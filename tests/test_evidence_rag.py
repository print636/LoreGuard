from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from sqlalchemy import create_engine, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.db import (
    Base,
    DocumentRow,
    EmbeddingProfileRow,
    EvidenceChunkRow,
    ProjectRow,
    enable_sqlite_foreign_keys,
)
from app.embeddings import (
    EmbeddingCallTelemetry,
    EmbeddingProfile,
    EmbeddingResponseError,
    EmbeddingResult,
    EmbeddingUsage,
)
from app.evidence_chunks import EvidenceChunker, EvidenceMatch, SnapshotDocumentKey
from app.evidence_rag import (
    EvidenceDocument,
    EvidenceIndexCoordinator,
    EvidenceIndexDiagnostics,
    EvidenceIndexResult,
    EvidenceQuery,
    EvidenceRrfRetriever,
    MAX_RETRIEVAL_CANDIDATES,
    RetrievalDiagnostics,
)
from app.evidence_store import SqlAlchemyEvidenceEmbeddingIndex


class MockEmbeddingProvider:
    def __init__(self, profile: EmbeddingProfile, *, session_counter=None):
        self._profile = profile
        self.calls: list[tuple[str, ...]] = []
        self.session_counter = session_counter

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def embed(self, texts):
        if self.session_counter is not None:
            if self.session_counter[0] != 0:
                raise AssertionError("embedding call happened while a DB session was open")
        prepared = tuple(texts)
        self.calls.append(prepared)
        vectors = tuple(
            (1.0, 0.0) if "白塔" in text else (0.0, 1.0) for text in prepared
        )
        return EmbeddingResult(
            profile=self.profile,
            vectors=vectors,
            usage=EmbeddingUsage(prompt_tokens=len(prepared), total_tokens=len(prepared)),
            telemetry=EmbeddingCallTelemetry(
                profile_id=self.profile.profile_id,
                input_count=len(prepared),
                input_chars=sum(len(text) for text in prepared),
                elapsed_ms=1,
                outcome="success",
            ),
        )


class FailingEmbeddingProvider(MockEmbeddingProvider):
    def embed(self, texts):
        self.calls.append(tuple(texts))
        raise EmbeddingResponseError("upstream secret body")


class ForbiddenProvider:
    @property
    def profile(self):
        raise AssertionError("provider profile was accessed")

    def embed(self, texts):
        raise AssertionError("provider was called")


class TrackingSessionContext:
    def __init__(
        self, session_factory, counter, *, race_once=None, commit_before_race=True
    ):
        self._session = session_factory()
        self._counter = counter
        self._race_once = race_once
        self._commit_before_race = commit_before_race

    def __enter__(self):
        self._counter[0] += 1
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self._session.close()
        finally:
            self._counter[0] -= 1

    def __getattr__(self, name):
        return getattr(self._session, name)

    def commit(self):
        if self._race_once is not None and self._race_once[0]:
            self._race_once[0] = False
            if self._commit_before_race:
                self._session.commit()
            raise IntegrityError("safe", {}, Exception("simulated race"))
        self._session.commit()


class FakeIndex:
    def __init__(self, ordered_chunks):
        self.ordered_chunks = tuple(ordered_chunks)
        self.calls = []

    def nearest(self, **kwargs):
        self.calls.append(kwargs)
        return tuple(
            EvidenceMatch(chunk=chunk, score=1.0 - index * 0.1)
            for index, chunk in enumerate(self.ordered_chunks)
        )


class EvidenceRagTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        enable_sqlite_foreign_keys(self.engine)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.content = "林澈抵达白塔。\n守卫确认星钥仍在林澈手中。\n夜晚，林澈离开白塔。"
        with self.Session() as session:
            session.add(ProjectRow(id="p", name="test", description=""))
            session.add(
                DocumentRow(
                    id="d",
                    project_id="p",
                    name="story.md",
                    content=self.content,
                    version=1,
                )
            )
            session.commit()
        self.snapshot = SnapshotDocumentKey(
            "p", "d", 1, hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        )
        self.document = EvidenceDocument(snapshot=self.snapshot, content=self.content)
        self.profile = EmbeddingProfile.openai_compatible(
            provider_namespace="unit",
            model_identifier="mock",
            model_revision="r1",
            deployment_fingerprint="unit-runtime-v1",
            dimensions=2,
        )
        self.chunker = EvidenceChunker(
            target_chars=12, min_chars=6, max_chars=18, overlap_chars=0
        )

    def tearDown(self):
        self.engine.dispose()

    def _coordinator(self, provider, **kwargs):
        return EvidenceIndexCoordinator(
            session_factory=self.Session,
            provider=provider,
            chunker=self.chunker,
            **kwargs,
        )

    def test_indexing_reuses_cache_and_never_holds_session_during_provider_call(self):
        counter = [0]

        def tracked_factory():
            return TrackingSessionContext(self.Session, counter)

        provider = MockEmbeddingProvider(self.profile, session_counter=counter)
        coordinator = EvidenceIndexCoordinator(
            session_factory=tracked_factory,
            provider=provider,
            chunker=self.chunker,
            batch_size=2,
        )
        first = coordinator.ensure_index([self.document])
        self.assertTrue(first.complete)
        self.assertGreater(len(first.chunks), 1)
        self.assertEqual(sum(len(call) for call in provider.calls), len(first.chunks))
        self.assertEqual(counter[0], 0)

        provider.calls.clear()
        second = coordinator.ensure_index([self.document])
        self.assertTrue(second.complete)
        self.assertEqual(provider.calls, [])
        self.assertEqual(second.diagnostics.reused_chunks, len(second.chunks))
        self.assertEqual(second.diagnostics.embedded_chunks, 0)

    def test_concurrent_unique_conflict_succeeds_only_after_exact_reread(self):
        counter = [0]
        race_once = [True]

        def racing_factory():
            return TrackingSessionContext(self.Session, counter, race_once=race_once)

        provider = MockEmbeddingProvider(self.profile, session_counter=counter)
        coordinator = EvidenceIndexCoordinator(
            session_factory=racing_factory,
            provider=provider,
            chunker=self.chunker,
            batch_size=64,
        )
        result = coordinator.ensure_index([self.document])
        self.assertTrue(result.complete)
        self.assertFalse(race_once[0])

    def test_concurrent_conflict_without_exact_rows_fails_closed(self):
        counter = [0]
        race_once = [True]

        def failed_race_factory():
            return TrackingSessionContext(
                self.Session,
                counter,
                race_once=race_once,
                commit_before_race=False,
            )

        provider = MockEmbeddingProvider(self.profile, session_counter=counter)
        coordinator = EvidenceIndexCoordinator(
            session_factory=failed_race_factory,
            provider=provider,
            chunker=self.chunker,
        )
        result = coordinator.ensure_index([self.document])
        self.assertFalse(result.complete)
        self.assertEqual(result.diagnostics.outcome, "write_conflict")
        self.assertEqual(result.diagnostics.reason, "concurrent_write_incomplete")

    def test_provider_failure_degrades_with_content_safe_diagnostics(self):
        provider = FailingEmbeddingProvider(self.profile)
        result = self._coordinator(provider).ensure_index([self.document])
        self.assertFalse(result.complete)
        safe = result.diagnostics.safe_dict()
        self.assertEqual(safe["reason"], "embedding_response_invalid")
        self.assertNotIn("secret", repr(result))
        self.assertNotIn(self.content, repr(result))
        self.assertNotIn("secret", repr(safe))

    def test_checkpoint_cancellation_propagates_before_network_call(self):
        class Cancelled(RuntimeError):
            pass

        provider = MockEmbeddingProvider(self.profile)

        def cancel():
            raise Cancelled("cancelled")

        with self.assertRaises(Cancelled):
            self._coordinator(provider, checkpoint=cancel).ensure_index([self.document])
        self.assertEqual(provider.calls, [])

    def test_index_scope_rejects_mixed_projects_and_multiple_document_versions(self):
        other_project_content = "另一个项目。"
        other_project = EvidenceDocument(
            snapshot=SnapshotDocumentKey(
                "p2",
                "d2",
                1,
                hashlib.sha256(other_project_content.encode("utf-8")).hexdigest(),
            ),
            content=other_project_content,
        )
        next_version_content = "同一文档的新版本。"
        next_version = EvidenceDocument(
            snapshot=SnapshotDocumentKey(
                "p",
                "d",
                2,
                hashlib.sha256(next_version_content.encode("utf-8")).hexdigest(),
            ),
            content=next_version_content,
        )
        coordinator = EvidenceIndexCoordinator(
            session_factory=self.Session,
            provider=ForbiddenProvider(),
            chunker=self.chunker,
        )
        with self.assertRaisesRegex(ValueError, "one project"):
            coordinator.ensure_index([self.document, other_project])
        with self.assertRaisesRegex(ValueError, "multiple versions"):
            coordinator.ensure_index([self.document, next_version])

    def test_chunk_limit_is_checked_before_provider_or_database_access(self):
        content = "界" * (MAX_RETRIEVAL_CANDIDATES + 1)
        document = EvidenceDocument(
            snapshot=SnapshotDocumentKey(
                "p",
                "d",
                1,
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
            ),
            content=content,
        )
        session_factory_calls = 0

        def counted_session_factory():
            nonlocal session_factory_calls
            session_factory_calls += 1
            return self.Session()

        coordinator = EvidenceIndexCoordinator(
            session_factory=counted_session_factory,
            provider=ForbiddenProvider(),
            chunker=EvidenceChunker(
                target_chars=1, min_chars=1, max_chars=1, overlap_chars=0
            ),
        )
        with self.assertRaisesRegex(ValueError, "chunk limit"):
            coordinator.ensure_index([document])
        self.assertEqual(session_factory_calls, 0)
        with self.Session() as session:
            self.assertEqual(
                session.scalar(select(func.count()).select_from(EvidenceChunkRow)), 0
            )
            self.assertEqual(
                session.scalar(select(func.count()).select_from(EmbeddingProfileRow)),
                0,
            )

    def test_entity_terms_must_be_observable_in_the_query(self):
        with self.assertRaisesRegex(ValueError, "not present"):
            EvidenceQuery(text="林澈在哪里", entity_terms=("白塔",))
        query = EvidenceQuery(text="LIN CHE 的去向", entity_terms=("lin che",))
        self.assertEqual(query.entity_terms, ("lin che",))

    def test_rrf_combines_real_vector_adapter_with_keyword_and_entity_ranks(self):
        provider = MockEmbeddingProvider(self.profile)
        indexed = self._coordinator(provider).ensure_index([self.document])
        provider.calls.clear()
        fake_index = FakeIndex(reversed(indexed.chunks))
        retriever = EvidenceRrfRetriever(index=fake_index, provider=provider)
        result = retriever.retrieve(
            indexed=indexed,
            allowed_snapshots=[self.snapshot],
            query=EvidenceQuery(text="林澈在哪里", entity_terms=("林澈",)),
            limit=3,
        )
        self.assertEqual(result.diagnostics.mode, "hybrid")
        self.assertEqual(result.diagnostics.vector_hits, len(indexed.chunks))
        self.assertEqual(len(provider.calls), 1)
        self.assertTrue(all(match.rrf_score > 0 for match in result.matches))
        self.assertTrue(any(match.vector_rank is not None for match in result.matches))
        self.assertNotIn("林澈在哪里", repr(result))

    def test_public_keyword_dense_and_rrf_strategies_use_only_requested_channels(self):
        provider = MockEmbeddingProvider(self.profile)
        indexed = self._coordinator(provider).ensure_index([self.document])
        query = EvidenceQuery(text="林澈的星钥", entity_terms=())

        keyword = EvidenceRrfRetriever(
            index=FakeIndex([]), provider=ForbiddenProvider()
        ).retrieve(
            indexed=indexed,
            allowed_snapshots=[self.snapshot],
            query=query,
            strategy="keyword-only",
        )
        self.assertEqual(keyword.diagnostics.strategy, "keyword-only")
        self.assertEqual(keyword.diagnostics.provider_calls, 0)
        self.assertEqual(keyword.diagnostics.vector_hits, 0)

        vector_index = FakeIndex(reversed(indexed.chunks))
        dense = EvidenceRrfRetriever(index=vector_index, provider=provider).retrieve(
            indexed=indexed,
            allowed_snapshots=[self.snapshot],
            query=query,
            strategy="dense-only",
        )
        self.assertEqual(dense.diagnostics.mode, "dense_only")
        self.assertEqual(dense.diagnostics.keyword_hits, 0)
        self.assertEqual(dense.diagnostics.entity_hits, 0)

        fused = EvidenceRrfRetriever(index=vector_index, provider=provider).retrieve(
            indexed=indexed,
            allowed_snapshots=[self.snapshot],
            query=query,
            strategy="keyword+dense-rrf",
        )
        self.assertEqual(fused.diagnostics.strategy, "keyword+dense-rrf")
        self.assertEqual(fused.diagnostics.entity_hits, 0)
        self.assertGreater(fused.diagnostics.keyword_hits, 0)
        self.assertGreater(fused.diagnostics.vector_hits, 0)

    def test_sqlite_vector_channel_fails_closed_but_lexical_retrieval_survives(self):
        provider = MockEmbeddingProvider(self.profile)
        indexed = self._coordinator(provider).ensure_index([self.document])
        retriever = EvidenceRrfRetriever(
            index=SqlAlchemyEvidenceEmbeddingIndex(self.Session), provider=provider
        )
        result = retriever.retrieve(
            indexed=indexed,
            allowed_snapshots=[self.snapshot],
            query=EvidenceQuery(text="林澈的星钥", entity_terms=("林澈",)),
        )
        self.assertEqual(result.diagnostics.mode, "lexical_only")
        self.assertEqual(result.diagnostics.reason, "vector_search_unavailable")
        self.assertGreater(len(result.matches), 0)
        self.assertTrue(all(match.vector_rank is None for match in result.matches))

    def test_vector_adapter_cannot_return_a_forged_out_of_scope_chunk(self):
        provider = MockEmbeddingProvider(self.profile)
        indexed = self._coordinator(provider).ensure_index([self.document])
        forged = replace(indexed.chunks[0])
        object.__setattr__(forged, "text", "leaked content")
        retriever = EvidenceRrfRetriever(index=FakeIndex([forged]), provider=provider)
        result = retriever.retrieve(
            indexed=indexed,
            allowed_snapshots=[self.snapshot],
            query=EvidenceQuery(text="林澈", entity_terms=()),
        )
        self.assertEqual(result.diagnostics.mode, "lexical_only")
        self.assertEqual(result.diagnostics.reason, "vector_scope_invalid")
        self.assertTrue(all(match.vector_rank is None for match in result.matches))

    def test_retriever_requires_exact_explicit_snapshot_authorization(self):
        provider = MockEmbeddingProvider(self.profile)
        indexed = self._coordinator(provider).ensure_index([self.document])
        other_content = "错误项目里的相似事实。"
        other_chunk = self.chunker.chunk(
            project_id="p2",
            document_id="d2",
            document_version=1,
            content=other_content,
        )[0]
        forged_diagnostics = replace(
            indexed.diagnostics, expected_chunks=len(indexed.chunks) + 1
        )
        retriever = EvidenceRrfRetriever(index=FakeIndex([]), provider=provider)

        forged_scope = replace(
            indexed,
            snapshots=(*indexed.snapshots, other_chunk.snapshot),
            chunks=(*indexed.chunks, other_chunk),
            diagnostics=forged_diagnostics,
        )
        with self.assertRaisesRegex(ValueError, "one project"):
            retriever.retrieve(
                indexed=forged_scope,
                allowed_snapshots=[self.snapshot],
                query=EvidenceQuery(text="林澈", entity_terms=("林澈",)),
            )

        forged_chunk_only = replace(
            indexed,
            chunks=(*indexed.chunks, other_chunk),
            diagnostics=forged_diagnostics,
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            retriever.retrieve(
                indexed=forged_chunk_only,
                allowed_snapshots=[self.snapshot],
                query=EvidenceQuery(text="林澈", entity_terms=("林澈",)),
            )

    def test_safe_diagnostics_reject_forged_content_and_unbounded_values(self):
        index_diagnostic = EvidenceIndexDiagnostics(
            outcome="complete",
            reason=None,
            profile_id=self.profile.profile_id,
            chunker_version=self.chunker.version,
            document_count=1,
            expected_chunks=2,
            reused_chunks=0,
            embedded_chunks=2,
            provider_calls=1,
            provider_input_chars=len(self.content),
            elapsed_ms=1,
        )
        object.__setattr__(index_diagnostic, "reason", {"secret": self.content})
        object.__setattr__(index_diagnostic, "profile_id", self.content * 100)
        object.__setattr__(index_diagnostic, "elapsed_ms", 10**100)
        safe = index_diagnostic.safe_dict()
        self.assertEqual(safe["reason"], "internal_failure")
        self.assertEqual(safe["profile_id"], "invalid")
        self.assertEqual(safe["elapsed_ms"], 0)
        self.assertNotIn(self.content, repr(safe))

        retrieval_diagnostic = RetrievalDiagnostics(
            strategy="keyword+dense-rrf",
            mode="hybrid",
            reason=None,
            profile_id=self.profile.profile_id,
            chunker_version=self.chunker.version,
            candidate_count=1,
            keyword_hits=1,
            vector_hits=1,
            entity_hits=1,
            result_count=1,
            provider_calls=1,
            provider_input_chars=1,
            elapsed_ms=1,
        )
        object.__setattr__(retrieval_diagnostic, "mode", [self.content])
        object.__setattr__(retrieval_diagnostic, "candidate_count", True)
        safe_retrieval = retrieval_diagnostic.safe_dict()
        self.assertEqual(safe_retrieval["mode"], "lexical_only")
        self.assertEqual(safe_retrieval["candidate_count"], 0)
        self.assertNotIn(self.content, repr(safe_retrieval))


if __name__ == "__main__":
    unittest.main()
