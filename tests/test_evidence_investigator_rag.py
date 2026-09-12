from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, DocumentRow, ProjectRow, enable_sqlite_foreign_keys
from app.domain import EvidenceSpan, ParsedDirective
from app.embeddings import (
    EmbeddingCallTelemetry,
    EmbeddingProfile,
    EmbeddingResult,
    EmbeddingUsage,
)
from app.evidence_authority import InvestigationScope, ScopedEvidenceDocument
from app.evidence_chunks import (
    EvidenceChunk,
    EvidenceChunker,
    EvidenceMatch,
    SnapshotDocumentKey,
)
from app.evidence_investigator import build_investigation_seeds
from app.evidence_investigator_loop import EvidenceInvestigatorToolLoop
from app.evidence_investigator_rag import (
    InvestigatorRagPolicy,
    InvestigatorRagRetriever,
    InvestigatorRagUnavailable,
)
from app.evidence_rag import (
    EvidenceIndexDiagnostics,
    EvidenceIndexResult,
    EvidenceQuery,
    EvidenceRetrievalResult,
    RankedEvidence,
    RetrievalDiagnostics,
)
from app.provider import ProviderToolCall, ToolCallResult


CONTENT = (
    "岚的发色是银色。\n"
    "赤羽在白塔交给岚一枚星钥。\n"
    "夜晚，岚在镜中看见自己的发色变成黑色。"
)


class MockEmbeddingProvider:
    def __init__(self, profile: EmbeddingProfile) -> None:
        self._profile = profile
        self.calls: list[tuple[str, ...]] = []

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

    def embed(self, texts):
        prepared = tuple(texts)
        self.calls.append(prepared)
        return EmbeddingResult(
            profile=self.profile,
            vectors=tuple((1.0, 0.0) for _ in prepared),
            usage=EmbeddingUsage(
                prompt_tokens=len(prepared),
                total_tokens=len(prepared),
            ),
            telemetry=EmbeddingCallTelemetry(
                profile_id=self.profile.profile_id,
                input_count=len(prepared),
                input_chars=sum(len(row) for row in prepared),
                elapsed_ms=1,
                outcome="success",
            ),
        )


class ScopeVectorIndex:
    def __init__(self, chunks: tuple[EvidenceChunk, ...]) -> None:
        self.chunks = chunks
        self.calls: list[dict[str, object]] = []

    def nearest(self, **kwargs):
        self.calls.append(kwargs)
        return tuple(
            EvidenceMatch(chunk=chunk, score=1.0 - (index * 0.01))
            for index, chunk in enumerate(reversed(self.chunks))
        )[: kwargs["limit"]]


class CountingCoordinator:
    def __init__(self, delegate=None, *, result=None, error=None) -> None:
        self.delegate = delegate
        self.result = result
        self.error = error
        self.calls = 0
        self.documents = []

    def ensure_index(self, documents):
        self.calls += 1
        self.documents.append(tuple(documents))
        if self.error is not None:
            raise self.error
        if self.delegate is not None:
            return self.delegate.ensure_index(documents)
        return self.result


class BlockingCoordinator(CountingCoordinator):
    def __init__(self, *, result) -> None:
        super().__init__(result=result)
        self.entered = threading.Event()
        self.release = threading.Event()

    def ensure_index(self, documents):
        self.calls += 1
        self.documents.append(tuple(documents))
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("test index release timed out")
        return self.result


class ScriptedRetriever:
    def __init__(self, result=None, *, error=None) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


class SearchOnceProvider:
    def __init__(self, seed_ref: str) -> None:
        self.seed_ref = seed_ref

    def complete_with_tools(self, system, user, *, tools, tool_choice, limits):
        return ToolCallResult(
            tool_calls=(
                ProviderToolCall(
                    id="call_1",
                    name="SEARCH_EVIDENCE",
                    arguments={
                        "seed_ref": self.seed_ref,
                        "query": "岚 发色 冲突",
                        "entity_terms": ["岚", "发色"],
                    },
                ),
            ),
            prompt_tokens=5,
            completion_tokens=2,
        )


@dataclass
class RuntimeContext:
    engine: object
    Session: object
    scope: InvestigationScope
    seed: object
    profile: EmbeddingProfile
    provider: MockEmbeddingProvider
    chunker: EvidenceChunker
    chunks: tuple[EvidenceChunk, ...]


@pytest.fixture
def runtime() -> RuntimeContext:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    enable_sqlite_foreign_keys(engine)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as session:
        session.add(ProjectRow(id="project-a", name="test", description=""))
        session.add(
            DocumentRow(
                id="doc-1",
                project_id="project-a",
                name="chapter.md",
                content=CONTENT,
                version=1,
            )
        )
        session.commit()
    snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-1",
        document_version=1,
        content_sha256=hashlib.sha256(CONTENT.encode("utf-8")).hexdigest(),
    )
    scope = InvestigationScope.create(
        run_id="run-a",
        project_id="project-a",
        documents=(ScopedEvidenceDocument(snapshot=snapshot, content=CONTENT),),
    )
    baseline = ParsedDirective(
        kind="fact",
        attrs={
            "subject": "岚",
            "predicate": "发色",
            "value": "银色",
            "modality": "asserted",
            "source_scope": "narrator",
            "certainty": "certain",
        },
        evidence=EvidenceSpan(
            document_id="doc-1",
            document_name="chapter.md",
            line_start=1,
            line_end=1,
            text=CONTENT.splitlines()[0],
        ),
        provenance_sources=frozenset({"baseline"}),
    )
    seed = build_investigation_seeds("run-a", (baseline,))[0]
    profile = EmbeddingProfile.openai_compatible(
        provider_namespace="investigator-test",
        model_identifier="mock-embedding",
        model_revision="r1",
        deployment_fingerprint="local-test-v1",
        dimensions=2,
    )
    provider = MockEmbeddingProvider(profile)
    chunker = EvidenceChunker(
        target_chars=18,
        min_chars=8,
        max_chars=28,
        overlap_chars=0,
    )
    chunks = chunker.chunk(
        project_id=snapshot.project_id,
        document_id=snapshot.document_id,
        document_version=snapshot.document_version,
        content=CONTENT,
        content_sha256=snapshot.content_sha256,
    )
    value = RuntimeContext(
        engine=engine,
        Session=Session,
        scope=scope,
        seed=seed,
        profile=profile,
        provider=provider,
        chunker=chunker,
        chunks=chunks,
    )
    try:
        yield value
    finally:
        engine.dispose()


def _complete_index(runtime: RuntimeContext) -> EvidenceIndexResult:
    return EvidenceIndexResult(
        profile=runtime.profile,
        snapshots=(runtime.scope.documents[0].snapshot,),
        chunks=runtime.chunks,
        diagnostics=EvidenceIndexDiagnostics(
            outcome="complete",
            reason=None,
            profile_id=runtime.profile.profile_id,
            chunker_version=runtime.chunker.version,
            document_count=1,
            expected_chunks=len(runtime.chunks),
            reused_chunks=len(runtime.chunks),
            embedded_chunks=0,
            provider_calls=0,
            provider_input_chars=0,
            elapsed_ms=1,
        ),
    )


def _retrieval_result(
    runtime: RuntimeContext,
    matches: tuple[RankedEvidence, ...],
    *,
    mode="hybrid",
    reason=None,
    strategy="keyword+vector+entity-rrf",
    keyword_hits=2,
    vector_hits=2,
    entity_hits=1,
    provider_calls=1,
) -> EvidenceRetrievalResult:
    return EvidenceRetrievalResult(
        matches=matches,
        diagnostics=RetrievalDiagnostics(
            strategy=strategy,
            mode=mode,
            reason=reason,
            profile_id=runtime.profile.profile_id,
            chunker_version=runtime.chunker.version,
            candidate_count=len(runtime.chunks),
            keyword_hits=min(keyword_hits, len(runtime.chunks)),
            vector_hits=min(vector_hits, len(runtime.chunks)),
            entity_hits=min(entity_hits, len(runtime.chunks)),
            result_count=len(matches),
            provider_calls=provider_calls,
            provider_input_chars=(len("岚 发色 冲突") if provider_calls else 0),
            elapsed_ms=1,
        ),
    )


def _adapter(runtime: RuntimeContext, **kwargs) -> InvestigatorRagRetriever:
    return InvestigatorRagRetriever(
        scope=runtime.scope,
        session_factory=runtime.Session,
        embedding_provider=runtime.provider,
        chunker=runtime.chunker,
        **kwargs,
    )


def test_real_coordinator_and_rrf_reuse_one_index_for_all_searches(runtime):
    from app.evidence_rag import EvidenceIndexCoordinator

    coordinator = CountingCoordinator(
        EvidenceIndexCoordinator(
            session_factory=runtime.Session,
            provider=runtime.provider,
            chunker=runtime.chunker,
        )
    )
    vector_index = ScopeVectorIndex(runtime.chunks)
    adapter = _adapter(
        runtime,
        index_coordinator=coordinator,
        embedding_index=vector_index,
        policy=InvestigatorRagPolicy(top_k=2, branch_limit=3),
    )
    query = EvidenceQuery(text="岚 发色 冲突", entity_terms=("岚", "发色"))

    first = adapter.search(seed=runtime.seed, query=query, limit=8)
    second = adapter.search(seed=runtime.seed, query=query, limit=1)

    assert coordinator.calls == 1
    assert 1 <= len(first) <= 2
    assert len(second) == 1
    assert len(vector_index.calls) == 2
    assert all(call["limit"] == 3 for call in vector_index.calls)
    assert all(
        tuple(call["snapshots"]) == (runtime.scope.documents[0].snapshot,)
        for call in vector_index.calls
    )
    assert coordinator.documents[0][0].content == CONTENT
    # Cold indexing embeds each chunk exactly once; every later provider call is
    # a query embedding, proving ensure_index was not repeated per seed/search.
    assert sum(len(call) for call in runtime.provider.calls) == len(runtime.chunks) + 2


def test_concurrent_prepare_and_search_publish_one_complete_index_state(runtime):
    match = RankedEvidence(
        chunk=runtime.chunks[0],
        rrf_score=0.03,
        keyword_rank=1,
        vector_rank=1,
        entity_rank=1,
    )
    coordinator = BlockingCoordinator(result=_complete_index(runtime))
    retriever = ScriptedRetriever(_retrieval_result(runtime, (match,)))
    adapter = _adapter(
        runtime,
        index_coordinator=coordinator,
        rrf_retriever=retriever,
    )
    query = EvidenceQuery(text="岚 发色 冲突", entity_terms=("岚",))

    with ThreadPoolExecutor(max_workers=6) as executor:
        prepare = executor.submit(adapter.prepare)
        assert coordinator.entered.wait(timeout=2)
        searches = [
            executor.submit(
                adapter.search,
                seed=runtime.seed,
                query=query,
                limit=2,
            )
            for _ in range(5)
        ]
        coordinator.release.set()
        assert prepare.result(timeout=3) is None
        results = [future.result(timeout=3) for future in searches]

    assert coordinator.calls == 1
    assert len(retriever.calls) == 5
    assert all(
        tuple(row.chunk_id for row in result) == (match.chunk.chunk_id,)
        for result in results
    )


def test_default_runtime_uses_real_sqlalchemy_embedding_index(runtime):
    adapter = _adapter(runtime)
    assert adapter._retriever._index.__class__.__name__ == "SqlAlchemyEvidenceEmbeddingIndex"
    # SQLite can persist the immutable vectors but intentionally cannot execute
    # pgvector nearest-neighbour search. Reaching this exact fail-closed reason
    # proves the default path used the real SQLAlchemy/pgvector adapter.
    with pytest.raises(InvestigatorRagUnavailable) as caught:
        adapter.search(
            seed=runtime.seed,
            query=EvidenceQuery(text="岚 发色 冲突", entity_terms=("岚",)),
            limit=2,
        )
    assert caught.value.reason_code == "vector_search_unavailable"


def test_cross_run_and_out_of_scope_anchor_are_rejected_before_index(runtime):
    coordinator = CountingCoordinator(result=_complete_index(runtime))
    adapter = _adapter(
        runtime,
        index_coordinator=coordinator,
        rrf_retriever=ScriptedRetriever(),
    )
    other_run_seed = build_investigation_seeds(
        "run-b", (runtime.seed.anchor,)
    )[0]
    with pytest.raises(ValueError, match="outside the run"):
        adapter.search(
            seed=other_run_seed,
            query=EvidenceQuery(text="岚 发色"),
            limit=2,
        )

    foreign_anchor = runtime.seed.anchor.model_copy(
        update={
            "evidence": runtime.seed.anchor.evidence.model_copy(
                update={"document_id": "doc-outside"}
            )
        }
    )
    foreign_seed = build_investigation_seeds("run-a", (foreign_anchor,))[0]
    with pytest.raises(ValueError, match="anchor is outside"):
        adapter.search(
            seed=foreign_seed,
            query=EvidenceQuery(text="岚 发色"),
            limit=2,
        )
    assert coordinator.calls == 0


def test_cross_snapshot_result_injection_fails_closed(runtime):
    foreign_text = runtime.chunks[0].text
    foreign_snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-foreign",
        document_version=1,
        content_sha256=hashlib.sha256(foreign_text.encode("utf-8")).hexdigest(),
    )
    foreign_chunk = EvidenceChunker(
        target_chars=100,
        min_chars=1,
        max_chars=100,
        overlap_chars=0,
    ).chunk(
        project_id=foreign_snapshot.project_id,
        document_id=foreign_snapshot.document_id,
        document_version=1,
        content=foreign_text,
        content_sha256=foreign_snapshot.content_sha256,
    )[0]
    match = RankedEvidence(
        chunk=foreign_chunk,
        rrf_score=0.02,
        keyword_rank=1,
        vector_rank=1,
        entity_rank=None,
    )
    result = _retrieval_result(runtime, (match,), keyword_hits=1, vector_hits=1)
    adapter = _adapter(
        runtime,
        index_coordinator=CountingCoordinator(result=_complete_index(runtime)),
        rrf_retriever=ScriptedRetriever(result),
    )

    with pytest.raises(InvestigatorRagUnavailable) as exc:
        adapter.search(
            seed=runtime.seed,
            query=EvidenceQuery(text="岚 发色 冲突"),
            limit=3,
        )
    assert exc.value.reason_code == "retrieval_invalid"
    assert foreign_text not in str(exc.value)


def test_duplicate_matches_are_deduplicated_and_sorted_stably(runtime):
    assert len(runtime.chunks) >= 2
    first, second = runtime.chunks[:2]
    matches = (
        RankedEvidence(
            chunk=first,
            rrf_score=0.01,
            keyword_rank=1,
            vector_rank=2,
            entity_rank=None,
        ),
        RankedEvidence(
            chunk=second,
            rrf_score=0.03,
            keyword_rank=2,
            vector_rank=1,
            entity_rank=1,
        ),
        RankedEvidence(
            chunk=first,
            rrf_score=0.02,
            keyword_rank=1,
            vector_rank=2,
            entity_rank=None,
        ),
    )
    result = _retrieval_result(runtime, matches)
    retriever = ScriptedRetriever(result)
    adapter = _adapter(
        runtime,
        index_coordinator=CountingCoordinator(result=_complete_index(runtime)),
        rrf_retriever=retriever,
        policy=InvestigatorRagPolicy(top_k=3, branch_limit=3),
    )

    chunks = adapter.search(
        seed=runtime.seed,
        query=EvidenceQuery(text="岚 发色 冲突", entity_terms=("岚",)),
        limit=3,
    )
    assert [row.chunk_id for row in chunks] == [second.chunk_id, first.chunk_id]
    assert len(set(row.chunk_id for row in chunks)) == len(chunks)
    assert retriever.calls[0]["limit"] == 3


def test_server_top_k_bounds_model_query_results(runtime):
    match = RankedEvidence(
        chunk=runtime.chunks[0],
        rrf_score=0.03,
        keyword_rank=1,
        vector_rank=1,
        entity_rank=1,
    )
    retriever = ScriptedRetriever(_retrieval_result(runtime, (match,)))
    adapter = _adapter(
        runtime,
        index_coordinator=CountingCoordinator(result=_complete_index(runtime)),
        rrf_retriever=retriever,
        policy=InvestigatorRagPolicy(top_k=1, branch_limit=2),
    )
    rows = adapter.search(
        seed=runtime.seed,
        query=EvidenceQuery(text="岚 发色 冲突", entity_terms=("岚",)),
        limit=20,
    )
    assert len(rows) == 1
    assert retriever.calls[0]["limit"] == 1
    assert retriever.calls[0]["branch_limit"] == 2
    assert retriever.calls[0]["strategy"] == "keyword+vector+entity-rrf"
    assert retriever.calls[0]["allowed_snapshots"] == (
        runtime.scope.documents[0].snapshot,
    )


def test_index_failure_propagates_once_and_is_never_retried(runtime):
    failure = RuntimeError("simulated index failure")
    coordinator = CountingCoordinator(error=failure)
    adapter = _adapter(
        runtime,
        index_coordinator=coordinator,
        rrf_retriever=ScriptedRetriever(),
    )
    query = EvidenceQuery(text="岚 发色")
    with pytest.raises(RuntimeError) as first:
        adapter.search(seed=runtime.seed, query=query, limit=2)
    assert first.value is failure
    with pytest.raises(InvestigatorRagUnavailable) as second:
        adapter.search(seed=runtime.seed, query=query, limit=2)
    assert second.value.reason_code == "index_incomplete"
    assert coordinator.calls == 1


def test_search_failure_propagates_to_loop_fail_closed_boundary(runtime):
    failure = RuntimeError("simulated search failure")
    adapter = _adapter(
        runtime,
        index_coordinator=CountingCoordinator(result=_complete_index(runtime)),
        rrf_retriever=ScriptedRetriever(error=failure),
    )
    with pytest.raises(RuntimeError) as caught:
        adapter.search(
            seed=runtime.seed,
            query=EvidenceQuery(text="岚 发色"),
            limit=2,
        )
    assert caught.value is failure


def test_checkpoint_cancellation_passes_through_without_index_retry(runtime):
    class Cancelled(RuntimeError):
        pass

    cancellation = Cancelled("cancelled")
    calls = 0

    def checkpoint():
        nonlocal calls
        calls += 1
        raise cancellation

    coordinator = CountingCoordinator(result=_complete_index(runtime))
    adapter = _adapter(
        runtime,
        index_coordinator=coordinator,
        rrf_retriever=ScriptedRetriever(),
        checkpoint=checkpoint,
    )
    with pytest.raises(Cancelled) as caught:
        adapter.search(
            seed=runtime.seed,
            query=EvidenceQuery(text="岚 发色"),
            limit=2,
        )
    assert caught.value is cancellation
    assert calls == 1
    assert coordinator.calls == 0


def test_loop_passthrough_contract_preserves_adapter_lease_exception(runtime):
    class LeaseLost(RuntimeError):
        pass

    lease_lost = LeaseLost("lease lost")

    def checkpoint():
        raise lease_lost

    adapter = _adapter(
        runtime,
        index_coordinator=CountingCoordinator(result=_complete_index(runtime)),
        rrf_retriever=ScriptedRetriever(),
        checkpoint=checkpoint,
    )
    loop = EvidenceInvestigatorToolLoop(
        provider=SearchOnceProvider(runtime.seed.seed_ref),
        retriever=adapter,
        scope=runtime.scope,
        seeds=(runtime.seed,),
        passthrough_exceptions=(LeaseLost,),
    )
    with pytest.raises(LeaseLost) as caught:
        loop.run()
    assert caught.value is lease_lost


def test_embedding_not_configured_index_result_is_terminal(runtime):
    incomplete = EvidenceIndexResult(
        profile=runtime.profile,
        snapshots=(runtime.scope.documents[0].snapshot,),
        chunks=runtime.chunks,
        diagnostics=EvidenceIndexDiagnostics(
            outcome="provider_unavailable",
            reason="embedding_not_configured",
            profile_id=runtime.profile.profile_id,
            chunker_version=runtime.chunker.version,
            document_count=1,
            expected_chunks=len(runtime.chunks),
            reused_chunks=0,
            embedded_chunks=0,
            provider_calls=0,
            provider_input_chars=0,
            elapsed_ms=1,
        ),
    )
    coordinator = CountingCoordinator(result=incomplete)
    adapter = _adapter(
        runtime,
        index_coordinator=coordinator,
        rrf_retriever=ScriptedRetriever(),
        policy=InvestigatorRagPolicy(require_hybrid=False),
    )
    with pytest.raises(InvestigatorRagUnavailable) as caught:
        adapter.search(
            seed=runtime.seed,
            query=EvidenceQuery(text="岚 发色"),
            limit=2,
        )
    assert caught.value.reason_code == "embedding_not_configured"
    assert coordinator.calls == 1


def test_hybrid_requirement_rejects_successful_lexical_only_mode(runtime):
    match = RankedEvidence(
        chunk=runtime.chunks[0],
        rrf_score=0.02,
        keyword_rank=1,
        vector_rank=None,
        entity_rank=None,
    )
    lexical = _retrieval_result(
        runtime,
        (match,),
        mode="lexical_only",
        keyword_hits=1,
        vector_hits=0,
        entity_hits=0,
    )
    adapter = _adapter(
        runtime,
        index_coordinator=CountingCoordinator(result=_complete_index(runtime)),
        rrf_retriever=ScriptedRetriever(lexical),
    )
    with pytest.raises(InvestigatorRagUnavailable) as caught:
        adapter.search(
            seed=runtime.seed,
            query=EvidenceQuery(text="岚 发色 冲突"),
            limit=2,
        )
    assert caught.value.reason_code == "hybrid_unavailable"


def test_vector_degradation_falls_back_to_lexical_when_hybrid_not_required(runtime):
    match = RankedEvidence(
        chunk=runtime.chunks[0],
        rrf_score=0.02,
        keyword_rank=1,
        vector_rank=None,
        entity_rank=None,
    )
    degraded = _retrieval_result(
        runtime,
        (match,),
        mode="lexical_only",
        reason="vector_search_unavailable",
        keyword_hits=1,
        vector_hits=0,
        entity_hits=0,
    )
    adapter = _adapter(
        runtime,
        index_coordinator=CountingCoordinator(result=_complete_index(runtime)),
        rrf_retriever=ScriptedRetriever(degraded),
        policy=InvestigatorRagPolicy(require_hybrid=False),
    )
    rows = adapter.search(
        seed=runtime.seed,
        query=EvidenceQuery(text="岚 发色 冲突"),
        limit=2,
    )

    assert tuple(row.chunk_id for row in rows) == (match.chunk.chunk_id,)
    diagnostics = adapter.safe_diagnostics()
    assert diagnostics["retrievals"][0]["mode"] == "lexical_only"
    assert diagnostics["retrievals"][0]["reason"] == "vector_search_unavailable"


def test_transient_index_failure_falls_back_lexically_and_is_published_once(runtime):
    incomplete = EvidenceIndexResult(
        profile=runtime.profile,
        snapshots=(runtime.scope.documents[0].snapshot,),
        chunks=runtime.chunks,
        diagnostics=EvidenceIndexDiagnostics(
            outcome="provider_unavailable",
            reason="embedding_retry_exhausted",
            profile_id=runtime.profile.profile_id,
            chunker_version=runtime.chunker.version,
            document_count=1,
            expected_chunks=len(runtime.chunks),
            reused_chunks=0,
            embedded_chunks=0,
            provider_calls=2,
            provider_input_chars=sum(len(row.text) for row in runtime.chunks),
            elapsed_ms=1,
        ),
    )
    match = RankedEvidence(
        chunk=runtime.chunks[0],
        rrf_score=0.02,
        keyword_rank=1,
        vector_rank=None,
        entity_rank=None,
    )
    lexical = _retrieval_result(
        runtime,
        (match,),
        mode="lexical_only",
        reason="index_incomplete",
        keyword_hits=1,
        vector_hits=0,
        entity_hits=0,
        provider_calls=0,
    )
    coordinator = CountingCoordinator(result=incomplete)
    retriever = ScriptedRetriever(lexical)
    adapter = _adapter(
        runtime,
        index_coordinator=coordinator,
        rrf_retriever=retriever,
        policy=InvestigatorRagPolicy(require_hybrid=False),
    )

    first = adapter.search(
        seed=runtime.seed,
        query=EvidenceQuery(text="岚 发色 冲突"),
        limit=2,
    )
    second = adapter.search(
        seed=runtime.seed,
        query=EvidenceQuery(text="岚 发色 冲突"),
        limit=2,
    )

    assert tuple(row.chunk_id for row in first) == (match.chunk.chunk_id,)
    assert tuple(row.chunk_id for row in second) == (match.chunk.chunk_id,)
    assert coordinator.calls == 1
    assert len(retriever.calls) == 2


@pytest.mark.parametrize(
    "reason",
    [
        "embedding_not_configured",
        "embedding_input_rejected",
        "embedding_response_invalid",
        "vector_scope_invalid",
    ],
)
def test_non_transient_query_failures_remain_terminal_without_hybrid_requirement(
    runtime, reason
):
    match = RankedEvidence(
        chunk=runtime.chunks[0],
        rrf_score=0.02,
        keyword_rank=1,
        vector_rank=None,
        entity_rank=None,
    )
    degraded = _retrieval_result(
        runtime,
        (match,),
        mode="lexical_only",
        reason=reason,
        keyword_hits=1,
        vector_hits=0,
        entity_hits=0,
    )
    adapter = _adapter(
        runtime,
        index_coordinator=CountingCoordinator(result=_complete_index(runtime)),
        rrf_retriever=ScriptedRetriever(degraded),
        policy=InvestigatorRagPolicy(require_hybrid=False),
    )

    with pytest.raises(InvestigatorRagUnavailable) as caught:
        adapter.search(
            seed=runtime.seed,
            query=EvidenceQuery(text="岚 发色 冲突"),
            limit=2,
        )
    assert caught.value.reason_code == reason


def test_index_chunk_content_must_match_frozen_scope(runtime):
    other_content = "伪造的索引内容。"
    other_snapshot = SnapshotDocumentKey(
        project_id="project-a",
        document_id="doc-foreign",
        document_version=1,
        content_sha256=hashlib.sha256(other_content.encode("utf-8")).hexdigest(),
    )
    foreign = EvidenceChunker(
        target_chars=20,
        min_chars=1,
        max_chars=20,
        overlap_chars=0,
        version="foreign",
    ).chunk(
        project_id=other_snapshot.project_id,
        document_id=other_snapshot.document_id,
        document_version=1,
        content=other_content,
        content_sha256=other_snapshot.content_sha256,
    )[0]
    poisoned = EvidenceIndexResult(
        profile=runtime.profile,
        snapshots=(runtime.scope.documents[0].snapshot,),
        chunks=(foreign,),
        diagnostics=EvidenceIndexDiagnostics(
            outcome="complete",
            reason=None,
            profile_id=runtime.profile.profile_id,
            chunker_version=runtime.chunker.version,
            document_count=1,
            expected_chunks=1,
            reused_chunks=1,
            embedded_chunks=0,
            provider_calls=0,
            provider_input_chars=0,
            elapsed_ms=1,
        ),
    )
    adapter = _adapter(
        runtime,
        index_coordinator=CountingCoordinator(result=poisoned),
        rrf_retriever=ScriptedRetriever(),
    )
    with pytest.raises(InvestigatorRagUnavailable) as caught:
        adapter.prepare()
    assert caught.value.reason_code == "index_invalid"


def test_policy_rejects_model_controllable_non_hybrid_configuration():
    with pytest.raises(ValueError, match="hybrid mode"):
        InvestigatorRagPolicy(strategy="keyword-only", require_hybrid=True)
