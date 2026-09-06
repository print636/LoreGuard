from __future__ import annotations

import hashlib
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .embeddings import (
    EmbeddingInputError,
    EmbeddingNotConfiguredError,
    EmbeddingProfile,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingResponseError,
    EmbeddingRetryExhaustedError,
)
from .evidence_chunks import (
    EvidenceChunk,
    EvidenceChunker,
    EvidenceEmbeddingIndex,
    SnapshotDocumentKey,
)
from .evidence_store import (
    MAX_STORE_BATCH,
    EvidenceStoreError,
    VectorSearchUnavailable,
    embedding_coverage,
    missing_embedding_chunks,
    store_embeddings,
)


MAX_DOCUMENTS = 256
MAX_QUERY_CHARS = 4_000
MAX_ENTITY_TERMS = 64
MAX_ENTITY_CHARS = 120
MAX_RETRIEVAL_CANDIDATES = 5_000
MAX_RETRIEVAL_LIMIT = 50

IndexOutcome = Literal["complete", "provider_unavailable", "write_conflict"]
RetrievalMode = Literal["hybrid", "lexical_only"]


@dataclass(frozen=True)
class EvidenceDocument:
    snapshot: SnapshotDocumentKey
    content: str = field(repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, SnapshotDocumentKey):
            raise ValueError("evidence document snapshot is invalid")
        if not isinstance(self.content, str):
            raise ValueError("evidence document content is invalid")
        content_hash = hashlib.sha256(self.content.encode("utf-8")).hexdigest()
        if content_hash != self.snapshot.content_sha256:
            raise ValueError("evidence document content does not match its snapshot")


@dataclass(frozen=True)
class EvidenceIndexDiagnostics:
    outcome: IndexOutcome
    reason: str | None
    profile_id: str
    chunker_version: str
    document_count: int
    expected_chunks: int
    reused_chunks: int
    embedded_chunks: int
    provider_calls: int
    provider_input_chars: int
    elapsed_ms: int

    def safe_dict(self) -> dict[str, object]:
        outcomes = {"complete", "provider_unavailable", "write_conflict"}
        reasons = {
            None,
            "embedding_not_configured",
            "embedding_input_rejected",
            "embedding_response_invalid",
            "embedding_retry_exhausted",
            "embedding_provider_failed",
            "concurrent_write_incomplete",
        }
        return {
            "outcome": (
                self.outcome
                if isinstance(self.outcome, str) and self.outcome in outcomes
                else "provider_unavailable"
            ),
            "reason": (
                self.reason
                if (self.reason is None or isinstance(self.reason, str))
                and self.reason in reasons
                else "internal_failure"
            ),
            "profile_id": _safe_identifier(self.profile_id, 68),
            "chunker_version": _safe_identifier(self.chunker_version, 80),
            "document_count": _safe_count(self.document_count),
            "expected_chunks": _safe_count(self.expected_chunks),
            "reused_chunks": _safe_count(self.reused_chunks),
            "embedded_chunks": _safe_count(self.embedded_chunks),
            "provider_calls": _safe_count(self.provider_calls),
            "provider_input_chars": _safe_count(self.provider_input_chars),
            "elapsed_ms": _safe_count(self.elapsed_ms),
        }


@dataclass(frozen=True)
class EvidenceIndexResult:
    profile: EmbeddingProfile
    snapshots: tuple[SnapshotDocumentKey, ...]
    chunks: tuple[EvidenceChunk, ...] = field(repr=False)
    diagnostics: EvidenceIndexDiagnostics

    @property
    def complete(self) -> bool:
        return self.diagnostics.outcome == "complete"


@dataclass(frozen=True)
class EvidenceQuery:
    text: str = field(repr=False)
    entity_terms: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.text, str)
            or not self.text.strip()
            or len(self.text) > MAX_QUERY_CHARS
        ):
            raise ValueError("evidence query text is invalid")
        if not isinstance(self.entity_terms, tuple) or len(self.entity_terms) > MAX_ENTITY_TERMS:
            raise ValueError("evidence query entity terms are invalid")
        seen: set[str] = set()
        folded_query = self.text.casefold()
        for term in self.entity_terms:
            if (
                not isinstance(term, str)
                or term != term.strip()
                or not term
                or len(term) > MAX_ENTITY_CHARS
            ):
                raise ValueError("evidence query entity term is invalid")
            folded = term.casefold()
            if folded in seen:
                raise ValueError("evidence query entity terms contain duplicates")
            if folded not in folded_query:
                raise ValueError("evidence query entity term is not present in the query")
            seen.add(folded)


@dataclass(frozen=True)
class RankedEvidence:
    chunk: EvidenceChunk = field(repr=False)
    rrf_score: float
    keyword_rank: int | None
    vector_rank: int | None
    entity_rank: int | None


@dataclass(frozen=True)
class RetrievalDiagnostics:
    mode: RetrievalMode
    reason: str | None
    profile_id: str
    chunker_version: str
    candidate_count: int
    keyword_hits: int
    vector_hits: int
    entity_hits: int
    result_count: int
    provider_calls: int
    provider_input_chars: int
    elapsed_ms: int

    def safe_dict(self) -> dict[str, object]:
        modes = {"hybrid", "lexical_only"}
        reasons = {
            None,
            "index_incomplete",
            "embedding_not_configured",
            "embedding_input_rejected",
            "embedding_response_invalid",
            "embedding_retry_exhausted",
            "embedding_provider_failed",
            "vector_search_unavailable",
            "vector_scope_invalid",
        }
        return {
            "mode": (
                self.mode
                if isinstance(self.mode, str) and self.mode in modes
                else "lexical_only"
            ),
            "reason": (
                self.reason
                if (self.reason is None or isinstance(self.reason, str))
                and self.reason in reasons
                else "internal_failure"
            ),
            "profile_id": _safe_identifier(self.profile_id, 68),
            "chunker_version": _safe_identifier(self.chunker_version, 80),
            "candidate_count": _safe_count(self.candidate_count),
            "keyword_hits": _safe_count(self.keyword_hits),
            "vector_hits": _safe_count(self.vector_hits),
            "entity_hits": _safe_count(self.entity_hits),
            "result_count": _safe_count(self.result_count),
            "provider_calls": _safe_count(self.provider_calls),
            "provider_input_chars": _safe_count(self.provider_input_chars),
            "elapsed_ms": _safe_count(self.elapsed_ms),
        }


@dataclass(frozen=True)
class EvidenceRetrievalResult:
    matches: tuple[RankedEvidence, ...] = field(repr=False)
    diagnostics: RetrievalDiagnostics


class EvidenceIndexCoordinator:
    """Build missing immutable vectors without holding a DB transaction on I/O."""

    def __init__(
        self,
        *,
        session_factory: Callable[[], Session],
        provider: EmbeddingProvider,
        chunker: EvidenceChunker | None = None,
        batch_size: int = 64,
        checkpoint: Callable[[], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session factory must be callable")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not (
            1 <= batch_size <= MAX_STORE_BATCH
        ):
            raise ValueError("embedding batch size is invalid")
        if checkpoint is not None and not callable(checkpoint):
            raise TypeError("checkpoint must be callable")
        self._session_factory = session_factory
        self._provider = provider
        self._chunker = chunker or EvidenceChunker()
        self._batch_size = batch_size
        self._checkpoint = checkpoint or (lambda: None)
        self._monotonic = monotonic

    def ensure_index(
        self, documents: Sequence[EvidenceDocument]
    ) -> EvidenceIndexResult:
        started = self._monotonic()
        prepared_documents = _validate_documents(documents)
        chunks = tuple(
            chunk
            for document in prepared_documents
            for chunk in self._chunker.chunk(
                project_id=document.snapshot.project_id,
                document_id=document.snapshot.document_id,
                document_version=document.snapshot.document_version,
                content=document.content,
                content_sha256=document.snapshot.content_sha256,
            )
        )
        if len(chunks) > MAX_RETRIEVAL_CANDIDATES:
            raise ValueError("evidence chunk limit was exceeded")
        _validate_candidate_chunks(chunks, self._chunker.version)
        profile = self._provider.profile
        missing = self._missing_chunks(profile, chunks)
        reused_chunks = len(chunks) - len(missing)
        embedded_chunks = 0
        provider_calls = 0
        provider_input_chars = 0

        for batch in _batches(missing, self._batch_size):
            self._checkpoint()
            provider_calls += 1
            provider_input_chars += sum(len(chunk.text) for chunk in batch)
            try:
                result = self._provider.embed(tuple(chunk.text for chunk in batch))
            except EmbeddingProviderError as exc:
                return self._index_result(
                    profile=profile,
                    chunks=chunks,
                    documents=prepared_documents,
                    outcome="provider_unavailable",
                    reason=_provider_reason(exc),
                    reused_chunks=reused_chunks,
                    embedded_chunks=embedded_chunks,
                    provider_calls=provider_calls,
                    provider_input_chars=provider_input_chars,
                    started=started,
                )
            self._checkpoint()
            if result.profile != profile or len(result.vectors) != len(batch):
                return self._index_result(
                    profile=profile,
                    chunks=chunks,
                    documents=prepared_documents,
                    outcome="provider_unavailable",
                    reason="embedding_response_invalid",
                    reused_chunks=reused_chunks,
                    embedded_chunks=embedded_chunks,
                    provider_calls=provider_calls,
                    provider_input_chars=provider_input_chars,
                    started=started,
                )
            try:
                with self._session_factory() as session:
                    store_embeddings(
                        session,
                        profile=profile,
                        chunks=batch,
                        vectors=result.vectors,
                    )
                    session.commit()
            except IntegrityError:
                # Another worker may have won the immutable unique-key race.
                # Only an exact, complete reread turns that race into success.
                if not self._coverage_complete(profile, batch):
                    return self._index_result(
                        profile=profile,
                        chunks=chunks,
                        documents=prepared_documents,
                        outcome="write_conflict",
                        reason="concurrent_write_incomplete",
                        reused_chunks=reused_chunks,
                        embedded_chunks=embedded_chunks,
                        provider_calls=provider_calls,
                        provider_input_chars=provider_input_chars,
                        started=started,
                    )
            embedded_chunks += len(batch)
            self._checkpoint()

        if not self._coverage_complete(profile, chunks):
            return self._index_result(
                profile=profile,
                chunks=chunks,
                documents=prepared_documents,
                outcome="write_conflict",
                reason="concurrent_write_incomplete",
                reused_chunks=reused_chunks,
                embedded_chunks=embedded_chunks,
                provider_calls=provider_calls,
                provider_input_chars=provider_input_chars,
                started=started,
            )
        return self._index_result(
            profile=profile,
            chunks=chunks,
            documents=prepared_documents,
            outcome="complete",
            reason=None,
            reused_chunks=reused_chunks,
            embedded_chunks=embedded_chunks,
            provider_calls=provider_calls,
            provider_input_chars=provider_input_chars,
            started=started,
        )

    def _missing_chunks(
        self, profile: EmbeddingProfile, chunks: tuple[EvidenceChunk, ...]
    ) -> tuple[EvidenceChunk, ...]:
        missing: list[EvidenceChunk] = []
        for batch in _batches(chunks, MAX_STORE_BATCH):
            self._checkpoint()
            with self._session_factory() as session:
                missing.extend(
                    missing_embedding_chunks(session, profile=profile, chunks=batch)
                )
            self._checkpoint()
        return tuple(missing)

    def _coverage_complete(
        self, profile: EmbeddingProfile, chunks: Sequence[EvidenceChunk]
    ) -> bool:
        for batch in _batches(tuple(chunks), MAX_STORE_BATCH):
            self._checkpoint()
            with self._session_factory() as session:
                if not embedding_coverage(
                    session, profile=profile, chunks=batch
                ).complete:
                    return False
            self._checkpoint()
        return True

    def _index_result(
        self,
        *,
        profile: EmbeddingProfile,
        chunks: tuple[EvidenceChunk, ...],
        documents: tuple[EvidenceDocument, ...],
        outcome: IndexOutcome,
        reason: str | None,
        reused_chunks: int,
        embedded_chunks: int,
        provider_calls: int,
        provider_input_chars: int,
        started: float,
    ) -> EvidenceIndexResult:
        return EvidenceIndexResult(
            profile=profile,
            snapshots=tuple(document.snapshot for document in documents),
            chunks=chunks,
            diagnostics=EvidenceIndexDiagnostics(
                outcome=outcome,
                reason=reason,
                profile_id=profile.profile_id,
                chunker_version=self._chunker.version,
                document_count=len(documents),
                expected_chunks=len(chunks),
                reused_chunks=reused_chunks,
                embedded_chunks=embedded_chunks,
                provider_calls=provider_calls,
                provider_input_chars=provider_input_chars,
                elapsed_ms=max(0, int((self._monotonic() - started) * 1000)),
            ),
        )


class EvidenceRrfRetriever:
    """Deterministic keyword/vector/entity RRF over an exact indexed snapshot."""

    def __init__(
        self,
        *,
        index: EvidenceEmbeddingIndex,
        provider: EmbeddingProvider,
        rrf_k: int = 60,
        checkpoint: Callable[[], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int) or not (1 <= rrf_k <= 1_000):
            raise ValueError("RRF constant is invalid")
        if checkpoint is not None and not callable(checkpoint):
            raise TypeError("checkpoint must be callable")
        self._index = index
        self._provider = provider
        self._rrf_k = rrf_k
        self._checkpoint = checkpoint or (lambda: None)
        self._monotonic = monotonic

    def retrieve(
        self,
        *,
        indexed: EvidenceIndexResult,
        allowed_snapshots: Sequence[SnapshotDocumentKey],
        query: EvidenceQuery,
        limit: int = 12,
        branch_limit: int = 30,
    ) -> EvidenceRetrievalResult:
        started = self._monotonic()
        if not isinstance(indexed, EvidenceIndexResult) or not isinstance(query, EvidenceQuery):
            raise TypeError("evidence retrieval input is invalid")
        if isinstance(limit, bool) or not isinstance(limit, int) or not (
            1 <= limit <= MAX_RETRIEVAL_LIMIT
        ):
            raise ValueError("retrieval limit is invalid")
        if isinstance(branch_limit, bool) or not isinstance(branch_limit, int) or not (
            1 <= branch_limit <= MAX_RETRIEVAL_LIMIT
        ):
            raise ValueError("retrieval branch limit is invalid")
        profile = indexed.profile
        if not isinstance(profile, EmbeddingProfile):
            raise ValueError("retrieval embedding profile is invalid")
        try:
            EmbeddingProfile(
                profile_id=profile.profile_id,
                provider_kind=profile.provider_kind,
                provider_namespace=profile.provider_namespace,
                model_identifier=profile.model_identifier,
                model_revision=profile.model_revision,
                deployment_fingerprint=profile.deployment_fingerprint,
                document_transform_identity=profile.document_transform_identity,
                query_transform_identity=profile.query_transform_identity,
                dimensions=profile.dimensions,
                normalized=profile.normalized,
            )
        except (TypeError, ValueError):
            raise ValueError("retrieval embedding profile is invalid") from None
        if not isinstance(indexed.diagnostics, EvidenceIndexDiagnostics):
            raise ValueError("retrieval index diagnostics are invalid")
        authorized_snapshots = _validate_snapshot_scope(allowed_snapshots)
        indexed_snapshots = _validate_snapshot_scope(indexed.snapshots)
        if set(authorized_snapshots) != set(indexed_snapshots):
            raise ValueError("retrieval snapshot authorization does not match the index")
        if (
            indexed.diagnostics.profile_id != profile.profile_id
            or type(indexed.diagnostics.expected_chunks) is not int
            or indexed.diagnostics.expected_chunks != len(indexed.chunks)
        ):
            raise ValueError("retrieval index scope is invalid")
        chunks = indexed.chunks
        if len(chunks) > MAX_RETRIEVAL_CANDIDATES:
            raise ValueError("retrieval candidate limit was exceeded")
        _validate_candidate_chunks(chunks, indexed.diagnostics.chunker_version)
        authorized_set = set(authorized_snapshots)
        if any(chunk.snapshot not in authorized_set for chunk in chunks):
            raise ValueError("retrieval candidate is outside the authorized snapshots")
        keyword_ids = _keyword_ranking(query.text, chunks, branch_limit)
        entity_ids = _entity_ranking(query.entity_terms, chunks, branch_limit)
        vector_ids: tuple[str, ...] = ()
        vector_reason: str | None = None
        provider_calls = 0
        provider_input_chars = 0
        if not indexed.complete:
            vector_reason = "index_incomplete"
        elif chunks:
            self._checkpoint()
            try:
                if self._provider.profile != profile:
                    vector_reason = "vector_scope_invalid"
                else:
                    provider_calls = 1
                    provider_input_chars = len(query.text)
                    embedded = self._provider.embed((query.text,))
                    self._checkpoint()
                    if embedded.profile != profile or len(embedded.vectors) != 1:
                        vector_reason = "embedding_response_invalid"
                    else:
                        vector_matches = self._index.nearest(
                            snapshots=_unique_snapshots(chunks),
                            profile_id=profile.profile_id,
                            chunker_version=indexed.diagnostics.chunker_version,
                            query_vector=embedded.vectors[0],
                            limit=branch_limit,
                        )
                        candidate_ids = {chunk.chunk_id for chunk in chunks}
                        candidates_by_id = {
                            chunk.chunk_id: chunk for chunk in chunks
                        }
                        returned_ids = tuple(
                            match.chunk.chunk_id for match in vector_matches
                        )
                        if (
                            len(set(returned_ids)) != len(returned_ids)
                            or any(
                                chunk_id not in candidate_ids
                                for chunk_id in returned_ids
                            )
                            or any(
                                not isinstance(match.score, (int, float))
                                or isinstance(match.score, bool)
                                or not math.isfinite(float(match.score))
                                or candidates_by_id.get(match.chunk.chunk_id)
                                != match.chunk
                                for match in vector_matches
                            )
                        ):
                            vector_reason = "vector_scope_invalid"
                        else:
                            vector_ids = returned_ids
            except EmbeddingProviderError as exc:
                vector_reason = _provider_reason(exc)
            except VectorSearchUnavailable:
                vector_reason = "vector_search_unavailable"
            except EvidenceStoreError:
                vector_reason = "vector_scope_invalid"

        rankings = (keyword_ids, vector_ids, entity_ids)
        ranks_by_id: dict[str, list[int | None]] = {}
        totals: dict[str, float] = {}
        for channel_index, ranking in enumerate(rankings):
            for rank, chunk_id in enumerate(ranking, start=1):
                ranks_by_id.setdefault(chunk_id, [None, None, None])[channel_index] = rank
                totals[chunk_id] = totals.get(chunk_id, 0.0) + 1.0 / (
                    self._rrf_k + rank
                )
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        ordered_ids = sorted(totals, key=lambda value: (-totals[value], value))[:limit]
        matches = tuple(
            RankedEvidence(
                chunk=by_id[chunk_id],
                rrf_score=totals[chunk_id],
                keyword_rank=ranks_by_id[chunk_id][0],
                vector_rank=ranks_by_id[chunk_id][1],
                entity_rank=ranks_by_id[chunk_id][2],
            )
            for chunk_id in ordered_ids
        )
        return EvidenceRetrievalResult(
            matches=matches,
            diagnostics=RetrievalDiagnostics(
                mode="hybrid" if vector_ids else "lexical_only",
                reason=vector_reason,
                profile_id=profile.profile_id,
                chunker_version=indexed.diagnostics.chunker_version,
                candidate_count=len(chunks),
                keyword_hits=len(keyword_ids),
                vector_hits=len(vector_ids),
                entity_hits=len(entity_ids),
                result_count=len(matches),
                provider_calls=provider_calls,
                provider_input_chars=provider_input_chars,
                elapsed_ms=max(0, int((self._monotonic() - started) * 1000)),
            ),
        )


def _validate_documents(
    documents: Sequence[EvidenceDocument],
) -> tuple[EvidenceDocument, ...]:
    try:
        prepared = tuple(documents)
    except TypeError:
        raise ValueError("evidence documents are invalid") from None
    if len(prepared) > MAX_DOCUMENTS:
        raise ValueError("evidence document limit was exceeded")
    identities: set[SnapshotDocumentKey] = set()
    projects: set[str] = set()
    document_identities: set[tuple[str, str]] = set()
    for document in prepared:
        if not isinstance(document, EvidenceDocument):
            raise ValueError("evidence document is invalid")
        # Reconstruct to reject forged frozen dataclasses.
        try:
            checked = EvidenceDocument(snapshot=document.snapshot, content=document.content)
        except (TypeError, ValueError):
            raise ValueError("evidence document is invalid") from None
        if checked.snapshot in identities:
            raise ValueError("evidence documents contain duplicate snapshots")
        identities.add(checked.snapshot)
        projects.add(checked.snapshot.project_id)
        document_identity = (
            checked.snapshot.project_id,
            checked.snapshot.document_id,
        )
        if document_identity in document_identities:
            raise ValueError("evidence documents contain multiple versions of a document")
        document_identities.add(document_identity)
    if len(projects) > 1:
        raise ValueError("evidence documents must belong to one project")
    return prepared


def _validate_snapshot_scope(
    snapshots: Sequence[SnapshotDocumentKey],
) -> tuple[SnapshotDocumentKey, ...]:
    try:
        prepared = tuple(snapshots)
    except TypeError:
        raise ValueError("retrieval snapshot scope is invalid") from None
    if len(prepared) > MAX_DOCUMENTS:
        raise ValueError("retrieval snapshot scope limit was exceeded")
    projects: set[str] = set()
    documents: set[tuple[str, str]] = set()
    identities: set[SnapshotDocumentKey] = set()
    for snapshot in prepared:
        if not isinstance(snapshot, SnapshotDocumentKey):
            raise ValueError("retrieval snapshot scope is invalid")
        try:
            checked = SnapshotDocumentKey(
                project_id=snapshot.project_id,
                document_id=snapshot.document_id,
                document_version=snapshot.document_version,
                content_sha256=snapshot.content_sha256,
            )
        except (TypeError, ValueError):
            raise ValueError("retrieval snapshot scope is invalid") from None
        if checked in identities:
            raise ValueError("retrieval snapshot scope contains duplicates")
        identity = (checked.project_id, checked.document_id)
        if identity in documents:
            raise ValueError("retrieval snapshot scope contains multiple document versions")
        identities.add(checked)
        documents.add(identity)
        projects.add(checked.project_id)
    if len(projects) > 1:
        raise ValueError("retrieval snapshot scope must belong to one project")
    return prepared


def _validate_candidate_chunks(
    chunks: Sequence[EvidenceChunk], chunker_version: str
) -> None:
    if (
        not isinstance(chunker_version, str)
        or not chunker_version
        or chunker_version != chunker_version.strip()
        or len(chunker_version) > 80
    ):
        raise ValueError("retrieval chunker version is invalid")
    ids: set[str] = set()
    for chunk in chunks:
        if not isinstance(chunk, EvidenceChunk) or chunk.chunker_version != chunker_version:
            raise ValueError("retrieval candidate chunk is invalid")
        try:
            EvidenceChunk(
                chunk_id=chunk.chunk_id,
                snapshot=chunk.snapshot,
                chunker_version=chunk.chunker_version,
                ordinal=chunk.ordinal,
                text=chunk.text,
                text_sha256=chunk.text_sha256,
                char_start=chunk.char_start,
                char_end=chunk.char_end,
                line_start=chunk.line_start,
                line_end=chunk.line_end,
            )
        except (TypeError, ValueError):
            raise ValueError("retrieval candidate chunk is invalid") from None
        if chunk.chunk_id in ids:
            raise ValueError("retrieval candidate chunks contain duplicates")
        ids.add(chunk.chunk_id)


def _batches(values: tuple[EvidenceChunk, ...], size: int):
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _provider_reason(exc: EmbeddingProviderError) -> str:
    if isinstance(exc, EmbeddingNotConfiguredError):
        return "embedding_not_configured"
    if isinstance(exc, EmbeddingInputError):
        return "embedding_input_rejected"
    if isinstance(exc, EmbeddingResponseError):
        return "embedding_response_invalid"
    if isinstance(exc, EmbeddingRetryExhaustedError):
        return "embedding_retry_exhausted"
    return "embedding_provider_failed"


def _safe_identifier(value: object, limit: int) -> str:
    if (
        isinstance(value, str)
        and value == value.strip()
        and value
        and len(value) <= limit
        and re.fullmatch(r"[A-Za-z0-9_.@-]+", value) is not None
    ):
        return value
    return "invalid"


def _safe_count(value: object) -> int:
    if type(value) is int and 0 <= value <= 1_000_000_000:
        return value
    return 0


def _tokenize(value: str) -> tuple[str, ...]:
    lowered = value.casefold()
    latin = re.findall(r"[a-z0-9_]+", lowered)
    chinese = re.findall(r"[\u4e00-\u9fff]", lowered)
    return tuple(latin + chinese)


def _keyword_ranking(
    query: str, chunks: tuple[EvidenceChunk, ...], limit: int
) -> tuple[str, ...]:
    query_tokens = set(_tokenize(query))
    if not query_tokens:
        return ()
    scored: list[tuple[float, str]] = []
    for chunk in chunks:
        chunk_tokens = set(_tokenize(chunk.text))
        overlap = len(query_tokens & chunk_tokens)
        if overlap:
            score = overlap / math.sqrt(max(1, len(query_tokens) * len(chunk_tokens)))
            scored.append((score, chunk.chunk_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(chunk_id for _, chunk_id in scored[:limit])


def _entity_ranking(
    terms: tuple[str, ...], chunks: tuple[EvidenceChunk, ...], limit: int
) -> tuple[str, ...]:
    if not terms:
        return ()
    folded_terms = tuple(term.casefold() for term in terms)
    scored: list[tuple[int, str]] = []
    for chunk in chunks:
        folded_text = chunk.text.casefold()
        hits = sum(folded_text.count(term) for term in folded_terms)
        if hits:
            scored.append((hits, chunk.chunk_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return tuple(chunk_id for _, chunk_id in scored[:limit])


def _unique_snapshots(
    chunks: tuple[EvidenceChunk, ...],
) -> tuple[SnapshotDocumentKey, ...]:
    values: dict[SnapshotDocumentKey, None] = {}
    for chunk in chunks:
        values.setdefault(chunk.snapshot, None)
    return tuple(values)
