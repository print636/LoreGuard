from __future__ import annotations

import math
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from .embeddings import EmbeddingProfile, EmbeddingProvider
from .evidence_authority import (
    InvestigationScope,
    ScopedEvidenceDocument,
    clone_investigation_scope,
)
from .evidence_chunks import (
    EvidenceChunk,
    EvidenceChunker,
    EvidenceEmbeddingIndex,
    SnapshotDocumentKey,
)
from .evidence_investigator import (
    InvestigationSeed,
    clone_investigation_seed,
)
from .evidence_rag import (
    MAX_RETRIEVAL_CANDIDATES,
    MAX_RETRIEVAL_LIMIT,
    EvidenceDocument,
    EvidenceIndexCoordinator,
    EvidenceIndexDiagnostics,
    EvidenceIndexResult,
    EvidenceQuery,
    EvidenceRetrievalResult,
    EvidenceRrfRetriever,
    RankedEvidence,
    RetrievalDiagnostics,
    RetrievalStrategy,
)
from .evidence_store import MAX_STORE_BATCH, SqlAlchemyEvidenceEmbeddingIndex


_INDEX_OUTCOMES = frozenset({"complete", "provider_unavailable", "write_conflict"})
_INDEX_REASONS = frozenset(
    {
        None,
        "embedding_not_configured",
        "embedding_input_rejected",
        "embedding_response_invalid",
        "embedding_retry_exhausted",
        "embedding_provider_failed",
        "concurrent_write_incomplete",
    }
)
_RETRIEVAL_MODES = frozenset(
    {"hybrid", "lexical_only", "dense_only", "unavailable"}
)
_RETRIEVAL_REASONS = frozenset(
    {
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
)
_SAFE_UNAVAILABLE_REASONS = frozenset(
    {
        "index_incomplete",
        "index_invalid",
        "embedding_not_configured",
        "embedding_input_rejected",
        "embedding_response_invalid",
        "embedding_retry_exhausted",
        "embedding_provider_failed",
        "concurrent_write_incomplete",
        "vector_search_unavailable",
        "vector_scope_invalid",
        "hybrid_unavailable",
        "retrieval_invalid",
    }
)
_STRATEGIES = frozenset(
    {
        "keyword-only",
        "dense-only",
        "keyword+dense-rrf",
        "keyword+vector+entity-rrf",
    }
)


class InvestigatorRagUnavailable(RuntimeError):
    """Content-free terminal RAG failure for the investigator runtime.

    The tool loop deliberately treats this like any retrieval failure and
    fails closed.  Cooperative cancellation/lease exceptions are *not*
    converted to this type; they pass through every adapter checkpoint.
    """

    def __init__(self, reason_code: str) -> None:
        safe = (
            reason_code
            if type(reason_code) is str and reason_code in _SAFE_UNAVAILABLE_REASONS
            else "retrieval_invalid"
        )
        super().__init__(safe)
        self.reason_code = safe


@dataclass(frozen=True, slots=True)
class InvestigatorRagPolicy:
    """Server-owned retrieval policy; no field is supplied by the model."""

    strategy: RetrievalStrategy = "keyword+vector+entity-rrf"
    top_k: int = 12
    branch_limit: int = 30
    require_hybrid: bool = True

    def __post_init__(self) -> None:
        if type(self.strategy) is not str or self.strategy not in _STRATEGIES:
            raise ValueError("investigator RAG strategy is invalid")
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or not 1 <= self.top_k <= MAX_RETRIEVAL_LIMIT
        ):
            raise ValueError("investigator RAG top-k is invalid")
        if (
            isinstance(self.branch_limit, bool)
            or not isinstance(self.branch_limit, int)
            or not 1 <= self.branch_limit <= MAX_RETRIEVAL_LIMIT
        ):
            raise ValueError("investigator RAG branch limit is invalid")
        if type(self.require_hybrid) is not bool:
            raise ValueError("investigator RAG hybrid policy is invalid")
        if self.require_hybrid and self.strategy in {"keyword-only", "dense-only"}:
            raise ValueError("hybrid mode requires a hybrid retrieval strategy")


@dataclass(frozen=True, slots=True)
class _ReadyIndex:
    """One atomically published immutable runtime index state."""

    indexed: EvidenceIndexResult
    chunks_by_id: Mapping[str, EvidenceChunk]


class InvestigatorRagRetriever:
    """Closed-scope real-RAG adapter for ``InvestigatorEvidenceRetriever``.

    One instance represents one analysis runtime.  It derives every document,
    snapshot and chunk authorization from a cloned :class:`InvestigationScope`,
    builds the immutable embedding index at most once, and reuses that index for
    every seed/query.  Only query text and observable entity terms cross the
    model boundary; retrieval strategy, profile, chunker and limits are fixed by
    the server at construction time.

    By default this composes :class:`EvidenceIndexCoordinator`,
    :class:`EvidenceRrfRetriever` and the real
    :class:`SqlAlchemyEvidenceEmbeddingIndex`.  Explicit component injection is
    retained as a narrow test seam and must still pass all scope validations.
    """

    def __init__(
        self,
        *,
        scope: InvestigationScope,
        session_factory: Callable,
        embedding_provider: EmbeddingProvider,
        policy: InvestigatorRagPolicy | None = None,
        chunker: EvidenceChunker | None = None,
        embedding_index: EvidenceEmbeddingIndex | None = None,
        batch_size: int = 64,
        rrf_k: int = 60,
        checkpoint: Callable[[], None] | None = None,
        index_coordinator: object | None = None,
        rrf_retriever: object | None = None,
    ) -> None:
        if not callable(session_factory):
            raise TypeError("session factory must be callable")
        if checkpoint is not None and not callable(checkpoint):
            raise TypeError("checkpoint must be callable")
        if not callable(getattr(embedding_provider, "embed", None)):
            raise TypeError("embedding provider is invalid")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or not 1 <= batch_size <= MAX_STORE_BATCH
        ):
            raise ValueError("embedding batch size is invalid")
        if (
            isinstance(rrf_k, bool)
            or not isinstance(rrf_k, int)
            or not 1 <= rrf_k <= 1_000
        ):
            raise ValueError("RRF constant is invalid")
        if policy is not None and type(policy) is not InvestigatorRagPolicy:
            raise TypeError("investigator RAG policy is invalid")
        if chunker is not None and type(chunker) is not EvidenceChunker:
            raise TypeError("investigator RAG chunker is invalid")
        if index_coordinator is not None and not callable(
            getattr(index_coordinator, "ensure_index", None)
        ):
            raise TypeError("investigator index coordinator is invalid")
        if rrf_retriever is not None and not callable(
            getattr(rrf_retriever, "retrieve", None)
        ):
            raise TypeError("investigator RRF retriever is invalid")
        if embedding_index is not None and rrf_retriever is not None:
            raise ValueError("embedding index and injected retriever conflict")

        try:
            checked_scope = clone_investigation_scope(scope)
        except (TypeError, ValueError):
            raise ValueError("investigator RAG scope is invalid") from None
        checked_policy = policy or InvestigatorRagPolicy()
        checked_chunker = chunker or EvidenceChunker()
        checked_checkpoint = checkpoint or (lambda: None)
        documents = tuple(
            EvidenceDocument(
                snapshot=_copy_snapshot(document.snapshot),
                content=document.content,
            )
            for document in checked_scope.documents
        )
        snapshots = tuple(document.snapshot for document in documents)
        expected_chunks = tuple(
            chunk
            for document in documents
            for chunk in checked_chunker.chunk(
                project_id=document.snapshot.project_id,
                document_id=document.snapshot.document_id,
                document_version=document.snapshot.document_version,
                content=document.content,
                content_sha256=document.snapshot.content_sha256,
            )
        )
        if len(expected_chunks) > MAX_RETRIEVAL_CANDIDATES:
            raise ValueError("investigator RAG chunk limit was exceeded")

        coordinator = (
            index_coordinator
            if index_coordinator is not None
            else EvidenceIndexCoordinator(
                session_factory=session_factory,
                provider=embedding_provider,
                chunker=checked_chunker,
                batch_size=batch_size,
                checkpoint=checked_checkpoint,
            )
        )
        retriever = (
            rrf_retriever
            if rrf_retriever is not None
            else EvidenceRrfRetriever(
                index=(
                    embedding_index
                    if embedding_index is not None
                    else SqlAlchemyEvidenceEmbeddingIndex(session_factory)
                ),
                provider=embedding_provider,
                rrf_k=rrf_k,
                checkpoint=checked_checkpoint,
            )
        )

        self._scope = checked_scope
        self._documents = documents
        self._snapshots = snapshots
        self._expected_chunks = expected_chunks
        self._documents_by_id = {
            document.snapshot.document_id: document
            for document in checked_scope.documents
        }
        self._policy = InvestigatorRagPolicy(
            strategy=checked_policy.strategy,
            top_k=checked_policy.top_k,
            branch_limit=checked_policy.branch_limit,
            require_hybrid=checked_policy.require_hybrid,
        )
        self._chunker_version = checked_chunker.version
        self._checkpoint = checked_checkpoint
        self._coordinator = coordinator
        self._retriever = retriever
        self._index_lock = threading.Lock()
        self._index_attempted = False
        self._ready: _ReadyIndex | None = None
        self._index_failure_reason: str | None = None
        self._diagnostic_lock = threading.Lock()
        self._retrieval_diagnostics: list[dict[str, object]] = []
        self._total_retrievals = 0

    def prepare(self) -> None:
        """Eagerly build/validate the runtime index without exposing it."""

        self._ensure_index()

    def safe_diagnostics(self) -> dict[str, object]:
        """Return bounded, content-free index and retrieval telemetry."""

        ready = self._ready
        index = (
            ready.indexed.diagnostics.safe_dict()
            if ready is not None
            else {
                "outcome": "unavailable",
                "reason": self._index_failure_reason,
                "provider_calls": 0,
                "provider_input_chars": 0,
            }
        )
        with self._diagnostic_lock:
            retrievals = [dict(row) for row in self._retrieval_diagnostics]
            total_retrievals = self._total_retrievals
        return {
            "index": index,
            "retrievals": retrievals,
            "total_retrievals": total_retrievals,
            "retrievals_truncated": total_retrievals > len(retrievals),
            "embedding_activity": {
                "index_calls": int(index.get("provider_calls", 0) or 0),
                "index_input_chars": int(
                    index.get("provider_input_chars", 0) or 0
                ),
                "query_calls": sum(
                    int(row.get("provider_calls", 0) or 0)
                    for row in retrievals
                ),
                "query_input_chars": sum(
                    int(row.get("provider_input_chars", 0) or 0)
                    for row in retrievals
                ),
                "accounting": "calls_and_characters_only_not_chat_tokens",
            },
        }

    def search(
        self,
        *,
        seed: InvestigationSeed,
        query: EvidenceQuery,
        limit: int,
    ) -> Sequence[EvidenceChunk]:
        checked_seed = _clone_and_validate_seed(
            seed,
            scope=self._scope,
            documents_by_id=self._documents_by_id,
        )
        checked_query = _copy_query(query)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= MAX_RETRIEVAL_LIMIT
        ):
            raise ValueError("investigator retrieval limit is invalid")
        effective_limit = min(limit, self._policy.top_k)
        ready = self._ensure_index()

        # Checkpoint failures intentionally pass through unchanged.  The loop's
        # passthrough exception contract distinguishes cancellation/lease loss
        # from an ordinary retrieval degradation.
        self._checkpoint()
        retrieved = self._retriever.retrieve(
            indexed=ready.indexed,
            allowed_snapshots=self._snapshots,
            query=checked_query,
            strategy=self._policy.strategy,
            limit=effective_limit,
            branch_limit=self._policy.branch_limit,
        )
        self._checkpoint()
        return self._validated_matches(
            retrieved,
            ready=ready,
            limit=effective_limit,
            seed=checked_seed,
            query=checked_query,
        )

    def _ensure_index(self) -> _ReadyIndex:
        ready = self._ready
        if ready is not None:
            return ready
        with self._index_lock:
            ready = self._ready
            if ready is not None:
                return ready
            if self._index_attempted:
                raise InvestigatorRagUnavailable(
                    self._index_failure_reason or "index_incomplete"
                )
            self._index_attempted = True
            self._checkpoint()
            indexed = self._coordinator.ensure_index(self._documents)
            self._checkpoint()
            try:
                checked, chunks_by_id = self._validate_index(indexed)
            except (TypeError, ValueError):
                self._index_failure_reason = "index_invalid"
                raise InvestigatorRagUnavailable("index_invalid") from None
            if not checked.complete:
                reason = checked.diagnostics.reason or "index_incomplete"
                self._index_failure_reason = (
                    reason if reason in _SAFE_UNAVAILABLE_REASONS else "index_incomplete"
                )
                raise InvestigatorRagUnavailable(self._index_failure_reason)
            # Publish the index and its exact-chunk lookup in one pointer write.
            # A concurrent search can therefore observe either no ready state
            # (and wait on the lock) or the complete immutable pair, never a
            # half-initialized index with an empty authorization map.
            ready = _ReadyIndex(
                indexed=checked,
                chunks_by_id=MappingProxyType(dict(chunks_by_id)),
            )
            self._ready = ready
            return ready

    def _validate_index(
        self, supplied: object
    ) -> tuple[EvidenceIndexResult, dict[str, EvidenceChunk]]:
        if type(supplied) is not EvidenceIndexResult:
            raise ValueError("investigator index result is invalid")
        if type(supplied.snapshots) is not tuple or type(supplied.chunks) is not tuple:
            raise ValueError("investigator index scope is invalid")
        profile = _copy_profile(supplied.profile)
        snapshots = tuple(_copy_snapshot(row) for row in supplied.snapshots)
        if snapshots != self._snapshots:
            raise ValueError("investigator index snapshots do not match the scope")
        diagnostics = _copy_index_diagnostics(supplied.diagnostics)
        if (
            diagnostics.outcome not in _INDEX_OUTCOMES
            or diagnostics.reason not in _INDEX_REASONS
            or diagnostics.profile_id != profile.profile_id
            or diagnostics.chunker_version != self._chunker_version
            or diagnostics.document_count != len(self._documents)
            or diagnostics.expected_chunks != len(supplied.chunks)
            or (diagnostics.outcome == "complete" and diagnostics.reason is not None)
        ):
            raise ValueError("investigator index diagnostics are invalid")
        chunks = tuple(_copy_chunk(row) for row in supplied.chunks)
        if len(chunks) > MAX_RETRIEVAL_CANDIDATES:
            raise ValueError("investigator index is too large")
        chunks_by_id: dict[str, EvidenceChunk] = {}
        scope_documents = {
            document.snapshot: document for document in self._scope.documents
        }
        for chunk in chunks:
            if chunk.chunker_version != self._chunker_version:
                raise ValueError("investigator chunker identity is invalid")
            document = scope_documents.get(chunk.snapshot)
            if document is None or not _chunk_matches_document(chunk, document.normalized_content):
                raise ValueError("investigator chunk is outside the scope")
            if chunk.chunk_id in chunks_by_id:
                raise ValueError("investigator index contains duplicate chunks")
            chunks_by_id[chunk.chunk_id] = chunk
        if chunks != self._expected_chunks:
            raise ValueError("investigator index does not exactly cover the scope")
        if diagnostics.reused_chunks + diagnostics.embedded_chunks > len(chunks):
            raise ValueError("investigator index accounting is invalid")
        if (
            diagnostics.outcome == "complete"
            and diagnostics.reused_chunks + diagnostics.embedded_chunks != len(chunks)
        ):
            raise ValueError("investigator index accounting is incomplete")
        return (
            EvidenceIndexResult(
                profile=profile,
                snapshots=snapshots,
                chunks=chunks,
                diagnostics=diagnostics,
            ),
            chunks_by_id,
        )

    def _validated_matches(
        self,
        supplied: object,
        *,
        ready: _ReadyIndex,
        limit: int,
        seed: InvestigationSeed,
        query: EvidenceQuery,
    ) -> tuple[EvidenceChunk, ...]:
        # ``seed`` is deliberately retained in this boundary even though RRF
        # does not use its semantics: validating it before index/search prevents
        # cross-run callers from probing an otherwise valid runtime index.
        indexed = ready.indexed
        if seed.run_hash != self._scope.run_hash:
            raise ValueError("investigator seed is outside the scope")
        if type(supplied) is not EvidenceRetrievalResult:
            raise InvestigatorRagUnavailable("retrieval_invalid")
        if type(supplied.matches) is not tuple:
            raise InvestigatorRagUnavailable("retrieval_invalid")
        diagnostics = _copy_retrieval_diagnostics(supplied.diagnostics)
        matches = supplied.matches
        if (
            diagnostics.strategy != self._policy.strategy
            or diagnostics.mode not in _RETRIEVAL_MODES
            or diagnostics.reason not in _RETRIEVAL_REASONS
            or diagnostics.profile_id != indexed.profile.profile_id
            or diagnostics.chunker_version != indexed.diagnostics.chunker_version
            or diagnostics.candidate_count != len(indexed.chunks)
            or diagnostics.result_count != len(matches)
            or len(matches) > limit
        ):
            raise InvestigatorRagUnavailable("retrieval_invalid")
        if not _retrieval_counts_are_valid(
            diagnostics,
            branch_limit=self._policy.branch_limit,
            query=query,
        ):
            raise InvestigatorRagUnavailable("retrieval_invalid")
        if diagnostics.mode != _expected_mode(diagnostics):
            raise InvestigatorRagUnavailable("retrieval_invalid")
        if diagnostics.reason is not None:
            raise InvestigatorRagUnavailable(diagnostics.reason)
        if self._policy.require_hybrid and diagnostics.mode != "hybrid":
            raise InvestigatorRagUnavailable("hybrid_unavailable")

        deduplicated: dict[str, tuple[float, EvidenceChunk]] = {}
        for supplied_match in matches:
            if type(supplied_match) is not RankedEvidence:
                raise InvestigatorRagUnavailable("retrieval_invalid")
            try:
                chunk = _copy_chunk(supplied_match.chunk)
            except (TypeError, ValueError):
                raise InvestigatorRagUnavailable("retrieval_invalid") from None
            indexed_chunk = ready.chunks_by_id.get(chunk.chunk_id)
            if indexed_chunk is None or chunk != indexed_chunk:
                raise InvestigatorRagUnavailable("retrieval_invalid")
            score = supplied_match.rrf_score
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
                or float(score) <= 0
                or not _rank_is_valid(
                    supplied_match.keyword_rank,
                    diagnostics.keyword_hits,
                )
                or not _rank_is_valid(
                    supplied_match.vector_rank,
                    diagnostics.vector_hits,
                )
                or not _rank_is_valid(
                    supplied_match.entity_rank,
                    diagnostics.entity_hits,
                )
                or (
                    supplied_match.keyword_rank is None
                    and supplied_match.vector_rank is None
                    and supplied_match.entity_rank is None
                )
            ):
                raise InvestigatorRagUnavailable("retrieval_invalid")
            current = deduplicated.get(chunk.chunk_id)
            candidate = (float(score), chunk)
            if current is None or candidate[0] > current[0]:
                deduplicated[chunk.chunk_id] = candidate

        ordered = sorted(
            deduplicated.values(),
            key=lambda row: (-row[0], row[1].chunk_id),
        )
        with self._diagnostic_lock:
            self._total_retrievals += 1
            if len(self._retrieval_diagnostics) < 16:
                self._retrieval_diagnostics.append(diagnostics.safe_dict())
        return tuple(_copy_chunk(row[1]) for row in ordered[:limit])


def _clone_and_validate_seed(
    seed: InvestigationSeed,
    *,
    scope: InvestigationScope,
    documents_by_id: dict[str, ScopedEvidenceDocument],
) -> InvestigationSeed:
    try:
        checked = clone_investigation_seed(seed)
    except (TypeError, ValueError):
        raise ValueError("investigator seed is invalid") from None
    if checked.run_hash != scope.run_hash:
        raise ValueError("investigator seed is outside the run")
    document = documents_by_id.get(checked.anchor.evidence.document_id)
    if document is None or not _anchor_matches_document(
        checked,
        document.lines,
    ):
        raise ValueError("investigator seed anchor is outside the scope")
    return checked


def _anchor_matches_document(seed: InvestigationSeed, lines: tuple[str, ...]) -> bool:
    evidence = seed.anchor.evidence
    start, end = evidence.line_start, evidence.line_end
    if (
        type(start) is not int
        or type(end) is not int
        or start < 1
        or end < start
        or end > len(lines)
        or type(evidence.text) is not str
    ):
        return False
    source = "\n".join(lines[start - 1 : end]).strip()
    if evidence.text == source:
        return True
    if start == end and source.startswith("@") and "|" in source:
        declared = source.partition("|")[2].strip() or source
        return evidence.text == declared
    return False


def _copy_query(query: EvidenceQuery) -> EvidenceQuery:
    if type(query) is not EvidenceQuery:
        raise ValueError("investigator evidence query is invalid")
    try:
        if type(query.entity_terms) is not tuple:
            raise ValueError
        return EvidenceQuery(text=query.text, entity_terms=tuple(query.entity_terms))
    except (AttributeError, TypeError, ValueError):
        raise ValueError("investigator evidence query is invalid") from None


def _copy_snapshot(snapshot: SnapshotDocumentKey) -> SnapshotDocumentKey:
    if type(snapshot) is not SnapshotDocumentKey:
        raise ValueError("snapshot is invalid")
    return SnapshotDocumentKey(
        project_id=snapshot.project_id,
        document_id=snapshot.document_id,
        document_version=snapshot.document_version,
        content_sha256=snapshot.content_sha256,
    )


def _copy_profile(profile: EmbeddingProfile) -> EmbeddingProfile:
    if type(profile) is not EmbeddingProfile:
        raise ValueError("embedding profile is invalid")
    return EmbeddingProfile(
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


def _copy_chunk(chunk: EvidenceChunk) -> EvidenceChunk:
    if type(chunk) is not EvidenceChunk:
        raise ValueError("evidence chunk is invalid")
    return EvidenceChunk(
        chunk_id=chunk.chunk_id,
        snapshot=_copy_snapshot(chunk.snapshot),
        chunker_version=chunk.chunker_version,
        ordinal=chunk.ordinal,
        text=chunk.text,
        text_sha256=chunk.text_sha256,
        char_start=chunk.char_start,
        char_end=chunk.char_end,
        line_start=chunk.line_start,
        line_end=chunk.line_end,
    )


def _copy_index_diagnostics(value: EvidenceIndexDiagnostics) -> EvidenceIndexDiagnostics:
    if type(value) is not EvidenceIndexDiagnostics:
        raise ValueError("index diagnostics are invalid")
    counts = (
        value.document_count,
        value.expected_chunks,
        value.reused_chunks,
        value.embedded_chunks,
        value.provider_calls,
        value.provider_input_chars,
        value.elapsed_ms,
    )
    if any(type(row) is not int or row < 0 for row in counts):
        raise ValueError("index diagnostics are invalid")
    return EvidenceIndexDiagnostics(
        outcome=value.outcome,
        reason=value.reason,
        profile_id=value.profile_id,
        chunker_version=value.chunker_version,
        document_count=value.document_count,
        expected_chunks=value.expected_chunks,
        reused_chunks=value.reused_chunks,
        embedded_chunks=value.embedded_chunks,
        provider_calls=value.provider_calls,
        provider_input_chars=value.provider_input_chars,
        elapsed_ms=value.elapsed_ms,
    )


def _copy_retrieval_diagnostics(value: RetrievalDiagnostics) -> RetrievalDiagnostics:
    if type(value) is not RetrievalDiagnostics:
        raise InvestigatorRagUnavailable("retrieval_invalid")
    counts = (
        value.candidate_count,
        value.keyword_hits,
        value.vector_hits,
        value.entity_hits,
        value.result_count,
        value.provider_calls,
        value.provider_input_chars,
        value.elapsed_ms,
    )
    if any(type(row) is not int or row < 0 for row in counts):
        raise InvestigatorRagUnavailable("retrieval_invalid")
    return RetrievalDiagnostics(
        strategy=value.strategy,
        mode=value.mode,
        reason=value.reason,
        profile_id=value.profile_id,
        chunker_version=value.chunker_version,
        candidate_count=value.candidate_count,
        keyword_hits=value.keyword_hits,
        vector_hits=value.vector_hits,
        entity_hits=value.entity_hits,
        result_count=value.result_count,
        provider_calls=value.provider_calls,
        provider_input_chars=value.provider_input_chars,
        elapsed_ms=value.elapsed_ms,
    )


def _chunk_matches_document(chunk: EvidenceChunk, normalized_content: str) -> bool:
    if chunk.char_end > len(normalized_content):
        return False
    if normalized_content[chunk.char_start : chunk.char_end] != chunk.text:
        return False
    line_start = normalized_content.count("\n", 0, chunk.char_start) + 1
    line_end = normalized_content.count("\n", 0, chunk.char_end - 1) + 1
    return chunk.line_start == line_start and chunk.line_end == line_end


def _retrieval_counts_are_valid(
    diagnostics: RetrievalDiagnostics,
    *,
    branch_limit: int,
    query: EvidenceQuery,
) -> bool:
    if any(
        value > min(branch_limit, diagnostics.candidate_count)
        for value in (
            diagnostics.keyword_hits,
            diagnostics.vector_hits,
            diagnostics.entity_hits,
        )
    ):
        return False
    if diagnostics.provider_calls not in {0, 1}:
        return False
    if (
        diagnostics.provider_input_chars
        != (len(query.text) if diagnostics.provider_calls else 0)
    ):
        return False
    if not query.entity_terms and diagnostics.entity_hits:
        return False
    if diagnostics.strategy == "keyword-only":
        return diagnostics.vector_hits == 0 and diagnostics.entity_hits == 0
    if diagnostics.strategy == "dense-only":
        return diagnostics.keyword_hits == 0 and diagnostics.entity_hits == 0
    if diagnostics.strategy == "keyword+dense-rrf":
        return diagnostics.entity_hits == 0
    return True


def _expected_mode(
    diagnostics: RetrievalDiagnostics,
) -> Literal["hybrid", "lexical_only", "dense_only", "unavailable"]:
    if diagnostics.vector_hits and (
        diagnostics.keyword_hits or diagnostics.entity_hits
    ):
        return "hybrid"
    if diagnostics.vector_hits:
        return "dense_only"
    if (
        diagnostics.keyword_hits
        or diagnostics.entity_hits
        or diagnostics.strategy == "keyword-only"
    ):
        return "lexical_only"
    return "unavailable"


def _rank_is_valid(value: int | None, hit_count: int) -> bool:
    return value is None or (
        type(value) is int and 1 <= value <= hit_count
    )
