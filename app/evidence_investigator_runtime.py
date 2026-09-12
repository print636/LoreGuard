from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import islice
from threading import Lock
from typing import Any

from .candidate_promotion import (
    CandidateEvidenceResolver,
    CandidatePromotionLimits,
    CandidatePromotionResult,
    TrustedDocumentContext,
    promote_investigator_candidates,
)
from .config import Settings, get_settings
from .domain import ConsistencyIssue, ParsedDirective, ProviderCallDiagnostics
from .embeddings import (
    EmbeddingInputError,
    EmbeddingProvider,
    EmbeddingRetryExhaustedError,
    OpenAICompatibleEmbeddingProvider,
)
from .evidence_authority import InvestigationScope, ScopedEvidenceDocument
from .evidence_chunks import EvidenceChunker, SnapshotDocumentKey
from .evidence_investigator import build_investigation_seeds
from .evidence_investigator_loop import (
    EvidenceInvestigatorToolLoop,
    InvestigatorLoopPolicy,
)
from .evidence_investigator_rag import (
    InvestigatorRagPolicy,
    InvestigatorRagRetriever,
    InvestigatorRagUnavailable,
)
from .evidence_investigator_state import InvestigatorLimits
from .evidence_rag import EvidenceDocument
from .pipeline import DocumentInput
from .provider import OpenAICompatibleProvider


# Keep the exact production provider type stable under constructor patching in
# integration tests; runtime decisions must not depend on a mutable module name.
_OPENAI_PROVIDER_TYPE = OpenAICompatibleProvider


_MAX_BUNDLE_DOCUMENTS = 256
_MAX_BASELINE_DIRECTIVES = 100_000
_MAX_BASELINE_ISSUES = 100_000
_MAX_SAFE_COUNTER = (1 << 63) - 1
_DOCUMENT_ROLES = frozenset(
    {"canon", "character_profile", "chapter", "reference"}
)
_SAFE_REASONS = frozenset(
    {
        "completed",
        "no_seeds",
        "chat_not_configured",
        "token_budget",
        "deadline",
        "embedding_not_configured",
        "embedding_input_rejected",
        "embedding_response_invalid",
        "embedding_retry_exhausted",
        "embedding_provider_failed",
        "concurrent_write_incomplete",
        "vector_search_unavailable",
        "vector_scope_invalid",
        "hybrid_unavailable",
        "index_incomplete",
        "index_invalid",
        "retrieval_invalid",
        "retrieval_failed",
        "loop_degraded",
        "promotion_failed",
        "internal_failure",
    }
)
_SAFE_PROVIDER_CATEGORIES = frozenset(
    {
        "success",
        "not_configured",
        "provider",
        "rate_limit",
        "upstream_5xx",
        "unauthorized",
        "forbidden",
        "nonretry_http",
        "body_json",
        "response_shape",
        "empty_content",
        "usage_shape",
        "truncated",
        "content_json",
        "connect_timeout",
        "read_timeout",
        "transport",
        "response_too_large",
        "tool_response_json",
        "tool_response_shape",
        "tool_response_finish_reason",
        "tool_calls_shape",
        "tool_calls_too_many",
        "tool_call_shape",
        "tool_call_id_invalid",
        "tool_call_id_duplicate",
        "tool_call_name_invalid",
        "tool_call_unknown",
        "tool_choice_mismatch",
        "tool_call_arguments_json",
        "tool_call_arguments_shape",
        "tool_call_arguments_too_large",
        "tool_calls_missing",
        "tool_request_rejected",
        "usage_unavailable",
        "provider_contract_invalid",
    }
)


class _InvestigatorDeadlineExceeded(RuntimeError):
    pass


@dataclass(frozen=True, slots=True, repr=False)
class FrozenInvestigationBundle:
    """One immutable projection of the already verified run-input snapshot."""

    project_id: str = field(repr=False)
    evidence_documents: tuple[EvidenceDocument, ...] = field(repr=False)
    scoped_documents: tuple[ScopedEvidenceDocument, ...] = field(repr=False)
    trusted_contexts: tuple[TrustedDocumentContext, ...] = field(repr=False)
    fingerprint: str

    def scope(self, run_id: str) -> InvestigationScope:
        return InvestigationScope.create(
            run_id=run_id,
            project_id=self.project_id,
            documents=self.scoped_documents,
        )


def build_frozen_investigation_bundle(
    *,
    project_id: str,
    documents: Sequence[DocumentInput],
    metadata: Sequence[dict],
) -> FrozenInvestigationBundle:
    """Bind three consumers to the exact same verified snapshot rows.

    No live ``DocumentRow`` is accepted here.  A mismatched name, version,
    hash, role, scope, cardinality, or ordinal invalidates the whole bundle.
    """

    prepared_documents = _bounded_tuple(
        documents, _MAX_BUNDLE_DOCUMENTS, "snapshot documents are invalid"
    )
    prepared_metadata = _bounded_tuple(
        metadata, _MAX_BUNDLE_DOCUMENTS, "snapshot metadata is invalid"
    )
    if not prepared_documents or len(prepared_documents) != len(prepared_metadata):
        raise ValueError("snapshot bundle cardinality is invalid")
    evidence_documents: list[EvidenceDocument] = []
    scoped_documents: list[ScopedEvidenceDocument] = []
    contexts: list[TrustedDocumentContext] = []
    fingerprint_rows: list[dict[str, object]] = []
    seen_documents: set[str] = set()

    for ordinal, (document, source) in enumerate(
        zip(prepared_documents, prepared_metadata, strict=True)
    ):
        if type(document) is not DocumentInput or type(source) is not dict:
            raise ValueError("snapshot bundle row is invalid")
        document_id = source.get("document_id")
        document_name = source.get("document_name")
        document_version = source.get("document_version")
        content_sha256 = source.get("content_sha256")
        document_role = source.get("document_role")
        story_scope = source.get("story_scope")
        if (
            type(document_id) is not str
            or type(document_name) is not str
            or type(document_version) is not int
            or isinstance(document_version, bool)
            or document_version < 1
            or type(content_sha256) is not str
            or type(document_role) is not str
            or document_role not in _DOCUMENT_ROLES
            or type(story_scope) is not str
            or not story_scope
            or type(source.get("context_explicit")) is not bool
            or type(source.get("ordinal")) is not int
            or isinstance(source.get("ordinal"), bool)
            or source.get("ordinal") != ordinal
            or type(source.get("char_count")) is not int
            or isinstance(source.get("char_count"), bool)
            or type(document.id) is not str
            or type(document.name) is not str
            or type(document.content) is not str
            or type(document.role) is not str
            or type(document.scope) is not str
            or source.get("char_count") != len(document.content)
            or document.id != document_id
            or document.name != document_name
            or document.role != document_role
            or document.scope != story_scope
            or document_id in seen_documents
            or hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            != content_sha256
        ):
            raise ValueError("snapshot bundle row is invalid")
        snapshot = SnapshotDocumentKey(
            project_id=project_id,
            document_id=document_id,
            document_version=document_version,
            content_sha256=content_sha256,
        )
        evidence_documents.append(
            EvidenceDocument(snapshot=snapshot, content=document.content)
        )
        scoped_documents.append(
            ScopedEvidenceDocument(snapshot=snapshot, content=document.content)
        )
        contexts.append(
            TrustedDocumentContext(
                document_id=document_id,
                document_name=document_name,
                story_scope=story_scope,
                document_role=document_role,
            )
        )
        fingerprint_rows.append(
            {
                "project_id": project_id,
                "document_id": document_id,
                "document_name": document_name,
                "document_version": document_version,
                "content_sha256": content_sha256,
                "document_role": document_role,
                "story_scope": story_scope,
                "ordinal": ordinal,
            }
        )
        seen_documents.add(document_id)

    fingerprint = hashlib.sha256(
        json.dumps(
            fingerprint_rows,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return FrozenInvestigationBundle(
        project_id=project_id,
        evidence_documents=tuple(evidence_documents),
        scoped_documents=tuple(scoped_documents),
        trusted_contexts=tuple(contexts),
        fingerprint=fingerprint,
    )


@dataclass(slots=True)
class InvestigatorUsageAccumulator:
    """Immediate, content-free ledger for native-tool chat calls only."""

    logical_calls: int = 0
    reported_prompt_tokens: int = 0
    reported_completion_tokens: int = 0
    charged_tokens: int = 0
    provider_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def prompt_tokens(self) -> int:
        return self.reported_prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self.reported_completion_tokens

    def record(
        self,
        telemetry: Any,
        succeeded: bool,
        prompt_tokens: int,
        completion_tokens: int,
        charged_tokens: int,
        category: str | None = None,
    ) -> None:
        prompt = _safe_count(prompt_tokens)
        completion = _safe_count(completion_tokens)
        charged = max(_safe_count(charged_tokens), prompt + completion)
        safe = ProviderCallDiagnostics.from_telemetry(
            telemetry,
            succeeded=succeeded,
            purpose="investigator",
        )
        if safe is None:
            call = {
                "status": "success" if succeeded else "failure",
                "category": _safe_provider_category(category),
                "purpose": "investigator",
            }
        else:
            call = safe.safe_dict()
            # Correlation IDs are unnecessary at this feature boundary and may
            # resemble secrets.  Persist only bounded operational counters.
            call.pop("request_id", None)
        self.provider_calls.append(call)
        self.logical_calls += 1
        self.reported_prompt_tokens += prompt
        self.reported_completion_tokens += completion
        self.charged_tokens += charged

    def safe_dict(self, *, terminal_status: str) -> dict[str, Any] | None:
        if not self.logical_calls:
            return None
        return {
            "completeness": "completed_calls",
            "scope": "evidence_investigator",
            "terminal_status": (
                terminal_status
                if terminal_status in {"running", "completed", "failed", "cancelled"}
                else "failed"
            ),
            "logical_calls": self.logical_calls,
            "prompt_tokens": self.reported_prompt_tokens,
            "completion_tokens": self.reported_completion_tokens,
            "reported_prompt_tokens": self.reported_prompt_tokens,
            "reported_completion_tokens": self.reported_completion_tokens,
            "charged_tokens": self.charged_tokens,
            "charged_token_semantics": "heuristic_or_reported_internal_debit",
            "provider_calls": [dict(row) for row in self.provider_calls[:64]],
        }


@dataclass(frozen=True, slots=True)
class EvidenceInvestigatorRuntimeResult:
    outcome: str
    reason_code: str
    diagnostics: dict[str, Any]
    promotion: CandidatePromotionResult | None = field(default=None, repr=False)

    @property
    def applied(self) -> bool:
        return bool(self.promotion and self.promotion.accepted_candidates)


class _DeadlineEmbeddingProvider:
    """Constrain embedding calls by one deadline and per-run char quota."""

    def __init__(
        self,
        provider: EmbeddingProvider,
        *,
        settings: Settings,
        deadline: float,
        checkpoint: Callable[[], None],
        monotonic: Callable[[], float],
        max_input_chars: int,
    ) -> None:
        if type(max_input_chars) is not int or not 1 <= max_input_chars <= 1_000_000:
            raise ValueError("investigator embedding input quota is invalid")
        self._provider = provider
        self._settings = settings
        self._deadline = deadline
        self._checkpoint = checkpoint
        self._monotonic = monotonic
        self._max_input_chars = max_input_chars
        self._activity_lock = Lock()
        self._calls = 0
        self._input_chars = 0

    @property
    def profile(self):
        self._checkpoint()
        profile = self._provider.profile
        self._checkpoint()
        return profile

    def embed(self, texts):
        self._checkpoint()
        if isinstance(texts, (str, bytes)):
            raise EmbeddingInputError("embedding input is invalid")
        try:
            prepared = tuple(
                islice(
                    iter(texts),
                    self._settings.embedding_batch_max_items + 1,
                )
            )
        except TypeError:
            raise EmbeddingInputError("embedding input is invalid") from None
        if (
            not prepared
            or len(prepared) > self._settings.embedding_batch_max_items
            or any(type(row) is not str or not row.strip() for row in prepared)
        ):
            raise EmbeddingInputError("embedding input is invalid")
        input_chars = sum(len(row) for row in prepared)
        if input_chars > self._settings.embedding_batch_max_chars:
            raise EmbeddingInputError("embedding input is invalid")
        with self._activity_lock:
            if self._input_chars + input_chars > self._max_input_chars:
                raise EmbeddingInputError(
                    "evidence investigator embedding resource quota exceeded"
                )
            self._calls += 1
            self._input_chars += input_chars
        remaining = self._deadline - self._monotonic()
        if remaining <= 0:
            raise EmbeddingRetryExhaustedError(
                "evidence investigator deadline exhausted"
            )
        provider = self._provider
        if isinstance(provider, OpenAICompatibleEmbeddingProvider):
            bounded = self._settings.model_copy(
                update={
                    "embedding_timeout_seconds": min(
                        self._settings.embedding_timeout_seconds, remaining
                    ),
                    "embedding_total_deadline_seconds": min(
                        self._settings.embedding_total_deadline_seconds,
                        remaining,
                    ),
                }
            )
            provider = OpenAICompatibleEmbeddingProvider(
                bounded,
                transport=getattr(self._provider, "_transport", None),
                monotonic=getattr(
                    self._provider, "_monotonic", time.monotonic
                ),
                sleeper=getattr(self._provider, "_sleeper", time.sleep),
            )
        result = provider.embed(prepared)
        self._checkpoint()
        return result

    def safe_activity(self) -> dict[str, object]:
        with self._activity_lock:
            calls = self._calls
            input_chars = self._input_chars
        return {
            "calls": calls,
            "input_chars": input_chars,
            "max_input_chars": self._max_input_chars,
            "accounting": "resource_quota_not_chat_tokens_or_api_cost",
        }


class _DeadlineNativeToolProvider:
    """Apply the one runtime deadline freshly to every native-tool turn.

    ``OpenAICompatibleProvider`` computes a relative deadline per logical
    call. Reusing one pre-forked instance would therefore grant the full
    runtime allowance again on every decision round. This adapter instead
    derives a new bounded fork from the absolute remaining time immediately
    before each call. The tool loop performs the post-call checkpoint after
    accounting returned telemetry; only the production provider can enforce
    the deadline inside a blocking transport call.
    """

    def __init__(
        self,
        provider: Any,
        *,
        deadline: float,
        checkpoint: Callable[[], None],
        monotonic: Callable[[], float],
    ) -> None:
        if not callable(getattr(provider, "complete_with_tools", None)):
            raise TypeError("investigator native provider is invalid")
        self._provider = provider
        self._deadline = deadline
        self._checkpoint = checkpoint
        self._monotonic = monotonic

    def complete_with_tools(self, system, user, *, tools, tool_choice, limits):
        self._checkpoint()
        remaining = self._deadline - self._monotonic()
        if remaining <= 0:
            raise _InvestigatorDeadlineExceeded(
                "evidence investigator deadline exhausted"
            )
        provider = self._provider
        if isinstance(provider, _OPENAI_PROVIDER_TYPE):
            provider = provider.fork_for_evidence_investigator(
                remaining_deadline_seconds=max(0.001, remaining)
            )
        # Do not checkpoint after this call. The loop must first detach the
        # successful result (or failure telemetry), account its reported usage,
        # and only then run its own checkpoint. Otherwise cancellation or lease
        # loss could erase telemetry from a provider call that already happened.
        return provider.complete_with_tools(
            system,
            user,
            tools=tools,
            tool_choice=tool_choice,
            limits=limits,
        )


class EvidenceInvestigatorRuntime:
    """Run the optional Investigator as one fail-closed post-pipeline stage."""

    def __init__(
        self,
        *,
        session_factory: Callable,
        settings: Settings | None = None,
        checkpoint: Callable[[], None] | None = None,
        usage: InvestigatorUsageAccumulator | None = None,
        passthrough_exceptions: tuple[type[Exception], ...] = (),
        native_provider: Any | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        rag_factory: Callable[..., Any] | None = None,
        loop_factory: Callable[..., Any] = EvidenceInvestigatorToolLoop,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(session_factory):
            raise TypeError("investigator session factory is invalid")
        if checkpoint is not None and not callable(checkpoint):
            raise TypeError("investigator checkpoint is invalid")
        if not callable(loop_factory) or not callable(monotonic):
            raise TypeError("investigator runtime dependency is invalid")
        if type(passthrough_exceptions) is not tuple or any(
            not isinstance(row, type) or not issubclass(row, Exception)
            for row in passthrough_exceptions
        ) or len(passthrough_exceptions) > 7:
            raise TypeError("investigator passthrough exceptions are invalid")
        self._session_factory = session_factory
        self._settings = settings or get_settings()
        self._checkpoint = checkpoint or (lambda: None)
        self._usage = usage or InvestigatorUsageAccumulator()
        self._passthrough_exceptions = passthrough_exceptions
        self._native_provider = native_provider
        self._embedding_provider = embedding_provider
        self._rag_factory = rag_factory
        self._loop_factory = loop_factory
        self._monotonic = monotonic

    @property
    def usage(self) -> InvestigatorUsageAccumulator:
        return self._usage

    def run(
        self,
        *,
        run_id: str,
        bundle: FrozenInvestigationBundle,
        baseline_directives: Sequence[ParsedDirective],
        baseline_issues: Sequence[ConsistencyIssue],
        remaining_run_tokens: int,
    ) -> EvidenceInvestigatorRuntimeResult:
        settings = self._settings
        if not settings.enable_evidence_investigator:
            raise ValueError("disabled investigator runtime must not be invoked")
        if type(bundle) is not FrozenInvestigationBundle:
            raise ValueError("investigator snapshot bundle is invalid")
        if (
            isinstance(remaining_run_tokens, bool)
            or not isinstance(remaining_run_tokens, int)
            or remaining_run_tokens < 0
        ):
            raise ValueError("investigator remaining budget is invalid")
        try:
            directives = _bounded_tuple(
                baseline_directives,
                _MAX_BASELINE_DIRECTIVES,
                "baseline directives are invalid",
            )
            issues = _bounded_tuple(
                baseline_issues,
                _MAX_BASELINE_ISSUES,
                "baseline issues are invalid",
            )
            seeds = build_investigation_seeds(
                run_id,
                directives,
                limit=settings.evidence_investigator_max_seeds,
            )
        except (TypeError, ValueError):
            return self._result("degraded", "internal_failure", bundle, 0)
        if not seeds:
            return self._result("skipped", "no_seeds", bundle, 0)
        try:
            scope = bundle.scope(run_id)
            _validate_seed_anchors(seeds, scope, bundle.trusted_contexts)
        except (AttributeError, TypeError, ValueError):
            return self._result(
                "degraded", "internal_failure", bundle, len(seeds)
            )
        if remaining_run_tokens <= 0:
            return self._result("skipped", "token_budget", bundle, len(seeds))

        deadline = (
            self._monotonic()
            + settings.evidence_investigator_total_deadline_seconds
        )

        def guarded_checkpoint() -> None:
            self._checkpoint()
            if self._monotonic() >= deadline:
                raise _InvestigatorDeadlineExceeded(
                    "evidence investigator deadline exhausted"
                )

        rag = None
        deadline_embedding = None
        preflight = None
        loop_result = None
        try:
            provider = self._native_provider
            if provider is None:
                base_provider = OpenAICompatibleProvider(settings)
                if not base_provider.evidence_investigator_configured:
                    return self._result(
                        "skipped", "chat_not_configured", bundle, len(seeds)
                    )
                provider = base_provider
            provider = _DeadlineNativeToolProvider(
                provider,
                deadline=deadline,
                checkpoint=guarded_checkpoint,
                monotonic=self._monotonic,
            )
            embedding = self._embedding_provider or (
                OpenAICompatibleEmbeddingProvider(settings)
            )
            deadline_embedding = _DeadlineEmbeddingProvider(
                embedding,
                settings=settings,
                deadline=deadline,
                checkpoint=guarded_checkpoint,
                monotonic=self._monotonic,
                max_input_chars=(
                    settings.evidence_investigator_embedding_max_input_chars
                ),
            )
            rag_policy = InvestigatorRagPolicy(
                strategy="keyword+vector+entity-rrf",
                top_k=settings.evidence_investigator_top_k,
                branch_limit=settings.evidence_investigator_branch_limit,
                require_hybrid=settings.evidence_investigator_require_hybrid,
            )
            rag = (
                self._rag_factory(
                    scope=scope,
                    session_factory=self._session_factory,
                    embedding_provider=deadline_embedding,
                    policy=rag_policy,
                    checkpoint=guarded_checkpoint,
                )
                if self._rag_factory is not None
                else InvestigatorRagRetriever(
                    scope=scope,
                    session_factory=self._session_factory,
                    embedding_provider=deadline_embedding,
                    policy=rag_policy,
                    chunker=EvidenceChunker(),
                    batch_size=settings.embedding_batch_max_items,
                    checkpoint=guarded_checkpoint,
                )
            )
            limits = InvestigatorLimits(
                max_decision_rounds=settings.evidence_investigator_max_decision_rounds,
                max_tool_calls=settings.evidence_investigator_max_tool_calls,
                max_searches=settings.evidence_investigator_max_searches,
                max_reads=settings.evidence_investigator_max_reads,
                max_results=settings.evidence_investigator_max_results,
                max_read_lines=settings.evidence_investigator_max_read_lines,
                max_span_chars=settings.evidence_investigator_max_span_chars,
                max_charged_tokens=max(
                    1,
                    min(
                        settings.evidence_investigator_token_budget,
                        remaining_run_tokens,
                    ),
                ),
                deadline_seconds=settings.evidence_investigator_total_deadline_seconds,
            )
            policy = InvestigatorLoopPolicy(
                retrieval_limit=settings.evidence_investigator_top_k,
                max_tool_argument_bytes=32 * 1_024,
                max_prompt_bytes=settings.evidence_investigator_max_prompt_bytes,
                completion_token_reserve=(
                    settings.evidence_investigator_max_completion_tokens
                ),
            )
            loop = self._loop_factory(
                provider=provider,
                retriever=rag,
                scope=scope,
                seeds=seeds,
                limits=limits,
                policy=policy,
                checkpoint=guarded_checkpoint,
                usage_callback=self._usage.record,
                passthrough_exceptions=(
                    *self._passthrough_exceptions,
                    _InvestigatorDeadlineExceeded,
                ),
                monotonic=self._monotonic,
            )
            preflight = loop.budget_preflight()
            if (
                not preflight.minimum_path_admissible
                or preflight.minimum_initial_reservation > remaining_run_tokens
            ):
                return self._result(
                    "skipped",
                    "token_budget",
                    bundle,
                    len(seeds),
                    preflight=preflight,
                    rag=rag,
                    embedding=deadline_embedding,
                )
            guarded_checkpoint()
            rag.prepare()
            guarded_checkpoint()
            loop_result = loop.run()
            if loop_result.outcome != "completed":
                return self._result(
                    "degraded",
                    "loop_degraded",
                    bundle,
                    len(seeds),
                    preflight=preflight,
                    loop_result=loop_result,
                    rag=rag,
                    embedding=deadline_embedding,
                )
            resolver = CandidateEvidenceResolver(
                scope=scope,
                documents=bundle.trusted_contexts,
                investigator_result=loop_result,
            )
            remaining_ms = max(
                1,
                min(5_000, int((deadline - self._monotonic()) * 1_000)),
            )
            promotion = promote_investigator_candidates(
                baseline_directives=directives,
                baseline_issues=issues,
                seeds=seeds,
                evidence_resolver=resolver,
                limits=CandidatePromotionLimits(
                    max_envelopes=max(1, len(seeds)),
                    max_candidates=max(1, min(32, len(seeds) * 2)),
                    max_total_payload_bytes=max(
                        128, min(32_000, len(seeds) * 8_192)
                    ),
                    max_elapsed_ms=remaining_ms,
                ),
                checkpoint=guarded_checkpoint,
                monotonic=self._monotonic,
            )
            return self._result(
                "completed",
                "completed",
                bundle,
                len(seeds),
                preflight=preflight,
                loop_result=loop_result,
                promotion=promotion,
                rag=rag,
                embedding=deadline_embedding,
            )
        except self._passthrough_exceptions:
            raise
        except _InvestigatorDeadlineExceeded:
            reason = "deadline"
        except InvestigatorRagUnavailable as exc:
            reason = _safe_reason(exc.reason_code)
        except Exception:
            reason = "promotion_failed" if loop_result is not None else "internal_failure"
        return self._result(
            "degraded",
            reason,
            bundle,
            len(seeds),
            preflight=preflight,
            loop_result=loop_result,
            rag=rag,
            embedding=deadline_embedding,
        )

    def _result(
        self,
        outcome: str,
        reason: str,
        bundle: FrozenInvestigationBundle,
        seed_count: int,
        *,
        preflight=None,
        loop_result=None,
        promotion: CandidatePromotionResult | None = None,
        rag=None,
        embedding: _DeadlineEmbeddingProvider | None = None,
    ) -> EvidenceInvestigatorRuntimeResult:
        safe_outcome = outcome if outcome in {"completed", "skipped", "degraded"} else "degraded"
        diagnostics: dict[str, Any] = {
            "enabled": True,
            "outcome": safe_outcome,
            "reason_code": _safe_reason(reason),
            "seed_count": _safe_count(seed_count),
            "snapshot_fingerprint": bundle.fingerprint,
            "budget_preflight": (
                preflight.safe_dict() if preflight is not None else None
            ),
            "loop": loop_result.safe_dict() if loop_result is not None else None,
            "rag": _safe_rag_diagnostics(rag, embedding=embedding),
            "promotion": promotion.safe_dict() if promotion is not None else None,
            "usage": self._usage.safe_dict(terminal_status="running"),
            "boundary": (
                "Optional additive stage; any degradation preserves the baseline "
                "directives, issues, IDs, order and provenance."
            ),
        }
        return EvidenceInvestigatorRuntimeResult(
            outcome=safe_outcome,
            reason_code=_safe_reason(reason),
            diagnostics=diagnostics,
            promotion=promotion,
        )


def _safe_rag_diagnostics(
    rag: Any,
    *,
    embedding: _DeadlineEmbeddingProvider | None = None,
) -> dict[str, Any] | None:
    runtime_activity = (
        embedding.safe_activity()
        if type(embedding) is _DeadlineEmbeddingProvider
        else None
    )
    getter = getattr(rag, "safe_diagnostics", None)
    if not callable(getter):
        return (
            {
                "index": _safe_index_diagnostics(None),
                "retrievals": [],
                "total_retrievals": 0,
                "retrievals_truncated": False,
                "embedding_activity": {
                    "index_calls": 0,
                    "index_input_chars": 0,
                    "query_calls": 0,
                    "query_input_chars": 0,
                    "accounting": "calls_and_characters_only_not_chat_tokens",
                },
                "runtime_embedding_quota": runtime_activity,
            }
            if runtime_activity is not None
            else None
        )
    try:
        value = getter()
    except Exception:
        value = {}
    if type(value) is not dict:
        value = {}
    index_source = value.get("index")
    index = _safe_index_diagnostics(index_source)
    retrieval_source = value.get("retrievals")
    retrievals = (
        [
            _safe_retrieval_diagnostics(row)
            for row in retrieval_source[:16]
            if type(row) is dict
        ]
        if type(retrieval_source) is list
        else []
    )
    activity_source = value.get("embedding_activity")
    activity = activity_source if type(activity_source) is dict else {}
    return {
        "index": index,
        "retrievals": retrievals,
        "total_retrievals": _safe_count(value.get("total_retrievals")),
        "retrievals_truncated": value.get("retrievals_truncated") is True,
        "embedding_activity": {
            "index_calls": _safe_count(activity.get("index_calls")),
            "index_input_chars": _safe_count(
                activity.get("index_input_chars")
            ),
            "query_calls": _safe_count(activity.get("query_calls")),
            "query_input_chars": _safe_count(
                activity.get("query_input_chars")
            ),
            "accounting": "calls_and_characters_only_not_chat_tokens",
        },
        "runtime_embedding_quota": runtime_activity,
    }


def _safe_index_diagnostics(value: object) -> dict[str, Any]:
    source = value if type(value) is dict else {}
    outcome = source.get("outcome")
    reason = source.get("reason")
    return {
        "outcome": (
            outcome
            if type(outcome) is str
            and outcome
            in {"complete", "provider_unavailable", "write_conflict", "unavailable"}
            else "unavailable"
        ),
        "reason": (
            reason
            if reason is None
            or type(reason) is str
            and reason
            in {
                "embedding_not_configured",
                "embedding_input_rejected",
                "embedding_response_invalid",
                "embedding_retry_exhausted",
                "embedding_provider_failed",
                "concurrent_write_incomplete",
            }
            else "internal_failure"
        ),
        "profile_fingerprint": _diagnostic_fingerprint(
            source.get("profile_id"), 68
        ),
        "chunker_fingerprint": _diagnostic_fingerprint(
            source.get("chunker_version"), 80
        ),
        **{
            key: _safe_count(source.get(key))
            for key in (
                "document_count",
                "expected_chunks",
                "reused_chunks",
                "embedded_chunks",
                "provider_calls",
                "provider_input_chars",
                "elapsed_ms",
            )
        },
    }


def _safe_retrieval_diagnostics(value: dict) -> dict[str, Any]:
    strategy = value.get("strategy")
    mode = value.get("mode")
    reason = value.get("reason")
    return {
        "strategy": (
            strategy
            if type(strategy) is str
            and strategy
            in {
                "keyword-only",
                "dense-only",
                "keyword+dense-rrf",
                "keyword+vector+entity-rrf",
            }
            else "keyword-only"
        ),
        "mode": (
            mode
            if type(mode) is str
            and mode
            in {"hybrid", "lexical_only", "dense_only", "unavailable"}
            else "unavailable"
        ),
        "reason": (
            reason
            if reason is None
            or type(reason) is str
            and reason
            in {
                "index_incomplete",
                "embedding_not_configured",
                "embedding_input_rejected",
                "embedding_response_invalid",
                "embedding_retry_exhausted",
                "embedding_provider_failed",
                "vector_search_unavailable",
                "vector_scope_invalid",
            }
            else "internal_failure"
        ),
        "profile_fingerprint": _diagnostic_fingerprint(
            value.get("profile_id"), 68
        ),
        "chunker_fingerprint": _diagnostic_fingerprint(
            value.get("chunker_version"), 80
        ),
        **{
            key: _safe_count(value.get(key))
            for key in (
                "candidate_count",
                "keyword_hits",
                "vector_hits",
                "entity_hits",
                "result_count",
                "provider_calls",
                "provider_input_chars",
                "elapsed_ms",
            )
        },
    }


def _diagnostic_fingerprint(value: object, maximum: int) -> str | None:
    if type(value) is not str or not 1 <= len(value) <= maximum:
        return None
    allowed = frozenset(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:@+-"
    )
    if not all(character in allowed for character in value):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_seed_anchors(
    seeds: Sequence,
    scope: InvestigationScope,
    contexts: Sequence[TrustedDocumentContext],
) -> None:
    context_by_id = {row.document_id: row for row in contexts}
    document_by_id = {
        row.snapshot.document_id: row for row in scope.documents
    }
    if (
        len(context_by_id) != len(contexts)
        or set(context_by_id) != set(document_by_id)
    ):
        raise ValueError("investigator scope contexts are invalid")
    for seed in seeds:
        if seed.run_hash != scope.run_hash:
            raise ValueError("investigator seed is outside the run")
        evidence = seed.anchor.evidence
        document = document_by_id.get(evidence.document_id)
        context = context_by_id.get(evidence.document_id)
        if (
            document is None
            or context is None
            or evidence.document_name != context.document_name
            or not _anchor_text_matches(document.lines, evidence)
        ):
            raise ValueError("investigator seed anchor is outside the snapshot")


def _anchor_text_matches(lines: tuple[str, ...], evidence: Any) -> bool:
    start = getattr(evidence, "line_start", None)
    end = getattr(evidence, "line_end", None)
    text = getattr(evidence, "text", None)
    if (
        type(start) is not int
        or type(end) is not int
        or start < 1
        or end < start
        or end > len(lines)
        or type(text) is not str
    ):
        return False
    source = "\n".join(lines[start - 1 : end]).strip()
    if source == text:
        return True
    if start == end and source.startswith("@") and "|" in source:
        declared = source.partition("|")[2].strip() or source
        return declared == text
    return False


def _safe_reason(value: object) -> str:
    return value if type(value) is str and value in _SAFE_REASONS else "internal_failure"


def _safe_provider_category(value: object) -> str:
    return (
        value
        if type(value) is str and value in _SAFE_PROVIDER_CATEGORIES
        else "provider"
    )


def _safe_count(value: object) -> int:
    return (
        value
        if type(value) is int and 0 <= value <= _MAX_SAFE_COUNTER
        else 0
    )


def _bounded_tuple(values: Sequence[Any], maximum: int, message: str) -> tuple:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(message)
    try:
        result = tuple(islice(iter(values), maximum + 1))
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if len(result) > maximum:
        raise ValueError(message)
    return result
