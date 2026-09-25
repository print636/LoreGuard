from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
from math import ceil, isfinite
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from sqlalchemy import and_, delete, exists, func, or_, select, update

from .config import get_settings
from .account_provider import runtime_for_analysis_run
from .db import (
    AnalysisDiagnosticRow,
    AnalysisRecordRow,
    AnalysisRunCharacterTraitInputRow,
    AnalysisRunExecutionRow,
    AnalysisRunInputContextRow,
    AnalysisRunInputNarrativeContextRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    CharacterTraitAxisRow,
    CharacterTraitCandidateRow,
    DocumentContextRow,
    DocumentRow,
    IssueRow,
    RunEventRow,
    SessionLocal,
)
from .character_traits import (
    CHARACTER_TRAIT_SCHEMA_VERSION,
    MAX_CONFIRMED_TRAITS_PER_RUN,
    candidate_snapshot_comparison_key,
    candidate_snapshot_payload,
    latest_confirm_reviews,
)
from .character_consistency_stage import (
    CharacterConsistencyStage,
    failed_character_consistency_stage,
)
from .character_scope_review_provider import SCOPE_REVIEW_SYSTEM_PROMPT
from .character_drift import CHARACTER_REVIEW_SYSTEM_PROMPT
from .character_trait_extraction import (
    CHARACTER_SIGNAL_CORE_SCOPE_PROMPT_V3,
    CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2,
    CHARACTER_SIGNAL_SCOPE_REVIEW_PROMPT_V1,
    CHARACTER_SIGNAL_SEMANTIC_SCOPE_REVIEW_PROMPT_V1,
    CHARACTER_SIGNAL_SEMANTIC_SCOPE_PROMPT_V5,
    CHARACTER_SIGNAL_SUPPORT_ID_PROMPT_V4,
    CHARACTER_SIGNAL_SYSTEM_PROMPT,
    CHARACTER_SIGNAL_SYSTEM_PROMPT_V5,
    TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT,
    _validate_signal_prompt_variant_settings,
    _bounded_provider,
)
from .domain import AnalysisCancelled
from .evidence_investigator_runtime import (
    EvidenceInvestigatorRuntime,
    InvestigatorUsageAccumulator,
    build_frozen_investigation_bundle,
)
from .issue_evidence_review import (
    IssueEvidenceReviewUsageAccumulator,
    IssueEvidenceReviewer,
    failed_issue_evidence_review,
)
from .pipeline import AnalysisPipeline, DocumentInput, build_result_provenance
from .provider import OpenAICompatibleProvider
from .narrative_context import (
    canonical_scope_payload,
    context_snapshot_payload,
    latest_context_revisions,
    narrative_context_semantic_sha256,
    payload_sha256,
)
from .runtime_provenance import safe_runtime_provenance
from .run_comparison import mark_comparison_unverifiable, materialize_run_comparison
from .time_utils import utc_now_naive
from .usage import configured_cost_usd, estimate_issue_evidence_review_tokens


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
TERMINAL_EVENT_STAGES = TERMINAL_STATUSES
WORKER_HEARTBEAT_JOIN_TIMEOUT_SECONDS = 5
DEFAULT_DOCUMENT_ROLE = "chapter"
DEFAULT_STORY_SCOPE = "global"
MISSING_SNAPSHOT_ERROR = (
    "RUN_INPUT_SNAPSHOT_MISSING: 此旧任务创建时未冻结输入，不能读取当前文档冒充原输入；"
    "请从项目重新发起分析以使用当前活动版本"
)
CORRUPT_SNAPSHOT_ERROR = (
    "RUN_INPUT_SNAPSHOT_CORRUPT: 分析输入快照校验失败；请从项目重新发起分析"
)
INTERNAL_ANALYSIS_ERROR = (
    "ANALYSIS_EXECUTION_FAILED: 分析执行失败，内部错误详情已隐藏；请重试任务"
)
DISPATCH_FAILED_ERROR = (
    "ANALYSIS_DISPATCH_FAILED: 分析任务未能进入执行队列；输入快照已保留，可安全重试"
)

_SAFE_INTERRUPTED_PROVIDER_CATEGORIES = {
    "success",
    "not_configured",
    "provider",
    "rate_limit",
    "upstream_5xx",
    "unauthorized",
    "forbidden",
    "nonretry_http",
    "unsupported_content_encoding",
    "response_decompression",
    "body_json",
    "response_shape",
    "empty_content",
    "usage_shape",
    "truncated",
    "content_json",
    "connect_timeout",
    "read_timeout",
    "transport",
}
_SIGNED_64_MAX = (1 << 63) - 1


class WorkerLeaseLost(RuntimeError):
    pass


class WorkerLeaseBusy(RuntimeError):
    def __init__(self, retry_after_seconds: int):
        super().__init__("analysis run is already owned by another worker")
        self.retry_after_seconds = retry_after_seconds


class WorkerLeaseHeartbeatError(WorkerLeaseLost):
    """The background renewal loop could not verify continued ownership."""


class CharacterTraitSnapshotLimitExceeded(RuntimeError):
    pass


def safe_persisted_analysis_error(value: object) -> str | None:
    """Return only an allowlisted analysis error suitable for REST or SSE."""
    if value is None:
        return None
    message = value if isinstance(value, str) else ""
    if message == MISSING_SNAPSHOT_ERROR:
        return MISSING_SNAPSHOT_ERROR
    if message == CORRUPT_SNAPSHOT_ERROR or message.startswith(
        "RUN_INPUT_SNAPSHOT_CORRUPT:"
    ):
        return CORRUPT_SNAPSHOT_ERROR
    if message == INTERNAL_ANALYSIS_ERROR:
        return INTERNAL_ANALYSIS_ERROR
    if message == DISPATCH_FAILED_ERROR:
        return DISPATCH_FAILED_ERROR
    return INTERNAL_ANALYSIS_ERROR


def safe_analysis_error(exc: BaseException) -> str:
    """Map internal exceptions to the small set of messages exposed to clients.

    Worker and provider exceptions can contain URLs, filesystem paths, response
    fragments, or credentials supplied by a deployment.  AnalysisRun.error is
    returned by both the REST API and the SSE terminal event, so arbitrary
    exception text must never be persisted there.
    """
    try:
        message = str(exc)
    except Exception:
        return INTERNAL_ANALYSIS_ERROR
    return safe_persisted_analysis_error(message) or INTERNAL_ANALYSIS_ERROR


def _lease_seconds() -> float:
    return get_settings().worker_lease_seconds


def _heartbeat_seconds() -> float:
    settings = get_settings()
    # Even a bad deployment value must not place renewal close to expiry.
    return min(settings.worker_lease_heartbeat_seconds, settings.worker_lease_seconds / 3)


def _renew_execution_lease(
    run_id: str,
    worker_token: str,
    *,
    now: object | None = None,
    lease_seconds: float | None = None,
) -> bool:
    """Renew only while ``worker_token`` still owns the row.

    Each call deliberately opens and closes its own session.  The heartbeat
    therefore keeps moving while the pipeline thread is blocked in a provider,
    index, or checker call and never shares an ORM session across threads.
    """
    heartbeat_at = now if now is not None else utc_now_naive()
    ttl = _lease_seconds() if lease_seconds is None else lease_seconds
    with SessionLocal() as db:
        renewed = db.execute(
            update(AnalysisRunExecutionRow)
            .where(
                AnalysisRunExecutionRow.run_id == run_id,
                AnalysisRunExecutionRow.worker_token == worker_token,
            )
            .values(
                last_heartbeat_at=heartbeat_at,
                lease_expires_at=heartbeat_at + timedelta(seconds=ttl),
            )
        ).rowcount
        if renewed != 1:
            db.rollback()
            return False
        db.commit()
        return True


class ExecutionLeaseHeartbeat:
    """Daemon renewal loop with a checkpoint-visible, thread-safe failure."""

    def __init__(
        self,
        run_id: str,
        worker_token: str,
        *,
        lease_seconds: float | None = None,
        interval_seconds: float | None = None,
        clock: Callable[[], object] = utc_now_naive,
        renew: Callable[..., bool] = _renew_execution_lease,
    ) -> None:
        self.run_id = run_id
        self.worker_token = worker_token
        self.lease_seconds = _lease_seconds() if lease_seconds is None else lease_seconds
        configured_interval = (
            _heartbeat_seconds() if interval_seconds is None else interval_seconds
        )
        self.interval_seconds = min(configured_interval, self.lease_seconds / 3)
        self.clock = clock
        self.renew = renew
        self._stop = Event()
        self._failed = Event()
        self._failure_lock = Lock()
        self._failure: WorkerLeaseLost | None = None
        self._thread: Thread | None = None

    @property
    def is_alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("execution lease heartbeat was already started")
        self._thread = Thread(
            target=self._run,
            name=f"analysis-lease-{self.run_id}",
            daemon=True,
        )
        self._thread.start()

    def _record_failure(self, failure: WorkerLeaseLost) -> None:
        with self._failure_lock:
            if self._failure is None:
                self._failure = failure
        self._failed.set()
        self._stop.set()

    def beat_once(self) -> bool:
        """Perform one renewal; public for deterministic tests."""
        try:
            renewed = self.renew(
                self.run_id,
                self.worker_token,
                now=self.clock(),
                lease_seconds=self.lease_seconds,
            )
        except Exception as exc:
            failure = WorkerLeaseHeartbeatError(
                f"analysis lease heartbeat failed: {exc}"
            )
            failure.__cause__ = exc
            self._record_failure(failure)
            return False
        if not renewed:
            self._record_failure(WorkerLeaseLost("analysis worker lease was lost"))
            return False
        return True

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            if not self.beat_once():
                return

    def raise_if_failed(self) -> None:
        if not self._failed.is_set():
            return
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise failure

    def stop(self, timeout: float = WORKER_HEARTBEAT_JOIN_TIMEOUT_SECONDS) -> None:
        self._stop.set()
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=max(0, timeout))
        if thread.is_alive():
            raise WorkerLeaseHeartbeatError(
                "analysis lease heartbeat did not stop before the join timeout"
            )


class CharacterConsistencyUsageAccumulator:
    """Content-free accounting for calls completed inside the character stage.

    The semantic stage normally returns aggregate usage, but cancellation and
    an unexpected integration failure can unwind it after one or more provider
    calls have already completed.  Recording immediately at the provider
    boundary keeps those calls chargeable without retaining prompts, source
    text, responses, request ids, URLs, or credentials.
    """

    def __init__(self) -> None:
        self.logical_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.charged_tokens = 0
        self.successful_calls = 0

    @staticmethod
    def _token_count(value: object) -> int:
        return value if type(value) is int and 0 <= value <= _SIGNED_64_MAX // 4 else 0

    def record(
        self,
        *,
        prompt_tokens: object,
        completion_tokens: object,
        charged_tokens: object,
        successful: bool,
    ) -> None:
        prompt = self._token_count(prompt_tokens)
        completion = self._token_count(completion_tokens)
        charged = self._token_count(charged_tokens)
        charged = max(charged, prompt + completion)
        self.logical_calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.charged_tokens += charged
        if successful:
            self.successful_calls += 1

    def safe_dict(self, *, terminal_status: str) -> dict[str, Any] | None:
        if self.logical_calls <= 0:
            return None
        values = (
            self.logical_calls,
            self.prompt_tokens,
            self.completion_tokens,
            self.charged_tokens,
        )
        if any(type(value) is not int or value < 0 for value in values):
            return None
        if self.charged_tokens < self.prompt_tokens + self.completion_tokens:
            return None
        return {
            "completeness": "completed_calls",
            "scope": "character_consistency",
            "terminal_status": terminal_status,
            "logical_calls": self.logical_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "charged_tokens": self.charged_tokens,
            "charged_token_semantics": "heuristic_or_reported_internal_debit",
            "provider_calls": None,
        }


class _CharacterConsistencyAccountingProvider:
    """Stage-bounded provider that records aggregate usage before unwinding."""

    def __init__(
        self,
        settings,
        usage: CharacterConsistencyUsageAccumulator,
        *,
        signal_provider=None,
        drift_provider=None,
        scope_review_provider=None,
        scope_review_completion_reserve: int | None = None,
    ) -> None:
        self.settings = settings
        self.usage = usage
        if signal_provider is None or drift_provider is None:
            base = OpenAICompatibleProvider(settings)
            signal_provider = signal_provider or _bounded_provider(
                base, settings, stage="signal"
            )
            drift_provider = drift_provider or _bounded_provider(
                base, settings, stage="drift"
            )
        self.signal_provider = signal_provider
        self.drift_provider = drift_provider
        self.scope_review_provider = scope_review_provider
        self.scope_review_completion_reserve = scope_review_completion_reserve

    def fork_for_character_consistency(
        self,
        *,
        settings,
        stage: str,
        remaining_deadline_seconds: float | None = None,
    ):
        """Preserve accounting while forwarding per-call resource bounds.

        In particular, package regeneration receives only the remainder of
        the extractor's shared logical deadline instead of starting a fresh
        full provider deadline behind this wrapper.
        """

        if stage == "signal":
            signal_provider = _bounded_provider(
                self.signal_provider,
                settings,
                stage="signal",
                remaining_deadline_seconds=remaining_deadline_seconds,
            )
            drift_provider = self.drift_provider
        elif stage == "drift":
            signal_provider = self.signal_provider
            drift_provider = _bounded_provider(
                self.drift_provider,
                settings,
                stage="drift",
                remaining_deadline_seconds=remaining_deadline_seconds,
            )
        else:
            raise ValueError("unsupported character consistency provider stage")
        return _CharacterConsistencyAccountingProvider(
            settings,
            self.usage,
            signal_provider=signal_provider,
            drift_provider=drift_provider,
        )

    def fork_for_character_scope_review(
        self,
        *,
        timeout_seconds: float,
        remaining_deadline_seconds: float,
        completion_reserve: int,
        max_response_bytes: int,
        max_attempts: int,
    ):
        """Share the gateway ledger while tightening review transport caps."""

        _validate_signal_prompt_variant_settings(self.settings)
        if not self.settings.character_signal_scope_review_v1:
            raise ValueError("character scope review is disabled")
        if any(
            type(value) not in {int, float}
            or not isfinite(float(value))
            or value <= 0
            for value in (timeout_seconds, remaining_deadline_seconds)
        ):
            raise ValueError("character scope review deadline is invalid")
        if (
            type(completion_reserve) is not int or completion_reserve < 64
            or type(max_response_bytes) is not int or max_response_bytes < 1
            or type(max_attempts) is not int or max_attempts < 1
        ):
            raise ValueError("character scope review limits are invalid")

        review_provider = self.signal_provider
        reserve = min(
            completion_reserve,
            self.settings.character_signal_scope_review_completion_tokens,
        )
        if isinstance(review_provider, OpenAICompatibleProvider):
            provider_settings = review_provider.settings
            deadline = min(
                value for value in (
                    float(timeout_seconds),
                    float(remaining_deadline_seconds),
                    float(self.settings.character_signal_total_deadline_seconds),
                    provider_settings.provider_total_deadline_seconds,
                    self.settings.provider_total_deadline_seconds,
                ) if value is not None
            )
            timeout = min(
                float(timeout_seconds),
                deadline,
                float(self.settings.character_signal_scope_review_timeout_seconds),
                float(self.settings.provider_timeout_seconds),
                float(provider_settings.provider_timeout_seconds),
            )
            completion = min(
                value for value in (
                    reserve,
                    provider_settings.provider_max_completion_tokens,
                    self.settings.provider_max_completion_tokens,
                ) if value is not None
            )
            response_bytes = min(
                value for value in (
                    max_response_bytes,
                    self.settings.character_signal_scope_review_max_response_bytes,
                    provider_settings.provider_max_response_bytes,
                    self.settings.provider_max_response_bytes,
                ) if value is not None
            )
            attempts = min(
                max_attempts,
                self.settings.character_signal_scope_review_max_attempts,
                self.settings.provider_max_attempts,
                provider_settings.provider_max_attempts,
                review_provider.retry_policy.max_attempts,
            )
            bounded_settings = provider_settings.model_copy(update={
                "enable_model_extraction": False,
                "enable_review_agent": False,
                "enable_issue_evidence_review": False,
                "enable_evidence_investigator": False,
                "enable_character_consistency": True,
                "provider_timeout_seconds": timeout,
                "provider_total_deadline_seconds": deadline,
                "provider_max_attempts": attempts,
                "provider_max_completion_tokens": completion,
                "provider_max_response_bytes": response_bytes,
            })
            review_provider = OpenAICompatibleProvider(
                bounded_settings,
                transport=review_provider.transport,
                retry_policy=replace(review_provider.retry_policy, max_attempts=attempts),
                sleep=review_provider.sleep,
                monotonic=review_provider.monotonic,
                wall_time=review_provider.wall_time,
                random_value=review_provider.random_value,
            )
        return _CharacterConsistencyAccountingProvider(
            self.settings,
            self.usage,
            signal_provider=self.signal_provider,
            drift_provider=self.drift_provider,
            scope_review_provider=review_provider,
            scope_review_completion_reserve=reserve,
        )

    def complete(self, system: str, user: str):
        # Settings.model_copy(update=...) can bypass model validators; fail
        # before purpose selection, estimation, provider call or charge.
        _validate_signal_prompt_variant_settings(self.settings)
        active_primary_system = (
            CHARACTER_SIGNAL_SYSTEM_PROMPT
            + (
                CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2
                if self.settings.character_signal_full_line_prompt_v2 else ""
            )
            + (
                CHARACTER_SIGNAL_CORE_SCOPE_PROMPT_V3
                if self.settings.character_signal_core_scope_prompt_v3 else ""
            )
        )
        allowed_primary_systems = {active_primary_system}
        if self.settings.character_signal_semantic_scope_v5:
            allowed_primary_systems.add(
                CHARACTER_SIGNAL_SYSTEM_PROMPT_V5
                + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2
                + (
                    CHARACTER_SIGNAL_SEMANTIC_SCOPE_REVIEW_PROMPT_V1
                    + CHARACTER_SIGNAL_SCOPE_REVIEW_PROMPT_V1
                    if self.settings.character_signal_scope_review_v1
                    else CHARACTER_SIGNAL_SEMANTIC_SCOPE_PROMPT_V5
                )
            )
        elif self.settings.character_signal_support_id_v4:
            allowed_primary_systems.add(
                active_primary_system + CHARACTER_SIGNAL_SUPPORT_ID_PROMPT_V4
            )
        if system in allowed_primary_systems | {TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT}:
            provider = self.signal_provider
            completion_reserve = self.settings.character_signal_max_completion_tokens
        elif (
            system == SCOPE_REVIEW_SYSTEM_PROMPT
            and self.settings.character_signal_scope_review_v1
        ):
            if self.scope_review_provider is None:
                bounded = self.fork_for_character_scope_review(
                    timeout_seconds=self.settings.character_signal_scope_review_timeout_seconds,
                    remaining_deadline_seconds=(
                        self.settings.character_signal_total_deadline_seconds
                    ),
                    completion_reserve=(
                        self.settings.character_signal_scope_review_completion_tokens
                    ),
                    max_response_bytes=(
                        self.settings.character_signal_scope_review_max_response_bytes
                    ),
                    max_attempts=self.settings.character_signal_scope_review_max_attempts,
                )
                provider = bounded.scope_review_provider
            else:
                provider = self.scope_review_provider
            completion_reserve = (
                self.scope_review_completion_reserve
                or self.settings.character_signal_scope_review_completion_tokens
            )
        elif system == CHARACTER_REVIEW_SYSTEM_PROMPT:
            provider = self.drift_provider
            completion_reserve = self.settings.character_drift_max_completion_tokens
        else:
            # This boundary is deliberately closed over the audited signal
            # (primary and targeted) plus drift-review prompt contracts. A new
            # model purpose must opt into its own limits and accounting instead
            # of silently borrowing either one.
            raise RuntimeError("unsupported character consistency provider purpose")
        estimate = estimate_issue_evidence_review_tokens(
            system,
            user,
            completion_reserve=completion_reserve,
        )
        try:
            result = provider.complete(system, user)
        except Exception:
            self.usage.record(
                prompt_tokens=0,
                completion_tokens=0,
                charged_tokens=estimate,
                successful=False,
            )
            raise
        prompt = getattr(result, "prompt_tokens", 0)
        completion = getattr(result, "completion_tokens", 0)
        self.usage.record(
            prompt_tokens=prompt,
            completion_tokens=completion,
            charged_tokens=max(
                estimate,
                (prompt if type(prompt) is int and prompt >= 0 else 0)
                + (
                    completion
                    if type(completion) is int and completion >= 0
                    else 0
                ),
            ),
            successful=True,
        )
        return result


class _EagerScalarRows:
    """Minimal detached ``ScalarResult`` surface consumed by the stage."""

    def __init__(self, rows: list[object]) -> None:
        self._rows = rows

    def all(self) -> list[object]:
        return list(self._rows)


class CharacterConsistencyDatabase:
    """Short-transaction database boundary for the optional character stage.

    Model calls must not inherit a SQLite writer (or reader) transaction from
    candidate persistence. Reads are eagerly materialized and detached in a
    short-lived session. Each ``begin_nested`` requested by the stage is mapped
    to one independent top-level candidate transaction and fenced against the
    current worker token before commit. PostgreSQL therefore keeps its normal
    row-level correctness while SQLite never holds the candidate write lock
    across a later provider call.
    """

    def __init__(
        self,
        session_factory,
        *,
        run_id: str,
        worker_token: str,
    ) -> None:
        self.session_factory = session_factory
        self.run_id = run_id
        self.worker_token = worker_token
        self._active = None
        self._ownership_lost = False

    @staticmethod
    def _detach(session, value: object) -> object:
        try:
            session.expunge(value)
        except Exception:
            pass
        return value

    def scalars(self, statement):
        if self._active is not None:
            return self._active.scalars(statement)
        with self.session_factory() as session:
            rows = list(session.scalars(statement).all())
            detached = [self._detach(session, row) for row in rows]
            session.rollback()
        return _EagerScalarRows(detached)

    def scalar(self, statement):
        if self._active is not None:
            return self._active.scalar(statement)
        with self.session_factory() as session:
            value = session.scalar(statement)
            value = self._detach(session, value)
            session.rollback()
            return value

    def get(self, entity, identifier):
        if self._active is not None:
            return self._active.get(entity, identifier)
        with self.session_factory() as session:
            value = session.get(entity, identifier)
            value = self._detach(session, value)
            session.rollback()
            return value

    def add(self, value) -> None:
        if self._active is None:
            raise RuntimeError("character candidate write is outside its transaction")
        self._active.add(value)

    def flush(self) -> None:
        if self._active is None:
            raise RuntimeError("character candidate flush is outside its transaction")
        self._active.flush()

    def execute(self, statement):
        if self._active is None:
            raise RuntimeError("character candidate execution is outside its transaction")
        return self._active.execute(statement)

    @contextmanager
    def begin_nested(self):
        if self._active is not None:
            with self._active.begin_nested():
                yield
            return
        with self.session_factory() as session:
            self._active = session
            try:
                with session.begin():
                    yield
                    owned = session.execute(
                        update(AnalysisRunExecutionRow)
                        .where(
                            AnalysisRunExecutionRow.run_id == self.run_id,
                            AnalysisRunExecutionRow.worker_token
                            == self.worker_token,
                        )
                        .values(worker_token=self.worker_token)
                    ).rowcount
                    if owned != 1:
                        self._ownership_lost = True
                        raise WorkerLeaseLost(
                            "analysis worker lease was lost during candidate persistence"
                        )
            finally:
                self._active = None

    def raise_if_ownership_lost(self) -> None:
        if self._ownership_lost:
            raise WorkerLeaseLost(
                "analysis worker lease was lost during candidate persistence"
            )


def document_content_sha256(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def _snapshot_narrative_contexts(
    db,
    snapshots: list[AnalysisRunInputRow],
    documents: list[DocumentRow],
    document_contexts: dict[str, DocumentContextRow],
) -> list[dict[str, Any]]:
    latest = latest_context_revisions(db, [document.id for document in documents])
    payloads: list[dict[str, Any]] = []
    for snapshot, document in zip(snapshots, documents, strict=True):
        legacy = document_contexts.get(document.id)
        role = legacy.document_role if legacy else DEFAULT_DOCUMENT_ROLE
        story_scope = legacy.story_scope if legacy else DEFAULT_STORY_SCOPE
        revision = latest.get(document.id)
        payload = context_snapshot_payload(
            revision,
            document_role=role,
            story_scope=story_scope,
        )
        payloads.append(payload)
        db.add(
            AnalysisRunInputNarrativeContextRow(
                input_id=snapshot.id,
                context_revision_id=revision.id if revision else None,
                schema_version=1,
                payload=payload,
                payload_sha256=payload_sha256(payload),
            )
        )
    return payloads


def _draft_target_release_ordinals(
    narrative_contexts: list[dict[str, Any]],
) -> tuple[int, ...]:
    """Return only explicit release ordinals for documents the stage treats as drafts."""

    result: set[int] = set()
    for context in narrative_contexts:
        if (
            context.get("resolution_state") != "confirmed"
            or context.get("publication_status") != "draft"
            or context.get("legacy_document_role") != "chapter"
        ):
            continue
        scope = context.get("scope")
        release = scope.get("release") if isinstance(scope, dict) else None
        ordinal = release.get("ordinal") if isinstance(release, dict) else None
        if type(ordinal) is int and 0 <= ordinal <= 2_147_483_647:
            result.add(ordinal)
    return tuple(sorted(result))


def _approved_axis_snapshot_fields(db, candidate, review) -> dict[str, Any]:
    """Freeze an author-approved axis without changing any legacy payload."""

    axis_id = candidate.approved_axis_id
    axis_version = candidate.approved_axis_version
    if (
        axis_id != review.approved_axis_id
        or axis_version != review.approved_axis_version
    ):
        raise ValueError("approved character axis does not match confirmation review")
    if axis_id is None and axis_version is None:
        return {}
    if not axis_id or type(axis_version) is not int or axis_version < 1:
        raise ValueError("incomplete approved character axis binding")
    axis = db.get(CharacterTraitAxisRow, axis_id)
    if (
        axis is None
        or axis.project_id != candidate.project_id
        or axis.trait_type != candidate.trait_type
        or axis.trait_type != "core_personality"
        or axis.version != axis_version
        or not isinstance(axis.definition, str)
        or not 1 <= len(axis.definition) <= 200
        or axis.definition != " ".join(axis.definition.split())
        or not isinstance(axis.display_name, str)
        or not 1 <= len(axis.display_name) <= 80
        or sha256(axis.definition.encode("utf-8")).hexdigest()
        != axis.definition_sha256
    ):
        raise ValueError("approved character axis binding is invalid")
    return {
        "approved_axis_id": axis.id,
        "approved_axis_version": axis.version,
        "approved_axis_display_name": axis.display_name,
        "approved_axis_definition": axis.definition,
        "approved_axis_definition_sha256": axis.definition_sha256,
    }


def _confirmed_trait_snapshot_payload(db, candidate, review) -> dict[str, Any]:
    payload = candidate_snapshot_payload(candidate, review)
    payload.update(_approved_axis_snapshot_fields(db, candidate, review))
    return payload


def _historical_trait_snapshot_payload(candidate, review, db=None) -> dict[str, Any]:
    """Snapshot a formerly confirmed, explicitly version-bounded profile row."""

    lower = candidate.valid_from_release_ordinal
    upper = candidate.valid_until_release_ordinal
    if (
        candidate.review_state != "superseded"
        or review.decision != "confirm"
        or upper is None
        or (lower is not None and upper < lower)
    ):
        raise ValueError("historical character trait is not safely bounded")
    scope = canonical_scope_payload(candidate.scope_payload)
    if payload_sha256(scope) != candidate.scope_sha256:
        raise ValueError("candidate scope hash mismatch")
    if payload_sha256(candidate.evidence) != candidate.evidence_sha256:
        raise ValueError("candidate evidence hash mismatch")
    payload = {
        "schema_version": CHARACTER_TRAIT_SCHEMA_VERSION,
        "candidate_id": candidate.id,
        "confirmation_review_id": review.id,
        "character_key": candidate.character_key,
        "character_display_name": candidate.character_display_name,
        "trait_type": candidate.trait_type,
        "trait_key": candidate.trait_key,
        **candidate_snapshot_comparison_key(candidate),
        "value": candidate.value,
        "polarity": candidate.polarity,
        "stability": candidate.stability,
        "contexts": candidate.contexts,
        "origin": candidate.origin,
        "authority_tier": candidate.authority_tier,
        "scope": scope,
        "scope_sha256": candidate.scope_sha256,
        "valid_from_release_ordinal": lower,
        "valid_until_release_ordinal": upper,
        "evidence": candidate.evidence,
        "evidence_sha256": candidate.evidence_sha256,
        "candidate_fingerprint": candidate.candidate_fingerprint,
        "generator_version": candidate.generator_version,
        "candidate_lock_version": candidate.lock_version,
    }
    if db is not None:
        payload.update(_approved_axis_snapshot_fields(db, candidate, review))
    elif candidate.approved_axis_id is not None:
        raise ValueError("approved character axis requires snapshot database")
    return payload


def _capture_confirmed_traits(
    db,
    run: AnalysisRunRow,
    narrative_contexts: list[dict[str, Any]],
) -> None:
    """Freeze active traits and historical traits applicable to a target release.

    Superseded rows are human-confirmed history, but an unbounded superseded row
    must never become a current baseline. They are included only when they have
    an explicit upper release bound and overlap an explicit, confirmed draft
    release in this run. The semantic stage still enforces scope compatibility.
    """

    target_ordinals = _draft_target_release_ordinals(narrative_contexts)
    historical_filters = [
        and_(
            CharacterTraitCandidateRow.review_state == "superseded",
            CharacterTraitCandidateRow.valid_until_release_ordinal.is_not(None),
            or_(
                CharacterTraitCandidateRow.valid_from_release_ordinal.is_(None),
                CharacterTraitCandidateRow.valid_from_release_ordinal <= ordinal,
            ),
            CharacterTraitCandidateRow.valid_until_release_ordinal >= ordinal,
            or_(
                CharacterTraitCandidateRow.valid_from_release_ordinal.is_(None),
                CharacterTraitCandidateRow.valid_from_release_ordinal
                <= CharacterTraitCandidateRow.valid_until_release_ordinal,
            ),
        )
        for ordinal in target_ordinals
    ]
    eligible_state = or_(
        CharacterTraitCandidateRow.review_state == "confirmed",
        *historical_filters,
    )
    candidates = list(
        db.scalars(
            select(CharacterTraitCandidateRow)
            .where(
                CharacterTraitCandidateRow.project_id == run.project_id,
                eligible_state,
            )
            .order_by(
                CharacterTraitCandidateRow.character_key,
                CharacterTraitCandidateRow.trait_type,
                CharacterTraitCandidateRow.trait_key,
                CharacterTraitCandidateRow.id,
            )
            .limit(MAX_CONFIRMED_TRAITS_PER_RUN + 1)
        ).all()
    )
    if len(candidates) > MAX_CONFIRMED_TRAITS_PER_RUN:
        raise CharacterTraitSnapshotLimitExceeded(
            "confirmed character profile exceeds the per-run snapshot limit"
        )
    reviews = latest_confirm_reviews(db, [row.id for row in candidates])
    if len(reviews) != len(candidates):
        raise ValueError("confirmed character profile is missing review provenance")
    for ordinal, candidate in enumerate(candidates):
        review = reviews[candidate.id]
        if review.project_id != run.project_id:
            raise ValueError("confirmed character profile crosses projects")
        payload = (
            _confirmed_trait_snapshot_payload(db, candidate, review)
            if candidate.review_state == "confirmed"
            else _historical_trait_snapshot_payload(candidate, review, db=db)
        )
        db.add(
            AnalysisRunCharacterTraitInputRow(
                run_id=run.id,
                project_id=run.project_id,
                candidate_id=candidate.id,
                confirmation_review_id=review.id,
                candidate_lock_version=candidate.lock_version,
                ordinal=ordinal,
                payload=payload,
                payload_sha256=payload_sha256(payload),
            )
        )


def capture_run_inputs(
    db,
    run: AnalysisRunRow,
    documents: list[DocumentRow],
    *,
    batch_roles: dict[str, str] | None = None,
) -> list[AnalysisRunInputRow]:
    """Freeze current document bodies in the caller's creation transaction."""
    resolved_batch_roles = batch_roles or {}
    invalid_roles = set(resolved_batch_roles.values()) - {"target", "background"}
    unknown_documents = set(resolved_batch_roles) - {
        document.id for document in documents
    }
    if invalid_roles or unknown_documents:
        raise ValueError("invalid analysis input batch role mapping")
    document_contexts = {
        row.document_id: row
        for row in db.scalars(
            select(DocumentContextRow).where(
                DocumentContextRow.document_id.in_([document.id for document in documents])
            )
        ).all()
    } if documents else {}
    snapshots: list[AnalysisRunInputRow] = []
    for ordinal, document in enumerate(documents):
        snapshot = AnalysisRunInputRow(
            run_id=run.id,
            document_id=document.id,
            document_name=document.name,
            document_version=document.version,
            content=document.content,
            content_sha256=document_content_sha256(document.content),
            ordinal=ordinal,
        )
        db.add(snapshot)
        snapshots.append(snapshot)
    db.flush()
    for snapshot, document in zip(snapshots, documents, strict=True):
        context = document_contexts.get(document.id)
        db.add(
            AnalysisRunInputContextRow(
                input_id=snapshot.id,
                document_role=(
                    context.document_role if context else DEFAULT_DOCUMENT_ROLE
                ),
                story_scope=context.story_scope if context else DEFAULT_STORY_SCOPE,
                batch_role=resolved_batch_roles.get(document.id, "target"),
            )
        )
    narrative_contexts = _snapshot_narrative_contexts(
        db, snapshots, documents, document_contexts
    )
    _capture_confirmed_traits(db, run, narrative_contexts)
    db.add(AnalysisRunExecutionRow(run_id=run.id))
    run.input_chars = sum(len(row.content) for row in snapshots)
    return snapshots


def copy_run_inputs(
    db, source_run_id: str, target_run: AnalysisRunRow
) -> list[AnalysisRunInputRow]:
    """Create a retry run from the original immutable snapshot."""
    source = db.scalars(
        select(AnalysisRunInputRow)
        .where(AnalysisRunInputRow.run_id == source_run_id)
        .order_by(AnalysisRunInputRow.ordinal)
    ).all()
    if not source:
        raise ValueError(MISSING_SNAPSHOT_ERROR)
    source_context = {
        row.input_id: row
        for row in db.scalars(
            select(AnalysisRunInputContextRow).where(
                AnalysisRunInputContextRow.input_id.in_([item.id for item in source])
            )
        ).all()
    }
    source_narrative_context = {
        row.input_id: row
        for row in db.scalars(
            select(AnalysisRunInputNarrativeContextRow).where(
                AnalysisRunInputNarrativeContextRow.input_id.in_(
                    [item.id for item in source]
                )
            )
        ).all()
    }
    copied: list[AnalysisRunInputRow] = []
    for row in source:
        snapshot = AnalysisRunInputRow(
            run_id=target_run.id,
            document_id=row.document_id,
            document_name=row.document_name,
            document_version=row.document_version,
            content=row.content,
            content_sha256=row.content_sha256,
            ordinal=row.ordinal,
        )
        db.add(snapshot)
        copied.append(snapshot)
    db.flush()
    for source_row, copied_row in zip(source, copied, strict=True):
        context = source_context.get(source_row.id)
        db.add(
            AnalysisRunInputContextRow(
                input_id=copied_row.id,
                document_role=(
                    context.document_role if context else DEFAULT_DOCUMENT_ROLE
                ),
                story_scope=context.story_scope if context else DEFAULT_STORY_SCOPE,
                batch_role=(
                    context.batch_role if context else "target"
                ),
            )
        )
        narrative = source_narrative_context.get(source_row.id)
        if narrative is not None:
            db.add(
                AnalysisRunInputNarrativeContextRow(
                    input_id=copied_row.id,
                    context_revision_id=narrative.context_revision_id,
                    schema_version=narrative.schema_version,
                    payload=narrative.payload,
                    payload_sha256=narrative.payload_sha256,
                )
            )
        else:
            payload = context_snapshot_payload(
                None,
                document_role=(
                    context.document_role if context else DEFAULT_DOCUMENT_ROLE
                ),
                story_scope=context.story_scope if context else DEFAULT_STORY_SCOPE,
            )
            db.add(
                AnalysisRunInputNarrativeContextRow(
                    input_id=copied_row.id,
                    context_revision_id=None,
                    schema_version=1,
                    payload=payload,
                    payload_sha256=payload_sha256(payload),
                )
            )
    source_traits = list(
        db.scalars(
            select(AnalysisRunCharacterTraitInputRow)
            .where(AnalysisRunCharacterTraitInputRow.run_id == source_run_id)
            .order_by(AnalysisRunCharacterTraitInputRow.ordinal)
        ).all()
    )
    for trait in source_traits:
        db.add(
            AnalysisRunCharacterTraitInputRow(
                run_id=target_run.id,
                project_id=trait.project_id,
                candidate_id=trait.candidate_id,
                confirmation_review_id=trait.confirmation_review_id,
                candidate_lock_version=trait.candidate_lock_version,
                ordinal=trait.ordinal,
                payload=trait.payload,
                payload_sha256=trait.payload_sha256,
            )
        )
    db.add(
        AnalysisRunExecutionRow(
            run_id=target_run.id,
            retried_from_run_id=source_run_id,
        )
    )
    target_run.input_chars = sum(len(row.content) for row in copied)
    return copied


def run_input_metadata(db, run_id: str) -> list[dict]:
    rows = db.scalars(
        select(AnalysisRunInputRow)
        .where(AnalysisRunInputRow.run_id == run_id)
        .order_by(AnalysisRunInputRow.ordinal)
    ).all()
    contexts = {
        context.input_id: context
        for context in db.scalars(
            select(AnalysisRunInputContextRow).where(
                AnalysisRunInputContextRow.input_id.in_([row.id for row in rows])
            )
        ).all()
    } if rows else {}
    narrative_contexts = {
        context.input_id: context
        for context in db.scalars(
            select(AnalysisRunInputNarrativeContextRow).where(
                AnalysisRunInputNarrativeContextRow.input_id.in_(
                    [row.id for row in rows]
                )
            )
        ).all()
    } if rows else {}
    return [
        {
            "document_id": row.document_id,
            "document_name": row.document_name,
            "document_version": row.document_version,
            "document_role": (
                contexts[row.id].document_role
                if row.id in contexts
                else DEFAULT_DOCUMENT_ROLE
            ),
            "story_scope": (
                contexts[row.id].story_scope
                if row.id in contexts
                else DEFAULT_STORY_SCOPE
            ),
            "batch_role": (
                contexts[row.id].batch_role
                if row.id in contexts
                else "target"
            ),
            "context_explicit": row.id in contexts,
            "narrative_context": (
                narrative_contexts[row.id].payload
                if row.id in narrative_contexts
                else context_snapshot_payload(
                    None,
                    document_role=(
                        contexts[row.id].document_role
                        if row.id in contexts
                        else DEFAULT_DOCUMENT_ROLE
                    ),
                    story_scope=(
                        contexts[row.id].story_scope
                        if row.id in contexts
                        else DEFAULT_STORY_SCOPE
                    ),
                )
            ),
            "narrative_context_sha256": (
                narrative_context_semantic_sha256(
                    narrative_contexts[row.id].payload
                )
                if row.id in narrative_contexts
                else narrative_context_semantic_sha256(
                    context_snapshot_payload(
                        None,
                        document_role=(
                            contexts[row.id].document_role
                            if row.id in contexts
                            else DEFAULT_DOCUMENT_ROLE
                        ),
                        story_scope=(
                            contexts[row.id].story_scope
                            if row.id in contexts
                            else DEFAULT_STORY_SCOPE
                        ),
                    )
                )
            ),
            "narrative_context_payload_sha256": (
                narrative_contexts[row.id].payload_sha256
                if row.id in narrative_contexts
                else payload_sha256(
                    context_snapshot_payload(
                        None,
                        document_role=(
                            contexts[row.id].document_role
                            if row.id in contexts
                            else DEFAULT_DOCUMENT_ROLE
                        ),
                        story_scope=(
                            contexts[row.id].story_scope
                            if row.id in contexts
                            else DEFAULT_STORY_SCOPE
                        ),
                    )
                )
            ),
            "content_sha256": row.content_sha256,
            "char_count": len(row.content),
            "ordinal": row.ordinal,
        }
        for row in rows
    ]


def run_trait_snapshot_metadata(db, run_id: str) -> dict:
    rows = list(
        db.scalars(
            select(AnalysisRunCharacterTraitInputRow)
            .where(AnalysisRunCharacterTraitInputRow.run_id == run_id)
            .order_by(AnalysisRunCharacterTraitInputRow.ordinal)
        ).all()
    )
    fingerprint = payload_sha256(
        [
            {
                "candidate_id": row.candidate_id,
                "candidate_lock_version": row.candidate_lock_version,
                "payload_sha256": row.payload_sha256,
                "ordinal": row.ordinal,
            }
            for row in rows
        ]
    )
    return {
        "confirmed_trait_count": len(rows),
        "character_profile_snapshot_sha256": fingerprint,
    }


def run_narrative_context_fingerprint(db, run_id: str) -> str:
    rows = run_input_metadata(db, run_id)
    return payload_sha256(
        [
            {
                "document_id": row["document_id"],
                "narrative_context_sha256": row["narrative_context_sha256"],
                "ordinal": row["ordinal"],
            }
            for row in rows
        ]
    )


def frozen_run_source_signature(db, run_id: str) -> str:
    documents = run_input_metadata(db, run_id)
    traits = run_trait_snapshot_metadata(db, run_id)
    return payload_sha256(
        {
            "documents": [
                {
                    "name": row["document_name"].casefold(),
                    "version": row["document_version"],
                    "content_sha256": row["content_sha256"],
                    "document_role": row["document_role"],
                    "story_scope": row["story_scope"],
                    "batch_role": row["batch_role"],
                    "narrative_context_sha256": row[
                        "narrative_context_sha256"
                    ],
                }
                for row in documents
            ],
            "character_profile_snapshot_sha256": traits[
                "character_profile_snapshot_sha256"
            ],
        }
    )


def current_project_source_signature(
    db,
    project_id: str,
    documents: list[DocumentRow],
    *,
    batch_roles: dict[str, str] | None = None,
) -> str:
    resolved_batch_roles = batch_roles or {}
    contexts = {
        row.document_id: row
        for row in db.scalars(
            select(DocumentContextRow).where(
                DocumentContextRow.document_id.in_(
                    [document.id for document in documents]
                )
            )
        ).all()
    } if documents else {}
    narrative = latest_context_revisions(
        db, [document.id for document in documents]
    )
    document_rows: list[dict] = []
    for document in documents:
        legacy = contexts.get(document.id)
        role = legacy.document_role if legacy else DEFAULT_DOCUMENT_ROLE
        story_scope = legacy.story_scope if legacy else DEFAULT_STORY_SCOPE
        context_payload = context_snapshot_payload(
            narrative.get(document.id),
            document_role=role,
            story_scope=story_scope,
        )
        document_rows.append(
            {
                "name": document.name.casefold(),
                "version": document.version,
                "content_sha256": document_content_sha256(document.content),
                "document_role": role,
                "story_scope": story_scope,
                "batch_role": resolved_batch_roles.get(document.id, "target"),
                "narrative_context_sha256": narrative_context_semantic_sha256(
                    context_payload
                ),
            }
        )
    candidates = list(
        db.scalars(
            select(CharacterTraitCandidateRow)
            .where(
                CharacterTraitCandidateRow.project_id == project_id,
                CharacterTraitCandidateRow.review_state == "confirmed",
            )
            .order_by(
                CharacterTraitCandidateRow.character_key,
                CharacterTraitCandidateRow.trait_type,
                CharacterTraitCandidateRow.trait_key,
                CharacterTraitCandidateRow.id,
            )
            .limit(MAX_CONFIRMED_TRAITS_PER_RUN + 1)
        ).all()
    )
    if len(candidates) > MAX_CONFIRMED_TRAITS_PER_RUN:
        raise CharacterTraitSnapshotLimitExceeded(
            "confirmed character profile exceeds the per-run snapshot limit"
        )
    reviews = latest_confirm_reviews(db, [row.id for row in candidates])
    if len(reviews) != len(candidates):
        raise ValueError("confirmed character profile is missing review provenance")
    trait_fingerprint = payload_sha256(
        [
            {
                "candidate_id": candidate.id,
                "candidate_lock_version": candidate.lock_version,
                "payload_sha256": payload_sha256(
                    _confirmed_trait_snapshot_payload(db, candidate, reviews[candidate.id])
                ),
                "ordinal": ordinal,
            }
            for ordinal, candidate in enumerate(candidates)
        ]
    )
    return payload_sha256(
        {
            "documents": document_rows,
            "character_profile_snapshot_sha256": trait_fingerprint,
        }
    )


def emit(db, run_id: str, stage: str, progress: int, message: str) -> bool:
    """Append a monotonic non-terminal event."""
    if stage in TERMINAL_EVENT_STAGES:
        raise ValueError("terminal events must be emitted with the status transition")
    terminal_exists = db.scalar(
        select(func.count())
        .select_from(RunEventRow)
        .where(
            RunEventRow.run_id == run_id,
            RunEventRow.stage.in_(TERMINAL_EVENT_STAGES),
        )
    )
    if terminal_exists:
        return False
    last_progress = db.scalar(
        select(func.max(RunEventRow.progress)).where(RunEventRow.run_id == run_id)
    )
    if last_progress is not None and progress < last_progress:
        return False
    db.add(
        RunEventRow(
            run_id=run_id, stage=stage, progress=progress, message=message
        )
    )
    db.commit()
    return True


def _emit_owned(
    db,
    run_id: str,
    worker_token: str,
    stage: str,
    progress: int,
    message: str,
) -> bool:
    """Fence an optional-stage event in the same transaction as ownership.

    The no-op conditional update obtains the execution-row write lock and is
    re-evaluated against the current owner before ``emit`` commits the event.
    A stale worker can therefore neither append progress nor overwrite a newer
    worker's lower progress after losing its lease.
    """

    owned = db.execute(
        update(AnalysisRunExecutionRow)
        .where(
            AnalysisRunExecutionRow.run_id == run_id,
            AnalysisRunExecutionRow.worker_token == worker_token,
        )
        .values(worker_token=worker_token)
    ).rowcount
    if owned != 1:
        db.rollback()
        raise WorkerLeaseLost("analysis worker lease was lost")
    emitted = emit(db, run_id, stage, progress, message)
    if not emitted:
        db.rollback()
    return emitted


def analysis_mode(result) -> tuple[str, bool]:
    execution = getattr(result, "diagnostics", {}).get("model")
    required_counters = (
        "succeeded_chunks",
        "failed_chunks",
        "skipped_chunks",
        "invalid_records",
        "empty_response_chunks",
    )
    if not isinstance(execution, dict) or any(
        type(execution.get(key)) is not int or execution[key] < 0
        for key in required_counters
    ):
        return "执行状态未知（缺少结构化记录）", True
    if not execution["enabled"]:
        return "确定性基线", False
    succeeded = execution["succeeded_chunks"]
    empty_responses = execution["empty_response_chunks"]
    if empty_responses > succeeded:
        return "执行状态未知（结构化记录不一致）", True
    raw_invalid = execution["invalid_records"]
    disposition_values = (
        execution.get("unresolved_invalid_records"),
        execution.get("recovered_invalid_records"),
    )
    has_final_disposition = any(value is not None for value in disposition_values)
    if has_final_disposition:
        unresolved, recovered = disposition_values
        repair_failed = execution.get("repair_failed")
        repair_post_invalid = execution.get("repair_post_invalid")
        if (
            any(type(value) is not int or value < 0 for value in disposition_values)
            or type(repair_failed) is not bool
            or type(repair_post_invalid) is not int
            or repair_post_invalid < 0
            or raw_invalid != unresolved + recovered
        ):
            return "执行状态未知（无效记录诊断不一致）", True
        final_invalid = unresolved
        repair_incomplete = repair_failed or repair_post_invalid > 0
    else:
        # Stored legacy runs expose only the observed invalid count.
        final_invalid = raw_invalid
        repair_incomplete = False
    partial_fallback = bool(
        execution["failed_chunks"]
        or execution["skipped_chunks"]
        or final_invalid
        or repair_incomplete
        or empty_responses
        or not execution["configured"]
    )
    if succeeded:
        if empty_responses:
            return "模型返回空结果，无法证明完整覆盖", True
        return (
            "模型增强（部分结果已降级）"
            if partial_fallback
            else "完整模型增强",
            partial_fallback,
        )
    return "确定性基线（模型未参与或已降级）", True


def _mark_legacy_run_unusable(run_id: str) -> None:
    """Fail a pre-snapshot nonterminal run without consulting live documents."""
    with SessionLocal() as db:
        now = utc_now_naive()
        changed = db.execute(
            update(AnalysisRunRow)
            .where(
                AnalysisRunRow.id == run_id,
                AnalysisRunRow.status.not_in(TERMINAL_STATUSES),
            )
            .values(status="failed", error=MISSING_SNAPSHOT_ERROR, completed_at=now)
        ).rowcount
        if changed:
            db.add(
                RunEventRow(
                    run_id=run_id,
                    stage="failed",
                    progress=100,
                    message="旧任务缺少不可变输入快照，已安全停止",
                )
            )
            db.commit()


def claim_analysis_run(run_id: str, worker_token: str) -> bool:
    """Atomically acquire the per-run execution record if its lease is free."""
    with SessionLocal() as db:
        run = db.get(AnalysisRunRow, run_id)
        if not run or run.status in {"completed", "cancelled"}:
            return False
        if run.status == "failed" and db.scalar(
            select(func.count())
            .select_from(RunEventRow)
            .where(RunEventRow.run_id == run_id, RunEventRow.stage == "failed")
        ):
            return False
        snapshot_count = db.scalar(
            select(func.count())
            .select_from(AnalysisRunInputRow)
            .where(AnalysisRunInputRow.run_id == run_id)
        )
        if not snapshot_count:
            db.rollback()
            _mark_legacy_run_unusable(run_id)
            return False

        now = utc_now_naive()
        lease_seconds = _lease_seconds()
        claimed = db.execute(
            update(AnalysisRunExecutionRow)
            .where(
                AnalysisRunExecutionRow.run_id == run_id,
                or_(
                    AnalysisRunExecutionRow.worker_token.is_(None),
                    AnalysisRunExecutionRow.lease_expires_at < now,
                ),
            )
            .values(
                worker_token=worker_token,
                attempt_no=AnalysisRunExecutionRow.attempt_no + 1,
                claimed_at=now,
                last_heartbeat_at=now,
                lease_expires_at=now + timedelta(seconds=lease_seconds),
            )
        ).rowcount
        if claimed != 1:
            db.rollback()
            return False
        moved_to_running = db.execute(
            update(AnalysisRunRow)
            .where(
                AnalysisRunRow.id == run_id,
                AnalysisRunRow.status.in_(("queued", "running", "failed")),
            )
            .values(
                status="running",
                started_at=func.coalesce(AnalysisRunRow.started_at, now),
                completed_at=None,
                error=None,
            )
        ).rowcount
        if moved_to_running != 1:
            db.rollback()
            return False
        db.commit()
        return True


def active_lease_retry_after(run_id: str) -> int | None:
    """Return a safe redelivery delay while another worker owns this run."""
    with SessionLocal() as db:
        execution = db.get(AnalysisRunExecutionRow, run_id)
        run = db.get(AnalysisRunRow, run_id)
        if (
            not execution
            or not execution.worker_token
            or not execution.lease_expires_at
            or not run
            or run.status != "running"
        ):
            return None
        remaining = (execution.lease_expires_at - utc_now_naive()).total_seconds()
        return max(1, ceil(remaining) + 1) if remaining > 0 else None


def _checkpoint(
    run_id: str,
    worker_token: str,
    heartbeat: ExecutionLeaseHeartbeat | None = None,
) -> None:
    """Refresh ownership and read cancellation from a fresh DB session."""
    if heartbeat is not None:
        heartbeat.raise_if_failed()
    now = utc_now_naive()
    if not _renew_execution_lease(run_id, worker_token, now=now):
        raise WorkerLeaseLost("analysis worker lease was lost")
    with SessionLocal() as db:
        cancel_requested = db.scalar(
            select(AnalysisRunRow.cancel_requested).where(
                AnalysisRunRow.id == run_id
            )
        )
        db.commit()
        if heartbeat is not None:
            heartbeat.raise_if_failed()
        if cancel_requested:
            raise AnalysisCancelled("analysis cancellation requested")


def _owned_run_clause(run_id: str, worker_token: str):
    return exists(
        select(AnalysisRunExecutionRow.run_id).where(
            AnalysisRunExecutionRow.run_id == run_id,
            AnalysisRunExecutionRow.worker_token == worker_token,
        )
    )


def _optional_nonnegative_int(
    value, *, maximum: int = _SIGNED_64_MAX
) -> int | None:
    return value if type(value) is int and 0 <= value <= maximum else None


def _interrupted_usage_limits() -> dict[str, int]:
    """Derive finite persistence limits from the bounded Agent settings."""
    settings = get_settings()
    decision_rounds = min(
        _SIGNED_64_MAX,
        max(1, int(settings.review_agent_max_decision_rounds)),
    )
    run_budget = max(0, int(settings.per_run_token_budget))
    agent_budget = max(0, int(settings.review_agent_token_budget))
    effective_budget = max(1, min(run_budget, agent_budget))
    # One provider response may legitimately overshoot a pre-call estimate.
    # Four bounded rounds-worth leaves headroom without accepting arbitrary
    # provider-controlled integers into cost arithmetic or the database.
    token_count = min(
        _SIGNED_64_MAX,
        effective_budget * max(4, decision_rounds + 2),
    )
    response_chars = min(
        _SIGNED_64_MAX,
        max(1, int(settings.review_agent_max_response_bytes)),
    )
    input_basis = max(
        response_chars,
        int(settings.model_batch_max_chars),
        int(settings.model_chunk_max_chars),
        int(settings.review_agent_max_span_chars),
    )
    input_chars = min(
        _SIGNED_64_MAX, input_basis * max(8, decision_rounds + 2)
    )
    elapsed_ms = min(
        _SIGNED_64_MAX,
        max(
            1,
            int(settings.review_agent_total_deadline_seconds * 1000)
            * max(4, decision_rounds + 2),
        ),
    )
    return {
        "logical_calls": decision_rounds,
        "token_count": token_count,
        "attempt": 1,
        "elapsed_ms": elapsed_ms,
        "input_chars": input_chars,
        "response_chars": response_chars,
    }


def _safe_interrupted_provider_call(row, limits: dict[str, int]) -> dict | None:
    """Re-allowlist already content-free telemetry at the DB boundary."""
    if not isinstance(row, dict):
        return None
    status = row.get("status")
    category = row.get("category")
    if status not in {"success", "failure"}:
        return None
    if category not in _SAFE_INTERRUPTED_PROVIDER_CATEGORIES:
        category = "provider"
    integer_limits = {
        "attempt": limits["attempt"],
        "elapsed_ms": limits["elapsed_ms"],
        "input_chars": limits["input_chars"],
        "response_chars": limits["response_chars"],
        "prompt_tokens": limits["token_count"],
        "completion_tokens": limits["token_count"],
        "total_tokens": limits["token_count"],
    }
    normalized = {
        key: _optional_nonnegative_int(row.get(key), maximum=maximum)
        for key, maximum in integer_limits.items()
    }
    if any(
        row.get(key) is not None and normalized[key] is None
        for key in integer_limits
    ):
        # One impossible per-call counter makes this optional series
        # unavailable; aggregate counters are validated independently.
        return None
    http_status = _optional_nonnegative_int(row.get("http_status"), maximum=599)
    if row.get("http_status") is not None and http_status is None:
        return None
    if http_status is not None and not 100 <= http_status <= 599:
        http_status = None
    safe = {
        "status": status,
        "category": category,
        **normalized,
        "http_status": http_status,
        # Request IDs are provider-controlled opaque strings and are not
        # needed to prove interrupted usage. Dropping them entirely avoids a
        # credential-like value passing a permissive identifier regex.
        "request_id": None,
        # This interrupted ledger is exclusively populated by Review Agent
        # calls. Never trust a caller-provided purpose at persistence time.
        "purpose": "agent",
    }
    return safe


def _interrupted_review_agent_usage(pipeline) -> dict | None:
    """Build a safe, explicitly incomplete lower-bound usage diagnostic."""
    getter = getattr(pipeline, "interrupted_model_usage", None)
    if not callable(getter):
        return None
    try:
        source = getter()
    except Exception:
        # Accounting must never prevent cancellation from reaching a durable
        # terminal state. A broken custom extractor simply has no proof.
        return None
    if not isinstance(source, dict):
        return None
    limits = _interrupted_usage_limits()
    counters = {
        "logical_calls": _optional_nonnegative_int(
            source.get("logical_calls"), maximum=limits["logical_calls"]
        ),
        **{
            key: _optional_nonnegative_int(
                source.get(key), maximum=limits["token_count"]
            )
            for key in (
                "prompt_tokens",
                "completion_tokens",
                "charged_tokens",
            )
        },
    }
    if any(value is None for value in counters.values()):
        return None
    logical_calls = counters["logical_calls"]
    prompt_tokens = counters["prompt_tokens"]
    completion_tokens = counters["completion_tokens"]
    charged_tokens = counters["charged_tokens"]
    if (
        not logical_calls
        or charged_tokens < prompt_tokens + completion_tokens
    ):
        return None

    raw_calls = source.get("provider_calls")
    provider_calls = None
    if isinstance(raw_calls, list):
        sanitized = [
            _safe_interrupted_provider_call(row, limits) for row in raw_calls
        ]
        # A partial or malformed series is unavailable, not a fake complete
        # list. Aggregate lower-bound counters remain independently valid.
        if (
            all(row is not None for row in sanitized)
            and len(sanitized) == logical_calls
        ):
            provider_calls = sanitized
    elif raw_calls is not None:
        provider_calls = None

    return {
        "completeness": "lower_bound",
        "scope": "review_agent_completed_calls_only",
        "terminal_status": "cancelled",
        "logical_calls": logical_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "charged_tokens": charged_tokens,
        "charged_token_semantics": "heuristic_or_reported_internal_debit",
        "provider_calls": provider_calls,
    }


def _combined_interrupted_usage(
    pipeline,
    investigator_usage: InvestigatorUsageAccumulator,
    issue_review_usage: IssueEvidenceReviewUsageAccumulator,
    *,
    terminal_status: str,
    character_usage: CharacterConsistencyUsageAccumulator | None = None,
) -> dict | None:
    """Combine independently content-free completed-call ledgers."""
    agent = _interrupted_review_agent_usage(pipeline) if pipeline is not None else None
    investigator = investigator_usage.safe_dict(
        terminal_status=terminal_status
    )
    evidence = issue_review_usage.safe_dict(terminal_status=terminal_status)
    character = (
        character_usage.safe_dict(terminal_status=terminal_status)
        if character_usage is not None
        else None
    )
    parts = [
        row for row in (agent, character, investigator, evidence) if row is not None
    ]
    if not parts:
        return None
    if len(parts) == 1:
        # Preserve the established Review Agent lower-bound contract when no
        # evidence-review call participated in this terminal attempt.
        return dict(parts[0])
    calls: list[dict] | None = []
    for row in parts:
        row_calls = row.get("provider_calls")
        if not isinstance(row_calls, list):
            calls = None
            break
        calls.extend(row_calls)
    return {
        "completeness": "completed_calls",
        "scope": "combined_model_usage",
        "terminal_status": terminal_status,
        "logical_calls": sum(int(row["logical_calls"]) for row in parts),
        "prompt_tokens": sum(int(row["prompt_tokens"]) for row in parts),
        "completion_tokens": sum(int(row["completion_tokens"]) for row in parts),
        "charged_tokens": sum(int(row["charged_tokens"]) for row in parts),
        "charged_token_semantics": "heuristic_or_reported_internal_debit",
        "provider_calls": calls,
    }


def _merge_usage_accounting(
    previous: dict | None,
    current: dict | None,
    *,
    terminal_status: str,
) -> dict | None:
    """Add two already-sanitized attempt ledgers without overwriting either."""

    def valid(row: object) -> bool:
        if not isinstance(row, dict):
            return False
        values = [
            row.get("logical_calls"),
            row.get("prompt_tokens"),
            row.get("completion_tokens"),
            row.get("charged_tokens"),
        ]
        return (
            all(
                type(value) is int
                and 0 <= value <= _SIGNED_64_MAX // 4
                for value in values
            )
            and values[3] >= values[1] + values[2]
            and row.get("charged_token_semantics")
            in {
                "conservative_internal_budget_debit",
                "heuristic_or_reported_internal_debit",
            }
        )

    older = previous if valid(previous) else None
    newer = current if valid(current) else None
    if older is None:
        if newer is None:
            return None
        result = dict(newer)
        result["terminal_status"] = terminal_status
        result[
            "charged_token_semantics"
        ] = "heuristic_or_reported_internal_debit"
        return result
    if newer is None:
        result = dict(older)
        result["terminal_status"] = terminal_status
        result[
            "charged_token_semantics"
        ] = "heuristic_or_reported_internal_debit"
        return result
    sums = {
        key: int(older[key]) + int(newer[key])
        for key in (
            "logical_calls",
            "prompt_tokens",
            "completion_tokens",
            "charged_tokens",
        )
    }
    return {
        "completeness": (
            "lower_bound"
            if "lower_bound"
            in {older.get("completeness"), newer.get("completeness")}
            else "completed_calls"
        ),
        "scope": "combined_model_usage",
        "terminal_status": terminal_status,
        **sums,
        "charged_token_semantics": "heuristic_or_reported_internal_debit",
        # Aggregate counters remain exact. A multi-attempt per-call series is
        # intentionally unavailable instead of pretending it is complete.
        "provider_calls": None,
    }


def _finalize_terminal(
    run_id: str,
    worker_token: str,
    status: str,
    *,
    error: str | None,
    message: str,
    interrupted_usage: dict | None = None,
) -> bool:
    with SessionLocal() as db:
        diagnostic = db.get(AnalysisDiagnosticRow, run_id)
        existing_payload = dict(diagnostic.payload) if diagnostic else {}
        cumulative_usage = _merge_usage_accounting(
            existing_payload.get("usage_accounting"),
            interrupted_usage,
            terminal_status=status,
        )
        conditions = [
            AnalysisRunRow.id == run_id,
            AnalysisRunRow.status.not_in(TERMINAL_STATUSES),
            _owned_run_clause(run_id, worker_token),
        ]
        if status == "completed":
            conditions.append(AnalysisRunRow.cancel_requested.is_(False))
        values = {
            "status": status,
            "error": safe_persisted_analysis_error(error),
            "completed_at": utc_now_naive(),
        }
        if status in {"cancelled", "failed"} and cumulative_usage is not None:
            values.update(
                prompt_tokens=cumulative_usage["prompt_tokens"],
                completion_tokens=cumulative_usage["completion_tokens"],
                estimated_cost_usd=(
                    configured_cost_usd(
                        cumulative_usage["prompt_tokens"],
                        cumulative_usage["completion_tokens"],
                        get_settings(),
                    )
                    or 0
                ),
            )
        changed = db.execute(
            update(AnalysisRunRow)
            .where(*conditions)
            .values(**values)
        ).rowcount
        if changed != 1:
            db.rollback()
            return False
        db.add(
            RunEventRow(
                run_id=run_id, stage=status, progress=100, message=message
            )
        )
        if status in {"cancelled", "failed"} and cumulative_usage is not None:
            payload = existing_payload
            payload["usage_accounting"] = cumulative_usage
            if diagnostic:
                diagnostic.payload = payload
            else:
                db.add(AnalysisDiagnosticRow(run_id=run_id, payload=payload))
        db.execute(
            update(AnalysisRunExecutionRow)
            .where(
                AnalysisRunExecutionRow.run_id == run_id,
                AnalysisRunExecutionRow.worker_token == worker_token,
            )
            .values(worker_token=None, lease_expires_at=None)
        )
        db.commit()
        return True


def _release_failed_attempt(
    run_id: str,
    worker_token: str,
    error: str,
    *,
    interrupted_usage: dict | None = None,
) -> None:
    """Release a retriable attempt without creating a terminal SSE event."""
    public_error = safe_persisted_analysis_error(error) or INTERNAL_ANALYSIS_ERROR
    with SessionLocal() as db:
        changed = db.execute(
            update(AnalysisRunRow)
            .where(
                AnalysisRunRow.id == run_id,
                AnalysisRunRow.status == "running",
                _owned_run_clause(run_id, worker_token),
            )
            .values(status="queued", error=public_error, completed_at=None)
        ).rowcount
        if changed != 1:
            db.rollback()
            return
        diagnostic = db.get(AnalysisDiagnosticRow, run_id)
        payload = dict(diagnostic.payload) if diagnostic else {}
        cumulative_usage = _merge_usage_accounting(
            payload.get("usage_accounting"),
            interrupted_usage,
            terminal_status="running",
        )
        if cumulative_usage is not None:
            payload["usage_accounting"] = cumulative_usage
            if diagnostic:
                diagnostic.payload = payload
            else:
                db.add(AnalysisDiagnosticRow(run_id=run_id, payload=payload))
            # Keep the run's reported counters aligned with the accumulated
            # ledger while it waits in queued backoff. Daily debit can then add
            # only the conservative charged-minus-reported delta.
            db.execute(
                update(AnalysisRunRow)
                .where(
                    AnalysisRunRow.id == run_id,
                    _owned_run_clause(run_id, worker_token),
                )
                .values(
                    prompt_tokens=cumulative_usage["prompt_tokens"],
                    completion_tokens=cumulative_usage["completion_tokens"],
                    estimated_cost_usd=(
                        configured_cost_usd(
                            cumulative_usage["prompt_tokens"],
                            cumulative_usage["completion_tokens"],
                            get_settings(),
                        )
                        or 0
                    ),
                )
            )
        db.execute(
            update(AnalysisRunExecutionRow)
            .where(
                AnalysisRunExecutionRow.run_id == run_id,
                AnalysisRunExecutionRow.worker_token == worker_token,
            )
            .values(worker_token=None, lease_expires_at=None)
        )
        last_progress = db.scalar(
            select(func.max(RunEventRow.progress)).where(RunEventRow.run_id == run_id)
        ) or 0
        db.add(
            RunEventRow(
                run_id=run_id,
                stage="attempt_failed",
                progress=last_progress,
                message="本次执行失败，等待受限的工作进程重试",
            )
        )
        db.commit()


def _load_verified_snapshot(db, run_id: str) -> tuple[list[DocumentInput], list[dict]]:
    rows = db.scalars(
        select(AnalysisRunInputRow)
        .where(AnalysisRunInputRow.run_id == run_id)
        .order_by(AnalysisRunInputRow.ordinal)
    ).all()
    if not rows:
        raise RuntimeError(MISSING_SNAPSHOT_ERROR)
    contexts = {
        context.input_id: context
        for context in db.scalars(
            select(AnalysisRunInputContextRow).where(
                AnalysisRunInputContextRow.input_id.in_([row.id for row in rows])
            )
        ).all()
    }
    narrative_contexts = {
        context.input_id: context
        for context in db.scalars(
            select(AnalysisRunInputNarrativeContextRow).where(
                AnalysisRunInputNarrativeContextRow.input_id.in_(
                    [row.id for row in rows]
                )
            )
        ).all()
    }
    if narrative_contexts and len(narrative_contexts) != len(rows):
        raise RuntimeError(
            "RUN_INPUT_SNAPSHOT_CORRUPT: narrative context snapshot is incomplete"
        )
    run = db.get(AnalysisRunRow, run_id)
    if run is None:
        raise RuntimeError("RUN_INPUT_SNAPSHOT_CORRUPT: analysis run is missing")
    batch_mode = run.batch_mode or "full_review"
    sensitivity = run.sensitivity or "balanced"
    if batch_mode not in {"draft_review", "baseline_build", "full_review"}:
        raise RuntimeError(
            "RUN_INPUT_SNAPSHOT_CORRUPT: review batch mode is invalid"
        )
    if sensitivity not in {"conservative", "balanced", "exploratory"}:
        raise RuntimeError(
            "RUN_INPUT_SNAPSHOT_CORRUPT: review batch sensitivity is invalid"
        )
    trait_rows = list(
        db.scalars(
            select(AnalysisRunCharacterTraitInputRow)
            .where(AnalysisRunCharacterTraitInputRow.run_id == run_id)
            .order_by(AnalysisRunCharacterTraitInputRow.ordinal)
        ).all()
    )
    if [row.ordinal for row in trait_rows] != list(range(len(trait_rows))):
        raise RuntimeError(
            "RUN_INPUT_SNAPSHOT_CORRUPT: character profile ordinals are invalid"
        )
    for trait in trait_rows:
        if (
            trait.project_id != run.project_id
            or payload_sha256(trait.payload) != trait.payload_sha256
            or trait.payload.get("candidate_id") != trait.candidate_id
            or trait.payload.get("confirmation_review_id")
            != trait.confirmation_review_id
            or trait.payload.get("candidate_lock_version")
            != trait.candidate_lock_version
        ):
            raise RuntimeError(
                "RUN_INPUT_SNAPSHOT_CORRUPT: character profile snapshot is invalid"
            )
    documents: list[DocumentInput] = []
    metadata: list[dict] = []
    frozen_batch_roles: list[str] = []
    for row in rows:
        context = contexts.get(row.id)
        batch_role = context.batch_role if context else "target"
        if batch_role not in {"target", "background"}:
            raise RuntimeError(
                "RUN_INPUT_SNAPSHOT_CORRUPT: input batch role is invalid"
            )
        frozen_batch_roles.append(batch_role)
        actual_hash = document_content_sha256(row.content)
        if actual_hash != row.content_sha256:
            raise RuntimeError(
                f"RUN_INPUT_SNAPSHOT_CORRUPT: document {row.document_id} hash mismatch"
            )
        narrative = narrative_contexts.get(row.id)
        if narrative is not None:
            if (
                narrative.schema_version != 1
                or payload_sha256(narrative.payload) != narrative.payload_sha256
            ):
                raise RuntimeError(
                    "RUN_INPUT_SNAPSHOT_CORRUPT: narrative context hash mismatch"
                )
            narrative_payload = narrative.payload
            narrative_hash = narrative_context_semantic_sha256(
                narrative.payload
            )
        else:
            narrative_payload = context_snapshot_payload(
                None,
                document_role=(
                    context.document_role if context else DEFAULT_DOCUMENT_ROLE
                ),
                story_scope=context.story_scope if context else DEFAULT_STORY_SCOPE,
            )
            narrative_hash = narrative_context_semantic_sha256(
                narrative_payload
            )
        documents.append(
            DocumentInput(
                id=row.document_id,
                name=row.document_name,
                content=row.content,
                role=context.document_role if context else DEFAULT_DOCUMENT_ROLE,
                scope=context.story_scope if context else DEFAULT_STORY_SCOPE,
            )
        )
        metadata.append(
            {
                "document_id": row.document_id,
                "document_name": row.document_name,
                "document_version": row.document_version,
                "document_role": (
                    context.document_role if context else DEFAULT_DOCUMENT_ROLE
                ),
                "story_scope": context.story_scope if context else DEFAULT_STORY_SCOPE,
                "batch_role": batch_role,
                "context_explicit": context is not None,
                "narrative_context": narrative_payload,
                "narrative_context_sha256": narrative_hash,
                "narrative_context_payload_sha256": (
                    narrative.payload_sha256
                    if narrative is not None
                    else payload_sha256(narrative_payload)
                ),
                "content_sha256": row.content_sha256,
                "char_count": len(row.content),
                "ordinal": row.ordinal,
            }
        )
    if batch_mode == "draft_review" and "target" not in frozen_batch_roles:
        raise RuntimeError(
            "RUN_INPUT_SNAPSHOT_CORRUPT: draft review target is missing"
        )
    if batch_mode == "baseline_build" and (
        not frozen_batch_roles or "target" in frozen_batch_roles
    ):
        raise RuntimeError(
            "RUN_INPUT_SNAPSHOT_CORRUPT: baseline inputs must be background"
        )
    return documents, metadata


def _enforce_draft_issue_boundary(
    result,
    *,
    batch_mode: str,
    input_metadata: list[dict],
    documents: list[DocumentInput],
) -> None:
    """Keep a draft report scoped to findings that actually touch its targets.

    Background material is deliberately available to extraction and retrieval,
    but a pre-existing background/background inconsistency is not a defect in
    the newly submitted draft.  Suppressed findings remain visible as an
    aggregate diagnostic so the boundary cannot silently hide its operation.
    """

    target_ids = {
        str(row.get("document_id"))
        for row in input_metadata
        if row.get("batch_role") == "target"
    }
    batch_diagnostic = {
        "mode": batch_mode,
        "target_document_ids": sorted(target_ids),
        "background_document_ids": sorted(
            str(row.get("document_id"))
            for row in input_metadata
            if row.get("batch_role") == "background"
        ),
        "suppressed_background_only_issues": 0,
    }
    if batch_mode != "draft_review":
        result.diagnostics["review_batch"] = batch_diagnostic
        return
    kept = [
        issue
        for issue in result.issues
        if any(span.document_id in target_ids for span in issue.evidence)
    ]
    suppressed = len(result.issues) - len(kept)
    result.issues = kept
    batch_diagnostic["suppressed_background_only_issues"] = suppressed
    result.diagnostics["review_batch"] = batch_diagnostic
    # Provenance is an integrity map over the final report and must therefore
    # be rebuilt after enforcing the target boundary.
    result.diagnostics["provenance"] = build_result_provenance(
        result.directives,
        result.issues,
        documents,
    )


def _character_stage_settings_for_run(settings, run: AnalysisRunRow | None):
    """Create an immutable per-run sensitivity view without mutating globals."""

    sensitivity = run.sensitivity if run is not None else "balanced"
    if sensitivity not in {"conservative", "balanced", "exploratory"}:
        raise RuntimeError(
            "RUN_INPUT_SNAPSHOT_CORRUPT: review batch sensitivity is invalid"
        )
    return settings.model_copy(
        update={"character_consistency_sensitivity": sensitivity}
    )


def execute_analysis(
    run_id: str,
    *,
    raise_on_failure: bool = False,
    finalize_failure: bool = True,
    worker_token: str | None = None,
    raise_on_busy: bool = False,
    heartbeat_factory: Callable[..., ExecutionLeaseHeartbeat] | None = None,
) -> None:
    service_started = perf_counter()
    token = worker_token or str(uuid4())
    if not claim_analysis_run(run_id, token):
        retry_after = active_lease_retry_after(run_id)
        if raise_on_busy and retry_after is not None:
            raise WorkerLeaseBusy(retry_after)
        return
    heartbeat = (heartbeat_factory or ExecutionLeaseHeartbeat)(run_id, token)
    heartbeat.start()
    pipeline = None
    investigator_usage = InvestigatorUsageAccumulator()
    issue_review_usage = IssueEvidenceReviewUsageAccumulator()
    character_usage_tracker = CharacterConsistencyUsageAccumulator()
    previous_usage: dict | None = None
    try:
        _checkpoint(run_id, token, heartbeat)
        with SessionLocal() as db:
            documents, input_metadata = _load_verified_snapshot(db, run_id)
            run = db.get(AnalysisRunRow, run_id)
            if run is None:
                raise ValueError("analysis run is unavailable")
            provider_runtime = runtime_for_analysis_run(
                db, run, base_settings=get_settings()
            )
            settings = provider_runtime.settings
            previous_diagnostic = db.get(AnalysisDiagnosticRow, run_id)
            if previous_diagnostic and isinstance(previous_diagnostic.payload, dict):
                candidate_usage = previous_diagnostic.payload.get("usage_accounting")
                previous_usage = _merge_usage_accounting(
                    None,
                    candidate_usage if isinstance(candidate_usage, dict) else None,
                    terminal_status="running",
                )
            if getattr(AnalysisPipeline, "accepts_run_local_settings", False):
                pipeline = AnalysisPipeline(settings=settings)
            else:
                # Preserve the documented injectable orchestration boundary
                # for zero-argument test/adaptor pipelines. The real pipeline
                # always takes the guarded branch above.
                pipeline = AnalysisPipeline()

            def on_stage(stage: str, progress: int, message: str) -> None:
                _checkpoint(run_id, token, heartbeat)
                emit(db, run_id, stage, progress, message)

            result = pipeline.run(
                documents,
                on_stage=on_stage,
                checkpoint=lambda: _checkpoint(run_id, token, heartbeat),
            )
            result.diagnostics["account_provider_execution"] = {
                "identity": provider_runtime.identity,
                "available": provider_runtime.available,
                "unavailable_reason": (
                    provider_runtime.unavailable_reason
                    if provider_runtime.unavailable_reason
                    in {
                        None,
                        "not_configured",
                        "credential_revoked",
                        "credential_unavailable",
                        "identity_mismatch",
                        "endpoint_rejected",
                    }
                    else "provider_unavailable"
                ),
            }
            _checkpoint(run_id, token, heartbeat)
            review_result = None
            baseline_reported_tokens = (
                result.prompt_tokens + result.completion_tokens
            )
            pipeline_budget_debit = baseline_reported_tokens
            if (
                settings.enable_evidence_investigator
                or settings.enable_character_consistency
            ):
                pipeline_budget_getter = getattr(
                    pipeline, "conservative_run_token_debit", None
                )
                if callable(pipeline_budget_getter):
                    try:
                        candidate_debit = pipeline_budget_getter(result)
                    except Exception:
                        candidate_debit = baseline_reported_tokens
                    if (
                        type(candidate_debit) is int
                        and candidate_debit >= baseline_reported_tokens
                    ):
                        pipeline_budget_debit = candidate_debit
            historical_charged_tokens = (
                int(previous_usage["charged_tokens"])
                if previous_usage is not None
                else 0
            )
            character_stage_settings = _character_stage_settings_for_run(
                settings, run
            )
            character_stage_result = None
            character_stage_db = CharacterConsistencyDatabase(
                SessionLocal,
                run_id=run_id,
                worker_token=token,
            )
            if settings.enable_character_consistency:
                _checkpoint(run_id, token, heartbeat)
                _emit_owned(
                    db,
                    run_id,
                    token,
                    "character_consistency",
                    76,
                    "正在从冻结设定与新稿中核对角色一致性",
                )
            try:
                if run is None:
                    raise ValueError("analysis run is unavailable")
                character_stage_result = CharacterConsistencyStage(
                    settings=character_stage_settings,
                    provider=_CharacterConsistencyAccountingProvider(
                        character_stage_settings,
                        character_usage_tracker,
                    ),
                    checkpoint=lambda: _checkpoint(
                        run_id, token, heartbeat
                    ),
                ).run(
                    character_stage_db,
                    run_id=run_id,
                    project_id=run.project_id,
                    documents=documents,
                    metadata=input_metadata,
                    remaining_run_tokens=max(
                        0,
                        settings.per_run_token_budget
                        - pipeline_budget_debit
                        - historical_charged_tokens,
                    ),
                )
                character_stage_db.raise_if_ownership_lost()
            except (AnalysisCancelled, WorkerLeaseLost):
                raise
            except Exception:
                character_stage_result = failed_character_consistency_stage()

            tracked_character_usage = character_usage_tracker.safe_dict(
                terminal_status="completed"
            )
            character_usage = (
                tracked_character_usage
                or character_stage_result.usage_accounting(
                    terminal_status="completed"
                )
            )
            character_charged_tokens = (
                int(character_usage["charged_tokens"])
                if character_usage is not None
                else 0
            )
            if tracked_character_usage is not None:
                # Keep diagnostics useful after an integration failure without
                # exposing prompts or provider payloads.
                character_stage_result.diagnostics["usage"] = {
                    "attempted_calls": tracked_character_usage["logical_calls"],
                    "input_tokens": tracked_character_usage["prompt_tokens"],
                    "completion_tokens": tracked_character_usage[
                        "completion_tokens"
                    ],
                    "charged_tokens": tracked_character_usage["charged_tokens"],
                }

            result.diagnostics["character_consistency"] = (
                character_stage_result.diagnostics
            )
            if settings.enable_character_consistency:
                if character_stage_result.issues:
                    try:
                        combined_issues = [
                            *result.issues,
                            *character_stage_result.issues,
                        ]
                        combined_provenance = build_result_provenance(
                            result.directives,
                            combined_issues,
                            documents,
                        )
                    except Exception:
                        result.diagnostics[
                            "character_consistency"
                        ] = failed_character_consistency_stage().diagnostics
                    else:
                        result.issues = combined_issues
                        result.diagnostics["provenance"] = combined_provenance
                if character_usage is not None:
                    result.prompt_tokens += int(character_usage["prompt_tokens"])
                    result.completion_tokens += int(
                        character_usage["completion_tokens"]
                    )
                result.model_used = (
                    result.model_used
                    or character_stage_result.model_used
                    or character_usage_tracker.successful_calls > 0
                )
                _checkpoint(run_id, token, heartbeat)
                _emit_owned(
                    db,
                    run_id,
                    token,
                    "character_consistency",
                    78,
                    (
                        f"角色一致性阶段新增 {len(character_stage_result.issues)} 条问题"
                        if character_stage_result.issues
                        else "角色一致性阶段未追加问题或已安全降级"
                    ),
                )
                _checkpoint(run_id, token, heartbeat)
            frozen_bundle = None
            if (
                settings.enable_evidence_investigator
                or settings.enable_issue_evidence_review
            ):
                try:
                    if run is None:
                        raise ValueError("analysis run is unavailable")
                    frozen_bundle = build_frozen_investigation_bundle(
                        project_id=run.project_id,
                        documents=documents,
                        metadata=input_metadata,
                    )
                except Exception:
                    # Optional model stages may only consume the verified
                    # in-memory projection. A binding defect therefore closes
                    # both stages without consulting live DocumentRow values.
                    frozen_bundle = None

            if settings.enable_evidence_investigator:
                _checkpoint(run_id, token, heartbeat)
                _emit_owned(
                    db,
                    run_id,
                    token,
                    "evidence_investigator",
                    80 if settings.enable_character_consistency else 76,
                    "正在用冻结版本建立证据调查范围",
                )
                investigator_runtime_result = None
                try:
                    if frozen_bundle is None:
                        raise ValueError("investigator snapshot bundle unavailable")
                    investigator_runtime_result = EvidenceInvestigatorRuntime(
                        session_factory=SessionLocal,
                        settings=settings,
                        checkpoint=lambda: _checkpoint(
                            run_id, token, heartbeat
                        ),
                        usage=investigator_usage,
                        passthrough_exceptions=(
                            AnalysisCancelled,
                            WorkerLeaseLost,
                        ),
                    ).run(
                        run_id=run_id,
                        bundle=frozen_bundle,
                        baseline_directives=tuple(result.directives),
                        baseline_issues=tuple(result.issues),
                        remaining_run_tokens=max(
                            0,
                            settings.per_run_token_budget
                            - pipeline_budget_debit
                            - historical_charged_tokens
                            - character_charged_tokens,
                        ),
                    )
                    investigator_diagnostics = dict(
                        investigator_runtime_result.diagnostics
                    )
                except (AnalysisCancelled, WorkerLeaseLost):
                    raise
                except Exception:
                    # Provider, RAG, promotion, and integration defects are
                    # local to this optional additive path. Never turn them
                    # into a Celery retry or erase deterministic baseline rows.
                    investigator_diagnostics = {
                        "enabled": True,
                        "outcome": "degraded",
                        "reason_code": "internal_failure",
                        "seed_count": 0,
                        "snapshot_fingerprint": (
                            frozen_bundle.fingerprint
                            if frozen_bundle is not None
                            else None
                        ),
                        "budget_preflight": None,
                        "loop": None,
                        "rag": None,
                        "promotion": None,
                        "usage": investigator_usage.safe_dict(
                            terminal_status="completed"
                        ),
                        "boundary": (
                            "Optional additive stage; any degradation preserves "
                            "the baseline directives, issues, IDs, order and "
                            "provenance."
                        ),
                    }

                completed_investigator_usage = investigator_usage.safe_dict(
                    terminal_status="completed"
                )
                investigator_diagnostics["usage"] = (
                    completed_investigator_usage
                )
                _checkpoint(run_id, token, heartbeat)
                _emit_owned(
                    db,
                    run_id,
                    token,
                    "evidence_investigator",
                    82 if settings.enable_character_consistency else 78,
                    (
                        "证据调查已完成受控检索"
                        if investigator_diagnostics.get("outcome") == "completed"
                        else "证据调查已安全跳过或降级"
                    ),
                )

                accepted_candidates = 0
                added_issues = 0
                if (
                    investigator_runtime_result is not None
                    and investigator_runtime_result.applied
                    and investigator_runtime_result.promotion is not None
                ):
                    promotion = investigator_runtime_result.promotion
                    try:
                        promoted_directives = list(promotion.directives)
                        promoted_issues = list(promotion.issues)
                        promoted_provenance = build_result_provenance(
                            promoted_directives,
                            promoted_issues,
                            documents,
                        )
                        # Compute every replacement before mutating the
                        # PipelineResult. The reviewer below can therefore see
                        # the complete promoted issue set, or the exact
                        # baseline—never a partially applied candidate.
                        promoted_diagnostics = dict(result.diagnostics)
                        promoted_diagnostics["provenance"] = promoted_provenance
                        promoted_diagnostics[
                            "evidence_investigator"
                        ] = investigator_diagnostics
                        accepted_candidates = promotion.accepted_candidates
                        added_issues = len(promotion.added_issues)
                    except (AnalysisCancelled, WorkerLeaseLost):
                        raise
                    except Exception:
                        investigator_diagnostics.update(
                            {
                                "outcome": "degraded",
                                "reason_code": "promotion_failed",
                                "promotion": None,
                            }
                        )
                    else:
                        result.directives = promoted_directives
                        result.issues = promoted_issues
                        result.diagnostics = promoted_diagnostics

                result.diagnostics[
                    "evidence_investigator"
                ] = investigator_diagnostics
                result.prompt_tokens += investigator_usage.prompt_tokens
                result.completion_tokens += (
                    investigator_usage.completion_tokens
                )
                _checkpoint(run_id, token, heartbeat)
                _emit_owned(
                    db,
                    run_id,
                    token,
                    "evidence_investigator",
                    85 if settings.enable_character_consistency else 81,
                    (
                        f"确定性复核接受 {accepted_candidates} 条候选，新增 "
                        f"{added_issues} 条问题"
                        if accepted_candidates
                        else "确定性复核未追加问题，基线结果保持原样"
                    ),
                )
                _checkpoint(run_id, token, heartbeat)

            _enforce_draft_issue_boundary(
                result,
                batch_mode=(run.batch_mode if run is not None else "full_review"),
                input_metadata=input_metadata,
                documents=documents,
            )

            if settings.enable_issue_evidence_review:
                _checkpoint(run_id, token, heartbeat)
                _emit_owned(
                    db,
                    run_id,
                    token,
                    "evidence_review",
                    (
                        86
                        if settings.enable_character_consistency
                        and settings.enable_evidence_investigator
                        else 82
                    ),
                    "正在用冻结版本的检索证据复核规则问题",
                )
                try:
                    if frozen_bundle is None:
                        raise ValueError("review snapshot bundle unavailable")
                    review_result = IssueEvidenceReviewer(
                        session_factory=SessionLocal,
                        settings=settings,
                        checkpoint=lambda: _checkpoint(run_id, token, heartbeat),
                        usage_accounting=issue_review_usage.record,
                    ).review(
                        documents=frozen_bundle.evidence_documents,
                        issues=tuple(result.issues),
                        remaining_run_tokens=max(
                            0,
                            settings.per_run_token_budget
                            - pipeline_budget_debit
                            - historical_charged_tokens
                            - character_charged_tokens
                            - investigator_usage.charged_tokens,
                        ),
                    )
                except (AnalysisCancelled, WorkerLeaseLost):
                    raise
                except Exception:
                    # The reviewer is an optional annotation path.  Its own
                    # unexpected defect must be explicit, content-free and
                    # unable to erase the deterministic report.
                    review_result = failed_issue_evidence_review()
                result.diagnostics["ai_evidence_review"] = review_result.diagnostics
                completed_usage = issue_review_usage.safe_dict(
                    terminal_status="completed"
                )
                if completed_usage is not None:
                    result.diagnostics["ai_evidence_review"][
                        "usage_accounting"
                    ] = completed_usage
                    result.prompt_tokens += issue_review_usage.prompt_tokens
                    result.completion_tokens += issue_review_usage.completion_tokens
                else:
                    # Compatibility for injected legacy/test reviewers that do
                    # not expose the immediate accounting callback.
                    result.prompt_tokens += review_result.prompt_tokens
                    result.completion_tokens += review_result.completion_tokens
                _checkpoint(run_id, token, heartbeat)
                _emit_owned(
                    db,
                    run_id,
                    token,
                    "evidence_review",
                    (
                        90
                        if settings.enable_character_consistency
                        and settings.enable_evidence_investigator
                        else 88
                    ),
                    (
                        f"证据复核已注释 {len(review_result.annotations)} 条规则问题"
                        if review_result.annotations
                        else "证据复核未生成注释，规则问题保持原样"
                    ),
                )
                _checkpoint(run_id, token, heartbeat)
            current_optional_usage = _merge_usage_accounting(
                character_usage,
                _merge_usage_accounting(
                    investigator_usage.safe_dict(terminal_status="completed"),
                    issue_review_usage.safe_dict(terminal_status="completed"),
                    terminal_status="completed",
                ),
                terminal_status="completed",
            )
            cumulative_usage = _merge_usage_accounting(
                previous_usage,
                current_optional_usage,
                terminal_status="completed",
            )
            if cumulative_usage is not None:
                result.diagnostics["usage_accounting"] = cumulative_usage
                if previous_usage is not None:
                    result.prompt_tokens += int(previous_usage["prompt_tokens"])
                    result.completion_tokens += int(
                        previous_usage["completion_tokens"]
                    )
            report_started = perf_counter()
            db.execute(delete(IssueRow).where(IssueRow.run_id == run_id))
            db.execute(
                delete(AnalysisRecordRow).where(AnalysisRecordRow.run_id == run_id)
            )
            db.execute(
                delete(AnalysisDiagnosticRow).where(
                    AnalysisDiagnosticRow.run_id == run_id
                )
            )
            for directive in result.directives:
                db.add(
                    AnalysisRecordRow(
                        run_id=run_id,
                        kind=directive.kind,
                        attrs=directive.attrs,
                        evidence=directive.evidence.model_dump(),
                    )
                )
            for issue in result.issues:
                extra = dict(issue.metadata)
                if review_result is not None:
                    annotation = review_result.annotations.get(str(issue.id))
                    if annotation is not None:
                        extra["ai_evidence_review"] = annotation
                db.add(
                    IssueRow(
                        run_id=run_id,
                        category=issue.category.value,
                        severity=issue.severity.value,
                        confidence=issue.confidence,
                        title=issue.title,
                        explanation=issue.explanation,
                        evidence=[span.model_dump() for span in issue.evidence],
                        suggestion=issue.suggestion,
                        extra=extra,
                    )
                )

            timings = result.diagnostics.setdefault("timings", {})
            timings["report_ms"] = round(
                float(timings.get("report_ms", 0))
                + (perf_counter() - report_started) * 1000,
                3,
            )
            timings["total_ms"] = round(
                (perf_counter() - service_started) * 1000, 3
            )
            mode, partial_fallback = analysis_mode(result)
            result.diagnostics.setdefault("model", {}).update(
                {
                    "used": result.model_used,
                    "partial_fallback": partial_fallback,
                    "mode": mode,
                }
            )
            result.diagnostics["input_snapshot"] = {
                "immutable": True,
                "documents": input_metadata,
                "narrative_context_snapshot_sha256": payload_sha256(
                    [
                        {
                            "document_id": row["document_id"],
                            "narrative_context_sha256": row[
                                "narrative_context_sha256"
                            ],
                            "ordinal": row["ordinal"],
                        }
                        for row in input_metadata
                    ]
                ),
                **run_trait_snapshot_metadata(db, run_id),
            }
            # API and worker each report the same content-free identity. The
            # live E2E runner compares them so a stale worker image or a
            # divergent model/RAG configuration fails provenance closed.
            result.diagnostics["runtime_provenance"] = safe_runtime_provenance(
                settings
            )
            db.add(AnalysisDiagnosticRow(run_id=run_id, payload=result.diagnostics))
            cost = configured_cost_usd(
                result.prompt_tokens, result.completion_tokens, get_settings()
            )
            completed = db.execute(
                update(AnalysisRunRow)
                .where(
                    AnalysisRunRow.id == run_id,
                    AnalysisRunRow.status == "running",
                    AnalysisRunRow.cancel_requested.is_(False),
                    _owned_run_clause(run_id, token),
                )
                .values(
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    estimated_cost_usd=cost if cost is not None else 0,
                    status="completed",
                    completed_at=utc_now_naive(),
                )
            ).rowcount
            if completed != 1:
                db.rollback()
                _checkpoint(run_id, token, heartbeat)
                raise WorkerLeaseLost("run could not be completed by this worker")

            # Comparison depends on the final diagnostics and on the guarded
            # completed transition above. A matcher defect must not discard a
            # valid analysis, but it must also never leave a completed recheck
            # looking silently pending: roll back its savepoint and persist an
            # explicit all-unverifiable comparison instead.
            try:
                with db.begin_nested():
                    materialize_run_comparison(db, run_id)
            except Exception:
                with db.begin_nested():
                    mark_comparison_unverifiable(db, run_id)

            last_progress = db.scalar(
                select(func.max(RunEventRow.progress)).where(
                    RunEventRow.run_id == run_id
                )
            ) or 0
            if result.warnings:
                db.add(
                    RunEventRow(
                        run_id=run_id,
                        stage="warning",
                        progress=max(last_progress, 90),
                        message="；".join(result.warnings[:5]),
                    )
                )
                last_progress = max(last_progress, 90)
            db.add(
                RunEventRow(
                    run_id=run_id,
                    stage="extract",
                    progress=max(last_progress, 92),
                    message=f"本次使用{mode}，保存 {len(result.directives)} 条记录",
                )
            )
            db.add(
                RunEventRow(
                    run_id=run_id,
                    stage="completed",
                    progress=100,
                    message="证据化报告生成完成",
                )
            )
            db.execute(
                update(AnalysisRunExecutionRow)
                .where(
                    AnalysisRunExecutionRow.run_id == run_id,
                    AnalysisRunExecutionRow.worker_token == token,
                )
                .values(worker_token=None, lease_expires_at=None)
            )
            db.commit()
    except AnalysisCancelled:
        interrupted_usage = _combined_interrupted_usage(
            pipeline,
            investigator_usage,
            issue_review_usage,
            terminal_status="cancelled",
            character_usage=character_usage_tracker,
        )
        _finalize_terminal(
            run_id,
            token,
            "cancelled",
            error=None,
            message="任务已按取消请求停止",
            interrupted_usage=interrupted_usage,
        )
    except WorkerLeaseHeartbeatError:
        if raise_on_failure:
            raise
        return
    except WorkerLeaseLost:
        return
    except Exception as exc:
        try:
            _checkpoint(run_id, token, heartbeat)
        except AnalysisCancelled:
            interrupted_usage = _combined_interrupted_usage(
                pipeline,
                investigator_usage,
                issue_review_usage,
                terminal_status="cancelled",
                character_usage=character_usage_tracker,
            )
            _finalize_terminal(
                run_id,
                token,
                "cancelled",
                error=None,
                message="任务已按取消请求停止",
                interrupted_usage=interrupted_usage,
            )
            return
        except WorkerLeaseHeartbeatError:
            # A provider error can race with a renewal database failure.  Do
            # not collapse that observable coordination failure into the
            # ordinary ownership-loss path used for a legitimate takeover.
            if raise_on_failure:
                raise
            return
        except WorkerLeaseLost:
            return
        public_error = safe_analysis_error(exc)
        if finalize_failure:
            interrupted_usage = _combined_interrupted_usage(
                pipeline,
                investigator_usage,
                issue_review_usage,
                terminal_status="failed",
                character_usage=character_usage_tracker,
            )
            _finalize_terminal(
                run_id,
                token,
                "failed",
                error=public_error,
                message="分析失败，可调用重试接口恢复",
                interrupted_usage=interrupted_usage,
            )
        else:
            interrupted_usage = _combined_interrupted_usage(
                pipeline,
                investigator_usage,
                issue_review_usage,
                terminal_status="running",
                character_usage=character_usage_tracker,
            )
            _release_failed_attempt(
                run_id,
                token,
                public_error,
                interrupted_usage=interrupted_usage,
            )
        if raise_on_failure:
            raise
    finally:
        heartbeat.stop()
