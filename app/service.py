from __future__ import annotations

from datetime import timedelta
from hashlib import sha256
from math import ceil
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Callable
from uuid import uuid4

from sqlalchemy import delete, exists, func, or_, select, update

from .config import get_settings
from .db import (
    AnalysisDiagnosticRow,
    AnalysisRecordRow,
    AnalysisRunExecutionRow,
    AnalysisRunInputContextRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    DocumentContextRow,
    DocumentRow,
    IssueRow,
    RunEventRow,
    SessionLocal,
)
from .domain import AnalysisCancelled
from .evidence_chunks import SnapshotDocumentKey
from .evidence_rag import EvidenceDocument
from .issue_evidence_review import (
    IssueEvidenceReviewUsageAccumulator,
    IssueEvidenceReviewer,
    failed_issue_evidence_review,
)
from .pipeline import AnalysisPipeline, DocumentInput
from .time_utils import utc_now_naive
from .usage import configured_cost_usd


TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
TERMINAL_EVENT_STAGES = TERMINAL_STATUSES
WORKER_HEARTBEAT_JOIN_TIMEOUT_SECONDS = 5
DEFAULT_DOCUMENT_ROLE = "chapter"
DEFAULT_STORY_SCOPE = "global"
MISSING_SNAPSHOT_ERROR = (
    "RUN_INPUT_SNAPSHOT_MISSING: 此旧任务创建时未冻结输入，不能读取当前文档冒充原输入；"
    "请从项目重新发起分析以使用当前活动版本"
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


def document_content_sha256(content: str) -> str:
    return sha256(content.encode("utf-8")).hexdigest()


def capture_run_inputs(
    db, run: AnalysisRunRow, documents: list[DocumentRow]
) -> list[AnalysisRunInputRow]:
    """Freeze current document bodies in the caller's creation transaction."""
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
            )
        )
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
            "context_explicit": row.id in contexts,
            "content_sha256": row.content_sha256,
            "char_count": len(row.content),
            "ordinal": row.ordinal,
        }
        for row in rows
    ]


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
            "模型增强（部分分块已降级）"
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
        "charged_token_semantics": "conservative_internal_budget_debit",
        "provider_calls": provider_calls,
    }


def _combined_interrupted_usage(
    pipeline,
    issue_review_usage: IssueEvidenceReviewUsageAccumulator,
    *,
    terminal_status: str,
) -> dict | None:
    """Combine independently content-free completed-call ledgers."""
    agent = _interrupted_review_agent_usage(pipeline) if pipeline is not None else None
    evidence = issue_review_usage.safe_dict(terminal_status=terminal_status)
    parts = [row for row in (agent, evidence) if row is not None]
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
        "charged_token_semantics": "conservative_internal_budget_debit",
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
        )

    older = previous if valid(previous) else None
    newer = current if valid(current) else None
    if older is None:
        if newer is None:
            return None
        result = dict(newer)
        result["terminal_status"] = terminal_status
        return result
    if newer is None:
        result = dict(older)
        result["terminal_status"] = terminal_status
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
        "charged_token_semantics": "conservative_internal_budget_debit",
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
            "error": error,
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
    with SessionLocal() as db:
        changed = db.execute(
            update(AnalysisRunRow)
            .where(
                AnalysisRunRow.id == run_id,
                AnalysisRunRow.status == "running",
                _owned_run_clause(run_id, worker_token),
            )
            .values(status="queued", error=error, completed_at=None)
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
    documents: list[DocumentInput] = []
    metadata: list[dict] = []
    for row in rows:
        context = contexts.get(row.id)
        actual_hash = document_content_sha256(row.content)
        if actual_hash != row.content_sha256:
            raise RuntimeError(
                f"RUN_INPUT_SNAPSHOT_CORRUPT: document {row.document_id} hash mismatch"
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
                "context_explicit": context is not None,
                "content_sha256": row.content_sha256,
                "char_count": len(row.content),
                "ordinal": row.ordinal,
            }
        )
    return documents, metadata


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
    issue_review_usage = IssueEvidenceReviewUsageAccumulator()
    previous_usage: dict | None = None
    try:
        _checkpoint(run_id, token, heartbeat)
        with SessionLocal() as db:
            documents, input_metadata = _load_verified_snapshot(db, run_id)
            previous_diagnostic = db.get(AnalysisDiagnosticRow, run_id)
            if previous_diagnostic and isinstance(previous_diagnostic.payload, dict):
                candidate_usage = previous_diagnostic.payload.get("usage_accounting")
                previous_usage = _merge_usage_accounting(
                    None,
                    candidate_usage if isinstance(candidate_usage, dict) else None,
                    terminal_status="running",
                )
            pipeline = AnalysisPipeline()

            def on_stage(stage: str, progress: int, message: str) -> None:
                _checkpoint(run_id, token, heartbeat)
                emit(db, run_id, stage, progress, message)

            result = pipeline.run(
                documents,
                on_stage=on_stage,
                checkpoint=lambda: _checkpoint(run_id, token, heartbeat),
            )
            _checkpoint(run_id, token, heartbeat)
            review_result = None
            settings = get_settings()
            if settings.enable_issue_evidence_review:
                emit(
                    db,
                    run_id,
                    "evidence_review",
                    82,
                    "正在用冻结版本的检索证据复核规则问题",
                )
                run = db.get(AnalysisRunRow, run_id)
                try:
                    frozen_documents = tuple(
                        EvidenceDocument(
                            snapshot=SnapshotDocumentKey(
                                project_id=run.project_id,
                                document_id=metadata["document_id"],
                                document_version=metadata["document_version"],
                                content_sha256=metadata["content_sha256"],
                            ),
                            content=document.content,
                        )
                        for document, metadata in zip(
                            documents, input_metadata, strict=True
                        )
                    )
                    review_result = IssueEvidenceReviewer(
                        session_factory=SessionLocal,
                        settings=settings,
                        checkpoint=lambda: _checkpoint(run_id, token, heartbeat),
                        usage_accounting=issue_review_usage.record,
                    ).review(
                        documents=frozen_documents,
                        issues=tuple(result.issues),
                        remaining_run_tokens=max(
                            0,
                            settings.per_run_token_budget
                            - result.prompt_tokens
                            - result.completion_tokens
                            - (
                                int(previous_usage["charged_tokens"])
                                if previous_usage is not None
                                else 0
                            ),
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
                emit(
                    db,
                    run_id,
                    "evidence_review",
                    88,
                    (
                        f"证据复核已注释 {len(review_result.annotations)} 条规则问题"
                        if review_result.annotations
                        else "证据复核未生成注释，规则问题保持原样"
                    ),
                )
                _checkpoint(run_id, token, heartbeat)
            cumulative_usage = _merge_usage_accounting(
                previous_usage,
                issue_review_usage.safe_dict(terminal_status="completed"),
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
            }
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
            issue_review_usage,
            terminal_status="cancelled",
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
                issue_review_usage,
                terminal_status="cancelled",
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
        if finalize_failure:
            interrupted_usage = _combined_interrupted_usage(
                pipeline,
                issue_review_usage,
                terminal_status="failed",
            )
            _finalize_terminal(
                run_id,
                token,
                "failed",
                error=str(exc),
                message="分析失败，可调用重试接口恢复",
                interrupted_usage=interrupted_usage,
            )
        else:
            interrupted_usage = _combined_interrupted_usage(
                pipeline,
                issue_review_usage,
                terminal_status="running",
            )
            _release_failed_attempt(
                run_id,
                token,
                str(exc),
                interrupted_usage=interrupted_usage,
            )
        if raise_on_failure:
            raise
    finally:
        heartbeat.stop()
