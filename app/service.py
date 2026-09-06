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


def _finalize_terminal(
    run_id: str,
    worker_token: str,
    status: str,
    *,
    error: str | None,
    message: str,
) -> bool:
    with SessionLocal() as db:
        conditions = [
            AnalysisRunRow.id == run_id,
            AnalysisRunRow.status.not_in(TERMINAL_STATUSES),
            _owned_run_clause(run_id, worker_token),
        ]
        if status == "completed":
            conditions.append(AnalysisRunRow.cancel_requested.is_(False))
        changed = db.execute(
            update(AnalysisRunRow)
            .where(*conditions)
            .values(status=status, error=error, completed_at=utc_now_naive())
        ).rowcount
        if changed != 1:
            db.rollback()
            return False
        db.add(
            RunEventRow(
                run_id=run_id, stage=status, progress=100, message=message
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
        db.commit()
        return True


def _release_failed_attempt(run_id: str, worker_token: str, error: str) -> None:
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
    try:
        _checkpoint(run_id, token, heartbeat)
        with SessionLocal() as db:
            documents, input_metadata = _load_verified_snapshot(db, run_id)
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
                        extra=issue.metadata,
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
        _finalize_terminal(
            run_id,
            token,
            "cancelled",
            error=None,
            message="任务已按取消请求停止",
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
            _finalize_terminal(
                run_id,
                token,
                "cancelled",
                error=None,
                message="任务已按取消请求停止",
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
            _finalize_terminal(
                run_id,
                token,
                "failed",
                error=str(exc),
                message="分析失败，可调用重试接口恢复",
            )
        else:
            _release_failed_attempt(run_id, token, str(exc))
        if raise_on_failure:
            raise
    finally:
        heartbeat.stop()
