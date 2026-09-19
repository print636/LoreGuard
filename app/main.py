from __future__ import annotations

import asyncio
import json
import re
from hashlib import sha256
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Thread
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import Scope

from .config import get_settings
from .auth import (
    AuthContext,
    ExactOriginMiddleware,
    get_auth_context,
    require_csrf,
    router as auth_router,
)
from .db import (
    AnalysisDiagnosticRow,
    AnalysisRecordRow,
    AnalysisRunComparisonRow,
    AnalysisRunExecutionRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    DocumentContextRow,
    DocumentRow,
    FeedbackRow,
    IssueComparisonItemRow,
    IssueRow,
    ProjectRow,
    RunEventRow,
    SessionLocal,
    init_db,
)
from .document_diff import build_document_diff
from .docx_import import DocxImportError, extract_docx_text
from .domain import CertaintyLevel, DocumentRole, EvidenceSpan, GraphResponse, SemanticModality, SourceScope, TimelineResponse
from .evaluation import run_evaluation
from .observability import AnalysisMetricsUnavailable, render_analysis_metrics
from .projections import project_graph, project_timeline, record_sort_key
from .provider import (
    OpenAICompatibleProvider,
    ProviderError,
    safe_thinking_configuration,
    sanitize_request_id,
)
from .rate_limit import SlidingWindowLimiter, WriteRateLimitMiddleware
from .runtime_provenance import safe_runtime_provenance
from .run_comparison import (
    MATCHER_VERSION,
    build_input_diff,
    mark_comparison_unverifiable,
    materialize_run_comparison,
)
from .service import (
    DEFAULT_DOCUMENT_ROLE,
    DEFAULT_STORY_SCOPE,
    DISPATCH_FAILED_ERROR,
    MISSING_SNAPSHOT_ERROR,
    capture_run_inputs,
    copy_run_inputs,
    execute_analysis,
    run_input_metadata,
    safe_persisted_analysis_error,
)
from .time_utils import utc_now_naive

@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="LoreGuard API", version="0.1.0", lifespan=lifespan)
settings = get_settings()
write_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute, settings.rate_limit_window_seconds)
app.add_middleware(WriteRateLimitMiddleware, limiter=write_limiter)
app.add_middleware(
    ExactOriginMiddleware,
    allowed_origins=settings.parsed_cors_origins(),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.parsed_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Accept", "Content-Type", "X-CSRF-Token", "Idempotency-Key"],
    expose_headers=["X-CSRF-Token"],
)
app.include_router(auth_router)


class SpaStaticFiles(StaticFiles):
    """Serve known client routes from ``index.html`` without hiding 404s.

    ``StaticFiles(html=True)`` only falls back to an index file for real
    directories. LoreGuard's browser routes are virtual, so a direct refresh
    of ``/login`` or ``/app/projects/...`` otherwise returns 404. The fallback
    is intentionally allow-listed: API paths, asset paths, file-like paths,
    unknown top-level paths, and traversal attempts retain normal static-file
    404 behaviour.
    """

    _LEGACY_WORKSPACE_ROUTES = frozenset(
        {"check", "projects", "diff", "visual", "audit", "report", "provider"}
    )

    @classmethod
    def _is_client_route(cls, path: str, request_path: str = "") -> bool:
        # Starlette builds nested static paths with the host OS separator, so
        # normalise Windows paths before applying URL-segment rules. Encoded
        # backslash traversal still becomes a ``..`` segment and is rejected.
        normalized_path = path.replace("\\", "/")
        normalized_request_path = request_path.replace("\\", "/")
        request_segments = [
            segment
            for segment in normalized_request_path.strip("/").split("/")
            if segment
        ]
        if any(segment in {".", ".."} for segment in request_segments):
            return False
        segments = [
            segment for segment in normalized_path.strip("/").split("/") if segment
        ]
        if any(segment in {".", ".."} for segment in segments):
            return False
        if not segments:
            return True
        if "." in segments[-1]:
            return False
        if len(segments) == 1:
            return segments[0] in {"login", "register", "app"} | cls._LEGACY_WORKSPACE_ROUTES
        return segments[0] == "app"

    async def get_response(self, path: str, scope: Scope) -> Response:
        not_found_response: Response | None = None
        try:
            response = await super().get_response(path, scope)
            if response.status_code != 404:
                return response
            not_found_response = response
        except StarletteHTTPException as exc:
            if exc.status_code != 404:
                raise

        if not self._is_client_route(path, str(scope.get("path", ""))):
            if not_found_response is not None:
                return not_found_response
            raise StarletteHTTPException(status_code=404)
        return await super().get_response("index.html", scope)


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class FeedbackIn(BaseModel):
    label: str
    comment: str = ""


class TextDocumentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)
    replace_document_id: str | None = None
    document_role: DocumentRole | None = None
    story_scope: str | None = Field(
        default=None,
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9_\-\u4e00-\u9fff]+$",
    )


ClarificationCategory = Literal[
    "scope_unknown",
    "missing_causal_bridge",
    "missing_state_transition",
    "ambiguous_reference",
    "insufficient_evidence",
]
ComparisonOutcome = Literal[
    "no_longer_detected", "persisting", "new", "unverifiable"
]


class SemanticReviewItemOut(BaseModel):
    id: str
    kind: Literal["clarification", "open_question"]
    category: ClarificationCategory | None = None
    modality: SemanticModality
    source_scope: SourceScope
    certainty: CertaintyLevel
    text: str
    evidence: EvidenceSpan


def _project_in_workspace(db, project_id: str, workspace_id: str) -> ProjectRow | None:
    return db.scalar(
        select(ProjectRow).where(
            ProjectRow.id == project_id,
            ProjectRow.workspace_id == workspace_id,
        )
    )


def _run_in_workspace(db, run_id: str, workspace_id: str) -> AnalysisRunRow | None:
    return db.scalar(
        select(AnalysisRunRow)
        .join(ProjectRow, ProjectRow.id == AnalysisRunRow.project_id)
        .where(
            AnalysisRunRow.id == run_id,
            ProjectRow.workspace_id == workspace_id,
        )
    )


def _issue_in_workspace(db, issue_id: str, workspace_id: str) -> IssueRow | None:
    return db.scalar(
        select(IssueRow)
        .join(AnalysisRunRow, AnalysisRunRow.id == IssueRow.run_id)
        .join(ProjectRow, ProjectRow.id == AnalysisRunRow.project_id)
        .where(
            IssueRow.id == issue_id,
            ProjectRow.workspace_id == workspace_id,
        )
    )


def serialize_run(row: AnalysisRunRow, db=None) -> dict:
    payload = {key: getattr(row, key) for key in ("id", "project_id", "status", "created_at", "started_at", "completed_at", "input_chars", "prompt_tokens", "completion_tokens", "error")}
    payload["error"] = safe_persisted_analysis_error(row.error)
    prices_configured = settings.model_input_price_per_million is not None and settings.model_output_price_per_million is not None
    payload["estimated_cost_usd"] = row.estimated_cost_usd if prices_configured else None
    if db is not None:
        payload["input_documents"] = run_input_metadata(db, row.id)
        diagnostic = db.get(AnalysisDiagnosticRow, row.id)
        usage_accounting = (
            diagnostic.payload.get("usage_accounting")
            if diagnostic and isinstance(diagnostic.payload, dict)
            else None
        )
        # Additive API qualifier: callers must not interpret interrupted
        # Agent-only token counters as a complete analysis-run total.
        payload["usage_accounting"] = usage_accounting
        execution = db.get(AnalysisRunExecutionRow, row.id)
        payload["attempt_no"] = execution.attempt_no if execution else None
        payload["retried_from"] = execution.retried_from_run_id if execution else None
        comparison = db.scalar(
            select(AnalysisRunComparisonRow).where(
                AnalysisRunComparisonRow.target_run_id == row.id
            )
        )
        payload["comparison_id"] = comparison.id if comparison else None
        payload["comparison_baseline_run_id"] = (
            comparison.baseline_run_id if comparison else None
        )
        payload["input_snapshot_available"] = bool(payload["input_documents"])
        if row.status == "completed":
            counts = dict(
                db.execute(
                    select(AnalysisRecordRow.kind, func.count())
                    .where(
                        AnalysisRecordRow.run_id == row.id,
                        AnalysisRecordRow.kind.in_(("clarification", "open_question")),
                    )
                    .group_by(AnalysisRecordRow.kind)
                ).all()
            )
            payload["clarification_count"] = counts.get("clarification", 0)
            payload["open_question_count"] = counts.get("open_question", 0)
        else:
            payload["clarification_count"] = None
            payload["open_question_count"] = None
    return payload


def serialize_document(row: DocumentRow, include_content: bool = True, db=None) -> dict:
    context = db.get(DocumentContextRow, row.id) if db is not None else None
    payload = {
        "id": row.id,
        "project_id": row.project_id,
        "name": row.name,
        "version": row.version,
        "active": row.active,
        "created_at": row.created_at,
        "document_role": context.document_role if context else DEFAULT_DOCUMENT_ROLE,
        "story_scope": context.story_scope if context else DEFAULT_STORY_SCOPE,
        "context_explicit": context is not None,
    }
    if include_content:
        payload["content"] = row.content
    return payload


def resolve_document_context(
    db,
    same_name: list[DocumentRow],
    replace_document_id: str | None,
    document_role: DocumentRole | None,
    story_scope: str | None,
) -> tuple[str, str]:
    """Inherit context from the replaced/latest version unless overridden."""
    base = None
    if replace_document_id:
        base = next((row for row in same_name if row.id == replace_document_id), None)
    if base is None and same_name:
        base = max(same_name, key=lambda row: row.version)
    inherited = db.get(DocumentContextRow, base.id) if base is not None else None
    role = (
        document_role.value
        if document_role is not None
        else inherited.document_role
        if inherited is not None
        else DEFAULT_DOCUMENT_ROLE
    )
    scope = (
        story_scope
        if story_scope is not None
        else inherited.story_scope
        if inherited is not None
        else DEFAULT_STORY_SCOPE
    )
    return role, scope


def dispatch_analysis(run_id: str) -> None:
    if settings.use_celery:
        from .tasks import analyze_project

        analyze_project.delay(run_id)
    else:
        Thread(target=execute_analysis, args=(run_id,), daemon=True).start()


def _normalize_idempotency_key(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise HTTPException(400, "Idempotency-Key 不能为空")
    if len(normalized) > 128 or re.fullmatch(r"[A-Za-z0-9._:-]+", normalized) is None:
        raise HTTPException(
            400,
            "Idempotency-Key 仅支持 1–128 位字母、数字及 . _ : -",
        )
    return normalized


def _idempotent_run(
    db, project_id: str, idempotency_key: str | None
) -> AnalysisRunRow | None:
    if idempotency_key is None:
        return None
    return db.scalar(
        select(AnalysisRunRow).where(
            AnalysisRunRow.project_id == project_id,
            AnalysisRunRow.idempotency_key == idempotency_key,
        )
    )


def _create_run_or_load_winner(
    db,
    run: AnalysisRunRow,
    idempotency_key: str | None,
    prepare: Callable[[AnalysisRunRow], object],
) -> tuple[AnalysisRunRow, bool]:
    """Create a run and snapshot, or load the concurrent winner for its key.

    The pre-insert lookup is only a fast path.  Correctness comes from the
    database unique index and this IntegrityError recovery path.
    """
    project_id = run.project_id
    try:
        db.add(run)
        db.flush()
        prepare(run)
        db.commit()
        return run, True
    except IntegrityError:
        db.rollback()
        winner = _idempotent_run(db, project_id, idempotency_key)
        if winner is None:
            raise
        return winner, False


def _accepted_run_payload(db, run: AnalysisRunRow, *, created: bool) -> dict:
    execution = db.get(AnalysisRunExecutionRow, run.id)
    return {
        "id": run.id,
        "project_id": run.project_id,
        "status": run.status,
        "retried_from": execution.retried_from_run_id if execution else None,
        "deduplicated": not created,
    }


def _require_idempotency_operation(
    db,
    run: AnalysisRunRow,
    *,
    retried_from_run_id: str | None,
    recheck_baseline_run_id: str | None = None,
) -> None:
    """Reject reuse of a project-scoped key for a different user intent."""
    execution = db.get(AnalysisRunExecutionRow, run.id)
    actual_source = execution.retried_from_run_id if execution else None
    comparison = db.scalar(
        select(AnalysisRunComparisonRow).where(
            AnalysisRunComparisonRow.target_run_id == run.id
        )
    )
    actual_baseline = comparison.baseline_run_id if comparison else None
    if (
        actual_source == retried_from_run_id
        and actual_baseline == recheck_baseline_run_id
    ):
        return
    raise HTTPException(
        409,
        detail={
            "code": "idempotency_key_conflict",
            "message": "这个 Idempotency-Key 已用于另一项分析操作，请为新操作生成新键",
        },
    )


def _mark_dispatch_failure(run_id: str) -> bool:
    """Persist a safe terminal state if scheduling fails before worker claim."""
    with SessionLocal() as db:
        changed = db.execute(
            update(AnalysisRunRow)
            .where(
                AnalysisRunRow.id == run_id,
                AnalysisRunRow.status == "queued",
            )
            .values(
                status="failed",
                error=DISPATCH_FAILED_ERROR,
                completed_at=utc_now_naive(),
            )
        ).rowcount
        if changed != 1:
            db.rollback()
            return False
        db.add(
            RunEventRow(
                run_id=run_id,
                stage="failed",
                progress=100,
                message="任务调度失败，输入快照已保留，可稍后重试",
            )
        )
        db.commit()
        return True


def _dispatch_created_run(run_id: str) -> None:
    try:
        dispatch_analysis(run_id)
    except Exception:
        marked_failed = _mark_dispatch_failure(run_id)
        if marked_failed:
            raise HTTPException(
                503,
                detail={
                    "code": "analysis_dispatch_failed",
                    "message": "任务暂未进入执行队列，请稍后重试",
                    "run_id": run_id,
                    "retryable": True,
                },
                headers={"Retry-After": "1"},
            ) from None
        # A worker may have claimed the run in the narrow interval after a
        # transport-ambiguous dispatch.  Do not overwrite that durable state.


def enforce_daily_model_budget(db, workspace_id: str | None = None) -> None:
    """Reject model-backed work after the local daily usage threshold.

    This is intentionally a single-database check, not a distributed quota
    reservation.  The per-run gate remains the hard fallback for concurrent
    local jobs.
    """
    model_requested = (
        settings.enable_model_extraction
        or settings.enable_evidence_investigator
        or settings.enable_issue_evidence_review
    ) and bool(settings.openai_api_key.strip())
    if not model_requested:
        return
    now = utc_now_naive()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    statement = (
        select(
            AnalysisRunRow.prompt_tokens,
            AnalysisRunRow.completion_tokens,
            AnalysisDiagnosticRow.payload,
        )
        .join(ProjectRow, ProjectRow.id == AnalysisRunRow.project_id)
        .outerjoin(
            AnalysisDiagnosticRow,
            AnalysisDiagnosticRow.run_id == AnalysisRunRow.id,
        )
        .where(
            AnalysisRunRow.created_at >= day_start,
            # Cancelled runs can contain an explicitly qualified lower-bound
            # Agent usage record. Those consumed tokens still count against
            # the local daily safety budget.
            AnalysisRunRow.status.in_(
                ("queued", "running", "completed", "failed", "cancelled")
            ),
        )
    )
    if workspace_id is not None:
        statement = statement.where(ProjectRow.workspace_id == workspace_id)
    usage_rows = db.execute(statement).all()
    daily_usage = sum(_conservative_run_token_debit(*row) for row in usage_rows)
    if settings.daily_token_budget <= 0 or daily_usage >= settings.daily_token_budget:
        seconds_to_reset = max(
            1, int(86400 - (now - day_start).total_seconds())
        )
        raise HTTPException(
            429,
            f"当日模型 Token 预算已用尽（{daily_usage}/{settings.daily_token_budget}）",
            headers={"Retry-After": str(seconds_to_reset)},
        )


def _conservative_run_token_debit(
    prompt_tokens: object,
    completion_tokens: object,
    diagnostic_payload: object,
) -> int:
    """Use reported usage plus any proven conservative review debit delta."""
    reported = sum(
        value if type(value) is int and value >= 0 else 0
        for value in (prompt_tokens, completion_tokens)
    )
    if not isinstance(diagnostic_payload, dict):
        return reported
    accounting = diagnostic_payload.get("usage_accounting")
    if not isinstance(accounting, dict):
        review = diagnostic_payload.get("ai_evidence_review")
        if isinstance(review, dict):
            nested = review.get("usage_accounting")
            accounting = nested if isinstance(nested, dict) else review
    if not isinstance(accounting, dict):
        return reported
    charged = accounting.get("charged_tokens")
    review_reported = sum(
        value if type(value) is int and value >= 0 else 0
        for value in (
            accounting.get("prompt_tokens"),
            accounting.get("completion_tokens"),
        )
    )
    if type(charged) is not int or charged < review_reported:
        return reported
    return reported + charged - review_reported


def prepare_document_version(
    db,
    project_id: str,
    name: str,
    replace_document_id: str | None,
    document_role: DocumentRole | None = None,
    story_scope: str | None = None,
) -> tuple[int, list[str], str, str]:
    # Serialize version allocation per project on PostgreSQL. The unique
    # logical-name/version and partial-active indexes remain authoritative and
    # also protect SQLite/tests where SELECT FOR UPDATE is advisory/no-op.
    locked_project = db.scalar(
        select(ProjectRow).where(ProjectRow.id == project_id).with_for_update()
    )
    if locked_project is None:
        raise HTTPException(404, "项目不存在")
    same_name = db.scalars(
        select(DocumentRow).where(
            DocumentRow.project_id == project_id,
            func.lower(DocumentRow.name) == name.lower(),
        )
    ).all()
    if replace_document_id:
        old = db.scalar(
            select(DocumentRow).where(
                DocumentRow.id == replace_document_id,
                DocumentRow.project_id == project_id,
            )
        )
        if not old:
            raise HTTPException(404, "待替换文档不存在")
        if old.name.lower() != name.lower():
            raise HTTPException(409, "替换文档必须保持同名；如需新文件请直接上传")
    resolved_role, resolved_scope = resolve_document_context(
        db,
        list(same_name),
        replace_document_id,
        document_role,
        story_scope,
    )
    version = max((row.version for row in same_name), default=0) + 1
    superseded = []
    for row in same_name:
        if row.active:
            row.active = False
            superseded.append(row.id)
    return version, superseded, resolved_role, resolved_scope


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "time": utc_now_naive().isoformat(),
        "model": {
            "configured": (
                settings.enable_model_extraction
                or settings.enable_evidence_investigator
                or settings.enable_issue_evidence_review
            )
            and bool(settings.openai_api_key.strip()),
            "thinking": safe_thinking_configuration(settings),
        },
        "runtime_provenance": safe_runtime_provenance(settings),
    }


def _provider_check_suggestions(category: str) -> list[str]:
    if category == "not_configured":
        return [
            "启用模型抽取、证据调查器或 AI 证据复核，并配置有效的 API 凭据后重试。"
        ]
    if category == "unauthorized":
        return [
            "检查 API 凭据是否有效、未过期，并确认凭据属于当前租户或项目。",
            "确认凭据已获得模型调用授权。",
        ]
    if category == "forbidden":
        return [
            "确认账户额度与计费状态可用。",
            "确认当前租户或项目有权访问所选模型。",
            "检查服务方的 IP 白名单、区域或网关访问策略。",
            "确认 API 凭据具有模型调用权限。",
        ]
    if category in {"rate_limit", "upstream_5xx"}:
        return ["服务已响应，请稍后重试；若持续失败，请检查额度和服务状态。"]
    if category in {"connect_timeout", "read_timeout", "transport"}:
        return ["检查到模型服务的网络连通性后重试。"]
    if category == "invalid_response":
        return ["模型服务已连通，但返回内容不符合最小 JSON 协议；请检查模型的 JSON 输出兼容性。"]
    if category == "success":
        return []
    return ["检查模型兼容性与访问权限后重试。"]


def _provider_check_metric(value: object) -> int | None:
    """Allowlist a bounded non-negative integer for the public preflight."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0 or value > 2_147_483_647:
        return None
    return value


def _provider_check_metrics(
    telemetry: object | None,
    *,
    fallback_elapsed_ms: object | None = None,
) -> dict[str, object]:
    elapsed_ms = _provider_check_metric(
        getattr(telemetry, "elapsed_ms", fallback_elapsed_ms)
    )
    prompt_tokens = _provider_check_metric(
        getattr(telemetry, "prompt_tokens", None)
    )
    completion_tokens = _provider_check_metric(
        getattr(telemetry, "completion_tokens", None)
    )
    return {
        "latency_ms": elapsed_ms,
        "token_usage": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": (
                prompt_tokens + completion_tokens
                if prompt_tokens is not None and completion_tokens is not None
                else None
            ),
        },
    }


@app.post("/api/v1/model/provider-check")
def check_model_provider(
    _context: AuthContext = Depends(require_csrf),
) -> dict:
    """Run an explicit, minimal provider preflight and return safe diagnostics."""
    provider = OpenAICompatibleProvider(settings)
    thinking = safe_thinking_configuration(settings)
    if not provider.configured:
        category = "not_configured"
        return {
            "status": category,
            "configured": False,
            "json_contract_ok": None,
            "reachable": False,
            "authorized": False,
            "category": category,
            "http_status": None,
            "request_id": None,
            "thinking": thinking,
            "suggestions": _provider_check_suggestions(category),
            **_provider_check_metrics(None),
        }

    try:
        result = provider.complete(
            "Return only valid JSON.",
            'Return {"status":"ok"}.',
        )
    except ProviderError as exc:
        category = exc.category
        http_status = exc.http_status
        reachable = http_status is not None
        authorized = False if category in {"unauthorized", "forbidden"} else None
        telemetry = getattr(exc, "telemetry", None)
        return {
            "status": category,
            "configured": True,
            "json_contract_ok": None,
            "reachable": reachable,
            "authorized": authorized,
            "category": category,
            "http_status": http_status,
            "request_id": sanitize_request_id(exc.request_id),
            "thinking": thinking,
            "suggestions": _provider_check_suggestions(category),
            **_provider_check_metrics(
                telemetry,
                fallback_elapsed_ms=getattr(exc, "elapsed_ms", None),
            ),
        }

    category = "success"
    try:
        contract_ok = json.loads(result.text) == {"status": "ok"}
    except (json.JSONDecodeError, TypeError):
        contract_ok = False
    if not contract_ok:
        category = "invalid_response"
    telemetry = getattr(result, "telemetry", None)
    return {
        "status": category,
        "configured": True,
        "json_contract_ok": contract_ok,
        "reachable": True,
        "authorized": True,
        "category": category,
        "http_status": getattr(telemetry, "http_status", 200),
        "request_id": sanitize_request_id(
            getattr(telemetry, "request_id", None)
        ),
        "thinking": thinking,
        "suggestions": _provider_check_suggestions(category),
        **_provider_check_metrics(telemetry),
    }


@app.post("/api/v1/projects", status_code=201)
def create_project(
    payload: ProjectIn,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    with SessionLocal() as db:
        row = ProjectRow(
            workspace_id=context.workspace_id,
            name=payload.name,
            description=payload.description,
        )
        db.add(row); db.commit()
        return {"id": row.id, "name": row.name, "description": row.description, "created_at": row.created_at}


@app.get("/api/v1/projects")
def list_projects(
    context: AuthContext = Depends(get_auth_context),
) -> list[dict]:
    with SessionLocal() as db:
        projects = db.scalars(
            select(ProjectRow)
            .where(ProjectRow.workspace_id == context.workspace_id)
            .order_by(ProjectRow.created_at.desc())
        ).all()
        result = []
        for project in projects:
            active_document_count = db.scalar(
                select(func.count()).select_from(DocumentRow).where(
                    DocumentRow.project_id == project.id, DocumentRow.active.is_(True)
                )
            ) or 0
            latest_run = db.scalar(
                select(AnalysisRunRow)
                .where(AnalysisRunRow.project_id == project.id)
                .order_by(AnalysisRunRow.created_at.desc())
                .limit(1)
            )
            result.append({
                "id": project.id,
                "name": project.name,
                "description": project.description,
                "created_at": project.created_at,
                "active_document_count": active_document_count,
                "latest_run": serialize_run(latest_run, db) if latest_run else None,
            })
        return result


@app.get("/api/v1/projects/{project_id}")
def get_project(
    project_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        project = _project_in_workspace(db, project_id, context.workspace_id)
        if not project:
            raise HTTPException(404, "项目不存在")
        documents = db.scalars(
            select(DocumentRow)
            .where(DocumentRow.project_id == project_id, DocumentRow.active.is_(True))
            .order_by(DocumentRow.created_at)
        ).all()
        return {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "documents": [serialize_document(d, db=db) for d in documents],
        }


@app.get("/api/v1/projects/{project_id}/documents")
def list_documents(
    project_id: str,
    include_history: bool = False,
    context: AuthContext = Depends(get_auth_context),
) -> list[dict]:
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        statement = select(DocumentRow).where(DocumentRow.project_id == project_id)
        if not include_history:
            statement = statement.where(DocumentRow.active.is_(True))
        rows = db.scalars(statement.order_by(DocumentRow.name, DocumentRow.version.desc())).all()
        # Version pickers and project history only need metadata. Returning every
        # historical body makes the browser download all revisions of a large
        # story before it can render the workspace.
        return [serialize_document(row, include_content=False, db=db) for row in rows]


@app.get("/api/v1/projects/{project_id}/documents/diff")
def compare_document_versions(
    project_id: str,
    from_document_id: str,
    to_document_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    """Compare two stored versions of one same-named document.

    This endpoint is deliberately local and deterministic: it never invokes the
    extraction provider and therefore cannot consume model tokens.
    """
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        old = db.scalar(
            select(DocumentRow)
            .join(ProjectRow, ProjectRow.id == DocumentRow.project_id)
            .where(
                DocumentRow.id == from_document_id,
                DocumentRow.project_id == project_id,
                ProjectRow.workspace_id == context.workspace_id,
            )
        )
        new = db.scalar(
            select(DocumentRow)
            .join(ProjectRow, ProjectRow.id == DocumentRow.project_id)
            .where(
                DocumentRow.id == to_document_id,
                DocumentRow.project_id == project_id,
                ProjectRow.workspace_id == context.workspace_id,
            )
        )
        if not old:
            raise HTTPException(404, "起始文档版本不存在于当前项目")
        if not new:
            raise HTTPException(404, "目标文档版本不存在于当前项目")
        if old.id == new.id:
            raise HTTPException(409, "请选择两个不同版本进行比较")
        if old.name.casefold() != new.name.casefold():
            raise HTTPException(409, "只能比较当前项目内同名文档的不同版本")

        diff = build_document_diff(
            old.content,
            new.content,
            max_lines=settings.diff_max_lines_per_version,
            max_chars=settings.diff_max_chars_per_version,
            max_output_lines=settings.diff_max_output_lines,
        )
        return {
            "from_document": {
                **serialize_document(old, include_content=False, db=db),
                "char_count": len(old.content),
                "line_count": len(old.content.splitlines()),
            },
            "to_document": {
                **serialize_document(new, include_content=False, db=db),
                "char_count": len(new.content),
                "line_count": len(new.content.splitlines()),
            },
            **diff,
        }


@app.post("/api/v1/projects/{project_id}/documents/text", status_code=201)
def create_text_document(
    project_id: str,
    payload: TextDocumentIn,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    if not payload.name.lower().endswith((".md", ".txt", ".json")):
        raise HTTPException(415, "名称必须以 .md、.txt 或 .json 结尾")
    if len(payload.content.encode("utf-8")) > settings.max_upload_bytes:
        raise HTTPException(413, "文本超过上传限制")
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        version, superseded, document_role, story_scope = prepare_document_version(
            db,
            project_id,
            payload.name,
            payload.replace_document_id,
            payload.document_role,
            payload.story_scope,
        )
        row = DocumentRow(project_id=project_id, name=payload.name, content=payload.content, version=version)
        try:
            db.add(row)
            db.flush()
            db.add(
                DocumentContextRow(
                    document_id=row.id,
                    document_role=document_role,
                    story_scope=story_scope,
                )
            )
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                409,
                detail={
                    "code": "document_version_conflict",
                    "message": "同名文档版本正在被更新，请刷新后重试",
                },
            ) from None
        return {**serialize_document(row, db=db), "superseded_document_ids": superseded}


@app.post("/api/v1/demo", status_code=201)
def create_demo(
    context: AuthContext = Depends(require_csrf),
) -> dict:
    data_dir = Path(__file__).resolve().parents[1] / "data" / "demo-natural"
    files = [
        (data_dir / "world.md", DocumentRole.canon),
        (data_dir / "chapter-01.md", DocumentRole.chapter),
    ]
    if not all(path.exists() for path, _ in files):
        raise HTTPException(500, "演示数据缺失")
    with SessionLocal() as db:
        project = ProjectRow(
            workspace_id=context.workspace_id,
            name="潮汐之门 · 自然文本体验",
            description="无需 API Key 的中文自然文本基线",
        )
        db.add(project); db.flush()
        for path, role in files:
            document = DocumentRow(project_id=project.id, name=path.name, content=path.read_text(encoding="utf-8"))
            db.add(document)
            db.flush()
            db.add(
                DocumentContextRow(
                    document_id=document.id,
                    document_role=role.value,
                    story_scope=DEFAULT_STORY_SCOPE,
                )
            )
        db.commit()
        return {"id": project.id, "name": project.name, "document_count": len(files)}


@app.post("/api/v1/demo/advanced", status_code=201)
def create_advanced_demo(
    context: AuthContext = Depends(require_csrf),
) -> dict:
    """Create the original multi-document acceptance scenario."""
    data_dir = Path(__file__).resolve().parents[1] / "data" / "advanced"
    files = [
        (data_dir / "world.md", DocumentRole.canon),
        (data_dir / "chapter-01.md", DocumentRole.chapter),
        (data_dir / "chapter-02.md", DocumentRole.chapter),
    ]
    if not all(path.exists() for path, _ in files):
        raise HTTPException(500, "复杂演示数据缺失")
    with SessionLocal() as db:
        project = ProjectRow(
            workspace_id=context.workspace_id,
            name="静默海域 · 复杂多章节验收",
            description="原创三文档场景，覆盖时间、地点、知识、物品与世界规则",
        )
        db.add(project); db.flush()
        for path, role in files:
            document = DocumentRow(project_id=project.id, name=path.name, content=path.read_text(encoding="utf-8"))
            db.add(document)
            db.flush()
            db.add(
                DocumentContextRow(
                    document_id=document.id,
                    document_role=role.value,
                    story_scope=DEFAULT_STORY_SCOPE,
                )
            )
        db.commit()
        return {"id": project.id, "name": project.name, "document_count": len(files)}


@app.post("/api/v1/projects/{project_id}/documents", status_code=201)
async def upload_document(
    project_id: str,
    file: UploadFile = File(...),
    replace_document_id: str | None = Form(None),
    document_role: DocumentRole | None = Form(None),
    story_scope: str | None = Form(
        None,
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9_\-\u4e00-\u9fff]+$",
    ),
    context: AuthContext = Depends(require_csrf),
) -> dict:
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "文件超过上传限制")
    if not file.filename or not file.filename.lower().endswith((".md", ".txt", ".json", ".docx")):
        raise HTTPException(415, "仅支持 Markdown、TXT、JSON 与标准 DOCX")
    if file.filename.lower().endswith(".docx"):
        try:
            content = extract_docx_text(data, max_text_bytes=settings.max_upload_bytes)
        except DocxImportError as exc:
            raise HTTPException(exc.status_code, str(exc)) from None
    else:
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(400, "Markdown、TXT 与 JSON 文件必须为 UTF-8 编码") from None
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        version, superseded, resolved_role, resolved_scope = prepare_document_version(
            db,
            project_id,
            file.filename,
            replace_document_id,
            document_role,
            story_scope,
        )
        row = DocumentRow(project_id=project_id, name=file.filename, content=content, version=version)
        try:
            db.add(row)
            db.flush()
            db.add(
                DocumentContextRow(
                    document_id=row.id,
                    document_role=resolved_role,
                    story_scope=resolved_scope,
                )
            )
            db.commit()
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                409,
                detail={
                    "code": "document_version_conflict",
                    "message": "同名文档版本正在被更新，请刷新后重试",
                },
            ) from None
        return {**serialize_document(row, db=db), "superseded_document_ids": superseded}


@app.post("/api/v1/projects/{project_id}/analysis-runs", status_code=202)
def start_analysis(
    project_id: str,
    idempotency_key_header: Annotated[
        str | None, Header(alias="Idempotency-Key", max_length=128)
    ] = None,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    idempotency_key = _normalize_idempotency_key(idempotency_key_header)
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        existing = _idempotent_run(db, project_id, idempotency_key)
        if existing is not None:
            _require_idempotency_operation(
                db, existing, retried_from_run_id=None
            )
            return _accepted_run_payload(db, existing, created=False)
        documents = db.scalars(
            select(DocumentRow)
            .where(DocumentRow.project_id == project_id, DocumentRow.active.is_(True))
            .order_by(DocumentRow.created_at, DocumentRow.id)
        ).all()
        if not documents:
            raise HTTPException(400, "项目没有可分析文档")
        enforce_daily_model_budget(db, context.workspace_id)
        run = AnalysisRunRow(
            project_id=project_id,
            requested_by_user_id=context.user_id,
            idempotency_key=idempotency_key,
        )
        run, created = _create_run_or_load_winner(
            db,
            run,
            idempotency_key,
            lambda created_run: capture_run_inputs(db, created_run, list(documents)),
        )
        _require_idempotency_operation(db, run, retried_from_run_id=None)
        payload = _accepted_run_payload(db, run, created=created)
        run_id = run.id
    if created:
        _dispatch_created_run(run_id)
    return payload


@app.get("/api/v1/analysis-runs/{run_id}")
def get_run(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        row = _run_in_workspace(db, run_id, context.workspace_id)
        if not row: raise HTTPException(404, "分析任务不存在")
        return serialize_run(row, db)


@app.get("/api/v1/projects/{project_id}/analysis-runs")
def list_analysis_runs(
    project_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> list[dict]:
    with SessionLocal() as db:
        if not _project_in_workspace(db, project_id, context.workspace_id):
            raise HTTPException(404, "项目不存在")
        rows = db.scalars(
            select(AnalysisRunRow)
            .where(AnalysisRunRow.project_id == project_id)
            .order_by(
                AnalysisRunRow.created_at.desc(),
                AnalysisRunRow.id.desc(),
            )
        ).all()
        return [serialize_run(row, db) for row in rows]


@app.post("/api/v1/analysis-runs/{run_id}/cancel", status_code=202)
def cancel_run(
    run_id: str,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    with SessionLocal() as db:
        row = _run_in_workspace(db, run_id, context.workspace_id)
        if not row: raise HTTPException(404, "分析任务不存在")
        if row.status == "cancelled" and row.cancel_requested:
            return {"id": row.id, "cancel_requested": True, "already_requested": True}
        if row.status in {"completed", "failed", "cancelled"}:
            raise HTTPException(409, f"终态任务不能取消：{row.status}")
        cancelled_before_start = db.execute(
            update(AnalysisRunRow)
            .where(
                AnalysisRunRow.id == run_id,
                AnalysisRunRow.status == "queued",
                AnalysisRunRow.cancel_requested.is_(False),
            )
            .values(
                status="cancelled",
                cancel_requested=True,
                completed_at=utc_now_naive(),
            )
        ).rowcount
        if cancelled_before_start == 1:
            db.add(
                RunEventRow(
                    run_id=run_id,
                    stage="cancelled",
                    progress=100,
                    message="任务在工作进程开始前已取消",
                )
            )
            db.commit()
            return {"id": row.id, "cancel_requested": True, "already_requested": False}
        changed = db.execute(
            update(AnalysisRunRow)
            .where(
                AnalysisRunRow.id == run_id,
                AnalysisRunRow.status.in_(("queued", "running")),
                AnalysisRunRow.cancel_requested.is_(False),
            )
            .values(cancel_requested=True)
        ).rowcount
        if changed != 1:
            db.rollback()
            db.refresh(row)
            if row.status == "cancelled" and row.cancel_requested:
                return {"id": row.id, "cancel_requested": True, "already_requested": True}
            if row.status in {"completed", "failed", "cancelled"}:
                raise HTTPException(409, f"终态任务不能取消：{row.status}")
            return {"id": row.id, "cancel_requested": True, "already_requested": True}
        db.commit()
        return {"id": row.id, "cancel_requested": True, "already_requested": False}


@app.post("/api/v1/analysis-runs/{run_id}/retry", status_code=202)
def retry_run(
    run_id: str,
    idempotency_key_header: Annotated[
        str | None, Header(alias="Idempotency-Key", max_length=128)
    ] = None,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    idempotency_key = _normalize_idempotency_key(idempotency_key_header)
    with SessionLocal() as db:
        old = _run_in_workspace(db, run_id, context.workspace_id)
        if not old: raise HTTPException(404, "分析任务不存在")
        existing = _idempotent_run(db, old.project_id, idempotency_key)
        if existing is not None:
            _require_idempotency_operation(
                db, existing, retried_from_run_id=run_id
            )
            return _accepted_run_payload(db, existing, created=False)
        if old.status not in {"failed", "cancelled"}:
            raise HTTPException(409, "仅失败或已取消任务可以重试")
        snapshot_count = db.scalar(
            select(func.count())
            .select_from(AnalysisRunInputRow)
            .where(AnalysisRunInputRow.run_id == run_id)
        )
        if not snapshot_count:
            raise HTTPException(409, MISSING_SNAPSHOT_ERROR)
        enforce_daily_model_budget(db, context.workspace_id)
        row = AnalysisRunRow(
            project_id=old.project_id,
            requested_by_user_id=context.user_id,
            idempotency_key=idempotency_key,
        )
        row, created = _create_run_or_load_winner(
            db,
            row,
            idempotency_key,
            lambda created_run: copy_run_inputs(db, old.id, created_run),
        )
        _require_idempotency_operation(
            db, row, retried_from_run_id=run_id
        )
        payload = _accepted_run_payload(db, row, created=created)
        new_id = row.id
    if created:
        _dispatch_created_run(new_id)
    return payload


@app.post("/api/v1/analysis-runs/{baseline_run_id}/rechecks", status_code=202)
def start_recheck(
    baseline_run_id: str,
    idempotency_key_header: Annotated[
        str | None, Header(alias="Idempotency-Key", max_length=128)
    ] = None,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    """Analyze the project's current revision against one frozen baseline."""
    idempotency_key = _normalize_idempotency_key(idempotency_key_header)
    with SessionLocal() as db:
        baseline = _run_in_workspace(db, baseline_run_id, context.workspace_id)
        if baseline is None:
            raise HTTPException(404, "基准分析任务不存在")
        existing = _idempotent_run(db, baseline.project_id, idempotency_key)
        if existing is not None:
            _require_idempotency_operation(
                db,
                existing,
                retried_from_run_id=None,
                recheck_baseline_run_id=baseline_run_id,
            )
            comparison = db.scalar(
                select(AnalysisRunComparisonRow).where(
                    AnalysisRunComparisonRow.target_run_id == existing.id
                )
            )
            payload = _accepted_run_payload(db, existing, created=False)
            return {
                **payload,
                "baseline_run_id": baseline_run_id,
                "comparison_id": comparison.id,
                "comparison_status": comparison.status,
            }
        if baseline.status != "completed":
            raise HTTPException(409, "只有已完成的分析任务可以作为复检基准")
        baseline_inputs = list(
            db.scalars(
                select(AnalysisRunInputRow)
                .where(AnalysisRunInputRow.run_id == baseline_run_id)
                .order_by(AnalysisRunInputRow.ordinal)
            ).all()
        )
        if not baseline_inputs:
            raise HTTPException(409, MISSING_SNAPSHOT_ERROR)
        documents = list(
            db.scalars(
                select(DocumentRow)
                .where(
                    DocumentRow.project_id == baseline.project_id,
                    DocumentRow.active.is_(True),
                )
                .order_by(DocumentRow.created_at, DocumentRow.id)
            ).all()
        )
        if not documents:
            raise HTTPException(400, "项目没有可复检文档")
        baseline_signature = sorted(
            (
                row.document_name.casefold(),
                row.document_version,
                row.content_sha256,
            )
            for row in baseline_inputs
        )
        current_signature = sorted(
            (
                row.name.casefold(),
                row.version,
                sha256(row.content.encode("utf-8")).hexdigest(),
            )
            for row in documents
        )
        if baseline_signature == current_signature:
            raise HTTPException(
                409,
                detail={
                    "code": "no_document_changes",
                    "message": "当前活动文档与基准运行的冻结输入相同，请先保存新版本",
                },
            )
        enforce_daily_model_budget(db, context.workspace_id)
        row = AnalysisRunRow(
            project_id=baseline.project_id,
            requested_by_user_id=context.user_id,
            idempotency_key=idempotency_key,
        )

        def prepare_recheck(created_run: AnalysisRunRow) -> None:
            capture_run_inputs(db, created_run, documents)
            db.flush()
            target_inputs = list(
                db.scalars(
                    select(AnalysisRunInputRow)
                    .where(AnalysisRunInputRow.run_id == created_run.id)
                    .order_by(AnalysisRunInputRow.ordinal)
                ).all()
            )
            db.add(
                AnalysisRunComparisonRow(
                    project_id=baseline.project_id,
                    baseline_run_id=baseline_run_id,
                    target_run_id=created_run.id,
                    matcher_version=MATCHER_VERSION,
                    status="pending",
                    summary={},
                    provenance={
                        "matcher_version": MATCHER_VERSION,
                        "input_diff": build_input_diff(
                            baseline_inputs, target_inputs
                        ),
                        "semantic_equivalence_guaranteed": False,
                    },
                )
            )

        row, created = _create_run_or_load_winner(
            db, row, idempotency_key, prepare_recheck
        )
        _require_idempotency_operation(
            db,
            row,
            retried_from_run_id=None,
            recheck_baseline_run_id=baseline_run_id,
        )
        comparison = db.scalar(
            select(AnalysisRunComparisonRow).where(
                AnalysisRunComparisonRow.target_run_id == row.id
            )
        )
        payload = {
            **_accepted_run_payload(db, row, created=created),
            "baseline_run_id": baseline_run_id,
            "comparison_id": comparison.id,
            "comparison_status": comparison.status,
        }
        target_run_id = row.id
    if created:
        _dispatch_created_run(target_run_id)
    return payload


def _serialize_issue(row: IssueRow | None) -> dict | None:
    if row is None:
        return None
    return {
        "id": row.id,
        "category": row.category,
        "severity": row.severity,
        "confidence": row.confidence,
        "title": row.title,
        "explanation": row.explanation,
        "evidence": row.evidence,
        "suggestion": row.suggestion,
        "metadata": row.extra,
    }


def _comparison_feedback_snapshot(provenance: object) -> dict | None:
    if not isinstance(provenance, dict):
        return None
    snapshot = provenance.get("baseline_feedback_snapshot")
    if not isinstance(snapshot, dict):
        return None
    identifier = snapshot.get("id")
    label = snapshot.get("label")
    comment = snapshot.get("comment")
    created_at = snapshot.get("created_at")
    if (
        not isinstance(identifier, str)
        or label not in {"accepted", "false_positive", "resolved"}
        or not isinstance(comment, str)
        or not isinstance(created_at, str)
    ):
        return None
    return {
        "id": identifier,
        "label": label,
        "comment": comment,
        "created_at": created_at,
    }


@app.get("/api/v1/analysis-runs/{target_run_id}/comparison")
def get_run_comparison(
    target_run_id: str,
    outcome: ComparisonOutcome | None = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        target = _run_in_workspace(db, target_run_id, context.workspace_id)
        if target is None:
            raise HTTPException(404, "分析任务不存在")
        comparison = db.scalar(
            select(AnalysisRunComparisonRow).where(
                AnalysisRunComparisonRow.target_run_id == target_run_id
            )
        )
        if comparison is None:
            raise HTTPException(404, "该分析任务不是复检任务")
        baseline = _run_in_workspace(
            db, comparison.baseline_run_id, context.workspace_id
        )
        if (
            baseline is None
            or baseline.project_id != target.project_id
            or comparison.project_id != target.project_id
        ):
            # Fail closed if persisted lineage was corrupted or manually
            # inserted across projects; never follow it into another workspace.
            raise HTTPException(404, "复检谱系不存在")
        if target.status == "completed" and comparison.status != "ready":
            try:
                with db.begin_nested():
                    comparison = materialize_run_comparison(db, target_run_id)
            except Exception:
                with db.begin_nested():
                    comparison = mark_comparison_unverifiable(db, target_run_id)
            db.commit()

        effective_status = (
            "ready"
            if comparison.status == "ready"
            else target.status
            if target.status in {"failed", "cancelled"}
            else "pending"
        )
        payload = {
            "id": comparison.id,
            "project_id": comparison.project_id,
            "baseline_run_id": comparison.baseline_run_id,
            "target_run_id": comparison.target_run_id,
            "status": effective_status,
            "target_run_status": target.status,
            "matcher": {
                "version": comparison.matcher_version,
                "kind": "deterministic_heuristic",
                "semantic_equivalence_guaranteed": False,
            },
            "summary": comparison.summary if comparison.status == "ready" else None,
            "provenance": comparison.provenance,
            "created_at": comparison.created_at,
            "completed_at": comparison.completed_at,
            "page": {
                "outcome": outcome,
                "limit": limit,
                "offset": offset,
                "returned": 0,
                "total": 0,
                "has_more": False,
            },
            "items": [],
        }
        if comparison.status != "ready":
            return payload
        item_filter = [IssueComparisonItemRow.comparison_id == comparison.id]
        if outcome is not None:
            item_filter.append(IssueComparisonItemRow.outcome == outcome)
        total_items = db.scalar(
            select(func.count()).select_from(IssueComparisonItemRow).where(*item_filter)
        ) or 0
        rows = list(
            db.scalars(
                select(IssueComparisonItemRow)
                .where(*item_filter)
                .order_by(IssueComparisonItemRow.outcome, IssueComparisonItemRow.id)
                .offset(offset)
                .limit(limit)
            ).all()
        )
        issue_ids = {
            issue_id
            for row in rows
            for issue_id in (row.baseline_issue_id, row.target_issue_id)
            if issue_id is not None
        }
        issues = (
            {
                row.id: row
                for row in db.scalars(
                    select(IssueRow).where(
                        IssueRow.id.in_(issue_ids),
                        IssueRow.run_id.in_(
                            (comparison.baseline_run_id, comparison.target_run_id)
                        ),
                    )
                ).all()
            }
            if issue_ids
            else {}
        )
        baseline_issues = {
            issue_id: issue
            for issue_id, issue in issues.items()
            if issue.run_id == comparison.baseline_run_id
        }
        target_issues = {
            issue_id: issue
            for issue_id, issue in issues.items()
            if issue.run_id == comparison.target_run_id
        }
        payload["items"] = [
            {
                "id": row.id,
                "outcome": row.outcome,
                "baseline_issue": _serialize_issue(
                    baseline_issues.get(row.baseline_issue_id)
                ),
                "target_issue": _serialize_issue(
                    target_issues.get(row.target_issue_id)
                ),
                "match_method": row.match_method,
                "match_score": row.match_score,
                "provenance": row.provenance,
                "baseline_latest_feedback": _comparison_feedback_snapshot(
                    row.provenance
                ),
            }
            for row in rows
        ]
        payload["page"]["returned"] = len(rows)
        payload["page"]["total"] = total_items
        payload["page"]["has_more"] = offset + len(rows) < total_items
        return payload


@app.get("/api/v1/analysis-runs/{run_id}/issues")
def get_issues(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> list[dict]:
    with SessionLocal() as db:
        if not _run_in_workspace(db, run_id, context.workspace_id):
            raise HTTPException(404, "分析任务不存在")
        rows = db.scalars(select(IssueRow).where(IssueRow.run_id == run_id)).all()
        return [_serialize_issue(row) for row in rows]


@app.get("/api/v1/analysis-runs/{run_id}/records")
def get_records(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        run = _run_in_workspace(db, run_id, context.workspace_id)
        if not run:
            raise HTTPException(404, "分析任务不存在")
        rows = list(db.scalars(
            select(AnalysisRecordRow)
            .where(AnalysisRecordRow.run_id == run_id)
        ).all())
        rows.sort(key=record_sort_key)
        warnings = [
            row.message
            for row in db.scalars(
                select(RunEventRow).where(
                    RunEventRow.run_id == run_id, RunEventRow.stage == "warning"
                )
            ).all()
        ]
        records = [
            {"id": row.id, "kind": row.kind, "attrs": row.attrs, "evidence": row.evidence}
            for row in rows
        ]
        return {"records": records, "warnings": warnings, "record_count": len(records)}


@app.get(
    "/api/v1/analysis-runs/{run_id}/clarifications",
    response_model=list[SemanticReviewItemOut],
)
def get_clarifications(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> list[SemanticReviewItemOut]:
    """Return only reviewable questions/gaps, never arbitrary record attrs."""
    with SessionLocal() as db:
        run = _run_in_workspace(db, run_id, context.workspace_id)
        if not run:
            raise HTTPException(404, "分析任务不存在")
        if run.status != "completed":
            raise HTTPException(409, f"仅已完成任务可读取待澄清结果：{run.status}")
        rows = db.scalars(
            select(AnalysisRecordRow)
            .where(
                AnalysisRecordRow.run_id == run_id,
                AnalysisRecordRow.kind.in_(("clarification", "open_question")),
            )
            .order_by(AnalysisRecordRow.id)
        ).all()
        result: list[SemanticReviewItemOut] = []
        for row in rows:
            attrs = row.attrs or {}
            is_question = row.kind == "open_question"
            result.append(
                SemanticReviewItemOut(
                    id=row.id,
                    kind=row.kind,
                    category=None if is_question else attrs.get("category"),
                    modality=attrs.get(
                        "modality",
                        SemanticModality.interrogative.value
                        if is_question
                        else SemanticModality.uncertain.value,
                    ),
                    source_scope=attrs.get("source_scope", SourceScope.unknown.value),
                    certainty=attrs.get("certainty", CertaintyLevel.unknown.value),
                    text=attrs.get("question" if is_question else "summary", ""),
                    evidence=EvidenceSpan.model_validate(row.evidence),
                )
            )
        return result


def _completed_visualization_rows(db, run_id: str, workspace_id: str):
    run = _run_in_workspace(db, run_id, workspace_id)
    if not run:
        raise HTTPException(404, "分析任务不存在")
    if run.status != "completed":
        raise HTTPException(409, f"仅已完成任务可生成可视化：{run.status}")
    records = list(db.scalars(
        select(AnalysisRecordRow).where(AnalysisRecordRow.run_id == run_id)
    ).all())
    issues = list(db.scalars(
        select(IssueRow).where(IssueRow.run_id == run_id)
    ).all())
    return records, issues


@app.get("/api/v1/analysis-runs/{run_id}/graph", response_model=GraphResponse)
def get_graph(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> GraphResponse:
    with SessionLocal() as db:
        records, issues = _completed_visualization_rows(
            db, run_id, context.workspace_id
        )
        return project_graph(run_id, records, issues)


@app.get("/api/v1/analysis-runs/{run_id}/timeline", response_model=TimelineResponse)
def get_timeline(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> TimelineResponse:
    with SessionLocal() as db:
        records, issues = _completed_visualization_rows(
            db, run_id, context.workspace_id
        )
        return project_timeline(run_id, records, issues)


@app.get("/api/v1/analysis-runs/{run_id}/diagnostics")
def get_diagnostics(
    run_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        if not _run_in_workspace(db, run_id, context.workspace_id):
            raise HTTPException(404, "分析任务不存在")
        row = db.get(AnalysisDiagnosticRow, run_id)
        return row.payload if row else {
            "chunking": {"total_chunks": 0, "documents": []},
            "aliases": {"declaration_count": 0, "trace_count": 0, "traces": []},
            "retrieval": {"candidate_count": 0, "consumed_count": 0, "traces": []},
        }


@app.get("/api/v1/analysis-runs/{run_id}/events")
async def stream_events(
    run_id: str,
    last_event_id: int = 0,
    last_event_id_header: Annotated[int | None, Header(alias="Last-Event-ID")] = None,
    context: AuthContext = Depends(get_auth_context),
):
    # Authorize before returning StreamingResponse so an unrelated run id is a
    # normal 404 and never establishes an SSE connection.
    with SessionLocal() as db:
        if not _run_in_workspace(db, run_id, context.workspace_id):
            raise HTTPException(404, "分析任务不存在")

    async def generate():
        cursor = max(last_event_id, last_event_id_header or 0)
        while True:
            with SessionLocal() as db:
                run = db.get(AnalysisRunRow, run_id)
                if not run:
                    yield "event: error\ndata: {\"message\":\"run not found\"}\n\n"; return
                rows = db.scalars(select(RunEventRow).where(RunEventRow.run_id == run_id, RunEventRow.id > cursor).order_by(RunEventRow.id)).all()
                for row in rows:
                    cursor = row.id
                    yield f"id: {row.id}\nevent: progress\ndata: {json.dumps({'stage': row.stage, 'progress': row.progress, 'message': row.message}, ensure_ascii=False)}\n\n"
                terminal = run.status in {"completed", "failed", "cancelled"}
                terminal_payload = {
                    "status": run.status,
                    "error": safe_persisted_analysis_error(run.error),
                }
            if terminal:
                yield f"event: terminal\ndata: {json.dumps(terminal_payload, ensure_ascii=False)}\n\n"
                return
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.35)
    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/v1/issues/{issue_id}/feedback", status_code=201)
def feedback(
    issue_id: str,
    payload: FeedbackIn,
    response: Response,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    if payload.label not in {"accepted", "false_positive", "resolved"}:
        raise HTTPException(422, "label 必须是 accepted、false_positive 或 resolved")
    with SessionLocal() as db:
        if not _issue_in_workspace(db, issue_id, context.workspace_id):
            raise HTTPException(404, "问题不存在")
        latest = db.scalar(
            select(FeedbackRow)
            .where(FeedbackRow.issue_id == issue_id)
            .order_by(FeedbackRow.created_at.desc())
            .limit(1)
        )
        if latest and latest.label == payload.label and latest.comment == payload.comment:
            response.status_code = 200
            count = db.scalar(
                select(func.count()).select_from(FeedbackRow).where(FeedbackRow.issue_id == issue_id)
            ) or 0
            return {
                "id": latest.id, "issue_id": issue_id, "label": latest.label,
                "comment": latest.comment, "created_at": latest.created_at,
                "history_count": count, "duplicate_ignored": True,
            }
        row = FeedbackRow(
            issue_id=issue_id,
            created_by_user_id=context.user_id,
            label=payload.label,
            comment=payload.comment,
        )
        db.add(row); db.commit()
        count = db.scalar(
            select(func.count()).select_from(FeedbackRow).where(FeedbackRow.issue_id == issue_id)
        ) or 0
        return {
            "id": row.id, "issue_id": issue_id, "label": row.label,
            "comment": row.comment, "created_at": row.created_at,
            "history_count": count, "duplicate_ignored": False,
        }


@app.get("/api/v1/issues/{issue_id}/feedback")
def feedback_history(
    issue_id: str,
    context: AuthContext = Depends(get_auth_context),
) -> dict:
    with SessionLocal() as db:
        if not _issue_in_workspace(db, issue_id, context.workspace_id):
            raise HTTPException(404, "问题不存在")
        rows = db.scalars(
            select(FeedbackRow)
            .where(FeedbackRow.issue_id == issue_id)
            .order_by(FeedbackRow.created_at.desc())
        ).all()
        history = [
            {"id": row.id, "label": row.label, "comment": row.comment, "created_at": row.created_at}
            for row in rows
        ]
        return {"issue_id": issue_id, "latest": history[0] if history else None, "history": history}


@app.get("/api/v1/evaluations/{evaluation_id}")
def evaluation(
    evaluation_id: str,
    _context: AuthContext = Depends(get_auth_context),
) -> dict:
    if evaluation_id not in {"baseline", "latest"}: raise HTTPException(404, "仅内置 baseline/latest 评测")
    return {
        "benchmark_kind": "rule-engine synthetic directive regression",
        "natural_language_evaluation": False,
        "warning": "显式 @directive 回归只验证规则接线，不代表自然文本准确率。",
        "metrics": run_evaluation().model_dump(),
    }


@app.get("/metrics")
def metrics():
    try:
        database_metrics = render_analysis_metrics(SessionLocal)
    except AnalysisMetricsUnavailable:
        body = (
            "# HELP loreguard_analysis_metrics_database_available "
            "Whether this analysis metrics scrape was derived from the database.\n"
            "# TYPE loreguard_analysis_metrics_database_available gauge\n"
            "loreguard_analysis_metrics_database_available 0\n"
        )
        return Response(
            content=body,
            status_code=503,
            media_type=CONTENT_TYPE_LATEST,
        )
    return Response(
        content=generate_latest() + database_metrics,
        media_type=CONTENT_TYPE_LATEST,
    )


# Production build can be experienced with one Python process. API routes are
# registered first, then the single-page app handles every remaining path.
frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", SpaStaticFiles(directory=frontend_dist, html=True), name="web")
