from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Thread
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from sqlalchemy import func, select

from .config import get_settings
from .db import AnalysisDiagnosticRow, AnalysisRecordRow, AnalysisRunRow, DocumentRow, FeedbackRow, IssueRow, ProjectRow, RunEventRow, SessionLocal, init_db
from .document_diff import build_document_diff
from .domain import GraphResponse, TimelineResponse
from .evaluation import run_evaluation
from .projections import project_graph, project_timeline, record_sort_key
from .rate_limit import SlidingWindowLimiter, WriteRateLimitMiddleware
from .service import execute_analysis
from .time_utils import utc_now_naive

RUNS = Counter("loreguard_analysis_runs_total", "Analysis runs", ["status"])
LATENCY = Histogram("loreguard_analysis_seconds", "Analysis duration")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(title="LoreGuard API", version="0.1.0", lifespan=lifespan)
settings = get_settings()
write_limiter = SlidingWindowLimiter(settings.rate_limit_per_minute, settings.rate_limit_window_seconds)
app.add_middleware(WriteRateLimitMiddleware, limiter=write_limiter)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:8080"], allow_methods=["*"], allow_headers=["*"])


class ProjectIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class FeedbackIn(BaseModel):
    label: str
    comment: str = ""


class TextDocumentIn(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1)
    replace_document_id: str | None = None


def serialize_run(row: AnalysisRunRow) -> dict:
    payload = {key: getattr(row, key) for key in ("id", "project_id", "status", "created_at", "started_at", "completed_at", "input_chars", "prompt_tokens", "completion_tokens", "error")}
    prices_configured = settings.model_input_price_per_million is not None and settings.model_output_price_per_million is not None
    payload["estimated_cost_usd"] = row.estimated_cost_usd if prices_configured else None
    return payload


def serialize_document(row: DocumentRow, include_content: bool = True) -> dict:
    payload = {
        "id": row.id,
        "project_id": row.project_id,
        "name": row.name,
        "version": row.version,
        "active": row.active,
        "created_at": row.created_at,
    }
    if include_content:
        payload["content"] = row.content
    return payload


def dispatch_analysis(run_id: str) -> None:
    if settings.use_celery:
        from .tasks import analyze_project

        analyze_project.delay(run_id)
    else:
        Thread(target=execute_analysis, args=(run_id,), daemon=True).start()


def enforce_daily_model_budget(db) -> None:
    """Reject model-backed work after the local daily usage threshold.

    This is intentionally a single-database check, not a distributed quota
    reservation.  The per-run gate remains the hard fallback for concurrent
    local jobs.
    """
    model_requested = settings.enable_model_extraction and bool(
        settings.openai_api_key.strip()
    )
    if not model_requested:
        return
    now = utc_now_naive()
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    daily_usage = db.scalar(
        select(
            func.coalesce(
                func.sum(
                    AnalysisRunRow.prompt_tokens + AnalysisRunRow.completion_tokens
                ),
                0,
            )
        ).where(
            AnalysisRunRow.created_at >= day_start,
            AnalysisRunRow.status.in_(("running", "completed")),
        )
    ) or 0
    if settings.daily_token_budget <= 0 or daily_usage >= settings.daily_token_budget:
        seconds_to_reset = max(
            1, int(86400 - (now - day_start).total_seconds())
        )
        raise HTTPException(
            429,
            f"当日模型 Token 预算已用尽（{daily_usage}/{settings.daily_token_budget}）",
            headers={"Retry-After": str(seconds_to_reset)},
        )


def prepare_document_version(db, project_id: str, name: str, replace_document_id: str | None) -> tuple[int, list[str]]:
    same_name = db.scalars(
        select(DocumentRow).where(
            DocumentRow.project_id == project_id,
            func.lower(DocumentRow.name) == name.lower(),
        )
    ).all()
    if replace_document_id:
        old = db.get(DocumentRow, replace_document_id)
        if not old or old.project_id != project_id:
            raise HTTPException(404, "待替换文档不存在")
        if old.name.lower() != name.lower():
            raise HTTPException(409, "替换文档必须保持同名；如需新文件请直接上传")
    version = max((row.version for row in same_name), default=0) + 1
    superseded = []
    for row in same_name:
        if row.active:
            row.active = False
            superseded.append(row.id)
    return version, superseded


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "time": utc_now_naive().isoformat()}


@app.post("/api/v1/projects", status_code=201)
def create_project(payload: ProjectIn) -> dict:
    with SessionLocal() as db:
        row = ProjectRow(name=payload.name, description=payload.description)
        db.add(row); db.commit()
        return {"id": row.id, "name": row.name, "description": row.description, "created_at": row.created_at}


@app.get("/api/v1/projects")
def list_projects() -> list[dict]:
    with SessionLocal() as db:
        projects = db.scalars(select(ProjectRow).order_by(ProjectRow.created_at.desc())).all()
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
                "latest_run": serialize_run(latest_run) if latest_run else None,
            })
        return result


@app.get("/api/v1/projects/{project_id}")
def get_project(project_id: str) -> dict:
    with SessionLocal() as db:
        project = db.get(ProjectRow, project_id)
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
            "documents": [serialize_document(d) for d in documents],
        }


@app.get("/api/v1/projects/{project_id}/documents")
def list_documents(project_id: str, include_history: bool = False) -> list[dict]:
    with SessionLocal() as db:
        if not db.get(ProjectRow, project_id):
            raise HTTPException(404, "项目不存在")
        statement = select(DocumentRow).where(DocumentRow.project_id == project_id)
        if not include_history:
            statement = statement.where(DocumentRow.active.is_(True))
        rows = db.scalars(statement.order_by(DocumentRow.name, DocumentRow.version.desc())).all()
        # Version pickers and project history only need metadata. Returning every
        # historical body makes the browser download all revisions of a large
        # story before it can render the workspace.
        return [serialize_document(row, include_content=False) for row in rows]


@app.get("/api/v1/projects/{project_id}/documents/diff")
def compare_document_versions(
    project_id: str, from_document_id: str, to_document_id: str
) -> dict:
    """Compare two stored versions of one same-named document.

    This endpoint is deliberately local and deterministic: it never invokes the
    extraction provider and therefore cannot consume model tokens.
    """
    with SessionLocal() as db:
        if not db.get(ProjectRow, project_id):
            raise HTTPException(404, "项目不存在")
        old = db.get(DocumentRow, from_document_id)
        new = db.get(DocumentRow, to_document_id)
        if not old or old.project_id != project_id:
            raise HTTPException(404, "起始文档版本不存在于当前项目")
        if not new or new.project_id != project_id:
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
                **serialize_document(old, include_content=False),
                "char_count": len(old.content),
                "line_count": len(old.content.splitlines()),
            },
            "to_document": {
                **serialize_document(new, include_content=False),
                "char_count": len(new.content),
                "line_count": len(new.content.splitlines()),
            },
            **diff,
        }


@app.post("/api/v1/projects/{project_id}/documents/text", status_code=201)
def create_text_document(project_id: str, payload: TextDocumentIn) -> dict:
    if not payload.name.lower().endswith((".md", ".txt", ".json")):
        raise HTTPException(415, "名称必须以 .md、.txt 或 .json 结尾")
    if len(payload.content.encode("utf-8")) > settings.max_upload_bytes:
        raise HTTPException(413, "文本超过上传限制")
    with SessionLocal() as db:
        if not db.get(ProjectRow, project_id):
            raise HTTPException(404, "项目不存在")
        version, superseded = prepare_document_version(
            db, project_id, payload.name, payload.replace_document_id
        )
        row = DocumentRow(project_id=project_id, name=payload.name, content=payload.content, version=version)
        db.add(row); db.commit()
        return {**serialize_document(row), "superseded_document_ids": superseded}


@app.post("/api/v1/demo", status_code=201)
def create_demo() -> dict:
    data_dir = Path(__file__).resolve().parents[1] / "data" / "demo-natural"
    files = [data_dir / "world.md", data_dir / "chapter-01.md"]
    if not all(path.exists() for path in files):
        raise HTTPException(500, "演示数据缺失")
    with SessionLocal() as db:
        project = ProjectRow(name="潮汐之门 · 自然文本体验", description="无需 API Key 的中文自然文本基线")
        db.add(project); db.flush()
        for path in files:
            db.add(DocumentRow(project_id=project.id, name=path.name, content=path.read_text(encoding="utf-8")))
        db.commit()
        return {"id": project.id, "name": project.name, "document_count": len(files)}


@app.post("/api/v1/demo/advanced", status_code=201)
def create_advanced_demo() -> dict:
    """Create the original multi-document acceptance scenario."""
    data_dir = Path(__file__).resolve().parents[1] / "data" / "advanced"
    files = [data_dir / "world.md", data_dir / "chapter-01.md", data_dir / "chapter-02.md"]
    if not all(path.exists() for path in files):
        raise HTTPException(500, "复杂演示数据缺失")
    with SessionLocal() as db:
        project = ProjectRow(
            name="静默海域 · 复杂多章节验收",
            description="原创三文档场景，覆盖时间、地点、知识、物品与世界规则",
        )
        db.add(project); db.flush()
        for path in files:
            db.add(DocumentRow(project_id=project.id, name=path.name, content=path.read_text(encoding="utf-8")))
        db.commit()
        return {"id": project.id, "name": project.name, "document_count": len(files)}


@app.post("/api/v1/projects/{project_id}/documents", status_code=201)
async def upload_document(project_id: str, file: UploadFile = File(...), replace_document_id: str | None = Form(None)) -> dict:
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "文件超过上传限制")
    if not file.filename or not file.filename.lower().endswith((".md", ".txt", ".json")):
        raise HTTPException(415, "仅支持 Markdown、TXT 与 JSON")
    try:
        content = data.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "文件必须为 UTF-8 编码")
    with SessionLocal() as db:
        if not db.get(ProjectRow, project_id):
            raise HTTPException(404, "项目不存在")
        version, superseded = prepare_document_version(
            db, project_id, file.filename, replace_document_id
        )
        row = DocumentRow(project_id=project_id, name=file.filename, content=content, version=version)
        db.add(row); db.commit()
        return {**serialize_document(row), "superseded_document_ids": superseded}


@app.post("/api/v1/projects/{project_id}/analysis-runs", status_code=202)
def start_analysis(project_id: str) -> dict:
    with SessionLocal() as db:
        if not db.get(ProjectRow, project_id):
            raise HTTPException(404, "项目不存在")
        if not db.scalar(select(func.count()).select_from(DocumentRow).where(DocumentRow.project_id == project_id, DocumentRow.active.is_(True))):
            raise HTTPException(400, "项目没有可分析文档")
        enforce_daily_model_budget(db)
        run = AnalysisRunRow(project_id=project_id)
        db.add(run); db.commit()
        run_id = run.id
    dispatch_analysis(run_id)
    RUNS.labels(status="queued").inc()
    return {"id": run_id, "status": "queued"}


@app.get("/api/v1/analysis-runs/{run_id}")
def get_run(run_id: str) -> dict:
    with SessionLocal() as db:
        row = db.get(AnalysisRunRow, run_id)
        if not row: raise HTTPException(404, "分析任务不存在")
        return serialize_run(row)


@app.get("/api/v1/projects/{project_id}/analysis-runs")
def list_analysis_runs(project_id: str) -> list[dict]:
    with SessionLocal() as db:
        if not db.get(ProjectRow, project_id):
            raise HTTPException(404, "项目不存在")
        rows = db.scalars(
            select(AnalysisRunRow)
            .where(AnalysisRunRow.project_id == project_id)
            .order_by(AnalysisRunRow.created_at.desc())
        ).all()
        return [serialize_run(row) for row in rows]


@app.post("/api/v1/analysis-runs/{run_id}/cancel", status_code=202)
def cancel_run(run_id: str) -> dict:
    with SessionLocal() as db:
        row = db.get(AnalysisRunRow, run_id)
        if not row: raise HTTPException(404, "分析任务不存在")
        if row.status in {"completed", "failed", "cancelled"}:
            raise HTTPException(409, f"终态任务不能取消：{row.status}")
        if row.cancel_requested:
            return {"id": row.id, "cancel_requested": True, "already_requested": True}
        row.cancel_requested = True; db.commit()
        return {"id": row.id, "cancel_requested": True, "already_requested": False}


@app.post("/api/v1/analysis-runs/{run_id}/retry", status_code=202)
def retry_run(run_id: str) -> dict:
    with SessionLocal() as db:
        old = db.get(AnalysisRunRow, run_id)
        if not old: raise HTTPException(404, "分析任务不存在")
        if old.status not in {"failed", "cancelled"}:
            raise HTTPException(409, "仅失败或已取消任务可以重试")
        enforce_daily_model_budget(db)
        row = AnalysisRunRow(project_id=old.project_id); db.add(row); db.commit(); new_id = row.id
    dispatch_analysis(new_id)
    return {"id": new_id, "status": "queued", "retried_from": run_id}


@app.get("/api/v1/analysis-runs/{run_id}/issues")
def get_issues(run_id: str) -> list[dict]:
    with SessionLocal() as db:
        if not db.get(AnalysisRunRow, run_id): raise HTTPException(404, "分析任务不存在")
        rows = db.scalars(select(IssueRow).where(IssueRow.run_id == run_id)).all()
        return [{"id": r.id, "category": r.category, "severity": r.severity, "confidence": r.confidence, "title": r.title, "explanation": r.explanation, "evidence": r.evidence, "suggestion": r.suggestion, "metadata": r.extra} for r in rows]


@app.get("/api/v1/analysis-runs/{run_id}/records")
def get_records(run_id: str) -> dict:
    with SessionLocal() as db:
        run = db.get(AnalysisRunRow, run_id)
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


def _completed_visualization_rows(db, run_id: str):
    run = db.get(AnalysisRunRow, run_id)
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
def get_graph(run_id: str) -> GraphResponse:
    with SessionLocal() as db:
        records, issues = _completed_visualization_rows(db, run_id)
        return project_graph(run_id, records, issues)


@app.get("/api/v1/analysis-runs/{run_id}/timeline", response_model=TimelineResponse)
def get_timeline(run_id: str) -> TimelineResponse:
    with SessionLocal() as db:
        records, issues = _completed_visualization_rows(db, run_id)
        return project_timeline(run_id, records, issues)


@app.get("/api/v1/analysis-runs/{run_id}/diagnostics")
def get_diagnostics(run_id: str) -> dict:
    with SessionLocal() as db:
        if not db.get(AnalysisRunRow, run_id):
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
):
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
                terminal_payload = {"status": run.status, "error": run.error}
            if terminal:
                yield f"event: terminal\ndata: {json.dumps(terminal_payload, ensure_ascii=False)}\n\n"
                return
            yield ": heartbeat\n\n"
            await asyncio.sleep(0.35)
    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/v1/issues/{issue_id}/feedback", status_code=201)
def feedback(issue_id: str, payload: FeedbackIn, response: Response) -> dict:
    if payload.label not in {"accepted", "false_positive", "resolved"}:
        raise HTTPException(422, "label 必须是 accepted、false_positive 或 resolved")
    with SessionLocal() as db:
        if not db.get(IssueRow, issue_id): raise HTTPException(404, "问题不存在")
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
        row = FeedbackRow(issue_id=issue_id, label=payload.label, comment=payload.comment)
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
def feedback_history(issue_id: str) -> dict:
    with SessionLocal() as db:
        if not db.get(IssueRow, issue_id):
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
def evaluation(evaluation_id: str) -> dict:
    if evaluation_id not in {"baseline", "latest"}: raise HTTPException(404, "仅内置 baseline/latest 评测")
    return {
        "benchmark_kind": "rule-engine synthetic directive regression",
        "natural_language_evaluation": False,
        "warning": "显式 @directive 回归只验证规则接线，不代表自然文本准确率。",
        "metrics": run_evaluation().model_dump(),
    }


@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# Production build can be experienced with one Python process. API routes are
# registered first, then the single-page app handles every remaining path.
frontend_dist = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="web")
