from __future__ import annotations

from time import perf_counter

from sqlalchemy import delete, select

from .db import AnalysisDiagnosticRow, AnalysisRecordRow, AnalysisRunRow, DocumentRow, IssueRow, RunEventRow, SessionLocal
from .pipeline import AnalysisPipeline, DocumentInput
from .config import get_settings
from .usage import configured_cost_usd
from .time_utils import utc_now_naive


def emit(db, run_id: str, stage: str, progress: int, message: str) -> None:
    db.add(RunEventRow(run_id=run_id, stage=stage, progress=progress, message=message))
    db.commit()


def execute_analysis(run_id: str, *, raise_on_failure: bool = False) -> None:
    service_started = perf_counter()
    with SessionLocal() as db:
        run = db.get(AnalysisRunRow, run_id)
        if not run:
            return
        try:
            if run.cancel_requested:
                run.status = "cancelled"
                run.completed_at = utc_now_naive()
                db.commit()
                emit(db, run_id, "cancelled", 100, "任务在开始前已取消")
                return
            run.status = "running"
            run.started_at = utc_now_naive()
            db.commit()
            docs = db.scalars(select(DocumentRow).where(DocumentRow.project_id == run.project_id, DocumentRow.active.is_(True))).all()
            run.input_chars = sum(len(doc.content) for doc in docs)
            pipeline = AnalysisPipeline()
            result = pipeline.run(
                [DocumentInput(id=doc.id, name=doc.name, content=doc.content) for doc in docs],
                on_stage=lambda stage, progress, message: emit(db, run_id, stage, progress, message),
            )
            if run.cancel_requested:
                run.status = "cancelled"
                run.completed_at = utc_now_naive()
                db.commit()
                emit(db, run_id, "cancelled", 100, "任务已取消")
                return
            issues = result.issues
            report_started = perf_counter()
            db.execute(delete(IssueRow).where(IssueRow.run_id == run_id))
            db.execute(delete(AnalysisRecordRow).where(AnalysisRecordRow.run_id == run_id))
            db.execute(delete(AnalysisDiagnosticRow).where(AnalysisDiagnosticRow.run_id == run_id))
            for directive in result.directives:
                db.add(AnalysisRecordRow(
                    run_id=run_id,
                    kind=directive.kind,
                    attrs=directive.attrs,
                    evidence=directive.evidence.model_dump(),
                ))
            for issue in issues:
                db.add(IssueRow(
                    run_id=run_id, category=issue.category.value, severity=issue.severity.value,
                    confidence=issue.confidence, title=issue.title, explanation=issue.explanation,
                    evidence=[span.model_dump() for span in issue.evidence], suggestion=issue.suggestion,
                    extra=issue.metadata,
                ))
            run.prompt_tokens = result.prompt_tokens
            run.completion_tokens = result.completion_tokens
            cost = configured_cost_usd(result.prompt_tokens, result.completion_tokens, get_settings())
            run.estimated_cost_usd = cost if cost is not None else 0
            run.status = "completed"
            run.completed_at = utc_now_naive()
            timings = result.diagnostics.setdefault("timings", {})
            timings["report_ms"] = round(float(timings.get("report_ms", 0)) + (perf_counter() - report_started) * 1000, 3)
            timings["total_ms"] = round((perf_counter() - service_started) * 1000, 3)
            db.add(AnalysisDiagnosticRow(run_id=run_id, payload=result.diagnostics))
            db.commit()
            if result.warnings:
                emit(db, run_id, "warning", 90, "；".join(result.warnings[:5]))
            mode = "模型增强" if result.model_used else "确定性基线"
            emit(db, run_id, "extract", 92, f"本次使用{mode}，保存 {len(result.directives)} 条记录")
            emit(db, run_id, "report", 100, "证据化报告生成完成")
        except Exception as exc:
            run.status = "failed"
            run.error = str(exc)
            run.completed_at = utc_now_naive()
            db.commit()
            emit(db, run_id, "failed", 100, "分析失败，可调用重试接口恢复")
            if raise_on_failure:
                raise
