from __future__ import annotations

from math import isfinite
from typing import Literal

from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from .db import AnalysisRunExecutionRow, AnalysisRunInputRow, AnalysisRunRow, ProjectRow


RunCatalogStatus = Literal["all", "queued", "running", "completed", "failed", "cancelled"]
NO_STORE_HEADERS = {"Cache-Control": "no-store"}
NOT_FOUND_MESSAGE = "项目不存在"


class RunCatalogNotFound(Exception):
    pass


def _counter(value: object) -> int:
    return max(0, value) if type(value) is int else 0


def build_run_catalog(
    db,
    *,
    project_id: str,
    workspace_id: str,
    page: int = 1,
    page_size: int = 20,
    status: RunCatalogStatus = "all",
    prices_configured: bool = False,
) -> dict:
    """Read one bounded metadata page, without hydrating run/input JSON or text.

    Reported usage counters are deliberately not qualified as complete usage:
    understanding partial/lower-bound accounting still requires the run detail.
    Snapshot count likewise describes stored rows, not their integrity or OOC
    readiness. This endpoint must never be used to select a character baseline.
    """
    project = db.scalar(select(ProjectRow.id).where(
        ProjectRow.id == project_id,
        ProjectRow.workspace_id == workspace_id,
    ))
    if project is None:
        raise RunCatalogNotFound()

    filters = [AnalysisRunRow.project_id == project_id]
    if status != "all":
        filters.append(AnalysisRunRow.status == status)
    total = db.scalar(select(func.count()).select_from(AnalysisRunRow).where(*filters)) or 0

    retry_source = aliased(AnalysisRunRow, name="catalog_retry_source")
    statement = (
        select(
            AnalysisRunRow.id,
            AnalysisRunRow.project_id,
            AnalysisRunRow.status,
            AnalysisRunRow.created_at,
            AnalysisRunRow.started_at,
            AnalysisRunRow.completed_at,
            AnalysisRunRow.input_chars,
            AnalysisRunRow.prompt_tokens,
            AnalysisRunRow.completion_tokens,
            AnalysisRunRow.estimated_cost_usd,
            AnalysisRunRow.batch_mode,
            retry_source.id.label("retried_from"),
        )
        .outerjoin(AnalysisRunExecutionRow, AnalysisRunExecutionRow.run_id == AnalysisRunRow.id)
        # Invalid historical lineage must not expose a different project's ID.
        .outerjoin(retry_source, (
            (retry_source.id == AnalysisRunExecutionRow.retried_from_run_id)
            & (retry_source.project_id == project_id)
        ))
        .where(*filters)
        .order_by(AnalysisRunRow.created_at.desc(), AnalysisRunRow.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    rows = db.execute(statement).mappings().all()
    run_ids = [row["id"] for row in rows]
    snapshot_counts = {}
    if run_ids:
        snapshot_counts = dict(db.execute(
            select(AnalysisRunInputRow.run_id, func.count(AnalysisRunInputRow.id))
            .where(AnalysisRunInputRow.run_id.in_(run_ids))
            .group_by(AnalysisRunInputRow.run_id)
        ).all())
    items = []
    for row in rows:
        cost = row["estimated_cost_usd"]
        items.append({
            "id": row["id"],
            "project_id": row["project_id"],
            "status": row["status"] if row["status"] in {
                "queued", "running", "completed", "failed", "cancelled",
            } else "unknown",
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "input_chars": _counter(row["input_chars"]),
            "prompt_tokens": _counter(row["prompt_tokens"]),
            "completion_tokens": _counter(row["completion_tokens"]),
            "estimated_cost_usd": (
                cost if prices_configured and type(cost) in (int, float)
                and isfinite(cost) and cost >= 0 else None
            ),
            "frozen_document_count": snapshot_counts.get(row["id"], 0),
            "retried_from": row["retried_from"],
            "batch_mode": row["batch_mode"] if row["batch_mode"] in {
                "full_review", "baseline_build", "draft_review",
            } else None,
        })
    return {
        "project_id": project_id,
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_more": page * page_size < total,
        "items": items,
    }
