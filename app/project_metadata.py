"""Small project metadata contract with an atomic revision compare-and-swap."""
from __future__ import annotations

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import and_, case, select, update

from .db import ProjectRow
from .project_sort import project_name_sort_key


NO_STORE_HEADERS = {"Cache-Control": "no-store"}


class ProjectMetadataInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    description: str
    expected_revision: int = Field(ge=1, strict=True)

    @field_validator("name", mode="before")
    @classmethod
    def trim_name(cls, value):
        return value.strip() if isinstance(value, str) else value


def _columns():
    return (
        ProjectRow.id, ProjectRow.name, ProjectRow.description,
        ProjectRow.created_at, ProjectRow.metadata_revision,
    )


def get_project_metadata(db, project_id: str, workspace_id: str) -> dict:
    row = db.execute(select(*_columns()).where(
        ProjectRow.id == project_id, ProjectRow.workspace_id == workspace_id,
    )).mappings().one_or_none()
    if row is None:
        raise HTTPException(404, "项目不存在", headers=NO_STORE_HEADERS)
    return dict(row)


def metadata_update_statement(project_id: str, workspace_id: str, payload: ProjectMetadataInput):
    # Check the revision even for a no-op, in the same atomic UPDATE. No-op
    # preserves the revision; it cannot accidentally increment or overwrite a
    # concurrent author's edits between a SELECT and an unconditional write.
    unchanged = and_(ProjectRow.name == payload.name, ProjectRow.description == payload.description)
    return update(ProjectRow).where(
        ProjectRow.id == project_id,
        ProjectRow.workspace_id == workspace_id,
        ProjectRow.metadata_revision == payload.expected_revision,
    ).values(
        name=payload.name,
        description=payload.description,
        name_sort_key=project_name_sort_key(payload.name),
        metadata_revision=case((unchanged, ProjectRow.metadata_revision), else_=ProjectRow.metadata_revision + 1),
    ).returning(*_columns())


def update_project_metadata(db, project_id: str, workspace_id: str, payload: ProjectMetadataInput) -> dict:
    row = db.execute(metadata_update_statement(project_id, workspace_id, payload)).mappings().one_or_none()
    if row is None:
        # Same 404 for missing/foreign projects; conflict reveals no current
        # name, description or revision. A separate authorized GET is needed.
        if db.scalar(select(ProjectRow.id).where(
            ProjectRow.id == project_id, ProjectRow.workspace_id == workspace_id,
        )) is None:
            raise HTTPException(404, "项目不存在", headers=NO_STORE_HEADERS)
        raise HTTPException(409, detail={
            "reason_code": "project_metadata_revision_conflict",
            "message": "项目名称或简介已在其他页面更新，请重新读取并核对后再保存。",
        }, headers=NO_STORE_HEADERS)
    db.commit()
    return dict(row)
