"""Finish comparison provenance and serialize logical document revisions.

Revision ID: 0008_document_concurrency
Revises: 0007_revision_comparisons
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0008_document_concurrency"
down_revision = "0007_revision_comparisons"
branch_labels = None
depends_on = None


def _document_index_names(bind) -> set[str]:
    if bind.dialect.name == "sqlite":
        return {
            str(row[1])
            for row in bind.exec_driver_sql("PRAGMA index_list('documents')").all()
        }
    return {row["name"] for row in sa.inspect(bind).get_indexes("documents")}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("documents"):
        return
    indexes = _document_index_names(bind)
    columns = ["project_id", sa.text("lower(name)"), "version"]
    if "uq_documents_project_lower_name_version" not in indexes:
        op.create_index(
            "uq_documents_project_lower_name_version",
            "documents",
            columns,
            unique=True,
        )
    if "uq_documents_project_lower_name_active" not in indexes:
        kwargs = (
            {"postgresql_where": sa.text("active")}
            if bind.dialect.name == "postgresql"
            else {"sqlite_where": sa.text("active = 1")}
        )
        op.create_index(
            "uq_documents_project_lower_name_active",
            "documents",
            ["project_id", sa.text("lower(name)")],
            unique=True,
            **kwargs,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("documents"):
        indexes = _document_index_names(bind)
        for name in (
            "uq_documents_project_lower_name_active",
            "uq_documents_project_lower_name_version",
        ):
            if name in indexes:
                op.drop_index(name, table_name="documents")
