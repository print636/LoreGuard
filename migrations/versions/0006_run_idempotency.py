"""Add project-scoped analysis-run idempotency and history ordering.

Revision ID: 0006_run_idempotency
Revises: 0005_workspace_isolation

The optional key preserves compatibility with existing clients.  The unique
index is the concurrency authority; request code also catches the losing
transaction's uniqueness error instead of relying on a check-before-insert.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0006_run_idempotency"
down_revision = "0005_workspace_isolation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    # Some explicitly stamped pre-Alembic/WIP databases contain only the
    # document and embedding substrate.  Match earlier additive revisions by
    # leaving an absent optional product table untouched.
    if not inspector.has_table("analysis_runs"):
        return
    columns = {item["name"] for item in inspector.get_columns("analysis_runs")}
    if "idempotency_key" not in columns:
        op.add_column(
            "analysis_runs",
            sa.Column("idempotency_key", sa.String(200), nullable=True),
        )
    indexes = {
        item["name"] for item in sa.inspect(bind).get_indexes("analysis_runs")
    }
    if "uq_analysis_runs_project_id_idempotency_key" not in indexes:
        op.create_index(
            "uq_analysis_runs_project_id_idempotency_key",
            "analysis_runs",
            ["project_id", "idempotency_key"],
            unique=True,
        )
    if "ix_analysis_runs_project_created_id" not in indexes:
        op.create_index(
            "ix_analysis_runs_project_created_id",
            "analysis_runs",
            ["project_id", "created_at", "id"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("analysis_runs"):
        return
    indexes = {
        item["name"] for item in inspector.get_indexes("analysis_runs")
    }
    if "ix_analysis_runs_project_created_id" in indexes:
        op.drop_index(
            "ix_analysis_runs_project_created_id", table_name="analysis_runs"
        )
    if "uq_analysis_runs_project_id_idempotency_key" in indexes:
        op.drop_index(
            "uq_analysis_runs_project_id_idempotency_key", table_name="analysis_runs"
        )
    columns = {
        item["name"] for item in sa.inspect(bind).get_columns("analysis_runs")
    }
    if "idempotency_key" in columns:
        with op.batch_alter_table("analysis_runs") as batch:
            batch.drop_column("idempotency_key")
