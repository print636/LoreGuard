"""Freeze review batch mode, sensitivity, and input roles.

Revision ID: 0011_review_batch_contract
Revises: 0010_character_trait_authority
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0011_review_batch_contract"
down_revision = "0010_character_trait_authority"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if inspector.has_table("analysis_runs"):
        columns = {row["name"] for row in inspector.get_columns("analysis_runs")}
        if "batch_mode" not in columns:
            op.add_column(
                "analysis_runs",
                sa.Column(
                    "batch_mode",
                    sa.String(24),
                    nullable=False,
                    server_default="full_review",
                ),
            )
        if "sensitivity" not in columns:
            op.add_column(
                "analysis_runs",
                sa.Column(
                    "sensitivity",
                    sa.String(24),
                    nullable=False,
                    server_default="balanced",
                ),
            )
        if "batch_coverage" not in columns:
            op.add_column(
                "analysis_runs",
                sa.Column(
                    "batch_coverage",
                    sa.JSON(),
                    nullable=False,
                    server_default="{}",
                ),
            )

    inspector = sa.inspect(bind)
    if inspector.has_table("analysis_run_input_context"):
        columns = {
            row["name"]
            for row in inspector.get_columns("analysis_run_input_context")
        }
        if "batch_role" not in columns:
            op.add_column(
                "analysis_run_input_context",
                sa.Column(
                    "batch_role",
                    sa.String(16),
                    nullable=False,
                    server_default="target",
                ),
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("analysis_run_input_context"):
        columns = {
            row["name"]
            for row in inspector.get_columns("analysis_run_input_context")
        }
        if "batch_role" in columns:
            op.drop_column("analysis_run_input_context", "batch_role")
    inspector = sa.inspect(bind)
    if inspector.has_table("analysis_runs"):
        columns = {row["name"] for row in inspector.get_columns("analysis_runs")}
        if "batch_coverage" in columns:
            op.drop_column("analysis_runs", "batch_coverage")
        if "sensitivity" in columns:
            op.drop_column("analysis_runs", "sensitivity")
        if "batch_mode" in columns:
            op.drop_column("analysis_runs", "batch_mode")
