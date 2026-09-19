"""Persist revision-run lineage and deterministic issue comparisons.

Revision ID: 0007_revision_comparisons
Revises: 0006_run_idempotency
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0007_revision_comparisons"
down_revision = "0006_run_idempotency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("analysis_runs") or not inspector.has_table("issues"):
        return

    if not inspector.has_table("analysis_run_comparisons"):
        op.create_table(
            "analysis_run_comparisons",
            sa.Column("id", sa.String(36), nullable=False),
            sa.Column("project_id", sa.String(36), nullable=False),
            sa.Column("baseline_run_id", sa.String(36), nullable=False),
            sa.Column("target_run_id", sa.String(36), nullable=False),
            sa.Column("matcher_version", sa.String(80), nullable=False),
            sa.Column("status", sa.String(24), nullable=False),
            sa.Column("summary", sa.JSON(), nullable=False),
            sa.Column("provenance", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(
                ["project_id"], ["projects.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["baseline_run_id"], ["analysis_runs.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["target_run_id"], ["analysis_runs.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "baseline_run_id",
                "target_run_id",
                name="uq_analysis_run_comparison_pair",
            ),
            sa.UniqueConstraint("target_run_id"),
        )
        op.create_index(
            "ix_analysis_run_comparisons_project_id",
            "analysis_run_comparisons",
            ["project_id"],
        )
        op.create_index(
            "ix_analysis_run_comparisons_baseline_run_id",
            "analysis_run_comparisons",
            ["baseline_run_id"],
        )
        op.create_index(
            "ix_analysis_run_comparisons_target_run_id",
            "analysis_run_comparisons",
            ["target_run_id"],
        )
        op.create_index(
            "ix_analysis_run_comparisons_baseline_created",
            "analysis_run_comparisons",
            ["baseline_run_id", "created_at"],
        )

    inspector = sa.inspect(bind)
    if not inspector.has_table("issue_comparison_items"):
        op.create_table(
            "issue_comparison_items",
            sa.Column("id", sa.String(36), nullable=False),
            sa.Column("comparison_id", sa.String(36), nullable=False),
            sa.Column("outcome", sa.String(24), nullable=False),
            sa.Column("baseline_issue_id", sa.String(36), nullable=True),
            sa.Column("target_issue_id", sa.String(36), nullable=True),
            sa.Column("match_method", sa.String(80), nullable=True),
            sa.Column("match_score", sa.Integer(), nullable=True),
            sa.Column("provenance", sa.JSON(), nullable=False),
            sa.CheckConstraint(
                "outcome IN ('no_longer_detected', 'persisting', 'new', 'unverifiable')",
                name="ck_issue_comparison_outcome",
            ),
            sa.ForeignKeyConstraint(
                ["comparison_id"],
                ["analysis_run_comparisons.id"],
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["baseline_issue_id"], ["issues.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["target_issue_id"], ["issues.id"], ondelete="CASCADE"
            ),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint(
                "comparison_id",
                "baseline_issue_id",
                name="uq_issue_comparison_baseline_issue",
            ),
            sa.UniqueConstraint(
                "comparison_id",
                "target_issue_id",
                name="uq_issue_comparison_target_issue",
            ),
        )
        op.create_index(
            "ix_issue_comparison_items_comparison_id",
            "issue_comparison_items",
            ["comparison_id"],
        )
        op.create_index(
            "ix_issue_comparison_items_outcome",
            "issue_comparison_items",
            ["outcome"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("issue_comparison_items"):
        op.drop_table("issue_comparison_items")
    if sa.inspect(bind).has_table("analysis_run_comparisons"):
        op.drop_table("analysis_run_comparisons")
