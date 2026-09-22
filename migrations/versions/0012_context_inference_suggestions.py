"""Persist safe model-inferred context suggestion details.

Revision ID: 0012_context_inference
Revises: 0011_review_batch_contract
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0012_context_inference"
down_revision = "0011_review_batch_contract"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("document_narrative_context_revisions"):
        return
    columns = {
        row["name"]
        for row in inspector.get_columns("document_narrative_context_revisions")
    }
    additions = (
        ("inference_reasoning", sa.Text()),
        ("inference_evidence", sa.JSON()),
        ("inference_usage", sa.JSON()),
    )
    for name, column_type in additions:
        if name not in columns:
            op.add_column(
                "document_narrative_context_revisions",
                sa.Column(name, column_type, nullable=True),
            )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("document_narrative_context_revisions"):
        return
    columns = {
        row["name"]
        for row in inspector.get_columns("document_narrative_context_revisions")
    }
    for name in (
        "inference_usage",
        "inference_evidence",
        "inference_reasoning",
    ):
        if name in columns:
            op.drop_column("document_narrative_context_revisions", name)
