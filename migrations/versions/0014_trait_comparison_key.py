"""Persist the server-derived character trait comparison identity.

Revision ID: 0014_trait_comparison_key
Revises: 0013_account_model_provider
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0014_trait_comparison_key"
down_revision = "0013_account_model_provider"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("character_trait_candidates"):
        return
    columns = {row["name"] for row in inspector.get_columns("character_trait_candidates")}
    if "comparison_key" not in columns:
        # Existing rows have no recoverable object anchor. Leave them NULL.
        op.add_column(
            "character_trait_candidates",
            sa.Column("comparison_key", sa.String(200), nullable=True),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("character_trait_candidates"):
        return
    columns = {row["name"] for row in inspector.get_columns("character_trait_candidates")}
    if "comparison_key" in columns:
        op.drop_column("character_trait_candidates", "comparison_key")
