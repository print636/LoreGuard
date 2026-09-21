"""Preserve authority tier on confirmed character traits.

Revision ID: 0010_character_trait_authority
Revises: 0009_narrative_authority_traits
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0010_character_trait_authority"
down_revision = "0009_narrative_authority_traits"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("character_trait_candidates"):
        return
    columns = {row["name"] for row in inspector.get_columns("character_trait_candidates")}
    if "authority_tier" in columns:
        return
    op.add_column(
        "character_trait_candidates",
        sa.Column(
            "authority_tier",
            sa.String(24),
            nullable=False,
            server_default="formal_record",
        ),
    )
    # SQLite's additive ALTER TABLE cannot add a named CHECK without rebuilding
    # and reflecting every foreign-key target.  Some supported legacy-adoption
    # fixtures intentionally omit those targets, so enforce the enum in the
    # application there and add the database constraint on full SQL databases.
    if bind.dialect.name != "sqlite":
        op.create_check_constraint(
            "ck_character_trait_candidate_authority",
            "character_trait_candidates",
            "authority_tier IN ('core_canon', 'formal_record')",
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("character_trait_candidates"):
        return
    columns = {row["name"] for row in inspector.get_columns("character_trait_candidates")}
    if "authority_tier" not in columns:
        return
    if bind.dialect.name != "sqlite":
        op.drop_constraint(
            "ck_character_trait_candidate_authority",
            "character_trait_candidates",
            type_="check",
        )
    op.drop_column("character_trait_candidates", "authority_tier")
