"""Optional server-bound character candidate support spans.

Revision ID: 0016_character_support_bindings
Revises: 0015_character_trait_axes
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0016_character_support_bindings"
down_revision = "0015_character_trait_axes"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {item["name"] for item in inspector.get_columns("character_trait_candidates")}
    constraint_name = "ck_character_trait_candidate_support_bindings_pair"
    constraints = {
        item["name"] for item in inspector.get_check_constraints(
            "character_trait_candidates"
        )
    }
    check_sql = (
        "support_binding_mode IS NOT NULL AND ("
        "(support_binding_mode = 'legacy_v1' "
        "AND support_bindings_v1 IS NULL "
        "AND support_bindings_sha256 IS NULL) OR "
        "(support_binding_mode = 'required_v1' "
        "AND support_bindings_v1 IS NOT NULL "
        "AND support_bindings_sha256 IS NOT NULL "
        "AND length(support_bindings_sha256) = 64))"
    )
    missing = {
        "support_binding_mode": sa.Column("support_binding_mode", sa.String(24), nullable=True),
        "support_bindings_v1": sa.Column(
            "support_bindings_v1", sa.JSON(none_as_null=True), nullable=True
        ),
        "support_bindings_sha256": sa.Column("support_bindings_sha256", sa.String(64), nullable=True),
    }
    adding_all_columns = all(name not in columns for name in missing)
    for name, column in missing.items():
        if name not in columns:
            op.add_column("character_trait_candidates", column)
    if adding_all_columns:
        # This is the authoritative pre-0016 population. Historical hashes
        # may encode a comparison key no longer stored in the row, so record
        # their legacy origin here instead of attempting to invert SHA-256.
        bind.execute(sa.text(
            "UPDATE character_trait_candidates SET support_binding_mode = 'legacy_v1' "
            "WHERE support_binding_mode IS NULL AND support_bindings_v1 IS NULL "
            "AND support_bindings_sha256 IS NULL"
        ))
    # If any column already existed (an interrupted/WIP migration), do not
    # silently launder triple-NULL new rows into legacy rows.
    if bind.dialect.name != "sqlite":
        op.alter_column(
            "character_trait_candidates", "support_binding_mode",
            existing_type=sa.String(24), nullable=False,
        )
    if bind.dialect.name != "sqlite" and constraint_name not in constraints:
        op.create_check_constraint(
            constraint_name, "character_trait_candidates", check_sql
        )
    # SQLite cannot ADD CHECK without rebuilding this referenced table. That
    # rebuild drops earlier named checks under Alembic reflection; preserve the
    # existing constraints and enforce this new contract in read/write paths.


def downgrade() -> None:
    bind = op.get_bind()
    has_bound_candidates = bind.execute(sa.text(
        "SELECT 1 FROM character_trait_candidates WHERE "
        "support_binding_mode IS NULL OR "
        "support_binding_mode != 'legacy_v1' OR "
        "support_bindings_v1 IS NOT NULL OR "
        "support_bindings_sha256 IS NOT NULL LIMIT 1"
    )).first()
    if has_bound_candidates is not None:
        raise RuntimeError(
            "cannot downgrade 0016 while character support bindings exist"
        )
    if bind.dialect.name != "sqlite":
        op.drop_constraint(
            "ck_character_trait_candidate_support_bindings_pair",
            "character_trait_candidates",
            type_="check",
        )
    op.drop_column("character_trait_candidates", "support_bindings_sha256")
    op.drop_column("character_trait_candidates", "support_bindings_v1")
    op.drop_column("character_trait_candidates", "support_binding_mode")
