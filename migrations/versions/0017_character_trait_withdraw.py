"""Allow explicit, audited withdrawal of an author-confirmed character trait.

Revision ID: 0017_character_trait_withdraw
Revises: 0016_character_support_bindings
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0017_character_trait_withdraw"
down_revision = "0016_character_support_bindings"
branch_labels = None
depends_on = None


_CHECKS = (
    (
        "character_trait_candidates",
        "ck_character_trait_candidate_review_state",
        "review_state IN ('pending', 'confirmed', 'rejected', 'superseded', 'withdrawn')",
        "review_state IN ('pending', 'confirmed', 'rejected', 'superseded')",
    ),
    (
        "character_trait_reviews",
        "ck_character_trait_review_decision",
        "decision IN ('confirm', 'reject', 'supersede', 'withdraw')",
        "decision IN ('confirm', 'reject', 'supersede')",
    ),
)


def _replace_check(table: str, name: str, expression: str) -> None:
    # SQLite cannot ALTER a CHECK directly. Alembic's reflected batch copy
    # keeps rows and the other named constraints/indexes while replacing it.
    if op.get_bind().dialect.name == "sqlite":
        # Some stamped legacy/WIP databases have dangling FK declarations;
        # do not resolve unrelated referenced tables during the local copy.
        with op.batch_alter_table(
            table, recreate="always", reflect_kwargs={"resolve_fks": False}
        ) as batch:
            batch.drop_constraint(name, type_="check")
            batch.create_check_constraint(name, expression)
        _restore_sqlite_axis_guards(table)
    else:
        op.drop_constraint(name, table, type_="check")
        op.create_check_constraint(name, table, expression)


def _restore_sqlite_axis_guards(table: str) -> None:
    """SQLite table copies drop the guards installed by revision 0015."""

    pair_name = (
        "ck_character_trait_candidate_axis_pair"
        if table == "character_trait_candidates"
        else "ck_character_trait_review_axis_pair"
    )
    for action in ("INSERT", "UPDATE"):
        action_key = action.lower()
        op.execute(sa.text(
            f"CREATE TRIGGER IF NOT EXISTS {pair_name}_{action_key} "
            f"BEFORE {action} ON {table} "
            "WHEN (NEW.approved_axis_id IS NULL AND "
            "NEW.approved_axis_version IS NOT NULL) OR "
            "(NEW.approved_axis_id IS NOT NULL AND "
            "(NEW.approved_axis_version IS NULL OR "
            "NEW.approved_axis_version != 1)) "
            f"BEGIN SELECT RAISE(ABORT, '{pair_name}'); END"
        ))
        op.execute(sa.text(
            f"CREATE TRIGGER IF NOT EXISTS fk_{table}_axis_project_{action_key} "
            f"BEFORE {action} ON {table} "
            "WHEN NEW.approved_axis_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM character_trait_axes "
            "WHERE id = NEW.approved_axis_id AND "
            "project_id = NEW.project_id) "
            "BEGIN SELECT RAISE(ABORT, "
            "'fk_character_trait_axis_project'); END"
        ))


def upgrade() -> None:
    for table, name, expanded, _previous in _CHECKS:
        _replace_check(table, name, expanded)


def downgrade() -> None:
    bind = op.get_bind()
    withdrawn_candidate = bind.execute(sa.text(
        "SELECT 1 FROM character_trait_candidates "
        "WHERE review_state = 'withdrawn' LIMIT 1"
    )).first()
    withdrawal_review = bind.execute(sa.text(
        "SELECT 1 FROM character_trait_reviews "
        "WHERE decision = 'withdraw' LIMIT 1"
    )).first()
    if withdrawn_candidate is not None or withdrawal_review is not None:
        raise RuntimeError(
            "cannot downgrade 0017 while withdrawn traits or withdrawal reviews exist"
        )
    for table, name, _expanded, previous in reversed(_CHECKS):
        _replace_check(table, name, previous)
