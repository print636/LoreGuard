"""Record an author axis's positive proposition and explicit direction alignment.

Revision ID: 0018_character_axis_direction
Revises: 0017_character_trait_withdraw
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0018_character_axis_direction"
down_revision = "0017_character_trait_withdraw"
branch_labels = None
depends_on = None


_AXIS = "character_trait_axes"
_CANDIDATES = "character_trait_candidates"
_REVIEWS = "character_trait_reviews"


def _columns(table: str) -> set[str]:
    return {item["name"] for item in sa.inspect(op.get_bind()).get_columns(table)}


def _has_check(table: str, name: str) -> bool:
    return any(
        item.get("name") == name
        for item in sa.inspect(op.get_bind()).get_check_constraints(table)
    )


def _sqlite_guard(table: str, action: str) -> str:
    return f"tr_{table}_axis_alignment_{action.lower()}"


def _sqlite_pair_when(table: str) -> str:
    pair = (
        "(NEW.axis_alignment IS NULL AND NEW.axis_polarity IS NULL "
        "AND NEW.axis_positive_proposition_sha256 IS NULL) OR "
        "(NEW.approved_axis_id IS NOT NULL AND "
        "NEW.axis_alignment IS NOT NULL AND NEW.axis_polarity IS NOT NULL AND "
        "NEW.axis_positive_proposition_sha256 IS NOT NULL AND "
        "NEW.axis_alignment IN ('same', 'opposite') AND "
        "NEW.axis_polarity IN ('positive', 'negative') AND "
        "length(NEW.axis_positive_proposition_sha256) = 64)"
    )
    if table == _CANDIDATES:
        pair += (
            " AND (NEW.axis_alignment IS NULL OR ("
            "(NEW.polarity = 'positive' AND NEW.axis_alignment = 'same' "
            "AND NEW.axis_polarity = 'positive') OR "
            "(NEW.polarity = 'positive' AND NEW.axis_alignment = 'opposite' "
            "AND NEW.axis_polarity = 'negative') OR "
            "(NEW.polarity = 'negative' AND NEW.axis_alignment = 'same' "
            "AND NEW.axis_polarity = 'negative') OR "
            "(NEW.polarity = 'negative' AND NEW.axis_alignment = 'opposite' "
            "AND NEW.axis_polarity = 'positive')))"
        )
    return pair


def _replace_review_decision_check(*, expanded: bool) -> None:
    expression = (
        "decision IN ('confirm', 'reject', 'supersede', 'withdraw', 'align')"
        if expanded else
        "decision IN ('confirm', 'reject', 'supersede', 'withdraw')"
    )
    if op.get_bind().dialect.name == "sqlite":
        # 0017 already uses a tested batch copy for this check. Execute before
        # adding new alignment columns so no new fields are lost on reflection.
        with op.batch_alter_table(
            _REVIEWS, recreate="always", reflect_kwargs={"resolve_fks": False}
        ) as batch:
            batch.drop_constraint("ck_character_trait_review_decision", type_="check")
            batch.create_check_constraint("ck_character_trait_review_decision", expression)
    else:
        op.drop_constraint("ck_character_trait_review_decision", _REVIEWS, type_="check")
        op.create_check_constraint(
            "ck_character_trait_review_decision", _REVIEWS, expression
        )


def _restore_sqlite_axis_guards(table: str) -> None:
    # The 0017 batch copy drops standalone triggers created in 0015.
    pair_name = (
        "ck_character_trait_candidate_axis_pair"
        if table == _CANDIDATES else "ck_character_trait_review_axis_pair"
    )
    for action in ("INSERT", "UPDATE"):
        suffix = action.lower()
        op.execute(sa.text(
            f"CREATE TRIGGER IF NOT EXISTS {pair_name}_{suffix} "
            f"BEFORE {action} ON {table} "
            "WHEN (NEW.approved_axis_id IS NULL AND NEW.approved_axis_version IS NOT NULL) "
            "OR (NEW.approved_axis_id IS NOT NULL AND "
            "(NEW.approved_axis_version IS NULL OR NEW.approved_axis_version != 1)) "
            f"BEGIN SELECT RAISE(ABORT, '{pair_name}'); END"
        ))
        op.execute(sa.text(
            f"CREATE TRIGGER IF NOT EXISTS fk_{table}_axis_project_{suffix} "
            f"BEFORE {action} ON {table} "
            "WHEN NEW.approved_axis_id IS NOT NULL AND NOT EXISTS "
            "(SELECT 1 FROM character_trait_axes WHERE id = NEW.approved_axis_id "
            "AND project_id = NEW.project_id) "
            "BEGIN SELECT RAISE(ABORT, 'fk_character_trait_axis_project'); END"
        ))


def _restore_sqlite_axis_identity_immutability() -> None:
    op.execute(sa.text(
        "CREATE TRIGGER IF NOT EXISTS tr_character_trait_axes_immutable "
        "BEFORE UPDATE OF id, project_id, trait_type, version, "
        "display_name, definition, definition_sha256, created_at "
        "ON character_trait_axes "
        "BEGIN SELECT RAISE(ABORT, 'character_trait_axis_immutable'); END"
    ))


def _install_sqlite_guards() -> None:
    for action in ("INSERT", "UPDATE"):
        op.execute(sa.text(
            f"CREATE TRIGGER IF NOT EXISTS {_sqlite_guard(_AXIS, action)} "
            f"BEFORE {action} ON {_AXIS} "
            "WHEN NOT ((NEW.positive_proposition IS NULL AND "
            "NEW.positive_proposition_sha256 IS NULL AND "
            "NEW.positive_proposition_authored_at IS NULL AND "
            "NEW.positive_proposition_authored_by_user_id IS NULL) OR "
            "(NEW.positive_proposition IS NOT NULL AND "
            "NEW.positive_proposition_sha256 IS NOT NULL AND "
            "NEW.positive_proposition_authored_at IS NOT NULL AND "
            "length(NEW.positive_proposition_sha256) = 64)) "
            "BEGIN SELECT RAISE(ABORT, 'character_trait_axis_proposition_pair'); END"
        ))
        for table in (_CANDIDATES, _REVIEWS):
            op.execute(sa.text(
                f"CREATE TRIGGER IF NOT EXISTS {_sqlite_guard(table, action)} "
                f"BEFORE {action} ON {table} "
                f"WHEN NOT ({_sqlite_pair_when(table)}) "
                "BEGIN SELECT RAISE(ABORT, 'character_trait_axis_alignment_pair'); END"
            ))
    op.execute(sa.text(
        "CREATE TRIGGER IF NOT EXISTS tr_character_trait_axis_proposition_immutable "
        f"BEFORE UPDATE OF positive_proposition, positive_proposition_sha256, "
        f"positive_proposition_authored_at ON {_AXIS} "
        "WHEN OLD.positive_proposition IS NOT NULL OR "
        "OLD.positive_proposition_sha256 IS NOT NULL OR "
        "OLD.positive_proposition_authored_at IS NOT NULL "
        "BEGIN SELECT RAISE(ABORT, 'character_trait_axis_proposition_immutable'); END"
    ))


def upgrade() -> None:
    bind = op.get_bind()
    _replace_review_decision_check(expanded=True)
    if bind.dialect.name == "sqlite":
        _restore_sqlite_axis_guards(_REVIEWS)
    axis_columns = _columns(_AXIS)
    if "positive_proposition" not in axis_columns:
        op.add_column(_AXIS, sa.Column("positive_proposition", sa.String(200), nullable=True))
    if "positive_proposition_sha256" not in axis_columns:
        op.add_column(
            _AXIS, sa.Column("positive_proposition_sha256", sa.String(64), nullable=True)
        )
    if "positive_proposition_authored_by_user_id" not in axis_columns:
        if bind.dialect.name == "sqlite":
            op.execute(sa.text(
                "ALTER TABLE character_trait_axes ADD COLUMN "
                "positive_proposition_authored_by_user_id VARCHAR(36) "
                "REFERENCES users(id) ON DELETE SET NULL"
            ))
        else:
            op.add_column(_AXIS, sa.Column(
                "positive_proposition_authored_by_user_id", sa.String(36),
                sa.ForeignKey("users.id", ondelete="SET NULL"), nullable=True,
            ))
    if "positive_proposition_authored_at" not in axis_columns:
        op.add_column(
            _AXIS, sa.Column("positive_proposition_authored_at", sa.DateTime(), nullable=True)
        )
    for table in (_CANDIDATES, _REVIEWS):
        existing_columns = _columns(table)
        if "axis_alignment" not in existing_columns:
            op.add_column(table, sa.Column("axis_alignment", sa.String(16), nullable=True))
        if "axis_polarity" not in existing_columns:
            op.add_column(table, sa.Column("axis_polarity", sa.String(16), nullable=True))
        if "axis_positive_proposition_sha256" not in existing_columns:
            op.add_column(
                table,
                sa.Column("axis_positive_proposition_sha256", sa.String(64), nullable=True),
            )
    if bind.dialect.name == "sqlite":
        _install_sqlite_guards()
    else:
        if not _has_check(_AXIS, "ck_character_trait_axis_proposition_pair"):
            op.create_check_constraint(
            "ck_character_trait_axis_proposition_pair", _AXIS,
            "(positive_proposition IS NULL AND positive_proposition_sha256 IS NULL "
            "AND positive_proposition_authored_at IS NULL "
            "AND positive_proposition_authored_by_user_id IS NULL) "
            "OR (positive_proposition IS NOT NULL AND "
            "positive_proposition_sha256 IS NOT NULL AND "
            "positive_proposition_authored_at IS NOT NULL AND "
            "length(positive_proposition_sha256) = 64)",
            )
        for table in (_CANDIDATES, _REVIEWS):
            check_name = (
                "ck_character_trait_candidate_alignment_pair"
                if table == _CANDIDATES else "ck_character_trait_review_alignment_pair"
            )
            if not _has_check(table, check_name):
                op.create_check_constraint(
                check_name,
                table,
                "(axis_alignment IS NULL AND axis_polarity IS NULL "
                "AND axis_positive_proposition_sha256 IS NULL) OR "
                "(approved_axis_id IS NOT NULL AND axis_alignment IS NOT NULL "
                "AND axis_polarity IS NOT NULL AND axis_positive_proposition_sha256 IS NOT NULL "
                "AND axis_alignment IN ('same', 'opposite') "
                "AND axis_polarity IN ('positive', 'negative') "
                "AND length(axis_positive_proposition_sha256) = 64)",
                )
        if not _has_check(_CANDIDATES, "ck_character_trait_candidate_alignment_direction"):
            op.create_check_constraint(
            "ck_character_trait_candidate_alignment_direction", _CANDIDATES,
            "axis_alignment IS NULL OR ("
            "(polarity = 'positive' AND axis_alignment = 'same' AND axis_polarity = 'positive') "
            "OR (polarity = 'positive' AND axis_alignment = 'opposite' AND axis_polarity = 'negative') "
            "OR (polarity = 'negative' AND axis_alignment = 'same' AND axis_polarity = 'negative') "
            "OR (polarity = 'negative' AND axis_alignment = 'opposite' AND axis_polarity = 'positive'))",
            )
        op.execute(sa.text(
            "CREATE OR REPLACE FUNCTION guard_character_trait_axis_proposition_immutable() "
            "RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
            "IF OLD.positive_proposition IS NOT NULL OR "
            "OLD.positive_proposition_sha256 IS NOT NULL OR "
            "OLD.positive_proposition_authored_at IS NOT NULL THEN "
            "IF NEW.positive_proposition IS DISTINCT FROM OLD.positive_proposition "
            "OR NEW.positive_proposition_sha256 IS DISTINCT FROM "
            "OLD.positive_proposition_sha256 OR "
            "NEW.positive_proposition_authored_at IS DISTINCT FROM "
            "OLD.positive_proposition_authored_at THEN "
            "RAISE EXCEPTION 'character_trait_axis_proposition_immutable' "
            "USING ERRCODE = '23514'; END IF; END IF; RETURN NEW; END; $$"
        ))
        op.execute(sa.text(
            "CREATE TRIGGER tr_character_trait_axis_proposition_immutable "
            "BEFORE UPDATE ON character_trait_axes FOR EACH ROW "
            "EXECUTE FUNCTION guard_character_trait_axis_proposition_immutable()"
        ))


def downgrade() -> None:
    bind = op.get_bind()
    if bind.execute(sa.text(
        "SELECT 1 FROM character_trait_axes WHERE positive_proposition IS NOT NULL "
        "OR positive_proposition_authored_at IS NOT NULL LIMIT 1"
    )).first() or bind.execute(sa.text(
        "SELECT 1 FROM character_trait_reviews WHERE decision = 'align' LIMIT 1"
    )).first() or bind.execute(sa.text(
        "SELECT 1 FROM character_trait_candidates WHERE axis_alignment IS NOT NULL LIMIT 1"
    )).first():
        raise RuntimeError("cannot downgrade 0018 while author direction decisions exist")
    if bind.dialect.name == "sqlite":
        for action in ("INSERT", "UPDATE"):
            for table in (_AXIS, _CANDIDATES, _REVIEWS):
                op.execute(sa.text(f"DROP TRIGGER IF EXISTS {_sqlite_guard(table, action)}"))
        op.execute(sa.text("DROP TRIGGER IF EXISTS tr_character_trait_axis_proposition_immutable"))
    else:
        op.execute(sa.text(
            "DROP TRIGGER IF EXISTS tr_character_trait_axis_proposition_immutable "
            "ON character_trait_axes"
        ))
        op.execute(sa.text(
            "DROP FUNCTION IF EXISTS guard_character_trait_axis_proposition_immutable()"
        ))
        op.drop_constraint("ck_character_trait_axis_proposition_pair", _AXIS, type_="check")
        op.drop_constraint(
            "ck_character_trait_candidate_alignment_direction", _CANDIDATES,
            type_="check",
        )
        for table in (_CANDIDATES, _REVIEWS):
            op.drop_constraint(
                f"ck_character_trait_{'candidate' if table == _CANDIDATES else 'review'}_alignment_pair",
                table, type_="check",
            )
    for table in (_REVIEWS, _CANDIDATES):
        new_columns = (
            "axis_positive_proposition_sha256", "axis_polarity", "axis_alignment"
        )
        new_checks = (
            ("ck_character_trait_review_alignment_pair",)
            if table == _REVIEWS else (
                "ck_character_trait_candidate_alignment_pair",
                "ck_character_trait_candidate_alignment_direction",
            )
        )
        present_checks = tuple(name for name in new_checks if _has_check(table, name))
        if bind.dialect.name == "sqlite" and present_checks:
            with op.batch_alter_table(
                table, recreate="always", reflect_kwargs={"resolve_fks": False}
            ) as batch:
                for name in present_checks:
                    batch.drop_constraint(name, type_="check")
                for name in new_columns:
                    batch.drop_column(name)
            _restore_sqlite_axis_guards(table)
        else:
            for name in new_columns:
                op.drop_column(table, name)
    axis_columns = (
        "positive_proposition_authored_at",
        "positive_proposition_authored_by_user_id",
        "positive_proposition_sha256",
        "positive_proposition",
    )
    if bind.dialect.name == "sqlite" and _has_check(
        _AXIS, "ck_character_trait_axis_proposition_pair"
    ):
        # The referenced table's temporary absence during batch replacement
        # invalidates 0015 cross-table guards. Drop only those exact known
        # triggers, then restore them and the axis identity immutability guard.
        for table in (_CANDIDATES, _REVIEWS):
            for action in ("insert", "update"):
                op.execute(sa.text(
                    f"DROP TRIGGER IF EXISTS fk_{table}_axis_project_{action}"
                ))
        with op.batch_alter_table(
            _AXIS, recreate="always", reflect_kwargs={"resolve_fks": False}
        ) as batch:
            batch.drop_constraint(
                "ck_character_trait_axis_proposition_pair", type_="check"
            )
            for name in axis_columns:
                batch.drop_column(name)
        _restore_sqlite_axis_identity_immutability()
        for table in (_CANDIDATES, _REVIEWS):
            _restore_sqlite_axis_guards(table)
    else:
        for name in axis_columns:
            op.drop_column(_AXIS, name)
    _replace_review_decision_check(expanded=False)
    if bind.dialect.name == "sqlite":
        _restore_sqlite_axis_guards(_REVIEWS)
