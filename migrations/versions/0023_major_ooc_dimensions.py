"""Add relationship-attitude and motivation-goal trait dimensions.

Revision ID: 0023_major_ooc_dimensions
Revises: 0022_character_formal_proof
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0023_major_ooc_dimensions"
down_revision = "0022_character_formal_proof"
branch_labels = None
depends_on = None


_TABLE = "character_trait_candidates"
_TYPE_NAME = "ck_character_trait_candidate_type"
_IDENTITY_NAME = "ck_character_trait_candidate_object_identity"
_NEW_DIMENSIONS = ("relationship_attitude", "motivation_goal")
_OLD_TYPE = (
    "trait_type IN ('core_personality', 'preference', 'value', "
    "'speech_pattern', 'behavior_boundary', 'contextual_behavior', "
    "'current_state')"
)
_NEW_TYPE = (
    "trait_type IN ('core_personality', 'preference', 'value', "
    "'relationship_attitude', 'motivation_goal', 'speech_pattern', "
    "'behavior_boundary', 'contextual_behavior', 'current_state')"
)
_OBJECT_IDENTITY = (
    "trait_type NOT IN ('relationship_attitude', 'motivation_goal') OR ("
    "key_object IS NOT NULL AND LENGTH(TRIM(key_object)) > 0 "
    "AND comparison_key IS NOT NULL "
    "AND comparison_key LIKE trait_type || ':%' "
    "AND LENGTH(comparison_key) <= 160)"
)


def _checks() -> dict[str, str]:
    return {
        item["name"]: str(item.get("sqltext") or "")
        for item in sa.inspect(op.get_bind()).get_check_constraints(_TABLE)
        if item.get("name")
    }


def _sqlite_triggers() -> tuple[str, ...]:
    rows = op.get_bind().exec_driver_sql(
        "SELECT sql FROM sqlite_master "
        "WHERE type = 'trigger' AND tbl_name = ? AND sql IS NOT NULL "
        "ORDER BY name",
        (_TABLE,),
    ).all()
    return tuple(row[0] for row in rows)


def _replace_checks(*, expanded: bool) -> None:
    bind = op.get_bind()
    checks = _checks()
    columns = {
        item["name"] for item in sa.inspect(bind).get_columns(_TABLE)
    }
    if _TYPE_NAME not in checks:
        raise RuntimeError(f"{_TABLE} is missing {_TYPE_NAME}")

    if bind.dialect.name == "sqlite":
        # Alembic's SQLite batch copy does not carry standalone triggers to
        # the replacement table. Preserve every trigger installed on this
        # table, including guards that are independent of this revision.
        triggers = _sqlite_triggers()
        with op.batch_alter_table(
            _TABLE, recreate="always", reflect_kwargs={"resolve_fks": False}
        ) as batch:
            if _IDENTITY_NAME in checks:
                batch.drop_constraint(_IDENTITY_NAME, type_="check")
            batch.drop_constraint(_TYPE_NAME, type_="check")
            if expanded and "key_object" not in columns:
                batch.add_column(
                    sa.Column("key_object", sa.String(80), nullable=True)
                )
            batch.create_check_constraint(
                _TYPE_NAME, _NEW_TYPE if expanded else _OLD_TYPE
            )
            if expanded:
                batch.create_check_constraint(_IDENTITY_NAME, _OBJECT_IDENTITY)
            elif "key_object" in columns:
                batch.drop_column("key_object")
        for trigger_sql in triggers:
            bind.exec_driver_sql(trigger_sql)
        return

    if expanded and "key_object" not in columns:
        op.add_column(
            _TABLE, sa.Column("key_object", sa.String(80), nullable=True)
        )
    if _IDENTITY_NAME in checks:
        op.drop_constraint(_IDENTITY_NAME, _TABLE, type_="check")
    op.drop_constraint(_TYPE_NAME, _TABLE, type_="check")
    op.create_check_constraint(
        _TYPE_NAME, _TABLE, _NEW_TYPE if expanded else _OLD_TYPE
    )
    if expanded:
        op.create_check_constraint(_IDENTITY_NAME, _TABLE, _OBJECT_IDENTITY)
    elif "key_object" in columns:
        op.drop_column(_TABLE, "key_object")


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table(_TABLE):
        return

    checks = _checks()
    if _IDENTITY_NAME in checks:
        columns = {
            item["name"] for item in inspector.get_columns(_TABLE)
        }
        type_sql = checks.get(_TYPE_NAME, "")
        identity_sql = checks[_IDENTITY_NAME].lower()
        if (
            all(value in type_sql for value in _NEW_DIMENSIONS)
            and "key_object" in columns
            and "comparison_key" in identity_sql
            and "key_object" in identity_sql
            and "trim" in identity_sql
            and "160" in identity_sql
        ):
            # Current ORM schemas can be created before Alembic adopts them.
            return
        raise RuntimeError("character trait dimension checks are incompatible with 0023")

    _replace_checks(expanded=True)


def downgrade() -> None:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table(_TABLE):
        return
    if bind.execute(sa.text(
        "SELECT 1 FROM character_trait_candidates "
        "WHERE trait_type IN ('relationship_attitude', 'motivation_goal') LIMIT 1"
    )).first():
        raise RuntimeError(
            "cannot downgrade 0023 while relationship-attitude or "
            "motivation-goal candidates exist"
        )
    _replace_checks(expanded=False)
