"""Store author-scoped value and behavior-boundary comparison axes.

Revision ID: 0020_value_boundary_author_axes
Revises: 0019_project_name_sort_key
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0020_value_boundary_author_axes"
down_revision = "0019_project_name_sort_key"
branch_labels = None
depends_on = None


_TABLE = "character_trait_axes"
_TYPE_NAME = "ck_character_trait_axis_type"
_SCOPE_NAME = "ck_character_trait_axis_object_scope"
_OLD_UNIQUE = "uq_character_trait_axis_definition"
_CORE_INDEX = "uq_character_trait_axis_core_definition"
_SCOPED_INDEX = "uq_character_trait_axis_scoped_definition"
_OLD_TYPE = "trait_type = 'core_personality'"
_NEW_TYPE = "trait_type IN ('core_personality', 'value', 'behavior_boundary')"
_OBJECT_SCOPE = (
    "(trait_type = 'core_personality' AND comparison_key IS NULL "
    "AND applicability_scope IS NULL AND applicability_scope_sha256 IS NULL) "
    "OR (trait_type IN ('value', 'behavior_boundary') "
    "AND comparison_key IS NOT NULL "
    "AND substr(comparison_key, 1, length(trait_type) + 1) = trait_type || ':' "
    "AND length(comparison_key) > length(trait_type) + 1 "
    "AND length(comparison_key) <= 200 "
    "AND applicability_scope IS NOT NULL "
    "AND length(applicability_scope) BETWEEN 1 AND 200 "
    "AND applicability_scope_sha256 IS NOT NULL "
    "AND length(applicability_scope_sha256) = 64 "
    "AND positive_proposition IS NOT NULL "
    "AND positive_proposition_sha256 IS NOT NULL "
    "AND positive_proposition_authored_at IS NOT NULL)"
)
_COLUMNS = (
    sa.Column("comparison_key", sa.String(200), nullable=True),
    sa.Column("applicability_scope", sa.String(200), nullable=True),
    sa.Column("applicability_scope_sha256", sa.String(64), nullable=True),
)


def _scoped_unique_indexes() -> None:
    core = sa.text("trait_type = 'core_personality'")
    scoped = sa.text("trait_type IN ('value', 'behavior_boundary')")
    op.create_index(
        _CORE_INDEX, _TABLE, ["project_id", "definition_sha256"],
        unique=True, sqlite_where=core, postgresql_where=core,
    )
    op.create_index(
        _SCOPED_INDEX, _TABLE,
        ["project_id", "trait_type", "definition_sha256", "comparison_key",
         "applicability_scope_sha256"],
        unique=True, sqlite_where=scoped, postgresql_where=scoped,
    )


def _sqlite_cross_table_guards(*, install: bool) -> None:
    # SQLite validates trigger SQL while renaming the replacement axis table.
    # Temporarily remove only the two 0015 guards that reference that table.
    for table in ("character_trait_candidates", "character_trait_reviews"):
        for action in ("INSERT", "UPDATE"):
            name = f"fk_{table}_axis_project_{action.lower()}"
            if not install:
                op.execute(sa.text(f"DROP TRIGGER IF EXISTS {name}"))
            else:
                op.execute(sa.text(
                    f"CREATE TRIGGER {name} BEFORE {action} ON {table} "
                    "WHEN NEW.approved_axis_id IS NOT NULL AND NOT EXISTS "
                    "(SELECT 1 FROM character_trait_axes "
                    "WHERE id = NEW.approved_axis_id AND "
                    "project_id = NEW.project_id) "
                    "BEGIN SELECT RAISE(ABORT, "
                    "'fk_character_trait_axis_project'); END"
                ))


def _sqlite_identity_guard(*, expanded: bool) -> None:
    identity_columns = (
        "id, project_id, trait_type, version, display_name, definition, "
        "definition_sha256, created_at"
    )
    if expanded:
        identity_columns += ", comparison_key, applicability_scope, applicability_scope_sha256"
    op.execute(sa.text(
        "CREATE TRIGGER tr_character_trait_axes_immutable "
        f"BEFORE UPDATE OF {identity_columns} ON {_TABLE} "
        "BEGIN SELECT RAISE(ABORT, 'character_trait_axis_immutable'); END"
    ))


def _sqlite_guards(*, expanded: bool) -> None:
    # Recreating the axis table drops its triggers, including the v15/v18
    # immutability and proposition guards. Reinstall all on the replacement.
    _sqlite_identity_guard(expanded=expanded)
    op.execute(sa.text(
        "CREATE TRIGGER tr_character_trait_axis_proposition_immutable "
        "BEFORE UPDATE OF positive_proposition, positive_proposition_sha256, "
        f"positive_proposition_authored_at ON {_TABLE} "
        "WHEN OLD.positive_proposition IS NOT NULL OR "
        "OLD.positive_proposition_sha256 IS NOT NULL OR "
        "OLD.positive_proposition_authored_at IS NOT NULL "
        "BEGIN SELECT RAISE(ABORT, 'character_trait_axis_proposition_immutable'); END"
    ))
    for action in ("INSERT", "UPDATE"):
        op.execute(sa.text(
            f"CREATE TRIGGER tr_character_trait_axes_axis_alignment_{action.lower()} "
            f"BEFORE {action} ON {_TABLE} "
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


def _postgres_identity_guard(*, expanded: bool) -> None:
    immutable = (
        "id", "project_id", "trait_type", "version", "display_name",
        "definition", "definition_sha256", "created_at",
    )
    if expanded:
        immutable += (
            "comparison_key", "applicability_scope", "applicability_scope_sha256",
        )
    changed = " OR ".join(
        f"NEW.{column} IS DISTINCT FROM OLD.{column}" for column in immutable
    )
    op.execute(sa.text(
        "CREATE OR REPLACE FUNCTION guard_character_trait_axis_immutable() "
        "RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN "
        f"IF {changed} THEN "
        "RAISE EXCEPTION 'character_trait_axis_immutable' "
        "USING ERRCODE = '23514'; END IF; RETURN NEW; END; $$"
    ))


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    existing_columns = {item["name"] for item in inspector.get_columns(_TABLE)}
    existing_checks = {
        item["name"]: item.get("sqltext") or ""
        for item in inspector.get_check_constraints(_TABLE)
    }
    existing_indexes = {item["name"] for item in inspector.get_indexes(_TABLE)}
    existing_uniques = {
        item["name"] for item in inspector.get_unique_constraints(_TABLE)
    }
    new_column_names = {column.name for column in _COLUMNS}
    if new_column_names <= existing_columns and _SCOPE_NAME in existing_checks:
        if (
            "behavior_boundary" not in existing_checks.get(_TYPE_NAME, "")
            or not {_CORE_INDEX, _SCOPED_INDEX} <= existing_indexes
            or _OLD_UNIQUE in existing_uniques
        ):
            raise RuntimeError("character axis identity is incompatible with 0020")
        # Direct Base.metadata.create_all adoption already has the 0020
        # columns/checks. Earlier revisions installed the old identity guard;
        # expand it to cover the already-present immutable object/scope fields.
        if op.get_bind().dialect.name == "sqlite":
            op.execute(sa.text("DROP TRIGGER IF EXISTS tr_character_trait_axes_immutable"))
            _sqlite_identity_guard(expanded=True)
        else:
            _postgres_identity_guard(expanded=True)
        return
    if new_column_names & existing_columns or _SCOPE_NAME in existing_checks:
        raise RuntimeError("character axis schema is partially migrated to 0020")
    if op.get_bind().dialect.name == "sqlite":
        # Alembic's migration connection leaves foreign_keys off for this
        # referenced-table copy. Existing rows, indices and FK declarations
        # are reflected; no candidate/review/snapshot table is rewritten.
        _sqlite_cross_table_guards(install=False)
        with op.batch_alter_table(
            _TABLE, recreate="always", reflect_kwargs={"resolve_fks": False}
        ) as batch:
            batch.drop_constraint(_TYPE_NAME, type_="check")
            batch.drop_constraint(_OLD_UNIQUE, type_="unique")
            for column in _COLUMNS:
                batch.add_column(column)
            batch.create_check_constraint(_TYPE_NAME, _NEW_TYPE)
            batch.create_check_constraint(_SCOPE_NAME, _OBJECT_SCOPE)
        _scoped_unique_indexes()
        _sqlite_guards(expanded=True)
        _sqlite_cross_table_guards(install=True)
    else:
        for column in _COLUMNS:
            op.add_column(_TABLE, column)
        op.drop_constraint(_TYPE_NAME, _TABLE, type_="check")
        op.drop_constraint(_OLD_UNIQUE, _TABLE, type_="unique")
        op.create_check_constraint(_TYPE_NAME, _TABLE, _NEW_TYPE)
        op.create_check_constraint(_SCOPE_NAME, _TABLE, _OBJECT_SCOPE)
        _scoped_unique_indexes()
        _postgres_identity_guard(expanded=True)


def downgrade() -> None:
    if op.get_bind().execute(sa.text(
        "SELECT 1 FROM character_trait_axes "
        "WHERE trait_type IN ('value', 'behavior_boundary') LIMIT 1"
    )).first():
        raise RuntimeError("cannot downgrade 0020 while value/boundary axes exist")
    op.drop_index(_SCOPED_INDEX, table_name=_TABLE)
    op.drop_index(_CORE_INDEX, table_name=_TABLE)
    if op.get_bind().dialect.name == "sqlite":
        _sqlite_cross_table_guards(install=False)
        with op.batch_alter_table(
            _TABLE, recreate="always", reflect_kwargs={"resolve_fks": False}
        ) as batch:
            batch.drop_constraint(_SCOPE_NAME, type_="check")
            batch.drop_constraint(_TYPE_NAME, type_="check")
            for column in _COLUMNS:
                batch.drop_column(column.name)
            batch.create_check_constraint(_TYPE_NAME, _OLD_TYPE)
            batch.create_unique_constraint(
                _OLD_UNIQUE, ["project_id", "trait_type", "definition_sha256"]
            )
        _sqlite_guards(expanded=False)
        _sqlite_cross_table_guards(install=True)
    else:
        op.drop_constraint(_SCOPE_NAME, _TABLE, type_="check")
        op.drop_constraint(_TYPE_NAME, _TABLE, type_="check")
        op.create_check_constraint(_TYPE_NAME, _TABLE, _OLD_TYPE)
        op.create_unique_constraint(
            _OLD_UNIQUE, _TABLE, ["project_id", "trait_type", "definition_sha256"]
        )
        _postgres_identity_guard(expanded=False)
        for column in _COLUMNS:
            op.drop_column(_TABLE, column.name)
