"""Add immutable author-approved personality axes and nullable bindings.

Revision ID: 0015_character_trait_axes
Revises: 0014_trait_comparison_key
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0015_character_trait_axes"
down_revision = "0014_trait_comparison_key"
branch_labels = None
depends_on = None


def _columns(bind, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    if not inspector.has_table(table_name):
        return set()
    return {item["name"] for item in inspector.get_columns(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    if not sa.inspect(bind).has_table("character_trait_axes"):
        op.create_table(
            "character_trait_axes",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "project_id",
                sa.String(36),
                sa.ForeignKey("projects.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("trait_type", sa.String(32), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("display_name", sa.String(80), nullable=False),
            sa.Column("definition", sa.String(200), nullable=False),
            sa.Column("definition_sha256", sa.String(64), nullable=False),
            sa.Column(
                "created_by_user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "id", "project_id", name="uq_character_trait_axis_project_identity"
            ),
            sa.UniqueConstraint(
                "project_id", "trait_type", "definition_sha256",
                name="uq_character_trait_axis_definition",
            ),
            sa.CheckConstraint(
                "trait_type = 'core_personality'",
                name="ck_character_trait_axis_type",
            ),
            sa.CheckConstraint("version = 1", name="ck_character_trait_axis_version"),
            sa.CheckConstraint(
                "length(definition_sha256) = 64",
                name="ck_character_trait_axis_definition_hash",
            ),
        )
        op.create_index(
            "ix_character_trait_axes_project_id",
            "character_trait_axes",
            ["project_id"],
        )
        op.create_index(
            "ix_character_trait_axes_created_by_user_id",
            "character_trait_axes",
            ["created_by_user_id"],
        )
        op.create_index(
            "ix_character_trait_axes_project_created",
            "character_trait_axes",
            ["project_id", "created_at"],
        )
    if bind.dialect.name == "sqlite":
        op.execute(
            sa.text(
                "CREATE TRIGGER IF NOT EXISTS tr_character_trait_axes_immutable "
                "BEFORE UPDATE OF id, project_id, trait_type, version, "
                "display_name, definition, definition_sha256, created_at "
                "ON character_trait_axes "
                "BEGIN SELECT RAISE(ABORT, 'character_trait_axis_immutable'); END"
            )
        )
    elif bind.dialect.name == "postgresql":
        op.execute(
            sa.text(
                "CREATE OR REPLACE FUNCTION guard_character_trait_axis_immutable() "
                "RETURNS trigger LANGUAGE plpgsql AS $$ "
                "BEGIN "
                "IF NEW.id IS DISTINCT FROM OLD.id "
                "OR NEW.project_id IS DISTINCT FROM OLD.project_id "
                "OR NEW.trait_type IS DISTINCT FROM OLD.trait_type "
                "OR NEW.version IS DISTINCT FROM OLD.version "
                "OR NEW.display_name IS DISTINCT FROM OLD.display_name "
                "OR NEW.definition IS DISTINCT FROM OLD.definition "
                "OR NEW.definition_sha256 IS DISTINCT FROM OLD.definition_sha256 "
                "OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN "
                "RAISE EXCEPTION 'character_trait_axis_immutable' "
                "USING ERRCODE = '23514'; "
                "END IF; "
                "RETURN NEW; "
                "END; $$"
            )
        )
        op.execute(
            sa.text(
                "DROP TRIGGER IF EXISTS tr_character_trait_axes_immutable "
                "ON character_trait_axes"
            )
        )
        op.execute(
            sa.text(
                "CREATE TRIGGER tr_character_trait_axes_immutable "
                "BEFORE UPDATE ON character_trait_axes FOR EACH ROW "
                "EXECUTE FUNCTION guard_character_trait_axis_immutable()"
            )
        )
    for table_name in ("character_trait_candidates", "character_trait_reviews"):
        columns = _columns(bind, table_name)
        if not columns:
            continue
        if "approved_axis_id" not in columns:
            if bind.dialect.name == "sqlite":
                # SQLite accepts a nullable REFERENCES column in ADD COLUMN,
                # while Alembic's separate ALTER CONSTRAINT is unsupported.
                op.execute(
                    sa.text(
                        f"ALTER TABLE {table_name} ADD COLUMN approved_axis_id "
                        "VARCHAR(36) REFERENCES character_trait_axes(id) ON DELETE RESTRICT"
                    )
                )
            else:
                op.add_column(
                    table_name,
                    sa.Column(
                        "approved_axis_id",
                        sa.String(36),
                        sa.ForeignKey("character_trait_axes.id", ondelete="RESTRICT"),
                        nullable=True,
                    ),
                )
        if "approved_axis_version" not in columns:
            op.add_column(
                table_name,
                sa.Column("approved_axis_version", sa.Integer(), nullable=True),
            )
        pair_rule = (
            "(approved_axis_id IS NULL AND approved_axis_version IS NULL) OR "
            "(approved_axis_id IS NOT NULL AND approved_axis_version = 1)"
        )
        pair_name = (
            "ck_character_trait_candidate_axis_pair"
            if table_name == "character_trait_candidates"
            else "ck_character_trait_review_axis_pair"
        )
        if bind.dialect.name == "sqlite":
            # Rebuilding a referenced SQLite table while FK enforcement is on
            # would disturb historical candidate/snapshot rows. Two triggers
            # enforce the same invariant without copying existing data.
            for action in ("INSERT", "UPDATE"):
                trigger_name = f"{pair_name}_{action.lower()}"
                op.execute(
                    sa.text(
                        f"CREATE TRIGGER IF NOT EXISTS {trigger_name} "
                        f"BEFORE {action} ON {table_name} "
                        "WHEN (NEW.approved_axis_id IS NULL AND "
                        "NEW.approved_axis_version IS NOT NULL) OR "
                        "(NEW.approved_axis_id IS NOT NULL AND "
                        "(NEW.approved_axis_version IS NULL OR "
                        "NEW.approved_axis_version != 1)) "
                        f"BEGIN SELECT RAISE(ABORT, '{pair_name}'); END"
                    )
                )
                project_trigger_name = f"fk_{table_name}_axis_project_{action.lower()}"
                op.execute(
                    sa.text(
                        f"CREATE TRIGGER IF NOT EXISTS {project_trigger_name} "
                        f"BEFORE {action} ON {table_name} "
                        "WHEN NEW.approved_axis_id IS NOT NULL AND NOT EXISTS "
                        "(SELECT 1 FROM character_trait_axes "
                        "WHERE id = NEW.approved_axis_id AND "
                        "project_id = NEW.project_id) "
                        "BEGIN SELECT RAISE(ABORT, "
                        "'fk_character_trait_axis_project'); END"
                    )
                )
        elif not any(
            item["name"] == pair_name
            for item in sa.inspect(bind).get_check_constraints(table_name)
        ):
            op.create_check_constraint(pair_name, table_name, pair_rule)
        if bind.dialect.name != "sqlite":
            project_fk_name = (
                "fk_character_trait_candidate_axis_project"
                if table_name == "character_trait_candidates"
                else "fk_character_trait_review_axis_project"
            )
            if not any(
                item.get("name") == project_fk_name
                for item in sa.inspect(bind).get_foreign_keys(table_name)
            ):
                op.create_foreign_key(
                    project_fk_name,
                    table_name,
                    "character_trait_axes",
                    ["approved_axis_id", "project_id"],
                    ["id", "project_id"],
                    ondelete="RESTRICT",
                )


def downgrade() -> None:
    bind = op.get_bind()
    if sa.inspect(bind).has_table("character_trait_axes"):
        if bind.dialect.name == "sqlite":
            op.execute(sa.text("DROP TRIGGER IF EXISTS tr_character_trait_axes_immutable"))
        elif bind.dialect.name == "postgresql":
            op.execute(
                sa.text(
                    "DROP TRIGGER IF EXISTS tr_character_trait_axes_immutable "
                    "ON character_trait_axes"
                )
            )
            op.execute(sa.text("DROP FUNCTION IF EXISTS guard_character_trait_axis_immutable()"))
    for table_name in ("character_trait_reviews", "character_trait_candidates"):
        columns = _columns(bind, table_name)
        pair_name = (
            "ck_character_trait_candidate_axis_pair"
            if table_name == "character_trait_candidates"
            else "ck_character_trait_review_axis_pair"
        )
        if bind.dialect.name == "sqlite":
            for action in ("INSERT", "UPDATE"):
                op.execute(sa.text(f"DROP TRIGGER IF EXISTS {pair_name}_{action.lower()}"))
                op.execute(
                    sa.text(
                        f"DROP TRIGGER IF EXISTS fk_{table_name}_axis_project_"
                        f"{action.lower()}"
                    )
                )
        elif columns and any(
            item["name"] == pair_name
            for item in sa.inspect(bind).get_check_constraints(table_name)
        ):
            op.drop_constraint(pair_name, table_name, type_="check")
        if bind.dialect.name != "sqlite" and columns:
            project_fk_name = (
                "fk_character_trait_candidate_axis_project"
                if table_name == "character_trait_candidates"
                else "fk_character_trait_review_axis_project"
            )
            if any(
                item.get("name") == project_fk_name
                for item in sa.inspect(bind).get_foreign_keys(table_name)
            ):
                op.drop_constraint(project_fk_name, table_name, type_="foreignkey")
        if bind.dialect.name == "sqlite" and (
            "approved_axis_version" in columns or "approved_axis_id" in columns
        ):
            # A later table rebuild can reify SQLite's original inline axis
            # reference as a table-level foreign key. DROP COLUMN then fails
            # because that key still names approved_axis_id. Recreate the
            # table while dropping both axis columns and their dependent key.
            with op.batch_alter_table(table_name, recreate="always") as batch:
                if "approved_axis_version" in columns:
                    batch.drop_column("approved_axis_version")
                if "approved_axis_id" in columns:
                    batch.drop_column("approved_axis_id")
        else:
            if "approved_axis_version" in columns:
                op.drop_column(table_name, "approved_axis_version")
            if "approved_axis_id" in columns:
                op.drop_column(table_name, "approved_axis_id")
    if sa.inspect(bind).has_table("character_trait_axes"):
        op.drop_table("character_trait_axes")
