"""Persist a locale-independent project-name sort key for catalog ordering.

Revision ID: 0019_project_name_sort_key
Revises: 0018_character_axis_direction
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0019_project_name_sort_key"
down_revision = "0018_character_axis_direction"
branch_labels = None
depends_on = None


_INDEX = "ix_projects_workspace_name_sort_id"
_BATCH_SIZE = 500


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "name_sort_key" not in {
        column["name"] for column in inspector.get_columns("projects")
    }:
        # SQLite can add a nullable BLOB in place; making it NOT NULL would
        # require copying projects and its existing foreign-key relationships.
        # Application writes populate the key for every new project.
        op.add_column(
            "projects", sa.Column("name_sort_key", sa.LargeBinary(), nullable=True)
        )

    from app.project_sort import project_name_sort_key_v1

    update = sa.text(
        "UPDATE projects SET name_sort_key = :sort_key WHERE id = :project_id"
    ).bindparams(sa.bindparam("sort_key", type_=sa.LargeBinary()))
    while True:
        rows = bind.execute(
            sa.text(
                "SELECT id, name FROM projects WHERE name_sort_key IS NULL "
                "ORDER BY id LIMIT :batch_size"
            ),
            {"batch_size": _BATCH_SIZE},
        ).all()
        if not rows:
            break
        bind.execute(
            update,
            [
                {
                    "project_id": row.id,
                    "sort_key": project_name_sort_key_v1(row.name),
                }
                for row in rows
            ],
        )

    if bind.execute(
        sa.text("SELECT 1 FROM projects WHERE name_sort_key IS NULL LIMIT 1")
    ).first():
        raise RuntimeError("project name sort key backfill is incomplete")

    if _INDEX not in {
        index["name"] for index in sa.inspect(bind).get_indexes("projects")
    }:
        op.create_index(
            _INDEX,
            "projects",
            ["workspace_id", "name_sort_key", "id"],
        )


def downgrade() -> None:
    bind = op.get_bind()
    if _INDEX in {
        index["name"] for index in sa.inspect(bind).get_indexes("projects")
    }:
        op.drop_index(_INDEX, table_name="projects")
    if "name_sort_key" in {
        column["name"] for column in sa.inspect(bind).get_columns("projects")
    }:
        op.drop_column("projects", "name_sort_key")
