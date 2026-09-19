"""Bind user-facing resources to workspaces and actors.

Revision ID: 0005_workspace_isolation
Revises: 0004_auth_foundation

Legacy product rows are assigned to the fixed local workspace/user.  Actor
columns remain nullable deliberately so future account deletion can preserve
run and feedback audit history through ``ON DELETE SET NULL``.
"""
from __future__ import annotations

from datetime import datetime

import sqlalchemy as sa
from alembic import op


revision = "0005_workspace_isolation"
down_revision = "0004_auth_foundation"
branch_labels = None
depends_on = None


LOCAL_USER_ID = "00000000-0000-0000-0000-000000000001"
LOCAL_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"
LOCAL_MEMBERSHIP_ID = "00000000-0000-0000-0000-000000000001"


def _columns(bind, table_name: str) -> dict[str, dict]:
    return {
        column["name"]: column
        for column in sa.inspect(bind).get_columns(table_name)
    }


def _table_exists(bind, table_name: str) -> bool:
    return sa.inspect(bind).has_table(table_name)


def _foreign_key_exists(bind, table_name: str, local_column: str) -> bool:
    return any(
        tuple(item.get("constrained_columns") or ()) == (local_column,)
        for item in sa.inspect(bind).get_foreign_keys(table_name)
    )


def _index_exists(bind, table_name: str, index_name: str) -> bool:
    return any(
        item.get("name") == index_name
        for item in sa.inspect(bind).get_indexes(table_name)
    )


def _seed_local_identity(bind) -> None:
    now = datetime.now()
    if bind.execute(
        sa.text("SELECT 1 FROM users WHERE id = :id"), {"id": LOCAL_USER_ID}
    ).first() is None:
        bind.execute(
            sa.text(
                "INSERT INTO users "
                "(id, email, display_name, password_hash, is_active, created_at) "
                "VALUES (:id, :email, :display_name, :password_hash, :active, :created_at)"
            ),
            {
                "id": LOCAL_USER_ID,
                "email": "local@loreguard.invalid",
                "display_name": "本地体验用户",
                "password_hash": "!anonymous-local-account",
                "active": True,
                "created_at": now,
            },
        )
    if bind.execute(
        sa.text("SELECT 1 FROM workspaces WHERE id = :id"),
        {"id": LOCAL_WORKSPACE_ID},
    ).first() is None:
        bind.execute(
            sa.text(
                "INSERT INTO workspaces (id, name, kind, created_at) "
                "VALUES (:id, :name, :kind, :created_at)"
            ),
            {
                "id": LOCAL_WORKSPACE_ID,
                "name": "本地工作区",
                "kind": "personal",
                "created_at": now,
            },
        )
    if bind.execute(
        sa.text(
            "SELECT 1 FROM workspace_members "
            "WHERE workspace_id = :workspace_id AND user_id = :user_id"
        ),
        {"workspace_id": LOCAL_WORKSPACE_ID, "user_id": LOCAL_USER_ID},
    ).first() is None:
        bind.execute(
            sa.text(
                "INSERT INTO workspace_members "
                "(id, workspace_id, user_id, role, created_at) "
                "VALUES (:id, :workspace_id, :user_id, :role, :created_at)"
            ),
            {
                "id": LOCAL_MEMBERSHIP_ID,
                "workspace_id": LOCAL_WORKSPACE_ID,
                "user_id": LOCAL_USER_ID,
                "role": "owner",
                "created_at": now,
            },
        )


def upgrade() -> None:
    bind = op.get_bind()
    _seed_local_identity(bind)

    project_columns = _columns(bind, "projects")
    if "workspace_id" not in project_columns:
        with op.batch_alter_table("projects") as batch:
            batch.add_column(sa.Column("workspace_id", sa.String(36), nullable=True))
    bind.execute(
        sa.text(
            "UPDATE projects SET workspace_id = :workspace_id "
            "WHERE workspace_id IS NULL"
        ),
        {"workspace_id": LOCAL_WORKSPACE_ID},
    )
    project_columns = _columns(bind, "projects")
    needs_project_fk = not _foreign_key_exists(bind, "projects", "workspace_id")
    if project_columns["workspace_id"].get("nullable", True) or needs_project_fk:
        with op.batch_alter_table("projects") as batch:
            batch.alter_column(
                "workspace_id",
                existing_type=sa.String(36),
                nullable=False,
            )
            if needs_project_fk:
                batch.create_foreign_key(
                    "fk_projects_workspace_id",
                    "workspaces",
                    ["workspace_id"],
                    ["id"],
                    ondelete="RESTRICT",
                )
    if not _index_exists(bind, "projects", "ix_projects_workspace_id"):
        op.create_index(
            "ix_projects_workspace_id", "projects", ["workspace_id"], unique=False
        )

    if _table_exists(bind, "analysis_runs"):
        run_columns = _columns(bind, "analysis_runs")
        if "requested_by_user_id" not in run_columns:
            with op.batch_alter_table("analysis_runs") as batch:
                batch.add_column(
                    sa.Column("requested_by_user_id", sa.String(36), nullable=True)
                )
        bind.execute(
            sa.text(
                "UPDATE analysis_runs SET requested_by_user_id = :user_id "
                "WHERE requested_by_user_id IS NULL"
            ),
            {"user_id": LOCAL_USER_ID},
        )
        if not _foreign_key_exists(bind, "analysis_runs", "requested_by_user_id"):
            with op.batch_alter_table("analysis_runs") as batch:
                batch.create_foreign_key(
                    "fk_analysis_runs_requested_by_user_id",
                    "users",
                    ["requested_by_user_id"],
                    ["id"],
                    ondelete="SET NULL",
                )
        if not _index_exists(
            bind, "analysis_runs", "ix_analysis_runs_requested_by_user_id"
        ):
            op.create_index(
                "ix_analysis_runs_requested_by_user_id",
                "analysis_runs",
                ["requested_by_user_id"],
                unique=False,
            )

    if _table_exists(bind, "issue_feedback"):
        feedback_columns = _columns(bind, "issue_feedback")
        if "created_by_user_id" not in feedback_columns:
            with op.batch_alter_table("issue_feedback") as batch:
                batch.add_column(
                    sa.Column("created_by_user_id", sa.String(36), nullable=True)
                )
        bind.execute(
            sa.text(
                "UPDATE issue_feedback SET created_by_user_id = :user_id "
                "WHERE created_by_user_id IS NULL"
            ),
            {"user_id": LOCAL_USER_ID},
        )
        if not _foreign_key_exists(bind, "issue_feedback", "created_by_user_id"):
            with op.batch_alter_table("issue_feedback") as batch:
                batch.create_foreign_key(
                    "fk_issue_feedback_created_by_user_id",
                    "users",
                    ["created_by_user_id"],
                    ["id"],
                    ondelete="SET NULL",
                )
        if not _index_exists(
            bind, "issue_feedback", "ix_issue_feedback_created_by_user_id"
        ):
            op.create_index(
                "ix_issue_feedback_created_by_user_id",
                "issue_feedback",
                ["created_by_user_id"],
                unique=False,
            )


def downgrade() -> None:
    with op.batch_alter_table("issue_feedback") as batch:
        batch.drop_index("ix_issue_feedback_created_by_user_id")
        batch.drop_constraint(
            "fk_issue_feedback_created_by_user_id", type_="foreignkey"
        )
        batch.drop_column("created_by_user_id")
    with op.batch_alter_table("analysis_runs") as batch:
        batch.drop_index("ix_analysis_runs_requested_by_user_id")
        batch.drop_constraint(
            "fk_analysis_runs_requested_by_user_id", type_="foreignkey"
        )
        batch.drop_column("requested_by_user_id")
    with op.batch_alter_table("projects") as batch:
        batch.drop_index("ix_projects_workspace_id")
        batch.drop_constraint("fk_projects_workspace_id", type_="foreignkey")
        batch.drop_column("workspace_id")
