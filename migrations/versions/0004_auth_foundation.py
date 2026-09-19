"""Add users, workspaces and opaque authentication sessions.

Revision ID: 0004_auth_foundation
Revises: 0003_embedding_identity
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0004_auth_foundation"
down_revision = "0003_embedding_identity"
branch_labels = None
depends_on = None


_EXPECTED = {
    "users": {"id", "email", "display_name", "password_hash", "is_active", "created_at"},
    "workspaces": {"id", "name", "kind", "created_at"},
    "workspace_members": {"id", "workspace_id", "user_id", "role", "created_at"},
    "auth_sessions": {
        "id",
        "user_id",
        "token_hash",
        "csrf_hash",
        "created_at",
        "expires_at",
        "last_seen_at",
        "revoked_at",
    },
}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing = set(inspector.get_table_names())
    for table_name, expected_columns in _EXPECTED.items():
        if table_name not in existing:
            continue
        actual = {item["name"] for item in inspector.get_columns(table_name)}
        if actual != expected_columns:
            raise RuntimeError(f"incompatible pre-existing table: {table_name}")

    if "users" not in existing:
        op.create_table(
            "users",
            sa.Column("id", sa.String(36), primary_key=True, nullable=False),
            sa.Column("email", sa.String(320), nullable=False),
            sa.Column("display_name", sa.String(80), nullable=False),
            sa.Column("password_hash", sa.String(512), nullable=False),
            sa.Column("is_active", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
        op.create_index("ix_users_email", "users", ["email"], unique=True)
    if "workspaces" not in existing:
        op.create_table(
            "workspaces",
            sa.Column("id", sa.String(36), primary_key=True, nullable=False),
            sa.Column("name", sa.String(120), nullable=False),
            sa.Column("kind", sa.String(24), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
    if "workspace_members" not in existing:
        op.create_table(
            "workspace_members",
            sa.Column("id", sa.String(36), primary_key=True, nullable=False),
            sa.Column(
                "workspace_id",
                sa.String(36),
                sa.ForeignKey("workspaces.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("role", sa.String(24), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "workspace_id", "user_id", name="uq_workspace_member"
            ),
        )
        op.create_index(
            "ix_workspace_members_workspace_id",
            "workspace_members",
            ["workspace_id"],
            unique=False,
        )
        op.create_index(
            "ix_workspace_members_user_id",
            "workspace_members",
            ["user_id"],
            unique=False,
        )
    if "auth_sessions" not in existing:
        op.create_table(
            "auth_sessions",
            sa.Column("id", sa.String(36), primary_key=True, nullable=False),
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("token_hash", sa.String(64), nullable=False),
            sa.Column("csrf_hash", sa.String(64), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
            sa.Column("last_seen_at", sa.DateTime(), nullable=False),
            sa.Column("revoked_at", sa.DateTime(), nullable=True),
        )
        op.create_index(
            "ix_auth_sessions_user_id", "auth_sessions", ["user_id"], unique=False
        )
        op.create_index(
            "ix_auth_sessions_token_hash",
            "auth_sessions",
            ["token_hash"],
            unique=True,
        )
        op.create_index(
            "ix_auth_sessions_expires_at",
            "auth_sessions",
            ["expires_at"],
            unique=False,
        )


def downgrade() -> None:
    op.drop_table("auth_sessions")
    op.drop_table("workspace_members")
    op.drop_table("workspaces")
    op.drop_table("users")
