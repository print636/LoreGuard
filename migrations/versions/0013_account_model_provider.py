"""Add encrypted account-owned chat-provider revisions and run bindings.

Revision ID: 0013_account_model_provider
Revises: 0012_context_inference
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0013_account_model_provider"
down_revision = "0012_context_inference"
branch_labels = None
depends_on = None


def _columns(bind, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    if not inspector.has_table(table_name):
        return set()
    return {row["name"] for row in inspector.get_columns(table_name)}


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


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("account_provider_configs"):
        op.create_table(
            "account_provider_configs",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("revision", sa.Integer(), nullable=False),
            sa.Column("base_url", sa.String(2048), nullable=False),
            sa.Column("model_name", sa.String(255), nullable=False),
            sa.Column("endpoint_sha256", sa.String(64), nullable=False),
            sa.Column("secret_ciphertext", sa.LargeBinary(), nullable=True),
            sa.Column("secret_nonce", sa.LargeBinary(), nullable=True),
            sa.Column("encryption_key_id", sa.String(64), nullable=True),
            sa.Column(
                "secret_schema_version",
                sa.Integer(),
                nullable=False,
                server_default="1",
            ),
            sa.Column(
                "state",
                sa.String(24),
                nullable=False,
                server_default="usable",
            ),
            sa.Column(
                "created_by_user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("superseded_at", sa.DateTime(), nullable=True),
            sa.Column("revoked_at", sa.DateTime(), nullable=True),
            sa.Column("secret_rewrapped_at", sa.DateTime(), nullable=True),
            sa.Column("last_tested_at", sa.DateTime(), nullable=True),
            sa.Column("last_test_status", sa.String(24), nullable=True),
            sa.Column("last_test_category", sa.String(80), nullable=True),
            sa.Column("last_test_payload", sa.JSON(), nullable=True),
            sa.UniqueConstraint(
                "user_id", "revision", name="uq_account_provider_config_revision"
            ),
            sa.CheckConstraint(
                "revision > 0", name="ck_account_provider_config_revision"
            ),
            sa.CheckConstraint(
                "state IN ('usable', 'superseded', 'revoked', 'scrubbed')",
                name="ck_account_provider_config_state",
            ),
            sa.CheckConstraint(
                "length(endpoint_sha256) = 64",
                name="ck_account_provider_config_endpoint_hash",
            ),
        )
        op.create_index(
            "ix_account_provider_configs_user_id",
            "account_provider_configs",
            ["user_id"],
        )
        op.create_index(
            "ix_account_provider_configs_state",
            "account_provider_configs",
            ["state"],
        )

    inspector = sa.inspect(bind)
    if not inspector.has_table("account_provider_bindings"):
        op.create_table(
            "account_provider_bindings",
            sa.Column(
                "user_id",
                sa.String(36),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "current_config_id",
                sa.String(36),
                sa.ForeignKey("account_provider_configs.id", ondelete="SET NULL"),
                nullable=True,
            ),
            sa.Column(
                "lock_version", sa.Integer(), nullable=False, server_default="0"
            ),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "current_config_id", name="uq_account_provider_binding_current"
            ),
        )

    if _columns(bind, "analysis_runs"):
        columns = _columns(bind, "analysis_runs")
        with op.batch_alter_table("analysis_runs") as batch:
            if "provider_config_id" not in columns:
                batch.add_column(
                    sa.Column("provider_config_id", sa.String(36), nullable=True)
                )
            if "provider_identity" not in columns:
                batch.add_column(sa.Column("provider_identity", sa.JSON(), nullable=True))
        if not _foreign_key_exists(bind, "analysis_runs", "provider_config_id"):
            with op.batch_alter_table("analysis_runs") as batch:
                batch.create_foreign_key(
                    "fk_analysis_runs_provider_config_id",
                    "account_provider_configs",
                    ["provider_config_id"],
                    ["id"],
                    ondelete="SET NULL",
                )
        if not _index_exists(bind, "analysis_runs", "ix_analysis_runs_provider_config_id"):
            op.create_index(
                "ix_analysis_runs_provider_config_id",
                "analysis_runs",
                ["provider_config_id"],
            )

    table = "document_narrative_context_revisions"
    if _columns(bind, table):
        columns = _columns(bind, table)
        with op.batch_alter_table(table) as batch:
            if "inference_provider_config_id" not in columns:
                batch.add_column(
                    sa.Column(
                        "inference_provider_config_id", sa.String(36), nullable=True
                    )
                )
            if "inference_provider_identity" not in columns:
                batch.add_column(
                    sa.Column("inference_provider_identity", sa.JSON(), nullable=True)
                )
        if not _foreign_key_exists(bind, table, "inference_provider_config_id"):
            with op.batch_alter_table(table) as batch:
                batch.create_foreign_key(
                    "fk_narrative_context_inference_provider_config",
                    "account_provider_configs",
                    ["inference_provider_config_id"],
                    ["id"],
                    ondelete="SET NULL",
                )
        index_name = "ix_document_narrative_context_inference_provider_config_id"
        if not _index_exists(bind, table, index_name):
            op.create_index(
                index_name, table, ["inference_provider_config_id"], unique=False
            )


def downgrade() -> None:
    bind = op.get_bind()
    table = "document_narrative_context_revisions"
    if "inference_provider_config_id" in _columns(bind, table):
        with op.batch_alter_table(table) as batch:
            batch.drop_index(
                "ix_document_narrative_context_inference_provider_config_id"
            )
            batch.drop_constraint(
                "fk_narrative_context_inference_provider_config",
                type_="foreignkey",
            )
            batch.drop_column("inference_provider_identity")
            batch.drop_column("inference_provider_config_id")
    if "provider_config_id" in _columns(bind, "analysis_runs"):
        with op.batch_alter_table("analysis_runs") as batch:
            batch.drop_index("ix_analysis_runs_provider_config_id")
            batch.drop_constraint(
                "fk_analysis_runs_provider_config_id", type_="foreignkey"
            )
            batch.drop_column("provider_identity")
            batch.drop_column("provider_config_id")
    inspector = sa.inspect(bind)
    if inspector.has_table("account_provider_bindings"):
        op.drop_table("account_provider_bindings")
    inspector = sa.inspect(bind)
    if inspector.has_table("account_provider_configs"):
        op.drop_index(
            "ix_account_provider_configs_state",
            table_name="account_provider_configs",
        )
        op.drop_index(
            "ix_account_provider_configs_user_id",
            table_name="account_provider_configs",
        )
        op.drop_table("account_provider_configs")
