"""Make embedding-space deployment and input transforms explicit.

Revision ID: 0003_embedding_identity
Revises: 0002_evidence_substrate

Legacy vectors cannot be proven compatible with the stronger profile identity,
so this migration invalidates only the reproducible embedding/profile cache.
Narrative source documents, runs, issues and feedback are never touched.
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0003_embedding_identity"
down_revision = "0002_evidence_substrate"
branch_labels = None
depends_on = None


_NEW_COLUMNS = {
    "deployment_fingerprint",
    "document_transform_identity",
    "query_transform_identity",
}
_OLD_UNIQUE = (
    "provider_kind",
    "provider_namespace",
    "model_identifier",
    "model_revision",
    "dimensions",
    "normalized",
)
_NEW_UNIQUE = (
    "provider_kind",
    "provider_namespace",
    "model_identifier",
    "model_revision",
    "deployment_fingerprint",
    "document_transform_identity",
    "query_transform_identity",
    "dimensions",
    "normalized",
)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {item["name"] for item in inspector.get_columns("embedding_profiles")}
    present = columns & _NEW_COLUMNS
    if present and present != _NEW_COLUMNS:
        raise RuntimeError("incompatible pre-existing table: embedding_profiles")
    if present == _NEW_COLUMNS:
        if _NEW_UNIQUE not in _unique_column_tuples(bind, "embedding_profiles"):
            raise RuntimeError("incompatible pre-existing table: embedding_profiles")
        return

    # Both tables are derived, reproducible caches. Deleting them before the
    # identity change prevents a legacy emb-* hash from being relabelled as a
    # stronger deployment/transform identity.
    op.execute(sa.text("DELETE FROM evidence_embeddings"))
    op.execute(sa.text("DELETE FROM embedding_profiles"))
    dialect = bind.dialect.name
    if dialect == "sqlite":
        _recreate_sqlite_tables(stronger_identity=True)
    else:
        with op.batch_alter_table("embedding_profiles") as batch:
            _alter_profile_identity(batch)


def downgrade() -> None:
    bind = op.get_bind()
    op.execute(sa.text("DELETE FROM evidence_embeddings"))
    op.execute(sa.text("DELETE FROM embedding_profiles"))
    dialect = bind.dialect.name
    if dialect == "sqlite":
        _recreate_sqlite_tables(stronger_identity=False)
    else:
        with op.batch_alter_table("embedding_profiles") as batch:
            _restore_legacy_identity(batch)


def _alter_profile_identity(batch) -> None:
    batch.drop_constraint("uq_embedding_profile_identity", type_="unique")
    batch.add_column(sa.Column("deployment_fingerprint", sa.String(160), nullable=False))
    batch.add_column(
        sa.Column("document_transform_identity", sa.String(80), nullable=False)
    )
    batch.add_column(sa.Column("query_transform_identity", sa.String(80), nullable=False))
    batch.create_unique_constraint("uq_embedding_profile_identity", list(_NEW_UNIQUE))


def _restore_legacy_identity(batch) -> None:
    batch.drop_constraint("uq_embedding_profile_identity", type_="unique")
    batch.drop_column("query_transform_identity")
    batch.drop_column("document_transform_identity")
    batch.drop_column("deployment_fingerprint")
    batch.create_unique_constraint("uq_embedding_profile_identity", list(_OLD_UNIQUE))


def _recreate_sqlite_tables(*, stronger_identity: bool) -> None:
    # SQLite cannot reliably drop a reflected unnamed UNIQUE constraint in a
    # batch migration. Both tables are empty derived caches at this point, so
    # rebuilding only these two tables is deterministic and leaves source and
    # analysis tables untouched.
    op.drop_index("ix_evidence_embeddings_profile_id", table_name="evidence_embeddings")
    op.drop_table("evidence_embeddings")
    op.drop_table("embedding_profiles")
    profile_columns = [
        sa.Column("id", sa.String(68), primary_key=True, nullable=False),
        sa.Column("provider_kind", sa.String(40), nullable=False),
        sa.Column("provider_namespace", sa.String(80), nullable=False),
        sa.Column("model_identifier", sa.String(255), nullable=False),
        sa.Column("model_revision", sa.String(120), nullable=False),
    ]
    if stronger_identity:
        profile_columns.extend(
            [
                sa.Column("deployment_fingerprint", sa.String(160), nullable=False),
                sa.Column(
                    "document_transform_identity", sa.String(80), nullable=False
                ),
                sa.Column("query_transform_identity", sa.String(80), nullable=False),
            ]
        )
    profile_columns.extend(
        [
            sa.Column("dimensions", sa.Integer(), nullable=False),
            sa.Column("normalized", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.CheckConstraint(
                "dimensions > 0 AND dimensions <= 16000",
                name="ck_embedding_profile_dimensions",
            ),
            sa.UniqueConstraint(
                *(_NEW_UNIQUE if stronger_identity else _OLD_UNIQUE),
                name="uq_embedding_profile_identity",
            ),
        ]
    )
    op.create_table("embedding_profiles", *profile_columns)
    op.create_table(
        "evidence_embeddings",
        sa.Column(
            "chunk_id",
            sa.String(68),
            sa.ForeignKey("evidence_chunks.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "profile_id",
            sa.String(68),
            sa.ForeignKey("embedding_profiles.id", ondelete="CASCADE"),
            primary_key=True,
            nullable=False,
        ),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("vector", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint(
            "dimensions > 0 AND dimensions <= 16000",
            name="ck_evidence_embedding_dimensions",
        ),
    )
    op.create_index(
        "ix_evidence_embeddings_profile_id",
        "evidence_embeddings",
        ["profile_id"],
        unique=False,
    )


def _unique_column_tuples(bind, table_name: str) -> set[tuple[str, ...]]:
    inspector = sa.inspect(bind)
    values = {
        tuple(item.get("column_names") or ())
        for item in inspector.get_unique_constraints(table_name)
    }
    values.update(
        tuple(item.get("column_names") or ())
        for item in inspector.get_indexes(table_name)
        if item.get("unique")
    )
    return values
