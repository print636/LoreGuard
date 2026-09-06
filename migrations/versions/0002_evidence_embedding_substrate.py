"""Add the opt-in Evidence RAG storage substrate.

Revision ID: 0002_evidence_substrate
Revises: 0001_legacy_schema_baseline
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector


revision = "0002_evidence_substrate"
down_revision = "0001_legacy_schema_baseline"
branch_labels = None
depends_on = None


_EXPECTED_TABLES = {
    "embedding_profiles": (
        {
            "id", "provider_kind", "provider_namespace", "model_identifier",
            "model_revision", "dimensions", "normalized", "created_at",
        },
        {"id"},
    ),
    "evidence_chunks": (
        {
            "id", "project_id", "document_id", "document_version", "content_sha256",
            "chunker_version", "ordinal", "text", "text_sha256", "char_start",
            "char_end", "line_start", "line_end", "created_at",
        },
        {"id"},
    ),
    "evidence_embeddings": (
        {"chunk_id", "profile_id", "dimensions", "vector", "created_at"},
        {"chunk_id", "profile_id"},
    ),
}


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name
    existing = set(sa.inspect(bind).get_table_names())
    _validate_existing_tables(bind, existing, dialect)
    needs_owner_fk = (
        "evidence_chunks" in existing and not _has_document_owner_fk(bind)
    )
    if needs_owner_fk:
        _validate_existing_chunk_ownership(bind)
    _ensure_document_owner_unique(bind)
    if dialect == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    if "embedding_profiles" not in existing:
        op.create_table(
            "embedding_profiles",
            sa.Column("id", sa.String(68), primary_key=True, nullable=False),
            sa.Column("provider_kind", sa.String(40), nullable=False),
            sa.Column("provider_namespace", sa.String(80), nullable=False),
            sa.Column("model_identifier", sa.String(255), nullable=False),
            sa.Column("model_revision", sa.String(120), nullable=False),
            sa.Column("dimensions", sa.Integer(), nullable=False),
            sa.Column("normalized", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.CheckConstraint(
                "dimensions > 0 AND dimensions <= 16000",
                name="ck_embedding_profile_dimensions",
            ),
            sa.UniqueConstraint(
                "provider_kind",
                "provider_namespace",
                "model_identifier",
                "model_revision",
                "dimensions",
                "normalized",
                name="uq_embedding_profile_identity",
            ),
        )
    if "evidence_chunks" not in existing:
        op.create_table(
            "evidence_chunks",
            sa.Column("id", sa.String(68), primary_key=True, nullable=False),
            sa.Column(
                "project_id", sa.String(36), sa.ForeignKey("projects.id"), nullable=False
            ),
            sa.Column("document_id", sa.String(36), nullable=False),
            sa.ForeignKeyConstraint(
                ["project_id", "document_id"],
                ["documents.project_id", "documents.id"],
                name="fk_evidence_chunk_document_owner",
            ),
            sa.Column("document_version", sa.Integer(), nullable=False),
            sa.Column("content_sha256", sa.String(64), nullable=False),
            sa.Column("chunker_version", sa.String(80), nullable=False),
            sa.Column("ordinal", sa.Integer(), nullable=False),
            sa.Column("text", sa.Text(), nullable=False),
            sa.Column("text_sha256", sa.String(64), nullable=False),
            sa.Column("char_start", sa.Integer(), nullable=False),
            sa.Column("char_end", sa.Integer(), nullable=False),
            sa.Column("line_start", sa.Integer(), nullable=False),
            sa.Column("line_end", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.CheckConstraint("document_version > 0", name="ck_evidence_chunk_version"),
            sa.CheckConstraint("ordinal >= 0", name="ck_evidence_chunk_ordinal"),
            sa.CheckConstraint("line_start > 0", name="ck_evidence_chunk_line_start"),
            sa.CheckConstraint("line_end >= line_start", name="ck_evidence_chunk_line_end"),
            sa.CheckConstraint("char_start >= 0", name="ck_evidence_chunk_char_start"),
            sa.CheckConstraint("char_end > char_start", name="ck_evidence_chunk_char_end"),
            sa.CheckConstraint(
                "length(content_sha256) = 64", name="ck_evidence_chunk_content_hash"
            ),
            sa.CheckConstraint(
                "length(text_sha256) = 64", name="ck_evidence_chunk_text_hash"
            ),
            sa.UniqueConstraint(
                "project_id",
                "document_id",
                "document_version",
                "content_sha256",
                "chunker_version",
                "ordinal",
                name="uq_evidence_chunk_snapshot_ordinal",
            ),
        )
    if "evidence_embeddings" not in existing:
        vector_type = Vector() if dialect == "postgresql" else sa.JSON()
        op.create_table(
            "evidence_embeddings",
            sa.Column(
                "chunk_id",
                sa.String(68),
                sa.ForeignKey("evidence_chunks.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column(
                "profile_id",
                sa.String(68),
                sa.ForeignKey("embedding_profiles.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("dimensions", sa.Integer(), nullable=False),
            sa.Column("vector", vector_type, nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.CheckConstraint(
                "dimensions > 0 AND dimensions <= 16000",
                name="ck_evidence_embedding_dimensions",
            ),
        )

    if needs_owner_fk:
        _add_document_owner_fk(dialect)
    if not _has_document_owner_fk(bind):
        raise RuntimeError("failed to establish evidence chunk document ownership")

    _ensure_index("evidence_chunks", "ix_evidence_chunks_project_id", ["project_id"])
    _ensure_index("evidence_chunks", "ix_evidence_chunks_document_id", ["document_id"])
    _ensure_index("evidence_embeddings", "ix_evidence_embeddings_profile_id", ["profile_id"])


def downgrade() -> None:
    # Existing create_all databases may already own these additive tables.
    # Preserve them rather than making a downgrade unexpectedly destructive.
    pass


def _ensure_index(table_name: str, index_name: str, columns: list[str]) -> None:
    indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table_name)}
    if index_name not in indexes:
        op.create_index(index_name, table_name, columns, unique=False)


def _validate_existing_tables(bind, existing: set[str], dialect: str) -> None:
    inspector = sa.inspect(bind)
    for table_name, (required_columns, required_pk) in _EXPECTED_TABLES.items():
        if table_name not in existing:
            continue
        actual_columns = {item["name"] for item in inspector.get_columns(table_name)}
        actual_pk = set(
            inspector.get_pk_constraint(table_name).get("constrained_columns") or ()
        )
        if not required_columns.issubset(actual_columns) or actual_pk != required_pk:
            raise RuntimeError(f"incompatible pre-existing table: {table_name}")

    required_uniques = {
        "embedding_profiles": {
            (
                "provider_kind",
                "provider_namespace",
                "model_identifier",
                "model_revision",
                "dimensions",
                "normalized",
            ),
            (
                "provider_kind",
                "provider_namespace",
                "model_identifier",
                "model_revision",
                "deployment_fingerprint",
                "document_transform_identity",
                "query_transform_identity",
                "dimensions",
                "normalized",
            ),
        },
        "evidence_chunks": {(
            "project_id",
            "document_id",
            "document_version",
            "content_sha256",
            "chunker_version",
            "ordinal",
        )},
    }
    for table_name, accepted_uniques in required_uniques.items():
        if table_name not in existing:
            continue
        available = _unique_column_tuples(bind, table_name)
        if not available.intersection(accepted_uniques):
            raise RuntimeError(f"incompatible pre-existing table: {table_name}")

    if "evidence_chunks" in existing:
        # The only supported pre-final WIP shape lacked the composite owner FK,
        # but did have both original single-column ownership references.  This
        # keeps adoption narrow rather than blessing an arbitrary lookalike.
        if not _has_document_owner_fk(bind) and not (
            _has_foreign_key(
                inspector,
                "evidence_chunks",
                ("project_id",),
                "projects",
                ("id",),
            )
            and _has_foreign_key(
                inspector,
                "evidence_chunks",
                ("document_id",),
                "documents",
                ("id",),
            )
        ):
            raise RuntimeError("incompatible pre-existing table: evidence_chunks")

    if "evidence_embeddings" in existing:
        if not _has_foreign_key(
            inspector,
            "evidence_embeddings",
            ("chunk_id",),
            "evidence_chunks",
            ("id",),
        ) or not _has_foreign_key(
            inspector,
            "evidence_embeddings",
            ("profile_id",),
            "embedding_profiles",
            ("id",),
        ):
            raise RuntimeError("incompatible pre-existing table: evidence_embeddings")
        if dialect == "postgresql":
            vector_type = bind.execute(
                sa.text(
                    "SELECT udt_name FROM information_schema.columns "
                    "WHERE table_schema = current_schema() "
                    "AND table_name = 'evidence_embeddings' AND column_name = 'vector'"
                )
            ).scalar_one_or_none()
            if vector_type != "vector":
                raise RuntimeError("incompatible pre-existing table: evidence_embeddings")
        elif dialect == "sqlite":
            reflected_vector = next(
                item["type"]
                for item in inspector.get_columns("evidence_embeddings")
                if item["name"] == "vector"
            )
            if not isinstance(reflected_vector, sa.JSON):
                raise RuntimeError("incompatible pre-existing table: evidence_embeddings")


def _has_document_owner_fk(bind) -> bool:
    inspector = sa.inspect(bind)
    if "evidence_chunks" not in inspector.get_table_names():
        return False
    return any(
        tuple(item.get("constrained_columns") or ()) == ("project_id", "document_id")
        and item.get("referred_table") == "documents"
        and tuple(item.get("referred_columns") or ()) == ("project_id", "id")
        for item in inspector.get_foreign_keys("evidence_chunks")
    )


def _has_foreign_key(
    inspector,
    table_name: str,
    local_columns: tuple[str, ...],
    referred_table: str,
    referred_columns: tuple[str, ...],
) -> bool:
    return any(
        tuple(item.get("constrained_columns") or ()) == local_columns
        and item.get("referred_table") == referred_table
        and tuple(item.get("referred_columns") or ()) == referred_columns
        for item in inspector.get_foreign_keys(table_name)
    )


def _validate_existing_chunk_ownership(bind) -> None:
    orphan_count = bind.execute(
        sa.text(
            "SELECT count(*) FROM evidence_chunks AS ec "
            "LEFT JOIN documents AS d "
            "ON d.project_id = ec.project_id AND d.id = ec.document_id "
            "WHERE d.id IS NULL"
        )
    ).scalar_one()
    if orphan_count:
        raise RuntimeError("incompatible evidence chunk document ownership")


def _ensure_document_owner_unique(bind) -> None:
    expected = ("project_id", "id")
    available = _unique_column_tuples(bind, "documents")
    if expected not in available:
        op.create_index(
            "uq_documents_project_id_id",
            "documents",
            ["project_id", "id"],
            unique=True,
        )


def _add_document_owner_fk(dialect: str) -> None:
    if dialect == "sqlite":
        with op.batch_alter_table("evidence_chunks", recreate="always") as batch_op:
            batch_op.create_foreign_key(
                "fk_evidence_chunk_document_owner",
                "documents",
                ["project_id", "document_id"],
                ["project_id", "id"],
            )
        return
    op.create_foreign_key(
        "fk_evidence_chunk_document_owner",
        "evidence_chunks",
        "documents",
        ["project_id", "document_id"],
        ["project_id", "id"],
    )


def _unique_column_tuples(bind, table_name: str) -> set[tuple[str, ...]]:
    inspector = sa.inspect(bind)
    result = {
        tuple(item.get("column_names") or ())
        for item in inspector.get_unique_constraints(table_name)
    }
    result.update(
        tuple(item.get("column_names") or ())
        for item in inspector.get_indexes(table_name)
        if item.get("unique")
    )
    if bind.dialect.name == "sqlite":
        for index_row in bind.exec_driver_sql(
            f'PRAGMA index_list("{table_name}")'
        ).mappings():
            if not index_row["unique"]:
                continue
            index_name = str(index_row["name"]).replace('"', '""')
            columns = bind.exec_driver_sql(
                f'PRAGMA index_info("{index_name}")'
            ).mappings()
            result.add(tuple(row["name"] for row in columns))
    return result
