"""Project metadata revisions and successful multipart upload receipts.

Revision ID: 0024_project_import_foundation
Revises: 0023_major_ooc_dimensions
"""
from __future__ import annotations

import re

import sqlalchemy as sa
from alembic import op


revision = "0024_project_import_foundation"
down_revision = "0023_major_ooc_dimensions"
branch_labels = None
depends_on = None


def metadata_revision_column():
    # A column-level CHECK lets SQLite add this non-null/defaulted column in
    # place, preserving all existing project foreign keys, indexes and data.
    return sa.Column(
        "metadata_revision", sa.Integer(),
        sa.CheckConstraint("metadata_revision > 0", name="ck_projects_metadata_revision"),
        nullable=False, server_default=sa.text("1"),
    )


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "metadata_revision" not in {column["name"] for column in inspector.get_columns("projects")}:
        op.add_column("projects", metadata_revision_column())
    if not inspector.has_table("document_upload_receipts"):
        op.create_table(
            "document_upload_receipts",
            sa.Column("id", sa.String(36), primary_key=True),
            sa.Column("project_id", sa.String(36), sa.ForeignKey("projects.id", ondelete="RESTRICT"), nullable=False),
            sa.Column("idempotency_key", sa.String(128), nullable=False),
            sa.Column("request_sha256", sa.String(64), nullable=False),
            sa.Column("document_id", sa.String(36), nullable=False),
            sa.Column("superseded_document_ids", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint("project_id", "idempotency_key", name="uq_document_upload_receipt_project_key"),
            sa.ForeignKeyConstraint(
                ["project_id", "document_id"], ["documents.project_id", "documents.id"],
                name="fk_document_upload_receipt_project_document", ondelete="RESTRICT",
            ),
            sa.CheckConstraint("length(idempotency_key) BETWEEN 1 AND 128", name="ck_document_upload_receipt_key_length"),
            sa.CheckConstraint("length(request_sha256) = 64", name="ck_document_upload_receipt_request_hash"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    has_receipts = inspector.has_table("document_upload_receipts")
    has_revision = "metadata_revision" in {column["name"] for column in inspector.get_columns("projects")}
    if ((has_receipts and bind.execute(sa.text("SELECT 1 FROM document_upload_receipts LIMIT 1")).first())
            or (has_revision and bind.execute(sa.text("SELECT 1 FROM projects WHERE metadata_revision > 1 LIMIT 1")).first())):
        # No DDL before this guard: dropping successful operation receipts or
        # edited revisions could make retries unexpectedly create new versions.
        raise RuntimeError("project import state exists; downgrade requires an explicit backup and controlled rollback")
    if has_revision and bind.dialect.name == "sqlite":
        ddl = bind.execute(sa.text("SELECT sql FROM sqlite_master WHERE type='table' AND name='projects'")).scalar_one()
        version = tuple(int(part) for part in bind.execute(sa.text("SELECT sqlite_version()")).scalar_one().split("."))
        inline_check = re.search(
            r"\bmetadata_revision\s+INTEGER\s+DEFAULT\s+1\s+NOT\s+NULL\s+"
            r"CONSTRAINT\s+ck_projects_metadata_revision\s+CHECK\s*\(metadata_revision\s*>\s*0\)",
            ddl, re.IGNORECASE,
        )
        if inline_check and version >= (3, 35, 0):
            # This revision's own column-level CHECK leaves with its column.
            # DROP COLUMN is in-place; never batch-recreate the projects parent.
            op.drop_column("projects", "metadata_revision")
        # create_all/adopted schemas can put the CHECK at table level. Without
        # a proven in-place drop, retain the harmless additive default-1 column
        # rather than copy the parent table. Old code ignores it; re-upgrade
        # adopts it. This also supports older SQLite without DROP COLUMN.
    elif has_revision:
        op.drop_constraint("ck_projects_metadata_revision", "projects", type_="check")
        op.drop_column("projects", "metadata_revision")
    if has_receipts:
        op.drop_table("document_upload_receipts")
