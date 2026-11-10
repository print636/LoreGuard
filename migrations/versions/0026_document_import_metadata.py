"""Record actual upload identity separately from the retained logical name."""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0026_document_import_metadata"
down_revision = "0025_character_observations"
branch_labels = None
depends_on = None


def upgrade() -> None:
    if "import_metadata" not in {c["name"] for c in sa.inspect(op.get_bind()).get_columns("document_upload_receipts")}:
        op.add_column("document_upload_receipts", sa.Column("import_metadata", sa.JSON(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    if "import_metadata" not in {c["name"] for c in sa.inspect(bind).get_columns("document_upload_receipts")}:
        return
    if bind.execute(sa.text("SELECT 1 FROM document_upload_receipts WHERE import_metadata IS NOT NULL LIMIT 1")).first():
        raise RuntimeError("document import metadata exists; downgrade requires an explicit backup and controlled rollback")
    if bind.dialect.name == "sqlite":
        version = tuple(int(p) for p in bind.execute(sa.text("SELECT sqlite_version()")).scalar_one().split("."))
        if version < (3, 35, 0):
            # Retain an unused nullable column rather than recreate this table.
            return
    op.drop_column("document_upload_receipts", "import_metadata")
