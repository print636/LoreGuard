"""The additive receipt column preserves historical operations and story data."""
from datetime import datetime

import pytest
from alembic import command
from sqlalchemy import MetaData, Table, create_engine, inspect, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from app.db import DocumentUploadReceiptRow, LOCAL_WORKSPACE_ID
from app.document_upload_receipts import import_metadata
from app.project_sort import project_name_sort_key
from tests.test_project_import_foundation_migration import _config


def test_upgrade_keeps_old_null_receipt_and_refuses_loss_of_new_metadata(tmp_path):
    url = f"sqlite:///{(tmp_path / 'old-receipt.db').as_posix()}"
    config = _config(url)
    previous, current = "0025_character_observations", "0026_document_import_metadata"
    command.upgrade(config, previous)
    engine = create_engine(url)
    metadata = MetaData()
    tables = {name: Table(name, metadata, autoload_with=engine)
              for name in ("projects", "documents", "document_upload_receipts")}
    when = datetime(2026, 11, 1)
    with engine.begin() as db:
        db.exec_driver_sql("PRAGMA foreign_keys=ON")
        db.execute(tables["projects"].insert(), {"id": "project", "workspace_id": LOCAL_WORKSPACE_ID,
                   "name": "旧项目", "name_sort_key": project_name_sort_key("旧项目"), "description": "", "created_at": when})
        db.execute(tables["documents"].insert(), [{"id": f"doc-{version}", "project_id": "project", "name": "old.md",
                   "content": f"正文{version}", "version": version, "active": version == 2, "created_at": when} for version in (1, 2)])
        db.execute(tables["document_upload_receipts"].insert(), {"id": "old-receipt", "project_id": "project",
                   "idempotency_key": "old-operation", "request_sha256": "a" * 64, "document_id": "doc-2",
                   "superseded_document_ids": ["doc-1"], "created_at": when})
        before = {name: db.execute(select(table).order_by(*table.primary_key.columns)).all() for name, table in tables.items()}
    try:
        command.upgrade(config, current)
        with engine.connect() as db:
            assert db.exec_driver_sql("SELECT import_metadata FROM document_upload_receipts").scalar_one() is None
            assert before == {name: db.execute(select(table).order_by(*table.primary_key.columns)).all() for name, table in tables.items()}
            assert db.exec_driver_sql("PRAGMA foreign_key_check").all() == []
        # A nullable, unused additive field can be removed without touching rows.
        command.downgrade(config, previous)
        with engine.connect() as db:
            assert "import_metadata" not in {c["name"] for c in inspect(db).get_columns("document_upload_receipts")}
            assert before == {name: db.execute(select(table).order_by(*table.primary_key.columns)).all() for name, table in tables.items()}
        command.upgrade(config, current)
        receipt = Table("document_upload_receipts", MetaData(), autoload_with=engine)
        actual = import_metadata(filename="renamed.md", logical_name="old.md", replaces_document_id="doc-1", replaces_document_version=1)
        with engine.begin() as db:
            db.execute(receipt.update().values(import_metadata=actual))
        with pytest.raises(RuntimeError, match="explicit backup"):
            command.downgrade(config, previous)
        with engine.connect() as db:
            assert db.execute(select(receipt.c.import_metadata)).scalar_one() == actual
            assert db.exec_driver_sql("SELECT version_num FROM alembic_version").scalar_one() == current
            assert before == {name: db.execute(select(table).order_by(*table.primary_key.columns)).all() for name, table in tables.items()}
            assert db.exec_driver_sql("PRAGMA quick_check").scalar_one() == "ok"
    finally:
        engine.dispose()


def test_postgresql_receipt_column_is_nullable_and_keeps_existing_constraints():
    sql = str(CreateTable(DocumentUploadReceiptRow.__table__).compile(dialect=postgresql.dialect()))
    assert "import_metadata JSON" in sql and "import_metadata JSON NOT NULL" not in sql
    assert "UNIQUE (project_id, idempotency_key)" in sql
    assert "FOREIGN KEY(project_id, document_id)" in sql
