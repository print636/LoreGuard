"""Additive upgrade checks on a synthetic pre-0024 database, never user data."""
from __future__ import annotations

import hashlib
import importlib.util
from datetime import datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from alembic.ddl.base import AddColumn
from sqlalchemy import MetaData, Table, create_engine, event, inspect, select, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable

from app.db import Base, DocumentUploadReceiptRow, LOCAL_WORKSPACE_ID
from app.project_sort import project_name_sort_key


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = "0023_major_ooc_dimensions"
HEAD = "0024_project_import_foundation"
WHEN = datetime(2026, 10, 1)


def _config(url):
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["database_url"] = url
    return config


def _revision():
    spec = importlib.util.spec_from_file_location("project_import_0024", ROOT / "migrations/versions/0024_project_metadata_upload_receipts.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_upgrade_preserves_all_old_identity_story_run_and_feedback_relationships(tmp_path):
    url = f"sqlite:///{(tmp_path / 'before0024.db').as_posix()}"
    config = _config(url)
    command.upgrade(config, PREVIOUS)
    engine = create_engine(url)
    metadata = MetaData()
    names = ("users", "workspaces", "workspace_members", "projects", "documents", "document_context",
             "analysis_runs", "analysis_run_inputs", "issues", "issue_feedback")
    tables = {name: Table(name, metadata, autoload_with=engine) for name in names}
    with engine.begin() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        values = {
            "users": {"id": "synthetic-user", "email": "author@synthetic.invalid", "display_name": "合成作者", "password_hash": "not-a-login-hash", "is_active": True, "created_at": WHEN},
            "workspaces": {"id": "synthetic-workspace", "name": "合成空间", "kind": "personal", "created_at": WHEN},
            "workspace_members": {"id": "synthetic-member", "workspace_id": "synthetic-workspace", "user_id": "synthetic-user", "role": "owner", "created_at": WHEN},
            "projects": {"id": "synthetic-project", "workspace_id": "synthetic-workspace", "name": "保全项目", "name_sort_key": project_name_sort_key("保全项目"), "description": "不得被升级改写", "created_at": WHEN},
            "analysis_runs": {"id": "synthetic-run", "project_id": "synthetic-project", "requested_by_user_id": "synthetic-user", "status": "completed", "created_at": WHEN, "started_at": WHEN, "completed_at": WHEN, "input_chars": 6, "prompt_tokens": 0, "completion_tokens": 0, "estimated_cost_usd": 0, "cancel_requested": False, "batch_mode": "draft_review", "sensitivity": "balanced", "batch_coverage": {"synthetic": True}},
            "analysis_run_inputs": {"id": "synthetic-input", "run_id": "synthetic-run", "document_id": "synthetic-doc-v1", "document_name": "chapter.md", "document_version": 1, "content": "冻结合成旧稿", "content_sha256": hashlib.sha256("冻结合成旧稿".encode()).hexdigest(), "ordinal": 0},
            "issues": {"id": "synthetic-issue", "run_id": "synthetic-run", "report_class": "formal", "category": "fact_conflict", "severity": "low", "confidence": 0.5, "title": "合成测试问题", "explanation": "仅验证关系保全", "evidence": [], "suggestion": "", "extra": {"synthetic": True}},
            "issue_feedback": {"id": "synthetic-feedback", "issue_id": "synthetic-issue", "created_by_user_id": "synthetic-user", "label": "accepted", "comment": "保留这个人工反馈", "created_at": WHEN},
        }
        for name in names:
            if name == "documents":
                connection.execute(tables[name].insert(), [
                    {"id": f"synthetic-doc-v{version}", "project_id": "synthetic-project", "name": "chapter.md", "content": body, "version": version, "active": version == 2, "created_at": WHEN}
                    for version, body in ((1, "冻结合成旧稿"), (2, "活动合成新稿"))
                ])
            elif name == "document_context":
                connection.execute(tables[name].insert(), [
                    {"document_id": f"synthetic-doc-v{version}", "document_role": "chapter", "story_scope": "global"}
                    for version in (1, 2)
                ])
            else:
                connection.execute(tables[name].insert(), values[name])
        before = {name: connection.execute(select(table).order_by(*table.primary_key.columns)).all() for name, table in tables.items()}
        indexes_before = inspect(connection).get_indexes("projects")
    engine.dispose()

    # Alembic opens its own engine; enforce FK checks on those connections too
    # during this upgrade, not only the fixture's verification connection.
    def enforce_foreign_keys(connection, _record):
        if connection.__class__.__module__ == "sqlite3":
            connection.execute("PRAGMA foreign_keys=ON")
    event.listen(Engine, "connect", enforce_foreign_keys)
    try:
        command.upgrade(config, HEAD)
        with engine.connect() as connection:
            after = {name: connection.execute(select(table).order_by(*table.primary_key.columns)).all() for name, table in tables.items()}
            assert before == after
            assert connection.execute(text("SELECT metadata_revision FROM projects WHERE id='synthetic-project'")).scalar_one() == 1
            assert connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one() == HEAD
            assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
            assert connection.exec_driver_sql("PRAGMA quick_check").scalar_one() == "ok"
            assert inspect(connection).get_indexes("projects") == indexes_before
            assert "document_upload_receipts" in inspect(connection).get_table_names()
        # An unused additive stage can round-trip in place, with FK enabled
        # and all old parent/child rows present. No table-copy is permitted.
        command.downgrade(config, PREVIOUS)
        with engine.connect() as connection:
            assert "metadata_revision" not in {column["name"] for column in inspect(connection).get_columns("projects")}
            assert "document_upload_receipts" not in inspect(connection).get_table_names()
            assert before == {name: connection.execute(select(table).order_by(*table.primary_key.columns)).all() for name, table in tables.items()}
            assert inspect(connection).get_indexes("projects") == indexes_before
            assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
        command.upgrade(config, HEAD)
        receipt = Table("document_upload_receipts", MetaData(), autoload_with=engine)
        with engine.begin() as connection:
            connection.execute(receipt.insert(), {"id": "synthetic-receipt", "project_id": "synthetic-project", "idempotency_key": "one-upload", "request_sha256": "a" * 64, "document_id": "synthetic-doc-v2", "superseded_document_ids": ["synthetic-doc-v1"], "created_at": WHEN})
        for statement in (
            "UPDATE projects SET metadata_revision=0 WHERE id='synthetic-project'",
            "UPDATE document_upload_receipts SET document_id='missing-document'",
            "UPDATE document_upload_receipts SET request_sha256='invalid'",
            "UPDATE document_upload_receipts SET idempotency_key=''",
            "INSERT INTO document_upload_receipts SELECT 'duplicate',project_id,idempotency_key,request_sha256,document_id,superseded_document_ids,created_at FROM document_upload_receipts",
        ):
            with pytest.raises(IntegrityError), engine.begin() as connection:
                connection.exec_driver_sql(statement)
        # Populated stage rollback is refused before any destructive DDL;
        # successful receipt and all old records survive the refusal.
        with pytest.raises(RuntimeError, match="explicit backup"):
            command.downgrade(config, PREVIOUS)
        with engine.connect() as connection:
            assert connection.execute(select(receipt.c.id)).all() == [("synthetic-receipt",)]
            assert before == {name: connection.execute(select(table).order_by(*table.primary_key.columns)).all() for name, table in tables.items()}
            assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
    finally:
        event.remove(Engine, "connect", enforce_foreign_keys)
        engine.dispose()


def test_adopted_create_all_schema_can_downgrade_without_parent_table_copy(tmp_path):
    url = f"sqlite:///{(tmp_path / 'adopted0024.db').as_posix()}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(text("INSERT INTO projects (id,workspace_id,name,name_sort_key,description,created_at) VALUES ('adopted-project',:workspace,'保留项目',:sort_key,'原简介',:created)"),
                           {"workspace": LOCAL_WORKSPACE_ID, "sort_key": project_name_sort_key("保留项目"), "created": WHEN})
    command.upgrade(_config(url), HEAD)
    command.downgrade(_config(url), PREVIOUS)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT name,description,metadata_revision FROM projects WHERE id='adopted-project'")).one() == ("保留项目", "原简介", 1)
        assert "document_upload_receipts" not in inspect(connection).get_table_names()
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
    command.upgrade(_config(url), HEAD)
    engine.dispose()


def test_postgresql_add_column_and_receipt_ddl_keep_scope_and_constraints():
    module = _revision()
    column_sql = str(AddColumn("projects", module.metadata_revision_column()).compile(dialect=postgresql.dialect()))
    assert "ALTER TABLE projects ADD COLUMN metadata_revision INTEGER DEFAULT 1 NOT NULL" in column_sql
    assert "CONSTRAINT ck_projects_metadata_revision CHECK (metadata_revision > 0)" in column_sql
    receipt_sql = str(CreateTable(DocumentUploadReceiptRow.__table__).compile(dialect=postgresql.dialect()))
    assert "UNIQUE (project_id, idempotency_key)" in receipt_sql
    assert "FOREIGN KEY(project_id, document_id) REFERENCES documents (project_id, id) ON DELETE RESTRICT" in receipt_sql
    assert "length(request_sha256) = 64" in receipt_sql
