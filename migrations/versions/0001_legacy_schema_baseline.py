"""Adopt or create the pre-Alembic application schema.

Revision ID: 0001_legacy_schema_baseline
Revises:
"""
from __future__ import annotations

from collections.abc import Callable

import sqlalchemy as sa
from alembic import op


revision = "0001_legacy_schema_baseline"
down_revision = None
branch_labels = None
depends_on = None


_EXPECTED_TABLES = {
    "projects": ({"id", "name", "description", "created_at"}, {"id"}),
    "documents": (
        {"id", "project_id", "name", "content", "version", "active", "created_at"},
        {"id"},
    ),
    "document_context": (
        {"document_id", "document_role", "story_scope"},
        {"document_id"},
    ),
    "analysis_runs": (
        {
            "id", "project_id", "status", "created_at", "started_at", "completed_at",
            "input_chars", "prompt_tokens", "completion_tokens", "estimated_cost_usd",
            "error", "cancel_requested",
        },
        {"id"},
    ),
    "analysis_run_inputs": (
        {
            "id", "run_id", "document_id", "document_name", "document_version",
            "content", "content_sha256", "ordinal",
        },
        {"id"},
    ),
    "analysis_run_input_context": (
        {"input_id", "document_role", "story_scope"},
        {"input_id"},
    ),
    "analysis_run_execution": (
        {
            "run_id", "attempt_no", "worker_token", "lease_expires_at", "claimed_at",
            "last_heartbeat_at", "retried_from_run_id",
        },
        {"run_id"},
    ),
    "issues": (
        {
            "id", "run_id", "category", "severity", "confidence", "title",
            "explanation", "evidence", "suggestion", "extra",
        },
        {"id"},
    ),
    "analysis_records": ({"id", "run_id", "kind", "attrs", "evidence"}, {"id"}),
    "analysis_diagnostics": ({"run_id", "payload", "created_at"}, {"run_id"}),
    "run_events": (
        {"id", "run_id", "stage", "progress", "message", "created_at"},
        {"id"},
    ),
    "issue_feedback": (
        {"id", "issue_id", "label", "comment", "created_at"},
        {"id"},
    ),
}


def upgrade() -> None:
    bind = op.get_bind()
    existing = set(sa.inspect(bind).get_table_names())
    _validate_existing_tables(bind, existing)
    _validate_existing_constraints(bind, existing)

    def create(name: str, operation: Callable[[], None]) -> None:
        if name not in existing:
            operation()
            existing.add(name)

    create(
        "projects",
        lambda: op.create_table(
            "projects",
            sa.Column("id", sa.String(36), primary_key=True, nullable=False),
            sa.Column("name", sa.String(200), nullable=False),
            sa.Column("description", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        ),
    )
    create(
        "documents",
        lambda: op.create_table(
            "documents",
            sa.Column("id", sa.String(36), primary_key=True, nullable=False),
            sa.Column(
                "project_id", sa.String(36), sa.ForeignKey("projects.id"), nullable=False
            ),
            sa.Column("name", sa.String(255), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("version", sa.Integer(), nullable=False),
            sa.Column("active", sa.Boolean(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.UniqueConstraint(
                "project_id", "id", name="uq_documents_project_id_id"
            ),
        ),
    )
    create(
        "document_context",
        lambda: op.create_table(
            "document_context",
            sa.Column(
                "document_id",
                sa.String(36),
                sa.ForeignKey("documents.id"),
                primary_key=True,
            ),
            sa.Column("document_role", sa.String(40), nullable=False),
            sa.Column("story_scope", sa.String(80), nullable=False),
        ),
    )
    create(
        "analysis_runs",
        lambda: op.create_table(
            "analysis_runs",
            sa.Column("id", sa.String(36), primary_key=True, nullable=False),
            sa.Column("project_id", sa.String(36), sa.ForeignKey("projects.id"), nullable=False),
            sa.Column("status", sa.String(32), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("started_at", sa.DateTime(), nullable=True),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.Column("input_chars", sa.Integer(), nullable=False),
            sa.Column("prompt_tokens", sa.Integer(), nullable=False),
            sa.Column("completion_tokens", sa.Integer(), nullable=False),
            sa.Column("estimated_cost_usd", sa.Float(), nullable=False),
            sa.Column("error", sa.Text(), nullable=True),
            sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        ),
    )
    create(
        "analysis_run_inputs",
        lambda: op.create_table(
            "analysis_run_inputs",
            sa.Column("id", sa.String(36), primary_key=True, nullable=False),
            sa.Column("run_id", sa.String(36), sa.ForeignKey("analysis_runs.id"), nullable=False),
            sa.Column("document_id", sa.String(36), nullable=False),
            sa.Column("document_name", sa.String(255), nullable=False),
            sa.Column("document_version", sa.Integer(), nullable=False),
            sa.Column("content", sa.Text(), nullable=False),
            sa.Column("content_sha256", sa.String(64), nullable=False),
            sa.Column("ordinal", sa.Integer(), nullable=False),
            sa.UniqueConstraint("run_id", "ordinal", name="uq_analysis_run_input_ordinal"),
            sa.UniqueConstraint("run_id", "document_id", name="uq_analysis_run_input_document"),
        ),
    )
    create(
        "analysis_run_input_context",
        lambda: op.create_table(
            "analysis_run_input_context",
            sa.Column(
                "input_id",
                sa.String(36),
                sa.ForeignKey("analysis_run_inputs.id"),
                primary_key=True,
            ),
            sa.Column("document_role", sa.String(40), nullable=True),
            sa.Column("story_scope", sa.String(200), nullable=True),
        ),
    )
    create(
        "analysis_run_execution",
        lambda: op.create_table(
            "analysis_run_execution",
            sa.Column("run_id", sa.String(36), sa.ForeignKey("analysis_runs.id"), primary_key=True),
            sa.Column("attempt_no", sa.Integer(), nullable=False),
            sa.Column("worker_token", sa.String(100), nullable=True),
            sa.Column("lease_expires_at", sa.DateTime(), nullable=True),
            sa.Column("claimed_at", sa.DateTime(), nullable=True),
            sa.Column("last_heartbeat_at", sa.DateTime(), nullable=True),
            sa.Column("retried_from_run_id", sa.String(36), nullable=True),
        ),
    )
    create(
        "issues",
        lambda: op.create_table(
            "issues",
            sa.Column("id", sa.String(36), primary_key=True, nullable=False),
            sa.Column("run_id", sa.String(36), sa.ForeignKey("analysis_runs.id"), nullable=False),
            sa.Column("category", sa.String(80), nullable=False),
            sa.Column("severity", sa.String(20), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("title", sa.String(255), nullable=False),
            sa.Column("explanation", sa.Text(), nullable=False),
            sa.Column("evidence", sa.JSON(), nullable=False),
            sa.Column("suggestion", sa.Text(), nullable=False),
            sa.Column("extra", sa.JSON(), nullable=False),
        ),
    )
    create(
        "analysis_records",
        lambda: op.create_table(
            "analysis_records",
            sa.Column("id", sa.String(36), primary_key=True, nullable=False),
            sa.Column("run_id", sa.String(36), sa.ForeignKey("analysis_runs.id"), nullable=False),
            sa.Column("kind", sa.String(40), nullable=False),
            sa.Column("attrs", sa.JSON(), nullable=False),
            sa.Column("evidence", sa.JSON(), nullable=False),
        ),
    )
    create(
        "analysis_diagnostics",
        lambda: op.create_table(
            "analysis_diagnostics",
            sa.Column("run_id", sa.String(36), sa.ForeignKey("analysis_runs.id"), primary_key=True),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        ),
    )
    create(
        "run_events",
        lambda: op.create_table(
            "run_events",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True, nullable=False),
            sa.Column("run_id", sa.String(36), sa.ForeignKey("analysis_runs.id"), nullable=False),
            sa.Column("stage", sa.String(80), nullable=False),
            sa.Column("progress", sa.Integer(), nullable=False),
            sa.Column("message", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        ),
    )
    create(
        "issue_feedback",
        lambda: op.create_table(
            "issue_feedback",
            sa.Column("id", sa.String(36), primary_key=True, nullable=False),
            sa.Column("issue_id", sa.String(36), sa.ForeignKey("issues.id"), nullable=False),
            sa.Column("label", sa.String(32), nullable=False),
            sa.Column("comment", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        ),
    )

    _ensure_index("documents", "ix_documents_project_id", ["project_id"])
    _ensure_index("analysis_runs", "ix_analysis_runs_project_id", ["project_id"])
    _ensure_index("analysis_run_inputs", "ix_analysis_run_inputs_run_id", ["run_id"])
    _ensure_index("issues", "ix_issues_run_id", ["run_id"])
    _ensure_index("analysis_records", "ix_analysis_records_run_id", ["run_id"])
    _ensure_index("run_events", "ix_run_events_run_id", ["run_id"])
    _ensure_index("issue_feedback", "ix_issue_feedback_issue_id", ["issue_id"])
    _ensure_unique_index(
        "documents", "uq_documents_project_id_id", ["project_id", "id"]
    )


def downgrade() -> None:
    # This revision can adopt tables created by pre-Alembic releases.  It is
    # intentionally non-destructive because it cannot distinguish adopted
    # user data from tables created during a fresh upgrade.
    pass


def _ensure_index(table_name: str, index_name: str, columns: list[str]) -> None:
    indexes = {item["name"] for item in sa.inspect(op.get_bind()).get_indexes(table_name)}
    if index_name not in indexes:
        op.create_index(index_name, table_name, columns, unique=False)


def _ensure_unique_index(table_name: str, index_name: str, columns: list[str]) -> None:
    expected = tuple(columns)
    unique_sets = _unique_column_tuples(op.get_bind(), table_name)
    if expected not in unique_sets:
        op.create_index(index_name, table_name, columns, unique=True)


def _validate_existing_tables(bind, existing: set[str]) -> None:
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


def _validate_existing_constraints(bind, existing: set[str]) -> None:
    inspector = sa.inspect(bind)
    if "analysis_run_inputs" in existing:
        unique_columns = _unique_column_tuples(bind, "analysis_run_inputs")
        required = {("run_id", "ordinal"), ("run_id", "document_id")}
        if not required.issubset(unique_columns):
            raise RuntimeError("incompatible pre-existing table: analysis_run_inputs")

    foreign_keys = {
        "documents": (("project_id",), "projects", ("id",)),
        "document_context": (("document_id",), "documents", ("id",)),
        "analysis_runs": (("project_id",), "projects", ("id",)),
        "analysis_run_inputs": (("run_id",), "analysis_runs", ("id",)),
        "analysis_run_input_context": (
            ("input_id",),
            "analysis_run_inputs",
            ("id",),
        ),
        "analysis_run_execution": (("run_id",), "analysis_runs", ("id",)),
        "issues": (("run_id",), "analysis_runs", ("id",)),
        "analysis_records": (("run_id",), "analysis_runs", ("id",)),
        "analysis_diagnostics": (("run_id",), "analysis_runs", ("id",)),
        "run_events": (("run_id",), "analysis_runs", ("id",)),
        "issue_feedback": (("issue_id",), "issues", ("id",)),
    }
    for table_name, (local_columns, referred_table, referred_columns) in foreign_keys.items():
        if table_name not in existing:
            continue
        if not _has_foreign_key(
            inspector,
            table_name,
            local_columns,
            referred_table,
            referred_columns,
        ):
            raise RuntimeError(f"incompatible pre-existing table: {table_name}")


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
    # SQLAlchemy cannot reflect every named, multiline SQLite UNIQUE
    # constraint, while SQLite always exposes its backing auto-index.
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
