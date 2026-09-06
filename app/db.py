from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from alembic import command
from alembic.config import Config
from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import get_settings
from .time_utils import utc_now_naive


class Base(DeclarativeBase):
    pass


def new_id() -> str:
    return str(uuid4())


class ProjectRow(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)
    documents: Mapped[list["DocumentRow"]] = relationship(cascade="all, delete-orphan")


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("project_id", "id", name="uq_documents_project_id_id"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    name: Mapped[str] = mapped_column(String(255))
    content: Mapped[str] = mapped_column(Text)
    version: Mapped[int] = mapped_column(Integer, default=1)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class DocumentContextRow(Base):
    """Typed document context stored additively for legacy DB compatibility."""

    __tablename__ = "document_context"
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id"), primary_key=True
    )
    document_role: Mapped[str] = mapped_column(String(40), default="chapter")
    story_scope: Mapped[str] = mapped_column(String(80), default="global")


class AnalysisRunRow(Base):
    __tablename__ = "analysis_runs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    input_chars: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)


class AnalysisRunInputRow(Base):
    """Immutable document body captured when an analysis run is created.

    This is a separate table rather than new columns on ``documents`` or
    ``analysis_runs`` so ``create_all`` can safely add it to existing local
    SQLite databases.  Workers must never reconstruct run input from the
    project's current active documents.
    """

    __tablename__ = "analysis_run_inputs"
    __table_args__ = (
        UniqueConstraint("run_id", "ordinal", name="uq_analysis_run_input_ordinal"),
        UniqueConstraint("run_id", "document_id", name="uq_analysis_run_input_document"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    document_id: Mapped[str] = mapped_column(String(36))
    document_name: Mapped[str] = mapped_column(String(255))
    document_version: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    content_sha256: Mapped[str] = mapped_column(String(64))
    ordinal: Mapped[int] = mapped_column(Integer)


class AnalysisRunInputContextRow(Base):
    """Optional semantic context kept separate for additive local upgrades."""

    __tablename__ = "analysis_run_input_context"
    input_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_run_inputs.id"), primary_key=True
    )
    document_role: Mapped[str | None] = mapped_column(String(40), nullable=True)
    story_scope: Mapped[str | None] = mapped_column(String(200), nullable=True)


class AnalysisRunExecutionRow(Base):
    """Small coordination record for best-effort worker ownership.

    A conditional update on this row prevents concurrent workers from running
    the same analysis.  The lease is deliberately described as a bounded
    ownership mechanism, not as an exactly-once guarantee.
    """

    __tablename__ = "analysis_run_execution"
    run_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_runs.id"), primary_key=True
    )
    attempt_no: Mapped[int] = mapped_column(Integer, default=0)
    worker_token: Mapped[str | None] = mapped_column(String(100), nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    retried_from_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True)


class IssueRow(Base):
    __tablename__ = "issues"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    category: Mapped[str] = mapped_column(String(80))
    severity: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)
    title: Mapped[str] = mapped_column(String(255))
    explanation: Mapped[str] = mapped_column(Text)
    evidence: Mapped[list] = mapped_column(JSON)
    suggestion: Mapped[str] = mapped_column(Text)
    extra: Mapped[dict] = mapped_column(JSON, default=dict)


class AnalysisRecordRow(Base):
    __tablename__ = "analysis_records"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    kind: Mapped[str] = mapped_column(String(40))
    attrs: Mapped[dict] = mapped_column(JSON)
    evidence: Mapped[dict] = mapped_column(JSON)


class AnalysisDiagnosticRow(Base):
    __tablename__ = "analysis_diagnostics"
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id"), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class RunEventRow(Base):
    __tablename__ = "run_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("analysis_runs.id"), index=True)
    stage: Mapped[str] = mapped_column(String(80))
    progress: Mapped[int] = mapped_column(Integer)
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class FeedbackRow(Base):
    __tablename__ = "issue_feedback"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    issue_id: Mapped[str] = mapped_column(ForeignKey("issues.id"), index=True)
    label: Mapped[str] = mapped_column(String(32))
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class EmbeddingProfileRow(Base):
    """A versioned embedding-space identity; never stores credentials or URLs."""

    __tablename__ = "embedding_profiles"
    __table_args__ = (
        UniqueConstraint(
            "provider_kind",
            "provider_namespace",
            "model_identifier",
            "model_revision",
            "dimensions",
            "normalized",
            name="uq_embedding_profile_identity",
        ),
        CheckConstraint(
            "dimensions > 0 AND dimensions <= 16000",
            name="ck_embedding_profile_dimensions",
        ),
    )
    id: Mapped[str] = mapped_column(String(68), primary_key=True)
    provider_kind: Mapped[str] = mapped_column(String(40))
    provider_namespace: Mapped[str] = mapped_column(String(80))
    model_identifier: Mapped[str] = mapped_column(String(255))
    model_revision: Mapped[str] = mapped_column(String(120))
    dimensions: Mapped[int] = mapped_column(Integer)
    normalized: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class EvidenceChunkRow(Base):
    """Deterministic chunk bound to one exact document snapshot."""

    __tablename__ = "evidence_chunks"
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id", "document_id"],
            ["documents.project_id", "documents.id"],
            name="fk_evidence_chunk_document_owner",
        ),
        UniqueConstraint(
            "project_id",
            "document_id",
            "document_version",
            "content_sha256",
            "chunker_version",
            "ordinal",
            name="uq_evidence_chunk_snapshot_ordinal",
        ),
        CheckConstraint("document_version > 0", name="ck_evidence_chunk_version"),
        CheckConstraint("ordinal >= 0", name="ck_evidence_chunk_ordinal"),
        CheckConstraint("line_start > 0", name="ck_evidence_chunk_line_start"),
        CheckConstraint("line_end >= line_start", name="ck_evidence_chunk_line_end"),
        CheckConstraint("char_start >= 0", name="ck_evidence_chunk_char_start"),
        CheckConstraint("char_end > char_start", name="ck_evidence_chunk_char_end"),
        CheckConstraint(
            "length(content_sha256) = 64", name="ck_evidence_chunk_content_hash"
        ),
        CheckConstraint(
            "length(text_sha256) = 64", name="ck_evidence_chunk_text_hash"
        ),
    )
    id: Mapped[str] = mapped_column(String(68), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    document_id: Mapped[str] = mapped_column(String(36), index=True)
    document_version: Mapped[int] = mapped_column(Integer)
    content_sha256: Mapped[str] = mapped_column(String(64))
    chunker_version: Mapped[str] = mapped_column(String(80))
    ordinal: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    text_sha256: Mapped[str] = mapped_column(String(64))
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    line_start: Mapped[int] = mapped_column(Integer)
    line_end: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class EvidenceEmbeddingRow(Base):
    """Vector for a chunk/profile pair.

    PostgreSQL gets a real pgvector column.  SQLite's JSON variant exists only
    for deterministic local and unit-test storage; it is not a similarity
    search implementation.
    """

    __tablename__ = "evidence_embeddings"
    __table_args__ = (
        CheckConstraint(
            "dimensions > 0 AND dimensions <= 16000",
            name="ck_evidence_embedding_dimensions",
        ),
    )
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey("evidence_chunks.id", ondelete="CASCADE"), primary_key=True
    )
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("embedding_profiles.id", ondelete="CASCADE"),
        primary_key=True,
        index=True,
    )
    dimensions: Mapped[int] = mapped_column(Integer)
    vector: Mapped[list[float]] = mapped_column(
        Vector().with_variant(JSON(), "sqlite")
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


def enable_sqlite_foreign_keys(target_engine: Engine) -> None:
    """Enable SQLite FK enforcement for application and explicit test engines."""

    @event.listens_for(target_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()


settings = get_settings()
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)
if settings.database_url.startswith("sqlite"):
    enable_sqlite_foreign_keys(engine)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    """Upgrade product databases through versioned, additive migrations."""

    root = Path(__file__).resolve().parents[1]
    config = Config(str(root / "alembic.ini"))
    config.set_main_option("script_location", str(root / "migrations"))
    config.attributes["database_url"] = settings.database_url
    command.upgrade(config, "head")
