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
    Index,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

from .config import get_settings
from .time_utils import utc_now_naive


LOCAL_USER_ID = "00000000-0000-0000-0000-000000000001"
LOCAL_WORKSPACE_ID = "00000000-0000-0000-0000-000000000001"
LOCAL_MEMBERSHIP_ID = "00000000-0000-0000-0000-000000000001"


def default_project_workspace_id() -> str:
    """Compatibility default for anonymous/local callers only.

    Required-auth code must always pass an explicit workspace.  Raising here
    turns any future forgotten ownership assignment into a failed transaction
    instead of silently leaking the resource into the local workspace.
    """

    if get_settings().auth_mode != "anonymous":
        raise RuntimeError("workspace_id is required when authentication is enabled")
    return LOCAL_WORKSPACE_ID


class Base(DeclarativeBase):
    pass


def new_id() -> str:
    return str(uuid4())


class UserRow(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(80))
    password_hash: Mapped[str] = mapped_column(String(512))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class WorkspaceRow(Base):
    __tablename__ = "workspaces"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120))
    kind: Mapped[str] = mapped_column(String(24), default="personal")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class WorkspaceMemberRow(Base):
    __tablename__ = "workspace_members"
    __table_args__ = (
        UniqueConstraint("workspace_id", "user_id", name="uq_workspace_member"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(24), default="member")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class AuthSessionRow(Base):
    __tablename__ = "auth_sessions"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    csrf_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AccountProviderConfigRow(Base):
    """One immutable account-owned chat-provider revision.

    Provider semantics never change after insertion.  Encryption envelopes may
    be rewrapped during master-key rotation, and revoked rows are scrubbed in
    place so historical foreign keys remain useful without retaining secrets.
    """

    __tablename__ = "account_provider_configs"
    __table_args__ = (
        UniqueConstraint(
            "user_id", "revision", name="uq_account_provider_config_revision"
        ),
        CheckConstraint("revision > 0", name="ck_account_provider_config_revision"),
        CheckConstraint(
            "state IN ('usable', 'superseded', 'revoked', 'scrubbed')",
            name="ck_account_provider_config_state",
        ),
        CheckConstraint(
            "length(endpoint_sha256) = 64",
            name="ck_account_provider_config_endpoint_hash",
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    revision: Mapped[int] = mapped_column(Integer)
    base_url: Mapped[str] = mapped_column(String(2048))
    model_name: Mapped[str] = mapped_column(String(255))
    endpoint_sha256: Mapped[str] = mapped_column(String(64))
    secret_ciphertext: Mapped[bytes | None] = mapped_column(
        LargeBinary, nullable=True
    )
    secret_nonce: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    encryption_key_id: Mapped[str | None] = mapped_column(
        String(64), nullable=True
    )
    secret_schema_version: Mapped[int] = mapped_column(Integer, default=1)
    state: Mapped[str] = mapped_column(String(24), default="usable", index=True)
    created_by_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    secret_rewrapped_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True
    )
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_test_status: Mapped[str | None] = mapped_column(String(24), nullable=True)
    last_test_category: Mapped[str | None] = mapped_column(String(80), nullable=True)
    last_test_payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class AccountProviderBindingRow(Base):
    """Current provider pointer and optimistic-lock revision for one account."""

    __tablename__ = "account_provider_bindings"
    user_id: Mapped[str] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    current_config_id: Mapped[str | None] = mapped_column(
        ForeignKey("account_provider_configs.id", ondelete="SET NULL"),
        nullable=True,
        unique=True,
    )
    lock_version: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class ProjectRow(Base):
    __tablename__ = "projects"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id", ondelete="RESTRICT"),
        default=default_project_workspace_id,
        nullable=False,
        index=True,
    )
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


# The project row is locked while allocating versions, and these constraints
# remain the final concurrency authority. The partial index guarantees there
# is never more than one active revision for a case-insensitive logical name.
Index(
    "uq_documents_project_lower_name_version",
    DocumentRow.project_id,
    func.lower(DocumentRow.name),
    DocumentRow.version,
    unique=True,
)
Index(
    "uq_documents_project_lower_name_active",
    DocumentRow.project_id,
    func.lower(DocumentRow.name),
    unique=True,
    sqlite_where=DocumentRow.active.is_(True),
    postgresql_where=DocumentRow.active.is_(True),
)


class DocumentContextRow(Base):
    """Typed document context stored additively for legacy DB compatibility."""

    __tablename__ = "document_context"
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id"), primary_key=True
    )
    document_role: Mapped[str] = mapped_column(String(40), default="chapter")
    story_scope: Mapped[str] = mapped_column(String(80), default="global")


class DocumentNarrativeContextRevisionRow(Base):
    """Append-only authority and structured-scope metadata for one document.

    ``document_context`` remains the compatibility contract consumed by the
    existing rule engine.  This table is deliberately revisioned so a run can
    freeze the exact metadata it observed without consulting a later edit.
    """

    __tablename__ = "document_narrative_context_revisions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["project_id", "document_id"],
            ["documents.project_id", "documents.id"],
            name="fk_document_narrative_context_document_owner",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "document_id",
            "revision",
            name="uq_document_narrative_context_revision",
        ),
        CheckConstraint(
            "revision > 0", name="ck_document_narrative_context_revision"
        ),
        CheckConstraint(
            "resolution_state IN ('unresolved', 'inferred', 'confirmed')",
            name="ck_document_narrative_context_resolution",
        ),
        CheckConstraint(
            "origin IN ('explicit', 'deterministic_import', 'model_inferred', 'legacy')",
            name="ck_document_narrative_context_origin",
        ),
        CheckConstraint(
            "authority_tier IN ('unresolved', 'core_canon', 'formal_record', 'draft', 'reference')",
            name="ck_document_narrative_context_authority",
        ),
        CheckConstraint(
            "publication_status IN ('draft', 'in_review', 'published', 'retired', 'unknown')",
            name="ck_document_narrative_context_publication",
        ),
        CheckConstraint(
            "inference_confidence IS NULL OR "
            "(inference_confidence >= 0 AND inference_confidence <= 1)",
            name="ck_document_narrative_context_confidence",
        ),
        CheckConstraint(
            "length(scope_sha256) = 64",
            name="ck_document_narrative_context_scope_hash",
        ),
        Index(
            "ix_document_narrative_context_latest",
            "project_id",
            "document_id",
            "revision",
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[str] = mapped_column(String(36), index=True)
    revision: Mapped[int] = mapped_column(Integer)
    resolution_state: Mapped[str] = mapped_column(String(24))
    origin: Mapped[str] = mapped_column(String(32))
    authority_tier: Mapped[str] = mapped_column(String(24))
    publication_status: Mapped[str] = mapped_column(String(24))
    scope_payload: Mapped[dict] = mapped_column(JSON)
    scope_sha256: Mapped[str] = mapped_column(String(64))
    inference_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    inference_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    inference_evidence: Mapped[list | None] = mapped_column(JSON, nullable=True)
    inference_usage: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    inference_provider_config_id: Mapped[str | None] = mapped_column(
        ForeignKey("account_provider_configs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    inference_provider_identity: Mapped[dict | None] = mapped_column(
        JSON, nullable=True
    )
    created_by_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class AnalysisRunRow(Base):
    __tablename__ = "analysis_runs"
    __table_args__ = (
        Index(
            "uq_analysis_runs_project_id_idempotency_key",
            "project_id",
            "idempotency_key",
            unique=True,
        ),
        Index(
            "ix_analysis_runs_project_created_id",
            "project_id",
            "created_at",
            "id",
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), index=True)
    # Optional so legacy and non-idempotent clients remain compatible.  A key
    # is scoped to one project by the unique index above; nullable keys may
    # appear on any number of ordinary runs on both PostgreSQL and SQLite.
    idempotency_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    # Nullable by design so audit history survives future user deletion.  The
    # HTTP creation paths always populate it; the ownership migration backfills
    # legacy rows to the fixed local identity.
    requested_by_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    provider_config_id: Mapped[str | None] = mapped_column(
        ForeignKey("account_provider_configs.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    provider_identity: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    # The batch contract is frozen on the run itself.  Legacy callers and
    # migrated rows keep the historical all-active-document behaviour through
    # the ``full_review`` default.
    batch_mode: Mapped[str] = mapped_column(String(24), default="full_review")
    sensitivity: Mapped[str] = mapped_column(String(24), default="balanced")
    batch_coverage: Mapped[dict] = mapped_column(JSON, default=dict)
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
    # ``target`` is the draft under review; ``background`` is server-selected
    # authority/history.  The worker receives both but may never infer the
    # authority boundary from client-controlled request fields.
    batch_role: Mapped[str] = mapped_column(String(16), default="target")


class AnalysisRunInputNarrativeContextRow(Base):
    """Frozen structured narrative context for one immutable run input."""

    __tablename__ = "analysis_run_input_narrative_context"
    __table_args__ = (
        CheckConstraint(
            "length(payload_sha256) = 64",
            name="ck_analysis_run_input_narrative_context_hash",
        ),
    )
    input_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_run_inputs.id", ondelete="CASCADE"), primary_key=True
    )
    context_revision_id: Mapped[str | None] = mapped_column(
        ForeignKey("document_narrative_context_revisions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    payload: Mapped[dict] = mapped_column(JSON)
    payload_sha256: Mapped[str] = mapped_column(String(64))


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


class AnalysisRunComparisonRow(Base):
    """Durable lineage between a completed baseline and one revision run.

    The matcher version and summary are frozen with the comparison so a later
    matcher upgrade cannot silently reinterpret an already presented report.
    """

    __tablename__ = "analysis_run_comparisons"
    __table_args__ = (
        UniqueConstraint(
            "baseline_run_id",
            "target_run_id",
            name="uq_analysis_run_comparison_pair",
        ),
        Index(
            "ix_analysis_run_comparisons_baseline_created",
            "baseline_run_id",
            "created_at",
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    baseline_run_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )
    target_run_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), unique=True, index=True
    )
    matcher_version: Mapped[str] = mapped_column(String(80))
    status: Mapped[str] = mapped_column(String(24), default="pending")
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class IssueComparisonItemRow(Base):
    """One conservative outcome in a revision comparison."""

    __tablename__ = "issue_comparison_items"
    __table_args__ = (
        CheckConstraint(
            "outcome IN ('no_longer_detected', 'persisting', 'new', 'unverifiable')",
            name="ck_issue_comparison_outcome",
        ),
        UniqueConstraint(
            "comparison_id",
            "baseline_issue_id",
            name="uq_issue_comparison_baseline_issue",
        ),
        UniqueConstraint(
            "comparison_id",
            "target_issue_id",
            name="uq_issue_comparison_target_issue",
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    comparison_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_run_comparisons.id", ondelete="CASCADE"), index=True
    )
    outcome: Mapped[str] = mapped_column(String(24), index=True)
    baseline_issue_id: Mapped[str | None] = mapped_column(
        ForeignKey("issues.id", ondelete="CASCADE"), nullable=True
    )
    target_issue_id: Mapped[str | None] = mapped_column(
        ForeignKey("issues.id", ondelete="CASCADE"), nullable=True
    )
    match_method: Mapped[str | None] = mapped_column(String(80), nullable=True)
    match_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)


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
    # Feedback remains attributable while the user exists but is retained when
    # an account is removed.  New API writes always set this field.
    created_by_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    label: Mapped[str] = mapped_column(String(32))
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class CharacterTraitAxisRow(Base):
    """Immutable, author-created project-local comparison identity (v1 only)."""

    __tablename__ = "character_trait_axes"
    __table_args__ = (
        UniqueConstraint(
            "id", "project_id", name="uq_character_trait_axis_project_identity"
        ),
        UniqueConstraint(
            "project_id", "trait_type", "definition_sha256",
            name="uq_character_trait_axis_definition",
        ),
        CheckConstraint(
            "trait_type = 'core_personality'",
            name="ck_character_trait_axis_type",
        ),
        CheckConstraint("version = 1", name="ck_character_trait_axis_version"),
        CheckConstraint(
            "length(definition_sha256) = 64",
            name="ck_character_trait_axis_definition_hash",
        ),
        Index("ix_character_trait_axes_project_created", "project_id", "created_at"),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    trait_type: Mapped[str] = mapped_column(String(32), default="core_personality")
    version: Mapped[int] = mapped_column(Integer, default=1)
    display_name: Mapped[str] = mapped_column(String(80))
    definition: Mapped[str] = mapped_column(String(200))
    definition_sha256: Mapped[str] = mapped_column(String(64))
    created_by_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class CharacterTraitCandidateRow(Base):
    """Evidence-bound candidate; confirmed rows double as profile entries."""

    __tablename__ = "character_trait_candidates"
    __table_args__ = (
        ForeignKeyConstraint(
            ["approved_axis_id", "project_id"],
            ["character_trait_axes.id", "character_trait_axes.project_id"],
            ondelete="RESTRICT",
            name="fk_character_trait_candidate_axis_project",
        ),
        UniqueConstraint(
            "project_id",
            "source_run_id",
            "candidate_fingerprint",
            name="uq_character_trait_candidate_fingerprint",
        ),
        CheckConstraint(
            "trait_type IN ('core_personality', 'preference', 'value', "
            "'speech_pattern', 'behavior_boundary', 'contextual_behavior', "
            "'current_state')",
            name="ck_character_trait_candidate_type",
        ),
        CheckConstraint(
            "polarity IN ('positive', 'negative', 'neutral', 'unclear')",
            name="ck_character_trait_candidate_polarity",
        ),
        CheckConstraint(
            "stability IN ('core', 'stable', 'temporary', 'situational', 'unknown')",
            name="ck_character_trait_candidate_stability",
        ),
        CheckConstraint(
            "origin IN ('explicit_setting', 'history_inference')",
            name="ck_character_trait_candidate_origin",
        ),
        CheckConstraint(
            "authority_tier IN ('core_canon', 'formal_record')",
            name="ck_character_trait_candidate_authority",
        ),
        CheckConstraint(
            "review_state IN ('pending', 'confirmed', 'rejected', 'superseded')",
            name="ck_character_trait_candidate_review_state",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_character_trait_candidate_confidence",
        ),
        CheckConstraint(
            "lock_version >= 0", name="ck_character_trait_candidate_lock_version"
        ),
        CheckConstraint(
            "(approved_axis_id IS NULL AND approved_axis_version IS NULL) OR "
            "(approved_axis_id IS NOT NULL AND approved_axis_version = 1)",
            name="ck_character_trait_candidate_axis_pair",
        ),
        CheckConstraint(
            "valid_from_release_ordinal IS NULL OR valid_from_release_ordinal >= 0",
            name="ck_character_trait_candidate_valid_from",
        ),
        CheckConstraint(
            "valid_until_release_ordinal IS NULL OR valid_until_release_ordinal >= 0",
            name="ck_character_trait_candidate_valid_until",
        ),
        CheckConstraint(
            "length(scope_sha256) = 64 AND length(evidence_sha256) = 64 "
            "AND length(candidate_fingerprint) = 64",
            name="ck_character_trait_candidate_hashes",
        ),
        CheckConstraint(
            "support_binding_mode IS NOT NULL AND ("
            "(support_binding_mode = 'legacy_v1' "
            "AND support_bindings_v1 IS NULL "
            "AND support_bindings_sha256 IS NULL) OR "
            "(support_binding_mode = 'required_v1' "
            "AND support_bindings_v1 IS NOT NULL "
            "AND support_bindings_sha256 IS NOT NULL "
            "AND length(support_bindings_sha256) = 64))",
            name="ck_character_trait_candidate_support_bindings_pair",
        ),
        Index(
            "ix_character_trait_candidates_project_state_character",
            "project_id",
            "review_state",
            "character_key",
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    source_run_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )
    character_key: Mapped[str] = mapped_column(String(160))
    character_display_name: Mapped[str] = mapped_column(String(160))
    trait_type: Mapped[str] = mapped_column(String(32))
    trait_key: Mapped[str] = mapped_column(String(160))
    # Bound only by an authenticated author decision. Never populated by extraction.
    approved_axis_id: Mapped[str | None] = mapped_column(
        ForeignKey("character_trait_axes.id", ondelete="RESTRICT"), nullable=True
    )
    approved_axis_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Pre-migration candidates have no recoverable object anchor.
    comparison_key: Mapped[str | None] = mapped_column(String(200), nullable=True)
    value: Mapped[str] = mapped_column(Text)
    polarity: Mapped[str] = mapped_column(String(24), default="unclear")
    stability: Mapped[str] = mapped_column(String(24))
    contexts: Mapped[list] = mapped_column(JSON, default=list)
    origin: Mapped[str] = mapped_column(String(32))
    authority_tier: Mapped[str] = mapped_column(
        String(24), default="formal_record"
    )
    confidence: Mapped[float] = mapped_column(Float)
    scope_payload: Mapped[dict] = mapped_column(JSON)
    scope_sha256: Mapped[str] = mapped_column(String(64))
    valid_from_release_ordinal: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    valid_until_release_ordinal: Mapped[int | None] = mapped_column(
        Integer, nullable=True
    )
    evidence: Mapped[list] = mapped_column(JSON)
    evidence_sha256: Mapped[str] = mapped_column(String(64))
    # Optional verified sub-line metadata; legacy evidence and hashes stay intact.
    support_binding_mode: Mapped[str] = mapped_column(String(24), nullable=False)
    support_bindings_v1: Mapped[dict | None] = mapped_column(
        JSON(none_as_null=True), nullable=True
    )
    support_bindings_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    candidate_fingerprint: Mapped[str] = mapped_column(String(64))
    generator_version: Mapped[str] = mapped_column(String(80))
    provenance: Mapped[dict] = mapped_column(JSON, default=dict)
    review_state: Mapped[str] = mapped_column(String(24), default="pending")
    lock_version: Mapped[int] = mapped_column(Integer, default=0)
    supersedes_candidate_id: Mapped[str | None] = mapped_column(
        ForeignKey("character_trait_candidates.id", ondelete="SET NULL"),
        nullable=True,
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    reviewed_by_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class CharacterTraitReviewRow(Base):
    """Append-only human decision audit for a candidate."""

    __tablename__ = "character_trait_reviews"
    __table_args__ = (
        ForeignKeyConstraint(
            ["approved_axis_id", "project_id"],
            ["character_trait_axes.id", "character_trait_axes.project_id"],
            ondelete="RESTRICT",
            name="fk_character_trait_review_axis_project",
        ),
        UniqueConstraint(
            "candidate_id",
            "idempotency_key",
            name="uq_character_trait_review_idempotency",
        ),
        CheckConstraint(
            "decision IN ('confirm', 'reject', 'supersede')",
            name="ck_character_trait_review_decision",
        ),
        CheckConstraint(
            "expected_lock_version >= 0",
            name="ck_character_trait_review_expected_version",
        ),
        CheckConstraint(
            "(approved_axis_id IS NULL AND approved_axis_version IS NULL) OR "
            "(approved_axis_id IS NOT NULL AND approved_axis_version = 1)",
            name="ck_character_trait_review_axis_pair",
        ),
        Index(
            "ix_character_trait_reviews_project_candidate_created",
            "project_id",
            "candidate_id",
            "created_at",
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("character_trait_candidates.id", ondelete="CASCADE"), index=True
    )
    decision: Mapped[str] = mapped_column(String(24))
    approved_axis_id: Mapped[str | None] = mapped_column(
        ForeignKey("character_trait_axes.id", ondelete="RESTRICT"), nullable=True
    )
    approved_axis_version: Mapped[int | None] = mapped_column(Integer, nullable=True)
    expected_lock_version: Mapped[int] = mapped_column(Integer)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    comment: Mapped[str] = mapped_column(Text, default="")
    created_by_user_id: Mapped[str | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now_naive)


class AnalysisRunCharacterTraitInputRow(Base):
    """Immutable confirmed-profile snapshot consumed by one analysis run."""

    __tablename__ = "analysis_run_character_trait_inputs"
    __table_args__ = (
        UniqueConstraint(
            "run_id", "candidate_id", name="uq_analysis_run_character_trait_candidate"
        ),
        UniqueConstraint(
            "run_id", "ordinal", name="uq_analysis_run_character_trait_ordinal"
        ),
        CheckConstraint(
            "ordinal >= 0", name="ck_analysis_run_character_trait_ordinal"
        ),
        CheckConstraint(
            "candidate_lock_version >= 0",
            name="ck_analysis_run_character_trait_candidate_version",
        ),
        CheckConstraint(
            "length(payload_sha256) = 64",
            name="ck_analysis_run_character_trait_payload_hash",
        ),
    )
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("analysis_runs.id", ondelete="CASCADE"), index=True
    )
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), index=True
    )
    candidate_id: Mapped[str] = mapped_column(
        ForeignKey("character_trait_candidates.id", ondelete="RESTRICT"), index=True
    )
    confirmation_review_id: Mapped[str] = mapped_column(
        ForeignKey("character_trait_reviews.id", ondelete="RESTRICT")
    )
    candidate_lock_version: Mapped[int] = mapped_column(Integer)
    ordinal: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSON)
    payload_sha256: Mapped[str] = mapped_column(String(64))


class EmbeddingProfileRow(Base):
    """A versioned embedding-space identity; never stores credentials or URLs."""

    __tablename__ = "embedding_profiles"
    __table_args__ = (
        UniqueConstraint(
            "provider_kind",
            "provider_namespace",
            "model_identifier",
            "model_revision",
            "deployment_fingerprint",
            "document_transform_identity",
            "query_transform_identity",
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
    deployment_fingerprint: Mapped[str] = mapped_column(String(160))
    document_transform_identity: Mapped[str] = mapped_column(String(80))
    query_transform_identity: Mapped[str] = mapped_column(String(80))
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


@event.listens_for(Base.metadata, "after_create")
def seed_create_all_local_identity(metadata, connection, **_kwargs) -> None:
    """Give direct ``create_all`` databases the same anonymous identity as migrations.

    Production startup uses Alembic.  This hook keeps isolated SDK/tests and
    local tools that intentionally use ``Base.metadata.create_all`` compatible
    with the non-null project workspace foreign key.
    """

    del metadata
    users = UserRow.__table__
    workspaces = WorkspaceRow.__table__
    memberships = WorkspaceMemberRow.__table__
    if connection.execute(
        select(users.c.id).where(users.c.id == LOCAL_USER_ID)
    ).first() is None:
        connection.execute(
            users.insert().values(
                id=LOCAL_USER_ID,
                email="local@loreguard.invalid",
                display_name="本地体验用户",
                password_hash="!anonymous-local-account",
                is_active=True,
                created_at=utc_now_naive(),
            )
        )
    if connection.execute(
        select(workspaces.c.id).where(workspaces.c.id == LOCAL_WORKSPACE_ID)
    ).first() is None:
        connection.execute(
            workspaces.insert().values(
                id=LOCAL_WORKSPACE_ID,
                name="本地工作区",
                kind="personal",
                created_at=utc_now_naive(),
            )
        )
    if connection.execute(
        select(memberships.c.id).where(
            memberships.c.workspace_id == LOCAL_WORKSPACE_ID,
            memberships.c.user_id == LOCAL_USER_ID,
        )
    ).first() is None:
        connection.execute(
            memberships.insert().values(
                id=LOCAL_MEMBERSHIP_ID,
                workspace_id=LOCAL_WORKSPACE_ID,
                user_id=LOCAL_USER_ID,
                role="owner",
                created_at=utc_now_naive(),
            )
        )


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
