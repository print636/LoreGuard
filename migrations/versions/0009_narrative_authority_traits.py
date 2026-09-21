"""Add frozen narrative authority and confirmed character profile substrate.

Revision ID: 0009_narrative_authority_traits
Revises: 0008_document_concurrency
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "0009_narrative_authority_traits"
down_revision = "0008_document_concurrency"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not inspector.has_table("document_narrative_context_revisions"):
        op.create_table(
            "document_narrative_context_revisions",
            sa.Column("id", sa.String(36), nullable=False),
            sa.Column("project_id", sa.String(36), nullable=False),
            sa.Column("document_id", sa.String(36), nullable=False),
            sa.Column("revision", sa.Integer(), nullable=False),
            sa.Column("resolution_state", sa.String(24), nullable=False),
            sa.Column("origin", sa.String(32), nullable=False),
            sa.Column("authority_tier", sa.String(24), nullable=False),
            sa.Column("publication_status", sa.String(24), nullable=False),
            sa.Column("scope_payload", sa.JSON(), nullable=False),
            sa.Column("scope_sha256", sa.String(64), nullable=False),
            sa.Column("inference_confidence", sa.Float(), nullable=True),
            sa.Column("created_by_user_id", sa.String(36), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.ForeignKeyConstraint(
                ["project_id"], ["projects.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["project_id", "document_id"],
                ["documents.project_id", "documents.id"],
                name="fk_document_narrative_context_document_owner",
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["created_by_user_id"], ["users.id"], ondelete="SET NULL"
            ),
            sa.UniqueConstraint(
                "document_id",
                "revision",
                name="uq_document_narrative_context_revision",
            ),
            sa.CheckConstraint(
                "revision > 0", name="ck_document_narrative_context_revision"
            ),
            sa.CheckConstraint(
                "resolution_state IN ('unresolved', 'inferred', 'confirmed')",
                name="ck_document_narrative_context_resolution",
            ),
            sa.CheckConstraint(
                "origin IN ('explicit', 'deterministic_import', 'model_inferred', 'legacy')",
                name="ck_document_narrative_context_origin",
            ),
            sa.CheckConstraint(
                "authority_tier IN ('unresolved', 'core_canon', 'formal_record', 'draft', 'reference')",
                name="ck_document_narrative_context_authority",
            ),
            sa.CheckConstraint(
                "publication_status IN ('draft', 'in_review', 'published', 'retired', 'unknown')",
                name="ck_document_narrative_context_publication",
            ),
            sa.CheckConstraint(
                "inference_confidence IS NULL OR "
                "(inference_confidence >= 0 AND inference_confidence <= 1)",
                name="ck_document_narrative_context_confidence",
            ),
            sa.CheckConstraint(
                "length(scope_sha256) = 64",
                name="ck_document_narrative_context_scope_hash",
            ),
        )
        op.create_index(
            "ix_document_narrative_context_revisions_project_id",
            "document_narrative_context_revisions",
            ["project_id"],
        )
        op.create_index(
            "ix_document_narrative_context_revisions_document_id",
            "document_narrative_context_revisions",
            ["document_id"],
        )
        op.create_index(
            "ix_document_narrative_context_revisions_created_by_user_id",
            "document_narrative_context_revisions",
            ["created_by_user_id"],
        )
        op.create_index(
            "ix_document_narrative_context_latest",
            "document_narrative_context_revisions",
            ["project_id", "document_id", "revision"],
        )

    inspector = sa.inspect(bind)
    if not inspector.has_table("analysis_run_input_narrative_context"):
        op.create_table(
            "analysis_run_input_narrative_context",
            sa.Column("input_id", sa.String(36), nullable=False),
            sa.Column("context_revision_id", sa.String(36), nullable=True),
            sa.Column("schema_version", sa.Integer(), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("payload_sha256", sa.String(64), nullable=False),
            sa.PrimaryKeyConstraint("input_id"),
            sa.ForeignKeyConstraint(
                ["input_id"], ["analysis_run_inputs.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["context_revision_id"],
                ["document_narrative_context_revisions.id"],
                ondelete="SET NULL",
            ),
            sa.CheckConstraint(
                "length(payload_sha256) = 64",
                name="ck_analysis_run_input_narrative_context_hash",
            ),
        )
        op.create_index(
            "ix_analysis_run_input_narrative_context_context_revision_id",
            "analysis_run_input_narrative_context",
            ["context_revision_id"],
        )

    inspector = sa.inspect(bind)
    if not inspector.has_table("character_trait_candidates"):
        op.create_table(
            "character_trait_candidates",
            sa.Column("id", sa.String(36), nullable=False),
            sa.Column("project_id", sa.String(36), nullable=False),
            sa.Column("source_run_id", sa.String(36), nullable=False),
            sa.Column("character_key", sa.String(160), nullable=False),
            sa.Column("character_display_name", sa.String(160), nullable=False),
            sa.Column("trait_type", sa.String(32), nullable=False),
            sa.Column("trait_key", sa.String(160), nullable=False),
            sa.Column("value", sa.Text(), nullable=False),
            sa.Column("polarity", sa.String(24), nullable=False),
            sa.Column("stability", sa.String(24), nullable=False),
            sa.Column("contexts", sa.JSON(), nullable=False),
            sa.Column("origin", sa.String(32), nullable=False),
            sa.Column("confidence", sa.Float(), nullable=False),
            sa.Column("scope_payload", sa.JSON(), nullable=False),
            sa.Column("scope_sha256", sa.String(64), nullable=False),
            sa.Column("valid_from_release_ordinal", sa.Integer(), nullable=True),
            sa.Column("valid_until_release_ordinal", sa.Integer(), nullable=True),
            sa.Column("evidence", sa.JSON(), nullable=False),
            sa.Column("evidence_sha256", sa.String(64), nullable=False),
            sa.Column("candidate_fingerprint", sa.String(64), nullable=False),
            sa.Column("generator_version", sa.String(80), nullable=False),
            sa.Column("provenance", sa.JSON(), nullable=False),
            sa.Column("review_state", sa.String(24), nullable=False),
            sa.Column("lock_version", sa.Integer(), nullable=False),
            sa.Column("supersedes_candidate_id", sa.String(36), nullable=True),
            sa.Column("reviewed_at", sa.DateTime(), nullable=True),
            sa.Column("reviewed_by_user_id", sa.String(36), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.ForeignKeyConstraint(
                ["project_id"], ["projects.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["source_run_id"], ["analysis_runs.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["supersedes_candidate_id"],
                ["character_trait_candidates.id"],
                ondelete="SET NULL",
            ),
            sa.ForeignKeyConstraint(
                ["reviewed_by_user_id"], ["users.id"], ondelete="SET NULL"
            ),
            sa.UniqueConstraint(
                "project_id",
                "source_run_id",
                "candidate_fingerprint",
                name="uq_character_trait_candidate_fingerprint",
            ),
            sa.CheckConstraint(
                "trait_type IN ('core_personality', 'preference', 'value', "
                "'speech_pattern', 'behavior_boundary', 'contextual_behavior', "
                "'current_state')",
                name="ck_character_trait_candidate_type",
            ),
            sa.CheckConstraint(
                "polarity IN ('positive', 'negative', 'neutral', 'unclear')",
                name="ck_character_trait_candidate_polarity",
            ),
            sa.CheckConstraint(
                "stability IN ('core', 'stable', 'temporary', 'situational', 'unknown')",
                name="ck_character_trait_candidate_stability",
            ),
            sa.CheckConstraint(
                "origin IN ('explicit_setting', 'history_inference')",
                name="ck_character_trait_candidate_origin",
            ),
            sa.CheckConstraint(
                "review_state IN ('pending', 'confirmed', 'rejected', 'superseded')",
                name="ck_character_trait_candidate_review_state",
            ),
            sa.CheckConstraint(
                "confidence >= 0 AND confidence <= 1",
                name="ck_character_trait_candidate_confidence",
            ),
            sa.CheckConstraint(
                "lock_version >= 0",
                name="ck_character_trait_candidate_lock_version",
            ),
            sa.CheckConstraint(
                "valid_from_release_ordinal IS NULL OR valid_from_release_ordinal >= 0",
                name="ck_character_trait_candidate_valid_from",
            ),
            sa.CheckConstraint(
                "valid_until_release_ordinal IS NULL OR valid_until_release_ordinal >= 0",
                name="ck_character_trait_candidate_valid_until",
            ),
            sa.CheckConstraint(
                "length(scope_sha256) = 64 AND length(evidence_sha256) = 64 "
                "AND length(candidate_fingerprint) = 64",
                name="ck_character_trait_candidate_hashes",
            ),
        )
        for name, columns in (
            ("ix_character_trait_candidates_project_id", ["project_id"]),
            ("ix_character_trait_candidates_source_run_id", ["source_run_id"]),
            ("ix_character_trait_candidates_reviewed_by_user_id", ["reviewed_by_user_id"]),
            (
                "ix_character_trait_candidates_project_state_character",
                ["project_id", "review_state", "character_key"],
            ),
        ):
            op.create_index(name, "character_trait_candidates", columns)

    inspector = sa.inspect(bind)
    if not inspector.has_table("character_trait_reviews"):
        op.create_table(
            "character_trait_reviews",
            sa.Column("id", sa.String(36), nullable=False),
            sa.Column("project_id", sa.String(36), nullable=False),
            sa.Column("candidate_id", sa.String(36), nullable=False),
            sa.Column("decision", sa.String(24), nullable=False),
            sa.Column("expected_lock_version", sa.Integer(), nullable=False),
            sa.Column("idempotency_key", sa.String(128), nullable=True),
            sa.Column("comment", sa.Text(), nullable=False),
            sa.Column("created_by_user_id", sa.String(36), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.ForeignKeyConstraint(
                ["project_id"], ["projects.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["candidate_id"],
                ["character_trait_candidates.id"],
                ondelete="CASCADE",
            ),
            sa.ForeignKeyConstraint(
                ["created_by_user_id"], ["users.id"], ondelete="SET NULL"
            ),
            sa.UniqueConstraint(
                "candidate_id",
                "idempotency_key",
                name="uq_character_trait_review_idempotency",
            ),
            sa.CheckConstraint(
                "decision IN ('confirm', 'reject', 'supersede')",
                name="ck_character_trait_review_decision",
            ),
            sa.CheckConstraint(
                "expected_lock_version >= 0",
                name="ck_character_trait_review_expected_version",
            ),
        )
        for name, columns in (
            ("ix_character_trait_reviews_project_id", ["project_id"]),
            ("ix_character_trait_reviews_candidate_id", ["candidate_id"]),
            ("ix_character_trait_reviews_created_by_user_id", ["created_by_user_id"]),
            (
                "ix_character_trait_reviews_project_candidate_created",
                ["project_id", "candidate_id", "created_at"],
            ),
        ):
            op.create_index(name, "character_trait_reviews", columns)

    inspector = sa.inspect(bind)
    if not inspector.has_table("analysis_run_character_trait_inputs"):
        op.create_table(
            "analysis_run_character_trait_inputs",
            sa.Column("id", sa.String(36), nullable=False),
            sa.Column("run_id", sa.String(36), nullable=False),
            sa.Column("project_id", sa.String(36), nullable=False),
            sa.Column("candidate_id", sa.String(36), nullable=False),
            sa.Column("confirmation_review_id", sa.String(36), nullable=False),
            sa.Column("candidate_lock_version", sa.Integer(), nullable=False),
            sa.Column("ordinal", sa.Integer(), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("payload_sha256", sa.String(64), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.ForeignKeyConstraint(
                ["run_id"], ["analysis_runs.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["project_id"], ["projects.id"], ondelete="CASCADE"
            ),
            sa.ForeignKeyConstraint(
                ["candidate_id"],
                ["character_trait_candidates.id"],
                ondelete="RESTRICT",
            ),
            sa.ForeignKeyConstraint(
                ["confirmation_review_id"],
                ["character_trait_reviews.id"],
                ondelete="RESTRICT",
            ),
            sa.UniqueConstraint(
                "run_id",
                "candidate_id",
                name="uq_analysis_run_character_trait_candidate",
            ),
            sa.UniqueConstraint(
                "run_id",
                "ordinal",
                name="uq_analysis_run_character_trait_ordinal",
            ),
            sa.CheckConstraint(
                "ordinal >= 0", name="ck_analysis_run_character_trait_ordinal"
            ),
            sa.CheckConstraint(
                "candidate_lock_version >= 0",
                name="ck_analysis_run_character_trait_candidate_version",
            ),
            sa.CheckConstraint(
                "length(payload_sha256) = 64",
                name="ck_analysis_run_character_trait_payload_hash",
            ),
        )
        for name, columns in (
            ("ix_analysis_run_character_trait_inputs_run_id", ["run_id"]),
            ("ix_analysis_run_character_trait_inputs_project_id", ["project_id"]),
            ("ix_analysis_run_character_trait_inputs_candidate_id", ["candidate_id"]),
        ):
            op.create_index(name, "analysis_run_character_trait_inputs", columns)


def downgrade() -> None:
    bind = op.get_bind()
    for table_name in (
        "analysis_run_character_trait_inputs",
        "character_trait_reviews",
        "character_trait_candidates",
        "analysis_run_input_narrative_context",
        "document_narrative_context_revisions",
    ):
        if sa.inspect(bind).has_table(table_name):
            op.drop_table(table_name)
