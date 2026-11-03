"""Durable admitted draft observations and frozen matching decisions.

Revision ID: 0025_character_observations
Revises: 0024_project_import_foundation
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0025_character_observations"
down_revision = "0024_project_import_foundation"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("character_observation_recordings"):
        op.create_table(
            "character_observation_recordings",
            sa.Column("run_id", sa.String(36), sa.ForeignKey("analysis_runs.id", ondelete="CASCADE"), primary_key=True),
            sa.Column("project_id", sa.String(36), sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
            sa.Column("schema_version", sa.String(48), nullable=False),
            sa.Column("status", sa.String(24), nullable=False),
            sa.Column("source_manifest_sha256", sa.String(64), nullable=False),
            sa.Column("baseline_manifest_sha256", sa.String(64), nullable=False),
            sa.Column("coverage", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.CheckConstraint("status IN ('in_progress', 'complete', 'partial', 'failed', 'not_applicable')", name="ck_character_observation_recording_status"),
            sa.CheckConstraint("length(source_manifest_sha256) = 64", name="ck_character_observation_source_hash"),
            sa.CheckConstraint("length(baseline_manifest_sha256) = 64", name="ck_character_observation_baseline_hash"),
        )
        op.create_index("ix_character_observation_recordings_project_id", "character_observation_recordings", ["project_id"])
    if not inspector.has_table("character_observations"):
        op.create_table(
            "character_observations",
            sa.Column("run_id", sa.String(36), sa.ForeignKey("character_observation_recordings.run_id", ondelete="CASCADE"), primary_key=True),
            sa.Column("observation_id", sa.String(36), primary_key=True),
            sa.Column("source_run_input_id", sa.String(36), sa.ForeignKey("analysis_run_inputs.id", ondelete="CASCADE"), nullable=False),
            sa.Column("ordinal", sa.Integer(), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("payload_sha256", sa.String(64), nullable=False),
            sa.UniqueConstraint("run_id", "ordinal", name="uq_character_observation_ordinal"),
            sa.CheckConstraint("ordinal >= 0", name="ck_character_observation_ordinal"),
            sa.CheckConstraint("length(payload_sha256) = 64", name="ck_character_observation_payload_hash"),
        )
        op.create_index("ix_character_observations_source_run_input_id", "character_observations", ["source_run_input_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("character_observation_recordings") and bind.execute(sa.text("SELECT 1 FROM character_observation_recordings LIMIT 1")).first():
        raise RuntimeError("character observation history exists; downgrade requires an explicit backup and controlled rollback")
    if inspector.has_table("character_observations"):
        op.drop_table("character_observations")
    if inspector.has_table("character_observation_recordings"):
        op.drop_table("character_observation_recordings")
