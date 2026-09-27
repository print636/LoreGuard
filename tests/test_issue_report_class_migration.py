"""The 0021 upgrade must conservatively separate historical character rows."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import MetaData, Table, create_engine, inspect, select
from sqlalchemy.engine import make_url

from app.db import Base


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = "0020_value_boundary_author_axes"
HEAD = "0021_issue_report_classes"
LOCAL_ID = "00000000-0000-0000-0000-000000000001"
WHEN = datetime(2026, 9, 1)


def _config(url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["database_url"] = url
    return config


def _span(document_id: str) -> dict:
    return {
        "document_id": document_id, "document_name": f"{document_id}.md",
        "line_start": 1, "line_end": 1, "text": "来源证据",
    }


def test_old_character_rows_are_backfilled_and_affected_comparison_is_reset(tmp_path):
    url = f"sqlite:///{(tmp_path / 'issue-classes.db').as_posix()}"
    config = _config(url)
    command.upgrade(config, PREVIOUS)
    engine = create_engine(url)
    tables = {
        name: Table(name, MetaData(), autoload_with=engine)
        for name in (
            "projects", "analysis_runs", "issues",
            "analysis_run_comparisons", "issue_comparison_items", "issue_feedback",
        )
    }
    try:
        with engine.begin() as connection:
            connection.execute(tables["projects"].insert().values(
                id="project-old", workspace_id=LOCAL_ID, name="旧项目",
                description="", created_at=WHEN,
            ))
            for run_id in ("run-old", "run-new"):
                connection.execute(tables["analysis_runs"].insert().values(
                    id=run_id, project_id="project-old", status="completed",
                    created_at=WHEN, completed_at=WHEN, input_chars=20,
                    prompt_tokens=0, completion_tokens=0, estimated_cost_usd=0.0,
                    cancel_requested=False, batch_mode="full_review",
                    sensitivity="balanced", batch_coverage={},
                ))
            for issue_id, category, title, judgement in (
                ("formal-old", "character_drift", "林澈的角色设定可能冲突", "contradicts"),
                ("partial-old", "character_drift", "林澈的角色表现需要确认", "contradicts"),
                ("review-old", "character_drift", "林澈的角色表现需要确认", "needs_confirmation"),
                ("unknown-old", "character_drift", "模糊历史标题", None),
                ("rule-old", "fact_conflict", "明确事实冲突", None),
            ):
                connection.execute(tables["issues"].insert().values(
                    id=issue_id, run_id="run-old", category=category,
                    severity="high", confidence=0.9, title=title,
                    explanation="历史说明", evidence=[_span("canon"), _span("draft")],
                    suggestion="检查", extra={"judgement": judgement},
                ))
            connection.execute(tables["analysis_run_comparisons"].insert().values(
                id="comparison-old", project_id="project-old",
                baseline_run_id="run-old", target_run_id="run-new",
                matcher_version="issue-match-v1", status="ready",
                summary={"total_baseline": 5}, provenance={"old": True},
                created_at=WHEN, completed_at=WHEN,
            ))
            connection.execute(tables["issue_comparison_items"].insert().values(
                id="item-old", comparison_id="comparison-old",
                outcome="no_longer_detected", baseline_issue_id="review-old",
                provenance={},
            ))
            connection.execute(tables["issue_feedback"].insert().values(
                id="feedback-old", issue_id="review-old", label="accepted",
                comment="旧反馈", created_at=WHEN,
            ))
    finally:
        engine.dispose()


    command.upgrade(config, HEAD)
    engine = create_engine(url)
    try:
        tables = {
            name: Table(name, MetaData(), autoload_with=engine)
            for name in (
                "issues", "analysis_run_comparisons", "issue_comparison_items",
                "issue_feedback",
            )
        }
        with engine.connect() as connection:
            migrated = {
                row.id: (row.report_class, row.extra)
                for row in connection.execute(select(
                    tables["issues"].c.id,
                    tables["issues"].c.report_class,
                    tables["issues"].c.extra,
                )).all()
            }
            comparison = connection.execute(
                select(tables["analysis_run_comparisons"])
            ).mappings().one()
            remaining_items = connection.execute(
                select(tables["issue_comparison_items"].c.id)
            ).all()
            old_feedback = connection.execute(
                select(tables["issue_feedback"].c.label)
            ).scalar_one()
        assert {key: value[0] for key, value in migrated.items()} == {
            "formal-old": "formal", "partial-old": "review_clue",
            "review-old": "review_clue", "unknown-old": "review_clue",
            "rule-old": "formal",
        }
        for issue_id in ("partial-old", "review-old", "unknown-old"):
            extra = migrated[issue_id][1]
            assert extra["final_outcome"] == "unverifiable"
            assert extra["review_reason"] == "legacy_report_reclassified"
            assert extra["legacy_report_reclassified"] is True
        assert comparison["status"] == "pending"
        assert comparison["summary"] == {}
        assert comparison["provenance"] == {}
        assert comparison["completed_at"] is None
        assert remaining_items == []
        assert old_feedback == "accepted"
        assert "ix_issues_run_report_class_id" in {
            index["name"] for index in inspect(engine).get_indexes("issues")
        }
    finally:
        engine.dispose()


def test_current_create_all_schema_adopts_without_rewriting_existing_clue(tmp_path):
    url = f"sqlite:///{(tmp_path / 'adopt-current.db').as_posix()}"
    config = _config(url)
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        projects = Table("projects", MetaData(), autoload_with=engine)
        runs = Table("analysis_runs", MetaData(), autoload_with=engine)
        issues = Table("issues", MetaData(), autoload_with=engine)
        connection.execute(projects.insert().values(
            id="project-current", workspace_id=LOCAL_ID,
            name="当前库", description="", created_at=WHEN,
        ))
        connection.execute(runs.insert().values(
            id="run-current", project_id="project-current",
            status="completed", created_at=WHEN, completed_at=WHEN,
            input_chars=20, prompt_tokens=0, completion_tokens=0,
            estimated_cost_usd=0.0, cancel_requested=False,
            batch_mode="full_review", sensitivity="balanced",
            batch_coverage={},
        ))
        connection.execute(issues.insert().values(
            id="clue-current", run_id="run-current",
            report_class="review_clue", category="character_drift",
            severity="low", confidence=0.52,
            title="林澈的角色表现需要确认", explanation="一次反向行为",
            evidence=[_span("canon"), _span("draft")],
            suggestion="检查", extra={
                "judgement": "needs_confirmation",
                "final_outcome": "needs_confirmation",
                "review_reason": "single_behavior_is_not_drift",
            },
        ))
    engine.dispose()

    command.stamp(config, PREVIOUS)
    command.upgrade(config, HEAD)
    engine = create_engine(url)
    try:
        issues = Table("issues", MetaData(), autoload_with=engine)
        with engine.connect() as connection:
            report_class, extra = connection.execute(select(
                issues.c.report_class, issues.c.extra,
            )).one()
        assert report_class == "review_clue"
        assert extra["final_outcome"] == "needs_confirmation"
        assert extra["review_reason"] == "single_behavior_is_not_drift"
    finally:
        engine.dispose()


@pytest.mark.skipif(
    not os.environ.get("LOREGUARD_TEST_POSTGRES_URL"),
    reason="LOREGUARD_TEST_POSTGRES_URL is unavailable",
)
def test_postgres_old_character_rows_backfill_in_isolated_schema():
    base_url = os.environ["LOREGUARD_TEST_POSTGRES_URL"]
    schema = f"loreguard_issue_classes_{uuid4().hex[:12]}"
    admin = create_engine(base_url)
    try:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        url = make_url(base_url).update_query_dict(
            {"options": f"-csearch_path={schema},public"}
        ).render_as_string(hide_password=False)
        config = _config(url)
        command.upgrade(config, PREVIOUS)
        engine = create_engine(url)
        try:
            tables = {
                name: Table(name, MetaData(), autoload_with=engine)
                for name in ("projects", "analysis_runs", "issues")
            }
            with engine.begin() as connection:
                connection.execute(tables["projects"].insert().values(
                    id="project-pg-old", workspace_id=LOCAL_ID,
                    name="Postgres 旧行", description="", created_at=WHEN,
                ))
                connection.execute(tables["analysis_runs"].insert().values(
                    id="run-pg-old", project_id="project-pg-old",
                    status="completed", created_at=WHEN, completed_at=WHEN,
                    input_chars=20, prompt_tokens=0, completion_tokens=0,
                    estimated_cost_usd=0.0, cancel_requested=False,
                    batch_mode="full_review", sensitivity="balanced",
                    batch_coverage={},
                ))
                for issue_id, title, judgement in (
                    ("pg-conflict", "林澈的角色设定可能冲突", "contradicts"),
                    ("pg-partial", "林澈的角色表现需要确认", "contradicts"),
                    ("pg-review", "林澈的角色表现需要确认", "needs_confirmation"),
                ):
                    connection.execute(tables["issues"].insert().values(
                        id=issue_id, run_id="run-pg-old",
                        category="character_drift", severity="high",
                        confidence=0.9, title=title,
                        explanation="历史说明",
                        evidence=[_span("canon"), _span("draft")],
                        suggestion="检查", extra={"judgement": judgement},
                    ))
        finally:
            engine.dispose()

        command.upgrade(config, HEAD)
        engine = create_engine(url)
        try:
            issues = Table("issues", MetaData(), autoload_with=engine)
            with engine.connect() as connection:
                classes = dict(connection.execute(
                    select(issues.c.id, issues.c.report_class)
                ).all())
            assert classes == {
                "pg-conflict": "formal",
                "pg-partial": "review_clue",
                "pg-review": "review_clue",
            }
        finally:
            engine.dispose()
    finally:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        admin.dispose()
