"""Historical character reports must satisfy the current formal-proof contract."""

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
HEAD = "0022_character_formal_proof"
LOCAL_ID = "00000000-0000-0000-0000-000000000001"
WHEN = datetime(2026, 9, 1)


def _config(url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["database_url"] = url
    return config


def _span(document_id: str, *, line: int = 1, text: str = "来源证据") -> dict:
    return {
        "document_id": document_id, "document_name": f"{document_id}.md",
        "line_start": line, "line_end": line, "text": text,
    }


def _formal_evidence() -> list[dict]:
    return [
        _span("canon", text="林澈一向内向。"),
        _span("draft", line=2, text="第一天林澈主动攀谈。"),
        _span("draft", line=8, text="三天后林澈主动主持聚会。"),
    ]


def _formal_extra() -> dict:
    return {
        "judgement": "contradicts",
        "final_outcome": "conflict",
        "evidence_binding": "review_citations_v1",
        "event_independence": "yes",
        "independent_event_citations": ["C01", "C02"],
        "event_identity_verification": "different_events",
        "material_coverage": "complete",
        "explanation_coverage": "complete",
        "explanation_review_executed": True,
        "review_citation_refs": [
            {
                "handle": "B01", "role": "B", "evidence_index": 0,
                "response_index": 0,
            },
            {
                "handle": "C01", "role": "C", "evidence_index": 1,
                "response_index": 1,
            },
            {
                "handle": "C02", "role": "C", "evidence_index": 2,
                "response_index": 2,
            },
        ],
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
                ("formal-proven", "character_drift", "林澈的角色设定可能冲突", "contradicts"),
                ("partial-old", "character_drift", "林澈的角色表现需要确认", "contradicts"),
                ("review-old", "character_drift", "林澈的角色表现需要确认", "needs_confirmation"),
                ("unknown-old", "character_drift", "模糊历史标题", None),
                ("rule-old", "fact_conflict", "明确事实冲突", None),
            ):
                proven = issue_id == "formal-proven"
                evidence = (
                    _formal_evidence()
                    if proven else [_span("canon"), _span("draft")]
                )
                extra = (
                    _formal_extra()
                    if proven else {"judgement": judgement}
                )
                connection.execute(tables["issues"].insert().values(
                    id=issue_id, run_id="run-old", category=category,
                    severity="high", confidence=0.9, title=title,
                    explanation="历史说明", evidence=evidence,
                    suggestion="检查", extra=extra,
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


    # First exercise the historical 0021 classification, then recreate a
    # ready derived comparison so 0022 itself must invalidate it when the
    # formerly formal row lacks the new proof contract.
    command.upgrade(config, "0021_issue_report_classes")
    engine = create_engine(url)
    try:
        comparisons = Table(
            "analysis_run_comparisons", MetaData(), autoload_with=engine
        )
        items = Table("issue_comparison_items", MetaData(), autoload_with=engine)
        with engine.begin() as connection:
            connection.execute(
                comparisons.update()
                .where(comparisons.c.id == "comparison-old")
                .values(
                    status="ready",
                    summary={"total_baseline": 2},
                    provenance={"pre_0022": True},
                    completed_at=WHEN,
                )
            )
            connection.execute(items.insert().values(
                id="item-pre-proof", comparison_id="comparison-old",
                outcome="no_longer_detected", baseline_issue_id="formal-old",
                provenance={},
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
            "formal-old": "review_clue", "formal-proven": "formal",
            "partial-old": "review_clue",
            "review-old": "review_clue", "unknown-old": "review_clue",
            "rule-old": "formal",
        }
        for issue_id in ("partial-old", "review-old", "unknown-old"):
            extra = migrated[issue_id][1]
            assert extra["final_outcome"] == "unverifiable"
            assert extra["review_reason"] == "legacy_report_reclassified"
            assert extra["legacy_report_reclassified"] is True
        demoted = migrated["formal-old"][1]
        assert demoted["final_outcome"] == "needs_confirmation"
        assert demoted["review_reason"] == "legacy_formal_proof_incomplete"
        assert demoted["legacy_formal_proof_incomplete"] is True
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


def test_0022_demotes_corrupt_formal_citation_bindings(tmp_path):
    url = f"sqlite:///{(tmp_path / 'corrupt-formal-proof.db').as_posix()}"
    config = _config(url)
    command.upgrade(config, "0021_issue_report_classes")
    engine = create_engine(url)
    tables = {
        name: Table(name, MetaData(), autoload_with=engine)
        for name in ("projects", "analysis_runs", "issues")
    }

    forged_handle = _formal_extra()
    forged_handle["review_citation_refs"][1]["handle"] = "B02"
    duplicate_response_index = _formal_extra()
    duplicate_response_index["review_citation_refs"][2]["response_index"] = 1
    missing_response_index = _formal_extra()
    missing_response_index["review_citation_refs"][1].pop("response_index")
    reordered_response_index = _formal_extra()
    reordered_response_index["review_citation_refs"][1]["response_index"] = 2
    reordered_response_index["review_citation_refs"][2]["response_index"] = 1
    broken_baseline = _formal_evidence()
    broken_baseline[0] = {**broken_baseline[0], "line_start": 0}
    cases = (
        ("forged-handle", _formal_evidence(), forged_handle),
        ("duplicate-response-index", _formal_evidence(), duplicate_response_index),
        ("missing-response-index", _formal_evidence(), missing_response_index),
        ("reordered-response-index", _formal_evidence(), reordered_response_index),
        ("broken-baseline-span", broken_baseline, _formal_extra()),
    )
    try:
        with engine.begin() as connection:
            connection.execute(tables["projects"].insert().values(
                id="project-corrupt-proof", workspace_id=LOCAL_ID,
                name="损坏证明项目", description="", created_at=WHEN,
            ))
            connection.execute(tables["analysis_runs"].insert().values(
                id="run-corrupt-proof", project_id="project-corrupt-proof",
                status="completed", created_at=WHEN, completed_at=WHEN,
                input_chars=20, prompt_tokens=0, completion_tokens=0,
                estimated_cost_usd=0.0, cancel_requested=False,
                batch_mode="full_review", sensitivity="balanced",
                batch_coverage={},
            ))
            for issue_id, evidence, extra in cases:
                connection.execute(tables["issues"].insert().values(
                    id=issue_id, run_id="run-corrupt-proof",
                    report_class="formal", category="character_drift",
                    severity="high", confidence=0.9,
                    title="林澈的角色设定可能冲突", explanation="历史说明",
                    evidence=evidence, suggestion="检查", extra=extra,
                ))
    finally:
        engine.dispose()

    command.upgrade(config, HEAD)
    engine = create_engine(url)
    try:
        issues = Table("issues", MetaData(), autoload_with=engine)
        with engine.connect() as connection:
            migrated = {
                row.id: (row.report_class, row.extra)
                for row in connection.execute(select(
                    issues.c.id, issues.c.report_class, issues.c.extra,
                )).all()
            }
        assert set(migrated) == {row[0] for row in cases}
        assert all(value[0] == "review_clue" for value in migrated.values())
        assert all(
            value[1]["review_reason"] == "legacy_formal_proof_incomplete"
            for value in migrated.values()
        )
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
                "pg-conflict": "review_clue",
                "pg-partial": "review_clue",
                "pg-review": "review_clue",
            }
        finally:
            engine.dispose()
    finally:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        admin.dispose()
