"""Review clues are visible to the owner without becoming formal issues."""

from __future__ import annotations

from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import (
    AnalysisDiagnosticRow, AnalysisRunComparisonRow, AnalysisRunInputRow,
    AnalysisRunRow,
    FeedbackRow, IssueComparisonItemRow, IssueRow, ProjectRow, SessionLocal,
    WorkspaceRow,
)
from app.character_consistency_stage import CharacterConsistencyStageResult
from app.domain import ConsistencyIssue, EvidenceSpan, IssueCategory, Severity
from app.main import app, settings
from app.run_comparison import MATCHER_VERSION, mark_comparison_unverifiable
from app.service import (
    _character_review_source_index,
    _verified_character_review_clues,
    document_content_sha256,
    execute_analysis,
)
from app.pipeline import DocumentInput


def _evidence(label: str) -> dict:
    return {
        "document_id": f"{label}-{uuid4().hex}",
        "document_name": f"{label}.md",
        "line_start": 1, "line_end": 1,
        "text": f"{label}来源原文",
    }


def _issue(run_id: str, report_class: str, *, label: str) -> IssueRow:
    return IssueRow(
        run_id=run_id, report_class=report_class,
        category="character_drift", severity="medium", confidence=0.68,
        title=f"{label}角色表现需要确认",
        explanation=f"{label}解释文本", suggestion="核对设定",
        evidence=[_evidence("baseline"), _evidence("draft")],
        extra={
            "character_key": label, "dimension": "core_personality",
            "trait_key": "社交主动性",
            "final_outcome": (
                "conflict" if report_class == "formal" else "needs_confirmation"
            ),
            "review_reason": "single_behavior_is_not_drift",
        },
    )


def _freeze_issue_evidence(db, run_id: str, *issues: IssueRow) -> None:
    frozen: dict[str, dict] = {}
    for issue in issues:
        for span in issue.evidence:
            existing = frozen.get(span["document_id"])
            if existing is not None:
                assert existing == span
                continue
            frozen[span["document_id"]] = span
    for ordinal, span in enumerate(frozen.values()):
        db.add(AnalysisRunInputRow(
            run_id=run_id,
            document_id=span["document_id"],
            document_name=span["document_name"],
            document_version=1,
            content=span["text"],
            content_sha256=document_content_sha256(span["text"]),
            ordinal=ordinal,
        ))


def test_modern_first_pass_clue_role_bindings_fail_closed():
    baseline = EvidenceSpan(
        document_id="shared", document_name="story.md",
        line_start=1, line_end=1, text="旧版本：祁雾说话直来直往。",
    )
    current = baseline.model_copy(
        update={"text": "新版本：祁雾从不直接回答问题。"}
    )
    refs = [
        {"handle": "B01", "role": "B", "evidence_index": 0, "response_index": 0},
        {"handle": "C01", "role": "C", "evidence_index": 1, "response_index": 1},
    ]
    valid = ConsistencyIssue(
        category=IssueCategory.character_drift,
        severity=Severity.medium,
        confidence=0.68,
        title="祁雾的角色表现需要确认",
        explanation="事件同一性复核未完成。",
        evidence=[baseline, current],
        suggestion="核对设定",
        metadata={
            "confirmed_candidate_id": "candidate-1",
            "final_outcome": "needs_confirmation",
            "review_reason": "event_identity_verification_unavailable",
            "evidence_binding": "first_pass_review_citations_v1",
            "review_citation_refs": refs,
        },
    )
    malformed_handle = valid.model_copy(update={"metadata": {
        **valid.metadata,
        "review_citation_refs": [{**refs[0], "handle": "Bgarbage"}, refs[1]],
    }})
    mixed_role_index = valid.model_copy(update={"metadata": {
        **valid.metadata,
        "review_citation_refs": [refs[0], {**refs[1], "evidence_index": 0}],
    }})
    unverified_event_metadata = valid.model_copy(update={"metadata": {
        **valid.metadata,
        "event_identity_verification": "different_events",
    }})

    accepted, rejected = _verified_character_review_clues(
        (valid, malformed_handle, mixed_role_index, unverified_event_metadata),
        [DocumentInput("shared", "story.md", current.text)],
        confirmed_trait_evidence_by_candidate={
            "candidate-1": frozenset({(
                baseline.document_id,
                baseline.document_name,
                baseline.line_start,
                baseline.line_end,
                baseline.text,
            )}),
        },
    )

    assert accepted == (valid,)
    assert rejected == 3


def test_clues_are_separate_from_all_formal_surfaces_and_feedback_writes():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"线索隔离-{uuid4().hex}"}
        ).json()
        with SessionLocal() as db:
            baseline_run = AnalysisRunRow(
                project_id=project["id"], status="completed",
            )
            target_run = AnalysisRunRow(
                project_id=project["id"], status="completed",
            )
            db.add_all((baseline_run, target_run))
            db.flush()
            formal = _issue(baseline_run.id, "formal", label="正式")
            clue = _issue(baseline_run.id, "review_clue", label="待复核")
            db.add_all((formal, clue))
            _freeze_issue_evidence(db, baseline_run.id, clue)
            db.flush()
            # A historical clue may already have feedback; retain it for audit.
            db.add(FeedbackRow(
                issue_id=clue.id, label="accepted", comment="历史反馈",
            ))
            db.commit()
            run_id, target_id, formal_id, clue_id = (
                baseline_run.id, target_run.id, formal.id, clue.id
            )

        issues = client.get(f"/api/v1/analysis-runs/{run_id}/issues")
        assert issues.status_code == 200
        assert [item["id"] for item in issues.json()] == [formal_id]
        assert issues.json()[0]["report_class"] == "formal"

        clues = client.get(f"/api/v1/analysis-runs/{run_id}/review-clues")
        assert clues.status_code == 200
        assert clues.headers["cache-control"] == "private, no-store"
        assert clues.json()["truncated"] is False
        assert [item["id"] for item in clues.json()["items"]] == [clue_id]
        assert clues.json()["items"][0]["report_class"] == "review_clue"
        assert len(clues.json()["items"][0]["evidence"]) == 2

        exported = client.get(f"/api/v1/analysis-runs/{run_id}/export.md")
        assert exported.status_code == 200
        assert "待复核解释文本" not in exported.text
        assert "正式解释文本" in exported.text
        drift = client.get(
            f"/api/v1/projects/{project['id']}/drift-issues"
        )
        assert drift.status_code == 200
        assert drift.json()["total"] == 1
        assert drift.json()["items"][0]["id"] == formal_id
        for surface in ("graph", "timeline"):
            response = client.get(
                f"/api/v1/analysis-runs/{run_id}/{surface}"
            )
            assert response.status_code == 200
            assert clue_id not in response.text

        rejected = client.post(
            f"/api/v1/issues/{clue_id}/feedback",
            json={"label": "resolved", "comment": "不应写入"},
        )
        assert rejected.status_code == 409
        old_history = client.get(f"/api/v1/issues/{clue_id}/feedback")
        assert old_history.status_code == 200
        assert old_history.json()["latest"]["comment"] == "历史反馈"

        with SessionLocal() as db:
            comparison = AnalysisRunComparisonRow(
                project_id=project["id"], baseline_run_id=run_id,
                target_run_id=target_id,
                matcher_version="issue-match-v1", status="ready",
                summary={"total_baseline": 2}, provenance={"old": True},
            )
            db.add(comparison)
            db.flush()
            db.add(IssueComparisonItemRow(
                comparison_id=comparison.id, outcome="no_longer_detected",
                baseline_issue_id=clue_id, provenance={},
            ))
            db.commit()
        refreshed = client.get(
            f"/api/v1/analysis-runs/{target_id}/comparison"
        )
        assert refreshed.status_code == 200
        assert refreshed.json()["matcher"]["version"] == MATCHER_VERSION
        assert refreshed.json()["summary"]["total_baseline"] == 1
        assert clue_id not in refreshed.text
        with SessionLocal() as db:
            fallback_target = AnalysisRunRow(
                project_id=project["id"], status="completed",
            )
            db.add(fallback_target)
            db.flush()
            db.add(AnalysisRunComparisonRow(
                project_id=project["id"], baseline_run_id=run_id,
                target_run_id=fallback_target.id,
                matcher_version=MATCHER_VERSION, status="pending",
            ))
            db.flush()
            result = mark_comparison_unverifiable(db, fallback_target.id)
            assert result.summary["total_baseline"] == 1
            assert result.summary["total_target"] == 0
            db.commit()


def test_clue_endpoint_is_bounded_and_workspace_scoped():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"线索分页-{uuid4().hex}"}
        ).json()
        with SessionLocal() as db:
            run = AnalysisRunRow(project_id=project["id"], status="completed")
            db.add(run)
            db.flush()
            clues = [
                _issue(run.id, "review_clue", label=f"线索{index}")
                for index in range(65)
            ]
            db.add_all(clues)
            _freeze_issue_evidence(db, run.id, *clues)
            other_workspace = WorkspaceRow(name="private workspace")
            db.add(other_workspace)
            db.flush()
            private_project = ProjectRow(
                workspace_id=other_workspace.id, name="private project",
            )
            db.add(private_project)
            db.flush()
            private_run = AnalysisRunRow(
                project_id=private_project.id, status="completed",
            )
            db.add(private_run)
            db.commit()
            run_id, private_run_id = run.id, private_run.id

        response = client.get(f"/api/v1/analysis-runs/{run_id}/review-clues")
        assert response.status_code == 200
        assert len(response.json()["items"]) == 64
        assert response.json()["truncated"] is True
        assert client.get(
            f"/api/v1/analysis-runs/{private_run_id}/review-clues"
        ).status_code == 404
        with SessionLocal() as db:
            assert len(db.scalars(select(IssueRow).where(
                IssueRow.run_id == run_id,
                IssueRow.report_class == "review_clue",
            )).all()) == 65


def test_scan_limit_reports_uninspected_rows_without_claiming_they_are_invalid():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"线索扫描上限-{uuid4().hex}"}
        ).json()
        with SessionLocal() as db:
            run = AnalysisRunRow(project_id=project["id"], status="completed")
            db.add(run)
            db.flush()
            for index in range(1025):
                db.add(IssueRow(
                    id=f"00000000-0000-0000-0000-{index:012x}",
                    run_id=run.id, report_class="review_clue",
                    category="character_drift", severity="low", confidence=0.1,
                    title="证据不完整的旧行", explanation="不可展示",
                    evidence=[], suggestion="重新分析", extra={},
                ))
            valid = _issue(run.id, "review_clue", label="末尾有效线索")
            valid.id = "ffffffff-ffff-ffff-ffff-ffffffffffff"
            db.add(valid)
            _freeze_issue_evidence(db, run.id, valid)
            db.commit()
            run_id = run.id
        response = client.get(f"/api/v1/analysis-runs/{run_id}/review-clues")
        assert response.status_code == 200
        assert response.json() == {
            "items": [],
            "truncated": False,
            "unavailable_count": 1025,
            "scan_limited": True,
        }


def test_service_persists_only_clues_rebound_to_frozen_run_input():
    baseline_text = "林澈平时很内向。"
    draft_text = "林澈主动与陌生人交谈。"
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"冻结线索-{uuid4().hex}"}
        ).json()
        baseline = client.post(
            f"/api/v1/projects/{project['id']}/documents/text",
            json={"name": "profile.md", "content": baseline_text},
        ).json()
        draft = client.post(
            f"/api/v1/projects/{project['id']}/documents/text",
            json={"name": "draft.md", "content": draft_text},
        ).json()
        with patch("app.main.dispatch_analysis"):
            start = client.post(
                f"/api/v1/projects/{project['id']}/analysis-runs"
            )
        assert start.status_code == 202, start.text
        run_id = start.json()["id"]
        baseline_span = EvidenceSpan(
            document_id=baseline["id"], document_name="profile.md",
            line_start=1, line_end=1, text=baseline_text,
        )
        draft_span = EvidenceSpan(
            document_id=draft["id"], document_name="draft.md",
            line_start=1, line_end=1, text=draft_text,
        )
        def clue_with(current_span: EvidenceSpan) -> ConsistencyIssue:
            return ConsistencyIssue(
                category=IssueCategory.character_drift,
                severity=Severity.low, confidence=0.52,
                title="林澈的角色表现需要确认",
                explanation="一次反向行为，等待作者复核。",
                evidence=[baseline_span, current_span],
                suggestion="核对角色设定",
                metadata={
                    "character_key": "林澈",
                    "dimension": "core_personality",
                    "trait_key": "社交主动性",
                    "final_outcome": "needs_confirmation",
                    "review_reason": "single_behavior_is_not_drift",
                },
            )

        valid_clue = clue_with(draft_span)
        forged_clue = clue_with(draft_span.model_copy(update={
            "text": "未出现在冻结文档中的描述"
        }))
        missing_current = valid_clue.model_copy(update={
            "evidence": [baseline_span]
        })
        stage_result = CharacterConsistencyStageResult(
            issues=(), review_clues=(valid_clue, forged_clue, missing_current),
            diagnostics={"outcome": "completed", "usage": {"attempted_calls": 0}},
        )
        with (
            patch.object(settings, "enable_character_consistency", True),
            patch("app.service.CharacterConsistencyStage.run", return_value=stage_result),
        ):
            execute_analysis(run_id, raise_on_failure=True)

        with SessionLocal() as db:
            run = db.get(AnalysisRunRow, run_id)
            rows = db.scalars(select(IssueRow).where(
                IssueRow.run_id == run_id,
                IssueRow.report_class == "review_clue",
            )).all()
            diagnostics = db.get(AnalysisDiagnosticRow, run_id).payload
            assert run.status == "completed"
            assert len(rows) == 1
            assert rows[0].evidence[1]["text"] == draft_text
            assert diagnostics["character_review_clues"] == {
                "stored": 1, "rejected_unbound": 2,
            }
            valid_id = rows[0].id
            rows[0].extra = {
                **rows[0].extra,
                "final_outcome": "unverifiable",
                "review_reason": "legacy_report_reclassified",
                "legacy_report_reclassified": True,
            }
            second_valid_id = str(uuid4())
            db.add(IssueRow(
                id=second_valid_id,
                run_id=run_id, report_class="review_clue",
                category="character_drift", severity="low", confidence=0.2,
                title="另一条旧线索", explanation="已定位两侧原文证据",
                evidence=[baseline_span.model_dump(), draft_span.model_dump()],
                suggestion="重新分析", extra=dict(rows[0].extra),
            ))
            db.add(IssueRow(
                run_id=run_id, report_class="review_clue",
                category="character_drift", severity="low", confidence=0.2,
                title="脏旧线索", explanation="只有单侧证据",
                evidence=[baseline_span.model_dump()], suggestion="重新分析",
                extra={
                    "final_outcome": "unverifiable",
                    "review_reason": "legacy_report_reclassified",
                    "legacy_report_reclassified": True,
                },
            ))
            db.add(IssueRow(
                run_id=run_id, report_class="review_clue",
                category="character_drift", severity="low", confidence=0.2,
                title="旧线索字段损坏", explanation="原文两侧仍在",
                evidence=[baseline_span.model_dump(), draft_span.model_dump()],
                suggestion="重新分析",
                extra={"legacy_report_reclassified": True},
            ))
            db.commit()
        response = client.get(f"/api/v1/analysis-runs/{run_id}/issues")
        assert response.status_code == 200
        assert all(row["id"] != valid_id for row in response.json())
        with patch(
            "app.main._character_review_source_index",
            wraps=_character_review_source_index,
        ) as index_builder:
            clue_response = client.get(
                f"/api/v1/analysis-runs/{run_id}/review-clues"
            )
        assert index_builder.call_count == 1
        assert clue_response.status_code == 200
        assert clue_response.json()["truncated"] is False
        assert clue_response.json()["unavailable_count"] == 2
        assert {row["id"] for row in clue_response.json()["items"]} == {
            valid_id, second_valid_id,
        }
