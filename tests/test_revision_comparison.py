from __future__ import annotations

import unittest
from hashlib import sha256
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from app.db import (
    AnalysisDiagnosticRow,
    AnalysisRunComparisonRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    DocumentRow,
    FeedbackRow,
    IssueComparisonItemRow,
    IssueRow,
    ProjectRow,
    SessionLocal,
    WorkspaceRow,
)
from app.main import app, settings, write_limiter
from app.run_comparison import (
    MATCHER_VERSION,
    _snapshot_valid,
    materialize_run_comparison,
)
from app.service import capture_run_inputs, execute_analysis


def _runtime() -> dict:
    return {
        "capabilities": {
            "model_extraction": False,
            "issue_evidence_review": False,
            "record_repair_agent": False,
            "evidence_investigator": False,
            "embeddings": False,
        },
        "chat_provider": {"model_alias": "test", "temperature": 0},
        "rag": {"strategy": "test"},
    }


def _diagnostic(*, partial_fallback: bool = False) -> dict:
    return {
        "model": {"partial_fallback": partial_fallback},
        "runtime_provenance": _runtime(),
    }


def _evidence(document_id: str, text: str, line: int = 1) -> list[dict]:
    return [
        {
            "document_id": document_id,
            "document_name": "chapter.md",
            "line_start": line,
            "line_end": line,
            "text": text,
        }
    ]


def _issue(
    run_id: str,
    document_id: str,
    subject: str,
    *,
    title: str | None = None,
    line: int = 1,
) -> IssueRow:
    return IssueRow(
        run_id=run_id,
        category="fact_conflict",
        severity="high",
        confidence=0.96,
        title=title or f"{subject}设定冲突",
        explanation="同一事实存在冲突",
        evidence=_evidence(document_id, f"{subject}的状态前后不一致", line),
        suggestion="统一设定",
        extra={"subject": subject, "predicate": "状态"},
    )


class RevisionComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        write_limiter.events.clear()

    def tearDown(self) -> None:
        write_limiter.events.clear()

    def _project_and_baseline(self, client: TestClient, content: str = "第一版"):
        project = client.post(
            "/api/v1/projects", json={"name": f"复检-{uuid4().hex}"}
        ).json()
        document = client.post(
            f"/api/v1/projects/{project['id']}/documents/text",
            json={"name": "chapter.md", "content": content},
        ).json()
        with patch("app.main.dispatch_analysis"):
            baseline = client.post(
                f"/api/v1/projects/{project['id']}/analysis-runs"
            ).json()
        return project, document, baseline

    def _set_completed(
        self, run_id: str, *, partial_fallback: bool = False
    ) -> None:
        with SessionLocal() as db:
            run = db.get(AnalysisRunRow, run_id)
            run.status = "completed"
            diagnostic = db.get(AnalysisDiagnosticRow, run_id)
            if diagnostic is None:
                db.add(
                    AnalysisDiagnosticRow(
                        run_id=run_id,
                        payload=_diagnostic(partial_fallback=partial_fallback),
                    )
                )
            else:
                diagnostic.payload = _diagnostic(
                    partial_fallback=partial_fallback
                )
            db.commit()

    def _new_revision_and_recheck(
        self,
        client: TestClient,
        project: dict,
        document: dict,
        baseline: dict,
        content: str = "第二版",
        key: str | None = None,
    ) -> dict:
        revision = client.post(
            f"/api/v1/projects/{project['id']}/documents/text",
            json={
                "name": "chapter.md",
                "content": content,
                "replace_document_id": document["id"],
            },
        )
        self.assertEqual(201, revision.status_code, revision.text)
        headers = {"Idempotency-Key": key} if key else {}
        with patch("app.main.dispatch_analysis"):
            response = client.post(
                f"/api/v1/analysis-runs/{baseline['id']}/rechecks",
                headers=headers,
            )
        self.assertEqual(202, response.status_code, response.text)
        return response.json()

    def test_comparison_is_conservative_feedback_aware_and_paginated(self):
        with TestClient(app) as client:
            project, document, baseline = self._project_and_baseline(client)
            self._set_completed(baseline["id"])
            with SessionLocal() as db:
                persisting = _issue(
                    baseline["id"], document["id"], "林澈", line=1
                )
                gone = _issue(baseline["id"], document["id"], "苏弦", line=2)
                false_positive = _issue(
                    baseline["id"], document["id"], "叶峤", line=3
                )
                db.add_all([persisting, gone, false_positive])
                db.flush()
                db.add_all(
                    [
                        FeedbackRow(issue_id=persisting.id, label="resolved"),
                        FeedbackRow(
                            issue_id=false_positive.id, label="false_positive"
                        ),
                    ]
                )
                db.commit()

            target = self._new_revision_and_recheck(
                client, project, document, baseline
            )
            self._set_completed(target["id"])
            with SessionLocal() as db:
                target_snapshot = db.scalar(
                    select(DocumentRow).where(
                        DocumentRow.project_id == project["id"],
                        DocumentRow.active.is_(True),
                    )
                )
                db.add_all(
                    [
                        _issue(
                            target["id"],
                            target_snapshot.id,
                            "林澈",
                            title="林澈仍然冲突",
                            line=8,
                        ),
                        _issue(
                            target["id"], target_snapshot.id, "新角色", line=9
                        ),
                    ]
                )
                db.commit()

            response = client.get(
                f"/api/v1/analysis-runs/{target['id']}/comparison?limit=2"
            )
            self.assertEqual(200, response.status_code, response.text)
            payload = response.json()
            self.assertEqual("ready", payload["status"])
            self.assertEqual("comparable", payload["provenance"]["compatibility"]["status"])
            self.assertEqual(
                {
                    "no_longer_detected": 2,
                    "persisting": 1,
                    "new": 1,
                    "unverifiable": 0,
                    "actionable_no_longer_detected": 1,
                    "baseline_false_positive": 1,
                    "total_baseline": 3,
                    "total_target": 2,
                },
                payload["summary"],
            )
            self.assertEqual(2, payload["page"]["returned"])
            all_items = client.get(
                f"/api/v1/analysis-runs/{target['id']}/comparison?limit=20"
            ).json()["items"]
            persistent_item = next(
                row for row in all_items if row["outcome"] == "persisting"
            )
            self.assertEqual(
                "resolved", persistent_item["baseline_latest_feedback"]["label"]
            )
            false_positive_item = next(
                row
                for row in all_items
                if row["baseline_latest_feedback"]
                and row["baseline_latest_feedback"]["label"] == "false_positive"
            )
            self.assertEqual("no_longer_detected", false_positive_item["outcome"])
            self.assertFalse(
                payload["matcher"]["semantic_equivalence_guaranteed"]
            )
            filtered = client.get(
                f"/api/v1/analysis-runs/{target['id']}/comparison"
                "?outcome=new&limit=1"
            ).json()
            self.assertEqual(1, filtered["page"]["total"])
            self.assertFalse(filtered["page"]["has_more"])
            self.assertEqual({"new"}, {row["outcome"] for row in filtered["items"]})
            self.assertEqual(
                422,
                client.get(
                    f"/api/v1/analysis-runs/{target['id']}/comparison"
                    "?outcome=resolved"
                ).status_code,
            )
            frozen_summary = dict(payload["summary"])
            feedback_change = client.post(
                f"/api/v1/issues/{persistent_item['baseline_issue']['id']}/feedback",
                json={"label": "false_positive", "comment": "比较完成后追加"},
            )
            self.assertEqual(201, feedback_change.status_code, feedback_change.text)
            frozen = client.get(
                f"/api/v1/analysis-runs/{target['id']}/comparison?limit=20"
            ).json()
            frozen_persisting = next(
                row for row in frozen["items"] if row["outcome"] == "persisting"
            )
            self.assertEqual(frozen_summary, frozen["summary"])
            self.assertEqual(
                "resolved",
                frozen_persisting["baseline_latest_feedback"]["label"],
            )

    def test_recheck_requires_changes_and_is_idempotent(self):
        with TestClient(app) as client:
            project, document, baseline = self._project_and_baseline(client)
            self._set_completed(baseline["id"])
            unchanged = client.post(
                f"/api/v1/analysis-runs/{baseline['id']}/rechecks"
            )
            self.assertEqual(409, unchanged.status_code)
            self.assertEqual(
                "no_document_changes", unchanged.json()["detail"]["code"]
            )
            revision = client.post(
                f"/api/v1/projects/{project['id']}/documents/text",
                json={
                    "name": "chapter.md",
                    "content": "第二版",
                    "replace_document_id": document["id"],
                },
            )
            self.assertEqual(201, revision.status_code, revision.text)
            headers = {"Idempotency-Key": "recheck-idempotent-1"}
            with patch("app.main.dispatch_analysis") as dispatch:
                first = client.post(
                    f"/api/v1/analysis-runs/{baseline['id']}/rechecks",
                    headers=headers,
                )
                second = client.post(
                    f"/api/v1/analysis-runs/{baseline['id']}/rechecks",
                    headers=headers,
                )
            self.assertEqual(first.json()["id"], second.json()["id"])
            self.assertTrue(second.json()["deduplicated"])
            dispatch.assert_called_once_with(first.json()["id"])
            exact = client.get(
                f"/api/v1/analysis-runs/{first.json()['id']}"
            ).json()
            self.assertEqual(2, exact["input_documents"][0]["document_version"])
            conflict = client.post(
                f"/api/v1/projects/{project['id']}/analysis-runs",
                headers=headers,
            )
            self.assertEqual(409, conflict.status_code)
            self.assertEqual(
                "idempotency_key_conflict", conflict.json()["detail"]["code"]
            )

    def test_ambiguous_identity_and_degraded_run_fail_closed(self):
        with TestClient(app) as client:
            project, document, baseline = self._project_and_baseline(client)
            self._set_completed(baseline["id"])
            with SessionLocal() as db:
                db.add(_issue(baseline["id"], document["id"], "林澈"))
                db.commit()
            target = self._new_revision_and_recheck(
                client, project, document, baseline
            )
            self._set_completed(target["id"])
            with SessionLocal() as db:
                target_document = db.scalar(
                    select(DocumentRow).where(
                        DocumentRow.project_id == project["id"],
                        DocumentRow.active.is_(True),
                    )
                )
                db.add_all(
                    [
                        _issue(target["id"], target_document.id, "林澈", line=2),
                        _issue(target["id"], target_document.id, "林澈", line=3),
                    ]
                )
                db.commit()
                materialize_run_comparison(db, target["id"])
                db.commit()
                comparison = db.scalar(
                    select(AnalysisRunComparisonRow).where(
                        AnalysisRunComparisonRow.target_run_id == target["id"]
                    )
                )
                self.assertEqual(3, comparison.summary["unverifiable"])
                self.assertEqual(0, comparison.summary["persisting"])
                reasons = {
                    row.provenance["reason"]
                    for row in db.scalars(
                        select(IssueComparisonItemRow).where(
                            IssueComparisonItemRow.comparison_id == comparison.id
                        )
                    ).all()
                }
                self.assertEqual({"ambiguous_identity"}, reasons)

            second_project, second_document, second_baseline = (
                self._project_and_baseline(client)
            )
            self._set_completed(second_baseline["id"])
            with SessionLocal() as db:
                db.add(
                    _issue(
                        second_baseline["id"], second_document["id"], "苏弦"
                    )
                )
                db.commit()
            second_target = self._new_revision_and_recheck(
                client,
                second_project,
                second_document,
                second_baseline,
            )
            self._set_completed(second_target["id"], partial_fallback=True)
            with SessionLocal() as db:
                target_document = db.scalar(
                    select(DocumentRow).where(
                        DocumentRow.project_id == second_project["id"],
                        DocumentRow.active.is_(True),
                    )
                )
                db.add(
                    _issue(
                        second_target["id"], target_document.id, "降级中新问题"
                    )
                )
                db.commit()
            degraded = client.get(
                f"/api/v1/analysis-runs/{second_target['id']}/comparison"
            ).json()
            self.assertEqual(0, degraded["summary"]["no_longer_detected"])
            self.assertEqual(0, degraded["summary"]["new"])
            self.assertEqual(2, degraded["summary"]["unverifiable"])
            target_only = next(
                row
                for row in degraded["items"]
                if row["target_issue"] is not None
                and row["baseline_issue"] is None
            )
            self.assertEqual("unverifiable", target_only["outcome"])
            self.assertEqual(
                "runs_not_comparable", target_only["provenance"]["reason"]
            )
            self.assertIn(
                "target_model_partial_fallback",
                degraded["provenance"]["compatibility"]["reasons"],
            )
            with SessionLocal() as db:
                db.get(AnalysisDiagnosticRow, second_baseline["id"]).payload = (
                    _diagnostic(partial_fallback=True)
                )
                db.get(AnalysisDiagnosticRow, second_target["id"]).payload = (
                    _diagnostic(partial_fallback=False)
                )
                comparison = db.scalar(
                    select(AnalysisRunComparisonRow).where(
                        AnalysisRunComparisonRow.target_run_id
                        == second_target["id"]
                    )
                )
                db.execute(
                    delete(IssueComparisonItemRow).where(
                        IssueComparisonItemRow.comparison_id == comparison.id
                    )
                )
                comparison.status = "pending"
                comparison.summary = {}
                db.commit()
            baseline_degraded = client.get(
                f"/api/v1/analysis-runs/{second_target['id']}/comparison"
            ).json()
            self.assertIn(
                "baseline_model_partial_fallback",
                baseline_degraded["provenance"]["compatibility"]["reasons"],
            )

    def test_snapshot_valid_rejects_casefold_duplicate_document_names(self):
        first_content = "第一份"
        second_content = "第二份"
        rows = [
            # Distinct document ids and ordinals ensure the failure is solely
            # the case-insensitive logical-name collision.
            AnalysisRunInputRow(
                run_id="run",
                document_id="doc-a",
                document_name="Chapter.md",
                document_version=1,
                content=first_content,
                content_sha256=sha256(first_content.encode("utf-8")).hexdigest(),
                ordinal=0,
            ),
            AnalysisRunInputRow(
                run_id="run",
                document_id="doc-b",
                document_name="chapter.MD",
                document_version=1,
                content=second_content,
                content_sha256=sha256(second_content.encode("utf-8")).hexdigest(),
                ordinal=1,
            ),
        ]
        self.assertFalse(_snapshot_valid(rows))

    def test_completed_worker_materializes_comparison_in_same_transaction(self):
        original = {
            name: getattr(settings, name)
            for name in (
                "enable_model_extraction",
                "enable_evidence_investigator",
                "enable_issue_evidence_review",
                "enable_review_agent",
                "openai_api_key",
            )
        }
        for name in original:
            setattr(settings, name, False if name != "openai_api_key" else "")
        try:
            with TestClient(app) as client:
                content = (
                    '@fact subject="林澈" predicate="发色" value="银色" | 银发\n'
                    '@fact subject="林澈" predicate="发色" value="黑色" | 黑发'
                )
                project, document, baseline = self._project_and_baseline(
                    client, content
                )
                execute_analysis(baseline["id"])
                self.assertEqual(
                    "completed",
                    client.get(
                        f"/api/v1/analysis-runs/{baseline['id']}"
                    ).json()["status"],
                )
                target = self._new_revision_and_recheck(
                    client,
                    project,
                    document,
                    baseline,
                    '@fact subject="林澈" predicate="发色" value="银色" | 银发',
                )
                execute_analysis(target["id"])
                comparison = client.get(
                    f"/api/v1/analysis-runs/{target['id']}/comparison"
                ).json()
                self.assertEqual("ready", comparison["status"])
                self.assertEqual(1, comparison["summary"]["no_longer_detected"])
                with SessionLocal() as db:
                    durable = db.scalar(
                        select(AnalysisRunComparisonRow).where(
                            AnalysisRunComparisonRow.target_run_id == target["id"]
                        )
                    )
                    self.assertEqual("ready", durable.status)

                fallback_project, fallback_document, fallback_baseline = (
                    self._project_and_baseline(client, content)
                )
                execute_analysis(fallback_baseline["id"])
                fallback_target = self._new_revision_and_recheck(
                    client,
                    fallback_project,
                    fallback_document,
                    fallback_baseline,
                    '@fact subject="林澈" predicate="发色" value="银色" | 银发',
                )
                with patch(
                    "app.service.materialize_run_comparison",
                    side_effect=RuntimeError("private matcher detail"),
                ):
                    execute_analysis(fallback_target["id"])
                fallback = client.get(
                    f"/api/v1/analysis-runs/{fallback_target['id']}/comparison"
                ).json()
                self.assertEqual("ready", fallback["status"])
                self.assertEqual(1, fallback["summary"]["unverifiable"])
                self.assertEqual(
                    ["comparison_internal_error"],
                    fallback["provenance"]["compatibility"]["reasons"],
                )
                self.assertNotIn("private matcher detail", str(fallback))
        finally:
            for name, value in original.items():
                setattr(settings, name, value)

    def test_comparison_endpoints_hide_other_workspace(self):
        with TestClient(app) as client, SessionLocal() as db:
            workspace = WorkspaceRow(name=f"隔离-{uuid4().hex}", kind="personal")
            db.add(workspace)
            db.flush()
            project = ProjectRow(
                workspace_id=workspace.id, name="不可见复检", description=""
            )
            db.add(project)
            db.flush()
            document = DocumentRow(
                project_id=project.id, name="chapter.md", content="private"
            )
            db.add(document)
            db.flush()
            baseline = AnalysisRunRow(project_id=project.id, status="completed")
            target = AnalysisRunRow(project_id=project.id, status="queued")
            db.add_all([baseline, target])
            db.flush()
            capture_run_inputs(db, baseline, [document])
            capture_run_inputs(db, target, [document])
            comparison = AnalysisRunComparisonRow(
                project_id=project.id,
                baseline_run_id=baseline.id,
                target_run_id=target.id,
                matcher_version=MATCHER_VERSION,
                provenance={},
            )
            db.add(comparison)
            db.commit()
            baseline_id = baseline.id
            target_id = target.id

            self.assertEqual(
                404,
                client.post(
                    f"/api/v1/analysis-runs/{baseline_id}/rechecks"
                ).status_code,
            )
            self.assertEqual(
                404,
                client.get(
                    f"/api/v1/analysis-runs/{target_id}/comparison"
                ).status_code,
            )

    def test_document_version_constraints_prevent_duplicate_active_revision(self):
        with TestClient(app) as client:
            project, document, _ = self._project_and_baseline(client)
            with SessionLocal() as db:
                db.add(
                    DocumentRow(
                        project_id=project["id"],
                        name="CHAPTER.md",
                        content="并发重复版本",
                        version=1,
                        active=True,
                    )
                )
                with self.assertRaises(IntegrityError):
                    db.commit()
                db.rollback()

            revision = client.post(
                f"/api/v1/projects/{project['id']}/documents/text",
                json={
                    "name": "chapter.md",
                    "content": "合法第二版",
                    "replace_document_id": document["id"],
                },
            )
            self.assertEqual(201, revision.status_code, revision.text)
            with SessionLocal() as db:
                rows = list(
                    db.scalars(
                        select(DocumentRow).where(
                            DocumentRow.project_id == project["id"]
                        )
                    ).all()
                )
                self.assertEqual({1, 2}, {row.version for row in rows})
                self.assertEqual(1, sum(row.active for row in rows))


if __name__ == "__main__":
    unittest.main()
