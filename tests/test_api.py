import os
import json
import time
import unittest

from fastapi.testclient import TestClient

# API flow tests must never inherit a developer's real model credential from .env.
os.environ["OPENAI_API_KEY"] = ""

from app.main import app, settings, write_limiter
from app.db import AnalysisRecordRow, AnalysisRunRow, DocumentRow, IssueRow, ProjectRow, RunEventRow, SessionLocal, init_db


class ApiFlowTests(unittest.TestCase):
    def test_completed_run_graph_timeline_and_record_order(self):
        init_db()
        with SessionLocal() as db:
            project = ProjectRow(name="可视化接口")
            db.add(project); db.flush()
            completed = AnalysisRunRow(project_id=project.id, status="completed")
            queued = AnalysisRunRow(project_id=project.id, status="queued")
            db.add_all([completed, queued]); db.flush()
            later_evidence = {"document_id": "doc", "document_name": "chapter.md", "line_start": 9, "line_end": 9, "text": "林澈在南塔。"}
            early_evidence = {"document_id": "doc", "document_name": "chapter.md", "line_start": 2, "line_end": 2, "text": "林澈在北港。"}
            db.add_all([
                AnalysisRecordRow(run_id=completed.id, kind="event", attrs={"participants": "林澈", "location": "南塔", "time": "1026-04-03 10:00"}, evidence=later_evidence),
                AnalysisRecordRow(run_id=completed.id, kind="event", attrs={"participants": "林澈", "location": "北港", "time": "1026-04-03 08:00"}, evidence=early_evidence),
                IssueRow(run_id=completed.id, category="location_collision", severity="high", confidence=0.9, title="地点冲突", explanation="测试", evidence=[early_evidence, later_evidence], suggestion="调整时间", extra={}),
            ])
            db.commit()
            completed_id, queued_id = completed.id, queued.id

        with TestClient(app) as client:
            records = client.get(f"/api/v1/analysis-runs/{completed_id}/records").json()["records"]
            self.assertEqual([2, 9], [row["evidence"]["line_start"] for row in records])
            self.assertTrue(all(row["id"] for row in records))
            graph = client.get(f"/api/v1/analysis-runs/{completed_id}/graph")
            self.assertEqual(200, graph.status_code)
            self.assertEqual(2, len(graph.json()["edges"]))
            self.assertTrue(all(edge["issue_ids"] for edge in graph.json()["edges"]))
            timeline = client.get(f"/api/v1/analysis-runs/{completed_id}/timeline")
            self.assertEqual(200, timeline.status_code)
            self.assertEqual(["1026-04-03 08:00", "1026-04-03 10:00"], [group["timestamp"] for group in timeline.json()["groups"]])
            self.assertEqual(409, client.get(f"/api/v1/analysis-runs/{queued_id}/graph").status_code)
            self.assertEqual(409, client.get(f"/api/v1/analysis-runs/{queued_id}/timeline").status_code)
            self.assertEqual(404, client.get("/api/v1/analysis-runs/missing/graph").status_code)

    def test_daily_model_budget_and_unconfigured_cost(self):
        original = (
            settings.enable_model_extraction,
            settings.openai_api_key,
            settings.daily_token_budget,
            settings.model_input_price_per_million,
            settings.model_output_price_per_million,
        )
        try:
            init_db()
            settings.enable_model_extraction = True
            settings.openai_api_key = "unit-test-placeholder"
            settings.daily_token_budget = 0
            settings.model_input_price_per_million = None
            settings.model_output_price_per_million = None
            with SessionLocal() as db:
                project = ProjectRow(name="预算测试")
                db.add(project); db.flush()
                db.add(DocumentRow(project_id=project.id, name="chapter.md", content="测试"))
                completed = AnalysisRunRow(
                    project_id=project.id,
                    status="completed",
                    prompt_tokens=3,
                    completion_tokens=2,
                    estimated_cost_usd=9.99,
                )
                failed = AnalysisRunRow(project_id=project.id, status="failed")
                db.add_all([completed, failed]); db.commit()
                project_id, completed_id, failed_id = project.id, completed.id, failed.id

            write_limiter.events.clear()
            with TestClient(app) as client:
                run = client.get(f"/api/v1/analysis-runs/{completed_id}")
                self.assertEqual(200, run.status_code)
                self.assertIsNone(run.json()["estimated_cost_usd"])
                limited = client.post(f"/api/v1/projects/{project_id}/analysis-runs")
                self.assertEqual(429, limited.status_code)
                self.assertIn("Retry-After", limited.headers)
                retry_limited = client.post(f"/api/v1/analysis-runs/{failed_id}/retry")
                self.assertEqual(429, retry_limited.status_code)
        finally:
            (
                settings.enable_model_extraction,
                settings.openai_api_key,
                settings.daily_token_budget,
                settings.model_input_price_per_million,
                settings.model_output_price_per_million,
            ) = original
            write_limiter.events.clear()

    def test_project_listing_and_same_name_document_versions(self):
        with TestClient(app) as client:
            project = client.post("/api/v1/projects", json={"name": "版本工作流"}).json()
            first = client.post(
                f"/api/v1/projects/{project['id']}/documents/text",
                json={"name": "chapter.md", "content": "第一版内容"},
            ).json()
            second = client.post(
                f"/api/v1/projects/{project['id']}/documents/text",
                json={"name": "chapter.md", "content": "第二版内容"},
            ).json()
            self.assertEqual(1, first["version"])
            self.assertEqual(2, second["version"])
            self.assertIn(first["id"], second["superseded_document_ids"])
            history = client.get(
                f"/api/v1/projects/{project['id']}/documents?include_history=true"
            ).json()
            self.assertEqual(2, len(history))
            self.assertEqual(1, len([row for row in history if row["active"]]))
            self.assertEqual(2, next(row for row in history if row["active"])["version"])
            self.assertTrue(all("content" not in row for row in history))
            conflict = client.post(
                f"/api/v1/projects/{project['id']}/documents/text",
                json={"name": "renamed.md", "content": "错误替换", "replace_document_id": second["id"]},
            )
            self.assertEqual(409, conflict.status_code)
            projects = client.get("/api/v1/projects").json()
            summary = next(row for row in projects if row["id"] == project["id"])
            self.assertEqual(1, summary["active_document_count"])

    def test_document_version_diff_boundaries_and_summary(self):
        with TestClient(app) as client:
            project = client.post("/api/v1/projects", json={"name": "版本差异"}).json()
            other_project = client.post("/api/v1/projects", json={"name": "其他项目"}).json()
            first = client.post(
                f"/api/v1/projects/{project['id']}/documents/text",
                json={"name": "chapter.md", "content": "第一行\n旧内容\n保留行"},
            ).json()
            second = client.post(
                f"/api/v1/projects/{project['id']}/documents/text",
                json={"name": "chapter.md", "content": "第一行\n新内容\n保留行\n新增行"},
            ).json()
            different_name = client.post(
                f"/api/v1/projects/{project['id']}/documents/text",
                json={"name": "world.md", "content": "设定"},
            ).json()
            other = client.post(
                f"/api/v1/projects/{other_project['id']}/documents/text",
                json={"name": "chapter.md", "content": "其他项目内容"},
            ).json()

            response = client.get(
                f"/api/v1/projects/{project['id']}/documents/diff",
                params={"from_document_id": first["id"], "to_document_id": second["id"]},
            )
            self.assertEqual(200, response.status_code)
            payload = response.json()
            self.assertEqual((1, 2), (payload["from_document"]["version"], payload["to_document"]["version"]))
            self.assertNotIn("content", payload["from_document"])
            self.assertEqual(2, payload["summary"]["added_lines"])
            self.assertEqual(1, payload["summary"]["removed_lines"])
            self.assertEqual([], payload["warnings"])

            same = client.get(
                f"/api/v1/projects/{project['id']}/documents/diff",
                params={"from_document_id": first["id"], "to_document_id": first["id"]},
            )
            self.assertEqual(409, same.status_code)
            wrong_name = client.get(
                f"/api/v1/projects/{project['id']}/documents/diff",
                params={"from_document_id": first["id"], "to_document_id": different_name["id"]},
            )
            self.assertEqual(409, wrong_name.status_code)
            cross_project = client.get(
                f"/api/v1/projects/{project['id']}/documents/diff",
                params={"from_document_id": first["id"], "to_document_id": other["id"]},
            )
            self.assertEqual(404, cross_project.status_code)
            self.assertEqual(
                404,
                client.get(
                    "/api/v1/projects/missing/documents/diff",
                    params={"from_document_id": first["id"], "to_document_id": second["id"]},
                ).status_code,
            )

    def test_run_history_cancel_retry_constraints_and_failed_sse_resume(self):
        with TestClient(app) as client:
            project = client.post("/api/v1/projects", json={"name": "运行历史"}).json()
            with SessionLocal() as db:
                queued = AnalysisRunRow(project_id=project["id"], status="queued")
                completed = AnalysisRunRow(project_id=project["id"], status="completed")
                failed = AnalysisRunRow(project_id=project["id"], status="failed", error="模拟失败")
                db.add_all([queued, completed, failed]); db.flush()
                failed_event = RunEventRow(run_id=failed.id, stage="failed", progress=100, message="失败")
                db.add(failed_event); db.commit()
                queued_id, completed_id, failed_id, event_id = queued.id, completed.id, failed.id, failed_event.id

            first_cancel = client.post(f"/api/v1/analysis-runs/{queued_id}/cancel").json()
            second_cancel = client.post(f"/api/v1/analysis-runs/{queued_id}/cancel").json()
            self.assertFalse(first_cancel["already_requested"])
            self.assertTrue(second_cancel["already_requested"])
            self.assertEqual(409, client.post(f"/api/v1/analysis-runs/{completed_id}/cancel").status_code)
            self.assertEqual(409, client.post(f"/api/v1/analysis-runs/{completed_id}/retry").status_code)
            retry = client.post(f"/api/v1/analysis-runs/{failed_id}/retry")
            self.assertEqual(202, retry.status_code)
            history = client.get(f"/api/v1/projects/{project['id']}/analysis-runs").json()
            self.assertGreaterEqual(len(history), 4)

            stream = client.get(
                f"/api/v1/analysis-runs/{failed_id}/events",
                headers={"Last-Event-ID": str(event_id)},
            ).text
            self.assertNotIn("event: progress", stream)
            self.assertIn("event: terminal", stream)
            self.assertIn('"status": "failed"', stream)
            self.assertIn("模拟失败", stream)

    def test_one_click_natural_text_demo(self):
        with TestClient(app) as client:
            project = client.post("/api/v1/demo").json()
            self.assertEqual(2, project["document_count"])
            run = client.post(f"/api/v1/projects/{project['id']}/analysis-runs").json()
            status = "queued"
            for _ in range(100):
                status = client.get(f"/api/v1/analysis-runs/{run['id']}").json()["status"]
                if status in {"completed", "failed"}:
                    break
                time.sleep(0.02)
            self.assertEqual("completed", status)
            issues = client.get(f"/api/v1/analysis-runs/{run['id']}/issues").json()
            records = client.get(f"/api/v1/analysis-runs/{run['id']}/records").json()
            self.assertEqual(5, len({issue["category"] for issue in issues}))
            self.assertGreaterEqual(records["record_count"], 10)
            diagnostics = client.get(f"/api/v1/analysis-runs/{run['id']}/diagnostics")
            self.assertEqual(200, diagnostics.status_code)
            payload = diagnostics.json()
            self.assertGreaterEqual(payload["chunking"]["total_chunks"], 2)
            self.assertIn("candidate_count", payload["retrieval"])
            self.assertNotIn("prompt", json.dumps(payload, ensure_ascii=False).lower())

    def test_end_to_end_analysis_and_feedback(self):
        with TestClient(app) as client:
            project = client.post("/api/v1/projects", json={"name": "API smoke"}).json()
            source = b'@fact subject="A" predicate="color" value="white" | first\n@fact subject="A" predicate="color" value="black" | second'
            upload = client.post(
                f"/api/v1/projects/{project['id']}/documents",
                files={"file": ("chapter.md", source, "text/markdown")},
            )
            self.assertEqual(201, upload.status_code)
            run = client.post(f"/api/v1/projects/{project['id']}/analysis-runs").json()
            status = "queued"
            for _ in range(100):
                status = client.get(f"/api/v1/analysis-runs/{run['id']}").json()["status"]
                if status in {"completed", "failed"}:
                    break
                time.sleep(0.02)
            self.assertEqual("completed", status)
            issues = client.get(f"/api/v1/analysis-runs/{run['id']}/issues").json()
            self.assertEqual("fact_conflict", issues[0]["category"])
            response = client.post(f"/api/v1/issues/{issues[0]['id']}/feedback", json={"label": "accepted"})
            self.assertEqual(201, response.status_code)
            duplicate = client.post(f"/api/v1/issues/{issues[0]['id']}/feedback", json={"label": "accepted"})
            self.assertEqual(200, duplicate.status_code)
            self.assertTrue(duplicate.json()["duplicate_ignored"])
            changed = client.post(
                f"/api/v1/issues/{issues[0]['id']}/feedback",
                json={"label": "resolved", "comment": "已统一设定"},
            )
            self.assertEqual(201, changed.status_code)
            audit = client.get(f"/api/v1/issues/{issues[0]['id']}/feedback").json()
            self.assertEqual("resolved", audit["latest"]["label"])
            self.assertEqual(2, len(audit["history"]))


if __name__ == "__main__":
    unittest.main()
