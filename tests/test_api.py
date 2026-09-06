import os
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from sqlalchemy import select

# API flow tests must never inherit a developer's real model credential from .env.
os.environ["OPENAI_API_KEY"] = ""
os.environ["ENABLE_MODEL_EXTRACTION"] = "false"

from app.main import app, settings, write_limiter
from app.db import AnalysisDiagnosticRow, AnalysisRecordRow, AnalysisRunRow, DocumentContextRow, DocumentRow, IssueRow, ProjectRow, RunEventRow, SessionLocal, init_db
from app.service import capture_run_inputs, execute_analysis
from app.provider import ProviderError

# Some settings loaders treat an empty environment value as absent and fall
# back to .env. Disable the already-cached application setting as a second,
# explicit guard so ordinary API/demo tests can never make provider requests.
settings.enable_model_extraction = False
settings.openai_api_key = ""


class ApiFlowTests(unittest.TestCase):
    def test_cancelled_run_exposes_agent_token_lower_bound_qualifier(self):
        with TestClient(app) as client:
            with SessionLocal() as db:
                project = ProjectRow(name="取消用量 API")
                db.add(project)
                db.flush()
                run = AnalysisRunRow(
                    project_id=project.id,
                    status="cancelled",
                    prompt_tokens=11,
                    completion_tokens=7,
                )
                db.add(run)
                db.flush()
                db.add(
                    AnalysisDiagnosticRow(
                        run_id=run.id,
                        payload={
                            "usage_accounting": {
                                "completeness": "lower_bound",
                                "scope": "review_agent_completed_calls_only",
                                "terminal_status": "cancelled",
                                "logical_calls": 1,
                                "prompt_tokens": 11,
                                "completion_tokens": 7,
                                "charged_tokens": 23,
                                "charged_token_semantics": (
                                    "conservative_internal_budget_debit"
                                ),
                                "provider_calls": None,
                            }
                        },
                    )
                )
                db.commit()
                run_id = run.id

            response = client.get(f"/api/v1/analysis-runs/{run_id}")
            self.assertEqual(200, response.status_code)
            payload = response.json()
            self.assertEqual(
                18, payload["prompt_tokens"] + payload["completion_tokens"]
            )
            self.assertEqual(
                "lower_bound", payload["usage_accounting"]["completeness"]
            )
            self.assertEqual(
                "review_agent_completed_calls_only",
                payload["usage_accounting"]["scope"],
            )

    def test_legacy_diagnostics_do_not_invent_zero_empty_responses(self):
        with TestClient(app) as client:
            with SessionLocal() as db:
                project = ProjectRow(name="旧诊断")
                db.add(project)
                db.flush()
                run = AnalysisRunRow(project_id=project.id, status="completed")
                db.add(run)
                db.flush()
                db.add(AnalysisDiagnosticRow(
                    run_id=run.id,
                    payload={
                        "model": {
                            "enabled": True,
                            "configured": True,
                            "total_chunks": 1,
                            "attempted_chunks": 1,
                            "succeeded_chunks": 1,
                            "failed_chunks": 0,
                            "skipped_chunks": 0,
                            "invalid_records": 0,
                            "mode": "完整模型增强",
                        }
                    },
                ))
                db.commit()
                run_id = run.id

            response = client.get(f"/api/v1/analysis-runs/{run_id}/diagnostics")
            self.assertEqual(200, response.status_code)
            model = response.json()["model"]
            self.assertNotIn("empty_response_chunks", model)
            self.assertNotEqual(0, model.get("empty_response_chunks"))
            self.assertNotIn("provenance", response.json())

    def test_health_is_passive_and_provider_check_is_safe_and_actionable(self):
        with patch("app.main.OpenAICompatibleProvider") as provider_class:
            with TestClient(app) as client:
                health = client.get("/health")
        self.assertEqual(200, health.status_code)
        self.assertIn("configured", health.json()["model"])
        expected_thinking = {
            "configured": settings.provider_thinking_mode is not None,
            "mode": settings.provider_thinking_mode,
        }
        self.assertEqual(expected_thinking, health.json()["model"]["thinking"])
        provider_class.assert_not_called()

        provider = SimpleNamespace(configured=True)
        provider.complete = lambda *_: (_ for _ in ()).throw(
            ProviderError(
                "safe failure",
                category="forbidden",
                http_status=403,
                request_id="cf-safe-ray",
            )
        )
        with patch("app.main.OpenAICompatibleProvider", return_value=provider):
            with TestClient(app) as client:
                response = client.post("/api/v1/model/provider-check")
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("forbidden", payload["status"])
        self.assertEqual("forbidden", payload["category"])
        self.assertIsNone(payload["json_contract_ok"])
        self.assertEqual(403, payload["http_status"])
        self.assertTrue(payload["reachable"])
        self.assertFalse(payload["authorized"])
        self.assertEqual("cf-safe-ray", payload["request_id"])
        self.assertEqual(expected_thinking, payload["thinking"])
        self.assertEqual(0, payload["latency_ms"])
        self.assertEqual(
            {"prompt": None, "completion": None, "total": None},
            payload["token_usage"],
        )
        serialized = json.dumps(payload, ensure_ascii=False)
        for secret in (
            settings.openai_api_key,
            settings.openai_base_url,
            settings.openai_model,
        ):
            if secret:
                self.assertNotIn(secret, serialized)
        for topic in ("额度", "租户", "IP", "模型"):
            self.assertIn(topic, serialized)

    def test_provider_check_reports_unconfigured_without_calling_complete(self):
        provider = SimpleNamespace(configured=False)
        with patch("app.main.OpenAICompatibleProvider", return_value=provider):
            with TestClient(app) as client:
                response = client.post("/api/v1/model/provider-check")
        self.assertEqual(200, response.status_code)
        payload = response.json()
        self.assertEqual("not_configured", payload["category"])
        self.assertFalse(payload["configured"])
        self.assertIsNone(payload["json_contract_ok"])
        self.assertIsNone(payload["latency_ms"])
        self.assertEqual(
            {"prompt": None, "completion": None, "total": None},
            payload["token_usage"],
        )
        self.assertIsNone(payload["request_id"])

    def test_provider_check_distinguishes_unauthorized_from_upstream_outage(self):
        cases = (
            ("unauthorized", 401, False),
            ("upstream_5xx", 503, None),
        )
        for category, status, authorized in cases:
            with self.subTest(category=category):
                error = ProviderError(
                    "safe failure",
                    category=category,
                    http_status=status,
                    request_id="request-safe-api",
                )
                error.endpoint = "https://private.invalid/v1"
                error.headers = {"Authorization": "private-key"}
                provider = SimpleNamespace(
                    configured=True,
                    complete=lambda *_args, failure=error: (
                        _ for _ in ()
                    ).throw(failure),
                )
                with patch(
                    "app.main.OpenAICompatibleProvider", return_value=provider
                ):
                    with TestClient(app) as client:
                        response = client.post("/api/v1/model/provider-check")
                payload = response.json()
                self.assertEqual(200, response.status_code)
                self.assertEqual(category, payload["category"])
                self.assertTrue(payload["reachable"])
                self.assertEqual(authorized, payload["authorized"])
                self.assertEqual(status, payload["http_status"])
                serialized = json.dumps(payload)
                self.assertNotIn("private.invalid", serialized)
                self.assertNotIn("private-key", serialized)

    def test_provider_check_success_response_uses_an_explicit_allowlist(self):
        telemetry = SimpleNamespace(
            http_status=200,
            request_id="bad id with spaces",
            elapsed_ms=321,
            prompt_tokens=9,
            completion_tokens=4,
            endpoint="https://private.invalid/v1",
            headers={"Authorization": "private-key"},
            response_body="private-response",
        )
        result = SimpleNamespace(text='{"status":"ok"}', telemetry=telemetry)
        provider = SimpleNamespace(configured=True, complete=lambda *_: result)
        with patch("app.main.OpenAICompatibleProvider", return_value=provider):
            with TestClient(app) as client:
                response = client.post("/api/v1/model/provider-check")
        payload = response.json()
        self.assertEqual("success", payload["status"])
        self.assertTrue(payload["json_contract_ok"])
        self.assertEqual(321, payload["latency_ms"])
        self.assertEqual(
            {"prompt": 9, "completion": 4, "total": 13},
            payload["token_usage"],
        )
        self.assertTrue(payload["authorized"])
        self.assertIsNone(payload["request_id"])
        serialized = json.dumps(payload)
        for secret in (
            "private.invalid",
            "Authorization",
            "private-key",
            "private-response",
            "bad id with spaces",
        ):
            self.assertNotIn(secret, serialized)

    def test_provider_check_marks_invalid_json_contract_without_raw_content(self):
        telemetry = SimpleNamespace(
            http_status=200,
            request_id=None,
            elapsed_ms=87,
            prompt_tokens=6,
            completion_tokens=3,
        )
        result = SimpleNamespace(
            text='{"status":"not-the-contract","private":"do-not-return"}',
            telemetry=telemetry,
        )
        provider = SimpleNamespace(configured=True, complete=lambda *_: result)
        with patch("app.main.OpenAICompatibleProvider", return_value=provider):
            with TestClient(app) as client:
                response = client.post("/api/v1/model/provider-check")

        payload = response.json()
        self.assertEqual("invalid_response", payload["category"])
        self.assertFalse(payload["json_contract_ok"])
        self.assertTrue(payload["reachable"])
        self.assertTrue(payload["authorized"])
        self.assertEqual(87, payload["latency_ms"])
        self.assertEqual(9, payload["token_usage"]["total"])
        self.assertNotIn("do-not-return", json.dumps(payload))
        self.assertIn("JSON", " ".join(payload["suggestions"]))

    def test_provider_check_reports_only_allowlisted_thinking_configuration(self):
        result = SimpleNamespace(text='{"status":"ok"}', telemetry=None)
        provider = SimpleNamespace(configured=True, complete=lambda *_: result)
        original_mode = settings.provider_thinking_mode
        try:
            settings.provider_thinking_mode = "enabled"
            with patch("app.main.OpenAICompatibleProvider", return_value=provider):
                with TestClient(app) as client:
                    response = client.post("/api/v1/model/provider-check")
        finally:
            settings.provider_thinking_mode = original_mode

        self.assertEqual(
            {"configured": True, "mode": "enabled"},
            response.json()["thinking"],
        )
        serialized = json.dumps(response.json())
        for secret in (settings.openai_api_key, settings.openai_base_url):
            if secret:
                self.assertNotIn(secret, serialized)

    def test_document_context_json_and_multipart_inherit_or_override_explicitly(self):
        write_limiter.events.clear()
        with patch("app.main.dispatch_analysis"):
            with TestClient(app) as client:
                project = client.post(
                    "/api/v1/projects", json={"name": "文档上下文"}
                ).json()
                first = client.post(
                    f"/api/v1/projects/{project['id']}/documents/text",
                    json={
                        "name": "story.md",
                        "content": "第一版正文。",
                        "document_role": "canon",
                        "story_scope": "route_a",
                    },
                )
                self.assertEqual(201, first.status_code)
                self.assertEqual("canon", first.json()["document_role"])
                self.assertEqual("route_a", first.json()["story_scope"])

                inherited = client.post(
                    f"/api/v1/projects/{project['id']}/documents/text",
                    json={"name": "story.md", "content": "第二版正文。"},
                )
                self.assertEqual("canon", inherited.json()["document_role"])
                self.assertEqual("route_a", inherited.json()["story_scope"])

                overridden = client.post(
                    f"/api/v1/projects/{project['id']}/documents/text",
                    json={
                        "name": "story.md",
                        "content": "第三版正文。",
                        "document_role": "chapter",
                        "story_scope": "route_b",
                    },
                )
                self.assertEqual("chapter", overridden.json()["document_role"])
                self.assertEqual("route_b", overridden.json()["story_scope"])

                upload = client.post(
                    f"/api/v1/projects/{project['id']}/documents",
                    files={"file": ("profile.txt", "角色资料", "text/plain")},
                    data={
                        "document_role": "character_profile",
                        "story_scope": "shared_cast",
                    },
                )
                self.assertEqual(201, upload.status_code)
                self.assertEqual("character_profile", upload.json()["document_role"])
                self.assertEqual("shared_cast", upload.json()["story_scope"])
                upload_v2 = client.post(
                    f"/api/v1/projects/{project['id']}/documents",
                    files={"file": ("profile.txt", "角色资料第二版", "text/plain")},
                )
                self.assertEqual("character_profile", upload_v2.json()["document_role"])
                self.assertEqual("shared_cast", upload_v2.json()["story_scope"])

                invalid_role = client.post(
                    f"/api/v1/projects/{project['id']}/documents/text",
                    json={
                        "name": "invalid.md",
                        "content": "测试",
                        "document_role": "arbitrary",
                    },
                )
                self.assertEqual(422, invalid_role.status_code)
                invalid_scope = client.post(
                    f"/api/v1/projects/{project['id']}/documents/text",
                    json={
                        "name": "invalid.md",
                        "content": "测试",
                        "story_scope": "bad/scope",
                    },
                )
                self.assertEqual(422, invalid_scope.status_code)

                run = client.post(
                    f"/api/v1/projects/{project['id']}/analysis-runs"
                ).json()
                status = client.get(
                    f"/api/v1/analysis-runs/{run['id']}"
                ).json()
                contexts = {
                    row["document_name"]: (
                        row["document_role"], row["story_scope"]
                    )
                    for row in status["input_documents"]
                }
                self.assertEqual(("chapter", "route_b"), contexts["story.md"])
                self.assertEqual(
                    ("character_profile", "shared_cast"), contexts["profile.txt"]
                )

    def test_legacy_document_gets_explicitly_reported_safe_default_context(self):
        write_limiter.events.clear()
        with TestClient(app) as client:
            project = client.post(
                "/api/v1/projects", json={"name": "旧文档上下文"}
            ).json()
            with SessionLocal() as db:
                document = DocumentRow(
                    project_id=project["id"],
                    name="legacy.txt",
                    content="只有正文也可以创建分析。",
                )
                db.add(document)
                db.commit()
                document_id = document.id
            details = client.get(f"/api/v1/projects/{project['id']}").json()
            stored = next(row for row in details["documents"] if row["id"] == document_id)
            self.assertEqual("chapter", stored["document_role"])
            self.assertEqual("global", stored["story_scope"])
            self.assertFalse(stored["context_explicit"])

            with patch("app.main.dispatch_analysis"):
                run = client.post(
                    f"/api/v1/projects/{project['id']}/analysis-runs"
                ).json()
            status = client.get(f"/api/v1/analysis-runs/{run['id']}").json()
            self.assertEqual("chapter", status["input_documents"][0]["document_role"])
            self.assertEqual("global", status["input_documents"][0]["story_scope"])

    def test_single_plain_body_document_can_complete_without_world_document(self):
        write_limiter.events.clear()
        with patch("app.main.dispatch_analysis"):
            with TestClient(app) as client:
                project = client.post(
                    "/api/v1/projects", json={"name": "单正文分析"}
                ).json()
                document = client.post(
                    f"/api/v1/projects/{project['id']}/documents/text",
                    json={
                        "name": "正文.txt",
                        "content": "林澈在雨夜抵达北港。他不确定守门人是否认识自己。",
                    },
                )
                self.assertEqual(201, document.status_code)
                self.assertEqual("chapter", document.json()["document_role"])
                self.assertEqual("global", document.json()["story_scope"])
                run = client.post(
                    f"/api/v1/projects/{project['id']}/analysis-runs"
                ).json()

                execute_analysis(run["id"])

                status = client.get(
                    f"/api/v1/analysis-runs/{run['id']}"
                ).json()
                self.assertEqual("completed", status["status"])
                self.assertEqual(1, len(status["input_documents"]))
                self.assertEqual(
                    "chapter", status["input_documents"][0]["document_role"]
                )
                self.assertEqual(
                    "global", status["input_documents"][0]["story_scope"]
                )

    def test_openapi_exposes_closed_document_context_and_typed_clarifications(self):
        schema = app.openapi()
        components = schema["components"]["schemas"]
        text_input = components["TextDocumentIn"]
        self.assertFalse(text_input["additionalProperties"])
        self.assertEqual(
            {"canon", "character_profile", "chapter", "reference"},
            set(components["DocumentRole"]["enum"]),
        )
        self.assertIn("document_role", text_input["properties"])
        self.assertEqual(
            r"^[A-Za-z0-9_\-\u4e00-\u9fff]+$",
            text_input["properties"]["story_scope"]["anyOf"][0]["pattern"],
        )
        upload_input = components[
            "Body_upload_document_api_v1_projects__project_id__documents_post"
        ]
        self.assertIn("document_role", upload_input["properties"])
        self.assertIn("story_scope", upload_input["properties"])
        response = schema["paths"][
            "/api/v1/analysis-runs/{run_id}/clarifications"
        ]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        self.assertEqual(
            "#/components/schemas/SemanticReviewItemOut",
            response["items"]["$ref"],
        )

    def test_clarifications_are_typed_sanitized_and_separate_from_issues(self):
        with TestClient(app) as client:
            project = client.post(
                "/api/v1/projects", json={"name": "待澄清只读结果"}
            ).json()
            with SessionLocal() as db:
                run = AnalysisRunRow(project_id=project["id"], status="completed")
                db.add(run)
                db.flush()
                evidence = {
                    "document_id": "doc",
                    "document_name": "chapter.md",
                    "line_start": 7,
                    "line_end": 7,
                    "text": "他是否通过暗渠抵达，需要作者确认。",
                }
                db.add_all(
                    [
                        AnalysisRecordRow(
                            run_id=run.id,
                            kind="open_question",
                            attrs={
                                "question": "他是否通过暗渠抵达？",
                                "question_type": "open",
                                "modality": "interrogative",
                                "source_scope": "narrator",
                                "certainty": "unknown",
                                "internal_debug": "不得泄漏",
                            },
                            evidence=evidence,
                        ),
                        AnalysisRecordRow(
                            run_id=run.id,
                            kind="clarification",
                            attrs={
                                "summary": "抵达路径需要作者确认",
                                "category": "missing_causal_bridge",
                                "modality": "uncertain",
                                "source_scope": "narrator",
                                "certainty": "unknown",
                                "internal_debug": "不得泄漏",
                            },
                            evidence=evidence,
                        ),
                    ]
                )
                db.commit()
                run_id = run.id

            response = client.get(
                f"/api/v1/analysis-runs/{run_id}/clarifications"
            )
            self.assertEqual(200, response.status_code)
            rows = response.json()
            self.assertEqual({"clarification", "open_question"}, {row["kind"] for row in rows})
            self.assertTrue(all("attrs" not in row for row in rows))
            self.assertNotIn("internal_debug", json.dumps(rows, ensure_ascii=False))
            self.assertEqual([], client.get(f"/api/v1/analysis-runs/{run_id}/issues").json())
            status = client.get(f"/api/v1/analysis-runs/{run_id}").json()
            self.assertEqual(1, status["clarification_count"])
            self.assertEqual(1, status["open_question_count"])

    def test_run_creation_and_retry_expose_frozen_document_versions(self):
        write_limiter.events.clear()
        with patch("app.main.dispatch_analysis"):
            with TestClient(app) as client:
                project = client.post(
                    "/api/v1/projects", json={"name": "运行输入冻结"}
                ).json()
                first = client.post(
                    f"/api/v1/projects/{project['id']}/documents/text",
                    json={"name": "chapter.md", "content": "第一版"},
                ).json()
                first_run = client.post(
                    f"/api/v1/projects/{project['id']}/analysis-runs"
                ).json()
                first_status = client.get(
                    f"/api/v1/analysis-runs/{first_run['id']}"
                ).json()
                self.assertTrue(first_status["input_snapshot_available"])
                self.assertEqual(
                    1, first_status["input_documents"][0]["document_version"]
                )
                self.assertEqual(64, len(first_status["input_documents"][0]["content_sha256"]))

                client.post(
                    f"/api/v1/projects/{project['id']}/documents/text",
                    json={"name": "chapter.md", "content": "第二版"},
                )
                with SessionLocal() as db:
                    run = db.get(AnalysisRunRow, first_run["id"])
                    run.status = "failed"
                    db.commit()
                retry = client.post(
                    f"/api/v1/analysis-runs/{first_run['id']}/retry"
                ).json()
                retry_status = client.get(
                    f"/api/v1/analysis-runs/{retry['id']}"
                ).json()
                self.assertEqual(first_run["id"], retry_status["retried_from"])
                self.assertEqual(
                    first["id"], retry_status["input_documents"][0]["document_id"]
                )
                self.assertEqual(
                    1, retry_status["input_documents"][0]["document_version"]
                )

                current_run = client.post(
                    f"/api/v1/projects/{project['id']}/analysis-runs"
                ).json()
                current_status = client.get(
                    f"/api/v1/analysis-runs/{current_run['id']}"
                ).json()
                self.assertEqual(
                    2, current_status["input_documents"][0]["document_version"]
                )

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
                document = DocumentRow(project_id=project.id, name="chapter.md", content="测试")
                db.add(document)
                completed = AnalysisRunRow(
                    project_id=project.id,
                    status="completed",
                    prompt_tokens=3,
                    completion_tokens=2,
                    estimated_cost_usd=9.99,
                )
                failed = AnalysisRunRow(project_id=project.id, status="failed")
                db.add_all([completed, failed]); db.flush()
                capture_run_inputs(db, failed, [document])
                db.commit()
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
                document = DocumentRow(project_id=project["id"], name="chapter.md", content="重试输入")
                queued = AnalysisRunRow(project_id=project["id"], status="queued")
                completed = AnalysisRunRow(project_id=project["id"], status="completed")
                failed = AnalysisRunRow(project_id=project["id"], status="failed", error="模拟失败")
                db.add_all([document, queued, completed, failed]); db.flush()
                capture_run_inputs(db, failed, [document])
                failed_event = RunEventRow(run_id=failed.id, stage="failed", progress=100, message="失败")
                db.add(failed_event); db.commit()
                queued_id, completed_id, failed_id, event_id = queued.id, completed.id, failed.id, failed_event.id

            first_cancel = client.post(f"/api/v1/analysis-runs/{queued_id}/cancel").json()
            second_cancel = client.post(f"/api/v1/analysis-runs/{queued_id}/cancel").json()
            self.assertFalse(first_cancel["already_requested"])
            self.assertTrue(second_cancel["already_requested"])
            self.assertEqual(
                "cancelled",
                client.get(f"/api/v1/analysis-runs/{queued_id}").json()["status"],
            )
            with SessionLocal() as db:
                cancel_events = db.scalars(
                    select(RunEventRow).where(
                        RunEventRow.run_id == queued_id,
                        RunEventRow.stage == "cancelled",
                    )
                ).all()
                self.assertEqual(1, len(cancel_events))
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
            self.assertFalse(payload["model"]["configured"])
            self.assertEqual([], payload["model"]["provider_calls"])
            self.assertEqual(1, payload["provenance"]["schema_version"])
            self.assertTrue(payload["provenance"]["directives"])
            self.assertTrue(all(
                row["sources"] == ["baseline"]
                for row in payload["provenance"]["directives"]
            ))
            self.assertTrue(all(
                row["derivation_type"] == "deterministic_rule"
                for row in payload["provenance"]["issues"]
            ))
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
