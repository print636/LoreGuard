"""Normal author authorization, old-run semantics and source-only reading."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import NARRATIVE_TIME_KNOWLEDGE_POLICY_V1, Settings
from app.db import AnalysisRunRow, AnalysisRunInputRow, IssueRow, SessionLocal
from app.main import app
from app.projections import project_graph, project_timeline
from app.provider import OpenAICompatibleProvider
from app.service import _narrative_settings_for_run, execute_analysis
from tests.test_account_provider import provider_test_settings, register, save_provider


@pytest.mark.parametrize("frozen", [None, "unknown-version", NARRATIVE_TIME_KNOWLEDGE_POLICY_V1])
def test_worker_uses_only_frozen_policy_and_preserves_legacy_default(frozen):
    base = Settings(_env_file=None, _narrative_time_knowledge_policy="client-cannot-enable")
    assert base._narrative_time_knowledge_policy is None
    base._narrative_time_knowledge_policy = NARRATIVE_TIME_KNOWLEDGE_POLICY_V1
    run = AnalysisRunRow(batch_coverage={"narrative_time_knowledge_policy": frozen})
    local = _narrative_settings_for_run(base, run)
    assert local._narrative_time_knowledge_policy == (frozen if frozen == NARRATIVE_TIME_KNOWLEDGE_POLICY_V1 else None)
    assert base._narrative_time_knowledge_policy == NARRATIVE_TIME_KNOWLEDGE_POLICY_V1
    assert "_narrative_time_knowledge_policy" not in local.model_dump()


def test_normal_account_freezes_policy_retries_keep_history_and_extra_evidence_is_readable():
    from app.model_extractor import SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT, SYSTEM_PROMPT_NARRATIVE_V1, BATCH_SYSTEM_PROMPT_NARRATIVE_V1

    sent = []
    def respond(request):
        body = json.loads(request.content)
        system, user = (message["content"] for message in body["messages"])
        sent.append(system)
        assert system in {SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT, SYSTEM_PROMPT_NARRATIVE_V1, BATCH_SYSTEM_PROMPT_NARRATIVE_V1}
        record = {"kind": "knowledge_constraint", "character": "林安", "fact": "换班口令", "time": "14:00",
                  "relation": "unknown_before", "knowledge_scope": "whole_content", "basis": "explicit",
                  "relation_quote": "14:00之前，林安明确不知道换班口令。", "source_line_start": 4, "source_line_end": 4,
                  "modality": "asserted", "source_scope": "narrator", "certainty": "certain"}
        rows = [record] if system in {SYSTEM_PROMPT_NARRATIVE_V1, BATCH_SYSTEM_PROMPT_NARRATIVE_V1} else []
        if system in {BATCH_SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT_NARRATIVE_V1}:
            documents = json.loads(user.split("\n", 1)[1])["documents"]
            content = {"documents": [{"doc_ref": row["doc_ref"], "records": rows} for row in documents]}
        else:
            content = {"records": rows}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(content)}}],
                                         "usage": {"prompt_tokens": 12, "completion_tokens": 8}})

    original_init = OpenAICompatibleProvider.__init__
    def provider_init(self, config=None, **kwargs):
        kwargs["transport"] = httpx.MockTransport(respond)
        original_init(self, config, **kwargs)

    with provider_test_settings(), TestClient(app) as client:
        _, csrf = register(client)
        client.headers["X-CSRF-Token"] = csrf
        assert save_provider(client, csrf, expected_revision=0, api_key="synthetic-only", model="narrative-test").status_code == 200
        project = client.post("/api/v1/projects", json={"name": "Narrative evidence"}).json()
        documents_url = f"/api/v1/projects/{project['id']}/documents/text"
        content = "12:00，林岑在西塔值守。\n12:00，林岑在东港签字。\n这两件事发生在同一天。\n14:00之前，林安明确不知道换班口令。"
        document = client.post(documents_url, json={"name": "draft.md", "content": content, "document_role": "chapter"}).json()
        runs_url = f"/api/v1/projects/{project['id']}/analysis-runs"
        with patch("app.main.dispatch_analysis"):
            created = client.post(runs_url, json={})
            assert created.status_code == 202, created.text
            run_id = created.json()["id"]
            for override in ({"_narrative_time_knowledge_policy": "other"}, {"narrative_time_knowledge_policy": "other"},
                             {"batch_coverage": {"narrative_time_knowledge_policy": "other"}}):
                assert client.post(runs_url, json=override).status_code == 422
        with SessionLocal() as db:
            frozen = db.get(AnalysisRunRow, run_id)
            assert frozen.batch_coverage["narrative_time_knowledge_policy"] == NARRATIVE_TIME_KNOWLEDGE_POLICY_V1
            assert not frozen.batch_coverage.get("character_review_enabled")
        with patch.object(OpenAICompatibleProvider, "__init__", provider_init):
            execute_analysis(run_id, raise_on_failure=True)
        assert sent
        assert client.get(f"/api/v1/analysis-runs/{run_id}").json()["status"] == "completed"
        records = client.get(f"/api/v1/analysis-runs/{run_id}/records").json()["records"]
        constraint, = [row for row in records if row["kind"] == "knowledge_constraint"]
        assert constraint["attrs"]["relation"] == "unknown_before"
        assert constraint["evidence"]["text"] == "14:00之前，林安明确不知道换班口令。"
        assert "relation_quote" not in constraint["attrs"] and "doc_ref" not in constraint["attrs"]
        assert not any(row["kind"] == "knows" for row in records)

        # Reading a third, separately cited time-identity source uses the same
        # authorized frozen reader. This fixture does not assert rule semantics.
        with SessionLocal() as db:
            source = db.scalar(select(AnalysisRunInputRow).where(AnalysisRunInputRow.run_id == run_id))
            spans = [{"document_id": source.document_id, "document_name": source.document_name,
                      "line_start": index, "line_end": index, "text": text}
                     for index, text in enumerate(source.content.splitlines(), 1)]
            issue = IssueRow(run_id=run_id, report_class="formal", category="location_collision", severity="high", confidence=.9,
                             title="时间地点需核对", explanation="第三条引用提供同日依据。", suggestion="核对原文。", evidence=spans, extra={})
            db.add(issue)
            db.commit()
            issue_id, frozen_hash = issue.id, source.content_sha256
        for index, span in enumerate(spans):
            response = client.get(f"/api/v1/analysis-runs/{run_id}/evidence-preview", params={"kind": "issue", "item_id": issue_id, "evidence_index": index})
            assert response.status_code == 200, response.text
            assert [row["text"] for row in response.json()["lines"] if row["is_evidence"]] == [span["text"]]
        assert client.get(f"/api/v1/analysis-runs/{run_id}/export.md").status_code == 200

        with patch("app.main.dispatch_analysis"):
            old_id = client.post(runs_url, json={}).json()["id"]
            with SessionLocal() as db:
                old = db.get(AnalysisRunRow, old_id)
                old.batch_coverage = {key: value for key, value in old.batch_coverage.items() if key != "narrative_time_knowledge_policy"}
                old.status = "failed"
                db.commit()
            retry = client.post(f"/api/v1/analysis-runs/{old_id}/retry")
            assert retry.status_code == 202, retry.text
            with SessionLocal() as db:
                copied = db.get(AnalysisRunRow, retry.json()["id"])
                assert "narrative_time_knowledge_policy" not in copied.batch_coverage
                assert db.scalar(select(AnalysisRunInputRow).where(AnalysisRunInputRow.run_id == copied.id)).content_sha256 == frozen_hash
            before = len(sent)
            with patch.object(OpenAICompatibleProvider, "__init__", provider_init):
                execute_analysis(retry.json()["id"], raise_on_failure=True)
            assert sent[before:] and set(sent[before:]) <= {SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT}
            replacement = client.post(documents_url, json={"name": "draft.md", "content": content.replace("东港", "西塔"), "replace_document_id": document["id"]})
            assert replacement.status_code == 201, replacement.text
            recheck = client.post(f"/api/v1/analysis-runs/{run_id}/rechecks")
            assert recheck.status_code == 202, recheck.text
            with SessionLocal() as db:
                assert db.get(AnalysisRunRow, recheck.json()["id"]).batch_coverage["narrative_time_knowledge_policy"] == NARRATIVE_TIME_KNOWLEDGE_POLICY_V1
        original = client.get(f"/api/v1/analysis-runs/{run_id}/evidence-preview", params={"kind": "issue", "item_id": issue_id, "evidence_index": 2})
        assert original.status_code == 200
        assert [row["text"] for row in original.json()["lines"] if row["is_evidence"]] == [spans[2]["text"]]
        with TestClient(app) as other:
            register(other)
            assert other.get(f"/api/v1/analysis-runs/{run_id}/evidence-preview", params={"kind": "issue", "item_id": issue_id, "evidence_index": 2}).status_code == 404


@pytest.mark.parametrize(("relation", "label"), [("unknown_before", "此前尚未知晓"), ("unknown_through", "截至此时仍未知晓"), ("known_by", "截至此时已知晓"), ("unclear", "知情时间待确认")])
def test_knowledge_constraint_projection_never_claims_an_acquisition_or_invents_a_date(relation, label):
    span = {"document_id": "doc", "document_name": "draft.md", "line_start": 1, "line_end": 1, "text": "正文依据。"}
    row = SimpleNamespace(id="constraint", kind="knowledge_constraint", attrs={"character": "乔予", "fact": "坐标", "time": "14:00", "relation": relation, "knowledge_scope": "whole_content"}, evidence=span)
    timeline = project_timeline("run", [row], []).model_dump(mode="json")
    assert label in json.dumps(timeline, ensure_ascii=False)
    assert "获知" not in json.dumps(timeline, ensure_ascii=False)
    assert "T14:00" not in json.dumps(timeline)
    assert not project_graph("run", [row], []).edges
