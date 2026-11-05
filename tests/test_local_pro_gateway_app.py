"""The optional launcher must never inherit another LoreGuard instance's config."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


def test_ordinary_deployment_rejects_requests_to_enable_the_local_profile():
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from app.main import app, write_limiter
    from tests.test_review_batch import _document, _project

    with (TestClient(app) as client, patch("app.main.dispatch_analysis") as dispatch,
          patch.dict(write_limiter.events, clear=True)):
        profile = client.get("/api/v1/account/model-provider").json()
        assert "character_review_opt_in_available" not in profile
        project = _project(client)["id"]
        _document(client, project, name="profile.md", role="character_profile", publication="published")
        response = client.post(f"/api/v1/projects/{project}/analysis-runs", json={
            "mode": "baseline_build", "character_review_enabled": True,
        })
        assert response.status_code == 422
        dispatch.assert_not_called()


@pytest.mark.parametrize("runtime", ["", "relative/path"])
def test_local_gateway_factory_rejects_missing_or_relative_runtime(monkeypatch, runtime):
    from scripts import local_pro_gateway_app

    monkeypatch.setenv("LOREGUARD_PRO_GATEWAY_RUNTIME", runtime)
    with pytest.raises(RuntimeError, match="fresh Python process|absolute runtime"):
        local_pro_gateway_app.create_app()


def test_local_gateway_factory_excludes_env_and_dotenv_and_keeps_login(tmp_path):
    runtime = tmp_path / "gateway"
    runtime.mkdir()
    (runtime / "loreguard-data").mkdir()
    (runtime / "installation.json").write_text(
        json.dumps(
            {
                "owner": "LoreGuardLocalProGateway/v1",
                "version": "8.0.33",
                "runtime": str(runtime),
            }
        ),
        encoding="utf-8",
    )
    # A synthetic conflicting dotenv must never be constructed/read by Settings.
    (tmp_path / ".env").write_text(
        "AUTH_MODE=anonymous\nDEPLOYMENT_ENVIRONMENT=production\n"
        "OPENAI_API_KEY=synthetic-must-not-inherit\nDATABASE_URL=not-a-database\n",
        encoding="utf-8",
    )
    code = r'''
import json
import sys
from pathlib import Path
from fastapi.testclient import TestClient
sys.path.insert(0, sys.argv[1])
from scripts import local_pro_gateway_app as factory
factory._RUNTIME = Path(sys.argv[2])
application = factory.create_app()
from app.config import get_settings
settings = get_settings()
assert settings.auth_mode == "required"
assert settings.deployment_environment == "local"
assert settings.openai_api_key == ""
assert settings.account_model_keyring_json == ""
assert settings.account_model_keyring_file == ""
assert settings.local_model_gateway_enabled is True
assert settings.local_model_gateway_origins == "http://127.0.0.1:8317"
assert settings.enable_model_extraction is True
assert settings.enable_review_agent is False
assert settings.enable_embeddings is False
assert settings.use_celery is False
assert settings.execution_timing_profile == "local_quality"
assert settings.provider_timeout_seconds == 180
assert settings.provider_total_deadline_seconds == 360
assert settings.analysis_run_timeout_seconds == 2400
assert str(Path(sys.argv[2]) / "loreguard-data") in settings.account_model_local_key_path
with TestClient(application, base_url="http://127.0.0.1:8787") as client:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.headers["x-loreguard-local-pro"] == "1"
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/projects").status_code == 401
    assert client.get("/api/v1/account/model-provider").status_code == 401
    assert client.get("/health", headers={"Host": "untrusted.example"}).status_code == 400
    from tests.test_local_pro_gateway_app import _exercise_author_review
    _exercise_author_review(client)
try:
    factory.create_app()
except RuntimeError as failure:
    assert "fresh Python process" in str(failure)
else:
    raise AssertionError("A second factory initialization must fail closed")
print("isolated gateway application checks passed")
'''
    environment = {
        "LOREGUARD_PRO_GATEWAY_RUNTIME": str(runtime),
        "AUTH_MODE": "anonymous",
        "DEPLOYMENT_ENVIRONMENT": "production",
        "DATABASE_URL": "not-a-database",
        "OPENAI_API_KEY": "synthetic-must-not-inherit",
        "ACCOUNT_MODEL_KEYRING_FILE": str(tmp_path / "must-not-read.key"),
        "PROVIDER_TIMEOUT_SECONDS": "1",
        "ANALYSIS_RUN_TIMEOUT_SECONDS": "1",
    }
    # Python on Windows needs the system directory; this is not an app setting.
    if "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    result = subprocess.run(
        [sys.executable, "-c", code, str(REPOSITORY), str(runtime)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "isolated gateway application checks passed" in result.stdout


def _exercise_author_review(client):
    """Use the actual factory, account, snapshots, worker and report with scripted HTTP."""
    from unittest.mock import patch
    import httpx
    from app import service
    from app.main import write_limiter
    from app.config import CHARACTER_TARGET_FOCUS_POLICY_V1, CLAIM_RESOLUTION_POLICY_V1, HABIT_ACTION_SELECTION_POLICY_V1, HABIT_OBJECT_CHAIN_POLICY_V1, HABIT_SCENE_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2, SELECTED_CLAIM_RESOLUTION_POLICY_V3
    from app.db import AnalysisRunRow, AnalysisRunCharacterTraitInputRow, IssueRow, SessionLocal
    from sqlalchemy import select
    from app.character_drift import CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT, CHARACTER_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT
    from app.character_explanation_review import EXPLANATION_REVIEW_SYSTEM_PROMPT
    from app.character_scope_review_provider import SCOPE_REVIEW_SYSTEM_PROMPT
    from app.character_trait_extraction import (
        CHARACTER_SIGNAL_SYSTEM_PROMPT, TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT,
        build_formal_character_signal_system_prompt,
    )
    from app.model_extractor import SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT
    from app.provider import OpenAICompatibleProvider
    from tests.test_account_provider import register, save_provider
    from tests.test_character_baseline_recovery import _record as formal_record
    from tests.test_character_scope_review_pipeline import ReviewingProvider
    from tests.test_character_consistency_stage import _confirm_only_candidate, _record, _response
    from tests.test_character_single_event_stage import _RuntimeSingleEventProvider
    from tests.test_review_batch import _document, _project

    def frozen_scene_policy(run, policy="habit_scene_policy"):
        with SessionLocal() as db:
            return db.get(AnalysisRunRow, run["id"]).batch_coverage.get(policy)

    _, csrf = register(client)
    client.headers.update({"X-CSRF-Token": csrf})
    config_url = "/api/v1/account/model-provider"
    assert client.get(config_url).json()["character_review_opt_in_available"] is True
    project = _project(client)["id"]
    runs_url = f"/api/v1/projects/{project}/analysis-runs"
    profile = "祁雾一贯说话简短。"
    draft = "祁雾在日常汇报时一口气说了很长一段话。"
    profile_doc = _document(client, project, name="profile.md", role="character_profile",
                            publication="published", content=profile)
    draft_doc = _document(client, project, name="draft.md", role="chapter", publication="draft", content=draft)
    baseline = {"mode": "baseline_build", "character_review_enabled": True}
    assert client.post(runs_url, json=baseline).json()["detail"]["code"] == "character_review_model_unavailable"
    assert save_provider(client, csrf, expected_revision=0, api_key="synthetic-only", model="mock-author",
                         base_url="http://127.0.0.1:8317/v1").status_code == 200
    formal_prompt = build_formal_character_signal_system_prompt(scope_review_v1=True)
    formal = ReviewingProvider([formal_record(profile, character="祁雾", dimension="speech_pattern",
                                              trait_key="concise_speech", key_object="")])
    draft_response = _response(_record(character="祁雾", evidence=draft, polarity="negative", kind="speech_sample",
                                       dimension="speech_pattern", trait_key="concise_speech"))
    reviewer = _RuntimeSingleEventProvider()
    calls = []
    draft_service_unavailable = False

    def respond(request):
        assert request.url.host == "127.0.0.1" and request.url.port == 8317
        body = json.loads(request.content)
        assert body["model"] == "mock-author"
        system, user = (message["content"] for message in body["messages"])
        calls.append(system)
        if draft_service_unavailable and system.startswith(
            (CHARACTER_SIGNAL_SYSTEM_PROMPT, TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT)
        ) and system != formal_prompt:
            return httpx.Response(503)
        if system == SYSTEM_PROMPT:
            content = '{"records":[]}'
        elif system == BATCH_SYSTEM_PROMPT:
            docs = json.loads(user.split("\n", 1)[1])["documents"]
            content = json.dumps({"documents": [{"doc_ref": row["doc_ref"], "records": []} for row in docs]})
        elif system in (formal_prompt, SCOPE_REVIEW_SYSTEM_PROMPT):
            content = formal.complete(system, user).text
        elif system == CHARACTER_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT:
            value = json.loads(reviewer.complete(CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT, user).text)
            request = json.loads(user)
            value["target_focus_sha256"] = request["target_focus_sha256"]
            for event in value["events"]:
                citation = event["current_citations"][0]
                quote = next(row["text"] for row in request["evidence"] if row["id"] == citation)
                event["target_focus"] = {"target_relation": "contradicts", "opposition_basis": "explicit",
                                         "counterevidence": [{"citation": citation, "quote": quote}]}
            content = json.dumps(value, ensure_ascii=False)
        elif system in (EXPLANATION_REVIEW_SYSTEM_PROMPT, CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT):
            content = reviewer.complete(system, user).text
        elif system.startswith(CHARACTER_SIGNAL_SYSTEM_PROMPT):
            content = draft_response
        else:
            raise AssertionError("Unexpected model contract in author workflow")
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}],
                                         "usage": {"prompt_tokens": 17, "completion_tokens": 9}})

    original_init = OpenAICompatibleProvider.__init__

    def with_mock_transport(self, config=None, **kwargs):
        kwargs["transport"] = httpx.MockTransport(respond)
        kwargs["sleep"] = lambda delay: None
        original_init(self, config, **kwargs)

    with (
        patch.object(OpenAICompatibleProvider, "__init__", with_mock_transport),
        patch("app.main.dispatch_analysis", side_effect=service.execute_analysis),
    ):
        def start(payload, **options):
            response = client.post(runs_url, json=payload, **options)
            assert response.status_code == 202, response.text
            return response.json()

        default = start({})
        assert frozen_scene_policy(default, "execution_limits") == {
            "schema_version": "execution-limits-v1", "read_seconds": 180,
            "logical_seconds": 360, "run_seconds": 2400,
        }
        assert default["review_batch"]["character_review_enabled"] is False
        assert calls and set(calls) <= {SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT}
        assert client.post(runs_url, json={"character_review_enabled": True}).status_code == 422
        assert client.post(runs_url, json={**baseline, "character_review_enabled": "true"}).status_code == 422

        created = start(baseline, headers={"Idempotency-Key": "baseline"})
        baseline_id = created["id"]
        assert created["review_batch"]["character_review_enabled"] is True
        assert formal_prompt in calls and SCOPE_REVIEW_SYSTEM_PROMPT in calls
        assert start(baseline, headers={"Idempotency-Key": "baseline"})["deduplicated"]
        assert client.post(runs_url, json={"mode": "baseline_build"},
                           headers={"Idempotency-Key": "baseline"}).status_code == 409
        chosen = {"mode": "draft_review", "target_document_ids": [draft_doc["id"]]}
        before_private_override = len(calls)
        # Keep these independent schema checks out of the long workflow's
        # existing write-rate window, then restore that window unchanged.
        with patch.dict(write_limiter.events, clear=True):
            for override in (
                {"execution_limits": {"run_seconds": 99999}},
                {"analysis_run_timeout_seconds": 99999},
                {"_character_habit_action_selection_v1": True},
                {"habit_action_selection_policy": HABIT_ACTION_SELECTION_POLICY_V1},
                {"batch_coverage": {"habit_action_selection_policy": HABIT_ACTION_SELECTION_POLICY_V1}},
                {"_character_selected_claim_resolution_policy": SELECTED_CLAIM_RESOLUTION_POLICY_V3},
                {"selected_claim_resolution_policy": SELECTED_CLAIM_RESOLUTION_POLICY_V3},
                {"batch_coverage": {"selected_claim_resolution_policy": SELECTED_CLAIM_RESOLUTION_POLICY_V3}},
                {"_character_target_focus_policy": CHARACTER_TARGET_FOCUS_POLICY_V1},
                {"target_focus_policy": CHARACTER_TARGET_FOCUS_POLICY_V1},
                {"batch_coverage": {"target_focus_policy": CHARACTER_TARGET_FOCUS_POLICY_V1}},
            ):
                assert client.post(runs_url, json={**chosen, **override}).status_code == 422
        assert len(calls) == before_private_override
        opted_in = {**chosen, "character_review_enabled": True, "character_baseline_run_id": baseline_id}
        assert client.post(runs_url, json=opted_in).json()["detail"]["code"] == "character_baseline_not_ready"
        _confirm_only_candidate(client, project, baseline_id)

        # A temporary draft-extraction outage can leave a completed report with no decision.
        draft_service_unavailable = True
        interrupted = start(opted_in)
        interrupted_url = "/api/v1/analysis-runs/" + interrupted["id"]
        interrupted_run = client.get(interrupted_url).json()
        interrupted_stage = client.get(interrupted_url + "/diagnostics").json()["character_consistency"]
        assert interrupted_run["status"] == "completed"
        assert interrupted_stage["outcome"] == "partial"
        assert interrupted_stage["reason_counts"]["upstream_5xx"] > 0
        assert interrupted_stage["reason_counts"]["targeted_pass_upstream_5xx"] > 0
        assert interrupted_stage["target_assessment_summary_v1"]["needs_review"] == 1
        assert client.get(interrupted_url + "/issues").json() == []

        draft_service_unavailable = False
        calls.clear()
        assert client.post(interrupted_url + "/retry").status_code == 409
        unchanged = client.post(interrupted_url + "/rechecks", json={"character_review_enabled": True})
        assert unchanged.status_code == 409 and unchanged.json()["detail"]["code"] == "no_document_changes"
        assert not calls
        ordinary = start(chosen)
        assert ordinary["review_batch"]["character_review_enabled"] is False
        assert calls and set(calls) <= {SYSTEM_PROMPT, BATCH_SYSTEM_PROMPT}
        reviewed = start(opted_in)
        assert frozen_scene_policy(reviewed) == HABIT_SCENE_POLICY_V1
        assert all(frozen_scene_policy(run) is None for run in (default, ordinary, created))
        assert frozen_scene_policy(reviewed, "habit_object_chain_policy") == HABIT_OBJECT_CHAIN_POLICY_V1
        assert all(frozen_scene_policy(run, "habit_object_chain_policy") is None for run in (default, ordinary, created))
        assert frozen_scene_policy(reviewed, "claim_resolution_policy") == CLAIM_RESOLUTION_POLICY_V1
        assert all(frozen_scene_policy(run, "claim_resolution_policy") is None for run in (default, ordinary, created))
        assert frozen_scene_policy(reviewed, "habit_action_selection_policy") == HABIT_ACTION_SELECTION_POLICY_V1
        assert all(frozen_scene_policy(run, "habit_action_selection_policy") is None for run in (default, ordinary, created))
        assert frozen_scene_policy(reviewed, "selected_claim_resolution_policy") == SELECTED_CLAIM_RESOLUTION_POLICY_V3
        assert all(frozen_scene_policy(run, "selected_claim_resolution_policy") is None for run in (default, ordinary, created))
        assert frozen_scene_policy(reviewed, "target_focus_policy") == CHARACTER_TARGET_FOCUS_POLICY_V1
        assert all(frozen_scene_policy(run, "target_focus_policy") is None for run in (default, ordinary, created))
        assert reviewed["review_batch"]["character_baseline_run_id"] == baseline_id
        assert calls.count(CHARACTER_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT) == 1
        run_url = "/api/v1/analysis-runs/" + reviewed["id"]
        recovered = client.get(run_url).json()
        assert recovered["status"] == "completed" and recovered["id"] != interrupted["id"]
        assert recovered["review_batch"]["character_review_enabled"] is True
        assert recovered["retried_from"] is None and recovered["comparison_id"] is None
        assert recovered["input_documents"] == interrupted_run["input_documents"]
        assert recovered["character_profile_snapshot_sha256"] == interrupted_run["character_profile_snapshot_sha256"]
        assert client.get(interrupted_url + "/diagnostics").json()["character_consistency"] == interrupted_stage
        assert client.get(interrupted_url + "/issues").json() == []
        worker = client.get(run_url + "/diagnostics").json()["runtime_provenance"]
        api = client.get("/health").json()["runtime_provenance"]
        assert worker["capabilities"] == api["capabilities"]
        assert worker["character_consistency_limits"] == api["character_consistency_limits"]
        assert worker["character_consistency_limits"]["drift_review_max_generations"] == 2
        issues = client.get(run_url + "/issues").json()
        issue, = issues
        assert issue["metadata"]["reviewed_events"]
        assert "target_focus_review_v1" not in issue["metadata"]
        assert "target_focus_review_v1" not in json.dumps(client.get(run_url + "/diagnostics").json())
        with SessionLocal() as db:
            persisted = db.get(IssueRow, issue["id"])
            frozen = db.scalar(select(AnalysisRunCharacterTraitInputRow).where(AnalysisRunCharacterTraitInputRow.run_id == reviewed["id"]))
            assert persisted.extra["target_focus_review_v1"]
            assert frozen.payload["value"] in issue["explanation"]
        assert {row["document_id"] for row in issue["evidence"]} == {profile_doc["id"], draft_doc["id"]}
        for index, citation in enumerate(issue["evidence"]):
            response = client.get(run_url + "/evidence-preview", params={
                "kind": "issue", "item_id": issue["id"], "evidence_index": index,
            })
            assert response.status_code == 200
            expected = profile if citation["document_id"] == profile_doc["id"] else draft
            assert [row["text"] for row in response.json()["lines"] if row["is_evidence"]] == [expected]
        for payload in (chosen, opted_in):
            with patch("app.main.dispatch_analysis", side_effect=RuntimeError("synthetic dispatch failure")):
                failed = client.post(runs_url, json=payload)
            assert failed.status_code == 503
            enabled = payload is opted_in
            with SessionLocal() as db:
                failed_run = db.get(AnalysisRunRow, failed.json()["detail"]["run_id"])
                assert failed_run.batch_coverage.get("habit_object_chain_policy") == (HABIT_OBJECT_CHAIN_POLICY_V1 if enabled else None)
                assert failed_run.batch_coverage.get("claim_resolution_policy") == (CLAIM_RESOLUTION_POLICY_V1 if enabled else None)
                assert failed_run.batch_coverage.get("habit_action_selection_policy") == (HABIT_ACTION_SELECTION_POLICY_V1 if enabled else None)
                assert failed_run.batch_coverage.get("selected_claim_resolution_policy") == (SELECTED_CLAIM_RESOLUTION_POLICY_V3 if enabled else None)
                assert failed_run.batch_coverage.get("target_focus_policy") == (CHARACTER_TARGET_FOCUS_POLICY_V1 if enabled else None)
                # Replay a genuine older snapshot shape after the application has upgraded.
                failed_run.batch_coverage = {key: value for key, value in failed_run.batch_coverage.items() if key not in {"habit_object_chain_policy", "claim_resolution_policy", "habit_action_selection_policy", "selected_claim_resolution_policy", "target_focus_policy", "execution_limits"}}
                db.commit()
            calls.clear()
            retry = client.post("/api/v1/analysis-runs/" + failed.json()["detail"]["run_id"] + "/retry")
            assert retry.status_code == 202
            assert retry.json()["review_batch"]["character_review_enabled"] is enabled
            assert frozen_scene_policy(retry.json()) == (HABIT_SCENE_POLICY_V1 if enabled else None)
            assert frozen_scene_policy(retry.json(), "habit_object_chain_policy") is None
            assert frozen_scene_policy(retry.json(), "claim_resolution_policy") is None
            assert frozen_scene_policy(retry.json(), "habit_action_selection_policy") is None
            assert frozen_scene_policy(retry.json(), "selected_claim_resolution_policy") is None
            assert frozen_scene_policy(retry.json(), "target_focus_policy") is None
            assert frozen_scene_policy(retry.json(), "execution_limits") is None
            assert (CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT in calls) is enabled
        # Earlier frozen runs retain their routing version after upgrade.
        with patch.dict(write_limiter.events, clear=True):
            for policy in (SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2):
                with patch("app.main.dispatch_analysis", side_effect=RuntimeError("synthetic dispatch failure")):
                    failed = client.post(runs_url, json=opted_in)
                assert failed.status_code == 503
                with SessionLocal() as db:
                    failed_run = db.get(AnalysisRunRow, failed.json()["detail"]["run_id"])
                    failed_run.batch_coverage = {**failed_run.batch_coverage, "selected_claim_resolution_policy": policy}
                    db.commit()
                retry = client.post("/api/v1/analysis-runs/" + failed.json()["detail"]["run_id"] + "/retry")
                assert retry.status_code == 202, retry.text
                assert frozen_scene_policy(retry.json(), "selected_claim_resolution_policy") == policy
        # A new draft version requires a fresh choice, even after a paid review.
        _document(client, project, name="draft.md", role="chapter", publication="draft",
                  content=draft, replace_document_id=draft_doc["id"])
        for source, choice in ((ordinary, None), (reviewed, None), (reviewed, True)):
            calls.clear()
            recheck_url = "/api/v1/analysis-runs/" + source["id"] + "/rechecks"
            options = {} if choice is None else {
                "json": {"character_review_enabled": True}, "headers": {"Idempotency-Key": "deep-recheck"},
            }
            recheck = client.post(recheck_url, **options)
            assert recheck.status_code == 202, recheck.text
            enabled = choice is True
            assert recheck.json()["review_batch"]["character_review_enabled"] is enabled
            assert frozen_scene_policy(recheck.json()) == (HABIT_SCENE_POLICY_V1 if enabled else None)
            assert frozen_scene_policy(recheck.json(), "habit_object_chain_policy") == (HABIT_OBJECT_CHAIN_POLICY_V1 if enabled else None)
            assert frozen_scene_policy(recheck.json(), "claim_resolution_policy") == (CLAIM_RESOLUTION_POLICY_V1 if enabled else None)
            assert frozen_scene_policy(recheck.json(), "habit_action_selection_policy") == (HABIT_ACTION_SELECTION_POLICY_V1 if enabled else None)
            assert frozen_scene_policy(recheck.json(), "selected_claim_resolution_policy") == (SELECTED_CLAIM_RESOLUTION_POLICY_V3 if enabled else None)
            assert frozen_scene_policy(recheck.json(), "target_focus_policy") == (CHARACTER_TARGET_FOCUS_POLICY_V1 if enabled else None)
            assert frozen_scene_policy(recheck.json(), "execution_limits")["run_seconds"] == 2400
            assert (CHARACTER_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT in calls) is enabled
        repeated = client.post(recheck_url, headers={"Idempotency-Key": "deep-recheck"})
        assert repeated.status_code == 409, repeated.text
        _document(client, project, name="profile.md", role="character_profile", publication="published",
                  content=profile + "新设定。", replace_document_id=profile_doc["id"])
        assert client.post(run_url + "/rechecks", json={"character_review_enabled": True}
                           ).json()["detail"]["code"] == "character_baseline_not_ready"
