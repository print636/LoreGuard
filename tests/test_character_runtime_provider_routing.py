"""Exercise the worker's real accounting/gateway boundary without a network."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app import service
from app.character_drift import (
    CHARACTER_REVIEW_SYSTEM_PROMPT,
    CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT,
    CharacterConsistencyReviewer,
    promote_character_drift,
)
from app.character_scope_review_provider import SCOPE_REVIEW_SYSTEM_PROMPT
from app.character_trait_extraction import (
    BASELINE_COVERAGE_SYSTEM_PROMPT_V1,
    CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2,
    CHARACTER_SIGNAL_SYSTEM_PROMPT,
    TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT,
    build_baseline_coverage_system_prompt,
    build_formal_character_signal_system_prompt,
    _bounded_provider,
)
from app.config import Settings
from app.db import (
    AnalysisDiagnosticRow, AnalysisRunRow, CharacterTraitCandidateRow, IssueRow,
    SessionLocal,
)
from app.main import app, write_limiter
from app.provider import OpenAICompatibleProvider, ProviderError, RetryPolicy
from app.service import (
    CharacterConsistencyUsageAccumulator, _CharacterConsistencyAccountingProvider,
)
from app.usage import estimate_issue_evidence_review_tokens
from tests.test_character_baseline_recovery import RecoveryProvider, _audit, _record
from tests.test_character_consistency_stage import (
    _confirmed_speech_project, _context, _create_document, _new_run,
    _record as draft_record, _response,
)
from tests.test_character_single_event_policy import event, payload, prepared
from tests.test_character_single_event_stage import _RuntimeSingleEventProvider


def _settings(**changes):
    values = {
        "openai_api_key": "unit-runtime-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_model_extraction": False,
        "enable_character_consistency": True,
        "provider_max_attempts": 1,
        "character_signal_max_attempts": 1,
        "character_drift_max_attempts": 1,
        "character_consistency_stage_token_budget": 60_000,
        "character_signal_token_budget": 40_000,
        "character_signal_package_max_attempts": 1,
        "character_signal_full_line_prompt_v2": True,
        "character_signal_support_id_v4": True,
        "character_signal_semantic_scope_v5": True,
        "character_signal_scope_review_v1": True,
    }
    values.update(changes)
    return Settings(_env_file=None, **values)


def _formal_system(settings):
    return build_formal_character_signal_system_prompt(
        scope_review_v1=settings.character_signal_scope_review_v1,
    )


def _response_http(value, *, prompt=17, completion=9):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": value}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    })


def _audit_system(settings):
    return build_baseline_coverage_system_prompt(_formal_system(settings))


def _http_accounting(settings, handler):
    base = OpenAICompatibleProvider(
        settings, transport=httpx.MockTransport(handler),
        retry_policy=RetryPolicy(max_attempts=1),
    )
    usage = CharacterConsistencyUsageAccumulator()
    return _CharacterConsistencyAccountingProvider(
        settings, usage,
        signal_provider=_bounded_provider(base, settings, stage="signal"),
        drift_provider=_bounded_provider(base, settings, stage="drift"),
    ), usage


@pytest.mark.parametrize("scope_review", [False, True])
def test_active_formal_audit_and_old_prompts_reach_accounted_http(scope_review):
    settings = _settings(character_signal_scope_review_v1=scope_review)
    requests = []

    def respond(request):
        requests.append(request)
        return _response_http("{}")

    accounting, usage = _http_accounting(settings, respond)
    formal = _formal_system(settings)
    audit = _audit_system(settings)
    primary_envelope = '只返回 JSON 对象 {"records":[...]}，不得返回 Markdown 或其他字段。'
    empty_primary_envelope = '没有有效记录时返回 {"records":[]}。'
    assert primary_envelope in formal and empty_primary_envelope in formal
    assert primary_envelope not in audit and empty_primary_envelope not in audit
    assert "schema_version、assessments、records" in audit
    systems = [formal, audit,
               CHARACTER_SIGNAL_SYSTEM_PROMPT + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2,
               TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT, CHARACTER_REVIEW_SYSTEM_PROMPT]
    if scope_review:
        systems.append(SCOPE_REVIEW_SYSTEM_PROMPT)
    for system in systems:
        accounting.complete(system, "source-bound request")
    assert [json.loads(row.content)["messages"][0]["content"] for row in requests] == systems
    expected_caps = [settings.character_signal_max_completion_tokens] * 4 + [
        settings.character_drift_max_completion_tokens,
    ]
    if scope_review:
        expected_caps.append(settings.character_signal_scope_review_completion_tokens)
    # Scope/provider bounds may tighten the reserve; they may never exceed it.
    for request, reserve in zip(requests, expected_caps, strict=True):
        body = json.loads(request.content)
        assert body["max_tokens"] <= reserve
        assert request.extensions["timeout"]["read"] <= settings.provider_timeout_seconds
    assert usage.logical_calls == usage.successful_calls == len(systems)
    assert (usage.prompt_tokens, usage.completion_tokens) == (17 * len(systems), 9 * len(systems))
    assert usage.charged_tokens == sum(estimate_issue_evidence_review_tokens(
        system, "source-bound request", completion_reserve=reserve,
    ) for system, reserve in zip(systems, expected_caps, strict=True))
    safe = usage.safe_dict(terminal_status="completed")
    assert safe["scope"] == "character_consistency"
    assert "source-bound request" not in json.dumps(safe)


@pytest.mark.parametrize("mutant", ["unknown", "prefix", "suffix", "wrong_parent", "targeted_parent", "repeat", "old_concat"])
def test_audit_purpose_is_exact_not_substring_or_fallback(mutant):
    settings = _settings()
    audit = _audit_system(settings)
    systems = {
        "unknown": "unregistered purpose",
        "prefix": "injected prefix" + audit,
        "suffix": audit + "injected suffix",
        "wrong_parent": CHARACTER_SIGNAL_SYSTEM_PROMPT + BASELINE_COVERAGE_SYSTEM_PROMPT_V1,
        "targeted_parent": TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT + BASELINE_COVERAGE_SYSTEM_PROMPT_V1,
        "repeat": audit + BASELINE_COVERAGE_SYSTEM_PROMPT_V1,
        "old_concat": _formal_system(settings) + BASELINE_COVERAGE_SYSTEM_PROMPT_V1,
    }
    requests = []
    accounting, usage = _http_accounting(settings, lambda req: requests.append(req))
    with pytest.raises(RuntimeError, match="unsupported character consistency provider purpose"):
        accounting.complete(systems[mutant], "private source")
    assert not requests and usage.logical_calls == usage.charged_tokens == 0


def test_disabled_or_inactive_formal_audit_rejected_before_accounting():
    configured = _settings()
    active_audit = _audit_system(configured)
    for settings in (
        configured.model_copy(update={"character_signal_scope_review_v1": False}),
        configured.model_copy(update={"character_signal_semantic_scope_v5": False,
                                      "character_signal_scope_review_v1": False}),
    ):
        requests = []
        accounting, usage = _http_accounting(settings, lambda req: requests.append(req))
        with pytest.raises(RuntimeError, match="unsupported character consistency provider purpose"):
            accounting.complete(active_audit, "private source")
        assert not requests and usage.logical_calls == usage.charged_tokens == 0


def test_new_reviewer_reaches_accounting_with_dynamic_reserve_and_one_charge():
    settings = _settings()
    requests = []

    def respond(request):
        requests.append(request)
        return _response_http(json.dumps(payload(event("C01")), ensure_ascii=False))

    accounting, usage = _http_accounting(settings, respond)
    candidate = prepared()
    result = CharacterConsistencyReviewer(accounting, settings=settings).review(candidate)
    assert result.diagnostics.outcome == "completed"
    assert promote_character_drift(candidate, result).outcome == "conflict"
    assert len(requests) == usage.logical_calls == usage.successful_calls == 1
    body = json.loads(requests[0].content)
    assert body["messages"][0]["content"] == CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT
    assert body["max_tokens"] == 384 + 320
    assert usage.charged_tokens == estimate_issue_evidence_review_tokens(
        body["messages"][0]["content"], body["messages"][1]["content"], completion_reserve=704,
    )


def test_explicit_small_review_cap_still_stops_before_any_gateway_call():
    settings = _settings(character_drift_max_completion_tokens=600)
    requests = []
    accounting, usage = _http_accounting(settings, lambda req: requests.append(req))
    result = CharacterConsistencyReviewer(accounting, settings=settings).review(prepared())
    assert result.diagnostics.outcome == "skipped"
    assert result.diagnostics.reason == "response_contract_budget"
    assert not requests and usage.logical_calls == usage.charged_tokens == 0


@pytest.mark.parametrize("purpose", ["audit", "single_event"])
def test_new_purpose_http_failure_is_charged_once_without_private_data(purpose):
    settings = _settings()
    system = (_audit_system(settings)
              if purpose == "audit" else CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT)
    requests = []

    def unavailable(request):
        requests.append(request)
        return httpx.Response(429, headers={"Retry-After": "0"},
                              json={"error": {"message": "private upstream message"}})

    accounting, usage = _http_accounting(settings, unavailable)
    with pytest.raises(ProviderError):
        accounting.complete(system, "private story")
    reserve = (settings.character_signal_max_completion_tokens if purpose == "audit"
               else settings.character_drift_max_completion_tokens)
    assert len(requests) == usage.logical_calls == 1
    assert usage.successful_calls == usage.prompt_tokens == usage.completion_tokens == 0
    assert usage.charged_tokens == estimate_issue_evidence_review_tokens(system, "private story", completion_reserve=reserve)
    safe = json.dumps(usage.safe_dict(terminal_status="failed"))
    assert all(value not in safe for value in ("private story", "private upstream", "mock.invalid", "unit-runtime-placeholder"))


class _QuietHeartbeat:
    def start(self):
        pass

    def raise_if_failed(self):
        pass

    def stop(self):
        pass


def _execute_service(run_id, settings, scripted, *, fail_purpose=None):
    """Keep worker, stage, accounting, gateway, and persistence real."""
    requests = []

    def respond(request):
        assert request.url.host == "mock.invalid"
        requests.append(request)
        body = json.loads(request.content)
        system, user = (item["content"] for item in body["messages"])
        if fail_purpose == "audit" and system.endswith(BASELINE_COVERAGE_SYSTEM_PROMPT_V1):
            return httpx.Response(429, headers={"Retry-After": "0"})
        if fail_purpose == "scope" and system == SCOPE_REVIEW_SYSTEM_PROMPT:
            return httpx.Response(503, headers={"X-Request-ID": "private-scope-header"},
                                  json={"error": {"message": "private-scope-body"}})
        result = scripted.complete(system, user)
        return _response_http(result.text, prompt=result.prompt_tokens, completion=result.completion_tokens)

    original_init = OpenAICompatibleProvider.__init__

    def with_mock_transport(self, config=None, **kwargs):
        kwargs.setdefault("transport", httpx.MockTransport(respond))
        original_init(self, config, **kwargs)

    runtime = SimpleNamespace(settings=settings, identity={"source": "unit-test"},
                              available=True, unavailable_reason=None)
    with (
        patch("app.service.get_settings", return_value=settings),
        patch("app.service.runtime_for_analysis_run", return_value=runtime),
        patch.object(OpenAICompatibleProvider, "__init__", with_mock_transport),
    ):
        service.execute_analysis(run_id, raise_on_failure=True,
                                 heartbeat_factory=lambda *_: _QuietHeartbeat())
    with SessionLocal() as db:
        run = db.get(AnalysisRunRow, run_id)
        assert run.status == "completed", run.error
        diagnostic = db.get(AnalysisDiagnosticRow, run_id).payload
        return requests, diagnostic, run.prompt_tokens, run.completion_tokens


@pytest.mark.parametrize("fail_purpose", [None, "audit", "scope"])
def test_worker_recovery_reaches_real_accounting_then_scope_and_persists_usage(fail_purpose):
    first, second = "林蔚长期喜欢枇杷。", "林蔚一贯说话简洁直接。"
    supplemental = _record(second, line=2, dimension="speech_pattern", trait_key="directness", key_object="")
    scripted = RecoveryProvider([_record(first)], _audit([supplemental]))
    settings = _settings()
    write_limiter.events.clear()
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "worker omission routing"}).json()
        _create_document(client, project["id"], name="profile.md", role="character_profile",
                         content=first + "\n" + second, narrative_context=_context(publication="published"))
        run_id = _new_run(client, project["id"])
        requests, diagnostic, prompt, completion = _execute_service(
            run_id, settings, scripted, fail_purpose=fail_purpose,
        )
        response = client.get(f"/api/v1/analysis-runs/{run_id}/diagnostics")
        assert response.status_code == 200
    assert len(requests) == 3
    systems = [json.loads(request.content)["messages"][0]["content"] for request in requests]
    assert systems == [_formal_system(settings),
                       _audit_system(settings),
                       SCOPE_REVIEW_SYSTEM_PROMPT]
    stage = diagnostic["character_consistency"]
    assert "stage_failed" not in stage.get("reason_counts", {})
    assert (stage["reason_counts"].get("baseline_coverage_partial", 0) > 0) == (fail_purpose is not None)
    failure_payload = stage["failure_diagnostics_v1"]
    assert response.json()["character_consistency"]["failure_diagnostics_v1"] == failure_payload
    assert failure_payload["chunks_omitted_count"] == 0
    chunk, = failure_payload["chunks"]
    assert (chunk["phase"], chunk["stage_chunk_ordinal"]) == ("primary_extraction", 1)
    series = chunk["diagnostics"]["scope_provider_failures"]
    assert series["omitted"] == 0
    if fail_purpose == "scope":
        # Both proposals abstained on the same failed logical provider call.
        assert stage["reason_counts"]["scope_review_provider_error"] == 2
        assert stage["material_coverage"] == "partial"
        assert stage["counts"]["signal_count"] == 0
        event, = series["events"]
        assert event["source_sha256"] == hashlib.sha256((first + "\n" + second).encode()).hexdigest()
        assert (event["origin"], event["round"], event["round_index"]) == ("complete_failure", "initial", 1)
        assert event["call_attempted"] is True and event["correction_reason"] is None
        assert (event["category"], event["http_status"], event["attempt_no"]) == ("upstream_5xx", 503, 1)
        attempt, = event["provider_attempts"]
        assert (attempt["category"], attempt["http_status"], attempt["attempt_no"]) == ("upstream_5xx", 503, 1)
        for canary in ("private-scope-header", "private-scope-body", "unit-runtime-placeholder"):
            assert canary not in response.text and canary not in json.dumps(diagnostic)
        projected = json.dumps(series, ensure_ascii=False)
        assert all(value not in projected for value in (first, second, "profile.md", "mock.invalid"))
    else:
        assert series["events"] == []
    with SessionLocal() as db:
        candidates = db.scalars(select(CharacterTraitCandidateRow).where(CharacterTraitCandidateRow.source_run_id == run_id)).all()
        assert len(candidates) == {None: 2, "audit": 1, "scope": 0}[fail_purpose]
    ledger = diagnostic["usage_accounting"]
    assert ledger["scope"] == "character_consistency"
    assert ledger["logical_calls"] == stage["usage"]["attempted_calls"] == 3
    assert (prompt, completion) == (ledger["prompt_tokens"], ledger["completion_tokens"])
    assert (prompt, completion) == {None: (49, 27), "audit": (30, 20), "scope": (36, 16)}[fail_purpose]
    expected_charge = sum(estimate_issue_evidence_review_tokens(
        body["messages"][0]["content"], body["messages"][1]["content"], completion_reserve=body["max_tokens"],
    ) for body in (json.loads(request.content) for request in requests))
    assert ledger["charged_tokens"] == stage["usage"]["charged_tokens"] == expected_charge
    assert first not in json.dumps(ledger, ensure_ascii=False)


def test_worker_single_event_reviewer_reaches_accounting_and_persists_formal_receipt():
    profile = "祁雾说话始终简短，这是她稳定的说话方式。"
    draft = "祁雾在日常汇报时一口气说了很长一段话。"
    settings = _settings(character_signal_support_id_v4=False, character_signal_semantic_scope_v5=False,
                         character_signal_scope_review_v1=False, character_explanation_review_v1=True)
    scripted = _RuntimeSingleEventProvider(
        _response(draft_record(character="祁雾", evidence=profile, polarity="positive", kind="explicit_declaration",
                               dimension="speech_pattern", trait_key="concise_speech", statement="说话始终简短")),
        _response(draft_record(character="祁雾", evidence=draft, polarity="negative", kind="speech_sample",
                               dimension="speech_pattern", trait_key="concise_speech")),
    )
    write_limiter.events.clear()
    with TestClient(app) as client:
        project = _confirmed_speech_project(client)
        _create_document(client, project["id"], name="draft.md", role="chapter", content=draft,
                         narrative_context=_context(publication="draft"))
        run_id = _new_run(client, project["id"])
        requests, diagnostic, prompt, completion = _execute_service(run_id, settings, scripted)
    systems = [json.loads(request.content)["messages"][0]["content"] for request in requests]
    assert systems.count(CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT) == 1
    trace = diagnostic["character_consistency"]["case_trace"][0]
    assert trace["review_outcome"] == "completed" and trace["final_outcome"] == "conflict"
    assert trace["target_coverage"]["status"] == "complete"
    with SessionLocal() as db:
        issues = db.scalars(select(IssueRow).where(IssueRow.run_id == run_id)).all()
        assert len(issues) == 1
        assert issues[0].extra["reviewed_events"] == trace["reviewed_events"]
    ledger = diagnostic["usage_accounting"]
    assert ledger["scope"] == "character_consistency"
    assert ledger["logical_calls"] == len(requests)
    assert (ledger["prompt_tokens"], ledger["completion_tokens"]) == (prompt, completion)
    assert draft not in json.dumps(ledger, ensure_ascii=False)
