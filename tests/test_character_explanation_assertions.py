"""Bounded source/semantic contracts with a synthetic provider, no model calls."""
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.character_explanation_assertions import (
    EXPLANATION_ASSERTION_REFS_KEY, assertion_index, valid_assertion_binding,
)
from app.character_explanation_review import (
    EXPLANATION_REVIEW_SCHEMA_V3, EXPLANATION_REVIEW_SYSTEM_PROMPT_V3,
    EXPLANATION_REVIEW_USER_PREFIX, ExplanationBaselineSummary,
    ExplanationCandidate, ExplanationObservationSummary,
    run_character_explanation_review,
)
from app.config import CHARACTER_EXPLANATION_ASSERTIONS_POLICY_V1
from app.domain import EvidenceSpan
from tests.test_character_explanation_review import _settings, _item


CONTENT = "许舟被委托宣读完整名单，她接着连续讲了很久。"
CAUSE = "许舟被委托宣读完整名单"
ACTION = "她接着连续讲了很久"


def fixtures(content=CONTENT):
    span = EvidenceSpan(document_id="draft", document_name="draft.md", line_start=1, line_end=1, text=content)
    candidate = ExplanationCandidate(citation="E01", source_kind="draft", publication_status="draft",
        authority_tier="draft", resolution_state="confirmed", source_ordinal=1,
        eligible_draft_document_ids=("draft",), promotion_cap="assertion_bound", evidence=span)
    current = ExplanationObservationSummary(citation="C01", observation_id="cs_assertion_current",
        statement="许舟连续讲了很久", document_id="draft", source_ordinal=1, evidence=span)
    baseline = ExplanationBaselineSummary(character="许舟", dimension="speech_pattern",
        trait_key="concise_speech", statement="许舟说话始终简短")
    return candidate, current, baseline


def selection_for(row, quote):
    assertion = next(item for item in row["assertion_index"]["assertions"] if quote in item["text"])
    return {"assertion_id": assertion["id"], "quote": quote}


def response_for(request, *, cause=CAUSE, action=ACTION, independence="independent_fact", updates=None):
    """Stable fixture hook for platform/native tests and the safe observer."""
    items = []
    for candidate in request["candidates"]:
        current = request["observations"][0]
        item = dict(citation=candidate["citation"], explanation_type="temporary_state_or_pressure",
            actuality="actual", actor_relation="same", axis_relation="same", temporal_relation="prior_or_active",
            causal_relation="explicit_causal", applicable_observation_citations=[current["citation"]],
            assertion_links=[dict(observation_citation=current["citation"], independence=independence,
                explanation=selection_for(candidate, cause), observation=selection_for(current, action))])
        item.update(updates or {})
        items.append(item)
    return dict(schema_version=EXPLANATION_REVIEW_SCHEMA_V3, request_digest=request["request_digest"], items=items)


class AssertionProvider:
    def __init__(self, transform=lambda value: value, **options):
        self.calls = []
        self.transform, self.options = transform, options

    def complete(self, system, user):
        request = json.loads(user.removeprefix(EXPLANATION_REVIEW_USER_PREFIX))
        self.calls.append((system, request))
        value = self.transform(response_for(request, **self.options))
        return SimpleNamespace(text=json.dumps(value, ensure_ascii=False), prompt_tokens=20, completion_tokens=80)


def run_fixture(provider, *, content=CONTENT, enabled=True, **settings_values):
    candidate, current, baseline = fixtures(content)
    config = _settings(**settings_values)
    if enabled:
        config._character_explanation_assertions_policy = CHARACTER_EXPLANATION_ASSERTIONS_POLICY_V1
    else:
        candidate = candidate.model_copy(update={"promotion_cap": "possible_only"})
    return run_character_explanation_review((candidate,), baseline=baseline,
        observations=(current,), provider=provider, settings=config)


def test_same_sentence_independent_assertions_are_bound_without_repeating_subject():
    provider = AssertionProvider()
    result = run_fixture(provider)
    assert result.coverage == "complete" and len(provider.calls) == 1
    assert provider.calls[0][0] == EXPLANATION_REVIEW_SYSTEM_PROMPT_V3
    support, = result.support_evidence
    assert support.kind == "exception" and support.summary == CAUSE
    assert support.evidence.text == CONTENT
    candidate, current, _ = fixtures()
    assert valid_assertion_binding(support.assertion_binding_v1, candidate.evidence, {current.observation_id: current.evidence})
    assert support.assertion_binding_v1["links"][0]["explanation"]["start"] == 0
    assert "assertion_binding_v1" not in support.model_dump()
    assert len(assertion_index(current.evidence)["assertions"]) == 1  # Commas do not force splitting.


@pytest.mark.parametrize("independence,updates,expected,coverage", [
    ("current_behavior", {}, None, "complete"),
    ("purpose_only", {}, None, "complete"),
    ("unverified_claim", {}, "possible_explanation", "partial"),
    ("unclear", {}, "possible_explanation", "partial"),
    ("independent_fact", {"causal_relation": "bounded_inference"}, "possible_explanation", "complete"),
    ("independent_fact", {"axis_relation": "different"}, None, "complete"),
    ("independent_fact", {"actuality": "hypothetical"}, None, "complete"),
    ("independent_fact", {"temporal_relation": "future"}, None, "complete"),
])
def test_semantic_slots_preserve_self_purpose_nonactual_object_and_uncertainty_boundaries(independence, updates, expected, coverage):
    result = run_fixture(AssertionProvider(independence=independence, updates=updates))
    assert [row.kind for row in result.support_evidence] == ([expected] if expected else [])
    assert result.coverage == coverage


@pytest.mark.parametrize("mode", ["same_span", "missing_quote", "wrong_handle", "wrong_digest", "old_schema"])
def test_source_and_protocol_failures_never_create_definitive_support(mode):
    def alter(value):
        link = value["items"][0]["assertion_links"][0]
        if mode == "same_span": link["explanation"] = dict(link["observation"])
        if mode == "missing_quote": link["explanation"]["quote"] = "原文未提供的原因"
        if mode == "wrong_handle": link["observation_citation"] = "C99"
        if mode == "wrong_digest": value["request_digest"] = "0" * 64
        if mode == "old_schema": value["schema_version"] = "character-explanation-review-v2"
        return value
    result = run_fixture(AssertionProvider(alter))
    assert not result.support_evidence and result.coverage == "partial"


def test_unicode_ref_offsets_and_tampering_are_source_bound():
    content = "📝" + CONTENT
    provider = AssertionProvider()
    result = run_fixture(provider, content=content)
    support, = result.support_evidence
    assert support.assertion_binding_v1["links"][0]["explanation"]["start"] == 1  # Not UTF-16 units.
    proof = json.loads(json.dumps(support.assertion_binding_v1))
    proof["links"][0]["explanation"]["start"] = 2
    candidate, current, _ = fixtures(content)
    assert not valid_assertion_binding(proof, candidate.evidence, {current.observation_id: current.evidence})


def test_target_counterstatement_is_not_an_independent_exception_even_at_different_offsets():
    cause = "许舟说自己从来不会简短表达"
    content = cause + "，" + ACTION + "。"
    provider = AssertionProvider(cause=cause, independence="current_behavior")
    result = run_fixture(provider, content=content)
    assert not result.support_evidence
    assert provider.calls[0][1]["baseline"]["statement"] == "许舟说话始终简短"
    assert "不能只说明动作动机" in provider.calls[0][0]


def test_old_v2_same_line_cap_remains_and_new_fields_do_not_upgrade_it():
    from tests.test_character_explanation_review import SequenceProvider, _response_envelope
    candidate, current, baseline = fixtures()
    candidate = candidate.model_copy(update={"promotion_cap": "possible_only"})
    response = _response_envelope((candidate,), [_item(candidate, explanation_type="temporary_state_or_pressure")],
                                  baseline=baseline, observations=(current,))
    provider = SequenceProvider(response)
    result = run_character_explanation_review((candidate,), baseline=baseline, observations=(current,),
                                             provider=provider, settings=_settings())
    assert result.support_evidence[0].kind == "possible_explanation"
    assert result.support_evidence[0].assertion_binding_v1 is None


def test_source_index_omission_and_small_budget_are_visible_without_extra_calls():
    provider = AssertionProvider()
    result = run_fixture(provider, content=CONTENT * 65)
    assert result.coverage == "partial" and "source_index_truncated" in result.diagnostics.failure_reasons
    assert not provider.calls
    result = run_fixture(provider, character_explanation_token_budget=512, character_explanation_max_completion_tokens=256)
    assert result.coverage == "partial" and "token_budget" in result.diagnostics.failure_reasons
    assert not provider.calls


def test_new_windows_keep_same_sentence_source_and_old_windows_keep_legacy_exclusion():
    from tests.test_character_explanation_window_coverage import _baseline, _observation, _source
    from app.character_consistency_stage import _explanation_window_specs_with_coverage
    from app.narrative_context import NarrativeScopeV1
    content = "林澈受命暂扮健谈商人，她随后与陌生人长谈。"
    observation = _observation().model_copy(update={"evidence": _observation().evidence.model_copy(update={"text": content})})
    source = _source(document_id="draft", content=content, ordinal=1, source_kind="draft")
    args = dict(baseline=_baseline(), baseline_scope=NarrativeScopeV1(), observations=(observation,),
        draft_scopes=(NarrativeScopeV1(),), draft_ordinals=(1,), draft_document_ids=("draft",), documents=[source])
    old, _ = _explanation_window_specs_with_coverage(**args)
    new, coverage = _explanation_window_specs_with_coverage(**args, assertion_policy=True)
    assert not old
    assert len(new) == 1 and new[0].promotion_cap == "assertion_bound"
    assert new[0].evidence.text == content and coverage.compound_observation_ids == (observation.id,)


def test_native_service_persists_same_line_assertion_proof_and_reviews_focused_explanation():
    """Real stage/journal/storage, synthetic model; no claim of real quality."""
    from fastapi.testclient import TestClient
    from sqlalchemy import func, select
    from app.character_consistency_stage import CharacterConsistencyStage
    from app.character_drift import CHARACTER_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT
    from app.config import (HABIT_SCENE_POLICY_V1, HABIT_OBJECT_CHAIN_POLICY_V1,
        HABIT_ACTION_SELECTION_POLICY_V1, CLAIM_RESOLUTION_POLICY_V1,
        SELECTED_CLAIM_RESOLUTION_POLICY_V3, CHARACTER_TARGET_FOCUS_POLICY_V1)
    from app.db import AnalysisRunRow, AnalysisDiagnosticRow, AccountProviderConfigRow, CharacterObservationRow, SessionLocal
    from app.main import app, write_limiter
    from app.service import execute_analysis, _CharacterConsistencyAccountingProvider
    from tests.test_habit_scene_policy import AuthorSettings
    from tests.test_character_consistency_stage import (
        _confirmed_speech_project, _create_document, _context, _new_run, _record, _response,
    )
    from tests.test_character_target_focus import focused_response

    cause = "祁雾正在按委托伪装成健谈商人"
    action = "祁雾一口气说了很长一段话"
    content = cause + "，" + action + "。"
    profile = "祁雾说话始终简短，这是她稳定的说话方式。"

    class NativeProvider:
        def __init__(self):
            self.calls, self.main_requests = [], []

        def complete(self, system, user):
            self.calls.append(system)
            if system == EXPLANATION_REVIEW_SYSTEM_PROMPT_V3:
                request = json.loads(user.removeprefix(EXPLANATION_REVIEW_USER_PREFIX))
                value = response_for(request, cause=cause, action=action,
                                     updates={"explanation_type": "disguise_or_role"})
            elif system == CHARACTER_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT:
                request = json.loads(user)
                self.main_requests.append(request)
                value = focused_response(request, verdict="explained", relation="compatible", basis="none")
                value["citations"].append("X01")
            else:
                formal = "character_profile" in user
                value = json.loads(_response(_record(character="祁雾", evidence=profile if formal else content,
                    polarity="positive" if formal else "negative", kind="explicit_declaration" if formal else "speech_sample",
                    dimension="speech_pattern", trait_key="concise_speech", statement="说话始终简短" if formal else action)))
            return SimpleNamespace(text=json.dumps(value, ensure_ascii=False), prompt_tokens=20, completion_tokens=80)

    provider, results = NativeProvider(), []
    original_run = CharacterConsistencyStage.run

    def observed_run(stage, *args, **kwargs):
        assert stage.settings._character_explanation_assertions_policy == CHARACTER_EXPLANATION_ASSERTIONS_POLICY_V1
        stage.provider = _CharacterConsistencyAccountingProvider(stage.settings, stage.provider.usage,
                                                                 signal_provider=provider, drift_provider=provider)
        result = original_run(stage, *args, **kwargs)
        results.append(result)
        return result

    write_limiter.events.clear()
    with TestClient(app) as client:
        project = _confirmed_speech_project(client)
        _create_document(client, project["id"], name="draft.md", role="chapter", content=content,
                         narrative_context=_context(publication="draft"))
        run_id = _new_run(client, project["id"])
        with SessionLocal() as db:
            run = db.get(AnalysisRunRow, run_id)
            owner = run.requested_by_user_id
            revision = (db.scalar(select(func.max(AccountProviderConfigRow.revision)).where(
                AccountProviderConfigRow.user_id == owner)) or 0) + 1
            config = AccountProviderConfigRow(user_id=owner, revision=revision, base_url="https://assertions.invalid/v1",
                model_name="fixture", endpoint_sha256="a" * 64, state="usable")
            db.add(config)
            db.flush()
            config_id = config.id
            baseline_run_id = db.scalar(select(AnalysisRunRow.id).where(
                AnalysisRunRow.project_id == project["id"], AnalysisRunRow.id != run_id
            ).order_by(AnalysisRunRow.created_at.desc()).limit(1))
            assert baseline_run_id
            run.provider_config_id, run.batch_mode = config_id, "draft_review"
            run.batch_coverage = {**(run.batch_coverage or {}), "character_review_enabled": True,
                "character_baseline_run_id": baseline_run_id, "habit_scene_policy": HABIT_SCENE_POLICY_V1,
                "habit_object_chain_policy": HABIT_OBJECT_CHAIN_POLICY_V1,
                "habit_action_selection_policy": HABIT_ACTION_SELECTION_POLICY_V1,
                "claim_resolution_policy": CLAIM_RESOLUTION_POLICY_V1,
                "selected_claim_resolution_policy": SELECTED_CLAIM_RESOLUTION_POLICY_V3,
                "target_focus_policy": CHARACTER_TARGET_FOCUS_POLICY_V1,
                "explanation_assertions_policy": CHARACTER_EXPLANATION_ASSERTIONS_POLICY_V1}
            db.commit()
        native = AuthorSettings(_env_file=None, **{**_settings().model_dump(),
            "enable_model_extraction": False, "enable_review_agent": False, "enable_embeddings": False,
            "character_explanation_review_v1": True, "provider_runtime_config_id": config_id,
            "provider_runtime_user_id": owner, "character_consistency_stage_token_budget": 100000,
            "per_run_token_budget": 120000})
        runtime = SimpleNamespace(settings=native, identity=None, available=True, unavailable_reason=None)
        with patch("app.service.runtime_for_analysis_run", return_value=runtime), patch.object(CharacterConsistencyStage, "run", observed_run):
            execute_analysis(run_id, raise_on_failure=True)
        assert len(results) == 1
        result = results[0]
        assert not result.issues and not result.review_clues
        assert result.diagnostics["case_trace"][0]["final_outcome"] == "no_issue"
        assert result.diagnostics["case_trace"][0]["explanation_coverage"] == "complete"
        payload = result.diagnostics[EXPLANATION_ASSERTION_REFS_KEY]
        receipt, = payload["receipts"]
        assert receipt["kind"] == "exception" and payload["omitted_count"] == 0
        assert provider.calls.count(EXPLANATION_REVIEW_SYSTEM_PROMPT_V3) == 1
        request, = provider.main_requests
        support = next(row for row in request["evidence"] if row["id"] == "X01")
        assert support["text"] == content
        assert support["assertion_focus"][0]["explanation_text"] == cause
        assert support["assertion_focus"][0]["current_text"] == action
        with SessionLocal() as db:
            durable = db.get(AnalysisDiagnosticRow, run_id).payload["character_consistency"]
            assert durable[EXPLANATION_ASSERTION_REFS_KEY] == payload
            observations = db.scalars(select(CharacterObservationRow).where(CharacterObservationRow.run_id == run_id)).all()
            assert receipt["binding"]["links"][0]["observation_id"] in {row.observation_id for row in observations}
