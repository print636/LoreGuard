"""Single-event runtime admission and frozen-source review receipts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.character_consistency_stage import (
    CHARACTER_CONSISTENCY_CHECKER_VERSION,
    _case_trace_comparison_key,
    _reviewed_event_group_id,
    _reviewed_events_receipt,
    _safe_case_trace,
    _target_assessment_summary,
    _target_has_sufficient_recall_evidence,
    _to_issue,
    _target_coverage_certificate,
)
from app.character_drift import (
    CHARACTER_REVIEW_POLICY_VERSION,
    CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT,
    CharacterDriftCase,
    CharacterReviewDiagnostics,
    CharacterReviewResult,
    ModelDriftDecision,
    prepare_character_drift,
    promote_character_drift,
)
from app.character_explanation_review import (
    EXPLANATION_REVIEW_SCHEMA_V2, EXPLANATION_REVIEW_SYSTEM_PROMPT,
    EXPLANATION_REVIEW_USER_PREFIX,
)
from app.character_trait_extraction import CharacterSignalTarget
from app.character_trait_extraction import (
    BaselineCoverageTraceV1, CharacterSignalChunk, CharacterSignalDiagnostics,
    CharacterSignalExtractionResult, CharacterSignalExtractor, BASELINE_COVERAGE_SYSTEM_PROMPT_V3,
)
from app.character_scope_review import ScopeReviewSourceIdentity
from app.config import Settings
from app.main import app, write_limiter
from app.runtime_provenance import safe_runtime_provenance
from app.service import CharacterConsistencyUsageAccumulator, _CharacterConsistencyAccountingProvider
from tests.test_character_consistency_stage import (
    QueueProvider, _confirmed_speech_project, _context, _create_document,
    _new_run, _record, _response, _run_stage, _settings,
    _confirm_only_candidate,
)
from tests.test_character_target_coverage import _fixture
from tests.test_character_baseline_recovery import (
    RecoveryProvider, _record as formal_record, _audit,
)
from tests.test_character_scope_review_pipeline import _settings as extraction_settings


@pytest.fixture(autouse=True)
def _clear_write_rate_limiter():
    # Isolate each case's write budget while retaining limits within the case.
    write_limiter.events.clear()
    try:
        yield
    finally:
        write_limiter.events.clear()


def _event(*handles, verdict="contradicts"):
    return {
        "current_citations": list(handles), "verdict": verdict,
        "explanation": "冻结原文显示该次表现与设定相反。",
        "actor_match": "same", "trait_match": "same", "actuality": "actual",
        "object_match": "same", "situation_match": "same",
    }


def _decision(*events, citations=("B01", "C01")):
    return ModelDriftDecision.model_validate_json(json.dumps({
        "policy_version": CHARACTER_REVIEW_POLICY_VERSION,
        "verdict": "contradicts", "explanation": "该次明确表现违反已确认设定。",
        "citations": list(citations), "events": list(events or (_event("C01"),)),
    }, ensure_ascii=False))


def _new_receipt_fixture():
    candidate, sources, baseline, old_prepared, _, _, _ = _fixture()
    prepared = prepare_character_drift(old_prepared.case, review_policy=CHARACTER_REVIEW_POLICY_VERSION)
    review = CharacterReviewResult(
        decision=_decision(),
        diagnostics=CharacterReviewDiagnostics(outcome="completed", reason="completed"),
    )
    return candidate, sources, baseline, prepared, review


def test_receipt_and_report_share_actual_reviewed_group_and_frozen_sources():
    candidate, sources, baseline, prepared, review = _new_receipt_fixture()
    comparison = _case_trace_comparison_key(baseline)
    receipt = _reviewed_events_receipt(
        prepared, review, candidate_id=candidate, comparison_key=comparison,
        frozen_by_document=sources,
    )
    assert receipt is not None
    promoted = promote_character_drift(prepared, review)
    assert promoted.outcome == "conflict"
    issue = _to_issue(
        promoted=promoted, prepared=prepared, review=review,
        confirmed_candidate_id=candidate, judgement="contradicts",
        frozen_by_document=sources, comparison_key=comparison,
    )
    trace = _safe_case_trace(
        character_key=baseline.character, baseline=baseline,
        baseline_entry=(SimpleNamespace(candidate_id=candidate), baseline, None, baseline.character),
        matched_observation_count=1, matched_observations=prepared.matching_observations,
        prepared=prepared, prepare_reason=prepared.reason, material_coverage="complete",
        review=review, final_outcome=promoted.outcome, visible=True,
        promote_reason=promoted.reason, frozen_by_document=sources,
    )
    assert receipt == issue.metadata["reviewed_events"] == trace["reviewed_events"]
    assert issue.metadata["evidence_binding"] == "review_citations_v1"
    assert receipt["assessment_policy"] == trace["review_policy"] == CHARACTER_REVIEW_POLICY_VERSION
    assert [ref["handle"] for ref in receipt["evidence_refs"]] == ["B01", "C01"]
    member = receipt["evidence_refs"][1]
    assert member["observation_id_sha256"] == hashlib.sha256(prepared.matching_observations[0].id.encode()).hexdigest()
    assert receipt["events"][0]["event_group_id"] == _reviewed_event_group_id(
        receipt["confirmed_candidate_id_sha256"], comparison, [member],
    )
    assert member["text_sha256"] == hashlib.sha256(issue.evidence[1].text.encode()).hexdigest()
    assert "text" not in member and "explanation" not in receipt["events"][0]
    assert "server_evidence_pair" not in trace


def test_source_change_or_legacy_decision_cannot_create_new_review_receipt():
    candidate, sources, baseline, prepared, review = _new_receipt_fixture()
    comparison = _case_trace_comparison_key(baseline)
    changed = dict(sources)
    source = changed["draft"]
    changed["draft"] = replace(source, content_sha256="f" * 64)
    assert _reviewed_events_receipt(
        prepared, review, candidate_id=candidate, comparison_key=comparison,
        frozen_by_document=changed,
    ) is None
    legacy = review.model_copy(update={"decision": ModelDriftDecision(
        verdict="contradicts", explanation="旧记录没有逐事件审查。", citations=("B01", "C01"),
    )})
    assert _reviewed_events_receipt(
        prepared, legacy, candidate_id=candidate, comparison_key=comparison,
        frozen_by_document=sources,
    ) is None


def test_recall_one_grounded_action_is_sufficient_but_wrong_axis_is_not():
    _, _, _, prepared, _ = _new_receipt_fixture()
    observation = prepared.matching_observations[0].model_copy(update={"observation_kind": "action"})
    target = CharacterSignalTarget(
        character="林澈", dimension="preference", trait_key=observation.trait_key,
        key_object="蜜瓜", requested_polarity="negative",
        comparison_key="preference:蜜瓜", baseline_polarity="positive", baseline_hint="林澈喜欢蜜瓜。",
    )
    assert _target_has_sufficient_recall_evidence(target, (observation,))
    assert not _target_has_sufficient_recall_evidence(
        target, (observation.model_copy(update={"dimension": "speech_pattern"}),),
    )


class _RuntimeSingleEventProvider(QueueProvider):
    def complete(self, system, user):
        if system == CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT:
            self.calls.append((system, user))
            value = _decision().model_dump(mode="json")
            # Excluded archive fields are explicitly supplied on the wire.
            value.update(policy_version=CHARACTER_REVIEW_POLICY_VERSION, events=[_event("C01")])
            return SimpleNamespace(text=json.dumps(value, ensure_ascii=False), prompt_tokens=17, completion_tokens=9)
        if system == EXPLANATION_REVIEW_SYSTEM_PROMPT:
            self.calls.append((system, user))
            request = json.loads(user.removeprefix(EXPLANATION_REVIEW_USER_PREFIX))
            value = {
                "schema_version": EXPLANATION_REVIEW_SCHEMA_V2, "request_digest": request["request_digest"],
                "items": [{
                    "citation": row["citation"], "explanation_type": "irrelevant", "actuality": "actual",
                    "actor_relation": "same", "axis_relation": "same", "temporal_relation": "prior_or_active",
                    "causal_relation": "none", "applicable_observation_citations": [],
                } for row in request["candidates"]],
            }
            return SimpleNamespace(text=json.dumps(value, ensure_ascii=False), prompt_tokens=17, completion_tokens=9)
        return super().complete(system, user)


def _single_action_run(client, provider_class=_RuntimeSingleEventProvider):
    profile = "祁雾说话始终简短，这是她稳定的说话方式。"
    draft = "祁雾在日常汇报时一口气说了很长一段话。"
    project = _confirmed_speech_project(client)
    _create_document(client, project["id"], name="draft.md", role="chapter", content=draft,
                     narrative_context=_context(publication="draft"))
    provider = provider_class(
        _response(_record(character="祁雾", evidence=profile, polarity="positive", kind="explicit_declaration",
                          dimension="speech_pattern", trait_key="concise_speech", statement="说话始终简短")),
        _response(_record(character="祁雾", evidence=draft, polarity="negative", kind="speech_sample",
                          dimension="speech_pattern", trait_key="concise_speech")),
    )
    return _new_run(client, project["id"]), provider


def test_actual_stage_single_action_calls_new_contract_and_emits_formal_receipt():
    with TestClient(app) as client:
        run_id, provider = _single_action_run(client)
        result = _run_stage(run_id, provider, character_explanation_review_v1=True)
    assert len(result.issues) == 1 and not result.review_clues
    assert sum(system == CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT for system, _ in provider.calls) == 1
    trace = result.diagnostics["case_trace"][0]
    assert trace["review_policy"] == CHARACTER_REVIEW_POLICY_VERSION
    assert trace["review_outcome"] == "completed" and trace["final_outcome"] == "conflict"
    assert result.issues[0].metadata["reviewed_events"] == trace["reviewed_events"]
    assert len(trace["reviewed_events"]["events"]) == 1
    assert trace["target_coverage"]["status"] == "complete"
    assert result.diagnostics["target_assessment_summary_v1"]["conflict"] == 1
    assert result.issues[0].metadata["checker_version"] == CHARACTER_CONSISTENCY_CHECKER_VERSION


@pytest.mark.parametrize(("correction", "expected_calls"), [
    ("recovered", 2), ("invalid_response", 2), ("attempt_limit", 1),
    ("token_budget", 1), ("cancelled", 1),
])
def test_stage_contract_correction_preserves_accounting_and_frozen_evidence(correction, expected_calls):
    from scripts.run_character_consistency_live import _safe_case_trace_summary

    class ContractProvider(_RuntimeSingleEventProvider):
        def complete(self, system, user):
            if system == CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT and (
                "contract_correction" not in json.loads(user) or correction == "invalid_response"
            ):
                self.calls.append((system, user))
                return SimpleNamespace(
                    text="malformed raw-model sentinel",
                    prompt_tokens=7900 if correction == "token_budget" else 17,
                    completion_tokens=9,
                )
            return super().complete(system, user)

    cancelled = RuntimeError("cancel before correction")
    settings = {
        "character_explanation_review_v1": True,
        "character_drift_review_max_generations": 1 if correction == "attempt_limit" else 2,
    }
    ledger = CharacterConsistencyUsageAccumulator()
    with TestClient(app) as client:
        run_id, model = _single_action_run(client, ContractProvider)
        provider = _CharacterConsistencyAccountingProvider(
            _settings(**settings), ledger, signal_provider=model, drift_provider=model,
        )

        def checkpoint():
            if correction == "cancelled" and any(
                system == CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT for system, _ in model.calls
            ):
                raise cancelled

        if correction == "cancelled":
            with pytest.raises(RuntimeError) as caught:
                _run_stage(run_id, provider, checkpoint=checkpoint, **settings)
            assert caught.value is cancelled
        else:
            result = _run_stage(run_id, provider, checkpoint=checkpoint, **settings)

    assert sum(system == CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT for system, _ in model.calls) == expected_calls
    assert ledger.logical_calls == len(model.calls)
    if correction == "cancelled":
        assert ledger.safe_dict(terminal_status="cancelled")["charged_tokens"] > 0
        return
    usage = result.diagnostics["usage"]
    assert usage["attempted_calls"] == ledger.logical_calls
    assert result.prompt_tokens == ledger.prompt_tokens
    assert result.completion_tokens == ledger.completion_tokens
    assert result.charged_tokens == ledger.charged_tokens
    counts = result.diagnostics["counts"]
    assert counts["planned_chunks"] == counts["model_called_chunks"] == counts["model_completed_chunks"] == 2
    assert counts["drift_reviewed"] == expected_calls
    assert counts["drift_correction_attempted_call_count"] == expected_calls - 1
    assert counts["drift_correction_recovered_case_count"] == int(correction == "recovered")
    trace = result.diagnostics["case_trace"][0]
    assert trace["review_contract_failure"] == "json"
    assert trace["review_correction_status"] == correction
    summary = _safe_case_trace_summary(trace)
    assert summary["review_contract_failure"] == "json" and summary["review_correction_status"] == correction
    assert "sentinel" not in json.dumps(result.diagnostics)
    assert "review_contract_failure" not in _safe_case_trace_summary({**trace, "review_contract_failure": ["raw"]})
    if correction == "recovered":
        assert trace["final_outcome"] == "conflict"
        assert result.issues[0].metadata["reviewed_events"] == trace["reviewed_events"]
    else:
        assert not result.issues and "reviewed_events" not in trace
        assert trace["review_outcome"] == "degraded"
        assert result.diagnostics["reason_counts"]["review_invalid_model_response"] == 1


def test_stage_limits_primary_correction_to_remaining_run_tokens():
    class InvalidPrimaryProvider(_RuntimeSingleEventProvider):
        def complete(self, system, user):
            if system == CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT:
                self.calls.append((system, user))
                return SimpleNamespace(text="{", prompt_tokens=17, completion_tokens=9)
            return super().complete(system, user)

    with TestClient(app) as client:
        # Measure an admitted first pass through extraction, explanation and review,
        # then allow only that charge and a small rounding margin for the whole run.
        reference_id, reference_model = _single_action_run(client, InvalidPrimaryProvider)
        reference = _run_stage(
            reference_id, reference_model, character_explanation_review_v1=True,
            character_drift_review_max_generations=1,
        )
        assert reference.diagnostics["case_trace"][0]["review_correction_status"] == "attempt_limit"
        run_id, model = _single_action_run(client, InvalidPrimaryProvider)
        limited = _run_stage(
            run_id, model, character_explanation_review_v1=True,
            remaining_run_tokens=reference.charged_tokens + 10,
        )
    assert limited.diagnostics["case_trace"][0]["review_correction_status"] == "token_budget"
    assert limited.diagnostics["counts"]["drift_reviewed"] == 1
    assert limited.charged_tokens <= reference.charged_tokens + 10


def test_provenance_records_runtime_policy_and_preserves_explicit_smaller_limits():
    defaults = Settings(_env_file=None)
    assert defaults.character_drift_token_budget == 8000
    assert defaults.character_drift_max_completion_tokens == 2500
    explicit = Settings(_env_file=None, character_drift_token_budget=1200, character_drift_max_completion_tokens=1000)
    limits = safe_runtime_provenance(explicit)["character_consistency_limits"]
    assert limits["drift_review_policy_version"] == CHARACTER_REVIEW_POLICY_VERSION
    assert limits["drift_review_schema_version"] == "character-single-event-review-v1"
    assert limits["reviewed_events_schema_version"] == "character-reviewed-events-v1"
    assert limits["checker_version"] == CHARACTER_CONSISTENCY_CHECKER_VERSION
    assert limits["drift_token_budget"] == 1200 and limits["drift_max_completion_tokens"] == 1000
    with pytest.raises(ValidationError):
        Settings(_env_file=None, character_drift_max_completion_tokens=4001)


def test_primary_generation_limit_and_prompt_identity_round_trip_for_old_and_current_runs():
    from scripts.run_character_axis_live import _safe_character_runtime_provenance
    from scripts.run_evidence_investigator_live import _safe_runtime_provenance
    from tests.test_runtime_provenance import configured_settings

    provenance = safe_runtime_provenance(configured_settings())
    limits = provenance["character_consistency_limits"]
    assert limits["drift_review_prompt_version"] == "character-single-event-review-prompt-v2"
    assert limits["drift_review_max_generations"] == 2
    for bound in (1, 2, 0, 3, True, "2"):
        limits["drift_review_max_generations"] = bound
        valid = type(bound) is int and bound in (1, 2)
        assert (_safe_runtime_provenance(provenance) == provenance) == valid
        assert (_safe_character_runtime_provenance(provenance) is not None) == valid
    del limits["drift_review_max_generations"]
    assert _safe_runtime_provenance(provenance) is None
    limits["drift_review_prompt_version"] = "character-single-event-review-prompt-v1"
    assert _safe_runtime_provenance(provenance) == provenance
    assert _safe_character_runtime_provenance(provenance)["character_consistency_limits"] == limits
    limits["drift_review_max_generations"] = 1
    assert _safe_runtime_provenance(provenance) is None
    assert _safe_character_runtime_provenance(provenance) is None


def test_stage_recovery_attempt_is_shared_across_formal_document_chunks(monkeypatch):
    calls = []

    class CapturingExtractor:
        def __init__(self, *args, **kwargs):
            pass

        def extract(self, chunk, **kwargs):
            calls.append(kwargs)
            used = kwargs["baseline_coverage_recovery_already_used"]
            return CharacterSignalExtractionResult(diagnostics=CharacterSignalDiagnostics(
                outcome="partial", attempted_calls=1 if used else 2,
                raw_records=0, accepted_records=0, rejected_records=0,
                prompt_tokens=17, completion_tokens=9, charged_tokens=26,
                baseline_coverage_trace=BaselineCoverageTraceV1(
                    status="partial", attempted_calls=0 if used else 1,
                    indexed_assertions=1, represented_assertions=0, unresolved_assertions=1,
                ),
            ))

    monkeypatch.setattr("app.character_consistency_stage.CharacterSignalExtractor", CapturingExtractor)
    text = "\n".join(f"林澈第{index}条设定坚持核对访客身份后才放行。" for index in range(36))
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "文档补查调用范围"}).json()
        _create_document(client, project["id"], name="profile.md", role="character_profile", content=text,
                         narrative_context=_context(publication="published"))
        result = _run_stage(
            _new_run(client, project["id"]), QueueProvider(),
            character_signal_max_chunk_chars=256, character_signal_full_line_prompt_v2=True,
            character_signal_support_id_v4=True, character_signal_semantic_scope_v5=True,
            character_signal_scope_review_v1=True,
        )
    assert len(calls) >= 2
    assert all(call["baseline_coverage_recovery"] is True for call in calls)
    assert [call["baseline_coverage_recovery_already_used"] for call in calls] == [False, *([True] * (len(calls) - 1))]
    assert result.diagnostics["outcome"] == "partial"
    assert result.diagnostics["reason_counts"]["baseline_coverage_partial"] == len(calls)
    assert result.prompt_tokens == 17 * len(calls)


def test_primary_package_regeneration_still_has_only_one_baseline_audit():
    first = "林蔚长期喜欢枇杷。"
    second = "林蔚一贯说话简洁直接。"
    supplemental = formal_record(second, line=2, dimension="speech_pattern", trait_key="directness", key_object="")

    class RegeneratingProvider(RecoveryProvider):
        primary_calls = 0

        def complete(self, system, user):
            if "角色信号抽取器" in system and BASELINE_COVERAGE_SYSTEM_PROMPT_V3 not in system:
                self.primary_calls += 1
                if self.primary_calls == 1:
                    self.calls.append((system, user))
                    invalid = {**formal_record(first), "evidence": "原文不存在的引用"}
                    return SimpleNamespace(text=json.dumps({"records": [invalid]}, ensure_ascii=False),
                                           prompt_tokens=17, completion_tokens=9)
            return super().complete(system, user)

    text = first + "\n" + second
    provider = RegeneratingProvider([formal_record(first)], _audit([supplemental]))
    chunk = CharacterSignalChunk("profile", "profile.md", text, 1, "formal_character_profile")
    identity = ScopeReviewSourceIdentity(run_input_id="source-input", document_id="profile", document_version=1,
                                        content_sha256=hashlib.sha256(text.encode()).hexdigest())
    result = CharacterSignalExtractor(provider, settings=extraction_settings(
        character_signal_token_budget=40000, character_signal_package_max_attempts=2,
    )).extract(chunk, source_identity=identity, frozen_content=text, baseline_coverage_recovery=True)
    assert provider.primary_calls == 2 and provider.audit_calls == 1
    assert result.diagnostics.baseline_coverage_trace.attempted_calls == 1
    assert result.diagnostics.baseline_coverage_trace.status == "audited"
    assert {candidate.dimension for candidate in result.pending_candidates} == {"preference", "speech_pattern"}


def test_new_runtime_provenance_versions_cannot_be_tampered_or_partially_removed():
    import copy
    from scripts.run_evidence_investigator_live import _safe_runtime_provenance
    from tests.test_runtime_provenance import configured_settings

    provenance = safe_runtime_provenance(configured_settings())
    assert _safe_runtime_provenance(provenance) == provenance
    for key in ("checker_version", "drift_review_policy_version", "drift_review_schema_version",
                "drift_review_prompt_version", "reviewed_events_schema_version"):
        changed = copy.deepcopy(provenance)
        changed["character_consistency_limits"][key] = "unknown-version"
        assert _safe_runtime_provenance(changed) is None
        del changed["character_consistency_limits"][key]
        assert _safe_runtime_provenance(changed) is None


@pytest.fixture(scope="module")
def outside_scope_runtime_result():
    profile = "林澈在陌生人面前从不主动交谈。"
    draft = "林澈主动和多年好友交谈。"
    baselines = []

    def capture_prepare(case, **kwargs):
        if case.observations:
            baselines.append(case.baseline)
        return prepare_character_drift(case, **kwargs)

    class OutsideProvider(QueueProvider):
        def complete(self, system, user):
            if system == CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT:
                self.calls.append((system, user))
                event = {**_event("C01", verdict="explained"), "situation_match": "different"}
                wire = {
                    "policy_version": CHARACTER_REVIEW_POLICY_VERSION, "events": [event],
                    "verdict": "explained", "explanation": "设定针对陌生人，当前交谈对象为多年好友。",
                    "citations": ["B01", "C01"],
                }
                return SimpleNamespace(text=json.dumps(wire, ensure_ascii=False), prompt_tokens=17, completion_tokens=9)
            return super().complete(system, user)

    formal = _record(evidence=profile, polarity="negative", kind="explicit_declaration",
                     dimension="core_personality", trait_key="社交主动性")
    current = _record(evidence=draft, polarity="positive", kind="action",
                      dimension="core_personality", trait_key="社交主动性")
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "明确范围外的单次表现"}).json()
        _create_document(client, project["id"], name="profile.md", role="character_profile", content=profile,
                         narrative_context=_context(publication="published"))
        seed = _new_run(client, project["id"])
        _run_stage(seed, QueueProvider(_response(formal)))
        _confirm_only_candidate(client, project["id"], seed)
        _create_document(client, project["id"], name="draft.md", role="chapter", content=draft,
                         narrative_context=_context(publication="draft"))
        with patch("app.character_consistency_stage.prepare_character_drift", side_effect=capture_prepare):
            result = _run_stage(_new_run(client, project["id"]), OutsideProvider(_response(formal), _response(current)))
    return result, baselines[-1]


def _recertify_outside(trace, baseline, **updates):
    values = {
        "baseline": baseline, "unresolved_alias_observations": (), "formal_signals": (),
        "frozen_by_document": {}, "target_aliases": (),
        "processing": {"planned_chunks": 2, "processed_chunks": 2, "model_completed_chunks": 2,
                       "model_incomplete_chunks": 0, "model_uncalled_chunks": 0},
        "pre_alias_partial": False, "baseline_limit": False, "explanation_processing_partial": True,
    }
    values.update(updates)
    return _target_coverage_certificate(trace, **values)


def test_complete_outside_scope_target_has_no_explanation_dependency_but_global_warning_remains(outside_scope_runtime_result):
    result, _ = outside_scope_runtime_result
    assert not result.issues and not result.review_clues
    trace = result.diagnostics["case_trace"][0]
    assert trace["final_outcome"] == "no_issue" and trace["promote_reason"] == "single_event_outside_comparison"
    assert result.diagnostics["target_assessment_summary_v1"]["no_issue"] == 1
    assert trace["explanation_coverage"] == "not_run"
    assert result.diagnostics["outcome"] == "partial"
    assert result.diagnostics["reason_counts"]["explanation_review_not_run"] == 1
    cert = trace["target_coverage"]
    assert cert["schema_version"] == "character-target-coverage-v2" and cert["status"] == "complete"
    assert cert["explanation_coverage"] == "not_run" and cert["non_alias_blockers"] == []
    requirement = cert["explanation_requirement"]
    assert requirement["state"] == "not_required_by_scope"
    from app.narrative_context import payload_sha256
    assert requirement["reviewed_events_sha256"] == payload_sha256(trace["reviewed_events"])
    assert requirement["target_observation_pool_sha256"] == payload_sha256(trace["target_observation_pool"])


def test_target_summary_counts_only_uniquely_bound_decisions(outside_scope_runtime_result):
    result, _ = outside_scope_runtime_result
    trace = result.diagnostics["case_trace"][0]
    target = trace["confirmed_candidate_id_sha256"]
    eligible = {target, "0" * 64}
    for traces in (
        [trace, trace],
        [{**trace, "reviewed_events": None}],
        [{**trace, "target_coverage": {**trace["target_coverage"], "status": "partial"}}],
        [{**trace, "reviewed_events": {**trace["reviewed_events"], "evidence_refs": []}}],
    ):
        summary = _target_assessment_summary(traces, eligible_targets=eligible)
        assert summary == {
            "schema_version": "character-target-assessment-summary-v1",
            "total_targets": 2, "needs_review": 1, "unprocessed": 1,
            "conflict": 0, "no_issue": 0, "no_matching_observation": 0,
        }


@pytest.mark.parametrize("failure", ["material", "processing", "unreviewed", "missing_pool", "extra_pool_member"])
def test_outside_scope_does_not_hide_missing_material_or_omitted_observations(outside_scope_runtime_result, failure):
    import copy
    result, baseline = outside_scope_runtime_result
    trace = copy.deepcopy(result.diagnostics["case_trace"][0])
    options = {}
    if failure == "material":
        trace["material_coverage"] = "partial"
        trace["material_coverage_reasons"] = ["case_draft_package_incomplete"]
    elif failure == "processing":
        options["processing"] = {"planned_chunks": 2, "processed_chunks": 2, "model_completed_chunks": 1,
                                 "model_incomplete_chunks": 1, "model_uncalled_chunks": 0}
    elif failure == "unreviewed":
        trace["unreviewed_observation_count"] = 1
    elif failure == "missing_pool":
        del trace["target_observation_pool"]
    else:
        pool = trace["target_observation_pool"]
        pool["observation_count"] += 1
        pool["observation_refs"].append({**pool["observation_refs"][0], "observation_id_sha256": "f" * 64})
        pool["observation_refs"].sort(key=lambda ref: ref["observation_id_sha256"])
    cert = _recertify_outside(trace, baseline, **options)
    assert cert["status"] == "partial"
    if failure in {"unreviewed", "missing_pool", "extra_pool_member"}:
        assert cert["schema_version"] == "character-target-coverage-v1"
        assert "explanation_requirement" not in cert


@pytest.mark.parametrize("gap,blocker", [
    ("post_alias_non_alias_partial", "post_alias_processing_gap"),
    ("other_explanation_processing_partial", "other_case_explanation_incomplete"),
    ("pre_alias_partial", "pre_alias_processing_gap"),
])
def test_scope_waives_only_this_targets_explanation_gap(outside_scope_runtime_result, gap, blocker):
    result, baseline = outside_scope_runtime_result
    trace = result.diagnostics["case_trace"][0]
    cert = _recertify_outside(trace, baseline, **{gap: True})
    assert cert["schema_version"] == "character-target-coverage-v2" and cert["status"] == "partial"
    assert blocker in cert["non_alias_blockers"]


def test_bounded_review_preserves_formal_event_but_reports_actual_unreviewed_pool():
    profile = "祁雾说话始终简短，这是她稳定的说话方式。"
    lines = [f"第{index + 1}天，祁雾一口气说了很长一段话。" for index in range(8)]

    class BoundedProvider(_RuntimeSingleEventProvider):
        def complete(self, system, user):
            if system == CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT:
                self.calls.append((system, user))
                request = json.loads(user)
                handles = [row["id"] for row in request["evidence"] if row["id"].startswith("C")]
                event = _event(*handles)
                wire = {"policy_version": CHARACTER_REVIEW_POLICY_VERSION, "events": [event],
                        "verdict": "contradicts", "explanation": "已审查的当前表现违反稳定设定。",
                        "citations": ["B01", *handles]}
                return SimpleNamespace(text=json.dumps(wire, ensure_ascii=False), prompt_tokens=17, completion_tokens=9)
            return super().complete(system, user)

    with TestClient(app) as client:
        project = _confirmed_speech_project(client)
        _create_document(client, project["id"], name="draft.md", role="chapter", content="\n".join(lines),
                         narrative_context=_context(publication="draft"))
        provider = BoundedProvider(
            _response(_record(character="祁雾", evidence=profile, polarity="positive", kind="explicit_declaration",
                              dimension="speech_pattern", trait_key="concise_speech", statement="说话始终简短")),
            _response(*[_record(character="祁雾", evidence=line, polarity="negative", kind="speech_sample",
                                dimension="speech_pattern", trait_key="concise_speech", line=index + 1)
                        for index, line in enumerate(lines)]),
        )
        result = _run_stage(_new_run(client, project["id"]), provider, character_explanation_review_v1=True)
    assert len(result.issues) == 1
    trace = result.diagnostics["case_trace"][0]
    assert result.diagnostics["outcome"] == "partial"
    assert result.diagnostics["reason_counts"]["review_observation_pool_incomplete"] == 1
    assert trace["unreviewed_observation_count"] == result.issues[0].metadata["unreviewed_observation_count"] == 2
    assert trace["matched_observation_count"] == 6
    assert trace["target_observation_pool"]["observation_count"] == 8
    assert trace["target_coverage"]["status"] == "partial"
    assert "review_observation_pool_incomplete" in trace["target_coverage"]["non_alias_blockers"]
