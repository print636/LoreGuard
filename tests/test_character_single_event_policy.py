"""Ordinary synthetic stories for the explicit v3 policy; no gold/model use."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.character_drift import (
    CHARACTER_REVIEW_POLICY_VERSION, CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT,
    CharacterConsistencyReviewer, CharacterDriftCase, CharacterReviewDiagnostics,
    CharacterReviewResult, ConfirmedTraitSnapshot, ModelDriftDecision,
    SupportEvidence, prepare_character_drift, promote_character_drift, validated_reviewed_events,
)
from app.character_trait_extraction import CharacterSignal
from app.config import Settings
from app.domain import EvidenceSpan
from app.provider import ProviderError


def span(text, line=1, document="chapter"):
    return EvidenceSpan(document_id=document, document_name=f"{document}.md",
                        line_start=line, line_end=line, text=text)


def prepared(*, count=1, support=(), coverage="complete", explanation="complete",
             same_document=True, texts=None):
    base = ConfirmedTraitSnapshot(
        id="ct_synthetic", character="沈遥", dimension="behavior_boundary",
        trait_key="委托保密", statement="沈遥绝不为私利泄露委托人的地址。",
        polarity="positive", stability="stable", origin="explicit_setting",
        evidence=(span("沈遥绝不为私利泄露委托人的地址。", document="profile"),),
    )
    observations = tuple(CharacterSignal(
        id=f"cs_{i:032x}", character="沈遥", dimension="behavior_boundary",
        trait_key="委托保密", statement=(texts[i-1] if texts else f"沈遥为赏金向买家主动透露了第{i}位委托人的住址。"),
        polarity="negative", stability="temporary", observation_kind="action", source_kind="draft",
        evidence=span(texts[i-1] if texts else f"沈遥为赏金向买家主动透露了第{i}位委托人的住址。",
                      i, "chapter" if same_document else f"chapter-{i}"),
    ) for i in range(1, count + 1))
    case = CharacterDriftCase(
        id="cdc_synthetic", baseline=base, observations=observations,
        support_evidence=support, scope_compatibility="compatible",
        material_coverage=coverage, explanation_coverage=explanation,
    )
    return prepare_character_drift(case, review_policy=CHARACTER_REVIEW_POLICY_VERSION)


def event(*citations, verdict="contradicts", **overrides):
    result = dict(current_citations=list(citations), verdict=verdict,
                  explanation="该行为由当前原文直接支持。", actor_match="same",
                  trait_match="same", actuality="actual", object_match="same",
                  situation_match="same")
    result.update(overrides)
    return result


def payload(*events, verdict="contradicts", citations=("B01", "C01"), **overrides):
    result = dict(policy_version=CHARACTER_REVIEW_POLICY_VERSION, events=list(events),
                  verdict=verdict, explanation="当前实际行为与已确认设定直接冲突。",
                  citations=list(citations))
    result.update(overrides)
    return result


class Provider:
    def __init__(self, value):
        self.value, self.calls = value, []

    def complete(self, system, user):
        self.calls.append((system, json.loads(user)))
        if isinstance(self.value, Exception):
            raise self.value
        return SimpleNamespace(text=json.dumps(self.value, ensure_ascii=False),
                               prompt_tokens=20, completion_tokens=20)


def review(candidate, response, **settings):
    provider = Provider(response)
    config = Settings(_env_file=None, openai_api_key="unit-placeholder",
                      enable_character_consistency=True,
                      **settings)
    return CharacterConsistencyReviewer(provider, settings=config).review(candidate), provider


def support_for(*observation_indexes, kind="exception"):
    return SupportEvidence(
        id="se_synthetic", kind=kind, summary="已发生的保护性安排使地址是获准使用的联络地址。",
        explicit=True, evidence=span("委托人已授权使用这处公开联络地址。", document="history"),
        source_kind="published_history", publication_status="published",
        authority_tier="formal_record", resolution_state="confirmed", source_ordinal=1,
        applicable_observation_ids=tuple(f"cs_{i:032x}" for i in observation_indexes),
        selection_basis="semantic_relation_v1",
    )


def test_one_actual_action_is_reviewed_and_can_be_formal_without_repetition():
    candidate = prepared()
    assert candidate.reviewer_eligible
    result, provider = review(candidate, payload(event("C01")))
    assert len(provider.calls) == 1
    assert provider.calls[0][0] == CHARACTER_SINGLE_EVENT_REVIEW_SYSTEM_PROMPT
    assert len(validated_reviewed_events(candidate, result)) == 1
    assert promote_character_drift(candidate, result).outcome == "conflict"


def test_one_ambiguous_action_remains_a_clue():
    candidate = prepared()
    result, _ = review(candidate, payload(event("C01", verdict="needs_confirmation",
                                               actor_match="unclear"), verdict="needs_confirmation"))
    assert promote_character_drift(candidate, result).outcome == "needs_confirmation"


def test_legacy_model_response_cannot_obtain_new_formal_eligibility():
    candidate = prepared()
    response = dict(verdict="contradicts", explanation="原文确有相反表现。", citations=["B01", "C01"])
    result, provider = review(candidate, response)
    assert len(provider.calls) == 2
    assert result.diagnostics.outcome == "degraded"
    assert result.diagnostics.contract_failure == "policy"
    assert result.diagnostics.correction_status == "invalid_response"
    old_result = CharacterReviewResult(
        decision=ModelDriftDecision(**{**response, "citations": tuple(response["citations"])}),
        diagnostics=CharacterReviewDiagnostics(outcome="completed", reason="completed"),
    )
    assert validated_reviewed_events(candidate, old_result) == ()
    assert promote_character_drift(candidate, old_result).outcome == "unverifiable"


def test_two_fragments_of_same_event_remain_one_reviewed_group():
    candidate = prepared(count=2)
    result, provider = review(candidate, payload(event("C01", "C02"), citations=("B01", "C01", "C02")))
    assert len(provider.calls) == 1
    assert len(validated_reviewed_events(candidate, result)) == 1
    assert promote_character_drift(candidate, result).outcome == "conflict"


def test_unrelated_uncertainty_does_not_veto_clear_selected_event():
    candidate = prepared(count=2, same_document=False)
    result, _ = review(candidate, payload(event("C01"), event("C02", verdict="needs_confirmation",
                                                              actuality="unclear")))
    assert promote_character_drift(candidate, result).outcome == "conflict"


@pytest.mark.parametrize("field,value", [("object_match", "different"), ("actuality", "non_actual"),
                                         ("actor_match", "different")])
def test_clearly_outside_comparison_may_be_no_issue_without_invented_growth(field, value):
    candidate = prepared()
    result, _ = review(candidate, payload(event("C01", verdict="needs_confirmation", **{field: value}),
                                          verdict="explained"))
    assert result.diagnostics.outcome == "completed"
    assert promote_character_drift(candidate, result).outcome == "no_issue"


def test_effective_exception_explains_single_event():
    candidate = prepared(support=(support_for(1),))
    result, _ = review(candidate, payload(event("C01", verdict="explained"), verdict="explained",
                                          citations=("B01", "C01", "X01")))
    assert promote_character_drift(candidate, result).outcome == "no_issue"


def test_effective_exception_cannot_be_ignored_by_conflict_vote():
    candidate = prepared(support=(support_for(1),))
    result, _ = review(candidate, payload(event("C01")))
    assert promote_character_drift(candidate, result).outcome == "needs_confirmation"


def test_support_for_one_fragment_applies_to_entire_selected_event():
    candidate = prepared(count=2, support=(support_for(1),))
    result, _ = review(candidate, payload(event("C01", "C02"), citations=("B01", "C01", "C02")))
    assert promote_character_drift(candidate, result).outcome == "needs_confirmation"


def test_splitting_nearby_related_fragments_does_not_bypass_exception():
    candidate = prepared(count=2, support=(support_for(1),))
    result, _ = review(candidate, payload(event("C01", verdict="explained"), event("C02"),
                                          citations=("B01", "C02")))
    promoted = promote_character_drift(candidate, result)
    assert promoted.outcome == "needs_confirmation"
    assert promoted.reason == "related_event_explanation_unresolved"


def test_other_event_exception_does_not_veto_clear_event_with_source_boundary():
    candidate = prepared(count=2, support=(support_for(1),), texts=(
        "沈遥按授权交出公开联络地址。", "次日沈遥为赏金主动泄露另一位委托人的私人住址。"))
    result, _ = review(candidate, payload(event("C01", verdict="explained"), event("C02"),
                                          citations=("B01", "C02")))
    assert promote_character_drift(candidate, result).outcome == "conflict"


def test_explicit_continuation_cannot_use_time_marker_to_avoid_prior_exception():
    candidate = prepared(count=2, support=(support_for(1),), texts=(
        "沈遥按授权交出公开联络地址。", "次日沈遥继续同一场交接，把那张获准公开的地址交给联络人。"))
    result, _ = review(candidate, payload(event("C01", verdict="explained"), event("C02"),
                                          citations=("B01", "C02")))
    assert promote_character_drift(candidate, result).outcome == "needs_confirmation"


@pytest.mark.parametrize("changed", [dict(coverage="partial"), dict(explanation="partial")])
def test_real_processing_gaps_remain_clues(changed):
    candidate = prepared(**changed)
    result, _ = review(candidate, payload(event("C01")))
    assert promote_character_drift(candidate, result).outcome == "needs_confirmation"


@pytest.mark.parametrize("change", [
    dict(events=[]), dict(events=[event("C02")]),
    dict(events=[event("C01"), event("C01")]),
    dict(events=[event("C01", actuality="unclear")]),
    dict(policy_version=None), dict(event_independence="yes"),
])
def test_malformed_or_unsupported_events_fail_closed(change):
    candidate = prepared()
    result, _ = review(candidate, payload(event("C01"), **change))
    assert result.diagnostics.outcome == "degraded"
    assert promote_character_drift(candidate, result).outcome == "unverifiable"


def test_selected_event_cannot_drop_one_of_its_fragments():
    candidate = prepared(count=2)
    result, _ = review(candidate, payload(event("C01", "C02")))
    assert result.diagnostics.outcome == "degraded"


def test_no_issue_cannot_hide_unresolved_event():
    candidate = prepared(count=2)
    result, _ = review(candidate, payload(event("C01", verdict="explained"),
                                          event("C02", verdict="needs_confirmation"),
                                          verdict="explained", citations=("B01", "C01")))
    assert result.diagnostics.outcome == "degraded"


def test_explicit_small_output_budget_skips_before_model_call():
    candidate = prepared(count=6)
    result, provider = review(candidate, {}, character_drift_max_completion_tokens=1000)
    assert result.diagnostics.reason == "response_contract_budget"
    assert provider.calls == []


def test_legacy_decision_serialization_is_unchanged():
    old = ModelDriftDecision(verdict="contradicts", explanation="原文行为方向相反。",
                             citations=("B01", "C01"))
    assert set(old.model_dump()) == {"verdict", "explanation", "citations"}
    assert prepare_character_drift(prepared().case).review_policy is None


@pytest.mark.parametrize("dimension,key,obj", [
    ("relationship_attitude", "trust_orientation", "守门人"),
    ("motivation_goal", "pursuit_commitment", "寻找姐姐"),
])
def test_relationship_and_long_term_goal_also_allow_one_clear_event(dimension, key, obj):
    case = prepared().case
    base = case.baseline.model_copy(update={"dimension": dimension, "trait_key": key, "key_object": obj})
    observation = case.observations[0].model_copy(update={
        "dimension": dimension, "trait_key": key, "key_object": obj,
    })
    candidate = prepare_character_drift(case.model_copy(update={
        "baseline": base, "observations": (observation,),
    }), review_policy=CHARACTER_REVIEW_POLICY_VERSION)
    result, _ = review(candidate, payload(event("C01")))
    assert candidate.reviewer_eligible
    assert promote_character_drift(candidate, result).outcome == "conflict"


def test_full_six_event_contract_fits_default_budget_without_chopping_candidates():
    candidate = prepared(count=6, same_document=False)
    response = payload(*(event(f"C{i:02d}") for i in range(1, 7)))
    result, provider = review(candidate, response)
    assert len(provider.calls) == 1
    assert len(validated_reviewed_events(candidate, result)) == 6
    assert promote_character_drift(candidate, result).outcome == "conflict"


def test_all_definitely_outside_events_do_not_need_a_causal_explanation():
    candidate = prepared(count=2, explanation="not_run")
    result, _ = review(candidate, payload(
        event("C01", verdict="needs_confirmation", actor_match="different", trait_match="unclear"),
        event("C02", verdict="needs_confirmation", object_match="different"),
        verdict="explained", citations=("B01", "C01", "C02"),
    ))
    assert result.diagnostics.outcome == "completed"
    promoted = promote_character_drift(candidate, result)
    assert promoted.outcome == "no_issue"
    assert promoted.reason == "single_event_outside_comparison"
    assert candidate.case.explanation_coverage == "not_run"


def test_outside_event_still_cannot_bypass_missing_material():
    candidate = prepared(coverage="partial", explanation="not_run")
    result, _ = review(candidate, payload(event("C01", verdict="needs_confirmation", actor_match="different"),
                                          verdict="explained"))
    lead = promote_character_drift(candidate, result)
    assert lead.reason == "material_coverage_incomplete"
    assert "不适合直接与这条设定比较" in lead.explanation and "未核实" in lead.explanation


def test_partial_coverage_lead_explains_only_the_cited_event():
    candidate = prepared(count=2, coverage="partial")
    classroom = "课堂上演的是另一个角色。"
    after_class = "散课后角色当面泄露了受托保密的地址。"
    result, _ = review(candidate, payload(
        event("C01", verdict="explained", situation_match="different", explanation=classroom),
        event("C02", explanation=after_class), citations=("B01", "C02"),
    ))
    lead = promote_character_drift(candidate, result)
    assert lead.outcome == "needs_confirmation" and lead.reason == "material_coverage_incomplete"
    assert after_class in lead.explanation and classroom not in lead.explanation
    assert "待复核" in lead.explanation and "未核实" in lead.explanation


@pytest.mark.parametrize("field", ["actor_match", "trait_match", "actuality",
                                   "object_match", "situation_match"])
def test_an_exception_cannot_turn_uncertain_in_scope_binding_into_no_issue(field):
    candidate = prepared(support=(support_for(1),))
    result, _ = review(candidate, payload(
        event("C01", verdict="explained", **{field: "unclear"}),
        verdict="explained", citations=("B01", "C01", "X01"),
    ))
    assert result.diagnostics.outcome == "degraded"
    assert validated_reviewed_events(candidate, result) == ()
    assert promote_character_drift(candidate, result).outcome == "unverifiable"


def test_medical_exception_with_clear_binding_and_real_support_can_explain():
    from tests.test_character_drift_medical_exception import _case

    candidate = prepare_character_drift(_case(), review_policy=CHARACTER_REVIEW_POLICY_VERSION)
    result, _ = review(candidate, payload(event("C01", verdict="explained"), verdict="explained",
                                          citations=("B01", "C01", "X01")))
    assert result.diagnostics.outcome == "completed"
    assert promote_character_drift(candidate, result).outcome == "no_issue"


def test_in_scope_explanation_still_requires_finished_explanation_review():
    candidate = prepared(support=(support_for(1),), explanation="not_run")
    result, _ = review(candidate, payload(event("C01", verdict="explained"), verdict="explained",
                                          citations=("B01", "C01", "X01")))
    assert result.diagnostics.outcome == "completed"
    assert promote_character_drift(candidate, result).reason == "explanation_search_incomplete"


@pytest.mark.parametrize("exit_kind", ["no_matching_observation", "no_opposition", "scope_unknown"])
def test_server_policy_survives_prepare_early_exits_without_creating_a_review(exit_kind):
    case = prepared().case
    if exit_kind == "no_matching_observation":
        case = case.model_copy(update={"observations": ()})
    elif exit_kind == "no_opposition":
        case = case.model_copy(update={
            "observations": (case.observations[0].model_copy(update={"polarity": "positive"}),),
        })
    else:
        case = case.model_copy(update={"scope_compatibility": "unknown"})
    candidate = prepare_character_drift(case, review_policy=CHARACTER_REVIEW_POLICY_VERSION)
    assert candidate.reason == exit_kind
    assert candidate.review_policy == CHARACTER_REVIEW_POLICY_VERSION
    result, provider = review(candidate, {})
    assert provider.calls == []
    assert validated_reviewed_events(candidate, result) == ()
    assert "review_policy" not in candidate.model_dump()


def test_no_issue_cannot_hide_related_observations_cut_from_bounded_pool():
    candidate = prepared(count=7)
    assert len(candidate.matching_observations) == 6
    assert candidate.unreviewed_observation_count == 1
    assert "unreviewed_observation_count" not in candidate.model_dump()
    result, _ = review(candidate, payload(
        *(event(f"C{i:02d}", verdict="needs_confirmation", actor_match="different") for i in range(1, 7)),
        verdict="explained", citations=("B01", *(f"C{i:02d}" for i in range(1, 7))),
    ))
    assert result.diagnostics.outcome == "completed"
    assert promote_character_drift(candidate, result).reason == "single_event_unreviewed_observations"
    formal_result, _ = review(candidate, payload(*(event(f"C{i:02d}") for i in range(1, 7))))
    assert promote_character_drift(candidate, formal_result).outcome == "conflict"

@pytest.mark.parametrize(("overrides", "category"), [
    ({"events": [event("C02")]}, "partition"),
    ({"policy_version": "private-value"}, "policy"),
    ({"scope_applicability": None}, "policy"),
    ({"unexpected": "private-value"}, "schema"),
    ({"citations": ["B01", "private-value"]}, "citations"),
    ({"events": [event("C01", actor_match="unclear")]}, "comparison"),
    ({"events": [event("C01", verdict="needs_confirmation")]}, "verdict"),
    ({"events": [event("C01", verdict="explained")], "verdict": "explained"}, "explanation"),
])
def test_v3_rejects_with_content_free_contract_categories(overrides, category):
    malformed = payload(event("C01"), **overrides)
    result, provider = review(prepared(), malformed, character_drift_review_max_generations=1)
    assert len(provider.calls) == 1
    assert result.decision is None
    assert result.diagnostics.reason == "invalid_model_response"
    assert result.diagnostics.contract_failure == category
    assert "private-value" not in result.model_dump_json()


@pytest.mark.parametrize("verdict", ["insufficient_evidence", "needs_confirmation"])
def test_valid_abstention_is_never_regenerated(verdict):
    result, provider = review(prepared(), payload(event("C01", verdict=verdict), verdict=verdict))
    assert result.decision.verdict == verdict
    assert len(provider.calls) == 1
    assert result.diagnostics.contract_failure is result.diagnostics.correction_status is None


def test_correction_is_bounded_to_one_sanitized_reassessment_with_same_evidence():
    from app.usage import estimate_issue_evidence_review_tokens

    class Queue:
        calls = []

        def complete(self, system, user):
            self.calls.append((system, user))
            # An injected transport may return unparsed content.
            text = ("private-invalid-json" if len(self.calls) == 1 else json.dumps(
                payload(event("C01", verdict="needs_confirmation"), verdict="needs_confirmation")
            ))
            return SimpleNamespace(text=text, prompt_tokens=20, completion_tokens=7)

    provider = Queue()
    result = CharacterConsistencyReviewer(provider, settings=Settings(
        _env_file=None, enable_character_consistency=True,
    )).review(prepared())
    assert result.decision.verdict == "needs_confirmation"
    assert result.diagnostics.contract_failure == "json"
    assert result.diagnostics.correction_status == "recovered"
    assert result.diagnostics.attempted_calls == len(provider.calls) == 2
    (system, original), (system2, corrected) = provider.calls
    assert system == system2
    assert json.loads(corrected) == {**json.loads(original), "contract_correction": "json"}
    assert "private-invalid-json" not in corrected + result.model_dump_json()
    assert result.diagnostics.prompt_tokens == 40
    assert result.diagnostics.completion_tokens == 14
    assert result.diagnostics.charged_tokens == sum(
        estimate_issue_evidence_review_tokens(s, u, completion_reserve=704) for s, u in provider.calls
    )


def test_undecidable_comparison_is_not_regenerated():
    result, provider = review(prepared(), payload(event("C01", trait_match="unclear")))
    assert result.diagnostics.contract_failure == "comparison"
    assert result.diagnostics.correction_status == "ineligible"
    assert len(provider.calls) == 1


def test_support_role_binding_and_required_context_are_visible_to_reviewer():
    case = prepared(support=(support_for(1),)).case
    candidate = prepare_character_drift(case.model_copy(update={
        "baseline": case.baseline.model_copy(update={"contexts": ("交接委托时",)}),
    }), review_policy=CHARACTER_REVIEW_POLICY_VERSION)
    result, provider = review(candidate, payload(event("C01", verdict="explained"),
        verdict="explained", citations=("B01", "C01", "X01")))
    assert result.diagnostics.outcome == "completed"
    system, user = provider.calls[0]
    assert user["candidate"]["baseline_contexts"] == ["交接委托时"]
    assert user["candidate"]["situation_match_required"] is True
    assert user["candidate"]["object_match_required"] is False
    assert [(r["id"], r["role"]) for r in user["evidence"]] == [
        ("B01", "baseline"), ("C01", "current"), ("X01", "exception"),
    ]
    assert user["evidence"][-1]["applicable_observation_citations"] == ["C01"]
    assert "P (possible_explanation)" in system
    # The exact example must itself pass the real parser and validator.
    sample = next(line for line in system.splitlines() if line.startswith("{"))
    accepted, _ = review(candidate, json.loads(sample))
    assert accepted.diagnostics.outcome == "completed"


def test_unknown_provider_categories_and_oversized_output_are_not_regenerated():
    for response, reason in [
        (ProviderError("private-exception", category="private-category"), "provider_error"),
        ("private-response" * 5000, "response_too_large"),
    ]:
        result, provider = review(prepared(), response)
        assert result.diagnostics.reason == reason
        assert result.diagnostics.contract_failure is result.diagnostics.correction_status is None
        assert len(provider.calls) == 1
        assert "private-" not in result.model_dump_json()


def test_correction_provider_setup_failure_keeps_initial_charge_and_clue():
    class FailingCorrectionFork(Provider):
        def fork_for_character_consistency(self, *, remaining_deadline_seconds=None, **kwargs):
            if remaining_deadline_seconds is not None:
                raise RuntimeError("private-setup-error")
            return self

    provider = FailingCorrectionFork({"unexpected": "field"})
    settings = Settings(_env_file=None, enable_character_consistency=True)
    case = prepared()
    result = CharacterConsistencyReviewer(provider, settings=settings).review(case)
    assert result.diagnostics.correction_status == "provider_error"
    assert result.diagnostics.reason == "invalid_model_response"
    assert result.diagnostics.attempted_calls == len(provider.calls) == 1
    assert result.diagnostics.charged_tokens >= 40
    assert promote_character_drift(case, result).outcome == "unverifiable"
    assert "private-setup-error" not in result.model_dump_json()
