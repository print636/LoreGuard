"""Fixed responses exercise source binding on the production single-event policy."""
import json

import pytest

from app.character_drift import (
    CHARACTER_HABIT_CONTEXT_REVIEW_SYSTEM_PROMPT, CHARACTER_HABIT_CONTEXT_SCHEMA_VERSION,
    CHARACTER_REVIEW_POLICY_VERSION, CharacterConsistencyReviewer,
    prepare_character_drift, promote_character_drift,
)
from tests.test_character_runtime_provider_routing import _http_accounting, _response_http, _settings
from tests.test_character_single_event_policy import event, payload, prepared, review, span
from tests.test_habit_claim_recovery import HABIT, SOURCE, extract, record


def habit(*texts, contexts=("封箱交接期间",), same_document=True):
    case = prepared(count=len(texts), texts=texts, same_document=same_document).case
    statement = "沈遥给每一只运箱贴编号"
    baseline = case.baseline.model_copy(update={
        "dimension": "contextual_behavior", "trait_key": "crate_numbering",
        "statement": statement, "contexts": contexts,
        "key_object": "", "evidence": (span(f"封箱交接期间，{statement}。", document="profile"),),
    })
    observations = tuple(row.model_copy(update={
        "dimension": baseline.dimension, "trait_key": baseline.trait_key,
        # A misleading model summary cannot substitute for the frozen C text.
        "context": "封箱交接期间",
    }) for row in case.observations)
    return prepare_character_drift(case.model_copy(update={
        "baseline": baseline, "observations": observations,
    }), review_policy=CHARACTER_REVIEW_POLICY_VERSION)


def check(citation, relation, quote="", **extra):
    return dict(citation=citation, relation=relation, quote=quote, **extra)


def test_paraphrased_habit_scope_uses_frozen_text_and_normal_accounting():
    candidate = habit("沈遥把箱盖钉好交给货车司机，没给运箱贴号。")
    response = payload(event("C01", context_checks=[check("C01", "same", "把箱盖钉好交给货车司机")]))
    settings = _settings()
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return _response_http(json.dumps(response))

    accounting, usage = _http_accounting(settings, respond)
    result = CharacterConsistencyReviewer(accounting, settings=settings).review(candidate)
    assert usage.logical_calls == len(requests) == 1
    system, user = requests[0]["messages"]
    assert system["content"] == CHARACTER_HABIT_CONTEXT_REVIEW_SYSTEM_PROMPT
    request = json.loads(user["content"])
    assert request["candidate"]["habit_context_schema_version"] == CHARACTER_HABIT_CONTEXT_SCHEMA_VERSION
    assert request["candidate"]["baseline_contexts"] == ["封箱交接期间"]
    assert promote_character_drift(candidate, result).outcome == "conflict"


@pytest.mark.parametrize(("relation", "quotation", "verdict", "outcome"), [
    ("different", "私人收藏的旧箱", "explained", "no_issue"),
    ("unclear", "", "needs_confirmation", "needs_confirmation"),
])
def test_other_or_unclear_situation_does_not_become_habit_conflict(relation, quotation, verdict, outcome):
    candidate = habit("沈遥给私人收藏的旧箱重新上漆，没有贴编号。")
    response = payload(event("C01", verdict=verdict, situation_match=relation,
                             context_checks=[check("C01", relation, quotation)]), verdict=verdict)
    result, _ = review(candidate, response)
    assert result.diagnostics.outcome == "completed"
    assert promote_character_drift(candidate, result).outcome == outcome


def test_an_explicit_habit_object_must_be_compared_even_in_the_same_situation():
    candidate = habit("沈遥把运箱交给司机，包里的练习卡没有贴号。")
    baseline = candidate.case.baseline.model_copy(update={"key_object": "运箱"})
    case = candidate.case.model_copy(update={"baseline": baseline})
    candidate = prepare_character_drift(case, review_policy=CHARACTER_REVIEW_POLICY_VERSION)
    checks = [check("C01", "same", "把运箱交给司机")]
    unexamined, provider = review(candidate, payload(event("C01", object_match="not_applicable", context_checks=checks)))
    request = provider.calls[0][1]
    assert request["candidate"]["object_match_required"] is True
    assert request["candidate"]["key_object"] == "运箱"
    assert next(row for row in request["evidence"] if row["id"] == "B01")["key_object"] == "运箱"
    assert unexamined.diagnostics.contract_failure == "comparison"

    outside, _ = review(candidate, payload(event("C01", verdict="explained", object_match="different",
                                                 context_checks=checks), verdict="explained"))
    assert promote_character_drift(candidate, outside).outcome == "no_issue"


def test_final_reviewer_receives_the_object_grounded_in_the_draft_action():
    extracted, _ = extract([dict(record("廖辰把行李标签装入口袋"), key_object="行李标签")], False)
    case = habit(SOURCE, contexts=HABIT.habit_contexts).case
    baseline = case.baseline.model_copy(update={
        "character": HABIT.character, "trait_key": HABIT.trait_key, "statement": HABIT.baseline_hint,
        "evidence": (span(HABIT.baseline_hint, document="profile"),),
    })
    candidate = prepare_character_drift(
        case.model_copy(update={"baseline": baseline, "observations": extracted.draft_observations}),
        review_policy=CHARACTER_REVIEW_POLICY_VERSION,
    )
    _, provider = review(candidate, payload(event("C01", context_checks=[check("C01", "same", "接过客人的手提箱")])))
    draft = next(row for row in provider.calls[0][1]["evidence"] if row["id"] == "C01")
    assert draft["key_object"] == "行李牌"
    assert draft["key_object"] in draft["text"]


def test_one_in_scope_event_does_not_vouch_for_an_unrelated_observation():
    candidate = habit("沈遥封好箱盖向司机交货，漏贴编号。",
                     "次日沈遥在家修补自己的旧箱，也没贴编号。")
    valid = check("C01", "same", "封好箱盖向司机交货")
    wrong_scope = check("C02", "different", "在家修补自己的旧箱")
    for checks in ([valid], [valid, wrong_scope], [valid, check("C02", "same", valid["quote"])]):
        response = payload(event("C01", "C02", context_checks=checks), citations=("B01", "C01", "C02"))
        result, _ = review(candidate, response)
        assert result.diagnostics.contract_failure == "comparison"
        assert promote_character_drift(candidate, result).outcome == "unverifiable"
    response = payload(event("C01", context_checks=[valid]),
                       event("C02", verdict="explained", situation_match="different", context_checks=[wrong_scope]))
    result, _ = review(candidate, response)
    assert promote_character_drift(candidate, result).outcome == "conflict"
    assert result.decision.citations == ("B01", "C01")


@pytest.mark.parametrize("same_document", [True, False])
def test_a_continuation_needs_its_own_quote_and_an_adjacent_direct_anchor(same_document):
    candidate = habit("沈遥合上头一只箱子的盖，直接交给承运人。",
                     "她随即将另一只也递了过去，两只箱上都没编号。",
                     same_document=same_document)
    checks = [check("C01", "same", "直接交给承运人"),
              check("C02", "same", "随即将另一只也递了过去", continues_from="C01")]
    response = payload(event("C01", "C02", context_checks=checks), citations=("B01", "C01", "C02"))
    result, _ = review(candidate, response)
    assert promote_character_drift(candidate, result).outcome == (
        "conflict" if same_document else "unverifiable")


def test_missing_confirmed_context_cannot_be_treated_as_unrestricted():
    candidate = habit("沈遥跳过了编号。", contexts=())
    for relation, checks in [("not_applicable", None), ("same", [check("C01", "same", "跳过了编号")])]:
        response = payload(event("C01", situation_match=relation, context_checks=checks))
        result, _ = review(candidate, response)
        assert result.diagnostics.contract_failure == "comparison"
        assert promote_character_drift(candidate, result).outcome == "unverifiable"
    response = payload(event("C01", verdict="needs_confirmation", situation_match="unclear",
                             context_checks=[check("C01", "unclear")]), verdict="needs_confirmation")
    result, _ = review(candidate, response)
    assert promote_character_drift(candidate, result).outcome == "needs_confirmation"


def test_unrelated_dimension_does_not_accept_the_habit_contract():
    candidate = prepared()
    response = payload(event("C01", context_checks=[check("C01", "same", "沈遥")]))
    result, _ = review(candidate, response)
    assert result.diagnostics.contract_failure == "policy"
