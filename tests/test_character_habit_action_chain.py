"""The habit verifier certifies the actor of each action in a bounded sentence."""
import json

import pytest

from app.character_draft_actor_review import (
    PrimaryDraftClaim, TARGET_BOUND_DRAFT_REVIEW_PROMPT_V8, TARGET_BOUND_DRAFT_REVIEW_PROMPT_V12,
    TARGET_BOUND_DRAFT_REVIEW_PROMPT_V13, TARGET_BOUND_DRAFT_REVIEW_PROMPT_V14, build_target_bound_draft_proposals,
    build_target_bound_draft_review_request, evaluate_target_bound_draft_review,
    target_bound_draft_review_request_digest, verify_target_bound_draft_review_source,
)
from app.character_draft_actor_review_provider import (
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V12, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13,
    build_target_bound_draft_review_prompts,
)
from tests.test_character_target_bound_draft_review_v4 import _source, _target, _item


def habit_request(text, *, claim=None, historical=False):
    target = _target(character="舟南", dimension="contextual_behavior", trait_key="封盖",
                     comparison_key="contextual_behavior:封盖", key_object="", habit_contexts=("装箱时",))
    source = _source(text)
    index, proposals = build_target_bound_draft_proposals(
        text, source, target, line_start=1, line_end=1, protocol_version="v4",
        primary_claim_review=claim is not None, allow_habit_chain=not historical,
    )
    request = build_target_bound_draft_review_request(
        index, target, proposals, frozen_content=text, expected_source=source,
        protocol_version="v4", primary_claim=claim, allow_habit_chain=not historical,
    )
    return source, request


def response_items(request):
    _, user = build_target_bound_draft_review_prompts((request,))
    hints = json.loads(user.split("：\n", 1)[1])["windows"][0]["basis_path_hints"]
    result = []
    for p, hint in zip(request.proposals, hints, strict=True):
        item = _item(request, p, object_relation="not_applicable", observation_kind="action")
        if "action_path" in hint:
            item["action_steps"] = [
                dict(clause_id=step["clause_id"], quote=step["text"], actor="target",
                     actuality="actual", scene="same") for step in hint["action_path"]
            ]
        if request.primary_claim:
            item.update(claim_digest=request.primary_claim.claim_digest, claim_relation="equivalent")
        result.append(item)
    return result


SENTENCE = "舟南打开木箱，逐件收进瓷器，却故意把箱盖搁在桌边。"


@pytest.mark.parametrize("with_claim", [False, True])
def test_ordered_chain_is_source_bound_for_recall_and_claim_recovery(with_claim):
    claim = PrimaryDraftClaim(package_attempt=1, record_ordinal=1, statement="舟南把箱盖搁在桌边",
                              claim_digest="7"*64) if with_claim else None
    source, req = habit_request(SENTENCE, claim=claim)
    assert req.prompt_version == (TARGET_BOUND_DRAFT_REVIEW_PROMPT_V13 if claim else TARGET_BOUND_DRAFT_REVIEW_PROMPT_V12)
    system, _ = build_target_bound_draft_review_prompts((req,))
    assert system == (TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13 if claim else TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V12)
    assert verify_target_bound_draft_review_source(req, source, frozen_content=SENTENCE, expected_primary_claim=claim)
    assert not verify_target_bound_draft_review_source(req, source, frozen_content=SENTENCE.replace("舟南", "叶驰"), expected_primary_claim=claim)
    items = response_items(req)
    third = next(item for item in items if len(item.get("action_steps", [])) == 3)
    third_id = third["proposal_id"]
    def decide(changed=third):
        raw = {"schema_version":req.schema_version,"request_digest":target_bound_draft_review_request_digest(req),
               "items":[changed if item is third else item for item in items]}
        evaluation = evaluate_target_bound_draft_review(req, json.dumps(raw), expected_source=source,
                        frozen_content=SENTENCE, expected_primary_claim=claim)
        return next(d for d in evaluation.decisions if d.proposal_id==third_id)
    assert decide().verdict=="supported"
    a,b,c = third["action_steps"]
    for wrong in ([], [a,c], [a,c,b], [a,b,b], [a,{**b, "actor":"other"},c],
                  [a,{**b, "actuality":"non_actual"},c], [a,b,{**c,"scene":"changed"}],
                  [a,b,{**c,"quote":b["quote"]}], [{**a,"quote":"打开木箱"},b,c]):
        assert decide({**third,"action_steps":wrong}).verdict == "uncertain"
    if claim:
        assert decide({**third,"claim_relation":"different"}).verdict != "supported"


@pytest.mark.parametrize("text",[
    "舟南打开木箱；逐件收进瓷器，却故意把箱盖搁在桌边。",
    "舟南打开木箱。逐件收进瓷器，却故意把箱盖搁在桌边。",
    "舟南打开木箱，叶驰接过木箱，却故意把箱盖搁在桌边。",
    "舟南打开木箱，次日回到学校，却故意把箱盖搁在桌边。",
    "舟南打开木箱，随后她接过木箱，却故意把箱盖搁在桌边。",
])
def test_source_boundaries_stop_a_habit_chain(text):
    _, request = habit_request(text)
    assert not any(p.binding_kind=="habit_action_chain" for p in request.proposals)


def test_other_domain_opens_a_chain_from_manner_then_a_disposal_construction():
    text = "舟南亲自收货，顺手拆开包裹，又将封条丢弃。"
    source, request = habit_request(text)
    items = response_items(request)
    last = next(item for item in items if len(item.get("action_steps", [])) == 3)
    assert [step["quote"] for step in last["action_steps"]] == ["舟南亲自收货", "顺手拆开包裹", "又将封条丢弃"]
    raw = {"schema_version": request.schema_version, "request_digest": target_bound_draft_review_request_digest(request),
           "items": items}
    result = evaluate_target_bound_draft_review(request, json.dumps(raw), expected_source=source, frozen_content=text)
    assert next(x for x in result.decisions if x.proposal_id == last["proposal_id"]).verdict == "supported"


@pytest.mark.parametrize("text", [
    "舟南打开木箱，假如有人进屋，便把箱盖搁在桌边。",
    "舟南打开木箱，喊道：“把箱盖搁在桌边。”",
    "舟南打开木箱，故意把箱盖搁在桌边吗？",
])
def test_non_narrative_windows_do_not_propose_a_habit_action(text):
    target = _target(character="舟南", dimension="contextual_behavior", trait_key="关箱",
                     comparison_key="contextual_behavior:关箱", key_object="", habit_contexts=("交接时",))
    _, proposals = build_target_bound_draft_proposals(text, _source(text), target,
                                                      line_start=1, line_end=1, protocol_version="v4")
    assert not proposals


def test_legacy_habit_request_reproduces_its_original_bytes_and_ordinary_prompt():
    source, req = habit_request(SENTENCE, historical=True)
    assert req.prompt_version == TARGET_BOUND_DRAFT_REVIEW_PROMPT_V8
    assert all(p.binding_kind != "habit_action_chain" for p in req.proposals)
    assert verify_target_bound_draft_review_source(req, source, frozen_content=SENTENCE)


def test_v14_requires_all_steps_and_grounding_in_the_same_fact_for_the_expected_record():
    claim = PrimaryDraftClaim(package_attempt=1, record_ordinal=2, statement="舟南将箱盖放在桌边",
                              key_object="箱盖", claim_digest="9"*64)
    source, req = habit_request(SENTENCE, claim=claim)
    assert req.prompt_version == TARGET_BOUND_DRAFT_REVIEW_PROMPT_V14
    assert not verify_target_bound_draft_review_source(req, source, frozen_content=SENTENCE,
                        expected_primary_claim=claim.model_copy(update={"record_ordinal": 3}))
    items = response_items(req)
    tail = next(item for item in items if len(item.get("action_steps", [])) == 3)
    obj = dict(role="event_object", grounding="explicit", referent_relation="same", fact_quote="箱盖",
               antecedent_clause_id=None, antecedent_quote=None)
    items = [{**item, "claim_object":obj} for item in items]

    def verdict(**changes):
        raw = {"schema_version": req.schema_version, "request_digest":target_bound_draft_review_request_digest(req),
               "items":[{**item, **changes} if item["proposal_id"] == tail["proposal_id"] else item for item in items]}
        ev = evaluate_target_bound_draft_review(req, json.dumps(raw), expected_source=source,
                    frozen_content=SENTENCE, expected_primary_claim=claim)
        return next(item.verdict for item in ev.decisions if item.proposal_id == tail["proposal_id"])

    assert verdict() == "supported"
    for changes in ({"claim_object":None}, {"claim_object":{**obj,"role":"axis_label"}},
                    {"claim_object":{**obj,"referent_relation":None}}, {"claim_object":{**obj,"fact_quote":"瓷器"}},
                    {"claim_object":{**obj,"fact_quote":"盖"}}, {"claim_digest":"6"*64},
                    {"action_steps":tail["action_steps"][:-1]}, {"polarity_relation":"opposite"}):
        assert verdict(**changes) != "supported"


@pytest.mark.parametrize("separator,expected", [("，", True), ("；", False), ("。", False)])
def test_pronoun_object_needs_an_antecedent_in_the_same_sentence(separator, expected):
    from app.character_draft_actor_review import ClaimObjectGrounding, valid_primary_object_grounding

    text = f"舟南打开木箱{separator}舟南将它搬出库房。"
    claim = PrimaryDraftClaim(package_attempt=1, record_ordinal=1, statement="舟南将木箱搬出库房",
                              key_object="木箱", claim_digest="9"*64)
    _, req = habit_request(text, claim=claim)
    proposal = next(p for p in req.proposals if "搬出" in p.canonical_statement)
    obj = ClaimObjectGrounding(role="event_object", grounding="unique_reference", referent_relation="same",
                               fact_quote="它", antecedent_clause_id=req.lines[0].clauses[0].support_id,
                               antecedent_quote="木箱")
    assert valid_primary_object_grounding(req, proposal, obj) is expected
