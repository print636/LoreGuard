"""V15 asks about a source-bound omission; it never infers conflict from negation."""
import json
from types import SimpleNamespace

import pytest

from app.character_draft_actor_review import (
    PrimaryDraftClaim, TARGET_BOUND_DRAFT_REVIEW_PROMPT_V14, TARGET_BOUND_DRAFT_REVIEW_PROMPT_V15,
    build_target_bound_draft_proposals, build_target_bound_draft_review_request,
    evaluate_target_bound_draft_review, target_bound_draft_review_request_digest,
    verify_target_bound_draft_review_source,
)
from app.character_draft_actor_review_provider import (
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V14, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15,
    TargetBoundDraftReviewBatchEntry, build_target_bound_draft_review_prompts,
    habit_object_chain_completion_reserve, run_target_bound_draft_review,
)
from app.character_trait_extraction import CharacterSignalChunk, CharacterSignalExtractor
from tests.test_character_target_bound_draft_review_v4 import _item, _source, _target
from tests.test_character_target_bound_draft_review_v3 import _settings
from tests.test_habit_claim_recovery import HABIT, record


PREFIX = "廖辰接过器材箱，把它们摆上货台，"
TAIL = "没逐件清点就装车发运"
CONTENT = PREFIX + TAIL + "。"


def make_request(text=CONTENT, *, enabled=True):
    target = _target(character="廖辰", dimension="contextual_behavior", trait_key="inventory",
                     comparison_key="contextual_behavior:inventory", key_object="", habit_contexts=("发货前",))
    source = _source(text)
    claim = PrimaryDraftClaim(package_attempt=1, record_ordinal=1, claim_digest="a" * 64,
                              statement="廖辰未清点器材箱便发货", key_object="器材箱")
    index, proposals = build_target_bound_draft_proposals(
        text, source, target, line_start=1, line_end=1, protocol_version="v4",
        primary_claim_review=True, allow_object_chain=enabled,
    )
    request = build_target_bound_draft_review_request(
        index, target, proposals, frozen_content=text, expected_source=source,
        protocol_version="v4", primary_claim=claim, allow_object_chain=enabled,
    )
    assert verify_target_bound_draft_review_source(request, source, frozen_content=text, expected_primary_claim=claim)
    return source, request


def object_path(path):
    negated = path[-1]["text"].split("就")[0].split("便")[0]
    return dict(
        role="event_object", grounding="action_path", referent_relation="same", fact_quote=negated,
        antecedent_clause_id=path[0]["clause_id"], antecedent_quote="器材箱",
        object_steps=[
            dict(clause_id=step["clause_id"], quote="器材箱" if i == 0 else negated if i == len(path) - 1 else "它们",
                 grounding="explicit" if i == 0 else "omitted" if i == len(path) - 1 else "reference",
                 referent_relation="same") for i, step in enumerate(path)
        ],
    )


def response_items(request, *, tail=TAIL):
    _, user = build_target_bound_draft_review_prompts((request,))
    hints = json.loads(user.split("：\n", 1)[1])["windows"][0]["basis_path_hints"]
    items = []
    for proposal, hint in zip(request.proposals, hints, strict=True):
        supported = proposal.canonical_statement == "廖辰" + tail
        item = _item(request, proposal, object_relation="not_applicable", observation_kind="action")
        item.update(verdict="supported" if supported else "rejected",
                    claim_digest=request.primary_claim.claim_digest,
                    claim_relation="equivalent" if supported else "different", claim_object=None)
        if path := hint.get("action_path"):
            item["action_steps"] = [dict(clause_id=step["clause_id"], quote=step["text"], actor="target",
                                         actuality="actual", scene="same") for step in path]
            if supported:
                item["claim_object"] = object_path(path)
        items.append(item)
    return items


def evaluate(request, source, items, text=CONTENT):
    return evaluate_target_bound_draft_review(
        request, json.dumps(dict(schema_version=request.schema_version,
                                request_digest=target_bound_draft_review_request_digest(request), items=items)),
        expected_source=source, frozen_content=text, expected_primary_claim=request.primary_claim,
    )


@pytest.mark.parametrize("tail", [TAIL, "未登记便直接交付", "并未核验就完成交接"])
@pytest.mark.parametrize("prefix", [PREFIX, "廖辰接过器材箱，"])
def test_generic_negative_continuations_enter_one_complete_versioned_request(prefix, tail):
    text = prefix + tail + "。"
    source, request = make_request(text)
    assert request.prompt_version == TARGET_BOUND_DRAFT_REVIEW_PROMPT_V15
    assert len(request.proposals) == (3 if prefix == PREFIX else 2)
    assert any(p.canonical_statement == "廖辰" + tail for p in request.proposals)
    assert build_target_bound_draft_review_prompts((request,))[0] == TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15
    result = evaluate(request, source, response_items(request, tail=tail), text)
    assert [d.verdict for d in result.decisions].count("supported") == 1


@pytest.mark.parametrize("text", [
    "廖辰接过器材箱；没逐件清点就装车发运。",
    "廖辰接过器材箱。没逐件清点就装车发运。",
    "廖辰接过器材箱，没逐件清点就他把货物装上车。",
    "廖辰接过器材箱，没逐件清点就没有装车发运。",
    "廖辰接过器材箱，如果没逐件清点就装车发运。",
    "廖辰说：“没逐件清点就装车发运。”",
])
def test_source_and_subject_boundaries_do_not_create_a_negative_chain(text):
    target = _target(character="廖辰", dimension="contextual_behavior", trait_key="inventory",
                     comparison_key="contextual_behavior:inventory", key_object="", habit_contexts=("发货前",))
    _, proposals = build_target_bound_draft_proposals(
        text, _source(text), target, line_start=1, line_end=1,
        protocol_version="v4", primary_claim_review=True, allow_object_chain=True,
    )
    assert not any(p.binding_kind == "habit_action_chain" for p in proposals)


@pytest.mark.parametrize("damage", ["object_unknown", "object_different", "wrong_quote", "wrong_order",
                                     "missing_step", "wrong_object", "wrong_fact", "actor", "actuality", "scene"])
def test_each_object_and_actor_step_must_be_source_bound_and_semantically_confirmed(damage):
    source, request = make_request()
    items = response_items(request)
    item = next(item for item in items if item["verdict"] == "supported")
    steps = item["claim_object"]["object_steps"]
    if damage in {"object_unknown", "object_different"}:
        steps[1]["referent_relation"] = "uncertain" if damage == "object_unknown" else "different"
    elif damage == "wrong_quote":
        steps[1]["quote"] = "货台"
    elif damage == "wrong_order":
        steps[0], steps[1] = steps[1], steps[0]
    elif damage == "missing_step":
        steps.pop(1)
    elif damage == "wrong_object":
        item["claim_object"]["antecedent_quote"] = "货台"
    elif damage == "wrong_fact":
        steps[-1]["quote"] = item["claim_object"]["fact_quote"] = "装车发运"
    else:
        item["action_steps"][1][damage] = {"actor": "other", "actuality": "non_actual", "scene": "changed"}[damage]
    assert all(d.verdict != "supported" for d in evaluate(request, source, items).decisions)


def test_multiple_source_objects_do_not_form_a_unique_antecedent():
    text = CONTENT.replace("接过器材箱", "接过器材箱和工具袋")
    source, request = make_request(text)
    assert all(d.verdict != "supported" for d in evaluate(request, source, response_items(request), text).decisions)


def qualified_object_certificate(phrase, raw="器材箱", *, omitted=True, quote=None):
    text = CONTENT.replace("接过器材箱", "接过" + phrase)
    source, request = make_request(text)
    claim = request.primary_claim.model_copy(update={
        "key_object": raw,
        "statement": TAIL if omitted else "廖辰未清点" + raw + "便发货",
    })
    request = request.model_copy(update={"primary_claim": claim})
    assert verify_target_bound_draft_review_source(request, source, frozen_content=text, expected_primary_claim=claim)
    items = response_items(request)
    item = next(item for item in items if item["verdict"] == "supported")
    obj = item["claim_object"]
    obj["antecedent_quote"] = obj["object_steps"][0]["quote"] = quote or phrase
    return source, request, items, text


@pytest.mark.parametrize("omitted", [False, True])
@pytest.mark.parametrize("phrase", ["器材箱", "新送来的器材箱", "三件器材箱"])
def test_complete_object_phrase_does_not_require_repeating_object_in_claim_statement(phrase, omitted):
    source, request, items, text = qualified_object_certificate(phrase, omitted=omitted)
    evaluated = evaluate(request, source, items, text)
    supported = [d for d in evaluated.decisions if d.verdict == "supported"]
    assert len(supported) == 1
    assert supported[0].verified_object.antecedent_quote == phrase
    assert request.primary_claim.statement == (TAIL if omitted else "廖辰未清点器材箱便发货")


@pytest.mark.parametrize("omitted", [False, True])
@pytest.mark.parametrize("phrase,raw,quote", [
    ("器材箱的标签", "器材箱", None),
    ("器材箱包装", "器材箱", None),
    ("器材箱的标签", "器材箱", "器材箱"),
    ("工具箱", "器材箱", None),
    ("器材箱", "新送来的器材箱", None),
    ("新送来的器材箱", "旧器材箱", None),
])
def test_object_head_guard_rejects_possessions_changed_objects_and_lost_qualifiers(phrase, raw, quote, omitted):
    source, request, items, text = qualified_object_certificate(phrase, raw, omitted=omitted, quote=quote)
    assert all(d.verdict != "supported" for d in evaluate(request, source, items, text).decisions)


def test_qualified_object_with_unconfirmed_middle_step_still_cannot_be_supported():
    source, request, items, text = qualified_object_certificate("新送来的器材箱")
    item = next(item for item in items if item["verdict"] == "supported")
    item["claim_object"]["object_steps"][1]["referent_relation"] = "uncertain"
    assert all(d.verdict != "supported" for d in evaluate(request, source, items, text).decisions)


def test_old_request_and_response_never_gain_the_new_semantics():
    source, old = make_request(enabled=False)
    _, new = make_request()
    assert old.prompt_version == TARGET_BOUND_DRAFT_REVIEW_PROMPT_V14
    assert len(old.proposals) == 2 and len(new.proposals) == 3
    assert build_target_bound_draft_review_prompts((old,))[0] == TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V14
    assert target_bound_draft_review_request_digest(old) != target_bound_draft_review_request_digest(new)
    assert all(d.verdict == "rejected" for d in evaluate(old, source, response_items(old)).decisions)
    injected = response_items(old)
    injected[0]["claim_object"] = next(i["claim_object"] for i in response_items(new) if i["verdict"] == "supported")
    assert all(d.reason == "response_mismatch" for d in evaluate(old, source, injected).decisions)
    # A V14-style missing-path certificate cannot certify a V15 omission either.
    incomplete = response_items(new)
    del next(i["claim_object"] for i in incomplete if i["verdict"] == "supported")["object_steps"]
    assert all(d.verdict != "supported" for d in evaluate(new, source, incomplete).decisions)


class ChainProvider:
    def __init__(self):
        self.requests = []
        self.generations = 0

    def complete(self, system, user):
        if system not in {TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V14, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15}:
            self.generations += 1
            output = {"records": [dict(record("廖辰未清点器材箱便发货"), key_object="器材箱", evidence=CONTENT)]}
        else:
            from app.character_draft_actor_review import TargetBoundDraftReviewRequest
            batch = json.loads(user.split("：\n", 1)[1])
            responses = []
            for window in batch["windows"]:
                request = TargetBoundDraftReviewRequest.model_validate_json(json.dumps(window["request"]))
                self.requests.append(request)
                responses.append(dict(schema_version=request.schema_version, request_digest=window["request_digest"],
                                      items=response_items(request)))
            output = dict(schema_version="character-target-bound-draft-review-batch-v4",
                          batch_digest=batch["batch_digest"], responses=responses)
        return SimpleNamespace(text=json.dumps(output, ensure_ascii=False), prompt_tokens=9, completion_tokens=8)


@pytest.mark.parametrize("targeted", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_frozen_policy_reaches_existing_claim_review_and_source_bound_observation(targeted, enabled):
    settings = _settings(character_signal_package_max_attempts=1, character_signal_draft_trace_v1=True)
    settings._character_habit_object_chain_v1 = enabled
    provider = ChainProvider()
    extractor = CharacterSignalExtractor(provider, settings=settings)
    chunk = CharacterSignalChunk("draft-v4", "draft.md", CONTENT, 1, "draft")
    kwargs = dict(source_identity=_source(CONTENT), frozen_content=CONTENT)
    if targeted:
        result = extractor.extract_targeted(chunk, (HABIT,), target_ordinal=1, **kwargs)
    else:
        result = extractor.extract(chunk, primary_value_targets=(HABIT,), **kwargs)
    request, = provider.requests
    assert provider.generations == 1
    assert request.prompt_version == (TARGET_BOUND_DRAFT_REVIEW_PROMPT_V15 if enabled else TARGET_BOUND_DRAFT_REVIEW_PROMPT_V14)
    assert len(request.proposals) == (3 if enabled else 2)
    assert result.diagnostics.reason_counts["habit_claim_review_calls"] == 1
    if enabled:
        observation, = result.draft_observations
        assert observation.statement == "廖辰" + TAIL and observation.key_object == "器材箱"
        assert observation.evidence.text == CONTENT
        assert result.diagnostics.outcome == "completed"
    else:
        assert not result.draft_observations and result.diagnostics.outcome == "partial"


def test_v15_response_reserve_is_bounded_and_debited_before_the_single_call():
    from app.usage import estimate_issue_evidence_review_tokens

    source, request = make_request()
    _, legacy = make_request(enabled=False)
    reserve = habit_object_chain_completion_reserve((request,), 2048)
    assert reserve == 3328
    assert habit_object_chain_completion_reserve((legacy,), 2048) == 2048
    assert habit_object_chain_completion_reserve((request,), 8192) == 4096
    entry = TargetBoundDraftReviewBatchEntry(request=request, expected_source=source,
                                            frozen_content=CONTENT, expected_primary_claim=request.primary_claim)

    class BoundedProvider(ChainProvider):
        def __init__(self):
            super().__init__()
            self.limits = []

        def fork_for_character_draft_actor_review(self, **limits):
            self.limits.append(limits)
            return self

    provider = BoundedProvider()
    kwargs = dict(provider=provider, completion_reserve=2048, timeout_seconds=30,
                  remaining_deadline_seconds=30, max_response_bytes=32768, max_attempts=1, monotonic=lambda: 0)
    system, user = build_target_bound_draft_review_prompts((request,))
    estimate = estimate_issue_evidence_review_tokens(system, user, completion_reserve=reserve)
    denied = run_target_bound_draft_review((entry,), token_budget=estimate - 1, **kwargs)
    assert denied.failure_reason == "token_budget" and denied.attempted_calls == 0
    assert not provider.limits and not provider.requests
    admitted = run_target_bound_draft_review((entry,), token_budget=estimate, **kwargs)
    assert admitted.attempted_calls == len(provider.requests) == len(provider.limits) == 1
    assert provider.limits[0]["completion_reserve"] == reserve
    assert admitted.charged_tokens == admitted.estimated_tokens == estimate
