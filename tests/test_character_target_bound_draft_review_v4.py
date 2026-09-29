from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from app.character_draft_actor_review import (
    TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V2,
    TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V3,
    TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V4,
    TARGET_BOUND_DRAFT_REVIEW_PROMPT_V8,
    TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V2,
    TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3,
    TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4,
    TargetBoundDraftReviewTarget,
    build_target_bound_draft_proposals,
    build_target_bound_draft_review_request,
    evaluate_target_bound_draft_review,
    evaluate_target_bound_draft_review_batch,
    required_target_bound_draft_basis_ids,
    target_bound_draft_review_batch_digest,
    target_bound_draft_review_request_digest,
)
from app.character_draft_actor_review_provider import (
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V8,
    TargetBoundDraftReviewBatchEntry,
    build_target_bound_draft_review_prompts,
    run_target_bound_draft_review,
)
from app.character_scope_review import ScopeReviewSourceIdentity


def _source(content: str, suffix: str = "v4") -> ScopeReviewSourceIdentity:
    return ScopeReviewSourceIdentity(
        run_input_id=f"run-input-{suffix}",
        document_id=f"draft-{suffix}",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _target(**updates) -> TargetBoundDraftReviewTarget:
    values = {
        "target_ordinal": 1,
        "character": "余霁",
        "dimension": "relationship_attitude",
        "trait_key": "trust_orientation",
        "comparison_key": "relationship_attitude:trust_orientation:周岚",
        "key_object": "周岚",
        "requested_polarity": "negative",
        "baseline_hint": "余霁长期信任周岚",
    }
    values.update(updates)
    return TargetBoundDraftReviewTarget(**values)


def _request(
    content: str,
    *,
    target: TargetBoundDraftReviewTarget | None = None,
    protocol: str = "v4",
    suffix: str = "v4",
):
    source = _source(content, suffix)
    target = target or _target()
    index, proposals = build_target_bound_draft_proposals(
        content,
        source,
        target,
        line_start=1,
        line_end=1,
        protocol_version=protocol,
    )
    assert proposals
    request = build_target_bound_draft_review_request(
        index,
        target,
        proposals,
        frozen_content=content,
        expected_source=source,
        protocol_version=protocol,
    )
    return source, request


def _item(request, proposal, **updates):
    item = {
        "proposal_id": proposal.proposal_id,
        "verdict": "supported",
        "actor": "proposed",
        "actuality": "asserted",
        "statement_relation": "supported",
        "axis_relation": (
            "matches_scoped_axis"
            if request.target.approved_axis_comparison_key is not None
            else "matches_target"
        ),
        "object_relation": (
            "not_applicable"
            if request.target.approved_axis_comparison_key is not None
            else "matches_target"
        ),
        "polarity_relation": "requested",
        "correction_relation": "none",
        "observation_kind": "interaction",
        "object_start_offset": None,
        "object_end_offset": None,
        "basis_ids": list(required_target_bound_draft_basis_ids(request)),
    }
    item.update(updates)
    return item


def _response(request, *, item_updates=None):
    items = [
        _item(request, proposal, **(item_updates or {}))
        for proposal in request.proposals
    ]
    return json.dumps(
        {
            "schema_version": request.schema_version,
            "request_digest": target_bound_draft_review_request_digest(request),
            "items": items,
        },
        ensure_ascii=False,
    )


def _batch_response(request, response: str) -> str:
    batch_schema = {
        TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V2:
            TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V2,
        TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3:
            TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V3,
        TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4:
            TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V4,
    }[request.schema_version]
    return json.dumps(
        {
            "schema_version": batch_schema,
            "batch_digest": target_bound_draft_review_batch_digest((request,)),
            "responses": [json.loads(response)],
        },
        ensure_ascii=False,
    )


def _evaluate(content, source, request, **item_updates):
    return evaluate_target_bound_draft_review(
        request,
        _response(request, item_updates=item_updates),
        expected_source=source,
        frozen_content=content,
    )


@pytest.mark.parametrize(
    "content",
    (
        "余霁公开指控搭档周岚撒谎。",
        "余霁拒绝把共同保管的钥匙交给周岚。",
        "余霁拒绝把钥匙交给周岚。",
    ),
)
def test_v4_unique_literal_null_offsets_bind_frozen_target(content):
    source, request = _request(content)
    decision = _evaluate(content, source, request).decisions[0]

    assert request.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4
    assert request.prompt_version == TARGET_BOUND_DRAFT_REVIEW_PROMPT_V8
    assert decision.verdict == "supported"
    assert decision.observed_object == "周岚"
    assert decision.object_basis_id == request.proposals[0].fact_clause_id


def test_v4_prompt_projects_server_binding_but_calls_it_coordinate_only():
    content = "沈砚明确表示自己讨厌甜食。"
    source, request = _request(
        content,
        target=_target(
            character="沈砚",
            dimension="preference",
            trait_key="food_preference",
            comparison_key="preference:food_preference:甜食",
            key_object="甜食",
            baseline_hint="沈砚长期喜欢甜食",
        ),
        suffix="shenyan",
    )
    system, user = build_target_bound_draft_review_prompts((request,))
    payload = json.loads(user.split("：\n", 1)[1])
    hint = payload["windows"][0]["basis_path_hints"][0]
    fact = next(
        clause.text
        for line in request.lines
        for clause in line.clauses
        if clause.support_id == request.proposals[0].fact_clause_id
    )

    assert system == TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V8
    assert "只是“若语义确为 matches_target 时从哪里取字”的坐标绑定" in system
    assert hint["object_binding_mode"] == "server_unique_target_literal"
    assert hint["server_object_start_offset"] == fact.index("甜食")
    assert hint["server_object_end_offset"] == fact.index("甜食") + 2
    decision = _evaluate(content, source, request).decisions[0]
    assert decision.observed_object == "甜食"


def test_v4_server_binding_rejects_even_a_correct_non_null_model_span():
    content = "余霁拒绝把钥匙交给周岚。"
    source, request = _request(content)
    fact = next(
        clause.text
        for line in request.lines
        for clause in line.clauses
        if clause.support_id == request.proposals[0].fact_clause_id
    )
    start = fact.index("周岚")
    decision = _evaluate(
        content,
        source,
        request,
        object_start_offset=start,
        object_end_offset=start + 2,
    ).decisions[0]

    assert decision.reason == "slot_conflict"
    assert decision.slot_conflicts == ("object_span",)
    assert decision.observed_object is None


@pytest.mark.parametrize("relation", ("broader", "narrower"))
def test_v4_broader_and_narrower_never_use_server_null_binding(relation):
    content = "余霁拒绝把钥匙交给周岚。"
    source, request = _request(content)
    decision = _evaluate(
        content, source, request, object_relation=relation
    ).decisions[0]
    assert decision.reason == "slot_conflict"
    assert decision.slot_conflicts == ("object_span",)


@pytest.mark.parametrize("relation", ("broader", "narrower"))
def test_v4_broader_and_narrower_accept_only_an_explicit_covering_span(relation):
    content = "余霁公开指控搭档周岚撒谎。"
    source, request = _request(content)
    fact = next(
        clause.text
        for line in request.lines
        for clause in line.clauses
        if clause.support_id == request.proposals[0].fact_clause_id
    )
    start = fact.index("搭档周岚")
    decision = _evaluate(
        content,
        source,
        request,
        object_relation=relation,
        object_start_offset=start,
        object_end_offset=start + len("搭档周岚"),
    ).decisions[0]
    assert decision.verdict == "supported"
    assert decision.observed_object == "搭档周岚"


@pytest.mark.parametrize(
    "content",
    (
        "余霁公开指控搭档撒谎。",
        "余霁拒绝把周岚的信交给周岚。",
    ),
    ids=("zero-occurrence", "two-occurrences"),
)
def test_v4_non_unique_literal_never_accepts_null_offsets(content):
    source, request = _request(content)
    system, user = build_target_bound_draft_review_prompts((request,))
    hint = json.loads(user.split("：\n", 1)[1])["windows"][0][
        "basis_path_hints"
    ][0]
    decision = _evaluate(content, source, request).decisions[0]

    assert system == TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V8
    assert hint["object_binding_mode"] == "model_span"
    assert hint["server_object_start_offset"] is None
    assert hint["server_object_end_offset"] is None
    assert decision.reason == "slot_conflict"
    assert decision.slot_conflicts == ("object_span",)


def test_v4_overlapping_raw_occurrences_are_counted_as_multiple():
    content = "余霁明确表示自己拒绝哈哈哈。"
    source, request = _request(
        content,
        target=_target(
            key_object="哈哈",
            comparison_key="relationship_attitude:trust_orientation:哈哈",
            baseline_hint="余霁长期信任哈哈",
        ),
        suffix="overlap",
    )
    _, user = build_target_bound_draft_review_prompts((request,))
    hint = json.loads(user.split("：\n", 1)[1])["windows"][0][
        "basis_path_hints"
    ][0]
    decision = _evaluate(content, source, request).decisions[0]

    assert hint["object_binding_mode"] == "model_span"
    assert decision.slot_conflicts == ("object_span",)


@pytest.mark.parametrize(
    ("updates", "slot"),
    (
        ({"actor": "other"}, "actor"),
        ({"actuality": "reported"}, "actuality"),
        ({"statement_relation": "ambiguous"}, "statement_relation"),
        ({"axis_relation": "different"}, "axis_relation"),
        ({"polarity_relation": "opposite"}, "polarity_relation"),
        ({"correction_relation": "corrected"}, "correction_relation"),
    ),
)
def test_v4_server_binding_never_clears_another_semantic_conflict(updates, slot):
    content = "余霁拒绝把钥匙交给周岚。"
    source, request = _request(content)
    decision = _evaluate(content, source, request, **updates).decisions[0]
    assert decision.reason == "slot_conflict"
    assert slot in decision.slot_conflicts
    assert decision.observed_object is None
    assert decision.object_basis_id is None


def test_v4_server_binding_never_repairs_invalid_basis_path():
    content = "余霁拒绝把钥匙交给周岚。"
    source, request = _request(content)
    decision = _evaluate(
        content, source, request, basis_ids=["L999:A1"]
    ).decisions[0]
    assert decision.reason == "basis_invalid"
    assert decision.observed_object is None
    assert decision.object_basis_id is None


def test_v4_source_and_trailing_context_gates_remain_fail_closed():
    content = "余霁拒绝把钥匙交给周岚。\n前述行动其实是一场梦。"
    source, request = _request(content)
    vetoed = _evaluate(content, source, request).decisions[0]
    assert vetoed.reason == "source_context_veto"
    assert vetoed.observed_object is None

    wrong_source = source.model_copy(update={"document_version": 2})
    mismatched = evaluate_target_bound_draft_review(
        request,
        _response(request),
        expected_source=wrong_source,
        frozen_content=content,
    ).decisions[0]
    assert mismatched.reason == "source_mismatch"
    assert mismatched.observed_object is None


def test_v4_scoped_axis_remains_not_applicable_even_when_key_literal_occurs():
    content = "顾潮明确表示自己拒绝牺牲平民。"
    scope = "战争决策中是否主动牺牲无辜平民"
    proposition = "顾潮拒绝主动牺牲无辜平民"
    definition = "对无辜者生命的底线"
    target = _target(
        character="顾潮",
        dimension="value",
        trait_key="civilian_boundary",
        comparison_key="value:civilian_boundary:牺牲平民",
        key_object="牺牲平民",
        baseline_hint="顾潮拒绝牺牲无辜平民",
        approved_axis_id="axis-civilian",
        approved_axis_version=1,
        approved_axis_definition=definition,
        approved_axis_definition_sha256=hashlib.sha256(
            definition.encode("utf-8")
        ).hexdigest(),
        approved_axis_comparison_key="value:civilian_boundary",
        approved_axis_applicability_scope=scope,
        approved_axis_applicability_scope_sha256=hashlib.sha256(
            scope.encode("utf-8")
        ).hexdigest(),
        axis_positive_proposition=proposition,
        axis_positive_proposition_sha256=hashlib.sha256(
            proposition.encode("utf-8")
        ).hexdigest(),
    )
    source, request = _request(content, target=target, suffix="scoped")
    _, user = build_target_bound_draft_review_prompts((request,))
    hint = json.loads(user.split("：\n", 1)[1])["windows"][0][
        "basis_path_hints"
    ][0]
    decision = _evaluate(
        content,
        source,
        request,
        object_relation="not_applicable",
        axis_relation="matches_scoped_axis",
        observation_kind="decision",
    ).decisions[0]

    assert hint["object_binding_mode"] == "not_applicable"
    assert decision.verdict == "supported"
    assert decision.observed_object is None
    assert decision.object_basis_id is None


def test_v2_and_v3_do_not_gain_v4_null_server_binding():
    v3_content = "余霁拒绝把钥匙交给周岚。"
    v3_source, v3_request = _request(v3_content, protocol="v3", suffix="v3")
    v3_decision = _evaluate(v3_content, v3_source, v3_request).decisions[0]
    assert v3_decision.reason == "slot_conflict"
    assert v3_decision.slot_conflicts == ("object_span",)

    v2_content = (
        "午餐时，旁白直接说明余霁现在一直拒绝周岚，"
        "这不是引语、谎言或伪装。"
    )
    v2_source, v2_request = _request(v2_content, protocol="v2", suffix="v2")
    v2_decision = _evaluate(v2_content, v2_source, v2_request).decisions[0]
    assert v2_request.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V2
    assert v2_decision.verdict == "supported"
    assert v2_decision.observed_object is None


@pytest.mark.parametrize(
    ("protocol", "wrong_schema"),
    (("v4", TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3),
     ("v3", TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4)),
)
def test_v3_v4_response_schema_cross_use_fails_closed(protocol, wrong_schema):
    content = "余霁拒绝把钥匙交给周岚。"
    source, request = _request(content, protocol=protocol, suffix=f"cross-{protocol}")
    payload = json.loads(_response(request))
    payload["schema_version"] = wrong_schema
    decision = evaluate_target_bound_draft_review(
        request,
        json.dumps(payload, ensure_ascii=False),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert decision.reason == "response_mismatch"
    assert decision.observed_object is None


def test_v4_binding_uses_raw_unicode_without_normalized_literal_matching():
    content = "余霁明确表示自己拒绝A。"
    source, request = _request(
        content,
        target=_target(
            key_object="Ａ",
            comparison_key="relationship_attitude:trust_orientation:Ａ",
            baseline_hint="余霁长期信任Ａ",
        ),
        suffix="raw-unicode",
    )
    _, user = build_target_bound_draft_review_prompts((request,))
    hint = json.loads(user.split("：\n", 1)[1])["windows"][0][
        "basis_path_hints"
    ][0]
    decision = _evaluate(content, source, request).decisions[0]
    assert hint["object_binding_mode"] == "model_span"
    assert decision.slot_conflicts == ("object_span",)


class _Provider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.responses.pop(0)


def _run(content, source, request, provider):
    return run_target_bound_draft_review(
        (TargetBoundDraftReviewBatchEntry(request, source, content),),
        provider=provider,
        token_budget=200_000,
        completion_reserve=512,
        timeout_seconds=5,
        remaining_deadline_seconds=20,
        max_response_bytes=32_768,
        max_attempts=1,
    )


def test_v4_object_span_failure_does_not_replay_but_contract_failure_still_can():
    content = "余霁拒绝把钥匙交给周岚。"
    source, request = _request(content)
    fact = next(
        clause.text
        for line in request.lines
        for clause in line.clauses
        if clause.support_id == request.proposals[0].fact_clause_id
    )
    start = fact.index("周岚")
    span_failure = _batch_response(
        request,
        _response(
            request,
            item_updates={
                "object_start_offset": start,
                "object_end_offset": start + 2,
            },
        ),
    )
    provider = _Provider(
        [SimpleNamespace(text=span_failure, prompt_tokens=10, completion_tokens=5)]
    )
    run = _run(content, source, request, provider)
    assert run.attempted_calls == 1
    assert run.evaluation.evaluations[0].decisions[0].slot_conflicts == (
        "object_span",
    )

    provider = _Provider(
        [
            SimpleNamespace(text="{}", prompt_tokens=3, completion_tokens=1),
            SimpleNamespace(text=span_failure, prompt_tokens=10, completion_tokens=5),
            SimpleNamespace(
                text=_batch_response(request, _response(request)),
                prompt_tokens=10,
                completion_tokens=5,
            ),
        ]
    )
    run = _run(content, source, request, provider)
    assert run.attempted_calls == 2
    assert len(provider.calls) == 2
    assert provider.calls[0] == provider.calls[1]
    assert run.evaluation.evaluations[0].decisions[0].slot_conflicts == (
        "object_span",
    )


def test_mixed_v3_v4_batch_is_rejected_before_provider_use():
    content = "余霁拒绝把钥匙交给周岚。"
    _, v3 = _request(content, protocol="v3", suffix="mixed-v3")
    _, v4 = _request(content, protocol="v4", suffix="mixed-v4")
    with pytest.raises(ValueError, match="target_bound_batch_duplicate_or_too_large"):
        target_bound_draft_review_batch_digest((v3, v4))


def test_v4_server_binding_is_recomputed_per_window_without_cross_binding():
    unique_content = "余霁拒绝把钥匙交给周岚。"
    repeated_content = "余霁拒绝把周岚的信交给周岚。"
    unique_source, unique = _request(unique_content, suffix="window-one")
    repeated_source, repeated = _request(repeated_content, suffix="window-two")
    requests = (unique, repeated)
    _, user = build_target_bound_draft_review_prompts(requests)
    windows = json.loads(user.split("：\n", 1)[1])["windows"]
    assert [
        window["basis_path_hints"][0]["object_binding_mode"]
        for window in windows
    ] == ["server_unique_target_literal", "model_span"]

    raw = json.dumps(
        {
            "schema_version": TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V4,
            "batch_digest": target_bound_draft_review_batch_digest(requests),
            "responses": [
                json.loads(_response(unique)),
                json.loads(_response(repeated)),
            ],
        },
        ensure_ascii=False,
    )
    evaluation = evaluate_target_bound_draft_review_batch(
        requests,
        raw,
        expected_sources=(unique_source, repeated_source),
        frozen_contents=(unique_content, repeated_content),
    )
    first, second = evaluation.evaluations
    assert first.decisions[0].verdict == "supported"
    assert first.decisions[0].observed_object == "周岚"
    assert second.decisions[0].reason == "slot_conflict"
    assert second.decisions[0].slot_conflicts == ("object_span",)
    assert second.decisions[0].observed_object is None
