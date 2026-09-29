from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from app.character_draft_actor_review import (
    TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V2,
    TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V2,
    TargetBoundDraftReviewTarget,
    build_target_bound_draft_proposals,
    build_target_bound_draft_review_request,
    evaluate_target_bound_draft_review,
    evaluate_target_bound_draft_review_batch,
    required_target_bound_draft_basis_ids,
    target_bound_draft_review_batch_digest,
    target_bound_draft_review_request_digest,
)
from app.character_scope_review import ScopeReviewSourceIdentity
from app.provider import ProviderError
from app.character_draft_actor_review_provider import (
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT,
    TargetBoundDraftReviewBatchEntry,
    build_target_bound_draft_review_prompts,
    run_target_bound_draft_review,
)


def _source(content: str, suffix: str = "1") -> ScopeReviewSourceIdentity:
    return ScopeReviewSourceIdentity(
        run_input_id=f"run-{suffix}",
        document_id=f"draft-{suffix}",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _target(**updates) -> TargetBoundDraftReviewTarget:
    values = {
        "target_ordinal": 2,
        "character": "沈砚",
        "dimension": "preference",
        "trait_key": "food_preference",
        "comparison_key": "preference:栗子糕",
        "key_object": "栗子糕",
        "requested_polarity": "negative",
        "baseline_hint": "沈砚稳定喜欢栗子糕",
    }
    values.update(updates)
    return TargetBoundDraftReviewTarget(**values)


def _request(content: str, target=None, *, line_start=1, line_end=1, suffix="1"):
    source = _source(content, suffix)
    target = target or _target()
    index, proposals = build_target_bound_draft_proposals(
        content,
        source,
        target,
        line_start=line_start,
        line_end=line_end,
    )
    assert proposals
    request = build_target_bound_draft_review_request(
        index,
        target,
        proposals,
        frozen_content=content,
        expected_source=source,
    )
    return source, request


def _item(request, proposal, **overrides):
    scoped = request.target.approved_axis_comparison_key is not None
    value = {
        "proposal_id": proposal.proposal_id,
        "verdict": "supported",
        "actor": "proposed",
        "actuality": "asserted",
        "statement_relation": "supported",
        "axis_relation": "requires_scoped_review" if scoped else "matches_target",
        "object_relation": (
            "not_applicable"
            if scoped or not request.target.key_object
            else "matches_target"
        ),
        "polarity_relation": "requested",
        "correction_relation": "none",
        "observation_kind": "preference_expression",
        "basis_ids": list(required_target_bound_draft_basis_ids(request)),
    }
    value.update(overrides)
    return value


def _single_response(request, items=None):
    return json.dumps(
        {
            "schema_version": TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V2,
            "request_digest": target_bound_draft_review_request_digest(request),
            "items": items or [
                _item(request, proposal) for proposal in request.proposals
            ],
        },
        ensure_ascii=False,
    )


class _Provider:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class _SequenceProvider:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.values.pop(0)


def _batch_response(request, items=None):
    return json.dumps(
        {
            "schema_version": TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V2,
            "batch_digest": target_bound_draft_review_batch_digest((request,)),
            "responses": [json.loads(_single_response(request, items))],
        },
        ensure_ascii=False,
    )


def _run_provider(content, source, request, provider, **updates):
    values = {
        "token_budget": 100_000,
        "completion_reserve": 512,
        "timeout_seconds": 5,
        "remaining_deadline_seconds": 20,
        "max_response_bytes": 32_768,
        "max_attempts": 1,
    }
    values.update(updates)
    return run_target_bound_draft_review(
        (TargetBoundDraftReviewBatchEntry(request, source, content),),
        provider=provider,
        **values,
    )


def test_narrator_and_adjacent_zero_subject_are_server_derived():
    narrator = (
        "午餐时，旁白直接说明沈砚现在一直最讨厌栗子糕，"
        "闻到甜味就想离开；这不是引语、谎言或伪装。"
    )
    source, request = _request(narrator)
    assert [(p.binding_kind, p.canonical_statement) for p in request.proposals] == [
        ("narrator_explicit_attribution", "沈砚现在一直最讨厌栗子糕")
    ]

    content = "顾潮又解散重建小组，明确表示永久放弃重建海灯塔。"
    target = _target(
        character="顾潮",
        dimension="motivation_goal",
        trait_key="lighthouse_commitment",
        comparison_key="motivation_goal:lighthouse_commitment:重建海灯塔",
        key_object="重建海灯塔",
        baseline_hint="顾潮长期致力于重建海灯塔",
    )
    _, request = _request(content, target, suffix="guchao")
    assert {p.binding_kind for p in request.proposals} == {
        "adjacent_zero_subject",
    }
    assert any(
        p.canonical_statement == "顾潮明确表示永久放弃重建海灯塔"
        for p in request.proposals
    )

    direct = "沈砚明确表示自己讨厌栗子糕。"
    direct_source = _source(direct, "direct")
    _, direct_proposals = build_target_bound_draft_proposals(
        direct, direct_source, _target(), line_start=1, line_end=1
    )
    assert direct_proposals == ()


@pytest.mark.parametrize(
    "content",
    (
        "如果宴会继续，沈砚最讨厌栗子糕。",
        "沈砚在梦境中最讨厌栗子糕。",
        "沈砚看见唐岫搬走货物。",
        "沈砚让唐岫搬走货物。",
        "旁白转述有人说沈砚最讨厌栗子糕。",
        "旁白写道：“沈砚最讨厌栗子糕”。",
    ),
)
def test_unsafe_condition_quote_dream_delegation_and_perception_do_not_propose(content):
    source = _source(content)
    _, proposals = build_target_bound_draft_proposals(
        content, source, _target(), line_start=1, line_end=1
    )
    assert proposals == ()


def test_supported_response_yields_only_server_certificate_fields():
    content = "旁白明确说明沈砚现在讨厌栗子糕。"
    source, request = _request(content)
    evaluation = evaluate_target_bound_draft_review(
        request,
        _single_response(request),
        expected_source=source,
        frozen_content=content,
    )
    decision = evaluation.decisions[0]
    assert (decision.verdict, decision.reason) == ("supported", "supported")
    assert decision.target_digest == request.target_digest
    assert decision.target_ordinal == 2
    assert decision.observation_kind == "preference_expression"

    injected = json.loads(_single_response(request))
    injected["items"][0]["character"] = "另一个角色"
    invalid = evaluate_target_bound_draft_review(
        request,
        json.dumps(injected, ensure_ascii=False),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (invalid.verdict, invalid.reason) == ("uncertain", "response_invalid")


def test_axis_object_direction_conflicts_and_basis_tampering_fail_closed():
    content = "旁白明确说明沈砚现在讨厌栗子糕。"
    source, request = _request(content)
    proposal = request.proposals[0]
    for overrides, reason in (
        ({"axis_relation": "different"}, "slot_conflict"),
        ({"object_relation": "different"}, "slot_conflict"),
        ({"polarity_relation": "opposite"}, "slot_conflict"),
        ({"basis_ids": []}, "basis_invalid"),
    ):
        decision = evaluate_target_bound_draft_review(
            request,
            _single_response(request, [_item(request, proposal, **overrides)]),
            expected_source=source,
            frozen_content=content,
        ).decisions[0]
        assert (decision.verdict, decision.reason) == ("uncertain", reason)


def test_frozen_source_hash_and_proposal_tampering_fail_closed():
    content = "旁白明确说明沈砚现在讨厌栗子糕。"
    source, request = _request(content)
    raw = json.loads(_single_response(request))

    changed_proposal = json.loads(json.dumps(raw, ensure_ascii=False))
    changed_proposal["items"][0]["proposal_id"] = "proposal-tampered"
    decision = evaluate_target_bound_draft_review(
        request,
        json.dumps(changed_proposal, ensure_ascii=False),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (decision.verdict, decision.reason) == (
        "uncertain", "response_invalid"
    )

    changed_source = source.model_copy(
        update={"content_sha256": "0" * 64}
    )
    for expected_source, frozen_content in (
        (changed_source, content),
        (source, content + "被篡改"),
    ):
        decision = evaluate_target_bound_draft_review(
            request,
            json.dumps(raw, ensure_ascii=False),
            expected_source=expected_source,
            frozen_content=frozen_content,
        ).decisions[0]
        assert (decision.verdict, decision.reason) == (
            "uncertain", "source_mismatch"
        )


def test_scoped_value_only_certifies_direction_and_defers_final_axis():
    definition = "遇到生命与货物冲突时优先救人"
    scope = "生命救援与货物保全发生直接冲突"
    proposition = "角色先救人再处理货物"
    target = _target(
        character="唐岫",
        dimension="value",
        trait_key="rescue_priority",
        comparison_key="value:先救人再保货",
        key_object="先救人再保货",
        baseline_hint="唐岫坚持先救人再保货",
        approved_axis_id="11111111-1111-4111-8111-111111111111",
        approved_axis_version=1,
        approved_axis_definition=definition,
        approved_axis_definition_sha256=hashlib.sha256(definition.encode()).hexdigest(),
        approved_axis_comparison_key="value:先救人再保货",
        approved_axis_applicability_scope=scope,
        approved_axis_applicability_scope_sha256=hashlib.sha256(scope.encode()).hexdigest(),
        axis_positive_proposition=proposition,
        axis_positive_proposition_sha256=hashlib.sha256(proposition.encode()).hexdigest(),
    )
    content = "唐岫再次要求先打捞货物，随后才允许救人。"
    source, request = _request(content, target, suffix="tang")
    evaluation = evaluate_target_bound_draft_review(
        request,
        _single_response(request, [
            _item(
                request,
                proposal,
                observation_kind="decision",
            )
            for proposal in request.proposals
        ]),
        expected_source=source,
        frozen_content=content,
    )
    assert {decision.verdict for decision in evaluation.decisions} == {"supported"}

    wrong_axis = evaluate_target_bound_draft_review(
        request,
        _single_response(request, [
            _item(request, proposal, axis_relation="matches_target")
            for proposal in request.proposals
        ]),
        expected_source=source,
        frozen_content=content,
    )
    assert {decision.reason for decision in wrong_axis.decisions} == {"slot_conflict"}


def test_later_correction_veto_and_batch_target_identity_are_strict():
    first = "旁白明确说明沈砚现在讨厌栗子糕。"
    content = first + "\n更正：上一行记录有误。"
    source, request = _request(content, line_start=1, line_end=1, suffix="corrected")
    decision = evaluate_target_bound_draft_review(
        request,
        _single_response(request),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (decision.verdict, decision.reason) == (
        "uncertain", "source_context_veto"
    )

    clean = "旁白明确说明沈砚现在讨厌栗子糕。"
    source2, request2 = _request(clean, suffix="clean")
    requests = (request2,)
    raw = json.dumps(
        {
            "schema_version": TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V2,
            "batch_digest": target_bound_draft_review_batch_digest(requests),
            "responses": [json.loads(_single_response(request2))],
        },
        ensure_ascii=False,
    )
    assert evaluate_target_bound_draft_review_batch(
        requests,
        raw,
        expected_sources=(source2,),
        frozen_contents=(clean,),
    ).evaluations[0].decisions[0].verdict == "supported"

    changed = request2.model_copy(
        update={"target": request2.target.model_copy(update={"target_ordinal": 3})}
    )
    replay = evaluate_target_bound_draft_review(
        changed,
        _single_response(request2),
        expected_source=source2,
        frozen_content=clean,
    ).decisions[0]
    assert replay.reason == "source_mismatch"


def test_v2_has_an_independent_prompt_and_bounded_failure_accounting():
    content = "旁白明确说明沈砚现在讨厌栗子糕。"
    source, request = _request(content)
    system, user = build_target_bound_draft_review_prompts((request,))
    assert system == TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT
    assert "character-target-bound-draft-review-batch-v2" in system
    payload = json.loads(user.split("：\n", 1)[1])
    assert "protocol" not in payload

    unused = _Provider(SimpleNamespace(text="unused", prompt_tokens=0, completion_tokens=0))
    budget = _run_provider(content, source, request, unused, token_budget=0)
    assert (budget.failure_reason, budget.attempted_calls, budget.charged_tokens) == (
        "token_budget", 0, 0
    )
    assert unused.calls == []

    secret = "sk-secret-must-not-leak"
    failed = _run_provider(content, source, request, _Provider(RuntimeError(secret)))
    assert failed.failure_reason == "provider_error"
    assert failed.attempted_calls == 1
    assert failed.charged_tokens == failed.estimated_tokens > 0
    assert secret not in repr(failed)

    timeout_secret = "sk-timeout-must-not-leak"
    timed_out = _run_provider(
        content,
        source,
        request,
        _Provider(ProviderError(timeout_secret, category="read_timeout")),
    )
    assert timed_out.failure_reason == "provider_timeout"
    assert timed_out.attempted_calls == 1
    assert timed_out.charged_tokens == timed_out.estimated_tokens > 0
    assert timeout_secret not in repr(timed_out)

    valid = json.dumps({
        "schema_version": TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V2,
        "batch_digest": target_bound_draft_review_batch_digest((request,)),
        "responses": [json.loads(_single_response(request))],
    }, ensure_ascii=False)
    oversized = _run_provider(
        content,
        source,
        request,
        _Provider(SimpleNamespace(text=valid, prompt_tokens=1, completion_tokens=1)),
        max_response_bytes=64,
    )
    assert oversized.failure_reason == "response_too_large"
    assert oversized.attempted_calls == 1
    assert oversized.charged_tokens == oversized.estimated_tokens > 0

    probe = _run_provider(content, source, request, unused, token_budget=0)
    over_usage = _run_provider(
        content,
        source,
        request,
        _Provider(SimpleNamespace(
            text=valid,
            prompt_tokens=probe.estimated_tokens + 1,
            completion_tokens=1,
        )),
        token_budget=probe.estimated_tokens,
    )
    assert over_usage.failure_reason == "token_budget"
    assert over_usage.attempted_calls == 1
    assert over_usage.charged_tokens > over_usage.estimated_tokens


def test_whole_batch_contract_failure_replays_same_frozen_prompts_once():
    content = "旁白明确说明沈砚现在讨厌栗子糕。"
    source, request = _request(content)
    provider = _SequenceProvider([
        SimpleNamespace(text="{invalid", prompt_tokens=7, completion_tokens=3),
        SimpleNamespace(
            text=_batch_response(request), prompt_tokens=11, completion_tokens=5
        ),
    ])

    run = _run_provider(content, source, request, provider)

    assert run.failure_reason is None
    assert run.attempted_calls == 2
    assert run.prompt_tokens == 18
    assert run.completion_tokens == 8
    assert run.charged_tokens == run.estimated_tokens
    assert run.evaluation.evaluations[0].decisions[0].verdict == "supported"
    assert len(provider.calls) == 2
    assert provider.calls[0] == provider.calls[1]


def test_contract_replay_exhaustion_fails_closed_and_semantic_rejection_does_not_retry():
    content = "旁白明确说明沈砚现在讨厌栗子糕。"
    source, request = _request(content)
    invalid_provider = _SequenceProvider([
        SimpleNamespace(text="{invalid", prompt_tokens=1, completion_tokens=1),
        SimpleNamespace(text="[]", prompt_tokens=2, completion_tokens=1),
    ])

    exhausted = _run_provider(content, source, request, invalid_provider)

    assert (exhausted.failure_reason, exhausted.attempted_calls) == (
        "response_invalid", 2
    )
    assert exhausted.prompt_tokens == 3
    assert exhausted.completion_tokens == 2
    assert {
        decision.reason
        for item in exhausted.evaluation.evaluations
        for decision in item.decisions
    } == {"response_invalid"}
    assert invalid_provider.calls[0] == invalid_provider.calls[1]

    proposal = request.proposals[0]
    slot_conflict = _batch_response(
        request, [_item(request, proposal, actor="other")]
    )
    semantic_provider = _SequenceProvider([
        SimpleNamespace(text=slot_conflict, prompt_tokens=1, completion_tokens=1),
        SimpleNamespace(text=_batch_response(request), prompt_tokens=1, completion_tokens=1),
    ])
    semantic = _run_provider(content, source, request, semantic_provider)
    assert (semantic.failure_reason, semantic.attempted_calls) == (
        "slot_conflict", 1
    )
    assert len(semantic_provider.calls) == 1

    v2_object_span = _batch_response(
        request,
        [_item(request, proposal, object_start_offset=0, object_end_offset=1)],
    )
    v2_provider = _SequenceProvider([
        SimpleNamespace(text=v2_object_span, prompt_tokens=1, completion_tokens=1),
        SimpleNamespace(text=_batch_response(request), prompt_tokens=1, completion_tokens=1),
    ])
    v2 = _run_provider(content, source, request, v2_provider)
    assert (v2.failure_reason, v2.attempted_calls) == ("slot_conflict", 1)
    assert v2.evaluation.evaluations[0].decisions[0].slot_conflicts == (
        "object_span",
    )
    assert len(v2_provider.calls) == 1


def test_contract_replay_rechecks_shared_budget_before_second_call():
    content = "旁白明确说明沈砚现在讨厌栗子糕。"
    source, request = _request(content)
    probe = _run_provider(
        content,
        source,
        request,
        _Provider(SimpleNamespace(text="unused", prompt_tokens=0, completion_tokens=0)),
        token_budget=0,
    )
    provider = _SequenceProvider([
        SimpleNamespace(text="{invalid", prompt_tokens=1, completion_tokens=1),
        SimpleNamespace(text=_batch_response(request), prompt_tokens=1, completion_tokens=1),
    ])

    run = _run_provider(
        content, source, request, provider, token_budget=probe.estimated_tokens
    )

    assert (run.failure_reason, run.attempted_calls) == ("token_budget", 1)
    assert run.estimated_tokens == probe.estimated_tokens * 2
    assert run.charged_tokens == probe.estimated_tokens
    assert len(provider.calls) == 1
