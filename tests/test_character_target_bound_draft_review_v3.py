from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.character_draft_actor_review import (
    TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V3,
    TARGET_BOUND_DRAFT_REVIEW_PROMPT_V7,
    TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3,
    TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V4,
    TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4,
    TargetBoundDraftReviewTarget,
    build_target_bound_draft_proposals,
    build_target_bound_draft_review_request,
    evaluate_target_bound_draft_review,
    evaluate_target_bound_draft_review_batch,
    required_target_bound_draft_basis_ids,
    target_bound_draft_target_digest,
    target_bound_draft_review_batch_digest,
    target_bound_draft_review_request_digest,
)
from app.character_draft_actor_review_provider import (
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V7,
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V8,
    TargetBoundDraftReviewBatchEntry,
    build_target_bound_draft_review_prompts,
    run_target_bound_draft_review,
)
from app.character_consistency_stage import (
    _explanation_observation_key_object,
    _observation_matches_baseline,
    _signal_matches_target,
)
from app.character_explanation_review import (
    ExplanationBaselineSummary,
    ExplanationCandidate,
    ExplanationObservationSummary,
    build_character_explanation_review_prompts,
)
from app.character_drift import (
    CharacterDriftCase,
    CharacterReviewDiagnostics,
    CharacterReviewResult,
    ConfirmedTraitSnapshot,
    EventIdentityDecision,
    ModelDriftDecision,
    prepare_character_drift,
    promote_character_drift,
)
from app.character_scope_review import ScopeReviewSourceIdentity
from app.character_trait_extraction import (
    CharacterSignal,
    CharacterSignalChunk,
    CharacterSignalExtractor,
    CharacterSignalTarget,
    _SignalValidationFailure,
    _TargetBoundDraftOutcome,
    _ValidatedSignalPackage,
    _resolve_completed_target_bound_discovery,
    _target_bound_review_target,
    stable_trait_identity,
)
from app.config import Settings
from app.domain import EvidenceSpan
from app.provider import ProviderError


def _source(content: str) -> ScopeReviewSourceIdentity:
    return ScopeReviewSourceIdentity(
        run_input_id="run-input-v3",
        document_id="draft-v3",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _review_target(**updates) -> TargetBoundDraftReviewTarget:
    values = {
        "target_ordinal": 1,
        "character": "沈砚",
        "dimension": "preference",
        "trait_key": "food_preference",
        "comparison_key": "preference:甜食",
        "key_object": "甜食",
        "requested_polarity": "negative",
        "baseline_hint": "沈砚长期喜欢甜食",
    }
    values.update(updates)
    return TargetBoundDraftReviewTarget(**values)


@pytest.mark.parametrize(
    "unsafe_literal",
    (
        "我", "我们", "你", "你们", "他", "她", "它们", "其", "本人", "自己",
        "这个", "那些", "此人", "该人", "对方", "they",
    ),
)
def test_v3_target_rejects_nonunique_pronoun_and_demonstrative_aliases(
    unsafe_literal,
):
    with pytest.raises(ValidationError, match="target_bound_actor_invalid"):
        _review_target(authorized_aliases=(unsafe_literal,))


@pytest.mark.parametrize("unsafe_character", ("我", "她", "本人", "这个", "they"))
def test_v3_actor_literal_boundary_rejects_pronoun_canonical_character(
    unsafe_character,
):
    with pytest.raises(ValidationError, match="target_bound_actor_invalid"):
        _review_target(character=unsafe_character)


def test_v3_keeps_a_unique_formal_alias_as_an_explicit_named_subject():
    content = "砚先生明确表示自己讨厌栗子糕。"
    target = _review_target(authorized_aliases=("砚先生",))
    source = _source(content)
    _, proposals = build_target_bound_draft_proposals(
        content, source, target, line_start=1, line_end=1,
        protocol_version="v3",
    )
    assert [(item.binding_kind, item.canonical_statement) for item in proposals] == [
        ("explicit_named_subject", "沈砚明确表示自己讨厌栗子糕")
    ]


def _request(content: str, target: TargetBoundDraftReviewTarget | None = None):
    source = _source(content)
    target = target or _review_target()
    index, proposals = build_target_bound_draft_proposals(
        content,
        source,
        target,
        line_start=1,
        line_end=min(2, len(content.splitlines())),
        protocol_version="v3",
    )
    assert proposals
    request = build_target_bound_draft_review_request(
        index,
        target,
        proposals,
        frozen_content=content,
        expected_source=source,
        protocol_version="v3",
    )
    return source, request


def _request_for_object_fact(content: str, object_text: str = "栗子糕"):
    source = _source(content)
    target = _review_target()
    index, proposals = build_target_bound_draft_proposals(
        content,
        source,
        target,
        line_start=1,
        line_end=1,
        protocol_version="v3",
    )
    selected = tuple(
        proposal
        for proposal in proposals
        if (
            (fact := index.resolve(proposal.fact_clause_id)) is not None
            and object_text in fact.text
        )
    )
    assert len(selected) == 1
    request = build_target_bound_draft_review_request(
        index,
        target,
        selected,
        frozen_content=content,
        expected_source=source,
        protocol_version="v3",
    )
    return source, request


def test_v3_request_uses_v7_scoped_axis_object_contract_identity():
    _, request = _request("沈砚明确表示自己讨厌栗子糕。")

    assert request.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3
    assert request.prompt_version == TARGET_BOUND_DRAFT_REVIEW_PROMPT_V7

    stale = request.model_dump(mode="python")
    stale["prompt_version"] = "character-target-bound-draft-review-prompt-v6"
    with pytest.raises(ValidationError, match="target_bound_protocol_identity_invalid"):
        type(request).model_validate(stale, strict=True)


def test_v7_prompt_spells_out_nested_batch_contract_every_enum_and_offset_origin():
    _, request = _request("沈砚明确表示自己讨厌栗子糕。")
    system, user = build_target_bound_draft_review_prompts((request,))

    assert system == TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V7
    for literal in (
        '"responses": [',
        '"request_digest":',
        '"items": [',
        "responses 必须与输入 windows 数量相同",
        "绝不能把 proposal 直接平铺成 responses",
        "verdict: supported | rejected | uncertain",
        "actor: proposed | other | ambiguous",
        "actuality: asserted | reported | hypothetical | question | ambiguous",
        "statement_relation: supported | contradicted | ambiguous",
        (
            "axis_relation: matches_target | matches_scoped_axis | different | "
            "requires_scoped_review | ambiguous"
        ),
        (
            "object_relation: matches_target | broader | narrower | different | "
            "not_applicable | ambiguous"
        ),
        "polarity_relation: requested | opposite | neutral | ambiguous",
        "correction_relation: none | corrected | ambiguous",
        (
            "observation_kind: preference_expression | speech_sample | action | "
            "decision | interaction | state_description"
        ),
        "basis_ids 只能为空数组或同样完整回显",
        "object_offset_origin 固定为 fact_clause_text[0]",
        "request.lines 和 proposal.evidence 保留完整冻结上下文供语义判断；它们绝不是 offset 坐标系",
        "comparison_key 尾部和 target.key_object 只是冻结的“轴身份”",
        "object_relation=not_applicable、object_start_offset=null、object_end_offset=null",
        "若 target.key_object 在该 fact_clause_text 中恰好逐字出现一次",
    ):
        assert literal in system
    payload = json.loads(user.split("：\n", 1)[1])
    assert payload["batch_digest"] == target_bound_draft_review_batch_digest(
        (request,)
    )
    assert payload["windows"][0]["request_digest"] == (
        target_bound_draft_review_request_digest(request)
    )
    hint = payload["windows"][0]["basis_path_hints"][0]
    proposal = request.proposals[0]
    clauses = {
        clause.support_id: clause.text
        for line in request.lines
        for clause in line.clauses
    }
    fact = clauses[proposal.fact_clause_id]
    assert hint == {
        "proposal_id": proposal.proposal_id,
        "required_basis_ids": list(required_target_bound_draft_basis_ids(request)),
        "fact_clause_id": proposal.fact_clause_id,
        "fact_clause_text": fact,
        "fact_clause_codepoint_length": len(fact),
        "object_offset_origin": "fact_clause_text[0]",
    }


def _response(request, *, object_text="栗子糕", **overrides):
    clauses = {
        clause["support_id"]: clause
        for line in request.model_dump(mode="json")["lines"]
        for clause in line["clauses"]
    }
    items = []
    for proposal in request.proposals:
        fact = clauses[proposal.fact_clause_id]["text"]
        start = fact.index(object_text) if object_text else None
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
            "object_relation": "narrower" if object_text else "not_applicable",
            "polarity_relation": "requested",
            "correction_relation": "none",
            "observation_kind": "preference_expression",
            "object_start_offset": start,
            "object_end_offset": start + len(object_text) if object_text else None,
            "basis_ids": list(required_target_bound_draft_basis_ids(request)),
        }
        item.update(overrides)
        items.append(item)
    return json.dumps(
        {
            "schema_version": TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3,
            "request_digest": target_bound_draft_review_request_digest(request),
            "items": items,
        },
        ensure_ascii=False,
    )


def _batch_response(requests, responses):
    return json.dumps(
        {
            "schema_version": TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V3,
            "batch_digest": target_bound_draft_review_batch_digest(requests),
            "responses": [json.loads(response) for response in responses],
        },
        ensure_ascii=False,
    )


class _ReplayProvider:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        return self.responses.pop(0)


def _run_review(entries, provider, **updates):
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
        tuple(
            TargetBoundDraftReviewBatchEntry(request, source, content)
            for content, source, request in entries
        ),
        provider=provider,
        **values,
    )


@pytest.mark.parametrize(
    "content",
    (
        "午餐时，旁白直接说明沈砚现在一直最讨厌栗子糕。",
        "沈砚走进餐厅，并明确表示自己讨厌栗子糕。",
    ),
    ids=("prefixed-narrator-clause", "same-line-second-clause"),
)
def test_v7_offsets_are_fact_clause_relative_and_evidence_relative_offsets_fail_closed(
    content,
):
    source, request = _request_for_object_fact(content)
    proposal = request.proposals[0]
    clauses = {
        clause.support_id: clause.text
        for line in request.lines
        for clause in line.clauses
    }
    fact = clauses[proposal.fact_clause_id]
    evidence_prefix_length = proposal.evidence.find(fact)
    assert evidence_prefix_length > 0

    fact_relative = _response(request)
    accepted = evaluate_target_bound_draft_review(
        request,
        fact_relative,
        expected_source=source,
        frozen_content=content,
    )
    assert accepted.decisions[0].verdict == "supported"
    assert accepted.decisions[0].reason == "supported"
    assert accepted.decisions[0].observed_object == "栗子糕"

    evidence_relative = json.loads(fact_relative)
    item = evidence_relative["items"][0]
    item["object_start_offset"] += evidence_prefix_length
    item["object_end_offset"] += evidence_prefix_length
    rejected = evaluate_target_bound_draft_review(
        request,
        json.dumps(evidence_relative, ensure_ascii=False),
        expected_source=source,
        frozen_content=content,
    )
    assert rejected.decisions[0].verdict == "uncertain"
    assert rejected.decisions[0].reason == "slot_conflict"
    assert rejected.decisions[0].slot_conflicts == ("object_span",)


def _relationship_request_for_literal_object(
    content: str,
    *,
    key_object: str = "周岚",
):
    target = _review_target(
        character="余霁",
        dimension="relationship_attitude",
        trait_key="trust_orientation",
        comparison_key=stable_trait_identity(
            "relationship_attitude", "trust_orientation", key_object
        ),
        key_object=key_object,
        baseline_hint=f"余霁长期信任{key_object}",
    )
    source = _source(content)
    index, proposals = build_target_bound_draft_proposals(
        content, source, target, line_start=1, line_end=1,
        protocol_version="v3",
    )
    selected = tuple(
        proposal
        for proposal in proposals
        if (
            (fact := index.resolve(proposal.fact_clause_id)) is not None
            and key_object in fact.text
        )
    )
    assert len(selected) == 1
    request = build_target_bound_draft_review_request(
        index,
        target,
        selected,
        frozen_content=content,
        expected_source=source,
        protocol_version="v3",
    )
    return source, request


@pytest.mark.parametrize(
    ("content", "bad_slice"),
    (
        ("余霁公开指控搭档周岚撒谎。", "搭档"),
        ("余霁拒绝把共同保管的钥匙交给周岚。", "匙交"),
        ("余霁拒绝把钥匙交给周岚。", "给周"),
    ),
)
def test_v3_unique_literal_target_rejects_adjacent_partial_object_span(
    content,
    bad_slice,
):
    source, request = _relationship_request_for_literal_object(content)
    decision = evaluate_target_bound_draft_review(
        request,
        _response(request, object_text=bad_slice),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]

    assert (decision.verdict, decision.reason) == ("uncertain", "slot_conflict")
    assert decision.slot_conflicts == ("object_span",)


@pytest.mark.parametrize("object_text", ("周岚", "搭档周岚"))
def test_v3_unique_literal_target_accepts_complete_target_covering_span(object_text):
    content = "余霁公开指控搭档周岚撒谎。"
    source, request = _relationship_request_for_literal_object(content)
    decision = evaluate_target_bound_draft_review(
        request,
        _response(request, object_text=object_text),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]

    assert (decision.verdict, decision.reason) == ("supported", "supported")
    assert decision.observed_object == object_text


def test_v3_no_exact_literal_target_keeps_existing_semantic_object_span_path():
    content = "沈砚明确表示自己讨厌栗子糕。"
    source, request = _request_for_object_fact(content)
    assert request.target.key_object == "甜食"
    assert request.target.key_object not in content

    decision = evaluate_target_bound_draft_review(
        request,
        _response(request, object_text="栗子糕"),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]

    assert (decision.verdict, decision.observed_object) == ("supported", "栗子糕")


def test_v3_unique_literal_target_conservatively_rejects_same_clause_narrower_object():
    content = "沈砚明确表示自己讨厌甜食中的栗子糕。"
    source, request = _request_for_object_fact(content)
    decision = evaluate_target_bound_draft_review(
        request,
        _response(request, object_text="栗子糕", object_relation="narrower"),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]

    assert (decision.verdict, decision.reason) == ("uncertain", "slot_conflict")
    assert decision.slot_conflicts == ("object_span",)


def test_v3_overlapping_literal_occurrences_are_ambiguous_without_normalization():
    content = "余霁拒绝周周周。"
    source, request = _relationship_request_for_literal_object(
        content,
        key_object="周周",
    )
    proposal = request.proposals[0]
    fact = next(
        clause.text
        for line in request.lines
        for clause in line.clauses
        if clause.support_id == proposal.fact_clause_id
    )
    first = fact.find("周周")
    second = fact.find("周周", first + 1)
    assert (first, second) == (4, 5)

    decision = evaluate_target_bound_draft_review(
        request,
        _response(
            request,
            object_text="周周",
            object_start_offset=second,
            object_end_offset=second + 2,
        ),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]

    assert (decision.verdict, decision.observed_object) == ("supported", "周周")


@pytest.mark.parametrize("key_object", (" 周岚", "周岚 ", "周\u200b岚"))
def test_v3_target_rejects_non_raw_literal_object_boundaries(key_object):
    with pytest.raises(ValidationError, match="target_bound_object_invalid"):
        _review_target(key_object=key_object)


def test_v3_literal_object_dimension_rejects_empty_target_object():
    with pytest.raises(ValidationError, match="target_bound_object_required"):
        _review_target(key_object="")


def test_v7_unique_literal_object_span_replays_same_frozen_batch_once():
    content = "余霁公开指控搭档周岚撒谎。"
    source, request = _relationship_request_for_literal_object(content)
    bad = _batch_response(
        (request,), (_response(request, object_text="搭档"),)
    )
    repaired = _batch_response(
        (request,), (_response(request, object_text="周岚"),)
    )
    provider = _ReplayProvider([
        SimpleNamespace(text=bad, prompt_tokens=7, completion_tokens=3),
        SimpleNamespace(text=repaired, prompt_tokens=11, completion_tokens=5),
    ])

    run = _run_review(((content, source, request),), provider)

    assert (run.failure_reason, run.attempted_calls) == (None, 2)
    assert (run.prompt_tokens, run.completion_tokens) == (18, 8)
    assert run.charged_tokens == run.estimated_tokens
    decision = run.evaluation.evaluations[0].decisions[0]
    assert (decision.verdict, decision.reason) == ("supported", "supported")
    assert decision.observed_object == "周岚"
    assert provider.calls[0] == provider.calls[1]


@pytest.mark.parametrize(
    ("second_response", "expected_conflicts"),
    (
        ("object_span", ("object_span",)),
        ("semantic", ("actor",)),
    ),
)
def test_v7_object_span_replay_exhaustion_uses_only_second_result(
    second_response,
    expected_conflicts,
):
    content = "余霁公开指控搭档周岚撒谎。"
    source, request = _relationship_request_for_literal_object(content)
    bad = _batch_response(
        (request,), (_response(request, object_text="搭档"),)
    )
    second = (
        bad
        if second_response == "object_span"
        else _batch_response(
            (request,),
            (_response(request, object_text="周岚", actor="other"),),
        )
    )
    provider = _ReplayProvider([
        SimpleNamespace(text=bad, prompt_tokens=1, completion_tokens=1),
        SimpleNamespace(text=second, prompt_tokens=1, completion_tokens=1),
        SimpleNamespace(
            text=_batch_response(
                (request,), (_response(request, object_text="周岚"),)
            ),
            prompt_tokens=1,
            completion_tokens=1,
        ),
    ])

    run = _run_review(((content, source, request),), provider)

    assert (run.failure_reason, run.attempted_calls) == ("slot_conflict", 2)
    decision = run.evaluation.evaluations[0].decisions[0]
    assert (decision.verdict, decision.reason) == ("uncertain", "slot_conflict")
    assert decision.slot_conflicts == expected_conflicts
    assert len(provider.calls) == 2


def test_v7_contract_replay_then_object_span_never_gets_a_third_call():
    content = "余霁公开指控搭档周岚撒谎。"
    source, request = _relationship_request_for_literal_object(content)
    bad_span = _batch_response(
        (request,), (_response(request, object_text="搭档"),)
    )
    provider = _ReplayProvider([
        SimpleNamespace(text="{invalid", prompt_tokens=1, completion_tokens=1),
        SimpleNamespace(text=bad_span, prompt_tokens=1, completion_tokens=1),
        SimpleNamespace(
            text=_batch_response(
                (request,), (_response(request, object_text="周岚"),)
            ),
            prompt_tokens=1,
            completion_tokens=1,
        ),
    ])

    run = _run_review(((content, source, request),), provider)

    assert (run.failure_reason, run.attempted_calls) == ("slot_conflict", 2)
    assert run.evaluation.evaluations[0].decisions[0].slot_conflicts == (
        "object_span",
    )
    assert len(provider.calls) == 2


def test_v7_object_span_replay_requires_exactly_one_raw_literal_occurrence():
    cases = []
    zero_content = "沈砚明确表示自己讨厌栗子糕。"
    zero_source, zero_request = _request_for_object_fact(zero_content)
    cases.append((zero_content, zero_source, zero_request, "栗子糕"))

    repeated_content = "余霁拒绝周周周。"
    repeated_source, repeated_request = _relationship_request_for_literal_object(
        repeated_content,
        key_object="周周",
    )
    cases.append(
        (repeated_content, repeated_source, repeated_request, "周周")
    )

    for content, source, request, object_text in cases:
        bad = _batch_response(
            (request,),
            (
                _response(
                    request,
                    object_text=object_text,
                    object_start_offset=999,
                    object_end_offset=1000,
                ),
            ),
        )
        provider = _ReplayProvider([
            SimpleNamespace(text=bad, prompt_tokens=1, completion_tokens=1),
            SimpleNamespace(text=bad, prompt_tokens=1, completion_tokens=1),
        ])

        run = _run_review(((content, source, request),), provider)

        assert (run.failure_reason, run.attempted_calls) == ("slot_conflict", 1)
        assert run.evaluation.evaluations[0].decisions[0].slot_conflicts == (
            "object_span",
        )
        assert len(provider.calls) == 1


def test_v7_mixed_or_semantic_results_do_not_trigger_object_span_replay():
    content = (
        "余霁公开指控周岚撒谎。\n"
        "余霁再次拒绝听周岚解释。"
    )
    target = _review_target(
        character="余霁",
        dimension="relationship_attitude",
        trait_key="trust_orientation",
        comparison_key=stable_trait_identity(
            "relationship_attitude", "trust_orientation", "周岚"
        ),
        key_object="周岚",
        baseline_hint="余霁长期信任周岚",
    )
    source, request = _request(content, target)
    assert len(request.proposals) >= 2
    mixed = json.loads(_response(request, object_text="周岚"))
    mixed["items"][0]["object_start_offset"] = 0
    mixed["items"][0]["object_end_offset"] = 1
    mixed_batch = _batch_response(
        (request,), (json.dumps(mixed, ensure_ascii=False),)
    )
    provider = _ReplayProvider([
        SimpleNamespace(text=mixed_batch, prompt_tokens=1, completion_tokens=1),
        SimpleNamespace(text=mixed_batch, prompt_tokens=1, completion_tokens=1),
    ])

    run = _run_review(((content, source, request),), provider)

    reasons = {
        decision.reason
        for decision in run.evaluation.evaluations[0].decisions
    }
    assert reasons == {"supported", "slot_conflict"}
    assert run.attempted_calls == 1
    assert len(provider.calls) == 1

    one_content = "余霁公开指控搭档周岚撒谎。"
    one_source, one_request = _relationship_request_for_literal_object(one_content)
    for overrides in (
        {"object_text": "搭档", "actor": "other"},
        {"object_text": "周岚", "verdict": "rejected", "actor": "other"},
    ):
        object_text = overrides.pop("object_text")
        raw = _batch_response(
            (one_request,),
            (_response(one_request, object_text=object_text, **overrides),),
        )
        semantic_provider = _ReplayProvider([
            SimpleNamespace(text=raw, prompt_tokens=1, completion_tokens=1),
            SimpleNamespace(text=raw, prompt_tokens=1, completion_tokens=1),
        ])
        semantic = _run_review(
            ((one_content, one_source, one_request),), semantic_provider
        )
        assert semantic.attempted_calls == 1
        assert len(semantic_provider.calls) == 1


def test_v7_scoped_axis_object_span_conflict_does_not_replay():
    definition = "救援与保全冲突时优先救人"
    scope = "救援与货物保全发生直接冲突"
    proposition = "先救人"
    target = _review_target(
        character="唐岫",
        dimension="value",
        trait_key="rescue_priority",
        comparison_key="value:救援优先",
        key_object="救援优先",
        baseline_hint="唐岫坚持救援优先",
        approved_axis_id="11111111-1111-4111-8111-111111111111",
        approved_axis_version=1,
        approved_axis_definition=definition,
        approved_axis_definition_sha256=hashlib.sha256(
            definition.encode("utf-8")
        ).hexdigest(),
        approved_axis_comparison_key="value:救援优先",
        approved_axis_applicability_scope=scope,
        approved_axis_applicability_scope_sha256=hashlib.sha256(
            scope.encode("utf-8")
        ).hexdigest(),
        axis_positive_proposition=proposition,
        axis_positive_proposition_sha256=hashlib.sha256(
            proposition.encode("utf-8")
        ).hexdigest(),
    )
    content = "唐岫明确决定先打捞货物。"
    source, request = _request(content, target)
    conflict = _batch_response(
        (request,),
        (
            _response(
                request,
                object_text="",
                object_start_offset=0,
                object_end_offset=1,
            ),
        ),
    )
    provider = _ReplayProvider([
        SimpleNamespace(text=conflict, prompt_tokens=1, completion_tokens=1),
        SimpleNamespace(text=conflict, prompt_tokens=1, completion_tokens=1),
    ])

    run = _run_review(((content, source, request),), provider)

    assert (run.failure_reason, run.attempted_calls) == ("slot_conflict", 1)
    assert run.evaluation.evaluations[0].decisions[0].slot_conflicts == (
        "object_span",
    )
    assert len(provider.calls) == 1


def test_v7_object_span_replay_rechecks_budget_and_deadline():
    content = "余霁公开指控搭档周岚撒谎。"
    source, request = _relationship_request_for_literal_object(content)
    bad = _batch_response(
        (request,), (_response(request, object_text="搭档"),)
    )
    entry = ((content, source, request),)
    probe = _run_review(
        entry,
        _ReplayProvider([
            SimpleNamespace(text=bad, prompt_tokens=0, completion_tokens=0)
        ]),
        token_budget=0,
    )
    budget_provider = _ReplayProvider([
        SimpleNamespace(text=bad, prompt_tokens=1, completion_tokens=1),
        SimpleNamespace(text=bad, prompt_tokens=1, completion_tokens=1),
    ])

    budget = _run_review(
        entry,
        budget_provider,
        token_budget=probe.estimated_tokens,
    )

    assert (budget.failure_reason, budget.attempted_calls) == ("token_budget", 1)
    assert budget.estimated_tokens == probe.estimated_tokens * 2
    assert len(budget_provider.calls) == 1

    times = iter((0.0, 0.6, 1.2, 1.8, 2.4, 3.0))
    deadline_provider = _ReplayProvider([
        SimpleNamespace(text=bad, prompt_tokens=1, completion_tokens=1),
        SimpleNamespace(text=bad, prompt_tokens=1, completion_tokens=1),
    ])
    deadline = _run_review(
        entry,
        deadline_provider,
        remaining_deadline_seconds=2,
        monotonic=lambda: next(times),
    )

    assert (deadline.failure_reason, deadline.attempted_calls) == ("deadline", 1)
    assert len(deadline_provider.calls) == 1


@pytest.mark.parametrize("dimension", ("value", "behavior_boundary"))
def test_v7_approved_scoped_axis_requires_no_literal_object_span(dimension):
    definition = "救援与保全冲突时优先救人"
    scope = "救援与货物保全发生直接冲突"
    proposition = "先救人"
    target = _review_target(
        character="唐岫",
        dimension=dimension,
        trait_key="rescue_priority",
        comparison_key=f"{dimension}:救援优先",
        key_object="救援优先",
        baseline_hint="唐岫坚持救援优先",
        approved_axis_id="11111111-1111-4111-8111-111111111111",
        approved_axis_version=1,
        approved_axis_definition=definition,
        approved_axis_definition_sha256=hashlib.sha256(
            definition.encode("utf-8")
        ).hexdigest(),
        approved_axis_comparison_key=f"{dimension}:救援优先",
        approved_axis_applicability_scope=scope,
        approved_axis_applicability_scope_sha256=hashlib.sha256(
            scope.encode("utf-8")
        ).hexdigest(),
        axis_positive_proposition=proposition,
        axis_positive_proposition_sha256=hashlib.sha256(
            proposition.encode("utf-8")
        ).hexdigest(),
    )
    content = "唐岫明确决定先打捞货物。"
    source, request = _request(content, target)

    accepted = evaluate_target_bound_draft_review(
        request,
        _response(request, object_text=""),
        expected_source=source,
        frozen_content=content,
    )
    assert accepted.decisions[0].verdict == "supported"
    assert accepted.decisions[0].object_relation == "not_applicable"
    assert accepted.decisions[0].observed_object is None
    assert accepted.decisions[0].object_basis_id is None

    literal_object = evaluate_target_bound_draft_review(
        request,
        _response(
            request,
            object_text="",
            object_relation="narrower",
            object_start_offset=0,
            object_end_offset=1,
        ),
        expected_source=source,
        frozen_content=content,
    )
    assert literal_object.decisions[0].verdict == "uncertain"
    assert literal_object.decisions[0].slot_conflicts == (
        "object_relation",
        "object_span",
    )


def test_v7_non_scoped_value_keeps_literal_object_and_span_gate():
    target = _review_target(
        character="唐岫",
        dimension="value",
        trait_key="rescue_priority",
        comparison_key="value:货物",
        key_object="货物",
        baseline_hint="唐岫坚持货物优先",
    )
    content = "唐岫明确决定先打捞货物。"
    source, request = _request(content, target)

    accepted = evaluate_target_bound_draft_review(
        request,
        _response(request, object_text="货物"),
        expected_source=source,
        frozen_content=content,
    )
    assert accepted.decisions[0].verdict == "supported"
    assert accepted.decisions[0].observed_object == "货物"

    missing = evaluate_target_bound_draft_review(
        request,
        _response(request, object_text=""),
        expected_source=source,
        frozen_content=content,
    )
    assert missing.decisions[0].verdict == "uncertain"
    assert missing.decisions[0].slot_conflicts == (
        "object_relation",
        "object_span",
    )


def test_v3_whitelists_explicit_named_narrator_and_strict_zero_subject():
    direct = "沈砚明确表示自己讨厌栗子糕。"
    source = _source(direct)
    _, proposals = build_target_bound_draft_proposals(
        direct, source, _review_target(), line_start=1, line_end=1,
        protocol_version="v3",
    )
    assert {item.binding_kind for item in proposals} == {"explicit_named_subject"}

    narrator = "旁白明确说明沈砚现在讨厌栗子糕。"
    source = _source(narrator)
    _, proposals = build_target_bound_draft_proposals(
        narrator, source, _review_target(), line_start=1, line_end=1,
        protocol_version="v3",
    )
    assert {item.binding_kind for item in proposals} == {
        "narrator_explicit_attribution"
    }

    adjacent = "沈砚走进餐厅。\n随后明确拒绝栗子糕。"
    source = _source(adjacent)
    _, proposals = build_target_bound_draft_proposals(
        adjacent, source, _review_target(), line_start=1, line_end=2,
        protocol_version="v3",
    )
    assert any(item.binding_kind == "adjacent_zero_subject" for item in proposals)


def test_v3_named_actor_is_not_rejected_by_joint_word_inside_its_object():
    content = "余霁又拒绝把共同保管的钥匙交给周岚。"
    target = _review_target(
        character="余霁",
        dimension="relationship_attitude",
        trait_key="trust_orientation",
        comparison_key=stable_trait_identity(
            "relationship_attitude", "trust_orientation", "周岚"
        ),
        key_object="周岚",
        baseline_hint="余霁长期信任周岚",
    )
    source = _source(content)

    _, proposals = build_target_bound_draft_proposals(
        content, source, target, line_start=1, line_end=1,
        protocol_version="v3",
    )

    assert [(item.binding_kind, item.canonical_statement) for item in proposals] == [
        ("explicit_named_subject", "余霁又拒绝把共同保管的钥匙交给周岚")
    ]


@pytest.mark.parametrize(
    "content",
    (
        "余霁和周岚一起拒绝交出钥匙。",
        "余霁与周岚共同拒绝交出钥匙。",
        "余霁共同拒绝交出钥匙。",
    ),
)
def test_v3_true_joint_named_subject_remains_outside_the_whitelist(content):
    target = _review_target(
        character="余霁",
        dimension="relationship_attitude",
        trait_key="trust_orientation",
        comparison_key=stable_trait_identity(
            "relationship_attitude", "trust_orientation", "周岚"
        ),
        key_object="周岚",
        baseline_hint="余霁长期信任周岚",
    )
    source = _source(content)

    _, proposals = build_target_bound_draft_proposals(
        content, source, target, line_start=1, line_end=1,
        protocol_version="v3",
    )

    assert proposals == ()


_SAFE_ZERO_SUBJECT_LEADS = (
    "又", "随后", "接着", "然后", "继而", "仍", "却", "再次", "并",
    "还", "才", "终于", "明确", "直接", "同时", "立即", "马上", "转而",
    "最终", "依然", "继续", "先", "再",
)
_OVERT_ZERO_SUBJECT_PRONOUNS = (
    "我", "我们", "咱", "咱们", "你", "你们", "您", "您们",
    "他", "他们", "她", "她们", "它", "它们", "祂", "祂们",
    "其", "本人", "自己",
)


@pytest.mark.parametrize("lead", _SAFE_ZERO_SUBJECT_LEADS)
def test_v3_all_safe_leads_still_admit_a_genuinely_zero_subject_clause(lead):
    content = f"沈砚走进餐厅。\n{lead}明确拒绝栗子糕。"
    source = _source(content)
    _, proposals = build_target_bound_draft_proposals(
        content, source, _review_target(), line_start=1, line_end=2,
        protocol_version="v3",
    )
    adjacent = tuple(
        item for item in proposals if item.binding_kind == "adjacent_zero_subject"
    )
    assert len(adjacent) == 1
    assert adjacent[0].canonical_statement == f"沈砚{lead}明确拒绝栗子糕"


def test_v3_every_safe_lead_rejects_overt_singular_and_plural_pronouns():
    for lead in _SAFE_ZERO_SUBJECT_LEADS:
        for pronoun in _OVERT_ZERO_SUBJECT_PRONOUNS:
            content = f"沈砚走进餐厅。\n{lead} {pronoun}明确拒绝栗子糕。"
            source = _source(content)
            _, proposals = build_target_bound_draft_proposals(
                content, source, _review_target(), line_start=1, line_end=2,
                protocol_version="v3",
            )
            adjacent = tuple(
                item
                for item in proposals
                if item.binding_kind == "adjacent_zero_subject"
            )
            assert adjacent == (), (lead, pronoun, adjacent)
            assert all(
                item.canonical_statement != f"沈砚{lead} {pronoun}明确拒绝栗子糕"
                for item in proposals
            )


@pytest.mark.parametrize(
    "intruding_clause",
    (
        "随后唐岫吃下栗子糕。",
        "随后陌生旅客带走栗子糕。",
        "随后唐岫喜欢栗子糕。",
        "随后阿橙拒绝栗子糕。",
        "随后灰斗篷拒绝栗子糕。",
    ),
)
def test_v3_zero_subject_whitelist_rejects_named_and_unknown_noun_subjects(
    intruding_clause,
):
    content = f"沈砚走进餐厅。\n{intruding_clause}"
    source = _source(content)
    _, proposals = build_target_bound_draft_proposals(
        content, source, _review_target(), line_start=1, line_end=2,
        protocol_version="v3",
    )
    assert not any(
        item.binding_kind == "adjacent_zero_subject" for item in proposals
    )


@pytest.mark.parametrize(
    "continuation",
    (
        "随后吃下栗子糕。",
        "随后带走栗子糕。",
        "随后喜欢栗子糕。",
        "随后明确表示讨厌栗子糕。",
        "随后主动拒绝栗子糕。",
        "随后把栗子糕带走。",
    ),
)
def test_v3_zero_subject_whitelist_keeps_bounded_omitted_subject_predicates(
    continuation,
):
    content = f"沈砚走进餐厅。\n{continuation}"
    source = _source(content)
    _, proposals = build_target_bound_draft_proposals(
        content, source, _review_target(), line_start=1, line_end=2,
        protocol_version="v3",
    )
    adjacent = tuple(
        item for item in proposals if item.binding_kind == "adjacent_zero_subject"
    )
    assert len(adjacent) == 1
    assert adjacent[0].canonical_statement == "沈砚" + continuation[:-1]


@pytest.mark.parametrize(
    "intruding_clause",
    (
        "随后她明确拒绝栗子糕。",
        "随后唐岫吃下栗子糕。",
        "随后陌生旅客带走栗子糕。",
    ),
)
def test_v3_reviewer_cannot_inject_supported_nonzero_subject_certificate(
    intruding_clause,
):
    content = f"沈砚走进餐厅。\n{intruding_clause}"
    source, request = _request(content)
    assert {item.binding_kind for item in request.proposals} == {
        "explicit_named_subject"
    }
    assert all(item.fact_clause_id == "L1:A1" for item in request.proposals)

    # Even if a reviewer reports a fabricated adjacent item as supported, its
    # ID was never issued by the server whitelist. Exact response-set binding
    # therefore invalidates the whole response before a certificate exists.
    payload = json.loads(_response(request, object_text="餐厅"))
    injected = dict(payload["items"][0])
    injected["proposal_id"] = "tdp_" + "f" * 32
    payload["items"].append(injected)
    evaluation = evaluate_target_bound_draft_review(
        request,
        json.dumps(payload, ensure_ascii=False),
        expected_source=source,
        frozen_content=content,
    )
    assert all(
        decision.verdict == "uncertain"
        and decision.reason == "response_mismatch"
        and decision.target_digest is None
        for decision in evaluation.decisions
    )


def test_v3_actor_boundary_rechecks_restored_unsafe_alias_and_supported_response():
    valid_target = _review_target(authorized_aliases=("砚先生",))
    target_payload = valid_target.model_dump(mode="python")
    target_payload["authorized_aliases"] = ("她",)
    # model_construct models a restored/tampered in-memory instance that did
    # not pass the normal Pydantic validator. The proposal boundary must still
    # fail closed for the pronoun sentence.
    unsafe_target = TargetBoundDraftReviewTarget.model_construct(**target_payload)
    unsafe_content = "她明确表示自己讨厌栗子糕。"
    unsafe_source = _source(unsafe_content)
    _, unsafe_proposals = build_target_bound_draft_proposals(
        unsafe_content,
        unsafe_source,
        unsafe_target,
        line_start=1,
        line_end=1,
        protocol_version="v3",
    )
    assert unsafe_proposals == ()

    canonical_payload = valid_target.model_dump(mode="python")
    canonical_payload.update({"character": "她", "authorized_aliases": ()})
    unsafe_canonical = TargetBoundDraftReviewTarget.model_construct(
        **canonical_payload
    )
    _, canonical_proposals = build_target_bound_draft_proposals(
        unsafe_content,
        unsafe_source,
        unsafe_canonical,
        line_start=1,
        line_end=1,
        protocol_version="v3",
    )
    assert canonical_proposals == ()

    content = "砚先生明确表示自己讨厌栗子糕。"
    source, request = _request(content, valid_target)
    unsafe_request = request.model_copy(update={
        "target": unsafe_target,
        "target_digest": target_bound_draft_target_digest(unsafe_target),
    })
    evaluation = evaluate_target_bound_draft_review(
        unsafe_request,
        _response(unsafe_request),
        expected_source=source,
        frozen_content=content,
    )
    assert all(
        decision.verdict == "uncertain"
        and decision.reason == "source_mismatch"
        and decision.target_digest is None
        for decision in evaluation.decisions
    )


@pytest.mark.parametrize(
    "content",
    (
        "唐岫明确表示自己讨厌栗子糕。",
        "沈砚在梦境中明确表示自己讨厌栗子糕。",
        "如果宴会继续，沈砚明确表示自己讨厌栗子糕。",
        "沈砚写道：“我讨厌栗子糕”。",
        "沈砚让唐岫拒绝栗子糕。",
    ),
)
def test_v3_structural_whitelist_excludes_wrong_actor_quote_unreal_and_condition(content):
    source = _source(content)
    _, proposals = build_target_bound_draft_proposals(
        content, source, _review_target(), line_start=1, line_end=1,
        protocol_version="v3",
    )
    assert proposals == ()


def test_v3_binds_actual_object_slice_and_fails_closed_on_semantic_or_digest_errors():
    content = "沈砚明确表示自己讨厌栗子糕。"
    source, request = _request(content)
    decision = evaluate_target_bound_draft_review(
        request,
        _response(request),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (decision.verdict, decision.observed_object) == ("supported", "栗子糕")
    assert decision.object_relation == "narrower"
    assert decision.object_basis_id == request.proposals[0].fact_clause_id

    for overrides, reason in (
        ({"actor": "other", "verdict": "supported"}, "slot_conflict"),
        ({"actuality": "hypothetical"}, "slot_conflict"),
        ({"axis_relation": "different"}, "slot_conflict"),
        ({"object_relation": "different"}, "slot_conflict"),
        ({"polarity_relation": "opposite"}, "slot_conflict"),
        ({"correction_relation": "corrected"}, "slot_conflict"),
        ({"object_start_offset": 999, "object_end_offset": 1000}, "slot_conflict"),
    ):
        value = evaluate_target_bound_draft_review(
            request,
            _response(request, **overrides),
            expected_source=source,
            frozen_content=content,
        ).decisions[0]
        assert (value.verdict, value.reason) == ("uncertain", reason)

    uncertain = evaluate_target_bound_draft_review(
        request,
        _response(
            request,
            verdict="uncertain",
            actor="ambiguous",
            actuality="ambiguous",
            statement_relation="ambiguous",
            axis_relation="ambiguous",
            object_relation="ambiguous",
            polarity_relation="ambiguous",
            correction_relation="ambiguous",
            basis_ids=[],
        ),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (uncertain.verdict, uncertain.reason) == (
        "uncertain", "reviewer_uncertain"
    )

    payload = json.loads(_response(request))
    payload["request_digest"] = "0" * 64
    mismatch = evaluate_target_bound_draft_review(
        request,
        json.dumps(payload, ensure_ascii=False),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert mismatch.reason == "response_mismatch"

    source_mismatch = evaluate_target_bound_draft_review(
        request,
        _response(request),
        expected_source=source,
        frozen_content=content + "被篡改",
    ).decisions[0]
    assert source_mismatch.reason == "source_mismatch"


def test_v3_rejected_item_may_omit_irrelevant_object_offsets_without_poisoning_batch():
    content = (
        "沈砚明确表示自己讨厌栗子糕。\n"
        "沈砚随后明确表示自己仍然喜欢栗子糕。"
    )
    source, request = _request(content)
    nested = json.loads(_response(request))
    assert len(nested["items"]) == 2
    rejected = nested["items"][1]
    rejected.update({
        "verdict": "rejected",
        "polarity_relation": "opposite",
    })
    rejected.pop("object_start_offset")
    rejected.pop("object_end_offset")
    raw = json.dumps({
        "schema_version": TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V3,
        "batch_digest": target_bound_draft_review_batch_digest((request,)),
        "responses": [nested],
    }, ensure_ascii=False)

    evaluation = evaluate_target_bound_draft_review_batch(
        (request,),
        raw,
        expected_sources=(source,),
        frozen_contents=(content,),
    ).evaluations[0]

    assert [(row.verdict, row.reason) for row in evaluation.decisions] == [
        ("supported", "supported"),
        ("rejected", "reviewer_rejected"),
    ]


def test_v3_supported_item_without_object_offsets_still_invalidates_whole_batch():
    content = "沈砚明确表示自己讨厌栗子糕。"
    source, request = _request(content)
    nested = json.loads(_response(request))
    nested["items"][0].pop("object_start_offset")
    nested["items"][0].pop("object_end_offset")
    raw = json.dumps({
        "schema_version": TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V3,
        "batch_digest": target_bound_draft_review_batch_digest((request,)),
        "responses": [nested],
    }, ensure_ascii=False)

    evaluation = evaluate_target_bound_draft_review_batch(
        (request,),
        raw,
        expected_sources=(source,),
        frozen_contents=(content,),
    ).evaluations[0]

    assert all(
        row.verdict == "uncertain" and row.reason == "response_invalid"
        for row in evaluation.decisions
    )


def test_v3_flattened_proposals_as_responses_fail_closed_for_whole_batch():
    content = "沈砚明确表示自己讨厌栗子糕。"
    source, request = _request(content)
    nested = json.loads(_response(request))
    # This mirrors the observed provider failure: proposal items were flattened
    # into top-level responses and one enum was invented.  The strict parser
    # must reject the complete batch rather than guessing a window grouping.
    flattened = []
    for item in nested["items"]:
        flattened.append({
            **item,
            "request_digest": nested["request_digest"],
            "semantic_verdict": "direct_conflict",
        })
    raw = json.dumps({
        "schema_version": TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V3,
        "batch_digest": target_bound_draft_review_batch_digest((request,)),
        "responses": flattened,
    }, ensure_ascii=False)

    evaluation = evaluate_target_bound_draft_review_batch(
        (request,),
        raw,
        expected_sources=(source,),
        frozen_contents=(content,),
    ).evaluations[0]

    assert all(
        row.verdict == "uncertain" and row.reason == "response_invalid"
        for row in evaluation.decisions
    )


class _V3Provider:
    def __init__(
        self,
        *,
        mode="supported",
        object_text="栗子糕",
        signal_records=None,
    ):
        self.mode = mode
        self.object_text = object_text
        self.signal_records = signal_records
        self.signal_calls = 0
        self.review_calls = 0

    def complete(self, system: str, user: str):
        if system not in {
            TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V7,
            TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V8,
        }:
            self.signal_calls += 1
            return SimpleNamespace(
                text=json.dumps({"records": self.signal_records or []}),
                prompt_tokens=5,
                completion_tokens=2,
            )
        self.review_calls += 1
        if self.mode == "timeout":
            raise ProviderError("private timeout", category="read_timeout")
        if self.mode == "invalid":
            return SimpleNamespace(text="{}", prompt_tokens=8, completion_tokens=2)
        payload = json.loads(user.split("：\n", 1)[1])
        responses = []
        for window in payload["windows"]:
            request = window["request"]
            clauses = {
                clause["support_id"]: clause
                for line in request["lines"]
                for clause in line["clauses"]
            }
            items = []
            for proposal, hint in zip(
                request["proposals"], window["basis_path_hints"]
            ):
                fact = clauses[proposal["fact_clause_id"]]["text"]
                has_object = self.object_text in fact
                start = fact.index(self.object_text) if has_object else None
                scoped_axis = (
                    request["target"]["dimension"] in {"value", "behavior_boundary"}
                    and request["target"]["approved_axis_comparison_key"] is not None
                )
                items.append({
                    "proposal_id": proposal["proposal_id"],
                    "verdict": "supported" if has_object else "rejected",
                    "actor": "proposed",
                    "actuality": "asserted",
                    "statement_relation": "supported",
                    "axis_relation": (
                        "matches_scoped_axis"
                        if has_object and scoped_axis
                        else "matches_target"
                        if has_object
                        else "different"
                    ),
                    "object_relation": (
                        "not_applicable"
                        if has_object and scoped_axis
                        else "narrower" if has_object else "different"
                    ),
                    "polarity_relation": "requested",
                    "correction_relation": "none",
                    "observation_kind": (
                        "decision"
                        if request["target"]["dimension"] != "preference"
                        else "preference_expression"
                    ),
                    "object_start_offset": None if scoped_axis else start,
                    "object_end_offset": (
                        start + len(self.object_text)
                        if start is not None and not scoped_axis else None
                    ),
                    "basis_ids": hint["required_basis_ids"],
                })
            responses.append({
                "schema_version": request["schema_version"],
                "request_digest": window["request_digest"],
                "items": items,
            })
        return SimpleNamespace(
            text=json.dumps({
                "schema_version": (
                    TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V4
                    if responses[0]["schema_version"]
                    == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4
                    else TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V3
                ),
                "batch_digest": payload["batch_digest"],
                "responses": responses,
            }, ensure_ascii=False),
            prompt_tokens=15,
            completion_tokens=10,
        )


def _settings(**updates) -> Settings:
    values = {
        "openai_api_key": "v3-test",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock",
        "enable_character_consistency": True,
        "character_signal_full_line_prompt_v2": True,
        "character_draft_actor_review_v1": True,
        "character_target_bound_draft_review_v3": True,
        "character_signal_package_max_attempts": 1,
        "character_signal_max_attempts": 1,
        "provider_max_attempts": 1,
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def _signal_target(**updates) -> CharacterSignalTarget:
    values = {
        "character": "沈砚",
        "dimension": "preference",
        "trait_key": "food_preference",
        "comparison_key": stable_trait_identity(
            "preference", "food_preference", "甜食"
        ),
        "key_object": "甜食",
        "baseline_polarity": "positive",
        "requested_polarity": "negative",
        "baseline_hint": "沈砚长期喜欢甜食",
    }
    values.update(updates)
    return CharacterSignalTarget(**values)


def _extract(content: str, provider: _V3Provider, target=None):
    return CharacterSignalExtractor(provider, settings=_settings()).extract_targeted(
        CharacterSignalChunk("draft-v3", "draft.md", content, 1, "draft"),
        (target or _signal_target(),),
        target_ordinal=1,
        source_identity=_source(content),
        frozen_content=content,
    )


def test_v3_clean_empty_recovery_preserves_actual_object_in_signal():
    content = "沈砚明确表示自己讨厌栗子糕。"
    provider = _V3Provider()
    result = _extract(content, provider)

    assert (provider.signal_calls, provider.review_calls) == (1, 1)
    assert result.diagnostics.outcome == "completed"
    assert len(result.draft_observations) == 1
    signal = result.draft_observations[0]
    assert signal.key_object == "栗子糕"
    assert signal.key_object != "甜食"
    assert signal.target_bound_object_relation == "narrower"
    assert signal.target_bound_object_basis_id == "L1:A1"


@pytest.mark.parametrize(
    ("signal_records", "generation_reason"),
    (
        ([{"unexpected": True}], "schema_validation"),
        (
            [
                {
                    "character": "沈砚",
                    "dimension": "preference",
                    "trait_key": "food_preference",
                    "statement": "沈砚明确表示自己讨厌不存在的甜点",
                    "polarity": "negative",
                    "stability": "temporary",
                    "observation_kind": "preference_expression",
                    "context": "",
                    "key_object": "栗子糕",
                    "source_line_start": 1,
                    "source_line_end": 1,
                    "evidence": "沈砚明确表示自己讨厌栗子糕。",
                }
            ],
            "statement_support",
        ),
    ),
)
def test_v3_complete_rejections_supersede_record_local_generation_noise(
    signal_records,
    generation_reason,
):
    content = "沈砚明确表示自己讨厌栗子糕。"
    provider = _V3Provider(
        object_text="不存在的对象",
        signal_records=signal_records,
    )

    result = _extract(content, provider)

    assert (provider.signal_calls, provider.review_calls) == (1, 1)
    assert result.diagnostics.outcome == "completed"
    assert result.draft_observations == ()
    assert result.diagnostics.reason_counts["target_bound_review_rejected"] > 0
    assert generation_reason not in result.diagnostics.reason_counts


def test_v3_unresolved_discovery_does_not_hide_generation_noise():
    content = "沈砚明确表示自己讨厌栗子糕。"
    provider = _V3Provider(
        mode="invalid",
        signal_records=[{"unexpected": True}],
    )

    result = _extract(content, provider)

    assert result.diagnostics.outcome == "partial"
    assert result.draft_observations == ()
    assert result.diagnostics.reason_counts["semantic_binding_unresolved"] > 0
    assert result.diagnostics.reason_counts["schema_validation"] > 0


@pytest.mark.parametrize(
    "deferred_field",
    ("draft_actor_deferred", "target_bound_draft_deferred"),
)
def test_v3_complete_discovery_never_clears_unmapped_deferred_debt(
    deferred_field,
):
    debt = object()
    validation = _ValidatedSignalPackage(
        parsed=True,
        raw_records=1,
        rejected_records=1,
        reason_counts={"statement_support": 1},
        failures=(_SignalValidationFailure(0, "statement_support"),),
        **{deferred_field: (debt,)},
    )

    resolved = _resolve_completed_target_bound_discovery(
        validation,
        _TargetBoundDraftOutcome(rejected_count=1),
        discovery_count=1,
        discovery_truncated=0,
    )

    assert resolved is validation
    assert getattr(resolved, deferred_field) == (debt,)
    assert resolved.reason_counts == {"statement_support": 1}


def test_v3_recovers_preference_object_only_from_round_trip_verified_frozen_key():
    content = "沈砚明确表示自己讨厌栗子糕。"
    target = _signal_target(key_object="")

    result = _extract(content, _V3Provider(), target)

    assert result.diagnostics.outcome == "completed"
    assert len(result.draft_observations) == 1
    signal = result.draft_observations[0]
    assert signal.key_object == "栗子糕"
    assert signal.target_bound_target is not None
    assert signal.target_bound_target.key_object == "甜食"


@pytest.mark.parametrize(
    "target",
    (
        _signal_target(
            key_object="",
            comparison_key="preference:甜 食",
        ),
        _signal_target(
            dimension="value",
            trait_key="rescue_before_goods_priority",
            comparison_key="value:先救人再保货",
            key_object="",
            baseline_hint="沈砚坚持先救人再保货",
        ),
    ),
)
def test_v3_never_guesses_missing_object_from_nonreversible_or_unapproved_target(
    target,
):
    with pytest.raises(ValidationError, match="target_bound_object_required"):
        _target_bound_review_target(target, target_ordinal=1)


@pytest.mark.parametrize("mode,reason", (
    ("timeout", "target_bound_provider_timeout"),
    ("invalid", "target_bound_response_invalid"),
))
def test_v3_provider_failures_are_partial_and_never_create_signal(mode, reason):
    result = _extract(
        "沈砚明确表示自己讨厌栗子糕。",
        _V3Provider(mode=mode),
    )
    assert result.diagnostics.outcome == "partial"
    assert result.draft_observations == ()
    assert result.diagnostics.reason_counts["semantic_binding_unresolved"] == 1
    assert result.diagnostics.reason_counts[reason] == 1


def test_v3_scoped_value_axis_is_bound_without_copying_baseline_object():
    definition = "生命与货物冲突时优先救人"
    scope = "救援与保全发生直接冲突"
    proposition = "先救人"
    target = _signal_target(
        character="唐岫",
        dimension="value",
        trait_key="rescue_priority",
        comparison_key="value:救援优先",
        key_object="",
        baseline_hint="唐岫坚持救援优先",
        approved_axis_id="11111111-1111-4111-8111-111111111111",
        approved_axis_version=1,
        approved_axis_definition=definition,
        approved_axis_definition_sha256=hashlib.sha256(definition.encode()).hexdigest(),
        approved_axis_comparison_key="value:救援优先",
        approved_axis_applicability_scope=scope,
        approved_axis_applicability_scope_sha256=hashlib.sha256(scope.encode()).hexdigest(),
        axis_positive_proposition=proposition,
        axis_positive_proposition_sha256=hashlib.sha256(proposition.encode()).hexdigest(),
    )
    content = "唐岫明确要求先打捞货物，随后才允许救人。"
    result = _extract(content, _V3Provider(object_text="先打捞货物"), target)
    assert result.diagnostics.outcome == "completed"
    assert len(result.draft_observations) == 1
    observation = result.draft_observations[0]
    assert observation.key_object == ""
    assert observation.target_bound_target is not None
    assert observation.target_bound_target.key_object == "救援优先"
    assert observation.target_bound_object_relation == "not_applicable"
    assert observation.target_bound_object_basis_id is None
    assert _signal_matches_target(observation, target)


def _certified_signal(
    target: TargetBoundDraftReviewTarget,
    *,
    key_object: str,
    signal_id: str = "a",
) -> CharacterSignal:
    return CharacterSignal(
        id="cs_" + signal_id * 32,
        character=target.character,
        dimension=target.dimension,
        trait_key=target.trait_key,
        statement=f"{target.character}明确背离既有设定",
        polarity=target.requested_polarity,
        stability="temporary",
        observation_kind=(
            "preference_expression"
            if target.dimension == "preference"
            else "explicit_declaration"
        ),
        key_object=key_object,
        source_kind="draft",
        evidence=EvidenceSpan(
            document_id="draft-v3",
            document_name="draft.md",
            line_start=1,
            line_end=1,
            text=f"{target.character}明确背离既有设定。",
        ),
        target_bound_target_digest=target_bound_draft_target_digest(target),
        target_bound_target=target,
        target_bound_target_ordinal=target.target_ordinal,
        target_bound_object_relation="narrower",
        target_bound_object_basis_id="L1:A1",
    )


def _confirmed_baseline(
    *,
    dimension="preference",
    trait_key="food_preference",
    key_object="甜食",
) -> ConfirmedTraitSnapshot:
    return ConfirmedTraitSnapshot(
        id="ct_target_bound_v3",
        character="沈砚",
        dimension=dimension,
        trait_key=trait_key,
        key_object=key_object,
        statement=f"沈砚长期坚持{key_object}",
        polarity="positive",
        stability="stable",
        origin="explicit_setting",
        evidence=(EvidenceSpan(
            document_id="canon",
            document_name="canon.md",
            line_start=1,
            line_end=1,
            text=f"沈砚长期坚持{key_object}。",
        ),),
    )


def test_v3_certificate_bypasses_only_the_exact_frozen_baseline_object_key():
    target = _review_target()
    certified = _certified_signal(target, key_object="栗子糕")
    baseline = _confirmed_baseline()
    entry = (
        SimpleNamespace(payload={"comparison_key": target.comparison_key}),
        baseline,
        None,
        "沈砚",
    )
    assert _observation_matches_baseline(entry, certified) is True

    plain = CharacterSignal.model_validate({
        **certified.model_dump(mode="python"),
        "id": "cs_" + "b" * 32,
    })
    assert _observation_matches_baseline(entry, plain) is False

    other_entry = (
        SimpleNamespace(payload={"comparison_key": "preference:咸食"}),
        baseline.model_copy(update={"key_object": "咸食"}),
        None,
        "沈砚",
    )
    assert _observation_matches_baseline(other_entry, certified) is False

    with pytest.raises(ValidationError, match="certificate is invalid"):
        CharacterSignal(
            **{
                **certified.model_dump(mode="python"),
                "id": "cs_" + "c" * 32,
                "target_bound_target_digest": "0" * 64,
                "target_bound_target": target,
                "target_bound_target_ordinal": target.target_ordinal,
                "target_bound_object_relation": "narrower",
                "target_bound_object_basis_id": "L1:A1",
            }
        )


def test_v3_certificate_keeps_actual_relationship_object_through_drift_matching():
    trait_key = "trust_orientation"
    baseline_object = "北境守备军"
    comparison_key = stable_trait_identity(
        "relationship_attitude", trait_key, baseline_object
    )
    target = _review_target(
        dimension="relationship_attitude",
        trait_key=trait_key,
        comparison_key=comparison_key,
        key_object=baseline_object,
        requested_polarity="negative",
        baseline_hint="沈砚长期信任北境守备军",
    )
    certified = _certified_signal(target, key_object="第三小队")
    baseline = _confirmed_baseline(
        dimension="relationship_attitude",
        trait_key=trait_key,
        key_object=baseline_object,
    )
    case = CharacterDriftCase(
        id="cdc_target_bound_v3",
        baseline=baseline,
        observations=(certified,),
        scope_compatibility="compatible",
    )
    assert prepare_character_drift(case).matching_observations == (certified,)
    entry = (
        SimpleNamespace(payload={"comparison_key": comparison_key}),
        baseline,
        None,
        "沈砚",
    )
    assert certified.key_object == "第三小队"
    assert _explanation_observation_key_object(entry, certified) == baseline_object
    # Projection is summary-only: source fact and certificate remain intact.
    assert certified.key_object == "第三小队"
    assert certified.target_bound_target == target

    plain = CharacterSignal.model_validate({
        **certified.model_dump(mode="python"),
        "id": "cs_" + "d" * 32,
    })
    assert prepare_character_drift(
        case.model_copy(update={"observations": (plain,)})
    ).matching_observations == ()
    assert _explanation_observation_key_object(entry, plain) == "第三小队"

    candidate = ExplanationCandidate(
        citation="E01",
        source_kind="published_history",
        publication_status="published",
        authority_tier="formal_record",
        resolution_state="confirmed",
        source_ordinal=0,
        eligible_draft_document_ids=(certified.evidence.document_id,),
        evidence=EvidenceSpan(
            document_id="history-v3",
            document_name="history.md",
            line_start=1,
            line_end=1,
            text="沈砚此前取得了新的确凿反证。",
        ),
    )
    summary = ExplanationObservationSummary(
        citation="C01",
        observation_id=certified.id,
        statement=certified.statement,
        key_object=_explanation_observation_key_object(entry, certified),
        document_id=certified.evidence.document_id,
        source_ordinal=1,
        evidence=certified.evidence.model_copy(deep=True),
    )
    _, user_prompt = build_character_explanation_review_prompts(
        (candidate,),
        baseline=ExplanationBaselineSummary(
            character=baseline.character,
            dimension=baseline.dimension,
            trait_key=baseline.trait_key,
            key_object=baseline.key_object,
            statement=baseline.statement,
        ),
        observations=(summary,),
    )
    assert '"key_object":"北境守备军"' in user_prompt


@pytest.mark.parametrize(
    ("dimension", "trait_key", "baseline_object"),
    (
        ("relationship_attitude", "trust_orientation", "周岚"),
        ("motivation_goal", "pursuit_commitment", "重建海灯塔"),
    ),
)
def test_v3_certificate_actual_objects_survive_final_repeated_behavior_gate(
    dimension: str,
    trait_key: str,
    baseline_object: str,
):
    comparison_key = stable_trait_identity(
        dimension, trait_key, baseline_object
    )
    target = _review_target(
        character="余霁",
        dimension=dimension,
        trait_key=trait_key,
        comparison_key=comparison_key,
        key_object=baseline_object,
        requested_polarity="negative",
        baseline_hint=f"余霁长期坚持{baseline_object}",
    )
    first = _certified_signal(
        target, key_object=f"搭档{baseline_object}", signal_id="e"
    ).model_copy(
        update={
            "statement": "余霁第一次明确背离既有设定",
            "evidence": EvidenceSpan(
                document_id="draft-v3",
                document_name="draft.md",
                line_start=14,
                line_end=14,
                text="余霁第一次明确背离既有设定。",
            ),
            "target_bound_object_basis_id": "L14:A1",
        }
    )
    second = _certified_signal(
        target, key_object=f"与{baseline_object}的共同约定", signal_id="f"
    ).model_copy(
        update={
            "statement": "次日，余霁再次明确背离既有设定",
            "evidence": EvidenceSpan(
                document_id="draft-v3",
                document_name="draft.md",
                line_start=15,
                line_end=15,
                text="次日，余霁再次明确背离既有设定。",
            ),
            "target_bound_object_basis_id": "L15:A1",
        }
    )
    case = CharacterDriftCase(
        id="cdc_target_bound_v3_repeated",
        baseline=_confirmed_baseline(
            dimension=dimension,
            trait_key=trait_key,
            key_object=baseline_object,
        ).model_copy(update={"character": "余霁"}),
        observations=(first, second),
        scope_compatibility="compatible",
        material_coverage="complete",
        explanation_coverage="complete",
    )
    prepared = prepare_character_drift(case)
    assert prepared.reason == "two_independent_behaviors"

    review = CharacterReviewResult(
        decision=ModelDriftDecision(
            verdict="contradicts",
            explanation="两次独立事件均与长期基线相反。",
            citations=("B01", "C01", "C02"),
            event_independence="yes",
            independent_event_citations=("C01", "C02"),
        ),
        event_identity_verification=EventIdentityDecision(
            relation="different_events",
            explanation="第二段以次日建立明确时间边界。",
            citations=("C01", "C02"),
        ),
        diagnostics=CharacterReviewDiagnostics(
            outcome="completed", reason="completed", attempted_calls=2
        ),
    )
    promoted = promote_character_drift(prepared, review)
    assert promoted.outcome == "conflict"
    assert promoted.reason == "model_contradicts"
