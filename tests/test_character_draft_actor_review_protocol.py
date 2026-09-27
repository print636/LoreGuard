from __future__ import annotations

import hashlib
import json

import pytest

from app.character_draft_actor_review import (
    DRAFT_ACTOR_REVIEW_BATCH_SCHEMA_V1,
    DRAFT_ACTOR_REVIEW_SCHEMA_V1,
    DraftActorProposal,
    build_draft_actor_clause_index,
    build_draft_actor_review_request,
    draft_actor_review_batch_digest,
    draft_actor_review_request_digest,
    evaluate_draft_actor_review,
    evaluate_draft_actor_review_batch,
    required_draft_actor_review_basis_ids,
    verify_draft_actor_review_source,
)
from app.character_scope_review import ScopeReviewSourceIdentity


def _source(content: str, suffix: str = "1") -> ScopeReviewSourceIdentity:
    return ScopeReviewSourceIdentity(
        run_input_id=f"run-{suffix}",
        document_id=f"draft-{suffix}",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _same_line_request(suffix: str = "1"):
    content = "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    source = _source(content, suffix)
    index = build_draft_actor_clause_index(
        content, source, line_start=1, line_end=1
    )
    proposal = DraftActorProposal(
        source_sha256=source.content_sha256,
        line_start=1,
        line_end=1,
        evidence=content,
        target_clause_id="L1:A1",
        actor_anchor_id="L1:A2",
        anchor_kind="same_line_corroboration",
        character="林澈",
        statement="林澈把原稿改成红色",
    )
    request = build_draft_actor_review_request(
        index, (proposal,), frozen_content=content, expected_source=source
    )
    return content, source, request


def _successor_request(successor: str, suffix: str):
    first = "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    content = f"{first}\n{successor}"
    source = _source(content, suffix)
    index = build_draft_actor_clause_index(
        content, source, line_start=1, line_end=1
    )
    proposal = DraftActorProposal(
        source_sha256=source.content_sha256,
        line_start=1,
        line_end=1,
        evidence=first,
        target_clause_id="L1:A1",
        actor_anchor_id="L1:A2",
        anchor_kind="same_line_corroboration",
        character="林澈",
        statement="林澈把原稿改成红色",
    )
    request = build_draft_actor_review_request(
        index, (proposal,), frozen_content=content, expected_source=source
    )
    return content, source, request


def _prior_anchor_request():
    content = "林澈亲自把灯台封好。\n她又把钥匙交给船长。"
    source = _source(content, "2")
    index = build_draft_actor_clause_index(
        content, source, line_start=1, line_end=2
    )
    proposal = DraftActorProposal(
        source_sha256=source.content_sha256,
        line_start=1,
        line_end=2,
        evidence=content,
        target_clause_id="L2:A1",
        actor_anchor_id="L1:A1",
        anchor_kind="verified_prior_named_anchor",
        character="林澈",
        statement="林澈又把钥匙交给船长",
    )
    request = build_draft_actor_review_request(
        index,
        (proposal,),
        frozen_content=content,
        expected_source=source,
        verified_prior_anchor_ids=frozenset({"L1:A1"}),
    )
    return content, source, request


def _item(request, **overrides):
    value = {
        "proposal_id": request.proposals[0].proposal_id,
        "verdict": "supported",
        "actor": "proposed",
        "actuality": "asserted",
        "statement_relation": "supported",
        "correction_relation": "none",
        "basis_ids": list(required_draft_actor_review_basis_ids(request)),
    }
    value.update(overrides)
    return value


def _single_response(request, **overrides) -> str:
    return json.dumps(
        {
            "schema_version": DRAFT_ACTOR_REVIEW_SCHEMA_V1,
            "request_digest": draft_actor_review_request_digest(request),
            "items": [_item(request, **overrides)],
        },
        ensure_ascii=False,
    )


def _batch_response(requests, items) -> str:
    return json.dumps(
        {
            "schema_version": DRAFT_ACTOR_REVIEW_BATCH_SCHEMA_V1,
            "batch_digest": draft_actor_review_batch_digest(requests),
            "responses": [
                {
                    "schema_version": DRAFT_ACTOR_REVIEW_SCHEMA_V1,
                    "request_digest": draft_actor_review_request_digest(request),
                    "items": [item],
                }
                for request, item in zip(requests, items)
            ],
        },
        ensure_ascii=False,
    )


def test_supported_and_explicit_other_actor_decisions_require_complete_basis():
    content, source, request = _same_line_request()
    supported = evaluate_draft_actor_review(
        request,
        _single_response(request),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (supported.verdict, supported.reason) == ("supported", "supported")
    assert supported.basis_ids == required_draft_actor_review_basis_ids(request)

    rejected = evaluate_draft_actor_review(
        request,
        _single_response(request, verdict="rejected", actor="other"),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (rejected.verdict, rejected.reason) == (
        "rejected", "reviewer_rejected"
    )


@pytest.mark.parametrize(
    "basis",
    (
        [],
        ["L1:A1"],
        ["L1:A2", "L1:A1"],
        ["L1:A1", "L1:A1"],
        ["L1:A1", "L1:A2", "L1:A9"],
    ),
)
def test_non_uncertain_basis_is_never_repaired_or_reordered(basis):
    content, source, request = _same_line_request()
    decision = evaluate_draft_actor_review(
        request,
        _single_response(request, basis_ids=basis),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (decision.verdict, decision.reason, decision.basis_ids) == (
        "uncertain", "basis_invalid", ()
    )


def test_slot_conflicts_fail_closed_and_rejection_needs_negative_slot():
    content, source, request = _same_line_request()
    ambiguous = evaluate_draft_actor_review(
        request,
        _single_response(request, actor="ambiguous"),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (ambiguous.verdict, ambiguous.reason) == ("uncertain", "slot_conflict")
    assert ambiguous.slot_conflicts == ("actor",)

    unsupported_rejection = evaluate_draft_actor_review(
        request,
        _single_response(request, verdict="rejected"),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (unsupported_rejection.verdict, unsupported_rejection.reason) == (
        "uncertain", "slot_conflict"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("evidence", "她把原稿改成蓝色；记录显示本次操作由林澈本人完成。"),
        ("statement", "林澈把原稿改成蓝色"),
        ("actor_anchor_id", "L1:A1"),
        ("anchor_kind", "verified_prior_named_anchor"),
    ),
)
def test_directly_constructed_request_cannot_bypass_structural_screen(field, value):
    content, source, request = _same_line_request()
    altered = request.proposals[0].model_copy(update={field: value})
    forged = request.model_copy(update={"proposals": (altered,)})
    assert not verify_draft_actor_review_source(
        forged, source, frozen_content=content
    )
    decision = evaluate_draft_actor_review(
        forged,
        _single_response(forged),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (decision.verdict, decision.reason) == ("uncertain", "source_mismatch")


def test_changed_frozen_source_and_response_identity_fail_closed():
    content, source, request = _same_line_request()
    changed = content.replace("红色", "蓝色")
    source_decision = evaluate_draft_actor_review(
        request,
        _single_response(request),
        expected_source=source,
        frozen_content=changed,
    ).decisions[0]
    assert source_decision.reason == "source_mismatch"

    payload = json.loads(_single_response(request))
    payload["request_digest"] = "0" * 64
    mismatch = evaluate_draft_actor_review(
        request,
        json.dumps(payload),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert mismatch.reason == "response_mismatch"


def test_one_batch_covers_multiple_windows_and_validates_each_basis_independently():
    content1, source1, request1 = _same_line_request()
    content2, source2, request2 = _prior_anchor_request()
    requests = (request1, request2)
    raw = _batch_response(requests, (_item(request1), _item(request2)))
    evaluation = evaluate_draft_actor_review_batch(
        requests,
        raw,
        expected_sources=(source1, source2),
        frozen_contents=(content1, content2),
    )
    assert [row.decisions[0].verdict for row in evaluation.evaluations] == [
        "supported", "supported"
    ]

    second_bad = _item(request2)
    second_bad["basis_ids"] = list(
        required_draft_actor_review_basis_ids(request2)[:-1]
    )
    mixed = evaluate_draft_actor_review_batch(
        requests,
        _batch_response(requests, (_item(request1), second_bad)),
        expected_sources=(source1, source2),
        frozen_contents=(content1, content2),
    )
    assert mixed.evaluations[0].decisions[0].reason == "supported"
    assert mixed.evaluations[1].decisions[0].reason == "basis_invalid"


def test_batch_must_return_every_window_exactly_once():
    content1, source1, request1 = _same_line_request()
    content2, source2, request2 = _prior_anchor_request()
    requests = (request1, request2)
    payload = json.loads(_batch_response(requests, (_item(request1), _item(request2))))
    payload["responses"] = payload["responses"][:1]
    result = evaluate_draft_actor_review_batch(
        requests,
        json.dumps(payload),
        expected_sources=(source1, source2),
        frozen_contents=(content1, content2),
    )
    assert {
        decision.reason
        for row in result.evaluations
        for decision in row.decisions
    } == {"response_mismatch"}


@pytest.mark.parametrize(
    ("successor", "suffix"),
    (
        ("下一幕揭示，上一行只是林澈梦境中的想象。", "veto-dream"),
        ("更正：上一行动作实际由周砚完成。", "veto-other"),
        ("上一行动作实际由周尧完成。", "veto-other-explicit"),
        ("上一行动作实际由周尧亲自完成。", "veto-other-personally"),
        ("上一行动作实际由周未然完成。", "veto-other-name-with-wei"),
        ("林澈并未执行上一行动作。", "veto-target-denial"),
    ),
)
def test_explicit_successor_veto_overrides_malicious_all_supported(
    successor: str, suffix: str,
):
    content, source, request = _successor_request(successor, suffix)
    decision = evaluate_draft_actor_review(
        request,
        _single_response(request),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (decision.verdict, decision.reason, decision.basis_ids) == (
        "uncertain", "source_context_veto", ()
    )


@pytest.mark.parametrize(
    "successor",
    (
        "上一幕是林澈追逐梦想的开始。",
        "上一幕是关于梦境的讨论。",
        "上一行动作发生后，她去了排练室。",
        "上一幕是模拟装置启动的结果。",
        "上一行动作实际由周尧协助完成。",
        "上一行动作由周尧建议完成。",
        "上一行动作实际由周尧未完成。",
        "上一行动作实际由周尧并未完成。",
        "上一行动作实际由周尧没有完成。",
        "上一行动作实际由周尧拒绝完成。",
        "上一行动作实际由周尧并未立即完成。",
        "上一行动作实际由周尧并未迅速完成。",
        "上一行动作实际由周尧并未按时完成。",
        "上一行动作实际由周尧并未真正完成。",
        "上一行动作实际由周尧没有实际完成。",
        "上一行动作实际由周尧从未独立完成。",
    ),
)
def test_benign_successor_keywords_remain_supported(successor: str):
    content, source, request = _successor_request(successor, "benign-keyword")
    decision = evaluate_draft_actor_review(
        request,
        _single_response(request),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (decision.verdict, decision.reason) == ("supported", "supported")
    assert decision.basis_ids == required_draft_actor_review_basis_ids(request)


def test_same_line_clause_after_anchor_vetoes_malicious_all_supported():
    content = (
        "她把原稿改成红色；记录显示本次操作由林澈本人完成；"
        "但那只是排练。"
    )
    source = _source(content, "same-line-veto")
    index = build_draft_actor_clause_index(
        content, source, line_start=1, line_end=1
    )
    proposal = DraftActorProposal(
        source_sha256=source.content_sha256,
        line_start=1,
        line_end=1,
        evidence=content,
        target_clause_id="L1:A1",
        actor_anchor_id="L1:A2",
        anchor_kind="same_line_corroboration",
        character="林澈",
        statement="林澈把原稿改成红色",
    )
    request = build_draft_actor_review_request(
        index, (proposal,), frozen_content=content, expected_source=source
    )
    decision = evaluate_draft_actor_review(
        request,
        _single_response(request),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (decision.verdict, decision.reason, decision.basis_ids) == (
        "uncertain", "source_context_veto", ()
    )


@pytest.mark.parametrize(
    ("successor", "negative_slot"),
    (
        ("随后记录更正：上一行所写动作并非真实发生。", "correction"),
        ("下一幕揭示，上一行只是林澈梦境中的想象。", "dream"),
    ),
)
def test_adjacent_correction_or_dream_is_inside_basis_and_cannot_pass_conflicting_slots(
    successor: str, negative_slot: str,
):
    first = "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    content = f"{first}\n{successor}"
    source = _source(content, f"next-{negative_slot}")
    index = build_draft_actor_clause_index(
        content, source, line_start=1, line_end=1
    )
    proposal = DraftActorProposal(
        source_sha256=source.content_sha256,
        line_start=1,
        line_end=1,
        evidence=first,
        target_clause_id="L1:A1",
        actor_anchor_id="L1:A2",
        anchor_kind="same_line_corroboration",
        character="林澈",
        statement="林澈把原稿改成红色",
    )
    request = build_draft_actor_review_request(
        index, (proposal,), frozen_content=content, expected_source=source
    )
    assert request.review_line_end == 2
    assert any(
        basis_id.startswith("L2:")
        for basis_id in required_draft_actor_review_basis_ids(request)
    )
    override = (
        {"correction_relation": "corrected"}
        if negative_slot == "correction"
        else {"actuality": "hypothetical"}
    )
    conflicting_support = evaluate_draft_actor_review(
        request,
        _single_response(request, **override),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (conflicting_support.verdict, conflicting_support.reason) == (
        "uncertain", "slot_conflict"
    )
    explicit_rejection = evaluate_draft_actor_review(
        request,
        _single_response(request, verdict="rejected", **override),
        expected_source=source,
        frozen_content=content,
    ).decisions[0]
    assert (explicit_rejection.verdict, explicit_rejection.reason) == (
        "rejected", "reviewer_rejected"
    )


def test_review_context_never_jumps_across_blank_line():
    first = "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    content = f"{first}\n\n下一段才出现无关说明。"
    source = _source(content, "blank")
    index = build_draft_actor_clause_index(
        content, source, line_start=1, line_end=1
    )
    proposal = DraftActorProposal(
        source_sha256=source.content_sha256,
        line_start=1,
        line_end=1,
        evidence=first,
        target_clause_id="L1:A1",
        actor_anchor_id="L1:A2",
        anchor_kind="same_line_corroboration",
        character="林澈",
        statement="林澈把原稿改成红色",
    )
    request = build_draft_actor_review_request(
        index, (proposal,), frozen_content=content, expected_source=source
    )
    assert (request.review_line_start, request.review_line_end) == (1, 1)
