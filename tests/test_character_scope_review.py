from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from app.character_scope_review import (
    MAX_SCOPE_REVIEW_RESPONSE_BYTES,
    SCOPE_REVIEW_SCHEMA_V1,
    SCOPE_REVIEW_PROMPT_V2,
    ScopeReviewClause,
    ScopeReviewItem,
    ScopeReviewLine,
    ScopeReviewProposal,
    ScopeReviewRequest,
    ScopeReviewSourceIdentity,
    classify_basis_invalid,
    evaluate_scope_review,
    request_digest,
    verify_frozen_source,
)


def _line(parts: tuple[str, ...], *, line_number: int = 2) -> ScopeReviewLine:
    text = "，".join(parts) + "。"
    clauses = []
    offset = 0
    for ordinal, part in enumerate(parts, start=1):
        clauses.append(ScopeReviewClause(
            support_id=f"L{line_number}:A{ordinal}",
            line_number=line_number,
            start_offset=offset,
            end_offset=offset + len(part),
            text=part,
        ))
        offset += len(part) + 1
    return ScopeReviewLine(line_number=line_number, text=text, clauses=tuple(clauses))


def _proposal(
    *,
    proposal_id: str = "p1",
    support_id: str = "L2:A3",
    actor_anchor_id: str | None = "L2:A1",
    label_anchor_id: str | None = "L2:A1",
    scope_relation: str = "labelled_elaboration",
    dimension: str = "core_personality",
    stability: str = "core",
    key_object: str = "风险",
    statement: str = "桑衍会提前说明影响搭档航船的危险",
    polarity: str = "positive",
) -> ScopeReviewProposal:
    return ScopeReviewProposal(
        proposal_id=proposal_id,
        support_id=support_id,
        actor_anchor_id=actor_anchor_id,
        label_anchor_id=label_anchor_id,
        scope_relation=scope_relation,
        character="桑衍",
        dimension=dimension,
        trait_key="risk_disclosure",
        statement=statement,
        polarity=polarity,
        stability=stability,
        observation_kind="explicit_declaration",
        context="",
        key_object=key_object,
    )


def _request(
    line: ScopeReviewLine | None = None,
    proposals: tuple[ScopeReviewProposal, ...] | None = None,
    *,
    version: int = 4,
    document_id: str = "doc-1",
) -> tuple[ScopeReviewRequest, ScopeReviewSourceIdentity, str]:
    line = line or _line((
        "桑衍的核心性格是让搭档提前知道需要承担的风险",
        "无封港令时",
        "她会在调整航路前说明可能影响搭档航船的危险",
    ))
    frozen_content = "标题\n" + line.text + "\n"
    identity = ScopeReviewSourceIdentity(
        run_input_id="input-1",
        document_id=document_id,
        document_version=version,
        content_sha256=hashlib.sha256(frozen_content.encode("utf-8")).hexdigest(),
    )
    request = ScopeReviewRequest(
        **identity.model_dump(),
        block_line_start=2,
        lines=(line,),
        proposals=proposals or (_proposal(),),
    )
    return request, identity, frozen_content


def _item(request: ScopeReviewRequest, *, proposal_id: str = "p1", **changes) -> dict:
    proposal = next(p for p in request.proposals if p.proposal_id == proposal_id)
    row = {
        "proposal_id": proposal_id,
        "support_id": proposal.support_id,
        "verdict": "supported",
        "actor": "proposed",
        "actuality": "asserted",
        "statement_relation": "supported",
        "label_relation": "same_axis" if proposal.label_anchor_id else "none",
        "object_relation": "same" if proposal.key_object else "not_applicable",
        "polarity_relation": (
            "same" if proposal.polarity in {"positive", "negative"} else "not_applicable"
        ),
        "level_supported": "yes",
        "basis_ids": [clause.support_id for clause in request.lines[0].clauses],
    }
    row.update(changes)
    return row


def _response(request: ScopeReviewRequest, *items: dict, **changes) -> str:
    payload = {
        "schema_version": SCOPE_REVIEW_SCHEMA_V1,
        "request_digest": request_digest(request),
        "items": list(items),
    }
    payload.update(changes)
    return json.dumps(payload, ensure_ascii=False)


def _evaluate(request: ScopeReviewRequest, identity: ScopeReviewSourceIdentity,
              frozen_content: str, *items: dict):
    return evaluate_scope_review(
        request,
        _response(request, *items),
        expected_source=identity,
        frozen_content=frozen_content,
        expected_block_line_start=2,
    )


def test_nonliteral_same_axis_and_situational_middle_clause_are_supported():
    request, identity, frozen_content = _request()
    assert request.prompt_version == SCOPE_REVIEW_PROMPT_V2
    result = _evaluate(request, identity, frozen_content, _item(request))
    assert result.decisions[0].verdict == "supported"
    assert result.decisions[0].reason == "supported"
    assert result.decisions[0].basis_ids == ("L2:A1", "L2:A2", "L2:A3")


def test_local_label_can_anchor_the_target_clause_itself():
    line = _line(("桑衍的核心性格是提前说明风险",))
    proposal = _proposal(
        support_id="L2:A1",
        actor_anchor_id=None,
        label_anchor_id="L2:A1",
        scope_relation="local",
        statement="桑衍提前说明风险",
    )
    request, identity, frozen_content = _request(line, (proposal,))
    result = _evaluate(request, identity, frozen_content, _item(request, basis_ids=["L2:A1"]))
    assert result.decisions[0].verdict == "supported"


def test_actor_anchor_must_precede_target_even_with_local_label():
    line = _line(("桑衍的核心性格是提前说明风险",))
    proposal = _proposal(
        support_id="L2:A1",
        actor_anchor_id="L2:A1",
        label_anchor_id="L2:A1",
        scope_relation="same_actor_continuation",
    )
    with pytest.raises(ValidationError):
        _request(line, (proposal,))


@pytest.mark.parametrize("label_anchor_id", ("L2:A2", "L3:A1"))
def test_label_anchor_cannot_point_forward_or_to_another_line(label_anchor_id: str):
    first = _line(("桑衍喜欢蜜瓜", "桑衍的核心性格是说明风险"))
    second = _line(("她说明风险",), line_number=3)
    frozen_content = "标题\n" + first.text + "\n" + second.text + "\n"
    identity = ScopeReviewSourceIdentity(
        run_input_id="input-1",
        document_id="doc-1",
        document_version=4,
        content_sha256=hashlib.sha256(frozen_content.encode("utf-8")).hexdigest(),
    )
    proposal = _proposal(
        support_id="L2:A1",
        actor_anchor_id=None,
        label_anchor_id=label_anchor_id,
        scope_relation="labelled_elaboration",
        statement="桑衍喜欢蜜瓜",
    )
    with pytest.raises(ValidationError):
        ScopeReviewRequest(
            **identity.model_dump(),
            block_line_start=2,
            lines=(first, second),
            proposals=(proposal,),
        )


@pytest.mark.parametrize("changes", (
    {"scope_relation": "local", "actor_anchor_id": "L2:A1", "label_anchor_id": None},
    {"scope_relation": "local", "actor_anchor_id": None, "label_anchor_id": "L2:A1"},
    {"scope_relation": "same_actor_continuation", "actor_anchor_id": None,
     "label_anchor_id": None},
    {"scope_relation": "same_actor_continuation", "actor_anchor_id": "L2:A1",
     "label_anchor_id": "L2:A1"},
    {"scope_relation": "labelled_elaboration", "actor_anchor_id": "L2:A1",
     "label_anchor_id": None},
    {"scope_relation": "labelled_elaboration", "actor_anchor_id": "L2:A1",
     "label_anchor_id": "L2:A3"},
))
def test_scope_relation_rejects_missing_or_forbidden_anchor_combinations(changes: dict):
    proposal = _proposal(**changes)
    with pytest.raises(ValidationError):
        _request(proposals=(proposal,))


@pytest.mark.parametrize("changes", (
    {"scope_relation": "same_actor_continuation", "actor_anchor_id": "L2:A1",
     "label_anchor_id": None},
    {"scope_relation": "same_actor_continuation", "actor_anchor_id": "L2:A1",
     "label_anchor_id": "L2:A3"},
    {"scope_relation": "labelled_elaboration", "actor_anchor_id": None,
     "label_anchor_id": "L2:A1"},
))
def test_scope_relation_allows_valid_anchor_combinations(changes: dict):
    proposal = _proposal(**changes)
    request, identity, frozen_content = _request(proposals=(proposal,))
    result = _evaluate(request, identity, frozen_content, _item(request))
    assert result.decisions[0].verdict == "supported"


def test_digest_binds_source_identity_and_source_bytes():
    request, identity, frozen_content = _request()
    changed_version, _, _ = _request(version=5)
    changed_document, _, _ = _request(document_id="doc-2")
    assert request_digest(request) != request_digest(changed_version)
    assert request_digest(request) != request_digest(changed_document)
    assert verify_frozen_source(
        request, identity, frozen_content=frozen_content, expected_block_line_start=2,
    )
    assert not verify_frozen_source(
        request, identity, frozen_content=frozen_content + "伪造", expected_block_line_start=2,
    )
    assert not verify_frozen_source(
        request, changed_version, frozen_content=frozen_content, expected_block_line_start=2,
    )
    assert not verify_frozen_source(
        request, identity, frozen_content=frozen_content, expected_block_line_start=1,
    )


@pytest.mark.parametrize("response_mutation, reason", [
    ({"schema_version": "scope-review-v0"}, "response_invalid"),
    ({"request_digest": "0" * 64}, "response_mismatch"),
    ({"document_id": "forged"}, "response_invalid"),
])
def test_wrong_response_version_digest_or_extra_source_field_fails_closed(
    response_mutation: dict, reason: str,
):
    request, identity, frozen_content = _request()
    response = _response(request, _item(request), **response_mutation)
    result = evaluate_scope_review(
        request, response, expected_source=identity, frozen_content=frozen_content,
        expected_block_line_start=2,
    )
    assert result.decisions[0].verdict == "uncertain"
    assert result.decisions[0].reason == reason


def test_wrong_frozen_document_version_or_hash_fails_closed():
    request, identity, frozen_content = _request()
    response = _response(request, _item(request))
    for changed in (
        identity.model_copy(update={"document_version": identity.document_version + 1}),
        identity.model_copy(update={"content_sha256": "a" * 64}),
    ):
        result = evaluate_scope_review(
            request, response, expected_source=changed, frozen_content=frozen_content,
            expected_block_line_start=2,
        )
        assert result.decisions[0].reason == "source_mismatch"
        assert result.decisions[0].verdict == "uncertain"


def test_duplicate_request_proposal_id_and_invalid_clause_offsets_are_rejected():
    request, _, _ = _request()
    with pytest.raises(ValidationError):
        ScopeReviewRequest(**{
            **request.model_dump(),
            "proposals": (request.proposals[0], request.proposals[0]),
        })
    line = request.lines[0]
    bad_clause = line.clauses[0].model_copy(update={"start_offset": 1})
    with pytest.raises(ValidationError):
        ScopeReviewLine(
            line_number=line.line_number,
            text=line.text,
            clauses=(bad_clause, *line.clauses[1:]),
        )


@pytest.mark.parametrize("kind", ("missing", "duplicate", "extra", "forged_support"))
def test_response_proposal_ids_must_match_exactly(kind: str):
    proposals = (_proposal(), _proposal(proposal_id="p2"))
    request, identity, frozen_content = _request(proposals=proposals)
    items = [_item(request, proposal_id="p1"), _item(request, proposal_id="p2")]
    if kind == "missing":
        items.pop()
    elif kind == "duplicate":
        items[1] = items[0].copy()
    elif kind == "extra":
        items.append(dict(items[0], proposal_id="p3"))
    else:
        items[1] = dict(items[1], support_id="L2:A2")
    result = _evaluate(request, identity, frozen_content, *items)
    assert len(result.decisions) == 2
    assert all(item.verdict == "uncertain" and item.reason == "response_mismatch"
               for item in result.decisions)


@pytest.mark.parametrize("basis_ids", (
    ["L2:A1", "L2:A2", "L2:A9"],
    ["L2:A1", "L2:A1", "L2:A2", "L2:A3"],
    ["L2:A1", "L2:A3"],
    ["L2:A1", "L2:A2"],
))
def test_basis_must_be_allowlisted_unique_and_cover_target_anchor_path(basis_ids: list[str]):
    request, identity, frozen_content = _request()
    result = _evaluate(
        request, identity, frozen_content, _item(request, basis_ids=basis_ids),
    )
    assert result.decisions[0].verdict == "uncertain"
    assert result.decisions[0].reason == "basis_invalid"


@pytest.mark.parametrize("basis_ids,verdict,subtype", (
    (["L2:A1", "L2:A2", "L2:A3"], "supported", None),
    (["L2:A1", "L2:A1", "L2:A2", "L2:A3"], "supported", "duplicate"),
    (["L2:A1", "L2:A2", "L2:A9"], "supported", "unknown_id"),
    (["L2:A1", "L2:A2"], "supported", "missing_target"),
    (["L2:A2", "L2:A3"], "supported", "missing_anchor"),
    (["L2:A1", "L2:A3"], "supported", "missing_intermediate"),
    ([], "rejected", None),
    ([], "uncertain", None),
    (["L2:A1", "L2:A1"], "rejected", "duplicate"),
))
def test_basis_invalid_subtype_is_fixed_and_agrees_with_acceptance_gate(
    basis_ids: list[str], verdict: str, subtype: str | None,
):
    request, identity, frozen_content = _request()
    row = _item(request, basis_ids=basis_ids, verdict=verdict)
    item = ScopeReviewItem.model_validate_json(json.dumps(row))
    assert classify_basis_invalid(request, request.proposals[0], item) == subtype
    result = _evaluate(request, identity, frozen_content, row)
    assert (result.decisions[0].reason == "basis_invalid") == (subtype is not None)


def test_basis_invalid_subtype_distinguishes_cross_line_without_returning_ids():
    request, _, _ = _request()
    other_line = _line(("独立旁注",), line_number=3)
    extended = ScopeReviewRequest(
        run_input_id=request.run_input_id,
        document_id=request.document_id,
        document_version=request.document_version,
        content_sha256=request.content_sha256,
        block_line_start=request.block_line_start,
        lines=(*request.lines, other_line),
        proposals=request.proposals,
    )
    item = ScopeReviewItem.model_validate_json(json.dumps(_item(extended, basis_ids=[
        "L2:A1", "L2:A2", "L2:A3", "L3:A1",
    ])))
    assert classify_basis_invalid(extended, extended.proposals[0], item) == "cross_line"


def test_basis_invalid_subtype_covers_both_distinct_anchor_paths():
    line = _line(("actor", "context", "label", "target"))
    proposal = ScopeReviewProposal(**{
        **_proposal().model_dump(),
        "support_id": "L2:A4",
        "actor_anchor_id": "L2:A1",
        "label_anchor_id": "L2:A3",
    })
    request, identity, frozen_content = _request(line, (proposal,))
    for ids, expected in (
        (["L2:A1", "L2:A2", "L2:A3", "L2:A4"], None),
        (["L2:A1", "L2:A3", "L2:A4"], "missing_intermediate"),
        (["L2:A1", "L2:A2", "L2:A4"], "missing_anchor"),
    ):
        row = _item(request, basis_ids=ids)
        item = ScopeReviewItem.model_validate_json(json.dumps(row))
        assert classify_basis_invalid(request, proposal, item) == expected
        result = _evaluate(request, identity, frozen_content, row)
        assert (result.decisions[0].reason == "basis_invalid") == (expected is not None)


def test_supported_basis_rejects_unrelated_same_line_clause_without_order_requirement():
    line = _line(("actor", "situation", "target", "unrelated"))
    request, identity, frozen_content = _request(line)
    proposal = request.proposals[0]
    valid = _item(request, basis_ids=["L2:A3", "L2:A1", "L2:A2"])
    accepted = _evaluate(request, identity, frozen_content, valid)
    assert accepted.decisions[0].verdict == "supported"
    assert classify_basis_invalid(
        request, proposal, ScopeReviewItem.model_validate_json(json.dumps(valid))
    ) is None

    extra = _item(request, basis_ids=["L2:A1", "L2:A2", "L2:A3", "L2:A4"])
    assert classify_basis_invalid(
        request, proposal, ScopeReviewItem.model_validate_json(json.dumps(extra))
    ) == "extra_unrelated"
    rejected = _evaluate(request, identity, frozen_content, extra)
    assert rejected.decisions[0].verdict == "uncertain"
    assert rejected.decisions[0].reason == "basis_invalid"
    assert rejected.decisions[0].basis_ids == ()


@pytest.mark.parametrize("wrong_slot", (
    {"actor": "other"},
    {"actuality": "reported"},
    {"label_relation": "different_axis"},
    {"object_relation": "different"},
    {"polarity_relation": "opposite"},
    {"level_supported": "no"},
))
def test_supported_with_contradictory_slot_becomes_uncertain(wrong_slot: dict):
    request, identity, frozen_content = _request()
    result = _evaluate(request, identity, frozen_content, _item(request, **wrong_slot))
    assert result.decisions[0].verdict == "uncertain"
    assert result.decisions[0].reason == "slot_conflict"


@pytest.mark.parametrize("statement_relation,verdict,expected,reason", (
    ("supported", "supported", "supported", "supported"),
    ("contradicted", "supported", "uncertain", "slot_conflict"),
    ("ambiguous", "supported", "uncertain", "slot_conflict"),
    ("contradicted", "rejected", "rejected", "reviewer_rejected"),
    ("ambiguous", "rejected", "uncertain", "slot_conflict"),
))
def test_statement_relation_controls_verdict(
    statement_relation: str, verdict: str, expected: str, reason: str,
):
    request, identity, frozen_content = _request()
    item = _item(request, statement_relation=statement_relation, verdict=verdict)
    result = _evaluate(request, identity, frozen_content, item)
    assert result.decisions[0].verdict == expected
    assert result.decisions[0].reason == reason


@pytest.mark.parametrize("change", ("missing", "extra", "invalid"))
def test_statement_relation_is_required_and_allowlisted(change: str):
    request, identity, frozen_content = _request()
    item = _item(request)
    if change == "missing":
        item.pop("statement_relation")
    elif change == "extra":
        item["statement_explanation"] = "ignored"
    else:
        item["statement_relation"] = "inferred"
    result = _evaluate(request, identity, frozen_content, item)
    assert result.decisions[0].verdict == "uncertain"
    assert result.decisions[0].reason == "response_invalid"


@pytest.mark.parametrize("parts, proposal, negative_slot", (
    (("桑衍相信周尧喜欢蜜瓜",),
     {"support_id": "L2:A1", "actor_anchor_id": None, "label_anchor_id": None,
      "scope_relation": "local", "dimension": "preference", "stability": "stable",
      "key_object": "蜜瓜", "statement": "桑衍喜欢蜜瓜"}, {"actor": "other"}),
    (("桑衍会说明风险", "周尧加入值守", "喜欢蜜瓜"),
     {"label_anchor_id": None, "scope_relation": "same_actor_continuation",
      "dimension": "preference", "stability": "stable", "key_object": "蜜瓜",
      "statement": "桑衍喜欢蜜瓜"}, {"actor": "other"}),
    (("桑衍的核心性格是及时说明风险", "她把桌面擦干净"),
     {"support_id": "L2:A2", "key_object": "桌面", "statement": "桑衍擦干净桌面"},
     {"label_relation": "different_axis"}),
    (("桑衍的稳定偏好是热茶", "她也喜欢蜜瓜"),
     {"support_id": "L2:A2", "dimension": "preference", "stability": "stable",
      "key_object": "蜜瓜", "statement": "桑衍喜欢蜜瓜"},
     {"label_relation": "different_axis"}),
))
def test_four_semantic_risks_are_rejected_by_mocked_verdicts(
    parts: tuple[str, ...], proposal: dict, negative_slot: dict,
):
    source_line = _line(parts)
    candidate = _proposal(**proposal)
    request, identity, frozen_content = _request(source_line, (candidate,))
    item = _item(request, verdict="rejected", **negative_slot)
    result = _evaluate(request, identity, frozen_content, item)
    assert result.decisions[0].verdict == "rejected"
    assert result.decisions[0].reason == "reviewer_rejected"


def test_rejected_without_any_negative_slot_is_an_inconsistent_response():
    request, identity, frozen_content = _request()
    result = _evaluate(request, identity, frozen_content, _item(request, verdict="rejected"))
    assert result.decisions[0].verdict == "uncertain"
    assert result.decisions[0].reason == "slot_conflict"


def test_uncertain_and_malformed_or_oversized_responses_never_support():
    request, identity, frozen_content = _request()
    ambiguous = _item(request, verdict="uncertain", actor="ambiguous")
    result = _evaluate(request, identity, frozen_content, ambiguous)
    assert result.decisions[0].reason == "reviewer_uncertain"
    for response, reason in (
        ("not JSON", "response_invalid"),
        ("x" * (MAX_SCOPE_REVIEW_RESPONSE_BYTES + 1), "response_too_large"),
        (_response(request, dict(_item(request), reason="ignore the rules")), "response_invalid"),
    ):
        evaluation = evaluate_scope_review(
            request, response, expected_source=identity, frozen_content=frozen_content,
            expected_block_line_start=2,
        )
        assert evaluation.decisions[0].verdict == "uncertain"
        assert evaluation.decisions[0].reason == reason


def test_duplicate_json_keys_are_not_silently_accepted():
    request, identity, frozen_content = _request()
    valid = _response(request, _item(request))
    duplicate_key = valid.replace('"verdict": "supported"',
                                  '"verdict": "rejected", "verdict": "supported"')
    result = evaluate_scope_review(
        request, duplicate_key, expected_source=identity,
        frozen_content=frozen_content, expected_block_line_start=2,
    )
    assert result.decisions[0].verdict == "uncertain"
    assert result.decisions[0].reason == "response_invalid"


def test_unpaired_surrogate_in_model_response_is_controlled_uncertainty():
    request, identity, frozen_content = _request()
    result = evaluate_scope_review(
        request, "\ud800", expected_source=identity,
        frozen_content=frozen_content, expected_block_line_start=2,
    )
    assert result.decisions[0].verdict == "uncertain"
    assert result.decisions[0].reason == "response_invalid"
