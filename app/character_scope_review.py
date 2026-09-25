"""Bounded, source-bound contract for the formal-profile semantic veto.

This module has no provider or persistence access. A caller supplies frozen
source data and a model response; only the caller may turn a supported decision
into a candidate eligible for the ordinary author-confirmation workflow.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


SCOPE_REVIEW_SCHEMA_V1 = "character-scope-review-v1"
SCOPE_REVIEW_PROMPT_V1 = "character-scope-review-prompt-v1"
SCOPE_REVIEW_PROMPT_V2 = "character-scope-review-prompt-v2"
ASSERTION_INDEX_V1 = "assertion-index-v1"
MAX_SCOPE_REVIEW_REQUEST_BYTES = 131_072
MAX_SCOPE_REVIEW_RESPONSE_BYTES = 32_768
MAX_SCOPE_REVIEW_ITEMS = 64

_SUPPORT_ID_PATTERN = r"^L[1-9][0-9]{0,7}:A[1-9][0-9]{0,2}$"
_PROPOSAL_ID_PATTERN = r"^[A-Za-z0-9:_-]{1,96}$"
_SHA256_PATTERN = r"^[a-f0-9]{64}$"
_SUPPORT_ID_PARTS = re.compile(r"^L([1-9][0-9]{0,7}):A([1-9][0-9]{0,2})$")

ReviewVerdict = Literal["supported", "rejected", "uncertain"]
BasisInvalidSubtype = Literal[
    "duplicate",
    "unknown_id",
    "cross_line",
    "missing_target",
    "missing_anchor",
    "missing_intermediate",
    "extra_unrelated",
]
ReviewReason = Literal[
    "supported",
    "reviewer_rejected",
    "reviewer_uncertain",
    "source_mismatch",
    "response_too_large",
    "response_invalid",
    "response_mismatch",
    "basis_invalid",
    "slot_conflict",
]


class ScopeReviewSourceIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    run_input_id: str = Field(min_length=1, max_length=200)
    document_id: str = Field(min_length=1, max_length=200)
    document_version: int = Field(ge=1)
    content_sha256: str = Field(pattern=_SHA256_PATTERN)


class ScopeReviewClause(BaseModel):
    """One server-created support clause, with Python codepoint offsets."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    support_id: str = Field(pattern=_SUPPORT_ID_PATTERN)
    line_number: int = Field(ge=1, le=10_000_000)
    start_offset: int = Field(ge=0, le=20_000)
    end_offset: int = Field(ge=1, le=20_000)
    text: str = Field(min_length=1, max_length=20_000)


class ScopeReviewLine(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    line_number: int = Field(ge=1, le=10_000_000)
    text: str = Field(min_length=1, max_length=20_000)
    clauses: tuple[ScopeReviewClause, ...] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def check_clauses(self) -> ScopeReviewLine:
        preceding_end = 0
        for ordinal, clause in enumerate(self.clauses, start=1):
            match = _SUPPORT_ID_PARTS.fullmatch(clause.support_id)
            if (
                match is None
                or int(match.group(1)) != self.line_number
                or int(match.group(2)) != ordinal
                or clause.line_number != self.line_number
                or clause.start_offset < preceding_end
                or clause.end_offset <= clause.start_offset
                or clause.end_offset > len(self.text)
                or self.text[clause.start_offset:clause.end_offset] != clause.text
            ):
                raise ValueError("scope_review_clause_invalid")
            preceding_end = clause.end_offset
        return self


class ScopeReviewProposal(BaseModel):
    """An immutable proposal for the reviewer to judge, never to edit."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    proposal_id: str = Field(pattern=_PROPOSAL_ID_PATTERN)
    support_id: str = Field(pattern=_SUPPORT_ID_PATTERN)
    actor_anchor_id: str | None = Field(default=None, pattern=_SUPPORT_ID_PATTERN)
    label_anchor_id: str | None = Field(default=None, pattern=_SUPPORT_ID_PATTERN)
    scope_relation: Literal["local", "same_actor_continuation", "labelled_elaboration"]
    character: str = Field(min_length=1, max_length=64)
    dimension: Literal[
        "core_personality", "preference", "value", "speech_pattern",
        "behavior_boundary", "contextual_behavior", "current_state",
    ]
    trait_key: str = Field(min_length=1, max_length=80)
    statement: str = Field(min_length=2, max_length=300)
    polarity: Literal["positive", "negative", "neutral", "unclear"]
    stability: Literal["core", "stable", "temporary", "situational", "unknown"]
    observation_kind: Literal[
        "explicit_declaration", "preference_expression", "dialogue", "speech_sample",
        "action", "decision", "interaction", "state_description",
    ]
    context: str = Field(default="", max_length=160)
    key_object: str = Field(default="", max_length=80)


class ScopeReviewRequest(ScopeReviewSourceIdentity):
    schema_version: Literal["character-scope-review-v1"] = SCOPE_REVIEW_SCHEMA_V1
    prompt_version: Literal["character-scope-review-prompt-v2"] = SCOPE_REVIEW_PROMPT_V2
    assertion_index_version: Literal["assertion-index-v1"] = ASSERTION_INDEX_V1
    block_line_start: int = Field(ge=1, le=10_000_000)
    lines: tuple[ScopeReviewLine, ...] = Field(min_length=1, max_length=64)
    proposals: tuple[ScopeReviewProposal, ...] = Field(min_length=1, max_length=MAX_SCOPE_REVIEW_ITEMS)

    @model_validator(mode="after")
    def check_index_and_proposals(self) -> ScopeReviewRequest:
        if tuple(line.line_number for line in self.lines) != tuple(sorted({
            line.line_number for line in self.lines
        })):
            raise ValueError("scope_review_lines_invalid")
        if self.lines[0].line_number < self.block_line_start:
            raise ValueError("scope_review_lines_invalid")
        clauses = {
            clause.support_id: clause
            for line in self.lines
            for clause in line.clauses
        }
        if len(clauses) > 256 or len(clauses) != sum(len(line.clauses) for line in self.lines):
            raise ValueError("scope_review_index_invalid")
        proposal_ids: set[str] = set()
        for proposal in self.proposals:
            if proposal.proposal_id in proposal_ids:
                raise ValueError("scope_review_proposal_duplicate")
            proposal_ids.add(proposal.proposal_id)
            target = clauses.get(proposal.support_id)
            if target is None:
                raise ValueError("scope_review_support_invalid")
            if proposal.scope_relation == "local":
                if (
                    proposal.actor_anchor_id is not None
                    or proposal.label_anchor_id not in {None, proposal.support_id}
                ):
                    raise ValueError("scope_review_relation_invalid")
            elif proposal.scope_relation == "same_actor_continuation":
                if (
                    proposal.actor_anchor_id is None
                    or proposal.label_anchor_id not in {None, proposal.support_id}
                ):
                    raise ValueError("scope_review_relation_invalid")
            elif proposal.label_anchor_id in {None, proposal.support_id}:
                raise ValueError("scope_review_relation_invalid")
            for anchor_id, may_be_target in (
                (proposal.actor_anchor_id, False),
                (proposal.label_anchor_id, True),
            ):
                if anchor_id is None:
                    continue
                anchor = clauses.get(anchor_id)
                if (
                    anchor is None
                    or anchor.line_number != target.line_number
                    or anchor.start_offset > target.start_offset
                    or (not may_be_target and anchor.start_offset == target.start_offset)
                ):
                    raise ValueError("scope_review_anchor_invalid")
        if len(_canonical_request_bytes(self)) > MAX_SCOPE_REVIEW_REQUEST_BYTES:
            raise ValueError("scope_review_request_too_large")
        return self


class ScopeReviewItem(BaseModel):
    """Only these model-authored fields are allowed in one review item."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    proposal_id: str = Field(pattern=_PROPOSAL_ID_PATTERN)
    support_id: str = Field(pattern=_SUPPORT_ID_PATTERN)
    verdict: ReviewVerdict
    actor: Literal["proposed", "other", "ambiguous"]
    actuality: Literal["asserted", "reported", "hypothetical", "question", "ambiguous"]
    statement_relation: Literal["supported", "contradicted", "ambiguous"]
    label_relation: Literal["same_axis", "different_axis", "none", "ambiguous"]
    object_relation: Literal["same", "different", "not_applicable", "ambiguous"]
    polarity_relation: Literal["same", "opposite", "not_applicable", "ambiguous"]
    level_supported: Literal["yes", "no", "ambiguous"]
    basis_ids: tuple[str, ...] = Field(max_length=64)


class ScopeReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["character-scope-review-v1"]
    request_digest: str = Field(pattern=_SHA256_PATTERN)
    items: tuple[ScopeReviewItem, ...] = Field(max_length=MAX_SCOPE_REVIEW_ITEMS)


class ScopeReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    proposal_id: str
    support_id: str
    verdict: ReviewVerdict
    reason: ReviewReason
    basis_ids: tuple[str, ...] = ()


class ScopeReviewEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    request_digest: str
    decisions: tuple[ScopeReviewDecision, ...]


def _canonical_request_bytes(request: ScopeReviewRequest) -> bytes:
    return json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def request_digest(request: ScopeReviewRequest) -> str:
    """Digest the server-validated request, including frozen source identity."""

    if not isinstance(request, ScopeReviewRequest):
        raise TypeError("request must be a ScopeReviewRequest")
    return hashlib.sha256(_canonical_request_bytes(request)).hexdigest()


def verify_frozen_source(
    request: ScopeReviewRequest,
    expected_source: ScopeReviewSourceIdentity,
    *,
    frozen_content: str,
    expected_block_line_start: int,
) -> bool:
    """Check the request against server-owned identity, block start, and full text."""

    if not isinstance(request, ScopeReviewRequest) or not isinstance(
        expected_source, ScopeReviewSourceIdentity
    ):
        raise TypeError("scope review source boundary is invalid")
    if not isinstance(frozen_content, str) or type(expected_block_line_start) is not int:
        raise TypeError("scope review frozen source arguments are invalid")
    if expected_block_line_start != request.block_line_start:
        return False
    if any(
        getattr(request, name) != getattr(expected_source, name)
        for name in ScopeReviewSourceIdentity.model_fields
    ):
        return False
    if hashlib.sha256(frozen_content.encode("utf-8")).hexdigest() != request.content_sha256:
        return False
    source_lines = frozen_content.splitlines()
    return all(
        line.line_number <= len(source_lines)
        and source_lines[line.line_number - 1] == line.text
        for line in request.lines
    )


def _uncertain_all(request: ScopeReviewRequest, reason: ReviewReason) -> ScopeReviewEvaluation:
    return ScopeReviewEvaluation(
        request_digest=request_digest(request),
        decisions=tuple(
            ScopeReviewDecision(
                proposal_id=proposal.proposal_id,
                support_id=proposal.support_id,
                verdict="uncertain",
                reason=reason,
            )
            for proposal in request.proposals
        ),
    )


def _required_basis_ids(
    request: ScopeReviewRequest,
    proposal: ScopeReviewProposal,
    clauses: dict[str, ScopeReviewClause],
) -> frozenset[str]:
    target = clauses[proposal.support_id]
    target_line = next(line for line in request.lines if line.line_number == target.line_number)
    required = {proposal.support_id}
    for anchor_id in (proposal.actor_anchor_id, proposal.label_anchor_id):
        if anchor_id is None:
            continue
        anchor = clauses[anchor_id]
        required.update(
            clause.support_id for clause in target_line.clauses
            if anchor.start_offset <= clause.start_offset <= target.start_offset
        )
    return frozenset(required)


def _basis_valid(
    request: ScopeReviewRequest,
    proposal: ScopeReviewProposal,
    item: ScopeReviewItem,
) -> bool:
    clauses = {
        clause.support_id: clause
        for line in request.lines
        for clause in line.clauses
    }
    target = clauses[proposal.support_id]
    if len(set(item.basis_ids)) != len(item.basis_ids):
        return False
    if any(
        basis_id not in clauses or clauses[basis_id].line_number != target.line_number
        for basis_id in item.basis_ids
    ):
        return False
    if item.verdict != "supported":
        return True
    return _required_basis_ids(request, proposal, clauses) == set(item.basis_ids)


def classify_basis_invalid(
    request: ScopeReviewRequest,
    proposal: ScopeReviewProposal,
    item: ScopeReviewItem,
) -> BasisInvalidSubtype | None:
    """Classify an invalid basis without returning source text or model-authored IDs.

    This diagnostic helper does not participate in the acceptance decision.
    Its precedence is fixed so a single item contributes at most one category.
    """
    clauses = {
        clause.support_id: clause
        for line in request.lines
        for clause in line.clauses
    }
    target = clauses[proposal.support_id]
    basis = set(item.basis_ids)
    if len(basis) != len(item.basis_ids):
        return "duplicate"
    if any(basis_id not in clauses for basis_id in basis):
        return "unknown_id"
    if any(clauses[basis_id].line_number != target.line_number for basis_id in basis):
        return "cross_line"
    if item.verdict != "supported":
        return None
    if proposal.support_id not in basis:
        return "missing_target"
    anchors = tuple(
        anchor_id for anchor_id in (proposal.actor_anchor_id, proposal.label_anchor_id)
        if anchor_id is not None
    )
    if any(anchor_id not in basis for anchor_id in anchors):
        return "missing_anchor"
    required = _required_basis_ids(request, proposal, clauses)
    if required - basis:
        return "missing_intermediate"
    if basis - required:
        return "extra_unrelated"
    return None


def _slots_support(proposal: ScopeReviewProposal, item: ScopeReviewItem) -> bool:
    if (
        item.actor != "proposed"
        or item.actuality != "asserted"
        or item.statement_relation != "supported"
    ):
        return False
    if proposal.label_anchor_id is not None:
        if item.label_relation != "same_axis":
            return False
    elif item.label_relation not in {"same_axis", "none"}:
        return False
    if item.object_relation != ("same" if proposal.key_object else "not_applicable"):
        return False
    expected_polarity = (
        "same" if proposal.polarity in {"positive", "negative"} else "not_applicable"
    )
    return item.polarity_relation == expected_polarity and item.level_supported == "yes"


def _slots_reject(proposal: ScopeReviewProposal, item: ScopeReviewItem) -> bool:
    return (
        item.actor == "other"
        or item.actuality in {"reported", "hypothetical", "question"}
        or item.statement_relation == "contradicted"
        or item.label_relation == "different_axis"
        or item.object_relation == "different"
        or item.polarity_relation == "opposite"
        or item.level_supported == "no"
        or (
            proposal.label_anchor_id is not None and item.label_relation == "none"
        )
        or (
            not proposal.key_object and item.object_relation == "same"
        )
        or (
            proposal.polarity in {"neutral", "unclear"}
            and item.polarity_relation == "same"
        )
    )


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


def evaluate_scope_review(
    request: ScopeReviewRequest,
    raw_response: str,
    *,
    expected_source: ScopeReviewSourceIdentity,
    frozen_content: str,
    expected_block_line_start: int,
) -> ScopeReviewEvaluation:
    """Normalize malformed or unsupported model output to controlled uncertainty.

    A model's echoed digest is checked only against a digest recomputed here.
    Source identity, block start, and full document are always supplied
    separately by the caller and checked against the presented lines.
    """

    if not isinstance(request, ScopeReviewRequest) or not isinstance(raw_response, str):
        raise TypeError("scope review arguments are invalid")
    if not verify_frozen_source(
        request,
        expected_source,
        frozen_content=frozen_content,
        expected_block_line_start=expected_block_line_start,
    ):
        return _uncertain_all(request, "source_mismatch")
    try:
        response_size = len(raw_response.encode("utf-8"))
    except UnicodeEncodeError:
        return _uncertain_all(request, "response_invalid")
    if response_size > MAX_SCOPE_REVIEW_RESPONSE_BYTES:
        return _uncertain_all(request, "response_too_large")
    try:
        json.loads(
            raw_response,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (ValueError, RecursionError):
        return _uncertain_all(request, "response_invalid")
    try:
        response = ScopeReviewResponse.model_validate_json(raw_response, strict=True)
    except ValidationError:
        return _uncertain_all(request, "response_invalid")
    if response.request_digest != request_digest(request):
        return _uncertain_all(request, "response_mismatch")
    by_id = {item.proposal_id: item for item in response.items}
    requested = {proposal.proposal_id: proposal for proposal in request.proposals}
    if (
        len(by_id) != len(response.items)
        or set(by_id) != set(requested)
        or any(by_id[proposal_id].support_id != proposal.support_id
               for proposal_id, proposal in requested.items())
    ):
        return _uncertain_all(request, "response_mismatch")
    decisions: list[ScopeReviewDecision] = []
    for proposal in request.proposals:
        item = by_id[proposal.proposal_id]
        if not _basis_valid(request, proposal, item):
            verdict: ReviewVerdict = "uncertain"
            reason: ReviewReason = "basis_invalid"
        elif item.verdict == "supported":
            verdict = "supported" if _slots_support(proposal, item) else "uncertain"
            reason = "supported" if verdict == "supported" else "slot_conflict"
        elif item.verdict == "rejected":
            verdict = "rejected" if _slots_reject(proposal, item) else "uncertain"
            reason = "reviewer_rejected" if verdict == "rejected" else "slot_conflict"
        else:
            verdict = "uncertain"
            reason = "reviewer_uncertain"
        decisions.append(ScopeReviewDecision(
            proposal_id=proposal.proposal_id,
            support_id=proposal.support_id,
            verdict=verdict,
            reason=reason,
            basis_ids=item.basis_ids if reason not in {"basis_invalid", "slot_conflict"} else (),
        ))
    return ScopeReviewEvaluation(request_digest=request_digest(request), decisions=tuple(decisions))
