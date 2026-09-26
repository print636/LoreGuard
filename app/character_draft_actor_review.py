"""Source-bound structural screening for draft pronoun attribution proposals.

This module is deliberately disconnected from extraction and issue creation.
An eligible proposal only proceeds to an independent semantic reviewer: a
named corroborating clause can refer to another action in the same scene.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .character_scope_review import ScopeReviewClause, ScopeReviewLine, ScopeReviewSourceIdentity


_SUPPORT_ID = r"^L[1-9][0-9]{0,7}:A[1-9][0-9]{0,2}$"
_SHA256 = r"^[a-f0-9]{64}$"
_QUOTE_MARKS = frozenset('"“”‘’「」『』')
_OPEN_QUOTES = {'"': '"', '“': '”', '‘': '’', '「': '」', '『': '』'}
_SPLIT = frozenset("，,。！？!?；;：:")
_BRANCH = re.compile(r"(?:如果|假如|假设|倘若|若是|否则|要么|或者|或是|另一条线|另一分支|分支|结局|可能|也许|设想)")
_VOICE = re.compile(
    r"(?:广播|录音|剪接|拼接|假播报|转述|转告|传闻|听说|据说|"
    r"声称|宣称|自称|说(?:道|自己|我|她|他|了|过)|回答|问道|写道|"
    r"据.{0,8}(?:说|称))"
)
_JOINT = re.compile(r"(?:两人|二人|双方|众人|大家|各自|另一人|另一个人|其他人|她们|他们|[她他](?:和|与|同|跟)|[她他]们|共同|一起)")
_DELEGATE = re.compile(r"^(?:让|请|叫|派|由|命令|指使|看见|看到|看着|望着|听见|听到|得知|记录|转述)")
_NEGATED_PERSONAL = re.compile(
    r"(?:没有|并未|未曾|从未|不是|并非|否认).{0,12}(?:本人|亲自)|"
    r"(?:本人|亲自).{0,12}(?:没有|并未|未曾|从未|不是|并非|否认)"
)
_NON_DIRECT_NAMED_TAIL = re.compile(
    r"^(?:本人|亲自|直接|自己|又|仍|正在|已|曾经){0,2}"
    r"(?:的|和|与|跟|同|、|让|请|叫|派|由|看见|看到|看着|望着|"
    r"听见|听到|得知|转述)"
)
_OTHER_SUBJECT = re.compile(r"^[\u4e00-\u9fffA-Za-z]{2,12}(?:又|也|亲自|直接|偷偷|悄悄)?(?:把|将|让|请|叫|派|由)")

MAX_DRAFT_ACTOR_EVIDENCE_CHARS = 2_000
MAX_DRAFT_ACTOR_CLAUSES = 128


class DraftActorProposal(BaseModel):
    """Model-proposed fields; IDs and text are checked against server source."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_sha256: str = Field(pattern=_SHA256)
    line_start: int = Field(ge=1, le=10_000_000)
    line_end: int = Field(ge=1, le=10_000_000)
    evidence: str = Field(min_length=1, max_length=MAX_DRAFT_ACTOR_EVIDENCE_CHARS)
    target_clause_id: str = Field(pattern=_SUPPORT_ID)
    actor_anchor_id: str = Field(pattern=_SUPPORT_ID)
    anchor_kind: Literal["same_line_corroboration", "verified_prior_named_anchor"]
    character: str = Field(min_length=1, max_length=64)
    statement: str = Field(min_length=2, max_length=300)


@dataclass(frozen=True, slots=True)
class DraftActorClauseIndex:
    source: ScopeReviewSourceIdentity
    line_start: int
    line_end: int
    lines: tuple[ScopeReviewLine, ...]

    def resolve(self, support_id: str) -> ScopeReviewClause | None:
        return next(
            (clause for line in self.lines for clause in line.clauses
             if clause.support_id == support_id), None,
        )


DraftActorReason = Literal[
    "eligible_for_semantic_review",
    "source_mismatch",
    "evidence_mismatch",
    "invalid_clause_reference",
    "unsafe_context",
    "statement_mismatch",
    "anchor_not_structurally_supported",
]


@dataclass(frozen=True, slots=True)
class DraftActorDecision:
    eligible_for_semantic_review: bool
    reason: DraftActorReason


def _segments(content: str) -> tuple[tuple[int, str, str], ...]:
    """Return exact source offsets, content lines, and line endings."""

    output = []
    cursor = 0
    for segment in content.splitlines(keepends=True):
        newline = "\r\n" if segment.endswith("\r\n") else (
            "\n" if segment.endswith("\n") else "\r" if segment.endswith("\r") else ""
        )
        output.append((cursor, segment[:-len(newline)] if newline else segment, newline))
        cursor += len(segment)
    return tuple(output)


def _indexed_line(line_number: int, text: str) -> ScopeReviewLine:
    clauses = []
    cursor = 0
    stack: list[str] = []

    def append(end: int) -> None:
        raw = text[cursor:end]
        if not raw.strip():
            return
        start_offset = cursor + len(raw) - len(raw.lstrip())
        end_offset = cursor + len(raw.rstrip())
        ordinal = len(clauses) + 1
        if ordinal > 64:
            raise ValueError("draft_actor_index_too_large")
        clauses.append(ScopeReviewClause(
            support_id=f"L{line_number}:A{ordinal}",
            line_number=line_number,
            start_offset=start_offset,
            end_offset=end_offset,
            text=text[start_offset:end_offset],
        ))

    for offset, char in enumerate(text):
        if stack and char == stack[-1]:
            stack.pop()
        elif char in _OPEN_QUOTES:
            stack.append(_OPEN_QUOTES[char])
        elif char in _QUOTE_MARKS:
            raise ValueError("draft_actor_unbalanced_quote")
        elif char in _SPLIT and not stack:
            append(offset)
            cursor = offset + 1
    if stack:
        raise ValueError("draft_actor_unbalanced_quote")
    append(len(text))
    if not clauses:
        raise ValueError("draft_actor_empty_line")
    return ScopeReviewLine(line_number=line_number, text=text, clauses=tuple(clauses))


def build_draft_actor_clause_index(
    frozen_content: str,
    source: ScopeReviewSourceIdentity,
    *,
    line_start: int,
    line_end: int,
) -> DraftActorClauseIndex:
    """Index one source line or two immediately adjacent, nonblank lines."""

    if not isinstance(frozen_content, str) or not isinstance(source, ScopeReviewSourceIdentity):
        raise TypeError("draft actor source invalid")
    if type(line_start) is not int or type(line_end) is not int:
        raise TypeError("draft actor lines invalid")
    if hashlib.sha256(frozen_content.encode("utf-8")).hexdigest() != source.content_sha256:
        raise ValueError("draft_actor_source_mismatch")
    segments = _segments(frozen_content)
    if not (1 <= line_start <= line_end <= len(segments) and line_end - line_start <= 1):
        raise ValueError("draft_actor_window_invalid")
    selected = segments[line_start - 1:line_end]
    if any(not text.strip() or len(text) > 20_000 for _, text, _ in selected):
        raise ValueError("draft_actor_window_invalid")
    lines = tuple(
        _indexed_line(line_number, text)
        for line_number, (_, text, _) in enumerate(selected, start=line_start)
    )
    if sum(len(line.clauses) for line in lines) > MAX_DRAFT_ACTOR_CLAUSES:
        raise ValueError("draft_actor_index_too_large")
    return DraftActorClauseIndex(source, line_start, line_end, lines)


def _exact_evidence(content: str, line_start: int, line_end: int) -> str:
    segments = _segments(content)
    first = segments[line_start - 1][0]
    last_offset, last_text, _ = segments[line_end - 1]
    return content[first:last_offset + len(last_text)]


def _between(index: DraftActorClauseIndex, first: ScopeReviewClause,
             second: ScopeReviewClause) -> tuple[ScopeReviewClause, ...]:
    ordered = tuple(clause for line in index.lines for clause in line.clauses)
    first_index = ordered.index(first)
    second_index = ordered.index(second)
    return ordered[first_index + 1:second_index]


def _unsafe_window(evidence: str) -> bool:
    return bool(
        any(char in evidence for char in _QUOTE_MARKS)
        or _BRANCH.search(evidence)
        or _VOICE.search(evidence)
        or _JOINT.search(evidence)
    )


def screen_draft_actor_proposal(
    index: DraftActorClauseIndex,
    proposal: DraftActorProposal,
    *,
    frozen_content: str,
    expected_source: ScopeReviewSourceIdentity,
    verified_prior_anchor_ids: frozenset[str] = frozenset(),
) -> DraftActorDecision:
    """Check structural eligibility only; never approve actor attribution.

    ``verified_prior_anchor_ids`` must be supplied by an independent trusted
    caller after reviewing direct named-actor clauses, never copied from the
    model proposal. This module rechecks their source location and syntax.
    """

    rejected = lambda reason: DraftActorDecision(False, reason)
    if not isinstance(index, DraftActorClauseIndex) or not isinstance(
        proposal, DraftActorProposal
    ) or not isinstance(expected_source, ScopeReviewSourceIdentity):
        raise TypeError("draft actor review input invalid")
    if index.source != expected_source or proposal.source_sha256 != expected_source.content_sha256:
        return rejected("source_mismatch")
    try:
        rebuilt = build_draft_actor_clause_index(
            frozen_content, expected_source,
            line_start=proposal.line_start, line_end=proposal.line_end,
        )
    except (TypeError, ValueError):
        return rejected("source_mismatch")
    if rebuilt != index or (proposal.line_start, proposal.line_end) != (
        index.line_start, index.line_end
    ):
        return rejected("source_mismatch")
    if proposal.evidence != _exact_evidence(
        frozen_content, proposal.line_start, proposal.line_end
    ):
        return rejected("evidence_mismatch")
    target = index.resolve(proposal.target_clause_id)
    anchor = index.resolve(proposal.actor_anchor_id)
    if target is None or anchor is None or target == anchor:
        return rejected("invalid_clause_reference")
    if (
        min(target.line_number, anchor.line_number),
        max(target.line_number, anchor.line_number),
    ) != (
        proposal.line_start, proposal.line_end
    ):
        return rejected("invalid_clause_reference")
    if _unsafe_window(proposal.evidence):
        return rejected("unsafe_context")
    if not target.text.startswith(("她", "他")) or target.text.startswith(("她们", "他们")):
        return rejected("statement_mismatch")
    if target.text[1:].startswith(("的",)) or _DELEGATE.match(target.text[1:]):
        return rejected("unsafe_context")
    if proposal.statement != proposal.character + target.text[1:]:
        return rejected("statement_mismatch")
    if len(proposal.statement) > 300 or not proposal.character.strip():
        return rejected("statement_mismatch")
    if proposal.anchor_kind == "same_line_corroboration":
        # This is a source-backed *lead* for the reviewer. It does not establish
        # which action the named clause describes.
        if (
            anchor.line_number != target.line_number
            or anchor.start_offset <= target.start_offset
            or len(_between(index, target, anchor)) > 3
            or proposal.character not in anchor.text
            or not any(marker in anchor.text for marker in ("本人", "亲自"))
            or _NEGATED_PERSONAL.search(anchor.text)
        ):
            return rejected("anchor_not_structurally_supported")
        between = _between(index, target, anchor)
    else:
        if proposal.actor_anchor_id not in verified_prior_anchor_ids:
            return rejected("anchor_not_structurally_supported")
        if anchor.line_number == target.line_number:
            adjacent = index.lines[0].clauses
            valid_order = adjacent.index(target) - adjacent.index(anchor) == 1
        else:
            valid_order = (
                anchor.line_number + 1 == target.line_number
                and index.lines[0].clauses[-1] == anchor
                and index.lines[1].clauses[0] == target
            )
        if (
            not valid_order
            or not anchor.text.startswith(proposal.character)
            or _NON_DIRECT_NAMED_TAIL.match(anchor.text[len(proposal.character):])
        ):
            return rejected("anchor_not_structurally_supported")
        between = ()
    other_clauses = (
        clause for line in index.lines for clause in line.clauses
        if clause not in {target, anchor} and not clause.text.startswith(proposal.character)
    )
    if any(_OTHER_SUBJECT.match(clause.text) for clause in other_clauses):
        return rejected("unsafe_context")
    return DraftActorDecision(True, "eligible_for_semantic_review")
