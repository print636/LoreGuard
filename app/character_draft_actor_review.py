"""Source-bound structural screening for draft pronoun attribution proposals.

This module is deliberately disconnected from extraction and issue creation.
An eligible proposal only proceeds to an independent semantic reviewer: a
named corroborating clause can refer to another action in the same scene.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

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

# These expressions are intentionally full-clause grammars, not keyword
# scans.  They exclude bare ``梦`` and require an exact backward reference plus
# an affirmative cancellation predicate, so ``梦想``/排练室/模拟装置 and
# negated or speculative cancellation language remain reviewer-owned.
_TRAILING_VETO_LEAD = (
    r"(?:但|然而|不过|其实|原来|随后(?:查明|发现|确认)?|"
    r"经(?:查|核实)|记录显示)?"
)
_TRAILING_CONTEXT_REFERENCE = (
    r"(?:"
    r"(?:上|前)一(?:行|句|段|幕|场|场景)"
    r"(?:(?:所写|所述|所记载|所描写)(?:的)?"
    r"(?:情节|场景|动作|行动|行为|事件|事情|操作|内容)?"
    r"|(?:的)?(?:情节|场景|动作|行动|行为|事件|事情|操作|内容))?"
    r"|(?:上|前)一(?:次)?(?:动作|行动|行为|事件|事情|操作)"
    r"|(?:前述|上述)(?:的)?"
    r"(?:情节|场景|动作|行动|行为|事件|事情|操作|记录|内容|说法)"
    r"|刚才(?:的|那一)?(?:幕|场|场景|动作|行动|行为|事件|事情|操作)"
    r"|(?:这|那)(?:一幕|一场|个场景|次动作|次行动|次操作|件事|一切)"
    r"|(?:该|此|本次)(?:动作|行动|行为|事件|事情|操作)"
    r")"
)
_TRAILING_ACTION_REFERENCE = (
    r"(?:(?:该|此|本次|这次|那次|上述|前述)"
    r"(?:动作|行动|行为|事件|事情|操作)"
    r"|(?:上|前)一(?:次)?(?:动作|行动|行为|事件|事情|操作)"
    r"|(?:上|前)一(?:行|句|段)(?:所写|所述)?(?:的)?"
    r"(?:动作|行动|行为|事件|事情|操作)"
    r"|这件事|那件事)"
)
_TRAILING_NON_REALITY = (
    r"(?:一场梦|梦境(?:(?:中|里)(?:的)?(?:想象|片段|场景|画面)?)?"
    r"|梦(?:中|里)(?:的)?(?:想象|片段|场景|画面)?"
    r"|幻觉|幻象|想象|假想|幻想"
    r"|(?:一场|一次|一段)?(?:排练|演练|彩排)(?:过程|片段|场景)?"
    r"|(?:一次|一段)?模拟(?:演练|场景|情境|片段|过程)?)"
)
_TRAILING_NON_REALITY_VETO = re.compile(
    rf"{_TRAILING_VETO_LEAD}(?:{_TRAILING_CONTEXT_REFERENCE}"
    r"(?:实际上|事实上)?"
    r"(?:只是|仅是|不过是|原(?:来)?是|其实是|实为|属于|是|为|"
    r"发生(?:在|于)|出自)"
    rf"|(?:这|那)(?:只是|仅是|不过是)){_TRAILING_NON_REALITY}"
    r"(?:而已|罢了)?"
)
_TRAILING_NON_OCCURRENCE_VETO = re.compile(
    rf"{_TRAILING_VETO_LEAD}(?:{_TRAILING_CONTEXT_REFERENCE}|这|那)"
    r"(?:其实|实际上|事实上)?"
    r"(?:(?:并未|未曾|不曾|从未|并没有|没有|没)"
    r"(?:在现实中)?(?:真正|真实|实际|真的)?"
    r"(?:发生|出现|上演|存在)(?:过)?"
    r"|(?:并非|不是|绝非)(?:事实|(?:真实|实际|真正)"
    r"(?:发生|出现|上演)(?:的)?(?:事实|事件)?)|(?:纯属|只是|仅是)虚构)"
)
_TRAILING_RECORD_REFERENCE = (
    r"(?:(?:上|前)一(?:行|句|段)(?:所写|所述|所载)?(?:的)?"
    r"(?:记录|记载|报告|档案|日志|描述|叙述|说法|内容)?"
    r"|(?:前述|上述)(?:的)?(?:记录|记载|报告|档案|日志|描述|叙述|说法)"
    r"|(?:该|此|这份|那份)(?:记录|记载|报告|档案|日志|描述|叙述|说法))"
)
_TRAILING_RECORD_VETO = re.compile(
    rf"{_TRAILING_VETO_LEAD}{_TRAILING_RECORD_REFERENCE}(?:本身)?"
    r"(?:有误|不实|失实|不属实|系误记|(?:是|为|系)错误的?|"
    r"(?:已|已经)?(?:作废|失效)|(?:已|已经)?(?:被)?(?:撤回|撤销|收回))"
)
_TRAILING_OTHER_PERFORMER = re.compile(
    rf"{_TRAILING_VETO_LEAD}{_TRAILING_CONTEXT_REFERENCE}"
    r"(?:实际(?:上)?|其实|实则|事实上|经查|经核实)?(?:是)?由"
    r"(?P<actor>[\u4e00-\u9fffA-Za-z0-9·._'-]{1,64}?)(?:本人|亲自)?(?:所)?"
    r"(?:完成|执行|实施|操作|做出)(?:的)?"
)
_TRAILING_OTHER_PERFORMER_AS = re.compile(
    rf"{_TRAILING_VETO_LEAD}{_TRAILING_CONTEXT_REFERENCE}"
    r"(?:实为|实际(?:是|为)|其实(?:是|为)|真正(?:是|为)|而是)"
    r"(?P<actor>[\u4e00-\u9fffA-Za-z0-9·._'-]{1,64}?)(?:所为|做的|干的)"
)
_TRAILING_OTHER_PERFORMER_ROLE = re.compile(
    rf"{_TRAILING_VETO_LEAD}{_TRAILING_CONTEXT_REFERENCE}(?:的)?(?:实际|真正)?"
    r"(?:执行者|操作者|实施者|完成者)(?:是|为|系)"
    r"(?P<actor>[\u4e00-\u9fffA-Za-z0-9·._'-]{1,64})"
)
_TRAILING_VETO_WEAKENER = re.compile(
    r"如果|假如|也许|可能|或许|据说|传闻|听说|猜测|怀疑|推测|"
    r"是否|难道|莫非|不能说|无法认定|未证实|尚未证实"
)
_TRAILING_NON_PERFORMER_QUALIFIER = re.compile(
    r"协助|帮助|参与|监督|指导|建议|要求|安排|委托|让|请|叫|派|"
    r"命令|指使|见证|确认|核实|记录|负责"
)
_TRAILING_EXPLICIT_NEGATED_PERFORMER_QUALIFIER = re.compile(
    r"(?:并未|尚未|未曾|不曾|从未|并没有|没有|没能|未能|"
    r"拒绝|拒不|否认|不愿意?|不肯|不能|无法)"
)
_TRAILING_BARE_NEGATED_PERFORMER_QUALIFIER = re.compile(r"(?:没|未)$")
_TRAILING_JOINT_PERFORMER = re.compile(r"和|与|同|跟|及|、|共同|一起")

MAX_DRAFT_ACTOR_EVIDENCE_CHARS = 2_000
MAX_DRAFT_ACTOR_CLAUSES = 128
MAX_DRAFT_ACTOR_REVIEW_ITEMS = 64
MAX_DRAFT_ACTOR_REVIEW_REQUEST_BYTES = 131_072
MAX_DRAFT_ACTOR_REVIEW_BATCH_REQUEST_BYTES = 262_144
MAX_DRAFT_ACTOR_REVIEW_RESPONSE_BYTES = 32_768

DRAFT_ACTOR_REVIEW_SCHEMA_V1 = "character-draft-actor-review-v1"
DRAFT_ACTOR_REVIEW_BATCH_SCHEMA_V1 = "character-draft-actor-review-batch-v1"
DRAFT_ACTOR_REVIEW_PROMPT_V1 = "character-draft-actor-review-prompt-v1"
DRAFT_ACTOR_CLAUSE_INDEX_V1 = "draft-actor-clause-index-v1"

_PROPOSAL_ID = r"^dap_[a-f0-9]{32}$"


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


class DraftActorReviewProposal(DraftActorProposal):
    """One screened proposal with a server-derived immutable identity."""

    proposal_id: str = Field(pattern=_PROPOSAL_ID)


class DraftActorReviewRequest(ScopeReviewSourceIdentity):
    """A single frozen one-or-two-line window reviewed as one batch."""

    schema_version: Literal["character-draft-actor-review-v1"] = (
        DRAFT_ACTOR_REVIEW_SCHEMA_V1
    )
    prompt_version: Literal["character-draft-actor-review-prompt-v1"] = (
        DRAFT_ACTOR_REVIEW_PROMPT_V1
    )
    clause_index_version: Literal["draft-actor-clause-index-v1"] = (
        DRAFT_ACTOR_CLAUSE_INDEX_V1
    )
    review_line_start: int = Field(ge=1, le=10_000_000, strict=True)
    review_line_end: int = Field(ge=1, le=10_000_000, strict=True)
    lines: tuple[ScopeReviewLine, ...] = Field(min_length=1, max_length=3)
    proposals: tuple[DraftActorReviewProposal, ...] = Field(
        min_length=1, max_length=MAX_DRAFT_ACTOR_REVIEW_ITEMS
    )

    @model_validator(mode="after")
    def check_index_and_proposals(self) -> DraftActorReviewRequest:
        expected_lines = tuple(
            range(self.review_line_start, self.review_line_end + 1)
        )
        if (
            self.review_line_end - self.review_line_start > 2
            or tuple(line.line_number for line in self.lines) != expected_lines
            or sum(len(line.clauses) for line in self.lines)
            > MAX_DRAFT_ACTOR_CLAUSES
        ):
            raise ValueError("draft_actor_review_index_invalid")
        clauses = {
            clause.support_id: clause
            for line in self.lines
            for clause in line.clauses
        }
        if len(clauses) != sum(len(line.clauses) for line in self.lines):
            raise ValueError("draft_actor_review_index_invalid")
        proposal_ids: set[str] = set()
        proposal_ranges: set[tuple[int, int]] = set()
        for proposal in self.proposals:
            target = clauses.get(proposal.target_clause_id)
            anchor = clauses.get(proposal.actor_anchor_id)
            if (
                proposal.proposal_id in proposal_ids
                or proposal.proposal_id != _draft_actor_proposal_id(proposal)
                or proposal.source_sha256 != self.content_sha256
                or not (
                    self.review_line_start <= proposal.line_start
                    <= proposal.line_end <= self.review_line_end
                    and proposal.line_end - proposal.line_start <= 1
                )
                or target is None
                or anchor is None
                or not (
                    proposal.line_start <= target.line_number <= proposal.line_end
                    and proposal.line_start <= anchor.line_number <= proposal.line_end
                )
                or proposal.target_clause_id == proposal.actor_anchor_id
            ):
                raise ValueError("draft_actor_review_proposal_invalid")
            proposal_ids.add(proposal.proposal_id)
            proposal_ranges.add((proposal.line_start, proposal.line_end))
        if len(proposal_ranges) != 1:
            raise ValueError("draft_actor_review_proposal_window_mismatch")
        if len(_canonical_review_request_bytes(self)) > (
            MAX_DRAFT_ACTOR_REVIEW_REQUEST_BYTES
        ):
            raise ValueError("draft_actor_review_request_too_large")
        return self


DraftActorReviewVerdict = Literal["supported", "rejected", "uncertain"]
DraftActorReviewReason = Literal[
    "supported",
    "reviewer_rejected",
    "reviewer_uncertain",
    "source_context_veto",
    "source_mismatch",
    "response_too_large",
    "response_invalid",
    "response_mismatch",
    "basis_invalid",
    "slot_conflict",
]
DraftActorReviewSlotConflict = Literal[
    "actor",
    "actuality",
    "statement_relation",
    "correction_relation",
    "rejected_without_negative_slot",
]


class DraftActorReviewItem(BaseModel):
    """The complete and only model-authored shape for one proposal."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    proposal_id: str = Field(pattern=_PROPOSAL_ID)
    verdict: DraftActorReviewVerdict
    actor: Literal["proposed", "other", "ambiguous"]
    actuality: Literal[
        "asserted", "reported", "hypothetical", "question", "ambiguous"
    ]
    statement_relation: Literal["supported", "contradicted", "ambiguous"]
    correction_relation: Literal["none", "corrected", "ambiguous"]
    basis_ids: tuple[str, ...] = Field(max_length=MAX_DRAFT_ACTOR_CLAUSES)


class DraftActorReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["character-draft-actor-review-v1"]
    request_digest: str = Field(pattern=_SHA256)
    items: tuple[DraftActorReviewItem, ...] = Field(
        max_length=MAX_DRAFT_ACTOR_REVIEW_ITEMS
    )


class DraftActorReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    proposal_id: str = Field(pattern=_PROPOSAL_ID)
    verdict: DraftActorReviewVerdict
    reason: DraftActorReviewReason
    basis_ids: tuple[str, ...] = Field(
        default=(), max_length=MAX_DRAFT_ACTOR_CLAUSES
    )
    slot_conflicts: tuple[DraftActorReviewSlotConflict, ...] = Field(
        default=(), exclude=True
    )


class DraftActorReviewEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    request_digest: str = Field(pattern=_SHA256)
    decisions: tuple[DraftActorReviewDecision, ...] = Field(
        max_length=MAX_DRAFT_ACTOR_REVIEW_ITEMS
    )


class DraftActorReviewBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal["character-draft-actor-review-batch-v1"]
    batch_digest: str = Field(pattern=_SHA256)
    responses: tuple[DraftActorReviewResponse, ...] = Field(
        min_length=1, max_length=MAX_DRAFT_ACTOR_REVIEW_ITEMS
    )


class DraftActorReviewBatchEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    batch_digest: str = Field(pattern=_SHA256)
    evaluations: tuple[DraftActorReviewEvaluation, ...] = Field(
        min_length=1, max_length=MAX_DRAFT_ACTOR_REVIEW_ITEMS
    )


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


def _review_context_end(
    frozen_content: str, *, line_start: int, proposal_line_end: int
) -> int:
    """Include one real adjacent successor without crossing a blank paragraph."""

    segments = _segments(frozen_content)
    review_end = proposal_line_end
    if (
        review_end < len(segments)
        and review_end - line_start + 1 < 3
        and segments[review_end][1].strip()
    ):
        review_end += 1
    return review_end


def _build_draft_actor_review_lines(
    frozen_content: str,
    source: ScopeReviewSourceIdentity,
    *,
    review_line_start: int,
    review_line_end: int,
) -> tuple[ScopeReviewLine, ...]:
    """Index a real, contiguous review context of at most three source lines."""

    if not isinstance(frozen_content, str) or not isinstance(
        source, ScopeReviewSourceIdentity
    ):
        raise TypeError("draft actor review source invalid")
    if hashlib.sha256(frozen_content.encode("utf-8")).hexdigest() != (
        source.content_sha256
    ):
        raise ValueError("draft_actor_source_mismatch")
    segments = _segments(frozen_content)
    if not (
        1 <= review_line_start <= review_line_end <= len(segments)
        and review_line_end - review_line_start <= 2
    ):
        raise ValueError("draft_actor_review_window_invalid")
    selected = segments[review_line_start - 1:review_line_end]
    if any(not text.strip() or len(text) > 20_000 for _, text, _ in selected):
        raise ValueError("draft_actor_review_window_invalid")
    lines = tuple(
        _indexed_line(line_number, text)
        for line_number, (_, text, _) in enumerate(
            selected, start=review_line_start
        )
    )
    if sum(len(line.clauses) for line in lines) > MAX_DRAFT_ACTOR_CLAUSES:
        raise ValueError("draft_actor_index_too_large")
    return lines


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


def _proposal_payload(
    proposal: DraftActorProposal | DraftActorReviewProposal,
) -> dict[str, object]:
    return {
        name: getattr(proposal, name)
        for name in DraftActorProposal.model_fields
    }


def _draft_actor_proposal_id(
    proposal: DraftActorProposal | DraftActorReviewProposal,
) -> str:
    payload = json.dumps(
        _proposal_payload(proposal),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"dap_{hashlib.sha256(payload).hexdigest()[:32]}"


def _canonical_review_request_bytes(request: DraftActorReviewRequest) -> bytes:
    return json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def draft_actor_review_request_digest(request: DraftActorReviewRequest) -> str:
    """Digest every frozen source and proposal field in canonical order."""

    if not isinstance(request, DraftActorReviewRequest):
        raise TypeError("draft actor review request is invalid")
    return hashlib.sha256(_canonical_review_request_bytes(request)).hexdigest()


def draft_actor_review_batch_digest(
    requests: tuple[DraftActorReviewRequest, ...],
) -> str:
    """Bind an ordered, unique, bounded set of source-window requests."""

    if type(requests) is not tuple or not 1 <= len(requests) <= (
        MAX_DRAFT_ACTOR_REVIEW_ITEMS
    ) or any(not isinstance(request, DraftActorReviewRequest) for request in requests):
        raise TypeError("draft actor review batch is invalid")
    digests = tuple(draft_actor_review_request_digest(request) for request in requests)
    if (
        len(set(digests)) != len(digests)
        or sum(len(request.proposals) for request in requests)
        > MAX_DRAFT_ACTOR_REVIEW_ITEMS
    ):
        raise ValueError("draft_actor_review_batch_duplicate_or_too_large")
    payload = json.dumps(
        [request.model_dump(mode="json") for request in requests],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > MAX_DRAFT_ACTOR_REVIEW_BATCH_REQUEST_BYTES:
        raise ValueError("draft_actor_review_batch_too_large")
    return hashlib.sha256(payload).hexdigest()


def required_draft_actor_review_basis_ids(
    request: DraftActorReviewRequest,
) -> tuple[str, ...]:
    """Return the full frozen review window in source order."""

    if not isinstance(request, DraftActorReviewRequest):
        raise TypeError("draft actor review request is invalid")
    return tuple(
        clause.support_id
        for line in request.lines
        for clause in line.clauses
    )


def build_draft_actor_review_request(
    index: DraftActorClauseIndex,
    proposals: tuple[DraftActorProposal, ...],
    *,
    frozen_content: str,
    expected_source: ScopeReviewSourceIdentity,
    verified_prior_anchor_ids: frozenset[str] = frozenset(),
) -> DraftActorReviewRequest:
    """Re-screen unique proposals and bind them to one immutable source window.

    The caller may provide only independently verified prior anchor IDs. Model
    fields never create that trust. An ineligible or duplicate proposal aborts
    the whole request rather than being silently dropped or repaired.
    """

    if not isinstance(index, DraftActorClauseIndex) or not isinstance(
        expected_source, ScopeReviewSourceIdentity
    ):
        raise TypeError("draft actor review source is invalid")
    if not isinstance(frozen_content, str) or type(proposals) is not tuple:
        raise TypeError("draft actor review proposals are invalid")
    if type(verified_prior_anchor_ids) is not frozenset or any(
        not isinstance(value, str) or re.fullmatch(_SUPPORT_ID, value) is None
        for value in verified_prior_anchor_ids
    ):
        raise TypeError("draft actor verified anchors are invalid")
    if not 1 <= len(proposals) <= MAX_DRAFT_ACTOR_REVIEW_ITEMS:
        raise ValueError("draft_actor_review_proposal_count_invalid")
    if index.source != expected_source:
        raise ValueError("draft_actor_review_source_mismatch")
    try:
        rebuilt = build_draft_actor_clause_index(
            frozen_content,
            expected_source,
            line_start=index.line_start,
            line_end=index.line_end,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("draft_actor_review_source_mismatch") from exc
    if rebuilt != index:
        raise ValueError("draft_actor_review_source_mismatch")

    review_proposals: list[DraftActorReviewProposal] = []
    seen_ids: set[str] = set()
    for proposal in proposals:
        if not isinstance(proposal, DraftActorProposal):
            raise TypeError("draft actor review proposal is invalid")
        decision = screen_draft_actor_proposal(
            index,
            proposal,
            frozen_content=frozen_content,
            expected_source=expected_source,
            verified_prior_anchor_ids=verified_prior_anchor_ids,
        )
        if not decision.eligible_for_semantic_review:
            raise ValueError("draft_actor_review_proposal_not_eligible")
        proposal_id = _draft_actor_proposal_id(proposal)
        if proposal_id in seen_ids:
            raise ValueError("draft_actor_review_proposal_duplicate")
        seen_ids.add(proposal_id)
        review_proposals.append(
            DraftActorReviewProposal(
                **proposal.model_dump(mode="python"), proposal_id=proposal_id
            )
        )
    review_line_end = _review_context_end(
        frozen_content,
        line_start=index.line_start,
        proposal_line_end=index.line_end,
    )
    review_lines = _build_draft_actor_review_lines(
        frozen_content,
        expected_source,
        review_line_start=index.line_start,
        review_line_end=review_line_end,
    )
    return DraftActorReviewRequest(
        **expected_source.model_dump(mode="python"),
        review_line_start=index.line_start,
        review_line_end=review_line_end,
        lines=review_lines,
        proposals=tuple(review_proposals),
    )


def verify_draft_actor_review_source(
    request: DraftActorReviewRequest,
    expected_source: ScopeReviewSourceIdentity,
    *,
    frozen_content: str,
) -> bool:
    """Rebuild the complete clause index from the independently frozen source."""

    if not isinstance(request, DraftActorReviewRequest) or not isinstance(
        expected_source, ScopeReviewSourceIdentity
    ):
        raise TypeError("draft actor review source boundary is invalid")
    if not isinstance(frozen_content, str):
        raise TypeError("draft actor review frozen source is invalid")
    if any(
        getattr(request, name) != getattr(expected_source, name)
        for name in ScopeReviewSourceIdentity.model_fields
    ):
        return False
    try:
        if hashlib.sha256(frozen_content.encode("utf-8")).hexdigest() != (
            request.content_sha256
        ):
            return False
        proposal_ranges = {
            (proposal.line_start, proposal.line_end)
            for proposal in request.proposals
        }
        if len(proposal_ranges) != 1:
            return False
        proposal_line_start, proposal_line_end = next(iter(proposal_ranges))
        expected_review_end = _review_context_end(
            frozen_content,
            line_start=proposal_line_start,
            proposal_line_end=proposal_line_end,
        )
        if (
            request.review_line_start != proposal_line_start
            or request.review_line_end != expected_review_end
        ):
            return False
        rebuilt_lines = _build_draft_actor_review_lines(
            frozen_content,
            expected_source,
            review_line_start=request.review_line_start,
            review_line_end=request.review_line_end,
        )
        proposal_index = build_draft_actor_clause_index(
            frozen_content,
            expected_source,
            line_start=proposal_line_start,
            line_end=proposal_line_end,
        )
    except (UnicodeError, TypeError, ValueError):
        return False
    if rebuilt_lines != request.lines:
        return False
    # A request is normally produced by ``build_draft_actor_review_request``,
    # but the evaluator treats even a directly constructed typed request as
    # untrusted. Recreate every base proposal and repeat the complete screen.
    # Prior-anchor membership is reconstructed only from this immutable
    # request; the screen still requires an adjacent, direct named-actor
    # clause and rejects possessors, delegation, perception, or other actors.
    verified_prior_anchor_ids = frozenset(
        proposal.actor_anchor_id
        for proposal in request.proposals
        if proposal.anchor_kind == "verified_prior_named_anchor"
    )
    for review_proposal in request.proposals:
        try:
            proposal = DraftActorProposal.model_validate(
                _proposal_payload(review_proposal), strict=True
            )
            decision = screen_draft_actor_proposal(
                proposal_index,
                proposal,
                frozen_content=frozen_content,
                expected_source=expected_source,
                verified_prior_anchor_ids=verified_prior_anchor_ids,
            )
        except (TypeError, ValueError, ValidationError):
            return False
        if not decision.eligible_for_semantic_review:
            return False
    return True


def _uncertain_review(
    request: DraftActorReviewRequest,
    reason: DraftActorReviewReason,
) -> DraftActorReviewEvaluation:
    return DraftActorReviewEvaluation(
        request_digest=draft_actor_review_request_digest(request),
        decisions=tuple(
            DraftActorReviewDecision(
                proposal_id=proposal.proposal_id,
                verdict="uncertain",
                reason=reason,
            )
            for proposal in request.proposals
        ),
    )


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


def _supported_slot_conflicts(
    item: DraftActorReviewItem,
) -> tuple[DraftActorReviewSlotConflict, ...]:
    conflicts: list[DraftActorReviewSlotConflict] = []
    if item.actor != "proposed":
        conflicts.append("actor")
    if item.actuality != "asserted":
        conflicts.append("actuality")
    if item.statement_relation != "supported":
        conflicts.append("statement_relation")
    if item.correction_relation != "none":
        conflicts.append("correction_relation")
    return tuple(conflicts)


def _explicit_negative_slots(item: DraftActorReviewItem) -> bool:
    return (
        item.actor == "other"
        or item.actuality in {"reported", "hypothetical", "question"}
        or item.statement_relation == "contradicted"
        or item.correction_relation == "corrected"
    )


def _draft_actor_trailing_context(
    request: DraftActorReviewRequest,
    proposal: DraftActorReviewProposal,
) -> tuple[str, ...]:
    """Return only clauses after either evidence endpoint, excluding both.

    The union boundary deliberately includes clauses between target and anchor,
    clauses later on the same line, and the server-added successor line.  It
    never feeds either model-selected evidence clause to the lexical veto.
    """

    ordered = tuple(
        (line.line_number, clause)
        for line in request.lines
        for clause in line.clauses
    )
    positions = {
        clause.support_id: position
        for position, (_line_number, clause) in enumerate(ordered)
    }
    try:
        boundary = min(
            positions[proposal.target_clause_id],
            positions[proposal.actor_anchor_id],
        )
    except KeyError:
        return ()
    evidence_ids = {proposal.target_clause_id, proposal.actor_anchor_id}
    trailing: list[str] = []
    lines = {line.line_number: line for line in request.lines}
    for line_number, clause in ordered[boundary + 1:]:
        if clause.support_id in evidence_ids:
            continue
        # Clause text excludes its separator. Preserve only a following
        # question mark so a question cannot look like an assertion here.
        suffix = lines[line_number].text[clause.end_offset:]
        terminal = next((char for char in suffix if not char.isspace()), "")
        trailing.append(
            clause.text + terminal if terminal in "？?" else clause.text
        )
    return tuple(trailing)


def _target_character_explicitly_excluded(
    text: str,
    proposal: DraftActorReviewProposal,
) -> bool:
    character = re.escape(re.sub(r"\s+", "", proposal.character))
    referenced_denial = re.fullmatch(
        rf"{_TRAILING_VETO_LEAD}{_TRAILING_CONTEXT_REFERENCE}"
        rf"(?:其实|实际上|事实上)?(?:并非|不是|绝非)(?:由)?"
        rf"{character}(?:本人|亲自)?"
        r"(?:所为|做的|干的|(?:所)?(?:完成|执行|实施|进行|操作|做出)(?:的)?)",
        text,
    )
    character_denial = re.fullmatch(
        rf"{_TRAILING_VETO_LEAD}{character}(?:本人)?"
        r"(?:并未|未曾|不曾|从未|并没有|没有)"
        r"(?:亲自)?(?:完成|执行|实施|进行|操作|做出|做|干)(?:了|过)?"
        rf"{_TRAILING_ACTION_REFERENCE}",
        text,
    )
    role_denial = re.fullmatch(
        rf"{_TRAILING_VETO_LEAD}{_TRAILING_CONTEXT_REFERENCE}(?:的)?"
        r"(?:实际|真正)?(?:执行者|实施者|操作者|完成者)"
        rf"(?:并非|不是|绝非){character}",
        text,
    )
    return any((referenced_denial, character_denial, role_denial))


def _target_character_dream_context(
    text: str,
    proposal: DraftActorReviewProposal,
) -> bool:
    character = re.escape(re.sub(r"\s+", "", proposal.character))
    return re.fullmatch(
        rf"{_TRAILING_VETO_LEAD}{_TRAILING_CONTEXT_REFERENCE}"
        r"(?:实际上|事实上)?"
        r"(?:只是|仅是|不过是|原(?:来)?是|其实是|实为|属于|是|为)"
        rf"{character}(?:的)?梦境(?:(?:中|里)(?:的)?"
        r"(?:想象|片段|场景|画面)?)?(?:而已|罢了)?",
        text,
    ) is not None


def _explicit_different_performer(
    text: str,
    proposal: DraftActorReviewProposal,
) -> bool:
    character = re.sub(r"\s+", "", proposal.character)
    patterns = (
        _TRAILING_OTHER_PERFORMER,
        _TRAILING_OTHER_PERFORMER_AS,
        _TRAILING_OTHER_PERFORMER_ROLE,
    )
    for pattern in patterns:
        match = pattern.fullmatch(text)
        if match is None:
            continue
        actor = re.sub(
            r"(?:本人|亲自)$", "", match.group("actor").strip()
        ).strip()
        # Joint or qualified performers remain reviewer-owned, including a
        # phrase that names the proposed character alongside someone else.
        # The lazy actor capture can also absorb a preverbal negation or
        # refusal (for example, ``周尧并未`` before ``完成``); that is not an
        # affirmative attribution to another performer.
        if (
            actor
            and actor != character
            and character not in actor
            and _TRAILING_JOINT_PERFORMER.search(actor) is None
            and _TRAILING_NON_PERFORMER_QUALIFIER.search(actor) is None
            and _TRAILING_EXPLICIT_NEGATED_PERFORMER_QUALIFIER.search(actor)
            is None
            and _TRAILING_BARE_NEGATED_PERFORMER_QUALIFIER.search(actor)
            is None
        ):
            return True
    return False


def _has_draft_actor_source_context_veto(
    request: DraftActorReviewRequest,
    proposal: DraftActorReviewProposal,
) -> bool:
    for raw_text in _draft_actor_trailing_context(request, proposal):
        text = re.sub(r"\s+", "", raw_text)
        if (
            not text
            or any(mark in text for mark in _QUOTE_MARKS)
            or _TRAILING_VETO_WEAKENER.search(text)
        ):
            continue
        if (
            _TRAILING_NON_REALITY_VETO.fullmatch(text)
            or _TRAILING_NON_OCCURRENCE_VETO.fullmatch(text)
            or _TRAILING_RECORD_VETO.fullmatch(text)
            or _target_character_explicitly_excluded(text, proposal)
            or _target_character_dream_context(text, proposal)
            or _explicit_different_performer(text, proposal)
        ):
            return True
    return False


def evaluate_draft_actor_review(
    request: DraftActorReviewRequest,
    raw_response: str,
    *,
    expected_source: ScopeReviewSourceIdentity,
    frozen_content: str,
) -> DraftActorReviewEvaluation:
    """Validate one strict reviewer response without repairing any field."""

    if not isinstance(request, DraftActorReviewRequest) or not isinstance(
        raw_response, str
    ):
        raise TypeError("draft actor review arguments are invalid")
    if not verify_draft_actor_review_source(
        request, expected_source, frozen_content=frozen_content
    ):
        return _uncertain_review(request, "source_mismatch")
    try:
        response_size = len(raw_response.encode("utf-8"))
    except UnicodeError:
        return _uncertain_review(request, "response_invalid")
    if response_size > MAX_DRAFT_ACTOR_REVIEW_RESPONSE_BYTES:
        return _uncertain_review(request, "response_too_large")
    try:
        json.loads(
            raw_response,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
        response = DraftActorReviewResponse.model_validate_json(
            raw_response, strict=True
        )
    except (ValidationError, ValueError, TypeError, RecursionError):
        return _uncertain_review(request, "response_invalid")
    if response.request_digest != draft_actor_review_request_digest(request):
        return _uncertain_review(request, "response_mismatch")
    requested = {proposal.proposal_id: proposal for proposal in request.proposals}
    returned = {item.proposal_id: item for item in response.items}
    if len(returned) != len(response.items) or set(returned) != set(requested):
        return _uncertain_review(request, "response_mismatch")

    required_basis = required_draft_actor_review_basis_ids(request)
    allowed_basis = set(required_basis)
    decisions: list[DraftActorReviewDecision] = []
    for proposal in request.proposals:
        item = returned[proposal.proposal_id]
        basis_valid = (
            len(set(item.basis_ids)) == len(item.basis_ids)
            and set(item.basis_ids) <= allowed_basis
            and (
                item.basis_ids == required_basis
                or item.verdict == "uncertain" and not item.basis_ids
            )
        )
        if not basis_valid:
            decisions.append(
                DraftActorReviewDecision(
                    proposal_id=proposal.proposal_id,
                    verdict="uncertain",
                    reason="basis_invalid",
                )
            )
            continue
        if item.verdict == "supported":
            conflicts = _supported_slot_conflicts(item)
            if conflicts:
                decisions.append(
                    DraftActorReviewDecision(
                        proposal_id=proposal.proposal_id,
                        verdict="uncertain",
                        reason="slot_conflict",
                        slot_conflicts=conflicts,
                    )
                )
            elif _has_draft_actor_source_context_veto(request, proposal):
                decisions.append(
                    DraftActorReviewDecision(
                        proposal_id=proposal.proposal_id,
                        verdict="uncertain",
                        reason="source_context_veto",
                    )
                )
            else:
                decisions.append(
                    DraftActorReviewDecision(
                        proposal_id=proposal.proposal_id,
                        verdict="supported",
                        reason="supported",
                        basis_ids=required_basis,
                    )
                )
        elif item.verdict == "rejected":
            explicit_negative = _explicit_negative_slots(item)
            decisions.append(
                DraftActorReviewDecision(
                    proposal_id=proposal.proposal_id,
                    verdict="rejected" if explicit_negative else "uncertain",
                    reason=(
                        "reviewer_rejected"
                        if explicit_negative else "slot_conflict"
                    ),
                    basis_ids=required_basis if explicit_negative else (),
                    slot_conflicts=(
                        () if explicit_negative
                        else ("rejected_without_negative_slot",)
                    ),
                )
            )
        else:
            decisions.append(
                DraftActorReviewDecision(
                    proposal_id=proposal.proposal_id,
                    verdict="uncertain",
                    reason="reviewer_uncertain",
                )
            )
    return DraftActorReviewEvaluation(
        request_digest=draft_actor_review_request_digest(request),
        decisions=tuple(decisions),
    )


def _uncertain_review_batch(
    requests: tuple[DraftActorReviewRequest, ...],
    reason: DraftActorReviewReason,
) -> DraftActorReviewBatchEvaluation:
    return DraftActorReviewBatchEvaluation(
        batch_digest=draft_actor_review_batch_digest(requests),
        evaluations=tuple(
            _uncertain_review(request, reason) for request in requests
        ),
    )


def evaluate_draft_actor_review_batch(
    requests: tuple[DraftActorReviewRequest, ...],
    raw_response: str,
    *,
    expected_sources: tuple[ScopeReviewSourceIdentity, ...],
    frozen_contents: tuple[str, ...],
) -> DraftActorReviewBatchEvaluation:
    """Validate one response covering every request window exactly once."""

    batch_digest = draft_actor_review_batch_digest(requests)
    if (
        not isinstance(raw_response, str)
        or type(expected_sources) is not tuple
        or type(frozen_contents) is not tuple
        or len(expected_sources) != len(requests)
        or len(frozen_contents) != len(requests)
        or any(
            not isinstance(source, ScopeReviewSourceIdentity)
            for source in expected_sources
        )
        or any(not isinstance(content, str) for content in frozen_contents)
    ):
        raise TypeError("draft actor review batch arguments are invalid")
    if any(
        not verify_draft_actor_review_source(
            request, source, frozen_content=content
        )
        for request, source, content in zip(
            requests, expected_sources, frozen_contents
        )
    ):
        return _uncertain_review_batch(requests, "source_mismatch")
    try:
        response_size = len(raw_response.encode("utf-8"))
    except UnicodeError:
        return _uncertain_review_batch(requests, "response_invalid")
    if response_size > MAX_DRAFT_ACTOR_REVIEW_RESPONSE_BYTES:
        return _uncertain_review_batch(requests, "response_too_large")
    try:
        json.loads(
            raw_response,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
        response = DraftActorReviewBatchResponse.model_validate_json(
            raw_response, strict=True
        )
    except (ValidationError, ValueError, TypeError, RecursionError):
        return _uncertain_review_batch(requests, "response_invalid")
    if response.batch_digest != batch_digest:
        return _uncertain_review_batch(requests, "response_mismatch")
    requested = {
        draft_actor_review_request_digest(request): request
        for request in requests
    }
    returned = {
        item.request_digest: item for item in response.responses
    }
    if len(returned) != len(response.responses) or set(returned) != set(requested):
        return _uncertain_review_batch(requests, "response_mismatch")
    sources = {
        draft_actor_review_request_digest(request): (source, content)
        for request, source, content in zip(
            requests, expected_sources, frozen_contents
        )
    }
    evaluations: list[DraftActorReviewEvaluation] = []
    for request in requests:
        digest = draft_actor_review_request_digest(request)
        source, content = sources[digest]
        evaluations.append(
            evaluate_draft_actor_review(
                request,
                returned[digest].model_dump_json(),
                expected_source=source,
                frozen_content=content,
            )
        )
    return DraftActorReviewBatchEvaluation(
        batch_digest=batch_digest,
        evaluations=tuple(evaluations),
    )
