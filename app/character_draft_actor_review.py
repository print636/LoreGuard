"""Source-bound structural screening for draft pronoun attribution proposals.

This module is deliberately disconnected from extraction and issue creation.
An eligible proposal only proceeds to an independent semantic reviewer: a
named corroborating clause can refer to another action in the same scene.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
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

# V2 is an additive protocol.  The V1 request/response models and canonical
# digests above remain unchanged so stored V1 diagnostics can still be read
# and verified byte-for-byte.  V2 binds one server-owned recall target to
# frozen source clauses; the reviewer never writes character, axis, object or
# statement fields back to the server.
TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V2 = "character-target-bound-draft-review-v2"
TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V2 = (
    "character-target-bound-draft-review-batch-v2"
)
TARGET_BOUND_DRAFT_REVIEW_PROMPT_V2 = (
    "character-target-bound-draft-review-prompt-v2"
)
TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3 = "character-target-bound-draft-review-v3"
TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V3 = (
    "character-target-bound-draft-review-batch-v3"
)
TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4 = "character-target-bound-draft-review-v4"
TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V4 = (
    "character-target-bound-draft-review-batch-v4"
)
TARGET_BOUND_DRAFT_REVIEW_PROMPT_V3 = (
    "character-target-bound-draft-review-prompt-v3"
)
# V4 keeps the V3 request/response schemas and semantic admission rules.  It
# versions only the model-facing output contract: the prompt now spells out
# the complete nested batch envelope and every legal enum so a provider cannot
# silently flatten ``windows[].request.proposals`` into top-level responses.
TARGET_BOUND_DRAFT_REVIEW_PROMPT_V4 = (
    "character-target-bound-draft-review-prompt-v4"
)
# V5 keeps every V3 semantic gate unchanged and removes ambiguity about the
# coordinate space used by model-authored object offsets.  The provider prompt
# projects the exact frozen fact clause next to each proposal and declares its
# first Unicode codepoint as offset zero.
TARGET_BOUND_DRAFT_REVIEW_PROMPT_V5 = (
    "character-target-bound-draft-review-prompt-v5"
)
# V6 keeps the V3 schemas and every semantic admission gate, but distinguishes
# an author-approved scoped value/boundary axis from a literal source object.
# The comparison-key tail names the frozen axis; it is not draft text that the
# reviewer may be required to locate as an object span.
TARGET_BOUND_DRAFT_REVIEW_PROMPT_V6 = (
    "character-target-bound-draft-review-prompt-v6"
)
# V7 adds a server-verifiable high-precision boundary for literal objects. If
# the raw target object occurs exactly once in the frozen fact clause, every
# accepted same/broader/narrower span must cover that complete occurrence.
TARGET_BOUND_DRAFT_REVIEW_PROMPT_V7 = (
    "character-target-bound-draft-review-prompt-v7"
)
# V8 accompanies the additive V4 wire protocol.  It lets the server provide
# exact coordinates for one uniquely occurring frozen target literal while
# leaving every semantic decision with the independent reviewer.
TARGET_BOUND_DRAFT_REVIEW_PROMPT_V8 = (
    "character-target-bound-draft-review-prompt-v8"
)

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


# ---------------------------------------------------------------------------
# Target-bound draft semantic binding protocol V2
# ---------------------------------------------------------------------------

_TARGET_DIGEST = r"^[a-f0-9]{64}$"
_TARGET_PROPOSAL_ID = r"^tdp_[a-f0-9]{32}$"
_TARGET_DIMENSIONS = (
    "core_personality",
    "preference",
    "value",
    "relationship_attitude",
    "motivation_goal",
    "speech_pattern",
    "behavior_boundary",
    "contextual_behavior",
    "current_state",
)
_TARGET_OBJECT_DIMENSIONS = frozenset(
    {
        "preference",
        "value",
        "relationship_attitude",
        "motivation_goal",
        "behavior_boundary",
        "current_state",
    }
)
_TARGET_SCOPED_AXIS_DIMENSIONS = frozenset({"value", "behavior_boundary"})
_TARGET_ACTOR_LITERAL = re.compile(r"^[^\s\x00-\x1f]{1,64}$")
UNSAFE_TARGET_ACTOR_LITERALS = frozenset(
    {
        # First, second and third person, including common plural/reflexive
        # forms. These strings cannot uniquely identify one canonical actor.
        "我", "我们", "咱", "咱们", "俺", "俺们", "吾", "吾等", "余", "予",
        "你", "你们", "您", "您们", "尔", "汝", "阁下",
        "他", "他们", "她", "她们", "它", "它们", "祂", "祂们",
        "其", "本人", "自己", "人家", "彼此", "对方",
        # Common demonstrative/generic referring expressions.
        "这", "那", "这个", "那个", "这些", "那些", "这位", "那位",
        "此", "该", "此人", "该人", "此位", "该位", "大家", "众人", "各位",
        # English literals are equally non-unique if present in a formal
        # profile or imported manuscript.
        "i", "me", "we", "us", "you", "he", "him", "she", "her", "it",
        "they", "them", "myself", "yourself", "himself", "herself", "itself",
        "ourselves", "yourselves", "themselves", "this", "that", "these", "those",
    }
)


def is_unsafe_target_actor_literal(value: object) -> bool:
    if not isinstance(value, str):
        return True
    normalized = re.sub(
        r"\s+", "", unicodedata.normalize("NFKC", value)
    ).casefold()
    return normalized in UNSAFE_TARGET_ACTOR_LITERALS


def _target_actor_literals_are_safe(
    character: object, aliases: object,
) -> bool:
    if not isinstance(character, str) or type(aliases) is not tuple:
        return False
    actors = (character, *aliases)
    normalized: set[str] = set()
    for actor in actors:
        if not isinstance(actor, str):
            return False
        actor_key = re.sub(
            r"\s+", "", unicodedata.normalize("NFKC", actor)
        ).casefold()
        if (
            actor != actor.strip()
            or _TARGET_ACTOR_LITERAL.fullmatch(actor) is None
            or not actor_key
            or actor_key in normalized
            or is_unsafe_target_actor_literal(actor)
        ):
            return False
        normalized.add(actor_key)
    return True


_TARGET_NARRATOR_ATTRIBUTION = re.compile(
    r"^(?P<prefix>(?:旁白|叙述者?|作者旁注)(?:直接|明确)?"
    r"(?:说明|写明|交代|指出|描述|叙述))(?P<actor>.+)$"
)
_TARGET_ZERO_SUBJECT_LEAD = re.compile(
    r"^(?:(?:又|随后|接着|然后|继而|仍|却|再次|并|还|才|终于|"
    r"明确|直接|同时|立即|马上|转而|最终|依然|继续|先|再){1,3})"
)
# A zero-subject continuation must actually omit its subject.  Check this
# *after* consuming the safe connective: checking only ``text.startswith``
# mistakes ``随后她……`` for an inherited subject and can manufacture a
# canonical statement such as ``沈砚随后她……``.  The tuple is deliberately
# conservative and covers first/second/third-person singular and plural
# pronouns plus the reflexive forms admitted elsewhere in this protocol.
_TARGET_ZERO_SUBJECT_OVERT_PRONOUNS = (
    "我", "我们", "咱", "咱们", "你", "你们", "您", "您们",
    "他", "他们", "她", "她们", "它", "它们", "祂", "祂们",
    "其", "本人", "自己",
)
# Chinese has no reliable surface delimiter between a short name and a verb.
# Therefore adjacency alone is not enough to infer an omitted subject.  After
# the connective (and any connective-like adverbs already consumed above), we
# admit only this bounded set of ordinary narrative predicates.  Unknown
# two/three-character names and noun phrases cannot become candidates merely
# because a reviewer might later call them the target actor.  Expanding this
# list is a protocol change and must be backed by positive and other-subject
# fixtures.
_TARGET_ZERO_SUBJECT_PREDICATE = re.compile(
    r"^(?:(?:主动|自愿|亲自|坚决|公开|当场|独自|悄悄|偷偷|"
    r"仍然|依旧|已经|正在|始终|一直|逐渐|渐渐|完全|正式|擅自){0,2})"
    r"(?:"
    r"表示|声明|承认|坦言|答应|允许|要求|命令|决定|选择|坚持|"
    r"拒绝|接受|放弃|撤回|反对|支持|喜欢|讨厌|厌恶|偏爱|"
    r"信任|怀疑|保护|帮助|背叛|服从|违抗|遵守|违反|隐瞒|坦白|道歉|原谅|"
    r"吃下|喝下|饮用|拿起|放下|带走|搬走|交出|归还|销毁|隐藏|"
    r"打开|关闭|进入|离开|走进|退出|救下|救援|攻击|阻止|"
    r"解散|重建|停止|保持|改变|转变|变得|开始|完成|执行|实施"
    r")"
)
_TARGET_ZERO_SUBJECT_BA_CONSTRUCTION = re.compile(
    r"^(?:把|将)[^，,。；;！？!?：:\n]{1,80}"
    r"(?:拿起|放下|带走|搬走|交出|交给|归还|销毁|隐藏|打开|关闭|"
    r"转卖|丢弃|扔掉|送出|救下|保护|移交|撤回|拆除|解散|重建)"
    r"(?:了|掉|出去|回来|完毕)?$"
)
_TARGET_EXPLICIT_OTHER_ACTOR = re.compile(
    r"^[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9·._'-]{1,31}"
    r"(?:本人|亲自|又|也|仍|却|再次)"
    r"(?:把|将|让|请|叫|派|命令|指使|拒绝|允许|要求|表示|声明|"
    r"决定|撤回|放弃|解散|搬|拿|交|救|看见|看到|听见|听到)"
)
_TARGET_NAMED_JOINT_TAIL = re.compile(
    r"^(?:们|两人|二人|双方|众人|大家|各自|另一人|另一个人|其他人|"
    r"她们|他们|共同|一起)"
)
_TARGET_UNSAFE_REALITY = re.compile(
    r"(?:梦境|梦中|幻觉|幻象|想象|假想|幻想|排练|演练|彩排|模拟)"
)
_TARGET_QUESTION = re.compile(r"[？?]$")
_TARGET_NON_FACTUAL = re.compile(
    r"(?:并非事实|不是事实|并未发生|没有发生|未曾发生|纯属虚构)"
)


class TargetBoundDraftReviewTarget(BaseModel):
    """Complete server-owned identity for one focused draft recall target."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    target_ordinal: int = Field(ge=1, le=12, strict=True)
    character: str = Field(min_length=1, max_length=64)
    authorized_aliases: tuple[str, ...] = ()
    dimension: Literal[
        "core_personality",
        "preference",
        "value",
        "relationship_attitude",
        "motivation_goal",
        "speech_pattern",
        "behavior_boundary",
        "contextual_behavior",
        "current_state",
    ]
    trait_key: str = Field(min_length=1, max_length=80)
    comparison_key: str = Field(min_length=1, max_length=160)
    key_object: str = Field(default="", max_length=80)
    requested_polarity: Literal["positive", "negative"]
    baseline_hint: str = Field(min_length=1, max_length=320)
    approved_axis_id: str | None = Field(default=None, max_length=64)
    approved_axis_version: int | None = Field(default=None, ge=1, strict=True)
    approved_axis_definition: str | None = Field(
        default=None, min_length=1, max_length=200
    )
    approved_axis_definition_sha256: str | None = Field(
        default=None, pattern=_SHA256
    )
    approved_axis_comparison_key: str | None = Field(
        default=None, min_length=1, max_length=160
    )
    approved_axis_applicability_scope: str | None = Field(
        default=None, min_length=1, max_length=200
    )
    approved_axis_applicability_scope_sha256: str | None = Field(
        default=None, pattern=_SHA256
    )
    axis_positive_proposition: str | None = Field(
        default=None, min_length=1, max_length=200
    )
    axis_positive_proposition_sha256: str | None = Field(
        default=None, pattern=_SHA256
    )

    @model_validator(mode="after")
    def validate_server_target(self) -> "TargetBoundDraftReviewTarget":
        if not _target_actor_literals_are_safe(
            self.character, self.authorized_aliases
        ):
            raise ValueError("target_bound_actor_invalid")
        if self.dimension in _TARGET_OBJECT_DIMENSIONS and not self.key_object.strip():
            raise ValueError("target_bound_object_required")
        if self.key_object and (
            self.key_object != self.key_object.strip()
            or any(
                unicodedata.category(character).startswith("C")
                for character in self.key_object
            )
        ):
            raise ValueError("target_bound_object_invalid")
        axis_values = (
            self.approved_axis_id,
            self.approved_axis_version,
            self.approved_axis_definition,
            self.approved_axis_definition_sha256,
        )
        if any(value is not None for value in axis_values) and any(
            value is None for value in axis_values
        ):
            raise ValueError("target_bound_axis_incomplete")
        scoped_values = (
            self.approved_axis_comparison_key,
            self.approved_axis_applicability_scope,
            self.approved_axis_applicability_scope_sha256,
            self.axis_positive_proposition,
            self.axis_positive_proposition_sha256,
        )
        if any(value is not None for value in scoped_values):
            if (
                self.dimension not in _TARGET_SCOPED_AXIS_DIMENSIONS
                or any(value is None for value in scoped_values)
                or self.approved_axis_id is None
            ):
                raise ValueError("target_bound_scoped_axis_incomplete")
        hashed_pairs = (
            (self.approved_axis_definition, self.approved_axis_definition_sha256),
            (
                self.approved_axis_applicability_scope,
                self.approved_axis_applicability_scope_sha256,
            ),
            (self.axis_positive_proposition, self.axis_positive_proposition_sha256),
        )
        for value, digest in hashed_pairs:
            if value is not None and hashlib.sha256(value.encode("utf-8")).hexdigest() != digest:
                raise ValueError("target_bound_axis_hash_invalid")
        return self


def target_bound_draft_target_digest(target: TargetBoundDraftReviewTarget) -> str:
    if not isinstance(target, TargetBoundDraftReviewTarget):
        raise TypeError("target bound draft target is invalid")
    payload = json.dumps(
        target.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TargetBoundDraftProposal(BaseModel):
    """Server-derived clause proposal; no field is authored by the reviewer."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_sha256: str = Field(pattern=_SHA256)
    line_start: int = Field(ge=1, le=10_000_000, strict=True)
    line_end: int = Field(ge=1, le=10_000_000, strict=True)
    evidence: str = Field(min_length=1, max_length=MAX_DRAFT_ACTOR_EVIDENCE_CHARS)
    fact_clause_id: str = Field(pattern=_SUPPORT_ID)
    actor_anchor_id: str | None = Field(default=None, pattern=_SUPPORT_ID)
    binding_kind: Literal[
        "explicit_named_subject",
        "narrator_explicit_attribution",
        "adjacent_zero_subject",
    ]
    target_digest: str = Field(pattern=_TARGET_DIGEST)
    canonical_statement: str = Field(min_length=2, max_length=300)


class TargetBoundDraftReviewProposal(TargetBoundDraftProposal):
    proposal_id: str = Field(pattern=_TARGET_PROPOSAL_ID)


class TargetBoundDraftReviewRequest(ScopeReviewSourceIdentity):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[
        "character-target-bound-draft-review-v2",
        "character-target-bound-draft-review-v3",
        "character-target-bound-draft-review-v4",
    ] = (
        TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V2
    )
    prompt_version: Literal[
        "character-target-bound-draft-review-prompt-v2",
        "character-target-bound-draft-review-prompt-v3",
        "character-target-bound-draft-review-prompt-v4",
        "character-target-bound-draft-review-prompt-v5",
        "character-target-bound-draft-review-prompt-v6",
        "character-target-bound-draft-review-prompt-v7",
        "character-target-bound-draft-review-prompt-v8",
    ] = (
        TARGET_BOUND_DRAFT_REVIEW_PROMPT_V2
    )
    clause_index_version: Literal["draft-actor-clause-index-v1"] = (
        DRAFT_ACTOR_CLAUSE_INDEX_V1
    )
    target: TargetBoundDraftReviewTarget
    target_digest: str = Field(pattern=_TARGET_DIGEST)
    review_line_start: int = Field(ge=1, le=10_000_000, strict=True)
    review_line_end: int = Field(ge=1, le=10_000_000, strict=True)
    lines: tuple[ScopeReviewLine, ...] = Field(min_length=1, max_length=3)
    proposals: tuple[TargetBoundDraftReviewProposal, ...] = Field(
        min_length=1, max_length=MAX_DRAFT_ACTOR_REVIEW_ITEMS
    )

    @model_validator(mode="after")
    def validate_target_bound_request(self) -> "TargetBoundDraftReviewRequest":
        if (
            self.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V2
            and self.prompt_version != TARGET_BOUND_DRAFT_REVIEW_PROMPT_V2
            or self.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3
            and self.prompt_version != TARGET_BOUND_DRAFT_REVIEW_PROMPT_V7
            or self.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4
            and self.prompt_version != TARGET_BOUND_DRAFT_REVIEW_PROMPT_V8
        ):
            raise ValueError("target_bound_protocol_identity_invalid")
        if self.target_digest != target_bound_draft_target_digest(self.target):
            raise ValueError("target_bound_target_digest_invalid")
        expected_lines = tuple(range(self.review_line_start, self.review_line_end + 1))
        if (
            self.review_line_end - self.review_line_start > 2
            or tuple(line.line_number for line in self.lines) != expected_lines
            or sum(len(line.clauses) for line in self.lines) > MAX_DRAFT_ACTOR_CLAUSES
        ):
            raise ValueError("target_bound_index_invalid")
        clauses = {
            clause.support_id: clause
            for line in self.lines
            for clause in line.clauses
        }
        if len(clauses) != sum(len(line.clauses) for line in self.lines):
            raise ValueError("target_bound_index_invalid")
        proposal_ids: set[str] = set()
        ranges: set[tuple[int, int]] = set()
        for proposal in self.proposals:
            fact = clauses.get(proposal.fact_clause_id)
            anchor = (
                clauses.get(proposal.actor_anchor_id)
                if proposal.actor_anchor_id is not None
                else None
            )
            if (
                proposal.proposal_id in proposal_ids
                or proposal.proposal_id != _target_bound_proposal_id(proposal)
                or proposal.source_sha256 != self.content_sha256
                or proposal.target_digest != self.target_digest
                or fact is None
                or not (
                    self.review_line_start <= proposal.line_start
                    <= proposal.line_end <= self.review_line_end
                    and proposal.line_start <= fact.line_number <= proposal.line_end
                )
                or proposal.binding_kind == "adjacent_zero_subject"
                and (
                    anchor is None
                    or anchor.support_id == fact.support_id
                    or not proposal.line_start <= anchor.line_number <= proposal.line_end
                )
                or proposal.binding_kind != "adjacent_zero_subject"
                and proposal.actor_anchor_id is not None
                or self.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V2
                and proposal.binding_kind == "explicit_named_subject"
            ):
                raise ValueError("target_bound_proposal_invalid")
            proposal_ids.add(proposal.proposal_id)
            ranges.add((proposal.line_start, proposal.line_end))
        if len(ranges) != 1:
            raise ValueError("target_bound_window_mismatch")
        if len(_canonical_target_bound_request_bytes(self)) > (
            MAX_DRAFT_ACTOR_REVIEW_REQUEST_BYTES
        ):
            raise ValueError("target_bound_request_too_large")
        return self


TargetBoundReviewVerdict = Literal["supported", "rejected", "uncertain"]
TargetBoundReviewReason = Literal[
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
TargetBoundSlotConflict = Literal[
    "actor",
    "actuality",
    "statement_relation",
    "axis_relation",
    "object_relation",
    "polarity_relation",
    "correction_relation",
    "observation_kind",
    "object_span",
    "rejected_without_negative_slot",
]


class TargetBoundDraftReviewItem(BaseModel):
    """Only enums and frozen IDs may cross back from the model."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    proposal_id: str = Field(pattern=_TARGET_PROPOSAL_ID)
    verdict: TargetBoundReviewVerdict
    actor: Literal["proposed", "other", "ambiguous"]
    actuality: Literal[
        "asserted", "reported", "hypothetical", "question", "ambiguous"
    ]
    statement_relation: Literal["supported", "contradicted", "ambiguous"]
    axis_relation: Literal[
        "matches_target", "matches_scoped_axis", "different",
        "requires_scoped_review", "ambiguous"
    ]
    object_relation: Literal[
        "matches_target", "broader", "narrower", "different",
        "not_applicable", "ambiguous"
    ]
    polarity_relation: Literal["requested", "opposite", "neutral", "ambiguous"]
    correction_relation: Literal["none", "corrected", "ambiguous"]
    observation_kind: Literal[
        "preference_expression",
        "speech_sample",
        "action",
        "decision",
        "interaction",
        "state_description",
    ]
    # V3/V4 return coordinates, never model-authored object text. Offsets are
    # codepoint positions inside the frozen fact clause and are converted to
    # an exact source slice only after every other semantic slot passes.
    object_start_offset: int | None = Field(default=None, ge=0, le=20_000, strict=True)
    object_end_offset: int | None = Field(default=None, ge=1, le=20_000, strict=True)
    basis_ids: tuple[str, ...] = Field(max_length=MAX_DRAFT_ACTOR_CLAUSES)


class TargetBoundDraftReviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[
        "character-target-bound-draft-review-v2",
        "character-target-bound-draft-review-v3",
        "character-target-bound-draft-review-v4",
    ]
    request_digest: str = Field(pattern=_SHA256)
    items: tuple[TargetBoundDraftReviewItem, ...] = Field(
        max_length=MAX_DRAFT_ACTOR_REVIEW_ITEMS
    )


class TargetBoundDraftReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    proposal_id: str = Field(pattern=_TARGET_PROPOSAL_ID)
    verdict: TargetBoundReviewVerdict
    reason: TargetBoundReviewReason
    basis_ids: tuple[str, ...] = Field(
        default=(), max_length=MAX_DRAFT_ACTOR_CLAUSES
    )
    target_digest: str | None = Field(default=None, pattern=_TARGET_DIGEST)
    target_ordinal: int | None = Field(default=None, ge=1, le=12, strict=True)
    observation_kind: Literal[
        "preference_expression",
        "speech_sample",
        "action",
        "decision",
        "interaction",
        "state_description",
    ] | None = None
    object_relation: Literal[
        "matches_target", "broader", "narrower", "not_applicable"
    ] | None = None
    observed_object: str | None = Field(default=None, max_length=80)
    object_basis_id: str | None = Field(default=None, pattern=_SUPPORT_ID)
    slot_conflicts: tuple[TargetBoundSlotConflict, ...] = Field(
        default=(), exclude=True
    )


class TargetBoundDraftReviewEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    request_digest: str = Field(pattern=_SHA256)
    decisions: tuple[TargetBoundDraftReviewDecision, ...] = Field(
        max_length=MAX_DRAFT_ACTOR_REVIEW_ITEMS
    )


class TargetBoundDraftReviewBatchResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[
        "character-target-bound-draft-review-batch-v2",
        "character-target-bound-draft-review-batch-v3",
        "character-target-bound-draft-review-batch-v4",
    ]
    batch_digest: str = Field(pattern=_SHA256)
    responses: tuple[TargetBoundDraftReviewResponse, ...] = Field(
        min_length=1, max_length=MAX_DRAFT_ACTOR_REVIEW_ITEMS
    )


class TargetBoundDraftReviewBatchEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    batch_digest: str = Field(pattern=_SHA256)
    evaluations: tuple[TargetBoundDraftReviewEvaluation, ...] = Field(
        min_length=1, max_length=MAX_DRAFT_ACTOR_REVIEW_ITEMS
    )


def _target_actor_literals(target: TargetBoundDraftReviewTarget) -> tuple[str, ...]:
    return (target.character, *target.authorized_aliases)


def _compact_actor(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _exact_actor_occurrences(text: str, target: TargetBoundDraftReviewTarget) -> tuple[str, ...]:
    compact = _compact_actor(text)
    return tuple(
        actor for actor in _target_actor_literals(target)
        if _compact_actor(actor) in compact
    )


def _unsafe_target_clause(text: str) -> bool:
    return bool(
        any(mark in text for mark in _QUOTE_MARKS)
        or _BRANCH.search(text)
        or _TARGET_QUESTION.search(text)
        or _TARGET_UNSAFE_REALITY.search(text)
        or _TARGET_NON_FACTUAL.search(text)
    )


def _canonical_named_statement(
    clause: ScopeReviewClause,
    target: TargetBoundDraftReviewTarget,
) -> tuple[str, Literal["explicit_named_subject", "narrator_explicit_attribution"]] | None:
    text = clause.text.strip()
    if _unsafe_target_clause(text):
        return None
    actors = _exact_actor_occurrences(text, target)
    if len(actors) != 1:
        return None
    actor = actors[0]
    if text.startswith(actor):
        tail = text[len(actor):]
        if (
            not tail
            or _NON_DIRECT_NAMED_TAIL.match(tail)
            or _TARGET_NAMED_JOINT_TAIL.match(tail)
        ):
            return None
        statement = target.character + tail
        kind: Literal[
            "explicit_named_subject", "narrator_explicit_attribution"
        ] = "explicit_named_subject"
    else:
        match = _TARGET_NARRATOR_ATTRIBUTION.fullmatch(text)
        if match is None or not match.group("actor").startswith(actor):
            return None
        actor_tail = match.group("actor")[len(actor):]
        if (
            not actor_tail
            or _NON_DIRECT_NAMED_TAIL.match(actor_tail)
            or _TARGET_NAMED_JOINT_TAIL.match(actor_tail)
        ):
            return None
        statement = target.character + actor_tail
        kind = "narrator_explicit_attribution"
    if not 2 <= len(statement) <= 300:
        return None
    return statement, kind


def _zero_subject_statement(
    clause: ScopeReviewClause,
    target: TargetBoundDraftReviewTarget,
) -> str | None:
    text = clause.text.strip()
    lead = _TARGET_ZERO_SUBJECT_LEAD.match(text)
    if (
        not text
        or len(text) + len(target.character) > 300
        or _unsafe_target_clause(text)
        or _VOICE.search(text)
        or _JOINT.search(text)
        or lead is None
        or _exact_actor_occurrences(text, target)
    ):
        return None
    tail = text[lead.end():].lstrip()
    if (
        not tail
        or tail.startswith(_TARGET_ZERO_SUBJECT_OVERT_PRONOUNS)
        or (
            _TARGET_ZERO_SUBJECT_PREDICATE.match(tail) is None
            and _TARGET_ZERO_SUBJECT_BA_CONSTRUCTION.fullmatch(tail) is None
        )
        or _TARGET_EXPLICIT_OTHER_ACTOR.match(tail)
    ):
        return None
    return target.character + text


def _target_bound_proposal_payload(
    proposal: TargetBoundDraftProposal | TargetBoundDraftReviewProposal,
) -> dict[str, object]:
    return {
        name: getattr(proposal, name)
        for name in TargetBoundDraftProposal.model_fields
    }


def _target_bound_proposal_id(
    proposal: TargetBoundDraftProposal | TargetBoundDraftReviewProposal,
) -> str:
    payload = json.dumps(
        _target_bound_proposal_payload(proposal),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"tdp_{hashlib.sha256(payload).hexdigest()[:32]}"


def build_target_bound_draft_proposals(
    frozen_content: str,
    source: ScopeReviewSourceIdentity,
    target: TargetBoundDraftReviewTarget,
    *,
    line_start: int,
    line_end: int,
    protocol_version: Literal["v2", "v3", "v4"] = "v2",
) -> tuple[DraftActorClauseIndex, tuple[TargetBoundDraftProposal, ...]]:
    """Create bounded structural leads; this function never certifies semantics."""

    if protocol_version not in {"v2", "v3", "v4"}:
        raise ValueError("target_bound_protocol_invalid")
    index = build_draft_actor_clause_index(
        frozen_content, source, line_start=line_start, line_end=line_end
    )
    # Recheck at the actor-literal boundary instead of relying exclusively on
    # model construction.  It keeps restored/tampered model instances from
    # using a generic pronoun as if it were a unique formal alias.
    if not _target_actor_literals_are_safe(
        target.character, target.authorized_aliases
    ):
        return index, ()
    evidence = _exact_evidence(frozen_content, line_start, line_end)
    if (
        any(mark in evidence for mark in _QUOTE_MARKS)
        or _BRANCH.search(evidence)
        or _TARGET_UNSAFE_REALITY.search(evidence)
        or _TARGET_QUESTION.search(evidence.rstrip())
    ):
        return index, ()
    digest = target_bound_draft_target_digest(target)
    ordered = tuple(clause for line in index.lines for clause in line.clauses)
    proposals: list[TargetBoundDraftProposal] = []
    named_positions: list[int] = []
    for position, clause in enumerate(ordered):
        named = _canonical_named_statement(clause, target)
        if named is None:
            continue
        statement, kind = named
        # V2 deliberately left ordinary named subjects to the primary
        # extractor. V3 is a bounded recall fallback, so the same frozen
        # clause may be proposed when that extractor returned no usable
        # observation. The independent reviewer still proves all semantics.
        if kind == "narrator_explicit_attribution" or protocol_version in {"v3", "v4"}:
            proposals.append(TargetBoundDraftProposal(
                source_sha256=source.content_sha256,
                line_start=line_start,
                line_end=line_end,
                evidence=evidence,
                fact_clause_id=clause.support_id,
                binding_kind=kind,
                target_digest=digest,
                canonical_statement=statement,
            ))
        named_positions.append(position)

    # A zero-subject fact can inherit only the immediately previous direct
    # named/narrator clause.  The semantic reviewer must still prove actor,
    # actuality, axis/object and direction; adjacency is never treated as that
    # proof by itself.
    for position in named_positions:
        next_position = position + 1
        if next_position >= len(ordered):
            continue
        anchor = ordered[position]
        fact = ordered[next_position]
        if not (
            fact.line_number == anchor.line_number
            or fact.line_number == anchor.line_number + 1
            and index.lines[0].clauses[-1] == anchor
            and index.lines[-1].clauses[0] == fact
        ):
            continue
        statement = _zero_subject_statement(fact, target)
        if statement is None:
            continue
        proposals.append(TargetBoundDraftProposal(
            source_sha256=source.content_sha256,
            line_start=line_start,
            line_end=line_end,
            evidence=evidence,
            fact_clause_id=fact.support_id,
            actor_anchor_id=anchor.support_id,
            binding_kind="adjacent_zero_subject",
            target_digest=digest,
            canonical_statement=statement,
        ))
    unique: dict[str, TargetBoundDraftProposal] = {}
    for proposal in proposals:
        unique.setdefault(_target_bound_proposal_id(proposal), proposal)
    return index, tuple(unique.values())


def _canonical_target_bound_request_bytes(
    request: TargetBoundDraftReviewRequest,
) -> bytes:
    return json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def target_bound_draft_review_request_digest(
    request: TargetBoundDraftReviewRequest,
) -> str:
    if not isinstance(request, TargetBoundDraftReviewRequest):
        raise TypeError("target bound draft review request is invalid")
    return hashlib.sha256(_canonical_target_bound_request_bytes(request)).hexdigest()


def target_bound_draft_review_batch_digest(
    requests: tuple[TargetBoundDraftReviewRequest, ...],
) -> str:
    if (
        type(requests) is not tuple
        or not 1 <= len(requests) <= MAX_DRAFT_ACTOR_REVIEW_ITEMS
        or any(not isinstance(request, TargetBoundDraftReviewRequest) for request in requests)
    ):
        raise TypeError("target bound draft review batch is invalid")
    digests = tuple(target_bound_draft_review_request_digest(item) for item in requests)
    if (
        len(set(digests)) != len(digests)
        or len({request.schema_version for request in requests}) != 1
        or sum(len(request.proposals) for request in requests)
        > MAX_DRAFT_ACTOR_REVIEW_ITEMS
    ):
        raise ValueError("target_bound_batch_duplicate_or_too_large")
    payload = json.dumps(
        [request.model_dump(mode="json") for request in requests],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > MAX_DRAFT_ACTOR_REVIEW_BATCH_REQUEST_BYTES:
        raise ValueError("target_bound_batch_too_large")
    return hashlib.sha256(payload).hexdigest()


def required_target_bound_draft_basis_ids(
    request: TargetBoundDraftReviewRequest,
) -> tuple[str, ...]:
    if not isinstance(request, TargetBoundDraftReviewRequest):
        raise TypeError("target bound draft review request is invalid")
    return tuple(
        clause.support_id for line in request.lines for clause in line.clauses
    )


def build_target_bound_draft_review_request(
    index: DraftActorClauseIndex,
    target: TargetBoundDraftReviewTarget,
    proposals: tuple[TargetBoundDraftProposal, ...],
    *,
    frozen_content: str,
    expected_source: ScopeReviewSourceIdentity,
    protocol_version: Literal["v2", "v3", "v4"] = "v2",
) -> TargetBoundDraftReviewRequest:
    if (
        not isinstance(index, DraftActorClauseIndex)
        or not isinstance(target, TargetBoundDraftReviewTarget)
        or type(proposals) is not tuple
        or not 1 <= len(proposals) <= MAX_DRAFT_ACTOR_REVIEW_ITEMS
        or index.source != expected_source
    ):
        raise TypeError("target bound draft review input is invalid")
    rebuilt_index, rebuilt_proposals = build_target_bound_draft_proposals(
        frozen_content,
        expected_source,
        target,
        line_start=index.line_start,
        line_end=index.line_end,
        protocol_version=protocol_version,
    )
    requested = {
        json.dumps(
            proposal.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ): proposal
        for proposal in proposals
    }
    rebuilt = {
        json.dumps(
            proposal.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ): proposal
        for proposal in rebuilt_proposals
    }
    if rebuilt_index != index or set(requested) - set(rebuilt):
        raise ValueError("target_bound_proposal_not_eligible")
    review_proposals = tuple(
        TargetBoundDraftReviewProposal(
            **requested[key].model_dump(mode="python"),
            proposal_id=_target_bound_proposal_id(requested[key]),
        )
        for key in sorted(requested)
    )
    review_end = _review_context_end(
        frozen_content,
        line_start=index.line_start,
        proposal_line_end=index.line_end,
    )
    lines = _build_draft_actor_review_lines(
        frozen_content,
        expected_source,
        review_line_start=index.line_start,
        review_line_end=review_end,
    )
    return TargetBoundDraftReviewRequest(
        **expected_source.model_dump(mode="python"),
        schema_version=(
            TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4
            if protocol_version == "v4"
            else (
                TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3
                if protocol_version == "v3"
                else TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V2
            )
        ),
        prompt_version=(
            TARGET_BOUND_DRAFT_REVIEW_PROMPT_V8
            if protocol_version == "v4"
            else (
                TARGET_BOUND_DRAFT_REVIEW_PROMPT_V7
                if protocol_version == "v3"
                else TARGET_BOUND_DRAFT_REVIEW_PROMPT_V2
            )
        ),
        target=target,
        target_digest=target_bound_draft_target_digest(target),
        review_line_start=index.line_start,
        review_line_end=review_end,
        lines=lines,
        proposals=review_proposals,
    )


def verify_target_bound_draft_review_source(
    request: TargetBoundDraftReviewRequest,
    expected_source: ScopeReviewSourceIdentity,
    *,
    frozen_content: str,
) -> bool:
    if not isinstance(request, TargetBoundDraftReviewRequest):
        raise TypeError("target bound draft request is invalid")
    if any(
        getattr(request, name) != getattr(expected_source, name)
        for name in ScopeReviewSourceIdentity.model_fields
    ):
        return False
    try:
        if hashlib.sha256(frozen_content.encode("utf-8")).hexdigest() != request.content_sha256:
            return False
        ranges = {(item.line_start, item.line_end) for item in request.proposals}
        if len(ranges) != 1:
            return False
        line_start, line_end = next(iter(ranges))
        index, eligible = build_target_bound_draft_proposals(
            frozen_content,
            expected_source,
            request.target,
            line_start=line_start,
            line_end=line_end,
            protocol_version=(
                "v4"
                if request.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4
                else (
                    "v3"
                    if request.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3
                    else "v2"
                )
            ),
        )
        eligible_ids = {_target_bound_proposal_id(item) for item in eligible}
        if any(item.proposal_id not in eligible_ids for item in request.proposals):
            return False
        rebuilt = build_target_bound_draft_review_request(
            index,
            request.target,
            tuple(
                TargetBoundDraftProposal.model_validate(
                    _target_bound_proposal_payload(item), strict=True
                )
                for item in request.proposals
            ),
            frozen_content=frozen_content,
            expected_source=expected_source,
            protocol_version=(
                "v4"
                if request.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4
                else (
                    "v3"
                    if request.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3
                    else "v2"
                )
            ),
        )
    except (TypeError, ValueError, ValidationError, UnicodeError):
        return False
    return rebuilt == request


def _uncertain_target_bound_review(
    request: TargetBoundDraftReviewRequest,
    reason: TargetBoundReviewReason,
) -> TargetBoundDraftReviewEvaluation:
    return TargetBoundDraftReviewEvaluation(
        request_digest=target_bound_draft_review_request_digest(request),
        decisions=tuple(
            TargetBoundDraftReviewDecision(
                proposal_id=proposal.proposal_id,
                verdict="uncertain",
                reason=reason,
            )
            for proposal in request.proposals
        ),
    )


def _target_bound_supported_conflicts(
    request: TargetBoundDraftReviewRequest,
    proposal: TargetBoundDraftReviewProposal,
    item: TargetBoundDraftReviewItem,
) -> tuple[TargetBoundSlotConflict, ...]:
    conflicts: list[TargetBoundSlotConflict] = []
    if item.actor != "proposed":
        conflicts.append("actor")
    if item.actuality != "asserted":
        conflicts.append("actuality")
    if item.statement_relation != "supported":
        conflicts.append("statement_relation")
    v3 = request.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3
    v4 = request.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4
    semantic_protocol = v3 or v4
    expected_axis = (
        "matches_scoped_axis" if semantic_protocol else "requires_scoped_review"
    ) if request.target.approved_axis_comparison_key is not None else "matches_target"
    if item.axis_relation != expected_axis:
        conflicts.append("axis_relation")
    literal_object_required = _target_bound_literal_object_required(request.target)
    if semantic_protocol and literal_object_required:
        object_valid = item.object_relation in {"matches_target", "broader", "narrower"}
    else:
        expected_object = (
            "matches_target"
            if request.target.dimension in _TARGET_OBJECT_DIMENSIONS
            and request.target.approved_axis_comparison_key is None
            else "not_applicable"
        )
        object_valid = item.object_relation == expected_object
    if not object_valid:
        conflicts.append("object_relation")
    if item.polarity_relation != "requested":
        conflicts.append("polarity_relation")
    if item.correction_relation != "none":
        conflicts.append("correction_relation")
    if not item.observation_kind:
        conflicts.append("observation_kind")
    if semantic_protocol:
        object_fields_present = {
            "object_start_offset", "object_end_offset"
        } <= item.model_fields_set
        if literal_object_required:
            server_binding = (
                v4
                and item.object_relation == "matches_target"
                and target_bound_object_binding_hint(
                    request, proposal
                ).mode == "server_unique_target_literal"
            )
            if server_binding:
                object_span_invalid = (
                    not object_fields_present
                    or item.object_start_offset is not None
                    or item.object_end_offset is not None
                )
            else:
                object_span_invalid = (
                    not object_fields_present
                    or item.object_start_offset is None
                    or item.object_end_offset is None
                    or item.object_start_offset >= item.object_end_offset
                )
            if object_span_invalid:
                conflicts.append("object_span")
        elif (
            not object_fields_present
            or item.object_start_offset is not None
            or item.object_end_offset is not None
        ):
            conflicts.append("object_span")
    elif item.object_start_offset is not None or item.object_end_offset is not None:
        conflicts.append("object_span")
    return tuple(conflicts)


def _target_bound_literal_object_required(
    target: TargetBoundDraftReviewTarget,
) -> bool:
    """Return whether the semantic protocols bind a draft object literal.

    An approved scoped value/behavior-boundary axis is already frozen by its
    definition, applicability scope, positive proposition and comparison key.
    That comparison-key tail is an axis identity, not necessarily a phrase in
    the draft.  All other object-bearing dimensions retain the literal span
    gate unchanged.
    """

    return bool(
        target.dimension in _TARGET_OBJECT_DIMENSIONS
        and not (
            target.dimension in _TARGET_SCOPED_AXIS_DIMENSIONS
            and target.approved_axis_comparison_key is not None
        )
    )


TargetBoundObjectBindingMode = Literal[
    "server_unique_target_literal", "model_span", "not_applicable"
]


@dataclass(frozen=True, slots=True)
class TargetBoundObjectBindingHint:
    """Server-recomputed V4 object-coordinate policy for one proposal."""

    mode: TargetBoundObjectBindingMode
    start_offset: int | None = None
    end_offset: int | None = None


def _raw_literal_occurrences(text: str, literal: str) -> tuple[int, ...]:
    """Return raw-codepoint starts, including overlapping occurrences."""

    if not literal:
        return ()
    starts: list[int] = []
    cursor = 0
    while cursor <= len(text) - len(literal):
        found = text.find(literal, cursor)
        if found < 0:
            break
        starts.append(found)
        cursor = found + 1
    return tuple(starts)


def target_bound_object_binding_hint(
    request: TargetBoundDraftReviewRequest,
    proposal: TargetBoundDraftReviewProposal,
) -> TargetBoundObjectBindingHint:
    """Derive one V4 binding mode only from the frozen request.

    The model-facing hint is never trusted on return.  Evaluation calls this
    function again against the request's own fact clause and target literal.
    """

    if (
        not isinstance(request, TargetBoundDraftReviewRequest)
        or not isinstance(proposal, TargetBoundDraftReviewProposal)
        or proposal not in request.proposals
    ):
        raise TypeError("target bound object binding input is invalid")
    if not _target_bound_literal_object_required(request.target):
        return TargetBoundObjectBindingHint(mode="not_applicable")
    if request.schema_version != TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4:
        return TargetBoundObjectBindingHint(mode="model_span")
    clauses = {
        clause.support_id: clause.text
        for line in request.lines
        for clause in line.clauses
    }
    fact = clauses.get(proposal.fact_clause_id)
    if not isinstance(fact, str):
        raise ValueError("target bound object binding fact is invalid")
    occurrences = _raw_literal_occurrences(fact, request.target.key_object)
    if len(occurrences) != 1:
        return TargetBoundObjectBindingHint(mode="model_span")
    start = occurrences[0]
    return TargetBoundObjectBindingHint(
        mode="server_unique_target_literal",
        start_offset=start,
        end_offset=start + len(request.target.key_object),
    )


def _target_bound_observed_object(
    request: TargetBoundDraftReviewRequest,
    proposal: TargetBoundDraftReviewProposal,
    item: TargetBoundDraftReviewItem,
) -> tuple[str | None, str | None]:
    """Validate and bind one exact immutable fact-clause object."""

    if request.schema_version not in {
        TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3,
        TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4,
    }:
        return None, None
    if not _target_bound_literal_object_required(request.target):
        return None, None
    clauses = {
        clause.support_id: clause
        for line in request.lines
        for clause in line.clauses
    }
    fact = clauses.get(proposal.fact_clause_id)
    if fact is None:
        return None, None
    if request.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4:
        binding = target_bound_object_binding_hint(request, proposal)
        if (
            binding.mode == "server_unique_target_literal"
            and item.object_relation == "matches_target"
        ):
            if (
                not {"object_start_offset", "object_end_offset"}
                <= item.model_fields_set
                or item.object_start_offset is not None
                or item.object_end_offset is not None
                or binding.start_offset is None
                or binding.end_offset is None
            ):
                return None, None
            return request.target.key_object, fact.support_id
    start = item.object_start_offset
    end = item.object_end_offset
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start < end <= len(fact.text)
    ):
        return None, None
    observed = fact.text[start:end]
    if (
        observed != observed.strip()
        or not observed
        or len(observed) > 80
        or all(not char.isalnum() for char in observed)
        or any(char in "\r\n\x00" for char in observed)
    ):
        return None, None
    # When the frozen target object has one unambiguous literal occurrence,
    # the model-selected span must cover that complete occurrence.  This is a
    # server-verifiable boundary: a nearby noun fragment cannot self-certify
    # as a broader/narrower object merely by choosing valid source offsets.
    literal = request.target.key_object
    occurrences = _raw_literal_occurrences(fact.text, literal)
    if request.schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3:
        if len(occurrences) == 1:
            literal_start = occurrences[0]
            literal_end = literal_start + len(literal)
            if start > literal_start or end < literal_end:
                return None, None
    elif occurrences and not any(
        start <= literal_start
        and end >= literal_start + len(literal)
        for literal_start in occurrences
    ):
        return None, None
    return observed, fact.support_id


def _target_bound_negative_slot(item: TargetBoundDraftReviewItem) -> bool:
    return (
        item.actor == "other"
        or item.actuality in {"reported", "hypothetical", "question"}
        or item.statement_relation == "contradicted"
        or item.axis_relation == "different"
        or item.object_relation == "different"
        or item.polarity_relation in {"opposite", "neutral"}
        or item.correction_relation == "corrected"
    )


def _target_bound_source_context_veto(
    request: TargetBoundDraftReviewRequest,
    proposal: TargetBoundDraftReviewProposal,
) -> bool:
    # Reuse the mature V1 trailing correction checks through a temporary V1
    # view.  The character/statement fields are server-frozen here, not model
    # output.  Direct/narrator proposals use their fact as both boundary ends;
    # the V1 helper only needs the two IDs to locate the trailing union.
    actor_anchor = proposal.actor_anchor_id or proposal.fact_clause_id
    view = DraftActorReviewProposal(
        source_sha256=proposal.source_sha256,
        line_start=proposal.line_start,
        line_end=proposal.line_end,
        evidence=proposal.evidence,
        target_clause_id=proposal.fact_clause_id,
        actor_anchor_id=actor_anchor,
        anchor_kind=(
            "verified_prior_named_anchor"
            if proposal.actor_anchor_id is not None
            else "same_line_corroboration"
        ),
        character=request.target.character,
        statement=proposal.canonical_statement,
        proposal_id="dap_" + "0" * 32,
    )
    return _has_draft_actor_source_context_veto(request, view)


def evaluate_target_bound_draft_review(
    request: TargetBoundDraftReviewRequest,
    raw_response: str,
    *,
    expected_source: ScopeReviewSourceIdentity,
    frozen_content: str,
) -> TargetBoundDraftReviewEvaluation:
    if not isinstance(request, TargetBoundDraftReviewRequest) or not isinstance(raw_response, str):
        raise TypeError("target bound draft review arguments are invalid")
    if not verify_target_bound_draft_review_source(
        request, expected_source, frozen_content=frozen_content
    ):
        return _uncertain_target_bound_review(request, "source_mismatch")
    try:
        if len(raw_response.encode("utf-8")) > MAX_DRAFT_ACTOR_REVIEW_RESPONSE_BYTES:
            return _uncertain_target_bound_review(request, "response_too_large")
        json.loads(
            raw_response,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
        response = TargetBoundDraftReviewResponse.model_validate_json(
            raw_response, strict=True
        )
    except (UnicodeError, ValidationError, ValueError, TypeError, RecursionError):
        return _uncertain_target_bound_review(request, "response_invalid")
    if response.schema_version != request.schema_version:
        return _uncertain_target_bound_review(request, "response_mismatch")
    if response.request_digest != target_bound_draft_review_request_digest(request):
        return _uncertain_target_bound_review(request, "response_mismatch")
    requested = {item.proposal_id: item for item in request.proposals}
    returned = {item.proposal_id: item for item in response.items}
    if len(returned) != len(response.items) or set(returned) != set(requested):
        return _uncertain_target_bound_review(request, "response_mismatch")
    basis = required_target_bound_draft_basis_ids(request)
    allowed = set(basis)
    decisions: list[TargetBoundDraftReviewDecision] = []
    for proposal in request.proposals:
        item = returned[proposal.proposal_id]
        basis_valid = (
            len(set(item.basis_ids)) == len(item.basis_ids)
            and set(item.basis_ids) <= allowed
            and (item.basis_ids == basis or item.verdict == "uncertain" and not item.basis_ids)
        )
        if not basis_valid:
            decisions.append(TargetBoundDraftReviewDecision(
                proposal_id=proposal.proposal_id,
                verdict="uncertain",
                reason="basis_invalid",
            ))
            continue
        if item.verdict == "supported":
            conflicts = _target_bound_supported_conflicts(request, proposal, item)
            observed_object: str | None = None
            object_basis_id: str | None = None
            if (
                not conflicts
                and request.schema_version in {
                    TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3,
                    TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4,
                }
                and _target_bound_literal_object_required(request.target)
            ):
                observed_object, object_basis_id = _target_bound_observed_object(
                    request, proposal, item
                )
                if observed_object is None or object_basis_id is None:
                    conflicts = ("object_span",)
            if conflicts:
                decisions.append(TargetBoundDraftReviewDecision(
                    proposal_id=proposal.proposal_id,
                    verdict="uncertain",
                    reason="slot_conflict",
                    slot_conflicts=conflicts,
                ))
            elif _target_bound_source_context_veto(request, proposal):
                decisions.append(TargetBoundDraftReviewDecision(
                    proposal_id=proposal.proposal_id,
                    verdict="uncertain",
                    reason="source_context_veto",
                ))
            else:
                decisions.append(TargetBoundDraftReviewDecision(
                    proposal_id=proposal.proposal_id,
                    verdict="supported",
                    reason="supported",
                    basis_ids=basis,
                    target_digest=request.target_digest,
                    target_ordinal=request.target.target_ordinal,
                    observation_kind=item.observation_kind,
                    object_relation=(
                        item.object_relation
                        if item.object_relation in {
                            "matches_target", "broader", "narrower", "not_applicable"
                        }
                        else None
                    ),
                    observed_object=observed_object,
                    object_basis_id=object_basis_id,
                ))
        elif item.verdict == "rejected":
            negative = _target_bound_negative_slot(item)
            decisions.append(TargetBoundDraftReviewDecision(
                proposal_id=proposal.proposal_id,
                verdict="rejected" if negative else "uncertain",
                reason="reviewer_rejected" if negative else "slot_conflict",
                basis_ids=basis if negative else (),
                slot_conflicts=() if negative else ("rejected_without_negative_slot",),
            ))
        else:
            decisions.append(TargetBoundDraftReviewDecision(
                proposal_id=proposal.proposal_id,
                verdict="uncertain",
                reason="reviewer_uncertain",
            ))
    return TargetBoundDraftReviewEvaluation(
        request_digest=target_bound_draft_review_request_digest(request),
        decisions=tuple(decisions),
    )


def uncertain_target_bound_draft_review_batch(
    requests: tuple[TargetBoundDraftReviewRequest, ...],
    reason: TargetBoundReviewReason = "reviewer_uncertain",
) -> TargetBoundDraftReviewBatchEvaluation:
    return TargetBoundDraftReviewBatchEvaluation(
        batch_digest=target_bound_draft_review_batch_digest(requests),
        evaluations=tuple(
            _uncertain_target_bound_review(request, reason) for request in requests
        ),
    )


def evaluate_target_bound_draft_review_batch(
    requests: tuple[TargetBoundDraftReviewRequest, ...],
    raw_response: str,
    *,
    expected_sources: tuple[ScopeReviewSourceIdentity, ...],
    frozen_contents: tuple[str, ...],
) -> TargetBoundDraftReviewBatchEvaluation:
    digest = target_bound_draft_review_batch_digest(requests)
    if (
        not isinstance(raw_response, str)
        or type(expected_sources) is not tuple
        or type(frozen_contents) is not tuple
        or len(expected_sources) != len(requests)
        or len(frozen_contents) != len(requests)
    ):
        raise TypeError("target bound draft batch arguments are invalid")
    if any(
        not verify_target_bound_draft_review_source(
            request, source, frozen_content=content
        )
        for request, source, content in zip(requests, expected_sources, frozen_contents)
    ):
        return uncertain_target_bound_draft_review_batch(requests, "source_mismatch")
    try:
        if len(raw_response.encode("utf-8")) > MAX_DRAFT_ACTOR_REVIEW_RESPONSE_BYTES:
            return uncertain_target_bound_draft_review_batch(requests, "response_too_large")
        json.loads(
            raw_response,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
        response = TargetBoundDraftReviewBatchResponse.model_validate_json(
            raw_response, strict=True
        )
    except (UnicodeError, ValidationError, ValueError, TypeError, RecursionError):
        return uncertain_target_bound_draft_review_batch(requests, "response_invalid")
    expected_batch_schema = (
        TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V4
        if requests[0].schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4
        else (
            TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V3
            if requests[0].schema_version == TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3
            else TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V2
        )
    )
    if response.schema_version != expected_batch_schema:
        return uncertain_target_bound_draft_review_batch(requests, "response_mismatch")
    if requests[0].schema_version in {
        TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V3,
        TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V4,
    } and any(
        item.verdict == "supported"
        and not {"object_start_offset", "object_end_offset"} <= item.model_fields_set
        for nested in response.responses
        for item in nested.items
    ):
        return uncertain_target_bound_draft_review_batch(requests, "response_invalid")
    if response.batch_digest != digest:
        return uncertain_target_bound_draft_review_batch(requests, "response_mismatch")
    requested = {
        target_bound_draft_review_request_digest(request): request
        for request in requests
    }
    returned = {item.request_digest: item for item in response.responses}
    if len(returned) != len(response.responses) or set(returned) != set(requested):
        return uncertain_target_bound_draft_review_batch(requests, "response_mismatch")
    source_map = {
        target_bound_draft_review_request_digest(request): (source, content)
        for request, source, content in zip(requests, expected_sources, frozen_contents)
    }
    evaluations: list[TargetBoundDraftReviewEvaluation] = []
    for request in requests:
        request_digest = target_bound_draft_review_request_digest(request)
        source, content = source_map[request_digest]
        evaluations.append(evaluate_target_bound_draft_review(
            request,
            returned[request_digest].model_dump_json(),
            expected_source=source,
            frozen_content=content,
        ))
    return TargetBoundDraftReviewBatchEvaluation(
        batch_digest=digest,
        evaluations=tuple(evaluations),
    )
