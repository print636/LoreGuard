from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .domain import IssueCategory, ParsedDirective
from .semantic_quality import eligible_for_deterministic_rules


MAX_LINE_NUMBER = 10_000_000
SEED_REF_PATTERN = r"^seed_[a-f0-9]{32}$"
_RESULT_REF_PATTERN = r"^result_[A-Za-z0-9_-]{16,64}$"
_SPAN_REF_PATTERN = r"^span_[A-Za-z0-9_-]{16,64}$"
HASH_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_FIELD_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

CandidateKind: TypeAlias = Literal[
    "fact",
    "event",
    "knows",
    "claims_knows",
    "item",
    "uses",
    "world_rule",
    "world_assert",
]


@dataclass(frozen=True, slots=True)
class CandidateFieldContract:
    """Shared immutable shape used for model guidance and promotion checks."""

    required: tuple[str, ...]
    optional: tuple[str, ...] = ()
    semantic_guidance: str = ""

    def __post_init__(self) -> None:
        fields = self.required + self.optional
        if (
            not self.required
            or len(fields) != len(set(fields))
            or any(_FIELD_PATTERN.fullmatch(value) is None for value in fields)
            or type(self.semantic_guidance) is not str
            or not self.semantic_guidance.strip()
            or len(self.semantic_guidance) > 512
        ):
            raise ValueError("candidate field contract is invalid")


_CANDIDATE_FIELD_CONTRACTS: dict[str, CandidateFieldContract] = {
    "fact": CandidateFieldContract(
        ("subject", "predicate", "value"),
        ("time",),
        (
            "候选必须与 anchor 的 subject、predicate 相同；冲突须为两条肯定事实的 value 不同，"
            "或同一 value 的一肯定一明确否定。"
        ),
    ),
    "event": CandidateFieldContract(
        ("time", "location", "participants"),
        semantic_guidance=(
            "候选必须与 anchor 共享参与者、精确 time 相同且 location 不同。"
        ),
    ),
    "knows": CandidateFieldContract(
        ("character", "fact", "time"),
        semantic_guidance=(
            "仅表示角色已经实际获得该知识。candidate 的 character、fact 必须与 "
            "anchor 分别指向同一角色、同一项知识，不得换入相关角色、相近知识或扩大"
            "知识含义。引用范围须同时支持角色、具体知识、精确 time 和实际获知关系。"
        ),
    ),
    "claims_knows": CandidateFieldContract(
        ("character", "fact", "time"),
        semantic_guidance=(
            "仅表示角色声称知道，不等同于实际获得知识。candidate 的 character、fact "
            "必须与 anchor 分别指向同一角色、同一项知识，不得换入相关角色、相近知识或"
            "扩大知识含义。引用范围须同时支持角色、具体知识、精确 time 和声称关系。"
        ),
    ),
    "item": CandidateFieldContract(
        ("item", "owner"),
        ("time",),
        "仅表示物品的所有或保管关系。",
    ),
    "uses": CandidateFieldContract(
        ("item", "user"),
        ("time",),
        "仅表示角色已经实际使用该物品。",
    ),
    "world_rule": CandidateFieldContract(
        ("key", "value"),
        semantic_guidance="仅表示文本明确建立的世界规则。",
    ),
    "world_assert": CandidateFieldContract(
        ("key", "value"),
        ("actor", "time"),
        (
            "value=performed 仅表示动作真实完成；命令、计划、转述或缺少执行结果时必须 ABSTAIN。"
        ),
    ),
}
if set(_CANDIDATE_FIELD_CONTRACTS) != set(get_args(CandidateKind)):
    raise RuntimeError("candidate field contracts do not cover candidate kinds")


def get_candidate_field_contract(kind: str) -> CandidateFieldContract:
    """Return the immutable contract for one allowed candidate kind."""

    if type(kind) is not str or kind not in _CANDIDATE_FIELD_CONTRACTS:
        raise ValueError("candidate kind is invalid")
    return _CANDIDATE_FIELD_CONTRACTS[kind]


def candidate_kinds() -> tuple[str, ...]:
    """Return every supported kind without exposing the mutable registry."""

    return tuple(_CANDIDATE_FIELD_CONTRACTS)


_FAMILY_SEMANTIC_GUIDANCE: dict[IssueCategory, str] = {
    IssueCategory.fact_conflict: (
        "只调查同一主体同一属性的冲突：肯定取值互异，或同一取值一肯定一明确否定。"
    ),
    IssueCategory.location_collision: (
        "只调查同一参与者在同一精确时间出现在不同地点的冲突。"
    ),
    IssueCategory.knowledge_without_acquisition: (
        "区分实际获得知识与仅声称知道；没有获得证据时不得把声称当作已知。"
        "candidate 的 character、fact 必须与 anchor 分别指向同一角色、同一项知识，"
        "不得换入相关角色、相近知识或扩大知识含义。若原文只说明角色出现于、"
        "接触、持有或拆封信息载体，却未明确说明该角色已听到、看到、读到或得知"
        "candidate 的具体知识，不得推断已获知。当前服务端不解析代词来建立知识关系；"
        "获知或声称关系句必须直接指向 candidate 的角色与具体知识，否则必须 ABSTAIN。候选的"
        "最小引用范围必须同时支持角色、具体知识、精确 time 和对应的获知或声称关系。"
    ),
    IssueCategory.item_ownership: (
        "区分物品的所有或保管关系与已经发生的实际使用。"
    ),
    IssueCategory.world_rule_conflict: (
        "区分世界规则与已经完成的规则相关行为；未完成行为不能作为已执行事实。"
    ),
}
if set(_FAMILY_SEMANTIC_GUIDANCE) != set(IssueCategory):
    raise RuntimeError("family semantic guidance does not cover issue categories")


def get_family_semantic_guidance(family: IssueCategory) -> str:
    """Return fixed model guidance for a server-selected issue family."""

    if not isinstance(family, IssueCategory):
        raise ValueError("issue family is invalid")
    return _FAMILY_SEMANTIC_GUIDANCE[family]


ToolName: TypeAlias = Literal[
    "SEARCH_EVIDENCE", "READ_SPAN", "SUBMIT_VERDICT", "ABSTAIN"
]

_FAMILY_KINDS: dict[IssueCategory, frozenset[str]] = {
    IssueCategory.fact_conflict: frozenset({"fact"}),
    IssueCategory.location_collision: frozenset({"event"}),
    IssueCategory.knowledge_without_acquisition: frozenset(
        {"knows", "claims_knows"}
    ),
    IssueCategory.item_ownership: frozenset({"item", "uses"}),
    IssueCategory.world_rule_conflict: frozenset({"world_rule", "world_assert"}),
}
_RESERVED_CANDIDATE_FIELDS = frozenset(
    {
        "kind",
        "span_ref",
        "source_line_start",
        "source_line_end",
        "run_id",
        "project_id",
        "document_id",
        "document_name",
        "document_version",
        "content_sha256",
        "snapshot",
        "doc_ref",
        "evidence",
        "issue",
        "issue_id",
        "title",
        "severity",
        "confidence",
        "suggestion",
        "metadata",
    }
)
SAFE_REASON_CODES = frozenset(
    {
        "accepted",
        "completed",
        "explicit_abstain",
        "invalid_tool_arguments",
        "unknown_tool",
        "invalid_state",
        "unknown_seed",
        "cross_run",
        "cross_seed",
        "snapshot_mismatch",
        "evidence_hash_mismatch",
        "evidence_range",
        "unknown_result_ref",
        "unknown_span_ref",
        "candidate_kind_forbidden",
        "repeated_action",
        "anchor_evidence_reused",
        "repeated_query",
        "no_progress",
        "round_budget",
        "tool_budget",
        "search_budget",
        "read_budget",
        "result_budget",
        "span_budget",
        "token_budget",
        "deadline",
        "unprocessed_seed",
        "internal_failure",
    }
)


class InvestigatorRejected(ValueError):
    """Content-free rejection safe for a trace or public diagnostic."""

    def __init__(self, reason_code: str) -> None:
        safe = reason_code if reason_code in SAFE_REASON_CODES else "internal_failure"
        super().__init__(safe)
        self.reason_code = safe


class _StrictToolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SearchEvidenceArgs(_StrictToolModel):
    seed_ref: str = Field(pattern=SEED_REF_PATTERN)
    query: str = Field(min_length=3, max_length=800, repr=False)
    entity_terms: list[str] = Field(default_factory=list, max_length=8, repr=False)

    @field_validator("query")
    @classmethod
    def query_is_not_only_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must contain visible text")
        return value

    @field_validator("entity_terms")
    @classmethod
    def entity_terms_are_bounded_and_observable(
        cls, values: list[str], info
    ) -> list[str]:
        query = str(info.data.get("query", "")).casefold()
        seen: set[str] = set()
        for value in values:
            if (
                not isinstance(value, str)
                or value != value.strip()
                or not value
                or len(value) > 80
            ):
                raise ValueError("entity term is invalid")
            folded = value.casefold()
            if folded in seen or folded not in query:
                raise ValueError("entity term must be unique and present in query")
            seen.add(folded)
        return values


class ReadSpanArgs(_StrictToolModel):
    seed_ref: str = Field(pattern=SEED_REF_PATTERN)
    result_ref: str = Field(pattern=_RESULT_REF_PATTERN)
    line_start: int = Field(ge=1, le=MAX_LINE_NUMBER)
    line_end: int = Field(ge=1, le=MAX_LINE_NUMBER)

    @model_validator(mode="after")
    def line_range_is_ordered(self) -> ReadSpanArgs:
        if self.line_end < self.line_start:
            raise ValueError("line range must be ordered")
        return self


class CandidateRecordSubmission(_StrictToolModel):
    """Model proposal shape; every field remains untrusted.

    P2 promotion must add per-kind positive field allowlists, byte limits, and
    rule semantics (for example single-owner validation).  This transport
    contract deliberately grants none of those properties.
    """

    kind: CandidateKind
    span_ref: str = Field(pattern=_SPAN_REF_PATTERN)
    source_line_start: int = Field(ge=1, le=MAX_LINE_NUMBER)
    source_line_end: int = Field(ge=1, le=MAX_LINE_NUMBER)
    fields: dict[str, Any] = Field(min_length=1, max_length=20, repr=False)

    @model_validator(mode="after")
    def candidate_shape_is_safe_but_still_untrusted(
        self,
    ) -> CandidateRecordSubmission:
        if self.source_line_end < self.source_line_start:
            raise ValueError("candidate line range must be ordered")
        if any(
            not isinstance(key, str)
            or _FIELD_PATTERN.fullmatch(key) is None
            or key in _RESERVED_CANDIDATE_FIELDS
            for key in self.fields
        ):
            raise ValueError("candidate contains a reserved or invalid field")
        if not _is_json_value(self.fields) or _contains_reserved_key(self.fields):
            raise ValueError("candidate fields must be finite JSON values")
        return self


class SubmitVerdictArgs(_StrictToolModel):
    seed_ref: str = Field(pattern=SEED_REF_PATTERN)
    verdict: Literal["candidate_conflict"]
    candidates: list[CandidateRecordSubmission] = Field(
        min_length=1, max_length=2, repr=False
    )

    @field_validator("candidates")
    @classmethod
    def candidates_are_unique(
        cls, values: list[CandidateRecordSubmission]
    ) -> list[CandidateRecordSubmission]:
        signatures = [_model_signature(value) for value in values]
        if len(signatures) != len(set(signatures)):
            raise ValueError("candidate submissions must be unique")
        return values


class AbstainArgs(_StrictToolModel):
    seed_ref: str = Field(pattern=SEED_REF_PATTERN)
    reason: Literal[
        "no_relevant_evidence",
        "insufficient_evidence",
        "ambiguous_source",
        "contextual_exception",
        "no_rule_validated_conflict",
    ]


ToolArguments: TypeAlias = (
    SearchEvidenceArgs | ReadSpanArgs | SubmitVerdictArgs | AbstainArgs
)
_TOOL_MODELS: dict[str, type[_StrictToolModel]] = {
    "SEARCH_EVIDENCE": SearchEvidenceArgs,
    "READ_SPAN": ReadSpanArgs,
    "SUBMIT_VERDICT": SubmitVerdictArgs,
    "ABSTAIN": AbstainArgs,
}


def parse_tool_arguments(name: str, arguments: Any) -> ToolArguments:
    """Validate one native tool call without echoing model-controlled values."""

    if type(name) is not str:
        raise InvestigatorRejected("unknown_tool")
    model = _TOOL_MODELS.get(name)
    if model is None:
        raise InvestigatorRejected("unknown_tool")
    if not isinstance(arguments, dict):
        raise InvestigatorRejected("invalid_tool_arguments")
    try:
        return model.model_validate(arguments)
    except (TypeError, ValueError, ValidationError):
        raise InvestigatorRejected("invalid_tool_arguments") from None


@dataclass(frozen=True, slots=True)
class InvestigationSeed:
    seed_ref: str
    run_hash: str
    family: IssueCategory
    anchor_hash: str
    join_key_hash: str
    anchor: ParsedDirective = field(repr=False, compare=False)
    allowed_candidate_kinds: frozenset[str]

    def __post_init__(self) -> None:
        if (
            type(self.seed_ref) is not str
            or re.fullmatch(SEED_REF_PATTERN, self.seed_ref) is None
        ):
            raise ValueError("seed ref is invalid")
        if type(self.run_hash) is not str or HASH_PATTERN.fullmatch(self.run_hash) is None:
            raise ValueError("seed run hash is invalid")
        if (
            type(self.anchor_hash) is not str
            or HASH_PATTERN.fullmatch(self.anchor_hash) is None
        ):
            raise ValueError("seed anchor hash is invalid")
        if (
            type(self.join_key_hash) is not str
            or HASH_PATTERN.fullmatch(self.join_key_hash) is None
        ):
            raise ValueError("seed join key hash is invalid")
        if not isinstance(self.family, IssueCategory):
            raise ValueError("seed family is invalid")
        if type(self.anchor) is not ParsedDirective:
            raise ValueError("seed anchor is invalid")
        if (
            type(self.allowed_candidate_kinds) is not frozenset
            or self.allowed_candidate_kinds != _FAMILY_KINDS[self.family]
        ):
            raise ValueError("seed candidate kinds are invalid")
        try:
            if not eligible_for_deterministic_rules(self.anchor):
                raise ValueError("seed anchor is not deterministic")
            if self.anchor_hash != _directive_hash(self.anchor):
                raise ValueError("seed anchor hash mismatch")
            identities = {
                (family, _sha256(join_key))
                for family, join_key in _seed_keys(self.anchor)
            }
        except (AttributeError, TypeError, ValueError):
            raise ValueError("seed anchor is invalid") from None
        if (self.family, self.join_key_hash) not in identities:
            raise ValueError("seed family or join key mismatch")
        expected_ref = (
            "seed_"
            + _sha256(f"{self.run_hash}:{self.family.value}:{self.join_key_hash}")[:32]
        )
        if self.seed_ref != expected_ref:
            raise ValueError("seed ref does not match its identity")


def build_investigation_seeds(
    run_id: str,
    directives: Sequence[ParsedDirective],
    *,
    limit: int = 8,
) -> tuple[InvestigationSeed, ...]:
    """Create stable, category-balanced seeds from deterministic-rule anchors."""

    if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 256:
        raise ValueError("run id is invalid")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 64:
        raise ValueError("seed limit is invalid")
    try:
        prepared = tuple(directives)
    except TypeError:
        raise ValueError("directives are invalid") from None
    if len(prepared) > 100_000:
        raise ValueError("directive limit was exceeded")

    run_hash = _sha256(run_id)
    candidates: dict[tuple[IssueCategory, str], tuple[str, ParsedDirective]] = {}
    for supplied_directive in prepared:
        if type(supplied_directive) is not ParsedDirective:
            raise ValueError("directive is invalid")
        try:
            directive = _clone_parsed_directive(supplied_directive)
        except (AttributeError, TypeError, ValueError):
            raise ValueError("directive is invalid") from None
        if not eligible_for_deterministic_rules(directive):
            continue
        for family, join_key in _seed_keys(directive):
            anchor_hash = _directive_hash(directive)
            key = (family, join_key)
            current = candidates.get(key)
            if current is None or anchor_hash < current[0]:
                candidates[key] = (anchor_hash, directive)

    by_family: dict[IssueCategory, list[InvestigationSeed]] = {
        family: [] for family in IssueCategory
    }
    for (family, join_key), (anchor_hash, anchor) in candidates.items():
        join_key_hash = _sha256(join_key)
        seed_ref = f"seed_{_sha256(f'{run_hash}:{family.value}:{join_key_hash}')[:32]}"
        by_family[family].append(
            InvestigationSeed(
                seed_ref=seed_ref,
                run_hash=run_hash,
                family=family,
                anchor_hash=anchor_hash,
                join_key_hash=join_key_hash,
                anchor=anchor,
                allowed_candidate_kinds=_FAMILY_KINDS[family],
            )
        )
    for values in by_family.values():
        values.sort(key=lambda row: (row.join_key_hash, row.anchor_hash))

    # Round-robin prevents ordinary facts starving the other four families.
    result: list[InvestigationSeed] = []
    while len(result) < limit and any(by_family.values()):
        for family in IssueCategory:
            if by_family[family] and len(result) < limit:
                result.append(by_family[family].pop(0))
    return tuple(result)


def clone_investigation_seed(seed: InvestigationSeed) -> InvestigationSeed:
    """Detach mutable Pydantic internals and re-run every identity invariant."""

    if type(seed) is not InvestigationSeed:
        raise ValueError("session seed is invalid")
    try:
        anchor = _clone_parsed_directive(seed.anchor)
        return InvestigationSeed(
            seed_ref=seed.seed_ref,
            run_hash=seed.run_hash,
            family=seed.family,
            anchor_hash=seed.anchor_hash,
            join_key_hash=seed.join_key_hash,
            anchor=anchor,
            allowed_candidate_kinds=seed.allowed_candidate_kinds,
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError("session seed is invalid") from None


def _clone_parsed_directive(directive: ParsedDirective) -> ParsedDirective:
    if type(directive) is not ParsedDirective:
        raise ValueError("directive is invalid")
    payload = json.loads(
        json.dumps(
            {
                "kind": directive.kind,
                "attrs": directive.attrs,
                "evidence": directive.evidence.model_dump(
                    mode="json", warnings="error"
                ),
                "noncanonical_frame": directive.noncanonical_frame,
                "provenance_sources": sorted(directive.provenance_sources),
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    return ParsedDirective.model_validate(payload)


def _seed_keys(directive: ParsedDirective) -> tuple[tuple[IssueCategory, str], ...]:
    attrs = directive.attrs
    if directive.kind == "fact":
        subject = attrs.get("subject", "").strip()
        predicate = attrs.get("predicate", "").strip()
        if (
            subject
            and predicate
            and predicate
            not in {"mobility_permission", "mobility_limit", "rule_exception"}
        ):
            return (
                (
                    IssueCategory.fact_conflict,
                    _canonical_join_key("fact", subject, predicate),
                ),
            )
    elif directive.kind == "event":
        timestamp = attrs.get("time", "").strip()
        location = attrs.get("location", "").strip()
        participants = {
            value.strip()
            for value in attrs.get("participants", "").split(",")
            if value.strip()
        }
        if (
            re.fullmatch(
                r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?", timestamp
            )
            and location
            and participants
        ):
            return tuple(
                (
                    IssueCategory.location_collision,
                    _canonical_join_key("event", participant, timestamp),
                )
                for participant in sorted(participants)
            )
    elif directive.kind in {"knows", "claims_knows"}:
        character = attrs.get("character", "").strip()
        fact = attrs.get("fact", "").strip()
        if character and fact:
            return (
                (
                    IssueCategory.knowledge_without_acquisition,
                    _canonical_join_key("knowledge", character, fact),
                ),
            )
    elif directive.kind in {"item", "uses"}:
        item = attrs.get("item", "").strip()
        if item:
            return (
                (IssueCategory.item_ownership, _canonical_join_key("item", item)),
            )
    elif directive.kind in {"world_rule", "world_assert"}:
        key = attrs.get("key", "").strip()
        if key:
            return (
                (
                    IssueCategory.world_rule_conflict,
                    _canonical_join_key("world_rule", key),
                ),
            )
    return ()


def _canonical_join_key(kind: str, *parts: str) -> str:
    return json.dumps(
        [kind, *parts],
        ensure_ascii=False,
        sort_keys=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def _directive_hash(directive: ParsedDirective) -> str:
    payload = {
        "kind": directive.kind,
        "attrs": directive.attrs,
        "document_id": directive.evidence.document_id,
        "document_name": directive.evidence.document_name,
        "line_start": directive.evidence.line_start,
        "line_end": directive.evidence.line_end,
        "evidence_text": directive.evidence.text,
        "noncanonical_frame": directive.noncanonical_frame,
        "provenance_sources": sorted(directive.provenance_sources),
    }
    return _sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _model_signature(model: BaseModel) -> str:
    return _sha256(
        json.dumps(
            model.model_dump(mode="json", warnings="error"),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    )


def _is_json_value(value: Any, *, depth: int = 0) -> bool:
    if depth > 16:
        return False
    if value is None or isinstance(value, (str, bool, int)):
        return True
    if isinstance(value, float):
        return value == value and value not in {float("inf"), float("-inf")}
    if isinstance(value, list):
        return all(_is_json_value(row, depth=depth + 1) for row in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(row, depth=depth + 1)
            for key, row in value.items()
        )
    return False


def _contains_reserved_key(value: Any, *, depth: int = 0) -> bool:
    if depth > 16:
        return True
    if isinstance(value, list):
        return any(_contains_reserved_key(row, depth=depth + 1) for row in value)
    if isinstance(value, dict):
        return any(
            key in _RESERVED_CANDIDATE_FIELDS
            or _contains_reserved_key(row, depth=depth + 1)
            for key, row in value.items()
        )
    return False


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
