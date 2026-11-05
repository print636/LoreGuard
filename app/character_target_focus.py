"""Bind a review to one confirmed proposition without replacing its context.

The references below prove source identity, not model semantics.  In particular,
an omission in the narrative cannot be upgraded into an observed counterexample.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .domain import EvidenceSpan
from .config import CHARACTER_TARGET_FOCUS_POLICY_V1


TARGET_FOCUS_POLICY_V1 = CHARACTER_TARGET_FOCUS_POLICY_V1
TARGET_FOCUS_REVIEW_KEY = "target_focus_review_v1"


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


class FocusAssertion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    citation: str = Field(pattern=r"^B[0-9]{2}$")
    support_id: str = Field(pattern=r"^L[1-9][0-9]*:A[1-9][0-9]*$")
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    text: str = Field(min_length=1)


class TargetFocus(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    policy: Literal["character-target-focus-v1"] = TARGET_FOCUS_POLICY_V1
    candidate_id: str
    statement: str
    contexts: tuple[str, ...]
    binding_status: Literal["source_bound", "statement_only"]
    target_assertions: tuple[FocusAssertion, ...]
    # These are contextual assertions, not other confirmed candidate objects.
    # They may contain a denial or qualification relevant to the target.
    surrounding_assertions: tuple[FocusAssertion, ...]
    source_sha256: tuple[str, ...]

    @property
    def digest(self) -> str:
        return _sha(self.model_dump(mode="json"))


class CounterEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    citation: str = Field(pattern=r"^C[0-9]{2}$")
    quote: str = Field(min_length=1, max_length=240)


class TargetFocusCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    target_relation: Literal[
        "contradicts", "compatible", "not_established", "different", "unclear"
    ]
    opposition_basis: Literal[
        "explicit", "supported_inference", "description_absence", "none"
    ]
    counterevidence: tuple[CounterEvidence, ...] = Field(max_length=8)


def build_target_focus(
    *, candidate_id: str, statement: str, contexts: tuple[str, ...],
    evidence: tuple[EvidenceSpan, ...], support_bindings: dict | None,
) -> TargetFocus:
    """Rebuild supplied target coordinates from full, immutable evidence lines.

    Old confirmed candidates without assertion bindings use their exact author
    confirmed statement.  Never guess a target span using substring similarity.
    """
    from .character_trait_extraction import CharacterSignalChunk, _assertion_index_v1

    source_hashes = tuple(hashlib.sha256(span.text.encode("utf-8")).hexdigest() for span in evidence)
    if support_bindings is None:
        return TargetFocus(
            candidate_id=candidate_id, statement=statement, contexts=contexts,
            binding_status="statement_only", target_assertions=(),
            surrounding_assertions=(), source_sha256=source_hashes,
        )
    clauses: dict[tuple[int, str], FocusAssertion] = {}
    for index, span in enumerate(evidence):
        chunk = CharacterSignalChunk(
            document_id=span.document_id, document_name=span.document_name,
            source_kind="formal_character_profile", global_line_start=span.line_start,
            content=span.text,
        )
        for clause in _assertion_index_v1(chunk).clauses:
            # Assertion index offsets are relative to their original line.
            lines = span.text.splitlines(keepends=True)
            relative_line = clause.line_number - span.line_start
            prefix = sum(len(line) for line in lines[:relative_line])
            row = FocusAssertion(
                citation=f"B{index + 1:02d}", support_id=clause.support_id,
                start_offset=prefix + clause.start_offset,
                end_offset=prefix + clause.end_offset, text=clause.text,
            )
            if span.text[row.start_offset:row.end_offset] != row.text:
                raise ValueError("target focus source mismatch")
            clauses[index, clause.support_id] = row

    targets: list[FocusAssertion] = []
    if support_bindings is not None:
        if (not isinstance(support_bindings, dict)
                or support_bindings.get("schema_version") != "character-support-bindings-v1"
                or support_bindings.get("index_version") != "assertion-index-v1"
                or not isinstance(support_bindings.get("bindings"), list)
                or not 1 <= len(support_bindings["bindings"]) <= 48):
            raise ValueError("target focus bindings invalid")
        for binding in support_bindings["bindings"]:
            raw = binding.get("target", {})
            key = (binding.get("evidence_index"), binding.get("support_id"))
            row = clauses.get(key)
            if row is None or raw.get("support_id") != row.support_id or raw.get("role") != "target":
                raise ValueError("target focus assertion missing")
            span = evidence[key[0]]
            lines = span.text.splitlines(keepends=True)
            line = int(row.support_id.split(":")[0][1:]) - span.line_start
            prefix = sum(len(value) for value in lines[:line])
            if (raw.get("start_offset") != row.start_offset - prefix
                    or raw.get("end_offset") != row.end_offset - prefix):
                raise ValueError("target focus assertion offsets invalid")
            if row not in targets:
                targets.append(row)
    selected = {(row.citation, row.support_id) for row in targets}
    return TargetFocus(
        candidate_id=candidate_id, statement=statement, contexts=contexts,
        binding_status="source_bound" if targets else "statement_only",
        target_assertions=tuple(targets),
        surrounding_assertions=tuple(
            row for row in clauses.values()
            if (row.citation, row.support_id) not in selected
        ),
        source_sha256=source_hashes,
    )


def validate_focus_check(
    check: TargetFocusCheck, *, verdict: str, trait_match: str,
    current_citations: tuple[str, ...], current: dict[str, EvidenceSpan],
) -> list[dict]:
    """Check source/contract consistency.  Meaning remains the model's judgment."""
    counter_refs = []
    seen = set()
    for reference in check.counterevidence:
        if reference.citation not in current_citations:
            raise ValueError("counterevidence outside reviewed event")
        span = current[reference.citation]
        start = span.text.find(reference.quote)
        if start < 0 or span.text.find(reference.quote, start + 1) >= 0:
            raise ValueError("counterevidence source not unique")
        identity = (reference.citation, start, start + len(reference.quote))
        if identity in seen:
            raise ValueError("duplicate counterevidence")
        seen.add(identity)
        counter_refs.append({
            "citation": reference.citation, "start_offset": start,
            "end_offset": start + len(reference.quote),
            "text_sha256": hashlib.sha256(reference.quote.encode("utf-8")).hexdigest(),
        })
    if check.target_relation == "contradicts":
        if (check.opposition_basis not in {"explicit", "supported_inference"}
                or not counter_refs or trait_match != "same"):
            raise ValueError("target opposition lacks counterevidence")
    elif check.opposition_basis in {"explicit", "supported_inference"} or counter_refs:
        raise ValueError("counterevidence without target opposition")
    if (check.opposition_basis == "description_absence"
            and check.target_relation not in {"not_established", "unclear"}):
        raise ValueError("description absence is not a counterexample")
    if verdict == "contradicts" and check.target_relation != "contradicts":
        raise ValueError("event does not contradict its confirmed target")
    return counter_refs


TARGET_FOCUS_REVIEW_INSTRUCTIONS = """
当前目标命题审查扩展（character-target-focus-v1）：target_focus.statement 是作者确认、此次唯一需要比较的命题，target_focus.contexts 是其限定。target_assertions 是该命题已核对的来源位置；surrounding_assertions 和 B 全文是完整上下文，可能含否认、条件或相邻其他设定。必须保留它们解释原意，但不得把邻近另一条设定的违背算成当前命题的违背。statement_only 表示旧候选没有断言位置，仍只比较准确的 statement，不猜测位置或扩大目标。trait_key、极性、摘要及 target_assertions 的存在本身都不证明冲突。
必须区别“原文没描写某行为”与“原文证实未做该行为”。未描写、只写另一动作或反应，不能当作违背证据。允许原文支持的同义改写和合理语义推断，不要求重复设定字词，也不强制出现否定词；推断必须有具体已发生的事实支撑，不能仅由叙事留白推导。解释须说明当前事实怎样违反这一个目标命题，不能改用相邻命题。
顶层额外必须返回 target_focus_sha256，逐字回显输入 target_focus_sha256。每个 event 额外必须返回 target_focus={"target_relation":"not_established","opposition_basis":"description_absence","counterevidence":[]}。target_relation 取 contradicts、compatible、not_established、different、unclear，分别表示当前事实反驳目标、可与目标并存、未建立目标方向、针对另一命题、无法判断。opposition_basis 取 explicit、supported_inference、description_absence、none。只有反驳当前命题时可取 explicit/supported_inference，并必须给 counterevidence 数组，每项为 {"citation":"C01","quote":"该事件C原文中可唯一定位的反例短引文"}，每条最多240字；非反驳时数组为空。仅缺描写使用 description_absence，不能返回 contradicts。反例引文必须在该事件组内；不能借另一事件或 B 原文充当当前反例。compatible/not_established/different/unclear 本身不证明全稿无问题，继续遵守原有覆盖和解释规则。例外可以解释一个表面反向事实，是否正式报告仍由原协议决定。
"""

TARGET_FOCUS_SIGNAL_INSTRUCTIONS = """
当前命题提示扩展（character-target-focus-v1）：confirmed_traits 中的 confirmed_target_statement 以及 targeted 的 baseline_hint 是对应作者确认命题，只作为比较语义提示，不是当前事实。不要只按 trait_key 字面猜方向，也不要把同一来源句中的另一条设定并入此目标。原文未描写某行为，不等于原文证实角色没有做该行为；不能仅凭留白标为该命题反面。允许有原文事实支持的同义表达和合理推断，不要求出现同字串或固定否定词。仍只输出原协议允许的可定位记录，目标提示/限定本身不是证据，抽取不能代替主审判断冲突。
"""
