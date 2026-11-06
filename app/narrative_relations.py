"""Source-bound temporal and epistemic relations from one extraction request.

The model judges meaning.  The server checks source identity, scope and a
bounded, direct relation; it never invents a calendar date or acquisition.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .domain import EvidenceSpan, ParsedDirective

RELATION_KINDS = frozenset({"narrative_time_relation", "knowledge_constraint"})
RELATION_SCHEMA = "narrative-time-knowledge-v1"


class TimePointRef(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    doc_ref: str = Field(pattern=r"^d[1-9][0-9]*$")
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    time: str = Field(min_length=1, max_length=80)


class RelationExtractionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_line_start: int = Field(ge=1)
    source_line_end: int = Field(ge=1)
    modality: Literal["asserted", "uncertain", "hypothetical", "reported"]
    source_scope: Literal["narrator", "world_rule", "character_dialogue", "quoted_material", "unverified_report", "unknown"]
    certainty: Literal["certain", "probable", "possible", "unknown"]
    basis: Literal["explicit", "supported_inference", "unclear"]
    relation_quote: str = Field(min_length=1, max_length=600)


class NarrativeTimeExtraction(RelationExtractionBase):
    kind: Literal["narrative_time_relation"]
    left: TimePointRef
    right: TimePointRef
    relation: Literal["same_day", "different_day", "before", "after", "same_time", "unclear"]


class KnowledgeConstraintExtraction(RelationExtractionBase):
    kind: Literal["knowledge_constraint"]
    character: str = Field(min_length=1, max_length=64)
    fact: str = Field(min_length=1, max_length=160)
    time: str = Field(min_length=1, max_length=80)
    relation: Literal["unknown_before", "unknown_through", "known_by", "unclear"]
    knowledge_scope: Literal["whole_content", "channel_only", "unclear"]


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _claim(row: ParsedDirective) -> dict:
    keys = ("character", "fact", "time", "relation", "basis", "knowledge_scope",
            "left_time", "right_time", "document_role", "story_scope",
            "modality", "source_scope", "certainty")
    return {"kind": row.kind, "attrs": {key: row.attrs[key] for key in keys if key in row.attrs},
            "evidence": row.evidence.model_dump(mode="json")}


def _compatible_scope(left: str, right: str) -> bool:
    return left == right or "global" in {left, right}


def bind_relation(record, directive: ParsedDirective, sources: dict) -> ParsedDirective:
    """Only request-supplied sources can become endpoints of a relation."""
    from .semantic_quality import document_context_has_noncanonical_frame

    quote = record.relation_quote
    if directive.evidence.text.count(quote) != 1:
        raise ValueError("narrative_relation_source")
    attrs = {key: str(value) for key, value in record.model_dump(exclude={
        "kind", "source_line_start", "source_line_end", "relation_quote", "left", "right",
    }).items()}
    attrs.update(document_role=directive.attrs.get("document_role", "chapter"),
                 story_scope=directive.attrs.get("story_scope", "global"))
    points = []
    if record.kind == "narrative_time_relation":
        for name in ("left", "right"):
            ref = getattr(record, name)
            if ref.doc_ref not in sources:
                raise ValueError("narrative_relation_source")
            document, chunk = sources[ref.doc_ref]
            if not (chunk.global_line_start <= ref.line_start <= ref.line_end <= chunk.global_line_end):
                raise ValueError("narrative_relation_range")
            text = "\n".join(document.content.splitlines()[ref.line_start - 1:ref.line_end]).strip()
            if not text or ref.time not in text:
                raise ValueError("narrative_relation_time_source")
            if not _compatible_scope(attrs["story_scope"], document.scope or "global"):
                raise ValueError("narrative_relation_scope")
            if document_context_has_noncanonical_frame(document.content, ref.line_start, ref.line_end):
                raise ValueError("narrative_relation_frame")
            points.append({"document_id": document.id, "line_start": ref.line_start,
                           "line_end": ref.line_end, "time": ref.time,
                           "story_scope": document.scope or "global", "text_sha256": _digest(text)})
            attrs[name + "_time"] = ref.time
        if not _compatible_scope(points[0]["story_scope"], points[1]["story_scope"]):
            raise ValueError("narrative_relation_scope")
        if points[0] == points[1]:
            raise ValueError("narrative_relation_same_endpoint")
    else:
        # These fields identify the source's subject/content/time.  The model
        # may infer their relation, but cannot invent names or content labels.
        if any(getattr(record, key) not in directive.evidence.text
               for key in ("character", "fact", "time")):
            raise ValueError("knowledge_constraint_source")
    result = directive.model_copy(update={"attrs": attrs})
    proof = {"schema_version": RELATION_SCHEMA, "claim_sha256": _digest(_claim(result)),
             "points": points, "quote_sha256": _digest(quote)}
    proof["payload_sha256"] = _digest(proof)
    return result.model_copy(update={"narrative_relation": proof})


def valid_relation(row: ParsedDirective) -> bool:
    proof = row.narrative_relation
    return bool(row.kind in RELATION_KINDS and isinstance(proof, dict)
                and proof.get("schema_version") == RELATION_SCHEMA
                and proof.get("claim_sha256") == _digest(_claim(row))
                and proof.get("payload_sha256") == _digest({k: v for k, v in proof.items() if k != "payload_sha256"}))


def relation_is_certain(row: ParsedDirective) -> bool:
    from .semantic_quality import (_source_scope, _HYPOTHETICAL_MARKERS,
                                   _QUESTION_MARKERS, _UNCERTAIN_MARKERS)
    text = row.evidence.text
    return bool(valid_relation(row) and not row.noncanonical_frame
                and _source_scope(text, row).value in {"narrator", "world_rule"}
                and not any(pattern.search(text) for pattern in (
                    _HYPOTHETICAL_MARKERS, _QUESTION_MARKERS, _UNCERTAIN_MARKERS))
                and row.attrs.get("modality") == "asserted"
                and row.attrs.get("source_scope") in {"narrator", "world_rule"}
                and row.attrs.get("certainty") == "certain"
                and row.attrs.get("basis") in {"explicit", "supported_inference"}
                and row.attrs.get("relation") != "unclear"
                and row.attrs.get("knowledge_scope") != "unclear")


def _point_matches(point: dict, row: ParsedDirective) -> bool:
    span = row.evidence
    return bool(point["document_id"] == span.document_id
                and point["line_start"] == span.line_start and point["line_end"] == span.line_end
                and point["time"] == row.attrs.get("time")
                and point["text_sha256"] == _digest(span.text))


def _clock(value: str) -> int | None:
    match = re.fullmatch(r"([0-9]{1,2}):([0-9]{2})(?::([0-9]{2}))?", value.strip())
    if match is None:
        return None
    hour, minute, second = (int(v or 0) for v in match.groups())
    return hour * 3600 + minute * 60 + second if hour < 24 and minute < 60 and second < 60 else None


def _absolute(value: str) -> datetime | None:
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}[ T][0-9]{2}:[0-9]{2}(?::[0-9]{2})?", value.strip()):
        return None
    try:
        return datetime.fromisoformat(value.strip())
    except ValueError:
        return None


def compare_time_points(left: ParsedDirective, right: ParsedDirective,
                        relations: list[ParsedDirective]) -> tuple[int | None, list[EvidenceSpan]]:
    """-1/0/1 only from absolute dates or direct grounded relations, no closure."""
    absolute_left, absolute_right = (_absolute(row.attrs.get("time", "")) for row in (left, right))
    conclusions: set[int] = set()
    evidence = []
    unknown = False
    different_day = same_day = False
    if absolute_left is not None and absolute_right is not None:
        conclusions.add((absolute_left > absolute_right) - (absolute_left < absolute_right))
    for relation in relations:
        if not valid_relation(relation):
            continue
        points = relation.narrative_relation["points"]
        forward = _point_matches(points[0], left) and _point_matches(points[1], right)
        backward = _point_matches(points[1], left) and _point_matches(points[0], right)
        if not (forward or backward):
            continue
        if not relation_is_certain(relation):
            unknown = True
            continue
        kind = relation.attrs["relation"]
        order = {"before": -1, "after": 1, "same_time": 0}.get(kind)
        if kind == "same_day":
            same_day = True
            a, b = _clock(left.attrs.get("time", "")), _clock(right.attrs.get("time", ""))
            if a is None or b is None:
                unknown = True
                continue
            order = (a > b) - (a < b)
        elif kind == "different_day":
            different_day = True
            continue
        elif backward and order is not None:
            order = -order
        if order is not None:
            conclusions.add(order)
            evidence.append(relation.evidence)
    if unknown or (different_day and (same_day or 0 in conclusions)) or len(conclusions) != 1:
        return None, []
    return next(iter(conclusions)), evidence


def relation_issues(directives: list[ParsedDirective]) -> list:
    """Extend the existing comparisons, leaving their legacy absolute path intact."""
    from .domain import IssueCategory
    from .semantic_quality import eligible_for_deterministic_rules
    from .rules import (_issue, _scopes_compatible, _same_evidence, _evidence_shares_place,
                        _event_precise_timestamp, _location_within, _mobility_limit_applies)

    relations = [row for row in directives if row.kind == "narrative_time_relation"]
    trusted = [row for row in directives if eligible_for_deterministic_rules(row)]
    constraints = [row for row in trusted if row.kind == "knowledge_constraint"
                   and row.attrs.get("knowledge_scope") == "whole_content"]
    if not relations and not constraints:
        return []
    issues = []

    def evidence(*spans):
        return list({(s.document_id, s.line_start, s.line_end, s.text): s for s in spans}.values())

    events = [row for row in trusted if row.kind == "event"]
    permissions = [row for row in trusted if row.kind == "fact" and row.attrs.get("predicate") == "mobility_permission"]
    limits = [row for row in trusted if row.kind == "fact" and row.attrs.get("predicate") == "mobility_limit"]
    for i, first in enumerate(events):
        for second in events[i + 1:]:
            if (_same_evidence(first, second) or not _scopes_compatible(first, second)
                    or _evidence_shares_place(first, second)):
                continue
            if all(_event_precise_timestamp(row.attrs.get("time", "")) is not None for row in (first, second)):
                continue  # Existing absolute-date behavior and exception gates.
            left_place, right_place = first.attrs.get("location", ""), second.attrs.get("location", "")
            if not left_place or not right_place or left_place == right_place:
                continue
            people = set(first.attrs.get("participants", "").split(",")) & set(second.attrs.get("participants", "").split(","))
            order, sources = compare_time_points(first, second, relations)
            if order != 0:
                continue
            for person in sorted(people - {""}):
                # An active permission with a calendar interval cannot be
                # evaluated against an undated day. Do not assume it expired.
                possible_permission = False
                for permission in permissions:
                    a = permission.attrs
                    if (a.get("subject") != person or a.get("status") != "active"
                            or not _scopes_compatible(permission, first)
                            or not _scopes_compatible(permission, second)):
                        continue
                    route = (a.get("origin", ""), a.get("destination", ""))
                    forward = _location_within(left_place, route[0]) and _location_within(right_place, route[1])
                    backward = (a.get("bidirectional") == "true" and _location_within(left_place, route[1])
                                and _location_within(right_place, route[0]))
                    if forward or backward:
                        possible_permission = True
                if possible_permission:
                    continue
                relevant_limits = [r.evidence for r in limits if _scopes_compatible(r, first)
                                   and _scopes_compatible(r, second)
                                   and _mobility_limit_applies(r, person, left_place, right_place)]
                issues.append(_issue(
                    IssueCategory.location_collision, f"{person}在同一时间出现在不同地点",
                    f"原文时间关系将两次记录定位到同一时刻：{first.attrs.get('time')}，地点分别为“{left_place}”与“{right_place}”。未补造日历日期。",
                    evidence(first.evidence, second.evidence, *sources, *relevant_limits[:1]),
                    "核对两段时间与地点，以及是否存在适用的移动规则。",
                    participant=person, timestamp=first.attrs.get("time", ""),
                    temporal_basis="source_bound_relation_v1",
                ))
    lower_bounds = [row for row in constraints if row.attrs["relation"] in {"unknown_before", "unknown_through"}]
    positives = [row for row in constraints if row.attrs["relation"] == "known_by"]
    positives.extend(row for row in trusted if row.kind == "knows")
    for bound in lower_bounds:
        for positive in positives:
            if (_same_evidence(bound, positive) or not _scopes_compatible(bound, positive)
                    or (bound.attrs["character"], bound.attrs["fact"])
                    != (positive.attrs.get("character"), positive.attrs.get("fact"))):
                continue
            order, sources = compare_time_points(positive, bound, relations)
            if order is None or order > 0 or (order == 0 and bound.attrs["relation"] == "unknown_before"):
                continue
            inclusion = "之前" if bound.attrs["relation"] == "unknown_before" else "及该时刻"
            issues.append(_issue(
                IssueCategory.knowledge_without_acquisition,
                f"{bound.attrs['character']}的知情时间与明确约束冲突",
                f"来源限定其在 {bound.attrs['time']} {inclusion}尚不知“{bound.attrs['fact']}”，另一处却已表现出知情。该约束不代表在边界时刻实际获知。",
                evidence(bound.evidence, positive.evidence, *sources),
                "核对知情约束、实际获知或使用信息的时间与来源。",
                character=bound.attrs["character"], fact=bound.attrs["fact"],
                claim_time=positive.attrs.get("time", ""), knowledge_basis="constraint_v1",
                constraint_relation=bound.attrs["relation"],
            ))
    return issues


NARRATIVE_RELATION_INSTRUCTIONS = """
叙事时间与知情约束扩展（narrative-time-knowledge-v1），不判断矛盾：保留所有原记录格式，另允许以下两类记录，使用同一次请求内实际给出的原文，不能补造日期或实际获知。它们也必须逐条给source_line_start/end、modality、source_scope、certainty，另给basis=explicit|supported_inference|unclear及relation_quote（逐字、可唯一定位的完整关系依据，最多600字）。推断必须有完整上下文来源；不明则保留unclear/uncertain，不按文档顺序、相同时钟或同一主角猜同一天。
1. narrative_time_relation：kind,left,right,relation及上述公共字段。left/right各为{doc_ref:'d1',line_start:1,line_end:1,time:'原文时间词组'}，必须定位到本次请求同一项event/knows/claims_knows/knowledge_constraint的完整证据行范围和原时间。relation=same_day|different_day|before|after|same_time|unclear。same_day只证明同日，before/after只比较两个具体时间点；same_time须确实同刻，不能因两个12:00就推断。关系依据可在另一句中，record本身放进该依据所属doc_ref组；只有left/right引用允许跨doc_ref，原子事实仍不得跨文档拼写。为本次有明确同日/先后依据的相关时间点给直接关系，禁止无依据地补连全部时间；不必生成传递闭包。单文档请求的doc_ref固定为d1。
2. knowledge_constraint：kind,character,fact,time,relation,knowledge_scope及公共字段。character/fact/time复制本记录引用范围内原文词组。relation=unknown_before（t之前明确尚不知，不含t）、unknown_through（截至t仍不知，含t）、known_by（截至t已表现掌握/准确引用内容，不声称在t才获知）、unclear。knowledge_scope=whole_content|channel_only|unclear。单纯没看某文件、没听某人说是渠道限制；未拆封不等于无人可知，不能自动升为whole_content。完整上下文确实支持没有此前知情渠道或明确不知时可作有来源的合理推断，也不能凭空假设无依据的渠道否定明确约束。猜测、碰巧说中、梦境、引语内的自称不证明实际知情。known_by须核同一角色、同一信息内容实际被掌握或使用；不能把它改写成knows获知事件。约束内容、方向、时间或主体不明时选unclear，保留原来源。它们的modality是对关系本身的断言，明确‘尚不知’约束可为asserted，仍须区分叙述者、假设和转述。
旧knows仍只表示原文明示的实际获知；claims_knows仍表示真实发生的角色声称。某时读到或听到不自动证明首次获知，只有较晚获知记录或缺少较早获取记录都不能推断此前不知；下界须有明确不知或最早可知的完整来源依据。不要用新约束替换旧事实或弱化已知事件。没有可核关系可只返回原格式records，不新增顶层字段。
"""
