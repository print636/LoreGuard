"""Finite, source-bound temporal membership; no calendar or graph completion."""
from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from .domain import ParsedDirective
from .narrative_relations import RelationExtractionBase, _digest, _clock, _compatible_scope

TIME_SCOPE_SCHEMA = "narrative-time-scope-v1"
TIME_SCOPE_REFS_KEY = "narrative_time_scope_refs_v1"
MAX_TIME_SCOPE_MEMBERS = 16


class TimeScopeBindingError(ValueError):
    """Expected, record-local source/shape failure."""


class TimeScopeMember(BaseModel):
    model_config = ConfigDict(extra="forbid")
    doc_ref: str = Field(pattern=r"^d[1-9][0-9]*$")
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    event_quote: str = Field(min_length=1, max_length=1200)
    time: str = Field(min_length=1, max_length=80)
    clock: str | None = Field(default=None, max_length=8)
    clock_relation: Literal["same", "equivalent", "unclear"]


class NarrativeTimeScopeExtraction(RelationExtractionBase):
    kind: Literal["narrative_time_scope"]
    relation: Literal["same_day", "ordered", "same_day_ordered", "unclear"]
    members: list[TimeScopeMember] = Field(min_length=2, max_length=MAX_TIME_SCOPE_MEMBERS)
    omitted_count: int = Field(ge=0, le=100000)


NARRATIVE_TIME_SCOPE_INSTRUCTIONS = """

【有源有限叙事时间范围 narrative-time-scope-v1】
保留既有记录，另可输出kind=narrative_time_scope，同一次请求内由正文判断有共同同日或顺序依据的一组具体事件。
完整结构：{"kind":"narrative_time_scope","source_line_start":1,"source_line_end":2,
"modality":"asserted","source_scope":"narrator","certainty":"certain","basis":"supported_inference",
"relation_quote":"逐字且唯一的完整时间范围依据","relation":"same_day",
"members":[{"doc_ref":"d1","line_start":1,"line_end":1,"event_quote":"逐字事件片段",
"time":"原时间短语","clock":"09:30","clock_relation":"equivalent"},
{"doc_ref":"d2","line_start":2,"line_end":2,"event_quote":"逐字事件片段",
"time":"原时间短语","clock":null,"clock_relation":"unclear"}],"omitted_count":0}。
所有非空成员均为该对象结构，不能字符串代替；不计算字符偏移。单文档doc_ref为d1。
relation=same_day|ordered|same_day_ordered|unclear；basis=explicit|supported_inference|unclear。
same_day只说明列明事件同一天，不补具体日期；ordered表示members按事件实际发生的从早到晚严格先后，绝不是文本排列顺序；same_day_ordered同时给出二者。
成员必须覆盖本次确有依据要比较的fact/event/item/uses/knows/claims_knows/knowledge_constraint的时间身份，
包括同物品的保管、实际交接、持有/使用等记录；一对事件的局部关系不能自动覆盖其他事件。
每个成员line范围须与相应原子记录完整证据行相同，event_quote在该范围唯一；time保留原文如‘当天九点半’、‘随后’，
与原子记录的原时间短语对应，不把原time改成虚构日期。clock另表示该短语明确/有依据的标准HH:MM[:SS]，不必原文含数字，
clock_relation=same|equivalent|unclear明确钟点同义；不知钟点就clock=null/unclear，可仅依正文先后ordered。
角色/物品名字重复时选更具体event_quote；允许同一行动的item与uses共享一个成员，不能因此合并独立行动。
同日/连续叙述可根据完整上下文合理推断，但必须有源；文档顺序、相同时钟、同人物都不能单独证明同日。
范围只作用列明成员；不得把局部同日断言扩散到整章，不把回忆/假设/转述成员混入当前真实事件。
只给直接有依据的一组关系，不建立跨范围传递闭包；遇矛盾或不确定用unclear/uncertain，不选有利的一条。
每组2至16成员，超出分成各自确有依据的有限组或omitted_count如实记录遗漏，不声称全部已审。不额外请求模型。
"""


def _ref(text: str, quote: str, within: dict | None = None) -> dict:
    start, end = (within["start"], within["end"]) if within else (0, len(text))
    local = text[start:end]
    if local.count(quote) != 1:
        raise TimeScopeBindingError("time_scope_source_not_unique")
    start += local.index(quote)
    return {"start": start, "end": start + len(quote), "text_sha256": _digest(quote)}


def _claim(row: ParsedDirective) -> dict:
    return {"kind": row.kind, "attrs": row.attrs, "evidence": row.evidence.model_dump(mode="json")}


def bind_time_scope(record: NarrativeTimeScopeExtraction, row: ParsedDirective, sources: dict) -> ParsedDirective:
    from .semantic_quality import document_context_has_noncanonical_frame
    source_ref = _ref(row.evidence.text, record.relation_quote)
    members = []
    total_chars = 0
    for raw in record.members:
        if raw.doc_ref not in sources:
            raise TimeScopeBindingError("time_scope_missing_source")
        document, chunk = sources[raw.doc_ref]
        if not chunk.global_line_start <= raw.line_start <= raw.line_end <= chunk.global_line_end:
            raise TimeScopeBindingError("time_scope_member_range")
        text = "\n".join(document.content.splitlines()[raw.line_start - 1:raw.line_end]).strip()
        total_chars += len(text)
        if not text or total_chars > 16000 or raw.line_end - raw.line_start > 16:
            raise TimeScopeBindingError("time_scope_source_limit")
        if (not _compatible_scope(row.attrs.get("story_scope", "global"), document.scope or "global")
                or document_context_has_noncanonical_frame(document.content, raw.line_start, raw.line_end)):
            raise TimeScopeBindingError("time_scope_member_context")
        event_ref = _ref(text, raw.event_quote)
        time_ref = _ref(text, raw.time, event_ref)
        if raw.clock is not None and _clock(raw.clock) is None:
            raise TimeScopeBindingError("time_scope_invalid_clock")
        members.append({"document_id": document.id, "line_start": raw.line_start, "line_end": raw.line_end,
                        "story_scope": document.scope or "global", "text_sha256": _digest(text),
                        "event_ref": event_ref, "time_ref": time_ref,
                        "clock": raw.clock, "clock_relation": raw.clock_relation})
    if any(not _compatible_scope(a["story_scope"], b["story_scope"]) for a in members for b in members):
        raise TimeScopeBindingError("time_scope_member_context")
    if len({_digest({k:v for k,v in m.items() if k not in {"clock", "clock_relation"}}) for m in members}) != len(members):
        raise TimeScopeBindingError("time_scope_duplicate_member")
    attrs = {key: str(getattr(record, key)) for key in ("modality", "source_scope", "certainty", "basis", "relation", "omitted_count")}
    attrs.update(member_count=str(len(members)), document_role=row.attrs.get("document_role", "chapter"),
                 story_scope=row.attrs.get("story_scope", "global"))
    result = row.model_copy(update={"attrs": attrs})
    proof = {"schema_version": TIME_SCOPE_SCHEMA, "claim_sha256": _digest(_claim(result)),
             "source_ref": source_ref, "members": members}
    proof["payload_sha256"] = _digest(proof)
    return result.model_copy(update={"narrative_time_scope": proof})


def _valid_ref(ref, text: str) -> bool:
    return bool(isinstance(ref, dict) and set(ref) == {"start", "end", "text_sha256"}
                and type(ref["start"]) is int and type(ref["end"]) is int
                and 0 <= ref["start"] < ref["end"] <= len(text)
                and _digest(text[ref["start"]:ref["end"]]) == ref["text_sha256"])


def valid_time_scope(row: ParsedDirective) -> bool:
    p = row.narrative_time_scope
    if not (row.kind == "narrative_time_scope" and isinstance(p, dict)
            and set(p) == {"schema_version", "claim_sha256", "source_ref", "members", "payload_sha256"}
            and p["schema_version"] == TIME_SCOPE_SCHEMA
            and p["claim_sha256"] == _digest(_claim(row))
            and p["payload_sha256"] == _digest({k:v for k,v in p.items() if k != "payload_sha256"})
            and _valid_ref(p["source_ref"], row.evidence.text)
            and isinstance(p["members"], list) and 2 <= len(p["members"]) <= MAX_TIME_SCOPE_MEMBERS):
        return False
    return all(isinstance(m, dict) and set(m) == {"document_id", "line_start", "line_end", "story_scope",
        "text_sha256", "event_ref", "time_ref", "clock", "clock_relation"}
        and type(m["line_start"]) is int and type(m["line_end"]) is int and 1 <= m["line_start"] <= m["line_end"]
        and m["clock_relation"] in {"same", "equivalent", "unclear"}
        and (m["clock"] is None or _clock(m["clock"]) is not None) for m in p["members"])


def validate_time_scope_sources(row: ParsedDirective, sources: dict[str, str]) -> bool:
    """Persistence recheck against this run's frozen document_id -> full text."""
    if not valid_time_scope(row):
        return False
    from .semantic_quality import document_context_has_noncanonical_frame
    span = row.evidence
    content = sources.get(span.document_id)
    if content is None or "\n".join(content.splitlines()[span.line_start - 1:span.line_end]).strip() != span.text:
        return False
    source_ref = row.narrative_time_scope["source_ref"]
    try:
        if _ref(span.text, span.text[source_ref["start"]:source_ref["end"]]) != source_ref:
            return False
    except TimeScopeBindingError:
        return False
    total_chars = 0
    for member in row.narrative_time_scope["members"]:
        content = sources.get(member["document_id"])
        if content is None:
            return False
        text = "\n".join(content.splitlines()[member["line_start"] - 1:member["line_end"]]).strip()
        total_chars += len(text)
        if (_digest(text) != member["text_sha256"] or not text or total_chars > 16000
                or member["line_end"] > len(content.splitlines())
                or member["line_end"] - member["line_start"] > 16
                or not _compatible_scope(member["story_scope"], row.attrs.get("story_scope", "global"))
                or document_context_has_noncanonical_frame(content, member["line_start"], member["line_end"])):
            return False
        event, time = member["event_ref"], member["time_ref"]
        if not (_valid_ref(event, text) and _valid_ref(time, text)
                and event["start"] <= time["start"] < time["end"] <= event["end"]):
            return False
        try:
            if _ref(text, text[event["start"]:event["end"]]) != event or _ref(text, text[time["start"]:time["end"]], event) != time:
                return False
        except TimeScopeBindingError:
            return False
    members = row.narrative_time_scope["members"]
    if (any(not _compatible_scope(a["story_scope"], b["story_scope"]) for a in members for b in members)
            or len({_digest({k:v for k,v in m.items() if k not in {"clock", "clock_relation"}}) for m in members}) != len(members)):
        return False
    return True


def time_scope_is_certain(row: ParsedDirective) -> bool:
    from .semantic_quality import _source_scope, _HYPOTHETICAL_MARKERS, _UNCERTAIN_MARKERS
    return bool(valid_time_scope(row) and not row.noncanonical_frame
                and row.attrs.get("modality") == "asserted" and row.attrs.get("certainty") == "certain"
                and row.attrs.get("source_scope") in {"narrator", "world_rule"}
                and _source_scope(row.evidence.text, row).value in {"narrator", "world_rule"}
                and not any(p.search(row.evidence.text) for p in (_HYPOTHETICAL_MARKERS, _UNCERTAIN_MARKERS))
                and row.attrs.get("basis") in {"explicit", "supported_inference"}
                and row.attrs.get("relation") != "unclear" and row.attrs.get("omitted_count") == "0")


def member_matches(member: dict, row: ParsedDirective) -> bool:
    e = row.evidence
    if (row.kind not in {"fact", "event", "item", "uses", "knows", "claims_knows", "knowledge_constraint"}
            or (member["document_id"], member["line_start"], member["line_end"], member["text_sha256"])
            != (e.document_id, e.line_start, e.line_end, _digest(e.text))
            or not _compatible_scope(member["story_scope"], row.attrs.get("story_scope", "global"))):
        return False
    raw_time = row.attrs.get("time", "")
    if not raw_time:
        return False
    event = member["event_ref"]
    if row.semantic_state is not None:
        from .semantic_state import supported_semantic_state
        if not supported_semantic_state(row):
            return False
        action = row.semantic_state["refs"]["proposition"]
        if not (event["start"] <= action["start"] < action["end"] <= event["end"]
                or action["start"] <= event["start"] < event["end"] <= action["end"]):
            return False
    else:
        # Existing event/knowledge records lack action offsets. Their source
        # identities must be present in this particular member, not elsewhere
        # on the same physical line. Untimed/legacy operations are never upgraded.
        event_text = e.text[event["start"]:event["end"]]
        if row.kind == "event":
            anchors = [row.attrs.get("location", ""), *row.attrs.get("participants", "").split(",")]
        elif row.kind in {"knows", "claims_knows", "knowledge_constraint"}:
            anchors = [row.attrs.get("character", ""), row.attrs.get("fact", "")]
        else:
            return False
        if not anchors or any(not a or a not in event_text for a in anchors):
            return False
    try:
        ref = _ref(e.text, raw_time, member["event_ref"])
    except TimeScopeBindingError:
        return False
    other = member["time_ref"]
    # Different spellings such as 当天09:30 and 09:30 may locate the same
    # original time occurrence. Equal strings elsewhere are not an identity.
    return max(ref["start"], other["start"]) < min(ref["end"], other["end"])


def scope_comparison(scope: ParsedDirective, left: ParsedDirective, right: ParsedDirective) -> tuple[bool, set[int]]:
    """Return (applies, possible conclusions); an empty set means unresolved."""
    if not valid_time_scope(scope):
        return False, set()
    members = scope.narrative_time_scope["members"]
    a = [i for i,m in enumerate(members) if member_matches(m, left)]
    b = [i for i,m in enumerate(members) if member_matches(m, right)]
    if not a or not b:
        return False, set()
    if len(a) != 1 or len(b) != 1 or not time_scope_is_certain(scope):
        return True, set()
    i, j = a[0], b[0]
    result = set()
    if i == j:
        return True, {0}
    relation = scope.attrs["relation"]
    if relation in {"ordered", "same_day_ordered"}:
        result.add((i > j) - (i < j))
    if relation in {"same_day", "same_day_ordered"}:
        x, y = members[i], members[j]
        if all(m["clock"] is not None and m["clock_relation"] != "unclear" for m in (x, y)):
            left_clock, right_clock = _clock(x["clock"]), _clock(y["clock"])
            result.add((left_clock > right_clock) - (left_clock < right_clock))
        elif relation == "same_day":
            return True, set()
    return True, result
