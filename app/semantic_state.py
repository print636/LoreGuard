"""Bounded source attestations for ordinary propositions and physical custody.

The model supplies meaning; the server binds it to the submitted source.  A
located quote is not an independent guarantee that the model's reading is true.
"""
from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .domain import ParsedDirective

SEMANTIC_STATE_SCHEMA = "semantic-state-v1"
SEMANTIC_STATE_REFS_KEY = "semantic_state_refs_v1"
CORE_FIELDS = {"fact": ("subject", "predicate", "value"),
               "item": ("item", "owner"), "uses": ("item", "user")}


class SemanticStateBindingError(ValueError):
    """Expected, record-local source binding failure after source attribution."""


class SourceRole(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quote: str = Field(min_length=1, max_length=600)
    relation: Literal["same", "equivalent", "different", "unclear"]


class SemanticRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    quote: str = Field(min_length=1, max_length=1600)
    roles: dict[str, SourceRole] = Field(min_length=2, max_length=3)
    relation: Literal["supported", "different", "unclear"]
    actuality: Literal["actual", "negated", "hypothetical", "reported", "unclear"]
    basis: Literal["explicit", "supported_inference", "unclear"]
    state_kind: Literal["observed_possession", "custody", "transfer", "retained_custody", "unclear"] | None = None
    retention_scope: Literal["ongoing", "bounded", "unclear"] | None = None
    transfer_from: SourceRole | None = None
    # A retained-custody constraint is bounded through this source time.  Its
    # start is the ordinary item.time; neither endpoint is invented.
    through_time: str | None = Field(default=None, min_length=1, max_length=80)


SEMANTIC_STATE_INSTRUCTIONS = """

【有源普通命题与物品状态 semantic-state-v1】
fact/item/uses 请附 semantic_relation；原字段仍保留，不为词面一致改写正确含义。
结构：{quote:原文完整命题片段, roles:{每个核心字段:{quote:原文对应片段,relation:same|equivalent|different|unclear}},
relation:supported|different|unclear, actuality:actual|negated|hypothetical|reported|unclear,
basis:explicit|supported_inference|unclear, state_kind:null, retention_scope:null, transfer_from:null, through_time:null}。
fact核心字段subject/predicate/value；item为item/owner；uses为item/user。quote必须逐字且在所报完整证据行中唯一；
允许完整上下文支持的同义、关系归纳、省略主体/对象承接，不要求归纳标签逐字出现或使用特定动作词。
roles的quote先在已选择的唯一命题quote内定位；同名在整行其他命题出现不影响此处定位。
若命题内仍重复，请给更具体的逐字原文片段区分；不计算偏移，不改正文，也不随便取第一个。
roles的quote可以共享或来自证据行的前文；跨命题承接须在完整证据行中唯一，必须确实证明当前角色/对象/属性；邻近别人的事实、另一物件不能借用。
完整证据行保留否定、条件、转述及限制。只有正文实际成立的关系才actual/supported；推测、计划、意图、否认不能当实际。
不同规范值请尽量统一同一属性predicate与同一物件名；不补不存在的事实、日期、交接。
item.owner仅表示物理持有/保管者，不是法律所有权。item必须给state_kind：
observed_possession=此刻观察到持有/拿取；custody=保管状态；transfer=正文证实实际交接并给transfer_from原持有人来源片段；
retained_custody=正文明确某期间仍由该人持有、未交给别人（不是仅没有写交接），retention_scope=ongoing表示有源持续约束、不要求写明终点；bounded表示有界区间才给through_time。
time起点照原文，不造日期或截止。范围不明用unclear；没有数字终点不等于范围不明。
retention_scope只用于retained_custody；through_time只用于其bounded范围；transfer_from只用于transfer，quote中的人物须保留完整姓名。
transfer_from非空时与roles的值使用完全相同的对象结构：{"quote":"原文中的出让人片段","relation":"same"}；
relation必须判断为same/equivalent/different/unclear之一。不得只输出姓名字符串；没有同指依据用unclear，不擅填same。
实际持有不证明合法交接，不得用观察到的新持有人抹去明确未交接的约束。
uses表示实际操作物品；不要仅因持有就虚构操作。其state_kind/retention_scope/transfer_from/through_time均null。
fact的state_kind/retention_scope/transfer_from/through_time也均null。
同钟点不等于同一天。需要比较物品状态的裸时间时，复用narrative_time_relation明确相关端点的直接同日/先后关系并给其原文依据；不能造日期。无可靠对应使用unclear并保留来源，不猜测。
"""


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def _claim(row: ParsedDirective, kind: str) -> dict:
    keys = (*CORE_FIELDS[kind], "time", "document_role", "story_scope",
            "state_kind", "retention_scope", "through_time", "transfer_from")
    return {"kind": kind, "attrs": {k: row.attrs[k] for k in keys if k in row.attrs},
            "evidence": row.evidence.model_dump(mode="json")}


def _ref(text: str, quote: str, *, proposition: dict | None = None) -> dict:
    if proposition is not None:
        start, end = proposition["start"], proposition["end"]
        local = text[start:end]
        if local.count(quote) == 1:
            start += local.index(quote)
            return {"start": start, "end": start + len(quote), "text_sha256": digest(quote)}
    if text.count(quote) != 1:
        raise SemanticStateBindingError("semantic_state_source_not_unique")
    start = text.index(quote)
    return {"start": start, "end": start + len(quote), "text_sha256": digest(quote)}


def bind_semantic_state(record, row: ParsedDirective) -> ParsedDirective:
    relation = record.semantic_relation
    if relation is None or record.kind not in CORE_FIELDS:
        raise SemanticStateBindingError("semantic_state_missing")
    if set(relation.roles) != set(CORE_FIELDS[record.kind]):
        raise SemanticStateBindingError("semantic_state_roles")
    if record.kind != "item" and any((relation.state_kind, relation.retention_scope, relation.transfer_from, relation.through_time)):
        raise SemanticStateBindingError("semantic_state_wrong_kind")
    if record.kind == "item" and relation.state_kind is None:
        raise SemanticStateBindingError("semantic_state_missing_state")
    if (relation.state_kind == "transfer") != (relation.transfer_from is not None):
        raise SemanticStateBindingError("semantic_state_transfer_source")
    if (relation.state_kind == "retained_custody") != (relation.retention_scope is not None):
        raise SemanticStateBindingError("semantic_state_retention_scope")
    if relation.retention_scope == "bounded" and not relation.through_time:
        raise SemanticStateBindingError("semantic_state_constraint_range")
    if relation.through_time is not None and relation.retention_scope != "bounded":
        raise SemanticStateBindingError("semantic_state_constraint_range")
    text = row.evidence.text
    refs = {"proposition": _ref(text, relation.quote)}
    refs.update({k: _ref(text, v.quote, proposition=refs["proposition"]) for k, v in relation.roles.items()})
    attrs = dict(row.attrs)
    if relation.state_kind:
        attrs["state_kind"] = relation.state_kind
    if relation.retention_scope:
        attrs["retention_scope"] = relation.retention_scope
    if relation.transfer_from:
        refs["transfer_from"] = _ref(text, relation.transfer_from.quote, proposition=refs["proposition"])
        attrs["transfer_from"] = relation.transfer_from.quote
    for key, value in (("time", attrs.get("time")), ("through_time", relation.through_time)):
        if value:
            refs[key] = _ref(text, value)
            attrs[key] = value
    result = row.model_copy(update={"attrs": attrs})
    proof = {"schema_version": SEMANTIC_STATE_SCHEMA, "kind": record.kind,
             "claim_sha256": digest(_claim(result, record.kind)), "refs": refs,
             "relation": relation.relation, "actuality": relation.actuality, "basis": relation.basis,
             "roles": {k: v.relation for k, v in relation.roles.items()},
             "transfer_relation": relation.transfer_from.relation if relation.transfer_from else None}
    proof["payload_sha256"] = digest(proof)
    return result.model_copy(update={"semantic_state": proof})


def valid_semantic_state(row: ParsedDirective) -> bool:
    proof = row.semantic_state
    if not isinstance(proof, dict) or proof.get("kind") not in CORE_FIELDS:
        return False
    if set(proof) != {"schema_version", "kind", "claim_sha256", "refs", "relation", "actuality",
                       "basis", "roles", "transfer_relation", "payload_sha256"}:
        return False
    kind = proof["kind"]
    if (proof.get("schema_version") != SEMANTIC_STATE_SCHEMA
            or proof.get("claim_sha256") != digest(_claim(row, kind))
            or proof.get("payload_sha256") != digest({k: v for k, v in proof.items() if k != "payload_sha256"})
            or row.kind not in {kind, "tentative_fact", "character_claim", "negative_statement", "open_question"}):
        return False
    required = {"proposition", *CORE_FIELDS[kind]}
    required.update(k for k in ("time", "through_time", "transfer_from") if row.attrs.get(k))
    if (not isinstance(proof.get("refs"), dict) or set(proof["refs"]) != required
            or not isinstance(proof.get("roles"), dict) or set(proof["roles"]) != set(CORE_FIELDS[kind])
            or any(v not in {"same", "equivalent", "different", "unclear"} for v in proof["roles"].values())
            or proof["relation"] not in {"supported", "different", "unclear"}
            or proof["actuality"] not in {"actual", "negated", "hypothetical", "reported", "unclear"}
            or proof["basis"] not in {"explicit", "supported_inference", "unclear"}
            or proof["transfer_relation"] not in {None, "same", "equivalent", "different", "unclear"}):
        return False
    for ref in proof["refs"].values():
        if not isinstance(ref, dict) or set(ref) != {"start", "end", "text_sha256"}:
            return False
        start, end = ref.get("start"), ref.get("end")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(row.evidence.text)
                or digest(row.evidence.text[start:end]) != ref.get("text_sha256")):
            return False
    # Recompute the same disambiguation as admission. A valid hash at another
    # equal-looking occurrence cannot replace the selected proposition's role.
    try:
        for name, ref in proof["refs"].items():
            quote = row.evidence.text[ref["start"]:ref["end"]]
            proposition = proof["refs"]["proposition"] if name in {*CORE_FIELDS[kind], "transfer_from"} else None
            if _ref(row.evidence.text, quote, proposition=proposition) != ref:
                return False
    except SemanticStateBindingError:
        return False
    return set(CORE_FIELDS[kind]).issubset(proof.get("refs", {}))


def supported_semantic_state(row: ParsedDirective) -> bool:
    if not valid_semantic_state(row):
        return False
    proof = row.semantic_state
    return bool(proof["relation"] == "supported" and proof["actuality"] == "actual"
                and proof["basis"] in {"explicit", "supported_inference"}
                and all(v in {"same", "equivalent"} for v in proof["roles"].values())
                and proof["transfer_relation"] in {None, "same", "equivalent"}
                and row.attrs.get("state_kind") != "unclear"
                and row.attrs.get("retention_scope") != "unclear")


def custody_comparisons(directives: list[ParsedDirective]) -> tuple[list, int]:
    """Compare physical states without treating an observed holder as transfer.

    No transitive ownership history is inferred. A single sourced handover may
    supersede a persistent structured baseline; an explicit bounded retention
    remains evidence even when another row says someone else possesses it.
    """
    from .domain import ConsistencyIssue, IssueCategory, Severity
    from .semantic_quality import eligible_for_deterministic_rules
    from .narrative_relations import compare_time_points
    from .rules import _scopes_compatible

    rows = [d for d in directives if d.kind in {"item", "uses"} and eligible_for_deterministic_rules(d)]
    relations = [d for d in directives if d.kind == "narrative_time_relation"]
    observations = [d for d in rows if d.kind == "uses" or d.attrs.get("state_kind") in {"observed_possession", "transfer"}]
    states = [d for d in rows if d.kind == "item" and d.attrs.get("state_kind") != "observed_possession"]
    issues, unknown, seen = [], 0, set()

    def sourced_handover_ends(state, observation, *, require_start: bool) -> bool:
        for transfer in states:
            if not (supported_semantic_state(transfer) and transfer.attrs.get("state_kind") == "transfer"
                    and transfer.attrs.get("item") == state.attrs.get("item")
                    and _scopes_compatible(state, transfer) and _scopes_compatible(transfer, observation)
                    and transfer.attrs.get("transfer_from") == state.attrs.get("owner")):
                continue
            if require_start:
                begins_after, _ = compare_time_points(state, transfer, relations)
                if begins_after not in {-1, 0}:
                    continue
            order, _ = compare_time_points(transfer, observation, relations)
            if transfer is observation or order in {-1, 0}:
                return True
        return False

    for observation in observations:
        who = observation.attrs.get("user", observation.attrs.get("owner", ""))
        item = observation.attrs.get("item")
        for state in states:
            if (state is observation or state.attrs.get("item") != item or not _scopes_compatible(state, observation)
                    or state.attrs.get("owner") == who):
                continue
            evidence = [state.evidence, observation.evidence]
            persistent = state.attrs.get("input_form") == "directive" and not state.attrs.get("time")
            if persistent:
                # Only a directly grounded transfer out of this exact holder
                # can update this persistent baseline. Unknown order never can.
                applicable = not sourced_handover_ends(state, observation, require_start=False)
            elif supported_semantic_state(state) and state.attrs.get("state_kind") == "retained_custody":
                if not state.attrs.get("time"):
                    unknown += 1
                    continue
                start_order, start_evidence = compare_time_points(state, observation, relations)
                end_order, end_evidence = 0, []
                if state.attrs.get("retention_scope") == "bounded":
                    end = state.model_copy(update={"attrs": {**state.attrs, "time": state.attrs["through_time"]}})
                    end_order, end_evidence = compare_time_points(observation, end, relations)
                if start_order is None or end_order is None:
                    unknown += 1
                    continue
                applicable = start_order <= 0 and end_order <= 0
                if state.attrs.get("retention_scope") == "ongoing" and sourced_handover_ends(
                        state, observation, require_start=True):
                    applicable = False
                evidence += start_evidence + end_evidence
            elif (supported_semantic_state(state) and state.attrs.get("state_kind") == "custody"
                  and sourced_handover_ends(state, observation, require_start=True)):
                # A located ordinary custody state also has a known successor
                # when an actual handover establishes both ordering boundaries.
                applicable = False
            else:
                order, time_evidence = compare_time_points(state, observation, relations)
                # Plain custody at an earlier time is not a no-transfer claim.
                applicable = order == 0
                evidence += time_evidence
                if order is None or order == -1:
                    unknown += 1
            if not applicable:
                continue
            key = (item, who, state.attrs.get("owner"), observation.evidence.document_id,
                   observation.evidence.line_start, observation.evidence.line_end)
            if key in seen:
                continue
            seen.add(key)
            unique = {digest(e.model_dump(mode="json")): e for e in evidence}
            issues.append(ConsistencyIssue(category=IssueCategory.item_ownership, severity=Severity.high,
                confidence=0.96, title=f"{item}的持有或保管状态存在冲突",
                explanation=f"来源约束表明{state.attrs.get('owner')}在适用时段持有或保管该物品，另一处却显示{who}持有或操作它。",
                evidence=list(unique.values()), suggestion="核对两处物品是否相同、交接是否发生及其时间；此处比较物理持有，不判断法律所有权。",
                metadata={"item": item, "expected_owner": state.attrs.get("owner"), "actual_user": who}))
    return issues, unknown
