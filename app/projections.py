from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import re
from typing import Any, Iterable

from .domain import (
    EvidenceSpan,
    GraphEdge,
    GraphNode,
    GraphResponse,
    TimelineEntry,
    TimelineGroup,
    TimelineResponse,
)


GRAPH_NODE_LIMIT = 500
GRAPH_EDGE_LIMIT = 1000
_EXACT_TIME = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})(?::(\d{2}))?$")
_DATE_TIME = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def record_sort_key(row: Any) -> tuple:
    evidence = row.evidence or {}
    return (
        str(evidence.get("document_name", "")).casefold(),
        int(evidence.get("line_start", 0) or 0),
        int(evidence.get("line_end", evidence.get("line_start", 0)) or 0),
        str(row.kind),
        str(row.id),
    )


def _evidence(payload: dict) -> EvidenceSpan:
    line_start = int(payload.get("line_start", 0) or 0)
    return EvidenceSpan(
        document_id=str(payload.get("document_id", "")),
        document_name=str(payload.get("document_name", "")),
        line_start=line_start,
        line_end=int(payload.get("line_end", line_start) or line_start),
        text=str(payload.get("text", "")),
    )


def _overlaps(left: dict, right: dict) -> bool:
    left_id, right_id = str(left.get("document_id", "")), str(right.get("document_id", ""))
    if left_id and right_id:
        if left_id != right_id:
            return False
    elif str(left.get("document_name", "")) != str(right.get("document_name", "")):
        return False
    left_start = int(left.get("line_start", 0) or 0)
    left_end = int(left.get("line_end", left_start) or left_start)
    right_start = int(right.get("line_start", 0) or 0)
    right_end = int(right.get("line_end", right_start) or right_start)
    return max(left_start, right_start) <= min(left_end, right_end)


def _issue_ids(evidence: dict, issues: Iterable[Any]) -> list[str]:
    matches = {
        str(issue.id)
        for issue in issues
        if any(_overlaps(evidence, span) for span in (issue.evidence or []))
    }
    return sorted(matches)


def _stable_id(kind: str, *parts: str) -> str:
    material = "\0".join((kind, *(str(part).strip() for part in parts)))
    return f"{kind}:{sha256(material.encode('utf-8')).hexdigest()[:16]}"


def _timestamp(attrs: dict[str, str]) -> str | None:
    value = str(attrs.get("time") or attrs.get("timestamp") or "").strip()
    return value or None


def _node(kind: str, label: str, *identity: str, **metadata: Any) -> dict:
    return {
        "id": _stable_id(kind, *(identity or (label,))),
        "label": label,
        "type": kind,
        "issue_ids": set(),
        "metadata": metadata,
    }


def _relations(row: Any, issue_ids: list[str]) -> tuple[list[dict], list[dict]]:
    attrs = {str(key): str(value) for key, value in (row.attrs or {}).items()}
    kind = str(row.kind)
    timestamp = _timestamp(attrs)
    evidence = _evidence(row.evidence or {})
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    def add_node(node: dict) -> str:
        nodes.setdefault(node["id"], node)
        return node["id"]

    def add_edge(source: str, target: str, label: str, *, suffix: str = "") -> None:
        edges.append({
            "id": _stable_id("edge", str(row.id), kind, source, target, suffix),
            "source": source,
            "target": target,
            "type": kind,
            "label": label,
            "record_id": str(row.id),
            "timestamp": timestamp,
            "evidence": evidence,
            "issue_ids": issue_ids,
            "metadata": attrs,
        })

    if kind == "entity":
        name = attrs.get("name", "").strip()
        if name:
            add_node(_node("entity", name, name, entity_kind=attrs.get("kind", "")))
    elif kind == "fact":
        subject, predicate, value = (attrs.get(key, "").strip() for key in ("subject", "predicate", "value"))
        if subject and predicate and value:
            source = add_node(_node("entity", subject, subject))
            state = add_node(_node("state", f"{predicate}：{value}", subject, predicate, value, predicate=predicate, value=value))
            add_edge(source, state, predicate)
    elif kind == "event":
        location = attrs.get("location", "").strip()
        participants = [part.strip() for part in re.split(r"[,，]", attrs.get("participants", "")) if part.strip()]
        if location:
            target = add_node(_node("location", location, location))
            for participant in participants:
                source = add_node(_node("entity", participant, participant))
                add_edge(source, target, "出现于", suffix=participant)
    elif kind in {"knows", "claims_knows"}:
        character, fact = attrs.get("character", "").strip(), attrs.get("fact", "").strip()
        if character and fact:
            source = add_node(_node("entity", character, character))
            target = add_node(_node("knowledge", fact, fact))
            add_edge(source, target, "获知" if kind == "knows" else "声称知道")
    elif kind in {"item", "uses"}:
        item = attrs.get("item", "").strip()
        actor_key = "owner" if kind == "item" else "user"
        actor = attrs.get(actor_key, "").strip()
        if item and actor:
            source = add_node(_node("entity", actor, actor))
            target = add_node(_node("item", item, item))
            add_edge(source, target, "持有" if kind == "item" else "使用")
    elif kind in {"world_rule", "world_assert"}:
        key, value = attrs.get("key", "").strip(), attrs.get("value", "").strip()
        if key and value:
            target = add_node(_node("rule_value", value, key, value, rule_key=key))
            actor = attrs.get("actor", "").strip()
            if kind == "world_assert" and actor:
                source = add_node(_node("entity", actor, actor))
                add_edge(source, target, f"剧情断言：{key}")
            else:
                source = add_node(_node("rule", key, key))
                add_edge(source, target, "权威规则" if kind == "world_rule" else "剧情断言")

    return list(nodes.values()), edges


def project_graph(
    run_id: str,
    records: Iterable[Any],
    issues: Iterable[Any],
    *,
    node_limit: int = GRAPH_NODE_LIMIT,
    edge_limit: int = GRAPH_EDGE_LIMIT,
) -> GraphResponse:
    issue_rows = list(issues)
    node_map: dict[str, dict] = {}
    output_edges: list[GraphEdge] = []
    truncated = False
    skipped_invalid = 0

    for row in sorted(records, key=record_sort_key):
        linked_issues = _issue_ids(row.evidence or {}, issue_rows)
        nodes, edges = _relations(row, linked_issues)
        if not nodes and not edges:
            skipped_invalid += 1
            continue
        new_node_ids = {node["id"] for node in nodes if node["id"] not in node_map}
        if len(node_map) + len(new_node_ids) > node_limit:
            truncated = True
            continue
        if edges and len(output_edges) + len(edges) > edge_limit:
            truncated = True
            continue
        for node in nodes:
            existing = node_map.setdefault(node["id"], node)
            existing["issue_ids"].update(linked_issues)
        output_edges.extend(GraphEdge(**edge) for edge in edges)

    warnings: list[str] = []
    if truncated:
        warnings.append(f"关系图已按 {node_limit} 个节点 / {edge_limit} 条边的上限截断；请缩小分析范围。")
    if skipped_invalid:
        warnings.append(f"{skipped_invalid} 条记录缺少构图所需字段，未进入关系图。")
    graph_nodes = [
        GraphNode(
            id=node["id"],
            label=node["label"],
            type=node["type"],
            issue_ids=sorted(node["issue_ids"]),
            metadata=node["metadata"],
        )
        for node in sorted(node_map.values(), key=lambda value: (value["type"], value["label"], value["id"]))
    ]
    output_edges.sort(key=lambda edge: edge.id)
    return GraphResponse(
        run_id=run_id,
        nodes=graph_nodes,
        edges=output_edges,
        truncated=truncated,
        warnings=warnings,
    )


def _time_classification(value: str | None) -> tuple[str, str | None, str | None]:
    if not value:
        return "unknown", None, None
    exact = _EXACT_TIME.fullmatch(value)
    if exact:
        seconds = exact.group(3) or "00"
        normalized = f"{exact.group(1)} {exact.group(2)}"
        if exact.group(3):
            normalized += f":{exact.group(3)}"
        return "exact", f"{exact.group(1)}T{exact.group(2)}:{seconds}", normalized
    if _DATE_TIME.fullmatch(value):
        return "date", f"{value}T00:00:00", value
    return "relative", None, value


def _timeline_title(kind: str, attrs: dict[str, str]) -> str:
    if kind == "fact":
        return f"{attrs.get('subject', '未知对象')} · {attrs.get('predicate', '状态')} = {attrs.get('value', '未知')}"
    if kind == "event":
        return f"{attrs.get('participants', '未知角色')} 出现在 {attrs.get('location', '未知地点')}"
    if kind == "knows":
        return f"{attrs.get('character', '未知角色')} 获知 {attrs.get('fact', '未知信息')}"
    if kind == "claims_knows":
        return f"{attrs.get('character', '未知角色')} 声称知道 {attrs.get('fact', '未知信息')}"
    if kind == "item":
        return f"{attrs.get('owner', '未知角色')} 持有 {attrs.get('item', '未知物品')}"
    if kind == "uses":
        return f"{attrs.get('user', '未知角色')} 使用 {attrs.get('item', '未知物品')}"
    if kind == "world_rule":
        return f"规则 {attrs.get('key', '未知')} = {attrs.get('value', '未知')}"
    if kind == "world_assert":
        return f"剧情断言 {attrs.get('key', '未知')} = {attrs.get('value', '未知')}"
    return f"实体 {attrs.get('name', '未知')}"


def project_timeline(run_id: str, records: Iterable[Any], issues: Iterable[Any]) -> TimelineResponse:
    issue_rows = list(issues)
    grouped: dict[tuple[str, str, str], list[TimelineEntry]] = defaultdict(list)
    unscheduled: list[TimelineEntry] = []

    for row in sorted(records, key=record_sort_key):
        if str(row.kind) == "entity":
            continue
        attrs = {str(key): str(value) for key, value in (row.attrs or {}).items()}
        timestamp = _timestamp(attrs)
        precision, sort_key, display = _time_classification(timestamp)
        entry = TimelineEntry(
            id=_stable_id("timeline", str(row.id)),
            record_id=str(row.id),
            kind=str(row.kind),
            title=_timeline_title(str(row.kind), attrs),
            timestamp=timestamp,
            precision=precision,
            evidence=_evidence(row.evidence or {}),
            issue_ids=_issue_ids(row.evidence or {}, issue_rows),
            attrs=attrs,
        )
        if sort_key and display:
            grouped[(sort_key, display, precision)].append(entry)
        else:
            unscheduled.append(entry)

    groups = [
        TimelineGroup(timestamp=display, sort_key=sort_key, precision=precision, entries=entries)
        for (sort_key, display, precision), entries in sorted(grouped.items(), key=lambda item: item[0][0])
    ]
    return TimelineResponse(
        run_id=run_id,
        groups=groups,
        unscheduled=unscheduled,
        warnings=["相对时间和无时间记录保留原文顺序，未被强行推断先后。"] if unscheduled else [],
    )
