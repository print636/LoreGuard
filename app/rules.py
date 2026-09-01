from __future__ import annotations

from collections import defaultdict
import re

from .domain import ConsistencyIssue, IssueCategory, ParsedDirective, Severity


def _issue(category, title, explanation, evidence, suggestion, severity=Severity.high, **metadata):
    return ConsistencyIssue(
        category=category,
        severity=severity,
        confidence=0.96,
        title=title,
        explanation=explanation,
        evidence=evidence,
        suggestion=suggestion,
        metadata=metadata,
    )


def _state_applies_at(state_time: str, action_time: str) -> bool:
    """An untimed ownership row is canonical; a timed row needs an ordered action."""
    if not state_time:
        return True
    if not action_time:
        return False
    return _time_key(state_time) <= _time_key(action_time)


def _time_key(value: str) -> str:
    return value.replace(" ", "").replace("T", "")


def _precise_timestamp(value: str) -> bool:
    """Only minute/second timestamps can prove simultaneous presence."""
    return bool(
        re.fullmatch(
            r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?",
            value.strip(),
        )
    )


def _explicit_state_active(state: ParsedDirective, action_time: str) -> bool:
    attrs = state.attrs
    if attrs.get("status") != "active":
        return False
    valid_from = attrs.get("valid_from", "")
    valid_until = attrs.get("valid_until", "")
    if action_time:
        if valid_from and _time_key(action_time) < _time_key(valid_from):
            return False
        if valid_until and _time_key(action_time) > _time_key(valid_until):
            return False
        return bool(attrs.get("current") == "true" or valid_from or valid_until)
    return attrs.get("current") == "true" and not valid_from


def _mobility_permission_applies(
    state: ParsedDirective,
    participant: str,
    timestamp: str,
    origin: str,
    destination: str,
) -> bool:
    attrs = state.attrs
    if attrs.get("subject") != participant or not _explicit_state_active(state, timestamp):
        return False
    route = (attrs.get("origin", ""), attrs.get("destination", ""))
    forward = _location_within(origin, route[0]) and _location_within(destination, route[1])
    if forward:
        return True
    reverse = _location_within(origin, route[1]) and _location_within(destination, route[0])
    return attrs.get("bidirectional") == "true" and reverse


def _location_within(location: str, permitted_endpoint: str) -> bool:
    return bool(
        location
        and permitted_endpoint
        and (location == permitted_endpoint or location.startswith(permitted_endpoint))
    )


def _place_tokens(text: str) -> set[str]:
    """Extract short explicit Chinese place spans for hierarchy checks."""
    suffixes = set("港塔室城岛站厅楼舱院宫门仓库")
    compact = re.sub(r"[^\u4e00-\u9fff]", " ", text)
    tokens: set[str] = set()
    for segment in compact.split():
        for end, char in enumerate(segment, start=1):
            if char not in suffixes:
                continue
            for length in range(2, min(6, end) + 1):
                tokens.add(segment[end - length:end])
    return tokens


def _evidence_shares_place(first: ParsedDirective, second: ParsedDirective) -> bool:
    return bool(_place_tokens(first.evidence.text) & _place_tokens(second.evidence.text))


def _rule_exception_applies(
    state: ParsedDirective, actor: str, key: str, action_time: str
) -> bool:
    attrs = state.attrs
    return bool(
        actor
        and attrs.get("subject") == actor
        and attrs.get("key") == key
        and _explicit_state_active(state, action_time)
    )


def _fact_label(predicate: str) -> str:
    if predicate.startswith("body_state:"):
        body = predicate.split(":", 1)[1]
        if body.startswith("左"):
            body = f"左侧{body[1:]}"
        elif body.startswith("右"):
            body = f"右侧{body[1:]}"
        return f"{body}状态"
    return predicate


def _world_rule_title(key: str) -> str:
    parts = key.split(":", 2)
    if len(parts) == 3 and parts[0] == "scope_action":
        return f"{parts[1]}中的{parts[2]}规则被违反"
    return f"世界规则“{key}”被违反"


def detect_issues(directives: list[ParsedDirective]) -> list[ConsistencyIssue]:
    issues: list[ConsistencyIssue] = []

    facts: dict[tuple[str, str], list[ParsedDirective]] = defaultdict(list)
    events: dict[tuple[str, str], list[ParsedDirective]] = defaultdict(list)
    knows: dict[tuple[str, str], list[ParsedDirective]] = defaultdict(list)
    claims: list[ParsedDirective] = []
    owners: dict[str, list[ParsedDirective]] = defaultdict(list)
    uses: list[ParsedDirective] = []
    rules: dict[str, list[ParsedDirective]] = defaultdict(list)
    assertions: list[ParsedDirective] = []
    mobility_permissions: list[ParsedDirective] = []
    rule_exceptions: list[ParsedDirective] = []

    for d in directives:
        a = d.attrs
        if d.kind == "fact":
            facts[(a.get("subject", ""), a.get("predicate", ""))].append(d)
            if a.get("predicate") == "mobility_permission":
                mobility_permissions.append(d)
            elif a.get("predicate") == "rule_exception":
                rule_exceptions.append(d)
        elif d.kind == "event":
            for participant in a.get("participants", "").split(","):
                if participant.strip():
                    events[(participant.strip(), a.get("time", ""))].append(d)
        elif d.kind == "knows":
            knows[(a.get("character", ""), a.get("fact", ""))].append(d)
        elif d.kind == "claims_knows":
            claims.append(d)
        elif d.kind == "item":
            owners[a.get("item", "")].append(d)
        elif d.kind == "uses":
            uses.append(d)
        elif d.kind == "world_rule":
            rules[a.get("key", "")].append(d)
        elif d.kind == "world_assert":
            assertions.append(d)

    for (subject, predicate), rows in facts.items():
        if predicate in {"mobility_permission", "rule_exception"}:
            continue
        conflict_pair: tuple[ParsedDirective, ParsedDirective] | None = None
        for index, first in enumerate(rows):
            for second in rows[index + 1:]:
                if not first.attrs.get("value") or first.attrs.get("value") == second.attrs.get("value"):
                    continue
                first_time, second_time = first.attrs.get("time", ""), second.attrs.get("time", "")
                if first_time and second_time and _time_key(first_time) != _time_key(second_time):
                    continue
                conflict_pair = (first, second)
                break
            if conflict_pair:
                break
        if conflict_pair:
            first, second = conflict_pair
            issues.append(_issue(
                IssueCategory.fact_conflict,
                f"{subject}的{_fact_label(predicate)}存在冲突",
                f"同一事实被描述为“{first.attrs.get('value')}”和“{second.attrs.get('value')}”。",
                [first.evidence, second.evidence],
                "确认权威设定并统一两处描述；如为阶段变化，请补充明确时间点。",
                subject=subject,
                predicate=predicate,
            ))

    for (participant, timestamp), rows in events.items():
        if not _precise_timestamp(timestamp):
            continue
        conflict_pair: tuple[ParsedDirective, ParsedDirective] | None = None
        for index, first in enumerate(rows):
            for second in rows[index + 1:]:
                first_location = first.attrs.get("location", "")
                second_location = second.attrs.get("location", "")
                if not first_location or not second_location or first_location == second_location:
                    continue
                first_span = (
                    first.evidence.document_id,
                    first.evidence.line_start,
                    first.evidence.line_end,
                )
                second_span = (
                    second.evidence.document_id,
                    second.evidence.line_start,
                    second.evidence.line_end,
                )
                if first_span == second_span:
                    continue
                if _evidence_shares_place(first, second):
                    continue
                if any(
                    _mobility_permission_applies(
                        permission,
                        participant,
                        timestamp,
                        first_location,
                        second_location,
                    )
                    for permission in mobility_permissions
                ):
                    continue
                conflict_pair = (first, second)
                break
            if conflict_pair:
                break
        if conflict_pair:
            first, second = conflict_pair
            issues.append(_issue(
                IssueCategory.location_collision,
                f"{participant}在同一时间出现在不同地点",
                f"{timestamp} 同时记录了“{first.attrs.get('location')}”与“{second.attrs.get('location')}”。",
                [first.evidence, second.evidence],
                "调整事件时间、补充瞬移规则，或修正其中一处地点。",
                participant=participant,
                timestamp=timestamp,
            ))

    for claim in claims:
        key = (claim.attrs.get("character", ""), claim.attrs.get("fact", ""))
        acquisitions = knows.get(key, [])
        claim_time = claim.attrs.get("time", "")
        # Absence of an acquisition record is incomplete information, not
        # proof of a continuity error.  A knowledge issue needs an explicit
        # later acquisition (or equivalent canonical constraint).
        if not acquisitions:
            continue
        claim_span = (
            claim.evidence.document_id,
            claim.evidence.line_start,
            claim.evidence.line_end,
        )
        if any(
            (
                acquisition.evidence.document_id,
                acquisition.evidence.line_start,
                acquisition.evidence.line_end,
            ) == claim_span
            for acquisition in acquisitions
        ):
            continue
        valid = [
            d for d in acquisitions
            if d.attrs.get("time", "") and _time_key(d.attrs.get("time", "")) <= _time_key(claim_time)
        ]
        if not valid:
            evidence = [claim.evidence]
            if acquisitions:
                evidence.append(sorted(acquisitions, key=lambda d: d.attrs.get("time", ""))[0].evidence)
            issues.append(_issue(
                IssueCategory.knowledge_without_acquisition,
                f"{key[0]}过早掌握“{key[1]}”",
                "角色在明确获知该信息之前就进行了引用或行动。",
                evidence,
                "提前安排信息获得事件，或改写当前台词使其符合角色认知。",
                character=key[0],
                fact=key[1],
                claim_time=claim_time,
            ))

    for use in uses:
        item = use.attrs.get("item", "")
        user = use.attrs.get("user", "")
        use_time = use.attrs.get("time", "")
        candidates = [
            d
            for d in owners.get(item, [])
            if _state_applies_at(d.attrs.get("time", ""), use_time)
        ]
        if candidates:
            owner = sorted(candidates, key=lambda d: d.attrs.get("time", ""))[-1]
            if owner.attrs.get("owner") != user:
                issues.append(_issue(
                    IssueCategory.item_ownership,
                    f"{user}使用了不属于自己的{item}",
                    f"使用发生时最近的持有者记录为“{owner.attrs.get('owner')}”。",
                    [owner.evidence, use.evidence],
                    "补充交接/借用事件，或修改使用者。",
                    item=item,
                    expected_owner=owner.attrs.get("owner"),
                    actual_user=user,
                ))

    for assertion in assertions:
        key = assertion.attrs.get("key", "")
        expected = rules.get(key, [])
        if expected and expected[-1].attrs.get("value") != assertion.attrs.get("value"):
            if any(
                _rule_exception_applies(
                    exception,
                    assertion.attrs.get("actor", ""),
                    key,
                    assertion.attrs.get("time", ""),
                )
                for exception in rule_exceptions
            ):
                continue
            issues.append(_issue(
                IssueCategory.world_rule_conflict,
                _world_rule_title(key),
                f"权威规则为“{expected[-1].attrs.get('value')}”，当前剧情写为“{assertion.attrs.get('value')}”。",
                [expected[-1].evidence, assertion.evidence],
                "遵循既有规则，或在世界观文档中正式引入规则例外及其代价。",
                key=key,
            ))

    return issues
