from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import re

from .domain import ConsistencyIssue, IssueCategory, ParsedDirective, Severity
from .semantic_quality import eligible_for_deterministic_rules


_CHINESE_ABSOLUTE_TIME = re.compile(
    r"(?P<year>[0-9]{4})[ \t]*年[ \t]*"
    r"(?P<month>[0-9]{1,2})[ \t]*月[ \t]*"
    r"(?P<day>[0-9]{1,2})[ \t]*日[ \t]*"
    r"(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2})"
    r"(?::(?P<second>[0-9]{2}))?"
)


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


def _event_precise_timestamp(value: str) -> datetime | None:
    """Validate event-only absolute time without widening knowledge ordering."""
    candidate = value.strip()
    if _precise_timestamp(candidate):
        normalized = candidate.replace("T", " ")
        time_format = (
            "%Y-%m-%d %H:%M:%S" if len(normalized) == 19 else "%Y-%m-%d %H:%M"
        )
        try:
            return datetime.strptime(normalized, time_format)
        except ValueError:
            return None
    match = _CHINESE_ABSOLUTE_TIME.fullmatch(candidate)
    if match is None:
        return None
    parts = {key: int(value) for key, value in match.groupdict(default="0").items()}
    try:
        return datetime(
            parts["year"], parts["month"], parts["day"],
            parts["hour"], parts["minute"], parts["second"],
        )
    except ValueError:
        return None


def _chinese_event_time_visible(value: str, evidence: str) -> bool:
    """Require the whole Chinese timestamp and reject nearby uncertainty."""
    candidate = value.strip()
    if _CHINESE_ABSOLUTE_TIME.fullmatch(candidate) is None:
        return True
    compact_time = re.sub(r"[ \t]+", "", candidate)
    compact_evidence = re.sub(r"[ \t]+", "", evidence)
    match = re.search(
        rf"(?<![0-9]){re.escape(compact_time)}(?![:0-9])",
        compact_evidence,
    )
    if match is None:
        return False
    before = compact_evidence[max(0, match.start() - 6) : match.start()]
    after = compact_evidence[match.end() : match.end() + 8]
    if re.search(r"(?:约|大约|大概|可能|也许|估计|不早于|不晚于)(?:在)?$", before):
        return False
    if re.match(r"(?:左右|前后|附近|或者|或|[/／~～—-]|(?:到|至)[0-9])", after):
        return False
    return True


def _event_permission_timestamp(value: str, parsed: datetime) -> str:
    """Keep display evidence original while comparing Chinese events to ISO bounds."""
    if _CHINESE_ABSOLUTE_TIME.fullmatch(value.strip()) is None:
        return value
    time_format = "%Y-%m-%d %H:%M:%S" if value.count(":") == 2 else "%Y-%m-%d %H:%M"
    return parsed.strftime(time_format)


def _ordered_knowledge_time(value: str) -> datetime | None:
    """Return only valid ISO-like times that have a total chronological order."""

    candidate = value.strip()
    if not _precise_timestamp(candidate):
        return None
    normalized = candidate.replace("T", " ")
    time_format = (
        "%Y-%m-%d %H:%M:%S" if len(normalized) == 19 else "%Y-%m-%d %H:%M"
    )
    try:
        return datetime.strptime(normalized, time_format)
    except ValueError:
        return None


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


_EXTERIOR_PLACE_CUE = re.compile(
    r"门外|门前|附近|旁边|对面|前方|后方|外港|外城|城外|港外|"
    r"塔外|楼外|院外|站外|室外|馆外|外围|外侧|外面|外部"
)
_INTERIOR_PLACE_SUFFIX = re.compile(
    r"(?:房间|书房|[\u4e00-\u9fffA-Za-z0-9·_-]{1,12}"
    r"(?:厅|室|馆|楼|院|库|站|舱|层))"
)


def _evidence_shares_place(first: ParsedDirective, second: ParsedDirective) -> bool:
    """Suppress only an explicit parent/inner-place relation in source text.

    Shared short tokens such as ``灯塔`` do not establish that two full place
    names are the same location. Mere cross-mentions do not establish nesting
    either: a character may be at one place while discussing another.
    """
    first_location = re.sub(r"\s+", "", first.attrs.get("location", ""))
    second_location = re.sub(r"\s+", "", second.attrs.get("location", ""))
    if not first_location or not second_location:
        return False
    evidence = tuple(
        re.sub(r"\s+", "", row.evidence.text) for row in (first, second)
    )
    for outer, inner in (
        (first_location, second_location),
        (second_location, first_location),
    ):
        # "X 的门外", "X 外港的档案厅" and similar nearby places are not inside X.
        if _EXTERIOR_PLACE_CUE.search(inner):
            continue
        # A full extracted name can itself encode an inner facility. Do not
        # silently trim an action tail from a malformed location here; that
        # belongs to extraction, not to the collision rule.
        if inner.startswith(outer):
            suffix = inner[len(outer):]
            if _INTERIOR_PLACE_SUFFIX.fullmatch(suffix) and any(
                inner in text for text in evidence
            ):
                return True
        possessive = re.compile(
            rf"{re.escape(outer)}(?:的|内的|里的|中的){re.escape(inner)}"
        )
        located_inside = re.compile(
            rf"{re.escape(inner)}(?:位于|坐落于|设于){re.escape(outer)}(?:内|里|中)"
        )
        if any(possessive.search(text) or located_inside.search(text) for text in evidence):
            return True
    return False


def _same_evidence(first: ParsedDirective, second: ParsedDirective) -> bool:
    return (
        first.evidence.document_id == second.evidence.document_id
        and first.evidence.line_start == second.evidence.line_start
        and first.evidence.line_end == second.evidence.line_end
    )


def _scopes_compatible(first: ParsedDirective, second: ParsedDirective) -> bool:
    """Global records apply to a branch; sibling branch records do not meet."""
    first_scope = first.attrs.get("story_scope", "") or "global"
    second_scope = second.attrs.get("story_scope", "") or "global"
    return (
        first_scope == "global"
        or second_scope == "global"
        or first_scope == second_scope
    )


def _mobility_limit_applies(
    state: ParsedDirective,
    participant: str,
    origin: str,
    destination: str,
) -> bool:
    attrs = state.attrs
    subject = attrs.get("subject", "")
    if subject and subject not in {participant, "*", "任何人", "普通人", "所有人"}:
        return False
    route = (attrs.get("origin", ""), attrs.get("destination", ""))
    forward = _location_within(origin, route[0]) and _location_within(destination, route[1])
    reverse = _location_within(origin, route[1]) and _location_within(destination, route[0])
    return bool(forward or (attrs.get("bidirectional") == "true" and reverse))


def _authorization_terms(text: str) -> set[str]:
    terms = set(re.findall(r"书面授权|授权|许可|豁免|批准|通行证|资格", text))
    return {"authorization"} if terms else set()


def _denied_authorization_applies(
    state: ParsedDirective,
    assertion: ParsedDirective,
    rule: ParsedDirective,
) -> bool:
    attrs = state.attrs
    actor = assertion.attrs.get("actor", "")
    if not actor or attrs.get("subject", "") != actor:
        return False
    if not _scopes_compatible(state, assertion) or not _scopes_compatible(state, rule):
        return False
    if attrs.get("predicate") == "rule_exception":
        return attrs.get("value") == "denied" and attrs.get("key") == rule.attrs.get("key")
    if attrs.get("polarity") != "negative" and attrs.get("modality") != "negated":
        return False
    state_terms = _authorization_terms(
        " ".join(
            [
                attrs.get("predicate", ""),
                attrs.get("value", ""),
                state.evidence.text,
            ]
        )
    )
    rule_terms = _authorization_terms(rule.evidence.text)
    return bool(state_terms & rule_terms)


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
    mobility_limits: list[ParsedDirective] = []
    rule_exceptions: list[ParsedDirective] = []
    denied_authorizations: list[ParsedDirective] = []

    for d in directives:
        if not eligible_for_deterministic_rules(d):
            continue
        a = d.attrs
        if d.kind == "fact":
            facts[(a.get("subject", ""), a.get("predicate", ""))].append(d)
            if a.get("predicate") == "mobility_permission":
                mobility_permissions.append(d)
            elif a.get("predicate") == "mobility_limit":
                mobility_limits.append(d)
            elif a.get("predicate") == "rule_exception":
                if a.get("value") == "denied":
                    denied_authorizations.append(d)
                else:
                    rule_exceptions.append(d)
            elif (
                a.get("polarity") == "negative"
                and _authorization_terms(
                    " ".join(
                        [a.get("predicate", ""), a.get("value", ""), d.evidence.text]
                    )
                )
            ):
                denied_authorizations.append(d)
        elif d.kind == "event":
            timestamp = a.get("time", "")
            parsed_time = _event_precise_timestamp(timestamp)
            if parsed_time is None or not _chinese_event_time_visible(
                timestamp, d.evidence.text
            ):
                continue
            # Normalize Chinese spelling/spacing only. Existing ISO grouping
            # remains exact to avoid an unrelated change in rule scope.
            event_key = (
                f"chinese:{parsed_time.isoformat()}"
                if _CHINESE_ABSOLUTE_TIME.fullmatch(timestamp.strip())
                else timestamp
            )
            for participant in a.get("participants", "").split(","):
                if participant.strip():
                    events[(participant.strip(), event_key)].append(d)
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
        if predicate in {"mobility_permission", "mobility_limit", "rule_exception"}:
            continue
        conflict_pair: tuple[ParsedDirective, ParsedDirective] | None = None
        for index, first in enumerate(rows):
            for second in rows[index + 1:]:
                if _same_evidence(first, second):
                    continue
                if not _scopes_compatible(first, second):
                    continue
                first_value = first.attrs.get("value", "")
                second_value = second.attrs.get("value", "")
                if not first_value or not second_value:
                    continue
                first_negative = first.attrs.get("polarity") == "negative"
                second_negative = second.attrs.get("polarity") == "negative"
                if first_negative == second_negative:
                    # Two affirmative values conflict when they differ. Two
                    # negative exclusions never prove which value is true.
                    if first_negative or first_value == second_value:
                        continue
                elif first_value != second_value:
                    # “不是银色” does not contradict “黑色”.
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

    for (participant, _event_key), rows in events.items():
        timestamp = rows[0].attrs.get("time", "")
        parsed_time = _event_precise_timestamp(timestamp)
        if parsed_time is None:
            continue
        permission_timestamp = _event_permission_timestamp(timestamp, parsed_time)
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
                if not _scopes_compatible(first, second):
                    continue
                if _evidence_shares_place(first, second):
                    continue
                if any(
                    _scopes_compatible(permission, first)
                    and _scopes_compatible(permission, second)
                    and _mobility_permission_applies(
                        permission,
                        participant,
                        permission_timestamp,
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
            supporting_limits = [
                limit
                for limit in mobility_limits
                if _scopes_compatible(limit, first)
                and _scopes_compatible(limit, second)
                and _mobility_limit_applies(
                    limit,
                    participant,
                    first.attrs.get("location", ""),
                    second.attrs.get("location", ""),
                )
            ]
            issue_evidence = [first.evidence, second.evidence]
            if supporting_limits:
                issue_evidence.insert(0, supporting_limits[0].evidence)
            issues.append(_issue(
                IssueCategory.location_collision,
                f"{participant}在同一时间出现在不同地点",
                f"{timestamp} 同时记录了“{first.attrs.get('location')}”与“{second.attrs.get('location')}”。",
                issue_evidence,
                "调整事件时间、补充瞬移规则，或修正其中一处地点。",
                participant=participant,
                timestamp=timestamp,
            ))

    for claim in claims:
        key = (claim.attrs.get("character", ""), claim.attrs.get("fact", ""))
        acquisitions = [
            row for row in knows.get(key, []) if _scopes_compatible(row, claim)
        ]
        claim_time = claim.attrs.get("time", "")
        claim_order = _ordered_knowledge_time(claim_time)
        if claim_order is None:
            continue
        # Absence of an acquisition record is incomplete information, not
        # proof of a continuity error. Only valid ISO-like times can establish
        # a before/after relation; natural labels remain displayable evidence
        # but must never be compared by string order.
        ordered_acquisitions = [
            (row, order)
            for row in acquisitions
            if (order := _ordered_knowledge_time(row.attrs.get("time", "")))
            is not None
        ]
        if not ordered_acquisitions:
            continue
        claim_span = (
            claim.evidence.document_id,
            claim.evidence.line_start,
            claim.evidence.line_end,
        )
        if any(
            (
                row.evidence.document_id,
                row.evidence.line_start,
                row.evidence.line_end,
            ) == claim_span
            for row, _ in ordered_acquisitions
        ):
            continue
        if not any(order <= claim_order for _, order in ordered_acquisitions):
            evidence = [claim.evidence]
            earliest, _ = min(ordered_acquisitions, key=lambda row: row[1])
            evidence.append(earliest.evidence)
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
            and _scopes_compatible(d, use)
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
        expected = [
            row for row in rules.get(key, []) if _scopes_compatible(row, assertion)
        ]
        canonical_rule = expected[-1] if expected else None
        if canonical_rule and _same_evidence(canonical_rule, assertion):
            continue
        if canonical_rule and canonical_rule.attrs.get("value") != assertion.attrs.get("value"):
            if any(
                _scopes_compatible(exception, assertion)
                and _rule_exception_applies(
                    exception,
                    assertion.attrs.get("actor", ""),
                    key,
                    assertion.attrs.get("time", ""),
                )
                for exception in rule_exceptions
            ):
                continue
            denied = [
                state
                for state in denied_authorizations
                if _denied_authorization_applies(state, assertion, canonical_rule)
            ]
            issue_evidence = [canonical_rule.evidence]
            issue_evidence.extend(state.evidence for state in denied[:1])
            issue_evidence.append(assertion.evidence)
            issues.append(_issue(
                IssueCategory.world_rule_conflict,
                _world_rule_title(key),
                f"权威规则为“{canonical_rule.attrs.get('value')}”，当前剧情写为“{assertion.attrs.get('value')}”。",
                issue_evidence,
                "遵循既有规则，或在世界观文档中正式引入规则例外及其代价。",
                key=key,
            ))

    return issues
