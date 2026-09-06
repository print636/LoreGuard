from __future__ import annotations

import re

from .domain import EvidenceSpan, ParsedDirective
from .semantic_quality import open_question_directives


TIME = r"(?P<time>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2})"
NAME = r"[\u4e00-\u9fffA-Za-z0-9·_-]{1,24}?"
PERSON = r"[\u4e00-\u9fffA-Za-z·_-]{2,12}?"
LOCATION_END = (
    rf"(?=(?:与{PERSON})?(?:检查|会面|见面|交谈|巡逻|等待|签署|执行|工作|停留|驻守|盘点|调查|把|向|告诉|展示|第一次)|[，,。；;]|$)"
)


def _directive(kind: str, attrs: dict[str, str], evidence: EvidenceSpan) -> ParsedDirective:
    return ParsedDirective(kind=kind, attrs={k: v.strip(" ，。；;：:") for k, v in attrs.items()}, evidence=evidence)


def _referenced_rule(text: str, evidence: EvidenceSpan) -> ParsedDirective | None:
    match = re.search(
        r"根据(?:世界观)?规则(?P<label>[^：:，,。；;]{0,12})[：:]\s*(?P<statement>[^。；;？?]{2,100})",
        text,
    )
    if not match:
        return None
    statement = match.group("statement").strip()
    copula = re.fullmatch(
        r"(?P<subject>[^，,。；;：:]{1,30}?)(?:是|为)(?P<value>[^，,。；;：:]{1,50})",
        statement,
    )
    if copula:
        key = copula.group("subject")
        value = copula.group("value")
    else:
        action = re.fullmatch(
            r"(?P<subject>[^，,。；;：:]{1,20}?)(?P<predicate>提供|禁止|允许|负责|保护|遵守|供应)"
            r"(?P<value>[^，,。；;：:]{1,50})",
            statement,
        )
        if action:
            key = f"{action.group('subject')}{action.group('predicate')}"
            value = action.group("value")
        else:
            key = f"规则{match.group('label') or '（未编号）'}"
            value = statement
    return _directive(
        "world_rule",
        {
            "key": key,
            "value": value,
            "modality": "asserted",
            "source_scope": "world_rule",
            "certainty": "certain",
        },
        evidence.model_copy(update={"text": match.group(0).strip()}),
    )


def _conditional_rule(text: str, evidence: EvidenceSpan) -> ParsedDirective | None:
    match = re.search(
        r"(?:如果|当|一旦|若(?:是)?)(?P<condition>[^，,。；;？?]{2,48})[，,]"
        r"(?:则|就|将|会|必然|一律)?(?P<consequence>[^。；;？?]{2,60})",
        text,
    ) or re.search(
        r"若(?P<condition>[^，,。；;？?]{2,30}?)(?:则|就)"
        r"(?P<consequence>[^。；;？?]{2,60})",
        text,
    )
    if not match or re.search(r"[？?]|可能|也许|或许|未必|不一定", match.group(0)):
        return None
    consequence = match.group("consequence").strip()
    if not re.search(r"必然|必须|不得|不能|禁止|一律|只能|失效|无效|无法|则|就|会|将", match.group(0)):
        return None
    condition = match.group("condition").strip()
    return _directive(
        "world_rule",
        {
            "key": f"条件:{condition}",
            "value": consequence,
            "condition": condition,
            "consequence": consequence,
            "modality": "conditional_rule",
            "source_scope": "world_rule",
            "certainty": "certain",
        },
        evidence.model_copy(update={"text": match.group(0).strip()}),
    )


def _extract_timed_event(text: str, evidence: EvidenceSpan) -> ParsedDirective | None:
    adverb = r"(?:仍|依然|正|正在|还|已经|才)?"
    location = rf"(?P<location>[^，,。；;]{{1,40}}?){LOCATION_END}"
    patterns = [
        # Reporting phrases must be consumed before matching the person. This
        # prevents “巡逻记录显示林澈仍” from becoming a participant.
        rf"{TIME}[，,\s]+(?:[^，,。；;]{{1,24}}?)(?:显示|记载|表明|确认|称)[:：]?\s*"
        rf"(?P<participant>{PERSON}){adverb}在{location}",
        rf"{TIME}[，,\s]+(?P<participant>{PERSON}){adverb}在{location}",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        attrs = match.groupdict()
        return _directive(
            "event",
            {
                "id": f"natural-{evidence.line_start}",
                "time": attrs["time"],
                "location": attrs["location"],
                "participants": attrs["participant"],
            },
            evidence,
        )
    return None


def _mobility_limit(text: str, evidence: EvidenceSpan) -> ParsedDirective | None:
    # Double negation and hedged possibility do not establish a hard travel
    # limit, even though they contain the literal token “不可能”.
    if re.search(r"(?:并)?不是不可能|并非不可能|未必不可能|不可能不|不太可能", text):
        return None
    route = re.search(
        rf"(?P<subject>{NAME})在(?P<duration>[^，,。；;]{{1,12}}?)内"
        r"(?:绝不可能|不可能|无法|不能)(?:及时)?往返"
        r"(?P<origin>[^，,。；;与]{2,18})与(?P<destination>[^，,。；;]{2,18})",
        text,
    )
    subject = route.group("subject") if route else "*"
    if route is None:
        route = re.search(
            r"从(?P<origin>[^，,。；;]{2,18})到(?P<destination>[^，,。；;]{2,18})"
            r"至少需要(?P<duration>[^，,。；;]{1,12})",
            text,
        )
    if route is None:
        return None
    return _directive(
        "fact",
        {
            "subject": subject,
            "predicate": "mobility_limit",
            "value": "forbidden",
            "origin": route.group("origin"),
            "destination": route.group("destination"),
            "duration": route.group("duration"),
            "bidirectional": "true" if "往返" in route.group(0) else "false",
            "status": "active",
            "current": "true",
        },
        evidence,
    )


def _denied_authorization(text: str, evidence: EvidenceSpan) -> ParsedDirective | None:
    match = re.search(
        rf"(?P<subject>{NAME})(?:没有|并未|未曾|未)(?:取得|获得|拿到|持有|得到)?"
        rf"(?P<value>[^，,。；;]{{0,16}}?(?:书面授权|授权|许可|豁免|批准|通行证|资格))",
        text,
    )
    if not match:
        return None
    return _directive(
        "fact",
        {
            "subject": match.group("subject"),
            "predicate": "authorization",
            "value": match.group("value"),
            "modality": "negated",
            "polarity": "negative",
            "certainty": "certain",
        },
        evidence,
    )


def extract_natural_line(evidence: EvidenceSpan) -> list[ParsedDirective]:
    """Conservatively extract explicit Chinese facts without an API key."""
    text = evidence.text.strip()
    rows: list[ParsedDirective] = open_question_directives(evidence)

    referenced_rule = _referenced_rule(text, evidence)
    if referenced_rule:
        rows.append(referenced_rule)

    conditional_rule = _conditional_rule(text, evidence)
    if conditional_rule:
        rows.append(conditional_rule)

    travel_limit = _mobility_limit(text, evidence)
    if travel_limit:
        rows.append(travel_limit)

    denied_authorization = _denied_authorization(text, evidence)
    if denied_authorization:
        rows.append(denied_authorization)

    # Explicit rule references are represented as world rules instead of a
    # duplicate ordinary fact.  Questions and open hypotheses are never fed to
    # the permissive copula pattern.
    if not referenced_rule:
        negative_fact = re.search(
            rf"(?:{TIME}[，,\s]+)?(?P<subject>{NAME})的"
            rf"(?P<predicate>[\u4e00-\u9fffA-Za-z0-9_-]{{1,16}})(?:不是|并非|不为)"
            rf"(?P<value>[^，。；;]{{1,32}})",
            text,
        )
        fact = negative_fact or re.search(
            rf"(?:{TIME}[，,\s]+)?(?P<subject>{NAME})的"
            rf"(?P<predicate>[\u4e00-\u9fffA-Za-z0-9_-]{{1,16}})(?:是(?!否)|为)"
            rf"(?P<value>[^，。；;]{{1,32}})",
            text,
        )
        if fact:
            attrs = fact.groupdict()
            attrs["time"] = attrs.get("time") or ""
            if negative_fact:
                attrs.update(modality="negated", polarity="negative")
            rows.append(_directive("fact", attrs, evidence))

    timed_patterns = [
        ("knows", rf"{TIME}[，,\s]+(?P<character>{NAME})(?:得知|获知|知道了)(?P<fact>[^，。；;]{{1,36}})"),
        (
            "claims_knows",
            rf"{TIME}[，,\s]+(?P<character>{PERSON})(?:对[^，。；;]{{0,24}})?"
            rf"(?:准确|清楚|完整)?(?:说出|提到|引用)(?:了)?(?P<fact>[^，。；;]{{1,36}})",
        ),
        ("item", rf"{TIME}[，,\s]+(?P<owner>{NAME})(?:获得|持有|保管)(?P<item>[^，。；;]{{1,30}})"),
        ("uses", rf"{TIME}[，,\s]+(?P<user>{NAME})(?:使用|用)(?P<item>[^，。；;]{{1,30}})"),
    ]
    for kind, pattern in timed_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        attrs = match.groupdict()
        rows.append(_directive(kind, attrs, evidence))
        break

    event = _extract_timed_event(text, evidence)
    if event:
        rows.append(event)

    world_rule = re.search(
        r"(?P<key>[^，。；;]{2,30}?)(?:只能|必须)(?:由|使用)(?P<value>[^，。；;]{1,30}?)(?:驱动|启动|开启)",
        text,
    )
    if world_rule:
        rows.append(_directive("world_rule", world_rule.groupdict(), evidence))
    elif "驱动" in text or "启动" in text or "开启" in text:
        assertion = re.search(
            r"(?P<key>[^，。；;]{2,30}?)(?:由|使用)(?P<value>[^，。；;]{1,30}?)(?:驱动|启动|开启)",
            text,
        )
        if assertion:
            rows.append(_directive("world_assert", assertion.groupdict(), evidence))
    return rows
