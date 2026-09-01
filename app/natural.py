from __future__ import annotations

import re

from .domain import EvidenceSpan, ParsedDirective


TIME = r"(?P<time>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2})"
NAME = r"[\u4e00-\u9fffA-Za-z0-9·_-]{1,24}?"
PERSON = r"[\u4e00-\u9fffA-Za-z·_-]{2,12}?"
LOCATION_END = (
    rf"(?=(?:与{PERSON})?(?:检查|会面|见面|交谈|巡逻|等待|签署|执行|工作|停留|驻守|盘点|调查|把|向|告诉|展示|第一次)|[，,。；;]|$)"
)


def _directive(kind: str, attrs: dict[str, str], evidence: EvidenceSpan) -> ParsedDirective:
    return ParsedDirective(kind=kind, attrs={k: v.strip(" ，。；;：:") for k, v in attrs.items()}, evidence=evidence)


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


def extract_natural_line(evidence: EvidenceSpan) -> list[ParsedDirective]:
    """Conservatively extract explicit Chinese facts without an API key."""
    text = evidence.text.strip()
    rows: list[ParsedDirective] = []

    fact = re.search(
        rf"(?P<subject>{NAME})的(?P<predicate>[\u4e00-\u9fffA-Za-z0-9_-]{{1,16}})(?:是|为)(?P<value>[^，。；;]{{1,32}})",
        text,
    )
    if fact:
        rows.append(_directive("fact", fact.groupdict(), evidence))

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
