from __future__ import annotations

import re
from dataclasses import dataclass, field

from .domain import EvidenceSpan, ParsedDirective
from .pipeline import DocumentInput


NAME = r"[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9·_-]{0,11}?"
ACTOR_NAME = r"[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9·_-]{1,11}?"
PRONOUNS = {"他", "她", "他们", "她们"}
USE_VERBS = r"使用|启用|挥动|按下|盖下|刷过|插入|开启|启动"


@dataclass(slots=True)
class NormalizationResult:
    directives: list[ParsedDirective] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _clean(value: str) -> str:
    return re.sub(r"\s+", "", value).strip("，。；;：:\"'“”‘’")


def _fingerprint(directive: ParsedDirective) -> tuple:
    attrs = tuple(sorted((key, _clean(value)) for key, value in directive.attrs.items()))
    evidence = directive.evidence
    return directive.kind, attrs, evidence.document_id, evidence.line_start, evidence.line_end


def _evidence(document: DocumentInput, line_no: int, text: str) -> EvidenceSpan:
    return EvidenceSpan(
        document_id=document.id,
        document_name=document.name,
        line_start=line_no,
        line_end=line_no,
        text=text.strip(),
    )


def _directive(
    kind: str,
    attrs: dict[str, str],
    document: DocumentInput,
    line_no: int,
    text: str,
) -> ParsedDirective:
    return ParsedDirective(
        kind=kind,
        attrs={key: _clean(value) for key, value in attrs.items()},
        evidence=_evidence(document, line_no, text),
    )


def _resolve_actor(raw: str, last_actor: str | None) -> str | None:
    actor = _clean(raw)
    if actor in PRONOUNS:
        return last_actor
    return actor or None


def _explicit_actor(text: str) -> str | None:
    pattern = re.compile(
        rf"(?:^|[，,。；;：:])(?:[^，,。；;：:]{{0,8}}[，,])?"
        rf"(?P<actor>{ACTOR_NAME})(?:独自|仍|依然|已经|才|正在|亲手|下意识地)?"
        rf"(?:在|驶入|进入|抵达|返回|发现|取出|拿出|掏出|拔出|使用|发动|说出|抓住)"
    )
    match = pattern.search(text)
    if not match:
        return None
    actor = _clean(match.group("actor"))
    # A pronoun plus a short prepositional phrase (for example “他从匣中”)
    # can satisfy the loose actor pattern. It must not replace the previously
    # resolved explicit character.
    return None if any(actor.startswith(pronoun) for pronoun in PRONOUNS) else actor


def _item_owner_candidate(
    document: DocumentInput, line_no: int, text: str
) -> ParsedDirective | None:
    owner_match = re.search(
        rf"(?:平时|一直|通常|当前)?由(?P<owner>{NAME})(?:亲自)?(?:保管|持有|掌管)",
        text,
    )
    if not owner_match:
        return None

    quoted = re.findall(r"[“\"]([^”\"]{1,24})[”\"]", text[: owner_match.start()])
    item = quoted[-1] if quoted else ""
    if not item:
        explicit = re.search(
            rf"(?P<item>[\u4e00-\u9fffA-Za-z0-9·_-]{{2,20}}?)"
            rf"(?:平时|一直|通常|当前)?由{re.escape(owner_match.group('owner'))}",
            text,
        )
        if explicit:
            item = explicit.group("item")
    if not item:
        return None
    return _directive(
        "item",
        {"item": item, "owner": owner_match.group("owner"), "time": ""},
        document,
        line_no,
        text,
    )


def _item_use_candidate(
    document: DocumentInput,
    line_no: int,
    text: str,
    last_actor: str | None,
) -> ParsedDirective | None:
    take_then_use = re.search(
        rf"(?:^|[，,。；;：:])(?P<actor>他们|她们|他|她|{ACTOR_NAME})"
        rf"(?:[^，,。；;]{{0,12}})?"
        rf"(?:取出|拿出|掏出|拔出)(?P<item>[^，,。；;]{{1,20}}?)"
        rf"(?:[，,]|并|后)[^。；;]{{0,80}}?(?:{USE_VERBS})",
        text,
    )
    if take_then_use:
        actor = _resolve_actor(take_then_use.group("actor"), last_actor)
        if actor:
            return _directive(
                "uses",
                {"item": take_then_use.group("item"), "user": actor, "time": ""},
                document,
                line_no,
                text,
            )

    return None


def _canonical_scope(value: str) -> str:
    return re.sub(r"(?:的)?(?:中央|中心|内部|境内|区域|范围|之中|中|内)$", "", _clean(value))


def _canonical_action(value: str) -> str:
    cleaned = re.sub(r"^(?:一切|所有|任何)", "", _clean(value))
    return re.sub(r"(?:能力|法术)$", lambda match: match.group(0) if cleaned == match.group(0) else "", cleaned)


def _validity_attrs(text: str) -> dict[str, str]:
    timestamps = re.findall(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", text)
    invalid = bool(re.search(r"当前无效|已经无效|已撤销|已失效|过期", text))
    attrs = {
        "status": "inactive" if invalid else "active",
        "valid_from": "",
        "valid_until": "",
    }
    if "过期" in text:
        attrs["status"] = "expired"
    if len(timestamps) >= 2 and re.search(r"(?:从|自).{0,12}(?:到|至)", text):
        attrs["valid_from"], attrs["valid_until"] = timestamps[-2:]
    elif timestamps and re.search(r"(?:截至|有效期至|在).{0,8}" + re.escape(timestamps[-1]), text):
        attrs["valid_until"] = timestamps[-1]
    if re.search(r"当前有效|仍有效|仍生效|处于有效状态|豁免状态有效|许可.*有效", text):
        attrs["current"] = "true"
    return attrs


def _mobility_permission_candidate(
    document: DocumentInput, line_no: int, text: str
) -> ParsedDirective | None:
    if not re.search(r"瞬时通行许可|传送通行证|跃迁权限|跃迁许可|折跃门|传送权限|定向传送许可", text):
        return None
    actor_patterns = [
        rf"(?:向|确认)(?P<actor>{ACTOR_NAME})(?:的|签发|获准|持有)",
        rf"(?P<actor>{ACTOR_NAME})(?:持有|获准|的(?:跃迁权限|传送权限|定向传送许可))",
        rf"(?P<actor>{ACTOR_NAME})持有有效许可",
    ]
    actor = next((match.group("actor") for pattern in actor_patterns if (match := re.search(pattern, text))), "")
    route_patterns = [
        r"从(?P<origin>[^，,。；;：:]{2,18}?)(?:瞬时)?(?:抵达|到|前往)(?P<destination>[^，,。；;]{2,18}?)(?=，|。|；|有效|$)",
        r"往返(?P<origin>[^，,。；;]{2,18}?)与(?P<destination>[^，,。；;]{2,18}?)(?=，|。|；|$)",
        r"路线为(?P<origin>[^，,。；;]{2,18}?)至(?P<destination>[^，,。；;]{2,18}?)(?=，|。|；|$)",
        r"覆盖(?P<origin>[^，,。；;]{2,18}?)和(?P<destination>[^，,。；;]{2,18}?)之间",
        r"在(?P<origin>[^，,。；;]{2,18}?)与(?P<destination>[^，,。；;]{2,18}?)之间瞬时移动",
    ]
    route = next((match for pattern in route_patterns if (match := re.search(pattern, text))), None)
    if not actor or route is None:
        return None
    bidirectional = bool(re.search(r"往返|双向|之间瞬时移动", text))
    return _directive(
        "fact",
        {
            "subject": actor,
            "predicate": "mobility_permission",
            "value": "instant_transport",
            "origin": route.group("origin"),
            "destination": route.group("destination"),
            "bidirectional": "true" if bidirectional else "false",
            **_validity_attrs(text),
        },
        document,
        line_no,
        text,
    )


def _rule_exception_candidate(
    document: DocumentInput, line_no: int, text: str
) -> ParsedDirective | None:
    badge = re.search(
        rf"(?:进入|位于|在)(?P<context>[^，,。；;]{{2,24}}?)(?:后|中|内)[，,]?"
        rf"(?:一切|所有|任何)?(?P<action>[^，,。；;]{{2,24}}?)(?:都会|会|将会|将)?(?:失效|无法(?:发动|使用|施展)|不能(?:发动|使用|施展))"
        rf"[^；;\n]*?(?:豁免徽记|豁免凭证)[^；;。]*?除外[；;，,](?P<actor>{ACTOR_NAME})持有有效(?:徽记|凭证)",
        text,
    )
    if badge:
        context = _canonical_scope(badge.group("context"))
        action = _canonical_action(badge.group("action"))
        return _directive(
            "fact",
            {
                "subject": badge.group("actor"),
                "predicate": "rule_exception",
                "value": "allowed",
                "key": f"scope_action:{context}:{action}",
                "status": "active",
                "current": "true",
            },
            document,
            line_no,
            text,
        )
    if not re.search(r"例外资格|规则豁免|特别许可|获准|授权", text):
        return None
    patterns = [
        rf"(?:只有|仅)?(?P<actor>{ACTOR_NAME})(?:拥有|获准)(?:在)(?P<context>[^，,。；;]{{2,24}}?)(?:中|内)?(?:使用|发动|施展)(?P<action>[^，,。；;]{{2,24}}?)(?:的例外资格|，|。|；|$)",
        rf"(?:议会)?授权(?P<actor>{ACTOR_NAME})在(?P<context>[^，,。；;]{{2,24}}?)(?:中|内)?(?:使用|发动|施展)(?P<action>[^，,。；;]{{2,24}}?)(?:，|。|；|$)",
        rf"(?P<actor>{ACTOR_NAME})持有[^，,。；;]{{0,18}}规则豁免[，,]可在(?P<context>[^，,。；;]{{2,24}}?)(?:中|内)?(?:使用|发动|施展)(?P<action>[^，,。；;]{{2,24}}?)(?:，|。|；|$)",
        rf"(?:确认[:：])?(?P<actor>{ACTOR_NAME})在(?P<context>[^，,。；;]{{2,24}}?)(?:中|内)?(?:使用|发动|施展)(?P<action>[^，,。；;]{{2,24}}?)的特别许可",
    ]
    match = next((candidate for pattern in patterns if (candidate := re.search(pattern, text))), None)
    if match is None:
        return None
    context = _canonical_scope(match.group("context"))
    action = _canonical_action(match.group("action"))
    if not context or not action:
        return None
    return _directive(
        "fact",
        {
            "subject": match.group("actor"),
            "predicate": "rule_exception",
            "value": "allowed",
            "key": f"scope_action:{context}:{action}",
            **_validity_attrs(text),
        },
        document,
        line_no,
        text,
    )


def _rule_candidate(
    document: DocumentInput, line_no: int, text: str
) -> ParsedDirective | None:
    patterns = [
        re.compile(
            r"(?:进入|位于|在)(?P<context>[^，,。；;]{2,24}?)(?:后|中|内)[，,]?"
            r"(?:一切|所有|任何)?(?P<action>[^，,。；;]{2,24}?)"
            r"(?:都会|会|将会|将)?(?:失效|无法(?:发动|使用|施展)|不能(?:发动|使用|施展))"
        ),
        re.compile(
            r"(?:进入|位于|在)(?P<context>[^，,。；;]{2,24}?)(?:后|中|内)[，,]?"
            r"(?:禁止|不得)(?:发动|使用|施展)?(?P<action>[^，,。；;]{2,24})"
        ),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if not match:
            continue
        context = _canonical_scope(match.group("context"))
        action = _canonical_action(match.group("action"))
        if context and action:
            return _directive(
                "world_rule",
                {"key": f"scope_action:{context}:{action}", "value": "disabled"},
                document,
                line_no,
                text,
            )
    return None


def _assertion_candidate(
    document: DocumentInput, line_no: int, text: str, last_actor: str | None
) -> ParsedDirective | None:
    match = re.search(
        rf"(?:(?P<actor>{ACTOR_NAME})(?:仍|依然|却|竟然)?)?在(?P<context>[^，,。；;]{{2,24}}?)"
        r"(?:中央|中心|内部|境内|区域|范围|之中|中|内)"
        r"(?:仍|依然|却|竟然)?(?:发动|使用|施展|启动|开启)"
        r"(?P<action>[^，,。；;]{2,24})",
        text,
    )
    if not match:
        return None
    context = _canonical_scope(match.group("context"))
    action = _canonical_action(match.group("action"))
    actor = _resolve_actor(match.group("actor") or "", last_actor)
    if not context or not action:
        return None
    return _directive(
        "world_assert",
        {"key": f"scope_action:{context}:{action}", "value": "performed", "actor": actor or ""},
        document,
        line_no,
        text,
    )


def _body_state_candidate(
    document: DocumentInput,
    line_no: int,
    text: str,
    last_actor: str | None,
) -> ParsedDirective | None:
    recovery_words = r"再生|恢复|重建|治愈|重新长出|接回"
    loss = re.search(
        r"(?P<actor>[\u4e00-\u9fffA-Za-z·_-]{2,12}?)(?:在[^，,。；;]{0,30}?)?"
        r"(?:失去|失去了|截去|截除了|没有了)"
        r"(?P<side>左|右)(?P<part>臂|手|腿|脚|眼|耳)",
        text,
    )
    if loss and not re.search(recovery_words, text[loss.start() :]):
        part = _canonical_body_part(loss.group("side"), loss.group("part"))
        return _directive(
            "fact",
            {
                "subject": loss.group("actor"),
                "predicate": f"body_state:{part}",
                "value": "missing_or_prosthetic",
            },
            document,
            line_no,
            text,
        )

    if re.search(r"看似|仿佛|伪装|幻象|投影|假装", text):
        return None
    intact = re.search(
        r"(?:伸出|抬起|举起|使用|挥动)[^，,。；;]{0,12}?"
        r"(?:完好|健全|没有受伤)(?:的)?(?P<side>左|右)(?P<part>手|臂|腿|脚|眼|耳)",
        text,
    )
    if intact and last_actor:
        part = _canonical_body_part(intact.group("side"), intact.group("part"))
        return _directive(
            "fact",
            {
                "subject": last_actor,
                "predicate": f"body_state:{part}",
                "value": "intact",
            },
            document,
            line_no,
            text,
        )
    return None


def _canonical_body_part(side: str, part: str) -> str:
    family = {"臂": "上肢", "手": "上肢", "腿": "下肢", "脚": "下肢"}.get(part, part)
    return f"{side}{family}"


def _canonical_knowledge_fact(value: str, source_text: str = "") -> str:
    fact = _clean(value)
    fact = re.sub(r"^(?:了|关于|有关)", "", fact)
    fact = fact.replace("经纬坐标", "位置").replace("坐标", "位置")
    fact = fact.replace("真实", "").replace("的", "")
    if fact.startswith("入口"):
        topics = re.findall(r"[“\"]([^”\"]{2,24})[”\"]", source_text)
        if topics:
            fact = f"{_clean(topics[-1])}{fact}"
    return fact


def _knowledge_candidates(
    document: DocumentInput,
    line_no: int,
    text: str,
    last_actor: str | None,
) -> list[ParsedDirective]:
    rows: list[ParsedDirective] = []
    timestamp = re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", text)
    if not timestamp:
        return rows
    time_value = timestamp.group(0)

    claim = re.search(
        rf"{re.escape(time_value)}[，,\s]+(?P<character>{ACTOR_NAME})"
        r"(?:对[^，,。；;]{0,24})?(?:准确|清楚|完整)?"
        r"(?:说出|提到|引用)(?:了)?(?P<fact>[^，,。；;]{2,48})",
        text,
    )
    if claim:
        rows.append(
            _directive(
                "claims_knows",
                {
                    "character": claim.group("character"),
                    "fact": _canonical_knowledge_fact(claim.group("fact"), text),
                    "time": time_value,
                },
                document,
                line_no,
                text,
            )
        )

    told = re.search(
        rf"第一次把(?P<fact>[^，,。；;]{{2,48}}?)告诉(?P<recipient>{ACTOR_NAME})",
        text,
    )
    if told:
        rows.append(
            _directive(
                "knows",
                {
                    "character": told.group("recipient"),
                    "fact": _canonical_knowledge_fact(told.group("fact"), text),
                    "time": time_value,
                },
                document,
                line_no,
                text,
            )
        )

    shown = re.search(
        rf"第一次向(?P<recipient>他们|她们|他|她|{ACTOR_NAME})[^，,。；;]{{0,36}}"
        r"[，,](?:正式)?告知(?P<fact>[^，,。；;]{2,48})",
        text,
    )
    if shown:
        recipient = _resolve_actor(shown.group("recipient"), last_actor)
        if recipient:
            rows.append(
                _directive(
                    "knows",
                    {
                        "character": recipient,
                        "fact": _canonical_knowledge_fact(shown.group("fact"), text),
                        "time": time_value,
                    },
                    document,
                    line_no,
                    text,
                )
            )
    source_patterns = [
        (
            "letter",
            rf"(?P<character>{ACTOR_NAME})(?:阅读(?:了)?(?:来信|信件|密函)后|通过(?:加密)?(?:来信|信件|密函))(?:得知|获知|知道了)(?P<fact>[^，,。；;]{{2,48}})",
        ),
        (
            "witness",
            rf"(?P<character>{ACTOR_NAME})亲眼目击(?:并)?(?:得知|获知|知道了)(?P<fact>[^，,。；;]{{2,48}})",
        ),
        (
            "told",
            rf"(?:{ACTOR_NAME})把(?P<fact>[^，,。；;]{{2,48}}?)告诉(?P<character>{ACTOR_NAME})",
        ),
        (
            "reading",
            rf"(?P<character>{ACTOR_NAME})(?:查阅|阅读)(?:了)?[^，,。；;]{{1,24}}[，,](?:得知|获知|知道了)(?P<fact>[^，,。；;]{{2,48}})",
        ),
        (
            "notice",
            rf"(?P<character>{ACTOR_NAME})从(?:公告|通告|记录|日志|档案)中(?:得知|获知|知道了)(?P<fact>[^，,。；;]{{2,48}})",
        ),
    ]
    for source_type, pattern in source_patterns:
        source = re.search(pattern, text)
        if not source:
            continue
        rows.append(
            _directive(
                "knows",
                {
                    "character": source.group("character"),
                    "fact": _canonical_knowledge_fact(source.group("fact"), text),
                    "time": time_value,
                    "source_type": source_type,
                },
                document,
                line_no,
                text,
            )
        )
        break
    return rows


def _dedupe_semantic_records(directives: list[ParsedDirective]) -> list[ParsedDirective]:
    """Merge item/use duplicates that differ only by optional time wording."""
    result: list[ParsedDirective] = []
    indexes: dict[tuple, int] = {}
    for directive in directives:
        evidence = directive.evidence
        if directive.kind == "uses":
            key = (
                "uses",
                _clean(directive.attrs.get("item", "")),
                _clean(directive.attrs.get("user", "")),
                evidence.document_id,
                evidence.line_start,
                evidence.line_end,
            )
        elif directive.kind == "item":
            key = (
                "item",
                _clean(directive.attrs.get("item", "")),
                _clean(directive.attrs.get("owner", "")),
                evidence.document_id,
                evidence.line_start,
                evidence.line_end,
            )
        elif directive.kind in {"knows", "claims_knows"}:
            canonical_fact = _canonical_knowledge_fact(directive.attrs.get("fact", ""))
            character = _clean(directive.attrs.get("character", ""))
            base = (
                directive.kind,
                canonical_fact,
                evidence.document_id,
                evidence.line_start,
                evidence.line_end,
            )
            related_key = next(
                (
                    existing_key
                    for existing_key in indexes
                    if existing_key[:5] == base
                    and (
                        character.startswith(existing_key[5])
                        or existing_key[5].startswith(character)
                    )
                ),
                None,
            )
            key = related_key or (*base, character)
        else:
            key = _fingerprint(directive)
        if key not in indexes:
            indexes[key] = len(result)
            result.append(directive)
            continue
        existing_index = indexes[key]
        existing = result[existing_index]
        if directive.kind in {"knows", "claims_knows"}:
            new_character = directive.attrs.get("character", "")
            old_character = existing.attrs.get("character", "")
            new_fact = directive.attrs.get("fact", "")
            old_fact = existing.attrs.get("fact", "")
            if (
                len(new_character) < len(old_character)
                or (
                    new_fact == _canonical_knowledge_fact(new_fact)
                    and old_fact != _canonical_knowledge_fact(old_fact)
                )
            ):
                result[existing_index] = directive
            continue
        if not existing.attrs.get("time") and directive.attrs.get("time"):
            result[existing_index] = directive
    return result


class DeterministicCandidateNormalizer:
    """Adds conservative canonical records derived only from cited source text."""

    def __init__(self, enable_state_modeling: bool = True) -> None:
        self.enable_state_modeling = enable_state_modeling

    def enrich(
        self, documents: list[DocumentInput], directives: list[ParsedDirective]
    ) -> NormalizationResult:
        result = NormalizationResult(directives=list(directives))
        seen = {_fingerprint(item) for item in result.directives}

        for document in documents:
            last_actor: str | None = None
            for line_no, source_line in enumerate(document.content.splitlines(), start=1):
                text = source_line.strip()
                if not text or text.startswith("#"):
                    continue
                actor = _explicit_actor(text)
                if actor:
                    last_actor = actor
                mobility_permission = _mobility_permission_candidate(document, line_no, text) if self.enable_state_modeling else None
                rule_exception = _rule_exception_candidate(document, line_no, text) if self.enable_state_modeling else None
                for candidate in (
                    _item_owner_candidate(document, line_no, text),
                    _item_use_candidate(document, line_no, text, last_actor),
                    _rule_candidate(document, line_no, text),
                    None if rule_exception else _assertion_candidate(document, line_no, text, last_actor),
                    _body_state_candidate(document, line_no, text, last_actor),
                    mobility_permission,
                    rule_exception,
                ):
                    if candidate is None:
                        continue
                    fingerprint = _fingerprint(candidate)
                    if fingerprint not in seen:
                        result.directives.append(candidate)
                        seen.add(fingerprint)
                for candidate in _knowledge_candidates(document, line_no, text, last_actor):
                    if not self.enable_state_modeling and candidate.attrs.get("source_type"):
                        continue
                    fingerprint = _fingerprint(candidate)
                    if fingerprint not in seen:
                        result.directives.append(candidate)
                        seen.add(fingerprint)
        result.directives = _dedupe_semantic_records(result.directives)
        return result
