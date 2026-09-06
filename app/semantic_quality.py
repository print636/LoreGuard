from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import TYPE_CHECKING

from .domain import (
    CertaintyLevel,
    EvidenceMedium,
    EvidenceSpan,
    ParsedDirective,
    SemanticModality,
    SourceScope,
)

if TYPE_CHECKING:
    from .pipeline import DocumentInput


CANONICAL_KINDS = {
    "fact",
    "event",
    "knows",
    "claims_knows",
    "item",
    "uses",
    "world_rule",
    "world_assert",
}

NON_CANONICAL_KINDS = {
    "open_question",
    "tentative_fact",
    "character_claim",
    "negative_statement",
    "clarification",
}

SEMANTIC_KINDS = CANONICAL_KINDS | NON_CANONICAL_KINDS | {"entity"}

_QUESTION_MARKERS = re.compile(
    r"[?？]|是否|能否|可否|会不会|是不是|有没有|究竟|为何|为什么|怎么可能|"
    r"(?:是|会|将|应该).{0,28}还是"
)
_RHETORICAL_MARKERS = re.compile(r"难道|岂(?:不|是)|怎么可能|何尝|莫非")
_HYPOTHETICAL_MARKERS = re.compile(r"(?:^|[，,；;。])\s*(?:如果|假如|倘若|若(?:是)?|假设|设想|要是)")
_UNCERTAIN_MARKERS = re.compile(
    r"(?:并)?不是不可能|并非不可能|绝非不可能|未必不可能|"
    r"没有什么不可能|没什么不可能|不可能不|不无可能|不太可能|"
    r"(?<!不)可能|也许|或许|大概|似乎|看起来|据说|传闻|未必|不一定|或将"
)
_NEGATION_MARKERS = re.compile(r"不是|并非|不为|从未|未曾|没有|并没有|绝非")
_DEFINITE_IMPOSSIBILITY_MARKERS = re.compile(r"(?<!未必)(?<!并非)(?<!不是)不可能|绝不可能|毫无可能")
_DETERMINISTIC_RULE_MARKERS = re.compile(
    r"必然|必须|不得|不能|禁止|一律|只能|失效|无效|无法|则|就|将|会"
)
_DIALOGUE_MARKERS = re.compile(
    r"(?:说|说道|声称|宣称|回答|反驳|喊道|问(?:道|[^：:]{0,12})|告诉[^，。；]{0,12})[：,:]?[“\"]|"
    r"[”\"](?:，|,)?(?:[^，。；]{0,10})?(?:说|说道|声称|宣称|回答|反驳|喊道|问道)|"
    r"(?:说|声称|宣称|回答|反驳|告诉|表示)[，,:：]\s*"
)
_QUOTED_PROPOSITION = r"(?:是|为|在|有|没有|已经|仍然|仍|从未|不得|必须|不能)"
_ACTUAL_QUOTE_MARKERS = re.compile(
    rf"“(?=[^”\n]{{0,80}}{_QUOTED_PROPOSITION})[^”\n]{{4,}}”|"
    rf'\"(?=[^\"\n]{{0,80}}{_QUOTED_PROPOSITION})[^\"\n]{{4,}}\"'
)
_VERIFIED_RECORD_MARKERS = re.compile(
    r"(?P<medium>航行日志|巡逻日志|日志|巡逻记录|审查记录|调度记录|记录|报告|值守簿)"
    r"(?:显示|表明|确认|证实|记载)"
)
_UNVERIFIED_SOURCE_MARKERS = re.compile(
    r"匿名(?:来源|(?:[^，。；:：]{0,8})?(?:信|来信|记录|日志|报告))?|"
    r"来源不明|未经(?:核实|核验|验证|证实)|"
    r"未(?:核实|核验|验证)|真伪未(?:知|验证|核实)|疑似(?:伪造|造假|篡改)|"
    r"(?:记录|日志|报告|档案|时间戳|时钟)(?:已?(?:被|遭)|已)?(?:伪造|造假|篡改|改写)|"
    r"(?:伪造|造假|篡改)的(?:记录|日志|报告|档案|时间戳)"
)
_QUOTED_SOURCE_MARKERS = re.compile(
    r"(?:设定(?:稿)?|档案|日志|记录|报告|手册|碑文|旧卷|卷宗|卷册|"
    r"书信|信件|札记|手稿|文书|传记|报道|资料)(?:中|称|写道|写明|记载)|"
    r"(?:设定(?:稿)?|档案|手册|碑文|旧卷|卷宗|卷册|书信|信件|札记|"
    r"手稿|文书|传记|报道|资料)(?:只|仅)?记载"
)
_REPORTED_SOURCE_MARKERS = re.compile(
    r"据[^，。；\n]{0,20}(?:所述|说法|声称|转述|口述)|"
    r"(?:转述|口述|传话)(?:中|称|提到|表示)"
)

# These expressions describe narrative frames, not isolated keywords.  In
# particular, a sentence that says an event is *not* a memory or illusion must
# remain eligible, while a section heading that establishes a dream can govern
# a following evidence line which contains no frame word of its own.
_NON_REALITY_DIRECT_MARKERS = re.compile(
    r"梦(?:里|中)|梦境(?:里|中)|(?:在|于)?(?:一段)?回忆(?:里|中)|"
    r"(?:在|于)?(?:幻象|幻觉|想象|假想)(?:里|中)|"
    r"(?:通讯)?投影(?:里|中|画面中)|"
    r"(?:梦境|回忆|幻象|幻觉|想象|假想|投影)(?:片段|段落|场景|画面)"
)
_NON_REALITY_DECLARATION_MARKERS = re.compile(
    r"(?:段落|片段|场景|画面|章节|下文|以下|本段|本节|镜中段落)"
    r"[^。；\n]{0,20}(?:均|都)?(?:为|属于|是|被标注为)"
    r"[^。；\n]{0,6}(?:梦境|回忆|幻象|幻觉|想象|假想|投影)|"
    r"(?:只是|均为|属于|被标注为)[^。；\n]{0,6}"
    r"(?:梦境|回忆|幻象|幻觉|想象|假想|投影)"
)
_NON_REALITY_CONTEXT_MARKERS = re.compile(
    r"(?:作者|编辑|旁白|叙事)?(?:旁注|注释|说明|标注|提示)[：:]"
    r"[^。；\n]{0,40}(?:下文|以下|本段|本节|镜中段落|该段|这些段落)"
    r"[^。；\n]{0,20}(?:梦境|回忆|幻象|幻觉|想象|假想|投影)|"
    r"^[\s#【\[]*(?:梦境|回忆|幻象|幻觉|想象|假想|投影)"
    r"(?:片段|段落|场景|章节)?[】\]:：\s]*$",
    re.MULTILINE,
)
_REALITY_CONFIRMATION_MARKERS = re.compile(
    r"(?:不是|并非|绝非|不属于|排除|明确(?:并)?非)"
    r"[^。；\n]{0,36}(?:回忆|梦境|幻象|幻觉|想象|假想|(?:通讯)?投影)"
)
_REALITY_RESUMPTION_MARKERS = re.compile(
    r"(?:从|自)(?:梦|回忆|幻象|幻觉|想象|假想)(?:里|中)"
    r"(?:醒来|走出|脱离|回到现实)(?:后|之后)?|"
    r"梦醒(?:后|之后)|(?:回忆|幻象|幻觉|想象|投影)(?:结束|消散|终止)"
    r"(?:后|之后)|回到现实(?:后|中)?"
)

_NON_AUTHORITATIVE_SCOPES = {
    SourceScope.character_dialogue,
    SourceScope.quoted_material,
    SourceScope.unverified_report,
    SourceScope.unknown,
}


def evidence_has_noncanonical_frame(text: str) -> bool:
    """Return whether text positively places its proposition outside reality.

    Reality confirmations and explicit transitions are removed before the
    positive frame scan.  Other mixed-frame evidence remains conservative: a
    caller should cite the smaller reality-only span if it wants promotion.
    """

    adversative_tails = re.findall(r"(?:而是|却是|其实是)([^。；\n]*)", text)
    if any(
        _NON_REALITY_DIRECT_MARKERS.search(tail)
        or _NON_REALITY_DECLARATION_MARKERS.search(tail)
        for tail in adversative_tails
    ):
        return True
    cleaned = _REALITY_CONFIRMATION_MARKERS.sub("", text)
    cleaned = _REALITY_RESUMPTION_MARKERS.sub("回到现实", cleaned)
    return bool(
        _NON_REALITY_DIRECT_MARKERS.search(cleaned)
        or _NON_REALITY_DECLARATION_MARKERS.search(cleaned)
    )


def document_context_has_noncanonical_frame(
    content: str, line_start: int, line_end: int
) -> bool:
    """Check cited evidence plus a narrowly scoped preceding frame marker."""

    lines = content.splitlines()
    if not (1 <= line_start <= line_end <= len(lines)):
        return False
    evidence = "\n".join(lines[line_start - 1 : line_end])
    if evidence_has_noncanonical_frame(evidence):
        return True
    # A line that explicitly resumes reality overrides a preceding dream or
    # recollection heading.  Only explicit annotation/heading syntax from the
    # previous two lines can otherwise establish cross-line scope.
    if _REALITY_RESUMPTION_MARKERS.search(evidence):
        return False
    preceding = "\n".join(lines[max(0, line_start - 3) : line_start - 1])
    preceding = _REALITY_CONFIRMATION_MARKERS.sub("", preceding)
    return bool(_NON_REALITY_CONTEXT_MARKERS.search(preceding))

_REQUIRED_ATTRS: dict[str, tuple[str, ...]] = {
    "fact": ("subject", "predicate", "value"),
    "event": ("time", "location", "participants"),
    "knows": ("character", "fact", "time"),
    "claims_knows": ("character", "fact", "time"),
    "item": ("item", "owner"),
    "uses": ("item", "user"),
    "world_rule": ("key", "value"),
    "world_assert": ("key", "value"),
    "open_question": ("question",),
    "clarification": ("summary", "category"),
}

_ENTITY_FIELDS = {"subject", "character", "owner", "user", "actor"}
_TITLE = (
    r"(?:船长|队长|站长|馆长|校长|局长|族长|组长|班长|店长|院长|"
    r"主任|医师|医生|护士|警官|教授|博士|导师|老师|守卫|侍卫|祭司|"
    r"领航员|档案官|工程师|调查员|管理员|书记员|审判官|指挥官|负责人)"
)
_PERSON_NAME = r"[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z·_-]{1,11}?"


def _title_declarations(text: str) -> list[tuple[str, str]]:
    """Return only syntactically explicit title-to-name bindings."""
    patterns = [
        rf"(?P<name>{_PERSON_NAME})[（(](?P<title>{_TITLE})[）)]",
        rf"(?P<title>{_TITLE})(?:名叫|名为|叫作|叫)(?P<name>{_PERSON_NAME})",
        rf"(?P<name>{_PERSON_NAME})(?:是|担任|出任|作为)(?:了)?(?P<title>{_TITLE})",
        rf"(?:^|[，,。；;：:\s])(?P<title>{_TITLE})"
        rf"(?P<name>(?!(?:名叫|名为|叫作|在|于|说|表示|确认|宣布|负责)){_PERSON_NAME})"
        rf"(?=在|于|说|表示|确认|宣布|负责|$|[，,。；;])",
    ]
    rows: list[tuple[str, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            title = _clean(match.group("title"))
            name = _clean(match.group("name"))
            if title and name and title != name:
                rows.append((title, name))
    return rows


def canonicalize_evidenced_titles(
    documents: list[DocumentInput], directives: list[ParsedDirective]
) -> tuple[list[ParsedDirective], list[dict], list[str]]:
    """Normalize a title only when the source explicitly binds it to one name.

    Global declarations apply to every branch. Branch declarations stay in
    their own branch, and ambiguous bindings are intentionally left unchanged.
    This is exact-value normalization; it never uses substring similarity.
    """
    declarations: dict[tuple[str, str], set[str]] = {}
    for document in documents:
        scope = document.scope or "global"
        for title, name in _title_declarations(document.content):
            declarations.setdefault((scope, title), set()).add(name)

    traces: list[dict] = []
    warnings: list[str] = []
    ambiguous = {
        (scope, title)
        for (scope, title), names in declarations.items()
        if len(names) != 1
    }
    for scope, title in sorted(ambiguous):
        warnings.append(f"作用域 {scope} 中称谓“{title}”对应多个姓名，已保持原称谓")

    def resolve(scope: str, value: str) -> str | None:
        title = value
        candidates: set[str] = set()
        for key in (("global", title), (scope, title)):
            if key in ambiguous:
                return None
            candidates.update(declarations.get(key, set()))
        if len(candidates) == 1:
            return next(iter(candidates))
        # A model may preserve the explicit surface form “船长顾青” while
        # another record uses “顾青”. Normalize that exact evidenced composite
        # as well, but never infer from two separated substrings.
        composite_candidates: set[str] = set()
        visible_scopes = {"global", scope}
        for (decl_scope, decl_title), names in declarations.items():
            if decl_scope not in visible_scopes or (decl_scope, decl_title) in ambiguous:
                continue
            for name in names:
                if value in {f"{decl_title}{name}", f"{name}{decl_title}"}:
                    composite_candidates.add(name)
        return next(iter(composite_candidates)) if len(composite_candidates) == 1 else None

    normalized: list[ParsedDirective] = []
    for directive in directives:
        attrs = dict(directive.attrs)
        scope = attrs.get("story_scope", "") or "global"
        for field_name in _ENTITY_FIELDS:
            original = attrs.get(field_name, "")
            canonical = resolve(scope, original)
            if not canonical:
                continue
            attrs[field_name] = canonical
            traces.append(
                {
                    "alias": original,
                    "canonical": canonical,
                    "kind": directive.kind,
                    "field": field_name,
                    "document_name": directive.evidence.document_name,
                    "line": directive.evidence.line_start,
                    "reason": "explicit_name_title_binding",
                }
            )
        participants = attrs.get("participants", "")
        if participants:
            values = [value.strip() for value in re.split(r"[,，]", participants) if value.strip()]
            replaced: list[str] = []
            for original in values:
                canonical = resolve(scope, original) or original
                replaced.append(canonical)
                if canonical != original:
                    traces.append(
                        {
                            "alias": original,
                            "canonical": canonical,
                            "kind": directive.kind,
                            "field": "participants",
                            "document_name": directive.evidence.document_name,
                            "line": directive.evidence.line_start,
                            "reason": "explicit_name_title_binding",
                        }
                    )
            attrs["participants"] = ",".join(replaced)
        normalized.append(directive.model_copy(update={"attrs": attrs}))
    return normalized, traces, warnings


@dataclass(slots=True)
class SemanticQualityResult:
    directives: list[ParsedDirective] = field(default_factory=list)
    rejected_count: int = 0
    transformed_count: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected_count += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def transform(self, reason: str) -> None:
        self.transformed_count += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    def warning(self) -> str | None:
        if not self.rejected_count and not self.transformed_count:
            return None
        details = "、".join(f"{key}={value}" for key, value in sorted(self.reasons.items()))
        return (
            "语义质量门已处理低可信候选："
            f"拒绝 {self.rejected_count} 条，转为非确定记录 {self.transformed_count} 条"
            f"（{details}）"
        )


def _clean(value: str) -> str:
    return re.sub(r"\s+", "", str(value)).strip("，,。；;：:\"'“”‘’")


def _semantic_units(text: str) -> list[str]:
    """Split prose while retaining punctuation that carries modality."""
    units = [part.strip() for part in re.findall(r"[^。！？?!；;]+[。！？?!；;]?", text)]
    return [part for part in units if part]


def _question_units(text: str) -> list[str]:
    questions: list[str] = []
    for unit in _semantic_units(text):
        if not _QUESTION_MARKERS.search(unit):
            continue
        clauses = [part.strip() for part in re.split(r"(?<=[，,])", unit) if part.strip()]
        first = next(
            (index for index, clause in enumerate(clauses) if _QUESTION_MARKERS.search(clause)),
            0,
        )
        # Preserve an opening condition because it defines what the question is
        # asking.  Do not drag an unrelated assertion before “but/whether” into
        # the clarification candidate.
        preceding_condition = [
            index
            for index, clause in enumerate(clauses[:first])
            if _HYPOTHETICAL_MARKERS.search(clause)
        ]
        if preceding_condition:
            first = preceding_condition[-1]
        questions.append("".join(clauses[first:]).strip())
    return questions


def open_question_directives(evidence: EvidenceSpan) -> list[ParsedDirective]:
    rows: list[ParsedDirective] = []
    for question in _question_units(evidence.text):
        question_type = "rhetorical" if _RHETORICAL_MARKERS.search(question) else "open"
        if _HYPOTHETICAL_MARKERS.search(question):
            question_type = "hypothetical"
        scope = (
            SourceScope.character_dialogue
            if _DIALOGUE_MARKERS.search(question) and not re.search(r"编辑(?:问|批注)", question)
            else SourceScope.narrator
        )
        rows.append(
            ParsedDirective(
                kind="open_question",
                attrs={
                    "question": question.strip(),
                    "question_type": question_type,
                    "modality": SemanticModality.interrogative.value,
                    "source_scope": scope.value,
                    "certainty": CertaintyLevel.unknown.value,
                },
                evidence=evidence.model_copy(update={"text": question.strip()}),
            )
        )
    return rows


def _anchors(directive: ParsedDirective) -> list[str]:
    attrs = directive.attrs
    keys = {
        "fact": ("subject", "predicate", "value"),
        "tentative_fact": ("subject", "predicate", "value"),
        "character_claim": ("subject", "predicate", "value"),
        "event": ("location", "participants"),
        "knows": ("character", "fact"),
        "claims_knows": ("character", "fact"),
        "item": ("item", "owner"),
        "uses": ("item", "user"),
        "world_rule": ("key", "value", "condition", "consequence"),
        "world_assert": ("key", "value", "actor"),
        "open_question": ("question",),
    }.get(directive.kind, ())
    return [_clean(attrs.get(key, "")) for key in keys if _clean(attrs.get(key, ""))]


def _support_text(directive: ParsedDirective) -> str:
    text = directive.evidence.text.strip()
    anchors = _anchors(directive)
    if not anchors:
        return text
    # A model may cite a whole multi-sentence line.  Judge the sentence that
    # actually supports its fields instead of inheriting modality from an
    # unrelated sentence. Commas are deliberately retained: an opening
    # “if” clause governs the consequence after the comma.
    units = _semantic_units(text)
    unit_scores = [
        (sum(anchor in _clean(unit) for anchor in anchors), unit) for unit in units
    ]
    best_unit_score = max((score for score, _ in unit_scores), default=0)
    best_units = [unit for score, unit in unit_scores if score == best_unit_score and score]
    if not best_units:
        return text
    unit = best_units[0]
    clauses = [part.strip() for part in re.split(r"(?<=[，,])", unit) if part.strip()]
    scored = [
        (sum(anchor in _clean(clause) for anchor in anchors), index, clause)
        for index, clause in enumerate(clauses)
    ]
    best_score = max((score for score, _, _ in scored), default=0)
    if best_score:
        selected = [index for score, index, _ in scored if score == best_score]
        first, last = min(selected), max(selected)
        if first and _HYPOTHETICAL_MARKERS.search(clauses[first - 1]):
            first -= 1
        return "".join(clauses[first : last + 1])
    return unit


def _source_scope(text: str, directive: ParsedDirective) -> SourceScope:
    full_evidence = directive.evidence.text
    if _UNVERIFIED_SOURCE_MARKERS.search(full_evidence) or re.search(
        r"信中声称", full_evidence
    ):
        return SourceScope.unverified_report
    # Attribution often follows a quoted sentence. `_support_text` may select
    # only the proposition before the full stop, so source authority must be
    # classified from the complete cited evidence rather than that slice.
    if _DIALOGUE_MARKERS.search(full_evidence) or _REPORTED_SOURCE_MARKERS.search(
        full_evidence
    ):
        return SourceScope.character_dialogue
    if _ACTUAL_QUOTE_MARKERS.search(full_evidence):
        return SourceScope.quoted_material
    if _QUOTED_SOURCE_MARKERS.search(full_evidence):
        return SourceScope.quoted_material
    if directive.kind == "world_rule" or re.search(
        r"根据(?:世界观)?规则[^：:]{0,12}[：:]", text
    ):
        return SourceScope.world_rule
    if _VERIFIED_RECORD_MARKERS.search(text):
        # The narrative adopts the record's stated observation as evidence;
        # this is not the same as quoting an embedded proposition.
        return SourceScope.narrator
    return SourceScope.narrator


def _evidence_medium(text: str) -> EvidenceMedium:
    match = _VERIFIED_RECORD_MARKERS.search(text)
    if match:
        token = match.group("medium")
        if "日志" in token:
            return EvidenceMedium.log
        if "报告" in token:
            return EvidenceMedium.report
        return EvidenceMedium.record
    if re.search(r"亲眼|目击|当场看见|当场看到|直接观察", text):
        return EvidenceMedium.direct_observation
    return EvidenceMedium.unspecified


def _declared_enum(attrs: dict[str, str], key: str, enum_type):
    if key not in attrs:
        return None
    try:
        return enum_type(attrs[key])
    except (TypeError, ValueError):
        return None


def _closed_conditional_rule(text: str, directive: ParsedDirective) -> bool:
    if directive.kind != "world_rule" or not _HYPOTHETICAL_MARKERS.search(text):
        return False
    return bool(
        not _QUESTION_MARKERS.search(text)
        and not _UNCERTAIN_MARKERS.search(text)
        and _DETERMINISTIC_RULE_MARKERS.search(text)
    )


def _closed_baseline_semantics(
    support: str, directive: ParsedDirective
) -> tuple[SemanticModality, SourceScope, CertaintyLevel] | None:
    """Infer labels only for syntax already recognized by closed baseline rules.

    This deliberately does not inspect provenance. Merely calling a row
    ``baseline`` can never make it eligible, and structured directives with a
    short free-form evidence label are excluded.
    """
    if directive.attrs.get("input_form") == "directive":
        return None
    attrs = directive.attrs
    compact = _clean(support)
    scope = _source_scope(support, directive)
    kind = directive.kind
    closed = False
    if kind == "fact":
        subject = _clean(attrs.get("subject", ""))
        value = _clean(attrs.get("value", ""))
        predicate = attrs.get("predicate", "")
        closed = bool(
            subject
            and value
            and (subject in compact or subject == "*")
            and (
                value in compact
                or predicate in {
                    "mobility_permission",
                    "mobility_limit",
                    "rule_exception",
                    "authorization",
                }
            )
            and re.search(
                r"(?:是(?!否)|为|属于|担任|获得|获准|允许|许可|"
                r"没有|并未|未曾|不可能|无法|不能)",
                support,
            )
        )
    elif kind == "event":
        closed = bool(
            re.search(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", support)
            and _clean(attrs.get("location", "")) in compact
            and any(
                _clean(part) in compact
                for part in re.split(r"[,，]", attrs.get("participants", ""))
                if _clean(part)
            )
        )
    elif kind in {"knows", "claims_knows"}:
        closed = bool(re.search(r"得知|获知|知道|说出|提到|引用", support))
    elif kind == "item":
        closed = bool(re.search(r"获得|持有|保管|掌管|交给|移交|归还|接收", support))
    elif kind == "uses":
        closed = bool(re.search(r"使用(?!权)|用(?!权)|取出|拿出|按下|插入|启用", support))
    elif kind == "world_rule":
        closed = bool(_DETERMINISTIC_RULE_MARKERS.search(support))
    elif kind == "world_assert":
        closed = bool(re.search(r"驱动|启动|开启|发动|进入|执行|使用", support))
    if not closed:
        return None
    if kind == "claims_knows":
        return (
            SemanticModality.reported,
            SourceScope.character_dialogue,
            CertaintyLevel.certain,
        )
    modality = (
        SemanticModality.negated
        if _NEGATION_MARKERS.search(support)
        or _DEFINITE_IMPOSSIBILITY_MARKERS.search(support)
        else SemanticModality.conditional_rule
        if _closed_conditional_rule(support, directive)
        else SemanticModality.asserted
    )
    return modality, scope, CertaintyLevel.certain


def _to_open_question(directive: ParsedDirective) -> ParsedDirective:
    attrs = directive.attrs
    questions = _question_units(directive.evidence.text)
    declared = attrs.get("question", "").strip()
    question = declared if declared and declared in directive.evidence.text else ""
    if not question:
        question = questions[0] if questions else directive.evidence.text.strip()
    question_type = "rhetorical" if _RHETORICAL_MARKERS.search(question) else "open"
    if _HYPOTHETICAL_MARKERS.search(question):
        question_type = "hypothetical"
    safe_metadata = {
        key: attrs[key]
        for key in (
            "repair_status",
            "document_role",
            "story_scope",
            "evidence_medium",
        )
        if key in attrs
    }
    return directive.model_copy(
        update={
            "kind": "open_question",
            "attrs": {
                "question": question,
                "question_type": question_type,
                "modality": SemanticModality.interrogative.value,
                "source_scope": _source_scope(question, directive).value,
                "certainty": CertaintyLevel.unknown.value,
                **safe_metadata,
            },
        }
    )


def _to_noncanonical(
    directive: ParsedDirective,
    *,
    kind: str,
    modality: SemanticModality,
    source_scope: SourceScope,
    certainty: CertaintyLevel,
) -> ParsedDirective:
    attrs = {
        **directive.attrs,
        "original_kind": directive.kind,
        "modality": modality.value,
        "source_scope": source_scope.value,
        "certainty": certainty.value,
        "evidence_medium": directive.attrs.get(
            "evidence_medium", _evidence_medium(directive.evidence.text).value
        ),
    }
    return directive.model_copy(update={"kind": kind, "attrs": attrs})


def assess_directive(directive: ParsedDirective) -> tuple[ParsedDirective | None, str | None]:
    """Apply evidence-aware semantic safety checks to one candidate.

    This gate does not decide whether a statement is true.  It only decides
    whether the source presents the candidate strongly enough for deterministic
    consistency rules.  Evidence modality always overrides a model's optimistic
    self-label.
    """
    if directive.kind not in SEMANTIC_KINDS:
        return None, "unsupported_kind"
    if not directive.evidence.text.strip():
        return None, "empty_evidence"

    attrs = dict(directive.attrs)
    for key in _REQUIRED_ATTRS.get(directive.kind, ()):
        if not _clean(attrs.get(key, "")):
            return None, "missing_required_field"

    missing_labels = {
        key for key in ("modality", "source_scope", "certainty") if key not in attrs
    }
    declared_modality = _declared_enum(attrs, "modality", SemanticModality)
    declared_scope = _declared_enum(attrs, "source_scope", SourceScope)
    declared_certainty = _declared_enum(attrs, "certainty", CertaintyLevel)
    if (
        ("modality" in attrs and declared_modality is None)
        or ("source_scope" in attrs and declared_scope is None)
        or ("certainty" in attrs and declared_certainty is None)
    ):
        return None, "invalid_semantic_enum"
    support = _support_text(directive)
    try:
        declared_medium = EvidenceMedium(
            attrs.get("evidence_medium", EvidenceMedium.unspecified.value)
        )
    except (TypeError, ValueError):
        declared_medium = EvidenceMedium.unspecified
    derived_medium = _evidence_medium(support)
    attrs["evidence_medium"] = (
        derived_medium
        if derived_medium != EvidenceMedium.unspecified
        else declared_medium
    ).value
    directive = directive.model_copy(update={"attrs": attrs})
    if missing_labels:
        if directive.kind == "open_question" or _QUESTION_MARKERS.search(support):
            return _to_open_question(directive), "missing_semantic_labels"
        if directive.kind == "clarification":
            attrs.update(
                modality=SemanticModality.uncertain.value,
                source_scope=SourceScope.unknown.value,
                certainty=CertaintyLevel.unknown.value,
            )
            return directive.model_copy(update={"attrs": attrs}), "missing_semantic_labels"
        if _HYPOTHETICAL_MARKERS.search(support):
            inferred = (
                SemanticModality.hypothetical,
                SourceScope.unknown,
                CertaintyLevel.unknown,
            )
        elif _UNCERTAIN_MARKERS.search(support):
            inferred = (
                SemanticModality.uncertain,
                SourceScope.unknown,
                CertaintyLevel.unknown,
            )
        elif _DIALOGUE_MARKERS.search(support):
            inferred = (
                SemanticModality.reported,
                _source_scope(support, directive),
                CertaintyLevel.unknown,
            )
        else:
            inferred = _closed_baseline_semantics(
                directive.evidence.text, directive
            )
        if inferred is None:
            attrs.update(
                original_kind=directive.kind,
                modality=SemanticModality.uncertain.value,
                source_scope=SourceScope.unknown.value,
                certainty=CertaintyLevel.unknown.value,
            )
            return (
                directive.model_copy(update={"kind": "tentative_fact", "attrs": attrs}),
                "missing_semantic_labels",
            )
        inferred_modality, inferred_scope, inferred_certainty = inferred
        declared_modality = declared_modality or inferred_modality
        declared_scope = declared_scope or inferred_scope
        declared_certainty = declared_certainty or inferred_certainty
        attrs.update(
            modality=declared_modality.value,
            source_scope=declared_scope.value,
            certainty=declared_certainty.value,
        )
    if declared_modality is None or declared_scope is None or declared_certainty is None:
        return None, "invalid_semantic_enum"

    support = _support_text(directive)
    if directive.kind == "clarification":
        attrs.update(
            modality=SemanticModality.uncertain.value,
            source_scope=_source_scope(support, directive).value,
            certainty=CertaintyLevel.unknown.value,
        )
        return directive.model_copy(update={"attrs": attrs}), None
    if directive.kind == "open_question":
        return _to_open_question(directive), None
    if _QUESTION_MARKERS.search(support):
        return _to_open_question(directive), "interrogative"

    if directive.noncanonical_frame and directive.kind in CANONICAL_KINDS:
        return (
            _to_noncanonical(
                directive,
                kind="tentative_fact",
                modality=SemanticModality.hypothetical,
                source_scope=SourceScope.unknown,
                certainty=CertaintyLevel.possible,
            ),
            "noncanonical_frame",
        )

    scope = _source_scope(support, directive)
    # Explicit structured directives are authoritative only when their evidence
    # has no source-risk marker. An author contract cannot launder an anonymous
    # report, dialogue, or quoted proposition into narrator/world truth.
    structured = attrs.get("input_form") == "directive"
    if structured and scope not in _NON_AUTHORITATIVE_SCOPES:
        scope = declared_scope
    elif declared_scope in _NON_AUTHORITATIVE_SCOPES:
        # Never promote a model's conservative source classification merely
        # because a short evidence span lacks an obvious speech marker.
        scope = declared_scope
    if (
        attrs.get("document_role") == "canon"
        and scope not in _NON_AUTHORITATIVE_SCOPES
        and declared_modality != SemanticModality.reported
    ):
        # A server-declared canon document is an authoritative source even for
        # item ownership or ordinary setting facts.  The current enum uses
        # ``world_rule`` for that authority tier; document_role remains the
        # more precise provenance field.
        scope = SourceScope.world_rule

    if directive.kind == "tentative_fact":
        modality = (
            declared_modality
            if declared_modality in {SemanticModality.hypothetical, SemanticModality.uncertain}
            else SemanticModality.uncertain
        )
        attrs.update(
            modality=modality.value,
            source_scope=scope.value,
            certainty=(
                declared_certainty.value
                if declared_certainty != CertaintyLevel.certain
                else CertaintyLevel.possible.value
            ),
        )
        return directive.model_copy(update={"attrs": attrs}), None
    if directive.kind == "character_claim":
        claim_scope = (
            scope
            if scope
            in {
                SourceScope.character_dialogue,
                SourceScope.quoted_material,
                SourceScope.unverified_report,
                SourceScope.unknown,
            }
            else SourceScope.character_dialogue
        )
        attrs.update(
            modality=SemanticModality.reported.value,
            source_scope=claim_scope.value,
            certainty=CertaintyLevel.unknown.value,
        )
        return directive.model_copy(update={"attrs": attrs}), None
    if directive.kind == "negative_statement":
        attrs.update(
            modality=SemanticModality.negated.value,
            source_scope=scope.value,
            certainty=CertaintyLevel.certain.value,
        )
        return directive.model_copy(update={"attrs": attrs}), None
    if (
        _closed_conditional_rule(support, directive)
        and scope not in _NON_AUTHORITATIVE_SCOPES
    ):
        attrs.update(
            modality=SemanticModality.conditional_rule.value,
            source_scope=SourceScope.world_rule.value,
            certainty=CertaintyLevel.certain.value,
            polarity="affirmative",
        )
        return directive.model_copy(update={"attrs": attrs}), None

    if _HYPOTHETICAL_MARKERS.search(support) or declared_modality == SemanticModality.hypothetical:
        return (
            _to_noncanonical(
                directive,
                kind="tentative_fact",
                modality=SemanticModality.hypothetical,
                source_scope=scope,
                certainty=CertaintyLevel.possible,
            ),
            "hypothetical",
        )

    if _UNCERTAIN_MARKERS.search(support) or declared_modality == SemanticModality.uncertain:
        return (
            _to_noncanonical(
                directive,
                kind="tentative_fact",
                modality=SemanticModality.uncertain,
                source_scope=scope,
                certainty=(
                    declared_certainty
                    if declared_certainty != CertaintyLevel.certain
                    else CertaintyLevel.possible
                ),
            ),
            "uncertain",
        )

    if scope == SourceScope.unknown:
        return (
            _to_noncanonical(
                directive,
                kind="tentative_fact",
                modality=SemanticModality.uncertain,
                source_scope=SourceScope.unknown,
                certainty=CertaintyLevel.unknown,
            ),
            "unknown_source",
        )

    if (
        scope in {
            SourceScope.character_dialogue,
            SourceScope.unverified_report,
        }
        or declared_modality == SemanticModality.reported
    ) and directive.kind != "claims_knows":
        return (
            _to_noncanonical(
                directive,
                kind="character_claim",
                modality=SemanticModality.reported,
                source_scope=scope,
                certainty=CertaintyLevel.unknown,
            ),
            "non_authoritative_source",
        )

    if declared_certainty != CertaintyLevel.certain:
        return (
            _to_noncanonical(
                directive,
                kind="tentative_fact",
                modality=SemanticModality.uncertain,
                source_scope=scope,
                certainty=declared_certainty,
            ),
            "uncertain_certainty",
        )

    negated = bool(
        _NEGATION_MARKERS.search(support)
        or _DEFINITE_IMPOSSIBILITY_MARKERS.search(support)
    )
    if negated and directive.kind not in {"fact", "world_rule"}:
        return (
            _to_noncanonical(
                directive,
                kind="negative_statement",
                modality=SemanticModality.negated,
                source_scope=scope,
                certainty=CertaintyLevel.certain,
            ),
            "negated_non_fact",
        )

    if directive.kind == "fact":
        # These fields must be atomic labels, not swallowed clauses.  This is
        # deliberately independent of who produced the JSON.
        if any(_QUESTION_MARKERS.search(attrs.get(key, "")) for key in ("subject", "predicate", "value")):
            return _to_open_question(directive), "question_inside_fact"
        if any(len(_clean(attrs.get(key, ""))) > limit for key, limit in (("subject", 32), ("predicate", 24), ("value", 96))):
            return None, "non_atomic_fact"
        attrs["polarity"] = "negative" if negated or declared_modality == SemanticModality.negated else "affirmative"

    modality = SemanticModality.negated if negated else declared_modality
    if modality not in {
        SemanticModality.asserted,
        SemanticModality.negated,
        SemanticModality.reported,
        SemanticModality.conditional_rule,
    }:
        return None, "unsafe_modality"
    attrs.update(
        modality=modality.value,
        source_scope=scope.value,
        certainty=declared_certainty.value,
    )
    if directive.kind == "claims_knows":
        # The claim itself is observed even though the proposition spoken by
        # the character is not promoted to narrator truth.
        if scope == SourceScope.character_dialogue:
            attrs.update(
                modality=SemanticModality.reported.value,
                source_scope=SourceScope.character_dialogue.value,
                certainty=CertaintyLevel.certain.value,
            )
        else:
            attrs.update(
                modality=SemanticModality.reported.value,
                source_scope=scope.value,
                certainty=CertaintyLevel.unknown.value,
            )
    return directive.model_copy(update={"attrs": attrs}), None


def apply_semantic_quality_gate(directives: list[ParsedDirective]) -> SemanticQualityResult:
    result = SemanticQualityResult()
    seen: set[tuple] = set()
    for directive in directives:
        assessed, reason = assess_directive(directive)
        if assessed is None:
            result.reject(reason or "rejected")
            continue
        if reason:
            result.transform(reason)
        evidence = assessed.evidence
        fingerprint = (
            assessed.kind,
            tuple(sorted((key, _clean(value)) for key, value in assessed.attrs.items())),
            evidence.document_id,
            evidence.line_start,
            evidence.line_end,
        )
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        result.directives.append(assessed)
    return result


def eligible_for_deterministic_rules(directive: ParsedDirective) -> bool:
    if directive.kind not in CANONICAL_KINDS:
        return False
    if directive.noncanonical_frame:
        return False
    attrs = directive.attrs
    if attrs.get("repair_status") == "failed":
        return False
    if directive.kind == "claims_knows":
        return (
            attrs.get("modality") == SemanticModality.reported.value
            and attrs.get("source_scope") == SourceScope.character_dialogue.value
            and attrs.get("certainty") == CertaintyLevel.certain.value
        )
    if attrs.get("source_scope") in {
        SourceScope.character_dialogue.value,
        SourceScope.quoted_material.value,
        SourceScope.unverified_report.value,
        SourceScope.unknown.value,
    }:
        return False
    if attrs.get("certainty") != CertaintyLevel.certain.value:
        return False
    return attrs.get("modality") in {
        SemanticModality.asserted.value,
        SemanticModality.negated.value,
        SemanticModality.conditional_rule.value,
    }
