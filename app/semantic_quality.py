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
_CONDITION_SCOPE_RESET = re.compile(r"^\s*(?:但|然而|不过|可是|却|只是)")
_UNCERTAIN_MARKERS = re.compile(
    r"并非不是|(?:并)?不是不(?:是|为)?|不为不(?:是|为)?|"
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
    r"(?:用户指南|维护规程|操作说明|使用说明)[：:]|"
    r"(?:设定(?:稿)?|档案|日志|记录|报告|手册|指南|规程|碑文|旧卷|卷宗|卷册|"
    r"书信|信件|札记|手稿|文书|传记|报道|资料)(?:中|称|写道|写明|记载)|"
    r"(?:设定(?:稿)?|档案|手册|指南|规程|碑文|旧卷|卷宗|卷册|书信|信件|札记|"
    r"手稿|文书|传记|报道|资料)(?:只|仅)?记载"
)
_REPORTED_SOURCE_MARKERS = re.compile(
    r"据[^，。；\n]{0,20}(?:所述|说法|声称|转述|口述)|"
    r"(?:转述|口述|传话)(?:中|称|提到|表示)|"
    r"(?:纪要|记录|日志|报告|录音|口供|旁白)?(?:的)?(?:转述|口述|传话)[：:]"
)
_SOURCE_CONTEXT_MARKERS = (
    _UNVERIFIED_SOURCE_MARKERS,
    _DIALOGUE_MARKERS,
    _REPORTED_SOURCE_MARKERS,
    _ACTUAL_QUOTE_MARKERS,
    _QUOTED_SOURCE_MARKERS,
    _VERIFIED_RECORD_MARKERS,
)

# Shared by semantic closure, candidate normalization, and promotion grounding.
# Keep this as a syntax fragment (rather than a compiled expression) so every
# layer recognizes the same concrete item-use verbs.
FOLLOWUP_USE_ACTION_VERB_PATTERN = (
    r"使用|启用|发动|操作|操控|挥动|借助|按下|插入|开启|启动|"
    r"盖下|刷过|(?<!作)用(?!途|处|法|量|时|户|例|品|具|权|意)"
)
USE_ACTION_VERB_PATTERN = (
    rf"{FOLLOWUP_USE_ACTION_VERB_PATTERN}|取出|拿出|掏出|拔出"
)

# An action mentioned as the content of an order, intention or unfinished
# attempt is not evidence that the action happened.  These are syntactic
# frames rather than a bag of keywords: a completed temporal bridge such as
# “接到命令后，某人在…启动…” starts a new, asserted action clause.
_ACTION_VERBS = re.compile(
    rf"驱动|施展|执行|(?:{USE_ACTION_VERB_PATTERN})"
)
_UNREALIZED_ACTION_FRAME = re.compile(
    r"命令|下令|责令|要求|吩咐|嘱咐|敦促|通知|建议|提议|"
    r"准许|准予|容许|批准|答应|授意|允许|授权(?!后)|应当|应该|需要|务必|"
    r"指示(?!灯|器|牌|标|线|针|状态|系统|图)|"
    r"计划|打算|预备|预定|意图|试图|尝试|决定|即将|将要|"
    r"准备(?!室|区|舱|间|站|台|工作|状态|材料|物资)|"
    r"拟定|拟(?=于|在|将|由|启动|开启|执行|使用|发动|施展|启用)|"
    r"将(?:会)?(?=(?:在|于)[^，,。；;！!？?：:]{0,36}$)"
)
_COMPLETED_ORDER_BRIDGE = re.compile(
    r"(?:接令|奉命)(?:后|之后|以后)|"
    r"(?:接到|收到|得到|接受|听从|遵照|按照|依照|执行完?|完成)"
    r"[^，,。；;！!？?：:]{0,20}?(?:命令|指令|要求|通知|指示|安排|决定)"
    r"(?:下达|发布|确认|批准|完成)?(?:后|之后|以后)|"
    r"(?:命令|指令|要求|通知)(?:下达|发布|确认|批准)(?:后|之后|以后)|"
    r"(?:下令|吩咐|嘱咐|决定)(?:后|之后|以后)|"
    r"(?:计划|准备|预备|尝试|安排)(?:已经|已)?(?:完成|结束|取消|中止|失败)"
    r"(?:后|之后|以后)|(?:准备好|准备就绪)(?:后|之后|以后)"
)
_UNREALIZED_HEADING_FRAME = re.compile(
    # A heading is recognized by its shape, not by a list of business topics:
    # it starts a punctuation-delimited segment, has a short punctuation-free
    # title, and ends in a document/category noun before the colon.  This
    # covers e.g. “核验计划”, “设备使用规范”, and future domain
    # headings without teaching the safety gate every possible topic.
    r"(?:^|[，,。；;！？?!：:\r\n])\s*"
    r"[^，,。；;！？?!：:\r\n]{0,24}?"
    r"(?:计划|方案|规范|预案|流程|安排|要求|范围|步骤|指引|说明)"
    r"(?:如下(?:所示)?|包括以下(?:内容|事项|步骤)?|为以下(?:内容|事项|步骤)?)?"
    r"\s*[：:]"
)
_ACTION_NEGATION = re.compile(
    r"尚未|还未|并未|从未|未曾|不曾|没有|并没有|未能|没能"
)
_NEGATION_SCOPE_RESET = re.compile(r"但|却|而是|反而")
_ACTION_ASSERTION_RESET = re.compile(
    r"但|却|而是|反而|仍然?|实际|确实|最终|随后|随即|终于|成功|当场"
)


def has_unrealized_heading_frame(text: str) -> bool:
    """Return whether text contains a compositional non-execution heading."""

    return bool(_UNREALIZED_HEADING_FRAME.search(text))

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
    # A line/paragraph boundary is also a semantic boundary.  Model evidence
    # can span several source lines without terminal punctuation; merging those
    # lines would let a later negation, attribution, or condition alter an
    # earlier proposition.
    units = [
        part.strip()
        for part in re.findall(r"[^。！？?!；;\r\n]+[。！？?!；;]?", text)
    ]
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
    if directive.kind in {"world_assert", "uses"}:
        # A cited line can quote a plan first and narrate the actual action
        # afterwards.  Prefer the local performed-action clause so source and
        # modality classification do not inherit the earlier quotation.
        action_support = _realized_action_support(directive, units)
        if action_support:
            return action_support
    unit_scores = [
        (sum(anchor in _clean(unit) for anchor in anchors), unit) for unit in units
    ]
    best_unit_score = max((score for score, _ in unit_scores), default=0)
    best_units = [unit for score, unit in unit_scores if score == best_unit_score and score]
    if not best_units:
        return text
    unit = _preferred_support(best_units, directive)
    if directive.kind == "fact" and any(
        re.search(r"[，,]", match.group(0))
        for match in find_bound_fact_relation_matches(
            unit,
            subject=_clean(directive.attrs.get("subject", "")),
            predicate=_clean(directive.attrs.get("predicate", "")),
            value=_clean(directive.attrs.get("value", "")),
        )
    ):
        # A state transition may deliberately put a comma between the
        # attribute and its transition verb ("属性，现已调整为值").  Once the
        # complete bound relation is proven, do not discard half of it during
        # generic comma-localization.
        return unit
    # Keep the original inter-clause whitespace so the selected support remains
    # an exact slice of ``evidence.text``.  Source attribution is recovered by
    # offset later; normalizing the spaces here could make that lookup fail and
    # accidentally promote a reported condition to world truth.
    clauses = [part for part in re.split(r"(?<=[，,])", unit) if part.strip()]
    scored = [
        (sum(anchor in _clean(clause) for anchor in anchors), index, clause)
        for index, clause in enumerate(clauses)
    ]
    best_score = max((score for score, _, _ in scored), default=0)
    if best_score:
        selected = [index for score, index, _ in scored if score == best_score]
        first, last = min(selected), max(selected)
        if first:
            # A condition can span more than one comma-delimited clause, for
            # example ``如果 A，且 B，则 C``.  Keep the nearest governing
            # condition, but do not cross a contrast that starts a new claim.
            for index in range(first - 1, -1, -1):
                if not _HYPOTHETICAL_MARKERS.search(clauses[index]):
                    continue
                if not any(
                    _CONDITION_SCOPE_RESET.search(clause)
                    for clause in clauses[index + 1 : first + 1]
                ):
                    first = index
                break
            if (
                first
                and directive.kind == "event"
                and re.search(
                    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}", clauses[first - 1]
                )
            ):
                first -= 1
        # Chinese prose may place a condition after its consequence (for
        # example ``航线必须停运，如果风暴发生``).  Retain only an immediately
        # adjacent post-condition in this semantic unit.  A sentence/paragraph
        # boundary has already been removed by ``_semantic_units``; a contrast
        # starts a new claim and closes the condition scope.
        if last + 1 < len(clauses) and _HYPOTHETICAL_MARKERS.search(
            clauses[last + 1]
        ):
            last += 1
            while (
                last + 1 < len(clauses)
                and not _CONDITION_SCOPE_RESET.search(clauses[last + 1])
                and not any(
                    marker.search(clauses[last + 1])
                    for marker in _SOURCE_CONTEXT_MARKERS
                )
            ):
                last += 1
        return "".join(clauses[first : last + 1]).strip()
    return unit


def _preferred_support(
    candidates: list[str], directive: ParsedDirective
) -> str:
    """Prefer the latest authoritative support among equally grounded spans."""

    if not candidates:
        return ""
    return max(
        enumerate(candidates),
        key=lambda row: (
            _source_scope(row[1], directive) not in _NON_AUTHORITATIVE_SCOPES,
            row[0],
        ),
    )[1]


def _action_signature(directive: ParsedDirective) -> tuple[str, str]:
    """Return the candidate's normalized action label and optional scope."""

    if directive.kind == "uses":
        return _clean(directive.attrs.get("item", "")), ""
    if directive.kind != "world_assert":
        return "", ""
    key = _clean(directive.attrs.get("key", ""))
    if not key.startswith("scope_action:"):
        return "", ""
    parts = key.split(":", 2)
    if len(parts) != 3:
        return "", ""
    return parts[2], parts[1]


_USE_PRE_ACTION_LINK_PATTERN = (
    r"(?:(?:已经|随后|立即|此时|正|正在|亲自|亲手|独自|实际|确实|最终|"
    r"随即|终于|成功|擅自|直接|获准后|得到许可后|获得授权后|取得授权后|"
    r"准备好后|准备完成后|准备就绪后|计划|打算|准备|试图|尝试|即将|将要|"
    r"尚未|还未|并未|从未|未曾|不曾|没有|并没有|未能|没能)|"
    r"(?:(?:从|在|于)[^，,。；;！？?!：:]{1,24}?(?:中|内)?))*"
)
_USE_PRE_ACTION_LINK = re.compile(_USE_PRE_ACTION_LINK_PATTERN)
_USE_ITEM_TERMINAL = re.compile(
    r"^(?:$|[，,。；;！？?!：:\"'“”‘’（）()])"
)
_USE_ITEM_POSTPOSITION = re.compile(
    r"^(?:之后|以后|后|之时|时)(?=$|[，,。；;！？?!：:])"
)
_USE_ITEM_CONTINUATION = re.compile(
    rf"^(?:并|而|但|随后|从而)(?=(?:又|再|随即|立即|随后)?"
    rf"(?:他|她|它|其)?(?:{FOLLOWUP_USE_ACTION_VERB_PATTERN}))"
)
_USE_ITEM_PURPOSE = re.compile(
    r"^(?:开门|(?:打开|开启|关闭|启动|触发|激活|完成|击退|解锁|进入|离开)"
    r"(?!器(?=$|[，,。；;！？?!：:]|之后|以后|后|之时|时|并|而|但|随后|从而))"
    r"[^，,。；;！？?!：:]{1,32})"
    r"(?=$|[，,。；;！？?!：:])"
)
_USE_ACTOR_LEAD = re.compile(
    r"(?:随后|然后|接着|此后|当时|最终|随即|终于|"
    r"命令|下令|责令|要求|吩咐|嘱咐|敦促|通知|建议|提议|"
    r"准许|准予|容许|批准|答应|授意)$"
)


def _use_actor_lead_is_bound(lead: str) -> bool:
    return not lead or bool(_USE_ACTOR_LEAD.search(lead))


def _use_suffix_binds_item(suffix: str, *, item: str, verb: str) -> bool:
    item_match = re.match(rf"(?:了)?{re.escape(item)}", suffix)
    if item_match is None:
        return False
    remainder = suffix[item_match.end() :]
    if (
        _USE_ITEM_TERMINAL.match(remainder)
        or _USE_ITEM_POSTPOSITION.match(remainder)
        or _USE_ITEM_CONTINUATION.match(remainder)
    ):
        return True
    # A purpose/result clause may directly follow the object only for verbs
    # whose grammar permits “使用/用/借助 X 打开 Y”.  Operation verbs such as
    # “操作 X” require an actual object boundary, so “X启动器/打开器” cannot
    # be mistaken for X itself.
    return bool(
        verb in {"使用", "用", "借助"}
        and _USE_ITEM_PURPOSE.match(remainder)
    )


def find_bound_use_action_matches(
    text: str,
    *,
    user: str,
    item: str,
    actor_aliases: tuple[str, ...] = (),
) -> list[re.Match[str]]:
    """Find item-use verbs explicitly bound to ``user`` and ``item``.

    Returned matches point at the action verb itself.  This lets callers apply
    command, plan, report, and negation scope to the exact relation instead of
    accepting a keyword elsewhere in the sentence.  Unrealized link tokens are
    intentionally recognized here; the semantic modality gate rejects them
    after locating the relation.
    """

    bound_user = _clean(user)
    bound_item = _clean(item)
    if not bound_user or not bound_item:
        return []
    actors = tuple(
        dict.fromkeys(
            candidate
            for candidate in (bound_user, *(_clean(row) for row in actor_aliases))
            if candidate
        )
    )
    item_pattern = re.escape(bound_item)
    matches: list[re.Match[str]] = []
    for match in re.finditer(USE_ACTION_VERB_PATTERN, text):
        clause_start, clause_end = _clause_bounds(text, match.start())
        prefix = re.sub(r"\s+", "", text[clause_start : match.start()])
        suffix = re.sub(r"\s+", "", text[match.end() : clause_end])
        for actor in actors:
            actor_pattern = re.escape(actor)
            subject_first = re.search(
                rf"{actor_pattern}(?P<link>[^，,。；;！？?!：:]{{0,32}})$",
                prefix,
            )
            if subject_first is not None:
                link = subject_first.group("link")
                if (
                    _use_actor_lead_is_bound(prefix[: subject_first.start()])
                    and
                    _USE_PRE_ACTION_LINK.fullmatch(link)
                    and _use_suffix_binds_item(
                        suffix,
                        item=bound_item,
                        verb=match.group(0),
                    )
                ):
                    matches.append(match)
                    break
            performed_on = re.search(
                rf"{actor_pattern}{_USE_PRE_ACTION_LINK_PATTERN}对"
                rf"{item_pattern}(?:进行|进行了)$",
                prefix,
            )
            if performed_on is not None and _use_actor_lead_is_bound(
                prefix[: performed_on.start()]
            ):
                matches.append(match)
                break
            object_first = re.search(
                rf"{actor_pattern}{_USE_PRE_ACTION_LINK_PATTERN}(?:将|把)"
                rf"{item_pattern}{_USE_PRE_ACTION_LINK_PATTERN}$",
                prefix,
            )
            if object_first is not None and _use_actor_lead_is_bound(
                prefix[: object_first.start()]
            ):
                matches.append(match)
                break
            passive = re.search(
                rf"{item_pattern}"
                rf"(?:已经|已|正|正在|随后|最终)?被{actor_pattern}"
                rf"{_USE_PRE_ACTION_LINK_PATTERN}$",
                prefix,
            )
            if passive is not None and _use_actor_lead_is_bound(
                prefix[: passive.start()]
            ):
                matches.append(match)
                break
    return matches


def _candidate_action_matches(
    text: str, directive: ParsedDirective
) -> list[re.Match[str]]:
    if directive.kind == "uses":
        return find_bound_use_action_matches(
            text,
            user=directive.attrs.get("user", ""),
            item=directive.attrs.get("item", ""),
            actor_aliases=("他", "她", "他们", "她们")
            if "baseline" in directive.provenance_sources
            else (),
        )
    action_label, context = _action_signature(directive)
    compact = _clean(text)
    if context and context not in compact:
        return []
    matches: list[re.Match[str]] = []
    for match in _ACTION_VERBS.finditer(text):
        if action_label:
            _, clause_end = _clause_bounds(text, match.start())
            tail = _clean(text[match.end() : min(clause_end, match.end() + 48)])
            if action_label not in tail:
                continue
        matches.append(match)
    return matches


def _clause_bounds(text: str, offset: int) -> tuple[int, int]:
    separators = "，,。；;！!？?：:"
    start = max((text.rfind(token, 0, offset) for token in separators), default=-1) + 1
    ends = [
        position
        for token in separators
        if (position := text.find(token, offset)) >= 0
    ]
    return start, min(ends, default=len(text))


def _negation_governs_action(prefix: str) -> bool:
    matches = list(_ACTION_NEGATION.finditer(prefix))
    if not matches:
        return False
    tail = prefix[matches[-1].end() :]
    if _ACTION_VERBS.search(tail) or _NEGATION_SCOPE_RESET.search(tail):
        return False
    compact = _clean(tail)
    # Only grammatical material that can sit between an action negator and
    # its verb is accepted.  This prevents “从未迟疑便启动” and a negated
    # different action from contaminating the candidate action.
    return bool(
        re.fullmatch(
            r"(?:真正|实际|成功|亲自|正式|立即|直接|擅自|继续|再次|再)*"
            r"(?:(?:在|于)[^，,。；;！!？?：:]{0,24}?(?:中|内)?)?"
            r"(?:(?:将|把)[^，,。；;！!？?：:]{0,48})?",
            compact,
        )
    )


def _action_match_is_unrealized(text: str, match: re.Match[str]) -> bool:
    clause_start, _ = _clause_bounds(text, match.start())
    # Fact relation matches may include their leading colon while action
    # matches start after it.  Inspect the nearest colon on either side of
    # that match boundary so both families inherit the same heading scope.
    heading_colon = max(
        text.rfind("：", 0, match.start() + 1),
        text.rfind(":", 0, match.start() + 1),
    )
    if heading_colon >= 0:
        heading_start = max(
            (text.rfind(token, 0, heading_colon) for token in "。；;！!？?\n"),
            default=-1,
        ) + 1
        heading = text[heading_start : heading_colon + 1]
        if has_unrealized_heading_frame(heading):
            return True
    prefix = text[clause_start : match.start()]
    # Once an instruction/plan is explicitly completed, a following action is
    # narration rather than the still-unrealized content of that instruction.
    prefix = _COMPLETED_ORDER_BRIDGE.sub("", prefix)
    if _negation_governs_action(prefix):
        return True
    frames = list(_UNREALIZED_ACTION_FRAME.finditer(prefix))
    if not frames:
        return False
    after_frame = prefix[frames[-1].end() :]
    if _ACTION_VERBS.search(after_frame):
        return False
    return not bool(_ACTION_ASSERTION_RESET.search(after_frame))


def _action_realization(text: str, directive: ParsedDirective) -> bool | None:
    matches = _candidate_action_matches(text, directive)
    if not matches:
        return None
    return any(not _action_match_is_unrealized(text, match) for match in matches)


def _realized_action_support(
    directive: ParsedDirective, units: list[str]
) -> str | None:
    candidates: list[str] = []
    for unit in units:
        for match in _candidate_action_matches(unit, directive):
            if _action_match_is_unrealized(unit, match):
                continue
            start, end = _clause_bounds(unit, match.start())
            candidates.append(unit[start:end].strip())
    return _preferred_support(candidates, directive) if candidates else None


def evidence_presents_unrealized_action(directive: ParsedDirective) -> bool:
    """Return whether cited prose presents an action without completing it.

    The check is deliberately conservative and evidence-bound.  It only
    applies when the candidate itself is an action record, and an order or
    intention must govern the candidate action inside the same punctuation
    clause.  This keeps ordinary narration, including narration after a
    completed-order bridge, eligible.
    """

    if directive.kind not in {"world_assert", "uses"}:
        return False
    states = [
        state
        for unit in _semantic_units(directive.evidence.text)
        if (state := _action_realization(unit, directive)) is not None
    ]
    # Missing telemetry is not proof of a missing action.  Downgrade only when
    # the cited candidate action is actually present and every matching mention
    # is syntactically an order, intention, attempt, or direct negation.
    return bool(states) and not any(states)


def _local_source_context(text: str, directive: ParsedDirective) -> str:
    """Keep source attribution attached to the supporting semantic unit."""

    full = directive.evidence.text
    support = text.strip()
    if not support or support == full.strip():
        return full
    start = full.find(support)
    if start < 0:
        return support
    end = start + len(support)
    context = support
    prefix = full[:start]
    boundary = max(
        (prefix.rfind(token) for token in "。；;！!？?\n"),
        default=-1,
    )
    direct_intro = prefix[boundary + 1 :]
    if re.search(r"[：:]\s*$", direct_intro) or any(
        marker.search(direct_intro) for marker in _SOURCE_CONTEXT_MARKERS
    ):
        # Source attribution is a separate axis from proposition polarity.
        # Reattach only an attribution in the same semantic unit; keep it out
        # of ``_support_text`` so an unrelated "没有" in that prefix cannot
        # negate an otherwise affirmative proposition.
        context = direct_intro + context
    # Attribution can immediately follow a closing quote in the next semantic
    # unit (for example “……”值班员说道).  Preserve only that adjacent suffix;
    # an earlier report must not taint a later independent narration sentence.
    if any(token in support for token in ('“', '"')):
        suffix = full[end:]
        adjacent = re.match(r'^[”"]?(?:，|,)?[^。！？?!；;]{0,32}[。！？?!；;]?', suffix)
        if adjacent:
            context += adjacent.group(0)
    # An attribution can also follow an unquoted proposition in the same
    # sentence (``……，据某人所述``).  Keep that provenance for the source
    # axis, but never add it to ``_support_text``: words such as ``没有`` in
    # the attribution must not flip the proposition's polarity.  Do not cross
    # a terminal punctuation or newline; the following sentence may describe
    # an unrelated source.
    if support.rstrip() and support.rstrip()[-1] not in "。！？?!；;\n":
        suffix = full[end:]
        adjacent = re.match(r"^[ \t]*[^。！？?!；;\n]{1,64}[。！？?!；;]?", suffix)
        if adjacent and any(
            marker.search(adjacent.group(0))
            for marker in _SOURCE_CONTEXT_MARKERS
        ):
            context += adjacent.group(0)
    return context


def _source_scope(text: str, directive: ParsedDirective) -> SourceScope:
    full_evidence = directive.evidence.text
    local_evidence = _local_source_context(text, directive)
    # Provenance belongs to the supporting semantic unit.  A reported claim in
    # one sentence must not taint a later, independent narrator confirmation;
    # `_local_source_context` still reattaches same-unit introductions and
    # adjacent attributions, preventing semantic laundering.
    authority_evidence = local_evidence
    if re.search(
        r"(?:记录|日志|报告|档案|时间戳|时钟)(?:已?(?:被|遭)|已)?"
        r"(?:伪造|造假|篡改|改写)|"
        r"(?:伪造|造假|篡改)的(?:记录|日志|报告|档案|时间戳)",
        full_evidence,
    ):
        return SourceScope.unverified_report
    if _UNVERIFIED_SOURCE_MARKERS.search(authority_evidence) or re.search(
        r"信中声称", authority_evidence
    ):
        return SourceScope.unverified_report
    # Attribution can follow the quoted proposition. `_local_source_context`
    # keeps that adjacent attribution while excluding unrelated earlier prose.
    if _DIALOGUE_MARKERS.search(local_evidence) or _REPORTED_SOURCE_MARKERS.search(
        local_evidence
    ):
        return SourceScope.character_dialogue
    if _ACTUAL_QUOTE_MARKERS.search(local_evidence):
        return SourceScope.quoted_material
    if _QUOTED_SOURCE_MARKERS.search(local_evidence):
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


_STRUCTURED_FACT_PREDICATES = {
    "mobility_permission",
    "mobility_limit",
    "rule_exception",
    "authorization",
}
_FACT_VALUE_BOUNDARY = (
    r"(?=$|[，,。；;！？?!：:\"'“”‘’（）()]|"
    r"而是|而为|却是|却为|"
    r"(?:了|的)(?=$|[，,。；;！？?!：:\"'“”‘’（）()]))"
)
_FACT_SUBJECT_LEAD = (
    r"(?:^|[，,。；;！？?!：:\"'“”‘’（）()]|随后|然后|接着|此后|当时|最终|"
    r"随即|终于|确认|显示|表明|指出|记载|修正|更新|宣布|认定|称)"
)


def _fact_predicate_variants(predicate: str) -> tuple[str, ...]:
    variants = [predicate]
    # Some model providers include the copula in the predicate label
    # (``身份是``).  Treat that as the same relation as ``身份`` + ``是``
    # without broadly trimming ordinary predicates such as ``行为``.
    if predicate.endswith("是") and len(predicate) > 1:
        variants.append(predicate[:-1])
    if predicate in {"身份", "身份是"}:
        # Identity prose commonly omits the normalized predicate label:
        # “林澈是领航员” is the closed surface form of 身份=领航员.
        variants.append("")
    return tuple(dict.fromkeys(variants))


def find_bound_fact_relation_matches(
    support: str, *, subject: str, predicate: str, value: str
) -> list[re.Match[str]]:
    """Return prose spans which explicitly bind one fact triple.

    Co-occurrence is not enough.  In particular, a copula or transition about
    some other object must not make an otherwise unrelated ``subject``,
    ``predicate`` and ``value`` eligible for deterministic rules.  The closed
    forms below cover ordinary attribute copulas, explicit state transitions,
    and a directly negated verbal relation.
    """

    if not subject or not predicate or not value:
        return []
    compact = re.sub(r"\s+", "", support)
    bound_subject = re.escape(subject)
    bound_value = re.escape(value)
    modifiers = r"(?:当前|现在|现已|已经|已|仍然|仍|最终|后来|随即|就)?"
    relations: list[re.Pattern[str]] = []
    for predicate_variant in _fact_predicate_variants(predicate):
        bound_predicate = re.escape(predicate_variant)
        attribute = (
            rf"{_FACT_SUBJECT_LEAD}{bound_subject}(?:的)?{bound_predicate}"
        )
        relations.extend(
            (
                re.compile(
                    rf"{attribute}(?:是否)?{modifiers}"
                    rf"(?:不是|并非|不为|是(?!否)|属于|担任|为)(?:了)?"
                    rf"{bound_value}{_FACT_VALUE_BOUNDARY}"
                ),
                re.compile(
                    rf"{attribute}(?:[，,])?{modifiers}"
                    rf"(?:(?:从|由)[^，,。；;！？?!]{{1,32}})?"
                    rf"(?:被(?![^，,。；;！？?!]{{0,12}}(?:计划|要求|提议|命令))"
                    rf"[^，,。；;！？?!]{{1,24}})?"
                    rf"(?:变成|变为|成为|转为|改为|更改为|更新为|调整为|切换为|"
                    rf"设置为|设置成|设为|设成|定为|定成)"
                    rf"(?:了)?{bound_value}{_FACT_VALUE_BOUNDARY}"
                ),
                re.compile(
                    rf"{attribute}{modifiers}"
                    rf"(?:不是|并非(?:是|为)?|不为)"
                    rf"[^，,。；;！？?!]{{1,32}}?"
                    rf"(?:而是|而为|却是|却为){bound_value}{_FACT_VALUE_BOUNDARY}"
                ),
                re.compile(
                    rf"{_FACT_SUBJECT_LEAD}"
                    rf"(?!(?:(?!将|把)[^，,。；;！？?!]){{0,24}}"
                    rf"(?:计划|要求|提议|命令|打算|准备|尚未|还未|并未|未曾))"
                    rf"(?:(?!将|把)[^，,。；;！？?!]){{0,24}}?"
                    rf"(?:最终|随后|随即|已经|已)?(?:将|把)"
                    rf"{bound_subject}(?:的)?{bound_predicate}{modifiers}"
                    rf"(?:变成|变为|成为|转为|改为|更改为|更新为|调整为|切换为|"
                    rf"设置为|设置成|设为|设成|定为|定成)"
                    rf"(?:了)?{bound_value}{_FACT_VALUE_BOUNDARY}"
                ),
                # Some facts use a verbal predicate rather than an attribute
                # copula, e.g. ``角色甲不可能抵达山门``.  Only an explicit,
                # immediately governing negation closes that relation.
                re.compile(
                    rf"{_FACT_SUBJECT_LEAD}{bound_subject}(?:明确)?"
                    rf"(?:从未|未曾|并未|没有|不可能|绝不可能|无法|不能)"
                    rf"(?:真正|实际|成功|及时)?{bound_predicate}(?:了|过)?"
                    rf"{bound_value}{_FACT_VALUE_BOUNDARY}"
                ),
            )
        )
    return [match for relation in relations for match in relation.finditer(compact)]


def _fact_has_bound_relation(
    support: str, *, subject: str, predicate: str, value: str
) -> bool:
    return bool(
        find_bound_fact_relation_matches(
            support,
            subject=subject,
            predicate=predicate,
            value=value,
        )
    )


def _fact_contrast_affirms_submitted_value(
    support: str, directive: ParsedDirective
) -> bool:
    attrs = directive.attrs
    return any(
        re.search(r"而是|而为|却是|却为", match.group(0))
        for match in find_bound_fact_relation_matches(
            support,
            subject=_clean(attrs.get("subject", "")),
            predicate=_clean(attrs.get("predicate", "")),
            value=_clean(attrs.get("value", "")),
        )
    )


def _fact_relation_is_unrealized(
    support: str, directive: ParsedDirective
) -> bool:
    if directive.kind != "fact":
        return False
    attrs = directive.attrs
    compact = re.sub(r"\s+", "", support)
    matches = find_bound_fact_relation_matches(
        compact,
        subject=_clean(attrs.get("subject", "")),
        predicate=_clean(attrs.get("predicate", "")),
        value=_clean(attrs.get("value", "")),
    )
    states = [not _action_match_is_unrealized(compact, match) for match in matches]
    return bool(states) and not any(states)


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
        predicate = _clean(attrs.get("predicate", ""))
        if predicate in _STRUCTURED_FACT_PREDICATES:
            # These normalized state records intentionally use server-owned
            # predicate/value labels which need not occur verbatim in prose.
            # Their extractors have separate relation-specific grammars.
            closed = bool(
                subject
                and (subject in compact or subject == "*")
                and re.search(
                    r"(?:获得|获准|允许|许可|持有|拥有|没有|并未|未曾|"
                    r"不可能|无法|不能|禁止|不得|至少需要)",
                    support,
                )
            )
        else:
            closed = _fact_has_bound_relation(
                support,
                subject=subject,
                predicate=predicate,
                value=value,
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
        closed = bool(_candidate_action_matches(support, directive))
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
    contrast_affirmed = (
        kind == "fact" and _fact_contrast_affirms_submitted_value(support, directive)
    )
    modality = (
        SemanticModality.negated
        if not contrast_affirmed
        and (
            _NEGATION_MARKERS.search(support)
            or _DEFINITE_IMPOSSIBILITY_MARKERS.search(support)
        )
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
        if _closed_conditional_rule(support, directive):
            inferred = (
                SemanticModality.conditional_rule,
                SourceScope.world_rule,
                CertaintyLevel.certain,
            )
        elif _HYPOTHETICAL_MARKERS.search(support):
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
            inferred = _closed_baseline_semantics(support, directive)
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

    if _fact_relation_is_unrealized(support, directive):
        relation_scope = _source_scope(support, directive)
        return (
            _to_noncanonical(
                directive,
                kind=(
                    "character_claim"
                    if relation_scope
                    in {
                        SourceScope.character_dialogue,
                        SourceScope.quoted_material,
                        SourceScope.unverified_report,
                    }
                    else "tentative_fact"
                ),
                modality=(
                    SemanticModality.reported
                    if relation_scope
                    in {
                        SourceScope.character_dialogue,
                        SourceScope.quoted_material,
                        SourceScope.unverified_report,
                    }
                    else SemanticModality.uncertain
                ),
                source_scope=relation_scope,
                certainty=CertaintyLevel.unknown,
            ),
            "unrealized_fact_relation",
        )

    if evidence_presents_unrealized_action(directive):
        action_scope = _source_scope(support, directive)
        if action_scope in {
            SourceScope.character_dialogue,
            SourceScope.quoted_material,
            SourceScope.unverified_report,
        }:
            return (
                _to_noncanonical(
                    directive,
                    kind="character_claim",
                    modality=SemanticModality.reported,
                    source_scope=action_scope,
                    certainty=CertaintyLevel.unknown,
                ),
                "unrealized_action",
            )
        return (
            _to_noncanonical(
                directive,
                kind="tentative_fact",
                modality=SemanticModality.uncertain,
                source_scope=action_scope,
                certainty=CertaintyLevel.unknown,
            ),
            "unrealized_action",
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

    # Relation closure is the final canonicality check, after modality and
    # source authority have had priority.  Keeping this ordering lets the
    # bounded repair Agent distinguish a lexical mismatch from an attempt to
    # promote hypothetical, reported, or otherwise non-authoritative evidence.
    if (
        directive.kind == "fact"
        and attrs.get("input_form") != "directive"
        and not attrs.get("predicate", "").startswith("body_state:")
        and _clean(attrs.get("predicate", "")) not in _STRUCTURED_FACT_PREDICATES
        and not _fact_has_bound_relation(
            support,
            subject=_clean(attrs.get("subject", "")),
            predicate=_clean(attrs.get("predicate", "")),
            value=_clean(attrs.get("value", "")),
        )
    ):
        return (
            _to_noncanonical(
                directive,
                kind="tentative_fact",
                modality=SemanticModality.uncertain,
                source_scope=scope,
                certainty=CertaintyLevel.unknown,
            ),
            "unbound_fact_relation",
        )

    if (
        directive.kind == "uses"
        and attrs.get("input_form") != "directive"
        and not _candidate_action_matches(support, directive)
    ):
        return (
            _to_noncanonical(
                directive,
                kind="tentative_fact",
                modality=SemanticModality.uncertain,
                source_scope=scope,
                certainty=CertaintyLevel.unknown,
            ),
            "unbound_use_relation",
        )

    negated = bool(
        _NEGATION_MARKERS.search(support)
        or _DEFINITE_IMPOSSIBILITY_MARKERS.search(support)
    )
    contrast_affirmed = bool(
        directive.kind == "fact"
        and _fact_contrast_affirms_submitted_value(support, directive)
    )
    if contrast_affirmed:
        # In "不是 A 而是 B", negation scopes over A; the submitted B value
        # is explicitly affirmed.
        negated = False
        declared_modality = SemanticModality.asserted
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
