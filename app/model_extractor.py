from __future__ import annotations

import re
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from .domain import EvidenceSpan, ParsedDirective
from .chunking import DocumentChunk, chunk_document, numbered_chunk
from .parser import ParsedDocument
from .pipeline import BaselineExtractor, DocumentInput
from .provider import OpenAICompatibleProvider, ProviderError
from .usage import estimate_request_tokens


class ExtractionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_line_start: int = Field(ge=1)
    source_line_end: int = Field(ge=1)


class FactExtraction(ExtractionBase):
    kind: Literal["fact"]
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    value: str = Field(min_length=1)
    time: str | None = None
    origin: str | None = None
    destination: str | None = None
    bidirectional: str | None = None
    status: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    current: str | None = None
    key: str | None = None


class EventExtraction(ExtractionBase):
    kind: Literal["event"]
    id: str | None = None
    time: str = Field(min_length=1)
    location: str = Field(min_length=1)
    participants: list[str] = Field(min_length=1)


class KnowledgeExtraction(ExtractionBase):
    kind: Literal["knows", "claims_knows"]
    character: str = Field(min_length=1)
    fact: str = Field(min_length=1)
    time: str = Field(min_length=1)


class ItemExtraction(ExtractionBase):
    kind: Literal["item"]
    item: str = Field(min_length=1)
    owner: str = Field(min_length=1)
    time: str = ""


class UsesExtraction(ExtractionBase):
    kind: Literal["uses"]
    item: str = Field(min_length=1)
    user: str = Field(min_length=1)
    time: str = ""


class WorldExtraction(ExtractionBase):
    kind: Literal["world_rule", "world_assert"]
    key: str = Field(min_length=1)
    value: str = Field(min_length=1)
    actor: str | None = None
    time: str | None = None


ExtractionRecord = Annotated[
    Union[
        FactExtraction,
        EventExtraction,
        KnowledgeExtraction,
        ItemExtraction,
        UsesExtraction,
        WorldExtraction,
    ],
    Field(discriminator="kind"),
]


class ExtractionEnvelope(BaseModel):
    """Validate the transport envelope without making one bad record fatal."""

    model_config = ConfigDict(extra="forbid")
    records: list[Any] = Field(default_factory=list, max_length=500)


ENVELOPE_ADAPTER = TypeAdapter(ExtractionEnvelope)
RECORD_ADAPTER = TypeAdapter(ExtractionRecord)

SYSTEM_PROMPT = """你是 LoreGuard 的叙事状态抽取器，只抽取状态，不判断矛盾。
只返回 JSON 对象 {"records": [...]}；顶层不得有其他字段，不得使用 Markdown 代码块。
每条记录只表达一个原子状态，且只能使用对应 kind 的字段；所有记录都必须包含 kind、source_line_start、source_line_end。

唯一允许的记录格式如下（未标 optional 的字段全部必需）：
- fact: kind, subject, predicate, value, time(optional), source_line_start, source_line_end。明确的移动许可可使用 predicate=mobility_permission、value=instant_transport，并附 origin、destination、bidirectional、status、valid_from、valid_until、current；明确的规则例外可使用 predicate=rule_exception、value=allowed，并附 key、status、valid_from、valid_until、current。不得把模糊许可猜成有效状态
- event: kind, id(optional), time, location, participants, source_line_start, source_line_end；participants 必须是非空字符串数组
- knows: kind, character, fact, time, source_line_start, source_line_end
- claims_knows: kind, character, fact, time, source_line_start, source_line_end
- item: kind, item, owner, time(optional), source_line_start, source_line_end
- uses: kind, item, user, time(optional), source_line_start, source_line_end
- world_rule: kind, key, value, source_line_start, source_line_end
- world_assert: kind, key, value, actor(optional), time(optional), source_line_start, source_line_end；原文明示执行者时必须填写 actor

绝对禁止 description、content、text、evidence 等未列出的字段。不要用 description 代替任何必需字段。
缺少某 kind 的必需信息时，省略该记录，不要猜测或填“未知”。普通叙述动作若没有明确时间、地点和参与者，不抽成 event。
角色取出/拿出某件工具后盖下、按下、插入或明确使用时，输出 uses；item 必须是被操作的工具本身，不是动作产生的印记或结果。
对于“进入某范围后某能力失效/禁止”的规则，优先输出 world_rule，key 使用 scope_action:<范围>:<能力>，value 使用 disabled。
对于“仍在某范围内发动/使用某能力”的明确行为，输出 world_assert，使用同样的 key 结构，value 使用 performed。只抽行为，不自行判断是否违规。
仅抽取原文明确陈述的内容，不补全常识，不做代词猜测，不创造角色、时间或地点。
source_line_start/end 必须引用输入的真实行号。world_rule 是设定约束，world_assert 是章节中的规则实现或主张。
"""


def _clean(value: str) -> str:
    return re.sub(r"\s+", "", value).strip("，。；;：:\"'")


def _attrs(record: ExtractionRecord, line_start: int) -> dict[str, str]:
    data = record.model_dump(
        exclude={"kind", "source_line_start", "source_line_end"}, exclude_none=True
    )
    if record.kind == "event":
        participants = data.pop("participants")
        data["participants"] = ",".join(participants)
        data["id"] = data.get("id") or f"model-{line_start}"
    return {key: str(value) for key, value in data.items()}


def _fingerprint(
    directive: ParsedDirective,
) -> tuple[str, tuple[tuple[str, str], ...], int, int]:
    attrs = dict(directive.attrs)
    if directive.kind == "event":
        attrs.pop("id", None)
        participants = re.split(r"[,，]", attrs.get("participants", ""))
        attrs["participants"] = ",".join(
            sorted(_clean(item) for item in participants if _clean(item))
        )
    normalized = tuple(sorted((key, _clean(value)) for key, value in attrs.items() if value))
    return (
        directive.kind,
        normalized,
        directive.evidence.line_start,
        directive.evidence.line_end,
    )


def merge_directives(
    baseline: list[ParsedDirective], model: list[ParsedDirective]
) -> list[ParsedDirective]:
    merged = list(baseline)
    seen = {_fingerprint(item) for item in baseline}
    for item in model:
        fingerprint = _fingerprint(item)
        if fingerprint not in seen:
            merged.append(item)
            seen.add(fingerprint)
    return merged


def _evidence_supports(record: ExtractionRecord, text: str) -> bool:
    """Conservative lexical guard for model records prone to false positives."""
    compact = _clean(text)
    if record.kind == "uses":
        if _clean(record.item) not in compact or _clean(record.user) not in compact:
            return False
        positive_operation = re.search(
            r"取出|拿出|掏出|拔出|盖下|按下|插入|启用|挥动|使用(?!权)|用(?!权)",
            compact,
        )
        if not positive_operation:
            positive_operation = re.search(
                r"拿(?:着|起)?[^。；]{0,24}(?:打开|开启|解锁|砸开|挡住)",
                compact,
            )
        if not positive_operation:
            return False
        if re.search(r"(?:没有|无人|无权|禁止|不得|不能|未曾)[^。；]{0,16}使用", compact):
            return bool(re.search(r"取出|拿出|掏出|拔出|盖下|按下|插入|启用|挥动", compact))
    elif record.kind == "item":
        return _clean(record.item) in compact and _clean(record.owner) in compact
    elif record.kind == "event":
        return all(_clean(participant) in compact for participant in record.participants)
    return True


class ModelEnhancedExtractor:
    """Baseline-first extractor: model failures are warnings, never run failures."""

    def __init__(
        self,
        provider: OpenAICompatibleProvider | None = None,
        baseline: BaselineExtractor | None = None,
    ) -> None:
        self.provider = provider or OpenAICompatibleProvider()
        self.baseline = baseline or BaselineExtractor()
        self._run_tokens_used = 0

    def begin_run(self) -> None:
        self._run_tokens_used = 0

    def extract(self, document: DocumentInput) -> ParsedDocument:
        parsed = self.baseline.extract(document)
        if not self.provider.configured:
            return parsed

        settings = self.provider.settings
        chunks = chunk_document(
            document,
            max_chars=settings.model_chunk_max_chars,
            overlap_lines=settings.model_chunk_overlap_lines,
        )
        selected = chunks[: settings.model_max_chunks_per_document]
        if len(selected) < len(chunks):
            parsed.warnings.append(
                f"模型分块超过上限：仅处理前 {len(selected)}/{len(chunks)} 块；未处理部分仍由全文 BaselineExtractor 覆盖"
            )

        model_directives: list[ParsedDirective] = []
        any_success = False
        failed_chunks = 0
        for chunk in selected:
            user_prompt = (
                f"文档名：{document.name}\n当前分块：{chunk.id}，原文全局行 {chunk.global_line_start}-{chunk.global_line_end}\n"
                f"以下文本使用原文全局行号：\n{numbered_chunk(chunk)}"
            )
            estimated_tokens = estimate_request_tokens(SYSTEM_PROMPT, user_prompt)
            used_tokens = self._run_tokens_used
            if used_tokens + estimated_tokens > settings.per_run_token_budget:
                parsed.warnings.append(
                    f"模型单次运行 Token 预算不足：已用 {used_tokens}，下一分块保守估算 {estimated_tokens}，预算 {settings.per_run_token_budget}；后续分块由全文基线覆盖"
                )
                break
            try:
                response = self.provider.complete(
                    SYSTEM_PROMPT,
                    user_prompt,
                )
                parsed.prompt_tokens += response.prompt_tokens
                parsed.completion_tokens += response.completion_tokens
                reported_tokens = response.prompt_tokens + response.completion_tokens
                # Some compatible providers omit usage or report fewer tokens
                # than our local preflight estimate.  Budget accounting stays
                # conservative even though persisted usage remains the actual
                # provider-reported value.
                self._run_tokens_used += max(reported_tokens, estimated_tokens)
                envelope = ENVELOPE_ADAPTER.validate_json(response.text)
                chunk_directives: list[ParsedDirective] = []
                invalid_count = 0
                for index, raw_record in enumerate(envelope.records, start=1):
                    try:
                        record = RECORD_ADAPTER.validate_python(raw_record)
                        chunk_directives.append(self._to_directive(document, chunk, record))
                    except (ValidationError, ValueError):
                        invalid_count += 1
                        parsed.warnings.append(
                            f"模型记录 #{index}（分块 {chunk.id}）不符合抽取协议，已安全跳过"
                        )
                if envelope.records and not chunk_directives:
                    raise ValueError(f"模型返回的 {invalid_count} 条记录全部无效")
                model_directives.extend(chunk_directives)
                any_success = True
            except (ProviderError, ValidationError, ValueError) as exc:
                failed_chunks += 1
                parsed.warnings.append(
                    f"模型分块 {chunk.id} 抽取不可用，已由全文基线覆盖（{type(exc).__name__}）"
                )
        parsed.directives = merge_directives(parsed.directives, model_directives)
        parsed.model_used = any_success
        if selected and failed_chunks == len(selected):
            parsed.warnings.append("模型抽取不可用，已降级到 BaselineExtractor（所有分块失败）")
        return parsed

    @staticmethod
    def _to_directive(
        document: DocumentInput, chunk: DocumentChunk, record: ExtractionRecord
    ) -> ParsedDirective:
        lines = document.content.splitlines()
        start, end = record.source_line_start, record.source_line_end
        if (
            end < start
            or end > len(lines)
            or start < chunk.global_line_start
            or end > chunk.global_line_end
        ):
            raise ValueError("模型返回的证据行号越界")
        evidence = EvidenceSpan(
            document_id=document.id,
            document_name=document.name,
            line_start=start,
            line_end=end,
            text="\n".join(lines[start - 1 : end]).strip(),
        )
        if not evidence.text:
            raise ValueError("模型返回了空证据区间")
        if not _evidence_supports(record, evidence.text):
            raise ValueError("模型记录缺少原文词面支持")
        return ParsedDirective(
            kind=record.kind, attrs=_attrs(record, start), evidence=evidence
        )
