from __future__ import annotations

import json
import shlex
from dataclasses import dataclass, field

from .domain import EvidenceSpan, ParsedDirective
from .natural import extract_natural_line


SUPPORTED_DIRECTIVES = {
    "entity",
    "fact",
    "event",
    "knows",
    "claims_knows",
    "item",
    "uses",
    "world_rule",
    "world_assert",
}


@dataclass(slots=True)
class ParsedDocument:
    document_id: str
    document_name: str
    directives: list[ParsedDirective] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_used: bool = False


def _parse_attrs(raw: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for token in shlex.split(raw, posix=True):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        attrs[key.strip()] = value.strip()
    return attrs


def parse_document(document_id: str, document_name: str, content: str) -> ParsedDocument:
    parsed = ParsedDocument(document_id=document_id, document_name=document_name)
    unmatched_natural_lines: list[int] = []
    if document_name.lower().endswith(".json"):
        try:
            payload = json.loads(content)
            if isinstance(payload, dict) and isinstance(payload.get("directives"), list):
                content = "\n".join(payload["directives"])
        except json.JSONDecodeError as exc:
            parsed.warnings.append(f"JSON parse error: {exc}")

    for line_no, source_line in enumerate(content.splitlines(), start=1):
        line = source_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith("@"):
            evidence = EvidenceSpan(
                document_id=document_id,
                document_name=document_name,
                line_start=line_no,
                line_end=line_no,
                text=source_line.strip(),
            )
            extracted = extract_natural_line(evidence)
            parsed.directives.extend(extracted)
            if not extracted and len(line) >= 8:
                unmatched_natural_lines.append(line_no)
            continue
        head, _, evidence_text = line.partition("|")
        parts = head[1:].strip().split(maxsplit=1)
        kind = parts[0].lower()
        if kind not in SUPPORTED_DIRECTIVES:
            parsed.warnings.append(f"Line {line_no}: unsupported directive @{kind}")
            continue
        attrs = _parse_attrs(parts[1] if len(parts) > 1 else "")
        evidence = EvidenceSpan(
            document_id=document_id,
            document_name=document_name,
            line_start=line_no,
            line_end=line_no,
            text=(evidence_text.strip() or source_line.strip()),
        )
        parsed.directives.append(ParsedDirective(kind=kind, attrs=attrs, evidence=evidence))
    if unmatched_natural_lines:
        preview = "、".join(str(line) for line in unmatched_natural_lines[:5])
        remaining = len(unmatched_natural_lines) - 5
        suffix = f"，另有 {remaining} 行" if remaining > 0 else ""
        parsed.warnings.append(
            f"确定性基线有 {len(unmatched_natural_lines)} 行未做高置信抽取"
            f"（示例行号：{preview}{suffix}）；普通叙述仍保留，启用模型可增强语义覆盖"
        )
    return parsed
