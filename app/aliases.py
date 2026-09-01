from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .domain import ParsedDirective
if TYPE_CHECKING:
    from .pipeline import DocumentInput


ENTITY_FIELDS = {"subject", "character", "owner", "user", "actor"}
NAME = r"[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9·_-]{0,15}"
ALIAS_PATTERN = re.compile(
    rf"(?:^|[，,。；;：:\s])(?P<canonical>{NAME})(?:的)?"
    rf"(?:又名|简称(?:为|是)?|化名(?:为|是)?|代号(?:为|是)?)[“\"]?"
    rf"(?P<alias>{NAME})[”\"]?"
)


@dataclass(slots=True)
class AliasResolutionResult:
    directives: list[ParsedDirective]
    alias_map: dict[str, str] = field(default_factory=dict)
    declarations: list[dict] = field(default_factory=list)
    traces: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _declarations(documents: list[DocumentInput]) -> tuple[dict[str, set[str]], list[dict]]:
    candidates: dict[str, set[str]] = {}
    rows: list[dict] = []
    for document in documents:
        for line_number, source_line in enumerate(document.content.splitlines(), start=1):
            for match in ALIAS_PATTERN.finditer(source_line):
                canonical = match.group("canonical").strip()
                alias = match.group("alias").strip(" ，,。；;：:\"“”")
                if not canonical or not alias or canonical == alias:
                    continue
                candidates.setdefault(alias, set()).add(canonical)
                rows.append({
                    "canonical": canonical,
                    "alias": alias,
                    "document_id": document.id,
                    "document_name": document.name,
                    "line": line_number,
                })
    return candidates, rows


def build_alias_map(documents: list[DocumentInput]) -> tuple[dict[str, str], list[dict], list[str]]:
    candidates, declarations = _declarations(documents)
    warnings: list[str] = []
    mapping: dict[str, str] = {}
    for alias, canonicals in sorted(candidates.items()):
        if len(canonicals) != 1:
            warnings.append(f"实体别名“{alias}”对应多个主名，已保持原名不合并")
            continue
        mapping[alias] = next(iter(canonicals))

    cyclic: set[str] = set()
    for alias in list(mapping):
        path: list[str] = []
        current = alias
        while current in mapping:
            if current in path:
                cyclic.update(path[path.index(current):])
                break
            path.append(current)
            current = mapping[current]
    if cyclic:
        warnings.append(f"检测到实体别名循环：{'、'.join(sorted(cyclic))}；相关名称不合并")
        for name in cyclic:
            mapping.pop(name, None)
        for alias, canonical in list(mapping.items()):
            if canonical in cyclic:
                mapping.pop(alias, None)

    def resolve(value: str) -> str:
        seen: set[str] = set()
        while value in mapping and value not in seen:
            seen.add(value)
            value = mapping[value]
        return value

    mapping = {alias: resolve(canonical) for alias, canonical in mapping.items()}
    return mapping, declarations, warnings


def canonicalize_entities(
    documents: list[DocumentInput], directives: list[ParsedDirective]
) -> AliasResolutionResult:
    alias_map, declarations, warnings = build_alias_map(documents)
    traces: list[dict] = []
    for directive in directives:
        for field_name in ENTITY_FIELDS:
            original = directive.attrs.get(field_name, "")
            canonical = alias_map.get(original)
            if not canonical:
                continue
            directive.attrs[field_name] = canonical
            traces.append({
                "alias": original,
                "canonical": canonical,
                "kind": directive.kind,
                "field": field_name,
                "document_name": directive.evidence.document_name,
                "line": directive.evidence.line_start,
            })
        participants = directive.attrs.get("participants", "")
        if participants:
            values = [value.strip() for value in re.split(r"[,，]", participants) if value.strip()]
            normalized = []
            for value in values:
                canonical = alias_map.get(value, value)
                normalized.append(canonical)
                if canonical != value:
                    traces.append({
                        "alias": value,
                        "canonical": canonical,
                        "kind": directive.kind,
                        "field": "participants",
                        "document_name": directive.evidence.document_name,
                        "line": directive.evidence.line_start,
                    })
            directive.attrs["participants"] = ",".join(normalized)
    return AliasResolutionResult(
        directives=directives,
        alias_map=alias_map,
        declarations=declarations,
        traces=traces,
        warnings=warnings,
    )
