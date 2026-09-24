from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING
import unicodedata

if TYPE_CHECKING:
    from .pipeline import DocumentInput


_PROFILE_H2 = re.compile(
    r"^##[ \t]+([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9·・]{0,19})"
    r"(?:[ \t]*[｜|][ \t]*([\u4e00-\u9fff]{2,16}))?[ \t]*$"
)
_PROFILE_ROLE_TITLE = re.compile(
    r"[\u4e00-\u9fff]{1,14}"
    r"(?:员|师|官|长|士|者|人|文书|调度|顾问|经理|主管|店主|"
    r"主编|导演|演员|策划|教授|学生|学徒)"
)
_PROFILE_NON_PERSON_NAME = re.compile(
    r"(?:第[一二三四五六七八九十百千零〇两0-9]+[章节幕卷回篇话集]|"
    r"序章|终章|番外|前言|后记|楔子|尾声|附录|角色[甲乙丙丁]|"
    r".{1,10}(?:和|与|及|同).{1,10})"
)
_PROFILE_H1 = re.compile(r"^#[ \t]+(.+?)[ \t]*$")
_PROFILE_FENCE = re.compile(r"^[ \t]*(?:```|~~~)")
_PROFILE_GENERIC_HEADINGS = frozenset(
    {
        "简介", "概览", "总览", "角色", "人物", "角色档案", "人物档案",
        "角色设定", "人物设定", "世界观", "背景", "身世", "经历",
        "性格", "核心性格", "人物性格", "性格特点", "偏好", "习惯",
        "能力", "外貌", "人物关系", "角色关系", "关系", "时间线",
        "设定", "故事", "章节", "事件", "备注", "补充", "其他",
    }
)


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    id: str
    document_id: str
    document_name: str
    global_line_start: int
    global_line_end: int
    content: str


def _make_chunk(document: DocumentInput, index: int, rows: list[tuple[int, str]]) -> DocumentChunk:
    return DocumentChunk(
        id=f"{document.id}:chunk:{index}",
        document_id=document.id,
        document_name=document.name,
        global_line_start=rows[0][0],
        global_line_end=rows[-1][0],
        content="\n".join(text for _, text in rows),
    )


def chunk_document(
    document: DocumentInput,
    max_chars: int,
    overlap_lines: int = 0,
) -> list[DocumentChunk]:
    """Split on source lines while retaining global line ranges.

    A single source line longer than ``max_chars`` is emitted as multiple safe
    fragments that all retain that source line's global number. Line overlap is
    applied only to ordinary multi-line chunks, never by silently dropping text.
    """
    if max_chars < 32:
        raise ValueError("model_chunk_max_chars must be at least 32")
    if overlap_lines < 0:
        raise ValueError("model_chunk_overlap_lines cannot be negative")
    lines = document.content.splitlines()
    if not lines:
        return []

    return _chunk_numbered_rows(
        document,
        list(enumerate(lines, start=1)),
        max_chars,
        overlap_lines=overlap_lines,
    )


def chunk_character_profile_document(
    document: DocumentInput,
    max_chars: int,
) -> list[DocumentChunk]:
    """Isolate clearly named character sections without changing source offsets.

    Ambiguous Markdown, preamble facts, and profiles without a unique named
    section layout retain ordinary line-based chunking.  Every original line
    belongs to exactly one section before the existing long-line splitter runs.
    """
    if max_chars < 32:
        raise ValueError("model_chunk_max_chars must be at least 32")
    if document.role != "character_profile":
        return chunk_document(document, max_chars, overlap_lines=0)
    lines = document.content.splitlines()
    if not lines or any(_PROFILE_FENCE.match(line) for line in lines):
        return chunk_document(document, max_chars, overlap_lines=0)

    headings = [
        index
        for index, line in enumerate(lines)
        if line.startswith("## ") or line.startswith("##\t")
    ]
    if len(headings) < 2:
        return chunk_document(document, max_chars, overlap_lines=0)

    preamble = [line for line in lines[: headings[0]] if line.strip()]
    if len(preamble) > 1 or (
        preamble
        and not (
            (match := _PROFILE_H1.fullmatch(preamble[0]))
            and re.search(r"角色|人物|人设|档案|character|profile", match.group(1), re.I)
        )
    ):
        return chunk_document(document, max_chars, overlap_lines=0)

    names: list[str] = []
    for index in headings:
        match = _PROFILE_H2.fullmatch(lines[index])
        if match is None:
            return chunk_document(document, max_chars, overlap_lines=0)
        name = match.group(1)
        if unicodedata.normalize("NFKC", name).casefold() in _PROFILE_GENERIC_HEADINGS:
            return chunk_document(document, max_chars, overlap_lines=0)
        role_title = match.group(2)
        if role_title and (
            _PROFILE_ROLE_TITLE.fullmatch(role_title) is None
            or _PROFILE_NON_PERSON_NAME.fullmatch(name)
        ):
            return chunk_document(document, max_chars, overlap_lines=0)
        names.append(name)
    normalized_names = [unicodedata.normalize("NFKC", name).casefold() for name in names]
    if len(set(normalized_names)) != len(names):
        return chunk_document(document, max_chars, overlap_lines=0)

    if any(name in "\n".join(preamble) for name in names):
        return chunk_document(document, max_chars, overlap_lines=0)
    boundaries = [*headings, len(lines)]
    for section_index, name in enumerate(names):
        body = lines[boundaries[section_index] + 1 : boundaries[section_index + 1]]
        if any(_PROFILE_H1.fullmatch(line) for line in body):
            return chunk_document(document, max_chars, overlap_lines=0)
        if not any(name in line and len(line.strip()) >= len(name) + 4 for line in body):
            return chunk_document(document, max_chars, overlap_lines=0)
        if any(other in line for other in names if other != name for line in body):
            return chunk_document(document, max_chars, overlap_lines=0)

    chunks: list[DocumentChunk] = []
    for section_index in range(len(names)):
        start = 0 if section_index == 0 else boundaries[section_index]
        end = boundaries[section_index + 1]
        numbered = [(index + 1, lines[index]) for index in range(start, end)]
        chunks.extend(
            _chunk_numbered_rows(
                document,
                numbered,
                max_chars,
                overlap_lines=0,
                first_chunk_index=len(chunks),
            )
        )
    return chunks


def _chunk_numbered_rows(
    document: DocumentInput,
    numbered_rows: list[tuple[int, str]],
    max_chars: int,
    *,
    overlap_lines: int,
    first_chunk_index: int = 0,
) -> list[DocumentChunk]:

    chunks: list[DocumentChunk] = []
    current: list[tuple[int, str]] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        chunks.append(_make_chunk(document, first_chunk_index + len(chunks), current))
        overlap: list[tuple[int, str]] = []
        if overlap_lines and len({line for line, _ in current}) > 1:
            wanted = set(sorted({line for line, _ in current})[-overlap_lines:])
            overlap = [row for row in current if row[0] in wanted]
            while overlap and sum(len(text) for _, text in overlap) + max(0, len(overlap) - 1) >= max_chars:
                overlap.pop(0)
        current = overlap
        current_chars = sum(len(text) for _, text in current) + max(0, len(current) - 1)

    for line_number, line in numbered_rows:
        if len(line) > max_chars:
            flush()
            current = []
            current_chars = 0
            for offset in range(0, len(line), max_chars):
                fragment = line[offset : offset + max_chars]
                chunks.append(
                    _make_chunk(
                        document,
                        first_chunk_index + len(chunks),
                        [(line_number, fragment)],
                    )
                )
            continue
        added = len(line) + (1 if current else 0)
        if current and current_chars + added > max_chars:
            flush()
            added = len(line) + (1 if current else 0)
        current.append((line_number, line))
        current_chars += added
    flush()
    return chunks


def numbered_chunk(chunk: DocumentChunk) -> str:
    lines = chunk.content.splitlines() or [chunk.content]
    if chunk.global_line_start == chunk.global_line_end:
        return "\n".join(f"{chunk.global_line_start}: {line}" for line in lines)
    return "\n".join(
        f"{chunk.global_line_start + index}: {line}"
        for index, line in enumerate(lines)
    )
