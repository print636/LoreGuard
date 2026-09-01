from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import DocumentInput


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

    chunks: list[DocumentChunk] = []
    current: list[tuple[int, str]] = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        chunks.append(_make_chunk(document, len(chunks), current))
        overlap: list[tuple[int, str]] = []
        if overlap_lines and len({line for line, _ in current}) > 1:
            wanted = set(sorted({line for line, _ in current})[-overlap_lines:])
            overlap = [row for row in current if row[0] in wanted]
            while overlap and sum(len(text) for _, text in overlap) + max(0, len(overlap) - 1) >= max_chars:
                overlap.pop(0)
        current = overlap
        current_chars = sum(len(text) for _, text in current) + max(0, len(current) - 1)

    for line_number, line in enumerate(lines, start=1):
        if len(line) > max_chars:
            flush()
            current = []
            current_chars = 0
            for offset in range(0, len(line), max_chars):
                fragment = line[offset : offset + max_chars]
                chunks.append(_make_chunk(document, len(chunks), [(line_number, fragment)]))
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
