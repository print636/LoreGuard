from __future__ import annotations

from hashlib import sha256
from typing import Any


def build_document_preview(
    content: str,
    *,
    offset: int = 0,
    limit: int = 120,
    query: str = "",
) -> dict[str, Any]:
    """Read a page of stored lines, preserving the evidence line numbering."""
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")
    if len(query) > 160:
        raise ValueError("query must contain at most 160 characters")

    query = query.strip()
    needle = query.casefold()
    source_lines = content.splitlines()
    lines: list[dict[str, Any]] = []
    total = 0
    for line_number, text in enumerate(source_lines, start=1):
        if needle and needle not in text.casefold():
            continue
        if offset <= total < offset + limit:
            lines.append({"line_number": line_number, "text": text})
        total += 1

    return {
        "document": {
            "char_count": len(content),
            "line_count": len(source_lines),
            "content_sha256": sha256(content.encode("utf-8")).hexdigest(),
        },
        "lines": lines,
        "page": {
            "offset": offset,
            "limit": limit,
            "total": total,
            "has_more": offset + len(lines) < total,
        },
        "query": query,
    }
