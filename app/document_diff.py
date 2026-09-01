from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any


def _clip_lines(content: str, max_lines: int, max_chars: int) -> tuple[list[str], int, bool]:
    """Return a bounded line view without silently treating it as the whole file."""
    all_lines = content.splitlines()
    selected: list[str] = []
    char_count = 0
    for line in all_lines:
        # Include the logical newline in the processing budget.
        cost = len(line) + 1
        if len(selected) >= max_lines or (selected and char_count + cost > max_chars):
            break
        if not selected and cost > max_chars:
            selected.append(line[:max_chars])
            char_count = max_chars
            break
        selected.append(line)
        char_count += cost
    return selected, len(all_lines), len(selected) < len(all_lines) or len(content) > max_chars


def build_document_diff(
    old_content: str,
    new_content: str,
    *,
    max_lines: int = 20_000,
    max_chars: int = 2_000_000,
    max_output_lines: int = 4_000,
    context_lines: int = 3,
) -> dict[str, Any]:
    """Build a bounded, deterministic line diff without any model/provider call."""
    old_lines, old_total_lines, old_clipped = _clip_lines(old_content, max_lines, max_chars)
    new_lines, new_total_lines, new_clipped = _clip_lines(new_content, max_lines, max_chars)
    matcher = SequenceMatcher(None, old_lines, new_lines, autojunk=True)
    opcodes = matcher.get_opcodes()

    added = sum(j2 - j1 for tag, _, _, j1, j2 in opcodes if tag in {"insert", "replace"})
    removed = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag in {"delete", "replace"})
    unchanged = sum(i2 - i1 for tag, i1, i2, _, _ in opcodes if tag == "equal")
    groups = list(matcher.get_grouped_opcodes(n=context_lines))

    hunks: list[dict[str, Any]] = []
    emitted = 0
    output_truncated = False
    for group in groups:
        first, last = group[0], group[-1]
        hunk: dict[str, Any] = {
            "old_start": first[1] + 1,
            "old_lines": last[2] - first[1],
            "new_start": first[3] + 1,
            "new_lines": last[4] - first[3],
            "lines": [],
        }
        for tag, i1, i2, j1, j2 in group:
            rows: list[tuple[str, str, int | None, int | None]] = []
            if tag == "equal":
                rows = [("unchanged", old_lines[index], index + 1, j1 + offset + 1) for offset, index in enumerate(range(i1, i2))]
            else:
                if tag in {"delete", "replace"}:
                    rows.extend(("removed", old_lines[index], index + 1, None) for index in range(i1, i2))
                if tag in {"insert", "replace"}:
                    rows.extend(("added", new_lines[index], None, index + 1) for index in range(j1, j2))
            for kind, content, old_line, new_line in rows:
                if emitted >= max_output_lines:
                    output_truncated = True
                    break
                hunk["lines"].append(
                    {"type": kind, "content": content, "old_line": old_line, "new_line": new_line}
                )
                emitted += 1
            if output_truncated:
                break
        if hunk["lines"]:
            hunks.append(hunk)
        if output_truncated:
            break

    warnings: list[str] = []
    if old_clipped or new_clipped:
        warnings.append(
            f"文档过大，本次仅比较每个版本前 {max_lines} 行且最多 {max_chars} 个字符；摘要只代表该范围。"
        )
    if output_truncated:
        warnings.append(
            f"差异展示超过 {max_output_lines} 行，响应已截断；摘要仍覆盖本次实际比较范围。"
        )

    return {
        "summary": {
            "added_lines": added,
            "removed_lines": removed,
            "unchanged_lines": unchanged,
            "changed_hunks": len(groups),
            "compared_old_lines": len(old_lines),
            "compared_new_lines": len(new_lines),
            "old_total_lines": old_total_lines,
            "new_total_lines": new_total_lines,
            "input_truncated": old_clipped or new_clipped,
            "output_truncated": output_truncated,
        },
        "hunks": hunks,
        "warnings": warnings,
    }
