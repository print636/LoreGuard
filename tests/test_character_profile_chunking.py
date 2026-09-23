from pathlib import Path

import pytest

from app.chunking import chunk_character_profile_document, chunk_document
from app.pipeline import DocumentInput


_ROOT = Path(__file__).resolve().parents[1]


def _profile(content: str, *, role: str = "character_profile") -> DocumentInput:
    return DocumentInput("profile-id", "profiles.md", content, role=role)


@pytest.mark.parametrize(
    "relative_path,expected_headings",
    (
        (
            "data/character-ooc-transfer-v1/02-character-profiles.md",
            ("## 霁青", "## 闻棠", "## 舟遥"),
        ),
        (
            "data/character-ooc-challenge-v1/02-character-profiles.md",
            ("## 南枝", "## 郁礼", "## 岑烛"),
        ),
        (
            "data/character-continuity-demo/02-character-profiles.md",
            ("## 林澈", "## 苏弦", "## 祁雾"),
        ),
    ),
)
def test_named_profile_sections_preserve_every_line_and_global_offsets(
    relative_path: str, expected_headings: tuple[str, ...]
):
    content = (_ROOT / relative_path).read_text(encoding="utf-8")
    chunks = chunk_character_profile_document(_profile(content), 8_000)

    assert len(chunks) == len(expected_headings)
    assert [chunk.id for chunk in chunks] == [
        f"profile-id:chunk:{index}" for index in range(len(chunks))
    ]
    assert "\n".join(chunk.content for chunk in chunks) == "\n".join(
        content.splitlines()
    )
    lines = content.splitlines()
    for chunk, heading in zip(chunks, expected_headings, strict=True):
        assert heading in chunk.content
        assert chunk.content.count("## ") == 1
        assert chunk.global_line_start == (
            1 if chunk is chunks[0] else lines.index(heading) + 1
        )
        assert chunk.global_line_end == (
            len(lines)
            if chunk is chunks[-1]
            else lines.index(expected_headings[expected_headings.index(heading) + 1])
        )


@pytest.mark.parametrize(
    "content",
    (
        "# 角色档案\n\n霁青的核心性格是谨慎。\n## 霁青\n霁青谨慎。\n## 闻棠\n闻棠外向。",
        "# 角色档案\n## 霁青\n霁青谨慎。\n## 霁青\n霁青外向。",
        "# 角色档案\n## 性格\n性格表现稳定。\n## 偏好\n偏好食物明确。",
        "# 角色档案\n## 霁青\n霁青谨慎。\n## 闻棠\n闻棠会与霁青讨论。",
        "# 角色档案\n```md\n## 霁青\n```\n## 闻棠\n闻棠外向。",
        "# 角色档案\n## 霁青\n霁青谨慎。\n# 第二组\n## 闻棠\n闻棠外向。",
    ),
)
def test_ambiguous_profile_layout_uses_original_chunking(content: str):
    document = _profile(content)
    assert chunk_character_profile_document(document, 8_000) == chunk_document(
        document, 8_000, overlap_lines=0
    )


def test_only_character_profile_role_uses_character_sections():
    content = "# 角色档案\n## 霁青\n霁青始终谨慎。\n## 闻棠\n闻棠始终外向。"
    document = _profile(content, role="canon")
    assert chunk_character_profile_document(document, 8_000) == chunk_document(
        document, 8_000, overlap_lines=0
    )


def test_long_line_keeps_existing_safe_fragments_and_global_line_number():
    long_fact = "霁青" + "谨慎" * 45
    content = f"# 角色档案\n## 霁青\n{long_fact}\n## 闻棠\n闻棠一向外向。"
    chunks = chunk_character_profile_document(_profile(content), 32)

    assert [chunk.id for chunk in chunks] == [
        f"profile-id:chunk:{index}" for index in range(len(chunks))
    ]
    fragments = [
        chunk.content
        for chunk in chunks
        if chunk.global_line_start == chunk.global_line_end == 3
    ]
    assert len(fragments) > 1
    assert "".join(fragments) == long_fact
    assert any(chunk.global_line_start == 4 and "## 闻棠" in chunk.content for chunk in chunks)


def test_invalid_chunk_size_keeps_existing_validation_contract():
    with pytest.raises(ValueError, match="at least 32"):
        chunk_character_profile_document(_profile("## 霁青\n霁青谨慎。"), 31)
