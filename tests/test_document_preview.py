from hashlib import sha256

import pytest

from app.document_preview import build_document_preview


def test_preview_preserves_blank_lines_and_evidence_line_numbers():
    content = "首行\r\n\r\n第三行\n末行\n"
    result = build_document_preview(content, offset=1, limit=2)
    assert result["document"] == {
        "char_count": len(content),
        "line_count": 4,
        "content_sha256": sha256(content.encode("utf-8")).hexdigest(),
    }
    assert result["lines"] == [
        {"line_number": 2, "text": ""},
        {"line_number": 3, "text": "第三行"},
    ]
    assert result["page"] == {
        "offset": 1, "limit": 2, "total": 4, "has_more": True,
    }
    assert result["query"] == ""


@pytest.mark.parametrize("offset", [3, 4, 50])
def test_preview_last_page_and_offsets_past_end(offset):
    result = build_document_preview("一\n二\n三\n四", offset=offset, limit=2)
    expected = [{"line_number": 4, "text": "四"}] if offset == 3 else []
    assert result["lines"] == expected
    assert result["page"] == {
        "offset": offset, "limit": 2, "total": 4, "has_more": False,
    }


@pytest.mark.parametrize("content, expected", [
    ("", []),
    ("\n", [{"line_number": 1, "text": ""}]),
    ("\n\n", [{"line_number": 1, "text": ""}, {"line_number": 2, "text": ""}]),
])
def test_preview_empty_content_and_empty_source_lines(content, expected):
    result = build_document_preview(content)
    assert result["lines"] == expected
    assert result["document"]["line_count"] == len(expected)
    assert result["page"] == {
        "offset": 0, "limit": 120, "total": len(expected), "has_more": False,
    }


def test_search_trims_chinese_query_and_pages_original_line_numbers():
    result = build_document_preview(
        "林澈登场。\n无关\n\n林澈说话。\n林澈离开。",
        offset=1,
        limit=1,
        query="  林澈 \t",
    )
    assert result["query"] == "林澈"
    assert result["lines"] == [{"line_number": 4, "text": "林澈说话。"}]
    assert result["page"] == {
        "offset": 1, "limit": 1, "total": 3, "has_more": True,
    }
    assert result["document"]["line_count"] == 5


def test_search_uses_unicode_casefold():
    result = build_document_preview("Straße\nSTRASSE\n其他", query="strasse")
    assert result["lines"] == [
        {"line_number": 1, "text": "Straße"},
        {"line_number": 2, "text": "STRASSE"},
    ]
    assert result["page"]["total"] == 2


def test_search_treats_regex_characters_as_literal_text():
    result = build_document_preview("alpha\n[a.*] 字面\naaa\n.* 字面", query=".*")
    assert result["lines"] == [
        {"line_number": 2, "text": "[a.*] 字面"},
        {"line_number": 4, "text": ".* 字面"},
    ]
    assert result["page"]["total"] == 2


def test_search_does_not_match_across_source_lines():
    result = build_document_preview("一\n二", query="一\n二")
    assert result["lines"] == []
    assert result["page"]["total"] == 0
    assert result["page"]["has_more"] is False


def test_whitespace_query_returns_unfiltered_page():
    result = build_document_preview("一\n\n三", query=" \t ")
    assert result["query"] == ""
    assert result["page"]["total"] == 3
    assert [line["line_number"] for line in result["lines"]] == [1, 2, 3]


def test_preview_enforces_maximum_page_and_query_lengths():
    result = build_document_preview("\n".join("x" * 160 for _ in range(201)),
                                    limit=200, query="x" * 160)
    assert len(result["lines"]) == 200
    assert result["page"]["total"] == 201
    assert result["page"]["has_more"] is True


@pytest.mark.parametrize("parameters", [
    {"offset": -1}, {"limit": 0}, {"limit": 201}, {"query": "x" * 161},
])
def test_preview_rejects_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        build_document_preview("正文", **parameters)
