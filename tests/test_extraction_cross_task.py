from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.run_extraction_cross_task import (
    FIXTURE,
    _documents,
    _score_case,
    _selected_cases,
)


def _issue(category: str, chapter_line: int = 5):
    return SimpleNamespace(
        category=SimpleNamespace(value=category),
        evidence=[
            SimpleNamespace(document_name="canon.md", line_start=3),
            SimpleNamespace(document_name="chapter.md", line_start=chapter_line),
        ],
    )


def test_transfer_runner_split_and_document_input_do_not_include_labels():
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    cases = _selected_cases(manifest, "dev")
    assert len(cases) == manifest["splits"]["dev"]["case_count"]
    documents, names = _documents(cases[0])
    assert names == {"canon": "canon.md", "chapter": "chapter.md"}
    assert len(documents) == 2
    assert all(document.role in {"canon", "chapter"} for document in documents)
    assert documents[0].content == (
        FIXTURE / cases[0]["sources"][0]["path"]
    ).read_text(encoding="utf-8")


def test_transfer_runner_requires_category_and_complete_evidence_set():
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    positive = next(
        case for case in _selected_cases(manifest, "holdout")
        if case["case_id"] == "EIL-H-01"
    )
    _, names = _documents(positive)
    expected_category = positive["expected"]["issue_category"]
    assert _score_case(positive, [_issue(expected_category)], names)[
        "exact_evidence_hit"
    ] is True
    assert _score_case(positive, [_issue(expected_category, 6)], names)[
        "exact_evidence_hit"
    ] is False
    assert _score_case(positive, [_issue("world_rule_conflict")], names)[
        "exact_evidence_hit"
    ] is False

    negative = next(
        case for case in _selected_cases(manifest, "holdout")
        if case["case_id"] == "EIL-H-06"
    )
    _, names = _documents(negative)
    assert _score_case(negative, [], names)["clean_abstain"] is True
    assert _score_case(negative, [_issue("location_collision")], names)[
        "clean_abstain"
    ] is False
