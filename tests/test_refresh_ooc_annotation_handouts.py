"""Presentation-only reissue tests with invented public inputs, never real cases."""

import json

import pytest

from scripts import refresh_ooc_annotation_handouts as handouts


def _prepare(tmp_path):
    directory = tmp_path / "handouts"
    directory.mkdir()
    for reviewer, name in zip(("synthetic-a", "synthetic-b"), handouts.PAGE_NAMES):
        data = {
            "page_schema_version": handouts.PAGE_SCHEMA_VERSION,
            "reviewer_id": reviewer,
            "manual_version": "ooc-manual-v1",
            "cases": [{"case_id": "synthetic-case", "note": "Synthetic public input only"}],
        }
        (directory / name).write_text(
            '<html><script id="annotation-data" type="application/json">'
            + json.dumps(data)
            + "</script></html>",
            encoding="utf-8",
        )
    (directory / handouts.MANUAL_HTML).write_text("Old HTML guide", encoding="utf-8")
    (directory / handouts.MANUAL_MD).write_text("Old Markdown guide", encoding="utf-8")
    html = tmp_path / "new-guide.html"
    markdown = tmp_path / "new-guide.md"
    html.write_text("<html><h1>New guide</h1></html>", encoding="utf-8")
    markdown.write_text("# New guide", encoding="utf-8")
    return directory, html, markdown


def test_default_preflight_does_not_write_or_backup(tmp_path):
    directory, html, markdown = _prepare(tmp_path)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    report = handouts.refresh_handouts(directory, html, markdown)
    assert report["mode"] == "check-only"
    assert report["public_inputs_unchanged"] is True
    assert report["case_count_per_reviewer"] == 1
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_apply_preserves_inputs_and_creates_verified_non_overwriting_backups(tmp_path):
    directory, html, markdown = _prepare(tmp_path)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    report = handouts.refresh_handouts(directory, html, markdown, apply=True)
    backup = directory / "原版备份-易读版升级"
    assert report["mode"] == "apply"
    assert {path.name: path.read_bytes() for path in backup.iterdir()} == before
    for name in handouts.PAGE_NAMES:
        assert handouts._page_data((directory / name).read_bytes()) == handouts._page_data(before[name])
    assert (directory / handouts.MANUAL_HTML).read_bytes() == html.read_bytes()
    assert (directory / handouts.MANUAL_MD).read_bytes() == markdown.read_bytes()
    second = handouts.refresh_handouts(directory, html, markdown, apply=True)
    assert second["backup_directory"] != report["backup_directory"]
    assert {path.name: path.read_bytes() for path in backup.iterdir()} == before


def test_mismatched_cases_are_rejected_without_touching_any_file(tmp_path):
    directory, html, markdown = _prepare(tmp_path)
    target = directory / handouts.PAGE_NAMES[1]
    target.write_bytes(target.read_bytes().replace(b"synthetic-case", b"different-case"))
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises(ValueError, match="public inputs differ"):
        handouts.refresh_handouts(directory, html, markdown, apply=True)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


def test_same_reviewer_identity_is_rejected_without_backup(tmp_path):
    directory, html, markdown = _prepare(tmp_path)
    target = directory / handouts.PAGE_NAMES[1]
    target.write_bytes(target.read_bytes().replace(b"synthetic-b", b"synthetic-a"))
    with pytest.raises(ValueError, match="different reviewer identities"):
        handouts.refresh_handouts(directory, html, markdown, apply=True)
    assert len(list(directory.iterdir())) == 4
