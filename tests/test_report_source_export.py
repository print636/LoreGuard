"""Frozen source annotations in the existing Markdown download, without writes."""
from sqlalchemy import select
import pytest
from unittest.mock import patch
from uuid import uuid4

from app.db import (AnalysisRunCharacterTraitInputRow, AnalysisRunInputRow,
                    DocumentRow, IssueRow)
from app.narrative_context import payload_sha256
from tests.test_report_evidence_preview_api import evidence_api, _read_only, _span


def exported(state, *, run_id=None):
    with _read_only(state):
        response = state.client.get(f"/api/v1/analysis-runs/{run_id or state.run.id}/export.md")
    return response


def test_export_independently_verifies_same_name_old_confirmation_and_current_run(evidence_api):
    state = evidence_api
    with state.sessions() as db:
        issue = db.get(IssueRow, state.trait_issue.id)
        issue.title = "同名双版本"
        issue.evidence = [issue.evidence[0], _span(state.replacement, 2)]
        db.get(IssueRow, state.trait_review.id).title = "不能导出为正式问题的线索"
        db.commit()
        before = [(r.id, r.evidence, r.extra, r.report_class) for r in db.scalars(
            select(IssueRow).where(IssueRow.run_id == state.run.id)).all()]
    response = exported(state)
    assert response.status_code == 200, response.text
    text = response.text
    assert "baseline\\.md，第 2 行，版本 1（已核对该版本原文）" in text
    assert "baseline\\.md，第 2 行，版本 2（已核对该版本原文）" in text
    assert "draft\\.md，第 14–16 行，版本 1（已核对该版本原文）" in text
    assert text.count("冻结基准证据。") == 2
    assert "一致性问题：2 条" in text and "不能导出为正式问题的线索" not in text
    assert "不表示问题判断必然正确" in text
    assert response.headers["Cache-Control"] == "private, no-store"
    for secret_metadata in (state.original.id, state.original.document_id, state.trait.id,
                            state.original.content_sha256, state.frozen.content_sha256):
        assert secret_metadata not in text
    with state.sessions() as db:
        current = db.get(DocumentRow, state.document.id)
        current.version, current.content = 9, "当前新版不得冒充报告旧原文"
        db.commit()
    assert exported(state).text == text
    with state.sessions() as db:
        assert [(r.id, r.evidence, r.extra, r.report_class) for r in db.scalars(
            select(IssueRow).where(IssueRow.run_id == state.run.id)).all()] == before
    assert not any("FROM documents" in query for query in state.statements)


@pytest.mark.parametrize("case, expected", [
    ("missing", "版本未核实（未核验：审查时保存的原文缺失）"),
    ("changed_body", "版本未核实（未核验：审查时保存的原文或版本无法核对）"),
    ("changed_old_version", "版本未核实（未核验：审查时保存的原文或版本无法核对）"),
    ("old_excerpt_only", "版本 1（仅能核对已保存的引文，完整旧稿暂不可用）"),
])
def test_export_retains_saved_excerpt_and_explains_unverified_source(evidence_api, case, expected):
    state = evidence_api
    with state.sessions() as db:
        if case == "missing":
            db.delete(db.get(AnalysisRunInputRow, state.frozen.id))
        elif case == "changed_body":
            db.get(AnalysisRunInputRow, state.frozen.id).content = "损坏内容不应导出"
        elif case == "changed_old_version":
            db.get(AnalysisRunInputRow, state.original.id).document_version = 7
        else:
            db.delete(db.get(AnalysisRunInputRow, state.original.id))
        db.commit()
    response = exported(state)
    assert response.status_code == 200, response.text
    assert expected in response.text
    assert "冻结基准证据。" in response.text
    assert "第十四行 😀" in response.text
    assert "损坏内容不应导出" not in response.text
    assert "一致性问题：2 条" in response.text


def test_export_does_not_borrow_another_project_source_or_expose_its_context(evidence_api):
    state = evidence_api
    with state.sessions() as db:
        source = db.get(AnalysisRunInputRow, state.foreign.id)
        source.content += "\n未授权完整上下文不得读取"
        import hashlib
        source.content_sha256 = hashlib.sha256(source.content.encode()).hexdigest()
        evidence = [{**_span(source, 1), "input_id":source.id, "document_version":source.document_version,
                     "content_sha256":source.content_sha256}]
        trait = db.get(AnalysisRunCharacterTraitInputRow, state.trait.id)
        trait.payload = {**trait.payload, "evidence":evidence, "evidence_sha256":payload_sha256(evidence)}
        trait.payload_sha256 = payload_sha256(trait.payload)
        issue = db.get(IssueRow, state.trait_issue.id)
        issue.evidence = [_span(source, 1), issue.evidence[1]]
        db.commit()
    response = exported(state)
    assert response.status_code == 200, response.text
    assert "版本未核实（未核验：来源不可用）" in response.text
    # The owned report's pre-existing excerpt remains; no foreign context is read.
    assert "另一工作区正文。" in response.text
    assert "未授权完整上下文不得读取" not in response.text
    assert exported(state, run_id=state.foreign_run.id).status_code == 404


def test_export_renderer_without_verified_annotations_does_not_guess_from_input_name():
    from app.report_export import render_markdown_report
    text = render_markdown_report(project_name="项目", run_id="run", completed_at=None,
        input_documents=[dict(document_name="同名.md",document_version=8)],
        issues=[dict(id="i",evidence=[dict(document_name="同名.md",line_start=2,line_end=2,text="保留引文")])],
        latest_feedback={},clarifications=[],diagnostics=None)
    assert "版本未核实（未核验：尚未核对审查时保存的原文）" in text
    assert "第 2 行，版本 8" not in text
    assert "> 保留引文" in text


def test_export_reuses_request_index_but_checks_each_saved_excerpt(evidence_api):
    state = evidence_api
    with state.sessions() as db:
        original = db.get(IssueRow, state.trait_issue.id)
        db.add(IssueRow(id=str(uuid4()), run_id=state.run.id, report_class="formal",
            category=original.category, severity=original.severity, confidence=original.confidence,
            title="另一条确认引用", explanation=original.explanation, suggestion=original.suggestion,
            evidence=list(original.evidence), extra=dict(original.extra)))
        ordinary = db.get(IssueRow, state.formal.id)
        ordinary.evidence = [ordinary.evidence[0], {**ordinary.evidence[0], "text":"原文中不存在的报告摘录"}]
        db.commit()
    import app.report_evidence_preview as preview
    with patch.object(preview, "_character_review_snapshot_evidence_index",
                      wraps=preview._character_review_snapshot_evidence_index) as index:
        response = exported(state)
    assert response.status_code == 200, response.text
    assert index.call_count == 1
    assert "证据 1：draft\\.md，第 14–16 行，版本 1（已核对该版本原文）" in response.text
    assert "证据 2：draft\\.md，第 14–16 行，版本未核实（未核验：审查时保存的原文或版本无法核对）" in response.text
    assert "> 原文中不存在的报告摘录" in response.text
    assert "不是 Word 页码或排版行号" in response.text
