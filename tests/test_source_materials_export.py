"""Explicit frozen text download: no provider, analysis or live source reads."""
from hashlib import sha256
from io import BytesIO
from zipfile import ZipFile

import pytest

from app.db import (AnalysisRunInputContextRow, AnalysisRunInputNarrativeContextRow,
                    AnalysisRunInputRow, AnalysisRunRow, DocumentRow)
from app.narrative_context import context_snapshot_payload, derive_authority_tier, payload_sha256
from tests.test_report_evidence_preview_api import evidence_api, _read_only


def download(state, ids, *, run_id=None):
    with _read_only(state):
        return state.client.get(f"/api/v1/analysis-runs/{run_id or state.run.id}/source-materials.zip",
                                params=[("document_id", value) for value in ids])


def contents(response):
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert 'filename="LoreGuard-source-materials.zip"' in response.headers["content-disposition"]
    with ZipFile(BytesIO(response.content)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_selected_frozen_texts_keep_exact_utf8_and_context_without_implicit_old_sources(evidence_api):
    state = evidence_api
    with state.sessions() as db:
        p = context_snapshot_payload(None, document_role="chapter", story_scope="main")
        p.update(resolution_state="confirmed", publication_status="published", origin="manual",
                 authority_tier=derive_authority_tier("chapter", "confirmed", "published"))
        p["scope"] = {"schema_version":1,"timeline_key":"主线", "release":{"key":"第一稿","ordinal":1},
                      "branch":{"path":["起点","分支甲"],"exclusive_group":"互斥结局"},"activity_key":"开航"}
        p["scope_sha256"] = payload_sha256(p["scope"])
        db.add(AnalysisRunInputNarrativeContextRow(input_id=state.frozen.id, payload=p,
                                                   payload_sha256=payload_sha256(p)))
        db.commit()
    first = download(state, [state.replacement.document_id, state.frozen.document_id])
    files = contents(first)
    assert set(files) == {"资料目录.md", "001-draft.md", "002-baseline.md"}
    assert files["001-draft.md"] == state.frozen.content.encode("utf-8")  # CRLF, blank line, Unicode.
    assert files["002-baseline.md"] == state.replacement.content.encode("utf-8")
    index = files["资料目录.md"].decode()
    assert "文稿名称：draft\\.md" in index and "原版本：1" in index and "原版本：2" in index
    assert "资料身份：故事正文" in index and "适用范围：main" in index
    assert "身份确认：作者已确认" in index and "发布状态：已发布" in index
    assert "分支：起点 → 分支甲" in index and "剧情版本：第一稿" in index
    assert "资料身份：未保存" in index and "结构化时间线／分支：历史记录未保存" in index
    assert "改名或跨格式上传仍保留所选文稿名称和版本链" in index
    assert "不一定是该次实际上传文件名或格式" in index and "不会自动转换回原 Word 排版" in index
    assert "不是 Word 页码" in index
    assert state.original.content.encode() not in files.values()
    assert all(value not in index for value in (state.run.id,state.frozen.id,state.frozen.content_sha256,state.trait.id))
    with state.sessions() as db:
        current = db.get(DocumentRow, state.document.id)
        current.content, current.version = "当前新版不能替换冻结正文", 9
        db.commit()
    assert download(state, [state.frozen.document_id, state.replacement.document_id]).content == first.content
    assert not any("FROM documents" in query for query in state.statements)


def test_one_docx_text_uses_safe_zip_entry_and_never_changes_body_lines(evidence_api):
    state = evidence_api
    with state.sessions() as db:
        source = db.get(AnalysisRunInputRow, state.frozen.id)
        source.document_name = "../\\<稿件>[修订].docx"
        db.commit()
    files = contents(download(state, [state.frozen.document_id]))
    bodies = [name for name in files if name != "资料目录.md"]
    assert len(bodies) == 1 and bodies[0].startswith("001-") and bodies[0].endswith(".md")
    assert not any("/" in name or "\\" in name or ".." in name for name in files)
    assert files[bodies[0]] == state.frozen.content.encode()
    assert "&lt;稿件&gt;" in files["资料目录.md"].decode()


@pytest.mark.parametrize("case", ["missing", "hash", "context", "narrative"])
def test_one_bad_selected_source_rejects_whole_archive(evidence_api, case):
    state = evidence_api
    with state.sessions() as db:
        if case == "missing":
            db.delete(db.get(AnalysisRunInputRow, state.frozen.id))
        elif case == "hash":
            db.get(AnalysisRunInputRow, state.frozen.id).content_sha256 = "0" * 64
        elif case == "context":
            db.get(AnalysisRunInputContextRow, state.frozen.id).document_role = "unsupported"
        else:
            db.add(AnalysisRunInputNarrativeContextRow(input_id=state.frozen.id, payload={}, payload_sha256="0" * 64))
        db.commit()
    response = download(state, [state.frozen.document_id, state.replacement.document_id])
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "source_materials_unavailable"
    assert response.json()["detail"]["documents"] == [{"name":None if case == "missing" else "draft.md",
        "reason":{"missing":"missing","hash":"content_mismatch","context":"context_mismatch","narrative":"context_mismatch"}[case]}]
    assert response.headers["cache-control"] == "private, no-store"
    assert not response.content.startswith(b"PK")


def test_only_explicit_owned_run_inputs_and_completed_run_can_be_exported(evidence_api):
    state = evidence_api
    # The old confirmed source is readable through the report's binding but is
    # not automatically a selectable input of the current report.
    assert download(state, [state.original.document_id]).status_code == 200  # same ID maps only to this run's v2.
    foreign = download(state, [state.foreign.document_id])
    assert foreign.status_code == 409 and "另一工作区" not in foreign.text
    assert download(state, [state.frozen.document_id], run_id=state.foreign_run.id).status_code == 404
    assert download(state, [state.foreign.document_id], run_id=state.other_run.id).status_code == 409
    with state.sessions() as db:
        db.get(AnalysisRunRow, state.run.id).status = "running"
        db.commit()
    response = download(state, [state.frozen.document_id])
    assert response.status_code == 409 and response.json()["detail"]["code"] == "run_not_completed"


@pytest.mark.parametrize("selection", [[], ["duplicate"] * 2, ["too_many"] * 101])
def test_empty_duplicate_or_overlong_selections_are_rejected(evidence_api, selection):
    response = download(evidence_api, selection)
    assert response.status_code == 422


def test_capacity_uses_exact_utf8_body_bytes_not_character_or_docx_upload_size(evidence_api, monkeypatch):
    state = evidence_api
    body = "甲乙\r\n"
    with state.sessions() as db:
        row = db.get(AnalysisRunInputRow, state.frozen.id)
        row.content, row.content_sha256 = body, sha256(body.encode()).hexdigest()
        db.commit()
    monkeypatch.setattr("app.source_materials_export.MAX_SOURCE_MATERIAL_BYTES", len(body.encode()) - 1)
    response = download(state, [state.frozen.document_id])
    assert response.status_code == 413 and response.json()["detail"]["code"] == "source_materials_too_large"
    monkeypatch.setattr("app.source_materials_export.MAX_SOURCE_MATERIAL_BYTES", len(body.encode()))
    files = contents(download(state, [state.frozen.document_id]))
    assert files["001-draft.md"] == body.encode()
