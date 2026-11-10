"""Explicit author mapping: real API/DB transactions, never model calls."""
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json

import pytest
from sqlalchemy import func, select

from app import main
from app.db import AnalysisRunInputRow, AnalysisRunRow, DocumentRow, DocumentUploadReceiptRow
from app.service import capture_run_inputs
from app.document_upload_receipts import upload_request_sha256
from tests.test_docx_import import make_docx
from tests.test_project_import_foundation_api import foundation_api, _upload


def test_renamed_markdown_keeps_word_logical_chain_context_snapshot_and_recheck(foundation_api):
    s = foundation_api
    confirmed_draft = json.dumps({"resolution_state": "confirmed", "publication_status": "draft", "scope": {"timeline_key": "main"}})
    original = _upload(s, key="original", name="原章.docx",
                       body=make_docx("<w:p><w:r><w:t>旧正文</w:t></w:r></w:p>"),
                       fields={"document_role": "chapter", "story_scope": "branch_a", "narrative_context": confirmed_draft}).json()
    unrelated = _upload(s, key="unrelated", name="编辑后的稿.md", body=b"unrelated").json()
    with s.sessions() as db:
        run = AnalysisRunRow(project_id=s.project.id, status="completed", batch_mode="draft_review",
                             batch_coverage={"selected_document_ids": [original["id"]], "no_formal_context_expected": True})
        db.add(run); db.flush()
        frozen = capture_run_inputs(db, run, [db.get(DocumentRow, original["id"])],
                                    batch_roles={original["id"]: "target"})[0]
        db.commit()
        run_id, input_id = run.id, frozen.id
    response = _upload(s, key="mapped", name="编辑后的稿.md", body="新正文\n第二行".encode(),
                       fields={"replace_document_id": original["id"], "narrative_context": confirmed_draft})
    assert response.status_code == 201, response.text
    row = response.json()
    assert (row["name"], row["version"], row["uploaded_filename"], row["uploaded_format"]) == ("原章.docx", 2, "编辑后的稿.md", "markdown")
    assert row["replaces_document_id"] == original["id"] and row["replaces_document_version"] == 1
    assert row["superseded_document_ids"] == [original["id"]]
    assert row["document_role"] == "chapter" and row["story_scope"] == "branch_a"
    assert row["upload_metadata_source"] == "stored_receipt"
    with s.sessions() as db:
        assert db.get(DocumentRow, row["id"]).content == "新正文\n第二行"
        assert db.get(DocumentRow, unrelated["id"]).active
        snapshot = db.get(AnalysisRunInputRow, input_id)
        assert (snapshot.document_name, snapshot.document_version, snapshot.content) == ("原章.docx", 1, "旧正文")
        assert snapshot.content_sha256 == sha256("旧正文".encode()).hexdigest()
        _, selected, roles, _ = main._recheck_material_selection(db, db.get(AnalysisRunRow, run_id), {})
        assert [d.id for d in selected if roles[d.id] == "target"] == [row["id"]]
        receipt = db.scalar(select(DocumentUploadReceiptRow).where(DocumentUploadReceiptRow.document_id == row["id"]))
        assert receipt.import_metadata == {
            "schema_version": "document-import-v1", "uploaded_filename": "编辑后的稿.md", "uploaded_format": "markdown",
            "logical_name": "原章.docx", "replaces_document_id": original["id"], "replaces_document_version": 1,
        }


def test_actual_docx_parser_and_non_key_client_keep_logical_markdown_name(foundation_api):
    s = foundation_api
    original = _upload(s, name="chapter.md").json()
    mapped = _upload(s, key=None, name="外部编辑.docx",
                     body=make_docx("<w:p><w:r><w:t>Word 解析文本</w:t></w:r></w:p>"),
                     fields={"replace_document_id": original["id"]})
    assert mapped.status_code == 201
    row = mapped.json()
    assert (row["name"], row["uploaded_format"], row["upload_metadata_source"]) == ("chapter.md", "docx", "current_request")
    assert "deduplicated" not in row
    with s.sessions() as db:
        assert db.get(DocumentRow, row["id"]).content == "Word 解析文本"
        assert db.scalar(select(func.count()).select_from(DocumentUploadReceiptRow)) == 1
    invalid = _upload(s, key="bad-docx", name="invalid.docx", body=b"not a docx",
                      fields={"replace_document_id": row["id"]})
    assert invalid.status_code == 400
    with s.sessions() as db:
        assert db.get(DocumentRow, row["id"]).active


def test_replay_precedes_active_guard_but_new_operation_cannot_redirect_stale_target(foundation_api):
    s = foundation_api
    target = _upload(s, key="old").json()
    args = {"key": "mapped", "name": "changed.md", "fields": {"replace_document_id": target["id"]}}
    first = _upload(s, **args).json()
    later = _upload(s, key="later", name="chapter.md", body=b"v3").json()
    replay = _upload(s, **args)
    assert replay.status_code == 201 and replay.json()["id"] == first["id"]
    assert replay.json()["deduplicated"] and not replay.json()["active"]
    assert replay.json()["upload_metadata_source"] == "stored_receipt"
    stale = _upload(s, **{**args, "key": "new-operation"})
    assert stale.status_code == 409 and stale.json()["detail"]["code"] == "replacement_target_stale"
    changed = _upload(s, **{**args, "fields": {"replace_document_id": later["id"]}})
    assert changed.status_code == 409 and changed.json()["detail"]["reason_code"] == "document_upload_idempotency_conflict"
    with s.sessions() as db:
        assert db.scalar(select(func.count()).select_from(DocumentRow)) == 3


@pytest.mark.parametrize("other_workspace", [False, True])
def test_explicit_target_must_belong_to_authorized_project(foundation_api, other_workspace):
    s = foundation_api
    with s.sessions() as db:
        target = DocumentRow(project_id=s.other.id if other_workspace else s.second.id,
                             name="secret.md", content="private source", version=1)
        db.add(target); db.commit(); target_id = target.id
    response = _upload(s, fields={"replace_document_id": target_id})
    assert response.status_code == 404 and "private source" not in response.text


@pytest.mark.parametrize("same_key", [True, False])
def test_explicit_replacement_concurrency_keeps_one_successful_new_version(foundation_api, same_key):
    s = foundation_api
    original = _upload(s, key="original").json()
    def submit(index):
        return _upload(s, key="same" if same_key else f"operation-{index}", name="new-name.md",
                       fields={"replace_document_id": original["id"]})
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(submit, range(2)))
    assert sorted(r.status_code for r in responses) == ([201, 201] if same_key else [201, 409])
    if same_key:
        assert len({r.json()["id"] for r in responses}) == 1
        assert sum(not r.json()["deduplicated"] for r in responses) == 1
    else:
        assert next(r for r in responses if r.status_code == 409).json()["detail"]["code"] == "replacement_target_stale"
    with s.sessions() as db:
        rows = db.scalars(select(DocumentRow).order_by(DocumentRow.version)).all()
        assert [(r.name, r.version, r.active) for r in rows] == [("chapter.md", 1, False), ("chapter.md", 2, True)]
        assert db.scalar(select(func.count()).select_from(DocumentUploadReceiptRow)) == 2


def test_failed_mapped_receipt_rolls_back_target_and_metadata(foundation_api, monkeypatch):
    s = foundation_api
    original = _upload(s, key="original").json()
    factory = main.DocumentUploadReceiptRow
    monkeypatch.setattr(main, "DocumentUploadReceiptRow", lambda **values: factory(**{**values, "request_sha256": "bad"}))
    response = _upload(s, key="mapped", name="different.md", fields={"replace_document_id": original["id"]})
    assert response.status_code == 409
    with s.sessions() as db:
        assert db.get(DocumentRow, original["id"]).active
        assert db.scalar(select(func.count()).select_from(DocumentRow)) == 1
        assert db.scalar(select(func.count()).select_from(DocumentUploadReceiptRow)) == 1


def test_old_null_receipt_echoes_verified_request_without_backfill(foundation_api):
    s = foundation_api
    target = _upload(s, key="old").json()
    args = {"key": "mapped", "fields": {"replace_document_id": target["id"]}}
    original = _upload(s, **args).json()
    with s.sessions() as db:
        receipt = db.scalar(select(DocumentUploadReceiptRow).where(DocumentUploadReceiptRow.document_id == original["id"]))
        receipt.import_metadata = None; db.commit(); receipt_id = receipt.id
    replay = _upload(s, **args)
    assert replay.status_code == 201 and replay.json()["id"] == original["id"]
    assert replay.json()["upload_metadata_source"] == "verified_request"
    assert replay.json()["replaces_document_id"] == target["id"]
    with s.sessions() as db:
        assert db.get(DocumentUploadReceiptRow, receipt_id).import_metadata is None


def test_mismatched_persisted_upload_metadata_does_not_claim_safe_replay(foundation_api):
    s = foundation_api
    first = _upload(s).json()
    with s.sessions() as db:
        receipt = db.scalar(select(DocumentUploadReceiptRow).where(DocumentUploadReceiptRow.document_id == first["id"]))
        receipt.import_metadata = {**receipt.import_metadata, "uploaded_format": "docx"}; db.commit()
    response = _upload(s)
    assert response.status_code == 409 and response.json()["detail"]["reason_code"] == "document_upload_receipt_unavailable"
    with s.sessions() as db:
        assert db.scalar(select(func.count()).select_from(DocumentRow)) == 1


def test_old_null_receipt_with_inactive_requested_target_does_not_invent_replacement(foundation_api):
    s = foundation_api
    old = _upload(s, key="v1").json()
    actual_target = _upload(s, key="v2").json()
    final = _upload(s, key="legacy").json()
    # Pre-change code allowed a stale explicit target while superseding the
    # current same-name version. Reconstruct that saved receipt, not a new import.
    digest = upload_request_sha256(filename="chapter.md", data=b"synthetic original",
                                   replace_document_id=old["id"], document_role=None, story_scope=None, narrative_context=None)
    with s.sessions() as db:
        receipt = db.scalar(select(DocumentUploadReceiptRow).where(DocumentUploadReceiptRow.document_id == final["id"]))
        receipt.import_metadata, receipt.request_sha256 = None, digest
        db.commit()
    response = _upload(s, key="legacy", fields={"replace_document_id": old["id"]})
    assert response.status_code == 201 and response.json()["id"] == final["id"]
    assert response.json()["superseded_document_ids"] == [actual_target["id"]]
    assert response.json()["replaces_document_id"] is None and response.json()["replaces_document_version"] is None
    assert response.json()["upload_metadata_source"] == "verified_request"
