from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import sessionmaker

from app import main
from app.auth import AuthContext, get_auth_context
from app.db import (
    Base, DocumentContextRow, DocumentNarrativeContextRevisionRow,
    DocumentRow, DocumentUploadReceiptRow, LOCAL_USER_ID, ProjectRow, WorkspaceRow,
)
from app.project_metadata import ProjectMetadataInput, metadata_update_statement
from app.project_sort import project_name_sort_key


@pytest.fixture()
def foundation_api(tmp_path, monkeypatch):
    # conftest changes cwd/environment before application imports. A second
    # disposable database here keeps every fixture independent of other tests.
    engine = create_engine(f"sqlite:///{(tmp_path / 'import.db').as_posix()}", connect_args={"check_same_thread": False})
    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _record):
        connection.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    with sessions() as db:
        workspace = WorkspaceRow(id=str(uuid4()), name="合成工作区")
        foreign = WorkspaceRow(id=str(uuid4()), name="另一个工作区")
        db.add_all([workspace, foreign]); db.flush()
        project = ProjectRow(workspace_id=workspace.id, name="原项目", description="原简介")
        second = ProjectRow(workspace_id=workspace.id, name="第二项目")
        other = ProjectRow(workspace_id=foreign.id, name="不应暴露", description="PRIVATE-FOREIGN-METADATA")
        db.add_all([project, second, other]); db.commit()
    monkeypatch.setattr(main, "SessionLocal", sessions)
    identity = AuthContext(user_id=LOCAL_USER_ID, workspace_id=workspace.id, role="owner", anonymous=True)
    monkeypatch.setitem(main.app.dependency_overrides, get_auth_context, lambda: identity)
    def forbidden(*_args, **_kwargs):
        pytest.fail("Import/metadata must never call a model or start analysis")
    for name in ("dispatch_analysis", "execute_analysis", "OpenAICompatibleProvider", "infer_narrative_context"):
        monkeypatch.setattr(main, name, forbidden)
    main.write_limiter.events.clear()
    state = SimpleNamespace(**locals())
    state.client = TestClient(main.app)  # deliberately no application lifespan
    try:
        yield state
    finally:
        state.client.close(); engine.dispose(); main.write_limiter.events.clear()


def _metadata(state, project=None):
    return f"/api/v1/projects/{project or state.project.id}/metadata"


def _upload(state, *, key="operation-1", name="chapter.md", body=b"synthetic original", fields=None, project=None):
    return state.client.post(
        f"/api/v1/projects/{project or state.project.id}/documents",
        files={"file": (name, body, "application/octet-stream")}, data=fields or {},
        headers={"Idempotency-Key": key} if key is not None else {},
    )


def test_metadata_reads_only_small_fields_and_preserves_old_contracts(foundation_api):
    state = foundation_api
    response = state.client.get(_metadata(state))
    assert response.status_code == 200
    assert set(response.json()) == {"id", "name", "description", "created_at", "metadata_revision"}
    assert response.json()["metadata_revision"] == 1
    assert response.headers["cache-control"] == "no-store"
    legacy = state.client.post("/api/v1/projects", json={"name": "旧创建入口", "description": ""})
    assert legacy.status_code == 201 and "metadata_revision" not in legacy.json()
    detail = state.client.get(f"/api/v1/projects/{state.project.id}")
    assert set(detail.json()) == {"id", "name", "description", "documents"}


def test_metadata_cas_noop_rename_sort_and_old_revision_rejection(foundation_api):
    state = foundation_api
    original = state.client.get(_metadata(state)).json()
    payload = {"name": "  原项目  ", "description": "原简介", "expected_revision": 1}
    noop = state.client.patch(_metadata(state), json=payload)
    assert noop.status_code == 200 and noop.json() == original
    payload.update(name="  安然 🕊  ", description="新简介\n多行")
    changed = state.client.patch(_metadata(state), json=payload)
    assert changed.status_code == 200 and changed.json()["metadata_revision"] == 2
    assert changed.json()["name"] == "安然 🕊" and changed.json()["created_at"] == original["created_at"]
    conflict = state.client.patch(_metadata(state), json=payload)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["reason_code"] == "project_metadata_revision_conflict"
    assert "新简介" not in conflict.text and conflict.headers["cache-control"] == "no-store"
    with state.sessions() as db:
        project = db.get(ProjectRow, state.project.id)
        assert project.name_sort_key == project_name_sort_key("安然 🕊")
        assert db.scalar(select(func.count()).select_from(DocumentRow)) == 0
    params = {"project_id": state.project.id, "page": 1, "page_size": 20}
    assert state.client.get("/api/v1/project-catalog", params=params).json()["items"][0]["name"] == "安然 🕊"


@pytest.mark.parametrize("payload", (
    {"name": " ", "description": "", "expected_revision": 1},
    {"name": "a" * 201, "description": "", "expected_revision": 1},
    {"name": "ok", "description": "", "expected_revision": True},
    {"name": "ok", "description": "", "expected_revision": 0},
    {"name": "ok", "description": "", "expected_revision": "1"},
    {"name": "ok", "description": "", "expected_revision": 1, "published": True},
))
def test_metadata_invalid_payload_is_422_no_store(foundation_api, payload):
    response = foundation_api.client.patch(_metadata(foundation_api), json=payload)
    assert response.status_code == 422 and response.headers["cache-control"] == "no-store"


def test_metadata_missing_and_foreign_are_same_404_and_csrf_origin_protected(foundation_api, monkeypatch):
    state = foundation_api
    payload = {"name": "attempt", "description": "", "expected_revision": 1}
    for method in ("get", "patch"):
        responses = [getattr(state.client, method)(_metadata(state, project), **({"json": payload} if method == "patch" else {}))
                     for project in (state.other.id, str(uuid4()))]
        assert responses[0].status_code == responses[1].status_code == 404
        assert responses[0].json() == responses[1].json() and "PRIVATE-FOREIGN" not in responses[0].text
        assert all(response.headers["cache-control"] == "no-store" for response in responses)
    monkeypatch.setitem(main.app.dependency_overrides, get_auth_context, lambda: AuthContext(
        user_id=LOCAL_USER_ID, workspace_id=state.workspace.id, role="owner", anonymous=False,
    ))
    assert state.client.patch(_metadata(state), json=payload).status_code == 403
    denied = state.client.patch(_metadata(state), json=payload, headers={"Origin": "https://foreign.invalid"})
    assert denied.status_code == 403 and denied.headers["cache-control"] == "no-store"


def test_metadata_concurrent_revision_has_only_one_winner(foundation_api):
    state = foundation_api
    def save(number):
        return state.client.patch(_metadata(state), json={"name": f"edit-{number}", "description": "", "expected_revision": 1})
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(save, range(4)))
    assert sorted(response.status_code for response in responses) == [200, 409, 409, 409]
    assert state.client.get(_metadata(state)).json()["metadata_revision"] == 2
    sql = str(metadata_update_statement("project", "workspace", ProjectMetadataInput(
        name="name", description="", expected_revision=1,
    )).compile(dialect=postgresql.dialect()))
    assert "RETURNING projects.id" in sql and "projects.metadata_revision =" in sql and "CASE WHEN" in sql


def test_keyed_upload_replays_one_document_context_and_receipt(foundation_api):
    state = foundation_api
    first, second = _upload(state), _upload(state)
    assert first.status_code == second.status_code == 201
    assert first.json()["id"] == second.json()["id"] and first.json()["version"] == second.json()["version"] == 1
    assert first.json()["deduplicated"] is False and second.json()["deduplicated"] is True
    assert "operation-1" not in first.text and "operation-1" not in second.text
    with state.sessions() as db:
        assert db.scalar(select(func.count()).select_from(DocumentRow)) == 1
        assert db.scalar(select(func.count()).select_from(DocumentContextRow)) == 1
        assert db.scalar(select(func.count()).select_from(DocumentNarrativeContextRevisionRow)) == 1
        assert db.scalar(select(func.count()).select_from(DocumentUploadReceiptRow)) == 1


def test_same_bytes_with_new_key_and_legacy_no_key_create_real_new_versions(foundation_api):
    state = foundation_api
    versions = [_upload(state, key=key) for key in ("one", "two", None, None)]
    assert [response.json()["version"] for response in versions] == [1, 2, 3, 4]
    assert all(response.status_code == 201 for response in versions)
    assert all("deduplicated" not in response.json() for response in versions[2:])
    replay = _upload(state, key="one")
    assert replay.json()["id"] == versions[0].json()["id"] and replay.json()["active"] is False
    assert replay.json()["version"] == 1 and replay.json()["superseded_document_ids"] == []


@pytest.mark.parametrize("change", (
    {"body": b"different bytes"}, {"name": "other.md"},
    {"fields": {"document_role": "reference"}}, {"fields": {"story_scope": "branch_a"}},
    {"fields": {"narrative_context": '{"publication_status":"draft"}'}},
    {"fields": {"replace_document_id": "different-target"}},
))
def test_upload_key_is_bound_to_all_explicit_operation_semantics(foundation_api, change):
    state = foundation_api
    assert _upload(state).status_code == 201
    conflict = _upload(state, **change)
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["reason_code"] == "document_upload_idempotency_conflict"
    with state.sessions() as db:
        assert db.scalar(select(func.count()).select_from(DocumentRow)) == 1


def test_parser_version_is_in_receipt_digest(foundation_api, monkeypatch):
    from app import document_upload_receipts
    assert _upload(foundation_api).status_code == 201
    monkeypatch.setattr(document_upload_receipts, "PARSER_VERSION", "multipart_document_parser_v2")
    assert _upload(foundation_api).status_code == 409


def test_import_role_scope_defaults_and_inheritance_remain_unchanged(foundation_api):
    state = foundation_api
    first = _upload(state, key="first", name="profile.json", body=b'{"directives": [], "document_role":"canon"}')
    assert first.json()["document_role"] == "chapter" and first.json()["story_scope"] == "global"
    second = _upload(state, key="second", name="profile.json", body=b"{}",
                     fields={"document_role": "character_profile", "story_scope": "branch_a"})
    third = _upload(state, key="third", name="profile.json", body=b"{}")
    assert second.json()["document_role"] == third.json()["document_role"] == "character_profile"
    assert third.json()["story_scope"] == "branch_a"
    assert third.json()["narrative_context"]["resolution_state"] != "confirmed"
    assert third.json()["superseded_document_ids"] == [second.json()["id"]]
    assert _upload(state, key="third", name="profile.json", body=b"{}").json()["superseded_document_ids"] == [second.json()["id"]]


@pytest.mark.parametrize("key,status", (("", 400), ("bad key", 400), ("not/valid", 400), ("x" * 129, 422)))
def test_invalid_upload_keys_do_not_create_documents(foundation_api, key, status):
    state = foundation_api
    assert _upload(state, key=key).status_code == status
    with state.sessions() as db:
        assert db.scalar(select(func.count()).select_from(DocumentRow)) == 0


def test_receipts_are_project_local_and_foreign_upload_is_404(foundation_api):
    state = foundation_api
    first = _upload(state)
    other = _upload(state, project=state.second.id)
    assert first.status_code == other.status_code == 201 and first.json()["id"] != other.json()["id"]
    assert _upload(state, project=state.other.id).status_code == _upload(state, project=str(uuid4())).status_code == 404


def test_upload_concurrent_same_key_creates_only_one_version(foundation_api):
    state = foundation_api
    with ThreadPoolExecutor(max_workers=5) as pool:
        responses = list(pool.map(lambda _: _upload(state), range(5)))
    assert all(response.status_code == 201 for response in responses)
    assert len({response.json()["id"] for response in responses}) == 1
    assert sum(not response.json()["deduplicated"] for response in responses) == 1
    with state.sessions() as db:
        assert db.scalar(select(func.count()).select_from(DocumentRow)) == 1
        assert db.scalar(select(func.count()).select_from(DocumentUploadReceiptRow)) == 1


def test_failed_receipt_insert_rolls_back_document_and_active_version(foundation_api, monkeypatch):
    state = foundation_api
    first = _upload(state, key="original")
    original_factory = main.DocumentUploadReceiptRow
    def invalid_receipt(**values):
        return original_factory(**{**values, "request_sha256": "invalid"})
    monkeypatch.setattr(main, "DocumentUploadReceiptRow", invalid_receipt)
    failure = _upload(state, key="broken", body=b"replacement")
    assert failure.status_code == 409
    with state.sessions() as db:
        rows = db.scalars(select(DocumentRow)).all()
        assert len(rows) == 1 and rows[0].id == first.json()["id"] and rows[0].active
        assert db.scalar(select(func.count()).select_from(DocumentNarrativeContextRevisionRow)) == 1
