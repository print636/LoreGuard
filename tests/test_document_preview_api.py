from __future__ import annotations

from hashlib import sha256
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app import main
from app.auth import AuthContext, get_auth_context
from app.db import (
    Base,
    DocumentContextRow,
    DocumentNarrativeContextRevisionRow,
    DocumentRow,
    ProjectRow,
    WorkspaceRow,
)
from app.narrative_context import payload_sha256


def _forbidden_runtime_call(*args, **kwargs):
    raise AssertionError("Document previews must not call providers or start analysis")


@pytest.fixture()
def preview_api(tmp_path, monkeypatch):
    # conftest isolates module imports; this fixture also owns every query made
    # by the endpoint. Do not enter TestClient's lifespan or initialize the
    # application's database or credential configuration.
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'document-preview.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    workspace_id, foreign_workspace_id = str(uuid4()), str(uuid4())
    with sessions() as db:
        db.add_all([
            WorkspaceRow(id=workspace_id, name="预览工作区"),
            WorkspaceRow(id=foreign_workspace_id, name="其他工作区"),
        ])
        db.flush()
        project = ProjectRow(workspace_id=workspace_id, name="当前项目")
        other_project = ProjectRow(workspace_id=workspace_id, name="同工作区另一项目")
        foreign_project = ProjectRow(workspace_id=foreign_workspace_id, name="其他工作区项目")
        db.add_all([project, other_project, foreign_project])
        db.flush()
        historical = DocumentRow(
            project_id=project.id, name="chapter.md", content="旧正文\n\n旧末行\n",
            version=1, active=False,
        )
        active = DocumentRow(
            project_id=project.id, name="chapter.md",
            content="林澈登场。\r\n\r\n第三行 [a.*]。\nSTRASSE\n林澈离开。\n",
            version=2, active=True,
        )
        empty = DocumentRow(project_id=project.id, name="empty.txt", content="")
        other_document = DocumentRow(
            project_id=other_project.id, name="other.txt", content="同工作区私有正文",
        )
        foreign_document = DocumentRow(
            project_id=foreign_project.id, name="foreign.txt", content="其他工作区私有正文",
        )
        db.add_all([historical, active, empty, other_document, foreign_document])
        db.flush()
        scope = {
            "schema_version": 1, "timeline_key": "main",
            "release": {"key": "v1", "ordinal": 1}, "branch": {"path": ["main"]},
        }
        for document in (historical, active):
            db.add(DocumentContextRow(
                document_id=document.id, document_role="chapter", story_scope="main",
            ))
            db.add(DocumentNarrativeContextRevisionRow(
                project_id=project.id, document_id=document.id, revision=1,
                resolution_state="confirmed", origin="explicit",
                authority_tier="formal_record" if document.active else "draft",
                publication_status="published" if document.active else "draft",
                scope_payload=scope, scope_sha256=payload_sha256(scope),
            ))
        db.commit()
        metadata = {
            document.id: jsonable_encoder(
                main.serialize_document(document, include_content=False, db=db)
            )
            for document in (historical, active, empty)
        }

    monkeypatch.setattr(main, "SessionLocal", sessions)
    monkeypatch.setitem(main.app.dependency_overrides, get_auth_context, lambda: AuthContext(
        user_id=str(uuid4()), workspace_id=workspace_id, role="owner", anonymous=False,
    ))
    for name in (
        "OpenAICompatibleProvider", "account_provider_runtime", "dispatch_analysis",
        "infer_narrative_context", "_narrative_context_inference_provider", "execute_analysis",
    ):
        monkeypatch.setattr(main, name, _forbidden_runtime_call)

    statements = []

    @event.listens_for(engine, "before_cursor_execute")
    def _only_read_queries(_connection, _cursor, statement, _parameters, _context, _many):
        statements.append(statement)
        assert statement.lstrip().upper().startswith("SELECT"), statement

    client = TestClient(main.app)
    try:
        yield SimpleNamespace(
            client=client, sessions=sessions, workspace_id=workspace_id,
            foreign_workspace_id=foreign_workspace_id, project=project,
            other_project=other_project, foreign_project=foreign_project,
            historical=historical, active=active, empty=empty,
            other_document=other_document, foreign_document=foreign_document,
            metadata=metadata, statements=statements,
        )
    finally:
        client.close()
        engine.dispose()


def _preview_path(state, document=None, project=None):
    document = document or state.active
    project = project or state.project
    return f"/api/v1/projects/{project.id}/documents/{document.id}/preview"


@pytest.mark.parametrize("version", ["historical", "active"])
def test_preview_reads_active_and_historical_versions_without_state_changes(preview_api, version):
    state = preview_api
    document = getattr(state, version)
    response = state.client.get(_preview_path(state, document))
    assert response.status_code == 200, response.text
    assert response.headers["Cache-Control"] == "no-store"
    payload = response.json()
    assert set(payload) == {"document", "lines", "page", "query"}
    assert payload["document"] == {
        **state.metadata[document.id],
        "char_count": len(document.content),
        "line_count": len(document.content.splitlines()),
        "content_sha256": sha256(document.content.encode("utf-8")).hexdigest(),
    }
    assert "content" not in payload and "content" not in payload["document"]
    assert payload["lines"] == [
        {"line_number": number, "text": text}
        for number, text in enumerate(document.content.splitlines(), start=1)
    ]
    assert payload["page"] == {
        "offset": 0, "limit": 120,
        "total": len(document.content.splitlines()), "has_more": False,
    }
    assert payload["query"] == ""
    with state.sessions() as db:
        stored = db.get(DocumentRow, document.id)
        assert stored.content == document.content
        assert stored.active == document.active
        assert stored.version == document.version
        assert jsonable_encoder(main.serialize_document(stored, include_content=False, db=db)) == (
            state.metadata[document.id]
        )
        assert len(db.scalars(select(DocumentNarrativeContextRevisionRow)).all()) == 2
    assert state.statements


def test_preview_keeps_blank_lines_and_original_page_numbers(preview_api):
    response = preview_api.client.get(
        _preview_path(preview_api), params={"offset": 1, "limit": 2},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["lines"] == [
        {"line_number": 2, "text": ""},
        {"line_number": 3, "text": "第三行 [a.*]。"},
    ]
    assert payload["page"] == {"offset": 1, "limit": 2, "total": 5, "has_more": True}


@pytest.mark.parametrize("offset, line_numbers", [(4, [5]), (5, []), (500, [])])
def test_preview_boundary_pages_and_offset_past_end(preview_api, offset, line_numbers):
    response = preview_api.client.get(
        _preview_path(preview_api), params={"offset": offset, "limit": 200},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert [row["line_number"] for row in payload["lines"]] == line_numbers
    assert payload["page"] == {
        "offset": offset, "limit": 200, "total": 5, "has_more": False,
    }


@pytest.mark.parametrize("query, offset, limit, numbers, total, normalized", [
    ("  林澈 \t", 0, 1, [1], 2, "林澈"),
    ("林澈", 1, 1, [5], 2, "林澈"),
    ("林澈", 2, 1, [], 2, "林澈"),
    (".*", 0, 120, [3], 1, ".*"),
    ("[a.*]", 0, 120, [3], 1, "[a.*]"),
    ("strasse", 0, 120, [4], 1, "strasse"),
    ("不存在", 0, 120, [], 0, "不存在"),
    (" \t ", 0, 120, [1, 2, 3, 4, 5], 5, ""),
    ("x" * 160, 0, 120, [], 0, "x" * 160),
])
def test_preview_literal_search_filters_before_pagination(
    preview_api, query, offset, limit, numbers, total, normalized,
):
    response = preview_api.client.get(
        _preview_path(preview_api), params={"query": query, "offset": offset, "limit": limit},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["query"] == normalized
    assert payload["document"]["line_count"] == 5
    assert [row["line_number"] for row in payload["lines"]] == numbers
    assert payload["page"] == {
        "offset": offset, "limit": limit, "total": total,
        "has_more": offset + len(numbers) < total,
    }


def test_preview_of_empty_stored_document_returns_empty_page(preview_api):
    response = preview_api.client.get(_preview_path(preview_api, preview_api.empty))
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["lines"] == []
    assert payload["page"] == {"offset": 0, "limit": 120, "total": 0, "has_more": False}
    assert payload["document"]["char_count"] == payload["document"]["line_count"] == 0
    assert payload["document"]["content_sha256"] == sha256(b"").hexdigest()


@pytest.mark.parametrize("parameters", [
    {"offset": -1}, {"offset": "not-an-integer"}, {"limit": 0}, {"limit": 201},
    {"limit": "not-an-integer"}, {"query": "x" * 161},
])
def test_preview_validates_pagination_and_query_before_reading(preview_api, parameters):
    response = preview_api.client.get(_preview_path(preview_api), params=parameters)
    assert response.status_code == 422
    assert preview_api.statements == []


def test_preview_uses_identical_not_found_responses_for_all_ownership_failures(preview_api):
    state = preview_api
    ids = [
        (state.project.id, state.other_document.id),
        (state.project.id, state.foreign_document.id),
        (state.other_project.id, state.active.id),
        (state.foreign_project.id, state.foreign_document.id),
        (state.foreign_project.id, state.active.id),
        (state.project.id, str(uuid4())),
        (str(uuid4()), state.active.id),
        ("invalid-project-id", state.active.id),
        (state.project.id, "invalid-document-id"),
        ("invalid-project-id", "invalid-document-id"),
    ]
    for project_id, document_id in ids:
        response = state.client.get(
            f"/api/v1/projects/{project_id}/documents/{document_id}/preview"
        )
        assert response.status_code == 404, response.text
        assert response.json() == {"detail": "文档不存在"}
        assert response.headers["Cache-Control"] == "no-store"


def test_preview_access_is_bound_to_current_authenticated_workspace(preview_api, monkeypatch):
    state = preview_api
    monkeypatch.setitem(main.app.dependency_overrides, get_auth_context, lambda: AuthContext(
        user_id=str(uuid4()), workspace_id=state.foreign_workspace_id,
        role="owner", anonymous=False,
    ))
    denied = state.client.get(_preview_path(state))
    assert denied.status_code == 404
    assert denied.json() == {"detail": "文档不存在"}
    permitted = state.client.get(_preview_path(state, state.foreign_document, state.foreign_project))
    assert permitted.status_code == 200, permitted.text
    assert permitted.json()["document"]["id"] == state.foreign_document.id


def test_preview_route_preserves_existing_diff_and_narrative_context_routes(preview_api):
    state = preview_api
    diff = state.client.get(
        f"/api/v1/projects/{state.project.id}/documents/diff",
        params={"from_document_id": state.historical.id, "to_document_id": state.active.id},
    )
    assert diff.status_code == 200, diff.text
    assert diff.json()["from_document"]["id"] == state.historical.id
    assert diff.json()["to_document"]["id"] == state.active.id
    context = state.client.get(
        f"/api/v1/projects/{state.project.id}/documents/{state.active.id}/narrative-context"
    )
    assert context.status_code == 200, context.text
    assert context.json()["current"]["publication_status"] == "published"
    route = f"/api/v1/projects/{{project_id}}/documents/{{document_id}}/publish"
    assert any(route == item.path and "POST" in item.methods for item in main.app.routes)
