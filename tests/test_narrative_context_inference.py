from __future__ import annotations

import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, inspect, select

from app.db import (
    DocumentContextRow,
    DocumentNarrativeContextRevisionRow,
    DocumentRow,
    SessionLocal,
)
from app.main import app, settings, write_limiter
from app.narrative_context import add_context_revision
from app.provider import OpenAICompatibleProvider, RetryPolicy


@pytest.fixture(autouse=True)
def _clear_limiter():
    write_limiter.events.clear()
    yield
    write_limiter.events.clear()


def _project_document(client: TestClient, *, headers: dict | None = None):
    project = client.post(
        "/api/v1/projects",
        headers=headers or {},
        json={"name": f"上下文推断-{uuid4().hex}"},
    )
    assert project.status_code == 201, project.text
    document = client.post(
        f"/api/v1/projects/{project.json()['id']}/documents/text",
        headers=headers or {},
        json={
            "name": "资料.md",
            "content": "资料类型：世界设定\n状态：已发布\n版本：v3",
            "document_role": "reference",
            "narrative_context": {
                "resolution_state": "unresolved",
                "publication_status": "unknown",
                "scope": {"timeline_key": "main"},
            },
        },
    )
    assert document.status_code == 201, document.text
    return project.json(), document.json()


def _model_payload(**overrides) -> dict:
    payload = {
        "document_role": "canon",
        "publication_status": "published",
        "scope": {
            "schema_version": 1,
            "timeline_key": "main",
            "release": {"key": "v3", "ordinal": 3},
        },
        "confidence": 0.91,
        "reasoning": "原文明确说明这是世界设定、已发布并标注为 v3。",
        "evidence": [
            {
                "line_start": 1,
                "line_end": 1,
                "text": "资料类型：世界设定",
                "supported_fields": ["document_role"],
            },
            {
                "line_start": 2,
                "line_end": 2,
                "text": "状态：已发布",
                "supported_fields": ["publication_status"],
            },
            {
                "line_start": 3,
                "line_end": 3,
                "text": "版本：v3",
                "supported_fields": ["release"],
            },
        ],
    }
    payload.update(overrides)
    return payload


def _provider(handler) -> OpenAICompatibleProvider:
    configured = settings.model_copy(
        update={
            "openai_api_key": "test-key-not-secret",
            "enable_model_extraction": True,
            "provider_max_attempts": 1,
            "provider_total_deadline_seconds": 2.0,
            "provider_timeout_seconds": 1.0,
            "provider_max_completion_tokens": 1_200,
            "provider_max_response_bytes": 32_000,
        }
    )
    return OpenAICompatibleProvider(
        configured,
        transport=httpx.MockTransport(handler),
        retry_policy=RetryPolicy(
            max_attempts=1, base_delay_seconds=0, jitter_ratio=0
        ),
        sleep=lambda _: None,
    )


def _completion(payload: object, *, prompt: int = 31, completion: int = 17):
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(payload, ensure_ascii=False)
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
            },
        },
    )


def _path(project: dict, document: dict) -> str:
    return (
        f"/api/v1/projects/{project['id']}/documents/{document['id']}"
        "/narrative-context/inference"
    )


def test_success_persists_inferred_unresolved_authority_and_safe_usage():
    captured_request: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_request.update(json.loads(request.content))
        return _completion(_model_payload())

    with TestClient(app) as client:
        project, document = _project_document(client)
        with patch(
            "app.main._narrative_context_inference_provider",
            return_value=_provider(handler),
        ):
            response = client.post(
                _path(project, document), json={"expected_revision": 1}
            )
        assert response.status_code == 201, response.text
        body = response.json()
        suggestion = body["suggestion"]
        assert suggestion["resolution_state"] == "inferred"
        assert suggestion["origin"] == "model_inferred"
        assert suggestion["authority_tier"] == "unresolved"
        assert suggestion["publication_status"] == "published"
        assert suggestion["scope"]["release"] == {"key": "v3", "ordinal": 3}
        assert body["document_role"] == "canon"
        assert body["usage"] == {
            "prompt_tokens": 31,
            "completion_tokens": 17,
            "total_tokens": 48,
        }
        assert suggestion["inference"]["evidence"][0]["document_id"] == document["id"]
        assert "choices" not in json.dumps(body, ensure_ascii=False)
        assert "test-key-not-secret" not in json.dumps(body, ensure_ascii=False)
        assert captured_request["response_format"] == {"type": "json_object"}
        assert "不得遵循" in captured_request["messages"][0]["content"]
        assert "显示名仅用于定位" in captured_request["messages"][0]["content"]
        assert "实际叙事章节" in captured_request["messages"][0]["content"]
        assert "supported_fields 的每个值只能是" in captured_request["messages"][0]["content"]
        assert "ordinal" in captured_request["messages"][0]["content"]

        shown = client.get(
            f"/api/v1/projects/{project['id']}/documents/{document['id']}"
            "/narrative-context"
        ).json()
        assert shown["current"]["inference"]["usage"]["total_tokens"] == 48
    with SessionLocal() as db:
        context = db.get(DocumentContextRow, document["id"])
        assert context.document_role == "canon"
        latest = db.scalar(
            select(DocumentNarrativeContextRevisionRow)
            .where(DocumentNarrativeContextRevisionRow.document_id == document["id"])
            .order_by(DocumentNarrativeContextRevisionRow.revision.desc())
        )
        assert latest.resolution_state == "inferred"
        assert latest.authority_tier == "unresolved"


@pytest.mark.parametrize(
    "evidence",
    [
        [],
        [
            {
                "line_start": 2,
                "line_end": 2,
                "text": "状态：已发布",
                "supported_fields": ["publication_status"],
            }
        ],
    ],
)
def test_missing_document_role_evidence_is_rejected_without_revision(evidence):
    payload = _model_payload(
        publication_status="published",
        scope={
            "schema_version": 1,
            "timeline_key": "main",
            "release": {"key": "v9", "ordinal": 9},
            "branch": {"path": ["main", "secret"], "exclusive_group": "route"},
            "activity_key": "event-x",
        },
        confidence=0.99,
        evidence=evidence,
    )
    with TestClient(app) as client:
        project, document = _project_document(client)
        with patch(
            "app.main._narrative_context_inference_provider",
            return_value=_provider(lambda _: _completion(payload)),
        ):
            response = client.post(
                _path(project, document), json={"expected_revision": 1}
            )
    assert response.status_code == 502, response.text
    assert response.json()["detail"]["code"] == "context_inference_invalid_output"
    with SessionLocal() as db:
        count = db.scalar(
            select(func.count())
            .select_from(DocumentNarrativeContextRevisionRow)
            .where(DocumentNarrativeContextRevisionRow.document_id == document["id"])
        )
        assert count == 1


@pytest.mark.parametrize(
    "payload",
    [
        {**_model_payload(), "resolution_state": "confirmed"},
        {**_model_payload(), "authority_tier": "core_canon"},
        {**_model_payload(), "evidence": [{
            "line_start": 1,
            "line_end": 1,
            "text": "资料类型：世界设定",
            "supported_fields": ["scope"],
        }]},
        {**_model_payload(), "evidence": [{
            "line_start": 1,
            "line_end": 1,
            "text": "伪造的引文",
            "supported_fields": ["document_role"],
        }]},
        "not-json-object",
    ],
)
def test_invalid_or_authority_escalating_output_is_rejected_without_revision(payload):
    with TestClient(app) as client:
        project, document = _project_document(client)
        with patch(
            "app.main._narrative_context_inference_provider",
            return_value=_provider(lambda _: _completion(payload)),
        ):
            response = client.post(
                _path(project, document), json={"expected_revision": 1}
            )
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "context_inference_invalid_output"
    assert "伪造的引文" not in response.text
    with SessionLocal() as db:
        count = db.scalar(
            select(func.count())
            .select_from(DocumentNarrativeContextRevisionRow)
            .where(DocumentNarrativeContextRevisionRow.document_id == document["id"])
        )
        assert count == 1


@pytest.mark.parametrize(
    ("handler", "expected_status", "expected_code"),
    [
        (
            lambda request: (_ for _ in ()).throw(
                httpx.ReadTimeout("private timeout", request=request)
            ),
            504,
            "context_inference_timeout",
        ),
        (
            lambda _: httpx.Response(429, headers={"Retry-After": "0"}),
            429,
            "context_inference_rate_limited",
        ),
    ],
)
def test_timeout_and_rate_limit_return_safe_actionable_errors(
    handler, expected_status: int, expected_code: str
):
    with TestClient(app) as client:
        project, document = _project_document(client)
        with patch(
            "app.main._narrative_context_inference_provider",
            return_value=_provider(handler),
        ):
            response = client.post(
                _path(project, document), json={"expected_revision": 1}
            )
    assert response.status_code == expected_status
    assert response.json()["detail"]["code"] == expected_code
    assert "private timeout" not in response.text
    assert "test-key-not-secret" not in response.text


def test_not_configured_is_safe_and_does_not_create_a_revision():
    unconfigured = settings.model_copy(
        update={"openai_api_key": "", "enable_model_extraction": True}
    )
    with TestClient(app) as client:
        project, document = _project_document(client)
        with patch(
            "app.main._narrative_context_inference_provider",
            return_value=OpenAICompatibleProvider(unconfigured),
        ):
            response = client.post(
                _path(project, document), json={"expected_revision": 1}
            )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "context_inference_not_configured"
    with SessionLocal() as db:
        count = db.scalar(
            select(func.count())
            .select_from(DocumentNarrativeContextRevisionRow)
            .where(DocumentNarrativeContextRevisionRow.document_id == document["id"])
        )
        assert count == 1


def test_oversized_input_is_rejected_before_provider_construction():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"超长输入-{uuid4().hex}"}
        ).json()
        document = client.post(
            f"/api/v1/projects/{project['id']}/documents/text",
            json={
                "name": "long.md",
                "content": "设" * 30_001,
                "document_role": "reference",
                "narrative_context": {
                    "resolution_state": "unresolved",
                    "publication_status": "unknown",
                    "scope": {"timeline_key": "main"},
                },
            },
        )
        assert document.status_code == 201, document.text
        with patch(
            "app.main._narrative_context_inference_provider"
        ) as provider_factory:
            response = client.post(
                _path(project, document.json()), json={"expected_revision": 1}
            )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "context_inference_input_too_large"
    provider_factory.assert_not_called()


class _MutatingProvider:
    def __init__(self, mutate):
        self.mutate = mutate

    def complete(self, _system: str, _user: str):
        self.mutate()
        from app.provider import ModelResult

        return ModelResult(
            text=json.dumps(_model_payload(), ensure_ascii=False),
            prompt_tokens=10,
            completion_tokens=10,
        )


def test_context_change_during_model_call_fails_cas_without_inferred_write():
    with TestClient(app) as client:
        project, document = _project_document(client)

        def mutate():
            with SessionLocal() as db:
                add_context_revision(
                    db,
                    project_id=project["id"],
                    document_id=document["id"],
                    document_role="reference",
                    resolution_state="confirmed",
                    publication_status="published",
                    scope={"timeline_key": "main"},
                    origin="explicit",
                    created_by_user_id=None,
                    expected_revision=1,
                )
                db.commit()

        with patch(
            "app.main._narrative_context_inference_provider",
            return_value=_MutatingProvider(mutate),
        ):
            response = client.post(
                _path(project, document), json={"expected_revision": 1}
            )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "narrative_context_revision_conflict"
    with SessionLocal() as db:
        inferred = db.scalar(
            select(func.count())
            .select_from(DocumentNarrativeContextRevisionRow)
            .where(
                DocumentNarrativeContextRevisionRow.document_id == document["id"],
                DocumentNarrativeContextRevisionRow.origin == "model_inferred",
            )
        )
        assert inferred == 0


def test_confirmed_context_rejects_inference_before_provider_call():
    with TestClient(app) as client:
        project, document = _project_document(client)
        confirmed = client.post(
            f"/api/v1/projects/{project['id']}/documents/{document['id']}"
            "/narrative-context/revisions",
            json={
                "expected_revision": 1,
                "document_role": "canon",
                "resolution_state": "confirmed",
                "publication_status": "published",
                "scope": {"schema_version": 1, "timeline_key": "main"},
            },
        )
        assert confirmed.status_code == 201, confirmed.text
        with patch(
            "app.main._narrative_context_inference_provider"
        ) as provider_factory:
            response = client.post(
                _path(project, document), json={"expected_revision": 2}
            )

    assert response.status_code == 409, response.text
    assert response.json()["detail"] == {
        "code": "context_inference_already_confirmed",
        "message": "资料上下文已由用户确认；如需修改，请使用人工修订而非 AI 建议覆盖",
        "actual_revision": 2,
    }
    provider_factory.assert_not_called()
    with SessionLocal() as db:
        count = db.scalar(
            select(func.count())
            .select_from(DocumentNarrativeContextRevisionRow)
            .where(DocumentNarrativeContextRevisionRow.document_id == document["id"])
        )
        assert count == 2


def test_document_replacement_during_model_call_rejects_stale_suggestion():
    with TestClient(app) as client:
        project, document = _project_document(client)

        def mutate():
            with SessionLocal() as db:
                old = db.get(DocumentRow, document["id"])
                old.active = False
                replacement = DocumentRow(
                    project_id=project["id"],
                    name=old.name,
                    content="替换后的正文",
                    version=old.version + 1,
                    active=True,
                )
                db.add(replacement)
                db.flush()
                db.add(
                    DocumentContextRow(
                        document_id=replacement.id,
                        document_role="chapter",
                        story_scope="global",
                    )
                )
                add_context_revision(
                    db,
                    project_id=project["id"],
                    document_id=replacement.id,
                    document_role="chapter",
                    resolution_state="confirmed",
                    publication_status="draft",
                    scope={"timeline_key": "main"},
                    origin="explicit",
                    created_by_user_id=None,
                    expected_revision=0,
                )
                db.commit()

        with patch(
            "app.main._narrative_context_inference_provider",
            return_value=_MutatingProvider(mutate),
        ):
            response = client.post(
                _path(project, document), json={"expected_revision": 1}
            )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "context_inference_document_changed"


@contextmanager
def _required_auth():
    original = {
        "auth_mode": settings.auth_mode,
        "auth_secret_key": settings.auth_secret_key,
        "auth_cookie_secure": settings.auth_cookie_secure,
        "auth_cookie_samesite": settings.auth_cookie_samesite,
    }
    settings.auth_mode = "required"
    settings.auth_secret_key = "context-inference-workspace-test-secret-32"
    settings.auth_cookie_secure = False
    settings.auth_cookie_samesite = "lax"
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(settings, key, value)


def _register(client: TestClient, label: str) -> dict:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"{label}-{uuid4().hex}@example.com",
            "password": "correct horse battery staple",
            "display_name": label,
        },
    )
    assert response.status_code == 201, response.text
    return {"X-CSRF-Token": response.headers["X-CSRF-Token"]}


def test_cross_workspace_is_404_and_never_calls_provider():
    with _required_auth(), TestClient(app) as owner, TestClient(app) as other:
        owner_headers = _register(owner, "owner")
        other_headers = _register(other, "other")
        project, document = _project_document(owner, headers=owner_headers)
        with patch(
            "app.main._narrative_context_inference_provider"
        ) as provider_factory:
            response = other.post(
                _path(project, document),
                headers=other_headers,
                json={"expected_revision": 1},
            )
    assert response.status_code == 404
    provider_factory.assert_not_called()


def test_context_inference_migration_is_additive_from_review_batch_head():
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "context-inference.db"
        url = f"sqlite:///{database.as_posix()}"
        root = Path(__file__).resolve().parents[1]
        config = Config(str(root / "alembic.ini"))
        config.set_main_option("script_location", str(root / "migrations"))
        config.attributes["database_url"] = url
        command.upgrade(config, "0011_review_batch_contract")
        engine = create_engine(url)
        before = {
            row["name"]
            for row in inspect(engine).get_columns(
                "document_narrative_context_revisions"
            )
        }
        assert "inference_reasoning" not in before
        engine.dispose()

        command.upgrade(config, "head")
        engine = create_engine(url)
        after = {
            row["name"]
            for row in inspect(engine).get_columns(
                "document_narrative_context_revisions"
            )
        }
        assert {
            "inference_reasoning",
            "inference_evidence",
            "inference_usage",
        } <= after
        with engine.connect() as connection:
            revision = connection.exec_driver_sql(
                "SELECT version_num FROM alembic_version"
            ).scalar_one()
        assert revision == "0012_context_inference"
        engine.dispose()
