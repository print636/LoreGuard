from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app, settings, write_limiter


@contextmanager
def _quiet_writes():
    write_limiter.events.clear()
    try:
        with patch("app.main.dispatch_analysis"):
            yield
    finally:
        write_limiter.events.clear()


def _project(client: TestClient, *, headers: dict | None = None) -> str:
    response = client.post(
        "/api/v1/projects",
        headers=headers,
        json={"name": f"章节发布-{uuid4().hex}"},
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _chapter(
    client: TestClient,
    project_id: str,
    *,
    name: str,
    publication: str = "draft",
    confirmed: bool = True,
    role: str = "chapter",
    headers: dict | None = None,
    scope: dict | None = None,
) -> dict:
    response = client.post(
        f"/api/v1/projects/{project_id}/documents/text",
        headers=headers,
        json={
            "name": name,
            "content": f"{name} 的原创剧情。",
            "document_role": role,
            "narrative_context": {
                "resolution_state": "confirmed" if confirmed else "unresolved",
                "publication_status": publication,
                "scope": scope or {"timeline_key": "main"},
            },
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _publish(
    client: TestClient,
    project_id: str,
    document: dict,
    *,
    headers: dict | None = None,
    revision: int | None = None,
    version: int | None = None,
):
    return client.post(
        f"/api/v1/projects/{project_id}/documents/{document['id']}/publish",
        headers=headers,
        json={
            "expected_revision": (
                revision if revision is not None
                else document["narrative_context"]["context_revision"]
            ),
            "expected_document_version": (
                version if version is not None else document["version"]
            ),
        },
    )


def test_author_publishes_without_report_and_next_draft_uses_frozen_history():
    scope = {
        "timeline_key": "main",
        "release": {"key": "v1", "ordinal": 1},
    }
    with TestClient(app) as client, _quiet_writes():
        project_id = _project(client)
        first = _chapter(
            client, project_id, name="chapter-01.md", publication="in_review",
            scope=scope,
        )
        # No analysis report, confirmed role traits, or issue verdict exists.
        published = _publish(client, project_id, first)
        assert published.status_code == 201, published.text
        result = published.json()
        assert result["document_id"] == first["id"]
        assert result["document_version"] == first["version"]
        assert result["revision"] == 2
        assert result["resolution_state"] == "confirmed"
        assert result["publication_status"] == "published"
        assert result["authority_tier"] == "formal_record"
        assert result["origin"] == "explicit"
        assert result["scope"] == first["narrative_context"]["scope"]

        second = _chapter(
            client, project_id, name="chapter-02.md", scope=scope,
        )
        run = client.post(
            f"/api/v1/projects/{project_id}/analysis-runs",
            json={"mode": "draft_review", "target_document_ids": [second["id"]]},
        )
        assert run.status_code == 202, run.text
        frozen = client.get(f"/api/v1/analysis-runs/{run.json()['id']}").json()
    assert frozen["review_batch"]["target_document_ids"] == [second["id"]]
    assert frozen["review_batch"]["background_document_ids"] == [first["id"]]
    assert {row["document_id"]: row["batch_role"] for row in frozen["input_documents"]} == {
        first["id"]: "background", second["id"]: "target",
    }
    assert next(
        row for row in frozen["input_documents"] if row["document_id"] == first["id"]
    )["narrative_context"]["publication_status"] == "published"


def test_publish_does_not_rewrite_existing_analysis_snapshot():
    with TestClient(app) as client, _quiet_writes():
        project_id = _project(client)
        chapter = _chapter(client, project_id, name="before.md")
        run = client.post(
            f"/api/v1/projects/{project_id}/analysis-runs",
            json={"mode": "draft_review", "target_document_ids": [chapter["id"]]},
        )
        assert run.status_code == 202, run.text
        assert _publish(client, project_id, chapter).status_code == 201
        frozen = client.get(f"/api/v1/analysis-runs/{run.json()['id']}").json()
        current = client.get(f"/api/v1/projects/{project_id}/documents").json()
    assert frozen["review_batch"]["target_document_ids"] == [chapter["id"]]
    assert frozen["input_documents"][0]["narrative_context"]["publication_status"] == "draft"
    assert current[0]["narrative_context"]["publication_status"] == "published"


def test_publish_rejects_stale_revision_version_and_duplicate_request():
    with TestClient(app) as client, _quiet_writes():
        project_id = _project(client)
        chapter = _chapter(client, project_id, name="chapter.md")
        stale_version = _publish(client, project_id, chapter, version=chapter["version"] + 1)
        assert stale_version.status_code == 409
        assert stale_version.json()["detail"]["code"] == "document_version_conflict"
        stale_context = _publish(client, project_id, chapter, revision=2)
        assert stale_context.status_code == 409
        assert stale_context.json()["detail"]["code"] == "narrative_context_revision_conflict"
        first = _publish(client, project_id, chapter)
        repeat = _publish(client, project_id, chapter)
    assert first.status_code == 201, first.text
    assert repeat.status_code == 409
    assert repeat.json()["detail"]["code"] == "chapter_already_published"


def test_two_concurrent_publish_requests_create_only_one_release_revision():
    with TestClient(app) as first_client, TestClient(app) as second_client, _quiet_writes():
        project_id = _project(first_client)
        chapter = _chapter(first_client, project_id, name="race.md")
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(_publish, first_client, project_id, chapter)
            second = pool.submit(_publish, second_client, project_id, chapter)
            responses = [first.result(timeout=8), second.result(timeout=8)]
        revisions = first_client.get(
            f"/api/v1/projects/{project_id}/documents/{chapter['id']}"
            "/narrative-context"
        )
    assert sorted(response.status_code for response in responses) == [201, 409]
    assert next(
        response.json()["detail"]["code"]
        for response in responses if response.status_code == 409
    ) == "chapter_already_published"
    assert revisions.status_code == 200, revisions.text
    assert [row["revision"] for row in revisions.json()["revisions"]] == [2, 1]


def test_publish_rejects_wrong_role_unconfirmed_and_non_draft_status():
    with TestClient(app) as client, _quiet_writes():
        project_id = _project(client)
        cases = (
            ("world.md", "canon", "draft", True, "document_not_publishable"),
            ("unknown.md", "chapter", "draft", False, "chapter_context_unconfirmed"),
            ("retired.md", "chapter", "retired", True, "chapter_not_draft_or_in_review"),
            ("published.md", "chapter", "published", True, "chapter_already_published"),
        )
        for name, role, publication, confirmed, expected in cases:
            document = _chapter(
                client, project_id, name=name, role=role,
                publication=publication, confirmed=confirmed,
            )
            result = _publish(client, project_id, document)
            assert result.status_code == 409, result.text
            assert result.json()["detail"]["code"] == expected


def test_publish_rejects_inactive_replaced_version_and_cross_project_document():
    with TestClient(app) as client, _quiet_writes():
        first_project = _project(client)
        second_project = _project(client)
        old = _chapter(client, first_project, name="chapter.md")
        replaced = client.post(
            f"/api/v1/projects/{first_project}/documents/text",
            json={
                "name": "chapter.md", "content": "第二版草稿。",
                "replace_document_id": old["id"],
                "document_role": "chapter",
                "narrative_context": {
                    "resolution_state": "confirmed", "publication_status": "draft",
                },
            },
        )
        assert replaced.status_code == 201, replaced.text
        inactive = _publish(client, first_project, old)
        foreign = _publish(client, second_project, old)
    assert inactive.status_code == 409
    assert inactive.json()["detail"]["code"] == "document_not_active"
    assert foreign.status_code == 404


def test_generic_revision_and_replacement_cannot_publish_existing_pending_chapter():
    with TestClient(app) as client, _quiet_writes():
        project_id = _project(client)
        chapter = _chapter(client, project_id, name="chapter.md")
        revision_url = (
            f"/api/v1/projects/{project_id}/documents/{chapter['id']}"
            "/narrative-context/revisions"
        )
        direct = client.post(
            revision_url,
            json={
                "expected_revision": 1,
                "resolution_state": "confirmed",
                "publication_status": "published",
                "scope": {"timeline_key": "main"},
            },
        )
        assert direct.status_code == 409, direct.text
        assert direct.json()["detail"]["code"] == "chapter_publish_endpoint_required"
        switched = client.post(
            revision_url,
            json={
                "expected_revision": 1, "document_role": "reference",
                "resolution_state": "confirmed", "publication_status": "published",
            },
        )
        assert switched.status_code == 201, switched.text
        switched_back = client.post(
            revision_url,
            json={
                "expected_revision": 2, "document_role": "chapter",
                "resolution_state": "confirmed", "publication_status": "published",
            },
        )
        assert switched_back.status_code == 409, switched_back.text
        assert switched_back.json()["detail"]["code"] == "chapter_publish_endpoint_required"

        replacement = client.post(
            f"/api/v1/projects/{project_id}/documents/text",
            json={
                "name": "chapter.md", "content": "新版本。",
                "replace_document_id": chapter["id"],
                "document_role": "chapter",
                "narrative_context": {
                    "resolution_state": "confirmed", "publication_status": "published",
                },
            },
        )
        assert replacement.status_code == 409, replacement.text
        assert replacement.json()["detail"]["code"] == "chapter_publish_endpoint_required"
        still_active = client.get(f"/api/v1/projects/{project_id}/documents").json()
    assert [row["id"] for row in still_active] == [chapter["id"]]


def test_importing_existing_published_history_is_still_allowed():
    with TestClient(app) as client, _quiet_writes():
        project_id = _project(client)
        historical = _chapter(
            client, project_id, name="archive.md", publication="published",
        )
        assert historical["narrative_context"]["publication_status"] == "published"
        unknown = _chapter(
            client, project_id, name="old-import.md", publication="unknown",
        )
        corrected = client.post(
            f"/api/v1/projects/{project_id}/documents/{unknown['id']}"
            "/narrative-context/revisions",
            json={
                "expected_revision": 1,
                "resolution_state": "confirmed",
                "publication_status": "published",
            },
        )
    assert corrected.status_code == 201, corrected.text


@contextmanager
def _required_auth():
    old = (
        settings.auth_mode,
        settings.auth_secret_key,
        settings.auth_cookie_secure,
        settings.auth_cookie_samesite,
    )
    settings.auth_mode = "required"
    settings.auth_secret_key = "chapter-publish-test-secret-key-32-bytes"
    settings.auth_cookie_secure = False
    settings.auth_cookie_samesite = "lax"
    write_limiter.events.clear()
    try:
        yield
    finally:
        (
            settings.auth_mode,
            settings.auth_secret_key,
            settings.auth_cookie_secure,
            settings.auth_cookie_samesite,
        ) = old
        write_limiter.events.clear()


def _register(client: TestClient, label: str) -> str:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"{label}-{uuid4().hex}@example.com",
            "password": "correct horse battery staple",
            "display_name": label,
        },
    )
    assert response.status_code == 201, response.text
    return response.headers["X-CSRF-Token"]


def test_publish_requires_owner_workspace_and_csrf():
    with _required_auth(), TestClient(app) as owner, TestClient(app) as stranger:
        owner_headers = {"X-CSRF-Token": _register(owner, "publisher")}
        stranger_headers = {"X-CSRF-Token": _register(stranger, "stranger")}
        project_id = _project(owner, headers=owner_headers)
        chapter = _chapter(
            owner, project_id, name="private.md", headers=owner_headers,
        )
        foreign = _publish(
            stranger, project_id, chapter, headers=stranger_headers,
        )
        missing_csrf = _publish(owner, project_id, chapter)
        allowed = _publish(owner, project_id, chapter, headers=owner_headers)
    assert foreign.status_code == 404
    assert missing_csrf.status_code == 403
    assert allowed.status_code == 201, allowed.text


def test_file_replacement_cannot_skip_explicit_publish():
    with TestClient(app) as client, _quiet_writes():
        project_id = _project(client)
        chapter = _chapter(client, project_id, name="chapter.md")
        attempt = client.post(
            f"/api/v1/projects/{project_id}/documents",
            data={
                "replace_document_id": chapter["id"],
                "document_role": "chapter",
                "narrative_context": json.dumps(
                    {"resolution_state": "confirmed", "publication_status": "published"}
                ),
            },
            files={"file": ("chapter.md", b"new chapter", "text/markdown")},
        )
        assert attempt.status_code == 409, attempt.text
        assert attempt.json()["detail"]["code"] == "chapter_publish_endpoint_required"
        active = client.get(f"/api/v1/projects/{project_id}/documents").json()
    assert [row["id"] for row in active] == [chapter["id"]]
