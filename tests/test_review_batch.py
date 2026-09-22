from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest
from sqlalchemy import select

from app.db import (
    AnalysisRunInputContextRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    DocumentContextRow,
    SessionLocal,
)
from app.domain import ConsistencyIssue, EvidenceSpan, IssueCategory, Severity
from app.main import app, settings, write_limiter
from app.pipeline import DocumentInput, PipelineResult
from app.service import (
    _character_stage_settings_for_run,
    _enforce_draft_issue_boundary,
)


@pytest.fixture(autouse=True)
def _clear_write_limiter():
    write_limiter.events.clear()
    yield
    write_limiter.events.clear()


def _scope(branch: str | None = None) -> dict:
    payload: dict = {"timeline_key": "main"}
    if branch is not None:
        payload["branch"] = {
            "path": ["main", branch],
            "exclusive_group": "route",
        }
    return payload


def _document(
    client: TestClient,
    project_id: str,
    *,
    name: str,
    role: str,
    publication: str,
    confirmed: bool = True,
    branch: str | None = None,
    content: str | None = None,
    replace_document_id: str | None = None,
) -> dict:
    response = client.post(
        f"/api/v1/projects/{project_id}/documents/text",
        json={
            "name": name,
            "content": content or f"{name} 的正文",
            "replace_document_id": replace_document_id,
            "document_role": role,
            "narrative_context": {
                "resolution_state": "confirmed" if confirmed else "unresolved",
                "publication_status": publication,
                "scope": _scope(branch),
            },
        },
    )
    assert response.status_code == 201, response.text
    return response.json()


def _project(client: TestClient) -> dict:
    response = client.post(
        "/api/v1/projects", json={"name": f"审查批次-{uuid4().hex}"}
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_legacy_missing_or_empty_body_keeps_full_review_contract():
    with TestClient(app) as client, patch("app.main.dispatch_analysis"):
        project = _project(client)
        document = _document(
            client,
            project["id"],
            name="legacy.md",
            role="chapter",
            publication="unknown",
            confirmed=False,
        )
        missing = client.post(
            f"/api/v1/projects/{project['id']}/analysis-runs"
        )
        empty = client.post(
            f"/api/v1/projects/{project['id']}/analysis-runs", json={}
        )
    assert missing.status_code == 202, missing.text
    assert empty.status_code == 202, empty.text
    for response in (missing, empty):
        batch = response.json()["review_batch"]
        assert batch["mode"] == "full_review"
        assert batch["sensitivity"] == "balanced"
        assert batch["target_document_ids"] == [document["id"]]


def test_draft_review_freezes_targets_authority_history_and_visible_exclusions():
    with TestClient(app) as client, patch("app.main.dispatch_analysis"):
        project = _project(client)
        canon = _document(
            client,
            project["id"],
            name="world.md",
            role="canon",
            publication="published",
        )
        profile = _document(
            client,
            project["id"],
            name="character.md",
            role="character_profile",
            publication="unknown",
        )
        history = _document(
            client,
            project["id"],
            name="chapter-01.md",
            role="chapter",
            publication="published",
            branch="route-a",
        )
        draft = _document(
            client,
            project["id"],
            name="chapter-02.md",
            role="chapter",
            publication="draft",
            branch="route-a",
        )
        unresolved = _document(
            client,
            project["id"],
            name="chapter-unconfirmed.md",
            role="chapter",
            publication="draft",
            confirmed=False,
        )
        incompatible = _document(
            client,
            project["id"],
            name="other-route.md",
            role="chapter",
            publication="published",
            branch="route-b",
        )
        retired = _document(
            client,
            project["id"],
            name="retired-world.md",
            role="canon",
            publication="retired",
        )
        response = client.post(
            f"/api/v1/projects/{project['id']}/analysis-runs",
            json={
                "mode": "draft_review",
                "target_document_ids": [draft["id"]],
                "sensitivity": "exploratory",
            },
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["id"]
        status = client.get(f"/api/v1/analysis-runs/{run_id}").json()

    batch = status["review_batch"]
    assert batch["mode"] == "draft_review"
    assert batch["sensitivity"] == "exploratory"
    assert batch["target_document_ids"] == [draft["id"]]
    assert set(batch["background_document_ids"]) == {
        canon["id"],
        profile["id"],
        history["id"],
    }
    excluded = {
        row["document_id"]: row["reason"]
        for row in batch["excluded_documents"]
    }
    assert excluded[unresolved["id"]] == "narrative_context_unconfirmed"
    assert excluded[incompatible["id"]] == "narrative_scope_incompatible"
    assert excluded[retired["id"]] == "retired_document"
    roles = {
        row["document_id"]: row["batch_role"]
        for row in status["input_documents"]
    }
    assert roles[draft["id"]] == "target"
    assert all(roles[row_id] == "background" for row_id in batch["background_document_ids"])


def test_auto_selection_excludes_unconfirmed_and_non_chapter_drafts():
    with TestClient(app) as client, patch("app.main.dispatch_analysis"):
        project = _project(client)
        valid = _document(
            client,
            project["id"],
            name="valid.md",
            role="chapter",
            publication="in_review",
        )
        unresolved = _document(
            client,
            project["id"],
            name="unresolved.md",
            role="chapter",
            publication="draft",
            confirmed=False,
        )
        false_target = _document(
            client,
            project["id"],
            name="draft-canon.md",
            role="canon",
            publication="draft",
        )
        response = client.post(
            f"/api/v1/projects/{project['id']}/analysis-runs",
            json={"mode": "draft_review"},
        )
    assert response.status_code == 202, response.text
    batch = response.json()["review_batch"]
    assert batch["target_document_ids"] == [valid["id"]]
    reasons = {
        row["document_id"]: row["reason"]
        for row in batch["excluded_documents"]
    }
    assert reasons[unresolved["id"]] == "narrative_context_unconfirmed"
    target_reasons = {
        row["document_id"]: row["reason"]
        for row in batch["target_selection_exclusions"]
    }
    assert target_reasons[false_target["id"]] == "target_must_be_chapter"


def test_explicit_invalid_and_cross_project_targets_fail_closed():
    with TestClient(app) as client, patch("app.main.dispatch_analysis"):
        first = _project(client)
        second = _project(client)
        unresolved = _document(
            client,
            first["id"],
            name="unresolved.md",
            role="chapter",
            publication="draft",
            confirmed=False,
        )
        foreign = _document(
            client,
            second["id"],
            name="foreign.md",
            role="chapter",
            publication="draft",
        )
        blocked = client.post(
            f"/api/v1/projects/{first['id']}/analysis-runs",
            json={"mode": "draft_review", "target_document_ids": [unresolved["id"]]},
        )
        hidden = client.post(
            f"/api/v1/projects/{first['id']}/analysis-runs",
            json={"mode": "draft_review", "target_document_ids": [foreign["id"]]},
        )
    assert blocked.status_code == 409
    assert blocked.json()["detail"]["code"] == "invalid_draft_review_targets"
    assert hidden.status_code == 404


def test_baseline_build_has_only_background_and_excludes_draft():
    with TestClient(app) as client, patch("app.main.dispatch_analysis"):
        project = _project(client)
        canon = _document(
            client, project["id"], name="world.md", role="canon", publication="unknown"
        )
        history = _document(
            client,
            project["id"],
            name="old.md",
            role="chapter",
            publication="published",
        )
        draft = _document(
            client,
            project["id"],
            name="new.md",
            role="chapter",
            publication="draft",
        )
        response = client.post(
            f"/api/v1/projects/{project['id']}/analysis-runs",
            json={"mode": "baseline_build", "sensitivity": "conservative"},
        )
    assert response.status_code == 202, response.text
    batch = response.json()["review_batch"]
    assert batch["target_document_ids"] == []
    assert set(batch["background_document_ids"]) == {canon["id"], history["id"]}
    assert any(
        row["document_id"] == draft["id"]
        and row["reason"] == "draft_excluded_from_baseline"
        for row in batch["excluded_documents"]
    )


def test_idempotency_binds_normalized_batch_intent():
    with TestClient(app) as client, patch("app.main.dispatch_analysis"):
        project = _project(client)
        draft = _document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            publication="draft",
        )
        path = f"/api/v1/projects/{project['id']}/analysis-runs"
        headers = {"Idempotency-Key": f"batch-{uuid4().hex}"}
        body = {
            "mode": "draft_review",
            "target_document_ids": [draft["id"]],
            "sensitivity": "balanced",
        }
        first = client.post(path, headers=headers, json=body)
        replay = client.post(path, headers=headers, json=body)
        changed = client.post(
            path,
            headers=headers,
            json={**body, "sensitivity": "exploratory"},
        )
    assert first.status_code == replay.status_code == 202
    assert first.json()["id"] == replay.json()["id"]
    assert replay.json()["deduplicated"] is True
    assert changed.status_code == 409
    assert changed.json()["detail"]["code"] == "idempotency_key_conflict"


def test_retry_copies_frozen_batch_semantics():
    with TestClient(app) as client, patch("app.main.dispatch_analysis"):
        project = _project(client)
        _document(
            client, project["id"], name="world.md", role="canon", publication="published"
        )
        draft = _document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            publication="draft",
        )
        original = client.post(
            f"/api/v1/projects/{project['id']}/analysis-runs",
            json={
                "mode": "draft_review",
                "target_document_ids": [draft["id"]],
                "sensitivity": "exploratory",
            },
        ).json()
        with SessionLocal() as db:
            db.get(AnalysisRunRow, original["id"]).status = "failed"
            db.commit()
        retry = client.post(f"/api/v1/analysis-runs/{original['id']}/retry")
        assert retry.status_code == 202, retry.text
        retried = client.get(f"/api/v1/analysis-runs/{retry.json()['id']}").json()
        original_status = client.get(
            f"/api/v1/analysis-runs/{original['id']}"
        ).json()
    assert retried["review_batch"] == original["review_batch"]
    assert [row["batch_role"] for row in retried["input_documents"]] == [
        row["batch_role"]
        for row in original_status["input_documents"]
    ]


def test_draft_recheck_replaces_only_logical_targets_and_rederives_background():
    with TestClient(app) as client, patch("app.main.dispatch_analysis"):
        project = _project(client)
        world = _document(
            client, project["id"], name="world.md", role="canon", publication="published"
        )
        draft = _document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            publication="draft",
            content="第一版",
        )
        baseline = client.post(
            f"/api/v1/projects/{project['id']}/analysis-runs",
            json={"mode": "draft_review", "target_document_ids": [draft["id"]]},
        ).json()
        with SessionLocal() as db:
            db.get(AnalysisRunRow, baseline["id"]).status = "completed"
            db.commit()
        current = _document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            publication="draft",
            content="第二版",
            replace_document_id=draft["id"],
        )
        unrelated = _document(
            client,
            project["id"],
            name="another-draft.md",
            role="chapter",
            publication="draft",
        )
        response = client.post(
            f"/api/v1/analysis-runs/{baseline['id']}/rechecks"
        )
        assert response.status_code == 202, response.text
        batch = response.json()["review_batch"]
    assert batch["target_document_ids"] == [current["id"]]
    assert batch["background_document_ids"] == [world["id"]]
    assert unrelated["id"] not in batch["target_document_ids"]


def _issue(*document_ids: str) -> ConsistencyIssue:
    return ConsistencyIssue(
        category=IssueCategory.fact_conflict,
        severity=Severity.high,
        confidence=0.9,
        title="冲突",
        explanation="冲突",
        evidence=[
            EvidenceSpan(
                document_id=document_id,
                document_name=f"{document_id}.md",
                line_start=1,
                line_end=1,
                text="证据",
            )
            for document_id in document_ids
        ],
        suggestion="核对",
    )


def test_draft_issue_boundary_suppresses_background_only_findings():
    result = PipelineResult(
        issues=[_issue("background"), _issue("background", "target")]
    )
    documents = [
        DocumentInput(id="background", name="world.md", content="设定"),
        DocumentInput(id="target", name="draft.md", content="新稿"),
    ]
    _enforce_draft_issue_boundary(
        result,
        batch_mode="draft_review",
        input_metadata=[
            {"document_id": "background", "batch_role": "background"},
            {"document_id": "target", "batch_role": "target"},
        ],
        documents=documents,
    )
    assert len(result.issues) == 1
    assert {span.document_id for span in result.issues[0].evidence} == {
        "background",
        "target",
    }
    assert result.diagnostics["review_batch"][
        "suppressed_background_only_issues"
    ] == 1


@pytest.mark.parametrize(
    "sensitivity", ["conservative", "balanced", "exploratory"]
)
def test_run_sensitivity_drives_stage_local_settings_without_global_mutation(
    sensitivity: str,
):
    global_value = settings.character_consistency_sensitivity
    run = AnalysisRunRow(project_id=str(uuid4()), sensitivity=sensitivity)
    local = _character_stage_settings_for_run(settings, run)
    assert local.character_consistency_sensitivity == sensitivity
    assert settings.character_consistency_sensitivity == global_value


def test_context_confirmation_can_correct_role_and_stale_cas_rolls_back_role():
    with TestClient(app) as client:
        project = _project(client)
        document = _document(
            client,
            project["id"],
            name="misclassified.md",
            role="chapter",
            publication="draft",
            confirmed=False,
        )
        path = (
            f"/api/v1/projects/{project['id']}/documents/{document['id']}"
            "/narrative-context/revisions"
        )
        corrected = client.post(
            path,
            json={
                "expected_revision": 1,
                "document_role": "canon",
                "resolution_state": "confirmed",
                "publication_status": "published",
                "scope": _scope(),
            },
        )
        assert corrected.status_code == 201, corrected.text
        assert corrected.json()["document_role"] == "canon"
        assert corrected.json()["authority_tier"] == "core_canon"
        stale = client.post(
            path,
            json={
                "expected_revision": 1,
                "document_role": "chapter",
                "resolution_state": "confirmed",
                "publication_status": "published",
                "scope": _scope(),
            },
        )
        assert stale.status_code == 409
        old_shape = client.post(
            path,
            json={
                "expected_revision": 2,
                "resolution_state": "confirmed",
                "publication_status": "published",
                "scope": _scope(),
            },
        )
        assert old_shape.status_code == 201, old_shape.text
        assert old_shape.json()["document_role"] == "canon"
        shown = client.get(f"/api/v1/projects/{project['id']}").json()
    assert shown["documents"][0]["document_role"] == "canon"
    assert shown["documents"][0]["narrative_context"]["authority_tier"] == "core_canon"


@contextmanager
def _required_auth():
    original = {
        "auth_mode": settings.auth_mode,
        "auth_secret_key": settings.auth_secret_key,
        "auth_cookie_secure": settings.auth_cookie_secure,
        "auth_cookie_samesite": settings.auth_cookie_samesite,
    }
    settings.auth_mode = "required"
    settings.auth_secret_key = "review-batch-permission-test-secret-32-bytes"
    settings.auth_cookie_secure = False
    settings.auth_cookie_samesite = "lax"
    write_limiter.events.clear()
    try:
        yield
    finally:
        for key, value in original.items():
            setattr(settings, key, value)
        write_limiter.events.clear()


def _register(client: TestClient, label: str) -> dict[str, str]:
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


def test_role_correction_and_batch_targets_are_workspace_isolated():
    with _required_auth(), TestClient(app) as owner, TestClient(app) as other:
        owner_headers = _register(owner, "owner")
        other_headers = _register(other, "other")
        project_response = owner.post(
            "/api/v1/projects", headers=owner_headers, json={"name": "private"}
        )
        assert project_response.status_code == 201, project_response.text
        project = project_response.json()
        document_response = owner.post(
            f"/api/v1/projects/{project['id']}/documents/text",
            headers=owner_headers,
            json={
                "name": "draft.md",
                "content": "私有草稿",
                "document_role": "chapter",
                "narrative_context": {
                    "resolution_state": "confirmed",
                    "publication_status": "draft",
                    "scope": _scope(),
                },
            },
        )
        assert document_response.status_code == 201, document_response.text
        document = document_response.json()
        correction = other.post(
            f"/api/v1/projects/{project['id']}/documents/{document['id']}"
            "/narrative-context/revisions",
            headers=other_headers,
            json={
                "expected_revision": 1,
                "document_role": "canon",
                "resolution_state": "confirmed",
                "publication_status": "published",
                "scope": _scope(),
            },
        )
        batch = other.post(
            f"/api/v1/projects/{project['id']}/analysis-runs",
            headers=other_headers,
            json={
                "mode": "draft_review",
                "target_document_ids": [document["id"]],
            },
        )
    assert correction.status_code == 404
    assert batch.status_code == 404


def test_frozen_signature_and_context_store_batch_roles():
    with TestClient(app) as client, patch("app.main.dispatch_analysis"):
        project = _project(client)
        _document(
            client, project["id"], name="world.md", role="canon", publication="published"
        )
        draft = _document(
            client,
            project["id"],
            name="draft.md",
            role="chapter",
            publication="draft",
        )
        response = client.post(
            f"/api/v1/projects/{project['id']}/analysis-runs",
            json={"mode": "draft_review", "target_document_ids": [draft["id"]]},
        )
        assert response.status_code == 202, response.text
        run_id = response.json()["id"]
    with SessionLocal() as db:
        rows = list(
            db.execute(
                select(AnalysisRunInputRow, AnalysisRunInputContextRow)
                .join(
                    AnalysisRunInputContextRow,
                    AnalysisRunInputContextRow.input_id == AnalysisRunInputRow.id,
                )
                .where(AnalysisRunInputRow.run_id == run_id)
                .order_by(AnalysisRunInputRow.ordinal)
            ).all()
        )
        assert {context.batch_role for _, context in rows} == {
            "target",
            "background",
        }
        assert db.get(DocumentContextRow, draft["id"]).document_role == "chapter"
