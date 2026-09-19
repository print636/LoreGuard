from __future__ import annotations

import unittest
from contextlib import contextmanager
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import StatementError

from app.db import (
    AnalysisRunRow,
    DocumentRow,
    FeedbackRow,
    IssueRow,
    ProjectRow,
    SessionLocal,
)
from app.main import app, settings, write_limiter
from app.service import capture_run_inputs


@contextmanager
def required_auth_settings():
    fields = {
        "auth_mode": settings.auth_mode,
        "auth_secret_key": settings.auth_secret_key,
        "auth_cookie_secure": settings.auth_cookie_secure,
        "auth_cookie_samesite": settings.auth_cookie_samesite,
    }
    settings.auth_mode = "required"
    settings.auth_secret_key = "workspace-isolation-test-secret-key-32-bytes"
    settings.auth_cookie_secure = False
    settings.auth_cookie_samesite = "lax"
    write_limiter.events.clear()
    try:
        yield
    finally:
        for name, value in fields.items():
            setattr(settings, name, value)
        write_limiter.events.clear()


def register(client: TestClient, label: str) -> tuple[dict, str]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"{label}-{uuid4().hex}@example.com",
            "password": "correct horse battery staple",
            "display_name": label,
        },
    )
    assert response.status_code == 201, response.text
    return response.json(), response.headers["X-CSRF-Token"]


class WorkspaceIsolationTests(unittest.TestCase):
    def test_required_mode_project_ownership_cannot_silently_default(self):
        with required_auth_settings(), TestClient(app):
            with SessionLocal() as db:
                db.add(ProjectRow(name="must fail closed"))
                with self.assertRaisesRegex(
                    StatementError, "workspace_id is required"
                ):
                    db.commit()
                db.rollback()

    def test_two_accounts_are_isolated_across_all_nested_resources(self):
        with required_auth_settings(), TestClient(app) as owner, TestClient(app) as other:
            owner_identity, owner_csrf = register(owner, "owner")
            _, other_csrf = register(other, "other")
            owner_headers = {"X-CSRF-Token": owner_csrf}
            other_headers = {"X-CSRF-Token": other_csrf}

            created = owner.post(
                "/api/v1/projects",
                headers=owner_headers,
                json={"name": "private story", "description": "private"},
            )
            self.assertEqual(201, created.status_code, created.text)
            project_id = created.json()["id"]
            first = owner.post(
                f"/api/v1/projects/{project_id}/documents/text",
                headers=owner_headers,
                json={"name": "chapter.md", "content": "first version"},
            )
            self.assertEqual(201, first.status_code, first.text)
            first_id = first.json()["id"]
            second = owner.post(
                f"/api/v1/projects/{project_id}/documents/text",
                headers=owner_headers,
                json={
                    "name": "chapter.md",
                    "content": "second version",
                    "replace_document_id": first_id,
                },
            )
            self.assertEqual(201, second.status_code, second.text)
            second_id = second.json()["id"]

            with SessionLocal() as db:
                active_document = db.scalar(
                    select(DocumentRow).where(DocumentRow.id == second_id)
                )
                completed = AnalysisRunRow(
                    project_id=project_id,
                    requested_by_user_id=owner_identity["user"]["id"],
                    status="completed",
                )
                running = AnalysisRunRow(
                    project_id=project_id,
                    requested_by_user_id=owner_identity["user"]["id"],
                    status="running",
                )
                failed = AnalysisRunRow(
                    project_id=project_id,
                    requested_by_user_id=owner_identity["user"]["id"],
                    status="failed",
                )
                db.add_all([completed, running, failed])
                db.flush()
                capture_run_inputs(db, failed, [active_document])
                issue = IssueRow(
                    run_id=completed.id,
                    category="fact_conflict",
                    severity="medium",
                    confidence=0.9,
                    title="private issue",
                    explanation="private explanation",
                    evidence=[],
                    suggestion="private suggestion",
                    extra={},
                )
                db.add(issue)
                db.commit()
                completed_id = completed.id
                running_id = running.id
                failed_id = failed.id
                issue_id = issue.id

            diff_path = (
                f"/api/v1/projects/{project_id}/documents/diff"
                f"?from_document_id={first_id}&to_document_id={second_id}"
            )
            self.assertEqual(200, owner.get(diff_path).status_code)
            self.assertEqual(
                202,
                owner.post(
                    f"/api/v1/analysis-runs/{running_id}/cancel",
                    headers=owner_headers,
                ).status_code,
            )
            with patch("app.main.dispatch_analysis"):
                retry = owner.post(
                    f"/api/v1/analysis-runs/{failed_id}/retry",
                    headers=owner_headers,
                )
            self.assertEqual(202, retry.status_code, retry.text)
            accepted = owner.post(
                f"/api/v1/issues/{issue_id}/feedback",
                headers=owner_headers,
                json={"label": "accepted", "comment": "confirmed"},
            )
            self.assertEqual(201, accepted.status_code, accepted.text)
            with SessionLocal() as db:
                feedback = db.scalar(
                    select(FeedbackRow).where(FeedbackRow.issue_id == issue_id)
                )
                self.assertEqual(
                    owner_identity["user"]["id"], feedback.created_by_user_id
                )

            owner_get_paths = [
                f"/api/v1/projects/{project_id}",
                f"/api/v1/projects/{project_id}/documents?include_history=true",
                f"/api/v1/projects/{project_id}/analysis-runs",
                f"/api/v1/analysis-runs/{completed_id}",
                f"/api/v1/analysis-runs/{completed_id}/issues",
                f"/api/v1/analysis-runs/{completed_id}/records",
                f"/api/v1/analysis-runs/{completed_id}/clarifications",
                f"/api/v1/analysis-runs/{completed_id}/graph",
                f"/api/v1/analysis-runs/{completed_id}/timeline",
                f"/api/v1/analysis-runs/{completed_id}/diagnostics",
                f"/api/v1/analysis-runs/{completed_id}/events",
                f"/api/v1/issues/{issue_id}/feedback",
            ]
            for path in owner_get_paths:
                with self.subTest(owner_path=path):
                    self.assertEqual(200, owner.get(path).status_code)

            cross_get_paths = [*owner_get_paths, diff_path]
            for path in cross_get_paths:
                with self.subTest(cross_path=path):
                    response = other.get(path)
                    self.assertEqual(404, response.status_code, response.text)

            cross_write_paths = [
                f"/api/v1/analysis-runs/{running_id}/cancel",
                f"/api/v1/analysis-runs/{failed_id}/retry",
            ]
            for path in cross_write_paths:
                with self.subTest(cross_write=path):
                    response = other.post(path, headers=other_headers)
                    self.assertEqual(404, response.status_code, response.text)
            cross_feedback = other.post(
                f"/api/v1/issues/{issue_id}/feedback",
                headers=other_headers,
                json={"label": "false_positive", "comment": "steal"},
            )
            self.assertEqual(404, cross_feedback.status_code, cross_feedback.text)

            self.assertEqual([], other.get("/api/v1/projects").json())
            self.assertEqual(
                404,
                other.post(
                    f"/api/v1/projects/{project_id}/documents/text",
                    headers=other_headers,
                    json={"name": "intrusion.md", "content": "blocked"},
                ).status_code,
            )

    def test_demo_and_started_run_are_bound_to_current_identity(self):
        with required_auth_settings(), TestClient(app) as owner, TestClient(app) as other:
            owner_identity, owner_csrf = register(owner, "demo-owner")
            register(other, "demo-other")
            owner_headers = {"X-CSRF-Token": owner_csrf}
            project_ids: list[str] = []
            for path in ("/api/v1/demo", "/api/v1/demo/advanced"):
                response = owner.post(path, headers=owner_headers)
                self.assertEqual(201, response.status_code, response.text)
                project_ids.append(response.json()["id"])

            with patch("app.main.dispatch_analysis") as dispatch:
                started = owner.post(
                    f"/api/v1/projects/{project_ids[0]}/analysis-runs",
                    headers=owner_headers,
                )
            self.assertEqual(202, started.status_code, started.text)
            dispatch.assert_called_once_with(started.json()["id"])
            with SessionLocal() as db:
                projects = db.scalars(
                    select(ProjectRow).where(ProjectRow.id.in_(project_ids))
                ).all()
                self.assertEqual(
                    {owner_identity["workspace"]["id"]},
                    {project.workspace_id for project in projects},
                )
                run = db.get(AnalysisRunRow, started.json()["id"])
                self.assertEqual(owner_identity["user"]["id"], run.requested_by_user_id)

            for project_id in project_ids:
                self.assertEqual(
                    404,
                    other.get(f"/api/v1/projects/{project_id}").status_code,
                )

    def test_business_endpoints_require_session_and_csrf(self):
        with required_auth_settings(), TestClient(app) as anonymous:
            self.assertEqual(401, anonymous.get("/api/v1/projects").status_code)
            self.assertEqual(
                401,
                anonymous.get("/api/v1/evaluations/latest").status_code,
            )
            self.assertEqual(
                401,
                anonymous.post("/api/v1/model/provider-check").status_code,
            )

        with required_auth_settings(), TestClient(app) as client:
            _, csrf = register(client, "csrf")
            self.assertEqual(
                403,
                client.post(
                    "/api/v1/projects",
                    json={"name": "missing csrf", "description": ""},
                ).status_code,
            )
            self.assertEqual(
                403,
                client.post("/api/v1/model/provider-check").status_code,
            )
            self.assertEqual(
                200,
                client.post(
                    "/api/v1/model/provider-check",
                    headers={"X-CSRF-Token": csrf},
                ).status_code,
            )
            self.assertEqual(
                200,
                client.get("/api/v1/evaluations/latest").status_code,
            )


if __name__ == "__main__":
    unittest.main()
