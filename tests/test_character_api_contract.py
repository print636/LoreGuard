from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.character_traits import upsert_character_trait_candidate
from app.db import AnalysisRunInputRow, AnalysisRunRow, SessionLocal
from app.main import app, write_limiter


@pytest.fixture(autouse=True)
def _clear_write_limiter():
    write_limiter.events.clear()
    yield
    write_limiter.events.clear()


def _project_and_document(client: TestClient) -> tuple[dict, dict]:
    project_response = client.post(
        "/api/v1/projects", json={"name": f"角色接口-{uuid4().hex}"}
    )
    assert project_response.status_code == 201, project_response.text
    project = project_response.json()
    document_response = client.post(
        f"/api/v1/projects/{project['id']}/documents/text",
        json={
            "name": "character.md",
            "content": "林澈一直喜欢蜜瓜。",
            "document_role": "character_profile",
        },
    )
    assert document_response.status_code == 201, document_response.text
    return project, document_response.json()


def _completed_run(client: TestClient, project_id: str) -> dict:
    with patch("app.main.dispatch_analysis"):
        response = client.post(f"/api/v1/projects/{project_id}/analysis-runs")
    assert response.status_code == 202, response.text
    with SessionLocal() as db:
        run = db.get(AnalysisRunRow, response.json()["id"])
        assert run is not None
        run.status = "completed"
        db.commit()
    return response.json()


def _candidate(
    project_id: str,
    run_id: str,
    *,
    trait_key: str = "食物偏好:蜜瓜",
    value: str = "喜欢蜜瓜",
    polarity: str = "positive",
) -> str:
    with SessionLocal() as db:
        snapshot = db.scalar(
            select(AnalysisRunInputRow).where(
                AnalysisRunInputRow.run_id == run_id
            )
        )
        assert snapshot is not None
        row, created = upsert_character_trait_candidate(
            db,
            project_id=project_id,
            source_run_id=run_id,
            candidate={
                "character_key": "林澈",
                "character_display_name": "林澈",
                "trait_type": "preference",
                "trait_key": trait_key,
                "value": value,
                "polarity": polarity,
                "stability": "stable",
                "contexts": ["日常饮食"],
                "origin": "explicit_setting",
                "authority_tier": "formal_record",
                "confidence": 0.95,
                "scope": {
                    "schema_version": 1,
                    "timeline_key": "main",
                    "release": {"key": "v1", "ordinal": 1},
                    "branch": {"path": ["main"]},
                },
                "evidence": [
                    {
                        "input_id": snapshot.id,
                        "document_id": snapshot.document_id,
                        "document_name": snapshot.document_name,
                        "document_version": snapshot.document_version,
                        "content_sha256": snapshot.content_sha256,
                        "line_start": 1,
                        "line_end": 1,
                        "text": snapshot.content,
                    }
                ],
                "generator_version": "api-contract-test-v1",
                "provenance": {"extractor": "test", "record_index": 0},
            },
        )
        assert created
        db.commit()
        return row.id


def _decision_path(project_id: str, candidate_id: str) -> str:
    return (
        f"/api/v1/projects/{project_id}/characters/{quote('林澈', safe='')}"
        f"/profile-candidates/{candidate_id}/decisions"
    )


def test_character_roster_envelope_distinguishes_readiness_states():
    with TestClient(app) as client:
        project_response = client.post(
            "/api/v1/projects", json={"name": f"角色空态-{uuid4().hex}"}
        )
        project = project_response.json()
        empty = client.get(f"/api/v1/projects/{project['id']}/characters")
        assert empty.status_code == 200, empty.text
        assert empty.json()["readiness"] == "no_documents"
        assert empty.json()["items"] == []

        document_response = client.post(
            f"/api/v1/projects/{project['id']}/documents/text",
            json={"name": "character.md", "content": "林澈喜欢蜜瓜。"},
        )
        assert document_response.status_code == 201, document_response.text
        waiting = client.get(f"/api/v1/projects/{project['id']}/characters")
        assert waiting.json()["readiness"] == "no_completed_run"

        run = _completed_run(client, project["id"])
        not_generated = client.get(
            f"/api/v1/projects/{project['id']}/characters"
        )
        assert not_generated.json()["readiness"] == "not_generated"
        assert not_generated.json()["source_run_id"] == run["id"]
        assert not_generated.json()["model_coverage"] == "unknown"


def test_replaced_source_marks_pending_candidate_stale_and_blocks_confirmation():
    with TestClient(app) as client:
        project, document = _project_and_document(client)
        run = _completed_run(client, project["id"])
        candidate_id = _candidate(project["id"], run["id"])

        roster = client.get(f"/api/v1/projects/{project['id']}/characters")
        assert roster.status_code == 200, roster.text
        assert roster.json()["readiness"] == "ready"
        assert roster.json()["total"] == 1

        replacement = client.post(
            f"/api/v1/projects/{project['id']}/documents/text",
            json={"name": document["name"], "content": "林澈现在喜欢苹果。"},
        )
        assert replacement.status_code == 201, replacement.text
        detail_path = (
            f"/api/v1/projects/{project['id']}/characters/"
            f"{quote('林澈', safe='')}/profile-candidates/{candidate_id}"
        )
        detail = client.get(detail_path)
        assert detail.status_code == 200, detail.text
        assert detail.json()["status"] == "stale"
        assert detail.json()["reviewable"] is False

        decision = client.post(
            _decision_path(project["id"], candidate_id),
            json={"decision": "confirm", "expected_revision": 0},
        )
        assert decision.status_code == 409, decision.text
        assert decision.json()["detail"]["code"] == "character_trait_candidate_stale"


def test_narrative_context_revision_marks_candidate_stale_and_blocks_confirmation():
    with TestClient(app) as client:
        project, document = _project_and_document(client)
        run = _completed_run(client, project["id"])
        candidate_id = _candidate(project["id"], run["id"])
        detail_path = (
            f"/api/v1/projects/{project['id']}/characters/"
            f"{quote('林澈', safe='')}/profile-candidates/{candidate_id}"
        )

        before = client.get(detail_path)
        assert before.status_code == 200, before.text
        assert before.json()["reviewable"] is True

        revision = client.post(
            f"/api/v1/projects/{project['id']}/documents/{document['id']}"
            "/narrative-context/revisions",
            json={
                "expected_revision": 1,
                "resolution_state": "confirmed",
                "publication_status": "published",
                "scope": {
                    "timeline_key": "main",
                    "branch": {
                        "path": ["main", "route-a"],
                        "exclusive_group": "route",
                    },
                },
            },
        )
        assert revision.status_code == 201, revision.text

        detail = client.get(detail_path)
        assert detail.status_code == 200, detail.text
        assert detail.json()["status"] == "stale"
        assert detail.json()["reviewable"] is False

        decision = client.post(
            _decision_path(project["id"], candidate_id),
            json={"decision": "confirm", "expected_revision": 0},
        )
        assert decision.status_code == 409, decision.text
        assert decision.json()["detail"]["code"] == "character_trait_candidate_stale"


def test_semantically_equivalent_trait_labels_cannot_confirm_opposite_values():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        positive = _candidate(project["id"], run["id"])
        negative = _candidate(
            project["id"],
            run["id"],
            trait_key="蜜瓜喜好",
            value="讨厌蜜瓜",
            polarity="negative",
        )
        confirmed = client.post(
            _decision_path(project["id"], positive),
            json={"decision": "confirm", "expected_revision": 0},
        )
        assert confirmed.status_code == 201, confirmed.text

        conflict = client.post(
            _decision_path(project["id"], negative),
            json={"decision": "confirm", "expected_revision": 0},
        )
        assert conflict.status_code == 409, conflict.text
        assert (
            conflict.json()["detail"]["code"]
            == "character_trait_confirmation_conflict"
        )
