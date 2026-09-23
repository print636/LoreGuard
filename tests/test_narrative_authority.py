from __future__ import annotations

import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect, select

from app.character_traits import (
    _validate_safe_provenance,
    upsert_character_trait_candidate,
)
from app.db import (
    AnalysisRunCharacterTraitInputRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    Base,
    CharacterTraitCandidateRow,
    DocumentNarrativeContextRevisionRow,
    FeedbackRow,
    IssueRow,
    SessionLocal,
)
from app.main import app, settings, write_limiter
from app.narrative_context import (
    NarrativeScopeV1,
    canonical_scope_payload,
    payload_sha256,
    scope_relation,
)
ROOT = Path(__file__).resolve().parents[1]
NARRATIVE_TABLES = {
    "document_narrative_context_revisions",
    "analysis_run_input_narrative_context",
    "character_trait_candidates",
    "character_trait_reviews",
    "analysis_run_character_trait_inputs",
}


@pytest.fixture(autouse=True)
def _clear_write_limiter():
    write_limiter.events.clear()
    yield
    write_limiter.events.clear()


def _project_and_document(
    client: TestClient,
    *,
    content: str = "林澈一直喜欢蜜瓜。",
    narrative_context: dict | None = None,
) -> tuple[dict, dict]:
    project = client.post(
        "/api/v1/projects",
        json={"name": f"叙事权威-{uuid4().hex}"},
    )
    assert project.status_code == 201, project.text
    body: dict = {
        "name": "chapter.md",
        "content": content,
        "document_role": "chapter",
    }
    if narrative_context is not None:
        body["narrative_context"] = narrative_context
    document = client.post(
        f"/api/v1/projects/{project.json()['id']}/documents/text",
        json=body,
    )
    assert document.status_code == 201, document.text
    return project.json(), document.json()


def _start_frozen_run(
    client: TestClient, project_id: str, *, status: str = "completed"
) -> dict:
    with patch("app.main.dispatch_analysis"):
        response = client.post(
            f"/api/v1/projects/{project_id}/analysis-runs"
        )
    assert response.status_code == 202, response.text
    with SessionLocal() as db:
        row = db.get(AnalysisRunRow, response.json()["id"])
        assert row is not None
        row.status = status
        db.commit()
    return response.json()


def _candidate_payload(snapshot: AnalysisRunInputRow, **overrides) -> dict:
    row = {
        "character_key": "林澈",
        "character_display_name": "林澈",
        "trait_type": "preference",
        "trait_key": "食物偏好:蜜瓜",
        "value": "喜欢蜜瓜",
        "polarity": "positive",
        "stability": "stable",
        "contexts": [" 日常　饮食 ", "日常 饮食"],
        "origin": "explicit_setting",
        "confidence": 0.94,
        "scope": {
            "schema_version": 1,
            "timeline_key": "main",
            "release": {"key": "v1", "ordinal": 1},
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
                "text": snapshot.content.splitlines()[0],
            }
        ],
        "generator_version": "test-v1",
        "provenance": {"extractor": "unit-test", "record_index": 0},
    }
    row.update(overrides)
    return row


def _create_candidate(project_id: str, run_id: str, **overrides) -> str:
    with SessionLocal() as db:
        snapshot = db.scalar(
            select(AnalysisRunInputRow).where(
                AnalysisRunInputRow.run_id == run_id
            )
        )
        assert snapshot is not None
        candidate, created = upsert_character_trait_candidate(
            db,
            project_id=project_id,
            source_run_id=run_id,
            candidate=_candidate_payload(snapshot, **overrides),
        )
        assert created
        db.commit()
        return candidate.id


def _confirm_candidate(
    client: TestClient,
    project_id: str,
    candidate_id: str,
    *,
    expected_revision: int = 0,
) -> dict:
    character = quote("林澈", safe="")
    response = client.post(
        (
            f"/api/v1/projects/{project_id}/characters/{character}"
            f"/profile-candidates/{candidate_id}/decisions"
        ),
        json={"decision": "confirm", "expected_revision": expected_revision},
        headers={"Idempotency-Key": f"confirm-{uuid4().hex}"},
    )
    assert response.status_code == 201, response.text
    return response.json()


@contextmanager
def _required_auth():
    fields = {
        "auth_mode": settings.auth_mode,
        "auth_secret_key": settings.auth_secret_key,
        "auth_cookie_secure": settings.auth_cookie_secure,
        "auth_cookie_samesite": settings.auth_cookie_samesite,
    }
    settings.auth_mode = "required"
    settings.auth_secret_key = "narrative-authority-test-secret-key-32-bytes"
    settings.auth_cookie_secure = False
    settings.auth_cookie_samesite = "lax"
    write_limiter.events.clear()
    try:
        yield
    finally:
        for key, value in fields.items():
            setattr(settings, key, value)
        write_limiter.events.clear()


def _register(client: TestClient, label: str) -> tuple[dict, dict[str, str]]:
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": f"{label}-{uuid4().hex}@example.com",
            "password": "correct horse battery staple",
            "display_name": label,
        },
    )
    assert response.status_code == 201, response.text
    return response.json(), {"X-CSRF-Token": response.headers["X-CSRF-Token"]}


def test_scope_hash_is_canonical_and_unresolved_scope_fails_closed():
    rich = {
        "schema_version": 1,
        "timeline_key": "main",
        "release": {"ordinal": 3, "key": "v3"},
        "branch": {"path": ["main", "route-a"], "exclusive_group": "route"},
    }
    parsed = NarrativeScopeV1.model_validate(rich)
    assert payload_sha256(canonical_scope_payload(rich)) == payload_sha256(
        canonical_scope_payload(parsed)
    )
    sibling = {
        **rich,
        "branch": {"path": ["main", "route-b"], "exclusive_group": "route"},
    }
    descendant = {
        **rich,
        "branch": {"path": ["main", "route-a", "chapter-2"], "exclusive_group": "route"},
    }
    assert (
        scope_relation(
            rich,
            sibling,
            first_resolution="confirmed",
            second_resolution="confirmed",
        )
        == "incompatible"
    )
    assert (
        scope_relation(
            rich,
            descendant,
            first_resolution="confirmed",
            second_resolution="confirmed",
        )
        == "compatible"
    )
    assert (
        scope_relation(
            rich,
            descendant,
            first_resolution="unresolved",
            second_resolution="confirmed",
        )
        == "unknown"
    )


@pytest.mark.parametrize(
    "forbidden_key",
    [
        "openai_api_key",
        "api-key",
        "system_prompt",
        "rawResponse",
        "provider_response",
        "response-body",
        "base.url",
        "clientSecret",
    ],
)
def test_candidate_provenance_rejects_normalized_sensitive_keys(forbidden_key):
    with pytest.raises(ValueError, match="forbidden field"):
        _validate_safe_provenance(
            {"safe": {"nested": {forbidden_key: "must-not-persist"}}}
        )


def test_upload_derives_authority_and_context_revision_uses_cas():
    with TestClient(app) as client:
        project = client.post(
            "/api/v1/projects", json={"name": f"设定-{uuid4().hex}"}
        ).json()
        response = client.post(
            f"/api/v1/projects/{project['id']}/documents/text",
            json={
                "name": "world.md",
                "content": "世界设定",
                "document_role": "canon",
                "narrative_context": {
                    "resolution_state": "confirmed",
                    "publication_status": "published",
                    "scope": {
                        "timeline_key": "main",
                        "release": {"key": "v1", "ordinal": 1},
                    },
                },
            },
        )
        assert response.status_code == 201, response.text
        document = response.json()
        assert document["narrative_context"]["authority_tier"] == "core_canon"
        assert document["narrative_context"]["context_revision"] == 1

        forbidden = client.post(
            f"/api/v1/projects/{project['id']}/documents/text",
            json={
                "name": "forbidden.md",
                "content": "不能自行提权",
                "narrative_context": {
                    "resolution_state": "confirmed",
                    "authority_tier": "core_canon",
                },
            },
        )
        assert forbidden.status_code == 422

        path = (
            f"/api/v1/projects/{project['id']}/documents/{document['id']}"
            "/narrative-context/revisions"
        )
        update_payload = {
            "expected_revision": 1,
            "resolution_state": "confirmed",
            "publication_status": "retired",
            "scope": {
                "timeline_key": "main",
                "release": {"key": "v2", "ordinal": 2},
            },
        }
        updated = client.post(path, json=update_payload)
        assert updated.status_code == 201, updated.text
        assert updated.json()["revision"] == 2
        stale = client.post(path, json=update_payload)
        assert stale.status_code == 409
        assert stale.json()["detail"]["actual_revision"] == 2


def test_run_snapshot_is_immutable_retry_copies_and_recheck_reads_current():
    initial_context = {
        "resolution_state": "confirmed",
        "publication_status": "published",
        "scope": {
            "timeline_key": "main",
            "release": {"key": "v1", "ordinal": 1},
        },
    }
    with TestClient(app) as client:
        project, document = _project_and_document(
            client, narrative_context=initial_context
        )
        failed = _start_frozen_run(client, project["id"], status="failed")
        before = client.get(f"/api/v1/analysis-runs/{failed['id']}").json()
        path = (
            f"/api/v1/projects/{project['id']}/documents/{document['id']}"
            "/narrative-context/revisions"
        )
        current = client.post(
            path,
            json={
                "expected_revision": 1,
                "resolution_state": "confirmed",
                "publication_status": "published",
                "scope": {
                    "timeline_key": "main",
                    "release": {"key": "v2", "ordinal": 2},
                },
            },
        )
        assert current.status_code == 201, current.text
        unchanged = client.get(f"/api/v1/analysis-runs/{failed['id']}").json()
        assert unchanged["input_documents"] == before["input_documents"]

        with patch("app.main.dispatch_analysis"):
            retry = client.post(f"/api/v1/analysis-runs/{failed['id']}/retry")
        assert retry.status_code == 202, retry.text
        retried = client.get(f"/api/v1/analysis-runs/{retry.json()['id']}").json()
        assert retried["input_documents"] == before["input_documents"]

        baseline = _start_frozen_run(client, project["id"], status="completed")
        candidate_id = _create_candidate(project["id"], baseline["id"])
        _confirm_candidate(client, project["id"], candidate_id)
        with patch("app.main.dispatch_analysis"):
            recheck = client.post(
                f"/api/v1/analysis-runs/{baseline['id']}/rechecks"
            )
        assert recheck.status_code == 202, recheck.text
        rechecked = client.get(
            f"/api/v1/analysis-runs/{recheck.json()['id']}"
        ).json()
        assert rechecked["confirmed_trait_count"] == 1
        assert (
            rechecked["input_documents"][0]["narrative_context"]["scope"]
            ["release"]["ordinal"]
            == 2
        )
        assert (
            before["input_documents"][0]["narrative_context"]["scope"]
            ["release"]["ordinal"]
            == 1
        )


def test_candidate_decision_is_idempotent_and_uses_revision_cas():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _start_frozen_run(client, project["id"])
        candidate_id = _create_candidate(project["id"], run["id"])
        character = quote("林澈", safe="")
        path = (
            f"/api/v1/projects/{project['id']}/characters/{character}"
            f"/profile-candidates/{candidate_id}/decisions"
        )
        key = f"review-{uuid4().hex}"
        body = {
            "decision": "confirm",
            "expected_revision": 0,
            "comment": "人工确认",
        }
        first = client.post(path, json=body, headers={"Idempotency-Key": key})
        assert first.status_code == 201, first.text
        assert first.json()["candidate"]["review_state"] == "confirmed"
        assert first.json()["candidate"]["revision"] == 1
        assert first.json()["candidate"]["origin"] == "explicit_setting"
        assert first.json()["candidate"]["contexts"] == ["日常 饮食"]
        assert "provenance" not in first.json()["candidate"]
        replay = client.post(path, json=body, headers={"Idempotency-Key": key})
        assert replay.status_code == 201, replay.text
        assert replay.json()["deduplicated"] is True
        conflicting_key_reuse = client.post(
            path,
            json={**body, "comment": "不同请求"},
            headers={"Idempotency-Key": key},
        )
        assert conflicting_key_reuse.status_code == 409
        stale = client.post(
            path,
            json={"decision": "reject", "expected_revision": 0},
        )
        assert stale.status_code == 409

        profile = client.get(
            f"/api/v1/projects/{project['id']}/characters/{character}"
        )
        assert profile.status_code == 200, profile.text
        assert len(profile.json()["confirmed_traits"]) == 1
        frozen = _start_frozen_run(client, project["id"])
        with SessionLocal() as db:
            snapshot = db.scalar(
                select(AnalysisRunCharacterTraitInputRow).where(
                    AnalysisRunCharacterTraitInputRow.run_id == frozen["id"]
                )
            )
            assert snapshot is not None
            assert snapshot.payload["origin"] == "explicit_setting"
            assert snapshot.payload["contexts"] == ["日常 饮食"]


def test_drift_issues_filter_character_before_pagination():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _start_frozen_run(client, project["id"], status="completed")
        with SessionLocal() as db:
            issues = {
                character: IssueRow(
                    run_id=run["id"],
                    category="character_drift",
                    severity="medium",
                    confidence=0.8,
                    title=f"{character}漂移",
                    explanation="测试",
                    evidence=[],
                    suggestion="",
                    extra={
                        "character_key": character,
                        "dimension": "preference",
                        "trait_key": "食物偏好",
                    },
                )
                for character in ("苏弦", "林澈")
            }
            db.add_all(issues.values())
            db.flush()
            db.add_all(
                [
                    FeedbackRow(
                        issue_id=issues["林澈"].id,
                        label="accepted",
                        created_at=datetime(2026, 1, 1, 12, 0, 0),
                    ),
                    FeedbackRow(
                        issue_id=issues["林澈"].id,
                        label="resolved",
                        created_at=datetime(2026, 1, 2, 12, 0, 0),
                    ),
                ]
            )
            db.commit()

        path = f"/api/v1/projects/{project['id']}/drift-issues"
        response = client.get(
            path,
            params={"character_key": "林 澈", "limit": 1, "offset": 0},
        )
        assert response.status_code == 200, response.text
        assert response.json()["total"] == 1
        assert len(response.json()["items"]) == 1
        assert response.json()["items"][0]["metadata"]["character_key"] == "林澈"
        assert response.json()["items"][0]["feedback_status"] == "resolved"
        assert response.json()["has_more"] is False
        unreviewed = client.get(
            path,
            params={"character_key": "苏弦", "limit": 1, "offset": 0},
        )
        assert unreviewed.status_code == 200, unreviewed.text
        assert unreviewed.json()["items"][0]["feedback_status"] == "unreviewed"
        assert client.get(path, params={"character_key": " "}).status_code == 422


def test_candidate_evidence_and_supersession_fail_closed_across_projects():
    with TestClient(app) as client:
        first_project, _ = _project_and_document(client, content="林澈喜欢蜜瓜。")
        second_project, _ = _project_and_document(client, content="苏弦喜欢茶。")
        first_run = _start_frozen_run(client, first_project["id"])
        second_run = _start_frozen_run(client, second_project["id"])
        with SessionLocal() as db:
            foreign_snapshot = db.scalar(
                select(AnalysisRunInputRow).where(
                    AnalysisRunInputRow.run_id == first_run["id"]
                )
            )
            assert foreign_snapshot is not None
            with pytest.raises(ValueError, match="outside the frozen run"):
                upsert_character_trait_candidate(
                    db,
                    project_id=second_project["id"],
                    source_run_id=second_run["id"],
                    candidate=_candidate_payload(foreign_snapshot),
                )


def test_supersession_requires_same_confirmed_identity_and_known_overlap():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        source = _start_frozen_run(client, project["id"], status="completed")
        original_scope = {
            "schema_version": 1,
            "timeline_key": "main",
            "release": {"key": "v1", "ordinal": 1},
            "branch": {
                "path": ["main", "route-a"],
                "exclusive_group": "route-a",
            },
        }
        original_id = _create_candidate(
            project["id"], source["id"], scope=original_scope
        )
        _confirm_candidate(client, project["id"], original_id)

        with SessionLocal() as db:
            snapshot = db.scalar(
                select(AnalysisRunInputRow).where(
                    AnalysisRunInputRow.run_id == source["id"]
                )
            )
            assert snapshot is not None
            with pytest.raises(ValueError, match="superseded candidate"):
                upsert_character_trait_candidate(
                    db,
                    project_id=project["id"],
                    source_run_id=source["id"],
                    candidate=_candidate_payload(
                        snapshot,
                        trait_type="current_state",
                        trait_key="当前状态",
                        value="紧张",
                        supersedes_candidate_id=original_id,
                        scope=original_scope,
                    ),
                )
            with pytest.raises(ValueError, match="superseded candidate"):
                upsert_character_trait_candidate(
                    db,
                    project_id=project["id"],
                    source_run_id=source["id"],
                    candidate=_candidate_payload(
                        snapshot,
                        value="不再喜欢蜜瓜",
                        polarity="negative",
                        supersedes_candidate_id=original_id,
                        scope={
                            "schema_version": 1,
                            "timeline_key": "main",
                            "release": {"key": "v1", "ordinal": 1},
                            "branch": {
                                "path": ["main", "route-b"],
                                "exclusive_group": "route-b",
                            },
                        },
                    ),
                )

        replacement_id = _create_candidate(
            project["id"],
            source["id"],
            value="不再喜欢蜜瓜",
            polarity="negative",
            supersedes_candidate_id=original_id,
            scope=original_scope,
        )
        with SessionLocal() as db:
            replacement = db.get(CharacterTraitCandidateRow, replacement_id)
            assert replacement is not None
            # Simulate a stale/legacy row whose supplied linkage no longer
            # matches the profile identity. Confirmation must re-check it.
            replacement.trait_type = "current_state"
            db.commit()

        character = quote("林澈", safe="")
        decision = client.post(
            (
                f"/api/v1/projects/{project['id']}/characters/{character}"
                f"/profile-candidates/{replacement_id}/decisions"
            ),
            json={"decision": "confirm", "expected_revision": 0},
        )
        assert decision.status_code == 409, decision.text
        assert decision.json()["detail"]["code"] == (
            "character_trait_supersession_conflict"
        )
        with SessionLocal() as db:
            assert db.get(CharacterTraitCandidateRow, original_id).review_state == (
                "confirmed"
            )
            assert db.get(CharacterTraitCandidateRow, replacement_id).review_state == (
                "pending"
            )


def test_retry_copies_exact_confirmed_profile_after_live_supersession():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        source = _start_frozen_run(client, project["id"], status="completed")
        original_id = _create_candidate(project["id"], source["id"])
        _confirm_candidate(client, project["id"], original_id)

        failed = _start_frozen_run(client, project["id"], status="failed")
        frozen = client.get(f"/api/v1/analysis-runs/{failed['id']}").json()
        assert frozen["confirmed_trait_count"] == 1

        replacement_id = _create_candidate(
            project["id"],
            source["id"],
            value="不再喜欢蜜瓜",
            polarity="negative",
            supersedes_candidate_id=original_id,
        )
        _confirm_candidate(client, project["id"], replacement_id)
        with patch("app.main.dispatch_analysis"):
            response = client.post(
                f"/api/v1/analysis-runs/{failed['id']}/retry"
            )
        assert response.status_code == 202, response.text
        retried = client.get(
            f"/api/v1/analysis-runs/{response.json()['id']}"
        ).json()
        assert retried["confirmed_trait_count"] == 1
        assert (
            retried["character_profile_snapshot_sha256"]
            == frozen["character_profile_snapshot_sha256"]
        )


def test_running_source_requires_explicit_internal_opt_in():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _start_frozen_run(client, project["id"], status="running")
        with SessionLocal() as db:
            snapshot = db.scalar(
                select(AnalysisRunInputRow).where(
                    AnalysisRunInputRow.run_id == run["id"]
                )
            )
            assert snapshot is not None
            candidate = _candidate_payload(snapshot)
            with pytest.raises(ValueError, match="source run is invalid"):
                upsert_character_trait_candidate(
                    db,
                    project_id=project["id"],
                    source_run_id=run["id"],
                    candidate=candidate,
                )
            row, created = upsert_character_trait_candidate(
                db,
                project_id=project["id"],
                source_run_id=run["id"],
                candidate=candidate,
                allow_running_source=True,
            )
            assert created
            assert row.review_state == "pending"
            db.rollback()


def test_new_resources_are_workspace_isolated_as_404():
    with _required_auth(), TestClient(app) as owner, TestClient(app) as other:
        _, owner_headers = _register(owner, "authority-owner")
        _, other_headers = _register(other, "authority-other")
        project_response = owner.post(
            "/api/v1/projects",
            headers=owner_headers,
            json={"name": f"私有叙事-{uuid4().hex}"},
        )
        assert project_response.status_code == 201, project_response.text
        project = project_response.json()
        document_response = owner.post(
            f"/api/v1/projects/{project['id']}/documents/text",
            headers=owner_headers,
            json={"name": "private.md", "content": "林澈喜欢蜜瓜。"},
        )
        assert document_response.status_code == 201, document_response.text
        document = document_response.json()
        with patch("app.main.dispatch_analysis"):
            run_response = owner.post(
                f"/api/v1/projects/{project['id']}/analysis-runs",
                headers=owner_headers,
            )
        assert run_response.status_code == 202, run_response.text
        run = run_response.json()
        with SessionLocal() as db:
            row = db.get(AnalysisRunRow, run["id"])
            row.status = "completed"
            db.commit()
        candidate_id = _create_candidate(project["id"], run["id"])
        character = quote("林澈", safe="")
        read_paths = [
            f"/api/v1/projects/{project['id']}/documents/{document['id']}/narrative-context",
            f"/api/v1/projects/{project['id']}/characters",
            f"/api/v1/projects/{project['id']}/characters/{character}",
            (
                f"/api/v1/projects/{project['id']}/characters/{character}"
                f"/profile-candidates/{candidate_id}"
            ),
            f"/api/v1/projects/{project['id']}/drift-issues",
        ]
        for path in read_paths:
            assert owner.get(path).status_code == 200
            assert other.get(path).status_code == 404
        decision = other.post(
            (
                f"/api/v1/projects/{project['id']}/characters/{character}"
                f"/profile-candidates/{candidate_id}/decisions"
            ),
            headers=other_headers,
            json={"decision": "confirm", "expected_revision": 0},
        )
        assert decision.status_code == 404


def test_narrative_authority_migrations_round_trip_have_exact_additive_tables():
    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / "narrative-authority.db"
        url = f"sqlite:///{database.as_posix()}"
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        config.attributes["database_url"] = url
        command.upgrade(config, "0008_document_concurrency")
        engine = create_engine(url)
        assert not (NARRATIVE_TABLES & set(inspect(engine).get_table_names()))
        engine.dispose()

        command.upgrade(config, "0009_narrative_authority_traits")
        engine = create_engine(url)
        inspector = inspect(engine)
        assert NARRATIVE_TABLES <= set(inspector.get_table_names())
        for table_name in NARRATIVE_TABLES:
            migrated_columns = {
                item["name"] for item in inspector.get_columns(table_name)
            }
            expected_columns = {
                column.name for column in Base.metadata.tables[table_name].columns
            }
            if table_name == "character_trait_candidates":
                expected_columns.remove("authority_tier")
            if table_name == "document_narrative_context_revisions":
                expected_columns -= {
                    "inference_reasoning",
                    "inference_evidence",
                    "inference_usage",
                    "inference_provider_config_id",
                    "inference_provider_identity",
                }
            assert migrated_columns == expected_columns
        engine.dispose()

        command.upgrade(config, "0010_character_trait_authority")
        engine = create_engine(url)
        inspector = inspect(engine)
        assert {
            item["name"]
            for item in inspector.get_columns("character_trait_candidates")
        } == {
            column.name
            for column in Base.metadata.tables["character_trait_candidates"].columns
        }
        engine.dispose()

        command.downgrade(config, "0008_document_concurrency")
        engine = create_engine(url)
        assert not (NARRATIVE_TABLES & set(inspect(engine).get_table_names()))
        engine.dispose()
