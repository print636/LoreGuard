from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.character_traits import upsert_character_trait_candidate
from app.db import (
    AnalysisDiagnosticRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    CharacterTraitAxisRow,
    CharacterTraitCandidateRow,
    CharacterTraitReviewRow,
    SessionLocal,
)
from app.main import app, settings, write_limiter
from app.narrative_context import payload_sha256


@pytest.fixture(autouse=True)
def _clear_write_limiter():
    write_limiter.events.clear()
    yield
    write_limiter.events.clear()


def _project_and_document(
    client: TestClient, *, headers: dict[str, str] | None = None
) -> tuple[dict, dict]:
    project_response = client.post(
        "/api/v1/projects",
        json={"name": f"角色接口-{uuid4().hex}"},
        headers=headers,
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
        headers=headers,
    )
    assert document_response.status_code == 201, document_response.text
    return project, document_response.json()


def _completed_run(
    client: TestClient,
    project_id: str,
    *,
    batch_mode: str = "baseline_build",
    headers: dict[str, str] | None = None,
) -> dict:
    with patch("app.main.dispatch_analysis"):
        response = client.post(
            f"/api/v1/projects/{project_id}/analysis-runs", headers=headers
        )
    assert response.status_code == 202, response.text
    with SessionLocal() as db:
        run = db.get(AnalysisRunRow, response.json()["id"])
        assert run is not None
        run.status = "completed"
        run.batch_mode = batch_mode
        db.commit()
    return response.json()


def _candidate(
    project_id: str,
    run_id: str,
    *,
    trait_key: str = "食物偏好:蜜瓜",
    trait_type: str = "preference",
    comparison_key: str | None = None,
    value: str = "喜欢蜜瓜",
    polarity: str = "positive",
    authority_tier: str = "formal_record",
    valid_from_release_ordinal: int | None = None,
    valid_until_release_ordinal: int | None = None,
    supersedes_candidate_id: str | None = None,
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
                "trait_type": trait_type,
                "trait_key": trait_key,
                "comparison_key": comparison_key,
                "value": value,
                "polarity": polarity,
                "stability": "stable",
                "contexts": ["日常饮食"],
                "origin": "explicit_setting",
                "authority_tier": authority_tier,
                "confidence": 0.95,
                "scope": {
                    "schema_version": 1,
                    "timeline_key": "main",
                    "release": {"key": "v1", "ordinal": 1},
                    "branch": {"path": ["main"]},
                },
                "valid_from_release_ordinal": valid_from_release_ordinal,
                "valid_until_release_ordinal": valid_until_release_ordinal,
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
                "supersedes_candidate_id": supersedes_candidate_id,
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


def _create_axis(
    client: TestClient,
    project_id: str,
    *,
    definition: str = "面对陌生人时，是否主动开启交谈",
    headers: dict[str, str] | None = None,
) -> dict:
    response = client.post(
        f"/api/v1/projects/{project_id}/character-trait-axes",
        json={
            "trait_type": "core_personality",
            "display_name": "社交主动性",
            "definition": definition,
        },
        headers=headers,
    )
    assert response.status_code == 201, response.text
    return response.json()


def _confirm(
    client: TestClient,
    project_id: str,
    candidate_id: str,
    *,
    headers: dict[str, str] | None = None,
):
    return client.post(
        _decision_path(project_id, candidate_id),
        json={"decision": "confirm", "expected_revision": 0},
        headers=headers,
    )


def _candidate_input_from_row(
    row: CharacterTraitCandidateRow,
    *,
    comparison_key: str | None,
    supersedes_candidate_id: str | None = None,
) -> dict:
    return {
        "character_key": row.character_key,
        "character_display_name": row.character_display_name,
        "trait_type": row.trait_type,
        "trait_key": row.trait_key,
        "comparison_key": comparison_key,
        "value": row.value,
        "polarity": row.polarity,
        "stability": row.stability,
        "contexts": row.contexts,
        "origin": row.origin,
        "authority_tier": row.authority_tier,
        "confidence": row.confidence,
        "scope": row.scope_payload,
        "valid_from_release_ordinal": row.valid_from_release_ordinal,
        "valid_until_release_ordinal": row.valid_until_release_ordinal,
        "evidence": row.evidence,
        "generator_version": row.generator_version,
        "provenance": row.provenance,
        "supersedes_candidate_id": supersedes_candidate_id,
    }


def _pre0014_fingerprint(row: CharacterTraitCandidateRow, comparison_key: str) -> str:
    semantic_evidence = [
        {
            key: item[key]
            for key in (
                "document_id", "document_version", "content_sha256",
                "line_start", "line_end", "text",
            )
        }
        for item in row.evidence
    ]
    return payload_sha256({
        "character_key": row.character_key,
        "trait_type": row.trait_type,
        "comparison_key": comparison_key,
        "polarity": row.polarity,
        "stability": row.stability,
        "contexts": row.contexts,
        "origin": row.origin,
        "authority_tier": row.authority_tier,
        "scope_sha256": row.scope_sha256,
        "valid_from_release_ordinal": row.valid_from_release_ordinal,
        "valid_until_release_ordinal": row.valid_until_release_ordinal,
        "semantic_evidence_sha256": payload_sha256(semantic_evidence),
        "supersedes_candidate_id": row.supersedes_candidate_id,
        "generator_version": row.generator_version,
    })


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


def test_character_roster_legacy_fallback_then_baseline_cutover_is_run_scoped():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        legacy = _completed_run(
            client, project["id"], batch_mode="full_review"
        )
        with SessionLocal() as db:
            db.add(
                AnalysisDiagnosticRow(
                    run_id=legacy["id"],
                    payload={
                        "character_consistency": {
                            "outcome": "completed",
                            "reason_code": "completed",
                        }
                    },
                )
            )
            db.commit()
        candidate_id = _candidate(project["id"], legacy["id"])
        confirmed = client.post(
            _decision_path(project["id"], candidate_id),
            json={"decision": "confirm", "expected_revision": 0},
        )
        assert confirmed.status_code == 201, confirmed.text

        legacy_roster = client.get(
            f"/api/v1/projects/{project['id']}/characters"
        )
        assert legacy_roster.status_code == 200, legacy_roster.text
        assert legacy_roster.json()["source_run_id"] == legacy["id"]
        assert legacy_roster.json()["readiness"] == "ready"
        assert legacy_roster.json()["items"][0]["confirmed_trait_count"] == 1

        first_draft = _completed_run(
            client, project["id"], batch_mode="draft_review"
        )
        with SessionLocal() as db:
            db.add(
                AnalysisDiagnosticRow(
                    run_id=first_draft["id"],
                    payload={
                        "character_consistency": {
                            "outcome": "skipped",
                            "reason_code": "no_eligible_frozen_documents",
                        }
                    },
                )
            )
            db.commit()

        after_first_draft = client.get(
            f"/api/v1/projects/{project['id']}/characters"
        )
        assert after_first_draft.json()["source_run_id"] == legacy["id"]
        assert after_first_draft.json()["readiness"] == "ready"

        baseline = _completed_run(client, project["id"])
        with SessionLocal() as db:
            db.add(
                AnalysisDiagnosticRow(
                    run_id=baseline["id"],
                    payload={
                        "character_consistency": {
                            "outcome": "partial",
                            "reason_code": "chunk_limit",
                        }
                    },
                )
            )
            db.commit()

        later_draft = _completed_run(
            client, project["id"], batch_mode="draft_review"
        )
        roster = client.get(f"/api/v1/projects/{project['id']}/characters")

    assert roster.status_code == 200, roster.text
    body = roster.json()
    assert body["source_run_id"] == baseline["id"]
    assert body["source_run_id"] != legacy["id"]
    assert body["source_run_id"] != later_draft["id"]
    assert body["model_coverage"] == "partial"
    assert body["readiness"] == "not_generated"
    assert body["items"] == []
    assert body["total"] == 0


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


def test_same_value_opposite_polarity_requires_supersession_in_overlapping_release():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        positive = _candidate(
            project["id"], run["id"], comparison_key="preference:蜜瓜",
            value="喜欢蜜瓜", polarity="positive",
        )
        negative = _candidate(
            project["id"], run["id"], comparison_key="preference:蜜瓜",
            value="喜欢蜜瓜", polarity="negative",
        )
        assert positive != negative
        assert _confirm(client, project["id"], positive).status_code == 201

        conflict = _confirm(client, project["id"], negative)
        assert conflict.status_code == 409, conflict.text
        assert conflict.json()["detail"]["code"] == "character_trait_confirmation_conflict"
        with SessionLocal() as db:
            assert db.get(CharacterTraitCandidateRow, positive).review_state == "confirmed"
            assert db.get(CharacterTraitCandidateRow, negative).review_state == "pending"


def test_same_value_opposite_polarity_can_coexist_in_disjoint_release_ranges():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        earlier = _candidate(
            project["id"], run["id"], comparison_key="preference:蜜瓜",
            value="喜欢蜜瓜", polarity="positive",
            valid_until_release_ordinal=1,
        )
        later = _candidate(
            project["id"], run["id"], comparison_key="preference:蜜瓜",
            value="喜欢蜜瓜", polarity="negative",
            valid_from_release_ordinal=2,
        )
        assert _confirm(client, project["id"], earlier).status_code == 201
        confirmed = _confirm(client, project["id"], later)
        assert confirmed.status_code == 201, confirmed.text


def test_distinct_object_preferences_with_generic_trait_key_can_both_be_confirmed():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        melon = _candidate(
            project["id"], run["id"],
            trait_key="food_preference",
            comparison_key="preference:蜜瓜",
            value="喜欢蜜瓜",
        )
        grape = _candidate(
            project["id"], run["id"],
            trait_key="food_preference",
            comparison_key="preference:葡萄",
            value="讨厌葡萄",
            polarity="negative",
        )

        assert _confirm(client, project["id"], melon).status_code == 201
        second = _confirm(client, project["id"], grape)
        assert second.status_code == 201, second.text
        with SessionLocal() as db:
            assert db.get(CharacterTraitCandidateRow, melon).review_state == "confirmed"
            assert db.get(CharacterTraitCandidateRow, grape).review_state == "confirmed"


def test_same_value_preferences_for_distinct_known_objects_can_both_be_confirmed():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        melon = _candidate(
            project["id"], run["id"],
            trait_key="food_preference", comparison_key="preference:蜜瓜",
            value="喜欢水果",
        )
        grape = _candidate(
            project["id"], run["id"],
            trait_key="food_preference", comparison_key="preference:葡萄",
            value="喜欢水果",
        )

        assert melon != grape
        assert _confirm(client, project["id"], melon).status_code == 201
        second = _confirm(client, project["id"], grape)
        assert second.status_code == 201, second.text


def test_same_object_key_conflicts_even_when_trait_labels_differ():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        first = _candidate(
            project["id"], run["id"],
            trait_key="melon_preference",
            comparison_key="preference:蜜瓜",
            value="喜欢蜜瓜",
        )
        second = _candidate(
            project["id"], run["id"],
            trait_key="fruit_choice",
            comparison_key="preference:蜜瓜",
            value="讨厌蜜瓜",
            polarity="negative",
        )

        assert _confirm(client, project["id"], first).status_code == 201
        conflict = _confirm(client, project["id"], second)
        assert conflict.status_code == 409, conflict.text
        assert conflict.json()["detail"]["code"] == "character_trait_confirmation_conflict"


def test_same_object_alias_can_explicitly_supersede_confirmed_candidate():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        first = _candidate(
            project["id"], run["id"],
            trait_key="melon_preference",
            comparison_key="preference:蜜瓜",
            value="喜欢蜜瓜",
        )
        assert _confirm(client, project["id"], first).status_code == 201
        replacement = _candidate(
            project["id"], run["id"],
            trait_key="fruit_choice",
            comparison_key="preference:蜜瓜",
            value="讨厌蜜瓜",
            polarity="negative",
            supersedes_candidate_id=first,
        )

        confirmed = _confirm(client, project["id"], replacement)
        assert confirmed.status_code == 201, confirmed.text
        with SessionLocal() as db:
            assert db.get(CharacterTraitCandidateRow, first).review_state == "superseded"
            assert db.get(CharacterTraitCandidateRow, replacement).review_state == "confirmed"


def test_same_object_different_value_axes_have_distinct_rows_and_can_coexist():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        trust = _candidate(
            project["id"], run["id"],
            trait_type="value",
            trait_key="companion_trust",
            comparison_key="value:同伴",
            value="信任同伴",
        )
        protect = _candidate(
            project["id"], run["id"],
            trait_type="value",
            trait_key="companion_protection",
            comparison_key="value:同伴",
            value="保护同伴",
        )

        assert trust != protect
        assert _confirm(client, project["id"], trust).status_code == 201
        second = _confirm(client, project["id"], protect)
        assert second.status_code == 201, second.text
        with SessionLocal() as db:
            first_row = db.get(CharacterTraitCandidateRow, trust)
            second_row = db.get(CharacterTraitCandidateRow, protect)
            assert first_row.candidate_fingerprint != second_row.candidate_fingerprint
            assert first_row.review_state == second_row.review_state == "confirmed"


@pytest.mark.parametrize("migrated_null_key", (False, True))
def test_old_object_only_fingerprint_reuses_same_axis_but_not_different_axis(
    migrated_null_key: bool,
):
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        trust_id = _candidate(
            project["id"], run["id"],
            trait_type="value", trait_key="companion_trust",
            comparison_key="value:同伴", value="信任同伴",
        )
        with SessionLocal() as db:
            original = db.get(CharacterTraitCandidateRow, trust_id)
            assert original is not None
            semantic_evidence = [
                {
                    "document_id": item["document_id"],
                    "document_version": item["document_version"],
                    "content_sha256": item["content_sha256"],
                    "line_start": item["line_start"],
                    "line_end": item["line_end"],
                    "text": item["text"],
                }
                for item in original.evidence
            ]
            original.candidate_fingerprint = payload_sha256({
                "character_key": original.character_key,
                "trait_type": original.trait_type,
                "comparison_key": original.comparison_key,
                "polarity": original.polarity,
                "stability": original.stability,
                "contexts": original.contexts,
                "origin": original.origin,
                "authority_tier": original.authority_tier,
                "scope_sha256": original.scope_sha256,
                "valid_from_release_ordinal": original.valid_from_release_ordinal,
                "valid_until_release_ordinal": original.valid_until_release_ordinal,
                "semantic_evidence_sha256": payload_sha256(semantic_evidence),
                "supersedes_candidate_id": original.supersedes_candidate_id,
                "generator_version": original.generator_version,
            })
            candidate = {
                "character_key": original.character_key,
                "character_display_name": original.character_display_name,
                "trait_type": original.trait_type,
                "trait_key": original.trait_key,
                "comparison_key": original.comparison_key,
                "value": original.value,
                "polarity": original.polarity,
                "stability": original.stability,
                "contexts": original.contexts,
                "origin": original.origin,
                "authority_tier": original.authority_tier,
                "confidence": original.confidence,
                "scope": original.scope_payload,
                "valid_from_release_ordinal": original.valid_from_release_ordinal,
                "valid_until_release_ordinal": original.valid_until_release_ordinal,
                "evidence": original.evidence,
                "generator_version": original.generator_version,
                "provenance": original.provenance,
            }
            if migrated_null_key:
                # 0014 adds a nullable key. A real pre-0014 row retains the
                # object-bearing old fingerprint but has no stored key.
                original.comparison_key = None
            db.commit()

            same, created = upsert_character_trait_candidate(
                db, project_id=project["id"], source_run_id=run["id"],
                candidate=candidate,
            )
            assert not created and same.id == trust_id
            different, created = upsert_character_trait_candidate(
                db, project_id=project["id"], source_run_id=run["id"],
                candidate={
                    **candidate,
                    "trait_key": "companion_protection",
                    "value": "保护同伴",
                },
            )
            assert created and different.id != trust_id
            assert different.candidate_fingerprint != original.candidate_fingerprint


def test_legacy_preference_link_cannot_lower_confirmed_authority():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        canon = _candidate(
            project["id"], run["id"], trait_key="food_preference",
            comparison_key=None, authority_tier="core_canon",
        )
        assert _confirm(client, project["id"], canon).status_code == 201
        formal = _candidate(
            project["id"], run["id"], trait_key="food_preference",
            comparison_key="preference:蜜瓜", authority_tier="formal_record",
        )

        link = client.post(
            f"/api/v1/projects/{project['id']}/characters/{quote('林澈', safe='')}"
            f"/profile-candidates/{formal}/supersession-links",
            json={"supersedes_candidate_id": canon, "expected_revision": 0},
        )
        assert link.status_code == 409, link.text
        assert link.json()["detail"]["code"] == "character_trait_supersession_conflict"
        with SessionLocal() as db:
            assert db.get(CharacterTraitCandidateRow, canon).review_state == "confirmed"
            assert db.get(CharacterTraitCandidateRow, formal).review_state == "pending"


def test_legacy_preference_confirmation_rechecks_authority_after_link():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        legacy = _candidate(
            project["id"], run["id"], trait_key="food_preference",
            comparison_key=None,
        )
        assert _confirm(client, project["id"], legacy).status_code == 201
        keyed = _candidate(
            project["id"], run["id"], trait_key="food_preference",
            comparison_key="preference:蜜瓜",
        )
        link = client.post(
            f"/api/v1/projects/{project['id']}/characters/{quote('林澈', safe='')}"
            f"/profile-candidates/{keyed}/supersession-links",
            json={"supersedes_candidate_id": legacy, "expected_revision": 0},
        )
        assert link.status_code == 201, link.text
        linked = link.json()["candidate"]["id"]

        # A previously linked pending row must not bypass the confirmation
        # guard if the confirmed baseline's authority changes in the meantime.
        with SessionLocal() as db:
            db.get(CharacterTraitCandidateRow, legacy).authority_tier = "core_canon"
            db.commit()
        blocked = _confirm(client, project["id"], linked)
        assert blocked.status_code == 409, blocked.text
        assert blocked.json()["detail"]["code"] == "character_trait_supersession_conflict"
        with SessionLocal() as db:
            assert db.get(CharacterTraitCandidateRow, legacy).review_state == "confirmed"
            assert db.get(CharacterTraitCandidateRow, linked).review_state == "pending"


def test_pre0014_preference_upgrade_needs_explicit_supersession():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        legacy_id = _candidate(
            project["id"], run["id"],
            trait_key="food_preference", comparison_key="preference:蜜瓜",
        )
        with SessionLocal() as db:
            legacy = db.get(CharacterTraitCandidateRow, legacy_id)
            assert legacy is not None
            candidate_input = _candidate_input_from_row(
                legacy, comparison_key="preference:蜜瓜"
            )
            old_fingerprint = _pre0014_fingerprint(legacy, "preference:蜜瓜")
            assert old_fingerprint != legacy.candidate_fingerprint
            # Simulate a confirmed pre-0014 row after migration: its old hash
            # contains the parsed object, while the new column remains NULL.
            legacy.candidate_fingerprint = old_fingerprint
            legacy.comparison_key = None
            db.commit()

        assert _confirm(client, project["id"], legacy_id).status_code == 201
        with SessionLocal() as db:
            legacy = db.get(CharacterTraitCandidateRow, legacy_id)
            legacy_before_upgrade = (
                legacy.candidate_fingerprint, legacy.comparison_key,
                legacy.review_state, legacy.lock_version, legacy.evidence_sha256,
            )
            keyed, created = upsert_character_trait_candidate(
                db, project_id=project["id"], source_run_id=run["id"],
                candidate=candidate_input,
            )
            assert created and keyed.id != legacy_id
            assert keyed.review_state == "pending"
            assert keyed.comparison_key == "preference:蜜瓜"
            assert keyed.candidate_fingerprint != old_fingerprint
            keyed_id = keyed.id
            db.commit()

        with SessionLocal() as db:
            repeated, created = upsert_character_trait_candidate(
                db, project_id=project["id"], source_run_id=run["id"],
                candidate=candidate_input,
            )
            assert not created and repeated.id == keyed_id
            legacy = db.get(CharacterTraitCandidateRow, legacy_id)
            assert (
                legacy.candidate_fingerprint, legacy.comparison_key,
                legacy.review_state, legacy.lock_version, legacy.evidence_sha256,
            ) == legacy_before_upgrade

        profile_path = (
            f"/api/v1/projects/{project['id']}/characters/{quote('林澈', safe='')}"
        )
        profile = client.get(profile_path)
        assert profile.status_code == 200, profile.text
        assert {item["id"] for item in profile.json()["confirmed_traits"]} == {legacy_id}
        assert profile.json()["pending_candidate_count"] == 1

        blocked = _confirm(client, project["id"], keyed_id)
        assert blocked.status_code == 409, blocked.text
        assert blocked.json()["detail"]["code"] == "character_trait_confirmation_conflict"
        with SessionLocal() as db:
            assert db.get(CharacterTraitCandidateRow, keyed_id).review_state == "pending"
            legacy = db.get(CharacterTraitCandidateRow, legacy_id)
            assert (
                legacy.candidate_fingerprint, legacy.comparison_key,
                legacy.review_state, legacy.lock_version, legacy.evidence_sha256,
            ) == legacy_before_upgrade

        link_path = (
            f"{profile_path}/profile-candidates/{keyed_id}/supersession-links"
        )
        link_request = {
            "supersedes_candidate_id": legacy_id,
            "expected_revision": 0,
        }
        wrong_target = client.post(
            link_path,
            json={**link_request, "supersedes_candidate_id": str(uuid4())},
        )
        assert wrong_target.status_code == 409, wrong_target.text
        linked_response = client.post(link_path, json=link_request)
        assert linked_response.status_code == 201, linked_response.text
        assert linked_response.json()["created"] is True
        linked_id = linked_response.json()["candidate"]["id"]
        assert linked_id not in {legacy_id, keyed_id}
        assert linked_response.json()["candidate"]["comparison_key"] == "preference:蜜瓜"
        assert linked_response.json()["candidate"]["supersedes_candidate_id"] == legacy_id
        repeated_link = client.post(link_path, json=link_request)
        assert repeated_link.status_code == 201, repeated_link.text
        assert repeated_link.json()["created"] is False
        assert repeated_link.json()["candidate"]["id"] == linked_id

        headers = {"Idempotency-Key": f"legacy-upgrade-{linked_id}"}
        confirmed = _confirm(client, project["id"], linked_id, headers=headers)
        assert confirmed.status_code == 201, confirmed.text
        retried = _confirm(client, project["id"], linked_id, headers=headers)
        assert retried.status_code == 201, retried.text
        assert retried.json()["deduplicated"] is True

        with SessionLocal() as db:
            legacy = db.get(CharacterTraitCandidateRow, legacy_id)
            linked = db.get(CharacterTraitCandidateRow, linked_id)
            assert legacy.review_state == "superseded"
            assert legacy.candidate_fingerprint == old_fingerprint
            assert legacy.comparison_key is None
            assert linked.review_state == "confirmed"
            assert linked.comparison_key == "preference:蜜瓜"

        repeated_link = client.post(link_path, json=link_request)
        assert repeated_link.status_code == 201, repeated_link.text
        assert repeated_link.json()["created"] is False
        assert repeated_link.json()["candidate"]["id"] == linked_id
        assert repeated_link.json()["candidate"]["review_state"] == "confirmed"

        profile = client.get(profile_path)
        assert profile.status_code == 200, profile.text
        assert {item["id"] for item in profile.json()["confirmed_traits"]} == {linked_id}


def test_keyed_preference_with_old_hash_is_reused_when_stored_key_matches():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        candidate_id = _candidate(
            project["id"], run["id"], comparison_key="preference:蜜瓜",
        )
        with SessionLocal() as db:
            row = db.get(CharacterTraitCandidateRow, candidate_id)
            candidate_input = _candidate_input_from_row(
                row, comparison_key="preference:蜜瓜"
            )
            old_fingerprint = _pre0014_fingerprint(row, "preference:蜜瓜")
            row.candidate_fingerprint = old_fingerprint
            db.commit()

        with SessionLocal() as db:
            existing, created = upsert_character_trait_candidate(
                db, project_id=project["id"], source_run_id=run["id"],
                candidate=candidate_input,
            )
            assert not created and existing.id == candidate_id
            assert existing.candidate_fingerprint == old_fingerprint


def test_same_object_same_value_axis_conflicts_and_wrong_axis_cannot_supersede():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        trust = _candidate(
            project["id"], run["id"],
            trait_type="value",
            trait_key="companion_trust",
            comparison_key="value:同伴",
            value="信任同伴",
        )
        assert _confirm(client, project["id"], trust).status_code == 201

        opposed = _candidate(
            project["id"], run["id"],
            trait_type="value",
            trait_key="companion_trust",
            comparison_key="value:同伴",
            value="不信任同伴",
            polarity="negative",
        )
        conflict = _confirm(client, project["id"], opposed)
        assert conflict.status_code == 409, conflict.text
        assert conflict.json()["detail"]["code"] == "character_trait_confirmation_conflict"

        with SessionLocal() as db:
            snapshot = db.scalar(
                select(AnalysisRunInputRow).where(AnalysisRunInputRow.run_id == run["id"])
            )
            assert snapshot is not None
            candidate = {
                "character_key": "林澈",
                "character_display_name": "林澈",
                "trait_type": "value",
                "trait_key": "companion_protection",
                "comparison_key": "value:同伴",
                "value": "保护同伴",
                "polarity": "positive",
                "stability": "stable",
                "origin": "explicit_setting",
                "authority_tier": "formal_record",
                "confidence": 0.95,
                "scope": {
                    "schema_version": 1,
                    "timeline_key": "main",
                    "release": {"key": "v1", "ordinal": 1},
                    "branch": {"path": ["main"]},
                },
                "evidence": [{
                    "input_id": snapshot.id,
                    "document_id": snapshot.document_id,
                    "document_name": snapshot.document_name,
                    "document_version": snapshot.document_version,
                    "content_sha256": snapshot.content_sha256,
                    "line_start": 1,
                    "line_end": 1,
                    "text": snapshot.content,
                }],
                "generator_version": "api-contract-test-v1",
                "supersedes_candidate_id": trust,
            }
            with pytest.raises(ValueError, match="superseded candidate is incompatible"):
                upsert_character_trait_candidate(
                    db,
                    project_id=project["id"],
                    source_run_id=run["id"],
                    candidate=candidate,
                )


@pytest.mark.parametrize("legacy_position", ["confirmed", "pending"])
def test_missing_legacy_object_key_preserves_conservative_confirmation_conflict(
    legacy_position: str,
):
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        first = _candidate(
            project["id"], run["id"],
            trait_key="food_preference",
            comparison_key=(
                None if legacy_position == "confirmed" else "preference:蜜瓜"
            ),
            value="喜欢蜜瓜",
        )
        second = _candidate(
            project["id"], run["id"],
            trait_key="food_preference",
            comparison_key=(
                None if legacy_position == "pending" else "preference:葡萄"
            ),
            value="讨厌葡萄",
            polarity="negative",
        )

        assert _confirm(client, project["id"], first).status_code == 201
        conflict = _confirm(client, project["id"], second)
        assert conflict.status_code == 409, conflict.text
        assert conflict.json()["detail"]["code"] == "character_trait_confirmation_conflict"
        with SessionLocal() as db:
            assert db.get(CharacterTraitCandidateRow, second).review_state == "pending"


def test_invalid_stored_object_key_uses_legacy_label_conflict_rule():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        first = _candidate(
            project["id"], run["id"],
            trait_key="food_preference",
            comparison_key="preference:蜜瓜",
            value="喜欢蜜瓜",
        )
        second = _candidate(
            project["id"], run["id"],
            trait_key="food_preference",
            comparison_key="preference:葡萄",
            value="讨厌葡萄",
            polarity="negative",
        )
        with SessionLocal() as db:
            db.get(CharacterTraitCandidateRow, second).comparison_key = "preference:bad key"
            db.commit()

        assert _confirm(client, project["id"], first).status_code == 201
        conflict = _confirm(client, project["id"], second)
        assert conflict.status_code == 409, conflict.text
        assert conflict.json()["detail"]["code"] == "character_trait_confirmation_conflict"


def test_non_object_dimension_keeps_trait_label_conflict_rule():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        first = _candidate(
            project["id"], run["id"],
            trait_type="core_personality",
            trait_key="social_initiative",
            comparison_key="core_personality:外向性",
            value="主动交谈",
        )
        second = _candidate(
            project["id"], run["id"],
            trait_type="core_personality",
            trait_key="social_initiative",
            comparison_key="core_personality:社交主动性",
            value="避免交谈",
            polarity="negative",
        )

        assert _confirm(client, project["id"], first).status_code == 201
        conflict = _confirm(client, project["id"], second)
        assert conflict.status_code == 409, conflict.text
        assert conflict.json()["detail"]["code"] == "character_trait_confirmation_conflict"


def test_object_confirmation_still_requires_author_workspace_and_csrf():
    with patch.multiple(
        settings,
        auth_mode="required",
        auth_secret_key="api-contract-test-secret-key-32-bytes",
        auth_cookie_secure=False,
        auth_cookie_samesite="lax",
    ), TestClient(app) as owner, TestClient(app) as other:
        owner_registration = owner.post(
            "/api/v1/auth/register",
            json={
                "email": f"owner-{uuid4().hex}@example.com",
                "password": "correct horse battery staple",
                "display_name": "Owner",
            },
        )
        other_registration = other.post(
            "/api/v1/auth/register",
            json={
                "email": f"other-{uuid4().hex}@example.com",
                "password": "correct horse battery staple",
                "display_name": "Other",
            },
        )
        assert owner_registration.status_code == 201, owner_registration.text
        assert other_registration.status_code == 201, other_registration.text
        owner_headers = {"X-CSRF-Token": owner_registration.headers["X-CSRF-Token"]}
        other_headers = {"X-CSRF-Token": other_registration.headers["X-CSRF-Token"]}
        project, _ = _project_and_document(owner, headers=owner_headers)
        run = _completed_run(owner, project["id"], headers=owner_headers)
        candidate_id = _candidate(
            project["id"], run["id"],
            trait_key="food_preference",
            comparison_key="preference:蜜瓜",
        )

        forbidden = _confirm(other, project["id"], candidate_id, headers=other_headers)
        assert forbidden.status_code == 404, forbidden.text
        csrf_missing = _confirm(owner, project["id"], candidate_id)
        assert csrf_missing.status_code == 403, csrf_missing.text
        with SessionLocal() as db:
            assert db.get(CharacterTraitCandidateRow, candidate_id).review_state == "pending"
        confirmed = _confirm(owner, project["id"], candidate_id, headers=owner_headers)
        assert confirmed.status_code == 201, confirmed.text


def test_legacy_preference_link_requires_author_workspace_and_csrf():
    with patch.multiple(
        settings,
        auth_mode="required",
        auth_secret_key="api-contract-test-secret-key-32-bytes",
        auth_cookie_secure=False,
        auth_cookie_samesite="lax",
    ), TestClient(app) as owner, TestClient(app) as other:
        owner_registration = owner.post(
            "/api/v1/auth/register",
            json={
                "email": f"owner-{uuid4().hex}@example.com",
                "password": "correct horse battery staple",
                "display_name": "Owner",
            },
        )
        other_registration = other.post(
            "/api/v1/auth/register",
            json={
                "email": f"other-{uuid4().hex}@example.com",
                "password": "correct horse battery staple",
                "display_name": "Other",
            },
        )
        assert owner_registration.status_code == 201, owner_registration.text
        assert other_registration.status_code == 201, other_registration.text
        owner_headers = {"X-CSRF-Token": owner_registration.headers["X-CSRF-Token"]}
        other_headers = {"X-CSRF-Token": other_registration.headers["X-CSRF-Token"]}
        project, _ = _project_and_document(owner, headers=owner_headers)
        run = _completed_run(owner, project["id"], headers=owner_headers)
        legacy_id = _candidate(
            project["id"], run["id"],
            trait_key="food_preference", comparison_key=None,
        )
        assert _confirm(owner, project["id"], legacy_id, headers=owner_headers).status_code == 201
        keyed_id = _candidate(
            project["id"], run["id"],
            trait_key="food_preference", comparison_key="preference:蜜瓜",
        )
        link_path = (
            f"/api/v1/projects/{project['id']}/characters/{quote('林澈', safe='')}"
            f"/profile-candidates/{keyed_id}/supersession-links"
        )
        link_request = {"supersedes_candidate_id": legacy_id, "expected_revision": 0}
        forbidden = other.post(link_path, json=link_request, headers=other_headers)
        assert forbidden.status_code == 404, forbidden.text
        csrf_missing = owner.post(link_path, json=link_request)
        assert csrf_missing.status_code == 403, csrf_missing.text
        with SessionLocal() as db:
            assert db.get(CharacterTraitCandidateRow, legacy_id).review_state == "confirmed"
            assert db.get(CharacterTraitCandidateRow, keyed_id).review_state == "pending"
        linked = owner.post(link_path, json=link_request, headers=owner_headers)
        assert linked.status_code == 201, linked.text
        assert linked.json()["candidate"]["supersedes_candidate_id"] == legacy_id


def test_object_confirmation_retried_with_stale_revision_is_rejected():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        candidate_id = _candidate(
            project["id"], run["id"],
            trait_key="food_preference",
            comparison_key="preference:蜜瓜",
        )

        first = _confirm(client, project["id"], candidate_id)
        stale = _confirm(client, project["id"], candidate_id)
        assert first.status_code == 201, first.text
        assert stale.status_code == 409, stale.text
        assert stale.json()["detail"]["code"] == "character_trait_revision_conflict"


def test_concurrent_confirmation_of_one_object_candidate_commits_once():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        candidate_id = _candidate(
            project["id"], run["id"],
            trait_key="food_preference",
            comparison_key="preference:蜜瓜",
        )
        ready = Barrier(2)

        def decide():
            ready.wait(timeout=5)
            return _confirm(client, project["id"], candidate_id)

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: decide(), range(2)))

        assert sorted(response.status_code for response in responses) == [201, 409]
        assert next(
            response.json()["detail"]["code"]
            for response in responses
            if response.status_code == 409
        ) == "character_trait_revision_conflict"
        with SessionLocal() as db:
            assert db.get(CharacterTraitCandidateRow, candidate_id).review_state == "confirmed"


def test_author_axis_creation_is_project_scoped_immutable_and_exact_duplicates_conflict():
    with TestClient(app) as client:
        first, _ = _project_and_document(client)
        second, _ = _project_and_document(client)
        axis = _create_axis(client, first["id"], definition="  是否主动\n向陌生人搭话  ")
        assert axis["version"] == 1
        assert axis["definition"] == "是否主动 向陌生人搭话"
        assert len(axis["definition_sha256"]) == 64
        listed = client.get(f"/api/v1/projects/{first['id']}/character-trait-axes")
        assert listed.status_code == 200, listed.text
        assert listed.json()["items"] == [axis]
        isolated = client.get(f"/api/v1/projects/{second['id']}/character-trait-axes")
        assert isolated.status_code == 200
        assert isolated.json()["items"] == []
        duplicate = client.post(
            f"/api/v1/projects/{first['id']}/character-trait-axes",
            json={
                "trait_type": "core_personality",
                "display_name": "另一个显示名称",
                "definition": "是否主动 向陌生人搭话",
            },
        )
        assert duplicate.status_code == 409, duplicate.text
        assert duplicate.json()["detail"]["code"] == "character_trait_axis_duplicate"
        unsafe = client.post(
            f"/api/v1/projects/{first['id']}/character-trait-axes",
            json={
                "trait_type": "core_personality",
                "display_name": "社交主动性",
                "definition": "https://example.invalid/secret",
            },
        )
        assert unsafe.status_code == 422, unsafe.text
        assert unsafe.json()["detail"]["code"] == "character_trait_axis_text_unsafe"
        with SessionLocal() as db:
            stored = db.get(CharacterTraitAxisRow, axis["id"])
            assert stored.definition == axis["definition"]


def test_author_axis_confirmation_binds_atomically_and_same_axis_blocks_different_raw_labels():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        axis = _create_axis(client, project["id"])
        first = _candidate(
            project["id"], run["id"], trait_type="core_personality",
            trait_key="social_initiative", comparison_key=None,
            value="主动与陌生人交谈", polarity="positive",
        )
        second = _candidate(
            project["id"], run["id"], trait_type="core_personality",
            trait_key="starts_conversations", comparison_key=None,
            value="回避与陌生人交谈", polarity="negative",
        )
        with SessionLocal() as db:
            original = db.get(CharacterTraitCandidateRow, first)
            original_fields = (
                original.trait_key, original.candidate_fingerprint,
                original.evidence_sha256, original.evidence,
            )
        request = {
            "decision": "confirm", "expected_revision": 0,
            "approved_axis_id": axis["id"], "expected_axis_version": 1,
        }
        headers = {"Idempotency-Key": f"approved-axis-{uuid4().hex}"}
        accepted = client.post(
            _decision_path(project["id"], first), json=request, headers=headers,
        )
        assert accepted.status_code == 201, accepted.text
        assert accepted.json()["candidate"]["approved_axis_id"] == axis["id"]
        assert accepted.json()["candidate"]["approved_axis_version"] == 1
        replay = client.post(
            _decision_path(project["id"], first), json=request, headers=headers,
        )
        assert replay.status_code == 201 and replay.json()["deduplicated"] is True
        changed_payload = client.post(
            _decision_path(project["id"], first),
            json={**request, "approved_axis_id": None, "expected_axis_version": None},
            headers=headers,
        )
        assert changed_payload.status_code == 409, changed_payload.text
        assert changed_payload.json()["detail"]["code"] == "idempotency_key_conflict"
        conflict = client.post(_decision_path(project["id"], second), json=request)
        assert conflict.status_code == 409, conflict.text
        assert conflict.json()["detail"]["code"] == "character_trait_confirmation_conflict"
        with SessionLocal() as db:
            current = db.get(CharacterTraitCandidateRow, first)
            assert (
                current.trait_key, current.candidate_fingerprint,
                current.evidence_sha256, current.evidence,
            ) == original_fields
            assert current.approved_axis_id == axis["id"]
            assert current.approved_axis_version == 1
            pending = db.get(CharacterTraitCandidateRow, second)
            assert pending.review_state == "pending"
            assert pending.approved_axis_id is None
            review = db.get(CharacterTraitReviewRow, accepted.json()["decision_id"])
            assert review.approved_axis_id == axis["id"]
            assert review.approved_axis_version == 1


def test_author_axis_rejects_cross_project_wrong_type_version_and_stale_source():
    with TestClient(app) as client:
        first, document = _project_and_document(client)
        second, _ = _project_and_document(client)
        run = _completed_run(client, first["id"])
        other_axis = _create_axis(client, second["id"])
        candidate_id = _candidate(
            first["id"], run["id"], trait_type="core_personality",
            trait_key="social_initiative", comparison_key=None,
        )
        request = {
            "decision": "confirm", "expected_revision": 0,
            "approved_axis_id": other_axis["id"], "expected_axis_version": 1,
        }
        foreign = client.post(_decision_path(first["id"], candidate_id), json=request)
        assert foreign.status_code == 404, foreign.text
        assert foreign.json()["detail"]["code"] == "character_trait_axis_not_found"
        with SessionLocal() as db:
            with pytest.raises(IntegrityError):
                db.execute(
                    update(CharacterTraitCandidateRow)
                    .where(CharacterTraitCandidateRow.id == candidate_id)
                    .values(
                        approved_axis_id=other_axis["id"],
                        approved_axis_version=1,
                    )
                )
                db.commit()
            db.rollback()
        local_axis = _create_axis(client, first["id"])
        wrong_version = client.post(
            _decision_path(first["id"], candidate_id),
            json={**request, "approved_axis_id": local_axis["id"], "expected_axis_version": 2},
        )
        assert wrong_version.status_code == 409, wrong_version.text
        assert wrong_version.json()["detail"]["code"] == "character_trait_axis_version_conflict"
        preference = _candidate(
            first["id"], run["id"], trait_type="preference",
            trait_key="food_preference", comparison_key="preference:蜜瓜",
        )
        wrong_type = client.post(
            _decision_path(first["id"], preference),
            json={**request, "approved_axis_id": local_axis["id"]},
        )
        assert wrong_type.status_code == 422, wrong_type.text
        assert wrong_type.json()["detail"]["code"] == "character_trait_axis_dimension_mismatch"
        rejected_with_axis = client.post(
            _decision_path(first["id"], candidate_id),
            json={**request, "decision": "reject", "approved_axis_id": local_axis["id"]},
        )
        assert rejected_with_axis.status_code == 422
        replaced = client.post(
            f"/api/v1/projects/{first['id']}/documents/text",
            json={
                "name": "character.md", "content": "林澈现在更常与陌生人交谈。",
                "replace_document_id": document["id"],
            },
        )
        assert replaced.status_code == 201, replaced.text
        stale = client.post(
            _decision_path(first["id"], candidate_id),
            json={**request, "approved_axis_id": local_axis["id"]},
        )
        assert stale.status_code == 409, stale.text
        assert stale.json()["detail"]["code"] == "character_trait_candidate_stale"
        with SessionLocal() as db:
            row = db.get(CharacterTraitCandidateRow, candidate_id)
            assert row.review_state == "pending" and row.approved_axis_id is None


def test_author_axis_workspace_and_csrf_are_required_in_account_mode():
    with patch.multiple(
        settings,
        auth_mode="required",
        auth_secret_key="axis-api-test-secret-key-32-bytes",
        auth_cookie_secure=False,
        auth_cookie_samesite="lax",
    ), TestClient(app) as owner, TestClient(app) as outsider:
        registered_owner = owner.post(
            "/api/v1/auth/register",
            json={
                "email": f"axis-owner-{uuid4().hex}@example.com",
                "password": "correct horse battery staple",
                "display_name": "Owner",
            },
        )
        registered_other = outsider.post(
            "/api/v1/auth/register",
            json={
                "email": f"axis-other-{uuid4().hex}@example.com",
                "password": "correct horse battery staple",
                "display_name": "Other",
            },
        )
        assert registered_owner.status_code == 201, registered_owner.text
        assert registered_other.status_code == 201, registered_other.text
        owner_headers = {"X-CSRF-Token": registered_owner.headers["X-CSRF-Token"]}
        other_headers = {"X-CSRF-Token": registered_other.headers["X-CSRF-Token"]}
        project, _ = _project_and_document(owner, headers=owner_headers)
        axis_url = f"/api/v1/projects/{project['id']}/character-trait-axes"
        body = {
            "trait_type": "core_personality", "display_name": "社交主动性",
            "definition": "面对陌生人时是否主动交谈",
        }
        without_csrf = owner.post(axis_url, json=body)
        assert without_csrf.status_code == 403, without_csrf.text
        assert outsider.get(axis_url).status_code == 404
        assert outsider.post(axis_url, json=body, headers=other_headers).status_code == 404
        axis = _create_axis(owner, project["id"], headers=owner_headers)
        run = _completed_run(owner, project["id"], headers=owner_headers)
        candidate_id = _candidate(
            project["id"], run["id"], trait_type="core_personality",
            trait_key="social_initiative", comparison_key=None,
        )
        decision = {
            "decision": "confirm", "expected_revision": 0,
            "approved_axis_id": axis["id"], "expected_axis_version": 1,
        }
        no_csrf_decision = owner.post(
            _decision_path(project["id"], candidate_id), json=decision,
        )
        assert no_csrf_decision.status_code == 403
        foreign_decision = outsider.post(
            _decision_path(project["id"], candidate_id), json=decision,
            headers=other_headers,
        )
        assert foreign_decision.status_code == 404
        own_decision = owner.post(
            _decision_path(project["id"], candidate_id), json=decision,
            headers=owner_headers,
        )
        assert own_decision.status_code == 201, own_decision.text


def test_concurrent_author_axis_confirmation_commits_one_binding():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        axis = _create_axis(client, project["id"])
        candidate_id = _candidate(
            project["id"], run["id"], trait_type="core_personality",
            trait_key="social_initiative", comparison_key=None,
        )
        body = {
            "decision": "confirm", "expected_revision": 0,
            "approved_axis_id": axis["id"], "expected_axis_version": 1,
        }
        ready = Barrier(2)

        def decide():
            ready.wait(timeout=5)
            return client.post(_decision_path(project["id"], candidate_id), json=body)

        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(pool.map(lambda _: decide(), range(2)))
        assert sorted(response.status_code for response in responses) == [201, 409]
        with SessionLocal() as db:
            row = db.get(CharacterTraitCandidateRow, candidate_id)
            assert row.review_state == "confirmed"
            assert row.approved_axis_id == axis["id"]
            reviews = list(
                db.scalars(
                    select(CharacterTraitReviewRow).where(
                        CharacterTraitReviewRow.candidate_id == candidate_id,
                        CharacterTraitReviewRow.decision == "confirm",
                    )
                )
            )
            assert len(reviews) == 1 and reviews[0].approved_axis_id == axis["id"]


def test_author_axis_supersession_requires_same_explicit_axis_and_preserves_old_binding():
    with TestClient(app) as client:
        project, _ = _project_and_document(client)
        run = _completed_run(client, project["id"])
        first_axis = _create_axis(client, project["id"])
        second_axis = _create_axis(
            client, project["id"], definition="面对熟人时是否主动开启交谈",
        )
        original_id = _candidate(
            project["id"], run["id"], trait_type="core_personality",
            trait_key="social_initiative", comparison_key=None,
            value="主动交谈", polarity="positive",
        )
        original_decision = {
            "decision": "confirm", "expected_revision": 0,
            "approved_axis_id": first_axis["id"], "expected_axis_version": 1,
        }
        assert client.post(
            _decision_path(project["id"], original_id), json=original_decision,
        ).status_code == 201
        replacement_id = _candidate(
            project["id"], run["id"], trait_type="core_personality",
            trait_key="social_initiative", comparison_key=None,
            value="不主动交谈", polarity="negative",
            supersedes_candidate_id=original_id,
        )
        wrong_axis = client.post(
            _decision_path(project["id"], replacement_id),
            json={**original_decision, "approved_axis_id": second_axis["id"]},
        )
        assert wrong_axis.status_code == 409, wrong_axis.text
        assert wrong_axis.json()["detail"]["code"] == (
            "character_trait_axis_supersession_mismatch"
        )
        with SessionLocal() as db:
            original = db.get(CharacterTraitCandidateRow, original_id)
            replacement = db.get(CharacterTraitCandidateRow, replacement_id)
            assert original.review_state == "confirmed"
            assert original.approved_axis_id == first_axis["id"]
            assert replacement.review_state == "pending"
            assert replacement.approved_axis_id is None
        accepted = client.post(
            _decision_path(project["id"], replacement_id), json=original_decision,
        )
        assert accepted.status_code == 201, accepted.text
        with SessionLocal() as db:
            original = db.get(CharacterTraitCandidateRow, original_id)
            replacement = db.get(CharacterTraitCandidateRow, replacement_id)
            assert original.review_state == "superseded"
            assert original.approved_axis_id == first_axis["id"]
            assert replacement.review_state == "confirmed"
            assert replacement.approved_axis_id == first_axis["id"]
