"""Frozen narrative-context identity must govern candidate deduplication."""

from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.character_traits import (
    _formal_target_fingerprint,
    _frozen_context_identities,
    _semantic_evidence_digest,
    upsert_character_trait_candidate,
)
from app.db import (
    AnalysisRunCharacterTraitInputRow,
    AnalysisRunInputNarrativeContextRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    CharacterTraitCandidateRow,
    CharacterTraitReviewRow,
    SessionLocal,
)
from app.main import app, write_limiter
from app.narrative_context import payload_sha256


@pytest.fixture(autouse=True)
def _clear_write_limiter():
    write_limiter.events.clear()
    yield
    write_limiter.events.clear()


def _project_and_document(client, *, role="character_profile"):
    created = client.post(
        "/api/v1/projects", json={"name": f"context-fingerprint-{uuid4().hex}"}
    )
    assert created.status_code == 201, created.text
    project_id = created.json()["id"]
    document = client.post(
        f"/api/v1/projects/{project_id}/documents/text",
        json={
            "name": "source.md",
            "content": (
                "林澈喜欢蜜瓜。\n林澈始终喜欢蜜瓜。"
                if role == "chapter" else "林澈喜欢蜜瓜。林澈喜欢葡萄。"
            ),
            "document_role": role,
            "narrative_context": {
                "resolution_state": "confirmed",
                "publication_status": "published",
            },
        },
    )
    assert document.status_code == 201, document.text
    return project_id, document.json()["id"]


def _run(client, project_id):
    with patch("app.main.dispatch_analysis"):
        response = client.post(f"/api/v1/projects/{project_id}/analysis-runs")
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]
    with SessionLocal() as db:
        row = db.get(AnalysisRunRow, run_id)
        row.status = "completed"
        row.batch_mode = "baseline_build"
        db.commit()
    return run_id


def _payload(db, run_id, *, role="character_profile", support_id=None,
             trait_type="preference", comparison_key="preference:蜜瓜",
             trait_key="食物偏好:蜜瓜"):
    source = db.scalar(select(AnalysisRunInputRow).where(
        AnalysisRunInputRow.run_id == run_id
    ))
    assert source is not None
    evidence = [{
        "input_id": source.id,
        "document_id": source.document_id,
        "document_name": source.document_name,
        "document_version": source.document_version,
        "content_sha256": source.content_sha256,
        "line_start": line_number,
        "line_end": line_number,
        "text": source.content.splitlines()[line_number - 1],
    } for line_number in (1, 2) if role == "chapter" or line_number == 1]
    return {
        "character_key": "林澈",
        "character_display_name": "林澈",
        "trait_type": trait_type,
        "trait_key": trait_key,
        "comparison_key": comparison_key,
        "value": "喜欢蜜瓜",
        "polarity": "positive",
        "stability": "stable",
        "contexts": ["日常"],
        "origin": "history_inference" if role == "chapter" else "explicit_setting",
        "authority_tier": "formal_record",
        "confidence": 0.9,
        "scope": {"schema_version": 1, "timeline_key": "main"},
        "evidence": evidence,
        **({"support_refs": [{
            "evidence_index": 0, "support_id": support_id,
            "scope_relation": "local",
        }]} if support_id is not None else {}),
        "generator_version": "context-fingerprint-test-v1",
        "provenance": {"extractor": "test"},
    }


def _upsert(db, project_id, run_id, **kwargs):
    return upsert_character_trait_candidate(
        db, project_id=project_id, source_run_id=run_id,
        candidate=_payload(db, run_id, **kwargs),
    )


def _context_revision(client, project_id, document_id, *, expected, status):
    response = client.post(
        f"/api/v1/projects/{project_id}/documents/{document_id}"
        "/narrative-context/revisions",
        json={
            "expected_revision": expected,
            "resolution_state": "confirmed",
            "publication_status": status,
        },
    )
    assert response.status_code == 201, response.text


def _legacy_fingerprint(row, *, formal=False):
    base = {
        "character_key": row.character_key,
        "trait_type": row.trait_type,
        "comparison_key": row.comparison_key or row.trait_key,
        "polarity": row.polarity,
        "stability": row.stability,
        "contexts": row.contexts,
        "origin": row.origin,
        "authority_tier": row.authority_tier,
        "scope_sha256": row.scope_sha256,
        "valid_from_release_ordinal": row.valid_from_release_ordinal,
        "valid_until_release_ordinal": row.valid_until_release_ordinal,
        "semantic_evidence_sha256": _semantic_evidence_digest(row.evidence),
        "supersedes_candidate_id": row.supersedes_candidate_id,
        "generator_version": row.generator_version,
    }
    if formal:
        return _formal_target_fingerprint(
            base, project_id=row.project_id,
            evidence=row.evidence, support_bindings=row.support_bindings_v1,
        )
    if row.trait_type == "preference" and row.comparison_key is not None:
        base["fingerprint_version"] = "keyed_preference_v2"
    elif row.trait_type in {"value", "behavior_boundary", "current_state"}:
        from app.character_trait_extraction import stable_trait_identity
        base["relation_axis"] = stable_trait_identity(row.trait_type, row.trait_key)
    return payload_sha256(base)


@pytest.mark.parametrize("variant", ["formal", "history", "keyed_relation"])
def test_same_context_reuses_but_roundtrip_revision_makes_new_reviewable_row(variant):
    role = "chapter" if variant == "history" else "character_profile"
    options = (
        {"role": role, "support_id": "L1:A1"} if variant == "formal"
        else {"role": role, "trait_type": "behavior_boundary",
              "comparison_key": "behavior_boundary:食物", "trait_key": "行为边界:食物"}
        if variant == "keyed_relation" else {"role": role}
    )
    with TestClient(app) as client:
        project_id, document_id = _project_and_document(client, role=role)
        first_run = _run(client, project_id)
        with SessionLocal() as db:
            first, created = _upsert(db, project_id, first_run, **options)
            assert created
            first_id, first_fingerprint = first.id, first.candidate_fingerprint
            db.commit()
            same, created = _upsert(db, project_id, first_run, **options)
            assert not created and same.id == first_id
        second_run = _run(client, project_id)
        with SessionLocal() as db:
            same, created = _upsert(db, project_id, second_run, **options)
            assert not created and same.id == first_id

        _context_revision(client, project_id, document_id, expected=1, status="retired")
        _context_revision(client, project_id, document_id, expected=2, status="published")
        third_run = _run(client, project_id)
        with SessionLocal() as db:
            fresh, created = _upsert(db, project_id, third_run, **options)
            assert created and fresh.id != first_id
            assert fresh.candidate_fingerprint != first_fingerprint
            fresh_id = fresh.id
            db.commit()
        path = (
            f"/api/v1/projects/{project_id}/characters/{quote('林澈', safe='')}"
            "/profile-candidates/"
        )
        stale = client.get(path + first_id)
        assert stale.status_code == 200, stale.text
        assert stale.json()["status"] == "stale"
        assert stale.json()["reviewable"] is False
        current = client.get(path + fresh_id)
        assert current.status_code == 200, current.text
        assert current.json()["reviewable"] is True
        if variant == "formal":
            assert current.json()["support_bindings_status"] == "verified"
        confirmed = client.post(
            path + fresh_id + "/decisions",
            json={"decision": "confirm", "expected_revision": 0},
        )
        assert confirmed.status_code == 201, confirmed.text


@pytest.mark.parametrize("variant", ["formal", "history", "keyed_relation"])
def test_old_fingerprint_is_readable_and_reused_only_with_same_frozen_context(variant):
    role = "chapter" if variant == "history" else "character_profile"
    options = (
        {"role": role, "support_id": "L1:A1"} if variant == "formal"
        else {"role": role, "trait_type": "behavior_boundary",
              "comparison_key": "behavior_boundary:食物", "trait_key": "行为边界:食物"}
        if variant == "keyed_relation" else {"role": role}
    )
    with TestClient(app) as client:
        project_id, document_id = _project_and_document(client, role=role)
        first_run = _run(client, project_id)
        with SessionLocal() as db:
            first, _ = _upsert(db, project_id, first_run, **options)
            first.candidate_fingerprint = _legacy_fingerprint(
                first, formal=variant == "formal"
            )
            first_id = first.id
            db.commit()
        if variant == "formal":
            detail = client.get(
                f"/api/v1/projects/{project_id}/characters/"
                f"{quote('林澈', safe='')}/profile-candidates/{first_id}"
            )
            assert detail.status_code == 200, detail.text
            assert detail.json()["support_bindings_status"] == "verified"
            assert detail.json()["reviewable"] is True
        second_run = _run(client, project_id)
        with SessionLocal() as db:
            same, created = _upsert(db, project_id, second_run, **options)
            assert not created and same.id == first_id
        _context_revision(client, project_id, document_id, expected=1, status="retired")
        _context_revision(client, project_id, document_id, expected=2, status="published")
        third_run = _run(client, project_id)
        with SessionLocal() as db:
            fresh, created = _upsert(db, project_id, third_run, **options)
            assert created and fresh.id != first_id
            db.rollback()


def test_frozen_context_damage_never_falls_back_to_current_document():
    with TestClient(app) as client:
        project_id, _ = _project_and_document(client)
        run_id = _run(client, project_id)
        with SessionLocal() as db:
            source = db.scalar(select(AnalysisRunInputRow).where(
                AnalysisRunInputRow.run_id == run_id
            ))
            frozen = db.get(AnalysisRunInputNarrativeContextRow, source.id)
            assert frozen is not None
            frozen.context_revision_id = None
            db.commit()
            with pytest.raises(ValueError, match="frozen narrative context"):
                _upsert(db, project_id, run_id, support_id="L1:A1")
            assert db.scalars(select(CharacterTraitCandidateRow).where(
                CharacterTraitCandidateRow.project_id == project_id
            )).all() == []


def test_multiple_formal_targets_remain_distinct_with_same_frozen_context():
    with TestClient(app) as client:
        project_id, _ = _project_and_document(client)
        run_id = _run(client, project_id)
        with SessionLocal() as db:
            first, created = _upsert(db, project_id, run_id, support_id="L1:A1")
            assert created
            second, created = _upsert(db, project_id, run_id, support_id="L1:A2")
            assert created and second.id != first.id
            assert second.candidate_fingerprint != first.candidate_fingerprint
            context_ids = _frozen_context_identities(db, run_id, first.evidence)
            assert len(context_ids) == 1


def test_context_identity_orders_multiple_evidence_documents_by_document_id():
    with TestClient(app) as client:
        project_id, _ = _project_and_document(client)
        second = client.post(
            f"/api/v1/projects/{project_id}/documents/text",
            json={
                "name": "second.md", "content": "林澈从不讨厌蜜瓜。",
                "document_role": "character_profile",
                "narrative_context": {
                    "resolution_state": "confirmed",
                    "publication_status": "published",
                },
            },
        )
        assert second.status_code == 201, second.text
        run_id = _run(client, project_id)
        with SessionLocal() as db:
            inputs = db.scalars(select(AnalysisRunInputRow).where(
                AnalysisRunInputRow.run_id == run_id
            )).all()
            assert len(inputs) == 2
            evidence = [{
                "input_id": item.id, "document_id": item.document_id,
                "document_version": item.document_version,
                "content_sha256": item.content_sha256,
            } for item in inputs]
            forward = _frozen_context_identities(db, run_id, evidence)
            backward = _frozen_context_identities(db, run_id, list(reversed(evidence)))
            assert forward == backward
            assert [document_id for document_id, _ in forward] == sorted(
                item.document_id for item in inputs
            )


@pytest.mark.parametrize("decision", ["confirm", "reject"])
@pytest.mark.parametrize("variant", ["formal", "history"])
def test_reviewed_fact_is_not_revoked_but_new_context_gets_new_candidate(decision, variant):
    role = "chapter" if variant == "history" else "character_profile"
    options = ({"role": role, "support_id": "L1:A1"}
               if variant == "formal" else {"role": role})
    with TestClient(app) as client:
        project_id, document_id = _project_and_document(client, role=role)
        first_run = _run(client, project_id)
        with SessionLocal() as db:
            first, _ = _upsert(db, project_id, first_run, **options)
            first_id = first.id
            db.commit()
        path = (
            f"/api/v1/projects/{project_id}/characters/{quote('林澈', safe='')}"
            f"/profile-candidates/{first_id}/decisions"
        )
        response = client.post(
            path, json={"decision": decision, "expected_revision": 0}
        )
        assert response.status_code == 201, response.text
        _context_revision(client, project_id, document_id, expected=1, status="retired")
        _context_revision(client, project_id, document_id, expected=2, status="published")
        next_run = _run(client, project_id)
        with SessionLocal() as db:
            fresh, created = _upsert(db, project_id, next_run, **options)
            assert created and fresh.id != first_id
            assert db.get(CharacterTraitCandidateRow, first_id).review_state == (
                "confirmed" if decision == "confirm" else "rejected"
            )
            frozen_confirmed = db.scalars(select(AnalysisRunCharacterTraitInputRow).where(
                AnalysisRunCharacterTraitInputRow.run_id == next_run
            )).all()
            assert [item.candidate_id for item in frozen_confirmed] == (
                [first_id] if decision == "confirm" else []
            )


@pytest.mark.parametrize("damage", ["scope", "fingerprint"])
def test_reviewed_nonformal_row_with_damaged_identity_is_never_reused(damage):
    with TestClient(app) as client:
        project_id, document_id = _project_and_document(client, role="chapter")
        first_run = _run(client, project_id)
        with SessionLocal() as db:
            old, _ = _upsert(db, project_id, first_run, role="chapter")
            old_id = old.id
            db.commit()
        reject = client.post(
            f"/api/v1/projects/{project_id}/characters/"
            f"{quote('林澈', safe='')}/profile-candidates/{old_id}/decisions",
            json={"decision": "reject", "expected_revision": 0},
        )
        assert reject.status_code == 201, reject.text
        new_run = _run(client, project_id)
        with SessionLocal() as db:
            old = db.get(CharacterTraitCandidateRow, old_id)
            if damage == "scope":
                old.scope_payload = {"schema_version": 1, "timeline_key": "different"}
            else:
                old.candidate_fingerprint = "f" * 64
            db.commit()
            if damage == "scope":
                with pytest.raises(ValueError, match="reused candidate scope is invalid"):
                    _upsert(db, project_id, new_run, role="chapter")
            else:
                fresh, created = _upsert(db, project_id, new_run, role="chapter")
                assert created and fresh.id != old_id


def test_forged_review_state_without_append_only_decision_cannot_be_reused():
    with TestClient(app) as client:
        project_id, _ = _project_and_document(client)
        first_run = _run(client, project_id)
        with SessionLocal() as db:
            first, _ = _upsert(db, project_id, first_run)
            first.review_state = "superseded"
            first.lock_version = 2
            first_id = first.id
            db.commit()
        second_run = _run(client, project_id)
        with SessionLocal() as db:
            with pytest.raises(ValueError, match="review state is invalid"):
                _upsert(db, project_id, second_run)
            assert db.get(CharacterTraitCandidateRow, first_id).review_state == "superseded"


def test_multihop_supersession_review_chain_can_reuse_first_candidate():
    with TestClient(app) as client:
        project_id, _ = _project_and_document(client)
        first_run = _run(client, project_id)
        with SessionLocal() as db:
            first, _ = _upsert(db, project_id, first_run)
            first_id = first.id
            db.commit()
        first_confirm = client.post(
            f"/api/v1/projects/{project_id}/characters/{quote('林澈', safe='')}"
            f"/profile-candidates/{first_id}/decisions",
            json={"decision": "confirm", "expected_revision": 0},
        )
        assert first_confirm.status_code == 201, first_confirm.text
        with SessionLocal() as db:
            successor_payload = _payload(db, first_run)
            successor_payload["supersedes_candidate_id"] = first_id
            successor, created = upsert_character_trait_candidate(
                db, project_id=project_id, source_run_id=first_run,
                candidate=successor_payload,
            )
            assert created
            successor_id = successor.id
            db.commit()
        successor_confirm = client.post(
            f"/api/v1/projects/{project_id}/characters/{quote('林澈', safe='')}"
            f"/profile-candidates/{successor_id}/decisions",
            json={"decision": "confirm", "expected_revision": 0},
        )
        assert successor_confirm.status_code == 201, successor_confirm.text
        with SessionLocal() as db:
            last_payload = _payload(db, first_run)
            last_payload["supersedes_candidate_id"] = successor_id
            last, created = upsert_character_trait_candidate(
                db, project_id=project_id, source_run_id=first_run,
                candidate=last_payload,
            )
            assert created
            last_id = last.id
            db.commit()
        last_confirm = client.post(
            f"/api/v1/projects/{project_id}/characters/{quote('林澈', safe='')}"
            f"/profile-candidates/{last_id}/decisions",
            json={"decision": "confirm", "expected_revision": 0},
        )
        assert last_confirm.status_code == 201, last_confirm.text
        with SessionLocal() as db:
            assert db.get(CharacterTraitCandidateRow, first_id).review_state == "superseded"
            assert db.get(CharacterTraitCandidateRow, successor_id).review_state == "superseded"
            assert db.get(CharacterTraitCandidateRow, last_id).review_state == "confirmed"
        second_run = _run(client, project_id)
        with SessionLocal() as db:
            same, created = _upsert(db, project_id, second_run)
            assert not created and same.id == first_id

        # A's supersession is genuine only while B's original confirmation
        # audit still proves the author approved B at lock version zero.
        with SessionLocal() as db:
            confirmation = db.scalar(select(CharacterTraitReviewRow).where(
                CharacterTraitReviewRow.candidate_id == successor_id,
                CharacterTraitReviewRow.decision == "confirm",
            ))
            assert confirmation is not None
            confirmation.expected_lock_version = 7
            db.commit()
        third_run = _run(client, project_id)
        with SessionLocal() as db:
            with pytest.raises(ValueError, match="supersession is invalid"):
                _upsert(db, project_id, third_run)
