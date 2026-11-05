"""A later observation only resolves a claim through a private bound receipt."""
from copy import deepcopy
from hashlib import sha256

import pytest

from app.character_claim_resolution import CLAIM_RESOLUTION_KEY, CLAIM_RESOLUTION_SCHEMA
from app.character_claim_resolution_storage import (
    ClaimResolutionIntegrityError, validate_claim_resolution_payload,
)
from app.character_observations import CharacterObservationRecorder, ObservationMatchDecision
from app.config import CLAIM_RESOLUTION_POLICY_V1, HABIT_OBJECT_CHAIN_POLICY_V1
from app.db import AnalysisDiagnosticRow, CharacterObservationRow
from app.narrative_context import payload_sha256
from tests.test_character_observations import baseline, db, observation, seed
from tests.test_character_target_bound_draft_review_v4 import _target
from tests.test_report_evidence_preview_api import evidence_api


def _hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def _seal(payload):
    result = deepcopy(payload)
    for receipt in result["receipts"]:
        previous = receipt["id"]
        receipt["id"] = "cr_" + payload_sha256({key: value for key, value in receipt.items() if key != "id"})
        for scope in result["final_scopes"]:
            scope["receipt_ids"] = [receipt["id"] if item == previous else item for item in scope["receipt_ids"]]
    result["payload_sha256"] = payload_sha256({key: value for key, value in result.items() if key != "payload_sha256"})
    return result


def _stored(db):
    project, run, source = seed(db)
    frozen, _ = baseline(db, project, run)
    run.batch_coverage = {
        "character_review_enabled": True, "habit_object_chain_policy": HABIT_OBJECT_CHAIN_POLICY_V1,
        "claim_resolution_policy": CLAIM_RESOLUTION_POLICY_V1,
    }
    target = _target(character="林舟", dimension="preference", trait_key="进食选择",
                     comparison_key="preference:餐食", key_object="餐食", baseline_hint="林舟通常正常进食")
    admitted = observation(source, target_bound_target=target, matches=(ObservationMatchDecision(
        baseline_input_id=frozen.id, status="matched", reason_code="matched", match_method="target_bound",
    ),))
    CharacterObservationRecorder().record(db, run_id=run.id, project_id=project.id, observations=[admitted])
    binding = dict(
        source_input_id=source.id, source_document_id=source.document_id,
        source_content_sha256=source.content_sha256, confirmed_candidate_id=frozen.candidate_id,
        confirmed_payload_sha256=frozen.payload_sha256,
        target_sha256=payload_sha256(target.model_dump(mode="json", exclude={"target_ordinal"})),
    )
    receipt = dict(
        **binding, id="cr_" + "0" * 64, method="original_claim_recheck",
        primary_claim_sha256="1" * 64, review_claim_sha256="2" * 64,
        request_sha256="3" * 64, proposal_id="tdp_" + "4" * 32,
        observation_id=admitted.observation_id, line_start=1, line_end=1,
        text_sha256=_hash(admitted.evidence.text), observation_statement_sha256=_hash(admitted.statement),
    )
    scope = dict(**binding, chunk_line_start=1, chunk_line_end=2, status="complete",
                 claim_count=1, resolved_count=1, unknown_count=0, omitted_count=0, receipt_ids=[receipt["id"]])
    payload = _seal(dict(schema_version=CLAIM_RESOLUTION_SCHEMA, receipts=[receipt], final_scopes=[scope], omitted_count=0))
    return run, source, admitted, payload


def test_bound_receipt_survives_private_storage_and_old_absence_stays_absent(db):
    run, source, admitted, payload = _stored(db)
    assert validate_claim_resolution_payload(db, run=run, payload=payload) == payload
    db.add(AnalysisDiagnosticRow(run_id=run.id, payload={"character_consistency": {CLAIM_RESOLUTION_KEY: payload}}))
    db.commit()
    stored = db.get(AnalysisDiagnosticRow, run.id).payload["character_consistency"][CLAIM_RESOLUTION_KEY]
    assert validate_claim_resolution_payload(db, run=run, payload=stored) == payload
    assert source.content not in str(stored) and admitted.statement not in str(stored)
    _, legacy, _ = seed(db, prefix="legacy-")
    assert validate_claim_resolution_payload(db, run=legacy, payload=None) is None
    with pytest.raises(ClaimResolutionIntegrityError):
        validate_claim_resolution_payload(db, run=legacy, payload=payload)


@pytest.mark.parametrize("mutation", ["raw_text", "source", "candidate", "target", "quote", "statement",
                                      "observation", "digest", "unknown_complete", "missing_receipt", "count", "omitted"])
def test_tampered_receipts_cannot_become_final_coverage_even_with_recomputed_envelope(db, mutation):
    run, _, _, payload = _stored(db)
    receipt, = payload["receipts"]
    scope, = payload["final_scopes"]
    if mutation == "raw_text":
        receipt["statement"] = "must never persist raw model text"
    elif mutation == "source":
        receipt["source_input_id"] = scope["source_input_id"] = "another-input"
    elif mutation == "candidate":
        receipt["confirmed_candidate_id"] = scope["confirmed_candidate_id"] = "another-candidate"
    elif mutation == "target":
        receipt["target_sha256"] = scope["target_sha256"] = "9" * 64
    elif mutation == "quote":
        receipt["line_start"] = receipt["line_end"] = 2
    elif mutation == "statement":
        receipt["observation_statement_sha256"] = "9" * 64
    elif mutation == "observation":
        receipt["observation_id"] = "cs_" + "9" * 32
    elif mutation == "digest":
        payload["payload_sha256"] = "9" * 64
    elif mutation == "unknown_complete":
        scope["unknown_count"] = 1
        scope["claim_count"] = 2
    elif mutation == "missing_receipt":
        scope["receipt_ids"] = []
    elif mutation == "count":
        scope["resolved_count"] = True
    else:
        payload["omitted_count"] = 1
    value = payload if mutation == "digest" else _seal(payload)
    with pytest.raises(ClaimResolutionIntegrityError):
        validate_claim_resolution_payload(db, run=run, payload=value)


def test_partial_scope_without_receipt_is_unknown_and_cannot_claim_a_resolution(db):
    run, _, _, payload = _stored(db)
    payload["receipts"] = []
    scope, = payload["final_scopes"]
    scope.update(status="partial", resolved_count=0, unknown_count=1, receipt_ids=[])
    payload = _seal(payload)
    assert validate_claim_resolution_payload(db, run=run, payload=payload) == payload


def test_receipts_recheck_observation_integrity_and_same_run_source(db):
    run, source, admitted, payload = _stored(db)
    stored = db.get(CharacterObservationRow, (run.id, admitted.observation_id))
    original = deepcopy(stored.payload)
    stored.payload = {**stored.payload, "statement": "changed after recording"}
    with pytest.raises(ClaimResolutionIntegrityError):
        validate_claim_resolution_payload(db, run=run, payload=payload)
    stored.payload = original
    _, other, other_source = seed(db, prefix="other-")
    other.batch_coverage = run.batch_coverage
    with pytest.raises(ClaimResolutionIntegrityError):
        validate_claim_resolution_payload(db, run=other, payload=payload)
    source.content = other_source.content + "new revision"
    with pytest.raises(ClaimResolutionIntegrityError):
        validate_claim_resolution_payload(db, run=run, payload=payload)


def test_receipt_is_hidden_from_author_diagnostics_and_export(evidence_api):
    state = evidence_api
    secret = "private claim correspondence sentinel"
    with state.sessions() as db:
        diagnostic = db.get(AnalysisDiagnosticRow, state.run.id)
        diagnostic.payload = {CLAIM_RESOLUTION_KEY: {"private": secret}, "character_consistency": {
            "outcome": "partial", CLAIM_RESOLUTION_KEY: {"private": secret},
        }}
        db.commit()
    url = f"/api/v1/analysis-runs/{state.run.id}"
    response = state.client.get(url + "/diagnostics")
    assert response.status_code == 200
    assert response.json() == {"character_consistency": {"outcome": "partial"}}
    exported = state.client.get(url + "/export.md")
    assert exported.status_code == 200
    assert secret not in exported.text and CLAIM_RESOLUTION_KEY not in exported.text
    assert state.client.get(f"/api/v1/analysis-runs/{state.foreign_run.id}/diagnostics").status_code == 404
