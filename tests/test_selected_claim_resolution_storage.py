"""Selected claim receipts bind actual private V16 observation provenance."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.character_action_selection import (
    ActionChoice, ActionSelectionResponse, build_selected_action_review_request,
    select_action_window, selection_request_digest,
)
from app.character_claim_resolution import CLAIM_RESOLUTION_SCHEMA
from app.config import SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2
from app.character_claim_resolution_storage import ClaimResolutionIntegrityError, validate_claim_resolution_payload
from app.character_draft_actor_review import target_bound_draft_review_request_digest
from app.character_observations import CharacterObservationRecorder, ObservationMatchDecision, project_character_observations
from app.character_scope_review import ScopeReviewSourceIdentity
from app.character_trait_extraction import _target_bound_signal_id
from app.db import AnalysisDiagnosticRow, CharacterObservationRow
from app.domain import EvidenceSpan
from app.narrative_context import payload_sha256
from tests.test_character_observations import db, seed, baseline, observation
from tests.test_character_target_bound_draft_review_v4 import _target
from tests.test_claim_resolution_storage import _hash, _seal
from tests.test_selected_claim_resolution_policy import POLICIES
from tests.test_report_evidence_preview_api import evidence_api, _read_only


def _stored_selected(db, *, workspace_id=None, policy=SELECTED_CLAIM_RESOLUTION_POLICY_V1):
    project, run, source = seed(db, **({"workspace_id": workspace_id} if workspace_id else {}))
    source.content = "林舟接过餐盘，把它推开。"
    source.content_sha256 = _hash(source.content)
    run.batch_coverage = {"character_review_enabled": True, **POLICIES, "selected_claim_resolution_policy": policy}
    frozen, _ = baseline(db, project, run)
    target = _target(character="林舟", dimension="contextual_behavior", trait_key="进食选择",
        comparison_key="contextual_behavior:进食选择", key_object="", baseline_hint="林舟用餐时习惯检查餐盘",
        habit_contexts=("用餐时",))
    source_id = ScopeReviewSourceIdentity(run_input_id=source.id, document_id=source.document_id,
        document_version=source.document_version, content_sha256=source.content_sha256)
    window, _ = select_action_window(source=source_id, frozen_content=source.content, target=target, line_start=1, line_end=1)
    path = tuple(c.support_id for c in window.window)
    response = ActionSelectionResponse(schema_version="habit-action-selection-v1", request_digest=selection_request_digest(window),
        status="selected", items=(ActionChoice(fact_clause_id=path[-1], actor_anchor_id=path[0], path_ids=path),))
    request = build_selected_action_review_request(window, response, frozen_content=source.content)
    proposal, = request.proposals
    origin = dict(request_sha256=target_bound_draft_review_request_digest(request), proposal_id=proposal.proposal_id,
        selection_digest=request.selection_digest, fact_clause_id=proposal.fact_clause_id,
        path_ids=list(path), window_ids=list(request.selection_window_ids))
    admitted = observation(source, observation_id=_target_bound_signal_id(chunk=SimpleNamespace(document_id=source.document_id),
        target=target, proposal=proposal, habit_event_object="餐盘"), dimension=target.dimension,
        key_object="餐盘", statement=proposal.canonical_statement, target_bound_target=target, selected_action_origin=origin,
        evidence=EvidenceSpan(document_id=source.document_id, document_name=source.document_name, line_start=1, line_end=1, text=source.content),
        matches=(ObservationMatchDecision(baseline_input_id=frozen.id, status="matched", reason_code="matched", match_method="target_bound"),))
    CharacterObservationRecorder().record(db, run_id=run.id, project_id=project.id, observations=[admitted])
    binding = dict(source_input_id=source.id, source_document_id=source.document_id, source_content_sha256=source.content_sha256,
        confirmed_candidate_id=frozen.candidate_id, confirmed_payload_sha256=frozen.payload_sha256,
        target_sha256=payload_sha256(target.model_dump(mode="json", exclude={"target_ordinal"})))
    receipt = dict(**binding, id="cr_" + "0" * 64,
        method="selected_claim_recheck" if policy == SELECTED_CLAIM_RESOLUTION_POLICY_V1 else "selected_claim_recheck_v2",
        primary_claim_sha256="1" * 64,
        review_claim_sha256="2" * 64, request_sha256="3" * 64, proposal_id=proposal.proposal_id,
        observation_id=admitted.observation_id, line_start=1, line_end=1, text_sha256=_hash(admitted.evidence.text),
        observation_statement_sha256=_hash(admitted.statement), origin_request_sha256=origin["request_sha256"],
        origin_observation_id=admitted.observation_id, fact_clause_id=proposal.fact_clause_id)
    scope = dict(**binding, chunk_line_start=1, chunk_line_end=1, status="complete", claim_count=1,
        resolved_count=1, unknown_count=0, omitted_count=0, receipt_ids=[receipt["id"]])
    payload = _seal(dict(schema_version=CLAIM_RESOLUTION_SCHEMA, receipts=[receipt], final_scopes=[scope], omitted_count=0))
    return run, admitted, payload


@pytest.mark.parametrize("policy", [SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2])
def test_selected_receipt_uses_persisted_origin_and_public_observation_has_no_private_hash(db, policy):
    run, admitted, payload = _stored_selected(db, policy=policy)
    assert validate_claim_resolution_payload(db, run=run, payload=payload) == payload
    stored = db.get(CharacterObservationRow, (run.id, admitted.observation_id))
    assert stored.payload["selected_action_origin"] == admitted.selected_action_origin.model_dump(mode="json")
    assert stored.payload_sha256 == payload_sha256(stored.payload)
    public, = project_character_observations(db, run)["items"]
    assert "selected_action_origin" not in public and "payload_sha256" not in public
    assert public["statement"] == admitted.statement
    assert public["target_bound_object_basis_id"] is None


@pytest.mark.parametrize("policy", [SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2])
def test_selected_receipt_method_cannot_cross_frozen_policy_versions(db, policy):
    run, _, payload = _stored_selected(db, policy=policy)
    payload["receipts"][0]["method"] = "selected_claim_recheck_v2" if policy == SELECTED_CLAIM_RESOLUTION_POLICY_V1 else "selected_claim_recheck"
    with pytest.raises(ClaimResolutionIntegrityError):
        validate_claim_resolution_payload(db, run=run, payload=_seal(payload))


@pytest.mark.parametrize("mutation", ["missing_origin", "wrong_origin", "wrong_request", "wrong_fact", "old_method",
                                      "missing_policy", "missing_selection", "null_extra", "missing_field"])
def test_selected_receipt_cannot_resolve_by_format_or_unrelated_origin(db, mutation):
    run, admitted, payload = _stored_selected(db)
    receipt = payload["receipts"][0]
    if mutation == "missing_origin":
        stored = db.get(CharacterObservationRow, (run.id, admitted.observation_id))
        stored.payload = {k: v for k, v in stored.payload.items() if k != "selected_action_origin"}
        stored.payload_sha256 = payload_sha256(stored.payload)
    elif mutation == "wrong_origin":
        receipt["origin_observation_id"] = "cs_" + "f" * 32
    elif mutation == "wrong_request":
        receipt["origin_request_sha256"] = "f" * 64
    elif mutation == "wrong_fact":
        receipt["fact_clause_id"] = "L1:A1"
    elif mutation == "old_method":
        receipt["method"] = "original_claim_recheck"
    elif mutation in {"missing_policy", "missing_selection"}:
        key = "selected_claim_resolution_policy" if mutation == "missing_policy" else "habit_action_selection_policy"
        run.batch_coverage = {k: v for k, v in run.batch_coverage.items() if k != key}
    elif mutation == "null_extra":
        receipt["origin_request_sha256"] = None
    else:
        del receipt["origin_request_sha256"]
    with pytest.raises(ClaimResolutionIntegrityError):
        validate_claim_resolution_payload(db, run=run, payload=_seal(payload))


def test_private_origin_stays_out_of_observation_api_diagnostics_and_export(evidence_api):
    state = evidence_api
    with state.sessions() as db:
        run, admitted, payload = _stored_selected(db, workspace_id=state.workspace.id, policy=SELECTED_CLAIM_RESOLUTION_POLICY_V2)
        run.status = "completed"
        db.add(AnalysisDiagnosticRow(run_id=run.id, payload={"character_consistency": {"claim_resolution_v1": payload}}))
        db.commit()
        run_id = run.id
    with _read_only(state):
        for route in ("character-observations", "diagnostics", "export.md"):
            response = state.client.get(f"/api/v1/analysis-runs/{run_id}/{route}")
            assert response.status_code == 200, response.text
            assert "selected_action_origin" not in response.text
            assert "origin_request_sha256" not in response.text
            assert admitted.selected_action_origin.request_sha256 not in response.text
        foreign = state.client.get(f"/api/v1/analysis-runs/{state.foreign_run.id}/character-observations")
        assert foreign.status_code == 404


@pytest.mark.parametrize("mutation", ["path", "window", "proposal", "statement", "target", "baseline", "source", "object"])
def test_rehashed_snapshot_still_requires_frozen_origin_structure_and_observation_identity(db, mutation):
    run, admitted, payload = _stored_selected(db)
    stored = db.get(CharacterObservationRow, (run.id, admitted.observation_id))
    value = deepcopy(stored.payload)
    if mutation == "path":
        value["selected_action_origin"]["path_ids"] = ["L1:A2"]
    elif mutation == "window":
        value["selected_action_origin"]["window_ids"] = ["L1:A2", "L1:A1"]
    elif mutation == "proposal":
        value["selected_action_origin"]["proposal_id"] = "tdp_" + "f" * 32
    elif mutation == "statement":
        value["statement"] = "篡改后的观察"
        payload["receipts"][0]["observation_statement_sha256"] = _hash(value["statement"])
    elif mutation == "target":
        value["target_bound_target"]["habit_contexts"] = ["另一个场景"]
    elif mutation == "baseline":
        value["matches"][0]["baseline"]["candidate_id"] = "another-candidate"
    elif mutation == "source":
        value["source"]["run_input_id"] = "another-run-input"
    else:
        value["key_object"] = "餐盘的标签"
    stored.payload = value
    stored.payload_sha256 = payload_sha256(value)
    with pytest.raises(ClaimResolutionIntegrityError):
        validate_claim_resolution_payload(db, run=run, payload=_seal(payload))
