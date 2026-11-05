"""Source retention never asserts a role, event, or object was verified."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from app.character_material_coverage import DraftMaterialGap, DraftMaterialTracker
from app.character_retained_habit_claims import RetainedHabitClaim, collect_retained_habit_claims, MAX_RETAINED_HABIT_CLAIMS
from app.character_scope_review import ScopeReviewSourceIdentity
from app.character_trait_extraction import CharacterSignalChunk
from app.config import SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2, SELECTED_CLAIM_RESOLUTION_POLICY_V3
from app.narrative_context import payload_sha256
from tests.test_character_draft_actor_stage_wiring import _draft_source, _partial_result, _result
from tests.test_selected_claim_resolution import paired, reconcile, RAW, CONTENT, TARGET, OBJECT, STATEMENT, CONTEXT


def retained_pair(**kwargs):
    return paired(policy=SELECTED_CLAIM_RESOLUTION_POLICY_V3, key_object="话筒", **kwargs)


def test_alias_record_is_source_retained_without_old_proposal_or_acceptance_then_independently_resolved():
    pair = retained_pair()
    source, _, primary, targeted, provider, _ = pair
    assert not primary.signals and not primary.draft_observations and not provider.original_requests
    assert primary.diagnostics.rejected_records == 1 and primary.diagnostics.outcome != "completed"
    gap, = primary.material_gaps
    assert gap.reason == "object_unverified" and gap.line_start == 1
    pending, = primary.claim_resolution_state.pending.values()
    assert pending.deferred is None and isinstance(pending.retained, RetainedHabitClaim)
    assert not hasattr(pending.retained, "proposal") and not hasattr(pending.retained, "preflight_signal")
    assert pending.original.record.statement == STATEMENT and pending.original.record.context == CONTEXT
    assert pending.original.record.key_object == "话筒" and "话筒" not in CONTENT
    assert pending.stable_key == gap.claim_digest == payload_sha256({"source": source.content_sha256,
        "document": source.document.id, "record": provider.raw})
    assert not primary.material_resolved_claims and not primary.claim_resolution_state.certificates
    manager, ledger, admitted, charges, _ = reconcile(pair)
    request, = provider.resolution_requests
    assert request.prompt_version.endswith("-v18") and request.original_claim.key_object == "话筒"
    assert request.lines == targeted.selected_action_certificates[0].request.lines
    assert len(charges) == 1 and admitted == list(targeted.signals) and admitted[0].key_object == OBJECT
    assert manager.receipts[0]["primary_claim_sha256"] == pending.stable_key
    assert manager.receipts[0]["method"] == "selected_claim_recheck_v2"
    assert manager.scopes[0]["status"] == "complete"
    assert (source.input_id, "candidate-1", pending.stable_key) in ledger.resolved_scopes
    assert (source.input_id, "candidate-2", pending.stable_key) not in ledger.resolved_scopes


@pytest.mark.parametrize("policy", [SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2])
def test_historical_policies_do_not_gain_source_retention(policy):
    pair = paired(policy=policy, key_object="话筒")
    assert not pair[2].claim_resolution_state.pending
    manager, _, _, charges, _ = reconcile(pair)
    assert not manager.receipts and not charges and not pair[4].resolution_requests


@pytest.mark.parametrize("damage", ["original_different", "original_uncertain", "multiple", "actor", "actuality", "object_quote"])
def test_retained_record_stays_unresolved_when_independent_semantics_fail(damage):
    pair = retained_pair(damage=damage)
    manager, _, admitted, charges, _ = reconcile(pair)
    assert len(charges) == len(pair[4].resolution_requests) == 1 and not pair[4].original_requests
    assert not admitted and not manager.receipts and manager.scopes[0]["status"] == "partial"


@pytest.mark.parametrize("problem", ["unadmitted_origin", "no_certificate", "unadmitted_final", "source", "target", "budget", "unknown", "omitted"])
def test_missing_admission_source_or_coverage_never_falls_back_to_legacy_review(problem):
    pair = retained_pair()
    primary, targeted = pair[2:4]
    if problem == "no_certificate":
        pair = (*pair[:3], targeted.model_copy(update={"selected_action_certificates": ()}), *pair[4:])
    elif problem in {"source", "target"}:
        pending = next(iter(primary.claim_resolution_state.pending.values()))
        if problem == "source":
            changed = replace(pending.retained, source=pending.retained.source.model_copy(update={"run_input_id": "other-run-input"}))
            pending = replace(pending, retained=changed)
        else:
            pending = replace(pending, target_sha256="0" * 64)
        primary.claim_resolution_state.pending = {pending.stable_key: pending}
    elif problem == "unknown":
        primary.claim_resolution_state.gaps = (*primary.claim_resolution_state.gaps, DraftMaterialGap(reason="source_unverified"))
    elif problem == "omitted":
        primary.claim_resolution_state.omitted_count = 1
    manager, _, _, _, _ = reconcile(pair, origin_ok=problem != "unadmitted_origin", admit_ok=problem != "unadmitted_final",
        budget=0 if problem == "budget" else 40000)
    assert not manager.repairs_document(pair[0].document.id, "candidate-1", 1)
    assert not pair[4].original_requests
    if problem in {"unadmitted_origin", "no_certificate", "source", "target", "budget"}:
        assert not pair[4].resolution_requests
    if problem == "unadmitted_final":
        assert not manager.receipts


def collector_fixture(rows, *, targets=(TARGET,), content=CONTENT, reason="key_object_support"):
    source = _draft_source(content)
    identity = ScopeReviewSourceIdentity(run_input_id=source.input_id, document_id=source.document.id,
        document_version=source.document_version, content_sha256=source.content_sha256)
    chunk = CharacterSignalChunk(source.document.id, source.document.name, content, 1, "draft")
    validation = SimpleNamespace(parsed=True, failures=tuple(SimpleNamespace(record_index=i, reason=reason) for i in range(len(rows))),
        target_bound_draft_deferred=(), draft_actor_deferred=(),
        draft_trace_attempt=SimpleNamespace(events=[SimpleNamespace(outcome="rejected", record_ordinal=i + 1) for i in range(len(rows))]))
    response = json.dumps({"records": rows}, ensure_ascii=False)
    retained, omitted = collect_retained_habit_claims(validation, response, chunk=chunk, targets=targets,
        source_identity=identity, frozen_content=content)
    validation.retained_habit_claims, validation.retained_habit_claims_omitted = retained, omitted
    return identity, chunk, validation, response, retained, omitted


@pytest.mark.parametrize("problem", ["quote", "line", "wrong_character", "absent_character", "wrong_target", "ambiguous_target", "other_dimension", "declaration", "other_failure"])
def test_retention_requires_exact_source_and_one_authorized_habit_without_relaxing_other_fields(problem):
    raw, content, targets, reason = {**RAW, "key_object": "话筒"}, CONTENT, (TARGET,), "key_object_support"
    if problem == "quote": raw["evidence"] = "不是冻结原文"
    elif problem == "line": raw["source_line_start"] = raw["source_line_end"] = 2
    elif problem == "wrong_character": raw["character"] = "另一个角色"
    elif problem == "absent_character":
        content = CONTENT.replace(TARGET.character, "他")
        raw["evidence"] = content
    elif problem == "wrong_target": targets = (TARGET.model_copy(update={"trait_key": "other", "comparison_key": "contextual_behavior:other"}),)
    elif problem == "ambiguous_target": targets = (TARGET, TARGET.model_copy(update={"habit_contexts": ("别的场合",)}))
    elif problem == "other_dimension": raw["dimension"] = "preference"
    elif problem == "declaration": raw["observation_kind"] = "explicit_declaration"
    elif problem == "other_failure": reason = "statement_support"
    *_, retained, omitted = collector_fixture([raw], targets=targets, content=content, reason=reason)
    assert not retained and not omitted


def test_retention_and_gap_caps_keep_omission_visible_and_exact_later_acceptance_clears_pending():
    rows = [{**RAW, "key_object": "话筒", "statement": f"第{i}项原主张"} for i in range(MAX_RETAINED_HABIT_CLAIMS + 1)]
    identity, chunk, validation, response, retained, omitted = collector_fixture(rows)
    assert len(retained) == MAX_RETAINED_HABIT_CLAIMS and omitted == 1
    tracker = DraftMaterialTracker(identity, CONTENT, chunk, claim_resolution_enabled=True, selected_claim_resolution_enabled=True)
    tracker.capture(response, validation)
    result = tracker.finish(_partial_result())
    assert len(result.claim_resolution_state.pending) == MAX_RETAINED_HABIT_CLAIMS
    assert result.material_gaps_omitted == 1 and not result.claim_resolution_state.enumeration_complete
    identity, chunk, validation, response, *_ = collector_fixture([{**RAW, "key_object": "话筒"}])
    tracker = DraftMaterialTracker(identity, CONTENT, chunk, claim_resolution_enabled=True, selected_claim_resolution_enabled=True)
    tracker.capture(response, validation)
    assert len(tracker.claim_state.pending) == 1
    validation.failures, validation.retained_habit_claims = (), ()
    validation.draft_trace_attempt.events[0].outcome = "accepted"
    tracker.capture(response, validation, package_attempt=2)
    assert not tracker.claim_state.pending and tracker.finish(_result()).material_gaps == ()


def test_retained_original_cannot_be_rechecked_after_a_policy_downgrade_or_cancellation():
    pair = retained_pair()
    pair[5]._character_selected_claim_resolution_policy = SELECTED_CLAIM_RESOLUTION_POLICY_V2
    manager, _, _, charges, _ = reconcile(pair)
    assert not manager.receipts and not charges and not pair[4].resolution_requests
    pair = retained_pair()
    visits = []
    def cancel():
        visits.append(1)
        if len(visits) == 2: raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        reconcile(pair, checkpoint=cancel)
    assert len(pair[4].resolution_requests) == 1 and not pair[4].original_requests
