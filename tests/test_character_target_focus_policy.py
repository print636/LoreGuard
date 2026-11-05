"""Server-only focus entitlement, frozen bindings and private projections."""
import json
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import CHARACTER_TARGET_FOCUS_POLICY_V1
from app.character_drift import (
    CHARACTER_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT,
    CHARACTER_HABIT_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT,
)
from app.character_target_focus import TARGET_FOCUS_SIGNAL_INSTRUCTIONS
from app.character_trait_extraction import CHARACTER_SIGNAL_SYSTEM_PROMPT, TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT
from app.character_observations import CharacterObservationRecorder, ObservationMatchDecision, project_character_observations
from app.db import AnalysisRunRow, AnalysisRunCharacterTraitInputRow, CharacterTraitCandidateRow, CharacterObservationRow, SessionLocal
from app.main import app, _public_diagnostics_payload, _serialize_issue, write_limiter
from app.narrative_context import payload_sha256
from app.service import (
    CharacterConsistencyUsageAccumulator, _CharacterConsistencyAccountingProvider,
    _character_stage_settings_for_run, _capture_confirmed_traits,
    _target_support_snapshot_fields, _safe_character_failure,
)
from app.usage import estimate_issue_evidence_review_tokens
from tests.test_habit_scene_policy import local_run
from tests.test_selected_claim_resolution_policy import POLICIES
from tests.test_habit_action_selection_policy import _RecordingProvider
from tests.test_character_observations import db, seed, baseline, observation


@pytest.mark.parametrize("missing", [None, "target_focus_policy", *POLICIES, "habit_scene_policy", "character_review_enabled"])
def test_focus_requires_frozen_full_authorization_and_never_upgrades_old_retry(missing):
    coverage = {**POLICIES, "target_focus_policy": CHARACTER_TARGET_FOCUS_POLICY_V1}
    if missing:
        coverage[missing] = None
    base, stage = local_run(coverage_changes=coverage)
    assert stage._character_target_focus_policy == (None if missing else CHARACTER_TARGET_FOCUS_POLICY_V1)
    assert base._character_target_focus_policy is None
    assert "_character_target_focus_policy" not in stage.model_dump()
    old = AnalysisRunRow(sensitivity="balanced", batch_coverage={})
    assert _character_stage_settings_for_run(stage, old)._character_target_focus_policy is None


@pytest.mark.parametrize("prompt", [
    CHARACTER_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT, CHARACTER_HABIT_TARGET_FOCUS_REVIEW_SYSTEM_PROMPT,
    CHARACTER_SIGNAL_SYSTEM_PROMPT + TARGET_FOCUS_SIGNAL_INSTRUCTIONS,
    TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT + TARGET_FOCUS_SIGNAL_INSTRUCTIONS,
], ids=["single-review", "habit-review", "primary", "targeted"])
def test_exact_new_purpose_requires_focus_policy_and_keeps_existing_metering(prompt):
    _, stage = local_run(coverage_changes={**POLICIES, "target_focus_policy": CHARACTER_TARGET_FOCUS_POLICY_V1})
    stage = stage.model_copy(update={"character_draft_actor_review_v1": False, "character_signal_full_line_prompt_v2": False})
    transport, usage = _RecordingProvider(), CharacterConsistencyUsageAccumulator()
    def provider(settings):
        return _CharacterConsistencyAccountingProvider(settings, usage, signal_provider=transport, drift_provider=transport)
    legacy = stage.model_copy()
    legacy._character_target_focus_policy = None
    for settings, system in ((legacy, prompt), (stage, prompt + "x"), (stage, prompt + TARGET_FOCUS_SIGNAL_INSTRUCTIONS)):
        with pytest.raises(RuntimeError, match="unsupported"):
            provider(settings).complete(system, "{}")
    assert not transport.sent and usage.logical_calls == 0
    provider(stage).complete(prompt, "{}")
    reserve = stage.character_signal_max_completion_tokens if prompt.endswith(TARGET_FOCUS_SIGNAL_INSTRUCTIONS) else stage.character_drift_max_completion_tokens
    assert usage.logical_calls == usage.successful_calls == 1
    assert usage.charged_tokens == estimate_issue_evidence_review_tokens(prompt, "{}", completion_reserve=reserve)
    assert transport.sent[0] == (prompt, "{}")


def test_new_snapshot_freezes_verified_target_without_changing_old_candidate_or_snapshot():
    from tests.test_character_api_contract import _project_and_document, _completed_run, _candidate, _confirm
    write_limiter.events.clear()
    with TestClient(app) as client:
        project, _ = _project_and_document(client, content="林澈喜欢蜜瓜。林澈喜欢葡萄。")
        source_run = _completed_run(client, project["id"])
        candidate_id = _candidate(project["id"], source_run["id"], value="喜欢蜜瓜", support_id="L1:A1")
        assert _confirm(client, project["id"], candidate_id).status_code == 201
        with SessionLocal() as session:
            candidate = session.get(CharacterTraitCandidateRow, candidate_id)
            for policy in (None, CHARACTER_TARGET_FOCUS_POLICY_V1):
                run = AnalysisRunRow(id=str(uuid4()), project_id=project["id"], status="queued", batch_coverage={"target_focus_policy": policy})
                session.add(run)
                session.flush()
                _capture_confirmed_traits(session, run, [])
                session.flush()
                frozen = session.scalar(select(AnalysisRunCharacterTraitInputRow).where(AnalysisRunCharacterTraitInputRow.run_id == run.id))
                assert frozen.payload["value"] == candidate.value == "喜欢蜜瓜"
                assert frozen.payload_sha256 == payload_sha256(frozen.payload)
                if policy is None:
                    old_payload = frozen.payload.copy()
                    assert "target_support_bindings_v1" not in frozen.payload
                else:
                    assert frozen.payload == {**old_payload, "target_support_bindings_v1": candidate.support_bindings_v1}
            original_fingerprint = candidate.candidate_fingerprint
            candidate.candidate_fingerprint = "0" * 64
            assert _target_support_snapshot_fields(session, candidate) == {}
            candidate.candidate_fingerprint = original_fingerprint
            candidate.source_run_id = run.id  # Real input from another run cannot supply this target.
            assert _target_support_snapshot_fields(session, candidate) == {}
            candidate.support_binding_mode = "legacy_v1"
            assert _target_support_snapshot_fields(session, candidate) == {}
            session.rollback()


def test_public_observation_filters_target_binding_and_hashes_the_visible_payload(db):
    project, run, source = seed(db)
    frozen, _ = baseline(db, project, run)
    frozen.payload = {**frozen.payload, "target_support_bindings_v1": {"private": "sentinel"}}
    frozen.payload_sha256 = payload_sha256(frozen.payload)
    recorder = CharacterObservationRecorder()
    item = observation(source, matches=(ObservationMatchDecision(baseline_input_id=frozen.id, status="matched", reason_code="matched", identity_match=True, match_method="comparison_identity"),))
    recorder.record(db, run_id=run.id, project_id=project.id, observations=[item])
    public = project_character_observations(db, run)
    visible = public["items"][0]["matches"][0]["baseline"]
    assert "target_support_bindings_v1" not in visible["payload"]
    assert visible["payload_sha256"] == payload_sha256(visible["payload"]) != frozen.payload_sha256
    stored = db.get(CharacterObservationRow, (run.id, item.observation_id))
    assert stored.payload["matches"][0]["baseline"]["payload_sha256"] == frozen.payload_sha256
    assert "sentinel" not in json.dumps(public)


def test_private_focus_and_failure_details_never_enter_public_issue_or_nested_diagnostics():
    private = {"target_focus_review_v1": {"private": "sentinel"}, "failure_diagnostic_v1": {"exception_type": "TypeError"}}
    diagnostic = {"character_consistency": {"outcome": "degraded", "reason": "internal_failure", **private, "case_trace": [{**private, "reviewed_events": []}]}}
    public = _public_diagnostics_payload(diagnostic)
    assert public == {"character_consistency": {"outcome": "degraded", "reason": "internal_failure", "case_trace": [{"reviewed_events": []}]}}
    row = SimpleNamespace(id="issue", report_class="formal", category="character_drift", severity="high", confidence=.9, title="title", explanation="explanation", evidence=[], suggestion="suggestion", extra={**private, "confirmed_candidate_id": "candidate"})
    assert _serialize_issue(row)["metadata"] == {"confirmed_candidate_id": "candidate"}
    detail = _safe_character_failure(TypeError("secret-source-and-key"), stage="character_stage_execution")
    assert detail == {"schema_version": "character-failure-diagnostic-v1", "stage": "character_stage_execution", "exception_type": "TypeError"}
    opaque = type("secret-source-and-key", (RuntimeError,), {})
    assert _safe_character_failure(opaque(), stage="character_stage_execution")["exception_type"] == "OtherException"
