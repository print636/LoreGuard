"""Frozen authorization and durable source validation for explanation V3."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.character_explanation_assertions import (
    EXPLANATION_ASSERTION_REFS_KEY, AssertionSelection, ExplanationAssertionLink,
    assertion_receipt_payload, make_assertion_binding, payload_sha256, text_sha256,
)
from app.character_explanation_assertion_storage import validate_explanation_assertion_payload
from app.character_explanation_review import (
    EXPLANATION_REVIEW_SYSTEM_PROMPT, EXPLANATION_REVIEW_SYSTEM_PROMPT_V3,
    ExplanationBaselineSummary,
)
from app.character_observations import CharacterObservationRecorder, ObservationMatchDecision
from app.config import CHARACTER_EXPLANATION_ASSERTIONS_POLICY_V1, CHARACTER_TARGET_FOCUS_POLICY_V1
from app.db import AnalysisRunRow, AnalysisDiagnosticRow, IssueRow, SessionLocal
from app.domain import EvidenceSpan, ConsistencyIssue
from app.main import _public_diagnostics_payload, app, write_limiter
from app.runtime_provenance import safe_runtime_provenance
from app.service import _character_stage_settings_for_run, _CharacterConsistencyAccountingProvider, CharacterConsistencyUsageAccumulator
from tests.test_character_observations import db, seed, baseline, observation
from tests.test_habit_scene_policy import local_run
from tests.test_selected_claim_resolution_policy import POLICIES
from tests.test_habit_action_selection_policy import _RecordingProvider


def _settings(**changes):
    base, _ = local_run()
    base.character_explanation_review_v1 = True
    run = AnalysisRunRow(sensitivity="balanced", batch_mode="draft_review", provider_config_id="account-config", requested_by_user_id="author", batch_coverage={
        "character_review_enabled": True, "character_baseline_run_id": "baseline", "habit_scene_policy": "habit-scene-windows-v1",
        **POLICIES, "target_focus_policy": CHARACTER_TARGET_FOCUS_POLICY_V1,
        "explanation_assertions_policy": CHARACTER_EXPLANATION_ASSERTIONS_POLICY_V1, **changes,
    })
    return base, _character_stage_settings_for_run(base, run)


@pytest.mark.parametrize("missing", [None, "explanation_assertions_policy", "target_focus_policy", "character_review_enabled"])
def test_fresh_explanation_policy_needs_authorized_deep_review_and_old_retry_stays_old(missing):
    base, local = _settings(**({missing: None} if missing else {}))
    assert local._character_explanation_assertions_policy == (None if missing else CHARACTER_EXPLANATION_ASSERTIONS_POLICY_V1)
    assert base._character_explanation_assertions_policy is None
    assert "_character_explanation_assertions_policy" not in local.model_dump()
    old = AnalysisRunRow(sensitivity="balanced", batch_coverage={})
    assert _character_stage_settings_for_run(local, old)._character_explanation_assertions_policy is None


def test_explanation_exact_versions_use_original_budget_and_runtime_reports_actual_protocol():
    _, current = _settings()
    _, old = _settings(explanation_assertions_policy=None)
    transport, usage = _RecordingProvider(), CharacterConsistencyUsageAccumulator()
    def wrapped(settings):
        return _CharacterConsistencyAccountingProvider(settings, usage, signal_provider=transport, drift_provider=transport)
    for settings, prompt in ((old, EXPLANATION_REVIEW_SYSTEM_PROMPT_V3), (current, EXPLANATION_REVIEW_SYSTEM_PROMPT),
                             (current, EXPLANATION_REVIEW_SYSTEM_PROMPT_V3 + "x")):
        with pytest.raises(RuntimeError, match="unsupported"):
            wrapped(settings).complete(prompt, "{}")
    assert not transport.sent
    for settings, prompt, version in ((old, EXPLANATION_REVIEW_SYSTEM_PROMPT, "v2"), (current, EXPLANATION_REVIEW_SYSTEM_PROMPT_V3, "v3")):
        wrapped(settings).complete(prompt, "{}")
        from tests.test_runtime_provenance import configured_settings
        provenance_settings = configured_settings(character_explanation_review_v1=True)
        provenance_settings._character_explanation_assertions_policy = settings._character_explanation_assertions_policy
        provenance = safe_runtime_provenance(provenance_settings)
        limits = provenance["character_consistency_limits"]
        assert limits["explanation_review_schema_version"] == f"character-explanation-review-{version}"
        assert limits["explanation_review_prompt_version"] == f"character-explanation-review-prompt-{version}"
        from scripts.run_evidence_investigator_live import _safe_runtime_provenance
        assert _safe_runtime_provenance(provenance) is not None
        changed = deepcopy(provenance)
        changed["character_consistency_limits"]["explanation_review_prompt_version"] = "character-explanation-review-prompt-v1"
        assert _safe_runtime_provenance(changed) is None
    assert usage.logical_calls == usage.successful_calls == 2


def _source_receipt(db):
    project, run, source = seed(db)
    run.batch_coverage = {"explanation_assertions_policy": CHARACTER_EXPLANATION_ASSERTIONS_POLICY_V1}
    source.content = "医生已要求暂时禁食，林舟按要求推开餐盘。"
    source.content_sha256 = text_sha256(source.content)
    frozen, _ = baseline(db, project, run)
    evidence = EvidenceSpan(document_id=source.document_id, document_name=source.document_name, line_start=1, line_end=1, text=source.content)
    item = observation(source, evidence=evidence, matches=(ObservationMatchDecision(baseline_input_id=frozen.id, status="matched", reason_code="matched", identity_match=True, match_method="comparison_identity"),))
    CharacterObservationRecorder().record(db, run_id=run.id, project_id=project.id, observations=[item])
    target = ExplanationBaselineSummary(character="林舟", dimension="preference", trait_key="进食选择", statement="林舟通常正常进食")
    binding = make_assertion_binding(candidate=SimpleNamespace(citation="E01", evidence=evidence),
        observations={"C01": SimpleNamespace(observation_id=item.observation_id, evidence=evidence)},
        links=[ExplanationAssertionLink(observation_citation="C01", independence="independent_fact",
            explanation=AssertionSelection(assertion_id="A1", quote="医生已要求暂时禁食"),
            observation=AssertionSelection(assertion_id="A1", quote="林舟按要求推开餐盘"))],
        request_sha256="1" * 64, baseline_sha256=payload_sha256(target.model_dump(mode="json")))
    payload = assertion_receipt_payload([{"case_id": "case", "confirmed_candidate_id": frozen.candidate_id,
        "baseline_payload_sha256": frozen.payload_sha256, "support_id": "support", "kind": "exception", "binding": binding}])
    return run, source, payload


@pytest.mark.parametrize("corruption", [None, "source", "observation", "baseline", "overlap", "policy"])
def test_private_receipt_requires_real_frozen_source_target_and_admitted_observation(db, corruption):
    run, source, payload = _source_receipt(db)
    item = payload["receipts"][0]
    link = item["binding"]["links"][0]
    if corruption == "source":
        source.content = "不同版本正文。"
        source.content_sha256 = text_sha256(source.content)
    elif corruption == "observation":
        link["observation_id"] = "cs_missing"
    elif corruption == "baseline":
        item["binding"]["baseline_sha256"] = "0" * 64
    elif corruption == "overlap":
        link["explanation"] = deepcopy(link["observation"])
    elif corruption == "policy":
        run.batch_coverage = {}
    item["binding"]["payload_sha256"] = payload_sha256({k: v for k, v in item["binding"].items() if k != "payload_sha256"})
    payload["payload_sha256"] = payload_sha256({k: v for k, v in payload.items() if k != "payload_sha256"})
    if corruption:
        with pytest.raises(ValueError, match="integrity mismatch"):
            validate_explanation_assertion_payload(db, run=run, payload=payload)
    else:
        assert validate_explanation_assertion_payload(db, run=run, payload=payload) == payload
        assert validate_explanation_assertion_payload(db, run=run, payload=None) is None
        public = _public_diagnostics_payload({"character_consistency": {EXPLANATION_ASSERTION_REFS_KEY: payload, "outcome": "completed"}})
        assert public == {"character_consistency": {"outcome": "completed"}}


def test_bad_assertion_receipt_fails_character_stage_without_retaining_formal_issue():
    from app.config import Settings
    from app.character_consistency_stage import failed_character_consistency_stage
    from app.service import execute_analysis
    from tests.test_character_observation_pipeline import _new_review
    from tests.test_character_consistency_stage import _context, _create_document
    write_limiter.events.clear()
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "Bad explanation receipt"}).json()
        document = _create_document(client, project["id"], name="draft.md", role="chapter", content="冻结正文。", narrative_context=_context(publication="draft"))
        run_id = _new_review(client, project["id"], document["id"])
        span = EvidenceSpan(document_id=document["id"], document_name="draft.md", line_start=1, line_end=1, text="冻结正文。")
        issue = ConsistencyIssue(category="character_drift", severity="high", confidence=.9, title="必须被丢弃的结果", explanation="坏收据不能保留正式结论。", suggestion="核对。", evidence=[span])
        result = failed_character_consistency_stage().model_copy(update={"issues": (issue,), "diagnostics": {EXPLANATION_ASSERTION_REFS_KEY: {"raw": "must-not-leak"}}})
        settings = Settings(_env_file=None, enable_model_extraction=False, enable_character_consistency=True)
        runtime = SimpleNamespace(settings=settings, identity="server_default", available=True, unavailable_reason=None)
        with patch("app.service.runtime_for_analysis_run", return_value=runtime), patch("app.service.CharacterConsistencyStage") as stage:
            stage.return_value.run.return_value = result
            stage.return_value.observation_recording_failed = False
            execute_analysis(run_id, raise_on_failure=True)
        with SessionLocal() as session:
            assert not session.scalars(select(IssueRow).where(IssueRow.run_id == run_id)).all()
            diagnostic = session.get(AnalysisDiagnosticRow, run_id).payload["character_consistency"]
            assert diagnostic["outcome"] == "degraded"
            assert diagnostic["failure_diagnostic_v1"]["stage"] == "character_explanation_assertion_validation"
            assert EXPLANATION_ASSERTION_REFS_KEY not in diagnostic
