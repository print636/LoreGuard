"""Coverage recovery requires real receipts and an admitted protocol binding."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import app.character_consistency_stage as stage_module
from app.character_claim_resolution import ClaimPackageState, ClaimResolutionManager
from app.character_drift import CharacterReviewDiagnostics, CharacterReviewResult, ConfirmedTraitSnapshot
from app.character_material_coverage import DraftMaterialGap
from app.character_trait_extraction import CharacterSignal, CharacterSignalTarget, _target_bound_review_target
from app.domain import EvidenceSpan
from app.narrative_context import NarrativeScopeV1
from tests.test_character_claim_resolution import paired, reconcile
from tests.test_character_draft_actor_stage_wiring import (
    _ExtractionRecorder, _FrozenStage, _draft_source, _partial_result, _result, _run, _settings,
)
from tests.test_habit_claim_recovery import HABIT


def ordinary_target():
    return CharacterSignalTarget(
        character="余霁", dimension="behavior_boundary", trait_key="seal_handover",
        comparison_key="behavior_boundary:印章", baseline_polarity="positive",
        requested_polarity="negative", baseline_hint="余霁离岗前会把印章交还值守柜",
    )


def test_no_receipt_does_not_convert_an_ordinary_target_or_clear_its_debt():
    source = _draft_source("余霁将印章放回柜中。")
    target = ordinary_target()
    # The narrower protocol remains strict; this fix does not fabricate its object.
    with pytest.raises(ValidationError):
        _target_bound_review_target(target, target_ordinal=1)
    gap = DraftMaterialGap(reason="record_unverified", claim_digest="f" * 64)
    result = _partial_result().model_copy(update={"material_gaps": (gap,)})
    manager = ClaimResolutionManager()
    manager.capture_targeted(source, None, target, result)
    assert not manager.repairs_targeted(source.input_id, "candidate-1")
    assert result.material_gaps == (gap,) and not manager.receipts and manager.payload() is None


@pytest.mark.parametrize("damage", [None, "source", "candidate", "target", "claim", "unbound", "uncovered", "recall_target"])
def test_targeted_repair_requires_the_scoped_receipt_bound_target_and_every_claim(damage):
    pair = paired(exact=True)
    source, chunk, primary, targeted, *_ = pair
    # A real binder/receipt path supplies the certificate. This package retains
    # the same raw claim as a terminal debt, as a partial targeted pass can do.
    targeted = targeted.model_copy(update={
        "diagnostics": targeted.diagnostics.model_copy(update={"outcome": "partial"}),
        "claim_resolution_state": ClaimPackageState(
            certificates=targeted.claim_resolution_state.certificates,
            gaps=primary.claim_resolution_state.gaps, enumeration_complete=True,
        ),
    })
    pair = (*pair[:3], targeted, *pair[4:])
    manager, _, _, _, _ = reconcile(pair)
    assert len(manager.receipts) == 1
    receipt = manager.receipts[0]
    if damage == "source":
        receipt["source_input_id"] = "other-input"
    elif damage == "candidate":
        receipt["confirmed_candidate_id"] = "other-candidate"
    elif damage == "target":
        receipt["target_sha256"] = "0" * 64
    elif damage == "claim":
        receipt["primary_claim_sha256"] = "0" * 64
    elif damage == "unbound":
        manager.targeted[0] = (source, chunk, HABIT, targeted.model_copy(update={"signals": ()}))
    elif damage == "uncovered":
        targeted.claim_resolution_state.gaps += (DraftMaterialGap(reason="source_unverified"),)
    elif damage == "recall_target":
        manager.targeted[0] = (source, chunk, HABIT.model_copy(update={"baseline_hint": "另一条已确认习惯"}), targeted)
    # Neither an unrelated ordinary target nor the same protocol in a different
    # source is permitted to interrupt or discharge the current target's debt.
    manager.targeted[:0] = [
        (source, chunk, ordinary_target(), _partial_result()),
        (replace(source, input_id="other-input"), chunk, ordinary_target(), _partial_result()),
    ]
    assert manager.repairs_targeted(source.input_id, "candidate-1") is (damage is None)
    assert targeted.diagnostics.outcome == "partial"


@pytest.mark.parametrize("other_package", ["unproved", "unknown", "enumerated_empty"])
def test_a_repaired_package_cannot_hide_another_unbound_package_for_the_same_target(other_package):
    pair = paired(exact=True)
    source, chunk, primary, targeted, *_ = pair
    manager, _, _, _, _ = reconcile(pair)
    repaired = targeted.model_copy(update={
        "diagnostics": targeted.diagnostics.model_copy(update={"outcome": "partial"}),
        "claim_resolution_state": ClaimPackageState(
            gaps=primary.claim_resolution_state.gaps, enumeration_complete=True,
        ),
    })
    other_state = None if other_package == "unknown" else ClaimPackageState(
        gaps=(DraftMaterialGap(reason="record_unverified", claim_digest="e" * 64),)
            if other_package == "unproved" else (),
        enumeration_complete=True,
    )
    verification = _partial_result().model_copy(update={"claim_resolution_state": other_state})
    manager.targeted = [
        (source, chunk, HABIT, repaired),
        (source, chunk, HABIT.model_copy(update={"existing_evidence_ranges": ((1, 1),)}), verification),
    ]
    assert manager.repairs_targeted(source.input_id, "candidate-1") is (other_package == "enumerated_empty")


def test_actual_admitted_object_reaches_stage_case_without_a_resolution_receipt(monkeypatch):
    """Downstream reconstruction of the public actor-self object, not model replay.

    The admitted statement/object and both frozen target meanings are preserved.
    Source IDs are test-local; extraction counters and the unavailable reviewer
    are controlled substitutes for the missing complete original responses.
    """
    content = "乔唯检查值守簿，余霁把档案室印章藏进袖中。"
    source = _draft_source(content)
    signal = CharacterSignal(
        id="cs_d219c72dbcc7b8b05d9d91a6cbd5d05a", character="余霁",
        dimension="behavior_boundary", trait_key="personal_retention",
        statement="余霁把档案室印章藏进袖中。", polarity="positive", stability="temporary",
        observation_kind="action", context="乔唯检查值守簿时", key_object="档案室印章",
        source_kind="draft", evidence=EvidenceSpan(document_id=source.document.id,
            document_name=source.document.name, line_start=1, line_end=1, text=content),
    )
    profile = "余霁从不私藏档案室印章，离岗前会把印章交还值守柜。"
    baselines = []
    for candidate, key, statement, polarity, context, comparison in (
        ("4549f778-8bb1-4907-8d43-2714b180f815", "personal_retention", "余霁从不私藏档案室印章", "negative", "档案室印章保管", "behavior_boundary:档案室印章"),
        ("b6e24ad3-6273-4dc7-ba4a-0145a4959b73", "seal_handover", "余霁离岗前会把印章交还值守柜", "positive", "离岗前的交还准则", "behavior_boundary:印章"),
    ):
        baseline = ConfirmedTraitSnapshot(
            id="ct_" + candidate, character="余霁", dimension="behavior_boundary", trait_key=key,
            statement=statement, polarity=polarity, stability="stable", contexts=(context,),
            origin="explicit_setting", evidence=(EvidenceSpan(document_id="profile-1",
                document_name="profile.md", line_start=1, line_end=1, text=profile),),
        )
        row = SimpleNamespace(id="input-" + candidate, candidate_id=candidate,
            payload={"comparison_key": comparison}, payload_sha256="b" * 64)
        baselines.append((row, baseline, NarrativeScopeV1(), "余霁"))
    pending = _partial_result().model_copy(update={"material_gaps": (
        DraftMaterialGap(reason="record_unverified", claim_digest="f" * 64),
    )})
    recorder = _ExtractionRecorder(primary_result=_result(signal), targeted_results=[pending])
    monkeypatch.setattr(stage_module, "CharacterSignalExtractor", recorder.extractor_type())
    reviewed = []

    class UnavailableReviewer:
        def __init__(self, *_args, **_kwargs):
            pass

        def review(self, prepared):
            reviewed.append(prepared)
            return CharacterReviewResult(diagnostics=CharacterReviewDiagnostics(
                outcome="skipped", reason="fixture_review_unavailable",
            ))

    monkeypatch.setattr(stage_module, "CharacterConsistencyReviewer", UnavailableReviewer)
    config = _settings(character_signal_full_line_prompt_v2=True,
        character_draft_actor_review_v1=True, character_target_bound_draft_review_v3=True,
        character_scoped_axis_drift_v1=True, character_explanation_review_v1=False)
    config._character_claim_resolution_v1 = True
    result = _run(_FrozenStage(source=source, baselines=baselines, settings=config), source)
    assert len(recorder.primary_calls) == len(recorder.targeted_calls) == 1
    assert len(reviewed) == 1 and reviewed[0].matching_observations == (signal,)
    assert result.diagnostics["counts"]["draft_observation_count"] == 1
    assert result.diagnostics["outcome"] == "partial" and not result.issues
    assert not result.diagnostics.get("claim_resolution_v1")
    missing = next(case for case in result.diagnostics["case_trace"] if case["prepare_reason"] == "no_matching_observation")
    assert missing["material_coverage"] == "partial"
    assert "targeted_package_incomplete" in missing["material_coverage_reasons"]
