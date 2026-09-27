"""A scoped author axis is reviewable only with object/situation evidence."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from app.character_drift import (
    CharacterConsistencyReviewer,
    CharacterDriftCase,
    CharacterReviewDiagnostics,
    CharacterReviewResult,
    ConfirmedTraitSnapshot,
    ModelDriftDecision,
    _validate_decision,
    prepare_character_drift,
    promote_character_drift,
)
from app.character_trait_extraction import CharacterSignal
from app.config import Settings
from app.domain import EvidenceSpan


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _case(
    *, coverage: str = "complete", dimension: str = "value", count: int = 2,
) -> CharacterDriftCase:
    definition = "家庭危机时是否保护家人"
    scope = "家庭危机时"
    proposition = "林澈在家庭危机时保护家人"
    baseline = ConfirmedTraitSnapshot(
        id="ct_scoped-review", character="林澈", dimension=dimension,
        trait_key="保护家人", statement=proposition,
        polarity="positive", stability="stable", origin="explicit_setting",
        evidence=(EvidenceSpan(
            document_id="profile", document_name="profile.md",
            line_start=1, line_end=1, text=proposition + "。",
        ),),
        approved_axis_id="00000000-0000-4000-8000-000000000001",
        approved_axis_version=1, approved_axis_display_name="家庭保护",
        approved_axis_definition=definition,
        approved_axis_definition_sha256=_sha(definition),
        approved_axis_comparison_key=f"{dimension}:家人",
        approved_axis_applicability_scope=scope,
        approved_axis_applicability_scope_sha256=_sha(scope),
        axis_positive_proposition=proposition,
        axis_positive_proposition_sha256=_sha(proposition),
        axis_alignment="same", axis_polarity="positive",
    )
    observations = tuple(
        CharacterSignal(
            id=f"cs_{index:032x}", character="林澈", dimension=dimension,
            trait_key="保护家人", statement="林澈在家庭危机时拒绝保护家人",
            polarity="negative", stability="situational",
            observation_kind="action", key_object="家人", source_kind="draft",
            evidence=EvidenceSpan(
                document_id="draft", document_name="draft.md",
                line_start=index, line_end=index,
                text="家庭危机时，林澈拒绝保护家人。",
            ),
        )
        for index in range(1, count + 1)
    )
    return CharacterDriftCase(
        id="cdc_scoped-review", baseline=baseline,
        observations=observations, scope_compatibility="compatible",
        material_coverage=coverage,
        approved_axis_bound_observation_ids=tuple(row.id for row in observations),
        approved_axis_observation_polarities=tuple(
            (row.id, "negative") for row in observations
        ),
    )


def _decision(
    object_match: str, situation_match: str,
    *, count: int = 2, independent_events: str | None = None,
) -> ModelDriftDecision:
    return ModelDriftDecision.model_validate({
        "verdict": "contradicts", "explanation": "两次行为都与已确认设定相反。",
        "citations": ("B01",) + tuple(f"C{index:02d}" for index in range(1, count + 1)),
        "scope_applicability": {
            "observations": tuple(
                {
                    "citation": f"C{index:02d}",
                    "object_match": object_match,
                    "situation_match": situation_match,
                }
                for index in range(1, count + 1)
            ),
            "independent_events": independent_events or (
                "yes" if count > 1 else "not_applicable"
            ),
        },
    })


def _review(decision: ModelDriftDecision) -> CharacterReviewResult:
    return CharacterReviewResult(
        decision=decision,
        diagnostics=CharacterReviewDiagnostics(
            outcome="completed", reason="completed", attempted_calls=1,
        ),
    )


@pytest.mark.parametrize("dimension", ("value", "behavior_boundary"))
def test_scoped_axis_two_independent_actions_can_reach_reviewer_and_conflict(
    dimension: str,
):
    prepared = prepare_character_drift(_case(dimension=dimension))
    assert prepared.reviewer_eligible
    assert prepared.candidate_level == "strong"
    result = promote_character_drift(prepared, _review(_decision("same", "same")))
    assert result.outcome == "conflict"
    assert result.reason == "model_contradicts"


@pytest.mark.parametrize("object_match,situation_match,expected,reason", [
    ("different", "same", "no_issue", "scoped_axis_outside_applicability"),
    ("same", "different", "no_issue", "scoped_axis_outside_applicability"),
    ("unclear", "same", "unverifiable", "scoped_axis_applicability_unclear"),
    ("same", "unclear", "unverifiable", "scoped_axis_applicability_unclear"),
])
def test_scoped_axis_never_promotes_without_both_applicability_parts(
    object_match: str, situation_match: str, expected: str, reason: str,
):
    result = promote_character_drift(
        prepare_character_drift(_case()),
        _review(_decision(object_match, situation_match)),
    )
    assert result.outcome == expected
    assert result.visible is False
    assert result.reason == reason


def test_scoped_axis_cannot_combine_object_from_one_action_with_scope_from_another():
    decision = _decision("same", "same")
    first, second = decision.scope_applicability.observations
    decision = decision.model_copy(update={
        "scope_applicability": decision.scope_applicability.model_copy(update={
            "observations": (
                first.model_copy(update={"situation_match": "different"}),
                second.model_copy(update={"object_match": "different"}),
            )
        })
    })
    _validate_decision(
        decision, frozenset({"B01", "C01", "C02"}),
        scoped_axis=True, current_count=2,
    )
    result = promote_character_drift(prepare_character_drift(_case()), _review(decision))
    assert result.outcome == "no_issue"
    assert result.visible is False


@pytest.mark.parametrize("independence", ("no", "unclear", "not_applicable"))
def test_scoped_axis_repeated_or_unproven_events_cannot_be_conflict(independence):
    result = promote_character_drift(
        prepare_character_drift(_case()),
        _review(_decision("same", "same", independent_events=independence)),
    )
    assert result.outcome == "unverifiable"
    assert result.visible is False
    assert result.reason == "scoped_axis_event_independence_unproven"


def test_scoped_axis_single_action_reviews_applicability_but_never_conflicts():
    prepared = prepare_character_drift(_case(count=1))
    assert prepared.reason == "single_behavior_is_not_drift"
    assert prepared.reviewer_eligible
    same = ModelDriftDecision.model_validate({
        "verdict": "contradicts", "explanation": "行为与设定方向相反。",
        "citations": ("B01", "C01"),
        "scope_applicability": {
            "observations": ({
                "citation": "C01", "object_match": "same",
                "situation_match": "same",
            },),
            "independent_events": "not_applicable",
        },
    })
    in_scope = promote_character_drift(prepared, _review(same))
    assert in_scope.outcome == "needs_confirmation"
    assert in_scope.reason == "single_behavior_is_not_drift"
    wrong_situation = same.model_copy(update={
        "scope_applicability": same.scope_applicability.model_copy(
            update={"observations": (
                same.scope_applicability.observations[0].model_copy(
                    update={"situation_match": "different"}
                ),
            )}
        ),
    })
    outside = promote_character_drift(prepared, _review(wrong_situation))
    assert outside.outcome == "no_issue"
    assert outside.visible is False
    assert outside.reason == "scoped_axis_outside_applicability"


def test_scoped_axis_incomplete_material_still_cannot_be_conflict():
    result = promote_character_drift(
        prepare_character_drift(_case(coverage="partial")),
        _review(_decision("same", "same")),
    )
    assert result.outcome == "needs_confirmation"
    assert result.reason == "material_coverage_incomplete"


def test_scoped_axis_reviewer_requires_every_current_citation():
    allowed = frozenset({"B01", "C01", "C02"})
    valid = _decision("same", "same")
    _validate_decision(valid, allowed, scoped_axis=True, current_count=2)
    missing_scope = valid.model_copy(update={
        "scope_applicability": valid.scope_applicability.model_copy(
            update={"observations": valid.scope_applicability.observations[:1]}
        )
    })
    with pytest.raises(ValueError, match="scoped_axis_citation_invalid"):
        _validate_decision(missing_scope, allowed, scoped_axis=True, current_count=2)
    incomplete_pending = missing_scope.model_copy(update={
        "verdict": "needs_confirmation"
    })
    with pytest.raises(ValueError, match="scoped_axis_citation_invalid"):
        _validate_decision(
            incomplete_pending, allowed, scoped_axis=True, current_count=2
        )
    with pytest.raises(ValueError, match="scoped_axis_applicability_required"):
        _validate_decision(
            valid.model_copy(update={"scope_applicability": None}),
            allowed, scoped_axis=True, current_count=2,
        )


class _Provider:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str):
        self.calls.append((system, user))
        return SimpleNamespace(
            text=json.dumps(self.payload, ensure_ascii=False),
            prompt_tokens=10, completion_tokens=10,
        )


def test_scoped_axis_reviewer_sends_author_scope_and_rejects_missing_gate():
    prepared = prepare_character_drift(_case())
    provider = _Provider({
        "verdict": "contradicts", "explanation": "表面相反。",
        "citations": ["B01", "C01", "C02"],
    })
    settings = Settings(
        _env_file=None, openai_api_key="test-key",
        openai_base_url="https://mock.invalid/v1",
        openai_model="mock", enable_character_consistency=True,
    )
    review = CharacterConsistencyReviewer(provider, settings=settings).review(prepared)
    assert review.decision is None
    assert review.diagnostics.reason == "invalid_model_response"
    system, user = provider.calls[0]
    assert "scope_applicability" in system
    parsed = json.loads(user)
    assert parsed["candidate"]["approved_axis_comparison_key"] == "value:家人"
    assert parsed["candidate"]["approved_axis_applicability_scope"] == "家庭危机时"


def test_scoped_axis_reviewer_accepts_complete_bounded_scope_assessment():
    provider = _Provider({
        "verdict": "contradicts", "explanation": "两次行为与已确认设定相反。",
        "citations": ["B01", "C01", "C02"],
        "scope_applicability": {
            "observations": [
                {"citation": "C01", "object_match": "same", "situation_match": "same"},
                {"citation": "C02", "object_match": "same", "situation_match": "same"},
            ],
            "independent_events": "yes",
        },
    })
    settings = Settings(
        _env_file=None, openai_api_key="test-key",
        openai_base_url="https://mock.invalid/v1", openai_model="mock",
        enable_character_consistency=True,
    )
    prepared = prepare_character_drift(_case())
    review = CharacterConsistencyReviewer(provider, settings=settings).review(prepared)
    assert review.diagnostics.outcome == "completed"
    assert review.decision is not None
    assert promote_character_drift(prepared, review).outcome == "conflict"
