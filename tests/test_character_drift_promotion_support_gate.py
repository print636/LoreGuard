from __future__ import annotations

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
    SupportEvidence,
    _evidence_rows,
    _validate_decision,
    prepare_character_drift,
    promote_character_drift,
)
from app.character_trait_extraction import CharacterSignal
from app.config import Settings
from app.domain import EvidenceSpan


def _span(document_id: str, line: int, text: str) -> EvidenceSpan:
    return EvidenceSpan(
        document_id=document_id,
        document_name=f"{document_id}.md",
        line_start=line,
        line_end=line,
        text=text,
    )


def _support(
    kind: str,
    *,
    semantic: bool = False,
    observation_ids: tuple[str, ...] | None = None,
) -> SupportEvidence:
    return SupportEvidence(
        id=f"se_{kind}",
        kind=kind,
        summary="与当前性格变化可能相关的既有事件",
        explicit=kind != "possible_explanation",
        evidence=_span("history", 7, "旧稿记录了角色态度发生变化的相关事件。"),
        source_kind="published_history",
        publication_status="published",
        authority_tier="formal_record",
        resolution_state="confirmed",
        source_ordinal=7,
        eligible_draft_document_ids=("draft",),
        selection_basis=(
            "semantic_relation_v1" if semantic else "deterministic_explicit_v1"
        ),
        applicable_observation_ids=(
            observation_ids
            if observation_ids is not None
            else (("cs_" + "a" * 32,) if semantic else ())
        ),
    )


def _prepared(*supports: SupportEvidence, two_observations: bool = False):
    baseline = ConfirmedTraitSnapshot(
        id="ct_promotion_gate",
        character="林澈",
        dimension="core_personality",
        trait_key="avoids_public_disputes",
        statement="林澈一向避免公开争执",
        polarity="positive",
        stability="core",
        origin="explicit_setting",
        evidence=(_span("profile", 2, "林澈一向避免公开争执。"),),
    )
    observation = CharacterSignal(
        id="cs_" + "a" * 32,
        character="林澈",
        dimension="core_personality",
        trait_key="avoids_public_disputes",
        statement="林澈声明自己今后会主动公开争执",
        polarity="negative",
        stability="situational",
        observation_kind="explicit_declaration",
        source_kind="draft",
        evidence=_span("draft", 12, "林澈说：以后我会主动在众人面前争执。"),
    )
    observations = (observation,)
    if two_observations:
        observations += (
            observation.model_copy(
                update={
                    "id": "cs_" + "b" * 32,
                    "statement": "林澈次日再次公开挑起争执",
                    "evidence": _span(
                        "draft", 20, "次日，林澈再次在众人面前挑起争执。"
                    ),
                }
            ),
        )
    prepared = prepare_character_drift(
        CharacterDriftCase(
            id="cdc_promotion_gate",
            baseline=baseline,
            observations=observations,
            support_evidence=supports,
            scope_compatibility="compatible",
            material_coverage="complete",
            explanation_coverage="complete",
        )
    )
    assert prepared.reason == "explicit_opposition"
    return prepared


def _review(verdict: str, *citations: str) -> CharacterReviewResult:
    return CharacterReviewResult(
        decision=ModelDriftDecision(
            verdict=verdict,
            explanation="最终语义审查的模型解释。",
            citations=citations,
        ),
        diagnostics=CharacterReviewDiagnostics(
            outcome="completed",
            reason="completed",
        ),
    )


class _Provider:
    def __init__(self, payload: dict[str, object]):
        self.payload = payload

    def complete(self, _system: str, _user: str):
        return SimpleNamespace(
            text=json.dumps(self.payload, ensure_ascii=False),
            prompt_tokens=7,
            completion_tokens=5,
        )


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="unit-test-placeholder",
        openai_base_url="https://mock.invalid/v1",
        openai_model="mock-model",
        enable_character_consistency=True,
        provider_max_attempts=1,
    )


def test_possible_explanation_ignored_by_contradiction_stays_visible_pending() -> None:
    result = promote_character_drift(
        _prepared(_support("possible_explanation", semantic=True)),
        _review("contradicts", "B01", "C01"),
    )

    assert result.outcome == "needs_confirmation"
    assert result.visible is True
    assert result.reason == "possible_explanation_unresolved"
    assert result.explanation == (
        "已发现可能相关的解释线索，但最终语义审查仍判定为冲突，且未能消解该线索；"
        "在确认其与当前变化的关系前，只能作为待复核线索。"
    )


@pytest.mark.parametrize("kind", ("causal_bridge", "exception"))
def test_semantic_definitive_support_disagrees_with_final_contradiction(
    kind: str,
) -> None:
    result = promote_character_drift(
        _prepared(_support(kind, semantic=True)),
        _review("contradicts", "B01", "C01"),
    )

    assert result.outcome == "needs_confirmation"
    assert result.visible is True
    assert result.reason == "semantic_explanation_review_disagreement"
    assert result.explanation == (
        "解释证据复核已将相关材料认定为明确的成长、因果或例外依据，"
        "但最终语义审查仍判定为冲突；两次语义复核结论不一致，需要人工确认。"
    )


def test_explained_with_definitive_support_remains_no_issue() -> None:
    result = promote_character_drift(
        _prepared(
            _support("possible_explanation", semantic=True),
            _support("causal_bridge", semantic=True).model_copy(
                update={"id": "se_definitive_growth"}
            ),
        ),
        _review("explained", "B01", "C01", "G01"),
    )

    assert result.outcome == "no_issue"
    assert result.visible is False
    assert result.reason == "model_explained"
    assert result.explanation == "最终语义审查的模型解释。"


def test_explained_cannot_use_one_bridge_for_an_entire_multi_observation_case() -> None:
    result = promote_character_drift(
        _prepared(
            _support("causal_bridge", semantic=True),
            two_observations=True,
        ),
        _review("explained", "B01", "C01", "C02", "G01"),
    )

    assert result.outcome == "needs_confirmation"
    assert result.visible is True
    assert result.reason == "explanation_observation_coverage_incomplete"


def test_reviewer_rejects_explained_when_bound_support_misses_one_current() -> None:
    prepared = _prepared(
        _support("causal_bridge", semantic=True),
        two_observations=True,
    )
    review = CharacterConsistencyReviewer(
        _Provider(
            {
                "verdict": "explained",
                "explanation": "既有事件解释了当前表现。",
                "citations": ["B01", "C01", "C02", "G01"],
            }
        ),
        settings=_settings(),
    ).review(prepared)

    assert review.decision is None
    assert review.diagnostics.outcome == "degraded"
    assert review.diagnostics.reason == "invalid_model_response"


def test_multi_event_explained_covers_selected_pair_not_unused_candidate_pool() -> None:
    decision = ModelDriftDecision(
        verdict="explained",
        explanation="所选两次表现均处于同一明确例外。",
        citations=("B01", "C01", "C02", "X01"),
        event_independence="yes",
        independent_event_citations=("C01", "C02"),
    )

    _validate_decision(
        decision,
        frozenset({"B01", "C01", "C02", "C03", "X01"}),
        current_count=3,
        require_independent_events=True,
        support_observation_bindings={
            "X01": frozenset({"C01", "C02"})
        },
    )


def test_legacy_semantic_support_without_binding_loads_but_fails_closed() -> None:
    support = _support(
        "causal_bridge", semantic=True, observation_ids=()
    )
    legacy_payload = support.model_dump()
    legacy_payload.pop("applicable_observation_ids")
    restored = SupportEvidence.model_validate(legacy_payload)

    result = promote_character_drift(
        _prepared(restored),
        _review("explained", "B01", "C01", "G01"),
    )

    assert restored.applicable_observation_ids == ()
    assert result.outcome == "needs_confirmation"
    assert result.reason == "explanation_observation_coverage_incomplete"


def test_stable_support_binding_follows_observation_id_after_reordering() -> None:
    original = _prepared(
        _support("causal_bridge", semantic=True),
        two_observations=True,
    )
    reordered = prepare_character_drift(
        original.case.model_copy(
            update={"observations": tuple(reversed(original.case.observations))}
        )
    )

    rows, _ = _evidence_rows(reordered)
    bridge = next(row for row in rows if row["id"] == "G01")

    assert bridge["applicable_observation_citations"] == ["C02"]


def test_explained_requires_definitive_support_for_each_current_observation() -> None:
    result = promote_character_drift(
        _prepared(
            _support("causal_bridge", semantic=True),
            _support(
                "exception",
                semantic=True,
                observation_ids=("cs_" + "b" * 32,),
            ).model_copy(update={"id": "se_second_exception"}),
            two_observations=True,
        ),
        _review(
            "explained", "B01", "C01", "C02", "G01", "X01"
        ),
    )

    assert result.outcome == "no_issue"
    assert result.visible is False


def test_possible_support_does_not_count_toward_explained_coverage() -> None:
    result = promote_character_drift(
        _prepared(
            _support("causal_bridge", semantic=True),
            _support(
                "possible_explanation",
                semantic=True,
                observation_ids=("cs_" + "b" * 32,),
            ).model_copy(update={"id": "se_second_possible"}),
            two_observations=True,
        ),
        _review("explained", "B01", "C01", "C02", "G01"),
    )

    assert result.outcome == "needs_confirmation"
    assert result.reason == "explanation_observation_coverage_incomplete"


def test_unrelated_possible_support_does_not_override_single_opposition_gate() -> None:
    result = promote_character_drift(
        _prepared(
            _support(
                "possible_explanation",
                semantic=True,
                observation_ids=("cs_" + "b" * 32,),
            ),
            two_observations=True,
        ),
        _review("contradicts", "B01", "C01"),
    )

    assert result.outcome == "needs_confirmation"
    assert result.reason == "single_opposition_not_repeated"


def test_partially_bound_semantic_exception_keeps_multi_c_conflict_pending() -> None:
    result = promote_character_drift(
        _prepared(
            _support(
                "exception",
                semantic=True,
                observation_ids=("cs_" + "b" * 32,),
            ),
            two_observations=True,
        ),
        _review("contradicts", "B01", "C01", "C02"),
    )

    assert result.outcome == "needs_confirmation"
    assert result.reason == "semantic_explanation_review_disagreement"


def test_single_explicit_contradiction_without_support_stays_review_clue() -> None:
    result = promote_character_drift(
        _prepared(),
        _review("contradicts", "B01", "C01"),
    )

    assert result.outcome == "needs_confirmation"
    assert result.visible is True
    assert result.reason == "single_opposition_not_repeated"
    assert "待复核线索" in result.explanation


def test_two_explicit_clauses_cannot_bypass_independent_event_review() -> None:
    prepared = _prepared(two_observations=True)
    assert prepared.reason == "explicit_opposition"

    result = promote_character_drift(
        prepared,
        _review("contradicts", "B01", "C01", "C02"),
    )

    assert result.outcome == "needs_confirmation"
    assert result.reason == "single_opposition_not_repeated"


def test_omitted_explanation_coverage_defaults_to_fail_closed() -> None:
    complete = _prepared().case.model_dump()
    complete.pop("explanation_coverage")
    restored = CharacterDriftCase.model_validate(complete)

    result = promote_character_drift(
        prepare_character_drift(restored),
        _review("contradicts", "B01", "C01"),
    )

    assert restored.explanation_coverage == "not_run"
    assert result.outcome == "needs_confirmation"
    assert result.reason == "explanation_search_incomplete"
