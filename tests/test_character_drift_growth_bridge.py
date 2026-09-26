from __future__ import annotations

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
    SupportEvidence,
    _axis_growth_surface_match,
    prepare_character_drift,
    promote_character_drift,
)
from app.character_trait_extraction import CharacterSignal
from app.config import Settings
from app.domain import EvidenceSpan


AXIS = "角色面对可能伤害居民的师父命令时，是否敢在当事人和其他居民面前直接提出反对。"
BASELINE = "温弦：核心人格是畏惧与师父在众人面前正面冲突；温弦通常不敢当众反驳师父的命令。"
CURRENT = (
    "温弦在东街听见顾衡要求继续开放已报警的侧门，便在居民和巡检员面前"
    "当场反对顾衡：『北台那次服从已经伤到人，我不能再让居民进入灰潮。』"
    "温弦随即带队让人群退回避难廊。"
)
GROWTH = (
    "事故复盘会上，温弦当着顾衡和居民的面说：「我以前不敢公开反对师父；"
    "这次服从错误命令伤到了人。以后如果命令会把居民送进灰潮，"
    "我会在现场直接提出反对，哪怕是顾衡下的命令。」"
    "顾衡没有把这句话解释成玩笑，也没有要求她收回。"
)


def _span(document_id: str, text: str, line: int) -> EvidenceSpan:
    return EvidenceSpan(
        document_id=document_id,
        document_name=f"{document_id}.md",
        line_start=line,
        line_end=line,
        text=text,
    )


def _case(
    *,
    current_text: str = CURRENT,
    growth_text: str = GROWTH,
    baseline_overrides: dict | None = None,
    observation_overrides: dict | None = None,
    support_overrides: dict | None = None,
    bound: bool = True,
    coverage: str = "complete",
) -> CharacterDriftCase:
    baseline = ConfirmedTraitSnapshot(
        id="ct_growth",
        character="温弦",
        dimension="core_personality",
        trait_key="public_mentor_challenge",
        statement="温弦通常不敢当众反驳师父的命令",
        polarity="negative",
        stability="core",
        origin="explicit_setting",
        evidence=(_span("profile", BASELINE, 9),),
        approved_axis_id="00000000-0000-4000-8000-000000000001",
        approved_axis_version=1,
        approved_axis_display_name="当众质疑师父命令",
        approved_axis_definition=AXIS,
        approved_axis_definition_sha256=hashlib.sha256(
            AXIS.encode("utf-8")
        ).hexdigest(),
        axis_positive_proposition="温弦当众反对可能伤害居民的师父命令",
        axis_positive_proposition_sha256=hashlib.sha256(
            "温弦当众反对可能伤害居民的师父命令".encode("utf-8")
        ).hexdigest(),
        axis_alignment="same",
        axis_polarity="negative",
    )
    if baseline_overrides:
        baseline = baseline.model_copy(update=baseline_overrides)
    observation = CharacterSignal(
        id="cs_" + "a" * 32,
        character="温弦",
        dimension="core_personality",
        trait_key="different_model_trait_key_is_not_the_axis_selector",
        statement="温弦当众反对师父命令",
        polarity="positive",
        stability="situational",
        observation_kind="action",
        key_object="",
        source_kind="draft",
        evidence=_span("draft", current_text, 9),
    )
    if observation_overrides:
        observation = observation.model_copy(update=observation_overrides)
    support_payload = {
        "id": "se_growth",
        "kind": "causal_bridge",
        "summary": "已发生的同角公开复盘",
        "explicit": True,
        "evidence": _span("published_history", growth_text, 11),
        "source_kind": "published_history",
        "publication_status": "published",
        "authority_tier": "formal_record",
        "resolution_state": "confirmed",
        "source_ordinal": 10,
        "eligible_draft_document_ids": ("draft",),
    }
    support_payload.update(support_overrides or {})
    return CharacterDriftCase(
        id="cdc_growth",
        baseline=baseline,
        observations=(observation,),
        support_evidence=(SupportEvidence(**support_payload),),
        scope_compatibility="compatible",
        material_coverage=coverage,
        approved_axis_bound_observation_ids=(observation.id,) if bound else (),
        approved_axis_observation_polarities=(
            ((observation.id, observation.polarity),) if bound else ()
        ),
    )


def _review(verdict: str, citations: tuple[str, ...]) -> CharacterReviewResult:
    return CharacterReviewResult(
        decision=ModelDriftDecision(
            verdict=verdict,
            explanation="该角色已在公开复盘中说明改变立场的事故。",
            citations=citations,
        ),
        diagnostics=CharacterReviewDiagnostics(outcome="completed", reason="completed"),
    )


def test_published_same_axis_growth_allows_only_explanatory_review() -> None:
    prepared = prepare_character_drift(_case())
    assert prepared.reviewer_eligible is True
    assert prepared.candidate_level == "possible"
    assert prepared.reason == "single_published_growth_review_only"
    assert not prepared.deterministic_conflict

    explained = promote_character_drift(
        prepared, _review("explained", ("B01", "C01", "G01"))
    )
    assert explained.outcome == "no_issue"
    assert explained.visible is False
    contradicted = promote_character_drift(
        prepared, _review("contradicts", ("B01", "C01"))
    )
    assert contradicted.outcome == "needs_confirmation"
    assert contradicted.outcome != "conflict"


def test_axis_surface_gate_needs_relation_and_action_not_only_mentor_word() -> None:
    assert _axis_growth_surface_match(AXIS, GROWTH)
    assert _axis_growth_surface_match(
        AXIS, "温弦向居民公开反对师父命令。"
    )
    assert not _axis_growth_surface_match(AXIS, "温弦与师父一起参加了绘图训练。")
    assert not _axis_growth_surface_match(
        AXIS, "师父在旁看着，温弦完成地图绘制训练。"
    )


@pytest.mark.parametrize(
    "change",
    [
        {"bound": False},
        {"coverage": "partial"},
        {"baseline_overrides": {"stability": "stable"}},
        {"observation_overrides": {"observation_kind": "interaction"}},
        {"observation_overrides": {"character": "顾衡"}},
        {"current_text": "如果温弦当众反对顾衡，居民就会先撤走。"},
        {"support_overrides": {"publication_status": None}},
        {"support_overrides": {"explicit": False}},
        {"support_overrides": {"source_kind": None}},
        {"support_overrides": {"authority_tier": None}},
        {"support_overrides": {"resolution_state": None}},
        {"support_overrides": {"source_ordinal": None}},
        {"support_overrides": {"eligible_draft_document_ids": ("other",)}},
        {"support_overrides": {"evidence": _span("draft", GROWTH, 11)}},
        {"growth_text": GROWTH.replace("温弦当着", "叶箫当着")},
        {"growth_text": "如果" + GROWTH},
        {
            "growth_text": (
                "事故复盘会上，温弦当着顾衡和居民的面说："
                "「以后如果师父命令伤害居民，我会直接提出反对。」"
            )
        },
        {"growth_text": "温弦与师父一起参加了绘图训练。"},
    ],
)
def test_unbound_unpublished_other_actor_or_nonactual_growth_fails_closed(
    change: dict,
) -> None:
    prepared = prepare_character_drift(_case(**change))
    assert prepared.reviewer_eligible is False
    assert promote_character_drift(prepared, None).outcome != "no_issue"
    assert promote_character_drift(prepared, None).outcome != "conflict"


def test_approved_axis_is_required_not_model_trait_key() -> None:
    prepared = prepare_character_drift(
        _case(
            baseline_overrides={
                "approved_axis_id": None,
                "approved_axis_version": None,
                "approved_axis_display_name": None,
                "approved_axis_definition": None,
                "approved_axis_definition_sha256": None,
                "axis_positive_proposition": None,
                "axis_positive_proposition_sha256": None,
                "axis_alignment": None,
                "axis_polarity": None,
            },
            bound=False,
        )
    )
    assert prepared.reason == "no_matching_observation"
    assert not prepared.reviewer_eligible

    # Even an identical model-authored trait key cannot replace a server
    # confirmed axis identity/binding for this single-action path.
    same_key = prepare_character_drift(
        _case(
            baseline_overrides={
                "approved_axis_id": None,
                "approved_axis_version": None,
                "approved_axis_display_name": None,
                "approved_axis_definition": None,
                "approved_axis_definition_sha256": None,
                "axis_positive_proposition": None,
                "axis_positive_proposition_sha256": None,
                "axis_alignment": None,
                "axis_polarity": None,
            },
            observation_overrides={"trait_key": "public_mentor_challenge"},
            bound=False,
        )
    )
    assert same_key.reason == "single_behavior_is_not_drift"
    assert not same_key.reviewer_eligible


def test_same_direction_is_not_a_growth_review_candidate() -> None:
    prepared = prepare_character_drift(
        _case(observation_overrides={"polarity": "negative"})
    )
    assert prepared.reason == "no_opposition"
    assert not prepared.reviewer_eligible


def test_opposite_author_axis_mapping_judges_per_case_not_raw_signal() -> None:
    case = _case()
    baseline = ConfirmedTraitSnapshot.model_validate({
        **case.baseline.model_dump(),
        "polarity": "positive",
        "axis_alignment": "opposite",
        "axis_polarity": "negative",
    })
    observation = case.observations[0].model_copy(
        update={"polarity": "negative"}
    )
    mapped = CharacterDriftCase.model_validate({
        **case.model_dump(),
        "baseline": baseline.model_dump(),
        "observations": [observation.model_dump()],
        "approved_axis_observation_polarities": [
            (observation.id, "positive")
        ],
    })
    prepared = prepare_character_drift(mapped)
    assert baseline.polarity == "positive"
    assert observation.polarity == "negative"
    assert prepared.matching_observations == (observation,)
    assert prepared.reviewer_eligible
    assert prepared.reason == "single_published_growth_review_only"
    # A second target may reuse this signal under another author axis. Its
    # direction is scoped to that case; the signal's raw polarity never flips.
    other = mapped.model_copy(update={
        "approved_axis_observation_polarities": ((observation.id, "negative"),)
    })
    assert prepare_character_drift(other).reason == "no_opposition"
    assert observation.polarity == "negative"


def test_legacy_bound_axis_abstains_even_with_model_opposition() -> None:
    case = _case()
    baseline = ConfirmedTraitSnapshot.model_validate({
        **case.baseline.model_dump(),
        "axis_positive_proposition": None,
        "axis_positive_proposition_sha256": None,
        "axis_alignment": "legacy_unverified",
        "axis_polarity": None,
    })
    legacy = case.model_copy(update={
        "baseline": baseline,
        "approved_axis_observation_polarities": (),
    })
    prepared = prepare_character_drift(legacy)
    assert prepared.reason == "author_alignment_required"
    assert not prepared.reviewer_eligible
    assert not promote_character_drift(prepared, _review("contradicts", ("B01", "C01"))).visible


def test_malformed_axis_snapshot_is_not_misclassified_as_legacy() -> None:
    baseline = _case().baseline.model_dump()
    with pytest.raises(ValueError, match="alignment is inconsistent"):
        ConfirmedTraitSnapshot.model_validate({
            **baseline, "axis_alignment": "opposite", "axis_polarity": "negative"
        })
    with pytest.raises(ValueError, match="legacy axis has a directional polarity"):
        ConfirmedTraitSnapshot.model_validate({
            **baseline, "axis_alignment": "legacy_unverified",
            "axis_polarity": "positive",
        })
    with pytest.raises(ValueError, match="proposition is incomplete"):
        ConfirmedTraitSnapshot.model_validate({
            **baseline, "axis_alignment": "legacy_unverified",
            "axis_polarity": None,
            "axis_positive_proposition_sha256": None,
        })


def test_unrelated_g_citation_cannot_replace_matching_g() -> None:
    case = _case()
    unrelated = case.support_evidence[0].model_copy(
        update={
            "id": "se_other",
            "evidence": _span("other_history", "温弦与师父一起参加了绘图训练。", 3),
        }
    )
    prepared = prepare_character_drift(
        case.model_copy(update={"support_evidence": (unrelated, case.support_evidence[0])})
    )
    assert prepared.reviewer_eligible
    assert promote_character_drift(
        prepared, _review("explained", ("B01", "C01", "G01"))
    ).outcome == "unverifiable"
    assert promote_character_drift(
        prepared, _review("explained", ("B01", "C01", "G02"))
    ).outcome == "no_issue"
    assert promote_character_drift(
        prepared, _review("explained", ("B99", "C01", "G02"))
    ).outcome == "unverifiable"


def test_reviewer_rejects_explanation_with_only_unrelated_g() -> None:
    case = _case()
    unrelated = case.support_evidence[0].model_copy(
        update={
            "id": "se_other",
            "evidence": _span("other_history", "温弦与师父一起参加了绘图训练。", 3),
        }
    )
    prepared = prepare_character_drift(
        case.model_copy(update={"support_evidence": (unrelated, case.support_evidence[0])})
    )

    class FakeProvider:
        def complete(self, system: str, user: str) -> SimpleNamespace:
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "verdict": "explained",
                        "explanation": "这个证据解释了当前行为。",
                        "citations": ["B01", "C01", "G01"],
                    },
                    ensure_ascii=False,
                ),
                prompt_tokens=10,
                completion_tokens=10,
            )

    review = CharacterConsistencyReviewer(
        FakeProvider(),
        settings=Settings(
            _env_file=None,
            openai_api_key="test-only",
            openai_model="fake-model",
            enable_character_consistency=True,
        ),
    ).review(prepared)
    assert review.decision is None
    assert review.diagnostics.reason == "invalid_model_response"


def test_no_growth_keeps_ordinary_single_behavior_gate() -> None:
    case = _case()
    prepared = prepare_character_drift(
        case.model_copy(update={"support_evidence": ()})
    )
    assert prepared.reason == "single_behavior_is_not_drift"
    assert not prepared.reviewer_eligible
