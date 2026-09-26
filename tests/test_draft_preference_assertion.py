"""A model's draft preference label is not proof of an actual self-assertion."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.character_drift import (
    CharacterConsistencyReviewer,
    CharacterDriftCase,
    ConfirmedTraitSnapshot,
    prepare_character_drift,
    promote_character_drift,
)
from app.character_trait_extraction import (
    CharacterSignalChunk,
    CharacterSignalExtractor,
    draft_preference_context_is_relevant,
    draft_preference_context_requires_review,
)
from app.config import Settings
from app.domain import EvidenceSpan


class _Provider:
    def __init__(self, payload: str):
        self.payload = payload
        self.calls = 0

    def complete(self, system: str, user: str):
        self.calls += 1
        return SimpleNamespace(text=self.payload, prompt_tokens=10, completion_tokens=5)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="unit-test-placeholder",
        openai_base_url="https://mock.invalid/v1",
        openai_model="mock-model",
        enable_character_consistency=True,
        provider_max_attempts=1,
    )


def _extract(
    source: str, *, statement: str = "林澈讨厌蜜瓜", polarity: str = "negative"
):
    record = {
        "character": "林澈",
        "dimension": "preference",
        "trait_key": "食物偏好:蜜瓜",
        "statement": statement,
        "polarity": polarity,
        "stability": "temporary",
        "observation_kind": "preference_expression",
        "context": "",
        "key_object": "蜜瓜",
        "source_line_start": 10,
        "source_line_end": 10,
        "evidence": source,
    }
    provider = _Provider(json.dumps({"records": [record]}, ensure_ascii=False))
    result = CharacterSignalExtractor(provider, settings=_settings()).extract(
        CharacterSignalChunk("draft", "draft.md", source, 10, "draft")
    )
    return result, provider


@pytest.mark.parametrize(
    "source",
    [
        "林澈说：「如果我讨厌蜜瓜，这一盘就给你。」",
        "林澈在排练中念道：「我讨厌蜜瓜。」",
        "林澈听周尧说：「我讨厌蜜瓜。」",
        "林澈说：「周尧说：『我讨厌蜜瓜。』」",
        "林澈转述周尧的话：「我讨厌蜜瓜。」",
        "林澈说：「我听周尧说过『我讨厌蜜瓜』。」",
        "林澈假装说：「我讨厌蜜瓜。」",
        "林澈说：「我讨厌蜜瓜？不，我喜欢蜜瓜。」",
        "林澈说：「我讨厌蜜瓜，不，我其实喜欢它。」",
        "林澈说：「我讨厌蜜瓜？周尧才会！」",
        "林澈说：「我讨厌蜜瓜，我怎么可能？」",
        "林澈说：「我讨厌蜜瓜，不对，我是说我不讨厌它。」",
        "林澈说：「我讨厌蜜瓜，其实没有。」",
        "林澈说：「我讨厌蜜瓜，开玩笑的。」",
        "如果明天下雨，林澈说：「我讨厌蜜瓜。」",
        "假如周尧走了，林澈说：「我讨厌蜜瓜。」",
        "林澈说：「我讨厌蜜瓜。」",
        "林澈转述自己刚才说过的话：「我讨厌蜜瓜。」",
        "林澈说：「我讨厌蜜瓜。」但那只是排练。",
        "林澈说：「我讨厌蜜瓜。」其实这是他背的台词。",
        "林澈说：「我讨厌蜜瓜。」那是周尧的台词。",
        "如果周尧讨厌苹果，他会拒食；林澈说：「我讨厌蜜瓜。」",
        "林澈说：「我喜欢葡萄，但我讨厌蜜瓜。」",
        "林澈说：「桌上的灯坏了，我讨厌蜜瓜。」",
        "林澈说：「我讨厌蜜瓜，但我喜欢葡萄。」",
        "林澈说：「我讨厌蜜瓜，因为它太甜。」",
        "林澈说：“我一直最讨厌蜜瓜，闻到味道就想离开。”",
        "林澈说我讨厌蜜瓜，但那只是排练。",
        "林澈说我讨厌蜜瓜。其实只是他梦中的一句话。",
    ],
)
def test_draft_model_preference_cannot_promote_nonactual_or_other_voice(source: str):
    result, provider = _extract(source)

    assert result.signals == ()
    assert result.diagnostics.outcome == "degraded"
    assert sum(result.diagnostics.reason_counts.values()) == 2
    assert set(result.diagnostics.reason_counts) <= {"statement_support", "character_support"}
    assert provider.calls == 2


@pytest.mark.parametrize(
    "source",
    [
        "林澈坦言自己讨厌蜜瓜。",
        "林澈转述自己刚才说过讨厌蜜瓜。",
        "林澈转述自己刚才说过的话：我讨厌蜜瓜。",
        "林澈说我讨厌蜜瓜。",
        "林澈讨厌蜜瓜。",
    ],
)
def test_draft_actual_self_assertion_survives_to_review(source: str):
    result, _ = _extract(source)

    assert result.diagnostics.outcome == "completed"
    assert len(result.signals) == 1
    assert result.signals[0].observation_kind == "preference_expression"

    baseline = ConfirmedTraitSnapshot(
        id="ct_melon",
        character="林澈",
        dimension="preference",
        trait_key="食物偏好:蜜瓜",
        statement="林澈一直喜欢蜜瓜",
        polarity="positive",
        stability="stable",
        origin="explicit_setting",
        evidence=(EvidenceSpan(
            document_id="profile", document_name="profile.md",
            line_start=1, line_end=1, text="林澈一直喜欢蜜瓜",
        ),),
    )
    case = CharacterDriftCase(
        id="cdc_melon",
        baseline=baseline,
        observations=result.signals,
        scope_compatibility="compatible",
        material_coverage="complete",
    )
    prepared = prepare_character_drift(case)
    assert prepared.reviewer_eligible
    reviewer = CharacterConsistencyReviewer(
        _Provider(json.dumps({
            "verdict": "contradicts",
            "explanation": "已确认偏好和当前本人表述相反。",
            "citations": ["B01", "C01"],
        }, ensure_ascii=False)),
        settings=_settings(),
    )
    review = reviewer.review(prepared)
    assert promote_character_drift(prepared, review).outcome == "conflict"


def test_same_line_other_speech_does_not_poison_later_own_assertion():
    source = (
        "林澈听周尧说：「我讨厌蜜瓜。」随后林澈说：「我也讨厌蜜瓜。」"
    )
    positive, _ = _extract(source, statement="林澈说我也讨厌蜜瓜")
    negative, _ = _extract(source, statement="林澈听周尧说我讨厌蜜瓜")

    # This release deliberately abstains even on the later quoted self-claim.
    assert positive.signals == ()
    assert negative.signals == ()


def test_unrelated_prior_condition_does_not_certify_quoted_claim():
    source = "林澈说：「如果明天有船，我就走；我仍讨厌蜜瓜。 」"
    result, _ = _extract(source, statement="林澈说我仍讨厌蜜瓜")

    assert result.signals == ()


def test_condition_governing_preference_is_not_actual_claim():
    source = "林澈说：「如果明天有船，我就讨厌蜜瓜。 」"
    result, _ = _extract(source, statement="林澈讨厌蜜瓜")

    assert result.signals == ()


def test_same_quote_repeated_object_predicates_without_span_identity_abstains():
    source = "林澈说：「我讨厌蜜瓜，后来我仍讨厌蜜瓜。」"
    result, _ = _extract(source, statement="林澈说我讨厌蜜瓜")

    assert result.signals == ()


def test_corrected_question_never_makes_earlier_negative_a_fact():
    source = "林澈说：「我讨厌蜜瓜？不，我喜欢蜜瓜。」"
    negative, _ = _extract(source)
    positive, _ = _extract(
        source, statement="林澈说我喜欢蜜瓜", polarity="positive"
    )

    assert negative.signals == ()
    # The older polarity binder cannot yet distinguish two opposite clauses
    # inside one quotation. It abstains even on the later positive assertion.
    assert positive.signals == ()


def test_preference_coverage_classifier_separates_relevance_from_proof():
    for source, complex_context in (
        ("林澈讨厌蜜瓜。", False),
        ("林澈说：「我讨厌蜜瓜。」但那只是排练。", True),
        ("如果明天下雨，林澈说我讨厌蜜瓜。", True),
        ("林澈梦见自己讨厌蜜瓜。", True),
    ):
        assert draft_preference_context_is_relevant(
            source, character="林澈", key_object="蜜瓜"
        )
        assert draft_preference_context_requires_review(
            source, character="林澈", key_object="蜜瓜"
        ) is complex_context
    assert not draft_preference_context_is_relevant(
        "林澈讨厌蜜瓜味糖。", character="林澈", key_object="蜜瓜"
    )
