from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.character_trait_extraction import (
    CharacterSignalChunk,
    CharacterSignalExtractor,
    _directional_trait_key,
)
from app.config import Settings


class _SequenceProvider:
    def __init__(self, *payloads: dict):
        self.payloads = list(payloads)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> SimpleNamespace:
        self.calls.append((system, user))
        if not self.payloads:
            raise AssertionError("unexpected model call")
        return SimpleNamespace(
            text=json.dumps(self.payloads.pop(0), ensure_ascii=False),
            prompt_tokens=17,
            completion_tokens=9,
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


def _record(**overrides: str) -> dict:
    record = {
        "character": "林澈",
        "dimension": "preference",
        "trait_key": "melon_preference",
        "statement": "林澈一直喜欢蜜瓜",
        "polarity": "positive",
        "stability": "stable",
        "observation_kind": "preference_expression",
        "context": "饮食",
        "key_object": "蜜瓜",
        "source_line_start": 10,
        "source_line_end": 10,
        "evidence": "林澈一直喜欢蜜瓜。",
    }
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    "key",
    (
        "impromptu_explanation_avoidance",
        "publicReprimandAvoidance",
        "social_anxieties",
        "melon_dislikes",
        "risk_refusal",
        "public_hates_speech",
        "food_liking",
    ),
)
def test_directional_english_trait_key_segments_are_rejected(key: str) -> None:
    assert _directional_trait_key(key)


@pytest.mark.parametrize(
    "key",
    (
        "public_rebuke_restraint",
        "public_speaking_participation",
        "route_decision_collaboration",
        "melon_preference",
        "interruption_restraint",
        "社交主动性",
    ),
)
def test_neutral_and_existing_trait_keys_remain_eligible(key: str) -> None:
    assert not _directional_trait_key(key)


def test_directional_trait_key_requires_clean_regeneration_without_raw_leak() -> None:
    valid = _record()
    invalid = _record(
        trait_key="private_melon_dislike_axis",
        context="private-model-context",
    )
    provider = _SequenceProvider(
        {"records": [valid, invalid]},
        {"records": [valid]},
    )

    result = CharacterSignalExtractor(provider, settings=_settings()).extract(
        CharacterSignalChunk(
            "profile-1", "profile.md", "林澈一直喜欢蜜瓜。", 10,
            "formal_character_profile",
        )
    )

    assert result.diagnostics.outcome == "completed"
    assert result.diagnostics.attempted_calls == 2
    assert result.diagnostics.reason_counts == {
        "regenerated_from_directional_trait_key": 1,
    }
    assert len(result.signals) == 1
    assert result.signals[0].trait_key == "melon_preference"
    retry_prompt = provider.calls[1][1]
    assert '"reason":"directional_trait_key"' in retry_prompt
    assert "中性可比较语义轴" in retry_prompt
    assert '"trait_key":"melon_preference"' in retry_prompt
    assert invalid["trait_key"] not in retry_prompt
    assert invalid["context"] not in retry_prompt


def test_two_directional_packages_fail_closed() -> None:
    invalid = _record(trait_key="public_reprimand_avoidance")
    provider = _SequenceProvider({"records": [invalid]}, {"records": [invalid]})

    result = CharacterSignalExtractor(provider, settings=_settings()).extract(
        CharacterSignalChunk(
            "profile-1", "profile.md", "林澈一直喜欢蜜瓜。", 10,
            "formal_character_profile",
        )
    )

    assert result.diagnostics.outcome == "degraded"
    assert result.diagnostics.attempted_calls == 2
    assert result.diagnostics.reason_counts == {"directional_trait_key": 2}
    assert result.signals == ()
    assert result.pending_candidates == ()
