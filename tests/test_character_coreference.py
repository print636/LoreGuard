from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.character_trait_extraction import (
    CHARACTER_SIGNAL_SYSTEM_PROMPT,
    TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT,
    CharacterSignalChunk,
    CharacterSignalExtractor,
    CharacterSignalTarget,
    safe_pronoun_evidence_range,
    stable_trait_identity,
)
from app.config import Settings


NANZHI_ANTECEDENT = "庆典组只安排南枝检查展台，接待商队原本由其他队员负责。"
NANZHI_OBSERVATION = (
    "清晨，她却主动走向第一次到站的商队代表，邀请对方坐下，"
    "兴致勃勃地聊起自己童年的远航，直到对方催促才结束谈话。"
)
NANZHI_STATEMENT = NANZHI_OBSERVATION.replace("她", "南枝", 1).rstrip("。")


def settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="unit-test-placeholder",
        openai_base_url="https://mock.invalid/v1",
        openai_model="mock-model",
        enable_character_consistency=True,
        provider_max_attempts=1,
    )


class FakeProvider:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str):
        self.calls.append((system, user))
        return SimpleNamespace(
            text=self.response,
            prompt_tokens=17,
            completion_tokens=9,
        )


def chunk(content: str, *, source_kind: str = "draft") -> CharacterSignalChunk:
    return CharacterSignalChunk(
        "draft-coreference",
        "draft.md",
        content,
        3,
        source_kind,
    )


def record(
    evidence: str,
    *,
    statement: str = NANZHI_STATEMENT,
    start: int = 3,
    end: int = 4,
) -> dict[str, object]:
    return {
        "character": "南枝",
        "dimension": "core_personality",
        "trait_key": "social_initiative",
        "statement": statement,
        "polarity": "positive",
        "stability": "situational",
        "observation_kind": "action",
        "context": "清晨接待陌生商队",
        "key_object": "",
        "source_line_start": start,
        "source_line_end": end,
        "evidence": evidence,
    }


def target() -> CharacterSignalTarget:
    return CharacterSignalTarget(
        character="南枝",
        dimension="core_personality",
        trait_key="social_initiative",
        comparison_key=stable_trait_identity(
            "core_personality",
            "social_initiative",
        ),
        baseline_polarity="negative",
        requested_polarity="positive",
        baseline_hint="长期回避同陌生人主动攀谈",
    )


def response(row: dict[str, object]) -> str:
    return json.dumps({"records": [row]}, ensure_ascii=False)


def test_generic_extraction_accepts_literal_two_line_pronoun_attribution():
    evidence = f"{NANZHI_ANTECEDENT}\n{NANZHI_OBSERVATION}"
    provider = FakeProvider(response(record(evidence)))

    result = CharacterSignalExtractor(provider, settings=settings()).extract(
        chunk(evidence)
    )

    assert result.diagnostics.outcome == "completed"
    assert result.diagnostics.reason_counts == {}
    assert len(result.draft_observations) == 1
    observation = result.draft_observations[0]
    assert observation.character == "南枝"
    assert observation.statement == NANZHI_STATEMENT
    assert (observation.evidence.line_start, observation.evidence.line_end) == (3, 4)
    assert observation.evidence.text == evidence


def test_targeted_candidate_view_accepts_only_the_server_listed_two_line_span():
    evidence = f"{NANZHI_ANTECEDENT}\n{NANZHI_OBSERVATION}"
    source = chunk(evidence)
    provider = FakeProvider(response(record(evidence)))

    assert safe_pronoun_evidence_range(source, "南枝", 3) == (3, 4)
    result = CharacterSignalExtractor(provider, settings=settings()).extract_targeted(
        source,
        (target(),),
        candidate_evidence_ranges=((3, 4),),
    )

    assert result.diagnostics.outcome == "completed"
    assert len(result.signals) == 1
    prompt = provider.calls[0][1]
    assert '"line_start":3,"line_end":4' in prompt
    assert f"3: {NANZHI_ANTECEDENT}" in prompt
    assert f"4: {NANZHI_OBSERVATION}" in prompt


def test_pronoun_candidate_range_is_draft_only():
    evidence = f"{NANZHI_ANTECEDENT}\n{NANZHI_OBSERVATION}"
    assert (
        safe_pronoun_evidence_range(
            chunk(evidence, source_kind="published_history"),
            "南枝",
            3,
        )
        is None
    )


@pytest.mark.parametrize(
    ("evidence", "statement", "end"),
    [
        (
            f"{NANZHI_ANTECEDENT}\n{NANZHI_OBSERVATION}",
            "南枝主动与陌生人长谈",
            4,
        ),
        (
            "庆典组只安排南枝和苏弦检查展台。\n清晨，她却主动与陌生人长谈。",
            "清晨，南枝却主动与陌生人长谈",
            4,
        ),
        (
            "南枝检查展台。\n清晨，她却主动与陌生人长谈。",
            "清晨，南枝却主动与陌生人长谈",
            4,
        ),
        (
            "现场只剩南枝。\n\n清晨，她却主动与陌生人长谈。",
            "清晨，南枝却主动与陌生人长谈",
            5,
        ),
        (
            "现场只剩南枝。\n清晨，“她却主动与陌生人长谈。”",
            "清晨，“南枝却主动与陌生人长谈",
            4,
        ),
        (
            "现场只剩南枝。\n清晨，如果她与陌生人长谈，就会错过集合。",
            "清晨，如果南枝与陌生人长谈，就会错过集合",
            4,
        ),
        (
            "现场只剩南枝。\n清晨，她请他坐下后主动长谈。",
            "清晨，南枝请他坐下后主动长谈",
            4,
        ),
    ],
    ids=(
        "non-literal-summary",
        "multiple-named-characters",
        "no-exclusive-antecedent",
        "blank-line-paragraph",
        "quotation",
        "conditional-branch",
        "multiple-gender-pronouns",
    ),
)
def test_unsafe_pronoun_attribution_is_rejected(
    evidence: str,
    statement: str,
    end: int,
):
    provider = FakeProvider(
        response(record(evidence, statement=statement, end=end))
    )

    result = CharacterSignalExtractor(provider, settings=settings()).extract(
        chunk(evidence)
    )

    assert result.signals == ()
    assert result.diagnostics.reason_counts == {"character_support": 2}


def test_unsafe_two_line_candidate_range_is_rejected_before_provider_call():
    evidence = "南枝检查展台。\n清晨，她却主动与陌生人长谈。"
    provider = FakeProvider(response(record(evidence)))

    result = CharacterSignalExtractor(provider, settings=settings()).extract_targeted(
        chunk(evidence),
        (target(),),
        candidate_evidence_ranges=((3, 4),),
    )

    assert provider.calls == []
    assert result.diagnostics.reason_counts == {
        "targeted_invalid_candidate_ranges": 1
    }


def test_directly_named_behavior_does_not_need_pronoun_resolution():
    evidence = "南枝和苏弦检查展台。\n清晨，南枝却主动与陌生人长谈。"
    statement = "清晨，南枝却主动与陌生人长谈"
    provider = FakeProvider(response(record(evidence, statement=statement)))

    result = CharacterSignalExtractor(provider, settings=settings()).extract(
        chunk(evidence)
    )

    assert len(result.signals) == 1
    assert result.diagnostics.reason_counts == {}


def test_both_prompts_describe_the_same_fail_closed_pronoun_rule():
    for prompt in (
        CHARACTER_SIGNAL_SYSTEM_PROMPT,
        TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT,
    ):
        assert "statement 必须逐字复制后句" in prompt or "statement 须逐字复制后句" in prompt
        assert "多先行词/代词" in prompt
        assert "条件/分支" in prompt
