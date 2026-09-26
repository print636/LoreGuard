"""Developer-visible narrative contracts, with deterministic mock model records.

These tests exercise production validation and drift promotion, not model
accuracy. The source prose is original and lives in one independent DEV fixture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from app.character_drift import (
    CharacterDriftCase,
    CharacterReviewDiagnostics,
    CharacterReviewResult,
    ConfirmedTraitSnapshot,
    ModelDriftDecision,
    SupportEvidence,
    prepare_character_drift,
    promote_character_drift,
)
from app.character_trait_extraction import (
    CharacterSignal,
    CharacterSignalChunk,
    CharacterSignalExtractor,
)
from app.config import Settings
from app.domain import EvidenceSpan


FIXTURE = Path(__file__).resolve().parents[1] / "data" / "character-ooc-complex-dev-v1"
AXIS = "闻岚面对可能伤害居民的师父命令时，是否敢在居民面前当众反对师父命令。"
PROPOSITION = "闻岚当众反对可能伤害居民的师父命令"


class _KnownAttributionLeak(AssertionError):
    """Only the observed unsafe attribution may count as an expected failure."""


def _line(name: str, number: int) -> str:
    return (FIXTURE / name).read_text(encoding="utf-8").splitlines()[number - 1]


def _span(name: str, number: int) -> EvidenceSpan:
    return EvidenceSpan(
        document_id=name.removesuffix(".md"),
        document_name=name,
        line_start=number,
        line_end=number,
        text=_line(name, number),
    )


def _review(verdict: str, *citations: str) -> CharacterReviewResult:
    return CharacterReviewResult(
        decision=ModelDriftDecision(
            verdict=verdict,
            explanation="按已发布事件和当次行为的证据作出判断。",
            citations=citations,
        ),
        diagnostics=CharacterReviewDiagnostics(outcome="completed", reason="completed"),
    )


def _support(name: str, number: int, kind: str) -> SupportEvidence:
    return SupportEvidence(
        id=f"se_{name.removesuffix('.md')}_{number}",
        kind=kind,
        summary="已发生且已发布的同一角色事件",
        explicit=True,
        evidence=_span(name, number),
        source_kind="published_history",
        publication_status="published",
        authority_tier="formal_record",
        resolution_state="confirmed",
        source_ordinal=10,
        eligible_draft_document_ids=("draft",),
    )


def _observation(
    *, character: str, dimension: str, trait_key: str, statement: str,
    polarity: str, key_object: str, line: int,
) -> CharacterSignal:
    return CharacterSignal(
        id="cs_" + "a" * 32,
        character=character,
        dimension=dimension,
        trait_key=trait_key,
        statement=statement,
        polarity=polarity,
        stability="situational",
        observation_kind="action",
        key_object=key_object,
        source_kind="draft",
        evidence=_span("draft.md", line),
    )


def _growth_case(support: SupportEvidence) -> CharacterDriftCase:
    baseline = ConfirmedTraitSnapshot(
        id="ct_wenlan_public_opposition",
        character="闻岚",
        dimension="core_personality",
        trait_key="public_opposition",
        statement="闻岚通常不敢当众反对师父命令",
        polarity="negative",
        stability="core",
        origin="explicit_setting",
        evidence=(_span("profiles.md", 2),),
        approved_axis_id="00000000-0000-4000-8000-000000000041",
        approved_axis_version=1,
        approved_axis_display_name="公开反对师父命令",
        approved_axis_definition=AXIS,
        approved_axis_definition_sha256=hashlib.sha256(AXIS.encode()).hexdigest(),
        axis_positive_proposition=PROPOSITION,
        axis_positive_proposition_sha256=hashlib.sha256(
            PROPOSITION.encode()
        ).hexdigest(),
        axis_alignment="same",
        axis_polarity="negative",
    )
    observation = _observation(
        character="闻岚", dimension="core_personality",
        trait_key="model_label_is_not_author_axis",
        statement="闻岚当众反对师父命令",
        polarity="positive", key_object="", line=2,
    )
    return CharacterDriftCase(
        id="cdc_growth_after_flood",
        baseline=baseline,
        observations=(observation,),
        support_evidence=(support,),
        scope_compatibility="compatible",
        material_coverage="complete",
        approved_axis_bound_observation_ids=(observation.id,),
        approved_axis_observation_polarities=((observation.id, "positive"),),
    )


def _medical_case(support: SupportEvidence) -> CharacterDriftCase:
    baseline = ConfirmedTraitSnapshot(
        id="ct_qitang_cake",
        character="祁棠",
        dimension="preference",
        trait_key="热栗椒饼",
        statement="祁棠长期喜欢热栗椒饼",
        polarity="positive",
        stability="stable",
        origin="explicit_setting",
        evidence=(_span("profiles.md", 5),),
    )
    observation = _observation(
        character="祁棠", dimension="preference", trait_key="热栗椒饼",
        statement="祁棠当场暂不吃热栗椒饼",
        polarity="negative", key_object="热栗椒饼", line=7,
    )
    return CharacterDriftCase(
        id="cdc_medical_food_exception",
        baseline=baseline,
        observations=(observation,),
        support_evidence=(support,),
        scope_compatibility="compatible",
        material_coverage="complete",
    )


class _QueueProvider:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload, ensure_ascii=False)
        self.calls = 0

    def complete(self, system: str, user: str):
        del system, user
        self.calls += 1
        return SimpleNamespace(text=self.payload, prompt_tokens=17, completion_tokens=9)


def _extract(
    source: str, *, line: int, records: list[dict],
) -> tuple[object, _QueueProvider]:
    provider = _QueueProvider({"records": records})
    settings = Settings(
        _env_file=None,
        openai_api_key="unit-test-placeholder",
        openai_base_url="https://mock.invalid/v1",
        openai_model="mock-model",
        enable_character_consistency=True,
        provider_max_attempts=1,
        character_signal_package_max_attempts=1,
        character_signal_full_line_prompt_v2=True,
        character_signal_support_id_v4=True,
    )
    result = CharacterSignalExtractor(provider, settings=settings).extract(
        CharacterSignalChunk("draft", "draft.md", source, line, "draft")
    )
    return result, provider


def _record(
    source: str, *, line_start: int, line_end: int | None = None,
    character: str, statement: str, trait_key: str,
) -> dict:
    return {
        "character": character,
        "dimension": "core_personality",
        "trait_key": trait_key,
        "statement": statement,
        "polarity": "negative",
        "stability": "situational",
        "observation_kind": "action",
        "context": "值守期间",
        "key_object": "",
        "source_line_start": line_start,
        "source_line_end": line_end or line_start,
        "evidence": source,
    }


def test_original_fixture_anchors_are_distinct_and_fixed() -> None:
    assert "当众反对师父命令" in _line("profiles.md", 2)
    assert "复盘会上" in _line("history.md", 3)
    assert "祁棠当日照做" in _line("history.md", 5)
    assert "乔唯当日照做" in _line("history.md", 6)
    assert "戏本念出台词" in _line("draft.md", 3)
    assert "乔唯此时真的" in _line("draft.md", 3)
    assert "余霁与乔唯" in _line("draft.md", 4)
    assert "只剩余霁" in _line("draft.md", 5)
    assert "祁棠当场把热栗椒饼推回" in _line("draft.md", 7)


def test_published_flood_growth_explains_actual_public_opposition() -> None:
    prepared = prepare_character_drift(_growth_case(
        _support("history.md", 3, "causal_bridge")
    ))
    assert prepared.reason == "single_published_growth_review_only"
    assert prepared.reviewer_eligible is True
    result = promote_character_drift(prepared, _review("explained", "B01", "C01", "G01"))
    assert result.outcome == "no_issue"
    assert result.visible is False

    # The harmful accident alone is not a completed change of stance.
    accident_only = prepare_character_drift(_growth_case(
        _support("history.md", 2, "causal_bridge")
    ))
    assert accident_only.reviewer_eligible is False
    assert promote_character_drift(accident_only, None).outcome != "no_issue"


def test_scripted_confession_cannot_become_the_reader_actor_action() -> None:
    source = _line("draft.md", 3)
    false_actor, false_provider = _extract(
        source, line=3, records=[_record(
            source, line_start=3, character="洛原",
            statement="洛原替同伴签了名", trait_key="unauthorized_signature",
        )],
    )
    assert false_provider.calls >= 1
    leaked = tuple(
        (row.character, row.statement, row.observation_kind, row.evidence.line_start)
        for row in false_actor.draft_observations
    )
    if leaked == (("洛原", "洛原替同伴签了名", "action", 3),):
        assert false_actor.diagnostics.reason_counts == {}
        raise _KnownAttributionLeak(
            "scripted_quote_leak: a verbatim scripted quote was accepted "
            "as 洛原's action with no rejection diagnostic"
        )
    assert leaked == (), (
        "scripted_quote_leak: expected no actual action for 洛原, "
        f"accepted={leaked!r}, reasons={false_actor.diagnostics.reason_counts!r}"
    )
    assert false_actor.diagnostics.reason_counts


def test_actual_other_actor_on_script_line_is_still_recoverable() -> None:
    source = _line("draft.md", 3)
    actual_actor, _ = _extract(
        source, line=3, records=[_record(
            source, line_start=3, character="乔唯",
            statement="乔唯此时真的在值守簿上替同伴签了名",
            trait_key="unauthorized_signature",
        )],
    )
    assert [row.character for row in actual_actor.draft_observations] == [
        "乔唯"
    ], actual_actor.diagnostics.reason_counts
    assert actual_actor.draft_observations[0].evidence.line_start == 3


def test_ambiguous_pronoun_cannot_be_assigned_to_first_named_actor() -> None:
    ambiguous = _line("draft.md", 4)
    wrong_actor, _ = _extract(
        ambiguous, line=4, records=[_record(
            ambiguous, line_start=4, character="余霁",
            statement=ambiguous.rstrip("。"), trait_key="seal_concealment",
        )],
    )
    leaked = tuple(
        (row.character, row.statement, row.observation_kind, row.evidence.line_start)
        for row in wrong_actor.draft_observations
    )
    if leaked == (("余霁", ambiguous.rstrip("。"), "action", 4),):
        assert wrong_actor.diagnostics.reason_counts == {}
        raise _KnownAttributionLeak(
            "ambiguous_pronoun_leak: the literal two-actor she-clause "
            "was accepted as 余霁's action with no rejection diagnostic"
        )
    assert leaked == (), (
        "ambiguous_pronoun_leak: she has two candidate antecedents, "
        f"accepted={leaked!r}, reasons={wrong_actor.diagnostics.reason_counts!r}"
    )
    assert wrong_actor.diagnostics.reason_counts


def test_unique_adjacent_pronoun_anchor_remains_recoverable() -> None:
    safe_source = "\n".join((_line("draft.md", 5), _line("draft.md", 6)))
    safe, _ = _extract(
        safe_source, line=5, records=[_record(
            safe_source, line_start=5, line_end=6, character="余霁",
            statement="随后，余霁把印章藏进袖中，没有交还值守柜",
            trait_key="seal_concealment",
        )],
    )
    assert [row.character for row in safe.draft_observations] == [
        "余霁"
    ], safe.diagnostics.reason_counts
    assert (safe.draft_observations[0].evidence.line_start,
            safe.draft_observations[0].evidence.line_end) == (5, 6)


def test_temporary_food_refusal_requires_same_patient_published_order() -> None:
    prepared = prepare_character_drift(_medical_case(
        _support("history.md", 5, "exception")
    ))
    assert prepared.reason == "single_medical_exception_review_only"
    assert prepared.reviewer_eligible is True
    result = promote_character_drift(prepared, _review("explained", "B01", "C01", "X01"))
    assert result.outcome == "no_issue"
    assert result.visible is False

    other_patient = _support("history.md", 6, "exception")
    wrong = prepare_character_drift(_medical_case(other_patient))
    assert wrong.reviewer_eligible is False
    assert promote_character_drift(wrong, None).outcome != "no_issue"
