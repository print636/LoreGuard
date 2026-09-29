from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.character_consistency_stage import (
    _FrozenDocument,
    _aliases_by_character,
    _character_appears_in_chunk,
    _safe_server_context,
    _signal_bound_to_target_character,
    _target_candidate_line_ranges,
    _unique_alias_map,
)
from app.character_drift import ConfirmedTraitSnapshot
from app.character_trait_extraction import (
    CharacterSignal,
    CharacterSignalChunk,
    CharacterSignalExtractor,
    CharacterSignalTarget,
)
from app.config import Settings
from app.domain import EvidenceSpan
from app.narrative_context import NarrativeScopeV1
from app.pipeline import DocumentInput


class _SequenceProvider:
    def __init__(self, *responses: str):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str):
        self.calls.append((system, user))
        if not self.responses:
            raise AssertionError("unexpected provider call")
        return SimpleNamespace(
            text=self.responses.pop(0),
            prompt_tokens=17,
            completion_tokens=9,
        )


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key="alias-test-placeholder",
        openai_base_url="https://mock.invalid/v1",
        openai_model="mock-model",
        enable_character_consistency=True,
        provider_max_attempts=1,
    )


def _baseline(character: str, candidate_id: str):
    return (
        SimpleNamespace(candidate_id=candidate_id),
        ConfirmedTraitSnapshot(
            id=f"ct_{candidate_id}",
            character=character,
            dimension="core_personality",
            trait_key="social_initiative",
            statement=f"{character}很少主动与陌生人交谈",
            polarity="negative",
            stability="core",
            origin="explicit_setting",
            evidence=(
                EvidenceSpan(
                    document_id="profile",
                    document_name="profile.md",
                    line_start=1,
                    line_end=1,
                    text=f"{character}很少主动与陌生人交谈。",
                ),
            ),
        ),
        NarrativeScopeV1(),
        character,
    )


def _source(
    content: str,
    *,
    source_kind: str = "formal_character_profile",
    authority_tier: str = "formal_record",
) -> _FrozenDocument:
    return _FrozenDocument(
        input_id="input-profile",
        document=DocumentInput(
            id="profile",
            name="profile.md",
            content=content,
            role="character_profile" if source_kind == "formal_character_profile" else "chapter",
        ),
        document_version=1,
        content_sha256="0" * 64,
        ordinal=0,
        source_kind=source_kind,
        source_reason="formal" if source_kind == "formal_character_profile" else "draft",
        scope=NarrativeScopeV1(),
        resolution_state="confirmed",
        publication_status="published" if source_kind == "formal_character_profile" else "draft",
        authority_tier=authority_tier,
    )


def test_formal_profile_alias_binds_to_one_confirmed_character() -> None:
    baselines = [_baseline("林澈", "00000000-0000-4000-8000-000000000101")]
    alias_map = _unique_alias_map(
        baselines,
        documents=[_source("林澈又名阿澈。\n林澈性格内向。")],
    )

    assert alias_map["阿澈"] == ("林澈",)
    profile = _source("林澈又名阿澈。\n林澈性格内向。")
    aliases = _aliases_by_character(
        baselines, alias_map=alias_map, documents=[profile]
    )
    assert aliases["林澈"] == ("阿澈",)
    assert _character_appears_in_chunk(
        "林澈",
        "阿澈在宴会上主动与陌生人长谈。",
        aliases=aliases["林澈"],
    )
    signal = CharacterSignal(
        id="cs_" + "1" * 32,
        character="阿澈",
        dimension="core_personality",
        trait_key="social_initiative",
        statement="阿澈主动与陌生人长谈",
        polarity="positive",
        stability="situational",
        observation_kind="action",
        source_kind="draft",
        evidence=EvidenceSpan(
            document_id="draft",
            document_name="draft.md",
            line_start=1,
            line_end=1,
            text="阿澈主动与陌生人长谈。",
        ),
    )
    target = CharacterSignalTarget(
        character="林澈",
        dimension="core_personality",
        trait_key="social_initiative",
        comparison_key="core_personality:socialinitiative",
        baseline_polarity="negative",
        requested_polarity="positive",
        baseline_hint="很少主动与陌生人交谈",
    )
    bound = _signal_bound_to_target_character(
        signal, target=target, alias_map=alias_map
    )
    assert bound.character == "林澈"
    assert bound.id == signal.id
    assert bound.evidence == signal.evidence


def test_formal_alias_flows_through_targeted_prompt_and_preserves_source() -> None:
    baselines = [_baseline("林澈", "00000000-0000-4000-8000-000000000105")]
    profile = _source("林澈又名阿澈。\n林澈性格内向。")
    draft = _source(
        "阿澈主动邀请陌生摊主长谈。",
        source_kind="draft",
        authority_tier="draft",
    )
    alias_map = _unique_alias_map(baselines, documents=[profile, draft])
    aliases = _aliases_by_character(
        baselines,
        alias_map=alias_map,
        documents=[profile, draft],
    )
    context = _safe_server_context(
        draft,
        baselines=baselines,
        authorized_aliases_by_character=aliases,
    )
    target = context.targets[0]
    assert target.character == "林澈"
    assert target.authorized_aliases == ("阿澈",)

    chunk = CharacterSignalChunk(
        document_id=draft.document.id,
        document_name=draft.document.name,
        content=draft.document.content,
        global_line_start=1,
        source_kind="draft",
    )
    candidate_ranges, omitted = _target_candidate_line_ranges(chunk, target)
    assert candidate_ranges == ((1, 1),)
    assert omitted == 0

    record = {
        "character": "阿澈",
        "dimension": "core_personality",
        "trait_key": "social_initiative",
        "statement": "阿澈主动邀请陌生摊主长谈",
        "polarity": "positive",
        "stability": "temporary",
        "observation_kind": "action",
        "context": "",
        "key_object": "",
        "source_line_start": 1,
        "source_line_end": 1,
        "evidence": draft.document.content,
    }
    provider = _SequenceProvider(
        json.dumps({"records": [record]}, ensure_ascii=False)
    )
    result = CharacterSignalExtractor(
        provider, settings=_settings()
    ).extract_targeted(
        chunk,
        (target,),
        candidate_evidence_ranges=candidate_ranges,
    )

    assert result.diagnostics.outcome == "completed"
    assert len(result.signals) == 1
    raw = result.signals[0]
    assert raw.character == "阿澈"
    assert raw.evidence.text == draft.document.content
    assert '"authorized_aliases":["阿澈"]' in provider.calls[0][1]
    bound = _signal_bound_to_target_character(
        raw, target=target, alias_map=alias_map
    )
    assert bound.character == "林澈"
    assert bound.id == raw.id
    assert bound.evidence == raw.evidence


def test_targeted_alias_must_match_authorized_literal_exactly() -> None:
    line = "阿澈主动邀请陌生摊主长谈。"
    target = CharacterSignalTarget(
        character="林澈",
        authorized_aliases=("阿澈",),
        dimension="core_personality",
        trait_key="social_initiative",
        comparison_key="core_personality:socialinitiative",
        baseline_polarity="negative",
        requested_polarity="positive",
        baseline_hint="很少主动与陌生人交谈",
    )
    record = {
        "character": "阿澈 ",
        "dimension": "core_personality",
        "trait_key": "social_initiative",
        "statement": "阿澈主动邀请陌生摊主长谈",
        "polarity": "positive",
        "stability": "temporary",
        "observation_kind": "action",
        "context": "",
        "key_object": "",
        "source_line_start": 1,
        "source_line_end": 1,
        "evidence": line,
    }
    payload = json.dumps({"records": [record]}, ensure_ascii=False)
    result = CharacterSignalExtractor(
        _SequenceProvider(payload, payload), settings=_settings()
    ).extract_targeted(
        CharacterSignalChunk("draft", "draft.md", line, 1, "draft"),
        (target,),
    )

    assert result.signals == ()
    assert result.diagnostics.outcome == "degraded"
    assert result.diagnostics.reason_counts == {"targeted_target_mismatch": 2}


@pytest.mark.parametrize(
    "unsafe_alias",
    (
        "我", "我们", "你", "你们", "他", "她", "它们", "其", "本人", "自己",
        "这个", "那些", "此人", "该人", "对方", "they",
    ),
)
def test_signal_target_service_boundary_rejects_nonunique_aliases(
    unsafe_alias: str,
) -> None:
    with pytest.raises(
        ValidationError, match="authorized target alias is invalid or ambiguous"
    ):
        CharacterSignalTarget(
            character="林澈",
            authorized_aliases=(unsafe_alias,),
            dimension="core_personality",
            trait_key="social_initiative",
            comparison_key="core_personality:socialinitiative",
            baseline_polarity="negative",
            requested_polarity="positive",
            baseline_hint="很少主动与陌生人交谈",
        )


def test_formal_pronoun_alias_is_not_authorized_for_character_recall() -> None:
    baselines = [_baseline("林澈", "00000000-0000-4000-8000-000000000106")]
    profile = _source("林澈又名她。\n林澈性格内向。")
    draft = _source(
        "她主动邀请陌生摊主长谈。",
        source_kind="draft",
        authority_tier="draft",
    )
    alias_map = _unique_alias_map(baselines, documents=[profile, draft])
    # General entity alias parsing remains untouched; only the character
    # recall authorization boundary rejects a non-unique actor literal.
    assert alias_map["她"] == ("林澈",)
    aliases = _aliases_by_character(
        baselines,
        alias_map=alias_map,
        documents=[profile, draft],
    )
    assert aliases["林澈"] == ()
    target = _safe_server_context(
        draft,
        baselines=baselines,
        authorized_aliases_by_character=aliases,
    ).targets[0]
    assert target.authorized_aliases == ()


def test_draft_cannot_grant_itself_a_character_alias() -> None:
    baselines = [_baseline("林澈", "00000000-0000-4000-8000-000000000102")]
    profile = _source("林澈性格内向。")
    draft = _source(
        "林澈又名阿澈。", source_kind="draft", authority_tier="draft"
    )
    alias_map = _unique_alias_map(
        baselines,
        documents=[profile, draft],
    )

    assert "阿澈" not in alias_map
    assert not _character_appears_in_chunk(
        "林澈", "阿澈在宴会上主动攀谈。"
    )
    aliases = _aliases_by_character(
        baselines, alias_map=alias_map, documents=[profile, draft]
    )
    target = _safe_server_context(
        draft,
        baselines=baselines,
        authorized_aliases_by_character=aliases,
    ).targets[0]
    assert target.authorized_aliases == ()


def test_ambiguous_formal_alias_and_name_collision_fail_closed() -> None:
    baselines = [
        _baseline("林澈", "00000000-0000-4000-8000-000000000103"),
        _baseline("苏弦", "00000000-0000-4000-8000-000000000104"),
    ]
    ambiguous_profile = _source("林澈又名小澈。\n苏弦又名小澈。")
    ambiguous = _unique_alias_map(
        baselines,
        documents=[ambiguous_profile],
    )
    assert "小澈" not in ambiguous
    ambiguous_aliases = _aliases_by_character(
        baselines, alias_map=ambiguous, documents=[ambiguous_profile]
    )
    assert all(
        "小澈" not in values for values in ambiguous_aliases.values()
    )

    collision_profile = _source("林澈又名苏弦。")
    collision = _unique_alias_map(
        baselines,
        documents=[collision_profile],
    )
    assert set(collision["苏弦"]) == {"林澈", "苏弦"}
    collision_aliases = _aliases_by_character(
        baselines, alias_map=collision, documents=[collision_profile]
    )
    assert all(
        "苏弦" not in values for values in collision_aliases.values()
    )
