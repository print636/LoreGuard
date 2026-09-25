from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.character_trait_extraction import (
    CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2,
    CHARACTER_SIGNAL_SEMANTIC_SCOPE_PROMPT_V5,
    CHARACTER_SIGNAL_SYSTEM_PROMPT_V5,
    CharacterSignalChunk,
    CharacterSignalExtractor,
)
from app.config import Settings


class QueueProvider:
    def __init__(self, *rows: str):
        self.rows = list(rows)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str):
        self.calls.append((system, user))
        return SimpleNamespace(
            text=self.rows.pop(0), prompt_tokens=17, completion_tokens=9,
        )


def settings(**overrides) -> Settings:
    values = {
        "openai_api_key": "unit-test-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_character_consistency": True,
        "provider_max_attempts": 1,
        "character_signal_package_max_attempts": 1,
        "character_signal_full_line_prompt_v2": True,
        "character_signal_support_id_v4": True,
        "character_signal_semantic_scope_v5": True,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def row(
    source: str,
    statement: str,
    *,
    support_id: str = "L10:A3",
    actor_anchor_id: str = "L10:A1",
    label_anchor_id: str = "L10:A1",
    scope_relation: str = "labelled_elaboration",
    character: str = "桑衍",
    dimension: str = "core_personality",
    stability: str = "core",
    key_object: str = "风险",
    trait_key: str = "risk_disclosure",
    polarity: str = "positive",
    line: int = 10,
) -> dict:
    return {
        "character": character,
        "dimension": dimension,
        "trait_key": trait_key,
        "statement": statement,
        "polarity": polarity,
        "stability": stability,
        "observation_kind": "explicit_declaration",
        "context": "",
        "key_object": key_object,
        "source_line_start": line,
        "source_line_end": line,
        "evidence": source,
        "support_id": support_id,
        "actor_anchor_id": actor_anchor_id,
        "label_anchor_id": label_anchor_id,
        "scope_relation": scope_relation,
    }


def extract(source: str, *records: dict, configured: Settings | None = None):
    provider = QueueProvider(json.dumps({"records": records}, ensure_ascii=False))
    result = CharacterSignalExtractor(provider, settings=configured or settings()).extract(
        CharacterSignalChunk("v5-doc", "profile.md", source, 10, "formal_character_profile")
    )
    return result, provider


def test_v5_nonliteral_core_relation_remains_unresolved_after_context_clause():
    source = (
        "桑衍的核心性格是让共同值守的搭档知晓自己要承担的风险："
        "在无封港令、无救援保密任务的日常航路调整中，"
        "她会先向共同值守的搭档说明会影响其航船的风险。"
    )
    result, provider = extract(
        source,
        row(source, "桑衍会先向共同值守的搭档说明会影响其航船的风险"),
    )
    assert result.signals == ()
    assert result.pending_candidates == ()
    assert result.diagnostics.reason_counts.get("semantic_scope_unresolved") == 1
    assert provider.calls[0][0] == (
        CHARACTER_SIGNAL_SYSTEM_PROMPT_V5
        + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2
        + CHARACTER_SIGNAL_SEMANTIC_SCOPE_PROMPT_V5
    )


def test_v5_literal_core_definition_repeated_across_a_context_clause_is_pending():
    source = "桑衍的核心性格是说明风险，在日常值守中，她会说明风险。"
    proposed = row(
        source, "桑衍会说明风险", support_id="L10:A3",
    )
    result, _ = extract(source, proposed)
    assert result.diagnostics.outcome == "completed"
    assert result.signals[0].dimension == "core_personality"
    assert result.signals[0].actor_anchor_id == "L10:A1"
    assert result.signals[0].label_anchor_id == "L10:A1"
    assert "actor_anchor_id" not in result.signals[0].model_dump()
    assert result.pending_candidates[0].status == "pending"
    assert result.pending_candidates[0].evidence[0].text == source


def test_v5_unrelated_action_cannot_inherit_core_from_same_line():
    source = "桑衍的核心性格是让搭档知道风险，他会把桌面擦干净。"
    proposed = row(
        source, "桑衍会把桌面擦干净", support_id="L10:A2",
        key_object="桌面", trait_key="desk_cleaning",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.pending_candidates == ()
    assert result.diagnostics.reason_counts.get("semantic_scope_unresolved") == 1


def test_v5_stable_preference_uses_target_object_and_label_anchor():
    source = "荀木有长期稳定偏好：夜航结束后喜欢热姜梅露，会请人把常温的一杯重新加热。"
    proposed = row(
        source, "荀木夜航结束后喜欢热姜梅露",
        support_id="L10:A2", character="荀木", dimension="preference",
        stability="stable", key_object="热姜梅露", trait_key="ginger_plum_drink_preference",
    )
    result, _ = extract(source, proposed)
    assert result.diagnostics.outcome == "completed"
    assert len(result.signals) == 1
    assert result.signals[0].stability == "stable"
    assert result.signals[0].key_object == "热姜梅露"


def test_v5_unrelated_liking_cannot_inherit_core_but_can_be_pending_preference():
    source = "桑衍的核心性格是说明风险，她喜欢蜜瓜。"
    false_core = row(
        source, "桑衍喜欢蜜瓜", support_id="L10:A2", key_object="蜜瓜",
    )
    rejected, _ = extract(source, false_core)
    assert rejected.signals == ()
    assert rejected.diagnostics.reason_counts.get("semantic_scope_unresolved") == 1

    independent = row(
        source, "桑衍喜欢蜜瓜", support_id="L10:A2", key_object="蜜瓜",
        label_anchor_id="", scope_relation="same_actor_continuation",
        dimension="preference", stability="stable", trait_key="melon_preference",
    )
    accepted, _ = extract(source, independent)
    assert accepted.diagnostics.outcome == "completed"
    assert accepted.signals[0].dimension == "preference"
    assert accepted.signals[0].label_anchor_id is None
    assert accepted.pending_candidates[0].status == "pending"
    assert accepted.pending_candidates[0].evidence[0].text == source


def test_v5_stable_preference_label_cannot_change_object_across_clauses():
    source = "桑衍的稳定偏好是热茶，他也喜欢蜜瓜。"
    proposed = row(
        source, "桑衍也喜欢蜜瓜", support_id="L10:A2",
        key_object="蜜瓜", dimension="preference", stability="stable",
        trait_key="melon_preference",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("support_label_scope") == 1

    proposed["label_anchor_id"] = ""
    proposed["scope_relation"] = "same_actor_continuation"
    independent, _ = extract(source, proposed)
    assert independent.diagnostics.outcome == "completed"
    assert independent.pending_candidates[0].status == "pending"


@pytest.mark.parametrize("target", (
    "她说周尧不喜欢蜜瓜",
    "她听到周尧喜欢蜜瓜",
    "她否认周尧喜欢蜜瓜",
    "她喜欢蜜瓜吗",
))
def test_v5_cannot_assign_nested_report_or_question_to_anchor_actor(target: str):
    source = f"桑衍的核心性格是说明风险，{target}。"
    proposed = row(
        source, "桑衍" + target[1:], support_id="L10:A2",
        label_anchor_id="", scope_relation="same_actor_continuation",
        dimension="preference", stability="stable", key_object="蜜瓜",
        trait_key="melon_preference",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("character_support") == 1


def test_v5_direct_negative_preference_remains_valid_without_label_borrowing():
    source = "桑衍的核心性格是说明风险，她不喜欢蜜瓜。"
    proposed = row(
        source, "桑衍不喜欢蜜瓜", support_id="L10:A2",
        label_anchor_id="", scope_relation="same_actor_continuation",
        dimension="preference", stability="stable", key_object="蜜瓜",
        trait_key="melon_preference", polarity="negative",
    )
    result, _ = extract(source, proposed)
    assert result.diagnostics.outcome == "completed"
    assert result.signals[0].polarity == "negative"


def test_v5_unparsed_direct_preference_cannot_fall_back_to_substring_match():
    source = "桑衍的核心性格是说明风险，她喜欢蜜瓜和热茶。"
    proposed = row(
        source, "桑衍喜欢蜜瓜和热茶", support_id="L10:A2",
        label_anchor_id="", scope_relation="same_actor_continuation",
        dimension="preference", stability="stable", key_object="蜜瓜",
        trait_key="melon_preference",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("key_object_support") == 1


def test_v5_temporal_prefix_does_not_hide_a_new_intervening_subject():
    source = "桑衍喜欢茶，随后周尧喜欢蜜瓜，还喜欢热茶。"
    proposed = row(
        source, "桑衍还喜欢热茶", support_id="L10:A3",
        label_anchor_id="", scope_relation="same_actor_continuation",
        dimension="preference", stability="stable", key_object="热茶",
        trait_key="hot_tea_preference",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("scope_anchor_invalid") == 1


def test_v5_unknown_intervening_predicate_cannot_silently_reassign_subject():
    source = "桑衍喜欢茶，周尧钟爱蜜瓜，还喜欢热茶。"
    proposed = row(
        source, "桑衍还喜欢热茶", support_id="L10:A3",
        label_anchor_id="", scope_relation="same_actor_continuation",
        dimension="preference", stability="stable", key_object="热茶",
        trait_key="hot_tea_preference",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("scope_anchor_invalid") == 1


def test_v5_observer_cannot_anchor_following_observed_person_preference():
    source = "桑衍看见周尧喜欢蜜瓜，还喜欢热茶。"
    proposed = row(
        source, "桑衍还喜欢热茶", support_id="L10:A2",
        label_anchor_id="", scope_relation="same_actor_continuation",
        dimension="preference", stability="stable", key_object="热茶",
        trait_key="hot_tea_preference",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("semantic_scope_unresolved") == 1


def test_v5_cognitive_subject_cannot_anchor_nested_preference_continuation():
    source = "桑衍知道周尧喜欢蜜瓜，还喜欢热茶。"
    proposed = row(
        source, "桑衍还喜欢热茶", support_id="L10:A2",
        label_anchor_id="", scope_relation="same_actor_continuation",
        dimension="preference", stability="stable", key_object="热茶",
        trait_key="hot_tea_preference",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("semantic_scope_unresolved") == 1


@pytest.mark.parametrize("source", (
    "桑衍相信周尧喜欢蜜瓜。",
    "桑衍华喜欢蜜瓜。",
))
def test_v5_local_preference_needs_actor_bound_predicate_and_name_boundary(source: str):
    proposed = row(
        source, source[:-1], support_id="L10:A1", actor_anchor_id="",
        label_anchor_id="", scope_relation="local", dimension="preference",
        stability="stable", key_object="蜜瓜", trait_key="melon_preference",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("semantic_scope_unresolved") == 1


def test_v5_nonpreference_name_prefix_requires_a_provable_subject_boundary():
    source = "桑衍的核心性格是说明风险，桑衍华会说明风险。"
    proposed = row(
        source, "桑衍华会说明风险", support_id="L10:A2",
        actor_anchor_id="", key_object="风险",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("semantic_scope_unresolved") == 1


def test_v5_explicit_absence_of_value_cannot_be_given_positive_polarity():
    source = "桑衍没有责任感。"
    proposed = row(
        source, "桑衍没有责任感", support_id="L10:A1",
        actor_anchor_id="", label_anchor_id="", scope_relation="local",
        dimension="value", stability="stable", key_object="责任感",
        trait_key="responsibility", polarity="positive",
    )
    rejected, _ = extract(source, proposed)
    assert rejected.signals == ()
    assert rejected.diagnostics.reason_counts.get("statement_support") == 1

    proposed["polarity"] = "negative"
    accepted, _ = extract(source, proposed)
    assert accepted.diagnostics.outcome == "completed"
    assert accepted.signals[0].polarity == "negative"
    assert accepted.pending_candidates[0].status == "pending"


def test_v5_negated_stable_heading_cannot_label_following_preference():
    source = "桑衍没有长期稳定偏好：喜欢蜜瓜。"
    proposed = row(
        source, "桑衍喜欢蜜瓜", support_id="L10:A2",
        dimension="preference", stability="stable", key_object="蜜瓜",
        trait_key="melon_preference",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("semantic_scope_unresolved") == 1


@pytest.mark.parametrize("relation", ("妻子", "丈夫", "代理人", "孩子"))
def test_v5_possessed_person_cannot_become_anchor_actor(relation: str):
    source = f"桑衍的{relation}喜欢蜜瓜，还喜欢热茶。"
    proposed = row(
        source, "桑衍还喜欢热茶", support_id="L10:A2",
        label_anchor_id="", scope_relation="same_actor_continuation",
        dimension="preference", stability="stable", key_object="热茶",
        trait_key="hot_tea_preference",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("semantic_scope_unresolved") == 1


@pytest.mark.parametrize("target", ("她不再喜欢蜜瓜", "她没有喜欢过蜜瓜"))
def test_v5_changed_or_denied_preference_cannot_be_misread_as_positive(target: str):
    source = f"桑衍的核心性格是说明风险，{target}。"
    proposed = row(
        source, "桑衍" + target[1:], support_id="L10:A2",
        label_anchor_id="", scope_relation="same_actor_continuation",
        dimension="preference", stability="stable", key_object="蜜瓜",
        trait_key="melon_preference", polarity="positive",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("statement_support", 0) + (
        result.diagnostics.reason_counts.get("key_object_support", 0)
    ) == 1


@pytest.mark.parametrize("source,character,statement,key_object", (
    ("曲霁：核心人格是对自己的工作失误主动担责；曲霁会如实承认自己算错的巡检数据。",
     "曲霁", "曲霁会如实承认自己算错的巡检数据", "巡检数据"),
    ("温弦：核心人格是畏惧与师父在众人面前正面冲突；温弦通常不敢当众反驳师父的命令。",
     "温弦", "温弦通常不敢当众反驳师父的命令", "师父的命令"),
))
def test_v5_semantic_verb_is_not_limited_to_v4_head_vocabulary(
    source: str, character: str, statement: str, key_object: str,
):
    proposed = row(
        source, statement, support_id="L10:A2", actor_anchor_id="",
        character=character, key_object=key_object, dimension="core_personality",
        trait_key="accountability" if character == "曲霁" else "public_disagreement",
        polarity="positive" if character == "曲霁" else "negative",
    )
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("semantic_scope_unresolved") == 1
    proposed["label_anchor_id"] = ""
    proposed["scope_relation"] = "local"
    proposed["dimension"] = "value"
    proposed["stability"] = "stable"
    independent, _ = extract(source, proposed)
    assert independent.diagnostics.outcome == "completed"
    assert independent.signals[0].dimension == "value"
    assert independent.pending_candidates[0].status == "pending"


@pytest.mark.parametrize("source,statement,mutations,reason", (
    ("桑衍的核心性格是说明风险，周尧喜欢热茶，她喜欢蜜瓜。",
     "桑衍喜欢蜜瓜", {"key_object": "蜜瓜", "dimension": "preference", "stability": "stable"},
     "scope_anchor_invalid"),
    ("桑衍的核心性格是说明风险，周尧说“她喜欢蜜瓜”，她喜欢蜜瓜。",
     "桑衍喜欢蜜瓜", {"key_object": "蜜瓜", "dimension": "preference", "stability": "stable"},
     "scope_anchor_invalid"),
    ("桑衍的核心性格是说明风险，如果她喜欢蜜瓜。",
     "桑衍如果她喜欢蜜瓜", {"support_id": "L10:A2", "key_object": "蜜瓜"},
     "character_support"),
    ("桑衍的核心性格是说明风险，林澈的妹妹喜欢蜜瓜。",
     "林澈的妹妹喜欢蜜瓜", {"support_id": "L10:A2", "character": "林澈", "actor_anchor_id": "L10:A1", "key_object": "蜜瓜"},
     "semantic_scope_unresolved"),
    ("桑衍的核心性格是说明风险，桑衍然喜欢蜜瓜。",
     "桑衍然喜欢蜜瓜", {"support_id": "L10:A2", "key_object": "蜜瓜"},
     "semantic_scope_unresolved"),
))
def test_v5_unsafe_actor_chain_is_not_promoted(source, statement, mutations, reason):
    proposed = row(source, statement)
    proposed.update(mutations)
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get(reason) == 1


@pytest.mark.parametrize("mutation,reason", (
    ({"actor_anchor_id": "L10:A9"}, "scope_anchor_invalid"),
    ({"actor_anchor_id": "L11:A1"}, "scope_anchor_invalid"),
    ({"actor_anchor_id": "L10:A4"}, "scope_anchor_invalid"),
    ({"label_anchor_id": "L10:A4"}, "scope_anchor_invalid"),
    ({"label_anchor_id": "", "scope_relation": "same_actor_continuation"}, "core_label_scope"),
    ({"scope_relation": "local"}, "scope_relation_invalid"),
))
def test_v5_anchor_id_and_relation_are_independently_checked(mutation, reason):
    source = "桑衍的核心性格是说明风险，在日常航路调整中，她会先说明风险，再发出指令。"
    proposed = row(source, "桑衍会先说明风险")
    proposed.update(mutation)
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get(reason) == 1


@pytest.mark.parametrize("source,mutations,reason", (
    ("桑衍不是核心性格是说明风险，她会先说明风险。", {}, "semantic_scope_unresolved"),
    ("桑衍的核心性格是说明风险，她喜欢蜜瓜味糖。",
     {"support_id": "L10:A2", "statement": "桑衍喜欢蜜瓜味糖", "key_object": "蜜瓜",
      "dimension": "preference", "stability": "stable"}, "key_object_support"),
    ("桑衍的核心性格是说明风险，她讨厌蜜瓜。",
     {"support_id": "L10:A2", "statement": "桑衍讨厌蜜瓜", "key_object": "蜜瓜",
      "dimension": "preference", "stability": "stable", "polarity": "positive"},
     "statement_support"),
    ("桑衍的核心性格是说明风险，她会先说明风险。",
     {"support_id": "L10:A2", "statement": "桑衍偶尔说明风险"}, "statement_support"),
))
def test_v5_label_preference_and_statement_stay_local(source, mutations, reason):
    proposed = row(source, "桑衍会先说明风险", support_id="L10:A2")
    proposed.update(mutations)
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get(reason) == 1


def test_v5_feature_dependency_is_default_off_and_model_copy_guarded():
    assert Settings(_env_file=None).character_signal_semantic_scope_v5 is False
    with pytest.raises(ValueError, match="semantic scope v5 requires support id v4"):
        settings(character_signal_support_id_v4=False)
    configured = settings().model_copy(update={"character_signal_support_id_v4": False})
    with pytest.raises(RuntimeError, match="semantic scope v5 requires support id v4"):
        CharacterSignalExtractor(QueueProvider(), settings=configured).extract(
            CharacterSignalChunk("v5-doc", "profile.md", "桑衍喜欢蜜瓜。", 10, "formal_character_profile")
        )


@pytest.mark.parametrize("change", (
    {"actor_anchor_id": None},
    {"label_anchor_id": None},
    {"scope_relation": "inferred"},
    {"untrusted_override": "confirmed"},
))
def test_v5_exact_schema_rejects_missing_or_extra_scope_fields(change: dict):
    source = "桑衍喜欢蜜瓜。"
    proposed = row(
        source, "桑衍喜欢蜜瓜", support_id="L10:A1",
        actor_anchor_id="", label_anchor_id="", scope_relation="local",
        dimension="preference", stability="stable", key_object="蜜瓜",
        trait_key="melon_preference",
    )
    proposed.update(change)
    result, _ = extract(source, proposed)
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("schema_validation") == 1


def test_v5_retry_cannot_swap_a_verified_actor_anchor():
    source = "桑衍喜欢茶，桑衍喜欢水，她喜欢蜜瓜。"
    original = row(
        source, "桑衍喜欢蜜瓜", actor_anchor_id="L10:A1",
        label_anchor_id="", scope_relation="same_actor_continuation",
        dimension="preference", stability="stable", key_object="蜜瓜",
        trait_key="melon_preference",
    )
    changed = dict(original, actor_anchor_id="L10:A2")
    invalid = dict(original, unknown="not part of protocol")
    provider = QueueProvider(
        json.dumps({"records": [original, invalid]}, ensure_ascii=False),
        json.dumps({"records": [changed]}, ensure_ascii=False),
    )
    result = CharacterSignalExtractor(
        provider, settings=settings(character_signal_package_max_attempts=2),
    ).extract(CharacterSignalChunk("v5-doc", "profile.md", source, 10, "formal_character_profile"))
    assert result.signals == ()
    assert result.diagnostics.reason_counts.get("regeneration_coverage_regression") == 1
    assert len(provider.calls) == 2
    assert '"actor_anchor_id":"L10:A1"' in provider.calls[1][1]


def test_v5_on_keeps_history_on_original_twelve_field_protocol():
    source = "桑衍喜欢蜜瓜。"
    proposed = row(
        source, "桑衍喜欢蜜瓜", support_id="L10:A1",
        actor_anchor_id="", label_anchor_id="", scope_relation="local",
        dimension="preference", stability="stable", key_object="蜜瓜",
        trait_key="melon_preference",
    )
    for key in ("support_id", "actor_anchor_id", "label_anchor_id", "scope_relation"):
        proposed.pop(key)
    provider = QueueProvider(json.dumps({"records": [proposed]}, ensure_ascii=False))
    result = CharacterSignalExtractor(provider, settings=settings()).extract(
        CharacterSignalChunk("history-doc", "history.md", source, 10, "published_history")
    )
    assert result.diagnostics.outcome == "completed"
    assert result.signals[0].support_id is None
    assert CHARACTER_SIGNAL_SEMANTIC_SCOPE_PROMPT_V5 not in provider.calls[0][0]
