"""A named actor can own a nearby omitted-subject action, not another's act."""

import pytest

from app.character_trait_extraction import (
    CharacterSignalChunk,
    _RawCharacterSignal,
    _bind_record,
)


def _bind(source: str, statement: str):
    record = _RawCharacterSignal(
        character="林澈", dimension="contextual_behavior",
        trait_key="record_integrity", statement=statement,
        polarity="negative", stability="situational",
        observation_kind="action", key_object="", context="",
        source_line_start=1, source_line_end=1, evidence=source,
    )
    return _bind_record(
        record,
        CharacterSignalChunk("doc", "draft.md", source, 1, "draft"),
    )


@pytest.mark.parametrize("source,statement", [
    (
        "林澈先核对原始记录，仍亲手把危险读数改成安全值。",
        "林澈亲手把危险读数改成安全值",
    ),
    (
        "林澈看到两份预测互相矛盾，尚无复测，却在公告上写已证实安全。",
        "林澈在公告上写已证实安全",
    ),
])
def test_named_actor_owns_bounded_same_sentence_action(source: str, statement: str):
    bound = _bind(source, statement)
    assert bound.character == "林澈"
    assert bound.evidence.text == source


@pytest.mark.parametrize("source,statement", [
    (
        "林澈先核对原始记录，周尧亲手把危险读数改成安全值。",
        "林澈亲手把危险读数改成安全值",
    ),
    (
        "林澈先核对原始记录，周尧换班，仍亲手把危险读数改成安全值。",
        "林澈亲手把危险读数改成安全值",
    ),
    (
        "林澈让周尧核对原始记录，周尧仍亲手把危险读数改成安全值。",
        "林澈亲手把危险读数改成安全值",
    ),
    (
        "林澈让周尧核对原始记录，仍亲手把危险读数改成安全值。",
        "林澈亲手把危险读数改成安全值",
    ),
    (
        "林澈看到周尧核对原始记录，仍亲手把危险读数改成安全值。",
        "林澈亲手把危险读数改成安全值",
    ),
    (
        "林澈先核对原始记录，仍由周尧亲手把危险读数改成安全值。",
        "林澈亲手把危险读数改成安全值",
    ),
    (
        "林澈先核对原始记录，尚无复测，周尧却在公告上写已证实安全。",
        "林澈在公告上写已证实安全",
    ),
    (
        "林澈先核对原始记录，仍亲手把危险读数改成安全值只是传言。",
        "林澈亲手把危险读数改成安全值",
    ),
    (
        "林澈先核对原始记录，仍亲手把危险读数改成安全值，只是传言。",
        "林澈亲手把危险读数改成安全值",
    ),
    (
        "林澈先核对原始记录，仍亲手把危险读数改成安全值，查明动手的是周尧。",
        "林澈亲手把危险读数改成安全值",
    ),
    (
        "如果林澈看到原始记录，她会把危险读数改成安全值。",
        "林澈把危险读数改成安全值",
    ),
])
def test_other_actor_and_non_event_never_borrow_named_actor(
    source: str, statement: str,
):
    with pytest.raises(ValueError):
        _bind(source, statement)
