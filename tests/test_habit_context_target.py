"""An author's habit conditions survive targeted extraction and review as data."""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.character_consistency_stage import _assess_observation_match, _safe_server_context
from app.character_draft_actor_review import (
    HABIT_CONTEXT_TARGET_GUIDE,
    target_bound_draft_target_digest,
)
from app.character_draft_actor_review_provider import build_target_bound_draft_review_prompts
from app.character_trait_extraction import (
    CharacterSignal, CharacterSignalChunk, CharacterSignalTarget,
    _matching_target, _target_bound_review_target, _targeted_chunk_prompt,
)
from app.narrative_context import NarrativeScopeV1
from tests.test_character_consistency_stage import _confirmed_trait
from tests.test_character_target_bound_draft_review_v4 import _request
from tests.test_scoped_axis_target_contract import _source


def entry(contexts, candidate_id="choice"):
    baseline = _confirmed_trait(dimension="contextual_behavior", trait_key="关门").model_copy(
        update={"contexts": contexts, "statement": "林澈习惯关门"}
    )
    return SimpleNamespace(candidate_id=candidate_id), baseline, NarrativeScopeV1(), "林澈"


def test_frozen_habit_travels_to_both_calls_and_never_becomes_draft_text():
    contexts = ("完成晚间盘点时", "离开库房前")
    context = _safe_server_context(_source(), baselines=[entry(contexts)])
    target, = context.targets
    assert target.habit_contexts == contexts
    assert "habit_contexts" not in json.loads(context.payload)["confirmed_traits"][0]
    draft = "林澈将门敞开。"
    prompt = _targeted_chunk_prompt(CharacterSignalChunk("draft", "稿", draft, 1, "draft"), (target,))
    assert HABIT_CONTEXT_TARGET_GUIDE in prompt
    assert json.loads(prompt.split("targets：", 1)[1].splitlines()[0])[0]["habit_contexts"] == list(contexts)
    assert prompt.split("原文如下：\n", 1)[1] == f"1: {draft}"
    review_target = _target_bound_review_target(target, target_ordinal=1)
    _, request = _request(draft, target=review_target)
    system, user = build_target_bound_draft_review_prompts((request,))
    data = json.loads(user.split("：\n", 1)[1])
    assert data["habit_context_guide"] == HABIT_CONTEXT_TARGET_GUIDE
    assert data["windows"][0]["request"]["target"]["habit_contexts"] == list(contexts)
    assert HABIT_CONTEXT_TARGET_GUIDE not in system


@pytest.mark.parametrize("bad", [
    (), ("盘点", ""), ("盘点", "api_key=example"), ("盘点", "阁楼\u2028地下室"),
    ("长" * 180, "长" * 141), ("教育",) * 13,
])
def test_partial_or_unsafe_context_never_broadens_a_habit(bad):
    context = _safe_server_context(_source(), baselines=[entry(bad)])
    assert not context.targets and context.eligible_traits == 1 and context.included_traits == 0
    assert json.loads(context.payload)["confirmed_traits_coverage"]["state"] == "partial"


def test_competing_habit_contexts_do_not_become_one_arbitrarily_chosen_scope():
    context = _safe_server_context(_source(), baselines=[entry(("早班",)), entry(("晚班",), "other")])
    assert not context.targets and context.ambiguous_traits == 1


def test_habit_certificate_cannot_move_to_another_context_but_raw_action_can():
    current = entry(("值晚班时",))
    target, = _safe_server_context(_source(), baselines=[current]).targets
    raw = CharacterSignal(
        id="cs_" + "a" * 32, character=target.character, dimension=target.dimension,
        trait_key=target.trait_key, statement="林澈将门敞开", polarity=target.requested_polarity,
        stability="temporary", observation_kind="action", source_kind="draft",
        evidence={"document_id": "draft", "document_name": "章", "line_start": 1, "line_end": 1,
                  "text": "林澈将门敞开。"},
    )
    assert _assess_observation_match(current, raw).result is True
    for scope in ((), ("白天打扫时",), target.habit_contexts):
        supplied = target.model_copy(update={"habit_contexts": scope})
        cert = _target_bound_review_target(supplied, target_ordinal=1)
        if not scope:
            assert "habit_contexts" not in cert.model_dump()
        signed = CharacterSignal.model_validate({
            **raw.model_dump(), "target_bound_target": cert, "target_bound_target_ordinal": 1,
            "target_bound_target_digest": target_bound_draft_target_digest(cert),
            "target_bound_object_relation": "not_applicable",
        })
        assert (_matching_target(signed, (target,)) == target) == (scope == target.habit_contexts)
        assert _assess_observation_match(current, signed).result == (scope == target.habit_contexts)
    with pytest.raises(ValidationError):
        CharacterSignalTarget.model_validate({**target.model_dump(), "dimension": "speech_pattern"})
