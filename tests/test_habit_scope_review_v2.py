"""Applicability explanations are independent of directional observations."""
import copy
import json
from types import SimpleNamespace

import pytest

import app.character_consistency_stage as stage_module
from app.character_habit_scope_review_v2 import (
    HABIT_SCOPE_SCHEMA_V2, HABIT_SCOPE_SYSTEM_PROMPT_V2,
    habit_scope_windows_v2, habit_scope_request_v2, bind_habit_scope_response_v2, review_habit_scope_v2,
)
from app.config import HABIT_SCOPE_POLICY_V1, HABIT_SCOPE_POLICY_V2
from app.narrative_context import payload_sha256
from tests.test_habit_scope_review import habit
from tests.test_habit_claim_recovery import HABIT, _source
from tests.test_character_draft_actor_stage_wiring import (
    _ExtractionRecorder, _FrozenStage, _draft_source, _partial_result, _run, _settings,
)


LINE = "祁雾给学徒演示运箱的装卸，练完便把运箱拆开，好让他们看清内部结构。"
ACTION = "练完便把运箱拆开"
OBJECT = "运箱"
PURPOSE = "好让他们看清内部结构"


def windows(text=LINE, *, excluded=(), source=None):
    baseline = habit()[1]
    target = HABIT.model_copy(update=dict(character=baseline.character, baseline_hint=baseline.statement,
                                          habit_contexts=baseline.contexts))
    return habit_scope_windows_v2(target, source or _source(text), text, start=1, chunk_text=text,
                                 excluded_lines=set(excluded))


def answer(request, *, action=ACTION, obj=OBJECT, purpose=PURPOSE, **changes):
    return dict(schema_version=HABIT_SCOPE_SCHEMA_V2, request_digest=request["request_digest"], items=[{
        "id": row["id"], "actor": "target", "actuality": "actual", "object_relation": "related",
        "use_relation": "same_object", "correction": "none", "action_quote": action,
        "object_quote": obj, "purpose_quote": purpose,
        "contexts": [dict(index=i, relation="outside", basis_quote=purpose)
                     for i, _ in enumerate(request["request"]["contexts"])], **changes,
    } for row in request["request"]["windows"]])


def bind(response, current=None, baseline=None):
    current, baseline = current or windows(), baseline or habit()[1]
    request = habit_scope_request_v2(current, baseline)
    return bind_habit_scope_response_v2(json.dumps(response, ensure_ascii=False), current, baseline,
                                       request_digest=request["request_digest"])


def test_non_template_use_and_omitted_actor_can_get_a_fragment_explanation():
    current = windows()
    request = habit_scope_request_v2(current, habit()[1])
    ref, = bind(answer(request), current)
    assert LINE[ref["action_start"]:ref["action_end"]] == ACTION + "，" + PURPOSE
    assert LINE[ref["purpose_start"]:ref["purpose_end"]] == PURPOSE
    assert "source" in request["request"] and request["request"]["windows"][0]["text"] == LINE
    assert not any(k in request["request"] for k in ("polarity", "primary_claim", "verdict"))


@pytest.mark.parametrize("changes", [
    {"actor": "other"}, {"actor": "uncertain"}, {"actuality": "nonactual"},
    {"object_relation": "different"}, {"use_relation": "different_object"}, {"correction": "corrected"},
    {"contexts": [dict(index=0, relation="inside", basis_quote=PURPOSE)]},
    {"contexts": [dict(index=0, relation="uncertain", basis_quote=None)]},
])
def test_negative_or_uncertain_semantic_slots_never_become_outside(changes):
    request = habit_scope_request_v2(windows(), habit()[1])
    assert bind(answer(request, **changes)) == []


@pytest.mark.parametrize("damage", ["source", "digest", "context", "missing_context", "missing_quote", "duplicate_id"])
def test_source_and_complete_context_set_are_required(damage):
    baseline = habit()[1].model_copy(update={"contexts": ("正式交付运箱时", "运箱移交验收时")})
    request = habit_scope_request_v2(windows(), baseline)
    response = answer(request)
    if damage == "source":
        response["items"][0]["purpose_quote"] = "另一个没有给出的场景"
    elif damage == "digest":
        response["request_digest"] = "a" * 64
    elif damage == "context":
        response["items"][0]["contexts"].reverse()
    elif damage == "missing_context":
        response["items"][0]["contexts"].pop()
    elif damage == "missing_quote":
        response["items"][0]["object_quote"] = None
    else:
        response["items"].append(copy.deepcopy(response["items"][0]))
    with pytest.raises(ValueError):
        bind(response, baseline=baseline)


def test_all_contexts_must_be_outside_and_repeated_objects_bind_locally():
    baseline = habit()[1].model_copy(update={"contexts": ("正式交付运箱时", "运箱移交验收时")})
    request = habit_scope_request_v2(windows(), baseline)
    response = answer(request)
    assert len(bind(response, baseline=baseline)) == 1  # 运箱 occurs twice, once in the selected action.
    response["items"][0]["contexts"][1]["relation"] = "uncertain"
    assert bind(response, baseline=baseline) == []
    response = answer(request, action=LINE)
    with pytest.raises(ValueError, match="ambiguous"):
        bind(response, baseline=baseline)


def test_windows_preserve_negation_other_actor_and_context_for_the_model():
    text = "此前有一份未经证实的传言。\n祁雾与文森清点道具，他把运箱拆开，好让学徒看清内部结构。\n后来核实这是文森的动作。"
    current, = windows(text)
    assert current.before == "此前有一份未经证实的传言。" and "文森" in current.after
    assert current.line == 2 and "他把" in current.text
    request = habit_scope_request_v2((current,), habit()[1])
    assert bind(answer(request, actor="uncertain", action=None, obj=None, purpose=None), (current,)) == []
    assert not windows(text, excluded={2})
    assert not windows(text, source=_source(text + "改动"))


def test_window_bound_does_not_claim_unselected_lines_are_reviewed():
    current = windows("\n".join([LINE] * 3))
    assert [w.line for w in current] == [1, 2]


class ScopeV2Provider:
    def __init__(self, *, changes=None):
        self.calls, self.forks, self.changes = [], [], changes or {}

    def fork_for_character_consistency(self, *, settings, stage, remaining_deadline_seconds=None):
        self.forks.append((stage, settings.character_drift_max_completion_tokens, settings.character_drift_max_attempts))
        return self

    def complete(self, system, user):
        assert system == HABIT_SCOPE_SYSTEM_PROMPT_V2
        request = json.loads(user)
        self.calls.append(request)
        return SimpleNamespace(text=json.dumps(answer(request, **self.changes), ensure_ascii=False),
                               prompt_tokens=19, completion_tokens=23)


def configured(**kwargs):
    result = _settings(**kwargs)
    result._character_habit_scope_review_policy = HABIT_SCOPE_POLICY_V2
    return result


def test_budget_small_limit_and_disabled_never_add_calls():
    provider = ScopeV2Provider()
    current = windows()
    baseline = habit()[1]
    config = configured(character_drift_max_completion_tokens=384)
    assert review_habit_scope_v2(current, baseline, provider=provider, settings=config, budget=1)[1].attempted_calls == 0
    refs, diagnostics = review_habit_scope_v2(current, baseline, provider=provider, settings=config, budget=10000)
    assert len(refs) == len(provider.calls) == 1 and diagnostics.attempted_calls == 1
    assert provider.forks == [("drift", 384, 1)] and diagnostics.prompt_tokens == 19
    config._character_habit_scope_review_policy = None
    assert review_habit_scope_v2(current, baseline, provider=provider, settings=config, budget=10000)[1].reason == "disabled"
    assert len(provider.calls) == 1


def test_stage_explains_excerpt_without_observation_or_debt_resolution(monkeypatch):
    source, baseline = _draft_source(LINE), habit()
    recorder = _ExtractionRecorder(primary_result=_partial_result(), targeted_results=[_partial_result()] * 3)
    monkeypatch.setattr(stage_module, "CharacterSignalExtractor", recorder.extractor_type())
    stage = _FrozenStage(source=source, baselines=[baseline], settings=configured(
        character_signal_full_line_prompt_v2=True, character_draft_actor_review_v1=True,
    ))
    stage.provider = ScopeV2Provider()
    result = _run(stage, source)
    assert not result.issues and not result.review_clues and not result.provisional_draft_clues
    assert result.diagnostics["outcome"] == "partial" and result.diagnostics["counts"]["draft_observation_count"] == 0
    ref, = result.diagnostics["habit_scope_refs_v1"]
    assert len(stage.provider.calls) == 1
    assert ref["draft_input_id"] == source.input_id and ref["baseline_input_id"] == baseline[0].id
    assert ref["ref_sha256"] == payload_sha256({k: v for k, v in ref.items() if k != "ref_sha256"})
    assert "运箱" not in json.dumps(ref, ensure_ascii=False)
    assert not result.diagnostics.get("claim_resolution_v1", {}).get("receipts")


@pytest.mark.parametrize("policy", [None, HABIT_SCOPE_POLICY_V1, HABIT_SCOPE_POLICY_V2, "future"])
def test_frozen_policy_and_exact_purpose_are_exclusive(policy):
    from tests.test_habit_scene_policy import local_run
    from app.character_habit_scope_review import HABIT_SCOPE_SYSTEM_PROMPT
    from app.service import CharacterConsistencyUsageAccumulator, _CharacterConsistencyAccountingProvider
    base, config = local_run(coverage_changes={"habit_scope_policy": policy})
    assert config._character_habit_scope_review_v1 is (policy == HABIT_SCOPE_POLICY_V1)
    assert config._character_habit_scope_review_policy == (HABIT_SCOPE_POLICY_V2 if policy == HABIT_SCOPE_POLICY_V2 else None)
    assert base._character_habit_scope_review_policy is None
    sent = []

    class Provider:
        def complete(self, system, user):
            sent.append(system)
            return SimpleNamespace(text="{}", prompt_tokens=9, completion_tokens=3)

    limited = config.model_copy(update={"character_drift_max_completion_tokens": 512, "character_drift_max_attempts": 1})
    usage = CharacterConsistencyUsageAccumulator()
    provider = _CharacterConsistencyAccountingProvider(limited, usage, drift_provider=Provider())
    for system, expected in ((HABIT_SCOPE_SYSTEM_PROMPT, HABIT_SCOPE_POLICY_V1), (HABIT_SCOPE_SYSTEM_PROMPT_V2, HABIT_SCOPE_POLICY_V2)):
        if policy == expected:
            provider.complete(system, "{}")
        else:
            with pytest.raises(RuntimeError, match="unsupported"):
                provider.complete(system, "{}")
    assert len(sent) == usage.logical_calls == int(policy in (HABIT_SCOPE_POLICY_V1, HABIT_SCOPE_POLICY_V2))


def test_v2_rejects_unbounded_accounting_call_and_wrong_account():
    from tests.test_habit_scene_policy import local_run
    from app.service import CharacterConsistencyUsageAccumulator, _CharacterConsistencyAccountingProvider
    _, config = local_run(coverage_changes={"habit_scope_policy": HABIT_SCOPE_POLICY_V2})
    with pytest.raises(RuntimeError, match="bounded"):
        _CharacterConsistencyAccountingProvider(config, CharacterConsistencyUsageAccumulator()).complete(HABIT_SCOPE_SYSTEM_PROMPT_V2, "{}")
    _, wrong = local_run(coverage_changes={"habit_scope_policy": HABIT_SCOPE_POLICY_V2}, requested_by_user_id="other")
    assert wrong._character_habit_scope_review_policy is None


@pytest.mark.parametrize(("text", "actor", "expected"), [
    ("周宁在门边清点，林砚把新送来的麦克风摆到台口。", "target", "supported"),
    ("林砚在门边清点，周宁把新送来的麦克风摆到台口。", "other", "uncertain"),
    ("林砚与周宁清点道具，她把新送来的麦克风摆到台口。", "uncertain", "uncertain"),
])
def test_current_selected_review_retains_multiple_actor_boundaries(text, actor, expected):
    # This tests source/slot handling with controlled model decisions, not
    # whether a real model resolves the pronoun correctly.
    from tests.test_habit_action_selection import selection, choice_response, evaluate
    from tests.test_character_target_bound_draft_review_v4 import _item
    from app.character_action_selection import build_selected_action_review_request, selected_action_path
    source, selected, _ = selection(text)
    request = build_selected_action_review_request(selected, choice_response(selected), frozen_content=text)
    proposal, = request.proposals
    path = selected_action_path(request, proposal)
    item = _item(request, proposal, object_relation="not_applicable", observation_kind="action")
    item["action_steps"] = [dict(clause_id=c.support_id, quote=c.text, actor="target", actuality="actual", scene="same") for c in path]
    item["action_steps"][-1]["actor"] = actor
    item["claim_object"] = dict(role="event_object", grounding="explicit", referent_relation="same",
                                fact_quote="新送来的麦克风", antecedent_clause_id=None, antecedent_quote=None)
    if actor == "uncertain":
        item.update(verdict="uncertain", actor="ambiguous")
    result, = evaluate(request, source, [item], content=text).decisions
    assert result.verdict == expected
