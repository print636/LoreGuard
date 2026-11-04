"""A fragment can have a scope judgement without a directional observation."""
import json
from types import SimpleNamespace

import pytest

import app.character_consistency_stage as stage_module
from app.character_habit_scope_review import (
    HABIT_SCOPE_SYSTEM_PROMPT, _bind_outside, habit_scope_windows, review_habit_scope,
)
from app.narrative_context import payload_sha256
from tests.test_character_draft_actor_stage_wiring import (
    _ExtractionRecorder, _FrozenStage, _baseline_entry, _draft_source, _partial_result, _run, _settings,
)
from tests.test_habit_claim_recovery import HABIT, _source


def habit():
    row, original, scope, character_key = _baseline_entry()
    row.id = "frozen-trait-input"
    row.payload_sha256 = "7" * 64
    statement = "正式交付运箱时，祁雾总要查验每只运箱的铅封。"
    baseline = original.model_copy(update=dict(
        dimension="contextual_behavior", trait_key="crate_seals", statement=statement,
        contexts=("正式交付运箱时",), evidence=(original.evidence[0].model_copy(update={"text": statement}),),
    ))
    return row, baseline, scope, character_key


def windows(line, *, source=None, excluded_lines=None):
    _, baseline, _, _ = habit()
    target = HABIT.model_copy(update=dict(character=baseline.character, baseline_hint=baseline.statement,
                                          habit_contexts=baseline.contexts))
    return habit_scope_windows(target, source or _source(line), line, start=1, chunk_text=line,
                              excluded_lines=excluded_lines or set())


def answer(obj, use, i=1):
    return json.dumps({"outside": [dict(id=i, object_phrase=obj, purpose_quote=use)]}, ensure_ascii=False)


@pytest.mark.parametrize(("text", "obj", "use"), [
    ("清点道具时，祁雾把两只专供搭台的运箱搬进剧场。当天由剧团收回。", "两只专供搭台的运箱", "专供搭台"),
    ("清点道具时，祁雾拿出两只运箱供剧团排演。演出后仍由剧团收回。", "两只运箱", "供剧团排演"),
    ("清点道具时，祁雾把运箱递给演员作为临时坐凳。", "运箱", "作为临时坐凳"),
])
def test_non_directional_actor_and_purpose_bind_to_one_frozen_action(text, obj, use):
    event, = windows(text)
    ref, = _bind_outside(answer(obj, use), (event,), habit()[1])
    assert text[ref["action_start"]:ref["action_end"]] == event.action
    assert text[ref["purpose_start"]:ref["purpose_end"]] == use


@pytest.mark.parametrize("text", [
    "祁雾说：“我把专供搭台的运箱搬上了站台。”",
    "文森把专供搭台的运箱搬上了站台，祁雾只在一旁观看。",
    "祁雾本想把专供搭台的运箱搬上站台。",
    "祁雾把原本只供搭台的运箱送入库房。",
    "祁雾把专供搭台的运箱搬上站台。\n但这件事并未发生。",
    "剧本中发生了下面一幕。\n祁雾把专供搭台的运箱搬上站台。",
    "祁雾把运箱搬上站台，演员拿出另一个纸盒供剧团排演。",
])
def test_framed_or_other_person_or_other_objects_use_does_not_enter_review(text):
    assert not windows(text)


@pytest.mark.parametrize(("text", "obj", "use"), [
    ("祁雾把不只供搭台的运箱搬进库房。", "不只供搭台的运箱", "只供搭台"),
    ("祁雾把并排着专供搭台的运箱的模型搬上车。", "并排着专供搭台的运箱的模型", "专供搭台"),
    ("祁雾把专供搭台的纸盒搬进库房。", "专供搭台的纸盒", "专供搭台"),
    ("祁雾把装着专供搭台的运箱的小车推进库房。", "装着专供搭台的运箱的小车", "专供搭台"),
    ("祁雾把运箱和专供搭台的运箱搬进库房。", "运箱和专供搭台的运箱", "专供搭台"),
    ("祁雾把专供搭台的运箱搬进库房。", "专供搭台的运箱", "只供练习"),
    ("祁雾把专供搭台的运箱搬进库房。", "运箱", "专供搭台"),
    ("祁雾拿出运箱旁的模型供剧团排演。", "运箱", "供剧团排演"),
])
def test_an_assertive_model_cannot_move_purpose_or_replace_the_affected_object(text, obj, use):
    with pytest.raises(ValueError):
        _bind_outside(answer(obj, use), windows(text), habit()[1])


class ScopeProvider:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def complete(self, system, user):
        assert system == HABIT_SCOPE_SYSTEM_PROMPT
        self.calls.append(json.loads(user))
        return SimpleNamespace(text=self.response, prompt_tokens=9, completion_tokens=10)


def test_abstention_and_snapshot_change_and_insufficient_budget():
    text = "祁雾拿出两只运箱供剧团排演。"
    assert not windows(text, source=_source(text + "。"))
    assert not windows(text, excluded_lines={1})
    provider = ScopeProvider('{"outside":[]}')
    opts = dict(provider=provider, settings=_settings())
    assert review_habit_scope(windows(text), habit()[1], budget=1, **opts)[1].attempted_calls == 0
    refs, diagnostics = review_habit_scope(windows(text), habit()[1], budget=5000, **opts)
    assert not refs and diagnostics.reason == "undetermined" and diagnostics.attempted_calls == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_stage_adds_only_safe_excerpt_refs_after_an_unresolved_extraction(monkeypatch, enabled):
    line = "祁雾拿出两只运箱供剧团排演。"
    source = _draft_source(line)
    baseline = habit()
    provider = ScopeProvider(answer("两只运箱", "供剧团排演"))
    recorder = _ExtractionRecorder(primary_result=_partial_result(), targeted_results=[_partial_result()] * 3)
    monkeypatch.setattr(stage_module, "CharacterSignalExtractor", recorder.extractor_type())
    settings = _settings(character_signal_full_line_prompt_v2=True, character_draft_actor_review_v1=True)
    settings._character_habit_scope_review_v1 = enabled
    stage = _FrozenStage(source=source, baselines=[baseline], settings=settings)
    stage.provider = provider
    result = _run(stage, source)
    assert not result.issues and not result.review_clues and not result.provisional_draft_clues
    assert result.diagnostics["outcome"] == "partial"
    assert result.diagnostics["counts"]["draft_observation_count"] == 0
    refs = result.diagnostics.get("habit_scope_refs_v1", [])
    assert len(refs) == len(provider.calls) == int(enabled)
    if enabled:
        (ref,) = refs
        assert ref["ref_sha256"] == payload_sha256({k: v for k, v in ref.items() if k != "ref_sha256"})
        assert ref["draft_input_id"] == source.input_id and ref["baseline_input_id"] == baseline[0].id
        assert line[ref["purpose_start"]:ref["purpose_end"]] == "供剧团排演"
        assert "运箱" not in json.dumps(ref, ensure_ascii=False)
