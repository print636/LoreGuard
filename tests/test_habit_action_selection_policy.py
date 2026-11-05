"""The frozen author choice authorizes each exact, bounded selection purpose."""
from types import SimpleNamespace

import pytest

from app.character_action_selection import (
    HABIT_ACTION_SELECTION_MAX_COMPLETION_TOKENS, HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1,
)
from app.character_draft_actor_review import SCOPED_DRAFT_DIRECTION_GUIDE
from app.character_draft_actor_review_provider import (
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16,
)
from app.config import HABIT_ACTION_SELECTION_POLICY_V1, HABIT_OBJECT_CHAIN_POLICY_V1
from app.db import AnalysisRunRow
from app.service import (
    CharacterConsistencyUsageAccumulator, _CharacterConsistencyAccountingProvider,
    _character_stage_settings_for_run,
)
from app.usage import estimate_issue_evidence_review_tokens
from tests.test_habit_scene_policy import LIMITS, local_run


def _stage(**updates):
    _, stage = local_run(coverage_changes={
        "habit_object_chain_policy": HABIT_OBJECT_CHAIN_POLICY_V1,
        "habit_action_selection_policy": HABIT_ACTION_SELECTION_POLICY_V1,
    })
    return stage.model_copy(update={
        "character_target_bound_draft_review_v3": True, "character_signal_full_line_prompt_v2": True,
        "character_signal_max_attempts": 1,
        "character_signal_max_completion_tokens": HABIT_ACTION_SELECTION_MAX_COMPLETION_TOKENS,
        **updates,
    })


class _RecordingProvider:
    def __init__(self, *, fail=False):
        self.sent = []
        self.fail = fail

    def complete(self, system, user):
        self.sent.append((system, user))
        if self.fail:
            raise RuntimeError("scripted provider failure")
        return SimpleNamespace(text="{}", prompt_tokens=9, completion_tokens=2)


@pytest.mark.parametrize(("changes", "enabled"), [
    ({}, False),
    ({"habit_action_selection_policy": HABIT_ACTION_SELECTION_POLICY_V1}, True),
    ({"habit_action_selection_policy": "unknown"}, False),
    ({"habit_action_selection_policy": HABIT_ACTION_SELECTION_POLICY_V1, "habit_object_chain_policy": None}, False),
    ({"habit_action_selection_policy": HABIT_ACTION_SELECTION_POLICY_V1, "habit_scene_policy": None}, False),
    ({"habit_action_selection_policy": HABIT_ACTION_SELECTION_POLICY_V1, "character_review_enabled": False}, False),
])
def test_selection_requires_its_frozen_version_and_existing_authorization(changes, enabled):
    base, stage = local_run(coverage_changes={"habit_object_chain_policy": HABIT_OBJECT_CHAIN_POLICY_V1, **changes})
    assert stage._character_habit_action_selection_v1 is enabled
    assert base._character_habit_action_selection_v1 is False
    assert "_character_habit_action_selection_v1" not in stage.model_dump()


@pytest.mark.parametrize("changes", [{"requested_by_user_id": "another-user"}, {"provider_config_id": "another-config"},
                                     {"batch_mode": "baseline_build"}])
def test_selection_policy_cannot_move_to_another_account_or_run_purpose(changes):
    _, stage = local_run(coverage_changes={
        "habit_object_chain_policy": HABIT_OBJECT_CHAIN_POLICY_V1,
        "habit_action_selection_policy": HABIT_ACTION_SELECTION_POLICY_V1,
    }, **changes)
    assert stage._character_habit_action_selection_v1 is False


def test_private_setting_override_cannot_upgrade_an_older_snapshot():
    base = _stage()
    run = AnalysisRunRow(sensitivity="balanced", batch_coverage={})
    assert not _character_stage_settings_for_run(base, run)._character_habit_action_selection_v1


def test_selector_requires_exact_purpose_authorization_and_single_bounded_call():
    stage = _stage()
    usage = CharacterConsistencyUsageAccumulator()
    transport = _RecordingProvider()

    def bounded(config):
        return _CharacterConsistencyAccountingProvider(
            config, usage, signal_provider=transport, drift_provider=transport,
        ).fork_for_character_consistency(settings=config, stage="signal", remaining_deadline_seconds=12)

    for private_flag in ("_character_habit_action_selection_v1", "_character_habit_scene_scheduling_v1",
                         "_character_habit_object_chain_v1"):
        unauthorized = stage.model_copy()
        setattr(unauthorized, private_flag, False)
        with pytest.raises(RuntimeError, match="unsupported"):
            bounded(unauthorized).complete(HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1, "{}")
    for prompt in (HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1 + "x",
                   HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1 + SCOPED_DRAFT_DIRECTION_GUIDE):
        with pytest.raises(RuntimeError, match="unsupported"):
            bounded(stage).complete(prompt, "{}")
    for config in (stage.model_copy(update={"character_signal_max_attempts": 2}),
                   stage.model_copy(update={"character_signal_max_completion_tokens": 769})):
        with pytest.raises(RuntimeError, match="bounded provider"):
            bounded(config).complete(HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1, "{}")
    assert not transport.sent and usage.logical_calls == 0
    bounded(stage).complete(HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1, "{}")
    assert transport.sent == [(HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1, "{}")]
    assert usage.logical_calls == usage.successful_calls == 1
    assert usage.charged_tokens == estimate_issue_evidence_review_tokens(
        HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1, "{}", completion_reserve=768,
    )


def test_selector_failure_is_charged_once_without_an_unmetered_retry():
    stage = _stage()
    usage = CharacterConsistencyUsageAccumulator()
    transport = _RecordingProvider(fail=True)
    accounting = _CharacterConsistencyAccountingProvider(stage, usage, signal_provider=transport, drift_provider=transport)
    with pytest.raises(RuntimeError, match="scripted provider failure"):
        accounting.complete(HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1, "{}")
    assert len(transport.sent) == usage.logical_calls == 1
    assert usage.successful_calls == 0
    assert usage.charged_tokens == estimate_issue_evidence_review_tokens(
        HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1, "{}", completion_reserve=768,
    )


@pytest.mark.parametrize(("outer_cap", "expected"), [(None, 768), (128, 128)])
def test_selector_transport_keeps_existing_signal_and_outer_limits(outer_cap, expected):
    stage = _stage(provider_max_completion_tokens=outer_cap)
    accounting = _CharacterConsistencyAccountingProvider(stage, CharacterConsistencyUsageAccumulator())
    bounded = accounting.fork_for_character_consistency(settings=stage, stage="signal", remaining_deadline_seconds=12)
    config = bounded.signal_provider.settings
    assert config.provider_max_completion_tokens == expected
    assert config.provider_max_attempts == bounded.signal_provider.retry_policy.max_attempts == 1
    assert config.provider_total_deadline_seconds <= 12


def test_v16_verification_requires_selection_authorization_and_keeps_v15_available():
    stage = _stage()
    usage = CharacterConsistencyUsageAccumulator()
    transport = _RecordingProvider()

    def bounded(config):
        return _CharacterConsistencyAccountingProvider(
            config, usage, signal_provider=transport, drift_provider=transport,
        ).fork_for_habit_scene_review(**{**LIMITS, "completion_reserve": 4096})

    for private_flag in ("_character_habit_action_selection_v1", "_character_habit_object_chain_v1"):
        unauthorized = stage.model_copy()
        setattr(unauthorized, private_flag, False)
        with pytest.raises(RuntimeError, match="unsupported"):
            bounded(unauthorized).complete(TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16, "{}")
    for prompt in (TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16 + "x",
                   TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16 + SCOPED_DRAFT_DIRECTION_GUIDE):
        with pytest.raises(RuntimeError, match="unsupported"):
            bounded(stage).complete(prompt, "{}")
    assert not transport.sent and usage.logical_calls == 0
    call = bounded(stage)
    assert call.draft_actor_review_completion_reserve == 4096
    call.complete(TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16, "{}")
    legacy = stage.model_copy()
    legacy._character_habit_action_selection_v1 = False
    bounded(legacy).complete(TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15, "{}")
    assert [system for system, _ in transport.sent] == [TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16,
                                                     TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15]
    assert usage.logical_calls == usage.successful_calls == 2
    assert usage.charged_tokens == sum(estimate_issue_evidence_review_tokens(
        system, "{}", completion_reserve=4096,
    ) for system, _ in transport.sent)
