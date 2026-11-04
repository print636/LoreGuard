"""Only an opted-in author draft runs the longer scene checks."""
import pytest

from app.config import HABIT_SCENE_POLICY_V1, HABIT_SCOPE_POLICY_V1, Settings
from app.db import AnalysisRunRow
from app.service import (
    CharacterConsistencyUsageAccumulator, _CharacterConsistencyAccountingProvider,
    _character_stage_settings_for_run,
)


class AuthorSettings(Settings):
    character_review_requires_opt_in = True
    habit_scene_policy_available = True


def local_run(*, settings_cls=AuthorSettings, total_deadline=None, coverage_changes=None, **run_changes):
    settings = settings_cls(
        _env_file=None, enable_character_consistency=True, character_draft_actor_review_v1=True,
        provider_runtime_config_id="account-config", provider_runtime_user_id="author",
        provider_total_deadline_seconds=total_deadline,
    )
    coverage = {
        "character_review_enabled": True, "character_baseline_run_id": "baseline",
        "habit_scene_policy": HABIT_SCENE_POLICY_V1, **(coverage_changes or {}),
    }
    run = AnalysisRunRow(**{
        "sensitivity": "balanced", "batch_mode": "draft_review", "batch_coverage": coverage,
        "provider_config_id": "account-config", "requested_by_user_id": "author", **run_changes,
    })
    return settings, _character_stage_settings_for_run(settings, run)


LIMITS = dict(timeout_seconds=60, remaining_deadline_seconds=110, completion_reserve=512,
              max_response_bytes=4096, max_attempts=1)


@pytest.mark.parametrize(("total_deadline", "expected"), [(None, 60), (12, 12)])
def test_local_habit_transport_has_its_own_bound_and_honors_outer_deadline(total_deadline, expected):
    base, stage = local_run(total_deadline=total_deadline)
    assert stage._character_habit_scene_scheduling_v1 and not base._character_habit_scene_scheduling_v1
    accounting = _CharacterConsistencyAccountingProvider(stage, CharacterConsistencyUsageAccumulator())
    transport = accounting.fork_for_habit_scene_review(**LIMITS).draft_actor_review_provider.settings
    normal = accounting.fork_for_character_draft_actor_review(**LIMITS).draft_actor_review_provider.settings
    assert transport.provider_timeout_seconds == transport.provider_total_deadline_seconds == expected
    assert transport.provider_max_attempts == 1 and transport.provider_max_response_bytes == 4096
    assert normal.provider_timeout_seconds == min(expected, 30)
    assert base.provider_timeout_seconds == stage.provider_timeout_seconds == 30


@pytest.mark.parametrize("changes", [
    {"settings_cls": Settings},
    {"batch_coverage": {"character_review_enabled": True, "character_baseline_run_id": "baseline"}},
    {"coverage_changes": {"character_review_enabled": False}},
    {"coverage_changes": {"habit_scene_policy": "unknown"}},
    {"coverage_changes": {"no_formal_context_expected": True}},
    {"batch_mode": "baseline_build"}, {"provider_config_id": None}, {"requested_by_user_id": "other"},
])
def test_older_snapshots_and_ineligible_runs_cannot_use_the_new_fork(changes):
    _, stage = local_run(**changes)
    assert not stage._character_habit_scene_scheduling_v1
    accounting = _CharacterConsistencyAccountingProvider(stage, CharacterConsistencyUsageAccumulator())
    with pytest.raises(ValueError, match="disabled"):
        accounting.fork_for_habit_scene_review(**LIMITS)


@pytest.mark.parametrize(("changes", "enabled"), [
    ({}, False),  # Earlier scene scheduling policy, with no scope check frozen.
    ({"habit_scope_policy": HABIT_SCOPE_POLICY_V1}, True),
    ({"habit_scope_policy": HABIT_SCOPE_POLICY_V1, "character_review_enabled": False}, False),
    ({"habit_scope_policy": "unknown"}, False),
])
def test_habit_scope_review_requires_its_own_frozen_version(changes, enabled):
    base, stage = local_run(coverage_changes=changes)
    assert stage._character_habit_scope_review_v1 is enabled
    assert base._character_habit_scope_review_v1 is False


def test_habit_scope_call_requires_explicit_consent_exact_purpose_and_one_attempt():
    from types import SimpleNamespace
    from app.character_habit_scope_review import HABIT_SCOPE_SYSTEM_PROMPT

    _, stage = local_run(coverage_changes={"habit_scope_policy": HABIT_SCOPE_POLICY_V1})
    sent = []

    class RecordingProvider:
        def complete(self, system, user):
            sent.append((system, user))
            return SimpleNamespace(text="{}", prompt_tokens=9, completion_tokens=2)

    usage = CharacterConsistencyUsageAccumulator()
    provider = RecordingProvider()
    accounting = _CharacterConsistencyAccountingProvider(stage, usage, signal_provider=provider, drift_provider=provider)
    with pytest.raises(RuntimeError, match="bounded provider"):
        accounting.complete(HABIT_SCOPE_SYSTEM_PROMPT, "{}")

    def bounded(config):
        limited = config.model_copy(update={"character_drift_max_attempts": 1, "character_drift_max_completion_tokens": 512})
        return accounting.fork_for_character_consistency(settings=limited, stage="drift")
    _, legacy = local_run()
    with pytest.raises(RuntimeError, match="unsupported"):
        bounded(legacy).complete(HABIT_SCOPE_SYSTEM_PROMPT, "{}")
    with pytest.raises(RuntimeError, match="unsupported"):
        bounded(stage).complete(HABIT_SCOPE_SYSTEM_PROMPT + "x", "{}")
    assert not sent and usage.logical_calls == 0
    bounded(stage).complete(HABIT_SCOPE_SYSTEM_PROMPT, "{}")
    assert len(sent) == usage.logical_calls == 1
    assert sent[0] == (HABIT_SCOPE_SYSTEM_PROMPT, "{}")
