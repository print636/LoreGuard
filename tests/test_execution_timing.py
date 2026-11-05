"""Shared execution time is a ceiling, not a new allowance at every call."""
import httpx
import pytest

from app.config import Settings, SELECTED_CLAIM_RESOLUTION_POLICY_V2
from app.execution_timing import (
    AnalysisDeadlineExceeded, ExecutionDeadline, execution_limits_payload,
    local_quality_timing_overrides, settings_for_execution_limits,
)
from app.provider import OpenAICompatibleProvider, ProviderError, RetryPolicy
from app.character_trait_extraction import _bounded_provider, CharacterSignalChunk


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def settings(**changes):
    return Settings(_env_file=None, **{
        "enable_model_extraction": True, "openai_base_url": "https://timing.invalid/v1",
        "openai_api_key": "unit-timing-placeholder", **local_quality_timing_overrides(), **changes,
    })


def reply():
    return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}],
                                     "usage": {"prompt_tokens": 3, "completion_tokens": 2}})


def provider(config, handler, clock):
    return OpenAICompatibleProvider(config, transport=httpx.MockTransport(handler),
        monotonic=clock, sleep=clock.advance,
        retry_policy=RetryPolicy(max_attempts=config.provider_max_attempts, base_delay_seconds=0, jitter_ratio=0))


def test_explicit_quality_profile_and_legacy_restore_do_not_change_permissions_or_attempts():
    old = Settings(_env_file=None)
    assert old.provider_timeout_seconds == 30 and old.provider_total_deadline_seconds is None
    assert execution_limits_payload(old) is None
    current = settings()
    assert current.character_signal_timeout_seconds == current.character_drift_timeout_seconds == 180
    assert current.character_habit_scene_total_deadline_seconds == 360
    assert current.character_signal_max_attempts == old.character_signal_max_attempts
    assert current.character_drift_max_attempts == old.character_drift_max_attempts
    assert current.character_draft_actor_review_max_attempts == old.character_draft_actor_review_max_attempts
    legacy = settings_for_execution_limits(current, None)
    assert legacy.provider_timeout_seconds == 30 and legacy.provider_total_deadline_seconds is None
    assert legacy.character_habit_scene_timeout_seconds == 60
    assert legacy.character_habit_scene_total_deadline_seconds == 180
    assert legacy.analysis_run_timeout_seconds is None and legacy.enable_model_extraction
    assert settings_for_execution_limits(old, None) is old
    snapshot = execution_limits_payload(current)
    tighter = settings(provider_timeout_seconds=90, character_draft_actor_review_timeout_seconds=45)
    restored = settings_for_execution_limits(tighter, snapshot)
    assert restored.provider_timeout_seconds == 90 and restored.character_draft_actor_review_timeout_seconds == 45
    for broken in ({**snapshot, "run_seconds": True}, {**snapshot, "logical_seconds": 999},
                   {**snapshot, "unexpected": 1}):
        with pytest.raises(ValueError):
            settings_for_execution_limits(current, broken)


def test_over_old_read_limit_can_succeed_without_an_extra_attempt():
    clock, seen = Clock(), []
    config = settings()
    config._execution_deadline = ExecutionDeadline.from_remaining(2400, monotonic=clock)
    def handler(request):
        seen.append(request.extensions["timeout"])
        clock.advance(80)
        return reply()
    gateway = provider(config, handler, clock)
    assert gateway.complete("s", "u").text == "{}"
    assert len(seen) == 1 and seen[0]["read"] == 180 and seen[0]["connect"] == 10
    assert config._execution_deadline.remaining_seconds() == 2320


def test_run_deadline_is_shared_across_calls_retries_and_character_forks():
    clock, caps = Clock(), []
    config = settings()
    config._execution_deadline = ExecutionDeadline.from_remaining(250, monotonic=clock)
    def handler(request):
        caps.append(request.extensions["timeout"]["read"])
        if len(caps) == 1:
            clock.advance(100)
            return reply()
        clock.advance(80 if len(caps) == 2 else request.extensions["timeout"]["read"])
        raise httpx.ReadTimeout("bounded test", request=request)
    gateway = provider(config, handler, clock)
    gateway.complete("first", "document")
    fork = _bounded_provider(gateway, config, stage="signal")
    assert fork.settings._execution_deadline is config._execution_deadline
    with pytest.raises(ProviderError):
        fork.complete("next", "stage")
    assert caps[:2] == [180, 150] and 0 < caps[2] < 70  # Existing retry backoff also consumes time.
    assert config._execution_deadline.remaining_seconds() == 0
    assert fork.last_telemetry is not None
    assert len(fork.last_telemetry.attempts) == 2
    with pytest.raises(AnalysisDeadlineExceeded):
        gateway.complete("never", "dispatch")
    assert len(caps) == 3
    assert "_execution_deadline" not in config.model_dump()


def test_logical_retry_budget_cannot_restart_before_run_deadline():
    clock, caps = Clock(), []
    config = settings(provider_max_attempts=3)
    config._execution_deadline = ExecutionDeadline.from_remaining(2400, monotonic=clock)
    def handler(request):
        caps.append(request.extensions["timeout"]["read"])
        clock.advance(180)
        raise httpx.ReadTimeout("bounded test", request=request)
    with pytest.raises(ProviderError):
        provider(config, handler, clock).complete("s", "u")
    assert caps == [180, 180]
    assert config._execution_deadline.remaining_seconds() == 2040


def test_smaller_transport_timeout_is_not_enlarged_by_character_fork():
    config = settings()
    gateway = provider(config.model_copy(update={"provider_timeout_seconds": 12}), lambda _: reply(), Clock())
    fork = _bounded_provider(gateway, config, stage="drift")
    assert fork.settings.provider_timeout_seconds == 12


def test_selection_and_independent_review_share_the_remaining_run_time():
    from app.character_action_selection import extract_selected_actions
    from tests.test_habit_action_selection import CONTENT, TARGET, SelectionProvider, settings as scene_settings, _source
    clock = Clock()
    config = scene_settings().model_copy(update=local_quality_timing_overrides())
    config._execution_deadline = ExecutionDeadline.from_remaining(250, monotonic=clock)
    class SlowSelection(SelectionProvider):
        def fork_for_habit_scene_review(self, **kwargs):
            self.review_limits = kwargs
            return super().fork_for_habit_scene_review(**kwargs)
        def complete(self, *args, **kwargs):
            clock.advance(80)
            return super().complete(*args, **kwargs)
    mock = SlowSelection()
    source = _source(CONTENT)
    result = extract_selected_actions(CharacterSignalChunk(source.document_id, "draft.md", CONTENT, 1, "draft"),
        TARGET, target_ordinal=1, source_identity=source, frozen_content=CONTENT,
        provider=mock, settings=config, monotonic=clock, checkpoint=config._execution_deadline.checkpoint)
    assert len(mock.calls) == 2 and len(result.signals) == 1
    assert mock.review_limits["timeout_seconds"] == 170
    assert mock.review_limits["remaining_deadline_seconds"] == 170
    assert config._execution_deadline.remaining_seconds() == 90


def test_original_claim_recheck_uses_remaining_time_without_another_opportunity():
    from tests.test_selected_claim_resolution import paired, reconcile
    pair = paired(policy=SELECTED_CLAIM_RESOLUTION_POLICY_V2, key_object="麦克风")
    *_, mock, old = pair
    clock = Clock()
    config = old.model_copy(update=local_quality_timing_overrides())
    config._execution_deadline = ExecutionDeadline.from_remaining(120, monotonic=clock)
    seen = []
    original = mock.fork_for_habit_scene_review
    def capture(**kwargs):
        seen.append(kwargs)
        return original(**kwargs)
    mock.fork_for_habit_scene_review = capture
    manager, _, admitted, _, _ = reconcile((*pair[:-1], config), checkpoint=config._execution_deadline.checkpoint)
    assert len(mock.resolution_requests) == len(manager.receipts) == len(admitted) == 1
    assert len(seen) == 1 and 0 < seen[0]["timeout_seconds"] <= 120
    assert 0 < seen[0]["remaining_deadline_seconds"] <= 120
