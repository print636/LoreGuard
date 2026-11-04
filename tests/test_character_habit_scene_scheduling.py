"""An opted-in slow reviewer shares a finite package deadline across manuscript windows."""
import json

import pytest

import app.character_trait_extraction as signals
from app.character_draft_actor_review_provider import (
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V14,
)
from tests.test_character_target_bound_draft_review_v3 import _V3Provider, _settings, _signal_target, _source
from tests.test_habit_claim_recovery import ClaimProvider, HABIT, SOURCE, record


class TimedClaimProvider(ClaimProvider):
    def __init__(self, records, *, generation_seconds, review_seconds, charge_first):
        super().__init__(records)
        self.clock = 0
        self.generation_seconds = generation_seconds
        self.review_seconds = review_seconds
        self.charge_first = charge_first
        self.order = []
        self.timeouts = []

    def fork_for_habit_scene_review(self, **limits):
        self.timeouts.append(("account", limits["timeout_seconds"]))
        return self

    def fork_for_character_draft_actor_review(self, **limits):
        self.timeouts.append(("normal", limits["timeout_seconds"]))
        return self

    def complete(self, system, user):
        response = super().complete(system, user)
        if system in {TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V14}:
            windows = json.loads(user.split("：\n", 1)[1])["windows"]
            (window,) = windows
            req = window["request"]
            self.order.append((req["proposals"][0]["line_start"], req["primary_claim"]["record_ordinal"],
                               len(req["proposals"])))
            self.clock += self.review_seconds
            if self.charge_first and len(self.order) == 1:
                response.prompt_tokens = self.charge_first
        else:
            self.clock += self.generation_seconds
        return response


def run_package(*, enabled, generation_seconds=0, review_seconds=0, deadline=None, charge_first=0,
                windows=(1, 1, 2), targeted=False):
    lines = [SOURCE.replace("客人", f"第{i}位客人") for i in range(1, max(windows) + 1)]
    text = "\n".join(lines)
    rows = [dict(record("廖辰把行李牌装入口袋"), key_object="行李牌" if i == 2 else "",
                 evidence=lines[n-1], source_line_start=n, source_line_end=n) for i, n in enumerate(windows, 1)]
    provider = TimedClaimProvider(rows, generation_seconds=generation_seconds, review_seconds=review_seconds,
                                  charge_first=charge_first)
    settings = _settings(provider_total_deadline_seconds=deadline, character_signal_token_budget=40_000)
    settings._character_habit_scene_scheduling_v1 = enabled
    extractor = signals.CharacterSignalExtractor(provider, settings=settings)
    extractor._monotonic = lambda: provider.clock
    chunk = signals.CharacterSignalChunk("draft-v3", "draft.md", text, 1, "draft")
    kwargs = dict(source_identity=_source(text), frozen_content=text)
    if targeted:
        result = extractor.extract_targeted(chunk, (HABIT,), target_ordinal=1, **kwargs)
    else:
        result = extractor.extract(chunk, primary_value_targets=(HABIT,), **kwargs)
    return provider, result


@pytest.mark.parametrize("targeted", [False, True])
def test_first_turn_per_source_window_preserves_complete_record_and_targeted_package_clock(targeted):
    provider, result = run_package(enabled=True, generation_seconds=20, review_seconds=40, targeted=targeted)
    assert provider.order == [(1, 1, 2), (2, 3, 2), (1, 2, 2)]
    assert provider.timeouts == [("account", 60)] * 3
    assert provider.clock == 140 and result.diagnostics.attempted_calls == 4
    assert {row.evidence.line_start for row in result.draft_observations} == {1, 2}


def test_profile_off_retains_original_record_order_and_regular_timeouts():
    provider, _ = run_package(enabled=False)
    assert provider.order == [(1, 1, 2), (1, 2, 2), (2, 3, 2)]
    assert provider.timeouts == [("normal", 30)] * 3


def test_provider_deadline_limits_first_wave_and_early_returns_release_capacity():
    provider, result = run_package(enabled=True, generation_seconds=20, review_seconds=10, deadline=96)
    assert provider.timeouts == [("account", 38), ("account", 60), ("account", 56)]
    assert result.diagnostics.reason_counts["habit_claim_recovery_supported"] == 3


def test_short_tail_can_serve_one_window_but_is_not_sliced_between_remaining_scenes():
    provider, result = run_package(enabled=True, review_seconds=16, deadline=24, windows=(1, 2, 3))
    assert provider.timeouts == [("account", 24)]
    assert result.diagnostics.attempted_calls == 2  # Primary extraction plus one actual receipt.
    assert result.diagnostics.reason_counts["target_bound_scene_time_insufficient"] == 2
    assert [row.evidence.line_start for row in result.draft_observations] == [1]
    assert result.diagnostics.outcome == "partial"


def test_single_window_can_use_an_eighteen_second_tail():
    provider, result = run_package(enabled=True, review_seconds=12, deadline=18, windows=(1,))
    assert provider.timeouts == [("account", 18)]
    assert result.diagnostics.outcome == "completed"


def test_round_robin_still_charges_actual_usage_before_next_admission():
    provider, result = run_package(enabled=True, charge_first=30_000)
    assert provider.order == [(1, 1, 2)]
    assert result.diagnostics.reason_counts["target_bound_token_budget"] == 2
    assert result.diagnostics.attempted_calls == 2 and result.diagnostics.outcome == "partial"
    assert [row.evidence.line_start for row in result.draft_observations] == [1]


def test_other_dimensions_keep_regular_transport_and_sixty_second_package_even_in_account_mode():
    class RegularProvider(_V3Provider):
        def fork_for_character_draft_actor_review(self, **limits):
            self.review_timeout = limits["timeout_seconds"]
            return self

        def fork_for_habit_scene_review(self, **limits):
            raise AssertionError("the preference protocol cannot spend a habit scene allowance")

    provider = RegularProvider()
    text = "沈砚明确表示自己讨厌栗子糕。"
    settings = _settings()
    settings._character_habit_scene_scheduling_v1 = True
    extractor = signals.CharacterSignalExtractor(provider, settings=settings)
    extractor._monotonic = lambda: 50.0 if provider.signal_calls else 0.0
    result = extractor.extract_targeted(signals.CharacterSignalChunk("draft-v3", "draft.md", text, 1, "draft"),
                                         (_signal_target(),), target_ordinal=1, source_identity=_source(text),
                                         frozen_content=text)
    assert result.diagnostics.outcome == "completed" and provider.review_timeout == 10
