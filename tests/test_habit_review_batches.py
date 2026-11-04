"""Context-bound checks share a deadline and budget but each source window can finish."""
import json

import pytest

from app.character_draft_actor_review_provider import TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V12
from app.character_trait_extraction import (
    CharacterSignalChunk, _discover_target_bound_draft_candidates_v3, _review_target_bound_discoveries_v3,
)
from app.provider import ProviderError
from tests.test_character_target_bound_draft_review_v3 import _settings, _source
from tests.test_habit_claim_recovery import HABIT, ClaimProvider, extract, record


CONTENT = "廖辰把客人的行李牌塞进口袋。\n廖辰把乘客的登记单收进口袋。"


class WindowProvider(ClaimProvider):
    def __init__(self, fail_line=None, prompt_tokens=9):
        super().__init__([], source_support=True)
        self.fail_line = fail_line
        self.prompt_tokens = prompt_tokens
        self.clock = 0
        self.lines = []
        self.deadlines = []

    def fork_for_character_draft_actor_review(self, **limits):
        self.deadlines.append(limits["timeout_seconds"])
        return self

    def complete(self, system, user):
        assert system == TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V12
        windows = json.loads(user.split("：\n", 1)[1])["windows"]
        (window,) = windows
        (line,) = {p["line_start"] for p in window["request"]["proposals"]}
        self.lines.append(line)
        self.clock += 4
        if line == self.fail_line:
            raise ProviderError("timed out", category="read_timeout")
        response = super().complete(system, user)
        response.prompt_tokens = self.prompt_tokens
        return response


def run_windows(provider, *, budget=25_000, deadline=45):
    chunk = CharacterSignalChunk("draft-v3", "draft.md", CONTENT, 1, "draft")
    source = _source(CONTENT)
    discoveries, truncated = _discover_target_bound_draft_candidates_v3(
        chunk=chunk, target=HABIT, target_ordinal=1, source_identity=source,
        frozen_content=CONTENT, allowed_ranges=((1, 1), (2, 2)),
    )
    assert truncated == 0 and len(discoveries) == 2
    dispatched = set()
    result = _review_target_bound_discoveries_v3(
        chunk=chunk, target=HABIT, target_ordinal=1, discoveries=discoveries, source_identity=source,
        frozen_content=CONTENT, provider=provider, settings=_settings(), token_budget=budget,
        remaining_deadline_seconds=deadline, monotonic=lambda: provider.clock, dispatch_ranges=dispatched,
    )
    return result, dispatched


@pytest.mark.parametrize("bad,good", [(1, 2), (2, 1)])
def test_only_timed_out_window_is_unresolved(bad, good):
    provider = WindowProvider(fail_line=bad)
    result, dispatched = run_windows(provider)
    assert provider.lines == [1, 2] and result.attempted_calls == 2
    assert dispatched == {(1, 1), (2, 2)}
    assert [row.evidence.line_start for row in result.signals] == [good]
    assert dict(result.reason_counts) == {"target_bound_provider_timeout": 1}
    assert result.unresolved_count == 1 and result.rejected_count == 0
    assert result.prompt_tokens == 9 and result.completion_tokens == 8
    assert result.charged_tokens > result.prompt_tokens + result.completion_tokens


def test_a_completed_window_keeps_its_evidence_when_next_cannot_afford_admission():
    provider = WindowProvider(prompt_tokens=18_000)
    result, dispatched = run_windows(provider, budget=20_000)
    assert dispatched == {(1, 1)} and provider.lines == [1]
    assert [row.evidence.line_start for row in result.signals] == [1]
    assert result.attempted_calls == 1 and result.charged_tokens == 18_008
    assert result.unresolved_count == 1 and dict(result.reason_counts) == {"target_bound_token_budget": 1}


def test_second_window_only_gets_time_left_on_original_deadline():
    provider = WindowProvider()
    result, _ = run_windows(provider, deadline=6)
    assert provider.lines == [1, 2] and provider.deadlines == [6, 2]
    assert [row.evidence.line_start for row in result.signals] == [1]
    assert result.unresolved_count == 1 and dict(result.reason_counts) == {"target_bound_deadline": 1}


def test_local_contract_failure_in_second_window_preserves_completed_first(monkeypatch):
    from app import character_trait_extraction as signals

    original = signals.run_target_bound_draft_review

    def fail_oversized_window(entries, **kwargs):
        if entries[0].request.proposals[0].line_start == 2:
            raise ValueError("target_bound_batch_too_large")
        return original(entries, **kwargs)

    monkeypatch.setattr(signals, "run_target_bound_draft_review", fail_oversized_window)
    result, dispatched = run_windows(WindowProvider())
    assert dispatched == {(1, 1)}
    assert [row.evidence.line_start for row in result.signals] == [1]
    assert result.attempted_calls == result.unresolved_count == 1
    assert dict(result.reason_counts) == {"target_bound_request_invalid": 1}


def test_same_window_keeps_object_and_objectless_record_checks_independent():
    statement = "廖辰把行李牌装入口袋"
    result, provider = extract([record(statement), dict(record(statement), key_object="行李牌")], False)
    assert [(request["prompt_version"].rsplit("-", 1)[-1], request["primary_claim"]["record_ordinal"])
            for request in provider.requests] == [("v13", 1), ("v14", 2)]
    assert all(len(request["proposals"]) == 2 for request in provider.requests)
    reasons = result.diagnostics.reason_counts
    assert reasons["habit_claim_review_calls"] == reasons["habit_claim_review_supported"] == 2
    assert reasons["habit_object_review_calls"] == reasons["habit_object_review_supported"] == 1
