"""An inconclusive paraphrase is distinct from a new, independently verified source action."""
import json

import pytest

from app.character_draft_actor_review_provider import (
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V12, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13,
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V14,
)
from app.character_trait_extraction import CharacterSignalChunk, CharacterSignalExtractor
from tests.test_character_target_bound_draft_review_v3 import _settings, _source
from tests.test_habit_claim_recovery import ClaimProvider, HABIT, SOURCE, record


class AmbiguousClaimProvider(ClaimProvider):
    def __init__(self, records, *, mode, source_support=True):
        super().__init__(records, source_support=source_support)
        self.mode = mode
        self.systems = []
        self.clock = 0

    def complete(self, system, user):
        self.systems.append(system)
        response = super().complete(system, user)
        if system not in {TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V14}:
            return response
        if self.mode == "bad_response":
            response.text = "{}"
            return response
        data = json.loads(response.text)
        for window in data["responses"]:
            for item in window["items"]:
                if self.mode == "resolved_sibling" and system == TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13:
                    continue
                if self.mode == "opposite":
                    item.update(verdict="rejected", claim_relation="different", polarity_relation="opposite")
                elif item["verdict"] == "rejected":
                    # The original claim does not map uniquely. This first fact also
                    # has a contrary axis and must not reappear in the source request.
                    item["claim_relation"] = "uncertain"
                elif self.mode == "bad_citation":
                    item["basis_ids"] = []
                elif self.mode == "bad_object":
                    item["claim_object"]["fact_quote"] = "不在原文"
                elif self.mode == "budget":
                    response.prompt_tokens = 1_000_000
                elif self.mode == "deadline":
                    self.clock = 10_000
        response.text = json.dumps(data)
        return response


def extract(mode="ambiguous", *, targeted=True, with_object=True, source_support=True, duplicate=False):
    statement = "廖辰把行李牌装入口袋"
    row = dict(record(statement), key_object="行李牌" if with_object else "")
    rows = [record(statement), row] if duplicate else [row]
    provider = AmbiguousClaimProvider(rows, mode=mode, source_support=source_support)
    extractor = CharacterSignalExtractor(provider, settings=_settings(character_signal_package_max_attempts=2))
    extractor._monotonic = lambda: provider.clock
    chunk = CharacterSignalChunk("draft-v3", "draft.md", SOURCE, 1, "draft")
    kwargs = dict(source_identity=_source(SOURCE), frozen_content=SOURCE)
    result = (extractor.extract_targeted(chunk, (HABIT,), target_ordinal=1, **kwargs) if targeted else
              extractor.extract(chunk, primary_value_targets=(HABIT,), **kwargs))
    return result, provider


@pytest.mark.parametrize("with_object", [True, False])
def test_ambiguous_claim_gets_one_new_source_check_without_clearing_original(with_object):
    result, provider = extract(with_object=with_object)
    claim, source = provider.requests
    assert claim["primary_claim"] and "primary_claim" not in source
    assert source["prompt_version"].endswith("-v12") and len(source["proposals"]) == 1
    assert "口袋" in source["proposals"][0]["canonical_statement"]
    reasons = result.diagnostics.reason_counts
    assert reasons["habit_after_claim_source_calls"] == reasons["habit_after_claim_source_supported"] == 1
    assert reasons["habit_claim_recovery_calls"] == reasons["habit_claim_recovery_unresolved"] == 1
    assert "habit_claim_recovery_supported" not in reasons and provider.generations == 1
    assert result.diagnostics.outcome == "partial" and result.diagnostics.rejected_records == 1
    assert result.diagnostics.attempted_calls == 3
    assert result.diagnostics.prompt_tokens == 27 and result.diagnostics.completion_tokens == 24
    signal, = result.draft_observations
    assert signal.evidence.text == SOURCE and "行李牌" in signal.statement


@pytest.mark.parametrize("mode", ["bad_citation", "bad_object", "bad_response", "opposite", "budget", "deadline"])
def test_hard_negative_contract_or_resource_failure_never_yields_source_observations(mode):
    result, provider = extract(mode)
    assert TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V12 not in provider.systems
    assert result.diagnostics.outcome == "partial" and not result.draft_observations
    assert not result.diagnostics.reason_counts.get("habit_after_claim_source_calls")


def test_existing_success_on_window_and_primary_phase_do_not_request_source_again():
    (sibling_result, sibling_provider), (primary_result, primary_provider) = (
        extract("resolved_sibling", duplicate=True), extract(targeted=False)
    )
    assert sibling_result.draft_observations and not primary_result.draft_observations
    for result, provider in [(sibling_result, sibling_provider), (primary_result, primary_provider)]:
        assert TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V12 not in provider.systems
        assert result.diagnostics.outcome == "partial"


def test_negative_independent_answer_keeps_partial_without_retries():
    result, provider = extract(source_support=False)
    assert provider.systems.count(TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V12) == 1
    assert result.diagnostics.outcome == "partial" and not result.draft_observations
    assert not result.diagnostics.reason_counts.get("habit_after_claim_source_supported")


def test_two_unresolved_source_windows_share_a_single_additional_dispatch():
    lines = (SOURCE, SOURCE.replace("客人", "旅客"))
    rows = [dict(record("廖辰把行李牌装入口袋"), evidence=text, source_line_start=i, source_line_end=i)
            for i, text in enumerate(lines, 1)]
    provider = AmbiguousClaimProvider(rows, mode="ambiguous")
    text = "\n".join(lines)
    result = CharacterSignalExtractor(provider, settings=_settings()).extract_targeted(
        CharacterSignalChunk("draft-v3", "draft.md", text, 1, "draft"), (HABIT,), target_ordinal=1,
        source_identity=_source(text), frozen_content=text,
    )
    assert provider.systems.count(TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13) == 2
    assert provider.systems.count(TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V12) == 1
    assert result.diagnostics.outcome == "partial" and result.diagnostics.rejected_records == 2
    assert [s.evidence.line_start for s in result.draft_observations] == [1]
