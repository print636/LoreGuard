"""A successful action cannot erase a different generated claim on its line."""
import json
from types import SimpleNamespace

import pytest

from app.character_draft_actor_review import TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V4
from app.character_draft_actor_review_provider import (
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V12, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13,
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V14,
)
from app.character_trait_extraction import CharacterSignalChunk, CharacterSignalExtractor, CharacterSignalTarget
from tests.test_character_target_bound_draft_review_v3 import _settings, _source


SOURCE = "班车进站时，廖辰接过客人的手提箱，随后亲手将行李牌收进了口袋。"
HABIT = CharacterSignalTarget(
    character="廖辰", dimension="contextual_behavior", trait_key="bag_labeling",
    comparison_key="contextual_behavior:baglabeling", requested_polarity="negative",
    baseline_polarity="positive", habit_contexts=("乘客交运行李时",),
    baseline_hint="廖辰给交运的箱子挂牌。",
)


def record(statement):
    return dict(character=HABIT.character, dimension=HABIT.dimension, trait_key=HABIT.trait_key,
                key_object="", polarity="negative", stability="temporary", observation_kind="action",
                statement=statement, context="", source_line_start=1, source_line_end=1, evidence=SOURCE)


class ClaimProvider:
    def __init__(self, records, *, generation_tokens=9, source_support=True):
        self.records = records
        self.generation_tokens = generation_tokens
        self.source_support = source_support
        self.requests = []
        self.generations = 0

    def complete(self, system, user):
        prompt_tokens = 9
        if system not in {TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V12, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13,
                          TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V14}:
            self.generations += 1
            prompt_tokens = self.generation_tokens
            result = {"records": self.records}
        else:
            batch = json.loads(user.split("：\n", 1)[1])
            responses = []
            for window in batch["windows"]:
                request = window["request"]
                self.requests.append(request)
                claim = request.get("primary_claim")
                items = []
                for proposal, hint in zip(request["proposals"], window["basis_path_hints"], strict=True):
                    same = "口袋" in hint["fact_clause_text"] and (
                        "口袋" in claim["statement"] if claim else self.source_support
                    )
                    item = dict(
                        proposal_id=proposal["proposal_id"], verdict="supported" if same else "rejected",
                        actor="proposed", actuality="asserted", statement_relation="supported",
                        axis_relation="matches_target" if same else "different", object_relation="not_applicable",
                        polarity_relation="requested", correction_relation="none", observation_kind="action",
                        object_start_offset=None, object_end_offset=None, basis_ids=hint["required_basis_ids"],
                    )
                    if claim:
                        item.update(claim_digest=claim["claim_digest"], claim_relation="equivalent" if same else "different")
                    if system == TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V14:
                        item["claim_object"] = (dict(
                            role="event_object", grounding="explicit", referent_relation="same", fact_quote="行李牌",
                            antecedent_clause_id=None, antecedent_quote=None,
                        ) if same else None)
                    if "action_path" in hint:
                        item["action_steps"] = [
                            dict(clause_id=step["clause_id"], quote=step["text"],
                                 actor="target", actuality="actual", scene="same") for step in hint["action_path"]
                        ]
                    items.append(item)
                responses.append(dict(schema_version=request["schema_version"],
                                      request_digest=window["request_digest"], items=items))
            result = dict(schema_version=TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V4,
                          batch_digest=batch["batch_digest"], responses=responses)
        return SimpleNamespace(text=json.dumps(result, ensure_ascii=False), prompt_tokens=prompt_tokens, completion_tokens=8)


def extract(records, targeted, *, generation_tokens=9, source_support=True, candidate_ranges=(), content=SOURCE,
            **target_changes):
    provider = ClaimProvider(records, generation_tokens=generation_tokens, source_support=source_support)
    target = HABIT.model_copy(update=target_changes)
    chunk = CharacterSignalChunk("draft-v3", "draft.md", content, 1, "draft")
    extractor = CharacterSignalExtractor(provider, settings=_settings(
        character_signal_package_max_attempts=2, character_signal_draft_trace_v1=True,
    ))
    source = dict(source_identity=_source(content), frozen_content=content)
    if targeted:
        result = extractor.extract_targeted(chunk, (target,), target_ordinal=3,
                                            candidate_evidence_ranges=candidate_ranges, **source)
    else:
        result = extractor.extract(chunk, primary_value_targets=(target,), **source)
    return result, provider


@pytest.mark.parametrize("targeted", [True, False])
@pytest.mark.parametrize("claim", ["廖辰把行李牌收进了口袋", "廖辰把行李牌装入口袋"])
def test_only_the_exact_record_can_clear_actor_or_statement_debt(targeted, claim):
    result, provider = extract([record(claim)], targeted)
    assert provider.generations == 1
    request, = provider.requests
    assert request["target"]["target_ordinal"] == (3 if targeted else 1)
    assert request["primary_claim"]["statement"] == claim
    assert request["primary_claim"]["record_ordinal"] == 1
    assert result.diagnostics.outcome == "completed"
    observation, = result.draft_observations
    assert "将行李牌收进了口袋" in observation.statement
    assert observation.target_bound_target.habit_contexts == HABIT.habit_contexts
    assert result.diagnostics.reason_counts["habit_claim_recovery_supported"] == 1


@pytest.mark.parametrize("targeted", [True, False])
def test_unrelated_claim_on_same_line_is_not_refunded(targeted):
    result, provider = extract([record("廖辰把行李牌装入口袋"), record("廖辰捏碎珍珠")], targeted)
    assert provider.generations == 1
    assert len({r["primary_claim"]["claim_digest"] for r in provider.requests}) == 2
    assert result.diagnostics.outcome == "partial"
    assert len(result.draft_observations) == 1
    assert result.diagnostics.reason_counts["statement_support"] == 1


def test_excluded_window_cannot_certify_a_targeted_record():
    result, provider = extract([record("廖辰把行李牌装入口袋")], True, existing_evidence_ranges=((1, 1),))
    assert not provider.requests and not result.draft_observations


@pytest.mark.parametrize("candidate_ranges,stage", [((), "targeted"), (((1, 1),), "verify")])
def test_independent_source_finds_action_but_leaves_wrong_original_partial(candidate_ranges, stage):
    bad = dict(record("廖辰把一枚纪念章装入口袋"), key_object="PRIVATE-RAW-OBJECT")
    result, provider = extract([bad], True, candidate_ranges=candidate_ranges)
    request, = provider.requests
    assert not request.get("primary_claim")
    assert provider.generations == 1
    observation, = result.draft_observations
    assert "行李牌" in observation.statement and "纪念章" not in observation.statement
    assert result.diagnostics.outcome == "partial"
    assert result.diagnostics.rejected_records == 1
    reasons = result.diagnostics.reason_counts
    assert reasons[f"habit_gate_{stage}_source_only"] == 1
    assert reasons[f"habit_gate_{stage}_object_unanchored"] == 1
    assert not any(reason.startswith("habit_claim_recovery") for reason in reasons)
    assert "PRIVATE" not in json.dumps(reasons)


@pytest.mark.parametrize("generation_tokens,source_support", [(9, False), (1_000_000, True)])
def test_no_supported_source_or_exhausted_budget_keeps_original_unresolved(generation_tokens, source_support):
    bad = dict(record("廖辰把一枚纪念章装入口袋"), key_object="PRIVATE-RAW-OBJECT")
    result, provider = extract([bad], True, generation_tokens=generation_tokens, source_support=source_support)
    assert provider.generations == 1
    assert len(provider.requests) == (0 if generation_tokens > 9 else 1)
    assert result.diagnostics.outcome == "partial"
    assert result.diagnostics.rejected_records == 1 and not result.draft_observations


@pytest.mark.parametrize("changes,gate", [({"character": "甲"}, "actor"), ({"polarity": "positive"}, "direction"),
                                          ({"trait_key": "other_key"}, "comparison_key")])
def test_primary_gate_counts_distinguish_parts_of_frozen_target(changes, gate):
    result, provider = extract([dict(record("廖辰把行李牌装入口袋"), **changes)], False)
    assert result.diagnostics.reason_counts[f"habit_gate_primary_{gate}"] == provider.generations
    assert not provider.requests


def test_quoted_story_action_remains_unobserved_and_hard_gate_is_identified():
    claim = "廖辰把行李牌收进口袋"
    text = f"演员只是朗读剧本：“{claim}”。"
    result, provider = extract([dict(record(claim), evidence=text)], False, content=text)
    assert result.diagnostics.reason_counts["habit_gate_primary_preflight_quoted_only"] == provider.generations
    assert not provider.requests and not result.draft_observations


@pytest.mark.parametrize("targeted", [True, False])
@pytest.mark.parametrize("raw_object", ["行李牌", "行李标签"])
def test_habit_object_review_retains_the_exact_affected_object_in_the_action(targeted, raw_object):
    result, provider = extract([dict(record(f"廖辰把{raw_object}装入口袋"), key_object=raw_object)], targeted)
    request, = provider.requests
    assert request["prompt_version"].endswith("-v14") and request["primary_claim"]["key_object"] == raw_object
    assert result.diagnostics.outcome == "completed"
    assert result.diagnostics.reason_counts["habit_claim_recovery_supported"] == 1
    observation, = result.draft_observations
    assert observation.key_object == "行李牌" and "行李牌" in observation.statement
    assert observation.evidence.text == SOURCE and observation.target_bound_target.habit_contexts == HABIT.habit_contexts


def test_another_object_on_the_same_line_cannot_repay_the_raw_habit_claim():
    result, provider = extract([dict(record("廖辰把手提箱装入口袋"), key_object="手提箱")], True)
    request, = provider.requests
    assert request["prompt_version"].endswith("-v14")
    assert result.diagnostics.outcome == "partial" and not result.draft_observations
    assert result.diagnostics.target_bound_slot_conflict_counts["object_span"]


def test_object_from_a_second_event_cannot_bypass_claim_review_even_with_literal_statement():
    text = "廖辰接过手提箱，廖辰把行李牌收进口袋。"
    bad = dict(record("廖辰把行李牌收进口袋"), evidence=text, key_object="手提箱")
    result, provider = extract([bad], False, content=text)
    request, = provider.requests
    assert request["prompt_version"].endswith("-v14")
    assert result.diagnostics.rejected_records == 1 and not result.draft_observations


def test_source_bound_object_is_one_observation_across_passes_and_claim_wording():
    primary, _ = extract([dict(record("廖辰把行李标签装入口袋"), key_object="行李标签")], False)
    targeted, _ = extract([dict(record("廖辰把行李牌装入口袋"), key_object="行李牌")], True)
    assert primary.draft_observations[0].id == targeted.draft_observations[0].id
