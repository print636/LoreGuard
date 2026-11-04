"""A successful action cannot erase a different generated claim on its line."""
import json
from types import SimpleNamespace

import pytest

from app.character_draft_actor_review import TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V4
from app.character_draft_actor_review_provider import TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13
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
    def __init__(self, records):
        self.records = records
        self.requests = []
        self.generations = 0

    def complete(self, system, user):
        if system != TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13:
            self.generations += 1
            result = {"records": self.records}
        else:
            batch = json.loads(user.split("：\n", 1)[1])
            responses = []
            for window in batch["windows"]:
                request = window["request"]
                self.requests.append(request)
                claim = request["primary_claim"]
                items = []
                for proposal, hint in zip(request["proposals"], window["basis_path_hints"], strict=True):
                    same = "口袋" in hint["fact_clause_text"] and "口袋" in claim["statement"]
                    item = dict(
                        proposal_id=proposal["proposal_id"], verdict="supported" if same else "rejected",
                        actor="proposed", actuality="asserted", statement_relation="supported",
                        axis_relation="matches_target" if same else "different", object_relation="not_applicable",
                        polarity_relation="requested", correction_relation="none", observation_kind="action",
                        object_start_offset=None, object_end_offset=None, basis_ids=hint["required_basis_ids"],
                        claim_digest=claim["claim_digest"], claim_relation="equivalent" if same else "different",
                    )
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
        return SimpleNamespace(text=json.dumps(result, ensure_ascii=False), prompt_tokens=9, completion_tokens=8)


def extract(records, targeted, **target_changes):
    provider = ClaimProvider(records)
    target = HABIT.model_copy(update=target_changes)
    chunk = CharacterSignalChunk("draft-v3", "draft.md", SOURCE, 1, "draft")
    extractor = CharacterSignalExtractor(provider, settings=_settings(
        character_signal_package_max_attempts=2, character_signal_draft_trace_v1=True,
    ))
    source = dict(source_identity=_source(SOURCE), frozen_content=SOURCE)
    if targeted:
        result = extractor.extract_targeted(chunk, (target,), target_ordinal=3, **source)
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
