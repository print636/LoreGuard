"""V17 cannot borrow an older run's selection or claim-review entitlement."""
import pytest

from app.config import (
    CLAIM_RESOLUTION_POLICY_V1, HABIT_ACTION_SELECTION_POLICY_V1,
    HABIT_OBJECT_CHAIN_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V1,
)
from app.db import AnalysisRunRow
from app.service import (
    CharacterConsistencyUsageAccumulator, _CharacterConsistencyAccountingProvider,
    _character_stage_settings_for_run,
)
from app.character_draft_actor_review import SCOPED_DRAFT_DIRECTION_GUIDE
from app.character_draft_actor_review_provider import (
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V17,
)
from app.usage import estimate_issue_evidence_review_tokens
from tests.test_habit_action_selection_policy import _RecordingProvider
from tests.test_habit_scene_policy import LIMITS, local_run


POLICIES = {
    "habit_object_chain_policy": HABIT_OBJECT_CHAIN_POLICY_V1,
    "claim_resolution_policy": CLAIM_RESOLUTION_POLICY_V1,
    "habit_action_selection_policy": HABIT_ACTION_SELECTION_POLICY_V1,
    "selected_claim_resolution_policy": SELECTED_CLAIM_RESOLUTION_POLICY_V1,
}


def _stage(**updates):
    _, stage = local_run(coverage_changes=POLICIES)
    return stage.model_copy(update={"character_target_bound_draft_review_v3": True,
        "character_signal_full_line_prompt_v2": True, **updates})


@pytest.mark.parametrize("missing", [None, *POLICIES, "habit_scene_policy", "character_review_enabled"])
def test_new_policy_requires_both_old_capabilities_and_fresh_frozen_authorization(missing):
    policies = {**POLICIES, **({missing: None} if missing else {})}
    base, stage = local_run(coverage_changes=policies)
    assert stage._character_selected_claim_resolution_v1 is (missing is None)
    assert not base._character_selected_claim_resolution_v1
    assert "_character_selected_claim_resolution_v1" not in stage.model_dump()


def test_old_retry_cannot_inherit_a_process_private_override():
    stage = _stage()
    old = AnalysisRunRow(sensitivity="balanced", batch_coverage={})
    assert not _character_stage_settings_for_run(stage, old)._character_selected_claim_resolution_v1


def test_v17_exact_purpose_requires_every_gate_and_uses_existing_accounting():
    stage = _stage()
    transport = _RecordingProvider()
    usage = CharacterConsistencyUsageAccumulator()

    def provider(config):
        return _CharacterConsistencyAccountingProvider(config, usage, signal_provider=transport, drift_provider=transport)

    for field in ("_character_selected_claim_resolution_v1", "_character_claim_resolution_v1",
                  "_character_habit_action_selection_v1", "_character_habit_object_chain_v1",
                  "_character_habit_scene_scheduling_v1", "character_target_bound_draft_review_v3"):
        denied = stage.model_copy()
        setattr(denied, field, False)
        with pytest.raises(RuntimeError, match="unsupported"):
            provider(denied).complete(TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V17, "{}")
    for suffix in ("x", SCOPED_DRAFT_DIRECTION_GUIDE):
        with pytest.raises(RuntimeError, match="unsupported"):
            provider(stage).complete(TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V17 + suffix, "{}")
    assert not transport.sent and usage.logical_calls == 0
    bounded = provider(stage).fork_for_habit_scene_review(**{**LIMITS, "completion_reserve": 4096})
    bounded.complete(TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V17, "{}")
    assert len(transport.sent) == usage.logical_calls == usage.successful_calls == 1
    assert usage.charged_tokens == estimate_issue_evidence_review_tokens(
        TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V17, "{}", completion_reserve=4096)
    # The extra capability does not rewrite the older selection protocol.
    old = stage.model_copy()
    old._character_selected_claim_resolution_v1 = False
    provider(old).fork_for_habit_scene_review(**LIMITS).complete(TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16, "{}")
    assert usage.logical_calls == 2


def test_v17_transport_stays_single_and_honors_outer_capacity():
    stage = _stage(provider_max_completion_tokens=1024)
    bounded = _CharacterConsistencyAccountingProvider(stage, CharacterConsistencyUsageAccumulator()).fork_for_habit_scene_review(
        **{**LIMITS, "completion_reserve": 8192})
    assert bounded.draft_actor_review_completion_reserve == 4096
    assert bounded.draft_actor_review_provider.settings.provider_max_completion_tokens == 1024
    assert bounded.draft_actor_review_provider.settings.provider_max_attempts == 1
