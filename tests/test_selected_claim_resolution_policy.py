"""V17 cannot borrow an older run's selection or claim-review entitlement."""
import pytest

from app.config import (
    CLAIM_RESOLUTION_POLICY_V1, HABIT_ACTION_SELECTION_POLICY_V1,
    HABIT_OBJECT_CHAIN_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2, SELECTED_CLAIM_RESOLUTION_POLICY_V3,
)
from app.db import AnalysisRunRow
from app.service import (
    CharacterConsistencyUsageAccumulator, _CharacterConsistencyAccountingProvider,
    _character_stage_settings_for_run,
)
from app.character_draft_actor_review import SCOPED_DRAFT_DIRECTION_GUIDE
from app.character_draft_actor_review_provider import (
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V17,
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V18,
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


def _stage(policy=SELECTED_CLAIM_RESOLUTION_POLICY_V1, **updates):
    _, stage = local_run(coverage_changes={**POLICIES, "selected_claim_resolution_policy": policy})
    return stage.model_copy(update={"character_target_bound_draft_review_v3": True,
        "character_signal_full_line_prompt_v2": True, **updates})


@pytest.mark.parametrize("missing", [None, *POLICIES, "habit_scene_policy", "character_review_enabled"])
@pytest.mark.parametrize("policy", [SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2, SELECTED_CLAIM_RESOLUTION_POLICY_V3])
def test_new_policy_requires_both_old_capabilities_and_fresh_frozen_authorization(missing, policy):
    policies = {**POLICIES, "selected_claim_resolution_policy": policy, **({missing: None} if missing else {})}
    base, stage = local_run(coverage_changes=policies)
    assert stage._character_selected_claim_resolution_policy == (policy if missing is None else None)
    assert base._character_selected_claim_resolution_policy is None
    assert "_character_selected_claim_resolution_policy" not in stage.model_dump()


def test_old_retry_cannot_inherit_a_process_private_override():
    stage = _stage()
    old = AnalysisRunRow(sensitivity="balanced", batch_coverage={})
    assert _character_stage_settings_for_run(stage, old)._character_selected_claim_resolution_policy is None


@pytest.mark.parametrize(("policy", "prompt"), [
    (SELECTED_CLAIM_RESOLUTION_POLICY_V1, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V17),
    (SELECTED_CLAIM_RESOLUTION_POLICY_V2, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V18),
    (SELECTED_CLAIM_RESOLUTION_POLICY_V3, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V18),
])
def test_exact_purpose_requires_its_policy_version_every_gate_and_existing_accounting(policy, prompt):
    stage = _stage(policy)
    transport = _RecordingProvider()
    usage = CharacterConsistencyUsageAccumulator()

    def provider(config):
        return _CharacterConsistencyAccountingProvider(config, usage, signal_provider=transport, drift_provider=transport)

    for field in ("_character_selected_claim_resolution_policy", "_character_claim_resolution_v1",
                  "_character_habit_action_selection_v1", "_character_habit_object_chain_v1",
                  "_character_habit_scene_scheduling_v1", "character_target_bound_draft_review_v3"):
        denied = stage.model_copy()
        setattr(denied, field, None if field == "_character_selected_claim_resolution_policy" else False)
        with pytest.raises(RuntimeError, match="unsupported"):
            provider(denied).complete(prompt, "{}")
    for suffix in ("x", SCOPED_DRAFT_DIRECTION_GUIDE):
        with pytest.raises(RuntimeError, match="unsupported"):
            provider(stage).complete(prompt + suffix, "{}")
    wrong = TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V18 if policy == SELECTED_CLAIM_RESOLUTION_POLICY_V1 else TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V17
    with pytest.raises(RuntimeError, match="unsupported"):
        provider(stage).complete(wrong, "{}")
    assert not transport.sent and usage.logical_calls == 0
    bounded = provider(stage).fork_for_habit_scene_review(**{**LIMITS, "completion_reserve": 4096})
    bounded.complete(prompt, "{}")
    assert len(transport.sent) == usage.logical_calls == usage.successful_calls == 1
    assert usage.charged_tokens == estimate_issue_evidence_review_tokens(
        prompt, "{}", completion_reserve=4096)
    # The extra capability does not rewrite the older selection protocol.
    old = stage.model_copy()
    old._character_selected_claim_resolution_policy = None
    provider(old).fork_for_habit_scene_review(**LIMITS).complete(TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16, "{}")
    assert usage.logical_calls == 2


@pytest.mark.parametrize("policy", [SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2, SELECTED_CLAIM_RESOLUTION_POLICY_V3])
def test_selected_claim_transport_stays_single_and_honors_outer_capacity(policy):
    stage = _stage(policy, provider_max_completion_tokens=1024)
    bounded = _CharacterConsistencyAccountingProvider(stage, CharacterConsistencyUsageAccumulator()).fork_for_habit_scene_review(
        **{**LIMITS, "completion_reserve": 8192})
    assert bounded.draft_actor_review_completion_reserve == 4096
    assert bounded.draft_actor_review_provider.settings.provider_max_completion_tokens == 1024
    assert bounded.draft_actor_review_provider.settings.provider_max_attempts == 1
