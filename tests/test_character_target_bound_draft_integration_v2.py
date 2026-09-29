from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from app.character_draft_actor_review import (
    TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V2,
    TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V2,
)
from app.character_draft_actor_review_provider import (
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT,
)
from app.character_scope_review import ScopeReviewSourceIdentity
from app.character_trait_extraction import (
    CharacterSignalChunk,
    CharacterSignalExtractor,
    CharacterSignalTarget,
    stable_trait_identity,
)
from app.config import Settings


def _settings(**updates) -> Settings:
    values = {
        "openai_api_key": "target-bound-test",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_character_consistency": True,
        "character_signal_full_line_prompt_v2": True,
        "character_draft_actor_review_v1": True,
        "character_target_bound_draft_review_v2": True,
        "character_signal_package_max_attempts": 1,
        "character_signal_max_attempts": 1,
        "provider_max_attempts": 1,
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def _source(content: str) -> ScopeReviewSourceIdentity:
    return ScopeReviewSourceIdentity(
        run_input_id="run-input-1",
        document_id="draft-1",
        document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


class _Provider:
    def __init__(self, records: list[dict], *, mode: str = "supported"):
        self.records = records
        self.mode = mode
        self.signal_calls = 0
        self.review_calls = 0

    def complete(self, system: str, user: str):
        if system != TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT:
            self.signal_calls += 1
            return SimpleNamespace(
                text=json.dumps({"records": self.records}, ensure_ascii=False),
                prompt_tokens=7,
                completion_tokens=3,
            )
        self.review_calls += 1
        payload = json.loads(user.split("：\n", 1)[1])
        if self.mode == "invalid":
            return SimpleNamespace(text="{}", prompt_tokens=9, completion_tokens=2)
        responses = []
        for window in payload["windows"]:
            request = window["request"]
            scoped = request["target"]["approved_axis_comparison_key"] is not None
            items = []
            for proposal, hint in zip(
                request["proposals"], window["basis_path_hints"]
            ):
                rejected = self.mode == "rejected"
                items.append({
                    "proposal_id": proposal["proposal_id"],
                    "verdict": "rejected" if rejected else "supported",
                    "actor": "other" if rejected else "proposed",
                    "actuality": "asserted",
                    "statement_relation": "supported",
                    "axis_relation": (
                        "requires_scoped_review" if scoped else "matches_target"
                    ),
                    "object_relation": (
                        "not_applicable"
                        if scoped or not request["target"]["key_object"]
                        else "matches_target"
                    ),
                    "polarity_relation": "requested",
                    "correction_relation": "none",
                    "observation_kind": (
                        "preference_expression"
                        if request["target"]["dimension"] == "preference"
                        else "decision"
                    ),
                    "basis_ids": hint["required_basis_ids"],
                })
            responses.append({
                "schema_version": TARGET_BOUND_DRAFT_REVIEW_SCHEMA_V2,
                "request_digest": window["request_digest"],
                "items": items,
            })
        return SimpleNamespace(
            text=json.dumps({
                "schema_version": TARGET_BOUND_DRAFT_REVIEW_BATCH_SCHEMA_V2,
                "batch_digest": payload["batch_digest"],
                "responses": responses,
            }, ensure_ascii=False),
            prompt_tokens=19,
            completion_tokens=11,
        )


def _record(
    *,
    target: CharacterSignalTarget,
    evidence: str,
    statement: str,
    line: int,
    observation_kind: str,
) -> dict:
    return {
        "character": target.character,
        "dimension": target.dimension,
        "trait_key": target.trait_key,
        "statement": statement,
        "polarity": target.requested_polarity,
        "stability": "temporary",
        "observation_kind": observation_kind,
        "context": "",
        "key_object": target.key_object,
        "source_line_start": line,
        "source_line_end": line,
        "evidence": evidence,
    }


def _extract(
    content: str,
    target: CharacterSignalTarget,
    provider: _Provider,
    **settings_updates,
):
    return CharacterSignalExtractor(
        provider, settings=_settings(**settings_updates)
    ).extract_targeted(
        CharacterSignalChunk("draft-1", "draft.md", content, 1, "draft"),
        (target,),
        target_ordinal=3,
        source_identity=_source(content),
        frozen_content=content,
    )


def test_narrator_target_binding_yields_exactly_one_server_bound_observation():
    content = (
        "午餐时，旁白直接说明沈砚现在一直最讨厌栗子糕，"
        "闻到甜味就想离开；这不是引语、谎言、伪装或食物异常。"
    )
    target = CharacterSignalTarget(
        character="沈砚",
        dimension="preference",
        trait_key="food_preference",
        comparison_key=stable_trait_identity(
            "preference", "food_preference", "栗子糕"
        ),
        key_object="栗子糕",
        baseline_polarity="positive",
        requested_polarity="negative",
        baseline_hint="沈砚稳定喜欢栗子糕",
    )
    provider = _Provider([
        _record(
            target=target,
            evidence=content,
            statement="沈砚现在一直最讨厌栗子糕",
            line=1,
            observation_kind="preference_expression",
        )
    ])

    result = _extract(content, target, provider)

    assert (provider.signal_calls, provider.review_calls) == (1, 1)
    assert result.diagnostics.outcome == "completed"
    assert len(result.draft_observations) == 1
    observation = result.draft_observations[0]
    assert (
        observation.character,
        observation.dimension,
        observation.trait_key,
        observation.key_object,
        observation.polarity,
    ) == ("沈砚", "preference", "food_preference", "栗子糕", "negative")
    assert observation.statement == "沈砚现在一直最讨厌栗子糕"


def test_two_guchao_lines_deduplicate_clause_certificates_to_two_observations():
    content = (
        "海灯塔仍未重建，顾潮却主动撤回重建申请，并把全部修复材料转卖。\n"
        "次日，顾潮又解散重建小组，明确表示永久放弃重建海灯塔。"
    )
    target = CharacterSignalTarget(
        character="顾潮",
        dimension="motivation_goal",
        trait_key="lighthouse_commitment",
        comparison_key=stable_trait_identity(
            "motivation_goal", "lighthouse_commitment", "重建海灯塔"
        ),
        key_object="重建海灯塔",
        baseline_polarity="positive",
        requested_polarity="negative",
        baseline_hint="顾潮长期致力于重建海灯塔",
    )
    lines = content.splitlines()
    result = _extract(content, target, _Provider([
        _record(
            target=target,
            evidence=lines[0],
            statement="顾潮并把全部修复材料转卖",
            line=1,
            observation_kind="decision",
        ),
        _record(
            target=target,
            evidence=lines[1],
            statement="顾潮明确表示永久放弃重建海灯塔",
            line=2,
            observation_kind="decision",
        ),
    ]))

    assert result.diagnostics.outcome == "completed"
    assert len(result.draft_observations) == 2
    assert {item.evidence.line_start for item in result.draft_observations} == {1, 2}
    assert all(item.key_object == "重建海灯塔" for item in result.draft_observations)


def test_protocol_failure_is_partial_unresolved_with_zero_formal_observation():
    content = "旁白明确说明沈砚现在讨厌栗子糕。"
    target = CharacterSignalTarget(
        character="沈砚",
        dimension="preference",
        trait_key="food_preference",
        comparison_key=stable_trait_identity(
            "preference", "food_preference", "栗子糕"
        ),
        key_object="栗子糕",
        baseline_polarity="positive",
        requested_polarity="negative",
        baseline_hint="沈砚稳定喜欢栗子糕",
    )
    result = _extract(content, target, _Provider([
        _record(
            target=target,
            evidence=content,
            statement="沈砚现在讨厌栗子糕",
            line=1,
            observation_kind="preference_expression",
        )
    ], mode="invalid"))

    assert result.diagnostics.outcome == "partial"
    assert result.diagnostics.reason_counts["semantic_binding_unresolved"] == 1
    assert result.diagnostics.reason_counts["target_bound_response_invalid"] == 1
    assert result.signals == result.draft_observations == ()


def test_scoped_tang_value_candidate_is_only_routed_for_later_axis_review():
    definition = "遇到生命与货物冲突时优先救人"
    scope = "生命救援与货物保全发生直接冲突"
    proposition = "角色先救人再处理货物"
    target = CharacterSignalTarget(
        character="唐岫",
        dimension="value",
        trait_key="rescue_priority",
        comparison_key="value:先救人再保货",
        key_object="先救人再保货",
        baseline_polarity="positive",
        requested_polarity="negative",
        baseline_hint="唐岫坚持先救人再保货",
        approved_axis_id="11111111-1111-4111-8111-111111111111",
        approved_axis_version=1,
        approved_axis_definition=definition,
        approved_axis_definition_sha256=hashlib.sha256(definition.encode()).hexdigest(),
        approved_axis_comparison_key="value:先救人再保货",
        approved_axis_applicability_scope=scope,
        approved_axis_applicability_scope_sha256=hashlib.sha256(scope.encode()).hexdigest(),
        axis_positive_proposition=proposition,
        axis_positive_proposition_sha256=hashlib.sha256(proposition.encode()).hexdigest(),
    )
    content = "唐岫再次要求先打捞货物，随后才允许救人。"

    result = _extract(content, target, _Provider([
        _record(
            target=target,
            evidence=content,
            statement="唐岫随后才允许救人",
            line=1,
            observation_kind="decision",
        )
    ]))

    assert result.diagnostics.outcome == "completed"
    assert len(result.draft_observations) == 1
    observation = result.draft_observations[0]
    assert (observation.dimension, observation.key_object) == (
        "value", "先救人再保货"
    )
    # The V2 signal carries no approved-axis ID. The stage's existing
    # bind_approved_axis/scoped reviewer remains the only final axis authority.
    assert observation.model_dump().get("approved_axis_id") is None


def test_explicit_rejection_is_clean_absence_not_unresolved_binding():
    content = "旁白明确说明沈砚现在讨厌栗子糕。"
    target = CharacterSignalTarget(
        character="沈砚",
        dimension="preference",
        trait_key="food_preference",
        comparison_key=stable_trait_identity(
            "preference", "food_preference", "栗子糕"
        ),
        key_object="栗子糕",
        baseline_polarity="positive",
        requested_polarity="negative",
        baseline_hint="沈砚稳定喜欢栗子糕",
    )
    result = _extract(content, target, _Provider([
        _record(
            target=target,
            evidence=content,
            statement="沈砚现在讨厌栗子糕",
            line=1,
            observation_kind="preference_expression",
        )
    ], mode="rejected"))

    assert result.diagnostics.outcome == "completed"
    assert "semantic_binding_unresolved" not in result.diagnostics.reason_counts
    assert result.diagnostics.reason_counts["target_bound_review_rejected"] == 1
    assert result.signals == ()


def test_clean_direct_binder_has_identical_call_count_with_v2_on_or_off():
    content = "顾潮明确表示永久放弃重建海灯塔。"
    target = CharacterSignalTarget(
        character="顾潮",
        dimension="motivation_goal",
        trait_key="lighthouse_commitment",
        comparison_key=stable_trait_identity(
            "motivation_goal", "lighthouse_commitment", "重建海灯塔"
        ),
        key_object="重建海灯塔",
        baseline_polarity="positive",
        requested_polarity="negative",
        baseline_hint="顾潮长期致力于重建海灯塔",
    )
    record = _record(
        target=target,
        evidence=content,
        statement="顾潮明确表示永久放弃重建海灯塔",
        line=1,
        observation_kind="decision",
    )
    enabled_provider = _Provider([record])
    disabled_provider = _Provider([record])

    enabled = _extract(content, target, enabled_provider)
    disabled = _extract(
        content,
        target,
        disabled_provider,
        character_target_bound_draft_review_v2=False,
    )

    assert enabled.diagnostics.outcome == disabled.diagnostics.outcome == "completed"
    assert len(enabled.signals) == len(disabled.signals) == 1
    assert (enabled_provider.signal_calls, enabled_provider.review_calls) == (1, 0)
    assert (disabled_provider.signal_calls, disabled_provider.review_calls) == (1, 0)


def test_non_binding_rejection_never_invokes_target_bound_reviewer():
    content = "顾潮明确表示永久放弃重建海灯塔。"
    target = CharacterSignalTarget(
        character="顾潮",
        dimension="motivation_goal",
        trait_key="lighthouse_commitment",
        comparison_key=stable_trait_identity(
            "motivation_goal", "lighthouse_commitment", "重建海灯塔"
        ),
        key_object="重建海灯塔",
        baseline_polarity="positive",
        requested_polarity="negative",
        baseline_hint="顾潮长期致力于重建海灯塔",
    )
    malformed = _record(
        target=target,
        evidence="被篡改的证据。",
        statement="顾潮明确表示永久放弃重建海灯塔",
        line=1,
        observation_kind="decision",
    )
    provider = _Provider([malformed])

    result = _extract(content, target, provider)

    assert provider.review_calls == 0
    assert provider.signal_calls == 1
    assert result.signals == ()
    assert "semantic_binding_unresolved" not in result.diagnostics.reason_counts
