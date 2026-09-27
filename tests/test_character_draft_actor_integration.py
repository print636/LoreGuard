from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

from app.character_draft_actor_review import (
    DRAFT_ACTOR_REVIEW_BATCH_SCHEMA_V1,
    DRAFT_ACTOR_REVIEW_SCHEMA_V1,
)
from app.character_draft_actor_review_provider import (
    DRAFT_ACTOR_REVIEW_SYSTEM_PROMPT,
    DRAFT_ACTOR_REVIEW_USER_PREFIX,
)
from app.character_scope_review import ScopeReviewSourceIdentity
from app.character_trait_extraction import (
    CHARACTER_SIGNAL_DRAFT_ACTOR_PROMPT_V1,
    CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2,
    CHARACTER_SIGNAL_SYSTEM_PROMPT,
    TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT,
    CharacterSignal,
    CharacterSignalChunk,
    CharacterSignalExtractor,
    CharacterSignalTarget,
    _draft_actor_clues_after_clean,
    _targeted_chunk_prompt,
    draft_actor_review_evidence_range,
    stable_trait_identity,
)
from app.config import Settings
from app.domain import EvidenceSpan
from app.provider import OpenAICompatibleProvider, RetryPolicy
from app.service import (
    CharacterConsistencyUsageAccumulator,
    _CharacterConsistencyAccountingProvider,
)
from app.usage import estimate_issue_evidence_review_tokens


def _settings(**overrides) -> Settings:
    values = {
        "openai_api_key": "unit-test-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_character_consistency": True,
        "provider_max_attempts": 1,
        "character_draft_actor_review_v1": True,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _source(content: str) -> ScopeReviewSourceIdentity:
    return ScopeReviewSourceIdentity(
        run_input_id="input-draft-1",
        document_id="draft-1",
        document_version=3,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _record(
    content: str,
    *,
    line_start: int = 1,
    line_end: int = 1,
    statement: str = "林澈把原稿改成红色",
    dimension: str = "core_personality",
    trait_key: str = "change_directness",
    polarity: str = "positive",
    key_object: str = "",
) -> dict:
    evidence = "\n".join(content.splitlines()[line_start - 1 : line_end])
    return {
        "character": "林澈",
        "dimension": dimension,
        "trait_key": trait_key,
        "statement": statement,
        "polarity": polarity,
        "stability": "temporary",
        "observation_kind": "action",
        "context": "",
        "key_object": key_object,
        "source_line_start": line_start,
        "source_line_end": line_end,
        "evidence": evidence,
    }


class _ActorReviewProvider:
    def __init__(self, primary_payloads: list[str], *, verdict: str = "supported"):
        self.primary_payloads = list(primary_payloads)
        self.verdict = verdict
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str):
        self.calls.append((system, user))
        if system != DRAFT_ACTOR_REVIEW_SYSTEM_PROMPT:
            if not self.primary_payloads:
                raise AssertionError("unexpected extraction call")
            return SimpleNamespace(
                text=self.primary_payloads.pop(0),
                prompt_tokens=13,
                completion_tokens=7,
            )

        request = json.loads(user.removeprefix(DRAFT_ACTOR_REVIEW_USER_PREFIX))
        responses = []
        for window in request["windows"]:
            frozen_request = window["request"]
            basis_ids = [
                clause["support_id"]
                for line in frozen_request["lines"]
                for clause in line["clauses"]
            ]
            items = []
            for proposal in frozen_request["proposals"]:
                if self.verdict == "supported":
                    slots = {
                        "actor": "proposed",
                        "actuality": "asserted",
                        "statement_relation": "supported",
                        "correction_relation": "none",
                        "basis_ids": basis_ids,
                    }
                elif self.verdict == "rejected":
                    slots = {
                        "actor": "other",
                        "actuality": "asserted",
                        "statement_relation": "supported",
                        "correction_relation": "none",
                        "basis_ids": basis_ids,
                    }
                else:
                    slots = {
                        "actor": "ambiguous",
                        "actuality": "ambiguous",
                        "statement_relation": "ambiguous",
                        "correction_relation": "ambiguous",
                        "basis_ids": [],
                    }
                items.append(
                    {
                        "proposal_id": proposal["proposal_id"],
                        "verdict": self.verdict,
                        **slots,
                    }
                )
            responses.append(
                {
                    "schema_version": DRAFT_ACTOR_REVIEW_SCHEMA_V1,
                    "request_digest": window["request_digest"],
                    "items": items,
                }
            )
        return SimpleNamespace(
            text=json.dumps(
                {
                    "schema_version": DRAFT_ACTOR_REVIEW_BATCH_SCHEMA_V1,
                    "batch_digest": request["batch_digest"],
                    "responses": responses,
                },
                ensure_ascii=False,
            ),
            prompt_tokens=19,
            completion_tokens=11,
        )

    @property
    def reviewer_calls(self) -> int:
        return sum(system == DRAFT_ACTOR_REVIEW_SYSTEM_PROMPT for system, _ in self.calls)


def _extract(content: str, provider: _ActorReviewProvider):
    source = _source(content)
    return CharacterSignalExtractor(provider, settings=_settings()).extract(
        CharacterSignalChunk("draft-1", "draft.md", content, 1, "draft"),
        source_identity=source,
        frozen_content=content,
    )


def test_supported_same_line_actor_review_rebinds_exact_record_once():
    content = "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    payload = json.dumps({"records": [_record(content)]}, ensure_ascii=False)
    provider = _ActorReviewProvider([payload])

    result = _extract(content, provider)

    assert result.diagnostics.outcome == "completed"
    assert provider.reviewer_calls == 1
    assert len(result.draft_observations) == 1
    assert result.draft_observations[0].statement == "林澈把原稿改成红色"


def test_supported_prior_named_anchor_allows_cross_line_continuation():
    content = "林澈亲自把灯台封好。\n她又把钥匙交给船长。"
    payload = json.dumps(
        {
            "records": [
                _record(
                    content,
                    line_start=1,
                    line_end=2,
                    statement="林澈又把钥匙交给船长",
                    trait_key="key_handover",
                )
            ]
        },
        ensure_ascii=False,
    )
    provider = _ActorReviewProvider([payload])

    result = _extract(content, provider)

    assert result.diagnostics.outcome == "completed"
    assert provider.reviewer_calls == 1
    assert [row.statement for row in result.draft_observations] == [
        "林澈又把钥匙交给船长"
    ]


def test_explicit_rejection_requires_clean_regeneration_but_not_coverage_debt():
    content = "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    first = json.dumps({"records": [_record(content)]}, ensure_ascii=False)
    clean = json.dumps({"records": []}, ensure_ascii=False)
    provider = _ActorReviewProvider([first, clean], verdict="rejected")

    result = _extract(content, provider)

    assert provider.reviewer_calls == 1
    assert result.diagnostics.outcome == "completed"
    assert result.signals == ()
    assert result.diagnostics.reason_counts["draft_actor_review_rejected"] == 1
    assert "draft_actor_review_incomplete" not in result.diagnostics.reason_counts


def test_uncertain_review_stays_partial_after_clean_regeneration():
    content = "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    first = json.dumps({"records": [_record(content)]}, ensure_ascii=False)
    clean = json.dumps({"records": []}, ensure_ascii=False)
    provider = _ActorReviewProvider([first, clean], verdict="uncertain")

    result = _extract(content, provider)

    assert provider.reviewer_calls == 1
    assert result.diagnostics.outcome == "partial"
    assert result.signals == ()
    assert result.diagnostics.reason_counts["draft_actor_review_incomplete"] == 1
    assert len(result.provisional_draft_clues) == 1


@pytest.mark.parametrize(
    ("accepted_polarity", "accepted_line", "expected_clues"),
    [
        ("neutral", 2, 1),
        ("unclear", 2, 1),
        ("negative", 2, 0),
        ("neutral", 1, 0),
    ],
)
def test_clean_signal_veto_preserves_nonopposing_nonoverlap_actor_clue(
    accepted_polarity: str,
    accepted_line: int,
    expected_clues: int,
):
    provisional = CharacterSignal(
        id="cs_" + "a" * 32,
        character="林澈",
        dimension="core_personality",
        trait_key="change_directness",
        statement="林澈把原稿改成红色",
        polarity="positive",
        stability="temporary",
        observation_kind="action",
        source_kind="draft",
        evidence=EvidenceSpan(
            document_id="draft-1",
            document_name="draft.md",
            line_start=1,
            line_end=1,
            text="第一行。",
        ),
    )
    accepted = provisional.model_copy(
        update={
            "id": "cs_" + "b" * 32,
            "statement": "林澈再次直接改稿",
            "polarity": accepted_polarity,
            "evidence": EvidenceSpan(
                document_id="draft-1",
                document_name="draft.md",
                line_start=accepted_line,
                line_end=accepted_line,
                text=f"第{accepted_line}行。",
            ),
        }
    )

    retained = _draft_actor_clues_after_clean(
        (provisional,), accepted_signals=(accepted,)
    )

    assert len(retained) == expected_clues


def test_invalid_reviewer_response_never_becomes_a_provisional_clue():
    content = "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    first = json.dumps({"records": [_record(content)]}, ensure_ascii=False)
    clean = json.dumps({"records": []}, ensure_ascii=False)
    provider = _ActorReviewProvider([first, clean], verdict="invalid")

    result = _extract(content, provider)

    assert provider.reviewer_calls == 1
    assert result.diagnostics.outcome == "partial"
    assert result.diagnostics.reason_counts["draft_actor_review_incomplete"] == 1
    assert result.signals == result.draft_observations == result.pending_candidates == ()
    assert result.provisional_draft_clues == ()


def test_successor_source_veto_remains_a_clue_after_clean_regeneration():
    content = (
        "她把原稿改成红色；记录显示本次操作由林澈本人完成。\n"
        "更正：上一行动作实际由周砚完成。"
    )
    first = json.dumps({"records": [_record(content)]}, ensure_ascii=False)
    clean = json.dumps({"records": []}, ensure_ascii=False)
    provider = _ActorReviewProvider([first, clean])

    result = _extract(content, provider)

    assert provider.reviewer_calls == 1
    assert provider.primary_payloads == []
    assert result.diagnostics.outcome == "partial"
    assert result.diagnostics.reason_counts["draft_actor_review_incomplete"] == 1
    assert "draft_actor_review_rejected" not in result.diagnostics.reason_counts
    assert result.signals == result.draft_observations == result.pending_candidates == ()
    assert len(result.provisional_draft_clues) == 1
    clue = result.provisional_draft_clues[0]
    assert clue.statement == "林澈把原稿改成红色"
    assert (clue.evidence.document_id, clue.evidence.line_start, clue.evidence.line_end) == (
        "draft-1",
        1,
        1,
    )


def test_actor_certificate_survives_record_reordering_but_requires_clean_package():
    content = (
        "林澈整理了索引。\n"
        "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    )
    actor = _record(content, line_start=2, line_end=2)
    first = json.dumps({"records": [actor, {"bad": "shape"}]}, ensure_ascii=False)
    direct = _record(
        content,
        line_start=1,
        line_end=1,
        statement="林澈整理了索引",
        trait_key="index_organization",
        polarity="neutral",
    )
    regenerated = json.dumps({"records": [direct, actor]}, ensure_ascii=False)
    provider = _ActorReviewProvider([first, regenerated])

    result = _extract(content, provider)

    assert provider.reviewer_calls == 1
    assert result.diagnostics.outcome == "completed"
    assert {row.statement for row in result.draft_observations} == {
        "林澈整理了索引",
        "林澈把原稿改成红色",
    }


def test_later_object_gate_failure_never_calls_actor_reviewer():
    content = "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    invalid = _record(
        content,
        dimension="value",
        trait_key="color_choice",
        key_object="蓝色",
    )
    payload = json.dumps({"records": [invalid]}, ensure_ascii=False)
    provider = _ActorReviewProvider([payload, payload])

    result = _extract(content, provider)

    assert provider.reviewer_calls == 0
    assert result.diagnostics.outcome == "degraded"
    assert result.diagnostics.reason_counts == {"key_object_support": 2}


def test_targeted_axis_mismatch_never_calls_actor_reviewer():
    content = "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    payload = json.dumps({"records": [_record(content)]}, ensure_ascii=False)
    provider = _ActorReviewProvider([payload, payload])
    source = _source(content)
    target = CharacterSignalTarget(
        character="林澈",
        dimension="speech_pattern",
        trait_key="directness",
        comparison_key=stable_trait_identity("speech_pattern", "directness"),
        baseline_polarity="negative",
        requested_polarity="positive",
        baseline_hint="林澈说话含蓄",
    )

    result = CharacterSignalExtractor(provider, settings=_settings()).extract_targeted(
        CharacterSignalChunk("draft-1", "draft.md", content, 1, "draft"),
        (target,),
        candidate_evidence_ranges=((1, 1),),
        source_identity=source,
        frozen_content=content,
    )

    assert provider.reviewer_calls == 0
    assert result.diagnostics.outcome == "degraded"
    assert result.diagnostics.reason_counts == {"targeted_target_mismatch": 2}


def test_targeted_prompt_marks_screened_actor_statement_as_proposal_not_proof():
    content = "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    source = _source(content)
    target = CharacterSignalTarget(
        character="林澈",
        dimension="core_personality",
        trait_key="change_directness",
        comparison_key=stable_trait_identity(
            "core_personality", "change_directness"
        ),
        baseline_polarity="negative",
        requested_polarity="positive",
        baseline_hint="林澈通常不直接改稿",
    )

    prompt = _targeted_chunk_prompt(
        CharacterSignalChunk("draft-1", "draft.md", content, 1, "draft"),
        (target,),
        candidate_evidence_ranges=((1, 1),),
        draft_actor_review_v1=True,
        source_identity=source,
        frozen_content=content,
    )
    candidates = json.loads(
        prompt.split("候选证据范围：", 1)[1].split("\n", 1)[0]
    )

    assert candidates[0]["actor_review_statement_proposals"] == [
        "林澈把原稿改成红色"
    ]
    assert "不证明主体、事件真实发生、target 语义轴或 requested_polarity" in prompt


def test_cross_line_range_is_added_only_after_frozen_structural_screen():
    content = "林澈亲自把灯台封好。\n她又把钥匙交给船长。"
    chunk = CharacterSignalChunk("draft-1", "draft.md", content, 1, "draft")
    source = _source(content)

    assert draft_actor_review_evidence_range(
        chunk,
        "林澈",
        1,
        source_identity=source,
        frozen_content=content,
    ) == (1, 2)
    assert draft_actor_review_evidence_range(
        chunk,
        "周尧",
        1,
        source_identity=source,
        frozen_content=content,
    ) is None


def test_feature_off_preserves_legacy_character_support_without_reviewer():
    content = "她把原稿改成红色；记录显示本次操作由林澈本人完成。"
    payload = json.dumps({"records": [_record(content)]}, ensure_ascii=False)
    provider = _ActorReviewProvider([payload, payload])

    result = CharacterSignalExtractor(
        provider,
        settings=_settings(character_draft_actor_review_v1=False),
    ).extract(CharacterSignalChunk("draft-1", "draft.md", content, 1, "draft"))

    assert provider.reviewer_calls == 0
    assert result.diagnostics.outcome == "degraded"
    assert result.diagnostics.reason_counts == {"character_support": 2}


class _RecordingProvider:
    def __init__(self):
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str):
        self.calls.append((system, user))
        return SimpleNamespace(text="{}", prompt_tokens=5, completion_tokens=3)


def test_accounting_accepts_exact_feature_on_primary_and_targeted_contracts():
    configured = _settings()
    signal = _RecordingProvider()
    usage = CharacterConsistencyUsageAccumulator()
    accounting = _CharacterConsistencyAccountingProvider(
        configured,
        usage,
        signal_provider=signal,
        drift_provider=_RecordingProvider(),
    )
    primary = (
        CHARACTER_SIGNAL_SYSTEM_PROMPT
        + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2
        + CHARACTER_SIGNAL_DRAFT_ACTOR_PROMPT_V1
    )
    targeted = (
        TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT
        + CHARACTER_SIGNAL_DRAFT_ACTOR_PROMPT_V1
    )

    accounting.complete(primary, "primary")
    accounting.complete(targeted, "targeted")

    assert [system for system, _ in signal.calls] == [primary, targeted]
    assert usage.logical_calls == 2


def test_accounting_forks_actor_reviewer_with_one_attempt_and_shared_usage():
    configured = _settings(
        provider_max_attempts=3,
        provider_timeout_seconds=20,
        provider_total_deadline_seconds=40,
        provider_max_completion_tokens=3_000,
        provider_max_response_bytes=20_000,
    )
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 17, "completion_tokens": 9},
            },
        )

    gateway = OpenAICompatibleProvider(
        configured,
        transport=httpx.MockTransport(respond),
        retry_policy=RetryPolicy(max_attempts=3),
    )
    usage = CharacterConsistencyUsageAccumulator()
    accounting = _CharacterConsistencyAccountingProvider(
        configured,
        usage,
        signal_provider=gateway,
        drift_provider=gateway,
    )
    bounded = accounting.fork_for_character_draft_actor_review(
        timeout_seconds=4,
        remaining_deadline_seconds=8,
        completion_reserve=320,
        max_response_bytes=6_000,
        max_attempts=1,
    )
    inner = bounded.draft_actor_review_provider

    assert isinstance(inner, OpenAICompatibleProvider)
    assert inner.settings.provider_timeout_seconds == 4
    assert inner.settings.provider_total_deadline_seconds == 4
    assert inner.settings.provider_max_completion_tokens == 320
    assert inner.settings.provider_max_response_bytes == 6_000
    assert inner.settings.provider_max_attempts == 1
    assert inner.retry_policy.max_attempts == 1
    bounded.complete(DRAFT_ACTOR_REVIEW_SYSTEM_PROMPT, "review request")
    assert len(seen) == 1
    assert json.loads(seen[0].content)["max_tokens"] == 320
    assert usage.logical_calls == 1
    assert usage.charged_tokens == estimate_issue_evidence_review_tokens(
        DRAFT_ACTOR_REVIEW_SYSTEM_PROMPT,
        "review request",
        completion_reserve=320,
    )

    preserved = bounded.fork_for_character_consistency(
        settings=configured,
        stage="signal",
        remaining_deadline_seconds=3,
    )
    assert preserved.draft_actor_review_provider is inner
    assert preserved.usage is usage

    with pytest.raises(ValueError, match="draft actor review limits"):
        accounting.fork_for_character_draft_actor_review(
            timeout_seconds=4,
            remaining_deadline_seconds=8,
            completion_reserve=320,
            max_response_bytes=6_000,
            max_attempts=2,
        )


def test_accounting_rejects_actor_purposes_when_feature_is_off():
    configured = _settings(character_draft_actor_review_v1=False)
    signal = _RecordingProvider()
    accounting = _CharacterConsistencyAccountingProvider(
        configured,
        CharacterConsistencyUsageAccumulator(),
        signal_provider=signal,
        drift_provider=_RecordingProvider(),
    )
    with pytest.raises(RuntimeError, match="unsupported character consistency provider purpose"):
        accounting.complete(DRAFT_ACTOR_REVIEW_SYSTEM_PROMPT, "review")
    with pytest.raises(RuntimeError, match="unsupported character consistency provider purpose"):
        accounting.complete(
            TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT
            + CHARACTER_SIGNAL_DRAFT_ACTOR_PROMPT_V1,
            "targeted",
        )
    assert signal.calls == []
