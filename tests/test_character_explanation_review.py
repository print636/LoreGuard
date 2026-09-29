from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

import app.character_explanation_review as explanation_review_module
from app.character_explanation_review import (
    EXPLANATION_REVIEW_PROMPT_V2,
    EXPLANATION_REVIEW_SCHEMA_V2,
    EXPLANATION_REVIEW_SYSTEM_PROMPT,
    EXPLANATION_REVIEW_USER_PREFIX,
    ExplanationBaselineSummary,
    ExplanationCandidate,
    ExplanationObservationSummary,
    build_character_explanation_review_prompts,
    run_character_explanation_review,
)
from app.config import Settings
from app.domain import EvidenceSpan
from app.provider import OpenAICompatibleProvider
from app.usage import estimate_issue_evidence_review_tokens


def _settings(**overrides) -> Settings:
    values = {
        "openai_api_key": "unit-test-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_character_consistency": True,
        "provider_max_attempts": 1,
        "character_explanation_token_budget": 60_000,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _evidence(
    document_id: str, line: int, text: str | None = None
) -> EvidenceSpan:
    return EvidenceSpan(
        document_id=document_id,
        document_name=f"{document_id}.md",
        line_start=line,
        line_end=line,
        text=text or f"林澈在事件 {line} 后逐渐恢复了公开表达。",
    )


def _baseline() -> ExplanationBaselineSummary:
    return ExplanationBaselineSummary(
        character="林澈",
        dimension="core_personality",
        trait_key="reserved_speech",
        statement="林澈长期寡言，不主动公开表达感受。",
        approved_axis_definition="面对群体时主动表达内心感受的稳定倾向。",
        axis_positive_proposition="林澈会主动向群体表达内心感受。",
    )


def _observations() -> tuple[ExplanationObservationSummary, ...]:
    evidence = _evidence("draft-current", 20, "林澈主动向所有同伴讲述自己的恐惧。")
    return (
        ExplanationObservationSummary(
            citation="C01",
            observation_id="cs_" + "1" * 32,
            statement="林澈主动向所有同伴讲述自己的恐惧。",
            document_id="draft-current",
            source_ordinal=20,
            evidence=evidence,
        ),
    )


def test_major_ooc_explanation_prompt_freezes_and_matches_key_object():
    baseline = ExplanationBaselineSummary(
        character="林澈",
        dimension="relationship_attitude",
        trait_key="trust_orientation",
        key_object="周尧",
        statement="林澈长期信任周尧。",
    )
    evidence = _evidence("draft-current", 20, "林澈明确表示不再信任周尧。")
    observations = (
        ExplanationObservationSummary(
            citation="C01",
            observation_id="cs_" + "1" * 32,
            statement="林澈明确表示不再信任周尧。",
            key_object="周尧",
            document_id="draft-current",
            source_ordinal=20,
            evidence=evidence,
        ),
    )

    system_prompt, user_prompt = build_character_explanation_review_prompts(
        (_candidate(),), baseline=baseline, observations=observations
    )
    packet = json.loads(user_prompt.removeprefix(EXPLANATION_REVIEW_USER_PREFIX))

    assert packet["baseline"]["key_object"] == "周尧"
    assert packet["observations"][0]["key_object"] == "周尧"
    assert "同一中性轴和同一 key_object" in system_prompt

    mismatched = observations[0].model_copy(update={"key_object": "顾岚"})
    with pytest.raises(ValueError, match="key_object"):
        build_character_explanation_review_prompts(
            (_candidate(),), baseline=baseline, observations=(mismatched,)
        )


def _candidate(
    citation: str = "E01",
    *,
    source_kind: str = "published_history",
    line: int = 3,
    source_ordinal: int = 3,
    evidence: EvidenceSpan | None = None,
) -> ExplanationCandidate:
    draft = source_kind == "draft"
    default_document = "draft-current" if draft else f"history-{citation}"
    return ExplanationCandidate(
        citation=citation,
        source_kind=source_kind,
        publication_status="draft" if draft else "published",
        authority_tier="draft" if draft else "formal_record",
        resolution_state="confirmed",
        source_ordinal=source_ordinal,
        eligible_draft_document_ids=("draft-current",),
        evidence=evidence or _evidence(default_document, line),
    )


def _item(candidate: ExplanationCandidate, **updates) -> dict[str, object]:
    value = {
        "citation": candidate.citation,
        "explanation_type": "growth_or_recovery",
        "actuality": "actual",
        "actor_relation": "same",
        "axis_relation": "same",
        "temporal_relation": "prior_or_active",
        "causal_relation": "explicit_causal",
        "applicable_observation_citations": ["C01"],
    }
    value.update(updates)
    return value


def _response_envelope(
    candidates,
    items,
    *,
    baseline=None,
    observations=None,
    top_updates=None,
) -> str:
    _, user = build_character_explanation_review_prompts(
        candidates,
        baseline=baseline or _baseline(),
        observations=observations or _observations(),
    )
    request = json.loads(user.removeprefix(EXPLANATION_REVIEW_USER_PREFIX))
    value = {
        "schema_version": EXPLANATION_REVIEW_SCHEMA_V2,
        "request_digest": request["request_digest"],
        "items": items,
    }
    value.update(top_updates or {})
    return json.dumps(value, ensure_ascii=False)


def _response(
    candidates,
    updates=None,
    *,
    baseline=None,
    observations=None,
) -> str:
    updates = updates or {}
    return _response_envelope(
        candidates,
        [
            _item(candidate, **updates.get(candidate.citation, {}))
            for candidate in candidates
        ],
        baseline=baseline,
        observations=observations,
    )


class SequenceProvider:
    def __init__(self, *values):
        self.values = list(values)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str):
        self.calls.append((system, user))
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(text=value, prompt_tokens=13, completion_tokens=7)


def _run(candidates, provider, *, settings=None):
    return run_character_explanation_review(
        tuple(candidates),
        baseline=_baseline(),
        observations=_observations(),
        provider=provider,
        settings=settings or _settings(),
    )


def test_prompt_contains_frozen_current_evidence_and_only_fixed_output_slots():
    candidate = _candidate()
    system, user = build_character_explanation_review_prompts(
        (candidate,), baseline=_baseline(), observations=_observations()
    )
    payload = json.loads(user.removeprefix(EXPLANATION_REVIEW_USER_PREFIX))
    assert payload["observations"][0]["evidence"]["text"].startswith("林澈主动")
    assert "observation_id" not in payload["observations"][0]
    assert payload["candidates"][0]["evidence"]["document_id"] == "history-E01"
    assert payload["schema_version"] == EXPLANATION_REVIEW_SCHEMA_V2
    assert len(payload["request_digest"]) == 64
    assert EXPLANATION_REVIEW_PROMPT_V2 in system
    assert '"schema_version":"character-explanation-review-v2"' in system
    assert "growth_or_recovery|disguise_or_role|temporary_state_or_pressure" in system
    assert "foreshadowing_or_ambiguous|irrelevant" in system
    assert "不得输出理由、摘要、置信度" in system
    assert "这种绑定只会形成 P 线索，不证明因果或解释成立" in system


def test_request_digest_is_stable_and_binds_every_model_visible_input():
    candidate = _candidate()
    _, first = build_character_explanation_review_prompts(
        (candidate,), baseline=_baseline(), observations=_observations()
    )
    _, repeated = build_character_explanation_review_prompts(
        (candidate,), baseline=_baseline(), observations=_observations()
    )
    changed = candidate.model_copy(
        update={"evidence": _evidence("history-E01", 3, "另一条冻结候选。")}
    )
    _, changed_prompt = build_character_explanation_review_prompts(
        (changed,), baseline=_baseline(), observations=_observations()
    )
    first_payload = json.loads(first.removeprefix(EXPLANATION_REVIEW_USER_PREFIX))
    repeated_payload = json.loads(
        repeated.removeprefix(EXPLANATION_REVIEW_USER_PREFIX)
    )
    changed_payload = json.loads(
        changed_prompt.removeprefix(EXPLANATION_REVIEW_USER_PREFIX)
    )

    assert first_payload["request_digest"] == repeated_payload["request_digest"]
    assert first_payload["request_digest"] != changed_payload["request_digest"]


def test_run_deep_snapshot_survives_checkpoint_mutation_of_all_original_inputs():
    candidate = _candidate(
        evidence=_evidence(
            "history-E01", 3, "原始成长经历明确解释了当前变化。"
        )
    )
    baseline = _baseline()
    observations = _observations()
    _, expected_user = build_character_explanation_review_prompts(
        (candidate,), baseline=baseline, observations=observations
    )
    expected_payload = json.loads(
        expected_user.removeprefix(EXPLANATION_REVIEW_USER_PREFIX)
    )
    response = _response(
        (candidate,), baseline=baseline, observations=observations
    )
    mutated = False

    def mutate_caller_owned_models():
        nonlocal mutated
        if mutated:
            return
        mutated = True
        candidate.evidence.text = "污染后的候选证据。"
        observations[0].evidence.text = "污染后的观察证据。"
        object.__setattr__(observations[0], "statement", "污染后的观察摘要。")
        object.__setattr__(baseline, "statement", "污染后的基线摘要。")

    provider = SequenceProvider(response)
    result = run_character_explanation_review(
        (candidate,),
        baseline=baseline,
        observations=observations,
        provider=provider,
        settings=_settings(),
        checkpoint=mutate_caller_owned_models,
    )

    actual_payload = json.loads(
        provider.calls[0][1].removeprefix(EXPLANATION_REVIEW_USER_PREFIX)
    )
    assert result.coverage == "complete"
    assert result.support_evidence[0].evidence.text == (
        "原始成长经历明确解释了当前变化。"
    )
    assert actual_payload == expected_payload
    assert actual_payload["request_digest"] == expected_payload["request_digest"]
    assert actual_payload["baseline"]["statement"] != "污染后的基线摘要。"
    assert actual_payload["observations"][0]["statement"] != "污染后的观察摘要。"
    assert "污染" not in provider.calls[0][1]


def test_protocol_subclasses_cannot_override_dump_or_attribute_access():
    touched: list[str] = []

    class HostileCandidate(ExplanationCandidate):
        def model_dump(self, *args, **kwargs):
            touched.append("model_dump")
            raise AssertionError("hostile model_dump executed")

        def __getattribute__(self, name):
            if name == "citation":
                touched.append("citation")
                raise AssertionError("hostile attribute access executed")
            return super().__getattribute__(name)

    original = _candidate()
    hostile = HostileCandidate.model_validate(original.model_dump())
    provider = SequenceProvider()

    with pytest.raises(TypeError, match="model type"):
        run_character_explanation_review(
            (hostile,),
            baseline=_baseline(),
            observations=_observations(),
            provider=provider,
            settings=_settings(),
        )

    assert touched == []
    assert provider.calls == []


def test_nested_evidence_subclass_is_rejected_before_its_dump_can_run():
    touched: list[str] = []

    class HostileEvidence(EvidenceSpan):
        def model_dump(self, *args, **kwargs):
            touched.append("model_dump")
            raise AssertionError("hostile evidence dump executed")

    candidate = _candidate()
    hostile_evidence = HostileEvidence.model_validate(
        candidate.evidence.model_dump()
    )
    object.__setattr__(candidate, "evidence", hostile_evidence)
    provider = SequenceProvider()

    with pytest.raises(TypeError, match="model type"):
        run_character_explanation_review(
            (candidate,),
            baseline=_baseline(),
            observations=_observations(),
            provider=provider,
            settings=_settings(),
        )

    assert touched == []
    assert provider.calls == []


def test_baseline_and_observation_subclasses_are_rejected_before_dump():
    touched: list[str] = []

    class HostileBaseline(ExplanationBaselineSummary):
        def model_dump(self, *args, **kwargs):
            touched.append("baseline")
            raise AssertionError("hostile baseline dump executed")

    class HostileObservation(ExplanationObservationSummary):
        def model_dump(self, *args, **kwargs):
            touched.append("observation")
            raise AssertionError("hostile observation dump executed")

    baseline = _baseline()
    observation = _observations()[0]
    hostile_baseline = HostileBaseline.model_validate(baseline.model_dump())
    hostile_observation = HostileObservation.model_validate(
        dict(object.__getattribute__(observation, "__dict__"))
    )

    with pytest.raises(TypeError, match="model type"):
        run_character_explanation_review(
            (_candidate(),),
            baseline=hostile_baseline,
            observations=(observation,),
            provider=SequenceProvider(),
            settings=_settings(),
        )
    with pytest.raises(TypeError, match="model type"):
        run_character_explanation_review(
            (_candidate(),),
            baseline=baseline,
            observations=(hostile_observation,),
            provider=SequenceProvider(),
            settings=_settings(),
        )

    assert touched == []


def test_explicit_published_growth_becomes_causal_bridge():
    candidate = _candidate()
    result = _run((candidate,), SequenceProvider(_response((candidate,))))
    assert result.coverage == "complete"
    assert result.diagnostics.outcome == "complete"
    assert result.diagnostics.attempted_calls == 1
    assert result.diagnostics.usage["charged_tokens"] > 0
    assert [(row.kind, row.explicit) for row in result.support_evidence] == [
        ("causal_bridge", True)
    ]
    assert result.support_evidence[0].applicable_observation_ids == (
        "cs_" + "1" * 32,
    )


def test_definitive_explanation_cannot_overlap_its_applicable_current_evidence():
    observation = _observations()[0]
    candidate = _candidate(
        source_kind="draft",
        source_ordinal=20,
        evidence=observation.evidence.model_copy(deep=True),
    )
    raw = _response(
        (candidate,),
        {candidate.citation: {"explanation_type": "disguise_or_role"}},
    )

    result = _run((candidate,), SequenceProvider(raw))

    assert result.support_evidence == ()
    assert result.coverage == "partial"
    assert result.diagnostics.outcome == "degraded"
    assert result.diagnostics.failure_reasons == ("response_invalid",)
    assert result.diagnostics.contract_failure_counts == {
        "promotion_invalid": 1
    }


def test_server_owned_same_line_candidate_is_capped_at_possible_support():
    observation = _observations()[0]
    candidate = _candidate(
        source_kind="draft",
        source_ordinal=20,
        evidence=observation.evidence.model_copy(deep=True),
    ).model_copy(update={"promotion_cap": "possible_only"})
    raw = _response(
        (candidate,),
        {candidate.citation: {"explanation_type": "disguise_or_role"}},
    )

    result = _run((candidate,), SequenceProvider(raw))

    assert result.coverage == "complete"
    assert [(row.kind, row.explicit) for row in result.support_evidence] == [
        ("possible_explanation", False)
    ]


def test_same_document_nonoverlapping_retrospective_explanation_still_promotes():
    candidate = _candidate(
        source_kind="draft",
        line=21,
        evidence=_evidence(
            "draft-current",
            21,
            "补记：林澈此前主动发言，是因为当时正在伪装身份。",
        ),
    )
    raw = _response(
        (candidate,),
        {candidate.citation: {"explanation_type": "disguise_or_role"}},
    )

    result = _run((candidate,), SequenceProvider(raw))

    assert result.coverage == "complete"
    assert [row.kind for row in result.support_evidence] == ["exception"]


def test_explanation_review_routes_through_accounting_as_strict_drift_purpose():
    from app.service import (
        CharacterConsistencyUsageAccumulator,
        _CharacterConsistencyAccountingProvider,
    )

    candidate = _candidate()
    configured = _settings(character_explanation_review_v1=True)
    drift = SequenceProvider(_response((candidate,)))
    usage = CharacterConsistencyUsageAccumulator()
    accounting = _CharacterConsistencyAccountingProvider(
        configured,
        usage,
        signal_provider=SequenceProvider(),
        drift_provider=drift,
    )

    result = _run((candidate,), accounting, settings=configured)

    system, user = build_character_explanation_review_prompts(
        (candidate,), baseline=_baseline(), observations=_observations()
    )
    expected_charge = estimate_issue_evidence_review_tokens(
        system,
        user,
        completion_reserve=configured.character_explanation_max_completion_tokens,
    )
    assert result.coverage == "complete"
    assert drift.calls == [(EXPLANATION_REVIEW_SYSTEM_PROMPT, user)]
    assert result.diagnostics.usage == {
        "estimated_tokens": expected_charge,
        "attempted_calls": 1,
        "prompt_tokens": 13,
        "completion_tokens": 7,
        "charged_tokens": expected_charge,
    }
    assert usage.safe_dict(terminal_status="completed") == {
        "completeness": "completed_calls",
        "scope": "character_consistency",
        "terminal_status": "completed",
        "logical_calls": 1,
        "prompt_tokens": 13,
        "completion_tokens": 7,
        "charged_tokens": expected_charge,
        "charged_token_semantics": "heuristic_or_reported_internal_debit",
        "provider_calls": None,
    }
    assert usage.successful_calls == 1

    with pytest.raises(
        RuntimeError,
        match="unsupported character consistency provider purpose",
    ):
        accounting.complete(EXPLANATION_REVIEW_SYSTEM_PROMPT + "\n", "unknown")
    assert len(drift.calls) == 1
    assert usage.logical_calls == 1


@pytest.mark.parametrize(
    "applicable",
    [
        ["C99"],
        ["C01", "C01"],
    ],
)
def test_forged_or_duplicate_observation_binding_fails_the_whole_batch(applicable):
    candidate = _candidate()
    raw = _response(
        (candidate,),
        {candidate.citation: {"applicable_observation_citations": applicable}},
    )

    result = _run((candidate,), SequenceProvider(raw, raw))

    assert result.support_evidence == ()
    assert result.coverage == "partial"
    assert result.diagnostics.outcome == "degraded"
    assert result.diagnostics.failure_reasons == ("response_invalid",)


def test_candidate_with_no_applicable_current_observation_emits_no_support():
    candidate = _candidate()
    raw = _response(
        (candidate,),
        {candidate.citation: {"applicable_observation_citations": []}},
    )

    result = _run((candidate,), SequenceProvider(raw))

    assert result.coverage == "complete"
    assert result.support_evidence == ()


def test_irrelevant_candidate_is_valid_but_can_never_emit_g_x_or_p():
    candidate = _candidate()
    raw = _response(
        (candidate,),
        {
            candidate.citation: {
                "explanation_type": "irrelevant",
                "causal_relation": "none",
                "applicable_observation_citations": [],
            }
        },
    )

    result = _run((candidate,), SequenceProvider(raw))

    assert result.coverage == "complete"
    assert result.support_evidence == ()
    assert result.diagnostics.reviewed_candidates == 1


def test_mixed_batch_keeps_relevant_support_and_discards_irrelevant_candidate():
    candidates = (_candidate("E01"), _candidate("E02", line=4))
    raw = _response(
        candidates,
        {
            "E02": {
                "explanation_type": "irrelevant",
                "causal_relation": "none",
                "applicable_observation_citations": [],
            }
        },
    )

    result = _run(candidates, SequenceProvider(raw))

    assert result.coverage == "complete"
    assert result.diagnostics.reviewed_candidates == 2
    assert len(result.support_evidence) == 1
    assert result.support_evidence[0].evidence.document_id == "history-E01"


@pytest.mark.parametrize(
    "updates",
    [
        {"causal_relation": "bounded_inference", "applicable_observation_citations": []},
        {"causal_relation": "none", "applicable_observation_citations": ["C01"]},
    ],
)
def test_irrelevant_with_causal_or_observation_slots_fails_closed(updates):
    candidate = _candidate()
    raw = _response(
        (candidate,),
        {candidate.citation: {"explanation_type": "irrelevant", **updates}},
    )

    result = _run((candidate,), SequenceProvider(raw, raw))

    assert result.coverage == "partial"
    assert result.support_evidence == ()
    assert result.diagnostics.contract_failure_counts == {
        "irrelevant_slots_invalid": 2
    }


def test_missing_observation_binding_field_fails_the_whole_batch():
    candidate = _candidate()
    item = _item(candidate)
    item.pop("applicable_observation_citations")
    raw = _response_envelope((candidate,), [item])

    result = _run((candidate,), SequenceProvider(raw, raw))

    assert result.support_evidence == ()
    assert result.coverage == "partial"
    assert result.diagnostics.failure_reasons == ("response_invalid",)


@pytest.mark.parametrize(
    "updates",
    [
        {"actor_relation": "different"},
        {"axis_relation": "different"},
        {"actuality": "reported"},
        {"actuality": "hypothetical"},
        {"temporal_relation": "future"},
    ],
)
def test_wrong_actor_axis_nonactual_or_future_never_emits_support(updates):
    candidate = _candidate()
    raw = _response((candidate,), {candidate.citation: updates})
    result = _run((candidate,), SequenceProvider(raw))
    assert result.coverage == "complete"
    assert result.support_evidence == ()


def test_published_source_uses_frozen_release_scope_not_import_ordinal():
    candidate = _candidate(source_ordinal=21)
    result = _run((candidate,), SequenceProvider(_response((candidate,))))
    assert [row.kind for row in result.support_evidence] == ["causal_bridge"]
    assert result.coverage == "complete"


def test_later_same_draft_bounded_inference_cannot_explain_earlier_observation():
    candidate = _candidate("E01", source_kind="draft", line=21)
    raw = _response(
        (candidate,),
        {candidate.citation: {"causal_relation": "bounded_inference"}},
    )

    result = _run((candidate,), SequenceProvider(raw))

    assert result.coverage == "complete"
    assert result.support_evidence == ()


def test_window_starting_before_but_ending_after_current_needs_explicit_retrospective():
    candidate = _candidate(
        "E01",
        source_kind="draft",
        evidence=EvidenceSpan(
            document_id="draft-current",
            document_name="draft-current.md",
            line_start=19,
            line_end=21,
            text="前置场景。\n林澈主动发言。\n后来发生了另一件事。",
        ),
    )
    raw = _response(
        (candidate,),
        {candidate.citation: {"causal_relation": "bounded_inference"}},
    )

    result = _run((candidate,), SequenceProvider(raw))

    assert result.support_evidence == ()


def test_later_same_draft_explicit_retrospective_can_bind_exact_observation():
    candidate = _candidate(
        "E01",
        source_kind="draft",
        line=21,
        evidence=_evidence(
            "draft-current",
            21,
            "补记：林澈在上一场合主动发言，是因为当时正在伪装身份。",
        ),
    )
    raw = _response(
        (candidate,),
        {candidate.citation: {"explanation_type": "disguise_or_role"}},
    )

    result = _run((candidate,), SequenceProvider(raw))

    assert [row.kind for row in result.support_evidence] == ["exception"]
    assert result.support_evidence[0].applicable_observation_ids == (
        "cs_" + "1" * 32,
    )


def test_one_bounded_inference_stays_possible():
    candidate = _candidate()
    raw = _response(
        (candidate,), {candidate.citation: {"causal_relation": "bounded_inference"}}
    )
    result = _run((candidate,), SequenceProvider(raw))
    assert [(row.kind, row.explicit) for row in result.support_evidence] == [
        ("possible_explanation", False)
    ]


def test_same_actor_and_axis_with_no_causal_relation_emits_nothing():
    candidate = _candidate()
    raw = _response(
        (candidate,), {candidate.citation: {"causal_relation": "none"}}
    )
    result = _run((candidate,), SequenceProvider(raw))
    assert result.coverage == "complete"
    assert result.support_evidence == ()


def test_even_separate_bounded_inferences_remain_possible():
    candidates = (_candidate("E01", line=3), _candidate("E02", line=7))
    raw = _response(
        candidates,
        {
            "E01": {"causal_relation": "bounded_inference"},
            "E02": {"causal_relation": "bounded_inference"},
        },
    )
    result = _run(candidates, SequenceProvider(raw))
    assert [row.kind for row in result.support_evidence] == [
        "possible_explanation",
        "possible_explanation",
    ]
    assert all(not row.explicit for row in result.support_evidence)


def test_repeated_same_evidence_cannot_satisfy_independence_or_duplicate_output():
    same = _evidence("history-same", 3, "训练后林澈逐渐愿意公开表达。")
    candidates = (
        _candidate("E01", evidence=same),
        _candidate("E02", evidence=same.model_copy(deep=True)),
    )
    raw = _response(
        candidates,
        {
            "E01": {"causal_relation": "bounded_inference"},
            "E02": {"causal_relation": "bounded_inference"},
        },
    )
    result = _run(candidates, SequenceProvider(raw))
    assert len(result.support_evidence) == 1
    assert result.support_evidence[0].kind == "possible_explanation"


def test_same_evidence_with_different_current_bindings_is_not_overwritten():
    same = _evidence("history-same", 3, "旧事明确解释了对应场合的公开表达。")
    candidates = (
        _candidate("E01", evidence=same),
        _candidate("E02", evidence=same.model_copy(deep=True)),
    )
    observations = (
        *_observations(),
        ExplanationObservationSummary(
            citation="C02",
            observation_id="cs_" + "2" * 32,
            statement="林澈次日再次向同伴公开讲述恐惧。",
            document_id="draft-current",
            source_ordinal=20,
            evidence=_evidence(
                "draft-current", 30, "林澈次日再次向同伴公开讲述恐惧。"
            ),
        ),
    )
    raw = _response(
        candidates,
        {
            "E02": {"applicable_observation_citations": ["C02"]},
        },
        observations=observations,
    )

    result = run_character_explanation_review(
        candidates,
        baseline=_baseline(),
        observations=observations,
        provider=SequenceProvider(raw),
        settings=_settings(),
    )

    assert len(result.support_evidence) == 2
    assert {
        row.applicable_observation_ids for row in result.support_evidence
    } == {
        ("cs_" + "1" * 32,),
        ("cs_" + "2" * 32,),
    }


def test_overlapping_or_adjacent_same_document_inferences_never_promote():
    first = EvidenceSpan(
        document_id="history-shared",
        document_name="history-shared.md",
        line_start=3,
        line_end=4,
        text="林澈参加训练。\n训练涉及公开表达。",
    )
    second = EvidenceSpan(
        document_id="history-shared",
        document_name="history-shared.md",
        line_start=4,
        line_end=5,
        text="训练涉及公开表达。\n林澈完成训练。",
    )
    candidates = (
        _candidate("E01", evidence=first),
        _candidate("E02", evidence=second),
    )
    raw = _response(
        candidates,
        {
            "E01": {"causal_relation": "bounded_inference"},
            "E02": {"causal_relation": "bounded_inference"},
        },
    )

    result = _run(candidates, SequenceProvider(raw))

    assert [row.kind for row in result.support_evidence] == [
        "possible_explanation",
        "possible_explanation",
    ]


def test_foreshadowing_is_always_possible_even_when_model_calls_it_explicit():
    candidate = _candidate()
    raw = _response(
        (candidate,),
        {candidate.citation: {"explanation_type": "foreshadowing_or_ambiguous"}},
    )
    result = _run((candidate,), SequenceProvider(raw))
    assert result.support_evidence[0].kind == "possible_explanation"
    assert result.support_evidence[0].explicit is False


@pytest.mark.parametrize(
    ("actuality", "temporal_relation"),
    [
        ("reported", "prior_or_active"),
        ("hypothetical", "future"),
        ("ambiguous", "ambiguous"),
    ],
)
def test_bound_foreshadowing_can_survive_nonactual_or_future_slots_as_p(
    actuality: str,
    temporal_relation: str,
):
    candidate = _candidate()
    raw = _response(
        (candidate,),
        {
            candidate.citation: {
                "explanation_type": "foreshadowing_or_ambiguous",
                "actuality": actuality,
                "temporal_relation": temporal_relation,
                "causal_relation": "ambiguous",
            }
        },
    )

    result = _run((candidate,), SequenceProvider(raw))

    assert [(row.kind, row.explicit) for row in result.support_evidence] == [
        ("possible_explanation", False)
    ]
    assert result.support_evidence[0].applicable_observation_ids == (
        "cs_" + "1" * 32,
    )


def test_prior_same_draft_future_plan_can_remain_a_bound_p_lead():
    candidate = _candidate("E01", source_kind="draft", line=2)
    raw = _response(
        (candidate,),
        {
            candidate.citation: {
                "explanation_type": "foreshadowing_or_ambiguous",
                "actuality": "hypothetical",
                "temporal_relation": "future",
                "causal_relation": "ambiguous",
            }
        },
    )

    result = _run((candidate,), SequenceProvider(raw))

    assert [row.kind for row in result.support_evidence] == [
        "possible_explanation"
    ]


def test_bound_foreshadowing_with_no_causal_claim_remains_a_p_lead():
    candidate = _candidate()
    raw = _response(
        (candidate,),
        {
            candidate.citation: {
                "explanation_type": "foreshadowing_or_ambiguous",
                "actuality": "hypothetical",
                "temporal_relation": "future",
                "causal_relation": "none",
            }
        },
    )

    result = _run((candidate,), SequenceProvider(raw))

    assert [(row.kind, row.explicit) for row in result.support_evidence] == [
        ("possible_explanation", False)
    ]


def test_later_same_draft_foreshadowing_cannot_bypass_retrospective_gate():
    candidate = _candidate("E01", source_kind="draft", line=21)
    raw = _response(
        (candidate,),
        {
            candidate.citation: {
                "explanation_type": "foreshadowing_or_ambiguous",
                "actuality": "hypothetical",
                "temporal_relation": "future",
                # Even this strongest causal slot cannot make a future plan
                # written after C into a retrospective explanation of C.
                "causal_relation": "explicit_causal",
            }
        },
    )

    result = _run((candidate,), SequenceProvider(raw))

    assert result.support_evidence == ()


@pytest.mark.parametrize(
    "updates",
    [
        {"actuality": "hypothetical"},
        {"temporal_relation": "future"},
    ],
)
def test_bounded_inference_still_requires_actual_prior_or_active(updates):
    candidate = _candidate()
    raw = _response(
        (candidate,),
        {
            candidate.citation: {
                "causal_relation": "bounded_inference",
                **updates,
            }
        },
    )

    result = _run((candidate,), SequenceProvider(raw))

    assert result.support_evidence == ()


def test_draft_can_prove_explicit_exception_but_not_growth_or_bounded_inference():
    growth = _candidate("E01", source_kind="draft", line=2)
    disguise = _candidate("E02", source_kind="draft", line=3)
    bounded_state = _candidate("E03", source_kind="draft", line=4)
    candidates = (growth, disguise, bounded_state)
    raw = _response(
        candidates,
        {
            "E02": {"explanation_type": "disguise_or_role"},
            "E03": {
                "explanation_type": "temporary_state_or_pressure",
                "causal_relation": "bounded_inference",
            },
        },
    )
    result = _run(candidates, SequenceProvider(raw))
    assert [row.kind for row in result.support_evidence] == [
        "possible_explanation",
        "exception",
        "possible_explanation",
    ]


def test_failed_second_batch_keeps_first_batch_and_marks_partial_without_raw_text():
    candidates = tuple(_candidate(f"E{index:02d}", line=index) for index in range(1, 6))
    secret = "private-provider-secret"
    provider = SequenceProvider(_response(candidates[:4]), RuntimeError(secret))
    result = _run(candidates, provider)
    assert result.coverage == "partial"
    assert result.diagnostics.outcome == "partial"
    assert (result.diagnostics.completed_batches, result.diagnostics.failed_batches) == (1, 1)
    assert len(result.support_evidence) == 4
    serialized = json.dumps(result.diagnostics.safe_dict())
    assert secret not in serialized
    assert "provider_error" in serialized


def test_contract_retry_recovers_without_echoing_rejected_provider_text():
    candidate = _candidate()
    secret = "private-provider-response-marker"
    invalid_item = _item(candidate)
    invalid_item["unexpected"] = secret
    invalid = _response_envelope((candidate,), [invalid_item])
    provider = SequenceProvider(invalid, _response((candidate,)))

    result = _run((candidate,), provider)

    assert result.coverage == "complete"
    assert result.diagnostics.failure_reasons == ()
    assert result.diagnostics.attempted_calls == 2
    assert result.diagnostics.contract_failure_counts == {"item_schema_invalid": 1}
    assert result.diagnostics.contract_regeneration_attempted_calls == 1
    assert result.diagnostics.contract_regeneration_recovered_batches == 1
    assert len(result.support_evidence) == 1
    assert secret not in json.dumps(result.diagnostics.safe_dict())
    assert all(secret not in system + user for system, user in provider.calls)
    retry_payload = json.loads(
        provider.calls[1][1].removeprefix(EXPLANATION_REVIEW_USER_PREFIX)
    )
    assert retry_payload["validation_retry"] == {
        "expected_candidate_citations": ["E01"],
        "expected_request_digest": retry_payload["request_digest"],
        "expected_schema_version": EXPLANATION_REVIEW_SCHEMA_V2,
        "failure_code": "item_schema_invalid",
        "required_item_keys": [
            "citation",
            "explanation_type",
            "actuality",
            "actor_relation",
            "axis_relation",
            "temporal_relation",
            "causal_relation",
            "applicable_observation_citations",
        ],
        "required_top_level_keys": ["schema_version", "request_digest", "items"],
    }


@pytest.mark.parametrize(
    ("top_updates", "failure_code"),
    [
        (
            {"schema_version": "character-explanation-review-v1"},
            "schema_version_invalid",
        ),
        ({"request_digest": "0" * 64}, "request_digest_invalid"),
    ],
)
def test_v1_or_wrong_request_digest_is_rejected_and_counted_safely(
    top_updates,
    failure_code,
):
    candidate = _candidate()
    invalid = _response_envelope(
        (candidate,), [_item(candidate)], top_updates=top_updates
    )

    result = _run((candidate,), SequenceProvider(invalid, invalid))

    assert result.coverage == "partial"
    assert result.support_evidence == ()
    assert result.diagnostics.contract_failure_counts == {failure_code: 2}
    assert json.dumps(result.diagnostics.safe_dict()).count("林澈") == 0


def test_digest_mismatch_contract_regeneration_can_recover():
    candidate = _candidate()
    invalid = _response_envelope(
        (candidate,),
        [_item(candidate)],
        top_updates={"request_digest": "0" * 64},
    )
    valid = _response((candidate,))

    result = _run((candidate,), SequenceProvider(invalid, valid))

    assert result.coverage == "complete"
    assert len(result.support_evidence) == 1
    assert result.diagnostics.contract_failure_counts == {
        "request_digest_invalid": 1
    }
    assert result.diagnostics.contract_regeneration_recovered_batches == 1


@pytest.mark.parametrize(
    ("mutate", "failure_code"),
    [
        (lambda value: {**value, "unexpected": True}, "item_schema_invalid"),
        (
            lambda value: {**value, "actuality": "invented"},
            "item_enum_invalid",
        ),
    ],
)
def test_pydantic_contract_failures_use_bounded_subclasses(mutate, failure_code):
    candidate = _candidate()
    invalid = _response_envelope((candidate,), [mutate(_item(candidate))])

    result = _run((candidate,), SequenceProvider(invalid, invalid))

    assert result.coverage == "partial"
    assert result.diagnostics.contract_failure_counts == {failure_code: 2}


def test_two_contract_invalid_responses_fail_closed_with_safe_counts():
    candidate = _candidate()
    invalid_item = _item(candidate)
    invalid_item.pop("axis_relation")
    invalid = _response_envelope((candidate,), [invalid_item])
    provider = SequenceProvider(invalid, invalid)

    result = _run((candidate,), provider)

    assert result.coverage == "partial"
    assert result.support_evidence == ()
    assert result.diagnostics.failure_reasons == ("response_invalid",)
    assert result.diagnostics.contract_failure_counts == {"item_schema_invalid": 2}
    assert result.diagnostics.contract_regeneration_attempted_calls == 1
    assert result.diagnostics.contract_regeneration_recovered_batches == 0
    assert len(provider.calls) == 2


def test_exact_candidate_set_is_normalized_to_server_order():
    candidates = (_candidate("E01"), _candidate("E02", line=4))
    reversed_response = _response_envelope(
        candidates,
        [_item(candidates[1]), _item(candidates[0])],
    )

    result = _run(candidates, SequenceProvider(reversed_response))

    assert result.coverage == "complete"
    assert result.diagnostics.citation_order_normalized_batches == 1
    assert [row.evidence.document_id for row in result.support_evidence] == [
        "history-E01",
        "history-E02",
    ]


@pytest.mark.parametrize(
    "items",
    [
        lambda candidates: [_item(candidates[0])],
        lambda candidates: [
            _item(candidates[0]),
            {**_item(candidates[1]), "citation": "E99"},
        ],
    ],
)
def test_missing_or_forged_candidate_set_still_fails_closed(items):
    candidates = (_candidate("E01"), _candidate("E02", line=4))
    invalid = _response_envelope(candidates, items(candidates))

    result = _run(candidates, SequenceProvider(invalid, invalid))

    assert result.coverage == "partial"
    assert result.support_evidence == ()
    assert result.diagnostics.failure_reasons == ("response_invalid",)
    assert result.diagnostics.contract_failure_counts == {
        "candidate_citations_invalid": 2
    }


def test_nine_candidates_use_four_four_one_batches_and_sum_all_call_usage():
    candidates = tuple(
        _candidate(f"E{index:02d}", line=index) for index in range(1, 10)
    )
    provider = SequenceProvider(
        _response(candidates[:4]),
        _response(candidates[4:8]),
        _response(candidates[8:]),
    )

    result = _run(candidates, provider)

    assert result.coverage == "complete"
    assert result.diagnostics.batch_count == 3
    assert result.diagnostics.completed_batches == 3
    assert result.diagnostics.failed_batches == 0
    assert result.diagnostics.attempted_calls == 3
    assert result.diagnostics.prompt_tokens == 39
    assert result.diagnostics.completion_tokens == 21
    assert len(result.support_evidence) == 9
    packets = [
        json.loads(user.removeprefix(EXPLANATION_REVIEW_USER_PREFIX))
        for _, user in provider.calls
    ]
    assert [len(packet["candidates"]) for packet in packets] == [4, 4, 1]


def test_all_provider_failure_is_degraded_and_fail_closed():
    candidate = _candidate()
    provider = SequenceProvider(RuntimeError("private raw"))
    result = _run((candidate,), provider)
    assert result.coverage == "partial"
    assert result.diagnostics.outcome == "degraded"
    assert result.support_evidence == ()
    assert result.diagnostics.failure_reasons == ("provider_error",)
    assert result.diagnostics.contract_regeneration_attempted_calls == 0
    assert len(provider.calls) == 1


def test_one_character_evidence_keeps_valid_summary_and_usage():
    candidate = _candidate(evidence=_evidence("history-short", 3, "变"))

    result = _run((candidate,), SequenceProvider(_response((candidate,))))

    assert result.diagnostics.outcome == "complete"
    assert result.diagnostics.charged_tokens > 0
    assert result.support_evidence[0].summary == "变。"


def test_promotion_failure_returns_fail_closed_diagnostics(monkeypatch):
    candidate = _candidate()

    def fail_promotion(*_args, **_kwargs):
        raise ValueError("private conversion detail")

    monkeypatch.setattr(
        explanation_review_module, "_make_support", fail_promotion
    )
    result = _run((candidate,), SequenceProvider(_response((candidate,))))

    assert result.coverage == "partial"
    assert result.support_evidence == ()
    assert result.diagnostics.outcome == "degraded"
    assert result.diagnostics.attempted_calls == 1
    assert result.diagnostics.charged_tokens > 0
    assert result.diagnostics.failure_reasons == ("response_invalid",)


def test_shared_budget_admits_first_batch_but_not_second():
    candidates = tuple(_candidate(f"E{index:02d}", line=index) for index in range(1, 6))
    settings = _settings()
    system, user = build_character_explanation_review_prompts(
        candidates[:4], baseline=_baseline(), observations=_observations()
    )
    first_estimate = estimate_issue_evidence_review_tokens(
        system,
        user,
        completion_reserve=settings.character_explanation_max_completion_tokens,
    )
    limited = settings.model_copy(
        update={"character_explanation_token_budget": first_estimate}
    )
    provider = SequenceProvider(_response(candidates[:4]))
    result = _run(candidates, provider, settings=limited)
    assert len(provider.calls) == 1
    assert result.coverage == "partial"
    assert result.diagnostics.failure_reasons == ("token_budget",)
    assert len(result.support_evidence) == 4


def test_contract_regeneration_must_pass_shared_budget_again():
    candidate = _candidate()
    invalid_item = _item(candidate)
    invalid_item.pop("axis_relation")
    invalid = _response_envelope((candidate,), [invalid_item])
    settings = _settings()
    system, user = build_character_explanation_review_prompts(
        (candidate,), baseline=_baseline(), observations=_observations()
    )
    first_estimate = estimate_issue_evidence_review_tokens(
        system,
        user,
        completion_reserve=settings.character_explanation_max_completion_tokens,
    )
    limited = settings.model_copy(
        update={"character_explanation_token_budget": first_estimate}
    )
    provider = SequenceProvider(invalid)

    result = _run((candidate,), provider, settings=limited)

    assert result.coverage == "partial"
    assert result.diagnostics.failure_reasons == ("token_budget",)
    assert result.diagnostics.contract_failure_counts == {"item_schema_invalid": 1}
    assert result.diagnostics.attempted_calls == 1
    assert result.diagnostics.contract_regeneration_attempted_calls == 0
    assert result.diagnostics.estimated_tokens > first_estimate
    assert len(provider.calls) == 1


def test_contract_regeneration_rechecks_shared_deadline(monkeypatch):
    candidate = _candidate()
    invalid_item = _item(candidate)
    invalid_item.pop("axis_relation")
    invalid = _response_envelope((candidate,), [invalid_item])
    provider = SequenceProvider(invalid)
    monotonic_values = iter((0.0, 0.0, 61.0))
    monkeypatch.setattr(
        explanation_review_module.time,
        "monotonic",
        lambda: next(monotonic_values),
    )

    result = _run((candidate,), provider)

    assert result.coverage == "partial"
    assert result.diagnostics.failure_reasons == ("deadline",)
    assert result.diagnostics.contract_failure_counts == {"item_schema_invalid": 1}
    assert result.diagnostics.attempted_calls == 1
    assert result.diagnostics.contract_regeneration_attempted_calls == 0
    assert len(provider.calls) == 1


def test_provider_payload_uses_independent_explanation_completion_limit():
    candidate = _candidate()
    response_text = _response((candidate,))
    observed_max_tokens: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        observed_max_tokens.append(payload["max_tokens"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 13, "completion_tokens": 7},
            },
        )

    configured = _settings(
        character_drift_max_completion_tokens=512,
        character_explanation_max_completion_tokens=2_048,
    )
    provider = OpenAICompatibleProvider(
        configured,
        transport=httpx.MockTransport(handler),
    )

    result = _run((candidate,), provider, settings=configured)

    assert result.coverage == "complete"
    assert observed_max_tokens == [2_048]


def test_explanation_completion_limit_is_finite_and_budget_bound():
    assert _settings().character_explanation_max_completion_tokens == 2_048
    with pytest.raises(ValidationError):
        _settings(character_explanation_max_completion_tokens=4_097)
    with pytest.raises(ValidationError, match="explanation budget"):
        _settings(
            character_explanation_token_budget=1_000,
            character_explanation_max_completion_tokens=2_048,
        )


def test_explanation_completion_reserve_only_binds_stage_when_feature_is_enabled():
    overrides = {
        "character_consistency_stage_token_budget": 1_500,
        "character_signal_max_completion_tokens": 64,
        "character_drift_max_completion_tokens": 64,
    }

    disabled = _settings(character_explanation_review_v1=False, **overrides)
    assert disabled.character_consistency_stage_token_budget == 1_500
    with pytest.raises(ValidationError, match="stage budget"):
        _settings(character_explanation_review_v1=True, **overrides)


def test_duplicate_or_nonwhitelisted_response_citations_fail_whole_batch():
    candidates = (_candidate("E01"), _candidate("E02", line=4))
    duplicate = _response_envelope(
        candidates, [_item(candidates[0]), _item(candidates[0])]
    )
    result = _run(candidates, SequenceProvider(duplicate, duplicate))
    assert result.diagnostics.outcome == "degraded"
    assert result.diagnostics.failure_reasons == ("response_invalid",)
    assert result.support_evidence == ()


def test_all_protocol_models_are_strict_and_forbid_extra_fields():
    payload = _candidate().model_dump()
    payload["unexpected"] = "model-owned"
    with pytest.raises(ValidationError):
        ExplanationCandidate.model_validate(payload)
    payload = _candidate().model_dump()
    payload["source_ordinal"] = True
    with pytest.raises(ValidationError):
        ExplanationCandidate.model_validate(payload)
