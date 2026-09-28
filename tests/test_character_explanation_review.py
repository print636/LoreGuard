from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import app.character_explanation_review as explanation_review_module
from app.character_explanation_review import (
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


def _response(candidates, updates=None) -> str:
    updates = updates or {}
    return json.dumps(
        {
            "items": [
                _item(candidate, **updates.get(candidate.citation, {}))
                for candidate in candidates
            ]
        },
        ensure_ascii=False,
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
    assert "不得输出理由、摘要、置信度" in system
    assert "这种绑定只会形成 P 线索，不证明因果或解释成立" in system


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
        completion_reserve=configured.character_drift_max_completion_tokens,
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

    result = _run((candidate,), SequenceProvider(raw))

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


def test_missing_observation_binding_field_fails_the_whole_batch():
    candidate = _candidate()
    item = _item(candidate)
    item.pop("applicable_observation_citations")
    raw = json.dumps({"items": [item]}, ensure_ascii=False)

    result = _run((candidate,), SequenceProvider(raw))

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
    candidates = tuple(_candidate(f"E{index:02d}", line=index) for index in range(1, 10))
    secret = "private-provider-secret"
    provider = SequenceProvider(_response(candidates[:8]), RuntimeError(secret))
    result = _run(candidates, provider)
    assert result.coverage == "partial"
    assert result.diagnostics.outcome == "partial"
    assert (result.diagnostics.completed_batches, result.diagnostics.failed_batches) == (1, 1)
    assert len(result.support_evidence) == 8
    serialized = json.dumps(result.diagnostics.safe_dict())
    assert secret not in serialized
    assert "provider_error" in serialized


def test_all_provider_failure_is_degraded_and_fail_closed():
    candidate = _candidate()
    result = _run((candidate,), SequenceProvider(RuntimeError("private raw")))
    assert result.coverage == "partial"
    assert result.diagnostics.outcome == "degraded"
    assert result.support_evidence == ()
    assert result.diagnostics.failure_reasons == ("provider_error",)


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
    candidates = tuple(_candidate(f"E{index:02d}", line=index) for index in range(1, 10))
    settings = _settings()
    system, user = build_character_explanation_review_prompts(
        candidates[:8], baseline=_baseline(), observations=_observations()
    )
    first_estimate = estimate_issue_evidence_review_tokens(
        system,
        user,
        completion_reserve=settings.character_drift_max_completion_tokens,
    )
    limited = settings.model_copy(
        update={"character_explanation_token_budget": first_estimate}
    )
    provider = SequenceProvider(_response(candidates[:8]))
    result = _run(candidates, provider, settings=limited)
    assert len(provider.calls) == 1
    assert result.coverage == "partial"
    assert result.diagnostics.failure_reasons == ("token_budget",)
    assert len(result.support_evidence) == 8


def test_duplicate_or_nonwhitelisted_response_citations_fail_whole_batch():
    candidates = (_candidate("E01"), _candidate("E02", line=4))
    duplicate = json.dumps({"items": [_item(candidates[0]), _item(candidates[0])]})
    result = _run(candidates, SequenceProvider(duplicate))
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
