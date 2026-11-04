"""Ordinary formal-profile omissions recovered without inventing canon."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from app.character_scope_review import ScopeReviewSourceIdentity
from app.character_trait_extraction import (
    BASELINE_COVERAGE_SCHEMA_V1,
    BASELINE_COVERAGE_SYSTEM_PROMPT_V3,
    CharacterSignalChunk,
    CharacterSignalExtractor,
    build_baseline_coverage_system_prompt,
    build_formal_character_signal_system_prompt,
    _assertion_index_v1,
    _recover_baseline_coverage,
)
from tests.test_character_scope_review_pipeline import ReviewingProvider, _settings


def _record(text, *, line=1, dimension="preference", trait_key="fruit_preference",
            key_object="枇杷", character="林蔚", stability="stable"):
    return {
        "character": character, "dimension": dimension, "trait_key": trait_key,
        "statement": text.rstrip("。"), "polarity": "positive",
        "stability": stability, "observation_kind": "explicit_declaration",
        "context": "", "key_object": key_object,
        "source_line_start": line, "source_line_end": line, "evidence": text,
        "support_id": f"L{line}:A1", "actor_anchor_id": "", "label_anchor_id": "",
        "scope_relation": "local",
    }


def _audit(records=(), *, choices=None):
    return {
        "schema_version": BASELINE_COVERAGE_SCHEMA_V1,
        "assessments": choices if choices is not None else [
            {"support_id": row["support_id"], "disposition": "record"} for row in records
        ],
        "records": list(records),
    }


class RecoveryProvider(ReviewingProvider):
    def __init__(self, primary, audit, verdicts=None):
        super().__init__(list(primary), verdicts)
        self.audit = audit
        self.audit_calls = 0

    def complete(self, system, user):
        if BASELINE_COVERAGE_SYSTEM_PROMPT_V3 in system:
            self.calls.append((system, user))
            self.audit_calls += 1
            if isinstance(self.audit, Exception):
                raise self.audit
            return SimpleNamespace(
                text=self.audit if isinstance(self.audit, str) else json.dumps(self.audit, ensure_ascii=False),
                prompt_tokens=19, completion_tokens=7,
            )
        return super().complete(system, user)


def _extract(source, provider, **options):
    chunk = CharacterSignalChunk("original-profile", "profile.md", source, 1, "formal_character_profile")
    source_identity = ScopeReviewSourceIdentity(
        run_input_id="frozen-profile-input", document_id=chunk.document_id,
        document_version=4, content_sha256=hashlib.sha256(source.encode()).hexdigest(),
    )
    result = CharacterSignalExtractor(provider, settings=_settings(
        character_signal_token_budget=40000, character_signal_package_max_attempts=1,
    )).extract(chunk, source_identity=source_identity, frozen_content=source,
               baseline_coverage_recovery=True, **options)
    return result


def test_omitted_stable_speech_is_recovered_and_independently_source_reviewed():
    first = "林蔚长期喜欢枇杷。"
    second = "林蔚一贯说话简洁直接。"
    speech = _record(second, line=2, dimension="speech_pattern", trait_key="directness", key_object="")
    provider = RecoveryProvider([_record(first)], _audit([speech]))
    result = _extract(first + "\n" + second, provider)

    assert {row.dimension for row in result.pending_candidates} == {"preference", "speech_pattern"}
    assert all(row.status == "pending" for row in result.pending_candidates)
    assert provider.audit_calls == 1
    assert provider.calls[1][0] == build_baseline_coverage_system_prompt(
        build_formal_character_signal_system_prompt(scope_review_v1=True)
    )
    assert len(provider.calls) == 3  # primary, one audit, one shared scope review
    trace = result.diagnostics.baseline_coverage_trace
    assert (trace.status, trace.added_records, trace.recovered_support_slots) == ("audited", 1, (2,))
    assert result.diagnostics.prompt_tokens == 17 + 19 + 13
    assert result.diagnostics.completion_tokens == 9 + 7 + 11
    assert result.diagnostics.attempted_calls == 3
    assert result.diagnostics.support_trace.final_accepted_slots == (1,)
    assert result.signals[1].source_run_input_id == "frozen-profile-input"
    assert result.pending_candidates[1].support_refs[0].run_input_id == "frozen-profile-input"
    assert provider.review_payload["request"]["document_version"] == 4
    assert "baseline_coverage_trace" not in result.diagnostics.model_dump()


def test_no_unrepresented_assertion_does_not_make_an_audit_call():
    text = "林蔚长期喜欢枇杷。"
    provider = RecoveryProvider([_record(text)], RuntimeError("must not call audit"))
    result = _extract(text, provider)
    assert result.diagnostics.baseline_coverage_trace.status == "covered"
    assert provider.audit_calls == 0
    assert len(result.pending_candidates) == 1
    assert len(provider.calls) == 2


@pytest.mark.parametrize("one_off", [False, True])
def test_rule_scope_remains_attached_without_promoting_a_one_off_decision(one_off):
    condition = "值班期间"
    fact = "林蔚这一次临时拒绝借出钥匙" if one_off else "林蔚一律拒绝借出钥匙"
    text = f"{condition}，{fact}。"
    row = dict(_record(text, dimension="behavior_boundary", trait_key="钥匙借用", key_object="钥匙",
                       stability="situational"), support_id="L1:A2", statement=fact)
    audit = _audit(
        [] if one_off else [dict(row, stability="stable")],
        choices=[
            {"support_id": "L1:A1", "disposition": "no_additional_trait"},
            {"support_id": "L1:A2", "disposition": "no_additional_trait" if one_off else "record"},
        ],
    )
    provider = RecoveryProvider([row], audit)
    result = _extract(text, provider)
    assert provider.audit_calls == 1 and provider.review_payload is not None
    if one_off:
        assert result.pending_candidates == ()
        assert result.signals[0].stability == "situational"
    else:
        candidate, = result.pending_candidates
        assert candidate.contexts == (condition,)
        assert candidate.statement == fact and candidate.evidence[0].text == text
        assert candidate.status == "pending" and candidate.support_refs[0].support_id == "L1:A2"
        assert provider.review_payload["request"]["proposals"][0]["context"] == condition
        assert result.diagnostics.baseline_coverage_trace.recovered_support_slots == (2,)


@pytest.mark.parametrize(("context", "review", "relation"), [
    ("演练当天，主管批准时例外", "supported", "supported"),
    ("演练日时", "rejected", "contradicted"),
    ("正式演出日", "rejected", "contradicted"),
    ("演练期间，主管或值班人员同意时除外", "uncertain", "ambiguous"),
])
def test_qualified_boundary_keeps_the_actual_proposal_through_independent_review(
    context, review, relation,
):
    fact = "岑青始终拒绝交出印章"
    text = f"演练日时，{fact}；除非演练主管许可。"
    row = dict(
        _record(text, character="岑青", dimension="behavior_boundary",
                trait_key="seal_handover", key_object="印章"),
        statement=fact, support_id="L1:A2", context=context, polarity="negative",
    )
    choices = [
        {"support_id": slot, "disposition": "no_additional_trait"}
        for slot in ("L1:A1", "L1:A3")
    ]
    # Slot values are fixed here. Exercise source binding and admission,
    # including a valid paraphrase that is not a literal source substring.
    provider = RecoveryProvider([row], _audit(choices=choices),
                                [{"verdict": review, "statement_relation": relation}])
    result = _extract(text, provider)
    request = provider.review_payload["request"]
    assert request["lines"][0]["text"] == text
    assert request["proposals"][0]["context"] == context
    assert len(provider.calls) == 3
    if review == "supported":
        candidate, = result.pending_candidates
        assert candidate.contexts == (context,)
        assert candidate.status == "pending" and candidate.evidence[0].text == text
    else:
        assert result.pending_candidates == () and result.signals == ()
        assert result.diagnostics.baseline_coverage_trace.unresolved_assertions == 1


@pytest.mark.parametrize(("text", "dimension", "trait_key", "key_object"), [
    ("林蔚的核心性格是沉着。", "core_personality", "composure", ""),
    ("林蔚长期信任顾昭。", "relationship_attitude", "trust_orientation", "顾昭"),
    ("林蔚长期坚持寻找失踪的姐姐。", "motivation_goal", "pursuit_commitment", "失踪的姐姐"),
    ("林蔚始终重视诚实。", "value", "integrity", "诚实"),
    ("林蔚始终拒绝泄露委托人住址。", "behavior_boundary", "client_confidentiality", "委托人住址"),
    ("林蔚在谈判时一贯先听完对方的意见。", "contextual_behavior", "listening_order", ""),
])
def test_supplementary_candidates_use_the_same_dimension_and_source_validation(
    text, dimension, trait_key, key_object,
):
    row = _record(text, dimension=dimension, trait_key=trait_key, key_object=key_object,
                  stability="core" if dimension == "core_personality" else "stable")
    result = _extract(text, RecoveryProvider([], _audit([row])))
    assert len(result.pending_candidates) == 1
    assert result.pending_candidates[0].dimension == dimension
    assert result.signals[0].evidence.text == text
    assert result.signals[0].source_run_input_id == "frozen-profile-input"


def test_nontrait_and_uncertain_assertions_are_not_turned_into_traits():
    first = "林蔚长期喜欢枇杷。"
    source = first + "\n这一页是草拟目录。\n林蔚是否依然沉默？"
    choices = [
        {"support_id": "L2:A1", "disposition": "no_additional_trait"},
        {"support_id": "L3:A1", "disposition": "uncertain"},
    ]
    result = _extract(source, RecoveryProvider([_record(first)], _audit(choices=choices)))
    assert len(result.pending_candidates) == 1
    trace = result.diagnostics.baseline_coverage_trace
    assert (trace.status, trace.nontrait_assertions, trace.unresolved_assertions) == ("partial", 1, 1)
    assert result.diagnostics.outcome == "partial"


@pytest.mark.parametrize("bad_audit", [
    "not json",
    _audit(choices=[]),
    _audit(choices=[{"support_id": "L2:A1", "disposition": "record"}]),
    _audit(choices=[{"support_id": "L999:A1", "disposition": "no_additional_trait"}]),
    RuntimeError("simulated transport failure"),
])
def test_failed_audit_preserves_original_candidate_and_marks_partial(bad_audit):
    first = "林蔚长期喜欢枇杷。"
    provider = RecoveryProvider([_record(first)], bad_audit)
    result = _extract(first + "\n林蔚一贯说话简洁直接。", provider)
    assert [row.dimension for row in result.pending_candidates] == ["preference"]
    assert result.diagnostics.outcome == "partial"
    assert result.diagnostics.baseline_coverage_trace.status == "partial"
    assert provider.audit_calls == 1


def test_bad_source_quote_in_supplement_cannot_bypass_original_validation():
    first = "林蔚长期喜欢枇杷。"
    second = "林蔚一贯说话简洁直接。"
    bad = _record(second, line=2, dimension="speech_pattern", trait_key="directness", key_object="")
    bad["evidence"] = "林蔚总爱含糊其辞。"
    result = _extract(first + "\n" + second, RecoveryProvider([_record(first)], _audit([bad])))
    assert [row.dimension for row in result.pending_candidates] == ["preference"]
    assert result.diagnostics.baseline_coverage_trace.unresolved_assertions == 1


def test_scope_review_rejecting_supplement_reopens_its_coverage():
    first = "林蔚长期喜欢枇杷。"
    second = "林蔚替顾昭转述她说话简洁直接。"
    wrong_actor = _record(second, line=2, dimension="speech_pattern", trait_key="directness", key_object="")
    provider = RecoveryProvider([_record(first)], _audit([wrong_actor]),
                                [{}, {"verdict": "rejected", "actor": "other"}])
    result = _extract(first + "\n" + second, provider)
    assert [row.dimension for row in result.pending_candidates] == ["preference"]
    assert result.diagnostics.baseline_coverage_trace.status == "partial"
    assert result.diagnostics.baseline_coverage_trace.recovered_support_slots == ()
    assert result.diagnostics.baseline_coverage_trace.unresolved_assertions == 1


def test_document_call_limit_keeps_unchecked_assertion_visible_without_call():
    first = "林蔚长期喜欢枇杷。"
    provider = RecoveryProvider([_record(first)], RuntimeError("must not call again"))
    result = _extract(first + "\n林蔚一贯说话简洁直接。", provider,
                      baseline_coverage_recovery_already_used=True)
    assert provider.audit_calls == 0
    assert result.diagnostics.outcome == "partial"
    assert result.diagnostics.reason_counts["baseline_coverage_document_call_limit"] == 1


def test_budget_and_deadline_admission_do_not_call_or_claim_a_completed_audit():
    text = "林蔚一贯说话简洁直接。"
    chunk = CharacterSignalChunk("profile", "profile.md", text, 1, "formal_character_profile")
    provider = RecoveryProvider([], RuntimeError("must not call"))
    for budget, deadline, reason in (
        (0, 30.0, "baseline_coverage_token_budget"),
        (200000, 0.0, "baseline_coverage_deadline"),
    ):
        outcome = _recover_baseline_coverage(
            (), index=_assertion_index_v1(chunk),
            system_prompt=build_formal_character_signal_system_prompt(scope_review_v1=True),
            user_prompt=text,
            settings=_settings(), provider=provider, token_budget=budget,
            remaining_deadline_seconds=deadline, already_used=False,
            validate_response=lambda _: pytest.fail("validation must not run"),
        )
        assert (outcome.reason, outcome.trace.status, outcome.trace.attempted_calls) == (reason, "partial", 0)
    assert provider.calls == []
