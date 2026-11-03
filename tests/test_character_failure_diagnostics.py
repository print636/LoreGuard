from __future__ import annotations

import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import app.character_trait_extraction as extraction
from app.character_consistency_stage import (
    _Usage, _diagnostics, _validated_failure_diagnostics_payload,
)
from app.character_draft_actor_review import (
    TargetBoundDraftReviewBatchEvaluation,
    TargetBoundDraftReviewTarget,
    build_target_bound_draft_proposals,
    build_target_bound_draft_review_request,
    evaluate_target_bound_draft_review,
    required_target_bound_draft_basis_ids,
    target_bound_draft_review_batch_digest,
    target_bound_draft_review_request_digest,
)
from app.character_draft_actor_review_provider import TargetBoundDraftReviewRun
from app.character_scope_review import (
    ScopeReviewDecision, ScopeReviewEvaluation, ScopeReviewSourceIdentity,
)
from app.character_scope_review_provider import SCOPE_REVIEW_USER_PREFIX
from app.config import Settings


CONTENT = "沈砚明确表示自己讨厌栗子糕。"


def _settings(**updates):
    return Settings(_env_file=None, **{
        "openai_api_key": "unit-test-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock",
        "enable_character_consistency": True,
        "character_signal_package_max_attempts": 1,
        "provider_max_attempts": 1,
        **updates,
    })


def _raw(**updates):
    return extraction._RawCharacterSignal(**{
        "character": "沈砚", "dimension": "speech_pattern",
        "trait_key": "speech_style", "statement": "沈砚语速很慢",
        "polarity": "neutral", "stability": "temporary",
        "observation_kind": "speech_sample", "key_object": "",
        "source_line_start": 1, "source_line_end": 1,
        "evidence": "沈砚说话很快。", **updates,
    })


class _Provider:
    def __init__(self, *responses):
        self.responses = list(responses)

    def complete(self, _system, _user):
        return SimpleNamespace(
            text=self.responses.pop(0), prompt_tokens=3, completion_tokens=2,
        )


@pytest.mark.parametrize("statement,evidence,polarity,kind", [
    ("沈砚喜欢栗子糕", "沈砚讨厌栗子糕。", "positive", "lexical_polarity"),
    ("沈砚一直", "沈砚说话很快。", "neutral", "anchor_empty"),
    ("沈砚语速很慢", "沈砚说话很快。", "neutral", "anchor_overlap"),
    ("沈砚说话很快", "沈砚说话很快。", "neutral", None),
])
def test_statement_subtype_identifies_the_existing_guard(statement, evidence, polarity, kind):
    record = _raw(statement=statement, evidence=evidence, polarity=polarity)
    assert extraction._statement_support_failure(record, evidence) == kind
    assert extraction._statement_supported(record, evidence) is (kind is None)
    if kind is not None:
        with pytest.raises(extraction._StatementSupportFailure) as caught:
            extraction._require_statement_supported(record, evidence)
        assert caught.value.kind == kind
        assert str(caught.value) == "statement_support"


def test_failed_and_regenerated_packages_conserve_statement_failure_counts():
    record = _raw()
    bad = json.dumps({"records": [record.model_dump()]}, ensure_ascii=False)
    chunk = extraction.CharacterSignalChunk(
        "private-document", "private.md", record.evidence, 1, "published_history"
    )
    failed = extraction.CharacterSignalExtractor(_Provider(bad), settings=_settings()).extract(chunk)
    assert failed.diagnostics.reason_counts == {"statement_support": 1}
    assert failed.diagnostics.failure_diagnostics.statement_support_counts == {"anchor_overlap": 1}
    assert failed.diagnostics.rejected_records == 1
    assert not failed.signals

    regenerated = extraction.CharacterSignalExtractor(
        _Provider(bad, '{"records":[]}'),
        settings=_settings(character_signal_package_max_attempts=2),
    ).extract(chunk)
    assert regenerated.diagnostics.outcome == "completed"
    assert regenerated.diagnostics.reason_counts == {"regenerated_from_statement_support": 1}
    assert regenerated.diagnostics.failure_diagnostics.statement_support_counts == {"anchor_overlap": 1}
    safe = _validated_failure_diagnostics_payload(regenerated.diagnostics.failure_diagnostics)
    serialized = json.dumps(safe, ensure_ascii=False)
    assert all(value not in serialized for value in (
        record.statement, record.evidence, "沈砚", "private-document", "private.md",
        "unit-test-placeholder", "https://mock.invalid/v1",
    ))
    assert "failure_diagnostics" not in regenerated.diagnostics.model_dump()


@pytest.mark.parametrize("updates,kind", [
    ({"statement": "沈砚很喜欢甜食"}, "assertion_rewrite"),
    ({"polarity": "negative"}, "claim_direction"),
])
def test_formal_statement_subtypes_name_the_actual_failing_guard(updates, kind):
    line = "沈砚喜欢甜食。"
    record = _raw(
        dimension="preference", trait_key="food_preference", key_object="甜食",
        statement="沈砚喜欢甜食", polarity="positive", evidence=line,
        observation_kind="explicit_declaration", stability="stable",
    ).model_dump()
    record.update(support_id="L1:A1", **updates)
    result = extraction.CharacterSignalExtractor(
        _Provider(json.dumps({"records": [record]}, ensure_ascii=False)),
        settings=_settings(character_signal_full_line_prompt_v2=True, character_signal_support_id_v4=True),
    ).extract(extraction.CharacterSignalChunk("document", "profile.md", line, 1, "formal_character_profile"))
    assert not result.signals
    assert result.diagnostics.reason_counts == {"statement_support": 1}
    assert result.diagnostics.failure_diagnostics.statement_support_counts == {kind: 1}


def _source(content=CONTENT):
    return ScopeReviewSourceIdentity(
        run_input_id="private-input", document_id="private-document", document_version=1,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def _target():
    return TargetBoundDraftReviewTarget(
        target_ordinal=1, character="沈砚", dimension="preference",
        trait_key="food_preference", comparison_key="preference:甜食",
        key_object="甜食", requested_polarity="negative", baseline_hint="沈砚长期喜欢甜食",
    )


def _semantic_evaluation(request, **updates):
    items = [{
        "proposal_id": proposal.proposal_id, "verdict": "rejected",
        "actor": "proposed", "actuality": "asserted", "statement_relation": "supported",
        "axis_relation": "matches_target", "object_relation": "matches_target",
        "polarity_relation": "requested", "correction_relation": "none",
        "observation_kind": "preference_expression",
        "basis_ids": list(required_target_bound_draft_basis_ids(request)),
        **updates,
    } for proposal in request.proposals]
    response = json.dumps({
        "schema_version": request.schema_version,
        "request_digest": target_bound_draft_review_request_digest(request), "items": items,
    })
    return evaluate_target_bound_draft_review(
        request, response, expected_source=_source(), frozen_content=CONTENT,
    )


@pytest.mark.parametrize("updates,slots,verdict", [
    ({"polarity_relation": "neutral"}, ("polarity_neutral",), "rejected"),
    ({"actor": "other"}, ("actor_other",), "rejected"),
    ({"actuality": "hypothetical", "polarity_relation": "opposite"},
     ("actuality_hypothetical", "polarity_opposite"), "rejected"),
    ({}, (), "uncertain"),
])
def test_negative_slots_come_from_validated_semantic_items(updates, slots, verdict):
    source, target = _source(), _target()
    index, proposals = build_target_bound_draft_proposals(
        CONTENT, source, target, line_start=1, line_end=1, protocol_version="v4",
    )
    request = build_target_bound_draft_review_request(
        index, target, proposals, frozen_content=CONTENT, expected_source=source,
        protocol_version="v4",
    )
    evaluation = _semantic_evaluation(request, **updates)
    assert evaluation.decisions
    for proposal, decision in zip(request.proposals, evaluation.decisions):
        assert decision.raw_verdict == "rejected"
        assert decision.verdict == verdict
        assert decision.negative_slots == slots
        event = extraction._target_bound_failure_event(request, proposal, decision)
        assert event.requested_polarity == "negative"
        assert event.negative_slots == slots
        assert event.source_sha256 == source.content_sha256
        assert "negative_slots" not in decision.model_dump()
        assert "raw_verdict" not in decision.model_dump()

    malformed = evaluate_target_bound_draft_review(
        request, "private-response", expected_source=source, frozen_content=CONTENT,
    )
    assert all(decision.raw_verdict is None and not decision.negative_slots
               for decision in malformed.decisions)


def test_targeted_semantic_rejection_reaches_bounded_safe_diagnostics(monkeypatch):
    def semantic_review(entries, **_kwargs):
        requests = tuple(entry.request for entry in entries)
        return TargetBoundDraftReviewRun(
            evaluation=TargetBoundDraftReviewBatchEvaluation(
                batch_digest=target_bound_draft_review_batch_digest(requests),
                evaluations=tuple(_semantic_evaluation(request, polarity_relation="neutral")
                                  for request in requests),
            ), estimated_tokens=10, attempted_calls=1, prompt_tokens=4,
            completion_tokens=2, charged_tokens=10,
        )

    monkeypatch.setattr(extraction, "run_target_bound_draft_review", semantic_review)
    target = extraction.CharacterSignalTarget(
        **{key: value for key, value in _target().model_dump().items()
           if key not in {"target_ordinal"}}, baseline_polarity="positive",
    )
    result = extraction.CharacterSignalExtractor(
        _Provider('{"records":[]}'), settings=_settings(
            character_signal_full_line_prompt_v2=True,
            character_draft_actor_review_v1=True, character_target_bound_draft_review_v3=True,
        ),
    ).extract_targeted(
        extraction.CharacterSignalChunk("private-document", "private.md", CONTENT, 1, "draft"),
        (target,), target_ordinal=1, source_identity=_source(), frozen_content=CONTENT,
    )
    failure = result.diagnostics.failure_diagnostics
    assert failure and failure.target_bound_events
    assert not result.signals
    assert all(event.negative_slots == ("polarity_neutral",) for event in failure.target_bound_events)
    assert len(failure.target_bound_events) == result.diagnostics.reason_counts["target_bound_review_rejected"]
    assert result.diagnostics.outcome == "completed"
    serialized = json.dumps(_validated_failure_diagnostics_payload(failure), ensure_ascii=False)
    assert all(value not in serialized for value in (CONTENT, "沈砚", "甜食", "private-document"))


def test_failure_projection_revalidates_closed_fields_and_preserves_legacy_shape():
    base = dict(outcome="completed", reason_code="ok", usage=_Usage(), reasons=Counter())
    assert "failure_diagnostics_v1" not in _diagnostics(**base)
    safe = extraction.CharacterSignalFailureDiagnosticsV1(statement_support_counts={"anchor_overlap": 1})
    assert _validated_failure_diagnostics_payload(safe)["statement_support_counts"] == {"anchor_overlap": 1}
    assert _validated_failure_diagnostics_payload({"raw_response": CONTENT}) is None
    assert _validated_failure_diagnostics_payload(safe.model_copy(update={
        "statement_support_counts": {CONTENT: 1},
    })) is None
    with pytest.raises(ValidationError):
        extraction.CharacterSignalFailureDiagnosticsV1(statement_support_counts={"anchor_overlap": 129})
    with pytest.raises(ValidationError):
        extraction.CharacterSignalFailureDiagnosticsV1(statement_support_counts={"anchor_overlap": True})
    with pytest.raises(ValidationError):
        extraction.CharacterSignalFailureDiagnosticsV1(raw_response=CONTENT)
    exported = _diagnostics(**base, failure_diagnostic_chunks=[{
        "diagnostics": _validated_failure_diagnostics_payload(safe),
    }], failure_diagnostic_chunks_omitted_count=2)
    assert exported["failure_diagnostics_v1"]["chunks_omitted_count"] == 2


def test_failure_event_limit_accounts_for_omitted_events():
    event = extraction.TargetBoundFailureEventV1(
        source_sha256="a" * 64, target_digest="b" * 64,
        source_line_start=1, source_line_end=1, requested_polarity="negative",
        raw_verdict="rejected", verdict="rejected", reason="reviewer_rejected",
        negative_slots=("polarity_neutral",),
    )
    diagnostics = extraction._failure_diagnostics_for_result([], [event] * 66)
    assert len(diagnostics.target_bound_events) == 64
    assert diagnostics.target_bound_events_omitted == 2
    recovery = extraction._ScopeReviewOutcome(
        supported=(), decisions=(), reasons={}, slot_recovery_attempts=1,
        slot_recovery_succeeded=0, slot_recovery_failed=1,
    )
    failed = extraction._failure_diagnostics_for_result([], scope_review=recovery)
    assert failed.scope_review_slot_recovery_attempts == 1
    assert failed.scope_review_slot_recovery_succeeded == 0
    assert failed.scope_review_slot_recovery_failed == 1


@pytest.mark.parametrize("retry_verdict", ["supported", "rejected", "uncertain"])
def test_scope_slot_history_and_actual_recovery_outcome_reach_extractor(retry_verdict):
    line = "沈砚喜欢甜食。"
    source = _source(line)
    record = _raw(
        dimension="preference", trait_key="food_preference", key_object="甜食",
        statement="沈砚喜欢甜食", polarity="positive", evidence=line,
        observation_kind="explicit_declaration", stability="stable",
    ).model_dump()
    record.update(support_id="L1:A1", actor_anchor_id="", label_anchor_id="", scope_relation="local")

    class ScopeProvider:
        reviews = 0

        def complete(self, _system, user):
            if not user.startswith(SCOPE_REVIEW_USER_PREFIX):
                payload = {"records": [record]}
            else:
                self.reviews += 1
                envelope = json.loads(user.removeprefix(SCOPE_REVIEW_USER_PREFIX))
                request = envelope["request"]
                items = [{
                    "proposal_id": proposal["proposal_id"], "support_id": proposal["support_id"],
                    "verdict": "supported" if self.reviews == 1 else retry_verdict,
                    "actor": "proposed", "actuality": "asserted",
                    "statement_relation": "supported", "label_relation": "none",
                    "object_relation": "same", "polarity_relation": (
                        "opposite" if self.reviews == 1 or retry_verdict == "rejected" else "same"
                    ), "level_supported": "yes", "basis_ids": [proposal["support_id"]],
                } for proposal in request["proposals"]]
                payload = {"schema_version": request["schema_version"],
                           "request_digest": envelope["request_digest"], "items": items}
            return SimpleNamespace(text=json.dumps(payload), prompt_tokens=3, completion_tokens=2)

    provider = ScopeProvider()
    result = extraction.CharacterSignalExtractor(provider, settings=_settings(
        character_signal_full_line_prompt_v2=True, character_signal_support_id_v4=True,
        character_signal_semantic_scope_v5=True, character_signal_scope_review_v1=True,
    )).extract(
        extraction.CharacterSignalChunk(source.document_id, "profile.md", line, 1, "formal_character_profile"),
        source_identity=source, frozen_content=line,
    )
    assert provider.reviews == 2
    diagnostics = result.diagnostics
    assert diagnostics.scope_review_slot_conflict_counts == {"polarity_relation": 1}
    recovery = diagnostics.failure_diagnostics
    assert recovery.scope_review_slot_recovery_attempts == 1
    assert recovery.scope_review_slot_recovery_succeeded == int(retry_verdict != "uncertain")
    assert recovery.scope_review_slot_recovery_failed == int(retry_verdict == "uncertain")
    assert bool(result.signals) is (retry_verdict == "supported")
    assert diagnostics.outcome == ("completed" if retry_verdict == "supported" else "partial")
    if retry_verdict == "rejected":
        assert diagnostics.reason_counts == {"scope_review_reviewer_rejected": 1}


def test_failed_slot_retry_does_not_relabel_an_unrelated_valid_rejection(monkeypatch):
    source = _source()
    chunk = extraction.CharacterSignalChunk(
        source.document_id, "profile.md", CONTENT, 1, "formal_character_profile",
    )
    initial = ScopeReviewEvaluation(request_digest="a" * 64, decisions=(
        ScopeReviewDecision(
            proposal_id="p1", support_id="L1:A1", verdict="rejected", reason="reviewer_rejected",
        ),
        ScopeReviewDecision(
            proposal_id="p2", support_id="L1:A2", verdict="uncertain", reason="slot_conflict",
            slot_conflicts=("polarity_relation",),
        ),
    ))
    run = SimpleNamespace(
        evaluation=initial, initial_evaluation=initial, failure_reason="token_budget",
        attempted_calls=1, prompt_tokens=3, completion_tokens=2, charged_tokens=10,
        slot_recovery_attempts=0, slot_recovery_succeeded=0, slot_recovery_failed=1,
    )
    monkeypatch.setattr(extraction, "_scope_review_request", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(extraction, "run_scope_review", lambda *_args, **_kwargs: run)
    outcome = extraction._review_clean_signals(
        (SimpleNamespace(support_id="L1:A1"), SimpleNamespace(support_id="L1:A2")),
        chunk=chunk, index=extraction._assertion_index_v1(chunk), source_identity=source,
        frozen_content=CONTENT, provider=_Provider(), settings=_settings(), token_budget=100,
        remaining_deadline_seconds=10, monotonic=lambda: 0,
    )
    assert outcome.reasons == {"scope_review_reviewer_rejected": 1, "scope_review_token_budget": 1}
    assert outcome.initial_decisions == initial.decisions
    assert outcome.supported == ()
