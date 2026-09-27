from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from app.character_history_semantic_review import (
    HISTORY_REVIEW_USER_PREFIX,
    HistoryReviewCandidate,
    HistoryReviewSourceIdentity,
    build_history_review_request,
    evaluate_history_review,
    history_review_request_digest,
    run_history_semantic_review,
    segment_history_line,
)
from app.provider import ProviderRetryExhausted


def _fixture(line: str = "林澈拒绝公开伤者姓名，后来证实这是林澈本人的决定。"):
    content = f"前情\n{line}\n后记\n"
    source = HistoryReviewSourceIdentity(
        run_input_id="run-input-1", document_id="doc-1", document_version=3,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        source_kind="published_history",
    )
    candidate = HistoryReviewCandidate(
        character="林澈", dimension="value", trait_key="privacy_protection",
        statement="林澈拒绝公开伤者姓名", polarity="positive", key_object="伤者姓名",
    )
    request = build_history_review_request(
        source=source, frozen_content=content, line_number=2,
        target_assertion_id="H2:A1", target_quote="拒绝公开伤者姓名",
        candidate=candidate,
    )
    return request, source, content


def _response(request, **changes):
    payload = {
        "schema_version": request.schema_version,
        "request_digest": history_review_request_digest(request),
        "target_assertion_id": request.target_assertion_id,
        "verdict": "supported",
        "actor": "proposed",
        "actuality": "asserted",
        "statement_relation": "supported",
        "axis_relation": "same",
        "object_relation": "same",
        "polarity_relation": "same",
        "whole_line_relation": "consistent",
        "basis_ids": [item.assertion_id for item in request.assertions],
    }
    payload.update(changes)
    return json.dumps(payload, ensure_ascii=False)


class MockProvider:
    def __init__(
        self, text: str = "", *, error=None, clock=None, advance=0,
        prompt_tokens=11, completion_tokens=7,
    ):
        self.text = text
        self.error = error
        self.clock = clock
        self.advance = advance
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.calls = []

    def complete(self, system: str, user: str):
        self.calls.append((system, user))
        if self.clock is not None:
            self.clock[0] += self.advance
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            text=self.text, prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )


def _run(request, source, content, provider, **overrides):
    settings = {
        "token_budget": 100_000,
        "completion_reserve": 256,
        "timeout_seconds": 5,
        "remaining_deadline_seconds": 20,
        "max_response_bytes": 8_192,
    }
    settings.update(overrides)
    return run_history_semantic_review(
        request, expected_source=source, frozen_content=content,
        provider=provider, **settings,
    )


def test_segmentation_covers_whole_line_and_keeps_post_target_correction():
    text = "林澈拒绝公开姓名，周砚却说：‘那只是传闻。’后来证实是周砚的决定。"
    assertions = segment_history_line(2, text)
    assert tuple(item.assertion_id for item in assertions) == (
        "H2:A1", "H2:A2", "H2:A3",
    )
    assert "".join(item.text for item in assertions) == text
    assert [(item.start_offset, item.end_offset) for item in assertions] == [
        (0, assertions[0].end_offset),
        (assertions[0].end_offset, assertions[1].end_offset),
        (assertions[1].end_offset, len(text)),
    ]


def test_positive_privacy_axis_can_pass_despite_surface_refusal():
    request, source, content = _fixture()
    provider = MockProvider(_response(request))
    run = _run(request, source, content, provider)
    assert run.evaluation.decision.verdict == "supported"
    assert run.failure_reason is None
    assert run.attempted_calls == 1
    assert run.charged_tokens == run.estimated_tokens
    assert "表面出现“拒绝”“不”等词" in provider.calls[0][0]
    body = json.loads(provider.calls[0][1].removeprefix(HISTORY_REVIEW_USER_PREFIX))
    assert body["request_digest"] == history_review_request_digest(request)
    assert body["request"]["line_text"] == request.line_text
    assert body["request"]["segmenter_version"] == request.segmenter_version


def test_supported_verdict_requires_all_post_target_assertions_in_order():
    request, source, content = _fixture()
    response = _response(request, basis_ids=[request.assertions[0].assertion_id])
    result = evaluate_history_review(
        request, response, expected_source=source, frozen_content=content
    )
    assert (result.decision.verdict, result.decision.reason) == (
        "uncertain", "basis_invalid"
    )


def test_later_correction_can_reject_but_cannot_support():
    request, source, content = _fixture(
        "林澈拒绝公开伤者姓名，后来查明拒绝的是周砚，林澈已同意公开。"
    )
    rejection = _response(
        request, verdict="rejected", actor="other", whole_line_relation="corrected"
    )
    result = evaluate_history_review(
        request, rejection, expected_source=source, frozen_content=content
    )
    assert result.decision.verdict == "rejected"
    unsupported = _response(request, whole_line_relation="corrected")
    result = evaluate_history_review(
        request, unsupported, expected_source=source, frozen_content=content
    )
    assert (result.decision.verdict, result.decision.reason) == (
        "uncertain", "slot_conflict"
    )


@pytest.mark.parametrize("line,slot,slot_value", [
    ("林澈记下周砚拒绝公开伤者姓名。", "actor", "other"),
    ("有人传闻林澈拒绝公开伤者姓名。", "actuality", "reported"),
    ("如果林澈拒绝公开伤者姓名，队伍就会等候。", "actuality", "hypothetical"),
])
def test_other_actor_report_and_hypothesis_are_rejection_signals(line, slot, slot_value):
    request, source, content = _fixture(line)
    response = _response(request, verdict="rejected", **{slot: slot_value})
    result = evaluate_history_review(
        request, response, expected_source=source, frozen_content=content
    )
    assert (result.decision.verdict, result.decision.reason) == (
        "rejected", "reviewer_rejected"
    )


@pytest.mark.parametrize("response_change,reason", [
    ({"request_digest": "0" * 64}, "response_mismatch"),
    ({"target_assertion_id": "H2:A99"}, "response_mismatch"),
    ({"extra": "instruction"}, "response_invalid"),
    ({"basis_ids": ["H2:A1", "H2:A1"]}, "basis_invalid"),
    ({"axis_relation": "ambiguous"}, "slot_conflict"),
])
def test_wrong_digest_id_path_or_slots_fail_closed(response_change, reason):
    request, source, content = _fixture()
    result = evaluate_history_review(
        request, _response(request, **response_change),
        expected_source=source, frozen_content=content,
    )
    assert (result.decision.verdict, result.decision.reason) == ("uncertain", reason)


@pytest.mark.parametrize("raw", [
    "```json\n{}\n```", "{broken", "{\"schema_version\":1,\"schema_version\":2}",
])
def test_malformed_or_duplicate_json_is_uncertain(raw):
    request, source, content = _fixture()
    result = evaluate_history_review(
        request, raw, expected_source=source, frozen_content=content
    )
    assert (result.decision.verdict, result.decision.reason) == (
        "uncertain", "response_invalid"
    )


def test_source_hash_and_line_mismatch_prevent_provider_call():
    request, source, content = _fixture()
    provider = MockProvider(_response(request))
    run = _run(request, source, content.replace("前情", "改写"), provider)
    assert run.evaluation.decision.reason == "source_mismatch"
    assert (run.attempted_calls, run.charged_tokens) == (0, 0)
    assert provider.calls == []
    with pytest.raises(ValueError):
        build_history_review_request(
            source=source, frozen_content=content.replace("前情", "改写"),
            line_number=2, target_assertion_id="H2:A1",
            target_quote="拒绝公开伤者姓名", candidate=request.candidate,
        )
    with pytest.raises(ValueError):
        build_history_review_request(
            source=source, frozen_content=content, line_number=2,
            target_assertion_id="H2:A2", target_quote="拒绝公开伤者姓名",
            candidate=request.candidate,
        )


def test_ambiguous_target_quote_is_rejected_before_provider():
    line = "林澈拒绝公开伤者姓名，周砚也拒绝公开伤者姓名。"
    content = line + "\n"
    source = HistoryReviewSourceIdentity(
        run_input_id="run", document_id="doc", document_version=1,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        source_kind="published_history",
    )
    candidate = _fixture()[0].candidate
    with pytest.raises(ValueError):
        build_history_review_request(
            source=source, frozen_content=content, line_number=1,
            target_assertion_id="H1:A1", target_quote="拒绝公开伤者姓名",
            candidate=candidate,
        )


def test_history_only_and_target_quote_must_bind_object_and_statement():
    request, source, content = _fixture()
    with pytest.raises(ValueError):
        HistoryReviewSourceIdentity(
            **{**source.model_dump(), "source_kind": "formal_character_profile"}
        )
    with pytest.raises(ValueError):
        build_history_review_request(
            source=source, frozen_content=content, line_number=2,
            target_assertion_id="H2:A1", target_quote="拒绝公开",
            candidate=request.candidate,
        )
    with pytest.raises(ValueError):
        build_history_review_request(
            source=source, frozen_content=content, line_number=2,
            target_assertion_id="H2:A2", target_quote="林澈本人的决定",
            candidate=request.candidate,
        )


def test_overlapping_duplicate_quote_is_ambiguous():
    line = "ababab"
    source = HistoryReviewSourceIdentity(
        run_input_id="run", document_id="doc", document_version=1,
        content_sha256=hashlib.sha256(line.encode()).hexdigest(),
        source_kind="published_history",
    )
    candidate = _fixture()[0].candidate.model_copy(update={
        "statement": "林澈abab", "key_object": "abab"
    })
    with pytest.raises(ValueError):
        build_history_review_request(
            source=source, frozen_content=line, line_number=1,
            target_assertion_id="H1:A1", target_quote="abab", candidate=candidate,
        )


def test_budget_and_deadline_gate_without_provider_call():
    request, source, content = _fixture()
    provider = MockProvider(_response(request))
    budget = _run(request, source, content, provider, token_budget=0)
    deadline = _run(request, source, content, provider, remaining_deadline_seconds=0)
    assert budget.failure_reason == "token_budget"
    assert deadline.failure_reason == "deadline"
    assert budget.evaluation.decision.verdict == deadline.evaluation.decision.verdict == "uncertain"
    assert provider.calls == []


def test_single_attempt_only_and_reported_token_overrun_abstains_with_real_charge():
    request, source, content = _fixture()
    provider = MockProvider(_response(request), prompt_tokens=60_000, completion_tokens=3_000)
    with pytest.raises(ValueError):
        _run(request, source, content, provider, max_attempts=2)
    assert provider.calls == []
    run = _run(request, source, content, provider, token_budget=10_000)
    assert run.estimated_tokens <= 10_000
    assert run.attempted_calls == len(provider.calls) == 1
    assert (run.prompt_tokens, run.completion_tokens, run.charged_tokens) == (
        60_000, 3_000, 63_000
    )
    assert (run.evaluation.decision.verdict, run.failure_reason) == (
        "uncertain", "token_budget"
    )


@pytest.mark.parametrize("error,failure", [
    (ProviderRetryExhausted("private", category="read_timeout"), "provider_timeout"),
    (ProviderRetryExhausted("private", category="rate_limit", http_status=429), "provider_rate_limit"),
    (RuntimeError("private"), "provider_error"),
])
def test_provider_failures_abstain_and_do_not_expose_error(error, failure):
    request, source, content = _fixture()
    run = _run(request, source, content, MockProvider(error=error))
    assert (run.failure_reason, run.evaluation.decision.verdict) == (failure, "uncertain")
    assert run.attempted_calls == 1
    assert run.charged_tokens == run.estimated_tokens
    assert "private" not in repr(run)


def test_elapsed_timeout_invalid_response_and_response_limit_abstain():
    request, source, content = _fixture()
    clock = [1.0]
    timeout = _run(
        request, source, content,
        MockProvider(_response(request), clock=clock, advance=6),
        monotonic=lambda: clock[0],
    )
    invalid = _run(request, source, content, MockProvider("{bad"))
    oversized = _run(
        request, source, content, MockProvider(_response(request)),
        max_response_bytes=16,
    )
    assert timeout.failure_reason == "provider_timeout"
    assert invalid.failure_reason == "response_invalid"
    assert oversized.failure_reason == "response_too_large"
    assert all(
        result.evaluation.decision.verdict == "uncertain"
        for result in (timeout, invalid, oversized)
    )
