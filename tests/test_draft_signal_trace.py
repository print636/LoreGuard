from __future__ import annotations

import json
from collections import Counter
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.character_consistency_stage import (
    _Usage,
    _diagnostics,
    _validated_draft_trace_payload,
)
from app.character_trait_extraction import (
    CharacterSignalChunk,
    CharacterSignalExtractor,
    CharacterSignalTarget,
    DraftSignalTraceAttemptV1,
    DraftSignalTraceEventV1,
    DraftSignalTraceV1,
    stable_trait_identity,
)
from app.config import Settings
from app.runtime_provenance import safe_runtime_provenance


PRIVATE_LINE = "林澈一直喜欢蜜瓜。"


class QueueProvider:
    def __init__(self, *responses: str | Exception) -> None:
        self.responses = list(responses)
        self.calls = 0

    def complete(self, _system: str, _user: str) -> SimpleNamespace:
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return SimpleNamespace(text=response, prompt_tokens=17, completion_tokens=9)


def _settings(**overrides: object) -> Settings:
    values = {
        "openai_api_key": "private-api-key",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_character_consistency": True,
        "provider_max_attempts": 1,
        "character_signal_draft_trace_v1": True,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def _draft() -> CharacterSignalChunk:
    return CharacterSignalChunk(
        "private-document-id", "private-draft.md", PRIVATE_LINE, 10, "draft"
    )


def _record() -> dict[str, object]:
    return {
        "character": "林澈",
        "dimension": "preference",
        "trait_key": "食物偏好:蜜瓜",
        "statement": "林澈一直喜欢蜜瓜",
        "polarity": "positive",
        "stability": "stable",
        "observation_kind": "explicit_declaration",
        "context": "日常饮食",
        "key_object": "蜜瓜",
        "source_line_start": 10,
        "source_line_end": 10,
        "evidence": PRIVATE_LINE,
    }


def _response(*records: dict[str, object]) -> str:
    return json.dumps({"records": records}, ensure_ascii=False)


def test_draft_trace_records_rejected_then_clean_retry_without_private_text():
    valid = _record()
    invalid = {**valid, "requested_polarity": "positive"}
    provider = QueueProvider(_response(valid, invalid), _response(valid))

    result = CharacterSignalExtractor(provider, settings=_settings()).extract(_draft())

    assert provider.calls == 2
    assert result.diagnostics.outcome == "completed"
    trace = result.diagnostics.draft_trace
    assert trace is not None
    assert trace.final_state == "clean"
    assert trace.final_accepted_record_ordinals == (1,)
    assert [(item.outcome, item.reason) for item in trace.attempts[0].events] == [
        ("accepted", None), ("rejected", "forbidden_server_field"),
    ]
    assert [
        (item.claimed_line_start_ordinal, item.claimed_line_end_ordinal)
        for item in trace.attempts[0].events
    ] == [
        (1, 1), (1, 1),
    ]
    assert [item.attempt for item in trace.attempts] == [1, 2]
    assert trace.attempts[1].submitted_record_count == 1
    assert trace.source_candidate_linkage == "unavailable"
    serialized = json.dumps(trace.model_dump(mode="json"), ensure_ascii=False)
    assert all(value not in serialized for value in (
        PRIVATE_LINE, "林澈", "蜜瓜", "private-document-id", "private-draft.md",
        "private-api-key", "requested_polarity",
    ))
    assert "draft_trace" not in result.diagnostics.model_dump()


def test_draft_trace_distinguishes_empty_target_from_rejected_proposal():
    target = CharacterSignalTarget(
        character="林澈",
        dimension="preference",
        trait_key="食物偏好:蜜瓜",
        comparison_key=stable_trait_identity("preference", "", "蜜瓜"),
        baseline_polarity="positive",
        requested_polarity="negative",
        baseline_hint="喜欢蜜瓜",
    )
    empty = CharacterSignalExtractor(
        QueueProvider(_response()), settings=_settings()
    ).extract_targeted(_draft(), (target,))
    rejected = CharacterSignalExtractor(
        QueueProvider(_response(_record())),
        settings=_settings(character_signal_package_max_attempts=1),
    ).extract_targeted(_draft(), (target,))

    assert empty.diagnostics.draft_trace is not None
    assert empty.diagnostics.draft_trace.final_state == "clean"
    assert empty.diagnostics.draft_trace.attempts[0].submitted_record_count == 0
    assert empty.diagnostics.draft_trace.attempts[0].events == ()
    assert rejected.diagnostics.draft_trace is not None
    assert rejected.diagnostics.draft_trace.final_state == "no_clean_package"
    assert rejected.diagnostics.draft_trace.attempts[0].submitted_record_count == 1
    assert rejected.diagnostics.draft_trace.attempts[0].events[0].outcome == "rejected"
    assert rejected.diagnostics.draft_trace.attempts[0].events[0].reason == (
        "targeted_polarity_mismatch"
    )
    assert rejected.diagnostics.draft_trace.final_accepted_record_ordinals == ()


def test_draft_trace_malformed_response_cannot_claim_zero_submissions_or_echo_raw():
    raw = '{"records": ["private secret 林澈'
    result = CharacterSignalExtractor(
        QueueProvider(raw),
        settings=_settings(character_signal_package_max_attempts=1),
    ).extract(_draft())

    trace = result.diagnostics.draft_trace
    assert trace is not None
    assert trace.final_state == "no_clean_package"
    assert trace.attempts[0].observability == "unparseable"
    assert trace.attempts[0].package_reason == "invalid_json"
    assert trace.attempts[0].submitted_record_count is None
    assert trace.attempts[0].events == ()
    assert raw not in json.dumps(trace.model_dump(mode="json"), ensure_ascii=False)


def test_draft_trace_invalid_model_coordinates_are_not_exported():
    invalid = {
        **_record(), "source_line_start": 999_999, "source_line_end": 999_999,
    }
    result = CharacterSignalExtractor(
        QueueProvider(_response(invalid)),
        settings=_settings(character_signal_package_max_attempts=1),
    ).extract(_draft())

    trace = result.diagnostics.draft_trace
    assert trace is not None
    event = trace.attempts[0].events[0]
    assert event.outcome == "rejected"
    assert event.claimed_line_start_ordinal is None
    assert event.claimed_line_end_ordinal is None
    assert "999999" not in json.dumps(trace.model_dump(mode="json"))


def test_draft_trace_preserves_cross_line_claim_as_bounded_pair():
    second_line = "林澈拿起蜜瓜。"
    chunk = CharacterSignalChunk(
        "private-document-id", "private-draft.md",
        f"{PRIVATE_LINE}\n{second_line}", 10, "draft",
    )
    cross_line = {
        **_record(), "source_line_end": 11,
        "evidence": f"{PRIVATE_LINE}\n{second_line}",
    }
    result = CharacterSignalExtractor(
        QueueProvider(_response(cross_line)),
        settings=_settings(character_signal_package_max_attempts=1),
    ).extract(chunk)

    trace = result.diagnostics.draft_trace
    assert trace is not None
    event = trace.attempts[0].events[0]
    assert (event.claimed_line_start_ordinal, event.claimed_line_end_ordinal) == (
        1, 2,
    )
    assert "source_line_start" not in json.dumps(trace.model_dump(mode="json"))


def test_draft_trace_retry_without_response_keeps_attempts_separate():
    rejected = {**_record(), "requested_polarity": "positive"}
    provider = QueueProvider(_response(rejected), RuntimeError("private failure"))
    result = CharacterSignalExtractor(provider, settings=_settings()).extract(_draft())

    trace = result.diagnostics.draft_trace
    assert trace is not None
    assert result.diagnostics.attempted_calls == provider.calls == len(trace.attempts) == 2
    assert trace.final_state == "no_clean_package"
    assert trace.attempts[0].observability == "parsed"
    assert trace.attempts[0].events[0].reason == "forbidden_server_field"
    assert trace.attempts[1].observability == "no_response"
    assert trace.attempts[1].submitted_record_count is None
    assert trace.attempts[1].events == ()
    assert "private failure" not in json.dumps(trace.model_dump(mode="json"))


def test_draft_trace_projection_failure_after_response_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        "app.character_trait_extraction._try_draft_trace_attempt",
        lambda *_args, **_kwargs: None,
    )
    rejected = {**_record(), "requested_polarity": "positive"}
    result = CharacterSignalExtractor(
        QueueProvider(_response(rejected)),
        settings=_settings(character_signal_package_max_attempts=1),
    ).extract(_draft())

    trace = result.diagnostics.draft_trace
    assert trace is not None
    assert trace.attempts[0].observability == "unavailable"
    assert trace.attempts[0].submitted_record_count is None
    assert trace.attempts[0].events == ()
    assert result.diagnostics.rejected_records == 1


def test_draft_trace_clean_projection_failure_does_not_claim_empty_acceptance(
    monkeypatch,
):
    monkeypatch.setattr(
        "app.character_trait_extraction._try_draft_trace_attempt",
        lambda *_args, **_kwargs: None,
    )
    result = CharacterSignalExtractor(
        QueueProvider(_response(_record())), settings=_settings()
    ).extract(_draft())

    assert result.diagnostics.outcome == "completed"
    assert result.diagnostics.accepted_records == 1
    assert result.diagnostics.draft_trace is None


def test_draft_trace_record_limit_marks_each_parsed_proposal_rejected():
    result = CharacterSignalExtractor(
        QueueProvider(_response(_record(), _record())),
        settings=_settings(
            character_signal_max_records=1,
            character_signal_package_max_attempts=1,
        ),
    ).extract(_draft())

    trace = result.diagnostics.draft_trace
    assert trace is not None
    assert trace.attempts[0].observability == "parsed"
    assert trace.attempts[0].submitted_record_count == 2
    assert [(event.record_ordinal, event.outcome, event.reason)
            for event in trace.attempts[0].events] == [
        (1, "rejected", "record_limit"), (2, "rejected", "record_limit"),
    ]


def test_draft_trace_is_default_off_and_schema_is_bounded():
    disabled = CharacterSignalExtractor(
        QueueProvider(_response()),
        settings=_settings(character_signal_draft_trace_v1=False),
    ).extract(_draft())
    assert disabled.diagnostics.draft_trace is None
    assert "draft_trace" not in disabled.diagnostics.model_dump()

    no_call_provider = QueueProvider()
    no_call = CharacterSignalExtractor(
        no_call_provider, settings=_settings()
    ).extract_targeted(_draft(), ())
    assert no_call_provider.calls == 0
    assert no_call.diagnostics.draft_trace is not None
    assert no_call.diagnostics.draft_trace.final_state == "no_call"
    assert no_call.diagnostics.draft_trace.attempts == ()

    with pytest.raises(ValidationError):
        DraftSignalTraceAttemptV1(
            attempt=1, observability="parsed", submitted_record_count=65,
            events=(),
        )
    with pytest.raises(ValidationError):
        DraftSignalTraceEventV1(
            record_ordinal=1, outcome="rejected", reason="private-response"
        )
    with pytest.raises(ValidationError):
        DraftSignalTraceEventV1(
            record_ordinal=1, outcome="accepted", claimed_line_start_ordinal=1,
        )
    with pytest.raises(ValidationError):
        DraftSignalTraceV1(
            attempts=(), final_state="no_call", final_accepted_record_ordinals=(),
            story_text="private story",
        )


def test_stage_draft_trace_export_has_opt_in_fixed_schema():
    trace = DraftSignalTraceV1(
        attempts=(DraftSignalTraceAttemptV1(
            attempt=1, observability="parsed", submitted_record_count=0,
            events=(),
        ),),
        final_state="clean", final_accepted_record_ordinals=(),
    )
    assert _validated_draft_trace_payload(trace) == trace.model_dump(mode="json")
    assert _validated_draft_trace_payload({"raw_response": PRIVATE_LINE}) is None
    base = dict(
        outcome="completed", reason_code="ok", usage=_Usage(),
        reasons=Counter(),
    )
    assert "draft_trace_chunks" not in _diagnostics(**base)
    enabled = _diagnostics(
        **base, draft_trace_enabled=True,
        draft_trace_chunks=[{"trace": _validated_draft_trace_payload(trace)}],
        draft_trace_chunks_omitted_count=2,
    )
    assert enabled["draft_trace_chunks"][0]["trace"]["schema_version"] == (
        "draft-signal-trace-v1"
    )
    assert enabled["draft_trace_chunks_omitted_count"] == 2


def test_draft_trace_runtime_provenance_records_flag_and_version():
    off = safe_runtime_provenance(
        _settings(character_signal_draft_trace_v1=False)
    )["character_consistency_limits"]
    on = safe_runtime_provenance(_settings())["character_consistency_limits"]
    assert (off["signal_draft_trace_v1"], off["signal_draft_trace_version"]) == (
        False, None,
    )
    assert (on["signal_draft_trace_v1"], on["signal_draft_trace_version"]) == (
        True, "draft-signal-trace-v1",
    )


def test_runtime_copies_reject_non_bool_draft_trace_flag():
    invalid = _settings().model_copy(
        update={"character_signal_draft_trace_v1": "false"}
    )
    with pytest.raises(RuntimeError, match="prompt variant flags must be bool"):
        CharacterSignalExtractor(QueueProvider(), settings=invalid).extract(_draft())
    with pytest.raises(RuntimeError, match="prompt variant flags must be bool"):
        CharacterSignalExtractor(QueueProvider(), settings=invalid).extract_targeted(
            _draft(), ()
        )
