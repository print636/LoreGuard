from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
import app.character_trait_extraction as extraction_module

from app.character_history_semantic_review import (
    HISTORY_REVIEW_SYSTEM_PROMPT,
    HISTORY_REVIEW_USER_PREFIX,
    HistoryReviewSourceIdentity,
)
from app.character_trait_extraction import (
    CharacterSignalChunk,
    CharacterSignalExtractor,
    _RECORD_ADAPTER,
    _history_target_anchor,
    _history_obviously_unresolved,
    _history_lexical_only_deferred,
)
from app.config import Settings
from app.provider import OpenAICompatibleProvider, ProviderRetryExhausted, RetryPolicy
from app.service import (
    CharacterConsistencyUsageAccumulator,
    _CharacterConsistencyAccountingProvider,
)
from app.usage import estimate_issue_evidence_review_tokens


def _settings(**changes) -> Settings:
    fields = {
        "openai_api_key": "unit-test-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_character_consistency": True,
        "character_history_semantic_review_v1": True,
        "character_signal_package_max_attempts": 1,
        "provider_max_attempts": 1,
    }
    fields.update(changes)
    return Settings(_env_file=None, **fields)


def _record(
    line: str,
    *,
    character: str,
    statement: str,
    key_object: str,
    line_number: int = 1,
) -> dict[str, object]:
    return {
        "character": character,
        "dimension": "value",
        "trait_key": "privacy_consent_respect",
        "statement": statement,
        "polarity": "positive",
        "stability": "stable",
        "observation_kind": "action",
        "context": "",
        "key_object": key_object,
        "source_line_start": line_number,
        "source_line_end": line_number,
        "evidence": line,
    }


class QueueProvider:
    def __init__(self, records: list[dict[str, object]], *, review_mode: str = "supported"):
        self.primary_text = json.dumps({"records": records}, ensure_ascii=False)
        self.review_mode = review_mode
        self.calls: list[str] = []
        self.review_requests: list[dict] = []

    def complete(self, system: str, user: str):
        self.calls.append(system)
        if system != HISTORY_REVIEW_SYSTEM_PROMPT:
            return SimpleNamespace(
                text=self.primary_text, prompt_tokens=100, completion_tokens=100
            )
        if self.review_mode == "timeout":
            raise ProviderRetryExhausted("timeout", category="read_timeout")
        if self.review_mode == "429":
            raise ProviderRetryExhausted(
                "rate limited", category="rate_limit", http_status=429
            )
        if self.review_mode == "invalid_json":
            return SimpleNamespace(text="not json", prompt_tokens=20, completion_tokens=10)
        body = json.loads(user.removeprefix(HISTORY_REVIEW_USER_PREFIX))
        self.review_requests.append(body["request"])
        request = body["request"]
        response = {
            "schema_version": request["schema_version"],
            "request_digest": body["request_digest"],
            "target_assertion_id": request["target_assertion_id"],
            "verdict": self.review_mode,
            "actor": "proposed",
            "actuality": "asserted",
            "statement_relation": "supported",
            "axis_relation": "same",
            "object_relation": "same",
            "polarity_relation": "same",
            "whole_line_relation": "consistent",
            "basis_ids": [row["assertion_id"] for row in request["assertions"]],
        }
        if self.review_mode == "rejected":
            response["actor"] = "other"
        return SimpleNamespace(
            text=json.dumps(response, ensure_ascii=False),
            prompt_tokens=20,
            completion_tokens=10,
        )


class RegenerationProvider(QueueProvider):
    def __init__(
        self, records, *, review_mode, second_records=None, review_modes=None
    ):
        super().__init__(records, review_mode=review_mode)
        self.primary_calls = 0
        self.second_records = second_records
        self.review_modes = list(review_modes or [])

    def complete(self, system: str, user: str):
        if system == HISTORY_REVIEW_SYSTEM_PROMPT and self.review_modes:
            self.review_mode = self.review_modes.pop(0)
        if system != HISTORY_REVIEW_SYSTEM_PROMPT:
            self.primary_calls += 1
            if self.primary_calls == 2:
                self.calls.append(system)
                return SimpleNamespace(
                    text=json.dumps(
                        {"records": self.second_records or []}, ensure_ascii=False
                    ),
                    prompt_tokens=100,
                    completion_tokens=20,
                )
        return super().complete(system, user)


def _extract(
    lines: list[str],
    records: list[dict[str, object]],
    *,
    review_mode: str = "supported",
    settings: Settings | None = None,
    frozen_content: str | None = None,
    source_hash: str | None = None,
    source_kind: str = "published_history",
):
    content = "\n".join(lines) + "\n"
    source = HistoryReviewSourceIdentity(
        run_input_id="input-1",
        document_id="doc-1",
        document_version=3,
        content_sha256=source_hash or hashlib.sha256(content.encode()).hexdigest(),
        source_kind="published_history",
    )
    provider = QueueProvider(records, review_mode=review_mode)
    extractor = CharacterSignalExtractor(provider, settings=settings or _settings())
    result = extractor.extract(
        CharacterSignalChunk(
            document_id="doc-1",
            document_name="history.md",
            content=content,
            global_line_start=1,
            source_kind=source_kind,
        ),
        source_identity=source,
        frozen_content=frozen_content if frozen_content is not None else content,
    )
    return result, provider


def test_real_h03_h04_history_lines_can_be_semantically_reviewed():
    source = (
        Path(__file__).resolve().parents[1]
        / "data/character-ooc-return-season-dev-v1/03-published-main-v1.0.md"
    ).read_text(encoding="utf-8").splitlines()
    h03, h04 = source[4], source[5]
    records = [
        _record(
            h03, character="迟蘅", statement="迟蘅拒绝了主持人的请求。",
            key_object="主持人的请求", line_number=1,
        ),
        _record(
            h04, character="柳泛",
            statement=(
                "柳泛在当天的公示上写“方向未核实”，原始读数留在观测簿中，"
                "复测结果另附一页。"
            ),
            key_object="方向未核实", line_number=2,
        ),
    ]
    result, provider = _extract([h03, h04], records)
    assert result.diagnostics.outcome == "completed"
    assert len(result.signals) == 2
    # H03 hits the lexical refusal bug. The exact H04 model record already
    # passes the ordinary binder, so it does not consume a review call.
    assert provider.calls.count(HISTORY_REVIEW_SYSTEM_PROMPT) == 1
    assert provider.review_requests[0]["candidate"]["character"] == "迟蘅"
    assert result.diagnostics.attempted_calls == 2
    assert result.pending_candidates == ()
    assert "history_deferred" not in result.model_dump_json()


def test_cross_clause_same_actor_continuation_can_reach_semantic_review():
    line = "柳泛发现风针卡住，便拒绝把未核实的方向写成确定预报。"
    record = _record(
        line, character="柳泛",
        statement="柳泛拒绝把未核实的方向写成确定预报。",
        key_object="未核实的方向",
    )
    raw = _RECORD_ADAPTER.validate_python(record)
    chunk = CharacterSignalChunk(
        document_id="doc-1", document_name="history.md", content=line + "\n",
        global_line_start=1, source_kind="published_history",
    )
    anchor = _history_target_anchor(raw, chunk)
    assert anchor is not None
    assert not _history_obviously_unresolved(raw, line, anchor[0])
    assert _history_lexical_only_deferred(0, raw, chunk) is not None
    result, provider = _extract([line], [record])
    assert len(result.signals) == 1, (
        result.diagnostics.reason_counts,
        provider.calls.count(HISTORY_REVIEW_SYSTEM_PROMPT),
    )
    assert provider.calls.count(HISTORY_REVIEW_SYSTEM_PROMPT) == 1
    assert provider.review_requests[0]["target_assertion_id"] == "H1:A2"


@pytest.mark.parametrize("review_mode", ["rejected", "invalid_json", "timeout", "429"])
def test_reviewer_failure_keeps_history_record_out_of_signals(review_mode):
    line = "林澈拒绝公开伤者姓名。"
    record = _record(
        line, character="林澈", statement="林澈拒绝公开伤者姓名。",
        key_object="伤者姓名",
    )
    result, provider = _extract([line], [record], review_mode=review_mode)
    assert result.diagnostics.outcome == "degraded"
    assert result.signals == ()
    assert result.pending_candidates == ()
    assert provider.calls.count(HISTORY_REVIEW_SYSTEM_PROMPT) == 1, (
        result.diagnostics.reason_counts
    )
    assert result.diagnostics.attempted_calls == 2
    serialized = result.model_dump_json()
    assert "拒绝公开伤者姓名" not in serialized
    assert "request_digest" not in serialized


@pytest.mark.parametrize("review_mode", ["timeout", "429", "invalid_json", "budget"])
def test_reviewer_uncertainty_cannot_be_erased_by_empty_regeneration(
    review_mode, monkeypatch
):
    line = "林澈拒绝公开伤者姓名。"
    record = _record(
        line, character="林澈", statement=line, key_object="伤者姓名"
    )
    provider = RegenerationProvider([record], review_mode=review_mode)
    if review_mode == "budget":
        actual_review = extraction_module.run_history_semantic_review

        def underfunded_review(request, **kwargs):
            return actual_review(request, **{**kwargs, "token_budget": 0})

        monkeypatch.setattr(
            extraction_module, "run_history_semantic_review", underfunded_review
        )
    content = line + "\n"
    source = HistoryReviewSourceIdentity(
        run_input_id="input-1", document_id="doc-1", document_version=3,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        source_kind="published_history",
    )
    result = CharacterSignalExtractor(
        provider, settings=_settings(
            character_signal_package_max_attempts=2,
            character_signal_token_budget=40_000,
        )
    ).extract(
        CharacterSignalChunk(
            document_id="doc-1", document_name="history.md", content=content,
            global_line_start=1, source_kind="published_history",
        ),
        source_identity=source, frozen_content=content,
    )
    assert provider.primary_calls == 2
    assert result.signals == ()
    assert result.pending_candidates == ()
    assert result.diagnostics.outcome == "partial", (
        result.diagnostics.reason_counts,
        provider.calls.count(HISTORY_REVIEW_SYSTEM_PROMPT),
    )
    assert result.diagnostics.reason_counts["history_semantic_review_incomplete"] == 1


def test_explicit_reviewer_rejection_keeps_existing_empty_regeneration_semantics():
    line = "林澈拒绝公开伤者姓名。"
    record = _record(
        line, character="林澈", statement=line, key_object="伤者姓名"
    )
    provider = RegenerationProvider([record], review_mode="rejected")
    content = line + "\n"
    source = HistoryReviewSourceIdentity(
        run_input_id="input-1", document_id="doc-1", document_version=3,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        source_kind="published_history",
    )
    result = CharacterSignalExtractor(
        provider, settings=_settings(
            character_signal_package_max_attempts=2,
            character_signal_token_budget=40_000,
        )
    ).extract(
        CharacterSignalChunk(
            document_id="doc-1", document_name="history.md", content=content,
            global_line_start=1, source_kind="published_history",
        ),
        source_identity=source, frozen_content=content,
    )
    assert result.diagnostics.outcome == "completed"
    assert result.signals == ()


def test_second_package_different_action_cannot_clear_first_review_debt():
    line = (
        "林澈拒绝公开伤者姓名，"
        "林澈又拒绝向记者透露伤者姓名。"
    )
    first = _record(
        line, character="林澈", statement="林澈拒绝公开伤者姓名",
        key_object="伤者姓名",
    )
    second = _record(
        line, character="林澈", statement="林澈又拒绝向记者透露伤者姓名。",
        key_object="伤者姓名",
    )
    provider = RegenerationProvider(
        [first], review_mode="timeout", second_records=[second],
        review_modes=["timeout", "supported"],
    )
    content = line + "\n"
    source = HistoryReviewSourceIdentity(
        run_input_id="input-1", document_id="doc-1", document_version=3,
        content_sha256=hashlib.sha256(content.encode()).hexdigest(),
        source_kind="published_history",
    )
    result = CharacterSignalExtractor(
        provider, settings=_settings(
            character_signal_package_max_attempts=2,
            character_signal_token_budget=40_000,
        )
    ).extract(
        CharacterSignalChunk(
            document_id="doc-1", document_name="history.md", content=content,
            global_line_start=1, source_kind="published_history",
        ),
        source_identity=source, frozen_content=content,
    )
    assert provider.primary_calls == 2
    assert result.diagnostics.outcome == "partial", (
        result.diagnostics.reason_counts,
        provider.calls.count(HISTORY_REVIEW_SYSTEM_PROMPT),
    )
    assert result.diagnostics.reason_counts["history_semantic_review_incomplete"] == 1


@pytest.mark.parametrize(
    "line,statement",
    [
        ("林澈看着周砚拒绝公开伤者姓名。", "林澈拒绝公开伤者姓名。"),
        ("林澈说周砚拒绝公开伤者姓名。", "林澈拒绝公开伤者姓名。"),
        ("林澈梦见自己拒绝公开伤者姓名。", "林澈拒绝公开伤者姓名。"),
        ("据说林澈拒绝公开伤者姓名。", "林澈拒绝公开伤者姓名。"),
        ("林澈本想拒绝公开伤者姓名，但最终同意公开。", "林澈拒绝公开伤者姓名"),
        ("林澈拒绝公开报告，周砚公开伤者姓名。", "林澈拒绝公开伤者姓名"),
    ],
)
def test_obvious_nonactual_or_other_actor_never_reaches_reviewer(line, statement):
    record = _record(
        line, character="林澈", statement=statement, key_object="伤者姓名"
    )
    result, provider = _extract([line], [record])
    assert result.signals == ()
    assert result.pending_candidates == ()
    assert HISTORY_REVIEW_SYSTEM_PROMPT not in provider.calls
    assert result.diagnostics.outcome == "degraded"


def test_wrong_object_in_neighbor_clause_cannot_be_semantically_rescued():
    line = "林澈拒绝公开报告，周砚把伤者姓名写上名单。"
    record = _record(
        line, character="林澈", statement="林澈拒绝公开伤者姓名",
        key_object="伤者姓名",
    )
    result, provider = _extract([line], [record])
    assert result.signals == ()
    assert HISTORY_REVIEW_SYSTEM_PROMPT not in provider.calls


@pytest.mark.parametrize(
    "line",
    [
        "周砚梦见自己公开伤者姓名。林澈拒绝公开伤者姓名。",
        "如果周砚公开伤者姓名，林澈拒绝公开伤者姓名。",
        "据说周砚公开了伤者姓名。林澈拒绝公开伤者姓名。",
    ],
)
def test_unrelated_prior_dream_or_hypothesis_does_not_block_real_target(line):
    record = _record(
        line, character="林澈", statement="林澈拒绝公开伤者姓名。",
        key_object="伤者姓名",
    )
    result, provider = _extract([line], [record])
    assert len(result.signals) == 1
    assert provider.calls.count(HISTORY_REVIEW_SYSTEM_PROMPT) == 1


def test_explicit_next_line_retraction_blocks_single_line_reviewer():
    first = "林澈拒绝公开伤者姓名。"
    second = "但上一幕只是梦，林澈并未做出这个决定。"
    record = _record(
        first, character="林澈", statement=first, key_object="伤者姓名"
    )
    result, provider = _extract([first, second], [record])
    assert result.signals == ()
    assert result.diagnostics.outcome == "degraded"
    assert HISTORY_REVIEW_SYSTEM_PROMPT not in provider.calls


def test_source_hash_mismatch_skips_before_any_provider_call():
    line = "林澈拒绝公开伤者姓名。"
    record = _record(
        line, character="林澈", statement=line, key_object="伤者姓名"
    )
    result, provider = _extract(
        [line], [record], frozen_content="另一个冻结版本\n"
    )
    assert result.diagnostics.outcome == "skipped"
    assert result.diagnostics.reason_counts == {"history_review_source_mismatch": 1}
    assert provider.calls == []


def test_budget_and_flag_off_do_not_expose_unreviewed_history():
    line = "林澈拒绝公开伤者姓名。"
    record = _record(
        line, character="林澈", statement=line, key_object="伤者姓名"
    )
    budget_result, budget_provider = _extract(
        [line], [record],
        settings=_settings(character_signal_token_budget=6_000),
    )
    assert budget_result.diagnostics.outcome == "skipped"
    assert budget_provider.calls == []
    off_result, off_provider = _extract(
        [line], [record], settings=_settings(
            character_history_semantic_review_v1=False
        ),
    )
    assert off_result.signals == ()
    assert HISTORY_REVIEW_SYSTEM_PROMPT not in off_provider.calls


def test_formal_and_draft_never_use_history_review_purpose():
    line = "林澈拒绝公开伤者姓名。"
    record = _record(
        line, character="林澈", statement=line, key_object="伤者姓名"
    )
    for source_kind in ("formal_character_profile", "draft"):
        result, provider = _extract([line], [record], source_kind=source_kind)
        assert result.signals == ()
        assert HISTORY_REVIEW_SYSTEM_PROMPT not in provider.calls


def test_accounting_provider_forks_history_purpose_with_one_transport_attempt():
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
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 9},
        })

    gateway = OpenAICompatibleProvider(
        configured, transport=httpx.MockTransport(respond),
        retry_policy=RetryPolicy(max_attempts=3),
    )
    usage = CharacterConsistencyUsageAccumulator()
    accounting = _CharacterConsistencyAccountingProvider(
        configured, usage, signal_provider=gateway, drift_provider=gateway,
    )
    bounded = accounting.fork_for_character_history_semantic_review(
        timeout_seconds=4, remaining_deadline_seconds=8,
        completion_reserve=320, max_response_bytes=6_000,
        max_attempts=1,
    )
    inner = bounded.history_review_provider
    assert isinstance(inner, OpenAICompatibleProvider)
    assert inner.settings.provider_timeout_seconds == 4
    assert inner.settings.provider_total_deadline_seconds == 4
    assert inner.settings.provider_max_completion_tokens == 320
    assert inner.settings.provider_max_response_bytes == 6_000
    assert inner.retry_policy.max_attempts == 1
    assert inner.settings.provider_max_attempts == 1
    bounded.complete(HISTORY_REVIEW_SYSTEM_PROMPT, "review request")
    assert len(seen) == 1
    assert json.loads(seen[0].content)["max_tokens"] == 320
    assert usage.logical_calls == 1
    assert usage.charged_tokens == estimate_issue_evidence_review_tokens(
        HISTORY_REVIEW_SYSTEM_PROMPT, "review request", completion_reserve=320,
    )


def test_accounting_provider_rejects_history_purpose_when_flag_off():
    off = _settings(character_history_semantic_review_v1=False)
    provider = QueueProvider([])
    accounting = _CharacterConsistencyAccountingProvider(
        off, CharacterConsistencyUsageAccumulator(),
        signal_provider=provider, drift_provider=provider,
    )
    with pytest.raises(RuntimeError, match="unsupported character consistency provider purpose"):
        accounting.complete(HISTORY_REVIEW_SYSTEM_PROMPT, "review request")
    assert provider.calls == []
