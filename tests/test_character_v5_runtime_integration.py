from __future__ import annotations

import json
from hashlib import sha256
from types import SimpleNamespace

import httpx
import pytest

from app.character_scope_review import ScopeReviewSourceIdentity
from app.character_scope_review_provider import SCOPE_REVIEW_SYSTEM_PROMPT
from app.character_trait_extraction import (
    CHARACTER_SIGNAL_CORE_SCOPE_PROMPT_V3,
    CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2,
    CHARACTER_SIGNAL_SCOPE_REVIEW_PROMPT_V1,
    CHARACTER_SIGNAL_SEMANTIC_SCOPE_REVIEW_PROMPT_V1,
    CHARACTER_SIGNAL_SEMANTIC_SCOPE_PROMPT_V5,
    CHARACTER_SIGNAL_SUPPORT_ID_PROMPT_V4,
    CHARACTER_SIGNAL_SYSTEM_PROMPT,
    CHARACTER_SIGNAL_SYSTEM_PROMPT_V5,
    TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT,
    CharacterSignalChunk,
    CharacterSignalExtractor,
)
from app.config import Settings
from app.provider import OpenAICompatibleProvider, ProviderError, RetryPolicy
from app.service import (
    CharacterConsistencyUsageAccumulator,
    _CharacterConsistencyAccountingProvider,
)
from app.usage import estimate_issue_evidence_review_tokens


class RecordingProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str):
        self.calls.append((system, user))
        return SimpleNamespace(text='{"records":[]}', prompt_tokens=17, completion_tokens=9)


def test_v5_accounting_routes_only_active_prompts_and_keeps_old_protocol_when_off():
    configured = Settings(
        _env_file=None,
        openai_api_key="unit-test-placeholder",
        openai_base_url="https://mock.invalid/v1",
        openai_model="mock-model",
        enable_character_consistency=True,
        character_signal_full_line_prompt_v2=True,
        character_signal_core_scope_prompt_v3=True,
        character_signal_support_id_v4=True,
        character_signal_semantic_scope_v5=True,
    )
    provider = RecordingProvider()
    usage = CharacterConsistencyUsageAccumulator()
    accounting = _CharacterConsistencyAccountingProvider(
        configured, usage, signal_provider=provider, drift_provider=provider,
    )
    history_system = (
        CHARACTER_SIGNAL_SYSTEM_PROMPT
        + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2
        + CHARACTER_SIGNAL_CORE_SCOPE_PROMPT_V3
    )
    formal_v4_system = history_system + CHARACTER_SIGNAL_SUPPORT_ID_PROMPT_V4
    formal_v5_system = (
        CHARACTER_SIGNAL_SYSTEM_PROMPT_V5
        + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2
        + CHARACTER_SIGNAL_SEMANTIC_SCOPE_PROMPT_V5
    )

    accounting.complete(formal_v5_system, "formal request")
    accounting.complete(history_system, "history request")
    accounting.complete(TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT, "targeted request")
    assert [system for system, _ in provider.calls] == [
        formal_v5_system, history_system, TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT,
    ]
    assert usage.logical_calls == 3

    with pytest.raises(
        RuntimeError, match="unsupported character consistency provider purpose"
    ):
        accounting.complete(formal_v4_system, "inactive formal request")
    assert usage.logical_calls == 3
    assert len(provider.calls) == 3

    off_provider = RecordingProvider()
    off_usage = CharacterConsistencyUsageAccumulator()
    off_accounting = _CharacterConsistencyAccountingProvider(
        configured.model_copy(update={"character_signal_semantic_scope_v5": False}),
        off_usage, signal_provider=off_provider, drift_provider=off_provider,
    )
    off_accounting.complete(formal_v4_system, "v4 formal request")
    assert off_provider.calls == [(formal_v4_system, "v4 formal request")]
    with pytest.raises(
        RuntimeError, match="unsupported character consistency provider purpose"
    ):
        off_accounting.complete(formal_v5_system, "disabled v5 request")
    assert off_usage.logical_calls == 1


def test_v5_extractor_prompt_reaches_accounting_for_formal_history_and_draft():
    configured = Settings(
        _env_file=None,
        openai_api_key="unit-test-placeholder",
        openai_base_url="https://mock.invalid/v1",
        openai_model="mock-model",
        enable_character_consistency=True,
        character_signal_full_line_prompt_v2=True,
        character_signal_core_scope_prompt_v3=True,
        character_signal_support_id_v4=True,
        character_signal_semantic_scope_v5=True,
    )
    provider = RecordingProvider()
    usage = CharacterConsistencyUsageAccumulator()
    accounting = _CharacterConsistencyAccountingProvider(
        configured, usage, signal_provider=provider, drift_provider=provider,
    )
    extractor = CharacterSignalExtractor(accounting, settings=configured)
    for kind in ("formal_character_profile", "published_history", "draft"):
        result = extractor.extract(CharacterSignalChunk(
            "doc", "profile.md", "林澈长期喜欢蜜瓜。", 1, kind,
        ))
        assert result.diagnostics.attempted_calls == 1
    assert [system for system, _ in provider.calls] == [
        CHARACTER_SIGNAL_SYSTEM_PROMPT_V5
        + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2
        + CHARACTER_SIGNAL_SEMANTIC_SCOPE_PROMPT_V5,
        CHARACTER_SIGNAL_SYSTEM_PROMPT
        + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2
        + CHARACTER_SIGNAL_CORE_SCOPE_PROMPT_V3,
        CHARACTER_SIGNAL_SYSTEM_PROMPT
        + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2
        + CHARACTER_SIGNAL_CORE_SCOPE_PROMPT_V3,
    ]
    assert usage.logical_calls == 3


def _scope_review_settings(**overrides) -> Settings:
    values = {
        "openai_api_key": "unit-test-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_character_consistency": True,
        "character_signal_full_line_prompt_v2": True,
        "character_signal_support_id_v4": True,
        "character_signal_semantic_scope_v5": True,
        "character_signal_scope_review_v1": True,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_scope_review_accounting_accepts_only_active_formal_and_review_prompts():
    configured = _scope_review_settings()
    provider = RecordingProvider()
    usage = CharacterConsistencyUsageAccumulator()
    accounting = _CharacterConsistencyAccountingProvider(
        configured, usage, signal_provider=provider, drift_provider=provider,
    )
    formal_v5_off = (
        CHARACTER_SIGNAL_SYSTEM_PROMPT_V5
        + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2
        + CHARACTER_SIGNAL_SEMANTIC_SCOPE_PROMPT_V5
    )
    formal_review_on = (
        CHARACTER_SIGNAL_SYSTEM_PROMPT_V5
        + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2
        + CHARACTER_SIGNAL_SEMANTIC_SCOPE_REVIEW_PROMPT_V1
        + CHARACTER_SIGNAL_SCOPE_REVIEW_PROMPT_V1
    )
    history = CHARACTER_SIGNAL_SYSTEM_PROMPT + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2
    assert formal_review_on != formal_v5_off
    accounting.complete(formal_review_on, "formal")
    accounting.complete(history, "history")
    accounting.complete(TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT, "targeted")
    before_review_charge = usage.charged_tokens
    accounting.complete(SCOPE_REVIEW_SYSTEM_PROMPT, "review")
    assert usage.charged_tokens - before_review_charge == (
        estimate_issue_evidence_review_tokens(
            SCOPE_REVIEW_SYSTEM_PROMPT, "review",
            completion_reserve=configured.character_signal_scope_review_completion_tokens,
        )
    )
    assert [system for system, _ in provider.calls] == [
        formal_review_on, history, TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT,
        SCOPE_REVIEW_SYSTEM_PROMPT,
    ]
    assert usage.logical_calls == 4
    with pytest.raises(RuntimeError, match="unsupported character consistency provider purpose"):
        accounting.complete(formal_v5_off, "inactive formal")
    assert usage.logical_calls == 4

    off_provider = RecordingProvider()
    off_usage = CharacterConsistencyUsageAccumulator()
    off_accounting = _CharacterConsistencyAccountingProvider(
        configured.model_copy(update={"character_signal_scope_review_v1": False}),
        off_usage, signal_provider=off_provider, drift_provider=off_provider,
    )
    off_accounting.complete(formal_v5_off, "v5 formal")
    for inactive in (formal_review_on, SCOPE_REVIEW_SYSTEM_PROMPT):
        with pytest.raises(
            RuntimeError, match="unsupported character consistency provider purpose"
        ):
            off_accounting.complete(inactive, "inactive")
    assert off_provider.calls == [(formal_v5_off, "v5 formal")]
    assert off_usage.logical_calls == 1


def test_scope_review_extractor_emits_active_formal_prompt_and_original_other_prompts():
    configured = _scope_review_settings()
    provider = RecordingProvider()
    usage = CharacterConsistencyUsageAccumulator()
    accounting = _CharacterConsistencyAccountingProvider(
        configured, usage, signal_provider=provider, drift_provider=provider,
    )
    extractor = CharacterSignalExtractor(accounting, settings=configured)
    content = "林澈长期喜欢蜜瓜。"
    source = ScopeReviewSourceIdentity(
        run_input_id="run-input", document_id="doc", document_version=1,
        content_sha256=sha256(content.encode("utf-8")).hexdigest(),
    )
    formal = extractor.extract(
        CharacterSignalChunk("doc", "profile.md", content, 1, "formal_character_profile"),
        source_identity=source, frozen_content=content,
    )
    history = extractor.extract(CharacterSignalChunk(
        "doc", "history.md", content, 1, "published_history",
    ))
    draft = extractor.extract(CharacterSignalChunk(
        "doc", "draft.md", content, 1, "draft",
    ))
    assert [result.diagnostics.attempted_calls for result in (formal, history, draft)] == [
        1, 1, 1,
    ]
    assert [system for system, _ in provider.calls] == [
        CHARACTER_SIGNAL_SYSTEM_PROMPT_V5
        + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2
        + CHARACTER_SIGNAL_SEMANTIC_SCOPE_REVIEW_PROMPT_V1
        + CHARACTER_SIGNAL_SCOPE_REVIEW_PROMPT_V1,
        CHARACTER_SIGNAL_SYSTEM_PROMPT + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2,
        CHARACTER_SIGNAL_SYSTEM_PROMPT + CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2,
    ]
    assert usage.logical_calls == 3


def test_scope_review_accounting_fork_enforces_dynamic_http_caps_and_one_charge():
    configured = _scope_review_settings(
        provider_max_attempts=3,
        provider_timeout_seconds=20,
        provider_total_deadline_seconds=40,
        provider_max_completion_tokens=3_000,
        provider_max_response_bytes=20_000,
        character_signal_scope_review_max_attempts=2,
    )
    seen: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 9},
        })

    provider = OpenAICompatibleProvider(
        configured,
        transport=httpx.MockTransport(respond),
        retry_policy=RetryPolicy(max_attempts=3),
    )
    usage = CharacterConsistencyUsageAccumulator()
    accounting = _CharacterConsistencyAccountingProvider(
        configured, usage, signal_provider=provider, drift_provider=provider,
    )
    bounded = accounting.fork_for_character_scope_review(
        timeout_seconds=4,
        remaining_deadline_seconds=8,
        completion_reserve=320,
        max_response_bytes=6_000,
        max_attempts=3,
    )
    inner = bounded.scope_review_provider
    assert isinstance(inner, OpenAICompatibleProvider)
    assert inner.settings.provider_timeout_seconds == 4
    assert inner.settings.provider_total_deadline_seconds == 4
    assert inner.settings.provider_max_completion_tokens == 320
    assert inner.settings.provider_max_response_bytes == 6_000
    assert inner.retry_policy.max_attempts == 2
    assert inner.settings.provider_max_attempts == 2
    assert bounded.usage is usage

    response = bounded.complete(SCOPE_REVIEW_SYSTEM_PROMPT, "review request")
    assert response.text == "{}"
    assert len(seen) == 1
    assert json.loads(seen[0].content)["max_tokens"] == 320
    assert seen[0].extensions["timeout"]["read"] <= 4
    assert usage.logical_calls == 1
    assert usage.prompt_tokens == 17
    assert usage.completion_tokens == 9
    assert usage.charged_tokens == estimate_issue_evidence_review_tokens(
        SCOPE_REVIEW_SYSTEM_PROMPT, "review request", completion_reserve=320,
    )
    tighter_provider = OpenAICompatibleProvider(
        configured, transport=httpx.MockTransport(respond),
        retry_policy=RetryPolicy(max_attempts=1),
    )
    tighter_accounting = _CharacterConsistencyAccountingProvider(
        configured, CharacterConsistencyUsageAccumulator(),
        signal_provider=tighter_provider, drift_provider=tighter_provider,
    )
    tighter = tighter_accounting.fork_for_character_scope_review(
        timeout_seconds=4, remaining_deadline_seconds=8,
        completion_reserve=320, max_response_bytes=6_000, max_attempts=3,
    )
    assert tighter.scope_review_provider.retry_policy.max_attempts == 1


def test_scope_review_http_rejects_oversize_and_one_attempt_is_one_charge():
    configured = _scope_review_settings(provider_max_attempts=3)
    seen: list[httpx.Request] = []

    def too_large(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "x" * 4_000}}],
            "usage": {"prompt_tokens": 17, "completion_tokens": 9},
        })

    provider = OpenAICompatibleProvider(
        configured, transport=httpx.MockTransport(too_large),
        retry_policy=RetryPolicy(max_attempts=3),
    )
    usage = CharacterConsistencyUsageAccumulator()
    accounting = _CharacterConsistencyAccountingProvider(
        configured, usage, signal_provider=provider, drift_provider=provider,
    )
    bounded = accounting.fork_for_character_scope_review(
        timeout_seconds=5,
        remaining_deadline_seconds=10,
        completion_reserve=320,
        max_response_bytes=1_024,
        max_attempts=3,
    )
    with pytest.raises(ProviderError):
        bounded.complete(SCOPE_REVIEW_SYSTEM_PROMPT, "review request")
    assert bounded.scope_review_provider.settings.provider_max_response_bytes == 1_024
    assert len(seen) == 1
    assert usage.logical_calls == 1
    assert usage.successful_calls == 0
    assert usage.charged_tokens == estimate_issue_evidence_review_tokens(
        SCOPE_REVIEW_SYSTEM_PROMPT, "review request", completion_reserve=320,
    )

    rate_limited: list[httpx.Request] = []

    def unavailable(request: httpx.Request) -> httpx.Response:
        rate_limited.append(request)
        return httpx.Response(429, headers={"Retry-After": "0"})

    one_attempt_provider = OpenAICompatibleProvider(
        configured,
        transport=httpx.MockTransport(unavailable),
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0, jitter_ratio=0),
    )
    retry_usage = CharacterConsistencyUsageAccumulator()
    retry_accounting = _CharacterConsistencyAccountingProvider(
        configured, retry_usage,
        signal_provider=one_attempt_provider, drift_provider=one_attempt_provider,
    )
    one_attempt = retry_accounting.fork_for_character_scope_review(
        timeout_seconds=5,
        remaining_deadline_seconds=10,
        completion_reserve=320,
        max_response_bytes=6_000,
        max_attempts=3,
    )
    with pytest.raises(ProviderError):
        one_attempt.complete(SCOPE_REVIEW_SYSTEM_PROMPT, "review request")
    assert one_attempt.scope_review_provider.retry_policy.max_attempts == 1
    assert len(rate_limited) == 1
    assert retry_usage.logical_calls == 1
    assert retry_usage.successful_calls == 0
