import json
import math
from pathlib import Path

import httpx
import pytest
import yaml
from pydantic import ValidationError

from app.config import Settings
from app.domain import ProviderCallDiagnostics
from app.provider import (
    OpenAICompatibleProvider,
    ProviderCallTelemetry,
    ToolCallLimits,
    ToolDefinition,
)


ROOT = Path(__file__).resolve().parents[1]


def settings(**overrides):
    values = {
        "openai_api_key": "unit-test-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock",
        "embedding_base_url": "https://embedding.invalid/v1",
        "embedding_model": "mock-embedding",
        "embedding_model_revision": "r1",
        "embedding_deployment_fingerprint": "test-deployment-v1",
        "embedding_dimensions": 2,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_evidence_investigator_defaults_are_bounded_and_disabled():
    configured = settings()

    assert configured.enable_evidence_investigator is False
    assert {
        "max_seeds": configured.evidence_investigator_max_seeds,
        "max_decision_rounds": (
            configured.evidence_investigator_max_decision_rounds
        ),
        "max_tool_calls": configured.evidence_investigator_max_tool_calls,
        "max_searches": configured.evidence_investigator_max_searches,
        "max_reads": configured.evidence_investigator_max_reads,
        "max_results": configured.evidence_investigator_max_results,
        "max_read_lines": configured.evidence_investigator_max_read_lines,
        "max_span_chars": configured.evidence_investigator_max_span_chars,
        "token_budget": configured.evidence_investigator_token_budget,
        "max_prompt_bytes": configured.evidence_investigator_max_prompt_bytes,
        "timeout_seconds": configured.evidence_investigator_timeout_seconds,
        "total_deadline_seconds": (
            configured.evidence_investigator_total_deadline_seconds
        ),
        "max_completion_tokens": (
            configured.evidence_investigator_max_completion_tokens
        ),
        "max_response_bytes": (
            configured.evidence_investigator_max_response_bytes
        ),
        "top_k": configured.evidence_investigator_top_k,
        "branch_limit": configured.evidence_investigator_branch_limit,
        "embedding_max_input_chars": (
            configured.evidence_investigator_embedding_max_input_chars
        ),
        "require_hybrid": configured.evidence_investigator_require_hybrid,
    } == {
        "max_seeds": 1,
        "max_decision_rounds": 6,
        "max_tool_calls": 6,
        "max_searches": 2,
        "max_reads": 2,
        "max_results": 12,
        "max_read_lines": 12,
        "max_span_chars": 12_000,
        "token_budget": 16_000,
        "max_prompt_bytes": 128 * 1_024,
        "timeout_seconds": 30.0,
        "total_deadline_seconds": 60.0,
        "max_completion_tokens": 768,
        "max_response_bytes": 64_000,
        "top_k": 6,
        "branch_limit": 30,
        "embedding_max_input_chars": 250_000,
        "require_hybrid": True,
    }
    assert (
        configured.evidence_investigator_token_budget
        < configured.per_run_token_budget
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("evidence_investigator_max_seeds", 9),
        ("evidence_investigator_max_decision_rounds", 33),
        ("evidence_investigator_max_tool_calls", 33),
        ("evidence_investigator_max_searches", 9),
        ("evidence_investigator_max_reads", 17),
        ("evidence_investigator_max_results", 49),
        ("evidence_investigator_max_read_lines", 21),
        ("evidence_investigator_max_span_chars", 24_001),
        ("evidence_investigator_token_budget", 20_001),
        ("evidence_investigator_max_prompt_bytes", 256 * 1_024 + 1),
        ("evidence_investigator_timeout_seconds", 30.001),
        ("evidence_investigator_total_deadline_seconds", 60.001),
        ("evidence_investigator_max_completion_tokens", 2_049),
        ("evidence_investigator_max_response_bytes", 128_001),
        ("evidence_investigator_top_k", 13),
        ("evidence_investigator_branch_limit", 51),
        ("evidence_investigator_embedding_max_input_chars", 1_000_001),
        ("evidence_investigator_require_hybrid", []),
    ],
)
def test_evidence_investigator_hard_upper_bounds(field, invalid):
    with pytest.raises(ValidationError):
        settings(**{field: invalid})


def test_evidence_investigator_maximum_seed_setting_has_a_valid_configuration():
    configured = settings(
        evidence_investigator_max_seeds=8,
        evidence_investigator_max_decision_rounds=24,
        evidence_investigator_max_tool_calls=24,
        evidence_investigator_max_searches=8,
        evidence_investigator_max_reads=8,
        evidence_investigator_max_results=48,
        evidence_investigator_token_budget=20_000,
    )

    assert configured.evidence_investigator_max_seeds == 8


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "evidence_investigator_max_seeds": 2,
            "evidence_investigator_max_decision_rounds": 5,
        },
        {
            "evidence_investigator_max_seeds": 2,
            "evidence_investigator_max_tool_calls": 5,
        },
        {
            "evidence_investigator_max_decision_rounds": 7,
            "evidence_investigator_max_tool_calls": 6,
        },
        {
            "evidence_investigator_max_seeds": 2,
            "evidence_investigator_max_searches": 1,
        },
        {
            "evidence_investigator_max_seeds": 2,
            "evidence_investigator_max_reads": 1,
        },
        {
            "evidence_investigator_max_results": 5,
            "evidence_investigator_top_k": 6,
        },
        {
            "evidence_investigator_max_seeds": 2,
            "evidence_investigator_max_results": 11,
            "evidence_investigator_top_k": 6,
        },
        {
            "evidence_investigator_branch_limit": 5,
            "evidence_investigator_top_k": 6,
        },
        {
            "evidence_investigator_timeout_seconds": 20,
            "evidence_investigator_total_deadline_seconds": 19,
        },
        {"evidence_investigator_token_budget": 2_000},
        {
            "enable_evidence_investigator": True,
            "enable_embeddings": False,
        },
    ],
)
def test_evidence_investigator_cross_field_invariants(overrides):
    with pytest.raises(ValidationError, match="evidence investigator|hybrid"):
        settings(**overrides)


def test_non_hybrid_investigator_still_requires_embedding_capability():
    with pytest.raises(ValidationError, match="requires the embeddings capability"):
        settings(
            enable_evidence_investigator=True,
            enable_embeddings=False,
            evidence_investigator_require_hybrid=False,
        )


def test_non_hybrid_investigator_allows_diagnosed_fallback_when_embeddings_enabled():
    configured = settings(
        enable_evidence_investigator=True,
        enable_embeddings=True,
        evidence_investigator_require_hybrid=False,
    )

    assert configured.enable_evidence_investigator is True
    assert configured.enable_embeddings is True
    assert configured.evidence_investigator_require_hybrid is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"embedding_base_url": ""},
        {"embedding_base_url": "http://embedding.invalid/v1"},
        {"embedding_base_url": "https://user:secret@embedding.invalid/v1"},
        {"embedding_model": ""},
        {"embedding_model_revision": "unspecified"},
        {"embedding_deployment_fingerprint": "unspecified"},
        {"embedding_dimensions": None},
    ],
)
def test_enabled_investigator_requires_a_complete_embedding_profile(overrides):
    with pytest.raises(ValidationError, match="complete embedding profile"):
        settings(
            enable_evidence_investigator=True,
            enable_embeddings=True,
            **overrides,
        )


def test_provider_configured_includes_but_does_not_borrow_investigator_capability():
    investigator = OpenAICompatibleProvider(
        settings(
            enable_evidence_investigator=True,
            enable_embeddings=True,
        )
    )
    extraction_only = OpenAICompatibleProvider(
        settings(
            enable_model_extraction=True,
            enable_evidence_investigator=False,
        )
    )

    assert investigator.configured is True
    assert investigator.evidence_investigator_configured is True
    assert extraction_only.configured is True
    assert extraction_only.evidence_investigator_configured is False


def test_investigator_provider_fork_applies_dedicated_limits_and_one_attempt():
    transport = httpx.MockTransport(lambda _: httpx.Response(500))
    sleep = lambda _: None
    monotonic = lambda: 10.0
    wall_time = lambda: 20.0
    random_value = lambda: 0.25
    provider = OpenAICompatibleProvider(
        settings(
            enable_evidence_investigator=True,
            enable_embeddings=True,
            enable_model_extraction=True,
            enable_issue_evidence_review=True,
            enable_review_agent=True,
            provider_max_attempts=4,
            provider_timeout_seconds=30,
            provider_total_deadline_seconds=None,
            provider_max_completion_tokens=None,
            provider_max_response_bytes=None,
        ),
        transport=transport,
        sleep=sleep,
        monotonic=monotonic,
        wall_time=wall_time,
        random_value=random_value,
    )

    fork = provider.fork_for_evidence_investigator()

    assert fork is not provider
    assert fork.configured is True
    assert fork.evidence_investigator_configured is True
    assert fork.settings.enable_model_extraction is False
    assert fork.settings.enable_issue_evidence_review is False
    assert fork.settings.enable_review_agent is False
    assert fork.settings.provider_timeout_seconds == 30
    assert fork.settings.provider_total_deadline_seconds == 60
    assert fork.settings.provider_max_completion_tokens == 768
    assert fork.settings.provider_max_response_bytes == 64_000
    assert fork.settings.provider_max_attempts == 1
    assert fork.retry_policy.max_attempts == 1
    assert fork.retry_policy.base_delay_seconds == 0
    assert fork.retry_policy.jitter_ratio == 0
    assert fork.transport is transport
    assert fork.sleep is sleep
    assert fork.monotonic is monotonic
    assert fork.wall_time is wall_time
    assert fork.random_value is random_value


def test_investigator_provider_fork_never_expands_base_or_remaining_limits():
    provider = OpenAICompatibleProvider(
        settings(
            enable_evidence_investigator=True,
            enable_embeddings=True,
            provider_timeout_seconds=7,
            provider_total_deadline_seconds=9,
            provider_max_completion_tokens=320,
            provider_max_response_bytes=32_000,
        )
    )

    fork = provider.fork_for_evidence_investigator(
        remaining_deadline_seconds=5
    )

    assert fork.settings.provider_timeout_seconds == 5
    assert fork.settings.provider_total_deadline_seconds == 5
    assert fork.settings.provider_max_completion_tokens == 320
    assert fork.settings.provider_max_response_bytes == 32_000


def test_investigator_provider_fork_caps_each_turn_by_remaining_total_deadline():
    provider = OpenAICompatibleProvider(
        settings(
            enable_evidence_investigator=True,
            enable_embeddings=True,
            provider_timeout_seconds=30,
            provider_total_deadline_seconds=None,
            provider_max_attempts=4,
        )
    )

    first_turn = provider.fork_for_evidence_investigator(
        remaining_deadline_seconds=60
    )
    final_turn = provider.fork_for_evidence_investigator(
        remaining_deadline_seconds=12.5
    )

    assert first_turn.settings.provider_timeout_seconds == 30
    assert first_turn.settings.provider_total_deadline_seconds == 60
    assert final_turn.settings.provider_timeout_seconds == 12.5
    assert final_turn.settings.provider_total_deadline_seconds == 12.5
    assert first_turn.settings.provider_max_attempts == 1
    assert final_turn.settings.provider_max_attempts == 1


def test_investigator_provider_fork_sends_its_completion_cap():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "ABSTAIN",
                                        "arguments": '{"reason":"done"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    provider = OpenAICompatibleProvider(
        settings(
            enable_evidence_investigator=True,
            enable_embeddings=True,
        ),
        transport=httpx.MockTransport(handler),
    ).fork_for_evidence_investigator()

    provider.complete_with_tools(
        "system",
        "user",
        tools=(
            ToolDefinition(
                name="ABSTAIN",
                description="Stop.",
                parameters={
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            ),
        ),
        limits=ToolCallLimits(max_calls=1, max_argument_bytes=1_024),
    )

    assert len(requests) == 1
    payload = json.loads(requests[0].content)
    assert payload["max_tokens"] == 768


def test_disabled_investigator_fork_is_not_enabled_by_extraction():
    provider = OpenAICompatibleProvider(
        settings(
            enable_model_extraction=True,
            enable_evidence_investigator=False,
        )
    )

    fork = provider.fork_for_evidence_investigator()

    assert provider.configured is True
    assert fork.configured is False
    assert fork.evidence_investigator_configured is False


@pytest.mark.parametrize(
    "remaining",
    [0, -1, math.inf, math.nan, True, "5"],
)
def test_investigator_provider_fork_rejects_invalid_remaining_deadline(remaining):
    provider = OpenAICompatibleProvider(settings())
    with pytest.raises(ValueError, match="remaining deadline"):
        provider.fork_for_evidence_investigator(
            remaining_deadline_seconds=remaining
        )


def test_investigator_provider_diagnostic_purpose_and_tool_category_are_safe():
    telemetry = ProviderCallTelemetry(
        input_chars=100,
        elapsed_ms=12,
        category="tool_response_shape",
    )

    diagnostic = ProviderCallDiagnostics.from_telemetry(
        telemetry,
        succeeded=False,
        purpose="investigator",
    )

    assert diagnostic is not None
    assert diagnostic.safe_dict()["purpose"] == "investigator"
    assert diagnostic.safe_dict()["category"] == "tool_response_shape"

    forged = ProviderCallDiagnostics.from_telemetry(
        telemetry,
        succeeded=False,
        purpose="prompt-content",  # type: ignore[arg-type]
    )
    assert forged is not None
    assert forged.safe_dict()["purpose"] == "extract"


def test_compose_passes_investigator_limits_without_rag_implicitly_enabling_it():
    compose = yaml.safe_load(
        (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    )
    defaults = {
        "ENABLE_EVIDENCE_INVESTIGATOR": "false",
        "EVIDENCE_INVESTIGATOR_MAX_SEEDS": "1",
        "EVIDENCE_INVESTIGATOR_MAX_DECISION_ROUNDS": "6",
        "EVIDENCE_INVESTIGATOR_MAX_TOOL_CALLS": "6",
        "EVIDENCE_INVESTIGATOR_MAX_SEARCHES": "2",
        "EVIDENCE_INVESTIGATOR_MAX_READS": "2",
        "EVIDENCE_INVESTIGATOR_MAX_RESULTS": "12",
        "EVIDENCE_INVESTIGATOR_MAX_READ_LINES": "12",
        "EVIDENCE_INVESTIGATOR_MAX_SPAN_CHARS": "12000",
        "EVIDENCE_INVESTIGATOR_TOKEN_BUDGET": "16000",
        "EVIDENCE_INVESTIGATOR_MAX_PROMPT_BYTES": "131072",
        "EVIDENCE_INVESTIGATOR_TIMEOUT_SECONDS": "30",
        "EVIDENCE_INVESTIGATOR_TOTAL_DEADLINE_SECONDS": "60",
        "EVIDENCE_INVESTIGATOR_MAX_COMPLETION_TOKENS": "768",
        "EVIDENCE_INVESTIGATOR_MAX_RESPONSE_BYTES": "64000",
        "EVIDENCE_INVESTIGATOR_TOP_K": "6",
        "EVIDENCE_INVESTIGATOR_BRANCH_LIMIT": "30",
        "EVIDENCE_INVESTIGATOR_EMBEDDING_MAX_INPUT_CHARS": "250000",
        "EVIDENCE_INVESTIGATOR_REQUIRE_HYBRID": "true",
    }
    expected = {
        key: "${" + key + ":-" + default + "}"
        for key, default in defaults.items()
    }
    for service_name in ("api", "worker"):
        environment = compose["services"][service_name]["environment"]
        assert {key: environment.get(key) for key in expected} == expected

    rag_override = (ROOT / "docker-compose.rag.yml").read_text(encoding="utf-8")
    assert "ENABLE_EVIDENCE_INVESTIGATOR" not in rag_override

    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    for key in expected:
        assert f"{key}=" in example
