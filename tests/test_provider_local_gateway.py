"""Synthetic SIWC wire-profile coverage; no gateway, OAuth or live model."""

from __future__ import annotations

import gzip
import json

import httpx
import pytest
from pydantic import ValidationError

from app.config import Settings
from app.model_extractor import ExtractionEnvelope
from app.provider import (
    NamedToolChoice,
    OpenAICompatibleProvider,
    ProviderError,
    ProviderRetryExhausted,
    ProviderToolCallError,
    RetryPolicy,
    ToolDefinition,
    safe_thinking_configuration,
)
from app.provider_credentials import provider_endpoint_sha256


LOCAL_ORIGIN = "http://127.0.0.1:8317"
LOCAL_BASE_URL = f"{LOCAL_ORIGIN}/v1"


def settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "auth_mode": "anonymous",
        "deployment_environment": "local",
        "enable_model_extraction": True,
        "local_model_gateway_enabled": True,
        "local_model_gateway_origins": LOCAL_ORIGIN,
        "openai_base_url": LOCAL_BASE_URL,
        "openai_model": "synthetic-siwc-model",
        "openai_api_key": "synthetic-fixed-client-key",
        "provider_timeout_seconds": 1,
        "provider_max_attempts": 1,
        "provider_max_completion_tokens": 64,
        "provider_thinking_mode": "disabled",
    }
    values.update(overrides)
    return Settings(**values)


def provider(handler, *, configured=None) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        configured or settings(),
        transport=httpx.MockTransport(handler),
        retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0, jitter_ratio=0),
        sleep=lambda _: None,
    )


def completion(content='{"records":[]}', *, completion_tokens=3) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": completion_tokens},
        },
    )


def tool() -> ToolDefinition:
    return ToolDefinition(
        name="read_span",
        description="Read an authorized synthetic span.",
        parameters={
            "type": "object",
            "properties": {"document_ref": {"type": "string"}},
            "required": ["document_ref"],
            "additionalProperties": False,
        },
    )


def tool_response(*, count=1, name="read_span", arguments='{"document_ref":"d1"}'):
    return httpx.Response(
        200,
        json={
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{index}",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                        for index in range(count)
                    ],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        },
    )


@pytest.mark.parametrize("host", ["127.0.0.1", "host.docker.internal"])
def test_authorized_gateway_uses_explicit_non_strict_object_schema(host):
    requests = []

    def handler(request):
        requests.append(request)
        return completion()

    origin = f"http://{host}:8317"
    result = provider(handler, configured=settings(
        local_model_gateway_origins=origin,
        openai_base_url=f"{origin}/v1",
    )).complete("system", "user")

    assert json.loads(result.text) == {"records": []}
    request = requests[0]
    assert str(request.url) == f"{origin}/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer synthetic-fixed-client-key"
    payload = json.loads(request.content)
    assert payload == {
        "model": "synthetic-siwc-model",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "loreguard_json_object",
                "strict": False,
                "schema": {"type": "object", "additionalProperties": True},
            },
        },
    }
    assert safe_thinking_configuration(settings(
        local_model_gateway_origins=origin,
        openai_base_url=f"{origin}/v1",
    )) == {"configured": False, "mode": None}


def test_ordinary_https_wire_contract_is_unchanged_even_when_local_opt_in_is_enabled():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return completion()

    configured = settings(openai_base_url="https://public-provider.invalid/v1")
    provider(handler, configured=configured).complete("system", "user")
    assert requests == [{
        "model": "synthetic-siwc-model",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": 64,
        "thinking": {"type": "disabled"},
    }]
    assert safe_thinking_configuration(configured) == {
        "configured": True, "mode": "disabled"
    }


@pytest.mark.parametrize("local", [True, False])
def test_only_local_profile_disables_environment_proxy_and_both_refuse_redirects(
    monkeypatch, local
):
    real_client = httpx.Client
    options = []

    def capture_client(**kwargs):
        options.append(kwargs)
        return real_client(**kwargs)

    monkeypatch.setattr("app.provider.httpx.Client", capture_client)
    configured = settings() if local else settings(
        openai_base_url="https://public-provider.invalid/v1"
    )
    provider(lambda _: completion(), configured=configured).complete("s", "u")
    assert options[0]["follow_redirects"] is False
    assert options[0]["trust_env"] is (not local)


def test_gateway_generation_token_cap_is_not_claimed_but_usage_is_preserved():
    result = provider(lambda _: completion(completion_tokens=321)).complete("s", "u")
    # A gateway may ignore generation caps. Report actual usage so the caller's
    # budget ledger can charge/stop later calls, not an invented 64-token cap.
    assert result.completion_tokens == 321
    assert result.telemetry.completion_tokens == 321


@pytest.mark.parametrize("content", ["not JSON", '{"value":1,"value":2}', "NaN"])
def test_gateway_invalid_json_still_fails_before_return(content):
    with pytest.raises(ProviderRetryExhausted) as caught:
        provider(lambda _: completion(content)).complete("s", "u")
    assert caught.value.category == "content_json"


@pytest.mark.parametrize("content", ['{"records":"not a list"}', '{"records":[],"extra":1}'])
def test_gateway_generic_format_does_not_bypass_business_pydantic_contract(content):
    result = provider(lambda _: completion(content)).complete("s", "u")
    with pytest.raises(ValidationError):
        ExtractionEnvelope.model_validate_json(result.text)


def test_gateway_gzip_response_uses_existing_bounded_decompression():
    encoded = gzip.compress(completion().content)
    result = provider(lambda _: httpx.Response(
        200,
        headers={"Content-Encoding": "gzip"},
        stream=httpx.ByteStream(encoded),
    )).complete("s", "u")
    assert json.loads(result.text) == {"records": []}
    assert result.telemetry.received_bytes == len(encoded)


def test_gateway_timeout_remains_a_safe_provider_failure():
    def handler(request):
        raise httpx.ReadTimeout("private synthetic failure", request=request)

    with pytest.raises(ProviderRetryExhausted) as caught:
        provider(handler).complete("s", "u")
    assert caught.value.category == "read_timeout"
    assert "private" not in str(caught.value)


def test_gateway_redirect_is_never_followed():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(307, headers={"Location": "https://other.invalid/collect"})

    with pytest.raises(ProviderError) as caught:
        provider(handler).complete("s", "u")
    assert caught.value.category == "nonretry_http"
    assert len(requests) == 1


@pytest.mark.parametrize("choice", ["required", NamedToolChoice("read_span")])
def test_gateway_preserves_tool_choice_schema_and_validates_single_return(choice):
    payloads = []

    def handler(request):
        payloads.append(json.loads(request.content))
        return tool_response()

    definition = tool()
    result = provider(handler).complete_with_tools("s", "u", tools=[definition], tool_choice=choice)
    assert len(result.tool_calls) == 1
    payload = payloads[0]
    assert payload["parallel_tool_calls"] is False
    assert payload["tools"][0]["function"] == {
        "name": definition.name,
        "description": definition.description,
        "parameters": definition.parameters,
        "strict": False,
    }
    assert payload["tool_choice"] == (
        "required" if choice == "required"
        else {"type": "function", "function": {"name": "read_span"}}
    )
    assert not {"temperature", "max_tokens", "thinking", "response_format"} & payload.keys()


def test_gateway_rejects_multiple_tool_calls_even_if_caller_accepts_more_by_default():
    with pytest.raises(ProviderToolCallError) as caught:
        provider(lambda _: tool_response(count=2)).complete_with_tools("s", "u", tools=[tool()])
    assert caught.value.category == "tool_calls_too_many"


@pytest.mark.parametrize(
    ("response", "category"),
    [
        (tool_response(name="unknown"), "tool_call_unknown"),
        (tool_response(arguments="[]"), "tool_call_arguments_shape"),
        (tool_response(arguments='{"a":1,"a":2}'), "tool_call_arguments_json"),
    ],
)
def test_gateway_retains_native_tool_allowlist_and_strict_argument_checks(response, category):
    with pytest.raises(ProviderToolCallError) as caught:
        provider(lambda _: response).complete_with_tools("s", "u", tools=[tool()])
    assert caught.value.category == category


@pytest.mark.parametrize(
    "overrides",
    [
        {"local_model_gateway_enabled": False, "local_model_gateway_origins": ""},
        {"openai_base_url": "http://127.0.0.1:8318/v1"},
        {"openai_base_url": "http://localhost:8317/v1"},
        {"openai_base_url": "http://169.254.169.254:8317/v1"},
        {"provider_transport_allowed_origins": "http://127.0.0.1:8318"},
    ],
)
def test_unapproved_local_or_frozen_origin_makes_no_transport_request(overrides):
    calls = []
    configured = settings(**overrides)
    with pytest.raises(ProviderError) as caught:
        provider(lambda request: calls.append(request) or completion(), configured=configured).complete("s", "u")
    assert caught.value.category == "endpoint_rejected"
    assert calls == []


def test_stale_worker_cannot_reenable_gateway_in_production_via_model_copy():
    calls = []
    # model_copy bypasses constructor validators, as can a stale/mutated runtime.
    configured = settings().model_copy(update={"deployment_environment": "production"})
    with pytest.raises(ProviderError) as caught:
        provider(lambda request: calls.append(request) or completion(), configured=configured).complete("s", "u")
    assert caught.value.category == "endpoint_rejected"
    assert calls == []


def test_gateway_still_checks_frozen_endpoint_fingerprint_before_transport():
    calls = []
    configured = settings(
        provider_runtime_config_id="synthetic-config",
        provider_runtime_endpoint_sha256=provider_endpoint_sha256(f"{LOCAL_ORIGIN}/other"),
    )
    with pytest.raises(ProviderError) as caught:
        provider(lambda request: calls.append(request) or completion(), configured=configured).complete("s", "u")
    assert caught.value.category == "endpoint_rejected"
    assert calls == []
