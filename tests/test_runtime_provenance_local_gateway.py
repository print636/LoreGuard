from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.provider_credentials import provider_endpoint_sha256
from app.runtime_provenance import safe_runtime_provenance


def settings(**overrides) -> Settings:
    values = {
        "_env_file": None,
        "deployment_environment": "local",
        "auth_mode": "anonymous",
        "openai_base_url": "http://127.0.0.1:8317/v1/",
        "openai_api_key": "synthetic-provenance-private-key",
        "openai_model": "synthetic-siwc-model",
        "local_model_gateway_enabled": True,
        "local_model_gateway_origins": "http://127.0.0.1:8317",
        "provider_max_completion_tokens": 512,
        "provider_thinking_mode": "disabled",
        "enable_embeddings": False,
    }
    values.update(overrides)
    return Settings(**values)


@pytest.mark.parametrize("host", ["127.0.0.1", "host.docker.internal"])
def test_authorized_local_provenance_records_wire_limits_honestly_without_secrets(host):
    configured = settings(
        openai_base_url=f"http://{host}:8317/v1/",
        local_model_gateway_origins=f"http://{host}:8317",
    )
    provenance = safe_runtime_provenance(configured)
    chat = provenance["chat_provider"]
    assert chat == {
        "model_alias": "synthetic-siwc-model",
        "endpoint_configuration_sha256": provider_endpoint_sha256(configured.openai_base_url.rstrip("/")),
        "temperature": None,
        "thinking_configured": False,
        "thinking_mode": None,
        "request_profile": "cli_proxy_siwc",
        "upstream_generation_token_limit_supported": False,
        "response_format": "json_schema",
        "response_schema_strict": False,
        "completion_limit_semantics": "local_admission_budget_not_upstream_cap",
    }
    assert len(chat["endpoint_configuration_sha256"]) == 64
    assert provenance["investigator_limits"]["max_completion_tokens"] == 512
    assert provenance["character_consistency_limits"]["signal_provider_max_completion_tokens"] == 512
    serialized = json.dumps(provenance)
    for forbidden in (host, configured.openai_base_url, configured.openai_api_key, "http://"):
        assert forbidden not in serialized


def test_ordinary_https_provenance_keeps_exact_legacy_provider_shape_and_limits():
    configured = settings(openai_base_url="https://ordinary.example/v1")
    provenance = safe_runtime_provenance(configured)
    assert provenance["chat_provider"] == {
        "model_alias": "synthetic-siwc-model",
        "endpoint_configuration_sha256": provider_endpoint_sha256(configured.openai_base_url),
        "temperature": 0,
        "thinking_configured": True,
        "thinking_mode": "disabled",
    }
    local = safe_runtime_provenance(settings())
    assert provenance["investigator_limits"] == local["investigator_limits"]
    assert provenance["character_consistency_limits"] == local["character_consistency_limits"]


@pytest.mark.parametrize(
    "changes",
    [
        {"local_model_gateway_enabled": False, "local_model_gateway_origins": ""},
        {"openai_base_url": "http://127.0.0.1:8318/v1"},
        {"deployment_environment": "production"},
    ],
)
def test_unapproved_or_stale_policy_never_claims_the_local_gateway_profile(changes):
    configured = settings().model_copy(update=changes)
    chat = safe_runtime_provenance(configured)["chat_provider"]
    assert "request_profile" not in chat
    assert "completion_limit_semantics" not in chat
