from __future__ import annotations

import json
import re

from app.config import Settings
from app.runtime_provenance import (
    RUNTIME_PROVENANCE_SCHEMA,
    safe_runtime_provenance,
    service_artifact_sha256,
)


def configured_settings(**overrides) -> Settings:
    values = {
        "loreguard_build_revision": "a" * 40,
        "openai_api_key": "test-only-secret",
        "openai_base_url": "https://Relay.Example:443/v1/",
        "openai_model": "fixture-model",
        "provider_thinking_mode": "disabled",
        "provider_timeout_seconds": 90,
        "provider_total_deadline_seconds": 90,
        "provider_max_completion_tokens": 1024,
        "enable_embeddings": True,
        "embedding_api_key": "embedding-test-only-secret",
        "embedding_base_url": "https://embedding.example/v1",
        "embedding_model": "fixture-embedding",
        "embedding_model_revision": "revision-1",
        "embedding_deployment_fingerprint": "deployment-1",
        "embedding_dimensions": 8,
        "enable_evidence_investigator": True,
        "evidence_investigator_timeout_seconds": 60,
        "evidence_investigator_total_deadline_seconds": 180,
        "evidence_investigator_max_completion_tokens": 2048,
        "evidence_investigator_token_budget": 20000,
    }
    values.update(overrides)
    return Settings(**values)


def test_runtime_provenance_is_content_free_and_records_effective_identity():
    result = safe_runtime_provenance(configured_settings())
    serialized = json.dumps(result, sort_keys=True)

    assert result["schema_version"] == RUNTIME_PROVENANCE_SCHEMA
    assert result["build"]["git_revision"] == "a" * 40
    assert re.fullmatch(
        r"[a-f0-9]{64}", result["build"]["service_artifact_sha256"]
    )
    assert result["chat_provider"] == {
        "model_alias": "fixture-model",
        "relay_hostname": "relay.example",
        "relay_configuration_sha256": result["chat_provider"][
            "relay_configuration_sha256"
        ],
        "temperature": 0,
        "thinking_configured": True,
        "thinking_mode": "disabled",
    }
    assert re.fullmatch(
        r"[a-f0-9]{64}",
        result["chat_provider"]["relay_configuration_sha256"],
    )
    assert result["investigator_limits"]["provider_call_timeout_seconds"] == 60
    assert result["investigator_limits"]["total_deadline_seconds"] == 90
    assert result["investigator_limits"]["max_completion_tokens"] == 1024
    assert result["investigator_limits"]["max_agent_input_bytes"] == 131072
    assert re.fullmatch(r"[a-f0-9]{64}", result["rag"]["profile_fingerprint"])
    assert re.fullmatch(r"[a-f0-9]{64}", result["rag"]["chunker_fingerprint"])
    assert "test-only-secret" not in serialized
    assert "https://" not in serialized
    assert "/v1" not in serialized


def test_runtime_provenance_marks_unversioned_build_without_inventing_revision():
    result = safe_runtime_provenance(
        configured_settings(loreguard_build_revision="")
    )

    assert result["build"]["git_revision"] is None
    assert service_artifact_sha256() == result["build"]["service_artifact_sha256"]
