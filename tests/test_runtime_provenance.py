from __future__ import annotations

import hashlib
import json
import re

import scripts.run_evidence_investigator_live as live_runner

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
        "endpoint_configuration_sha256": result["chat_provider"][
            "endpoint_configuration_sha256"
        ],
        "temperature": 0,
        "thinking_configured": True,
        "thinking_mode": "disabled",
    }
    assert re.fullmatch(
        r"[a-f0-9]{64}",
        result["chat_provider"]["endpoint_configuration_sha256"],
    )
    assert result["investigator_limits"]["provider_call_timeout_seconds"] == 60
    assert result["investigator_limits"]["total_deadline_seconds"] == 90
    assert result["investigator_limits"]["max_completion_tokens"] == 1024
    assert result["investigator_limits"]["max_agent_input_bytes"] == 131072
    assert result["character_consistency_limits"][
        "signal_provider_max_attempts"
    ] == 2
    assert result["character_consistency_limits"][
        "signal_package_max_attempts"
    ] == 2
    assert result["character_consistency_limits"]["signal_token_budget"] == 22_000
    assert result["character_consistency_limits"][
        "signal_max_completion_tokens"
    ] == 4_096
    assert result["character_consistency_limits"]["signal_max_records"] == 48
    assert result["character_consistency_limits"]["signal_full_line_echo_v2"] is False
    assert result["character_consistency_limits"]["signal_core_scope_v3"] is False
    assert result["character_consistency_limits"]["signal_support_id_v4"] is False
    assert result["character_consistency_limits"][
        "signal_support_segmenter_version"
    ] is None
    assert result["character_consistency_limits"][
        "signal_targeted_max_targets_per_chunk"
    ] == 12
    assert result["character_consistency_limits"][
        "signal_provider_max_completion_tokens"
    ] == 1_024
    assert result["character_consistency_limits"][
        "signal_provider_max_response_bytes"
    ] == 64_000
    assert result["character_consistency_limits"][
        "drift_provider_max_attempts"
    ] == 2
    assert result["character_consistency_limits"]["drift_token_budget"] == 4_000
    assert result["character_consistency_limits"][
        "drift_provider_max_completion_tokens"
    ] == 1_000
    assert result["character_consistency_limits"][
        "signal_total_deadline_seconds"
    ] == 60
    assert result["character_consistency_limits"][
        "drift_total_deadline_seconds"
    ] == 60
    assert re.fullmatch(r"[a-f0-9]{64}", result["rag"]["profile_fingerprint"])
    assert re.fullmatch(r"[a-f0-9]{64}", result["rag"]["chunker_fingerprint"])
    assert "test-only-secret" not in serialized
    assert "https://" not in serialized
    assert "/v1" not in serialized
    assert "relay.example" not in serialized
    assert live_runner._safe_runtime_provenance(result) == result


def _character_limits_digest(settings: Settings) -> str:
    limits = safe_runtime_provenance(settings)["character_consistency_limits"]
    encoded = json.dumps(
        limits, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def test_character_runtime_fingerprint_tracks_stage_and_effective_provider_limits():
    baseline = configured_settings()
    baseline_limits = safe_runtime_provenance(baseline)["character_consistency_limits"]
    baseline_digest = _character_limits_digest(baseline)
    variants = (
        {"character_signal_full_line_prompt_v2": True},
        {
            "character_signal_full_line_prompt_v2": True,
            "character_signal_core_scope_prompt_v3": True,
        },
        {
            "character_signal_full_line_prompt_v2": True,
            "character_signal_support_id_v4": True,
        },
        {"character_signal_max_records": 47},
        {"character_signal_targeted_max_targets_per_chunk": 11},
        {"character_signal_timeout_seconds": 20},
        {"character_signal_max_response_bytes": 32_000},
        {"character_drift_max_evidence_chars": 7_000},
        {"character_drift_token_budget": 3_000},
        {"character_drift_timeout_seconds": 20},
        {"character_drift_max_completion_tokens": 900},
        {"character_drift_max_response_bytes": 16_000},
        {"provider_total_deadline_seconds": 20},
        {"provider_max_completion_tokens": 512},
        {"provider_max_response_bytes": 16_000},
        {"per_run_token_budget": 90_000},
        {"daily_token_budget": 200_000},
    )
    for override in variants:
        assert _character_limits_digest(configured_settings(**override)) != baseline_digest, override

    constrained = safe_runtime_provenance(configured_settings(
        provider_total_deadline_seconds=12,
        provider_max_completion_tokens=512,
        provider_max_response_bytes=1_024,
    ))["character_consistency_limits"]
    assert constrained["signal_total_deadline_seconds"] == 12.0
    assert constrained["drift_total_deadline_seconds"] == 12.0
    assert constrained["signal_provider_timeout_seconds"] == 12.0
    assert constrained["drift_provider_timeout_seconds"] == 12.0
    assert constrained["signal_provider_max_completion_tokens"] == 512
    assert constrained["drift_provider_max_completion_tokens"] == 512
    assert constrained["signal_provider_max_response_bytes"] == 1_024
    assert constrained["drift_provider_max_response_bytes"] == 1_024
    assert type(baseline_limits["signal_timeout_seconds"]) is float
    assert type(baseline_limits["drift_timeout_seconds"]) is float
    assert type(baseline_limits["per_run_token_budget"]) is int
    assert type(baseline_limits["daily_token_budget"]) is int


def test_health_api_provenance_exposes_only_safe_core_scope_flag(monkeypatch):
    from app import main

    configured = configured_settings(
        character_signal_full_line_prompt_v2=True,
        character_signal_core_scope_prompt_v3=True,
    )
    monkeypatch.setattr(main, "settings", configured)

    payload = main.health()
    limits = payload["runtime_provenance"]["character_consistency_limits"]
    assert limits["signal_full_line_echo_v2"] is True
    assert limits["signal_core_scope_v3"] is True
    assert live_runner._safe_runtime_provenance(payload["runtime_provenance"]) == payload[
        "runtime_provenance"
    ]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "character_signal_core_scope_prompt_v3" not in serialized
    assert "test-only-secret" not in serialized
    assert "https://" not in serialized


def test_health_api_provenance_exposes_versioned_support_id_without_prompt_or_source(
    monkeypatch,
):
    from app import main

    configured = configured_settings(
        character_signal_full_line_prompt_v2=True,
        character_signal_support_id_v4=True,
    )
    monkeypatch.setattr(main, "settings", configured)
    payload = main.health()
    limits = payload["runtime_provenance"]["character_consistency_limits"]

    assert limits["signal_full_line_echo_v2"] is True
    assert limits["signal_support_id_v4"] is True
    assert limits["signal_support_segmenter_version"] == "assertion-index-v1"
    assert live_runner._safe_runtime_provenance(payload["runtime_provenance"]) == payload[
        "runtime_provenance"
    ]
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "character_signal_support_id_v4" not in serialized
    assert "unit-test-placeholder" not in serialized
    assert "https://" not in serialized


def test_runtime_provenance_support_trace_is_versioned_and_requires_v4():
    off = safe_runtime_provenance(configured_settings())
    limits = off["character_consistency_limits"]
    assert limits["signal_support_trace_v1"] is False
    assert limits["signal_support_trace_version"] is None
    assert live_runner._safe_runtime_provenance(off) == off

    enabled = safe_runtime_provenance(configured_settings(
        character_signal_full_line_prompt_v2=True,
        character_signal_support_id_v4=True,
        character_signal_support_trace_v1=True,
    ))
    assert enabled["character_consistency_limits"]["signal_support_trace_v1"] is True
    assert enabled["character_consistency_limits"]["signal_support_trace_version"] == "support-trace-v1"
    assert live_runner._safe_runtime_provenance(enabled) == enabled
    for mutation in (
        {"signal_support_trace_v1": "false"},
        {"signal_support_trace_v1": True, "signal_support_id_v4": False},
        {"signal_support_trace_version": "untrusted"},
    ):
        tampered = json.loads(json.dumps(enabled))
        tampered["character_consistency_limits"].update(mutation)
        assert live_runner._safe_runtime_provenance(tampered) is None
    legacy = json.loads(json.dumps(enabled))
    del legacy["character_consistency_limits"]["signal_support_trace_v1"]
    del legacy["character_consistency_limits"]["signal_support_trace_version"]
    assert live_runner._safe_runtime_provenance(legacy) == legacy


def test_runtime_provenance_marks_unversioned_build_without_inventing_revision():
    result = safe_runtime_provenance(
        configured_settings(loreguard_build_revision="")
    )

    assert result["build"]["git_revision"] is None
    assert service_artifact_sha256() == result["build"]["service_artifact_sha256"]


def test_service_and_runner_independently_hash_the_same_source_bundle():
    assert service_artifact_sha256() == live_runner._local_service_artifact_sha256()
