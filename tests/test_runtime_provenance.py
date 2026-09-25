from __future__ import annotations

import hashlib
import json
import re

import pytest

import scripts.run_evidence_investigator_live as live_runner
import scripts.run_character_axis_live as axis_live

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
    assert result["character_consistency_limits"]["signal_semantic_scope_v5"] is False
    assert result["character_consistency_limits"]["signal_semantic_scope_version"] is None
    assert result["character_consistency_limits"]["signal_scope_review_v1"] is False
    assert result["character_consistency_limits"][
        "signal_scope_review_schema_version"
    ] is None
    assert result["character_consistency_limits"][
        "signal_scope_review_prompt_version"
    ] is None
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
        {
            "character_signal_full_line_prompt_v2": True,
            "character_signal_support_id_v4": True,
            "character_signal_semantic_scope_v5": True,
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


def test_v5_runtime_identity_and_both_runner_boundaries_are_versioned_and_strict():
    disabled = safe_runtime_provenance(configured_settings())
    assert live_runner._safe_runtime_provenance(disabled) == disabled
    assert axis_live._safe_character_runtime_provenance(disabled) is not None
    assert axis_live._runtime_summary({"runtime_provenance": disabled})[
        "signal_semantic_scope_v5"
    ] is False

    base = {
        "character_signal_full_line_prompt_v2": True,
        "character_signal_core_scope_prompt_v3": True,
        "character_signal_support_id_v4": True,
    }
    v4 = safe_runtime_provenance(configured_settings(**base))
    v5 = safe_runtime_provenance(configured_settings(
        **base, character_signal_semantic_scope_v5=True,
    ))
    v5_limits = v5["character_consistency_limits"]
    assert v5_limits["signal_support_id_v4"] is True
    assert v5_limits["signal_semantic_scope_v5"] is True
    assert v5_limits["signal_semantic_scope_version"] == "semantic-scope-v5"
    assert live_runner._safe_runtime_provenance(v5) == v5
    assert axis_live._safe_character_runtime_provenance(v5) is not None
    assert axis_live._runtime_provenance_digest(v5) != axis_live._runtime_provenance_digest(v4)
    assert axis_live._runtime_summary({"runtime_provenance": v4})[
        "signal_semantic_scope_v5"
    ] is False
    summary = axis_live._runtime_summary({"runtime_provenance": v5})
    assert summary["signal_semantic_scope_v5"] is True
    assert summary["signal_semantic_scope_version"] == "semantic-scope-v5"
    assert "character_signal_semantic_scope_v5" not in json.dumps(v5)
    assert "test-only-secret" not in json.dumps(v5)

    legacy = json.loads(json.dumps(v4))
    del legacy["character_consistency_limits"]["signal_semantic_scope_v5"]
    del legacy["character_consistency_limits"]["signal_semantic_scope_version"]
    for key in live_runner._CHARACTER_SIGNAL_SCOPE_REVIEW_KEYS:
        del legacy["character_consistency_limits"][key]
    assert live_runner._safe_runtime_provenance(legacy) == legacy
    assert axis_live._safe_character_runtime_provenance(legacy) is not None
    assert axis_live._runtime_summary({"runtime_provenance": legacy})[
        "signal_semantic_scope_v5"
    ] is None

    traced = safe_runtime_provenance(configured_settings(
        **base, character_signal_semantic_scope_v5=True,
        character_signal_support_trace_v1=True,
    ))
    assert live_runner._safe_runtime_provenance(traced) == traced
    assert axis_live._safe_character_runtime_provenance(traced) is not None

    mutations = (
        {"signal_semantic_scope_v5": "true"},
        {"signal_semantic_scope_version": "unknown"},
        {"signal_semantic_scope_v5": False},
        {"signal_support_id_v4": False, "signal_support_segmenter_version": None},
        {"signal_full_line_echo_v2": False},
        {"unexpected_source_text": "private"},
    )
    for changes in mutations:
        malformed = json.loads(json.dumps(v5))
        malformed["character_consistency_limits"].update(changes)
        assert live_runner._safe_runtime_provenance(malformed) is None, changes
        assert axis_live._safe_character_runtime_provenance(malformed) is None, changes
    for missing in ("signal_semantic_scope_v5", "signal_semantic_scope_version"):
        malformed = json.loads(json.dumps(v5))
        del malformed["character_consistency_limits"][missing]
        assert live_runner._safe_runtime_provenance(malformed) is None, missing
        assert axis_live._safe_character_runtime_provenance(malformed) is None, missing


def test_scope_review_runtime_identity_is_bounded_versioned_and_back_compatible():
    base = {
        "character_signal_full_line_prompt_v2": True,
        "character_signal_support_id_v4": True,
        "character_signal_semantic_scope_v5": True,
    }
    off = safe_runtime_provenance(configured_settings(**base))
    on = safe_runtime_provenance(configured_settings(
        **base,
        character_signal_scope_review_v1=True,
        provider_timeout_seconds=7,
        provider_total_deadline_seconds=20,
        provider_max_response_bytes=8_192,
    ))
    limits = on["character_consistency_limits"]
    assert limits["signal_scope_review_v1"] is True
    assert limits["signal_scope_review_schema_version"] == "character-scope-review-v1"
    assert limits["signal_scope_review_prompt_version"] == "character-scope-review-prompt-v2"
    assert limits["signal_scope_review_token_reserve"] == 6_000
    assert limits["signal_scope_review_completion_tokens"] == 2_048
    assert limits["signal_scope_review_max_response_bytes"] == 32_768
    assert limits["signal_scope_review_max_attempts"] == 1
    assert limits["signal_scope_review_provider_timeout_seconds"] == 7
    assert limits["signal_scope_review_provider_total_deadline_seconds"] == 20
    assert limits["signal_scope_review_provider_max_completion_tokens"] == 1_024
    assert limits["signal_scope_review_provider_max_response_bytes"] == 8_192
    assert limits["signal_scope_review_provider_max_attempts"] == 1
    assert live_runner._safe_runtime_provenance(on) == on
    assert axis_live._safe_character_runtime_provenance(on) is not None
    assert axis_live._runtime_provenance_digest(on) != axis_live._runtime_provenance_digest(off)
    summary = axis_live._runtime_summary({"runtime_provenance": on})
    assert summary["signal_scope_review_v1"] is True
    assert summary["signal_scope_review_schema_version"] == "character-scope-review-v1"
    assert summary["signal_scope_review_prompt_version"] == "character-scope-review-prompt-v2"
    assert summary["signal_scope_review_limits"][
        "signal_scope_review_provider_max_completion_tokens"
    ] == 1_024
    assert "test-only-secret" not in json.dumps(summary)
    assert "https://" not in json.dumps(summary)

    old_v1_report = json.loads(json.dumps(on))
    old_v1_limits = old_v1_report["character_consistency_limits"]
    old_v1_limits["signal_scope_review_prompt_version"] = "character-scope-review-prompt-v1"
    assert live_runner._safe_runtime_provenance(old_v1_report) == old_v1_report
    old_v1_axis = axis_live._safe_character_runtime_provenance(old_v1_report)
    assert old_v1_axis is not None
    assert old_v1_axis["character_consistency_limits"] == old_v1_limits
    assert axis_live._runtime_summary({"runtime_provenance": old_v1_report})[
        "signal_scope_review_prompt_version"
    ] == "character-scope-review-prompt-v1"
    assert not live_runner._valid_character_scope_review_limits(old_v1_limits)
    assert live_runner._valid_character_scope_review_limits(
        old_v1_limits, allow_legacy_prompt_version=True,
    )
    with pytest.raises(axis_live.SafeFailure) as exc:
        axis_live._service_preflight_gate(
            {"runtime_provenance": old_v1_report},
            {"git_head": old_v1_report["build"]["git_revision"]},
            old_v1_report["build"]["service_artifact_sha256"],
        )
    assert exc.value.payload["code"] == "runtime_provenance_invalid"

    legacy = json.loads(json.dumps(off))
    for key in live_runner._CHARACTER_SIGNAL_SCOPE_REVIEW_KEYS:
        del legacy["character_consistency_limits"][key]
    assert live_runner._safe_runtime_provenance(legacy) == legacy
    assert axis_live._safe_character_runtime_provenance(legacy) is not None
    assert axis_live._runtime_summary({"runtime_provenance": legacy})[
        "signal_scope_review_v1"
    ] is None

    traced = safe_runtime_provenance(configured_settings(
        **base, character_signal_scope_review_v1=True,
        character_signal_support_trace_v1=True,
    ))
    assert live_runner._safe_runtime_provenance(traced) == traced
    assert axis_live._safe_character_runtime_provenance(traced) is not None

    mutations = (
        {"signal_scope_review_v1": "true"},
        {"signal_scope_review_v1": False},
        {"signal_scope_review_schema_version": "unknown"},
        {"signal_scope_review_prompt_version": None},
        {"signal_scope_review_prompt_version": "character-scope-review-prompt-v0"},
        {"signal_scope_review_prompt_version": ["character-scope-review-prompt-v2"]},
        {"signal_semantic_scope_v5": False, "signal_semantic_scope_version": None},
        {"signal_scope_review_completion_tokens": True},
        {"signal_scope_review_provider_max_completion_tokens": 1_025},
        {"signal_scope_review_provider_timeout_seconds": 31},
        {"source_text": "private"},
    )
    for changes in mutations:
        malformed = json.loads(json.dumps(on))
        malformed["character_consistency_limits"].update(changes)
        assert live_runner._safe_runtime_provenance(malformed) is None, changes
        assert axis_live._safe_character_runtime_provenance(malformed) is None, changes
    for missing in live_runner._CHARACTER_SIGNAL_SCOPE_REVIEW_KEYS:
        malformed = json.loads(json.dumps(on))
        del malformed["character_consistency_limits"][missing]
        assert live_runner._safe_runtime_provenance(malformed) is None, missing
        assert axis_live._safe_character_runtime_provenance(malformed) is None, missing


def test_runtime_provenance_marks_unversioned_build_without_inventing_revision():
    result = safe_runtime_provenance(
        configured_settings(loreguard_build_revision="")
    )

    assert result["build"]["git_revision"] is None
    assert service_artifact_sha256() == result["build"]["service_artifact_sha256"]


def test_service_and_runner_independently_hash_the_same_source_bundle():
    assert service_artifact_sha256() == live_runner._local_service_artifact_sha256()
