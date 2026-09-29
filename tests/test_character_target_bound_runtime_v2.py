from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.runtime_provenance import safe_runtime_provenance
from scripts import run_character_axis_live as axis_runner
from scripts import run_evidence_investigator_live as investigator_runner


def _settings(**updates) -> Settings:
    values = {
        "loreguard_build_revision": "a" * 40,
        "openai_api_key": "runtime-test",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "provider_thinking_mode": "disabled",
        "enable_embeddings": True,
        "embedding_api_key": "embedding-runtime-test",
        "embedding_base_url": "https://embedding.invalid/v1",
        "embedding_model": "runtime-embedding",
        "embedding_model_revision": "revision-1",
        "embedding_deployment_fingerprint": "deployment-1",
        "embedding_dimensions": 8,
        "enable_evidence_investigator": True,
        "character_signal_full_line_prompt_v2": True,
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def test_target_bound_v2_requires_explicit_v1_source_boundary():
    with pytest.raises(ValidationError, match="requires draft actor review v1"):
        _settings(character_target_bound_draft_review_v2=True)


def test_target_bound_v2_runtime_bundle_is_atomic_and_versioned():
    off = safe_runtime_provenance(_settings())["character_consistency_limits"]
    assert off["target_bound_draft_review_v2"] is False
    assert off["target_bound_draft_review_schema_version"] is None
    assert off["target_bound_draft_review_batch_schema_version"] is None
    assert off["target_bound_draft_review_prompt_version"] is None
    assert off["target_bound_draft_review_signal_prompt_version"] is None
    assert off["target_bound_draft_review_clause_index_version"] is None

    on = safe_runtime_provenance(_settings(
        character_draft_actor_review_v1=True,
        character_target_bound_draft_review_v2=True,
    ))["character_consistency_limits"]
    assert on["target_bound_draft_review_v2"] is True
    assert on["target_bound_draft_review_schema_version"] == (
        "character-target-bound-draft-review-v2"
    )
    assert on["target_bound_draft_review_batch_schema_version"] == (
        "character-target-bound-draft-review-batch-v2"
    )
    assert on["target_bound_draft_review_prompt_version"] == (
        "character-target-bound-draft-review-prompt-v2"
    )
    assert on["target_bound_draft_review_signal_prompt_version"] == (
        "character-target-bound-draft-signal-prompt-v2"
    )
    assert on["target_bound_draft_review_clause_index_version"] == (
        "draft-actor-clause-index-v1"
    )


def test_target_bound_v2_runtime_bundle_round_trips_strict_consumers():
    off = safe_runtime_provenance(_settings())
    on = safe_runtime_provenance(_settings(
        character_draft_actor_review_v1=True,
        character_target_bound_draft_review_v2=True,
    ))
    assert investigator_runner._safe_runtime_provenance(off) == off
    assert investigator_runner._safe_runtime_provenance(on) == on
    assert axis_runner._safe_character_runtime_provenance(off) is not None
    assert axis_runner._safe_character_runtime_provenance(on) is not None

    for key in investigator_runner._CHARACTER_TARGET_BOUND_DRAFT_REVIEW_KEYS:
        malformed = {
            **on,
            "character_consistency_limits": {
                **on["character_consistency_limits"],
                key: None,
            },
        }
        assert investigator_runner._safe_runtime_provenance(malformed) is None
        assert axis_runner._safe_character_runtime_provenance(malformed) is None
