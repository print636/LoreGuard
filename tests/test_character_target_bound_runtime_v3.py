import pytest
from pydantic import ValidationError

from app.config import Settings
from app.runtime_provenance import safe_runtime_provenance
from scripts.run_character_consistency_live import (
    _TARGET_BOUND_DRAFT_REVIEW_V3,
    _safe_runtime_provenance,
    _target_bound_semantic_v3_ready,
)


def _settings(**updates):
    values = {
        "openai_api_key": "test",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock",
        "enable_character_consistency": True,
        "character_signal_full_line_prompt_v2": True,
        "character_draft_actor_review_v1": True,
        "character_target_bound_draft_review_v3": True,
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def test_v3_is_explicit_exclusive_and_requires_v1_boundary():
    with pytest.raises(ValidationError, match="requires draft actor review v1"):
        _settings(
            character_draft_actor_review_v1=False,
            character_target_bound_draft_review_v3=True,
        )
    with pytest.raises(ValidationError, match="versions are exclusive"):
        _settings(character_target_bound_draft_review_v2=True)


def test_v3_runtime_identity_round_trips_strict_consumer():
    provenance = safe_runtime_provenance(_settings())
    limits = provenance["character_consistency_limits"]
    assert limits["target_bound_draft_review_v2"] is False
    assert limits["target_bound_draft_review_v3"] is True
    assert all(
        limits[field] == expected
        for field, expected in _TARGET_BOUND_DRAFT_REVIEW_V3.items()
    )
    assert limits["target_bound_draft_review_prompt_version"] == (
        "character-target-bound-draft-review-prompt-v9"
    )
    assert _target_bound_semantic_v3_ready(limits)
    assert _safe_runtime_provenance(provenance) == provenance


@pytest.mark.parametrize(
    "field",
    (
        "target_bound_draft_review_schema_version",
        "target_bound_draft_review_batch_schema_version",
        "target_bound_draft_review_prompt_version",
        "primary_scoped_claim_review_prompt_version",
        "primary_scoped_object_review_prompt_version",
        "scoped_draft_direction_version",
        "scoped_draft_direction_guide_sha256",
        "target_bound_draft_review_signal_prompt_version",
        "target_bound_draft_review_clause_index_version",
    ),
)
def test_v3_runtime_identity_tampering_is_rejected(field):
    provenance = safe_runtime_provenance(_settings())
    provenance["character_consistency_limits"][field] = "unknown"
    assert _safe_runtime_provenance(provenance) is None


@pytest.mark.parametrize("previous_version", (9, 10, 11))
def test_older_receipt_remains_readable_but_cannot_pass_current_live_preflight(previous_version):
    provenance = safe_runtime_provenance(_settings())
    limits = provenance["character_consistency_limits"]
    del limits["scoped_draft_direction_version"]
    del limits["scoped_draft_direction_guide_sha256"]
    if previous_version < 11:
        del limits["primary_scoped_object_review_prompt_version"]
    if previous_version < 10:
        del limits["primary_scoped_claim_review_prompt_version"]
    assert _safe_runtime_provenance(provenance) == provenance
    assert not _target_bound_semantic_v3_ready(limits)


def test_object_review_cannot_be_advertised_without_claim_verification():
    provenance = safe_runtime_provenance(_settings())
    del provenance["character_consistency_limits"]["primary_scoped_claim_review_prompt_version"]
    assert _safe_runtime_provenance(provenance) is None
