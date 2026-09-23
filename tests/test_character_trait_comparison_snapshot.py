from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.character_traits import (
    _validated_comparison_key,
    candidate_snapshot_payload,
    validate_character_trait_supersession,
)
from app.narrative_context import canonical_scope_payload, payload_sha256
from app.service import _historical_trait_snapshot_payload


def _candidate(*, comparison_key: str | None = None) -> SimpleNamespace:
    scope = canonical_scope_payload({"schema_version": 1, "timeline_key": "main"})
    evidence = [{"document_id": "document-1", "text": "林澈喜欢蜜瓜。"}]
    return SimpleNamespace(
        id="candidate-1",
        review_state="confirmed",
        character_key="林澈",
        character_display_name="林澈",
        trait_type="preference",
        trait_key="食物偏好",
        comparison_key=comparison_key,
        value="喜欢蜜瓜",
        polarity="positive",
        stability="stable",
        contexts=["日常饮食"],
        origin="explicit_setting",
        authority_tier="formal_record",
        scope_payload=scope,
        scope_sha256=payload_sha256(scope),
        valid_from_release_ordinal=1,
        valid_until_release_ordinal=None,
        evidence=evidence,
        evidence_sha256=payload_sha256(evidence),
        candidate_fingerprint="f" * 64,
        generator_version="test-v1",
        lock_version=1,
    )


def test_new_and_legacy_confirmed_snapshot_contracts_keep_distinct_hashes() -> None:
    candidate = _candidate()
    review = SimpleNamespace(id="review-1", decision="confirm")
    legacy = candidate_snapshot_payload(candidate, review)

    assert legacy == {
        "schema_version": 1,
        "candidate_id": "candidate-1",
        "confirmation_review_id": "review-1",
        "character_key": "林澈",
        "character_display_name": "林澈",
        "trait_type": "preference",
        "trait_key": "食物偏好",
        "value": "喜欢蜜瓜",
        "polarity": "positive",
        "stability": "stable",
        "contexts": ["日常饮食"],
        "origin": "explicit_setting",
        "authority_tier": "formal_record",
        "scope": candidate.scope_payload,
        "scope_sha256": candidate.scope_sha256,
        "valid_from_release_ordinal": 1,
        "valid_until_release_ordinal": None,
        "evidence": candidate.evidence,
        "evidence_sha256": candidate.evidence_sha256,
        "candidate_fingerprint": "f" * 64,
        "generator_version": "test-v1",
        "candidate_lock_version": 1,
    }
    legacy_hash = payload_sha256(legacy)
    assert legacy_hash == "5ab0d862bf95f8a33aaaaa56c845fdc06efd659fc69d21b0036959ab19c9e0dc"

    candidate.comparison_key = "preference:蜜瓜"
    current = candidate_snapshot_payload(candidate, review)
    assert current == {**legacy, "comparison_key": "preference:蜜瓜"}
    assert payload_sha256(current) != legacy_hash
    assert payload_sha256(legacy) == legacy_hash


def test_historical_snapshot_preserves_stored_identity_and_legacy_shape() -> None:
    candidate = _candidate()
    candidate.review_state = "superseded"
    candidate.valid_until_release_ordinal = 5
    review = SimpleNamespace(id="review-1", decision="confirm")

    legacy = _historical_trait_snapshot_payload(candidate, review)
    assert "comparison_key" not in legacy
    legacy_hash = payload_sha256(legacy)
    assert legacy_hash == "418bbf9919c9a6d42643ecfbd9f87a551ae0e566577cb2aa65f7f8120151fdd8"

    candidate.comparison_key = "preference:蜜瓜"
    current = _historical_trait_snapshot_payload(candidate, review)
    assert current == {**legacy, "comparison_key": "preference:蜜瓜"}
    assert payload_sha256(current) != legacy_hash
    assert payload_sha256(legacy) == legacy_hash


def test_object_supersession_rejects_distinct_known_keys_and_allows_legacy_link() -> None:
    original = _candidate(comparison_key="preference:蜜瓜")
    original.project_id = "project-1"
    original.scope_payload = canonical_scope_payload(
        {"timeline_key": "main", "release": {"key": "v1", "ordinal": 1}}
    )

    def validate(new_key: str | None) -> None:
        validate_character_trait_supersession(
            project_id="project-1",
            character_key="林澈",
            trait_type="preference",
            trait_key="食物偏好",
            comparison_key=new_key,
            origin=original.origin,
            authority_tier=original.authority_tier,
            scope=original.scope_payload,
            valid_from_release_ordinal=1,
            valid_until_release_ordinal=None,
            superseded=original,
        )

    validate("preference:蜜瓜")
    with pytest.raises(ValueError, match="superseded candidate is incompatible"):
        validate("preference:葡萄")
    validate(None)
    validate_character_trait_supersession(
        project_id="project-1",
        character_key="林澈",
        trait_type="preference",
        trait_key="melon_preference",
        comparison_key="preference:蜜瓜",
        origin=original.origin,
        authority_tier=original.authority_tier,
        scope=original.scope_payload,
        valid_from_release_ordinal=1,
        valid_until_release_ordinal=None,
        superseded=original,
    )
    original.comparison_key = None
    validate("preference:蜜瓜")
    with pytest.raises(ValueError, match="superseded candidate is incompatible"):
        validate_character_trait_supersession(
            project_id="project-1",
            character_key="林澈",
            trait_type="preference",
            trait_key="melon_preference",
            comparison_key="preference:蜜瓜",
            origin=original.origin,
            authority_tier=original.authority_tier,
            scope=original.scope_payload,
            valid_from_release_ordinal=1,
            valid_until_release_ordinal=None,
            superseded=original,
        )
    original.comparison_key = "preference:蜜瓜\u202e"
    validate("preference:蜜瓜")

    original.trait_type = "core_personality"
    original.trait_key = "社交主动性"
    validate_character_trait_supersession(
        project_id="project-1",
        character_key="林澈",
        trait_type="core_personality",
        trait_key="社交主动性",
        comparison_key=None,
        origin=original.origin,
        authority_tier=original.authority_tier,
        scope=original.scope_payload,
        valid_from_release_ordinal=1,
        valid_until_release_ordinal=None,
        superseded=original,
    )


def test_nonpreference_legacy_supersession_keeps_old_label_rule() -> None:
    original = _candidate(comparison_key=None)
    original.project_id = "project-1"
    original.trait_type = "value"
    original.trait_key = "companion_trust"
    original.scope_payload = canonical_scope_payload(
        {"timeline_key": "main", "release": {"key": "v1", "ordinal": 1}}
    )

    def validate(trait_key: str) -> None:
        validate_character_trait_supersession(
            project_id="project-1",
            character_key="林澈",
            trait_type="value",
            trait_key=trait_key,
            comparison_key="value:同伴",
            origin=original.origin,
            authority_tier=original.authority_tier,
            scope=original.scope_payload,
            valid_from_release_ordinal=1,
            valid_until_release_ordinal=None,
            superseded=original,
        )

    validate("companion_trust")
    with pytest.raises(ValueError, match="superseded candidate is incompatible"):
        validate("companion_protection")


@pytest.mark.parametrize(
    ("old_tier", "old_origin", "new_tier", "new_origin", "allowed"),
    [
        ("core_canon", "explicit_setting", "formal_record", "explicit_setting", False),
        ("formal_record", "explicit_setting", "formal_record", "history_inference", False),
        ("formal_record", "history_inference", "formal_record", "explicit_setting", True),
        ("formal_record", "explicit_setting", "formal_record", "explicit_setting", True),
        ("core_canon", "explicit_setting", "core_canon", "explicit_setting", True),
    ],
)
def test_supersession_preserves_authority_order(
    old_tier: str,
    old_origin: str,
    new_tier: str,
    new_origin: str,
    allowed: bool,
) -> None:
    original = _candidate(comparison_key=None)
    original.project_id = "project-1"
    original.authority_tier = old_tier
    original.origin = old_origin

    def validate() -> None:
        validate_character_trait_supersession(
            project_id="project-1",
            character_key="林澈",
            trait_type="preference",
            trait_key="食物偏好",
            comparison_key="preference:蜜瓜",
            origin=new_origin,
            authority_tier=new_tier,
            scope=original.scope_payload,
            valid_from_release_ordinal=1,
            valid_until_release_ordinal=None,
            superseded=original,
        )

    if allowed:
        validate()
    else:
        with pytest.raises(ValueError, match="superseded candidate is incompatible"):
            validate()


@pytest.mark.parametrize(
    "comparison_key",
    [
        "preference:",
        "preference:蜜瓜:葡萄",
        "preference:蜜 瓜",
        "preference:蜜瓜\nsecret",
        "preference:蜜瓜\u202e",
        "value:蜜瓜",
        "preference:" + "a" * 190,
    ],
)
def test_invalid_or_oversized_comparison_key_is_not_frozen(
    comparison_key: str,
) -> None:
    with pytest.raises(ValueError, match="comparison key is invalid"):
        _validated_comparison_key(comparison_key, trait_type="preference")
    snapshot = candidate_snapshot_payload(
        _candidate(comparison_key=comparison_key),
        SimpleNamespace(id="review-1", decision="confirm"),
    )
    assert "comparison_key" not in snapshot
