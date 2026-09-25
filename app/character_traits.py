from __future__ import annotations

import re
import unicodedata
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from .character_trait_extraction import stable_trait_identity
from .character_support_bindings import (
    TraitSupportRef,
    bind_support_refs,
    support_bindings_sha256,
    verify_stored_support_bindings,
)
from .db import (
    AnalysisRunInputNarrativeContextRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    CharacterTraitCandidateRow,
    CharacterTraitReviewRow,
)
from .narrative_context import (
    NarrativeScopeV1,
    canonical_json,
    canonical_scope_payload,
    payload_sha256,
    scope_relation,
)


CHARACTER_TRAIT_SCHEMA_VERSION = 1
MAX_CONFIRMED_TRAITS_PER_RUN = 5_000
MAX_CANDIDATES_PER_SOURCE_RUN = 500
_OBJECT_BEARING_TRAIT_DIMENSIONS = frozenset(
    {"preference", "value", "behavior_boundary", "current_state"}
)
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_FORBIDDEN_PROVENANCE_KEY_FRAGMENTS = (
    "apikey",
    "authorization",
    "baseurl",
    "credential",
    "password",
    "privatekey",
    "prompt",
    "providerresponse",
    "rawresponse",
    "responsebody",
    "secret",
)


class TraitEvidenceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_id: str = Field(min_length=1, max_length=36)
    document_id: str = Field(min_length=1, max_length=36)
    document_name: str = Field(min_length=1, max_length=255)
    document_version: int = Field(ge=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=8_000)

    @model_validator(mode="after")
    def validate_range(self) -> "TraitEvidenceInput":
        if self.line_end < self.line_start:
            raise ValueError("evidence range is invalid")
        return self


class TraitCandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    character_key: str = Field(min_length=1, max_length=160)
    character_display_name: str = Field(min_length=1, max_length=160)
    trait_type: Literal[
        "core_personality",
        "preference",
        "value",
        "speech_pattern",
        "behavior_boundary",
        "contextual_behavior",
        "current_state",
    ]
    trait_key: str = Field(min_length=1, max_length=160)
    comparison_key: str | None = Field(default=None, min_length=1, max_length=200)
    value: str = Field(min_length=1, max_length=2_000)
    polarity: Literal["positive", "negative", "neutral", "unclear"] = "unclear"
    stability: Literal["core", "stable", "temporary", "situational", "unknown"]
    contexts: list[str] = Field(default_factory=list, max_length=12)
    origin: Literal["explicit_setting", "history_inference"]
    authority_tier: Literal["core_canon", "formal_record"] = "formal_record"
    confidence: float = Field(ge=0, le=1)
    scope: NarrativeScopeV1 = Field(default_factory=NarrativeScopeV1)
    valid_from_release_ordinal: int | None = Field(default=None, ge=0)
    valid_until_release_ordinal: int | None = Field(default=None, ge=0)
    evidence: list[TraitEvidenceInput] = Field(min_length=1, max_length=12)
    support_refs: list[TraitSupportRef] = Field(default_factory=list, max_length=48)
    generator_version: str = Field(min_length=1, max_length=80)
    provenance: dict[str, Any] = Field(default_factory=dict)
    supersedes_candidate_id: str | None = Field(default=None, max_length=36)

    @model_validator(mode="after")
    def validate_release_range(self) -> "TraitCandidateInput":
        if (
            self.valid_from_release_ordinal is not None
            and self.valid_until_release_ordinal is not None
            and self.valid_until_release_ordinal < self.valid_from_release_ordinal
        ):
            raise ValueError("trait release range is invalid")
        normalized_contexts: dict[str, str] = {}
        for value in self.contexts:
            if not isinstance(value, str):
                raise ValueError("trait context is invalid")
            normalized = re.sub(
                r"\s+", " ", unicodedata.normalize("NFKC", value)
            ).strip()
            if (
                not normalized
                or len(normalized) > 160
                or _CONTROL.search(normalized)
            ):
                raise ValueError("trait context is invalid")
            normalized_contexts.setdefault(normalized.casefold(), normalized)
        self.contexts = sorted(
            normalized_contexts.values(), key=lambda item: item.casefold()
        )
        if self.trait_type == "contextual_behavior" and not self.contexts:
            raise ValueError("contextual behavior requires a context label")
        if self.origin == "history_inference" and len(self.evidence) < 2:
            raise ValueError("history inference requires independent evidence")
        if self.stability not in {"core", "stable"}:
            raise ValueError("temporary or situational signals cannot become stable traits")
        return self


def normalize_character_key(value: str) -> str:
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()
    if not normalized or len(normalized) > 160 or _CONTROL.search(normalized):
        raise ValueError("character key is invalid")
    return normalized


def normalize_trait_key(value: str) -> str:
    normalized = re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", value)
    ).strip().casefold()
    if not normalized or len(normalized) > 160 or _CONTROL.search(normalized):
        raise ValueError("trait key is invalid")
    return normalized


def _clean_text(value: str, *, maximum: int, label: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).strip()
    if not normalized or len(normalized) > maximum or _CONTROL.search(normalized):
        raise ValueError(f"{label} is invalid")
    return normalized


def _validated_comparison_key(value: str | None, *, trait_type: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) > 200
        or _CONTROL.search(value)
        or any(unicodedata.category(char).startswith("C") for char in value)
    ):
        raise ValueError("comparison key is invalid")
    prefix = f"{trait_type}:"
    anchor = value[len(prefix) :] if value.startswith(prefix) else ""
    if (
        not anchor
        or ":" in anchor
        or re.search(r"\s", anchor)
        or unicodedata.normalize("NFKC", anchor).casefold() != anchor
    ):
        raise ValueError("comparison key is invalid")
    return value


def _stored_comparison_key(value: object, *, trait_type: str) -> str | None:
    if value is not None and not isinstance(value, str):
        return None
    try:
        return _validated_comparison_key(value, trait_type=trait_type)
    except ValueError:
        return None


def candidate_snapshot_comparison_key(
    candidate: CharacterTraitCandidateRow,
) -> dict[str, str]:
    """Freeze only an actual stored identity; legacy rows remain keyless."""

    key = _stored_comparison_key(
        candidate.comparison_key, trait_type=candidate.trait_type
    )
    return {"comparison_key": key} if key is not None else {}


def _validate_safe_provenance(value: object, *, depth: int = 0) -> None:
    if depth > 8:
        raise ValueError("candidate provenance is too deep")
    if isinstance(value, dict):
        for key, nested in value.items():
            canonical_key = (
                re.sub(
                    r"[^a-z0-9]",
                    "",
                    unicodedata.normalize("NFKC", key).casefold(),
                )
                if isinstance(key, str)
                else ""
            )
            if not canonical_key or any(
                fragment in canonical_key
                for fragment in _FORBIDDEN_PROVENANCE_KEY_FRAGMENTS
            ):
                raise ValueError("candidate provenance contains a forbidden field")
            _validate_safe_provenance(nested, depth=depth + 1)
    elif isinstance(value, list):
        if len(value) > 256:
            raise ValueError("candidate provenance is too large")
        for nested in value:
            _validate_safe_provenance(nested, depth=depth + 1)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError("candidate provenance contains an invalid value")


def _validate_evidence(
    db, source_run_id: str, evidence: list[TraitEvidenceInput]
) -> list[dict[str, Any]]:
    input_ids = [row.input_id for row in evidence]
    evidence_keys = [
        (row.input_id, row.line_start, row.line_end) for row in evidence
    ]
    if len(set(evidence_keys)) != len(evidence_keys):
        raise ValueError("candidate evidence contains duplicate spans")
    snapshots = {
        row.id: row
        for row in db.scalars(
            select(AnalysisRunInputRow).where(
                AnalysisRunInputRow.run_id == source_run_id,
                AnalysisRunInputRow.id.in_(set(input_ids)),
            )
        ).all()
    }
    if set(snapshots) != set(input_ids):
        raise ValueError("candidate evidence is outside the frozen run")
    result: list[dict[str, Any]] = []
    for item in evidence:
        snapshot = snapshots[item.input_id]
        lines = snapshot.content.splitlines()
        if (
            snapshot.document_id != item.document_id
            or snapshot.document_name != item.document_name
            or snapshot.document_version != item.document_version
            or snapshot.content_sha256 != item.content_sha256
            or item.line_end > len(lines)
        ):
            raise ValueError("candidate evidence does not match the frozen run")
        source = "\n".join(lines[item.line_start - 1 : item.line_end])
        if source.strip() != item.text.strip():
            raise ValueError("candidate evidence text does not match the frozen run")
        if not source.strip() or len(source) > 8_000:
            raise ValueError("candidate evidence full line exceeds the supported length")
        # The extractor may trim boundary whitespace. Persist the verified
        # frozen line itself so new review records retain exact source text.
        result.append({**item.model_dump(mode="json"), "text": source})
    return result


def _semantic_evidence_digest(evidence: list[dict[str, Any]]) -> str:
    return payload_sha256([
        {
            "document_id": item["document_id"],
            "document_version": item["document_version"],
            "content_sha256": item["content_sha256"],
            "line_start": item["line_start"],
            "line_end": item["line_end"],
            "text": item["text"],
        }
        for item in evidence
    ])


def _frozen_context_identities(
    db, source_run_id: str, evidence: list[dict[str, Any]]
) -> tuple[tuple[str, str], ...]:
    """Identify every evidence document by its *frozen* context revision.

    Input ids change between equivalent runs, whereas document ids and the
    frozen payload hashes do not.  The latter include the revision id, so an
    intervening retirement cannot silently revive an earlier pending row.
    """

    try:
        input_documents = {item["input_id"]: item["document_id"] for item in evidence}
        if not input_documents:
            raise ValueError("candidate has no frozen context inputs")
        if any(
            not isinstance(input_id, str) or not isinstance(document_id, str)
            for input_id, document_id in input_documents.items()
        ):
            raise ValueError("candidate frozen context identity is invalid")
        snapshots = {
            row.id: row for row in db.scalars(
                select(AnalysisRunInputRow).where(
                    AnalysisRunInputRow.run_id == source_run_id,
                    AnalysisRunInputRow.id.in_(input_documents),
                )
            ).all()
        }
        contexts = {
            row.input_id: row for row in db.scalars(
                select(AnalysisRunInputNarrativeContextRow).where(
                    AnalysisRunInputNarrativeContextRow.input_id.in_(input_documents)
                )
            ).all()
        }
        if (
            set(snapshots) != set(input_documents)
            or set(contexts) != set(input_documents)
            or len(set(input_documents.values())) != len(input_documents)
        ):
            raise ValueError("candidate frozen narrative context is missing")
        for item in evidence:
            snapshot = snapshots[item["input_id"]]
            if (
                snapshot.document_id != item["document_id"]
                or snapshot.document_version != item["document_version"]
                or snapshot.content_sha256 != item["content_sha256"]
            ):
                raise ValueError("candidate frozen evidence identity is invalid")
        identities: dict[str, str] = {}
        for input_id, document_id in input_documents.items():
            snapshot = snapshots[input_id]
            context = contexts[input_id]
            payload = context.payload
            if (
                snapshot.document_id != document_id
                or sha256(snapshot.content.encode("utf-8")).hexdigest()
                != snapshot.content_sha256
                or context.schema_version != 1
                or not isinstance(payload, dict)
                or payload.get("schema_version") != 1
                or payload_sha256(payload) != context.payload_sha256
                or payload.get("context_revision_id") != context.context_revision_id
                or type(payload.get("context_revision")) is not int
                or payload["context_revision"] < 0
                or (context.context_revision_id is None)
                != (payload["context_revision"] == 0)
            ):
                raise ValueError("candidate frozen narrative context is invalid")
            if document_id in identities:
                raise ValueError("candidate frozen narrative context is ambiguous")
            identities[document_id] = context.payload_sha256
        return tuple(sorted(identities.items()))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("candidate frozen narrative context is invalid") from exc


def _reused_candidate_context_identities(
    db, row: CharacterTraitCandidateRow
) -> tuple[tuple[str, str], ...]:
    """A legacy hash is reusable only when its evidence and context survive."""

    if not isinstance(row.evidence, list) or payload_sha256(row.evidence) != row.evidence_sha256:
        raise ValueError("reused candidate evidence is invalid")
    try:
        if (
            not isinstance(row.scope_payload, dict)
            or canonical_scope_payload(row.scope_payload) != row.scope_payload
            or payload_sha256(row.scope_payload) != row.scope_sha256
        ):
            raise ValueError("reused candidate scope is invalid")
    except (TypeError, AttributeError) as exc:
        raise ValueError("reused candidate scope is invalid") from exc
    try:
        parsed = [TraitEvidenceInput.model_validate(item) for item in row.evidence]
        if _validate_evidence(db, row.source_run_id, parsed) != row.evidence:
            raise ValueError("reused candidate evidence is not exact")
    except (TypeError, AttributeError) as exc:
        raise ValueError("reused candidate evidence is invalid") from exc
    return _frozen_context_identities(db, row.source_run_id, row.evidence)


def _verify_reused_review_chain(db, row: CharacterTraitCandidateRow) -> None:
    """A mutable state label alone is not evidence of an author decision."""

    reviews = list(db.scalars(
        select(CharacterTraitReviewRow)
        .where(CharacterTraitReviewRow.candidate_id == row.id)
        .order_by(CharacterTraitReviewRow.expected_lock_version, CharacterTraitReviewRow.id)
        .limit(4)
    ).all())
    if row.review_state == "pending":
        if row.lock_version == 0 and not reviews:
            return
        raise ValueError("reused candidate review state is invalid")
    expected = {
        "confirmed": ("confirm",),
        "rejected": ("reject",),
        "superseded": ("confirm", "supersede"),
        "withdrawn": ("confirm", "withdraw"),
    }.get(row.review_state)
    if (
        expected is None
        or row.lock_version != len(expected)
        or len(reviews) != len(expected)
        or row.reviewed_at is None
        or any(
            review.project_id != row.project_id
            or review.decision != decision
            or review.expected_lock_version != index
            or review.approved_axis_id != row.approved_axis_id
            or review.approved_axis_version != row.approved_axis_version
            for index, (review, decision) in enumerate(zip(reviews, expected, strict=True))
        )
    ):
        raise ValueError("reused candidate review state is invalid")
    if row.review_state == "superseded":
        successor_id = reviews[1].comment.removeprefix("由候选 ").removesuffix(" 替代")
        successor = db.get(CharacterTraitCandidateRow, successor_id)
        if (
            reviews[1].comment != f"由候选 {successor_id} 替代"
            or successor is None
            or successor.project_id != row.project_id
            or successor.character_key != row.character_key
            or successor.trait_type != row.trait_type
            or successor.supersedes_candidate_id != row.id
            or successor.review_state not in {"confirmed", "superseded", "withdrawn"}
            or successor.lock_version < 1
            or successor.reviewed_at is None
        ):
            raise ValueError("reused candidate supersession is invalid")
        successor_confirms = db.scalars(select(CharacterTraitReviewRow).where(
            CharacterTraitReviewRow.candidate_id == successor.id,
            CharacterTraitReviewRow.project_id == row.project_id,
            CharacterTraitReviewRow.decision == "confirm",
        ).limit(2)).all()
        if (
            len(successor_confirms) != 1
            or successor_confirms[0].expected_lock_version != 0
            or successor_confirms[0].approved_axis_id != successor.approved_axis_id
            or successor_confirms[0].approved_axis_version
            != successor.approved_axis_version
        ):
            raise ValueError("reused candidate supersession is invalid")


def _verify_reused_nonformal_fingerprint(
    row: CharacterTraitCandidateRow,
    contexts: tuple[tuple[str, str], ...],
    *,
    comparison_key_override: str | None = None,
) -> None:
    if (
        row.support_binding_mode != "legacy_v1"
        or row.support_bindings_v1 is not None
        or row.support_bindings_sha256 is not None
    ):
        raise ValueError("reused candidate support mode is invalid")
    comparison_key = row.comparison_key or comparison_key_override
    base = {
        "character_key": row.character_key,
        "trait_type": row.trait_type,
        "comparison_key": comparison_key or normalize_trait_key(row.trait_key),
        "polarity": row.polarity,
        "stability": row.stability,
        "contexts": row.contexts,
        "origin": row.origin,
        "authority_tier": row.authority_tier,
        "scope_sha256": row.scope_sha256,
        "valid_from_release_ordinal": row.valid_from_release_ordinal,
        "valid_until_release_ordinal": row.valid_until_release_ordinal,
        "semantic_evidence_sha256": _semantic_evidence_digest(row.evidence),
        "supersedes_candidate_id": row.supersedes_candidate_id,
        "generator_version": row.generator_version,
    }
    valid = {payload_sha256(base)}
    if comparison_key is not None and row.trait_type == "preference":
        base["fingerprint_version"] = "keyed_preference_v2"
        valid.add(payload_sha256(base))
    elif comparison_key is not None and row.trait_type in _OBJECT_BEARING_TRAIT_DIMENSIONS:
        base["relation_axis"] = stable_trait_identity(row.trait_type, row.trait_key)
        valid.add(payload_sha256(base))
    base["fingerprint_version"] = "context_bound_v1"
    base["frozen_contexts"] = [
        {"document_id": document_id, "payload_sha256": context_sha256}
        for document_id, context_sha256 in contexts
    ]
    valid.add(payload_sha256(base))
    if row.candidate_fingerprint not in valid:
        raise ValueError("reused candidate fingerprint is invalid")


def _validate_reused_candidate_identity(
    db, row: CharacterTraitCandidateRow, *, comparison_key_override: str | None = None
) -> tuple[tuple[str, str], ...]:
    contexts = _reused_candidate_context_identities(db, row)
    _verify_reused_review_chain(db, row)
    if row.support_binding_mode == "required_v1":
        _verify_reused_formal_target(db, row)
    else:
        _verify_reused_nonformal_fingerprint(
            row, contexts, comparison_key_override=comparison_key_override
        )
    return contexts


def _formal_target_fingerprint(
    base_payload: dict[str, Any],
    *,
    project_id: str,
    evidence: list[dict[str, Any]],
    support_bindings: dict,
    frozen_context_sha256: str | None = None,
) -> str:
    if (
        support_bindings.get("index_version") != "assertion-index-v1"
        or not isinstance(support_bindings.get("bindings"), list)
        or len(support_bindings["bindings"]) != 1
    ):
        raise ValueError("formal target fingerprint requires one verified target")
    target = support_bindings["bindings"][0]
    source = evidence[target["evidence_index"]]
    support_identity = {
        "project_id": project_id,
        "document_id": source["document_id"],
        "document_version": source["document_version"],
        "content_sha256": source["content_sha256"],
        "line_number": source["line_start"],
        "support_id": target["support_id"],
        "actor_anchor_id": target["actor_anchor_id"],
        "label_anchor_id": target["label_anchor_id"],
        "scope_relation": target["scope_relation"],
        "index_version": support_bindings["index_version"],
    }
    if frozen_context_sha256 is not None:
        support_identity["frozen_context_sha256"] = frozen_context_sha256
    return payload_sha256({
        **base_payload,
        "fingerprint_version": (
            "formal_target_v2" if frozen_context_sha256 is not None
            else "formal_target_v1"
        ),
        "support_identity": support_identity,
    })


def formal_target_fingerprint_matches(
    row: CharacterTraitCandidateRow, support_bindings: dict, *, db=None
) -> bool:
    """Bind persisted target metadata to the candidate's stable semantic ID."""

    if row.support_binding_mode != "required_v1":
        return False
    try:
        if (
            not isinstance(row.evidence, list)
            or payload_sha256(row.evidence) != row.evidence_sha256
            or payload_sha256(row.scope_payload) != row.scope_sha256
            or normalize_character_key(row.character_key) != row.character_key
        ):
            return False
        base_payload = {
            "character_key": row.character_key,
            "trait_type": row.trait_type,
            "comparison_key": row.comparison_key or normalize_trait_key(row.trait_key),
            "polarity": row.polarity,
            "stability": row.stability,
            "contexts": row.contexts,
            "origin": row.origin,
            "authority_tier": row.authority_tier,
            "scope_sha256": row.scope_sha256,
            "valid_from_release_ordinal": row.valid_from_release_ordinal,
            "valid_until_release_ordinal": row.valid_until_release_ordinal,
            "semantic_evidence_sha256": _semantic_evidence_digest(row.evidence),
            "supersedes_candidate_id": row.supersedes_candidate_id,
            "generator_version": row.generator_version,
        }
        if _formal_target_fingerprint(
            base_payload,
            project_id=row.project_id,
            evidence=row.evidence,
            support_bindings=support_bindings,
        ) == row.candidate_fingerprint:
            return True  # Historical v1 rows retain their original identity.
        if db is None:
            return False
        frozen_contexts = dict(_frozen_context_identities(
            db, row.source_run_id, row.evidence
        ))
        target = support_bindings["bindings"][0]
        source = row.evidence[target["evidence_index"]]
        return _formal_target_fingerprint(
            base_payload,
            project_id=row.project_id,
            evidence=row.evidence,
            support_bindings=support_bindings,
            frozen_context_sha256=frozen_contexts[source["document_id"]],
        ) == row.candidate_fingerprint
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def _verify_reused_formal_target(db, row: CharacterTraitCandidateRow) -> None:
    """Never count a damaged prior formal target as a successful reuse."""

    if row.support_binding_mode != "required_v1" or not isinstance(row.evidence, list):
        raise ValueError("reused formal target lacks required support binding")
    try:
        input_ids = {item["input_id"] for item in row.evidence}
        frozen_by_id = {
            item.id: item for item in db.scalars(
                select(AnalysisRunInputRow).where(
                    AnalysisRunInputRow.run_id == row.source_run_id,
                    AnalysisRunInputRow.id.in_(input_ids),
                )
            ).all()
        }
        if input_ids != set(frozen_by_id):
            raise ValueError("reused formal target frozen source is missing")
        binding = verify_stored_support_bindings(
            row.support_bindings_v1,
            row.support_bindings_sha256,
            evidence=row.evidence,
            frozen_by_id=frozen_by_id,
        )
        if binding is None or not formal_target_fingerprint_matches(
            row, binding, db=db
        ):
            raise ValueError("reused formal target binding identity is invalid")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("reused formal target binding is invalid") from exc


def validate_character_trait_supersession(
    *,
    project_id: str,
    character_key: str,
    trait_type: str,
    trait_key: str,
    comparison_key: str | None,
    origin: str,
    authority_tier: str,
    scope: NarrativeScopeV1 | dict[str, Any],
    valid_from_release_ordinal: int | None,
    valid_until_release_ordinal: int | None,
    superseded: CharacterTraitCandidateRow,
) -> None:
    """Fail closed unless ``superseded`` is the same live profile identity.

    A client-supplied internal id is never enough to establish that two rows
    represent the same trait.  Lower-authority evidence cannot replace a
    higher-authority baseline. Narrative scopes must be provably compatible,
    and their release ranges must overlap; unknown scope relations are rejected.
    """

    try:
        same_identity = (
            superseded.project_id == project_id
            and superseded.review_state == "confirmed"
            and normalize_character_key(superseded.character_key)
            == normalize_character_key(character_key)
            and superseded.trait_type == trait_type
        )
        if same_identity and trait_type in _OBJECT_BEARING_TRAIT_DIMENSIONS:
            new_key = _stored_comparison_key(comparison_key, trait_type=trait_type)
            old_key = _stored_comparison_key(
                superseded.comparison_key, trait_type=trait_type
            )
            # Explicitly linked legacy rows without a usable anchor retain the
            # earlier label rule; two valid keys identify the object directly.
            if new_key is not None and old_key is not None:
                same_identity = new_key == old_key and (
                    trait_type == "preference"
                    or stable_trait_identity(trait_type, superseded.trait_key)
                    == stable_trait_identity(trait_type, trait_key)
                )
            else:
                same_identity = normalize_trait_key(
                    superseded.trait_key
                ) == normalize_trait_key(trait_key)
        elif same_identity:
            same_identity = normalize_trait_key(
                superseded.trait_key
            ) == normalize_trait_key(trait_key)
    except (TypeError, ValueError):
        same_identity = False
    if not same_identity:
        raise ValueError("superseded candidate is incompatible")

    # Match the snapshot authority order: canon, explicit formal setting,
    # inferred formal history. Unknown stored values fail closed.
    authority_rank = {
        ("core_canon", "explicit_setting"): 2,
        ("core_canon", "history_inference"): 2,
        ("formal_record", "explicit_setting"): 1,
        ("formal_record", "history_inference"): 0,
    }
    new_rank = authority_rank.get((authority_tier, origin))
    old_rank = authority_rank.get((superseded.authority_tier, superseded.origin))
    if new_rank is None or old_rank is None or new_rank < old_rank:
        raise ValueError("superseded candidate is incompatible")

    first_start = valid_from_release_ordinal or 0
    second_start = superseded.valid_from_release_ordinal or 0
    first_end = (
        valid_until_release_ordinal
        if valid_until_release_ordinal is not None
        else 2_147_483_647
    )
    second_end = (
        superseded.valid_until_release_ordinal
        if superseded.valid_until_release_ordinal is not None
        else 2_147_483_647
    )
    if max(first_start, second_start) > min(first_end, second_end):
        raise ValueError("superseded candidate is incompatible")

    if scope_relation(
        scope,
        superseded.scope_payload,
        first_resolution="confirmed",
        second_resolution="confirmed",
    ) != "compatible":
        raise ValueError("superseded candidate is incompatible")


def upsert_character_trait_candidate(
    db,
    *,
    project_id: str,
    source_run_id: str,
    candidate: TraitCandidateInput | dict[str, Any],
    allow_running_source: bool = False,
) -> tuple[CharacterTraitCandidateRow, bool]:
    """Persist one bounded pending candidate from a frozen run.

    This is the only write helper intended for the later semantic module.  It
    validates evidence against immutable run inputs and strips the temptation
    to persist provider prompts, credentials, or raw responses.  The default
    only accepts completed runs; the owned analysis worker must explicitly opt
    in while its source run is still running.
    """

    parsed = (
        candidate
        if isinstance(candidate, TraitCandidateInput)
        else TraitCandidateInput.model_validate(candidate)
    )
    run = db.get(AnalysisRunRow, source_run_id)
    allowed_statuses = {"completed", "running"} if allow_running_source else {"completed"}
    if (
        run is None
        or run.project_id != project_id
        or run.status not in allowed_statuses
    ):
        raise ValueError("candidate source run is invalid")
    character_key = normalize_character_key(parsed.character_key)
    display_name = _clean_text(
        parsed.character_display_name, maximum=160, label="character name"
    )
    trait_key = _clean_text(parsed.trait_key, maximum=160, label="trait key")
    trait_value = _clean_text(parsed.value, maximum=2_000, label="trait value")
    comparison_key = _validated_comparison_key(
        parsed.comparison_key, trait_type=parsed.trait_type
    )
    _validate_safe_provenance(parsed.provenance)
    if len(canonical_json(parsed.provenance).encode("utf-8")) > 32_000:
        raise ValueError("candidate provenance is too large")
    evidence = _validate_evidence(db, source_run_id, parsed.evidence)
    frozen_contexts = _frozen_context_identities(db, source_run_id, evidence)
    support_bindings = None
    if parsed.support_refs:
        if parsed.origin != "explicit_setting":
            raise ValueError("candidate support binding requires a formal setting")
        if len(parsed.support_refs) != 1 or len(evidence) != 1:
            raise ValueError("reviewed formal candidate must bind one target")
        frozen_by_id = {
            row.id: row for row in db.scalars(
                select(AnalysisRunInputRow).where(
                    AnalysisRunInputRow.run_id == source_run_id,
                    AnalysisRunInputRow.id.in_({item["input_id"] for item in evidence}),
                )
            ).all()
        }
        support_bindings = bind_support_refs(
            parsed.support_refs, evidence=evidence, frozen_by_id=frozen_by_id
        )
    evidence_hash = payload_sha256(evidence)
    semantic_evidence_hash = _semantic_evidence_digest(evidence)
    scope = canonical_scope_payload(parsed.scope)
    scope_hash = payload_sha256(scope)
    fingerprint_payload = {
        "character_key": character_key,
        "trait_type": parsed.trait_type,
        "comparison_key": comparison_key or normalize_trait_key(trait_key),
        "polarity": parsed.polarity,
        "stability": parsed.stability,
        "contexts": parsed.contexts,
        "origin": parsed.origin,
        "authority_tier": parsed.authority_tier,
        "scope_sha256": scope_hash,
        "valid_from_release_ordinal": parsed.valid_from_release_ordinal,
        "valid_until_release_ordinal": parsed.valid_until_release_ordinal,
        "semantic_evidence_sha256": semantic_evidence_hash,
        "supersedes_candidate_id": parsed.supersedes_candidate_id,
        "generator_version": parsed.generator_version,
    }
    keyed_preference = comparison_key is not None and parsed.trait_type == "preference"
    keyed_relation = (
        comparison_key is not None
        and parsed.trait_type in _OBJECT_BEARING_TRAIT_DIMENSIONS
        and parsed.trait_type != "preference"
    )
    legacy_fingerprints = {payload_sha256(fingerprint_payload)}
    if support_bindings is not None:
        old_formal_fingerprint = _formal_target_fingerprint(
            fingerprint_payload,
            project_id=project_id,
            evidence=evidence,
            support_bindings=support_bindings,
        )
        legacy_fingerprints = {old_formal_fingerprint}
        fingerprint = _formal_target_fingerprint(
            fingerprint_payload,
            project_id=project_id,
            evidence=evidence,
            support_bindings=support_bindings,
            frozen_context_sha256=dict(frozen_contexts)[evidence[0]["document_id"]],
        )
    elif keyed_preference:
        # Pre-0014 hashes could contain the parsed object key even though the
        # migrated row has no stored key. A versioned hash makes a new,
        # reviewable keyed candidate without changing that historical row.
        fingerprint_payload["fingerprint_version"] = "keyed_preference_v2"
        legacy_fingerprints.add(payload_sha256(fingerprint_payload))
    elif keyed_relation:
        # Keep historical hashes unchanged. New keyed rows cannot collide when
        # they share an object and evidence but describe distinct relations.
        fingerprint_payload["relation_axis"] = stable_trait_identity(
            parsed.trait_type, trait_key
        )
        legacy_fingerprints.add(payload_sha256(fingerprint_payload))
    if support_bindings is None:
        fingerprint_payload["fingerprint_version"] = "context_bound_v1"
        fingerprint_payload["frozen_contexts"] = [
            {"document_id": document_id, "payload_sha256": context_sha256}
            for document_id, context_sha256 in frozen_contexts
        ]
        fingerprint = payload_sha256(fingerprint_payload)
    possible_existing = db.scalars(
        select(CharacterTraitCandidateRow)
        .join(
            AnalysisRunRow,
            AnalysisRunRow.id == CharacterTraitCandidateRow.source_run_id,
        )
        .where(
            CharacterTraitCandidateRow.project_id == project_id,
            CharacterTraitCandidateRow.candidate_fingerprint.in_(
                (fingerprint, *sorted(legacy_fingerprints))
            ),
            (
                (CharacterTraitCandidateRow.source_run_id == source_run_id)
                | (AnalysisRunRow.status == "completed")
            ),
        )
        .order_by(CharacterTraitCandidateRow.created_at, CharacterTraitCandidateRow.id)
    ).all()
    for existing in possible_existing:
        prior_contexts = _validate_reused_candidate_identity(
            db, existing,
            comparison_key_override=(comparison_key if support_bindings is None else None),
        )
        if prior_contexts != frozen_contexts:
            continue
        if existing.candidate_fingerprint == fingerprint:
            return existing, False
        if (
            support_bindings is None
            and keyed_preference
            and existing.candidate_fingerprint in legacy_fingerprints
            and _stored_comparison_key(
                existing.comparison_key, trait_type=parsed.trait_type
            ) == comparison_key
        ):
            return existing, False
        if (
            support_bindings is None
            and keyed_relation
            and existing.candidate_fingerprint in legacy_fingerprints
            and stable_trait_identity(parsed.trait_type, existing.trait_key)
            == stable_trait_identity(parsed.trait_type, trait_key)
            and (
                _stored_comparison_key(
                    existing.comparison_key, trait_type=parsed.trait_type
                ) == comparison_key
                # Rows migrated from before the comparison_key column have a
                # NULL key, but their old fingerprint already included the
                # parsed object key. Match that hash and the relation axis;
                # never infer an object from the prose value.
                or existing.comparison_key is None
            )
        ):
            return existing, False
        if (
            support_bindings is not None
            and existing.candidate_fingerprint in legacy_fingerprints
        ):
            return existing, False
        if (
            support_bindings is None
            and not keyed_preference
            and not keyed_relation
            and existing.candidate_fingerprint in legacy_fingerprints
        ):
            return existing, False

    if parsed.supersedes_candidate_id:
        superseded = db.get(
            CharacterTraitCandidateRow, parsed.supersedes_candidate_id
        )
        if superseded is None:
            raise ValueError("superseded candidate is invalid")
        validate_character_trait_supersession(
            project_id=project_id,
            character_key=character_key,
            trait_type=parsed.trait_type,
            trait_key=trait_key,
            comparison_key=comparison_key,
            origin=parsed.origin,
            authority_tier=parsed.authority_tier,
            scope=scope,
            valid_from_release_ordinal=parsed.valid_from_release_ordinal,
            valid_until_release_ordinal=parsed.valid_until_release_ordinal,
            superseded=superseded,
        )
    from sqlalchemy import func

    total = db.scalar(
        select(func.count())
        .select_from(CharacterTraitCandidateRow)
        .where(CharacterTraitCandidateRow.source_run_id == source_run_id)
    ) or 0
    if total >= MAX_CANDIDATES_PER_SOURCE_RUN:
        raise ValueError("candidate limit exceeded")
    row = CharacterTraitCandidateRow(
        project_id=project_id,
        source_run_id=source_run_id,
        character_key=character_key,
        character_display_name=display_name,
        trait_type=parsed.trait_type,
        trait_key=trait_key,
        comparison_key=comparison_key,
        value=trait_value,
        polarity=parsed.polarity,
        stability=parsed.stability,
        contexts=parsed.contexts,
        origin=parsed.origin,
        authority_tier=parsed.authority_tier,
        confidence=parsed.confidence,
        scope_payload=scope,
        scope_sha256=scope_hash,
        valid_from_release_ordinal=parsed.valid_from_release_ordinal,
        valid_until_release_ordinal=parsed.valid_until_release_ordinal,
        evidence=evidence,
        evidence_sha256=evidence_hash,
        support_binding_mode=("required_v1" if support_bindings is not None else "legacy_v1"),
        support_bindings_v1=support_bindings,
        support_bindings_sha256=(
            support_bindings_sha256(support_bindings)
            if support_bindings is not None else None
        ),
        candidate_fingerprint=fingerprint,
        generator_version=parsed.generator_version,
        provenance=parsed.provenance,
        supersedes_candidate_id=parsed.supersedes_candidate_id,
    )
    db.add(row)
    db.flush()
    return row, True


def candidate_snapshot_payload(
    candidate: CharacterTraitCandidateRow,
    review: CharacterTraitReviewRow,
) -> dict[str, Any]:
    if candidate.review_state != "confirmed" or review.decision != "confirm":
        raise ValueError("only confirmed candidates can be snapshotted")
    scope = canonical_scope_payload(candidate.scope_payload)
    if payload_sha256(scope) != candidate.scope_sha256:
        raise ValueError("candidate scope hash mismatch")
    if payload_sha256(candidate.evidence) != candidate.evidence_sha256:
        raise ValueError("candidate evidence hash mismatch")
    return {
        "schema_version": CHARACTER_TRAIT_SCHEMA_VERSION,
        "candidate_id": candidate.id,
        "confirmation_review_id": review.id,
        "character_key": candidate.character_key,
        "character_display_name": candidate.character_display_name,
        "trait_type": candidate.trait_type,
        "trait_key": candidate.trait_key,
        **candidate_snapshot_comparison_key(candidate),
        "value": candidate.value,
        "polarity": candidate.polarity,
        "stability": candidate.stability,
        "contexts": candidate.contexts,
        "origin": candidate.origin,
        "authority_tier": candidate.authority_tier,
        "scope": scope,
        "scope_sha256": candidate.scope_sha256,
        "valid_from_release_ordinal": candidate.valid_from_release_ordinal,
        "valid_until_release_ordinal": candidate.valid_until_release_ordinal,
        "evidence": candidate.evidence,
        "evidence_sha256": candidate.evidence_sha256,
        "candidate_fingerprint": candidate.candidate_fingerprint,
        "generator_version": candidate.generator_version,
        "candidate_lock_version": candidate.lock_version,
    }


def latest_confirm_reviews(
    db, candidate_ids: list[str]
) -> dict[str, CharacterTraitReviewRow]:
    if not candidate_ids:
        return {}
    rows = list(
        db.scalars(
            select(CharacterTraitReviewRow)
            .where(
                CharacterTraitReviewRow.candidate_id.in_(candidate_ids),
                CharacterTraitReviewRow.decision == "confirm",
            )
            .order_by(
                CharacterTraitReviewRow.candidate_id,
                CharacterTraitReviewRow.created_at.desc(),
                CharacterTraitReviewRow.id.desc(),
            )
        ).all()
    )
    result: dict[str, CharacterTraitReviewRow] = {}
    for row in rows:
        result.setdefault(row.candidate_id, row)
    return result
