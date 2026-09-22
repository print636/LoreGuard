from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from .db import DocumentNarrativeContextRevisionRow


NARRATIVE_CONTEXT_SCHEMA_VERSION = 1
ResolutionState = Literal["unresolved", "inferred", "confirmed"]
PublicationStatus = Literal[
    "draft", "in_review", "published", "retired", "unknown"
]
ScopeRelation = Literal["compatible", "incompatible", "unknown"]

_KEY_PATTERN = r"^[A-Za-z0-9_.:\-一-鿿]+$"
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


class NarrativeRelease(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1, max_length=80, pattern=_KEY_PATTERN)
    ordinal: int = Field(ge=0, le=2_147_483_647)


class NarrativeBranch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: list[str] = Field(min_length=1, max_length=16)
    exclusive_group: str | None = Field(
        default=None, min_length=1, max_length=80, pattern=_KEY_PATTERN
    )

    @model_validator(mode="after")
    def validate_path(self) -> "NarrativeBranch":
        normalized: list[str] = []
        for value in self.path:
            if not isinstance(value, str):
                raise ValueError("branch path is invalid")
            candidate = unicodedata.normalize("NFKC", value).strip()
            if (
                not candidate
                or len(candidate) > 80
                or _CONTROL.search(candidate)
                or re.fullmatch(_KEY_PATTERN, candidate) is None
            ):
                raise ValueError("branch path is invalid")
            normalized.append(candidate)
        if len(set(normalized)) != len(normalized):
            raise ValueError("branch path contains a cycle")
        self.path = normalized
        return self


class NarrativeScopeV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    timeline_key: str = Field(
        default="main", min_length=1, max_length=80, pattern=_KEY_PATTERN
    )
    release: NarrativeRelease | None = None
    branch: NarrativeBranch | None = None
    activity_key: str | None = Field(
        default=None, min_length=1, max_length=80, pattern=_KEY_PATTERN
    )


class NarrativeContextInput(BaseModel):
    """Client-visible fields; effective authority is never client supplied."""

    model_config = ConfigDict(extra="forbid")

    resolution_state: Literal["unresolved", "confirmed"] = "unresolved"
    publication_status: PublicationStatus = "unknown"
    scope: NarrativeScopeV1 = Field(default_factory=NarrativeScopeV1)


class NarrativeContextRevisionInput(NarrativeContextInput):
    expected_revision: int = Field(ge=0)
    # Optional so every existing confirmation payload remains valid.  A user
    # may correct an initially misclassified import in the same optimistic
    # transaction that confirms its authority metadata.
    document_role: Literal[
        "chapter", "canon", "character_profile", "reference"
    ] | None = None


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def payload_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def narrative_context_semantic_sha256(payload: dict[str, Any]) -> str:
    """Identity used for recheck comparability, excluding revision lineage."""
    return payload_sha256(
        {
            "schema_version": payload.get("schema_version"),
            "resolution_state": payload.get("resolution_state"),
            "authority_tier": payload.get("authority_tier"),
            "publication_status": payload.get("publication_status"),
            "scope": payload.get("scope"),
            "legacy_document_role": payload.get("legacy_document_role"),
            "legacy_story_scope": payload.get("legacy_story_scope"),
        }
    )


def canonical_scope_payload(scope: NarrativeScopeV1 | dict[str, Any]) -> dict[str, Any]:
    parsed = scope if isinstance(scope, NarrativeScopeV1) else NarrativeScopeV1.model_validate(scope)
    # Keep explicit nulls out of identities so equivalent API representations
    # cannot produce different snapshot hashes.
    return parsed.model_dump(mode="json", exclude_none=True)


def derive_authority_tier(
    document_role: str,
    resolution_state: ResolutionState,
    publication_status: PublicationStatus,
) -> str:
    if resolution_state != "confirmed":
        return "unresolved"
    if document_role == "canon":
        return "core_canon"
    if document_role == "character_profile":
        return "formal_record"
    if document_role == "chapter":
        return (
            "formal_record"
            if publication_status in {"published", "retired"}
            else "draft"
        )
    return "reference"


def latest_context_revisions(
    db, document_ids: list[str]
) -> dict[str, DocumentNarrativeContextRevisionRow]:
    if not document_ids:
        return {}
    rows = list(
        db.scalars(
            select(DocumentNarrativeContextRevisionRow)
            .where(DocumentNarrativeContextRevisionRow.document_id.in_(document_ids))
            .order_by(
                DocumentNarrativeContextRevisionRow.document_id,
                DocumentNarrativeContextRevisionRow.revision.desc(),
            )
        ).all()
    )
    result: dict[str, DocumentNarrativeContextRevisionRow] = {}
    for row in rows:
        result.setdefault(row.document_id, row)
    return result


def context_snapshot_payload(
    row: DocumentNarrativeContextRevisionRow | None,
    *,
    document_role: str,
    story_scope: str,
) -> dict[str, Any]:
    if row is None:
        scope = canonical_scope_payload(NarrativeScopeV1())
        return {
            "schema_version": NARRATIVE_CONTEXT_SCHEMA_VERSION,
            "context_revision_id": None,
            "context_revision": 0,
            "resolution_state": "unresolved",
            "origin": "legacy",
            "authority_tier": "unresolved",
            "publication_status": "unknown",
            "scope": scope,
            "scope_sha256": payload_sha256(scope),
            "legacy_document_role": document_role,
            "legacy_story_scope": story_scope,
        }
    scope = canonical_scope_payload(row.scope_payload)
    scope_hash = payload_sha256(scope)
    if scope_hash != row.scope_sha256:
        raise ValueError("narrative context scope hash mismatch")
    expected_authority = derive_authority_tier(
        document_role,
        row.resolution_state,
        row.publication_status,
    )
    if row.authority_tier != expected_authority:
        raise ValueError("narrative context authority mismatch")
    return {
        "schema_version": NARRATIVE_CONTEXT_SCHEMA_VERSION,
        "context_revision_id": row.id,
        "context_revision": row.revision,
        "resolution_state": row.resolution_state,
        "origin": row.origin,
        "authority_tier": row.authority_tier,
        "publication_status": row.publication_status,
        "scope": scope,
        "scope_sha256": scope_hash,
        "legacy_document_role": document_role,
        "legacy_story_scope": story_scope,
    }


def add_context_revision(
    db,
    *,
    project_id: str,
    document_id: str,
    document_role: str,
    resolution_state: ResolutionState,
    publication_status: PublicationStatus,
    scope: NarrativeScopeV1 | dict[str, Any],
    origin: Literal["explicit", "deterministic_import", "model_inferred", "legacy"],
    created_by_user_id: str | None,
    expected_revision: int | None = None,
    inference_confidence: float | None = None,
    inference_reasoning: str | None = None,
    inference_evidence: list[dict[str, Any]] | None = None,
    inference_usage: dict[str, int] | None = None,
) -> DocumentNarrativeContextRevisionRow:
    latest = db.scalar(
        select(DocumentNarrativeContextRevisionRow)
        .where(
            DocumentNarrativeContextRevisionRow.project_id == project_id,
            DocumentNarrativeContextRevisionRow.document_id == document_id,
        )
        .order_by(DocumentNarrativeContextRevisionRow.revision.desc())
        .limit(1)
    )
    actual_revision = latest.revision if latest is not None else 0
    if expected_revision is not None and expected_revision != actual_revision:
        raise NarrativeContextRevisionConflict(actual_revision)
    if origin == "model_inferred" and resolution_state != "inferred":
        raise ValueError("model inference cannot confirm narrative context")
    scope_payload = canonical_scope_payload(scope)
    row = DocumentNarrativeContextRevisionRow(
        project_id=project_id,
        document_id=document_id,
        revision=actual_revision + 1,
        resolution_state=resolution_state,
        origin=origin,
        authority_tier=derive_authority_tier(
            document_role, resolution_state, publication_status
        ),
        publication_status=publication_status,
        scope_payload=scope_payload,
        scope_sha256=payload_sha256(scope_payload),
        inference_confidence=inference_confidence,
        inference_reasoning=inference_reasoning,
        inference_evidence=inference_evidence,
        inference_usage=inference_usage,
        created_by_user_id=created_by_user_id,
    )
    db.add(row)
    return row


class NarrativeContextRevisionConflict(RuntimeError):
    def __init__(self, actual_revision: int):
        super().__init__("narrative context revision conflict")
        self.actual_revision = actual_revision


def scope_relation(
    first_scope: NarrativeScopeV1 | dict[str, Any],
    second_scope: NarrativeScopeV1 | dict[str, Any],
    *,
    first_resolution: str,
    second_resolution: str,
) -> ScopeRelation:
    if first_resolution != "confirmed" or second_resolution != "confirmed":
        return "unknown"
    try:
        first = (
            first_scope
            if isinstance(first_scope, NarrativeScopeV1)
            else NarrativeScopeV1.model_validate(first_scope)
        )
        second = (
            second_scope
            if isinstance(second_scope, NarrativeScopeV1)
            else NarrativeScopeV1.model_validate(second_scope)
        )
    except (TypeError, ValueError):
        return "unknown"
    if first.timeline_key != second.timeline_key:
        return "incompatible"
    if (
        first.activity_key is not None
        and second.activity_key is not None
        and first.activity_key != second.activity_key
    ):
        return "incompatible"
    if first.branch is None or second.branch is None:
        return "compatible"
    left = tuple(first.branch.path)
    right = tuple(second.branch.path)
    common = min(len(left), len(right))
    if left[:common] == right[:common]:
        return "compatible"
    if (
        first.branch.exclusive_group
        and first.branch.exclusive_group == second.branch.exclusive_group
    ):
        return "incompatible"
    return "unknown"
