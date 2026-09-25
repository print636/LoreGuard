"""Server-derived, versioned sub-line evidence for reviewed character traits.

This metadata supplements, but never replaces, the immutable full-line
candidate evidence or the author confirmation contract.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from .character_trait_extraction import CharacterSignalChunk, _assertion_index_v1


SUPPORT_BINDINGS_SCHEMA_V1 = "character-support-bindings-v1"
SUPPORT_INDEX_VERSION_V1 = "assertion-index-v1"
MAX_SUPPORT_BINDINGS = 48
MAX_SUPPORT_BINDINGS_BYTES = 32_768
_SUPPORT_ID = r"^L[1-9][0-9]{0,7}:A[1-9][0-9]{0,2}$"


class TraitSupportRef(BaseModel):
    """Server-internal candidate-to-frozen-evidence reference."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    evidence_index: int = Field(ge=0, le=11)
    support_id: str = Field(pattern=_SUPPORT_ID)
    actor_anchor_id: str | None = Field(default=None, pattern=_SUPPORT_ID)
    label_anchor_id: str | None = Field(default=None, pattern=_SUPPORT_ID)
    scope_relation: Literal[
        "local", "same_actor_continuation", "labelled_elaboration"
    ]


def _canonical_bytes(value: dict) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def support_bindings_sha256(value: dict) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _span(clause, *, role: str) -> dict:
    return {
        "support_id": clause.support_id,
        "start_offset": clause.start_offset,
        "end_offset": clause.end_offset,
        "role": role,
    }


def bind_support_refs(
    refs: list[TraitSupportRef],
    *,
    evidence: list[dict],
    frozen_by_id: Mapping[str, object],
) -> dict:
    """Recompute every codepoint span from the exact frozen input.

    The caller must have already verified full-line evidence. Model-authored
    IDs are never accepted as coordinates or source authority.
    """

    if not refs or len(refs) > MAX_SUPPORT_BINDINGS:
        raise ValueError("candidate support reference count is invalid")
    bound: dict[tuple, dict] = {}
    for ref in refs:
        if not isinstance(ref, TraitSupportRef) or ref.evidence_index >= len(evidence):
            raise ValueError("candidate support reference is invalid")
        item = evidence[ref.evidence_index]
        input_id = item.get("input_id")
        snapshot = frozen_by_id.get(input_id) if isinstance(input_id, str) else None
        if snapshot is None or getattr(snapshot, "id", None) != input_id:
            raise ValueError("candidate support source is missing")
        try:
            frozen_content = snapshot.content
            lines = frozen_content.splitlines()
            line_number = item["line_start"]
            raw_line = lines[line_number - 1]
            expected_hash = hashlib.sha256(frozen_content.encode("utf-8")).hexdigest()
            if (
                item["line_end"] != line_number
                or line_number < 1
                or getattr(snapshot, "document_id", None) != item["document_id"]
                or getattr(snapshot, "document_version", None) != item["document_version"]
                or getattr(snapshot, "content_sha256", None) != item["content_sha256"]
                or expected_hash != item["content_sha256"]
                or raw_line != item["text"]
                or len(raw_line) > 8_000
            ):
                raise ValueError("candidate support source does not match frozen evidence")
            index = _assertion_index_v1(CharacterSignalChunk(
                document_id=item["document_id"],
                document_name=item["document_name"],
                content=raw_line,
                global_line_start=line_number,
                source_kind="formal_character_profile",
            ))
        except (AttributeError, IndexError, KeyError, TypeError, UnicodeEncodeError) as exc:
            raise ValueError("candidate support source is invalid") from exc
        target = index.resolve(ref.support_id)
        actor = index.resolve(ref.actor_anchor_id) if ref.actor_anchor_id else None
        label = index.resolve(ref.label_anchor_id) if ref.label_anchor_id else None
        if target is None or target.line_number != line_number:
            raise ValueError("candidate support target is invalid")
        if (ref.actor_anchor_id is not None and actor is None) or (
            ref.label_anchor_id is not None and label is None
        ):
            raise ValueError("candidate support anchor is invalid")
        for anchor in (actor, label):
            if anchor is not None and (
                anchor.line_number != line_number or anchor.start_offset > target.start_offset
            ):
                raise ValueError("candidate support anchor is outside the target line")
        if actor is not None and actor.start_offset == target.start_offset:
            raise ValueError("candidate support actor anchor must precede target")
        if ref.scope_relation == "local":
            valid_relation = actor is None and label in (None, target)
        elif ref.scope_relation == "same_actor_continuation":
            valid_relation = actor is not None and label in (None, target)
        else:
            valid_relation = label is not None and label != target
        if not valid_relation:
            raise ValueError("candidate support relation is invalid")
        # Only the server-determined anchor-to-target chain is context. The
        # reviewer may have supplied extra same-line basis IDs; ignore those.
        anchors = [anchor for anchor in (actor, label) if anchor is not None and anchor != target]
        context = []
        if anchors:
            first = min(anchor.start_offset for anchor in anchors)
            for clause in index.clauses:
                if first <= clause.start_offset < target.start_offset:
                    role = (
                        "actor_anchor" if actor == clause else
                        "label_anchor" if label == clause else "bridge"
                    )
                    context.append(_span(clause, role=role))
        item_bound = {
            "evidence_index": ref.evidence_index,
            "support_id": ref.support_id,
            "target": _span(target, role="target"),
            "actor_anchor_id": ref.actor_anchor_id,
            "label_anchor_id": ref.label_anchor_id,
            "scope_relation": ref.scope_relation,
            "context": context,
        }
        key = (
            ref.evidence_index, target.start_offset, ref.support_id,
            ref.actor_anchor_id or "", ref.label_anchor_id or "", ref.scope_relation,
        )
        bound[key] = item_bound
    payload = {
        "schema_version": SUPPORT_BINDINGS_SCHEMA_V1,
        "index_version": SUPPORT_INDEX_VERSION_V1,
        "bindings": [bound[key] for key in sorted(bound)],
    }
    if len(_canonical_bytes(payload)) > MAX_SUPPORT_BINDINGS_BYTES:
        raise ValueError("candidate support bindings are too large")
    return payload


def verify_stored_support_bindings(
    payload: object,
    digest: object,
    *,
    evidence: list[dict],
    frozen_by_id: Mapping[str, object],
) -> dict | None:
    """Return verified metadata; NULL denotes a legacy candidate."""

    if payload is None and digest is None:
        return None
    if not isinstance(payload, dict) or not isinstance(digest, str):
        raise ValueError("candidate support bindings are malformed")
    try:
        if (
            len(_canonical_bytes(payload)) > MAX_SUPPORT_BINDINGS_BYTES
            or support_bindings_sha256(payload) != digest
            or set(payload) != {"schema_version", "index_version", "bindings"}
            or payload["schema_version"] != SUPPORT_BINDINGS_SCHEMA_V1
            or payload["index_version"] != SUPPORT_INDEX_VERSION_V1
            or not isinstance(payload["bindings"], list)
            or not payload["bindings"]
            or len(payload["bindings"]) > MAX_SUPPORT_BINDINGS
        ):
            raise ValueError("candidate support bindings failed validation")
        refs = [TraitSupportRef.model_validate({
            name: row[name] for name in TraitSupportRef.model_fields
        }, strict=True) for row in payload["bindings"]]
        rebound = bind_support_refs(refs, evidence=evidence, frozen_by_id=frozen_by_id)
        if rebound != payload:
            raise ValueError("candidate support bindings do not match frozen input")
        return rebound
    except (KeyError, TypeError, UnicodeEncodeError) as exc:
        raise ValueError("candidate support bindings are malformed") from exc
