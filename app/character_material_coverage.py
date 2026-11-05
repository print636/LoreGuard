"""Content-free, final-state material gaps for an already reviewed event."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .narrative_context import payload_sha256
from .character_claim_resolution import ClaimPackageState, PendingMaterialClaim, claim_resolution_target_sha256

MATERIAL_REVIEW_KEY = "material_review_v1"
MATERIAL_REVIEW_SCHEMA = "character-material-review-v1"
MAX_MATERIAL_GAPS = 12
MAX_PACKAGE_GAPS = 32
MaterialReason = Literal[
    "actor_unverified", "statement_unverified", "object_unverified", "source_unverified",
    "record_unverified", "response_unavailable", "processing_incomplete",
]
MaterialPhase = Literal["primary_extraction", "targeted_recall", "targeted_verification", "case_coverage"]


class DraftMaterialGap(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    reason: MaterialReason
    claim_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    text_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


def material_reason(reason: str) -> MaterialReason:
    if reason == "character_support":
        return "actor_unverified"
    if reason == "statement_support":
        return "statement_unverified"
    if reason == "key_object_support":
        return "object_unverified"
    if reason in {"evidence_mismatch", "source_line_range"}:
        return "source_unverified"
    if reason in {"invalid_json", "response_too_large"}:
        return "response_unavailable"
    return "record_unverified"


@dataclass
class DraftMaterialTracker:
    source: object
    content: str | None
    chunk: object
    gaps: tuple[DraftMaterialGap, ...] = ()
    resolved: set[str] = field(default_factory=set)
    omitted: int = 0
    captured: bool = False
    pending_response: bool = False
    claim_resolution_enabled: bool = False
    claim_state: ClaimPackageState = field(default_factory=ClaimPackageState)
    final_review_reasons: dict[int, MaterialReason] = field(default_factory=dict)

    def remember_review(self, response: str, outcome) -> None:
        if not self.claim_resolution_enabled:
            return
        self.final_review_reasons.update(getattr(outcome, "material_reasons", ()))
        try:
            records = json.loads(response)["records"]
        except (ValueError, KeyError, TypeError):
            return
        for proof in getattr(outcome, "claim_certificates", ()):
            index = proof["record_index"]
            if not isinstance(records, list) or not 0 <= index < len(records) or not self._location(records[index]):
                continue
            key = payload_sha256({"source": self.source.content_sha256, "document": self.source.document_id,
                                  "record": records[index]})
            self.claim_state.certificates.append({**proof, "primary_claim_sha256": key})

    def _location(self, raw: object) -> dict:
        if not isinstance(raw, dict) or not isinstance(self.content, str) or self.source is None:
            return {}
        if (self.source.document_id != self.chunk.document_id
                or hashlib.sha256(self.content.encode()).hexdigest() != self.source.content_sha256):
            return {}
        start, end = raw.get("source_line_start"), raw.get("source_line_end")
        lines = self.content.splitlines()
        if not (type(start) is int and type(end) is int
                and self.chunk.global_line_start <= start <= end <= self.chunk.global_line_end
                and 1 <= start <= end <= len(lines)):
            return {}
        text = "\n".join(lines[start - 1:end])
        if not text or raw.get("evidence") != text:
            return {}
        return dict(line_start=start, line_end=end, text_sha256=hashlib.sha256(text.encode()).hexdigest())

    def capture(self, response: str, validation, *, package_attempt: int = 1) -> None:
        """Only the post-correction validation of this attempt is current."""
        self.captured = True
        self.pending_response = False
        try:
            records = json.loads(response)["records"] if validation.parsed else []
        except (ValueError, TypeError, KeyError):
            records = []
        if not isinstance(records, list):
            records = []
        keys = {}
        locations = {}
        for ordinal, raw in enumerate(records):
            location = self._location(raw)
            if location:
                locations[ordinal] = location
                try:
                    keys[ordinal] = payload_sha256({
                        "source": self.source.content_sha256, "document": self.source.document_id,
                        "record": raw,
                    })
                except (TypeError, ValueError, UnicodeError):
                    pass
        trace = validation.draft_trace_attempt
        self.resolved = set()
        if trace is not None:
            self.resolved.update(
                keys[event.record_ordinal - 1] for event in trace.events
                if event.outcome == "accepted" and event.record_ordinal - 1 in keys
            )
        failures = validation.failures
        remaining = {
            gap.claim_digest or payload_sha256(gap.model_dump()): gap
            for gap in self.gaps if gap.claim_digest not in self.resolved
        }
        for failure in failures:
            gap = DraftMaterialGap(
                reason=self.final_review_reasons.get(failure.record_index, material_reason(failure.reason)),
                claim_digest=keys.get(failure.record_index),
                **locations.get(failure.record_index, {}),
            )
            identity = gap.claim_digest or payload_sha256(gap.model_dump())
            if identity in remaining or len(remaining) < MAX_PACKAGE_GAPS:
                remaining[identity] = gap
            else:
                self.omitted += 1
        self.gaps = tuple(remaining.values())
        if self.claim_resolution_enabled:
            self.claim_state.enumeration_complete = bool(
                validation.parsed and trace is not None and len(trace.events) == len(records)
                and len(keys) == len(records) and not any(f.record_index is None for f in failures)
            )
            for deferred in validation.target_bound_draft_deferred:
                key = keys.get(deferred.record_index)
                if (key is None or not deferred.claim_review or not deferred.object_review
                        or deferred.target.dimension != "contextual_behavior" or not deferred.target.habit_contexts):
                    continue
                # The target model is already frozen on the generated proposal;
                # constructing its review identity does not adjudicate the claim.
                from .character_trait_extraction import _target_bound_review_target
                target = _target_bound_review_target(deferred.target, target_ordinal=deferred.target_ordinal)
                self.claim_state.pending[key] = PendingMaterialClaim(
                    key, deferred, package_attempt, claim_resolution_target_sha256(target),
                )
            self.claim_state.pending = {k: v for k, v in self.claim_state.pending.items() if k not in self.resolved}
            self.final_review_reasons.clear()

    def finish(self, result):
        if result.diagnostics.outcome == "completed":
            gaps, omitted = (), 0
        else:
            gaps, omitted = self.gaps, self.omitted
            if self.pending_response:
                gaps = (*gaps, DraftMaterialGap(reason="response_unavailable"))
            if not gaps and not omitted:
                reason = "processing_incomplete" if self.captured else "response_unavailable"
                gaps = (DraftMaterialGap(reason=reason),)
        self.claim_state.gaps, self.claim_state.omitted_count = gaps, omitted
        if self.pending_response:
            self.claim_state.enumeration_complete = False
        return result.model_copy(update={
            "material_gaps": gaps, "material_gaps_omitted": omitted,
            "material_resolved_claims": tuple(sorted(self.resolved)),
            **({"claim_resolution_state": self.claim_state} if self.claim_resolution_enabled else {}),
        })


def material_review_payload(
    *, case_id: str, candidate_id: str, reviewed_events: dict,
    gaps: list[dict], omitted_count: int = 0,
) -> dict:
    """Bind source coordinates and closed enums to this review receipt."""
    rows = []
    for gap in gaps[:MAX_MATERIAL_GAPS]:
        data = dict(gap)
        data["id"] = "mg_" + payload_sha256(data)
        rows.append(data)
    payload = dict(
        schema_version=MATERIAL_REVIEW_SCHEMA, case_id=case_id,
        confirmed_candidate_id=candidate_id,
        reviewed_events_sha256=payload_sha256(reviewed_events),
        gaps=rows,
        omitted_count=omitted_count + sum(gap.get("count", 1) for gap in gaps[MAX_MATERIAL_GAPS:]),
        total_count=omitted_count + sum(gap.get("count", 1) for gap in gaps),
    )
    return {**payload, "payload_sha256": payload_sha256(payload)}


@dataclass
class MaterialGapLedger:
    """Final unresolved packages, bounded independently of diagnostic history."""
    entries: list[dict] = field(default_factory=list)
    resolved: set[str] = field(default_factory=set)
    omitted_count: int = 0
    scoped_resolution: bool = False
    resolved_scopes: set[tuple[str, str, str]] = field(default_factory=set)

    def add(self, result, source, phase: MaterialPhase, target_key=None) -> None:
        if not self.scoped_resolution:
            self.resolved.update(getattr(result, "material_resolved_claims", ()))
        gaps = getattr(result, "material_gaps", None)
        if gaps is None:
            gaps = () if result.diagnostics.outcome == "completed" else (
                DraftMaterialGap(reason="processing_incomplete"),
            )
        self.omitted_count += getattr(result, "material_gaps_omitted", 0)
        for gap in gaps:
            if gap.claim_digest is not None:
                self.resolved.discard(gap.claim_digest)
                self.entries = [row for row in self.entries if row.get("claim_digest") != gap.claim_digest
                                or self.scoped_resolution and (
                                    row["source_input_id"] != source.input_id or row["phase"] != phase
                                    or row.get("target_key") != target_key)]
            if len(self.entries) >= 128:
                self.omitted_count += 1
                continue
            self.entries.append(dict(
                phase=phase, source_input_id=source.input_id,
                source_document_id=source.document.id,
                source_content_sha256=source.content_sha256, target_key=target_key,
                **gap.model_dump(exclude_none=True),
            ))

    def for_case(self, *, case_id: str, candidate_id: str, reviewed_events: dict,
                 target_key, draft_document_ids: set[str], case_reasons: set[str]) -> dict:
        current = [row for row in reviewed_events["evidence_refs"] if row["role"] == "C"]
        grouped = {}
        seen_claims = set()
        for entry in reversed(self.entries) if self.scoped_resolution else self.entries:
            if (entry["source_document_id"] not in draft_document_ids
                    or entry.get("target_key") is not None and entry["target_key"] != target_key
                    or entry.get("claim_digest") in self.resolved
                    or (entry["source_input_id"], candidate_id, entry.get("claim_digest")) in self.resolved_scopes):
                continue
            claim_key = (entry["source_input_id"], entry.get("claim_digest"))
            if self.scoped_resolution and claim_key[1] is not None:
                if claim_key in seen_claims:
                    continue
                seen_claims.add(claim_key)
            data = {k: v for k, v in entry.items() if k not in {"claim_digest", "target_key"}}
            located = "line_start" in data and "text_sha256" in data
            data["location"] = "located" if located else "unknown"
            data["relation"] = "unknown"
            if located:
                data["relation"] = "overlap" if any(
                    row.get("source_input_id_sha256") == hashlib.sha256(data["source_input_id"].encode()).hexdigest()
                    and row["line_start"] <= data["line_end"] and data["line_start"] <= row["line_end"]
                    for row in current
                ) else "elsewhere"
            identity = payload_sha256(data)
            if identity not in grouped:
                grouped[identity] = {**data, "count": 0}
            grouped[identity]["count"] += 1
        explained_reasons = {"case_draft_package_incomplete", "targeted_package_incomplete",
                             "semantic_binding_unresolved"}
        if case_reasons and (not grouped or case_reasons - explained_reasons):
            grouped["case"] = dict(
                phase="case_coverage", reason="processing_incomplete",
                location="unknown", relation="unknown", count=1,
            )
        return material_review_payload(
            case_id=case_id, candidate_id=candidate_id, reviewed_events=reviewed_events,
            gaps=list(grouped.values()), omitted_count=self.omitted_count,
        )
