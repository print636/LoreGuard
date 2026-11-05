"""Private, immutable replay of admitted draft observations and match decisions.

Only the validated character pipeline writes here. This repository checks source
identity and integrity, never character truth, matching semantics, or model output.
It deliberately has no logging and no client-facing write endpoint.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from collections.abc import Sequence
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .db import (
    AnalysisRunCharacterTraitInputRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    CharacterObservationRecordingRow,
    CharacterObservationRow,
)
from .domain import EvidenceSpan
from .character_draft_actor_review import (
    TargetBoundDraftReviewTarget, TargetBoundDraftProposal, build_draft_actor_clause_index,
    _exact_actor_occurrences, _target_bound_proposal_id, target_bound_draft_target_digest,
)
from .character_scope_review import ScopeReviewSourceIdentity
from .config import (
    CLAIM_RESOLUTION_POLICY_V1, HABIT_ACTION_SELECTION_POLICY_V1,
    HABIT_OBJECT_CHAIN_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2, SELECTED_CLAIM_RESOLUTION_POLICY_V3,
)
from .time_utils import utc_now_naive


SCHEMA_VERSION = "character-observations-v1"
MAX_TRACKED_BASELINES = 256
RecordingTerminalStatus = Literal["complete", "partial", "failed", "not_applicable"]
MatchReason = Literal[
    "matched", "unmatched_character_alias", "ambiguous_character_alias",
    "no_confirmed_baseline", "character_mismatch", "dimension_mismatch",
    "comparison_identity_unavailable", "comparison_mismatch",
    "axis_direction_ambiguous", "source_scope_unavailable",
    "lower_authority_shadowed", "release_unknown", "release_inapplicable",
    "scope_unknown", "scope_incompatible", "observation_limit",
    "baseline_not_evaluated",
    "approved_axis_binding_mismatch", "trait_identity_mismatch",
    "object_identity_unavailable", "author_alignment_required",
    "scoped_axis_review_disabled", "equivalent_axis_collapsed",
    "lower_authority_baseline_shadowed", "baseline_limit",
]


class ObservationMatchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    baseline_input_id: str | None = Field(default=None, max_length=36)
    status: Literal["matched", "not_matched", "unresolved"]
    reason_code: MatchReason
    match_method: Literal["trait_identity", "target_bound", "comparison_identity", "preference_modifier", "approved_axis"] | None = None
    identity_match: bool | None = None


class ObservationRecordingCoverage(BaseModel):
    """Mutually exclusive omitted baseline counts, never free-form diagnostics."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    baseline_not_evaluated_count: int = Field(default=0, ge=0, le=5000)
    excluded_baseline_counts: dict[MatchReason, int] = Field(default_factory=dict, max_length=32)

    @model_validator(mode="after")
    def valid_counts(self):
        if any(type(count) is not int or count < 0 or count > 5000 for count in self.excluded_baseline_counts.values()) or self.omitted_baseline_count > 5000:
            raise ValueError("character observation coverage counts invalid")
        return self

    @property
    def omitted_baseline_count(self) -> int:
        return self.baseline_not_evaluated_count + sum(self.excluded_baseline_counts.values())


class SelectedActionOrigin(BaseModel):
    """Private provenance from an actually admitted V16 review."""
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    proposal_id: str = Field(pattern=r"^tdp_[a-f0-9]{32}$")
    selection_digest: str = Field(pattern=r"^[a-f0-9]{64}$")
    fact_clause_id: str = Field(pattern=r"^L[1-9][0-9]{0,7}:A[1-9][0-9]{0,2}$")
    path_ids: list[Annotated[str, Field(pattern=r"^L[1-9][0-9]{0,7}:A[1-9][0-9]{0,2}$")]] = Field(min_length=1, max_length=3)
    window_ids: list[Annotated[str, Field(pattern=r"^L[1-9][0-9]{0,7}:A[1-9][0-9]{0,2}$")]] = Field(min_length=1, max_length=3)


class ValidatedCharacterObservation(BaseModel):
    """Explicit allowlist supplied only after the existing admission checks."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    observation_id: str = Field(pattern=r"^cs_[a-f0-9]{32}$")
    source_run_input_id: str = Field(min_length=1, max_length=36)
    character: str = Field(min_length=1, max_length=128)
    resolved_character_key: str | None = Field(default=None, max_length=128)
    dimension: str = Field(min_length=1, max_length=80)
    trait_key: str = Field(min_length=1, max_length=200)
    key_object: str = Field(default="", max_length=200)
    comparison_key: str | None = Field(default=None, max_length=500)
    statement: str = Field(min_length=1, max_length=1000)
    polarity: str = Field(min_length=1, max_length=24)
    stability: str = Field(min_length=1, max_length=24)
    observation_kind: str = Field(min_length=1, max_length=40)
    context: str = Field(default="", max_length=1000)
    target_bound_target_digest: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    target_bound_object_relation: Literal["matches_target", "broader", "narrower", "not_applicable"] | None = None
    target_bound_target: TargetBoundDraftReviewTarget | None = None
    target_bound_target_ordinal: int | None = Field(default=None, ge=1, le=12)
    target_bound_object_basis_id: str | None = Field(default=None, pattern=r"^L[1-9][0-9]{0,7}:A[1-9][0-9]{0,2}$")
    selected_action_origin: SelectedActionOrigin | None = None
    evidence: EvidenceSpan
    matches: tuple[ObservationMatchDecision, ...] = Field(min_length=1, max_length=MAX_TRACKED_BASELINES)


class CharacterObservationIntegrityError(ValueError):
    """A fixed message only: do not interpolate private payloads or exceptions."""


def validate_selected_action_origin(origin, *, source, target, evidence, observation_id, statement, key_object):
    """Rebuild reference structure from frozen text, without deciding semantics."""
    from .character_action_selection import _contiguous_window
    from .character_trait_extraction import _target_bound_signal_id

    try:
        proof = SelectedActionOrigin.model_validate(origin)
        if target is None or evidence.line_start != evidence.line_end:
            raise ValueError("selected origin binding")
        identity = ScopeReviewSourceIdentity(run_input_id=source.id, document_id=source.document_id,
            document_version=source.document_version, content_sha256=source.content_sha256)
        index = build_draft_actor_clause_index(source.content, identity,
            line_start=evidence.line_start, line_end=evidence.line_end)
        window = tuple(index.resolve(value) for value in proof.window_ids)
        if (any(value is None for value in window) or len(set(proof.window_ids)) != len(window)
                or not _contiguous_window(window, index.lines[0].text)
                or proof.fact_clause_id != proof.path_ids[-1]):
            raise ValueError("selected origin window")
        first, last = proof.window_ids.index(proof.path_ids[0]), proof.window_ids.index(proof.path_ids[-1])
        if proof.window_ids[first:last + 1] != proof.path_ids or not _exact_actor_occurrences(window[first].text, target):
            raise ValueError("selected origin path")
        fact = index.resolve(proof.fact_clause_id)
        canonical_statement = fact.text if _exact_actor_occurrences(fact.text, target) else target.character + fact.text
        proposal = TargetBoundDraftProposal(
            source_sha256=source.content_sha256, line_start=evidence.line_start, line_end=evidence.line_end,
            evidence=index.lines[0].text, fact_clause_id=proof.fact_clause_id,
            actor_anchor_id=proof.path_ids[0], path_ids=tuple(proof.path_ids), binding_kind="selected_action_path",
            target_digest=target_bound_draft_target_digest(target), canonical_statement=canonical_statement,
        )
        if (_target_bound_proposal_id(proposal) != proof.proposal_id or statement != canonical_statement
                or not isinstance(key_object, str) or not key_object.strip()
                or _target_bound_signal_id(chunk=SimpleNamespace(document_id=source.document_id),
                    target=target, proposal=proposal, habit_event_object=key_object) != observation_id):
            raise ValueError("selected origin proposal")
        return proof
    except (ValueError, TypeError, AttributeError, IndexError):
        raise CharacterObservationIntegrityError("character observation selected origin invalid") from None


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _run(db: Session, run_id: str, project_id: str) -> AnalysisRunRow:
    run = db.get(AnalysisRunRow, run_id)
    if run is None or run.project_id != project_id:
        raise CharacterObservationIntegrityError("character observation run ownership mismatch")
    return run


def _frozen(db: Session, run_id: str, project_id: str):
    inputs = list(db.scalars(select(AnalysisRunInputRow).where(AnalysisRunInputRow.run_id == run_id).order_by(AnalysisRunInputRow.ordinal, AnalysisRunInputRow.id)))
    baselines = list(db.scalars(select(AnalysisRunCharacterTraitInputRow).where(AnalysisRunCharacterTraitInputRow.run_id == run_id).order_by(AnalysisRunCharacterTraitInputRow.ordinal, AnalysisRunCharacterTraitInputRow.id)))
    for row in inputs:
        if hashlib.sha256(row.content.encode("utf-8")).hexdigest() != row.content_sha256:
            raise CharacterObservationIntegrityError("character observation source integrity mismatch")
    for row in baselines:
        if row.project_id != project_id or not isinstance(row.payload, dict) or _hash(row.payload) != row.payload_sha256 or row.payload.get("candidate_id") != row.candidate_id:
            raise CharacterObservationIntegrityError("character observation baseline integrity mismatch")
    source_hash = _hash([_source(row) for row in inputs])
    baseline_hash = _hash([_baseline(row) for row in baselines])
    return {row.id: row for row in inputs}, {row.id: row for row in baselines}, source_hash, baseline_hash


def _source(row: AnalysisRunInputRow) -> dict:
    return {"run_input_id": row.id, "document_id": row.document_id, "document_name": row.document_name, "document_version": row.document_version, "content_sha256": row.content_sha256, "input_ordinal": row.ordinal}


def _baseline(row: AnalysisRunCharacterTraitInputRow) -> dict:
    # The payload already lives in this run's immutable frozen input table.
    # Persisting only this reference avoids copying it for every observation.
    return {"input_id": row.id, "candidate_id": row.candidate_id, "confirmation_review_id": row.confirmation_review_id, "candidate_lock_version": row.candidate_lock_version, "payload_sha256": row.payload_sha256}


class CharacterObservationRecorder:
    """Transaction-scoped recorder. Caller owns commit/rollback and lifecycle."""

    def begin(self, db: Session, *, run_id: str, project_id: str) -> CharacterObservationRecordingRow:
        _run(db, run_id, project_id)
        _, _, source_hash, baseline_hash = _frozen(db, run_id, project_id)
        header = db.get(CharacterObservationRecordingRow, run_id)
        if header is not None:
            if (header.project_id != project_id or header.schema_version != SCHEMA_VERSION or header.source_manifest_sha256 != source_hash or header.baseline_manifest_sha256 != baseline_hash):
                raise CharacterObservationIntegrityError("character observation recording identity mismatch")
            return header
        header = CharacterObservationRecordingRow(run_id=run_id, project_id=project_id, schema_version=SCHEMA_VERSION, status="in_progress", source_manifest_sha256=source_hash, baseline_manifest_sha256=baseline_hash)
        db.add(header)
        db.flush()
        return header

    def record(self, db: Session, *, run_id: str, project_id: str, observations: Sequence[ValidatedCharacterObservation]) -> None:
        header = self.begin(db, run_id=run_id, project_id=project_id)
        sources, baselines, _, _ = _frozen(db, run_id, project_id)
        pending: dict[str, tuple[dict, str, str]] = {}
        # Validate the entire batch before inserting any part of it.
        for observation in observations:
            if not isinstance(observation, ValidatedCharacterObservation):
                raise CharacterObservationIntegrityError("character observation validated input required")
            source = sources.get(observation.source_run_input_id)
            evidence = observation.evidence
            if source is None or source.document_id != evidence.document_id or source.document_name != evidence.document_name:
                raise CharacterObservationIntegrityError("character observation source binding mismatch")
            lines = source.content.splitlines()
            if (evidence.line_start < 1 or evidence.line_end < evidence.line_start or evidence.line_end > len(lines)):
                raise CharacterObservationIntegrityError("character observation evidence range mismatch")
            frozen_text = "\n".join(lines[evidence.line_start - 1:evidence.line_end])
            if not evidence.text.strip() or evidence.text not in frozen_text:
                raise CharacterObservationIntegrityError("character observation evidence text mismatch")
            decisions = []
            seen_baselines = set()
            for decision in observation.matches:
                baseline = baselines.get(decision.baseline_input_id) if decision.baseline_input_id is not None else None
                if (decision.baseline_input_id is not None and baseline is None) or decision.baseline_input_id in seen_baselines:
                    raise CharacterObservationIntegrityError("character observation baseline binding mismatch")
                if decision.status == "matched" and (baseline is None or decision.reason_code != "matched"):
                    raise CharacterObservationIntegrityError("character observation match reference required")
                seen_baselines.add(decision.baseline_input_id)
                decisions.append({"status": decision.status, "reason_code": decision.reason_code, "match_method": decision.match_method, "identity_match": decision.identity_match, "baseline": _baseline(baseline) if baseline is not None else None})
            decisions.sort(key=lambda item: (item["baseline"]["input_id"] if item["baseline"] else "", item["reason_code"]))
            payload = observation.model_dump(mode="json", exclude={"source_run_input_id", "matches", "selected_action_origin"})
            if observation.selected_action_origin is not None:
                run = _run(db, run_id, project_id)
                coverage = run.batch_coverage if isinstance(run.batch_coverage, dict) else {}
                if (run.batch_mode != "draft_review" or coverage.get("character_review_enabled") is not True
                        or coverage.get("habit_object_chain_policy") != HABIT_OBJECT_CHAIN_POLICY_V1
                        or coverage.get("claim_resolution_policy") != CLAIM_RESOLUTION_POLICY_V1
                        or coverage.get("habit_action_selection_policy") != HABIT_ACTION_SELECTION_POLICY_V1
                        or coverage.get("selected_claim_resolution_policy") not in (
                            SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2,
                            SELECTED_CLAIM_RESOLUTION_POLICY_V3)):
                    raise CharacterObservationIntegrityError("character observation selected origin unauthorized")
                validate_selected_action_origin(observation.selected_action_origin, source=source,
                    target=observation.target_bound_target, evidence=evidence,
                    observation_id=observation.observation_id, statement=observation.statement,
                    key_object=observation.key_object)
                payload["selected_action_origin"] = observation.selected_action_origin.model_dump(mode="json")
            payload.update(source=_source(source), matches=decisions)
            digest = _hash(payload)
            previous = pending.get(observation.observation_id)
            if previous is not None and previous[1] != digest:
                raise CharacterObservationIntegrityError("character observation immutable identity conflict")
            existing = db.get(CharacterObservationRow, (run_id, observation.observation_id))
            if existing is not None:
                if existing.payload_sha256 != digest or _hash(existing.payload) != digest:
                    raise CharacterObservationIntegrityError("character observation immutable identity conflict")
                continue
            if header.status != "in_progress":
                raise CharacterObservationIntegrityError("character observation recording is sealed")
            pending[observation.observation_id] = (payload, digest, source.id)
        ordinal = db.scalar(select(func.max(CharacterObservationRow.ordinal)).where(CharacterObservationRow.run_id == run_id))
        next_ordinal = 0 if ordinal is None else ordinal + 1
        for observation_id, (payload, digest, source_id) in sorted(pending.items(), key=lambda item: (item[1][0]["source"]["input_ordinal"], item[1][0]["evidence"]["line_start"], item[0])):
            db.add(CharacterObservationRow(run_id=run_id, observation_id=observation_id, source_run_input_id=source_id, ordinal=next_ordinal, payload=payload, payload_sha256=digest))
            next_ordinal += 1
        db.flush()

    def finish(self, db: Session, *, run_id: str, project_id: str, status: RecordingTerminalStatus, coverage: ObservationRecordingCoverage | None = None) -> None:
        if status not in {"complete", "partial", "failed", "not_applicable"}:
            raise CharacterObservationIntegrityError("character observation recording status invalid")
        header = self.begin(db, run_id=run_id, project_id=project_id)
        if coverage is not None:
            coverage_payload = {**coverage.model_dump(mode="json"), "omitted_baseline_count": coverage.omitted_baseline_count}
            if header.status != "in_progress" and header.coverage != coverage_payload:
                raise CharacterObservationIntegrityError("character observation recording is sealed")
            if coverage.omitted_baseline_count and status == "complete":
                status = "partial"
        else:
            coverage_payload = header.coverage
        if header.status == status:
            return
        if header.status != "in_progress":
            raise CharacterObservationIntegrityError("character observation recording is sealed")
        if status == "not_applicable" and db.scalar(select(func.count()).select_from(CharacterObservationRow).where(CharacterObservationRow.run_id == run_id)):
            raise CharacterObservationIntegrityError("character observation recording has admitted data")
        header.status = status
        header.coverage = coverage_payload
        header.completed_at = utc_now_naive()
        db.flush()


def project_character_observations(db: Session, run: AnalysisRunRow, *, offset: int = 0, limit: int = 50) -> dict:
    """Bounded stable projection. Missing history is never called a clean result."""
    if offset < 0 or not 1 <= limit <= 200:
        raise ValueError("character observation pagination invalid")
    header = db.get(CharacterObservationRecordingRow, run.id)
    count = db.scalar(select(func.count()).select_from(CharacterObservationRow).where(CharacterObservationRow.run_id == run.id)) or 0
    rows = list(db.scalars(select(CharacterObservationRow).where(CharacterObservationRow.run_id == run.id).order_by(CharacterObservationRow.ordinal, CharacterObservationRow.observation_id).offset(offset).limit(limit)))
    if header is not None and header.project_id != run.project_id:
        raise CharacterObservationIntegrityError("character observation run ownership mismatch")
    for row in rows:
        if _hash(row.payload) != row.payload_sha256:
            raise CharacterObservationIntegrityError("character observation payload integrity mismatch")
    baseline_ids = {decision["baseline"]["input_id"] for row in rows for decision in row.payload["matches"] if decision["baseline"] is not None}
    baselines = {}
    # A page may refer to the same baseline many times. Fetch each frozen
    # payload once; chunks keep SQLite parameter counts bounded as well.
    ordered_ids = sorted(baseline_ids)
    for start in range(0, len(ordered_ids), 400):
        baselines.update({row.id: row for row in db.scalars(select(AnalysisRunCharacterTraitInputRow).where(AnalysisRunCharacterTraitInputRow.run_id == run.id, AnalysisRunCharacterTraitInputRow.project_id == run.project_id, AnalysisRunCharacterTraitInputRow.id.in_(ordered_ids[start:start + 400])))})
    items = []
    for row in rows:
        decisions = []
        for decision in row.payload["matches"]:
            reference = decision["baseline"]
            if reference is not None:
                baseline = baselines.get(reference["input_id"])
                if baseline is None or _baseline(baseline) != reference or _hash(baseline.payload) != reference["payload_sha256"]:
                    raise CharacterObservationIntegrityError("character observation baseline integrity mismatch")
                public_baseline = {
                    key: value for key, value in baseline.payload.items()
                    if key != "target_support_bindings_v1"
                }
                decisions.append({**decision, "baseline": {
                    **reference, "payload": public_baseline,
                    # This digest describes the public body, while persisted
                    # observation references keep the complete private digest.
                    "payload_sha256": _hash(public_baseline),
                }})
            else:
                decisions.append(decision)
        public_payload = {key: value for key, value in row.payload.items() if key != "selected_action_origin"}
        items.append({**public_payload, "ordinal": row.ordinal, "matches": decisions})
    if header is None:
        status = "legacy_unavailable"
        if run.batch_mode == "baseline_build":
            status = "not_applicable"
    else:
        status = header.status
    incomplete_reason = None
    if header is not None and header.status == "in_progress" and run.status in {"completed", "failed", "cancelled"}:
        # A stale/cancelled worker must respect its write fence. A read may
        # explain the final run state without rewriting that worker's history.
        status = "partial" if run.status == "completed" else "failed"
        incomplete_reason = "run_ended_before_recording_completed"
    return {
        "schema_version": SCHEMA_VERSION, "run_id": run.id, "project_id": run.project_id,
        "match_semantics": "matched means selected for baseline comparison; it is not a consistency verdict",
        "recording": {"status": status, "stored_status": header.status if header else None, "incomplete_reason": incomplete_reason, "coverage": header.coverage if header else {}, "recorded_at": header.created_at.isoformat() if header else None, "completed_at": header.completed_at.isoformat() if header and header.completed_at else None, "source_manifest_sha256": header.source_manifest_sha256 if header else None, "baseline_manifest_sha256": header.baseline_manifest_sha256 if header else None, "observation_count": count},
        "page": {"offset": offset, "limit": limit, "total_items": count, "has_more": offset + len(rows) < count},
        "items": items,
    }
