"""Content-free claim receipts bound to this run's immutable observations.

The extraction stage establishes semantic equivalence. This boundary checks
only storage identity and integrity; it cannot infer a resolution from a quote.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select

from .character_claim_resolution import (
    CLAIM_RESOLUTION_SCHEMA, MAX_CLAIM_RESOLUTION_RECEIPTS,
    MAX_CLAIM_RESOLUTION_SCOPES, claim_resolution_target_sha256,
)
from .character_draft_actor_review import TargetBoundDraftReviewTarget
from .character_observations import validate_selected_action_origin, CharacterObservationIntegrityError
from .domain import EvidenceSpan
from .config import (
    CLAIM_RESOLUTION_POLICY_V1, HABIT_ACTION_SELECTION_POLICY_V1,
    HABIT_OBJECT_CHAIN_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2, SELECTED_CLAIM_RESOLUTION_POLICY_V3,
)
from .db import (
    AnalysisRunCharacterTraitInputRow, AnalysisRunInputRow, AnalysisRunRow,
    CharacterObservationRow,
)
from .narrative_context import payload_sha256


Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Count = Annotated[int, Field(ge=0, le=100000)]


class _Binding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source_input_id: str = Field(min_length=1, max_length=36)
    source_document_id: str = Field(min_length=1, max_length=36)
    source_content_sha256: Digest
    confirmed_candidate_id: str = Field(min_length=1, max_length=36)
    confirmed_payload_sha256: Digest
    target_sha256: Digest


class _Receipt(_Binding):
    id: str = Field(pattern=r"^cr_[a-f0-9]{64}$")
    method: Literal["exact_claim", "original_claim_recheck"]
    primary_claim_sha256: Digest
    review_claim_sha256: Digest
    request_sha256: Digest
    proposal_id: str = Field(pattern=r"^tdp_[a-f0-9]{32}$")
    observation_id: str = Field(pattern=r"^cs_[a-f0-9]{32}$")
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    text_sha256: Digest
    observation_statement_sha256: Digest


class _SelectedReceipt(_Receipt):
    method: Literal["selected_claim_recheck", "selected_claim_recheck_v2"]
    origin_request_sha256: Digest
    origin_observation_id: str = Field(pattern=r"^cs_[a-f0-9]{32}$")
    fact_clause_id: str = Field(pattern=r"^L[1-9][0-9]{0,7}:A[1-9][0-9]{0,2}$")


class _Scope(_Binding):
    chunk_line_start: int = Field(ge=1)
    chunk_line_end: int = Field(ge=1)
    status: Literal["complete", "partial"]
    claim_count: Count
    resolved_count: Count
    unknown_count: Count
    omitted_count: Count
    receipt_ids: list[str] = Field(max_length=MAX_CLAIM_RESOLUTION_RECEIPTS)


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["character-claim-resolution-v1"]
    receipts: list[Annotated[_Receipt | _SelectedReceipt, Field(discriminator="method")]] = Field(max_length=MAX_CLAIM_RESOLUTION_RECEIPTS)
    final_scopes: list[_Scope] = Field(max_length=MAX_CLAIM_RESOLUTION_SCOPES)
    omitted_count: Count
    payload_sha256: Digest


class ClaimResolutionIntegrityError(ValueError):
    """Fixed messages only: private payloads must never enter diagnostics."""


def _require(condition: bool) -> None:
    if not condition:
        raise ClaimResolutionIntegrityError("character claim resolution integrity mismatch")


def _text_sha256(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _validate_origin(db, *, run, item: _SelectedReceipt, source, baseline, final_value):
    origin = db.get(CharacterObservationRow, (run.id, item.origin_observation_id))
    _require(origin is not None and origin.source_run_input_id == source.id)
    value = origin.payload
    _require(isinstance(value, dict) and payload_sha256(value) == origin.payload_sha256)
    stored_source = value.get("source")
    _require(isinstance(stored_source, dict) and value.get("observation_id") == item.origin_observation_id
        and stored_source.get("run_input_id") == source.id
        and stored_source.get("document_id") == source.document_id
        and stored_source.get("content_sha256") == source.content_sha256)
    try:
        evidence = EvidenceSpan.model_validate(value.get("evidence"))
        target = TargetBoundDraftReviewTarget.model_validate_json(json.dumps(value.get("target_bound_target")))
    except ValidationError:
        raise ClaimResolutionIntegrityError("character claim resolution origin invalid") from None
    lines = source.content.splitlines()
    _require(
        evidence.document_id == source.document_id and evidence.document_name == source.document_name
        and 1 <= evidence.line_start <= evidence.line_end <= len(lines)
        and bool(evidence.text.strip()) and evidence.text in "\n".join(lines[evidence.line_start - 1:evidence.line_end])
        and claim_resolution_target_sha256(target) == item.target_sha256
        and isinstance(value.get("matches"), list)
    )
    _require(any(isinstance(match, dict) and isinstance(match.get("baseline"), dict)
        and match["baseline"].get("input_id") == baseline.id
        and match["baseline"].get("candidate_id") == baseline.candidate_id
        and match["baseline"].get("payload_sha256") == baseline.payload_sha256
        for match in value["matches"]))
    try:
        proof = validate_selected_action_origin(value.get("selected_action_origin"), source=source,
            target=target, evidence=evidence, observation_id=item.origin_observation_id,
            statement=value.get("statement"), key_object=value.get("key_object"))
        validate_selected_action_origin(proof, source=source, target=target,
            evidence=EvidenceSpan.model_validate(final_value["evidence"]), observation_id=item.observation_id,
            statement=final_value.get("statement"), key_object=final_value.get("key_object"))
    except CharacterObservationIntegrityError:
        raise ClaimResolutionIntegrityError("character claim resolution origin invalid") from None
    _require(proof.request_sha256 == item.origin_request_sha256 and proof.fact_clause_id == item.fact_clause_id)


def validate_claim_resolution_payload(db, *, run: AnalysisRunRow, payload: object) -> dict | None:
    """Validate before durable publication; missing historical data stays missing."""
    if payload is None:
        return None
    try:
        parsed = _Payload.model_validate(payload)
    except ValidationError:
        raise ClaimResolutionIntegrityError("character claim resolution shape invalid") from None
    coverage = run.batch_coverage if isinstance(run.batch_coverage, dict) else {}
    _require(
        run.batch_mode == "draft_review"
        and coverage.get("character_review_enabled") is True
        and coverage.get("habit_object_chain_policy") == HABIT_OBJECT_CHAIN_POLICY_V1
        and coverage.get("claim_resolution_policy") == CLAIM_RESOLUTION_POLICY_V1
        and parsed.schema_version == CLAIM_RESOLUTION_SCHEMA
        and parsed.payload_sha256 == payload_sha256(parsed.model_dump(exclude={"payload_sha256"}))
    )
    sources = {row.id: row for row in db.scalars(select(AnalysisRunInputRow).where(AnalysisRunInputRow.run_id == run.id)).all()}
    baselines = {row.candidate_id: row for row in db.scalars(select(AnalysisRunCharacterTraitInputRow).where(
        AnalysisRunCharacterTraitInputRow.run_id == run.id,
        AnalysisRunCharacterTraitInputRow.project_id == run.project_id,
    )).all()}

    def bound(item: _Binding):
        source = sources.get(item.source_input_id)
        baseline = baselines.get(item.confirmed_candidate_id)
        _require(source is not None and baseline is not None)
        _require(
            source.document_id == item.source_document_id
            and source.content_sha256 == item.source_content_sha256 == _text_sha256(source.content)
            and isinstance(baseline.payload, dict)
            and baseline.payload_sha256 == item.confirmed_payload_sha256 == payload_sha256(baseline.payload)
            and baseline.payload.get("candidate_id") == item.confirmed_candidate_id
        )
        return source, baseline

    receipts = {}
    for item in parsed.receipts:
        _require(item.id not in receipts and item.id == "cr_" + payload_sha256(item.model_dump(exclude={"id"})))
        source, baseline = bound(item)
        observation = db.get(CharacterObservationRow, (run.id, item.observation_id))
        _require(observation is not None and observation.source_run_input_id == source.id)
        value = observation.payload
        _require(isinstance(value, dict) and payload_sha256(value) == observation.payload_sha256)
        evidence = value.get("evidence")
        stored_source = value.get("source")
        _require(isinstance(evidence, dict) and isinstance(stored_source, dict))
        lines = source.content.splitlines()
        _require(
            value.get("observation_id") == item.observation_id
            and stored_source.get("run_input_id") == source.id
            and stored_source.get("document_id") == source.document_id
            and stored_source.get("content_sha256") == source.content_sha256
            and evidence.get("document_id") == source.document_id
            and evidence.get("document_name") == source.document_name
            and evidence.get("line_start") == item.line_start
            and evidence.get("line_end") == item.line_end
            and 1 <= item.line_start <= item.line_end <= len(lines)
            and isinstance(evidence.get("text"), str) and bool(evidence["text"].strip())
            and evidence["text"] in "\n".join(lines[item.line_start - 1:item.line_end])
            and _text_sha256(evidence["text"]) == item.text_sha256
            and isinstance(value.get("statement"), str)
            and _text_sha256(value["statement"]) == item.observation_statement_sha256
        )
        try:
            target = TargetBoundDraftReviewTarget.model_validate_json(json.dumps(value.get("target_bound_target")))
        except ValidationError:
            raise ClaimResolutionIntegrityError("character claim resolution target invalid") from None
        _require(claim_resolution_target_sha256(target) == item.target_sha256)
        _require(isinstance(value.get("matches"), list))
        _require(any(
            isinstance(match, dict) and isinstance(match.get("baseline"), dict)
            and match["baseline"].get("input_id") == baseline.id
            and match["baseline"].get("candidate_id") == baseline.candidate_id
            and match["baseline"].get("payload_sha256") == baseline.payload_sha256
            for match in value.get("matches", ())
        ))
        if isinstance(item, _SelectedReceipt):
            _require(
                coverage.get("habit_action_selection_policy") == HABIT_ACTION_SELECTION_POLICY_V1
                and coverage.get("selected_claim_resolution_policy") in {
                    "selected_claim_recheck": (SELECTED_CLAIM_RESOLUTION_POLICY_V1,),
                    "selected_claim_recheck_v2": (SELECTED_CLAIM_RESOLUTION_POLICY_V2, SELECTED_CLAIM_RESOLUTION_POLICY_V3),
                }[item.method]
            )
            _validate_origin(db, run=run, item=item, source=source, baseline=baseline, final_value=value)
        receipts[item.id] = item

    seen_scopes = set()
    for item in parsed.final_scopes:
        source, _ = bound(item)
        _require(1 <= item.chunk_line_start <= item.chunk_line_end <= len(source.content.splitlines()))
        key = (source.id, item.confirmed_candidate_id, item.target_sha256, item.chunk_line_start, item.chunk_line_end)
        _require(key not in seen_scopes and len(set(item.receipt_ids)) == len(item.receipt_ids))
        seen_scopes.add(key)
        linked = [receipts.get(receipt_id) for receipt_id in item.receipt_ids]
        _require(all(receipt is not None and all(
            getattr(receipt, field) == getattr(item, field) for field in _Binding.model_fields
        ) for receipt in linked))
        _require(
            item.resolved_count == len({receipt.primary_claim_sha256 for receipt in linked})
            and item.resolved_count + item.unknown_count + item.omitted_count <= item.claim_count
        )
        if item.status == "complete":
            _require(
                item.claim_count > 0 and item.claim_count == item.resolved_count
                and item.unknown_count == item.omitted_count == parsed.omitted_count == 0
            )
    return parsed.model_dump()
