"""Validate private explanation references against this run's durable sources."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from .character_explanation_assertions import (
    AssertionRef, ExplanationAssertionBinding, MAX_ASSERTION_RECEIPTS,
    payload_sha256, text_sha256, valid_assertion_binding,
)
from .character_explanation_review import ExplanationBaselineSummary
from .config import CHARACTER_EXPLANATION_ASSERTIONS_POLICY_V1
from .db import AnalysisRunCharacterTraitInputRow, AnalysisRunInputRow, CharacterObservationRow
from .domain import EvidenceSpan


class _Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    case_id: str = Field(min_length=1, max_length=200)
    confirmed_candidate_id: str = Field(min_length=1, max_length=200)
    baseline_payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    support_id: str = Field(min_length=1, max_length=200)
    kind: Literal["causal_bridge", "exception", "possible_explanation"]
    binding: ExplanationAssertionBinding


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["explanation-assertions-v1"]
    receipts: list[_Receipt] = Field(max_length=MAX_ASSERTION_RECEIPTS)
    omitted_count: int = Field(ge=0, le=100000)
    payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _require(condition):
    if not condition:
        raise ValueError("character explanation assertion integrity mismatch")


def _source_span(ref: AssertionRef, sources: dict) -> EvidenceSpan:
    source = sources.get(ref.document_id)
    _require(source is not None and text_sha256(source.content) == source.content_sha256)
    lines = source.content.splitlines()
    _require(1 <= ref.line_start <= ref.line_end <= len(lines))
    text = "\n".join(lines[ref.line_start - 1:ref.line_end])
    matches = [value for value in (text, text.strip()) if text_sha256(value) == ref.evidence_sha256]
    _require(bool(matches))
    return EvidenceSpan(document_id=source.document_id, document_name=source.document_name,
                        line_start=ref.line_start, line_end=ref.line_end, text=matches[0])


def validate_explanation_assertion_payload(db, *, run, payload):
    if payload is None:
        return None  # Old runs and new runs without assertion review retain their meaning.
    try:
        parsed = _Payload.model_validate_json(json.dumps(payload), strict=True)
        coverage = run.batch_coverage if isinstance(run.batch_coverage, dict) else {}
        _require(coverage.get("explanation_assertions_policy") == CHARACTER_EXPLANATION_ASSERTIONS_POLICY_V1)
        _require(payload_sha256({k: v for k, v in payload.items() if k != "payload_sha256"}) == parsed.payload_sha256)
        sources = {row.document_id: row for row in db.scalars(select(AnalysisRunInputRow).where(AnalysisRunInputRow.run_id == run.id)).all()}
        baselines = {row.candidate_id: row for row in db.scalars(select(AnalysisRunCharacterTraitInputRow).where(
            AnalysisRunCharacterTraitInputRow.run_id == run.id,
            AnalysisRunCharacterTraitInputRow.project_id == run.project_id,
        )).all()}
        seen = set()
        for item in parsed.receipts:
            identity = (item.case_id, item.support_id)
            _require(identity not in seen)
            seen.add(identity)
            baseline = baselines.get(item.confirmed_candidate_id)
            _require(baseline is not None and baseline.payload_sha256 == item.baseline_payload_sha256
                     and payload_sha256(baseline.payload) == item.baseline_payload_sha256)
            value = baseline.payload
            _require(value.get("candidate_id") == item.confirmed_candidate_id)
            summary = ExplanationBaselineSummary(
                character=value.get("character_display_name"), dimension=value.get("trait_type"),
                trait_key=value.get("trait_key"), key_object=value.get("key_object", ""),
                statement=value.get("value"), approved_axis_definition=value.get("approved_axis_definition"),
                axis_positive_proposition=value.get("axis_positive_proposition"),
            )
            _require(payload_sha256(summary.model_dump(mode="json")) == item.binding.baseline_sha256)
            observations = {}
            explanation = None
            for link in item.binding.links:
                current_explanation = _source_span(link.explanation, sources)
                _require(explanation is None or explanation == current_explanation)
                explanation = current_explanation
                current = _source_span(link.observation, sources)
                stored = db.get(CharacterObservationRow, (run.id, link.observation_id))
                _require(stored is not None and payload_sha256(stored.payload) == stored.payload_sha256)
                source = sources[current.document_id]
                observed = stored.payload
                _require(stored.source_run_input_id == source.id and observed.get("observation_id") == link.observation_id
                         and observed.get("source", {}).get("run_input_id") == source.id
                         and observed.get("source", {}).get("content_sha256") == source.content_sha256
                         and EvidenceSpan.model_validate(observed.get("evidence")) == current)
                _require(any(match.get("status") == "matched" and match.get("identity_match") is True
                             and isinstance(match.get("baseline"), dict)
                             and match["baseline"].get("input_id") == baseline.id
                             and match["baseline"].get("candidate_id") == item.confirmed_candidate_id
                             and match["baseline"].get("payload_sha256") == item.baseline_payload_sha256
                             for match in observed.get("matches", [])))
                observations[link.observation_id] = current
            _require(valid_assertion_binding(item.binding.model_dump(mode="json"), explanation, observations))
        return parsed.model_dump(mode="json")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
        raise ValueError("character explanation assertion integrity mismatch") from None
