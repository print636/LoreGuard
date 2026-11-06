"""Source coordinates for independently reviewed explanation assertions.

Offsets are Python Unicode code points in the complete frozen EvidenceSpan
text, with an exclusive end. Sentence boundaries are navigation aids only.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .domain import EvidenceSpan

EXPLANATION_ASSERTION_REFS_KEY = "explanation_assertion_refs_v1"
EXPLANATION_ASSERTION_SCHEMA = "explanation-assertions-v1"
MAX_ASSERTIONS = 64
MAX_ASSERTION_RECEIPTS = 64


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def payload_sha256(value: object) -> str:
    return text_sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def assertion_index(evidence: EvidenceSpan) -> dict:
    """Keep full clauses, including commas, and preserve every original byte."""
    rows = []
    for match in re.finditer(r"[^。！？!?；;\n]+(?:[。！？!?；;]+|\n|$)", evidence.text):
        if not match.group().strip():
            continue
        rows.append({"id": f"A{len(rows) + 1}", "start": match.start(), "end": match.end(),
                     "text": match.group(), "text_sha256": text_sha256(match.group())})
    return {"offset_unit": "unicode_code_point", "assertions": rows[:MAX_ASSERTIONS],
            "omitted_count": max(0, len(rows) - MAX_ASSERTIONS)}


class AssertionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    assertion_id: str = Field(pattern=r"^A[1-9][0-9]{0,2}$")
    quote: str = Field(min_length=2, max_length=1200)


class ExplanationAssertionLink(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    observation_citation: str = Field(pattern=r"^C[0-9]{2,3}$")
    independence: Literal["independent_fact", "current_behavior", "purpose_only", "unverified_claim", "unclear"]
    explanation: AssertionSelection | None
    observation: AssertionSelection | None

    @model_validator(mode="after")
    def require_independent_sources(self):
        if self.independence == "independent_fact" and (self.explanation is None or self.observation is None):
            raise ValueError("independent assertion sources required")
        return self


class AssertionRef(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    document_id: str = Field(min_length=1, max_length=200)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    evidence_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    assertion_id: str = Field(pattern=r"^A[1-9][0-9]{0,2}$")
    start: int = Field(ge=0)
    end: int = Field(ge=1)
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class AssertionBindingLink(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    observation_id: str = Field(pattern=r"^cs_[A-Za-z0-9_-]{1,80}$")
    independence: Literal["independent_fact"]
    explanation: AssertionRef
    observation: AssertionRef


class ExplanationAssertionBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["explanation-assertions-v1"]
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    baseline_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    candidate_citation: str = Field(pattern=r"^E[0-9]{2,3}$")
    links: tuple[AssertionBindingLink, ...] = Field(min_length=1, max_length=24)
    payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def selection_ref(selection: AssertionSelection, evidence: EvidenceSpan) -> AssertionRef:
    rows = assertion_index(evidence)["assertions"]
    row = next((row for row in rows if row["id"] == selection.assertion_id), None)
    if row is None or row["text"].count(selection.quote) != 1 or selection.quote != selection.quote.strip():
        raise ValueError("assertion selection does not uniquely bind source")
    start = row["start"] + row["text"].index(selection.quote)
    return AssertionRef(document_id=evidence.document_id, line_start=evidence.line_start,
                        line_end=evidence.line_end, evidence_sha256=text_sha256(evidence.text),
                        assertion_id=selection.assertion_id, start=start,
                        end=start + len(selection.quote), text_sha256=text_sha256(selection.quote))


def valid_assertion_ref(ref: AssertionRef, evidence: EvidenceSpan) -> bool:
    if (ref.document_id != evidence.document_id or ref.line_start != evidence.line_start
            or ref.line_end != evidence.line_end or ref.evidence_sha256 != text_sha256(evidence.text)
            or not 0 <= ref.start < ref.end <= len(evidence.text)):
        return False
    quote = evidence.text[ref.start:ref.end]
    try:
        return selection_ref(AssertionSelection(assertion_id=ref.assertion_id, quote=quote), evidence) == ref
    except ValueError:
        return False


def _position(evidence: EvidenceSpan, offset: int) -> tuple[int, int]:
    prefix = evidence.text[:offset]
    line_index = prefix.count("\n")
    line = evidence.text.split("\n")[line_index]
    column = len(prefix.rsplit("\n", 1)[-1])
    # EvidenceSpan historically trims outer whitespace. Align columns of a
    # shared physical line without treating indentation as a semantic span.
    return evidence.line_start + line_index, max(0, column - (len(line) - len(line.lstrip())))


def independent_refs(left: AssertionRef, left_evidence: EvidenceSpan,
                     right: AssertionRef, right_evidence: EvidenceSpan) -> bool:
    if not valid_assertion_ref(left, left_evidence) or not valid_assertion_ref(right, right_evidence):
        return False
    if left.document_id != right.document_id:
        return True
    # Overlapping frozen lines must agree, including all negation/context;
    # never compare offsets from two inconsistent source versions.
    for line_number in range(max(left.line_start, right.line_start), min(left.line_end, right.line_end) + 1):
        a = left_evidence.text.split("\n")[line_number - left.line_start].strip()
        b = right_evidence.text.split("\n")[line_number - right.line_start].strip()
        if a != b:
            return False
    return (_position(left_evidence, left.end) <= _position(right_evidence, right.start)
            or _position(right_evidence, right.end) <= _position(left_evidence, left.start))


def make_assertion_binding(*, candidate, links, observations, request_sha256: str,
                           baseline_sha256: str) -> dict | None:
    refs = []
    for link in links:
        observation = observations[link.observation_citation]
        # Validate supplied coordinates even for non-definitive opinions.
        e_ref = selection_ref(link.explanation, candidate.evidence) if link.explanation else None
        c_ref = selection_ref(link.observation, observation.evidence) if link.observation else None
        if link.independence != "independent_fact":
            continue
        if not independent_refs(e_ref, candidate.evidence, c_ref, observation.evidence):
            raise ValueError("explanation and observation assertions overlap")
        refs.append({"observation_id": observation.observation_id, "independence": "independent_fact",
                     "explanation": e_ref.model_dump(mode="json"), "observation": c_ref.model_dump(mode="json")})
    if not refs:
        return None
    payload = {"schema_version": EXPLANATION_ASSERTION_SCHEMA, "request_sha256": request_sha256,
               "baseline_sha256": baseline_sha256, "candidate_citation": candidate.citation, "links": refs}
    payload["payload_sha256"] = payload_sha256(payload)
    return payload


def valid_assertion_binding(value: dict | None, explanation: EvidenceSpan, observations: dict) -> bool:
    if not isinstance(value, dict):
        return False
    try:
        proof = ExplanationAssertionBinding.model_validate_json(json.dumps(value), strict=True)
        if payload_sha256({k: v for k, v in value.items() if k != "payload_sha256"}) != proof.payload_sha256:
            return False
        ids = [row.observation_id for row in proof.links]
        return len(ids) == len(set(ids)) and all(
            row.observation_id in observations and independent_refs(
                row.explanation, explanation, row.observation, observations[row.observation_id])
            for row in proof.links)
    except (ValueError, TypeError, IndexError):
        return False


def assertion_receipt_payload(receipts: list[dict], omitted_count: int = 0) -> dict:
    payload = {"schema_version": EXPLANATION_ASSERTION_SCHEMA,
               "receipts": receipts[:MAX_ASSERTION_RECEIPTS],
               "omitted_count": omitted_count + max(0, len(receipts) - MAX_ASSERTION_RECEIPTS)}
    payload["payload_sha256"] = payload_sha256(payload)
    return payload
