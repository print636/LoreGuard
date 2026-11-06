"""Bind private time-scope receipts to this run's actual sources and records."""
from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from .config import NARRATIVE_TIME_SCOPE_POLICY_V1
from .db import AnalysisRunInputRow, AnalysisRunInputContextRow
from .narrative_relations import _digest
from .narrative_time_scope import TIME_SCOPE_SCHEMA, member_matches, validate_time_scope_sources


class _Ref(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class _Member(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    document_id: str = Field(min_length=1, max_length=200)
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    story_scope: str = Field(min_length=1, max_length=200)
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_ref: _Ref
    time_ref: _Ref
    clock: str | None = Field(max_length=8)
    clock_relation: Literal["same", "equivalent", "unclear"]


class _Proof(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["narrative-time-scope-v1"]
    claim_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_ref: _Ref
    members: list[_Member] = Field(min_length=2, max_length=16)
    payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def time_scope_storage_payload(db, *, run, records, directives):
    """Source binding is distinct from semantic correctness or final comparison.

    A sourced member may have no admitted record. Its empty match list remains
    explicit and never creates an observation or a formal result.
    """
    if len(records) != len(directives) or len(records) > 100_000:
        raise ValueError("narrative time scope storage integrity mismatch")
    receipts, sources = [], None
    for stored, row in zip(records, directives):
        proof = getattr(row, "narrative_time_scope", None)
        if proof is None and row.kind != "narrative_time_scope":
            continue
        try:
            parsed = _Proof.model_validate(proof)
            if (run.batch_coverage or {}).get("narrative_time_scope_policy") != NARRATIVE_TIME_SCOPE_POLICY_V1:
                raise ValueError
            if sources is None:
                sources = {item.document_id:item for item in db.scalars(select(AnalysisRunInputRow).where(
                    AnalysisRunInputRow.run_id == run.id)).all()}
            source = sources[row.evidence.document_id]
            contexts = {}
            used = {row.evidence.document_id, *(member.document_id for member in parsed.members)}
            for document_id in used:
                item = sources[document_id]
                if hashlib.sha256(item.content.encode("utf-8")).hexdigest() != item.content_sha256:
                    raise ValueError
                contexts[document_id] = db.get(AnalysisRunInputContextRow, item.id)
            context = contexts[source.document_id]
            if context is not None and any(getattr(context,key) is not None and row.attrs.get(key) != getattr(context,key)
                                           for key in ("document_role","story_scope")):
                raise ValueError
            lines = source.content.splitlines()
            span = row.evidence
            if not 1 <= span.line_start <= span.line_end <= len(lines):
                raise ValueError
            text = "\n".join(lines[span.line_start-1:span.line_end])
            if span.text not in (text,text.strip()) or span.document_name != source.document_name:
                raise ValueError
            if not validate_time_scope_sources(row,{key:sources[key].content for key in used}):
                raise ValueError
            members = []
            for index, member in enumerate(parsed.members):
                item = sources[member.document_id]
                context = contexts[member.document_id]
                expected_scope = (context.story_scope if context else None) or "global"
                if member.story_scope != expected_scope or not 1 <= member.line_start <= member.line_end <= len(item.content.splitlines()):
                    raise ValueError
                matched = []
                for actual, candidate in zip(records,directives):
                    if member_matches(proof["members"][index],candidate):
                        _check_record(run,actual,candidate)
                        matched.append(actual.id)
                members.append({"member_index":index,"source_input_id":item.id,
                                "source_content_sha256":item.content_sha256,"matched_record_ids":matched})
            _check_record(run,stored,row)
            receipts.append({"record_id":stored.id,"source_input_id":source.id,
                             "source_content_sha256":source.content_sha256,
                             "record_sha256":_digest({"kind":stored.kind,"attrs":stored.attrs,"evidence":stored.evidence}),
                             "binding":parsed.model_dump(mode="json"),"members":members})
        except (ValueError, TypeError, KeyError, AttributeError, IndexError):
            raise ValueError("narrative time scope storage integrity mismatch") from None
    if not receipts:
        return None
    payload = {"schema_version":TIME_SCOPE_SCHEMA,"receipts":receipts}
    return {**payload,"payload_sha256":_digest(payload)}


def _check_record(run, stored, row):
    if (not stored.id or stored.run_id != run.id or stored.kind != row.kind
            or stored.attrs != row.attrs or stored.evidence != row.evidence.model_dump()):
        raise ValueError
