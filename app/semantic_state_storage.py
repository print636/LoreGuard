"""Persist source bindings separately from public ordinary records."""
from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from .config import SEMANTIC_STATE_POLICY_V1
from .db import AnalysisRunInputRow, AnalysisRunInputContextRow
from .semantic_state import SEMANTIC_STATE_SCHEMA, digest, valid_semantic_state


class _Ref(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class _Proof(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal["semantic-state-v1"]
    kind: Literal["fact", "item", "uses"]
    claim_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    refs: dict[str, _Ref]
    relation: Literal["supported", "different", "unclear"]
    actuality: Literal["actual", "negated", "hypothetical", "reported", "unclear"]
    basis: Literal["explicit", "supported_inference", "unclear"]
    roles: dict[str, Literal["same", "equivalent", "different", "unclear"]]
    transfer_relation: Literal["same", "equivalent", "different", "unclear"] | None
    payload_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def semantic_state_storage_payload(db, *, run, records, directives):
    """Reject damaged bindings before committing either records or issues.

    A binding attests to source identity, not to the model's semantic accuracy.
    Records without the new proof retain their historical serialization.
    """
    if len(records) != len(directives) or len(records) > 100_000:
        raise ValueError("semantic state storage integrity mismatch")
    sources = None
    receipts = []
    for record, directive in zip(records, directives):
        proof = getattr(directive, "semantic_state", None)
        if proof is None:
            if any(key in directive.attrs for key in ("state_kind", "through_time", "transfer_from", "retention_scope")):
                raise ValueError("semantic state storage integrity mismatch")
            continue
        try:
            parsed = _Proof.model_validate(proof)
            if (run.batch_coverage or {}).get("semantic_state_policy") != SEMANTIC_STATE_POLICY_V1:
                raise ValueError
            if not valid_semantic_state(directive):
                raise ValueError
            if sources is None:
                sources = {row.document_id: row for row in db.scalars(select(AnalysisRunInputRow).where(
                    AnalysisRunInputRow.run_id == run.id)).all()}
            source = sources[directive.evidence.document_id]
            context = db.get(AnalysisRunInputContextRow, source.id)
            if context is not None and any(
                getattr(context, key) is not None and directive.attrs.get(key) != getattr(context, key)
                for key in ("document_role", "story_scope")
            ):
                raise ValueError
            evidence = directive.evidence
            if hashlib.sha256(source.content.encode("utf-8")).hexdigest() != source.content_sha256:
                raise ValueError
            lines = source.content.splitlines()
            if not 1 <= evidence.line_start <= evidence.line_end <= len(lines):
                raise ValueError
            text = "\n".join(lines[evidence.line_start - 1:evidence.line_end])
            if evidence.text not in (text, text.strip()) or evidence.document_name != source.document_name:
                raise ValueError
            if (record.run_id != run.id or record.kind != directive.kind
                    or record.attrs != directive.attrs or record.evidence != evidence.model_dump()):
                raise ValueError
            receipts.append({"record_id": record.id, "source_input_id": source.id,
                             "source_document_id": source.document_id,
                             "source_content_sha256": source.content_sha256,
                             "record_sha256": digest({"kind": record.kind, "attrs": record.attrs, "evidence": record.evidence}),
                             "binding": parsed.model_dump(mode="json")})
        except (ValueError, TypeError, KeyError, AttributeError, IndexError):
            raise ValueError("semantic state storage integrity mismatch") from None
    if not receipts:
        return None
    result = {"schema_version": SEMANTIC_STATE_SCHEMA, "receipts": receipts}
    return {**result, "payload_sha256": digest(result)}
