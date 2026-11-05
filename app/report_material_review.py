from __future__ import annotations

from hashlib import sha256
from typing import get_args

from sqlalchemy import select

from .character_consistency_stage import _reviewed_event_group_id
from .character_drift import (
    CHARACTER_REVIEW_POLICY_VERSION,
    CHARACTER_REVIEW_SCHEMA_VERSION,
    CHARACTER_REVIEWED_EVENTS_SCHEMA_VERSION,
)
from .character_material_coverage import (
    MATERIAL_REVIEW_KEY, MATERIAL_REVIEW_SCHEMA, MAX_MATERIAL_GAPS, MaterialPhase, MaterialReason,
)
from .db import AnalysisRunCharacterTraitInputRow, AnalysisRunInputRow
from .narrative_context import payload_sha256


_PHASES = get_args(MaterialPhase)
_REASONS = get_args(MaterialReason)
_OPINIONS = {"contradicts", "explained", "needs_confirmation", "insufficient_evidence"}


def _text_hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _comparisons(row, receipt, sources) -> list[dict] | None:
    """Validate the cited groups against this clue and this run's frozen inputs."""
    if not isinstance(receipt, dict):
        return None
    candidate_id = row.extra.get("confirmed_candidate_id")
    if (
        not isinstance(candidate_id, str)
        or receipt.get("schema_version") != CHARACTER_REVIEWED_EVENTS_SCHEMA_VERSION
        or receipt.get("assessment_policy") != CHARACTER_REVIEW_POLICY_VERSION
        or receipt.get("review_schema_version") != CHARACTER_REVIEW_SCHEMA_VERSION
        or receipt.get("confirmed_candidate_id_sha256") != _text_hash(candidate_id)
        or not isinstance(receipt.get("comparison_key"), str)
        or not receipt["comparison_key"]
        or row.extra.get("evidence_binding") != "review_citations_v1"
    ):
        return None
    refs, events = receipt.get("evidence_refs"), receipt.get("events")
    citations = row.extra.get("review_citation_refs")
    if (
        not isinstance(refs, list) or not 2 <= len(refs) <= 8
        or not isinstance(events, list) or not 1 <= len(events) <= 8
        or not isinstance(citations, list) or len(citations) != len(refs)
    ):
        return None
    by_handle = {}
    for ref, citation in zip(refs, citations):
        if not isinstance(ref, dict) or not isinstance(citation, dict):
            return None
        if any(ref.get(key) != citation.get(key) for key in ("handle", "role", "evidence_index", "response_index")):
            return None
        index, handle = ref.get("evidence_index"), ref.get("handle")
        if type(index) is not int or not 0 <= index < len(row.evidence) or not isinstance(handle, str) or handle in by_handle:
            return None
        span = row.evidence[index]
        source = sources.get(span["document_id"])
        if source is None or (
            ref.get("source_input_id_sha256") != _text_hash(source.id)
            or ref.get("document_version") != source.document_version
            or ref.get("content_sha256") != source.content_sha256
            or any(ref.get(key) != span[key] for key in ("document_name", "line_start", "line_end"))
            or ref.get("text_sha256") != _text_hash(span["text"])
            or source.document_name != span["document_name"]
        ):
            return None
        literal = "\n".join(source.content.splitlines()[span["line_start"] - 1:span["line_end"]])
        if literal != span["text"] or _text_hash(source.content) != source.content_sha256:
            return None
        by_handle[handle] = ref
    current = {handle for handle, ref in by_handle.items() if ref["role"] == "C"}
    if not current or not any(ref["role"] == "B" for ref in refs):
        return None
    covered, comparisons = set(), []
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("verdict"), str) or event["verdict"] not in _OPINIONS:
            return None
        handles = event.get("current_citations")
        if (
            not isinstance(handles, list) or not handles
            or any(not isinstance(h, str) or h not in current or h in covered for h in handles)
            or len(set(handles)) != len(handles)
            or event.get("event_group_id") != _reviewed_event_group_id(
                receipt["confirmed_candidate_id_sha256"], receipt["comparison_key"],
                [by_handle[h] for h in handles],
            )
        ):
            return None
        covered.update(handles)
        comparisons.append({
            "evidence_indices": sorted({by_handle[h]["evidence_index"] for h in handles}),
            "opinion": event["verdict"],
        })
    return comparisons if covered == current else None


def _located_gap(gap, sources):
    if not isinstance(gap.get("source_document_id"), str):
        return None
    source = sources.get(gap.get("source_document_id"))
    start, end = gap.get("line_start"), gap.get("line_end")
    if (
        source is None or source.id != gap.get("source_input_id")
        or source.content_sha256 != gap.get("source_content_sha256")
        or _text_hash(source.content) != source.content_sha256
        or type(start) is not int or type(end) is not int or not 1 <= start <= end
    ):
        return None
    lines = source.content.splitlines()
    if end > len(lines):
        return None
    text = "\n".join(lines[start - 1:end])
    if not text.strip() or len(text) > 20_000 or _text_hash(text) != gap.get("text_sha256"):
        return None
    return {
        "document_id": source.document_id, "document_name": source.document_name,
        "line_start": start, "line_end": end, "text": text,
    }


def project_material_review(db, run, row) -> dict | None:
    """Additional context for an already verified clue. Never establishes a finding."""
    metadata = row.extra if isinstance(row.extra, dict) else {}
    ledger = metadata.get(MATERIAL_REVIEW_KEY)
    if not isinstance(ledger, dict) or run.status != "completed" or row.run_id != run.id or row.report_class != "review_clue":
        return None
    receipt = metadata.get("reviewed_events")
    candidate_id = metadata.get("confirmed_candidate_id")
    if (
        ledger.get("schema_version") != MATERIAL_REVIEW_SCHEMA
        or not isinstance(candidate_id, str)
        or ledger.get("confirmed_candidate_id") != candidate_id
        or ledger.get("reviewed_events_sha256") != payload_sha256(receipt)
        or ledger.get("payload_sha256") != payload_sha256({k: v for k, v in ledger.items() if k != "payload_sha256"})
        or not isinstance(ledger.get("case_id"), str) or not ledger["case_id"].startswith("cdc_")
        or not isinstance(ledger.get("gaps"), list) or len(ledger["gaps"]) > MAX_MATERIAL_GAPS
        or type(ledger.get("omitted_count")) is not int or ledger["omitted_count"] < 0
        or type(ledger.get("total_count")) is not int or ledger["total_count"] < 1
    ):
        return None
    trait = db.scalar(select(AnalysisRunCharacterTraitInputRow).where(
        AnalysisRunCharacterTraitInputRow.run_id == run.id,
        AnalysisRunCharacterTraitInputRow.project_id == run.project_id,
        AnalysisRunCharacterTraitInputRow.candidate_id == candidate_id,
    ))
    if trait is None or payload_sha256(trait.payload) != trait.payload_sha256:
        return None
    sources = {source.document_id: source for source in db.scalars(select(AnalysisRunInputRow).where(
        AnalysisRunInputRow.run_id == run.id,
    ))}
    comparisons = _comparisons(row, receipt, sources)
    if comparisons is None:
        return None
    current_spans = [row.evidence[i] for event in comparisons for i in event["evidence_indices"]]
    gaps = []
    for index, gap in enumerate(ledger["gaps"]):
        if (
            not isinstance(gap, dict) or gap.get("phase") not in _PHASES
            or gap.get("reason") not in _REASONS
            or not isinstance(gap.get("location"), str) or gap["location"] not in {"located", "unknown"}
            or type(gap.get("count")) is not int or gap["count"] < 1
        ):
            return None
        # Recompute the location relationship. Separate lines never prove separate events.
        evidence = _located_gap(gap, sources) if gap["location"] == "located" else None
        relation = "unknown"
        if evidence is not None:
            relation = "overlap" if any(
                evidence["document_id"] == span["document_id"]
                and evidence["line_start"] <= span["line_end"]
                and span["line_start"] <= evidence["line_end"]
                for span in current_spans
            ) else "elsewhere"
        gaps.append({
            "index": index, "count": gap["count"], "reason": gap["reason"],
            "location": "located" if evidence is not None else "unknown",
            "relation": relation, **({"evidence": evidence} if evidence is not None else {}),
        })
    if sum(gap["count"] for gap in gaps) + ledger["omitted_count"] != ledger["total_count"]:
        return None
    return {"comparisons": comparisons, "gaps": gaps, "omitted_count": ledger["omitted_count"], "total_count": ledger["total_count"]}
