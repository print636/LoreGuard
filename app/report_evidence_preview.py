from __future__ import annotations

from collections.abc import Callable
from hashlib import sha256
from typing import Any, Literal

from sqlalchemy import select

from .character_consistency_stage import (
    PROVISIONAL_DRAFT_CLUES_PAYLOAD_KEY,
    ProvisionalCluesUnavailable,
    project_provisional_draft_clues,
)
from .character_traits import TraitEvidenceInput
from .db import (
    AnalysisDiagnosticRow,
    AnalysisRunCharacterTraitInputRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    IssueRow,
    ProjectRow,
)
from .domain import ConsistencyIssue
from .service import (
    _character_review_clue_roles,
    _character_review_snapshot_evidence_index,
    _character_review_source_index,
    _load_verified_snapshot,
)


EvidencePreviewKind = Literal["issue", "review_clue", "provisional_clue"]
NO_STORE_HEADERS = {"Cache-Control": "no-store"}
NOT_FOUND_MESSAGE = "报告证据不存在"


class EvidencePreviewUnavailable(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _not_found() -> None:
    raise EvidencePreviewUnavailable(404, NOT_FOUND_MESSAGE)


def _snapshot_unavailable(*, missing: bool = False) -> None:
    raise EvidencePreviewUnavailable(
        409, "报告证据原文快照缺失" if missing else "报告证据原文快照无法核验"
    )


def _reference(span: object) -> dict[str, Any]:
    if not isinstance(span, dict):
        _snapshot_unavailable()
    document_id, document_name, text = (
        span.get("document_id"), span.get("document_name"), span.get("text")
    )
    line_start, line_end = span.get("line_start"), span.get("line_end")
    if (
        not isinstance(document_id, str) or not 1 <= len(document_id.strip()) <= 200
        or not isinstance(document_name, str) or not 1 <= len(document_name.strip()) <= 255
        or not isinstance(text, str) or not 1 <= len(text.strip()) <= 20_000
        or type(line_start) is not int or type(line_end) is not int
        or not 1 <= line_start <= line_end <= 10_000_000
    ):
        _snapshot_unavailable()
    return {
        "document_id": document_id, "document_name": document_name,
        "line_start": line_start, "line_end": line_end, "text": text,
    }


def _identity(span: dict[str, Any]) -> tuple[str, str, int, int, str]:
    return tuple(span[key] for key in (
        "document_id", "document_name", "line_start", "line_end", "text"
    ))


def _verified_lines(source: AnalysisRunInputRow, reference: dict[str, Any]) -> list[str]:
    if (
        not isinstance(source.content, str)
        or sha256(source.content.encode("utf-8")).hexdigest() != source.content_sha256
        or source.document_id != reference["document_id"]
        or source.document_name != reference["document_name"]
        or type(source.document_version) is not int or source.document_version < 1
    ):
        _snapshot_unavailable()
    lines = source.content.splitlines()
    start, end = reference["line_start"], reference["line_end"]
    if end > len(lines):
        _snapshot_unavailable()
    literal = "\n".join(lines[start - 1:end])
    if reference["text"] not in {literal, literal.strip()}:
        _snapshot_unavailable()
    return lines


def _confirmed_source(
    db, run: AnalysisRunRow, workspace_id: str, row: IssueRow,
    reference: dict[str, Any], evidence_index: int,
    verified_evidence: dict[str, frozenset] | None,
) -> tuple[AnalysisRunInputRow | None, TraitEvidenceInput | None]:
    metadata = row.extra if isinstance(row.extra, dict) else {}
    candidate_id = metadata.get("confirmed_candidate_id")
    roles = {}
    if metadata.get("evidence_binding") is not None:
        try:
            roles = _character_review_clue_roles(ConsistencyIssue(
                id=row.id, category=row.category, severity=row.severity,
                confidence=row.confidence, title=row.title, explanation=row.explanation,
                evidence=row.evidence, suggestion=row.suggestion, metadata=metadata,
            ))
        except (TypeError, ValueError):
            _snapshot_unavailable()
        if roles is None:
            _snapshot_unavailable()
    selected_roles = roles.get(evidence_index, set())
    if selected_roles & {"C", "G", "X", "P"}:
        return None, None
    baseline_selected = "B" in selected_roles
    if not isinstance(candidate_id, str):
        if baseline_selected:
            _snapshot_unavailable()
        return None, None
    if verified_evidence is None:
        try:
            verified_evidence = _character_review_snapshot_evidence_index(db, run.id)
        except (RuntimeError, TypeError, ValueError, AttributeError):
            _snapshot_unavailable()
    if _identity(reference) not in verified_evidence.get(candidate_id, frozenset()):
        if baseline_selected:
            _snapshot_unavailable()
        return None, None
    # The index has already checked the payload hash, row identity and evidence
    # hash. Only this case's candidate and exact evidence may resolve an old input.
    trait = db.scalar(select(AnalysisRunCharacterTraitInputRow).where(
        AnalysisRunCharacterTraitInputRow.run_id == run.id,
        AnalysisRunCharacterTraitInputRow.candidate_id == candidate_id,
    ))
    matches = [
        TraitEvidenceInput.model_validate(raw)
        for raw in trait.payload["evidence"]
        if _identity(raw) == _identity(reference)
    ]
    if not matches or len({span.model_dump_json() for span in matches}) != 1:
        _snapshot_unavailable()
    binding = matches[0]
    # Do not inspect a foreign input's body before proving its owning run.
    owned_run_id = db.scalar(select(AnalysisRunInputRow.run_id).where(
        AnalysisRunInputRow.id == binding.input_id,
    ))
    if owned_run_id is None:
        return None, binding
    source_run = db.scalar(
        select(AnalysisRunRow)
        .join(ProjectRow, ProjectRow.id == AnalysisRunRow.project_id)
        .where(
            AnalysisRunRow.id == owned_run_id,
            AnalysisRunRow.project_id == run.project_id,
            ProjectRow.workspace_id == workspace_id,
        )
    )
    if source_run is None:
        _not_found()
    source = db.scalar(select(AnalysisRunInputRow).where(
        AnalysisRunInputRow.id == binding.input_id,
        AnalysisRunInputRow.run_id == source_run.id,
    ))
    if source is None:
        return None, binding
    if (
        source.document_version != binding.document_version
        or source.content_sha256 != binding.content_sha256
    ):
        _snapshot_unavailable()
    return source, binding


def _provisional_reference(db, run: AnalysisRunRow, item_id: str, evidence_index: int):
    if evidence_index != 0:
        _not_found()
    diagnostic = db.get(AnalysisDiagnosticRow, run.id)
    if diagnostic is None:
        _not_found()
    if not isinstance(diagnostic.payload, dict):
        _snapshot_unavailable()
    payload = diagnostic.payload.get(PROVISIONAL_DRAFT_CLUES_PAYLOAD_KEY)
    if payload is None:
        _not_found()
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        _snapshot_unavailable()
    selected = next((
        raw for raw in payload["items"]
        if isinstance(raw, dict) and raw.get("id") == item_id
    ), None)
    if selected is None:
        _not_found()
    if run.status != "completed":
        _snapshot_unavailable()
    inputs = list(db.scalars(
        select(AnalysisRunInputRow).where(AnalysisRunInputRow.run_id == run.id)
        .order_by(AnalysisRunInputRow.ordinal, AnalysisRunInputRow.id)
    ).all())
    try:
        # Reuse the public projection's complete private-input/hash validation.
        public = project_provisional_draft_clues(payload, inputs)
    except ProvisionalCluesUnavailable:
        _snapshot_unavailable(missing=not inputs)
    clue = next(item for item in public["items"] if item["id"] == item_id)
    reference = _reference({**clue, "text": clue["evidence"]})
    source = next(row for row in inputs if row.id == selected["input_id"])
    return reference, source


def build_report_evidence_preview(
    db, run: AnalysisRunRow, *, workspace_id: str, kind: EvidencePreviewKind,
    item_id: str, evidence_index: int, offset: int | None, limit: int,
    safe_review_clue_payload: Callable[..., dict | None],
) -> dict[str, Any]:
    """Resolve report-owned evidence against immutable bodies using SELECT only."""
    binding = None
    if kind == "provisional_clue":
        reference, source = _provisional_reference(db, run, item_id, evidence_index)
    else:
        row = db.scalar(select(IssueRow).where(
            IssueRow.run_id == run.id, IssueRow.id == item_id,
            IssueRow.report_class == ("formal" if kind == "issue" else "review_clue"),
        ))
        if row is None or not isinstance(row.evidence, list) or evidence_index >= len(row.evidence):
            _not_found()
        reference = _reference(row.evidence[evidence_index])
        verified_evidence = None
        if kind == "review_clue":
            try:
                documents, _ = _load_verified_snapshot(db, run.id)
                verified_evidence = _character_review_snapshot_evidence_index(db, run.id)
            except (RuntimeError, TypeError, ValueError, AttributeError):
                _snapshot_unavailable()
            if safe_review_clue_payload(
                row, legacy_frozen_documents=documents,
                legacy_source_index=_character_review_source_index(documents),
                confirmed_trait_evidence_by_candidate=verified_evidence,
            ) is None:
                _not_found()
        source, binding = _confirmed_source(
            db, run, workspace_id, row, reference, evidence_index, verified_evidence,
        )
        if binding is None:
            source = db.scalar(select(AnalysisRunInputRow).where(
                AnalysisRunInputRow.run_id == run.id,
                AnalysisRunInputRow.document_id == reference["document_id"],
            ))
            if source is None:
                _snapshot_unavailable(missing=True)

    response: dict[str, Any] = {
        "run": {"id": run.id, "project_id": run.project_id},
        "reference": {
            "kind": kind, "item_id": item_id, "evidence_index": evidence_index,
            **reference,
        },
        "source": {
            "availability": "full_context" if source is not None else "excerpt_only",
            "provenance": "confirmed_trait_source" if binding else "run_input",
            "source_run_id": source.run_id if source is not None else None,
            "document_version": source.document_version if source is not None else binding.document_version,
            "content_sha256": source.content_sha256 if source is not None else binding.content_sha256,
            "char_count": None,
            "line_count": None,
        },
        "lines": [], "page": None,
        "message": None,
    }
    if source is None:
        response["message"] = "完整原文快照已缺失，仅可查看已冻结的证据摘录。"
        return response
    lines = _verified_lines(source, reference)
    offset = max(0, reference["line_start"] - 1 - 12) if offset is None else offset
    response["source"].update(char_count=len(source.content), line_count=len(lines))
    response["lines"] = [
        {
            "line_number": number, "text": text,
            "is_evidence": reference["line_start"] <= number <= reference["line_end"],
        }
        for number, text in enumerate(lines[offset:offset + limit], start=offset + 1)
    ]
    response["page"] = {
        "offset": offset, "limit": limit, "total": len(lines),
        "has_more": offset + len(response["lines"]) < len(lines),
    }
    return response
