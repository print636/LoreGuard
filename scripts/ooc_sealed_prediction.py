"""Fail-closed projection from a sealed live run into public predictions.

This module contains no gold/oracle loader and no scorer.  It accepts only a
frozen public input plus a structured runtime snapshot collected from the
LoreGuard HTTP API.  Ambiguous report states, incomplete coverage or citation
mapping failures become ``unassessed`` rather than guessed labels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scripts.ooc_eval_contract import (
    CasePrediction,
    Citation,
    OpaqueId,
    PredictionBundle,
    PublicInputBundle,
    PublicInputCase,
    Sha256,
    StageFunnel,
    canonical_sha256,
)


SEALED_PREDICTION_SCHEMA_VERSION = "loreguard-ooc-sealed-prediction-v1"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class RuntimeEvidenceRef(StrictModel):
    role: Literal["B", "C", "G", "X", "P"]
    document_id: OpaqueId
    line_start: int = Field(strict=True, ge=1, le=10_000_000)
    line_end: int = Field(strict=True, ge=1, le=10_000_000)
    event_group_id: OpaqueId | None = None

    @model_validator(mode="after")
    def validate_range(self) -> RuntimeEvidenceRef:
        if self.line_end < self.line_start:
            raise ValueError("runtime_evidence_range_invalid")
        if self.role != "C" and self.event_group_id is not None:
            raise ValueError("only_current_evidence_may_have_event_group")
        return self


class RuntimeCaseSnapshot(StrictModel):
    case_id: OpaqueId
    stage_status: Literal["completed", "partial", "failed"]
    material_coverage: Literal["complete", "partial", "unknown"]
    explanation_coverage: Literal["complete", "partial", "unknown"]
    report_collections_complete: bool
    citation_refs_complete: bool
    trace_count: int = Field(strict=True, ge=0, le=10_000)
    final_outcome: Literal[
        "conflict",
        "needs_confirmation",
        "unverifiable",
        "no_issue",
        "unknown",
    ]
    formal_issue_count: int = Field(strict=True, ge=0, le=10_000)
    review_clue_count: int = Field(strict=True, ge=0, le=10_000)
    provisional_clue_count: int = Field(strict=True, ge=0, le=10_000)
    # ``citations`` are the final decision-trace refs.  For a surfaced row,
    # ``report_evidence`` is independently reconstructed from that row's
    # concrete evidence plus its role/index binding.  Keeping both prevents a
    # trace from silently supplying evidence that the user-visible report did
    # not actually contain.
    citations: tuple[RuntimeEvidenceRef, ...] = Field(default=(), max_length=10_000)
    report_evidence: tuple[RuntimeEvidenceRef, ...] = Field(
        default=(), max_length=10_000
    )
    stage_funnel: StageFunnel


class SealedPredictionArtifact(StrictModel):
    schema_version: Literal["loreguard-ooc-sealed-prediction-v1"] = (
        SEALED_PREDICTION_SCHEMA_VERSION
    )
    dataset_id: OpaqueId
    public_input_sha256: Sha256
    author_setup_sha256: Sha256
    execution_freeze_sha256: Sha256
    runtime_provenance_sha256: Sha256
    prediction: PredictionBundle

    @model_validator(mode="after")
    def validate_inner_binding(self) -> SealedPredictionArtifact:
        if (
            self.prediction.dataset_id != self.dataset_id
            or self.prediction.public_input_sha256 != self.public_input_sha256
        ):
            raise ValueError("sealed_prediction_inner_binding_mismatch")
        return self


class PredictionProjectionError(ValueError):
    pass


def _unassessed(snapshot: RuntimeCaseSnapshot) -> CasePrediction:
    return CasePrediction(
        case_id=snapshot.case_id,
        outcome="unassessed",
        surface="none",
        citations=(),
        stage_funnel=snapshot.stage_funnel,
    )


def _project_citations(
    public_case: PublicInputCase,
    runtime_refs: Sequence[RuntimeEvidenceRef],
) -> tuple[Citation, ...] | None:
    documents = {row.document_id: row for row in public_case.documents}
    anchors_by_document: dict[str, list] = {}
    for anchor in public_case.evidence_catalog:
        anchors_by_document.setdefault(anchor.document_id, []).append(anchor)
    citations: list[Citation] = []
    assigned: dict[str, str] = {}
    for runtime_ref in runtime_refs:
        document = documents.get(runtime_ref.document_id)
        if document is None:
            return None
        anchors = [
            anchor
            for anchor in anchors_by_document.get(runtime_ref.document_id, ())
            if runtime_ref.line_start <= anchor.line_start
            and anchor.line_end <= runtime_ref.line_end
        ]
        if (
            not anchors
            or min(anchor.line_start for anchor in anchors) != runtime_ref.line_start
            or max(anchor.line_end for anchor in anchors) != runtime_ref.line_end
        ):
            return None
        for anchor in anchors:
            prior_role = assigned.setdefault(anchor.evidence_id, runtime_ref.role)
            if prior_role != runtime_ref.role:
                return None
            if any(row.evidence_id == anchor.evidence_id for row in citations):
                continue
            citations.append(
                Citation(
                    evidence_id=anchor.evidence_id,
                    role=runtime_ref.role,
                    document_id=document.document_id,
                    snapshot_version=document.version,
                    snapshot_sha256=document.sha256,
                    line_start=anchor.line_start,
                    line_end=anchor.line_end,
                )
            )
    return tuple(citations)


def _surface_refs_are_exact(snapshot: RuntimeCaseSnapshot) -> bool:
    """Require one lossless report/trace binding for a surfaced decision.

    Ordering is part of the binding.  Both server projections originate from
    the review response order, so accepting only a set match would needlessly
    permit duplicated, reordered, or role-swapped evidence.
    """

    return bool(
        snapshot.report_evidence
        and len(snapshot.report_evidence) == len(snapshot.citations)
        and snapshot.report_evidence == snapshot.citations
    )


def project_case_prediction(
    public_case: PublicInputCase,
    snapshot: RuntimeCaseSnapshot,
) -> CasePrediction:
    if snapshot.case_id != public_case.case_id:
        raise PredictionProjectionError("runtime_case_id_mismatch")
    complete = bool(
        snapshot.stage_status == "completed"
        and snapshot.material_coverage == "complete"
        and snapshot.explanation_coverage == "complete"
        and snapshot.report_collections_complete
        and snapshot.citation_refs_complete
        and snapshot.trace_count == 1
    )
    if not complete:
        return _unassessed(snapshot)

    surfaced = snapshot.final_outcome in {
        "conflict",
        "needs_confirmation",
        "unverifiable",
    }
    if surfaced and not _surface_refs_are_exact(snapshot):
        return _unassessed(snapshot)
    if not surfaced and snapshot.report_evidence:
        return _unassessed(snapshot)

    # Surfaced predictions are projected only from evidence proven to exist on
    # the bound report row.  A no-issue decision has no report row, so its
    # strictly validated trace refs remain the explanatory source.
    projection_refs = (
        snapshot.report_evidence if surfaced else snapshot.citations
    )
    citations = _project_citations(public_case, projection_refs)
    if citations is None:
        return _unassessed(snapshot)

    if snapshot.final_outcome == "conflict":
        formal_shape = bool(
            snapshot.formal_issue_count == 1
            and snapshot.review_clue_count == 0
            and snapshot.provisional_clue_count == 0
        )
        current_refs = [row for row in projection_refs if row.role == "C"]
        event_groups = {row.event_group_id for row in current_refs}
        has_two_events = bool(
            len(current_refs) >= 2
            and None not in event_groups
            and len(event_groups) >= 2
        )
        nonoverlapping = all(
            left.document_id != right.document_id
            or max(left.line_start, right.line_start)
            > min(left.line_end, right.line_end)
            for index, left in enumerate(current_refs)
            for right in current_refs[index + 1 :]
        )
        roles = [row.role for row in citations]
        if not (
            formal_shape
            and has_two_events
            and nonoverlapping
            and "B" in roles
            and roles.count("C") >= 2
            and not {"G", "X", "P"}.intersection(roles)
        ):
            return _unassessed(snapshot)
        try:
            return CasePrediction(
                case_id=snapshot.case_id,
                outcome="conflict",
                surface="formal_issue",
                citations=citations,
                stage_funnel=snapshot.stage_funnel,
            )
        except ValueError:
            return _unassessed(snapshot)

    if snapshot.final_outcome in {"needs_confirmation", "unverifiable"}:
        if not (
            snapshot.formal_issue_count == 0
            and snapshot.review_clue_count == 1
            and snapshot.provisional_clue_count == 0
            and citations
        ):
            return _unassessed(snapshot)
        return CasePrediction(
            case_id=snapshot.case_id,
            outcome="indeterminate",
            surface="review_clue",
            citations=citations,
            stage_funnel=snapshot.stage_funnel,
        )

    if snapshot.final_outcome == "no_issue":
        if any(
            (
                snapshot.formal_issue_count,
                snapshot.review_clue_count,
                snapshot.provisional_clue_count,
            )
        ) or any(row.role == "P" for row in citations):
            return _unassessed(snapshot)
        return CasePrediction(
            case_id=snapshot.case_id,
            outcome="no_issue",
            surface="none",
            citations=citations,
            stage_funnel=snapshot.stage_funnel,
        )

    return _unassessed(snapshot)


def build_prediction_artifact(
    public: PublicInputBundle,
    snapshots: Sequence[RuntimeCaseSnapshot],
    *,
    run_id: str,
    repeat_id: str,
    author_setup_sha256: str,
    execution_freeze_sha256: str,
    runtime_provenance_sha256: str,
) -> SealedPredictionArtifact:
    by_case = {row.case_id: row for row in snapshots}
    if len(by_case) != len(snapshots):
        raise PredictionProjectionError("duplicate_runtime_case_id")
    public_ids = {row.case_id for row in public.cases}
    if set(by_case) != public_ids:
        raise PredictionProjectionError("runtime_snapshots_must_cover_every_public_case")
    predictions = tuple(
        project_case_prediction(case, by_case[case.case_id]) for case in public.cases
    )
    run_status = (
        "complete"
        if all(row.outcome != "unassessed" for row in predictions)
        else "partial"
    )
    public_digest = canonical_sha256(public)
    prediction = PredictionBundle(
        dataset_id=public.dataset_id,
        public_input_sha256=public_digest,
        run_id=run_id,
        repeat_id=repeat_id,
        run_status=run_status,
        predictions=predictions,
    )
    return SealedPredictionArtifact(
        dataset_id=public.dataset_id,
        public_input_sha256=public_digest,
        author_setup_sha256=author_setup_sha256,
        execution_freeze_sha256=execution_freeze_sha256,
        runtime_provenance_sha256=runtime_provenance_sha256,
        prediction=prediction,
    )


__all__ = [
    "SEALED_PREDICTION_SCHEMA_VERSION",
    "PredictionProjectionError",
    "RuntimeCaseSnapshot",
    "RuntimeEvidenceRef",
    "SealedPredictionArtifact",
    "build_prediction_artifact",
    "project_case_prediction",
]
