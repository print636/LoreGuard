from __future__ import annotations

import hashlib
from uuid import uuid4

import pytest

from scripts import ooc_eval_contract as contract
from scripts.ooc_sealed_prediction import (
    PredictionProjectionError,
    RuntimeCaseSnapshot,
    RuntimeEvidenceRef,
    build_prediction_artifact,
    project_case_prediction,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _id() -> str:
    return str(uuid4())


def _public(*, case_count: int = 1) -> contract.PublicInputBundle:
    cases = []
    for index in range(case_count):
        group_id = _id()
        baseline_id = _id()
        target_id = _id()
        baseline_sha = hashlib.sha256(f"baseline-{index}".encode()).hexdigest()
        target_sha = hashlib.sha256(f"target-{index}".encode()).hexdigest()
        axis_sha = hashlib.sha256(f"axis-{index}".encode()).hexdigest()
        cases.append(
            contract.PublicInputCase(
                case_id=_id(),
                split_id="closed-holdout-v1",
                group_id=group_id,
                world_id=_id(),
                documents=(
                    contract.DocumentSnapshot(
                        document_id=baseline_id,
                        path=f"cases/{group_id}/{baseline_id}.md",
                        version=1,
                        sha256=baseline_sha,
                    ),
                    contract.DocumentSnapshot(
                        document_id=target_id,
                        path=f"cases/{group_id}/{target_id}.md",
                        version=1,
                        sha256=target_sha,
                    ),
                ),
                axis=contract.AxisSnapshot(
                    axis_id=_id(), version=1, sha256=axis_sha
                ),
                evidence_catalog=(
                    contract.EvidenceAnchor(
                        evidence_id=_id(),
                        document_id=baseline_id,
                        line_start=1,
                        line_end=1,
                    ),
                    contract.EvidenceAnchor(
                        evidence_id=_id(),
                        document_id=target_id,
                        line_start=1,
                        line_end=1,
                    ),
                    contract.EvidenceAnchor(
                        evidence_id=_id(),
                        document_id=target_id,
                        line_start=2,
                        line_end=2,
                    ),
                    contract.EvidenceAnchor(
                        evidence_id=_id(),
                        document_id=target_id,
                        line_start=3,
                        line_end=3,
                    ),
                ),
            )
        )
    return contract.PublicInputBundle(dataset_id=_id(), cases=tuple(cases))


def _funnel(*, surfaced: int = 1) -> contract.StageFunnel:
    return contract.StageFunnel(
        input_candidates=3,
        retrieved_candidates=3,
        grounded_candidates=3,
        adjudicated_candidates=1,
        surfaced_candidates=surfaced,
    )


def _ref(
    case: contract.PublicInputCase,
    role: str,
    line_start: int,
    line_end: int | None = None,
    *,
    event_group_id: str | None = None,
) -> RuntimeEvidenceRef:
    document = case.documents[0] if role == "B" else case.documents[1]
    return RuntimeEvidenceRef(
        role=role,
        document_id=document.document_id,
        line_start=line_start,
        line_end=line_start if line_end is None else line_end,
        event_group_id=event_group_id,
    )


def _snapshot(
    case: contract.PublicInputCase,
    *,
    outcome: str = "conflict",
    citations: tuple[RuntimeEvidenceRef, ...] | None = None,
    **updates,
) -> RuntimeCaseSnapshot:
    if citations is None:
        citations = (
            _ref(case, "B", 1),
            _ref(case, "C", 1, event_group_id=_id()),
            _ref(case, "C", 2, event_group_id=_id()),
        )
    values = {
        "case_id": case.case_id,
        "stage_status": "completed",
        "material_coverage": "complete",
        "explanation_coverage": "complete",
        "report_collections_complete": True,
        "citation_refs_complete": True,
        "trace_count": 1,
        "final_outcome": outcome,
        "formal_issue_count": 1 if outcome == "conflict" else 0,
        "review_clue_count": (
            1 if outcome in {"needs_confirmation", "unverifiable"} else 0
        ),
        "provisional_clue_count": 0,
        "citations": citations,
        "report_evidence": (
            citations
            if outcome in {"conflict", "needs_confirmation", "unverifiable"}
            else ()
        ),
        "stage_funnel": _funnel(surfaced=0 if outcome == "no_issue" else 1),
    }
    values.update(updates)
    return RuntimeCaseSnapshot.model_validate(values)


def test_formal_projection_expands_line_anchors_and_preserves_snapshot_binding():
    case = _public().cases[0]

    prediction = project_case_prediction(case, _snapshot(case))

    assert prediction.outcome == "conflict"
    assert prediction.surface == "formal_issue"
    assert [row.role for row in prediction.citations] == ["B", "C", "C"]
    for citation in prediction.citations:
        document = next(
            row for row in case.documents if row.document_id == citation.document_id
        )
        assert citation.snapshot_version == document.version
        assert citation.snapshot_sha256 == document.sha256


@pytest.mark.parametrize("mismatch", ("empty", "strict_subset", "extra_trace_c"))
def test_surface_projection_requires_exact_report_evidence_trace_binding(mismatch):
    case = _public().cases[0]
    report_refs = (
        _ref(case, "B", 1),
        _ref(case, "C", 1, event_group_id=_id()),
        _ref(case, "C", 2, event_group_id=_id()),
    )
    trace_refs = report_refs
    if mismatch == "empty":
        report_refs = ()
    elif mismatch == "strict_subset":
        report_refs = report_refs[:-1]
    else:
        trace_refs = (
            *trace_refs,
            _ref(case, "C", 3, event_group_id=_id()),
        )

    prediction = project_case_prediction(
        case,
        _snapshot(
            case,
            citations=trace_refs,
            report_evidence=report_refs,
        ),
    )

    assert prediction.outcome == "unassessed"
    assert prediction.citations == ()


@pytest.mark.parametrize(
    "updates",
    (
        {"stage_status": "partial"},
        {"stage_status": "failed"},
        {"material_coverage": "partial"},
        {"explanation_coverage": "unknown"},
        {"report_collections_complete": False},
        {"citation_refs_complete": False},
        {"trace_count": 0},
        {"trace_count": 2},
    ),
)
def test_incomplete_or_ambiguous_runtime_state_fails_closed(updates):
    case = _public().cases[0]
    prediction = project_case_prediction(case, _snapshot(case, **updates))
    assert prediction.outcome == "unassessed"
    assert prediction.surface == "none"
    assert prediction.citations == ()


@pytest.mark.parametrize("role", ("G", "X", "P"))
def test_formal_projection_rejects_explanation_or_provisional_roles(role):
    case = _public().cases[0]
    refs = (
        _ref(case, "B", 1),
        _ref(case, "C", 1, event_group_id=_id()),
        _ref(case, "C", 2, event_group_id=_id()),
        _ref(case, role, 3),
    )
    prediction = project_case_prediction(case, _snapshot(case, citations=refs))
    assert prediction.outcome == "unassessed"


def test_formal_projection_requires_two_distinct_nonoverlapping_events():
    case = _public().cases[0]
    same_event = _id()
    same_group = (
        _ref(case, "B", 1),
        _ref(case, "C", 1, event_group_id=same_event),
        _ref(case, "C", 2, event_group_id=same_event),
    )
    assert (
        project_case_prediction(case, _snapshot(case, citations=same_group)).outcome
        == "unassessed"
    )

    overlapping = (
        _ref(case, "B", 1),
        _ref(case, "C", 1, 2, event_group_id=_id()),
        _ref(case, "C", 2, 3, event_group_id=_id()),
    )
    assert (
        project_case_prediction(case, _snapshot(case, citations=overlapping)).outcome
        == "unassessed"
    )


def test_unknown_document_anchor_and_conflicting_roles_fail_closed():
    case = _public().cases[0]
    unknown_document = RuntimeEvidenceRef(
        role="B",
        document_id=_id(),
        line_start=1,
        line_end=1,
    )
    assert (
        project_case_prediction(
            case, _snapshot(case, citations=(unknown_document,))
        ).outcome
        == "unassessed"
    )

    missing_anchor = (_ref(case, "B", 99),)
    assert (
        project_case_prediction(
            case, _snapshot(case, citations=missing_anchor)
        ).outcome
        == "unassessed"
    )

    overbroad = (_ref(case, "B", 1, 99),)
    assert (
        project_case_prediction(case, _snapshot(case, citations=overbroad)).outcome
        == "unassessed"
    )

    target = case.documents[1]
    conflicting = (
        RuntimeEvidenceRef(
            role="B", document_id=target.document_id, line_start=1, line_end=1
        ),
        _ref(case, "C", 1, event_group_id=_id()),
        _ref(case, "C", 2, event_group_id=_id()),
    )
    assert (
        project_case_prediction(case, _snapshot(case, citations=conflicting)).outcome
        == "unassessed"
    )


@pytest.mark.parametrize("outcome", ("needs_confirmation", "unverifiable"))
def test_review_clue_projects_to_indeterminate(outcome):
    case = _public().cases[0]
    prediction = project_case_prediction(
        case,
        _snapshot(
            case,
            outcome=outcome,
            citations=(_ref(case, "P", 1),),
        ),
    )
    assert prediction.outcome == "indeterminate"
    assert prediction.surface == "review_clue"
    assert [row.role for row in prediction.citations] == ["P"]


def test_provisional_clue_and_empty_review_evidence_are_unassessed():
    case = _public().cases[0]
    provisional = _snapshot(
        case,
        outcome="needs_confirmation",
        citations=(_ref(case, "P", 1),),
        review_clue_count=0,
        provisional_clue_count=1,
    )
    assert project_case_prediction(case, provisional).outcome == "unassessed"

    empty = _snapshot(case, outcome="unverifiable", citations=())
    assert project_case_prediction(case, empty).outcome == "unassessed"


def test_explained_no_issue_preserves_gx_evidence_but_rejects_provisional_support():
    case = _public().cases[0]
    refs = (
        _ref(case, "B", 1),
        _ref(case, "G", 1),
        _ref(case, "X", 2),
    )
    prediction = project_case_prediction(
        case, _snapshot(case, outcome="no_issue", citations=refs)
    )
    assert prediction.outcome == "no_issue"
    assert prediction.surface == "none"
    assert [row.role for row in prediction.citations] == ["B", "G", "X"]

    provisional = _snapshot(
        case,
        outcome="no_issue",
        citations=(*refs, _ref(case, "P", 3)),
    )
    assert project_case_prediction(case, provisional).outcome == "unassessed"

    surfaced = _snapshot(
        case,
        outcome="no_issue",
        citations=refs,
        review_clue_count=1,
    )
    assert project_case_prediction(case, surfaced).outcome == "unassessed"


def test_case_id_mismatch_is_rejected_instead_of_silently_projected():
    case = _public().cases[0]
    snapshot = _snapshot(case).model_copy(update={"case_id": _id()})
    with pytest.raises(PredictionProjectionError, match="runtime_case_id_mismatch"):
        project_case_prediction(case, snapshot)


def test_artifact_binds_frozen_inputs_and_marks_partial_on_any_unassessed():
    public = _public(case_count=2)
    complete = _snapshot(public.cases[0], outcome="no_issue", citations=())
    partial = _snapshot(public.cases[1], stage_status="partial")

    artifact = build_prediction_artifact(
        public,
        (complete, partial),
        run_id="sealed-run-1",
        repeat_id="repeat-1",
        author_setup_sha256=SHA_A,
        execution_freeze_sha256=SHA_B,
        runtime_provenance_sha256=SHA_D,
    )

    assert artifact.dataset_id == public.dataset_id
    assert artifact.public_input_sha256 == contract.canonical_sha256(public)
    assert artifact.author_setup_sha256 == SHA_A
    assert artifact.execution_freeze_sha256 == SHA_B
    assert artifact.runtime_provenance_sha256 == SHA_D
    assert artifact.prediction.run_status == "partial"
    assert [row.outcome for row in artifact.prediction.predictions] == [
        "no_issue",
        "unassessed",
    ]


def test_artifact_requires_exactly_one_snapshot_for_every_public_case():
    public = _public(case_count=2)
    first = _snapshot(public.cases[0])
    kwargs = {
        "run_id": "sealed-run-1",
        "repeat_id": "repeat-1",
        "author_setup_sha256": SHA_A,
        "execution_freeze_sha256": SHA_B,
        "runtime_provenance_sha256": SHA_D,
    }
    with pytest.raises(
        PredictionProjectionError,
        match="runtime_snapshots_must_cover_every_public_case",
    ):
        build_prediction_artifact(public, (first,), **kwargs)
    with pytest.raises(PredictionProjectionError, match="duplicate_runtime_case_id"):
        build_prediction_artifact(public, (first, first), **kwargs)
