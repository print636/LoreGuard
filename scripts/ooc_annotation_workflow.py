"""Offline workflow for two-person, independent OOC holdout annotation.

The annotation files, adjudication file and private gold are deliberately
separate envelopes.  This module never calls a model, the LoreGuard API or a
network service.  It also never treats an AI-generated judgement as a human
annotation: reviewer files carry an explicit independence declaration and are
only accepted when two distinct reviewer identities cover the same frozen
public bundle.

Private annotations and gold should live outside the repository.  Only the
public input bundle, document snapshots and an HMAC commitment are suitable
for version control.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import ooc_eval_contract as contract


ANNOTATION_SCHEMA_VERSION = "loreguard-ooc-reviewer-annotation-v1"
ADJUDICATION_SCHEMA_VERSION = "loreguard-ooc-adjudication-v1"
AGREEMENT_SCHEMA_VERSION = "loreguard-ooc-annotation-agreement-v1"
CLOSED_COMMITMENT_SCHEMA_VERSION = "loreguard-ooc-closed-commitment-v2"
CLOSED_COMMITMENT_DOMAIN = b"LoreGuard/OOC/closed-evaluation/v2\x00"
SCORER_MANIFEST_SCHEMA_VERSION = "loreguard-ooc-scorer-manifest-v1"
SINGLE_RUN_FORMAL_GATE_POLICY_ID = "single-run-formal-v1"

# V1 authorizes exactly one fixed real run.  Every threshold is written out so
# a later change to the contract defaults cannot silently change the formal
# decision rule.  Cross-repeat consistency is undefined for one run and is
# therefore deliberately disabled rather than treated as a failed metric.
SINGLE_RUN_FORMAL_GATE_POLICY = contract.GatePolicy(
    min_formal_precision=1.0,
    min_worst_case_recall=1.0,
    min_hard_negative_specificity=1.0,
    min_bc_completeness=1.0,
    max_forbidden_reference_rate=0.0,
    min_joint_accuracy=1.0,
    max_sliced_formal_false_positive_rate=0.0,
    min_coverage=1.0,
    min_outcome_consistency=None,
    minimum_repeats=1,
    max_invalid_citation_count=0,
)
FORMAL_GATE_POLICY_REGISTRY: Mapping[str, contract.GatePolicy] = MappingProxyType(
    {SINGLE_RUN_FORMAL_GATE_POLICY_ID: SINGLE_RUN_FORMAL_GATE_POLICY}
)

Confidence = Annotated[int, Field(strict=True, ge=1, le=5)]
ReviewNote = Annotated[str, Field(strict=True, min_length=1, max_length=8_000)]
CharacterDimension = Literal[
    "core_trait",
    "stable_preference",
    "speech_pattern",
    "value_boundary",
    "relationship_attitude",
    "motivation_goal",
]
ConflictLevel = Literal["L0", "L1", "L2", "L3"]
MaterialCoverage = Literal["complete", "partial", "missing"]
_UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
_TRAIT_DIMENSION = {
    "core_personality": "core_trait",
    "preference": "stable_preference",
    "speech_pattern": "speech_pattern",
    "value": "value_boundary",
    "relationship_attitude": "relationship_attitude",
    "motivation_goal": "motivation_goal",
}


class EvidenceAssignment(contract.StrictModel):
    evidence_id: contract.OpaqueId
    role: contract.EvidenceRole


class ExplanationLink(contract.StrictModel):
    support_evidence_id: contract.OpaqueId
    applicable_current_ids: tuple[contract.OpaqueId, ...] = Field(
        min_length=1, max_length=contract.MAX_EVIDENCE_PER_CASE
    )
    causal_relation: Literal["explicit", "bounded", "ambiguous"]
    temporal_relation: Literal["before", "active_during", "after", "unknown"]

    @model_validator(mode="after")
    def validate_unique_targets(self) -> ExplanationLink:
        if len(self.applicable_current_ids) != len(set(self.applicable_current_ids)):
            raise ValueError("duplicate_explanation_current_evidence_id")
        return self


def _validate_label_shape(
    *,
    outcome: contract.GoldOutcome,
    surface: contract.Surface,
    evidence: Sequence[EvidenceAssignment],
    event_groups: Sequence[contract.IndependentEventGroup],
    explanation_links: Sequence[ExplanationLink],
    conflict_level: ConflictLevel,
    material_coverage: MaterialCoverage,
) -> None:
    expected_surface = {
        "conflict": "formal_issue",
        "no_issue": "none",
        "indeterminate": "review_clue",
    }[outcome]
    if surface != expected_surface:
        raise ValueError("outcome_surface_pair_invalid")
    expected_levels: Mapping[str, set[str]] = {
        "conflict": {"L3"},
        "no_issue": {"L0"},
        "indeterminate": {"L1", "L2"},
    }
    if conflict_level not in expected_levels[outcome]:
        raise ValueError("outcome_conflict_level_pair_invalid")
    if material_coverage != "complete" and outcome != "indeterminate":
        raise ValueError("incomplete_material_must_be_indeterminate")

    evidence_ids = tuple(row.evidence_id for row in evidence)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise ValueError("evidence_id_may_have_only_one_role")
    role_ids = {
        role: tuple(row.evidence_id for row in evidence if row.role == role)
        for role in ("B", "C", "G", "X", "P")
    }
    group_ids = tuple(row.event_group_id for row in event_groups)
    grouped = tuple(
        evidence_id for group in event_groups for evidence_id in group.evidence_ids
    )
    if len(group_ids) != len(set(group_ids)):
        raise ValueError("duplicate_independent_event_group_id")
    if len(grouped) != len(set(grouped)):
        raise ValueError("evidence_cannot_span_independent_event_groups")
    if not set(grouped) <= set(evidence_ids):
        raise ValueError("event_group_evidence_not_assigned")

    explanation_support = {link.support_evidence_id for link in explanation_links}
    if len(explanation_support) != len(explanation_links):
        raise ValueError("duplicate_explanation_support_evidence_id")
    valid_support = set(role_ids["G"] + role_ids["X"] + role_ids["P"])
    if not explanation_support <= valid_support:
        raise ValueError("explanation_link_requires_G_X_or_P_support")
    valid_current = set(role_ids["C"])
    if any(
        not set(link.applicable_current_ids) <= valid_current
        for link in explanation_links
    ):
        raise ValueError("explanation_link_target_requires_C_evidence")
    if set(role_ids["G"] + role_ids["X"]) != explanation_support & set(
        role_ids["G"] + role_ids["X"]
    ):
        raise ValueError("every_G_or_X_requires_an_explanation_link")

    if outcome == "conflict":
        if (
            not role_ids["B"]
            or len(role_ids["C"]) < 2
            or role_ids["G"]
            or role_ids["X"]
            or role_ids["P"]
        ):
            raise ValueError("formal_annotation_requires_B_two_C_and_no_explanation")
        event_by_evidence = {
            evidence_id: group.event_group_id
            for group in event_groups
            for evidence_id in group.evidence_ids
        }
        current_groups = {event_by_evidence.get(value) for value in role_ids["C"]}
        if None in current_groups or len(current_groups) < 2:
            raise ValueError("formal_annotation_requires_two_independent_current_events")
    elif outcome == "indeterminate" and not evidence:
        raise ValueError("indeterminate_annotation_requires_evidence")

    role_by_evidence = {row.evidence_id: row.role for row in evidence}
    for link in explanation_links:
        role = role_by_evidence[link.support_evidence_id]
        if role in {"G", "X"} and link.causal_relation == "ambiguous":
            raise ValueError("ambiguous_support_must_be_classified_as_P")
        if role == "G" and link.temporal_relation != "before":
            raise ValueError("growth_explanation_must_precede_current_event")
        if role == "X" and link.temporal_relation != "active_during":
            raise ValueError("exception_must_be_active_during_current_event")


class ReviewerCaseAnnotation(contract.StrictModel):
    case_id: contract.OpaqueId
    dimension: CharacterDimension
    outcome: contract.GoldOutcome
    surface: contract.Surface
    conflict_level: ConflictLevel
    material_coverage: MaterialCoverage
    evidence: tuple[EvidenceAssignment, ...] = Field(
        default=(), max_length=contract.MAX_EVIDENCE_PER_CASE
    )
    independent_event_groups: tuple[contract.IndependentEventGroup, ...] = Field(
        default=(), max_length=contract.MAX_EVIDENCE_PER_CASE
    )
    explanation_links: tuple[ExplanationLink, ...] = Field(
        default=(), max_length=contract.MAX_EVIDENCE_PER_CASE
    )
    phenomena: tuple[contract.OpaqueTag, ...] = Field(min_length=1, max_length=64)
    reason_codes: tuple[contract.OpaqueTag, ...] = Field(min_length=1, max_length=64)
    confidence: Confidence
    review_note: ReviewNote

    @model_validator(mode="after")
    def validate_annotation(self) -> ReviewerCaseAnnotation:
        if len(self.phenomena) != len(set(self.phenomena)):
            raise ValueError("duplicate_phenomenon")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("duplicate_reason_code")
        _validate_label_shape(
            outcome=self.outcome,
            surface=self.surface,
            evidence=self.evidence,
            event_groups=self.independent_event_groups,
            explanation_links=self.explanation_links,
            conflict_level=self.conflict_level,
            material_coverage=self.material_coverage,
        )
        return self


class ReviewerAnnotationBundle(contract.StrictModel):
    schema_version: Literal["loreguard-ooc-reviewer-annotation-v1"] = (
        ANNOTATION_SCHEMA_VERSION
    )
    dataset_id: contract.OpaqueId
    public_input_sha256: contract.Sha256
    reviewer_kind: Literal["human"]
    reviewer_id: contract.OpaqueId
    manual_version: contract.OpaqueTag
    blinded_to_peer: Literal[True]
    blinded_to_system_prediction: Literal[True]
    independence_declaration: Literal[True]
    cases: tuple[ReviewerCaseAnnotation, ...] = Field(
        min_length=1, max_length=contract.MAX_CASES
    )

    @model_validator(mode="after")
    def validate_unique_cases(self) -> ReviewerAnnotationBundle:
        case_ids = tuple(row.case_id for row in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("duplicate_annotation_case_id")
        return self


class AdjudicatedCase(contract.StrictModel):
    case_id: contract.OpaqueId
    dimension: CharacterDimension
    final_outcome: contract.GoldOutcome
    final_surface: contract.Surface
    conflict_level: ConflictLevel
    material_coverage: MaterialCoverage
    required_evidence: tuple[EvidenceAssignment, ...] = Field(
        default=(), max_length=contract.MAX_EVIDENCE_PER_CASE
    )
    forbidden_evidence_ids: tuple[contract.OpaqueId, ...] = Field(
        default=(), max_length=contract.MAX_EVIDENCE_PER_CASE
    )
    independent_event_groups: tuple[contract.IndependentEventGroup, ...] = Field(
        default=(), max_length=contract.MAX_EVIDENCE_PER_CASE
    )
    explanation_links: tuple[ExplanationLink, ...] = Field(
        default=(), max_length=contract.MAX_EVIDENCE_PER_CASE
    )
    phenomena: tuple[contract.OpaqueTag, ...] = Field(min_length=1, max_length=64)
    reason_codes: tuple[contract.OpaqueTag, ...] = Field(min_length=1, max_length=64)
    resolution_mode: Literal["reviewer_consensus", "third_human"]
    adjudicator_id: contract.OpaqueId | None = None
    adjudicator_kind: Literal["human"] | None = None
    adjudicator_blinded_to_system_prediction: Literal[True] | None = None
    resolution_note: ReviewNote

    @model_validator(mode="after")
    def validate_resolution(self) -> AdjudicatedCase:
        evidence_ids = tuple(row.evidence_id for row in self.required_evidence)
        if len(self.forbidden_evidence_ids) != len(set(self.forbidden_evidence_ids)):
            raise ValueError("duplicate_forbidden_evidence_id")
        if set(evidence_ids) & set(self.forbidden_evidence_ids):
            raise ValueError("evidence_cannot_be_required_and_forbidden")
        if len(self.phenomena) != len(set(self.phenomena)):
            raise ValueError("duplicate_phenomenon")
        if len(self.reason_codes) != len(set(self.reason_codes)):
            raise ValueError("duplicate_reason_code")
        third_party_fields = (
            self.adjudicator_id,
            self.adjudicator_kind,
            self.adjudicator_blinded_to_system_prediction,
        )
        if self.resolution_mode == "third_human" and any(
            value is None for value in third_party_fields
        ):
            raise ValueError("third_human_resolution_requires_adjudicator_id")
        if self.resolution_mode == "reviewer_consensus" and any(
            value is not None for value in third_party_fields
        ):
            raise ValueError("reviewer_consensus_cannot_claim_third_human")
        _validate_label_shape(
            outcome=self.final_outcome,
            surface=self.final_surface,
            evidence=self.required_evidence,
            event_groups=self.independent_event_groups,
            explanation_links=self.explanation_links,
            conflict_level=self.conflict_level,
            material_coverage=self.material_coverage,
        )
        return self


class AdjudicationBundle(contract.StrictModel):
    schema_version: Literal["loreguard-ooc-adjudication-v1"] = (
        ADJUDICATION_SCHEMA_VERSION
    )
    dataset_id: contract.OpaqueId
    public_input_sha256: contract.Sha256
    reviewer_ids: tuple[contract.OpaqueId, contract.OpaqueId]
    annotation_sha256: tuple[contract.Sha256, contract.Sha256]
    consensus_confirmed_by: tuple[contract.OpaqueId, contract.OpaqueId]
    cases: tuple[AdjudicatedCase, ...] = Field(
        min_length=1, max_length=contract.MAX_CASES
    )

    @model_validator(mode="after")
    def validate_resolution_bundle(self) -> AdjudicationBundle:
        if len(set(self.reviewer_ids)) != 2:
            raise ValueError("adjudication_requires_two_distinct_reviewers")
        if set(self.consensus_confirmed_by) != set(self.reviewer_ids):
            raise ValueError("both_reviewers_must_confirm_consensus")
        case_ids = tuple(row.case_id for row in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("duplicate_adjudication_case_id")
        return self


class CaseAgreement(contract.StrictModel):
    case_id: contract.OpaqueId
    outcome_agrees: bool
    surface_agrees: bool
    dimension_agrees: bool
    conflict_level_agrees: bool
    material_coverage_agrees: bool
    evidence_agrees: bool
    event_groups_agree: bool
    explanation_links_agree: bool
    phenomena_agree: bool
    reason_codes_agree: bool


class AnnotationAgreementReport(contract.StrictModel):
    schema_version: Literal["loreguard-ooc-annotation-agreement-v1"] = (
        AGREEMENT_SCHEMA_VERSION
    )
    dataset_id: contract.OpaqueId
    public_input_sha256: contract.Sha256
    reviewer_ids: tuple[contract.OpaqueId, contract.OpaqueId]
    case_count: int = Field(strict=True, ge=1)
    exact_label_agreement: float = Field(ge=0.0, le=1.0)
    outcome_kappa: float = Field(ge=-1.0, le=1.0)
    surface_kappa: float = Field(ge=-1.0, le=1.0)
    mean_evidence_jaccard: float = Field(ge=0.0, le=1.0)
    disagreement_case_ids: tuple[contract.OpaqueId, ...]
    cases: tuple[CaseAgreement, ...]


class ScorerSourceManifest(contract.StrictModel):
    """Source identities that determine the formal scoring result."""

    schema_version: Literal["loreguard-ooc-scorer-manifest-v1"] = (
        SCORER_MANIFEST_SCHEMA_VERSION
    )
    scorer_source_sha256: contract.Sha256
    eval_contract_source_sha256: contract.Sha256


class ClosedGoldCommitment(contract.StrictModel):
    schema_version: Literal["loreguard-ooc-closed-commitment-v2"] = (
        CLOSED_COMMITMENT_SCHEMA_VERSION
    )
    scheme: Literal["hmac-sha256-v1"]
    public_input_sha256: contract.Sha256
    author_setup_sha256: contract.Sha256
    execution_freeze_sha256: contract.Sha256
    run_config_sha256: contract.Sha256
    annotation_sha256: tuple[contract.Sha256, contract.Sha256]
    adjudication_sha256: contract.Sha256
    manual_version: contract.OpaqueTag
    gate_policy_id: Literal["single-run-formal-v1"]
    gate_policy: contract.GatePolicy
    scorer_manifest: ScorerSourceManifest
    digest: contract.Sha256


class AnnotationWorkflowError(ValueError):
    """Stable error for invalid bindings or unsafe workflow transitions."""


def _path_chain_has_reparse_point(
    path: str | Path,
    *,
    stop: str | Path | None = None,
) -> bool:
    """Inspect lexical ancestors without resolving through a reparse point."""

    absolute = Path(path).absolute()
    stop_path = Path(stop).absolute() if stop is not None else None
    candidates = [absolute, *absolute.parents]
    for candidate in candidates:
        try:
            is_junction = bool(
                getattr(candidate, "is_junction", lambda: False)()
            )
            if candidate.is_symlink() or is_junction:
                return True
        except OSError:
            return True
        if stop_path is not None and candidate == stop_path:
            break
    return False


def validate_annotation_binding(
    public: contract.PublicInputBundle,
    annotation: ReviewerAnnotationBundle,
) -> None:
    if annotation.dataset_id != public.dataset_id:
        raise AnnotationWorkflowError("annotation_dataset_id_mismatch")
    public_digest = contract.canonical_sha256(public)
    if annotation.public_input_sha256 != public_digest:
        raise AnnotationWorkflowError("annotation_public_input_binding_mismatch")
    public_cases = {row.case_id: row for row in public.cases}
    annotated_cases = {row.case_id: row for row in annotation.cases}
    if set(annotated_cases) != set(public_cases):
        raise AnnotationWorkflowError("annotation_must_cover_every_public_case_once")
    for case_id, row in annotated_cases.items():
        known = {evidence.evidence_id for evidence in public_cases[case_id].evidence_catalog}
        referenced = {evidence.evidence_id for evidence in row.evidence}
        referenced.update(
            evidence_id
            for group in row.independent_event_groups
            for evidence_id in group.evidence_ids
        )
        referenced.update(link.support_evidence_id for link in row.explanation_links)
        referenced.update(
            evidence_id
            for link in row.explanation_links
            for evidence_id in link.applicable_current_ids
        )
        if not referenced <= known:
            raise AnnotationWorkflowError("annotation_evidence_id_not_in_public_catalog")


def _validate_case_evidence_against_setup(
    public_case: contract.PublicInputCase,
    setup_case: Any,
    evidence: Sequence[EvidenceAssignment],
) -> None:
    anchors = {row.evidence_id: row for row in public_case.evidence_catalog}
    setup_documents = {row.document_id: row for row in setup_case.documents}
    selector = setup_case.candidate_selector
    for assignment in evidence:
        anchor = anchors.get(assignment.evidence_id)
        if anchor is None:
            raise AnnotationWorkflowError(
                "annotation_evidence_id_not_in_public_catalog"
            )
        document = setup_documents.get(anchor.document_id)
        if document is None:
            raise AnnotationWorkflowError("annotation_setup_document_unknown")
        if assignment.role == "B" and not (
            anchor.document_id == selector.source_document_id
            and selector.source_line_start <= anchor.line_start
            and anchor.line_end <= selector.source_line_end
        ):
            raise AnnotationWorkflowError("baseline_evidence_must_match_frozen_selector")
        if assignment.role == "C" and anchor.document_id != setup_case.target_document_id:
            raise AnnotationWorkflowError("current_evidence_must_come_from_target_document")
        if assignment.role == "G" and document.phase != "baseline":
            raise AnnotationWorkflowError("growth_evidence_must_precede_target_document")


def _validate_annotation_setup_binding(
    public: contract.PublicInputBundle,
    setup: Any,
    annotation: ReviewerAnnotationBundle,
) -> None:
    """Bind human dimensions and semantic evidence roles to author setup."""

    validate_annotation_binding(public, annotation)
    public_cases = {row.case_id: row for row in public.cases}
    setup_cases = {row.case_id: row for row in setup.cases}
    annotation_cases = {row.case_id: row for row in annotation.cases}
    if set(setup_cases) != set(public_cases):
        raise AnnotationWorkflowError("annotation_setup_case_set_mismatch")
    for case_id, row in annotation_cases.items():
        setup_case = setup_cases[case_id]
        expected_dimension = _TRAIT_DIMENSION[setup_case.candidate_selector.trait_type]
        if row.dimension != expected_dimension:
            raise AnnotationWorkflowError("annotation_dimension_mismatch")
        _validate_case_evidence_against_setup(
            public_cases[case_id], setup_case, row.evidence
        )


def _validate_adjudication_setup_binding(
    public: contract.PublicInputBundle,
    setup: Any,
    adjudication: AdjudicationBundle,
) -> None:
    public_cases = {row.case_id: row for row in public.cases}
    setup_cases = {row.case_id: row for row in setup.cases}
    adjudicated_cases = {row.case_id: row for row in adjudication.cases}
    if set(adjudicated_cases) != set(public_cases) or set(setup_cases) != set(
        public_cases
    ):
        raise AnnotationWorkflowError("adjudication_setup_case_set_mismatch")
    for row in adjudication.cases:
        setup_case = setup_cases[row.case_id]
        expected_dimension = _TRAIT_DIMENSION[setup_case.candidate_selector.trait_type]
        if row.dimension != expected_dimension:
            raise AnnotationWorkflowError("adjudication_dimension_mismatch")
        _validate_case_evidence_against_setup(
            public_cases[row.case_id], setup_case, row.required_evidence
        )


def validate_exhaustive_line_catalog(
    public: contract.PublicInputBundle,
    bundle_root: str | Path,
) -> None:
    """Verify frozen document bytes and an oracle-independent line catalogue.

    Every non-empty UTF-8 line must have exactly one single-line anchor and no
    anchor may point at a blank line.  This prevents a hand-picked public
    evidence catalogue from revealing where the private answer is located.
    """

    root_text = str(bundle_root)
    if "://" in root_text or root_text.startswith(("\\\\", "//")):
        raise AnnotationWorkflowError("network_locations_are_forbidden")
    root = Path(bundle_root)
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise AnnotationWorkflowError("public_bundle_root_unreadable") from exc
    if not resolved_root.is_dir() or _path_chain_has_reparse_point(root):
        raise AnnotationWorkflowError("public_bundle_root_invalid")
    if not _UUID4.fullmatch(public.dataset_id):
        raise AnnotationWorkflowError("public_dataset_id_must_be_neutral_uuid4")
    for case in public.cases:
        if not re.fullmatch(r"closed-(?:pilot|holdout)-v[1-9][0-9]*", case.split_id):
            raise AnnotationWorkflowError("public_split_id_must_be_neutral")
        identities = (
            case.case_id,
            case.group_id,
            case.world_id,
            case.axis.axis_id,
            *(document.document_id for document in case.documents),
            *(anchor.evidence_id for anchor in case.evidence_catalog),
        )
        if any(not _UUID4.fullmatch(value) for value in identities):
            raise AnnotationWorkflowError("public_identifiers_must_be_neutral_uuid4")
        anchors_by_document: dict[str, list[contract.EvidenceAnchor]] = {}
        for anchor in case.evidence_catalog:
            anchors_by_document.setdefault(anchor.document_id, []).append(anchor)
        for document in case.documents:
            path_parts = Path(document.path).parts
            if (
                len(path_parts) != 3
                or path_parts[0] != "cases"
                or path_parts[1] != case.group_id
                or Path(path_parts[2]).suffix.lower() != ".md"
                or Path(path_parts[2]).stem.lower() != document.document_id
            ):
                raise AnnotationWorkflowError("public_document_path_must_be_neutral")
            path = resolved_root.joinpath(*Path(document.path).parts)
            try:
                if _path_chain_has_reparse_point(path, stop=resolved_root):
                    raise AnnotationWorkflowError("public_document_symlink_forbidden")
                resolved = path.resolve(strict=True)
                if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
                    raise AnnotationWorkflowError("public_document_path_escapes_bundle")
                size = resolved.stat().st_size
                if size > contract.MAX_JSON_BYTES:
                    raise AnnotationWorkflowError("public_document_too_large")
                payload = resolved.read_bytes()
            except OSError as exc:
                raise AnnotationWorkflowError("public_document_unreadable") from exc
            if hashlib.sha256(payload).hexdigest() != document.sha256:
                raise AnnotationWorkflowError("public_document_sha256_mismatch")
            if b"\r" in payload:
                raise AnnotationWorkflowError("public_document_must_use_lf")
            try:
                lines = payload.decode("utf-8").splitlines()
            except UnicodeDecodeError as exc:
                raise AnnotationWorkflowError("public_document_must_be_utf8") from exc
            expected = {index for index, line in enumerate(lines, start=1) if line.strip()}
            anchors = anchors_by_document.get(document.document_id, [])
            actual: list[int] = []
            for anchor in anchors:
                if anchor.line_start != anchor.line_end:
                    raise AnnotationWorkflowError("public_catalog_anchor_must_be_single_line")
                if anchor.line_start not in expected:
                    raise AnnotationWorkflowError("public_catalog_anchor_must_target_nonempty_line")
                actual.append(anchor.line_start)
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise AnnotationWorkflowError("public_catalog_must_cover_each_nonempty_line_once")


def _kappa(left: Sequence[str], right: Sequence[str]) -> float:
    if len(left) != len(right) or not left:
        raise AnnotationWorkflowError("kappa_requires_equal_nonempty_sequences")
    observed = sum(a == b for a, b in zip(left, right)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    labels = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[label] / len(left)) * (right_counts[label] / len(right))
        for label in labels
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def _evidence_identity(row: ReviewerCaseAnnotation) -> set[tuple[str, str]]:
    return {(item.role, item.evidence_id) for item in row.evidence}


def _event_identity(row: ReviewerCaseAnnotation) -> set[frozenset[str]]:
    return {frozenset(group.evidence_ids) for group in row.independent_event_groups}


def _explanation_identity(
    row: ReviewerCaseAnnotation,
) -> set[tuple[str, frozenset[str], str, str]]:
    return {
        (
            link.support_evidence_id,
            frozenset(link.applicable_current_ids),
            link.causal_relation,
            link.temporal_relation,
        )
        for link in row.explanation_links
    }


def compare_annotations(
    public: contract.PublicInputBundle,
    left: ReviewerAnnotationBundle,
    right: ReviewerAnnotationBundle,
) -> AnnotationAgreementReport:
    validate_annotation_binding(public, left)
    validate_annotation_binding(public, right)
    if left.reviewer_id == right.reviewer_id:
        raise AnnotationWorkflowError("two_distinct_reviewers_required")
    if left.manual_version != right.manual_version:
        raise AnnotationWorkflowError("reviewers_must_use_same_manual_version")
    left_by_case = {row.case_id: row for row in left.cases}
    right_by_case = {row.case_id: row for row in right.cases}
    ordered_ids = tuple(row.case_id for row in public.cases)
    cases: list[CaseAgreement] = []
    evidence_scores: list[float] = []
    disagreements: list[str] = []
    for case_id in ordered_ids:
        first = left_by_case[case_id]
        second = right_by_case[case_id]
        first_evidence = _evidence_identity(first)
        second_evidence = _evidence_identity(second)
        union = first_evidence | second_evidence
        evidence_jaccard = len(first_evidence & second_evidence) / len(union) if union else 1.0
        evidence_scores.append(evidence_jaccard)
        row = CaseAgreement(
            case_id=case_id,
            outcome_agrees=first.outcome == second.outcome,
            surface_agrees=first.surface == second.surface,
            dimension_agrees=first.dimension == second.dimension,
            conflict_level_agrees=first.conflict_level == second.conflict_level,
            material_coverage_agrees=(
                first.material_coverage == second.material_coverage
            ),
            evidence_agrees=first_evidence == second_evidence,
            event_groups_agree=_event_identity(first) == _event_identity(second),
            explanation_links_agree=(
                _explanation_identity(first) == _explanation_identity(second)
            ),
            phenomena_agree=set(first.phenomena) == set(second.phenomena),
            reason_codes_agree=set(first.reason_codes) == set(second.reason_codes),
        )
        cases.append(row)
        if not all(
            (
                row.outcome_agrees,
                row.surface_agrees,
                row.dimension_agrees,
                row.conflict_level_agrees,
                row.material_coverage_agrees,
                row.evidence_agrees,
                row.event_groups_agree,
                row.explanation_links_agree,
                row.phenomena_agree,
                row.reason_codes_agree,
            )
        ):
            disagreements.append(case_id)
    exact_labels = sum(
        left_by_case[case_id].outcome == right_by_case[case_id].outcome
        and left_by_case[case_id].surface == right_by_case[case_id].surface
        for case_id in ordered_ids
    ) / len(ordered_ids)
    return AnnotationAgreementReport(
        dataset_id=public.dataset_id,
        public_input_sha256=contract.canonical_sha256(public),
        reviewer_ids=(left.reviewer_id, right.reviewer_id),
        case_count=len(ordered_ids),
        exact_label_agreement=exact_labels,
        outcome_kappa=_kappa(
            [left_by_case[case_id].outcome for case_id in ordered_ids],
            [right_by_case[case_id].outcome for case_id in ordered_ids],
        ),
        surface_kappa=_kappa(
            [left_by_case[case_id].surface for case_id in ordered_ids],
            [right_by_case[case_id].surface for case_id in ordered_ids],
        ),
        mean_evidence_jaccard=sum(evidence_scores) / len(evidence_scores),
        disagreement_case_ids=tuple(disagreements),
        cases=tuple(cases),
    )


def _build_private_gold(
    public: contract.PublicInputBundle,
    left: ReviewerAnnotationBundle,
    right: ReviewerAnnotationBundle,
    adjudication: AdjudicationBundle,
) -> contract.PrivateGoldBundle:
    validate_annotation_binding(public, left)
    validate_annotation_binding(public, right)
    if left.reviewer_id == right.reviewer_id:
        raise AnnotationWorkflowError("two_distinct_reviewers_required")
    if left.manual_version != right.manual_version:
        raise AnnotationWorkflowError("reviewers_must_use_same_manual_version")
    public_digest = contract.canonical_sha256(public)
    if (
        adjudication.dataset_id != public.dataset_id
        or adjudication.public_input_sha256 != public_digest
    ):
        raise AnnotationWorkflowError("adjudication_public_input_binding_mismatch")
    annotations = (left, right)
    annotation_digests = tuple(contract.canonical_sha256(value) for value in annotations)
    if adjudication.reviewer_ids != tuple(value.reviewer_id for value in annotations):
        raise AnnotationWorkflowError("adjudication_reviewer_order_mismatch")
    if adjudication.annotation_sha256 != annotation_digests:
        raise AnnotationWorkflowError("adjudication_annotation_binding_mismatch")
    public_cases = {row.case_id: row for row in public.cases}
    resolutions = {row.case_id: row for row in adjudication.cases}
    if set(resolutions) != set(public_cases):
        raise AnnotationWorkflowError("adjudication_must_cover_every_public_case_once")
    reviewer_cases = (
        {row.case_id: row for row in left.cases},
        {row.case_id: row for row in right.cases},
    )
    gold_cases: list[contract.PrivateGoldCase] = []
    for case in public.cases:
        resolution = resolutions[case.case_id]
        if (
            resolution.adjudicator_id is not None
            and resolution.adjudicator_id in adjudication.reviewer_ids
        ):
            raise AnnotationWorkflowError("third_human_must_be_distinct_from_reviewers")
        known = {row.evidence_id for row in case.evidence_catalog}
        required_ids = {row.evidence_id for row in resolution.required_evidence}
        referenced = required_ids | set(resolution.forbidden_evidence_ids)
        referenced.update(
            evidence_id
            for group in resolution.independent_event_groups
            for evidence_id in group.evidence_ids
        )
        referenced.update(
            link.support_evidence_id for link in resolution.explanation_links
        )
        referenced.update(
            evidence_id
            for link in resolution.explanation_links
            for evidence_id in link.applicable_current_ids
        )
        if not referenced <= known:
            raise AnnotationWorkflowError("adjudication_evidence_id_not_in_public_catalog")
        roles: dict[str, tuple[str, ...]] = {
            role: tuple(
                row.evidence_id
                for row in resolution.required_evidence
                if row.role == role
            )
            for role in ("B", "C", "G", "X", "P")
        }
        first = reviewer_cases[0][case.case_id]
        second = reviewer_cases[1][case.case_id]
        gold_cases.append(
            contract.PrivateGoldCase(
                case_id=case.case_id,
                outcome=resolution.final_outcome,
                surface=resolution.final_surface,
                required_evidence=contract.RequiredEvidenceGroups(**roles),
                forbidden_evidence_ids=resolution.forbidden_evidence_ids,
                independent_event_groups=resolution.independent_event_groups,
                phenomena=resolution.phenomena,
                adjudication=contract.DualAdjudication(
                    adjudicated=True,
                    reviewers=(
                        contract.ReviewerJudgement(
                            reviewer_id=left.reviewer_id,
                            outcome=first.outcome,
                            surface=first.surface,
                        ),
                        contract.ReviewerJudgement(
                            reviewer_id=right.reviewer_id,
                            outcome=second.outcome,
                            surface=second.surface,
                        ),
                    ),
                    final_outcome=resolution.final_outcome,
                    final_surface=resolution.final_surface,
                ),
            )
        )
    gold = contract.PrivateGoldBundle(
        dataset_id=public.dataset_id,
        public_input_sha256=public_digest,
        cases=tuple(gold_cases),
    )
    contract.validate_evaluation_contract(public, gold)
    return gold


def _closed_commitment_metadata_bytes(
    *,
    public_input_sha256: str,
    author_setup_sha256: str,
    execution_freeze_sha256: str,
    run_config_sha256: str,
    annotation_sha256: tuple[str, str],
    adjudication_sha256: str,
    manual_version: str,
    gate_policy_id: str,
    gate_policy: contract.GatePolicy,
    scorer_manifest: ScorerSourceManifest,
) -> bytes:
    return json.dumps(
        {
            "schema_version": CLOSED_COMMITMENT_SCHEMA_VERSION,
            "scheme": "hmac-sha256-v1",
            "public_input_sha256": public_input_sha256,
            "author_setup_sha256": author_setup_sha256,
            "execution_freeze_sha256": execution_freeze_sha256,
            "run_config_sha256": run_config_sha256,
            "annotation_sha256": annotation_sha256,
            "adjudication_sha256": adjudication_sha256,
            "manual_version": manual_version,
            "gate_policy_id": gate_policy_id,
            "gate_policy": gate_policy.model_dump(mode="json"),
            "scorer_manifest": scorer_manifest.model_dump(mode="json"),
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _source_sha256(path: Path) -> str:
    """Hash a scoring source file without importing any additional code."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_scorer_manifest() -> ScorerSourceManifest:
    """Rebuild the canonical identity of the scorer and its score contract."""

    contract_path = Path(contract.__file__).resolve()
    return ScorerSourceManifest(
        scorer_source_sha256=_source_sha256(Path(__file__).resolve()),
        eval_contract_source_sha256=_source_sha256(contract_path),
    )


def _create_closed_gold_commitment(
    gold: contract.PrivateGoldBundle,
    left: ReviewerAnnotationBundle,
    right: ReviewerAnnotationBundle,
    adjudication: AdjudicationBundle,
    *,
    author_setup_sha256: str,
    execution_freeze_sha256: str,
    run_config_sha256: str,
    hmac_key: bytes,
) -> ClosedGoldCommitment:
    # Reuse the base contract's entropy validation rather than silently
    # accepting a password-like or public low-entropy key.
    contract.create_gold_commitment(gold, hmac_key=hmac_key)
    annotation_sha256 = (
        contract.canonical_sha256(left),
        contract.canonical_sha256(right),
    )
    adjudication_sha256 = contract.canonical_sha256(adjudication)
    gate_policy = FORMAL_GATE_POLICY_REGISTRY[SINGLE_RUN_FORMAL_GATE_POLICY_ID]
    scorer_manifest = _current_scorer_manifest()
    metadata = _closed_commitment_metadata_bytes(
        public_input_sha256=gold.public_input_sha256,
        author_setup_sha256=author_setup_sha256,
        execution_freeze_sha256=execution_freeze_sha256,
        run_config_sha256=run_config_sha256,
        annotation_sha256=annotation_sha256,
        adjudication_sha256=adjudication_sha256,
        manual_version=left.manual_version,
        gate_policy_id=SINGLE_RUN_FORMAL_GATE_POLICY_ID,
        gate_policy=gate_policy,
        scorer_manifest=scorer_manifest,
    )
    payload = (
        CLOSED_COMMITMENT_DOMAIN
        + contract.canonical_json_bytes(gold)
        + b"\x00"
        + metadata
    )
    return ClosedGoldCommitment(
        scheme="hmac-sha256-v1",
        public_input_sha256=gold.public_input_sha256,
        author_setup_sha256=author_setup_sha256,
        execution_freeze_sha256=execution_freeze_sha256,
        run_config_sha256=run_config_sha256,
        annotation_sha256=annotation_sha256,
        adjudication_sha256=adjudication_sha256,
        manual_version=left.manual_version,
        gate_policy_id=SINGLE_RUN_FORMAL_GATE_POLICY_ID,
        gate_policy=gate_policy,
        scorer_manifest=scorer_manifest,
        digest=hmac.new(hmac_key, payload, hashlib.sha256).hexdigest(),
    )


def _verify_closed_gold_commitment(
    gold: contract.PrivateGoldBundle,
    commitment: ClosedGoldCommitment,
    *,
    hmac_key: bytes,
) -> bool:
    # This call performs the same high-entropy key validation used by the
    # base contract before the workflow-specific envelope is checked.
    contract.create_gold_commitment(gold, hmac_key=hmac_key)
    if commitment.public_input_sha256 != gold.public_input_sha256:
        return False
    metadata = _closed_commitment_metadata_bytes(
        public_input_sha256=commitment.public_input_sha256,
        author_setup_sha256=commitment.author_setup_sha256,
        execution_freeze_sha256=commitment.execution_freeze_sha256,
        run_config_sha256=commitment.run_config_sha256,
        annotation_sha256=commitment.annotation_sha256,
        adjudication_sha256=commitment.adjudication_sha256,
        manual_version=commitment.manual_version,
        gate_policy_id=commitment.gate_policy_id,
        gate_policy=commitment.gate_policy,
        scorer_manifest=commitment.scorer_manifest,
    )
    payload = (
        CLOSED_COMMITMENT_DOMAIN
        + contract.canonical_json_bytes(gold)
        + b"\x00"
        + metadata
    )
    expected = hmac.new(hmac_key, payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, commitment.digest)


def _require_sealed_scoring_configuration(
    commitment: ClosedGoldCommitment,
) -> None:
    """Reject unregistered policy choices and scoring-source drift."""

    registered_policy = FORMAL_GATE_POLICY_REGISTRY.get(commitment.gate_policy_id)
    if registered_policy is None or commitment.gate_policy != registered_policy:
        raise AnnotationWorkflowError("sealed_gate_policy_mismatch")
    if commitment.scorer_manifest != _current_scorer_manifest():
        raise AnnotationWorkflowError("sealed_scorer_manifest_mismatch")


def _score_sealed_evaluation(
    public: contract.PublicInputBundle,
    gold: contract.PrivateGoldBundle,
    commitment: ClosedGoldCommitment,
    predictions: Sequence[contract.PredictionBundle],
    *,
    author_setup_sha256: str,
    execution_freeze_sha256: str,
    run_config_sha256: str,
    hmac_key: bytes,
) -> contract.EvaluationReport:
    """Verify the frozen gold, policy and scorer before scoring is allowed."""

    public_input_sha256 = contract.canonical_sha256(public)
    _require_closed_commitment_for_execution(
        gold,
        commitment,
        public_input_sha256=public_input_sha256,
        author_setup_sha256=author_setup_sha256,
        execution_freeze_sha256=execution_freeze_sha256,
        run_config_sha256=run_config_sha256,
        hmac_key=hmac_key,
    )
    return contract.score_evaluation(
        public,
        gold,
        predictions,
        gate_policy=commitment.gate_policy,
    )


def _require_closed_commitment_for_execution(
    gold: contract.PrivateGoldBundle,
    commitment: ClosedGoldCommitment,
    *,
    public_input_sha256: str,
    author_setup_sha256: str,
    execution_freeze_sha256: str,
    run_config_sha256: str,
    hmac_key: bytes,
) -> None:
    """Fail before prediction loading when any frozen execution input differs."""

    if (
        gold.public_input_sha256 != public_input_sha256
        or commitment.public_input_sha256 != public_input_sha256
        or commitment.author_setup_sha256 != author_setup_sha256
        or commitment.execution_freeze_sha256 != execution_freeze_sha256
        or commitment.run_config_sha256 != run_config_sha256
        or not _verify_closed_gold_commitment(gold, commitment, hmac_key=hmac_key)
    ):
        raise AnnotationWorkflowError("private_gold_commitment_mismatch")
    _require_sealed_scoring_configuration(commitment)


def _extract_bound_prediction(
    public: contract.PublicInputBundle,
    artifact: Any,
    *,
    author_setup_sha256: str,
    execution_freeze_sha256: str,
) -> contract.PredictionBundle:
    """Return the inner prediction only after checking the sealed envelope."""

    public_digest = contract.canonical_sha256(public)
    if artifact.dataset_id != public.dataset_id:
        raise AnnotationWorkflowError("sealed_prediction_dataset_mismatch")
    if artifact.public_input_sha256 != public_digest:
        raise AnnotationWorkflowError("sealed_prediction_public_hash_mismatch")
    if artifact.author_setup_sha256 != author_setup_sha256:
        raise AnnotationWorkflowError("sealed_prediction_author_setup_hash_mismatch")
    if artifact.execution_freeze_sha256 != execution_freeze_sha256:
        raise AnnotationWorkflowError("sealed_prediction_execution_freeze_hash_mismatch")
    if (
        artifact.prediction.dataset_id != public.dataset_id
        or artifact.prediction.public_input_sha256 != public_digest
    ):
        raise AnnotationWorkflowError("sealed_prediction_inner_binding_mismatch")
    return artifact.prediction


def _write_json(path: Path, value: Any) -> None:
    text_path = str(path)
    if "://" in text_path or text_path.startswith(("\\\\", "//")):
        raise AnnotationWorkflowError("network_locations_are_forbidden")
    if _path_chain_has_reparse_point(path.parent):
        raise AnnotationWorkflowError("reparse_point_paths_are_forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        payload = value.model_dump(mode="json")
    else:
        payload = value
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError as exc:
        raise AnnotationWorkflowError("output_file_already_exists") from exc
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _read_limited_local(path: Path, *, limit: int) -> bytes:
    text_path = str(path)
    if "://" in text_path or text_path.startswith(("\\\\", "//")):
        raise AnnotationWorkflowError("network_locations_are_forbidden")
    if _path_chain_has_reparse_point(path):
        raise AnnotationWorkflowError("reparse_point_paths_are_forbidden")
    try:
        size = path.stat().st_size
        if size > limit:
            raise AnnotationWorkflowError("input_file_too_large")
        payload = path.read_bytes()
    except OSError as exc:
        raise AnnotationWorkflowError("input_file_unreadable") from exc
    if len(payload) > limit:
        raise AnnotationWorkflowError("input_file_too_large")
    return payload


def _load_annotation(path: Path) -> ReviewerAnnotationBundle:
    return ReviewerAnnotationBundle.model_validate_json(
        _read_limited_local(path, limit=contract.MAX_JSON_BYTES)
    )


def _load_adjudication(path: Path) -> AdjudicationBundle:
    return AdjudicationBundle.model_validate_json(
        _read_limited_local(path, limit=contract.MAX_JSON_BYTES)
    )


def _load_and_validate_execution_binding(
    public: contract.PublicInputBundle,
    *,
    author_setup_path: str | Path,
    execution_freeze_path: str | Path,
) -> tuple[Any, Any]:
    # Local imports avoid a module cycle: the bundle builder reuses this
    # module's public-document verifier.
    from scripts import ooc_closed_bundle as closed_bundle

    setup = closed_bundle.PublicAuthorSetupBundle.model_validate_json(
        _read_limited_local(
            Path(author_setup_path), limit=contract.MAX_JSON_BYTES
        )
    )
    freeze = closed_bundle.ExecutionFreeze.model_validate_json(
        _read_limited_local(
            Path(execution_freeze_path), limit=contract.MAX_JSON_BYTES
        )
    )
    closed_bundle.validate_execution_freeze_binding(public, setup, freeze)
    return setup, freeze


def _load_sealed_prediction_artifact(path: str | Path) -> Any:
    from scripts.ooc_sealed_prediction import SealedPredictionArtifact

    return SealedPredictionArtifact.model_validate_json(
        _read_limited_local(Path(path), limit=contract.MAX_JSON_BYTES)
    )


def _load_and_validate_run_config(path: str | Path) -> Any:
    from scripts import run_ooc_sealed_http as sealed_runner

    config = sealed_runner.SealedHttpRunConfig.model_validate_json(
        _read_limited_local(Path(path), limit=contract.MAX_JSON_BYTES)
    )
    # The historical field name remains wire-compatible.  Its value is the
    # runner's canonical execution-code manifest digest (runner + sealed
    # prediction + evaluation contract), not merely one file's byte hash.
    if (
        config.expected_runner_source_sha256
        != sealed_runner.current_runner_source_sha256()
    ):
        raise AnnotationWorkflowError("run_config_runner_source_hash_mismatch")
    return config


def _load_sealed_http_run_report(path: str | Path) -> Any:
    from scripts.run_ooc_sealed_http import SealedHttpRunReport

    return SealedHttpRunReport.model_validate_json(
        _read_limited_local(Path(path), limit=contract.MAX_JSON_BYTES)
    )


def _extract_prediction_from_run_report(
    public: contract.PublicInputBundle,
    report: Any,
    run_config: Any,
    *,
    author_setup_sha256: str,
    execution_freeze_sha256: str,
) -> contract.PredictionBundle:
    if report.phase not in {"completed", "partial"}:
        raise AnnotationWorkflowError("sealed_run_report_not_scorable")
    if report.run_config != run_config:
        raise AnnotationWorkflowError("sealed_run_report_config_mismatch")
    expected_config_sha256 = contract.canonical_sha256(run_config)
    if report.run_config_sha256 != expected_config_sha256:
        raise AnnotationWorkflowError("sealed_run_report_config_hash_mismatch")
    if report.runtime_provenance_sha256 != run_config.expected_runtime_provenance_sha256:
        raise AnnotationWorkflowError("sealed_run_report_runtime_mismatch")
    if report.runner_source_sha256 != run_config.expected_runner_source_sha256:
        raise AnnotationWorkflowError("sealed_run_report_runner_mismatch")
    if report.execution_id != run_config.execution_id:
        raise AnnotationWorkflowError("sealed_run_report_execution_mismatch")
    if report.prediction_artifact is None:
        raise AnnotationWorkflowError("sealed_run_report_prediction_missing")
    return _extract_bound_prediction(
        public,
        report.prediction_artifact,
        author_setup_sha256=author_setup_sha256,
        execution_freeze_sha256=execution_freeze_sha256,
    )


def _load_hmac_key(path: Path) -> bytes:
    payload = _read_limited_local(path, limit=4_096)
    if payload.endswith(b"\r\n"):
        payload = payload[:-2]
    elif payload.endswith(b"\n"):
        payload = payload[:-1]
    return payload


def _require_distinct_local_paths(paths: Sequence[str | Path]) -> None:
    resolved: list[Path] = []
    for value in paths:
        text_path = str(value)
        if "://" in text_path or text_path.startswith(("\\\\", "//")):
            raise AnnotationWorkflowError("network_locations_are_forbidden")
        if _path_chain_has_reparse_point(Path(value)):
            raise AnnotationWorkflowError("reparse_point_paths_are_forbidden")
        resolved.append(Path(value).resolve(strict=False))
    if len(resolved) != len(set(resolved)):
        raise AnnotationWorkflowError("input_and_output_files_must_be_distinct")


def _command_validate(args: argparse.Namespace) -> int:
    _require_distinct_local_paths(
        (
            args.public,
            args.author_setup,
            args.execution_freeze,
            args.bundle_root,
            args.annotation,
        )
    )
    public = contract.load_public_input_file(args.public)
    setup, _ = _load_and_validate_execution_binding(
        public,
        author_setup_path=args.author_setup,
        execution_freeze_path=args.execution_freeze,
    )
    validate_exhaustive_line_catalog(public, args.bundle_root)
    annotation = _load_annotation(args.annotation)
    _validate_annotation_setup_binding(public, setup, annotation)
    print(json.dumps({"valid": True, "case_count": len(annotation.cases)}))
    return 0


def _command_verify_public(args: argparse.Namespace) -> int:
    _require_distinct_local_paths((args.public, args.bundle_root))
    public = contract.load_public_input_file(args.public)
    validate_exhaustive_line_catalog(public, args.bundle_root)
    print(
        json.dumps(
            {
                "valid": True,
                "case_count": len(public.cases),
                "public_input_sha256": contract.canonical_sha256(public),
            }
        )
    )
    return 0


def _command_compare(args: argparse.Namespace) -> int:
    _require_distinct_local_paths(
        (
            args.public,
            args.author_setup,
            args.execution_freeze,
            args.bundle_root,
            args.left,
            args.right,
            args.output,
        )
    )
    public = contract.load_public_input_file(args.public)
    setup, _ = _load_and_validate_execution_binding(
        public,
        author_setup_path=args.author_setup,
        execution_freeze_path=args.execution_freeze,
    )
    validate_exhaustive_line_catalog(public, args.bundle_root)
    left = _load_annotation(args.left)
    right = _load_annotation(args.right)
    _validate_annotation_setup_binding(public, setup, left)
    _validate_annotation_setup_binding(public, setup, right)
    report = compare_annotations(public, left, right)
    _write_json(Path(args.output), report)
    print(
        json.dumps(
            {
                "case_count": report.case_count,
                "disagreement_count": len(report.disagreement_case_ids),
                "exact_label_agreement": report.exact_label_agreement,
                "outcome_kappa": report.outcome_kappa,
                "surface_kappa": report.surface_kappa,
                "mean_evidence_jaccard": report.mean_evidence_jaccard,
            },
            ensure_ascii=False,
        )
    )
    return 0


def _command_build_gold(args: argparse.Namespace) -> int:
    _require_distinct_local_paths(
        (
            args.public,
            args.author_setup,
            args.execution_freeze,
            args.run_config,
            args.bundle_root,
            args.left,
            args.right,
            args.adjudication,
            args.hmac_key_file,
            args.gold_output,
            args.commitment_output,
        )
    )
    public = contract.load_public_input_file(args.public)
    setup, freeze = _load_and_validate_execution_binding(
        public,
        author_setup_path=args.author_setup,
        execution_freeze_path=args.execution_freeze,
    )
    run_config = _load_and_validate_run_config(args.run_config)
    validate_exhaustive_line_catalog(public, args.bundle_root)
    left = _load_annotation(args.left)
    right = _load_annotation(args.right)
    adjudication = _load_adjudication(args.adjudication)
    _validate_annotation_setup_binding(public, setup, left)
    _validate_annotation_setup_binding(public, setup, right)
    _validate_adjudication_setup_binding(public, setup, adjudication)
    gold = _build_private_gold(public, left, right, adjudication)
    commitment = _create_closed_gold_commitment(
        gold,
        left,
        right,
        adjudication,
        author_setup_sha256=contract.canonical_sha256(setup),
        execution_freeze_sha256=contract.canonical_sha256(freeze),
        run_config_sha256=contract.canonical_sha256(run_config),
        hmac_key=_load_hmac_key(Path(args.hmac_key_file)),
    )
    _write_json(Path(args.gold_output), gold)
    try:
        _write_json(Path(args.commitment_output), commitment)
    except Exception:
        Path(args.gold_output).unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "case_count": len(gold.cases),
                "gold_written": True,
                "commitment_written": True,
            }
        )
    )
    return 0


def _command_score(args: argparse.Namespace) -> int:
    _require_distinct_local_paths(
        (
            args.public,
            args.author_setup,
            args.execution_freeze,
            args.run_config,
            args.bundle_root,
            args.gold,
            args.commitment,
            args.hmac_key_file,
            args.sealed_run_report,
            args.output,
        )
    )
    public = contract.load_public_input_file(args.public)
    setup, freeze = _load_and_validate_execution_binding(
        public,
        author_setup_path=args.author_setup,
        execution_freeze_path=args.execution_freeze,
    )
    run_config = _load_and_validate_run_config(args.run_config)
    validate_exhaustive_line_catalog(public, args.bundle_root)
    gold = contract.load_private_gold_file(args.gold)
    commitment = ClosedGoldCommitment.model_validate_json(
        _read_limited_local(Path(args.commitment), limit=contract.MAX_JSON_BYTES)
    )
    hmac_key = _load_hmac_key(Path(args.hmac_key_file))
    author_setup_sha256 = contract.canonical_sha256(setup)
    execution_freeze_sha256 = contract.canonical_sha256(freeze)
    run_config_sha256 = contract.canonical_sha256(run_config)
    public_input_sha256 = contract.canonical_sha256(public)
    # Keep this check before opening any model-produced prediction artifact.
    _require_closed_commitment_for_execution(
        gold,
        commitment,
        public_input_sha256=public_input_sha256,
        author_setup_sha256=author_setup_sha256,
        execution_freeze_sha256=execution_freeze_sha256,
        run_config_sha256=run_config_sha256,
        hmac_key=hmac_key,
    )
    report = _load_sealed_http_run_report(args.sealed_run_report)
    prediction = _extract_prediction_from_run_report(
        public,
        report,
        run_config,
        author_setup_sha256=author_setup_sha256,
        execution_freeze_sha256=execution_freeze_sha256,
    )
    report = _score_sealed_evaluation(
        public,
        gold,
        commitment,
        (prediction,),
        author_setup_sha256=author_setup_sha256,
        execution_freeze_sha256=execution_freeze_sha256,
        run_config_sha256=run_config_sha256,
        hmac_key=hmac_key,
    )
    _write_json(Path(args.output), report)
    print(
        json.dumps(
            {
                "case_count": report.metrics.case_count,
                "repeat_count": report.metrics.repeat_count,
                "gate_passed": report.gate.passed,
                "failure_count": len(report.gate.failures),
            }
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and seal two-person OOC holdout annotations."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    verify = commands.add_parser("verify-public")
    verify.add_argument("--public", required=True)
    verify.add_argument("--bundle-root", required=True)
    verify.set_defaults(handler=_command_verify_public)

    validate = commands.add_parser("validate-annotation")
    validate.add_argument("--public", required=True)
    validate.add_argument("--author-setup", required=True)
    validate.add_argument("--execution-freeze", required=True)
    validate.add_argument("--bundle-root", required=True)
    validate.add_argument("--annotation", required=True)
    validate.set_defaults(handler=_command_validate)

    compare = commands.add_parser("compare")
    compare.add_argument("--public", required=True)
    compare.add_argument("--author-setup", required=True)
    compare.add_argument("--execution-freeze", required=True)
    compare.add_argument("--bundle-root", required=True)
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(handler=_command_compare)

    build = commands.add_parser("build-gold")
    build.add_argument("--public", required=True)
    build.add_argument("--author-setup", required=True)
    build.add_argument("--execution-freeze", required=True)
    build.add_argument("--run-config", required=True)
    build.add_argument("--bundle-root", required=True)
    build.add_argument("--left", required=True)
    build.add_argument("--right", required=True)
    build.add_argument("--adjudication", required=True)
    build.add_argument("--hmac-key-file", required=True)
    build.add_argument("--gold-output", required=True)
    build.add_argument("--commitment-output", required=True)
    build.set_defaults(handler=_command_build_gold)

    score = commands.add_parser("score")
    score.add_argument("--public", required=True)
    score.add_argument("--author-setup", required=True)
    score.add_argument("--execution-freeze", required=True)
    score.add_argument("--run-config", required=True)
    score.add_argument("--bundle-root", required=True)
    score.add_argument("--gold", required=True)
    score.add_argument("--commitment", required=True)
    score.add_argument("--hmac-key-file", required=True)
    score.add_argument("--sealed-run-report", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(handler=_command_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ADJUDICATION_SCHEMA_VERSION",
    "AGREEMENT_SCHEMA_VERSION",
    "ANNOTATION_SCHEMA_VERSION",
    "CLOSED_COMMITMENT_SCHEMA_VERSION",
    "SCORER_MANIFEST_SCHEMA_VERSION",
    "SINGLE_RUN_FORMAL_GATE_POLICY",
    "SINGLE_RUN_FORMAL_GATE_POLICY_ID",
    "AdjudicatedCase",
    "AdjudicationBundle",
    "AnnotationAgreementReport",
    "AnnotationWorkflowError",
    "ClosedGoldCommitment",
    "EvidenceAssignment",
    "ExplanationLink",
    "ReviewerAnnotationBundle",
    "ReviewerCaseAnnotation",
    "ScorerSourceManifest",
    "compare_annotations",
    "validate_annotation_binding",
    "validate_exhaustive_line_catalog",
]
