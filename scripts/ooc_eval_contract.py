"""Offline-only contracts and scoring for a frozen OOC evaluation.

This module deliberately contains no provider, HTTP, database, or application
imports.  The public input manifest, private gold file, and prediction files
are separate JSON envelopes.  Document paths are metadata: none of the loader
or scoring helpers opens a referenced document.

The scorer is deliberately pessimistic.  A missing case, a ``partial`` or
``failed`` run, or an ``unassessed`` outcome can never receive correctness
credit.  Consequently, incomplete runs remain false negatives in recall and
in the denominators of coverage, specificity, and joint accuracy.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


PUBLIC_SCHEMA_VERSION = "loreguard-ooc-public-v1"
PRIVATE_GOLD_SCHEMA_VERSION = "loreguard-ooc-private-gold-v1"
PREDICTION_SCHEMA_VERSION = "loreguard-ooc-prediction-v1"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_CASES = 10_000
MAX_DOCUMENTS_PER_CASE = 128
MAX_EVIDENCE_PER_CASE = 1_024
MAX_CITATIONS_PER_CASE = 1_024
MAX_COUNTER = 10_000_000

# IDs are intentionally opaque.  In particular, there is no UUID, prefix,
# ASCII, or slug assumption: only a non-empty length bound is imposed.
OpaqueId = Annotated[str, Field(strict=True, min_length=1, max_length=200)]
OpaqueTag = Annotated[str, Field(strict=True, min_length=1, max_length=100)]
Sha256 = Annotated[
    str,
    Field(strict=True, pattern=r"^[0-9a-f]{64}$"),
]
SnapshotVersion = Annotated[int, Field(strict=True, ge=1, le=2_147_483_647)]
BoundedCount = Annotated[int, Field(strict=True, ge=0, le=MAX_COUNTER)]
RateThreshold = Annotated[float, Field(strict=True, ge=0.0, le=1.0)]

GoldOutcome = Literal["conflict", "no_issue", "indeterminate"]
PredictionOutcome = Literal[
    "conflict", "no_issue", "indeterminate", "unassessed"
]
Surface = Literal["formal_issue", "review_clue", "none"]
EvidenceRole = Literal["B", "C", "G", "X", "P"]
RunStatus = Literal["complete", "partial", "failed"]


class ContractError(ValueError):
    """Raised when otherwise valid envelopes do not bind to one another."""


class CommitmentError(ValueError):
    """Raised when a gold commitment would be insecure or cannot be checked."""


class StrictModel(BaseModel):
    """Shared fail-closed settings for every persisted contract model."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class DocumentSnapshot(StrictModel):
    document_id: OpaqueId
    path: str = Field(strict=True, min_length=1, max_length=500)
    version: SnapshotVersion
    sha256: Sha256

    @field_validator("path")
    @classmethod
    def validate_portable_relative_path(cls, value: str) -> str:
        # Inputs may be relocated as a frozen bundle, so paths must not escape
        # the bundle root.  This validates only the string and never reads it.
        if "\\" in value:
            raise ValueError("document_path_must_use_forward_slashes")
        if (
            "://" in value
            or value.startswith("//")
            or (len(value) >= 2 and value[0].isalpha() and value[1] == ":")
            or any(":" in part for part in value.split("/"))
        ):
            raise ValueError("document_path_must_be_portable_and_local")
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or str(parsed) != value
        ):
            raise ValueError("document_path_must_be_normalized_and_relative")
        return value


class AxisSnapshot(StrictModel):
    axis_id: OpaqueId
    version: SnapshotVersion
    sha256: Sha256


class EvidenceAnchor(StrictModel):
    evidence_id: OpaqueId
    document_id: OpaqueId
    line_start: int = Field(strict=True, ge=1, le=10_000_000)
    line_end: int = Field(strict=True, ge=1, le=10_000_000)

    @model_validator(mode="after")
    def validate_range(self) -> EvidenceAnchor:
        if self.line_end < self.line_start:
            raise ValueError("evidence_line_range_invalid")
        return self


class PublicInputCase(StrictModel):
    """One model-visible case.  It contains no expected labels or roles."""

    case_id: OpaqueId
    split_id: OpaqueId
    group_id: OpaqueId
    world_id: OpaqueId
    documents: tuple[DocumentSnapshot, ...] = Field(
        min_length=1, max_length=MAX_DOCUMENTS_PER_CASE
    )
    axis: AxisSnapshot
    evidence_catalog: tuple[EvidenceAnchor, ...] = Field(
        min_length=1, max_length=MAX_EVIDENCE_PER_CASE
    )

    @model_validator(mode="after")
    def validate_local_references(self) -> PublicInputCase:
        document_ids = tuple(row.document_id for row in self.documents)
        document_paths = tuple(row.path for row in self.documents)
        document_hashes = tuple(row.sha256 for row in self.documents)
        evidence_ids = tuple(row.evidence_id for row in self.evidence_catalog)
        evidence_anchors = tuple(
            (row.document_id, row.line_start, row.line_end)
            for row in self.evidence_catalog
        )
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("duplicate_document_id")
        if len(document_paths) != len(set(document_paths)):
            raise ValueError("duplicate_document_path")
        if len(document_hashes) != len(set(document_hashes)):
            raise ValueError("duplicate_document_bytes")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("duplicate_evidence_id")
        if len(evidence_anchors) != len(set(evidence_anchors)):
            raise ValueError("duplicate_evidence_anchor")
        known_documents = set(document_ids)
        if any(row.document_id not in known_documents for row in self.evidence_catalog):
            raise ValueError("evidence_document_unknown")
        return self


def _leakage_tokens(case: PublicInputCase) -> tuple[tuple[str, str], ...]:
    """Return identities that force cases into one leakage group.

    Reuse is allowed inside a group.  Reusing a world, document identity,
    document bytes, path, axis identity/snapshot, or evidence identity across
    groups would make those groups non-independent and is rejected.
    """

    tokens: list[tuple[str, str]] = [
        ("world_id", case.world_id),
        ("axis_id", case.axis.axis_id),
        ("axis_bytes", case.axis.sha256),
        ("axis_snapshot", f"{case.axis.version}:{case.axis.sha256}"),
    ]
    for document in case.documents:
        tokens.extend(
            (
                ("document_id", document.document_id),
                ("document_path", document.path),
                ("document_bytes", document.sha256),
                ("document_snapshot", f"{document.version}:{document.sha256}"),
            )
        )
    tokens.extend(
        ("evidence_id", evidence.evidence_id)
        for evidence in case.evidence_catalog
    )
    return tuple(tokens)


def validate_no_cross_split_group_leakage(
    cases: Sequence[PublicInputCase],
) -> None:
    """Reject correlated identities assigned to different groups or splits."""

    group_splits: dict[str, str] = {}
    token_owners: dict[tuple[str, str], tuple[str, str]] = {}
    case_ids: set[str] = set()
    for case in cases:
        if case.case_id in case_ids:
            raise ContractError("duplicate_case_id")
        case_ids.add(case.case_id)
        prior_split = group_splits.setdefault(case.group_id, case.split_id)
        if prior_split != case.split_id:
            raise ContractError("cross_split_group_leakage")
        owner = (case.split_id, case.group_id)
        for token in _leakage_tokens(case):
            prior_owner = token_owners.setdefault(token, owner)
            if prior_owner != owner:
                raise ContractError("cross_split_or_group_identity_leakage")


class PublicInputBundle(StrictModel):
    schema_version: Literal["loreguard-ooc-public-v1"] = PUBLIC_SCHEMA_VERSION
    dataset_id: OpaqueId
    cases: tuple[PublicInputCase, ...] = Field(
        min_length=1, max_length=MAX_CASES
    )

    @model_validator(mode="after")
    def validate_isolation(self) -> PublicInputBundle:
        validate_no_cross_split_group_leakage(self.cases)
        return self


class RequiredEvidenceGroups(StrictModel):
    """Gold-only role assignments; every key is present, even when empty."""

    B: tuple[OpaqueId, ...] = Field(default=(), max_length=MAX_EVIDENCE_PER_CASE)
    C: tuple[OpaqueId, ...] = Field(default=(), max_length=MAX_EVIDENCE_PER_CASE)
    G: tuple[OpaqueId, ...] = Field(default=(), max_length=MAX_EVIDENCE_PER_CASE)
    X: tuple[OpaqueId, ...] = Field(default=(), max_length=MAX_EVIDENCE_PER_CASE)
    P: tuple[OpaqueId, ...] = Field(default=(), max_length=MAX_EVIDENCE_PER_CASE)

    @model_validator(mode="after")
    def validate_unique_role_assignments(self) -> RequiredEvidenceGroups:
        all_ids = self.all_ids()
        if len(all_ids) != len(set(all_ids)):
            raise ValueError("required_evidence_id_reused_or_duplicated")
        return self

    def all_ids(self) -> tuple[str, ...]:
        return self.B + self.C + self.G + self.X + self.P

    def by_role(self) -> Mapping[EvidenceRole, tuple[str, ...]]:
        return {role: getattr(self, role) for role in ("B", "C", "G", "X", "P")}


class IndependentEventGroup(StrictModel):
    event_group_id: OpaqueId
    evidence_ids: tuple[OpaqueId, ...] = Field(
        min_length=1, max_length=MAX_EVIDENCE_PER_CASE
    )

    @model_validator(mode="after")
    def validate_unique_evidence(self) -> IndependentEventGroup:
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("duplicate_event_group_evidence_id")
        return self


class ReviewerJudgement(StrictModel):
    reviewer_id: OpaqueId
    outcome: GoldOutcome
    surface: Surface


class DualAdjudication(StrictModel):
    """Two independent labels plus their recorded adjudicated resolution."""

    adjudicated: Literal[True]
    reviewers: tuple[ReviewerJudgement, ReviewerJudgement]
    final_outcome: GoldOutcome
    final_surface: Surface

    @model_validator(mode="after")
    def validate_two_distinct_reviewers(self) -> DualAdjudication:
        if self.reviewers[0].reviewer_id == self.reviewers[1].reviewer_id:
            raise ValueError("adjudication_requires_two_distinct_reviewers")
        return self


class PrivateGoldCase(StrictModel):
    case_id: OpaqueId
    outcome: GoldOutcome
    surface: Surface
    required_evidence: RequiredEvidenceGroups
    forbidden_evidence_ids: tuple[OpaqueId, ...] = Field(
        default=(), max_length=MAX_EVIDENCE_PER_CASE
    )
    independent_event_groups: tuple[IndependentEventGroup, ...] = Field(
        default=(), max_length=MAX_EVIDENCE_PER_CASE
    )
    phenomena: tuple[OpaqueTag, ...] = Field(
        min_length=1, max_length=64
    )
    adjudication: DualAdjudication

    @model_validator(mode="after")
    def validate_gold_consistency(self) -> PrivateGoldCase:
        if (
            self.adjudication.final_outcome != self.outcome
            or self.adjudication.final_surface != self.surface
        ):
            raise ValueError("adjudicated_resolution_mismatch")
        if len(self.forbidden_evidence_ids) != len(set(self.forbidden_evidence_ids)):
            raise ValueError("duplicate_forbidden_evidence_id")
        if set(self.required_evidence.all_ids()) & set(self.forbidden_evidence_ids):
            raise ValueError("evidence_cannot_be_required_and_forbidden")
        if len(self.phenomena) != len(set(self.phenomena)):
            raise ValueError("duplicate_phenomenon")
        event_ids = tuple(row.event_group_id for row in self.independent_event_groups)
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("duplicate_independent_event_group_id")
        grouped_evidence = tuple(
            evidence_id
            for group in self.independent_event_groups
            for evidence_id in group.evidence_ids
        )
        if len(grouped_evidence) != len(set(grouped_evidence)):
            raise ValueError("evidence_cannot_span_independent_event_groups")
        if self.surface == "formal_issue" and (
            not self.required_evidence.B or not self.required_evidence.C
        ):
            raise ValueError("formal_issue_requires_B_and_C_evidence")
        if (self.surface == "formal_issue") != (self.outcome == "conflict"):
            raise ValueError("formal_issue_must_match_conflict_outcome")
        if self.surface == "formal_issue":
            if self.required_evidence.P:
                raise ValueError("formal_issue_cannot_require_P_evidence")
            if len(self.required_evidence.C) < 2:
                raise ValueError("formal_issue_requires_two_current_evidence")
            group_by_evidence = {
                evidence_id: group.event_group_id
                for group in self.independent_event_groups
                for evidence_id in group.evidence_ids
            }
            current_groups = {
                group_by_evidence.get(evidence_id)
                for evidence_id in self.required_evidence.C
            }
            if None in current_groups or len(current_groups) < 2:
                raise ValueError("formal_issue_requires_two_independent_current_events")
        return self


class PrivateGoldBundle(StrictModel):
    schema_version: Literal["loreguard-ooc-private-gold-v1"] = (
        PRIVATE_GOLD_SCHEMA_VERSION
    )
    dataset_id: OpaqueId
    public_input_sha256: Sha256
    cases: tuple[PrivateGoldCase, ...] = Field(
        min_length=1, max_length=MAX_CASES
    )

    @model_validator(mode="after")
    def validate_unique_cases(self) -> PrivateGoldBundle:
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("duplicate_gold_case_id")
        return self


class Citation(StrictModel):
    evidence_id: OpaqueId
    role: EvidenceRole
    document_id: OpaqueId
    snapshot_version: SnapshotVersion
    snapshot_sha256: Sha256
    line_start: int = Field(strict=True, ge=1, le=10_000_000)
    line_end: int = Field(strict=True, ge=1, le=10_000_000)

    @model_validator(mode="after")
    def validate_range(self) -> Citation:
        if self.line_end < self.line_start:
            raise ValueError("citation_line_range_invalid")
        return self


class StageFunnel(StrictModel):
    """Bounded, monotonically narrowing per-case pipeline counters."""

    input_candidates: BoundedCount
    retrieved_candidates: BoundedCount
    grounded_candidates: BoundedCount
    adjudicated_candidates: BoundedCount
    surfaced_candidates: BoundedCount

    @model_validator(mode="after")
    def validate_monotonic_funnel(self) -> StageFunnel:
        values = (
            self.input_candidates,
            self.retrieved_candidates,
            self.grounded_candidates,
            self.adjudicated_candidates,
            self.surfaced_candidates,
        )
        if any(left < right for left, right in zip(values, values[1:])):
            raise ValueError("stage_funnel_must_be_monotonic")
        return self


class CasePrediction(StrictModel):
    case_id: OpaqueId
    outcome: PredictionOutcome
    surface: Surface
    citations: tuple[Citation, ...] = Field(
        default=(), max_length=MAX_CITATIONS_PER_CASE
    )
    stage_funnel: StageFunnel

    @model_validator(mode="after")
    def validate_assessment_shape(self) -> CasePrediction:
        identities = tuple(
            (row.role, row.evidence_id) for row in self.citations
        )
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate_prediction_citation")
        if self.outcome == "unassessed" and (
            self.surface != "none" or self.citations
        ):
            raise ValueError("unassessed_prediction_must_be_empty")
        if (self.surface == "formal_issue") != (self.outcome == "conflict"):
            raise ValueError("formal_prediction_must_match_conflict_outcome")
        if self.surface == "formal_issue":
            roles = [row.role for row in self.citations]
            current_ids = {
                row.evidence_id for row in self.citations if row.role == "C"
            }
            if "P" in roles:
                raise ValueError("formal_prediction_cannot_cite_P")
            if "B" not in roles or len(current_ids) < 2:
                raise ValueError("formal_prediction_requires_B_and_two_C")
        return self


class PredictionBundle(StrictModel):
    schema_version: Literal["loreguard-ooc-prediction-v1"] = (
        PREDICTION_SCHEMA_VERSION
    )
    dataset_id: OpaqueId
    public_input_sha256: Sha256
    run_id: OpaqueId
    repeat_id: OpaqueId
    run_status: RunStatus
    predictions: tuple[CasePrediction, ...] = Field(
        default=(), max_length=MAX_CASES
    )

    @model_validator(mode="after")
    def validate_run_shape(self) -> PredictionBundle:
        case_ids = tuple(row.case_id for row in self.predictions)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("duplicate_prediction_case_id")
        if self.run_status == "failed" and self.predictions:
            raise ValueError("failed_run_must_not_claim_predictions")
        return self


def canonical_json_bytes(model: BaseModel) -> bytes:
    """Canonical UTF-8 JSON used for bindings and commitments."""

    return json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(model: BaseModel) -> str:
    """A non-secret content binding, suitable for the *public* input file."""

    return hashlib.sha256(canonical_json_bytes(model)).hexdigest()


class GoldCommitment(StrictModel):
    scheme: Literal["hmac-sha256-v1"]
    digest: Sha256


def _require_high_entropy_material(value: bytes, *, name: str, minimum: int) -> None:
    if not isinstance(value, bytes):
        raise CommitmentError(f"{name}_must_be_bytes")
    # Length is the primary security requirement; the diversity check catches
    # common test/dev mistakes such as b"x" * 32 that meet it syntactically.
    if len(value) < minimum or len(value) > 4_096 or len(set(value)) < 8:
        raise CommitmentError(f"{name}_is_too_short_or_low_entropy")


def _coerce_gold(value: PrivateGoldBundle | bytes) -> PrivateGoldBundle:
    if isinstance(value, PrivateGoldBundle):
        return value
    if not isinstance(value, bytes):
        raise CommitmentError("gold_must_be_a_validated_model_or_json_bytes")
    if len(value) > MAX_JSON_BYTES:
        raise CommitmentError("gold_json_too_large")
    try:
        return PrivateGoldBundle.model_validate_json(value)
    except ValidationError as exc:
        raise CommitmentError("gold_json_invalid") from exc


def create_gold_commitment(
    gold: PrivateGoldBundle | bytes,
    *,
    hmac_key: bytes | None = None,
    salt: bytes | None = None,
) -> GoldCommitment:
    """Commit to canonical private gold without publishing a bare hash.

    A secret HMAC key is required. Publishing a salt beside the digest does
    not hide this contract's small outcome/surface label space from offline
    enumeration, so the legacy ``salt`` argument is rejected explicitly.
    """

    if salt is not None:
        raise CommitmentError("public_salt_cannot_hide_low_entropy_gold")
    if hmac_key is None:
        raise CommitmentError("hmac_key_required")
    payload = b"LoreGuard/OOC/private-gold/v1\x00" + canonical_json_bytes(
        _coerce_gold(gold)
    )
    _require_high_entropy_material(hmac_key, name="hmac_key", minimum=32)
    digest = hmac.new(hmac_key, payload, hashlib.sha256).hexdigest()
    return GoldCommitment(scheme="hmac-sha256-v1", digest=digest)


def verify_gold_commitment(
    gold: PrivateGoldBundle | bytes,
    commitment: GoldCommitment,
    *,
    hmac_key: bytes | None = None,
) -> bool:
    """Verify a commitment in constant time; no document paths are accessed."""

    if hmac_key is None:
        raise CommitmentError("hmac_key_required_for_verification")
    expected = create_gold_commitment(gold, hmac_key=hmac_key)
    return hmac.compare_digest(expected.digest, commitment.digest)


ModelT = TypeVar("ModelT", bound=BaseModel)


def _local_envelope_path(path: str | Path) -> Path:
    """Reject network-like names before any filesystem normalization occurs."""

    text_path = str(path)
    if "://" in text_path or text_path.startswith(("\\\\", "//")):
        raise ContractError("network_locations_are_forbidden")
    return Path(path)


def _load_model_file(path: str | Path, model_type: type[ModelT]) -> ModelT:
    local_path = _local_envelope_path(path)
    try:
        size = local_path.stat().st_size
    except OSError as exc:
        raise ContractError("evaluation_json_unreadable") from exc
    if size > MAX_JSON_BYTES:
        raise ContractError("evaluation_json_too_large")
    try:
        payload = local_path.read_bytes()
        if len(payload) > MAX_JSON_BYTES:
            raise ContractError("evaluation_json_too_large")
        return model_type.model_validate_json(payload)
    except OSError as exc:
        raise ContractError("evaluation_json_unreadable") from exc
    except ValidationError as exc:
        raise ContractError("evaluation_json_schema_invalid") from exc


def load_public_input_file(path: str | Path) -> PublicInputBundle:
    return _load_model_file(path, PublicInputBundle)


def load_private_gold_file(path: str | Path) -> PrivateGoldBundle:
    return _load_model_file(path, PrivateGoldBundle)


def load_prediction_file(path: str | Path) -> PredictionBundle:
    return _load_model_file(path, PredictionBundle)


# Short aliases are useful to callers without weakening the separate loaders.
load_public_inputs = load_public_input_file
load_private_gold = load_private_gold_file
load_predictions = load_prediction_file


def validate_evaluation_contract(
    public: PublicInputBundle,
    gold: PrivateGoldBundle,
    prediction_runs: Sequence[PredictionBundle] = (),
) -> None:
    """Validate cross-file bindings without opening referenced documents."""

    if public.dataset_id != gold.dataset_id:
        raise ContractError("dataset_id_mismatch")
    public_digest = canonical_sha256(public)
    if gold.public_input_sha256 != public_digest:
        raise ContractError("public_input_binding_mismatch")
    public_cases = {case.case_id: case for case in public.cases}
    gold_cases = {case.case_id: case for case in gold.cases}
    if set(public_cases) != set(gold_cases):
        raise ContractError("public_gold_case_set_mismatch")
    for case_id, gold_case in gold_cases.items():
        public_case = public_cases[case_id]
        evidence_by_id = {
            row.evidence_id: row for row in public_case.evidence_catalog
        }
        known_evidence = set(evidence_by_id)
        referenced = (
            set(gold_case.required_evidence.all_ids())
            | set(gold_case.forbidden_evidence_ids)
            | {
                evidence_id
                for group in gold_case.independent_event_groups
                for evidence_id in group.evidence_ids
            }
        )
        if not referenced <= known_evidence:
            raise ContractError("gold_evidence_id_not_in_public_catalog")
        required = set(gold_case.required_evidence.all_ids())
        if any(
            not set(group.evidence_ids) <= required
            for group in gold_case.independent_event_groups
        ):
            raise ContractError("event_group_evidence_must_be_required")
        if gold_case.surface == "formal_issue":
            current = tuple(
                evidence_by_id[evidence_id]
                for evidence_id in gold_case.required_evidence.C
            )
            for index, left in enumerate(current):
                for right in current[index + 1 :]:
                    if (
                        left.document_id == right.document_id
                        and max(left.line_start, right.line_start)
                        <= min(left.line_end, right.line_end)
                    ):
                        raise ContractError(
                            "formal_current_evidence_ranges_overlap"
                        )

    repeat_ids: set[str] = set()
    run_ids: set[str] = set()
    for run in prediction_runs:
        if run.dataset_id != public.dataset_id:
            raise ContractError("prediction_dataset_id_mismatch")
        if run.public_input_sha256 != public_digest:
            raise ContractError("prediction_public_input_binding_mismatch")
        if run.repeat_id in repeat_ids:
            raise ContractError("duplicate_repeat_id")
        if run.run_id in run_ids:
            raise ContractError("duplicate_run_id")
        repeat_ids.add(run.repeat_id)
        run_ids.add(run.run_id)
        if not {row.case_id for row in run.predictions} <= set(public_cases):
            raise ContractError("prediction_case_not_in_public_inputs")
        for prediction in run.predictions:
            if prediction.surface != "formal_issue":
                continue
            current = tuple(
                citation
                for citation in prediction.citations
                if citation.role == "C"
            )
            if len({citation.evidence_id for citation in current}) < 2:
                raise ContractError(
                    "formal_prediction_requires_two_nonoverlapping_current_evidence"
                )
            for index, left in enumerate(current):
                for right in current[index + 1 :]:
                    if (
                        left.document_id == right.document_id
                        and max(left.line_start, right.line_start)
                        <= min(left.line_end, right.line_end)
                    ):
                        raise ContractError(
                            "formal_prediction_current_evidence_ranges_overlap"
                        )


def load_evaluation_files(
    public_path: str | Path,
    gold_path: str | Path,
    prediction_paths: Sequence[str | Path] = (),
) -> tuple[PublicInputBundle, PrivateGoldBundle, tuple[PredictionBundle, ...]]:
    """Load separated files and verify their bindings.

    Only the explicitly supplied JSON envelope files are read.  Paths inside
    ``DocumentSnapshot`` remain inert metadata.
    """

    # Guard raw strings before ``resolve``: resolving a UNC path on Windows can
    # contact SMB and leak credentials even if the later loader rejects it.
    all_paths = tuple(
        _local_envelope_path(path)
        for path in (public_path, gold_path, *prediction_paths)
    )
    resolved = tuple(path.resolve() for path in all_paths)
    if len(resolved) != len(set(resolved)):
        raise ContractError("public_gold_prediction_files_must_be_distinct")
    public = load_public_input_file(public_path)
    gold = load_private_gold_file(gold_path)
    runs = tuple(load_prediction_file(path) for path in prediction_paths)
    validate_evaluation_contract(public, gold, runs)
    return public, gold, runs


class RateMetric(StrictModel):
    numerator: int = Field(strict=True, ge=0)
    denominator: int = Field(strict=True, ge=0)
    value: float | None

    @model_validator(mode="after")
    def validate_fraction(self) -> RateMetric:
        if self.numerator > self.denominator:
            raise ValueError("rate_numerator_exceeds_denominator")
        if self.denominator == 0:
            if self.value not in {None, 0.0}:
                raise ValueError("empty_rate_must_be_none_or_zero")
        else:
            expected = self.numerator / self.denominator
            if self.value is None or not math.isclose(
                self.value, expected, rel_tol=0.0, abs_tol=1e-15
            ):
                raise ValueError("rate_value_mismatch")
        return self


class SliceFalsePositiveMetric(StrictModel):
    phenomenon: OpaqueTag
    formal_false_positives: int = Field(strict=True, ge=0)
    eligible_nonformal_predictions: int = Field(strict=True, ge=0)
    rate: float | None


class EvaluationMetrics(StrictModel):
    case_count: int = Field(strict=True, ge=1)
    repeat_count: int = Field(strict=True, ge=1)
    formal_precision: RateMetric
    worst_case_recall: RateMetric
    hard_negative_specificity: RateMetric
    bc_completeness: RateMetric
    forbidden_reference_rate: RateMetric
    forbidden_reference_count: int = Field(strict=True, ge=0)
    invalid_citation_count: int = Field(strict=True, ge=0)
    joint_accuracy: RateMetric
    formal_false_positive_slices: tuple[SliceFalsePositiveMetric, ...]
    coverage: RateMetric
    outcome_consistency: RateMetric


class GatePolicy(StrictModel):
    """Explicit thresholds; defaults describe a strict freeze gate."""

    min_formal_precision: RateThreshold = 1.0
    min_worst_case_recall: RateThreshold = 1.0
    min_hard_negative_specificity: RateThreshold = 1.0
    min_bc_completeness: RateThreshold = 1.0
    max_forbidden_reference_rate: RateThreshold = 0.0
    min_joint_accuracy: RateThreshold = 1.0
    max_sliced_formal_false_positive_rate: RateThreshold = 0.0
    min_coverage: RateThreshold = 1.0
    min_outcome_consistency: RateThreshold | None = 1.0
    minimum_repeats: int = Field(default=2, strict=True, ge=1, le=100)
    max_invalid_citation_count: int = Field(default=0, strict=True, ge=0)


DEFAULT_GATE_POLICY = GatePolicy()


class GateResult(StrictModel):
    passed: bool
    failures: tuple[str, ...] = Field(max_length=256)


class CaseScore(StrictModel):
    case_id: OpaqueId
    repeat_id: OpaqueId
    run_status: RunStatus
    prediction_present: bool
    assessed_complete: bool
    outcome_surface_correct: bool
    bc_complete: bool
    all_required_evidence_complete: bool
    forbidden_reference_count: int = Field(strict=True, ge=0)
    invalid_citation_count: int = Field(strict=True, ge=0)
    joint_correct: bool


class EvaluationReport(StrictModel):
    metrics: EvaluationMetrics
    gate: GateResult
    case_scores: tuple[CaseScore, ...] = Field(max_length=1_000_000)


def _rate(numerator: int, denominator: int, *, zero_when_empty: bool = False) -> RateMetric:
    value = (
        numerator / denominator
        if denominator
        else 0.0 if zero_when_empty else None
    )
    return RateMetric(numerator=numerator, denominator=denominator, value=value)


def _citation_state(
    public_case: PublicInputCase,
    prediction: CasePrediction | None,
) -> tuple[dict[EvidenceRole, set[str]], int, int]:
    valid: dict[EvidenceRole, set[str]] = {
        role: set() for role in ("B", "C", "G", "X", "P")
    }
    if prediction is None:
        return valid, 0, 0
    evidence = {row.evidence_id: row for row in public_case.evidence_catalog}
    documents = {row.document_id: row for row in public_case.documents}
    invalid = 0
    for citation in prediction.citations:
        anchor = evidence.get(citation.evidence_id)
        document = documents.get(citation.document_id)
        if (
            anchor is None
            or document is None
            or anchor.document_id != citation.document_id
            or anchor.line_start != citation.line_start
            or anchor.line_end != citation.line_end
            or document.version != citation.snapshot_version
            or document.sha256 != citation.snapshot_sha256
        ):
            invalid += 1
            continue
        valid[citation.role].add(citation.evidence_id)
    return valid, invalid, len(prediction.citations)


def _required_complete(
    required: RequiredEvidenceGroups,
    valid: Mapping[EvidenceRole, set[str]],
    roles: Sequence[EvidenceRole],
) -> bool:
    return all(set(getattr(required, role)) <= valid[role] for role in roles)


def _apply_gate(metrics: EvaluationMetrics, policy: GatePolicy) -> GateResult:
    failures: list[str] = []

    def minimum(name: str, metric: RateMetric, threshold: float) -> None:
        if metric.value is None:
            failures.append(f"{name}:undefined")
        elif metric.value < threshold:
            failures.append(f"{name}:below_minimum")

    def maximum(name: str, metric: RateMetric, threshold: float) -> None:
        if metric.value is None:
            failures.append(f"{name}:undefined")
        elif metric.value > threshold:
            failures.append(f"{name}:above_maximum")

    minimum("formal_precision", metrics.formal_precision, policy.min_formal_precision)
    minimum(
        "worst_case_recall", metrics.worst_case_recall,
        policy.min_worst_case_recall,
    )
    minimum(
        "hard_negative_specificity", metrics.hard_negative_specificity,
        policy.min_hard_negative_specificity,
    )
    minimum("bc_completeness", metrics.bc_completeness, policy.min_bc_completeness)
    maximum(
        "forbidden_reference_rate", metrics.forbidden_reference_rate,
        policy.max_forbidden_reference_rate,
    )
    minimum("joint_accuracy", metrics.joint_accuracy, policy.min_joint_accuracy)
    minimum("coverage", metrics.coverage, policy.min_coverage)
    if metrics.repeat_count < policy.minimum_repeats:
        failures.append("repeat_count:below_minimum")
    if policy.min_outcome_consistency is not None:
        minimum(
            "outcome_consistency", metrics.outcome_consistency,
            policy.min_outcome_consistency,
        )
    if metrics.invalid_citation_count > policy.max_invalid_citation_count:
        failures.append("invalid_citation_count:above_maximum")
    for row in metrics.formal_false_positive_slices:
        if (
            row.rate is not None
            and row.rate > policy.max_sliced_formal_false_positive_rate
        ):
            failures.append("sliced_formal_false_positive_rate:above_maximum")
            break
    return GateResult(passed=not failures, failures=tuple(failures))


def score_evaluation(
    public: PublicInputBundle,
    gold: PrivateGoldBundle,
    prediction_runs: Sequence[PredictionBundle],
    *,
    gate_policy: GatePolicy = DEFAULT_GATE_POLICY,
) -> EvaluationReport:
    """Score one or more offline repeats against sealed gold.

    ``joint_accuracy`` requires a complete assessed run, exact outcome and
    surface, every required B/C/G/X/P citation with an exact public snapshot,
    no forbidden evidence ID, and no invalid citation.  B+C completeness is
    separately reported for formal gold cases.  Outcome consistency is
    pairwise across repeats; incomplete/missing/unassessed pairs disagree.
    """

    runs = tuple(prediction_runs)
    if not runs:
        raise ContractError("at_least_one_prediction_repeat_is_required")
    validate_evaluation_contract(public, gold, runs)

    public_cases = {case.case_id: case for case in public.cases}
    gold_cases = {case.case_id: case for case in gold.cases}
    prediction_maps = {
        run.repeat_id: {row.case_id: row for row in run.predictions}
        for run in runs
    }

    formal_predicted = 0
    formal_true_positive = 0
    formal_true_positive_by_repeat = {
        run.repeat_id: 0 for run in runs
    }
    formal_gold_per_repeat = sum(
        case.surface == "formal_issue" for case in gold.cases
    )
    formal_gold_total = 0
    hard_negative_total = 0
    hard_negative_true_negative = 0
    bc_complete_count = 0
    forbidden_count = 0
    citation_count = 0
    invalid_count = 0
    joint_count = 0
    covered_count = 0
    total_slots = len(public.cases) * len(runs)
    case_scores: list[CaseScore] = []
    slice_counts: dict[str, list[int]] = {}

    for run in runs:
        predictions = prediction_maps[run.repeat_id]
        for case_id, public_case in public_cases.items():
            gold_case = gold_cases[case_id]
            prediction = predictions.get(case_id)
            assessed_complete = bool(
                run.run_status == "complete"
                and prediction is not None
                and prediction.outcome != "unassessed"
            )
            if assessed_complete:
                covered_count += 1
            formal_prediction = bool(
                prediction is not None and prediction.surface == "formal_issue"
            )
            if formal_prediction:
                formal_predicted += 1
            formal_gold = gold_case.surface == "formal_issue"
            if formal_gold:
                formal_gold_total += 1
            hard_negative = gold_case.outcome == "no_issue"
            if hard_negative:
                hard_negative_total += 1

            valid, invalid, submitted_citations = _citation_state(
                public_case, prediction
            )
            invalid_count += invalid
            citation_count += submitted_citations
            forbidden = (
                sum(
                    citation.evidence_id in set(gold_case.forbidden_evidence_ids)
                    for citation in prediction.citations
                )
                if prediction is not None else 0
            )
            forbidden_count += forbidden
            outcome_surface_correct = bool(
                assessed_complete
                and prediction is not None
                and prediction.outcome == gold_case.outcome
                and prediction.surface == gold_case.surface
            )
            formal_correct = bool(
                outcome_surface_correct and formal_gold and formal_prediction
            )
            if formal_correct:
                formal_true_positive += 1
                formal_true_positive_by_repeat[run.repeat_id] += 1
            if hard_negative and outcome_surface_correct:
                hard_negative_true_negative += 1

            bc_complete = bool(
                formal_correct
                and _required_complete(
                    gold_case.required_evidence, valid, ("B", "C")
                )
            )
            if formal_gold and bc_complete:
                bc_complete_count += 1
            all_required_complete = _required_complete(
                gold_case.required_evidence,
                valid,
                ("B", "C", "G", "X", "P"),
            )
            joint_correct = bool(
                outcome_surface_correct
                and all_required_complete
                and forbidden == 0
                and invalid == 0
            )
            if joint_correct:
                joint_count += 1

            if not formal_gold:
                for phenomenon in gold_case.phenomena:
                    counts = slice_counts.setdefault(phenomenon, [0, 0])
                    counts[1] += 1
                    if formal_prediction:
                        counts[0] += 1

            case_scores.append(
                CaseScore(
                    case_id=case_id,
                    repeat_id=run.repeat_id,
                    run_status=run.run_status,
                    prediction_present=prediction is not None,
                    assessed_complete=assessed_complete,
                    outcome_surface_correct=outcome_surface_correct,
                    bc_complete=bc_complete,
                    all_required_evidence_complete=all_required_complete,
                    forbidden_reference_count=forbidden,
                    invalid_citation_count=invalid,
                    joint_correct=joint_correct,
                )
            )

    consistency_matches = 0
    consistency_total = 0
    for left_index, left in enumerate(runs):
        for right in runs[left_index + 1:]:
            left_predictions = prediction_maps[left.repeat_id]
            right_predictions = prediction_maps[right.repeat_id]
            for case_id in public_cases:
                consistency_total += 1
                left_row = left_predictions.get(case_id)
                right_row = right_predictions.get(case_id)
                if (
                    left.run_status == right.run_status == "complete"
                    and left_row is not None
                    and right_row is not None
                    and left_row.outcome != "unassessed"
                    and right_row.outcome != "unassessed"
                    and left_row.outcome == right_row.outcome
                ):
                    consistency_matches += 1

    slices = tuple(
        SliceFalsePositiveMetric(
            phenomenon=phenomenon,
            formal_false_positives=counts[0],
            eligible_nonformal_predictions=counts[1],
            rate=counts[0] / counts[1] if counts[1] else None,
        )
        for phenomenon, counts in sorted(slice_counts.items())
    )
    metrics = EvaluationMetrics(
        case_count=len(public.cases),
        repeat_count=len(runs),
        formal_precision=_rate(formal_true_positive, formal_predicted),
        worst_case_recall=_rate(
            min(formal_true_positive_by_repeat.values()),
            formal_gold_per_repeat,
        ),
        hard_negative_specificity=_rate(
            hard_negative_true_negative, hard_negative_total
        ),
        bc_completeness=_rate(bc_complete_count, formal_gold_total),
        forbidden_reference_rate=_rate(
            forbidden_count, citation_count, zero_when_empty=True
        ),
        forbidden_reference_count=forbidden_count,
        invalid_citation_count=invalid_count,
        joint_accuracy=_rate(joint_count, total_slots),
        formal_false_positive_slices=slices,
        coverage=_rate(covered_count, total_slots),
        outcome_consistency=_rate(consistency_matches, consistency_total),
    )
    return EvaluationReport(
        metrics=metrics,
        gate=_apply_gate(metrics, gate_policy),
        case_scores=tuple(case_scores),
    )


__all__ = [
    "AxisSnapshot",
    "CasePrediction",
    "Citation",
    "CommitmentError",
    "ContractError",
    "DEFAULT_GATE_POLICY",
    "DocumentSnapshot",
    "DualAdjudication",
    "EvaluationMetrics",
    "EvaluationReport",
    "EvidenceAnchor",
    "GatePolicy",
    "GoldCommitment",
    "IndependentEventGroup",
    "PredictionBundle",
    "PrivateGoldBundle",
    "PrivateGoldCase",
    "PublicInputBundle",
    "PublicInputCase",
    "RateMetric",
    "RequiredEvidenceGroups",
    "ReviewerJudgement",
    "StageFunnel",
    "canonical_json_bytes",
    "canonical_sha256",
    "create_gold_commitment",
    "load_evaluation_files",
    "load_prediction_file",
    "load_predictions",
    "load_private_gold",
    "load_private_gold_file",
    "load_public_input_file",
    "load_public_inputs",
    "score_evaluation",
    "validate_evaluation_contract",
    "validate_no_cross_split_group_leakage",
    "verify_gold_commitment",
]
