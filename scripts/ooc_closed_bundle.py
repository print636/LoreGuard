"""Build an answer-free, neutral-ID public package for closed OOC evaluation.

The source plan contains author intent (document roles and baseline trait
selectors) but no expected result, evidence role or gold label.  The generated
package randomizes every case/document/axis/evidence identity, copies only the
frozen public documents and emits a hash-bound author setup plus freeze file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import ooc_annotation_workflow as annotation
from scripts import ooc_eval_contract as contract


SOURCE_SCHEMA_VERSION = "loreguard-ooc-source-plan-v1"
AUTHOR_SETUP_SCHEMA_VERSION = "loreguard-ooc-author-setup-v1"
FREEZE_SCHEMA_VERSION = "loreguard-ooc-execution-freeze-v1"
MAX_SOURCE_PLAN_BYTES = 2 * 1024 * 1024
MAX_PUBLIC_DOCUMENT_BYTES = 8 * 1024 * 1024

SourceKey = Annotated[str, Field(strict=True, pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
TraitType = Literal[
    "core_personality",
    "preference",
    "speech_pattern",
    "value",
    "relationship_attitude",
    "motivation_goal",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SourceDocument(StrictModel):
    document_key: SourceKey
    source_path: str = Field(strict=True, min_length=1, max_length=500)
    phase: Literal["baseline", "target"]
    document_role: Literal["canon", "character_profile", "chapter"]
    story_scope: str = Field(strict=True, min_length=1, max_length=100)
    resolution_state: Literal["confirmed", "draft"]
    publication_status: Literal["published", "draft"]
    import_order: int = Field(strict=True, ge=1, le=1_000)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> SourceDocument:
        if self.phase == "baseline" and (
            self.resolution_state != "confirmed"
            or self.publication_status != "published"
        ):
            raise ValueError("baseline_document_must_be_confirmed_and_published")
        if self.phase == "target" and (
            self.resolution_state != "draft" or self.publication_status != "draft"
        ):
            raise ValueError("target_document_must_remain_draft")
        return self


class SourceAuthorAxis(StrictModel):
    display_name: str = Field(strict=True, min_length=1, max_length=120)
    definition: str = Field(strict=True, min_length=1, max_length=1_000)
    positive_proposition: str = Field(strict=True, min_length=1, max_length=1_000)
    applicability_scope: str = Field(strict=True, min_length=1, max_length=1_000)
    axis_alignment: Literal["same", "opposite"]


class SourceCandidateSelector(StrictModel):
    character_key: str = Field(strict=True, min_length=1, max_length=120)
    trait_type: TraitType
    source_document_key: SourceKey
    source_line_start: int = Field(strict=True, ge=1, le=10_000_000)
    source_line_end: int = Field(strict=True, ge=1, le=10_000_000)
    origin: Literal["explicit_setting", "published_history"]
    polarity: Literal["positive", "negative", "neutral"]
    stability: Literal["core", "stable"]

    @model_validator(mode="after")
    def validate_range(self) -> SourceCandidateSelector:
        if self.source_line_end < self.source_line_start:
            raise ValueError("selector_line_range_invalid")
        return self


class SourceCase(StrictModel):
    case_key: SourceKey
    candidate_selector: SourceCandidateSelector
    author_axis: SourceAuthorAxis | None = None

    @model_validator(mode="after")
    def validate_author_axis_dimension(self) -> SourceCase:
        if (
            self.author_axis is not None
            and self.candidate_selector.trait_type
            not in {"core_personality", "value"}
        ):
            raise ValueError("author_axis_dimension_not_supported_by_product_api")
        return self


class SourceGroup(StrictModel):
    group_key: SourceKey
    documents: tuple[SourceDocument, ...] = Field(min_length=2, max_length=32)
    cases: tuple[SourceCase, ...] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_group(self) -> SourceGroup:
        document_keys = tuple(row.document_key for row in self.documents)
        case_keys = tuple(row.case_key for row in self.cases)
        if len(document_keys) != len(set(document_keys)):
            raise ValueError("duplicate_source_document_key")
        if len(case_keys) != len(set(case_keys)):
            raise ValueError("duplicate_source_case_key")
        targets = [row for row in self.documents if row.phase == "target"]
        if len(targets) != 1:
            raise ValueError("group_requires_exactly_one_target_document")
        baseline_keys = {row.document_key for row in self.documents if row.phase == "baseline"}
        for case in self.cases:
            if case.candidate_selector.source_document_key not in baseline_keys:
                raise ValueError("candidate_selector_must_reference_baseline_document")
        return self


class SourcePlan(StrictModel):
    schema_version: Literal["loreguard-ooc-source-plan-v1"] = SOURCE_SCHEMA_VERSION
    split_id: str = Field(strict=True, min_length=1, max_length=100)
    groups: tuple[SourceGroup, ...] = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def validate_unique_source_keys(self) -> SourcePlan:
        groups = tuple(row.group_key for row in self.groups)
        cases = tuple(case.case_key for group in self.groups for case in group.cases)
        if len(groups) != len(set(groups)):
            raise ValueError("duplicate_source_group_key")
        if len(cases) != len(set(cases)):
            raise ValueError("duplicate_source_case_key_across_groups")
        return self


class PublicDocumentSetup(StrictModel):
    document_id: contract.OpaqueId
    phase: Literal["baseline", "target"]
    document_role: Literal["canon", "character_profile", "chapter"]
    story_scope: str = Field(strict=True, min_length=1, max_length=100)
    resolution_state: Literal["confirmed", "draft"]
    publication_status: Literal["published", "draft"]
    import_order: int = Field(strict=True, ge=1, le=1_000)


class PublicCandidateSelector(StrictModel):
    character_key: str = Field(strict=True, min_length=1, max_length=120)
    trait_type: TraitType
    source_document_id: contract.OpaqueId
    source_line_start: int = Field(strict=True, ge=1, le=10_000_000)
    source_line_end: int = Field(strict=True, ge=1, le=10_000_000)
    origin: Literal["explicit_setting", "published_history"]
    polarity: Literal["positive", "negative", "neutral"]
    stability: Literal["core", "stable"]


class PublicAuthorSetupCase(StrictModel):
    case_id: contract.OpaqueId
    group_id: contract.OpaqueId
    target_document_id: contract.OpaqueId
    documents: tuple[PublicDocumentSetup, ...] = Field(min_length=2, max_length=32)
    candidate_selector: PublicCandidateSelector
    author_axis: SourceAuthorAxis | None = None
    unselected_candidate_policy: Literal["reject"] = "reject"


class PublicAuthorSetupBundle(StrictModel):
    schema_version: Literal["loreguard-ooc-author-setup-v1"] = (
        AUTHOR_SETUP_SCHEMA_VERSION
    )
    dataset_id: contract.OpaqueId
    public_input_sha256: contract.Sha256
    cases: tuple[PublicAuthorSetupCase, ...] = Field(
        min_length=1, max_length=contract.MAX_CASES
    )


class ExecutionFreeze(StrictModel):
    schema_version: Literal["loreguard-ooc-execution-freeze-v1"] = (
        FREEZE_SCHEMA_VERSION
    )
    dataset_id: contract.OpaqueId
    public_input_sha256: contract.Sha256
    author_setup_sha256: contract.Sha256
    case_count: int = Field(strict=True, ge=1, le=contract.MAX_CASES)
    group_count: int = Field(strict=True, ge=1, le=contract.MAX_CASES)


class BundleBuildError(ValueError):
    pass


def _load_source_plan(path: Path) -> SourcePlan:
    text = str(path)
    if "://" in text or text.startswith(("\\\\", "//")):
        raise BundleBuildError("network_locations_are_forbidden")
    if annotation._path_chain_has_reparse_point(path):
        raise BundleBuildError("reparse_point_paths_are_forbidden")
    try:
        if path.stat().st_size > MAX_SOURCE_PLAN_BYTES:
            raise BundleBuildError("source_plan_too_large")
        payload = path.read_bytes()
    except OSError as exc:
        raise BundleBuildError("source_plan_unreadable") from exc
    return SourcePlan.model_validate_json(payload)


def _read_source_document(root: Path, relative: str) -> bytes:
    root_text = str(root)
    if "://" in root_text or root_text.startswith(("\\\\", "//")):
        raise BundleBuildError("network_locations_are_forbidden")
    if annotation._path_chain_has_reparse_point(root):
        raise BundleBuildError("reparse_point_paths_are_forbidden")
    if (
        "\\" in relative
        or ":" in relative
        or relative.startswith("//")
    ):
        raise BundleBuildError("source_document_path_invalid")
    parsed = Path(relative)
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise BundleBuildError("source_document_path_invalid")
    try:
        resolved_root = root.resolve(strict=True)
        path = resolved_root.joinpath(*parsed.parts)
        if path.is_symlink():
            raise BundleBuildError("source_document_symlink_forbidden")
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(resolved_root):
            raise BundleBuildError("source_document_path_escapes_root")
        if resolved.stat().st_size > MAX_PUBLIC_DOCUMENT_BYTES:
            raise BundleBuildError("source_document_too_large")
        payload = resolved.read_bytes()
    except OSError as exc:
        raise BundleBuildError("source_document_unreadable") from exc
    if b"\r" in payload:
        raise BundleBuildError("source_document_must_use_lf")
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleBuildError("source_document_must_be_utf8") from exc
    return payload


def _canonical_sha(model: BaseModel) -> str:
    return hashlib.sha256(contract.canonical_json_bytes(model)).hexdigest()


def _write_exclusive_json(path: Path, model: BaseModel) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n"
        )


def validate_author_setup_binding(
    public: contract.PublicInputBundle,
    setup: PublicAuthorSetupBundle,
) -> None:
    if setup.dataset_id != public.dataset_id:
        raise BundleBuildError("author_setup_dataset_mismatch")
    if setup.public_input_sha256 != contract.canonical_sha256(public):
        raise BundleBuildError("author_setup_public_hash_mismatch")
    public_cases = {row.case_id: row for row in public.cases}
    setup_cases = {row.case_id: row for row in setup.cases}
    if len(setup_cases) != len(setup.cases):
        raise BundleBuildError("duplicate_author_setup_case_id")
    if set(public_cases) != set(setup_cases):
        raise BundleBuildError("author_setup_case_set_mismatch")
    for case_id, public_case in public_cases.items():
        row = setup_cases[case_id]
        if row.group_id != public_case.group_id:
            raise BundleBuildError("author_setup_group_mismatch")
        public_docs = {value.document_id for value in public_case.documents}
        setup_docs = {value.document_id for value in row.documents}
        if len(setup_docs) != len(row.documents):
            raise BundleBuildError("duplicate_author_setup_document_id")
        if public_docs != setup_docs:
            raise BundleBuildError("author_setup_document_set_mismatch")
        if row.target_document_id not in public_docs:
            raise BundleBuildError("author_setup_target_document_unknown")
        if row.candidate_selector.source_document_id not in public_docs:
            raise BundleBuildError("author_setup_selector_document_unknown")
        setup_by_document = {value.document_id: value for value in row.documents}
        import_orders = tuple(value.import_order for value in row.documents)
        if len(import_orders) != len(set(import_orders)):
            raise BundleBuildError("duplicate_author_setup_import_order")
        targets = [value for value in row.documents if value.phase == "target"]
        if len(targets) != 1 or targets[0].document_id != row.target_document_id:
            raise BundleBuildError("author_setup_requires_exactly_one_target")
        for document in row.documents:
            expected_lifecycle = (
                ("confirmed", "published")
                if document.phase == "baseline"
                else ("draft", "draft")
            )
            if (
                document.resolution_state,
                document.publication_status,
            ) != expected_lifecycle:
                raise BundleBuildError("author_setup_document_lifecycle_mismatch")
        target = setup_by_document[row.target_document_id]
        if target.phase != "target":
            raise BundleBuildError("author_setup_target_document_phase_mismatch")
        baseline = setup_by_document[row.candidate_selector.source_document_id]
        if baseline.phase != "baseline":
            raise BundleBuildError("author_setup_selector_must_reference_baseline")
        selector = row.candidate_selector
        selector_anchors = [
            value
            for value in public_case.evidence_catalog
            if value.document_id == selector.source_document_id
            and selector.source_line_start <= value.line_start
            and value.line_end <= selector.source_line_end
        ]
        if (
            not selector_anchors
            or min(value.line_start for value in selector_anchors)
            != selector.source_line_start
            or max(value.line_end for value in selector_anchors)
            != selector.source_line_end
        ):
            raise BundleBuildError("author_setup_selector_not_bound_to_public_lines")
        axis_payload = json.dumps(
            {
                "candidate_selector": selector.model_dump(mode="json"),
                "author_axis": (
                    row.author_axis.model_dump(mode="json")
                    if row.author_axis is not None
                    else None
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if (
            public_case.axis.version != 1
            or hashlib.sha256(axis_payload).hexdigest()
            != public_case.axis.sha256
        ):
            raise BundleBuildError("author_setup_axis_binding_mismatch")


def validate_execution_freeze_binding(
    public: contract.PublicInputBundle,
    setup: PublicAuthorSetupBundle,
    freeze: ExecutionFreeze,
) -> None:
    """Validate the full public/setup/freeze envelope as one immutable unit."""

    validate_author_setup_binding(public, setup)
    public_digest = contract.canonical_sha256(public)
    setup_digest = contract.canonical_sha256(setup)
    if freeze.dataset_id != public.dataset_id:
        raise BundleBuildError("execution_freeze_dataset_mismatch")
    if freeze.public_input_sha256 != public_digest:
        raise BundleBuildError("execution_freeze_public_hash_mismatch")
    if freeze.author_setup_sha256 != setup_digest:
        raise BundleBuildError("execution_freeze_author_setup_hash_mismatch")
    if freeze.case_count != len(public.cases):
        raise BundleBuildError("execution_freeze_case_count_mismatch")
    public_group_ids = {row.group_id for row in public.cases}
    setup_group_ids = {row.group_id for row in setup.cases}
    if setup_group_ids != public_group_ids:
        raise BundleBuildError("execution_freeze_group_set_mismatch")
    if freeze.group_count != len(public_group_ids):
        raise BundleBuildError("execution_freeze_group_count_mismatch")


def build_bundle(source_plan: Path, source_root: Path, output_root: Path) -> ExecutionFreeze:
    plan = _load_source_plan(source_plan)
    output_text = str(output_root)
    if "://" in output_text or output_text.startswith(("\\\\", "//")):
        raise BundleBuildError("network_locations_are_forbidden")
    if annotation._path_chain_has_reparse_point(output_root.parent):
        raise BundleBuildError("reparse_point_paths_are_forbidden")
    try:
        output_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise BundleBuildError("output_root_already_exists") from exc

    dataset_id = str(uuid4())
    public_cases: list[contract.PublicInputCase] = []
    setup_cases_raw: list[dict[str, object]] = []
    try:
        for source_group in plan.groups:
            group_id = str(uuid4())
            world_id = str(uuid4())
            group_dir = output_root / "cases" / group_id
            group_dir.mkdir(parents=True)
            documents: list[contract.DocumentSnapshot] = []
            document_ids: dict[str, str] = {}
            document_line_counts: dict[str, int] = {}
            evidence: list[contract.EvidenceAnchor] = []
            setup_documents: list[PublicDocumentSetup] = []
            for source_document in sorted(
                source_group.documents, key=lambda value: value.import_order
            ):
                payload = _read_source_document(source_root, source_document.source_path)
                document_id = str(uuid4())
                document_ids[source_document.document_key] = document_id
                document_line_counts[source_document.document_key] = len(
                    payload.decode("utf-8").splitlines()
                )
                if Path(source_document.source_path).suffix.lower() != ".md":
                    raise BundleBuildError("public_documents_must_be_markdown")
                destination = group_dir / f"{document_id}.md"
                destination.write_bytes(payload)
                relative_path = destination.relative_to(output_root).as_posix()
                documents.append(
                    contract.DocumentSnapshot(
                        document_id=document_id,
                        path=relative_path,
                        version=1,
                        sha256=hashlib.sha256(payload).hexdigest(),
                    )
                )
                setup_documents.append(
                    PublicDocumentSetup(
                        document_id=document_id,
                        phase=source_document.phase,
                        document_role=source_document.document_role,
                        story_scope=source_document.story_scope,
                        resolution_state=source_document.resolution_state,
                        publication_status=source_document.publication_status,
                        import_order=source_document.import_order,
                    )
                )
                for line_number, line in enumerate(
                    payload.decode("utf-8").splitlines(), start=1
                ):
                    if line.strip():
                        evidence.append(
                            contract.EvidenceAnchor(
                                evidence_id=str(uuid4()),
                                document_id=document_id,
                                line_start=line_number,
                                line_end=line_number,
                            )
                        )

            target_document_id = next(
                document_ids[row.document_key]
                for row in source_group.documents
                if row.phase == "target"
            )
            for source_case in source_group.cases:
                case_id = str(uuid4())
                axis_id = str(uuid4())
                selector = source_case.candidate_selector
                if selector.source_line_end > document_line_counts[
                    selector.source_document_key
                ]:
                    raise BundleBuildError("candidate_selector_line_out_of_range")
                setup_case_payload: dict[str, object] = {
                    "case_id": case_id,
                    "group_id": group_id,
                    "target_document_id": target_document_id,
                    "documents": tuple(setup_documents),
                    "candidate_selector": PublicCandidateSelector(
                        character_key=selector.character_key,
                        trait_type=selector.trait_type,
                        source_document_id=document_ids[selector.source_document_key],
                        source_line_start=selector.source_line_start,
                        source_line_end=selector.source_line_end,
                        origin=selector.origin,
                        polarity=selector.polarity,
                        stability=selector.stability,
                    ),
                    "author_axis": source_case.author_axis,
                    "unselected_candidate_policy": "reject",
                }
                axis_payload = json.dumps(
                    {
                        "candidate_selector": setup_case_payload[
                            "candidate_selector"
                        ].model_dump(mode="json"),
                        "author_axis": (
                            source_case.author_axis.model_dump(mode="json")
                            if source_case.author_axis is not None
                            else None
                        ),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                public_cases.append(
                    contract.PublicInputCase(
                        case_id=case_id,
                        split_id=plan.split_id,
                        group_id=group_id,
                        world_id=world_id,
                        documents=tuple(documents),
                        axis=contract.AxisSnapshot(
                            axis_id=axis_id,
                            version=1,
                            sha256=hashlib.sha256(axis_payload).hexdigest(),
                        ),
                        evidence_catalog=tuple(evidence),
                    )
                )
                setup_cases_raw.append(setup_case_payload)

        public = contract.PublicInputBundle(
            dataset_id=dataset_id,
            cases=tuple(public_cases),
        )
        setup = PublicAuthorSetupBundle(
            dataset_id=dataset_id,
            public_input_sha256=contract.canonical_sha256(public),
            cases=tuple(
                PublicAuthorSetupCase.model_validate(value)
                for value in setup_cases_raw
            ),
        )
        validate_author_setup_binding(public, setup)
        annotation.validate_exhaustive_line_catalog(public, output_root)
        freeze = ExecutionFreeze(
            dataset_id=dataset_id,
            public_input_sha256=contract.canonical_sha256(public),
            author_setup_sha256=_canonical_sha(setup),
            case_count=len(public.cases),
            group_count=len(plan.groups),
        )
        validate_execution_freeze_binding(public, setup, freeze)
        _write_exclusive_json(output_root / "public-input.json", public)
        _write_exclusive_json(output_root / "author-setup.json", setup)
        _write_exclusive_json(output_root / "freeze.json", freeze)
        return freeze
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a neutral-ID, answer-free public OOC holdout package."
    )
    parser.add_argument("--source-plan", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    freeze = build_bundle(
        Path(args.source_plan), Path(args.source_root), Path(args.output_root)
    )
    print(
        json.dumps(
            {
                "dataset_id": freeze.dataset_id,
                "case_count": freeze.case_count,
                "group_count": freeze.group_count,
                "public_input_sha256": freeze.public_input_sha256,
                "author_setup_sha256": freeze.author_setup_sha256,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHOR_SETUP_SCHEMA_VERSION",
    "FREEZE_SCHEMA_VERSION",
    "SOURCE_SCHEMA_VERSION",
    "BundleBuildError",
    "ExecutionFreeze",
    "PublicAuthorSetupBundle",
    "SourcePlan",
    "build_bundle",
    "validate_author_setup_binding",
    "validate_execution_freeze_binding",
]
