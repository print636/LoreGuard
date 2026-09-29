"""Run a frozen public OOC package through LoreGuard's HTTP API.

The runner is deliberately one-way: it loads the three public control files,
the Markdown documents named by the public manifest, and one runtime config.
It has no private-label input and no scoring capability.  Every case gets a
fresh project.  Any ambiguity is projected as ``unassessed``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol
from urllib.parse import quote, urlsplit
from uuid import UUID, uuid4

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import ooc_eval_contract as contract
from scripts.ooc_sealed_prediction import (
    RuntimeCaseSnapshot,
    RuntimeEvidenceRef,
    SealedPredictionArtifact,
    build_prediction_artifact,
)


RUN_CONFIG_SCHEMA = "loreguard-ooc-sealed-http-config-v1"
RUN_REPORT_SCHEMA = "loreguard-ooc-sealed-http-report-v1"
RUNNER_CODE_MANIFEST_SCHEMA = "loreguard-ooc-sealed-runner-code-v1"
AUTHOR_SETUP_SCHEMA = "loreguard-ooc-author-setup-v1"
EXECUTION_FREEZE_SCHEMA = "loreguard-ooc-execution-freeze-v1"
ISOLATION_SCHEMA = "character-axis-evaluation-isolation-v1"
RUNNER_CODE_PATHS = (
    "scripts/run_ooc_sealed_http.py",
    "scripts/ooc_sealed_prediction.py",
    "scripts/ooc_eval_contract.py",
)
MAX_CONTROL_BYTES = 8 * 1024 * 1024
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_PENDING_CANDIDATES = 5_000
MAX_REPORT_ROWS = 1_024
TERMINAL_RUN_STATES = {"completed", "failed", "cancelled"}
_UUID_TEXT = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class RunnerCodeFile(StrictModel):
    path: Literal[
        "scripts/run_ooc_sealed_http.py",
        "scripts/ooc_sealed_prediction.py",
        "scripts/ooc_eval_contract.py",
    ]
    sha256: contract.Sha256


class RunnerCodeManifest(StrictModel):
    schema_version: Literal["loreguard-ooc-sealed-runner-code-v1"] = (
        RUNNER_CODE_MANIFEST_SCHEMA
    )
    files: tuple[RunnerCodeFile, ...] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_exact_file_set(self) -> "RunnerCodeManifest":
        if tuple(row.path for row in self.files) != RUNNER_CODE_PATHS:
            raise ValueError("runner_code_manifest_file_set_mismatch")
        return self


TraitType = Literal[
    "core_personality",
    "preference",
    "speech_pattern",
    "value",
    "relationship_attitude",
    "motivation_goal",
]


class AuthorAxis(StrictModel):
    display_name: str = Field(strict=True, min_length=1, max_length=120)
    definition: str = Field(strict=True, min_length=1, max_length=1_000)
    positive_proposition: str = Field(strict=True, min_length=1, max_length=1_000)
    applicability_scope: str = Field(strict=True, min_length=1, max_length=1_000)
    axis_alignment: Literal["same", "opposite"]


class PublicDocumentSetup(StrictModel):
    document_id: contract.OpaqueId
    phase: Literal["baseline", "target"]
    document_role: Literal["canon", "character_profile", "chapter"]
    story_scope: str = Field(
        strict=True,
        min_length=1,
        max_length=80,
        pattern=r"^[A-Za-z0-9_\-\u4e00-\u9fff]+$",
    )
    resolution_state: Literal["confirmed", "draft"]
    publication_status: Literal["published", "draft"]
    import_order: int = Field(strict=True, ge=1, le=1_000)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> "PublicDocumentSetup":
        if self.phase == "baseline" and (
            self.resolution_state != "confirmed"
            or self.publication_status != "published"
        ):
            raise ValueError("baseline_lifecycle_invalid")
        if self.phase == "target" and (
            self.resolution_state != "draft"
            or self.publication_status != "draft"
        ):
            raise ValueError("target_lifecycle_invalid")
        return self


class PublicCandidateSelector(StrictModel):
    character_key: str = Field(strict=True, min_length=1, max_length=120)
    trait_type: TraitType
    source_document_id: contract.OpaqueId
    source_line_start: int = Field(strict=True, ge=1, le=10_000_000)
    source_line_end: int = Field(strict=True, ge=1, le=10_000_000)
    origin: Literal["explicit_setting", "published_history"]
    polarity: Literal["positive", "negative", "neutral"]
    stability: Literal["core", "stable"]

    @model_validator(mode="after")
    def validate_range(self) -> "PublicCandidateSelector":
        if self.source_line_end < self.source_line_start:
            raise ValueError("selector_line_range_invalid")
        return self


class PublicAuthorSetupCase(StrictModel):
    case_id: contract.OpaqueId
    group_id: contract.OpaqueId
    target_document_id: contract.OpaqueId
    documents: tuple[PublicDocumentSetup, ...] = Field(min_length=2, max_length=128)
    candidate_selector: PublicCandidateSelector
    author_axis: AuthorAxis | None = None
    unselected_candidate_policy: Literal["reject"] = "reject"

    @model_validator(mode="after")
    def validate_case(self) -> "PublicAuthorSetupCase":
        ids = [row.document_id for row in self.documents]
        orders = [row.import_order for row in self.documents]
        if len(ids) != len(set(ids)) or len(orders) != len(set(orders)):
            raise ValueError("setup_document_identity_or_order_duplicate")
        target = [row for row in self.documents if row.phase == "target"]
        if len(target) != 1 or target[0].document_id != self.target_document_id:
            raise ValueError("setup_requires_exact_target")
        baseline_ids = {row.document_id for row in self.documents if row.phase == "baseline"}
        if self.candidate_selector.source_document_id not in baseline_ids:
            raise ValueError("selector_must_reference_baseline")
        axis_dimensions = {"core_personality", "value"}
        if (
            self.candidate_selector.trait_type in axis_dimensions
            and self.author_axis is None
        ):
            raise ValueError("author_axis_required_for_supported_dimension")
        if (
            self.candidate_selector.trait_type not in axis_dimensions
            and self.author_axis is not None
        ):
            raise ValueError("author_axis_for_unsupported_dimension")
        return self


class PublicAuthorSetupBundle(StrictModel):
    schema_version: Literal["loreguard-ooc-author-setup-v1"] = AUTHOR_SETUP_SCHEMA
    dataset_id: contract.OpaqueId
    public_input_sha256: contract.Sha256
    cases: tuple[PublicAuthorSetupCase, ...] = Field(
        min_length=1, max_length=contract.MAX_CASES
    )


class ExecutionFreeze(StrictModel):
    schema_version: Literal["loreguard-ooc-execution-freeze-v1"] = (
        EXECUTION_FREEZE_SCHEMA
    )
    dataset_id: contract.OpaqueId
    public_input_sha256: contract.Sha256
    author_setup_sha256: contract.Sha256
    case_count: int = Field(strict=True, ge=1, le=contract.MAX_CASES)
    group_count: int = Field(strict=True, ge=1, le=contract.MAX_CASES)


class IsolationExpectation(StrictModel):
    schema_version: Literal["character-axis-evaluation-isolation-v1"] = (
        ISOLATION_SCHEMA
    )
    isolated_sqlite: Literal[True] = True
    eval_db_id: str = Field(strict=True, min_length=36, max_length=36)
    database_path_sha256: contract.Sha256
    instance_id_sha256: contract.Sha256

    @model_validator(mode="after")
    def validate_eval_id(self) -> "IsolationExpectation":
        if not _is_uuid4(self.eval_db_id):
            raise ValueError("evaluation_database_id_must_be_uuid4")
        return self


class SealedHttpRunConfig(StrictModel):
    schema_version: Literal["loreguard-ooc-sealed-http-config-v1"] = RUN_CONFIG_SCHEMA
    base_url: str = Field(strict=True, min_length=1, max_length=500)
    isolation: IsolationExpectation
    execution_id: str = Field(strict=True, min_length=36, max_length=36)
    repeat_id: contract.OpaqueId
    expected_runtime_provenance_sha256: contract.Sha256
    expected_runner_source_sha256: contract.Sha256
    sensitivity: Literal["conservative", "balanced", "exploratory"] = "balanced"
    run_timeout_seconds: float = Field(strict=True, ge=1.0, le=7_200.0)
    poll_interval_seconds: float = Field(strict=True, ge=0.01, le=10.0)

    @model_validator(mode="after")
    def validate_local_url(self) -> "SealedHttpRunConfig":
        try:
            parsed = urlsplit(self.base_url)
            port = parsed.port
        except ValueError as exc:
            raise ValueError("sealed_base_url_invalid") from exc
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
            or port is None
            or port in {8000, 8080}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("sealed_base_url_invalid")
        if not _is_uuid4(self.execution_id):
            raise ValueError("execution_id_must_be_uuid4")
        return self


class ProjectReceipt(StrictModel):
    case_id: contract.OpaqueId
    project_id_sha256: contract.Sha256


class SealedHttpRunReport(StrictModel):
    schema_version: Literal["loreguard-ooc-sealed-http-report-v1"] = RUN_REPORT_SCHEMA
    phase: Literal["preflight", "completed", "partial", "failed"]
    dataset_id: contract.OpaqueId | None = None
    public_input_sha256: contract.Sha256 | None = None
    author_setup_sha256: contract.Sha256 | None = None
    execution_freeze_sha256: contract.Sha256 | None = None
    runtime_provenance_sha256: contract.Sha256 | None = None
    run_config_sha256: contract.Sha256 | None = None
    run_config: SealedHttpRunConfig | None = None
    runner_source_sha256: contract.Sha256 | None = None
    execution_id: str | None = Field(default=None, min_length=36, max_length=36)
    project_set_sha256: contract.Sha256
    project_receipts: tuple[ProjectReceipt, ...] = Field(
        default=(), max_length=contract.MAX_CASES
    )
    independent_project_count: int = Field(strict=True, ge=0, le=contract.MAX_CASES)
    assessed_case_count: int = Field(strict=True, ge=0, le=contract.MAX_CASES)
    unassessed_case_count: int = Field(strict=True, ge=0, le=contract.MAX_CASES)
    failure_code: str | None = Field(default=None, max_length=100)
    prediction_artifact: SealedPredictionArtifact | None = None

    @model_validator(mode="after")
    def validate_receipt_and_artifact_binding(self) -> "SealedHttpRunReport":
        if self.execution_id is not None and not _is_uuid4(self.execution_id):
            raise ValueError("report_execution_id_must_be_uuid4")
        case_ids = [row.case_id for row in self.project_receipts]
        project_hashes = [row.project_id_sha256 for row in self.project_receipts]
        if (
            len(case_ids) != len(set(case_ids))
            or len(project_hashes) != len(set(project_hashes))
            or self.independent_project_count != len(self.project_receipts)
            or self.project_set_sha256 != _project_set_sha(self.project_receipts)
        ):
            raise ValueError("project_receipt_binding_mismatch")
        artifact = self.prediction_artifact
        if self.run_config is not None:
            if (
                self.run_config_sha256 != _canonical_model_sha(self.run_config)
                or self.execution_id != self.run_config.execution_id
                or self.runner_source_sha256
                != self.run_config.expected_runner_source_sha256
                or (
                    self.runtime_provenance_sha256 is not None
                    and self.runtime_provenance_sha256
                    != self.run_config.expected_runtime_provenance_sha256
                )
            ):
                raise ValueError("report_run_config_binding_mismatch")
        if artifact is None:
            if self.phase in {"completed", "partial"}:
                raise ValueError("successful_report_requires_prediction_artifact")
            return self
        if self.phase not in {"completed", "partial"}:
            raise ValueError("prediction_artifact_phase_mismatch")
        if (
            artifact.dataset_id != self.dataset_id
            or artifact.public_input_sha256 != self.public_input_sha256
            or artifact.author_setup_sha256 != self.author_setup_sha256
            or artifact.execution_freeze_sha256 != self.execution_freeze_sha256
            or artifact.runtime_provenance_sha256 != self.runtime_provenance_sha256
            or self.run_config_sha256 is None
            or self.run_config is None
            or self.runner_source_sha256 is None
            or self.execution_id is None
        ):
            raise ValueError("report_prediction_artifact_binding_mismatch")
        prediction_case_ids = {
            row.case_id for row in artifact.prediction.predictions
        }
        if (
            not set(case_ids).issubset(prediction_case_ids)
            or len(self.project_receipts) > len(prediction_case_ids)
            or (
                self.phase == "completed"
                and set(case_ids) != prediction_case_ids
            )
        ):
            raise ValueError("project_receipt_case_set_mismatch")
        prediction_complete = artifact.prediction.run_status == "complete"
        if (self.phase == "completed") != prediction_complete:
            raise ValueError("report_prediction_status_mismatch")
        if (
            self.assessed_case_count + self.unassessed_case_count
            != len(artifact.prediction.predictions)
            or self.assessed_case_count
            != sum(
                row.outcome != "unassessed"
                for row in artifact.prediction.predictions
            )
        ):
            raise ValueError("report_prediction_count_mismatch")
        return self


class ResponseLike(Protocol):
    status_code: int
    content: bytes

    def json(self) -> Any: ...


class TransportLike(Protocol):
    def request(self, method: str, url: str, **kwargs: Any) -> ResponseLike: ...


class RunnerFailure(ValueError):
    def __init__(self, code: str, stage: str):
        super().__init__(code)
        self.code = code
        self.stage = stage


class LoadedBundle(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    root: Path
    public: contract.PublicInputBundle
    setup: PublicAuthorSetupBundle
    freeze: ExecutionFreeze
    documents: Mapping[str, str]
    author_setup_sha256: str
    execution_freeze_sha256: str


def _is_uuid4(value: object) -> bool:
    if not isinstance(value, str) or _UUID_TEXT.fullmatch(value) is None:
        return False
    try:
        parsed = UUID(value)
    except ValueError:
        return False
    return parsed.version == 4 and str(parsed) == value


def _has_reparse_component(path: Path) -> bool:
    current = path.absolute()
    existing: list[Path] = []
    while True:
        if current.exists() or current.is_symlink():
            existing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for item in existing:
        try:
            stat = item.lstat()
        except OSError:
            return True
        if item.is_symlink() or bool(getattr(stat, "st_file_attributes", 0) & 0x400):
            return True
    return False


def _local_path(path: Path, *, must_exist: bool) -> Path:
    text = str(path)
    if "://" in text or text.startswith(("\\\\", "//")):
        raise RunnerFailure("network_path_forbidden", "local_preflight")
    if not path.is_absolute():
        raise RunnerFailure("absolute_local_path_required", "local_preflight")
    if _has_reparse_component(path if must_exist else path.parent):
        raise RunnerFailure("reparse_path_forbidden", "local_preflight")
    try:
        return path.resolve(strict=must_exist)
    except (OSError, RuntimeError) as exc:
        raise RunnerFailure("local_path_unavailable", "local_preflight") from exc


def _read_bounded(path: Path, maximum: int) -> bytes:
    resolved = _local_path(path, must_exist=True)
    try:
        if not resolved.is_file() or resolved.stat().st_size > maximum:
            raise RunnerFailure("input_file_invalid_or_too_large", "local_preflight")
        return resolved.read_bytes()
    except OSError as exc:
        raise RunnerFailure("input_file_unreadable", "local_preflight") from exc


def _canonical_model_sha(model: BaseModel) -> str:
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_value_sha(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_runner_code_manifest() -> RunnerCodeManifest:
    """Return the explicit, ordered code closure used by the sealed runner."""

    return RunnerCodeManifest(
        files=tuple(
            RunnerCodeFile(
                path=relative,
                sha256=hashlib.sha256(
                    (ROOT / Path(*relative.split("/"))).read_bytes()
                ).hexdigest(),
            )
            for relative in RUNNER_CODE_PATHS
        )
    )


def current_runner_source_sha256() -> str:
    """Bind a run to the canonical runner code-manifest, not one file."""

    return _canonical_model_sha(current_runner_code_manifest())


def _project_id_sha(project_id: str) -> str:
    return hashlib.sha256(project_id.encode("utf-8")).hexdigest()


def _project_set_sha(receipts: Sequence[ProjectReceipt]) -> str:
    rows = sorted(
        (
            {"case_id": row.case_id, "project_id_sha256": row.project_id_sha256}
            for row in receipts
        ),
        key=lambda row: row["case_id"],
    )
    return _canonical_value_sha(rows)


def _validate_bundle_binding(
    public: contract.PublicInputBundle,
    setup: PublicAuthorSetupBundle,
    freeze: ExecutionFreeze,
) -> None:
    if not _is_uuid4(public.dataset_id):
        raise RunnerFailure("public_dataset_identity_not_neutral", "local_preflight")
    public_sha = contract.canonical_sha256(public)
    setup_sha = _canonical_model_sha(setup)
    if (
        setup.dataset_id != public.dataset_id
        or setup.public_input_sha256 != public_sha
        or freeze.dataset_id != public.dataset_id
        or freeze.public_input_sha256 != public_sha
        or freeze.author_setup_sha256 != setup_sha
        or freeze.case_count != len(public.cases)
        or freeze.group_count != len({row.group_id for row in public.cases})
    ):
        raise RunnerFailure("frozen_control_binding_mismatch", "local_preflight")
    public_cases = {row.case_id: row for row in public.cases}
    setup_cases = {row.case_id: row for row in setup.cases}
    if len(setup_cases) != len(setup.cases) or set(setup_cases) != set(public_cases):
        raise RunnerFailure("setup_case_set_mismatch", "local_preflight")
    group_documents: dict[str, tuple[PublicDocumentSetup, ...]] = {}
    for case_id, public_case in public_cases.items():
        if (
            re.fullmatch(r"closed-(?:pilot|holdout)-v[1-9][0-9]*", public_case.split_id)
            is None
            or public_case.axis.version != 1
            or any(not _is_uuid4(item.evidence_id) for item in public_case.evidence_catalog)
        ):
            raise RunnerFailure("public_split_or_evidence_identity_invalid", "local_preflight")
        row = setup_cases[case_id]
        if row.group_id != public_case.group_id:
            raise RunnerFailure("setup_group_mismatch", "local_preflight")
        public_docs = {value.document_id: value for value in public_case.documents}
        setup_docs = {value.document_id: value for value in row.documents}
        if set(public_docs) != set(setup_docs):
            raise RunnerFailure("setup_document_set_mismatch", "local_preflight")
        prior = group_documents.setdefault(row.group_id, row.documents)
        if prior != row.documents:
            raise RunnerFailure("group_document_setup_changed", "local_preflight")
        source = setup_docs[row.candidate_selector.source_document_id]
        expected_origin = (
            "published_history" if source.document_role == "chapter" else "explicit_setting"
        )
        if row.candidate_selector.origin != expected_origin:
            raise RunnerFailure("selector_origin_lifecycle_mismatch", "local_preflight")
        axis_payload = {
            "candidate_selector": row.candidate_selector.model_dump(mode="json"),
            "author_axis": (
                row.author_axis.model_dump(mode="json")
                if row.author_axis is not None
                else None
            ),
        }
        if _canonical_value_sha(axis_payload) != public_case.axis.sha256:
            raise RunnerFailure("public_axis_binding_mismatch", "local_preflight")


def load_public_bundle(bundle_root: Path) -> LoadedBundle:
    root = _local_path(bundle_root, must_exist=True)
    if not root.is_dir():
        raise RunnerFailure("bundle_root_not_directory", "local_preflight")
    public = contract.PublicInputBundle.model_validate_json(
        _read_bounded(root / "public-input.json", MAX_CONTROL_BYTES)
    )
    setup = PublicAuthorSetupBundle.model_validate_json(
        _read_bounded(root / "author-setup.json", MAX_CONTROL_BYTES)
    )
    freeze = ExecutionFreeze.model_validate_json(
        _read_bounded(root / "freeze.json", MAX_CONTROL_BYTES)
    )
    _validate_bundle_binding(public, setup, freeze)

    documents: dict[str, str] = {}
    document_snapshots: dict[str, contract.DocumentSnapshot] = {}
    for case in public.cases:
        for document in case.documents:
            expected = f"cases/{case.group_id}/{document.document_id}.md"
            if (
                not _is_uuid4(case.case_id)
                or not _is_uuid4(case.group_id)
                or not _is_uuid4(case.world_id)
                or not _is_uuid4(case.axis.axis_id)
                or not _is_uuid4(document.document_id)
                or document.version != 1
                or document.path != expected
            ):
                raise RunnerFailure("public_identity_or_path_not_neutral", "local_preflight")
            if document.document_id in documents:
                if document_snapshots[document.document_id] != document:
                    raise RunnerFailure("shared_document_snapshot_changed", "local_preflight")
                continue
            file_path = _local_path(root / Path(*document.path.split("/")), must_exist=True)
            if not file_path.is_relative_to(root):
                raise RunnerFailure("document_path_escapes_bundle", "local_preflight")
            payload = _read_bounded(file_path, MAX_DOCUMENT_BYTES)
            if b"\r" in payload or hashlib.sha256(payload).hexdigest() != document.sha256:
                raise RunnerFailure("document_snapshot_mismatch", "local_preflight")
            try:
                text = payload.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise RunnerFailure("document_not_utf8", "local_preflight") from exc
            documents[document.document_id] = text
            document_snapshots[document.document_id] = document

        anchors = {
            (row.document_id, row.line_start, row.line_end): row.evidence_id
            for row in case.evidence_catalog
        }
        if len(anchors) != len(case.evidence_catalog):
            raise RunnerFailure("evidence_catalog_duplicate", "local_preflight")
        expected_anchors: set[tuple[str, int, int]] = set()
        for document in case.documents:
            for number, line in enumerate(documents[document.document_id].splitlines(), 1):
                if line.strip():
                    expected_anchors.add((document.document_id, number, number))
        if set(anchors) != expected_anchors:
            raise RunnerFailure("evidence_catalog_not_exhaustive", "local_preflight")
        setup_case = next(row for row in setup.cases if row.case_id == case.case_id)
        selector = setup_case.candidate_selector
        if any(
            (selector.source_document_id, line, line) not in anchors
            for line in range(selector.source_line_start, selector.source_line_end + 1)
        ):
            raise RunnerFailure("selector_not_bound_to_nonempty_lines", "local_preflight")

    return LoadedBundle(
        root=root,
        public=public,
        setup=setup,
        freeze=freeze,
        documents=documents,
        author_setup_sha256=_canonical_model_sha(setup),
        execution_freeze_sha256=_canonical_model_sha(freeze),
    )


def load_run_config(path: Path) -> SealedHttpRunConfig:
    config = SealedHttpRunConfig.model_validate_json(
        _read_bounded(path, MAX_CONTROL_BYTES)
    )
    if config.expected_runner_source_sha256 != current_runner_source_sha256():
        raise RunnerFailure("runner_source_hash_mismatch", "local_preflight")
    return config


def _request(
    client: TransportLike,
    method: str,
    path: str,
    stage: str,
    **kwargs: Any,
) -> Any:
    try:
        response = client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        raise RunnerFailure("http_transport_failure", stage) from exc
    if response.status_code >= 400:
        raise RunnerFailure("http_status_failure", stage)
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise RunnerFailure("http_response_too_large", stage)
    try:
        return response.json()
    except ValueError as exc:
        raise RunnerFailure("http_json_invalid", stage) from exc


def _verify_isolation(
    client: TransportLike,
    expected: IsolationExpectation,
    *,
    require_fresh: bool,
) -> None:
    value = _request(
        client,
        "GET",
        "/api/v1/evaluation/isolation-identity",
        "isolation_preflight",
    )
    expected_payload = expected.model_dump(mode="json")
    if (
        type(value) is not dict
        or set(value) != set(expected_payload) | {"fresh_for_prepare"}
        or any(value.get(key) != item for key, item in expected_payload.items())
        or type(value.get("fresh_for_prepare")) is not bool
        or (require_fresh and value["fresh_for_prepare"] is not True)
    ):
        raise RunnerFailure("evaluation_isolation_mismatch", "isolation_preflight")


def _runtime_preflight(
    client: TransportLike, *, expected_digest: str | None = None
) -> str:
    health = _request(client, "GET", "/health", "runtime_preflight")
    if type(health) is not dict or type(health.get("runtime_provenance")) is not dict:
        raise RunnerFailure("runtime_provenance_unavailable", "runtime_preflight")
    model = health.get("model")
    provenance = health["runtime_provenance"]
    capabilities = provenance.get("capabilities")
    limits = provenance.get("character_consistency_limits")
    if (
        type(model) is not dict
        or model.get("configured") is not True
        or type(capabilities) is not dict
        or capabilities.get("character_consistency") is not True
        or type(limits) is not dict
        or limits.get("signal_support_id_v4") is not True
        or limits.get("signal_full_line_echo_v2") is not True
    ):
        raise RunnerFailure("sealed_runtime_capability_missing", "runtime_preflight")
    digest = _canonical_value_sha(provenance)
    if expected_digest is not None and digest != expected_digest:
        raise RunnerFailure("runtime_provenance_expectation_mismatch", "runtime_preflight")
    return digest


def _check_runtime(client: TransportLike, expected_digest: str, diagnostics: object) -> None:
    if (
        type(diagnostics) is not dict
        or type(diagnostics.get("runtime_provenance")) is not dict
        or _canonical_value_sha(diagnostics["runtime_provenance"]) != expected_digest
    ):
        raise RunnerFailure("worker_runtime_provenance_mismatch", "runtime_postrun")
    if _runtime_preflight(client, expected_digest=expected_digest) != expected_digest:
        raise RunnerFailure("api_runtime_provenance_changed", "runtime_postrun")


def _wait_run(
    client: TransportLike,
    run_id: str,
    config: SealedHttpRunConfig,
    *,
    sleeper: Callable[[float], None],
    monotonic: Callable[[], float],
) -> dict[str, Any]:
    started = monotonic()
    while True:
        value = _request(
            client, "GET", f"/api/v1/analysis-runs/{quote(run_id, safe='')}", "run_status"
        )
        if type(value) is not dict or value.get("id") != run_id:
            raise RunnerFailure("run_status_contract", "run_status")
        if value.get("status") in TERMINAL_RUN_STATES:
            return value
        if monotonic() - started >= config.run_timeout_seconds:
            raise RunnerFailure("run_timeout", "run_status")
        sleeper(config.poll_interval_seconds)


def _start_run(
    client: TransportLike,
    project_id: str,
    config: SealedHttpRunConfig,
    *,
    mode: Literal["baseline_build", "draft_review"],
    target_document_id: str | None = None,
) -> str:
    body: dict[str, Any] = {"mode": mode, "sensitivity": config.sensitivity}
    if target_document_id is not None:
        body["target_document_ids"] = [target_document_id]
    value = _request(
        client,
        "POST",
        f"/api/v1/projects/{quote(project_id, safe='')}/analysis-runs",
        "start_run",
        headers={"Idempotency-Key": f"sealed-ooc-{uuid4().hex}"},
        json=body,
    )
    if type(value) is not dict or not _is_uuid4(value.get("id")):
        raise RunnerFailure("run_create_contract", "start_run")
    return value["id"]


def _upload_document(
    client: TransportLike,
    project_id: str,
    public_document: contract.DocumentSnapshot,
    setup: PublicDocumentSetup,
    content: str,
) -> str:
    resolution = "confirmed"  # A draft is an author-confirmed draft identity.
    scope_key = setup.story_scope
    value = _request(
        client,
        "POST",
        f"/api/v1/projects/{quote(project_id, safe='')}/documents/text",
        "document_upload",
        json={
            "name": f"{public_document.document_id}.md",
            "content": content,
            "document_role": setup.document_role,
            "story_scope": setup.story_scope,
            "narrative_context": {
                "resolution_state": resolution,
                "publication_status": setup.publication_status,
                "scope": {
                    "schema_version": 1,
                    "timeline_key": scope_key,
                    "release": {
                        "key": f"r{setup.import_order}",
                        "ordinal": setup.import_order,
                    },
                    "branch": {"path": ["main"]},
                },
            },
        },
    )
    if (
        type(value) is not dict
        or not _is_uuid4(value.get("id"))
        or value.get("project_id") != project_id
        or value.get("version") != 1
        or value.get("name") != f"{public_document.document_id}.md"
        or value.get("content") != content
        or value.get("document_role") != setup.document_role
        or value.get("story_scope") != setup.story_scope
    ):
        raise RunnerFailure("document_upload_contract", "document_upload")
    return value["id"]


def _run_stage_complete(run: object, diagnostics: object) -> bool:
    if type(run) is not dict or run.get("status") != "completed":
        return False
    if type(diagnostics) is not dict:
        return False
    stage = diagnostics.get("character_consistency")
    counts = stage.get("counts") if type(stage) is dict else None
    return bool(
        type(stage) is dict
        and stage.get("outcome") == "completed"
        and stage.get("material_coverage") == "complete"
        and type(counts) is dict
        and type(counts.get("planned_chunks")) is int
        and counts["planned_chunks"] > 0
        and counts.get("processed_chunks") == counts["planned_chunks"]
        and counts.get("model_incomplete_chunks", 0) == 0
        and counts.get("model_uncalled_chunks", 0) == 0
    )


def _list_pending(client: TransportLike, project_id: str) -> list[dict[str, Any]]:
    characters: list[str] = []
    page = 1
    while True:
        value = _request(
            client,
            "GET",
            f"/api/v1/projects/{quote(project_id, safe='')}/characters",
            "character_roster",
            params={"page": page, "page_size": 100},
        )
        if type(value) is not dict or type(value.get("items")) is not list:
            raise RunnerFailure("character_roster_contract", "candidate_inventory")
        for row in value["items"]:
            if type(row) is not dict or type(row.get("character_key")) is not str:
                raise RunnerFailure("character_roster_contract", "candidate_inventory")
            characters.append(row["character_key"])
        if value.get("has_more") is False:
            break
        if value.get("has_more") is not True or page >= 100:
            raise RunnerFailure("character_roster_pagination_invalid", "candidate_inventory")
        page += 1
    if len(characters) != len(set(characters)):
        raise RunnerFailure("character_roster_duplicate", "candidate_inventory")

    candidates: list[dict[str, Any]] = []
    for character in characters:
        offset = 0
        while True:
            value = _request(
                client,
                "GET",
                f"/api/v1/projects/{quote(project_id, safe='')}/characters/"
                f"{quote(character, safe='')}/profile-candidates",
                "candidate_list",
                params={"state": "pending", "limit": 200, "offset": offset},
            )
            if type(value) is not dict or type(value.get("items")) is not list:
                raise RunnerFailure("candidate_list_contract", "candidate_inventory")
            rows = value["items"]
            if any(type(row) is not dict for row in rows):
                raise RunnerFailure("candidate_list_contract", "candidate_inventory")
            candidates.extend(rows)
            if len(candidates) > MAX_PENDING_CANDIDATES:
                raise RunnerFailure("candidate_inventory_too_large", "candidate_inventory")
            if value.get("has_more") is False:
                break
            if value.get("has_more") is not True or not rows:
                raise RunnerFailure("candidate_pagination_invalid", "candidate_inventory")
            offset += len(rows)
    ids = [row.get("id") for row in candidates]
    if any(not _is_uuid4(value) for value in ids) or len(ids) != len(set(ids)):
        raise RunnerFailure("candidate_identity_invalid", "candidate_inventory")
    return candidates


def _candidate_metadata_matches(
    candidate: Mapping[str, Any],
    selector: PublicCandidateSelector,
    *,
    project_id: str,
    run_id: str,
) -> bool:
    return bool(
        candidate.get("project_id") == project_id
        and candidate.get("source_run_id") == run_id
        and candidate.get("review_state") == "pending"
        and candidate.get("reviewable") is True
        and candidate.get("source_verified") is True
        and candidate.get("support_bindings_status") == "verified"
        and candidate.get("character_key") == selector.character_key
        and candidate.get("trait_type") == selector.trait_type
        and candidate.get("origin")
        == ({"explicit_setting": "explicit_setting", "published_history": "history_inference"})[
            selector.origin
        ]
        and candidate.get("polarity") == selector.polarity
        and candidate.get("stability") == selector.stability
        and type(candidate.get("revision")) is int
        and candidate["revision"] >= 0
    )


def _candidate_exactly_matches(
    candidate: Mapping[str, Any],
    selector: PublicCandidateSelector,
    *,
    project_id: str,
    run_id: str,
    runtime_document_id: str,
    public_document: contract.DocumentSnapshot,
    source_content: str,
) -> bool:
    if not _candidate_metadata_matches(
        candidate, selector, project_id=project_id, run_id=run_id
    ):
        return False
    evidence = candidate.get("evidence")
    if type(evidence) is not list or len(evidence) != 1 or type(evidence[0]) is not dict:
        return False
    row = evidence[0]
    source_lines = source_content.splitlines()
    expected_text = "\n".join(
        source_lines[selector.source_line_start - 1 : selector.source_line_end]
    )
    if not (
        row.get("document_id") == runtime_document_id
        and row.get("document_name") == f"{public_document.document_id}.md"
        and row.get("document_version") == public_document.version
        and row.get("content_sha256") == public_document.sha256
        and row.get("line_start") == selector.source_line_start
        and row.get("line_end") == selector.source_line_end
        and row.get("text") in {expected_text, expected_text.strip()}
        and type(row.get("input_id")) is str
        and bool(row["input_id"])
    ):
        return False
    binding = candidate.get("support_bindings_v1")
    if (
        type(binding) is not dict
        or binding.get("schema_version") != "character-support-bindings-v1"
        or binding.get("index_version") != "assertion-index-v1"
        or type(binding.get("bindings")) is not list
        or len(binding["bindings"]) != 1
    ):
        return False
    binding_row = binding["bindings"][0]
    target = binding_row.get("target") if type(binding_row) is dict else None
    support_id = binding_row.get("support_id") if type(binding_row) is dict else None
    return bool(
        type(target) is dict
        and binding_row.get("evidence_index") == 0
        and type(support_id) is str
        and support_id.startswith(f"L{selector.source_line_start}:A")
        and target.get("support_id") == support_id
        and target.get("role") == "target"
        and type(target.get("start_offset")) is int
        and type(target.get("end_offset")) is int
        and 0 <= target["start_offset"] < target["end_offset"]
        and target["end_offset"] <= len(expected_text)
    )


def _decide_candidate(
    client: TransportLike,
    project_id: str,
    candidate: Mapping[str, Any],
    *,
    decision: Literal["confirm", "reject"],
    axis: Mapping[str, Any] | None = None,
    author_axis: AuthorAxis | None = None,
) -> dict[str, Any]:
    character = candidate.get("character_key")
    candidate_id = candidate.get("id")
    revision = candidate.get("revision")
    if type(character) is not str or not _is_uuid4(candidate_id) or type(revision) is not int:
        raise RunnerFailure("candidate_decision_identity_invalid", "candidate_decision")
    body: dict[str, Any] = {
        "decision": decision,
        "expected_revision": revision,
        "comment": "封闭评测按冻结作者配置执行；未选候选一律拒绝。",
    }
    if decision == "confirm" and axis is not None and author_axis is not None:
        body.update(
            {
                "approved_axis_id": axis["id"],
                "expected_axis_version": axis["version"],
                "axis_alignment": author_axis.axis_alignment,
                "expected_axis_positive_proposition_sha256": axis[
                    "positive_proposition_sha256"
                ],
            }
        )
        if candidate.get("trait_type") == "value":
            body.update(
                {
                    "expected_axis_applicability_scope_sha256": axis[
                        "applicability_scope_sha256"
                    ],
                    "scope_applicability_confirmed": True,
                }
            )
    value = _request(
        client,
        "POST",
        f"/api/v1/projects/{quote(project_id, safe='')}/characters/"
        f"{quote(character, safe='')}/profile-candidates/{quote(candidate_id, safe='')}/decisions",
        "candidate_decision",
        headers={"Idempotency-Key": f"sealed-ooc-decision-{uuid4().hex}"},
        json=body,
    )
    result = value.get("candidate") if type(value) is dict else None
    expected_state = "confirmed" if decision == "confirm" else "rejected"
    if (
        type(result) is not dict
        or value.get("deduplicated") is not False
        or not _is_uuid4(value.get("decision_id"))
        or result.get("id") != candidate_id
        or result.get("review_state") != expected_state
        or result.get("revision") != revision + 1
    ):
        raise RunnerFailure("candidate_decision_contract", "candidate_decision")
    return result


def _create_axis(
    client: TransportLike,
    project_id: str,
    candidate: Mapping[str, Any],
    author_axis: AuthorAxis,
) -> dict[str, Any]:
    trait_type = candidate.get("trait_type")
    if trait_type not in {"core_personality", "value"}:
        raise RunnerFailure("author_axis_dimension_unsupported", "axis_create")
    if any(
        len(value) > maximum
        for value, maximum in (
            (author_axis.display_name, 80),
            (author_axis.definition, 200),
            (author_axis.positive_proposition, 200),
            (author_axis.applicability_scope, 200),
        )
    ):
        raise RunnerFailure("author_axis_exceeds_api_limit", "axis_create")
    body: dict[str, Any] = {
        "trait_type": trait_type,
        "display_name": author_axis.display_name,
        "definition": author_axis.definition,
        "positive_proposition": author_axis.positive_proposition,
    }
    if trait_type == "value":
        comparison_key = candidate.get("comparison_key")
        if type(comparison_key) is not str or not comparison_key:
            raise RunnerFailure("candidate_comparison_key_missing", "axis_create")
        body.update(
            {
                "comparison_key": comparison_key,
                "applicability_scope": author_axis.applicability_scope,
            }
        )
    value = _request(
        client,
        "POST",
        f"/api/v1/projects/{quote(project_id, safe='')}/character-trait-axes",
        "axis_create",
        json=body,
    )
    if (
        type(value) is not dict
        or not _is_uuid4(value.get("id"))
        or value.get("project_id") != project_id
        or value.get("version") != 1
        or value.get("trait_type") != trait_type
        or value.get("definition_sha256")
        != hashlib.sha256(author_axis.definition.encode("utf-8")).hexdigest()
        or value.get("positive_proposition_sha256")
        != hashlib.sha256(author_axis.positive_proposition.encode("utf-8")).hexdigest()
    ):
        raise RunnerFailure("axis_create_contract", "axis_create")
    if trait_type == "value" and (
        value.get("comparison_key") != candidate.get("comparison_key")
        or value.get("applicability_scope_sha256")
        != hashlib.sha256(author_axis.applicability_scope.encode("utf-8")).hexdigest()
    ):
        raise RunnerFailure("axis_scope_binding_mismatch", "axis_create")
    return value


def _zero_funnel() -> contract.StageFunnel:
    return contract.StageFunnel(
        input_candidates=0,
        retrieved_candidates=0,
        grounded_candidates=0,
        adjudicated_candidates=0,
        surfaced_candidates=0,
    )


def _unassessed_snapshot(
    case_id: str,
    *,
    funnel: contract.StageFunnel | None = None,
    stage_status: Literal["partial", "failed"] = "failed",
) -> RuntimeCaseSnapshot:
    return RuntimeCaseSnapshot(
        case_id=case_id,
        stage_status=stage_status,
        material_coverage="unknown",
        explanation_coverage="unknown",
        report_collections_complete=False,
        citation_refs_complete=False,
        trace_count=0,
        final_outcome="unknown",
        formal_issue_count=0,
        review_clue_count=0,
        provisional_clue_count=0,
        citations=(),
        stage_funnel=funnel or _zero_funnel(),
    )


def _candidate_hash(candidate_id: str) -> str:
    return hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()


def _report_items(value: object, *, kind: Literal["formal", "review", "provisional"]):
    if kind == "formal":
        if type(value) is not list or len(value) > MAX_REPORT_ROWS:
            return None
        return value
    if type(value) is not dict or type(value.get("items")) is not list:
        return None
    rows = value["items"]
    if len(rows) > MAX_REPORT_ROWS or value.get("truncated") is not False:
        return None
    if kind == "review" and (
        value.get("unavailable_count") != 0 or value.get("scan_limited") is not False
    ):
        return None
    return rows


def _bound_report_rows(
    rows: Sequence[object],
    candidate_id: str,
    *,
    kind: Literal["formal", "review"],
) -> list[Mapping[str, Any]] | None:
    result: list[Mapping[str, Any]] = []
    for row in rows:
        if type(row) is not dict or row.get("category") != "character_drift":
            return None
        metadata = row.get("metadata")
        expected_class = "formal" if kind == "formal" else "review_clue"
        allowed_outcomes = {"conflict"} if kind == "formal" else {
            "needs_confirmation", "unverifiable"
        }
        if (
            row.get("report_class") != expected_class
            or type(metadata) is not dict
            or metadata.get("confirmed_candidate_id") != candidate_id
            or metadata.get("final_outcome") not in allowed_outcomes
        ):
            return None
        result.append(row)
    return result


def _runtime_ref_role_allowed(
    setup_case: PublicAuthorSetupCase,
    *,
    role: str,
    document_id: str,
    line_start: int,
    line_end: int,
) -> bool:
    selector = setup_case.candidate_selector
    if role == "B":
        return bool(
            document_id == selector.source_document_id
            and selector.source_line_start <= line_start
            and line_end <= selector.source_line_end
        )
    if role == "C":
        return document_id == setup_case.target_document_id
    if role == "G":
        return document_id in {
            row.document_id for row in setup_case.documents if row.phase == "baseline"
        }
    return role in {"X", "P"}


def _strict_runtime_refs(
    public_case: contract.PublicInputCase,
    setup_case: PublicAuthorSetupCase,
    trace: Mapping[str, Any],
) -> tuple[RuntimeEvidenceRef, ...] | None:
    raw = trace.get("citation_refs")
    if trace.get("citation_refs_incomplete") is not False or type(raw) is not list:
        return None
    if len(raw) > 64:
        return None
    by_name = {f"{row.document_id}.md": row.document_id for row in public_case.documents}
    anchors = {
        (row.document_id, row.line_start, row.line_end)
        for row in public_case.evidence_catalog
    }
    scoped = trace.get("scoped_axis_review")
    independent = type(scoped) is dict and scoped.get("independent_events") == "yes"
    reviewed_c = set()
    if type(scoped) is dict and type(scoped.get("observations")) is list:
        for row in scoped["observations"]:
            if (
                type(row) is not dict
                or row.get("object_match") != "same"
                or row.get("situation_match") != "same"
                or type(row.get("citation")) is not str
            ):
                continue
            reviewed_c.add(row["citation"])
    if trace.get("event_independence") == "yes":
        raw_independent = trace.get("independent_event_citations")
        if (
            type(raw_independent) is list
            and len(raw_independent) >= 2
            and len(raw_independent) == len(set(raw_independent))
            and all(
                type(value) is str and re.fullmatch(r"C[0-9]{2}", value)
                for value in raw_independent
            )
        ):
            independent = True
            reviewed_c.update(raw_independent)
    refs: list[RuntimeEvidenceRef] = []
    seen_handles: set[str] = set()
    for row in raw:
        if type(row) is not dict:
            return None
        handle = row.get("handle")
        role = row.get("role")
        name = row.get("document_name")
        start = row.get("line_start")
        end = row.get("line_end")
        document_id = by_name.get(name)
        if (
            type(handle) is not str
            or re.fullmatch(r"[BCGXP][0-9]{2}", handle) is None
            or handle in seen_handles
            or role not in {"B", "C", "G", "X", "P"}
            or handle[0] != role
            or document_id is None
            or type(start) is not int
            or type(end) is not int
            or start != end
            or (document_id, start, end) not in anchors
        ):
            return None
        seen_handles.add(handle)
        if not _runtime_ref_role_allowed(
            setup_case,
            role=role,
            document_id=document_id,
            line_start=start,
            line_end=end,
        ):
            return None
        event_group = None
        if role == "C" and independent and handle in reviewed_c:
            event_group = hashlib.sha256(
                f"{public_case.case_id}\0{handle}".encode("utf-8")
            ).hexdigest()
        refs.append(
            RuntimeEvidenceRef(
                role=role,
                document_id=document_id,
                line_start=start,
                line_end=end,
                event_group_id=event_group,
            )
        )
    return tuple(refs)


def _strict_report_evidence_refs(
    public_case: contract.PublicInputCase,
    setup_case: PublicAuthorSetupCase,
    row: Mapping[str, Any],
    *,
    allowed_roles: frozenset[str],
) -> tuple[RuntimeEvidenceRef, ...] | None:
    """Bind a surfaced row's concrete evidence to its reviewed role map."""

    evidence = row.get("evidence")
    metadata = row.get("metadata")
    if (
        type(evidence) is not list
        or not evidence
        or len(evidence) > 64
        or type(metadata) is not dict
        or metadata.get("evidence_binding") != "review_citations_v1"
    ):
        return None
    raw_bindings = metadata.get("review_citation_refs")
    if (
        type(raw_bindings) is not list
        or not raw_bindings
        or len(raw_bindings) != len(evidence)
        or len(raw_bindings) > 64
    ):
        return None

    by_name = {f"{item.document_id}.md": item.document_id for item in public_case.documents}
    anchors = {
        (item.document_id, item.line_start, item.line_end)
        for item in public_case.evidence_catalog
    }
    coordinates: list[tuple[str, int, int]] = []
    for item in evidence:
        if type(item) is not dict:
            return None
        document_id = by_name.get(item.get("document_name"))
        line_start = item.get("line_start")
        line_end = item.get("line_end")
        if (
            document_id is None
            or type(line_start) is not int
            or type(line_end) is not int
            or line_start != line_end
            or (document_id, line_start, line_end) not in anchors
        ):
            return None
        coordinates.append((document_id, line_start, line_end))
    if len(coordinates) != len(set(coordinates)):
        return None

    bindings: list[tuple[int, str, str, int]] = []
    seen_handles: set[str] = set()
    seen_evidence_indexes: set[int] = set()
    seen_response_indexes: set[int] = set()
    for binding in raw_bindings:
        if type(binding) is not dict:
            return None
        handle = binding.get("handle")
        role = binding.get("role")
        evidence_index = binding.get("evidence_index")
        response_index = binding.get("response_index")
        if (
            type(handle) is not str
            or re.fullmatch(r"[BCGXP][0-9]{2}", handle) is None
            or handle in seen_handles
            or role not in allowed_roles
            or handle[0] != role
            or type(evidence_index) is not int
            or not 0 <= evidence_index < len(coordinates)
            or evidence_index in seen_evidence_indexes
            or type(response_index) is not int
            or not 0 <= response_index < len(raw_bindings)
            or response_index in seen_response_indexes
        ):
            return None
        seen_handles.add(handle)
        seen_evidence_indexes.add(evidence_index)
        seen_response_indexes.add(response_index)
        bindings.append((response_index, handle, role, evidence_index))
    if (
        seen_evidence_indexes != set(range(len(evidence)))
        or seen_response_indexes != set(range(len(raw_bindings)))
    ):
        return None

    refs: list[RuntimeEvidenceRef] = []
    for _, _handle, role, evidence_index in sorted(bindings):
        document_id, line_start, line_end = coordinates[evidence_index]
        if not _runtime_ref_role_allowed(
            setup_case,
            role=role,
            document_id=document_id,
            line_start=line_start,
            line_end=line_end,
        ):
            return None
        refs.append(
            RuntimeEvidenceRef(
                role=role,
                document_id=document_id,
                line_start=line_start,
                line_end=line_end,
            )
        )
    return tuple(refs)


def _bind_report_refs_to_trace(
    report_refs: Sequence[RuntimeEvidenceRef],
    trace_refs: Sequence[RuntimeEvidenceRef],
) -> tuple[RuntimeEvidenceRef, ...] | None:
    """Cross-check exact role/coordinates, then copy only event identities."""

    def identity(ref: RuntimeEvidenceRef) -> tuple[str, str, int, int]:
        return (ref.role, ref.document_id, ref.line_start, ref.line_end)

    report_identities = tuple(identity(ref) for ref in report_refs)
    trace_identities = tuple(identity(ref) for ref in trace_refs)
    if (
        not report_identities
        or len(set(report_identities)) != len(report_identities)
        or len(set(trace_identities)) != len(trace_identities)
        or report_identities != trace_identities
    ):
        return None
    return tuple(
        report_ref.model_copy(
            update={"event_group_id": trace_ref.event_group_id}
        )
        for report_ref, trace_ref in zip(report_refs, trace_refs)
    )


def _snapshot_from_reports(
    public_case: contract.PublicInputCase,
    setup_case: PublicAuthorSetupCase,
    candidate_id: str,
    run: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    formal_raw: object,
    review_raw: object,
    provisional_raw: object,
    funnel: contract.StageFunnel,
) -> RuntimeCaseSnapshot:
    stage = diagnostics.get("character_consistency")
    if type(stage) is not dict:
        return _unassessed_snapshot(public_case.case_id, funnel=funnel)
    formal_items = _report_items(formal_raw, kind="formal")
    review_items = _report_items(review_raw, kind="review")
    provisional_items = _report_items(provisional_raw, kind="provisional")
    if formal_items is None or review_items is None or provisional_items is None:
        return _unassessed_snapshot(public_case.case_id, funnel=funnel, stage_status="partial")
    formal = _bound_report_rows(formal_items, candidate_id, kind="formal")
    review = _bound_report_rows(review_items, candidate_id, kind="review")
    if formal is None or review is None or any(type(row) is not dict for row in provisional_items):
        return _unassessed_snapshot(public_case.case_id, funnel=funnel, stage_status="partial")
    candidate_digest = _candidate_hash(candidate_id)
    all_traces = stage.get("case_trace")
    if type(all_traces) is not list or any(type(row) is not dict for row in all_traces):
        return _unassessed_snapshot(public_case.case_id, funnel=funnel, stage_status="partial")
    traces = [
        row
        for row in all_traces
        if row.get("confirmed_candidate_id_sha256") == candidate_digest
    ]
    if len(all_traces) != 1 or len(traces) != 1:
        return _unassessed_snapshot(public_case.case_id, funnel=funnel, stage_status="partial")
    trace = traces[0]
    refs = _strict_runtime_refs(public_case, setup_case, trace)
    if refs is None:
        return _unassessed_snapshot(public_case.case_id, funnel=funnel, stage_status="partial")
    final_outcome = trace.get("final_outcome")
    if final_outcome not in {
        "conflict", "needs_confirmation", "unverifiable", "no_issue", "unknown"
    }:
        final_outcome = "unknown"
    report_evidence: tuple[RuntimeEvidenceRef, ...] = ()
    reports_match = False
    metadata_matches = not (
        any(row["metadata"].get("final_outcome") != final_outcome for row in formal)
        or any(row["metadata"].get("final_outcome") != final_outcome for row in review)
    )
    surfaced_row: Mapping[str, Any] | None = None
    allowed_roles = frozenset[str]()
    if (
        final_outcome == "conflict"
        and len(formal) == 1
        and not review
        and not provisional_items
    ):
        surfaced_row = formal[0]
        allowed_roles = frozenset({"B", "C"})
    elif (
        final_outcome in {"needs_confirmation", "unverifiable"}
        and not formal
        and len(review) == 1
        and not provisional_items
    ):
        surfaced_row = review[0]
        allowed_roles = frozenset({"B", "C", "G", "X", "P"})
    elif (
        final_outcome == "no_issue"
        and not formal
        and not review
        and not provisional_items
    ):
        # A no-issue result has no surfaced row.  Its explanatory B/G/X refs
        # therefore remain trace-backed and are checked by the projector.
        reports_match = True

    if surfaced_row is not None and metadata_matches:
        raw_report_refs = _strict_report_evidence_refs(
            public_case,
            setup_case,
            surfaced_row,
            allowed_roles=allowed_roles,
        )
        if raw_report_refs is not None:
            verified_report_refs = _bind_report_refs_to_trace(raw_report_refs, refs)
            if verified_report_refs is not None:
                report_evidence = verified_report_refs
                reports_match = True
    if not metadata_matches:
        reports_match = False
    stage_complete = _run_stage_complete(run, diagnostics)
    explanation = trace.get("explanation_coverage")
    material = trace.get("material_coverage")
    stage_status: Literal["completed", "partial", "failed"] = (
        "completed" if stage_complete else "partial"
    )
    material_value = material if material in {"complete", "partial"} else "unknown"
    explanation_value = (
        explanation if explanation in {"complete", "partial"} else "unknown"
    )
    surfaced = len(formal) + len(review) + len(provisional_items)
    safe_funnel = funnel.model_copy(
        update={"surfaced_candidates": min(funnel.adjudicated_candidates, surfaced)}
    )
    return RuntimeCaseSnapshot(
        case_id=public_case.case_id,
        stage_status=stage_status,
        material_coverage=material_value,
        explanation_coverage=explanation_value,
        report_collections_complete=reports_match,
        citation_refs_complete=refs is not None,
        trace_count=len(traces),
        final_outcome=final_outcome,
        formal_issue_count=len(formal),
        review_clue_count=len(review),
        provisional_clue_count=len(provisional_items),
        citations=refs,
        report_evidence=report_evidence,
        stage_funnel=safe_funnel,
    )


def _create_case_project(
    client: TransportLike, public_case: contract.PublicInputCase
) -> str:
    project = _request(
        client,
        "POST",
        "/api/v1/projects",
        "project_create",
        json={
            "name": f"sealed-ooc-{public_case.case_id}-{uuid4().hex[:8]}",
            "description": "冻结公共输入的隔离 OOC 执行项目",
        },
    )
    if type(project) is not dict or not _is_uuid4(project.get("id")):
        raise RunnerFailure("project_create_contract", "project_create")
    return project["id"]


def _case_execution(
    client: TransportLike,
    bundle: LoadedBundle,
    public_case: contract.PublicInputCase,
    setup_case: PublicAuthorSetupCase,
    config: SealedHttpRunConfig,
    runtime_digest: str,
    project_id: str,
    *,
    sleeper: Callable[[float], None],
    monotonic: Callable[[], float],
) -> RuntimeCaseSnapshot:
    public_docs = {row.document_id: row for row in public_case.documents}
    runtime_docs: dict[str, str] = {}
    for setup_document in sorted(setup_case.documents, key=lambda row: row.import_order):
        if setup_document.phase != "baseline":
            continue
        public_document = public_docs[setup_document.document_id]
        runtime_docs[setup_document.document_id] = _upload_document(
            client,
            project_id,
            public_document,
            setup_document,
            bundle.documents[setup_document.document_id],
        )
    baseline_id = _start_run(client, project_id, config, mode="baseline_build")
    baseline_run = _wait_run(
        client, baseline_id, config, sleeper=sleeper, monotonic=monotonic
    )
    baseline_diagnostics = _request(
        client,
        "GET",
        f"/api/v1/analysis-runs/{quote(baseline_id, safe='')}/diagnostics",
        "baseline_diagnostics",
    )
    _check_runtime(client, runtime_digest, baseline_diagnostics)
    if not _run_stage_complete(baseline_run, baseline_diagnostics):
        return _unassessed_snapshot(public_case.case_id)

    candidates = _list_pending(client, project_id)
    selector = setup_case.candidate_selector
    source_public = public_docs[selector.source_document_id]
    retrieved = [
        row
        for row in candidates
        if _candidate_metadata_matches(
            row, selector, project_id=project_id, run_id=baseline_id
        )
    ]
    grounded = [
        row
        for row in retrieved
        if _candidate_exactly_matches(
            row,
            selector,
            project_id=project_id,
            run_id=baseline_id,
            runtime_document_id=runtime_docs[selector.source_document_id],
            public_document=source_public,
            source_content=bundle.documents[selector.source_document_id],
        )
    ]
    funnel = contract.StageFunnel(
        input_candidates=len(candidates),
        retrieved_candidates=len(retrieved),
        grounded_candidates=len(grounded),
        adjudicated_candidates=0,
        surfaced_candidates=0,
    )
    if len(grounded) != 1:
        for candidate in candidates:
            _decide_candidate(client, project_id, candidate, decision="reject")
        return _unassessed_snapshot(public_case.case_id, funnel=funnel)
    chosen = grounded[0]
    for candidate in candidates:
        if candidate["id"] != chosen["id"]:
            _decide_candidate(client, project_id, candidate, decision="reject")
    axis = (
        _create_axis(client, project_id, chosen, setup_case.author_axis)
        if setup_case.author_axis is not None
        else None
    )
    confirmed = _decide_candidate(
        client,
        project_id,
        chosen,
        decision="confirm",
        axis=axis,
        author_axis=setup_case.author_axis,
    )
    funnel = funnel.model_copy(update={"adjudicated_candidates": 1})

    target_setup = next(row for row in setup_case.documents if row.phase == "target")
    target_public = public_docs[target_setup.document_id]
    target_runtime_id = _upload_document(
        client,
        project_id,
        target_public,
        target_setup,
        bundle.documents[target_setup.document_id],
    )
    draft_id = _start_run(
        client,
        project_id,
        config,
        mode="draft_review",
        target_document_id=target_runtime_id,
    )
    draft_run = _wait_run(
        client, draft_id, config, sleeper=sleeper, monotonic=monotonic
    )
    diagnostics = _request(
        client,
        "GET",
        f"/api/v1/analysis-runs/{quote(draft_id, safe='')}/diagnostics",
        "draft_diagnostics",
    )
    _check_runtime(client, runtime_digest, diagnostics)
    formal = _request(
        client,
        "GET",
        f"/api/v1/analysis-runs/{quote(draft_id, safe='')}/issues",
        "formal_issues",
    )
    review = _request(
        client,
        "GET",
        f"/api/v1/analysis-runs/{quote(draft_id, safe='')}/review-clues",
        "review_clues",
    )
    provisional = _request(
        client,
        "GET",
        f"/api/v1/analysis-runs/{quote(draft_id, safe='')}/provisional-clues",
        "provisional_clues",
    )
    return _snapshot_from_reports(
        public_case,
        setup_case,
        confirmed["id"],
        draft_run,
        diagnostics,
        formal,
        review,
        provisional,
        funnel,
    )


def execute_with_transport(
    bundle: LoadedBundle,
    config: SealedHttpRunConfig,
    client: TransportLike,
    *,
    sleeper: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> SealedHttpRunReport:
    runner_digest = current_runner_source_sha256()
    if config.expected_runner_source_sha256 != runner_digest:
        raise RunnerFailure("runner_source_hash_mismatch", "local_preflight")
    _verify_isolation(client, config.isolation, require_fresh=True)
    runtime_digest = _runtime_preflight(
        client, expected_digest=config.expected_runtime_provenance_sha256
    )
    config_digest = _canonical_model_sha(config)
    setup_by_case = {row.case_id: row for row in bundle.setup.cases}
    snapshots: list[RuntimeCaseSnapshot] = []
    project_ids: set[str] = set()
    receipts: list[ProjectReceipt] = []
    failure_code: str | None = None
    for public_case in bundle.public.cases:
        project_id: str | None = None
        try:
            project_id = _create_case_project(client, public_case)
            if project_id in project_ids:
                raise RunnerFailure("project_not_independent", "case_execution")
            project_ids.add(project_id)
            receipts.append(
                ProjectReceipt(
                    case_id=public_case.case_id,
                    project_id_sha256=_project_id_sha(project_id),
                )
            )
            snapshot = _case_execution(
                client,
                bundle,
                public_case,
                setup_by_case[public_case.case_id],
                config,
                runtime_digest,
                project_id,
                sleeper=sleeper,
                monotonic=monotonic,
            )
            snapshots.append(snapshot)
            _verify_isolation(client, config.isolation, require_fresh=False)
        except Exception as exc:
            snapshots.append(_unassessed_snapshot(public_case.case_id))
            if failure_code is None:
                failure_code = (
                    exc.code
                    if isinstance(exc, RunnerFailure)
                    else "case_execution_failed"
                )
            if project_id is not None:
                try:
                    _verify_isolation(client, config.isolation, require_fresh=False)
                except Exception:
                    if failure_code is None:
                        failure_code = "evaluation_isolation_postcase_mismatch"

    artifact = build_prediction_artifact(
        bundle.public,
        snapshots,
        run_id=str(uuid4()),
        repeat_id=config.repeat_id,
        author_setup_sha256=bundle.author_setup_sha256,
        execution_freeze_sha256=bundle.execution_freeze_sha256,
        runtime_provenance_sha256=runtime_digest,
    )
    assessed = sum(row.outcome != "unassessed" for row in artifact.prediction.predictions)
    phase: Literal["completed", "partial"] = (
        "completed" if assessed == len(bundle.public.cases) else "partial"
    )
    return SealedHttpRunReport(
        phase=phase,
        dataset_id=bundle.public.dataset_id,
        public_input_sha256=contract.canonical_sha256(bundle.public),
        author_setup_sha256=bundle.author_setup_sha256,
        execution_freeze_sha256=bundle.execution_freeze_sha256,
        runtime_provenance_sha256=runtime_digest,
        run_config_sha256=config_digest,
        run_config=config,
        runner_source_sha256=runner_digest,
        execution_id=config.execution_id,
        project_set_sha256=_project_set_sha(receipts),
        project_receipts=tuple(receipts),
        independent_project_count=len(project_ids),
        assessed_case_count=assessed,
        unassessed_case_count=len(bundle.public.cases) - assessed,
        failure_code=failure_code,
        prediction_artifact=artifact,
    )


def _write_exclusive(path: Path, model: BaseModel) -> None:
    resolved_parent = _local_path(path.parent, must_exist=True)
    target = resolved_parent / path.name
    if target.exists() or target.is_symlink():
        raise RunnerFailure("output_already_exists", "output")
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2)
            + "\n"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--allow-provider-call", action="store_true")
    args = parser.parse_args(argv)
    if not args.preflight_only and not args.allow_provider_call:
        parser.error("live execution requires --allow-provider-call")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output_json)
    try:
        bundle = load_public_bundle(Path(args.bundle_root))
        config = load_run_config(Path(args.run_config))
        if args.preflight_only:
            receipts: tuple[ProjectReceipt, ...] = ()
            report = SealedHttpRunReport(
                phase="preflight",
                dataset_id=bundle.public.dataset_id,
                public_input_sha256=contract.canonical_sha256(bundle.public),
                author_setup_sha256=bundle.author_setup_sha256,
                execution_freeze_sha256=bundle.execution_freeze_sha256,
                run_config_sha256=_canonical_model_sha(config),
                run_config=config,
                runner_source_sha256=current_runner_source_sha256(),
                execution_id=config.execution_id,
                project_set_sha256=_project_set_sha(receipts),
                project_receipts=receipts,
                independent_project_count=0,
                assessed_case_count=0,
                unassessed_case_count=len(bundle.public.cases),
            )
        else:
            with httpx.Client(
                base_url=config.base_url.rstrip("/"),
                timeout=httpx.Timeout(30.0),
                trust_env=False,
            ) as client:
                report = execute_with_transport(bundle, config, client)
        _write_exclusive(output, report)
        return 0 if report.phase in {"preflight", "completed", "partial"} else 1
    except Exception as exc:
        failure = exc.code if isinstance(exc, RunnerFailure) else "sealed_runner_failed"
        report = SealedHttpRunReport(
            phase="failed",
            project_set_sha256=_project_set_sha(()),
            independent_project_count=0,
            assessed_case_count=0,
            unassessed_case_count=0,
            failure_code=failure,
        )
        try:
            _write_exclusive(output, report)
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExecutionFreeze",
    "LoadedBundle",
    "PublicAuthorSetupBundle",
    "RunnerCodeFile",
    "RunnerCodeManifest",
    "RunnerFailure",
    "SealedHttpRunConfig",
    "SealedHttpRunReport",
    "execute_with_transport",
    "current_runner_code_manifest",
    "current_runner_source_sha256",
    "load_public_bundle",
    "load_run_config",
    "parse_args",
]
