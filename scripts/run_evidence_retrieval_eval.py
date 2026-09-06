from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.db import DocumentRow, ProjectRow
from app.embeddings import EmbeddingProfile, OpenAICompatibleEmbeddingProvider
from app.evidence_chunks import EvidenceChunk, EvidenceChunker, SnapshotDocumentKey
from app.evidence_rag import (
    EvidenceDocument,
    EvidenceIndexCoordinator,
    EvidenceIndexDiagnostics,
    EvidenceIndexResult,
    EvidenceQuery,
    EvidenceRetrievalResult,
    EvidenceRrfRetriever,
)
from app.evidence_store import (
    SqlAlchemyEvidenceEmbeddingIndex,
    store_chunks,
    store_embeddings,
)


DEFAULT_SUITE_ROOT = ROOT / "data" / "evidence-retrieval-v1"
PINNED_MANIFEST_SHA256 = "e7f7100116c52e55585a55258914bb22b26b97711e35c0a50d3dae4a8ebbbdca"
PINNED_FREEZE_SHA256 = "3b9b14df06b427318acec0a01d9f30e6225936ed3792799fcd63f3eae1e25e9d"
PREPARED_KEYS = {
    "schema_version",
    "dataset_id",
    "split",
    "manifest_sha256",
    "logical_profile",
    "documents",
    "tasks",
}
DOCUMENT_KEYS = {
    "world_id",
    "project_id",
    "document_id",
    "version",
    "role",
    "active",
    "content_sha256",
    "content",
}
TASK_KEYS = {"task_id", "split", "world_id", "query", "allowed_snapshots"}
SNAPSHOT_KEYS = {"project_id", "document_id", "version", "content_sha256"}
PROFILE_KEYS = {
    "profile_id",
    "model_identifier",
    "model_revision",
    "dimensions",
    "normalized",
}
PREDICTION_KEYS = {
    "schema_version",
    "dataset_id",
    "split",
    "prepared_sha256",
    "configuration",
    "profile",
    "index",
    "code_commit",
    "tasks",
}
PREDICTION_TASK_KEYS = {
    "task_id",
    "status",
    "failure_reason",
    "diagnostics",
    "latency_ms",
    "rankings",
}
RESULT_KEYS = {
    "chunk_id",
    "project_id",
    "document_id",
    "version",
    "content_sha256",
    "chunker_version",
    "line_start",
    "line_end",
    "logical_profile_id",
    "canonical_profile_id",
    "keyword_rank",
    "vector_rank",
    "rrf_score",
}
FORBIDDEN_ORACLE_KEYS = {
    "expected",
    "expected_evidence",
    "labels",
    "negative_candidates",
    "difficulty",
    "oracle",
}
STRATEGIES = {"keyword-only", "dense-only", "keyword+dense-rrf"}
TOP_K = 5
WARM_REPEATS = 3
ALL_EVIDENCE_MINIMUM = {"dev": 12 / 16, "holdout": 20 / 28}

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SafeId = Annotated[str, Field(min_length=1, max_length=100)]


class _StrictRunInput(BaseModel):
    """Closed schema for every value crossing the prepare -> run boundary."""

    model_config = ConfigDict(extra="forbid", strict=True)


class PreparedProfile(_StrictRunInput):
    profile_id: SafeId
    model_identifier: Annotated[str, Field(min_length=1, max_length=255)]
    model_revision: Annotated[str, Field(min_length=1, max_length=120)]
    dimensions: Annotated[int, Field(ge=1, le=16_000)]
    normalized: bool


class PreparedSnapshot(_StrictRunInput):
    project_id: Annotated[str, Field(min_length=1, max_length=36)]
    document_id: Annotated[str, Field(min_length=1, max_length=36)]
    version: Annotated[int, Field(ge=1)]
    content_sha256: Sha256


class PreparedDocument(PreparedSnapshot):
    world_id: SafeId
    role: Literal["canon", "chapter", "isolation_decoy"]
    active: bool
    content: str


class PreparedTask(_StrictRunInput):
    task_id: SafeId
    split: Literal["dev", "holdout"]
    world_id: SafeId
    query: Annotated[str, Field(min_length=1, max_length=4_000)]
    allowed_snapshots: list[PreparedSnapshot]


class PreparedBundle(_StrictRunInput):
    schema_version: Literal["1.0"]
    dataset_id: SafeId
    split: Literal["dev", "holdout"]
    manifest_sha256: Sha256
    logical_profile: PreparedProfile
    documents: list[PreparedDocument]
    tasks: list[PreparedTask]


class PredictionConfiguration(_StrictRunInput):
    strategy: Literal["keyword-only", "dense-only", "keyword+dense-rrf"]
    chunk_target_chars: Annotated[int, Field(ge=1)]
    chunk_min_chars: Annotated[int, Field(ge=1)]
    chunk_max_chars: Annotated[int, Field(ge=1)]
    chunk_overlap_chars: Annotated[int, Field(ge=0)]
    rrf_k: Annotated[int, Field(ge=1, le=1_000)]
    branch_limit: Annotated[int, Field(ge=1, le=50)]
    top_k: Literal[5]
    warm_repeats: Literal[3]

    @model_validator(mode="after")
    def validate_chunk_bounds(self) -> "PredictionConfiguration":
        if not (
            self.chunk_min_chars
            <= self.chunk_target_chars
            <= self.chunk_max_chars
            and self.chunk_overlap_chars < self.chunk_min_chars
        ):
            raise ValueError("prediction chunk bounds are invalid")
        return self


class PredictionProfile(_StrictRunInput):
    logical_profile_id: SafeId
    canonical_profile_id: Annotated[str, Field(pattern=r"^emb-[0-9a-f]{64}$")]
    provider_kind: Annotated[str, Field(min_length=1, max_length=40)]
    provider_namespace: Annotated[str, Field(min_length=1, max_length=80)]
    model_identifier: Annotated[str, Field(min_length=1, max_length=255)]
    model_revision: Annotated[str, Field(min_length=1, max_length=120)]
    deployment_fingerprint: Annotated[str, Field(min_length=1, max_length=160)]
    document_transform_identity: Annotated[str, Field(min_length=1, max_length=80)]
    query_transform_identity: Annotated[str, Field(min_length=1, max_length=80)]
    dimensions: Annotated[int, Field(ge=1, le=16_000)]
    normalized: bool


class IndexStats(_StrictRunInput):
    expected_chunks: Annotated[int, Field(ge=0)]
    reused_chunks: Annotated[int, Field(ge=0)]
    embedded_chunks: Annotated[int, Field(ge=0)]
    provider_calls: Annotated[int, Field(ge=0)]
    provider_input_chars: Annotated[int, Field(ge=0)]
    elapsed_ms: Annotated[int, Field(ge=0)]
    complete_documents: Annotated[int, Field(ge=0)]
    incomplete_documents: Annotated[int, Field(ge=0)]
    failure_reasons: dict[SafeId, Annotated[int, Field(ge=1)]]


class CorpusStats(_StrictRunInput):
    visible_documents: Annotated[int, Field(ge=1)]
    active_authorizable_documents: Annotated[int, Field(ge=1)]
    inactive_versions: Annotated[int, Field(ge=0)]
    decoy_documents: Annotated[int, Field(ge=0)]
    projects: Annotated[int, Field(ge=1)]


class ProviderStats(_StrictRunInput):
    used_for_index: bool
    cold_complete: bool
    reuse_complete: bool


class IsolationFixtureStats(_StrictRunInput):
    wrong_profile_vectors: Literal[1]
    wrong_profile_logical_id: Literal["distractor-wrong-profile-768-v0"]
    wrong_profile_canonical_id: Annotated[str, Field(pattern=r"^emb-[0-9a-f]{64}$")]
    wrong_profile_dimensions: Literal[768]
    wrong_profile_on_authorized_chunk: Literal[True]
    old_versions_and_decoys_loaded: Annotated[int, Field(ge=1)]


class IndexMetadata(_StrictRunInput):
    chunker_version: Annotated[str, Field(min_length=1, max_length=80)]
    cold: IndexStats
    reuse: IndexStats
    corpus: CorpusStats
    provider: ProviderStats
    isolation_fixtures: IsolationFixtureStats
    pg_storage_bytes: Annotated[int, Field(ge=0)]


class RetrievalDiagnostic(_StrictRunInput):
    strategy: Literal["keyword-only", "dense-only", "keyword+dense-rrf"]
    mode: Literal["hybrid", "lexical_only", "dense_only", "unavailable"]
    reason: Literal[
        "index_incomplete",
        "embedding_not_configured",
        "embedding_input_rejected",
        "embedding_response_invalid",
        "embedding_retry_exhausted",
        "embedding_provider_failed",
        "vector_search_unavailable",
        "vector_scope_invalid",
    ] | None
    profile_id: Annotated[str, Field(pattern=r"^emb-[0-9a-f]{64}$")]
    chunker_version: Annotated[str, Field(min_length=1, max_length=80)]
    candidate_count: Annotated[int, Field(ge=0)]
    keyword_hits: Annotated[int, Field(ge=0)]
    vector_hits: Annotated[int, Field(ge=0)]
    entity_hits: Annotated[int, Field(ge=0)]
    result_count: Annotated[int, Field(ge=0, le=50)]
    provider_calls: Annotated[int, Field(ge=0)]
    provider_input_chars: Annotated[int, Field(ge=0)]
    elapsed_ms: Annotated[int, Field(ge=0)]


class PredictionResult(_StrictRunInput):
    chunk_id: Annotated[str, Field(pattern=r"^chk-[0-9a-f]{64}$")]
    project_id: Annotated[str, Field(min_length=1, max_length=36)]
    document_id: Annotated[str, Field(min_length=1, max_length=36)]
    version: Annotated[int, Field(ge=1)]
    content_sha256: Sha256
    chunker_version: Annotated[str, Field(min_length=1, max_length=80)]
    line_start: Annotated[int, Field(ge=1)]
    line_end: Annotated[int, Field(ge=1)]
    logical_profile_id: SafeId
    canonical_profile_id: Annotated[str, Field(pattern=r"^emb-[0-9a-f]{64}$")]
    keyword_rank: Annotated[int, Field(ge=1, le=50)] | None
    vector_rank: Annotated[int, Field(ge=1, le=50)] | None
    rrf_score: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @model_validator(mode="after")
    def validate_lines(self) -> "PredictionResult":
        if self.line_end < self.line_start:
            raise ValueError("prediction line range is invalid")
        if self.keyword_rank is None and self.vector_rank is None:
            raise ValueError("prediction result has no retrieval rank")
        return self


FiniteLatency = Annotated[float, Field(ge=0, le=3_600_000, allow_inf_nan=False)]


class PredictionTask(_StrictRunInput):
    task_id: SafeId
    status: Literal["completed", "failed"]
    failure_reason: Literal["retrieval_execution_failed"] | None
    diagnostics: RetrievalDiagnostic | None
    latency_ms: list[FiniteLatency]
    rankings: list[list[PredictionResult]]

    @model_validator(mode="after")
    def validate_status_contract(self) -> "PredictionTask":
        if self.status == "completed":
            if (
                self.failure_reason is not None
                or self.diagnostics is None
                or len(self.latency_ms) != WARM_REPEATS
                or len(self.rankings) != WARM_REPEATS
                or any(len(ranking) > TOP_K for ranking in self.rankings)
            ):
                raise ValueError("completed prediction task is invalid")
        elif (
            self.failure_reason != "retrieval_execution_failed"
            or self.diagnostics is not None
            or self.latency_ms
            or self.rankings
        ):
            raise ValueError("failed prediction task is invalid")
        return self


class PredictionBundle(_StrictRunInput):
    schema_version: Literal["1.0"]
    dataset_id: SafeId
    split: Literal["dev", "holdout"]
    prepared_sha256: Sha256
    configuration: PredictionConfiguration
    profile: PredictionProfile
    index: IndexMetadata
    code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{7,64}$")]
    tasks: list[PredictionTask]


Rate = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class LatencyMetrics(_StrictRunInput):
    samples: Annotated[int, Field(ge=0)]
    p50: FiniteLatency
    p95: FiniteLatency
    maximum: FiniteLatency


class ScoreMetrics(_StrictRunInput):
    task_count: Annotated[int, Field(ge=1)]
    evidence_recall_at_5: Rate
    all_evidence_at_5: Rate
    mrr: Rate
    low_lexical_evidence_recall_at_5: Rate
    isolation_failures: Annotated[int, Field(ge=0)]
    isolation_failure_tasks: Annotated[int, Field(ge=0)]
    fallback_tasks: Annotated[int, Field(ge=0)]
    failed_tasks: Annotated[int, Field(ge=0)]
    failure_rate: Rate
    unstable_warm_rankings: Annotated[int, Field(ge=0)]
    latency_ms: LatencyMetrics


class LabelMetrics(_StrictRunInput):
    task_count: Annotated[int, Field(ge=1)]
    evidence_recall_at_5: Rate
    all_evidence_at_5: Rate
    mrr: Rate


class TaskScore(_StrictRunInput):
    task_id: SafeId
    labels: list[SafeId]
    expected: Annotated[int, Field(ge=1)]
    recalled: Annotated[int, Field(ge=0)]
    all_evidence: bool
    reciprocal_rank: Rate
    low_lexical_overlap: bool
    status: Literal["completed", "failed"]
    fallback: bool
    unstable: bool
    isolation_failures: Annotated[int, Field(ge=0)]
    latency_ms: list[FiniteLatency]

    @model_validator(mode="after")
    def validate_task_metrics(self) -> "TaskScore":
        if (
            not self.labels
            or len(set(self.labels)) != len(self.labels)
            or self.recalled > self.expected
            or self.all_evidence is not (self.recalled == self.expected)
            or self.low_lexical_overlap is not ("low_lexical_overlap" in self.labels)
            or (self.recalled == 0) is not (self.reciprocal_rank == 0)
            or (self.status == "completed" and len(self.latency_ms) != WARM_REPEATS)
            or (
                self.status == "failed"
                and (
                    self.latency_ms
                    or self.recalled != 0
                    or self.reciprocal_rank != 0
                    or self.fallback
                    or self.unstable
                    or self.isolation_failures != 0
                )
            )
        ):
            raise ValueError("task score metrics are inconsistent")
        return self


class ScoreReport(_StrictRunInput):
    schema_version: Literal["1.0"]
    dataset_id: SafeId
    split: Literal["dev", "holdout"]
    claim_scope: Literal["developer-visible frozen split; not a blind or open-text result"]
    prediction_sha256: Sha256
    prepared_sha256: Sha256
    manifest_sha256: Literal[
        "e7f7100116c52e55585a55258914bb22b26b97711e35c0a50d3dae4a8ebbbdca"
    ]
    code_commit: Annotated[str, Field(pattern=r"^[0-9a-f]{7,64}$")]
    configuration: PredictionConfiguration
    profile: PredictionProfile
    index: IndexMetadata
    valid_run: bool
    metrics: ScoreMetrics
    per_label: dict[SafeId, LabelMetrics]
    tasks: list[TaskScore]


class RetrievalRunnerError(RuntimeError):
    pass


@dataclass(frozen=True)
class RunnerConfiguration:
    strategy: str
    chunk_target_chars: int = 450
    chunk_min_chars: int = 300
    chunk_max_chars: int = 600
    chunk_overlap_chars: int = 80
    rrf_k: int = 60
    branch_limit: int = 30
    top_k: int = TOP_K
    warm_repeats: int = WARM_REPEATS

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError("runner strategy is invalid")
        EvidenceChunker(
            target_chars=self.chunk_target_chars,
            min_chars=self.chunk_min_chars,
            max_chars=self.chunk_max_chars,
            overlap_chars=self.chunk_overlap_chars,
        )
        if type(self.rrf_k) is not int or not (1 <= self.rrf_k <= 1_000):
            raise ValueError("runner RRF constant is invalid")
        if type(self.branch_limit) is not int or not (1 <= self.branch_limit <= 50):
            raise ValueError("runner branch limit is invalid")
        if self.top_k != TOP_K or self.warm_repeats != WARM_REPEATS:
            raise ValueError("runner top-k and warm-repeat contract is fixed")

    def safe_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "chunk_target_chars": self.chunk_target_chars,
            "chunk_min_chars": self.chunk_min_chars,
            "chunk_max_chars": self.chunk_max_chars,
            "chunk_overlap_chars": self.chunk_overlap_chars,
            "rrf_k": self.rrf_k,
            "branch_limit": self.branch_limit,
            "top_k": self.top_k,
            "warm_repeats": self.warm_repeats,
        }


@dataclass(frozen=True)
class RuntimeEmbeddingConfiguration:
    base_url: str = field(repr=False)
    model_identifier: str
    model_revision: str
    deployment_fingerprint: str
    dimensions: int
    api_key: str = field(default="", repr=False)
    provider_namespace: str = "evidence-retrieval-v1"
    allow_insecure_http: bool = False

    def provider(self) -> OpenAICompatibleEmbeddingProvider:
        settings = Settings(
            _env_file=None,
            enable_embeddings=True,
            embedding_api_key=self.api_key,
            embedding_base_url=self.base_url,
            embedding_model=self.model_identifier,
            embedding_model_revision=self.model_revision,
            embedding_deployment_fingerprint=self.deployment_fingerprint,
            embedding_dimensions=self.dimensions,
            embedding_profile_namespace=self.provider_namespace,
            embedding_allow_insecure_http=self.allow_insecure_http,
        )
        return OpenAICompatibleEmbeddingProvider(settings)


class RetrievalExecutor(Protocol):
    @property
    def metadata(self) -> dict[str, object]: ...

    def retrieve(self, task: dict, repeat_index: int) -> EvidenceRetrievalResult: ...


def prepare_suite(
    *, suite_root: Path = DEFAULT_SUITE_ROOT, split: str = "dev"
) -> dict[str, object]:
    if split not in {"dev", "holdout"}:
        raise RetrievalRunnerError("split is invalid")
    manifest_path = suite_root / "manifest.json"
    freeze_path = suite_root / "freeze.json"
    manifest_bytes = manifest_path.read_bytes()
    freeze_bytes = freeze_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    freeze_sha = hashlib.sha256(freeze_bytes).hexdigest()
    if manifest_sha != PINNED_MANIFEST_SHA256 or freeze_sha != PINNED_FREEZE_SHA256:
        raise RetrievalRunnerError("frozen evaluation suite does not match the pinned v1 hashes")
    manifest = _json_object(manifest_bytes, "manifest")
    freeze = _json_object(freeze_bytes, "freeze")
    if freeze.get("manifest", {}).get("sha256") != manifest_sha:
        raise RetrievalRunnerError("frozen manifest hash does not match")
    profiles = {
        row["profile_id"]: row for row in manifest.get("profiles", [])
    }
    logical_profile = profiles.get(manifest.get("target_profile_id"))
    if not isinstance(logical_profile, dict) or set(logical_profile) != PROFILE_KEYS:
        raise RetrievalRunnerError("logical embedding profile is invalid")
    frozen_documents = {
        row["path"]: row for row in freeze.get("frozen_documents", [])
    }
    selected_worlds = [
        world for world in manifest.get("worlds", []) if world.get("split") == split
    ]
    documents: list[dict[str, object]] = []
    allowed_by_world: dict[str, list[dict[str, object]]] = {}
    for world in selected_worlds:
        world_id = world["world_id"]
        allowed: list[dict[str, object]] = []
        for row in world["documents"]:
            relative = row["path"]
            path = (suite_root / relative).resolve()
            path.relative_to(suite_root.resolve())
            content_bytes = path.read_bytes()
            frozen = frozen_documents.get(relative)
            if (
                not isinstance(frozen, dict)
                or hashlib.sha256(content_bytes).hexdigest() != frozen.get("sha256")
                or len(content_bytes) != frozen.get("bytes")
            ):
                raise RetrievalRunnerError("frozen document integrity failed")
            content = content_bytes.decode("utf-8")
            content_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
            document = {
                "world_id": world_id,
                "project_id": world["project_id"],
                "document_id": row["document_id"],
                "version": row["version"],
                "role": row["role"],
                "active": row["active"],
                "content_sha256": content_sha,
                "content": content,
            }
            documents.append(document)
            if row["active"] and row["role"] in {"canon", "chapter"}:
                allowed.append(
                    {
                        "project_id": world["project_id"],
                        "document_id": row["document_id"],
                        "version": row["version"],
                        "content_sha256": content_sha,
                    }
                )
        allowed_by_world[world_id] = allowed
    _validate_manifest_negative_fixtures(
        manifest=manifest,
        documents=documents,
        split=split,
    )
    tasks = [
        {
            "task_id": task["task_id"],
            "split": task["split"],
            "world_id": task["world_id"],
            "query": task["query"],
            "allowed_snapshots": allowed_by_world[task["world_id"]],
        }
        for task in manifest.get("tasks", [])
        if task.get("split") == split
    ]
    prepared = {
        "schema_version": "1.0",
        "dataset_id": manifest["dataset_id"],
        "split": split,
        "manifest_sha256": manifest_sha,
        "logical_profile": logical_profile,
        "documents": documents,
        "tasks": tasks,
    }
    validate_prepared(prepared)
    return prepared


def validate_prepared(value: object) -> dict:
    _reject_oracle_keys(value)
    try:
        # All nested models use ``extra='forbid'`` and strict primitive types.
        # This is the only parser used by the execution phase.
        value = PreparedBundle.model_validate(value).model_dump(mode="python")
    except ValidationError:
        raise RetrievalRunnerError("prepared bundle schema is invalid") from None
    if value["schema_version"] != "1.0" or value["split"] not in {"dev", "holdout"}:
        raise RetrievalRunnerError("prepared bundle identity is invalid")
    if not _safe_id(value["dataset_id"], 100) or not _sha(value["manifest_sha256"]):
        raise RetrievalRunnerError("prepared bundle identity is invalid")
    profile = value["logical_profile"]
    if not isinstance(profile, dict) or set(profile) != PROFILE_KEYS:
        raise RetrievalRunnerError("prepared logical profile is invalid")
    documents = value["documents"]
    tasks = value["tasks"]
    if not isinstance(documents, list) or not isinstance(tasks, list):
        raise RetrievalRunnerError("prepared rows are invalid")
    document_keys: set[tuple[str, str, int, str]] = set()
    allowed_documents: dict[str, set[tuple[str, str, int, str]]] = {}
    for row in documents:
        if not isinstance(row, dict) or set(row) != DOCUMENT_KEYS:
            raise RetrievalRunnerError("prepared document schema is invalid")
        if (
            not _safe_id(row["world_id"], 100)
            or not _safe_id(row["project_id"], 36)
            or not _safe_id(row["document_id"], 36)
            or type(row["version"]) is not int
            or row["version"] < 1
            or row["role"] not in {"canon", "chapter", "isolation_decoy"}
            or type(row["active"]) is not bool
            or not _sha(row["content_sha256"])
            or not isinstance(row["content"], str)
            or hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
            != row["content_sha256"]
        ):
            raise RetrievalRunnerError("prepared document is invalid")
        identity = (
            row["project_id"], row["document_id"], row["version"], row["content_sha256"]
        )
        if identity in document_keys:
            raise RetrievalRunnerError("prepared documents contain duplicates")
        document_keys.add(identity)
        if row["active"] and row["role"] in {"canon", "chapter"}:
            allowed_documents.setdefault(row["world_id"], set()).add(identity)
    task_ids: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != TASK_KEYS:
            raise RetrievalRunnerError("prepared task schema is invalid")
        if (
            task["split"] != value["split"]
            or not _safe_id(task["task_id"], 100)
            or not _safe_id(task["world_id"], 100)
            or not isinstance(task["query"], str)
            or not task["query"].strip()
            or task["task_id"] in task_ids
            or not isinstance(task["allowed_snapshots"], list)
        ):
            raise RetrievalRunnerError("prepared task is invalid")
        task_ids.add(task["task_id"])
        allowed: set[tuple[str, str, int, str]] = set()
        for snapshot in task["allowed_snapshots"]:
            if not isinstance(snapshot, dict) or set(snapshot) != SNAPSHOT_KEYS:
                raise RetrievalRunnerError("prepared snapshot schema is invalid")
            identity = (
                snapshot["project_id"],
                snapshot["document_id"],
                snapshot["version"],
                snapshot["content_sha256"],
            )
            if identity in allowed:
                raise RetrievalRunnerError("prepared snapshot scope has duplicates")
            allowed.add(identity)
        if allowed != allowed_documents.get(task["world_id"], set()):
            raise RetrievalRunnerError("prepared snapshot scope is not the active corpus")
    return value


def _assert_canonical_prepared(
    prepared: dict, *, serialized: bytes | None = None
) -> None:
    canonical = prepare_suite(split=prepared["split"])
    canonical_bytes = _canonical_bytes(canonical)
    if _canonical_bytes(prepared) != canonical_bytes:
        raise RetrievalRunnerError("prepared bundle is not the complete pinned split")
    if serialized is not None and serialized != canonical_bytes:
        raise RetrievalRunnerError("prepared bundle bytes are not canonical")


def _validate_manifest_negative_fixtures(
    *, manifest: dict, documents: list[dict[str, object]], split: str
) -> None:
    profiles = {
        profile.get("profile_id"): profile for profile in manifest.get("profiles", [])
        if isinstance(profile, dict)
    }
    target_profile_id = manifest.get("target_profile_id")
    wrong_profile = profiles.get("distractor-wrong-profile-768-v0")
    if (
        not isinstance(wrong_profile, dict)
        or wrong_profile.get("dimensions") != 768
        or wrong_profile.get("normalized") is not True
    ):
        raise RetrievalRunnerError("frozen wrong-profile fixture is invalid")
    by_identity = {
        (row["project_id"], row["document_id"], row["version"]): row
        for row in documents
    }
    worlds = {
        world.get("world_id"): world for world in manifest.get("worlds", [])
        if isinstance(world, dict) and world.get("split") == split
    }
    isolation_tasks = 0
    required_kinds = {
        "stale_version",
        "wrong_profile",
        "wrong_document",
        "wrong_project",
    }
    for task in manifest.get("tasks", []):
        if not isinstance(task, dict) or task.get("split") != split:
            continue
        candidates = task.get("negative_candidates")
        if candidates is None:
            continue
        isolation_tasks += 1
        if (
            not isinstance(candidates, list)
            or len(candidates) != len(required_kinds)
            or {candidate.get("kind") for candidate in candidates if isinstance(candidate, dict)}
            != required_kinds
        ):
            raise RetrievalRunnerError("frozen isolation candidates are incomplete")
        world = worlds.get(task.get("world_id"))
        if not isinstance(world, dict):
            raise RetrievalRunnerError("frozen isolation task world is invalid")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise RetrievalRunnerError("frozen isolation candidate is invalid")
            kind = candidate["kind"]
            row = by_identity.get(
                (
                    candidate.get("project_id"),
                    candidate.get("document_id"),
                    candidate.get("version"),
                )
            )
            if row is None:
                raise RetrievalRunnerError("frozen isolation candidate was not prepared")
            start = candidate.get("start_line")
            end = candidate.get("end_line")
            if (
                type(start) is not int
                or type(end) is not int
                or start < 1
                or end < start
                or end > len(str(row["content"]).splitlines())
            ):
                raise RetrievalRunnerError("frozen isolation candidate range is invalid")
            if kind == "stale_version":
                valid = (
                    row["project_id"] == world.get("project_id")
                    and row["active"] is False
                    and candidate.get("profile_id") == target_profile_id
                )
            elif kind == "wrong_profile":
                valid = (
                    row["project_id"] == world.get("project_id")
                    and row["active"] is True
                    and row["role"] in {"canon", "chapter"}
                    and candidate.get("profile_id") == wrong_profile["profile_id"]
                )
            elif kind == "wrong_document":
                valid = (
                    row["project_id"] == world.get("project_id")
                    and row["role"] == "isolation_decoy"
                    and candidate.get("profile_id") == target_profile_id
                )
            else:
                valid = (
                    row["project_id"] != world.get("project_id")
                    and row["role"] == "isolation_decoy"
                    and candidate.get("profile_id") == target_profile_id
                )
            if not valid:
                raise RetrievalRunnerError("frozen isolation candidate semantics are invalid")
    expected_tasks = 1 if split == "dev" else 2
    if isolation_tasks != expected_tasks:
        raise RetrievalRunnerError("frozen isolation task count is invalid")


def run_prepared(
    prepared: dict,
    *,
    prepared_sha256: str,
    configuration: RunnerConfiguration,
    executor: RetrievalExecutor,
    canonical_profile: EmbeddingProfile,
    code_commit: str,
) -> dict[str, object]:
    prepared = validate_prepared(prepared)
    _assert_canonical_prepared(prepared)
    if not _sha(prepared_sha256) or not _safe_commit(code_commit):
        raise RetrievalRunnerError("run provenance is invalid")
    logical_profile_id = prepared["logical_profile"]["profile_id"]
    tasks: list[dict[str, object]] = []
    for task in prepared["tasks"]:
        rankings: list[list[dict[str, object]]] = []
        latencies: list[float] = []
        diagnostics: dict[str, object] | None = None
        status = "completed"
        failure_reason: str | None = None
        try:
            for repeat_index in range(configuration.warm_repeats):
                repeat_started = time.perf_counter()
                result = executor.retrieve(task, repeat_index)
                latencies.append(
                    round(max(0.0, (time.perf_counter() - repeat_started) * 1000), 3)
                )
                diagnostics = result.diagnostics.safe_dict()
                rankings.append(
                    [
                        _safe_result_row(
                            match.chunk,
                            match,
                            logical_profile_id=logical_profile_id,
                            canonical_profile_id=canonical_profile.profile_id,
                        )
                        for match in result.matches[: configuration.top_k]
                    ]
                )
        except Exception:
            status = "failed"
            failure_reason = "retrieval_execution_failed"
            rankings = []
            latencies = []
            diagnostics = None
        tasks.append(
            {
                "task_id": task["task_id"],
                "status": status,
                "failure_reason": failure_reason,
                "diagnostics": diagnostics,
                "latency_ms": latencies,
                "rankings": rankings,
            }
        )
    output = {
        "schema_version": "1.0",
        "dataset_id": prepared["dataset_id"],
        "split": prepared["split"],
        "prepared_sha256": prepared_sha256,
        "configuration": configuration.safe_dict(),
        "profile": {
            "logical_profile_id": logical_profile_id,
            "canonical_profile_id": canonical_profile.profile_id,
            "provider_kind": canonical_profile.provider_kind,
            "provider_namespace": canonical_profile.provider_namespace,
            "model_identifier": canonical_profile.model_identifier,
            "model_revision": canonical_profile.model_revision,
            "deployment_fingerprint": canonical_profile.deployment_fingerprint,
            "document_transform_identity": canonical_profile.document_transform_identity,
            "query_transform_identity": canonical_profile.query_transform_identity,
            "dimensions": canonical_profile.dimensions,
            "normalized": canonical_profile.normalized,
        },
        "index": executor.metadata,
        "code_commit": code_commit,
        "tasks": tasks,
    }
    validate_predictions(output, prepared)
    return output


def validate_predictions(value: object, prepared: dict) -> dict:
    _reject_oracle_keys(value)
    try:
        value = PredictionBundle.model_validate(value).model_dump(mode="python")
    except ValidationError:
        raise RetrievalRunnerError("prediction bundle schema is invalid") from None
    if value["schema_version"] != "1.0" or value["dataset_id"] != prepared["dataset_id"]:
        raise RetrievalRunnerError("prediction bundle identity is invalid")
    if value["split"] != prepared["split"]:
        raise RetrievalRunnerError("prediction bundle provenance is invalid")
    configuration = value["configuration"]
    profile = value["profile"]
    index = value["index"]
    logical = prepared["logical_profile"]
    if (
        profile["logical_profile_id"] != logical["profile_id"]
        or profile["model_identifier"] != logical["model_identifier"]
        or profile["model_revision"] != logical["model_revision"]
        or profile["dimensions"] != logical["dimensions"]
        or profile["normalized"] != logical["normalized"]
    ):
        raise RetrievalRunnerError("prediction profile does not match the logical profile")
    try:
        EmbeddingProfile(**{
            "profile_id": profile["canonical_profile_id"],
            "provider_kind": profile["provider_kind"],
            "provider_namespace": profile["provider_namespace"],
            "model_identifier": profile["model_identifier"],
            "model_revision": profile["model_revision"],
            "deployment_fingerprint": profile["deployment_fingerprint"],
            "document_transform_identity": profile["document_transform_identity"],
            "query_transform_identity": profile["query_transform_identity"],
            "dimensions": profile["dimensions"],
            "normalized": profile["normalized"],
        })
    except (TypeError, ValueError):
        raise RetrievalRunnerError("prediction canonical profile is invalid") from None
    expected_chunker = EvidenceChunker(
        target_chars=configuration["chunk_target_chars"],
        min_chars=configuration["chunk_min_chars"],
        max_chars=configuration["chunk_max_chars"],
        overlap_chars=configuration["chunk_overlap_chars"],
    ).version
    if index["chunker_version"] != expected_chunker:
        raise RetrievalRunnerError("prediction chunker identity is invalid")
    if index["corpus"]["visible_documents"] != len(prepared["documents"]):
        raise RetrievalRunnerError("prediction corpus count is invalid")
    uses_vector = configuration["strategy"] != "keyword-only"
    if index["provider"]["used_for_index"] is not uses_vector:
        raise RetrievalRunnerError("prediction provider mode is inconsistent")
    tasks = value["tasks"]
    if len(tasks) != len(prepared["tasks"]):
        raise RetrievalRunnerError("prediction task count is invalid")
    expected_ids = [task["task_id"] for task in prepared["tasks"]]
    for expected_id, row in zip(expected_ids, tasks, strict=True):
        if row["task_id"] != expected_id:
            raise RetrievalRunnerError("prediction task identity is invalid")
        if row["status"] == "completed":
            diagnostics = row["diagnostics"]
            if (
                diagnostics["strategy"] != configuration["strategy"]
                or diagnostics["profile_id"] != profile["canonical_profile_id"]
                or diagnostics["chunker_version"] != index["chunker_version"]
            ):
                raise RetrievalRunnerError("prediction diagnostics are inconsistent")
            for ranking in row["rankings"]:
                if diagnostics["result_count"] != len(ranking):
                    raise RetrievalRunnerError("prediction result count is inconsistent")
                chunk_ids: set[str] = set()
                for result in ranking:
                    if (
                        result["chunk_id"] in chunk_ids
                        or result["logical_profile_id"] != profile["logical_profile_id"]
                        or result["canonical_profile_id"] != profile["canonical_profile_id"]
                        or result["chunker_version"] != index["chunker_version"]
                        or (
                            configuration["strategy"] == "keyword-only"
                            and result["vector_rank"] is not None
                        )
                        or (
                            configuration["strategy"] == "dense-only"
                            and result["keyword_rank"] is not None
                        )
                    ):
                        raise RetrievalRunnerError("prediction result is inconsistent")
                    chunk_ids.add(result["chunk_id"])
    return value


def score_prediction_files(
    *,
    prepared_path: Path,
    predictions_path: Path,
    expected_prediction_sha256: str,
    manifest_path: Path,
) -> dict[str, object]:
    if not _sha(expected_prediction_sha256):
        raise RetrievalRunnerError("prediction hash is invalid")
    prediction_bytes = predictions_path.read_bytes()
    if hashlib.sha256(prediction_bytes).hexdigest() != expected_prediction_sha256:
        raise RetrievalRunnerError("prediction hash does not match")
    prepared_bytes = prepared_path.read_bytes()
    prepared = validate_prepared(_json_object(prepared_bytes, "prepared bundle"))
    _assert_canonical_prepared(prepared, serialized=prepared_bytes)
    predictions = validate_predictions(
        _json_object(prediction_bytes, "prediction bundle"), prepared
    )
    if predictions["prepared_sha256"] != hashlib.sha256(prepared_bytes).hexdigest():
        raise RetrievalRunnerError("prepared bundle hash does not match")

    # Oracle-bearing fields are loaded only after the immutable prediction
    # bytes have been written and verified above.
    manifest_bytes = manifest_path.read_bytes()
    if (
        hashlib.sha256(manifest_bytes).hexdigest() != PINNED_MANIFEST_SHA256
        or prepared["manifest_sha256"] != PINNED_MANIFEST_SHA256
    ):
        raise RetrievalRunnerError("scorer manifest hash does not match")
    manifest = _json_object(manifest_bytes, "scorer manifest")
    oracle_by_id = {
        task["task_id"]: task
        for task in manifest["tasks"]
        if task["split"] == prepared["split"]
    }
    prepared_by_id = {task["task_id"]: task for task in prepared["tasks"]}
    expected_total = 0
    expected_hit = 0
    all_evidence_hits = 0
    reciprocal_ranks: list[float] = []
    low_expected = 0
    low_hit = 0
    failures = 0
    fallbacks = 0
    unstable = 0
    isolation_failures = 0
    isolation_failure_tasks: set[str] = set()
    latency_values: list[float] = []
    task_scores: list[dict[str, object]] = []
    label_totals: dict[str, dict[str, float | int]] = {}
    vector_strategy = predictions["configuration"].get("strategy") in {
        "dense-only",
        "keyword+dense-rrf",
    }
    canonical_profile_id = predictions["profile"].get("canonical_profile_id")
    logical_profile_id = prepared["logical_profile"]["profile_id"]
    chunker_version = predictions["index"].get("chunker_version")
    authorized_chunks = _authorized_chunk_catalog(prepared, predictions)
    for row in predictions["tasks"]:
        task_id = row["task_id"]
        oracle = oracle_by_id.get(task_id)
        prepared_task = prepared_by_id[task_id]
        if not isinstance(oracle, dict):
            raise RetrievalRunnerError("scorer task is missing")
        task_fallback = False
        task_unstable = False
        task_isolation_failures = 0
        if row["status"] != "completed":
            failures += 1
            ranking: list[dict] = []
        else:
            ranking = row["rankings"][0]
            latency_values.extend(float(value) for value in row["latency_ms"])
            if any(candidate != ranking for candidate in row["rankings"][1:]):
                unstable += 1
                task_unstable = True
            diagnostics = row["diagnostics"]
            if vector_strategy and (
                diagnostics.get("reason") is not None
                or diagnostics.get("vector_hits", 0) < 1
            ):
                fallbacks += 1
                task_fallback = True
        allowed = {
            (
                item["project_id"],
                item["document_id"],
                item["version"],
                item["content_sha256"],
            )
            for item in prepared_task["allowed_snapshots"]
        }
        for result in ranking:
            identity = (
                result["project_id"],
                result["document_id"],
                result["version"],
                result["content_sha256"],
            )
            exact_chunk = authorized_chunks.get(result["chunk_id"])
            if (
                identity not in allowed
                or result["canonical_profile_id"] != canonical_profile_id
                or result["logical_profile_id"] != logical_profile_id
                or result["chunker_version"] != chunker_version
                or exact_chunk is None
                or exact_chunk != (
                    *identity,
                    result["chunker_version"],
                    result["line_start"],
                    result["line_end"],
                )
            ):
                isolation_failures += 1
                task_isolation_failures += 1
                isolation_failure_tasks.add(task_id)
        expected_rows = oracle["expected_evidence"]
        expected_total += len(expected_rows)
        hits = [
            any(_covers(result, expected) for result in ranking)
            for expected in expected_rows
        ]
        hit_count = sum(hits)
        expected_hit += hit_count
        if hits and all(hits):
            all_evidence_hits += 1
        first_rank = next(
            (
                rank
                for rank, result in enumerate(ranking, start=1)
                if any(_covers(result, expected) for expected in expected_rows)
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        is_low = "low_lexical_overlap" in oracle.get("labels", [])
        if is_low:
            low_expected += len(expected_rows)
            low_hit += hit_count
        for label in oracle.get("labels", []):
            if not _safe_id(label, 100):
                raise RetrievalRunnerError("scorer task label is invalid")
            totals = label_totals.setdefault(
                label,
                {
                    "task_count": 0,
                    "expected_evidence": 0,
                    "recalled_evidence": 0,
                    "all_evidence_tasks": 0,
                    "reciprocal_rank_sum": 0.0,
                },
            )
            totals["task_count"] += 1
            totals["expected_evidence"] += len(expected_rows)
            totals["recalled_evidence"] += hit_count
            totals["all_evidence_tasks"] += int(bool(hits and all(hits)))
            totals["reciprocal_rank_sum"] += 1.0 / first_rank if first_rank else 0.0
        task_scores.append(
            {
                "task_id": task_id,
                "labels": sorted(oracle.get("labels", [])),
                "expected": len(expected_rows),
                "recalled": hit_count,
                "all_evidence": bool(hits and all(hits)),
                "reciprocal_rank": round(1.0 / first_rank, 6) if first_rank else 0.0,
                "low_lexical_overlap": is_low,
                "status": row["status"],
                "fallback": task_fallback,
                "unstable": task_unstable,
                "isolation_failures": task_isolation_failures,
                "latency_ms": row["latency_ms"],
            }
        )
    aggregates = _aggregate_task_scores(task_scores)
    report = {
        "schema_version": "1.0",
        "dataset_id": prepared["dataset_id"],
        "split": prepared["split"],
        "claim_scope": "developer-visible frozen split; not a blind or open-text result",
        "prediction_sha256": expected_prediction_sha256,
        "prepared_sha256": predictions["prepared_sha256"],
        "manifest_sha256": prepared["manifest_sha256"],
        "code_commit": predictions["code_commit"],
        "configuration": predictions["configuration"],
        "profile": predictions["profile"],
        "index": predictions["index"],
        "valid_run": aggregates["valid_run"],
        "metrics": aggregates["metrics"],
        "per_label": aggregates["per_label"],
        "tasks": task_scores,
    }
    _reject_sensitive_report_keys(report)
    validate_score_report(report)
    return report


def _aggregate_task_scores(task_values: list[dict]) -> dict[str, object]:
    try:
        tasks = [TaskScore.model_validate(value).model_dump(mode="python") for value in task_values]
    except ValidationError:
        raise RetrievalRunnerError("task score schema is invalid") from None
    if not tasks or not any(task["low_lexical_overlap"] for task in tasks):
        raise RetrievalRunnerError("task score coverage is incomplete")
    expected_total = sum(task["expected"] for task in tasks)
    recalled_total = sum(task["recalled"] for task in tasks)
    low_tasks = [task for task in tasks if task["low_lexical_overlap"]]
    low_expected = sum(task["expected"] for task in low_tasks)
    low_recalled = sum(task["recalled"] for task in low_tasks)
    latency_values = [
        float(value) for task in tasks for value in task["latency_ms"]
    ]
    failures = sum(task["status"] == "failed" for task in tasks)
    fallbacks = sum(task["fallback"] for task in tasks)
    unstable = sum(task["unstable"] for task in tasks)
    isolation_failures = sum(task["isolation_failures"] for task in tasks)
    isolation_failure_tasks = sum(task["isolation_failures"] > 0 for task in tasks)
    label_totals: dict[str, dict[str, float | int]] = {}
    for task in tasks:
        for label in task["labels"]:
            totals = label_totals.setdefault(
                label,
                {
                    "task_count": 0,
                    "expected": 0,
                    "recalled": 0,
                    "all_evidence": 0,
                    "reciprocal_rank": 0.0,
                },
            )
            totals["task_count"] += 1
            totals["expected"] += task["expected"]
            totals["recalled"] += task["recalled"]
            totals["all_evidence"] += int(task["all_evidence"])
            totals["reciprocal_rank"] += task["reciprocal_rank"]
    per_label = {
        label: {
            "task_count": totals["task_count"],
            "evidence_recall_at_5": round(totals["recalled"] / totals["expected"], 6),
            "all_evidence_at_5": round(
                totals["all_evidence"] / totals["task_count"], 6
            ),
            "mrr": round(totals["reciprocal_rank"] / totals["task_count"], 6),
        }
        for label, totals in sorted(label_totals.items())
    }
    task_count = len(tasks)
    valid = not any((failures, fallbacks, unstable, isolation_failures))
    return {
        "valid_run": valid,
        "metrics": {
            "task_count": task_count,
            "evidence_recall_at_5": round(recalled_total / expected_total, 6),
            "all_evidence_at_5": round(
                sum(task["all_evidence"] for task in tasks) / task_count, 6
            ),
            "mrr": round(
                sum(task["reciprocal_rank"] for task in tasks) / task_count, 6
            ),
            "low_lexical_evidence_recall_at_5": round(
                low_recalled / low_expected, 6
            ),
            "isolation_failures": isolation_failures,
            "isolation_failure_tasks": isolation_failure_tasks,
            "fallback_tasks": fallbacks,
            "failed_tasks": failures,
            "failure_rate": round(failures / task_count, 6),
            "unstable_warm_rankings": unstable,
            "latency_ms": {
                "samples": len(latency_values),
                "p50": _percentile(latency_values, 50),
                "p95": _percentile(latency_values, 95),
                "maximum": round(max(latency_values), 3) if latency_values else 0.0,
            },
        },
        "per_label": per_label,
    }


def _expected_task_semantics(split: str) -> list[tuple[object, ...]]:
    manifest_bytes = (DEFAULT_SUITE_ROOT / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != PINNED_MANIFEST_SHA256:
        raise RetrievalRunnerError("pinned scorer manifest is unavailable")
    manifest = _json_object(manifest_bytes, "pinned scorer manifest")
    return [
        (
            task["task_id"],
            tuple(sorted(task["labels"])),
            len(task["expected_evidence"]),
            "low_lexical_overlap" in task["labels"],
        )
        for task in manifest["tasks"]
        if task["split"] == split
    ]


def validate_score_report(value: object) -> dict:
    _reject_sensitive_report_keys(value)
    try:
        report = ScoreReport.model_validate(value).model_dump(mode="python")
    except ValidationError:
        raise RetrievalRunnerError("score report schema is invalid") from None
    metrics = report["metrics"]
    aggregates = _aggregate_task_scores(report["tasks"])
    expected_prepared_sha = hashlib.sha256(
        _canonical_bytes(prepare_suite(split=report["split"]))
    ).hexdigest()
    semantics = [
        (
            task["task_id"],
            tuple(task["labels"]),
            task["expected"],
            task["low_lexical_overlap"],
        )
        for task in report["tasks"]
    ]
    uses_vector = report["configuration"]["strategy"] != "keyword-only"
    if (
        semantics != _expected_task_semantics(report["split"])
        or report["prepared_sha256"] != expected_prepared_sha
        or report["metrics"] != aggregates["metrics"]
        or report["per_label"] != aggregates["per_label"]
        or report["index"]["provider"]["used_for_index"] is not uses_vector
        or report["valid_run"] is not aggregates["valid_run"]
    ):
        raise RetrievalRunnerError("score report metrics are inconsistent")
    return report


def compare_score_reports(
    report_paths: list[Path], expected_report_sha256s: list[str]
) -> dict[str, object]:
    if len(report_paths) != 3 or len(expected_report_sha256s) != 3:
        raise RetrievalRunnerError("compare requires exactly three score reports")
    if any(not _sha(value) for value in expected_report_sha256s):
        raise RetrievalRunnerError("compare report hash is invalid")
    _assert_distinct_paths(*report_paths)
    reports: dict[str, dict] = {}
    report_hashes: dict[str, str] = {}
    for path, expected_sha in zip(report_paths, expected_report_sha256s, strict=True):
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_sha:
            raise RetrievalRunnerError("compare report hash does not match")
        report = validate_score_report(_json_object(payload, "score report"))
        strategy = report["configuration"]["strategy"]
        if strategy in reports:
            raise RetrievalRunnerError("compare received a duplicate retrieval strategy")
        reports[strategy] = report
        report_hashes[strategy] = expected_sha
    if set(reports) != STRATEGIES:
        raise RetrievalRunnerError("compare requires keyword, dense and RRF reports")
    keyword = reports["keyword-only"]
    dense = reports["dense-only"]
    hybrid = reports["keyword+dense-rrf"]
    for candidate in (dense, hybrid):
        if any(
            candidate[key] != keyword[key]
            for key in (
                "dataset_id",
                "split",
                "manifest_sha256",
                "prepared_sha256",
                "code_commit",
                "profile",
            )
        ):
            raise RetrievalRunnerError("score reports do not share one evaluation provenance")
        if candidate["index"]["chunker_version"] != keyword["index"]["chunker_version"]:
            raise RetrievalRunnerError("score reports do not share one chunker")
        baseline_configuration = dict(keyword["configuration"])
        candidate_configuration = dict(candidate["configuration"])
        baseline_configuration.pop("strategy")
        candidate_configuration.pop("strategy")
        if candidate_configuration != baseline_configuration:
            raise RetrievalRunnerError(
                "score report configurations differ beyond retrieval strategy"
            )

    metric_names = (
        "evidence_recall_at_5",
        "all_evidence_at_5",
        "mrr",
        "low_lexical_evidence_recall_at_5",
    )

    def deltas(left: dict, right: dict) -> dict[str, float]:
        return {
            metric: round(right["metrics"][metric] - left["metrics"][metric], 6)
            for metric in metric_names
        }

    criteria = {
        "all_three_runs_valid": all(report["valid_run"] for report in reports.values()),
        "hybrid_evidence_recall_at_5_at_least_0_80": (
            hybrid["metrics"]["evidence_recall_at_5"] >= 0.80
        ),
        "hybrid_all_evidence_at_5_meets_split_minimum": (
            hybrid["metrics"]["all_evidence_at_5"]
            >= ALL_EVIDENCE_MINIMUM[hybrid["split"]]
        ),
        "hybrid_mrr_at_least_0_80": hybrid["metrics"]["mrr"] >= 0.80,
        "hybrid_low_lexical_recall_at_5_at_least_0_80": (
            hybrid["metrics"]["low_lexical_evidence_recall_at_5"] >= 0.80
        ),
        "hybrid_warm_p95_at_most_1500ms": (
            hybrid["metrics"]["latency_ms"]["p95"] <= 1_500
        ),
        "hybrid_cold_index_at_most_120s": hybrid["index"]["cold"]["elapsed_ms"] <= 120_000,
        "hybrid_reuse_makes_no_embedding_calls": (
            hybrid["index"]["reuse"]["embedded_chunks"] == 0
            and hybrid["index"]["reuse"]["provider_calls"] == 0
        ),
        "hybrid_zero_isolation_failure": (
            hybrid["metrics"]["isolation_failures"] == 0
            and hybrid["metrics"]["isolation_failure_tasks"] == 0
        ),
        "hybrid_zero_fallback_failure_or_instability": all(
            hybrid["metrics"][key] == 0
            for key in ("fallback_tasks", "failed_tasks", "unstable_warm_rankings")
        ),
    }
    improvement_criteria = {
        "hybrid_recall_gain_vs_keyword_at_least_0_05": (
            round(
                hybrid["metrics"]["evidence_recall_at_5"]
                - keyword["metrics"]["evidence_recall_at_5"],
                6,
            )
            >= 0.05
        ),
        "hybrid_recall_gain_vs_dense_at_least_0_05": (
            round(
                hybrid["metrics"]["evidence_recall_at_5"]
                - dense["metrics"]["evidence_recall_at_5"],
                6,
            )
            >= 0.05
        ),
        "hybrid_all_evidence_not_below_keyword": (
            hybrid["metrics"]["all_evidence_at_5"]
            >= keyword["metrics"]["all_evidence_at_5"]
        ),
        "hybrid_all_evidence_not_below_dense": (
            hybrid["metrics"]["all_evidence_at_5"]
            >= dense["metrics"]["all_evidence_at_5"]
        ),
        "all_three_runs_valid": all(report["valid_run"] for report in reports.values()),
    }
    comparison = {
        "schema_version": "1.0",
        "dataset_id": keyword["dataset_id"],
        "split": keyword["split"],
        "manifest_sha256": keyword["manifest_sha256"],
        "prepared_sha256": keyword["prepared_sha256"],
        "code_commit": keyword["code_commit"],
        "canonical_profile_id": keyword["profile"]["canonical_profile_id"],
        "chunker_version": keyword["index"]["chunker_version"],
        "top_k": TOP_K,
        "warm_repeats": WARM_REPEATS,
        "report_sha256s": dict(sorted(report_hashes.items())),
        "strategies": {
            strategy: {
                metric: report["metrics"][metric] for metric in metric_names
            }
            for strategy, report in sorted(reports.items())
        },
        "deltas": {
            "keyword_to_dense": deltas(keyword, dense),
            "keyword_to_hybrid": deltas(keyword, hybrid),
            "dense_to_hybrid": deltas(dense, hybrid),
        },
        "gate": {"passed": all(criteria.values()), "criteria": criteria},
        "hybrid_improvement_claim": {
            "supported": all(improvement_criteria.values()),
            "criteria": improvement_criteria,
        },
    }
    _reject_sensitive_report_keys(comparison)
    return comparison


class ProductionRetrievalExecutor:
    def __init__(
        self,
        *,
        prepared: dict,
        configuration: RunnerConfiguration,
        embedding: RuntimeEmbeddingConfiguration,
        database_url: str,
    ) -> None:
        self._prepared = validate_prepared(prepared)
        self._configuration = configuration
        self._engine = create_engine(database_url)
        self._sessions = sessionmaker(bind=self._engine, expire_on_commit=False)
        with self._sessions() as session:
            revision = session.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        if revision != "0003_embedding_identity":
            raise RetrievalRunnerError("database schema is not at the required revision")
        self._provider = embedding.provider()
        self.profile = self._provider.profile
        logical = self._prepared["logical_profile"]
        if (
            self.profile.model_identifier != logical["model_identifier"]
            or self.profile.model_revision != logical["model_revision"]
            or self.profile.dimensions != logical["dimensions"]
            or self.profile.normalized != logical["normalized"]
        ):
            raise RetrievalRunnerError("runtime profile does not match logical profile")
        self._chunker = EvidenceChunker(
            target_chars=configuration.chunk_target_chars,
            min_chars=configuration.chunk_min_chars,
            max_chars=configuration.chunk_max_chars,
            overlap_chars=configuration.chunk_overlap_chars,
        )
        self._results: dict[SnapshotDocumentKey, EvidenceIndexResult] = {}
        self._metadata = self._load_corpus()
        self._retriever = EvidenceRrfRetriever(
            index=SqlAlchemyEvidenceEmbeddingIndex(self._sessions),
            provider=self._provider,
            rrf_k=configuration.rrf_k,
        )

    @property
    def metadata(self) -> dict[str, object]:
        return self._metadata

    def _load_corpus(self) -> dict[str, object]:
        documents = [_document_from_prepared(row) for row in self._prepared["documents"]]
        with self._sessions() as session:
            project_ids = sorted({document.snapshot.project_id for document in documents})
            if session.scalar(
                select(ProjectRow.id).where(ProjectRow.id.in_(project_ids)).limit(1)
            ) is not None:
                raise RetrievalRunnerError("evaluation database already contains suite projects")
            session.add_all(
                ProjectRow(id=project_id, name=project_id, description="")
                for project_id in project_ids
            )
            by_document: dict[tuple[str, str], dict] = {}
            for row in self._prepared["documents"]:
                key = (row["project_id"], row["document_id"])
                current = by_document.get(key)
                if current is None or row["active"]:
                    by_document[key] = row
            session.add_all(
                DocumentRow(
                    id=row["document_id"],
                    project_id=row["project_id"],
                    name=f"{row['document_id']}.md",
                    content=row["content"],
                    version=row["version"],
                    active=row["active"],
                )
                for row in by_document.values()
            )
            session.commit()
        cold = _empty_index_stats()
        reuse = _empty_index_stats()
        coordinator = EvidenceIndexCoordinator(
            session_factory=self._sessions,
            provider=self._provider,
            chunker=self._chunker,
        )
        uses_vector = self._configuration.strategy != "keyword-only"
        for evidence_document in documents:
            if uses_vector:
                result = coordinator.ensure_index([evidence_document])
                self._results[evidence_document.snapshot] = result
                _merge_index_stats(cold, result.diagnostics)
            else:
                chunks = self._chunker.chunk(
                    project_id=evidence_document.snapshot.project_id,
                    document_id=evidence_document.snapshot.document_id,
                    document_version=evidence_document.snapshot.document_version,
                    content=evidence_document.content,
                    content_sha256=evidence_document.snapshot.content_sha256,
                )
                with self._sessions() as session:
                    store_chunks(session, chunks)
                    session.commit()
                self._results[evidence_document.snapshot] = _synthetic_index_result(
                    profile=self.profile,
                    snapshot=evidence_document.snapshot,
                    chunks=chunks,
                    chunker_version=self._chunker.version,
                    vector_complete=False,
                )
                cold["expected_chunks"] += len(chunks)
                cold["embedded_chunks"] += 0
        if uses_vector:
            for evidence_document in documents:
                result = coordinator.ensure_index([evidence_document])
                _merge_index_stats(reuse, result.diagnostics)
        fixture_count = self._insert_wrong_profile_fixture()
        visible_rows = self._prepared["documents"]
        return {
            "chunker_version": self._chunker.version,
            "cold": cold,
            "reuse": reuse,
            "corpus": {
                "visible_documents": len(documents),
                "active_authorizable_documents": sum(
                    bool(row["active"] and row["role"] in {"canon", "chapter"})
                    for row in visible_rows
                ),
                "inactive_versions": sum(not row["active"] for row in visible_rows),
                "decoy_documents": sum(
                    row["role"] == "isolation_decoy" for row in visible_rows
                ),
                "projects": len({row["project_id"] for row in visible_rows}),
            },
            "provider": {
                "used_for_index": uses_vector,
                "cold_complete": cold["incomplete_documents"] == 0,
                "reuse_complete": reuse["incomplete_documents"] == 0,
            },
            "isolation_fixtures": {
                **fixture_count,
                "old_versions_and_decoys_loaded": sum(
                    (not row["active"]) or row["role"] == "isolation_decoy"
                    for row in visible_rows
                ),
            },
            "pg_storage_bytes": self._pg_storage_bytes(),
        }

    def _insert_wrong_profile_fixture(self) -> dict[str, object]:
        authorized = {
            _snapshot_from_prepared(snapshot)
            for task in self._prepared["tasks"]
            for snapshot in task["allowed_snapshots"]
        }
        first_chunk = next(
            (
                chunk
                for snapshot, result in self._results.items()
                if snapshot in authorized
                for chunk in result.chunks
            ),
            None,
        )
        if first_chunk is None:
            raise RetrievalRunnerError("authorized wrong-profile fixture chunk is unavailable")
        wrong_profile = EmbeddingProfile.openai_compatible(
            provider_namespace="evidence-retrieval-v1-fixture",
            model_identifier="fixture-only/wrong-space",
            model_revision="fixture-v0",
            deployment_fingerprint="frozen-wrong-profile-768-v0",
            document_transform_identity="fixture-vector-v0",
            query_transform_identity="fixture-vector-v0",
            dimensions=768,
        )
        vector = (1.0, *([0.0] * 767))
        with self._sessions() as session:
            store_embeddings(
                session,
                profile=wrong_profile,
                chunks=[first_chunk],
                vectors=[vector],
            )
            session.commit()
        return {
            "wrong_profile_vectors": 1,
            "wrong_profile_logical_id": "distractor-wrong-profile-768-v0",
            "wrong_profile_canonical_id": wrong_profile.profile_id,
            "wrong_profile_dimensions": wrong_profile.dimensions,
            "wrong_profile_on_authorized_chunk": first_chunk.snapshot in authorized,
        }

    def _pg_storage_bytes(self) -> int:
        with self._sessions() as session:
            value = session.execute(
                text(
                    "SELECT "
                    "pg_total_relation_size('embedding_profiles') + "
                    "pg_total_relation_size('evidence_chunks') + "
                    "pg_total_relation_size('evidence_embeddings')"
                )
            ).scalar_one()
        if type(value) is not int or value < 0:
            raise RetrievalRunnerError("PostgreSQL storage measurement is invalid")
        return value

    def retrieve(self, task: dict, repeat_index: int) -> EvidenceRetrievalResult:
        del repeat_index
        snapshots = tuple(_snapshot_from_prepared(row) for row in task["allowed_snapshots"])
        chunks = tuple(
            chunk for snapshot in snapshots for chunk in self._results[snapshot].chunks
        )
        complete = all(self._results[snapshot].complete for snapshot in snapshots)
        indexed = EvidenceIndexResult(
            profile=self.profile,
            snapshots=snapshots,
            chunks=chunks,
            diagnostics=EvidenceIndexDiagnostics(
                outcome="complete" if complete else "provider_unavailable",
                reason=None if complete else "embedding_not_configured",
                profile_id=self.profile.profile_id,
                chunker_version=self._chunker.version,
                document_count=len(snapshots),
                expected_chunks=len(chunks),
                reused_chunks=0,
                embedded_chunks=len(chunks) if complete else 0,
                provider_calls=0,
                provider_input_chars=0,
                elapsed_ms=0,
            ),
        )
        return self._retriever.retrieve(
            indexed=indexed,
            allowed_snapshots=snapshots,
            query=EvidenceQuery(text=task["query"], entity_terms=()),
            strategy=self._configuration.strategy,
            limit=TOP_K,
            branch_limit=self._configuration.branch_limit,
        )


def _synthetic_index_result(
    *,
    profile: EmbeddingProfile,
    snapshot: SnapshotDocumentKey,
    chunks: tuple[EvidenceChunk, ...],
    chunker_version: str,
    vector_complete: bool,
) -> EvidenceIndexResult:
    return EvidenceIndexResult(
        profile=profile,
        snapshots=(snapshot,),
        chunks=chunks,
        diagnostics=EvidenceIndexDiagnostics(
            outcome="complete" if vector_complete else "provider_unavailable",
            reason=None if vector_complete else "embedding_not_configured",
            profile_id=profile.profile_id,
            chunker_version=chunker_version,
            document_count=1,
            expected_chunks=len(chunks),
            reused_chunks=0,
            embedded_chunks=len(chunks) if vector_complete else 0,
            provider_calls=0,
            provider_input_chars=0,
            elapsed_ms=0,
        ),
    )


def _document_from_prepared(row: dict) -> EvidenceDocument:
    return EvidenceDocument(snapshot=_snapshot_from_prepared(row), content=row["content"])


def _snapshot_from_prepared(row: dict) -> SnapshotDocumentKey:
    return SnapshotDocumentKey(
        project_id=row["project_id"],
        document_id=row["document_id"],
        document_version=row["version"],
        content_sha256=row["content_sha256"],
    )


def _safe_result_row(
    chunk: EvidenceChunk,
    match,
    *,
    logical_profile_id: str,
    canonical_profile_id: str,
) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "project_id": chunk.snapshot.project_id,
        "document_id": chunk.snapshot.document_id,
        "version": chunk.snapshot.document_version,
        "content_sha256": chunk.snapshot.content_sha256,
        "chunker_version": chunk.chunker_version,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "logical_profile_id": logical_profile_id,
        "canonical_profile_id": canonical_profile_id,
        "keyword_rank": match.keyword_rank,
        "vector_rank": match.vector_rank,
        "rrf_score": round(float(match.rrf_score), 12),
    }


def _empty_index_stats() -> dict[str, object]:
    return {
        "expected_chunks": 0,
        "reused_chunks": 0,
        "embedded_chunks": 0,
        "provider_calls": 0,
        "provider_input_chars": 0,
        "elapsed_ms": 0,
        "complete_documents": 0,
        "incomplete_documents": 0,
        "failure_reasons": {},
    }


def _merge_index_stats(target: dict[str, object], diagnostic: EvidenceIndexDiagnostics) -> None:
    target["expected_chunks"] += diagnostic.expected_chunks
    target["reused_chunks"] += diagnostic.reused_chunks
    target["embedded_chunks"] += diagnostic.embedded_chunks
    target["provider_calls"] += diagnostic.provider_calls
    target["provider_input_chars"] += diagnostic.provider_input_chars
    target["elapsed_ms"] += diagnostic.elapsed_ms
    if diagnostic.outcome == "complete":
        target["complete_documents"] += 1
    else:
        target["incomplete_documents"] += 1
        reason = diagnostic.reason or "unknown"
        reasons = target["failure_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1


def _covers(result: dict, expected: dict) -> bool:
    return (
        result["document_id"] == expected["document_id"]
        and result["version"] == expected["version"]
        and result["line_start"] <= expected["start_line"]
        and result["line_end"] >= expected["end_line"]
    )


def _authorized_chunk_catalog(
    prepared: dict, predictions: dict
) -> dict[str, tuple[object, ...]]:
    configuration = predictions["configuration"]
    try:
        chunker = EvidenceChunker(
            target_chars=configuration["chunk_target_chars"],
            min_chars=configuration["chunk_min_chars"],
            max_chars=configuration["chunk_max_chars"],
            overlap_chars=configuration["chunk_overlap_chars"],
        )
    except (KeyError, TypeError, ValueError):
        raise RetrievalRunnerError("prediction chunker configuration is invalid") from None
    if predictions["index"].get("chunker_version") != chunker.version:
        raise RetrievalRunnerError("prediction chunker identity is invalid")
    allowed = {
        (
            snapshot["project_id"],
            snapshot["document_id"],
            snapshot["version"],
            snapshot["content_sha256"],
        )
        for task in prepared["tasks"]
        for snapshot in task["allowed_snapshots"]
    }
    catalog: dict[str, tuple[object, ...]] = {}
    for row in prepared["documents"]:
        identity = (
            row["project_id"],
            row["document_id"],
            row["version"],
            row["content_sha256"],
        )
        if identity not in allowed:
            continue
        chunks = chunker.chunk(
            project_id=row["project_id"],
            document_id=row["document_id"],
            document_version=row["version"],
            content=row["content"],
            content_sha256=row["content_sha256"],
        )
        for chunk in chunks:
            catalog[chunk.chunk_id] = (
                *identity,
                chunk.chunker_version,
                chunk.line_start,
                chunk.line_end,
            )
    return catalog


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 3)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction, 3)


def _reject_oracle_keys(value: object) -> None:
    if isinstance(value, dict):
        if any(key in FORBIDDEN_ORACLE_KEYS for key in value):
            raise RetrievalRunnerError("oracle-bearing field is forbidden during run")
        for child in value.values():
            _reject_oracle_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_oracle_keys(child)


def _reject_sensitive_report_keys(value: object) -> None:
    forbidden = {
        "api_key",
        "base_url",
        "endpoint",
        "content",
        "text",
        "prompt",
        "response",
        "raw_response",
        "expected_evidence",
        "negative_candidates",
    }
    if isinstance(value, dict):
        if any(key in forbidden for key in value):
            raise RetrievalRunnerError("report contains a forbidden field")
        for child in value.values():
            _reject_sensitive_report_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_report_keys(child)


def _json_object(payload: bytes, label: str) -> dict:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise RetrievalRunnerError(f"{label} JSON is invalid") from None
    if not isinstance(value, dict):
        raise RetrievalRunnerError(f"{label} must be an object")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> str:
    payload = _canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return hashlib.sha256(payload).hexdigest()


def _assert_distinct_paths(*paths: Path) -> None:
    normalized: list[tuple[Path, str]] = []
    for raw in paths:
        path = Path(raw)
        resolved = path.resolve(strict=False)
        folded = os.path.normcase(str(resolved))
        for prior_path, prior_folded in normalized:
            if folded == prior_folded:
                raise RetrievalRunnerError("evaluation input and output paths must be distinct")
            if path.exists() and prior_path.exists():
                try:
                    if os.path.samefile(path, prior_path):
                        raise RetrievalRunnerError(
                            "evaluation input and output paths must not alias"
                        )
                except OSError:
                    raise RetrievalRunnerError("evaluation path identity is unavailable") from None
        normalized.append((path, folded))


def _assert_clean_tracked_worktree() -> None:
    completed = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RetrievalRunnerError("tracked worktree state is unavailable")
    if completed.stdout.strip():
        raise RetrievalRunnerError(
            "tracked worktree must be clean before producing predictions"
        )


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _safe_id(value: object, limit: int) -> bool:
    return isinstance(value, str) and value == value.strip() and 0 < len(value) <= limit


def _safe_commit(value: object) -> bool:
    return isinstance(value, str) and 7 <= len(value) <= 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip().lower()
    if not _safe_commit(value):
        raise RetrievalRunnerError("git commit is unavailable")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen Evidence Retrieval v1 runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    prepare.add_argument("--split", choices=("dev", "holdout"), default="dev")
    prepare.add_argument("--output", type=Path, required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--prepared", type=Path, required=True)
    run.add_argument("--predictions", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    run.add_argument("--database-url", required=True)
    run.add_argument("--strategy", choices=sorted(STRATEGIES), required=True)
    run.add_argument("--embedding-base-url", required=True)
    run.add_argument("--embedding-model", required=True)
    run.add_argument("--embedding-revision", required=True)
    run.add_argument("--embedding-deployment-fingerprint", required=True)
    run.add_argument("--embedding-dimensions", type=int, required=True)
    run.add_argument("--embedding-provider-namespace", default="evidence-retrieval-v1")
    run.add_argument("--allow-insecure-http", action="store_true")
    run.add_argument("--chunk-target", type=int, default=450)
    run.add_argument("--chunk-min", type=int, default=300)
    run.add_argument("--chunk-max", type=int, default=600)
    run.add_argument("--chunk-overlap", type=int, default=80)
    run.add_argument("--rrf-k", type=int, default=60)
    run.add_argument("--branch-limit", type=int, default=30)

    score = subparsers.add_parser("score")
    score.add_argument("--prepared", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--prediction-sha256", required=True)
    score.add_argument("--manifest", type=Path, default=DEFAULT_SUITE_ROOT / "manifest.json")
    score.add_argument("--report", type=Path, required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--keyword-report", type=Path, required=True)
    compare.add_argument("--keyword-report-sha256", required=True)
    compare.add_argument("--dense-report", type=Path, required=True)
    compare.add_argument("--dense-report-sha256", required=True)
    compare.add_argument("--hybrid-report", type=Path, required=True)
    compare.add_argument("--hybrid-report-sha256", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        _assert_distinct_paths(
            args.output,
            args.suite_root / "manifest.json",
            args.suite_root / "freeze.json",
        )
        digest = _write_json(
            args.output, prepare_suite(suite_root=args.suite_root, split=args.split)
        )
        print(digest)
        return 0
    if args.command == "run":
        if not args.execute:
            raise RetrievalRunnerError("real retrieval requires explicit --execute")
        _assert_distinct_paths(args.prepared, args.predictions)
        _assert_clean_tracked_worktree()
        prepared_bytes = args.prepared.read_bytes()
        prepared = validate_prepared(_json_object(prepared_bytes, "prepared bundle"))
        _assert_canonical_prepared(prepared, serialized=prepared_bytes)
        configuration = RunnerConfiguration(
            strategy=args.strategy,
            chunk_target_chars=args.chunk_target,
            chunk_min_chars=args.chunk_min,
            chunk_max_chars=args.chunk_max,
            chunk_overlap_chars=args.chunk_overlap,
            rrf_k=args.rrf_k,
            branch_limit=args.branch_limit,
        )
        executor = ProductionRetrievalExecutor(
            prepared=prepared,
            configuration=configuration,
            embedding=RuntimeEmbeddingConfiguration(
                base_url=args.embedding_base_url,
                api_key=os.environ.get("LOREGUARD_EVAL_EMBEDDING_API_KEY", ""),
                model_identifier=args.embedding_model,
                model_revision=args.embedding_revision,
                deployment_fingerprint=args.embedding_deployment_fingerprint,
                dimensions=args.embedding_dimensions,
                provider_namespace=args.embedding_provider_namespace,
                allow_insecure_http=args.allow_insecure_http,
            ),
            database_url=args.database_url,
        )
        predictions = run_prepared(
            prepared,
            prepared_sha256=hashlib.sha256(prepared_bytes).hexdigest(),
            configuration=configuration,
            executor=executor,
            canonical_profile=executor.profile,
            code_commit=_git_commit(),
        )
        prediction_sha = _write_json(args.predictions, predictions)
        print(prediction_sha)
        return 0
    if args.command == "score":
        _assert_distinct_paths(
            args.prepared, args.predictions, args.manifest, args.report
        )
        report = score_prediction_files(
            prepared_path=args.prepared,
            predictions_path=args.predictions,
            expected_prediction_sha256=args.prediction_sha256,
            manifest_path=args.manifest,
        )
        print(_write_json(args.report, report))
        return 0 if report["valid_run"] else 2
    report_paths = [args.keyword_report, args.dense_report, args.hybrid_report]
    report_hashes = [
        args.keyword_report_sha256,
        args.dense_report_sha256,
        args.hybrid_report_sha256,
    ]
    _assert_distinct_paths(*report_paths, args.output)
    comparison = compare_score_reports(report_paths, report_hashes)
    print(_write_json(args.output, comparison))
    return 0 if comparison["gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
