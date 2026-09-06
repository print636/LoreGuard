from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "issue-review-v1"
DEFAULT_MANIFEST = DATASET_ROOT / "manifest.json"
DEFAULT_FREEZE = DATASET_ROOT / "freeze.json"

EXECUTION_SCHEMA = "issue-review-eval-execution-v1"
PREDICTION_SCHEMA = "issue-review-eval-predictions-v1"
REPORT_SCHEMA = "issue-review-eval-report-v1"
COMPARISON_SCHEMA = "issue-review-eval-comparison-v1"
PROVIDER_CHECK_SCHEMA = "issue-review-provider-check-v1"

Mode = Literal["local-context", "rag-evidence"]
VERDICTS = {
    "supports_issue",
    "contextual_exception",
    "insufficient_evidence",
}
CASE_SPLITS = {"dev", "holdout"}
PACKAGE_SPLITS = {"dev", "holdout", "full"}
MODES = {"local-context", "rag-evidence"}
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_CASES = 32
MAX_DOCUMENTS = 64
MAX_REPEATS = 5
EXPECTED_HEAD_REVISION = "0003_embedding_identity"
PINNED_MANIFEST_SHA256 = "2e216d6452917d08dec0b909012dde6fe39caa1b967847b79f49a204c7febcee"
PINNED_FREEZE_SHA256 = "d2e418072ecd51ceb8a3183edebed04b7175b42d736a55573741a05a7592f271"
PINNED_EXECUTION_SHA256 = {
    "dev": "e6c50cb40c54ed91249b04aa26cbfc6ed96cd360acc3b891bd44538fd604aa67",
    "holdout": "c893a0307819c03c79c896b9e16b6444426a6a7d86dd8c8ec8bf122fd17779fa",
    "full": "790bdef30c16531199d7a9890edfab75c9c553b63dc0c53fe6279acc14668dca",
}

_FORBIDDEN_EXECUTION_KEYS = {
    "expected",
    "labels",
    "rationale",
    "required_reasoning_tags",
    "retrievable_evidence_ids",
    "excluded_evidence_ids",
}
_SENSITIVE_REPORT_KEYS = {
    "api_key",
    "authorization",
    "base_url",
    "database_url",
    "endpoint",
    "prompt",
    "raw_prompt",
    "raw_response",
    "response_text",
    "text",
    "content",
    "note",
}


class EvalContractError(ValueError):
    """A safe, content-free evaluation contract failure."""


class EvaluationExecutor(Protocol):
    metadata: dict[str, Any]

    def execute(self, case: dict[str, Any], mode: Mode) -> dict[str, Any]: ...


@dataclass(frozen=True)
class RuntimeConfiguration:
    mode: Mode
    repeats: int
    chat_model: str
    chat_endpoint_fingerprint: str
    thinking_mode: str | None
    top_k: int
    token_budget: int
    timeout_seconds: float
    total_deadline_seconds: float
    embedding_allow_insecure_http: bool = False

    def safe_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "repeats": self.repeats,
            "chat_model": _identifier(self.chat_model, 255),
            "chat_endpoint_fingerprint": self.chat_endpoint_fingerprint,
            "thinking_mode": self.thinking_mode,
            "top_k": self.top_k,
            "token_budget": self.token_budget,
            "timeout_seconds": self.timeout_seconds,
            "total_deadline_seconds": self.total_deadline_seconds,
            "embedding_allow_insecure_http": self.embedding_allow_insecure_http,
        }


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(_read_bytes(path))


def _read_bytes(path: Path) -> bytes:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise EvalContractError("input path is not a regular file")
    size = resolved.stat().st_size
    if size < 1 or size > MAX_FILE_BYTES:
        raise EvalContractError("input file size is invalid")
    return resolved.read_bytes()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_bytes(path))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise EvalContractError("input JSON is invalid") from None
    if not isinstance(value, dict):
        raise EvalContractError("input JSON root must be an object")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any], *, protected: Sequence[Path]) -> str:
    target = path.resolve()
    frozen_root = DATASET_ROOT.resolve()
    if target == frozen_root or frozen_root in target.parents:
        raise EvalContractError("output cannot modify the frozen dataset")
    if any(target == candidate.resolve() for candidate in protected):
        raise EvalContractError("output cannot overwrite an input")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return _sha256_bytes(payload)


def _exact_keys(value: dict[str, Any], expected: set[str], location: str) -> None:
    if set(value) != expected:
        raise EvalContractError(f"{location} schema is invalid")


def _identifier(value: Any, limit: int = 160) -> str:
    if (
        isinstance(value, str)
        and value == value.strip()
        and 1 <= len(value) <= limit
        and all(character.isalnum() or character in "-_.@:/+" for character in value)
    ):
        return value
    raise EvalContractError("identifier is invalid")


def _positive_int(value: Any, *, maximum: int) -> int:
    if type(value) is not int or not 1 <= value <= maximum:
        raise EvalContractError("positive integer is invalid")
    return value


def _nonnegative_int(value: Any, *, maximum: int = 1_000_000_000) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise EvalContractError("nonnegative integer is invalid")
    return value


def _safe_number(value: Any, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvalContractError("numeric value is invalid")
    prepared = float(value)
    if not 0 <= prepared <= maximum:
        raise EvalContractError("numeric value is invalid")
    return prepared


def _positive_number(value: Any, *, maximum: float) -> float:
    prepared = _safe_number(value, maximum=maximum)
    if prepared <= 0:
        raise EvalContractError("positive numeric value is invalid")
    return prepared


def _reject_execution_oracle(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or key in _FORBIDDEN_EXECUTION_KEYS:
                raise EvalContractError("execution package contains an oracle field")
            _reject_execution_oracle(child)
    elif isinstance(value, list):
        for child in value:
            _reject_execution_oracle(child)


def _reject_sensitive_report(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or key.casefold() in _SENSITIVE_REPORT_KEYS:
                raise EvalContractError("report contains a sensitive field")
            _reject_sensitive_report(child)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_report(child)


def _line_count(raw: bytes) -> int:
    return len(raw.decode("utf-8").splitlines())


def _verify_frozen_dataset(manifest_path: Path, freeze_path: Path) -> tuple[dict[str, Any], str, str]:
    manifest_bytes = _read_bytes(manifest_path)
    freeze_bytes = _read_bytes(freeze_path)
    try:
        manifest = json.loads(manifest_bytes)
        freeze = json.loads(freeze_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise EvalContractError("frozen dataset JSON is invalid") from None
    if not isinstance(manifest, dict) or not isinstance(freeze, dict):
        raise EvalContractError("frozen dataset root is invalid")
    if _sha256_bytes(manifest_bytes) != PINNED_MANIFEST_SHA256:
        raise EvalContractError("manifest does not match the pinned issue-review-v1 hash")
    if _sha256_bytes(freeze_bytes) != PINNED_FREEZE_SHA256:
        raise EvalContractError("freeze does not match the pinned issue-review-v1 hash")
    if manifest.get("dataset_id") != "issue-review-v1" or freeze.get("dataset_id") != "issue-review-v1":
        raise EvalContractError("frozen dataset identity is invalid")
    rows = freeze.get("frozen_files")
    if not isinstance(rows, list) or not rows:
        raise EvalContractError("freeze inventory is invalid")
    inventory: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise EvalContractError("freeze inventory row is invalid")
        _exact_keys(row, {"path", "sha256", "bytes", "line_count"}, "freeze row")
        relative = row["path"]
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise EvalContractError("freeze path is invalid")
        if relative in inventory:
            raise EvalContractError("freeze inventory contains duplicates")
        inventory[relative] = row
        raw = _read_bytes(DATASET_ROOT / relative)
        if (
            _sha256_bytes(raw) != row["sha256"]
            or len(raw) != row["bytes"]
            or _line_count(raw) != row["line_count"]
        ):
            raise EvalContractError("frozen file verification failed")
    manifest_row = inventory.get("manifest.json")
    if manifest_row is None or manifest_row["sha256"] != _sha256_bytes(manifest_bytes):
        raise EvalContractError("manifest is not bound to the freeze")
    return manifest, _sha256_bytes(manifest_bytes), _sha256_bytes(freeze_bytes)


def _evidence_text(document_text: str, start_line: int, end_line: int) -> str:
    lines = document_text.splitlines()
    if not (1 <= start_line <= end_line <= len(lines)):
        raise EvalContractError("evidence range is outside its document")
    return "\n".join(lines[start_line - 1 : end_line])


def prepare_execution_package(
    *, manifest_path: Path = DEFAULT_MANIFEST, freeze_path: Path = DEFAULT_FREEZE, split: str
) -> dict[str, Any]:
    if split not in PACKAGE_SPLITS:
        raise EvalContractError("split is invalid")
    manifest, manifest_sha, freeze_sha = _verify_frozen_dataset(manifest_path, freeze_path)
    worlds = manifest.get("worlds")
    evidence_rows = manifest.get("evidence")
    cases = manifest.get("cases")
    if not isinstance(worlds, list) or not isinstance(evidence_rows, list) or not isinstance(cases, list):
        raise EvalContractError("manifest collections are invalid")

    world_map: dict[str, dict[str, Any]] = {}
    document_map: dict[tuple[str, str, int], dict[str, Any]] = {}
    prepared_documents: dict[str, list[dict[str, Any]]] = {}
    for world in worlds:
        if (
            not isinstance(world, dict)
            or world.get("split") not in CASE_SPLITS
            or (split != "full" and world.get("split") != split)
        ):
            continue
        world_id = _identifier(world.get("world_id"), 80)
        project_id = _identifier(world.get("project_id"), 80)
        if world_id in world_map:
            raise EvalContractError("manifest world is duplicated")
        documents = world.get("documents")
        if not isinstance(documents, list) or not documents or len(documents) > MAX_DOCUMENTS:
            raise EvalContractError("manifest world documents are invalid")
        world_map[world_id] = world
        output_documents: list[dict[str, Any]] = []
        for document in documents:
            if not isinstance(document, dict):
                raise EvalContractError("manifest document is invalid")
            document_id = _identifier(document.get("document_id"), 80)
            version = _positive_int(document.get("version"), maximum=1_000_000)
            relative = document.get("path")
            if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise EvalContractError("manifest document path is invalid")
            raw = _read_bytes(DATASET_ROOT / relative)
            content = raw.decode("utf-8")
            prepared = {
                "project_id": project_id,
                "document_id": document_id,
                "document_version": version,
                "role": _identifier(document.get("role"), 80),
                "active": document.get("active") is True,
                "authorized_for_review": document.get("authorized_for_review") is True,
                "content_sha256": _sha256_bytes(raw),
                "line_count": len(content.splitlines()),
                "content": content,
            }
            key = (world_id, document_id, version)
            if key in document_map:
                raise EvalContractError("manifest document snapshot is duplicated")
            document_map[key] = prepared
            output_documents.append(prepared)
        prepared_documents[world_id] = output_documents

    evidence_map: dict[str, dict[str, Any]] = {}
    for row in evidence_rows:
        if not isinstance(row, dict) or row.get("world_id") not in world_map:
            continue
        evidence_id = _identifier(row.get("evidence_id"), 80)
        document = document_map.get((row["world_id"], row.get("document_id"), row.get("version")))
        if document is None or evidence_id in evidence_map:
            raise EvalContractError("manifest evidence identity is invalid")
        start = _positive_int(row.get("start_line"), maximum=document["line_count"])
        end = _positive_int(row.get("end_line"), maximum=document["line_count"])
        if end < start:
            raise EvalContractError("manifest evidence range is invalid")
        evidence_map[evidence_id] = {
            "evidence_id": evidence_id,
            "project_id": document["project_id"],
            "document_id": document["document_id"],
            "document_version": document["document_version"],
            "content_sha256": document["content_sha256"],
            "line_start": start,
            "line_end": end,
            "text": _evidence_text(document["content"], start, end),
        }

    prepared_cases: list[dict[str, Any]] = []
    for case in cases:
        if (
            not isinstance(case, dict)
            or case.get("split") not in CASE_SPLITS
            or (split != "full" and case.get("split") != split)
        ):
            continue
        world_id = _identifier(case.get("world_id"), 80)
        if world_id not in world_map:
            raise EvalContractError("case references an unknown world")
        current_ids = case.get("current_evidence_ids")
        if not isinstance(current_ids, list) or not current_ids:
            raise EvalContractError("case current evidence is invalid")
        current = []
        for evidence_id in current_ids:
            if not isinstance(evidence_id, str) or evidence_id not in evidence_map:
                raise EvalContractError("case current evidence identity is invalid")
            current.append(dict(evidence_map[evidence_id]))
        documents = prepared_documents[world_id]
        allowed = [
            {
                "project_id": document["project_id"],
                "document_id": document["document_id"],
                "document_version": document["document_version"],
                "content_sha256": document["content_sha256"],
            }
            for document in documents
            if document["active"] and document["authorized_for_review"]
        ]
        candidate = case.get("candidate_issue")
        if not isinstance(candidate, dict):
            raise EvalContractError("candidate issue is invalid")
        _exact_keys(
            candidate,
            {"issue_type", "rule_category", "severity", "summary", "detector_basis"},
            "candidate issue",
        )
        prepared_cases.append(
            {
                "case_id": _identifier(case.get("case_id"), 80),
                "split": case["split"],
                "world_id": world_id,
                "project_id": _identifier(world_map[world_id].get("project_id"), 80),
                "candidate_issue": dict(candidate),
                "current_evidence": current,
                "allowed_snapshots": allowed,
                "documents": documents,
            }
        )
    if split == "full":
        declared_count = sum(
            row.get("case_count", 0)
            for row in manifest.get("splits", {}).values()
            if isinstance(row, dict) and type(row.get("case_count")) is int
        )
    else:
        declared_split = manifest.get("splits", {}).get(split, {})
        declared_count = declared_split.get("case_count") if isinstance(declared_split, dict) else None
    if not prepared_cases or len(prepared_cases) > MAX_CASES or len(prepared_cases) != declared_count:
        raise EvalContractError("prepared split is incomplete")
    package = {
        "schema_version": EXECUTION_SCHEMA,
        "dataset_id": "issue-review-v1",
        "split": split,
        "manifest_sha256": manifest_sha,
        "freeze_sha256": freeze_sha,
        "cases": prepared_cases,
    }
    validate_execution_package(package)
    if _sha256_bytes(_canonical_bytes(package)) != PINNED_EXECUTION_SHA256[split]:
        raise EvalContractError("prepared execution package does not match its pinned hash")
    return package


def validate_execution_package(package: dict[str, Any]) -> None:
    if not isinstance(package, dict):
        raise EvalContractError("execution package is invalid")
    _reject_execution_oracle(package)
    _exact_keys(
        package,
        {"schema_version", "dataset_id", "split", "manifest_sha256", "freeze_sha256", "cases"},
        "execution package",
    )
    if package["schema_version"] != EXECUTION_SCHEMA or package["dataset_id"] != "issue-review-v1":
        raise EvalContractError("execution package identity is invalid")
    if package["split"] not in PACKAGE_SPLITS:
        raise EvalContractError("execution package split is invalid")
    for key in ("manifest_sha256", "freeze_sha256"):
        if not isinstance(package[key], str) or len(package[key]) != 64:
            raise EvalContractError("execution package hash is invalid")
    cases = package["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES:
        raise EvalContractError("execution cases are invalid")
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise EvalContractError("execution case is invalid")
        _exact_keys(
            case,
            {"case_id", "split", "world_id", "project_id", "candidate_issue", "current_evidence", "allowed_snapshots", "documents"},
            "execution case",
        )
        case_id = _identifier(case["case_id"], 80)
        if (
            case_id in case_ids
            or case["split"] not in CASE_SPLITS
            or (package["split"] != "full" and case["split"] != package["split"])
        ):
            raise EvalContractError("execution case identity is invalid")
        case_ids.add(case_id)
        _identifier(case["world_id"], 80)
        project_id = _identifier(case["project_id"], 80)
        candidate = case["candidate_issue"]
        if not isinstance(candidate, dict):
            raise EvalContractError("execution candidate issue is invalid")
        _exact_keys(candidate, {"issue_type", "rule_category", "severity", "summary", "detector_basis"}, "execution candidate issue")
        for key in ("issue_type", "rule_category", "severity", "summary", "detector_basis"):
            if not isinstance(candidate[key], str) or not candidate[key].strip() or len(candidate[key]) > 2_000:
                raise EvalContractError("execution candidate field is invalid")
        documents = case["documents"]
        if not isinstance(documents, list) or not 1 <= len(documents) <= MAX_DOCUMENTS:
            raise EvalContractError("execution documents are invalid")
        snapshots: set[tuple[str, str, int, str]] = set()
        document_lines: dict[tuple[str, int], int] = {}
        for document in documents:
            if not isinstance(document, dict):
                raise EvalContractError("execution document is invalid")
            _exact_keys(document, {"project_id", "document_id", "document_version", "role", "active", "authorized_for_review", "content_sha256", "line_count", "content"}, "execution document")
            if document["project_id"] != project_id:
                raise EvalContractError("execution document project scope is invalid")
            document_id = _identifier(document["document_id"], 80)
            version = _positive_int(document["document_version"], maximum=1_000_000)
            _identifier(document["role"], 80)
            if type(document["active"]) is not bool or type(document["authorized_for_review"]) is not bool:
                raise EvalContractError("execution document authorization is invalid")
            content = document["content"]
            if not isinstance(content, str) or not content or len(content.encode("utf-8")) > MAX_FILE_BYTES:
                raise EvalContractError("execution document content is invalid")
            content_hash = _sha256_bytes(content.encode("utf-8"))
            if content_hash != document["content_sha256"] or document["line_count"] != len(content.splitlines()):
                raise EvalContractError("execution document snapshot is invalid")
            identity = (project_id, document_id, version, content_hash)
            if identity in snapshots:
                raise EvalContractError("execution document snapshot is duplicated")
            snapshots.add(identity)
            document_lines[(document_id, version)] = document["line_count"]
        allowed = case["allowed_snapshots"]
        if not isinstance(allowed, list) or not allowed:
            raise EvalContractError("execution allowed snapshots are invalid")
        allowed_set: set[tuple[str, str, int, str]] = set()
        for snapshot in allowed:
            if not isinstance(snapshot, dict):
                raise EvalContractError("execution allowed snapshot is invalid")
            _exact_keys(snapshot, {"project_id", "document_id", "document_version", "content_sha256"}, "execution allowed snapshot")
            identity = (snapshot["project_id"], snapshot["document_id"], snapshot["document_version"], snapshot["content_sha256"])
            if identity not in snapshots or identity in allowed_set:
                raise EvalContractError("execution allowed snapshot scope is invalid")
            allowed_set.add(identity)
        expected_allowed = {
            (document["project_id"], document["document_id"], document["document_version"], document["content_sha256"])
            for document in documents
            if document["active"] and document["authorized_for_review"]
        }
        if allowed_set != expected_allowed:
            raise EvalContractError("execution allowed snapshots do not match authorization")
        current = case["current_evidence"]
        if not isinstance(current, list) or not current:
            raise EvalContractError("execution current evidence is invalid")
        evidence_ids: set[str] = set()
        for evidence in current:
            if not isinstance(evidence, dict):
                raise EvalContractError("execution evidence is invalid")
            _exact_keys(evidence, {"evidence_id", "project_id", "document_id", "document_version", "content_sha256", "line_start", "line_end", "text"}, "execution evidence")
            evidence_id = _identifier(evidence["evidence_id"], 80)
            if evidence_id in evidence_ids:
                raise EvalContractError("execution evidence is duplicated")
            evidence_ids.add(evidence_id)
            identity = (evidence["project_id"], evidence["document_id"], evidence["document_version"], evidence["content_sha256"])
            if identity not in allowed_set:
                raise EvalContractError("current evidence is outside the allowed scope")
            maximum = document_lines[(evidence["document_id"], evidence["document_version"])]
            start = _positive_int(evidence["line_start"], maximum=maximum)
            end = _positive_int(evidence["line_end"], maximum=maximum)
            if end < start or not isinstance(evidence["text"], str) or not evidence["text"]:
                raise EvalContractError("execution evidence range is invalid")
            document = next(row for row in documents if row["document_id"] == evidence["document_id"] and row["document_version"] == evidence["document_version"])
            if evidence["text"] != _evidence_text(document["content"], start, end):
                raise EvalContractError("execution evidence text is not bound to its source")


def _endpoint_fingerprint(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip() or len(base_url) > 2_048:
        raise EvalContractError("provider base URL is invalid")
    return "endpoint-sha256:" + hashlib.sha256(base_url.strip().encode("utf-8")).hexdigest()


def _git_commit_and_clean(root: Path = ROOT) -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=root, check=True, capture_output=True, text=True, timeout=10
        ).stdout
    except (OSError, subprocess.SubprocessError):
        raise EvalContractError("git state could not be verified") from None
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise EvalContractError("git commit is invalid")
    if dirty:
        raise EvalContractError("worktree must be clean before a real run")
    return commit


def _validate_result(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise EvalContractError("executor result is invalid")
    _exact_keys(
        result,
        {"status", "verdict", "citations", "review_outcome", "failure_category", "retrieval_mode", "retrieval_strategy", "latency_ms", "prompt_tokens", "completion_tokens", "charged_tokens", "provider_calls"},
        "executor result",
    )
    if result["status"] not in {"success", "degraded", "failed"}:
        raise EvalContractError("executor status is invalid")
    if result["verdict"] is not None and result["verdict"] not in VERDICTS:
        raise EvalContractError("executor verdict is invalid")
    if result["status"] != "success" and result["verdict"] is not None:
        raise EvalContractError("failed executor result cannot contain a verdict")
    if not isinstance(result["citations"], list) or len(result["citations"]) > 12:
        raise EvalContractError("executor citations are invalid")
    for citation in result["citations"]:
        if not isinstance(citation, dict):
            raise EvalContractError("executor citation is invalid")
        _exact_keys(citation, {"project_id", "document_id", "document_version", "content_sha256", "line_start", "line_end"}, "executor citation")
        _identifier(citation["project_id"], 80)
        _identifier(citation["document_id"], 80)
        _positive_int(citation["document_version"], maximum=1_000_000)
        if not isinstance(citation["content_sha256"], str) or len(citation["content_sha256"]) != 64:
            raise EvalContractError("executor citation hash is invalid")
        start = _positive_int(citation["line_start"], maximum=1_000_000)
        end = _positive_int(citation["line_end"], maximum=1_000_000)
        if end < start:
            raise EvalContractError("executor citation range is invalid")
    for key in ("latency_ms", "prompt_tokens", "completion_tokens", "charged_tokens"):
        _nonnegative_int(result[key])
    if not isinstance(result["provider_calls"], list) or len(result["provider_calls"]) > 16:
        raise EvalContractError("executor provider diagnostics are invalid")
    for call in result["provider_calls"]:
        if not isinstance(call, dict) or any(key.casefold() in _SENSITIVE_REPORT_KEYS for key in call):
            raise EvalContractError("executor provider diagnostics are unsafe")
    for key in ("review_outcome", "failure_category", "retrieval_mode", "retrieval_strategy"):
        if result[key] is not None:
            _identifier(result[key], 80)
    return result


def run_execution_package(
    package: dict[str, Any],
    *,
    execution_sha256: str,
    configuration: RuntimeConfiguration,
    executor: EvaluationExecutor,
    code_commit: str,
) -> dict[str, Any]:
    validate_execution_package(package)
    canonical_execution_sha = _sha256_bytes(_canonical_bytes(package))
    if execution_sha256 != canonical_execution_sha:
        raise EvalContractError("execution package hash does not match")
    if canonical_execution_sha != PINNED_EXECUTION_SHA256.get(package["split"]):
        raise EvalContractError("execution package does not match the pinned suite")
    if configuration.mode not in MODES or not 1 <= configuration.repeats <= MAX_REPEATS:
        raise EvalContractError("run configuration is invalid")
    _identifier(code_commit, 64)
    case_results: list[dict[str, Any]] = []
    for case in package["cases"]:
        repeats: list[dict[str, Any]] = []
        for repeat_index in range(1, configuration.repeats + 1):
            try:
                result = _validate_result(executor.execute(case, configuration.mode))
            except Exception as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                result = {
                    "status": "failed",
                    "verdict": None,
                    "citations": [],
                    "review_outcome": "failed",
                    "failure_category": "execution_failure",
                    "retrieval_mode": None,
                    "retrieval_strategy": None,
                    "latency_ms": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "charged_tokens": 0,
                    "provider_calls": [],
                }
            repeats.append({"repeat": repeat_index, **result})
        case_results.append({"case_id": case["case_id"], "runs": repeats})
    metadata = getattr(executor, "metadata", None)
    if not isinstance(metadata, dict):
        raise EvalContractError("executor metadata is invalid")
    _reject_sensitive_report(metadata)
    predictions = {
        "schema_version": PREDICTION_SCHEMA,
        "dataset_id": package["dataset_id"],
        "split": package["split"],
        "manifest_sha256": package["manifest_sha256"],
        "freeze_sha256": package["freeze_sha256"],
        "execution_sha256": execution_sha256,
        "code_commit": code_commit,
        "configuration": configuration.safe_dict(),
        "executor_metadata": metadata,
        "cases": case_results,
    }
    validate_predictions(predictions, package)
    return predictions


def validate_predictions(predictions: dict[str, Any], package: dict[str, Any]) -> None:
    if not isinstance(predictions, dict):
        raise EvalContractError("predictions are invalid")
    _exact_keys(predictions, {"schema_version", "dataset_id", "split", "manifest_sha256", "freeze_sha256", "execution_sha256", "code_commit", "configuration", "executor_metadata", "cases"}, "predictions")
    if predictions["schema_version"] != PREDICTION_SCHEMA:
        raise EvalContractError("prediction artifact type is invalid")
    for key in ("dataset_id", "split", "manifest_sha256", "freeze_sha256"):
        if predictions[key] != package[key]:
            raise EvalContractError("predictions are not bound to the execution package")
    if predictions["execution_sha256"] != _sha256_bytes(_canonical_bytes(package)):
        raise EvalContractError("prediction execution hash is invalid")
    _identifier(predictions["code_commit"], 64)
    configuration = predictions["configuration"]
    if not isinstance(configuration, dict):
        raise EvalContractError("prediction configuration is invalid")
    _exact_keys(configuration, {"mode", "repeats", "chat_model", "chat_endpoint_fingerprint", "thinking_mode", "top_k", "token_budget", "timeout_seconds", "total_deadline_seconds", "embedding_allow_insecure_http"}, "prediction configuration")
    if configuration["mode"] not in MODES:
        raise EvalContractError("prediction mode is invalid")
    repeats = _positive_int(configuration["repeats"], maximum=MAX_REPEATS)
    _identifier(configuration["chat_model"], 255)
    fingerprint = configuration["chat_endpoint_fingerprint"]
    if not isinstance(fingerprint, str) or not fingerprint.startswith("endpoint-sha256:") or len(fingerprint) != 80:
        raise EvalContractError("prediction provider fingerprint is invalid")
    if configuration["thinking_mode"] not in {None, "disabled", "enabled"}:
        raise EvalContractError("prediction thinking mode is invalid")
    _positive_int(configuration["top_k"], maximum=6)
    _positive_int(configuration["token_budget"], maximum=6_000)
    timeout_seconds = _positive_number(configuration["timeout_seconds"], maximum=20)
    total_deadline_seconds = _positive_number(
        configuration["total_deadline_seconds"], maximum=45
    )
    if total_deadline_seconds < timeout_seconds:
        raise EvalContractError("prediction deadline is shorter than its timeout")
    if type(configuration["embedding_allow_insecure_http"]) is not bool:
        raise EvalContractError("prediction embedding HTTP opt-in is invalid")
    metadata = predictions["executor_metadata"]
    if not isinstance(metadata, dict):
        raise EvalContractError("prediction executor metadata is invalid")
    _reject_sensitive_report(metadata)
    case_ids = [case["case_id"] for case in package["cases"]]
    result_cases = predictions["cases"]
    if not isinstance(result_cases, list) or [case.get("case_id") if isinstance(case, dict) else None for case in result_cases] != case_ids:
        raise EvalContractError("prediction case coverage is incomplete")
    for case in result_cases:
        _exact_keys(case, {"case_id", "runs"}, "prediction case")
        if not isinstance(case["runs"], list) or len(case["runs"]) != repeats:
            raise EvalContractError("prediction repeat coverage is incomplete")
        for index, run in enumerate(case["runs"], start=1):
            if not isinstance(run, dict) or run.get("repeat") != index:
                raise EvalContractError("prediction repeat identity is invalid")
            _validate_result({key: value for key, value in run.items() if key != "repeat"})


def _oracle_maps(manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    cases = manifest.get("cases")
    evidence = manifest.get("evidence")
    worlds = manifest.get("worlds")
    if not isinstance(cases, list) or not isinstance(evidence, list) or not isinstance(worlds, list):
        raise EvalContractError("oracle manifest is invalid")
    return (
        {row["case_id"]: row for row in cases if isinstance(row, dict) and isinstance(row.get("case_id"), str)},
        {row["evidence_id"]: row for row in evidence if isinstance(row, dict) and isinstance(row.get("evidence_id"), str)},
        {row["world_id"]: row for row in worlds if isinstance(row, dict) and isinstance(row.get("world_id"), str)},
    )


def _citation_covers(
    citation: dict[str, Any],
    evidence: dict[str, Any],
    project_id: str,
    content_sha256: str,
) -> bool:
    return (
        citation["project_id"] == project_id
        and citation["document_id"] == evidence.get("document_id")
        and citation["document_version"] == evidence.get("version")
        and citation["content_sha256"] == content_sha256
        and citation["line_start"] <= evidence.get("start_line", 0)
        and citation["line_end"] >= evidence.get("end_line", 0)
    )


def score_predictions(
    *,
    prediction_path: Path,
    prediction_sha256: str,
    execution_path: Path,
    execution_sha256: str,
    manifest_path: Path = DEFAULT_MANIFEST,
    freeze_path: Path = DEFAULT_FREEZE,
) -> dict[str, Any]:
    prediction_bytes = _read_bytes(prediction_path)
    if _sha256_bytes(prediction_bytes) != prediction_sha256:
        raise EvalContractError("prediction artifact hash does not match")
    execution_bytes = _read_bytes(execution_path)
    if _sha256_bytes(execution_bytes) != execution_sha256:
        raise EvalContractError("execution artifact hash does not match")
    try:
        predictions = json.loads(prediction_bytes)
        package = json.loads(execution_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise EvalContractError("evaluation artifact JSON is invalid") from None
    if not isinstance(predictions, dict) or not isinstance(package, dict):
        raise EvalContractError("evaluation artifact root is invalid")
    validate_execution_package(package)
    if execution_bytes != _canonical_bytes(package):
        raise EvalContractError("execution artifact is not canonical JSON")
    if execution_sha256 != PINNED_EXECUTION_SHA256.get(package["split"]):
        raise EvalContractError("execution artifact does not match the pinned suite")
    validate_predictions(predictions, package)
    # Oracle loading deliberately happens only after both immutable prediction
    # and execution artifact hashes and strict schemas have been accepted.
    rebuilt = prepare_execution_package(
        manifest_path=manifest_path,
        freeze_path=freeze_path,
        split=package["split"],
    )
    if execution_bytes != _canonical_bytes(rebuilt):
        raise EvalContractError("execution artifact differs from the pinned reconstruction")
    manifest, manifest_sha, freeze_sha = _verify_frozen_dataset(manifest_path, freeze_path)
    if manifest_sha != package["manifest_sha256"] or freeze_sha != package["freeze_sha256"]:
        raise EvalContractError("scoring oracle does not match the execution package")
    case_oracle, evidence_map, worlds = _oracle_maps(manifest)

    expected_case_ids = [case["case_id"] for case in package["cases"]]
    selected_oracle = [
        row
        for row in manifest["cases"]
        if package["split"] == "full" or row.get("split") == package["split"]
    ]
    if [row.get("case_id") for row in selected_oracle] != expected_case_ids:
        raise EvalContractError("oracle case coverage does not match predictions")
    result_map = {row["case_id"]: row for row in predictions["cases"]}
    execution_case_map = {row["case_id"]: row for row in package["cases"]}
    case_scores: list[dict[str, Any]] = []
    class_totals = {verdict: {"cases": 0, "correct_cases": 0} for verdict in sorted(VERDICTS)}
    latency_values: list[int] = []
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "charged_tokens": 0}
    failed_runs = degraded_runs = excluded_leaks = non_allowlisted = 0
    total_citations = allowlisted_citations = 0
    contextual_exception_to_supports = 0
    for case_id in expected_case_ids:
        oracle = case_oracle.get(case_id)
        if not isinstance(oracle, dict):
            raise EvalContractError("oracle case is missing")
        expected = oracle.get("expected")
        if not isinstance(expected, dict) or expected.get("verdict") not in VERDICTS:
            raise EvalContractError("oracle expected result is invalid")
        expected_verdict = expected["verdict"]
        world = worlds.get(oracle.get("world_id"))
        if not isinstance(world, dict):
            raise EvalContractError("oracle world is missing")
        project_id = world.get("project_id")
        execution_case = execution_case_map[case_id]
        snapshot_hashes = {
            (document["document_id"], document["document_version"]): document[
                "content_sha256"
            ]
            for document in execution_case["documents"]
        }
        authorized = {
            (
                snapshot["document_id"],
                snapshot["document_version"],
                snapshot["content_sha256"],
            )
            for snapshot in execution_case["allowed_snapshots"]
        }
        excluded_ids = oracle.get("excluded_evidence_ids", [])
        gold_ids = expected.get("citation_evidence_ids")
        if not isinstance(excluded_ids, list) or not isinstance(gold_ids, list) or not gold_ids:
            raise EvalContractError("oracle evidence labels are invalid")
        run_scores: list[dict[str, Any]] = []
        for run in result_map[case_id]["runs"]:
            status = run["status"]
            if status == "failed":
                failed_runs += 1
            elif status == "degraded":
                degraded_runs += 1
            if (
                expected_verdict == "contextual_exception"
                and status == "success"
                and run["verdict"] == "supports_issue"
            ):
                contextual_exception_to_supports += 1
            latency_values.append(run["latency_ms"])
            for key in token_totals:
                token_totals[key] += run[key]
            citations = run["citations"]
            citation_allowlist = [
                citation["project_id"] == project_id
                and (
                    citation["document_id"],
                    citation["document_version"],
                    citation["content_sha256"],
                )
                in authorized
                for citation in citations
            ]
            total_citations += len(citation_allowlist)
            allowlisted_citations += sum(citation_allowlist)
            allowlist_ok = all(citation_allowlist)
            if not allowlist_ok:
                non_allowlisted += 1
            leaked_ids = [
                evidence_id
                for evidence_id in excluded_ids
                if evidence_id in evidence_map
                and (
                    content_hash := snapshot_hashes.get(
                        (
                            evidence_map[evidence_id].get("document_id"),
                            evidence_map[evidence_id].get("version"),
                        )
                    )
                )
                is not None
                and any(
                    _citation_covers(
                        citation,
                        evidence_map[evidence_id],
                        project_id,
                        content_hash,
                    )
                    for citation in citations
                )
            ]
            if leaked_ids:
                excluded_leaks += len(leaked_ids)
            citation_coverage = all(
                evidence_id in evidence_map
                and (
                    content_hash := snapshot_hashes.get(
                        (
                            evidence_map[evidence_id].get("document_id"),
                            evidence_map[evidence_id].get("version"),
                        )
                    )
                )
                is not None
                and any(
                    _citation_covers(
                        citation,
                        evidence_map[evidence_id],
                        project_id,
                        content_hash,
                    )
                    for citation in citations
                )
                for evidence_id in gold_ids
            )
            run_scores.append(
                {
                    "repeat": run["repeat"],
                    "status": status,
                    "verdict": run["verdict"],
                    "successful": status == "success",
                    "verdict_correct": status == "success" and run["verdict"] == expected_verdict,
                    "citation_allowlist_ok": allowlist_ok,
                    "citation_count": len(citations),
                    "allowlisted_citation_count": sum(citation_allowlist),
                    "minimal_citation_coverage": citation_coverage,
                    "excluded_evidence_leak": bool(leaked_ids),
                    "excluded_evidence_leak_count": len(leaked_ids),
                    "failure_category": run["failure_category"],
                    "latency_ms": run["latency_ms"],
                    "prompt_tokens": run["prompt_tokens"],
                    "completion_tokens": run["completion_tokens"],
                    "charged_tokens": run["charged_tokens"],
                }
            )
        verdict_correct = all(row["verdict_correct"] for row in run_scores)
        citation_covered = all(row["minimal_citation_coverage"] for row in run_scores)
        allowlist_ok = all(row["citation_allowlist_ok"] for row in run_scores)
        no_excluded_leak = not any(row["excluded_evidence_leak"] for row in run_scores)
        class_totals[expected_verdict]["cases"] += 1
        class_totals[expected_verdict]["correct_cases"] += int(verdict_correct)
        case_scores.append(
            {
                "case_id": case_id,
                "expected_class": expected_verdict,
                "verdict_correct_all_repeats": verdict_correct,
                "minimal_citation_coverage_all_repeats": citation_covered,
                "citation_allowlist_ok_all_repeats": allowlist_ok,
                "no_excluded_evidence_leak": no_excluded_leak,
                "runs": run_scores,
            }
        )
    correct_cases = sum(row["verdict_correct_all_repeats"] for row in case_scores)
    covered_cases = sum(row["minimal_citation_coverage_all_repeats"] for row in case_scores)
    stable_cases = sum(
        len({run["verdict"] for run in result_map[row["case_id"]]["runs"]}) == 1
        and all(run["status"] == "success" for run in result_map[row["case_id"]]["runs"])
        for row in case_scores
    )
    ordered_latency = sorted(latency_values)
    p95 = ordered_latency[max(0, (95 * len(ordered_latency) + 99) // 100 - 1)] if ordered_latency else 0
    split_is_full = package["split"] == "full" and len(case_scores) == 12
    absolute_gate = (
        correct_cases >= 10
        and class_totals["contextual_exception"]["correct_cases"] >= 3
        and contextual_exception_to_supports == 0
        and non_allowlisted == 0
        and excluded_leaks == 0
        and covered_cases >= 11
        and split_is_full
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "dataset_id": package["dataset_id"],
        "split": package["split"],
        "manifest_sha256": manifest_sha,
        "freeze_sha256": freeze_sha,
        "execution_sha256": execution_sha256,
        "prediction_sha256": prediction_sha256,
        "code_commit": predictions["code_commit"],
        "configuration": predictions["configuration"],
        "executor_metadata": predictions["executor_metadata"],
        "summary": {
            "case_count": len(case_scores),
            "repeats": predictions["configuration"]["repeats"],
            "correct_cases_all_repeats": correct_cases,
            "minimal_citation_coverage_cases_all_repeats": covered_cases,
            "stable_cases": stable_cases,
            "non_allowlisted_runs": non_allowlisted,
            "citation_allowlist": {
                "allowed": allowlisted_citations,
                "total": total_citations,
                "rate": (
                    allowlisted_citations / total_citations
                    if total_citations
                    else None
                ),
            },
            "excluded_evidence_leaks": excluded_leaks,
            "failed_runs": failed_runs,
            "degraded_runs": degraded_runs,
            "contextual_exception_to_supports_runs": contextual_exception_to_supports,
            "latency_ms": {
                "count": len(latency_values),
                "mean": int(sum(latency_values) / len(latency_values)) if latency_values else 0,
                "p95": p95,
            },
            **token_totals,
            "by_expected_class": class_totals,
        },
        "cases": case_scores,
        "absolute_gate": {
            "evaluable": split_is_full,
            "passed": absolute_gate if split_is_full else None,
            "policy": "frozen-readme-v1",
            "reason": None if split_is_full else "full_12_case_suite_required",
        },
        "claim_scope": "frozen developer-visible constrained evidence review only",
    }
    _reject_sensitive_report(report)
    validate_report(report, manifest=manifest)
    return report


def validate_report(
    report: dict[str, Any],
    *,
    manifest: dict[str, Any] | None = None,
    manifest_path: Path = DEFAULT_MANIFEST,
    freeze_path: Path = DEFAULT_FREEZE,
) -> None:
    if not isinstance(report, dict):
        raise EvalContractError("comparison input is not a scored report")
    _reject_sensitive_report(report)
    _exact_keys(
        report,
        {
            "schema_version",
            "dataset_id",
            "split",
            "manifest_sha256",
            "freeze_sha256",
            "execution_sha256",
            "prediction_sha256",
            "code_commit",
            "configuration",
            "executor_metadata",
            "summary",
            "cases",
            "absolute_gate",
            "claim_scope",
        },
        "scored report",
    )
    if report["schema_version"] != REPORT_SCHEMA or report["dataset_id"] != "issue-review-v1":
        raise EvalContractError("comparison input is not a scored report")
    split = report["split"]
    if split not in PACKAGE_SPLITS:
        raise EvalContractError("report split is invalid")
    if report["manifest_sha256"] != PINNED_MANIFEST_SHA256 or report["freeze_sha256"] != PINNED_FREEZE_SHA256:
        raise EvalContractError("report is not bound to the pinned suite")
    if report["execution_sha256"] != PINNED_EXECUTION_SHA256[split]:
        raise EvalContractError("report execution binding is invalid")
    for key in ("prediction_sha256", "code_commit"):
        value = report[key]
        expected_length = 64 if key == "prediction_sha256" else 40
        if not isinstance(value, str) or len(value) != expected_length:
            raise EvalContractError("report artifact identity is invalid")
        if any(character not in "0123456789abcdef" for character in value):
            raise EvalContractError("report artifact identity is invalid")
    configuration = report["configuration"]
    if not isinstance(configuration, dict):
        raise EvalContractError("report configuration is invalid")
    _exact_keys(
        configuration,
        {"mode", "repeats", "chat_model", "chat_endpoint_fingerprint", "thinking_mode", "top_k", "token_budget", "timeout_seconds", "total_deadline_seconds", "embedding_allow_insecure_http"},
        "report configuration",
    )
    if configuration["mode"] not in MODES:
        raise EvalContractError("report mode is invalid")
    repeats = _positive_int(configuration["repeats"], maximum=MAX_REPEATS)
    _identifier(configuration["chat_model"], 255)
    fingerprint = configuration["chat_endpoint_fingerprint"]
    if not isinstance(fingerprint, str) or not fingerprint.startswith("endpoint-sha256:") or len(fingerprint) != 80:
        raise EvalContractError("report provider fingerprint is invalid")
    if configuration["thinking_mode"] not in {None, "disabled", "enabled"}:
        raise EvalContractError("report thinking mode is invalid")
    _positive_int(configuration["top_k"], maximum=6)
    _positive_int(configuration["token_budget"], maximum=6_000)
    timeout = _positive_number(configuration["timeout_seconds"], maximum=20)
    deadline = _positive_number(configuration["total_deadline_seconds"], maximum=45)
    if deadline < timeout or type(configuration["embedding_allow_insecure_http"]) is not bool:
        raise EvalContractError("report runtime bounds are invalid")
    if not isinstance(report["executor_metadata"], dict):
        raise EvalContractError("report executor metadata is invalid")

    if manifest is None:
        manifest, manifest_sha, freeze_sha = _verify_frozen_dataset(manifest_path, freeze_path)
        if manifest_sha != report["manifest_sha256"] or freeze_sha != report["freeze_sha256"]:
            raise EvalContractError("report oracle binding is invalid")
    oracle_cases = [
        row
        for row in manifest.get("cases", [])
        if isinstance(row, dict)
        and (split == "full" or row.get("split") == split)
    ]
    oracle_ids = [row.get("case_id") for row in oracle_cases]
    cases = report["cases"]
    if not isinstance(cases, list) or [row.get("case_id") if isinstance(row, dict) else None for row in cases] != oracle_ids:
        raise EvalContractError("report case coverage is invalid")

    class_totals = {verdict: {"cases": 0, "correct_cases": 0} for verdict in sorted(VERDICTS)}
    latency_values: list[int] = []
    token_totals = {"prompt_tokens": 0, "completion_tokens": 0, "charged_tokens": 0}
    correct_cases = covered_cases = stable_cases = 0
    non_allowlisted = excluded_leaks = failed_runs = degraded_runs = 0
    total_citations = allowlisted_citations = contextual_exception_to_supports = 0
    run_keys = {
        "repeat", "status", "verdict", "successful", "verdict_correct",
        "citation_allowlist_ok", "citation_count", "allowlisted_citation_count",
        "minimal_citation_coverage", "excluded_evidence_leak",
        "excluded_evidence_leak_count", "failure_category", "latency_ms",
        "prompt_tokens", "completion_tokens", "charged_tokens",
    }
    for case, oracle in zip(cases, oracle_cases, strict=True):
        _exact_keys(
            case,
            {"case_id", "expected_class", "verdict_correct_all_repeats", "minimal_citation_coverage_all_repeats", "citation_allowlist_ok_all_repeats", "no_excluded_evidence_leak", "runs"},
            "report case",
        )
        expected = oracle.get("expected")
        expected_class = expected.get("verdict") if isinstance(expected, dict) else None
        if expected_class not in VERDICTS or case["expected_class"] != expected_class:
            raise EvalContractError("report case semantics are invalid")
        runs = case["runs"]
        if not isinstance(runs, list) or len(runs) != repeats:
            raise EvalContractError("report repeat coverage is invalid")
        verdict_correct_values: list[bool] = []
        coverage_values: list[bool] = []
        allowlist_values: list[bool] = []
        leak_values: list[bool] = []
        verdict_values: list[str | None] = []
        successful_values: list[bool] = []
        for repeat_index, run in enumerate(runs, start=1):
            if not isinstance(run, dict):
                raise EvalContractError("report run is invalid")
            _exact_keys(run, run_keys, "report run")
            if run["repeat"] != repeat_index or run["status"] not in {"success", "degraded", "failed"}:
                raise EvalContractError("report run identity is invalid")
            verdict = run["verdict"]
            if verdict is not None and verdict not in VERDICTS:
                raise EvalContractError("report run verdict is invalid")
            successful = run["status"] == "success"
            if run["successful"] is not successful or (not successful and verdict is not None):
                raise EvalContractError("report run success semantics are invalid")
            verdict_correct = successful and verdict == expected_class
            if run["verdict_correct"] is not verdict_correct:
                raise EvalContractError("report verdict score was not derived from the run")
            for key in ("citation_allowlist_ok", "minimal_citation_coverage", "excluded_evidence_leak"):
                if type(run[key]) is not bool:
                    raise EvalContractError("report citation score is invalid")
            citation_count = _nonnegative_int(run["citation_count"], maximum=12)
            allowed_count = _nonnegative_int(run["allowlisted_citation_count"], maximum=12)
            leak_count = _nonnegative_int(run["excluded_evidence_leak_count"], maximum=64)
            if allowed_count > citation_count or run["citation_allowlist_ok"] is not (allowed_count == citation_count):
                raise EvalContractError("report allowlist score was not derived from the run")
            if run["excluded_evidence_leak"] is not (leak_count > 0):
                raise EvalContractError("report leakage score was not derived from the run")
            for key in ("latency_ms", "prompt_tokens", "completion_tokens", "charged_tokens"):
                _nonnegative_int(run[key])
            if run["failure_category"] is not None:
                _identifier(run["failure_category"], 80)
            if successful and run["failure_category"] is not None:
                raise EvalContractError("successful report run has a failure category")
            if run["status"] == "failed":
                failed_runs += 1
            elif run["status"] == "degraded":
                degraded_runs += 1
            if expected_class == "contextual_exception" and successful and verdict == "supports_issue":
                contextual_exception_to_supports += 1
            latency_values.append(run["latency_ms"])
            for key in token_totals:
                token_totals[key] += run[key]
            total_citations += citation_count
            allowlisted_citations += allowed_count
            non_allowlisted += int(not run["citation_allowlist_ok"])
            excluded_leaks += leak_count
            verdict_correct_values.append(verdict_correct)
            coverage_values.append(run["minimal_citation_coverage"])
            allowlist_values.append(run["citation_allowlist_ok"])
            leak_values.append(run["excluded_evidence_leak"])
            verdict_values.append(verdict)
            successful_values.append(successful)
        derived_case = {
            "verdict_correct_all_repeats": all(verdict_correct_values),
            "minimal_citation_coverage_all_repeats": all(coverage_values),
            "citation_allowlist_ok_all_repeats": all(allowlist_values),
            "no_excluded_evidence_leak": not any(leak_values),
        }
        if any(case[key] is not value for key, value in derived_case.items()):
            raise EvalContractError("report case aggregate was not derived from its runs")
        correct_cases += int(derived_case["verdict_correct_all_repeats"])
        covered_cases += int(derived_case["minimal_citation_coverage_all_repeats"])
        stable_cases += int(all(successful_values) and len(set(verdict_values)) == 1)
        class_totals[expected_class]["cases"] += 1
        class_totals[expected_class]["correct_cases"] += int(derived_case["verdict_correct_all_repeats"])

    ordered_latency = sorted(latency_values)
    p95 = ordered_latency[max(0, (95 * len(ordered_latency) + 99) // 100 - 1)] if ordered_latency else 0
    expected_summary = {
        "case_count": len(cases),
        "repeats": repeats,
        "correct_cases_all_repeats": correct_cases,
        "minimal_citation_coverage_cases_all_repeats": covered_cases,
        "stable_cases": stable_cases,
        "non_allowlisted_runs": non_allowlisted,
        "citation_allowlist": {
            "allowed": allowlisted_citations,
            "total": total_citations,
            "rate": allowlisted_citations / total_citations if total_citations else None,
        },
        "excluded_evidence_leaks": excluded_leaks,
        "failed_runs": failed_runs,
        "degraded_runs": degraded_runs,
        "contextual_exception_to_supports_runs": contextual_exception_to_supports,
        "latency_ms": {
            "count": len(latency_values),
            "mean": int(sum(latency_values) / len(latency_values)) if latency_values else 0,
            "p95": p95,
        },
        **token_totals,
        "by_expected_class": class_totals,
    }
    if report["summary"] != expected_summary:
        raise EvalContractError("report summary was not derived from its runs")
    full_suite = split == "full" and len(cases) == 12
    passed = (
        full_suite
        and correct_cases >= 10
        and class_totals["contextual_exception"]["correct_cases"] >= 3
        and contextual_exception_to_supports == 0
        and non_allowlisted == 0
        and excluded_leaks == 0
        and covered_cases >= 11
    )
    expected_gate = {
        "evaluable": full_suite,
        "passed": passed if full_suite else None,
        "policy": "frozen-readme-v1",
        "reason": None if full_suite else "full_12_case_suite_required",
    }
    if report["absolute_gate"] != expected_gate:
        raise EvalContractError("report gate was not derived from the frozen policy")
    if report["claim_scope"] != "frozen developer-visible constrained evidence review only":
        raise EvalContractError("report claim scope is invalid")


def compare_reports(
    *, local_path: Path, local_sha256: str, rag_path: Path, rag_sha256: str
) -> dict[str, Any]:
    local_bytes = _read_bytes(local_path)
    rag_bytes = _read_bytes(rag_path)
    if _sha256_bytes(local_bytes) != local_sha256 or _sha256_bytes(rag_bytes) != rag_sha256:
        raise EvalContractError("report artifact hash does not match")
    try:
        local = json.loads(local_bytes)
        rag = json.loads(rag_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise EvalContractError("report JSON is invalid") from None
    for report, mode in ((local, "local-context"), (rag, "rag-evidence")):
        validate_report(report)
        if report.get("configuration", {}).get("mode") != mode:
            raise EvalContractError("comparison report mode is invalid")
    same_fields = ("dataset_id", "split", "manifest_sha256", "freeze_sha256", "execution_sha256", "code_commit")
    if any(local.get(key) != rag.get(key) for key in same_fields):
        raise EvalContractError("comparison reports are not from the same evaluation")
    local_config = local["configuration"]
    rag_config = rag["configuration"]
    for key in ("repeats", "chat_model", "chat_endpoint_fingerprint", "thinking_mode", "top_k", "token_budget", "timeout_seconds", "total_deadline_seconds"):
        if local_config.get(key) != rag_config.get(key):
            raise EvalContractError("comparison reports do not use the same chat configuration")
    for key in ("backend", "parser_contract", "citation_validation"):
        if local.get("executor_metadata", {}).get(key) != rag.get("executor_metadata", {}).get(key):
            raise EvalContractError("comparison reports do not share production provenance")
    local_ids = [row.get("case_id") for row in local.get("cases", [])]
    rag_ids = [row.get("case_id") for row in rag.get("cases", [])]
    if local_ids != rag_ids or not local_ids:
        raise EvalContractError("comparison case coverage is different")
    local_correct = local["summary"]["correct_cases_all_repeats"]
    rag_correct = rag["summary"]["correct_cases_all_repeats"]
    local_exception = local["summary"]["by_expected_class"]["contextual_exception"]["correct_cases"]
    rag_exception = rag["summary"]["by_expected_class"]["contextual_exception"]["correct_cases"]
    full_suite = len(local_ids) == 12
    rag_gate = rag.get("absolute_gate", {}).get("passed") is True
    comparison_passed = (
        full_suite
        and rag_gate
        and rag_correct - local_correct >= 2
        and rag_exception - local_exception >= 2
    )
    comparison = {
        "schema_version": COMPARISON_SCHEMA,
        "dataset_id": local["dataset_id"],
        "split": local["split"],
        "manifest_sha256": local["manifest_sha256"],
        "freeze_sha256": local["freeze_sha256"],
        "execution_sha256": local["execution_sha256"],
        "code_commit": local["code_commit"],
        "chat_model": local_config["chat_model"],
        "repeats": local_config["repeats"],
        "local_report_sha256": local_sha256,
        "rag_report_sha256": rag_sha256,
        "gains": {
            "verdict_correct_cases": rag_correct - local_correct,
            "contextual_exception_correct_cases": rag_exception - local_exception,
            "minimal_citation_coverage_cases": rag["summary"]["minimal_citation_coverage_cases_all_repeats"] - local["summary"]["minimal_citation_coverage_cases_all_repeats"],
        },
        "gate": {
            "evaluable": full_suite,
            "passed": comparison_passed if full_suite else None,
            "policy": "frozen-readme-v1",
            "reason": None if full_suite else "full_12_case_suite_required",
        },
        "claim_scope": "paired frozen-suite A/B only",
    }
    _reject_sensitive_report(comparison)
    return comparison


def _load_production_modules() -> dict[str, Any]:
    # app.db creates its legacy global engine at import time. Import from an
    # empty temporary cwd so Settings' relative `.env` source cannot discover
    # the repository's ignored credential file. Runtime settings below are
    # always built explicitly with `_env_file=None`.
    root_text = str(ROOT)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    original = Path.cwd()
    with tempfile.TemporaryDirectory(prefix="loreguard-eval-import-") as temporary:
        os.chdir(temporary)
        try:
            modules = {
                name: importlib.import_module(name)
                for name in (
                    "app.config",
                    "app.db",
                    "app.domain",
                    "app.embeddings",
                    "app.evidence_chunks",
                    "app.evidence_rag",
                    "app.issue_evidence_review",
                    "app.provider",
                    "app.usage",
                )
            }
        finally:
            os.chdir(original)
    return modules


def _read_secret_file(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if resolved.name == ".env" or resolved.name.startswith(".env."):
        raise EvalContractError(".env files are forbidden for this runner")
    raw = _read_bytes(resolved)
    try:
        value = raw.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise EvalContractError("secret file encoding is invalid") from None
    if not value or "\n" in value or len(value) > 4_096:
        raise EvalContractError("secret file content is invalid")
    return value


def _read_secret_env(name: str) -> str:
    """Read one explicitly named process variable without loading dotenv files."""
    if (
        not isinstance(name, str)
        or not 1 <= len(name) <= 64
        or not (name[0].isupper() or name[0] == "_")
        or any(
            not (character.isupper() or character.isdigit() or character == "_")
            for character in name
        )
    ):
        raise EvalContractError("secret environment variable name is invalid")
    value = os.environ.get(name)
    if value is None:
        raise EvalContractError("required secret environment variable is unavailable")
    value = value.strip()
    if not value or "\n" in value or "\r" in value or len(value) > 4_096:
        raise EvalContractError("secret environment variable content is invalid")
    return value


def _read_secret_source(path: Path | None, environment_name: str | None) -> str | None:
    if path is not None and environment_name is not None:
        raise EvalContractError("secret sources are mutually exclusive")
    if path is not None:
        return _read_secret_file(path)
    if environment_name is not None:
        return _read_secret_env(environment_name)
    return None


class ProductionEvaluationExecutor:
    def __init__(
        self,
        *,
        chat_api_key: str,
        chat_base_url: str,
        chat_model: str,
        embedding_api_key: str | None,
        embedding_base_url: str | None,
        embedding_model: str | None,
        embedding_revision: str | None,
        embedding_deployment: str | None,
        embedding_namespace: str | None,
        embedding_dimensions: int | None,
        database_url: str | None,
        configuration: RuntimeConfiguration,
    ) -> None:
        modules = _load_production_modules()
        self._m = modules
        Settings = modules["app.config"].Settings
        settings_values: dict[str, Any] = {
            "enable_model_extraction": True,
            "enable_issue_evidence_review": True,
            "openai_api_key": chat_api_key,
            "openai_base_url": chat_base_url,
            "openai_model": chat_model,
            "provider_thinking_mode": configuration.thinking_mode,
            "provider_timeout_seconds": configuration.timeout_seconds,
            "provider_total_deadline_seconds": configuration.total_deadline_seconds,
            "provider_max_attempts": 1,
            "issue_evidence_review_max_issues": 1,
            "issue_evidence_review_top_k": configuration.top_k,
            "issue_evidence_review_batch_size": 1,
            "issue_evidence_review_token_budget": configuration.token_budget,
            "issue_evidence_review_timeout_seconds": configuration.timeout_seconds,
            "issue_evidence_review_total_deadline_seconds": configuration.total_deadline_seconds,
        }
        if configuration.mode == "rag-evidence":
            if not all((embedding_api_key, embedding_base_url, embedding_model, embedding_revision, embedding_deployment, embedding_namespace, embedding_dimensions, database_url)):
                raise EvalContractError("RAG runtime configuration is incomplete")
            settings_values.update(
                {
                    "database_url": database_url,
                    "enable_embeddings": True,
                    "embedding_api_key": embedding_api_key,
                    "embedding_base_url": embedding_base_url,
                    "embedding_model": embedding_model,
                    "embedding_model_revision": embedding_revision,
                    "embedding_deployment_fingerprint": embedding_deployment,
                    "embedding_profile_namespace": embedding_namespace,
                    "embedding_dimensions": embedding_dimensions,
                }
            )
            if configuration.embedding_allow_insecure_http:
                settings_values["embedding_allow_insecure_http"] = True
        self._settings = Settings(_env_file=None, **settings_values)
        self._chat = modules["app.provider"].OpenAICompatibleProvider(self._settings)
        self._engine = None
        self._Session = None
        if configuration.mode == "rag-evidence":
            sqlalchemy = importlib.import_module("sqlalchemy")
            orm = importlib.import_module("sqlalchemy.orm")
            self._engine = sqlalchemy.create_engine(database_url, pool_pre_ping=True)
            if self._engine.dialect.name != "postgresql":
                raise EvalContractError("RAG evaluation requires PostgreSQL with pgvector")
            self._Session = orm.sessionmaker(bind=self._engine, expire_on_commit=False)
            with self._engine.connect() as connection:
                revision = connection.execute(sqlalchemy.text("SELECT version_num FROM alembic_version")).scalar_one_or_none()
            if revision != EXPECTED_HEAD_REVISION:
                raise EvalContractError("evaluation database migration is not current")
        self._loaded_projects: set[str] = set()
        profile_metadata: dict[str, Any] = {"kind": "local-context"}
        if configuration.mode == "rag-evidence":
            provider = modules["app.embeddings"].OpenAICompatibleEmbeddingProvider(self._settings)
            profile = provider.profile
            profile_metadata = {
                "kind": "real-embedding-pgvector",
                "canonical_profile_id": profile.profile_id,
                "model_identifier": profile.model_identifier,
                "model_revision": profile.model_revision,
                "deployment_fingerprint": profile.deployment_fingerprint,
                "document_transform_identity": profile.document_transform_identity,
                "query_transform_identity": profile.query_transform_identity,
                "dimensions": profile.dimensions,
                "normalized": profile.normalized,
            }
        self.metadata = {
            "backend": "production-issue-evidence-review",
            "parser_contract": "issue-evidence-review-v1",
            "citation_validation": "production-strict",
            "embedding": profile_metadata,
        }

    def close(self) -> None:
        if self._engine is not None:
            self._engine.dispose()

    def _issue(self, case: dict[str, Any]):
        domain = self._m["app.domain"]
        candidate = case["candidate_issue"]
        category_map = {
            "knowledge_overreach": "knowledge_without_acquisition",
            "object_state_conflict": "item_ownership",
            "location_conflict": "location_collision",
            "world_rule_conflict": "world_rule_conflict",
            "authorization_conflict": "world_rule_conflict",
            "character_state_conflict": "fact_conflict",
        }
        category = category_map.get(candidate["issue_type"], "fact_conflict")
        evidence = [
            domain.EvidenceSpan(
                document_id=row["document_id"],
                document_name=row["document_id"],
                line_start=row["line_start"],
                line_end=row["line_end"],
                text=row["text"],
            )
            for row in case["current_evidence"]
        ]
        return domain.ConsistencyIssue(
            category=domain.IssueCategory(category),
            severity=domain.Severity(candidate["severity"]),
            confidence=1.0,
            title=candidate["summary"],
            explanation=candidate["detector_basis"],
            evidence=evidence,
            suggestion="复核当前候选问题的证据充分性。",
            metadata={"rule_category": candidate["rule_category"]},
        )

    @staticmethod
    def _safe_calls(value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        calls = []
        for row in value[:16]:
            if isinstance(row, dict) and not any(str(key).casefold() in _SENSITIVE_REPORT_KEYS for key in row):
                calls.append(dict(row))
        return calls

    @staticmethod
    def _citations(annotation: dict[str, Any]) -> list[dict[str, Any]]:
        result = []
        for row in annotation.get("consumed_evidence", []):
            if not isinstance(row, dict) or row.get("cited") is not True:
                continue
            result.append(
                {
                    "project_id": row.get("project_id"),
                    "document_id": row.get("document_id"),
                    "document_version": row.get("document_version"),
                    "content_sha256": row.get("content_sha256"),
                    "line_start": row.get("line_start"),
                    "line_end": row.get("line_end"),
                }
            )
        return result

    def _local(self, case: dict[str, Any], issue: Any) -> dict[str, Any]:
        review = self._m["app.issue_evidence_review"]
        domain = self._m["app.domain"]
        provider_module = self._m["app.provider"]
        usage = self._m["app.usage"]
        evidence_by_label: dict[str, dict[str, Any]] = {}
        prompt_evidence = []
        for index, row in enumerate(case["current_evidence"], start=1):
            label = f"E{index:02d}"
            evidence_by_label[label] = {
                "citation_id": label,
                "chunk_id": "local-" + hashlib.sha256((case["case_id"] + label).encode()).hexdigest(),
                "project_id": row["project_id"],
                "document_id": row["document_id"],
                "document_version": row["document_version"],
                "content_sha256": row["content_sha256"],
                "chunker_version": "local-current-evidence-v1",
                "profile_id": "local-context",
                "line_start": row["line_start"],
                "line_end": row["line_end"],
                "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                "provided_chars": len(row["text"]),
            }
            prompt_evidence.append(
                {
                    "citation_id": label,
                    "document_id": row["document_id"],
                    "document_version": row["document_version"],
                    "line_start": row["line_start"],
                    "line_end": row["line_end"],
                    "text": row["text"],
                }
            )
        prepared = review._PreparedIssue(
            issue_ref="I01",
            issue_id=str(issue.id),
            payload={
                "issue_ref": "I01",
                "category": issue.category.value,
                "title": issue.title,
                "explanation": issue.explanation,
                "rule_evidence": [
                    {"document_id": span.document_id, "line_start": span.line_start, "line_end": span.line_end, "text": span.text}
                    for span in issue.evidence
                ],
            },
            evidence_by_label=evidence_by_label,
            prompt_evidence=prompt_evidence,
            retrieval={
                "strategy": "local-context-only",
                "mode": "local_only",
                "reason": None,
                "profile_id": "local-context",
                "chunker_version": "local-current-evidence-v1",
                "candidate_count": len(prompt_evidence),
                "result_count": len(prompt_evidence),
                "elapsed_ms": 0,
            },
        )
        user_prompt = review._batch_prompt([prepared])
        estimated_charge = usage.estimate_issue_evidence_review_tokens(
            review.ISSUE_EVIDENCE_REVIEW_SYSTEM_PROMPT,
            user_prompt,
            completion_reserve=self._settings.issue_evidence_review_max_completion_tokens,
        )
        if estimated_charge > self._settings.issue_evidence_review_token_budget:
            return {
                "annotation": None,
                "diagnostics": {
                    "outcome": "degraded",
                    "reason_codes": ["token_budget"],
                    "provider_calls": [],
                },
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "charged_tokens": 0,
            }
        try:
            response = self._chat.complete(
                review.ISSUE_EVIDENCE_REVIEW_SYSTEM_PROMPT, user_prompt
            )
        except provider_module.ProviderError as exc:
            call = domain.ProviderCallDiagnostics.from_telemetry(
                getattr(exc, "telemetry", None),
                succeeded=False,
                purpose="evidence_review",
            )
            category = review._safe_provider_category(getattr(exc, "category", None))
            return {
                "annotation": None,
                "diagnostics": {
                    "outcome": "degraded",
                    "reason_codes": [category],
                    "provider_calls": [
                        call.safe_dict()
                        if call is not None
                        else {
                            "status": "failure",
                            "category": category,
                            "purpose": "evidence_review",
                        }
                    ],
                },
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "charged_tokens": estimated_charge,
            }
        call = domain.ProviderCallDiagnostics.from_telemetry(response.telemetry, succeeded=True, purpose="evidence_review")
        safe_call = call.safe_dict() if call else {"status": "success", "purpose": "evidence_review"}
        charged_tokens = max(
            estimated_charge, response.prompt_tokens + response.completion_tokens
        )
        try:
            parsed = review._parse_batch(response.text, [prepared])
        except (ValueError, TypeError):
            return {
                "annotation": None,
                "diagnostics": {
                    "outcome": "degraded",
                    "reason_codes": ["invalid_model_response"],
                    "provider_calls": [safe_call],
                },
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "charged_tokens": charged_tokens,
            }
        annotation = review._annotation(prepared, parsed[0], safe_call)
        return {
            "annotation": annotation,
            "diagnostics": {"outcome": "completed", "reason_codes": [], "provider_calls": [annotation["provider_call"]]},
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "charged_tokens": charged_tokens,
        }

    def _seed_and_documents(self, case: dict[str, Any]) -> tuple[list[Any], list[Any]]:
        db = self._m["app.db"]
        rag = self._m["app.evidence_rag"]
        chunks = self._m["app.evidence_chunks"]
        embeddings = self._m["app.embeddings"]
        assert self._Session is not None
        project_id = case["project_id"]
        if project_id not in self._loaded_projects:
            active_by_id: dict[str, dict[str, Any]] = {}
            for row in case["documents"]:
                current = active_by_id.get(row["document_id"])
                if current is None or row["active"]:
                    active_by_id[row["document_id"]] = row
            with self._Session() as session:
                if session.get(db.ProjectRow, project_id) is None:
                    session.add(db.ProjectRow(id=project_id, name=project_id, description="frozen issue review evaluation"))
                for document_id, row in active_by_id.items():
                    existing = session.get(db.DocumentRow, document_id)
                    if existing is None:
                        session.add(db.DocumentRow(id=document_id, project_id=project_id, name=document_id, content=row["content"], version=row["document_version"], active=row["active"]))
                    elif existing.project_id != project_id:
                        raise EvalContractError("evaluation database document scope conflicts")
                session.commit()
            provider = embeddings.OpenAICompatibleEmbeddingProvider(self._settings)
            coordinator = rag.EvidenceIndexCoordinator(session_factory=self._Session, provider=provider)
            # Index forbidden snapshots separately. They are physically present
            # in the same store but never included in the authorized result
            # passed to the production reviewer/retriever.
            for row in case["documents"]:
                if row["active"] and row["authorized_for_review"]:
                    continue
                document = rag.EvidenceDocument(
                    snapshot=chunks.SnapshotDocumentKey(
                        project_id=row["project_id"], document_id=row["document_id"], document_version=row["document_version"], content_sha256=row["content_sha256"]
                    ),
                    content=row["content"],
                )
                indexed = coordinator.ensure_index([document])
                if not indexed.complete:
                    raise EvalContractError("forbidden-snapshot isolation fixture could not be indexed")
            self._loaded_projects.add(project_id)
        all_documents = []
        allowed_documents = []
        allowed_set = {(row["project_id"], row["document_id"], row["document_version"], row["content_sha256"]) for row in case["allowed_snapshots"]}
        for row in case["documents"]:
            document = rag.EvidenceDocument(
                snapshot=chunks.SnapshotDocumentKey(project_id=row["project_id"], document_id=row["document_id"], document_version=row["document_version"], content_sha256=row["content_sha256"]),
                content=row["content"],
            )
            all_documents.append(document)
            identity = (row["project_id"], row["document_id"], row["document_version"], row["content_sha256"])
            if identity in allowed_set:
                allowed_documents.append(document)
        return all_documents, allowed_documents

    def _rag(self, case: dict[str, Any], issue: Any) -> dict[str, Any]:
        review = self._m["app.issue_evidence_review"]
        embeddings = self._m["app.embeddings"]
        _, allowed_documents = self._seed_and_documents(case)
        assert self._Session is not None
        reviewer = review.IssueEvidenceReviewer(
            session_factory=self._Session,
            settings=self._settings,
            embedding_provider=embeddings.OpenAICompatibleEmbeddingProvider(self._settings),
            chat_provider=self._chat,
        )
        result = reviewer.review(documents=allowed_documents, issues=[issue], remaining_run_tokens=self._settings.issue_evidence_review_token_budget)
        annotation = result.annotations.get(str(issue.id))
        return {
            "annotation": annotation,
            "diagnostics": result.diagnostics,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "charged_tokens": result.charged_tokens,
        }

    def execute(self, case: dict[str, Any], mode: Mode) -> dict[str, Any]:
        started = time.monotonic()
        issue = self._issue(case)
        try:
            raw = self._local(case, issue) if mode == "local-context" else self._rag(case, issue)
        except Exception as exc:
            provider = self._m["app.provider"]
            category = exc.category if isinstance(exc, provider.ProviderError) else "execution_failure"
            return {
                "status": "failed",
                "verdict": None,
                "citations": [],
                "review_outcome": "failed",
                "failure_category": category if isinstance(category, str) else "execution_failure",
                "retrieval_mode": None,
                "retrieval_strategy": None,
                "latency_ms": max(0, int((time.monotonic() - started) * 1000)),
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "charged_tokens": 0,
                "provider_calls": [],
            }
        annotation = raw.get("annotation")
        diagnostics = raw.get("diagnostics") if isinstance(raw.get("diagnostics"), dict) else {}
        outcome = diagnostics.get("outcome") if isinstance(diagnostics.get("outcome"), str) else "failed"
        provider_calls = self._safe_calls(diagnostics.get("provider_calls"))
        retrieval = annotation.get("retrieval", {}) if isinstance(annotation, dict) else {}
        success = isinstance(annotation, dict) and annotation.get("verdict") in VERDICTS and outcome == "completed"
        reasons = diagnostics.get("reason_codes")
        failure = None
        if not success:
            failure = reasons[0] if isinstance(reasons, list) and reasons and isinstance(reasons[0], str) else "review_degraded"
        return {
            "status": "success" if success else "degraded",
            "verdict": annotation.get("verdict") if success else None,
            "citations": self._citations(annotation) if success else [],
            "review_outcome": outcome,
            "failure_category": failure,
            "retrieval_mode": retrieval.get("mode") if isinstance(retrieval.get("mode"), str) else ("local_only" if mode == "local-context" else None),
            "retrieval_strategy": retrieval.get("strategy") if isinstance(retrieval.get("strategy"), str) else ("local-context-only" if mode == "local-context" else None),
            "latency_ms": max(0, int((time.monotonic() - started) * 1000)),
            "prompt_tokens": max(0, int(raw.get("prompt_tokens") or 0)),
            "completion_tokens": max(0, int(raw.get("completion_tokens") or 0)),
            "charged_tokens": max(0, int(raw.get("charged_tokens") or 0)),
            "provider_calls": provider_calls,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen Issue Review v1 A/B evaluator")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--split", choices=sorted(PACKAGE_SPLITS), required=True)
    prepare.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    prepare.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    prepare.add_argument("--output", type=Path, required=True)

    run = commands.add_parser("run")
    run.add_argument("--execution", type=Path, required=True)
    run.add_argument("--execution-sha256", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--mode", choices=sorted(MODES), required=True)
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--chat-base-url", required=True)
    chat_secret = run.add_mutually_exclusive_group(required=True)
    chat_secret.add_argument("--chat-api-key-file", type=Path)
    chat_secret.add_argument("--chat-api-key-env")
    run.add_argument("--chat-model", required=True)
    run.add_argument("--thinking-mode", choices=("disabled", "enabled"))
    run.add_argument("--top-k", type=int, default=6)
    run.add_argument("--token-budget", type=int, default=6_000)
    run.add_argument("--timeout-seconds", type=float, default=20.0)
    run.add_argument("--total-deadline-seconds", type=float, default=45.0)
    run.add_argument("--embedding-base-url")
    embedding_secret = run.add_mutually_exclusive_group()
    embedding_secret.add_argument("--embedding-api-key-file", type=Path)
    embedding_secret.add_argument("--embedding-api-key-env")
    run.add_argument("--embedding-model")
    run.add_argument("--embedding-revision")
    run.add_argument("--embedding-deployment")
    run.add_argument("--embedding-namespace")
    run.add_argument("--embedding-dimensions", type=int)
    run.add_argument("--embedding-allow-insecure-http", action="store_true")
    database_secret = run.add_mutually_exclusive_group()
    database_secret.add_argument("--database-url-file", type=Path)
    database_secret.add_argument("--database-url-env")
    run.add_argument("--execute", action="store_true")

    score = commands.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--prediction-sha256", required=True)
    score.add_argument("--execution", type=Path, required=True)
    score.add_argument("--execution-sha256", required=True)
    score.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    score.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    score.add_argument("--output", type=Path, required=True)

    compare = commands.add_parser("compare")
    compare.add_argument("--local-report", type=Path, required=True)
    compare.add_argument("--local-report-sha256", required=True)
    compare.add_argument("--rag-report", type=Path, required=True)
    compare.add_argument("--rag-report-sha256", required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        package = prepare_execution_package(manifest_path=args.manifest, freeze_path=args.freeze, split=args.split)
        digest = _atomic_write_json(args.output, package, protected=(args.manifest, args.freeze))
    elif args.command == "run":
        if not args.execute:
            raise EvalContractError("real evaluation requires explicit --execute")
        commit = _git_commit_and_clean()
        package = _read_json(args.execution)
        execution_raw_sha = _sha256_file(args.execution)
        if execution_raw_sha != args.execution_sha256:
            raise EvalContractError("execution artifact hash does not match")
        configuration = RuntimeConfiguration(
            mode=args.mode,
            repeats=_positive_int(args.repeats, maximum=MAX_REPEATS),
            chat_model=_identifier(args.chat_model, 255),
            chat_endpoint_fingerprint=_endpoint_fingerprint(args.chat_base_url),
            thinking_mode=args.thinking_mode,
            top_k=_positive_int(args.top_k, maximum=6),
            token_budget=_positive_int(args.token_budget, maximum=6_000),
            timeout_seconds=_positive_number(args.timeout_seconds, maximum=20),
            total_deadline_seconds=_positive_number(args.total_deadline_seconds, maximum=45),
            embedding_allow_insecure_http=args.embedding_allow_insecure_http,
        )
        chat_api_key = _read_secret_source(args.chat_api_key_file, args.chat_api_key_env)
        embedding_key = _read_secret_source(args.embedding_api_key_file, args.embedding_api_key_env)
        database_url = _read_secret_source(args.database_url_file, args.database_url_env)
        if chat_api_key is None:
            raise EvalContractError("chat credential is unavailable")
        executor = ProductionEvaluationExecutor(
            chat_api_key=chat_api_key, chat_base_url=args.chat_base_url, chat_model=args.chat_model,
            embedding_api_key=embedding_key, embedding_base_url=args.embedding_base_url, embedding_model=args.embedding_model,
            embedding_revision=args.embedding_revision, embedding_deployment=args.embedding_deployment,
            embedding_namespace=args.embedding_namespace, embedding_dimensions=args.embedding_dimensions,
            database_url=database_url, configuration=configuration,
        )
        try:
            predictions = run_execution_package(package, execution_sha256=args.execution_sha256, configuration=configuration, executor=executor, code_commit=commit)
        finally:
            executor.close()
        digest = _atomic_write_json(args.output, predictions, protected=(args.execution,))
    elif args.command == "score":
        report = score_predictions(
            prediction_path=args.predictions, prediction_sha256=args.prediction_sha256,
            execution_path=args.execution, execution_sha256=args.execution_sha256,
            manifest_path=args.manifest, freeze_path=args.freeze,
        )
        digest = _atomic_write_json(args.output, report, protected=(args.predictions, args.execution, args.manifest, args.freeze))
    else:
        comparison = compare_reports(
            local_path=args.local_report, local_sha256=args.local_report_sha256,
            rag_path=args.rag_report, rag_sha256=args.rag_report_sha256,
        )
        digest = _atomic_write_json(args.output, comparison, protected=(args.local_report, args.rag_report))
    print(digest)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvalContractError as exc:
        print(f"evaluation contract error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
