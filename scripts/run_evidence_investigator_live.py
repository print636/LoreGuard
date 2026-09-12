"""Run the frozen Evidence Investigator fixture through the public HTTP API.

This is evaluator-side code.  Ground truth is loaded only for local scoring
after each service run has finished; only source documents and their production
role/scope metadata are sent to LoreGuard.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "evaluation" / "evidence_investigator_live"
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "evidence-investigator-live"

ARTIFACT_SCHEMA = "evidence-investigator-live-http-v5"
HISTORICAL_ARTIFACT_SCHEMAS = frozenset(
    {
        "evidence-investigator-live-http-v2",
        "evidence-investigator-live-http-v3",
        "evidence-investigator-live-http-v4",
    }
)
RUNTIME_PROVENANCE_SCHEMA = "loreguard-runtime-provenance-v2"
DEV_PAIR_SCHEMA = "evidence-investigator-live-dev-pair-v3"
DATASET_ID = "evidence-investigator-live-v2"
PINNED_MANIFEST_SHA256 = "64ded0988cb70ea591548facfc26f4867e39818d1b868a17dbadde402fc6d316"
PINNED_FREEZE_SHA256 = "ed86c164004e62d018cee014170b2b1f0d00bb91fb378d11ffcf73e7051ecf70"
MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_SOURCE_BYTES = 512 * 1024
MAX_BUNDLE_FILES = 512
MAX_BUNDLE_FILE_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_TOTAL_BYTES = 32 * 1024 * 1024
MAX_SAFE_COUNTER = (1 << 63) - 1
TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})
ACTIVE_STATUSES = frozenset({"queued", "running"})
ISSUE_CATEGORIES = frozenset(
    {
        "fact_conflict",
        "location_collision",
        "knowledge_without_acquisition",
        "item_ownership",
        "world_rule_conflict",
    }
)
DOCUMENT_ROLES = frozenset({"canon", "character_profile", "chapter", "reference"})
SAFE_INVESTIGATOR_OUTCOMES = frozenset({"completed", "skipped", "degraded"})
SAFE_LOOP_OUTCOMES = frozenset({"completed", "degraded"})
SAFE_INVESTIGATOR_REASON_CODES = frozenset(
    {
        "completed",
        "no_seeds",
        "chat_not_configured",
        "token_budget",
        "deadline",
        "embedding_not_configured",
        "embedding_input_rejected",
        "embedding_response_invalid",
        "embedding_retry_exhausted",
        "embedding_provider_failed",
        "concurrent_write_incomplete",
        "vector_search_unavailable",
        "vector_scope_invalid",
        "hybrid_unavailable",
        "index_incomplete",
        "index_invalid",
        "retrieval_invalid",
        "retrieval_failed",
        "loop_degraded",
        "promotion_failed",
        "internal_failure",
    }
)
SAFE_LOOP_REASON_CODES = frozenset(
    {
        "completed",
        "usage_unavailable",
        "prompt_too_large",
        "provider_not_configured",
        "provider_rate_limit",
        "provider_timeout",
        "provider_rejected",
        "provider_contract_invalid",
        "provider_unavailable",
        "no_tool_call",
        "multiple_tool_calls",
        "unknown_tool",
        "invalid_tool_arguments",
        "cross_seed",
        "retrieval_failed",
        "internal_failure",
        "accepted",
        "explicit_abstain",
        "invalid_state",
        "unknown_seed",
        "cross_run",
        "snapshot_mismatch",
        "evidence_hash_mismatch",
        "evidence_range",
        "unknown_result_ref",
        "unknown_span_ref",
        "candidate_kind_forbidden",
        "repeated_action",
        "anchor_evidence_reused",
        "repeated_query",
        "no_progress",
        "round_budget",
        "tool_budget",
        "search_budget",
        "read_budget",
        "result_budget",
        "span_budget",
        "token_budget",
        "deadline",
        "unprocessed_seed",
    }
)
SAFE_RAG_INDEX_OUTCOMES = frozenset(
    {"complete", "provider_unavailable", "write_conflict", "unavailable"}
)
SAFE_RAG_INDEX_REASONS = frozenset(
    {
        "embedding_not_configured",
        "embedding_input_rejected",
        "embedding_response_invalid",
        "embedding_retry_exhausted",
        "embedding_provider_failed",
        "concurrent_write_incomplete",
        "vector_search_unavailable",
        "vector_scope_invalid",
        "hybrid_unavailable",
        "index_incomplete",
        "index_invalid",
        "retrieval_invalid",
        "internal_failure",
    }
)
SAFE_RAG_MODES = frozenset({"hybrid", "lexical_only", "dense_only", "unavailable"})
SAFE_RAG_STRATEGIES = frozenset(
    {"keyword-only", "dense-only", "keyword+dense-rrf", "keyword+vector+entity-rrf"}
)
SAFE_CLASSIFICATIONS = frozenset(
    {
        "positive_added_issue",
        "active_abstain",
        "promotion_rejection",
        "agent_protocol_failure",
        "isolation_failure",
        "provider_failure",
        "timeout_or_budget",
        "wrong_candidate",
        "runner_failure",
    }
)
DISQUALIFYING_EXECUTION_CLASSIFICATIONS = frozenset(
    {
        "agent_protocol_failure",
        "isolation_failure",
        "provider_failure",
        "timeout_or_budget",
        "runner_failure",
    }
)
DEV_CASE_IDS = frozenset(f"EIL-D-{index:02d}" for index in range(1, 9))
SAFE_REJECTION_REASONS = frozenset(
    {
        "invalid_envelope",
        "unknown_seed",
        "candidate_budget",
        "candidate_payload_too_large",
        "candidate_fields_forbidden",
        "candidate_shape_invalid",
        "candidate_not_grounded",
        "candidate_join_mismatch",
        "candidate_evidence_unauthorized",
        "candidate_evidence_reused",
        "candidate_semantics_rejected",
        "candidate_no_rule_conflict",
        "candidate_duplicate",
    }
)
SAFE_PROVIDER_CATEGORIES = frozenset(
    {
        "success",
        "provider",
        "not_configured",
        "rate_limit",
        "upstream_5xx",
        "unauthorized",
        "forbidden",
        "nonretry_http",
        "body_json",
        "response_shape",
        "empty_content",
        "usage_shape",
        "truncated",
        "content_json",
        "connect_timeout",
        "read_timeout",
        "transport",
        "response_too_large",
        "tool_response_json",
        "tool_response_shape",
        "tool_response_finish_reason",
        "tool_calls_shape",
        "tool_calls_too_many",
        "tool_call_shape",
        "tool_call_id_invalid",
        "tool_call_id_duplicate",
        "tool_call_name_invalid",
        "tool_call_unknown",
        "tool_choice_mismatch",
        "tool_call_arguments_json",
        "tool_call_arguments_shape",
        "tool_call_arguments_too_large",
        "tool_calls_missing",
        "tool_request_rejected",
        "usage_unavailable",
        "provider_contract_invalid",
    }
)
TIME_OR_BUDGET_REASONS = frozenset(
    {
        "deadline",
        "token_budget",
        "prompt_too_large",
        "round_budget",
        "tool_budget",
        "search_budget",
        "read_budget",
        "result_budget",
        "span_budget",
        "provider_timeout",
    }
)
PROVIDER_FAILURE_REASONS = frozenset(
    {
        "chat_not_configured",
        "provider_not_configured",
        "provider_rate_limit",
        "provider_rejected",
        "provider_contract_invalid",
        "provider_unavailable",
        "usage_unavailable",
    }
)
AGENT_PROTOCOL_FAILURE_REASONS = frozenset(
    {
        "no_tool_call",
        "multiple_tool_calls",
        "unknown_tool",
        "invalid_tool_arguments",
        "cross_seed",
        "unknown_seed",
        "unknown_result_ref",
        "unknown_span_ref",
        "candidate_kind_forbidden",
        "anchor_evidence_reused",
        "repeated_action",
        "repeated_query",
        "no_progress",
        "unprocessed_seed",
    }
)
SAFE_FAILURE_CODES = frozenset(
    {
        "confirmation_required",
        "fixture_integrity_failed",
        "service_not_ready",
        "configuration_invalid",
        "http_client_error",
        "http_server_error",
        "http_rate_limit_exhausted",
        "http_transport_error",
        "http_redirect_rejected",
        "response_too_large",
        "response_json_invalid",
        "response_schema_invalid",
        "run_timeout",
        "run_status_invalid",
        "artifact_exists",
        "internal_runner_error",
    }
)
_CAPABILITY_KEYS = frozenset(
    {
        "model_extraction",
        "issue_evidence_review",
        "record_repair_agent",
        "evidence_investigator",
        "embeddings",
    }
)
_INVESTIGATOR_INTEGER_LIMIT_KEYS = frozenset(
    {
        "max_seeds",
        "max_decision_rounds",
        "max_tool_calls",
        "max_searches",
        "max_reads",
        "max_results",
        "max_read_lines",
        "max_span_chars",
        "token_budget",
        "max_agent_input_bytes",
        "max_completion_tokens",
        "max_response_bytes",
        "provider_attempts_per_decision",
        "top_k",
        "branch_limit",
        "embedding_max_input_chars",
        "daily_token_budget",
    }
)
_INVESTIGATOR_NUMBER_LIMIT_KEYS = frozenset(
    {"provider_call_timeout_seconds", "total_deadline_seconds"}
)
_INVESTIGATOR_LIMIT_KEYS = (
    _INVESTIGATOR_INTEGER_LIMIT_KEYS
    | _INVESTIGATOR_NUMBER_LIMIT_KEYS
    | {"require_hybrid"}
)
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_GIT_REVISION = re.compile(r"^[a-f0-9]{40,64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")
_SAFE_TRACE_SCHEMA = "evidence_investigator_safe_trace_v1"
_SAFE_TRACE_ACTIONS = frozenset(
    {"SEARCH_EVIDENCE", "READ_SPAN", "SUBMIT_VERDICT", "ABSTAIN"}
)
_SAFE_TRACE_PHASES = frozenset({"search", "read", "verdict"})
_SAFE_TRACE_ABSTAIN_REASONS = frozenset(
    {
        "no_relevant_evidence",
        "insufficient_evidence",
        "ambiguous_source",
        "contextual_exception",
        "no_rule_validated_conflict",
    }
)
_SAFE_TRACE_CANDIDATE_FIELDS = {
    "fact": frozenset({"subject", "predicate", "value", "time"}),
    "event": frozenset({"time", "location", "participants"}),
    "knows": frozenset({"character", "fact", "time"}),
    "claims_knows": frozenset({"character", "fact", "time"}),
    "item": frozenset({"item", "owner", "time"}),
    "uses": frozenset({"item", "user", "time"}),
    "world_rule": frozenset({"key", "value"}),
    "world_assert": frozenset({"key", "value", "actor", "time"}),
}
_SAFE_TRACE_VALIDATION_OUTCOMES = frozenset(
    {"verified", "invalid", "unavailable"}
)
_TRACE_ATTRIBUTION_KEYS = frozenset(
    {"validation_outcome", "action_counts"}
)
_MAX_TRACE_ACTIONS = 64
_MAX_TRACE_SEED_ORDINAL = 64
_MAX_TRACE_RESULT_COUNT = 512
_MAX_TRACE_READ_LINES = 20
_MAX_TRACE_LINE_NUMBER = 10_000_000
_PRIVACY_BOUNDARY = {
    "source_bodies_persisted": False,
    "provider_payloads_persisted": False,
    "credentials_persisted": False,
    "service_address_persisted": False,
    "decision_trace_actions_persisted": False,
    "document_pseudonyms_persisted": False,
    "evidence_line_ranges_persisted": False,
    "ground_truth_coordinates_persisted": False,
}
_DIAGNOSTIC_BOUNDARY = (
    "Raw Evidence Investigator decision-trace actions are parsed only in evaluator "
    "memory and are discarded before export. The artifact retains only a strict "
    "validation outcome, fixed-enum action-count aggregate, and a developer-only "
    "nullable boolean indicating whether one READ_SPAN fully covered the target "
    "coordinate; null means unavailable or not applicable. It never "
    "retains document pseudonyms, line ranges, result ranks, candidate field names, "
    "source or query text, raw identifiers, or provider payloads. Runtime provenance "
    "exposes only a model alias, irreversible endpoint hash, content hash and bounded "
    "configuration; it never exposes credentials, a hostname, or a complete base URL."
)
_FORBIDDEN_ARTIFACT_KEYS = frozenset(
    {
        "actions",
        "document_ref_hash",
        "line_start",
        "line_end",
        "selected_result_rank",
        "field_names",
        "candidate_shapes",
        "candidate_count",
        "result_count",
        "provider_decision_index",
        "seed_ordinal",
        "phase",
        "overlaps_anchor",
        "covers_entire_result",
        "source_line_count",
        "kind",
        "reason",
        "read_target_evidence",
        "candidate",
        "lure_candidate",
        "document_id",
        "lines",
        "fields",
        "query",
        "provider_payload",
        "source_body",
    }
)
_DEV_ARTIFACT_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "split",
        "execution_source",
        "started_at",
        "completed_at",
        "fixture",
        "git",
        "safe_configuration",
        "safe_configuration_fingerprint",
        "reproducibility_fingerprint",
        "provenance_gate",
        "development_gate",
        "summary",
        "cases",
        "privacy_boundary",
        "diagnostic_boundary",
    }
)
_SAFE_CONFIGURATION_KEYS = frozenset(
    {
        "artifact_schema",
        "dataset_id",
        "split",
        "manifest_sha256",
        "freeze_sha256",
        "case_timeout_seconds",
        "request_timeout_seconds",
        "poll_interval_seconds",
        "runner_git",
        "runner_service_artifact_sha256",
        "service_observation",
        "service_reported_effective_limits",
        "worker_runtime_provenance_matches_api_cases",
        "rag_configuration_verified_cases",
    }
)
_PROVENANCE_GATE_KEYS = frozenset(
    {
        "passed",
        "reason_codes",
        "api_runtime_provenance",
        "runner_service_artifact_sha256",
        "worker_runtime_provenance_observed_cases",
        "worker_runtime_provenance_matches_api_cases",
        "rag_configuration_verified_cases",
        "case_count",
    }
)
_DEVELOPMENT_GATE_KEYS = frozenset(
    {"applicable", "passed", "checks", "reason_codes", "boundary"}
)
_CASE_KEYS = frozenset(
    {
        "case_id",
        "expected_decision",
        "expected_issue_category",
        "classification",
        "passed",
        "run_status",
        "project_ref_hash",
        "run_ref_hash",
        "case_wall_latency_ms",
        "issue_category_counts",
        "citation_scope_authorized",
        "target_evidence_match",
        "read_target_evidence_match",
        "decision_trace_attribution",
        "analysis_usage",
        "investigator",
        "runtime_provenance",
        "capability_isolation",
        "failure_code",
    }
)
_INVESTIGATOR_KEYS = frozenset(
    {
        "available",
        "enabled",
        "outcome",
        "reason_code",
        "seed_count",
        "loop",
        "promotion",
        "usage",
        "budget_preflight",
        "effective_limits",
        "rag",
    }
)
_LOOP_KEYS = frozenset(
    {
        "outcome",
        "reason_code",
        "provider_decision_calls",
        "tool_calls",
        "tool_call_count_basis",
        "searches",
        "reads",
        "recoverable_rejections",
        "completed_seeds",
        "abstained_seeds",
        "submitted_envelopes",
        "authorized_candidates",
    }
)
_PROMOTION_KEYS = frozenset(
    {"submitted_candidates", "accepted_candidates", "added_issues", "rejection_counts"}
)
_USAGE_KEYS = frozenset(
    {
        "reported_prompt_tokens",
        "reported_completion_tokens",
        "charged_tokens",
        "token_count_basis",
        "provider_category_counts",
    }
)
_BUDGET_PREFLIGHT_KEYS = frozenset(
    {
        "seed_count",
        "minimum_path_admissible",
        "minimum_required_rounds",
        "minimum_initial_reservation",
        "maximum_local_round_reservation",
        "maximum_local_run_reservation",
        "max_charged_tokens",
        "oversized_initial_prompts",
    }
)
_RAG_KEYS = frozenset(
    {
        "index_outcome",
        "index_reason",
        "profile_fingerprint",
        "chunker_fingerprint",
        "modes",
        "strategies",
        "total_retrievals",
    }
)
_ISOLATION_KEYS = frozenset(
    {
        "verified",
        "main_extraction",
        "issue_evidence_review",
        "aggregate_chat_usage_matches_investigator",
        "reason_codes",
    }
)


class LiveEvaluationError(ValueError):
    """Content-free runner failure safe to display or persist."""

    def __init__(self, code: str):
        self.code = code if code in SAFE_FAILURE_CODES else "internal_runner_error"
        super().__init__(self.code)


class HttpApiError(LiveEvaluationError):
    def __init__(self, code: str, *, status: int | None = None, retry_after: int = 0):
        super().__init__(code)
        self.status = status if type(status) is int and 100 <= status <= 599 else None
        self.retry_after = (
            retry_after if type(retry_after) is int and 0 <= retry_after <= 60 else 0
        )


class JsonApi(Protocol):
    def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float,
    ) -> Any: ...


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class UrllibJsonApi:
    """Small JSON client that never follows redirects or exposes response bodies."""

    def __init__(self, base_url: str):
        self._base_url = _normalize_base_url(base_url)
        self._opener = urllib.request.build_opener(_NoRedirect())

    def request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float,
    ) -> Any:
        if method not in {"GET", "POST"} or not path.startswith("/") or "?" in path:
            raise LiveEvaluationError("internal_runner_error")
        body = None
        headers = {"Accept": "application/json", "User-Agent": "LoreGuard-EIL-Eval/1"}
        if payload is not None:
            try:
                body = _canonical_json_bytes(payload)
            except (TypeError, ValueError, UnicodeError):
                raise LiveEvaluationError("internal_runner_error") from None
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self._base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with self._opener.open(request, timeout=timeout) as response:
                raw = response.read(MAX_JSON_BYTES + 1)
        except urllib.error.HTTPError as exc:
            retry_after = _retry_after_seconds(exc.headers.get("Retry-After"))
            if 300 <= exc.code <= 399:
                code = "http_redirect_rejected"
            elif exc.code == 429:
                code = "http_rate_limit_exhausted"
            elif 400 <= exc.code <= 499:
                code = "http_client_error"
            else:
                code = "http_server_error"
            raise HttpApiError(code, status=exc.code, retry_after=retry_after) from None
        except (OSError, TimeoutError, urllib.error.URLError):
            raise HttpApiError("http_transport_error") from None
        if len(raw) > MAX_JSON_BYTES:
            raise HttpApiError("response_too_large")
        try:
            return json.loads(
                raw.decode("utf-8"),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise HttpApiError("response_json_invalid") from None


@dataclass(frozen=True, slots=True)
class SourceDocument:
    logical_id: str
    filename: str
    content: str
    role: str
    story_scope: str
    line_count: int

    def upload_payload(self) -> dict[str, str]:
        # This exact allowlist is the model-facing boundary.  Ground truth is
        # deliberately held in a different object and cannot enter this dict.
        return {
            "name": self.filename,
            "content": self.content,
            "document_role": self.role,
            "story_scope": self.story_scope,
        }


@dataclass(frozen=True, slots=True)
class GroundTruth:
    decision: Literal["added_issue", "abstain"]
    issue_category: str
    added_issue_count: int
    allowed_evidence: tuple[tuple[str, int, int], ...]
    read_target_evidence: tuple[str, int, int] | None = None


@dataclass(frozen=True, slots=True)
class DecisionTraceRead:
    """One server-issued READ observation kept only in evaluator memory."""

    document_ref_hash: str
    line_start: int
    line_end: int


@dataclass(frozen=True, slots=True)
class ParsedDecisionTrace:
    """Content-free trace attribution; detailed reads are never serialized."""

    validation_outcome: Literal["verified", "invalid", "unavailable"]
    action_counts: tuple[tuple[str, int], ...] = ()
    reads: tuple[DecisionTraceRead, ...] = ()

    def artifact_summary(self) -> dict[str, Any]:
        return {
            "validation_outcome": self.validation_outcome,
            "action_counts": (
                dict(self.action_counts)
                if self.validation_outcome == "verified"
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class CasePlan:
    case_id: str
    split: Literal["dev", "holdout"]
    documents: tuple[SourceDocument, ...]
    ground_truth: GroundTruth


@dataclass(frozen=True, slots=True)
class VerifiedDataset:
    manifest: dict[str, Any]
    manifest_sha256: str
    freeze_sha256: str
    source_payloads: tuple[tuple[str, bytes], ...]


@dataclass(frozen=True, slots=True)
class RunnerOptions:
    split: Literal["dev", "holdout"]
    base_url: str
    artifact_path: Path
    case_timeout_seconds: float = 180.0
    request_timeout_seconds: float = 30.0
    poll_interval_seconds: float = 1.0
    confirm_live_provider: bool = False


def _reject_json_constant(_: str) -> None:
    raise ValueError("non-finite JSON value")


def _reject_duplicate_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _exact_object(value: Any, keys: frozenset[str] | set[str]) -> bool:
    return (
        type(value) is dict
        and all(type(key) is str for key in value)
        and set(value) == set(keys)
    )


def _canonical_json_bytes(value: Any) -> bytes:
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


def _local_service_artifact_sha256(root: Path = ROOT) -> str | None:
    """Independently fingerprint the local service bundle used for evaluation."""

    try:
        resolved_root = root.resolve(strict=True)
        app_root = (resolved_root / "app").resolve(strict=True)
        files = sorted(
            (path for path in app_root.rglob("*.py") if path.is_file()),
            key=lambda path: path.relative_to(resolved_root).as_posix(),
        )
        requirements = resolved_root / "requirements.txt"
        if requirements.is_file():
            files.append(requirements)
        if not 1 <= len(files) <= MAX_BUNDLE_FILES:
            return None
        digest = hashlib.sha256()
        total_size = 0
        for path in files:
            relative = path.relative_to(resolved_root).as_posix().encode("utf-8")
            size = path.stat().st_size
            if size < 0 or size > MAX_BUNDLE_FILE_BYTES:
                return None
            payload = path.read_bytes()
            if len(payload) != size:
                return None
            total_size += len(payload)
            if total_size > MAX_BUNDLE_TOTAL_BYTES:
                return None
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()
    except (OSError, ValueError):
        return None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_base_url(value: str) -> str:
    if not isinstance(value, str) or value != value.strip() or len(value) > 2048:
        raise LiveEvaluationError("configuration_invalid")
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or any(ord(character) < 32 for character in value)
    ):
        raise LiveEvaluationError("configuration_invalid")
    return value.rstrip("/")


def _retry_after_seconds(value: Any) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return 1
    return min(max(parsed, 1), 60)


def _safe_int(value: Any) -> int | None:
    if type(value) is int and 0 <= value <= MAX_SAFE_COUNTER:
        return value
    return None


def _safe_identifier(value: Any) -> str | None:
    if type(value) is str and _SAFE_ID.fullmatch(value):
        return value
    return None


def _safe_bool(value: Any) -> bool | None:
    return value if type(value) is bool else None


def _safe_object_list_count(value: Any, *, maximum: int = 64) -> int | None:
    if (
        type(value) is list
        and len(value) <= maximum
        and all(type(row) is dict for row in value)
    ):
        return len(value)
    return None


def _require_object(value: Any) -> dict[str, Any]:
    if type(value) is not dict:
        raise LiveEvaluationError("response_schema_invalid")
    return value


def _require_id(value: Any) -> str:
    prepared = _safe_identifier(value)
    if prepared is None:
        raise LiveEvaluationError("response_schema_invalid")
    return prepared


def _safe_source_path(dataset_root: Path, relative: Any) -> Path:
    if type(relative) is not str or not relative or Path(relative).is_absolute():
        raise LiveEvaluationError("internal_runner_error")
    root = dataset_root.resolve(strict=True)
    try:
        candidate = (root / relative).resolve(strict=True)
        candidate.relative_to(root)
    except (OSError, ValueError):
        raise LiveEvaluationError("internal_runner_error") from None
    if not candidate.is_file() or candidate.stat().st_size > MAX_SOURCE_BYTES:
        raise LiveEvaluationError("internal_runner_error")
    return candidate


def _read_bounded(path: Path, maximum: int) -> bytes:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not 1 <= resolved.stat().st_size <= maximum:
            raise OSError
        return resolved.read_bytes()
    except OSError:
        raise LiveEvaluationError("fixture_integrity_failed") from None


def _authenticated_dev_oracle() -> dict[str, tuple[str, str]] | None:
    """Read only authenticated shared manifest metadata, never story bodies."""

    try:
        payload = _read_bounded(DATASET_ROOT / "manifest.json", MAX_JSON_BYTES)
        if _sha256_bytes(payload) != PINNED_MANIFEST_SHA256:
            return None
        manifest = json.loads(
            payload,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_object,
        )
    except (LiveEvaluationError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    cases = manifest.get("cases") if type(manifest) is dict else None
    if type(cases) is not list:
        return None
    oracle: dict[str, tuple[str, str]] = {}
    for case in cases:
        if type(case) is not dict or case.get("split") != "dev":
            continue
        expected = case.get("expected")
        case_id = case.get("case_id")
        if (
            case_id not in DEV_CASE_IDS
            or case_id in oracle
            or type(expected) is not dict
            or expected.get("decision") not in {"added_issue", "abstain"}
            or expected.get("issue_category") not in ISSUE_CATEGORIES
        ):
            return None
        oracle[case_id] = (
            expected["decision"],
            expected["issue_category"],
        )
    return oracle if set(oracle) == DEV_CASE_IDS else None


def verify_frozen_dataset(
    dataset_root: Path = DATASET_ROOT,
    *,
    split: Literal["dev", "holdout"] | None = None,
) -> VerifiedDataset:
    """Verify the selected split before it is allowed near the network.

    The shared manifest and freeze index are always authenticated. A split run
    hashes only its own source files, so a dev run never opens holdout bodies.
    Passing ``split=None`` remains an explicit offline whole-fixture audit.
    """

    if split not in {None, "dev", "holdout"}:
        raise LiveEvaluationError("configuration_invalid")
    try:
        root = dataset_root.resolve(strict=True)
        manifest_bytes = _read_bounded(root / "manifest.json", MAX_JSON_BYTES)
        freeze_bytes = _read_bounded(root / "freeze.json", MAX_JSON_BYTES)
        if (
            _sha256_bytes(manifest_bytes) != PINNED_MANIFEST_SHA256
            or _sha256_bytes(freeze_bytes) != PINNED_FREEZE_SHA256
        ):
            raise LiveEvaluationError("fixture_integrity_failed")
        manifest = json.loads(
            manifest_bytes,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_object,
        )
        freeze = json.loads(
            freeze_bytes,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_object,
        )
    except LiveEvaluationError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise LiveEvaluationError("fixture_integrity_failed") from None
    if (
        type(manifest) is not dict
        or type(freeze) is not dict
        or manifest.get("dataset_id") != DATASET_ID
        or freeze.get("dataset_id") != DATASET_ID
        or manifest.get("status") != "ready_unscored"
    ):
        raise LiveEvaluationError("fixture_integrity_failed")
    rows = freeze.get("frozen_files")
    cases = manifest.get("cases")
    if (
        type(rows) is not list
        or not 1 <= len(rows) <= 300
        or type(cases) is not list
        or not 1 <= len(cases) <= 64
    ):
        raise LiveEvaluationError("fixture_integrity_failed")
    declared_paths: set[str] = set()
    expected_paths = {"manifest.json", "README.md"}
    selected_paths = {"README.md"}
    selected_source_paths: set[str] = set()
    for case in cases:
        if (
            type(case) is not dict
            or case.get("split") not in {"dev", "holdout"}
            or type(case.get("sources")) is not list
            or not 1 <= len(case["sources"]) <= 8
        ):
            raise LiveEvaluationError("fixture_integrity_failed")
        for source in case["sources"]:
            if type(source) is not dict or type(source.get("path")) is not str:
                raise LiveEvaluationError("fixture_integrity_failed")
            expected_paths.add(source["path"])
            if split is None or case["split"] == split:
                selected_paths.add(source["path"])
                selected_source_paths.add(source["path"])
    frozen_by_path: dict[str, str] = {}
    for row in rows:
        if type(row) is not dict or set(row) != {"path", "sha256"}:
            raise LiveEvaluationError("fixture_integrity_failed")
        relative = row.get("path")
        expected_hash = row.get("sha256")
        if (
            type(relative) is not str
            or relative in declared_paths
            or type(expected_hash) is not str
            or _SHA256.fullmatch(expected_hash) is None
        ):
            raise LiveEvaluationError("fixture_integrity_failed")
        frozen_by_path[relative] = expected_hash
        declared_paths.add(relative)
    if declared_paths != expected_paths or not selected_paths.issubset(declared_paths):
        raise LiveEvaluationError("fixture_integrity_failed")
    if frozen_by_path.get("manifest.json") != _sha256_bytes(manifest_bytes):
        raise LiveEvaluationError("fixture_integrity_failed")
    verified_payloads: list[tuple[str, bytes]] = []
    for relative in sorted(selected_paths):
        try:
            path = _safe_source_path(root, relative)
        except LiveEvaluationError:
            raise LiveEvaluationError("fixture_integrity_failed") from None
        payload = _read_bounded(path, MAX_SOURCE_BYTES)
        if _sha256_bytes(payload) != frozen_by_path[relative]:
            raise LiveEvaluationError("fixture_integrity_failed")
        if relative in selected_source_paths:
            verified_payloads.append((relative, payload))
    return VerifiedDataset(
        manifest=manifest,
        manifest_sha256=_sha256_bytes(manifest_bytes),
        freeze_sha256=_sha256_bytes(freeze_bytes),
        source_payloads=tuple(verified_payloads),
    )


def load_case_plans(
    split: Literal["dev", "holdout"],
    *,
    dataset_root: Path = DATASET_ROOT,
) -> tuple[tuple[CasePlan, ...], str, str]:
    if split not in {"dev", "holdout"}:
        raise LiveEvaluationError("internal_runner_error")
    verified = verify_frozen_dataset(dataset_root, split=split)
    manifest = verified.manifest
    source_payloads = dict(verified.source_payloads)
    plans: list[CasePlan] = []
    seen_cases: set[str] = set()
    for case in manifest["cases"]:
        if case.get("split") != split:
            continue
        case_id = _require_id(case.get("case_id"))
        if case_id in seen_cases or case.get("prompt_example_allowed") is not False:
            raise LiveEvaluationError("internal_runner_error")
        expected = case.get("expected")
        if type(expected) is not dict:
            raise LiveEvaluationError("internal_runner_error")
        decision = expected.get("decision")
        category = expected.get("issue_category")
        added_count = expected.get("added_issue_count")
        if (
            decision not in {"added_issue", "abstain"}
            or category not in ISSUE_CATEGORIES
            or type(added_count) is not int
            or added_count not in {0, 1}
            or (decision == "added_issue") != (added_count == 1)
        ):
            raise LiveEvaluationError("internal_runner_error")
        read_target: tuple[str, int, int] | None = None
        if split == "dev":
            target_key = "candidate" if decision == "added_issue" else "lure_candidate"
            target = case.get(target_key)
            if not _exact_object(
                target, {"document_id", "lines", "kind", "fields"}
            ):
                raise LiveEvaluationError("internal_runner_error")
            target_document = _require_id(target.get("document_id"))
            target_lines = target.get("lines")
            if (
                type(target_lines) is not list
                or len(target_lines) != 2
                or any(type(value) is not int or value < 1 for value in target_lines)
                or target_lines[1] < target_lines[0]
            ):
                raise LiveEvaluationError("internal_runner_error")
            read_target = (target_document, target_lines[0], target_lines[1])
        allowed: list[tuple[str, int, int]] = []
        for row in expected.get("allowed_evidence", []):
            if type(row) is not dict or set(row) != {"document_id", "lines"}:
                raise LiveEvaluationError("internal_runner_error")
            line_range = row.get("lines")
            if (
                type(line_range) is not list
                or len(line_range) != 2
                or any(type(value) is not int or value < 1 for value in line_range)
                or line_range[1] < line_range[0]
            ):
                raise LiveEvaluationError("internal_runner_error")
            allowed.append((str(row["document_id"]), line_range[0], line_range[1]))
        documents: list[SourceDocument] = []
        logical_ids: set[str] = set()
        for source in case["sources"]:
            logical_id = _require_id(source.get("document_id"))
            role = source.get("role")
            scope = _safe_identifier(source.get("story_scope"))
            if logical_id in logical_ids or role not in DOCUMENT_ROLES or scope is None:
                raise LiveEvaluationError("internal_runner_error")
            relative_path = source.get("path")
            if type(relative_path) is not str or relative_path not in source_payloads:
                raise LiveEvaluationError("internal_runner_error")
            try:
                content = source_payloads[relative_path].decode("utf-8")
            except UnicodeDecodeError:
                raise LiveEvaluationError("internal_runner_error") from None
            if not content or (role == "chapter" and any(
                line.lstrip().startswith("@") for line in content.splitlines()
            )):
                raise LiveEvaluationError("internal_runner_error")
            documents.append(
                SourceDocument(
                    logical_id=logical_id,
                    filename=Path(relative_path).name,
                    content=content,
                    role=role,
                    story_scope=scope,
                    line_count=len(content.splitlines()),
                )
            )
            logical_ids.add(logical_id)
        line_count_by_logical = {
            document.logical_id: document.line_count for document in documents
        }
        if (
            not documents
            or any(row[0] not in logical_ids for row in allowed)
            or (
                read_target is not None
                and (
                    read_target[0] not in logical_ids
                    or read_target[2] > line_count_by_logical[read_target[0]]
                )
            )
        ):
            raise LiveEvaluationError("internal_runner_error")
        plans.append(
            CasePlan(
                case_id=case_id,
                split=split,
                documents=tuple(documents),
                ground_truth=GroundTruth(
                    decision=decision,
                    issue_category=category,
                    added_issue_count=added_count,
                    allowed_evidence=tuple(allowed),
                    read_target_evidence=read_target,
                ),
            )
        )
        seen_cases.add(case_id)
    declared = manifest.get("splits", {}).get(split, {}).get("case_count")
    if not plans or type(declared) is not int or len(plans) != declared:
        raise LiveEvaluationError("internal_runner_error")
    return (
        tuple(plans),
        verified.manifest_sha256,
        verified.freeze_sha256,
    )


def _request_with_rate_limit(
    api: JsonApi,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None,
    deadline: float,
    request_timeout: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> Any:
    while True:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise LiveEvaluationError("run_timeout")
        try:
            return api.request_json(
                method,
                path,
                payload=payload,
                timeout=min(request_timeout, max(0.05, remaining)),
            )
        except HttpApiError as exc:
            if exc.status != 429:
                raise
            delay = max(1, exc.retry_after)
            if delay >= remaining:
                raise LiveEvaluationError("run_timeout") from None
            sleep(delay)


def _poll_run(
    api: JsonApi,
    run_id: str,
    *,
    deadline: float,
    request_timeout: float,
    poll_interval: float,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    path = f"/api/v1/analysis-runs/{run_id}"
    while True:
        payload = _require_object(
            _request_with_rate_limit(
                api,
                "GET",
                path,
                payload=None,
                deadline=deadline,
                request_timeout=request_timeout,
                monotonic=monotonic,
                sleep=sleep,
            )
        )
        status = payload.get("status")
        if status in TERMINAL_STATUSES:
            return payload
        if status not in ACTIVE_STATUSES:
            raise LiveEvaluationError("run_status_invalid")
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise LiveEvaluationError("run_timeout")
        sleep(min(poll_interval, remaining))


def _best_effort_cancel(api: JsonApi, run_id: str, timeout: float) -> None:
    try:
        api.request_json(
            "POST", f"/api/v1/analysis-runs/{run_id}/cancel", payload=None, timeout=timeout
        )
    except Exception:
        pass


def _trace_result(
    outcome: Literal["verified", "invalid", "unavailable"],
    *,
    action_counts: Counter[str] | None = None,
    reads: Sequence[DecisionTraceRead] = (),
) -> ParsedDecisionTrace:
    if outcome != "verified":
        return ParsedDecisionTrace(validation_outcome=outcome)
    counts = action_counts or Counter()
    return ParsedDecisionTrace(
        validation_outcome="verified",
        action_counts=tuple(
            (action, counts.get(action, 0))
            for action in sorted(_SAFE_TRACE_ACTIONS)
        ),
        reads=tuple(reads),
    )


def _parse_safe_decision_trace(value: Any) -> ParsedDecisionTrace:
    """Strictly validate the service trace and retain READ coordinates in memory only."""

    root = value if type(value) is dict else None
    investigator = (
        root.get("evidence_investigator") if root is not None else None
    )
    loop = (
        investigator.get("loop")
        if type(investigator) is dict
        else None
    )
    if type(loop) is not dict or "decision_trace" not in loop:
        return _trace_result("unavailable")
    trace = loop.get("decision_trace")
    if not _exact_object(trace, {"schema_version", "complete", "actions"}):
        return _trace_result("invalid")
    if (
        type(trace.get("schema_version")) is not str
        or trace.get("schema_version") != _SAFE_TRACE_SCHEMA
        or trace.get("complete") is not True
        or type(trace.get("actions")) is not list
        or len(trace["actions"]) > _MAX_TRACE_ACTIONS
    ):
        return _trace_result("invalid")

    loop_outcome = loop.get("outcome")
    loop_reason = loop.get("reason_code")
    if (
        type(loop_outcome) is not str
        or loop_outcome not in SAFE_LOOP_OUTCOMES
        or type(loop_reason) is not str
        or loop_reason not in SAFE_LOOP_REASON_CODES
        or (loop_outcome == "completed" and loop_reason != "completed")
    ):
        return _trace_result("invalid")
    counter_names = (
        "provider_calls",
        "executed_tool_calls",
        "executed_searches",
        "executed_reads",
        "recoverable_rejections",
        "completed_seeds",
        "abstained_seeds",
        "submitted_envelopes",
        "authorized_candidates",
    )
    counters = {name: _safe_int(loop.get(name)) for name in counter_names}
    seed_count = _safe_int(investigator.get("seed_count"))
    if (
        any(value is None for value in counters.values())
        or seed_count is None
        or seed_count > _MAX_TRACE_SEED_ORDINAL
        or counters["provider_calls"] > _MAX_TRACE_ACTIONS
        or counters["executed_tool_calls"] > _MAX_TRACE_ACTIONS
        or counters["completed_seeds"] + counters["abstained_seeds"]
        > _MAX_TRACE_SEED_ORDINAL
    ):
        return _trace_result("invalid")

    action_counts: Counter[str] = Counter()
    reads: list[DecisionTraceRead] = []
    positions: list[
        tuple[int, int, str, str, int | None, int | None]
    ] = []
    previous_decision_index = 0
    for row in trace["actions"]:
        if type(row) is not dict:
            return _trace_result("invalid")
        action = row.get("action")
        if type(action) is not str or action not in _SAFE_TRACE_ACTIONS:
            return _trace_result("invalid")
        expected_keys = {
            "SEARCH_EVIDENCE": {
                "provider_decision_index",
                "seed_ordinal",
                "phase",
                "action",
                "result_count",
            },
            "READ_SPAN": {
                "provider_decision_index",
                "seed_ordinal",
                "phase",
                "action",
                "document_ref_hash",
                "line_start",
                "line_end",
                "selected_result_rank",
                "overlaps_anchor",
                "covers_entire_result",
            },
            "SUBMIT_VERDICT": {
                "provider_decision_index",
                "seed_ordinal",
                "phase",
                "action",
                "candidate_count",
                "candidate_shapes",
            },
            "ABSTAIN": {
                "provider_decision_index",
                "seed_ordinal",
                "phase",
                "action",
                "reason",
            },
        }[action]
        if not _exact_object(row, expected_keys):
            return _trace_result("invalid")
        decision_index = row.get("provider_decision_index")
        seed_ordinal = row.get("seed_ordinal")
        phase = row.get("phase")
        if (
            type(decision_index) is not int
            or not 1 <= decision_index <= _MAX_TRACE_ACTIONS
            or decision_index <= previous_decision_index
            or decision_index > counters["provider_calls"]
            or type(seed_ordinal) is not int
            or not 1 <= seed_ordinal <= _MAX_TRACE_SEED_ORDINAL
            or type(phase) is not str
            or phase not in _SAFE_TRACE_PHASES
        ):
            return _trace_result("invalid")
        previous_decision_index = decision_index
        phase_payload: int | None = None
        line_count_payload: int | None = None
        if action == "SEARCH_EVIDENCE":
            result_count = row.get("result_count")
            if (
                phase != "search"
                or type(result_count) is not int
                or not 0 <= result_count <= _MAX_TRACE_RESULT_COUNT
            ):
                return _trace_result("invalid")
            phase_payload = result_count
        elif action == "READ_SPAN":
            document_ref_hash = row.get("document_ref_hash")
            line_start = row.get("line_start")
            line_end = row.get("line_end")
            selected_rank = row.get("selected_result_rank")
            if (
                phase != "read"
                or type(document_ref_hash) is not str
                or _SHA256.fullmatch(document_ref_hash) is None
                or type(line_start) is not int
                or type(line_end) is not int
                or not 1 <= line_start <= line_end <= _MAX_TRACE_LINE_NUMBER
                or line_end - line_start + 1 > _MAX_TRACE_READ_LINES
                or type(selected_rank) is not int
                or not 1 <= selected_rank <= _MAX_TRACE_RESULT_COUNT
                or type(row.get("overlaps_anchor")) is not bool
                or type(row.get("covers_entire_result")) is not bool
            ):
                return _trace_result("invalid")
            phase_payload = selected_rank
            line_count_payload = line_end - line_start + 1
            reads.append(
                DecisionTraceRead(
                    document_ref_hash=document_ref_hash,
                    line_start=line_start,
                    line_end=line_end,
                )
            )
        elif action == "SUBMIT_VERDICT":
            shapes = row.get("candidate_shapes")
            if (
                phase != "verdict"
                or type(row.get("candidate_count")) is not int
                or row.get("candidate_count") != 1
                or type(shapes) is not list
                or len(shapes) != 1
            ):
                return _trace_result("invalid")
            shape = shapes[0]
            if not _exact_object(
                shape, {"kind", "field_names", "source_line_count"}
            ):
                return _trace_result("invalid")
            kind = shape.get("kind")
            field_names = shape.get("field_names")
            source_line_count = shape.get("source_line_count")
            allowed_fields = (
                _SAFE_TRACE_CANDIDATE_FIELDS.get(kind)
                if type(kind) is str
                else None
            )
            if (
                allowed_fields is None
                or type(field_names) is not list
                or len(field_names) > 20
                or any(type(field) is not str for field in field_names)
                or field_names != sorted(set(field_names))
                or not set(field_names).issubset(allowed_fields)
                or type(source_line_count) is not int
                or not 1 <= source_line_count <= _MAX_TRACE_READ_LINES
            ):
                return _trace_result("invalid")
            line_count_payload = source_line_count
        else:
            reason = row.get("reason")
            if (
                type(reason) is not str
                or reason not in _SAFE_TRACE_ABSTAIN_REASONS
            ):
                return _trace_result("invalid")
        positions.append(
            (
                decision_index,
                seed_ordinal,
                phase,
                action,
                phase_payload,
                line_count_payload,
            )
        )
        action_counts[action] += 1

    if (
        len(positions) != counters["executed_tool_calls"]
        or action_counts["SEARCH_EVIDENCE"] != counters["executed_searches"]
        or action_counts["READ_SPAN"] != counters["executed_reads"]
        or action_counts["SUBMIT_VERDICT"] != counters["completed_seeds"]
        or action_counts["ABSTAIN"] != counters["abstained_seeds"]
        or counters["recoverable_rejections"]
        > counters["provider_calls"] - counters["executed_tool_calls"]
    ):
        return _trace_result("invalid")
    if loop_outcome == "completed":
        if (
            action_counts["SUBMIT_VERDICT"]
            != counters["submitted_envelopes"]
            or action_counts["SUBMIT_VERDICT"]
            != counters["authorized_candidates"]
            or counters["provider_calls"]
            != counters["executed_tool_calls"]
            + counters["recoverable_rejections"]
            or (positions and positions[-1][0] != counters["provider_calls"])
            or (not positions and counters["provider_calls"] != 0)
        ):
            return _trace_result("invalid")
    elif (
        counters["submitted_envelopes"] != 0
        or counters["authorized_candidates"] != 0
        or counters["provider_calls"]
        - counters["executed_tool_calls"]
        - counters["recoverable_rejections"]
        not in {0, 1}
    ):
        return _trace_result("invalid")

    current_seed = 0
    current_terminal = False
    latest_search_result_count = 0
    read_completed = False
    read_line_count = 0
    for _, seed_ordinal, phase, action, payload, action_line_count in positions:
        if current_seed == 0:
            if seed_ordinal != 1:
                return _trace_result("invalid")
            current_seed = 1
        elif seed_ordinal == current_seed + 1:
            if not current_terminal:
                return _trace_result("invalid")
            current_seed += 1
            current_terminal = False
            latest_search_result_count = 0
            read_completed = False
            read_line_count = 0
        elif seed_ordinal != current_seed or current_terminal:
            return _trace_result("invalid")
        expected_phase = (
            "verdict"
            if read_completed
            else "read"
            if latest_search_result_count > 0
            else "search"
        )
        if phase != expected_phase:
            return _trace_result("invalid")
        if action == "SEARCH_EVIDENCE":
            latest_search_result_count = payload or 0
        elif action == "READ_SPAN":
            if (
                latest_search_result_count < 1
                or read_completed
                or payload is None
                or payload > latest_search_result_count
            ):
                return _trace_result("invalid")
            read_completed = True
            read_line_count = action_line_count or 0
        elif action == "SUBMIT_VERDICT":
            if (
                not read_completed
                or action_line_count is None
                or action_line_count > read_line_count
            ):
                return _trace_result("invalid")
            current_terminal = True
        else:
            current_terminal = True

    terminal_seeds = counters["completed_seeds"] + counters["abstained_seeds"]
    if terminal_seeds > seed_count or current_seed > seed_count:
        return _trace_result("invalid")
    if loop_outcome == "completed":
        if (
            not current_terminal
            or current_seed != terminal_seeds
            or terminal_seeds != seed_count
        ):
            return _trace_result("invalid")
    elif current_seed not in {terminal_seeds, terminal_seeds + 1}:
        return _trace_result("invalid")
    return _trace_result("verified", action_counts=action_counts, reads=reads)


def _run_scoped_document_ref_hash(*, run_id: str, document_id: str) -> str:
    if (
        type(run_id) is not str
        or _SAFE_ID.fullmatch(run_id) is None
        or type(document_id) is not str
        or _SAFE_ID.fullmatch(document_id) is None
    ):
        raise LiveEvaluationError("response_schema_invalid")
    run_hash = _sha256_bytes(run_id.encode("utf-8"))
    material = (
        "loreguard:evidence-read-ref:v1\0" + run_hash + "\0" + document_id
    )
    return _sha256_bytes(material.encode("utf-8"))


def _bind_decision_trace_to_case(
    trace: ParsedDecisionTrace,
    *,
    run_id: str,
    actual_to_logical: dict[str, str],
    line_counts: dict[str, int],
) -> ParsedDecisionTrace:
    """Bind READ pseudonyms to this run snapshot without exporting the mapping."""

    if trace.validation_outcome != "verified":
        return trace
    try:
        ref_to_logical: dict[str, str] = {}
        for actual_id, logical_id in actual_to_logical.items():
            if (
                type(actual_id) is not str
                or type(logical_id) is not str
                or logical_id not in line_counts
                or type(line_counts[logical_id]) is not int
                or line_counts[logical_id] < 1
            ):
                return _trace_result("invalid")
            document_ref = _run_scoped_document_ref_hash(
                run_id=run_id, document_id=actual_id
            )
            if document_ref in ref_to_logical:
                return _trace_result("invalid")
            ref_to_logical[document_ref] = logical_id
        for read in trace.reads:
            logical_id = ref_to_logical.get(read.document_ref_hash)
            if logical_id is None or read.line_end > line_counts[logical_id]:
                return _trace_result("invalid")
    except (AttributeError, TypeError, ValueError, LiveEvaluationError):
        return _trace_result("invalid")
    return trace


def _read_target_evidence_match(
    trace: ParsedDecisionTrace,
    *,
    ground_truth: GroundTruth,
    run_id: str,
    actual_to_logical: dict[str, str],
) -> bool | None:
    target = ground_truth.read_target_evidence
    if trace.validation_outcome != "verified" or target is None:
        return None
    logical_id, target_start, target_end = target
    actual_ids = [
        actual_id
        for actual_id, logical in actual_to_logical.items()
        if logical == logical_id
    ]
    if len(actual_ids) != 1:
        return None
    expected_document_ref = _run_scoped_document_ref_hash(
        run_id=run_id, document_id=actual_ids[0]
    )
    return any(
        read.document_ref_hash == expected_document_ref
        and read.line_start <= target_start
        and read.line_end >= target_end
        for read in trace.reads
    )


def _safe_diagnostics(value: Any) -> dict[str, Any]:
    root = value if type(value) is dict else {}
    investigator = root.get("evidence_investigator")
    source = investigator if type(investigator) is dict else {}
    outcome = source.get("outcome")
    reason_source = source.get("reason_code")
    reason = (
        reason_source
        if type(reason_source) is str
        and reason_source in SAFE_INVESTIGATOR_REASON_CODES
        else None
        if reason_source is None
        else "internal_failure"
    )
    loop_source = source.get("loop") if type(source.get("loop")) is dict else {}
    loop_outcome = loop_source.get("outcome")
    loop_reason_source = loop_source.get("reason_code")
    loop_reason = (
        loop_reason_source
        if type(loop_reason_source) is str
        and loop_reason_source in SAFE_LOOP_REASON_CODES
        else None
        if loop_reason_source is None
        else "internal_failure"
    )
    promotion_source = (
        source.get("promotion") if type(source.get("promotion")) is dict else {}
    )
    rejection_source = promotion_source.get("rejection_counts")
    rejections = {
        key: count
        for key, value in (rejection_source.items() if type(rejection_source) is dict else ())
        if type(key) is str
        and key in SAFE_REJECTION_REASONS
        and (count := _safe_int(value)) is not None
    }
    rag_source = source.get("rag") if type(source.get("rag")) is dict else {}
    index_source = rag_source.get("index") if type(rag_source.get("index")) is dict else {}
    retrievals = rag_source.get("retrievals")
    modes: set[str] = set()
    strategies: set[str] = set()
    for row in retrievals[:16] if type(retrievals) is list else ():
        if type(row) is not dict:
            continue
        if type(row.get("mode")) is str and row.get("mode") in SAFE_RAG_MODES:
            modes.add(row["mode"])
        if (
            type(row.get("strategy")) is str
            and row.get("strategy") in SAFE_RAG_STRATEGIES
        ):
            strategies.add(row["strategy"])
    usage_source = source.get("usage") if type(source.get("usage")) is dict else {}
    provider_categories: Counter[str] = Counter()
    calls = usage_source.get("provider_calls")
    for row in calls[:64] if type(calls) is list else ():
        if (
            type(row) is dict
            and type(row.get("category")) is str
            and row.get("category") in SAFE_PROVIDER_CATEGORIES
        ):
            provider_categories[row["category"]] += 1
    usage_token_keys = (
        "reported_prompt_tokens",
        "reported_completion_tokens",
        "charged_tokens",
    )
    if any(key in usage_source for key in usage_token_keys):
        reported_prompt_tokens = _safe_int(
            usage_source.get("reported_prompt_tokens")
        )
        reported_completion_tokens = _safe_int(
            usage_source.get("reported_completion_tokens")
        )
        charged_tokens = _safe_int(usage_source.get("charged_tokens"))
        usage_token_basis = (
            "investigator_usage_ledger"
            if None
            not in {
                reported_prompt_tokens,
                reported_completion_tokens,
                charged_tokens,
            }
            else "investigator_usage_ledger_invalid"
        )
    else:
        # Compatibility for diagnostics emitted before the runtime-level usage
        # accumulator became the authoritative ledger. Never mix fields from
        # the two sources: a partial modern ledger must fail closed.
        reported_prompt_tokens = _safe_int(
            loop_source.get("reported_prompt_tokens")
        )
        reported_completion_tokens = _safe_int(
            loop_source.get("reported_completion_tokens")
        )
        charged_tokens = _safe_int(loop_source.get("charged_tokens"))
        usage_token_basis = (
            "legacy_loop_compatibility"
            if None
            not in {
                reported_prompt_tokens,
                reported_completion_tokens,
                charged_tokens,
            }
            else "unavailable"
        )
    provider_calls = _safe_int(loop_source.get("provider_calls"))
    reported_tool_calls = _safe_int(loop_source.get("executed_tool_calls"))
    if (
        reported_tool_calls is not None
        and provider_calls is not None
        and reported_tool_calls <= provider_calls
    ):
        exact_tool_calls = reported_tool_calls
        tool_call_count_basis = "server_reported_executed_tools"
    elif loop_outcome == "completed" and provider_calls is not None:
        # Backward-compatible fallback for artifacts produced before degraded
        # loop diagnostics exposed an explicit execution counter.
        exact_tool_calls = provider_calls
        tool_call_count_basis = "completed_loop_protocol_invariant"
    else:
        exact_tool_calls = None
        tool_call_count_basis = "unavailable"
    budget_source = (
        source.get("budget_preflight")
        if type(source.get("budget_preflight")) is dict
        else {}
    )
    budget_preflight = {
        "seed_count": _safe_int(budget_source.get("seed_count")),
        "minimum_path_admissible": _safe_bool(
            budget_source.get("minimum_path_admissible")
        ),
        "minimum_required_rounds": _safe_int(
            budget_source.get("minimum_required_rounds")
        ),
        "minimum_initial_reservation": _safe_int(
            budget_source.get("minimum_initial_reservation")
        ),
        "maximum_local_round_reservation": _safe_int(
            budget_source.get("maximum_local_round_reservation")
        ),
        "maximum_local_run_reservation": _safe_int(
            budget_source.get("maximum_local_run_reservation")
        ),
        "max_charged_tokens": _safe_int(
            budget_source.get("max_charged_tokens")
        ),
        "oversized_initial_prompts": _safe_int(
            budget_source.get("oversized_initial_prompts")
        ),
    }
    profile_fingerprint = index_source.get("profile_fingerprint")
    chunker_fingerprint = index_source.get("chunker_fingerprint")
    index_outcome_source = index_source.get("outcome")
    index_reason_source = index_source.get("reason")
    return {
        "available": bool(source),
        "enabled": source.get("enabled") is True,
        "outcome": (
            outcome
            if type(outcome) is str and outcome in SAFE_INVESTIGATOR_OUTCOMES
            else None
        ),
        "reason_code": reason,
        "seed_count": _safe_int(source.get("seed_count")),
        "loop": {
            "outcome": (
                loop_outcome
                if type(loop_outcome) is str and loop_outcome in SAFE_LOOP_OUTCOMES
                else None
            ),
            "reason_code": loop_reason,
            "provider_decision_calls": provider_calls,
            "tool_calls": exact_tool_calls,
            "tool_call_count_basis": tool_call_count_basis,
            "searches": _safe_int(loop_source.get("executed_searches")),
            "reads": _safe_int(loop_source.get("executed_reads")),
            "recoverable_rejections": _safe_int(
                loop_source.get("recoverable_rejections")
            ),
            "completed_seeds": _safe_int(loop_source.get("completed_seeds")),
            "abstained_seeds": _safe_int(loop_source.get("abstained_seeds")),
            "submitted_envelopes": _safe_int(loop_source.get("submitted_envelopes")),
            "authorized_candidates": _safe_int(loop_source.get("authorized_candidates")),
        },
        "promotion": {
            "submitted_candidates": _safe_int(promotion_source.get("submitted_candidates")),
            "accepted_candidates": _safe_int(promotion_source.get("accepted_candidates")),
            "added_issues": _safe_int(promotion_source.get("added_issues")),
            "rejection_counts": dict(sorted(rejections.items())),
        },
        "usage": {
            "reported_prompt_tokens": reported_prompt_tokens,
            "reported_completion_tokens": reported_completion_tokens,
            "charged_tokens": charged_tokens,
            "token_count_basis": usage_token_basis,
            "provider_category_counts": dict(sorted(provider_categories.items())),
        },
        "budget_preflight": budget_preflight,
        "effective_limits": {
            "max_charged_tokens": budget_preflight["max_charged_tokens"],
            "source": "service_budget_preflight",
        },
        "rag": {
            "index_outcome": (
                index_outcome_source
                if type(index_outcome_source) is str
                and index_outcome_source in SAFE_RAG_INDEX_OUTCOMES
                else None
            ),
            "index_reason": (
                index_reason_source
                if index_reason_source is None
                else index_reason_source
                if type(index_reason_source) is str
                and index_reason_source in SAFE_RAG_INDEX_REASONS
                else "internal_failure"
            ),
            "profile_fingerprint": (
                profile_fingerprint
                if type(profile_fingerprint) is str and _SHA256.fullmatch(profile_fingerprint)
                else None
            ),
            "chunker_fingerprint": (
                chunker_fingerprint
                if type(chunker_fingerprint) is str
                and _SHA256.fullmatch(chunker_fingerprint)
                else None
            ),
            "modes": sorted(modes),
            "strategies": sorted(strategies),
            "total_retrievals": _safe_int(rag_source.get("total_retrievals")),
        },
    }


def _safe_capability_isolation(
    value: Any,
    *,
    status_payload: Any,
    investigator: dict[str, Any],
) -> dict[str, Any]:
    """Return a content-free, fail-closed proof that only Investigator used chat.

    ``model`` is always emitted by the production analysis pipeline. By
    contrast, ``ai_evidence_review`` is added only when that optional service
    branch is enabled, so absence is the switch-off proof for that stage.
    Aggregate chat accounting must also equal the Investigator ledger; this
    catches a future chat-backed stage that is not yet known to this runner.
    """

    root = value if type(value) is dict else {}
    status = status_payload if type(status_payload) is dict else {}
    model_source = root.get("model") if type(root.get("model")) is dict else None
    if model_source is None:
        main = {
            "diagnostics_present": False,
            "enabled": None,
            "configured": None,
            "used": None,
            "logical_calls": None,
            "provider_calls": None,
        }
    else:
        main = {
            "diagnostics_present": True,
            "enabled": _safe_bool(model_source.get("enabled")),
            "configured": _safe_bool(model_source.get("configured")),
            "used": _safe_bool(model_source.get("used")),
            "logical_calls": _safe_int(model_source.get("logical_call_count")),
            "provider_calls": _safe_object_list_count(
                model_source.get("provider_calls")
            ),
        }

    review_source = (
        root.get("ai_evidence_review")
        if type(root.get("ai_evidence_review")) is dict
        else None
    )
    if review_source is None:
        review = {
            "diagnostics_present": False,
            "enabled": False,
            "logical_calls": 0,
            "provider_calls": 0,
            "switch_proof": "stage_diagnostics_absent",
        }
    else:
        review_usage = (
            review_source.get("usage_accounting")
            if type(review_source.get("usage_accounting")) is dict
            else {}
        )
        review_calls = _safe_object_list_count(review_source.get("provider_calls"))
        logical_calls = _safe_int(review_usage.get("logical_calls"))
        if logical_calls is None and review_calls is not None:
            logical_calls = review_calls
        review = {
            "diagnostics_present": True,
            "enabled": _safe_bool(review_source.get("enabled")),
            "logical_calls": logical_calls,
            "provider_calls": review_calls,
            "switch_proof": "explicit_stage_diagnostics",
        }

    investigator_usage = (
        investigator.get("usage")
        if type(investigator.get("usage")) is dict
        else {}
    )
    run_prompt = _safe_int(status.get("prompt_tokens"))
    run_completion = _safe_int(status.get("completion_tokens"))
    usage_accounting = (
        status.get("usage_accounting")
        if type(status.get("usage_accounting")) is dict
        else {}
    )
    run_charged = _safe_int(usage_accounting.get("charged_tokens"))
    investigator_prompt = _safe_int(
        investigator_usage.get("reported_prompt_tokens")
    )
    investigator_completion = _safe_int(
        investigator_usage.get("reported_completion_tokens")
    )
    investigator_charged = _safe_int(investigator_usage.get("charged_tokens"))
    usage_matches = (
        None
        not in {
            run_prompt,
            run_completion,
            run_charged,
            investigator_prompt,
            investigator_completion,
            investigator_charged,
        }
        and run_prompt == investigator_prompt
        and run_completion == investigator_completion
        and run_charged == investigator_charged
    )

    reasons: list[str] = []
    if not main["diagnostics_present"]:
        reasons.append("main_extraction_diagnostics_missing")
    if main["enabled"] is not False:
        reasons.append("main_extraction_not_disabled")
    if main["configured"] is not False:
        reasons.append("main_extraction_configured_or_unknown")
    if main["used"] is not False:
        reasons.append("main_extraction_used_or_unknown")
    if main["logical_calls"] != 0 or main["provider_calls"] != 0:
        reasons.append("main_extraction_calls_nonzero_or_unknown")
    if review["enabled"] is not False:
        reasons.append("issue_evidence_review_not_disabled")
    if review["logical_calls"] != 0 or review["provider_calls"] != 0:
        reasons.append("issue_evidence_review_calls_nonzero_or_unknown")
    if not usage_matches:
        reasons.append("aggregate_chat_usage_mismatch_or_unknown")

    return {
        "verified": not reasons,
        "main_extraction": main,
        "issue_evidence_review": review,
        "aggregate_chat_usage_matches_investigator": usage_matches,
        "reason_codes": reasons,
    }


def _observed_issues(
    value: Any,
    *,
    actual_to_logical: dict[str, str],
    line_counts: dict[str, int],
) -> tuple[Counter[str], list[dict[str, Any]], bool]:
    if type(value) is not list or len(value) > 256:
        raise LiveEvaluationError("response_schema_invalid")
    categories: Counter[str] = Counter()
    sanitized: list[dict[str, Any]] = []
    all_authorized = True
    for issue in value:
        if type(issue) is not dict or issue.get("category") not in ISSUE_CATEGORIES:
            raise LiveEvaluationError("response_schema_invalid")
        category = issue["category"]
        categories[category] += 1
        spans: list[tuple[str, int, int]] = []
        evidence = issue.get("evidence")
        if type(evidence) is not list or not evidence or len(evidence) > 16:
            all_authorized = False
            evidence = []
        for row in evidence:
            if type(row) is not dict:
                all_authorized = False
                continue
            logical = actual_to_logical.get(row.get("document_id"))
            start = row.get("line_start")
            end = row.get("line_end")
            if (
                logical is None
                or type(start) is not int
                or type(end) is not int
                or start < 1
                or end < start
                or end > line_counts[logical]
            ):
                all_authorized = False
                continue
            spans.append((logical, start, end))
        sanitized.append({"category": category, "spans": tuple(spans)})
    return categories, sanitized, all_authorized


def _target_evidence_match(
    issues: Sequence[dict[str, Any]], ground_truth: GroundTruth
) -> bool:
    required_spans = frozenset(ground_truth.allowed_evidence)
    if not required_spans:
        return False
    return any(
        issue["category"] == ground_truth.issue_category
        and required_spans.issubset(frozenset(issue["spans"]))
        for issue in issues
    )


def _reason_from_diagnostics(diagnostics: dict[str, Any]) -> str | None:
    loop = diagnostics.get("loop", {})
    if loop.get("outcome") != "completed":
        return loop.get("reason_code") or diagnostics.get("reason_code")
    return diagnostics.get("reason_code") or loop.get("reason_code")


def _classify(
    *,
    run_status: str,
    categories: Counter[str],
    diagnostics: dict[str, Any],
    ground_truth: GroundTruth,
    target_evidence_match: bool,
    citation_scope_authorized: bool,
    capability_isolation: dict[str, Any],
) -> tuple[str, bool]:
    reason = _reason_from_diagnostics(diagnostics)
    loop = diagnostics.get("loop", {})
    promotion = diagnostics.get("promotion", {})
    investigator_applied = (
        diagnostics.get("available") is True
        and diagnostics.get("enabled") is True
        and diagnostics.get("outcome") == "completed"
        and loop.get("outcome") == "completed"
        and (promotion.get("accepted_candidates") or 0) >= 1
        and promotion.get("added_issues") == ground_truth.added_issue_count
    )
    if capability_isolation.get("verified") is not True:
        classification = "isolation_failure"
    elif reason in TIME_OR_BUDGET_REASONS:
        classification = "timeout_or_budget"
    elif reason in AGENT_PROTOCOL_FAILURE_REASONS:
        classification = "agent_protocol_failure"
    elif run_status != "completed" or reason in PROVIDER_FAILURE_REASONS:
        classification = "provider_failure"
    elif diagnostics.get("outcome") == "degraded":
        classification = "provider_failure"
    elif sum(diagnostics.get("promotion", {}).get("rejection_counts", {}).values()) > 0:
        classification = "promotion_rejection"
    elif (diagnostics.get("loop", {}).get("abstained_seeds") or 0) > 0:
        classification = "active_abstain"
    elif (
        ground_truth.decision == "added_issue"
        and categories.get(ground_truth.issue_category, 0) == ground_truth.added_issue_count
        and sum(categories.values()) == ground_truth.added_issue_count
        and target_evidence_match
        and citation_scope_authorized
        and investigator_applied
    ):
        classification = "positive_added_issue"
    else:
        classification = "wrong_candidate"

    if ground_truth.decision == "added_issue":
        passed = classification == "positive_added_issue"
    else:
        passed = (
            sum(categories.values()) == 0
            and classification in {"active_abstain", "promotion_rejection"}
        )
    return classification, passed


def _run_one_case(
    plan: CasePlan,
    *,
    api: JsonApi,
    options: RunnerOptions,
    monotonic: Callable[[], float],
    sleep: Callable[[float], None],
) -> dict[str, Any]:
    started = monotonic()
    deadline = started + options.case_timeout_seconds
    project_id: str | None = None
    run_id: str | None = None
    actual_to_logical: dict[str, str] = {}
    line_counts = {document.logical_id: document.line_count for document in plan.documents}
    try:
        project = _require_object(
            _request_with_rate_limit(
                api,
                "POST",
                "/api/v1/projects",
                payload={
                    "name": f"EIL isolated {uuid4().hex[:16]}",
                    "description": "",
                },
                deadline=deadline,
                request_timeout=options.request_timeout_seconds,
                monotonic=monotonic,
                sleep=sleep,
            )
        )
        project_id = _require_id(project.get("id"))
        for document in plan.documents:
            upload = _require_object(
                _request_with_rate_limit(
                    api,
                    "POST",
                    f"/api/v1/projects/{project_id}/documents/text",
                    payload=document.upload_payload(),
                    deadline=deadline,
                    request_timeout=options.request_timeout_seconds,
                    monotonic=monotonic,
                    sleep=sleep,
                )
            )
            actual_id = _require_id(upload.get("id"))
            if (
                upload.get("document_role") != document.role
                or upload.get("story_scope") != document.story_scope
                or actual_id in actual_to_logical
            ):
                raise LiveEvaluationError("response_schema_invalid")
            actual_to_logical[actual_id] = document.logical_id
        started_run = _require_object(
            _request_with_rate_limit(
                api,
                "POST",
                f"/api/v1/projects/{project_id}/analysis-runs",
                payload=None,
                deadline=deadline,
                request_timeout=options.request_timeout_seconds,
                monotonic=monotonic,
                sleep=sleep,
            )
        )
        run_id = _require_id(started_run.get("id"))
        status_payload = _poll_run(
            api,
            run_id,
            deadline=deadline,
            request_timeout=options.request_timeout_seconds,
            poll_interval=options.poll_interval_seconds,
            monotonic=monotonic,
            sleep=sleep,
        )
        status = status_payload["status"]
        diagnostics_payload = _request_with_rate_limit(
            api,
            "GET",
            f"/api/v1/analysis-runs/{run_id}/diagnostics",
            payload=None,
            deadline=deadline,
            request_timeout=options.request_timeout_seconds,
            monotonic=monotonic,
            sleep=sleep,
        )
        decision_trace = _bind_decision_trace_to_case(
            _parse_safe_decision_trace(diagnostics_payload),
            run_id=run_id,
            actual_to_logical=actual_to_logical,
            line_counts=line_counts,
        )
        diagnostics = _safe_diagnostics(diagnostics_payload)
        worker_runtime_provenance = _safe_runtime_provenance(
            diagnostics_payload.get("runtime_provenance")
            if type(diagnostics_payload) is dict
            else None
        )
        capability_isolation = _safe_capability_isolation(
            diagnostics_payload,
            status_payload=status_payload,
            investigator=diagnostics,
        )
        issue_payload: Any = []
        if status == "completed":
            issue_payload = _request_with_rate_limit(
                api,
                "GET",
                f"/api/v1/analysis-runs/{run_id}/issues",
                payload=None,
                deadline=deadline,
                request_timeout=options.request_timeout_seconds,
                monotonic=monotonic,
                sleep=sleep,
            )
        categories, issues, authorized = _observed_issues(
            issue_payload,
            actual_to_logical=actual_to_logical,
            line_counts=line_counts,
        )
        evidence_match = _target_evidence_match(issues, plan.ground_truth)
        read_target_match = _read_target_evidence_match(
            decision_trace,
            ground_truth=plan.ground_truth,
            run_id=run_id,
            actual_to_logical=actual_to_logical,
        )
        classification, passed = _classify(
            run_status=status,
            categories=categories,
            diagnostics=diagnostics,
            ground_truth=plan.ground_truth,
            target_evidence_match=evidence_match,
            citation_scope_authorized=authorized,
            capability_isolation=capability_isolation,
        )
        usage_accounting = (
            status_payload.get("usage_accounting")
            if type(status_payload.get("usage_accounting")) is dict
            else {}
        )
        return {
            "case_id": plan.case_id,
            "expected_decision": plan.ground_truth.decision,
            "expected_issue_category": plan.ground_truth.issue_category,
            "classification": classification,
            "passed": passed,
            "run_status": status,
            "project_ref_hash": _sha256_bytes(project_id.encode("utf-8")),
            "run_ref_hash": _sha256_bytes(run_id.encode("utf-8")),
            "case_wall_latency_ms": max(
                0, round((monotonic() - started) * 1000)
            ),
            "issue_category_counts": dict(sorted(categories.items())),
            "citation_scope_authorized": authorized,
            "target_evidence_match": evidence_match,
            "read_target_evidence_match": read_target_match,
            "decision_trace_attribution": decision_trace.artifact_summary(),
            "analysis_usage": {
                "reported_prompt_tokens": _safe_int(status_payload.get("prompt_tokens")),
                "reported_completion_tokens": _safe_int(
                    status_payload.get("completion_tokens")
                ),
                "charged_tokens": _safe_int(usage_accounting.get("charged_tokens")),
            },
            "investigator": diagnostics,
            "runtime_provenance": worker_runtime_provenance,
            "capability_isolation": capability_isolation,
            "failure_code": None,
        }
    except LiveEvaluationError as exc:
        if run_id is not None and exc.code == "run_timeout":
            _best_effort_cancel(api, run_id, min(options.request_timeout_seconds, 5.0))
        classification = (
            "timeout_or_budget" if exc.code == "run_timeout" else "runner_failure"
        )
        return {
            "case_id": plan.case_id,
            "expected_decision": plan.ground_truth.decision,
            "expected_issue_category": plan.ground_truth.issue_category,
            "classification": classification,
            "passed": False,
            "run_status": None,
            "project_ref_hash": (
                _sha256_bytes(project_id.encode("utf-8")) if project_id else None
            ),
            "run_ref_hash": _sha256_bytes(run_id.encode("utf-8")) if run_id else None,
            "case_wall_latency_ms": max(
                0, round((monotonic() - started) * 1000)
            ),
            "issue_category_counts": {},
            "citation_scope_authorized": None,
            "target_evidence_match": None,
            "read_target_evidence_match": None,
            "decision_trace_attribution": _trace_result(
                "unavailable"
            ).artifact_summary(),
            "analysis_usage": {
                "reported_prompt_tokens": None,
                "reported_completion_tokens": None,
                "charged_tokens": None,
            },
            "investigator": None,
            "runtime_provenance": None,
            "capability_isolation": None,
            "failure_code": exc.code,
        }
    except Exception:
        return {
            "case_id": plan.case_id,
            "expected_decision": plan.ground_truth.decision,
            "expected_issue_category": plan.ground_truth.issue_category,
            "classification": "runner_failure",
            "passed": False,
            "run_status": None,
            "project_ref_hash": None,
            "run_ref_hash": None,
            "case_wall_latency_ms": max(
                0, round((monotonic() - started) * 1000)
            ),
            "issue_category_counts": {},
            "citation_scope_authorized": None,
            "target_evidence_match": None,
            "read_target_evidence_match": None,
            "decision_trace_attribution": _trace_result(
                "unavailable"
            ).artifact_summary(),
            "analysis_usage": {
                "reported_prompt_tokens": None,
                "reported_completion_tokens": None,
                "charged_tokens": None,
            },
            "investigator": None,
            "runtime_provenance": None,
            "capability_isolation": None,
            "failure_code": "internal_runner_error",
        }


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        dirty_output = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "tracked_worktree_dirty": None}
    return {
        "commit": commit if re.fullmatch(r"[a-f0-9]{40,64}", commit) else None,
        "tracked_worktree_dirty": bool(dirty_output),
    }


def _stable_git_state(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    commit = before.get("commit")
    dirty = before.get("tracked_worktree_dirty")
    stable = (
        type(commit) is str
        and _GIT_REVISION.fullmatch(commit) is not None
        and type(dirty) is bool
        and before == after
    )
    return {
        "commit": commit if stable else None,
        "tracked_worktree_dirty": dirty if stable else None,
        "stable_during_run": stable,
    }


def _percentile(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percentile) - 1)
    return ordered[index]


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _trace_attribution_is_well_formed(value: Any) -> bool:
    if not _exact_object(value, _TRACE_ATTRIBUTION_KEYS):
        return False
    outcome = value.get("validation_outcome")
    if type(outcome) is not str or outcome not in _SAFE_TRACE_VALIDATION_OUTCOMES:
        return False
    action_counts = value.get("action_counts")
    if outcome != "verified":
        return action_counts is None
    return (
        type(action_counts) is dict
        and set(action_counts) == _SAFE_TRACE_ACTIONS
        and all(type(key) is str for key in action_counts)
        and all(_safe_int(count) is not None for count in action_counts.values())
        and sum(action_counts.values()) <= _MAX_TRACE_ACTIONS
    )


def _trace_attribution_matches_sanitized_loop(
    attribution: Any, investigator: Any
) -> bool:
    """Bind exported trace counts to the independently sanitized loop counters."""

    if not _trace_attribution_is_well_formed(attribution):
        return False
    if attribution["validation_outcome"] != "verified":
        return True
    if type(investigator) is not dict:
        return False
    loop = investigator.get("loop")
    if type(loop) is not dict or set(loop) != _LOOP_KEYS:
        return False
    counter_keys = {
        "tool_calls",
        "searches",
        "reads",
        "completed_seeds",
        "abstained_seeds",
    }
    if any(_safe_int(loop.get(key)) is None for key in counter_keys):
        return False
    expected = {
        "ABSTAIN": loop["abstained_seeds"],
        "READ_SPAN": loop["reads"],
        "SEARCH_EVIDENCE": loop["searches"],
        "SUBMIT_VERDICT": loop["completed_seeds"],
    }
    return (
        attribution["action_counts"] == expected
        and loop["tool_calls"] == sum(expected.values())
    )


def _erroneous_added_issue_count(result: dict[str, Any]) -> int | None:
    """Count final issues that cannot be credited to the frozen expectation.

    A missing/failed final response is unknown rather than zero.  For a positive
    case, at most the one fixture-authorized, correctly cited target issue can be
    credited; every other returned issue is an erroneous addition.  Every issue
    returned for an abstain case is erroneous.
    """

    counts = result.get("issue_category_counts")
    if (
        result.get("run_status") != "completed"
        or type(counts) is not dict
        or result.get("citation_scope_authorized") is None
    ):
        return None
    safe_counts: dict[str, int] = {}
    for category, count in counts.items():
        if category not in ISSUE_CATEGORIES or _safe_int(count) is None:
            return None
        safe_counts[category] = count
    total = sum(safe_counts.values())
    if result.get("expected_decision") == "abstain":
        return total
    if result.get("expected_decision") != "added_issue":
        return None
    expected_category = result.get("expected_issue_category")
    if expected_category not in ISSUE_CATEGORIES:
        return None
    credited = 0
    if (
        result.get("citation_scope_authorized") is True
        and result.get("target_evidence_match") is True
    ):
        # Frozen v1 cases allow exactly one expected addition.
        credited = min(safe_counts.get(expected_category, 0), 1)
    return total - credited


def _build_outcome_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Build orthogonal safety, capability and liveness metrics.

    These intentionally do not replace the strict per-case ``passed`` score.
    In particular, a deterministic promotion rejection may keep a negative case
    safe without proving that the Agent itself chose the correct abstain action.
    """

    case_count = len(results)
    positive_cases = sum(
        result.get("expected_decision") == "added_issue" for result in results
    )
    negative_cases = sum(
        result.get("expected_decision") == "abstain" for result in results
    )
    positive_passed = sum(
        result.get("expected_decision") == "added_issue"
        and result.get("passed") is True
        for result in results
    )

    trace_validation_counts: Counter[str] = Counter()
    trace_action_counts: Counter[str] = Counter()
    trace_verified_cases = 0
    read_target_observed = 0
    read_target_matched = 0
    for result in results:
        attribution = result.get("decision_trace_attribution")
        if not _trace_attribution_is_well_formed(attribution):
            continue
        outcome = attribution["validation_outcome"]
        trace_validation_counts[outcome] += 1
        if outcome == "verified":
            trace_verified_cases += 1
            trace_action_counts.update(attribution["action_counts"])
        read_match = result.get("read_target_evidence_match")
        if type(read_match) is bool:
            read_target_observed += 1
            read_target_matched += read_match is True

    erroneous_counts = [_erroneous_added_issue_count(result) for result in results]
    final_output_observed = sum(count is not None for count in erroneous_counts)
    system_safe_cases = sum(count == 0 for count in erroneous_counts)
    erroneous_added_issues = sum(
        count for count in erroneous_counts if count is not None
    )
    negative_safe_cases = sum(
        result.get("expected_decision") == "abstain" and count == 0
        for result, count in zip(results, erroneous_counts, strict=True)
    )
    negative_active_abstain_cases = sum(
        result.get("expected_decision") == "abstain"
        and result.get("classification") == "active_abstain"
        for result in results
    )

    isolation_verified = sum(
        type(result.get("capability_isolation")) is dict
        and result["capability_isolation"].get("verified") is True
        for result in results
    )
    promotion_submitted = 0
    promotion_accepted = 0
    promotion_accounting_observed = 0
    for result in results:
        investigator = result.get("investigator")
        promotion = (
            investigator.get("promotion")
            if type(investigator) is dict
            and type(investigator.get("promotion")) is dict
            else {}
        )
        submitted = _safe_int(promotion.get("submitted_candidates"))
        accepted = _safe_int(promotion.get("accepted_candidates"))
        if submitted is None or accepted is None or accepted > submitted:
            continue
        promotion_accounting_observed += 1
        promotion_submitted += submitted
        promotion_accepted += accepted

    normal_termination_classes = {
        "positive_added_issue",
        "active_abstain",
        "promotion_rejection",
        "wrong_candidate",
    }
    normally_terminated = sum(
        result.get("classification") in normal_termination_classes for result in results
    )
    return {
        "final_output_observed_cases": final_output_observed,
        "system_safe_cases": system_safe_cases,
        "system_safety_rate": _rate(system_safe_cases, case_count),
        "erroneous_added_issue_count": erroneous_added_issues,
        "capability_isolation_verified": isolation_verified,
        "capability_isolation_rate": _rate(isolation_verified, case_count),
        "positive_cases": positive_cases,
        "positive_passed": positive_passed,
        "positive_recall": _rate(positive_passed, positive_cases),
        "negative_cases": negative_cases,
        "negative_safe_cases": negative_safe_cases,
        "negative_safety_rate": _rate(negative_safe_cases, negative_cases),
        "negative_active_abstain_cases": negative_active_abstain_cases,
        "agent_active_abstain_rate": _rate(
            negative_active_abstain_cases, negative_cases
        ),
        "promotion_accounting_observed_cases": promotion_accounting_observed,
        "promotion_submitted_candidates": promotion_submitted,
        "promotion_accepted_candidates": promotion_accepted,
        "promotion_acceptance_rate": _rate(
            promotion_accepted, promotion_submitted
        ),
        "normally_terminated_cases": normally_terminated,
        "normal_termination_rate": _rate(normally_terminated, case_count),
        "decision_trace_validation_counts": dict(
            sorted(trace_validation_counts.items())
        ),
        "decision_trace_verified_cases": trace_verified_cases,
        "decision_trace_action_counts": {
            action: trace_action_counts.get(action, 0)
            for action in sorted(_SAFE_TRACE_ACTIONS)
        },
        "read_target_evidence_observed_cases": read_target_observed,
        "read_target_evidence_matched_cases": read_target_matched,
        "read_target_evidence_match_rate": _rate(
            read_target_matched, read_target_observed
        ),
    }


def _build_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    classifications = Counter(result["classification"] for result in results)
    case_wall_latencies = [result["case_wall_latency_ms"] for result in results]
    passed = sum(result["passed"] is True for result in results)
    return {
        "case_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        **_build_outcome_summary(results),
        "disqualifying_execution_failure_cases": sum(
            classifications.get(classification, 0)
            for classification in DISQUALIFYING_EXECUTION_CLASSIFICATIONS
        ),
        "classification_counts": dict(sorted(classifications.items())),
        "case_wall_latency_ms_p50": _percentile(case_wall_latencies, 0.50),
        "case_wall_latency_ms_p95": _percentile(case_wall_latencies, 0.95),
    }


def _safe_runtime_provenance(value: Any) -> dict[str, Any] | None:
    root = value if type(value) is dict else None
    if root is None or set(root) != {
        "schema_version",
        "build",
        "chat_provider",
        "capabilities",
        "investigator_limits",
        "rag",
    }:
        return None
    if root.get("schema_version") != RUNTIME_PROVENANCE_SCHEMA:
        return None

    build = root.get("build")
    if type(build) is not dict or set(build) != {
        "git_revision",
        "service_artifact_sha256",
    }:
        return None
    revision = build.get("git_revision")
    artifact_hash = build.get("service_artifact_sha256")
    if (
        revision is not None
        and (type(revision) is not str or _GIT_REVISION.fullmatch(revision) is None)
    ) or type(artifact_hash) is not str or _SHA256.fullmatch(artifact_hash) is None:
        return None

    provider = root.get("chat_provider")
    if type(provider) is not dict or set(provider) != {
        "model_alias",
        "endpoint_configuration_sha256",
        "temperature",
        "thinking_configured",
        "thinking_mode",
    }:
        return None
    model_alias = provider.get("model_alias")
    endpoint_hash = provider.get("endpoint_configuration_sha256")
    if (
        type(model_alias) is not str
        or model_alias != model_alias.strip()
        or not model_alias
        or len(model_alias) > 255
        or any(ord(character) < 32 for character in model_alias)
        or type(endpoint_hash) is not str
        or _SHA256.fullmatch(endpoint_hash) is None
        or provider.get("temperature") != 0
        or type(provider.get("thinking_configured")) is not bool
        or (
            provider.get("thinking_mode") is not None
            and type(provider.get("thinking_mode")) is not str
        )
        or provider.get("thinking_mode") not in {None, "disabled", "enabled"}
        or provider.get("thinking_configured")
        != (provider.get("thinking_mode") is not None)
    ):
        return None

    capabilities = root.get("capabilities")
    if (
        type(capabilities) is not dict
        or set(capabilities) != _CAPABILITY_KEYS
        or any(type(value) is not bool for value in capabilities.values())
    ):
        return None

    limits = root.get("investigator_limits")
    if type(limits) is not dict or set(limits) != _INVESTIGATOR_LIMIT_KEYS:
        return None
    safe_limits: dict[str, Any] = {}
    for key in _INVESTIGATOR_INTEGER_LIMIT_KEYS:
        parsed = _safe_int(limits.get(key))
        if parsed is None or parsed <= 0:
            return None
        safe_limits[key] = parsed
    for key in _INVESTIGATOR_NUMBER_LIMIT_KEYS:
        parsed = limits.get(key)
        if (
            isinstance(parsed, bool)
            or not isinstance(parsed, (int, float))
            or not math.isfinite(float(parsed))
            or float(parsed) <= 0
        ):
            return None
        safe_limits[key] = float(parsed)
    if type(limits.get("require_hybrid")) is not bool:
        return None
    safe_limits["require_hybrid"] = limits["require_hybrid"]

    rag = root.get("rag")
    if type(rag) is not dict or set(rag) != {
        "strategy",
        "profile_fingerprint",
        "chunker_fingerprint",
        "require_hybrid",
        "top_k",
        "branch_limit",
    }:
        return None
    profile = rag.get("profile_fingerprint")
    chunker = rag.get("chunker_fingerprint")
    top_k = _safe_int(rag.get("top_k"))
    branch_limit = _safe_int(rag.get("branch_limit"))
    if (
        rag.get("strategy") != "keyword+vector+entity-rrf"
        or type(profile) is not str
        or _SHA256.fullmatch(profile) is None
        or type(chunker) is not str
        or _SHA256.fullmatch(chunker) is None
        or type(rag.get("require_hybrid")) is not bool
        or top_k is None
        or top_k <= 0
        or branch_limit is None
        or branch_limit <= 0
        or rag["require_hybrid"] != safe_limits["require_hybrid"]
        or top_k != safe_limits["top_k"]
        or branch_limit != safe_limits["branch_limit"]
    ):
        return None

    return {
        "schema_version": RUNTIME_PROVENANCE_SCHEMA,
        "build": {
            "git_revision": revision,
            "service_artifact_sha256": artifact_hash,
        },
        "chat_provider": {
            "model_alias": model_alias,
            "endpoint_configuration_sha256": endpoint_hash,
            "temperature": 0,
            "thinking_configured": provider["thinking_configured"],
            "thinking_mode": provider["thinking_mode"],
        },
        "capabilities": dict(sorted(capabilities.items())),
        "investigator_limits": dict(sorted(safe_limits.items())),
        "rag": {
            "strategy": rag["strategy"],
            "profile_fingerprint": profile,
            "chunker_fingerprint": chunker,
            "require_hybrid": rag["require_hybrid"],
            "top_k": top_k,
            "branch_limit": branch_limit,
        },
    }


def _safe_health(value: Any) -> dict[str, Any]:
    root = _require_object(value)
    model = root.get("model") if type(root.get("model")) is dict else {}
    thinking = model.get("thinking") if type(model.get("thinking")) is dict else {}
    mode = thinking.get("mode")
    return {
        "service_ok": root.get("status") == "ok",
        "model_configured": model.get("configured") is True,
        "thinking_configured": thinking.get("configured") is True,
        "thinking_mode": mode if mode in {"enabled", "disabled"} else None,
        "runtime_provenance": _safe_runtime_provenance(
            root.get("runtime_provenance")
        ),
    }


def _service_reported_effective_limits(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    snapshots: list[dict[str, Any]] = []
    for row in results:
        investigator = row.get("investigator")
        limits = (
            investigator.get("effective_limits")
            if type(investigator) is dict
            and type(investigator.get("effective_limits")) is dict
            else None
        )
        if (
            limits is not None
            and _safe_int(limits.get("max_charged_tokens")) is not None
            and limits.get("source") == "service_budget_preflight"
        ):
            snapshots.append(limits)
    unique = {
        _canonical_json_bytes(snapshot): snapshot
        for snapshot in snapshots
    }
    consistent = len(snapshots) == len(results) and len(unique) == 1
    return {
        "observed_cases": len(snapshots),
        "consistent_across_cases": consistent,
        "values": next(iter(unique.values())) if consistent else None,
        "distinct_snapshot_count": len(unique),
    }


def _build_runtime_provenance_gate(
    api_provenance: dict[str, Any] | None,
    results: Sequence[dict[str, Any]],
    git_state: dict[str, Any],
    runner_service_artifact_sha256: str | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    case_count = len(results)
    if api_provenance is None:
        reasons.append("api_runtime_provenance_missing_or_invalid")
    if (
        type(runner_service_artifact_sha256) is not str
        or _SHA256.fullmatch(runner_service_artifact_sha256) is None
    ):
        reasons.append("runner_service_artifact_unavailable")

    worker_rows = [
        row.get("runtime_provenance")
        for row in results
        if type(row.get("runtime_provenance")) is dict
    ]
    worker_matches = sum(
        api_provenance is not None and row == api_provenance for row in worker_rows
    )
    if len(worker_rows) != case_count:
        reasons.append("worker_runtime_provenance_incomplete")
    if worker_matches != case_count:
        reasons.append("api_worker_runtime_provenance_mismatch")

    git_commit = git_state.get("commit")
    if (
        type(git_commit) is not str
        or _GIT_REVISION.fullmatch(git_commit) is None
        or git_state.get("tracked_worktree_dirty") is not False
        or git_state.get("stable_during_run") is not True
    ):
        reasons.append("runner_git_state_not_frozen")

    rag_verified = 0
    if api_provenance is not None:
        build = api_provenance["build"]
        provider = api_provenance["chat_provider"]
        capabilities = api_provenance["capabilities"]
        expected_rag = api_provenance["rag"]
        if (
            build.get("service_artifact_sha256")
            != runner_service_artifact_sha256
        ):
            reasons.append("service_artifact_mismatch_with_runner")
        if build.get("git_revision") != git_commit:
            reasons.append("service_build_revision_mismatch")
        if capabilities != {
            "embeddings": True,
            "evidence_investigator": True,
            "issue_evidence_review": False,
            "model_extraction": False,
            "record_repair_agent": False,
        }:
            reasons.append("capability_configuration_not_isolated")
        if (
            provider.get("thinking_configured") is not True
            or provider.get("thinking_mode") != "disabled"
            or provider.get("temperature") != 0
        ):
            reasons.append("provider_sampling_configuration_not_frozen")
        if expected_rag.get("require_hybrid") is not True:
            reasons.append("hybrid_rag_not_required")

        for result in results:
            investigator = result.get("investigator")
            rag = (
                investigator.get("rag")
                if type(investigator) is dict
                and type(investigator.get("rag")) is dict
                else {}
            )
            if (
                rag.get("index_outcome") == "complete"
                and rag.get("index_reason") is None
                and rag.get("profile_fingerprint")
                == expected_rag.get("profile_fingerprint")
                and rag.get("chunker_fingerprint")
                == expected_rag.get("chunker_fingerprint")
                and rag.get("modes") == ["hybrid"]
                and rag.get("strategies")
                == [expected_rag.get("strategy")]
                and type(rag.get("total_retrievals")) is int
                and rag["total_retrievals"] >= 1
            ):
                rag_verified += 1
    if rag_verified != case_count:
        reasons.append("observed_rag_configuration_incomplete_or_mismatched")

    return {
        "passed": not reasons,
        "reason_codes": reasons,
        "api_runtime_provenance": api_provenance,
        "runner_service_artifact_sha256": runner_service_artifact_sha256,
        "worker_runtime_provenance_observed_cases": len(worker_rows),
        "worker_runtime_provenance_matches_api_cases": worker_matches,
        "rag_configuration_verified_cases": rag_verified,
        "case_count": case_count,
    }


def _build_development_gate(
    split: str,
    summary: dict[str, Any],
    provenance_gate: dict[str, Any],
) -> dict[str, Any]:
    if split != "dev":
        return {
            "applicable": False,
            "passed": None,
            "checks": {},
            "reason_codes": [],
            "boundary": "Development qualification thresholds apply only to dev.",
        }
    case_count = summary.get("case_count")
    checks = {
        "fixture_shape_8_with_5_positive_3_negative": (
            case_count == 8
            and summary.get("positive_cases") == 5
            and summary.get("negative_cases") == 3
        ),
        "strict_pass_at_least_5_of_8": (
            type(summary.get("passed")) is int and summary["passed"] >= 5
        ),
        "positive_pass_at_least_3_of_5": (
            type(summary.get("positive_passed")) is int
            and summary["positive_passed"] >= 3
        ),
        "negative_final_safety_3_of_3": (
            summary.get("negative_safe_cases") == 3
        ),
        "negative_active_abstain_at_least_2_of_3": (
            type(summary.get("negative_active_abstain_cases")) is int
            and summary["negative_active_abstain_cases"] >= 2
        ),
        "normal_termination_8_of_8": (
            type(summary.get("normally_terminated_cases")) is int
            and summary["normally_terminated_cases"] == 8
        ),
        "zero_disqualifying_execution_failures": (
            summary.get("disqualifying_execution_failure_cases") == 0
        ),
        "zero_erroneous_added_issues": (
            summary.get("erroneous_added_issue_count") == 0
        ),
        "final_output_observed_8_of_8": (
            summary.get("final_output_observed_cases") == 8
        ),
        "capability_isolation_8_of_8": (
            summary.get("capability_isolation_verified") == 8
        ),
        "promotion_accounting_observed_8_of_8": (
            summary.get("promotion_accounting_observed_cases") == 8
        ),
        "decision_trace_verified_8_of_8": (
            summary.get("decision_trace_verified_cases") == 8
        ),
        "read_target_attribution_observed_8_of_8": (
            summary.get("read_target_evidence_observed_cases") == 8
        ),
        "read_target_evidence_matched_8_of_8": (
            summary.get("read_target_evidence_matched_cases") == 8
        ),
        "runtime_provenance_and_rag_verified": (
            provenance_gate.get("passed") is True
        ),
    }
    return {
        "applicable": True,
        "passed": all(checks.values()),
        "checks": checks,
        "reason_codes": [key for key, passed in checks.items() if not passed],
        "boundary": (
            "A promotion rejection may preserve system safety, but it does not "
            "contribute to the active-abstain threshold. This small developer-visible "
            "fixture is a pre-holdout qualification gate, not a quality claim."
        ),
    }


def _safe_counter_dict(value: Any, allowed_keys: frozenset[str]) -> bool:
    return (
        type(value) is dict
        and set(value).issubset(allowed_keys)
        and all(_safe_int(count) is not None for count in value.values())
    )


def _qualified_case_is_well_formed(
    value: Any, dev_oracle: dict[str, tuple[str, str]]
) -> bool:
    """Validate the complete sanitized case contract used by the dev gate."""

    if type(value) is not dict or set(value) != _CASE_KEYS:
        return False
    case_id = value.get("case_id")
    classification = value.get("classification")
    if (
        type(case_id) is not str
        or case_id not in dev_oracle
        or (
            value.get("expected_decision"),
            value.get("expected_issue_category"),
        )
        != dev_oracle.get(case_id)
        or type(classification) is not str
        or classification not in SAFE_CLASSIFICATIONS
        or type(value.get("passed")) is not bool
        or value.get("run_status") != "completed"
        or type(value.get("project_ref_hash")) is not str
        or _SHA256.fullmatch(value["project_ref_hash"]) is None
        or type(value.get("run_ref_hash")) is not str
        or _SHA256.fullmatch(value["run_ref_hash"]) is None
        or _safe_int(value.get("case_wall_latency_ms")) is None
        or type(value.get("citation_scope_authorized")) is not bool
        or type(value.get("target_evidence_match")) is not bool
        or value.get("read_target_evidence_match") is not True
        or value.get("failure_code") is not None
    ):
        return False
    issue_counts = value.get("issue_category_counts")
    if (
        not _safe_counter_dict(issue_counts, ISSUE_CATEGORIES)
        or any(count < 1 for count in issue_counts.values())
    ):
        return False

    analysis_usage = value.get("analysis_usage")
    if (
        type(analysis_usage) is not dict
        or set(analysis_usage)
        != {"reported_prompt_tokens", "reported_completion_tokens", "charged_tokens"}
        or any(_safe_int(count) is None for count in analysis_usage.values())
    ):
        return False

    provenance = value.get("runtime_provenance")
    if _safe_runtime_provenance(provenance) != provenance:
        return False

    investigator = value.get("investigator")
    if type(investigator) is not dict or set(investigator) != _INVESTIGATOR_KEYS:
        return False
    if (
        investigator.get("available") is not True
        or investigator.get("enabled") is not True
        or investigator.get("outcome") != "completed"
        or investigator.get("reason_code") != "completed"
        or _safe_int(investigator.get("seed_count")) is None
    ):
        return False

    loop = investigator.get("loop")
    if type(loop) is not dict or set(loop) != _LOOP_KEYS:
        return False
    if (
        loop.get("outcome") != "completed"
        or loop.get("reason_code") != "completed"
        or loop.get("tool_call_count_basis")
        != "server_reported_executed_tools"
        or any(
            _safe_int(loop.get(key)) is None
            for key in _LOOP_KEYS
            - {"outcome", "reason_code", "tool_call_count_basis"}
        )
    ):
        return False
    provider_decisions = loop["provider_decision_calls"]
    tool_calls = loop["tool_calls"]
    recoverable_rejections = loop["recoverable_rejections"]
    terminal_seeds = loop["completed_seeds"] + loop["abstained_seeds"]
    if (
        provider_decisions < 1
        or tool_calls > provider_decisions
        or provider_decisions != tool_calls + recoverable_rejections
        or loop["searches"] + loop["reads"] + terminal_seeds != tool_calls
        or terminal_seeds != investigator["seed_count"]
        or loop["reads"] < 1
        or loop["searches"] < loop["reads"]
        or loop["completed_seeds"] > loop["reads"]
    ):
        return False
    trace_attribution = value.get("decision_trace_attribution")
    if (
        not _trace_attribution_matches_sanitized_loop(
            trace_attribution, investigator
        )
        or trace_attribution.get("validation_outcome") != "verified"
    ):
        return False

    promotion = investigator.get("promotion")
    if type(promotion) is not dict or set(promotion) != _PROMOTION_KEYS:
        return False
    submitted = _safe_int(promotion.get("submitted_candidates"))
    accepted = _safe_int(promotion.get("accepted_candidates"))
    added = _safe_int(promotion.get("added_issues"))
    if (
        None in {submitted, accepted, added}
        or accepted > submitted
        or added > accepted
        or not _safe_counter_dict(
            promotion.get("rejection_counts"), SAFE_REJECTION_REASONS
        )
    ):
        return False

    usage = investigator.get("usage")
    if type(usage) is not dict or set(usage) != _USAGE_KEYS:
        return False
    if (
        any(
            _safe_int(usage.get(key)) is None
            for key in {
                "reported_prompt_tokens",
                "reported_completion_tokens",
                "charged_tokens",
            }
        )
        or usage.get("token_count_basis") != "investigator_usage_ledger"
        or usage["charged_tokens"]
        < usage["reported_prompt_tokens"] + usage["reported_completion_tokens"]
        or not _safe_counter_dict(
            usage.get("provider_category_counts"), frozenset({"success"})
        )
        or usage["provider_category_counts"].get("success", 0)
        != loop["provider_decision_calls"]
        or analysis_usage
        != {
            "reported_prompt_tokens": usage["reported_prompt_tokens"],
            "reported_completion_tokens": usage["reported_completion_tokens"],
            "charged_tokens": usage["charged_tokens"],
        }
    ):
        return False

    budget = investigator.get("budget_preflight")
    if type(budget) is not dict or set(budget) != _BUDGET_PREFLIGHT_KEYS:
        return False
    if (
        type(budget.get("minimum_path_admissible")) is not bool
        or any(
            _safe_int(budget.get(key)) is None
            for key in _BUDGET_PREFLIGHT_KEYS - {"minimum_path_admissible"}
        )
    ):
        return False
    effective = investigator.get("effective_limits")
    if (
        type(effective) is not dict
        or set(effective) != {"max_charged_tokens", "source"}
        or _safe_int(effective.get("max_charged_tokens")) is None
        or effective.get("max_charged_tokens") != budget.get("max_charged_tokens")
        or effective.get("source") != "service_budget_preflight"
    ):
        return False

    rag = investigator.get("rag")
    if type(rag) is not dict or set(rag) != _RAG_KEYS:
        return False
    if (
        rag.get("index_outcome") != "complete"
        or rag.get("index_reason") is not None
        or type(rag.get("profile_fingerprint")) is not str
        or _SHA256.fullmatch(rag["profile_fingerprint"]) is None
        or type(rag.get("chunker_fingerprint")) is not str
        or _SHA256.fullmatch(rag["chunker_fingerprint"]) is None
        or rag.get("modes") != ["hybrid"]
        or rag.get("strategies") != ["keyword+vector+entity-rrf"]
        or _safe_int(rag.get("total_retrievals")) is None
        or rag["total_retrievals"] < 1
    ):
        return False

    isolation = value.get("capability_isolation")
    if type(isolation) is not dict or set(isolation) != _ISOLATION_KEYS:
        return False
    main = isolation.get("main_extraction")
    review = isolation.get("issue_evidence_review")
    if (
        isolation.get("verified") is not True
        or isolation.get("aggregate_chat_usage_matches_investigator") is not True
        or isolation.get("reason_codes") != []
        or type(main) is not dict
        or set(main)
        != {
            "diagnostics_present",
            "enabled",
            "configured",
            "used",
            "logical_calls",
            "provider_calls",
        }
        or main
        != {
            "diagnostics_present": True,
            "enabled": False,
            "configured": False,
            "used": False,
            "logical_calls": 0,
            "provider_calls": 0,
        }
        or type(review) is not dict
        or set(review)
        != {
            "diagnostics_present",
            "enabled",
            "logical_calls",
            "provider_calls",
            "switch_proof",
        }
        or review.get("enabled") is not False
        or review.get("logical_calls") != 0
        or review.get("provider_calls") != 0
        or review.get("switch_proof")
        not in {"stage_diagnostics_absent", "explicit_stage_diagnostics"}
    ):
        return False

    expected_count = 1 if value["expected_decision"] == "added_issue" else 0
    expected_classification, expected_passed = _classify(
        run_status="completed",
        categories=Counter(issue_counts),
        diagnostics=investigator,
        ground_truth=GroundTruth(
            decision=value["expected_decision"],
            issue_category=value["expected_issue_category"],
            added_issue_count=expected_count,
            allowed_evidence=(),
        ),
        target_evidence_match=value["target_evidence_match"],
        citation_scope_authorized=value["citation_scope_authorized"],
        capability_isolation=isolation,
    )
    return (
        value["classification"] == expected_classification
        and value["passed"] is expected_passed
    )


def _artifact_tree_avoids_trace_details(value: Any) -> bool:
    if type(value) is dict:
        if (
            any(type(key) is not str for key in value)
            or any(key in _FORBIDDEN_ARTIFACT_KEYS for key in value)
        ):
            return False
        return all(_artifact_tree_avoids_trace_details(row) for row in value.values())
    if type(value) is list:
        return all(_artifact_tree_avoids_trace_details(row) for row in value)
    if value is None or type(value) in {str, int, bool}:
        return True
    return type(value) is float and math.isfinite(value)


def _nullable_safe_int(value: Any) -> bool:
    return value is None or _safe_int(value) is not None


def _nullable_bool(value: Any) -> bool:
    return value is None or type(value) is bool


def _sanitized_investigator_is_export_safe(value: Any) -> bool:
    if type(value) is not dict or set(value) != _INVESTIGATOR_KEYS:
        return False
    if (
        type(value.get("available")) is not bool
        or type(value.get("enabled")) is not bool
        or (
            value.get("outcome") is not None
            and (
                type(value.get("outcome")) is not str
                or value.get("outcome") not in SAFE_INVESTIGATOR_OUTCOMES
            )
        )
        or (
            value.get("reason_code") is not None
            and (
                type(value.get("reason_code")) is not str
                or value.get("reason_code") not in SAFE_INVESTIGATOR_REASON_CODES
            )
        )
        or not _nullable_safe_int(value.get("seed_count"))
    ):
        return False
    loop = value.get("loop")
    if type(loop) is not dict or set(loop) != _LOOP_KEYS:
        return False
    if (
        loop.get("outcome") is not None
        and (
            type(loop.get("outcome")) is not str
            or loop.get("outcome") not in SAFE_LOOP_OUTCOMES
        )
    ) or (
        loop.get("reason_code") is not None
        and (
            type(loop.get("reason_code")) is not str
            or loop.get("reason_code") not in SAFE_LOOP_REASON_CODES
        )
    ):
        return False
    if loop.get("tool_call_count_basis") not in {
        "server_reported_executed_tools",
        "completed_loop_protocol_invariant",
        "unavailable",
    } or any(
        not _nullable_safe_int(loop.get(key))
        for key in _LOOP_KEYS
        - {"outcome", "reason_code", "tool_call_count_basis"}
    ):
        return False
    promotion = value.get("promotion")
    if type(promotion) is not dict or set(promotion) != _PROMOTION_KEYS:
        return False
    if any(
        not _nullable_safe_int(promotion.get(key))
        for key in {"submitted_candidates", "accepted_candidates", "added_issues"}
    ) or not _safe_counter_dict(
        promotion.get("rejection_counts"), SAFE_REJECTION_REASONS
    ):
        return False
    usage = value.get("usage")
    if type(usage) is not dict or set(usage) != _USAGE_KEYS:
        return False
    if any(
        not _nullable_safe_int(usage.get(key))
        for key in {
            "reported_prompt_tokens",
            "reported_completion_tokens",
            "charged_tokens",
        }
    ) or usage.get("token_count_basis") not in {
        "investigator_usage_ledger",
        "investigator_usage_ledger_invalid",
        "legacy_loop_compatibility",
        "unavailable",
    } or not _safe_counter_dict(
        usage.get("provider_category_counts"), SAFE_PROVIDER_CATEGORIES
    ):
        return False
    budget = value.get("budget_preflight")
    if type(budget) is not dict or set(budget) != _BUDGET_PREFLIGHT_KEYS:
        return False
    if not _nullable_bool(budget.get("minimum_path_admissible")) or any(
        not _nullable_safe_int(budget.get(key))
        for key in _BUDGET_PREFLIGHT_KEYS - {"minimum_path_admissible"}
    ):
        return False
    effective = value.get("effective_limits")
    if not _exact_object(effective, {"max_charged_tokens", "source"}) or (
        not _nullable_safe_int(effective.get("max_charged_tokens"))
        or effective.get("source") != "service_budget_preflight"
    ):
        return False
    rag = value.get("rag")
    if type(rag) is not dict or set(rag) != _RAG_KEYS:
        return False
    if (
        rag.get("index_outcome") is not None
        and (
            type(rag.get("index_outcome")) is not str
            or rag.get("index_outcome") not in SAFE_RAG_INDEX_OUTCOMES
        )
    ) or (
        rag.get("index_reason") is not None
        and (
            type(rag.get("index_reason")) is not str
            or rag.get("index_reason") not in SAFE_RAG_INDEX_REASONS
        )
    ):
        return False
    for key in ("profile_fingerprint", "chunker_fingerprint"):
        fingerprint = rag.get(key)
        if fingerprint is not None and (
            type(fingerprint) is not str or _SHA256.fullmatch(fingerprint) is None
        ):
            return False
    if (
        type(rag.get("modes")) is not list
        or rag["modes"] != sorted(set(rag["modes"]))
        or any(
            type(mode) is not str or mode not in SAFE_RAG_MODES
            for mode in rag["modes"]
        )
        or type(rag.get("strategies")) is not list
        or rag["strategies"] != sorted(set(rag["strategies"]))
        or any(
            type(strategy) is not str or strategy not in SAFE_RAG_STRATEGIES
            for strategy in rag["strategies"]
        )
        or not _nullable_safe_int(rag.get("total_retrievals"))
    ):
        return False
    return True


def _capability_isolation_is_export_safe(value: Any) -> bool:
    if type(value) is not dict or set(value) != _ISOLATION_KEYS:
        return False
    reasons = value.get("reason_codes")
    allowed_reasons = {
        "main_extraction_diagnostics_missing",
        "main_extraction_not_disabled",
        "main_extraction_configured_or_unknown",
        "main_extraction_used_or_unknown",
        "main_extraction_calls_nonzero_or_unknown",
        "issue_evidence_review_not_disabled",
        "issue_evidence_review_calls_nonzero_or_unknown",
        "aggregate_chat_usage_mismatch_or_unknown",
    }
    if (
        type(value.get("verified")) is not bool
        or type(value.get("aggregate_chat_usage_matches_investigator")) is not bool
        or type(reasons) is not list
        or len(reasons) > len(allowed_reasons)
        or any(type(reason) is not str or reason not in allowed_reasons for reason in reasons)
        or len(set(reasons)) != len(reasons)
        or value["verified"] is not (not reasons)
    ):
        return False
    main = value.get("main_extraction")
    if not _exact_object(
        main,
        {
            "diagnostics_present",
            "enabled",
            "configured",
            "used",
            "logical_calls",
            "provider_calls",
        },
    ) or (
        type(main.get("diagnostics_present")) is not bool
        or not _nullable_bool(main.get("enabled"))
        or not _nullable_bool(main.get("configured"))
        or not _nullable_bool(main.get("used"))
        or not _nullable_safe_int(main.get("logical_calls"))
        or not _nullable_safe_int(main.get("provider_calls"))
    ):
        return False
    review = value.get("issue_evidence_review")
    return bool(
        _exact_object(
            review,
            {
                "diagnostics_present",
                "enabled",
                "logical_calls",
                "provider_calls",
                "switch_proof",
            },
        )
        and type(review.get("diagnostics_present")) is bool
        and _nullable_bool(review.get("enabled"))
        and _nullable_safe_int(review.get("logical_calls"))
        and _nullable_safe_int(review.get("provider_calls"))
        and type(review.get("switch_proof")) is str
        and review.get("switch_proof")
        in {"stage_diagnostics_absent", "explicit_stage_diagnostics"}
    )


def _export_case_shape_is_valid(value: Any, *, split: str) -> bool:
    if type(value) is not dict or set(value) != _CASE_KEYS:
        return False
    expected_decision = value.get("expected_decision")
    expected_category = value.get("expected_issue_category")
    classification = value.get("classification")
    run_status = value.get("run_status")
    failure_code = value.get("failure_code")
    if (
        _safe_identifier(value.get("case_id")) is None
        or type(expected_decision) is not str
        or expected_decision not in {"added_issue", "abstain"}
        or type(expected_category) is not str
        or expected_category not in ISSUE_CATEGORIES
        or type(classification) is not str
        or classification not in SAFE_CLASSIFICATIONS
        or type(value.get("passed")) is not bool
        or run_status is not None
        and (type(run_status) is not str or run_status not in TERMINAL_STATUSES)
        or _safe_int(value.get("case_wall_latency_ms")) is None
        or not _nullable_bool(value.get("citation_scope_authorized"))
        or not _nullable_bool(value.get("target_evidence_match"))
        or not _nullable_bool(value.get("read_target_evidence_match"))
        or failure_code is not None
        and (
            type(failure_code) is not str or failure_code not in SAFE_FAILURE_CODES
        )
    ):
        return False
    for key in ("project_ref_hash", "run_ref_hash"):
        ref = value.get(key)
        if ref is not None and (
            type(ref) is not str or _SHA256.fullmatch(ref) is None
        ):
            return False
    if not _safe_counter_dict(value.get("issue_category_counts"), ISSUE_CATEGORIES):
        return False
    attribution = value.get("decision_trace_attribution")
    if not _trace_attribution_is_well_formed(attribution):
        return False
    if attribution["validation_outcome"] == "verified":
        expected_read_type = bool if split == "dev" else type(None)
        if type(value.get("read_target_evidence_match")) is not expected_read_type:
            return False
    elif value.get("read_target_evidence_match") is not None:
        return False
    analysis_usage = value.get("analysis_usage")
    if not _exact_object(
        analysis_usage,
        {"reported_prompt_tokens", "reported_completion_tokens", "charged_tokens"},
    ) or any(
        count is not None and _safe_int(count) is None
        for count in analysis_usage.values()
    ):
        return False
    investigator = value.get("investigator")
    if investigator is not None and not _sanitized_investigator_is_export_safe(
        investigator
    ):
        return False
    if not _trace_attribution_matches_sanitized_loop(attribution, investigator):
        return False
    provenance = value.get("runtime_provenance")
    if provenance is not None and _safe_runtime_provenance(provenance) != provenance:
        return False
    isolation = value.get("capability_isolation")
    if isolation is not None and not _capability_isolation_is_export_safe(isolation):
        return False
    return True


def _artifact_export_contract_is_valid_unchecked(value: Any) -> bool:
    """Validate the exact privacy/export schema immediately before persistence."""

    if (
        type(value) is not dict
        or set(value) != _DEV_ARTIFACT_KEYS
        or value.get("schema_version") != ARTIFACT_SCHEMA
        or value.get("dataset_id") != DATASET_ID
        or value.get("split") not in {"dev", "holdout"}
        or value.get("execution_source") != "live_http_service"
        or value.get("diagnostic_boundary") != _DIAGNOSTIC_BOUNDARY
        or not _artifact_tree_avoids_trace_details(value)
    ):
        return False
    privacy = value.get("privacy_boundary")
    if not _exact_object(privacy, set(_PRIVACY_BOUNDARY)) or any(
        privacy.get(key) is not False for key in _PRIVACY_BOUNDARY
    ):
        return False
    git = value.get("git")
    if not _exact_object(
        git, {"commit", "tracked_worktree_dirty", "stable_during_run"}
    ) or (
        type(git.get("stable_during_run")) is not bool
        or git.get("commit") is not None
        and (
            type(git.get("commit")) is not str
            or _GIT_REVISION.fullmatch(git["commit"]) is None
        )
        or git.get("tracked_worktree_dirty") is not None
        and type(git.get("tracked_worktree_dirty")) is not bool
    ):
        return False
    fixture = value.get("fixture")
    if not _exact_object(fixture, {"manifest_sha256", "freeze_sha256"}) or any(
        type(fixture.get(key)) is not str
        or _SHA256.fullmatch(fixture[key]) is None
        for key in fixture
    ):
        return False
    if fixture != {
        "manifest_sha256": PINNED_MANIFEST_SHA256,
        "freeze_sha256": PINNED_FREEZE_SHA256,
    }:
        return False
    safe_configuration = value.get("safe_configuration")
    fingerprint = value.get("safe_configuration_fingerprint")
    service_observation = (
        safe_configuration.get("service_observation")
        if type(safe_configuration) is dict
        else None
    )
    if (
        type(safe_configuration) is not dict
        or set(safe_configuration) != _SAFE_CONFIGURATION_KEYS
        or safe_configuration.get("artifact_schema") != ARTIFACT_SCHEMA
        or safe_configuration.get("dataset_id") != DATASET_ID
        or safe_configuration.get("split") != value["split"]
        or safe_configuration.get("runner_git") != git
        or type(fingerprint) is not str
        or _SHA256.fullmatch(fingerprint) is None
        or value.get("reproducibility_fingerprint") != fingerprint
        or _sha256_bytes(_canonical_json_bytes(safe_configuration)) != fingerprint
        or safe_configuration.get("manifest_sha256") != fixture["manifest_sha256"]
        or safe_configuration.get("freeze_sha256") != fixture["freeze_sha256"]
        or type(safe_configuration.get("runner_service_artifact_sha256")) is not str
        or _SHA256.fullmatch(
            safe_configuration["runner_service_artifact_sha256"]
        )
        is None
        or any(
            type(safe_configuration.get(key)) not in {int, float}
            or not math.isfinite(float(safe_configuration[key]))
            or float(safe_configuration[key]) <= 0
            for key in {
                "case_timeout_seconds",
                "request_timeout_seconds",
                "poll_interval_seconds",
            }
        )
        or not 1 <= float(safe_configuration["case_timeout_seconds"]) <= 3600
        or not 0.1 <= float(safe_configuration["request_timeout_seconds"]) <= 120
        or not 0.05 <= float(safe_configuration["poll_interval_seconds"]) <= 30
        or not _exact_object(
            service_observation,
            {
                "service_ok",
                "model_configured",
                "thinking_configured",
                "thinking_mode",
                "runtime_provenance",
            },
        )
        or type(service_observation.get("service_ok")) is not bool
        or type(service_observation.get("model_configured")) is not bool
        or type(service_observation.get("thinking_configured")) is not bool
        or service_observation.get("service_ok") is not True
        or service_observation.get("model_configured") is not True
        or service_observation.get("thinking_mode") not in {
            None,
            "disabled",
            "enabled",
        }
        or service_observation.get("thinking_configured")
        is not (service_observation.get("thinking_mode") is not None)
        or _safe_runtime_provenance(
            service_observation.get("runtime_provenance")
        )
        != service_observation.get("runtime_provenance")
        or type(service_observation.get("runtime_provenance")) is not dict
        or _safe_int(
            safe_configuration.get("worker_runtime_provenance_matches_api_cases")
        )
        is None
        or _safe_int(safe_configuration.get("rag_configuration_verified_cases"))
        is None
    ):
        return False
    cases = value.get("cases")
    if (
        type(cases) is not list
        or not cases
        or len(cases) > 64
        or not all(
            _export_case_shape_is_valid(case, split=value["split"])
            for case in cases
        )
        or len({case["case_id"] for case in cases}) != len(cases)
    ):
        return False
    try:
        if value.get("summary") != _build_summary(cases):
            return False
    except (KeyError, TypeError, ValueError):
        return False
    provenance_gate = value.get("provenance_gate")
    expected_provenance_gate = _build_runtime_provenance_gate(
        service_observation["runtime_provenance"],
        cases,
        git,
        safe_configuration["runner_service_artifact_sha256"],
    )
    development_gate = value.get("development_gate")
    expected_development_gate = _build_development_gate(
        value["split"], value["summary"], expected_provenance_gate
    )
    return bool(
        _exact_object(provenance_gate, _PROVENANCE_GATE_KEYS)
        and provenance_gate == expected_provenance_gate
        and _exact_object(development_gate, _DEVELOPMENT_GATE_KEYS)
        and development_gate == expected_development_gate
        and safe_configuration.get("service_reported_effective_limits")
        == _service_reported_effective_limits(cases)
        and safe_configuration.get(
            "worker_runtime_provenance_matches_api_cases"
        )
        == provenance_gate.get("worker_runtime_provenance_matches_api_cases")
        and safe_configuration.get("rag_configuration_verified_cases")
        == provenance_gate.get("rag_configuration_verified_cases")
    )


def _artifact_export_contract_is_valid(value: Any) -> bool:
    try:
        return _artifact_export_contract_is_valid_unchecked(value)
    except (
        AttributeError,
        KeyError,
        OverflowError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        return False


def compare_dev_artifact_payloads(
    first: Any, second: Any
) -> dict[str, Any]:
    """Fail closed unless two dev artifacts are complete and self-consistent."""

    reasons: list[str] = []
    fingerprints: list[str] = []
    timestamps: list[tuple[datetime, datetime]] = []
    case_identity_sets: list[tuple[set[str], set[str]]] = []
    current_bundle_hash = _local_service_artifact_sha256()
    dev_oracle = _authenticated_dev_oracle()
    for label, supplied_payload in (("first", first), ("second", second)):
        if type(supplied_payload) is not dict:
            reasons.append(f"{label}_artifact_invalid")
            continue
        try:
            payload = json.loads(
                _canonical_json_bytes(supplied_payload),
                parse_constant=_reject_json_constant,
                object_pairs_hook=_reject_duplicate_json_object,
            )
        except (
            OverflowError,
            RecursionError,
            TypeError,
            UnicodeError,
            ValueError,
        ):
            reasons.append(f"{label}_artifact_invalid")
            continue
        schema = payload.get("schema_version")
        if type(schema) is str and schema in HISTORICAL_ARTIFACT_SCHEMAS:
            reasons.append(f"{label}_artifact_historical_only")
            continue
        if type(schema) is not str or schema != ARTIFACT_SCHEMA:
            reasons.append(f"{label}_artifact_schema_invalid")
            continue
        if not _artifact_export_contract_is_valid(payload):
            reasons.append(f"{label}_artifact_shape_invalid")
        if payload.get("dataset_id") != DATASET_ID or payload.get("split") != "dev":
            reasons.append(f"{label}_artifact_not_dev_fixture")
        if payload.get("execution_source") != "live_http_service":
            reasons.append(f"{label}_execution_source_invalid")
        fixture = payload.get("fixture")
        if (
            type(fixture) is not dict
            or set(fixture) != {"manifest_sha256", "freeze_sha256"}
            or fixture.get("manifest_sha256") != PINNED_MANIFEST_SHA256
            or fixture.get("freeze_sha256") != PINNED_FREEZE_SHA256
        ):
            reasons.append(f"{label}_fixture_identity_invalid")

        cases = payload.get("cases")
        case_ids_are_safe = (
            type(cases) is list
            and all(
                type(case) is dict and type(case.get("case_id")) is str
                for case in cases
            )
        )
        cases_valid = (
            case_ids_are_safe
            and len(cases) == 8
            and {case["case_id"] for case in cases} == DEV_CASE_IDS
            and dev_oracle is not None
            and all(
                _qualified_case_is_well_formed(case, dev_oracle)
                for case in cases
            )
        )
        if not cases_valid:
            reasons.append(f"{label}_case_results_incomplete")
        else:
            project_refs = {case["project_ref_hash"] for case in cases}
            run_refs = {case["run_ref_hash"] for case in cases}
            if len(project_refs) != len(cases) or len(run_refs) != len(cases):
                reasons.append(f"{label}_case_identity_reused")
            else:
                case_identity_sets.append((project_refs, run_refs))
        computed_summary = _build_summary(cases) if cases_valid else None
        if payload.get("summary") != computed_summary:
            reasons.append(f"{label}_summary_not_reproducible_from_cases")

        fingerprint = payload.get("reproducibility_fingerprint")
        if type(fingerprint) is not str or _SHA256.fullmatch(fingerprint) is None:
            reasons.append(f"{label}_reproducibility_fingerprint_invalid")
        else:
            fingerprints.append(fingerprint)
        safe_configuration = payload.get("safe_configuration")
        configuration_valid = not (
            type(safe_configuration) is not dict
            or set(safe_configuration) != _SAFE_CONFIGURATION_KEYS
            or payload.get("safe_configuration_fingerprint") != fingerprint
            or _sha256_bytes(_canonical_json_bytes(safe_configuration))
            != fingerprint
            or safe_configuration.get("artifact_schema") != ARTIFACT_SCHEMA
            or safe_configuration.get("dataset_id") != DATASET_ID
            or safe_configuration.get("split") != "dev"
            or safe_configuration.get("runner_git") != payload.get("git")
            or safe_configuration.get("runner_service_artifact_sha256")
            != current_bundle_hash
            or safe_configuration.get("manifest_sha256")
            != PINNED_MANIFEST_SHA256
            or safe_configuration.get("freeze_sha256") != PINNED_FREEZE_SHA256
            or type(safe_configuration.get("service_observation")) is not dict
            or set(safe_configuration.get("service_observation", {}))
            != {
                "service_ok",
                "model_configured",
                "thinking_configured",
                "thinking_mode",
                "runtime_provenance",
            }
            or safe_configuration["service_observation"].get("service_ok")
            is not True
            or safe_configuration["service_observation"].get("model_configured")
            is not True
        )
        if not configuration_valid:
            reasons.append(f"{label}_configuration_fingerprint_mismatch")

        provenance_gate = payload.get("provenance_gate")
        api_provenance = (
            _safe_runtime_provenance(
                safe_configuration["service_observation"].get(
                    "runtime_provenance"
                )
            )
            if configuration_valid
            else None
        )
        runner_bundle_hash = (
            safe_configuration.get("runner_service_artifact_sha256")
            if configuration_valid
            else None
        )
        expected_provenance_gate = (
            _build_runtime_provenance_gate(
                api_provenance,
                cases,
                payload.get("git") if type(payload.get("git")) is dict else {},
                runner_bundle_hash,
            )
            if cases_valid
            else None
        )
        if (
            type(provenance_gate) is not dict
            or set(provenance_gate) != _PROVENANCE_GATE_KEYS
            or provenance_gate.get("passed") is not True
            or provenance_gate != expected_provenance_gate
            or not configuration_valid
            or safe_configuration.get(
                "worker_runtime_provenance_matches_api_cases"
            )
            != provenance_gate.get("worker_runtime_provenance_matches_api_cases")
            or safe_configuration.get("rag_configuration_verified_cases")
            != provenance_gate.get("rag_configuration_verified_cases")
            or safe_configuration.get("service_reported_effective_limits")
            != _service_reported_effective_limits(cases if cases_valid else [])
        ):
            reasons.append(f"{label}_provenance_gate_not_reproducible")

        development_gate = payload.get("development_gate")
        expected_development_gate = (
            _build_development_gate(
                "dev", computed_summary, expected_provenance_gate
            )
            if computed_summary is not None
            and expected_provenance_gate is not None
            else None
        )
        if (
            type(development_gate) is not dict
            or set(development_gate) != _DEVELOPMENT_GATE_KEYS
            or development_gate.get("applicable") is not True
            or development_gate.get("passed") is not True
            or development_gate != expected_development_gate
        ):
            reasons.append(f"{label}_development_gate_not_reproducible")
        try:
            started = datetime.fromisoformat(payload["started_at"])
            completed = datetime.fromisoformat(payload["completed_at"])
            if started.tzinfo is None or completed.tzinfo is None or completed < started:
                raise ValueError
            timestamps.append((started, completed))
        except (KeyError, TypeError, ValueError):
            reasons.append(f"{label}_timestamps_invalid")

    if len(fingerprints) == 2 and fingerprints[0] != fingerprints[1]:
        reasons.append("runtime_configuration_changed_between_dev_runs")
    if len(case_identity_sets) == 2 and (
        case_identity_sets[0][0] & case_identity_sets[1][0]
        or case_identity_sets[0][1] & case_identity_sets[1][1]
    ):
        reasons.append("dev_case_identity_reused_between_runs")
    if len(timestamps) == 2 and timestamps[1][0] < timestamps[0][1]:
        reasons.append("dev_runs_overlap_or_are_out_of_order")
    return {
        "schema_version": DEV_PAIR_SCHEMA,
        "passed": not reasons,
        "reason_codes": reasons,
        "shared_reproducibility_fingerprint": (
            fingerprints[0]
            if len(fingerprints) == 2 and fingerprints[0] == fingerprints[1]
            else None
        ),
        "boundary": (
            "This checks that two developer-visible artifacts are internally "
            "self-consistent and match the evaluator's current source bundle. It is "
            "not a signature, authenticity proof, or blind benchmark."
        ),
    }


def compare_dev_artifacts(first_path: Path, second_path: Path) -> dict[str, Any]:
    payloads: list[Any] = []
    for path in (first_path, second_path):
        try:
            resolved = path.resolve(strict=True)
            if (
                not resolved.is_file()
                or not 1 <= resolved.stat().st_size <= MAX_JSON_BYTES
            ):
                raise OSError
            payloads.append(
                json.loads(
                    resolved.read_text(encoding="utf-8"),
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_reject_duplicate_json_object,
                )
            )
        except (
            OSError,
            OverflowError,
            RecursionError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            ValueError,
        ):
            payloads.append(None)
    return compare_dev_artifact_payloads(*payloads)


def _default_artifact_path(split: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return DEFAULT_ARTIFACT_ROOT / f"{split}-{stamp}-{uuid4().hex[:8]}.json"


def write_artifact_exclusive(path: Path, artifact: dict[str, Any]) -> str:
    target = _validate_artifact_target(path)
    payload = _canonical_json_bytes(artifact)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise LiveEvaluationError("artifact_exists") from None
    except OSError:
        raise LiveEvaluationError("internal_runner_error") from None
    return _sha256_bytes(payload)


def _validate_artifact_target(path: Path) -> Path:
    try:
        target = path.resolve()
        dataset = DATASET_ROOT.resolve()
    except OSError:
        raise LiveEvaluationError("configuration_invalid") from None
    if target == dataset or dataset in target.parents:
        raise LiveEvaluationError("configuration_invalid")
    if target.exists():
        raise LiveEvaluationError("artifact_exists")
    return target


def run_live_evaluation(
    options: RunnerOptions,
    *,
    api: JsonApi | None = None,
    dataset_root: Path = DATASET_ROOT,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], str]:
    if not options.confirm_live_provider:
        raise LiveEvaluationError("confirmation_required")
    if (
        options.split not in {"dev", "holdout"}
        or not 1 <= options.case_timeout_seconds <= 3600
        or not 0.1 <= options.request_timeout_seconds <= 120
        or not 0.05 <= options.poll_interval_seconds <= 30
    ):
        raise LiveEvaluationError("configuration_invalid")
    _normalize_base_url(options.base_url)
    _validate_artifact_target(options.artifact_path)
    runner_service_artifact_sha256 = _local_service_artifact_sha256()
    if runner_service_artifact_sha256 is None:
        raise LiveEvaluationError("service_not_ready")
    git_before = _git_state()
    # Freeze verification intentionally occurs before construction/use of the
    # HTTP client.  A modified holdout can therefore never create a project.
    plans, manifest_hash, freeze_hash = load_case_plans(
        options.split, dataset_root=dataset_root
    )
    prepared_api = api or UrllibJsonApi(options.base_url)
    health = _safe_health(
        prepared_api.request_json(
            "GET",
            "/health",
            payload=None,
            timeout=options.request_timeout_seconds,
        )
    )
    if (
        not health["service_ok"]
        or not health["model_configured"]
        or health["runtime_provenance"] is None
        or health["runtime_provenance"]["build"].get(
            "service_artifact_sha256"
        )
        != runner_service_artifact_sha256
    ):
        raise LiveEvaluationError("service_not_ready")

    started_at = _utc_now()
    results = [
        _run_one_case(
            plan,
            api=prepared_api,
            options=options,
            monotonic=monotonic,
            sleep=sleep,
        )
        for plan in plans
    ]
    git_state = _stable_git_state(git_before, _git_state())
    effective_limits = _service_reported_effective_limits(results)
    summary = _build_summary(results)
    provenance_gate = _build_runtime_provenance_gate(
        health["runtime_provenance"],
        results,
        git_state,
        runner_service_artifact_sha256,
    )
    development_gate = _build_development_gate(
        options.split, summary, provenance_gate
    )
    safe_config = {
        "artifact_schema": ARTIFACT_SCHEMA,
        "dataset_id": DATASET_ID,
        "split": options.split,
        "manifest_sha256": manifest_hash,
        "freeze_sha256": freeze_hash,
        "case_timeout_seconds": options.case_timeout_seconds,
        "request_timeout_seconds": options.request_timeout_seconds,
        "poll_interval_seconds": options.poll_interval_seconds,
        "runner_git": git_state,
        "runner_service_artifact_sha256": runner_service_artifact_sha256,
        "service_observation": health,
        "service_reported_effective_limits": effective_limits,
        "worker_runtime_provenance_matches_api_cases": provenance_gate[
            "worker_runtime_provenance_matches_api_cases"
        ],
        "rag_configuration_verified_cases": provenance_gate[
            "rag_configuration_verified_cases"
        ],
    }
    config_fingerprint = _sha256_bytes(_canonical_json_bytes(safe_config))
    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "dataset_id": DATASET_ID,
        "split": options.split,
        "execution_source": "live_http_service",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "fixture": {
            "manifest_sha256": manifest_hash,
            "freeze_sha256": freeze_hash,
        },
        "git": git_state,
        "safe_configuration": safe_config,
        "safe_configuration_fingerprint": config_fingerprint,
        "reproducibility_fingerprint": config_fingerprint,
        "provenance_gate": provenance_gate,
        "development_gate": development_gate,
        "summary": summary,
        "cases": results,
        "privacy_boundary": dict(_PRIVACY_BOUNDARY),
        "diagnostic_boundary": _DIAGNOSTIC_BOUNDARY,
    }
    if not _artifact_export_contract_is_valid(artifact):
        raise LiveEvaluationError("internal_runner_error")
    artifact_hash = write_artifact_exclusive(options.artifact_path, artifact)
    return artifact, artifact_hash


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Evidence Investigator fixture through a live LoreGuard HTTP "
            "service. This can consume real provider quota."
        )
    )
    parser.add_argument("--split", required=True, choices=("dev", "holdout"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--case-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--poll-interval-seconds", type=float, default=1.0)
    parser.add_argument(
        "--confirm-live-provider",
        action="store_true",
        help="Required acknowledgement that this run may invoke and bill the live provider.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    artifact_path = args.artifact or _default_artifact_path(args.split)
    options = RunnerOptions(
        split=args.split,
        base_url=args.base_url,
        artifact_path=artifact_path,
        case_timeout_seconds=args.case_timeout_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
        confirm_live_provider=args.confirm_live_provider,
    )
    try:
        artifact, artifact_hash = run_live_evaluation(options)
    except LiveEvaluationError as exc:
        print(f"live evaluation refused or failed: {exc.code}", file=sys.stderr)
        return 2
    print(
        f"saved {artifact['summary']['case_count']} sanitized case results to "
        f"{artifact_path.resolve()} (sha256={artifact_hash})"
    )
    return 0 if artifact["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
