"""Evaluate the author-approved character-axis workflow through the HTTP API.

This is a developer-visible test, not a blind benchmark. The review plan is
loaded before the model runs; the separate oracle is parsed only for scoring.
No credential, story text, prompt, provider response, or endpoint is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.character_trait_extraction import SupportTraceV1, stable_trait_identity
from scripts.run_character_consistency_live import (
    _baseline_admission,
    _candidate_id_sha256,
    _resolve_output_json,
    _safe_case_trace_summary,
    _safe_token_admission_events,
    _safe_visible_issue,
)
from scripts.run_evidence_investigator_live import (
    _CAPABILITY_KEYS,
    _CHARACTER_CONSISTENCY_INTEGER_LIMIT_BOUNDS,
    _CHARACTER_CONSISTENCY_LIMIT_KEYS,
    _CHARACTER_SIGNAL_CORE_SCOPE_KEY,
    _CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY,
    _CHARACTER_SIGNAL_SUPPORT_ID_KEY,
    _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_KEY,
    _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_VERSION,
    _CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY,
    _CHARACTER_SIGNAL_SEMANTIC_SCOPE_VERSION_KEY,
    _CHARACTER_SIGNAL_SEMANTIC_SCOPE_VERSION,
    _CHARACTER_SIGNAL_SCOPE_REVIEW_KEY,
    _CHARACTER_SIGNAL_SCOPE_REVIEW_KEYS,
    _valid_character_scope_review_limits,
    _CHARACTER_SIGNAL_SUPPORT_TRACE_KEY,
    _CHARACTER_SIGNAL_SUPPORT_TRACE_VERSION_KEY,
    _CHARACTER_SIGNAL_SUPPORT_TRACE_VERSION,
    _CHARACTER_CONSISTENCY_NUMBER_LIMIT_BOUNDS,
    _local_service_artifact_sha256,
)


DATASET = ROOT / "data" / "character-axis-challenge-v1"
DATASET_V2 = ROOT / "data" / "character-axis-challenge-v2"
SUITES = ("dev", "transfer")
BASELINE_FILES = (
    ("01-world-setting.md", "canon"),
    ("02-character-profiles.md", "character_profile"),
    ("03-published-history-v1.0.md", "chapter"),
)
DRAFT_FILE = "04-draft-event-v1.1.md"
FROZEN_FILES = frozenset(
    [name for name, _ in BASELINE_FILES]
    + [DRAFT_FILE, "review-plan.json", "oracle.json"]
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_HASH = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
SAFE_KEY = re.compile(r"[A-Za-z0-9_.:-]{1,100}\Z")
SUPPORT_ID = re.compile(r"L[1-9][0-9]{0,7}:A[1-9][0-9]{0,2}\Z")
SUPPORT_LOCATION_DIAGNOSTIC_V1 = "support-location-diagnostic-v1"
SAFE_REASON_KEYS = frozenset({
    "source_formal", "source_history",
    "regenerated_from_evidence_mismatch", "evidence_mismatch",
    "regenerated_from_core_label_scope", "core_label_scope",
    "support_index_invalid", "support_id_invalid", "support_label_scope",
    "regenerated_from_support_id_invalid", "regenerated_from_support_label_scope",
    "scope_anchor_invalid", "scope_relation_invalid",
    "regenerated_from_scope_anchor_invalid", "regenerated_from_scope_relation_invalid",
    "semantic_scope_unresolved", "regenerated_from_semantic_scope_unresolved",
    "scope_review_source_mismatch", "scope_review_request_invalid",
    "scope_review_internal_error", "scope_review_token_budget",
    "scope_review_deadline", "scope_review_provider_timeout",
    "scope_review_provider_rate_limit", "scope_review_provider_error",
    "scope_review_response_too_large", "scope_review_response_invalid",
    "scope_review_response_mismatch", "scope_review_reviewer_rejected",
    "scope_review_reviewer_uncertain", "scope_review_basis_invalid",
    "scope_review_slot_conflict",
    "character_support", "regenerated_from_character_support",
    "key_object_support", "statement_support",
    "lower_authority_baseline_shadowed", "invalid_confirmed_trait_snapshot",
    "chunk_limit", "confirmed_trait_hint_ambiguous",
    "confirmed_trait_context_truncated", "stage_token_budget",
    "targeted_target_limit", "targeted_reviewer_budget_reserve",
    "targeted_verification_candidate_line_limit",
    "targeted_verification_reviewer_budget_reserve",
    "approved_axis_ambiguous_evidence", "candidate_limit",
    "candidate_scope", "candidate_persistence",
    "targeted_no_supported_observation", "no_draft_signals",
    "no_confirmed_character_traits", "ambiguous_character_alias",
    "unmatched_character_alias", "object_baseline_identity_unavailable",
    "drift_scope_unknown", "lower_authority_draft_scope_shadowed",
    "drift_release_unknown", "drift_release_inapplicable",
    "observation_limit", "below_sensitivity_threshold", "baseline_limit",
})
SAFE_SIGNAL_SOURCE_KINDS = frozenset({
    "formal_character_profile", "published_history", "draft",
})
SAFE_SIGNAL_STABILITIES = frozenset({
    "core", "stable", "temporary", "situational", "unknown",
})
SAFE_SIGNAL_DIMENSIONS = frozenset({
    "core_personality", "preference", "value", "speech_pattern",
    "behavior_boundary", "contextual_behavior", "current_state",
})
SAFE_CANDIDATE_ELIGIBILITY_KEYS = frozenset({
    "stable_or_core_formal_signals", "stable_or_core_history_signals",
    "prelimit_candidates",
})
SAFE_EVIDENCE_MISMATCH_CATEGORIES = frozenset({
    "presentation_difference", "unique_other_line", "multiline_omission",
    "source_excerpt", "other",
})
SAFE_CORE_LABEL_SCOPE_CATEGORIES = frozenset({
    "selected_other_assertion", "selected_literal_unbound",
    "anchor_unresolved", "other",
})
SAFE_EVIDENCE_MISMATCH_ROLES = frozenset({
    "chapter", "canon", "character_profile", "reference", "unknown",
})
SAFE_EVIDENCE_MISMATCH_PHASES = frozenset({
    "primary_extraction", "targeted_recall", "targeted_verification",
})
SAFE_EVIDENCE_MISMATCH_OUTCOMES = frozenset({
    "disabled", "completed", "partial", "degraded", "skipped",
})
TERMINAL = {"completed", "failed", "cancelled"}
SAFE_PUBLIC_STAGE_OUTCOMES = frozenset({
    "disabled", "skipped", "degraded", "partial", "completed",
})
SAFE_PUBLIC_STAGE_REASONS = frozenset({
    "feature_disabled", "invalid_remaining_budget", "run_token_budget",
    "no_eligible_frozen_documents", "model_stage_unavailable",
    "bounded_partial", "completed", "internal_failure",
})
SAFE_PUBLIC_MATERIAL_COVERAGE = frozenset({
    "unknown", "partial", "complete",
})
SAFE_PUBLIC_CASE_OUTCOMES = frozenset({
    "conflict", "needs_confirmation", "no_issue", "unverifiable",
})
SAFE_PUBLIC_REVIEW_VERDICTS = frozenset({
    "contradicts", "explained", "needs_confirmation", "insufficient_evidence",
})
PUBLIC_RUN_COUNTER_LIMITS = {
    "prompt_tokens": 100_000_000,
    "completion_tokens": 100_000_000,
    "planned_chunks": 1_000_000,
    "processed_chunks": 1_000_000,
    "draft_observations": 1_000_000,
    "targeted_record_rejection_events": 1_000_000,
}
PUBLIC_STAGE_USAGE_LIMITS = {
    "attempted_calls": 1_000_000,
    "input_tokens": 100_000_000,
    "completion_tokens": 100_000_000,
    "charged_tokens": 100_000_000,
}
# Each official fixture has its own committed digest. Custom fixtures require
# an explicit CLI digest and remain ineligible for strict quality claims.
PINNED_MANIFEST_SHA256: str | None = (
    "650dad19bf54fad726b5c0f2f5dfb94d05fc4462babbbb558b0e9aaf40f61140"
)
PINNED_MANIFEST_SHA256_V2 = (
    "3d428f156d2b5fd752a46eddb8a23ebeefff83cb221c2d02eaa8ac87891755b3"
)


class SafeFailure(RuntimeError):
    def __init__(self, code: str, stage: str, **counts: int | str | bool):
        super().__init__(code)
        self.payload = {"code": code, "stage": stage, "details": counts}


@dataclass(frozen=True)
class VerifiedSuite:
    name: str
    world_id: str
    case_count: int
    files: dict[str, bytes]
    hashes: dict[str, str]
    plan: dict[str, Any]


@dataclass(frozen=True)
class VerifiedFixture:
    manifest_sha256: str
    suites: dict[str, VerifiedSuite]


def _sha256(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _json_object(data: bytes, *, stage: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise SafeFailure("invalid_json", stage) from exc
    if not isinstance(value, dict):
        raise SafeFailure("invalid_json_object", stage)
    return value


def _safe_key(value: Any) -> bool:
    return isinstance(value, str) and SAFE_KEY.fullmatch(value) is not None


def _source_lines(data: bytes) -> list[str]:
    try:
        return data.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise SafeFailure("invalid_document_encoding", "fixture_preflight") from exc


def _validate_plan(plan: dict[str, Any], suite: VerifiedSuite) -> None:
    if (
        plan.get("schema_version") != "character-axis-review-plan-v1"
        or plan.get("suite") != suite.name
        or plan.get("world_id") != suite.world_id
        or not isinstance(plan.get("approved_axes"), list)
        or not isinstance(plan.get("candidate_decisions"), list)
    ):
        raise SafeFailure("review_plan_contract", "fixture_preflight")
    axes: set[str] = set()
    for row in plan["approved_axes"]:
        if (
            not isinstance(row, dict)
            or not _safe_key(row.get("axis_key"))
            or row["axis_key"] in axes
            or row.get("trait_type") != "core_personality"
            or not isinstance(row.get("display_name"), str)
            or not 1 <= len(row["display_name"]) <= 80
            or not isinstance(row.get("definition"), str)
            or not 1 <= len(row["definition"]) <= 200
        ):
            raise SafeFailure("review_plan_axis_contract", "fixture_preflight")
        axes.add(row["axis_key"])
    keys: set[str] = set()
    baseline_names = {name for name, _ in BASELINE_FILES}
    for row in plan["candidate_decisions"]:
        if not isinstance(row, dict):
            raise SafeFailure("review_plan_candidate_contract", "fixture_preflight")
        key = row.get("candidate_key")
        source_name = row.get("source_document")
        line = row.get("source_line")
        source_quote = row.get("source_quote")
        dimension = row.get("trait_type")
        axis_key = row.get("approved_axis_key")
        object_key = row.get("key_object")
        if (
            not _safe_key(key) or key in keys
            or not isinstance(row.get("character_key"), str)
            or not 1 <= len(row["character_key"]) <= 64
            or dimension not in {
                "core_personality", "speech_pattern", "preference", "value",
                "behavior_boundary", "current_state", "contextual_behavior",
            }
            or source_name not in baseline_names
            or type(line) is not int or line < 1
            or not isinstance(source_quote, str) or not source_quote.strip()
            or len(source_quote) > 200
            or row.get("polarity") not in {"positive", "negative", "neutral"}
            or row.get("stability") not in {"core", "stable"}
            or (object_key is not None and (not isinstance(object_key, str) or not object_key.strip()))
            or (dimension == "core_personality") != (axis_key in axes)
            or (dimension != "core_personality" and axis_key is not None)
        ):
            raise SafeFailure("review_plan_candidate_contract", "fixture_preflight")
        source_lines = _source_lines(suite.files[source_name])
        if line > len(source_lines) or source_quote not in source_lines[line - 1]:
            raise SafeFailure("review_plan_source_anchor", "fixture_preflight")
        keys.add(key)
    if not keys:
        raise SafeFailure("review_plan_empty", "fixture_preflight")


def verify_fixture(
    dataset: Path = DATASET,
    *,
    expected_manifest_sha256: str | None = PINNED_MANIFEST_SHA256,
) -> VerifiedFixture:
    """Hash every frozen input before opening an HTTP client or scoring gold."""
    if not isinstance(expected_manifest_sha256, str) or SHA256.fullmatch(expected_manifest_sha256) is None:
        raise SafeFailure("manifest_digest_required", "fixture_preflight")
    manifest_path = dataset / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise SafeFailure("manifest_unavailable", "fixture_preflight") from exc
    manifest_hash = _sha256(manifest_bytes)
    if manifest_hash != expected_manifest_sha256:
        raise SafeFailure("manifest_hash_mismatch", "fixture_preflight")
    manifest = _json_object(manifest_bytes, stage="fixture_preflight")
    boundary = manifest.get("dataset_boundary")
    entries = manifest.get("suites")
    if (
        manifest.get("schema_version") != "character-axis-fixture-manifest-v1"
        or not isinstance(boundary, dict)
        or boundary.get("developer_visible") is not True
        or boundary.get("blind_holdout") is not False
        or boundary.get("production_quality") is not False
        or not isinstance(entries, dict)
        or set(entries) != set(SUITES)
    ):
        raise SafeFailure("manifest_contract", "fixture_preflight")
    verified: dict[str, VerifiedSuite] = {}
    for suite_name in SUITES:
        meta = entries[suite_name]
        if (
            not isinstance(meta, dict)
            or not _safe_key(meta.get("world_id"))
            or type(meta.get("case_count")) is not int
            or not 1 <= meta["case_count"] <= 100
            or not isinstance(meta.get("files"), dict)
            or set(meta["files"]) != FROZEN_FILES
        ):
            raise SafeFailure("manifest_suite_contract", "fixture_preflight")
        contents: dict[str, bytes] = {}
        for filename in sorted(FROZEN_FILES):
            expected_hash = meta["files"][filename]
            if not isinstance(expected_hash, str) or SHA256.fullmatch(expected_hash) is None:
                raise SafeFailure("manifest_file_hash_contract", "fixture_preflight")
            path = dataset / suite_name / filename
            try:
                if path.is_symlink() or not path.is_file():
                    raise OSError("invalid fixture file")
                contents[filename] = path.read_bytes()
            except OSError as exc:
                raise SafeFailure("fixture_file_unavailable", "fixture_preflight") from exc
            if _sha256(contents[filename]) != expected_hash:
                raise SafeFailure("fixture_file_hash_mismatch", "fixture_preflight")
        suite = VerifiedSuite(
            name=suite_name,
            world_id=meta["world_id"],
            case_count=meta["case_count"],
            files=contents,
            hashes=dict(meta["files"]),
            plan=_json_object(contents["review-plan.json"], stage="fixture_preflight"),
        )
        for name, _ in BASELINE_FILES:
            _source_lines(contents[name])
        _source_lines(contents[DRAFT_FILE])
        _validate_plan(suite.plan, suite)
        verified[suite_name] = suite
    return VerifiedFixture(manifest_sha256=manifest_hash, suites=verified)


def _request(client: httpx.Client, method: str, path: str, route: str, **kwargs: Any) -> Any:
    try:
        response = client.request(method, path, **kwargs)
    except httpx.HTTPError as exc:
        raise SafeFailure("http_transport", route) from exc
    if response.status_code >= 400:
        raise SafeFailure("http_status", route, status_code=response.status_code)
    try:
        return response.json()
    except ValueError as exc:
        raise SafeFailure("http_json", route) from exc


def _context(*, published: bool) -> dict[str, Any]:
    return {
        "resolution_state": "confirmed",
        "publication_status": "published" if published else "draft",
        "scope": {
            "schema_version": 1,
            "timeline_key": "main",
            "release": {"key": "v1.0" if published else "v1.1", "ordinal": 10 if published else 11},
            "branch": {"path": ["main"]},
        },
    }


def _upload(client: httpx.Client, project_id: str, suite: VerifiedSuite, filename: str, role: str, *, published: bool) -> str:
    try:
        content = suite.files[filename].decode("utf-8")
    except UnicodeError as exc:
        raise SafeFailure("invalid_document_encoding", "upload") from exc
    payload = _request(
        client, "POST", f"/api/v1/projects/{project_id}/documents/text", "upload",
        json={
            "name": filename, "content": content, "document_role": role,
            "story_scope": "main", "narrative_context": _context(published=published),
        },
    )
    if not isinstance(payload, dict) or not isinstance(payload.get("id"), str):
        raise SafeFailure("document_response_contract", "upload")
    return payload["id"]


def _wait_run(client: httpx.Client, run_id: str, *, timeout_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    while True:
        value = _request(client, "GET", f"/api/v1/analysis-runs/{run_id}", "run_status")
        if not isinstance(value, dict):
            raise SafeFailure("run_response_contract", "run_status")
        if value.get("status") in TERMINAL:
            value["_elapsed_seconds"] = round(time.monotonic() - started, 3)
            return value
        if time.monotonic() - started >= timeout_seconds:
            raise SafeFailure("run_timeout", "run_status")
        time.sleep(1)


def _start_run(client: httpx.Client, project_id: str, *, mode: str, target_id: str | None = None) -> str:
    body: dict[str, Any] = {"mode": mode, "sensitivity": "balanced"}
    if target_id is not None:
        body["target_document_ids"] = [target_id]
    response = _request(
        client, "POST", f"/api/v1/projects/{project_id}/analysis-runs", "start_run",
        headers={"Idempotency-Key": f"axis-live-{uuid4().hex}"}, json=body,
    )
    if not isinstance(response, dict) or not isinstance(response.get("id"), str):
        raise SafeFailure("run_create_contract", "start_run")
    return response["id"]


def _safe_citation_refs(value: object, *, known_documents: set[str]) -> list[dict[str, Any]] | None:
    """Return None for old/incomplete diagnostics; never infer missing refs."""
    if not isinstance(value, list):
        return None
    refs: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            return None
        role = row.get("role")
        name = row.get("document_name")
        start = row.get("line_start")
        end = row.get("line_end")
        if (
            role not in {"B", "C", "G", "X"}
            or name not in known_documents
            or type(start) is not int or type(end) is not int
            or not 1 <= start <= end <= 10_000_000
        ):
            return None
        refs.append({"role": role, "document_name": name, "line_start": start, "line_end": end})
    return refs if len(refs) <= 8 else None


def _actor_digest(value: str) -> str:
    normalized = re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()
    return _sha256(normalized)


def _safe_accepted_draft_refs(
    stage: dict[str, Any], *, known_documents: set[str],
) -> list[dict[str, Any]] | None:
    """Project the complete accepted-signal set without actor names or prose."""
    value = stage.get("accepted_draft_observation_refs")
    total = stage.get("accepted_draft_observation_total")
    if (
        not isinstance(value, list) or len(value) > 64
        or type(total) is not int or total != len(value)
        or stage.get("accepted_draft_observation_refs_truncated") is not False
    ):
        return None
    refs: list[dict[str, Any]] = []
    for row in value:
        if not isinstance(row, dict):
            return None
        actor = row.get("character_key")
        document = row.get("document_name")
        start = row.get("line_start")
        end = row.get("line_end")
        dimension = row.get("dimension")
        polarity = row.get("polarity")
        if (
            not isinstance(actor, str) or not 1 <= len(actor) <= 64
            or actor.startswith("redacted_")
            or document not in known_documents
            or type(start) is not int or type(end) is not int
            or not 1 <= start <= end <= 10_000_000
            or dimension not in {
                "core_personality", "preference", "value", "speech_pattern",
                "behavior_boundary", "contextual_behavior", "current_state",
            }
            or polarity not in {"positive", "negative", "neutral", "unclear"}
        ):
            return None
        refs.append({
            "actor_sha256": _actor_digest(actor),
            "dimension": dimension,
            "polarity": polarity,
            "document_name": document,
            "line_start": start,
            "line_end": end,
        })
    return refs


def _safe_accepted_signal_diagnostics(
    stage: dict[str, Any], counts: dict[str, Any],
) -> tuple[list[dict[str, str | int]] | None, dict[str, int] | None]:
    """Keep only bounded enum/count aggregates, never model text or identity."""
    raw_histogram = stage.get("accepted_signal_histogram")
    raw_eligibility = stage.get("candidate_eligibility")
    if (
        not isinstance(raw_histogram, list)
        or len(raw_histogram) > (
            len(SAFE_SIGNAL_SOURCE_KINDS)
            * len(SAFE_SIGNAL_STABILITIES)
            * len(SAFE_SIGNAL_DIMENSIONS)
        )
        or not isinstance(raw_eligibility, dict)
        or set(raw_eligibility) != SAFE_CANDIDATE_ELIGIBILITY_KEYS
    ):
        return None, None
    if any(
        type(value) is not int or not 0 <= value <= 100_000
        for value in raw_eligibility.values()
    ):
        return None, None
    histogram: list[dict[str, str | int]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in raw_histogram:
        if not isinstance(row, dict) or set(row) != {
            "source_kind", "stability", "dimension", "count"
        }:
            return None, None
        source_kind = row["source_kind"]
        stability = row["stability"]
        dimension = row["dimension"]
        count = row["count"]
        if (
            type(source_kind) is not str or source_kind not in SAFE_SIGNAL_SOURCE_KINDS
            or type(stability) is not str or stability not in SAFE_SIGNAL_STABILITIES
            or type(dimension) is not str or dimension not in SAFE_SIGNAL_DIMENSIONS
            or type(count) is not int or not 1 <= count <= 100_000
        ):
            return None, None
        identity = (source_kind, stability, dimension)
        if identity in seen:
            return None, None
        seen.add(identity)
        histogram.append({
            "source_kind": source_kind, "stability": stability,
            "dimension": dimension, "count": count,
        })
    signal_count = counts.get("signal_count")
    if signal_count is None and not histogram:
        pass  # Empty/degraded stage has no counts, but still uses this schema.
    elif (
        type(signal_count) is not int or not 0 <= signal_count <= 100_000
        or sum(row["count"] for row in histogram) != signal_count
    ):
        return None, None
    formal = sum(
        row["count"] for row in histogram
        if row["source_kind"] == "formal_character_profile"
        and row["stability"] in {"core", "stable"}
    )
    history = sum(
        row["count"] for row in histogram
        if row["source_kind"] == "published_history"
        and row["stability"] in {"core", "stable"}
    )
    if (
        raw_eligibility["stable_or_core_formal_signals"] != formal
        or raw_eligibility["stable_or_core_history_signals"] != history
        or raw_eligibility["prelimit_candidates"] > formal + history
    ):
        return None, None
    return sorted(
        histogram,
        key=lambda row: (row["source_kind"], row["stability"], row["dimension"]),
    ), dict(raw_eligibility)


def _safe_evidence_mismatch_diagnostics(
    stage: dict[str, Any],
) -> tuple[dict[str, int] | None, list[dict[str, Any]] | None, int | None]:
    """Project only fixed enums, ordinals and bounded counts from worker data."""
    raw_counts = stage.get("evidence_mismatch_counts")
    raw_chunks = stage.get("evidence_mismatch_chunks")
    omitted = stage.get("evidence_mismatch_chunks_omitted_count")
    if (
        not isinstance(raw_counts, dict)
        or not isinstance(raw_chunks, list) or len(raw_chunks) > 128
        or type(omitted) is not int or not 0 <= omitted <= 1_000_000
        or not set(raw_counts) <= SAFE_EVIDENCE_MISMATCH_CATEGORIES
        or any(type(value) is not int or not 1 <= value <= 1_000_000
               for value in raw_counts.values())
    ):
        return None, None, None
    safe_chunks: list[dict[str, Any]] = []
    emitted_counts: dict[str, int] = {}
    row_keys = {
        "source_document_ordinal", "document_chunk_ordinal", "stage_chunk_ordinal",
        "document_role", "source_kind", "phase", "target_ordinal", "outcome",
        "counts",
    }
    for row in raw_chunks:
        if not isinstance(row, dict) or set(row) != row_keys:
            return None, None, None
        counts = row["counts"]
        target = row["target_ordinal"]
        if (
            type(row["source_document_ordinal"]) is not int
            or not 0 <= row["source_document_ordinal"] <= 1_000_000
            or type(row["document_chunk_ordinal"]) is not int
            or not 1 <= row["document_chunk_ordinal"] <= 1_000_000
            or type(row["stage_chunk_ordinal"]) is not int
            or not 1 <= row["stage_chunk_ordinal"] <= 1_000_000
            or type(row["document_role"]) is not str
            or row["document_role"] not in SAFE_EVIDENCE_MISMATCH_ROLES
            or type(row["source_kind"]) is not str
            or row["source_kind"] not in SAFE_SIGNAL_SOURCE_KINDS
            or type(row["phase"]) is not str
            or row["phase"] not in SAFE_EVIDENCE_MISMATCH_PHASES
            or type(row["outcome"]) is not str
            or row["outcome"] not in SAFE_EVIDENCE_MISMATCH_OUTCOMES
            or (target is not None and (
                type(target) is not int or not 1 <= target <= 1_000_000
            ))
            or not isinstance(counts, dict) or not counts
            or not set(counts) <= SAFE_EVIDENCE_MISMATCH_CATEGORIES
            or any(type(value) is not int or not 1 <= value <= 1_000_000
                   for value in counts.values())
        ):
            return None, None, None
        safe_chunks.append({
            "source_document_ordinal": row["source_document_ordinal"],
            "document_chunk_ordinal": row["document_chunk_ordinal"],
            "stage_chunk_ordinal": row["stage_chunk_ordinal"],
            "document_role": row["document_role"],
            "source_kind": row["source_kind"],
            "phase": row["phase"],
            "target_ordinal": target,
            "outcome": row["outcome"],
            "counts": {key: counts[key] for key in sorted(counts)},
        })
        for key, value in counts.items():
            emitted_counts[key] = emitted_counts.get(key, 0) + value
    if (
        (omitted == 0 and emitted_counts != raw_counts)
        or (omitted > 0 and (
            not raw_counts
            or any(value > raw_counts.get(key, 0)
                   for key, value in emitted_counts.items())
        ))
    ):
        return None, None, None
    return (
        {key: raw_counts[key] for key in sorted(raw_counts)},
        safe_chunks,
        omitted,
    )


def _safe_core_label_scope_diagnostics(
    stage: dict[str, Any], safe_reasons: dict[str, int],
    *, safe_signal_histogram: list[dict[str, str | int]] | None,
    signal_count: object,
) -> tuple[dict[str, int] | None, int | None]:
    """Project bounded categories against the independently validated signal count."""
    raw_counts = stage.get("core_label_scope_counts")
    raw_reasons = stage.get("reason_counts")
    accepted_unlabeled = stage.get(
        "accepted_model_core_without_literal_label_count"
    )
    if (
        not isinstance(raw_counts, dict)
        or not isinstance(raw_reasons, dict)
        or not set(raw_counts) <= SAFE_CORE_LABEL_SCOPE_CATEGORIES
        or any(type(value) is not int or not 1 <= value <= 1_000_000
               for value in raw_counts.values())
        or type(accepted_unlabeled) is not int
        or not 0 <= accepted_unlabeled <= 1_000_000
        or safe_signal_histogram is None
        or type(signal_count) is not int
        or not 0 <= signal_count <= 100_000
    ):
        return None, None
    formal_core_upper_bound = sum(
        row["count"] for row in safe_signal_histogram
        if row["source_kind"] == "formal_character_profile"
        and (row["stability"] == "core" or row["dimension"] == "core_personality")
    )
    if (
        formal_core_upper_bound > signal_count
        or accepted_unlabeled > formal_core_upper_bound
        or accepted_unlabeled > signal_count
    ):
        return None, None
    for key in ("core_label_scope", "regenerated_from_core_label_scope"):
        value = raw_reasons.get(key, 0)
        if (
            type(value) is not int or not 0 <= value <= 1_000_000
            or value != safe_reasons.get(key, 0)
        ):
            return None, None
    expected = (
        safe_reasons.get("core_label_scope", 0)
        + safe_reasons.get("regenerated_from_core_label_scope", 0)
    )
    if sum(raw_counts.values()) != expected or expected > 1_000_000:
        return None, None
    return {key: raw_counts[key] for key in sorted(raw_counts)}, accepted_unlabeled


def _safe_character_runtime_provenance(value: object) -> dict[str, Any] | None:
    """Whitelist only model, character-stage, and build identity fields.

    Unlike the investigator validator, this deliberately permits an
    unconfigured embedding profile: character OOC does not require RAG.
    """
    if not isinstance(value, dict) or value.get("schema_version") != "loreguard-runtime-provenance-v3":
        return None
    build = value.get("build")
    provider = value.get("chat_provider")
    capabilities = value.get("capabilities")
    limits = value.get("character_consistency_limits")
    if not all(isinstance(row, dict) for row in (build, provider, capabilities, limits)):
        return None
    if set(build) != {"git_revision", "service_artifact_sha256"}:
        return None
    revision = build.get("git_revision")
    artifact = build.get("service_artifact_sha256")
    if (
        not isinstance(revision, str) or GIT_HASH.fullmatch(revision) is None
        or not isinstance(artifact, str) or SHA256.fullmatch(artifact) is None
    ):
        return None
    if set(provider) != {
        "model_alias", "endpoint_configuration_sha256", "temperature",
        "thinking_configured", "thinking_mode",
    }:
        return None
    alias = provider.get("model_alias")
    endpoint = provider.get("endpoint_configuration_sha256")
    thinking_mode = provider.get("thinking_mode")
    if (
        not isinstance(alias, str) or not 1 <= len(alias) <= 255
        or alias != alias.strip() or any(ord(char) < 32 for char in alias)
        or not isinstance(endpoint, str) or SHA256.fullmatch(endpoint) is None
        or type(provider.get("temperature")) not in {int, float}
        or provider["temperature"] != 0
        or type(provider.get("thinking_configured")) is not bool
        or thinking_mode not in {None, "disabled", "enabled"}
        or provider["thinking_configured"] != (thinking_mode is not None)
    ):
        return None
    if (
        set(capabilities) != _CAPABILITY_KEYS
        or any(type(flag) is not bool for flag in capabilities.values())
        or set(limits) not in {
            _CHARACTER_CONSISTENCY_LIMIT_KEYS,
            _CHARACTER_CONSISTENCY_LIMIT_KEYS
            | {_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY},
            _CHARACTER_CONSISTENCY_LIMIT_KEYS
            | {_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY, _CHARACTER_SIGNAL_CORE_SCOPE_KEY},
            _CHARACTER_CONSISTENCY_LIMIT_KEYS
            | {
                _CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY,
                _CHARACTER_SIGNAL_CORE_SCOPE_KEY,
                _CHARACTER_SIGNAL_SUPPORT_ID_KEY,
                _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_KEY,
            },
            _CHARACTER_CONSISTENCY_LIMIT_KEYS
            | {
                _CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY,
                _CHARACTER_SIGNAL_CORE_SCOPE_KEY,
                _CHARACTER_SIGNAL_SUPPORT_ID_KEY,
                _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_KEY,
                _CHARACTER_SIGNAL_SUPPORT_TRACE_KEY,
                _CHARACTER_SIGNAL_SUPPORT_TRACE_VERSION_KEY,
            },
            _CHARACTER_CONSISTENCY_LIMIT_KEYS
            | {
                _CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY,
                _CHARACTER_SIGNAL_CORE_SCOPE_KEY,
                _CHARACTER_SIGNAL_SUPPORT_ID_KEY,
                _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_KEY,
                _CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY,
                _CHARACTER_SIGNAL_SEMANTIC_SCOPE_VERSION_KEY,
            },
            _CHARACTER_CONSISTENCY_LIMIT_KEYS
            | {
                _CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY,
                _CHARACTER_SIGNAL_CORE_SCOPE_KEY,
                _CHARACTER_SIGNAL_SUPPORT_ID_KEY,
                _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_KEY,
                _CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY,
                _CHARACTER_SIGNAL_SEMANTIC_SCOPE_VERSION_KEY,
                _CHARACTER_SIGNAL_SUPPORT_TRACE_KEY,
                _CHARACTER_SIGNAL_SUPPORT_TRACE_VERSION_KEY,
            },
            _CHARACTER_CONSISTENCY_LIMIT_KEYS
            | {
                _CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY,
                _CHARACTER_SIGNAL_CORE_SCOPE_KEY,
                _CHARACTER_SIGNAL_SUPPORT_ID_KEY,
                _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_KEY,
                _CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY,
                _CHARACTER_SIGNAL_SEMANTIC_SCOPE_VERSION_KEY,
            }
            | _CHARACTER_SIGNAL_SCOPE_REVIEW_KEYS,
            _CHARACTER_CONSISTENCY_LIMIT_KEYS
            | {
                _CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY,
                _CHARACTER_SIGNAL_CORE_SCOPE_KEY,
                _CHARACTER_SIGNAL_SUPPORT_ID_KEY,
                _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_KEY,
                _CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY,
                _CHARACTER_SIGNAL_SEMANTIC_SCOPE_VERSION_KEY,
                _CHARACTER_SIGNAL_SUPPORT_TRACE_KEY,
                _CHARACTER_SIGNAL_SUPPORT_TRACE_VERSION_KEY,
            }
            | _CHARACTER_SIGNAL_SCOPE_REVIEW_KEYS,
        }
        or limits.get("sensitivity") not in {"conservative", "balanced", "exploratory"}
    ):
        return None
    if (
        _CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY in limits
        and type(limits[_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY]) is not bool
    ):
        return None
    if _CHARACTER_SIGNAL_CORE_SCOPE_KEY in limits and (
        type(limits[_CHARACTER_SIGNAL_CORE_SCOPE_KEY]) is not bool
        or (
            limits[_CHARACTER_SIGNAL_CORE_SCOPE_KEY]
            and limits[_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY] is not True
        )
    ):
        return None
    if _CHARACTER_SIGNAL_SUPPORT_ID_KEY in limits and (
        type(limits[_CHARACTER_SIGNAL_SUPPORT_ID_KEY]) is not bool
        or (
            limits[_CHARACTER_SIGNAL_SUPPORT_ID_KEY]
            and limits[_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY] is not True
        )
        or limits[_CHARACTER_SIGNAL_SUPPORT_SEGMENTER_KEY] != (
            _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_VERSION
            if limits[_CHARACTER_SIGNAL_SUPPORT_ID_KEY] else None
        )
    ):
        return None
    if _CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY in limits and (
        type(limits[_CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY]) is not bool
        or (
            limits[_CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY]
            and limits[_CHARACTER_SIGNAL_SUPPORT_ID_KEY] is not True
        )
        or limits[_CHARACTER_SIGNAL_SEMANTIC_SCOPE_VERSION_KEY] != (
            _CHARACTER_SIGNAL_SEMANTIC_SCOPE_VERSION
            if limits[_CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY] else None
        )
    ):
        return None
    if (
        _CHARACTER_SIGNAL_SCOPE_REVIEW_KEY in limits
        and not _valid_character_scope_review_limits(
            limits, allow_legacy_prompt_version=True,
        )
    ):
        return None
    if _CHARACTER_SIGNAL_SUPPORT_TRACE_KEY in limits and (
        type(limits[_CHARACTER_SIGNAL_SUPPORT_TRACE_KEY]) is not bool
        or (
            limits[_CHARACTER_SIGNAL_SUPPORT_TRACE_KEY]
            and limits[_CHARACTER_SIGNAL_SUPPORT_ID_KEY] is not True
        )
        or limits[_CHARACTER_SIGNAL_SUPPORT_TRACE_VERSION_KEY] != (
            _CHARACTER_SIGNAL_SUPPORT_TRACE_VERSION
            if limits[_CHARACTER_SIGNAL_SUPPORT_TRACE_KEY] else None
        )
    ):
        return None
    for key, (minimum, maximum) in _CHARACTER_CONSISTENCY_INTEGER_LIMIT_BOUNDS.items():
        number = limits.get(key)
        if type(number) is not int or not minimum <= number <= maximum:
            return None
    for key, (minimum, maximum) in _CHARACTER_CONSISTENCY_NUMBER_LIMIT_BOUNDS.items():
        number = limits.get(key)
        if (
            type(number) not in {int, float}
            or not math.isfinite(float(number))
            or not minimum < float(number) <= maximum
        ):
            return None
    for prefix in ("signal", "drift"):
        if (
            limits[f"{prefix}_provider_max_completion_tokens"]
            > limits[f"{prefix}_max_completion_tokens"]
            or limits[f"{prefix}_provider_max_response_bytes"]
            > limits[f"{prefix}_max_response_bytes"]
            or limits[f"{prefix}_provider_timeout_seconds"]
            > min(
                limits[f"{prefix}_timeout_seconds"],
                limits[f"{prefix}_total_deadline_seconds"],
            )
        ):
            return None
    return {
        "schema_version": value["schema_version"],
        "build": dict(build),
        "chat_provider": dict(provider),
        "capabilities": dict(capabilities),
        "character_consistency_limits": dict(limits),
    }


def _runtime_provenance_digest(value: object) -> str | None:
    safe = _safe_character_runtime_provenance(value)
    if safe is None:
        return None
    return _sha256(json.dumps(
        safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ))


def _safe_support_trace_chunks(
    stage: dict[str, Any],
) -> tuple[list[dict[str, Any]], int] | None:
    """Final HTTP boundary for the anonymous V4 trace; never echo API data."""

    raw = stage.get("support_trace_chunks")
    omitted = stage.get("support_trace_chunks_omitted_count")
    counts = stage.get("counts")
    processed = (
        stage.get("processed_chunks")
        if "processed_chunks" in stage
        else counts.get("processed_chunks")
        if type(counts) is dict
        else None
    )
    if (
        type(raw) is not list
        or len(raw) > 128
        or type(omitted) is not int
        or not 0 <= omitted <= 128
        or len(raw) + omitted > 128
        or (
            processed is not None
            and (type(processed) is not int or not 0 <= processed <= 128)
        )
    ):
        return None
    safe: list[dict[str, Any]] = []
    previous_ordinal = 0
    for row in raw:
        if type(row) is not dict or len(row) != 4 or set(row) != {
            "stage_chunk_ordinal", "outcome", "availability", "trace"
        }:
            return None
        ordinal = row["stage_chunk_ordinal"]
        outcome = row["outcome"]
        availability = row["availability"]
        if (
            type(ordinal) is not int
            or not previous_ordinal < ordinal <= 128
            or (processed is not None and ordinal > processed)
            or type(outcome) is not str
            or outcome not in {"disabled", "completed", "partial", "degraded", "skipped"}
            or type(availability) is not str
            or availability not in {"available", "unavailable"}
        ):
            return None
        previous_ordinal = ordinal
        if availability == "unavailable":
            if row["trace"] is not None:
                return None
            trace = None
        else:
            if type(row["trace"]) is not dict:
                return None
            raw_trace = row["trace"]
            if len(raw_trace) != 6 or set(raw_trace) != {
                "schema_version", "indexed_support_count", "attempts",
                "final_state", "final_accepted_slots", "candidate_transition",
            } or type(raw_trace["attempts"]) is not list or len(raw_trace["attempts"]) > 2:
                return None
            if (
                type(raw_trace["final_accepted_slots"]) is not list
                or len(raw_trace["final_accepted_slots"]) > 64
            ):
                return None
            for attempt in raw_trace["attempts"]:
                if type(attempt) is not dict or len(attempt) != 5 or set(attempt) != {
                    "attempt", "observability", "submitted_slots", "events",
                    "unbound_record_events",
                } or type(attempt["events"]) is not list or len(attempt["events"]) > 64:
                    return None
                if (
                    type(attempt["submitted_slots"]) is not list
                    or len(attempt["submitted_slots"]) > 64
                ):
                    return None
                if any(
                    type(event) is not dict
                    or len(event) != 3
                    or set(event) != {"slot", "outcome", "reason"}
                    for event in attempt["events"]
                ):
                    return None
            try:
                trace = SupportTraceV1.model_validate(raw_trace).model_dump(
                    mode="json"
                )
            except (ValueError, TypeError):
                return None
        safe.append({
            "stage_chunk_ordinal": ordinal,
            "outcome": outcome,
            "availability": availability,
            "trace": trace,
        })
    return safe, omitted


def _run_summary(client: httpx.Client, run: dict[str, Any], *, known_documents: set[str]) -> dict[str, Any]:
    run_id = run.get("id")
    if not isinstance(run_id, str):
        raise SafeFailure("run_response_contract", "run_summary")
    diagnostics = _request(client, "GET", f"/api/v1/analysis-runs/{run_id}/diagnostics", "diagnostics")
    issues = _request(client, "GET", f"/api/v1/analysis-runs/{run_id}/issues", "issues")
    if not isinstance(diagnostics, dict) or not isinstance(issues, list):
        raise SafeFailure("run_summary_contract", "run_summary")
    stage = diagnostics.get("character_consistency")
    stage = stage if isinstance(stage, dict) else {}
    counts = stage.get("counts")
    counts = counts if isinstance(counts, dict) else {}
    usage = stage.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    reasons = stage.get("reason_counts")
    reasons = reasons if isinstance(reasons, dict) else {}
    safe_reasons = {
        key: value for key, value in reasons.items()
        if key in SAFE_REASON_KEYS and type(value) is int and 0 <= value <= 1_000_000
    }
    signal_histogram, candidate_eligibility = _safe_accepted_signal_diagnostics(
        stage, counts
    )
    mismatch_counts, mismatch_chunks, mismatch_omitted = (
        _safe_evidence_mismatch_diagnostics(stage)
    )
    core_scope_counts, accepted_unlabeled_core = (
        _safe_core_label_scope_diagnostics(
            stage, safe_reasons,
            safe_signal_histogram=signal_histogram,
            signal_count=counts.get("signal_count"),
        )
    )
    accepted_draft_refs = _safe_accepted_draft_refs(
        stage, known_documents=known_documents
    )
    raw_trace = stage.get("case_trace")
    raw_trace = raw_trace if isinstance(raw_trace, list) else []
    safe_trace = []
    for row in raw_trace:
        if not isinstance(row, dict):
            continue
        safe = _safe_case_trace_summary(row)
        if row.get("observation_axis_binding") == "server_targeted_evidence":
            safe["observation_axis_binding"] = "server_targeted_evidence"
        definition_hash = row.get("approved_axis_definition_sha256")
        if isinstance(definition_hash, str) and SHA256.fullmatch(definition_hash):
            safe["approved_axis_definition_sha256"] = definition_hash
        # The server addition is optional. Absence or incomplete binding must
        # not be promoted into an exact G/X evidence score.
        safe["citation_refs"] = _safe_citation_refs(
            row.get("citation_refs"), known_documents=known_documents
        ) if (
            row.get("review_outcome") == "completed"
            and row.get("citation_refs_incomplete") is False
        ) else None
        safe_trace.append(safe)
    safe_issues = [
        _safe_visible_issue(row, case_trace=safe_trace)
        for row in issues if isinstance(row, dict) and row.get("category") == "character_drift"
    ]
    summary = {
        "run_id": run_id,
        "runtime_provenance_sha256": _runtime_provenance_digest(
            diagnostics.get("runtime_provenance")
        ),
        "status": run.get("status"),
        "elapsed_seconds": run.get("_elapsed_seconds"),
        "prompt_tokens": run.get("prompt_tokens"),
        "completion_tokens": run.get("completion_tokens"),
        "stage_outcome": stage.get("outcome"),
        "stage_reason": stage.get("reason_code"),
        "material_coverage": stage.get("material_coverage"),
        "planned_chunks": counts.get("planned_chunks"),
        "processed_chunks": counts.get("processed_chunks"),
        "draft_observations": counts.get("draft_observation_count"),
        "accepted_draft_observation_total": (
            stage.get("accepted_draft_observation_total")
            if type(stage.get("accepted_draft_observation_total")) is int
            and 0 <= stage["accepted_draft_observation_total"] <= 100_000
            else None
        ),
        "accepted_draft_observation_refs_complete": accepted_draft_refs is not None,
        "targeted_record_rejection_events": counts.get("targeted_record_rejected_count"),
        "accepted_signal_histogram": signal_histogram,
        "candidate_eligibility": candidate_eligibility,
        "evidence_mismatch_counts": mismatch_counts,
        "evidence_mismatch_chunks": mismatch_chunks,
        "evidence_mismatch_chunks_omitted_count": mismatch_omitted,
        "core_label_scope_counts": core_scope_counts,
        "accepted_model_core_without_literal_label_count": accepted_unlabeled_core,
        "stage_usage": {key: usage.get(key) for key in ("attempted_calls", "input_tokens", "completion_tokens", "charged_tokens")},
        "reason_counts": safe_reasons,
        "unreported_reason_entries": len(reasons) - len(safe_reasons),
        "token_admission_events": _safe_token_admission_events(stage.get("token_admission_events")),
        "case_trace": safe_trace,
        "accepted_draft_observation_refs": accepted_draft_refs,
        "visible_issue_cases": safe_issues,
    }
    provenance = _safe_character_runtime_provenance(
        diagnostics.get("runtime_provenance")
    )
    if (
        provenance is not None
        and provenance["character_consistency_limits"].get(
            _CHARACTER_SIGNAL_SUPPORT_TRACE_KEY
        ) is True
    ):
        projected = _safe_support_trace_chunks(stage)
        summary["support_trace_chunks"] = projected[0] if projected else None
        summary["support_trace_chunks_omitted_count"] = (
            projected[1] if projected else None
        )
    return summary


def _list_pending(client: httpx.Client, project_id: str) -> list[dict[str, Any]]:
    characters: list[str] = []
    page = 1
    while True:
        roster = _request(
            client, "GET", f"/api/v1/projects/{project_id}/characters", "characters",
            params={"page": page, "page_size": 100},
        )
        if not isinstance(roster, dict) or not isinstance(roster.get("items"), list):
            raise SafeFailure("character_roster_contract", "candidate_review")
        for row in roster["items"]:
            if not isinstance(row, dict) or not isinstance(row.get("character_key"), str):
                raise SafeFailure("character_roster_contract", "candidate_review")
            characters.append(row["character_key"])
        if not roster.get("has_more"):
            break
        page += 1
        if page > 20:
            raise SafeFailure("character_roster_limit", "candidate_review")
    candidates: list[dict[str, Any]] = []
    for character in characters:
        offset = 0
        while True:
            result = _request(
                client, "GET",
                f"/api/v1/projects/{project_id}/characters/{quote(character, safe='')}/profile-candidates",
                "candidate_list", params={"state": "pending", "limit": 200, "offset": offset},
            )
            if not isinstance(result, dict) or not isinstance(result.get("items"), list):
                raise SafeFailure("candidate_list_contract", "candidate_review")
            if any(not isinstance(row, dict) for row in result["items"]):
                raise SafeFailure("candidate_list_contract", "candidate_review")
            candidates.extend(result["items"])
            if not result.get("has_more"):
                break
            offset += len(result["items"])
            if offset > 5000 or not result["items"]:
                raise SafeFailure("candidate_list_limit", "candidate_review")
    return candidates


def _reference_has_line(evidence: object, *, document: str, line: int, source_quote: str | None = None) -> bool:
    if not isinstance(evidence, list):
        return False
    for row in evidence:
        if not isinstance(row, dict):
            continue
        start = row.get("line_start")
        end = row.get("line_end")
        if (
            row.get("document_name") == document
            and type(start) is int and type(end) is int
            and start <= line <= end
            and (source_quote is None or (
                isinstance(row.get("text"), str) and source_quote in row["text"]
            ))
        ):
            return True
    return False


def _normalized_clause(value: object) -> str:
    """Normalize presentation only; preserve words and negation direction."""
    if not isinstance(value, str):
        return ""
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", value).casefold()
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "Z"))
    )


def _candidate_matches(row: dict[str, Any], selector: dict[str, Any]) -> bool:
    source_quote = _normalized_clause(selector.get("source_quote"))
    # The API exposes the extracted model statement as `value`. Evidence is
    # the whole source line, which can contain several unrelated clauses.
    statement = _normalized_clause(row.get("value"))
    trait_key = row.get("trait_key")
    if (
        not source_quote
        or source_quote not in statement
        or not isinstance(trait_key, str)
        or not trait_key.strip()
        or row.get("reviewable") is not True
        or row.get("character_key") != selector["character_key"]
        or row.get("trait_type") != selector["trait_type"]
        or row.get("polarity") != selector["polarity"]
        or row.get("stability") != selector["stability"]
        or not _reference_has_line(
            row.get("evidence"), document=selector["source_document"],
            line=selector["source_line"], source_quote=selector["source_quote"],
        )
    ):
        return False
    if selector["source_document"] == "03-published-history-v1.0.md":
        if row.get("origin") != "history_inference":
            return False
    elif row.get("origin") != "explicit_setting":
        return False
    key_object = selector["key_object"]
    # The key is checked only for internal consistency. It is model-authored
    # for objectless traits and must never become a free-form gold selector.
    # For object-bearing traits, the frozen key retains the exact object fence.
    expected_key = stable_trait_identity(
        selector["trait_type"], trait_key, key_object or ""
    )
    return row.get("comparison_key") == expected_key


def _verified_target_span(
    row: dict[str, Any],
) -> tuple[str, tuple[str, int, int] | None]:
    """Read only the API-verified target, never a statement or context span."""
    if "support_bindings_status" not in row or "support_bindings_v1" not in row:
        return "missing", None
    status = row["support_bindings_status"]
    payload = row["support_bindings_v1"]
    if status == "legacy" and payload is None:
        return "legacy", None
    if status == "invalid":
        return "invalid", None
    if status != "verified":
        return "invalid", None
    if payload is None:
        return "invalid", None
    if (
        type(payload) is not dict
        or set(payload) != {"schema_version", "index_version", "bindings"}
        or payload.get("schema_version") != "character-support-bindings-v1"
        or payload.get("index_version") != "assertion-index-v1"
        or type(payload.get("bindings")) is not list
        or len(payload["bindings"]) != 1
    ):
        return "invalid", None
    binding = payload["bindings"][0]
    if type(binding) is not dict or set(binding) != {
        "evidence_index", "support_id", "target", "actor_anchor_id",
        "label_anchor_id", "scope_relation", "context",
    }:
        return "invalid", None
    target = binding.get("target")
    if (
        type(binding.get("evidence_index")) is not int
        or binding["evidence_index"] != 0
        or type(binding.get("support_id")) is not str
        or SUPPORT_ID.fullmatch(binding["support_id"]) is None
        or type(target) is not dict
        or set(target) != {"support_id", "start_offset", "end_offset", "role"}
        or target.get("support_id") != binding["support_id"]
        or target.get("role") != "target"
        or type(target.get("start_offset")) is not int
        or type(target.get("end_offset")) is not int
        or target["start_offset"] < 0
        or target["end_offset"] <= target["start_offset"]
    ):
        return "invalid", None
    return "verified", (
        binding["support_id"], target["start_offset"], target["end_offset"]
    )


def _support_location_diagnostic(
    pending: list[dict[str, Any]], project_id: str, suite: VerifiedSuite,
    *, source_run_id: str, document_ids: dict[str, str],
    statement_matches: list[list[str]],
) -> dict[str, Any]:
    """Count source-bound target locations; this never selects a candidate."""
    status_counts = {key: 0 for key in ("verified", "legacy", "invalid", "missing")}
    targets: dict[str, tuple[str, int, int]] = {}
    for row in pending:
        status, target = _verified_target_span(row)
        status_counts[status] += 1
        if target is not None:
            targets[row["id"]] = target

    matches_by_selector: list[list[str]] = []
    for selector in suite.plan["candidate_decisions"]:
        name = selector["source_document"]
        line = selector["source_line"]
        expected_lines = _source_lines(suite.files[name])
        expected_line = expected_lines[line - 1]
        normalized_quote = _normalized_clause(selector["source_quote"])
        expected_document_id = document_ids.get(name)
        matched: list[str] = []
        for row in pending:
            target = targets.get(row["id"])
            trait_key = row.get("trait_key")
            if (
                target is None
                or row.get("reviewable") is not True
                or row.get("project_id") != project_id
                or row.get("source_run_id") != source_run_id
                or row.get("character_key") != selector["character_key"]
                or row.get("trait_type") != selector["trait_type"]
                or row.get("polarity") != selector["polarity"]
                or row.get("stability") != selector["stability"]
                or row.get("origin") != "explicit_setting"
                or name == "03-published-history-v1.0.md"
                or not isinstance(trait_key, str)
                or not trait_key.strip()
                or row.get("comparison_key") != stable_trait_identity(
                    selector["trait_type"], trait_key, selector["key_object"] or ""
                )
                or not isinstance(expected_document_id, str)
                or not expected_document_id
                or not normalized_quote
            ):
                continue
            evidence = row.get("evidence")
            if type(evidence) is not list or len(evidence) != 1 or type(evidence[0]) is not dict:
                continue
            ref = evidence[0]
            support_id, start, end = target
            if (
                ref.get("document_id") != expected_document_id
                or ref.get("document_name") != name
                or type(ref.get("document_version")) is not int
                or ref["document_version"] != 1
                or ref.get("content_sha256") != suite.hashes[name]
                or ref.get("line_start") != line
                or ref.get("line_end") != line
                or ref.get("text") != expected_line
                or not isinstance(ref.get("input_id"), str)
                or not ref["input_id"]
                or not support_id.startswith(f"L{line}:A")
                or end > len(expected_line)
            ):
                continue
            if normalized_quote in _normalized_clause(expected_line[start:end]):
                matched.append(row["id"])
        matches_by_selector.append(matched)

    frequency: dict[str, int] = {}
    for matches in matches_by_selector:
        for candidate_id in matches:
            frequency[candidate_id] = frequency.get(candidate_id, 0) + 1
    selector_counts = {
        f"selector_{ordinal}": {
            "match_count": len(matches),
            "unique": len(matches) == 1 and frequency[matches[0]] == 1,
        }
        for ordinal, matches in enumerate(matches_by_selector, start=1)
    }
    return {
        "schema_version": SUPPORT_LOCATION_DIAGNOSTIC_V1,
        "binding_status_counts": status_counts,
        "statement_zero_target_one_count": sum(
            len(old_matches) == 0 and len(target_matches) == 1
            for old_matches, target_matches in zip(statement_matches, matches_by_selector)
        ),
        "selectors": selector_counts,
    }


def _support_location_unavailable(reason_code: str) -> dict[str, Any]:
    """Report only a fixed diagnostic status, never an exception or source text."""
    if reason_code not in {"snapshot_unavailable", "diagnostic_unavailable"}:
        reason_code = "diagnostic_unavailable"
    return {
        "schema_version": SUPPORT_LOCATION_DIAGNOSTIC_V1,
        "available": False,
        "reason_code": reason_code,
    }


def _safe_support_location_diagnostic(
    pending: list[dict[str, Any]], project_id: str, suite: VerifiedSuite,
    *, source_run_id: str, document_ids: dict[str, str],
    statement_matches: list[list[str]],
) -> dict[str, Any]:
    try:
        return _support_location_diagnostic(
            pending, project_id, suite, source_run_id=source_run_id,
            document_ids=document_ids, statement_matches=statement_matches,
        )
    except Exception:
        return _support_location_unavailable("diagnostic_unavailable")


def _partial_baseline_inventory(
    client: httpx.Client, project_id: str, suite: VerifiedSuite,
    *, source_run_id: str | None = None, document_ids: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Read-only, redacted inventory after a partial baseline fails admission."""
    pending = _list_pending(client, project_id)
    candidate_ids: set[str] = set()
    required_text = (
        "character_key", "trait_type", "trait_key", "value", "polarity",
        "stability", "origin",
    )
    for row in pending:
        candidate_id = row.get("id")
        evidence = row.get("evidence")
        if (
            not isinstance(candidate_id, str) or not candidate_id
            or candidate_id in candidate_ids
            or type(row.get("reviewable")) is not bool
            or any(not isinstance(row.get(key), str) for key in required_text)
            or not (
                row.get("comparison_key") is None
                or isinstance(row.get("comparison_key"), str)
            )
            or not isinstance(evidence, list)
            or any(
                not isinstance(ref, dict)
                or not isinstance(ref.get("document_name"), str)
                or type(ref.get("line_start")) is not int
                or type(ref.get("line_end")) is not int
                or not isinstance(ref.get("text"), str)
                for ref in evidence
            )
        ):
            raise SafeFailure("candidate_inventory_contract", "candidate_inventory")
        candidate_ids.add(candidate_id)

    reviewable = [row for row in pending if row["reviewable"]]
    matched_ids: set[str] = set()
    matches_by_selector = [
        [row["id"] for row in reviewable if _candidate_matches(row, selector)]
        for selector in suite.plan["candidate_decisions"]
    ]
    match_frequency: dict[str, int] = {}
    for match_ids in matches_by_selector:
        matched_ids.update(match_ids)
        for candidate_id in match_ids:
            match_frequency[candidate_id] = match_frequency.get(candidate_id, 0) + 1
    selector_counts: dict[str, dict[str, int | bool]] = {}
    for ordinal, match_ids in enumerate(matches_by_selector, start=1):
        unique = len(match_ids) == 1 and match_frequency[match_ids[0]] == 1
        selector_counts[f"selector_{ordinal}"] = {
            "match_count": len(match_ids), "unique": unique,
        }
    inventory = {
        "available": True,
        "reviewable_total": len(reviewable),
        "matched_reviewable_total": len(matched_ids),
        "extra_reviewable_total": len(reviewable) - len(matched_ids),
        "selectors": selector_counts,
    }
    if suite.name == "dev" and source_run_id is not None and document_ids is not None:
        inventory["support_location_diagnostic_v1"] = _safe_support_location_diagnostic(
            pending, project_id, suite, source_run_id=source_run_id,
            document_ids=document_ids, statement_matches=matches_by_selector,
        )
    return inventory


def _review_candidates(
    client: httpx.Client, project_id: str, suite: VerifiedSuite,
    state: dict[str, Any], *, source_run_id: str | None = None,
    document_ids: dict[str, str] | None = None,
) -> None:
    plan = suite.plan
    all_pending = _list_pending(client, project_id)
    reviewable = [row for row in all_pending if row.get("reviewable") is True]
    selectors = plan["candidate_decisions"]
    chosen: dict[str, dict[str, Any]] = {}
    reused: set[str] = set()
    selection_checks: dict[str, dict[str, Any]] = {}
    statement_matches: list[list[str]] = []
    for selector in selectors:
        matches = [row for row in reviewable if _candidate_matches(row, selector)]
        statement_matches.append([
            row["id"] for row in matches if isinstance(row.get("id"), str)
        ])
        key = selector["candidate_key"]
        selection_checks[key] = {"match_count": len(matches), "unique": len(matches) == 1}
        if len(matches) == 1:
            candidate_id = matches[0].get("id")
            if not isinstance(candidate_id, str) or candidate_id in reused:
                selection_checks[key]["unique"] = False
            else:
                reused.add(candidate_id)
                chosen[key] = matches[0]
    state["candidate_review"] = {
        "expected": len(selectors),
        "reviewable": len(reviewable),
        "unique_matches": sum(row["unique"] for row in selection_checks.values()),
        "unselected_reviewable": len(reviewable) - len(reused),
        "selectors": selection_checks,
    }
    if suite.name == "dev" and source_run_id is not None and document_ids is not None:
        state["support_location_diagnostic_v1"] = _safe_support_location_diagnostic(
            all_pending, project_id, suite, source_run_id=source_run_id,
            document_ids=document_ids, statement_matches=statement_matches,
        )
    if len(chosen) != len(selectors) or any(not row["unique"] for row in selection_checks.values()):
        raise SafeFailure("candidate_not_unique", "candidate_review")

    axes: dict[str, dict[str, Any]] = {}
    for spec in plan["approved_axes"]:
        result = _request(
            client, "POST", f"/api/v1/projects/{project_id}/character-trait-axes", "axis_create",
            json={
                "trait_type": "core_personality",
                "display_name": spec["display_name"],
                "definition": spec["definition"],
            },
        )
        if (
            not isinstance(result, dict)
            or not isinstance(result.get("id"), str)
            or result.get("version") != 1
            or result.get("definition_sha256") != _sha256(" ".join(spec["definition"].split()))
        ):
            raise SafeFailure("axis_create_contract", "axis_create")
        axes[spec["axis_key"]] = result

    selected: dict[str, dict[str, Any]] = {}
    for selector in selectors:
        candidate = chosen[selector["candidate_key"]]
        character_key = quote(selector["character_key"], safe="")
        candidate_id = candidate["id"]
        body: dict[str, Any] = {
            "decision": "confirm", "expected_revision": candidate.get("revision"),
            "comment": "冻结作者审核计划：唯一来源锚点与角色、维度、方向、稳定性一致",
        }
        axis_key = selector["approved_axis_key"]
        if axis_key is not None:
            body["approved_axis_id"] = axes[axis_key]["id"]
            body["expected_axis_version"] = 1
        result = _request(
            client, "POST",
            f"/api/v1/projects/{project_id}/characters/{character_key}/profile-candidates/{quote(candidate_id, safe='')}/decisions",
            "candidate_decision",
            headers={"Idempotency-Key": f"axis-{uuid4().hex}"}, json=body,
        )
        confirmed = result.get("candidate") if isinstance(result, dict) else None
        if (
            not isinstance(confirmed, dict)
            or confirmed.get("review_state") != "confirmed"
            or confirmed.get("id") != candidate_id
            or confirmed.get("approved_axis_id") != (axes[axis_key]["id"] if axis_key else None)
        ):
            raise SafeFailure("candidate_decision_contract", "candidate_review")
        selected[selector["candidate_key"]] = confirmed
    state["selected"] = selected
    state["axis_count"] = len(axes)


def _execute_trial(client: httpx.Client, suite: VerifiedSuite, trial: int, state: dict[str, Any], *, timeout_seconds: float) -> None:
    project = _request(
        client, "POST", "/api/v1/projects", "project_create",
        json={
            "name": f"axis-{suite.name}-T{trial}-{uuid4().hex[:8]}",
            "description": "原创、开发者可见角色轴验收；非盲测",
        },
    )
    if not isinstance(project, dict) or not isinstance(project.get("id"), str):
        raise SafeFailure("project_create_contract", "project_create")
    project_id = project["id"]
    state["project_id_sha256"] = _sha256(project_id)
    baseline_document_ids = {
        name: _upload(client, project_id, suite, name, role, published=True)
        for name, role in BASELINE_FILES
    }
    baseline_id = _start_run(client, project_id, mode="baseline_build")
    baseline_run = _wait_run(client, baseline_id, timeout_seconds=timeout_seconds)
    known_docs = {name for name, _ in BASELINE_FILES} | {DRAFT_FILE}
    baseline = _run_summary(client, baseline_run, known_documents=known_docs)
    state["baseline"] = baseline
    state["baseline_admission"] = _baseline_admission(baseline)
    if suite.name == "dev":
        state["support_location_diagnostic_v1"] = _support_location_unavailable(
            "snapshot_unavailable"
        )
    if state["baseline_admission"]["admitted"] is not True:
        if baseline.get("status") == "completed" and baseline.get("stage_outcome") == "partial":
            try:
                state["partial_baseline_inventory"] = _partial_baseline_inventory(
                    client, project_id, suite, source_run_id=baseline_id,
                    document_ids=baseline_document_ids,
                )
                if suite.name == "dev":
                    state["support_location_diagnostic_v1"] = state[
                        "partial_baseline_inventory"
                    ].get(
                        "support_location_diagnostic_v1",
                        _support_location_unavailable("diagnostic_unavailable"),
                    )
            except Exception as exc:
                state["partial_baseline_inventory"] = {
                    "available": False, "reason_code": _failure(exc)["code"],
                }
        raise SafeFailure("baseline_admission_failed", "baseline")
    _review_candidates(
        client, project_id, suite, state, source_run_id=baseline_id,
        document_ids=baseline_document_ids,
    )
    draft_document_id = _upload(
        client, project_id, suite, DRAFT_FILE, "chapter", published=False
    )
    draft_id = _start_run(client, project_id, mode="draft_review", target_id=draft_document_id)
    draft_run = _wait_run(client, draft_id, timeout_seconds=timeout_seconds)
    state["draft"] = _run_summary(client, draft_run, known_documents=known_docs)


def _load_oracle(suite: VerifiedSuite) -> list[dict[str, Any]]:
    oracle = _json_object(suite.files["oracle.json"], stage="oracle_scoring")
    cases = oracle.get("expected_cases")
    if (
        oracle.get("schema_version") != "character-axis-oracle-v1"
        or oracle.get("suite") != suite.name
        or oracle.get("world_id") != suite.world_id
        or oracle.get("developer_visible") is not True
        or not isinstance(cases, list)
        or len(cases) != suite.case_count
    ):
        raise SafeFailure("oracle_contract", "oracle_scoring")
    candidate_keys = {row["candidate_key"] for row in suite.plan["candidate_decisions"]}
    case_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise SafeFailure("oracle_case_contract", "oracle_scoring")
        evidence = case.get("evidence")
        roles = case.get("required_citation_roles")
        outcomes = case.get("allowed_final_outcomes")
        if (
            not _safe_key(case.get("case_id")) or case["case_id"] in case_ids
            or case.get("candidate_key") not in candidate_keys
            or case.get("gold_class") not in {"conflict", "explained", "hard_negative", "abstain"}
            or not isinstance(outcomes, list) or not outcomes
            or any(value not in {"conflict", "no_issue", "needs_confirmation", "unverifiable"} for value in outcomes)
            or not isinstance(roles, list)
            or any(value not in {"B", "C", "G", "X"} for value in roles)
            or type(case.get("min_independent_observations")) is not int
            or not 0 <= case["min_independent_observations"] <= 24
            or not isinstance(evidence, dict)
            or set(evidence) != {"B", "C", "G", "X", "forbidden"}
        ):
            raise SafeFailure("oracle_case_contract", "oracle_scoring")
        for role in ("B", "C", "G", "X", "forbidden"):
            refs = evidence[role]
            if not isinstance(refs, list):
                raise SafeFailure("oracle_evidence_contract", "oracle_scoring")
            for ref in refs:
                if not isinstance(ref, dict) or ref.get("document_name") not in suite.files or type(ref.get("line")) is not int:
                    raise SafeFailure("oracle_evidence_contract", "oracle_scoring")
                source_lines = _source_lines(suite.files[ref["document_name"]])
                if not 1 <= ref["line"] <= len(source_lines):
                    raise SafeFailure("oracle_evidence_contract", "oracle_scoring")
                if role == "forbidden" and (
                    not isinstance(ref.get("character_key"), str)
                    or ref.get("role") not in {"C", "G", "X"}
                    or ref.get("polarity") not in {"positive", "negative", "neutral", None}
                ):
                    raise SafeFailure("oracle_evidence_contract", "oracle_scoring")
        case_ids.add(case["case_id"])
    return cases


def _ref_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    start = actual.get("line_start")
    end = actual.get("line_end")
    return (
        actual.get("document_name") == expected["document_name"]
        and type(start) is int and type(end) is int
        and start == expected["line"] == end
    )


def _evidence_scores(
    case: dict[str, Any], candidate: dict[str, Any], trace: dict[str, Any],
) -> dict[str, str]:
    expected = case["evidence"]
    baseline_refs = candidate.get("evidence")
    baseline_refs = baseline_refs if isinstance(baseline_refs, list) else []
    observation_refs = trace.get("matched_observation_refs")
    observation_refs = observation_refs if isinstance(observation_refs, list) else []
    citation_refs = trace.get("citation_refs")
    scores: dict[str, str] = {}
    for role, actual in (("B", baseline_refs), ("C", observation_refs)):
        required = expected[role]
        source_covered = (role != "C" or not actual) if not required else all(
            any(_ref_matches(ref, anchor) for ref in actual)
            for anchor in required
        )
        if role in case["required_citation_roles"] and citation_refs is None:
            scores[role] = "unavailable"
            continue
        cited = (
            [ref for ref in citation_refs if ref["role"] == role]
            if citation_refs is not None else []
        )
        citation_covered = (
            role not in case["required_citation_roles"]
            or bool(cited)
            and all(any(_ref_matches(ref, anchor) for ref in cited) for anchor in required)
            and all(any(_ref_matches(ref, anchor) for anchor in required) for ref in cited)
        )
        scores[role] = "matched" if source_covered and citation_covered else "missed"
    for role in ("G", "X"):
        required = expected[role]
        if not required:
            scores[role] = "not_required"
            continue
        if citation_refs is None:
            scores[role] = "unavailable"
            continue
        role_refs = [ref for ref in citation_refs if ref["role"] == role]
        scores[role] = "matched" if (
            role_refs
            and all(any(_ref_matches(ref, anchor) for ref in role_refs) for anchor in required)
            and all(any(_ref_matches(ref, anchor) for anchor in required) for ref in role_refs)
        ) else "missed"
    return scores


def _visible_issue_evidence_score(
    case: dict[str, Any], visible: list[dict[str, Any]], trace: dict[str, Any],
) -> str:
    """Score the evidence actually delivered in the visible HTTP issue."""
    if not visible:
        return "missed" if case["gold_class"] == "conflict" else "not_required"
    if len(visible) != 1:
        return "missed"
    required = case["evidence"]["B"] + case["evidence"]["C"]
    # G/X may accompany a visible issue only when separately registered in
    # the frozen oracle. Their presence is allowed context, not proof that
    # the reviewer used them as an explanation; citation scoring is separate.
    registered = required + case["evidence"]["G"] + case["evidence"]["X"]
    if not case["evidence"]["C"]:
        # A visible issue without pre-registered current evidence cannot be
        # endorsed merely because its internal trace has an expected label.
        return "unavailable"
    actual = visible[0].get("evidence_refs")
    if not isinstance(actual, list) or not actual:
        return "unavailable"
    if not all(any(_ref_matches(ref, anchor) for ref in actual) for anchor in required):
        return "missed"
    if not all(any(_ref_matches(ref, anchor) for anchor in registered) for ref in actual):
        return "missed"
    citations = trace.get("citation_refs")
    if isinstance(citations, list) and any(
        citation["line_start"] != citation["line_end"] or not any(_ref_matches(ref, {
            "document_name": citation["document_name"],
            "line": citation["line_start"],
        }) for ref in actual)
        for citation in citations
        if citation["role"] in {"B", "C", "G", "X"}
    ):
        return "missed"
    return "matched"


def _forbidden_hits(
    case: dict[str, Any], trace: dict[str, Any], draft: dict[str, Any],
    *, target_dimension: str,
) -> tuple[int, bool]:
    hits = 0
    accepted = draft.get("accepted_draft_observation_refs")
    actor_attribution_unavailable = (
        any(row["role"] == "C" for row in case["evidence"]["forbidden"])
        and not isinstance(accepted, list)
    )
    citations = trace.get("citation_refs")
    citations = citations if isinstance(citations, list) else []
    for forbidden in case["evidence"]["forbidden"]:
        if forbidden["role"] == "C":
            actual = accepted if isinstance(accepted, list) else []
            actor_matches = lambda ref: (
                ref.get("actor_sha256") == _actor_digest(forbidden["character_key"])
                and ref.get("dimension") == target_dimension
            )
        else:
            actual = [ref for ref in citations if ref["role"] == forbidden["role"]]
            actor_matches = lambda _ref: trace.get("character_key") == forbidden["character_key"]
        if any(
            actor_matches(ref) and _ref_matches(ref, forbidden)
            and (forbidden.get("polarity") is None or ref.get("polarity") == forbidden["polarity"])
            for ref in actual
        ):
            hits += 1
    return hits, actor_attribution_unavailable


def _score_case(case: dict[str, Any], state: dict[str, Any], suite: VerifiedSuite) -> dict[str, Any]:
    draft = state.get("draft") or {}
    selected = state.get("selected") or {}
    candidate = selected.get(case["candidate_key"])
    digest = _candidate_id_sha256(candidate.get("id")) if isinstance(candidate, dict) else None
    traces = [
        row for row in draft.get("case_trace", [])
        if row.get("confirmed_candidate_id_sha256") == digest
    ] if digest else []
    trace = traces[0] if len(traces) == 1 else None
    visible = [
        row for row in draft.get("visible_issue_cases", [])
        if row.get("confirmed_candidate_id_sha256") == digest
    ] if digest else []
    selector = next(
        row for row in suite.plan["candidate_decisions"]
        if row["candidate_key"] == case["candidate_key"]
    )
    forbidden_hits, actor_unavailable = _forbidden_hits(
        case, trace or {}, draft, target_dimension=selector["trait_type"]
    )
    visible_false_positive_count = (
        sum(row.get("judgement") == "contradicts" for row in visible)
        if case["gold_class"] != "conflict" else 0
    )
    if trace is None or candidate is None:
        evaluation_state = (
            "pipeline_not_reached" if state.get("draft") is None
            else "case_result_unavailable"
        )
        return {
            "case_id": case["case_id"], "gold_class": case["gold_class"],
            "evaluation_state": evaluation_state,
            "candidate_confirmed": candidate is not None,
            "trace_unique": len(traces) == 1,
            "actual_outcome": None,
            "evidence": {role: "unavailable" for role in ("B", "C", "G", "X")},
            "visible_issue_evidence": "unavailable",
            "false_positive": visible_false_positive_count > 0,
            "visible_false_positive_count": visible_false_positive_count,
            "trace_false_positive": None,
            "false_positive_unknown": case["gold_class"] != "conflict" and not visible_false_positive_count,
            "false_negative": None,
            "target_dimension_suspect_attribution_hits": forbidden_hits,
            "target_dimension_attribution_unavailable": actor_unavailable,
            "actor_recall_assessable": bool(case["evidence"]["C"]),
            "abstained": False, "passed": False,
        }
    observation_refs = trace.get("matched_observation_refs")
    observation_refs = observation_refs if isinstance(observation_refs, list) else []
    independent = {
        (ref.get("document_name"), ref.get("line_start"), ref.get("line_end"))
        for ref in observation_refs
    }
    evidence = _evidence_scores(case, candidate, trace)
    outcome = trace.get("final_outcome")
    gold = case["gold_class"]
    axis_key = selector["approved_axis_key"]
    if axis_key is None:
        axis_applicability_valid = candidate.get("approved_axis_id") is None
        baseline_axis_snapshot_matched = None
    else:
        axis = next(
            row for row in suite.plan["approved_axes"] if row["axis_key"] == axis_key
        )
        baseline_axis_snapshot_matched = (
            isinstance(candidate.get("approved_axis_id"), str)
            and trace.get("comparison_key_sha256")
            == _sha256(f"approved_axis:{candidate['approved_axis_id']}:1")
            and trace.get("approved_axis_definition_sha256")
            == _sha256(" ".join(axis["definition"].split()))
            and trace.get("observation_axis_binding") == "server_targeted_evidence"
        )
        axis_applicability_valid = baseline_axis_snapshot_matched
    role_check = set(case["required_citation_roles"]) <= set(trace.get("citation_roles") or [])
    outcome_check = outcome in case["allowed_final_outcomes"]
    review_check = (
        trace.get("review_verdict") == "contradicts" if gold == "conflict"
        else trace.get("review_verdict") == "explained" if gold == "explained"
        else outcome != "conflict"
    )
    raw_review_verdict = trace.get("review_verdict")
    public_review_verdict = _safe_public_enum(
        raw_review_verdict, SAFE_PUBLIC_REVIEW_VERDICTS
    )
    review_verdict_contract = (
        raw_review_verdict is None or public_review_verdict is not None
    )
    if gold == "conflict":
        visible_check = len(visible) == 1 and visible[0].get("judgement") == "contradicts"
    elif outcome in {"no_issue", "unverifiable"}:
        visible_check = not visible
    else:
        visible_check = len(visible) <= 1 and not any(
            row.get("judgement") == "contradicts" for row in visible
        )
    evidence_check = all(value in {"matched", "not_required"} for value in evidence.values())
    visible_issue_evidence = _visible_issue_evidence_score(case, visible, trace)
    coverage_check = not trace.get("matched_observation_refs_truncated")
    axis_observation_binding_evidenced = (
        baseline_axis_snapshot_matched is True
        and bool(independent)
        and evidence["C"] == "matched"
        if axis_key is not None and case["min_independent_observations"] > 0
        else None
    )
    passed = all((
        outcome_check, review_check, review_verdict_contract,
        visible_check, role_check, evidence_check,
        visible_issue_evidence in {"matched", "not_required"},
        len(independent) >= case["min_independent_observations"],
        forbidden_hits == 0, not actor_unavailable, coverage_check,
        axis_applicability_valid,
    ))
    return {
        "case_id": case["case_id"], "gold_class": gold,
        "evaluation_state": "case_result_available",
        "candidate_confirmed": True, "trace_unique": True,
        "actual_outcome": _safe_public_enum(outcome, SAFE_PUBLIC_CASE_OUTCOMES),
        "review_verdict": public_review_verdict,
        "matched_observations": len(independent),
        "baseline_axis_snapshot_matched": baseline_axis_snapshot_matched,
        "axis_applicability_valid": axis_applicability_valid,
        "axis_observation_binding_evidenced": axis_observation_binding_evidenced,
        "citation_roles_matched": role_check,
        "evidence": evidence,
        "visible_issue_evidence": visible_issue_evidence,
        "target_dimension_suspect_attribution_hits": forbidden_hits,
        "target_dimension_attribution_unavailable": actor_unavailable,
        "actor_recall_assessable": bool(case["evidence"]["C"]),
        "false_positive": gold != "conflict" and (
            outcome == "conflict" or visible_false_positive_count > 0
        ),
        "visible_false_positive_count": visible_false_positive_count,
        "trace_false_positive": gold != "conflict" and outcome == "conflict",
        "false_positive_unknown": False,
        "false_negative": gold == "conflict" and outcome != "conflict",
        "abstained": outcome in {"needs_confirmation", "unverifiable"},
        "passed": passed,
    }


def _safe_public_enum(value: object, allowed: frozenset[str] | set[str]) -> str | None:
    return value if type(value) is str and value in allowed else None


def _safe_public_counter(value: object, *, maximum: int) -> int | None:
    return value if type(value) is int and 0 <= value <= maximum else None


def _safe_public_stage_usage(value: object) -> dict[str, int] | None:
    if not isinstance(value, dict) or set(value) != set(PUBLIC_STAGE_USAGE_LIMITS):
        return None
    safe = {
        key: _safe_public_counter(value[key], maximum=maximum)
        for key, maximum in PUBLIC_STAGE_USAGE_LIMITS.items()
    }
    return safe if all(number is not None for number in safe.values()) else None


def _safe_public_elapsed(value: object) -> float | int | None:
    if type(value) is int:
        return value if 0 <= value <= 1_000_000 else None
    if type(value) is float and math.isfinite(value) and 0 <= value <= 1_000_000:
        return value
    return None


def _public_run(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if summary is None:
        return None
    public = {
        key: summary.get(key)
        for key in (
            "status", "runtime_provenance_sha256", "elapsed_seconds", "prompt_tokens", "completion_tokens",
            "stage_outcome", "stage_reason", "material_coverage", "planned_chunks",
            "processed_chunks", "draft_observations", "accepted_draft_observation_total",
            "accepted_draft_observation_refs_complete", "targeted_record_rejection_events",
            "accepted_signal_histogram", "candidate_eligibility",
            "evidence_mismatch_counts", "evidence_mismatch_chunks",
            "evidence_mismatch_chunks_omitted_count",
            "core_label_scope_counts",
            "accepted_model_core_without_literal_label_count",
            "stage_usage", "reason_counts", "unreported_reason_entries",
            "token_admission_events",
        )
    }
    # The summary is assembled from HTTP responses and persisted diagnostics.
    # Even fields normally populated by server enums or integers must not
    # become a prose, URL, or credential channel in the published report.
    public["status"] = _safe_public_enum(summary.get("status"), TERMINAL)
    public["stage_outcome"] = _safe_public_enum(
        summary.get("stage_outcome"), SAFE_PUBLIC_STAGE_OUTCOMES
    )
    public["stage_reason"] = _safe_public_enum(
        summary.get("stage_reason"), SAFE_PUBLIC_STAGE_REASONS
    )
    public["material_coverage"] = _safe_public_enum(
        summary.get("material_coverage"), SAFE_PUBLIC_MATERIAL_COVERAGE
    )
    public["elapsed_seconds"] = _safe_public_elapsed(
        summary.get("elapsed_seconds")
    )
    for key, maximum in PUBLIC_RUN_COUNTER_LIMITS.items():
        public[key] = _safe_public_counter(summary.get(key), maximum=maximum)
    public["stage_usage"] = _safe_public_stage_usage(summary.get("stage_usage"))
    if "support_trace_chunks" in summary or "support_trace_chunks_omitted_count" in summary:
        projected = _safe_support_trace_chunks(summary)
        public["support_trace_chunks"] = projected[0] if projected else None
        public["support_trace_chunks_omitted_count"] = projected[1] if projected else None
    return public


def _score_trial(
    suite: VerifiedSuite, state: dict[str, Any], cases: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = state.get("baseline")
    draft = state.get("draft")
    scores = [_score_case(case, state, suite) for case in cases]
    raw_review = state.get("candidate_review")
    review = raw_review if isinstance(raw_review, dict) else {}
    admission = state.get("baseline_admission") or {}
    selected = state.get("selected") or {}
    unselected = review.get("unselected_reviewable")
    unselected_count = (
        unselected if type(unselected) is int and 0 <= unselected <= 100_000 else None
    )
    expected_count = len(suite.plan["candidate_decisions"])
    known_case_digests = {
        _candidate_id_sha256(selected[case["candidate_key"]]["id"])
        for case in cases if case["candidate_key"] in selected
    }
    unexpected_conflicts = sum(
        row.get("judgement") == "contradicts"
        and row.get("confirmed_candidate_id_sha256") not in known_case_digests
        for row in (draft or {}).get("visible_issue_cases", [])
    )
    visible_false_positives = sum(
        row.get("visible_false_positive_count", 0) for row in scores
    ) + unexpected_conflicts
    trace_only_false_positives = sum(
        row.get("trace_false_positive") is True
        and row.get("visible_false_positive_count", 0) == 0
        for row in scores
    )
    complete = (
        draft is not None
        and draft.get("status") == "completed"
        and draft.get("stage_outcome") == "completed"
        and draft.get("material_coverage") == "complete"
        and type(draft.get("planned_chunks")) is int
        and draft["planned_chunks"] > 0
        and draft.get("processed_chunks") == draft["planned_chunks"]
        and type((draft.get("stage_usage") or {}).get("attempted_calls")) is int
        and draft["stage_usage"]["attempted_calls"] > 0
    )
    passed = (
        state.get("failure") is None
        and admission.get("admitted") is True
        and type(review.get("expected")) is int
        and review["expected"] == expected_count
        and type(review.get("unique_matches")) is int
        and review["unique_matches"] == expected_count
        and unselected_count == 0
        and len(selected) == expected_count
        and complete
        and all(row["passed"] for row in scores)
        and unexpected_conflicts == 0
    )
    return {
        "trial": state["trial"],
        "project_id_sha256": state.get("project_id_sha256"),
        "failure": state.get("failure"),
        "baseline": _public_run(baseline),
        "baseline_admitted": admission.get("admitted") is True,
        **({"partial_baseline_inventory": state["partial_baseline_inventory"]}
           if "partial_baseline_inventory" in state else {}),
        **({"support_location_diagnostic_v1": state.get(
            "support_location_diagnostic_v1",
            _support_location_unavailable("snapshot_unavailable"),
        )} if suite.name == "dev" and baseline is not None else {}),
        "candidate_review": review,
        "axis_count": state.get("axis_count", 0),
        "draft": _public_run(draft),
        "draft_complete": complete,
        "case_scores": scores,
        "counts": {
            "cases_passed": sum(row["passed"] for row in scores),
            "cases_total": len(scores),
            "unselected_reviewable_candidates": unselected_count,
            "false_positives": visible_false_positives + trace_only_false_positives,
            "visible_false_positives": visible_false_positives,
            "false_positive_unknown_cases": sum(row.get("false_positive_unknown") is True for row in scores),
            "false_negatives": sum(row.get("false_negative") is True for row in scores),
            "unevaluated_cases": sum(
                row.get("evaluation_state") != "case_result_available" for row in scores
            ),
            "abstained": sum(row["abstained"] for row in scores),
            "target_dimension_suspect_attribution_hits": sum(row.get("target_dimension_suspect_attribution_hits", 0) for row in scores),
            "target_dimension_attribution_unavailable_cases": sum(row.get("target_dimension_attribution_unavailable") is True for row in scores),
            "actor_recall_unassessed_cases": sum(row.get("actor_recall_assessable") is False for row in scores),
            "baseline_axis_snapshots_matched": sum(row.get("baseline_axis_snapshot_matched") is True for row in scores),
            "baseline_axis_snapshot_eligible_cases": sum(row.get("baseline_axis_snapshot_matched") is not None for row in scores),
            "axis_observation_bindings_evidenced": sum(row.get("axis_observation_binding_evidenced") is True for row in scores),
            "axis_observation_binding_eligible_cases": sum(row.get("axis_observation_binding_evidenced") is not None for row in scores),
            "unavailable_evidence_roles": sum(
                value == "unavailable" for row in scores for value in row["evidence"].values()
            ),
            "unexpected_conflicts": unexpected_conflicts,
        },
        "passed": passed,
    }


def _failure(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, SafeFailure):
        return exc.payload
    if isinstance(exc, (OSError, httpx.HTTPError)):
        code = "environment_error"
    elif isinstance(exc, (KeyError, TypeError, ValueError)):
        code = "runner_contract_error"
    else:
        code = "runner_error"
    return {"code": code, "stage": "runner", "details": {"exception_type": type(exc).__name__}}


def _bounded_number(value: Any) -> int:
    return value if type(value) is int and 0 <= value <= 100_000_000 else 0


def _runtime_summary(health: dict[str, Any]) -> dict[str, Any]:
    provenance = health.get("runtime_provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    build = provenance.get("build")
    build = build if isinstance(build, dict) else {}
    provider = provenance.get("chat_provider")
    provider = provider if isinstance(provider, dict) else {}
    limits = provenance.get("character_consistency_limits")
    limits = limits if isinstance(limits, dict) else {}
    scope_review_valid = (
        _CHARACTER_SIGNAL_SCOPE_REVIEW_KEY in limits
        and _valid_character_scope_review_limits(
            limits, allow_legacy_prompt_version=True,
        )
        and (
            limits[_CHARACTER_SIGNAL_SCOPE_REVIEW_KEY] is False
            or (
                limits.get(_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY) is True
                and limits.get(_CHARACTER_SIGNAL_SUPPORT_ID_KEY) is True
                and limits.get(_CHARACTER_SIGNAL_SUPPORT_SEGMENTER_KEY)
                == _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_VERSION
            )
        )
    )
    alias = provider.get("model_alias")
    return {
        "schema_version": (
            "loreguard-runtime-provenance-v3"
            if provenance.get("schema_version") == "loreguard-runtime-provenance-v3"
            else None
        ),
        "build_revision": (
            build.get("git_revision")
            if isinstance(build.get("git_revision"), str)
            and GIT_HASH.fullmatch(build["git_revision"])
            else None
        ),
        "service_artifact_sha256": (
            build.get("service_artifact_sha256")
            if isinstance(build.get("service_artifact_sha256"), str)
            and SHA256.fullmatch(build["service_artifact_sha256"])
            else None
        ),
        "requested_model_alias_sha256": _sha256(alias) if isinstance(alias, str) else None,
        "endpoint_configuration_sha256": (
            provider.get("endpoint_configuration_sha256")
            if isinstance(provider.get("endpoint_configuration_sha256"), str)
            and SHA256.fullmatch(provider["endpoint_configuration_sha256"])
            else None
        ),
        "character_limits": {
            key: value for key in (
                "stage_token_budget", "signal_token_budget", "max_chunks_per_run",
                "max_candidates_per_run",
            )
            if type(value := limits.get(key)) is int and value >= 0
        },
        "signal_full_line_echo_v2": (
            limits.get(_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY)
            if type(limits.get(_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY)) is bool
            else None
        ),
        "signal_core_scope_v3": (
            limits.get(_CHARACTER_SIGNAL_CORE_SCOPE_KEY)
            if type(limits.get(_CHARACTER_SIGNAL_CORE_SCOPE_KEY)) is bool
            and type(limits.get(_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY)) is bool
            and (
                limits[_CHARACTER_SIGNAL_CORE_SCOPE_KEY] is False
                or limits[_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY] is True
            )
            else None
        ),
        "signal_support_id_v4": (
            limits.get(_CHARACTER_SIGNAL_SUPPORT_ID_KEY)
            if type(limits.get(_CHARACTER_SIGNAL_SUPPORT_ID_KEY)) is bool
            and type(limits.get(_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY)) is bool
            and (
                limits[_CHARACTER_SIGNAL_SUPPORT_ID_KEY] is False
                or limits[_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY] is True
            )
            and limits.get(_CHARACTER_SIGNAL_SUPPORT_SEGMENTER_KEY) == (
                _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_VERSION
                if limits[_CHARACTER_SIGNAL_SUPPORT_ID_KEY] else None
            )
            else None
        ),
        "signal_support_segmenter_version": (
            _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_VERSION
            if limits.get(_CHARACTER_SIGNAL_SUPPORT_ID_KEY) is True
            and limits.get(_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY) is True
            and limits.get(_CHARACTER_SIGNAL_SUPPORT_SEGMENTER_KEY)
            == _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_VERSION
            else None
        ),
        "signal_semantic_scope_v5": (
            limits.get(_CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY)
            if type(limits.get(_CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY)) is bool
            and (
                limits[_CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY] is False
                or (
                    limits.get(_CHARACTER_SIGNAL_SUPPORT_ID_KEY) is True
                    and limits.get(_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY) is True
                    and limits.get(_CHARACTER_SIGNAL_SUPPORT_SEGMENTER_KEY)
                    == _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_VERSION
                )
            )
            and limits.get(_CHARACTER_SIGNAL_SEMANTIC_SCOPE_VERSION_KEY) == (
                _CHARACTER_SIGNAL_SEMANTIC_SCOPE_VERSION
                if limits[_CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY] else None
            )
            else None
        ),
        "signal_semantic_scope_version": (
            _CHARACTER_SIGNAL_SEMANTIC_SCOPE_VERSION
            if limits.get(_CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY) is True
            and limits.get(_CHARACTER_SIGNAL_SUPPORT_ID_KEY) is True
            and limits.get(_CHARACTER_SIGNAL_FULL_LINE_ECHO_KEY) is True
            and limits.get(_CHARACTER_SIGNAL_SUPPORT_SEGMENTER_KEY)
            == _CHARACTER_SIGNAL_SUPPORT_SEGMENTER_VERSION
            and limits.get(_CHARACTER_SIGNAL_SEMANTIC_SCOPE_VERSION_KEY)
            == _CHARACTER_SIGNAL_SEMANTIC_SCOPE_VERSION
            else None
        ),
        "signal_scope_review_v1": (
            limits[_CHARACTER_SIGNAL_SCOPE_REVIEW_KEY] if scope_review_valid else None
        ),
        "signal_scope_review_schema_version": (
            limits["signal_scope_review_schema_version"] if scope_review_valid else None
        ),
        "signal_scope_review_prompt_version": (
            limits["signal_scope_review_prompt_version"] if scope_review_valid else None
        ),
        "signal_scope_review_limits": (
            {
                key: limits[key]
                for key in sorted(_CHARACTER_SIGNAL_SCOPE_REVIEW_KEYS - {
                    _CHARACTER_SIGNAL_SCOPE_REVIEW_KEY,
                    "signal_scope_review_schema_version",
                    "signal_scope_review_prompt_version",
                })
            }
            if scope_review_valid else None
        ),
        "signal_support_trace_v1": (
            limits.get(_CHARACTER_SIGNAL_SUPPORT_TRACE_KEY)
            if type(limits.get(_CHARACTER_SIGNAL_SUPPORT_TRACE_KEY)) is bool
            and (
                limits[_CHARACTER_SIGNAL_SUPPORT_TRACE_KEY] is False
                or limits.get(_CHARACTER_SIGNAL_SUPPORT_ID_KEY) is True
            )
            and limits.get(_CHARACTER_SIGNAL_SUPPORT_TRACE_VERSION_KEY) == (
                _CHARACTER_SIGNAL_SUPPORT_TRACE_VERSION
                if limits[_CHARACTER_SIGNAL_SUPPORT_TRACE_KEY] else None
            )
            else None
        ),
        "signal_support_trace_version": (
            _CHARACTER_SIGNAL_SUPPORT_TRACE_VERSION
            if limits.get(_CHARACTER_SIGNAL_SUPPORT_TRACE_KEY) is True
            and limits.get(_CHARACTER_SIGNAL_SUPPORT_ID_KEY) is True
            and limits.get(_CHARACTER_SIGNAL_SUPPORT_TRACE_VERSION_KEY)
            == _CHARACTER_SIGNAL_SUPPORT_TRACE_VERSION
            else None
        ),
    }


def _service_preflight_gate(
    health: dict[str, Any], code_state: dict[str, Any], local_hash: str | None,
) -> str:
    """Require the API service bundle and revision to equal the frozen runner."""
    if not isinstance(local_hash, str) or SHA256.fullmatch(local_hash) is None:
        raise SafeFailure("local_service_artifact_unavailable", "runtime_preflight")
    provenance = _safe_character_runtime_provenance(health.get("runtime_provenance"))
    if provenance is None:
        raise SafeFailure("runtime_provenance_invalid", "runtime_preflight")
    build = provenance["build"]
    if build["git_revision"] != code_state["git_head"]:
        raise SafeFailure("service_build_revision_mismatch", "runtime_preflight")
    if build["service_artifact_sha256"] != local_hash:
        raise SafeFailure("service_artifact_mismatch", "runtime_preflight")
    limits = provenance["character_consistency_limits"]
    if (
        _CHARACTER_SIGNAL_SCOPE_REVIEW_KEY in limits
        and not _valid_character_scope_review_limits(limits)
    ):
        raise SafeFailure("runtime_provenance_invalid", "runtime_preflight")
    if provenance["capabilities"]["character_consistency"] is not True:
        raise SafeFailure("character_consistency_disabled", "runtime_preflight")
    digest = _runtime_provenance_digest(provenance)
    if digest is None:
        raise SafeFailure("runtime_provenance_invalid", "runtime_preflight")
    return digest


def _code_state() -> dict[str, Any]:
    """Expose only the revision and cleanliness, never dirty file names."""
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, check=True, timeout=10,
        ).stdout.strip().lower()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=ROOT, capture_output=True, text=True, check=True, timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"git_head": None, "worktree_clean": None}
    return {
        "git_head": head if GIT_HASH.fullmatch(head) is not None else None,
        "worktree_clean": status == "",
    }


def _suite_report(
    suite: VerifiedSuite, states: list[dict[str, Any]], cases: list[dict[str, Any]],
) -> dict[str, Any]:
    trials = [_score_trial(suite, state, cases) for state in states]
    project_hashes = [trial.get("project_id_sha256") for trial in trials]
    independent_projects = (
        len(project_hashes) == 3
        and all(isinstance(value, str) and SHA256.fullmatch(value) for value in project_hashes)
        and len(set(project_hashes)) == 3
    )
    stability = {
        case["case_id"]: {
            "passed_trials": sum(
                next((row["passed"] for row in trial["case_scores"] if row["case_id"] == case["case_id"]), False)
                for trial in trials
            ),
            "total_trials": 3,
        }
        for case in cases
    }
    return {
        "world_id": suite.world_id,
        "developer_visible": True,
        "dataset_sha256": suite.hashes,
        "case_count": suite.case_count,
        "trials": trials,
        "independent_projects": independent_projects,
        "stability": stability,
        "aggregate": {
            "complete_trials": sum(trial["draft_complete"] for trial in trials),
            "passed_trials": sum(trial["passed"] for trial in trials),
            "candidate_matches": sum((trial["candidate_review"] or {}).get("unique_matches", 0) for trial in trials),
            "candidate_expected": len(suite.plan["candidate_decisions"]) * 3,
            "candidate_reviewable": sum((trial["candidate_review"] or {}).get("reviewable", 0) for trial in trials),
            "unselected_reviewable_candidates": sum(
                trial["counts"]["unselected_reviewable_candidates"] or 0 for trial in trials
            ),
            "unselected_reviewable_unknown_trials": sum(
                trial["counts"]["unselected_reviewable_candidates"] is None for trial in trials
            ),
            "axes_bound": sum(trial["axis_count"] for trial in trials),
            "false_positives": sum(trial["counts"]["false_positives"] for trial in trials),
            "visible_false_positives": sum(trial["counts"]["visible_false_positives"] for trial in trials),
            "false_positive_unknown_cases": sum(trial["counts"]["false_positive_unknown_cases"] for trial in trials),
            "false_negatives": sum(trial["counts"]["false_negatives"] for trial in trials),
            "unevaluated_cases": sum(trial["counts"]["unevaluated_cases"] for trial in trials),
            "abstained": sum(trial["counts"]["abstained"] for trial in trials),
            "target_dimension_suspect_attribution_hits": sum(trial["counts"]["target_dimension_suspect_attribution_hits"] for trial in trials),
            "target_dimension_attribution_unavailable_cases": sum(
                trial["counts"]["target_dimension_attribution_unavailable_cases"] for trial in trials
            ),
            "actor_recall_unassessed_cases": sum(
                trial["counts"]["actor_recall_unassessed_cases"] for trial in trials
            ),
            "baseline_axis_snapshots_matched": sum(
                trial["counts"]["baseline_axis_snapshots_matched"] for trial in trials
            ),
            "baseline_axis_snapshot_eligible_cases": sum(
                trial["counts"]["baseline_axis_snapshot_eligible_cases"] for trial in trials
            ),
            "axis_observation_bindings_evidenced": sum(
                trial["counts"]["axis_observation_bindings_evidenced"] for trial in trials
            ),
            "axis_observation_binding_eligible_cases": sum(
                trial["counts"]["axis_observation_binding_eligible_cases"] for trial in trials
            ),
            "unavailable_evidence_roles": sum(trial["counts"]["unavailable_evidence_roles"] for trial in trials),
            "prompt_tokens": sum(
                _bounded_number((trial.get(stage) or {}).get("prompt_tokens"))
                for trial in trials for stage in ("baseline", "draft")
            ),
            "completion_tokens": sum(
                _bounded_number((trial.get(stage) or {}).get("completion_tokens"))
                for trial in trials for stage in ("baseline", "draft")
            ),
            "elapsed_seconds": round(sum(
                value for trial in trials for stage in ("baseline", "draft")
                for value in [(trial.get(stage) or {}).get("elapsed_seconds")]
                if isinstance(value, (int, float)) and 0 <= value <= 1_000_000
            ), 3),
        },
        "passed": independent_projects and all(trial["passed"] for trial in trials),
    }


def _emit_report(report: dict[str, Any], output_json: str | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if output_json:
        path = _resolve_output_json(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output:
            output.write(rendered + "\n")
    print(rendered)


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    requested_dataset = Path(args.dataset)
    resolved_dataset = requested_dataset.resolve()
    if resolved_dataset == DATASET.resolve():
        official_digest = PINNED_MANIFEST_SHA256
        dataset_kind = "pinned_challenge"
    elif resolved_dataset == DATASET_V2.resolve():
        official_digest = PINNED_MANIFEST_SHA256_V2
        dataset_kind = "pinned_challenge_v2"
    else:
        official_digest = None
        dataset_kind = "custom"
    official_dataset = official_digest is not None
    mode = "developer_diagnostic" if args.diagnostic_dev_one_trial else "strict"
    try:
        # Fail on an invalid report path before spending model tokens.
        if args.output_json:
            _resolve_output_json(args.output_json)
        supplied_digest = args.manifest_sha256
        if official_dataset and supplied_digest is not None and supplied_digest != official_digest:
            raise SafeFailure("official_manifest_digest_override", "fixture_preflight")
        expected_digest = official_digest if official_dataset else supplied_digest
        fixture = verify_fixture(
            requested_dataset, expected_manifest_sha256=expected_digest
        )
    except Exception as exc:
        return {
            "schema_version": "character-axis-live-v1",
            "mode": mode,
            "dataset_kind": dataset_kind,
            "claims": {"blind_holdout": False, "production_quality": False,
                       "independent_full_workflow_trials": False},
            "failure": _failure(exc), "passed": False,
        }, 1

    code_state = _code_state()
    if args.preflight_only:
        return {
            "schema_version": "character-axis-live-v1",
            "dataset_kind": dataset_kind,
            "mode": "preflight_only", "manifest_sha256": fixture.manifest_sha256,
            "dataset_sha256": {name: suite.hashes for name, suite in fixture.suites.items()},
            "code_state": code_state,
            "claims": {"blind_holdout": False, "production_quality": False,
                       "independent_full_workflow_trials": False},
            "preflight_verified": True,
            "passed": False,
        }, 0

    if mode == "strict" and not official_dataset:
        return {
            "schema_version": "character-axis-live-v1",
            "mode": mode, "dataset_kind": dataset_kind,
            "manifest_sha256": fixture.manifest_sha256,
            "code_state": code_state,
            "claims": {"blind_holdout": False, "production_quality": False,
                       "independent_full_workflow_trials": False},
            "failure": {"code": "strict_requires_pinned_fixture", "stage": "fixture_preflight", "details": {}},
            "passed": False,
        }, 1

    if mode == "strict" and (
        code_state["git_head"] is None or code_state["worktree_clean"] is not True
    ):
        return {
            "schema_version": "character-axis-live-v1",
            "mode": mode, "dataset_kind": dataset_kind,
            "manifest_sha256": fixture.manifest_sha256,
            "code_state": code_state,
            "claims": {"blind_holdout": False, "production_quality": False,
                       "independent_full_workflow_trials": False},
            "failure": {"code": "strict_worktree_not_clean", "stage": "code_preflight", "details": {}},
            "passed": False,
        }, 1

    local_service_hash = (
        _local_service_artifact_sha256(ROOT) if mode == "strict" else None
    )
    if mode == "strict" and (
        not isinstance(local_service_hash, str)
        or SHA256.fullmatch(local_service_hash) is None
    ):
        return {
            "schema_version": "character-axis-live-v1",
            "mode": mode, "dataset_kind": dataset_kind,
            "manifest_sha256": fixture.manifest_sha256,
            "code_state": code_state,
            "claims": {"blind_holdout": False, "production_quality": False,
                       "independent_full_workflow_trials": False},
            "failure": {"code": "local_service_artifact_unavailable", "stage": "code_preflight", "details": {}},
            "passed": False,
        }, 1

    states: dict[str, list[dict[str, Any]]] = {name: [] for name in SUITES}
    runtime: dict[str, Any] | None = None
    api_runtime_digest: str | None = None
    post_api_runtime_digest: str | None = None
    post_health_failure: str | None = None
    try:
        with httpx.Client(
            base_url=args.base_url.rstrip("/"), timeout=httpx.Timeout(30.0)
        ) as client:
            health = _request(client, "GET", "/health", "health")
            if not isinstance(health, dict):
                raise SafeFailure("health_contract", "runtime_preflight")
            if mode == "strict":
                api_runtime_digest = _service_preflight_gate(
                    health, code_state, local_service_hash
                )
            capabilities = health.get("runtime_provenance", {}).get("capabilities", {}) if isinstance(health.get("runtime_provenance"), dict) else {}
            if capabilities.get("character_consistency") is not True:
                raise SafeFailure("character_consistency_disabled", "runtime_preflight")
            runtime = _runtime_summary(health)
            model = health.get("model") if isinstance(health, dict) else None
            configured = isinstance(model, dict) and model.get("configured") is True
            if not configured:
                provider = _request(client, "GET", "/api/v1/account/model-provider", "model_provider")
                configured = isinstance(provider, dict) and (
                    provider.get("configured") is True
                    or provider.get("service_default_available") is True
                )
            if not configured:
                raise SafeFailure("model_not_configured", "runtime_preflight")
            run_suites = ("dev",) if mode == "developer_diagnostic" else SUITES
            trial_numbers = (1,) if mode == "developer_diagnostic" else range(1, 4)
            for suite_name in run_suites:
                suite = fixture.suites[suite_name]
                for trial in trial_numbers:
                    state: dict[str, Any] = {"trial": trial}
                    try:
                        _execute_trial(
                            client, suite, trial, state,
                            timeout_seconds=args.run_timeout_seconds,
                        )
                    except Exception as exc:
                        state["failure"] = _failure(exc)
                    states[suite_name].append(state)
            if mode == "strict":
                try:
                    after_health = _request(client, "GET", "/health", "health_postrun")
                    post_api_runtime_digest = _runtime_provenance_digest(
                        after_health.get("runtime_provenance")
                        if isinstance(after_health, dict) else None
                    )
                except Exception:
                    post_health_failure = "health_postrun_unavailable"
    except Exception as exc:
        return {
            "schema_version": "character-axis-live-v1",
            "mode": mode, "dataset_kind": dataset_kind,
            "manifest_sha256": fixture.manifest_sha256,
            "code_state": code_state,
            "claims": {"blind_holdout": False, "production_quality": False,
                       "independent_full_workflow_trials": False},
            "failure": _failure(exc), "passed": False,
        }, 1

    code_state_after = _code_state() if mode == "strict" else None
    local_service_hash_after = (
        _local_service_artifact_sha256(ROOT) if mode == "strict" else None
    )
    worker_digests = [
        (state.get(stage) or {}).get("runtime_provenance_sha256")
        for suite_states in states.values() for state in suite_states
        for stage in ("baseline", "draft")
    ]
    worker_observed = sum(
        isinstance(value, str) and SHA256.fullmatch(value) is not None
        for value in worker_digests
    )
    worker_matched = sum(
        api_runtime_digest is not None and value == api_runtime_digest
        for value in worker_digests
    )
    gate_reasons = []
    if mode == "strict":
        if len(worker_digests) != 12 or worker_observed != 12:
            gate_reasons.append("worker_runtime_provenance_incomplete")
        if worker_matched != 12:
            gate_reasons.append("api_worker_runtime_provenance_mismatch")
        if post_health_failure or post_api_runtime_digest != api_runtime_digest:
            gate_reasons.append("api_runtime_provenance_changed_or_unavailable")
        if code_state_after != code_state or local_service_hash_after != local_service_hash:
            gate_reasons.append("runner_code_changed_during_run")
    provenance_gate = {
        "applicable": mode == "strict",
        "passed": mode == "strict" and not gate_reasons,
        "reason_codes": gate_reasons,
        "local_service_artifact_sha256": local_service_hash,
        "api_runtime_provenance_sha256": api_runtime_digest,
        "worker_runtime_provenance_observed_runs": worker_observed,
        "worker_runtime_provenance_matches_api_runs": worker_matched,
        "expected_worker_runs": 12 if mode == "strict" else 2,
        "postrun_code_state_stable": (
            code_state_after == code_state and local_service_hash_after == local_service_hash
            if mode == "strict" else None
        ),
        "postrun_api_runtime_stable": (
            post_health_failure is None and post_api_runtime_digest == api_runtime_digest
            if mode == "strict" else None
        ),
    }

    suites: dict[str, Any] = {}
    for suite_name in run_suites:
        suite = fixture.suites[suite_name]
        try:
            # The gold file was hashed before HTTP, but is first parsed here.
            cases = _load_oracle(suite)
            if mode == "developer_diagnostic":
                trial = _score_trial(suite, states[suite_name][0], cases)
                suites[suite_name] = {
                    "world_id": suite.world_id,
                    "developer_visible": True,
                    "dataset_sha256": suite.hashes,
                    "case_count": suite.case_count,
                    "trials": [trial],
                    "aggregate": {
                        "false_negatives": trial["counts"]["false_negatives"],
                        "unevaluated_cases": trial["counts"]["unevaluated_cases"],
                    },
                    "diagnostic_completed": trial["draft_complete"],
                    "passed": False,
                }
            else:
                suites[suite_name] = _suite_report(suite, states[suite_name], cases)
        except Exception as exc:
            suites[suite_name] = {
                "world_id": suite.world_id,
                "failure": _failure(exc), "passed": False,
                "attempted_trials": len(states[suite_name]),
            }
    project_hashes = [
        state.get("project_id_sha256") for suite_states in states.values()
        for state in suite_states
    ]
    independent_projects = (
        len(project_hashes) == 6
        and all(isinstance(value, str) and SHA256.fullmatch(value) for value in project_hashes)
        and len(set(project_hashes)) == 6
    )
    complete = mode == "strict" and provenance_gate["passed"] and independent_projects and all(
        suites[name].get("passed") is True for name in SUITES
    )
    report = {
        "schema_version": "character-axis-live-v1",
        "mode": mode,
        "dataset_kind": dataset_kind,
        "manifest_sha256": fixture.manifest_sha256,
        "code_state": code_state,
        "runtime": runtime,
        "provenance_gate": provenance_gate,
        "claims": {
            "blind_holdout": False,
            "production_quality": False,
            "open_text_generalization": False,
            "complete_actor_attribution_accuracy": False,
            "true_actor_recall": False,
            "independent_full_workflow_trials": complete,
        },
        "metric_definitions": {
            "targeted_record_rejection_events": (
                "校验拒收与重试事件计数；非独立原文行数、原始模型错误数或召回分母"
            ),
            "accepted_draft_observation_total": (
                "证据校验后接纳的草稿信号数；不包含拒收或模型原始输出"
            ),
            "target_dimension_suspect_attribution_hits": (
                "仅计已接纳草稿信号在预注册目标角色、目标维度、极性与精确行上的可疑归属；"
                "不能证明行为语义，也不是总体角色归属准确率或 true-actor recall"
            ),
            "actor_recall_assessable": (
                "Oracle 未预注册正向角色召回锚点的 C=[] 案例为 false；不评价真实行动者的 trait 抽取召回"
            ),
            "evidence": (
                "B/C 为预注册来源行的覆盖率；完成审查时另核 reviewer 引用行，"
                "可见 issue 再核全部实际证据行；不声称覆盖了每条可能相关的原文证据"
            ),
            "visible_issue_evidence": (
                "最终可见 issue 必须覆盖预注册 B/C 精确单行；其余实际引用只允许落在"
                "预注册 B/C/G/X 行。G/X 在此仅为允许的背景，是否用于解释另看 reviewer citation_refs"
            ),
            "false_positives": (
                "可见 contradicts issue 数加仅在内部 trace 观测到的冲突数；"
                "trace 缺失且无可见冲突的案例另计为 false_positive_unknown_cases"
            ),
            "false_negatives": (
                "仅对已取得可评分 case result 的预注册 conflict 案统计；"
                "未进入草稿或缺少 case result 的案例另计 unevaluated_cases，绝不追认为漏报"
            ),
            "reason_counts": (
                "仅展示服务端静态原因码白名单；动态或未知键只计入 unreported_reason_entries，"
                "不输出其文本"
            ),
            "accepted_signal_histogram": (
                "已接纳信号的 source_kind×stability×dimension 有界枚举计数；不含姓名、"
                "正文、trait_key 或源行，且不代表原始模型记录或被拒收记录"
            ),
            "candidate_eligibility": (
                "正式/历史 core 或 stable 信号数与真实候选构建后的限额前候选数；"
                "不公开历史精确分组，不能仅凭本简表推断每条信号被滤除的原因"
            ),
            "evidence_mismatch_counts": (
                "各次模型抽取尝试中的证据拒收/重试后恢复事件分类计数；"
                "不是独立原文行数、最终失败次数或剧情错误数"
            ),
            "core_label_scope_counts": (
                "正式角色设定抽取中，核心标签作用域拒收事件的固定类别计数；"
                "含重试后恢复事件，与两个 core_label_scope 原因码之和守恒；"
                "不是独立原文行数、模型准确率或解析器错误数；旧报告缺字段为 unavailable"
            ),
            "accepted_model_core_without_literal_label_count": (
                "阶段最终按服务端信号 ID 去重后，正式设定中模型声明 core 但证据未出现"
                "字面核心标签的接纳信号数；不同于抽取器的分块局部观察数，独立于拒收计数，"
                "仅供观察，不代表该声明被验证正确；缺少可信信号总数/类别直方图时为 unavailable"
            ),
            "evidence_mismatch_chunks": (
                "每个分块或目标抽取调用汇总其各次模型尝试中的证据拒收/恢复事件；"
                "仅展示固定类别、"
                "来源/阶段枚举、文档与分块序号及有界计数，不代表独立原文行或最终失败数；"
                "不含原文、文档名、候选 ID、密钥或服务地址，也不改变准入与评分"
            ),
            "partial_baseline_inventory": (
                "仅在基线部分完成且准入失败时，对待审核候选执行只读查询；"
                "以匿名序号槽报告预注册锚点的脱敏匹配计数和额外可审核候选数，不确认候选、"
                "不上传草稿，也不改变基线失败或试验通过状态"
            ),
            "support_location_diagnostic_v1": (
                "DEV 基线候选快照可用时，独立统计后端已验证的 target 分句与冻结短句的"
                "归一化逐字定位；要求本轮项目、来源运行、文档 ID/哈希和精确行一致。"
                "不使用 statement、整行其他分句或承接桥段；仅输出匿名计数，不选候选、"
                "不改变冻结 selector 或准入 gate，也不代表语义正确率；诊断不可用时"
                "只输出固定原因码，不将诊断失败扩展为试验失败"
            ),
            "unselected_reviewable_candidates": (
                "基线运行产生、但不在预注册作者审核计划内的可审核候选数；字段缺失或无效"
                "时记为未知并使试验不通过，不得只凭计划内候选匹配判通过"
            ),
            "provenance_gate": (
                "核对本地源码包摘要、Git HEAD、API 与十二次 worker 阶段摘要及运行后稳定性；"
                "源码包摘要不是 OCI 镜像摘要或运行内存证明"
            ),
        },
        "preflight_verified_suites": list(SUITES),
        "independent_projects": independent_projects if mode == "strict" else False,
        "suites": suites,
        "passed": complete,
    }
    return report, 0 if complete else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default=str(DATASET),
        help="official v1 (default), official v2, or an explicitly pinned custom fixture",
    )
    parser.add_argument(
        "--manifest-sha256",
        help="required for custom fixtures; official fixtures use their built-in pins",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--run-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--diagnostic-dev-one-trial", action="store_true",
        help="one DEV project only; never passes the strict quality gate",
    )
    parser.add_argument("--output-json", help="new JSON path inside artifacts/")
    args = parser.parse_args()
    if args.run_timeout_seconds <= 0:
        parser.error("--run-timeout-seconds must be positive")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    result, exit_code = run(parsed)
    try:
        _emit_report(result, parsed.output_json)
    except (OSError, ValueError):
        # The report has already been redacted; a failed artifact write must
        # not reveal the filesystem path or suppress the verdict.
        print(json.dumps(result, ensure_ascii=False, indent=2))
        exit_code = 1
    raise SystemExit(exit_code)
