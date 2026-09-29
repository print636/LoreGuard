"""Run independent character-continuity workflows through the real HTTP API.

This is a developer-visible acceptance runner, not a blind benchmark.  It
never reads a Provider credential: the already running API owns all model
configuration.  Output is limited to fixture labels, hashes, evidence file/line
references, counters, verdict labels and token usage; source text, credentials,
the configured endpoint and upstream responses are not printed.

Passing uses model-completion counters and complete per-case material/explanation
coverage. Formal issues, review clues and provisional clues are fetched and
scored as separate report classes; an unassessed or partial case never receives
clean-case credit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import UUID, uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.character_trait_extraction import stable_trait_identity
from app.character_traits import (
    _OBJECT_BEARING_TRAIT_DIMENSIONS,
    _validated_comparison_key,
)
from scripts.run_evidence_investigator_live import (
    _local_service_artifact_sha256,
    _safe_runtime_provenance as _safe_evidence_runtime_provenance,
)


FIXTURES = {
    "demo": ROOT / "data" / "character-continuity-demo",
    "alpha-v1": ROOT / "data" / "character-ooc-alpha-v1",
    "ooc-v1": ROOT / "data" / "character-ooc-challenge-v1",
    "ooc-transfer-v1": ROOT / "data" / "character-ooc-transfer-v1",
}
DEMO = FIXTURES["demo"]
BASELINE_FILES = (
    ("01-world-setting.md", "canon"),
    ("02-character-profiles.md", "character_profile"),
    ("03-published-history-v1.0.md", "chapter"),
)
DRAFT_FILE = "04-draft-event-v1.1.md"
ORACLE_FILE = "acceptance-oracle.json"
TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
DIAGNOSTIC_RECORD_REASONS = frozenset({
    "evidence_range", "evidence_mismatch", "character_support",
    "key_object_required", "key_object_support", "statement_support",
    "schema_validation", "record_validation", "directional_trait_key",
})
DIAGNOSTIC_SOURCE_COUNTS = frozenset({"source_formal", "source_history"})
DIAGNOSTIC_REGENERATION_COUNTS = frozenset(
    f"regenerated_from_{reason}" for reason in DIAGNOSTIC_RECORD_REASONS
)
_TRACE_OBSERVATION_KINDS = frozenset({
    "explicit_declaration", "preference_expression", "dialogue",
    "speech_sample", "action", "decision", "interaction",
    "state_description",
})
_TRACE_POLARITIES = frozenset({"positive", "negative", "neutral", "unclear"})
_ALPHA_OOC_DIMENSIONS = (
    "core_trait", "stable_preference", "speech_pattern", "value_boundary",
    "relationship_attitude", "motivation_goal",
)
_ALPHA_FORMAL_SUPPORT_BINDING_IDENTITY = {
    "signal_support_id_v4": True,
    "signal_support_segmenter_version": "assertion-index-v1",
    "signal_semantic_scope_v5": True,
    "signal_semantic_scope_version": "semantic-scope-v6",
    "signal_scope_review_v1": True,
    "signal_scope_review_schema_version": "character-scope-review-v2",
    "signal_scope_review_prompt_version": "character-scope-review-prompt-v4",
}
_ALPHA_EXPLANATION_REVIEW_IDENTITY = {
    "explanation_review_v1": True,
    "explanation_review_schema_version": "character-explanation-review-v2",
    "explanation_review_prompt_version": (
        "character-explanation-review-prompt-v2"
    ),
}
_STABILITY_OBSERVATION_KEY = (
    "stable_visible_issue_semantic_identity_across_independent_trials"
)
_ALPHA_FIXTURE = "alpha-v1"
_LEGACY_MINIMUM_TRIALS = 3
_TOKEN_ADMISSION_FIELDS = (
    "stage_phase", "signal_phase", "chunk_ordinal", "target_ordinal",
    "estimated_tokens", "available_tokens", "stage_remaining_before",
    "reviewer_reserve_tokens", "model_calls_before_failure",
)
_TRACE_SENSITIVE_LABEL = re.compile(
    r"(?:api[_\s-]?key|base[_\s-]?url|authorization|bearer\s+|password|"
    r"credential|private[_\s-]?key|(?:^|[^a-z0-9])sk-[a-z0-9]{8,}|密钥|密码)",
    re.IGNORECASE,
)
_AUTHOR_AXIS_UNSAFE_TEXT = re.compile(
    r"(?:[a-z][a-z0-9+.-]{1,15}://|www\.)|"
    r"(?:api[_\s-]?key|base[_\s-]?url|authorization|bearer\s+|password|"
    r"credential|private[_\s-]?key|(?:^|[^a-z0-9])sk-[a-z0-9]{8,}|\u5bc6\u94a5|\u5bc6\u7801)",
    re.IGNORECASE,
)
_AUTHOR_AXIS_PLAN_ID = re.compile(r"[a-z][a-z0-9_]{0,63}")
_SCOPED_AXIS_TRAIT_TYPES = frozenset({"value", "behavior_boundary"})
_AUTHOR_AXIS_PLAN_FIELDS = frozenset({
    "plan_id", "case_id", "trait_type", "display_name", "definition",
    "positive_proposition", "applicability_scope", "axis_alignment",
})
_TARGET_BOUND_DRAFT_REVIEW_V2 = {
    "target_bound_draft_review_schema_version": (
        "character-target-bound-draft-review-v2"
    ),
    "target_bound_draft_review_batch_schema_version": (
        "character-target-bound-draft-review-batch-v2"
    ),
    "target_bound_draft_review_prompt_version": (
        "character-target-bound-draft-review-prompt-v2"
    ),
    "target_bound_draft_review_signal_prompt_version": (
        "character-target-bound-draft-signal-prompt-v2"
    ),
    "target_bound_draft_review_clause_index_version": (
        "draft-actor-clause-index-v1"
    ),
}
_TARGET_BOUND_DRAFT_REVIEW_V3 = {
    "target_bound_draft_review_schema_version": (
        "character-target-bound-draft-review-v4"
    ),
    "target_bound_draft_review_batch_schema_version": (
        "character-target-bound-draft-review-batch-v4"
    ),
    "target_bound_draft_review_prompt_version": (
        "character-target-bound-draft-review-prompt-v8"
    ),
    "target_bound_draft_review_signal_prompt_version": (
        "character-target-bound-draft-signal-prompt-v3"
    ),
    "target_bound_draft_review_clause_index_version": (
        "draft-actor-clause-index-v1"
    ),
}


def _safe_runtime_provenance(value: object) -> dict[str, Any] | None:
    """Strict character-safe projection built on the shared consumer.

    The shared parser validates the exact content-free schema. Character runs
    may legitimately disable embeddings, so that one producer-supported form
    is validated through a non-mutating sentinel adaptation and restored in
    the projection. Character-only scalar and effective-provider constraints
    then match the stricter character-axis consumer.
    """

    safe = _safe_evidence_runtime_provenance(value)
    if safe is None and type(value) is dict:
        capabilities = value.get("capabilities")
        rag = value.get("rag")
        if (
            type(capabilities) is dict
            and capabilities.get("embeddings") is False
            and type(rag) is dict
            and rag.get("profile_fingerprint") is None
        ):
            adapted = {
                **value,
                "capabilities": {**capabilities, "embeddings": True},
                "rag": {**rag, "profile_fingerprint": "0" * 64},
            }
            safe = _safe_evidence_runtime_provenance(adapted)
            if safe is not None:
                safe["capabilities"]["embeddings"] = False
                safe["rag"]["profile_fingerprint"] = None
    if safe is None:
        return None
    if type(value) is not dict:
        return None
    raw_build = value.get("build")
    raw_provider = value.get("chat_provider")
    if type(raw_build) is not dict or type(raw_provider) is not dict:
        return None
    revision = raw_build.get("git_revision")
    temperature = raw_provider.get("temperature")
    if (
        revision is not None
        and (type(revision) is not str or len(revision) not in {40, 64})
    ) or type(temperature) not in {int, float}:
        return None
    embeddings_enabled = safe["capabilities"]["embeddings"]
    profile = safe["rag"]["profile_fingerprint"]
    if (
        embeddings_enabled
        and not isinstance(profile, str)
    ) or (not embeddings_enabled and profile is not None):
        return None
    limits = safe["character_consistency_limits"]
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
    return safe


def _dataset_label() -> str:
    return (
        "original-developer-visible-demo"
        if DEMO == FIXTURES["demo"]
        else DEMO.name
    )


class AcceptanceFailure(RuntimeError):
    def __init__(self, message: str, *, safe_payload: dict[str, Any]) -> None:
        super().__init__(message)
        self.safe_payload = safe_payload


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _comparison_identity(
    *, dimension: str, trait_key: str, key_object: str = ""
) -> dict[str, str]:
    """Return the legacy label-derived identity for a confirmed trait.

    The raw model label is intentionally not copied into the acceptance report.
    New object-axis runs bind by candidate ID and use the frozen trace key.
    """

    comparison_key = stable_trait_identity(dimension, trait_key, key_object)
    return {
        "dimension": dimension,
        "comparison_key_sha256": _sha256_text(comparison_key),
    }


def _trace_comparison_sha256(trace: dict[str, Any]) -> str | None:
    value = trace.get("comparison_key")
    if not isinstance(value, str) or not value:
        return None
    return _sha256_text(value)


def _candidate_id_sha256(value: object) -> str | None:
    return _safe_uuid_sha256(value)


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _safe_uuid_sha256(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        if str(UUID(value)) != value:
            return None
    except ValueError:
        return None
    return _sha256_text(value)


def _safe_author_axis_text(value: object, *, maximum: int) -> bool:
    """Mirror the API's bounded author-text envelope during preflight."""

    return (
        isinstance(value, str)
        and 1 <= len(value) <= maximum
        and value == " ".join(value.split())
        and not any(ord(character) < 32 or ord(character) == 127 for character in value)
        and _AUTHOR_AXIS_UNSAFE_TEXT.search(value) is None
    )


def _trace_for_candidate(
    trace: list[dict[str, Any]],
    *,
    candidate_id_sha256: str | None,
    character_key: str,
    dimension: str,
    legacy_comparison_sha256: str | None = None,
) -> dict[str, Any] | None:
    """Bind only a unique candidate trace, or a unique pre-ID legacy trace."""

    if not _valid_sha256(candidate_id_sha256):
        return None
    rows = [row for row in trace if isinstance(row, dict)]
    matches = [
        row for row in rows
        if row.get("confirmed_candidate_id_sha256") == candidate_id_sha256
    ]
    if matches:
        if len(matches) != 1:
            return None
        match = matches[0]
        if (
            match.get("character_key") != character_key
            or match.get("dimension") != dimension
            or not _valid_sha256(match.get("comparison_key_sha256"))
        ):
            return None
        return match
    if (
        not _valid_sha256(legacy_comparison_sha256)
        or any(row.get("confirmed_candidate_id_sha256") is not None for row in rows)
    ):
        return None
    matches = [
        row for row in rows
        if row.get("character_key") == character_key
        and row.get("dimension") == dimension
        and row.get("comparison_key_sha256") == legacy_comparison_sha256
    ]
    return matches[0] if len(matches) == 1 else None


def _request(
    client: httpx.Client,
    method: str,
    path: str,
    **kwargs: Any,
) -> dict[str, Any]:
    response = client.request(method, path, **kwargs)
    if response.status_code >= 400:
        raise AcceptanceFailure(
            "http_request_failed",
            safe_payload={
                "code": "http_request_failed",
                "stage": "http_api",
                "details": {
                    "method": method,
                    "route_kind": path.split("?", 1)[0],
                    "status_code": response.status_code,
                },
            },
        )
    try:
        payload = response.json()
    except (ValueError, TypeError) as exc:
        raise AcceptanceFailure(
            "http_response_invalid_json",
            safe_payload={
                "code": "http_response_invalid_json",
                "stage": "http_api",
                "details": {"method": method, "route_kind": path.split("?", 1)[0]},
            },
        ) from exc
    if not isinstance(payload, dict):
        raise AcceptanceFailure(
            "http_response_not_object",
            safe_payload={
                "code": "http_response_not_object",
                "stage": "http_api",
                "details": {"method": method, "route_kind": path.split("?", 1)[0]},
            },
        )
    return payload


def _request_list(
    client: httpx.Client,
    method: str,
    path: str,
    **kwargs: Any,
) -> list[Any]:
    """Request a JSON list without weakening the object-only API helper."""

    response = client.request(method, path, **kwargs)
    if response.status_code >= 400:
        raise AcceptanceFailure(
            "http_request_failed",
            safe_payload={
                "code": "http_request_failed",
                "stage": "http_api",
                "details": {
                    "method": method,
                    "route_kind": path.split("?", 1)[0],
                    "status_code": response.status_code,
                },
            },
        )
    try:
        payload = response.json()
    except (ValueError, TypeError) as exc:
        raise AcceptanceFailure(
            "http_response_invalid_json",
            safe_payload={
                "code": "http_response_invalid_json",
                "stage": "http_api",
                "details": {"method": method, "route_kind": path.split("?", 1)[0]},
            },
        ) from exc
    if not isinstance(payload, list):
        raise AcceptanceFailure(
            "http_response_not_list",
            safe_payload={
                "code": "http_response_not_list",
                "stage": "http_api",
                "details": {"method": method, "route_kind": path.split("?", 1)[0]},
            },
        )
    return payload


def _context(*, release_key: str, ordinal: int, published: bool) -> dict[str, Any]:
    return {
        "resolution_state": "confirmed",
        "publication_status": "published" if published else "draft",
        "scope": {
            "schema_version": 1,
            "timeline_key": "main",
            "release": {"key": release_key, "ordinal": ordinal},
            "branch": {"path": ["main"]},
        },
    }


def _upload(
    client: httpx.Client,
    project_id: str,
    filename: str,
    role: str,
    *,
    release_key: str,
    ordinal: int,
    published: bool,
) -> dict[str, Any]:
    path = DEMO / filename
    return _request(
        client,
        "POST",
        f"/api/v1/projects/{project_id}/documents/text",
        json={
            "name": filename,
            "content": path.read_text(encoding="utf-8"),
            "document_role": role,
            "story_scope": "main",
            "narrative_context": _context(
                release_key=release_key,
                ordinal=ordinal,
                published=published,
            ),
        },
    )


def _start_run(
    client: httpx.Client, project_id: str, *, mode: str | None = None
) -> str:
    payload = _request(
        client,
        "POST",
        f"/api/v1/projects/{project_id}/analysis-runs",
        headers={"Idempotency-Key": f"character-live-{uuid4().hex}"},
        json={"mode": mode} if mode is not None else {},
    )
    run_id = payload.get("id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("analysis response is missing a run id")
    return run_id


def _wait_run(
    client: httpx.Client,
    run_id: str,
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    deadline = time.monotonic() + timeout_seconds
    while True:
        payload = _request(client, "GET", f"/api/v1/analysis-runs/{run_id}")
        status = payload.get("status")
        if status in TERMINAL_STATUSES:
            return {
                **payload,
                "_acceptance_elapsed_seconds": round(time.monotonic() - started, 3),
            }
        if time.monotonic() >= deadline:
            raise RuntimeError(f"analysis {run_id} did not finish before timeout")
        time.sleep(1.0)


def _validated_author_axis_plans(
    value: object,
    *,
    selectors_by_case: dict[str, dict[str, Any]],
    expected_by_case: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate author intent without accepting a guessed comparison key."""

    if not isinstance(value, list):
        raise RuntimeError("acceptance oracle author axis plan must be a list")
    plan_ids: set[str] = set()
    case_ids: set[str] = set()
    plans: list[dict[str, Any]] = []
    for plan in value:
        if not isinstance(plan, dict) or set(plan) != _AUTHOR_AXIS_PLAN_FIELDS:
            raise RuntimeError(
                "acceptance oracle author axis plan fields are invalid"
            )
        plan_id = plan.get("plan_id")
        case_id = plan.get("case_id")
        trait_type = plan.get("trait_type")
        if (
            not isinstance(plan_id, str)
            or _AUTHOR_AXIS_PLAN_ID.fullmatch(plan_id) is None
            or plan_id in plan_ids
        ):
            raise RuntimeError(
                "acceptance oracle author axis ids are invalid or duplicated"
            )
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in case_ids
        ):
            raise RuntimeError(
                "acceptance oracle author axis cases are invalid or duplicated"
            )
        selector = selectors_by_case.get(case_id)
        expected = expected_by_case.get(case_id)
        if selector is None or expected is None:
            raise RuntimeError(
                "acceptance oracle author axis references an unknown case"
            )
        # Scope is meaningful only for these product-supported dimensions.
        # The exact-field check above also rejects oracle-owned comparison_key,
        # key_object, trait_key, and comparison hashes: identity must come from
        # the uniquely selected, frozen API candidate instead.
        if (
            trait_type not in _SCOPED_AXIS_TRAIT_TYPES
            or selector.get("trait_type") != trait_type
            or expected.get("dimension") != trait_type
        ):
            raise RuntimeError(
                "acceptance oracle author axis scope or dimension is invalid"
            )
        if (
            not _safe_author_axis_text(plan.get("display_name"), maximum=80)
            or not _safe_author_axis_text(plan.get("definition"), maximum=200)
            or not _safe_author_axis_text(
                plan.get("positive_proposition"), maximum=200
            )
            or not _safe_author_axis_text(
                plan.get("applicability_scope"), maximum=200
            )
            or plan.get("axis_alignment") not in {"same", "opposite"}
        ):
            raise RuntimeError(
                "acceptance oracle author axis author intent is invalid"
            )
        plan_ids.add(plan_id)
        case_ids.add(case_id)
        plans.append(plan)
    return plans


def _validate_oracle_payload(
    payload: object, *, require_frozen_semantic_axis: bool = False
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("acceptance oracle must be a JSON object")
    if payload.get("schema_version") != "character-continuity-live-oracle-v2":
        raise RuntimeError("acceptance oracle schema version is unsupported")
    boundary = payload.get("dataset_boundary")
    if not isinstance(boundary, dict) or boundary != {
        "developer_visible": True,
        "blind_holdout": False,
        "production_quality_claim": False,
        "open_text_generalization_claim": False,
    }:
        raise RuntimeError("acceptance oracle dataset boundary is invalid")
    selector_policy = payload.get("selector_policy")
    if selector_policy not in {None, "frozen_semantic_axis_v1"}:
        raise RuntimeError("acceptance oracle selector policy is invalid")
    if require_frozen_semantic_axis and selector_policy != "frozen_semantic_axis_v1":
        raise RuntimeError("frozen transfer oracle selector policy is missing")
    selectors = payload.get("confirm_candidates")
    expected = payload.get("expected_cases")
    if not isinstance(selectors, list) or not selectors:
        raise RuntimeError("acceptance oracle has no candidate selectors")
    if not isinstance(expected, list) or not expected:
        raise RuntimeError("acceptance oracle has no expected cases")
    selector_ids = {
        row.get("case_id") for row in selectors if isinstance(row, dict)
    }
    expected_ids = {row.get("case_id") for row in expected if isinstance(row, dict)}
    if (
        len(selector_ids) != len(selectors)
        or len(expected_ids) != len(expected)
        or selector_ids != expected_ids
        or not all(isinstance(value, str) and value for value in selector_ids)
    ):
        raise RuntimeError("acceptance oracle case ids are invalid or inconsistent")
    selectors_by_case = {row["case_id"]: row for row in selectors}
    expected_by_case = {row["case_id"]: row for row in expected}
    author_axes = _validated_author_axis_plans(
        payload.get("author_axes", []),
        selectors_by_case=selectors_by_case,
        expected_by_case=expected_by_case,
    )
    for selector in selectors:
        if not isinstance(selector, dict):
            raise RuntimeError("acceptance oracle selector is invalid")
        if selector_policy == "frozen_semantic_axis_v1":
            value_options = selector.get("value_contains_any")
            if (
                not {
                    "case_id", "character_key", "trait_type", "source_document",
                    "source_line", "stability", "value_contains_any",
                    "trait_polarity_options",
                } <= set(selector)
                or not all(
                    isinstance(selector[field], str) and bool(selector[field])
                    for field in (
                        "case_id", "character_key", "trait_type",
                        "source_document",
                    )
                )
                or type(selector["source_line"]) is not int
                or selector["source_line"] < 1
                or selector["stability"] not in {"core", "stable"}
                or not isinstance(value_options, list)
                or not value_options
                or any(not isinstance(value, str) or not value for value in value_options)
                or len(set(value_options)) != len(value_options)
                or "value_contains" in selector
                or "trait_polarity_options" not in selector
            ):
                raise RuntimeError("frozen selector lacks independent semantic anchors")
        variants = selector.get("trait_polarity_options")
        if variants is None:
            continue
        if (
            "polarity" in selector
            or not isinstance(variants, list)
            or not variants
            or any(
                not isinstance(row, dict)
                or set(row) != {"trait_key", "polarity"}
                or not isinstance(row["trait_key"], str)
                or not row["trait_key"]
                or not isinstance(row["polarity"], str)
                or row["polarity"] not in {"positive", "negative"}
                for row in variants
            )
            or len({(row["trait_key"], row["polarity"]) for row in variants})
            != len(variants)
        ):
            raise RuntimeError("acceptance oracle trait polarity variants are invalid")
    for row in expected:
        if not isinstance(row, dict):
            raise RuntimeError("acceptance oracle case must be an object")
        if selector_policy == "frozen_semantic_axis_v1":
            required_fields = {
                "case_id", "character_key", "dimension", "review_outcome",
                "final_outcome", "promote_reason", "review_verdict", "visible",
                "expected_citation_roles",
            }
            exact_observation_count = row.get("matched_observation_count")
            minimum_observation_count = row.get("min_matched_observation_count")
            maximum_observation_count = row.get("max_matched_observation_count")
            has_exact_count = "matched_observation_count" in row
            has_range_count = (
                "min_matched_observation_count" in row
                or "max_matched_observation_count" in row
            )
            count_contract_valid = has_exact_count != has_range_count and (
                (
                    has_exact_count
                    and type(exact_observation_count) is int
                    and exact_observation_count >= 0
                )
                or (
                    has_range_count
                    and type(minimum_observation_count) is int
                    and type(maximum_observation_count) is int
                    and 0 <= minimum_observation_count <= maximum_observation_count
                )
            )
            review_contract_valid = (
                (
                    row.get("review_outcome") == "not_run"
                    and row.get("review_verdict") is None
                )
                or (
                    row.get("review_outcome") == "completed"
                    and row.get("review_verdict")
                    in {"contradicts", "explained", "uncertain"}
                )
            )
            if (
                not required_fields <= set(row)
                or not all(
                    isinstance(row[field], str) and bool(row[field])
                    for field in (
                        "case_id", "character_key", "dimension",
                        "final_outcome", "promote_reason",
                    )
                )
                or not review_contract_valid
                or not count_contract_valid
            ):
                raise RuntimeError("frozen expected case lacks an explicit contract")
        has_exact_prepare_reason = "prepare_reason" in row
        has_prepare_reason_allowlist = "prepare_reason_any_of" in row
        if has_exact_prepare_reason == has_prepare_reason_allowlist:
            raise RuntimeError(
                "acceptance oracle case must define exactly one prepare reason contract"
            )
        if has_exact_prepare_reason:
            prepare_reason = row.get("prepare_reason")
            if not isinstance(prepare_reason, str) or not prepare_reason.strip():
                raise RuntimeError("acceptance oracle prepare reason is invalid")
        else:
            prepare_reasons = row.get("prepare_reason_any_of")
            if (
                not isinstance(prepare_reasons, list)
                or not prepare_reasons
                or not all(
                    isinstance(value, str) and bool(value.strip())
                    for value in prepare_reasons
                )
                or len(prepare_reasons) != len(set(prepare_reasons))
            ):
                raise RuntimeError(
                    "acceptance oracle prepare reason allowlist is invalid"
                )
        roles = row.get("expected_citation_roles")
        if (
            not isinstance(roles, list)
            or not all(isinstance(role, str) for role in roles)
            or len(roles) != len(set(roles))
            or not set(roles) <= {"B", "C", "G", "X"}
            or type(row.get("visible")) is not bool
            or row.get("review_outcome") not in {"completed", "not_run"}
        ):
            raise RuntimeError("acceptance oracle case contract is invalid")
        report_class = row.get("expected_report_class")
        if report_class is not None and report_class not in {
            "formal", "review_clue", "none"
        }:
            raise RuntimeError("acceptance oracle report class is invalid")
        if report_class == "formal" and (
            row["visible"] is not True
            or not isinstance(row.get("visible_issue"), dict)
        ):
            raise RuntimeError("formal oracle case lacks a full issue signature")
        if report_class == "review_clue" and (
            row["visible"] is not True
            or not isinstance(row.get("review_clue"), dict)
        ):
            raise RuntimeError("review clue oracle case lacks a full signature")
        if report_class == "none" and row["visible"] is not False:
            raise RuntimeError("non-reporting oracle case cannot be visible")
        if (
            report_class is None
            and row["visible"] is True
            and not isinstance(row.get("visible_issue"), dict)
        ):
            raise RuntimeError("visible oracle case lacks a full issue signature")
    return {**payload, "author_axes": author_axes}


def _load_oracle() -> dict[str, Any]:
    payload = json.loads((DEMO / ORACLE_FILE).read_text(encoding="utf-8"))
    return _validate_oracle_payload(
        payload, require_frozen_semantic_axis=DEMO == FIXTURES["ooc-transfer-v1"]
    )


def _matches_selector(candidate: dict[str, Any], selector: dict[str, Any]) -> bool:
    if not _matches_source_anchored_selector(candidate, selector):
        return False
    variants = selector.get("trait_polarity_options")
    if variants is not None and not any(
        candidate.get("trait_key") == row.get("trait_key")
        and candidate.get("polarity") == row.get("polarity")
        for row in variants
    ):
        return False
    for field in ("polarity", "stability", "key_object"):
        expected_value = selector.get(field)
        if expected_value is not None and candidate.get(field) != expected_value:
            return False
    return True


def _matches_source_anchored_selector(
    candidate: dict[str, Any], selector: dict[str, Any]
) -> bool:
    """DEV diagnosis only: identify a setting candidate by source, not trait semantics."""
    if (
        candidate.get("character_key") != selector.get("character_key")
        or candidate.get("trait_type") != selector.get("trait_type")
        or candidate.get("origin") != "explicit_setting"
        or candidate.get("reviewable") is not True
    ):
        return False
    document = selector.get("source_document")
    line = selector.get("source_line")
    if not isinstance(document, str) or type(line) is not int:
        return False
    value_contains = selector.get("value_contains")
    evidence = candidate.get("evidence")
    evidence_texts = (
        [str(row.get("text") or "") for row in evidence if isinstance(row, dict)]
        if isinstance(evidence, list)
        else []
    )
    if value_contains is not None:
        if not isinstance(value_contains, str) or not value_contains:
            return False
        if (
            value_contains not in str(candidate.get("value") or "")
            and not any(value_contains in text for text in evidence_texts)
        ):
            return False
    value_contains_any = selector.get("value_contains_any")
    if value_contains_any is not None and (
        not isinstance(value_contains_any, list)
        or not value_contains_any
        or not any(
            isinstance(value, str)
            and value
            and (
                value in str(candidate.get("value") or "")
                or any(value in text for text in evidence_texts)
            )
            for value in value_contains_any
        )
    ):
        return False
    if not isinstance(evidence, list):
        return False
    return any(
        isinstance(row, dict)
        and row.get("document_name") == document
        and type(row.get("line_start")) is int
        and type(row.get("line_end")) is int
        and row["line_start"] <= line <= row["line_end"]
        for row in evidence
    )


def _author_axis_failure(
    code: str,
    *,
    case_id: str,
    candidate_id_sha256: str | None = None,
    contract_check: str,
) -> AcceptanceFailure:
    details: dict[str, Any] = {
        "case_id": case_id,
        "contract_check": contract_check,
    }
    if _valid_sha256(candidate_id_sha256):
        details["candidate_id_sha256"] = candidate_id_sha256
    return AcceptanceFailure(
        code,
        safe_payload={
            "code": code,
            "stage": "baseline_author_axis",
            "details": details,
        },
    )


def _create_and_verify_author_axes(
    client: httpx.Client,
    project_id: str,
    selected: dict[str, tuple[str, dict[str, Any]]],
    plans: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Create every planned axis before candidate review mutates any state."""

    selected_by_case = {
        case_id: (candidate_id, candidate)
        for candidate_id, (case_id, candidate) in selected.items()
    }
    bindings: dict[str, dict[str, Any]] = {}
    for plan in plans:
        case_id = str(plan["case_id"])
        selected_row = selected_by_case.get(case_id)
        if selected_row is None:
            raise _author_axis_failure(
                "author_axis_candidate_missing",
                case_id=case_id,
                contract_check="selected_case",
            )
        candidate_id, candidate = selected_row
        candidate_digest = _candidate_id_sha256(candidate_id)
        trait_type = plan.get("trait_type")
        comparison_key = candidate.get("comparison_key")
        try:
            validated_key = _validated_comparison_key(
                comparison_key, trait_type=str(trait_type)
            )
        except ValueError:
            validated_key = None
        if (
            candidate_digest is None
            or trait_type not in _SCOPED_AXIS_TRAIT_TYPES
            or candidate.get("trait_type") != trait_type
            or validated_key is None
            or validated_key != comparison_key
            or not _safe_author_axis_text(validated_key, maximum=200)
        ):
            raise _author_axis_failure(
                "author_axis_candidate_comparison_key_invalid",
                case_id=case_id,
                candidate_id_sha256=candidate_digest,
                contract_check="candidate_frozen_comparison_key",
            )
        created = _request(
            client,
            "POST",
            f"/api/v1/projects/{project_id}/character-trait-axes",
            json={
                "trait_type": trait_type,
                "display_name": plan["display_name"],
                "definition": plan["definition"],
                "positive_proposition": plan["positive_proposition"],
                # This value is intentionally taken only from the selected
                # frozen candidate; the oracle is forbidden from supplying it.
                "comparison_key": validated_key,
                "applicability_scope": plan["applicability_scope"],
            },
        )
        axis_id = created.get("id")
        axis_id_sha256 = _safe_uuid_sha256(axis_id)
        version = created.get("version")
        proposition_sha256 = created.get("positive_proposition_sha256")
        scope_sha256 = created.get("applicability_scope_sha256")
        expected_definition_sha256 = _sha256_text(plan["definition"])
        expected_proposition_sha256 = _sha256_text(
            plan["positive_proposition"]
        )
        expected_scope_sha256 = _sha256_text(plan["applicability_scope"])
        response_valid = (
            axis_id_sha256 is not None
            and created.get("project_id") == project_id
            and created.get("trait_type") == trait_type
            and type(version) is int
            and version == 1
            and created.get("display_name") == plan["display_name"]
            and created.get("definition") == plan["definition"]
            and created.get("definition_sha256") == expected_definition_sha256
            and created.get("positive_proposition")
            == plan["positive_proposition"]
            and proposition_sha256 == expected_proposition_sha256
            and created.get("comparison_key") == validated_key
            and created.get("applicability_scope")
            == plan["applicability_scope"]
            and scope_sha256 == expected_scope_sha256
            and isinstance(
                created.get("positive_proposition_authored_at"), str
            )
            and bool(created.get("positive_proposition_authored_at"))
        )
        if not response_valid:
            raise _author_axis_failure(
                "author_axis_response_invalid",
                case_id=case_id,
                candidate_id_sha256=candidate_digest,
                contract_check="created_axis_snapshot",
            )
        bindings[candidate_id] = {
            # Raw identity and object key are transient request state only.
            # They are never copied into the returned acceptance report.
            "_axis_id": axis_id,
            "_comparison_key": validated_key,
            "axis_alignment": plan["axis_alignment"],
            "axis_id_sha256": axis_id_sha256,
            "axis_version": version,
            "axis_positive_proposition_sha256": proposition_sha256,
            "axis_applicability_scope_sha256": scope_sha256,
        }
    return bindings


def _verify_bound_candidate_decision(
    response: dict[str, Any],
    *,
    binding: dict[str, Any],
    candidate: dict[str, Any],
    case_id: str,
    candidate_id_sha256: str,
) -> None:
    reviewed = response.get("candidate")
    polarity = candidate.get("polarity")
    expected_axis_polarity = polarity
    if binding["axis_alignment"] == "opposite":
        expected_axis_polarity = {
            "positive": "negative", "negative": "positive"
        }.get(polarity)
    if not (
        isinstance(reviewed, dict)
        and reviewed.get("id") == candidate.get("id")
        and reviewed.get("character_key") == candidate.get("character_key")
        and reviewed.get("review_state") == "confirmed"
        and reviewed.get("approved_axis_id") == binding["_axis_id"]
        and type(reviewed.get("approved_axis_version")) is int
        and reviewed.get("approved_axis_version") == binding["axis_version"]
        and reviewed.get("axis_alignment") == binding["axis_alignment"]
        and reviewed.get("axis_polarity") == expected_axis_polarity
        and reviewed.get("axis_positive_proposition_sha256")
        == binding["axis_positive_proposition_sha256"]
        and reviewed.get("comparison_key") == binding["_comparison_key"]
        and reviewed.get("trait_type") == candidate.get("trait_type")
    ):
        raise _author_axis_failure(
            "author_axis_decision_response_invalid",
            case_id=case_id,
            candidate_id_sha256=candidate_id_sha256,
            contract_check="confirmed_axis_binding",
        )


def _review_explicit_candidates(
    client: httpx.Client,
    project_id: str,
    selectors: list[dict[str, Any]],
    *,
    author_axes: list[dict[str, Any]] | None = None,
    diagnostic_source_anchored: bool = False,
) -> dict[str, Any]:
    if author_axes is None:
        author_axes = []
    if author_axes:
        selector_map = {
            row["case_id"]: row
            for row in selectors
            if isinstance(row, dict) and isinstance(row.get("case_id"), str)
        }
        author_axes = _validated_author_axis_plans(
            author_axes,
            selectors_by_case=selector_map,
            expected_by_case={
                case_id: {"dimension": row.get("trait_type")}
                for case_id, row in selector_map.items()
            },
        )
    roster = _request(
        client,
        "GET",
        f"/api/v1/projects/{project_id}/characters",
        params={"page": 1, "page_size": 100},
    )
    items = roster.get("items")
    if not isinstance(items, list):
        raise AcceptanceFailure(
            "character_roster_invalid",
            safe_payload={
                "code": "character_roster_invalid",
                "stage": "baseline_candidate_review",
                "details": {},
            },
        )
    candidates_by_id: dict[str, dict[str, Any]] = {}
    characters: list[str] = []
    for summary in items:
        if not isinstance(summary, dict):
            continue
        character_key = summary.get("character_key") or summary.get("id")
        if not isinstance(character_key, str):
            continue
        characters.append(character_key)
        page = _request(
            client,
            "GET",
            (
                f"/api/v1/projects/{project_id}/characters/"
                f"{quote(character_key, safe='')}/profile-candidates"
            ),
            params={"state": "pending", "limit": 200, "offset": 0},
        )
        candidates = page.get("items")
        if not isinstance(candidates, list):
            raise AcceptanceFailure(
                "candidate_page_invalid",
                safe_payload={
                    "code": "candidate_page_invalid",
                    "stage": "baseline_candidate_review",
                    "details": {
                        "character_key_sha256": _sha256_text(character_key),
                    },
                },
            )
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_id = candidate.get("id")
            if (
                candidate.get("origin") == "explicit_setting"
                and candidate.get("reviewable") is True
                and isinstance(candidate_id, str)
            ):
                candidates_by_id[candidate_id] = candidate

    selected: dict[str, tuple[str, dict[str, Any]]] = {}
    relaxed_identities: list[dict[str, Any]] = []
    used_candidate_ids: set[str] = set()
    for selector in selectors:
        case_id = selector.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise AcceptanceFailure(
                "oracle_selector_invalid",
                safe_payload={
                    "code": "oracle_selector_invalid",
                    "stage": "baseline_candidate_review",
                    "details": {},
                },
            )
        matches = [
            candidate
            for candidate in candidates_by_id.values()
            if (
                _matches_source_anchored_selector(candidate, selector)
                if diagnostic_source_anchored
                else _matches_selector(candidate, selector)
            )
        ]
        if len(matches) != 1:
            raise AcceptanceFailure(
                (
                    "diagnostic_source_anchor_match_count"
                    if diagnostic_source_anchored
                    else "oracle_selector_match_count"
                ),
                safe_payload={
                    "code": (
                        "diagnostic_source_anchor_match_count"
                        if diagnostic_source_anchored
                        else "oracle_selector_match_count"
                    ),
                    "stage": "baseline_candidate_review",
                    "details": {
                        "case_id": case_id,
                        "matched_candidates": len(matches),
                        "expected_candidates": 1,
                    },
                },
            )
        candidate_id = str(matches[0]["id"])
        if candidate_id in used_candidate_ids:
            raise AcceptanceFailure(
                "oracle_candidate_reused",
                safe_payload={
                    "code": "oracle_candidate_reused",
                    "stage": "baseline_candidate_review",
                    "details": {"case_id": case_id},
                },
            )
        used_candidate_ids.add(candidate_id)
        selected[candidate_id] = (case_id, matches[0])
        if diagnostic_source_anchored:
            relaxed_identities.append({
                "case_id": case_id,
                "candidate_id_sha256": _sha256_text(candidate_id),
                "character_key": selector.get("character_key"),
                "trait_type": selector.get("trait_type"),
                "trait_key_sha256": _sha256_text(str(matches[0].get("trait_key") or "")),
                "polarity": (
                    matches[0]["polarity"]
                    if matches[0].get("polarity") in {"positive", "negative", "neutral"}
                    else None
                ),
                "stability": (
                    matches[0]["stability"]
                    if matches[0].get("stability") in {"core", "stable", "temporary"}
                    else None
                ),
                "source_document": selector.get("source_document"),
                "source_line": selector.get("source_line"),
                "strict_selector_match": _matches_selector(matches[0], selector),
            })

    for candidate in candidates_by_id.values():
        revision = candidate.get("revision")
        character_key = candidate.get("character_key")
        if type(revision) is not int or not isinstance(character_key, str):
            raise AcceptanceFailure(
                "candidate_contract_invalid",
                safe_payload={
                    "code": "candidate_contract_invalid",
                    "stage": "baseline_candidate_review",
                    "details": {},
                },
            )

    selected_traits: dict[str, dict[str, Any]] = {}
    for candidate_id, (case_id, candidate) in selected.items():
        trait_key = candidate.get("trait_key")
        dimension = candidate.get("trait_type")
        key_object = candidate.get("key_object")
        candidate_id_sha256 = _candidate_id_sha256(candidate_id)
        if (
            not isinstance(trait_key, str)
            or not trait_key
            or not isinstance(dimension, str)
            or not dimension
            or candidate_id_sha256 is None
        ):
            raise AcceptanceFailure(
                "candidate_trait_identity_invalid",
                safe_payload={
                    "code": "candidate_trait_identity_invalid",
                    "stage": "baseline_candidate_review",
                    "details": {"case_id": case_id},
                },
            )
        selected_traits[candidate_id] = {
            "case_id": case_id,
            "trait_key": trait_key,
            "dimension": dimension,
            "key_object": key_object if isinstance(key_object, str) else "",
            "candidate_id_sha256": candidate_id_sha256,
        }

    # This must finish for every planned axis before the first candidate
    # decision is sent. A failed creation or snapshot check therefore leaves
    # every candidate pending and makes the trial fail closed.
    axis_bindings = _create_and_verify_author_axes(
        client, project_id, selected, author_axes
    )

    confirmed = rejected = 0
    case_traits: dict[str, dict[str, Any]] = {}
    for candidate_id, candidate in sorted(candidates_by_id.items()):
        revision = candidate.get("revision")
        character_key = candidate.get("character_key")
        chosen = selected.get(candidate_id)
        decision = "confirm" if chosen is not None else "reject"
        decision_payload: dict[str, Any] = {
            "decision": decision,
            "expected_revision": revision,
            "comment": (
                (
                    "原创 DEV 诊断：仅按设定来源锚点模拟确认，非正式验收"
                    if diagnostic_source_anchored
                    else "原创 DEV 演示集：命中人工编写的确认清单"
                )
                if chosen is not None
                else "原创 DEV 演示集：未命中人工确认清单，自动拒绝"
            ),
        }
        binding = axis_bindings.get(candidate_id)
        if binding is not None:
            decision_payload.update({
                "approved_axis_id": binding["_axis_id"],
                "expected_axis_version": binding["axis_version"],
                "axis_alignment": binding["axis_alignment"],
                "expected_axis_positive_proposition_sha256": (
                    binding["axis_positive_proposition_sha256"]
                ),
                "expected_axis_applicability_scope_sha256": (
                    binding["axis_applicability_scope_sha256"]
                ),
                "scope_applicability_confirmed": True,
            })
        decision_response = _request(
            client,
            "POST",
            (
                f"/api/v1/projects/{project_id}/characters/"
                f"{quote(character_key, safe='')}/profile-candidates/"
                f"{quote(candidate_id, safe='')}/decisions"
            ),
            headers={"Idempotency-Key": f"{decision}-{candidate_id}"},
            json=decision_payload,
        )
        if chosen is None:
            rejected += 1
            continue
        metadata = selected_traits[candidate_id]
        case_id = metadata["case_id"]
        dimension = metadata["dimension"]
        candidate_id_sha256 = metadata["candidate_id_sha256"]
        if binding is not None:
            _verify_bound_candidate_decision(
                decision_response,
                binding=binding,
                candidate=candidate,
                case_id=case_id,
                candidate_id_sha256=candidate_id_sha256,
            )
            comparison_identity = {
                "dimension": dimension,
                "comparison_key_sha256": _sha256_text(
                    binding["_comparison_key"]
                ),
            }
            axis_metadata: dict[str, Any] = {
                "axis_bound": True,
                "axis_id_sha256": binding["axis_id_sha256"],
                "axis_version": binding["axis_version"],
                # The worker trace identifies an approved axis by its
                # server-owned UUID/version, while ``comparison_key_sha256``
                # above intentionally retains the author-authored semantic
                # comparison key.  Freeze both identities at the verified
                # creation/decision boundary; a later trace must not be able
                # to nominate its own expected identity.
                "approved_axis_runtime_comparison_key_sha256": _sha256_text(
                    f"approved_axis:{binding['_axis_id']}:"
                    f"{binding['axis_version']}"
                ),
                "axis_alignment": binding["axis_alignment"],
                "axis_positive_proposition_sha256": (
                    binding["axis_positive_proposition_sha256"]
                ),
                "axis_applicability_scope_sha256": (
                    binding["axis_applicability_scope_sha256"]
                ),
                "axis_creation_status": "created",
                "axis_decision_status": "confirmed_bound",
            }
        else:
            comparison_identity = _comparison_identity(
                dimension=dimension,
                trait_key=metadata["trait_key"],
                key_object=metadata["key_object"],
            )
            axis_metadata = {
                "axis_bound": False,
                "axis_creation_status": "not_requested",
                "axis_decision_status": "confirmed_unbound",
            }
        case_traits[case_id] = {
            **comparison_identity,
            "confirmed_candidate_id_sha256": candidate_id_sha256,
            **axis_metadata,
        }
        confirmed += 1
    review = {
        "confirmed": confirmed,
        "rejected": rejected,
        "expected_confirmed": len(selectors),
        "explicit_reviewable_candidate_count": len(candidates_by_id),
        "author_axis_plan_count": len(author_axes),
        "author_axes_created": len(axis_bindings),
        "author_axis_operation_status": (
            "created_and_bound" if axis_bindings else "not_requested"
        ),
        "recognized_characters": sorted(set(characters)),
        "case_traits": case_traits,
    }
    if diagnostic_source_anchored:
        review["diagnostic_relaxed_candidate_identities"] = relaxed_identities
    return review


def _safe_trace_observation_refs(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    refs: list[dict[str, Any]] = []
    for row in value[:12]:
        if not isinstance(row, dict):
            continue
        name = row.get("document_name")
        line_start = row.get("line_start")
        line_end = row.get("line_end")
        kind = row.get("observation_kind")
        polarity = row.get("polarity")
        key_object_sha256 = row.get("key_object_sha256")
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 128
            or any(ord(char) < 32 or char in "/\\" for char in name)
            or _TRACE_SENSITIVE_LABEL.search(name)
            or type(line_start) is not int
            or type(line_end) is not int
            or not 1 <= line_start <= line_end <= 10_000_000
            or not isinstance(kind, str)
            or kind not in _TRACE_OBSERVATION_KINDS
            or not isinstance(polarity, str)
            or polarity not in _TRACE_POLARITIES
            or not (
                key_object_sha256 is None
                or (
                    isinstance(key_object_sha256, str)
                    and re.fullmatch(r"[0-9a-f]{64}", key_object_sha256)
                )
            )
        ):
            continue
        refs.append({
            "document_name": name,
            "line_start": line_start,
            "line_end": line_end,
            "observation_kind": kind,
            "polarity": polarity,
            "key_object_sha256": key_object_sha256,
        })
    return refs


def _safe_case_trace_summary(row: dict[str, Any]) -> dict[str, Any]:
    roles = row.get("citation_roles")
    candidate_id_sha256 = row.get("confirmed_candidate_id_sha256")
    if not _valid_sha256(candidate_id_sha256):
        # A present but invalid binding is not a pre-ID legacy trace.
        candidate_id_sha256 = (
            "" if "confirmed_candidate_id_sha256" in row else None
        )
    scoped_review = _safe_scoped_axis_review(row.get("scoped_axis_review"))
    return {
        "character_key": row.get("character_key"),
        "dimension": row.get("dimension"),
        "comparison_key_sha256": _trace_comparison_sha256(row),
        "confirmed_candidate_id_sha256": candidate_id_sha256,
        "matched_observation_count": row.get("matched_observation_count"),
        "matched_observation_refs": _safe_trace_observation_refs(
            row.get("matched_observation_refs")
        ),
        "matched_observation_refs_truncated": (
            row.get("matched_observation_refs_truncated") is True
        ),
        "prepare_reason": row.get("prepare_reason"),
        "material_coverage": row.get("material_coverage"),
        "explanation_coverage": row.get("explanation_coverage"),
        "review_outcome": row.get("review_outcome"),
        "review_verdict": row.get("review_verdict"),
        "citation_roles": (
            sorted(value for value in roles if value in {"B", "C", "G", "X"})
            if isinstance(roles, list)
            else []
        ),
        "final_outcome": row.get("final_outcome"),
        "visible": row.get("visible"),
        "promote_reason": row.get("promote_reason"),
        **({"scoped_axis_review": scoped_review} if scoped_review is not None else {}),
    }


def _safe_scoped_axis_review(value: object) -> dict[str, Any] | None:
    """Retain only reviewed C handles and fixed enums, never source/model text."""

    if not isinstance(value, dict) or set(value) != {
        "observations", "independent_events"
    }:
        return None
    rows = value.get("observations")
    independence = value.get("independent_events")
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= 8
        or independence not in {"yes", "no", "unclear", "not_applicable"}
    ):
        return None
    safe_rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "citation", "object_match", "situation_match"
        }:
            return None
        citation = row.get("citation")
        object_match = row.get("object_match")
        situation_match = row.get("situation_match")
        if (
            not isinstance(citation, str)
            or re.fullmatch(r"C[0-9]{2}", citation) is None
            or citation in seen
            or object_match not in {"same", "different", "unclear"}
            or situation_match not in {"same", "different", "unclear"}
        ):
            return None
        seen.add(citation)
        safe_rows.append({
            "citation": citation,
            "object_match": object_match,
            "situation_match": situation_match,
        })
    return {"observations": safe_rows, "independent_events": independence}


def _safe_evidence_refs(evidence: object) -> list[dict[str, Any]]:
    if not isinstance(evidence, list):
        return []
    refs: set[tuple[str, int, int]] = set()
    for row in evidence:
        if not isinstance(row, dict):
            continue
        document_name = row.get("document_name")
        line_start = row.get("line_start")
        line_end = row.get("line_end")
        if (
            not isinstance(document_name, str)
            or not document_name
            or len(document_name) > 255
            or any(ord(character) < 32 for character in document_name)
            or type(line_start) is not int
            or type(line_end) is not int
            or line_start < 1
            or line_end < line_start
        ):
            continue
        refs.add((document_name, line_start, line_end))
    return [
        {
            "document_name": document_name,
            "line_start": line_start,
            "line_end": line_end,
        }
        for document_name, line_start, line_end in sorted(refs)
    ]


def _safe_visible_issue(
    issue: dict[str, Any], *, case_trace: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    metadata = issue.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    dimension = metadata.get("dimension")
    trait_key = metadata.get("trait_key")
    key_object = metadata.get("key_object")
    character_key = metadata.get("character_key")
    candidate_id_sha256 = _candidate_id_sha256(
        metadata.get("confirmed_candidate_id")
    )
    comparison_key_sha256 = None
    legacy_comparison_sha256 = None
    if (
        isinstance(dimension, str)
        and dimension
        and isinstance(trait_key, str)
        and trait_key
        and (
            dimension not in {"relationship_attitude", "motivation_goal"}
            or (isinstance(key_object, str) and bool(key_object.strip()))
        )
    ):
        legacy_comparison_sha256 = _comparison_identity(
            dimension=dimension,
            trait_key=trait_key,
            key_object=(key_object if isinstance(key_object, str) else ""),
        )["comparison_key_sha256"]
    if (
        case_trace is not None
        and isinstance(character_key, str)
        and isinstance(dimension, str)
    ):
        matched_trace = _trace_for_candidate(
            case_trace,
            candidate_id_sha256=candidate_id_sha256,
            character_key=character_key,
            dimension=dimension,
            legacy_comparison_sha256=legacy_comparison_sha256,
        )
        if matched_trace is not None:
            comparison_key_sha256 = matched_trace["comparison_key_sha256"]
    if case_trace is None or (
        comparison_key_sha256 is None
        and candidate_id_sha256 is None
        and dimension not in _OBJECT_BEARING_TRAIT_DIMENSIONS
    ):
        comparison_key_sha256 = legacy_comparison_sha256
    return {
        "character_key": character_key,
        "dimension": dimension,
        "comparison_key_sha256": comparison_key_sha256,
        "confirmed_candidate_id_sha256": candidate_id_sha256,
        "subtype": metadata.get("subtype"),
        "judgement": metadata.get("judgement"),
        "evidence_refs": _safe_evidence_refs(issue.get("evidence")),
    }


def _safe_provisional_clue(value: object) -> dict[str, Any] | None:
    """Retain only source coordinates and bounded enums from an unverified clue."""

    if not isinstance(value, dict):
        return None
    clue_id = value.get("id")
    document_name = value.get("document_name")
    line_start = value.get("line_start")
    line_end = value.get("line_end")
    dimension = value.get("dimension")
    if (
        not isinstance(clue_id, str)
        or re.fullmatch(r"pc_[a-f0-9]{32}", clue_id) is None
        or not isinstance(document_name, str)
        or not document_name
        or len(document_name) > 255
        or any(ord(char) < 32 or char in "/\\" for char in document_name)
        or _TRACE_SENSITIVE_LABEL.search(document_name)
        or type(line_start) is not int
        or type(line_end) is not int
        or not 1 <= line_start <= line_end <= 10_000_000
        or dimension not in {
            "core_personality", "preference", "value",
            "relationship_attitude", "motivation_goal", "speech_pattern",
            "behavior_boundary", "contextual_behavior", "current_state",
        }
        or value.get("reason") != "partial_model_package"
    ):
        return None
    return {
        "id_sha256": _sha256_text(clue_id),
        "document_name": document_name,
        "line_start": line_start,
        "line_end": line_end,
        "dimension": dimension,
        "reason": "partial_model_package",
    }


def _safe_token_admission_events(value: object) -> list[dict[str, int | str | None]]:
    """Copy only bounded numeric admission facts; never copy model or source text."""

    if not isinstance(value, list):
        return []
    safe: list[dict[str, int | str | None]] = []
    for row in value[:24]:
        if not isinstance(row, dict) or set(row) != set(_TOKEN_ADMISSION_FIELDS):
            continue
        stage_phase = row["stage_phase"]
        signal_phase = row["signal_phase"]
        chunk_ordinal = row["chunk_ordinal"]
        target_ordinal = row["target_ordinal"]
        numeric_limits = {
            "estimated_tokens": 1_000_000,
            "available_tokens": 1_000_000,
            # Bound this by the Settings ceiling rather than the deployment's
            # current default. Smaller configured values remain valid while
            # impossible values are discarded from the safe artifact.
            "stage_remaining_before": 500_000,
            "reviewer_reserve_tokens": 8_000,
            "model_calls_before_failure": 2,
        }
        if (
            not isinstance(stage_phase, str)
            or stage_phase not in {
                "primary_extraction", "targeted_recall", "targeted_verification"
            }
            or not isinstance(signal_phase, str)
            or signal_phase not in {"initial", "regeneration"}
            or type(chunk_ordinal) is not int
            or not 1 <= chunk_ordinal <= 128
            or (
                target_ordinal is not None
                and (type(target_ordinal) is not int or not 1 <= target_ordinal <= 12)
            )
            or (stage_phase == "primary_extraction") != (target_ordinal is None)
            or any(
                type(row[key]) is not int or not 0 <= row[key] <= limit
                for key, limit in numeric_limits.items()
            )
            or row["estimated_tokens"] <= row["available_tokens"]
        ):
            continue
        safe.append({key: row[key] for key in _TOKEN_ADMISSION_FIELDS})
    return safe


def _run_summary(
    client: httpx.Client,
    project_id: str,
    run: dict[str, Any],
) -> dict[str, Any]:
    run_id = str(run["id"])
    diagnostics = _request(
        client, "GET", f"/api/v1/analysis-runs/{run_id}/diagnostics"
    )
    worker_runtime_provenance = _safe_runtime_provenance(
        diagnostics.get("runtime_provenance")
    )
    stage = diagnostics.get("character_consistency")
    if not isinstance(stage, dict):
        stage = {}
    counts = stage.get("counts")
    counts = counts if isinstance(counts, dict) else {}
    reason_counts = stage.get("reason_counts")
    reason_counts = reason_counts if isinstance(reason_counts, dict) else {}
    usage = stage.get("usage")
    usage = usage if isinstance(usage, dict) else {}
    token_admission_events = _safe_token_admission_events(
        stage.get("token_admission_events")
    )
    case_trace = stage.get("case_trace")
    case_trace = (
        [_safe_case_trace_summary(row) for row in case_trace if isinstance(row, dict)]
        if isinstance(case_trace, list)
        else []
    )
    formal_items = _request_list(
        client, "GET", f"/api/v1/analysis-runs/{run_id}/issues"
    )
    review_page = _request(
        client, "GET", f"/api/v1/analysis-runs/{run_id}/review-clues"
    )
    provisional_page = _request(
        client, "GET", f"/api/v1/analysis-runs/{run_id}/provisional-clues"
    )
    review_items = review_page.get("items")
    provisional_items = provisional_page.get("items")
    if not isinstance(review_items, list):
        raise RuntimeError("review clue page is missing items")
    if not isinstance(provisional_items, list):
        raise RuntimeError("provisional clue page is missing items")
    signatures: list[str] = []
    contradiction_characters: set[str] = set()
    formal_issue_cases: list[dict[str, Any]] = []
    review_clue_cases: list[dict[str, Any]] = []
    report_class_integrity = True
    for issue in formal_items:
        if (
            not isinstance(issue, dict)
            or issue.get("report_class") != "formal"
        ):
            report_class_integrity = False
            continue
        if issue.get("category") != "character_drift":
            continue
        safe_issue = _safe_visible_issue(issue, case_trace=case_trace)
        signatures.append(
            json.dumps(safe_issue, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        formal_issue_cases.append(safe_issue)
        if safe_issue["judgement"] == "contradicts" and isinstance(
            safe_issue["character_key"], str
        ):
            contradiction_characters.add(safe_issue["character_key"])
    for clue in review_items:
        if (
            not isinstance(clue, dict)
            or clue.get("report_class") != "review_clue"
            or clue.get("category") != "character_drift"
        ):
            report_class_integrity = False
            continue
        review_clue_cases.append(
            _safe_visible_issue(clue, case_trace=case_trace)
        )
    safe_provisional = [
        safe
        for item in provisional_items
        for safe in [_safe_provisional_clue(item)]
        if safe is not None
    ]
    if len(safe_provisional) != len(provisional_items):
        report_class_integrity = False
    review_unavailable_count = review_page.get("unavailable_count")
    review_collection_complete = (
        type(review_unavailable_count) is int
        and review_unavailable_count == 0
        and review_page.get("scan_limited") is False
        and review_page.get("truncated") is False
    )
    provisional_collection_complete = (
        provisional_page.get("truncated") is False
        and len(safe_provisional) == len(provisional_items)
    )
    return {
        "run_id": run_id,
        "status": run.get("status"),
        # Strictly validated, content-free projection only. Invalid worker
        # payloads become ``None`` and fail the Alpha API/worker identity gate;
        # the raw diagnostics object is never copied into the report.
        "runtime_provenance": worker_runtime_provenance,
        "elapsed_seconds": run.get("_acceptance_elapsed_seconds"),
        "prompt_tokens": run.get("prompt_tokens"),
        "completion_tokens": run.get("completion_tokens"),
        "stage_outcome": stage.get("outcome"),
        "stage_reason": stage.get("reason_code"),
        "material_coverage": stage.get("material_coverage"),
        "planned_chunks": counts.get("planned_chunks"),
        "processed_chunks": counts.get("processed_chunks"),
        "model_called_chunks": counts.get("model_called_chunks"),
        "model_completed_chunks": counts.get("model_completed_chunks"),
        "model_uncalled_chunks": counts.get("model_uncalled_chunks"),
        "model_incomplete_chunks": counts.get("model_incomplete_chunks"),
        "case_material_complete_count": counts.get(
            "case_material_complete_count"
        ),
        "case_material_partial_count": counts.get(
            "case_material_partial_count"
        ),
        "explanation_review_required_case_count": counts.get(
            "explanation_review_required_case_count"
        ),
        "explanation_review_complete_case_count": counts.get(
            "explanation_review_complete_case_count"
        ),
        "explanation_coverage": stage.get("explanation_coverage"),
        "signals": counts.get("signal_count"),
        "draft_observations": counts.get("draft_observation_count"),
        "drift_considered": counts.get("drift_considered"),
        "reason_counts": reason_counts,
        "stage_usage": usage,
        "token_admission_events": token_admission_events,
        "case_trace": case_trace,
        "issues": sorted(signatures),
        # Keep the old key as a report-schema compatibility alias.  It now
        # means formal character issues only; review/provisional clues are
        # always separate collections below.
        "visible_issue_cases": sorted(
            formal_issue_cases,
            key=lambda row: (
                str(row["character_key"]),
                str(row["dimension"]),
                str(row["comparison_key_sha256"]),
            ),
        ),
        "formal_issue_cases": sorted(
            formal_issue_cases,
            key=lambda row: (
                str(row["character_key"]),
                str(row["dimension"]),
                str(row["comparison_key_sha256"]),
            ),
        ),
        "review_clue_cases": sorted(
            review_clue_cases,
            key=lambda row: (
                str(row["character_key"]),
                str(row["dimension"]),
                str(row["comparison_key_sha256"]),
            ),
        ),
        "provisional_clue_cases": safe_provisional,
        "formal_character_issue_count": len(formal_issue_cases),
        "review_clue_count": len(review_clue_cases),
        "provisional_clue_count": len(safe_provisional),
        "report_class_integrity": report_class_integrity,
        "review_clue_collection_complete": review_collection_complete,
        "provisional_clue_collection_complete": provisional_collection_complete,
        "contradiction_characters": sorted(contradiction_characters),
    }


def _runtime_identity(
    expected: dict[str, Any],
    case_traits: dict[str, dict[str, Any]],
    trace: list[dict[str, Any]],
) -> tuple[str, str, str] | None:
    case_id = expected.get("case_id")
    character_key = expected.get("character_key")
    dimension = expected.get("dimension")
    if (
        not isinstance(case_id, str)
        or not isinstance(character_key, str)
        or not isinstance(dimension, str)
    ):
        return None
    runtime = case_traits.get(case_id)
    if not isinstance(runtime, dict) or runtime.get("dimension") != dimension:
        return None
    comparison_hash = runtime.get("comparison_key_sha256")
    candidate_id_sha256 = runtime.get("confirmed_candidate_id_sha256")
    if candidate_id_sha256 is not None:
        if not _valid_sha256(candidate_id_sha256):
            return None
        matched_trace = _trace_for_candidate(
            trace,
            candidate_id_sha256=candidate_id_sha256,
            character_key=character_key,
            dimension=dimension,
            legacy_comparison_sha256=comparison_hash,
        )
        if matched_trace is None:
            return None
        if runtime.get("axis_bound") is True:
            approved_axis_hash = _verified_approved_axis_runtime_hash(runtime)
            if (
                not _valid_sha256(comparison_hash)
                or approved_axis_hash is None
                or matched_trace["comparison_key_sha256"] != approved_axis_hash
            ):
                return None
        comparison_hash = matched_trace["comparison_key_sha256"]
    if not _valid_sha256(comparison_hash):
        return None
    return character_key, dimension, comparison_hash


def _verified_approved_axis_runtime_hash(
    runtime: dict[str, Any],
) -> str | None:
    """Return the pre-bound worker identity for one fully verified axis.

    The semantic comparison key remains useful candidate-review provenance,
    but it is not the key emitted by an approved-axis case trace.  Require the
    complete safe metadata created after axis creation and candidate decision
    verification before accepting the separately frozen runtime key.
    """

    runtime_hash = runtime.get("approved_axis_runtime_comparison_key_sha256")
    if not (
        runtime.get("axis_bound") is True
        and runtime.get("axis_creation_status") == "created"
        and runtime.get("axis_decision_status") == "confirmed_bound"
        and _valid_sha256(runtime.get("axis_id_sha256"))
        and type(runtime.get("axis_version")) is int
        and runtime["axis_version"] >= 1
        and runtime.get("axis_alignment") in {"same", "opposite"}
        and _valid_sha256(runtime.get("axis_positive_proposition_sha256"))
        and _valid_sha256(runtime.get("axis_applicability_scope_sha256"))
        and _valid_sha256(runtime_hash)
    ):
        return None
    return runtime_hash


def _trace_identity(trace: dict[str, Any]) -> tuple[str, str, str] | None:
    character_key = trace.get("character_key")
    dimension = trace.get("dimension")
    comparison_hash = trace.get("comparison_key_sha256")
    if not all(isinstance(value, str) and value for value in (
        character_key,
        dimension,
        comparison_hash,
    )):
        return None
    return character_key, dimension, comparison_hash


def _case_matches_expected(
    trace: dict[str, Any],
    expected: dict[str, Any],
    *,
    runtime_identity: tuple[str, str, str],
) -> bool:
    if (
        _trace_identity(trace) != runtime_identity
    ):
        return False
    # A semantic verdict is not an assessed Alpha result when either the
    # relevant material or its explanation search was incomplete.
    if (
        trace.get("material_coverage") != "complete"
        or trace.get("explanation_coverage") != "complete"
    ):
        return False
    prepare_reason = trace.get("prepare_reason")
    if "prepare_reason" in expected:
        if prepare_reason != expected.get("prepare_reason"):
            return False
    else:
        allowed_prepare_reasons = expected.get("prepare_reason_any_of")
        if (
            not isinstance(allowed_prepare_reasons, list)
            or prepare_reason not in allowed_prepare_reasons
        ):
            return False
    for field in (
        "review_outcome",
        "review_verdict",
        "final_outcome",
        "promote_reason",
        "visible",
    ):
        if field in expected and trace.get(field) != expected[field]:
            return False
    expected_roles = expected.get("expected_citation_roles", [])
    actual_roles = trace.get("citation_roles")
    if not (
        isinstance(expected_roles, list)
        and isinstance(actual_roles, list)
        and sorted(expected_roles) == sorted(actual_roles)
    ):
        return False
    matched_count = trace.get("matched_observation_count")
    if type(matched_count) is not int:
        return False
    exact_count = expected.get("matched_observation_count")
    minimum_count = expected.get("min_matched_observation_count")
    maximum_count = expected.get("max_matched_observation_count")
    if type(exact_count) is int and matched_count != exact_count:
        return False
    if type(minimum_count) is int and matched_count < minimum_count:
        return False
    if type(maximum_count) is int and matched_count > maximum_count:
        return False
    return True


def _evidence_ref_matches(
    reference: dict[str, Any], selector: dict[str, Any]
) -> bool:
    if reference.get("document_name") != selector.get("document_name"):
        return False
    line = selector.get("line")
    if type(line) is int:
        start = reference.get("line_start")
        end = reference.get("line_end")
        return type(start) is int and type(end) is int and start <= line <= end
    expected_start = selector.get("line_start")
    expected_end = selector.get("line_end")
    return (
        type(expected_start) is int
        and type(expected_end) is int
        and reference.get("line_start") == expected_start
        and reference.get("line_end") == expected_end
    )


def _visible_issue_matches(
    issue: dict[str, Any],
    expected: dict[str, Any],
    *,
    runtime_identity: tuple[str, str, str],
    contract_field: str = "visible_issue",
) -> bool:
    issue_identity = (
        issue.get("character_key"),
        issue.get("dimension"),
        issue.get("comparison_key_sha256"),
    )
    if issue_identity != runtime_identity:
        return False
    visible_expectation = expected.get(contract_field)
    if not isinstance(visible_expectation, dict):
        return False
    if (
        issue.get("subtype") != visible_expectation.get("subtype")
        or issue.get("judgement") != visible_expectation.get("judgement")
    ):
        return False
    refs = issue.get("evidence_refs")
    if not isinstance(refs, list) or not all(isinstance(row, dict) for row in refs):
        return False
    required = visible_expectation.get("required_evidence", [])
    required_any = visible_expectation.get("required_any_evidence", [])
    allowed = visible_expectation.get("allowed_evidence")
    required_any_minimum = visible_expectation.get(
        "required_any_evidence_min_matches",
        1 if required_any else 0,
    )
    if (
        not isinstance(required, list)
        or not isinstance(required_any, list)
        or not isinstance(allowed, list)
        or not allowed
        or type(required_any_minimum) is not int
        or required_any_minimum < 0
        or required_any_minimum > len(required_any)
        or not all(isinstance(selector, dict) for selector in allowed)
    ):
        return False
    if any(
        not any(_evidence_ref_matches(ref, selector) for selector in allowed)
        for ref in refs
    ):
        return False
    if any(
        not isinstance(selector, dict)
        or not any(_evidence_ref_matches(ref, selector) for ref in refs)
        for selector in required
    ):
        return False
    if any(not isinstance(selector, dict) for selector in required_any):
        return False
    matched_any_refs = {
        (
            ref.get("document_name"),
            ref.get("line_start"),
            ref.get("line_end"),
        )
        for ref in refs
        if any(_evidence_ref_matches(ref, selector) for selector in required_any)
    }
    if len(matched_any_refs) < required_any_minimum:
        return False
    return True


def _expected_report_class(expected: dict[str, Any]) -> str:
    explicit = expected.get("expected_report_class")
    if explicit in {"formal", "review_clue", "none"}:
        return explicit
    outcome = expected.get("final_outcome")
    if outcome == "conflict":
        return "formal"
    if outcome in {"needs_confirmation", "unverifiable"} and expected.get(
        "visible"
    ) is True:
        return "review_clue"
    return "none"


def _evaluate_oracle_trial(
    trial: dict[str, Any],
    expected_cases: list[dict[str, Any]],
    case_traits: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    trace = trial.get("case_trace")
    trace = trace if isinstance(trace, list) else []
    checks: dict[str, bool] = {}
    expected_identities: list[tuple[str, str, str]] = []
    visible_checks: dict[str, bool] = {}
    review_clue_checks: dict[str, bool] = {}
    report_class_checks: dict[str, bool] = {}
    visible_issues = trial.get("formal_issue_cases")
    if not isinstance(visible_issues, list):
        visible_issues = trial.get("visible_issue_cases")
    visible_issues = visible_issues if isinstance(visible_issues, list) else []
    review_clues = trial.get("review_clue_cases")
    review_clues = review_clues if isinstance(review_clues, list) else []
    for expected in expected_cases:
        case_id = expected.get("case_id")
        if not isinstance(case_id, str):
            continue
        runtime_identity = _runtime_identity(expected, case_traits, trace)
        if runtime_identity is None:
            checks[case_id] = False
            expected_class = _expected_report_class(expected)
            if expected_class == "formal":
                visible_checks[case_id] = False
            elif expected_class == "review_clue":
                review_clue_checks[case_id] = False
            report_class_checks[case_id] = False
            continue
        candidates = [
            row
            for row in trace
            if isinstance(row, dict)
            and _trace_identity(row) == runtime_identity
        ]
        checks[case_id] = len(candidates) == 1 and _case_matches_expected(
            candidates[0], expected, runtime_identity=runtime_identity
        )
        expected_identities.append(runtime_identity)
        expected_class = _expected_report_class(expected)
        formal_candidates = [
            row
            for row in visible_issues
            if isinstance(row, dict)
            and (
                row.get("character_key"),
                row.get("dimension"),
                row.get("comparison_key_sha256"),
            )
            == runtime_identity
        ]
        clue_candidates = [
            row
            for row in review_clues
            if isinstance(row, dict)
            and (
                row.get("character_key"),
                row.get("dimension"),
                row.get("comparison_key_sha256"),
            )
            == runtime_identity
        ]
        report_class_checks[case_id] = (
            (
                expected_class == "formal"
                and len(formal_candidates) == 1
                and not clue_candidates
            )
            or (
                expected_class == "review_clue"
                and len(clue_candidates) == 1
                and not formal_candidates
            )
            or (
                expected_class == "none"
                and not formal_candidates
                and not clue_candidates
            )
        )
        if expected_class == "formal":
            visible_checks[case_id] = (
                len(formal_candidates) == 1
                and _visible_issue_matches(
                    formal_candidates[0],
                    expected,
                    runtime_identity=runtime_identity,
                )
            )
        elif expected_class == "review_clue":
            review_clue_checks[case_id] = (
                len(clue_candidates) == 1
                and _visible_issue_matches(
                    clue_candidates[0],
                    expected,
                    runtime_identity=runtime_identity,
                    contract_field="review_clue",
                )
            )

    actual_identities = [
        identity
        for row in trace
        if isinstance(row, dict)
        for identity in [_trace_identity(row)]
        if identity is not None
    ]
    valid_visible_issues = [row for row in visible_issues if isinstance(row, dict)]
    valid_review_clues = [row for row in review_clues if isinstance(row, dict)]
    return {
        "cases": checks,
        "all_cases_passed": bool(checks) and all(checks.values()),
        "trace_has_no_unexpected_cases": (
            Counter(actual_identities) == Counter(expected_identities)
            and len(actual_identities) == len(trace)
        ),
        "visible_issue_cases": visible_checks,
        "visible_issues_exact": (
            all(visible_checks.values())
            and len(valid_visible_issues) == len(visible_checks)
        ),
        "review_clue_cases": review_clue_checks,
        "review_clues_exact": (
            all(review_clue_checks.values())
            and len(valid_review_clues) == len(review_clue_checks)
        ),
        "report_classes": report_class_checks,
        "report_classes_exact": (
            bool(report_class_checks) and all(report_class_checks.values())
        ),
    }


def _dataset_hashes() -> dict[str, str]:
    filenames = [name for name, _ in BASELINE_FILES]
    filenames.extend((DRAFT_FILE, ORACLE_FILE))
    return {
        name: hashlib.sha256((DEMO / name).read_bytes()).hexdigest()
        for name in filenames
    }


def _resolve_output_json(value: str) -> Path:
    artifacts_root = (ROOT / "artifacts").resolve()
    requested = Path(value)
    if not requested.is_absolute():
        requested = (
            ROOT / requested
            if requested.parts and requested.parts[0].lower() == "artifacts"
            else artifacts_root / requested
        )
    resolved = requested.resolve()
    try:
        resolved.relative_to(artifacts_root)
    except ValueError as exc:
        raise ValueError("--output-json must stay inside the artifacts directory") from exc
    if resolved.suffix.lower() != ".json" or resolved == artifacts_root:
        raise ValueError("--output-json must name a .json file")
    return resolved


def _emit_report(report: dict[str, Any], output_json: str | None) -> None:
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if output_json:
        path = _resolve_output_json(output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as output:
            output.write(rendered + "\n")
    print(rendered)


def _safe_unexpected_failure(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, httpx.HTTPError):
        code = "http_client_error"
    elif isinstance(exc, OSError):
        code = "filesystem_error"
    elif isinstance(exc, (KeyError, TypeError)):
        code = "runner_contract_error"
    elif isinstance(exc, ValueError):
        code = "runner_value_error"
    else:
        code = "runner_runtime_error"
    return {
        "code": code,
        "stage": "runner",
        "details": {"exception_type": type(exc).__name__},
    }


def _failure_report(failure: dict[str, Any]) -> dict[str, Any]:
    try:
        hashes = _dataset_hashes()
    except (OSError, ValueError):
        hashes = {}
    return {
        "schema_version": "character-continuity-live-dev-v3",
        "dataset": _dataset_label(),
        "claims": {
            "blind_holdout": False,
            "production_quality": False,
            "open_text_generalization": False,
            "semantic_coverage": False,
        },
        "dataset_sha256": hashes,
        "failure": failure,
        "passed": False,
    }


def _emit_failure_report(
    failure: dict[str, Any], output_json: str | None
) -> dict[str, Any]:
    report = _failure_report(failure)
    try:
        _emit_report(report, output_json)
    except (OSError, ValueError):
        # An invalid/unwritable output target must not suppress the safe report
        # or cause the original exception text to leak through a traceback.
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def _stage_model_tokens_reported(summary: dict[str, Any]) -> bool:
    usage = summary.get("stage_usage")
    return (
        isinstance(usage, dict)
        and type(usage.get("attempted_calls")) is int
        and usage["attempted_calls"] > 0
        and type(usage.get("input_tokens")) is int
        and usage["input_tokens"] > 0
    )


def _all_planned_chunks_processed(summary: dict[str, Any]) -> bool:
    planned = summary.get("planned_chunks")
    completed = summary.get("model_completed_chunks")
    incomplete = summary.get("model_incomplete_chunks")
    called = summary.get("model_called_chunks")
    uncalled = summary.get("model_uncalled_chunks")
    return (
        summary.get("status") == "completed"
        and summary.get("stage_outcome") == "completed"
        and summary.get("material_coverage") == "complete"
        and type(planned) is int
        and planned > 0
        and type(completed) is int
        and completed == planned
        and type(incomplete) is int
        and incomplete == 0
        and type(called) is int
        and called == planned
        and type(uncalled) is int
        and uncalled == 0
    )


def _case_review_coverage_complete(summary: dict[str, Any]) -> bool:
    trace = summary.get("case_trace")
    complete = summary.get("case_material_complete_count")
    partial = summary.get("case_material_partial_count")
    explanation_required = summary.get("explanation_review_required_case_count")
    explanation_complete = summary.get("explanation_review_complete_case_count")
    return (
        isinstance(trace, list)
        and bool(trace)
        and type(complete) is int
        and complete == len(trace)
        and type(partial) is int
        and partial == 0
        and all(
            isinstance(row, dict)
            and row.get("material_coverage") == "complete"
            and row.get("explanation_coverage") == "complete"
            for row in trace
        )
        and summary.get("explanation_coverage") == "complete"
        and type(explanation_required) is int
        and type(explanation_complete) is int
        and explanation_required == explanation_complete == len(trace)
    )


def _baseline_admission(summary: dict[str, Any]) -> dict[str, Any]:
    usage = summary.get("stage_usage")
    usage = usage if isinstance(usage, dict) else {}
    planned = summary.get("planned_chunks")
    completed = summary.get("model_completed_chunks")
    incomplete = summary.get("model_incomplete_chunks")
    called = summary.get("model_called_chunks")
    uncalled = summary.get("model_uncalled_chunks")
    checks = {
        "run_completed": summary.get("status") == "completed",
        "character_stage_completed_without_degradation": (
            summary.get("stage_outcome") == "completed"
            and summary.get("material_coverage") == "complete"
        ),
        "all_planned_chunks_model_completed": (
            type(planned) is int
            and planned > 0
            and type(completed) is int
            and completed == planned
            and type(incomplete) is int
            and incomplete == 0
            and type(called) is int
            and called == planned
            and type(uncalled) is int
            and uncalled == 0
        ),
        "model_calls_reported": (
            type(usage.get("attempted_calls")) is int
            and usage["attempted_calls"] > 0
        ),
        "model_input_tokens_reported": (
            type(usage.get("input_tokens")) is int and usage["input_tokens"] > 0
        ),
    }
    return {
        "admitted": all(checks.values()),
        "checks": checks,
        "safe_summary": {
            "status": summary.get("status"),
            "stage_outcome": summary.get("stage_outcome"),
            "stage_reason": summary.get("stage_reason"),
            "material_coverage": summary.get("material_coverage"),
            "planned_chunks": planned,
            "processed_chunks": summary.get("processed_chunks"),
            "model_called_chunks": called,
            "model_completed_chunks": completed,
            "model_uncalled_chunks": uncalled,
            "model_incomplete_chunks": incomplete,
            "stage_usage": {
                key: usage.get(key)
                for key in (
                    "attempted_calls",
                    "input_tokens",
                    "completion_tokens",
                    "charged_tokens",
                )
            },
            "reason_counts": summary.get("reason_counts", {}),
        },
    }


def _diagnostic_partial_baseline_allowed(summary: dict[str, Any]) -> bool:
    """Permit observation only, never a passing full-workflow gate."""
    usage = summary.get("stage_usage")
    usage = usage if isinstance(usage, dict) else {}
    counts = summary.get("reason_counts")
    counts = counts if isinstance(counts, dict) else {}
    planned = summary.get("planned_chunks")
    completed = summary.get("model_completed_chunks")
    incomplete = summary.get("model_incomplete_chunks")
    return (
        summary.get("status") == "completed"
        and summary.get("stage_outcome") == "partial"
        and summary.get("material_coverage") == "partial"
        and type(planned) is int
        and planned > 0
        and completed == planned
        and incomplete == 0
        and type(usage.get("attempted_calls")) is int
        and usage["attempted_calls"] > 0
        and any(
            type(counts.get(reason)) is int and counts[reason] > 0
            for reason in DIAGNOSTIC_RECORD_REASONS
        )
        and set(counts) <= (
            DIAGNOSTIC_RECORD_REASONS
            | DIAGNOSTIC_SOURCE_COUNTS
            | DIAGNOSTIC_REGENERATION_COUNTS
        )
    )


def _candidate_review_exact(review: dict[str, Any]) -> bool:
    expected = review.get("expected_confirmed")
    return (
        type(expected) is int
        and expected > 0
        and review.get("confirmed") == expected
        and review.get("rejected") == 0
        and review.get("explicit_reviewable_candidate_count") == expected
        and isinstance(review.get("case_traits"), dict)
        and len(review["case_traits"]) == expected
    )


def _visible_issue_semantic_identity(
    issue: dict[str, Any],
) -> tuple[str, str, str, str, str] | None:
    identity = (
        issue.get("character_key"),
        issue.get("dimension"),
        issue.get("comparison_key_sha256"),
        issue.get("subtype"),
        issue.get("judgement"),
    )
    if not all(isinstance(value, str) and value for value in identity):
        return None
    if len(identity[2]) != 64:
        return None
    return identity


def _semantic_signatures(
    trials: list[dict[str, Any]],
) -> tuple[bool, list[tuple[tuple[str, str, str, str, str], ...]]]:
    """Return validated per-trial issue identities without claiming stability."""

    signatures: list[tuple[tuple[str, str, str, str, str], ...]] = []
    valid = True
    for trial in trials:
        draft = trial.get("draft", {})
        visible_issues = (
            draft.get("visible_issue_cases") if isinstance(draft, dict) else None
        )
        if not isinstance(visible_issues, list) or not visible_issues:
            valid = False
            continue
        identities = [
            _visible_issue_semantic_identity(row)
            if isinstance(row, dict)
            else None
            for row in visible_issues
        ]
        if any(identity is None for identity in identities):
            valid = False
            continue
        signatures.append(tuple(sorted(identities)))
    return valid, signatures


def _independent_project_trials(
    trials: list[dict[str, Any]], *, minimum: int
) -> bool:
    project_ids = [row.get("project_id") for row in trials]
    return (
        len(trials) >= minimum
        and all(isinstance(project_id, str) and project_id for project_id in project_ids)
        and len(set(project_ids)) == len(project_ids)
    )


def _minimum_trial_count(fixture: str) -> int:
    if fixture not in FIXTURES:
        raise ValueError("unknown character acceptance fixture")
    return 1 if fixture == _ALPHA_FIXTURE else _LEGACY_MINIMUM_TRIALS


def _alpha_worker_runtime_provenance_matches_api(
    trials: list[dict[str, Any]], api_runtime_provenance: object,
) -> bool:
    """Require every worker snapshot to equal the strict API projection."""

    safe_api = _safe_runtime_provenance(api_runtime_provenance)
    if safe_api is None or not trials:
        return False
    for trial in trials:
        for phase in ("baseline", "draft"):
            summary = trial.get(phase)
            worker = (
                _safe_runtime_provenance(summary.get("runtime_provenance"))
                if isinstance(summary, dict)
                else None
            )
            if worker is None or worker != safe_api:
                return False
    return True


def _evaluate_gates(
    trials: list[dict[str, Any]],
    *,
    fixture: str = "demo",
    api_runtime_provenance: object = None,
    post_run_api_runtime_provenance: object = None,
) -> dict[str, bool]:
    """Evaluate the required gates for the selected fixture contract.

    Alpha is a one-run functional-development fixture, so cross-run stability
    remains observational there. The older fixtures retain their historical
    three-independent-run requirement and required semantic-stability gate.
    """

    baselines = [row.get("baseline", {}) for row in trials]
    drafts = [row.get("draft", {}) for row in trials]
    reviews = [row.get("candidate_review", {}) for row in trials]
    oracle_results = [row.get("oracle", {}) for row in trials]
    semantic_signatures_valid, semantic_signatures = _semantic_signatures(trials)
    minimum_trials = _minimum_trial_count(fixture)
    independent_trials = _independent_project_trials(
        trials, minimum=minimum_trials
    )
    gates = {
        (
            "at_least_one_fresh_full_workflow_trial"
            if fixture == _ALPHA_FIXTURE
            else "at_least_three_independent_full_workflow_trials"
        ): independent_trials,
        "all_baseline_runs_completed": bool(baselines)
        and all(row.get("status") == "completed" for row in baselines),
        "baseline_character_stage_model_tokens_reported": bool(baselines)
        and all(_stage_model_tokens_reported(row) for row in baselines),
        "baseline_all_planned_chunks_model_completed_without_degradation": bool(baselines)
        and all(_all_planned_chunks_processed(row) for row in baselines),
        "baseline_explicit_candidate_set_matches_oracle_exactly": bool(reviews)
        and all(_candidate_review_exact(row) for row in reviews),
        "all_draft_runs_completed": bool(drafts)
        and all(row.get("status") == "completed" for row in drafts),
        "draft_character_stage_model_tokens_reported": bool(drafts)
        and all(_stage_model_tokens_reported(row) for row in drafts),
        "draft_all_planned_chunks_model_completed_without_degradation": bool(drafts)
        and all(_all_planned_chunks_processed(row) for row in drafts),
        "draft_case_material_and_explanation_coverage_complete": bool(drafts)
        and all(_case_review_coverage_complete(row) for row in drafts),
        "draft_report_classes_are_separate_and_complete": bool(drafts)
        and all(
            row.get("report_class_integrity") is True
            and row.get("review_clue_collection_complete") is True
            and row.get("provisional_clue_collection_complete") is True
            for row in drafts
        ),
        "draft_has_no_unverified_provisional_clues": bool(drafts)
        and all(row.get("provisional_clue_count") == 0 for row in drafts),
        "visible_issue_semantic_identity_valid_in_each_trial": bool(trials)
        and semantic_signatures_valid
        and len(semantic_signatures) == len(trials),
        "all_runtime_bound_expected_cases_passed": bool(oracle_results)
        and all(row.get("all_cases_passed") is True for row in oracle_results),
        "trace_has_no_unexpected_runtime_bound_cases": bool(oracle_results)
        and all(
            row.get("trace_has_no_unexpected_cases") is True
            for row in oracle_results
        ),
        "visible_issue_full_safe_signatures_match_oracle": bool(oracle_results)
        and all(row.get("visible_issues_exact") is True for row in oracle_results),
        "review_clue_full_safe_signatures_match_oracle": bool(oracle_results)
        and all(row.get("review_clues_exact") is True for row in oracle_results),
        "report_class_assignment_matches_oracle": bool(oracle_results)
        and all(row.get("report_classes_exact") is True for row in oracle_results),
    }
    if fixture == _ALPHA_FIXTURE:
        safe_preflight = _safe_runtime_provenance(api_runtime_provenance)
        safe_post_run = _safe_runtime_provenance(
            post_run_api_runtime_provenance
        )
        gates["api_and_worker_runtime_provenance_match"] = (
            _alpha_worker_runtime_provenance_matches_api(
                trials, api_runtime_provenance
            )
        )
        gates["post_run_api_runtime_provenance_matches_preflight"] = (
            safe_preflight is not None
            and safe_post_run is not None
            and safe_post_run == safe_preflight
        )
    else:
        gates[_STABILITY_OBSERVATION_KEY] = (
            independent_trials
            and semantic_signatures_valid
            and len(semantic_signatures) == len(trials)
            and len(set(semantic_signatures)) == 1
        )
    return gates


def _evaluate_stability_observations(
    trials: list[dict[str, Any]],
    *,
    fixture: str = "demo",
) -> dict[str, bool]:
    """Evaluate stability at the selected fixture's evidentiary threshold."""

    semantic_signatures_valid, semantic_signatures = _semantic_signatures(trials)
    observation_minimum = (
        2 if fixture == _ALPHA_FIXTURE else _LEGACY_MINIMUM_TRIALS
    )
    multiple_independent_trials = _independent_project_trials(
        trials, minimum=observation_minimum
    )
    return {
        "multiple_independent_full_workflow_trials": multiple_independent_trials,
        _STABILITY_OBSERVATION_KEY: (
            multiple_independent_trials
            and semantic_signatures_valid
            and len(semantic_signatures) == len(trials)
            and len(set(semantic_signatures)) == 1
        ),
    }


def _evaluation_claims(
    trials: list[dict[str, Any]],
    gates: dict[str, bool],
    observations: dict[str, bool],
    *,
    diagnostic_source_anchored: bool,
) -> dict[str, Any]:
    """Build explicit claims without treating one run as stability evidence."""

    multiple_trials = observations[
        "multiple_independent_full_workflow_trials"
    ]
    stable = observations[
        _STABILITY_OBSERVATION_KEY
    ]
    return {
        "blind_holdout": False,
        "production_quality": False,
        "open_text_generalization": False,
        "semantic_coverage": False,
        "diagnostic_source_anchored_candidate_review": diagnostic_source_anchored,
        # For Alpha this means at least two fresh projects. For legacy fixtures
        # it preserves the historical three-project evidentiary threshold.
        "independent_full_workflow_trials": multiple_trials,
        "full_workflow_trial_count": len(trials),
        "visible_issue_semantic_identity_valid_in_each_trial": gates[
            "visible_issue_semantic_identity_valid_in_each_trial"
        ],
        "semantic_identity_stability_assessed": multiple_trials,
        "semantic_identity_stable_across_independent_trials": stable,
    }


def _report_gates(
    required_gates: dict[str, bool], observations: dict[str, bool]
) -> dict[str, bool]:
    """Expose stability while preserving whether policy makes it required."""

    return {
        **required_gates,
        _STABILITY_OBSERVATION_KEY: observations[_STABILITY_OBSERVATION_KEY],
    }


def _execute_trial(
    client: httpx.Client,
    oracle: dict[str, Any],
    *,
    trial_number: int,
    timeout_seconds: float,
    guided_review: bool = False,
    diagnostic_continue_partial: bool = False,
    diagnostic_source_anchored: bool = False,
) -> dict[str, Any]:
    project = _request(
        client,
        "POST",
        "/api/v1/projects",
        json={
            "name": (
                f"{'浮光列车' if DEMO == FIXTURES['demo'] else DEMO.name}"
                f"·角色连续性真实验收·T{trial_number}·"
                f"{uuid4().hex[:8]}"
            ),
            "description": "原创 DEV 演示集；非盲测、非开放文本质量结论",
        },
    )
    project_id = str(project["id"])
    for filename, role in BASELINE_FILES:
        _upload(
            client,
            project_id,
            filename,
            role,
            release_key="v1.0",
            ordinal=10,
            published=True,
        )
    baseline_id = _start_run(
        client,
        project_id,
        **({"mode": "baseline_build"} if guided_review else {}),
    )
    baseline_run = _wait_run(
        client,
        baseline_id,
        timeout_seconds=timeout_seconds,
    )
    baseline = _run_summary(client, project_id, baseline_run)
    admission = _baseline_admission(baseline)
    if admission["admitted"] is not True and not (
        diagnostic_continue_partial
        and _diagnostic_partial_baseline_allowed(baseline)
    ):
        raise AcceptanceFailure(
            "baseline_admission_failed",
            safe_payload={
                "code": "baseline_admission_failed",
                "stage": "baseline_admission",
                "details": {
                    "trial": trial_number,
                    "baseline_admission": admission,
                },
            },
        )
    candidate_review = _review_explicit_candidates(
        client,
        project_id,
        oracle["confirm_candidates"],
        author_axes=oracle.get("author_axes", []),
        diagnostic_source_anchored=diagnostic_source_anchored,
    )
    if candidate_review["confirmed"] != len(oracle["confirm_candidates"]):
        raise AcceptanceFailure(
            "baseline_confirmation_count_mismatch",
            safe_payload={
                "code": "baseline_confirmation_count_mismatch",
                "stage": "baseline_candidate_review",
                "details": {
                    "trial": trial_number,
                    "confirmed": candidate_review["confirmed"],
                    "expected": len(oracle["confirm_candidates"]),
                },
            },
        )
    _upload(
        client,
        project_id,
        DRAFT_FILE,
        "chapter",
        release_key="v1.1",
        ordinal=11,
        published=False,
    )
    draft_id = _start_run(
        client,
        project_id,
        **({"mode": "draft_review"} if guided_review else {}),
    )
    draft_run = _wait_run(
        client,
        draft_id,
        timeout_seconds=timeout_seconds,
    )
    draft = _run_summary(client, project_id, draft_run)
    oracle_result = _evaluate_oracle_trial(
        draft,
        oracle["expected_cases"],
        candidate_review["case_traits"],
    )
    return {
        "trial": trial_number,
        "project_id": project_id,
        "baseline": baseline,
        "baseline_admission": admission,
        "candidate_review": candidate_review,
        "draft": draft,
        "oracle": oracle_result,
    }


def _model_available_for_session(client: httpx.Client, health: dict[str, Any]) -> bool:
    model = health.get("model")
    if isinstance(model, dict) and model.get("configured") is True:
        return True
    profile = _request(client, "GET", "/api/v1/account/model-provider")
    return profile.get("configured") is True or profile.get("service_default_available") is True


def _target_bound_actor_v2_ready(limits: object) -> bool:
    return (
        isinstance(limits, dict)
        and limits.get("draft_actor_review_v1") is True
        and limits.get("target_bound_draft_review_v2") is True
        and all(
            limits.get(field) == expected
            for field, expected in _TARGET_BOUND_DRAFT_REVIEW_V2.items()
        )
    )


def _target_bound_semantic_v3_ready(limits: object) -> bool:
    return (
        isinstance(limits, dict)
        and limits.get("draft_actor_review_v1") is True
        and limits.get("target_bound_draft_review_v2") is False
        and limits.get("target_bound_draft_review_v3") is True
        and all(
            limits.get(field) == expected
            for field, expected in _TARGET_BOUND_DRAFT_REVIEW_V3.items()
        )
    )


def _alpha_runtime_ready(provenance: dict[str, Any]) -> bool:
    """Check Alpha-only switches on an already strict-safe projection."""

    limits = provenance.get("character_consistency_limits")
    return (
        isinstance(limits, dict)
        and all(
            limits.get(field) == expected
            for field, expected in _ALPHA_EXPLANATION_REVIEW_IDENTITY.items()
        )
        and limits.get("ooc_protocol_version") == "character-ooc-v1"
        and limits.get("ooc_supported_dimensions") == list(_ALPHA_OOC_DIMENSIONS)
        # V4 alone only identifies an assertion. V5 supplies its scope
        # relation, and the V1 reviewer binds the accepted proposal to the
        # frozen run input; all three are needed for required_v1 persistence.
        and all(
            limits.get(field) == expected
            for field, expected in _ALPHA_FORMAL_SUPPORT_BINDING_IDENTITY.items()
        )
        # The versioned switch is the public runtime protocol identity for
        # scoped value/boundary review; Alpha never enables it implicitly.
        and limits.get("scoped_axis_drift_v1") is True
        and _target_bound_semantic_v3_ready(limits)
    )


def _alpha_service_artifact_binding(
    provenance: dict[str, Any], *, allow_remote_service_artifact: bool
) -> dict[str, bool]:
    """Bind Alpha to this checkout unless an explicit remote override is used."""

    build = provenance.get("build")
    live_hash = (
        build.get("service_artifact_sha256")
        if isinstance(build, dict)
        else None
    )
    local_hash = _local_service_artifact_sha256(ROOT)
    local_available = isinstance(local_hash, str)
    matches = local_available and local_hash == live_hash
    if not allow_remote_service_artifact:
        if not local_available:
            raise AcceptanceFailure(
                "local_service_artifact_unavailable",
                safe_payload={
                    "code": "local_service_artifact_unavailable",
                    "stage": "runner_preflight",
                    "details": {},
                },
            )
        if not matches:
            raise AcceptanceFailure(
                "service_artifact_mismatch",
                safe_payload={
                    "code": "service_artifact_mismatch",
                    "stage": "runner_preflight",
                    "details": {},
                },
            )
    return {
        "local_service_artifact_match_enforced": (
            not allow_remote_service_artifact
        ),
        "local_service_artifact_matches_live_api": matches,
        "remote_service_artifact_override": allow_remote_service_artifact,
    }


def run(args: argparse.Namespace) -> int:
    global DEMO
    DEMO = FIXTURES[args.fixture]
    trial_count = (
        _minimum_trial_count(args.fixture)
        if getattr(args, "trials", None) is None
        else args.trials
    )
    diagnostic_source_anchored = bool(
        getattr(args, "diagnostic_review_source_anchored_candidates", False)
    )
    allow_remote_service_artifact = bool(
        getattr(args, "allow_remote_service_artifact", False)
    )
    if allow_remote_service_artifact and args.fixture != _ALPHA_FIXTURE:
        raise AcceptanceFailure(
            "remote_service_artifact_override_scope_invalid",
            safe_payload={
                "code": "remote_service_artifact_override_scope_invalid",
                "stage": "runner_preflight",
                "details": {"allowed_fixture": _ALPHA_FIXTURE},
            },
        )
    if diagnostic_source_anchored and (
        args.fixture not in {"ooc-v1", "ooc-transfer-v1"} or trial_count != 1
    ):
        raise AcceptanceFailure(
            "diagnostic_source_anchored_scope_invalid",
            safe_payload={
                "code": "diagnostic_source_anchored_scope_invalid",
                "stage": "runner_preflight",
                "details": {"allowed_fixtures": ["ooc-v1", "ooc-transfer-v1"], "trials": 1},
            },
        )
    missing = [name for name, _ in BASELINE_FILES if not (DEMO / name).is_file()]
    if not (DEMO / DRAFT_FILE).is_file():
        missing.append(DRAFT_FILE)
    if not (DEMO / ORACLE_FILE).is_file():
        missing.append(ORACLE_FILE)
    if missing:
        raise AcceptanceFailure(
            "demo_files_missing",
            safe_payload={
                "code": "demo_files_missing",
                "stage": "runner_preflight",
                "details": {"missing_files": sorted(missing)},
            },
        )
    oracle = _load_oracle()
    with httpx.Client(
        base_url=args.base_url.rstrip("/"),
        timeout=httpx.Timeout(30.0),
    ) as client:
        health = _request(client, "GET", "/health")
        api_runtime_provenance = _safe_runtime_provenance(
            health.get("runtime_provenance")
        )
        if api_runtime_provenance is None:
            raise AcceptanceFailure(
                "runtime_provenance_invalid",
                safe_payload={
                    "code": "runtime_provenance_invalid",
                    "stage": "runner_preflight",
                    "details": {},
                },
            )
        capabilities = api_runtime_provenance["capabilities"]
        if capabilities.get("character_consistency") is not True:
            raise AcceptanceFailure(
                "character_consistency_disabled",
                safe_payload={
                    "code": "character_consistency_disabled",
                    "stage": "runner_preflight",
                    "details": {},
                },
            )
        if not _model_available_for_session(client, health):
            raise AcceptanceFailure(
                "model_not_configured",
                safe_payload={
                    "code": "model_not_configured",
                    "stage": "runner_preflight",
                    "details": {},
                },
            )
        if (
            args.fixture == _ALPHA_FIXTURE
            and not _alpha_runtime_ready(api_runtime_provenance)
        ):
            raise AcceptanceFailure(
                "alpha_ooc_capabilities_unavailable",
                safe_payload={
                    "code": "alpha_ooc_capabilities_unavailable",
                    "stage": "runner_preflight",
                    "details": {
                        "required_ooc_protocol_version": "character-ooc-v1",
                        "required_explanation_review": True,
                        "required_formal_support_binding_identity": dict(
                            _ALPHA_FORMAL_SUPPORT_BINDING_IDENTITY
                        ),
                        "required_scoped_axis_drift_v1": True,
                        "required_draft_actor_review_v1": True,
                        "required_target_bound_draft_review_v3": True,
                        "required_target_bound_protocol_versions": dict(
                            _TARGET_BOUND_DRAFT_REVIEW_V3
                        ),
                        "required_dimensions": list(_ALPHA_OOC_DIMENSIONS),
                    },
                },
            )
        artifact_binding = (
            _alpha_service_artifact_binding(
                api_runtime_provenance,
                allow_remote_service_artifact=allow_remote_service_artifact,
            )
            if args.fixture == _ALPHA_FIXTURE
            else {}
        )
        trials = [
            _execute_trial(
                client,
                oracle,
                trial_number=index,
                timeout_seconds=args.run_timeout_seconds,
                guided_review=args.fixture in {
                    "alpha-v1", "ooc-v1", "ooc-transfer-v1"
                },
                diagnostic_continue_partial=(
                    args.fixture in {"ooc-v1", "ooc-transfer-v1"}
                    and args.diagnostic_continue_partial_baseline
                ),
                diagnostic_source_anchored=diagnostic_source_anchored,
            )
            for index in range(1, trial_count + 1)
        ]
        post_run_api_runtime_provenance = None
        try:
            post_run_health = _request(client, "GET", "/health")
        except (AcceptanceFailure, httpx.HTTPError):
            # The Alpha post-run identity gate below fails closed. Do not copy
            # an unavailable response or its failure details into the report.
            pass
        else:
            post_run_api_runtime_provenance = _safe_runtime_provenance(
                post_run_health.get("runtime_provenance")
            )

    required_gates = _evaluate_gates(
        trials,
        fixture=args.fixture,
        api_runtime_provenance=api_runtime_provenance,
        post_run_api_runtime_provenance=post_run_api_runtime_provenance,
    )
    required_gates["strict_candidate_selector_identity"] = (
        not diagnostic_source_anchored
    )
    observations = _evaluate_stability_observations(
        trials, fixture=args.fixture
    )
    # Keep the historical field available for report consumers. It is
    # observational-only for Alpha, but remains a required gate for every
    # pre-existing fixture.
    reported_gates = _report_gates(required_gates, observations)
    report = {
        "schema_version": "character-continuity-live-dev-v3",
        "dataset": _dataset_label(),
        "claims": _evaluation_claims(
            trials,
            required_gates,
            observations,
            diagnostic_source_anchored=diagnostic_source_anchored,
        ) | artifact_binding,
        "gate_policy": {
            "required_for_pass": list(required_gates),
            "observational_only": (
                [_STABILITY_OBSERVATION_KEY]
                if args.fixture == _ALPHA_FIXTURE
                else []
            ),
        },
        "observations": observations,
        "runtime_provenance": api_runtime_provenance,
        "dataset_sha256": _dataset_hashes(),
        "trials": trials,
        "gates": reported_gates,
        "passed": all(required_gates.values()) and not diagnostic_source_anchored,
    }
    _emit_report(report, args.output_json)
    return 0 if report["passed"] else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", choices=tuple(FIXTURES), default="demo")
    parser.add_argument(
        "--diagnostic-continue-partial-baseline",
        action="store_true",
        help="OOC DEV only: continue after record-level partial baseline; full gate stays false",
    )
    parser.add_argument(
        "--diagnostic-review-source-anchored-candidates",
        action="store_true",
        help=(
            "one-trial OOC DEV only: simulate candidate confirmation from unique "
            "source anchors to observe draft; strict acceptance always fails"
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--allow-remote-service-artifact",
        action="store_true",
        help=(
            "Alpha only: explicitly allow a live service built from a different "
            "checkout; the override and match result are recorded in the report"
        ),
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        choices=range(1, 6),
        help="fresh workflow count (default: 1 for alpha-v1, 3 otherwise)",
    )
    parser.add_argument("--run-timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--output-json",
        help=(
            "optional UTF-8 report path inside artifacts/; a bare filename is "
            "placed there automatically"
        ),
    )
    args = parser.parse_args(argv)
    if args.trials is None:
        args.trials = _minimum_trial_count(args.fixture)
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        raise SystemExit(run(parsed_args))
    except AcceptanceFailure as exc:
        _emit_failure_report(exc.safe_payload, parsed_args.output_json)
        print(
            f"character live acceptance failed: {exc.safe_payload.get('code')}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
    except (
        httpx.HTTPError,
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
    ) as exc:
        failure = _safe_unexpected_failure(exc)
        _emit_failure_report(failure, parsed_args.output_json)
        print(
            f"character live acceptance failed: {failure['code']}",
            file=sys.stderr,
        )
        raise SystemExit(2) from None
