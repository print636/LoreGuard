"""Run independent character-continuity workflows through the real HTTP API.

This is a developer-visible acceptance runner, not a blind benchmark.  It
never reads a Provider credential: the already running API owns all model
configuration.  Output is limited to fixture labels, hashes, evidence file/line
references, counters, verdict labels and token usage; source text, credentials,
the configured endpoint and upstream responses are not printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.character_trait_extraction import stable_trait_identity


FIXTURES = {
    "demo": ROOT / "data" / "character-continuity-demo",
    "ooc-v1": ROOT / "data" / "character-ooc-challenge-v1",
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
    "source_formal", "source_history", "schema_validation", "record_validation",
})


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


def _comparison_identity(*, dimension: str, trait_key: str) -> dict[str, str]:
    """Return a report-safe binding to one runtime-confirmed trait.

    The raw model label is intentionally not copied into the acceptance report.
    The trace and the confirmed candidate are instead joined through the digest
    of the same server-owned comparison identity.
    """

    comparison_key = stable_trait_identity(dimension, trait_key)
    return {
        "dimension": dimension,
        "comparison_key_sha256": _sha256_text(comparison_key),
    }


def _trace_comparison_sha256(trace: dict[str, Any]) -> str | None:
    value = trace.get("comparison_key")
    if not isinstance(value, str) or not value:
        return None
    return _sha256_text(value)


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


def _validate_oracle_payload(payload: object) -> dict[str, Any]:
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
    for selector in selectors:
        if not isinstance(selector, dict):
            raise RuntimeError("acceptance oracle selector is invalid")
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
        if row["visible"] is True and not isinstance(row.get("visible_issue"), dict):
            raise RuntimeError("visible oracle case lacks a full issue signature")
    return payload


def _load_oracle() -> dict[str, Any]:
    payload = json.loads((DEMO / ORACLE_FILE).read_text(encoding="utf-8"))
    return _validate_oracle_payload(payload)


def _matches_selector(candidate: dict[str, Any], selector: dict[str, Any]) -> bool:
    if (
        candidate.get("character_key") != selector.get("character_key")
        or candidate.get("trait_type") != selector.get("trait_type")
        or candidate.get("origin") != "explicit_setting"
        or candidate.get("reviewable") is not True
    ):
        return False
    variants = selector.get("trait_polarity_options")
    if variants is not None and not any(
        candidate.get("trait_key") == row.get("trait_key")
        and candidate.get("polarity") == row.get("polarity")
        for row in variants
    ):
        return False
    for field in ("polarity", "stability"):
        expected_value = selector.get(field)
        if expected_value is not None and candidate.get(field) != expected_value:
            return False
    document = selector.get("source_document")
    line = selector.get("source_line")
    if not isinstance(document, str) or type(line) is not int:
        return False
    value_contains = selector.get("value_contains")
    if value_contains is not None and (
        not isinstance(value_contains, str)
        or not value_contains
        or value_contains not in str(candidate.get("value") or "")
    ):
        return False
    evidence = candidate.get("evidence")
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


def _review_explicit_candidates(
    client: httpx.Client,
    project_id: str,
    selectors: list[dict[str, Any]],
) -> dict[str, Any]:
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
            if _matches_selector(candidate, selector)
        ]
        if len(matches) != 1:
            raise AcceptanceFailure(
                "oracle_selector_match_count",
                safe_payload={
                    "code": "oracle_selector_match_count",
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

    confirmed = rejected = 0
    case_traits: dict[str, dict[str, str]] = {}
    for candidate_id, candidate in sorted(candidates_by_id.items()):
        revision = candidate.get("revision")
        character_key = candidate.get("character_key")
        if not isinstance(revision, int) or not isinstance(character_key, str):
            raise AcceptanceFailure(
                "candidate_contract_invalid",
                safe_payload={
                    "code": "candidate_contract_invalid",
                    "stage": "baseline_candidate_review",
                    "details": {},
                },
            )
        chosen = selected.get(candidate_id)
        decision = "confirm" if chosen is not None else "reject"
        _request(
            client,
            "POST",
            (
                f"/api/v1/projects/{project_id}/characters/"
                f"{quote(character_key, safe='')}/profile-candidates/"
                f"{quote(candidate_id, safe='')}/decisions"
            ),
            headers={"Idempotency-Key": f"{decision}-{candidate_id}"},
            json={
                "decision": decision,
                "expected_revision": revision,
                "comment": (
                    "原创 DEV 演示集：命中人工编写的确认清单"
                    if chosen is not None
                    else "原创 DEV 演示集：未命中人工确认清单，自动拒绝"
                ),
            },
        )
        if chosen is None:
            rejected += 1
            continue
        case_id, _ = chosen
        trait_key = candidate.get("trait_key")
        dimension = candidate.get("trait_type")
        if (
            not isinstance(trait_key, str)
            or not trait_key
            or not isinstance(dimension, str)
            or not dimension
        ):
            raise AcceptanceFailure(
                "candidate_trait_identity_invalid",
                safe_payload={
                    "code": "candidate_trait_identity_invalid",
                    "stage": "baseline_candidate_review",
                    "details": {"case_id": case_id},
                },
            )
        case_traits[case_id] = _comparison_identity(
            dimension=dimension,
            trait_key=trait_key,
        )
        confirmed += 1
    return {
        "confirmed": confirmed,
        "rejected": rejected,
        "expected_confirmed": len(selectors),
        "explicit_reviewable_candidate_count": len(candidates_by_id),
        "recognized_characters": sorted(set(characters)),
        "case_traits": case_traits,
    }


def _safe_case_trace_summary(row: dict[str, Any]) -> dict[str, Any]:
    roles = row.get("citation_roles")
    return {
        "character_key": row.get("character_key"),
        "dimension": row.get("dimension"),
        "comparison_key_sha256": _trace_comparison_sha256(row),
        "matched_observation_count": row.get("matched_observation_count"),
        "prepare_reason": row.get("prepare_reason"),
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
    }


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


def _safe_visible_issue(issue: dict[str, Any]) -> dict[str, Any]:
    metadata = issue.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    dimension = metadata.get("dimension")
    trait_key = metadata.get("trait_key")
    comparison_key_sha256 = None
    if (
        isinstance(dimension, str)
        and dimension
        and isinstance(trait_key, str)
        and trait_key
    ):
        comparison_key_sha256 = _comparison_identity(
            dimension=dimension,
            trait_key=trait_key,
        )["comparison_key_sha256"]
    return {
        "character_key": metadata.get("character_key"),
        "dimension": dimension,
        "comparison_key_sha256": comparison_key_sha256,
        "subtype": metadata.get("subtype"),
        "judgement": metadata.get("judgement"),
        "evidence_refs": _safe_evidence_refs(issue.get("evidence")),
    }


def _run_summary(
    client: httpx.Client,
    project_id: str,
    run: dict[str, Any],
) -> dict[str, Any]:
    run_id = str(run["id"])
    diagnostics = _request(
        client, "GET", f"/api/v1/analysis-runs/{run_id}/diagnostics"
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
    case_trace = stage.get("case_trace")
    case_trace = (
        [_safe_case_trace_summary(row) for row in case_trace if isinstance(row, dict)]
        if isinstance(case_trace, list)
        else []
    )
    issues = _request(
        client,
        "GET",
        f"/api/v1/projects/{project_id}/drift-issues",
        params={"limit": 200, "offset": 0},
    ).get("items")
    if not isinstance(issues, list):
        raise RuntimeError("drift issue page is missing items")
    signatures: list[str] = []
    contradiction_characters: set[str] = set()
    visible_issue_cases: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, dict) or issue.get("run_id") != run_id:
            continue
        safe_issue = _safe_visible_issue(issue)
        signatures.append(
            json.dumps(safe_issue, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )
        visible_issue_cases.append(safe_issue)
        if safe_issue["judgement"] == "contradicts" and isinstance(
            safe_issue["character_key"], str
        ):
            contradiction_characters.add(safe_issue["character_key"])
    return {
        "run_id": run_id,
        "status": run.get("status"),
        "elapsed_seconds": run.get("_acceptance_elapsed_seconds"),
        "prompt_tokens": run.get("prompt_tokens"),
        "completion_tokens": run.get("completion_tokens"),
        "stage_outcome": stage.get("outcome"),
        "stage_reason": stage.get("reason_code"),
        "material_coverage": stage.get("material_coverage"),
        "planned_chunks": counts.get("planned_chunks"),
        "processed_chunks": counts.get("processed_chunks"),
        "signals": counts.get("signal_count"),
        "draft_observations": counts.get("draft_observation_count"),
        "drift_considered": counts.get("drift_considered"),
        "reason_counts": reason_counts,
        "stage_usage": usage,
        "case_trace": case_trace,
        "issues": sorted(signatures),
        "visible_issue_cases": sorted(
            visible_issue_cases,
            key=lambda row: (
                str(row["character_key"]),
                str(row["dimension"]),
                str(row["comparison_key_sha256"]),
            ),
        ),
        "contradiction_characters": sorted(contradiction_characters),
    }


def _runtime_identity(
    expected: dict[str, Any], case_traits: dict[str, dict[str, str]]
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
    if not isinstance(comparison_hash, str) or len(comparison_hash) != 64:
        return None
    return character_key, dimension, comparison_hash


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
) -> bool:
    issue_identity = (
        issue.get("character_key"),
        issue.get("dimension"),
        issue.get("comparison_key_sha256"),
    )
    if issue_identity != runtime_identity:
        return False
    visible_expectation = expected.get("visible_issue")
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


def _evaluate_oracle_trial(
    trial: dict[str, Any],
    expected_cases: list[dict[str, Any]],
    case_traits: dict[str, dict[str, str]],
) -> dict[str, Any]:
    trace = trial.get("case_trace")
    trace = trace if isinstance(trace, list) else []
    checks: dict[str, bool] = {}
    expected_identities: list[tuple[str, str, str]] = []
    visible_checks: dict[str, bool] = {}
    visible_issues = trial.get("visible_issue_cases")
    visible_issues = visible_issues if isinstance(visible_issues, list) else []
    for expected in expected_cases:
        case_id = expected.get("case_id")
        if not isinstance(case_id, str):
            continue
        runtime_identity = _runtime_identity(expected, case_traits)
        if runtime_identity is None:
            checks[case_id] = False
            if expected.get("visible") is True:
                visible_checks[case_id] = False
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
        if expected.get("visible") is True:
            issue_candidates = [
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
            visible_checks[case_id] = (
                len(issue_candidates) == 1
                and _visible_issue_matches(
                    issue_candidates[0],
                    expected,
                    runtime_identity=runtime_identity,
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
    return {
        "cases": checks,
        "all_cases_passed": bool(checks) and all(checks.values()),
        "trace_has_no_unexpected_cases": (
            Counter(actual_identities) == Counter(expected_identities)
            and len(actual_identities) == len(trace)
        ),
        "visible_issue_cases": visible_checks,
        "visible_issues_exact": (
            bool(visible_checks)
            and all(visible_checks.values())
            and len(valid_visible_issues) == len(visible_checks)
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
        "schema_version": "character-continuity-live-dev-v2",
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
    processed = summary.get("processed_chunks")
    return (
        summary.get("status") == "completed"
        and summary.get("stage_outcome") == "completed"
        and summary.get("material_coverage") == "complete"
        and type(planned) is int
        and planned > 0
        and type(processed) is int
        and processed == planned
    )


def _baseline_admission(summary: dict[str, Any]) -> dict[str, Any]:
    usage = summary.get("stage_usage")
    usage = usage if isinstance(usage, dict) else {}
    planned = summary.get("planned_chunks")
    processed = summary.get("processed_chunks")
    checks = {
        "run_completed": summary.get("status") == "completed",
        "character_stage_completed_without_degradation": (
            summary.get("stage_outcome") == "completed"
            and summary.get("material_coverage") == "complete"
        ),
        "all_planned_chunks_processed": (
            type(planned) is int
            and planned > 0
            and type(processed) is int
            and processed == planned
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
            "processed_chunks": processed,
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
    processed = summary.get("processed_chunks")
    return (
        summary.get("status") == "completed"
        and summary.get("stage_outcome") == "partial"
        and summary.get("material_coverage") == "partial"
        and type(planned) is int
        and planned > 0
        and processed == planned
        and type(usage.get("attempted_calls")) is int
        and usage["attempted_calls"] > 0
        and bool(counts)
        and set(counts) <= DIAGNOSTIC_RECORD_REASONS
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


def _evaluate_gates(trials: list[dict[str, Any]]) -> dict[str, bool]:
    baselines = [row.get("baseline", {}) for row in trials]
    drafts = [row.get("draft", {}) for row in trials]
    reviews = [row.get("candidate_review", {}) for row in trials]
    oracle_results = [row.get("oracle", {}) for row in trials]
    project_ids = [row.get("project_id") for row in trials]
    semantic_signatures: list[tuple[tuple[str, str, str, str, str], ...]] = []
    semantic_signatures_valid = True
    for draft in drafts:
        visible_issues = (
            draft.get("visible_issue_cases") if isinstance(draft, dict) else None
        )
        if not isinstance(visible_issues, list) or not visible_issues:
            semantic_signatures_valid = False
            continue
        identities = [
            _visible_issue_semantic_identity(row)
            if isinstance(row, dict)
            else None
            for row in visible_issues
        ]
        if any(identity is None for identity in identities):
            semantic_signatures_valid = False
            continue
        semantic_signatures.append(tuple(sorted(identities)))
    enough_independent_trials = (
        len(trials) >= 3
        and all(isinstance(project_id, str) and project_id for project_id in project_ids)
        and len(set(project_ids)) == len(project_ids)
    )
    return {
        "at_least_three_independent_full_workflow_trials": enough_independent_trials,
        "all_baseline_runs_completed": bool(baselines)
        and all(row.get("status") == "completed" for row in baselines),
        "baseline_character_stage_model_tokens_reported": bool(baselines)
        and all(_stage_model_tokens_reported(row) for row in baselines),
        "baseline_all_planned_chunks_processed_without_degradation": bool(baselines)
        and all(_all_planned_chunks_processed(row) for row in baselines),
        "baseline_explicit_candidate_set_matches_oracle_exactly": bool(reviews)
        and all(_candidate_review_exact(row) for row in reviews),
        "all_draft_runs_completed": bool(drafts)
        and all(row.get("status") == "completed" for row in drafts),
        "draft_character_stage_model_tokens_reported": bool(drafts)
        and all(_stage_model_tokens_reported(row) for row in drafts),
        "draft_all_planned_chunks_processed_without_degradation": bool(drafts)
        and all(_all_planned_chunks_processed(row) for row in drafts),
        "stable_visible_issue_semantic_identity_across_independent_trials": (
            enough_independent_trials
            and semantic_signatures_valid
            and len(semantic_signatures) == len(trials)
            and len(set(semantic_signatures)) == 1
        ),
        "all_runtime_bound_expected_cases_passed": bool(oracle_results)
        and all(row.get("all_cases_passed") is True for row in oracle_results),
        "trace_has_no_unexpected_runtime_bound_cases": bool(oracle_results)
        and all(
            row.get("trace_has_no_unexpected_cases") is True
            for row in oracle_results
        ),
        "visible_issue_full_safe_signatures_match_oracle": bool(oracle_results)
        and all(row.get("visible_issues_exact") is True for row in oracle_results),
    }


def _execute_trial(
    client: httpx.Client,
    oracle: dict[str, Any],
    *,
    trial_number: int,
    timeout_seconds: float,
    guided_review: bool = False,
    diagnostic_continue_partial: bool = False,
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


def run(args: argparse.Namespace) -> int:
    global DEMO
    DEMO = FIXTURES[args.fixture]
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
        capabilities = health.get("runtime_provenance", {}).get("capabilities", {})
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
        trials = [
            _execute_trial(
                client,
                oracle,
                trial_number=index,
                timeout_seconds=args.run_timeout_seconds,
                guided_review=args.fixture == "ooc-v1",
                diagnostic_continue_partial=(
                    args.fixture == "ooc-v1"
                    and args.diagnostic_continue_partial_baseline
                ),
            )
            for index in range(1, args.trials + 1)
        ]

    gates = _evaluate_gates(trials)
    report = {
        "schema_version": "character-continuity-live-dev-v2",
        "dataset": _dataset_label(),
        "claims": {
            "blind_holdout": False,
            "production_quality": False,
            "open_text_generalization": False,
            "semantic_coverage": False,
            "independent_full_workflow_trials": gates[
                "at_least_three_independent_full_workflow_trials"
            ],
        },
        "runtime_provenance": health.get("runtime_provenance"),
        "dataset_sha256": _dataset_hashes(),
        "trials": trials,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _emit_report(report, args.output_json)
    return 0 if report["passed"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", choices=tuple(FIXTURES), default="demo")
    parser.add_argument(
        "--diagnostic-continue-partial-baseline",
        action="store_true",
        help="OOC DEV only: continue after record-level partial baseline; full gate stays false",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--trials", type=int, default=3, choices=range(1, 6))
    parser.add_argument("--run-timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--output-json",
        help=(
            "optional UTF-8 report path inside artifacts/; a bare filename is "
            "placed there automatically"
        ),
    )
    return parser.parse_args()


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
