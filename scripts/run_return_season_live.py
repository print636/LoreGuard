"""Run the frozen Return Season DEV fixture against an isolated HTTP service.

The default preflight makes no network or model call. A live run needs an
explicitly provisioned evaluation database, its live instance identity, a
non-default loopback port, a new report file, and provider-call consent.
Arm A leaves all candidates pending; optional arm B confirms uniquely matched
value/boundary candidates as a simulated author, without binding an axis.
Neither arm treats an unbound or partially covered draft as a quality result.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.character_trait_extraction import DraftSignalTraceV1
from app.chunking import chunk_document
from app.pipeline import DocumentInput
from scripts import run_character_axis_live as legacy
from scripts import run_character_axis_direction_live as isolation


DATASET = ROOT / "data" / "character-ooc-return-season-dev-v1"
SCHEMA = "return-season-live-diagnostic-v1"
PINNED_SHA256 = {
    "01-world-setting.md": "f1f62c6b328c4e498c4358b57fd6e436b56df07371c7c5fabd1a6c531050d8c2",
    "02-character-profiles.md": "c1ce95bc2b0a73ad5c09afe130915fc3a797d4055495148664b6e66573961ec7",
    "03-published-main-v1.0.md": "e645113e1fd25010ecc51c2a05b050757284aceda6c292cd52b56692bb8adc66",
    "04-published-main-v1.1.md": "d9579e2e1b31440c50b4d8b2ad231ab7cda9647ceca3292612d866e324542dd8",
    "05-draft-event-v1.2.md": "a15766ecd51d0608d7ceba4836162e9cd688d2d5b5a07af32f69608a09c62c43",
    "manifest.json": "2691dc7538aadc8125613062c1421435a739cf9f69405251a8031dd828c37cfb",
    "review-plan.json": "7a5c8cf052bcf46d9b36422091158325df54f023f343d967b2baf020e087397e",
    "oracle.json": "ae6bb92b4a0e434b4af5049beb5b3ffffbbfdda140fa35e07339150aafbfd2d3",
}
BASELINE = tuple(PINNED_SHA256)[:4]
DRAFT = "05-draft-event-v1.2.md"
SAFE_KEY = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
# These are literal profile clauses, selected from the author review plan's
# source lines. They are not oracle expectations or inferred labels.
TARGET_CLAUSES = {
    "shelter_eligibility_without_registry": "不会仅凭居籍状态取消他们的避风席位",
    "notify_affected_waiters": "先把变更告诉受影响的人",
    "medical_identity_consent": "核对许可的内容与受众",
    "forecast_uncertainty_disclosure": "说明不确定性",
    "raw_observation_integrity": "保留原始读数，另附更正说明",
}


@dataclass(frozen=True)
class Fixture:
    files: dict[str, bytes]
    manifest: dict[str, Any]
    plan: dict[str, Any]


def _fail(code: str, stage: str) -> None:
    raise legacy.SafeFailure(code, stage)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_object(data: bytes, stage: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError):
        _fail("invalid_json", stage)
    if type(value) is not dict:
        _fail("json_object_required", stage)
    return value


def verify_fixture(root: Path = DATASET) -> Fixture:
    """Verify every frozen hash, without retaining or parsing oracle bytes."""
    if root.resolve() != DATASET.resolve():
        _fail("fixture_path_not_pinned", "fixture_preflight")
    files: dict[str, bytes] = {}
    for name, digest in PINNED_SHA256.items():
        try:
            data = (root / name).read_bytes()
        except OSError:
            _fail("fixture_file_unavailable", "fixture_preflight")
        if _digest(data) != digest:
            _fail("fixture_hash_mismatch", "fixture_preflight")
        if name != "oracle.json":
            files[name] = data
    manifest = _json_object(files["manifest.json"], "fixture_preflight")
    plan = _json_object(files["review-plan.json"], "fixture_preflight")
    docs = manifest.get("documents_in_import_order")
    if (
        manifest.get("schema_version") != "character-ooc-return-season-manifest-v1"
        or manifest.get("world_id") != "shanchao-return-season"
        or type(docs) is not list
        or [row.get("file") for row in docs if type(row) is dict] != [*BASELINE, DRAFT]
        or len(docs) != 5
        or manifest.get("oracle_file") != "oracle.json"
        or manifest.get("author_review_plan_file") != "review-plan.json"
        or plan.get("schema_version") != "character-ooc-return-season-review-plan-v1"
        or plan.get("world_id") != manifest["world_id"]
        or plan.get("status") != "proposed_not_confirmed"
        or plan.get("author_confirmation_required") is not True
    ):
        _fail("fixture_contract", "fixture_preflight")
    roles = ["canon", "character_profile", "chapter", "chapter", "chapter"]
    for index, row in enumerate(docs):
        if (
            type(row) is not dict or row.get("document_role") != roles[index]
            or row.get("resolution_state") != "confirmed"
            or row.get("publication_status") != ("draft" if index == 4 else "published")
            or type(row.get("scope")) is not dict
        ):
            _fail("manifest_document_contract", "fixture_preflight")
        try:
            files[row["file"]].decode("utf-8")
        except UnicodeError:
            _fail("invalid_document_encoding", "fixture_preflight")
    axes = plan.get("approved_axes_to_review")
    if type(axes) is not list or len(axes) != 5:
        _fail("review_plan_contract", "fixture_preflight")
    seen: set[str] = set()
    profile = files["02-character-profiles.md"].decode("utf-8").splitlines()
    for axis in axes:
        if type(axis) is not dict:
            _fail("review_plan_contract", "fixture_preflight")
        key = axis.get("axis_key")
        source = axis.get("source")
        if (
            type(key) is not str or SAFE_KEY.fullmatch(key) is None
            or key not in TARGET_CLAUSES or key in seen
            or axis.get("trait_type") != "core_personality"
            or type(axis.get("character_key")) is not str
            or type(source) is not dict
            or source.get("document_name") != "02-character-profiles.md"
            or type(source.get("line")) is not int
            or not 1 <= source["line"] <= len(profile)
            or f"【{source.get('fragment_id')}】" not in profile[source["line"] - 1]
            or TARGET_CLAUSES[key] not in profile[source["line"] - 1]
        ):
            _fail("review_plan_contract", "fixture_preflight")
        seen.add(key)
    if seen != set(TARGET_CLAUSES):
        _fail("review_plan_contract", "fixture_preflight")
    return Fixture(files=files, manifest=manifest, plan=plan)


def _upload(client: httpx.Client, project_id: str, fixture: Fixture, row: dict[str, Any]) -> str:
    name = row["file"]
    payload = legacy._request(
        client, "POST", f"/api/v1/projects/{project_id}/documents/text", "upload",
        json={
            "name": name,
            "content": fixture.files[name].decode("utf-8"),
            "document_role": row["document_role"],
            "story_scope": "main",
            "narrative_context": {
                "resolution_state": row["resolution_state"],
                "publication_status": row["publication_status"],
                "scope": row["scope"],
            },
        },
    )
    if type(payload) is not dict or type(payload.get("id")) is not str:
        _fail("document_response_contract", "upload")
    return payload["id"]


def _candidate_matches(candidate: dict[str, Any], axis: dict[str, Any], *, strict_type: bool) -> bool:
    source = axis["source"]
    statement = legacy._normalized_clause(candidate.get("value"))
    anchor = legacy._normalized_clause(TARGET_CLAUSES[axis["axis_key"]])
    candidate_type = candidate.get("trait_type")
    return (
        candidate.get("reviewable") is True
        and candidate.get("character_key") == axis["character_key"]
        and candidate_type == ("core_personality" if strict_type else candidate_type)
        and (strict_type or candidate_type in {"value", "behavior_boundary"})
        and candidate.get("origin") == "explicit_setting"
        and candidate.get("polarity") in {"positive", "negative"}
        and candidate.get("stability") in {"core", "stable"}
        and bool(anchor) and anchor in statement
        and legacy._reference_has_line(
            candidate.get("evidence"), document=source["document_name"],
            line=source["line"], source_quote=TARGET_CLAUSES[axis["axis_key"]],
        )
    )


def _inventory(pending: list[dict[str, Any]], fixture: Fixture) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows: dict[str, Any] = {}
    unique_semantic: dict[str, dict[str, Any]] = {}
    for axis in fixture.plan["approved_axes_to_review"]:
        key = axis["axis_key"]
        strict = [row for row in pending if _candidate_matches(row, axis, strict_type=True)]
        semantic = [row for row in pending if _candidate_matches(row, axis, strict_type=False)]
        rows[key] = {
            "strict_core_personality_matches": min(len(strict), 1000),
            "source_clause_value_or_boundary_matches": min(len(semantic), 1000),
            "source_clause_unique": len(semantic) == 1,
        }
        if len(semantic) == 1:
            unique_semantic[key] = semantic[0]
    return rows, unique_semantic


def _confirm_simulated(
    client: httpx.Client, project_id: str, fixture: Fixture,
    inventory: dict[str, Any], chosen: dict[str, dict[str, Any]],
) -> int:
    if len(chosen) != len(TARGET_CLAUSES):
        _fail("semantic_source_not_unique", "simulated_review")
    candidate_ids = [candidate.get("id") for candidate in chosen.values()]
    if any(type(value) is not str for value in candidate_ids) or len(set(candidate_ids)) != len(candidate_ids):
        _fail("candidate_identity_ambiguous", "simulated_review")
    confirmed = 0
    for axis in fixture.plan["approved_axes_to_review"]:
        key = axis["axis_key"]
        if inventory[key]["strict_core_personality_matches"] != 0:
            _fail("strict_type_mismatch_not_isolated", "simulated_review")
        candidate = chosen[key]
        if type(candidate.get("revision")) is not int or candidate["revision"] < 0:
            _fail("candidate_revision_invalid", "simulated_review")
        response = legacy._request(
            client, "POST",
            f"/api/v1/projects/{project_id}/characters/{quote(axis['character_key'], safe='')}/profile-candidates/{quote(candidate['id'], safe='')}/decisions",
            "simulated_review",
            headers={"Idempotency-Key": f"return-season-sim-{uuid4().hex}"},
            json={
                "decision": "confirm", "expected_revision": candidate["revision"],
                "comment": "Developer-only simulated author confirmation of one frozen source clause",
            },
        )
        reviewed = response.get("candidate") if type(response) is dict else None
        if (
            type(reviewed) is not dict or reviewed.get("id") != candidate["id"]
            or reviewed.get("review_state") != "confirmed"
            or reviewed.get("approved_axis_id") is not None
        ):
            _fail("simulated_confirmation_contract", "simulated_review")
        confirmed += 1
    return confirmed


def _runtime_preflight(
    client: httpx.Client, *, require_draft_trace: bool = False,
) -> tuple[str, int, bool]:
    health = legacy._request(client, "GET", "/health", "health")
    if type(health) is not dict:
        _fail("health_contract", "runtime_preflight")
    code = legacy._code_state()
    digest = legacy._service_preflight_gate(
        health, code, legacy._local_service_artifact_sha256(ROOT)
    )
    model = health.get("model")
    configured = type(model) is dict and model.get("configured") is True
    if not configured:
        provider = legacy._request(
            client, "GET", "/api/v1/account/model-provider", "model_provider"
        )
        configured = type(provider) is dict and (
            provider.get("configured") is True
            or provider.get("service_default_available") is True
        )
    if not configured:
        _fail("model_not_configured", "runtime_preflight")
    provenance = legacy._safe_character_runtime_provenance(
        health.get("runtime_provenance")
    )
    max_chunk_chars = (
        provenance["character_consistency_limits"]["signal_max_chunk_chars"]
        if provenance is not None else None
    )
    if type(max_chunk_chars) is not int or not 256 <= max_chunk_chars <= 12_000:
        _fail("runtime_provenance_invalid", "runtime_preflight")
    limits = provenance["character_consistency_limits"]
    trace_enabled = (
        limits.get("signal_draft_trace_v1") is True
        and limits.get("signal_draft_trace_version") == "draft-signal-trace-v1"
    )
    if require_draft_trace and not trace_enabled:
        _fail("draft_trace_required", "runtime_preflight")
    return digest, max_chunk_chars, trace_enabled


def _check_runtime(client: httpx.Client, digest: str, summary: dict[str, Any]) -> None:
    if summary.get("runtime_provenance_sha256") != digest:
        _fail("worker_runtime_provenance_mismatch", "runtime_postrun")
    health = legacy._request(client, "GET", "/health", "health_postrun")
    if type(health) is not dict or legacy._runtime_provenance_digest(
        health.get("runtime_provenance")
    ) != digest:
        _fail("api_runtime_provenance_changed", "runtime_postrun")


def _safe_draft_trace_chunks(stage: object) -> tuple[list[dict[str, Any]], int] | None:
    """Accept only the server's fixed content-free trace schema."""
    if type(stage) is not dict:
        return None
    raw = stage.get("draft_trace_chunks")
    omitted = stage.get("draft_trace_chunks_omitted_count")
    if (
        type(raw) is not list or len(raw) > 128
        or type(omitted) is not int or not 0 <= omitted <= 1_000_000
    ):
        return None
    safe: list[dict[str, Any]] = []
    for row in raw:
        if type(row) is not dict or set(row) != {
            "source_document_ordinal", "document_chunk_ordinal",
            "stage_chunk_ordinal", "phase", "target_ordinal", "outcome",
            "availability", "trace",
        }:
            return None
        source_ordinal = row["source_document_ordinal"]
        document_chunk = row["document_chunk_ordinal"]
        stage_chunk = row["stage_chunk_ordinal"]
        target = row["target_ordinal"]
        if (
            type(source_ordinal) is not int or not 0 <= source_ordinal <= 1000
            or type(document_chunk) is not int or not 1 <= document_chunk <= 128
            or type(stage_chunk) is not int or not 1 <= stage_chunk <= 128
            or (target is not None and (type(target) is not int or not 1 <= target <= 128))
            or type(row["phase"]) is not str or row["phase"] not in {
                "primary_extraction", "targeted_recall", "targeted_verification",
            }
            or type(row["outcome"]) is not str or row["outcome"] not in {
                "disabled", "completed", "partial", "degraded", "skipped",
            }
            or type(row["availability"]) is not str
            or row["availability"] not in {"available", "unavailable"}
        ):
            return None
        if row["availability"] == "unavailable":
            if row["trace"] is not None:
                return None
            trace = None
        else:
            try:
                trace = DraftSignalTraceV1.model_validate(row["trace"]).model_dump(
                    mode="json"
                )
            except (TypeError, ValueError):
                return None
            last_attempt = len(trace["attempts"])
            final_ordinals = set(trace["final_accepted_record_ordinals"])
            for attempt in trace["attempts"]:
                for event in attempt["events"]:
                    event["final_accepted"] = (
                        trace["final_state"] == "clean"
                        and attempt["attempt"] == last_attempt
                        and event["record_ordinal"] in final_ordinals
                    )
        safe.append({
            "source_document_ordinal": source_ordinal,
            "document_chunk_ordinal": document_chunk,
            "stage_chunk_ordinal": stage_chunk,
            "phase": row["phase"], "target_ordinal": target,
            "outcome": row["outcome"],
            "availability": row["availability"], "trace": trace,
        })
    return safe, omitted


def _execute_arm(
    client: httpx.Client, fixture: Fixture, *, simulated: bool,
    timeout_seconds: float, runtime_digest: str, draft_trace_enabled: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    project = legacy._request(
        client, "POST", "/api/v1/projects", "project_create",
        json={
            "name": f"return-season-{'sim' if simulated else 'baseline'}-{uuid4().hex[:8]}",
            "description": "Frozen developer-visible character consistency diagnostic",
        },
    )
    if type(project) is not dict or type(project.get("id")) is not str:
        _fail("project_create_contract", "project_create")
    project_id = project["id"]
    documents = fixture.manifest["documents_in_import_order"]
    for row in documents[:4]:
        _upload(client, project_id, fixture, row)
    baseline_id = legacy._start_run(client, project_id, mode="baseline_build")
    baseline_run = legacy._wait_run(client, baseline_id, timeout_seconds=timeout_seconds)
    known_documents = set(BASELINE) | {DRAFT}
    baseline = legacy._run_summary(client, baseline_run, known_documents=known_documents)
    _check_runtime(client, runtime_digest, baseline)
    admitted = legacy._baseline_admission(baseline)["admitted"] is True
    public = {
        "arm": "simulated_author" if simulated else "unconfirmed_baseline",
        "project_id_sha256": legacy._sha256(project_id),
        "baseline": legacy._public_run(baseline),
        "baseline_admitted": admitted,
        "candidate_inventory": None,
        "simulated_confirmed_count": 0,
        "approved_axis_bound_count": 0,
        "draft": None,
        "draft_trace_chunks": None,
        "draft_trace_chunks_omitted_count": None,
        "draft_trace_status": "trace_unavailable",
        "quality_scored": False,
    }
    if not admitted:
        public["arm_stop_reason"] = "baseline_admission_failed"
        return public, None
    pending = legacy._list_pending(client, project_id)
    inventory, chosen = _inventory(pending, fixture)
    public["candidate_inventory"] = inventory
    if simulated and (
        len(chosen) != len(TARGET_CLAUSES)
        or any(inventory[key]["strict_core_personality_matches"] != 0 for key in chosen)
    ):
        public["arm_stop_reason"] = "semantic_source_not_unique_or_type_mixed"
        return public, None
    confirmed = _confirm_simulated(client, project_id, fixture, inventory, chosen) if simulated else 0
    public["simulated_confirmed_count"] = confirmed
    draft_id = _upload(client, project_id, fixture, documents[4])
    run_id = legacy._start_run(client, project_id, mode="draft_review", target_id=draft_id)
    draft_run = legacy._wait_run(client, run_id, timeout_seconds=timeout_seconds)
    draft = legacy._run_summary(client, draft_run, known_documents=known_documents)
    _check_runtime(client, runtime_digest, draft)
    public["draft"] = legacy._public_run(draft)
    diagnostics = legacy._request(
        client, "GET", f"/api/v1/analysis-runs/{run_id}/diagnostics",
        "draft_trace_diagnostics",
    )
    stage = diagnostics.get("character_consistency") if type(diagnostics) is dict else None
    trace = _safe_draft_trace_chunks(stage)
    public["draft_trace_chunks"] = trace[0] if trace is not None else None
    public["draft_trace_chunks_omitted_count"] = trace[1] if trace is not None else None
    public["draft_trace_claim_status"] = (
        "model_start_claim_unverified" if trace is not None and trace[0] else None
    )
    public["draft_trace_status"] = (
        "disabled" if not draft_trace_enabled else
        "trace_unavailable" if trace is None or (not trace[0] and trace[1] > 0) else
        "no_trace_records" if not trace[0] else "available"
    )
    return public, draft


def _score_after_runs(
    fixture: Fixture, arms: list[tuple[dict[str, Any], dict[str, Any] | None]],
    *, max_chunk_chars: int | None = None,
) -> list[dict[str, Any]]:
    """The oracle is decoded only here, after all HTTP/model activity ends."""
    try:
        oracle_bytes = (DATASET / "oracle.json").read_bytes()
    except OSError:
        _fail("oracle_unavailable", "oracle_scoring")
    if _digest(oracle_bytes) != PINNED_SHA256["oracle.json"]:
        _fail("oracle_hash_mismatch", "oracle_scoring")
    oracle = _json_object(oracle_bytes, "oracle_scoring")
    cases = oracle.get("cases")
    if (
        oracle.get("schema_version") != "character-ooc-return-season-oracle-v1"
        or oracle.get("world_id") != fixture.manifest["world_id"]
        or type(cases) is not list or len(cases) != 11
        or sum(type(row) is dict and row.get("kind") == "potential_ooc" for row in cases) != 5
        or sum(type(row) is dict and row.get("kind") == "hard_negative" for row in cases) != 6
    ):
        _fail("oracle_contract", "oracle_scoring")
    draft_lines = fixture.files[DRAFT].decode("utf-8").splitlines()
    line_to_fragment: dict[int, str] = {}
    fragment = re.compile(r"D[0-9]{2}\Z")
    for case in cases:
        if type(case) is not dict or type(case.get("draft")) is not list:
            _fail("oracle_case_contract", "oracle_scoring")
        for ref in case["draft"]:
            if type(ref) is not dict:
                _fail("oracle_case_contract", "oracle_scoring")
            line = ref.get("line")
            marker = ref.get("fragment_id")
            if (
                ref.get("document_name") != DRAFT
                or type(line) is not int or not 1 <= line <= len(draft_lines)
                or type(marker) is not str or fragment.fullmatch(marker) is None
                or not draft_lines[line - 1].startswith(f"【{marker}】")
                or (line in line_to_fragment and line_to_fragment[line] != marker)
            ):
                _fail("oracle_case_contract", "oracle_scoring")
            line_to_fragment[line] = marker
    chunks = (
        chunk_document(
            DocumentInput(id="frozen-draft", name=DRAFT,
                          content=fixture.files[DRAFT].decode("utf-8")),
            max_chunk_chars, overlap_lines=0,
        )
        if type(max_chunk_chars) is int and 256 <= max_chunk_chars <= 12_000
        else []
    )
    scored: list[dict[str, Any]] = []
    for public, draft in arms:
        trace_rows = public.get("draft_trace_chunks")
        if type(trace_rows) is list:
            for row in trace_rows:
                trace = row.get("trace")
                if type(trace) is not dict:
                    continue
                index = row["document_chunk_ordinal"] - 1
                chunk = chunks[index] if 0 <= index < len(chunks) else None
                for attempt in trace["attempts"]:
                    for event in attempt["events"]:
                        claimed_start = event["claimed_line_start_ordinal"]
                        claimed_end = event["claimed_line_end_ordinal"]
                        chunk_lines = len(chunk.content.splitlines()) if chunk else 0
                        valid_span = (
                            type(claimed_start) is int and type(claimed_end) is int
                            and 1 <= claimed_start <= claimed_end <= chunk_lines
                        )
                        line = (
                            chunk.global_line_start + claimed_start - 1
                            if valid_span and chunk is not None else None
                        )
                        start_fragment = line_to_fragment.get(line)
                        event["claimed_start_Dxx"] = start_fragment
                        event["claimed_single_line_Dxx"] = (
                            start_fragment
                            if valid_span and claimed_start == claimed_end else None
                        )
                        event["claimed_start_basis"] = "unverified_model_claim"
                        event["claimed_span_status"] = (
                            "single_line_claim" if valid_span and claimed_start == claimed_end
                            else "multi_line_claim" if valid_span else "unavailable"
                        )
        draft = draft or {}
        coverage_complete = (
            draft.get("status") == "completed"
            and draft.get("stage_outcome") == "completed"
            and draft.get("material_coverage") == "complete"
            and type(draft.get("planned_chunks")) is int
            and draft["planned_chunks"] > 0
            and draft.get("processed_chunks") == draft["planned_chunks"]
        )
        observations = draft.get("accepted_draft_observation_total")
        if public.get("baseline_admitted") is not True:
            reason = "baseline_coverage_incomplete"
        elif public.get("arm_stop_reason") is not None:
            reason = "simulated_review_unavailable"
        elif not coverage_complete:
            reason = "draft_coverage_incomplete"
        elif type(observations) is not int or observations <= 0:
            reason = "no_accepted_draft_observation"
        else:
            reason = "approved_axis_unbound"
        public["case_assessment"] = [
            {"case_id": row["case_id"], "kind": row["kind"],
             "status": "unassessed", "reason_code": reason}
            for row in cases
            if type(row) is dict and type(row.get("case_id")) is str
            and SAFE_KEY.fullmatch(row["case_id"]) is not None
        ]
        if len(public["case_assessment"]) != 11:
            _fail("oracle_case_contract", "oracle_scoring")
        public["assessed_case_count"] = 0
        public["unassessed_case_count"] = 11
        public["complete_draft_coverage"] = coverage_complete
        scored.append(public)
    return scored


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "developer_visible": True,
        "blind_holdout": False,
        "production_quality": False,
        "quality_scored": False,
        "passed": False,
    }
    try:
        fixture = verify_fixture()
        report["fixture_sha256"] = dict(PINNED_SHA256)
        if args.preflight_only:
            report["phase"] = "preflight"
            report["preflight_verified"] = True
            return report, 0
        if not args.allow_provider_call or not args.output_json:
            _fail("provider_opt_in_and_output_required", "run_preflight")
        output = legacy._resolve_output_json(args.output_json)
        if output.exists():
            _fail("output_report_exists", "run_preflight")
        base_url = isolation._isolated_base_url(args.base_url)
        if urlsplit(base_url).port == 8080:
            _fail("isolation_base_url_invalid", "run_preflight")
        expected = isolation._expected_isolation(
            args.expected_eval_db_id, args.expected_eval_instance_id,
            args.expected_eval_db_path,
        )
        with httpx.Client(
            base_url=base_url, timeout=httpx.Timeout(30.0), trust_env=False
        ) as client:
            isolation._verify_live_isolation(
                client, expected, require_fresh=True, stage="isolation_preflight"
            )
            runtime_digest, max_chunk_chars, draft_trace_enabled = _runtime_preflight(
                client, require_draft_trace=args.require_draft_trace
            )
            arms = [_execute_arm(
                client, fixture, simulated=False,
                timeout_seconds=args.run_timeout_seconds, runtime_digest=runtime_digest,
                draft_trace_enabled=draft_trace_enabled,
            )]
            isolation._verify_live_isolation(
                client, expected, require_fresh=False, stage="isolation_post_arm_a"
            )
            if args.simulate_author:
                arms.append(_execute_arm(
                    client, fixture, simulated=True,
                    timeout_seconds=args.run_timeout_seconds, runtime_digest=runtime_digest,
                    draft_trace_enabled=draft_trace_enabled,
                ))
                isolation._verify_live_isolation(
                    client, expected, require_fresh=False, stage="isolation_post_arm_b"
                )
        report["phase"] = "completed"
        report["runtime_provenance_sha256"] = runtime_digest
        report["draft_trace_enabled"] = draft_trace_enabled
        report["draft_trace_required"] = args.require_draft_trace
        project_hashes = [public.get("project_id_sha256") for public, _ in arms]
        report["independent_projects"] = (
            len(project_hashes) == len(set(project_hashes))
            and all(
                type(value) is str and legacy.SHA256.fullmatch(value) is not None
                for value in project_hashes
            )
        )
        report["arms"] = _score_after_runs(
            fixture, arms, max_chunk_chars=max_chunk_chars
        )
        return report, 0
    except Exception as exc:
        report["phase"] = "failed"
        report["failure"] = legacy._failure(exc)
        return report, 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-eval-db-id")
    parser.add_argument("--expected-eval-instance-id")
    parser.add_argument("--expected-eval-db-path")
    parser.add_argument("--allow-provider-call", action="store_true")
    parser.add_argument("--simulate-author", action="store_true")
    parser.add_argument("--require-draft-trace", action="store_true")
    parser.add_argument("--run-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--output-json")
    args = parser.parse_args()
    if args.run_timeout_seconds <= 0:
        parser.error("--run-timeout-seconds must be positive")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    result, exit_code = run(parsed)
    try:
        legacy._emit_report(result, parsed.output_json)
    except (OSError, ValueError):
        print(json.dumps(result, ensure_ascii=False, indent=2))
        exit_code = 1
    raise SystemExit(exit_code)
