"""Diagnose ten independent, synthetic scoped-character cases over isolated HTTP.

This is a developer-visible DEV diagnostic, not an accuracy benchmark. The
default phase only verifies frozen files. Live phases require an explicit
provider opt-in, a fresh evaluation database, and a non-user loopback port.
``prepare`` produces a checkpoint; ``review`` requires a separately written,
explicit simulated-author decision file. The sealed oracle is parsed only
after all review-phase HTTP/model activity has ended.
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

from scripts import run_character_axis_direction_live as isolation
from scripts import run_character_axis_live as legacy


DATASET = ROOT / "data" / "character-ooc-scoped-dev-v1"
SCHEMA = "scoped-ooc-dev-runner-v1"
DECISIONS_SCHEMA = "scoped-ooc-dev-author-decisions-v1"
REVIEWER_DECLARATION = "developer_simulated_author_reviewed_frozen_source_axis_and_direction"
PINNED_SHA256 = {
    "manifest.json": "7fd745c870e6a07284a98b08f650f7655831eca690f4efd1f531b6d1fc126b07",
    "author-plan.json": "a788b6ac60ee355278a97d7c7a9fa508f182241abda86228bb178220b84624f0",
    "oracle.json": "99d33a78489b2848bda8970ac960a7ea2e99b880616d0f589a5585e13635a792",
    "01-world-setting.md": "35ee640df223059d163fce53b37c9d21e59ee71a9c98c6867e146dce0e6d98a7",
    "02-character-profiles.md": "52130f59ab5462d08a641de44e1afa4915ab612c67c069b7e17ccfac178dcdf0",
    "03-published-history-v1.1.md": "375a3207332ae6a466a5c9e0372d4d88127e92d4d494b872bcde7183d5632e9e",
    "drafts/01-two-real-patients.md": "49a8b1f844ff91606b270098064a1e8af53a8e03b40b1aace399cca98bfcbb28",
    "drafts/02-omitted-subject-announcements.md": "d162ce94dc0d8902054e5871bcf07689b6b8b46c7e52d0f4a72e869a3bc42b8e",
    "drafts/03-fictional-patients.md": "ec5827ae933d0214be0876605e494a202b4e38f739440b7c9448fd6ed4550f0a",
    "drafts/04-red-alert-private-channel.md": "2bba14831acc6abd3dd81e9e9212372b4b4adaa51edeb494880edabee0c5980d",
    "drafts/05-do-not-combine-rows.md": "ead080aa5b5982fd7be98ef0f2a085a8064c71ffdb18a2f68b90b49723f5c219",
    "drafts/06-one-event-two-mentions.md": "84560dc8dda53e7b32bea43c4da5f7d6fe0400667aaa34134c23d17e87e1030c",
    "drafts/07-other-actor-upload.md": "4d188ca6c260255672e7ae2272742147a98a59c7aaef1e27e079b0bc085572cf",
    "drafts/08-rehearsal-and-counterfactual.md": "8cc666cfbedec29e1ea985d23218b1897d4018842e430fe2a263dc47998a7f80",
    "drafts/09-published-growth-in-format.md": "3bfa78a4be553cad919f5a1aa72cd9fc729dac66c26c08e64b9fdb64a86daf5b",
    "drafts/10-vague-consent.md": "f355694100b936982d258e9c7bc1e074d01532f95ec50ae67b6677546d18b118",
}
COMMON = (
    "01-world-setting.md", "02-character-profiles.md",
    "03-published-history-v1.1.md",
)
SAFE_CASE = re.compile(r"[a-z][a-z0-9_]{1,79}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class Fixture:
    manifest: dict[str, Any]
    plan: dict[str, Any]
    files: dict[str, bytes]


def _fail(code: str, stage: str) -> None:
    raise legacy.SafeFailure(code, stage)


def _digest(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(data: bytes, stage: str) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeError, ValueError):
        _fail("invalid_json", stage)
    if type(value) is not dict:
        _fail("json_object_required", stage)
    return value


def verify_fixture() -> Fixture:
    """Pin all story input and plans; hash, but do not parse, the oracle."""
    files: dict[str, bytes] = {}
    for name, expected in PINNED_SHA256.items():
        path = DATASET / name
        try:
            if _file_digest(path) != expected:
                _fail("fixture_hash_mismatch", "fixture_preflight")
            if name != "oracle.json":
                files[name] = path.read_bytes()
        except OSError:
            _fail("fixture_file_unavailable", "fixture_preflight")
    manifest = _json_object(files["manifest.json"], "fixture_preflight")
    plan = _json_object(files["author-plan.json"], "fixture_preflight")
    cases = manifest.get("cases")
    common = manifest.get("common_documents_in_import_order")
    if (
        manifest.get("schema_version") != "character-ooc-scoped-dev-manifest-v1"
        or manifest.get("world_id") != "shanchao-scoped-ooc"
        or manifest.get("author_plan_file") != "author-plan.json"
        or manifest.get("oracle_file") != "oracle.json"
        or manifest.get("author_plan_sha256") != PINNED_SHA256["author-plan.json"]
        or type(common) is not list or len(common) != 3
        or [row.get("file") for row in common if type(row) is dict] != list(COMMON)
        or type(cases) is not list or len(cases) != 10
        or plan.get("schema_version") != "character-ooc-scoped-dev-author-plan-v1"
        or plan.get("world_id") != manifest["world_id"]
        or plan.get("status") != "proposed_not_confirmed"
        or plan.get("dataset_boundary", {}).get("human_author_confirmed") is not False
        or manifest.get("dataset_boundary", {}).get("production_quality") is not False
    ):
        _fail("fixture_contract", "fixture_preflight")
    for row, role in zip(common, ("canon", "character_profile", "chapter")):
        if (
            type(row) is not dict or row.get("document_role") != role
            or row.get("resolution_state") != "confirmed"
            or row.get("publication_status") != "published"
            or type(row.get("scope")) is not dict
        ):
            _fail("common_document_contract", "fixture_preflight")
    defaults = manifest.get("case_draft_defaults")
    if (
        type(defaults) is not dict
        or defaults.get("document_role") != "chapter"
        or defaults.get("resolution_state") != "confirmed"
        or defaults.get("publication_status") != "draft"
        or type(defaults.get("scope")) is not dict
    ):
        _fail("draft_document_contract", "fixture_preflight")
    seen: set[str] = set()
    draft_names: set[str] = set()
    for row in cases:
        if type(row) is not dict or set(row) != {"case_id", "draft_file", "axis_key"}:
            _fail("case_contract", "fixture_preflight")
        key, draft = row["case_id"], row["draft_file"]
        if (
            type(key) is not str or SAFE_CASE.fullmatch(key) is None or key in seen
            or type(draft) is not str or draft not in PINNED_SHA256
            or not draft.startswith("drafts/") or draft in draft_names
            or Path(draft).name != draft.removeprefix("drafts/")
        ):
            _fail("case_contract", "fixture_preflight")
        seen.add(key)
        draft_names.add(draft)
    if draft_names != {name for name in PINNED_SHA256 if name.startswith("drafts/")}:
        _fail("case_set_mismatch", "fixture_preflight")
    axes = plan.get("proposed_axes")
    if type(axes) is not list or len(axes) != 2:
        _fail("author_plan_contract", "fixture_preflight")
    profile_lines = files["02-character-profiles.md"].decode("utf-8").splitlines()
    axis_keys: set[str] = set()
    source_lines: set[int] = set()
    for axis in axes:
        if type(axis) is not dict or type(axis.get("source")) is not dict:
            _fail("author_plan_contract", "fixture_preflight")
        source = axis["source"]
        key = axis.get("axis_key")
        line_number = source.get("line")
        quote_text = source.get("source_quote")
        if (
            type(key) is not str or SAFE_CASE.fullmatch(key) is None
            or key in axis_keys or axis.get("dimension") not in {"value", "behavior_boundary"}
            or not isinstance(axis.get("character_key"), str)
            or not isinstance(axis.get("comparison_key"), str)
            or not all(isinstance(axis.get(field), str) and 1 <= len(axis[field]) <= 200
                       for field in ("display_name", "definition", "positive_proposition", "applicability_scope"))
            or source.get("document_name") != COMMON[1]
            or type(line_number) is not int or not 1 <= line_number <= len(profile_lines)
            or line_number in source_lines
            or not isinstance(quote_text, str) or not 2 <= len(quote_text) <= 100
            or profile_lines[line_number - 1].count(quote_text) != 1
            or f"【{source.get('fragment_id')}】" not in profile_lines[line_number - 1]
        ):
            _fail("author_axis_contract", "fixture_preflight")
        axis_keys.add(key)
        source_lines.add(line_number)
    if any(row["axis_key"] not in axis_keys for row in cases):
        _fail("case_axis_contract", "fixture_preflight")
    for name, data in files.items():
        if name.endswith(".md") and name != "manifest.json":
            try:
                data.decode("utf-8")
            except UnicodeError:
                _fail("document_encoding_invalid", "fixture_preflight")
    return Fixture(manifest=manifest, plan=plan, files=files)


def _axis_for_case(fixture: Fixture, case: dict[str, Any]) -> dict[str, Any]:
    return next(row for row in fixture.plan["proposed_axes"]
                if row["axis_key"] == case["axis_key"])


def _case_document_name(case: dict[str, Any]) -> str:
    return Path(case["draft_file"]).name


def _upload(client: httpx.Client, project_id: str, fixture: Fixture,
            row: dict[str, Any], *, draft: bool = False) -> str:
    name = row["draft_file"] if draft else row["file"]
    metadata = fixture.manifest["case_draft_defaults"] if draft else row
    result = legacy._request(
        client, "POST", f"/api/v1/projects/{project_id}/documents/text", "upload",
        json={
            "name": Path(name).name, "content": fixture.files[name].decode("utf-8"),
            "document_role": metadata["document_role"], "story_scope": "main",
            "narrative_context": {
                "resolution_state": metadata["resolution_state"],
                "publication_status": metadata["publication_status"],
                "scope": metadata["scope"],
            },
        },
    )
    if type(result) is not dict or type(result.get("id")) is not str:
        _fail("upload_response_contract", "upload")
    return result["id"]


def _candidate_matches(row: dict[str, Any], axis: dict[str, Any], fixture: Fixture,
                       *, project_id: str, baseline_run_id: str,
                       source_document_id: str) -> bool:
    source = axis["source"]
    line_number = source["line"]
    line = fixture.files[COMMON[1]].decode("utf-8").splitlines()[line_number - 1]
    evidence = row.get("evidence")
    if (
        row.get("id") is None or row.get("project_id") != project_id
        or row.get("source_run_id") != baseline_run_id
        or row.get("review_state") != "pending"
        or row.get("reviewable") is not True or row.get("source_verified") is not True
        or row.get("character_key") != axis["character_key"]
        or row.get("trait_type") != axis["dimension"]
        or row.get("comparison_key") != axis["comparison_key"]
        or row.get("origin") != "explicit_setting"
        or row.get("stability") not in {"stable", "core"}
        or row.get("polarity") not in {"positive", "negative"}
        or type(row.get("revision")) is not int or row["revision"] < 0
        or type(evidence) is not list or len(evidence) != 1
    ):
        return False
    ref = evidence[0]
    if (
        type(ref) is not dict or ref.get("document_id") != source_document_id
        or ref.get("document_version") != 1
        or ref.get("content_sha256") != PINNED_SHA256[COMMON[1]]
        or type(ref.get("input_id")) is not str or not ref["input_id"]
        or ref.get("document_name") != COMMON[1]
        or ref.get("line_start") != line_number or ref.get("line_end") != line_number
        or ref.get("text") not in {line, line.strip()}
    ):
        return False
    status, target = legacy._verified_target_span(row)
    if status != "verified" or target is None:
        return False
    support_id, start, end = target
    return (
        support_id.startswith(f"L{line_number}:A")
        and end <= len(line)
        and axis["source"]["source_quote"] in line[start:end]
    )


def _inventory(pending: list[dict[str, Any]], axis: dict[str, Any], fixture: Fixture,
               *, project_id: str, baseline_run_id: str,
               source_document_id: str) -> dict[str, Any]:
    matches = [row for row in pending if _candidate_matches(
        row, axis, fixture, project_id=project_id, baseline_run_id=baseline_run_id,
        source_document_id=source_document_id,
    )]
    ids = [row["id"] for row in matches]
    if any(type(value) is not str for value in ids) or len(set(ids)) != len(ids):
        _fail("candidate_identity_invalid", "candidate_inventory")
    return {
        "match_count": len(matches), "unique": len(matches) == 1,
        "candidate_revisions": {row["id"]: row["revision"] for row in matches},
    }


def _isolation_args(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    base_url = isolation._isolated_base_url(args.base_url)
    if urlsplit(base_url).port == 8080:
        _fail("isolation_base_url_invalid", "run_preflight")
    expected = isolation._expected_isolation(
        args.expected_eval_db_id, args.expected_eval_instance_id,
        args.expected_eval_db_path,
    )
    return base_url, expected


def _service_preflight(client: httpx.Client) -> tuple[dict[str, Any], str, dict[str, Any]]:
    code = legacy._code_state()
    if code.get("git_head") is None or code.get("worktree_clean") is not True:
        _fail("strict_worktree_not_clean", "code_preflight")
    health = legacy._request(client, "GET", "/health", "health")
    if type(health) is not dict:
        _fail("health_contract", "runtime_preflight")
    digest = legacy._service_preflight_gate(
        health, code, legacy._local_service_artifact_sha256(ROOT)
    )
    provenance = legacy._safe_character_runtime_provenance(
        health.get("runtime_provenance")
    )
    limits = provenance.get("character_consistency_limits") if provenance else None
    if (
        type(limits) is not dict
        or limits.get("scoped_axis_drift_v1") is not True
        or limits.get("signal_full_line_echo_v2") is not True
    ):
        _fail("scoped_review_runtime_not_enabled", "runtime_preflight")
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
    safe_runtime = legacy._runtime_summary(health)
    return code, digest, safe_runtime


def _check_runtime(client: httpx.Client, digest: str,
                   summary: dict[str, Any]) -> None:
    if summary.get("runtime_provenance_sha256") != digest:
        _fail("worker_runtime_provenance_mismatch", "runtime_postrun")
    health = legacy._request(client, "GET", "/health", "health_postrun")
    if type(health) is not dict or legacy._runtime_provenance_digest(
        health.get("runtime_provenance")
    ) != digest:
        _fail("api_runtime_provenance_changed", "runtime_postrun")


def _prepare(client: httpx.Client, fixture: Fixture, *, args: argparse.Namespace,
             expected: dict[str, Any]) -> dict[str, Any]:
    isolation._verify_live_isolation(
        client, expected, require_fresh=True, stage="isolation_preflight"
    )
    code, digest, safe_runtime = _service_preflight(client)
    prepared: list[dict[str, Any]] = []
    for case in fixture.manifest["cases"]:
        project = legacy._request(
            client, "POST", "/api/v1/projects", "project_create",
            json={
                "name": f"scoped-ooc-{case['case_id']}-{uuid4().hex[:8]}",
                "description": "Developer-visible isolated synthetic OOC diagnostic",
            },
        )
        if type(project) is not dict or type(project.get("id")) is not str:
            _fail("project_response_contract", "prepare")
        project_id = project["id"]
        if project_id in {row["project_id"] for row in prepared}:
            _fail("project_not_independent", "prepare")
        document_ids = {
            document["file"]: _upload(client, project_id, fixture, document)
            for document in fixture.manifest["common_documents_in_import_order"]
        }
        run_id = legacy._start_run(client, project_id, mode="baseline_build")
        run = legacy._wait_run(
            client, run_id, timeout_seconds=args.run_timeout_seconds
        )
        known = set(COMMON) | {_case_document_name(case)}
        summary = legacy._run_summary(client, run, known_documents=known)
        _check_runtime(client, digest, summary)
        admission = legacy._baseline_admission(summary)
        pending = legacy._list_pending(client, project_id)
        axis = _axis_for_case(fixture, case)
        inventory = _inventory(
            pending, axis, fixture, project_id=project_id,
            baseline_run_id=run_id, source_document_id=document_ids[COMMON[1]],
        )
        prepared.append({
            "case_id": case["case_id"], "axis_key": case["axis_key"],
            "project_id": project_id, "baseline_run_id": run_id,
            "document_ids": document_ids,
            "baseline_admitted": admission["admitted"],
            "baseline": legacy._public_run(summary),
            "baseline_checks": admission["checks"],
            "candidate_inventory": inventory,
        })
        isolation._verify_live_isolation(
            client, expected, require_fresh=False,
            stage="isolation_post_baseline",
        )
    return {
        "schema_version": SCHEMA, "phase": "prepare",
        "dataset_boundary": fixture.manifest["dataset_boundary"],
        "fixture_sha256": dict(PINNED_SHA256),
        "code_state": code, "runtime_provenance_sha256": digest,
        "runtime": safe_runtime, "evaluation_isolation": expected,
        "cases": prepared, "author_review_required": True,
        "quality_scored": False, "aggregate_accuracy": None,
    }


def _checkpoint(data: bytes, fixture: Fixture) -> dict[str, Any]:
    checkpoint = _json_object(data, "checkpoint_preflight")
    rows = checkpoint.get("cases")
    expected_cases = fixture.manifest["cases"]
    if (
        checkpoint.get("schema_version") != SCHEMA
        or checkpoint.get("phase") != "prepare"
        or checkpoint.get("fixture_sha256") != PINNED_SHA256
        or type(rows) is not list or len(rows) != 10
        or [row.get("case_id") for row in rows if type(row) is dict]
        != [case["case_id"] for case in expected_cases]
        or any(type(row) is not dict or row.get("axis_key") != case["axis_key"]
               for row, case in zip(rows, expected_cases))
        or len({row.get("project_id") for row in rows}) != 10
        or len({row.get("baseline_run_id") for row in rows}) != 10
    ):
        _fail("checkpoint_contract", "checkpoint_preflight")
    for row in rows:
        inventory = row.get("candidate_inventory")
        revisions = inventory.get("candidate_revisions") if type(inventory) is dict else None
        document_ids = row.get("document_ids")
        if (
            type(row.get("project_id")) is not str
            or type(row.get("baseline_run_id")) is not str
            or type(document_ids) is not dict or set(document_ids) != set(COMMON)
            or any(type(value) is not str or not value for value in document_ids.values())
            or len(set(document_ids.values())) != len(COMMON)
            or type(row.get("baseline_admitted")) is not bool
            or type(inventory) is not dict or type(revisions) is not dict
            or type(inventory.get("match_count")) is not int
            or inventory["match_count"] != len(revisions)
            or inventory.get("unique") is not (len(revisions) == 1)
            or any(type(key) is not str or type(value) is not int or value < 0
                   for key, value in revisions.items())
        ):
            _fail("checkpoint_case_contract", "checkpoint_preflight")
    return checkpoint


def _parse_decisions(data: bytes, fixture: Fixture,
                     checkpoint: dict[str, Any], checkpoint_hash: str) -> dict[str, dict[str, Any]]:
    decisions = _json_object(data, "author_decisions")
    choices = decisions.get("choices")
    if (
        set(decisions) != {"schema_version", "fixture_manifest_sha256",
                           "checkpoint_sha256", "reviewer_declaration", "choices"}
        or decisions["schema_version"] != DECISIONS_SCHEMA
        or decisions["fixture_manifest_sha256"] != PINNED_SHA256["manifest.json"]
        or decisions["checkpoint_sha256"] != checkpoint_hash
        or decisions["reviewer_declaration"] != REVIEWER_DECLARATION
        or type(choices) is not list or len(choices) != 10
    ):
        _fail("author_decisions_contract", "author_decisions")
    by_case = {row["case_id"]: row for row in checkpoint["cases"]}
    expected = {case["case_id"]: case for case in fixture.manifest["cases"]}
    parsed: dict[str, dict[str, Any]] = {}
    for choice in choices:
        if (
            type(choice) is not dict
            or set(choice) != {"case_id", "axis_key", "candidate_id", "decision", "axis_alignment"}
            or choice.get("case_id") not in expected
            or choice["case_id"] in parsed
            or choice.get("axis_key") != expected[choice["case_id"]]["axis_key"]
            or choice.get("decision") not in {"confirm", "defer"}
        ):
            _fail("author_choice_contract", "author_decisions")
        case = by_case[choice["case_id"]]
        inventory = case["candidate_inventory"]
        if choice["decision"] == "defer":
            if choice["candidate_id"] is not None or choice["axis_alignment"] is not None:
                _fail("author_defer_contract", "author_decisions")
        elif (
            case["baseline_admitted"] is not True
            or inventory["unique"] is not True
            or choice["candidate_id"] not in inventory["candidate_revisions"]
            or choice["axis_alignment"] not in {"same", "opposite"}
        ):
            _fail("author_confirmation_not_safe", "author_decisions")
        parsed[choice["case_id"]] = choice
    if set(parsed) != set(expected):
        _fail("author_choice_set_mismatch", "author_decisions")
    return parsed


def _candidate_detail(client: httpx.Client, project_id: str,
                      axis: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    result = legacy._request(
        client, "GET",
        f"/api/v1/projects/{project_id}/characters/"
        f"{quote(axis['character_key'], safe='')}/profile-candidates/"
        f"{quote(candidate_id, safe='')}", "candidate_detail",
    )
    if type(result) is not dict or result.get("id") != candidate_id:
        _fail("candidate_detail_contract", "author_confirmation")
    return result


def _confirm_candidate(client: httpx.Client, fixture: Fixture,
                       case: dict[str, Any], checkpoint_case: dict[str, Any],
                       choice: dict[str, Any]) -> dict[str, Any]:
    """Post an explicit decision only for a unique, still-frozen exact source."""
    axis = _axis_for_case(fixture, case)
    project_id = checkpoint_case["project_id"]
    candidate_id = choice["candidate_id"]
    baseline_run_id = checkpoint_case["baseline_run_id"]
    revision = checkpoint_case["candidate_inventory"]["candidate_revisions"][candidate_id]
    baseline = legacy._request(
        client, "GET", f"/api/v1/analysis-runs/{baseline_run_id}", "baseline_status",
    )
    if (
        type(baseline) is not dict or baseline.get("id") != baseline_run_id
        or baseline.get("project_id") != project_id
        or baseline.get("status") != "completed"
    ):
        _fail("baseline_status_changed", "author_confirmation")
    pending = legacy._list_pending(client, project_id)
    if _inventory(pending, axis, fixture, project_id=project_id,
                  baseline_run_id=baseline_run_id,
                  source_document_id=checkpoint_case["document_ids"][COMMON[1]]) != checkpoint_case["candidate_inventory"]:
        _fail("candidate_inventory_changed", "author_confirmation")
    candidate = _candidate_detail(client, project_id, axis, candidate_id)
    if (
        not _candidate_matches(candidate, axis, fixture, project_id=project_id,
                               baseline_run_id=baseline_run_id,
                               source_document_id=checkpoint_case["document_ids"][COMMON[1]])
        or candidate.get("revision") != revision
        or candidate.get("decisions") != []
    ):
        _fail("candidate_source_or_revision_changed", "author_confirmation")
    body = {
        "trait_type": axis["dimension"], "display_name": axis["display_name"],
        "definition": axis["definition"],
        "positive_proposition": axis["positive_proposition"],
        "comparison_key": axis["comparison_key"],
        "applicability_scope": axis["applicability_scope"],
    }
    created_axis = legacy._request(
        client, "POST", f"/api/v1/projects/{project_id}/character-trait-axes",
        "axis_create", json=body,
    )
    if (
        type(created_axis) is not dict
        or type(created_axis.get("id")) is not str
        or created_axis.get("project_id") != project_id
        or created_axis.get("version") != 1
        or created_axis.get("trait_type") != axis["dimension"]
        or created_axis.get("definition_sha256") != _digest(axis["definition"])
        or created_axis.get("positive_proposition_sha256")
        != _digest(axis["positive_proposition"])
        or created_axis.get("comparison_key") != axis["comparison_key"]
        or created_axis.get("applicability_scope_sha256")
        != _digest(axis["applicability_scope"])
    ):
        _fail("axis_create_contract", "author_confirmation")
    polarity = isolation._axis_polarity_from_choice(
        candidate["polarity"], choice["axis_alignment"]
    )
    if polarity is None:
        _fail("author_direction_invalid", "author_confirmation")
    decision_body = {
        "decision": "confirm", "expected_revision": revision,
        "comment": "开发者模拟作者确认：已核对冻结原文、对象、适用情境和轴方向；非真实作者确认。",
        "approved_axis_id": created_axis["id"], "expected_axis_version": 1,
        "axis_alignment": choice["axis_alignment"],
        "expected_axis_positive_proposition_sha256": created_axis[
            "positive_proposition_sha256"
        ],
        "expected_axis_applicability_scope_sha256": created_axis[
            "applicability_scope_sha256"
        ],
        "scope_applicability_confirmed": True,
    }
    key_material = [SCHEMA, PINNED_SHA256["manifest.json"], project_id,
                    baseline_run_id, case["case_id"], candidate_id, revision,
                    choice["axis_alignment"]]
    idempotency_key = "scoped-ooc-v1-" + _digest(json.dumps(
        key_material, ensure_ascii=False, separators=(",", ":")
    ))
    confirmed_result = legacy._request(
        client, "POST",
        f"/api/v1/projects/{project_id}/characters/"
        f"{quote(axis['character_key'], safe='')}/profile-candidates/"
        f"{quote(candidate_id, safe='')}/decisions",
        "author_decision", headers={"Idempotency-Key": idempotency_key},
        json=decision_body,
    )
    confirmed = confirmed_result.get("candidate") if type(confirmed_result) is dict else None
    if (
        type(confirmed) is not dict or type(confirmed_result.get("decision_id")) is not str
        or confirmed_result.get("deduplicated") is not False
        or confirmed.get("id") != candidate_id
        or confirmed.get("source_run_id") != baseline_run_id
        or confirmed.get("review_state") != "confirmed"
        or confirmed.get("revision") != revision + 1
        or confirmed.get("approved_axis_id") != created_axis["id"]
        or confirmed.get("approved_axis_version") != 1
        or confirmed.get("axis_alignment") != choice["axis_alignment"]
        or confirmed.get("axis_polarity") != polarity
        or confirmed.get("axis_positive_proposition_sha256")
        != created_axis["positive_proposition_sha256"]
    ):
        _fail("author_confirmation_contract", "author_confirmation")
    after = _candidate_detail(client, project_id, axis, candidate_id)
    chain = after.get("decisions")
    if (
        after.get("review_state") != "confirmed"
        or after.get("revision") != revision + 1
        or after.get("approved_axis_id") != created_axis["id"]
        or after.get("axis_polarity") != polarity
        or type(chain) is not list or len(chain) != 1
        or chain[0].get("id") != confirmed_result["decision_id"]
        or chain[0].get("decision") != "confirm"
        or chain[0].get("expected_revision") != revision
    ):
        _fail("author_confirmation_readback_mismatch", "author_confirmation")
    return {
        "confirmed": True, "candidate_id_sha256": _digest(candidate_id),
        "axis_id_sha256": _digest(created_axis["id"]),
        "axis_alignment": choice["axis_alignment"], "axis_polarity": polarity,
        "source_exact": True, "scope_applicability_explicit": True,
    }


def _safe_clue_response(value: Any, *, kind: str,
                        known_documents: set[str]) -> dict[str, Any]:
    if type(value) is not dict or type(value.get("items")) is not list:
        _fail("clue_response_contract", "draft_report")
    if type(value.get("truncated")) is not bool or len(value["items"]) > 64:
        _fail("clue_response_contract", "draft_report")
    if kind == "review":
        if (type(value.get("unavailable_count")) is not int
                or type(value.get("scan_limited")) is not bool):
            _fail("clue_response_contract", "draft_report")
    refs: list[list[dict[str, Any]]] = []
    for item in value["items"]:
        if type(item) is not dict:
            _fail("clue_response_contract", "draft_report")
        if kind == "review":
            refs.append(legacy._safe_visible_issue(item)["evidence_refs"])
        else:
            name, start, end = (item.get("document_name"), item.get("line_start"),
                                item.get("line_end"))
            if (name not in known_documents or type(start) is not int
                    or type(end) is not int or not 1 <= start <= end):
                _fail("provisional_source_contract", "draft_report")
            refs.append([{"document_name": name, "line_start": start,
                          "line_end": end}])
    return {
        "count": len(value["items"]), "refs": refs,
        "truncated": value["truncated"],
        **({"unavailable_count": value["unavailable_count"],
            "scan_limited": value["scan_limited"]} if kind == "review" else {}),
    }


def _review(client: httpx.Client, fixture: Fixture, checkpoint: dict[str, Any],
            decisions: dict[str, dict[str, Any]], *, args: argparse.Namespace,
            expected: dict[str, Any]) -> dict[str, Any]:
    if checkpoint.get("evaluation_isolation") != expected:
        _fail("checkpoint_isolation_mismatch", "review_preflight")
    isolation._verify_live_isolation(
        client, expected, require_fresh=False, stage="isolation_preflight"
    )
    code, digest, safe_runtime = _service_preflight(client)
    if (checkpoint.get("code_state") != code
            or checkpoint.get("runtime_provenance_sha256") != digest):
        _fail("checkpoint_runtime_changed", "review_preflight")
    reviewed: list[dict[str, Any]] = []
    for case, prepared in zip(fixture.manifest["cases"], checkpoint["cases"]):
        choice = decisions[case["case_id"]]
        row: dict[str, Any] = {
            "case_id": case["case_id"], "axis_key": case["axis_key"],
            "expected_axis_polarity": _axis_for_case(fixture, case)[
                "expected_direction_if_confirmed"
            ],
            "project_id_sha256": _digest(prepared["project_id"]),
            "baseline_run_id_sha256": _digest(prepared["baseline_run_id"]),
            "baseline_admitted": prepared["baseline_admitted"],
            "baseline_checks": prepared["baseline_checks"],
            "candidate_match_count": prepared["candidate_inventory"]["match_count"],
            "author_decision": choice["decision"], "author_confirmation": None,
            "draft": None, "draft_run_id_sha256": None,
            "formal_issues": None, "review_clues": None,
            "provisional_clues": None, "case_trace": None,
            "accepted_draft_observation_refs": None,
            "unassessed_reason": None,
        }
        if not prepared["baseline_admitted"]:
            row["unassessed_reason"] = "baseline_incomplete"
        elif choice["decision"] == "defer":
            row["unassessed_reason"] = "author_confirmation_deferred"
        elif not prepared["candidate_inventory"]["unique"]:
            row["unassessed_reason"] = "candidate_not_unique"
        else:
            row["author_confirmation"] = _confirm_candidate(
                client, fixture, case, prepared, choice
            )
            draft_id = _upload(client, prepared["project_id"], fixture, case, draft=True)
            draft_run_id = legacy._start_run(
                client, prepared["project_id"], mode="draft_review", target_id=draft_id
            )
            draft_run = legacy._wait_run(
                client, draft_run_id, timeout_seconds=args.run_timeout_seconds
            )
            known = set(COMMON) | {_case_document_name(case)}
            summary = legacy._run_summary(client, draft_run, known_documents=known)
            _check_runtime(client, digest, summary)
            row["draft_run_id_sha256"] = _digest(draft_run_id)
            row["draft"] = legacy._public_run(summary)
            row["case_trace"] = summary["case_trace"]
            row["accepted_draft_observation_refs"] = summary[
                "accepted_draft_observation_refs"
            ]
            row["formal_issues"] = summary["visible_issue_cases"]
            if summary.get("status") != "completed":
                row["unassessed_reason"] = "draft_run_not_completed"
            else:
                review_clues = legacy._request(
                    client, "GET", f"/api/v1/analysis-runs/{draft_run_id}/review-clues",
                    "review_clues",
                )
                provisional = legacy._request(
                    client, "GET", f"/api/v1/analysis-runs/{draft_run_id}/provisional-clues",
                    "provisional_clues",
                )
                row["review_clues"] = _safe_clue_response(
                    review_clues, kind="review", known_documents=known
                )
                row["provisional_clues"] = _safe_clue_response(
                    provisional, kind="provisional", known_documents=known
                )
                planned = summary.get("planned_chunks")
                processed = summary.get("processed_chunks")
                usage = summary.get("stage_usage")
                usage = usage if type(usage) is dict else {}
                if (summary.get("stage_outcome") != "completed"
                        or summary.get("material_coverage") != "complete"
                        or type(planned) is not int or planned <= 0
                        or type(processed) is not int or processed != planned
                        or type(usage.get("attempted_calls")) is not int
                        or usage["attempted_calls"] <= 0
                        or type(usage.get("input_tokens")) is not int
                        or usage["input_tokens"] <= 0
                        or summary.get("accepted_draft_observation_refs") is None
                        or row["review_clues"]["truncated"]
                        or row["review_clues"]["unavailable_count"] > 0
                        or row["review_clues"]["scan_limited"]
                        or row["provisional_clues"]["truncated"]):
                    row["unassessed_reason"] = "draft_coverage_or_report_incomplete"
        reviewed.append(row)
        isolation._verify_live_isolation(
            client, expected, require_fresh=False, stage="isolation_post_case"
        )
    return {
        "schema_version": SCHEMA, "phase": "review", "dataset_boundary": fixture.manifest[
            "dataset_boundary"
        ], "fixture_sha256": dict(PINNED_SHA256), "code_state": code,
        "runtime_provenance_sha256": digest, "runtime": safe_runtime,
        "evaluation_isolation": expected, "cases": reviewed,
        "author_review_provenance": "separate_decision_file_developer_simulation_not_independently_verified",
        "independent_human_annotation": False, "quality_scored": False,
        "aggregate_accuracy": None,
    }


def _source_refs_for_case(oracle_case: dict[str, Any],
                          case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(oracle_case["baseline"], 1):
        refs[f"B{index:02d}"] = {
            "role": "B", "document_name": source["document_name"],
            "line_start": source["line"], "line_end": source["line"],
        }
    for index, source in enumerate(oracle_case["current"], 1):
        refs[f"C{index:02d}"] = {
            "role": "C", "document_name": _case_document_name(case),
            "line_start": source["line"], "line_end": source["line"],
        }
    return refs


def _assess_case(oracle_case: dict[str, Any], case: dict[str, Any],
                 observed: dict[str, Any]) -> dict[str, Any]:
    expected = oracle_case["expected_when_confirmed_and_complete"]
    result: dict[str, Any] = {
        **observed,
        "oracle_kind": oracle_case["kind"],
        "expected_outcome": expected["outcome"],
        "expected_formal_issue_count": expected["formal_issue_count"],
        "assessment_state": "unassessed", "formal_count_matches_oracle": None,
        "citation_coverage": None, "scope_review_comparison": None,
        "clean_pass": None, "wrong_reason_flags": [],
    }
    if observed.get("unassessed_reason"):
        return result
    confirmation = observed.get("author_confirmation")
    if (
        type(confirmation) is not dict
        or confirmation.get("axis_polarity") != observed.get("expected_axis_polarity")
    ):
        result["unassessed_reason"] = "author_direction_disagrees_with_frozen_plan"
        return result
    draft = observed.get("draft")
    if (type(draft) is not dict or draft.get("status") != "completed"
            or draft.get("stage_outcome") != "completed"
            or draft.get("material_coverage") != "complete"
            or observed.get("accepted_draft_observation_refs") is None
            or type(observed.get("formal_issues")) is not list
            or type(observed.get("review_clues")) is not dict
            or type(observed.get("provisional_clues")) is not dict):
        result["unassessed_reason"] = "draft_incomplete_or_evidence_unavailable"
        return result
    candidate_hash = observed["author_confirmation"]["candidate_id_sha256"]
    traces = [row for row in observed["case_trace"]
              if type(row) is dict
              and row.get("confirmed_candidate_id_sha256") == candidate_hash]
    issues = [row for row in observed["formal_issues"]
              if type(row) is dict
              and row.get("confirmed_candidate_id_sha256") == candidate_hash]
    unrelated = len(observed["formal_issues"]) - len(issues)
    result["target_formal_issue_count"] = len(issues)
    result["other_character_drift_issue_count"] = unrelated
    result["review_clue_count"] = observed["review_clues"]["count"]
    result["provisional_clue_count"] = observed["provisional_clues"]["count"]
    result["formal_count_matches_oracle"] = len(issues) == expected["formal_issue_count"]
    result["assessment_state"] = "observed_developer_visible"
    result["clean_pass"] = False if expected.get("clean_pass") is False else None
    if unrelated:
        result["wrong_reason_flags"].append("unrelated_formal_issue_in_case_project")
    if len(traces) > 1:
        result["assessment_state"] = "unassessed"
        result["unassessed_reason"] = "ambiguous_target_trace"
        return result
    trace = traces[0] if traces else None
    result["target_trace"] = {
        key: trace.get(key) for key in (
            "prepare_reason", "review_outcome", "review_verdict", "final_outcome",
            "visible", "matched_observation_count", "matched_observation_refs",
            "citation_refs", "scoped_axis_review",
        )
    } if trace else None
    if expected["formal_issue_count"] > 0:
        if trace is None:
            result["assessment_state"] = "unassessed"
            result["unassessed_reason"] = "target_trace_missing"
            return result
        if (trace.get("review_outcome") != "completed"
                or trace.get("final_outcome") != "conflict"
                or trace.get("review_verdict") != "contradicts"):
            result["wrong_reason_flags"].append("expected_final_conflict_not_reached")
        expected_refs = _source_refs_for_case(oracle_case, case)
        required = expected.get("required_citations")
        citation_refs = trace.get("citation_refs")
        if type(required) is not list or type(citation_refs) is not list:
            result["assessment_state"] = "unassessed"
            result["unassessed_reason"] = "required_citation_trace_unavailable"
            return result
        required_rows = [expected_refs.get(handle) for handle in required]
        if any(row is None for row in required_rows):
            result["assessment_state"] = "unassessed"
            result["unassessed_reason"] = "oracle_citation_contract_invalid"
            return result
        result["citation_coverage"] = all(row in citation_refs for row in required_rows)
        issue_refs = [ref for issue in issues for ref in issue.get("evidence_refs", [])]
        result["formal_issue_evidence_coverage"] = all(
            {key: value for key, value in row.items() if key != "role"} in issue_refs
            for row in required_rows
        )
        if not result["citation_coverage"] or not result["formal_issue_evidence_coverage"]:
            result["wrong_reason_flags"].append("formal_citation_or_evidence_missing")
    elif issues:
        result["wrong_reason_flags"].append("unexpected_formal_conflict")
    if expected["outcome"] == "needs_confirmation_or_unverifiable":
        # Missing material is not a clean pass, even when the formal list is empty.
        result["clean_pass"] = False
        if not trace or trace.get("final_outcome") not in {
            "needs_confirmation", "unverifiable"
        }:
            result["wrong_reason_flags"].append("uncertain_material_not_marked_for_review")
    if expected["outcome"] == "single_behavior_clue_or_unverifiable":
        if trace and trace.get("final_outcome") == "conflict":
            result["wrong_reason_flags"].append("single_event_counted_as_repeated_behavior")
    expected_scope = oracle_case.get("applicability_if_both_current_observations_bound")
    expected_independence = oracle_case.get("independent_events")
    scoped = trace.get("scoped_axis_review") if trace else None
    if type(scoped) is dict and type(expected_scope) is list:
        by_handle = {row.get("citation"): row for row in scoped.get("observations", [])
                     if type(row) is dict}
        result["scope_review_comparison"] = {
            "observations": {
                row["citation"]: (
                    by_handle.get(row["citation"], {}).get("object_match")
                    == row["object_match"]
                    and by_handle.get(row["citation"], {}).get("situation_match")
                    == row["situation_match"]
                ) for row in expected_scope
            },
            "independent_events": scoped.get("independent_events")
            == expected_independence,
        }
        if not all(result["scope_review_comparison"]["observations"].values()):
            result["wrong_reason_flags"].append("object_or_situation_misread")
        if not result["scope_review_comparison"]["independent_events"]:
            result["wrong_reason_flags"].append("event_independence_misread")
    else:
        result["scope_review_comparison"] = None
    return result


def _seal_and_assess(report: dict[str, Any], fixture: Fixture) -> dict[str, Any]:
    """The first parse of the oracle happens after the caller closes HTTP."""
    oracle_bytes = (DATASET / "oracle.json").read_bytes()
    if _digest(oracle_bytes) != PINNED_SHA256["oracle.json"]:
        _fail("sealed_oracle_changed_after_inference", "sealed_assessment")
    oracle = _json_object(oracle_bytes, "sealed_assessment")
    oracle_cases = oracle.get("cases")
    manifest_cases = fixture.manifest["cases"]
    if (
        oracle.get("schema_version") != "character-ooc-scoped-dev-oracle-v1"
        or oracle.get("world_id") != fixture.manifest["world_id"]
        or type(oracle_cases) is not list or len(oracle_cases) != 10
        or [row.get("case_id") for row in oracle_cases if type(row) is dict]
        != [row["case_id"] for row in manifest_cases]
        or any(row.get("axis_key") != case["axis_key"]
               or row.get("draft_file") != case["draft_file"]
               for row, case in zip(oracle_cases, manifest_cases))
    ):
        _fail("sealed_oracle_contract", "sealed_assessment")
    report["cases"] = [
        _assess_case(gold, case, observed)
        for gold, case, observed in zip(oracle_cases, manifest_cases, report["cases"])
    ]
    report["oracle_opened_after_inference"] = True
    report["unassessed_count"] = sum(
        row["assessment_state"] == "unassessed" for row in report["cases"]
    )
    report["aggregate_accuracy"] = None
    report["quality_scored"] = False
    return report


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    try:
        output_path = legacy._resolve_output_json(args.output_json) if args.output_json else None
        if output_path is not None and output_path.exists():
            _fail("output_report_exists", "run_preflight")
        fixture = verify_fixture()
        if args.phase == "preflight":
            return {
                "schema_version": SCHEMA, "phase": "preflight",
                "fixture_sha256": dict(PINNED_SHA256),
                "cases": len(fixture.manifest["cases"]),
                "developer_visible": True, "oracle_parsed": False,
                "quality_scored": False, "aggregate_accuracy": None,
            }, 0
        if args.phase not in {"prepare", "review"}:
            _fail("phase_invalid", "run_preflight")
        if not args.allow_provider_call or output_path is None:
            _fail("explicit_provider_call_and_output_required", "run_preflight")
        base_url, expected = _isolation_args(args)
        checkpoint = None
        decisions = None
        if args.phase == "review":
            if not args.checkpoint_json or not args.author_decisions_json:
                _fail("checkpoint_and_author_decisions_required", "run_preflight")
            checkpoint_path = legacy._resolve_output_json(args.checkpoint_json)
            decisions_path = legacy._resolve_output_json(args.author_decisions_json)
            if checkpoint_path == decisions_path or output_path in {
                checkpoint_path, decisions_path
            }:
                _fail("report_path_collision", "run_preflight")
            checkpoint_bytes = checkpoint_path.read_bytes()
            checkpoint = _checkpoint(checkpoint_bytes, fixture)
            decisions = _parse_decisions(
                decisions_path.read_bytes(), fixture, checkpoint,
                _digest(checkpoint_bytes),
            )
        with httpx.Client(base_url=base_url, timeout=httpx.Timeout(30.0)) as client:
            if args.phase == "prepare":
                report = _prepare(client, fixture, args=args, expected=expected)
            else:
                report = _review(
                    client, fixture, checkpoint, decisions, args=args, expected=expected
                )
        if args.phase == "review":
            report = _seal_and_assess(report, fixture)
        return report, 0
    except Exception as exc:
        return {
            "schema_version": SCHEMA, "phase": args.phase,
            "quality_scored": False, "aggregate_accuracy": None,
            "failure": legacy._failure(exc),
        }, 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "prepare", "review"),
                        default="preflight")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--expected-eval-db-id")
    parser.add_argument("--expected-eval-instance-id")
    parser.add_argument("--expected-eval-db-path")
    parser.add_argument("--run-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--allow-provider-call", action="store_true")
    parser.add_argument("--checkpoint-json")
    parser.add_argument("--author-decisions-json")
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
