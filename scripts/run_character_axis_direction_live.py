"""First-stage real-model evaluation of explicit author-axis direction review.

The v2 story and oracle bytes remain immutable. `preflight` has no network or
model call. `prepare` requires an explicit provider-call opt-in and stops after
the baseline; `confirm` reads a separately authored decision file and has no
provider call. Neither phase claims draft or OOC quality.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import quote, urlsplit
from uuid import uuid4

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.character_trait_extraction import stable_trait_identity
from app.evaluation_isolation import (
    EvaluationIsolationUnavailable,
    SCHEMA_VERSION as ISOLATION_SCHEMA,
    database_path_sha256,
    instance_id_sha256,
    _uuid4,
)
from scripts import run_character_axis_live as legacy


DATASET = ROOT / "data" / "character-axis-direction-v1"
SOURCE_DATASET = ROOT / "data" / "character-axis-challenge-v2"
PINNED_MANIFEST_SHA256 = "3d0104601356d84ec30be70a483ad7c54d1302440502ae7ac68ea7cd22b172f3"
MANIFEST_SCHEMA = "character-axis-direction-manifest-v1"
PLAN_SCHEMA = "character-axis-direction-author-plan-v1"
DECISIONS_SCHEMA = "character-axis-direction-decisions-v1"
CHECKPOINT_SCHEMA = "character-axis-direction-checkpoint-v1"
REPORT_SCHEMA = "character-axis-direction-report-v1"
HUMAN_DECLARATION = (
    "I reviewed each actual candidate, its frozen source clause, and the axis "
    "positive proposition without consulting the scoring oracle."
)
CONFIRM_COMMENT = "Explicit author decision for developer-visible direction evaluation"
ISOLATION_IDENTITY_PATH = "/api/v1/evaluation/isolation-identity"
PROTOCOL = {
    "reviewer": "human_author",
    "show": [
        "frozen_target_clause", "candidate_raw_trait_key", "candidate_raw_polarity",
        "axis_positive_proposition", "candidate_evidence",
    ],
    "choices": ["same", "opposite", "uncertain"],
    "uncertain_action": "leave_pending",
    "alignment_source": "explicit_human_review_of_candidate_and_source",
}
@dataclass(frozen=True)
class DirectionSuite:
    source: legacy.VerifiedSuite
    plan: dict[str, Any]
    plan_sha256: str


@dataclass(frozen=True)
class DirectionFixture:
    manifest_sha256: str
    source_manifest_sha256: str
    suites: dict[str, DirectionSuite]


def _digest(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def _fail(code: str, stage: str) -> None:
    raise legacy.SafeFailure(code, stage)


def _json_file(path: Path, *, stage: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, ValueError):
        _fail("json_unavailable_or_invalid", stage)
    if type(value) is not dict:
        _fail("json_object_required", stage)
    return value


def _normalized(value: str) -> str:
    return " ".join(value.split())


def _isolated_base_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        _fail("isolation_base_url_invalid", "run_preflight")
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or port is None
        or port == 8000
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        _fail("isolation_base_url_invalid", "run_preflight")
    return value.rstrip("/")


def _expected_isolation(
    eval_db_id: str | None, instance_id: str | None, db_path: str | None,
) -> dict[str, Any]:
    if not _uuid4(eval_db_id) or not _uuid4(instance_id) or not db_path:
        _fail("isolation_expectation_required", "run_preflight")
    try:
        path = Path(db_path)
        if not path.is_absolute():
            _fail("isolation_expectation_invalid", "run_preflight")
        path_sha = database_path_sha256(path)
        instance_sha = instance_id_sha256(instance_id)
    except EvaluationIsolationUnavailable:
        _fail("isolation_expectation_invalid", "run_preflight")
    return {
        "schema_version": ISOLATION_SCHEMA,
        "isolated_sqlite": True,
        "eval_db_id": eval_db_id,
        "database_path_sha256": path_sha,
        "instance_id_sha256": instance_sha,
    }


def _verify_live_isolation(
    client: httpx.Client, expected: dict[str, Any], *,
    require_fresh: bool, stage: str,
) -> dict[str, Any]:
    observed = legacy._request(
        client, "GET", ISOLATION_IDENTITY_PATH, "evaluation_isolation"
    )
    if (
        type(observed) is not dict
        or set(observed) != set(expected) | {"fresh_for_prepare"}
        or any(observed.get(key) != value for key, value in expected.items())
        or type(observed.get("fresh_for_prepare")) is not bool
        or (require_fresh and observed["fresh_for_prepare"] is not True)
    ):
        _fail("evaluation_isolation_mismatch", stage)
    return observed


def _validate_plan(plan: dict[str, Any], source: legacy.VerifiedSuite) -> None:
    if (
        set(plan) != {"schema_version", "suite", "world_id", "review_protocol", "axes", "targets"}
        or plan["schema_version"] != PLAN_SCHEMA
        or plan["suite"] != source.name
        or plan["world_id"] != source.world_id
        or plan["review_protocol"] != PROTOCOL
        or type(plan["axes"]) is not list
        or type(plan["targets"]) is not list
    ):
        _fail("direction_plan_contract", "fixture_preflight")
    source_axes = {row["axis_key"]: row for row in source.plan["approved_axes"]}
    seen_axes: set[str] = set()
    for axis in plan["axes"]:
        if (
            type(axis) is not dict
            or set(axis) != {"axis_key", "positive_proposition"}
            or axis.get("axis_key") not in source_axes
            or axis["axis_key"] in seen_axes
            or type(axis.get("positive_proposition")) is not str
            or not 1 <= len(axis["positive_proposition"]) <= 200
            or axis["positive_proposition"] != _normalized(axis["positive_proposition"])
        ):
            _fail("direction_axis_contract", "fixture_preflight")
        seen_axes.add(axis["axis_key"])
    if seen_axes != set(source_axes):
        _fail("direction_axis_set_mismatch", "fixture_preflight")

    source_targets = {
        row["candidate_key"]: row for row in source.plan["candidate_decisions"]
    }
    seen_targets: set[str] = set()
    for target in plan["targets"]:
        if type(target) is not dict or set(target) != {
            "candidate_key", "source_document", "source_line", "target_quote",
            "expected_axis_polarity",
        }:
            _fail("direction_target_contract", "fixture_preflight")
        key = target.get("candidate_key")
        if key not in source_targets or key in seen_targets:
            _fail("direction_target_key", "fixture_preflight")
        original = source_targets[key]
        if (
            target["source_document"] != original["source_document"]
            or target["source_line"] != original["source_line"]
            or target["target_quote"] != original["source_quote"]
            or (
                type(target["expected_axis_polarity"]) is str
                and target["expected_axis_polarity"] in {"positive", "negative"}
            ) != (original["approved_axis_key"] is not None)
            or (
                original["approved_axis_key"] is None
                and target["expected_axis_polarity"] is not None
            )
        ):
            _fail("direction_target_source_mismatch", "fixture_preflight")
        source_line = source.files[target["source_document"]].decode("utf-8").splitlines()[
            target["source_line"] - 1
        ]
        if source_line.count(target["target_quote"]) != 1:
            _fail("direction_target_anchor_ambiguous", "fixture_preflight")
        seen_targets.add(key)
    if seen_targets != set(source_targets):
        _fail("direction_target_set_mismatch", "fixture_preflight")


def verify_fixture(
    dataset: Path = DATASET,
    *,
    source_dataset: Path = SOURCE_DATASET,
    expected_manifest_sha256: str = PINNED_MANIFEST_SHA256,
) -> DirectionFixture:
    try:
        manifest_bytes = (dataset / "manifest.json").read_bytes()
    except OSError:
        _fail("direction_manifest_unavailable", "fixture_preflight")
    if _digest(manifest_bytes) != expected_manifest_sha256:
        _fail("direction_manifest_hash_mismatch", "fixture_preflight")
    manifest = _json_file(dataset / "manifest.json", stage="fixture_preflight")
    if (
        set(manifest) != {
            "schema_version", "dataset_boundary", "source_dataset",
            "source_manifest_sha256", "suites",
        }
        or manifest["schema_version"] != MANIFEST_SCHEMA
        or manifest["dataset_boundary"] != {
            "developer_visible": True, "blind_holdout": False,
            "production_quality": False,
        }
        or manifest["source_dataset"] != "character-axis-challenge-v2"
        or manifest["source_manifest_sha256"] != legacy.PINNED_MANIFEST_SHA256_V2
        or type(manifest["suites"]) is not dict
        or set(manifest["suites"]) != set(legacy.SUITES)
    ):
        _fail("direction_manifest_contract", "fixture_preflight")
    # This reads oracle bytes only to check their existing v2 hashes. It does
    # not parse labels; no oracle object is available to model-facing code.
    source = legacy.verify_fixture(
        source_dataset,
        expected_manifest_sha256=manifest["source_manifest_sha256"],
    )
    suites: dict[str, DirectionSuite] = {}
    for name in legacy.SUITES:
        entry = manifest["suites"][name]
        original = source.suites[name]
        if (
            type(entry) is not dict
            or set(entry) != {"world_id", "case_count", "author_plan_sha256"}
            or entry["world_id"] != original.world_id
            or entry["case_count"] != original.case_count
            or type(entry["author_plan_sha256"]) is not str
            or legacy.SHA256.fullmatch(entry["author_plan_sha256"]) is None
        ):
            _fail("direction_suite_manifest_contract", "fixture_preflight")
        path = dataset / name / "author-plan.json"
        try:
            plan_bytes = path.read_bytes()
        except OSError:
            _fail("direction_plan_unavailable", "fixture_preflight")
        if _digest(plan_bytes) != entry["author_plan_sha256"]:
            _fail("direction_plan_hash_mismatch", "fixture_preflight")
        plan = _json_file(path, stage="fixture_preflight")
        _validate_plan(plan, original)
        suites[name] = DirectionSuite(original, plan, entry["author_plan_sha256"])
    return DirectionFixture(_digest(manifest_bytes), source.manifest_sha256, suites)


def _target_selectors(suite: DirectionSuite) -> dict[str, tuple[dict[str, Any], dict[str, Any]]]:
    originals = {row["candidate_key"]: row for row in suite.source.plan["candidate_decisions"]}
    return {
        row["candidate_key"]: (row, originals[row["candidate_key"]])
        for row in suite.plan["targets"]
    }


def _matches_target(
    row: dict[str, Any], target: dict[str, Any], selector: dict[str, Any],
    suite: DirectionSuite, *, project_id: str, baseline_run_id: str,
    required_state: str = "pending",
) -> bool:
    if (
        (required_state == "pending" and row.get("reviewable") is not True)
        or row.get("source_verified") is not True
        or row.get("review_state") != required_state
        or row.get("project_id") != project_id
        or row.get("source_run_id") != baseline_run_id
        or row.get("character_key") != selector["character_key"]
        or row.get("trait_type") != selector["trait_type"]
        or row.get("stability") != selector["stability"]
        or row.get("origin") != "explicit_setting"
        or not legacy._reference_has_line(
            row.get("evidence"), document=target["source_document"],
            line=target["source_line"], source_quote=target["target_quote"],
        )
    ):
        return False
    status, bound = legacy._verified_target_span(row)
    if status != "verified" or bound is None:
        return False
    _, start, end = bound
    line = suite.source.files[target["source_document"]].decode("utf-8").splitlines()[
        target["source_line"] - 1
    ]
    if end > len(line) or target["target_quote"] not in line[start:end]:
        return False
    if selector["approved_axis_key"] is None:
        trait_key = row.get("trait_key")
        return (
            row.get("polarity") == selector["polarity"]
            and isinstance(trait_key, str)
            and row.get("comparison_key") == stable_trait_identity(
                selector["trait_type"], trait_key, selector["key_object"] or ""
            )
        )
    # The v2 raw-polarity lock is deliberately omitted. A human must inspect
    # the actual raw key/direction and explicitly choose its axis alignment.
    return row.get("polarity") in {"positive", "negative"}


def _candidate_inventory(
    pending: list[dict[str, Any]], suite: DirectionSuite, *,
    project_id: str, baseline_run_id: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {}
    public: dict[str, Any] = {}
    for key, (target, selector) in _target_selectors(suite).items():
        matches = [
            row for row in pending if _matches_target(
                row, target, selector, suite,
                project_id=project_id, baseline_run_id=baseline_run_id,
            )
        ]
        if any(
            not isinstance(row.get("id"), str)
            or type(row.get("revision")) is not int
            or row["revision"] < 0
            for row in matches
        ):
            _fail("candidate_inventory_contract", "candidate_review")
        candidates[key] = matches
        public[key] = {
            "match_count": len(matches),
            "candidate_ids": [row["id"] for row in matches],
            "candidate_revisions": {
                row["id"]: row["revision"] for row in matches
            },
            "unique": len(matches) == 1,
        }
    return candidates, public


def _service_preflight(client: httpx.Client) -> tuple[dict[str, Any], str]:
    code = legacy._code_state()
    if code.get("git_head") is None or code.get("worktree_clean") is not True:
        _fail("strict_worktree_not_clean", "code_preflight")
    health = legacy._request(client, "GET", "/health", "health")
    if type(health) is not dict:
        _fail("health_contract", "runtime_preflight")
    digest = legacy._service_preflight_gate(
        health, code, legacy._local_service_artifact_sha256(ROOT)
    )
    safe_provenance = legacy._safe_character_runtime_provenance(
        health.get("runtime_provenance")
    )
    limits = safe_provenance["character_consistency_limits"] if safe_provenance else {}
    if not all(limits.get(key) is True for key in (
        legacy._CHARACTER_SIGNAL_SUPPORT_ID_KEY,
        legacy._CHARACTER_SIGNAL_SEMANTIC_SCOPE_KEY,
        legacy._CHARACTER_SIGNAL_SCOPE_REVIEW_KEY,
    )):
        _fail("precise_scope_review_not_enabled", "runtime_preflight")
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
    return code, digest


def prepare(
    client: httpx.Client, fixture: DirectionFixture, suite_name: str, *,
    timeout_seconds: float, expected_isolation: dict[str, Any],
) -> dict[str, Any]:
    suite = fixture.suites[suite_name]
    _verify_live_isolation(
        client, expected_isolation, require_fresh=True, stage="isolation_preflight"
    )
    code, runtime_digest = _service_preflight(client)
    project = legacy._request(
        client, "POST", "/api/v1/projects", "project_create",
        json={
            "name": f"axis-direction-{suite_name}-{uuid4().hex[:8]}",
            "description": "Developer-visible author-axis direction evaluation",
        },
    )
    if type(project) is not dict or not isinstance(project.get("id"), str):
        _fail("project_create_contract", "project_create")
    project_id = project["id"]
    for name, role in legacy.BASELINE_FILES:
        legacy._upload(client, project_id, suite.source, name, role, published=True)
    run_id = legacy._start_run(client, project_id, mode="baseline_build")
    run = legacy._wait_run(client, run_id, timeout_seconds=timeout_seconds)
    summary = legacy._run_summary(
        client, run,
        known_documents={name for name, _ in legacy.BASELINE_FILES} | {legacy.DRAFT_FILE},
    )
    admission = legacy._baseline_admission(summary)
    if summary.get("runtime_provenance_sha256") != runtime_digest:
        _fail("worker_runtime_provenance_mismatch", "baseline")
    post_health = legacy._request(client, "GET", "/health", "health_postrun")
    if legacy._runtime_provenance_digest(
        post_health.get("runtime_provenance") if type(post_health) is dict else None
    ) != runtime_digest:
        _fail("api_runtime_provenance_changed", "baseline")
    _verify_live_isolation(
        client, expected_isolation, require_fresh=False, stage="isolation_postrun"
    )
    pending = legacy._list_pending(client, project_id)
    _, inventory = _candidate_inventory(
        pending, suite, project_id=project_id, baseline_run_id=run_id,
    )
    return {
        "schema_version": CHECKPOINT_SCHEMA,
        "phase": "prepare",
        "suite": suite_name,
        "manifest_sha256": fixture.manifest_sha256,
        "source_manifest_sha256": fixture.source_manifest_sha256,
        "author_plan_sha256": suite.plan_sha256,
        "code_state": code,
        "runtime_provenance_sha256": runtime_digest,
        "evaluation_isolation": expected_isolation,
        "project_id": project_id,
        "baseline_run_id": run_id,
        "baseline_admitted": admission.get("admitted") is True,
        "baseline": legacy._public_run(summary),
        "candidate_inventory": inventory,
        "author_review_required": True,
        "independent_human_annotation": False,
        "quality_scored": False,
        "passed": False,
    }


def _load_decisions(
    path: Path, checkpoint: dict[str, Any], suite: DirectionSuite,
) -> dict[str, dict[str, Any]]:
    document = _json_file(path, stage="author_decisions")
    if (
        set(document) != {
            "schema_version", "project_id", "baseline_run_id", "reviewer_declaration",
            "choices",
        }
        or document["schema_version"] != DECISIONS_SCHEMA
        or document["project_id"] != checkpoint["project_id"]
        or document["baseline_run_id"] != checkpoint["baseline_run_id"]
        or document["reviewer_declaration"] != HUMAN_DECLARATION
        or type(document["choices"]) is not list
    ):
        _fail("author_decisions_contract", "author_decisions")
    result: dict[str, dict[str, Any]] = {}
    expected = set(_target_selectors(suite))
    used_ids: set[str] = set()
    for choice in document["choices"]:
        if (
            type(choice) is not dict
            or set(choice) != {"candidate_key", "candidate_id", "decision", "axis_alignment"}
            or choice.get("candidate_key") not in expected
            or choice["candidate_key"] in result
            or not isinstance(choice.get("candidate_id"), str)
            or choice["candidate_id"] in used_ids
            or choice.get("decision") not in {"confirm", "defer"}
        ):
            _fail("author_choice_contract", "author_decisions")
        selector = _target_selectors(suite)[choice["candidate_key"]][1]
        alignment = choice["axis_alignment"]
        if choice["decision"] == "defer":
            if alignment != "uncertain":
                _fail("author_uncertain_contract", "author_decisions")
        elif selector["approved_axis_key"] is None:
            if alignment is not None:
                _fail("non_axis_alignment_contract", "author_decisions")
        elif alignment not in {"same", "opposite"}:
            _fail("author_alignment_contract", "author_decisions")
        result[choice["candidate_key"]] = choice
        used_ids.add(choice["candidate_id"])
        _frozen_revision(checkpoint, choice["candidate_key"], choice["candidate_id"])
    if set(result) != expected:
        _fail("author_choice_set_mismatch", "author_decisions")
    return result


def _frozen_revision(checkpoint: dict[str, Any], key: str, candidate_id: str) -> int:
    inventory = checkpoint.get("candidate_inventory")
    entry = inventory.get(key) if type(inventory) is dict else None
    revisions = entry.get("candidate_revisions") if type(entry) is dict else None
    ids = entry.get("candidate_ids") if type(entry) is dict else None
    revision = revisions.get(candidate_id) if type(revisions) is dict else None
    if (
        type(ids) is not list
        or not all(isinstance(value, str) for value in ids)
        or len(ids) != len(set(ids))
        or type(revisions) is not dict
        or set(revisions) != set(ids)
        or not all(type(value) is int and value >= 0 for value in revisions.values())
        or type(entry.get("match_count")) is not int
        or entry.get("match_count") != len(ids)
        or entry.get("unique") is not (len(ids) == 1)
        or candidate_id not in ids
        or type(revision) is not int
        or revision < 0
    ):
        _fail("checkpoint_candidate_mismatch", "checkpoint_preflight")
    return revision


def _decision_key(
    fixture: DirectionFixture, checkpoint: dict[str, Any],
    key: str, choice: dict[str, Any], initial_revision: int,
) -> str:
    identity = [
        DECISIONS_SCHEMA, fixture.manifest_sha256, checkpoint["project_id"],
        checkpoint["baseline_run_id"], key, choice["candidate_id"],
        initial_revision, choice["decision"], choice["axis_alignment"],
    ]
    encoded = json.dumps(identity, ensure_ascii=False, separators=(",", ":"))
    return f"axisdir-v1-{_digest(encoded)}"


def _candidate_detail(
    client: httpx.Client, project_id: str, selector: dict[str, Any], candidate_id: str,
) -> dict[str, Any]:
    result = legacy._request(
        client, "GET",
        f"/api/v1/projects/{project_id}/characters/"
        f"{quote(selector['character_key'], safe='')}/profile-candidates/"
        f"{quote(candidate_id, safe='')}",
        "candidate_detail",
    )
    if type(result) is not dict or result.get("id") != candidate_id:
        _fail("candidate_detail_contract", "candidate_review")
    return result


def _axis_polarity_from_choice(raw: Any, alignment: Any) -> str | None:
    if raw not in {"positive", "negative"} or alignment not in {"same", "opposite"}:
        return None
    if alignment == "same":
        return raw
    return "negative" if raw == "positive" else "positive"


def _check_candidate_detail(
    row: dict[str, Any], target: dict[str, Any], selector: dict[str, Any],
    suite: DirectionSuite, choice: dict[str, Any], *, project_id: str,
    baseline_run_id: str, initial_revision: int,
) -> str:
    state = row.get("review_state")
    if state not in {"pending", "confirmed"} or not _matches_target(
        row, target, selector, suite, project_id=project_id,
        baseline_run_id=baseline_run_id, required_state=state,
    ):
        _fail("candidate_source_or_state_mismatch", "candidate_review")
    chain = row.get("decisions")
    if type(chain) is not list:
        _fail("candidate_review_chain_unavailable", "candidate_review")
    if state == "pending":
        if row.get("revision") != initial_revision or chain:
            _fail("pending_candidate_changed", "candidate_review")
        return state
    if choice["decision"] != "confirm":
        _fail("deferred_candidate_already_confirmed", "candidate_review")
    if row.get("revision") != initial_revision + 1 or len(chain) != 1:
        _fail("confirmed_candidate_review_chain_mismatch", "candidate_review")
    review = chain[0]
    if type(review) is not dict or not isinstance(review.get("id"), str):
        _fail("confirmed_candidate_review_chain_mismatch", "candidate_review")
    axis_key = selector["approved_axis_key"]
    axis_spec = next(
        (item for item in suite.plan["axes"] if item["axis_key"] == axis_key), None
    )
    proposition_sha = _digest(axis_spec["positive_proposition"]) if axis_spec else None
    polarity = (
        _axis_polarity_from_choice(row.get("polarity"), choice["axis_alignment"])
        if axis_key else None
    )
    if (
        review.get("decision") != "confirm"
        or review.get("expected_revision") != initial_revision
        or review.get("comment") != CONFIRM_COMMENT
        or review.get("axis_alignment") != (choice["axis_alignment"] if axis_key else None)
        or review.get("axis_polarity") != polarity
        or review.get("axis_positive_proposition_sha256") != proposition_sha
        or row.get("axis_alignment") != (choice["axis_alignment"] if axis_key else None)
        or row.get("axis_polarity") != polarity
        or row.get("axis_positive_proposition_sha256") != proposition_sha
        or (
            (not isinstance(row.get("approved_axis_id"), str)
             or row.get("approved_axis_version") != 1)
            if axis_key else (
                row.get("approved_axis_id") is not None
                or row.get("approved_axis_version") is not None
            )
        )
    ):
        _fail("confirmed_candidate_review_chain_mismatch", "candidate_review")
    return state


def _axes_for_confirmation(
    client: httpx.Client, project_id: str, suite: DirectionSuite,
    axis_keys: set[str], *, replay_axis_ids: dict[str, str],
) -> dict[str, dict[str, Any]]:
    if not axis_keys:
        return {}
    planned = {row["axis_key"]: row for row in suite.plan["axes"]}
    if not axis_keys <= set(planned) or not set(replay_axis_ids) <= axis_keys:
        _fail("selected_axis_set_invalid", "axis_create")
    existing = legacy._request(
        client, "GET", f"/api/v1/projects/{project_id}/character-trait-axes", "axis_list",
        params={"limit": 100, "offset": 0},
    )
    if (
        type(existing) is not dict
        or type(existing.get("items")) is not list
        or type(existing.get("total")) is not int
        or existing["total"] != len(existing["items"])
        or existing["total"] > 100
    ):
        _fail("axis_list_contract", "axis_create")
    source_axes = {row["axis_key"]: row for row in suite.source.plan["approved_axes"]}
    result: dict[str, dict[str, Any]] = {}
    create_later: list[str] = []
    for key in sorted(axis_keys):
        spec = planned[key]
        source_axis = source_axes[key]
        definition_sha = _digest(_normalized(source_axis["definition"]))
        proposition_sha = _digest(spec["positive_proposition"])
        matches = [
            row for row in existing["items"]
            if type(row) is dict and row.get("definition_sha256") == definition_sha
        ]
        if len(matches) > 1:
            _fail("axis_identity_ambiguous", "axis_create")
        if matches:
            axis = matches[0]
        else:
            if key in replay_axis_ids:
                _fail("replay_axis_missing", "axis_create")
            create_later.append(key)
            continue
        if (
            type(axis) is not dict
            or not isinstance(axis.get("id"), str)
            or axis.get("version") != 1
            or axis.get("trait_type") != "core_personality"
            or axis.get("definition_sha256") != definition_sha
            or axis.get("positive_proposition_sha256") != proposition_sha
            or (key in replay_axis_ids and axis["id"] != replay_axis_ids[key])
        ):
            _fail("axis_create_contract", "axis_create")
        result[key] = axis
    for key in create_later:
        spec = planned[key]
        source_axis = source_axes[key]
        definition_sha = _digest(_normalized(source_axis["definition"]))
        proposition_sha = _digest(spec["positive_proposition"])
        axis = legacy._request(
                client, "POST", f"/api/v1/projects/{project_id}/character-trait-axes",
                "axis_create", json={
                    "trait_type": "core_personality",
                    "display_name": source_axis["display_name"],
                    "definition": source_axis["definition"],
                    "positive_proposition": spec["positive_proposition"],
                },
        )
        if (
            type(axis) is not dict
            or not isinstance(axis.get("id"), str)
            or axis.get("version") != 1
            or axis.get("trait_type") != "core_personality"
            or axis.get("definition_sha256") != definition_sha
            or axis.get("positive_proposition_sha256") != proposition_sha
        ):
            _fail("axis_create_contract", "axis_create")
        result[key] = axis
    return result


def confirm(
    client: httpx.Client, fixture: DirectionFixture, checkpoint: dict[str, Any],
    decisions: dict[str, dict[str, Any]], *, expected_isolation: dict[str, Any],
) -> dict[str, Any]:
    suite_name = checkpoint.get("suite")
    if suite_name not in fixture.suites:
        _fail("checkpoint_suite_invalid", "checkpoint_preflight")
    suite = fixture.suites[suite_name]
    if checkpoint.get("evaluation_isolation") != expected_isolation:
        _fail("checkpoint_isolation_mismatch", "checkpoint_preflight")
    _verify_live_isolation(
        client, expected_isolation, require_fresh=False, stage="isolation_preflight"
    )
    code, runtime_digest = _service_preflight(client)
    if (
        checkpoint.get("schema_version") != CHECKPOINT_SCHEMA
        or checkpoint.get("phase") != "prepare"
        or checkpoint.get("manifest_sha256") != fixture.manifest_sha256
        or checkpoint.get("source_manifest_sha256") != fixture.source_manifest_sha256
        or checkpoint.get("author_plan_sha256") != suite.plan_sha256
        or checkpoint.get("baseline_admitted") is not True
        or checkpoint.get("code_state") != code
        or checkpoint.get("runtime_provenance_sha256") != runtime_digest
        or not isinstance(checkpoint.get("project_id"), str)
        or not isinstance(checkpoint.get("baseline_run_id"), str)
    ):
        _fail("checkpoint_contract", "checkpoint_preflight")
    project_id = checkpoint["project_id"]
    baseline_run_id = checkpoint["baseline_run_id"]
    baseline_run = legacy._request(
        client, "GET", f"/api/v1/analysis-runs/{baseline_run_id}", "baseline_status"
    )
    if (
        type(baseline_run) is not dict
        or baseline_run.get("id") != baseline_run_id
        or baseline_run.get("status") != "completed"
        or ("project_id" in baseline_run and baseline_run["project_id"] != project_id)
    ):
        _fail("baseline_status_changed", "checkpoint_preflight")
    baseline_summary = legacy._run_summary(
        client, baseline_run,
        known_documents={name for name, _ in legacy.BASELINE_FILES} | {legacy.DRAFT_FILE},
    )
    if (
        legacy._baseline_admission(baseline_summary).get("admitted") is not True
        or baseline_summary.get("runtime_provenance_sha256") != runtime_digest
    ):
        _fail("baseline_admission_changed", "checkpoint_preflight")
    selectors = _target_selectors(suite)
    if set(decisions) != set(selectors):
        _fail("author_choice_set_mismatch", "author_decisions")
    selected: dict[str, dict[str, Any]] = {}
    initial_revisions: dict[str, int] = {}
    previous_states: dict[str, str] = {}
    selected_axis_keys: set[str] = set()
    replay_axis_ids: dict[str, str] = {}
    # Validate the entire frozen selection before creating an axis or posting a
    # decision. A replayed confirmation must still have its exact review chain.
    for key, (target, selector) in selectors.items():
        choice = decisions[key]
        candidate_id = choice.get("candidate_id")
        if not isinstance(candidate_id, str):
            _fail("author_choice_contract", "author_decisions")
        initial_revision = _frozen_revision(checkpoint, key, candidate_id)
        row = _candidate_detail(client, project_id, selector, candidate_id)
        state = _check_candidate_detail(
            row, target, selector, suite, choice, project_id=project_id,
            baseline_run_id=baseline_run_id, initial_revision=initial_revision,
        )
        selected[key] = row
        initial_revisions[key] = initial_revision
        previous_states[key] = state
        axis_key = selector["approved_axis_key"]
        if choice["decision"] == "confirm" and axis_key is not None:
            selected_axis_keys.add(axis_key)
            if state == "confirmed":
                previous_id = replay_axis_ids.get(axis_key)
                approved_id = row["approved_axis_id"]
                if previous_id is not None and previous_id != approved_id:
                    _fail("replay_axis_conflict", "candidate_review")
                replay_axis_ids[axis_key] = approved_id
    axes = _axes_for_confirmation(
        client, project_id, suite, selected_axis_keys,
        replay_axis_ids=replay_axis_ids,
    )
    public: dict[str, Any] = {}
    confirmed_count = 0
    deferred_count = 0
    compared_count = 0
    matched_count = 0
    for key, (target, selector) in selectors.items():
        choice = decisions[key]
        candidate = selected[key]
        item: dict[str, Any] = {
            "candidate_id_sha256": _digest(candidate["id"]),
            "author_decision": choice["decision"],
            "author_axis_alignment": choice["axis_alignment"],
            "source_match_count": checkpoint["candidate_inventory"][key]["match_count"],
            "confirmed": False,
            "axis_polarity_matches_plan": None,
        }
        if choice["decision"] == "defer":
            deferred_count += 1
            public[key] = item
            continue
        body: dict[str, Any] = {
            "decision": "confirm", "expected_revision": initial_revisions[key],
            "comment": CONFIRM_COMMENT,
        }
        axis_key = selector["approved_axis_key"]
        if axis_key is not None:
            axis = axes[axis_key]
            if (
                previous_states[key] == "confirmed"
                and candidate["approved_axis_id"] != axis["id"]
            ):
                _fail("replay_axis_conflict", "candidate_review")
            body.update({
                "approved_axis_id": axis["id"],
                "expected_axis_version": 1,
                "axis_alignment": choice["axis_alignment"],
                "expected_axis_positive_proposition_sha256": axis[
                    "positive_proposition_sha256"
                ],
            })
        result = legacy._request(
            client, "POST",
            f"/api/v1/projects/{project_id}/characters/"
            f"{quote(selector['character_key'], safe='')}/profile-candidates/"
            f"{quote(candidate['id'], safe='')}/decisions",
            "candidate_decision",
            headers={"Idempotency-Key": _decision_key(
                fixture, checkpoint, key, choice, initial_revisions[key]
            )}, json=body,
        )
        confirmed = result.get("candidate") if type(result) is dict else None
        expected_polarity = (
            _axis_polarity_from_choice(candidate.get("polarity"), choice["axis_alignment"])
            if axis_key else None
        )
        if (
            type(confirmed) is not dict
            or not isinstance(result.get("decision_id"), str)
            or result.get("deduplicated") is not (previous_states[key] == "confirmed")
            or confirmed.get("id") != candidate["id"]
            or confirmed.get("review_state") != "confirmed"
            or confirmed.get("source_run_id") != baseline_run_id
            or confirmed.get("approved_axis_id") != (axes[axis_key]["id"] if axis_key else None)
            or confirmed.get("axis_alignment") != (choice["axis_alignment"] if axis_key else None)
            or confirmed.get("axis_polarity") != expected_polarity
            or confirmed.get("axis_positive_proposition_sha256") != (
                axes[axis_key]["positive_proposition_sha256"] if axis_key else None
            )
        ):
            _fail("candidate_confirmation_contract", "candidate_review")
        after = _candidate_detail(client, project_id, selector, candidate["id"])
        if (
            _check_candidate_detail(
                after, target, selector, suite, choice, project_id=project_id,
                baseline_run_id=baseline_run_id,
                initial_revision=initial_revisions[key],
            ) != "confirmed"
            or after["decisions"][0]["id"] != result["decision_id"]
            or after.get("approved_axis_id") != (axes[axis_key]["id"] if axis_key else None)
        ):
            _fail("candidate_confirmation_contract", "candidate_review")
        item["confirmed"] = True
        confirmed_count += 1
        if axis_key is not None:
            item["axis_polarity_matches_plan"] = (
                expected_polarity == target["expected_axis_polarity"]
            )
            compared_count += 1
            matched_count += item["axis_polarity_matches_plan"]
        public[key] = item
    post_health = legacy._request(client, "GET", "/health", "health_postrun")
    if legacy._runtime_provenance_digest(
        post_health.get("runtime_provenance") if type(post_health) is dict else None
    ) != runtime_digest:
        _fail("api_runtime_provenance_changed", "candidate_review")
    _verify_live_isolation(
        client, expected_isolation, require_fresh=False, stage="isolation_postrun"
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "phase": "confirm",
        "suite": suite_name,
        "manifest_sha256": fixture.manifest_sha256,
        "source_manifest_sha256": fixture.source_manifest_sha256,
        "author_plan_sha256": suite.plan_sha256,
        "code_state": code,
        "runtime_provenance_sha256": runtime_digest,
        "evaluation_isolation": expected_isolation,
        "project_id_sha256": _digest(project_id),
        "baseline_run_id_sha256": _digest(baseline_run_id),
        "author_review_provenance": "decision_file_self_attested_not_independently_verified",
        "independent_human_annotation": False,
        "candidate_inventory": checkpoint["candidate_inventory"],
        "decisions": public,
        "api_decision_integrity": True,
        "coverage": {
            "targets_total": len(selectors), "confirmed": confirmed_count,
            "deferred": deferred_count,
        },
        "plan_agreement": {
            "compared": compared_count, "matched": matched_count,
            "all_compared_match": (
                matched_count == compared_count if compared_count else None
            ),
        },
        "quality_scored": False,
        "passed": False,
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    try:
        if args.output_json:
            output_path = legacy._resolve_output_json(args.output_json)
            if output_path.exists():
                _fail("output_report_exists", "run_preflight")
        fixture = verify_fixture()
        if args.preflight_only or args.phase == "preflight":
            return {
                "schema_version": REPORT_SCHEMA,
                "phase": "preflight",
                "manifest_sha256": fixture.manifest_sha256,
                "source_manifest_sha256": fixture.source_manifest_sha256,
                "author_plan_sha256": {
                    name: suite.plan_sha256 for name, suite in fixture.suites.items()
                },
                "developer_visible": True,
                "blind_holdout": False,
                "preflight_verified": True,
                "quality_scored": False,
                "passed": False,
            }, 0
        if args.phase == "prepare" and (
            not args.allow_provider_call or not args.output_json
        ):
            _fail("explicit_provider_call_and_output_required", "run_preflight")
        base_url = _isolated_base_url(args.base_url)
        expected_isolation = _expected_isolation(
            args.expected_eval_db_id, args.expected_eval_instance_id,
            args.expected_eval_db_path,
        )
        if args.phase == "prepare":
            with httpx.Client(
                base_url=base_url, timeout=httpx.Timeout(30.0)
            ) as client:
                report = prepare(
                    client, fixture, args.suite,
                    timeout_seconds=args.run_timeout_seconds,
                    expected_isolation=expected_isolation,
                )
            return report, 0
        if args.phase == "confirm":
            if not args.output_json or not args.checkpoint_json or not args.author_decisions_json:
                _fail("checkpoint_and_author_decisions_required", "run_preflight")
            checkpoint = _json_file(
                legacy._resolve_output_json(args.checkpoint_json), stage="checkpoint_preflight"
            )
            if checkpoint.get("suite") != args.suite:
                _fail("checkpoint_suite_mismatch", "checkpoint_preflight")
            suite = fixture.suites[args.suite]
            decisions = _load_decisions(
                legacy._resolve_output_json(args.author_decisions_json), checkpoint, suite
            )
            with httpx.Client(
                base_url=base_url, timeout=httpx.Timeout(30.0)
            ) as client:
                report = confirm(
                    client, fixture, checkpoint, decisions,
                    expected_isolation=expected_isolation,
                )
            return report, 0 if report["api_decision_integrity"] else 1
        _fail("phase_invalid", "run_preflight")
    except Exception as exc:
        return {
            "schema_version": REPORT_SCHEMA,
            "phase": args.phase,
            "quality_scored": False,
            "passed": False,
            "failure": legacy._failure(exc),
        }, 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "prepare", "confirm"), default="preflight")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--suite", choices=legacy.SUITES, default="dev")
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
