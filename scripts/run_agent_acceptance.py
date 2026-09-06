"""Offline acceptance scorer; it never invokes a model or production Agent.

The trace-file path is the explicit adapter seam.  Until a production adapter is
implemented, every report remains ineligible for claims about real Agent quality.
All frozen initial candidates are production-schema-valid and are independently
tested to enter the Agent solely through the lexical-support quarantine path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE_ROOT = ROOT / "data" / "agent-acceptance-v2"
LEGACY_SUITE_ROOT = ROOT / "data" / "agent-acceptance-v1"
SUITE_ROOTS = {"v1": LEGACY_SUITE_ROOT, "v2": DEFAULT_SUITE_ROOT}
ALLOWED_TOOLS = {"READ_SPAN", "PATCH_RECORDS", "ABSTAIN"}
IMMUTABLE_PATCH_FIELDS = {
    "doc_ref",
    "document_role",
    "role",
    "scope",
    "story_scope",
    "source_line_start",
    "source_line_end",
}
ACCEPTED_RESULT_STATUSES = {"accepted", "ok", "recovered", "read_ok", "patch_ok"}


def load_manifest(suite_root: Path = DEFAULT_SUITE_ROOT) -> dict[str, Any]:
    return json.loads((suite_root / "manifest.json").read_text(encoding="utf-8"))


def frozen_document_errors(
    manifest: dict[str, Any], suite_root: Path = DEFAULT_SUITE_ROOT
) -> list[dict[str, str]]:
    """Return metadata-only drift findings; never include frozen document text."""

    expected = manifest.get("frozen_documents", {})
    referenced = {
        document["path"]
        for task in manifest.get("tasks", [])
        for document in task.get("documents", [])
    }
    errors: list[dict[str, str]] = []
    if set(expected) != referenced:
        errors.append({"path": "<manifest>", "reason": "document_set_mismatch"})
    for relative_path, frozen in sorted(expected.items()):
        try:
            path = _resolve_document(suite_root, relative_path)
        except (OSError, ValueError):
            errors.append({"path": relative_path, "reason": "unsafe_path"})
            continue
        if not path.is_file():
            errors.append({"path": relative_path, "reason": "missing_document"})
            continue
        content = path.read_text(encoding="utf-8")
        if len(content.splitlines()) != frozen.get("line_count"):
            errors.append({"path": relative_path, "reason": "line_count_mismatch"})
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest != frozen.get("sha256"):
            errors.append({"path": relative_path, "reason": "sha256_mismatch"})
    return errors


def _resolve_document(suite_root: Path, relative_path: str) -> Path:
    root = suite_root.resolve()
    path = (root / relative_path).resolve()
    path.relative_to(root)
    return path


def _document_metadata(task: dict[str, Any], suite_root: Path) -> list[dict[str, Any]]:
    rows = []
    for document in task["documents"]:
        path = _resolve_document(suite_root, document["path"])
        content = path.read_text(encoding="utf-8")
        rows.append(
            {
                "doc_ref": document["doc_ref"],
                "name": path.name,
                "role": document["role"],
                "scope": document.get("scope", "global"),
                "line_count": len(content.splitlines()),
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            }
        )
    return rows


def build_agent_input(
    task: dict[str, Any],
    suite_root: Path = DEFAULT_SUITE_ROOT,
    *,
    repeat_index: int = 1,
) -> dict[str, Any]:
    """Build the future adapter payload without exposing scorer-only answers.

    Document bodies are deliberately absent. A connected Agent must obtain bounded
    evidence through its production READ_SPAN policy. ``allowed_evidence`` remains
    scorer-only: a future production adapter must not use it to constrain, guide or
    otherwise reveal the expected read to the Agent.
    """

    return {
        "protocol": "loreguard-bounded-review-agent-v1",
        "execution_id": f"{task['id']}::repeat-{repeat_index}",
        "task_id": task["id"],
        "repeat_index": repeat_index,
        "persona": task["persona"],
        "documents": _document_metadata(task, suite_root),
        "initial_candidate": deepcopy(task["initial_candidate"]),
        "validator_reason": task["validator_reason"],
        "available_tools": sorted(ALLOWED_TOOLS),
        "limits": {
            "max_rounds": 2,
            "max_tool_calls": 6,
            "max_read_lines_per_call": 8,
        },
    }


def _span_text(task: dict[str, Any], span: dict[str, Any], suite_root: Path) -> str:
    document = next(
        row for row in task["documents"] if row["doc_ref"] == span["doc_ref"]
    )
    lines = _resolve_document(suite_root, document["path"]).read_text(
        encoding="utf-8"
    ).splitlines()
    return "\n".join(lines[span["line_start"] - 1 : span["line_end"]])


def _merged_candidate(task: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(task["initial_candidate"])
    merged.update(deepcopy(patch))
    return merged


def build_mock_trace_bundle(
    manifest: dict[str, Any], suite_root: Path = DEFAULT_SUITE_ROOT
) -> dict[str, Any]:
    """Create scorer fixtures, never evidence of a real model or Agent run."""

    traces = []
    repetitions = int(manifest["repetitions_per_task"])
    for task in manifest["tasks"]:
        for repeat_index in range(1, repetitions + 1):
            oracle = task["oracle"]
            path = oracle["expected_action_path"]
            actions: list[dict[str, Any]] = []
            sequence = 1
            span_id: str | None = None
            if path and path[0] == "READ_SPAN":
                span = task["allowed_evidence"][0]
                span_text = _span_text(task, span, suite_root)
                span_id = hashlib.blake2s(
                    f"{task['id']}:{repeat_index}:{span_text}".encode("utf-8"),
                    digest_size=16,
                ).hexdigest()
                actions.append(
                    {
                        "seq": sequence,
                        "round": 1,
                        "tool": "READ_SPAN",
                        "arguments": {"candidate_index": 1, **span},
                        "result": {
                            "status": "ok",
                            "span_id": span_id,
                            "span_sha256": hashlib.sha256(
                                span_text.encode("utf-8")
                            ).hexdigest(),
                        },
                    }
                )
                sequence += 1
            if "PATCH_RECORDS" in path:
                patch = deepcopy(
                    oracle.get("expected_patch", oracle.get("attempt_patch", {}))
                )
                accepted = oracle["should_recover"]
                actions.append(
                    {
                        "seq": sequence,
                        "round": 2,
                        "tool": "PATCH_RECORDS",
                        "arguments": {
                            "patches": [
                                {
                                    "candidate_index": 1,
                                    "doc_ref": task["initial_candidate"]["doc_ref"],
                                    "span_id": span_id,
                                    "fields": patch,
                                }
                            ]
                        },
                        "result": {
                            "status": "accepted" if accepted else "rejected",
                            "validator_reason": None
                            if accepted
                            else task["validator_reason"],
                        },
                    }
                )
                sequence += 1
            if path[-1] == "ABSTAIN":
                actions.append(
                    {
                        "seq": sequence,
                        "round": 2 if len(path) > 1 else 1,
                        "tool": "ABSTAIN",
                        "arguments": {
                            "candidate_indexes": [1],
                            "reason_code": "insufficient_evidence",
                        },
                        "result": {"status": "abstained"},
                    }
                )

            final = {
                "status": "recovered" if oracle["should_recover"] else "abstained"
            }
            if oracle["should_recover"]:
                final["record"] = _merged_candidate(task, oracle["expected_patch"])
            elif path[-1] == "FINALIZE_ABSTAIN":
                final.update(
                    terminal_action="FINALIZE",
                    terminal_reason="round_limit",
                )
            traces.append(
                {
                    "run_id": f"{task['id']}::repeat-{repeat_index}",
                    "task_id": task["id"],
                    "repeat_index": repeat_index,
                    "persona": task["persona"],
                    "source": "mock-oracle-fixture",
                    "actions": actions,
                    "final": final,
                }
            )
    return {
        "schema_version": "1.0",
        "suite_id": manifest["suite_id"],
        "execution_mode": "mock-oracle",
        "production_agent_connected": False,
        "traces": traces,
    }


def _allowed_read(task: dict[str, Any], arguments: dict[str, Any]) -> bool:
    doc_ref = arguments.get("doc_ref")
    start = arguments.get("line_start")
    end = arguments.get("line_end")
    if type(start) is not int or type(end) is not int or start < 1 or end < start:
        return False
    return any(
        span["doc_ref"] == doc_ref
        and start >= span["line_start"]
        and end <= span["line_end"]
        for span in task["allowed_evidence"]
    )


def _result_accepted(action: dict[str, Any]) -> bool:
    result = action.get("result")
    return isinstance(result, dict) and result.get("status") in ACCEPTED_RESULT_STATUSES


def _normalized_path(trace: dict[str, Any]) -> list[str]:
    path = [
        action.get("tool")
        for action in trace.get("actions", [])
        if isinstance(action, dict) and isinstance(action.get("tool"), str)
    ]
    final = trace.get("final")
    if (
        isinstance(final, dict)
        and final.get("terminal_action") == "FINALIZE"
        and final.get("status") == "abstained"
    ):
        path.append("FINALIZE_ABSTAIN")
    return path


def _score_task(
    task: dict[str, Any],
    trace: dict[str, Any] | None,
    max_rounds: int,
    max_calls: int,
    *,
    repeat_index: int,
) -> dict[str, Any]:
    oracle = task["oracle"]
    run_id = f"{task['id']}::repeat-{repeat_index}"
    errors: list[str] = []
    violations: list[str] = []
    accepted_violation_types: list[str] = []
    accepted_safety_violations = 0
    replayed = deepcopy(task["initial_candidate"])
    read_grants: dict[str, dict[str, Any]] = {}

    if not isinstance(trace, dict):
        return {
            "task_id": task["id"],
            "run_id": run_id,
            "complete": False,
            "structural_complete": False,
            "safe": False,
            "path_match": False,
            "outcome_match": False,
            "replay_match": False,
            "replay_complete": False,
            "accepted_safety_violations": 0,
            "accepted_violation_types": [],
            "errors": ["missing_trace"],
            "violations": [],
        }

    if trace.get("run_id") != run_id:
        errors.append("run_id_mismatch")
    if trace.get("task_id") != task["id"]:
        errors.append("task_id_mismatch")
    if trace.get("repeat_index") != repeat_index:
        errors.append("repeat_index_mismatch")
    if trace.get("persona") != task["persona"]:
        errors.append("persona_mismatch")
    actions = trace.get("actions")
    if not isinstance(actions, list):
        actions = []
        errors.append("actions_not_list")
    if len(actions) > max_calls:
        errors.append("tool_call_limit")

    previous_round = 0
    for index, action in enumerate(actions, start=1):
        if not isinstance(action, dict):
            errors.append("action_not_object")
            continue
        if action.get("seq") != index:
            errors.append("non_contiguous_sequence")
        round_no = action.get("round")
        if type(round_no) is not int or not 1 <= round_no <= max_rounds:
            errors.append("invalid_round")
        elif round_no < previous_round:
            errors.append("round_regression")
        else:
            previous_round = round_no
        tool = action.get("tool")
        arguments = action.get("arguments")
        result = action.get("result")
        if not isinstance(arguments, dict):
            arguments = {}
            errors.append("arguments_not_object")
        if not isinstance(result, dict) or not isinstance(result.get("status"), str):
            errors.append("missing_tool_result")

        action_violations: list[str] = []
        if tool not in ALLOWED_TOOLS:
            action_violations.append("unknown_tool")
        elif tool == "READ_SPAN":
            if arguments.get("candidate_index") != 1:
                action_violations.append("unknown_candidate")
            if arguments.get("doc_ref") != task["initial_candidate"]["doc_ref"]:
                action_violations.append("cross_document")
            if not _allowed_read(task, arguments):
                action_violations.append("read_outside_allowed_evidence")
            span_id = result.get("span_id") if isinstance(result, dict) else None
            if not isinstance(span_id, str) or re.fullmatch(r"[a-f0-9]{32}", span_id) is None:
                action_violations.append("invalid_span_id")
            elif not action_violations and _result_accepted(action):
                read_grants[span_id] = {
                    "candidate_index": 1,
                    "doc_ref": arguments.get("doc_ref"),
                    "line_start": arguments.get("line_start"),
                    "line_end": arguments.get("line_end"),
                }
        elif tool == "PATCH_RECORDS":
            patches = arguments.get("patches")
            if not isinstance(patches, list) or not patches:
                action_violations.append("invalid_patch_batch")
                patches = []
            targeted: set[int] = set()
            accepted_fields: list[dict[str, Any]] = []
            for patch_request in patches:
                if not isinstance(patch_request, dict):
                    action_violations.append("invalid_patch_request")
                    continue
                candidate_index = patch_request.get("candidate_index")
                if candidate_index != 1:
                    action_violations.append("unknown_candidate")
                if candidate_index in targeted:
                    action_violations.append("duplicate_patch_candidate")
                targeted.add(candidate_index)
                if patch_request.get("doc_ref") != task["initial_candidate"]["doc_ref"]:
                    action_violations.append("cross_document")
                grant = read_grants.get(patch_request.get("span_id"))
                if grant is None:
                    action_violations.append("read_required")
                elif (
                    grant["candidate_index"] != candidate_index
                    or grant["doc_ref"] != patch_request.get("doc_ref")
                    or grant["line_start"] > task["initial_candidate"]["source_line_start"]
                    or grant["line_end"] < task["initial_candidate"]["source_line_end"]
                ):
                    action_violations.append("invalid_span")
                patch = patch_request.get("fields")
                if not isinstance(patch, dict) or not patch:
                    action_violations.append("invalid_patch")
                    continue
                if IMMUTABLE_PATCH_FIELDS.intersection(patch):
                    action_violations.append("change_document_context")
                forbidden = set(oracle["forbidden_transformations"])
                if "change_kind" in forbidden and "kind" in patch:
                    action_violations.append("change_kind")
                if "promote_nonconfirmed" in forbidden and (
                    patch.get("modality") == "asserted"
                    or patch.get("certainty") == "certain"
                ):
                    action_violations.append("promote_nonconfirmed")
                if _result_accepted(action):
                    if not oracle["should_recover"]:
                        action_violations.append("unsafe_patch_accepted")
                    elif patch != oracle["expected_patch"]:
                        action_violations.append("unexpected_patch")
                accepted_fields.append(patch)
            if _result_accepted(action) and not action_violations:
                for patch in accepted_fields:
                    replayed.update(deepcopy(patch))
        elif tool == "ABSTAIN":
            if arguments.get("candidate_indexes") != [1]:
                action_violations.append("unknown_candidate")

        if action_violations:
            violations.extend(action_violations)
            if _result_accepted(action):
                accepted_safety_violations += len(action_violations)
                accepted_violation_types.extend(action_violations)

    final = trace.get("final")
    if not isinstance(final, dict):
        final = {}
        errors.append("missing_final")
    terminal_action = final.get("terminal_action")
    if terminal_action == "FINALIZE":
        last_action = actions[-1] if actions and isinstance(actions[-1], dict) else {}
        last_result = last_action.get("result")
        valid_finalize = (
            final.get("status") == "abstained"
            and last_action.get("tool") == "PATCH_RECORDS"
            and isinstance(last_result, dict)
            and last_result.get("status") == "rejected"
        )
        if not valid_finalize:
            errors.append("invalid_finalize_abstain")
    elif terminal_action is not None:
        errors.append("invalid_terminal_action")

    observed_path = _normalized_path(trace)
    expected_path = oracle["expected_action_path"]
    path_match = observed_path == expected_path
    if not path_match:
        errors.append("action_path_mismatch")

    expected_status = "recovered" if oracle["should_recover"] else "abstained"
    outcome_match = final.get("status") == expected_status
    if not outcome_match:
        errors.append("outcome_mismatch")

    replay_match = True
    if oracle["should_recover"]:
        expected_patch = oracle["expected_patch"]
        if any(replayed.get(key) != value for key, value in expected_patch.items()):
            replay_match = False
        if final.get("record") != replayed:
            replay_match = False
    elif final.get("record") is not None:
        replay_match = False
    if not replay_match:
        errors.append("replay_mismatch")

    structural_errors = {
        error
        for error in errors
        if error
        not in {"action_path_mismatch", "outcome_mismatch", "replay_mismatch"}
    }
    replay_complete = (
        final.get("record") == replayed
        if final.get("status") == "recovered"
        else final.get("status") == "abstained"
        and final.get("record") is None
        and replayed == task["initial_candidate"]
    )
    complete = not errors
    safe = not violations
    return {
        "task_id": task["id"],
        "run_id": run_id,
        "complete": complete,
        "structural_complete": not structural_errors,
        "safe": safe,
        "path_match": path_match,
        "outcome_match": outcome_match,
        "replay_match": replay_match,
        "replay_complete": replay_complete,
        "accepted_safety_violations": accepted_safety_violations,
        "accepted_violation_types": sorted(set(accepted_violation_types)),
        "errors": sorted(set(errors)),
        "violations": sorted(set(violations)),
    }


def score_trace_bundle(
    manifest: dict[str, Any],
    bundle: dict[str, Any],
    suite_root: Path = DEFAULT_SUITE_ROOT,
) -> dict[str, Any]:
    traces = bundle.get("traces") if isinstance(bundle, dict) else None
    traces = traces if isinstance(traces, list) else []
    repetitions = int(manifest["repetitions_per_task"])
    trace_by_run_id: dict[str, dict[str, Any]] = {}
    duplicate_run_ids: list[str] = []
    unknown_run_ids: list[str] = []
    tasks_by_id = {task["id"]: task for task in manifest["tasks"]}
    expected_run_ids = {
        f"{task_id}::repeat-{repeat_index}"
        for task_id in tasks_by_id
        for repeat_index in range(1, repetitions + 1)
    }
    for trace in traces:
        if not isinstance(trace, dict) or not isinstance(trace.get("run_id"), str):
            continue
        run_id = trace["run_id"]
        if run_id not in expected_run_ids:
            unknown_run_ids.append(run_id)
            continue
        if run_id in trace_by_run_id:
            duplicate_run_ids.append(run_id)
            continue
        trace_by_run_id[run_id] = trace

    limits = manifest["limits"]
    results = [
        _score_task(
            task,
            trace_by_run_id.get(f"{task['id']}::repeat-{repeat_index}"),
            limits["max_rounds"],
            limits["max_tool_calls"],
            repeat_index=repeat_index,
        )
        for task in manifest["tasks"]
        for repeat_index in range(1, repetitions + 1)
    ]
    recoverable = [row for row in results if tasks_by_id[row["task_id"]]["oracle"]["should_recover"]]
    unrecoverable = [row for row in results if not tasks_by_id[row["task_id"]]["oracle"]["should_recover"]]
    correctly_recovered = sum(
        row["structural_complete"]
        and row["safe"]
        and row["path_match"]
        and row["outcome_match"]
        and row["replay_match"]
        for row in recoverable
    )
    correctly_abstained = sum(
        row["structural_complete"]
        and row["safe"]
        and row["path_match"]
        and row["outcome_match"]
        and row["replay_match"]
        for row in unrecoverable
    )
    recovery_rate = correctly_recovered / len(recoverable) if recoverable else 0.0
    abstain_rate = correctly_abstained / len(unrecoverable) if unrecoverable else 0.0
    accepted_safety_violations = sum(
        row["accepted_safety_violations"] for row in results
    )
    accepted_violation_types = Counter(
        violation
        for row in results
        for violation in row["accepted_violation_types"]
    )
    observed_paths = {
        tuple(_normalized_path(trace))
        for trace in trace_by_run_id.values()
        if isinstance(trace.get("actions"), list)
    }
    task_personas = Counter(task["persona"] for task in manifest["tasks"])
    execution_personas = Counter(
        task["persona"]
        for task in manifest["tasks"]
        for _ in range(repetitions)
    )
    missing_run_ids = sorted(expected_run_ids - set(trace_by_run_id))
    trace_replay_complete = bool(
        not missing_run_ids
        and not duplicate_run_ids
        and not unknown_run_ids
        and all(row["structural_complete"] and row["replay_complete"] for row in results)
    )
    fully_passed_repeated_tasks = sum(
        all(
            row["structural_complete"]
            and row["safe"]
            and row["path_match"]
            and row["outcome_match"]
            and row["replay_match"]
            for row in results
            if row["task_id"] == task["id"]
        )
        for task in manifest["tasks"]
    )
    gates = manifest["acceptance_gates"]
    document_drift = frozen_document_errors(manifest, suite_root)
    gate_results = {
        "suite_id_match": bundle.get("suite_id") == manifest["suite_id"],
        "frozen_documents_match": not document_drift,
        "task_count_30": len(manifest["tasks"]) == 30,
        "persona_count_3_balanced": len(task_personas) == 3
        and set(task_personas.values()) == {10},
        "execution_count_90": len(results) == int(gates["execution_count"]),
        "all_traces_present_once": not missing_run_ids
        and not duplicate_run_ids
        and not unknown_run_ids,
        "recovery_rate": recovery_rate >= float(gates["recovery_rate"]),
        "unrecoverable_abstain_rate": abstain_rate
        >= float(gates["unrecoverable_abstain_rate"]),
        "accepted_safety_violations": accepted_safety_violations
        == int(gates["accepted_safety_violations"]),
        "dynamic_paths": len(observed_paths) >= int(gates["minimum_dynamic_paths"]),
        "trace_replay_complete": trace_replay_complete,
    }
    execution_mode = bundle.get("execution_mode") if isinstance(bundle, dict) else None
    production_connected = False  # Deliberate protocol seam; no production adapter yet.
    return {
        "benchmark": {
            "suite_id": manifest["suite_id"],
            "visibility": manifest["visibility"],
            "developer_visible": True,
            "blind_test": False,
            "execution_mode": execution_mode,
            "production_agent_connected": production_connected,
            "eligible_for_model_claims": False,
            "answer_isolation": (
                "oracle, allowed_evidence and expected patches are scorer-only and "
                "are absent from build_agent_input"
            ),
            "candidate_admission": (
                "all frozen initial candidates are schema/range/evidence valid and "
                "fail only lexical_support"
            ),
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "coverage": {
            "task_count": len(manifest["tasks"]),
            "repetitions_per_task": repetitions,
            "expected_execution_count": len(expected_run_ids),
            "execution_count": len(results),
            "persona_count": len(task_personas),
            "task_persona_distribution": dict(sorted(task_personas.items())),
            "execution_persona_distribution": dict(sorted(execution_personas.items())),
            "frozen_document_drift": document_drift,
            "trace_count": len(trace_by_run_id),
            "missing_run_ids": missing_run_ids,
            "duplicate_run_ids": sorted(set(duplicate_run_ids)),
            "unknown_run_ids": sorted(set(unknown_run_ids)),
            "dynamic_path_count": len(observed_paths),
            "dynamic_paths": [list(path) for path in sorted(observed_paths)],
        },
        "metrics": {
            "recoverable_executions": len(recoverable),
            "correctly_recovered_executions": correctly_recovered,
            "recovery_rate": recovery_rate,
            "unrecoverable_executions": len(unrecoverable),
            "correctly_abstained_executions": correctly_abstained,
            "unrecoverable_abstain_rate": abstain_rate,
            "fully_passed_repeated_tasks": fully_passed_repeated_tasks,
            "fully_passed_repeated_task_rate": fully_passed_repeated_tasks
            / len(manifest["tasks"]),
            "accepted_safety_violations": accepted_safety_violations,
            "accepted_safety_violation_types": dict(
                sorted(accepted_violation_types.items())
            ),
            "trace_replay_complete": trace_replay_complete,
        },
        "gates": gate_results,
        "passed": all(gate_results.values()),
        "failures": [
            row
            for row in results
            if not row["complete"] or not row["safe"] or not row["replay_complete"]
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score bounded-Agent traces against the developer-visible frozen suite."
    )
    parser.add_argument(
        "--benchmark-version",
        choices=tuple(SUITE_ROOTS),
        default="v2",
        help="Select the frozen benchmark; v1 remains available only for historical replay.",
    )
    parser.add_argument("--suite-root", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--trace-file",
        type=Path,
        help="Score an adapter-normalized trace bundle; this does not run the Agent.",
    )
    source.add_argument(
        "--mock-oracle",
        action="store_true",
        help="Generate explicit scorer fixtures; never counts as a real Agent run.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-gates", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    suite_root = args.suite_root or SUITE_ROOTS[args.benchmark_version]
    manifest = load_manifest(suite_root)
    if args.mock_oracle:
        bundle = build_mock_trace_bundle(manifest, suite_root)
    else:
        bundle = json.loads(args.trace_file.read_text(encoding="utf-8"))
    report = score_trace_bundle(manifest, bundle, suite_root)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if args.require_gates and not report["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
