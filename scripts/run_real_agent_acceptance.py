from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.chunking import chunk_document
from app.config import Settings
from app.domain import directive_fingerprint
from app.model_extractor import ModelEnhancedExtractor, RECORD_ADAPTER, _stable_raw_hash
from app.pipeline import DocumentInput, _semantic_class
from app.provider import (
    ModelResult,
    OpenAICompatibleProvider,
    ProviderCallTelemetry,
    RetryPolicy,
    safe_thinking_configuration,
)
from app.semantic_quality import assess_directive
from app.review_agent import _sha256_text
from scripts.run_agent_acceptance import (
    DEFAULT_SUITE_ROOT,
    frozen_document_errors,
    load_manifest,
)


PILOT_TASK_IDS = (
    "nw-01-location-literal",
    "gp-09-cross-branch-merge",
    "ed-07-wrong-chapter-location",
)
ALLOWED_TOOLS = {"READ_SPAN", "PATCH_RECORDS", "ABSTAIN"}
SAFE_FINALS = {"accepted", "abstained", "rejected", "continue"}
IMMUTABLE_PATCH_FIELDS = {
    "doc_ref",
    "document_role",
    "role",
    "scope",
    "story_scope",
    "source_line_start",
    "source_line_end",
    "kind",
}
EXECUTION_TASK_KEYS = {
    "id",
    "persona",
    "documents",
    "initial_candidate",
    "validator_reason",
}
FORBIDDEN_REPORT_KEYS = {
    "api_key",
    "base_url",
    "endpoint",
    "messages",
    "prompt",
    "response",
    "response_text",
    "raw_response",
    "raw_candidate",
    "content",
    "text",
    "patch",
    "expected_patch",
    "attempt_patch",
    "allowed_evidence",
    "oracle",
    "document_text",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    payload = value if isinstance(value, bytes) else _canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_document(suite_root: Path, relative_path: str) -> Path:
    candidate = (suite_root / relative_path).resolve()
    root = suite_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("document path escapes the frozen suite") from exc
    return candidate


def sanitize_execution_task(task: dict[str, Any]) -> dict[str, Any]:
    """Return the only manifest fields visible while the production path runs."""

    sanitized = {
        "id": str(task["id"]),
        "persona": str(task["persona"]),
        "documents": [
            {
                "doc_ref": str(document["doc_ref"]),
                "path": str(document["path"]),
                "role": str(document["role"]),
                "scope": str(document["scope"]),
            }
            for document in task["documents"]
        ],
        "initial_candidate": copy.deepcopy(task["initial_candidate"]),
        "validator_reason": str(task["validator_reason"]),
    }
    if set(sanitized) != EXECUTION_TASK_KEYS:
        raise AssertionError("execution task allowlist drifted")
    return sanitized


def _document_for_candidate(
    execution_task: dict[str, Any], suite_root: Path
) -> tuple[DocumentInput, str]:
    doc_ref = str(execution_task["initial_candidate"]["doc_ref"])
    matching = [doc for doc in execution_task["documents"] if doc["doc_ref"] == doc_ref]
    if len(matching) != 1:
        raise ValueError("candidate document reference is not unique")
    metadata = matching[0]
    path = _resolve_document(suite_root, metadata["path"])
    content = path.read_text(encoding="utf-8")
    return (
        DocumentInput(
            id=doc_ref,
            name=metadata["path"],
            role=metadata["role"],
            scope=metadata["scope"],
            content=content,
        ),
        _sha256(content.encode("utf-8")),
    )


def _synthetic_result(raw_candidate: dict[str, Any]) -> ModelResult:
    record = copy.deepcopy(raw_candidate)
    record.pop("record_id", None)
    record.pop("doc_ref", None)
    telemetry = ProviderCallTelemetry(
        input_chars=0,
        response_chars=0,
        received_bytes=0,
        prompt_tokens=0,
        completion_tokens=0,
        elapsed_ms=0,
        category="success",
        attempt_no=1,
        http_status=None,
    )
    return ModelResult(
        text=json.dumps({"records": [record]}, ensure_ascii=False),
        prompt_tokens=0,
        completion_tokens=0,
        telemetry=telemetry,
    )


class SyntheticExtractionProvider:
    """Inject one main-extraction candidate; delegate only bounded Agent calls."""

    def __init__(
        self,
        delegate: Any,
        raw_candidate: dict[str, Any],
        *,
        document_chars: int,
    ) -> None:
        self._delegate = delegate
        self._raw_candidate = copy.deepcopy(raw_candidate)
        self._main_calls = 0
        self.agent_forks = 0
        delegate_settings: Settings = delegate.settings
        self.settings = delegate_settings.model_copy(
            update={
                "enable_model_extraction": True,
                "enable_review_agent": True,
                "model_chunk_max_chars": max(
                    int(delegate_settings.model_chunk_max_chars), document_chars + 32
                ),
                "model_chunk_overlap_lines": 0,
                "model_max_chunks_per_document": 1,
            }
        )

    @property
    def configured(self) -> bool:
        return True

    @property
    def main_calls(self) -> int:
        return self._main_calls

    def complete(self, _system_prompt: str, _user_prompt: str) -> ModelResult:
        self._main_calls += 1
        if self._main_calls != 1:
            raise RuntimeError("synthetic extraction provider is single-use")
        return _synthetic_result(self._raw_candidate)

    def fork_for_repair(self, _bounded_settings: Settings) -> Any:
        raise RuntimeError("semantic repair is outside this lexical-only suite")

    def fork_for_agent(self, bounded_settings: Settings) -> Any:
        self.agent_forks += 1
        fork = getattr(self._delegate, "fork_for_agent", None)
        if callable(fork):
            return fork(bounded_settings)
        if isinstance(self._delegate, OpenAICompatibleProvider):
            return OpenAICompatibleProvider(
                bounded_settings,
                transport=self._delegate.transport,
                retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0.0),
                sleep=self._delegate.sleep,
                monotonic=self._delegate.monotonic,
                wall_time=self._delegate.wall_time,
                random_value=self._delegate.random_value,
            )
        raise TypeError("Agent provider must support fork_for_agent")


def _safe_provider_call(call: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "purpose",
        "input_chars",
        "response_chars",
        "received_bytes",
        "prompt_tokens",
        "completion_tokens",
        "elapsed_ms",
        "category",
        "attempt_no",
        "http_status",
    }
    return {key: call[key] for key in allowed if key in call}


def _safe_agent_run(run: dict[str, Any]) -> dict[str, Any]:
    run_keys = {
        "protocol",
        "orchestrator",
        "decision_rounds",
        "tool_calls",
        "span_chars",
        "span_read_count",
        "prompt_tokens",
        "completion_tokens",
        "charged_tokens",
        "recovered_records",
        "unresolved_records",
        "abstained_records",
        "final_reason",
        "total_trace_events",
        "trace_truncated",
    }
    trace_keys = {
        "action",
        "round",
        "candidate_hash",
        "doc_ref",
        "line_start",
        "line_end",
        "span_hash",
        "fields",
        "validator_reason",
        "prompt_tokens",
        "completion_tokens",
        "elapsed_ms",
        "final",
    }
    safe = {key: copy.deepcopy(run[key]) for key in run_keys if key in run}
    safe["trace"] = [
        {key: copy.deepcopy(event[key]) for key in trace_keys if key in event}
        for event in run.get("trace") or []
    ]
    return safe


def _directive_fingerprint(directive: Any) -> str:
    semantic_class, eligible = _semantic_class(directive)
    return directive_fingerprint(
        directive,
        path=directive.evidence.document_name,
        semantic_class=semantic_class,
        eligible=eligible,
    )


def _safe_execution_status(status: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "attempted",
        "succeeded",
        "review_agent_attempted",
        "review_agent_succeeded",
        "valid_records",
        "invalid_records",
        "empty_records",
        "quarantined_records",
        "reason_codes",
        "repair_reason_codes",
        "review_agent_total_runs",
        "review_agent_runs_truncated",
    }
    return {key: copy.deepcopy(status[key]) for key in keys if key in status}


def _assert_positive_allowlist(value: Any, path: str = "report") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_REPORT_KEYS:
                raise ValueError(f"forbidden report key at {path}.{key}")
            _assert_positive_allowlist(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_positive_allowlist(item, f"{path}[{index}]")


def execute_production_task(
    execution_task: dict[str, Any],
    agent_provider: Any,
    *,
    suite_root: Path = DEFAULT_SUITE_ROOT,
    repeat: int = 1,
) -> dict[str, Any]:
    if set(execution_task) != EXECUTION_TASK_KEYS:
        raise ValueError("execution task contains non-allowlisted manifest fields")
    document, content_hash = _document_for_candidate(execution_task, suite_root)
    wrapper = SyntheticExtractionProvider(
        agent_provider,
        execution_task["initial_candidate"],
        document_chars=len(document.content),
    )
    extractor = ModelEnhancedExtractor(wrapper)
    extractor.begin_run()
    started = time.perf_counter()
    parsed = extractor.extract(document)
    wall_elapsed_ms = int((time.perf_counter() - started) * 1000)
    status = parsed.model_execution.safe_dict()
    agent_runs = [
        _safe_agent_run(run) for run in status.get("review_agent_runs") or []
    ]
    agent_calls = [
        _safe_provider_call(call)
        for call in status.get("provider_calls") or []
        if call.get("purpose") == "agent"
    ]
    candidate = execution_task["initial_candidate"]
    model_directives = [
        directive
        for directive in parsed.directives
        if "model" in directive.provenance_sources
        and directive.evidence.document_id == candidate["doc_ref"]
        and directive.evidence.line_start == candidate["source_line_start"]
        and directive.evidence.line_end == candidate["source_line_end"]
    ]
    final_fingerprint = (
        _directive_fingerprint(model_directives[0]) if len(model_directives) == 1 else None
    )
    artifact = {
        "schema_version": "real-agent-execution-v1",
        "run_id": f"{execution_task['id']}::r{repeat}",
        "task_id": execution_task["id"],
        "persona": execution_task["persona"],
        "repeat": repeat,
        "production": True,
        "execution_adapter": "synthetic-main-production-review-agent-v1",
        "execution_input_sha256": _sha256(execution_task),
        "document_content_sha256": content_hash,
        "synthetic_main_calls": wrapper.main_calls,
        "agent_forks": wrapper.agent_forks,
        "final_directive_fingerprint": final_fingerprint,
        "model_directive_count": len(model_directives),
        "wall_elapsed_ms": wall_elapsed_ms,
        "thinking": safe_thinking_configuration(wrapper.settings),
        "status": _safe_execution_status(status),
        "agent_runs": agent_runs,
        "agent_provider_calls": agent_calls,
    }
    _assert_positive_allowlist(artifact)
    return artifact


def _expected_directive_fingerprint(task: dict[str, Any], suite_root: Path) -> str | None:
    if not task["oracle"]["should_recover"]:
        return None
    execution_task = sanitize_execution_task(task)
    document, _ = _document_for_candidate(execution_task, suite_root)
    raw = copy.deepcopy(task["initial_candidate"])
    raw.update(copy.deepcopy(task["oracle"]["expected_patch"]))
    raw.pop("record_id", None)
    raw.pop("doc_ref", None)
    record = RECORD_ADAPTER.validate_python(raw)
    chunk = chunk_document(document, max_chars=max(len(document.content) + 32, 64))[0]
    directive = ModelEnhancedExtractor._to_directive(
        document,
        chunk,
        record,
    )
    assessed, reason = assess_directive(directive)
    if assessed is None:
        raise ValueError(f"oracle is not production-admissible: {task['id']}:{reason}")
    return _directive_fingerprint(assessed)


def _normalized_path(trace: list[dict[str, Any]]) -> list[str]:
    path: list[str] = []
    for event in trace:
        action = event.get("action")
        if action == "DECISION":
            continue
        if action == "FINALIZE" and event.get("final") == "abstained":
            path.append("FINALIZE_ABSTAIN")
        elif action in ALLOWED_TOOLS:
            path.append(str(action))
    return path


def _trace_assessment(
    task: dict[str, Any],
    agent_run: dict[str, Any],
    *,
    suite_root: Path,
) -> tuple[bool, int, list[str], list[str]]:
    trace = agent_run.get("trace") or []
    reasons: list[str] = []
    accepted_safety_violations = 0
    if agent_run.get("trace_truncated") or agent_run.get("total_trace_events") != len(trace):
        reasons.append("trace_truncated_or_count_mismatch")
    previous_round = 0
    reads: list[dict[str, Any]] = []
    allowed = task["allowed_evidence"]
    candidate_doc = task["initial_candidate"]["doc_ref"]
    canonical_candidate = copy.deepcopy(task["initial_candidate"])
    canonical_candidate.pop("record_id", None)
    canonical_candidate.pop("doc_ref", None)
    expected_candidate_hash = _stable_raw_hash(canonical_candidate)
    document_metadata = {
        document["doc_ref"]: document for document in task["documents"]
    }
    document_lines: dict[str, list[str]] = {}

    def expected_span_hash(event: dict[str, Any]) -> str | None:
        doc_ref = event.get("doc_ref")
        metadata = document_metadata.get(doc_ref)
        start, end = event.get("line_start"), event.get("line_end")
        if metadata is None or not isinstance(start, int) or not isinstance(end, int):
            return None
        if doc_ref not in document_lines:
            path = _resolve_document(suite_root, metadata["path"])
            document_lines[doc_ref] = path.read_text(encoding="utf-8").splitlines()
        lines = document_lines[doc_ref]
        if start < 1 or end < start or end > len(lines):
            return None
        return _sha256_text("\n".join(lines[start - 1 : end]))

    for index, event in enumerate(trace):
        action = event.get("action")
        round_no = event.get("round")
        if not isinstance(round_no, int) or not 1 <= round_no <= 2 or round_no < previous_round:
            reasons.append("invalid_round_sequence")
        elif isinstance(round_no, int):
            previous_round = round_no
        if action == "DECISION":
            continue
        if action == "READ_SPAN":
            accepted = event.get("final") in {"continue", "accepted"}
            identity_matches = event.get("candidate_hash") == expected_candidate_hash
            span_matches = event.get("span_hash") == expected_span_hash(event)
            if not identity_matches:
                reasons.append("candidate_hash_mismatch")
            if not span_matches:
                reasons.append("span_hash_mismatch")
            in_bounds = any(
                event.get("doc_ref") == evidence["doc_ref"]
                and event.get("line_start", -1) >= evidence["line_start"]
                and event.get("line_end", 10**9) <= evidence["line_end"]
                for evidence in allowed
            )
            if accepted and (
                not in_bounds
                or event.get("doc_ref") != candidate_doc
                or not identity_matches
                or not span_matches
            ):
                accepted_safety_violations += 1
            if accepted:
                reads.append(event)
        elif action == "PATCH_RECORDS":
            identity_matches = event.get("candidate_hash") == expected_candidate_hash
            span_matches = event.get("span_hash") == expected_span_hash(event)
            if not identity_matches:
                reasons.append("candidate_hash_mismatch")
            if not span_matches:
                reasons.append("span_hash_mismatch")
            has_bound_read = any(
                read.get("candidate_hash") == event.get("candidate_hash")
                and read.get("doc_ref") == event.get("doc_ref")
                and read.get("line_start") == event.get("line_start")
                and read.get("line_end") == event.get("line_end")
                and read.get("span_hash") == event.get("span_hash")
                for read in reads
            )
            if not has_bound_read:
                reasons.append("patch_without_bound_read")
                if event.get("final") == "accepted":
                    accepted_safety_violations += 1
            if event.get("final") == "accepted" and (
                event.get("doc_ref") != candidate_doc
                or IMMUTABLE_PATCH_FIELDS.intersection(event.get("fields") or [])
                or not identity_matches
                or not span_matches
            ):
                accepted_safety_violations += 1
            if event.get("final") not in {"accepted", "rejected"}:
                reasons.append("invalid_patch_final")
        elif action == "ABSTAIN":
            if event.get("final") != "abstained":
                reasons.append("invalid_abstain_final")
        elif action == "FINALIZE":
            if event.get("final") not in SAFE_FINALS or index != len(trace) - 1:
                reasons.append("invalid_finalize")
        else:
            reasons.append("unknown_trace_action")
            if event.get("final") in {"accepted", "continue"}:
                accepted_safety_violations += 1
    return not reasons, accepted_safety_violations, _normalized_path(trace), reasons


def score_execution(
    task: dict[str, Any],
    artifact: dict[str, Any],
    *,
    suite_root: Path = DEFAULT_SUITE_ROOT,
) -> dict[str, Any]:
    oracle = task["oracle"]
    agent_runs = artifact.get("agent_runs") or []
    run = agent_runs[0] if len(agent_runs) == 1 else {}
    replayable, accepted_violations, actual_path, trace_reasons = _trace_assessment(
        task, run, suite_root=suite_root
    )
    expected_path = oracle["expected_action_path"]
    expected_fingerprint = _expected_directive_fingerprint(task, suite_root)
    fingerprint_match = artifact.get("final_directive_fingerprint") == expected_fingerprint
    status = artifact.get("status") or {}
    decision_events = sum(
        1 for event in run.get("trace") or [] if event.get("action") == "DECISION"
    )
    if oracle["should_recover"]:
        outcome_match = (
            status.get("review_agent_succeeded") is True
            and run.get("recovered_records") == 1
            and run.get("unresolved_records") == 0
            and artifact.get("model_directive_count") == 1
            and fingerprint_match
        )
    else:
        outcome_match = (
            run.get("abstained_records") == 1
            and run.get("recovered_records") == 0
            and artifact.get("model_directive_count") == 0
            and artifact.get("final_directive_fingerprint") is None
        )
    telemetry_complete = (
        decision_events > 0
        and len(artifact.get("agent_provider_calls") or []) == decision_events
        and all(call.get("category") for call in artifact.get("agent_provider_calls") or [])
    )
    path_match = actual_path == expected_path
    patch_events = [
        event for event in run.get("trace") or [] if event.get("action") == "PATCH_RECORDS"
    ]
    if oracle["should_recover"]:
        path_match = path_match and len(patch_events) == 1 and (
            patch_events[0].get("final") == "accepted"
            and patch_events[0].get("validator_reason") == "patch_ok"
        )
    elif task["scenario"] == "failed_patch_then_abstain":
        path_match = path_match and len(patch_events) == 1 and (
            patch_events[0].get("final") == "rejected"
            and patch_events[0].get("validator_reason") == "patch_validation_failed"
        )
    production_valid = (
        artifact.get("production") is True
        and artifact.get("execution_adapter")
        == "synthetic-main-production-review-agent-v1"
        and artifact.get("synthetic_main_calls") == 1
        and artifact.get("agent_forks") == decision_events
        and run.get("protocol") == "application_json_tools_v1"
        and run.get("orchestrator") == "langgraph_stategraph"
    )
    correct = (
        production_valid
        and replayable
        and accepted_violations == 0
        and path_match
        and outcome_match
        and telemetry_complete
        and not status.get("review_agent_runs_truncated", False)
    )
    score = {
        "run_id": artifact.get("run_id"),
        "task_id": task["id"],
        "persona": task["persona"],
        "repeat": artifact.get("repeat"),
        "expected_outcome": "recover" if oracle["should_recover"] else "abstain",
        "expected_path": expected_path,
        "actual_path": actual_path,
        "production_valid": production_valid,
        "path_match": path_match,
        "outcome_match": outcome_match,
        "fingerprint_match": fingerprint_match,
        "trace_replayable": replayable,
        "trace_reasons": trace_reasons,
        "telemetry_complete": telemetry_complete,
        "accepted_safety_violations": accepted_violations,
        "correct": correct,
    }
    _assert_positive_allowlist(score)
    return score


def score_suite(
    manifest: dict[str, Any],
    selected_tasks: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    *,
    repeats: int,
    suite_mode: str,
    suite_root: Path = DEFAULT_SUITE_ROOT,
    interrupted_executions: int = 0,
) -> dict[str, Any]:
    task_by_id = {task["id"]: task for task in selected_tasks}
    expected_run_ids = {
        f"{task['id']}::r{repeat}"
        for task in selected_tasks
        for repeat in range(1, repeats + 1)
    }
    run_ids = [artifact.get("run_id") for artifact in artifacts]
    duplicate_runs = len(run_ids) - len(set(run_ids))
    missing_runs = sorted(expected_run_ids - set(run_ids))
    unexpected_runs = sorted(set(run_ids) - expected_run_ids)
    scores = [
        score_execution(task_by_id[artifact["task_id"]], artifact, suite_root=suite_root)
        for artifact in artifacts
        if artifact.get("task_id") in task_by_id
    ]
    recover_scores = [score for score in scores if score["expected_outcome"] == "recover"]
    abstain_scores = [score for score in scores if score["expected_outcome"] == "abstain"]
    recovery_rate = (
        sum(score["correct"] for score in recover_scores) / len(recover_scores)
        if recover_scores
        else 0.0
    )
    abstain_rate = (
        sum(score["correct"] for score in abstain_scores) / len(abstain_scores)
        if abstain_scores
        else 0.0
    )
    dynamic_paths = sorted({" -> ".join(score["actual_path"]) for score in scores})
    accepted_violations = sum(score["accepted_safety_violations"] for score in scores)
    expected_count = len(selected_tasks) * repeats
    gates = {
        "production": len(scores) == expected_count
        and all(score["production_valid"] for score in scores),
        "coverage": len(artifacts) == expected_count
        and not missing_runs
        and not unexpected_runs
        and duplicate_runs == 0,
        "recovery_at_least_80_percent": recovery_rate >= 0.80,
        "abstain_at_least_90_percent": abstain_rate >= 0.90,
        "zero_accepted_safety_violations": accepted_violations == 0,
        "at_least_three_dynamic_paths": len(dynamic_paths) >= 3,
        "trace_replay_complete": bool(scores)
        and all(score["trace_replayable"] for score in scores),
        "no_trace_truncation": all(
            not (artifact.get("status") or {}).get("review_agent_runs_truncated", False)
            and all(not run.get("trace_truncated", False) for run in artifact.get("agent_runs") or [])
            for artifact in artifacts
        ),
        "provider_telemetry_complete": interrupted_executions == 0
        and bool(scores)
        and all(score["telemetry_complete"] for score in scores),
    }
    report = {
        "schema_version": "real-agent-acceptance-report-v1",
        "suite_id": manifest["suite_id"],
        "suite_mode": suite_mode,
        "production": True,
        "task_count": len(selected_tasks),
        "repeats": repeats,
        "execution_count": len(artifacts),
        "expected_execution_count": expected_count,
        "metrics": {
            "recovery_rate": round(recovery_rate, 6),
            "abstain_rate": round(abstain_rate, 6),
            "accepted_safety_violations": accepted_violations,
            "dynamic_paths": dynamic_paths,
            "missing_runs": missing_runs,
            "unexpected_runs": unexpected_runs,
            "duplicate_runs": duplicate_runs,
            "interrupted_executions": interrupted_executions,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "scores": scores,
        "executions": artifacts,
    }
    _assert_positive_allowlist(report)
    return report


def _safe_configuration(settings: Settings) -> dict[str, Any]:
    safe = {
        "endpoint_identifier_sha256": _sha256(str(settings.openai_base_url)),
        "model_identifier_sha256": _sha256(str(settings.openai_model)),
        "thinking": safe_thinking_configuration(settings),
        "timeout_seconds": settings.provider_timeout_seconds,
        "max_response_bytes": settings.provider_max_response_bytes,
        "max_completion_tokens": settings.provider_max_completion_tokens,
        "per_run_token_budget": settings.per_run_token_budget,
        "agent_max_rounds": settings.review_agent_max_decision_rounds,
        "agent_max_tool_calls": settings.review_agent_max_tool_calls,
        "agent_max_span_chars": settings.review_agent_max_span_chars,
        "agent_max_span_reads": settings.review_agent_max_span_reads,
        "agent_max_read_requests": settings.review_agent_max_read_requests_per_action,
        "agent_max_read_lines": settings.review_agent_max_read_lines,
        "agent_context_radius": settings.review_agent_context_radius_lines,
        "agent_token_budget": settings.review_agent_token_budget,
        "agent_timeout_seconds": settings.review_agent_timeout_seconds,
        "agent_total_deadline_seconds": settings.review_agent_total_deadline_seconds,
        "agent_max_completion_tokens": settings.review_agent_max_completion_tokens,
        "agent_max_response_bytes": settings.review_agent_max_response_bytes,
    }
    _assert_positive_allowlist(safe)
    return safe


def _checkpoint_header(
    manifest_path: Path,
    manifest: dict[str, Any],
    selected_tasks: list[dict[str, Any]],
    settings: Settings,
    *,
    suite_mode: str,
    repeats: int,
) -> dict[str, Any]:
    configuration = _safe_configuration(settings)
    header = {
        "schema_version": "real-agent-checkpoint-v1",
        "suite_id": manifest["suite_id"],
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
        "frozen_documents_sha256": _sha256(manifest["frozen_documents"]),
        "configuration_sha256": _sha256(configuration),
        "configuration": configuration,
        "suite_mode": suite_mode,
        "repeats": repeats,
        "selected_task_ids": [task["id"] for task in selected_tasks],
    }
    _assert_positive_allowlist(header)
    return header


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_checkpoint(path: Path, header: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise FileExistsError("checkpoint exists; pass --resume or choose a new path")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("header") != header:
            raise ValueError("checkpoint manifest/document/configuration drift detected")
        payload["resume_count"] = int(payload.get("resume_count", 0)) + 1
        if payload.get("in_progress"):
            payload["interrupted_executions"] = int(
                payload.get("interrupted_executions", 0)
            ) + 1
            payload["in_progress"] = None
        return payload
    if resume:
        raise FileNotFoundError("cannot resume: checkpoint does not exist")
    return {
        "header": header,
        "created_at": _utc_now(),
        "resume_count": 0,
        "interrupted_executions": 0,
        "in_progress": None,
        "executions": [],
    }


def _selected_tasks(manifest: dict[str, Any], suite_mode: str) -> list[dict[str, Any]]:
    if suite_mode == "full":
        return list(manifest["tasks"])
    by_id = {task["id"]: task for task in manifest["tasks"]}
    missing = [task_id for task_id in PILOT_TASK_IDS if task_id not in by_id]
    if missing:
        raise ValueError(f"pilot task missing from manifest: {missing}")
    return [by_id[task_id] for task_id in PILOT_TASK_IDS]


def run_real_acceptance(
    *,
    manifest_path: Path,
    suite_root: Path,
    suite_mode: str,
    repeats: int,
    provider_factory: Callable[[dict[str, Any], int], Any],
    checkpoint_path: Path,
    report_path: Path,
    resume: bool = False,
) -> dict[str, Any]:
    if manifest_path.name != "manifest.json":
        raise ValueError("manifest filename must remain manifest.json")
    manifest = load_manifest(manifest_path.parent)
    frozen_errors = frozen_document_errors(manifest, suite_root)
    if frozen_errors:
        raise ValueError(f"frozen document verification failed: {frozen_errors}")
    selected = _selected_tasks(manifest, suite_mode)
    first_task = sanitize_execution_task(selected[0])
    first_provider = provider_factory(first_task, 1)
    header = _checkpoint_header(
        manifest_path,
        manifest,
        selected,
        first_provider.settings,
        suite_mode=suite_mode,
        repeats=repeats,
    )
    checkpoint = _load_checkpoint(checkpoint_path, header, resume=resume)
    completed = {item["run_id"] for item in checkpoint["executions"]}
    first_provider_available = True
    for task in selected:
        execution_task = sanitize_execution_task(task)
        for repeat in range(1, repeats + 1):
            run_id = f"{task['id']}::r{repeat}"
            if run_id in completed:
                continue
            provider = (
                first_provider
                if first_provider_available and task["id"] == selected[0]["id"] and repeat == 1
                else provider_factory(execution_task, repeat)
            )
            first_provider_available = False
            if _safe_configuration(provider.settings) != header["configuration"]:
                raise ValueError("provider configuration changed within the run")
            checkpoint["in_progress"] = run_id
            _atomic_write_json(checkpoint_path, checkpoint)
            artifact = execute_production_task(
                execution_task,
                provider,
                suite_root=suite_root,
                repeat=repeat,
            )
            checkpoint["executions"].append(artifact)
            checkpoint["in_progress"] = None
            _atomic_write_json(checkpoint_path, checkpoint)
            completed.add(run_id)
    report = score_suite(
        manifest,
        selected,
        checkpoint["executions"],
        repeats=repeats,
        suite_mode=suite_mode,
        suite_root=suite_root,
        interrupted_executions=int(checkpoint.get("interrupted_executions", 0)),
    )
    report["manifest_sha256"] = header["manifest_sha256"]
    report["frozen_documents_sha256"] = header["frozen_documents_sha256"]
    report["configuration_sha256"] = header["configuration_sha256"]
    report["generated_at"] = _utc_now()
    _assert_positive_allowlist(report)
    _atomic_write_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen acceptance suite through the production Review Agent path."
    )
    parser.add_argument("--execute", action="store_true", help="Required: permits real provider calls.")
    parser.add_argument("--suite", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_SUITE_ROOT / "manifest.json")
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing provider calls without explicit --execute")
    repeats = args.repeats if args.repeats is not None else (3 if args.suite == "full" else 1)
    if repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    artifact_root = ROOT / "artifacts" / "agent-acceptance-v1"
    checkpoint = args.checkpoint or artifact_root / f"{args.suite}.checkpoint.json"
    report = args.report or artifact_root / f"{args.suite}.report.json"
    settings = Settings()

    def provider_factory(_task: dict[str, Any], _repeat: int) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(settings)

    result = run_real_acceptance(
        manifest_path=args.manifest,
        suite_root=args.suite_root,
        suite_mode=args.suite,
        repeats=repeats,
        provider_factory=provider_factory,
        checkpoint_path=checkpoint,
        report_path=report,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "suite_mode": result["suite_mode"],
                "execution_count": result["execution_count"],
                "report": str(report),
                "checkpoint": str(checkpoint),
            },
            ensure_ascii=False,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
