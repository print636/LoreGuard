from __future__ import annotations

import argparse
import copy
from collections import Counter
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
from app.review_agent import AGENT_SYSTEM_PROMPT, _sha256_text
from scripts.run_agent_acceptance import (
    DEFAULT_SUITE_ROOT,
    LEGACY_SUITE_ROOT,
    frozen_document_errors,
    load_manifest,
)


PILOT_TASK_IDS = (
    "nw-01-location-literal",
    "gp-09-cross-branch-merge",
    "ed-07-wrong-chapter-location",
)
EVALUATION_IMPLEMENTATION_FILES = (
    "requirements.txt",
    "app/chunking.py",
    "app/config.py",
    "app/domain.py",
    "app/model_extractor.py",
    "app/natural.py",
    "app/parser.py",
    "app/pipeline.py",
    "app/provider.py",
    "app/review_agent.py",
    "app/semantic_quality.py",
    "app/usage.py",
    "scripts/run_agent_acceptance.py",
    "scripts/run_real_agent_acceptance.py",
)
SUITE_ROOTS = {"v1": LEGACY_SUITE_ROOT, "v2": DEFAULT_SUITE_ROOT}
CHECKPOINT_GENESIS_SHA256 = "0" * 64
NORMAL_RUNTIME_FINAL_REASONS = {"completed", "explicit_abstain", "round_limit"}
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


def _implementation_bundle_sha256() -> str:
    digest = hashlib.sha256()
    for relative_path in EVALUATION_IMPLEMENTATION_FILES:
        path = ROOT / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _execution_source(provider: Any) -> str:
    return (
        "real_provider"
        if isinstance(provider, OpenAICompatibleProvider)
        else "scripted_harness"
    )


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
        self.agent_patch_field_sha256s: list[str] = []
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
            agent_delegate = fork(bounded_settings)
        elif isinstance(self._delegate, OpenAICompatibleProvider):
            agent_delegate = OpenAICompatibleProvider(
                bounded_settings,
                transport=self._delegate.transport,
                retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0.0),
                sleep=self._delegate.sleep,
                monotonic=self._delegate.monotonic,
                wall_time=self._delegate.wall_time,
                random_value=self._delegate.random_value,
            )
        else:
            raise TypeError("Agent provider must support fork_for_agent")
        return PatchHashCapturingProvider(
            agent_delegate, self.agent_patch_field_sha256s
        )


class PatchHashCapturingProvider:
    """Capture only hashes of model-proposed PATCH fields for post-run scoring."""

    def __init__(self, delegate: Any, captured: list[str]) -> None:
        self._delegate = delegate
        self._captured = captured
        self.settings = delegate.settings

    @property
    def configured(self) -> bool:
        return bool(self._delegate.configured)

    def complete(self, system_prompt: str, user_prompt: str) -> ModelResult:
        result = self._delegate.complete(system_prompt, user_prompt)
        try:
            payload = json.loads(result.text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return result
        actions = payload.get("actions") if isinstance(payload, dict) else None
        if not isinstance(actions, list):
            return result
        for action in actions:
            if not isinstance(action, dict) or action.get("action") != "PATCH_RECORDS":
                continue
            patches = action.get("patches")
            if not isinstance(patches, list):
                continue
            for patch_request in patches:
                fields = (
                    patch_request.get("fields")
                    if isinstance(patch_request, dict)
                    else None
                )
                if isinstance(fields, dict):
                    self._captured.append(_sha256(fields))
        return result


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
    execution_source = _execution_source(agent_provider)
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
        "execution_source": execution_source,
        "real_provider_connected": execution_source == "real_provider",
        "execution_adapter": "synthetic-main-production-review-agent-v1",
        "execution_input_sha256": _sha256(execution_task),
        "document_content_sha256": content_hash,
        "synthetic_main_calls": wrapper.main_calls,
        "agent_forks": wrapper.agent_forks,
        "final_directive_fingerprint": final_fingerprint,
        "model_patch_field_sha256s": list(wrapper.agent_patch_field_sha256s),
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


def _expected_surface_patch_sha256(task: dict[str, Any]) -> str | None:
    if not task["oracle"]["should_recover"]:
        return None
    return _sha256(task["oracle"]["expected_patch"])


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


def _runtime_assessment(
    run: dict[str, Any], artifact: dict[str, Any]
) -> tuple[bool, list[str]]:
    failures: set[str] = set()
    trace = run.get("trace") or []
    decisions = [event for event in trace if event.get("action") == "DECISION"]
    calls = artifact.get("agent_provider_calls") or []
    if not decisions:
        failures.add("runtime:no_model_decision")
    if len(calls) != len(decisions):
        failures.add("runtime:provider_telemetry_incomplete")
    for call in calls:
        category = str(call.get("category") or "missing")
        if category != "success":
            failures.add(f"provider:{category}")
    for event in decisions:
        reason = str(event.get("validator_reason") or "missing")
        if reason != "accepted_protocol":
            failures.add(f"decision:{reason}")
    final_reason = str(run.get("final_reason") or "missing")
    if final_reason not in NORMAL_RUNTIME_FINAL_REASONS:
        failures.add(f"terminal:{final_reason}")
    return not failures, sorted(failures)


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
            if accepted and not in_bounds:
                # ``allowed_evidence`` is a post-run oracle span, not the
                # production READ_SPAN authorization policy.  A different
                # same-document neighbourhood read therefore fails benchmark
                # replay, but is not misreported as a server safety breach.
                reasons.append("read_outside_oracle_evidence")
            if accepted and (
                event.get("doc_ref") != candidate_doc
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
    expected_surface_patch_sha256 = _expected_surface_patch_sha256(task)
    fingerprint_match = artifact.get("final_directive_fingerprint") == expected_fingerprint
    model_patch_hashes = artifact.get("model_patch_field_sha256s")
    surface_patch_match = (
        model_patch_hashes == [expected_surface_patch_sha256]
        if oracle["should_recover"]
        else not model_patch_hashes
    )
    status = artifact.get("status") or {}
    decision_events = sum(
        1 for event in run.get("trace") or [] if event.get("action") == "DECISION"
    )
    runtime_success, runtime_failure_categories = _runtime_assessment(run, artifact)
    if oracle["should_recover"]:
        outcome_match = (
            status.get("review_agent_succeeded") is True
            and run.get("recovered_records") == 1
            and run.get("unresolved_records") == 0
            and artifact.get("model_directive_count") == 1
            and fingerprint_match
            and surface_patch_match
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
    reference_path_match = actual_path == expected_path
    patch_events = [
        event for event in run.get("trace") or [] if event.get("action") == "PATCH_RECORDS"
    ]
    rejected_patch_events = [
        event for event in patch_events if event.get("final") == "rejected"
    ]
    accepted_patch_events = [
        event for event in patch_events if event.get("final") == "accepted"
    ]
    bad_patch_rejection_reasons = {
        str(event.get("validator_reason") or "missing")
        for event in rejected_patch_events
    }
    if accepted_patch_events:
        if not oracle["should_recover"]:
            bad_patch_rejection_reasons.add("oracle_unrecoverable_patch")
        elif not surface_patch_match or not fingerprint_match:
            bad_patch_rejection_reasons.add("oracle_patch_mismatch")
    bad_patch_rejection_reasons = sorted(bad_patch_rejection_reasons)
    # Every rejected PATCH and every scorer-observed wrong accepted PATCH is a
    # bad model proposal for quality accounting.  Only the former can prove
    # fail-closed containment; an oracle mismatch accepted by the service did
    # not get contained and must remain a quality failure.
    bad_patch_proposal = bool(bad_patch_rejection_reasons)
    explicit_abstain = any(
        event.get("action") == "ABSTAIN" and event.get("final") == "abstained"
        for event in run.get("trace") or []
    )
    normal_safe_finalize = any(
        event.get("action") == "FINALIZE" and event.get("final") == "abstained"
        for event in run.get("trace") or []
    )
    safe_containment_success = (
        runtime_success
        and bad_patch_proposal
        and bool(rejected_patch_events)
        and not accepted_patch_events
        and normal_safe_finalize
        and run.get("final_reason") == "round_limit"
        and run.get("abstained_records") == 1
        and run.get("recovered_records") == 0
        and artifact.get("model_directive_count") == 0
        and artifact.get("final_directive_fingerprint") is None
    )
    semantic_abstain = (
        not oracle["should_recover"]
        and outcome_match
        and runtime_success
        and not bad_patch_proposal
        and explicit_abstain
        and run.get("final_reason") == "explicit_abstain"
    )
    recovery_action_valid = False
    if oracle["should_recover"]:
        recovery_action_valid = actual_path == ["READ_SPAN", "PATCH_RECORDS"] and len(
            patch_events
        ) == 1 and (
            patch_events[0].get("final") == "accepted"
            and patch_events[0].get("validator_reason") == "patch_ok"
        )
    outcome_policy_match = runtime_success and outcome_match and (
        recovery_action_valid if oracle["should_recover"] else semantic_abstain
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
        and outcome_policy_match
        and telemetry_complete
        and not status.get("review_agent_runs_truncated", False)
    )
    score = {
        "run_id": artifact.get("run_id"),
        "task_id": task["id"],
        "persona": task["persona"],
        "repeat": artifact.get("repeat"),
        "expected_outcome": "recover" if oracle["should_recover"] else "abstain",
        "reference_path": expected_path,
        "actual_path": actual_path,
        "production_valid": production_valid,
        "execution_source": artifact.get("execution_source", "unknown"),
        "real_provider_connected": artifact.get("real_provider_connected") is True,
        "reference_path_match": reference_path_match,
        "outcome_match": outcome_match,
        "outcome_policy_match": outcome_policy_match,
        "semantic_abstain": semantic_abstain,
        "safe_containment_success": safe_containment_success,
        "runtime_success": runtime_success,
        "runtime_failure_categories": runtime_failure_categories,
        "bad_patch_proposal": bad_patch_proposal,
        "bad_patch_rejection_reasons": bad_patch_rejection_reasons,
        "fingerprint_match": fingerprint_match,
        "surface_patch_match": surface_patch_match,
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
    evaluation_configuration: dict[str, Any] | None = None,
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
    development_ids = set(
        manifest.get("evaluation_subgroups", {}).get(
            "development_tuned_task_ids", PILOT_TASK_IDS
        )
    )
    for score in scores:
        score["evaluation_subgroup"] = (
            "development_tuned" if score["task_id"] in development_ids else "holdout"
        )
    recover_scores = [score for score in scores if score["expected_outcome"] == "recover"]
    abstain_scores = [score for score in scores if score["expected_outcome"] == "abstain"]
    recovery_rate = (
        sum(score["correct"] for score in recover_scores) / len(recover_scores)
        if recover_scores
        else 0.0
    )
    semantic_abstain_rate = (
        sum(score["correct"] for score in abstain_scores) / len(abstain_scores)
        if abstain_scores
        else 0.0
    )

    def contributes_semantic_path(score: dict[str, Any]) -> bool:
        if not score["runtime_success"] or not score["trace_replayable"]:
            return False
        if score["expected_outcome"] == "recover":
            return score["correct"] and score["actual_path"] == [
                "READ_SPAN",
                "PATCH_RECORDS",
            ]
        return (
            score["correct"]
            and score["semantic_abstain"]
            and not score["bad_patch_proposal"]
            and score["actual_path"]
            in (["ABSTAIN"], ["READ_SPAN", "ABSTAIN"])
        )

    dynamic_paths = sorted(
        {
            " -> ".join(score["actual_path"])
            for score in scores
            if contributes_semantic_path(score)
        }
    )
    accepted_violations = sum(score["accepted_safety_violations"] for score in scores)
    expected_count = len(selected_tasks) * repeats

    def subgroup_summary(
        name: str, subgroup_scores: list[dict[str, Any]], task_ids: set[str]
    ) -> dict[str, Any]:
        task_personas = Counter(task_by_id[task_id]["persona"] for task_id in task_ids)
        execution_personas = Counter(score["persona"] for score in subgroup_scores)
        recover = [
            score for score in subgroup_scores if score["expected_outcome"] == "recover"
        ]
        abstain = [
            score for score in subgroup_scores if score["expected_outcome"] == "abstain"
        ]
        patch_proposals = [
            score
            for score in subgroup_scores
            if "PATCH_RECORDS" in score["actual_path"]
        ]
        bad_patch_proposals = [
            score for score in patch_proposals if score["bad_patch_proposal"]
        ]
        safe_containments = [
            score for score in subgroup_scores if score["safe_containment_success"]
        ]
        runtime_failures = [
            score for score in subgroup_scores if not score["runtime_success"]
        ]
        runtime_failure_categories = Counter(
            category
            for score in runtime_failures
            for category in score["runtime_failure_categories"]
        )
        bad_patch_rejection_reasons = Counter(
            reason
            for score in bad_patch_proposals
            for reason in score["bad_patch_rejection_reasons"]
        )
        return {
            "name": name,
            "task_count": len(task_ids),
            "task_persona_distribution": dict(sorted(task_personas.items())),
            "expected_execution_count": len(task_ids) * repeats,
            "execution_count": len(subgroup_scores),
            "execution_persona_distribution": dict(
                sorted(execution_personas.items())
            ),
            "recoverable_executions": len(recover),
            "correctly_recovered_executions": sum(score["correct"] for score in recover),
            "recovery_rate": round(
                sum(score["correct"] for score in recover) / len(recover), 6
            )
            if recover
            else 0.0,
            "unrecoverable_executions": len(abstain),
            "correctly_abstained_executions": sum(score["correct"] for score in abstain),
            "semantic_abstain_rate": round(
                sum(score["correct"] for score in abstain) / len(abstain), 6
            )
            if abstain
            else 0.0,
            "runtime_success_count": len(subgroup_scores) - len(runtime_failures),
            "runtime_failure_count": len(runtime_failures),
            "runtime_failure_categories": dict(
                sorted(runtime_failure_categories.items())
            ),
            "patch_proposal_executions": len(patch_proposals),
            "bad_patch_proposal_executions": len(bad_patch_proposals),
            "bad_patch_proposal_rate": round(
                len(bad_patch_proposals) / len(patch_proposals), 6
            )
            if patch_proposals
            else 0.0,
            "bad_patch_rejection_reasons": dict(
                sorted(bad_patch_rejection_reasons.items())
            ),
            "safe_containment_success_count": len(safe_containments),
            "accepted_safety_violations": sum(
                score["accepted_safety_violations"] for score in subgroup_scores
            ),
            "trace_replay_complete": bool(subgroup_scores)
            and all(score["trace_replayable"] for score in subgroup_scores),
            "provider_telemetry_complete": bool(subgroup_scores)
            and all(score["telemetry_complete"] for score in subgroup_scores),
            "dynamic_paths": sorted(
                {
                    " -> ".join(score["actual_path"])
                    for score in subgroup_scores
                    if contributes_semantic_path(score)
                }
            ),
        }

    selected_ids = set(task_by_id)
    development_task_ids = selected_ids & development_ids
    holdout_task_ids = selected_ids - development_ids
    development_scores = [
        score for score in scores if score["evaluation_subgroup"] == "development_tuned"
    ]
    holdout_scores = [score for score in scores if score["evaluation_subgroup"] == "holdout"]
    subgroup_metrics = {
        "development_tuned": subgroup_summary(
            "development_tuned", development_scores, development_task_ids
        ),
        "holdout": subgroup_summary("holdout", holdout_scores, holdout_task_ids),
    }
    provider_categories = Counter(
        str(call.get("category"))
        for artifact in artifacts
        for call in artifact.get("agent_provider_calls") or []
        if call.get("category")
    )
    runtime_failure_scores = [score for score in scores if not score["runtime_success"]]
    runtime_failure_categories = Counter(
        category
        for score in runtime_failure_scores
        for category in score["runtime_failure_categories"]
    )
    patch_proposal_scores = [
        score for score in scores if "PATCH_RECORDS" in score["actual_path"]
    ]
    bad_patch_proposal_scores = [
        score for score in patch_proposal_scores if score["bad_patch_proposal"]
    ]
    bad_patch_rejection_reasons = Counter(
        reason
        for score in bad_patch_proposal_scores
        for reason in score["bad_patch_rejection_reasons"]
    )
    safe_containment_scores = [
        score for score in scores if score["safe_containment_success"]
    ]
    selected_task_personas = Counter(task["persona"] for task in selected_tasks)
    execution_personas = Counter(score["persona"] for score in scores)
    real_provider_connected = len(scores) == expected_count and all(
        score["real_provider_connected"] for score in scores
    )
    gates = {
        "production": len(scores) == expected_count
        and all(score["production_valid"] for score in scores),
        "real_provider_connected": real_provider_connected,
        "coverage": len(artifacts) == expected_count
        and not missing_runs
        and not unexpected_runs
        and duplicate_runs == 0,
        "recovery_at_least_80_percent": recovery_rate >= 0.80,
        "semantic_abstain_at_least_90_percent": semantic_abstain_rate >= 0.90,
        "zero_accepted_safety_violations": accepted_violations == 0,
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
        "selected_persona_balance": len(selected_task_personas) == 3
        and len(set(selected_task_personas.values())) == 1,
    }
    if suite_mode == "full":
        holdout = subgroup_metrics["holdout"]
        development = subgroup_metrics["development_tuned"]
        holdout_paths = set(holdout["dynamic_paths"])
        required_semantic_paths = {
            "READ_SPAN -> PATCH_RECORDS",
            "READ_SPAN -> ABSTAIN",
            "ABSTAIN",
        }
        gates.update(
            {
                "at_least_three_runtime_success_semantic_paths": len(holdout_paths) >= 3,
                "runtime_success_recover_path_present": (
                    "READ_SPAN -> PATCH_RECORDS" in holdout_paths
                ),
                "runtime_success_direct_abstain_path_present": (
                    "ABSTAIN" in holdout_paths
                ),
                "runtime_success_semantic_abstain_path_present": (
                    "READ_SPAN -> ABSTAIN" in holdout_paths
                ),
                "semantic_paths_are_expected_set": holdout_paths
                == required_semantic_paths,
                "full_repetitions_3": repeats
                == int(manifest["repetitions_per_task"])
                == 3,
                "holdout_task_count_27": holdout["task_count"] == 27,
                "manifest_persona_tasks_10_each": dict(
                    sorted(Counter(task["persona"] for task in manifest["tasks"]).items())
                )
                == {persona: 10 for persona in manifest["personas"]},
                "development_persona_tasks_1_each": development[
                    "task_persona_distribution"
                ]
                == {persona: 1 for persona in manifest["personas"]},
                "holdout_persona_tasks_9_each": holdout[
                    "task_persona_distribution"
                ]
                == {persona: 9 for persona in manifest["personas"]},
                "holdout_persona_executions_27_each": holdout[
                    "execution_persona_distribution"
                ]
                == {persona: 27 for persona in manifest["personas"]},
                "holdout_expected_executions_81": holdout[
                    "expected_execution_count"
                ]
                == 81,
                "holdout_execution_coverage": holdout["execution_count"]
                == holdout["expected_execution_count"],
                "holdout_recovery_at_least_80_percent": holdout["recovery_rate"]
                >= 0.80,
                "holdout_semantic_abstain_at_least_90_percent": holdout[
                    "semantic_abstain_rate"
                ]
                >= 0.90,
                "holdout_runtime_accounting_complete": holdout[
                    "runtime_success_count"
                ]
                + holdout["runtime_failure_count"]
                == holdout["execution_count"],
                "holdout_zero_accepted_safety_violations": holdout[
                    "accepted_safety_violations"
                ]
                == 0,
                "holdout_trace_replay_complete": holdout["trace_replay_complete"],
                "holdout_provider_telemetry_complete": holdout[
                    "provider_telemetry_complete"
                ],
            }
        )
    else:
        gates["pilot_is_development_tuned_subgroup"] = (
            selected_ids == development_ids and len(selected_ids) == 3
        )
    report = {
        "schema_version": "real-agent-acceptance-report-v2",
        "suite_id": manifest["suite_id"],
        "benchmark_revision": manifest.get(
            "benchmark_revision", "outcome-first-v2-post-pilot"
        ),
        "suite_mode": suite_mode,
        "production": True,
        "real_provider_connected": real_provider_connected,
        "eligible_for_model_quality_claims": suite_mode == "full"
        and real_provider_connected
        and all(gates.values()),
        "claim_scope": (
            "full_with_separate_holdout_gate"
            if suite_mode == "full"
            else "development_tuned_pilot_only_not_holdout"
        ),
        "task_count": len(selected_tasks),
        "repeats": repeats,
        "execution_count": len(artifacts),
        "expected_execution_count": expected_count,
        "metrics": {
            "recovery_rate": round(recovery_rate, 6),
            "semantic_abstain_rate": round(semantic_abstain_rate, 6),
            "runtime_success_count": len(scores) - len(runtime_failure_scores),
            "runtime_failure_count": len(runtime_failure_scores),
            "runtime_failure_categories": dict(
                sorted(runtime_failure_categories.items())
            ),
            "patch_proposal_executions": len(patch_proposal_scores),
            "bad_patch_proposal_executions": len(bad_patch_proposal_scores),
            "bad_patch_proposal_rate": round(
                len(bad_patch_proposal_scores) / len(patch_proposal_scores), 6
            )
            if patch_proposal_scores
            else 0.0,
            "bad_patch_rejection_reasons": dict(
                sorted(bad_patch_rejection_reasons.items())
            ),
            "safe_containment_success_count": len(safe_containment_scores),
            "accepted_safety_violations": accepted_violations,
            "dynamic_paths": dynamic_paths,
            "missing_runs": missing_runs,
            "unexpected_runs": unexpected_runs,
            "duplicate_runs": duplicate_runs,
            "interrupted_executions": interrupted_executions,
            "agent_provider_call_categories": dict(sorted(provider_categories.items())),
            "read_timeout_calls": provider_categories.get("read_timeout", 0),
            "task_persona_distribution": dict(sorted(selected_task_personas.items())),
            "execution_persona_distribution": dict(sorted(execution_personas.items())),
        },
        "evaluation_coverage": {
            "development_tuned_task_count": len(development_task_ids),
            "holdout_task_count": len(holdout_task_ids),
            "configured_provider_timeout_seconds": (
                evaluation_configuration or {}
            ).get("timeout_seconds"),
            "configured_agent_timeout_seconds": (
                evaluation_configuration or {}
            ).get("agent_timeout_seconds"),
            "configured_agent_total_deadline_seconds": (
                evaluation_configuration or {}
            ).get("agent_total_deadline_seconds"),
        },
        "subgroups": subgroup_metrics,
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
    execution_source: str,
) -> dict[str, Any]:
    configuration = _safe_configuration(settings)
    header = {
        "schema_version": "real-agent-checkpoint-v2",
        "suite_id": manifest["suite_id"],
        "manifest_sha256": _sha256(manifest_path.read_bytes()),
        "frozen_documents_sha256": _sha256(manifest["frozen_documents"]),
        "agent_prompt_sha256": _sha256(AGENT_SYSTEM_PROMPT),
        "evaluation_implementation_bundle_sha256": _implementation_bundle_sha256(),
        "configuration_sha256": _sha256(configuration),
        "configuration": configuration,
        "suite_mode": suite_mode,
        "repeats": repeats,
        "execution_source": execution_source,
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


def _seal_execution(
    artifact: dict[str, Any], previous_chain_sha256: str
) -> dict[str, Any]:
    if set(artifact).intersection(
        {
            "execution_content_sha256",
            "previous_execution_chain_sha256",
            "execution_chain_sha256",
        }
    ):
        raise ValueError("execution already contains checkpoint seal fields")
    content_sha256 = _sha256(artifact)
    chain_sha256 = _sha256(
        {
            "previous_execution_chain_sha256": previous_chain_sha256,
            "execution_content_sha256": content_sha256,
        }
    )
    sealed = {
        **artifact,
        "execution_content_sha256": content_sha256,
        "previous_execution_chain_sha256": previous_chain_sha256,
        "execution_chain_sha256": chain_sha256,
    }
    _assert_positive_allowlist(sealed)
    return sealed


def _validate_execution_chain(executions: Any) -> str:
    if not isinstance(executions, list):
        raise ValueError("checkpoint executions must be a list")
    previous = CHECKPOINT_GENESIS_SHA256
    for index, sealed in enumerate(executions):
        if not isinstance(sealed, dict):
            raise ValueError(f"checkpoint execution {index} is not an object")
        content_sha256 = sealed.get("execution_content_sha256")
        previous_sha256 = sealed.get("previous_execution_chain_sha256")
        chain_sha256 = sealed.get("execution_chain_sha256")
        artifact = {
            key: value
            for key, value in sealed.items()
            if key
            not in {
                "execution_content_sha256",
                "previous_execution_chain_sha256",
                "execution_chain_sha256",
            }
        }
        if content_sha256 != _sha256(artifact) or previous_sha256 != previous:
            raise ValueError("checkpoint execution content/chain drift detected")
        expected_chain = _sha256(
            {
                "previous_execution_chain_sha256": previous,
                "execution_content_sha256": content_sha256,
            }
        )
        if chain_sha256 != expected_chain:
            raise ValueError("checkpoint execution content/chain drift detected")
        previous = chain_sha256
    return previous


def _load_checkpoint(path: Path, header: dict[str, Any], *, resume: bool) -> dict[str, Any]:
    if path.exists():
        if not resume:
            raise FileExistsError("checkpoint exists; pass --resume or choose a new path")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("header") != header:
            raise ValueError("checkpoint manifest/document/configuration drift detected")
        _validate_execution_chain(payload.get("executions"))
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
    pilot_ids = tuple(
        manifest.get("evaluation_subgroups", {}).get(
            "development_tuned_task_ids", PILOT_TASK_IDS
        )
    )
    missing = [task_id for task_id in pilot_ids if task_id not in by_id]
    if missing:
        raise ValueError(f"pilot task missing from manifest: {missing}")
    return [by_id[task_id] for task_id in pilot_ids]


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
    required_full_repetitions = int(manifest["repetitions_per_task"])
    if suite_mode == "full" and (
        repeats != required_full_repetitions or required_full_repetitions != 3
    ):
        raise ValueError(
            "full suite requires manifest repetitions_per_task=3 and --repeats 3"
        )
    frozen_errors = frozen_document_errors(manifest, suite_root)
    if frozen_errors:
        raise ValueError(f"frozen document verification failed: {frozen_errors}")
    selected = _selected_tasks(manifest, suite_mode)
    first_task = sanitize_execution_task(selected[0])
    first_provider = provider_factory(first_task, 1)
    execution_source = _execution_source(first_provider)
    header = _checkpoint_header(
        manifest_path,
        manifest,
        selected,
        first_provider.settings,
        suite_mode=suite_mode,
        repeats=repeats,
        execution_source=execution_source,
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
            if _execution_source(provider) != execution_source:
                raise ValueError("provider execution source changed within the run")
            checkpoint["in_progress"] = run_id
            _atomic_write_json(checkpoint_path, checkpoint)
            artifact = execute_production_task(
                execution_task,
                provider,
                suite_root=suite_root,
                repeat=repeat,
            )
            previous_chain = (
                checkpoint["executions"][-1]["execution_chain_sha256"]
                if checkpoint["executions"]
                else CHECKPOINT_GENESIS_SHA256
            )
            checkpoint["executions"].append(
                _seal_execution(artifact, previous_chain)
            )
            checkpoint["in_progress"] = None
            _atomic_write_json(checkpoint_path, checkpoint)
            completed.add(run_id)
    _validate_execution_chain(checkpoint["executions"])
    report = score_suite(
        manifest,
        selected,
        checkpoint["executions"],
        repeats=repeats,
        suite_mode=suite_mode,
        suite_root=suite_root,
        interrupted_executions=int(checkpoint.get("interrupted_executions", 0)),
        evaluation_configuration=header["configuration"],
    )
    report["manifest_sha256"] = header["manifest_sha256"]
    report["frozen_documents_sha256"] = header["frozen_documents_sha256"]
    report["agent_prompt_sha256"] = header["agent_prompt_sha256"]
    report["evaluation_implementation_bundle_sha256"] = header[
        "evaluation_implementation_bundle_sha256"
    ]
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
    parser.add_argument(
        "--benchmark-version",
        choices=tuple(SUITE_ROOTS),
        default="v2",
        help="Select v2 by default; choose v1 only to replay the historical benchmark.",
    )
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--suite-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not args.execute:
        raise SystemExit("refusing provider calls without explicit --execute")
    suite_root = args.suite_root or SUITE_ROOTS[args.benchmark_version]
    manifest_path = args.manifest or suite_root / "manifest.json"
    if manifest_path.parent.resolve() != suite_root.resolve():
        raise SystemExit("--manifest and --suite-root must identify the same suite")
    repeats = args.repeats if args.repeats is not None else (3 if args.suite == "full" else 1)
    if repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.suite == "full":
        if manifest_path.name != "manifest.json":
            raise SystemExit("manifest filename must remain manifest.json")
        manifest = load_manifest(manifest_path.parent)
        required_full_repetitions = int(manifest["repetitions_per_task"])
        if repeats != required_full_repetitions or required_full_repetitions != 3:
            raise SystemExit(
                "--suite full requires manifest repetitions_per_task=3 and --repeats 3"
            )
    artifact_root = ROOT / "artifacts" / str(load_manifest(suite_root)["suite_id"])
    checkpoint = args.checkpoint or artifact_root / f"{args.suite}.checkpoint.json"
    report = args.report or artifact_root / f"{args.suite}.report.json"
    settings = Settings()

    def provider_factory(_task: dict[str, Any], _repeat: int) -> OpenAICompatibleProvider:
        return OpenAICompatibleProvider(settings)

    result = run_real_acceptance(
        manifest_path=manifest_path,
        suite_root=suite_root,
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
