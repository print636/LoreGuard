from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
import hashlib
import json
import re
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from .time_utils import utc_now_naive


class AnalysisCancelled(RuntimeError):
    """Cooperative cancellation raised only at durable pipeline boundaries."""


_PROVIDER_TELEMETRY_CATEGORIES = {
    "success",
    "not_configured",
    "provider",
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
}

_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_SAFE_AGENT_HASH = re.compile(r"^[a-f0-9]{64}$")
_SAFE_AGENT_DOC_REF = re.compile(r"^[A-Za-z0-9._-]{1,16}$")
_SAFE_AGENT_FIELD = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SAFE_AGENT_MAX_LINE_NUMBER = 10_000_000
_SAFE_AGENT_ACTIONS = {
    "DECISION",
    "READ_SPAN",
    "PATCH_RECORDS",
    "ABSTAIN",
    "FINALIZE",
}
_SAFE_AGENT_FINALS = {"continue", "accepted", "abstained", "rejected"}
_SAFE_AGENT_REASONS = {
    "accepted_protocol",
    "read_ok",
    "patch_ok",
    "explicit_abstain",
    "invalid_json",
    "invalid_action",
    "unknown_tool",
    "cross_document",
    "evidence_range",
    "span_budget",
    "span_count_budget",
    "tool_budget",
    "token_budget",
    "deadline",
    "provider_error",
    "response_too_large",
    "repeated_loop",
    "patch_field_forbidden",
    "semantic_field_forbidden",
    "semantic_promotion",
    "patch_duplicate_candidate",
    "patch_validation_failed",
    "read_required",
    "invalid_span",
    "round_limit",
    "completed",
}
_SAFE_PROTOCOL_STAGES = {
    "json",
    "envelope",
    "actions",
    "action_row",
    "action_name",
    "action_schema",
}
_SAFE_PROTOCOL_ROOT_SHAPES = {
    "unparsed",
    "object",
    "array",
    "string",
    "number",
    "boolean",
    "null",
}
_SAFE_PROTOCOL_ACTION_NAMES = {"READ_SPAN", "PATCH_RECORDS", "ABSTAIN"}
_SAFE_PROTOCOL_SCHEMA_LOCATION_PARTS = {
    "*",
    "action",
    "actions",
    "requests",
    "patches",
    "candidate_index",
    "candidate_indexes",
    "doc_ref",
    "line_start",
    "line_end",
    "span_id",
    "fields",
    "reason_code",
    "unknown_field",
}
_SAFE_PROTOCOL_SCHEMA_ERROR_TYPES = {
    "missing",
    "extra_forbidden",
    "literal_error",
    "list_type",
    "dict_type",
    "int_type",
    "int_parsing",
    "string_type",
    "string_too_short",
    "string_too_long",
    "greater_than_equal",
    "less_than_equal",
    "too_short",
    "too_long",
    "string_pattern_mismatch",
    "validation_error",
}


def _optional_nonnegative_int(value: Any) -> int | None:
    """Normalize provider counters without turning missing values into zeroes."""
    return value if type(value) is int and value >= 0 else None


def _optional_agent_line_number(value: Any) -> int | None:
    return (
        value
        if type(value) is int and 1 <= value <= _SAFE_AGENT_MAX_LINE_NUMBER
        else None
    )


def _optional_request_id(value: Any) -> str | None:
    return value if isinstance(value, str) and _SAFE_REQUEST_ID.fullmatch(value) else None


def _safe_protocol_diagnostic(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    stage = value.get("stage")
    root_shape = value.get("root_shape")
    action_count = value.get("action_count")
    action_names = value.get("action_names")
    locations = value.get("schema_error_locations")
    error_types = value.get("schema_error_types")

    def safe_location(location: Any) -> str:
        if not isinstance(location, str):
            return "unknown_field"
        location = location[:256]
        return ".".join(
            part
            if part in _SAFE_PROTOCOL_SCHEMA_LOCATION_PARTS
            else "unknown_field"
            for part in location.split(".")[:6]
        ) or "unknown_field"

    return {
        "stage": (
            stage
            if isinstance(stage, str) and stage in _SAFE_PROTOCOL_STAGES
            else "action_schema"
        ),
        "root_shape": (
            root_shape
            if isinstance(root_shape, str)
            and root_shape in _SAFE_PROTOCOL_ROOT_SHAPES
            else "unparsed"
        ),
        "action_count": (
            min(action_count, 7)
            if type(action_count) is int and action_count >= 0
            else None
        ),
        "action_names": [
            name
            if isinstance(name, str) and name in _SAFE_PROTOCOL_ACTION_NAMES
            else "unknown"
            for name in action_names[:6]
        ]
        if isinstance(action_names, list)
        else [],
        "schema_error_locations": [safe_location(row) for row in locations[:8]]
        if isinstance(locations, list)
        else [],
        "schema_error_types": [
            row
            if isinstance(row, str) and row in _SAFE_PROTOCOL_SCHEMA_ERROR_TYPES
            else "validation_error"
            for row in error_types[:8]
        ]
        if isinstance(error_types, list)
        else [],
    }


def _safe_review_agent_trace(row: Any) -> dict[str, Any]:
    source = row if isinstance(row, dict) else {}
    action = source.get("action")
    final = source.get("final")
    reason = source.get("validator_reason")
    candidate_hash = source.get("candidate_hash")
    doc_ref = source.get("doc_ref")
    span_hash = source.get("span_hash")
    fields = source.get("fields")
    return {
        "action": action if action in _SAFE_AGENT_ACTIONS else "FINALIZE",
        "round": _optional_nonnegative_int(source.get("round")) or 0,
        "candidate_hash": (
            candidate_hash
            if isinstance(candidate_hash, str)
            and _SAFE_AGENT_HASH.fullmatch(candidate_hash)
            else None
        ),
        "doc_ref": (
            doc_ref
            if isinstance(doc_ref, str) and _SAFE_AGENT_DOC_REF.fullmatch(doc_ref)
            else None
        ),
        "line_start": _optional_agent_line_number(source.get("line_start")),
        "line_end": _optional_agent_line_number(source.get("line_end")),
        "allowed_line_start": _optional_agent_line_number(
            source.get("allowed_line_start")
        ),
        "allowed_line_end": _optional_agent_line_number(
            source.get("allowed_line_end")
        ),
        "span_hash": (
            span_hash
            if isinstance(span_hash, str) and _SAFE_AGENT_HASH.fullmatch(span_hash)
            else None
        ),
        "fields": [
            field
            for field in fields if isinstance(field, str) and _SAFE_AGENT_FIELD.fullmatch(field)
        ][:20]
        if isinstance(fields, list)
        else [],
        "validator_reason": (
            reason if reason in _SAFE_AGENT_REASONS else "invalid_action"
        ),
        "prompt_tokens": _optional_nonnegative_int(source.get("prompt_tokens")) or 0,
        "completion_tokens": (
            _optional_nonnegative_int(source.get("completion_tokens")) or 0
        ),
        "elapsed_ms": _optional_nonnegative_int(source.get("elapsed_ms")) or 0,
        "final": final if final in _SAFE_AGENT_FINALS else "rejected",
        "protocol_diagnostic": _safe_protocol_diagnostic(
            source.get("protocol_diagnostic")
        ),
    }


def _safe_review_agent_run(row: Any) -> dict[str, Any]:
    source = row if isinstance(row, dict) else {}
    reason = source.get("final_reason")
    trace = source.get("trace")
    trace_length = len(trace) if isinstance(trace, list) else 0
    reported_trace_total = _optional_nonnegative_int(
        source.get("total_trace_events")
    )
    total_trace_events = max(trace_length, reported_trace_total or 0)
    return {
        "protocol": (
            "application_json_tools_v1"
            if source.get("protocol") == "application_json_tools_v1"
            else "unknown"
        ),
        "orchestrator": (
            "langgraph_stategraph"
            if source.get("orchestrator") == "langgraph_stategraph"
            else "unknown"
        ),
        **{
            key: _optional_nonnegative_int(source.get(key)) or 0
            for key in (
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
            )
        },
        "final_reason": reason if reason in _SAFE_AGENT_REASONS else "invalid_action",
        "total_trace_events": total_trace_events,
        "trace_truncated": bool(source.get("trace_truncated") is True)
        or total_trace_events > min(trace_length, 64),
        "trace": (
            [_safe_review_agent_trace(trace_row) for trace_row in trace[:64]]
            if isinstance(trace, list)
            else []
        ),
    }


@dataclass(frozen=True, slots=True)
class ProviderCallDiagnostics:
    """Persistable, content-free diagnostics for one logical model call."""

    status: str
    category: str
    attempt: int | None
    elapsed_ms: int | None
    input_chars: int | None
    response_chars: int | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    http_status: int | None
    request_id: str | None
    purpose: Literal["extract", "repair", "agent", "evidence_review"] = "extract"

    @classmethod
    def from_telemetry(
        cls,
        telemetry: Any,
        *,
        succeeded: bool,
        purpose: Literal[
            "extract", "repair", "agent", "evidence_review"
        ] = "extract",
    ) -> ProviderCallDiagnostics | None:
        if telemetry is None:
            return None
        raw_category = getattr(telemetry, "category", None)
        category = (
            raw_category
            if raw_category in _PROVIDER_TELEMETRY_CATEGORIES
            else "provider"
        )
        prompt_tokens = _optional_nonnegative_int(
            getattr(telemetry, "prompt_tokens", None)
        )
        completion_tokens = _optional_nonnegative_int(
            getattr(telemetry, "completion_tokens", None)
        )
        total_tokens = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None
            else None
        )
        http_status = _optional_nonnegative_int(
            getattr(telemetry, "http_status", None)
        )
        if http_status is not None and not 100 <= http_status <= 599:
            http_status = None
        return cls(
            status="success" if succeeded else "failure",
            category=category,
            attempt=_optional_nonnegative_int(
                getattr(telemetry, "attempt_no", None)
            ),
            elapsed_ms=_optional_nonnegative_int(
                getattr(telemetry, "elapsed_ms", None)
            ),
            input_chars=_optional_nonnegative_int(
                getattr(telemetry, "input_chars", None)
            ),
            response_chars=_optional_nonnegative_int(
                getattr(telemetry, "response_chars", None)
            ),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            http_status=http_status,
            request_id=_optional_request_id(
                getattr(telemetry, "request_id", None)
            ),
            purpose=purpose,
        )

    def safe_dict(self) -> dict[str, str | int | None]:
        # Keep this explicit allowlist at the persistence boundary. Provider
        # URLs, headers, prompts and response bodies can never enter it.
        return {
            "status": self.status,
            "category": self.category,
            "attempt": self.attempt,
            "elapsed_ms": self.elapsed_ms,
            "input_chars": self.input_chars,
            "response_chars": self.response_chars,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "http_status": self.http_status,
            "request_id": self.request_id,
            "purpose": self.purpose,
        }


@dataclass(slots=True)
class ModelExecutionDiagnostics:
    """Structured model execution plus safe logical-call telemetry.

    ``provider_calls=None`` means telemetry was not exposed (legacy provider or
    legacy stored data). An empty list is a known zero-call execution.
    """

    enabled: bool = False
    configured: bool = False
    total_chunks: int = 0
    attempted_chunks: int = 0
    succeeded_chunks: int = 0
    failed_chunks: int = 0
    skipped_chunks: int = 0
    # Legacy observed/raw invalid count. Keep this additive counter stable for
    # stored diagnostics and distinguish final disposition below.
    invalid_records: int = 0
    # ``None`` means a legacy producer that did not expose final disposition.
    unresolved_invalid_records: int | None = 0
    recovered_invalid_records: int | None = 0
    # ``None`` is reserved for legacy execution objects that predate this
    # counter.  Treating a missing value as zero would manufacture proof that
    # the model covered every non-empty chunk.
    empty_response_chunks: int | None = 0
    batch_used: bool = False
    batch_document_count: int = 0
    batch_estimated_tokens: int = 0
    repair_attempted: bool = False
    repair_succeeded: bool = False
    repair_failed: bool = False
    repair_skipped_reason: str | None = None
    repair_pre_invalid: int = 0
    repair_post_invalid: int = 0
    repair_salvaged: int = 0
    repair_dropped: int = 0
    repair_final_path: str = "not_needed"
    review_agent_attempted: bool = False
    review_agent_succeeded: bool = False
    review_agent_abstained: bool = False
    # Each entry is produced by ReviewAgentRun.safe_dict(), whose persistence
    # boundary is a positive allowlist with no prompt, raw response or text.
    review_agent_runs: list[dict[str, Any]] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    provider_calls: list[ProviderCallDiagnostics] | None = field(default_factory=list)

    @classmethod
    def from_legacy(cls, execution: Any) -> ModelExecutionDiagnostics:
        """Upgrade the parser's pre-telemetry execution record additively."""
        return cls(
            enabled=bool(getattr(execution, "enabled", False)),
            configured=bool(getattr(execution, "configured", False)),
            total_chunks=_optional_nonnegative_int(
                getattr(execution, "total_chunks", None)
            ) or 0,
            attempted_chunks=_optional_nonnegative_int(
                getattr(execution, "attempted_chunks", None)
            ) or 0,
            succeeded_chunks=_optional_nonnegative_int(
                getattr(execution, "succeeded_chunks", None)
            ) or 0,
            failed_chunks=_optional_nonnegative_int(
                getattr(execution, "failed_chunks", None)
            ) or 0,
            skipped_chunks=_optional_nonnegative_int(
                getattr(execution, "skipped_chunks", None)
            ) or 0,
            invalid_records=_optional_nonnegative_int(
                getattr(execution, "invalid_records", None)
            ) or 0,
            unresolved_invalid_records=_optional_nonnegative_int(
                getattr(execution, "unresolved_invalid_records", None)
            ),
            recovered_invalid_records=_optional_nonnegative_int(
                getattr(execution, "recovered_invalid_records", None)
            ),
            empty_response_chunks=_optional_nonnegative_int(
                getattr(execution, "empty_response_chunks", None)
            ),
            batch_used=bool(getattr(execution, "batch_used", False)),
            batch_document_count=_optional_nonnegative_int(
                getattr(execution, "batch_document_count", None)
            ) or 0,
            batch_estimated_tokens=_optional_nonnegative_int(
                getattr(execution, "batch_estimated_tokens", None)
            ) or 0,
            reason_codes=list(getattr(execution, "reason_codes", [])),
        )

    def note(self, reason: str) -> None:
        if reason not in self.reason_codes:
            self.reason_codes.append(reason)

    def record_provider_call(
        self,
        telemetry: Any,
        *,
        succeeded: bool,
        purpose: Literal[
            "extract", "repair", "agent", "evidence_review"
        ] = "extract",
    ) -> None:
        if self.provider_calls is None:
            return
        safe = ProviderCallDiagnostics.from_telemetry(
            telemetry, succeeded=succeeded, purpose=purpose
        )
        if safe is None:
            # One uninstrumented logical call makes the per-call series
            # incomplete. Preserve that as unavailable, never as a fake zero.
            self.provider_calls = None
            return
        self.provider_calls.append(safe)

    def safe_dict(self) -> dict[str, Any]:
        repair = {
            "attempted": self.repair_attempted,
            "succeeded": self.repair_succeeded,
            "failed": self.repair_failed,
            "skipped_reason": self.repair_skipped_reason,
            "pre_invalid": self.repair_pre_invalid,
            "post_invalid": self.repair_post_invalid,
            "salvaged": self.repair_salvaged,
            "dropped": self.repair_dropped,
            "observed_invalid": self.invalid_records,
            "unresolved_invalid": self.unresolved_invalid_records,
            "recovered_invalid": self.recovered_invalid_records,
            "final_path": self.repair_final_path,
        }
        return {
            "enabled": self.enabled,
            "configured": self.configured,
            "total_chunks": self.total_chunks,
            "attempted_chunks": self.attempted_chunks,
            "succeeded_chunks": self.succeeded_chunks,
            "failed_chunks": self.failed_chunks,
            "skipped_chunks": self.skipped_chunks,
            "invalid_records": self.invalid_records,
            "unresolved_invalid_records": self.unresolved_invalid_records,
            "recovered_invalid_records": self.recovered_invalid_records,
            "empty_response_chunks": self.empty_response_chunks,
            "batch_used": self.batch_used,
            "batch_document_count": self.batch_document_count,
            "batch_estimated_tokens": self.batch_estimated_tokens,
            "repair_attempted": self.repair_attempted,
            "repair_succeeded": self.repair_succeeded,
            "repair_failed": self.repair_failed,
            "repair_skipped_reason": self.repair_skipped_reason,
            "repair_pre_invalid": self.repair_pre_invalid,
            "repair_post_invalid": self.repair_post_invalid,
            "repair_salvaged": self.repair_salvaged,
            "repair_dropped": self.repair_dropped,
            "repair_final_path": self.repair_final_path,
            "repair": repair,
            "review_agent_attempted": self.review_agent_attempted,
            "review_agent_succeeded": self.review_agent_succeeded,
            "review_agent_abstained": self.review_agent_abstained,
            "review_agent_runs": [
                _safe_review_agent_run(row) for row in self.review_agent_runs[:8]
            ],
            "review_agent_total_runs": len(self.review_agent_runs),
            "review_agent_runs_truncated": len(self.review_agent_runs) > 8,
            "reason_codes": list(self.reason_codes),
            "provider_calls": (
                [row.safe_dict() for row in self.provider_calls]
                if self.provider_calls is not None
                else None
            ),
        }


class Severity(StrEnum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class RunStatus(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class SemanticModality(StrEnum):
    asserted = "asserted"
    negated = "negated"
    uncertain = "uncertain"
    interrogative = "interrogative"
    hypothetical = "hypothetical"
    conditional_rule = "conditional_rule"
    reported = "reported"


class SourceScope(StrEnum):
    narrator = "narrator"
    world_rule = "world_rule"
    character_dialogue = "character_dialogue"
    quoted_material = "quoted_material"
    unverified_report = "unverified_report"
    unknown = "unknown"


class EvidenceMedium(StrEnum):
    record = "record"
    log = "log"
    report = "report"
    direct_observation = "direct_observation"
    unspecified = "unspecified"


class DocumentRole(StrEnum):
    canon = "canon"
    character_profile = "character_profile"
    chapter = "chapter"
    reference = "reference"


class CertaintyLevel(StrEnum):
    certain = "certain"
    probable = "probable"
    possible = "possible"
    unknown = "unknown"


class IssueCategory(StrEnum):
    fact_conflict = "fact_conflict"
    location_collision = "location_collision"
    knowledge_without_acquisition = "knowledge_without_acquisition"
    item_ownership = "item_ownership"
    world_rule_conflict = "world_rule_conflict"


class EvidenceSpan(BaseModel):
    document_id: str
    document_name: str
    line_start: int
    line_end: int
    text: str


class NarrativeEntity(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    kind: str
    name: str
    aliases: list[str] = Field(default_factory=list)


class NarrativeEvent(BaseModel):
    id: str
    timestamp: str
    location: str
    participants: list[str]
    evidence: EvidenceSpan


class CanonicalFact(BaseModel):
    subject: str
    predicate: str
    value: str
    timestamp: str | None = None
    evidence: EvidenceSpan


class KnowledgeAcquisition(BaseModel):
    character: str
    fact: str
    timestamp: str
    evidence: EvidenceSpan


class ParsedDirective(BaseModel):
    kind: str
    attrs: dict[str, str]
    evidence: EvidenceSpan
    # Server-derived narrative-frame safety bit.  A preceding section marker
    # such as "the following passage is a dream" may govern an otherwise
    # ordinary-looking evidence line.  Keep that context through model_copy
    # and normalization without exposing it in records, APIs or fingerprints.
    noncanonical_frame: bool = Field(default=False, exclude=True, repr=False)
    # Internal-only lineage carried through normalization/model_copy.  It is
    # deliberately excluded from record/API serialization; the public,
    # allowlisted representation lives in diagnostics.provenance.
    provenance_sources: frozenset[Literal["baseline", "model"]] = Field(
        default_factory=frozenset,
        exclude=True,
        repr=False,
    )


class ConsistencyIssue(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    category: IssueCategory
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    title: str
    explanation: str
    evidence: list[EvidenceSpan]
    suggestion: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def directive_fingerprint(
    directive: ParsedDirective,
    *,
    path: str,
    semantic_class: str,
    eligible: bool,
) -> str:
    """Stable v1 identity for a final, normalized directive.

    Only semantic fields and an evidence location participate. Evidence text,
    document ids, prompts and provider details are intentionally excluded.
    """
    payload = {
        "kind": directive.kind,
        "class": semantic_class,
        "eligible": eligible,
        "path": path,
        "line_start": directive.evidence.line_start,
        "line_end": directive.evidence.line_end,
        "attrs": sorted(directive.attrs.items()),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _semantic_provenance_key(directive: ParsedDirective) -> tuple:
    evidence = directive.evidence

    def clean(value: Any) -> str:
        return re.sub(r"\s+", "", str(value)).strip("，,。；;：:\"'“”‘’")

    return (
        directive.kind,
        tuple(sorted((key, clean(value)) for key, value in directive.attrs.items())),
        evidence.document_id,
        evidence.line_start,
        evidence.line_end,
    )


def apply_semantic_quality_gate_with_provenance(
    directives: list[ParsedDirective],
):
    """Run the semantic gate while unioning sources of converged duplicates."""
    from .semantic_quality import apply_semantic_quality_gate, assess_directive

    sources_by_result: dict[tuple, set[str]] = {}
    for directive in directives:
        assessed, _ = assess_directive(directive)
        if assessed is None:
            continue
        sources_by_result.setdefault(_semantic_provenance_key(assessed), set()).update(
            directive.provenance_sources
        )

    result = apply_semantic_quality_gate(directives)
    result.directives = [
        directive.model_copy(
            update={
                "provenance_sources": frozenset(
                    sources_by_result.get(
                        _semantic_provenance_key(directive),
                        directive.provenance_sources,
                    )
                )
            }
        )
        for directive in result.directives
    ]
    return result


def issue_fingerprint(
    issue: ConsistencyIssue,
    *,
    document_paths: dict[str, str],
) -> str:
    """Stable v1 identity for a deterministic issue and its evidence set."""
    evidence = sorted(
        {
            (
                document_paths.get(span.document_id, span.document_name),
                span.line_start,
            )
            for span in issue.evidence
        }
    )
    payload = {
        "category": issue.category.value,
        "evidence": [[path, line] for path, line in evidence],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


class AnalysisRun(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    status: RunStatus = RunStatus.queued
    created_at: datetime = Field(default_factory=utc_now_naive)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    input_chars: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0
    error: str | None = None


class EvaluationResult(BaseModel):
    sample_count: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    evidence_hit_rate: float
    category_scores: dict[str, dict[str, float | int]]
    generated_at: datetime = Field(default_factory=utc_now_naive)


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    issue_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    type: str
    label: str
    record_id: str
    timestamp: str | None = None
    evidence: EvidenceSpan
    issue_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphResponse(BaseModel):
    run_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    id: str
    record_id: str
    kind: str
    title: str
    timestamp: str | None = None
    precision: str
    evidence: EvidenceSpan
    issue_ids: list[str] = Field(default_factory=list)
    attrs: dict[str, str] = Field(default_factory=dict)


class TimelineGroup(BaseModel):
    timestamp: str
    sort_key: str
    precision: str
    entries: list[TimelineEntry]


class TimelineResponse(BaseModel):
    run_id: str
    groups: list[TimelineGroup]
    unscheduled: list[TimelineEntry]
    warnings: list[str] = Field(default_factory=list)
