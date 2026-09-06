from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from time import perf_counter
from typing import Callable, Protocol

from .domain import (
    ConsistencyIssue,
    ModelExecutionDiagnostics,
    ParsedDirective,
    apply_semantic_quality_gate_with_provenance,
    directive_fingerprint,
    issue_fingerprint,
)
from .parser import ParsedDocument, parse_document
from .rules import detect_issues


@dataclass(frozen=True, slots=True)
class DocumentInput:
    id: str
    name: str
    content: str
    role: str | None = None
    scope: str | None = None


@dataclass(slots=True)
class PipelineResult:
    directives: list[ParsedDirective] = field(default_factory=list)
    issues: list[ConsistencyIssue] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model_used: bool = False
    diagnostics: dict = field(default_factory=dict)


class NarrativeExtractor(Protocol):
    def extract(self, document: DocumentInput) -> ParsedDocument: ...


class ConsistencyChecker(Protocol):
    def check(self, directives: list[ParsedDirective]) -> list[ConsistencyIssue]: ...


class BaselineExtractor:
    def extract(self, document: DocumentInput) -> ParsedDocument:
        parsed = parse_document(document.id, document.name, document.content)
        parsed.directives = [
            directive.model_copy(
                update={"provenance_sources": frozenset({"baseline"})}
            )
            for directive in _bind_document_context(parsed.directives, document)
        ]
        return parsed


class RuleChecker:
    def check(self, directives: list[ParsedDirective]) -> list[ConsistencyIssue]:
        return detect_issues(directives)

    def check_with_candidates(self, directives, candidate_pairs):
        compatible = {
            frozenset(("fact", "fact")): "fact_evidence_shortlist",
            frozenset(("event", "event")): "location_evidence_shortlist",
            frozenset(("knows", "claims_knows")): "knowledge_evidence_shortlist",
            frozenset(("item", "uses")): "item_evidence_shortlist",
            frozenset(("world_rule", "world_assert")): "world_rule_evidence_shortlist",
        }
        traces = []
        for pair in candidate_pairs:
            kinds = frozenset((directives[pair.left_index].kind, directives[pair.right_index].kind))
            family = compatible.get(kinds)
            traces.append(pair.to_trace(
                directives,
                consumed_by=family,
                consumed=bool(family),
            ))
        # Exact canonical rules remain direct and cannot be lost to a retrieval top-k.
        return detect_issues(directives), traces


class AnalysisPipeline:
    """Stable orchestration boundary for replacing extraction and checking layers."""

    def __init__(self, extractor: NarrativeExtractor | None = None, checker: ConsistencyChecker | None = None, normalizer=None, retriever=None):
        if extractor is None:
            from .model_extractor import ModelEnhancedExtractor

            extractor = ModelEnhancedExtractor()
        self.extractor = extractor
        self.checker = checker or RuleChecker()
        if normalizer is None:
            from .candidate_normalizer import DeterministicCandidateNormalizer

            normalizer = DeterministicCandidateNormalizer()
        self.normalizer = normalizer
        if retriever is None:
            from .retrieval import HybridRetriever

            retriever = HybridRetriever()
        self.retriever = retriever

    def interrupted_model_usage(self) -> dict | None:
        """Return content-free Agent accounting after an interrupted run.

        Completed runs use ``PipelineResult`` as their accounting boundary.
        This narrow escape hatch exists only because cooperative cancellation
        can happen immediately after an upstream Agent response and before a
        ``PipelineResult`` exists.  The service revalidates and labels this as
        an incomplete lower bound before persisting it.
        """
        getter = getattr(self.extractor, "review_agent_safe_accounting", None)
        if not callable(getter):
            return None
        accounting = getter()
        return accounting if isinstance(accounting, dict) else None

    def run(
        self,
        documents: list[DocumentInput],
        on_stage: Callable[[str, int, str], None] | None = None,
        checkpoint: Callable[[], None] | None = None,
    ) -> PipelineResult:
        notify = on_stage or (lambda *_: None)
        check = checkpoint or (lambda: None)
        started = perf_counter()
        first_progress_ms: float | None = None

        def emit(stage: str, progress: int, message: str) -> None:
            nonlocal first_progress_ms
            if first_progress_ms is None:
                first_progress_ms = (perf_counter() - started) * 1000
            notify(stage, progress, message)

        result = PipelineResult()
        if hasattr(self.extractor, "begin_run"):
            self.extractor.begin_run(checkpoint=checkpoint)
        from .aliases import canonicalize_entities
        from .chunking import chunk_document
        from .config import get_settings

        settings = get_settings()
        chunk_started = perf_counter()
        document_chunk_counts = []
        for document in documents:
            chunks = chunk_document(document, settings.model_chunk_max_chars, settings.model_chunk_overlap_lines)
            document_chunk_counts.append({
                "document_id": document.id, "document_name": document.name,
                "chars": len(document.content), "lines": len(document.content.splitlines()),
                "chunk_count": len(chunks), "model_chunk_limit": settings.model_max_chunks_per_document,
                "would_truncate_model_chunks": len(chunks) > settings.model_max_chunks_per_document,
            })
        chunk_ms = (perf_counter() - chunk_started) * 1000

        emit("extract", 10, "开始抽取叙事状态")
        extract_started = perf_counter()
        model_documents = []
        parsed_documents = None
        if len(documents) >= 2 and hasattr(self.extractor, "extract_batch"):
            check()
            parsed_documents = self.extractor.extract_batch(documents)
            check()
        if parsed_documents is None:
            parsed_documents = []
            for document in documents:
                check()
                parsed_documents.append(self.extractor.extract(document))
                check()
        if len(parsed_documents) != len(documents):
            raise RuntimeError("extractor returned a different document count")
        for document, parsed in zip(documents, parsed_documents, strict=True):
            check()
            execution = parsed.model_execution
            execution_diagnostics = _safe_execution_dict(execution)
            model_documents.append({
                "document_id": document.id,
                "document_name": document.name,
                **execution_diagnostics,
            })
            result.directives.extend(parsed.directives)
            result.warnings.extend(f"{document.name}: {warning}" for warning in parsed.warnings)
            result.prompt_tokens += parsed.prompt_tokens
            result.completion_tokens += parsed.completion_tokens
            result.model_used = result.model_used or parsed.model_used
        extract_ms = (perf_counter() - extract_started) * 1000

        index_started = perf_counter()
        extracted_ids = {id(directive) for directive in result.directives}
        normalized = self.normalizer.enrich(documents, result.directives)
        contexts = {document.id: document for document in documents}
        result.directives = [
            _bind_directive_context(
                (
                    _label_closed_normalizer_directive(directive)
                    if id(directive) not in extracted_ids
                    and not directive.provenance_sources
                    else directive
                ),
                contexts.get(directive.evidence.document_id),
            )
            for directive in normalized.directives
        ]
        result.warnings.extend(normalized.warnings)
        from .semantic_quality import (
            canonicalize_evidenced_titles,
        )

        semantic_quality = apply_semantic_quality_gate_with_provenance(
            result.directives
        )
        result.directives = semantic_quality.directives
        if warning := semantic_quality.warning():
            result.warnings.append(warning)
        alias_result = canonicalize_entities(documents, result.directives)
        result.directives, title_traces, title_warnings = canonicalize_evidenced_titles(
            documents, alias_result.directives
        )
        result.warnings.extend(alias_result.warnings)
        result.warnings.extend(title_warnings)
        result.directives = _sort_directives(
            documents, _merge_final_directive_sources(result.directives)
        )
        candidate_pairs = self.retriever.candidate_pairs(result.directives)
        index_ms = (perf_counter() - index_started) * 1000
        emit("index", 45, f"已建立 {len(result.directives)} 条可追溯状态记录")
        check_started = perf_counter()
        if hasattr(self.checker, "check_with_candidates"):
            checked, retrieval_traces = self.checker.check_with_candidates(result.directives, candidate_pairs)
        else:
            checked = self.checker.check(result.directives)
            retrieval_traces = [pair.to_trace(result.directives, consumed=False, consumed_by=None) for pair in candidate_pairs]
        check_ms = (perf_counter() - check_started) * 1000
        report_started = perf_counter()
        result.issues = _dedupe_issues(checked)
        report_ms = (perf_counter() - report_started) * 1000
        provider_calls = (
            [
                call
                for row in model_documents
                for call in row["provider_calls"]
            ]
            if model_documents
            and all(isinstance(row.get("provider_calls"), list) for row in model_documents)
            else None
        )
        review_agent_runs = [
            run
            for row in model_documents
            for run in row.get("review_agent_runs", [])
        ]
        review_agent_total_runs = sum(
            int(row.get("review_agent_total_runs", 0) or 0)
            for row in model_documents
        )
        review_agent_runs_truncated = (
            any(
                row.get("review_agent_runs_truncated", False)
                for row in model_documents
            )
            or len(review_agent_runs) > 8
            or review_agent_total_runs > len(review_agent_runs)
        )
        for row in model_documents:
            # The run-level series below is the sole persisted telemetry series.
            # Keeping a second copy per document invites accidental double sums.
            row.pop("provider_calls", None)
        provenance = _build_provenance(result.directives, result.issues, documents)
        result.diagnostics = {
            "model": {
                "enabled": any(row["enabled"] for row in model_documents),
                "configured": any(row["configured"] for row in model_documents),
                **{
                    key: _sum_known_counter(model_documents, key)
                    for key in (
                        "total_chunks", "attempted_chunks", "succeeded_chunks",
                        "failed_chunks", "skipped_chunks", "invalid_records",
                        "unresolved_invalid_records", "recovered_invalid_records",
                        "empty_response_chunks",
                    )
                },
                "reason_codes": sorted({
                    reason for row in model_documents for reason in row["reason_codes"]
                }),
                "provider_calls": provider_calls,
                "logical_call_count": (
                    len(provider_calls) if isinstance(provider_calls, list) else None
                ),
                "batch_used": any(row.get("batch_used", False) for row in model_documents),
                "batch_document_count": max(
                    (row.get("batch_document_count", 0) for row in model_documents),
                    default=0,
                ),
                "repair_attempted": any(
                    row.get("repair_attempted", False) for row in model_documents
                ),
                "repair_succeeded": any(
                    row.get("repair_succeeded", False) for row in model_documents
                ),
                "repair_failed": any(
                    row.get("repair_failed", False) for row in model_documents
                ),
                "repair_skipped_reason": next(
                    (
                        row.get("repair_skipped_reason")
                        for row in model_documents
                        if row.get("repair_skipped_reason")
                    ),
                    None,
                ),
                **{
                    key: sum(int(row.get(key, 0) or 0) for row in model_documents)
                    for key in (
                        "repair_pre_invalid",
                        "repair_post_invalid",
                        "repair_salvaged",
                        "repair_dropped",
                    )
                },
                "repair_final_path": _aggregate_repair_final_path(model_documents),
                "review_agent_attempted": any(
                    row.get("review_agent_attempted", False)
                    for row in model_documents
                ),
                "review_agent_succeeded": any(
                    row.get("review_agent_succeeded", False)
                    for row in model_documents
                ),
                "review_agent_abstained": any(
                    row.get("review_agent_abstained", False)
                    for row in model_documents
                ),
                "review_agent_runs": review_agent_runs[:8],
                "review_agent_total_runs": review_agent_total_runs,
                "review_agent_runs_truncated": review_agent_runs_truncated,
                "repair": {
                    "attempted": any(
                        row.get("repair_attempted", False)
                        for row in model_documents
                    ),
                    "succeeded": any(
                        row.get("repair_succeeded", False)
                        for row in model_documents
                    ),
                    "failed": any(
                        row.get("repair_failed", False)
                        for row in model_documents
                    ),
                    "skipped_reason": next(
                        (
                            row.get("repair_skipped_reason")
                            for row in model_documents
                            if row.get("repair_skipped_reason")
                        ),
                        None,
                    ),
                    "pre_invalid": sum(
                        int(row.get("repair_pre_invalid", 0) or 0)
                        for row in model_documents
                    ),
                    "post_invalid": sum(
                        int(row.get("repair_post_invalid", 0) or 0)
                        for row in model_documents
                    ),
                    "salvaged": sum(
                        int(row.get("repair_salvaged", 0) or 0)
                        for row in model_documents
                    ),
                    "dropped": sum(
                        int(row.get("repair_dropped", 0) or 0)
                        for row in model_documents
                    ),
                    "observed_invalid": _sum_known_counter(
                        model_documents, "invalid_records"
                    ),
                    "unresolved_invalid": _sum_known_counter(
                        model_documents, "unresolved_invalid_records"
                    ),
                    "recovered_invalid": _sum_known_counter(
                        model_documents, "recovered_invalid_records"
                    ),
                    "final_path": _aggregate_repair_final_path(model_documents),
                },
                "batch_budget_control": {
                    "mode": "local_conservative_estimate_only",
                    "provider_output_hard_limit": False,
                    "estimated_tokens": max(
                        (row.get("batch_estimated_tokens", 0) for row in model_documents),
                        default=0,
                    ),
                    "max_estimated_tokens": settings.model_batch_max_estimated_tokens,
                    "max_records_per_document": 40,
                },
                "documents": model_documents,
            },
            "chunking": {
                "max_chars": settings.model_chunk_max_chars,
                "overlap_lines": settings.model_chunk_overlap_lines,
                "max_chunks_per_document": settings.model_max_chunks_per_document,
                "total_chunks": sum(row["chunk_count"] for row in document_chunk_counts),
                "documents": document_chunk_counts,
            },
            "aliases": {
                "map": alias_result.alias_map,
                "declaration_count": len(alias_result.declarations),
                "trace_count": len(alias_result.traces) + len(title_traces),
                "traces": [*alias_result.traces, *title_traces],
            },
            "semantic_quality": {
                "rejected_after_normalization": semantic_quality.rejected_count,
                "transformed_after_normalization": semantic_quality.transformed_count,
                "reason_counts": semantic_quality.reasons,
                "canonical_count": sum(
                    row.kind in {
                        "fact", "event", "knows", "claims_knows", "item",
                        "uses", "world_rule", "world_assert",
                    }
                    for row in result.directives
                ),
                "non_canonical_count": sum(
                    row.kind in {
                        "open_question", "tentative_fact", "character_claim",
                        "negative_statement", "clarification",
                    }
                    for row in result.directives
                ),
                "boundary": (
                    "Only certain asserted/negated/closed-rule records enter deterministic checks; "
                    "questions, open hypotheses, uncertain statements and dialogue propositions remain traceable."
                ),
            },
            "provenance": provenance,
            "retrieval": {
                "implementation": "local stable SHA-256 character n-gram + keyword + canonical-entity graph",
                "candidate_count": len(candidate_pairs),
                "consumed_count": sum(bool(row.get("consumed")) for row in retrieval_traces),
                "traces": retrieval_traces,
                "boundary": "Evidence shortlist only; exact canonical rules bypass retrieval top-k.",
            },
            "timings": {
                "clock": "time.perf_counter monotonic",
                "chunk_ms": round(chunk_ms, 3),
                "extract_ms": round(extract_ms, 3),
                "index_ms": round(index_ms, 3),
                "check_ms": round(check_ms, 3),
                "report_ms": round(report_ms, 3),
                "first_progress_ms": round(first_progress_ms or 0.0, 3),
                "total_ms": round((perf_counter() - started) * 1000, 3),
            },
        }
        emit("check", 75, f"检查器发现 {len(result.issues)} 个候选问题")
        result.diagnostics["timings"]["total_ms"] = round((perf_counter() - started) * 1000, 3)
        return result


def _safe_execution_dict(execution) -> dict:
    if isinstance(execution, ModelExecutionDiagnostics):
        return execution.safe_dict()
    # Compatibility for baseline/custom extractors that still return the
    # parser's original ModelExecution dataclass. Keep serialization explicit.
    return {
        "enabled": bool(getattr(execution, "enabled", False)),
        "configured": bool(getattr(execution, "configured", False)),
        **{
            key: getattr(execution, key, None)
            for key in (
                "total_chunks",
                "attempted_chunks",
                "succeeded_chunks",
                "failed_chunks",
                "skipped_chunks",
                "invalid_records",
                "unresolved_invalid_records",
                "recovered_invalid_records",
                "empty_response_chunks",
            )
        },
        "reason_codes": list(getattr(execution, "reason_codes", [])),
        "batch_used": bool(getattr(execution, "batch_used", False)),
        "batch_document_count": getattr(execution, "batch_document_count", 0),
        "batch_estimated_tokens": getattr(execution, "batch_estimated_tokens", 0),
        "repair_attempted": False,
        "repair_succeeded": False,
        "repair_failed": False,
        "repair_skipped_reason": None,
        "repair_pre_invalid": 0,
        "repair_post_invalid": 0,
        "repair_salvaged": 0,
        "repair_dropped": 0,
        "repair_final_path": "not_needed",
        "review_agent_attempted": False,
        "review_agent_succeeded": False,
        "review_agent_abstained": False,
        "review_agent_runs": [],
        "review_agent_total_runs": 0,
        "review_agent_runs_truncated": False,
        "repair": {
            "attempted": False,
            "succeeded": False,
            "failed": False,
            "skipped_reason": None,
            "pre_invalid": 0,
            "post_invalid": 0,
            "salvaged": 0,
            "dropped": 0,
            "observed_invalid": getattr(execution, "invalid_records", None),
            "unresolved_invalid": None,
            "recovered_invalid": None,
            "final_path": "not_needed",
        },
        "provider_calls": None,
    }


def _aggregate_repair_final_path(rows: list[dict]) -> str:
    """Compose document repair paths without losing mixed run outcomes."""
    succeeded = any(row.get("repair_succeeded", False) for row in rows)
    post_invalid = sum(int(row.get("repair_post_invalid", 0) or 0) for row in rows)
    salvaged = sum(int(row.get("repair_salvaged", 0) or 0) for row in rows)
    dropped = sum(int(row.get("repair_dropped", 0) or 0) for row in rows)
    skipped = any(row.get("repair_skipped_reason") for row in rows)
    failed = any(row.get("repair_failed", False) for row in rows)
    if succeeded:
        if post_invalid or salvaged or dropped or skipped or failed:
            return "partial_repair"
        return "repaired"
    if salvaged and dropped:
        return "salvaged_and_baseline"
    if salvaged:
        return "salvaged"
    if dropped or post_invalid or skipped or failed:
        return "baseline"
    return "not_needed"


def _sum_known_counter(rows: list[dict], key: str) -> int | None:
    """Aggregate counters without rewriting legacy unknowns as known zeroes."""
    values = [row.get(key) for row in rows]
    if any(type(value) is not int or value < 0 for value in values):
        return None
    return sum(values)


def _sort_directives(
    documents: list[DocumentInput], directives: list[ParsedDirective]
) -> list[ParsedDirective]:
    """Return a reproducible order independent of model group/record ordering."""
    document_order: dict[str, int] = {}
    for index, document in enumerate(documents):
        document_order.setdefault(document.id, index)

    def key(directive: ParsedDirective) -> tuple[int, int, int, str, str]:
        canonical = json.dumps(
            {
                "document_id": directive.evidence.document_id,
                "line_start": directive.evidence.line_start,
                "line_end": directive.evidence.line_end,
                "kind": directive.kind,
                "attrs": directive.attrs,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return (
            document_order.get(directive.evidence.document_id, len(documents)),
            directive.evidence.line_start,
            directive.evidence.line_end,
            directive.kind,
            fingerprint,
        )

    return sorted(directives, key=key)


def _final_directive_key(directive: ParsedDirective) -> tuple:
    evidence = directive.evidence
    return (
        directive.kind,
        json.dumps(directive.attrs, ensure_ascii=False, sort_keys=True),
        evidence.document_id,
        evidence.document_name,
        evidence.line_start,
        evidence.line_end,
    )


def _merge_final_directive_sources(
    directives: list[ParsedDirective],
) -> list[ParsedDirective]:
    """Deduplicate final normalized rows without losing either extractor."""
    merged: list[ParsedDirective] = []
    indexes: dict[tuple, int] = {}
    for directive in directives:
        key = _final_directive_key(directive)
        existing_index = indexes.get(key)
        if existing_index is None:
            indexes[key] = len(merged)
            merged.append(directive)
            continue
        existing = merged[existing_index]
        sources = existing.provenance_sources | directive.provenance_sources
        if sources != existing.provenance_sources:
            merged[existing_index] = existing.model_copy(
                update={"provenance_sources": frozenset(sources)}
            )
    return merged


def _semantic_class(directive: ParsedDirective) -> tuple[str, bool]:
    from .semantic_quality import eligible_for_deterministic_rules

    eligible = eligible_for_deterministic_rules(directive)
    if directive.kind == "open_question":
        return "open_question", eligible
    if directive.kind == "tentative_fact":
        return "tentative", eligible
    if directive.kind == "character_claim":
        return "quoted_claim", eligible
    if directive.kind == "clarification":
        return "clarification", eligible
    if not eligible:
        return "noncanonical_other", eligible
    if (
        directive.kind == "world_rule"
        or directive.attrs.get("document_role") == "canonical_setting"
        or directive.attrs.get("source_scope") == "world_rule"
    ):
        return "confirmed_canonical", eligible
    return "confirmed_narrative", eligible


def _document_provenance_path(document: DocumentInput) -> str:
    """Prefer an explicit path embedded by file-based acceptance inputs."""
    if ":" in document.id:
        candidate = document.id.split(":", 1)[1].replace("\\", "/")
        name = document.name.replace("\\", "/")
        if candidate == name or candidate.endswith(f"/{name}"):
            return candidate
    return document.name.replace("\\", "/")


def _build_provenance(
    directives: list[ParsedDirective],
    issues: list[ConsistencyIssue],
    documents: list[DocumentInput],
) -> dict:
    """Build the content-free, stable v1 per-result provenance sidecar."""
    document_paths = {
        document.id: _document_provenance_path(document) for document in documents
    }
    directive_rows: list[dict] = []
    evidence_source_map: dict[tuple, set[str]] = {}
    for directive in directives:
        path = document_paths.get(
            directive.evidence.document_id, directive.evidence.document_name
        )
        semantic_class, eligible = _semantic_class(directive)
        sources = sorted(directive.provenance_sources)
        row = {
            "fingerprint": directive_fingerprint(
                directive,
                path=path,
                semantic_class=semantic_class,
                eligible=eligible,
            ),
            "sources": sources,
            "contributing_evidence": (
                [
                    [path, line]
                    for line in range(
                        directive.evidence.line_start,
                        directive.evidence.line_end + 1,
                    )
                ]
                if directive.kind == "clarification"
                else []
            ),
        }
        if directive.kind == "clarification":
            row["contributing_evidence_sources"] = [
                {"path": path, "line": line, "sources": sources}
                for line in range(
                    directive.evidence.line_start,
                    directive.evidence.line_end + 1,
                )
            ]
        directive_rows.append(row)
        evidence = directive.evidence
        key = (
            evidence.document_id,
            evidence.document_name,
            evidence.line_start,
            evidence.line_end,
            evidence.text,
        )
        evidence_source_map.setdefault(key, set()).update(sources)

    issue_rows: list[dict] = []
    for issue in issues:
        evidence_details = []
        aggregate_sources: set[str] = set()
        for evidence in issue.evidence:
            key = (
                evidence.document_id,
                evidence.document_name,
                evidence.line_start,
                evidence.line_end,
                evidence.text,
            )
            sources = sorted(evidence_source_map.get(key, set()))
            aggregate_sources.update(sources)
            evidence_details.append(
                {
                    "path": document_paths.get(
                        evidence.document_id, evidence.document_name
                    ),
                    "line": evidence.line_start,
                    "sources": sources,
                }
            )
        issue_rows.append(
            {
                "fingerprint": issue_fingerprint(
                    issue, document_paths=document_paths
                ),
                # `deterministic` is the frozen runner token; the explicit type
                # below documents the derivation without widening that enum.
                "derivation": ["deterministic"],
                "derivation_type": "deterministic_rule",
                "evidence_sources": sorted(aggregate_sources),
                "evidence_source_details": sorted(
                    evidence_details,
                    key=lambda row: (row["path"], row["line"], row["sources"]),
                ),
                "contributing_evidence": sorted(
                    evidence_details,
                    key=lambda row: (row["path"], row["line"], row["sources"]),
                ),
            }
        )

    return {
        "schema_version": 1,
        "directives": sorted(directive_rows, key=lambda row: row["fingerprint"]),
        "issues": sorted(issue_rows, key=lambda row: row["fingerprint"]),
    }


def _dedupe_issues(issues: list[ConsistencyIssue]) -> list[ConsistencyIssue]:
    """Keep one issue per category, canonical metadata and evidence pair."""
    result: list[ConsistencyIssue] = []
    seen: set[tuple] = set()
    for issue in issues:
        evidence_key = tuple(
            sorted(
                (
                    span.document_id,
                    span.line_start,
                    span.line_end,
                )
                for span in issue.evidence
            )
        )
        metadata_key = json.dumps(issue.metadata, ensure_ascii=False, sort_keys=True)
        key = (issue.category.value, metadata_key, evidence_key)
        if key not in seen:
            result.append(issue)
            seen.add(key)
    return result


def _bind_directive_context(
    directive: ParsedDirective, document: DocumentInput | None
) -> ParsedDirective:
    if document is None:
        return directive
    attrs = dict(directive.attrs)
    if document.role:
        attrs["document_role"] = document.role
    if document.scope:
        attrs["story_scope"] = document.scope
    return directive.model_copy(update={"attrs": attrs})


def _label_closed_normalizer_directive(
    directive: ParsedDirective,
) -> ParsedDirective:
    """Attach labels at the deterministic normalizer's output boundary.

    This applies only to newly created rows from its closed recognizers. It is
    deliberately unrelated to a generic ``baseline`` provenance label.
    """
    attrs = dict(directive.attrs)
    text = directive.evidence.text
    kind = directive.kind
    predicate = attrs.get("predicate", "")
    closed = bool(
        (kind == "item" and re.search(r"保管|持有|掌管|移交|归还|接收", text))
        or (kind == "uses" and re.search(r"取出|拿出|使用|启用|按下|盖下|插入", text))
        or (kind == "world_rule" and re.search(r"失效|无法|不能|禁止|不得", text))
        or (kind == "world_assert" and re.search(r"发动|使用|施展|启动|开启", text))
        or (
            kind == "fact"
            and predicate.startswith("body_state:")
            and re.search(r"失去|截去|没有了|完好|健全|没有受伤", text)
        )
        or (
            kind == "fact"
            and predicate in {"mobility_permission", "rule_exception"}
            and re.search(r"许可|权限|通行证|豁免|获准|授权|例外资格", text)
        )
        or (
            kind in {"knows", "claims_knows"}
            and re.search(r"得知|获知|知道|说出|提到|引用|告诉|告知|展示", text)
        )
    )
    if not closed:
        return directive.model_copy(
            update={"provenance_sources": frozenset({"baseline"})}
        )
    if kind == "claims_knows":
        attrs.update(
            modality="reported",
            source_scope="character_dialogue",
            certainty="certain",
        )
    elif kind in {
        "fact", "event", "knows", "item", "uses", "world_rule", "world_assert"
    }:
        attrs.update(
            modality=(
                "negated"
                if attrs.get("polarity") == "negative"
                else "conditional_rule"
                if kind == "world_rule" and attrs.get("condition")
                else "asserted"
            ),
            source_scope="world_rule" if kind == "world_rule" else "narrator",
            certainty="certain",
        )
    return directive.model_copy(
        update={
            "attrs": attrs,
            "provenance_sources": frozenset({"baseline"}),
        }
    )


def _bind_document_context(
    directives: list[ParsedDirective], document: DocumentInput
) -> list[ParsedDirective]:
    return [_bind_directive_context(directive, document) for directive in directives]
