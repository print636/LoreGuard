from __future__ import annotations

from dataclasses import dataclass, field
import json
from time import perf_counter
from typing import Callable, Protocol

from .domain import ConsistencyIssue, ParsedDirective
from .parser import ParsedDocument, parse_document
from .rules import detect_issues


@dataclass(frozen=True, slots=True)
class DocumentInput:
    id: str
    name: str
    content: str


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
        return parse_document(document.id, document.name, document.content)


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

    def run(self, documents: list[DocumentInput], on_stage: Callable[[str, int, str], None] | None = None) -> PipelineResult:
        notify = on_stage or (lambda *_: None)
        started = perf_counter()
        first_progress_ms: float | None = None

        def emit(stage: str, progress: int, message: str) -> None:
            nonlocal first_progress_ms
            if first_progress_ms is None:
                first_progress_ms = (perf_counter() - started) * 1000
            notify(stage, progress, message)

        result = PipelineResult()
        if hasattr(self.extractor, "begin_run"):
            self.extractor.begin_run()
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
        for document in documents:
            parsed = self.extractor.extract(document)
            result.directives.extend(parsed.directives)
            result.warnings.extend(f"{document.name}: {warning}" for warning in parsed.warnings)
            result.prompt_tokens += parsed.prompt_tokens
            result.completion_tokens += parsed.completion_tokens
            result.model_used = result.model_used or parsed.model_used
        extract_ms = (perf_counter() - extract_started) * 1000

        index_started = perf_counter()
        normalized = self.normalizer.enrich(documents, result.directives)
        result.directives = normalized.directives
        result.warnings.extend(normalized.warnings)
        alias_result = canonicalize_entities(documents, result.directives)
        result.directives = alias_result.directives
        result.warnings.extend(alias_result.warnings)
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
        result.diagnostics = {
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
                "trace_count": len(alias_result.traces),
                "traces": alias_result.traces,
            },
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
