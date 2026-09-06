from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import re
import sys
from time import perf_counter
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domain import ConsistencyIssue, ParsedDirective
from app.model_extractor import ModelEnhancedExtractor
from app.pipeline import AnalysisPipeline, DocumentInput
from app.provider import OpenAICompatibleProvider, safe_thinking_configuration
from app.semantic_quality import eligible_for_deterministic_rules
from scripts.provider_instrumentation import CountingProvider


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE_ROOT = ROOT / "data" / "user-acceptance-v1"
NONCONFIRMED_CLASSES = {"open_question", "tentative", "quoted_claim", "clarification"}
NON_SEMANTIC_ATTRS = {
    "modality", "source_scope", "certainty", "document_role", "story_scope",
    "input_form", "original_kind", "polarity", "question_type", "category",
}

# The manifest intentionally freezes prohibitions as reviewer-facing prose.  A
# prose sentence must never be treated as executable merely because a fuzzy
# matcher found (or failed to find) similar words.  This table is the explicit
# scorer contract for every frozen v1 prohibition.  A newly added/edited
# manifest assertion is reported as unevaluated and fails closed until an
# executable rule is added here.
FORBIDDEN_ASSERTIONS: dict[str, dict[str, tuple[str, ...]]] = {
    "不得把是否等同于中的‘是’抽成肯定事实": {
        "pollution_ids": ("g-cart-question",),
    },
    "不得把两种可能食用结果抽成已发生事件": {
        "pollution_ids": ("g-food-hypothesis",),
    },
    "不得把已知谎言与旁白事实报成普通事实冲突": {
        "pollution_ids": ("g-lie",),
    },
    "不得把A路线观星厅与B路线货舱报成同刻多地点": {
        "unexpected_issue_all_ids": ("g-a-location-1", "g-b-location"),
    },
    "不得因陆遥站在门另一侧就擅自断言他穿过了石门": {
        "unexpected_directive_ids": ("n-at-inner-side",),
        "unexpected_issue_all_ids": ("n-bell-rule", "n-at-inner-side"),
    },
    "不得把暗渠当作已存在的地点或既成路线": {
        "pollution_ids": ("n-channel-hypothesis",),
    },
    "不得从‘若下雨则熄灯’反推‘不下雨就不会熄灯’": {
        "unexpected_directive_ids": ("n-rain-condition",),
        "unexpected_issue_all_ids": ("n-rain-condition", "n-lamp-event"),
    },
    "不得把‘门没有打开’抽成开门事件": {
        "unexpected_directive_ids": ("n-door-negative",),
    },
    "不得把角色谎言当作旁白事实": {
        "pollution_ids": ("n-lie",),
    },
    "不得把匿名信内容当作蓝色账册已被烧毁": {
        "pollution_ids": ("e-letter-claim",),
    },
    "不得因信标为蓝色且北门开启而违反单向条件规则": {
        "unexpected_issue_all_ids": ("e-red-rule", "e-blue-gate"),
    },
    "不得用尚未回答的另一把星钥问题自动消除或自动证明冲突": {
        "pollution_ids": ("e-duplicate-key-question",),
        "missing_conflict_ids": ("e-issue-key",),
    },
}


def model_pipeline() -> AnalysisPipeline:
    provider = CountingProvider(OpenAICompatibleProvider())
    return AnalysisPipeline(extractor=ModelEnhancedExtractor(provider=provider))


def load_manifest(suite_root: Path = DEFAULT_SUITE_ROOT) -> dict:
    return json.loads((suite_root / "manifest.json").read_text(encoding="utf-8"))


def select_cases(manifest: dict, suite: str, case_id: str | None = None) -> list[dict]:
    cases = list(manifest["cases"])
    if case_id:
        selected = [case for case in cases if case["id"] == case_id]
        if not selected:
            raise ValueError(f"unknown case id: {case_id}")
        return selected
    return cases[:1] if suite == "pilot" else cases


def _execution_classification(execution: dict | None) -> tuple[bool, bool]:
    if not isinstance(execution, dict):
        return False, False
    fields = (
        "total_chunks", "attempted_chunks", "succeeded_chunks",
        "failed_chunks", "skipped_chunks", "invalid_records",
        "empty_response_chunks",
    )
    if any(type(execution.get(key)) is not int or execution[key] < 0 for key in fields):
        return False, False
    if (
        execution["attempted_chunks"]
        != execution["succeeded_chunks"] + execution["failed_chunks"]
        or execution["total_chunks"]
        != execution["attempted_chunks"] + execution["skipped_chunks"]
        or execution["empty_response_chunks"] > execution["succeeded_chunks"]
    ):
        return False, False
    participating = execution["succeeded_chunks"] > 0
    disposition_supplied = any(
        execution.get(key) is not None
        for key in ("unresolved_invalid_records", "recovered_invalid_records")
    )
    if disposition_supplied:
        disposition_fields = (
            "unresolved_invalid_records",
            "recovered_invalid_records",
            "repair_post_invalid",
        )
        if (
            any(
                type(execution.get(key)) is not int or execution[key] < 0
                for key in disposition_fields
            )
            or type(execution.get("repair_failed")) is not bool
            or execution["invalid_records"]
            != execution["unresolved_invalid_records"]
            + execution["recovered_invalid_records"]
        ):
            return participating, False
        final_invalid = execution["unresolved_invalid_records"]
        repair_complete = bool(
            execution["repair_failed"] is False
            and execution["repair_post_invalid"] == 0
            and (
                execution["recovered_invalid_records"] == 0
                or execution.get("repair_succeeded") is True
            )
        )
    else:
        # Legacy reports have no final-disposition fields; preserve the old
        # fail-closed interpretation of observed invalid records/reason codes.
        final_invalid = execution["invalid_records"]
        repair_complete = not execution.get("reason_codes")
    complete = bool(
        execution.get("enabled") is True
        and execution.get("configured") is True
        and execution["total_chunks"] > 0
        and execution["total_chunks"] == execution["attempted_chunks"]
        == execution["succeeded_chunks"]
        and execution["failed_chunks"] == 0
        and execution["skipped_chunks"] == 0
        and final_invalid == 0
        and execution["empty_response_chunks"] == 0
        and repair_complete
    )
    return participating, complete


def _provider_stats(pipeline: AnalysisPipeline) -> dict:
    provider = getattr(getattr(pipeline, "extractor", None), "provider", None)
    instrumented = all(
        hasattr(provider, field) for field in ("requested", "succeeded", "failed")
    )
    return {
        "instrumented": instrumented,
        "requested": int(getattr(provider, "requested", 0)) if instrumented else None,
        "succeeded": int(getattr(provider, "succeeded", 0)) if instrumented else None,
        "failed": int(getattr(provider, "failed", 0)) if instrumented else None,
        "last_error_type": getattr(provider, "last_error_type", None),
    }


def _runtime_provider(provider: Any) -> Any:
    """Return the object that owns the retry policy behind instrumentation."""
    return getattr(provider, "delegate", provider)


def _first_setting(settings: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        value = getattr(settings, name, None)
        if value is not None:
            return value
    return None


def _model_identifier_sha256(model: Any) -> str | None:
    if not isinstance(model, str) or not model.strip():
        return None
    return hashlib.sha256(model.strip().encode("utf-8")).hexdigest()


def _effective_provider_configuration(provider: Any) -> dict:
    """Capture reproducibility fields without retaining keys or endpoints."""
    settings = getattr(provider, "settings", None)
    runtime_provider = _runtime_provider(provider)
    retry_policy = getattr(runtime_provider, "retry_policy", None)
    model_hash = _model_identifier_sha256(getattr(settings, "openai_model", None))
    return {
        "model_identifier_kind": "sha256" if model_hash else None,
        "model_identifier_sha256": model_hash,
        "timeout_seconds": getattr(settings, "provider_timeout_seconds", None),
        "max_attempts": getattr(
            retry_policy,
            "max_attempts",
            getattr(settings, "provider_max_attempts", None),
        ),
        "circuit_failure_threshold": getattr(
            settings, "model_circuit_breaker_failed_documents", None
        ),
        "per_run_token_budget": getattr(settings, "per_run_token_budget", None),
        "completion_token_limit": _first_setting(
            settings,
            (
                "provider_max_completion_tokens",
                "model_max_completion_tokens",
                "max_completion_tokens",
            ),
        ),
        "generation_limit": _first_setting(
            settings,
            ("provider_generation_limit", "model_generation_limit"),
        ),
        "thinking": safe_thinking_configuration(settings),
    }


def _safe_model_execution(execution: Any) -> dict | None:
    if not isinstance(execution, dict):
        return None
    safe_fields = (
        "enabled",
        "configured",
        "total_chunks",
        "attempted_chunks",
        "succeeded_chunks",
        "failed_chunks",
        "skipped_chunks",
        "invalid_records",
        "unresolved_invalid_records",
        "recovered_invalid_records",
        "empty_response_chunks",
        "repair_attempted",
        "repair_succeeded",
        "repair_failed",
        "repair_post_invalid",
        "repair_salvaged",
        "repair_dropped",
        "repair_final_path",
        "reason_codes",
    )
    safe = {key: execution[key] for key in safe_fields if key in execution}
    provider_call_fields = (
        "status",
        "category",
        "attempt",
        "elapsed_ms",
        "input_chars",
        "response_chars",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "http_status",
    )
    provider_calls = execution.get("provider_calls")
    if isinstance(provider_calls, list):
        safe_calls = []
        for row in provider_calls:
            if not isinstance(row, dict):
                continue
            safe_call = {
                key: row[key] for key in provider_call_fields if key in row
            }
            if row.get("purpose") in {"extract", "repair"}:
                safe_call["purpose"] = row["purpose"]
            safe_calls.append(safe_call)
        safe["provider_calls"] = safe_calls
    elif provider_calls is None and "provider_calls" in execution:
        safe["provider_calls"] = None
    document_fields = ("document_id", "document_name", *safe_fields)
    documents = execution.get("documents")
    if isinstance(documents, list):
        safe["documents"] = [
            {key: row[key] for key in document_fields if key in row}
            for row in documents
            if isinstance(row, dict)
        ]
    return safe


def _documents(case: dict, suite_root: Path) -> tuple[list[DocumentInput], dict[str, str]]:
    documents = []
    id_to_path = {}
    for row in case["documents"]:
        relative_path = row["path"]
        path = (suite_root / relative_path).resolve()
        path.relative_to(suite_root.resolve())
        document_id = f"{case['id']}:{relative_path}"
        id_to_path[document_id] = relative_path
        documents.append(
            DocumentInput(
                id=document_id,
                name=Path(relative_path).name,
                content=path.read_text(encoding="utf-8"),
                role=row.get("role"),
                scope=row.get("scope"),
            )
        )
    return documents, id_to_path


def _semantic_class(directive: ParsedDirective) -> str:
    if directive.kind == "open_question":
        return "open_question"
    if directive.kind == "tentative_fact":
        return "tentative"
    if directive.kind == "character_claim":
        return "quoted_claim"
    if directive.kind == "clarification":
        return "clarification"
    if not eligible_for_deterministic_rules(directive):
        return "noncanonical_other"
    if (
        directive.kind == "world_rule"
        or directive.attrs.get("document_role") == "canonical_setting"
        or directive.attrs.get("source_scope") == "world_rule"
    ):
        return "confirmed_canonical"
    return "confirmed_narrative"


def _semantic_text(directive: ParsedDirective) -> str:
    return " ".join(
        str(value)
        for key, value in sorted(directive.attrs.items())
        if key not in NON_SEMANTIC_ATTRS and str(value).strip()
    )


def _directive_fingerprint(
    directive: ParsedDirective, path: str | None = None
) -> str:
    evidence = directive.evidence
    return hashlib.sha256(
        json.dumps(
            {
                "kind": directive.kind,
                "class": _semantic_class(directive),
                "eligible": eligible_for_deterministic_rules(directive),
                "path": path or evidence.document_name,
                "line_start": evidence.line_start,
                "line_end": evidence.line_end,
                "attrs": sorted(directive.attrs.items()),
            },
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _provenance_lookup(provenance: Any, key: str) -> dict[str, dict]:
    """Read only the frozen v1 provenance sidecar; legacy data is unknown."""
    if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
        return {}
    rows = provenance.get(key)
    if not isinstance(rows, list):
        return {}
    result: dict[str, dict] = {}
    ambiguous: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        fingerprint = row.get("fingerprint")
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            continue
        # Duplicate entries are ambiguous rather than last-write-wins.
        if fingerprint in result or fingerprint in ambiguous:
            result.pop(fingerprint, None)
            ambiguous.add(fingerprint)
            continue
        result[fingerprint] = row
    return result


def _safe_sources(value: Any) -> list[str]:
    allowed = {"baseline", "model", "deterministic"}
    if not isinstance(value, list) or any(item not in allowed for item in value):
        return []
    return sorted(set(value))


def _safe_contributing_evidence(value: Any) -> list[list[str | int]]:
    if not isinstance(value, list):
        return []
    rows = []
    for item in value:
        if (
            isinstance(item, list)
            and len(item) == 2
            and isinstance(item[0], str)
            and type(item[1]) is int
            and item[1] >= 1
        ):
            rows.append([item[0], item[1]])
    return sorted(rows, key=lambda row: (str(row[0]), int(row[1])))


def _units(value: str) -> set[str]:
    compact = "".join(re.findall(r"[\u3400-\u9fffA-Za-z0-9]+", value)).lower()
    chars = set(compact)
    bigrams = {compact[index:index + 2] for index in range(max(0, len(compact) - 1))}
    return chars | bigrams


def _similarity(expected_summary: str, actual_text: str) -> float:
    expected = _units(expected_summary)
    actual = _units(actual_text)
    if not expected or not actual:
        return 0.0
    common = len(expected & actual)
    precision = common / len(actual)
    recall = common / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _actual_directives(
    directives: list[ParsedDirective],
    id_to_path: dict[str, str],
    provenance: dict | None = None,
) -> list[dict]:
    provenance_rows = _provenance_lookup(provenance, "directives")
    rows = []
    for index, directive in enumerate(directives):
        evidence = directive.evidence
        path = id_to_path.get(evidence.document_id, evidence.document_name)
        semantic_text = _semantic_text(directive)
        safe_fingerprint = _directive_fingerprint(directive, path)
        provenance_row = provenance_rows.get(safe_fingerprint, {})
        sources = _safe_sources(provenance_row.get("sources"))
        contributing_evidence = _safe_contributing_evidence(
            provenance_row.get("contributing_evidence")
        )
        rows.append(
            {
                "index": index,
                "directive": directive,
                "kind": directive.kind,
                "class": _semantic_class(directive),
                "eligible": eligible_for_deterministic_rules(directive),
                "path": path,
                "line_start": evidence.line_start,
                "line_end": evidence.line_end,
                "semantic_text": semantic_text,
                "fingerprint": safe_fingerprint,
                "sources": sources,
                "model_source": "model" in sources,
                "contributing_evidence": contributing_evidence,
            }
        )
    return rows


def _score_semantics(case: dict, actual: list[dict]) -> dict:
    candidates = []
    for expected_index, expected in enumerate(case["required_semantics"]):
        source = expected["evidence"]
        for row in actual:
            if row["path"] != source["path"]:
                continue
            if not (row["line_start"] <= source["line"] <= row["line_end"]):
                continue
            if row["class"] != expected["class"]:
                continue
            if row["eligible"] is not expected["enters_conflict_engine"]:
                continue
            score = _similarity(expected["summary"], row["semantic_text"])
            if score >= 0.18:
                candidates.append((score, expected_index, row["index"]))

    matched_expected: set[int] = set()
    matched_actual: set[int] = set()
    matches = []
    for score, expected_index, actual_index in sorted(candidates, reverse=True):
        if expected_index in matched_expected or actual_index in matched_actual:
            continue
        matched_expected.add(expected_index)
        matched_actual.add(actual_index)
        expected = case["required_semantics"][expected_index]
        row = next(item for item in actual if item["index"] == actual_index)
        matches.append(
            {
                "expected_id": expected["id"],
                "class": expected["class"],
                "path": row["path"],
                "line": expected["evidence"]["line"],
                "actual_kind": row["kind"],
                "similarity": round(score, 4),
                "sources": row["sources"],
                "model_source": row["model_source"],
                "actual_index": actual_index,
            }
        )

    missing = []
    for index, expected in enumerate(case["required_semantics"]):
        if index in matched_expected:
            continue
        source = expected["evidence"]
        observed = sorted(
            {
                f"{row['kind']}:{row['class']}:eligible={row['eligible']}"
                for row in actual
                if row["path"] == source["path"]
                and row["line_start"] <= source["line"] <= row["line_end"]
            }
        )
        missing.append(
            {
                "expected_id": expected["id"],
                "class": expected["class"],
                "path": source["path"],
                "line": source["line"],
                "observed": observed,
            }
        )
    missing_model_provenance = [
        {
            "expected_id": row["expected_id"],
            "actual_kind": row["actual_kind"],
            "sources": row["sources"],
        }
        for row in matches
        if not row["model_source"]
    ]
    return {
        "expected": len(case["required_semantics"]),
        "matched": len(matches),
        "evidence_location_hit_rate": (
            len(matches) / len(case["required_semantics"])
            if case["required_semantics"] else 1.0
        ),
        "matches": sorted(matches, key=lambda row: row["expected_id"]),
        "missing": missing,
        "model_provenance_matched": len(matches) - len(missing_model_provenance),
        "missing_model_provenance": missing_model_provenance,
        "unmatched_actual_count": len(actual) - len(matched_actual),
        "matched_actual_indices": sorted(matched_actual),
    }


def _issue_fingerprint(issue: ConsistencyIssue, id_to_path: dict[str, str]) -> str:
    row = _issue_row(issue, id_to_path)
    return hashlib.sha256(
        json.dumps(
            {"category": row["category"], "evidence": row["evidence"]},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _issue_row(
    issue: ConsistencyIssue,
    id_to_path: dict[str, str],
    provenance_rows: dict[str, dict] | None = None,
) -> dict:
    evidence = sorted(
        {
            (id_to_path.get(span.document_id, span.document_name), span.line_start)
            for span in issue.evidence
        }
    )
    row = {"category": issue.category.value, "evidence": [[path, line] for path, line in evidence]}
    fingerprint = hashlib.sha256(
        json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    provenance = (provenance_rows or {}).get(fingerprint, {})
    evidence_sources = _safe_sources(provenance.get("evidence_sources"))
    derivation = _safe_sources(provenance.get("derivation"))
    return {
        **row,
        "fingerprint": fingerprint,
        "evidence_sources": evidence_sources,
        "derivation": derivation,
        "model_evidence": "model" in evidence_sources,
        "deterministic_derived": "deterministic" in derivation,
    }


def _expected_refs(case: dict, finding: dict) -> set[tuple[str, int]]:
    semantics = {row["id"]: row for row in case["required_semantics"]}
    return {
        (semantics[semantic_id]["evidence"]["path"], semantics[semantic_id]["evidence"]["line"])
        for semantic_id in finding["evidence_ids"]
    }


def _score_conflicts(
    case: dict,
    issues: list[ConsistencyIssue],
    id_to_path: dict[str, str],
    provenance: dict | None = None,
) -> dict:
    expected = [row for row in case["expected_findings"] if row["result"] == "confirmed_conflict"]
    provenance_rows = _provenance_lookup(provenance, "issues")
    actual = [_issue_row(issue, id_to_path, provenance_rows) for issue in issues]
    candidates = []
    for expected_index, finding in enumerate(expected):
        required = _expected_refs(case, finding)
        for actual_index, issue in enumerate(actual):
            if issue["category"] != finding["category"]:
                continue
            observed = {(path, line) for path, line in issue["evidence"]}
            if required.issubset(observed):
                candidates.append((len(required), expected_index, actual_index))
    matched_expected: set[int] = set()
    matched_actual: set[int] = set()
    matches = []
    for _, expected_index, actual_index in sorted(candidates, reverse=True):
        if expected_index in matched_expected or actual_index in matched_actual:
            continue
        matched_expected.add(expected_index)
        matched_actual.add(actual_index)
        matches.append({"expected_id": expected[expected_index]["id"], **actual[actual_index]})
    missing = []
    for index, finding in enumerate(expected):
        if index in matched_expected:
            continue
        missing.append(
            {
                "expected_id": finding["id"],
                "category": finding["category"],
                "required_evidence": [list(row) for row in sorted(_expected_refs(case, finding))],
                "same_category_predictions": [row for row in actual if row["category"] == finding["category"]],
            }
        )
    missing_model_provenance = [
        {"expected_id": row["expected_id"], "evidence_sources": row["evidence_sources"]}
        for row in matches
        if not row["model_evidence"]
    ]
    missing_deterministic_derivation = [
        {"expected_id": row["expected_id"], "derivation": row["derivation"]}
        for row in matches
        if not row["deterministic_derived"]
    ]
    return {
        "expected": len(expected),
        "matched_with_complete_evidence": len(matches),
        "category_hits": sum(any(issue["category"] == row["category"] for issue in actual) for row in expected),
        "matches": matches,
        "model_evidence_matched": len(matches) - len(missing_model_provenance),
        "missing_model_evidence": missing_model_provenance,
        "deterministic_derived_matched": (
            len(matches) - len(missing_deterministic_derivation)
        ),
        "missing_deterministic_derivation": missing_deterministic_derivation,
        "missing_or_incomplete": missing,
        "actual": actual,
        "unexpected": [row for index, row in enumerate(actual) if index not in matched_actual],
    }


def _score_clarifications(case: dict, actual: list[dict]) -> dict:
    expected = [row for row in case["expected_findings"] if row["result"] == "clarification"]
    clarification_rows = [row for row in actual if row["kind"] == "clarification"]
    matched_actual: set[int] = set()
    matches = []
    missing = []
    for finding in expected:
        refs = _expected_refs(case, finding)
        candidate = next(
            (
                row for row in clarification_rows
                if row["index"] not in matched_actual
                and row["directive"].attrs.get("category") == finding["category"]
                and refs.issubset(
                    {
                        (row["path"], line)
                        for line in range(row["line_start"], row["line_end"] + 1)
                    }
                    | {
                        (str(path), int(line))
                        for path, line in row["contributing_evidence"]
                    }
                )
            ),
            None,
        )
        if candidate is None:
            missing.append(
                {
                    "expected_id": finding["id"],
                    "category": finding["category"],
                    "required_evidence": [list(row) for row in sorted(refs)],
                }
            )
            continue
        matched_actual.add(candidate["index"])
        matches.append(
            {
                "expected_id": finding["id"],
                "category": finding["category"],
                "evidence": [list(row) for row in sorted(refs)],
                "sources": candidate["sources"],
                "model_source": candidate["model_source"],
            }
        )
    missing_model_provenance = [
        {"expected_id": row["expected_id"], "sources": row["sources"]}
        for row in matches
        if not row["model_source"]
    ]
    return {
        "expected": len(expected),
        "matched_with_complete_evidence": len(matches),
        "model_provenance_matched": len(matches) - len(missing_model_provenance),
        "matches": matches,
        "missing_or_incomplete": missing,
        "missing_model_provenance": missing_model_provenance,
    }


def _pollution(case: dict, actual: list[dict], issues: list[ConsistencyIssue], id_to_path: dict[str, str]) -> dict:
    issue_refs = {
        (id_to_path.get(span.document_id, span.document_name), span.line_start)
        for issue in issues for span in issue.evidence
    }
    polluted = []
    expected = [row for row in case["required_semantics"] if row["class"] in NONCONFIRMED_CLASSES]
    for semantic in expected:
        source = semantic["evidence"]
        eligible_rows = [
            row for row in actual
            if row["path"] == source["path"]
            and row["line_start"] <= source["line"] <= row["line_end"]
            and row["eligible"]
        ]
        used_by_issue = (source["path"], source["line"]) in issue_refs
        if eligible_rows or used_by_issue:
            polluted.append(
                {
                    "expected_id": semantic["id"],
                    "path": source["path"],
                    "line": source["line"],
                    "eligible_kinds": sorted({row["kind"] for row in eligible_rows}),
                    "used_by_issue": used_by_issue,
                }
            )
    return {"expected_nonconfirmed": len(expected), "polluted": len(polluted), "details": polluted}


def _forbidden_findings(
    case: dict,
    actual: list[dict],
    semantics: dict,
    conflicts: dict,
    pollution: dict,
) -> dict:
    """Execute and report every frozen prohibition; unknown prose fails closed."""
    semantic_by_id = {row["id"]: row for row in case["required_semantics"]}
    matched_actual = set(semantics["matched_actual_indices"])
    unmatched = [row for row in actual if row["index"] not in matched_actual]
    actual_issues = conflicts["actual"]
    results = []
    for index, finding in enumerate(case.get("forbidden_findings", []), start=1):
        rule = FORBIDDEN_ASSERTIONS.get(finding)
        if rule is None:
            results.append(
                {
                    "index": index,
                    "finding": finding,
                    "status": "unevaluated",
                    "rule": None,
                    "related_semantic_ids": [],
                    "violations": [{"type": "unsupported_manifest_assertion"}],
                }
            )
            continue
        configured_ids = sorted({
            semantic_id
            for key, semantic_ids in rule.items()
            if key.endswith("_ids") and key != "missing_conflict_ids"
            for semantic_id in semantic_ids
        })
        violations = []
        pollution_ids = set(rule.get("pollution_ids", ()))
        for detail in pollution["details"]:
            if detail["expected_id"] in pollution_ids:
                violations.append(
                    {"type": "nonconfirmed_pollution", "expected_id": detail["expected_id"]}
                )
        directive_ids = set(rule.get("unexpected_directive_ids", ()))
        for row in unmatched:
            matched_ids = []
            for semantic_id in directive_ids:
                source = semantic_by_id[semantic_id]["evidence"]
                if (
                    row["path"] == source["path"]
                    and row["line_start"] <= source["line"] <= row["line_end"]
                ):
                    matched_ids.append(semantic_id)
            if matched_ids:
                violations.append(
                    {
                        "type": "unexpected_directive",
                        "semantic_ids": sorted(matched_ids),
                        "fingerprint": row["fingerprint"],
                        "kind": row["kind"],
                    }
                )
        issue_ids = tuple(rule.get("unexpected_issue_all_ids", ()))
        if issue_ids:
            required_refs = {
                (
                    semantic_by_id[semantic_id]["evidence"]["path"],
                    semantic_by_id[semantic_id]["evidence"]["line"],
                )
                for semantic_id in issue_ids
            }
            for issue in actual_issues:
                observed = {(path, line) for path, line in issue["evidence"]}
                if required_refs.issubset(observed):
                    violations.append(
                        {
                            "type": "forbidden_conflict",
                            "semantic_ids": list(issue_ids),
                            "fingerprint": issue["fingerprint"],
                            "category": issue["category"],
                        }
                    )
        missing_conflict_ids = set(rule.get("missing_conflict_ids", ()))
        for missing in conflicts["missing_or_incomplete"]:
            if missing["expected_id"] in missing_conflict_ids:
                violations.append(
                    {
                        "type": "required_conflict_missing_or_incomplete",
                        "expected_id": missing["expected_id"],
                    }
                )
        results.append(
            {
                "index": index,
                "finding": finding,
                "status": "hit" if violations else "passed",
                "rule": sorted(rule),
                "related_semantic_ids": configured_ids,
                "violations": violations,
            }
        )
    return {
        "expected": len(case.get("forbidden_findings", [])),
        "evaluated": sum(row["status"] != "unevaluated" for row in results),
        "unevaluated": sum(row["status"] == "unevaluated" for row in results),
        "hit_count": sum(row["status"] == "hit" for row in results),
        "results": results,
    }


def score_case(
    case: dict,
    directives: list[ParsedDirective],
    issues: list[ConsistencyIssue],
    id_to_path: dict[str, str],
    provenance: dict | None = None,
) -> dict:
    actual = _actual_directives(directives, id_to_path, provenance)
    semantics = _score_semantics(case, actual)
    conflicts = _score_conflicts(case, issues, id_to_path, provenance)
    clarifications = _score_clarifications(case, actual)
    pollution = _pollution(case, actual, issues, id_to_path)
    forbidden = _forbidden_findings(case, actual, semantics, conflicts, pollution)
    signature_payload = {
        "directives": sorted(row["fingerprint"] for row in actual),
        "issues": sorted(
            json.dumps(row, ensure_ascii=False, sort_keys=True)
            for row in [_issue_row(issue, id_to_path) for issue in issues]
        ),
    }
    signature = hashlib.sha256(
        json.dumps(signature_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    gate_pass = bool(
        semantics["matched"] == semantics["expected"]
        and semantics["model_provenance_matched"] == semantics["expected"]
        and conflicts["matched_with_complete_evidence"] == conflicts["expected"]
        and conflicts["model_evidence_matched"] == conflicts["expected"]
        and conflicts["deterministic_derived_matched"] == conflicts["expected"]
        and not conflicts["unexpected"]
        and clarifications["matched_with_complete_evidence"] == clarifications["expected"]
        and clarifications["model_provenance_matched"] == clarifications["expected"]
        and pollution["polluted"] == 0
        and forbidden["evaluated"] == forbidden["expected"]
        and forbidden["hit_count"] == 0
    )
    return {
        "case_id": case["id"],
        "persona": case["persona"],
        "gate_pass": gate_pass,
        "semantics": semantics,
        "conflicts": conflicts,
        "clarifications": clarifications,
        "nonconfirmed_pollution": pollution,
        "forbidden_findings": forbidden,
        "provenance": {
            "available": isinstance(provenance, dict),
            "schema_version": provenance.get("schema_version")
            if isinstance(provenance, dict)
            else None,
            "schema_supported": (
                isinstance(provenance, dict) and provenance.get("schema_version") == 1
            ),
            "directive_rows_available": bool(
                _provenance_lookup(provenance, "directives")
            ),
            "issue_rows_available": bool(_provenance_lookup(provenance, "issues")),
            "minimum_backend_contract": {
                "schema_version": 1,
                "directives": "final directive fingerprint -> sources; clarifications also list every contributing evidence path/line",
                "issues": "final issue fingerprint -> derivation including deterministic and evidence_sources including model",
                "dedupe_rule": "union baseline/model sources when final records are deduplicated",
            },
        },
        "result_signature_sha256": signature,
    }


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.999999)))
    return ordered[index]


def _stability(attempts: list[dict], selected_cases: list[dict], repeats: int) -> dict:
    rows = []
    for case in selected_cases:
        eligible = [
            row for row in attempts
            if row["case_id"] == case["id"] and row["full_model_attempt"]
        ]
        signatures = Counter(row["score"]["result_signature_sha256"] for row in eligible)
        rows.append(
            {
                "case_id": case["id"],
                "eligible_repeats": len(eligible),
                "requested_repeats": repeats,
                "distinct_result_sets": len(signatures),
                "stable": len(eligible) == repeats and len(signatures) == 1,
            }
        )
    return {"all_cases_stable": bool(rows) and all(row["stable"] for row in rows), "cases": rows}


def _contract_counts(cases: list[dict]) -> dict:
    semantics = [row for case in cases for row in case["required_semantics"]]
    findings = [row for case in cases for row in case["expected_findings"]]
    return {
        "cases": len(cases),
        "required_semantics": len(semantics),
        "confirmed_conflicts": sum(row["result"] == "confirmed_conflict" for row in findings),
        "clarifications": sum(row["result"] == "clarification" for row in findings),
        "nonconfirmed_semantics": sum(row["class"] in NONCONFIRMED_CLASSES for row in semantics),
        "forbidden_findings": sum(len(case.get("forbidden_findings", [])) for case in cases),
    }


def run_acceptance(
    manifest: dict,
    selected_cases: list[dict],
    *,
    suite_root: Path = DEFAULT_SUITE_ROOT,
    repeats: int = 1,
    max_total_tokens: int = 600_000,
    per_case_token_budget: int = 60_000,
    timeout_seconds: float | None = None,
    max_attempts: int | None = None,
    circuit_failure_threshold: int | None = None,
    pipeline_factory: Callable[[], AnalysisPipeline] = model_pipeline,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    if max_total_tokens <= 0 or per_case_token_budget <= 0:
        raise ValueError("token budgets must be positive")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if max_attempts is not None and max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    if circuit_failure_threshold is not None and circuit_failure_threshold <= 0:
        raise ValueError("circuit_failure_threshold must be positive")
    attempts = []
    accounted_total = prompt_total = completion_total = 0
    stop_reason: str | None = None
    started_all = perf_counter()

    for repeat_index in range(1, repeats + 1):
        for case in selected_cases:
            if accounted_total >= max_total_tokens:
                stop_reason = "token_safety_limit"
                break
            pipeline = pipeline_factory()
            provider = getattr(getattr(pipeline, "extractor", None), "provider", None)
            settings = getattr(provider, "settings", None)
            runtime_provider = _runtime_provider(provider)
            original_settings: list[tuple[str, Any]] = []
            setting_overrides = {
                "provider_timeout_seconds": timeout_seconds,
                "provider_max_attempts": max_attempts,
                "model_circuit_breaker_failed_documents": circuit_failure_threshold,
                "per_run_token_budget": min(
                    per_case_token_budget, max_total_tokens - accounted_total
                ),
            }
            if settings is not None:
                for name, value in setting_overrides.items():
                    if value is not None and hasattr(settings, name):
                        original_settings.append((name, getattr(settings, name)))
                        setattr(settings, name, value)
            original_retry_policy = getattr(runtime_provider, "retry_policy", None)
            original_retry_max_attempts = None
            retry_policy_replaced = False
            if max_attempts is not None and original_retry_policy is not None:
                try:
                    runtime_provider.retry_policy = replace(
                        original_retry_policy, max_attempts=max_attempts
                    )
                    retry_policy_replaced = True
                except TypeError:
                    original_retry_max_attempts = getattr(
                        original_retry_policy, "max_attempts", None
                    )
                    if original_retry_max_attempts is not None:
                        setattr(original_retry_policy, "max_attempts", max_attempts)
            effective_configuration = _effective_provider_configuration(provider)
            started = perf_counter()
            try:
                documents, id_to_path = _documents(case, suite_root)
                result = pipeline.run(documents)
            finally:
                if retry_policy_replaced:
                    runtime_provider.retry_policy = original_retry_policy
                elif original_retry_max_attempts is not None:
                    setattr(
                        original_retry_policy,
                        "max_attempts",
                        original_retry_max_attempts,
                    )
                if settings is not None:
                    for name, value in reversed(original_settings):
                        setattr(settings, name, value)
            case_total_duration_ms = (perf_counter() - started) * 1000
            provider_stats = _provider_stats(pipeline)
            execution = _safe_model_execution(
                getattr(result, "diagnostics", {}).get("model")
            )
            participating, full_model = _execution_classification(execution)
            reported = int(result.prompt_tokens + result.completion_tokens)
            conservative = int(getattr(getattr(pipeline, "extractor", None), "_run_tokens_used", 0))
            accounted = max(reported, conservative)
            accounted_total += accounted
            prompt_total += int(result.prompt_tokens)
            completion_total += int(result.completion_tokens)
            provenance = getattr(result, "diagnostics", {}).get("provenance")
            score = score_case(
                case, result.directives, result.issues, id_to_path, provenance
            )
            provenance_verified = bool(
                score["semantics"]["model_provenance_matched"]
                == score["semantics"]["expected"]
                and score["conflicts"]["model_evidence_matched"]
                == score["conflicts"]["expected"]
                and score["conflicts"]["deterministic_derived_matched"]
                == score["conflicts"]["expected"]
                and score["clarifications"]["model_provenance_matched"]
                == score["clarifications"]["expected"]
            )
            attempt = {
                "repeat_index": repeat_index,
                "case_id": case["id"],
                "document_count": len(documents),
                "case_total_duration_ms": round(case_total_duration_ms, 3),
                "duration_ms": round(case_total_duration_ms, 3),
                "model_participated": participating,
                "full_model_attempt": full_model,
                "provenance_verified": provenance_verified,
                "included_in_acceptance_metrics": full_model and provenance_verified,
                "model_execution": execution,
                "effective_model_configuration": effective_configuration,
                "provider_completions": provider_stats,
                "prompt_tokens": int(result.prompt_tokens),
                "completion_tokens": int(result.completion_tokens),
                "reported_tokens": reported,
                "conservative_accounted_tokens": accounted,
                "warning_count": len(result.warnings),
                "score": score,
            }
            attempts.append(attempt)
            if progress:
                progress(
                    {
                        "event": "case_completed",
                        "repeat_index": repeat_index,
                        "case_id": case["id"],
                        "full_model_attempt": full_model,
                        "gate_pass": score["gate_pass"] if full_model else False,
                        "provenance_verified": provenance_verified,
                        "reported_tokens_so_far": prompt_total + completion_total,
                        "case_total_duration_ms": round(case_total_duration_ms, 3),
                    }
                )
            # One all-failed real-provider attempt is enough evidence of a systemic
            # compatibility/transport problem. Do not burn further requests.
            if (
                provider_stats["instrumented"]
                and provider_stats["requested"] > 0
                and provider_stats["succeeded"] == 0
                and provider_stats["failed"] == provider_stats["requested"]
            ):
                stop_reason = "systemic_provider_failure"
                break
        if stop_reason:
            break

    full_model_attempts = [row for row in attempts if row["full_model_attempt"]]
    strict_attempts = [row for row in attempts if row["included_in_acceptance_metrics"]]
    durations = [float(row["case_total_duration_ms"]) for row in attempts]
    stability = _stability(attempts, selected_cases, repeats)
    requested_attempts = len(selected_cases) * repeats
    aggregate = {
        "required_semantics": sum(row["score"]["semantics"]["expected"] for row in strict_attempts),
        "matched_semantics": sum(row["score"]["semantics"]["matched"] for row in strict_attempts),
        "expected_conflicts": sum(row["score"]["conflicts"]["expected"] for row in strict_attempts),
        "matched_conflicts": sum(row["score"]["conflicts"]["matched_with_complete_evidence"] for row in strict_attempts),
        "expected_clarifications": sum(row["score"]["clarifications"]["expected"] for row in strict_attempts),
        "matched_clarifications": sum(
            row["score"]["clarifications"]["matched_with_complete_evidence"]
            for row in strict_attempts
        ),
        "expected_nonconfirmed": sum(row["score"]["nonconfirmed_pollution"]["expected_nonconfirmed"] for row in strict_attempts),
        "nonconfirmed_polluted": sum(row["score"]["nonconfirmed_pollution"]["polluted"] for row in strict_attempts),
        "forbidden_findings_evaluated": sum(
            row["score"]["forbidden_findings"]["evaluated"] for row in strict_attempts
        ),
        "forbidden_findings_hit": sum(
            row["score"]["forbidden_findings"]["hit_count"] for row in strict_attempts
        ),
    }
    aggregate["semantic_evidence_hit_rate"] = (
        aggregate["matched_semantics"] / aggregate["required_semantics"]
        if aggregate["required_semantics"] else 0.0
    )
    full_success = bool(
        len(attempts) == requested_attempts
        and len(strict_attempts) == requested_attempts
        and all(row["score"]["gate_pass"] for row in strict_attempts)
        and stability["all_cases_stable"]
        and stop_reason is None
    )
    errors = [
        {
            "repeat_index": row["repeat_index"],
            "case_id": row["case_id"],
            "full_model_attempt": row["full_model_attempt"],
            "model_reason_codes": (row["model_execution"] or {}).get("reason_codes", []),
            "observed_invalid_records": (row["model_execution"] or {}).get(
                "invalid_records"
            ),
            "unresolved_invalid_records": (row["model_execution"] or {}).get(
                "unresolved_invalid_records"
            ),
            "recovered_invalid_records": (row["model_execution"] or {}).get(
                "recovered_invalid_records"
            ),
            "repair_final_path": (row["model_execution"] or {}).get(
                "repair_final_path"
            ),
            "empty_response_chunks": (row["model_execution"] or {}).get(
                "empty_response_chunks"
            ),
            "last_provider_error_type": row["provider_completions"]["last_error_type"],
            "semantic_missing": row["score"]["semantics"]["missing"],
            "conflict_missing_or_incomplete": row["score"]["conflicts"]["missing_or_incomplete"],
            "unexpected_conflicts": row["score"]["conflicts"]["unexpected"],
            "clarification_missing_or_incomplete": row["score"]["clarifications"]["missing_or_incomplete"],
            "semantic_missing_model_provenance": row["score"]["semantics"]["missing_model_provenance"],
            "conflict_missing_model_evidence": row["score"]["conflicts"]["missing_model_evidence"],
            "conflict_missing_deterministic_derivation": row["score"]["conflicts"]["missing_deterministic_derivation"],
            "clarification_missing_model_provenance": row["score"]["clarifications"]["missing_model_provenance"],
            "nonconfirmed_pollution": row["score"]["nonconfirmed_pollution"]["details"],
            "forbidden_findings_hit": [
                finding
                for finding in row["score"]["forbidden_findings"]["results"]
                if finding["status"] != "passed"
            ],
        }
        for row in attempts
        if not row["full_model_attempt"] or not row["score"]["gate_pass"]
    ]
    provider_counters_available = bool(attempts) and all(
        row["provider_completions"]["instrumented"] for row in attempts
    )
    provider_telemetry_available = bool(attempts) and all(
        isinstance((row["model_execution"] or {}).get("provider_calls"), list)
        for row in attempts
    )
    per_call_telemetry = (
        [
            call
            for row in attempts
            for call in row["model_execution"]["provider_calls"]
        ]
        if provider_telemetry_available
        else None
    )
    effective_configurations = []
    for row in attempts:
        configuration = row["effective_model_configuration"]
        if configuration not in effective_configurations:
            effective_configurations.append(configuration)
    return {
        "benchmark": {
            "name": "LoreGuard phase-1 AI-first model acceptance",
            "suite_id": manifest["suite_id"],
            "visibility": manifest["visibility"],
            "developer_visible": True,
            "blind_test": False,
            "human_annotated": False,
            "annotation_status": "Developer-authored expected manifest; not independent human or blind annotation.",
            "answer_isolation": "Only documents and their declared role/scope enter the pipeline; manifest expectations are scorer-only.",
            "metric_scope": "Acceptance metrics include only attempts whose structured execution shows all chunks succeeded with zero failed or skipped chunks, zero final unresolved invalid records, zero repair post-invalid records, no repair failure, conserved observed/recovered/unresolved counts, and zero empty responses. Legacy execution without final-disposition counters falls back to the observed invalid count and reason codes. Final records still require verified model provenance; a schema-valid empty response does not prove semantic coverage.",
            "provenance_contract": "Every matched semantic and clarification must have explicit model in its final directive sources; every conflict must be deterministic-derived from evidence_sources containing model. Missing, duplicate, unsupported or legacy provenance is unknown and fails closed.",
            "semantic_match_contract": "One-to-one match on exact document/line, expected semantic class and rule eligibility, then normalized lexical F1 >= 0.18. This fixed scorer is not a semantic judge.",
            "prompts_or_raw_response_bodies_recorded": False,
            "credentials_or_endpoint_recorded": False,
            "token_policy": "Accuracy first; token limits are high safety ceilings, not optimization targets.",
            "generated_at": datetime.now(UTC).isoformat(),
        },
        "coverage": {
            "selected_case_count": len(selected_cases),
            "requested_repeats": repeats,
            "requested_attempt_count": requested_attempts,
            "attempted_count": len(attempts),
            "full_model_attempt_count": len(full_model_attempts),
            "provenance_verified_attempt_count": len(strict_attempts),
            "unverified_provenance_attempt_count": sum(
                row["full_model_attempt"] and not row["provenance_verified"]
                for row in attempts
            ),
            "acceptance_metric_attempt_count": len(strict_attempts),
            "empty_response_attempt_count": sum(
                bool((row["model_execution"] or {}).get("empty_response_chunks"))
                for row in attempts
            ),
        },
        "contract": {
            "frozen_suite": _contract_counts(list(manifest["cases"])),
            "selected_cases_per_repeat": _contract_counts(selected_cases),
            "aggregate_note": "Aggregate counters below are repeat-weighted and include only complete-model attempts with verified final-record provenance.",
        },
        "stop_reason": stop_reason,
        "passed": full_success,
        "aggregate": aggregate,
        "effective_model_configurations": effective_configurations,
        "provider_completions": {
            "aggregation_level": "logical_provider_complete_calls",
            "availability": (
                "available" if provider_counters_available else "unavailable"
            ),
            **{
                key: (
                    sum(row["provider_completions"][key] for row in attempts)
                    if provider_counters_available
                    else None
                )
                for key in ("requested", "succeeded", "failed")
            },
            "per_call_telemetry_status": (
                "available" if provider_telemetry_available else "unavailable"
            ),
            "per_call_telemetry": per_call_telemetry,
            "per_call_telemetry_note": (
                "Each entry is one logical provider call; retry attempts are already "
                "aggregated by the provider. elapsed_ms is provider-call time and is "
                "distinct from case_total_duration_ms. Token telemetry is diagnostic "
                "and is not added again to the report's usage totals."
                if provider_telemetry_available
                else "Safe per-call telemetry is unavailable for legacy execution data."
            ),
        },
        "tokens": {
            "max_total_tokens": max_total_tokens,
            "per_case_safety_budget": per_case_token_budget,
            "prompt_tokens": prompt_total,
            "completion_tokens": completion_total,
            "reported_tokens": prompt_total + completion_total,
            "conservative_accounted_tokens": accounted_total,
        },
        "timing": {
            "wall_duration_ms": round((perf_counter() - started_all) * 1000, 3),
            "case_total_p50_ms": _percentile(durations, 0.50),
            "case_total_p95_ms": _percentile(durations, 0.95),
            "case_total_max_ms": max(durations) if durations else 0.0,
            "attempt_p50_ms": _percentile(durations, 0.50),
            "attempt_p95_ms": _percentile(durations, 0.95),
            "attempt_max_ms": max(durations) if durations else 0.0,
        },
        "stability": stability,
        "attempts": attempts,
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen phase-1 real-model acceptance suite.",
        epilog=(
            "pilot is a first-case preflight only. For the strict three-case, "
            "one-repeat pilot use: --suite full --repeats 1"
        ),
    )
    parser.add_argument(
        "--suite",
        choices=("pilot", "full"),
        default="pilot",
        help=(
            "pilot runs only the first case as a preflight; full selects all "
            "three frozen cases"
        ),
    )
    parser.add_argument("--case-id")
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-total-tokens", type=int)
    parser.add_argument("--per-case-token-budget", type=int, default=60_000)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        help="temporarily override the provider request timeout for this process",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        help="temporarily override the provider retry attempt limit for this process",
    )
    parser.add_argument(
        "--circuit-failure-threshold",
        type=int,
        help="temporarily override failed documents before circuit-open for this process",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repeats = args.repeats if args.repeats is not None else (1 if args.suite == "pilot" else 3)
    max_total_tokens = args.max_total_tokens if args.max_total_tokens is not None else (80_000 if args.suite == "pilot" else 600_000)
    manifest = load_manifest()
    cases = select_cases(manifest, args.suite, args.case_id)
    report = run_acceptance(
        manifest,
        cases,
        repeats=repeats,
        max_total_tokens=max_total_tokens,
        per_case_token_budget=args.per_case_token_budget,
        timeout_seconds=args.timeout_seconds,
        max_attempts=args.max_attempts,
        circuit_failure_threshold=args.circuit_failure_threshold,
        progress=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True),
    )
    output = args.output or Path(
        "artifacts/phase1-model-acceptance-"
        + args.suite
        + "-"
        + datetime.now().strftime("%Y%m%d-%H%M%S")
        + ".json"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "suite": args.suite,
                "passed": report["passed"],
                "stop_reason": report["stop_reason"],
                **report["coverage"],
                **report["aggregate"],
                **report["provider_completions"],
                **report["tokens"],
                **report["timing"],
                "report": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["passed"]:
        return 0
    if report["coverage"]["full_model_attempt_count"] < report["coverage"]["requested_attempt_count"]:
        return 2
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
