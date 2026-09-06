from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import httpx

from app.domain import ConsistencyIssue, EvidenceSpan, IssueCategory, ParsedDirective, Severity
from app.model_extractor import ModelEnhancedExtractor
from app.pipeline import AnalysisPipeline, DocumentInput, PipelineResult
from app.provider import OpenAICompatibleProvider, RetryPolicy
from scripts.run_phase1_model_acceptance import (
    CountingProvider,
    DEFAULT_SUITE_ROOT,
    FORBIDDEN_ASSERTIONS,
    _directive_fingerprint,
    _execution_classification,
    _issue_fingerprint,
    _safe_model_execution,
    build_parser,
    load_manifest,
    run_acceptance,
    score_case,
    select_cases,
)
from tests.test_model_extractor import completion, settings


def _source_line(relative_path: str, line: int) -> str:
    return (DEFAULT_SUITE_ROOT / relative_path).read_text(encoding="utf-8").splitlines()[line - 1]


def _evidence(case_id: str, source: dict) -> EvidenceSpan:
    relative_path = source["path"]
    return EvidenceSpan(
        document_id=f"{case_id}:{relative_path}",
        document_name=Path(relative_path).name,
        line_start=source["line"],
        line_end=source["line"],
        text=_source_line(relative_path, source["line"]),
    )


def _directive(case: dict, semantic: dict) -> ParsedDirective:
    semantic_class = semantic["class"]
    common = {"document_role": "chapter", "modality": "asserted", "source_scope": "narrator", "certainty": "certain"}
    if semantic_class == "confirmed_canonical":
        kind = "world_rule"
        attrs = {**common, "key": semantic["summary"], "value": "confirmed", "source_scope": "world_rule"}
    elif semantic_class == "confirmed_narrative":
        kind = "fact"
        attrs = {**common, "subject": semantic["summary"], "predicate": "状态", "value": "confirmed"}
    elif semantic_class == "open_question":
        kind = "open_question"
        attrs = {**common, "question": semantic["summary"], "question_type": "open", "modality": "interrogative", "certainty": "unknown"}
    elif semantic_class == "tentative":
        kind = "tentative_fact"
        attrs = {**common, "subject": semantic["summary"], "predicate": "状态", "value": "possible", "original_kind": "fact", "modality": "hypothetical", "certainty": "possible"}
    elif semantic_class == "quoted_claim":
        kind = "character_claim"
        attrs = {**common, "subject": semantic["summary"], "predicate": "声称", "value": "reported", "original_kind": "fact", "modality": "reported", "source_scope": "character_dialogue", "certainty": "unknown"}
    else:
        kind = "clarification"
        linked = next(
            finding for finding in case["expected_findings"]
            if finding["result"] == "clarification" and semantic["id"] in finding["evidence_ids"]
        )
        attrs = {**common, "summary": semantic["summary"], "category": linked["category"], "modality": "uncertain", "certainty": "unknown"}
    return ParsedDirective(kind=kind, attrs=attrs, evidence=_evidence(case["id"], semantic["evidence"]))


def perfect_result(case: dict) -> PipelineResult:
    semantics = {row["id"]: row for row in case["required_semantics"]}
    directives = [_directive(case, semantic) for semantic in case["required_semantics"]]
    for finding in case["expected_findings"]:
        if finding["result"] != "clarification":
            continue
        if any(row.kind == "clarification" and row.attrs.get("category") == finding["category"] for row in directives):
            continue
        semantic = semantics[finding["evidence_ids"][-1]]
        directives.append(
            ParsedDirective(
                kind="clarification",
                attrs={
                    "summary": semantic["summary"],
                    "category": finding["category"],
                    "modality": "uncertain",
                    "source_scope": "narrator",
                    "certainty": "unknown",
                },
                evidence=_evidence(case["id"], semantic["evidence"]),
            )
        )
    issues = []
    for finding in case["expected_findings"]:
        if finding["result"] != "confirmed_conflict":
            continue
        issues.append(
            ConsistencyIssue(
                category=IssueCategory(finding["category"]),
                severity=Severity.high,
                confidence=1.0,
                title=finding["id"],
                explanation="fixed fake result",
                suggestion="review",
                evidence=[_evidence(case["id"], semantics[row]["evidence"]) for row in finding["evidence_ids"]],
            )
        )
    chunks = len(case["documents"])
    id_to_path = {
        f"{case['id']}:{row['path']}": row["path"] for row in case["documents"]
    }
    clarification_evidence = {
        finding["category"]: [
            [semantics[semantic_id]["evidence"]["path"], semantics[semantic_id]["evidence"]["line"]]
            for semantic_id in finding["evidence_ids"]
        ]
        for finding in case["expected_findings"]
        if finding["result"] == "clarification"
    }
    provenance = {
        "schema_version": 1,
        "directives": [
            {
                "fingerprint": _directive_fingerprint(
                    directive,
                    id_to_path[directive.evidence.document_id],
                ),
                "sources": ["model"],
                "contributing_evidence": (
                    clarification_evidence.get(directive.attrs.get("category"), [])
                    if directive.kind == "clarification"
                    else []
                ),
            }
            for directive in directives
        ],
        "issues": [
            {
                "fingerprint": _issue_fingerprint(issue, id_to_path),
                "derivation": ["deterministic"],
                "evidence_sources": ["model"],
            }
            for issue in issues
        ],
    }
    return PipelineResult(
        directives=directives,
        issues=issues,
        prompt_tokens=10,
        completion_tokens=5,
        model_used=True,
        diagnostics={
            "model": {
                "enabled": True,
                "configured": True,
                "total_chunks": chunks,
                "attempted_chunks": chunks,
                "succeeded_chunks": chunks,
                "failed_chunks": 0,
                "skipped_chunks": 0,
                "invalid_records": 0,
                "empty_response_chunks": 0,
                "reason_codes": [],
            },
            "provenance": provenance,
        },
    )


class FakeProvider:
    def __init__(self, *, failed: bool = False) -> None:
        self.settings = SimpleNamespace(
            per_run_token_budget=20_000,
            openai_model="fake-model",
            provider_timeout_seconds=11.0,
            provider_max_attempts=4,
            model_circuit_breaker_failed_documents=2,
            provider_thinking_mode=None,
        )
        self.retry_policy = RetryPolicy(max_attempts=4, base_delay_seconds=0.1)
        self.requested = 1
        self.succeeded = 0 if failed else 1
        self.failed = 1 if failed else 0
        self.last_error_type = "ProviderError" if failed else None


class PerfectPipeline:
    def __init__(self, manifest: dict) -> None:
        self.manifest = manifest
        self.extractor = SimpleNamespace(provider=FakeProvider(), _run_tokens_used=15)

    def run(self, documents):
        case_id = documents[0].id.split(":", 1)[0]
        case = next(row for row in self.manifest["cases"] if row["id"] == case_id)
        return perfect_result(case)


class FailedPipeline:
    def __init__(self) -> None:
        self.extractor = SimpleNamespace(provider=FakeProvider(failed=True), _run_tokens_used=0)

    def run(self, documents):
        return PipelineResult(
            diagnostics={
                "model": {
                    "enabled": True,
                    "configured": True,
                    "total_chunks": 1,
                    "attempted_chunks": 1,
                    "succeeded_chunks": 0,
                    "failed_chunks": 1,
                    "skipped_chunks": 0,
                    "invalid_records": 0,
                    "empty_response_chunks": 0,
                    "reason_codes": ["provider_error"],
                }
            }
        )


class Phase1ModelAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest()

    def test_execution_gate_requires_every_chunk_and_zero_invalid(self):
        good = {
            "enabled": True, "configured": True, "total_chunks": 2,
            "attempted_chunks": 2, "succeeded_chunks": 2,
            "failed_chunks": 0, "skipped_chunks": 0, "invalid_records": 0,
            "empty_response_chunks": 0,
            "reason_codes": [],
        }
        self.assertEqual((True, True), _execution_classification(good))
        empty = {**good, "empty_response_chunks": 1}
        self.assertEqual((True, False), _execution_classification(empty))
        impossible_empty = {**good, "empty_response_chunks": 3}
        self.assertEqual((False, False), _execution_classification(impossible_empty))
        legacy = {key: value for key, value in good.items() if key != "empty_response_chunks"}
        self.assertEqual((False, False), _execution_classification(legacy))
        for field in ("failed_chunks", "skipped_chunks", "invalid_records"):
            row = {**good, field: 1}
            if field == "failed_chunks":
                row["succeeded_chunks"] = 1
            if field == "skipped_chunks":
                row["attempted_chunks"] = row["succeeded_chunks"] = 1
            self.assertFalse(_execution_classification(row)[1], field)

    def test_execution_gate_uses_final_invalid_disposition_with_legacy_fallback(self):
        legacy = {
            "enabled": True, "configured": True, "total_chunks": 1,
            "attempted_chunks": 1, "succeeded_chunks": 1,
            "failed_chunks": 0, "skipped_chunks": 0,
            "invalid_records": 1, "empty_response_chunks": 0,
            "reason_codes": ["schema_validation"],
        }
        self.assertEqual((True, False), _execution_classification(legacy))

        repaired = {
            **legacy,
            "invalid_records": 4,
            "unresolved_invalid_records": 0,
            "recovered_invalid_records": 4,
            "repair_attempted": True,
            "repair_succeeded": True,
            "repair_failed": False,
            "repair_post_invalid": 0,
            "repair_final_path": "repaired",
            "reason_codes": [
                "semantic_labels_quarantined", "schema_validation", "repair_succeeded"
            ],
        }
        self.assertEqual((True, True), _execution_classification(repaired))
        safe = _safe_model_execution(repaired)
        self.assertEqual(4, safe["invalid_records"])
        self.assertEqual(0, safe["unresolved_invalid_records"])
        self.assertEqual(4, safe["recovered_invalid_records"])
        self.assertEqual("repaired", safe["repair_final_path"])

        for updates in (
            {"unresolved_invalid_records": 1, "recovered_invalid_records": 3},
            {"repair_failed": True, "repair_post_invalid": 1,
             "unresolved_invalid_records": 1, "recovered_invalid_records": 3},
            {"repair_post_invalid": 1},
            {"recovered_invalid_records": 3},
        ):
            with self.subTest(updates=updates):
                self.assertFalse(_execution_classification({**repaired, **updates})[1])

    def test_perfect_manifest_projection_passes_all_frozen_gates(self):
        for case in self.manifest["cases"]:
            result = perfect_result(case)
            id_to_path = {
                f"{case['id']}:{row['path']}": row["path"] for row in case["documents"]
            }
            score = score_case(
                case,
                result.directives,
                result.issues,
                id_to_path,
                result.diagnostics["provenance"],
            )
            self.assertTrue(score["gate_pass"], case["id"])
            self.assertEqual(score["semantics"]["expected"], score["semantics"]["matched"])
            self.assertEqual(0, score["nonconfirmed_pollution"]["polluted"])
            forbidden = score["forbidden_findings"]
            self.assertEqual(forbidden["expected"], forbidden["evaluated"])
            self.assertEqual(0, forbidden["unevaluated"])
            self.assertTrue(all(row["status"] == "passed" for row in forbidden["results"]))

    def test_every_frozen_forbidden_finding_has_an_executable_failure_probe(self):
        probed = 0
        for case in self.manifest["cases"]:
            semantics = {row["id"]: row for row in case["required_semantics"]}
            id_to_path = {
                f"{case['id']}:{row['path']}": row["path"] for row in case["documents"]
            }
            for finding in case["forbidden_findings"]:
                with self.subTest(case=case["id"], finding=finding):
                    result = perfect_result(case)
                    rule = FORBIDDEN_ASSERTIONS[finding]
                    if rule.get("pollution_ids"):
                        semantic = semantics[rule["pollution_ids"][0]]
                        result.directives.append(
                            ParsedDirective(
                                kind="fact",
                                attrs={
                                    "subject": "forbidden-sentinel",
                                    "predicate": "asserted",
                                    "value": "true",
                                    "modality": "asserted",
                                    "source_scope": "narrator",
                                    "certainty": "certain",
                                },
                                evidence=_evidence(case["id"], semantic["evidence"]),
                            )
                        )
                    elif rule.get("unexpected_directive_ids"):
                        semantic = semantics[rule["unexpected_directive_ids"][0]]
                        result.directives.append(
                            ParsedDirective(
                                kind="event",
                                attrs={
                                    "subject": "forbidden-sentinel",
                                    "predicate": "derived",
                                    "value": "occurred",
                                    "modality": "asserted",
                                    "source_scope": "narrator",
                                    "certainty": "certain",
                                },
                                evidence=_evidence(case["id"], semantic["evidence"]),
                            )
                        )
                    elif rule.get("unexpected_issue_all_ids"):
                        result.issues.append(
                            ConsistencyIssue(
                                category=IssueCategory.fact_conflict,
                                severity=Severity.high,
                                confidence=1.0,
                                title="forbidden-sentinel",
                                explanation="fixed failure probe",
                                suggestion="review",
                                evidence=[
                                    _evidence(case["id"], semantics[semantic_id]["evidence"])
                                    for semantic_id in rule["unexpected_issue_all_ids"]
                                ],
                            )
                        )
                    else:
                        self.fail(f"no failure probe for: {finding}")
                    score = score_case(
                        case,
                        result.directives,
                        result.issues,
                        id_to_path,
                        result.diagnostics["provenance"],
                    )
                    row = next(
                        row for row in score["forbidden_findings"]["results"]
                        if row["finding"] == finding
                    )
                    self.assertEqual("hit", row["status"])
                    self.assertTrue(row["violations"])
                    self.assertFalse(score["gate_pass"])
                    probed += 1
        self.assertEqual(12, probed)

    def test_unknown_forbidden_manifest_text_is_unevaluated_and_fails_closed(self):
        case = deepcopy(self.manifest["cases"][0])
        case["forbidden_findings"].append("新增但尚无执行规则的禁止项")
        result = perfect_result(case)
        id_to_path = {
            f"{case['id']}:{row['path']}": row["path"] for row in case["documents"]
        }
        score = score_case(
            case,
            result.directives,
            result.issues,
            id_to_path,
            result.diagnostics["provenance"],
        )
        self.assertEqual(1, score["forbidden_findings"]["unevaluated"])
        self.assertEqual("unevaluated", score["forbidden_findings"]["results"][-1]["status"])
        self.assertFalse(score["gate_pass"])

    def test_forbidden_conflict_is_reported_even_if_it_also_matches_an_expected_issue(self):
        case = self.manifest["cases"][0]
        semantics = {row["id"]: row for row in case["required_semantics"]}
        result = perfect_result(case)
        location_issue = next(row for row in result.issues if row.title == "g-issue-location")
        location_issue.evidence.append(
            _evidence(case["id"], semantics["g-b-location"]["evidence"])
        )
        id_to_path = {
            f"{case['id']}:{row['path']}": row["path"] for row in case["documents"]
        }
        score = score_case(
            case,
            result.directives,
            result.issues,
            id_to_path,
            result.diagnostics["provenance"],
        )
        forbidden = next(
            row for row in score["forbidden_findings"]["results"]
            if row["finding"] == "不得把A路线观星厅与B路线货舱报成同刻多地点"
        )
        self.assertEqual("hit", forbidden["status"])
        self.assertTrue(any(row["type"] == "forbidden_conflict" for row in forbidden["violations"]))
        self.assertFalse(score["gate_pass"])

    def test_clarification_requires_every_manifest_evidence_reference(self):
        for case in self.manifest["cases"]:
            finding = next(
                row for row in case["expected_findings"] if row["result"] == "clarification"
            )
            result = perfect_result(case)
            provenance = deepcopy(result.diagnostics["provenance"])
            clarification = next(
                row for row in result.directives
                if row.kind == "clarification"
                and row.attrs.get("category") == finding["category"]
            )
            id_to_path = {
                f"{case['id']}:{row['path']}": row["path"] for row in case["documents"]
            }
            fingerprint = _directive_fingerprint(
                clarification, id_to_path[clarification.evidence.document_id]
            )
            provenance_row = next(
                row for row in provenance["directives"] if row["fingerprint"] == fingerprint
            )
            primary = (
                id_to_path[clarification.evidence.document_id],
                clarification.evidence.line_start,
            )
            removable = next(
                row for row in provenance_row["contributing_evidence"]
                if tuple(row) != primary
            )
            provenance_row["contributing_evidence"].remove(removable)
            score = score_case(
                case, result.directives, result.issues, id_to_path, provenance
            )
            self.assertEqual(0, score["clarifications"]["matched_with_complete_evidence"])
            self.assertFalse(score["gate_pass"])

    def test_baseline_only_results_cannot_masquerade_as_ai_first(self):
        pipeline = PerfectPipeline(self.manifest)
        original_run = pipeline.run

        def baseline_only(documents):
            result = original_run(documents)
            for row in result.diagnostics["provenance"]["directives"]:
                row["sources"] = ["baseline"]
            for row in result.diagnostics["provenance"]["issues"]:
                row["evidence_sources"] = ["baseline"]
            return result

        pipeline.run = baseline_only
        report = run_acceptance(
            self.manifest,
            select_cases(self.manifest, "pilot"),
            pipeline_factory=lambda: pipeline,
        )
        self.assertEqual(1, report["coverage"]["full_model_attempt_count"])
        self.assertEqual(0, report["coverage"]["provenance_verified_attempt_count"])
        self.assertFalse(report["attempts"][0]["provenance_verified"])
        self.assertFalse(report["attempts"][0]["included_in_acceptance_metrics"])
        self.assertFalse(report["passed"])

    def test_model_and_baseline_deduped_sources_preserve_model_credit(self):
        case = self.manifest["cases"][0]
        result = perfect_result(case)
        provenance = deepcopy(result.diagnostics["provenance"])
        for row in provenance["directives"]:
            row["sources"] = ["baseline", "model"]
        for row in provenance["issues"]:
            row["evidence_sources"] = ["baseline", "model"]
        id_to_path = {
            f"{case['id']}:{row['path']}": row["path"] for row in case["documents"]
        }
        score = score_case(
            case, result.directives, result.issues, id_to_path, provenance
        )
        self.assertTrue(score["gate_pass"])
        self.assertTrue(
            all(match["model_source"] for match in score["semantics"]["matches"])
        )

    def test_legacy_result_without_record_provenance_fails_closed(self):
        case = self.manifest["cases"][0]
        result = perfect_result(case)
        id_to_path = {
            f"{case['id']}:{row['path']}": row["path"] for row in case["documents"]
        }
        score = score_case(case, result.directives, result.issues, id_to_path)
        self.assertFalse(score["provenance"]["available"])
        self.assertEqual(0, score["semantics"]["model_provenance_matched"])
        self.assertEqual(0, score["conflicts"]["model_evidence_matched"])
        self.assertFalse(score["gate_pass"])

    def test_conflicts_require_deterministic_derivation_over_model_evidence(self):
        case = self.manifest["cases"][0]
        result = perfect_result(case)
        provenance = deepcopy(result.diagnostics["provenance"])
        provenance["issues"][0]["derivation"] = ["model"]
        id_to_path = {
            f"{case['id']}:{row['path']}": row["path"] for row in case["documents"]
        }
        score = score_case(
            case, result.directives, result.issues, id_to_path, provenance
        )
        self.assertEqual(
            score["conflicts"]["expected"] - 1,
            score["conflicts"]["deterministic_derived_matched"],
        )
        self.assertFalse(score["gate_pass"])

    def test_eligible_record_on_question_line_is_pollution(self):
        case = self.manifest["cases"][0]
        result = perfect_result(case)
        question = next(row for row in case["required_semantics"] if row["id"] == "g-cart-question")
        result.directives.append(
            ParsedDirective(
                kind="fact",
                attrs={
                    "subject": "推车食物", "predicate": "安全", "value": "是",
                    "modality": "asserted", "source_scope": "narrator", "certainty": "certain",
                },
                evidence=_evidence(case["id"], question["evidence"]),
            )
        )
        id_to_path = {f"{case['id']}:{row['path']}": row["path"] for row in case["documents"]}
        score = score_case(case, result.directives, result.issues, id_to_path)
        self.assertEqual(1, score["nonconfirmed_pollution"]["polluted"])
        self.assertFalse(score["gate_pass"])

    def test_full_fake_run_is_stable_and_report_contains_no_raw_transport_data(self):
        cases = select_cases(self.manifest, "full")
        report = run_acceptance(
            self.manifest,
            cases,
            repeats=3,
            max_total_tokens=10_000,
            pipeline_factory=lambda: PerfectPipeline(self.manifest),
        )
        self.assertTrue(report["passed"])
        self.assertEqual(9, report["coverage"]["full_model_attempt_count"])
        self.assertEqual(
            {
                "cases": 3,
                "required_semantics": 40,
                "confirmed_conflicts": 5,
                "clarifications": 3,
                "nonconfirmed_semantics": 11,
                "forbidden_findings": 12,
            },
            report["contract"]["frozen_suite"],
        )
        self.assertTrue(report["stability"]["all_cases_stable"])
        self.assertFalse(report["benchmark"]["human_annotated"])
        self.assertTrue(report["benchmark"]["developer_visible"])
        self.assertFalse(report["benchmark"]["blind_test"])
        self.assertFalse(report["benchmark"]["prompts_or_raw_response_bodies_recorded"])
        self.assertFalse(report["benchmark"]["credentials_or_endpoint_recorded"])
        self.assertEqual("available", report["provider_completions"]["availability"])
        self.assertEqual(9, report["provider_completions"]["requested"])
        self.assertIsNone(report["provider_completions"]["per_call_telemetry"])
        self.assertEqual(
            "unavailable",
            report["provider_completions"]["per_call_telemetry_status"],
        )
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("fake-model", serialized)
        self.assertNotIn("endpoint", serialized.lower().replace(
            "credentials_or_endpoint_recorded", ""
        ))

    def test_effective_overrides_are_reported_and_restored_in_process(self):
        pipeline = PerfectPipeline(self.manifest)
        original_retry_policy = pipeline.extractor.provider.retry_policy
        report = run_acceptance(
            self.manifest,
            select_cases(self.manifest, "pilot"),
            max_total_tokens=100_000,
            per_case_token_budget=12_345,
            timeout_seconds=7.5,
            max_attempts=6,
            circuit_failure_threshold=3,
            pipeline_factory=lambda: pipeline,
        )
        configuration = report["attempts"][0]["effective_model_configuration"]
        self.assertEqual(
            hashlib.sha256(b"fake-model").hexdigest(),
            configuration["model_identifier_sha256"],
        )
        self.assertEqual("sha256", configuration["model_identifier_kind"])
        self.assertEqual(7.5, configuration["timeout_seconds"])
        self.assertEqual(6, configuration["max_attempts"])
        self.assertEqual(3, configuration["circuit_failure_threshold"])
        self.assertEqual(12_345, configuration["per_run_token_budget"])
        self.assertIsNone(configuration["completion_token_limit"])
        self.assertIsNone(configuration["generation_limit"])
        self.assertEqual(
            {"configured": False, "mode": None}, configuration["thinking"]
        )
        settings = pipeline.extractor.provider.settings
        self.assertEqual(11.0, settings.provider_timeout_seconds)
        self.assertEqual(4, settings.provider_max_attempts)
        self.assertEqual(2, settings.model_circuit_breaker_failed_documents)
        self.assertEqual(20_000, settings.per_run_token_budget)
        self.assertIs(original_retry_policy, pipeline.extractor.provider.retry_policy)

    def test_acceptance_report_records_only_safe_thinking_mode(self):
        pipeline = PerfectPipeline(self.manifest)
        settings = pipeline.extractor.provider.settings
        settings.provider_thinking_mode = "disabled"
        settings.openai_api_key = "acceptance-private-key"
        settings.openai_base_url = "https://acceptance-private.invalid/v1"

        report = run_acceptance(
            self.manifest,
            select_cases(self.manifest, "pilot"),
            pipeline_factory=lambda: pipeline,
        )

        configuration = report["attempts"][0]["effective_model_configuration"]
        self.assertEqual(
            {"configured": True, "mode": "disabled"},
            configuration["thinking"],
        )
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("acceptance-private-key", serialized)
        self.assertNotIn("acceptance-private.invalid", serialized)

    def test_pilot_help_distinguishes_preflight_from_three_case_strict_pilot(self):
        parser = build_parser()
        help_text = parser.format_help()
        self.assertIn("first-case preflight", help_text)
        self.assertIn("--suite full --repeats 1", help_text)
        self.assertEqual(1, len(select_cases(self.manifest, "pilot")))
        self.assertEqual(3, len(select_cases(self.manifest, "full")))
        args = parser.parse_args([
            "--timeout-seconds", "8.5",
            "--max-attempts", "5",
            "--circuit-failure-threshold", "2",
        ])
        self.assertEqual((8.5, 5, 2), (
            args.timeout_seconds,
            args.max_attempts,
            args.circuit_failure_threshold,
        ))

    def test_model_execution_diagnostics_are_allowlisted(self):
        safe = _safe_model_execution({
            "enabled": True,
            "configured": True,
            "total_chunks": 1,
            "endpoint": "https://should-not-be-recorded.invalid",
            "api_key": "should-not-be-recorded",
            "documents": [{
                "document_id": "case:chapter.md",
                "failed_chunks": 0,
                "api_key": "nested-secret",
            }],
        })
        serialized = json.dumps(safe)
        self.assertNotIn("endpoint", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("nested-secret", serialized)

    def test_missing_provider_counters_are_unavailable_not_fake_zeroes(self):
        pipeline = PerfectPipeline(self.manifest)
        provider = pipeline.extractor.provider
        for field in ("requested", "succeeded", "failed"):
            delattr(provider, field)
        report = run_acceptance(
            self.manifest,
            select_cases(self.manifest, "pilot"),
            pipeline_factory=lambda: pipeline,
        )
        telemetry = report["provider_completions"]
        self.assertEqual("unavailable", telemetry["availability"])
        self.assertIsNone(telemetry["requested"])
        self.assertIsNone(telemetry["succeeded"])
        self.assertIsNone(telemetry["failed"])
        self.assertEqual("unavailable", telemetry["per_call_telemetry_status"])
        self.assertIsNone(telemetry["per_call_telemetry"])

    def test_acceptance_report_consumes_safe_per_call_telemetry(self):
        pipeline = PerfectPipeline(self.manifest)
        original_run = pipeline.run

        def run_with_telemetry(documents):
            result = original_run(documents)
            result.diagnostics["model"]["provider_calls"] = [{
                "status": "success",
                "category": "success",
                "attempt": 2,
                "elapsed_ms": 17,
                "input_chars": 120,
                "response_chars": 45,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "http_status": 200,
                "url": "https://must-not-leak.invalid",
                "prompt": "must-not-leak",
            }]
            return result

        pipeline.run = run_with_telemetry
        report = run_acceptance(
            self.manifest,
            select_cases(self.manifest, "pilot"),
            pipeline_factory=lambda: pipeline,
        )
        completions = report["provider_completions"]
        self.assertEqual("available", completions["per_call_telemetry_status"])
        self.assertEqual(17, completions["per_call_telemetry"][0]["elapsed_ms"])
        serialized = json.dumps(report)
        self.assertNotIn("must-not-leak.invalid", serialized)
        self.assertNotIn("must-not-leak", serialized)
        self.assertIn("distinct from case_total_duration_ms", completions["per_call_telemetry_note"])

    def test_counting_provider_wraps_repair_child_and_keeps_report_content_free(self):
        raw = {
            "kind": "fact",
            "subject": "林澈",
            "predicate": "身份",
            "value": "领航员",
            "source_line_start": 1,
            "source_line_end": 1,
        }
        responses = iter([
            {"records": [raw]},
            {"patches": [{
                "record_index": 1,
                "modality": "asserted",
                "source_scope": "narrator",
                "certainty": "certain",
            }]},
        ])
        request_payloads = []

        def handler(request):
            request_payloads.append(json.loads(request.content))
            return completion(json.dumps(next(responses), ensure_ascii=False))

        delegate = OpenAICompatibleProvider(
            settings(),
            transport=httpx.MockTransport(handler),
            retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0),
            sleep=lambda _: None,
        )
        provider = CountingProvider(delegate)
        result = AnalysisPipeline(
            extractor=ModelEnhancedExtractor(provider)
        ).run([DocumentInput("doc", "chapter.md", "林澈的身份是领航员。")])

        self.assertEqual(2, len(request_payloads))
        self.assertEqual((2, 2, 0), (
            provider.requested, provider.succeeded, provider.failed
        ))
        safe = _safe_model_execution(result.diagnostics["model"])
        self.assertEqual(
            ["extract", "repair"],
            [row["purpose"] for row in safe["provider_calls"]],
        )
        serialized = json.dumps(safe, ensure_ascii=False)
        for forbidden in (
            "林澈的身份是领航员",
            "unit-test-placeholder",
            "mock.invalid",
            '"records"',
            '"patches"',
        ):
            self.assertNotIn(forbidden, serialized)

        unsafe = _safe_model_execution({
            "provider_calls": [{"purpose": "attacker-controlled", "status": "success"}]
        })
        self.assertNotIn("purpose", unsafe["provider_calls"][0])

    def test_systemic_provider_failure_halts_after_one_attempt(self):
        report = run_acceptance(
            self.manifest,
            select_cases(self.manifest, "full"),
            repeats=3,
            pipeline_factory=FailedPipeline,
        )
        self.assertEqual("systemic_provider_failure", report["stop_reason"])
        self.assertEqual(1, report["coverage"]["attempted_count"])
        self.assertEqual(0, report["coverage"]["full_model_attempt_count"])
        self.assertFalse(report["passed"])


if __name__ == "__main__":
    unittest.main()
