import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.domain import IssueCategory
from app.natural_evaluation import load_cases
from app.pipeline import AnalysisPipeline, BaselineExtractor
from app.natural_evaluation import score_case
from scripts.run_complex_model_evaluation import (
    _exact_metrics,
    _strict_errors,
    build_report,
    rescore_report,
    run_evaluation,
    select_cases,
)


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "evaluation-complex-v3"


class FakeModelPipeline:
    def __init__(self, tokens: int = 10, model_used: bool = True) -> None:
        self.pipeline = AnalysisPipeline(extractor=BaselineExtractor())
        self.tokens = tokens
        self.model_used = model_used

    def run(self, documents):
        result = self.pipeline.run(documents)
        result.model_used = self.model_used
        result.prompt_tokens = self.tokens
        result.diagnostics["model"] = {
            "enabled": True, "configured": True, "total_chunks": 1,
            "attempted_chunks": 1, "succeeded_chunks": int(self.model_used),
            "failed_chunks": int(not self.model_used), "skipped_chunks": 0,
            "invalid_records": 0, "reason_codes": [] if self.model_used else ["provider_error"],
        }
        return result


class FakeSystemicFailurePipeline(FakeModelPipeline):
    def __init__(self) -> None:
        super().__init__(model_used=False)
        self.extractor = SimpleNamespace(
            provider=SimpleNamespace(
                requested=4,
                succeeded=0,
                failed=4,
                last_error_type="ProviderError",
            )
        )


class FakePartialFallbackPipeline(FakeModelPipeline):
    def run(self, documents):
        result = super().run(documents)
        result.warnings.append("模型分块 x 抽取不可用，已由全文基线覆盖（ProviderError）")
        result.diagnostics["model"].update(
            total_chunks=2, attempted_chunks=2, failed_chunks=1,
            reason_codes=["provider_error"],
        )
        return result


class ComplexModelEvaluationTests(unittest.TestCase):
    def test_structured_partial_status_overrides_model_used_without_warning(self):
        selected = select_cases(load_cases("test", DATASET_ROOT), "pilot")[:1]
        for status in (
            {"invalid_records": 1, "reason_codes": ["schema_validation"]},
            {"total_chunks": 2, "skipped_chunks": 1, "reason_codes": ["token_budget"]},
            {"total_chunks": 2, "skipped_chunks": 1, "reason_codes": ["chunk_limit"]},
        ):
            with self.subTest(status=status):
                class PartialPipeline(FakeModelPipeline):
                    def run(self, documents):
                        result = super().run(documents)
                        result.warnings.clear()
                        result.diagnostics["model"].update(status)
                        return result

                report = run_evaluation(selected, 100, pipeline_factory=PartialPipeline)
                self.assertEqual(1, report["coverage"]["model_scored_case_count"])
                self.assertEqual(0, report["coverage"]["strict_model_scored_case_count"])
                self.assertEqual("structured_execution", report["attempts"][0]["classification_source"])

    def test_unknown_new_execution_never_counts_as_full_model(self):
        class UnknownPipeline(FakeModelPipeline):
            def run(self, documents):
                result = super().run(documents)
                result.diagnostics.pop("model")
                result.warnings.clear()
                return result

        selected = select_cases(load_cases("test", DATASET_ROOT), "pilot")[:1]
        report = run_evaluation(selected, 100, pipeline_factory=UnknownPipeline)
        self.assertEqual(0, report["coverage"]["strict_model_scored_case_count"])
        self.assertEqual("unknown_execution", report["attempts"][0]["classification_source"])

    def test_structured_complete_ignores_warning_wording_and_zero_call_budget_stops(self):
        selected = select_cases(load_cases("test", DATASET_ROOT), "pilot")

        class CompletePipeline(FakeModelPipeline):
            def run(self, documents):
                result = super().run(documents)
                result.warnings = ["旧文案：模型单次运行 Token 预算不足，抽取不可用"]
                return result

        complete = run_evaluation(selected[:1], 100, pipeline_factory=CompletePipeline)
        self.assertEqual(1, complete["coverage"]["strict_model_scored_case_count"])

        class BudgetPipeline(FakeModelPipeline):
            def run(self, documents):
                result = super().run(documents)
                result.model_used = False
                result.warnings.clear()
                result.diagnostics["model"].update(
                    attempted_chunks=0, succeeded_chunks=0, skipped_chunks=1,
                    reason_codes=["token_budget"],
                )
                return result

        exhausted = run_evaluation(selected, 100, pipeline_factory=BudgetPipeline)
        self.assertEqual(1, exhausted["coverage"]["attempted_case_count"])
        self.assertEqual("token_budget_exhausted_before_model_success", exhausted["stop_reason"])
        self.assertEqual(0, exhausted["coverage"]["strict_model_scored_case_count"])

    def test_rescore_structured_status_overrides_stale_flags_but_preserves_explicit_legacy(self):
        selected = select_cases(load_cases("test", DATASET_ROOT), "pilot")[:1]
        report = run_evaluation(selected, 100, repeats=3, pipeline_factory=FakeModelPipeline)
        report["benchmark"] = {}
        report["attempts"][0]["model_execution"]["invalid_records"] = 1
        legacy = report["attempts"][1]
        legacy.pop("model_execution")
        unknown = report["attempts"][2]
        unknown.pop("model_execution")
        unknown.pop("included_in_strict_metrics")
        unknown.pop("included_in_model_metrics")
        rescored = rescore_report(report)
        self.assertEqual(1, rescored["coverage"]["strict_model_scored_case_count"])
        self.assertEqual(0, rescored["repeat_summaries"][0]["full_model_case_count"])
        self.assertEqual("legacy_stored_flags_unverified", legacy["classification_source"])
        self.assertTrue(legacy["included_in_strict_metrics"])
        self.assertFalse(unknown["included_in_strict_metrics"])

    def test_pilot_has_five_category_positive_and_hard_negative_coverage(self):
        selected = select_cases(load_cases("test", DATASET_ROOT), "pilot")
        expected_categories = {
            issue.category
            for case in selected
            for issue in case.expected_issues
        }
        negative_focus = {
            case.category_focus for case in selected if not case.expected_issues
        }
        self.assertEqual(set(IssueCategory), expected_categories)
        self.assertEqual(set(IssueCategory), negative_focus)
        self.assertEqual(6, len(selected))

    def test_budget_stops_before_unbounded_case_count(self):
        selected = select_cases(load_cases("test", DATASET_ROOT), "pilot")
        report = run_evaluation(
            selected,
            max_total_tokens=30,
            pipeline_factory=lambda: FakeModelPipeline(tokens=10),
        )
        self.assertEqual(3, report["coverage"]["attempted_case_count"])
        self.assertEqual(3, report["coverage"]["model_scored_case_count"])
        self.assertTrue(report["budget"]["exhausted"])
        self.assertEqual(0, report["budget"]["overshoot_tokens"])

    def test_baseline_only_fallback_is_not_counted_as_model_metric(self):
        selected = select_cases(load_cases("test", DATASET_ROOT), "pilot")[:1]
        report = run_evaluation(
            selected,
            max_total_tokens=100,
            pipeline_factory=lambda: FakeModelPipeline(model_used=False),
        )
        self.assertEqual(1, report["coverage"]["attempted_case_count"])
        self.assertEqual(0, report["coverage"]["model_scored_case_count"])
        self.assertEqual(0, report["metrics"]["overall"]["tp"])

    def test_systemic_provider_failure_stops_after_first_case(self):
        selected = select_cases(load_cases("test", DATASET_ROOT), "pilot")
        report = run_evaluation(
            selected,
            max_total_tokens=100,
            pipeline_factory=FakeSystemicFailurePipeline,
        )
        self.assertEqual(1, report["coverage"]["attempted_case_count"])
        self.assertEqual("systemic_provider_failure", report["stop_reason"])
        self.assertEqual(
            "ProviderError",
            report["attempts"][0]["provider_completions"]["last_error_type"],
        )
        self.assertNotIn(
            "last_safe_error", report["attempts"][0]["provider_completions"]
        )

    def test_full_suite_repeats_have_per_repeat_and_cross_repeat_summaries(self):
        selected = select_cases(load_cases("test", DATASET_ROOT), "full")
        report = run_evaluation(
            selected,
            max_total_tokens=10_000,
            repeats=3,
            pipeline_factory=lambda: FakeModelPipeline(tokens=10),
        )
        self.assertEqual(42, report["coverage"]["requested_attempt_count"])
        self.assertEqual(42, report["coverage"]["attempted_case_count"])
        self.assertEqual(42, report["coverage"]["strict_model_scored_case_count"])
        self.assertEqual(420, report["budget"]["provider_reported_tokens"])
        self.assertEqual(3, len(report["repeat_summaries"]))
        self.assertTrue(
            all(row["completed_all_selected_cases"] for row in report["repeat_summaries"])
        )
        self.assertEqual(14, report["stability"]["case_count_with_all_repeats_eligible"])
        self.assertEqual(1.0, report["stability"]["mean_modal_prediction_set_rate"])

    def test_partial_model_fallback_is_separate_from_strict_metrics(self):
        selected = select_cases(load_cases("test", DATASET_ROOT), "pilot")[:1]
        report = run_evaluation(
            selected,
            max_total_tokens=100,
            pipeline_factory=FakePartialFallbackPipeline,
        )
        self.assertEqual(1, report["coverage"]["model_scored_case_count"])
        self.assertEqual(0, report["coverage"]["strict_model_scored_case_count"])
        self.assertEqual(
            0, report["exact_evidence_metrics"]["full_model_only"]["overall"]["tp"]
        )
        self.assertEqual(1, report["stability"]["case_count_without_any_eligible_repeat"])
        self.assertEqual(0, report["stability"]["cases"][0]["eligible_repeat_count"])

    def test_wrong_evidence_is_not_an_exact_true_positive(self):
        case = select_cases(load_cases("test", DATASET_ROOT), "pilot")[0]
        expected_issue = case.expected_issues[0]
        scored = score_case(
            case,
            [
                {
                    "category": expected_issue.category.value,
                    "title": "right category, wrong evidence",
                    "evidence": [{"document": case.documents[0].name, "line": 1}],
                    "metadata": {},
                }
            ],
        )
        exact = _exact_metrics([scored])["overall"]
        self.assertEqual({"tp": 0, "fp": 1, "fn": 5}, {key: exact[key] for key in ("tp", "fp", "fn")})

    def test_report_metadata_declares_answer_isolation_and_no_raw_bodies(self):
        report = build_report(
            DATASET_ROOT,
            "pilot",
            max_total_tokens=100,
            repeats=1,
            pipeline_factory=lambda: FakeModelPipeline(tokens=10),
        )
        self.assertIn("never included in provider input", report["benchmark"]["answer_isolation"])
        self.assertFalse(report["benchmark"]["prompts_or_response_bodies_recorded"])
        self.assertFalse(report["benchmark"]["credentials_or_endpoint_recorded"])
        self.assertNotIn("safe rejection", json.dumps(report, ensure_ascii=False))

    def test_progress_callback_receives_safe_case_summary(self):
        selected = select_cases(load_cases("test", DATASET_ROOT), "pilot")[:1]
        events = []
        run_evaluation(
            selected,
            max_total_tokens=100,
            pipeline_factory=lambda: FakeModelPipeline(tokens=10),
            progress=events.append,
        )
        self.assertEqual(1, len(events))
        self.assertEqual("case_completed", events[0]["event"])
        self.assertNotIn("warnings", events[0])
        self.assertNotIn("score", events[0])

    def test_strict_errors_keep_repeat_identity_and_rescore_is_offline(self):
        selected = select_cases(load_cases("test", DATASET_ROOT), "pilot")[:1]
        report = run_evaluation(
            selected,
            max_total_tokens=100,
            repeats=2,
            pipeline_factory=lambda: FakeModelPipeline(tokens=10),
        )
        report["attempts"][1]["score"]["false_positives"].append(
            {"category": "fact_conflict", "evidence": []}
        )
        errors = _strict_errors(report["attempts"])
        self.assertEqual(2, errors[0]["repeat_index"])
        wrapped = {"benchmark": {}, **report}
        rescored = rescore_report(wrapped)
        self.assertTrue(rescored["benchmark"]["rescored_without_provider_call"])
        self.assertGreater(rescored["timing"]["case_attempt_p95_ms"], 0)


if __name__ == "__main__":
    unittest.main()
