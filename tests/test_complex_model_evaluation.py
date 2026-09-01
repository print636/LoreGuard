import unittest
from pathlib import Path
from types import SimpleNamespace

from app.domain import IssueCategory
from app.natural_evaluation import load_cases
from app.pipeline import AnalysisPipeline, BaselineExtractor
from scripts.run_complex_model_evaluation import run_evaluation, select_cases


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
                last_safe_error="safe rejection",
            )
        )


class ComplexModelEvaluationTests(unittest.TestCase):
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
            "safe rejection",
            report["attempts"][0]["provider_completions"]["last_safe_error"],
        )


if __name__ == "__main__":
    unittest.main()
