import json
import tempfile
import unittest
from pathlib import Path

from app.natural_evaluation import (
    NaturalEvaluationCase,
    aggregate_results,
    load_cases,
    run_natural_evaluation,
    score_case,
)
from scripts.generate_natural_evaluation import SEED, generate
from scripts.generate_challenge_v2 import SEED as CHALLENGE_SEED, build as build_challenge


def sample_case(expected=True) -> NaturalEvaluationCase:
    payload = {
        "case_id": "test-sample-01",
        "scenario_id": "test-sample-01",
        "split": "test",
        "category_focus": "fact_conflict",
        "polarity": "positive" if expected else "hard-negative",
        "documents": [
            {"name": "world.md", "content": "甲的身份是领航员。"},
            {"name": "chapter.md", "content": "甲的身份是守卫。"},
        ],
        "expected_issues": (
            [{"category": "fact_conflict", "evidence": [{"document": "world.md", "line": 1}, {"document": "chapter.md", "line": 1}]}]
            if expected
            else []
        ),
        "generation": {"kind": "synthetic natural-language", "seed": SEED, "template": "unit"},
    }
    return NaturalEvaluationCase.model_validate(payload)


class MetricTests(unittest.TestCase):
    def test_empty_expected_case_counts_prediction_as_false_positive(self):
        result = score_case(
            sample_case(expected=False),
            [{"category": "fact_conflict", "title": "误报", "evidence": [{"document": "chapter.md", "line": 1}], "metadata": {}}],
        )
        metrics = aggregate_results([result])["overall"]
        self.assertEqual(0, metrics["tp"])
        self.assertEqual(1, metrics["fp"])
        self.assertEqual(0, metrics["fn"])

    def test_category_match_and_exact_evidence_are_scored_separately(self):
        exact = score_case(
            sample_case(),
            [{"category": "fact_conflict", "title": "命中", "evidence": [{"document": "chapter.md", "line": 1}, {"document": "world.md", "line": 1}], "metadata": {}}],
        )
        wrong_line = score_case(
            sample_case(),
            [{"category": "fact_conflict", "title": "类别命中但证据错误", "evidence": [{"document": "chapter.md", "line": 1}], "metadata": {}}],
        )
        self.assertTrue(exact["matches"][0]["evidence_exact"])
        self.assertFalse(wrong_line["matches"][0]["evidence_exact"])
        metrics = aggregate_results([exact, wrong_line])["overall"]
        self.assertEqual(2, metrics["tp"])
        self.assertEqual(1, metrics["evidence_hits"])
        self.assertEqual(0.5, metrics["evidence_pair_line_hit_rate"])

    def test_multiple_same_category_issues_are_matched_by_evidence(self):
        payload = sample_case().model_dump(mode="json")
        payload["documents"] = [
            {"name": "story.md", "content": "第一处。\n第二处。\n第三处。\n第四处。"}
        ]
        payload["expected_issues"] = [
            {"category": "fact_conflict", "evidence": [{"document": "story.md", "line": 1}, {"document": "story.md", "line": 2}]},
            {"category": "fact_conflict", "evidence": [{"document": "story.md", "line": 3}, {"document": "story.md", "line": 4}]},
        ]
        case = NaturalEvaluationCase.model_validate(payload)
        predictions = [
            {"category": "fact_conflict", "title": "第二组", "evidence": [{"document": "story.md", "line": 3}, {"document": "story.md", "line": 4}], "metadata": {}},
            {"category": "fact_conflict", "title": "第一组", "evidence": [{"document": "story.md", "line": 1}, {"document": "story.md", "line": 2}], "metadata": {}},
        ]
        result = score_case(case, predictions)
        self.assertEqual(2, len(result["matches"]))
        self.assertTrue(all(match["evidence_exact"] for match in result["matches"]))


class DatasetTests(unittest.TestCase):
    def test_dataset_has_positive_and_hard_negative_cases_with_isolated_scenarios(self):
        dev = load_cases("dev")
        test = load_cases("test")
        self.assertGreaterEqual(len(dev) + len(test), 80)
        self.assertFalse({row.scenario_id for row in dev} & {row.scenario_id for row in test})
        for split in (dev, test):
            for category in {row.category_focus for row in split}:
                rows = [row for row in split if row.category_focus == category]
                self.assertTrue(any(row.expected_issues for row in rows))
                self.assertTrue(any(not row.expected_issues for row in rows))

    def test_generator_is_reproducible_with_fixed_seed(self):
        first = generate()
        second = generate()
        self.assertEqual(SEED, 20260901)
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True),
        )

    def test_loading_test_does_not_read_dev_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "dev.jsonl").write_text("not valid json\n", encoding="utf-8")
            (root / "test.jsonl").write_text(sample_case().model_dump_json() + "\n", encoding="utf-8")
            loaded = load_cases("test", root)
            self.assertEqual(["test-sample-01"], [row.case_id for row in loaded])

    def test_current_baseline_report_uses_test_only_and_is_self_contained(self):
        report = run_natural_evaluation("test")
        self.assertFalse(report["benchmark"]["model_enabled"])
        self.assertEqual(60, report["benchmark"]["case_count"])
        self.assertEqual(30, report["benchmark"]["negative_case_count"])
        self.assertEqual([], report["errors"])
        self.assertEqual(
            {"tp": 30, "fp": 0, "fn": 0},
            {key: report["metrics"]["overall"][key] for key in ("tp", "fp", "fn")},
        )

    def test_challenge_v2_is_fixed_developer_visible_synthetic_data(self):
        root = Path(__file__).resolve().parents[1] / "data" / "evaluation-challenge-v2"
        rows = load_cases("test", root)
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(50, len(rows))
        self.assertEqual(CHALLENGE_SEED, manifest["seed"])
        self.assertFalse(manifest["human_annotated"])
        self.assertTrue(manifest["developer_visible"])
        self.assertFalse(manifest["blind_test"])
        self.assertEqual(
            [row.case_id for row in rows],
            [row["case_id"] for row in build_challenge()],
        )


if __name__ == "__main__":
    unittest.main()
