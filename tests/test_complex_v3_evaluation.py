import hashlib
import json
import unittest
from pathlib import Path

from app.domain import IssueCategory
from app.natural_evaluation import load_cases
from scripts.generate_complex_v3 import DATASET_VERSION, WORLD, build, validate
from scripts.run_complex_v3_evaluation import build_report


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "evaluation-complex-v3"


class ComplexV3DatasetTests(unittest.TestCase):
    def test_fixed_dataset_matches_developer_authored_source(self):
        stored = [
            json.loads(line)
            for line in (DATASET_ROOT / "test.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        generated = build()
        validate(generated)
        self.assertEqual(generated, stored)

    def test_manifest_and_hash_make_dataset_identity_explicit(self):
        manifest = json.loads((DATASET_ROOT / "manifest.json").read_text(encoding="utf-8"))
        digest = hashlib.sha256((DATASET_ROOT / "test.jsonl").read_bytes()).hexdigest()
        self.assertEqual(DATASET_VERSION, manifest["version"])
        self.assertEqual(WORLD, manifest["world"])
        self.assertEqual(digest, manifest["sha256"])
        self.assertFalse(manifest["human_annotated"])
        self.assertTrue(manifest["developer_visible"])
        self.assertFalse(manifest["blind_test"])
        self.assertFalse(manifest["model_required"])

    def test_all_five_categories_have_positive_and_hard_negative_coverage(self):
        rows = load_cases("test", DATASET_ROOT)
        self.assertEqual(14, len(rows))
        self.assertEqual(10, sum(len(row.expected_issues) for row in rows))
        self.assertGreaterEqual(min(len(row.documents) for row in rows), 4)
        for category in IssueCategory:
            focused = [row for row in rows if row.category_focus == category]
            self.assertTrue(any(row.expected_issues for row in focused), category.value)
            self.assertTrue(any(not row.expected_issues for row in focused), category.value)

    def test_integrated_case_fixes_all_five_evidence_pairs(self):
        rows = load_cases("test", DATASET_ROOT)
        integrated = next(row for row in rows if row.case_id == "complex-v3-integrated-conflicts")
        self.assertEqual(set(IssueCategory), {issue.category for issue in integrated.expected_issues})
        self.assertEqual(5, len(integrated.documents))
        self.assertEqual(5, len(integrated.expected_issues))
        self.assertTrue(
            all(len(issue.evidence) == 2 for issue in integrated.expected_issues)
        )


class ComplexV3RunnerTests(unittest.TestCase):
    def test_no_model_runner_reports_exact_expected_issues(self):
        report = build_report(DATASET_ROOT)
        benchmark = report["benchmark"]
        metrics = report["metrics"]["overall"]
        self.assertFalse(benchmark["model_enabled"])
        self.assertEqual(0, benchmark["provider_calls"])
        self.assertEqual("developer-authored, developer-visible acceptance regression", benchmark["evaluation_role"])
        self.assertEqual(
            {"tp": 10, "fp": 0, "fn": 0, "evidence_hits": 10},
            {key: metrics[key] for key in ("tp", "fp", "fn", "evidence_hits")},
        )
        self.assertEqual(1.0, metrics["evidence_pair_line_hit_rate"])
        self.assertEqual([], report["errors"])


if __name__ == "__main__":
    unittest.main()
