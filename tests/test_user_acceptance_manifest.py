import json
import re
from pathlib import Path
import unittest

import httpx

from app.domain import DocumentRole, IssueCategory
from app.model_extractor import ModelEnhancedExtractor
from app.pipeline import DocumentInput
from app.provider import OpenAICompatibleProvider
from app.rules import detect_issues
from app.semantic_quality import eligible_for_deterministic_rules
from tests.test_model_extractor import completion, settings


SUITE_ROOT = Path(__file__).resolve().parents[1] / "data" / "user-acceptance-v1"
MANIFEST_PATH = SUITE_ROOT / "manifest.json"


STATE_TESTS = {
    "snapshot-after-submit-update": (
        "tests.test_run_reliability",
        "RunReliabilityTests",
        "test_worker_reads_frozen_body_and_diagnostic_lists_exact_version",
    ),
    "retry-reuses-snapshot": (
        "tests.test_run_reliability",
        "RunReliabilityTests",
        "test_retry_copies_original_snapshot_not_current_document",
    ),
    "new-analysis-reads-new-version": (
        "tests.test_api",
        "ApiFlowTests",
        "test_run_creation_and_retry_expose_frozen_document_versions",
    ),
    "duplicate-worker-delivery": (
        "tests.test_run_reliability",
        "RunReliabilityTests",
        "test_conditional_claim_allows_only_one_concurrent_worker",
    ),
    "redispatch-completed-run": (
        "tests.test_run_reliability",
        "RunReliabilityTests",
        "test_completed_run_cannot_be_claimed_or_emit_more_events",
    ),
    "cancel-in-flight-run": (
        "tests.test_run_reliability",
        "RunReliabilityTests",
        "test_cross_session_cancel_stops_at_next_major_stage",
    ),
    "legacy-run-without-snapshot": (
        "tests.test_run_reliability",
        "RunReliabilityTests",
        "test_legacy_queued_run_fails_without_reading_live_documents",
    ),
}


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def evidence_text(relative_path: str, line_number: int) -> str:
    path = (SUITE_ROOT / relative_path).resolve()
    path.relative_to(SUITE_ROOT.resolve())
    lines = path.read_text(encoding="utf-8").splitlines()
    if line_number < 1 or line_number > len(lines):
        raise AssertionError(f"evidence line outside document: {relative_path}:{line_number}")
    return lines[line_number - 1].strip()


class UserAcceptanceManifestTests(unittest.TestCase):
    def test_manifest_references_and_frozen_counts_are_machine_checked(self):
        manifest = load_manifest()
        self.assertEqual("user-acceptance-v1", manifest["suite_id"])
        self.assertEqual("developer-visible-non-blind", manifest["visibility"])
        self.assertTrue(manifest["policy"]["ai_first"])
        self.assertEqual(3, len(manifest["cases"]))

        case_ids: set[str] = set()
        semantic_ids: set[str] = set()
        finding_ids: set[str] = set()
        semantic_count = confirmed_count = clarification_count = 0
        allowed_categories = {category.value for category in IssueCategory}

        for case in manifest["cases"]:
            self.assertNotIn(case["id"], case_ids)
            case_ids.add(case["id"])
            declared_paths = {row["path"] for row in case["documents"]}
            allowed_roles = {role.value for role in DocumentRole}
            for document in case["documents"]:
                relative_path = document["path"]
                self.assertIn(document["role"], allowed_roles)
                self.assertRegex(
                    document.get("scope", "global"),
                    r"^[A-Za-z0-9_\-\u4e00-\u9fff]{1,80}$",
                )
                path = (SUITE_ROOT / relative_path).resolve()
                path.relative_to(SUITE_ROOT.resolve())
                self.assertTrue(path.is_file())
                self.assertTrue(path.read_text(encoding="utf-8").strip())

            case_semantics: set[str] = set()
            for semantic in case["required_semantics"]:
                semantic_count += 1
                self.assertNotIn(semantic["id"], semantic_ids)
                semantic_ids.add(semantic["id"])
                case_semantics.add(semantic["id"])
                self.assertIn(semantic["class"], manifest["semantic_classes"])
                self.assertEqual(
                    manifest["semantic_classes"][semantic["class"]][
                        "eligible_for_confirmed_conflict"
                    ],
                    semantic["enters_conflict_engine"],
                )
                evidence = semantic["evidence"]
                self.assertIn(evidence["path"], declared_paths)
                self.assertTrue(evidence_text(evidence["path"], evidence["line"]))

            for finding in case["expected_findings"]:
                self.assertNotIn(finding["id"], finding_ids)
                finding_ids.add(finding["id"])
                self.assertTrue(set(finding["evidence_ids"]).issubset(case_semantics))
                if finding["result"] == "confirmed_conflict":
                    confirmed_count += 1
                    self.assertIn(finding["category"], allowed_categories)
                else:
                    self.assertEqual("clarification", finding["result"])
                    clarification_count += 1

        self.assertEqual(40, semantic_count)
        self.assertEqual(5, confirmed_count)
        self.assertEqual(3, clarification_count)
        self.assertEqual(7, len(manifest["state_machine_scenarios"]))
        self.assertEqual(
            set(STATE_TESTS),
            {row["id"] for row in manifest["state_machine_scenarios"]},
        )

        # Keep the manifest-to-regression mapping executable and auditable.
        for module_name, class_name, method_name in STATE_TESTS.values():
            module = __import__(module_name, fromlist=[class_name])
            case_class = getattr(module, class_name)
            self.assertTrue(callable(getattr(case_class, method_name)))

    def test_manifest_noncanonical_classes_remain_outside_rules_with_mock_model(self):
        manifest = load_manifest()
        representatives = {}
        for case in manifest["cases"]:
            for semantic in case["required_semantics"]:
                representatives.setdefault(semantic["class"], (case, semantic))

        expected_kinds = {
            "open_question": "open_question",
            "tentative": "tentative_fact",
            "quoted_claim": "character_claim",
            "clarification": "clarification",
        }
        for semantic_class, expected_kind in expected_kinds.items():
            case, semantic = representatives[semantic_class]
            source = semantic["evidence"]
            text = evidence_text(source["path"], source["line"])
            if semantic_class == "open_question":
                record = {
                    "kind": "open_question",
                    "question": text,
                    "question_type": "open",
                    "modality": "interrogative",
                    "source_scope": "narrator",
                    "certainty": "unknown",
                }
            elif semantic_class == "clarification":
                linked = next(
                    finding
                    for finding in case["expected_findings"]
                    if finding["result"] == "clarification"
                    and semantic["id"] in finding["evidence_ids"]
                )
                record = {
                    "kind": "clarification",
                    "summary": text,
                    "category": linked["category"],
                    "modality": "uncertain",
                    "source_scope": "narrator",
                    "certainty": "unknown",
                }
            else:
                subject = "".join(re.findall(r"[\u4e00-\u9fff]", text))[:2]
                self.assertTrue(subject)
                record = {
                    "kind": "fact",
                    "subject": subject,
                    "predicate": "状态",
                    "value": "未确认",
                    "modality": (
                        "reported" if semantic_class == "quoted_claim" else "hypothetical"
                    ),
                    "source_scope": (
                        "character_dialogue"
                        if semantic_class == "quoted_claim"
                        else "narrator"
                    ),
                    "certainty": (
                        "unknown" if semantic_class == "quoted_claim" else "possible"
                    ),
                }
            record.update(source_line_start=1, source_line_end=1)
            payload = json.dumps({"records": [record]}, ensure_ascii=False)
            provider = OpenAICompatibleProvider(
                settings(), transport=httpx.MockTransport(lambda _: completion(payload))
            )
            result = ModelEnhancedExtractor(provider).extract(
                DocumentInput(semantic["id"], Path(source["path"]).name, text)
            )
            matching = [row for row in result.directives if row.kind == expected_kind]
            self.assertTrue(matching, semantic["id"])
            self.assertTrue(
                all(not eligible_for_deterministic_rules(row) for row in matching)
            )
            self.assertEqual([], detect_issues(result.directives))


if __name__ == "__main__":
    unittest.main()
