from __future__ import annotations

from collections import Counter
import unittest

from app.chunking import chunk_document
from app.config import Settings
from app.domain import DocumentRole
from app.model_extractor import ModelEnhancedExtractor, RECORD_ADAPTER
from app.pipeline import DocumentInput
from app.provider import OpenAICompatibleProvider
from scripts.run_agent_acceptance import (
    DEFAULT_SUITE_ROOT,
    build_agent_input,
    build_mock_trace_bundle,
    frozen_document_errors,
    load_manifest,
    score_trace_bundle,
)


class AgentAcceptanceManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = DEFAULT_SUITE_ROOT.resolve()
        cls.manifest = load_manifest(cls.root)

    def test_frozen_suite_has_30_balanced_tasks_and_90_executions(self):
        manifest = self.manifest
        tasks = manifest["tasks"]
        personas = Counter(task["persona"] for task in tasks)
        scenarios = Counter(task["scenario"] for task in tasks)
        safety_tasks = [task for task in tasks if task["safety_focus"]]
        safety_focus = {
            focus for task in safety_tasks for focus in task["safety_focus"]
        }

        self.assertEqual("agent-acceptance-v1", manifest["suite_id"])
        self.assertEqual("developer-visible-non-blind", manifest["visibility"])
        self.assertEqual(
            "trace-import-adapter-only-production-invocation-disabled",
            manifest["agent_connection"],
        )
        self.assertEqual(30, len(tasks))
        self.assertEqual(30, len({task["id"] for task in tasks}))
        self.assertEqual(set(manifest["personas"]), set(personas))
        self.assertEqual({10}, set(personas.values()))
        self.assertEqual(3, manifest["repetitions_per_task"])
        self.assertEqual(90, len(tasks) * manifest["repetitions_per_task"])
        self.assertEqual(90, manifest["acceptance_gates"]["execution_count"])
        self.assertEqual([], frozen_document_errors(manifest, self.root))
        self.assertEqual(
            {"lexical_support"}, {task["validator_reason"] for task in tasks}
        )
        self.assertGreaterEqual(scenarios["recover_after_read_patch"], 10)
        self.assertGreaterEqual(scenarios["direct_abstain"], 8)
        self.assertGreaterEqual(scenarios["failed_patch_then_abstain"], 6)
        self.assertGreaterEqual(len(safety_tasks), 6)
        self.assertTrue(
            {"semantic_labels", "question", "quotation", "cross_branch"}
            <= safety_focus
        )

    def test_every_task_has_bounded_documents_evidence_and_oracle_contract(self):
        manifest = self.manifest
        allowed_roles = {role.value for role in DocumentRole}
        allowed_reasons = set(manifest["validator_reasons"])
        forbidden_catalog = set(manifest["forbidden_transformations_catalog"])
        expected_paths = {
            "recover_after_read_patch": ["READ_SPAN", "PATCH_RECORDS"],
            "direct_abstain": ["ABSTAIN"],
            "failed_patch_then_abstain": [
                "READ_SPAN",
                "PATCH_RECORDS",
                "FINALIZE_ABSTAIN",
            ],
        }
        self.assertEqual(
            {"READ_SPAN", "PATCH_RECORDS", "ABSTAIN"}, set(manifest["tools"])
        )
        self.assertEqual(2, manifest["limits"]["max_rounds"])
        self.assertEqual(6, manifest["limits"]["max_tool_calls"])

        for task in manifest["tasks"]:
            with self.subTest(task=task["id"]):
                documents = task["documents"]
                doc_refs = {document["doc_ref"] for document in documents}
                self.assertEqual(len(documents), len(doc_refs))
                self.assertIn(task["initial_candidate"]["doc_ref"], doc_refs)
                self.assertIn(task["validator_reason"], allowed_reasons)
                self.assertEqual(
                    expected_paths[task["scenario"]],
                    task["oracle"]["expected_action_path"],
                )
                self.assertTrue(task["oracle"]["forbidden_transformations"])
                self.assertTrue(
                    set(task["oracle"]["forbidden_transformations"])
                    <= forbidden_catalog
                )

                if task["scenario"] == "recover_after_read_patch":
                    self.assertTrue(task["oracle"]["should_recover"])
                    self.assertTrue(task["oracle"]["expected_patch"])
                    self.assertNotIn("attempt_patch", task["oracle"])
                elif task["scenario"] == "failed_patch_then_abstain":
                    self.assertFalse(task["oracle"]["should_recover"])
                    self.assertTrue(task["oracle"]["attempt_patch"])
                    self.assertNotIn("expected_patch", task["oracle"])
                else:
                    self.assertFalse(task["oracle"]["should_recover"])
                    self.assertNotIn("expected_patch", task["oracle"])

                line_counts: dict[str, int] = {}
                for document in documents:
                    self.assertIn(document["role"], allowed_roles)
                    self.assertRegex(document["scope"], r"^[a-z][a-z0-9_]{0,31}$")
                    path = (self.root / document["path"]).resolve()
                    path.relative_to(self.root)
                    self.assertTrue(path.is_file())
                    text = path.read_text(encoding="utf-8")
                    self.assertTrue(text.strip())
                    line_counts[document["doc_ref"]] = len(text.splitlines())

                candidate = task["initial_candidate"]
                self.assertGreaterEqual(candidate["source_line_start"], 1)
                self.assertGreaterEqual(
                    candidate["source_line_end"], candidate["source_line_start"]
                )
                self.assertLessEqual(
                    candidate["source_line_end"], line_counts[candidate["doc_ref"]]
                )
                patch = task["oracle"].get(
                    "expected_patch", task["oracle"].get("attempt_patch", {})
                )
                self.assertFalse(
                    {
                        "kind",
                        "doc_ref",
                        "role",
                        "scope",
                        "source_line_start",
                        "source_line_end",
                    }.intersection(patch)
                )

                self.assertTrue(task["allowed_evidence"])
                for span in task["allowed_evidence"]:
                    self.assertIn(span["doc_ref"], doc_refs)
                    self.assertGreaterEqual(span["line_start"], 1)
                    self.assertGreaterEqual(span["line_end"], span["line_start"])
                    self.assertLessEqual(span["line_end"], line_counts[span["doc_ref"]])
                    self.assertLessEqual(
                        span["line_end"] - span["line_start"] + 1,
                        manifest["limits"]["max_read_lines_per_call"],
                    )

    def test_recoverable_oracles_pass_current_schema_and_frozen_evidence_gates(self):
        settings = Settings(
            _env_file=None,
            openai_api_key="unit-test-placeholder",
            openai_base_url="https://mock.invalid/v1",
            openai_model="mock-model",
            enable_model_extraction=True,
            enable_review_agent=False,
            provider_timeout_seconds=1,
            provider_max_attempts=1,
            provider_thinking_mode=None,
        )
        extractor = ModelEnhancedExtractor(OpenAICompatibleProvider(settings))

        for task in self.manifest["tasks"]:
            if not task["oracle"]["should_recover"]:
                continue
            with self.subTest(task=task["id"]):
                raw = {
                    **task["initial_candidate"],
                    **task["oracle"]["expected_patch"],
                }
                raw.pop("record_id")
                raw.pop("doc_ref")
                RECORD_ADAPTER.validate_python(raw)

                doc_ref = task["initial_candidate"]["doc_ref"]
                spec = next(
                    document
                    for document in task["documents"]
                    if document["doc_ref"] == doc_ref
                )
                path = (self.root / spec["path"]).resolve()
                document = DocumentInput(
                    id=task["id"],
                    name=path.name,
                    content=path.read_text(encoding="utf-8"),
                    role=spec["role"],
                    scope=spec["scope"],
                )
                chunk = chunk_document(document, max_chars=100_000)[0]
                assessed, repair, _ = extractor._validate_or_quarantine(
                    document,
                    chunk,
                    raw,
                    source_record_index=0,
                    repair_index=0,
                    doc_ref=doc_ref,
                )
                self.assertIsNotNone(assessed)
                self.assertIsNone(repair)

    def test_all_initial_candidates_enter_only_the_production_lexical_gate(self):
        settings = Settings(
            _env_file=None,
            openai_api_key="unit-test-placeholder",
            openai_base_url="https://mock.invalid/v1",
            openai_model="mock-model",
            enable_model_extraction=True,
            enable_review_agent=True,
            provider_timeout_seconds=1,
            provider_max_attempts=1,
            provider_thinking_mode=None,
        )
        extractor = ModelEnhancedExtractor(OpenAICompatibleProvider(settings))

        for task in self.manifest["tasks"]:
            with self.subTest(task=task["id"]):
                raw = dict(task["initial_candidate"])
                raw.pop("record_id")
                doc_ref = raw.pop("doc_ref")
                RECORD_ADAPTER.validate_python(raw)

                spec = next(
                    document
                    for document in task["documents"]
                    if document["doc_ref"] == doc_ref
                )
                path = (self.root / spec["path"]).resolve()
                document = DocumentInput(
                    id=task["id"],
                    name=path.name,
                    content=path.read_text(encoding="utf-8"),
                    role=spec["role"],
                    scope=spec["scope"],
                )
                chunk = chunk_document(document, max_chars=100_000)[0]
                assessed, repair, semantic_reason = extractor._validate_or_quarantine(
                    document,
                    chunk,
                    raw,
                    source_record_index=0,
                    repair_index=1,
                    doc_ref=doc_ref,
                )
                self.assertIsNone(assessed)
                self.assertIsNotNone(repair)
                self.assertIsNone(semantic_reason)
                self.assertEqual(("lexical_support",), repair.error_codes)
                self.assertEqual(doc_ref, repair.doc_ref)
                self.assertEqual(spec["role"], repair.document.role)
                self.assertEqual(spec["scope"], repair.document.scope)
                self.assertTrue(repair.provisional_directive.evidence.text)
                self.assertEqual(
                    raw["source_line_start"],
                    repair.provisional_directive.evidence.line_start,
                )
                self.assertEqual(
                    raw["source_line_end"],
                    repair.provisional_directive.evidence.line_end,
                )

                if task["scenario"] == "failed_patch_then_abstain":
                    attempted = {**raw, **task["oracle"]["attempt_patch"]}
                    attempted_assessed, attempted_repair, _ = (
                        extractor._validate_or_quarantine(
                            document,
                            chunk,
                            attempted,
                            source_record_index=0,
                            repair_index=1,
                            doc_ref=doc_ref,
                        )
                    )
                    self.assertIsNone(attempted_assessed)
                    self.assertIsNotNone(attempted_repair)
                    self.assertEqual(
                        ("lexical_support",), attempted_repair.error_codes
                    )

    def test_agent_payload_excludes_manifest_answers_and_document_bodies(self):
        forbidden_keys = {
            "allowed_evidence",
            "oracle",
            "expected_action_path",
            "should_recover",
            "expected_patch",
            "attempt_patch",
            "forbidden_transformations",
        }

        def keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)

        for task in self.manifest["tasks"]:
            with self.subTest(task=task["id"]):
                payload = build_agent_input(task, self.root, repeat_index=2)
                self.assertEqual(f"{task['id']}::repeat-2", payload["execution_id"])
                self.assertFalse(forbidden_keys.intersection(keys(payload)))
                for document in payload["documents"]:
                    self.assertNotIn("content", document)
                    self.assertNotIn("path", document)
                    self.assertRegex(document["content_sha256"], r"^[a-f0-9]{64}$")

    def test_mock_trace_exercises_scorer_without_claiming_agent_results(self):
        bundle = build_mock_trace_bundle(self.manifest, self.root)
        report = score_trace_bundle(self.manifest, bundle)

        self.assertEqual(90, len(bundle["traces"]))
        self.assertTrue(report["passed"])
        self.assertTrue(report["gates"]["suite_id_match"])
        self.assertEqual(90, report["coverage"]["trace_count"])
        self.assertEqual(3, report["coverage"]["dynamic_path_count"])
        self.assertEqual(1.0, report["metrics"]["recovery_rate"])
        self.assertEqual(1.0, report["metrics"]["unrecoverable_abstain_rate"])
        self.assertEqual(30, report["metrics"]["fully_passed_repeated_tasks"])
        self.assertEqual(1.0, report["metrics"]["fully_passed_repeated_task_rate"])
        self.assertEqual(0, report["metrics"]["accepted_safety_violations"])
        self.assertTrue(report["metrics"]["trace_replay_complete"])
        self.assertFalse(report["benchmark"]["production_agent_connected"])
        self.assertFalse(report["benchmark"]["eligible_for_model_claims"])
        self.assertEqual(
            {"mock-oracle-fixture"}, {trace["source"] for trace in bundle["traces"]}
        )
        failed_task_ids = {
            task["id"]
            for task in self.manifest["tasks"]
            if task["scenario"] == "failed_patch_then_abstain"
        }
        failed_traces = [
            trace for trace in bundle["traces"] if trace["task_id"] in failed_task_ids
        ]
        self.assertEqual(18, len(failed_traces))
        for trace in failed_traces:
            self.assertEqual(
                ["READ_SPAN", "PATCH_RECORDS"],
                [action["tool"] for action in trace["actions"]],
            )
            self.assertEqual(2, max(action["round"] for action in trace["actions"]))
            self.assertEqual("rejected", trace["actions"][-1]["result"]["status"])
            self.assertEqual("abstained", trace["final"]["status"])
            self.assertEqual("FINALIZE", trace["final"]["terminal_action"])
            self.assertEqual("round_limit", trace["final"]["terminal_reason"])

    def test_scorer_fails_closed_on_accepted_boundary_violations(self):
        mutators = {
            "cross_document": lambda action: action["arguments"].update(doc_ref="d9"),
            "read_outside_allowed_evidence": lambda action: action["arguments"].update(
                line_start=0
            ),
            "unknown_tool": lambda action: action.update(tool="SEARCH_WEB"),
        }

        for expected_violation, mutate in mutators.items():
            with self.subTest(violation=expected_violation):
                bundle = build_mock_trace_bundle(self.manifest, self.root)
                trace = next(
                    row
                    for row in bundle["traces"]
                    if row["actions"] and row["actions"][0]["tool"] == "READ_SPAN"
                )
                action = trace["actions"][0]
                mutate(action)
                action["result"]["status"] = "ok"
                report = score_trace_bundle(self.manifest, bundle)
                self.assertFalse(report["passed"])
                self.assertGreater(
                    report["metrics"]["accepted_safety_violations"], 0
                )
                self.assertIn(
                    expected_violation,
                    report["metrics"]["accepted_safety_violation_types"],
                )


if __name__ == "__main__":
    unittest.main()
