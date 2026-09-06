import json
import unittest

import httpx

from app.domain import AnalysisCancelled
from app.model_extractor import ModelEnhancedExtractor
from app.pipeline import AnalysisPipeline, BaselineExtractor, DocumentInput
from app.provider import OpenAICompatibleProvider, RetryPolicy
from tests.test_model_extractor import completion, settings


def fact(subject, value, line=1, **extra):
    return {
        "kind": "fact",
        "subject": subject,
        "predicate": "身份",
        "value": value,
        "source_line_start": line,
        "source_line_end": line,
        "modality": "asserted",
        "source_scope": "narrator",
        "certainty": "certain",
        **extra,
    }


class BatchModelExtractionTests(unittest.TestCase):
    def run_batch(self, payload, documents=None, **overrides):
        calls = []

        def handler(request):
            calls.append(request)
            return completion(json.dumps(payload, ensure_ascii=False))

        provider = OpenAICompatibleProvider(
            settings(**overrides),
            transport=httpx.MockTransport(handler),
            retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0),
            sleep=lambda _: None,
        )
        docs = documents or [
            DocumentInput("real-a", "a.md", "林澈的身份是领航员。", "canon", "global"),
            DocumentInput("real-b", "b.md", "苏弦的身份是档案官。", "chapter", "route_b"),
        ]
        result = AnalysisPipeline(extractor=ModelEnhancedExtractor(provider)).run(docs)
        return result, calls

    def test_success_uses_one_call_binds_context_and_sorts_by_input_document(self):
        payload = {
            "documents": [
                {"doc_ref": "d2", "records": [fact("苏弦", "档案官")]},
                {"doc_ref": "d1", "records": [fact("林澈", "领航员")]},
            ]
        }
        result, calls = self.run_batch(payload)
        self.assertEqual(1, len(calls))
        request = json.loads(calls[0].content)
        self.assertNotIn("max_tokens", request)
        self.assertNotIn("max_completion_tokens", request)
        self.assertNotIn("reasoning_effort", request)
        system = request["messages"][0]["content"]
        self.assertIn("同一 doc_ref、所填行号内的原文词组直接支持", system)
        self.assertIn("不要用摘要、近义改写、常识补全或跨行/跨文档拼接", system)
        user = request["messages"][1]["content"]
        self.assertIn('"doc_ref":"d1"', user)
        self.assertNotIn("real-a", user)
        ordered_ids = [row.evidence.document_id for row in result.directives]
        self.assertEqual(ordered_ids, sorted(ordered_ids, key={"real-a": 0, "real-b": 1}.get))
        self.assertEqual("canon", result.directives[0].attrs["document_role"])
        self.assertEqual("route_b", result.directives[-1].attrs["story_scope"])
        status = result.diagnostics["model"]
        self.assertTrue(status["batch_used"])
        self.assertEqual(2, status["batch_document_count"])
        self.assertEqual(1, status["logical_call_count"])
        self.assertEqual((12, 8), (result.prompt_tokens, result.completion_tokens))

    def test_empty_group_conserves_success_counts_but_is_not_model_coverage(self):
        result, calls = self.run_batch({
            "documents": [
                {"doc_ref": "d1", "records": []},
                {"doc_ref": "d2", "records": []},
            ]
        })
        self.assertEqual(1, len(calls))
        self.assertEqual(2, result.diagnostics["model"]["empty_response_chunks"])
        self.assertEqual(2, result.diagnostics["model"]["succeeded_chunks"])
        self.assertFalse(result.model_used)
        self.assertTrue(any("模型返回空结果，无法证明完整覆盖" in row for row in result.warnings))

    def test_one_empty_group_marks_only_that_document_and_degrades_whole_run(self):
        result, calls = self.run_batch({
            "documents": [
                {"doc_ref": "d1", "records": [fact("林澈", "领航员")]},
                {"doc_ref": "d2", "records": []},
            ]
        })
        self.assertEqual(1, len(calls))
        status = result.diagnostics["model"]
        self.assertEqual(1, status["empty_response_chunks"])
        self.assertEqual(2, status["succeeded_chunks"])
        self.assertTrue(result.model_used)
        self.assertEqual(
            [0, 1],
            [row["empty_response_chunks"] for row in status["documents"]],
        )
        self.assertTrue(any(
            warning.startswith("b.md: 模型返回空结果，无法证明完整覆盖")
            for warning in result.warnings
        ))

    def test_unknown_duplicate_and_missing_refs_fail_without_fanout(self):
        cases = [
            [{"doc_ref": "d1", "records": []}, {"doc_ref": "d3", "records": []}],
            [{"doc_ref": "d1", "records": []}, {"doc_ref": "d1", "records": []}],
            [{"doc_ref": "d1", "records": []}],
        ]
        for groups in cases:
            with self.subTest(groups=groups):
                result, calls = self.run_batch({"documents": groups})
                self.assertEqual(1, len(calls))
                self.assertFalse(result.model_used)
                self.assertEqual(1, result.diagnostics["model"]["failed_chunks"])
                self.assertEqual(1, result.diagnostics["model"]["invalid_records"])
                self.assertEqual(
                    1, result.diagnostics["model"]["unresolved_invalid_records"]
                )
                self.assertIn("batch_protocol", result.diagnostics["model"]["reason_codes"])

    def test_last_group_failure_atomically_discards_earlier_valid_model_output(self):
        calls = []
        payload = {"documents": [
            {"doc_ref": "d1", "records": [fact("林澈", "领航员")]},
            {"doc_ref": "d2", "records": [fact("苏弦", "档案官", line=9)]},
        ]}
        provider = OpenAICompatibleProvider(
            settings(),
            transport=httpx.MockTransport(
                lambda request: calls.append(request)
                or completion(json.dumps(payload, ensure_ascii=False))
            ),
        )
        extractor = ModelEnhancedExtractor(provider)
        extractor.begin_run()
        documents = [
            DocumentInput("a", "a.md", "林澈的身份是领航员。"),
            DocumentInput("b", "b.md", "苏弦的身份是档案官。"),
        ]
        parsed = extractor.extract_batch(documents)
        self.assertIsNotNone(parsed)
        self.assertEqual(1, len(calls))
        for document, actual in zip(documents, parsed, strict=True):
            expected = BaselineExtractor().extract(document)
            self.assertEqual(
                [row.model_dump() for row in expected.directives],
                [row.model_dump() for row in actual.directives],
            )
            status = actual.model_execution
            self.assertEqual(
                status.attempted_chunks,
                status.succeeded_chunks + status.failed_chunks,
            )
            self.assertEqual(
                status.total_chunks,
                status.attempted_chunks + status.skipped_chunks,
            )
        self.assertEqual(1, sum(row.model_execution.attempted_chunks for row in parsed))
        self.assertEqual(1, sum(row.model_execution.failed_chunks for row in parsed))
        self.assertEqual(1, sum(row.model_execution.invalid_records for row in parsed))
        self.assertEqual(1, sum(
            row.model_execution.unresolved_invalid_records or 0 for row in parsed
        ))
        self.assertEqual(0, sum(
            row.model_execution.recovered_invalid_records or 0 for row in parsed
        ))

    def test_label_candidate_before_fatal_schema_is_atomically_unresolved(self):
        candidate = fact("林澈", "领航员")
        for field in ("modality", "source_scope", "certainty"):
            candidate.pop(field)
        fatal = fact("苏弦", "档案官")
        fatal.pop("predicate")
        result, calls = self.run_batch({"documents": [
            {"doc_ref": "d1", "records": [candidate]},
            {"doc_ref": "d2", "records": [fatal]},
        ]})

        self.assertEqual(1, len(calls))
        self.assertFalse(result.model_used)
        status = result.diagnostics["model"]
        self.assertEqual((2, 2, 0), (
            status["invalid_records"],
            status["unresolved_invalid_records"],
            status["recovered_invalid_records"],
        ))
        self.assertEqual(
            status["invalid_records"],
            status["unresolved_invalid_records"]
            + status["recovered_invalid_records"],
        )
        self.assertIn("semantic_labels_quarantined", status["reason_codes"])
        self.assertIn("schema_validation", status["reason_codes"])
        self.assertIn("batch_protocol", status["reason_codes"])

    def test_content_invalid_record_is_isolated_without_weakening_attribution(self):
        documents = [
            DocumentInput("a", "a.md", "林澈的身份是领航员。"),
            DocumentInput("b", "b.md", "苏弦的身份是档案官。"),
        ]
        unsupported = fact("顾青", "守卫")
        result, calls = self.run_batch({"documents": [
            {"doc_ref": "d1", "records": [fact("林澈", "领航员")]},
            {"doc_ref": "d2", "records": [unsupported]},
        ]}, documents)
        self.assertEqual(1, len(calls))
        self.assertTrue(result.model_used)
        status = result.diagnostics["model"]
        self.assertEqual((1, 1, 0), (
            status["invalid_records"],
            status["unresolved_invalid_records"],
            status["recovered_invalid_records"],
        ))
        self.assertEqual((1, 1), (status["succeeded_chunks"], status["failed_chunks"]))
        self.assertIn("lexical_support", status["reason_codes"])
        self.assertNotIn("batch_protocol", status["reason_codes"])
        self.assertEqual(
            [(1, 0), (0, 1)],
            [
                (row["succeeded_chunks"], row["failed_chunks"])
                for row in status["documents"]
            ],
        )
        model_sources = {
            directive.evidence.document_id: directive.provenance_sources
            for directive in result.directives
        }
        self.assertIn("model", model_sources["a"])
        self.assertNotIn("model", model_sources["b"])

    def test_mixed_valid_and_content_invalid_records_keep_document_success_partial(self):
        result, calls = self.run_batch({"documents": [
            {"doc_ref": "d1", "records": [
                fact("林澈", "领航员"),
                fact("不存在的人", "虚构身份"),
            ]},
            {"doc_ref": "d2", "records": [fact("苏弦", "档案官")]},
        ]})
        self.assertEqual(1, len(calls))
        status = result.diagnostics["model"]
        self.assertEqual((1, 1, 2, 0), (
            status["invalid_records"],
            status["unresolved_invalid_records"],
            status["succeeded_chunks"],
            status["failed_chunks"],
        ))
        self.assertTrue(all(
            row["succeeded_chunks"] == 1 for row in status["documents"]
        ))

    def test_cross_document_line_and_context_deception_fail_whole_batch(self):
        documents = [
            DocumentInput("a", "a.md", "林澈的身份是领航员。", "canon", "route_a"),
            DocumentInput("b", "b.md", "空行。\n苏弦的身份是档案官。", "chapter", "route_b"),
        ]
        cross = {
            "documents": [
                {"doc_ref": "d1", "records": [fact("苏弦", "档案官", line=2)]},
                {"doc_ref": "d2", "records": []},
            ]
        }
        deceptive = {
            "documents": [
                {"doc_ref": "d1", "records": [fact("林澈", "领航员", story_scope="route_b")]},
                {"doc_ref": "d2", "records": []},
            ]
        }
        for payload in (cross, deceptive):
            with self.subTest(payload=payload):
                result, calls = self.run_batch(payload, documents)
                self.assertEqual(1, len(calls))
                self.assertFalse(result.model_used)
                self.assertIn("batch_protocol", result.diagnostics["model"]["reason_codes"])
                self.assertEqual({"route_a", "route_b"}, {
                    row.attrs.get("story_scope") for row in result.directives
                })

    def test_shared_subject_predicate_cannot_import_another_documents_value(self):
        documents = [
            DocumentInput("a", "a.md", "林澈的身份是领航员。", "chapter", "route_a"),
            DocumentInput("b", "b.md", "林澈的身份是档案官。", "chapter", "route_b"),
        ]
        for modality, source_scope in (
            ("asserted", "narrator"),
            ("uncertain", "unknown"),
            ("reported", "character_dialogue"),
            ("hypothetical", "unknown"),
        ):
            wrong = fact("林澈", "档案官")
            wrong.update(modality=modality, source_scope=source_scope)
            payload = {"documents": [
                {"doc_ref": "d1", "records": [wrong]},
                {"doc_ref": "d2", "records": []},
            ]}
            with self.subTest(modality=modality):
                result, calls = self.run_batch(payload, documents)
                self.assertEqual(1, len(calls))
                self.assertFalse(result.model_used)
                self.assertFalse(any(
                    row.evidence.document_id == "a"
                    and row.attrs.get("value") == "档案官"
                    for row in result.directives
                ))
                self.assertFalse(any(issue.category.value == "fact_conflict" for issue in result.issues))

        valid = {"documents": [
            {"doc_ref": "d1", "records": [fact("林澈", "领航员")]},
            {"doc_ref": "d2", "records": [fact("林澈", "档案官")]},
        ]}
        result, calls = self.run_batch(valid, documents)
        self.assertEqual(1, len(calls))
        self.assertTrue(result.model_used)
        self.assertEqual(2, result.diagnostics["model"]["succeeded_chunks"])
        self.assertFalse(any(issue.category.value == "fact_conflict" for issue in result.issues))

    def test_nonasserted_fact_still_requires_contiguous_predicate_and_value(self):
        documents = [
            DocumentInput("a", "a.md", "旅人甲的身份是领航员。", scope="route_a"),
            DocumentInput("b", "b.md", "旅人甲的代号是领航员。", scope="route_b"),
        ]
        for modality, source_scope in (
            ("uncertain", "unknown"),
            ("hypothetical", "unknown"),
            ("reported", "character_dialogue"),
        ):
            wrong = fact("旅人甲", "领航员")
            wrong.update(
                predicate="代号",
                modality=modality,
                source_scope=source_scope,
                certainty="possible" if modality != "reported" else "unknown",
            )
            with self.subTest(modality=modality):
                result, calls = self.run_batch(
                    {"documents": [
                        {"doc_ref": "d1", "records": [wrong]},
                        {"doc_ref": "d2", "records": []},
                    ]},
                    documents,
                )
                self.assertEqual(1, len(calls))
                self.assertFalse(result.model_used)
                self.assertFalse(any(
                    row.evidence.document_id == "a"
                    and row.attrs.get("predicate") == "代号"
                    for row in result.directives
                ))
                self.assertFalse(any(
                    issue.category.value == "fact_conflict" for issue in result.issues
                ))

    def test_batch_threshold_falls_back_to_legacy_per_document_path(self):
        result, calls = self.run_batch(
            {"records": []}, model_batch_max_estimated_tokens=1
        )
        self.assertEqual(2, len(calls))
        self.assertFalse(result.diagnostics["model"]["batch_used"])

    def test_batch_budget_boundary_is_inclusive_and_one_below_falls_back(self):
        payload = {"documents": [
            {"doc_ref": "d1", "records": []},
            {"doc_ref": "d2", "records": []},
        ]}
        admitted, calls = self.run_batch(payload)
        estimate = admitted.diagnostics["model"]["batch_budget_control"]["estimated_tokens"]
        at_boundary, boundary_calls = self.run_batch(
            payload, model_batch_max_estimated_tokens=estimate
        )
        below, below_calls = self.run_batch(
            {"records": []}, model_batch_max_estimated_tokens=estimate - 1
        )
        self.assertEqual(1, len(calls))
        self.assertEqual(1, len(boundary_calls))
        self.assertTrue(at_boundary.diagnostics["model"]["batch_used"])
        self.assertEqual(2, len(below_calls))
        self.assertFalse(below.diagnostics["model"]["batch_used"])

    def test_oversized_batch_completion_is_rejected_atomically_without_fanout(self):
        payload = {"documents": [
            {"doc_ref": "d1", "records": [fact("旅人甲", "领航员")] * 41},
            {"doc_ref": "d2", "records": []},
        ]}
        documents = [
            DocumentInput("a", "a.md", "旅人甲的身份是领航员。"),
            DocumentInput("b", "b.md", "旅人乙的身份是档案官。"),
        ]
        result, calls = self.run_batch(payload, documents)
        self.assertEqual(1, len(calls))
        self.assertFalse(result.model_used)
        self.assertEqual(1, result.diagnostics["model"]["logical_call_count"])
        self.assertIn("batch_protocol", result.diagnostics["model"]["reason_codes"])
        for document in documents:
            expected = BaselineExtractor().extract(document)
            actual = [
                row.model_dump() for row in result.directives
                if row.evidence.document_id == document.id
            ]
            self.assertEqual([row.model_dump() for row in expected.directives], actual)

    def test_provider_failure_is_one_logical_call_and_never_fans_out(self):
        calls = []

        def handler(request):
            calls.append(request)
            return completion("{}", status=503)

        provider = OpenAICompatibleProvider(
            settings(),
            transport=httpx.MockTransport(handler),
            retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0),
            sleep=lambda _: None,
        )
        result = AnalysisPipeline(extractor=ModelEnhancedExtractor(provider)).run([
            DocumentInput("a", "a.md", "旅人甲的身份是领航员。"),
            DocumentInput("b", "b.md", "旅人乙的身份是档案官。"),
        ])
        model = result.diagnostics["model"]
        self.assertEqual(1, len(calls))
        self.assertEqual(1, model["logical_call_count"])
        self.assertEqual(model["attempted_chunks"], model["failed_chunks"])
        self.assertEqual(model["total_chunks"], model["attempted_chunks"] + model["skipped_chunks"])
        self.assertFalse(result.model_used)

    def test_batch_budget_diagnostics_disclose_local_estimate_without_hard_cap(self):
        result, _ = self.run_batch({"documents": [
            {"doc_ref": "d1", "records": []},
            {"doc_ref": "d2", "records": []},
        ]})
        budget = result.diagnostics["model"]["batch_budget_control"]
        self.assertEqual("local_conservative_estimate_only", budget["mode"])
        self.assertFalse(budget["provider_output_hard_limit"])
        self.assertGreater(budget["estimated_tokens"], 0)
        self.assertEqual(40, budget["max_records_per_document"])

    def test_record_and_group_order_do_not_change_stable_result_order(self):
        documents = [
            DocumentInput(
                "a", "a.md", "设定同时记载林澈的身份是领航员、档案官。"
            ),
            DocumentInput("b", "b.md", "苏弦的身份是档案官。"),
        ]
        first_fact = fact("林澈", "领航员")
        second_fact = fact("林澈", "档案官")
        payloads = [
            {"documents": [
                {"doc_ref": "d1", "records": [first_fact, second_fact]},
                {"doc_ref": "d2", "records": []},
            ]},
            {"documents": [
                {"doc_ref": "d2", "records": []},
                {"doc_ref": "d1", "records": [second_fact, first_fact]},
            ]},
        ]
        orders = []
        for payload in payloads:
            result, _ = self.run_batch(payload, documents)
            orders.append([
                (
                    row.evidence.document_id,
                    row.evidence.line_start,
                    row.evidence.line_end,
                    row.kind,
                    json.dumps(row.attrs, ensure_ascii=False, sort_keys=True),
                )
                for row in result.directives
            ])
        self.assertEqual(orders[0], orders[1])

    def test_cancellation_is_checked_before_and_after_shared_call(self):
        payload = {"documents": [
            {"doc_ref": "d1", "records": []},
            {"doc_ref": "d2", "records": []},
        ]}
        for cancel_at, expected_calls in ((1, 0), (3, 1)):
            calls = []
            provider = OpenAICompatibleProvider(
                settings(),
                transport=httpx.MockTransport(
                    lambda request: calls.append(request) or completion(json.dumps(payload))
                ),
            )
            checkpoints = 0

            def checkpoint():
                nonlocal checkpoints
                checkpoints += 1
                if checkpoints == cancel_at:
                    raise AnalysisCancelled("cancelled")

            with self.assertRaises(AnalysisCancelled):
                AnalysisPipeline(extractor=ModelEnhancedExtractor(provider)).run(
                    [
                        DocumentInput("a", "a.md", "林澈的身份是领航员。"),
                        DocumentInput("b", "b.md", "苏弦的身份是档案官。"),
                    ],
                    checkpoint=checkpoint,
                )
            self.assertEqual(expected_calls, len(calls))

    def test_multi_chunk_document_keeps_legacy_path(self):
        calls = []
        provider = OpenAICompatibleProvider(
            settings(model_chunk_max_chars=32, model_chunk_overlap_lines=0),
            transport=httpx.MockTransport(
                lambda request: calls.append(request) or completion('{"records":[]}')
            ),
        )
        result = AnalysisPipeline(extractor=ModelEnhancedExtractor(provider)).run([
            DocumentInput("long", "long.md", "普通背景文字。" * 20),
            DocumentInput("short", "short.md", "苏弦的身份是档案官。"),
        ])
        self.assertGreater(len(calls), 2)
        self.assertFalse(result.diagnostics["model"]["batch_used"])


if __name__ == "__main__":
    unittest.main()
