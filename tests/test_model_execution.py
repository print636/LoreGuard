import json
import unittest

import httpx

from app.model_extractor import ModelEnhancedExtractor
from app.pipeline import AnalysisPipeline, DocumentInput
from app.provider import OpenAICompatibleProvider, RetryPolicy
from app.service import analysis_mode
from tests.test_model_extractor import completion, settings


class ModelExecutionTests(unittest.TestCase):
    def run_pipeline(self, payloads, documents=None, **overrides):
        calls = []
        responses = iter(payloads)

        def handler(request):
            calls.append(request)
            value = next(responses)
            if isinstance(value, Exception):
                raise value
            return completion(json.dumps(value, ensure_ascii=False))

        provider = OpenAICompatibleProvider(
            settings(**overrides), transport=httpx.MockTransport(handler),
            retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0),
            sleep=lambda _: None,
        )
        pipeline = AnalysisPipeline(extractor=ModelEnhancedExtractor(provider))
        result = pipeline.run(documents or [DocumentInput("doc", "chapter.md", "林澈的身份是领航员。")])
        status = result.diagnostics["model"]
        self.assertEqual(status["attempted_chunks"], len(calls))
        self.assertEqual(status["total_chunks"], status["attempted_chunks"] + status["skipped_chunks"])
        self.assertEqual(status["attempted_chunks"], status["succeeded_chunks"] + status["failed_chunks"])
        return result, status, pipeline

    def test_explicit_empty_records_is_success_but_missing_records_is_invalid(self):
        result, status, _ = self.run_pipeline([{"records": []}])
        self.assertEqual(1, status["empty_response_chunks"])
        self.assertEqual(("完整模型增强", False), analysis_mode(result))
        result, status, _ = self.run_pipeline([{}])
        self.assertEqual(0, status["empty_response_chunks"])
        self.assertEqual(1, status["failed_chunks"])
        self.assertIn("schema_validation", status["reason_codes"])
        self.assertEqual(("确定性基线（模型未参与或已降级）", True), analysis_mode(result))

    def test_partial_valid_records_are_never_complete_and_warning_text_is_irrelevant(self):
        payload = {"records": [
            {"kind": "fact", "subject": "林澈", "predicate": "身份", "value": "领航员",
             "source_line_start": 1, "source_line_end": 1},
            {"kind": "event", "description": "invalid"},
        ]}
        result, status, _ = self.run_pipeline([payload])
        self.assertEqual(1, status["succeeded_chunks"])
        self.assertEqual(1, status["invalid_records"])
        self.assertEqual(0, status["failed_chunks"])
        result.warnings.clear()
        self.assertEqual(("模型增强（部分分块已降级）", True), analysis_mode(result))
        complete, _, _ = self.run_pipeline([{"records": []}])
        complete.warnings = ["模型分块 运行级熔断 后续模型分块已停止 降级到 BaselineExtractor"]
        self.assertEqual(("完整模型增强", False), analysis_mode(complete))

    def test_budget_before_first_call_is_zero_call_fallback(self):
        result, status, _ = self.run_pipeline([], per_run_token_budget=1)
        self.assertEqual(0, status["attempted_chunks"])
        self.assertEqual(1, status["skipped_chunks"])
        self.assertEqual(["token_budget"], status["reason_codes"])
        self.assertTrue(result.directives)
        self.assertEqual(("确定性基线（模型未参与或已降级）", True), analysis_mode(result))

    def test_budget_after_success_and_document_limit_are_partial(self):
        documents = [DocumentInput("doc", "chapter.md", "林澈的身份是领航员。" * 7)]
        for overrides, reason in [
            ({"per_run_token_budget": 2500}, "token_budget"),
            ({"model_max_chunks_per_document": 1}, "chunk_limit"),
        ]:
            with self.subTest(reason=reason):
                result, status, _ = self.run_pipeline(
                    [{"records": []}] * 10, documents,
                    model_chunk_max_chars=32, model_chunk_overlap_lines=0, **overrides,
                )
                self.assertGreater(status["succeeded_chunks"], 0)
                self.assertGreater(status["skipped_chunks"], 0)
                self.assertIn(reason, status["reason_codes"])
                self.assertEqual(("模型增强（部分分块已降级）", True), analysis_mode(result))

    def test_provider_failure_aborts_document_then_circuit_skips_later_document(self):
        documents = [
            DocumentInput("first", "first.md", "林澈的身份是领航员。" * 7),
            DocumentInput("second", "second.md", "苏弦的身份是档案官。"),
        ]
        result, status, pipeline = self.run_pipeline(
            [httpx.ReadTimeout("private upstream payload must not appear")], documents,
            model_chunk_max_chars=32, model_chunk_overlap_lines=0,
        )
        self.assertEqual(1, status["failed_chunks"])
        self.assertIn("document_aborted", status["reason_codes"])
        self.assertIn("provider_error", status["reason_codes"])
        self.assertEqual(0, status["documents"][1]["attempted_chunks"])
        self.assertEqual(["circuit_open"], status["documents"][1]["reason_codes"])
        self.assertNotIn("private upstream", json.dumps(status))
        self.assertTrue(result.directives)
        pipeline.extractor.begin_run()
        self.assertFalse(pipeline.extractor._circuit_open)

    def test_all_invalid_records_are_not_empty_success_and_usage_is_preserved(self):
        result, status, _ = self.run_pipeline([{"records": [{"kind": "bad"}, {"kind": "bad"}]}])
        self.assertEqual(2, status["invalid_records"])
        self.assertEqual(1, status["failed_chunks"])
        self.assertEqual(0, status["succeeded_chunks"])
        self.assertEqual(0, status["empty_response_chunks"])
        self.assertEqual((12, 8), (result.prompt_tokens, result.completion_tokens))

    def test_disabled_unconfigured_and_empty_documents_never_claim_complete(self):
        result, status, _ = self.run_pipeline([], enable_model_extraction=False)
        self.assertFalse(status["enabled"])
        self.assertEqual(("确定性基线", False), analysis_mode(result))
        result, status, _ = self.run_pipeline([], openai_api_key="")
        self.assertTrue(status["enabled"])
        self.assertFalse(status["configured"])
        self.assertEqual(["not_configured"], status["reason_codes"])
        self.assertTrue(analysis_mode(result)[1])
        result, status, _ = self.run_pipeline([], [DocumentInput("empty", "empty.md", "")])
        self.assertEqual(0, status["total_chunks"])
        self.assertNotEqual("完整模型增强", analysis_mode(result)[0])


if __name__ == "__main__":
    unittest.main()
