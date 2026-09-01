import json
import unittest

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.chunking import chunk_document, numbered_chunk
from app.config import Settings
from app.model_extractor import ModelEnhancedExtractor, SYSTEM_PROMPT
from app.pipeline import AnalysisPipeline, BaselineExtractor, DocumentInput
from app.provider import OpenAICompatibleProvider
from app.rate_limit import SlidingWindowLimiter, WriteRateLimitMiddleware
from app.usage import configured_cost_usd, estimate_request_tokens
from scripts.run_model_stability import run_stability


def completion(prompt_tokens=3, completion_tokens=2):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": '{"records":[]}'}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    })


def provider_settings(**overrides):
    values = {
        "openai_api_key": "unit-test-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_model_extraction": True,
        "provider_max_attempts": 1,
        "model_chunk_max_chars": 64,
        "model_chunk_overlap_lines": 0,
        "model_max_chunks_per_document": 20,
        "per_run_token_budget": 20_000,
    }
    values.update(overrides)
    return Settings(**values)


class TokenBudgetTests(unittest.TestCase):
    def test_exact_conservative_budget_allows_request_and_one_less_rejects(self):
        document = DocumentInput(id="doc", name="doc.md", content="普通叙述没有结构化状态。")
        chunk = chunk_document(document, 64, 0)[0]
        user = f"文档名：{document.name}\n当前分块：{chunk.id}，原文全局行 {chunk.global_line_start}-{chunk.global_line_end}\n以下文本使用原文全局行号：\n{numbered_chunk(chunk)}"
        exact = estimate_request_tokens(SYSTEM_PROMPT, user)
        for budget, expected_calls in ((exact, 1), (exact - 1, 0)):
            calls = 0

            def handler(_):
                nonlocal calls
                calls += 1
                return completion()

            provider = OpenAICompatibleProvider(
                provider_settings(per_run_token_budget=budget),
                transport=httpx.MockTransport(handler),
            )
            result = ModelEnhancedExtractor(provider).extract(document)
            self.assertEqual(expected_calls, calls)
            if not expected_calls:
                self.assertTrue(any("Token 预算不足" in warning for warning in result.warnings))

    def test_budget_exhaustion_between_chunks_keeps_full_baseline(self):
        calls = 0

        def handler(_):
            nonlocal calls
            calls += 1
            return completion(prompt_tokens=estimate, completion_tokens=0)

        document = DocumentInput(
            id="multi", name="multi.md",
            content="普通背景一没有状态。\n普通背景二没有状态。\n林澈的身份是领航员。\n普通背景四没有状态。",
        )
        first = chunk_document(document, 32, 0)[0]
        user = f"文档名：{document.name}\n当前分块：{first.id}，原文全局行 {first.global_line_start}-{first.global_line_end}\n以下文本使用原文全局行号：\n{numbered_chunk(first)}"
        estimate = estimate_request_tokens(SYSTEM_PROMPT, user)
        provider = OpenAICompatibleProvider(
            # Exactly reserve the first conservative request estimate.  The
            # provider reports that amount as actual usage, so no later chunk
            # can pass the next conservative preflight gate.
            provider_settings(model_chunk_max_chars=32, per_run_token_budget=estimate),
            transport=httpx.MockTransport(handler),
        )
        result = ModelEnhancedExtractor(provider).extract(document)
        self.assertEqual(1, calls)
        self.assertTrue(any(row.kind == "fact" and row.attrs.get("subject") == "林澈" for row in result.directives))
        self.assertTrue(any("后续分块由全文基线覆盖" in warning for warning in result.warnings))

    def test_unconfigured_and_configured_cost(self):
        self.assertIsNone(configured_cost_usd(1_000, 500, Settings()))
        value = configured_cost_usd(
            1_000, 500,
            Settings(model_input_price_per_million=2.0, model_output_price_per_million=6.0),
        )
        self.assertEqual(0.005, value)

    def test_missing_provider_usage_still_consumes_conservative_budget(self):
        calls = 0

        def handler(_):
            nonlocal calls
            calls += 1
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": '{"records":[]}'}}]},
            )

        document = DocumentInput(
            id="missing-usage",
            name="missing-usage.md",
            content=(
                "普通背景一没有状态。\n普通背景二没有状态。\n"
                "普通背景三没有状态。\n普通背景四没有状态。"
            ),
        )
        first = chunk_document(document, 32, 0)[0]
        user = (
            f"文档名：{document.name}\n当前分块：{first.id}，原文全局行 "
            f"{first.global_line_start}-{first.global_line_end}\n"
            f"以下文本使用原文全局行号：\n{numbered_chunk(first)}"
        )
        budget = estimate_request_tokens(SYSTEM_PROMPT, user)
        provider = OpenAICompatibleProvider(
            provider_settings(
                model_chunk_max_chars=32,
                per_run_token_budget=budget,
            ),
            transport=httpx.MockTransport(handler),
        )
        result = ModelEnhancedExtractor(provider).extract(document)
        self.assertEqual(1, calls)
        self.assertTrue(any("Token 预算不足" in warning for warning in result.warnings))


class RateLimitTests(unittest.TestCase):
    def test_sliding_window_and_retry_after_header(self):
        now = [0.0]
        limiter = SlidingWindowLimiter(2, 10, clock=lambda: now[0])
        self.assertTrue(limiter.check("client")[0])
        self.assertTrue(limiter.check("client")[0])
        allowed, retry_after = limiter.check("client")
        self.assertFalse(allowed)
        self.assertEqual(10, retry_after)
        now[0] = 11
        self.assertTrue(limiter.check("client")[0])

        app = FastAPI()
        app.add_middleware(WriteRateLimitMiddleware, limiter=SlidingWindowLimiter(1, 30, clock=lambda: 0.0))

        @app.post("/api/v1/write")
        def write():
            return {"ok": True}

        @app.get("/api/v1/read")
        def read():
            return {"ok": True}

        with TestClient(app) as client:
            self.assertEqual(200, client.post("/api/v1/write").status_code)
            limited = client.post("/api/v1/write")
            self.assertEqual(429, limited.status_code)
            self.assertEqual("30", limited.headers["Retry-After"])
            self.assertEqual(200, client.get("/api/v1/read").status_code)


class TimingAndStabilityTests(unittest.TestCase):
    def test_pipeline_exposes_monotonic_stage_timings(self):
        result = AnalysisPipeline(extractor=BaselineExtractor()).run([
            DocumentInput(id="doc", name="doc.md", content="林澈的发色是银色。")
        ])
        timings = result.diagnostics["timings"]
        self.assertIn("monotonic", timings["clock"])
        for name in ("chunk_ms", "extract_ms", "index_ms", "check_ms", "report_ms", "total_ms", "first_progress_ms"):
            self.assertGreaterEqual(timings[name], 0)
        self.assertGreaterEqual(timings["total_ms"], timings["check_ms"])

    def test_stability_aggregation_stops_on_total_token_budget(self):
        class FakePipeline:
            def run(self, documents, on_stage=None):
                if on_stage:
                    on_stage("extract", 10, "start")
                result = AnalysisPipeline(extractor=BaselineExtractor()).run(documents)
                result.model_used = True
                result.prompt_tokens = 4
                result.completion_tokens = 2
                return result

        report = run_stability(
            "advanced", repeats=5, max_total_tokens=10,
            pipeline_factory=FakePipeline,
        )
        self.assertEqual(2, len(report["runs"]))
        self.assertTrue(report["budget_exhausted"])
        self.assertEqual(12, report["summary"]["total_tokens"])
        self.assertEqual(1.0, report["summary"]["complete_target_success_rate"])
        serialized = json.dumps(report, ensure_ascii=False)
        self.assertNotIn("unit-test-placeholder", serialized)


if __name__ == "__main__":
    unittest.main()
