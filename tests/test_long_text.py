import json
import re
import unittest
from pathlib import Path

import httpx

from app.chunking import chunk_document
from app.config import Settings
from app.model_extractor import ModelEnhancedExtractor
from app.pipeline import AnalysisPipeline, BaselineExtractor, DocumentInput
from app.provider import OpenAICompatibleProvider


def model_settings(**overrides) -> Settings:
    values = {
        "openai_api_key": "unit-test-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_model_extraction": True,
        "provider_max_attempts": 1,
        "model_chunk_max_chars": 35,
        "model_chunk_overlap_lines": 1,
        "model_max_chunks_per_document": 20,
    }
    values.update(overrides)
    return Settings(**values)


def completion(payload: dict) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(payload, ensure_ascii=False)}}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    })


class ChunkingTests(unittest.TestCase):
    def test_long_single_line_is_never_silently_dropped(self):
        document = DocumentInput(id="long", name="long.md", content="短行\n" + "潮" * 95 + "\n尾行")
        chunks = chunk_document(document, max_chars=32, overlap_lines=1)
        self.assertGreaterEqual(len(chunks), 5)
        self.assertTrue(all(len(chunk.content) <= 32 for chunk in chunks))
        long_line_chunks = [chunk for chunk in chunks if chunk.global_line_start == chunk.global_line_end == 2]
        self.assertEqual("潮" * 95, "".join(chunk.content for chunk in long_line_chunks))

    def test_model_chunks_use_global_lines_dedupe_overlap_and_isolate_invalid_chunk(self):
        prompts = []

        def handler(request: httpx.Request) -> httpx.Response:
            prompt = json.loads(request.content)["messages"][1]["content"]
            prompts.append(prompt)
            start, end = map(int, re.search(r"原文全局行 (\d+)-(\d+)", prompt).groups())
            records = []
            if start <= 2 <= end:
                records.append({"kind": "fact", "subject": "林澈", "predicate": "身份", "value": "领航员", "source_line_start": 2, "source_line_end": 2})
            if start == 3:
                records.append({"kind": "fact", "subject": "林澈", "predicate": "身份", "value": "守卫", "source_line_start": 3, "source_line_end": 3})
            if start == 4:
                records.append({"kind": "event", "description": "非法记录", "source_line_start": 4, "source_line_end": 4})
            return completion({"records": records})

        provider = OpenAICompatibleProvider(model_settings(), transport=httpx.MockTransport(handler))
        content = "\n".join(f"普通叙述段落{index}没有结构化句式。" for index in range(1, 8))
        result = AnalysisPipeline(extractor=ModelEnhancedExtractor(provider)).run([
            DocumentInput(id="doc", name="story.md", content=content)
        ])
        facts = [row for row in result.directives if row.kind == "fact"]
        self.assertEqual(2, len(facts))
        self.assertEqual({2, 3}, {row.evidence.line_start for row in facts})
        self.assertEqual("fact_conflict", result.issues[0].category.value)
        self.assertGreater(len(prompts), 2)
        self.assertTrue(all("原文全局行" in prompt for prompt in prompts))
        self.assertTrue(any("已安全跳过" in warning for warning in result.warnings))

    def test_chunk_limit_is_explicit_and_full_baseline_remains(self):
        provider = OpenAICompatibleProvider(
            model_settings(model_chunk_max_chars=32, model_max_chunks_per_document=1),
            transport=httpx.MockTransport(lambda _: completion({"records": []})),
        )
        content = "\n".join(["林澈的身份是领航员。", *[f"普通背景段落{index}没有状态。" for index in range(12)]])
        result = ModelEnhancedExtractor(provider).extract(DocumentInput(id="limit", name="limit.md", content=content))
        self.assertTrue(any("仅处理前 1/" in warning for warning in result.warnings))
        self.assertTrue(any(row.kind == "fact" and row.attrs["subject"] == "林澈" for row in result.directives))


class AliasAndRetrievalTests(unittest.TestCase):
    def test_explicit_alias_cross_document_conflict_and_trace(self):
        result = AnalysisPipeline(extractor=BaselineExtractor()).run([
            DocumentInput(id="world", name="world.md", content="林澈又名银羽。\n林澈的发色是银色。"),
            DocumentInput(id="chapter", name="chapter.md", content="银羽的发色是黑色。"),
        ])
        self.assertEqual("fact_conflict", result.issues[0].category.value)
        self.assertEqual("林澈", result.issues[0].metadata["subject"])
        self.assertIn("银羽的发色是黑色。", [span.text for span in result.issues[0].evidence])
        self.assertEqual("林澈", result.diagnostics["aliases"]["map"]["银羽"])
        self.assertGreaterEqual(result.diagnostics["aliases"]["trace_count"], 1)
        self.assertGreater(result.diagnostics["retrieval"]["candidate_count"], 0)
        self.assertGreater(result.diagnostics["retrieval"]["consumed_count"], 0)
        trace = next(row for row in result.diagnostics["retrieval"]["traces"] if row["consumed"])
        self.assertIn("shared_canonical_entity", trace["selected_reason"])

    def test_ambiguous_or_cyclic_aliases_are_not_merged(self):
        ambiguous = AnalysisPipeline(extractor=BaselineExtractor()).run([
            DocumentInput(id="a", name="a.md", content="林澈又名银羽。\n苏弦又名银羽。\n林澈的发色是银色。\n银羽的发色是黑色。")
        ])
        self.assertNotIn("银羽", ambiguous.diagnostics["aliases"]["map"])
        self.assertFalse(ambiguous.issues)
        self.assertTrue(any("多个主名" in warning for warning in ambiguous.warnings))
        cyclic = AnalysisPipeline(extractor=BaselineExtractor()).run([
            DocumentInput(id="c", name="c.md", content="林澈又名银羽。\n银羽又名林澈。")
        ])
        self.assertFalse(cyclic.diagnostics["aliases"]["map"])
        self.assertTrue(any("别名循环" in warning for warning in cyclic.warnings))


class LongTextSmokeTests(unittest.TestCase):
    def test_generated_twenty_thousand_character_fixture_has_exact_five_issues(self):
        root = Path(__file__).resolve().parents[1] / "data" / "long-text-smoke"
        paths = sorted(root.glob("*.md"))
        documents = [DocumentInput(id=path.stem, name=path.name, content=path.read_text(encoding="utf-8")) for path in paths]
        self.assertGreaterEqual(sum(len(document.content) for document in documents), 20_000)
        result = AnalysisPipeline(extractor=BaselineExtractor()).run(documents)
        self.assertEqual(
            {"fact_conflict", "location_collision", "knowledge_without_acquisition", "item_ownership", "world_rule_conflict"},
            {issue.category.value for issue in result.issues},
        )
        self.assertEqual(5, len(result.issues))
        sources = {document.name: document.content.splitlines() for document in documents}
        for issue in result.issues:
            for evidence in issue.evidence:
                self.assertEqual(sources[evidence.document_name][evidence.line_start - 1].strip(), evidence.text)
        self.assertGreaterEqual(result.diagnostics["chunking"]["total_chunks"], 4)
        self.assertGreaterEqual(result.diagnostics["aliases"]["trace_count"], 1)


if __name__ == "__main__":
    unittest.main()
