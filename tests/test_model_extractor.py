import json
import unittest

import httpx

from app.config import Settings
from app.model_extractor import ModelEnhancedExtractor
from app.pipeline import AnalysisPipeline, DocumentInput
from app.provider import (
    OpenAICompatibleProvider,
    ProviderError,
    ProviderRetryExhausted,
    RetryPolicy,
)


def settings(**overrides) -> Settings:
    values = {
        "openai_api_key": "unit-test-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_model_extraction": True,
        "provider_timeout_seconds": 1,
        "provider_max_attempts": 3,
    }
    values.update(overrides)
    return Settings(**values)


def completion(content: str, status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        },
    )


class ProviderTests(unittest.TestCase):
    def test_success_and_usage(self):
        transport = httpx.MockTransport(lambda _: completion('{"records":[]}'))
        provider = OpenAICompatibleProvider(settings(), transport=transport)
        result = provider.complete("system", "user")
        self.assertEqual('{"records":[]}', result.text)
        self.assertEqual(12, result.prompt_tokens)
        self.assertEqual(8, result.completion_tokens)

    def test_429_and_5xx_are_retried_then_succeed(self):
        statuses = iter([429, 503, 200])

        def handler(_: httpx.Request) -> httpx.Response:
            status = next(statuses)
            return completion('{"records":[]}', status=status)

        provider = OpenAICompatibleProvider(
            settings(),
            transport=httpx.MockTransport(handler),
            retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=0),
            sleep=lambda _: None,
        )
        self.assertEqual('{"records":[]}', provider.complete("s", "u").text)

    def test_timeout_exhaustion_has_safe_error(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("simulated", request=request)

        provider = OpenAICompatibleProvider(
            settings(),
            transport=httpx.MockTransport(handler),
            retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0),
            sleep=lambda _: None,
        )
        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")
        self.assertEqual(2, calls)
        self.assertNotIn("unit-test-placeholder", str(caught.exception))

    def test_invalid_json_and_empty_content_exhaust_retries(self):
        for response in (
            httpx.Response(200, text="not-json"),
            completion(""),
            completion("not-json"),
        ):
            provider = OpenAICompatibleProvider(
                settings(),
                transport=httpx.MockTransport(lambda _, value=response: value),
                retry_policy=RetryPolicy(max_attempts=2, base_delay_seconds=0),
                sleep=lambda _: None,
            )
            with self.assertRaises(ProviderRetryExhausted):
                provider.complete("s", "u")

    def test_non_retryable_4xx_fails_immediately(self):
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return completion("{}", status=401)

        provider = OpenAICompatibleProvider(
            settings(), transport=httpx.MockTransport(handler), sleep=lambda _: None
        )
        with self.assertRaises(ProviderError):
            provider.complete("s", "u")
        self.assertEqual(1, calls)


class ModelExtractorTests(unittest.TestCase):
    def provider_for(self, payload: dict) -> OpenAICompatibleProvider:
        content = json.dumps(payload, ensure_ascii=False)
        return OpenAICompatibleProvider(
            settings(), transport=httpx.MockTransport(lambda _: completion(content))
        )

    def test_pipeline_opens_circuit_after_one_fully_failed_document(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("simulated", request=request)

        provider = OpenAICompatibleProvider(
            settings(
                model_circuit_breaker_failed_documents=1,
                model_chunk_max_chars=32,
                model_chunk_overlap_lines=0,
            ),
            transport=httpx.MockTransport(handler),
            retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0),
            sleep=lambda _: None,
        )
        result = AnalysisPipeline(extractor=ModelEnhancedExtractor(provider)).run(
            [
                DocumentInput(
                    id="first",
                    name="first.md",
                    content="林澈的身份是领航员。" + "这是一段没有额外状态的航海背景。" * 8,
                ),
                DocumentInput(id="second", name="second.md", content="苏弦的身份是档案官。"),
                DocumentInput(id="third", name="third.md", content="闻序的身份是观察员。"),
            ]
        )
        self.assertEqual(1, calls)
        self.assertFalse(result.model_used)
        self.assertEqual(3, len([row for row in result.directives if row.kind == "fact"]))
        self.assertTrue(any("运行级熔断已开启" in warning for warning in result.warnings))

    def test_validates_all_supported_model_kinds_and_binds_original_evidence(self):
        text = """林澈的身份是领航员。
1026-04-03 10:00，林澈抵达北港。
午后，林澈得知星门口令。
清晨，林澈说出了星门口令。
苏弦保管潮汐钥匙。
林澈使用潮汐钥匙开门。
星门只能由潮汐晶核驱动。
这一章里星门由普通火焰启动。"""
        records = [
            {"kind": "fact", "subject": "林澈", "predicate": "身份", "value": "领航员", "source_line_start": 1, "source_line_end": 1},
            {"kind": "event", "time": "1026-04-03 10:00", "location": "北港", "participants": ["林澈"], "source_line_start": 2, "source_line_end": 2},
            {"kind": "knows", "character": "林澈", "fact": "星门口令", "time": "午后", "source_line_start": 3, "source_line_end": 3},
            {"kind": "claims_knows", "character": "林澈", "fact": "星门口令", "time": "清晨", "source_line_start": 4, "source_line_end": 4},
            {"kind": "item", "item": "潮汐钥匙", "owner": "苏弦", "time": "未注明", "source_line_start": 5, "source_line_end": 5},
            {"kind": "uses", "item": "潮汐钥匙", "user": "林澈", "time": "未注明", "source_line_start": 6, "source_line_end": 6},
            {"kind": "world_rule", "key": "星门能源", "value": "潮汐晶核", "source_line_start": 7, "source_line_end": 7},
            {"kind": "world_assert", "key": "星门能源", "value": "普通火焰", "source_line_start": 8, "source_line_end": 8},
        ]
        result = ModelEnhancedExtractor(self.provider_for({"records": records})).extract(
            DocumentInput(id="doc", name="chapter.md", content=text)
        )
        self.assertEqual(
            {"fact", "event", "knows", "claims_knows", "item", "uses", "world_rule", "world_assert"},
            {item.kind for item in result.directives},
        )
        self.assertTrue(result.model_used)
        self.assertEqual(12, result.prompt_tokens)
        knowledge = next(item for item in result.directives if item.kind == "knows")
        self.assertEqual("午后，林澈得知星门口令。", knowledge.evidence.text)

    def test_model_and_baseline_duplicates_are_merged(self):
        payload = {
            "records": [
                {
                    "kind": "fact",
                    "subject": "林澈",
                    "predicate": "发色",
                    "value": "银色",
                    "source_line_start": 1,
                    "source_line_end": 1,
                }
            ]
        }
        result = ModelEnhancedExtractor(self.provider_for(payload)).extract(
            DocumentInput(id="doc", name="chapter.md", content="林澈的发色是银色。")
        )
        self.assertEqual(1, len(result.directives))

    def test_boolean_permission_fields_are_valid_and_normalized(self):
        payload = {
            "records": [
                {
                    "kind": "fact",
                    "subject": "祁霁",
                    "predicate": "mobility_permission",
                    "value": "instant_transport",
                    "origin": "沉钟港",
                    "destination": "月沫城",
                    "bidirectional": True,
                    "status": "active",
                    "current": True,
                    "source_line_start": 1,
                    "source_line_end": 1,
                }
            ]
        }
        result = ModelEnhancedExtractor(self.provider_for(payload)).extract(
            DocumentInput(
                id="permit",
                name="permit.md",
                content="议会向祁霁签发瞬时通行许可，路线为沉钟港至月沫城，当前有效。",
            )
        )
        permission = next(
            row for row in result.directives
            if row.kind == "fact" and row.attrs.get("predicate") == "mobility_permission"
        )
        self.assertEqual("true", permission.attrs["bidirectional"])
        self.assertEqual("true", permission.attrs["current"])
        self.assertFalse(any("schema_validation" in warning for warning in result.warnings))

    def test_model_fact_inherits_explicit_source_time(self):
        payload = {
            "records": [
                {"kind": "fact", "subject": "叶峤", "predicate": "行动能力", "value": "受限", "source_line_start": 1, "source_line_end": 1},
                {"kind": "fact", "subject": "叶峤", "predicate": "行动能力", "value": "恢复", "source_line_start": 2, "source_line_end": 2},
            ]
        }
        result = AnalysisPipeline(
            extractor=ModelEnhancedExtractor(self.provider_for(payload))
        ).run([
            DocumentInput(
                id="state",
                name="state.md",
                content=(
                    "1027-01-01 08:00，叶峤的行动能力是受限。\n"
                    "1027-02-01 08:00，叶峤的行动能力是恢复。"
                ),
            )
        ])
        self.assertFalse(any(row.category.value == "fact_conflict" for row in result.issues))
        self.assertEqual(
            ["1027-01-01 08:00", "1027-02-01 08:00"],
            sorted({row.attrs.get("time") for row in result.directives if row.kind == "fact"}),
        )

    def test_use_wording_cannot_create_a_fake_item_owner(self):
        payload = {
            "records": [
                {"kind": "item", "item": "潮汐钥", "owner": "莫行", "source_line_start": 1, "source_line_end": 1},
                {"kind": "uses", "item": "潮汐钥", "user": "莫行", "source_line_start": 1, "source_line_end": 1},
            ]
        }
        result = ModelEnhancedExtractor(self.provider_for(payload)).extract(
            DocumentInput(
                id="chapter",
                name="chapter.md",
                content="1044-06-18 09:30，莫行取出潮汐钥，并按下启动机关。",
            )
        )
        self.assertFalse(any(row.kind == "item" for row in result.directives))
        self.assertTrue(any(row.kind == "uses" for row in result.directives))
        self.assertTrue(any("lexical_support" in warning for warning in result.warnings))

    def test_model_use_record_requires_positive_lexical_support(self):
        payload = {
            "records": [
                {
                    "kind": "item",
                    "item": "白潮印",
                    "owner": "苏弦",
                    "source_line_start": 1,
                    "source_line_end": 1,
                },
                {
                    "kind": "uses",
                    "item": "白潮印",
                    "user": "任何人",
                    "source_line_start": 1,
                    "source_line_end": 1,
                },
            ]
        }
        result = ModelEnhancedExtractor(self.provider_for(payload)).extract(
            DocumentInput(
                id="permissions",
                name="world.md",
                content="白潮印由苏弦保管，任何人都没有使用权。",
            )
        )
        self.assertTrue(any(row.kind == "item" for row in result.directives))
        self.assertFalse(any(row.kind == "uses" for row in result.directives))
        self.assertTrue(any("不符合抽取协议" in warning for warning in result.warnings))

    def test_invalid_structure_or_line_range_falls_back_to_baseline(self):
        payloads = [
            {"records": [{"kind": "unknown", "source_line_start": 1, "source_line_end": 1}]},
            {"records": [{"kind": "fact", "subject": "林澈", "predicate": "发色", "value": "银色", "source_line_start": 99, "source_line_end": 99}]},
        ]
        for payload in payloads:
            result = ModelEnhancedExtractor(self.provider_for(payload)).extract(
                DocumentInput(id="doc", name="chapter.md", content="林澈的发色是银色。")
            )
            self.assertEqual(1, len(result.directives))
            self.assertFalse(result.model_used)
            self.assertTrue(any("已降级" in warning for warning in result.warnings))

    def test_partial_valid_records_survive_invalid_description_record(self):
        payload = {
            "records": [
                {
                    "kind": "fact",
                    "subject": "林澈",
                    "predicate": "身份",
                    "value": "领航员",
                    "source_line_start": 1,
                    "source_line_end": 1,
                },
                {
                    "kind": "event",
                    "description": "林澈抵达北港",
                    "source_line_start": 2,
                    "source_line_end": 2,
                },
            ]
        }
        result = ModelEnhancedExtractor(self.provider_for(payload)).extract(
            DocumentInput(
                id="doc",
                name="chapter.md",
                content="设定稿称林澈是领航员。\n林澈沿石阶抵达北港。",
            )
        )
        self.assertTrue(result.model_used)
        self.assertEqual(1, len(result.directives))
        self.assertEqual("fact", result.directives[0].kind)
        self.assertTrue(any("模型记录 #2" in warning for warning in result.warnings))
        self.assertFalse(any("已降级" in warning for warning in result.warnings))

    def test_all_invalid_records_fall_back_without_losing_baseline(self):
        payload = {
            "records": [
                {
                    "kind": "world_rule",
                    "description": "星门需要晶核",
                    "source_line_start": 1,
                    "source_line_end": 1,
                },
                {
                    "kind": "item",
                    "description": "苏弦拿着钥匙",
                    "source_line_start": 1,
                    "source_line_end": 1,
                },
            ]
        }
        result = ModelEnhancedExtractor(self.provider_for(payload)).extract(
            DocumentInput(id="doc", name="chapter.md", content="林澈的发色是银色。")
        )
        self.assertFalse(result.model_used)
        self.assertEqual(1, len(result.directives))
        self.assertEqual("fact", result.directives[0].kind)
        self.assertEqual(2, len([w for w in result.warnings if "安全跳过" in w]))
        self.assertTrue(any("已降级" in warning for warning in result.warnings))

    def test_untimed_item_and_use_are_valid_canonical_states(self):
        payload = {
            "records": [
                {"kind": "item", "item": "潮汐钥匙", "owner": "苏弦", "source_line_start": 1, "source_line_end": 1},
                {"kind": "uses", "item": "潮汐钥匙", "user": "林澈", "source_line_start": 2, "source_line_end": 2},
            ]
        }
        pipeline = AnalysisPipeline(
            extractor=ModelEnhancedExtractor(self.provider_for(payload))
        )
        result = pipeline.run(
            [
                DocumentInput(
                    id="doc",
                    name="chapter.md",
                    content="潮汐钥匙一直由苏弦保管。\n林澈用潮汐钥匙打开石门。",
                )
            ]
        )
        self.assertEqual("", result.directives[0].attrs["time"])
        self.assertEqual("", result.directives[1].attrs["time"])
        self.assertEqual("item_ownership", result.issues[0].category.value)

    def test_no_key_keeps_baseline_without_http_request(self):
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return completion('{"records":[]}')

        provider = OpenAICompatibleProvider(
            settings(openai_api_key=""), transport=httpx.MockTransport(handler)
        )
        result = ModelEnhancedExtractor(provider).extract(
            DocumentInput(id="doc", name="chapter.md", content="林澈的发色是银色。")
        )
        self.assertEqual(0, calls)
        self.assertEqual(1, len(result.directives))

    def test_model_extractor_is_consumed_by_analysis_pipeline(self):
        payload = {
            "records": [
                {"kind": "fact", "subject": "林澈", "predicate": "发色", "value": "银色", "source_line_start": 1, "source_line_end": 1},
                {"kind": "fact", "subject": "林澈", "predicate": "发色", "value": "黑色", "source_line_start": 2, "source_line_end": 2},
            ]
        }
        pipeline = AnalysisPipeline(
            extractor=ModelEnhancedExtractor(self.provider_for(payload))
        )
        result = pipeline.run(
            [
                DocumentInput(
                    id="doc",
                    name="chapter.md",
                    content="角色设定稿记载林澈天生银发。\n同一版本的另一份设定稿记载林澈天生黑发。",
                )
            ]
        )
        self.assertTrue(result.model_used)
        self.assertEqual(12, result.prompt_tokens)
        self.assertEqual(2, len(result.directives))
        self.assertEqual("fact_conflict", result.issues[0].category.value)


if __name__ == "__main__":
    unittest.main()
