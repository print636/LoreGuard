import gzip
import json
import unittest

import httpx

from app.config import Settings
from app.provider import (
    NamedToolChoice,
    OpenAICompatibleProvider,
    ProviderError,
    ProviderRetryExhausted,
    ProviderToolCallError,
    RetryPolicy,
    ToolCallLimits,
    ToolDefinition,
)


def tool(name: str = "read_span") -> ToolDefinition:
    return ToolDefinition(
        name=name,
        description="读取已授权的证据片段。",
        parameters={
            "type": "object",
            "properties": {
                "document_ref": {"type": "string"},
                "line": {"type": "integer", "minimum": 1},
            },
            "required": ["document_ref", "line"],
            "additionalProperties": False,
        },
    )


def call(
    call_id: str = "call_safe_1",
    name: str = "read_span",
    arguments='{"document_ref":"d1","line":7}',
) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def completion(
    calls: list[dict] | None,
    *,
    content=None,
    finish_reason="tool_calls",
    usage=None,
    status=200,
) -> httpx.Response:
    message = {"content": content}
    if calls is not None:
        message["tool_calls"] = calls
    return httpx.Response(
        status,
        json={
            "choices": [{"message": message, "finish_reason": finish_reason}],
            "usage": usage
            if usage is not None
            else {"prompt_tokens": 11, "completion_tokens": 4},
        },
    )


class ProviderNativeToolCallingTests(unittest.TestCase):
    def settings(self, **overrides) -> Settings:
        values = {
            "_env_file": None,
            "enable_model_extraction": True,
            "openai_base_url": "https://tool-test.invalid/v1",
            "openai_model": "mock-tool-model",
            "openai_api_key": "unit-test-secret-key",
            "provider_timeout_seconds": 1,
            "provider_max_attempts": 2,
            "provider_thinking_mode": None,
            "provider_max_completion_tokens": None,
        }
        values.update(overrides)
        return Settings(**values)

    def provider(self, handler, *, settings=None, attempts=1):
        return OpenAICompatibleProvider(
            settings or self.settings(),
            transport=httpx.MockTransport(handler),
            retry_policy=RetryPolicy(
                max_attempts=attempts,
                base_delay_seconds=0,
                jitter_ratio=0,
            ),
            sleep=lambda _: None,
        )

    def test_success_sends_native_contract_and_returns_only_safe_structures(self):
        requests = []

        def handler(request):
            requests.append(json.loads(request.content))
            return completion([call()])

        provider = self.provider(
            handler,
            settings=self.settings(
                provider_max_completion_tokens=256,
                provider_thinking_mode="disabled",
            ),
        )
        result = provider.complete_with_tools(
            "private system prompt",
            "private user prompt",
            tools=[tool()],
        )

        self.assertEqual(1, len(requests))
        payload = requests[0]
        self.assertNotIn("response_format", payload)
        self.assertEqual("required", payload["tool_choice"])
        self.assertEqual("function", payload["tools"][0]["type"])
        self.assertEqual("read_span", payload["tools"][0]["function"]["name"])
        self.assertEqual(256, payload["max_tokens"])
        self.assertEqual({"type": "disabled"}, payload["thinking"])
        self.assertEqual(1, len(result.tool_calls))
        self.assertEqual("call_safe_1", result.tool_calls[0].id)
        self.assertEqual("read_span", result.tool_calls[0].name)
        self.assertEqual(
            {"document_ref": "d1", "line": 7}, result.tool_calls[0].arguments
        )
        self.assertNotIn("document_ref", repr(result.tool_calls[0]))
        self.assertEqual((11, 4), (result.prompt_tokens, result.completion_tokens))
        self.assertEqual("success", result.telemetry.category)
        expected_tool_chars = len(
            json.dumps(
                payload["tools"][0],
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
        self.assertEqual(
            len("private system prompt")
            + len("private user prompt")
            + expected_tool_chars,
            result.telemetry.input_chars,
        )

        # The envelope has no raw prompt/response fields.  Tool arguments are
        # still model-authored, untrusted content and must not be logged merely
        # because the transport wrapper is sanitized.
        serialized_transport_result = json.dumps(
            {
                "calls": [
                    {
                        "id": row.id,
                        "name": row.name,
                        "arguments": row.arguments,
                    }
                    for row in result.tool_calls
                ],
                "telemetry": result.telemetry.model_dump(),
            }
        )
        for forbidden in (
            "unit-test-secret-key",
            "tool-test.invalid",
            "private system prompt",
            "private user prompt",
        ):
            self.assertNotIn(forbidden, serialized_transport_result)

    def test_gzip_success_uses_the_same_bounded_reader_for_native_tools(self):
        raw = completion([call()]).content
        encoded = gzip.compress(raw)

        provider = self.provider(
            lambda _: httpx.Response(
                200,
                headers={
                    "Content-Encoding": "gzip",
                    "Content-Length": str(len(encoded)),
                },
                stream=httpx.ByteStream(encoded),
            )
        )
        result = provider.complete_with_tools("s", "u", tools=[tool()])

        self.assertEqual(1, len(result.tool_calls))
        self.assertEqual("read_span", result.tool_calls[0].name)
        self.assertEqual(len(encoded), result.telemetry.received_bytes)

    def test_encoded_representation_failure_is_availability_not_tool_contract(self):
        provider = self.provider(
            lambda _: httpx.Response(
                200,
                headers={"Content-Encoding": "br"},
                stream=httpx.ByteStream(b"private encoded response"),
            )
        )

        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete_with_tools("s", "u", tools=[tool()])

        self.assertIs(type(caught.exception), ProviderRetryExhausted)
        self.assertEqual(
            "unsupported_content_encoding", caught.exception.category
        )
        self.assertNotIn("private", str(caught.exception))

    def test_multiple_calls_and_object_arguments_are_supported_within_limits(self):
        response_calls = [
            {**call("call_1"), "index": 0},
            call(
                "call_2",
                "search_evidence",
                {"query": "moon gate", "top_k": 3},
            ),
        ]
        provider = self.provider(lambda _: completion(response_calls))
        result = provider.complete_with_tools(
            "s",
            "u",
            tools=[
                tool(),
                ToolDefinition(
                    "search_evidence",
                    "Search evidence.",
                    {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer"},
                        },
                    },
                ),
            ],
            limits=ToolCallLimits(max_calls=2),
        )

        self.assertEqual(("call_1", "call_2"), tuple(row.id for row in result.tool_calls))
        self.assertEqual(3, result.tool_calls[1].arguments["top_k"])

    def test_parallel_tool_calls_is_disabled_only_for_single_call_limit(self):
        payloads = []

        def single_handler(request):
            payloads.append(json.loads(request.content))
            return completion([call()])

        self.provider(single_handler).complete_with_tools(
            "s",
            "u",
            tools=[tool()],
            limits=ToolCallLimits(max_calls=1),
        )
        self.assertIs(payloads[0]["parallel_tool_calls"], False)

        def multiple_handler(request):
            payloads.append(json.loads(request.content))
            return completion([call("call_1"), call("call_2")])

        self.provider(multiple_handler).complete_with_tools(
            "s",
            "u",
            tools=[tool()],
            limits=ToolCallLimits(max_calls=2),
        )
        self.assertNotIn("parallel_tool_calls", payloads[1])

    def test_prompt_derived_arguments_are_untrusted_and_hidden_from_repr(self):
        echoed = "private user prompt copied by the model"
        provider = self.provider(
            lambda _: completion([call(arguments=json.dumps({"echo": echoed}))])
        )
        result = provider.complete_with_tools(
            "private system prompt",
            echoed,
            tools=[tool()],
        )

        # A native tool call can legitimately derive arguments from the prompt.
        # The caller receives those arguments for validation, but routine repr
        # output must not turn them into accidental log content.
        self.assertEqual(echoed, result.tool_calls[0].arguments["echo"])
        self.assertNotIn(echoed, repr(result.tool_calls[0]))
        self.assertNotIn(echoed, repr(result))

    def test_named_choice_is_sent_and_mismatched_response_fails_closed(self):
        payloads = []

        def handler(request):
            payloads.append(json.loads(request.content))
            return completion([call(name="other_tool")])

        provider = self.provider(handler)
        with self.assertRaises(ProviderToolCallError) as caught:
            provider.complete_with_tools(
                "s",
                "u",
                tools=[tool(), tool("other_tool")],
                tool_choice=NamedToolChoice("read_span"),
            )

        self.assertEqual(
            {"type": "function", "function": {"name": "read_span"}},
            payloads[0]["tool_choice"],
        )
        self.assertEqual("tool_choice_mismatch", caught.exception.category)

    def test_plain_text_without_tool_calls_is_never_disguised_as_success(self):
        calls = 0

        def handler(_):
            nonlocal calls
            calls += 1
            return completion(None, content="ordinary assistant text", finish_reason="stop")

        with self.assertRaises(ProviderToolCallError) as caught:
            self.provider(handler, attempts=2).complete_with_tools(
                "s", "u", tools=[tool()]
            )

        self.assertEqual(1, calls)
        self.assertEqual("tool_calls_missing", caught.exception.category)
        self.assertNotIn("ordinary assistant text", str(caught.exception))

    def test_unknown_tool_invalid_and_duplicate_ids_fail_closed(self):
        cases = (
            ("tool_call_unknown", [call(name="not_allowlisted")]),
            ("tool_call_id_invalid", [call(call_id="../../forged")]),
            ("tool_call_id_invalid", [call(call_id="")]),
            (
                "tool_call_id_duplicate",
                [call(call_id="call_same"), call(call_id="call_same")],
            ),
        )
        for category, response_calls in cases:
            with self.subTest(category=category):
                with self.assertRaises(ProviderToolCallError) as caught:
                    provider = self.provider(
                        lambda _, rows=response_calls: completion(rows)
                    )
                    provider.complete_with_tools("s", "u", tools=[tool()])
                self.assertEqual(category, caught.exception.category)
                self.assertEqual(category, caught.exception.telemetry.category)

    def test_malformed_arguments_fail_with_precise_safe_categories(self):
        cases = (
            ("tool_call_arguments_json", "not-json"),
            ("tool_call_arguments_json", '{"line":1,"line":2}'),
            ("tool_call_arguments_json", '{"line":NaN}'),
            ("tool_call_arguments_shape", "[]"),
            ("tool_call_arguments_shape", ["not", "an", "object"]),
            ("tool_call_arguments_shape", None),
        )
        for category, arguments in cases:
            with self.subTest(category=category, arguments=arguments):
                with self.assertRaises(ProviderToolCallError) as caught:
                    self.provider(
                        lambda _, value=arguments: completion(
                            [call(arguments=value)]
                        )
                    ).complete_with_tools("s", "u", tools=[tool()])
                self.assertEqual(category, caught.exception.category)
                self.assertNotIn("not-json", str(caught.exception))

    def test_call_and_argument_limits_are_enforced_after_bounded_read(self):
        with self.assertRaises(ProviderToolCallError) as caught:
            self.provider(
                lambda _: completion([call("call_1"), call("call_2")])
            ).complete_with_tools(
                "s",
                "u",
                tools=[tool()],
                limits=ToolCallLimits(max_calls=1),
            )
        self.assertEqual("tool_calls_too_many", caught.exception.category)

        with self.assertRaises(ProviderToolCallError) as caught:
            provider = self.provider(
                lambda _: completion([call(arguments='{"long":"value"}')])
            )
            provider.complete_with_tools(
                "s", "u", tools=[tool()], limits=ToolCallLimits(max_argument_bytes=4)
            )
        self.assertEqual("tool_call_arguments_too_large", caught.exception.category)

        with self.assertRaises(ProviderToolCallError) as caught:
            self.provider(
                lambda _: completion([call()]),
                settings=self.settings(provider_max_response_bytes=8),
            ).complete_with_tools("s", "u", tools=[tool()])
        self.assertEqual("response_too_large", caught.exception.category)
        self.assertEqual(0, caught.exception.telemetry.response_chars)

    def test_truncation_and_strict_outer_json_are_rejected(self):
        with self.assertRaises(ProviderToolCallError) as caught:
            self.provider(
                lambda _: completion([call()], finish_reason="length")
            ).complete_with_tools("s", "u", tools=[tool()])
        self.assertEqual("truncated", caught.exception.category)
        self.assertEqual(11, caught.exception.telemetry.prompt_tokens)

        duplicate_choices = (
            b'{"choices":[{"message":{"tool_calls":[]}}],'
            b'"choices":[],"usage":{}}'
        )
        with self.assertRaises(ProviderToolCallError) as caught:
            provider = self.provider(
                lambda _: httpx.Response(200, content=duplicate_choices)
            )
            provider.complete_with_tools("s", "u", tools=[tool()])
        self.assertEqual("tool_response_json", caught.exception.category)

    def test_429_retries_then_succeeds_and_timeout_exhaustion_stays_retryable(self):
        statuses = iter((429, 200))

        def recover(_):
            status = next(statuses)
            if status == 429:
                return httpx.Response(429, text="private throttle body")
            return completion([call()])

        result = self.provider(recover, attempts=2).complete_with_tools(
            "s", "u", tools=[tool()]
        )
        self.assertEqual(
            ["rate_limit", "success"],
            [row.category for row in result.telemetry.attempts],
        )

        timeout_calls = 0

        def timeout(request):
            nonlocal timeout_calls
            timeout_calls += 1
            raise httpx.ReadTimeout("private timeout detail", request=request)

        with self.assertRaises(ProviderRetryExhausted) as caught:
            self.provider(timeout, attempts=2).complete_with_tools(
                "s", "u", tools=[tool()]
            )
        self.assertEqual(2, timeout_calls)
        self.assertEqual("read_timeout", caught.exception.category)
        self.assertNotIn("private timeout detail", str(caught.exception))

    def test_rejected_tool_request_has_explicit_category_without_body_leak(self):
        provider = self.provider(
            lambda _: httpx.Response(400, text="provider-private unsupported detail")
        )
        with self.assertRaises(ProviderToolCallError) as caught:
            provider.complete_with_tools("s", "u", tools=[tool()])
        self.assertEqual("tool_request_rejected", caught.exception.category)
        self.assertEqual(400, caught.exception.http_status)
        self.assertNotIn("provider-private", str(caught.exception))

    def test_auth_failures_keep_the_existing_provider_error_classification(self):
        for status, category in ((401, "unauthorized"), (403, "forbidden")):
            with self.subTest(status=status):
                provider = self.provider(lambda _, code=status: httpx.Response(code))
                with self.assertRaises(ProviderError) as caught:
                    provider.complete_with_tools("s", "u", tools=[tool()])
                self.assertIs(type(caught.exception), ProviderError)
                self.assertEqual(category, caught.exception.category)

    def test_invalid_local_contracts_make_no_upstream_request(self):
        calls = 0

        def handler(_):
            nonlocal calls
            calls += 1
            return completion([call()])

        invalid_invocations = (
            lambda provider: provider.complete_with_tools("s", "u", tools=[]),
            lambda provider: provider.complete_with_tools(
                "s", "u", tools=[tool(), tool()]
            ),
            lambda provider: provider.complete_with_tools(
                "s",
                "u",
                tools=[ToolDefinition("bad name", "d", {"type": "object"})],
            ),
            lambda provider: provider.complete_with_tools(
                "s",
                "u",
                tools=[ToolDefinition("valid", "d", {"type": "array"})],
            ),
            lambda provider: provider.complete_with_tools(
                "s", "u", tools=[tool()], tool_choice=NamedToolChoice("missing")
            ),
            lambda provider: provider.complete_with_tools(
                "s", "u", tools=[tool()], tool_choice=NamedToolChoice([])  # type: ignore[arg-type]
            ),
            lambda provider: provider.complete_with_tools(
                "s", "u", tools=[tool()], tool_choice="auto"  # type: ignore[arg-type]
            ),
            lambda provider: provider.complete_with_tools(
                "s", "u", tools=[tool()], limits=ToolCallLimits(max_calls=0)
            ),
        )
        provider = self.provider(handler)
        for invoke in invalid_invocations:
            with self.subTest(invoke=invoke):
                with self.assertRaises(ProviderToolCallError):
                    invoke(provider)
        self.assertEqual(0, calls)


if __name__ == "__main__":
    unittest.main()
