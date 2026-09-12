import gzip
import json
import unittest

import httpx

from app.config import Settings
from app.provider import (
    OpenAICompatibleProvider,
    ProviderError,
    ProviderRetryExhausted,
    RetryPolicy,
)
from scripts.check_provider import check_provider


class TrackingStream(httpx.SyncByteStream):
    def __init__(self, chunks, *, error=None):
        self.chunks = list(chunks)
        self.error = error
        self.yielded = 0
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            self.yielded += 1
            yield chunk
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


class ProviderProbeTests(unittest.TestCase):
    def provider(self, handler):
        settings = Settings(_env_file=None, enable_model_extraction=True,
            openai_base_url="https://probe.invalid/v1", openai_model="mock-model",
            openai_api_key="unit-test-private-value", provider_thinking_mode=None,
            provider_max_completion_tokens=None)
        return OpenAICompatibleProvider(settings, transport=httpx.MockTransport(handler),
            retry_policy=RetryPolicy(max_attempts=1), sleep=lambda _: None)

    def test_probe_uses_production_json_contract(self):
        def handler(request):
            self.assertEqual({"type": "json_object"}, json.loads(request.content)["response_format"])
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"status":"ok"}'}}]})
        report, code = check_provider(self.provider(handler))
        self.assertEqual(0, code)
        self.assertTrue(report["json_contract_ok"])
        self.assertEqual(
            {"configured": False, "mode": None}, report["thinking"]
        )

    def test_probe_does_not_echo_rejected_body(self):
        handler = lambda _: httpx.Response(403, text="unit-test-private-value")
        report, code = check_provider(self.provider(handler))
        self.assertEqual(1, code)
        self.assertEqual(403, report["http_status"])
        self.assertNotIn("unit-test-private-value", json.dumps(report))

    def test_probe_reports_only_allowlisted_thinking_mode(self):
        payloads = []

        def handler(request):
            payloads.append(json.loads(request.content))
            return httpx.Response(200, json={
                "choices": [{"message": {"content": '{"status":"ok"}'}}]
            })

        provider = self.provider(handler)
        provider.settings.provider_thinking_mode = "enabled"
        report, code = check_provider(provider)

        self.assertEqual(0, code)
        self.assertEqual(
            {"configured": True, "mode": "enabled"}, report["thinking"]
        )
        self.assertEqual({"type": "enabled"}, payloads[0]["thinking"])
        serialized = json.dumps(report)
        self.assertNotIn("unit-test-private-value", serialized)
        self.assertNotIn("probe.invalid", serialized)

    def test_provider_rejects_redirect_without_second_request(self):
        calls = []
        def handler(request):
            calls.append(request.url.host)
            return httpx.Response(307, headers={"location": "https://other.invalid/collect"})
        with self.assertRaises(ProviderError):
            self.provider(handler).complete("s", "u")
        self.assertEqual(["probe.invalid"], calls)

    def test_invalid_usage_never_leaks_provider_controlled_value(self):
        def handler(_):
            return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": "unit-test-private-value"}})
        with self.assertRaises(ProviderRetryExhausted) as caught:
            self.provider(handler).complete("s", "u")
        self.assertNotIn("unit-test-private-value", str(caught.exception))
        self.assertEqual("usage_shape", caught.exception.category)

    def test_unexpected_json_is_not_success(self):
        handler = lambda _: httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
        report, code = check_provider(self.provider(handler))
        self.assertEqual(1, code)
        self.assertEqual("UnexpectedJSON", report["error_type"])


class ProviderHardeningTests(unittest.TestCase):
    def settings(self, **overrides):
        values = {
            "_env_file": None,
            "enable_model_extraction": True,
            "openai_base_url": "https://anonymous.invalid/v1",
            "openai_model": "mock-model",
            "openai_api_key": "anonymous-unit-test-key",
            "provider_timeout_seconds": 1,
            "provider_max_attempts": 3,
            "provider_thinking_mode": None,
            "provider_max_completion_tokens": None,
        }
        values.update(overrides)
        return Settings(**values)

    @staticmethod
    def success(*, finish_reason="stop"):
        return httpx.Response(200, json={
            "choices": [{
                "message": {"content": '{"records":[]}'},
                "finish_reason": finish_reason,
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        })

    def provider(self, handler, **kwargs):
        retry_policy = kwargs.pop(
            "retry_policy",
            RetryPolicy(max_attempts=3, base_delay_seconds=0, jitter_ratio=0),
        )
        return OpenAICompatibleProvider(
            kwargs.pop("settings", self.settings()),
            transport=httpx.MockTransport(handler),
            retry_policy=retry_policy,
            sleep=kwargs.pop("sleep", lambda _: None),
            **kwargs,
        )

    @staticmethod
    def success_bytes():
        return json.dumps({
            "choices": [{
                "message": {"content": '{"records":[]}'},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        }, separators=(",", ":")).encode("utf-8")

    def test_content_length_over_cap_rejects_without_reading(self):
        stream = TrackingStream([self.success_bytes()])
        provider = self.provider(
            lambda _: httpx.Response(
                200,
                headers={"Content-Length": "9"},
                stream=stream,
            ),
            settings=self.settings(provider_max_response_bytes=8),
        )

        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")

        self.assertEqual("response_too_large", caught.exception.category)
        self.assertEqual(0, stream.yielded)
        self.assertTrue(stream.closed)
        self.assertEqual(0, caught.exception.telemetry.received_bytes)
        self.assertEqual(0, caught.exception.telemetry.response_chars)
        self.assertEqual(1, len(caught.exception.telemetry.attempts))

    def test_chunked_response_stops_on_first_crossing_chunk(self):
        stream = TrackingStream([b"123", b"456", b"must-not-be-read"])
        provider = self.provider(
            lambda _: httpx.Response(200, stream=stream),
            settings=self.settings(provider_max_response_bytes=5),
        )

        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")

        self.assertEqual("response_too_large", caught.exception.category)
        self.assertEqual(2, stream.yielded)
        self.assertTrue(stream.closed)
        self.assertEqual(6, caught.exception.telemetry.received_bytes)
        self.assertEqual(0, caught.exception.telemetry.response_chars)

    def test_response_exactly_at_byte_boundary_succeeds(self):
        body = self.success_bytes()
        stream = TrackingStream([body[:11], body[11:]])
        provider = self.provider(
            lambda _: httpx.Response(
                200,
                headers={"Content-Length": str(len(body))},
                stream=stream,
            ),
            settings=self.settings(provider_max_response_bytes=len(body)),
        )

        result = provider.complete("s", "u")

        self.assertEqual('{"records":[]}', result.text)
        self.assertEqual(len(body), result.telemetry.received_bytes)
        self.assertTrue(stream.closed)

    def test_gzip_success_is_decoded_across_wire_chunk_boundaries(self):
        body = self.success_bytes()
        compressed = gzip.compress(body)
        stream = TrackingStream(
            [compressed[:1], compressed[1:7], compressed[7:19], compressed[19:]]
        )
        observed_accept_encoding = []

        def handler(request):
            observed_accept_encoding.append(request.headers["accept-encoding"])
            return httpx.Response(
                200,
                headers={
                    "Content-Encoding": "GZip",
                    "Content-Length": str(len(compressed)),
                },
                stream=stream,
            )

        provider = self.provider(
            handler,
            settings=self.settings(
                provider_max_response_bytes=max(len(body), len(compressed))
            ),
        )

        result = provider.complete("s", "u")

        self.assertEqual('{"records":[]}', result.text)
        self.assertEqual(["gzip"], observed_accept_encoding)
        self.assertEqual(len(compressed), result.telemetry.received_bytes)
        self.assertEqual(len(body.decode("utf-8")), result.telemetry.response_chars)
        self.assertEqual(4, stream.yielded)
        self.assertTrue(stream.closed)

    def test_gzip_decoded_body_cannot_expand_beyond_response_cap(self):
        compressed = gzip.compress(b"private-expanded-body" * 1_000)
        self.assertLess(len(compressed), 256)
        stream = TrackingStream([compressed])
        provider = self.provider(
            lambda _: httpx.Response(
                200,
                headers={
                    "Content-Encoding": "gzip",
                    "Content-Length": str(len(compressed)),
                },
                stream=stream,
            ),
            settings=self.settings(provider_max_response_bytes=256),
        )

        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")

        self.assertEqual("response_too_large", caught.exception.category)
        self.assertEqual(len(compressed), caught.exception.telemetry.received_bytes)
        self.assertEqual(0, caught.exception.telemetry.response_chars)
        self.assertNotIn("private-expanded-body", str(caught.exception))
        self.assertTrue(stream.closed)

    def test_preconsumed_gzip_response_fails_closed_without_body_access(self):
        encoded = gzip.compress(self.success_bytes())
        provider = self.provider(
            lambda _: httpx.Response(
                200,
                headers={"Content-Encoding": "gzip"},
                content=encoded,
            )
        )

        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")

        self.assertEqual("response_decompression", caught.exception.category)
        self.assertEqual(0, caught.exception.telemetry.received_bytes)
        self.assertEqual(0, caught.exception.telemetry.response_chars)

    def test_gzip_wire_bytes_are_bounded_before_decompression(self):
        encoded = gzip.compress(b"{}")
        wire_cap = len(encoded) - 1
        stream = TrackingStream([encoded[:wire_cap], encoded[wire_cap:]])
        provider = self.provider(
            lambda _: httpx.Response(
                200,
                headers={"Content-Encoding": "gzip"},
                stream=stream,
            ),
            settings=self.settings(provider_max_response_bytes=wire_cap),
        )

        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")

        self.assertEqual("response_too_large", caught.exception.category)
        self.assertEqual(len(encoded), caught.exception.telemetry.received_bytes)
        self.assertEqual(2, stream.yielded)
        self.assertTrue(stream.closed)

    def test_gzip_corruption_truncation_and_trailing_bytes_fail_closed(self):
        valid = gzip.compress(self.success_bytes())
        crc_corrupt = bytearray(valid)
        crc_corrupt[-8] ^= 0xFF
        cases = (
            ("invalid_header", [b"not-a-gzip-stream-private"]),
            ("truncated_trailer", [valid[:-3]]),
            ("crc_mismatch", [bytes(crc_corrupt)]),
            ("trailing_garbage_same_chunk", [valid + b"private trailing bytes"]),
            ("trailing_garbage_next_chunk", [valid, b"private trailing bytes"]),
            (
                "second_member_same_chunk",
                [valid + gzip.compress(b"private second member")],
            ),
            (
                "second_member_at_chunk_boundary",
                [valid, gzip.compress(b"private second member")],
            ),
        )
        for name, chunks in cases:
            with self.subTest(name=name):
                stream = TrackingStream(chunks)
                provider = self.provider(
                    lambda _, candidate=stream: httpx.Response(
                        200,
                        headers={"Content-Encoding": "gzip"},
                        stream=candidate,
                    ),
                    settings=self.settings(provider_max_response_bytes=4_096),
                )

                with self.assertRaises(ProviderRetryExhausted) as caught:
                    provider.complete("s", "u")

                self.assertEqual(
                    "response_decompression", caught.exception.category
                )
                self.assertEqual(0, caught.exception.telemetry.response_chars)
                self.assertNotIn("private", str(caught.exception))
                self.assertTrue(stream.closed)

    def test_unknown_or_multiple_content_encodings_are_rejected_without_read(self):
        cases = (
            [(b"Content-Encoding", b"br")],
            [(b"Content-Encoding", b"deflate")],
            [(b"Content-Encoding", b"zstd")],
            [(b"Content-Encoding", b"private-coding")],
            [(b"Content-Encoding", b"")],
            [(b"Content-Encoding", b"gzip, br")],
            [
                (b"Content-Encoding", b"gzip"),
                (b"Content-Encoding", b"gzip"),
            ],
            [
                (b"Content-Encoding", b"gzip"),
                (b"Content-Encoding", b"identity"),
            ],
        )
        for headers in cases:
            with self.subTest(header_count=len(headers)):
                stream = TrackingStream([b"private body must not be consumed"])
                provider = self.provider(
                    lambda _, values=headers, candidate=stream: httpx.Response(
                        200, headers=values, stream=candidate
                    )
                )

                with self.assertRaises(ProviderRetryExhausted) as caught:
                    provider.complete("s", "u")

                self.assertEqual(
                    "unsupported_content_encoding", caught.exception.category
                )
                self.assertEqual(0, stream.yielded)
                self.assertEqual(0, caught.exception.telemetry.received_bytes)
                self.assertNotIn("private", str(caught.exception))
                self.assertTrue(stream.closed)

    def test_none_response_cap_still_has_an_absolute_wire_ceiling(self):
        stream = TrackingStream([self.success_bytes()])
        provider = self.provider(
            lambda _: httpx.Response(
                200,
                headers={"Content-Length": str(16 * 1_024 * 1_024 + 1)},
                stream=stream,
            ),
            settings=self.settings(provider_max_response_bytes=None),
        )

        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")

        self.assertEqual("response_too_large", caught.exception.category)
        self.assertEqual(0, stream.yielded)
        self.assertEqual(0, caught.exception.telemetry.received_bytes)
        self.assertTrue(stream.closed)

    def test_untrusted_content_length_cannot_bypass_stream_limit(self):
        stream = TrackingStream([b"1234", b"56", b"must-not-be-read"])
        provider = self.provider(
            lambda _: httpx.Response(
                200,
                headers={"Content-Length": "4-not-really"},
                stream=stream,
            ),
            settings=self.settings(provider_max_response_bytes=5),
        )

        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")

        self.assertEqual("response_too_large", caught.exception.category)
        self.assertEqual(2, stream.yielded)
        self.assertEqual(6, caught.exception.telemetry.received_bytes)
        self.assertNotIn(
            "4-not-really", json.dumps(caught.exception.telemetry.model_dump())
        )

    def test_response_limit_counts_utf8_bytes_not_characters(self):
        body = json.dumps({
            "choices": [{
                "message": {"content": '{"地点":"山门"}'},
                "finish_reason": "stop",
            }],
        }, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        character_count = len(body.decode("utf-8"))
        self.assertGreater(len(body), character_count)
        stream = TrackingStream([body])
        provider = self.provider(
            lambda _: httpx.Response(200, stream=stream),
            settings=self.settings(provider_max_response_bytes=character_count),
        )

        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")

        self.assertEqual("response_too_large", caught.exception.category)
        self.assertEqual(len(body), caught.exception.telemetry.received_bytes)
        self.assertEqual(0, caught.exception.telemetry.response_chars)

    def test_large_http_error_body_is_never_read(self):
        stream = TrackingStream([b"private" * 100_000])
        provider = self.provider(lambda _: httpx.Response(
            403,
            headers={
                "X-Request-ID": "safe-error-request",
                # Encoded HTTP error bodies must never enter the success-body
                # reader or decompressor.
                "Content-Encoding": "gzip",
            },
            stream=stream,
        ))

        with self.assertRaises(ProviderError) as caught:
            provider.complete("s", "u")

        self.assertEqual("forbidden", caught.exception.category)
        self.assertEqual("safe-error-request", caught.exception.request_id)
        self.assertEqual(0, stream.yielded)
        self.assertTrue(stream.closed)
        self.assertEqual(0, caught.exception.telemetry.received_bytes)

    def test_stream_transport_failure_discards_body_and_closes_connection(self):
        streams = []

        def handler(_):
            stream = TrackingStream(
                [b"private partial body"],
                error=httpx.ReadError("private connection detail"),
            )
            streams.append(stream)
            return httpx.Response(200, stream=stream)

        provider = self.provider(
            handler,
            retry_policy=RetryPolicy(
                max_attempts=2, base_delay_seconds=0, jitter_ratio=0
            ),
        )
        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")

        self.assertEqual("transport", caught.exception.category)
        self.assertEqual(2, len(streams))
        self.assertTrue(all(stream.closed for stream in streams))
        self.assertEqual(
            2 * len(b"private partial body"),
            caught.exception.telemetry.received_bytes,
        )
        self.assertEqual(0, caught.exception.telemetry.response_chars)
        self.assertNotIn(
            "private partial body",
            json.dumps(caught.exception.telemetry.model_dump()),
        )

    def test_stream_decoding_and_generic_request_errors_have_safe_categories(self):
        cases = (
            (
                "response_decompression",
                httpx.DecodingError("private decoding detail"),
                False,
            ),
            ("transport", httpx.RequestError("private request detail"), True),
        )
        for expected, error, retryable in cases:
            with self.subTest(category=expected):
                streams = []

                def handler(_):
                    stream = TrackingStream([b"private partial body"], error=error)
                    streams.append(stream)
                    return httpx.Response(200, stream=stream)

                attempts = 2 if retryable else 1
                provider = self.provider(
                    handler,
                    retry_policy=RetryPolicy(
                        max_attempts=attempts,
                        base_delay_seconds=0,
                        jitter_ratio=0,
                    ),
                )
                with self.assertRaises(ProviderRetryExhausted) as caught:
                    provider.complete("s", "u")

                self.assertEqual(expected, caught.exception.category)
                self.assertEqual(attempts, len(streams))
                self.assertTrue(all(stream.closed for stream in streams))
                self.assertEqual(0, caught.exception.telemetry.response_chars)
                serialized = json.dumps(caught.exception.telemetry.model_dump())
                self.assertNotIn("private", serialized)
                self.assertNotIn("private", str(caught.exception))

    def test_decoding_error_raised_before_response_context_is_sanitized(self):
        def handler(_):
            raise httpx.DecodingError("private pre-response decoding detail")

        with self.assertRaises(ProviderRetryExhausted) as caught:
            self.provider(handler).complete("s", "u")

        self.assertEqual("response_decompression", caught.exception.category)
        self.assertEqual(1, caught.exception.attempt_no)
        self.assertEqual(0, caught.exception.telemetry.received_bytes)
        self.assertEqual(0, caught.exception.telemetry.response_chars)
        self.assertNotIn("private", str(caught.exception))

    def test_response_failure_categories_are_structured_and_not_retried(self):
        cases = (
            ("body_json", lambda: httpx.Response(200, text="not-json")),
            ("response_shape", lambda: httpx.Response(200, json=[])),
            ("response_shape", lambda: httpx.Response(200, json={"choices": []})),
            ("empty_content", lambda: httpx.Response(200, json={
                "choices": [{"message": {"content": ""}}],
            })),
            ("content_json", lambda: httpx.Response(200, json={
                "choices": [{"message": {"content": "not-json"}}],
            })),
            ("usage_shape", lambda: httpx.Response(200, json={
                "choices": [{"message": {"content": "{}"}}], "usage": [],
            })),
        )
        for expected_category, response_factory in cases:
            with self.subTest(category=expected_category):
                calls = 0

                def handler(_):
                    nonlocal calls
                    calls += 1
                    return response_factory()

                with self.assertRaises(ProviderRetryExhausted) as caught:
                    self.provider(handler).complete("system", "user")
                self.assertEqual(1, calls)
                self.assertEqual(expected_category, caught.exception.category)
                self.assertIsNone(caught.exception.http_status)
                self.assertEqual(1, caught.exception.attempt_no)
                self.assertGreaterEqual(caught.exception.elapsed_ms, 0)

    def test_timeout_transport_and_http_categories(self):
        exception_cases = (
            ("connect_timeout", httpx.ConnectTimeout),
            ("read_timeout", httpx.ReadTimeout),
            ("transport", httpx.ConnectError),
        )
        for expected_category, exception_type in exception_cases:
            with self.subTest(category=expected_category):
                calls = 0

                def handler(request):
                    nonlocal calls
                    calls += 1
                    raise exception_type("anonymous failure", request=request)

                with self.assertRaises(ProviderRetryExhausted) as caught:
                    self.provider(handler).complete("s", "u")
                self.assertEqual(3, calls)
                self.assertEqual(expected_category, caught.exception.category)
                self.assertEqual(3, caught.exception.attempt_no)

        for status, category, error_type in (
            (429, "rate_limit", ProviderRetryExhausted),
            (503, "upstream_5xx", ProviderRetryExhausted),
            (401, "unauthorized", ProviderError),
            (403, "forbidden", ProviderError),
        ):
            with self.subTest(status=status):
                calls = 0

                def handler(_, response_status=status):
                    nonlocal calls
                    calls += 1
                    return httpx.Response(response_status, text="anonymous body")

                with self.assertRaises(error_type) as caught:
                    self.provider(handler).complete("s", "u")
                self.assertEqual(1 if status in {401, 403} else 3, calls)
                self.assertEqual(category, caught.exception.category)
                self.assertEqual(status, caught.exception.http_status)
                self.assertIn(f"HTTP {status}", str(caught.exception))
                self.assertNotIn("anonymous body", str(caught.exception))

    def test_correlation_id_is_allowlisted_sanitized_and_propagated(self):
        provider = self.provider(lambda _: httpx.Response(
            403,
            text="private response body",
            headers={
                "X-Request-ID": "request-safe_123",
                "Authorization": "private response credential",
                "Set-Cookie": "private-cookie=true",
            },
        ))
        with self.assertRaises(ProviderError) as caught:
            provider.complete("s", "u")
        error = caught.exception
        self.assertEqual("request-safe_123", error.request_id)
        self.assertEqual("request-safe_123", error.telemetry.request_id)
        self.assertEqual("request-safe_123", error.telemetry.attempts[0].request_id)
        serialized = json.dumps(error.telemetry.model_dump())
        for forbidden in (
            "private response body",
            "private response credential",
            "private-cookie",
            "anonymous.invalid",
        ):
            self.assertNotIn(forbidden, serialized)

        malicious = "A" * 129
        provider = self.provider(lambda _: httpx.Response(
            401,
            headers={"x-request-id": malicious, "trace-id": "trace-safe/456"},
        ))
        with self.assertRaises(ProviderError) as caught:
            provider.complete("s", "u")
        self.assertEqual("trace-safe/456", caught.exception.request_id)
        self.assertNotIn(malicious, json.dumps(caught.exception.telemetry.model_dump()))

        provider = self.provider(lambda _: httpx.Response(403))
        with self.assertRaises(ProviderError) as caught:
            provider.complete("s", "u")
        self.assertIsNone(caught.exception.request_id)
        self.assertIsNone(caught.exception.telemetry.request_id)

    def test_success_and_failure_expose_content_free_telemetry(self):
        provider = self.provider(lambda _: self.success())
        result = provider.complete("sys", "user")
        telemetry = result.telemetry
        self.assertIs(telemetry, provider.last_telemetry)
        self.assertEqual("success", telemetry.category)
        self.assertEqual((1, 200), (telemetry.attempt_no, telemetry.http_status))
        self.assertEqual(7, telemetry.input_chars)
        self.assertEqual((7, 3), (telemetry.prompt_tokens, telemetry.completion_tokens))
        self.assertEqual(1, len(telemetry.attempts))
        self.assertEqual("success", telemetry.attempts[0].category)
        self.assertGreater(telemetry.response_chars, len(result.text))
        self.assertNotIn("telemetry", result.model_dump())
        safe_metrics = json.dumps(telemetry.model_dump())
        for forbidden in ("anonymous.invalid", "Authorization", "anonymous-unit-test-key", result.text):
            self.assertNotIn(forbidden, safe_metrics)

        def fail(request):
            raise httpx.ReadTimeout("private upstream detail", request=request)

        provider = self.provider(fail)
        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("sys", "private input")
        self.assertIs(caught.exception.telemetry, provider.last_telemetry)
        self.assertEqual("read_timeout", caught.exception.telemetry.category)
        self.assertEqual(3, caught.exception.telemetry.attempt_no)
        self.assertEqual(3, len(caught.exception.telemetry.attempts))
        self.assertNotIn("private input", json.dumps(caught.exception.telemetry.model_dump()))

    def test_completion_limit_is_opt_in_and_truncation_is_failure(self):
        payloads = []

        def handler(request):
            payloads.append(json.loads(request.content))
            return self.success()

        self.provider(handler).complete("s", "u")
        self.assertNotIn("max_tokens", payloads[-1])

        limited_settings = self.settings(provider_max_completion_tokens=321)
        self.provider(handler, settings=limited_settings).complete("s", "u")
        self.assertEqual(321, payloads[-1]["max_tokens"])

        calls = 0

        def truncated(_):
            nonlocal calls
            calls += 1
            return self.success(finish_reason="length")

        with self.assertRaises(ProviderRetryExhausted) as caught:
            self.provider(truncated, settings=limited_settings).complete("s", "u")
        self.assertEqual(1, calls)
        self.assertEqual("truncated", caught.exception.category)

    def test_thinking_extension_is_opt_in_and_uses_only_top_level_type(self):
        payloads = []

        def handler(request):
            payloads.append(json.loads(request.content))
            return self.success()

        self.provider(handler).complete("s", "u")
        self.assertNotIn("thinking", payloads[-1])

        for mode in ("disabled", "enabled"):
            with self.subTest(mode=mode):
                configured = self.settings(provider_thinking_mode=mode)
                self.provider(handler, settings=configured).complete("s", "u")
                payload = payloads[-1]
                self.assertEqual({"type": mode}, payload["thinking"])
                for forbidden in (
                    "reasoning",
                    "reasoning_effort",
                    "max_completion_tokens",
                    "strict",
                ):
                    self.assertNotIn(forbidden, payload)

        legacy = self.settings(
            provider_thinking_mode="disabled",
            provider_max_completion_tokens=321,
        )
        self.provider(handler, settings=legacy).complete("s", "u")
        self.assertEqual(321, payloads[-1]["max_tokens"])
        self.assertEqual({"type": "disabled"}, payloads[-1]["thinking"])

    def test_retry_after_is_respected_with_cap_and_jitter(self):
        statuses = iter((429, 200))
        sleeps = []

        def handler(_):
            status = next(statuses)
            if status == 429:
                return httpx.Response(429, headers={"Retry-After": "60"})
            return self.success()

        provider = self.provider(
            handler,
            retry_policy=RetryPolicy(
                max_attempts=2,
                base_delay_seconds=0.25,
                max_delay_seconds=5,
                jitter_ratio=0.1,
            ),
            sleep=sleeps.append,
            random_value=lambda: 1,
        )
        result = provider.complete("s", "u")
        self.assertEqual([5], sleeps)
        self.assertEqual(
            ["rate_limit", "success"],
            [row.category for row in result.telemetry.attempts],
        )
        self.assertEqual(429, result.telemetry.attempts[0].http_status)

    def test_total_deadline_bounds_retry_sleep(self):
        now = 100.0
        calls = 0

        def monotonic():
            return now

        def sleep(seconds):
            nonlocal now
            now += seconds

        def handler(request):
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("anonymous", request=request)

        provider = self.provider(
            handler,
            settings=self.settings(provider_total_deadline_seconds=0.2),
            retry_policy=RetryPolicy(
                max_attempts=3,
                base_delay_seconds=1,
                jitter_ratio=0,
            ),
            sleep=sleep,
            monotonic=monotonic,
        )
        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")
        self.assertEqual(1, calls)
        self.assertEqual("read_timeout", caught.exception.category)
        self.assertEqual(200, caught.exception.elapsed_ms)

    def test_total_deadline_is_checked_between_success_body_chunks(self):
        clock = {"now": 100.0}

        class DeadlineCrossingStream(httpx.SyncByteStream):
            def __init__(self):
                self.yielded = 0
                self.closed = False

            def __iter__(self):
                self.yielded += 1
                yield b"{}"
                clock["now"] = 102.0
                self.yielded += 1
                yield b"private after deadline"

            def close(self):
                self.closed = True

        stream = DeadlineCrossingStream()
        provider = self.provider(
            lambda _: httpx.Response(200, stream=stream),
            settings=self.settings(provider_total_deadline_seconds=1.0),
            retry_policy=RetryPolicy(
                max_attempts=1,
                base_delay_seconds=0,
                jitter_ratio=0,
            ),
            monotonic=lambda: clock["now"],
        )

        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")

        self.assertEqual("read_timeout", caught.exception.category)
        self.assertEqual(2, caught.exception.telemetry.received_bytes)
        self.assertEqual(0, caught.exception.telemetry.response_chars)
        self.assertEqual(2, stream.yielded)
        self.assertTrue(stream.closed)
        self.assertNotIn("private", str(caught.exception))

    def test_total_deadline_precedes_preconsumed_identity_body_access(self):
        clock = {"now": 100.0}

        def handler(_):
            clock["now"] = 102.0
            return self.success()

        provider = self.provider(
            handler,
            settings=self.settings(provider_total_deadline_seconds=1.0),
            retry_policy=RetryPolicy(
                max_attempts=1,
                base_delay_seconds=0,
                jitter_ratio=0,
            ),
            monotonic=lambda: clock["now"],
        )

        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")

        self.assertEqual("read_timeout", caught.exception.category)
        self.assertEqual(0, caught.exception.telemetry.received_bytes)
        self.assertEqual(0, caught.exception.telemetry.response_chars)

    def test_total_deadline_is_checked_after_waiting_for_stream_eof(self):
        clock = {"now": 100.0}
        body = self.success_bytes()

        class SlowEofStream(httpx.SyncByteStream):
            def __init__(self):
                self.closed = False

            def __iter__(self):
                yield body
                # Simulate waiting for EOF after the complete last body chunk.
                clock["now"] = 102.0

            def close(self):
                self.closed = True

        stream = SlowEofStream()
        provider = self.provider(
            lambda _: httpx.Response(200, stream=stream),
            settings=self.settings(provider_total_deadline_seconds=1.0),
            retry_policy=RetryPolicy(
                max_attempts=1,
                base_delay_seconds=0,
                jitter_ratio=0,
            ),
            monotonic=lambda: clock["now"],
        )

        with self.assertRaises(ProviderRetryExhausted) as caught:
            provider.complete("s", "u")

        self.assertEqual("read_timeout", caught.exception.category)
        self.assertEqual(len(body), caught.exception.telemetry.received_bytes)
        self.assertEqual(0, caught.exception.telemetry.response_chars)
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
