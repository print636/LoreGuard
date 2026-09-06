from __future__ import annotations

import json
import unittest
from dataclasses import replace
from unittest.mock import patch

import httpx

from app.config import Settings
from app.embeddings import (
    EmbeddingInputError,
    EmbeddingNotConfiguredError,
    EmbeddingResponseError,
    EmbeddingRetryExhaustedError,
    OpenAICompatibleEmbeddingProvider,
)


class _PlainStream(httpx.SyncByteStream):
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __iter__(self):
        yield self.body


class EmbeddingProviderTests(unittest.TestCase):
    def settings(self, **overrides) -> Settings:
        values = {
            "enable_embeddings": True,
            "embedding_api_key": "embedding-secret-marker",
            "embedding_base_url": "https://embedding.invalid/v1",
            "embedding_model": "embedding-model",
            "embedding_model_revision": "rev-1",
            "embedding_dimensions": 2,
            "embedding_max_attempts": 2,
            "embedding_total_deadline_seconds": 5,
            "embedding_timeout_seconds": 1,
            "openai_api_key": "chat-secret-marker",
            "openai_model": "chat-model-must-not-be-used",
        }
        values.update(overrides)
        return Settings(_env_file=None, **values)

    def test_chat_configuration_does_not_enable_embeddings(self):
        settings = Settings(
            _env_file=None,
            enable_embeddings=False,
            openai_api_key="configured-chat-only",
            openai_model="chat-only",
        )
        provider = OpenAICompatibleEmbeddingProvider(settings)
        with self.assertRaisesRegex(EmbeddingNotConfiguredError, "disabled"):
            _ = provider.profile

    def test_profile_id_is_derived_from_the_complete_canonical_identity(self):
        profile = OpenAICompatibleEmbeddingProvider(self.settings()).profile
        with self.assertRaisesRegex(ValueError, "profile id"):
            replace(profile, profile_id="emb-" + ("0" * 64))
        changed = OpenAICompatibleEmbeddingProvider(
            self.settings(embedding_model_revision="rev-2")
        ).profile
        self.assertNotEqual(profile.profile_id, changed.profile_id)

    def test_success_reorders_indexes_normalizes_and_emits_safe_telemetry(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "application/json", "x-request-id": "safe-id_1"},
                json={
                    "data": [
                        {"index": 1, "embedding": [0, 5]},
                        {"index": 0, "embedding": [3, 4]},
                    ],
                    "usage": {"prompt_tokens": 7, "total_tokens": 7},
                },
            )

        provider = OpenAICompatibleEmbeddingProvider(
            self.settings(), transport=httpx.MockTransport(handler)
        )
        result = provider.embed(["银河列车", "角色记忆"])

        self.assertEqual(result.vectors, ((0.6, 0.8), (0.0, 1.0)))
        self.assertEqual(result.usage.total_tokens, 7)
        self.assertEqual(result.telemetry.input_count, 2)
        self.assertEqual(result.telemetry.input_chars, 8)
        request_token = result.telemetry.attempts[0].request_id
        self.assertRegex(request_token or "", r"^rid-[0-9a-f]{16}$")
        self.assertNotIn("safe-id_1", request_token or "")
        self.assertEqual(seen[0].url.path, "/v1/embeddings")
        payload = json.loads(seen[0].content)
        self.assertEqual(payload["model"], "embedding-model")
        self.assertEqual(seen[0].headers["accept-encoding"], "identity")
        self.assertEqual(
            seen[0].headers["authorization"], "Bearer embedding-secret-marker"
        )
        rendered = repr(result)
        for sensitive in (
            "embedding-secret-marker",
            "chat-secret-marker",
            "embedding.invalid",
            "银河列车",
            "0.6",
        ):
            self.assertNotIn(sensitive, rendered)
        other_namespace = OpenAICompatibleEmbeddingProvider(
            self.settings(embedding_profile_namespace="another-provider")
        ).profile
        self.assertNotEqual(result.profile.profile_id, other_namespace.profile_id)

    def test_empty_key_is_allowed_without_chat_fallback_or_authorization(self):
        seen: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "data": [{"index": 0, "embedding": [1, 0]}],
                    "usage": {"prompt_tokens": 10_000_000_001},
                },
            )

        provider = OpenAICompatibleEmbeddingProvider(
            self.settings(embedding_api_key="", openai_api_key="chat-must-not-leak"),
            transport=httpx.MockTransport(handler),
        )
        result = provider.embed(["local TEI"])
        self.assertNotIn("authorization", seen[0].headers)
        self.assertNotIn("chat-must-not-leak", repr(seen[0].headers))
        self.assertIsNone(result.usage.prompt_tokens)

    def test_revision_and_insecure_http_require_explicit_configuration(self):
        for revision in ("", "unspecified", "UNSPECIFIED"):
            provider = OpenAICompatibleEmbeddingProvider(
                self.settings(embedding_model_revision=revision)
            )
            with self.subTest(revision=revision), self.assertRaisesRegex(
                EmbeddingNotConfiguredError, "revision"
            ):
                _ = provider.profile

        blocked = OpenAICompatibleEmbeddingProvider(
            self.settings(embedding_base_url="http://tei:8080")
        )
        with self.assertRaisesRegex(EmbeddingNotConfiguredError, "base URL"):
            _ = blocked.profile
        allowed = OpenAICompatibleEmbeddingProvider(
            self.settings(
                embedding_base_url="http://tei:8080",
                embedding_allow_insecure_http=True,
            )
        )
        self.assertTrue(allowed.profile.profile_id.startswith("emb-"))

    def test_redirect_is_not_followed_and_error_is_sanitized(self):
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(
                307,
                headers={"location": "https://secret.invalid/leak"},
                content=b"secret-upstream-body",
            )

        provider = OpenAICompatibleEmbeddingProvider(
            self.settings(), transport=httpx.MockTransport(handler)
        )
        with self.assertRaises(EmbeddingResponseError) as caught:
            provider.embed(["safe input"])
        self.assertEqual(calls, 1)
        self.assertEqual(str(caught.exception), "embedding redirects are not permitted")
        self.assertNotIn("secret.invalid", repr(caught.exception.telemetry))

    def test_retryable_status_and_transport_are_bounded(self):
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    429,
                    headers={"retry-after": "0"},
                    content=b"sensitive quota body",
                    request=request,
                )
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={"data": [{"index": 0, "embedding": [1, 0]}]},
                request=request,
            )

        provider = OpenAICompatibleEmbeddingProvider(
            self.settings(), transport=httpx.MockTransport(handler), sleeper=lambda _: None
        )
        result = provider.embed(["one"])
        self.assertEqual(calls, 2)
        self.assertEqual(
            [attempt.outcome for attempt in result.telemetry.attempts],
            ["retryable_status", "success"],
        )

        def timeout(_: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("sensitive transport detail")

        failing = OpenAICompatibleEmbeddingProvider(
            self.settings(), transport=httpx.MockTransport(timeout), sleeper=lambda _: None
        )
        with self.assertRaises(EmbeddingRetryExhaustedError) as caught:
            failing.embed(["one"])
        self.assertEqual(str(caught.exception), "embedding retries were exhausted")
        self.assertEqual(len(caught.exception.telemetry.attempts), 2)
        self.assertNotIn("sensitive", repr(caught.exception.telemetry))

    def test_retry_after_values_are_always_finite_and_bounded(self):
        cases = (("nan", 0.1), ("inf", 0.1), ("-5", 0.0), ("999999", 2.0))
        for header, expected_delay in cases:
            calls = 0
            delays: list[float] = []

            def handler(_: httpx.Request) -> httpx.Response:
                nonlocal calls
                calls += 1
                if calls == 1:
                    return httpx.Response(429, headers={"retry-after": header})
                return httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                    json={"data": [{"index": 0, "embedding": [1, 0]}]},
                )

            provider = OpenAICompatibleEmbeddingProvider(
                self.settings(),
                transport=httpx.MockTransport(handler),
                sleeper=delays.append,
            )
            provider.embed(["one"])
            self.assertEqual(delays, [expected_delay])

    def test_total_deadline_applies_while_success_body_is_streaming(self):
        class Clock:
            now = 0.0

            def monotonic(self) -> float:
                return self.now

        class DelayedBody(httpx.SyncByteStream):
            def __iter__(self):
                clock.now = 6.0
                yield b'{"data":[{"index":0,"embedding":[1,0]}]}'

        clock = Clock()
        provider = OpenAICompatibleEmbeddingProvider(
            self.settings(embedding_total_deadline_seconds=5),
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                    stream=DelayedBody(),
                )
            ),
            monotonic=clock.monotonic,
        )
        with self.assertRaisesRegex(EmbeddingRetryExhaustedError, "deadline") as caught:
            provider.embed(["one"])
        self.assertEqual(caught.exception.telemetry.attempts[0].outcome, "deadline_exceeded")

    def test_response_caps_and_contract_validation_fail_closed(self):
        oversized = OpenAICompatibleEmbeddingProvider(
            self.settings(embedding_max_response_bytes=10),
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={"content-type": "application/json", "content-length": "999"},
                    content=b"secret body",
                )
            ),
        )
        with self.assertRaisesRegex(EmbeddingResponseError, "byte limit"):
            oversized.embed(["one"])

        invalid_payloads = [
            {"data": []},
            {"data": [{"index": 0, "embedding": [1]}]},
            {"data": [{"index": True, "embedding": [1, 0]}]},
            {"data": [{"index": 0, "embedding": [0, 0]}]},
            {"data": [{"index": 0, "embedding": [float("nan"), 1]}]},
            {"data": [{"index": 0, "embedding": ["bad", 1]}]},
            {
                "data": [
                    {"index": 0, "embedding": [1, 0]},
                    {"index": 0, "embedding": [0, 1]},
                ]
            },
        ]
        for payload in invalid_payloads:
            with self.subTest(payload=repr(payload)):
                provider = OpenAICompatibleEmbeddingProvider(
                    self.settings(embedding_max_attempts=1),
                    transport=httpx.MockTransport(
                        lambda _, current=payload: httpx.Response(
                            200,
                            headers={"content-type": "application/json"},
                            content=json.dumps(current).encode("utf-8"),
                        )
                    ),
                )
                with self.assertRaises(EmbeddingResponseError) as caught:
                    provider.embed(["one"])
                self.assertEqual(caught.exception.telemetry.attempts[0].outcome, "invalid_response")
                self.assertNotIn("embedding-secret-marker", str(caught.exception))

        huge_integer = "1" + ("0" * 400)
        provider = OpenAICompatibleEmbeddingProvider(
            self.settings(embedding_max_attempts=1),
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={"content-type": "application/json"},
                    content=(
                        '{"data":[{"index":0,"embedding":['
                        + huge_integer
                        + ",1]}]}"
                    ).encode(),
                )
            ),
        )
        with self.assertRaisesRegex(EmbeddingResponseError, "vector value"):
            provider.embed(["one"])

        encoded = OpenAICompatibleEmbeddingProvider(
            self.settings(embedding_max_attempts=1),
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    headers={
                        "content-type": "application/json",
                        "content-encoding": "gzip",
                    },
                    stream=_PlainStream(b"not inspected"),
                )
            ),
        )
        with self.assertRaisesRegex(EmbeddingResponseError, "encoding"):
            encoded.embed(["one"])

        with patch("app.embeddings.json.loads", side_effect=ValueError("unsafe")):
            malformed = OpenAICompatibleEmbeddingProvider(
                self.settings(embedding_max_attempts=1),
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(
                        200,
                        headers={"content-type": "application/json"},
                        content=b"{}",
                    )
                ),
            )
            with self.assertRaises(EmbeddingResponseError) as caught:
                malformed.embed(["one"])
            self.assertIsNotNone(caught.exception.telemetry)

    def test_input_and_endpoint_limits_fail_before_transport(self):
        calls = 0

        def handler(_: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            raise AssertionError("transport must not be called")

        provider = OpenAICompatibleEmbeddingProvider(
            self.settings(embedding_batch_max_items=1, embedding_batch_max_chars=3),
            transport=httpx.MockTransport(handler),
        )
        for values in ([], [""], ["abcd"], ["a", "b"], "abc"):
            with self.subTest(values=values), self.assertRaises(EmbeddingInputError):
                provider.embed(values)  # type: ignore[arg-type]
        self.assertEqual(calls, 0)

        bad_url = OpenAICompatibleEmbeddingProvider(
            self.settings(embedding_base_url="https://user:secret@invalid/v1?leak=1"),
            transport=httpx.MockTransport(handler),
        )
        with self.assertRaisesRegex(EmbeddingNotConfiguredError, "base URL is invalid"):
            _ = bad_url.profile
        self.assertEqual(calls, 0)


if __name__ == "__main__":
    unittest.main()
