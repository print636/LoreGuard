import json
import unittest

import httpx

from app.config import Settings
from app.provider import OpenAICompatibleProvider, ProviderError, ProviderRetryExhausted, RetryPolicy
from scripts.check_provider import check_provider


class ProviderProbeTests(unittest.TestCase):
    def provider(self, handler):
        settings = Settings(_env_file=None, enable_model_extraction=True,
            openai_base_url="https://probe.invalid/v1", openai_model="mock-model",
            openai_api_key="unit-test-private-value")
        return OpenAICompatibleProvider(settings, transport=httpx.MockTransport(handler),
            retry_policy=RetryPolicy(max_attempts=1), sleep=lambda _: None)

    def test_probe_uses_production_json_contract(self):
        def handler(request):
            self.assertEqual({"type": "json_object"}, json.loads(request.content)["response_format"])
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"status":"ok"}'}}]})
        report, code = check_provider(self.provider(handler))
        self.assertEqual(0, code)
        self.assertTrue(report["json_contract_ok"])

    def test_probe_does_not_echo_rejected_body(self):
        handler = lambda _: httpx.Response(403, text="unit-test-private-value")
        report, code = check_provider(self.provider(handler))
        self.assertEqual(1, code)
        self.assertEqual(403, report["http_status"])
        self.assertNotIn("unit-test-private-value", json.dumps(report))

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
        self.assertIn("ValueError", str(caught.exception))

    def test_unexpected_json_is_not_success(self):
        handler = lambda _: httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})
        report, code = check_provider(self.provider(handler))
        self.assertEqual(1, code)
        self.assertEqual("UnexpectedJSON", report["error_type"])


if __name__ == "__main__":
    unittest.main()
