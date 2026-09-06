from __future__ import annotations

import json
import math
import unittest
from pathlib import Path
from unittest.mock import Mock

import httpx

from scripts.run_tei_smoke import (
    DEFAULT_DIMENSIONS,
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    SmokeConfig,
    SmokeFailure,
    verify_embeddings,
    verify_info,
    verify_oversized_input_rejected,
)


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = (ROOT / "docker-compose.rag.yml").read_text(encoding="utf-8")
BASE_COMPOSE = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")


def config() -> SmokeConfig:
    return SmokeConfig(
        base_url="http://embeddings:80",
        expected_model=DEFAULT_MODEL,
        expected_revision=DEFAULT_REVISION,
        expected_dimensions=DEFAULT_DIMENSIONS,
        startup_timeout_seconds=1,
        request_timeout_seconds=1,
    )


def vector(index: int, values: list[float]) -> dict:
    return {"object": "embedding", "index": index, "embedding": values}


class TeiComposeStructureTests(unittest.TestCase):
    def test_overlay_pins_private_cpu_tei_and_model_identity(self):
        self.assertIn(
            "ghcr.io/huggingface/text-embeddings-inference@sha256:"
            "ad950d30878eceb72aaf32024d26fa2b1d04a75304fa0b4776b49aa1941fea07",
            OVERLAY,
        )
        self.assertGreaterEqual(OVERLAY.count(DEFAULT_MODEL), 5)
        self.assertGreaterEqual(OVERLAY.count(DEFAULT_REVISION), 4)
        for required in (
            "- float32",
            "- cls",
            "- --served-model-name",
            '- "false"',
            "tei_model_cache:/data",
            "http://localhost:80/health",
            "condition: service_healthy",
        ):
            self.assertIn(required, OVERLAY)
        embeddings_block = OVERLAY.split("  embeddings:", 1)[1].split("\n  api:", 1)[0]
        self.assertNotIn("ports:", embeddings_block)
        self.assertIn("expose:", embeddings_block)

    def test_overlay_enables_an_independent_keyless_profile_for_api_and_worker(self):
        self.assertEqual(2, OVERLAY.count('ENABLE_EMBEDDINGS: "true"'))
        self.assertEqual(2, OVERLAY.count('EMBEDDING_API_KEY: ""'))
        self.assertEqual(2, OVERLAY.count("EMBEDDING_BASE_URL: http://embeddings:80/v1"))
        self.assertEqual(2, OVERLAY.count('EMBEDDING_DIMENSIONS: "512"'))
        self.assertEqual(2, OVERLAY.count('EMBEDDING_ALLOW_INSECURE_HTTP: "true"'))
        self.assertNotIn("OPENAI_", OVERLAY)
        self.assertIn('ENABLE_EMBEDDINGS: ${ENABLE_EMBEDDINGS:-false}', BASE_COMPOSE)
        self.assertNotIn("text-embeddings-inference", BASE_COMPOSE)

    def test_smoke_is_explicit_and_has_no_host_or_key_configuration(self):
        self.assertIn('profiles: ["rag-smoke"]', OVERLAY)
        self.assertIn("scripts/run_tei_smoke.py", OVERLAY)
        self.assertNotIn("HF_TOKEN", OVERLAY)
        self.assertNotIn("API_KEY:", OVERLAY.replace('EMBEDDING_API_KEY: ""', ""))


class TeiSmokeContractTests(unittest.TestCase):
    def test_info_requires_the_exact_model_and_revision(self):
        response = httpx.Response(
            200,
            json={
                "model_id": DEFAULT_MODEL,
                "model_sha": DEFAULT_REVISION,
                "max_input_length": 512,
            },
            request=httpx.Request("GET", "http://embeddings/info"),
        )
        client = Mock()
        client.request.return_value = response
        self.assertEqual(DEFAULT_REVISION, verify_info(client, config())["model_sha"])

        response._content = json.dumps(
            {
                "model_id": DEFAULT_MODEL,
                "model_sha": "moving-branch",
                "max_input_length": 512,
            }
        ).encode()
        with self.assertRaisesRegex(SmokeFailure, "SHA"):
            verify_info(client, config())

    def test_embedding_contract_checks_batch_single_norm_dimension_and_ranking(self):
        dim = DEFAULT_DIMENSIONS
        query = [1.0] + [0.0] * (dim - 1)
        related = [0.8, 0.6] + [0.0] * (dim - 2)
        unrelated = [0.0, 1.0] + [0.0] * (dim - 2)
        batches = [query, related, unrelated]
        responses = [
            httpx.Response(
                200,
                json={
                    "model": DEFAULT_MODEL,
                    "data": [vector(index, row) for index, row in enumerate(batches)],
                },
                request=httpx.Request("POST", "http://embeddings/v1/embeddings"),
            ),
            *[
                httpx.Response(
                    200,
                    json={"model": DEFAULT_MODEL, "data": [vector(0, row)]},
                    request=httpx.Request("POST", "http://embeddings/v1/embeddings"),
                )
                for row in batches
            ],
        ]
        client = Mock()
        client.request.side_effect = responses
        result = verify_embeddings(client, config())
        self.assertEqual(dim, result["dimensions"])
        self.assertGreater(result["related_cosine"], result["unrelated_cosine"])
        self.assertTrue(math.isclose(1.0, result["minimum_batch_single_cosine"]))
        for call in client.request.call_args_list:
            self.assertNotIn("Authorization", call.kwargs["headers"])

    def test_oversized_input_requires_a_client_error(self):
        client = Mock()
        client.request.return_value = httpx.Response(
            413,
            text="input too long",
            request=httpx.Request("POST", "http://embeddings/v1/embeddings"),
        )
        self.assertEqual(
            413,
            verify_oversized_input_rejected(
                client,
                config(),
                {"max_input_length": 512},
            ),
        )
        sent_text = client.request.call_args.kwargs["json"]["input"][0]
        self.assertGreater(len(sent_text), 512)


if __name__ == "__main__":
    unittest.main()
