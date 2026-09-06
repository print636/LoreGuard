from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from typing import Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx


MAX_RESPONSE_BYTES = 4 * 1024 * 1024
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_REVISION = "7999e1d3359715c523056ef9478215996d62a620"
DEFAULT_DIMENSIONS = 512


class SmokeFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class SmokeConfig:
    base_url: str
    expected_model: str
    expected_revision: str
    expected_dimensions: int
    startup_timeout_seconds: float
    request_timeout_seconds: float


def _endpoint(base_url: str, path: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or not parsed.hostname
        or parsed.hostname not in {"embeddings", "localhost", "127.0.0.1", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise SmokeFailure("TEI smoke base URL must be an internal plain HTTP endpoint")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def _bounded_json(response: httpx.Response) -> dict:
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise SmokeFailure("TEI response exceeded the smoke byte limit")
    try:
        payload = response.json()
    except (UnicodeError, ValueError):
        raise SmokeFailure("TEI returned invalid JSON") from None
    if not isinstance(payload, dict):
        raise SmokeFailure("TEI returned an invalid JSON shape")
    return payload


def _request(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    payload: dict | None = None,
) -> httpx.Response:
    response = client.request(
        method,
        url,
        json=payload,
        headers={"Accept": "application/json", "Accept-Encoding": "identity"},
    )
    if 300 <= response.status_code < 400:
        raise SmokeFailure("TEI returned an unexpected redirect")
    return response


def wait_for_health(client: httpx.Client, config: SmokeConfig) -> None:
    deadline = time.monotonic() + config.startup_timeout_seconds
    health_url = _endpoint(config.base_url, "/health")
    last_status = None
    while time.monotonic() < deadline:
        try:
            response = _request(client, "GET", health_url)
            last_status = response.status_code
            if response.status_code == 200:
                return
        except httpx.RequestError:
            pass
        time.sleep(2)
    suffix = f" (last status {last_status})" if last_status is not None else ""
    raise SmokeFailure(f"TEI did not become healthy before the deadline{suffix}")


def verify_info(client: httpx.Client, config: SmokeConfig) -> dict:
    response = _request(client, "GET", _endpoint(config.base_url, "/info"))
    if response.status_code != 200:
        raise SmokeFailure(f"TEI info endpoint returned HTTP {response.status_code}")
    info = _bounded_json(response)
    if info.get("model_id") != config.expected_model:
        raise SmokeFailure("TEI info model identity does not match the pinned model")
    if info.get("model_sha") != config.expected_revision:
        raise SmokeFailure("TEI info model SHA does not match the pinned revision")
    return info


def _parse_vectors(
    payload: dict,
    *,
    expected_count: int,
    expected_dimensions: int,
    expected_model: str,
) -> tuple[tuple[float, ...], ...]:
    if payload.get("model") not in {None, expected_model}:
        raise SmokeFailure("TEI embedding response used an unexpected served model")
    rows = payload.get("data")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise SmokeFailure("TEI embedding response count is invalid")
    ordered: list[tuple[float, ...] | None] = [None] * expected_count
    for row in rows:
        if not isinstance(row, dict):
            raise SmokeFailure("TEI embedding item is invalid")
        index = row.get("index")
        vector = row.get("embedding")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= expected_count
            or ordered[index] is not None
            or not isinstance(vector, list)
            or len(vector) != expected_dimensions
        ):
            raise SmokeFailure("TEI embedding index or dimension is invalid")
        try:
            converted = tuple(float(value) for value in vector)
        except (OverflowError, TypeError, ValueError):
            raise SmokeFailure("TEI embedding contains an invalid value") from None
        if not all(math.isfinite(value) for value in converted):
            raise SmokeFailure("TEI embedding contains a non-finite value")
        norm = math.sqrt(sum(value * value for value in converted))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-4:
            raise SmokeFailure("TEI embedding is not L2-normalized")
        ordered[index] = converted
    if any(vector is None for vector in ordered):
        raise SmokeFailure("TEI embedding indexes are incomplete")
    return tuple(vector for vector in ordered if vector is not None)


def embed(
    client: httpx.Client,
    config: SmokeConfig,
    texts: Sequence[str],
) -> tuple[tuple[float, ...], ...]:
    response = _request(
        client,
        "POST",
        _endpoint(config.base_url, "/v1/embeddings"),
        payload={"model": config.expected_model, "input": list(texts)},
    )
    if response.status_code != 200:
        raise SmokeFailure(f"TEI embedding endpoint returned HTTP {response.status_code}")
    return _parse_vectors(
        _bounded_json(response),
        expected_count=len(texts),
        expected_dimensions=config.expected_dimensions,
        expected_model=config.expected_model,
    )


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def verify_embeddings(client: httpx.Client, config: SmokeConfig) -> dict:
    query = "潮汐钥匙开启北港的星门"
    related = "在北港，只有潮汐钥匙能够打开星门。"
    unrelated = "厨房今天准备了苹果、面包和热茶。"
    texts = (query, related, unrelated)
    batched = embed(client, config, texts)
    single = tuple(embed(client, config, (text,))[0] for text in texts)

    agreement = tuple(cosine(batch, one) for batch, one in zip(batched, single, strict=True))
    if min(agreement) < 0.9999:
        raise SmokeFailure("TEI batch and single embeddings are inconsistent")
    related_score = cosine(batched[0], batched[1])
    unrelated_score = cosine(batched[0], batched[2])
    if related_score <= unrelated_score:
        raise SmokeFailure("TEI failed the frozen Chinese relevance ordering smoke")
    return {
        "dimensions": config.expected_dimensions,
        "minimum_batch_single_cosine": round(min(agreement), 6),
        "related_cosine": round(related_score, 6),
        "unrelated_cosine": round(unrelated_score, 6),
    }


def verify_oversized_input_rejected(
    client: httpx.Client,
    config: SmokeConfig,
    info: dict,
) -> int:
    reported_limit = info.get("max_input_length")
    if isinstance(reported_limit, bool) or not isinstance(reported_limit, int):
        raise SmokeFailure("TEI info did not report a finite input-token limit")
    if reported_limit < 1 or reported_limit > 1_000_000:
        raise SmokeFailure("TEI reported an invalid input-token limit")
    long_text = "潮汐星门超长输入片段 " * (reported_limit + 64)
    response = _request(
        client,
        "POST",
        _endpoint(config.base_url, "/v1/embeddings"),
        payload={"model": config.expected_model, "input": [long_text]},
    )
    if not 400 <= response.status_code < 500:
        raise SmokeFailure("TEI did not reject an input beyond the model token limit")
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise SmokeFailure("TEI oversized-input response exceeded the byte limit")
    return response.status_code


def run(config: SmokeConfig) -> dict:
    with httpx.Client(
        timeout=httpx.Timeout(config.request_timeout_seconds),
        follow_redirects=False,
    ) as client:
        wait_for_health(client, config)
        info = verify_info(client, config)
        result = verify_embeddings(client, config)
        result["oversized_input_status"] = verify_oversized_input_rejected(
            client, config, info
        )
        result["model"] = config.expected_model
        result["revision"] = config.expected_revision
        result["max_input_length"] = info["max_input_length"]
        return result


def parse_args(argv: Sequence[str] | None = None) -> SmokeConfig:
    parser = argparse.ArgumentParser(description="Verify the pinned private TEI service")
    parser.add_argument("--base-url", default="http://embeddings:80")
    parser.add_argument("--expected-model", default=DEFAULT_MODEL)
    parser.add_argument("--expected-revision", default=DEFAULT_REVISION)
    parser.add_argument("--expected-dimensions", type=int, default=DEFAULT_DIMENSIONS)
    parser.add_argument("--startup-timeout-seconds", type=float, default=900)
    parser.add_argument("--request-timeout-seconds", type=float, default=60)
    args = parser.parse_args(argv)
    if args.expected_dimensions < 1 or args.expected_dimensions > 16_000:
        parser.error("expected dimensions must be between 1 and 16000")
    if args.startup_timeout_seconds <= 0 or args.request_timeout_seconds <= 0:
        parser.error("timeouts must be positive")
    return SmokeConfig(
        base_url=args.base_url.rstrip("/"),
        expected_model=args.expected_model,
        expected_revision=args.expected_revision,
        expected_dimensions=args.expected_dimensions,
        startup_timeout_seconds=args.startup_timeout_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(argv))
    except (SmokeFailure, httpx.RequestError) as exc:
        print(f"TEI smoke failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": "passed", **result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
