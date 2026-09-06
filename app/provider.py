from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

import httpx
from pydantic import BaseModel, Field

from .config import Settings, get_settings


_REQUEST_ID_HEADERS = ("x-request-id", "request-id", "trace-id", "cf-ray")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")


def sanitize_request_id(value: Any) -> str | None:
    """Return one safe correlation token, never an arbitrary header value."""
    if not isinstance(value, str) or not _REQUEST_ID_PATTERN.fullmatch(value):
        return None
    return value


def safe_thinking_configuration(settings: Any) -> dict[str, Any]:
    """Return the allowlisted thinking mode without other provider settings."""
    mode = getattr(settings, "provider_thinking_mode", None)
    if mode not in {"disabled", "enabled"}:
        mode = None
    return {"configured": mode is not None, "mode": mode}


class ProviderAttemptTelemetry(BaseModel):
    """Safe, content-free metrics for one upstream request attempt."""

    attempt_no: int
    input_chars: int
    response_chars: int = 0
    received_bytes: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: int
    category: str
    http_status: int | None = None
    request_id: str | None = Field(
        default=None, max_length=128, pattern=_REQUEST_ID_PATTERN.pattern
    )


class ProviderCallTelemetry(BaseModel):
    """Safe aggregate metrics for one logical provider call."""

    input_chars: int
    response_chars: int = 0
    received_bytes: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: int
    category: str
    attempt_no: int = 0
    http_status: int | None = None
    request_id: str | None = Field(
        default=None, max_length=128, pattern=_REQUEST_ID_PATTERN.pattern
    )
    attempts: list[ProviderAttemptTelemetry] = Field(default_factory=list)


class ModelResult(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Additive compatibility: callers can inspect telemetry, while existing
    # serialized ModelResult payloads retain their previous shape.
    telemetry: ProviderCallTelemetry | None = Field(default=None, exclude=True)


class ProviderError(RuntimeError):
    """Safe provider failure that never includes credentials or response bodies."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "provider",
        http_status: int | None = None,
        attempt_no: int = 0,
        elapsed_ms: int = 0,
        telemetry: ProviderCallTelemetry | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.http_status = http_status
        self.attempt_no = attempt_no
        self.elapsed_ms = elapsed_ms
        self.telemetry = telemetry
        self.request_id = sanitize_request_id(request_id)


class ProviderNotConfigured(ProviderError):
    pass


class ProviderRetryExhausted(ProviderError):
    pass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 30.0
    jitter_ratio: float = 0.1


@dataclass(frozen=True, slots=True)
class _Failure:
    category: str
    attempt_no: int
    http_status: int | None = None
    retryable: bool = False
    retry_after_seconds: float | None = None
    request_id: str | None = None


class OpenAICompatibleProvider:
    """Small, injectable OpenAI-compatible gateway with bounded retries."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.perf_counter,
        wall_time: Callable[[], float] = time.time,
        random_value: Callable[[], float] = random.random,
    ) -> None:
        self.settings = settings or get_settings()
        self.transport = transport
        self.retry_policy = retry_policy or RetryPolicy(
            max_attempts=self.settings.provider_max_attempts
        )
        self.sleep = sleep
        self.monotonic = monotonic
        self.wall_time = wall_time
        self.random_value = random_value
        self.last_telemetry: ProviderCallTelemetry | None = None

    @property
    def configured(self) -> bool:
        return self.settings.enable_model_extraction and bool(
            self.settings.openai_api_key.strip()
        )

    def complete(self, system: str, user: str) -> ModelResult:
        call_started = self.monotonic()
        input_chars = len(system) + len(user)
        attempts: list[ProviderAttemptTelemetry] = []
        self.last_telemetry = None

        if not self.configured:
            telemetry = self._call_telemetry(
                call_started, input_chars, "not_configured", attempts
            )
            self.last_telemetry = telemetry
            raise ProviderNotConfigured(
                "模型未配置",
                category="not_configured",
                elapsed_ms=telemetry.elapsed_ms,
                telemetry=telemetry,
            )

        payload: dict[str, Any] = {
            "model": self.settings.openai_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        if self.settings.provider_max_completion_tokens is not None:
            # max_tokens remains the most broadly supported Chat Completions
            # parameter across OpenAI-compatible gateways.
            payload["max_tokens"] = self.settings.provider_max_completion_tokens
        if self.settings.provider_thinking_mode is not None:
            payload["thinking"] = {"type": self.settings.provider_thinking_mode}

        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"
        deadline = (
            call_started + self.settings.provider_total_deadline_seconds
            if self.settings.provider_total_deadline_seconds is not None
            else None
        )
        last_failure: _Failure | None = None

        with httpx.Client(
            timeout=self._request_timeout(deadline),
            follow_redirects=False,
            transport=self.transport,
        ) as client:
            for attempt_no in range(1, self.retry_policy.max_attempts + 1):
                if deadline is not None and self.monotonic() >= deadline:
                    break

                attempt_started = self.monotonic()
                response_chars = 0
                received_bytes = 0
                prompt_tokens = 0
                completion_tokens = 0
                request_id = None
                failure: _Failure | None = None
                try:
                    with client.stream(
                        "POST",
                        endpoint,
                        json=payload,
                        headers=headers,
                        timeout=self._request_timeout(deadline),
                    ) as response:
                        status = response.status_code
                        request_id = self._request_id(response)

                        if status == 429:
                            failure = _Failure(
                                "rate_limit",
                                attempt_no,
                                http_status=status,
                                retryable=True,
                                retry_after_seconds=self._retry_after(response),
                                request_id=request_id,
                            )
                        elif status >= 500:
                            failure = _Failure(
                                "upstream_5xx",
                                attempt_no,
                                http_status=status,
                                retryable=True,
                                retry_after_seconds=self._retry_after(response),
                                request_id=request_id,
                            )
                        elif status >= 300:
                            failure = _Failure(
                                self._http_failure_category(status),
                                attempt_no,
                                http_status=status,
                                request_id=request_id,
                            )
                        else:
                            raw_body, received_bytes, failure = (
                                self._read_success_body(response, attempt_no)
                            )
                            if failure is None:
                                assert raw_body is not None
                                response_chars = self._response_chars(raw_body)
                                body, failure = self._response_body(
                                    raw_body, attempt_no
                                )
                                if failure is None:
                                    (
                                        text,
                                        prompt_tokens,
                                        completion_tokens,
                                        failure,
                                    ) = self._response_content(body, attempt_no)

                        # Error responses are intentionally never consumed. A
                        # successful oversized stream is closed by the bounded
                        # reader as soon as the boundary is crossed.
                        if failure is not None:
                            response.close()

                    if failure is not None and failure.request_id is None:
                        failure = replace(failure, request_id=request_id)

                    if failure is None:
                        attempts.append(
                            self._attempt_telemetry(
                                attempt_no,
                                input_chars,
                                response_chars,
                                received_bytes,
                                prompt_tokens,
                                completion_tokens,
                                attempt_started,
                                "success",
                                status,
                                request_id,
                            )
                        )
                        telemetry = self._call_telemetry(
                            call_started, input_chars, "success", attempts
                        )
                        self.last_telemetry = telemetry
                        return ModelResult(
                            text=text,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            telemetry=telemetry,
                        )
                except httpx.ConnectTimeout:
                    failure = _Failure("connect_timeout", attempt_no, retryable=True)
                except httpx.ReadTimeout:
                    failure = _Failure("read_timeout", attempt_no, retryable=True)
                except (httpx.TimeoutException, httpx.TransportError):
                    failure = _Failure("transport", attempt_no, retryable=True)

                # Every branch above either returned or recorded a safe failure.
                assert failure is not None
                last_failure = failure
                attempts.append(
                    self._attempt_telemetry(
                        attempt_no,
                        input_chars,
                        response_chars,
                        received_bytes,
                        prompt_tokens,
                        completion_tokens,
                        attempt_started,
                        failure.category,
                        failure.http_status,
                        failure.request_id,
                    )
                )

                if not failure.retryable:
                    break
                if attempt_no >= self.retry_policy.max_attempts:
                    break

                delay = self._retry_delay(attempt_no, failure.retry_after_seconds)
                if deadline is not None:
                    remaining = max(0.0, deadline - self.monotonic())
                    if remaining <= 0:
                        break
                    delay = min(delay, remaining)
                if delay > 0:
                    self.sleep(delay)

        # max_attempts <= 0 is invalid operationally but keep the failure safe.
        failure = last_failure or _Failure("transport", 0)
        telemetry = self._call_telemetry(
            call_started, input_chars, failure.category, attempts
        )
        self.last_telemetry = telemetry
        message = self._safe_error_message(failure)
        error_type = (
            ProviderError
            if failure.category in {"unauthorized", "forbidden", "nonretry_http"}
            else ProviderRetryExhausted
        )
        raise error_type(
            message,
            category=failure.category,
            http_status=failure.http_status,
            attempt_no=failure.attempt_no,
            elapsed_ms=telemetry.elapsed_ms,
            telemetry=telemetry,
            request_id=failure.request_id,
        )

    def _read_success_body(
        self, response: httpx.Response, attempt_no: int
    ) -> tuple[bytes | None, int, _Failure | None]:
        max_bytes = self.settings.provider_max_response_bytes
        if max_bytes is not None and self._content_length_exceeds(
            response, max_bytes
        ):
            response.close()
            return None, 0, _Failure(
                "response_too_large",
                attempt_no,
                http_status=response.status_code,
            )

        # MockTransport commonly receives pre-buffered Response objects from
        # test handlers. Preserve that injection contract while production
        # responses still take the streaming path below.
        if response.is_stream_consumed:
            raw_body = response.content
            received_bytes = len(raw_body)
            if max_bytes is not None and received_bytes > max_bytes:
                response.close()
                return None, received_bytes, _Failure(
                    "response_too_large",
                    attempt_no,
                    http_status=response.status_code,
                )
            return raw_body, received_bytes, None

        body = bytearray()
        received_bytes = 0
        try:
            for chunk in response.iter_raw():
                received_bytes += len(chunk)
                if max_bytes is not None and received_bytes > max_bytes:
                    body.clear()
                    response.close()
                    return None, received_bytes, _Failure(
                        "response_too_large",
                        attempt_no,
                        http_status=response.status_code,
                    )
                body.extend(chunk)
        except httpx.ConnectTimeout:
            body.clear()
            return None, received_bytes, _Failure(
                "connect_timeout", attempt_no, retryable=True
            )
        except httpx.ReadTimeout:
            body.clear()
            return None, received_bytes, _Failure(
                "read_timeout", attempt_no, retryable=True
            )
        except (httpx.TimeoutException, httpx.TransportError):
            body.clear()
            return None, received_bytes, _Failure(
                "transport", attempt_no, retryable=True
            )
        return bytes(body), received_bytes, None

    @staticmethod
    def _content_length_exceeds(
        response: httpx.Response, max_bytes: int
    ) -> bool:
        values = response.headers.get_list("content-length")
        if len(values) != 1:
            return False
        value = values[0]
        if re.fullmatch(r"[0-9]+", value) is None:
            return False

        # Compare canonical decimal strings instead of parsing an
        # provider-controlled, potentially enormous integer.
        canonical = value.lstrip("0") or "0"
        boundary = str(max_bytes)
        return len(canonical) > len(boundary) or (
            len(canonical) == len(boundary) and canonical > boundary
        )

    def _response_body(
        self, raw_body: bytes, attempt_no: int
    ) -> tuple[dict[str, Any], _Failure | None]:
        try:
            body: Any = json.loads(raw_body)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return {}, _Failure("body_json", attempt_no)
        if not isinstance(body, dict):
            return {}, _Failure("response_shape", attempt_no)
        return body, None

    @staticmethod
    def _response_content(
        body: dict[str, Any], attempt_no: int
    ) -> tuple[str, int, int, _Failure | None]:
        try:
            choices = body["choices"]
            if not isinstance(choices, list) or not choices:
                raise TypeError
            choice = choices[0]
            if not isinstance(choice, dict):
                raise TypeError
            message = choice["message"]
            if not isinstance(message, dict):
                raise TypeError
            text = message["content"]
        except (KeyError, IndexError, TypeError):
            return "", 0, 0, _Failure("response_shape", attempt_no)

        if not isinstance(text, str):
            return "", 0, 0, _Failure("response_shape", attempt_no)
        if not text.strip():
            return "", 0, 0, _Failure("empty_content", attempt_no)

        usage = body.get("usage", {})
        if not isinstance(usage, dict):
            return "", 0, 0, _Failure("usage_shape", attempt_no)
        try:
            prompt_tokens = OpenAICompatibleProvider._token_count(
                usage.get("prompt_tokens", 0)
            )
            completion_tokens = OpenAICompatibleProvider._token_count(
                usage.get("completion_tokens", 0)
            )
        except (TypeError, ValueError, OverflowError):
            return "", 0, 0, _Failure("usage_shape", attempt_no)

        if choice.get("finish_reason") == "length":
            return text, prompt_tokens, completion_tokens, _Failure(
                "truncated", attempt_no
            )
        try:
            json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text, prompt_tokens, completion_tokens, _Failure(
                "content_json", attempt_no
            )
        return text, prompt_tokens, completion_tokens, None

    @staticmethod
    def _token_count(value: Any) -> int:
        if isinstance(value, bool):
            raise TypeError
        count = int(value or 0)
        if count < 0:
            raise ValueError
        return count

    def _request_timeout(self, deadline: float | None) -> httpx.Timeout:
        timeout = self.settings.provider_timeout_seconds
        if deadline is not None:
            timeout = min(timeout, max(0.001, deadline - self.monotonic()))
        return httpx.Timeout(timeout, connect=min(10.0, timeout))

    @staticmethod
    def _response_chars(raw_body: bytes) -> int:
        try:
            return len(raw_body.decode("utf-8"))
        except UnicodeDecodeError:
            return len(raw_body)

    def _retry_after(self, response: httpx.Response) -> float | None:
        value = response.headers.get("Retry-After")
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                now = datetime.fromtimestamp(self.wall_time(), tz=timezone.utc)
                return max(0.0, (retry_at - now).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    @staticmethod
    def _http_failure_category(status: int) -> str:
        if status == 401:
            return "unauthorized"
        if status == 403:
            return "forbidden"
        return "nonretry_http"

    @staticmethod
    def _request_id(response: httpx.Response) -> str | None:
        # httpx headers are case-insensitive. Only these four values are ever
        # observed; the rest of the upstream header collection is discarded.
        for name in _REQUEST_ID_HEADERS:
            value = sanitize_request_id(response.headers.get(name))
            if value is not None:
                return value
        return None

    def _retry_delay(
        self, attempt_no: int, retry_after_seconds: float | None
    ) -> float:
        exponential = self.retry_policy.base_delay_seconds * (2 ** (attempt_no - 1))
        base = max(exponential, retry_after_seconds or 0.0)
        jitter = base * max(0.0, self.retry_policy.jitter_ratio) * max(
            0.0, min(1.0, self.random_value())
        )
        return min(max(0.0, self.retry_policy.max_delay_seconds), base + jitter)

    def _attempt_telemetry(
        self,
        attempt_no: int,
        input_chars: int,
        response_chars: int,
        received_bytes: int,
        prompt_tokens: int,
        completion_tokens: int,
        attempt_started: float,
        category: str,
        http_status: int | None,
        request_id: str | None,
    ) -> ProviderAttemptTelemetry:
        return ProviderAttemptTelemetry(
            attempt_no=attempt_no,
            input_chars=input_chars,
            response_chars=response_chars,
            received_bytes=received_bytes,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            elapsed_ms=self._elapsed_ms(attempt_started),
            category=category,
            http_status=http_status,
            request_id=request_id,
        )

    def _call_telemetry(
        self,
        call_started: float,
        input_chars: int,
        category: str,
        attempts: list[ProviderAttemptTelemetry],
    ) -> ProviderCallTelemetry:
        return ProviderCallTelemetry(
            input_chars=input_chars,
            response_chars=sum(row.response_chars for row in attempts),
            received_bytes=sum(row.received_bytes for row in attempts),
            prompt_tokens=sum(row.prompt_tokens for row in attempts),
            completion_tokens=sum(row.completion_tokens for row in attempts),
            elapsed_ms=self._elapsed_ms(call_started),
            category=category,
            attempt_no=attempts[-1].attempt_no if attempts else 0,
            http_status=attempts[-1].http_status if attempts else None,
            request_id=attempts[-1].request_id if attempts else None,
            attempts=list(attempts),
        )

    def _elapsed_ms(self, started: float) -> int:
        return max(0, round((self.monotonic() - started) * 1000))

    @staticmethod
    def _safe_error_message(failure: _Failure) -> str:
        if failure.http_status is not None:
            return (
                f"模型服务请求失败（HTTP {failure.http_status}；"
                f"category={failure.category}；attempt={failure.attempt_no}）"
            )
        return (
            f"模型服务请求失败（category={failure.category}；"
            f"attempt={failure.attempt_no}）"
        )
