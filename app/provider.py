from __future__ import annotations

import json
import math
import random
import re
import time
import zlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Literal, TypeVar

import httpx
from pydantic import BaseModel, Field

from .config import Settings, get_settings


_REQUEST_ID_HEADERS = ("x-request-id", "request-id", "trace-id", "cf-ray")
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_TOOL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_TOOL_CALL_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_MAX_TOOL_DEFINITIONS = 64
_MAX_TOOL_DESCRIPTION_CHARS = 4_096
_MAX_TOOL_SCHEMA_BYTES = 64 * 1_024
_MAX_TOOL_REQUEST_BYTES = 256 * 1_024
_ABSOLUTE_MAX_TOOL_CALLS = 32
_ABSOLUTE_MAX_TOOL_ARGUMENT_BYTES = 256 * 1_024
# ``provider_max_response_bytes=None`` only disables a deployment-specific
# tighter ceiling.  Successful response transport and decompression must still
# have a hard memory bound, especially when an upstream enables compression.
_ABSOLUTE_MAX_PROVIDER_RESPONSE_BYTES = 16 * 1_024 * 1_024

_ParsedResponse = TypeVar("_ParsedResponse")


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


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """One caller-owned function contract sent to an OpenAI-compatible API."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass(frozen=True, slots=True)
class NamedToolChoice:
    """Force the provider to select one named tool from the supplied allowlist."""

    name: str


@dataclass(frozen=True, slots=True)
class ToolCallLimits:
    """Defense-in-depth limits below the provider response-byte ceiling."""

    max_calls: int = 8
    max_argument_bytes: int = 64 * 1_024


@dataclass(frozen=True, slots=True)
class ProviderToolCall:
    """Structurally checked call with no raw response or prompt attached.

    ``arguments`` remains untrusted until a tool-specific Pydantic model and
    runtime authorization policy validate it immediately before execution.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ToolCallResult:
    """Sanitized transport result; it is not an authorization to run a tool."""

    tool_calls: tuple[ProviderToolCall, ...]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    telemetry: ProviderCallTelemetry | None = None


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


class ProviderToolCallError(ProviderError):
    """A safe local request or upstream native-tool contract failure."""

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
        capability_enabled = (
            self.settings.enable_model_extraction
            or self.settings.enable_issue_evidence_review
            or self.settings.enable_evidence_investigator
        )
        return capability_enabled and bool(self.settings.openai_api_key.strip())

    @property
    def model_extraction_configured(self) -> bool:
        """Report extraction readiness without borrowing another capability."""

        return self.settings.enable_model_extraction and bool(
            self.settings.openai_api_key.strip()
        )

    @property
    def evidence_investigator_configured(self) -> bool:
        """Report this capability without borrowing another feature's switch."""

        return self.settings.enable_evidence_investigator and bool(
            self.settings.openai_api_key.strip()
        )

    def fork_for_evidence_investigator(
        self,
        *,
        remaining_deadline_seconds: float | None = None,
    ) -> OpenAICompatibleProvider:
        """Create a single-attempt provider bounded for Investigator calls.

        Generic provider limits can only tighten the dedicated limits.  Other
        chat capabilities are disabled in the fork so an enabled extractor or
        evidence reviewer cannot accidentally make a disabled Investigator
        appear configured.
        """

        if remaining_deadline_seconds is not None and (
            isinstance(remaining_deadline_seconds, bool)
            or not isinstance(remaining_deadline_seconds, (int, float))
            or not math.isfinite(float(remaining_deadline_seconds))
            or float(remaining_deadline_seconds) <= 0
        ):
            raise ValueError("investigator remaining deadline is invalid")
        settings = self.settings
        deadline_caps = [settings.evidence_investigator_total_deadline_seconds]
        if settings.provider_total_deadline_seconds is not None:
            deadline_caps.append(settings.provider_total_deadline_seconds)
        if remaining_deadline_seconds is not None:
            deadline_caps.append(float(remaining_deadline_seconds))
        bounded_deadline = min(deadline_caps)

        completion_caps = [settings.evidence_investigator_max_completion_tokens]
        if settings.provider_max_completion_tokens is not None:
            completion_caps.append(settings.provider_max_completion_tokens)
        response_caps = [settings.evidence_investigator_max_response_bytes]
        if settings.provider_max_response_bytes is not None:
            response_caps.append(settings.provider_max_response_bytes)

        bounded_settings = settings.model_copy(
            update={
                "enable_model_extraction": False,
                "enable_review_agent": False,
                "enable_issue_evidence_review": False,
                "provider_timeout_seconds": min(
                    settings.provider_timeout_seconds,
                    settings.evidence_investigator_timeout_seconds,
                    bounded_deadline,
                ),
                "provider_total_deadline_seconds": bounded_deadline,
                "provider_max_attempts": 1,
                "provider_max_completion_tokens": min(completion_caps),
                "provider_max_response_bytes": min(response_caps),
            }
        )
        return OpenAICompatibleProvider(
            bounded_settings,
            transport=self.transport,
            retry_policy=RetryPolicy(
                max_attempts=1,
                base_delay_seconds=0,
                jitter_ratio=0,
            ),
            sleep=self.sleep,
            monotonic=self.monotonic,
            wall_time=self.wall_time,
            random_value=self.random_value,
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

        text, prompt_tokens, completion_tokens, telemetry = self._execute_payload(
            payload=payload,
            input_chars=input_chars,
            call_started=call_started,
            response_parser=self._parse_completion_response,
        )
        return ModelResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            telemetry=telemetry,
        )

    def complete_with_tools(
        self,
        system: str,
        user: str,
        *,
        tools: tuple[ToolDefinition, ...] | list[ToolDefinition],
        tool_choice: Literal["required"] | NamedToolChoice = "required",
        limits: ToolCallLimits | None = None,
    ) -> ToolCallResult:
        """Request and structurally validate native ``assistant.tool_calls``.

        This is intentionally separate from :meth:`complete`: callers cannot
        accidentally interpret ordinary JSON content as native tool calls, and
        no raw provider response, prompt, endpoint, or credential is retained in
        the returned value.  The contract always requires at least one call;
        final assistant text belongs to a separate, future Agent turn contract.
        Arguments are checked as bounded JSON objects but deliberately are not
        treated as schema-valid or authorized for execution at this layer.
        """
        call_started = self.monotonic()
        self.last_telemetry = None
        checked_limits = self._validate_tool_limits(limits or ToolCallLimits())
        tool_payloads, allowed_names, tool_payload_chars = self._tool_payloads(tools)
        choice_payload, forced_name = self._tool_choice_payload(
            tool_choice, allowed_names
        )
        input_chars = len(system) + len(user) + tool_payload_chars
        attempts: list[ProviderAttemptTelemetry] = []

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
            "tools": tool_payloads,
            "tool_choice": choice_payload,
        }
        if checked_limits.max_calls == 1:
            payload["parallel_tool_calls"] = False
        if self.settings.provider_max_completion_tokens is not None:
            payload["max_tokens"] = self.settings.provider_max_completion_tokens
        if self.settings.provider_thinking_mode is not None:
            payload["thinking"] = {"type": self.settings.provider_thinking_mode}

        def parse_tool_response(
            raw_body: bytes, attempt_no: int
        ) -> tuple[tuple[ProviderToolCall, ...], int, int, _Failure | None]:
            return self._parse_tool_response(
                raw_body,
                attempt_no,
                allowed_names=allowed_names,
                forced_name=forced_name,
                limits=checked_limits,
            )

        calls, prompt_tokens, completion_tokens, telemetry = self._execute_payload(
            payload=payload,
            input_chars=input_chars,
            call_started=call_started,
            response_parser=parse_tool_response,
            tool_contract=True,
        )
        return ToolCallResult(
            tool_calls=calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            telemetry=telemetry,
        )

    def _execute_payload(
        self,
        *,
        payload: dict[str, Any],
        input_chars: int,
        call_started: float,
        response_parser: Callable[
            [bytes, int], tuple[_ParsedResponse, int, int, _Failure | None]
        ],
        tool_contract: bool = False,
    ) -> tuple[_ParsedResponse, int, int, ProviderCallTelemetry]:
        attempts: list[ProviderAttemptTelemetry] = []
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
            # Advertise only the transfer coding that the bounded success-body
            # reader implements. Identity remains acceptable by HTTP default.
            "Accept-Encoding": "gzip",
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
                parsed_response: _ParsedResponse | None = None
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
                                (
                                    self._tool_http_failure_category(status)
                                    if tool_contract
                                    else self._http_failure_category(status)
                                ),
                                attempt_no,
                                http_status=status,
                                request_id=request_id,
                            )
                        else:
                            raw_body, received_bytes, failure = (
                                self._read_success_body(
                                    response, attempt_no, deadline=deadline
                                )
                            )
                            if failure is None:
                                assert raw_body is not None
                                response_chars = self._response_chars(raw_body)
                                (
                                    parsed_response,
                                    prompt_tokens,
                                    completion_tokens,
                                    failure,
                                ) = response_parser(raw_body, attempt_no)

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
                        assert parsed_response is not None
                        return (
                            parsed_response,
                            prompt_tokens,
                            completion_tokens,
                            telemetry,
                        )
                except httpx.ConnectTimeout:
                    failure = _Failure("connect_timeout", attempt_no, retryable=True)
                except httpx.ReadTimeout:
                    failure = _Failure("read_timeout", attempt_no, retryable=True)
                except httpx.DecodingError:
                    failure = _Failure("response_decompression", attempt_no)
                except httpx.RequestError:
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
        if failure.retryable:
            error_type: type[ProviderError] = ProviderRetryExhausted
        elif failure.category in {"unauthorized", "forbidden", "nonretry_http"}:
            error_type = ProviderError
        elif failure.category in {
            "unsupported_content_encoding",
            "response_decompression",
        }:
            # These are upstream representation/availability failures, not a
            # defect in a caller's native-tool schema or tool arguments.
            error_type = ProviderRetryExhausted
        elif tool_contract:
            error_type = ProviderToolCallError
        else:
            error_type = ProviderRetryExhausted
        raise error_type(
            message,
            category=failure.category,
            http_status=failure.http_status,
            attempt_no=failure.attempt_no,
            elapsed_ms=telemetry.elapsed_ms,
            telemetry=telemetry,
            request_id=failure.request_id,
        )

    def _parse_completion_response(
        self, raw_body: bytes, attempt_no: int
    ) -> tuple[str, int, int, _Failure | None]:
        body, failure = self._response_body(raw_body, attempt_no)
        if failure is not None:
            return "", 0, 0, failure
        return self._response_content(body, attempt_no)

    @staticmethod
    def _tool_input_error(category: str) -> ProviderToolCallError:
        return ProviderToolCallError(
            f"工具调用请求无效（category={category}）",
            category=category,
        )

    def _validate_tool_limits(self, limits: ToolCallLimits) -> ToolCallLimits:
        if not isinstance(limits, ToolCallLimits):
            raise self._tool_input_error("tool_limits_invalid")
        if (
            isinstance(limits.max_calls, bool)
            or not isinstance(limits.max_calls, int)
            or not 1 <= limits.max_calls <= _ABSOLUTE_MAX_TOOL_CALLS
        ):
            raise self._tool_input_error("tool_limits_invalid")
        if (
            isinstance(limits.max_argument_bytes, bool)
            or not isinstance(limits.max_argument_bytes, int)
            or not 1
            <= limits.max_argument_bytes
            <= _ABSOLUTE_MAX_TOOL_ARGUMENT_BYTES
        ):
            raise self._tool_input_error("tool_limits_invalid")
        return limits

    def _tool_payloads(
        self, tools: tuple[ToolDefinition, ...] | list[ToolDefinition]
    ) -> tuple[list[dict[str, Any]], frozenset[str], int]:
        if not isinstance(tools, (tuple, list)) or not 1 <= len(tools) <= _MAX_TOOL_DEFINITIONS:
            raise self._tool_input_error("tool_definitions_invalid")

        payloads: list[dict[str, Any]] = []
        names: set[str] = set()
        total_bytes = 0
        total_chars = 0
        for tool in tools:
            if not isinstance(tool, ToolDefinition):
                raise self._tool_input_error("tool_definition_invalid")
            if (
                not isinstance(tool.name, str)
                or _TOOL_NAME_PATTERN.fullmatch(tool.name) is None
                or tool.name in names
            ):
                raise self._tool_input_error("tool_definition_invalid")
            if (
                not isinstance(tool.description, str)
                or len(tool.description) > _MAX_TOOL_DESCRIPTION_CHARS
            ):
                raise self._tool_input_error("tool_definition_invalid")
            if (
                not isinstance(tool.parameters, dict)
                or tool.parameters.get("type") != "object"
                or not self._is_json_value(tool.parameters)
            ):
                raise self._tool_input_error("tool_definition_invalid")

            try:
                schema_json = json.dumps(
                    tool.parameters,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                schema_bytes = len(schema_json.encode("utf-8"))
                parameters = self._strict_json_loads(schema_json)
            except (TypeError, ValueError, UnicodeError, RecursionError):
                raise self._tool_input_error("tool_definition_invalid") from None
            if schema_bytes > _MAX_TOOL_SCHEMA_BYTES:
                raise self._tool_input_error("tool_definition_too_large")

            payload = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": parameters,
                },
            }
            try:
                payload_json = json.dumps(
                    payload,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                payload_bytes = len(payload_json.encode("utf-8"))
            except (TypeError, ValueError, UnicodeError, RecursionError):
                raise self._tool_input_error("tool_definition_invalid") from None
            total_bytes += payload_bytes
            total_chars += len(payload_json)
            if total_bytes > _MAX_TOOL_REQUEST_BYTES:
                raise self._tool_input_error("tool_definitions_too_large")
            names.add(tool.name)
            payloads.append(payload)
        return payloads, frozenset(names), total_chars

    def _tool_choice_payload(
        self,
        tool_choice: Literal["required"] | NamedToolChoice,
        allowed_names: frozenset[str],
    ) -> tuple[str | dict[str, Any], str | None]:
        if isinstance(tool_choice, str):
            if tool_choice != "required":
                raise self._tool_input_error("tool_choice_invalid")
            return tool_choice, None
        if not isinstance(tool_choice, NamedToolChoice):
            raise self._tool_input_error("tool_choice_invalid")
        if (
            not isinstance(tool_choice.name, str)
            or _TOOL_NAME_PATTERN.fullmatch(tool_choice.name) is None
        ):
            raise self._tool_input_error("tool_choice_invalid")
        if tool_choice.name not in allowed_names:
            raise self._tool_input_error("tool_choice_unknown")
        return (
            {"type": "function", "function": {"name": tool_choice.name}},
            tool_choice.name,
        )

    def _parse_tool_response(
        self,
        raw_body: bytes,
        attempt_no: int,
        *,
        allowed_names: frozenset[str],
        forced_name: str | None,
        limits: ToolCallLimits,
    ) -> tuple[tuple[ProviderToolCall, ...], int, int, _Failure | None]:
        try:
            body = self._strict_json_loads(raw_body)
        except (TypeError, ValueError, UnicodeError, RecursionError):
            return (), 0, 0, _Failure("tool_response_json", attempt_no)
        if not isinstance(body, dict):
            return (), 0, 0, _Failure("tool_response_shape", attempt_no)

        choices = body.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            return (), 0, 0, _Failure("tool_response_shape", attempt_no)
        choice = choices[0]
        if not isinstance(choice, dict):
            return (), 0, 0, _Failure("tool_response_shape", attempt_no)
        message = choice.get("message")
        if not isinstance(message, dict):
            return (), 0, 0, _Failure("tool_response_shape", attempt_no)

        prompt_tokens, completion_tokens, usage_failure = self._response_usage(
            body, attempt_no
        )
        if usage_failure is not None:
            return (), 0, 0, usage_failure

        finish_reason = choice.get("finish_reason")
        if finish_reason == "length":
            return (), prompt_tokens, completion_tokens, _Failure(
                "truncated", attempt_no
            )
        if finish_reason is not None and finish_reason not in {"tool_calls", "stop"}:
            return (), prompt_tokens, completion_tokens, _Failure(
                "tool_response_finish_reason", attempt_no
            )

        raw_calls = message.get("tool_calls")
        if raw_calls is None or raw_calls == []:
            return (), prompt_tokens, completion_tokens, _Failure(
                "tool_calls_missing", attempt_no
            )
        if not isinstance(raw_calls, list):
            return (), prompt_tokens, completion_tokens, _Failure(
                "tool_calls_shape", attempt_no
            )
        if len(raw_calls) > limits.max_calls:
            return (), prompt_tokens, completion_tokens, _Failure(
                "tool_calls_too_many", attempt_no
            )

        calls: list[ProviderToolCall] = []
        seen_ids: set[str] = set()
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict) or raw_call.get("type") != "function":
                return (), prompt_tokens, completion_tokens, _Failure(
                    "tool_call_shape", attempt_no
                )
            call_id = raw_call.get("id")
            if (
                not isinstance(call_id, str)
                or _TOOL_CALL_ID_PATTERN.fullmatch(call_id) is None
            ):
                return (), prompt_tokens, completion_tokens, _Failure(
                    "tool_call_id_invalid", attempt_no
                )
            if call_id in seen_ids:
                return (), prompt_tokens, completion_tokens, _Failure(
                    "tool_call_id_duplicate", attempt_no
                )

            function = raw_call.get("function")
            if not isinstance(function, dict):
                return (), prompt_tokens, completion_tokens, _Failure(
                    "tool_call_shape", attempt_no
                )
            name = function.get("name")
            if not isinstance(name, str) or _TOOL_NAME_PATTERN.fullmatch(name) is None:
                return (), prompt_tokens, completion_tokens, _Failure(
                    "tool_call_name_invalid", attempt_no
                )
            if name not in allowed_names:
                return (), prompt_tokens, completion_tokens, _Failure(
                    "tool_call_unknown", attempt_no
                )
            if forced_name is not None and name != forced_name:
                return (), prompt_tokens, completion_tokens, _Failure(
                    "tool_choice_mismatch", attempt_no
                )

            arguments, argument_failure = self._tool_arguments(
                function.get("arguments"), attempt_no, limits.max_argument_bytes
            )
            if argument_failure is not None:
                return (), prompt_tokens, completion_tokens, argument_failure
            assert arguments is not None
            seen_ids.add(call_id)
            calls.append(ProviderToolCall(id=call_id, name=name, arguments=arguments))

        return tuple(calls), prompt_tokens, completion_tokens, None

    def _tool_arguments(
        self, raw_arguments: Any, attempt_no: int, max_bytes: int
    ) -> tuple[dict[str, Any] | None, _Failure | None]:
        if isinstance(raw_arguments, str):
            try:
                argument_bytes = len(raw_arguments.encode("utf-8"))
            except UnicodeError:
                return None, _Failure("tool_call_arguments_json", attempt_no)
            if argument_bytes > max_bytes:
                return None, _Failure("tool_call_arguments_too_large", attempt_no)
            try:
                arguments = self._strict_json_loads(raw_arguments)
            except (TypeError, ValueError, UnicodeError, RecursionError):
                return None, _Failure("tool_call_arguments_json", attempt_no)
        elif isinstance(raw_arguments, dict):
            if not self._is_json_value(raw_arguments):
                return None, _Failure("tool_call_arguments_json", attempt_no)
            try:
                canonical = json.dumps(
                    raw_arguments,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                argument_bytes = len(canonical.encode("utf-8"))
                arguments = self._strict_json_loads(canonical)
            except (TypeError, ValueError, UnicodeError, RecursionError):
                return None, _Failure("tool_call_arguments_json", attempt_no)
            if argument_bytes > max_bytes:
                return None, _Failure("tool_call_arguments_too_large", attempt_no)
        else:
            return None, _Failure("tool_call_arguments_shape", attempt_no)

        if not isinstance(arguments, dict) or not self._is_json_value(arguments):
            return None, _Failure("tool_call_arguments_shape", attempt_no)
        return arguments, None

    @staticmethod
    def _response_usage(
        body: dict[str, Any], attempt_no: int
    ) -> tuple[int, int, _Failure | None]:
        usage = body.get("usage", {})
        if not isinstance(usage, dict):
            return 0, 0, _Failure("usage_shape", attempt_no)
        try:
            prompt_tokens = OpenAICompatibleProvider._token_count(
                usage.get("prompt_tokens", 0)
            )
            completion_tokens = OpenAICompatibleProvider._token_count(
                usage.get("completion_tokens", 0)
            )
        except (TypeError, ValueError, OverflowError):
            return 0, 0, _Failure("usage_shape", attempt_no)
        return prompt_tokens, completion_tokens, None

    @staticmethod
    def _strict_json_loads(value: str | bytes) -> Any:
        def object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = item
            return result

        def reject_constant(_: str) -> None:
            raise ValueError("non-finite JSON number")

        return json.loads(
            value,
            object_pairs_hook=object_without_duplicates,
            parse_constant=reject_constant,
        )

    @classmethod
    def _is_json_value(cls, value: Any, *, depth: int = 0) -> bool:
        if depth > 32:
            return False
        if value is None or isinstance(value, (str, bool, int)):
            return True
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, list):
            return all(cls._is_json_value(item, depth=depth + 1) for item in value)
        if isinstance(value, dict):
            return all(
                isinstance(key, str)
                and cls._is_json_value(item, depth=depth + 1)
                for key, item in value.items()
            )
        return False

    def _read_success_body(
        self,
        response: httpx.Response,
        attempt_no: int,
        *,
        deadline: float | None = None,
    ) -> tuple[bytes | None, int, _Failure | None]:
        encoding, encoding_failure = self._success_content_encoding(
            response, attempt_no
        )
        if encoding_failure is not None:
            response.close()
            return None, 0, encoding_failure

        configured_max = self.settings.provider_max_response_bytes
        max_bytes = min(
            configured_max
            if configured_max is not None
            else _ABSOLUTE_MAX_PROVIDER_RESPONSE_BYTES,
            _ABSOLUTE_MAX_PROVIDER_RESPONSE_BYTES,
        )
        if self._content_length_exceeds(response, max_bytes):
            response.close()
            return None, 0, _Failure(
                "response_too_large",
                attempt_no,
                http_status=response.status_code,
            )
        if deadline is not None and self.monotonic() >= deadline:
            response.close()
            return None, 0, _Failure(
                "read_timeout", attempt_no, retryable=True
            )

        # MockTransport commonly receives pre-buffered Response objects from
        # test handlers. HTTPX has already decoded an encoded response at this
        # point, so its original wire length, checksum and member boundary can
        # no longer be verified. Fail closed instead of trusting the decoded
        # cache; production responses take the raw bounded path below.
        if response.is_stream_consumed:
            if encoding != "identity":
                response.close()
                return None, 0, _Failure(
                    "response_decompression",
                    attempt_no,
                    http_status=response.status_code,
                )
            raw_body = response.content
            received_bytes = len(raw_body)
            if received_bytes > max_bytes:
                response.close()
                return None, received_bytes, _Failure(
                    "response_too_large",
                    attempt_no,
                    http_status=response.status_code,
                )
            return raw_body, received_bytes, None

        body = bytearray()
        received_bytes = 0
        decompressor = (
            zlib.decompressobj(zlib.MAX_WBITS | 16)
            if encoding == "gzip"
            else None
        )
        try:
            for chunk in response.iter_raw():
                if deadline is not None and self.monotonic() >= deadline:
                    body.clear()
                    response.close()
                    return None, received_bytes, _Failure(
                        "read_timeout", attempt_no, retryable=True
                    )
                received_bytes += len(chunk)
                if received_bytes > max_bytes:
                    body.clear()
                    response.close()
                    return None, received_bytes, _Failure(
                        "response_too_large",
                        attempt_no,
                        http_status=response.status_code,
                    )
                if decompressor is None:
                    decoded = chunk
                else:
                    # Ask zlib for at most one byte beyond the remaining
                    # decoded allowance.  This detects expansion beyond the
                    # ceiling without ever materializing an unbounded output.
                    remaining = max_bytes - len(body)
                    try:
                        decoded = decompressor.decompress(chunk, remaining + 1)
                    except zlib.error:
                        body.clear()
                        return None, received_bytes, _Failure(
                            "response_decompression",
                            attempt_no,
                            http_status=response.status_code,
                        )
                if len(body) + len(decoded) > max_bytes:
                    body.clear()
                    response.close()
                    return None, received_bytes, _Failure(
                        "response_too_large",
                        attempt_no,
                        http_status=response.status_code,
                    )
                body.extend(decoded)

                # A second gzip member or arbitrary trailing bytes are not a
                # second HTTP content encoding and are rejected rather than
                # being silently ignored.
                if decompressor is not None and decompressor.unused_data:
                    body.clear()
                    return None, received_bytes, _Failure(
                        "response_decompression",
                        attempt_no,
                        http_status=response.status_code,
                    )
                if deadline is not None and self.monotonic() >= deadline:
                    body.clear()
                    response.close()
                    return None, received_bytes, _Failure(
                        "read_timeout", attempt_no, retryable=True
                    )
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
        except httpx.DecodingError:
            body.clear()
            return None, received_bytes, _Failure(
                "response_decompression", attempt_no
            )
        except httpx.RequestError:
            body.clear()
            return None, received_bytes, _Failure(
                "transport", attempt_no, retryable=True
            )
        if deadline is not None and self.monotonic() >= deadline:
            body.clear()
            response.close()
            return None, received_bytes, _Failure(
                "read_timeout", attempt_no, retryable=True
            )
        if decompressor is not None and not decompressor.eof:
            body.clear()
            return None, received_bytes, _Failure(
                "response_decompression",
                attempt_no,
                http_status=response.status_code,
            )
        return bytes(body), received_bytes, None

    @staticmethod
    def _success_content_encoding(
        response: httpx.Response, attempt_no: int
    ) -> tuple[Literal["identity", "gzip"] | None, _Failure | None]:
        values = response.headers.get_list("content-encoding", split_commas=True)
        if not values:
            return "identity", None
        if len(values) != 1:
            return None, _Failure(
                "unsupported_content_encoding",
                attempt_no,
                http_status=response.status_code,
            )
        encoding = values[0].strip().lower()
        if encoding not in {"identity", "gzip"}:
            return None, _Failure(
                "unsupported_content_encoding",
                attempt_no,
                http_status=response.status_code,
            )
        return encoding, None

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
    def _tool_http_failure_category(status: int) -> str:
        if status == 401:
            return "unauthorized"
        if status == 403:
            return "forbidden"
        # Error bodies remain unread.  A safe gateway therefore reports only
        # that the native-tool request was rejected, not an invented claim that
        # a particular OpenAI-compatible extension is unsupported.
        return "tool_request_rejected"

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
