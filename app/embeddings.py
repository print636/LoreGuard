from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

import httpx

from .config import Settings


_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class EmbeddingProfile:
    """Versioned identity of vectors that may safely share an index."""

    profile_id: str
    provider_kind: str
    provider_namespace: str
    model_identifier: str
    model_revision: str
    dimensions: int
    normalized: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.provider_kind, str)
            or self.provider_kind != self.provider_kind.strip()
            or not self.provider_kind
            or len(self.provider_kind) > 40
        ):
            raise ValueError("embedding provider kind is invalid")
        if (
            not isinstance(self.provider_namespace, str)
            or self.provider_namespace != self.provider_namespace.strip()
            or not self.provider_namespace
            or len(self.provider_namespace) > 80
        ):
            raise ValueError("embedding provider namespace is invalid")
        if (
            not isinstance(self.model_identifier, str)
            or self.model_identifier != self.model_identifier.strip()
            or not self.model_identifier
            or len(self.model_identifier) > 255
        ):
            raise ValueError("embedding model identifier is invalid")
        if (
            not isinstance(self.model_revision, str)
            or self.model_revision != self.model_revision.strip()
            or not self.model_revision
            or len(self.model_revision) > 120
        ):
            raise ValueError("embedding model revision is invalid")
        if (
            isinstance(self.dimensions, bool)
            or not isinstance(self.dimensions, int)
            or not (1 <= self.dimensions <= 16_000)
        ):
            raise ValueError("embedding dimensions are invalid")
        if not isinstance(self.normalized, bool):
            raise ValueError("embedding normalization flag is invalid")
        expected_id = _embedding_profile_id(
            provider_kind=self.provider_kind,
            provider_namespace=self.provider_namespace,
            model_identifier=self.model_identifier,
            model_revision=self.model_revision,
            dimensions=self.dimensions,
            normalized=self.normalized,
        )
        if self.profile_id != expected_id:
            raise ValueError("embedding profile id is invalid")

    @classmethod
    def openai_compatible(
        cls,
        *,
        provider_namespace: str = "default",
        model_identifier: str,
        model_revision: str,
        dimensions: int,
    ) -> "EmbeddingProfile":
        return cls(
            profile_id=_embedding_profile_id(
                provider_kind="openai-compatible",
                provider_namespace=provider_namespace,
                model_identifier=model_identifier,
                model_revision=model_revision,
                dimensions=dimensions,
                normalized=True,
            ),
            provider_kind="openai-compatible",
            provider_namespace=provider_namespace,
            model_identifier=model_identifier,
            model_revision=model_revision,
            dimensions=dimensions,
            normalized=True,
        )


@dataclass(frozen=True)
class EmbeddingUsage:
    prompt_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class EmbeddingAttemptTelemetry:
    attempt: int
    elapsed_ms: int
    outcome: str
    status_code: int | None = None
    response_bytes: int | None = None
    request_id: str | None = None


@dataclass(frozen=True)
class EmbeddingCallTelemetry:
    profile_id: str
    input_count: int
    input_chars: int
    elapsed_ms: int
    outcome: str
    attempts: tuple[EmbeddingAttemptTelemetry, ...] = ()


@dataclass(frozen=True)
class EmbeddingResult:
    profile: EmbeddingProfile
    vectors: tuple[tuple[float, ...], ...] = field(repr=False)
    usage: EmbeddingUsage
    telemetry: EmbeddingCallTelemetry


class EmbeddingProvider(Protocol):
    @property
    def profile(self) -> EmbeddingProfile:
        ...

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        ...


class EmbeddingProviderError(RuntimeError):
    """Safe provider failure whose message never includes upstream content."""

    def __init__(
        self, message: str, *, telemetry: EmbeddingCallTelemetry | None = None
    ) -> None:
        super().__init__(message)
        self.telemetry = telemetry


class EmbeddingNotConfiguredError(EmbeddingProviderError):
    pass


class EmbeddingInputError(EmbeddingProviderError):
    pass


class EmbeddingResponseError(EmbeddingProviderError):
    pass


class EmbeddingRetryExhaustedError(EmbeddingProviderError):
    pass


class OpenAICompatibleEmbeddingProvider:
    """Bounded OpenAI-compatible ``/embeddings`` client.

    The implementation deliberately owns no fallback to the chat provider.  A
    deployment must explicitly enable and configure the embedding capability.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.BaseTransport | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._monotonic = monotonic
        self._sleeper = sleeper
        self._profile: EmbeddingProfile | None = None

    @property
    def profile(self) -> EmbeddingProfile:
        if self._profile is not None:
            return self._profile
        settings = self._settings
        if not settings.enable_embeddings:
            raise EmbeddingNotConfiguredError("embedding provider is disabled")
        if not settings.embedding_model.strip():
            raise EmbeddingNotConfiguredError("embedding model is not configured")
        revision = settings.embedding_model_revision.strip()
        if not revision or revision.lower() == "unspecified":
            raise EmbeddingNotConfiguredError("embedding model revision is not configured")
        if settings.embedding_dimensions is None:
            raise EmbeddingNotConfiguredError("embedding dimensions are not configured")
        self._validated_endpoint()
        self._profile = EmbeddingProfile.openai_compatible(
            provider_namespace=(
                settings.embedding_profile_namespace.strip() or "default"
            ),
            model_identifier=settings.embedding_model.strip(),
            model_revision=revision,
            dimensions=settings.embedding_dimensions,
        )
        return self._profile

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        profile = self.profile
        prepared = self._validate_inputs(texts)
        settings = self._settings
        started = self._monotonic()
        deadline = started + settings.embedding_total_deadline_seconds
        attempts: list[EmbeddingAttemptTelemetry] = []
        last_retryable = False

        for attempt_no in range(1, settings.embedding_max_attempts + 1):
            remaining = deadline - self._monotonic()
            if remaining <= 0:
                break
            attempt_started = self._monotonic()
            response: httpx.Response | None = None
            try:
                timeout = min(settings.embedding_timeout_seconds, remaining)
                with httpx.Client(
                    transport=self._transport,
                    timeout=httpx.Timeout(timeout),
                    follow_redirects=False,
                ) as client:
                    headers = {
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Accept-Encoding": "identity",
                    }
                    if settings.embedding_api_key.strip():
                        headers["Authorization"] = (
                            f"Bearer {settings.embedding_api_key.strip()}"
                        )
                    request = client.build_request(
                        "POST",
                        self._validated_endpoint(),
                        headers=headers,
                        json={"model": profile.model_identifier, "input": prepared},
                    )
                    response = client.send(request, stream=True)
                    request_id = _safe_request_id(response.headers.get("x-request-id"))
                    status_code = response.status_code
                    if 300 <= status_code < 400:
                        attempts.append(
                            self._attempt(
                                attempt_no, attempt_started, "redirect_rejected",
                                status_code=status_code, request_id=request_id,
                            )
                        )
                        raise self._failure(
                            EmbeddingResponseError,
                            "embedding redirects are not permitted",
                            started,
                            prepared,
                            profile,
                            attempts,
                            "failed",
                        )
                    if status_code >= 400:
                        last_retryable = status_code in _RETRYABLE_STATUS
                        attempts.append(
                            self._attempt(
                                attempt_no,
                                attempt_started,
                                "retryable_status" if last_retryable else "rejected",
                                status_code=status_code,
                                request_id=request_id,
                            )
                        )
                        if not last_retryable:
                            raise self._failure(
                                EmbeddingResponseError,
                                "embedding request was rejected",
                                started,
                                prepared,
                                profile,
                                attempts,
                                "failed",
                            )
                        delay = self._retry_delay(response, attempt_no)
                        continue_retry = attempt_no < settings.embedding_max_attempts
                        if continue_retry and self._bounded_sleep(delay, deadline):
                            continue
                        break

                    try:
                        body = self._read_bounded_success(response, deadline)
                        if self._monotonic() >= deadline:
                            raise _EmbeddingDeadlineExceeded
                        vectors, usage = self._parse_success(body, len(prepared), profile)
                        if self._monotonic() >= deadline:
                            raise _EmbeddingDeadlineExceeded
                    except _EmbeddingDeadlineExceeded:
                        attempts.append(
                            self._attempt(
                                attempt_no,
                                attempt_started,
                                "deadline_exceeded",
                                status_code=status_code,
                                request_id=request_id,
                            )
                        )
                        raise self._failure(
                            EmbeddingRetryExhaustedError,
                            "embedding deadline was exceeded",
                            started,
                            prepared,
                            profile,
                            attempts,
                            "failed",
                        ) from None
                    except EmbeddingResponseError as exc:
                        attempts.append(
                            self._attempt(
                                attempt_no,
                                attempt_started,
                                "invalid_response",
                                status_code=status_code,
                                request_id=request_id,
                            )
                        )
                        raise self._failure(
                            EmbeddingResponseError,
                            str(exc),
                            started,
                            prepared,
                            profile,
                            attempts,
                            "failed",
                        ) from None
                    attempts.append(
                        self._attempt(
                            attempt_no,
                            attempt_started,
                            "success",
                            status_code=status_code,
                            response_bytes=len(body),
                            request_id=request_id,
                        )
                    )
                    telemetry = self._call_telemetry(
                        started, prepared, profile, attempts, "success"
                    )
                    return EmbeddingResult(
                        profile=profile,
                        vectors=vectors,
                        usage=usage,
                        telemetry=telemetry,
                    )
            except EmbeddingProviderError:
                raise
            except httpx.RequestError:
                last_retryable = True
                attempts.append(
                    self._attempt(attempt_no, attempt_started, "transport_error")
                )
                if attempt_no < settings.embedding_max_attempts:
                    if self._bounded_sleep(self._retry_delay(None, attempt_no), deadline):
                        continue
                break
            finally:
                if response is not None:
                    response.close()

        message = (
            "embedding retries were exhausted"
            if last_retryable
            else "embedding deadline was exceeded"
        )
        raise self._failure(
            EmbeddingRetryExhaustedError,
            message,
            started,
            prepared,
            profile,
            attempts,
            "failed",
        )

    def _validate_inputs(self, texts: Sequence[str]) -> tuple[str, ...]:
        if isinstance(texts, (str, bytes)):
            raise EmbeddingInputError("embedding input must be a sequence of text")
        try:
            prepared = tuple(texts)
        except TypeError:
            raise EmbeddingInputError(
                "embedding input must be a sequence of text"
            ) from None
        if not prepared:
            raise EmbeddingInputError("embedding input must not be empty")
        if len(prepared) > self._settings.embedding_batch_max_items:
            raise EmbeddingInputError("embedding input item limit was exceeded")
        if any(not isinstance(item, str) or not item.strip() for item in prepared):
            raise EmbeddingInputError("embedding inputs must be non-empty text")
        if sum(len(item) for item in prepared) > self._settings.embedding_batch_max_chars:
            raise EmbeddingInputError("embedding input character limit was exceeded")
        return prepared

    def _validated_endpoint(self) -> str:
        raw = self._settings.embedding_base_url.strip()
        try:
            parts = urlsplit(raw)
            if (
                parts.scheme not in {"http", "https"}
                or not parts.hostname
                or parts.username is not None
                or parts.password is not None
                or parts.query
                or parts.fragment
            ):
                raise ValueError
            if parts.scheme == "http" and not self._settings.embedding_allow_insecure_http:
                raise ValueError
            path = parts.path.rstrip("/") + "/embeddings"
            endpoint = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
            parsed = httpx.URL(endpoint)
            if not parsed.is_absolute_url or parsed.host is None:
                raise ValueError
            return str(parsed)
        except (httpx.InvalidURL, TypeError, ValueError):
            raise EmbeddingNotConfiguredError("embedding base URL is invalid") from None

    def _read_bounded_success(self, response: httpx.Response, deadline: float) -> bytes:
        limit = self._settings.embedding_max_response_bytes
        raw_length = response.headers.get("content-length")
        if raw_length:
            try:
                if int(raw_length) > limit:
                    raise EmbeddingResponseError("embedding response exceeded byte limit")
            except ValueError:
                raise EmbeddingResponseError("embedding response metadata is invalid") from None
        media_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type and media_type != "application/json":
            raise EmbeddingResponseError("embedding response type is invalid")
        content_encoding = response.headers.get("content-encoding", "").strip().lower()
        if content_encoding not in {"", "identity"}:
            raise EmbeddingResponseError("embedding response encoding is invalid")
        body = bytearray()
        for chunk in response.iter_bytes():
            if self._monotonic() >= deadline:
                raise _EmbeddingDeadlineExceeded
            body.extend(chunk)
            if len(body) > limit:
                raise EmbeddingResponseError("embedding response exceeded byte limit")
        return bytes(body)

    def _parse_success(
        self, body: bytes, expected_count: int, profile: EmbeddingProfile
    ) -> tuple[tuple[tuple[float, ...], ...], EmbeddingUsage]:
        try:
            payload = json.loads(body)
        except (RecursionError, UnicodeError, ValueError):
            raise EmbeddingResponseError("embedding response JSON is invalid") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise EmbeddingResponseError("embedding response shape is invalid")
        rows = payload["data"]
        if len(rows) != expected_count:
            raise EmbeddingResponseError("embedding response count is invalid")
        ordered: list[tuple[float, ...] | None] = [None] * expected_count
        for row in rows:
            if not isinstance(row, dict):
                raise EmbeddingResponseError("embedding response item is invalid")
            index = row.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise EmbeddingResponseError("embedding response index is invalid")
            if index < 0 or index >= expected_count or ordered[index] is not None:
                raise EmbeddingResponseError("embedding response index is invalid")
            vector = row.get("embedding")
            if not isinstance(vector, list) or len(vector) != profile.dimensions:
                raise EmbeddingResponseError("embedding vector dimension is invalid")
            converted: list[float] = []
            for value in vector:
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise EmbeddingResponseError("embedding vector value is invalid")
                try:
                    number = float(value)
                except (OverflowError, ValueError):
                    raise EmbeddingResponseError(
                        "embedding vector value is invalid"
                    ) from None
                if not math.isfinite(number):
                    raise EmbeddingResponseError("embedding vector value is invalid")
                converted.append(number)
            norm = math.sqrt(sum(value * value for value in converted))
            if not math.isfinite(norm) or norm <= 0:
                raise EmbeddingResponseError("embedding vector norm is invalid")
            ordered[index] = tuple(value / norm for value in converted)
        if any(vector is None for vector in ordered):
            raise EmbeddingResponseError("embedding response index is invalid")

        usage_payload = payload.get("usage")
        usage = EmbeddingUsage()
        if isinstance(usage_payload, dict):
            usage = EmbeddingUsage(
                prompt_tokens=_safe_nonnegative_int(usage_payload.get("prompt_tokens")),
                total_tokens=_safe_nonnegative_int(usage_payload.get("total_tokens")),
            )
        return tuple(vector for vector in ordered if vector is not None), usage

    def _retry_delay(self, response: httpx.Response | None, attempt_no: int) -> float:
        if response is not None:
            value = response.headers.get("retry-after", "").strip()
            try:
                parsed = float(value)
                if math.isfinite(parsed):
                    return min(max(parsed, 0.0), 2.0)
            except ValueError:
                pass
        return min(0.1 * (2 ** (attempt_no - 1)), 1.0)

    def _bounded_sleep(self, delay: float, deadline: float) -> bool:
        remaining = deadline - self._monotonic()
        if remaining <= 0 or delay >= remaining:
            return False
        self._sleeper(delay)
        return self._monotonic() < deadline

    def _attempt(
        self,
        attempt_no: int,
        started: float,
        outcome: str,
        *,
        status_code: int | None = None,
        response_bytes: int | None = None,
        request_id: str | None = None,
    ) -> EmbeddingAttemptTelemetry:
        return EmbeddingAttemptTelemetry(
            attempt=attempt_no,
            elapsed_ms=max(0, int((self._monotonic() - started) * 1000)),
            outcome=outcome,
            status_code=status_code,
            response_bytes=response_bytes,
            request_id=request_id,
        )

    def _call_telemetry(
        self,
        started: float,
        texts: tuple[str, ...],
        profile: EmbeddingProfile,
        attempts: list[EmbeddingAttemptTelemetry],
        outcome: str,
    ) -> EmbeddingCallTelemetry:
        return EmbeddingCallTelemetry(
            profile_id=profile.profile_id,
            input_count=len(texts),
            input_chars=sum(len(text) for text in texts),
            elapsed_ms=max(0, int((self._monotonic() - started) * 1000)),
            outcome=outcome,
            attempts=tuple(attempts),
        )

    def _failure(
        self,
        error_type: type[EmbeddingProviderError],
        message: str,
        started: float,
        texts: tuple[str, ...],
        profile: EmbeddingProfile,
        attempts: list[EmbeddingAttemptTelemetry],
        outcome: str,
    ) -> EmbeddingProviderError:
        return error_type(
            message,
            telemetry=self._call_telemetry(started, texts, profile, attempts, outcome),
        )


def _safe_request_id(value: str | None) -> str | None:
    if not value or len(value) > 128:
        return None
    if not all(character.isalnum() or character in "-_." for character in value):
        return None
    return f"rid-{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _safe_nonnegative_int(value: object) -> int | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > 10_000_000_000
    ):
        return None
    return value


class _EmbeddingDeadlineExceeded(Exception):
    pass


def _embedding_profile_id(
    *,
    provider_kind: str,
    provider_namespace: str,
    model_identifier: str,
    model_revision: str,
    dimensions: int,
    normalized: bool,
) -> str:
    canonical = json.dumps(
        {
            "provider_kind": provider_kind,
            "provider_namespace": provider_namespace,
            "model_identifier": model_identifier,
            "model_revision": model_revision,
            "dimensions": dimensions,
            "normalized": normalized,
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"emb-{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"
