"""Content-free diagnostics for exceptions at the two scope provider boundaries."""

from __future__ import annotations

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .provider import ProviderAttemptTelemetry, ProviderCallTelemetry, ProviderError


ScopeReviewContractFailure = Literal["response_invalid", "response_mismatch", "slot_conflict"]
ScopeFailureOrigin = Literal["limit_fork_failure", "complete_failure"]
ScopeFailureCategory = Literal[
    "provider", "not_configured", "rate_limit", "upstream_5xx", "unauthorized",
    "forbidden", "nonretry_http", "unsupported_content_encoding", "response_decompression",
    "body_json", "response_shape", "empty_content", "usage_shape", "truncated", "content_json",
    "connect_timeout", "read_timeout", "transport", "response_too_large", "deadline_exceeded",
    "credential_invalid", "credential_revoked", "endpoint_rejected", "other",
]
_SAFE_CATEGORIES = frozenset(get_args(ScopeFailureCategory))
_MAX_ELAPSED_MS = 86_400_000
_MAX_PROVIDER_ATTEMPTS = 4


class ScopeProviderAttemptV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    category: ScopeFailureCategory
    http_status: int | None = Field(default=None, ge=100, le=599)
    attempt_no: int | None = Field(default=None, ge=0, le=_MAX_PROVIDER_ATTEMPTS)
    elapsed_ms: int | None = Field(default=None, ge=0, le=_MAX_ELAPSED_MS)


class ScopeProviderFailureEventV1(ScopeProviderAttemptV1):
    purpose: Literal["scope_review"] = "scope_review"
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    origin: ScopeFailureOrigin
    round: Literal["initial", "contract_regeneration"]
    round_index: int = Field(ge=1, le=2)
    correction_reason: ScopeReviewContractFailure | None
    call_attempted: bool
    provider_attempts: tuple[ScopeProviderAttemptV1, ...] | None = Field(
        default=None, max_length=_MAX_PROVIDER_ATTEMPTS
    )
    provider_attempts_omitted: int = Field(default=0, ge=0, le=(1 << 63) - 1)

    @model_validator(mode="after")
    def _consistent_call(self):
        initial = self.round_index == 1
        if initial != (self.round == "initial") or initial != (self.correction_reason is None):
            raise ValueError("scope failure round is inconsistent")
        if self.call_attempted != (self.origin == "complete_failure"):
            raise ValueError("scope failure call flag is inconsistent")
        return self


class ScopeProviderFailureSeriesV1(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    events: tuple[ScopeProviderFailureEventV1, ...] = Field(default=(), max_length=2)
    omitted: int = Field(default=0, ge=0, le=2)


def _safe_int(value: object, minimum: int, maximum: int) -> int | None:
    return value if type(value) is int and minimum <= value <= maximum else None


def _safe_metrics(
    data: ProviderError | ProviderCallTelemetry | ProviderAttemptTelemetry,
    fallback: ProviderCallTelemetry | None = None,
) -> ScopeProviderAttemptV1:
    def metric(name: str, minimum: int, maximum: int) -> int | None:
        value = getattr(data, name, None)
        if value is None and fallback is not None:
            value = getattr(fallback, name, None)
        return _safe_int(value, minimum, maximum)

    category = getattr(data, "category", None)
    if category is None and fallback is not None:
        category = getattr(fallback, "category", None)
    return ScopeProviderAttemptV1(
        category=category if type(category) is str and category in _SAFE_CATEGORIES else "other",
        http_status=metric("http_status", 100, 599),
        attempt_no=metric("attempt_no", 0, _MAX_PROVIDER_ATTEMPTS),
        elapsed_ms=metric("elapsed_ms", 0, _MAX_ELAPSED_MS),
    )


def scope_provider_failure_event(
    exc: Exception, *, origin: ScopeFailureOrigin, source_sha256: str,
    round_index: int, correction_reason: ScopeReviewContractFailure | None,
) -> ScopeProviderFailureEventV1:
    metrics = ScopeProviderAttemptV1(category="other")
    attempts = None
    omitted = 0
    if isinstance(exc, ProviderError):
        telemetry = getattr(exc, "telemetry", None)
        if not isinstance(telemetry, ProviderCallTelemetry):
            telemetry = None
        metrics = _safe_metrics(exc, fallback=telemetry)
        raw_attempts = getattr(telemetry, "attempts", None)
        if type(raw_attempts) is list:
            attempts = tuple(
                _safe_metrics(item) for item in raw_attempts[:_MAX_PROVIDER_ATTEMPTS]
                if isinstance(item, ProviderAttemptTelemetry)
            )
            omitted = len(raw_attempts) - len(attempts)
    return ScopeProviderFailureEventV1(
        **metrics.model_dump(),
        source_sha256=source_sha256, origin=origin,
        round="initial" if round_index == 1 else "contract_regeneration",
        round_index=round_index, correction_reason=correction_reason,
        call_attempted=origin == "complete_failure",
        provider_attempts=attempts, provider_attempts_omitted=omitted,
    )
