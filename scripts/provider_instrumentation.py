from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.provider import OpenAICompatibleProvider, RetryPolicy


@dataclass(slots=True)
class _ProviderCallCounts:
    requested: int = 0
    succeeded: int = 0
    failed: int = 0
    last_error_type: str | None = None


class CountingProvider:
    """Content-free logical-call counters shared with bounded repair children."""

    def __init__(
        self,
        delegate: OpenAICompatibleProvider,
        *,
        _counts: _ProviderCallCounts | None = None,
    ) -> None:
        self.delegate = delegate
        self.settings = delegate.settings
        self._counts = _counts or _ProviderCallCounts()

    @property
    def transport(self):
        return self.delegate.transport

    @property
    def sleep(self):
        return self.delegate.sleep

    @property
    def monotonic(self):
        return self.delegate.monotonic

    @property
    def wall_time(self):
        return self.delegate.wall_time

    @property
    def random_value(self):
        return self.delegate.random_value

    @property
    def requested(self) -> int:
        return self._counts.requested

    @property
    def succeeded(self) -> int:
        return self._counts.succeeded

    @property
    def failed(self) -> int:
        return self._counts.failed

    @property
    def last_error_type(self) -> str | None:
        return self._counts.last_error_type

    @property
    def configured(self) -> bool:
        return self.delegate.configured

    def fork_for_repair(self, settings: Any) -> CountingProvider:
        """Create a bounded child without losing the parent counter state."""
        child = OpenAICompatibleProvider(
            settings,
            transport=self.transport,
            retry_policy=RetryPolicy(max_attempts=1, base_delay_seconds=0),
            sleep=self.sleep,
            monotonic=self.monotonic,
            wall_time=self.wall_time,
            random_value=self.random_value,
        )
        return CountingProvider(child, _counts=self._counts)

    def complete(self, system: str, user: str):
        self._counts.requested += 1
        try:
            result = self.delegate.complete(system, user)
        except Exception as exc:
            self._counts.failed += 1
            self._counts.last_error_type = type(exc).__name__
            raise
        self._counts.succeeded += 1
        return result
