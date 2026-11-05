"""Server-owned timing limits and a shared, non-renewable run deadline."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings

EXECUTION_LIMITS_KEY = "execution_limits"
EXECUTION_LIMITS_SCHEMA = "execution-limits-v1"

_READ_FIELDS = (
    "provider_timeout_seconds", "character_signal_timeout_seconds",
    "character_signal_scope_review_timeout_seconds",
    "character_history_semantic_review_timeout_seconds",
    "character_draft_actor_review_timeout_seconds", "character_drift_timeout_seconds",
    "character_habit_scene_timeout_seconds",
)
_LOGICAL_FIELDS = (
    "provider_total_deadline_seconds", "character_signal_total_deadline_seconds",
    "character_drift_total_deadline_seconds", "character_habit_scene_total_deadline_seconds",
)


class AnalysisDeadlineExceeded(RuntimeError):
    """The analysis run's shared wall-clock allowance has expired."""


@dataclass(frozen=True)
class ExecutionDeadline:
    deadline: float
    monotonic: Callable[[], float] = field(repr=False, compare=False)

    @classmethod
    def from_remaining(cls, seconds: float, *, monotonic=time.perf_counter):
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
            raise ValueError("invalid analysis remaining time")
        return cls(monotonic() + max(0.0, float(seconds)), monotonic)

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline - self.monotonic())

    def checkpoint(self) -> None:
        if self.remaining_seconds() <= 0:
            raise AnalysisDeadlineExceeded("analysis execution deadline exceeded")


def _timing_values(read: float, logical: float, run: float) -> dict:
    return {
        **dict.fromkeys(_READ_FIELDS, read),
        **dict.fromkeys(_LOGICAL_FIELDS, logical),
        "analysis_run_timeout_seconds": run,
        "execution_timing_profile": "local_quality",
    }


def local_quality_timing_overrides() -> dict:
    """Explicit factory arguments; ordinary deployment defaults remain unchanged."""
    return _timing_values(180.0, 360.0, 2400.0)


def execution_limits_payload(settings: Settings) -> dict | None:
    if settings.execution_timing_profile != "local_quality":
        return None
    payload = {
        "schema_version": EXECUTION_LIMITS_SCHEMA,
        "read_seconds": settings.provider_timeout_seconds,
        "logical_seconds": settings.provider_total_deadline_seconds,
        "run_seconds": settings.analysis_run_timeout_seconds,
    }
    _validated_limits(payload)
    return payload


def _validated_limits(payload: object) -> tuple[float, float, float]:
    if (not isinstance(payload, dict) or set(payload) != {
        "schema_version", "read_seconds", "logical_seconds", "run_seconds"
    } or payload["schema_version"] != EXECUTION_LIMITS_SCHEMA):
        raise ValueError("invalid execution limits snapshot")
    values = tuple(payload[key] for key in ("read_seconds", "logical_seconds", "run_seconds"))
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("invalid execution limits snapshot")
    read, logical, run = map(float, values)
    if not (read <= logical <= run and read <= 180 and logical <= 360 and run <= 2400):
        raise ValueError("execution limits snapshot exceeds profile bounds")
    return read, logical, run


def settings_for_execution_limits(settings: Settings, payload: object) -> Settings:
    """Restore new frozen limits, or the legacy factory defaults for old runs."""
    if payload is None:
        if settings.execution_timing_profile != "local_quality":
            return settings
        from .config import Settings
        updates = {name: Settings.model_fields[name].default for name in (
            *_READ_FIELDS, *_LOGICAL_FIELDS, "analysis_run_timeout_seconds", "execution_timing_profile"
        )}
    else:
        read, logical, run = _validated_limits(payload)
        updates = _timing_values(read, logical, run)
        # Deployment limits may tighten a frozen run; they may never extend it.
        for name in (*_READ_FIELDS, *_LOGICAL_FIELDS, "analysis_run_timeout_seconds"):
            current = getattr(settings, name)
            if current is not None:
                updates[name] = min(updates[name], current)
    restored = settings.model_copy(update=updates)
    restored._execution_deadline = None
    return restored


def remaining_execution_seconds(settings: Settings, allowance: float) -> float:
    context = settings._execution_deadline
    if context is None:
        return float(allowance)
    context.checkpoint()
    return min(float(allowance), context.remaining_seconds())
