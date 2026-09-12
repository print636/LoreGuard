from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite
from typing import Callable

from prometheus_client import CollectorRegistry, generate_latest
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily
from prometheus_client.core import HistogramMetricFamily
from sqlalchemy import and_, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from .db import AnalysisRunRow, RunEventRow


RUN_STATES = ("queued", "running", "completed", "failed", "cancelled")
EXPORTED_RUN_STATES = (*RUN_STATES, "unknown")
TERMINAL_RUN_STATES = ("completed", "failed", "cancelled")
DURATION_BUCKETS_SECONDS = (0.5, 1, 2, 5, 10, 30, 60, 90, 120, 300, 600, 1800)


class AnalysisMetricsUnavailable(RuntimeError):
    """Raised without database details when persisted metrics cannot be read."""


@dataclass(frozen=True)
class DurationSnapshot:
    bucket_counts: tuple[int, ...]
    count: int
    sum_seconds: float
    unavailable: int


@dataclass(frozen=True)
class AnalysisMetricsSnapshot:
    submitted: int
    current_by_status: dict[str, int]
    terminal_transitions_by_status: dict[str, int]
    duration_by_status: dict[str, DurationSnapshot]


@dataclass
class _DurationAccumulator:
    bucket_counts: list[int] = field(
        default_factory=lambda: [0] * len(DURATION_BUCKETS_SECONDS)
    )
    count: int = 0
    sum_seconds: float = 0.0
    unavailable: int = 0


def _duration_seconds(
    started_at: datetime | None,
    completed_at: datetime | None,
) -> float | None:
    if started_at is None or completed_at is None:
        return None
    try:
        value = (completed_at - started_at).total_seconds()
    except (OverflowError, TypeError):
        return None
    if value < 0 or not isfinite(value):
        return None
    return value


def load_analysis_metrics(
    session_factory: Callable[[], Session],
) -> AnalysisMetricsSnapshot:
    """Build a content-free metrics snapshot from the shared application DB."""

    current = {status: 0 for status in EXPORTED_RUN_STATES}
    terminal_transitions = {status: 0 for status in TERMINAL_RUN_STATES}
    duration_accumulators = {
        status: _DurationAccumulator() for status in TERMINAL_RUN_STATES
    }
    try:
        with session_factory() as db:
            terminal_events = (
                select(RunEventRow.run_id, RunEventRow.stage)
                .where(RunEventRow.stage.in_(TERMINAL_RUN_STATES))
                .distinct()
                .subquery()
            )
            rows = db.execute(
                select(
                    AnalysisRunRow.status,
                    AnalysisRunRow.started_at,
                    AnalysisRunRow.completed_at,
                    terminal_events.c.stage,
                )
                .outerjoin(
                    terminal_events,
                    and_(
                        terminal_events.c.run_id == AnalysisRunRow.id,
                        terminal_events.c.stage == AnalysisRunRow.status,
                    ),
                )
            ).all()
    except SQLAlchemyError:
        # Do not put a database URL, driver message, SQL text, or deployment
        # detail into the public scrape response.
        raise AnalysisMetricsUnavailable("analysis metrics database unavailable") from None

    for raw_status, started_at, completed_at, terminal_event_status in rows:
        current_status = raw_status if raw_status in RUN_STATES else "unknown"
        current[current_status] += 1
        if terminal_event_status not in TERMINAL_RUN_STATES:
            continue
        terminal_transitions[terminal_event_status] += 1
        accumulator = duration_accumulators[terminal_event_status]
        duration = _duration_seconds(started_at, completed_at)
        if duration is None:
            accumulator.unavailable += 1
            continue
        accumulator.count += 1
        accumulator.sum_seconds += duration
        for index, upper_bound in enumerate(DURATION_BUCKETS_SECONDS):
            if duration <= upper_bound:
                accumulator.bucket_counts[index] += 1

    durations = {}
    for status, accumulator in duration_accumulators.items():
        durations[status] = DurationSnapshot(
            bucket_counts=tuple(accumulator.bucket_counts),
            count=accumulator.count,
            sum_seconds=accumulator.sum_seconds,
            unavailable=accumulator.unavailable,
        )
    return AnalysisMetricsSnapshot(
        submitted=len(rows),
        current_by_status=current,
        terminal_transitions_by_status=terminal_transitions,
        duration_by_status=durations,
    )


class AnalysisRunMetricsCollector:
    """Render one immutable database snapshot as bounded Prometheus families."""

    def __init__(self, snapshot: AnalysisMetricsSnapshot):
        self.snapshot = snapshot

    def collect(self):
        lifecycle = CounterMetricFamily(
            "loreguard_analysis_runs",
            "Analysis-run records accepted in the current database lifecycle.",
        )
        lifecycle.add_metric([], self.snapshot.submitted)
        yield lifecycle

        terminal = CounterMetricFamily(
            "loreguard_analysis_run_terminal_transitions",
            (
                "Distinct persisted terminal run events corroborated by the "
                "run's matching current terminal state."
            ),
            labels=["status"],
        )
        for status in TERMINAL_RUN_STATES:
            terminal.add_metric(
                [status], self.snapshot.terminal_transitions_by_status[status]
            )
        yield terminal

        current = GaugeMetricFamily(
            "loreguard_analysis_runs_current",
            "Analysis-run records currently stored in each bounded state.",
            labels=["status"],
        )
        for status in EXPORTED_RUN_STATES:
            current.add_metric([status], self.snapshot.current_by_status[status])
        yield current

        duration = HistogramMetricFamily(
            "loreguard_analysis_seconds",
            (
                "Elapsed time from first worker claim to a persisted terminal "
                "event matching the current terminal state, including "
                "retry/backoff waits, for valid intervals."
            ),
            labels=["status"],
        )
        unavailable = GaugeMetricFamily(
            "loreguard_analysis_duration_unavailable",
            (
                "Terminal analysis-run records excluded from the duration "
                "histogram because a matching terminal event has missing or "
                "invalid timing fields."
            ),
            labels=["status"],
        )
        for status in TERMINAL_RUN_STATES:
            snapshot = self.snapshot.duration_by_status[status]
            buckets = [
                (format(upper_bound, "g"), count)
                for upper_bound, count in zip(
                    DURATION_BUCKETS_SECONDS,
                    snapshot.bucket_counts,
                    strict=True,
                )
            ]
            buckets.append(("+Inf", snapshot.count))
            duration.add_metric([status], buckets, snapshot.sum_seconds)
            unavailable.add_metric([status], snapshot.unavailable)
        yield duration
        yield unavailable

        database_available = GaugeMetricFamily(
            "loreguard_analysis_metrics_database_available",
            "Whether this analysis metrics scrape was derived from the database.",
        )
        database_available.add_metric([], 1)
        yield database_available


def render_analysis_metrics(session_factory: Callable[[], Session]) -> bytes:
    snapshot = load_analysis_metrics(session_factory)
    registry = CollectorRegistry()
    registry.register(AnalysisRunMetricsCollector(snapshot))
    return generate_latest(registry)
