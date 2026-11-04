"""Opt-in, in-process HTTP evidence for isolated author-acceptance runs.

Use `with observer.capture():` around the entire run, including worker completion.
Capture in the process running the character stage. A separate Celery process is
outside its scope. snapshot() contains only counts, timing and transport flags.
Install any other review wrappers before entering capture(). The first 64 POST
writes confirmed by httpcore are kept in send order. An untraced transport or
a failure before the POST headers were written is counted separately. Rejection
before entering the HTTP client does not produce either sort of receipt.

Timeouts are the per-operation values on the actual HTTP request. elapsed_ms is
measured from starting to write its headers through stream close (None while
in flight). The header/body flags describe the configured HTTP endpoint only.
An HTTP response is independent of the model's semantic decision.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from functools import wraps
from math import isfinite
from threading import Lock
from time import monotonic
from unittest.mock import patch

import httpx

from app import character_trait_extraction


_CAPTURE_LOCK = Lock()
_MAX_EVENTS = 64


@dataclass
class _Attempt:
    original_record_entries: int
    window_entries: int
    read_timeout_ms: float | None
    connect_timeout_ms: float | None
    elapsed_ms: int | None = None
    request_body_sent: bool = False
    response_headers_received: bool = False
    response_body_complete: bool = False


def _timeout_ms(value: object) -> float | None:
    if type(value) not in (float, int) or not isfinite(value) or value < 0:
        return None
    return round(1000 * value, 3)


class HabitHttpObserver:
    def __init__(self):
        self._lock = Lock()
        self._attempts: list[_Attempt] = []
        self._omitted = 0
        self._unconfirmed = 0

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "schema_version": "habit-http-observer-v1",
                "attempts": [asdict(row) for row in self._attempts],
                "omitted_attempts": self._omitted,
                "streams_without_observed_post": self._unconfirmed,
            }

    @contextmanager
    def capture(self):
        """Install scoped probes; unrelated requests and worker threads stay untouched."""
        if not _CAPTURE_LOCK.acquire(blocking=False):
            raise RuntimeError("an isolated HTTP observation is already active")
        try:
            stage = character_trait_extraction.run_target_bound_draft_review
            stream = httpx.Client.stream
            scope = ContextVar("habit_http_scope", default=None)

            @wraps(stage)
            def review(entries, *args, **kwargs):
                habits = [entry.request for entry in entries if entry.request.target.dimension == "contextual_behavior"]
                originals = sum(req.primary_claim is not None for req in habits)
                token = scope.set((originals, len(habits) - originals) if habits else None)
                try:
                    return stage(entries, *args, **kwargs)
                finally:
                    scope.reset(token)

            @contextmanager
            @wraps(stream)
            def traced_stream(client, method, url, **kwargs):
                current = scope.get()
                if current is None or method.upper() != "POST":
                    with stream(client, method, url, **kwargs) as response:
                        yield response
                    return
                extensions = dict(kwargs.get("extensions") or {})
                existing_trace = extensions.get("trace")
                observed = False
                started = None
                active = None
                read_ms = connect_ms = None

                def trace(event, info):
                    nonlocal started, active, observed, read_ms, connect_ms
                    if event.startswith(("http11.", "http2.")):
                        action = event.split(".", 1)[1]
                        with self._lock:
                            if action == "send_request_headers.started":
                                if active is not None:
                                    active.elapsed_ms = round((monotonic() - started) * 1000)
                                request = info.get("request")
                                # A proxy CONNECT is not a model POST.
                                started = monotonic() if getattr(request, "method", None) == b"POST" else None
                                active = None
                                if started is not None:
                                    limits = request.extensions.get("timeout") or {}
                                    read_ms = _timeout_ms(limits.get("read"))
                                    connect_ms = _timeout_ms(limits.get("connect"))
                            elif started is not None and action == "send_request_headers.complete":
                                observed = True
                                if len(self._attempts) < _MAX_EVENTS:
                                    active = _Attempt(*current, read_ms, connect_ms)
                                    self._attempts.append(active)
                                else:
                                    self._omitted += 1
                            elif active is not None:
                                if action == "send_request_body.complete":
                                    active.request_body_sent = True
                                elif action == "receive_response_headers.complete":
                                    active.response_headers_received = True
                                elif action == "receive_response_body.complete":
                                    active.response_body_complete = True
                    if existing_trace is not None:
                        existing_trace(event, info)

                extensions["trace"] = trace
                try:
                    with stream(client, method, url, **{**kwargs, "extensions": extensions}) as response:
                        yield response
                finally:
                    with self._lock:
                        if active is not None:
                            active.elapsed_ms = round((monotonic() - started) * 1000)
                        if not observed:
                            self._unconfirmed += 1

            with patch.object(character_trait_extraction, "run_target_bound_draft_review", review), \
                    patch.object(httpx.Client, "stream", traced_stream):
                yield self
        finally:
            _CAPTURE_LOCK.release()
