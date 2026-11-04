from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Event, Thread
from types import SimpleNamespace

import httpx
import pytest

from app import character_trait_extraction as extraction
from scripts import habit_http_observer as probe


def entry(*, claim=False, habit=True):
    return SimpleNamespace(request=SimpleNamespace(
        target=SimpleNamespace(dimension="contextual_behavior" if habit else "preference"),
        primary_claim=object() if claim else None,
    ))


@contextmanager
def local_endpoint():
    release = Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            if self.path == "/hold":
                release.wait(5)
                return
            self.send_response(200)
            self.send_header("Content-Length", "10" if self.path == "/truncated" else "2")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = Thread(target=server.serve_forever)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        worker.join()


def test_real_posts_record_effective_transport_limits_and_completed_streams(monkeypatch):
    observer = probe.HabitHttpObserver()
    chained_events = []
    original_stream = httpx.Client.stream
    with local_endpoint() as base, httpx.Client(trust_env=False) as client:
        def review(entries, path):
            if path == "budget":
                return
            with client.stream(
                "POST", base + path, content=b"secret-story-sentinel",
                headers={"Authorization": "test-private-key"},
                timeout=httpx.Timeout(.08 if path == "/hold" else 3, connect=1),
                extensions={"trace": lambda name, info: chained_events.append(name)},
            ) as response:
                assert response.read() == b"ok"

        monkeypatch.setattr(extraction, "run_target_bound_draft_review", review)
        with observer.capture(), ThreadPoolExecutor(max_workers=2) as executor:
            executor.submit(extraction.run_target_bound_draft_review, (entry(), entry()), "/ok").result()
            executor.submit(extraction.run_target_bound_draft_review, (entry(habit=False),), "/private").result()
            extraction.run_target_bound_draft_review((entry(),), "budget")
            with pytest.raises(httpx.ReadTimeout):
                extraction.run_target_bound_draft_review((entry(claim=True),), "/hold")
            with pytest.raises(httpx.RemoteProtocolError):
                extraction.run_target_bound_draft_review((entry(),), "/truncated")
        assert httpx.Client.stream is original_stream
        assert extraction.run_target_bound_draft_review is review

    report = observer.snapshot()
    assert report["streams_without_observed_post"] == report["omitted_attempts"] == 0
    completed, timed_out, truncated = report["attempts"]
    assert (completed["original_record_entries"], completed["window_entries"]) == (0, 2)
    assert (timed_out["original_record_entries"], timed_out["window_entries"]) == (1, 0)
    assert completed["read_timeout_ms"] == 3000 and timed_out["read_timeout_ms"] == 80
    assert completed["connect_timeout_ms"] == timed_out["connect_timeout_ms"] == 1000
    assert completed["response_headers_received"] and completed["response_body_complete"]
    assert timed_out["request_body_sent"] and not timed_out["response_headers_received"]
    assert not timed_out["response_body_complete"]
    assert truncated["response_headers_received"] and not truncated["response_body_complete"]
    assert all(row["elapsed_ms"] >= 0 for row in report["attempts"])
    assert "http11.receive_response_body.complete" in chained_events
    assert not any(word in json.dumps(report) for word in ("sentinel", "private", "http://"))


def test_untraced_stream_is_not_a_wire_attempt_and_receipt_stays_bounded(monkeypatch):
    observer = probe.HabitHttpObserver()
    monkeypatch.setattr(probe, "_MAX_EVENTS", 1)
    with local_endpoint() as base, httpx.Client(trust_env=False) as real, \
            httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200))) as silent:
        def review(entries, client):
            with client.stream("POST", base + "/ok", content=b"x") as response:
                response.read()
        monkeypatch.setattr(extraction, "run_target_bound_draft_review", review)
        with observer.capture():
            extraction.run_target_bound_draft_review((entry(),), silent)
            for _ in range(3):
                extraction.run_target_bound_draft_review((entry(),), real)
    report = observer.snapshot()
    assert len(report["attempts"]) == 1
    assert report["streams_without_observed_post"] == 1
    assert report["omitted_attempts"] == 2
