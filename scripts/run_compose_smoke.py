from __future__ import annotations

import argparse
import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


EXPECTED_CATEGORIES = {
    "fact_conflict",
    "location_collision",
    "knowledge_without_acquisition",
    "item_ownership",
    "world_rule_conflict",
}


def request_json(base_url: str, path: str, method: str = "GET"):
    request = Request(f"{base_url}{path}", method=method)
    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_until_ready(base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if request_json(base_url, "/health").get("status") == "ok":
                return
        except (URLError, TimeoutError, ValueError) as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"API did not become ready: {type(last_error).__name__}")


def run(base_url: str, timeout_seconds: float) -> dict:
    wait_until_ready(base_url, timeout_seconds)
    project = request_json(base_url, "/api/v1/demo/advanced", "POST")
    run_row = request_json(
        base_url, f"/api/v1/projects/{project['id']}/analysis-runs", "POST"
    )
    deadline = time.monotonic() + timeout_seconds
    status = "queued"
    while time.monotonic() < deadline:
        run_row = request_json(base_url, f"/api/v1/analysis-runs/{run_row['id']}")
        status = run_row["status"]
        if status in {"completed", "failed", "cancelled"}:
            break
        time.sleep(1)
    if status != "completed":
        raise RuntimeError(f"Celery-backed analysis ended as {status}")

    run_id = run_row["id"]
    issues = request_json(base_url, f"/api/v1/analysis-runs/{run_id}/issues")
    categories = {issue["category"] for issue in issues}
    if categories != EXPECTED_CATEGORIES:
        raise RuntimeError(
            f"Unexpected issue categories: {sorted(categories)}"
        )
    graph = request_json(base_url, f"/api/v1/analysis-runs/{run_id}/graph")
    timeline = request_json(base_url, f"/api/v1/analysis-runs/{run_id}/timeline")
    timeline_entry_count = sum(len(group["entries"]) for group in timeline["groups"])
    timeline_entry_count += len(timeline["unscheduled"])
    with urlopen(f"{base_url}/api/v1/analysis-runs/{run_id}/events", timeout=10) as response:
        events = response.read().decode("utf-8")
    if "event: terminal" not in events or '"status": "completed"' not in events:
        raise RuntimeError("SSE stream did not expose the completed terminal state")
    if not graph["nodes"] or not timeline_entry_count:
        raise RuntimeError("Completed run projections are unexpectedly empty")
    return {
        "status": status,
        "document_count": project["document_count"],
        "issue_count": len(issues),
        "categories": sorted(categories),
        "graph_nodes": len(graph["nodes"]),
        "timeline_entries": timeline_entry_count,
        "sse_terminal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.base_url.rstrip("/"), args.timeout_seconds),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
