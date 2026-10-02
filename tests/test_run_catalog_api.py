from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app import main
from app.auth import AuthContext, get_auth_context
from app.db import (
    AnalysisDiagnosticRow,
    AnalysisRunExecutionRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    Base,
    ProjectRow,
    WorkspaceRow,
)


STATUSES = ("queued", "running", "completed", "failed", "cancelled")
SENTINEL = "CATALOG_PRIVATE_SYNTHETIC_SENTINEL"


def _forbidden_call(*_args, **_kwargs):
    raise AssertionError("Run catalog must not hydrate details, call models or dispatch tasks")


@pytest.fixture()
def catalog_api(tmp_path, monkeypatch):
    # conftest isolates imports from the developer .env; this fixture uses an
    # additional disposable DB and never starts the application lifespan.
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'run-catalog.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    with sessions() as db:
        workspace = WorkspaceRow(id=str(uuid4()), name="运行目录工作区")
        foreign_workspace = WorkspaceRow(id=str(uuid4()), name="另一工作区")
        db.add_all([workspace, foreign_workspace])
        db.flush()
        project = ProjectRow(workspace_id=workspace.id, name="运行目录项目")
        empty_project = ProjectRow(workspace_id=workspace.id, name="空项目")
        other_project = ProjectRow(workspace_id=workspace.id, name="另一项目")
        foreign_project = ProjectRow(workspace_id=foreign_workspace.id, name="另一工作区项目")
        db.add_all([project, empty_project, other_project, foreign_project])
        db.flush()
        start = datetime(2026, 1, 1, 8)
        runs = []
        for number in range(1, 68):
            row = AnalysisRunRow(
                id=str(UUID(int=number)), project_id=project.id,
                status=STATUSES[(number - 1) % len(STATUSES)],
                # Three rows share a timestamp: the ID tie-breaker matters.
                created_at=start + timedelta(minutes=number // 3),
                started_at=start if number % 2 == 0 else None,
                completed_at=start + timedelta(hours=1) if number % 3 == 0 else None,
                input_chars=number * 10, prompt_tokens=number * 2,
                completion_tokens=number, estimated_cost_usd=number / 100,
                provider_identity={"secret": SENTINEL},
                batch_coverage={"private_note": SENTINEL}, error=SENTINEL,
                batch_mode=("full_review", "baseline_build", "draft_review")[(number - 1) % 3],
            )
            runs.append(row)
        foreign_run = AnalysisRunRow(
            id=str(UUID(int=1000)), project_id=foreign_project.id, status="completed",
        )
        other_run = AnalysisRunRow(
            id=str(UUID(int=1001)), project_id=other_project.id, status="completed",
        )
        db.add_all([*runs, foreign_run, other_run])
        db.flush()
        for index in range(2):
            db.add(AnalysisRunInputRow(
                run_id=runs[-1].id, document_id=str(uuid4()),
                document_name=f"{SENTINEL}-{index}.md", document_version=1,
                content=SENTINEL, content_sha256="a" * 64, ordinal=index,
            ))
        db.add(AnalysisDiagnosticRow(run_id=runs[-1].id, payload={"private": SENTINEL}))
        db.add_all([
            AnalysisRunExecutionRow(run_id=runs[-1].id, retried_from_run_id=runs[0].id),
            AnalysisRunExecutionRow(run_id=runs[-2].id, retried_from_run_id=foreign_run.id),
            AnalysisRunExecutionRow(run_id=runs[-3].id, retried_from_run_id=other_run.id),
        ])
        db.commit()

    monkeypatch.setattr(main, "SessionLocal", sessions)
    monkeypatch.setitem(main.app.dependency_overrides, get_auth_context, lambda: AuthContext(
        user_id=str(uuid4()), workspace_id=workspace.id, role="owner", anonymous=False,
    ))
    original_serialize = main.serialize_run
    for name in (
        "serialize_run", "run_input_metadata", "OpenAICompatibleProvider",
        "account_provider_runtime", "dispatch_analysis", "execute_analysis",
    ):
        monkeypatch.setattr(main, name, _forbidden_call)
    state = SimpleNamespace(**locals(), reading=False, statements=[])

    @event.listens_for(engine, "before_cursor_execute")
    def only_select(_connection, _cursor, statement, _params, _context, _many):
        if state.reading:
            state.statements.append(statement)
            assert statement.lstrip().upper().startswith("SELECT"), statement

    state.client = TestClient(main.app)
    try:
        yield state
    finally:
        state.client.close()
        engine.dispose()


@contextmanager
def _read_only(state):
    state.reading = True
    try:
        yield
    finally:
        state.reading = False


def _get(state, suffix="", *, project_id=None):
    with _read_only(state):
        return state.client.get(
            f"/api/v1/projects/{project_id or state.project.id}/run-catalog{suffix}"
        )


def test_default_page_is_stable_bounded_and_metadata_only(catalog_api):
    state = catalog_api
    response = _get(state)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = response.json()
    assert {key: payload[key] for key in ("project_id", "page", "page_size", "total", "has_more")} == {
        "project_id": state.project.id, "page": 1, "page_size": 20,
        "total": 67, "has_more": True,
    }
    expected = sorted(state.runs, key=lambda row: (row.created_at, row.id), reverse=True)
    assert [row["id"] for row in payload["items"]] == [row.id for row in expected[:20]]
    assert SENTINEL not in response.text
    assert payload["items"][0]["frozen_document_count"] == 2
    assert payload["items"][1]["frozen_document_count"] == 0
    assert payload["items"][0]["retried_from"] == state.runs[0].id
    assert payload["items"][1]["retried_from"] is None
    assert payload["items"][2]["retried_from"] is None
    assert set(payload["items"][0]) == {
        "id", "project_id", "status", "created_at", "started_at", "completed_at",
        "input_chars", "prompt_tokens", "completion_tokens", "estimated_cost_usd",
        "frozen_document_count", "retried_from", "batch_mode",
    }
    selected_sql = "\n".join(state.statements).lower()
    for forbidden in (
        "analysis_runs.provider_identity", "analysis_runs.batch_coverage", "analysis_runs.error",
        "analysis_diagnostics", "analysis_run_inputs.content", "analysis_run_inputs.document_name",
        "documents.", "account_provider_configs",
    ):
        assert forbidden not in selected_sql
    assert len(state.statements) == 4


def test_pages_cover_every_run_once_and_out_of_range_is_empty(catalog_api):
    state = catalog_api
    identifiers = []
    for page in range(1, 5):
        payload = _get(state, f"?page={page}").json()
        identifiers.extend(row["id"] for row in payload["items"])
        assert payload["has_more"] is (page < 4)
        assert payload["total"] == 67
    assert len(identifiers) == len(set(identifiers)) == 67
    beyond = _get(state, "?page=100000").json()
    assert beyond["items"] == [] and beyond["total"] == 67 and beyond["has_more"] is False


@pytest.mark.parametrize("status", STATUSES)
def test_status_filter_counts_and_pages_only_matching_runs(catalog_api, status):
    state = catalog_api
    expected = sorted(
        (row for row in state.runs if row.status == status),
        key=lambda row: (row.created_at, row.id), reverse=True,
    )
    payload = _get(state, f"?status={status}&page_size=3&page=2").json()
    assert payload["total"] == len(expected)
    assert payload["page"] == 2 and payload["page_size"] == 3
    assert payload["has_more"] is (6 < len(expected))
    assert [row["id"] for row in payload["items"]] == [row.id for row in expected[3:6]]
    assert {row["status"] for row in payload["items"]} == {status}


def test_max_page_size_and_empty_owned_project(catalog_api):
    state = catalog_api
    largest = _get(state, "?page_size=50").json()
    assert len(largest["items"]) == 50 and largest["has_more"] is True
    empty = _get(state, project_id=state.empty_project.id)
    assert empty.status_code == 200
    assert empty.json()["total"] == 0 and empty.json()["items"] == []
    assert empty.json()["has_more"] is False


def test_invalid_legacy_status_remains_visible_as_unknown_without_new_filter(catalog_api):
    state = catalog_api
    with state.sessions() as db:
        row = db.get(AnalysisRunRow, state.runs[-1].id)
        row.status = "invalid-legacy-state"
        db.commit()
    response = _get(state)
    assert response.status_code == 200
    first = response.json()["items"][0]
    assert first["id"] == state.runs[-1].id and first["status"] == "unknown"
    assert response.json()["total"] == 67
    assert "invalid-legacy-state" not in response.text
    assert _get(state, "?status=unknown").status_code == 422


def test_missing_and_foreign_project_share_safe_not_found(catalog_api):
    state = catalog_api
    missing = _get(state, project_id=str(uuid4()))
    foreign = _get(state, project_id=state.foreign_project.id)
    assert missing.status_code == foreign.status_code == 404
    assert missing.json() == foreign.json() == {"detail": "项目不存在"}
    assert missing.headers["cache-control"] == foreign.headers["cache-control"] == "no-store"
    other = _get(state, project_id=state.other_project.id).json()
    assert other["total"] == 1 and other["items"][0]["id"] == state.other_run.id
    assert other["items"][0]["project_id"] == state.other_project.id


@pytest.mark.parametrize("suffix", (
    "?page=0", "?page=-1", "?page=100001", "?page=abc", "?page_size=0",
    "?page_size=51", "?page_size=1.5", "?status=unknown", "?status=COMPLETED",
))
def test_invalid_query_is_422_without_business_reads(catalog_api, suffix):
    state = catalog_api
    response = _get(state, suffix)
    assert response.status_code == 422
    assert response.headers["cache-control"] == "no-store"
    assert not state.statements
    assert SENTINEL not in response.text


def test_reported_counters_and_optional_cost_are_not_full_usage_claims(catalog_api, monkeypatch):
    state = catalog_api
    monkeypatch.setattr(main.settings, "model_input_price_per_million", None)
    monkeypatch.setattr(main.settings, "model_output_price_per_million", None)
    row = _get(state).json()["items"][0]
    assert row["prompt_tokens"] == 134 and row["completion_tokens"] == 67
    assert row["input_chars"] == 670
    assert row["estimated_cost_usd"] is None
    assert "usage_accounting" not in row and "model_execution" not in row
    monkeypatch.setattr(main.settings, "model_input_price_per_million", 1.0)
    monkeypatch.setattr(main.settings, "model_output_price_per_million", 1.0)
    priced = _get(state).json()["items"][0]
    assert priced["estimated_cost_usd"] == pytest.approx(0.67)


def test_legacy_all_runs_endpoint_still_returns_its_original_array(catalog_api, monkeypatch):
    state = catalog_api
    # Check compatibility without broad detail-hydration work. The legacy
    # endpoint remains responsible for its full serializer, unlike the catalog.
    monkeypatch.setattr(main, "serialize_run", lambda row, db: {"id": row.id, "status": row.status})
    with _read_only(state):
        response = state.client.get(f"/api/v1/projects/{state.project.id}/analysis-runs")
    assert response.status_code == 200
    assert isinstance(response.json(), list) and len(response.json()) == 67
    assert response.json()[0]["id"] == state.runs[-1].id
