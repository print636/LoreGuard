from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from prometheus_client.parser import text_string_to_metric_families
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from app.db import AnalysisRunRow, Base, DocumentRow, ProjectRow, RunEventRow
from app.main import app
from app.observability import AnalysisMetricsUnavailable, render_analysis_metrics
from app.service import execute_analysis


def _sample_value(
    payload: str,
    name: str,
    **labels: str,
) -> float:
    for family in text_string_to_metric_families(payload):
        for sample in family.samples:
            if sample.name == name and sample.labels == labels:
                return float(sample.value)
    raise AssertionError(f"missing metric sample {name} with labels {labels}")


class DatabaseAnalysisMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "metrics.db"
        database_url = f"sqlite:///{database_path.as_posix()}"
        self.writer_engine = create_engine(database_url)
        self.reader_engine = create_engine(database_url)
        Base.metadata.create_all(self.writer_engine)
        self.WriterSession = sessionmaker(
            bind=self.writer_engine,
            expire_on_commit=False,
        )
        self.ReaderSession = sessionmaker(
            bind=self.reader_engine,
            expire_on_commit=False,
        )

    def tearDown(self) -> None:
        self.reader_engine.dispose()
        self.writer_engine.dispose()
        self.temp_dir.cleanup()

    def _seed_runs(self) -> None:
        started = datetime(2026, 9, 12, 8, 0, 0)
        secret = "sk-private-key https://private-provider.invalid/v1"
        with self.WriterSession() as db:
            project = ProjectRow(name="never-export-this-project")
            db.add(project)
            db.flush()
            db.add(
                DocumentRow(
                    project_id=project.id,
                    name="never-export-this-file.md",
                    content="never-export-this-story-content",
                )
            )
            runs = [
                AnalysisRunRow(project_id=project.id, status="queued"),
                AnalysisRunRow(
                    project_id=project.id,
                    status="running",
                    started_at=started,
                ),
                AnalysisRunRow(
                    project_id=project.id,
                    status="completed",
                    started_at=started,
                    completed_at=started + timedelta(seconds=0.4),
                ),
                AnalysisRunRow(
                    project_id=project.id,
                    status="completed",
                    started_at=started,
                    completed_at=started + timedelta(seconds=12),
                ),
                AnalysisRunRow(
                    project_id=project.id,
                    status="failed",
                    started_at=started,
                    completed_at=started + timedelta(seconds=3),
                    error=secret,
                ),
                AnalysisRunRow(
                    project_id=project.id,
                    status="cancelled",
                    completed_at=started,
                ),
                AnalysisRunRow(
                    project_id=project.id,
                    status="cancelled",
                    started_at=started,
                    completed_at=started - timedelta(seconds=1),
                ),
                AnalysisRunRow(
                    project_id=project.id,
                    status="never-export-this-status",
                ),
            ]
            db.add_all(runs)
            db.flush()
            terminal_events = (
                (2, "completed"),
                (4, "failed"),
                (5, "cancelled"),
                (6, "cancelled"),
            )
            for index, stage in terminal_events:
                db.add(
                    RunEventRow(
                        run_id=runs[index].id,
                        stage=stage,
                        progress=100,
                        message=f"terminal {stage}",
                    )
                )
            # Duplicate terminal events must not inflate a lifecycle counter.
            db.add(
                RunEventRow(
                    run_id=runs[2].id,
                    stage="completed",
                    progress=100,
                    message="duplicate terminal completed",
                )
            )
            # An event that contradicts the current state is not exported as a
            # terminal transition or latency sample.
            db.add(
                RunEventRow(
                    run_id=runs[3].id,
                    stage="failed",
                    progress=100,
                    message="mismatched stale event",
                )
            )
            db.commit()

    def test_metrics_are_database_derived_bounded_and_content_free(self) -> None:
        self._seed_runs()

        payload = render_analysis_metrics(self.ReaderSession).decode("utf-8")
        repeated_payload = render_analysis_metrics(self.ReaderSession).decode(
            "utf-8"
        )

        self.assertEqual(
            8,
            _sample_value(
                payload,
                "loreguard_analysis_runs_total",
            ),
        )
        self.assertEqual(
            2,
            _sample_value(
                payload,
                "loreguard_analysis_runs_current",
                status="completed",
            ),
        )
        self.assertEqual(
            1,
            _sample_value(
                payload,
                "loreguard_analysis_runs_current",
                status="failed",
            ),
        )
        self.assertEqual(
            _sample_value(payload, "loreguard_analysis_runs_total"),
            _sample_value(repeated_payload, "loreguard_analysis_runs_total"),
        )
        self.assertEqual(
            1,
            _sample_value(
                payload,
                "loreguard_analysis_run_terminal_transitions_total",
                status="completed",
            ),
        )
        self.assertEqual(
            1,
            _sample_value(
                payload,
                "loreguard_analysis_run_terminal_transitions_total",
                status="failed",
            ),
        )
        self.assertEqual(
            2,
            _sample_value(
                payload,
                "loreguard_analysis_run_terminal_transitions_total",
                status="cancelled",
            ),
        )
        self.assertEqual(
            0,
            _sample_value(
                payload,
                "loreguard_analysis_duration_unavailable",
                status="completed",
            ),
        )
        self.assertEqual(
            1,
            _sample_value(
                payload,
                "loreguard_analysis_runs_current",
                status="unknown",
            ),
        )
        self.assertEqual(
            1,
            _sample_value(
                payload,
                "loreguard_analysis_seconds_count",
                status="completed",
            ),
        )
        self.assertAlmostEqual(
            0.4,
            _sample_value(
                payload,
                "loreguard_analysis_seconds_sum",
                status="completed",
            ),
        )
        self.assertEqual(
            1,
            _sample_value(
                payload,
                "loreguard_analysis_seconds_bucket",
                status="completed",
                le="0.5",
            ),
        )
        self.assertEqual(
            1,
            _sample_value(
                payload,
                "loreguard_analysis_seconds_bucket",
                status="completed",
                le="30",
            ),
        )
        self.assertEqual(
            2,
            _sample_value(
                payload,
                "loreguard_analysis_duration_unavailable",
                status="cancelled",
            ),
        )
        self.assertEqual(
            1,
            _sample_value(
                payload,
                "loreguard_analysis_seconds_count",
                status="failed",
            ),
        )
        self.assertEqual(
            3,
            _sample_value(
                payload,
                "loreguard_analysis_seconds_sum",
                status="failed",
            ),
        )
        self.assertEqual(
            1,
            _sample_value(
                payload,
                "loreguard_analysis_metrics_database_available",
            ),
        )
        for forbidden in (
            "never-export-this-project",
            "never-export-this-file",
            "never-export-this-story-content",
            "never-export-this-status",
            "sk-private-key",
            "private-provider.invalid",
        ):
            self.assertNotIn(forbidden, payload)

        expected_current_statuses = {
            "queued",
            "running",
            "completed",
            "failed",
            "cancelled",
            "unknown",
        }
        expected_terminal_statuses = {"completed", "failed", "cancelled"}
        for family in text_string_to_metric_families(payload):
            for sample in family.samples:
                if sample.name == "loreguard_analysis_runs_current":
                    self.assertEqual({"status"}, set(sample.labels))
                    self.assertIn(
                        sample.labels["status"],
                        expected_current_statuses,
                    )
                if sample.name.startswith(
                    "loreguard_analysis_run_terminal_transitions"
                ):
                    self.assertEqual({"status"}, set(sample.labels))
                    self.assertIn(
                        sample.labels["status"],
                        expected_terminal_statuses,
                    )
                if sample.name.startswith("loreguard_analysis_seconds_"):
                    self.assertIn(
                        sample.labels["status"],
                        expected_terminal_statuses,
                    )

    def test_database_error_is_replaced_with_safe_public_category(self) -> None:
        secret = "postgresql://user:secret@private-db.invalid/loreguard"

        def broken_session():
            raise SQLAlchemyError(secret)

        with self.assertRaises(AnalysisMetricsUnavailable) as raised:
            render_analysis_metrics(broken_session)

        self.assertEqual(
            "analysis metrics database unavailable",
            str(raised.exception),
        )
        self.assertNotIn("secret", str(raised.exception))

    def test_new_database_connection_and_exporter_restart_keep_db_state(self) -> None:
        self._seed_runs()

        first_payload = render_analysis_metrics(self.ReaderSession).decode("utf-8")
        self.reader_engine.dispose()
        database_url = str(self.writer_engine.url)
        restarted_engine = create_engine(database_url)
        RestartedSession = sessionmaker(
            bind=restarted_engine,
            expire_on_commit=False,
        )
        try:
            restarted_payload = render_analysis_metrics(RestartedSession).decode(
                "utf-8"
            )
        finally:
            restarted_engine.dispose()

        for name, labels in (
            ("loreguard_analysis_runs_total", {}),
            (
                "loreguard_analysis_run_terminal_transitions_total",
                {"status": "completed"},
            ),
            ("loreguard_analysis_runs_current", {"status": "failed"}),
            ("loreguard_analysis_seconds_count", {"status": "completed"}),
            ("loreguard_analysis_seconds_sum", {"status": "completed"}),
        ):
            self.assertEqual(
                _sample_value(first_payload, name, **labels),
                _sample_value(restarted_payload, name, **labels),
            )


class MetricsEndpointTests(unittest.TestCase):
    def test_successful_endpoint_has_prometheus_content_type_and_db_metrics(self) -> None:
        with TestClient(app) as client:
            response = client.get("/metrics")

        self.assertEqual(200, response.status_code)
        self.assertIn("text/plain", response.headers["content-type"])
        self.assertIn("version=0.0.4", response.headers["content-type"])
        self.assertIn("loreguard_analysis_runs_total", response.text)
        self.assertIn(
            "loreguard_analysis_metrics_database_available 1",
            response.text,
        )

    def test_worker_terminal_commit_is_visible_to_api_metrics(self) -> None:
        with TestClient(app) as client:
            before = client.get("/metrics").text
            project = client.post(
                "/api/v1/projects",
                json={"name": "metrics worker boundary"},
            ).json()
            document = client.post(
                f"/api/v1/projects/{project['id']}/documents/text",
                json={
                    "name": "chapter.md",
                    "content": "林澈在第一日抵达北港。",
                },
            )
            self.assertEqual(201, document.status_code)
            with patch("app.main.dispatch_analysis"):
                started = client.post(
                    f"/api/v1/projects/{project['id']}/analysis-runs"
                )
            self.assertEqual(202, started.status_code)

            execute_analysis(started.json()["id"])
            after = client.get("/metrics").text

        comparisons = (
            ("loreguard_analysis_runs_total", {}),
            (
                "loreguard_analysis_run_terminal_transitions_total",
                {"status": "completed"},
            ),
            ("loreguard_analysis_runs_current", {"status": "completed"}),
            ("loreguard_analysis_seconds_count", {"status": "completed"}),
        )
        for name, labels in comparisons:
            self.assertEqual(
                _sample_value(before, name, **labels) + 1,
                _sample_value(after, name, **labels),
            )

    def test_database_failure_returns_503_without_internal_details(self) -> None:
        secret = "postgresql://user:secret@private-db.invalid/loreguard"
        with patch(
            "app.main.render_analysis_metrics",
            side_effect=AnalysisMetricsUnavailable(secret),
        ):
            with TestClient(app) as client:
                response = client.get("/metrics")

        self.assertEqual(503, response.status_code)
        self.assertTrue(response.headers["content-type"].startswith("text/plain"))
        self.assertIn(
            "loreguard_analysis_metrics_database_available 0",
            response.text,
        )
        self.assertNotIn("secret", response.text)
        self.assertNotIn("private-db", response.text)


if __name__ == "__main__":
    unittest.main()
