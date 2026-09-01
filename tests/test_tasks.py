import unittest
from unittest.mock import patch
from types import SimpleNamespace

from app.service import execute_analysis
from app.tasks import analyze_project


class CeleryTaskTests(unittest.TestCase):
    def test_worker_requests_service_failure_reraise_for_autoretry(self):
        task_body = analyze_project.run
        with patch("app.tasks.execute_analysis") as execute:
            task_body("run-123")
        execute.assert_called_once_with("run-123", raise_on_failure=True)

    def test_service_persists_failure_then_reraises_for_worker(self):
        run = SimpleNamespace(
            cancel_requested=False,
            status="queued",
            started_at=None,
            completed_at=None,
            error=None,
            project_id="project-123",
            input_chars=0,
        )
        with (
            patch("app.service.SessionLocal") as session_factory,
            patch("app.service.AnalysisPipeline") as pipeline_factory,
            patch("app.service.emit") as emit,
        ):
            db = session_factory.return_value.__enter__.return_value
            db.get.return_value = run
            db.scalars.return_value.all.return_value = []
            pipeline_factory.return_value.run.side_effect = RuntimeError("retry me")
            with self.assertRaisesRegex(RuntimeError, "retry me"):
                execute_analysis("run-123", raise_on_failure=True)

        self.assertEqual("failed", run.status)
        self.assertEqual("retry me", run.error)
        self.assertIsNotNone(run.completed_at)
        emit.assert_called_with(
            db, "run-123", "failed", 100, "分析失败，可调用重试接口恢复"
        )


if __name__ == "__main__":
    unittest.main()
