import unittest
from unittest.mock import patch
from types import SimpleNamespace

from app.db import AnalysisDiagnosticRow
from app.service import analysis_mode, execute_analysis
from app.tasks import analyze_project


class CeleryTaskTests(unittest.TestCase):
    def test_analysis_mode_distinguishes_complete_partial_and_baseline(self):
        def result(**overrides):
            status = dict(enabled=True, configured=True, succeeded_chunks=1,
                          failed_chunks=0, skipped_chunks=0, invalid_records=0)
            status.update(overrides)
            return SimpleNamespace(diagnostics={"model": status}, warnings=[])

        complete = result()
        partial = result(skipped_chunks=1)
        failed = result(succeeded_chunks=0, failed_chunks=1)
        baseline = result(enabled=False, configured=False, succeeded_chunks=0)
        self.assertEqual(("完整模型增强", False), analysis_mode(complete))
        self.assertEqual(("模型增强（部分分块已降级）", True), analysis_mode(partial))
        self.assertEqual(("确定性基线（模型未参与或已降级）", True), analysis_mode(failed))
        self.assertEqual(("确定性基线", False), analysis_mode(baseline))
        self.assertEqual(
            ("执行状态未知（缺少结构化记录）", True),
            analysis_mode(SimpleNamespace(model_used=True, warnings=[])),
        )

    def test_worker_requests_service_failure_reraise_for_autoretry(self):
        task_body = analyze_project.run
        with patch("app.tasks.execute_analysis") as execute:
            task_body("run-123")
        execute.assert_called_once_with("run-123", raise_on_failure=True)

    def test_service_preserves_structured_execution_counts_in_persisted_diagnostic(self):
        run = SimpleNamespace(cancel_requested=False, project_id="project-123")
        execution = dict(enabled=True, configured=True, total_chunks=2,
                         attempted_chunks=1, succeeded_chunks=1, failed_chunks=0,
                         skipped_chunks=1, invalid_records=0,
                         reason_codes=["token_budget"], documents=[{"document_id": "doc"}])
        result = SimpleNamespace(
            directives=[], issues=[], warnings=[], prompt_tokens=12, completion_tokens=8,
            model_used=True, diagnostics={"model": execution.copy()},
        )
        with (
            patch("app.service.SessionLocal") as session_factory,
            patch("app.service.AnalysisPipeline") as pipeline_factory,
            patch("app.service.emit"),
        ):
            db = session_factory.return_value.__enter__.return_value
            db.get.return_value = run
            db.scalars.return_value.all.return_value = []
            pipeline_factory.return_value.run.return_value = result
            execute_analysis("run-123", raise_on_failure=True)
        persisted = next(
            call.args[0] for call in db.add.call_args_list
            if isinstance(call.args[0], AnalysisDiagnosticRow)
        )
        payload = persisted.payload["model"]
        for key, value in execution.items():
            self.assertEqual(value, payload[key])
        self.assertEqual("模型增强（部分分块已降级）", payload["mode"])
        self.assertTrue(payload["partial_fallback"])

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
