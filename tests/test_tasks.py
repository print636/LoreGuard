import os
import unittest
from unittest.mock import patch
from types import SimpleNamespace

# Worker unit tests must not load a real provider from the repository .env.
os.environ["ENABLE_MODEL_EXTRACTION"] = "false"
os.environ["OPENAI_API_KEY"] = ""

from celery.exceptions import Retry

from app.domain import AnalysisCancelled
from app.service import WorkerLeaseBusy, analysis_mode, execute_analysis
from app.tasks import analyze_project


class CeleryTaskTests(unittest.TestCase):
    def test_analysis_mode_distinguishes_complete_partial_and_baseline(self):
        def result(**overrides):
            status = dict(enabled=True, configured=True, succeeded_chunks=1,
                          failed_chunks=0, skipped_chunks=0, invalid_records=0,
                          empty_response_chunks=0)
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
            ("模型返回空结果，无法证明完整覆盖", True),
            analysis_mode(result(empty_response_chunks=1)),
        )
        self.assertEqual(
            ("执行状态未知（缺少结构化记录）", True),
            analysis_mode(SimpleNamespace(model_used=True, warnings=[])),
        )
        self.assertEqual(
            ("执行状态未知（缺少结构化记录）", True),
            analysis_mode(result(**{"empty_response_chunks": None})),
        )
        self.assertEqual(
            ("执行状态未知（结构化记录不一致）", True),
            analysis_mode(result(succeeded_chunks=1, empty_response_chunks=2)),
        )
        repaired = result(
            invalid_records=4,
            unresolved_invalid_records=0,
            recovered_invalid_records=4,
            repair_succeeded=True,
            repair_failed=False,
            repair_post_invalid=0,
        )
        self.assertEqual(("完整模型增强", False), analysis_mode(repaired))
        unresolved = result(
            invalid_records=1,
            unresolved_invalid_records=1,
            recovered_invalid_records=0,
            repair_failed=True,
            repair_post_invalid=1,
        )
        self.assertEqual(
            ("模型增强（部分分块已降级）", True), analysis_mode(unresolved)
        )
        # Missing final-disposition fields use the legacy invalid_records rule.
        self.assertEqual(
            ("模型增强（部分分块已降级）", True),
            analysis_mode(result(invalid_records=1)),
        )
        inconsistent = result(
            invalid_records=4,
            unresolved_invalid_records=1,
            recovered_invalid_records=2,
            repair_failed=False,
            repair_post_invalid=0,
        )
        self.assertEqual(
            ("执行状态未知（无效记录诊断不一致）", True),
            analysis_mode(inconsistent),
        )

    def test_worker_requests_service_failure_reraise_for_autoretry(self):
        task_body = analyze_project.run
        with patch("app.tasks.execute_analysis") as execute:
            task_body("run-123")
        execute.assert_called_once_with(
            "run-123",
            raise_on_failure=True,
            finalize_failure=False,
            worker_token=None,
            raise_on_busy=True,
        )

    def test_service_does_not_construct_pipeline_when_claim_is_rejected(self):
        with (
            patch("app.service.claim_analysis_run", return_value=False),
            patch("app.service.AnalysisPipeline") as pipeline_factory,
        ):
            execute_analysis("run-123", raise_on_failure=True)
        pipeline_factory.assert_not_called()

    def test_worker_reschedules_busy_late_ack_delivery_after_lease(self):
        task_body = analyze_project.run
        with (
            patch("app.tasks.execute_analysis", side_effect=WorkerLeaseBusy(17)),
            patch.object(analyze_project, "retry", side_effect=Retry()) as retry,
        ):
            with self.assertRaises(Retry):
                task_body("run-123")
        retry.assert_called_once()
        self.assertEqual(17, retry.call_args.kwargs["countdown"])

    def test_worker_never_autoretries_requested_cancellation(self):
        task_body = analyze_project.run
        with (
            patch(
                "app.tasks.execute_analysis",
                side_effect=AnalysisCancelled("requested"),
            ),
            patch.object(analyze_project, "retry") as retry,
        ):
            self.assertIsNone(task_body("run-123"))
        retry.assert_not_called()


if __name__ == "__main__":
    unittest.main()
