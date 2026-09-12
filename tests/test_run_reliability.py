from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
from threading import Event
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from types import SimpleNamespace
import unittest
from unittest.mock import patch

# Reliability tests must never enable the real model path or inherit a
# developer credential from .env.
os.environ["ENABLE_MODEL_EXTRACTION"] = "false"
os.environ["OPENAI_API_KEY"] = ""

from sqlalchemy import create_engine, inspect, select, update
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.domain import AnalysisCancelled
from app.db import (
    AnalysisDiagnosticRow,
    AnalysisRunExecutionRow,
    AnalysisRunInputContextRow,
    AnalysisRunInputRow,
    AnalysisRunRow,
    Base,
    DocumentRow,
    DocumentContextRow,
    ProjectRow,
    RunEventRow,
)
from app.service import (
    CORRUPT_SNAPSHOT_ERROR,
    INTERNAL_ANALYSIS_ERROR,
    MISSING_SNAPSHOT_ERROR,
    ExecutionLeaseHeartbeat,
    WorkerLeaseBusy,
    WorkerLeaseHeartbeatError,
    WorkerLeaseLost,
    _finalize_terminal,
    _interrupted_review_agent_usage,
    capture_run_inputs,
    claim_analysis_run,
    copy_run_inputs,
    emit,
    execute_analysis,
    run_input_metadata,
    safe_analysis_error,
    safe_persisted_analysis_error,
)


def pipeline_result():
    return SimpleNamespace(
        directives=[],
        issues=[],
        warnings=[],
        prompt_tokens=12,
        completion_tokens=8,
        model_used=True,
        diagnostics={
            "model": {
                "enabled": True,
                "configured": True,
                "succeeded_chunks": 1,
                "failed_chunks": 0,
                "skipped_chunks": 0,
                "invalid_records": 0,
            }
        },
    )


class RunReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        database = Path(self.temp_dir.name) / "run-reliability.db"
        self.engine = create_engine(
            f"sqlite:///{database.as_posix()}",
            connect_args={"check_same_thread": False, "timeout": 10},
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.session_patch = patch("app.service.SessionLocal", self.Session)
        self.session_patch.start()

    def tearDown(self):
        self.session_patch.stop()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def create_snapshotted_run(
        self,
        content="第一版",
        version=1,
        document_role="chapter",
        story_scope="global",
    ):
        with self.Session() as db:
            project = ProjectRow(name="快照测试")
            db.add(project)
            db.flush()
            document = DocumentRow(
                project_id=project.id,
                name="chapter.md",
                content=content,
                version=version,
            )
            run = AnalysisRunRow(project_id=project.id)
            db.add_all([document, run])
            db.flush()
            db.add(
                DocumentContextRow(
                    document_id=document.id,
                    document_role=document_role,
                    story_scope=story_scope,
                )
            )
            capture_run_inputs(db, run, [document])
            db.commit()
            return project.id, document.id, run.id

    def test_create_all_adds_snapshot_tables_without_changing_legacy_run(self):
        legacy_path = Path(self.temp_dir.name) / "legacy-upgrade.db"
        legacy_engine = create_engine(f"sqlite:///{legacy_path.as_posix()}")
        ProjectRow.__table__.create(legacy_engine)
        AnalysisRunRow.__table__.create(legacy_engine)
        LegacySession = sessionmaker(bind=legacy_engine, expire_on_commit=False)
        with LegacySession() as db:
            project = ProjectRow(name="旧项目")
            db.add(project)
            db.flush()
            old_run = AnalysisRunRow(project_id=project.id, status="completed")
            db.add(old_run)
            db.commit()
            old_run_id = old_run.id

        Base.metadata.create_all(legacy_engine)
        table_names = set(inspect(legacy_engine).get_table_names())
        self.assertIn("analysis_run_inputs", table_names)
        self.assertIn("analysis_run_execution", table_names)
        self.assertIn("document_context", table_names)
        with LegacySession() as db:
            self.assertEqual("completed", db.get(AnalysisRunRow, old_run_id).status)
        legacy_engine.dispose()

    def test_worker_reads_frozen_body_and_diagnostic_lists_exact_version(self):
        _, document_id, run_id = self.create_snapshotted_run("冻结内容", 3)
        with self.Session() as db:
            document = db.get(DocumentRow, document_id)
            document.content = "稍后修改的内容"
            document.version = 4
            db.commit()

        captured = []

        class FakePipeline:
            def run(_, documents, on_stage, checkpoint):
                captured.extend(documents)
                checkpoint()
                on_stage("extract", 10, "开始")
                return pipeline_result()

        with patch("app.service.AnalysisPipeline", FakePipeline):
            execute_analysis(run_id)

        self.assertEqual("冻结内容", captured[0].content)
        with self.Session() as db:
            run = db.get(AnalysisRunRow, run_id)
            diagnostic = db.get(AnalysisDiagnosticRow, run_id)
            execution = db.get(AnalysisRunExecutionRow, run_id)
            self.assertEqual("completed", run.status)
            self.assertEqual(1, execution.attempt_no)
            metadata = diagnostic.payload["input_snapshot"]
            self.assertTrue(metadata["immutable"])
            self.assertEqual(3, metadata["documents"][0]["document_version"])
            self.assertEqual(64, len(metadata["documents"][0]["content_sha256"]))

    def test_retry_copies_original_snapshot_not_current_document(self):
        project_id, document_id, old_run_id = self.create_snapshotted_run(
            "原始输入", document_role="reference", story_scope="route_a"
        )
        with self.Session() as db:
            db.execute(
                update(AnalysisRunRow)
                .where(AnalysisRunRow.id == old_run_id)
                .values(status="failed")
            )
            db.get(DocumentRow, document_id).content = "项目当前输入"
            retry = AnalysisRunRow(project_id=project_id)
            db.add(retry)
            db.flush()
            copy_run_inputs(db, old_run_id, retry)
            db.commit()
            retry_id = retry.id

        with self.Session() as db:
            copied = db.scalar(
                select(AnalysisRunInputRow).where(
                    AnalysisRunInputRow.run_id == retry_id
                )
            )
            execution = db.get(AnalysisRunExecutionRow, retry_id)
            context = db.get(AnalysisRunInputContextRow, copied.id)
            self.assertEqual("原始输入", copied.content)
            self.assertEqual(old_run_id, execution.retried_from_run_id)
            self.assertEqual("reference", context.document_role)
            self.assertEqual("route_a", context.story_scope)

    def test_legacy_snapshot_context_defaults_and_retry_materializes_context(self):
        project_id, _, source_run_id = self.create_snapshotted_run("旧快照")
        with self.Session() as db:
            source_input = db.scalar(
                select(AnalysisRunInputRow).where(
                    AnalysisRunInputRow.run_id == source_run_id
                )
            )
            db.delete(db.get(AnalysisRunInputContextRow, source_input.id))
            retry = AnalysisRunRow(project_id=project_id)
            db.add(retry)
            db.flush()
            copy_run_inputs(db, source_run_id, retry)
            db.commit()
            retry_id = retry.id

        with self.Session() as db:
            source_metadata = run_input_metadata(db, source_run_id)[0]
            self.assertEqual("chapter", source_metadata["document_role"])
            self.assertEqual("global", source_metadata["story_scope"])
            self.assertFalse(source_metadata["context_explicit"])
            copied = db.scalar(
                select(AnalysisRunInputRow).where(
                    AnalysisRunInputRow.run_id == retry_id
                )
            )
            copied_context = db.get(AnalysisRunInputContextRow, copied.id)
            self.assertEqual("chapter", copied_context.document_role)
            self.assertEqual("global", copied_context.story_scope)

        captured = []

        class FakePipeline:
            def run(_, documents, on_stage, checkpoint):
                captured.extend(documents)
                return pipeline_result()

        with patch("app.service.AnalysisPipeline", FakePipeline):
            execute_analysis(retry_id)
        self.assertEqual("chapter", captured[0].role)
        self.assertEqual("global", captured[0].scope)

    def test_conditional_claim_allows_only_one_concurrent_worker(self):
        _, _, run_id = self.create_snapshotted_run()
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda token: claim_analysis_run(run_id, token),
                    ("worker-one", "worker-two"),
                )
            )
        self.assertEqual([False, True], sorted(results))
        with self.Session() as db:
            execution = db.get(AnalysisRunExecutionRow, run_id)
            self.assertEqual(1, execution.attempt_no)
            self.assertIn(execution.worker_token, {"worker-one", "worker-two"})

        with self.assertRaises(WorkerLeaseBusy) as caught:
            execute_analysis(
                run_id,
                worker_token="redelivered-worker",
                raise_on_busy=True,
            )
        self.assertGreaterEqual(caught.exception.retry_after_seconds, 1)

    def test_heartbeat_prevents_takeover_during_call_longer_than_lease(self):
        _, _, run_id = self.create_snapshotted_run()
        entered = Event()
        release = Event()
        heartbeats = []

        class BlockingPipeline:
            def run(_, documents, on_stage, checkpoint):
                entered.set()
                self.assertTrue(release.wait(3))
                checkpoint()
                return pipeline_result()

        def heartbeat_factory(run_id, token):
            heartbeat = ExecutionLeaseHeartbeat(
                run_id,
                token,
                lease_seconds=0.18,
                interval_seconds=0.03,
            )
            heartbeats.append(heartbeat)
            return heartbeat

        with patch("app.service.AnalysisPipeline", BlockingPipeline):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    execute_analysis,
                    run_id,
                    worker_token="slow-worker",
                    heartbeat_factory=heartbeat_factory,
                )
                self.assertTrue(entered.wait(2))
                # Wait beyond the original lease.  A worker relying only on
                # pipeline checkpoints would now be vulnerable to takeover.
                sleep(0.3)
                self.assertFalse(claim_analysis_run(run_id, "duplicate-worker"))
                release.set()
                future.result(timeout=3)

        self.assertEqual(1, len(heartbeats))
        self.assertFalse(heartbeats[0].is_alive)
        with self.Session() as db:
            self.assertEqual("completed", db.get(AnalysisRunRow, run_id).status)
            self.assertEqual(1, db.get(AnalysisRunExecutionRow, run_id).attempt_no)

    def test_ownership_loss_fences_commit_and_next_external_call(self):
        _, _, run_id = self.create_snapshotted_run()
        first_call = Event()
        allow_checkpoint = Event()
        calls = []

        class TwoCallPipeline:
            def run(_, documents, on_stage, checkpoint):
                calls.append("first-provider-call")
                first_call.set()
                self.assertTrue(allow_checkpoint.wait(3))
                checkpoint()
                calls.append("second-provider-call")
                return pipeline_result()

        def heartbeat_factory(run_id, token):
            return ExecutionLeaseHeartbeat(
                run_id,
                token,
                lease_seconds=1,
                interval_seconds=0.02,
            )

        with patch("app.service.AnalysisPipeline", TwoCallPipeline):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    execute_analysis,
                    run_id,
                    worker_token="old-worker",
                    heartbeat_factory=heartbeat_factory,
                )
                self.assertTrue(first_call.wait(2))
                with self.Session() as db:
                    db.execute(
                        update(AnalysisRunExecutionRow)
                        .where(AnalysisRunExecutionRow.run_id == run_id)
                        .values(worker_token="new-worker")
                    )
                    db.commit()
                deadline = monotonic() + 2
                while monotonic() < deadline:
                    with self.Session() as db:
                        owner = db.get(AnalysisRunExecutionRow, run_id).worker_token
                    if owner == "new-worker":
                        break
                    sleep(0.01)
                # Give the short heartbeat interval time to publish its loss
                # signal, then let the pipeline reach its next checkpoint.
                sleep(0.08)
                allow_checkpoint.set()
                future.result(timeout=3)

        self.assertEqual(["first-provider-call"], calls)
        with self.Session() as db:
            self.assertEqual("running", db.get(AnalysisRunRow, run_id).status)
            self.assertEqual(
                "new-worker", db.get(AnalysisRunExecutionRow, run_id).worker_token
            )
            self.assertIsNone(db.get(AnalysisDiagnosticRow, run_id))

    def test_heartbeat_db_exception_is_visible_to_checkpoint_and_stops(self):
        database_error = RuntimeError("database unavailable")

        def broken_renew(*args, **kwargs):
            raise database_error

        heartbeat = ExecutionLeaseHeartbeat(
            "run",
            "worker",
            lease_seconds=3,
            interval_seconds=1,
            renew=broken_renew,
        )
        self.assertFalse(heartbeat.beat_once())
        with self.assertRaisesRegex(
            WorkerLeaseHeartbeatError, "database unavailable"
        ) as caught:
            heartbeat.raise_if_failed()
        self.assertIs(database_error, caught.exception.__cause__)
        heartbeat.stop(timeout=0.1)
        self.assertFalse(heartbeat.is_alive)

    def test_provider_error_does_not_hide_concurrent_heartbeat_db_failure(self):
        _, _, run_id = self.create_snapshotted_run()

        class FailedHeartbeat:
            def __init__(self):
                self.checks = 0
                self.stopped = False

            def start(self):
                pass

            def raise_if_failed(self):
                self.checks += 1
                if self.checks >= 3:
                    raise WorkerLeaseHeartbeatError("database unavailable")

            def stop(self):
                self.stopped = True

        heartbeat = FailedHeartbeat()

        class FailingPipeline:
            def run(_, documents, on_stage, checkpoint):
                raise RuntimeError("provider failed at the same time")

        with patch("app.service.AnalysisPipeline", FailingPipeline):
            with self.assertRaisesRegex(
                WorkerLeaseHeartbeatError, "database unavailable"
            ):
                execute_analysis(
                    run_id,
                    raise_on_failure=True,
                    worker_token="worker",
                    heartbeat_factory=lambda *_: heartbeat,
                )
        self.assertTrue(heartbeat.stopped)
        with self.Session() as db:
            self.assertEqual("running", db.get(AnalysisRunRow, run_id).status)
            self.assertEqual(
                "worker", db.get(AnalysisRunExecutionRow, run_id).worker_token
            )

    def test_heartbeat_stop_uses_a_bounded_join(self):
        entered_renewal = Event()
        release_renewal = Event()

        def blocking_renew(*args, **kwargs):
            entered_renewal.set()
            self.assertTrue(release_renewal.wait(2))
            return True

        heartbeat = ExecutionLeaseHeartbeat(
            "run",
            "worker",
            lease_seconds=3,
            interval_seconds=0.01,
            renew=blocking_renew,
        )
        heartbeat.start()
        self.assertTrue(entered_renewal.wait(1))
        started = monotonic()
        with self.assertRaisesRegex(
            WorkerLeaseHeartbeatError, "did not stop before the join timeout"
        ):
            heartbeat.stop(timeout=0.02)
        self.assertLess(monotonic() - started, 0.5)
        release_renewal.set()
        heartbeat.stop(timeout=1)
        self.assertFalse(heartbeat.is_alive)

    def test_completed_and_duplicate_delivery_leave_no_heartbeat_thread(self):
        _, _, run_id = self.create_snapshotted_run()
        heartbeats = []

        def heartbeat_factory(run_id, token):
            heartbeat = ExecutionLeaseHeartbeat(
                run_id,
                token,
                lease_seconds=2,
                interval_seconds=0.1,
            )
            heartbeats.append(heartbeat)
            return heartbeat

        class ImmediatePipeline:
            def run(_, documents, on_stage, checkpoint):
                return pipeline_result()

        with patch("app.service.AnalysisPipeline", ImmediatePipeline):
            execute_analysis(run_id, heartbeat_factory=heartbeat_factory)
            execute_analysis(run_id, heartbeat_factory=heartbeat_factory)

        self.assertEqual(1, len(heartbeats))
        self.assertFalse(heartbeats[0].is_alive)

    def test_failed_and_cancelled_runs_release_lease_and_stop_heartbeat(self):
        for terminal_status in ("failed", "cancelled"):
            with self.subTest(terminal_status=terminal_status):
                _, _, run_id = self.create_snapshotted_run()
                heartbeats = []

                def heartbeat_factory(run_id, token):
                    heartbeat = ExecutionLeaseHeartbeat(
                        run_id,
                        token,
                        lease_seconds=2,
                        interval_seconds=0.1,
                    )
                    heartbeats.append(heartbeat)
                    return heartbeat

                class TerminalPipeline:
                    def run(_, documents, on_stage, checkpoint):
                        if terminal_status == "cancelled":
                            raise AnalysisCancelled("requested")
                        raise RuntimeError("provider failed")

                with patch("app.service.AnalysisPipeline", TerminalPipeline):
                    execute_analysis(run_id, heartbeat_factory=heartbeat_factory)

                self.assertEqual(1, len(heartbeats))
                self.assertFalse(heartbeats[0].is_alive)
                with self.Session() as db:
                    run = db.get(AnalysisRunRow, run_id)
                    execution = db.get(AnalysisRunExecutionRow, run_id)
                    self.assertEqual(terminal_status, run.status)
                    self.assertEqual(
                        INTERNAL_ANALYSIS_ERROR if terminal_status == "failed" else None,
                        run.error,
                    )
                    self.assertIsNone(execution.worker_token)
                    self.assertIsNone(execution.lease_expires_at)

    def test_failure_persists_only_allowlisted_public_error(self):
        _, _, run_id = self.create_snapshotted_run()
        secret = "sk-secret https://private.invalid C:\\private\\story.txt"

        class LeakingPipeline:
            def run(_, documents, on_stage, checkpoint):
                raise RuntimeError(secret)

        with patch("app.service.AnalysisPipeline", LeakingPipeline):
            execute_analysis(run_id)

        with self.Session() as db:
            run = db.get(AnalysisRunRow, run_id)
            self.assertEqual("failed", run.status)
            self.assertEqual(INTERNAL_ANALYSIS_ERROR, run.error)
            self.assertNotIn(secret, run.error)

        self.assertEqual(
            MISSING_SNAPSHOT_ERROR,
            safe_analysis_error(RuntimeError(MISSING_SNAPSHOT_ERROR)),
        )
        self.assertEqual(
            CORRUPT_SNAPSHOT_ERROR,
            safe_analysis_error(RuntimeError("RUN_INPUT_SNAPSHOT_CORRUPT: document private-id hash mismatch")),
        )
        self.assertEqual(
            INTERNAL_ANALYSIS_ERROR,
            safe_persisted_analysis_error("legacy private provider detail"),
        )

    def test_cancel_persists_agent_usage_as_safe_lower_bound_once(self):
        _, _, run_id = self.create_snapshotted_run()
        instances = []

        class CancelAfterAgentResponsePipeline:
            def __init__(self):
                self.accounting_reads = 0
                instances.append(self)

            def run(self, documents, on_stage, checkpoint):
                raise AnalysisCancelled("cancelled after Agent response")

            def interrupted_model_usage(self):
                self.accounting_reads += 1
                return {
                    "logical_calls": 1,
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "charged_tokens": 23,
                    "provider_calls": [
                        {
                            "status": "success",
                            "category": "success",
                            "attempt": 1,
                            "elapsed_ms": 19,
                            "input_chars": 321,
                            "response_chars": 45,
                            "prompt_tokens": 11,
                            "completion_tokens": 7,
                            "total_tokens": 18,
                            "http_status": 200,
                            "request_id": "s" + "k-MUST_NOT_PERSIST",
                            "purpose": "extract",
                            "url": "must-not-persist",
                            "raw_response": "must-not-persist",
                            "api_key": "must-not-persist",
                        }
                    ],
                    "prompt": "must-not-persist",
                }

        with patch(
            "app.service.AnalysisPipeline", CancelAfterAgentResponsePipeline
        ):
            execute_analysis(run_id)
            # A redelivery of the same terminal run cannot add the usage again.
            execute_analysis(run_id)

        self.assertEqual(1, len(instances))
        self.assertEqual(1, instances[0].accounting_reads)
        with self.Session() as db:
            run = db.get(AnalysisRunRow, run_id)
            diagnostic = db.get(AnalysisDiagnosticRow, run_id)
            self.assertEqual("cancelled", run.status)
            self.assertEqual((11, 7), (run.prompt_tokens, run.completion_tokens))
            usage = diagnostic.payload["usage_accounting"]
            self.assertEqual("lower_bound", usage["completeness"])
            self.assertEqual(
                "review_agent_completed_calls_only", usage["scope"]
            )
            self.assertEqual("cancelled", usage["terminal_status"])
            self.assertEqual(23, usage["charged_tokens"])
            self.assertEqual(
                "heuristic_or_reported_internal_debit",
                usage["charged_token_semantics"],
            )
            self.assertEqual(1, usage["logical_calls"])
            self.assertEqual("agent", usage["provider_calls"][0]["purpose"])
            self.assertIsNone(usage["provider_calls"][0]["request_id"])
            serialized = str(diagnostic.payload)
            for forbidden in (
                "must-not-persist",
                "api_key",
                "raw_response",
                "url",
                "s" + "k-MUST_NOT_PERSIST",
            ):
                self.assertNotIn(forbidden, serialized)

    def test_broken_interrupted_accounting_cannot_block_cancellation(self):
        _, _, run_id = self.create_snapshotted_run()

        class BrokenAccountingPipeline:
            def run(self, documents, on_stage, checkpoint):
                raise AnalysisCancelled("requested")

            def interrupted_model_usage(self):
                raise RuntimeError("accounting unavailable")

        with patch("app.service.AnalysisPipeline", BrokenAccountingPipeline):
            execute_analysis(run_id)

        with self.Session() as db:
            run = db.get(AnalysisRunRow, run_id)
            self.assertEqual("cancelled", run.status)
            self.assertEqual((0, 0), (run.prompt_tokens, run.completion_tokens))
            self.assertIsNone(db.get(AnalysisDiagnosticRow, run_id))

    def test_huge_interrupted_usage_fails_closed_without_blocking_cancel(self):
        _, _, run_id = self.create_snapshotted_run()
        huge = 10 ** 1000

        class HugeAccountingPipeline:
            def run(self, documents, on_stage, checkpoint):
                raise AnalysisCancelled("requested")

            def interrupted_model_usage(self):
                return {
                    "logical_calls": 1,
                    "prompt_tokens": huge,
                    "completion_tokens": huge,
                    "charged_tokens": huge,
                    "provider_calls": [],
                }

        priced_settings = Settings(
            _env_file=None,
            model_input_price_per_million=1.0,
            model_output_price_per_million=2.0,
        )
        with (
            patch("app.service.AnalysisPipeline", HugeAccountingPipeline),
            patch("app.service.get_settings", return_value=priced_settings),
        ):
            execute_analysis(run_id)

        with self.Session() as db:
            run = db.get(AnalysisRunRow, run_id)
            self.assertEqual("cancelled", run.status)
            self.assertEqual((0, 0), (run.prompt_tokens, run.completion_tokens))
            self.assertEqual(0, run.estimated_cost_usd)
            self.assertIsNone(db.get(AnalysisDiagnosticRow, run_id))

    def test_huge_optional_provider_counter_hides_only_call_series(self):
        pipeline = SimpleNamespace(
            interrupted_model_usage=lambda: {
                "logical_calls": 1,
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "charged_tokens": 23,
                "provider_calls": [
                    {
                        "status": "success",
                        "category": "success",
                        "attempt": 1,
                        "elapsed_ms": 10 ** 1000,
                        "input_chars": 321,
                        "response_chars": 45,
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                        "http_status": 200,
                    }
                ],
            }
        )
        usage = _interrupted_review_agent_usage(pipeline)
        self.assertIsNotNone(usage)
        self.assertEqual(
            (11, 7),
            (usage["prompt_tokens"], usage["completion_tokens"]),
        )
        self.assertIsNone(usage["provider_calls"])

    def test_completed_run_cannot_be_claimed_or_emit_more_events(self):
        _, _, run_id = self.create_snapshotted_run()
        self.assertTrue(claim_analysis_run(run_id, "winner"))
        self.assertTrue(
            _finalize_terminal(
                run_id,
                "winner",
                "completed",
                error=None,
                message="完成",
            )
        )
        self.assertFalse(claim_analysis_run(run_id, "duplicate"))
        self.assertFalse(
            _finalize_terminal(
                run_id,
                "winner",
                "completed",
                error=None,
                message="重复完成",
            )
        )
        with self.Session() as db:
            self.assertFalse(emit(db, run_id, "extract", 99, "终态后事件"))
            terminal = db.scalars(
                select(RunEventRow).where(
                    RunEventRow.run_id == run_id,
                    RunEventRow.stage == "completed",
                )
            ).all()
            self.assertEqual(1, len(terminal))

    def test_cross_session_cancel_stops_at_next_major_stage(self):
        _, _, run_id = self.create_snapshotted_run()
        Session = self.Session

        class CancellingPipeline:
            def run(_, documents, on_stage, checkpoint):
                on_stage("extract", 10, "开始")
                with Session() as other_process:
                    other_process.execute(
                        update(AnalysisRunRow)
                        .where(AnalysisRunRow.id == run_id)
                        .values(cancel_requested=True)
                    )
                    other_process.commit()
                on_stage("index", 45, "不应写入")
                return pipeline_result()

        with patch("app.service.AnalysisPipeline", CancellingPipeline):
            execute_analysis(run_id)

        with self.Session() as db:
            run = db.get(AnalysisRunRow, run_id)
            events = db.scalars(
                select(RunEventRow)
                .where(RunEventRow.run_id == run_id)
                .order_by(RunEventRow.id)
            ).all()
            self.assertEqual("cancelled", run.status)
            self.assertEqual(["extract", "cancelled"], [row.stage for row in events])
            self.assertEqual([10, 100], [row.progress for row in events])

    def test_intermediate_worker_failure_can_retry_same_snapshot(self):
        _, _, run_id = self.create_snapshotted_run("始终相同")

        class FailingPipeline:
            def run(_, documents, on_stage, checkpoint):
                on_stage("extract", 10, "开始")
                raise RuntimeError("transient")

        with patch("app.service.AnalysisPipeline", FailingPipeline):
            with self.assertRaisesRegex(RuntimeError, "transient"):
                execute_analysis(
                    run_id,
                    raise_on_failure=True,
                    finalize_failure=False,
                    worker_token="attempt-one",
                )
        with self.Session() as db:
            waiting = db.get(AnalysisRunRow, run_id)
            self.assertEqual("queued", waiting.status)
            self.assertEqual(INTERNAL_ANALYSIS_ERROR, waiting.error)
        captured = []

        class SuccessfulPipeline:
            def run(_, documents, on_stage, checkpoint):
                captured.append(documents[0].content)
                on_stage("extract", 10, "重试")
                return pipeline_result()

        with patch("app.service.AnalysisPipeline", SuccessfulPipeline):
            execute_analysis(run_id, worker_token="attempt-two")
        with self.Session() as db:
            run = db.get(AnalysisRunRow, run_id)
            execution = db.get(AnalysisRunExecutionRow, run_id)
            events = db.scalars(
                select(RunEventRow)
                .where(RunEventRow.run_id == run_id)
                .order_by(RunEventRow.id)
            ).all()
            self.assertEqual("completed", run.status)
            self.assertEqual(2, execution.attempt_no)
            self.assertEqual("始终相同", captured[0])
            self.assertEqual(1, len([row for row in events if row.stage == "completed"]))
            self.assertEqual(sorted(row.progress for row in events), [row.progress for row in events])

    def test_legacy_queued_run_fails_without_reading_live_documents(self):
        with self.Session() as db:
            project = ProjectRow(name="旧数据")
            db.add(project)
            db.flush()
            db.add(
                DocumentRow(
                    project_id=project.id, name="chapter.md", content="当前内容"
                )
            )
            run = AnalysisRunRow(project_id=project.id)
            db.add(run)
            db.commit()
            run_id = run.id

        with patch("app.service.AnalysisPipeline") as pipeline:
            execute_analysis(run_id)
        pipeline.assert_not_called()
        with self.Session() as db:
            run = db.get(AnalysisRunRow, run_id)
            self.assertEqual("failed", run.status)
            self.assertEqual(MISSING_SNAPSHOT_ERROR, run.error)

    def test_corrupt_snapshot_failure_never_persists_document_identifier(self):
        _, document_id, run_id = self.create_snapshotted_run("原始冻结内容")
        with self.Session() as db:
            snapshot = db.scalar(
                select(AnalysisRunInputRow).where(
                    AnalysisRunInputRow.run_id == run_id
                )
            )
            snapshot.content = "数据库中被篡改的冻结内容"
            db.commit()

        with patch("app.service.AnalysisPipeline") as pipeline:
            execute_analysis(run_id)
        pipeline.assert_not_called()

        with self.Session() as db:
            run = db.get(AnalysisRunRow, run_id)
            self.assertEqual("failed", run.status)
            self.assertEqual(CORRUPT_SNAPSHOT_ERROR, run.error)
            self.assertNotIn(document_id, run.error)


if __name__ == "__main__":
    unittest.main()
