"""Native service timing/ownership boundaries; no real provider or waiting."""
from datetime import timedelta
from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app import service
from app.config import Settings
from app.db import AnalysisDiagnosticRow, AnalysisRunExecutionRow, AnalysisRunRow, RunEventRow
from app.execution_timing import ExecutionDeadline, execution_limits_payload, local_quality_timing_overrides
from app.runtime_provenance import safe_runtime_provenance


class QuietHeartbeat:
    def __init__(self, *_args):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def raise_if_failed(self):
        pass


@pytest.fixture
def timed_service(monkeypatch):
    from tests.test_run_reliability import RunReliabilityTests
    harness = RunReliabilityTests()
    harness.setUp()
    settings = Settings(_env_file=None, **local_quality_timing_overrides(),
                        enable_character_consistency=True, enable_embeddings=False,
                        openai_api_key="", enable_model_extraction=False)
    clock = [0.0]
    monkeypatch.setattr(service, "get_settings", lambda: settings)
    monkeypatch.setattr(service, "runtime_for_analysis_run", lambda *_args, **_kwargs:
        SimpleNamespace(settings=settings, identity={}, available=False, unavailable_reason="not_configured"))
    monkeypatch.setattr(service, "ExecutionDeadline", SimpleNamespace(from_remaining=lambda remaining:
        ExecutionDeadline.from_remaining(remaining, monotonic=lambda: clock[0])))
    try:
        yield harness, settings, clock
    finally:
        harness.tearDown()


def _run(harness, settings, *, old=False, expired=False):
    _, _, run_id = harness.create_snapshotted_run()
    with harness.Session() as db:
        row = db.get(AnalysisRunRow, run_id)
        if not old:
            limits = execution_limits_payload(settings)
            limits.update(read_seconds=2, logical_seconds=5, run_seconds=10)
            row.batch_coverage = {"execution_limits": limits}
        if expired:
            row.started_at = service.utc_now_naive() - timedelta(seconds=11)
            row.status = "running"
            execution = db.get(AnalysisRunExecutionRow, run_id)
            execution.worker_token = "previous-worker"
            execution.lease_expires_at = service.utc_now_naive() - timedelta(seconds=1)
        db.commit()
    return run_id


@pytest.mark.parametrize("ending", ["after_pipeline", "swallowed_optional", "cancel", "takeover"])
def test_shared_run_deadline_survives_stage_fallback_and_preserves_usage(timed_service, monkeypatch, ending):
    from tests.test_run_reliability import pipeline_result
    harness, settings, clock = timed_service
    run_id = _run(harness, settings)
    seen = {}

    class Pipeline:
        accepts_run_local_settings = True

        def __init__(self, settings):
            seen["context"] = settings._execution_deadline
            assert settings.provider_timeout_seconds == 2

        def run(self, *_args, **_kwargs):
            clock[0] += 11 if ending == "after_pipeline" else 4
            return pipeline_result()

    class Stage:
        def __init__(self, *, settings, provider, **_kwargs):
            assert settings._execution_deadline is seen["context"]
            assert 5 < settings._execution_deadline.remaining_seconds() <= 6
            self.usage = provider.usage

        def run(self, *_args, **_kwargs):
            self.usage.record(prompt_tokens=11, completion_tokens=7, charged_tokens=23, successful=True)
            clock[0] += 7
            if ending in {"cancel", "takeover"}:
                with harness.Session() as db:
                    if ending == "cancel":
                        db.get(AnalysisRunRow, run_id).cancel_requested = True
                    else:
                        db.get(AnalysisRunExecutionRow, run_id).worker_token = "new-owner"
                    db.commit()
            # Simulates a broad optional-stage handler swallowing a timeout.
            raise RuntimeError("optional stage stopped; no private text")

    monkeypatch.setattr(service, "AnalysisPipeline", Pipeline)
    monkeypatch.setattr(service, "CharacterConsistencyStage", Stage)
    monkeypatch.setattr(service, "_CharacterConsistencyAccountingProvider",
                        lambda _settings, usage: SimpleNamespace(usage=usage))
    service.execute_analysis(run_id, finalize_failure=False, raise_on_failure=True,
                             heartbeat_factory=QuietHeartbeat)
    with harness.Session() as db:
        row = db.get(AnalysisRunRow, run_id)
        events = db.scalars(select(RunEventRow).where(RunEventRow.run_id == run_id)).all()
        assert not any(event.stage == "completed" for event in events)
        if ending == "takeover":
            assert row.status == "running"
            assert db.get(AnalysisRunExecutionRow, run_id).worker_token == "new-owner"
        elif ending == "cancel":
            assert row.status == "cancelled" and row.error is None
        else:
            assert row.status == "failed" and row.error == service.ANALYSIS_TIMEOUT_ERROR
            assert any(event.message == service.ANALYSIS_TIMEOUT_ERROR for event in events)
            usage = db.get(AnalysisDiagnosticRow, run_id).payload["usage_accounting"]
            expected = (12, 8) if ending == "after_pipeline" else (23, 15)
            assert (usage["prompt_tokens"], usage["completion_tokens"]) == expected
            assert (row.prompt_tokens, row.completion_tokens) == expected
            assert usage["terminal_status"] == "failed"


def test_takeover_reuses_persisted_start_and_expires_before_pipeline(timed_service, monkeypatch):
    harness, settings, _clock = timed_service
    run_id = _run(harness, settings, expired=True)
    def forbidden(*_args, **_kwargs):
        raise AssertionError("expired run must not construct a pipeline")
    monkeypatch.setattr(service, "AnalysisPipeline", forbidden)
    service.execute_analysis(run_id, heartbeat_factory=QuietHeartbeat)
    with harness.Session() as db:
        row = db.get(AnalysisRunRow, run_id)
        assert row.status == "failed" and row.error == service.ANALYSIS_TIMEOUT_ERROR
        assert (service.utc_now_naive() - row.started_at).total_seconds() >= 11


def test_old_run_without_snapshot_keeps_legacy_timing(timed_service, monkeypatch):
    from tests.test_run_reliability import pipeline_result
    harness, settings, clock = timed_service
    run_id = _run(harness, settings, old=True)
    seen = []
    class Pipeline:
        accepts_run_local_settings = True
        def __init__(self, settings):
            assert settings._execution_deadline is None
            assert settings.provider_timeout_seconds == 30
            assert settings.analysis_run_timeout_seconds is None
            seen.append(settings)
        def run(self, *_args, **_kwargs):
            clock[0] += 3000
            return pipeline_result()
    monkeypatch.setattr(service, "AnalysisPipeline", Pipeline)
    monkeypatch.setattr(service, "CharacterConsistencyStage", lambda **_kwargs:
                        SimpleNamespace(run=lambda *_args, **_kw: service.failed_character_consistency_stage()))
    service.execute_analysis(run_id, raise_on_failure=True, heartbeat_factory=QuietHeartbeat)
    with harness.Session() as db:
        assert db.get(AnalysisRunRow, run_id).status == "completed"
    assert len(seen) == 1


def test_quality_habit_fork_and_existing_provenance_use_configured_timing():
    settings = Settings(_env_file=None, **local_quality_timing_overrides(), openai_api_key="",
                        character_draft_actor_review_v1=True, character_signal_full_line_prompt_v2=True)
    settings._character_habit_scene_scheduling_v1 = True
    context = ExecutionDeadline.from_remaining(100)
    settings._execution_deadline = context
    provider = service._CharacterConsistencyAccountingProvider(settings, service.CharacterConsistencyUsageAccumulator())
    fork = provider.fork_for_habit_scene_review(timeout_seconds=180, remaining_deadline_seconds=360,
        completion_reserve=2048, max_response_bytes=65536, max_attempts=1)
    assert fork.draft_actor_review_provider.settings.provider_timeout_seconds == 180
    assert fork.draft_actor_review_provider.settings._execution_deadline is context
    limits = safe_runtime_provenance(settings)["character_consistency_limits"]
    assert limits["signal_scope_review_provider_timeout_seconds"] == 180
    assert limits["signal_scope_review_provider_total_deadline_seconds"] == 180
    assert limits["signal_total_deadline_seconds"] == 360
    assert service.safe_persisted_analysis_error(service.ANALYSIS_TIMEOUT_ERROR) == service.ANALYSIS_TIMEOUT_ERROR


def test_native_batch_response_usage_survives_checkpoint_before_pipeline_return(timed_service, monkeypatch):
    import json
    import httpx
    from app import provider as provider_module
    from app.db import DocumentRow, ProjectRow
    from app.model_extractor import BATCH_SYSTEM_PROMPT

    harness, settings, clock = timed_service
    settings.enable_model_extraction = True
    settings.openai_api_key = "unit-test-placeholder"
    settings.openai_base_url = "https://mock.invalid/v1"
    settings.enable_character_consistency = False
    with harness.Session() as db:
        project = ProjectRow(name="响应后超时")
        db.add(project)
        db.flush()
        documents = [DocumentRow(project_id=project.id, name=f"chapter-{index}.md",
                                 content=f"第{index}章的未持久化私密句子。", version=1)
                     for index in (1, 2)]
        limits = execution_limits_payload(settings)
        limits.update(read_seconds=2, logical_seconds=5, run_seconds=10)
        run = AnalysisRunRow(project_id=project.id, batch_coverage={"execution_limits": limits})
        db.add_all([*documents, run])
        db.flush()
        service.capture_run_inputs(db, run, documents)
        db.commit()
        run_id = run.id
    requests = []

    def respond(request):
        body = json.loads(request.content)
        assert body["messages"][0]["content"] == BATCH_SYSTEM_PROMPT
        requests.append("batch")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"documents":[]}'}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8},
        })

    class RespondThenExpire(provider_module.OpenAICompatibleProvider):
        def __init__(self, settings=None, **kwargs):
            super().__init__(settings, transport=httpx.MockTransport(respond), **kwargs)

        def complete(self, *args, **kwargs):
            response = super().complete(*args, **kwargs)
            # Full transport and usage parse succeeded; time expires between
            # the returned response and the extractor's post-response check.
            clock[0] += 11
            return response

    monkeypatch.setattr(provider_module, "OpenAICompatibleProvider", RespondThenExpire)
    service.execute_analysis(run_id, heartbeat_factory=QuietHeartbeat)
    service.execute_analysis(run_id, heartbeat_factory=QuietHeartbeat)
    assert requests == ["batch"]
    with harness.Session() as db:
        run = db.get(AnalysisRunRow, run_id)
        diagnostic = db.get(AnalysisDiagnosticRow, run_id)
        assert run.status == "failed" and run.error == service.ANALYSIS_TIMEOUT_ERROR
        assert (run.prompt_tokens, run.completion_tokens) == (12, 8)
        usage = diagnostic.payload["usage_accounting"]
        assert usage["scope"] == "model_extraction_and_agent_completed_calls"
        assert usage["logical_calls"] == 1 and usage["provider_calls"] is None
        assert (usage["prompt_tokens"], usage["completion_tokens"]) == (12, 8)
        assert usage["charged_tokens"] >= 20 and usage["completeness"] == "lower_bound"
        assert "未持久化私密句子" not in str(diagnostic.payload)
        assert "unit-test-placeholder" not in str(diagnostic.payload)
