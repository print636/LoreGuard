from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, select, update
from sqlalchemy.orm import sessionmaker

from app.candidate_promotion import CandidatePromotionResult
from app.config import Settings
from app.db import (
    AnalysisDiagnosticRow,
    AnalysisRecordRow,
    AnalysisRunExecutionRow,
    AnalysisRunRow,
    Base,
    DocumentContextRow,
    DocumentRow,
    IssueRow,
    ProjectRow,
    RunEventRow,
)
from app.domain import (
    AnalysisCancelled,
    ConsistencyIssue,
    EvidenceSpan,
    IssueCategory,
    ParsedDirective,
    Severity,
)
from app.evidence_investigator_runtime import EvidenceInvestigatorRuntimeResult
from app.evidence_chunks import EvidenceChunker
from app.issue_evidence_review import IssueEvidenceReviewResult
from app.provider import OpenAICompatibleProvider, ProviderToolCall, ToolCallResult
from app.service import (
    WorkerLeaseLost,
    _emit_owned,
    capture_run_inputs,
    execute_analysis,
)


CONTENT = "岚的发色是银色。\n岚的发色是黑色。"


def find_nested_string(value, key: str) -> str:
    if isinstance(value, dict):
        if isinstance(value.get(key), str):
            return value[key]
        for child in value.values():
            try:
                return find_nested_string(child, key)
            except LookupError:
                pass
    elif isinstance(value, list):
        for child in value:
            try:
                return find_nested_string(child, key)
            except LookupError:
                pass
    raise LookupError(key)


@pytest.fixture()
def db_session(tmp_path):
    engine = create_engine(
        f"sqlite:///{(tmp_path / 'investigator-service.db').as_posix()}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)
    try:
        yield session
    finally:
        engine.dispose()


def create_run(session, *, historical_usage=None):
    with session() as db:
        project = ProjectRow(name="证据调查接线")
        db.add(project)
        db.flush()
        document = DocumentRow(
            project_id=project.id,
            name="chapter.md",
            content=CONTENT,
            version=4,
        )
        run = AnalysisRunRow(project_id=project.id)
        db.add_all([document, run])
        db.flush()
        db.add(
            DocumentContextRow(
                document_id=document.id,
                document_role="chapter",
                story_scope="main",
            )
        )
        capture_run_inputs(db, run, [document])
        if historical_usage is not None:
            db.add(
                AnalysisDiagnosticRow(
                    run_id=run.id,
                    payload={"usage_accounting": historical_usage},
                )
            )
        db.commit()
        return project.id, document.id, run.id


def directive(document_id: str, *, value: str, line: int) -> ParsedDirective:
    return ParsedDirective(
        kind="fact",
        attrs={
            "subject": "岚",
            "predicate": "发色",
            "value": value,
            "modality": "asserted",
            "source_scope": "narrator",
            "certainty": "certain",
            "story_scope": "main",
            "document_role": "chapter",
        },
        evidence=EvidenceSpan(
            document_id=document_id,
            document_name="chapter.md",
            line_start=line,
            line_end=line,
            text=CONTENT.splitlines()[line - 1],
        ),
        provenance_sources=frozenset(
            {"baseline"} if line == 1 else {"model"}
        ),
    )


def conflict_issue(document_id: str) -> ConsistencyIssue:
    return ConsistencyIssue(
        category=IssueCategory.fact_conflict,
        severity=Severity.high,
        confidence=0.98,
        title="发色冲突",
        explanation="同一角色的发色记录不一致",
        evidence=[
            directive(document_id, value="银色", line=1).evidence,
            directive(document_id, value="黑色", line=2).evidence,
        ],
        suggestion="核对发色设定",
        metadata={"rule": "fact_conflict"},
    )


def pipeline_factory(
    document_id: str, *, mutate_live=None, issue=None, budget_debit=None
):
    baseline_provenance = {
        "schema_version": 1,
        "directives": [{"fingerprint": "baseline-fingerprint"}],
        "issues": [{"fingerprint": "baseline-issue"}] if issue else [],
    }

    class FakePipeline:
        last_result = None

        def conservative_run_token_debit(self, result):
            if budget_debit is not None:
                return budget_debit
            return result.prompt_tokens + result.completion_tokens

        def run(self, documents, on_stage, checkpoint):
            assert documents[0].content == CONTENT
            if mutate_live is not None:
                mutate_live()
            checkpoint()
            on_stage("check", 75, "规则检查完成")
            result = SimpleNamespace(
                directives=[directive(document_id, value="银色", line=1)],
                issues=[issue] if issue is not None else [],
                warnings=[],
                prompt_tokens=12,
                completion_tokens=8,
                model_used=False,
                diagnostics={
                    "model": {
                        "enabled": False,
                        "configured": False,
                        "succeeded_chunks": 0,
                        "failed_chunks": 0,
                        "skipped_chunks": 0,
                        "invalid_records": 0,
                        "empty_response_chunks": 0,
                    },
                    "provenance": deepcopy(baseline_provenance),
                },
            )
            FakePipeline.last_result = result
            return result

    return FakePipeline, baseline_provenance


def enabled_settings(*, review=False, per_run_token_budget=20_000):
    return Settings(
        _env_file=None,
        openai_api_key="unit-secret",
        openai_base_url="https://mock.invalid/v1",
        openai_model="mock",
        enable_evidence_investigator=True,
        enable_embeddings=True,
        embedding_base_url="https://embedding.invalid/v1",
        embedding_model="mock-embedding",
        embedding_model_revision="r1",
        embedding_deployment_fingerprint="test-deployment-v1",
        embedding_dimensions=2,
        evidence_investigator_require_hybrid=False,
        evidence_investigator_max_completion_tokens=64,
        evidence_investigator_token_budget=8_000,
        enable_issue_evidence_review=review,
        issue_evidence_review_require_hybrid=False,
        per_run_token_budget=per_run_token_budget,
    )


def test_switch_off_performs_no_runtime_or_bundle_work_and_emits_no_new_stage(db_session):
    _, document_id, run_id = create_run(db_session)
    FakePipeline, baseline_provenance = pipeline_factory(document_id)
    expected_directive_bytes = directive(
        document_id, value="银色", line=1
    ).model_dump_json()
    settings = Settings(_env_file=None)

    with (
        patch("app.service.SessionLocal", db_session),
        patch("app.service.AnalysisPipeline", FakePipeline),
        patch(
            "app.service.build_frozen_investigation_bundle",
            side_effect=AssertionError("disabled path must not freeze optional bundle"),
        ),
        patch(
            "app.service.EvidenceInvestigatorRuntime",
            side_effect=AssertionError("disabled path must not create runtime"),
        ),
        patch("app.service.get_settings", return_value=settings),
    ):
        execute_analysis(run_id, raise_on_failure=True)

    with db_session() as db:
        diagnostic = db.get(AnalysisDiagnosticRow, run_id)
        events = db.scalars(
            select(RunEventRow).where(RunEventRow.run_id == run_id)
        ).all()
    assert "evidence_investigator" not in diagnostic.payload
    assert not [row for row in events if row.stage == "evidence_investigator"]
    assert [
        row.model_dump_json() for row in FakePipeline.last_result.directives
    ] == [expected_directive_bytes]
    assert FakePipeline.last_result.issues == []
    assert FakePipeline.last_result.diagnostics["provenance"] == baseline_provenance


def test_enabled_run_with_no_seed_skips_before_any_chat_or_embedding(db_session):
    _, _, run_id = create_run(db_session)

    class EmptyPipeline:
        def run(self, documents, on_stage, checkpoint):
            checkpoint()
            on_stage("check", 75, "规则检查完成")
            return SimpleNamespace(
                directives=[],
                issues=[],
                warnings=[],
                prompt_tokens=0,
                completion_tokens=0,
                model_used=False,
                diagnostics={
                    "model": {
                        "enabled": False,
                        "configured": False,
                        "succeeded_chunks": 0,
                        "failed_chunks": 0,
                        "skipped_chunks": 0,
                        "invalid_records": 0,
                        "empty_response_chunks": 0,
                    },
                    "provenance": {
                        "schema_version": 1,
                        "directives": [],
                        "issues": [],
                    },
                },
            )

    with (
        patch("app.service.SessionLocal", db_session),
        patch("app.service.AnalysisPipeline", EmptyPipeline),
        patch("app.service.get_settings", return_value=enabled_settings()),
        patch(
            "app.evidence_investigator_runtime.OpenAICompatibleProvider",
            side_effect=AssertionError("no seed must not construct chat provider"),
        ),
        patch(
            "app.evidence_investigator_runtime.OpenAICompatibleEmbeddingProvider",
            side_effect=AssertionError("no seed must not construct embedding provider"),
        ),
    ):
        execute_analysis(run_id, raise_on_failure=True)

    with db_session() as db:
        diagnostic = db.get(AnalysisDiagnosticRow, run_id)
    investigator = diagnostic.payload["evidence_investigator"]
    assert investigator["outcome"] == "skipped"
    assert investigator["reason_code"] == "no_seeds"
    assert investigator["usage"] is None


def test_real_runtime_tool_calls_are_merged_into_service_usage(db_session):
    _, document_id, run_id = create_run(db_session)
    FakePipeline, _ = pipeline_factory(document_id, budget_debit=200)
    created_rag = []

    class NativeProvider:
        def __init__(self):
            self.calls = 0

        def complete_with_tools(self, _system, user, **_kwargs):
            self.calls += 1
            state = json.loads(user)
            seed_ref = state["current_seed"]["seed_ref"]
            if self.calls == 1:
                name = "SEARCH_EVIDENCE"
                arguments = {
                    "seed_ref": seed_ref,
                    "query": "岚的发色是否冲突",
                    "entity_terms": ["岚"],
                }
            elif self.calls == 2:
                name = "READ_SPAN"
                arguments = {
                    "seed_ref": seed_ref,
                    "result_ref": find_nested_string(state, "result_ref"),
                    "line_start": 2,
                    "line_end": 2,
                }
            else:
                name = "SUBMIT_VERDICT"
                arguments = {
                    "seed_ref": seed_ref,
                    "verdict": "candidate_conflict",
                    "candidates": [
                        {
                            "kind": "fact",
                            "span_ref": find_nested_string(state, "span_ref"),
                            "source_line_start": 2,
                            "source_line_end": 2,
                            "fields": {
                                "subject": "岚",
                                "predicate": "发色",
                                "value": "黑色",
                            },
                        }
                    ],
                }
            return ToolCallResult(
                tool_calls=(
                    ProviderToolCall(
                        id=f"call_{self.calls}",
                        name=name,
                        arguments=arguments,
                    ),
                ),
                prompt_tokens=7,
                completion_tokens=2,
            )

    native = NativeProvider()

    class BaseProvider(OpenAICompatibleProvider):
        evidence_investigator_configured = True

        def __init__(self):
            self.remaining_deadlines = []

        def fork_for_evidence_investigator(self, **kwargs):
            assert kwargs["remaining_deadline_seconds"] > 0
            self.remaining_deadlines.append(kwargs["remaining_deadline_seconds"])
            return native

    base = BaseProvider()

    class RuntimeRag:
        def __init__(self, *, scope, **_kwargs):
            self.prepare_calls = 0
            self.search_calls = 0
            document = scope.documents[0]
            self.chunk = EvidenceChunker(
                target_chars=100,
                min_chars=1,
                max_chars=120,
                overlap_chars=0,
            ).chunk(
                project_id=document.snapshot.project_id,
                document_id=document.snapshot.document_id,
                document_version=document.snapshot.document_version,
                content=document.content,
                content_sha256=document.snapshot.content_sha256,
            )[0]
            created_rag.append(self)

        def prepare(self):
            self.prepare_calls += 1

        def search(self, **kwargs):
            self.search_calls += 1
            return (self.chunk,)[: kwargs["limit"]]

        def safe_diagnostics(self):
            return {
                "index": {
                    "outcome": "complete",
                    "reason": None,
                    "profile_id": "unit-profile",
                    "chunker_version": self.chunk.chunker_version,
                    "document_count": 1,
                    "expected_chunks": 1,
                    "reused_chunks": 1,
                    "embedded_chunks": 0,
                    "provider_calls": 0,
                    "provider_input_chars": 0,
                    "elapsed_ms": 1,
                },
                "retrievals": [],
                "total_retrievals": self.search_calls,
                "retrievals_truncated": False,
                "embedding_activity": {
                    "index_calls": 0,
                    "index_input_chars": 0,
                    "query_calls": 0,
                    "query_input_chars": 0,
                },
            }

    with (
        patch("app.service.SessionLocal", db_session),
        patch("app.service.AnalysisPipeline", FakePipeline),
        patch("app.service.get_settings", return_value=enabled_settings()),
        patch(
            "app.evidence_investigator_runtime.OpenAICompatibleProvider",
            return_value=base,
        ),
        patch(
            "app.evidence_investigator_runtime.InvestigatorRagRetriever",
            RuntimeRag,
        ),
    ):
        execute_analysis(run_id, raise_on_failure=True)

    with db_session() as db:
        saved_run = db.get(AnalysisRunRow, run_id)
        diagnostic = db.get(AnalysisDiagnosticRow, run_id)
        issues = db.scalars(
            select(IssueRow).where(IssueRow.run_id == run_id)
        ).all()
    usage = diagnostic.payload["usage_accounting"]
    assert saved_run.status == "completed"
    assert native.calls == 3
    assert created_rag[0].prepare_calls == 1
    assert created_rag[0].search_calls == 1
    assert len(base.remaining_deadlines) == 3
    assert base.remaining_deadlines == sorted(
        base.remaining_deadlines, reverse=True
    )
    assert len(issues) == 1
    assert usage["logical_calls"] == 3
    assert usage["prompt_tokens"] == 21
    assert usage["completion_tokens"] == 6
    assert usage["charged_tokens"] >= 27
    assert saved_run.prompt_tokens == 12 + 21
    assert saved_run.completion_tokens == 8 + 6


def test_success_is_atomically_appended_before_reviewer_and_usage_budgets_merge(db_session):
    historical = {
        "completeness": "completed_calls",
        "scope": "previous_attempt",
        "terminal_status": "running",
        "logical_calls": 1,
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "charged_tokens": 500,
        "charged_token_semantics": "conservative_internal_budget_debit",
        "provider_calls": None,
    }
    _, document_id, run_id = create_run(
        db_session, historical_usage=historical
    )
    FakePipeline, _ = pipeline_factory(document_id, budget_debit=200)
    baseline = directive(document_id, value="银色", line=1)
    candidate = directive(document_id, value="黑色", line=2)
    added = conflict_issue(document_id)
    captured = {}

    class FakeRuntime:
        def __init__(self, **kwargs):
            self.usage = kwargs["usage"]

        def run(self, **kwargs):
            captured["bundle_content"] = kwargs["bundle"].evidence_documents[0].content
            captured["investigator_remaining"] = kwargs["remaining_run_tokens"]
            self.usage.record(None, True, 5, 2, 100, "success")
            promotion = CandidatePromotionResult(
                directives=(baseline, candidate),
                issues=(added,),
                promoted_directives=(candidate,),
                added_issues=(added,),
                submitted_candidates=1,
                accepted_candidates=1,
                rejection_counts=(),
            )
            return EvidenceInvestigatorRuntimeResult(
                outcome="completed",
                reason_code="completed",
                diagnostics={
                    "enabled": True,
                    "outcome": "completed",
                    "reason_code": "completed",
                    "seed_count": 1,
                    "snapshot_fingerprint": "a" * 64,
                    "budget_preflight": {},
                    "loop": {},
                    "rag": {},
                    "promotion": promotion.safe_dict(),
                    "usage": None,
                },
                promotion=promotion,
            )

    class FakeReviewer:
        def __init__(self, **kwargs):
            self.account = kwargs["usage_accounting"]

        def review(self, *, documents, issues, remaining_run_tokens):
            captured["review_documents"] = documents
            captured["review_issues"] = issues
            captured["review_remaining"] = remaining_run_tokens
            self.account(None, True, 3, 1, 20, "success")
            return IssueEvidenceReviewResult(
                annotations={},
                diagnostics={"enabled": True, "outcome": "completed"},
                prompt_tokens=3,
                completion_tokens=1,
                charged_tokens=20,
            )

    settings = enabled_settings(review=True)
    with (
        patch("app.service.SessionLocal", db_session),
        patch("app.service.AnalysisPipeline", FakePipeline),
        patch("app.service.EvidenceInvestigatorRuntime", FakeRuntime),
        patch("app.service.IssueEvidenceReviewer", FakeReviewer),
        patch("app.service.get_settings", return_value=settings),
    ):
        execute_analysis(run_id, raise_on_failure=True)

    assert captured["bundle_content"] == CONTENT
    assert captured["investigator_remaining"] == 20_000 - 200 - 500
    assert captured["review_documents"][0].content == CONTENT
    assert captured["review_issues"] == (added,)
    assert captured["review_remaining"] == 20_000 - 200 - 500 - 100
    with db_session() as db:
        saved_run = db.get(AnalysisRunRow, run_id)
        diagnostic = db.get(AnalysisDiagnosticRow, run_id)
        records = db.scalars(
            select(AnalysisRecordRow)
            .where(AnalysisRecordRow.run_id == run_id)
        ).all()
        issues = db.scalars(
            select(IssueRow).where(IssueRow.run_id == run_id)
        ).all()
        events = db.scalars(
            select(RunEventRow)
            .where(RunEventRow.run_id == run_id)
            .order_by(RunEventRow.id)
        ).all()

    records.sort(key=lambda row: row.evidence["line_start"])
    assert [row.attrs["value"] for row in records] == ["银色", "黑色"]
    assert [row.category for row in issues] == ["fact_conflict"]
    assert diagnostic.payload["provenance"]["schema_version"] == 1
    assert len(diagnostic.payload["provenance"]["directives"]) == 2
    assert diagnostic.payload["evidence_investigator"]["outcome"] == "completed"
    accounting = diagnostic.payload["usage_accounting"]
    assert accounting["logical_calls"] == 3
    assert accounting["prompt_tokens"] == 18
    assert accounting["completion_tokens"] == 8
    assert accounting["charged_tokens"] == 620
    assert (
        accounting["charged_token_semantics"]
        == "heuristic_or_reported_internal_debit"
    )
    assert saved_run.prompt_tokens == 12 + 5 + 3 + 10
    assert saved_run.completion_tokens == 8 + 2 + 1 + 5
    investigator_events = [
        row.progress for row in events if row.stage == "evidence_investigator"
    ]
    assert investigator_events == [76, 78, 81]
    assert [row.progress for row in events] == sorted(
        row.progress for row in events
    )


def test_optional_failure_preserves_baseline_rows_ids_order_and_provenance(db_session):
    _, document_id, run_id = create_run(db_session)
    baseline_issue = conflict_issue(document_id)
    FakePipeline, baseline_provenance = pipeline_factory(
        document_id, issue=baseline_issue
    )

    class BrokenRuntime:
        def __init__(self, **_kwargs):
            pass

        def run(self, **_kwargs):
            raise RuntimeError("CANARY story https://secret.invalid sk-secret")

    with (
        patch("app.service.SessionLocal", db_session),
        patch("app.service.AnalysisPipeline", FakePipeline),
        patch("app.service.EvidenceInvestigatorRuntime", BrokenRuntime),
        patch("app.service.get_settings", return_value=enabled_settings()),
    ):
        execute_analysis(run_id, raise_on_failure=True)

    with db_session() as db:
        saved_run = db.get(AnalysisRunRow, run_id)
        diagnostic = db.get(AnalysisDiagnosticRow, run_id)
        records = db.scalars(
            select(AnalysisRecordRow).where(AnalysisRecordRow.run_id == run_id)
        ).all()
        issues = db.scalars(
            select(IssueRow).where(IssueRow.run_id == run_id)
        ).all()
    assert saved_run.status == "completed"
    assert [row.attrs["value"] for row in records] == ["银色"]
    assert [row.category for row in issues] == ["fact_conflict"]
    assert diagnostic.payload["provenance"] == baseline_provenance
    assert diagnostic.payload["evidence_investigator"]["outcome"] == "degraded"
    serialized = str(diagnostic.payload)
    assert "CANARY" not in serialized
    assert "secret.invalid" not in serialized
    assert "sk-secret" not in serialized


def test_bundle_uses_loaded_snapshot_even_when_live_document_mutates(db_session):
    _, document_id, run_id = create_run(db_session)

    def mutate_live():
        with db_session() as db:
            live = db.get(DocumentRow, document_id)
            live.content = "稍后写入的实时版本"
            live.version = 5
            db.commit()

    FakePipeline, _ = pipeline_factory(document_id, mutate_live=mutate_live)
    captured = {}

    class SkippingRuntime:
        def __init__(self, **_kwargs):
            pass

        def run(self, **kwargs):
            captured["content"] = kwargs["bundle"].evidence_documents[0].content
            return EvidenceInvestigatorRuntimeResult(
                outcome="skipped",
                reason_code="no_seeds",
                diagnostics={
                    "enabled": True,
                    "outcome": "skipped",
                    "reason_code": "no_seeds",
                    "seed_count": 0,
                },
            )

    with (
        patch("app.service.SessionLocal", db_session),
        patch("app.service.AnalysisPipeline", FakePipeline),
        patch("app.service.EvidenceInvestigatorRuntime", SkippingRuntime),
        patch("app.service.get_settings", return_value=enabled_settings()),
    ):
        execute_analysis(run_id, raise_on_failure=True)

    assert captured["content"] == CONTENT


def test_cancel_passthrough_persists_immediate_investigator_charge(db_session):
    _, document_id, run_id = create_run(db_session)
    FakePipeline, _ = pipeline_factory(document_id)

    class CancellingRuntime:
        def __init__(self, **kwargs):
            self.usage = kwargs["usage"]

        def run(self, **_kwargs):
            self.usage.record(None, False, 0, 0, 1_234, "provider")
            raise AnalysisCancelled("stop after investigator provider call")

    with (
        patch("app.service.SessionLocal", db_session),
        patch("app.service.AnalysisPipeline", FakePipeline),
        patch("app.service.EvidenceInvestigatorRuntime", CancellingRuntime),
        patch("app.service.get_settings", return_value=enabled_settings()),
    ):
        execute_analysis(run_id, raise_on_failure=True)

    with db_session() as db:
        saved_run = db.get(AnalysisRunRow, run_id)
        diagnostic = db.get(AnalysisDiagnosticRow, run_id)
    assert saved_run.status == "cancelled"
    assert diagnostic.payload["usage_accounting"]["charged_tokens"] == 1_234
    assert diagnostic.payload["usage_accounting"]["prompt_tokens"] == 0
    assert diagnostic.payload["usage_accounting"]["completion_tokens"] == 0


def test_lease_loss_fences_all_candidate_and_diagnostic_persistence(db_session):
    _, document_id, run_id = create_run(db_session)
    FakePipeline, _ = pipeline_factory(document_id)

    class LeaseLosingRuntime:
        def __init__(self, **kwargs):
            self.usage = kwargs["usage"]

        def run(self, **_kwargs):
            self.usage.record(None, False, 0, 0, 1_234, "provider")
            raise WorkerLeaseLost("ownership changed")

    with (
        patch("app.service.SessionLocal", db_session),
        patch("app.service.AnalysisPipeline", FakePipeline),
        patch("app.service.EvidenceInvestigatorRuntime", LeaseLosingRuntime),
        patch("app.service.get_settings", return_value=enabled_settings()),
    ):
        execute_analysis(run_id, raise_on_failure=True)

    with db_session() as db:
        saved_run = db.get(AnalysisRunRow, run_id)
        diagnostic = db.get(AnalysisDiagnosticRow, run_id)
        records = db.scalars(
            select(AnalysisRecordRow).where(AnalysisRecordRow.run_id == run_id)
        ).all()
        issues = db.scalars(
            select(IssueRow).where(IssueRow.run_id == run_id)
        ).all()
    assert saved_run.status == "running"
    assert diagnostic is None
    assert records == []
    assert issues == []


def test_lease_takeover_between_runtime_and_progress_fences_later_events(db_session):
    _, document_id, run_id = create_run(db_session)
    FakePipeline, _ = pipeline_factory(document_id)

    class TakeoverRuntime:
        def __init__(self, **_kwargs):
            pass

        def run(self, **_kwargs):
            with db_session() as db:
                changed = db.execute(
                    update(AnalysisRunExecutionRow)
                    .where(AnalysisRunExecutionRow.run_id == run_id)
                    .values(worker_token="replacement-worker")
                ).rowcount
                assert changed == 1
                db.commit()
            return EvidenceInvestigatorRuntimeResult(
                outcome="skipped",
                reason_code="no_seeds",
                diagnostics={
                    "enabled": True,
                    "outcome": "skipped",
                    "reason_code": "no_seeds",
                    "seed_count": 0,
                },
            )

    with (
        patch("app.service.SessionLocal", db_session),
        patch("app.service.AnalysisPipeline", FakePipeline),
        patch("app.service.EvidenceInvestigatorRuntime", TakeoverRuntime),
        patch("app.service.get_settings", return_value=enabled_settings()),
    ):
        execute_analysis(run_id, worker_token="original-worker", raise_on_failure=True)

    with db_session() as db:
        execution = db.get(AnalysisRunExecutionRow, run_id)
        run = db.get(AnalysisRunRow, run_id)
        events = db.scalars(
            select(RunEventRow)
            .where(RunEventRow.run_id == run_id)
            .order_by(RunEventRow.id)
        ).all()
    assert execution.worker_token == "replacement-worker"
    assert run.status == "running"
    assert [row.progress for row in events if row.stage == "evidence_investigator"] == [
        76
    ]


def test_stale_owned_emit_cannot_block_replacement_worker_progress(db_session):
    _, _, run_id = create_run(db_session)
    with db_session() as db:
        execution = db.get(AnalysisRunExecutionRow, run_id)
        execution.worker_token = "replacement-worker"
        db.commit()

    with db_session() as db:
        with pytest.raises(WorkerLeaseLost):
            _emit_owned(
                db,
                run_id,
                "stale-worker",
                "evidence_investigator",
                81,
                "stale progress",
            )

    with db_session() as db:
        assert _emit_owned(
            db,
            run_id,
            "replacement-worker",
            "evidence_investigator",
            10,
            "replacement progress",
        )

    with db_session() as db:
        events = db.scalars(
            select(RunEventRow).where(RunEventRow.run_id == run_id)
        ).all()
    assert [(row.progress, row.message) for row in events] == [
        (10, "replacement progress")
    ]
