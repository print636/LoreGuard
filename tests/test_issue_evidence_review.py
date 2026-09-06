from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from dataclasses import replace
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.db import (
    AnalysisDiagnosticRow,
    AnalysisRunRow,
    Base,
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
    Severity,
)
from app.embeddings import EmbeddingProfile
from app.evidence_chunks import EvidenceChunker, SnapshotDocumentKey
from app.evidence_rag import (
    EvidenceDocument,
    EvidenceIndexDiagnostics,
    EvidenceIndexResult,
    EvidenceRetrievalResult,
    RankedEvidence,
    RetrievalDiagnostics,
)
from app.issue_evidence_review import (
    IssueEvidenceReviewer,
    IssueEvidenceReviewResult,
    IssueEvidenceReviewUsageAccumulator,
)
from app.main import enforce_daily_model_budget
from app.provider import ModelResult, ProviderCallTelemetry, ProviderError
from app.service import capture_run_inputs, execute_analysis


class _ForbiddenChat:
    def complete(self, system, user):
        raise AssertionError("chat provider must not be called")


class _ScriptedChat:
    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = []

    def complete(self, system, user):
        self.calls.append((system, user))
        if self.error is not None:
            raise self.error
        return ModelResult(
            text=json.dumps(self.payload, ensure_ascii=False),
            prompt_tokens=100,
            completion_tokens=20,
            telemetry=ProviderCallTelemetry(
                input_chars=len(system) + len(user),
                response_chars=80,
                received_bytes=80,
                prompt_tokens=100,
                completion_tokens=20,
                elapsed_ms=9,
                category="success",
                attempt_no=1,
            ),
        )


class _Coordinator:
    def __init__(self, result):
        self.result = result
        self.documents = None

    def ensure_index(self, documents):
        self.documents = tuple(documents)
        return self.result


class _Retriever:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def retrieve(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class IssueEvidenceReviewTests(unittest.TestCase):
    def setUp(self):
        self.content = "守则写明银钥只能由林澈持有。\n第二章中顾青使用了银钥开门。"
        self.snapshot = SnapshotDocumentKey(
            project_id="project-1",
            document_id="document-1",
            document_version=3,
            content_sha256=hashlib.sha256(self.content.encode()).hexdigest(),
        )
        self.document = EvidenceDocument(
            snapshot=self.snapshot, content=self.content
        )
        self.profile = EmbeddingProfile.openai_compatible(
            provider_namespace="unit",
            model_identifier="mock",
            model_revision="r1",
            deployment_fingerprint="unit-runtime-v1",
            dimensions=2,
        )
        chunker = EvidenceChunker(
            target_chars=20, min_chars=10, max_chars=40, overlap_chars=0
        )
        self.chunks = chunker.chunk(
            project_id=self.snapshot.project_id,
            document_id=self.snapshot.document_id,
            document_version=self.snapshot.document_version,
            content=self.content,
            content_sha256=self.snapshot.content_sha256,
        )
        self.indexed = EvidenceIndexResult(
            profile=self.profile,
            snapshots=(self.snapshot,),
            chunks=self.chunks,
            diagnostics=EvidenceIndexDiagnostics(
                outcome="complete",
                reason=None,
                profile_id=self.profile.profile_id,
                chunker_version=self.chunks[0].chunker_version,
                document_count=1,
                expected_chunks=len(self.chunks),
                reused_chunks=0,
                embedded_chunks=len(self.chunks),
                provider_calls=1,
                provider_input_chars=len(self.content),
                elapsed_ms=1,
            ),
        )
        matches = tuple(
            RankedEvidence(
                chunk=chunk,
                rrf_score=0.03,
                keyword_rank=index,
                vector_rank=index,
                entity_rank=None,
            )
            for index, chunk in enumerate(self.chunks, start=1)
        )
        self.retrieved = EvidenceRetrievalResult(
            matches=matches,
            diagnostics=RetrievalDiagnostics(
                strategy="keyword+dense-rrf",
                mode="hybrid",
                reason=None,
                profile_id=self.profile.profile_id,
                chunker_version=self.chunks[0].chunker_version,
                candidate_count=len(self.chunks),
                keyword_hits=len(self.chunks),
                vector_hits=len(self.chunks),
                entity_hits=0,
                result_count=len(self.chunks),
                provider_calls=1,
                provider_input_chars=10,
                elapsed_ms=1,
            ),
        )
        self.issue = ConsistencyIssue(
            category=IssueCategory.item_ownership,
            severity=Severity.high,
            confidence=0.9,
            title="银钥持有冲突",
            explanation="顾青使用了仅限林澈持有的银钥",
            evidence=[
                EvidenceSpan(
                    document_id="document-1",
                    document_name="story.md",
                    line_start=1,
                    line_end=2,
                    text=self.content,
                )
            ],
            suggestion="确认银钥是否发生转移",
            metadata={"rule": "item_state"},
        )

    def settings(self, **overrides):
        values = {
            "enable_issue_evidence_review": True,
            "openai_api_key": "unit-secret",
            "openai_base_url": "https://unit.invalid/v1",
            "openai_model": "unit-model",
        }
        values.update(overrides)
        return Settings(**values)

    def reviewer(self, chat, *, indexed=None, retrieved=None, checkpoint=None, settings=None):
        return IssueEvidenceReviewer(
            session_factory=lambda: None,
            settings=settings or self.settings(),
            embedding_provider=object(),
            chat_provider=chat,
            coordinator=_Coordinator(indexed or self.indexed),
            retriever=_Retriever(retrieved or self.retrieved),
            checkpoint=checkpoint,
        )

    def test_success_adds_safe_annotation_without_mutating_issue(self):
        before = self.issue.model_dump(mode="json")
        chat = _ScriptedChat(
            {
                "reviews": [
                    {
                        "issue_ref": "I01",
                        "verdict": "supports_issue",
                        "note": "两处证据直接构成持有与使用冲突。",
                        "citations": ["E01"],
                    }
                ]
            }
        )
        coordinator = _Coordinator(self.indexed)
        retriever = _Retriever(self.retrieved)
        reviewer = IssueEvidenceReviewer(
            session_factory=lambda: None,
            settings=self.settings(),
            embedding_provider=object(),
            chat_provider=chat,
            coordinator=coordinator,
            retriever=retriever,
        )
        result = reviewer.review(
            documents=[self.document], issues=[self.issue], remaining_run_tokens=8_000
        )

        self.assertEqual(before, self.issue.model_dump(mode="json"))
        self.assertEqual("completed", result.diagnostics["outcome"])
        annotation = result.annotations[str(self.issue.id)]
        self.assertEqual("supports_issue", annotation["verdict"])
        consumed = annotation["consumed_evidence"][0]
        self.assertEqual(self.snapshot.project_id, consumed["project_id"])
        self.assertEqual(self.snapshot.document_version, consumed["document_version"])
        self.assertEqual(self.snapshot.content_sha256, consumed["content_sha256"])
        self.assertEqual("hybrid", annotation["retrieval"]["mode"])
        self.assertNotIn(self.content, repr(annotation))
        self.assertNotIn("unit-secret", repr(result.diagnostics))
        self.assertEqual((self.snapshot,), retriever.calls[0]["allowed_snapshots"])
        self.assertEqual((self.document,), coordinator.documents)
        prompt_payload = json.loads(chat.calls[0][1])
        self.assertEqual(
            self.content,
            prompt_payload["issues"][0]["retrieved_evidence"][0]["text"],
        )

    def test_forged_citation_rejects_entire_batch(self):
        chat = _ScriptedChat(
            {
                "reviews": [
                    {
                        "issue_ref": "I01",
                        "verdict": "supports_issue",
                        "note": "伪造引用",
                        "citations": ["E99"],
                    }
                ]
            }
        )
        result = self.reviewer(chat).review(
            documents=[self.document], issues=[self.issue], remaining_run_tokens=8_000
        )
        self.assertEqual({}, result.annotations)
        self.assertEqual(1, result.diagnostics["rejected_batches"])
        self.assertIn("invalid_model_response", result.diagnostics["reason_codes"])

    def test_provider_failure_is_content_free_and_preserves_issue(self):
        before = self.issue.model_dump(mode="json")
        error = ProviderError(
            "secret upstream body",
            category="rate_limit",
            http_status=429,
        )
        usage = IssueEvidenceReviewUsageAccumulator()
        reviewer = IssueEvidenceReviewer(
            session_factory=lambda: None,
            settings=self.settings(),
            embedding_provider=object(),
            chat_provider=_ScriptedChat(error=error),
            coordinator=_Coordinator(self.indexed),
            retriever=_Retriever(self.retrieved),
            usage_accounting=usage.record,
        )
        result = reviewer.review(
            documents=[self.document], issues=[self.issue], remaining_run_tokens=8_000
        )
        self.assertEqual(before, self.issue.model_dump(mode="json"))
        self.assertEqual({}, result.annotations)
        self.assertIn("rate_limit", result.diagnostics["reason_codes"])
        self.assertNotIn("secret upstream body", repr(result.diagnostics))
        self.assertEqual(1, usage.logical_calls)
        self.assertGreater(usage.charged_tokens, 0)
        self.assertEqual(0, usage.prompt_tokens)

    def test_incomplete_embedding_index_is_explicitly_degraded(self):
        incomplete = replace(
            self.indexed,
            diagnostics=replace(
                self.indexed.diagnostics,
                outcome="provider_unavailable",
                reason="embedding_not_configured",
            ),
        )
        result = self.reviewer(_ForbiddenChat(), indexed=incomplete).review(
            documents=[self.document], issues=[self.issue], remaining_run_tokens=8_000
        )
        self.assertEqual("degraded", result.diagnostics["outcome"])
        self.assertIn("embedding_not_configured", result.diagnostics["reason_codes"])
        self.assertEqual([], result.diagnostics["provider_calls"])

    def test_lexical_fallback_cannot_masquerade_as_hybrid_rag(self):
        lexical = replace(
            self.retrieved,
            diagnostics=replace(
                self.retrieved.diagnostics,
                mode="lexical_only",
                reason="vector_search_unavailable",
                vector_hits=0,
            ),
        )
        result = self.reviewer(_ForbiddenChat(), retrieved=lexical).review(
            documents=[self.document], issues=[self.issue], remaining_run_tokens=8_000
        )
        self.assertEqual({}, result.annotations)
        self.assertEqual("degraded", result.diagnostics["outcome"])
        self.assertIn("hybrid_unavailable", result.diagnostics["reason_codes"])

    def test_default_disabled_does_not_touch_index_or_chat(self):
        coordinator = _Coordinator(self.indexed)
        reviewer = IssueEvidenceReviewer(
            session_factory=lambda: None,
            settings=Settings(),
            embedding_provider=object(),
            chat_provider=_ForbiddenChat(),
            coordinator=coordinator,
            retriever=_Retriever(self.retrieved),
        )
        result = reviewer.review(
            documents=[self.document], issues=[self.issue], remaining_run_tokens=8_000
        )
        self.assertEqual("disabled", result.diagnostics["outcome"])
        self.assertIsNone(coordinator.documents)

    def test_checkpoint_cancellation_propagates_before_indexing(self):
        class Cancelled(RuntimeError):
            pass

        def cancel():
            raise Cancelled("cancelled")

        coordinator = _Coordinator(self.indexed)
        reviewer = IssueEvidenceReviewer(
            session_factory=lambda: None,
            settings=self.settings(),
            embedding_provider=object(),
            chat_provider=_ForbiddenChat(),
            coordinator=coordinator,
            retriever=_Retriever(self.retrieved),
            checkpoint=cancel,
        )
        with self.assertRaises(Cancelled):
            reviewer.review(
                documents=[self.document],
                issues=[self.issue],
                remaining_run_tokens=8_000,
            )
        self.assertIsNone(coordinator.documents)

    def test_one_deadline_includes_cold_indexing(self):
        clock = [0.0]

        class SlowCoordinator(_Coordinator):
            def ensure_index(_, documents):
                clock[0] = 2.0
                return super().ensure_index(documents)

        reviewer = IssueEvidenceReviewer(
            session_factory=lambda: None,
            settings=self.settings(
                issue_evidence_review_total_deadline_seconds=1.0
            ),
            embedding_provider=object(),
            chat_provider=_ForbiddenChat(),
            coordinator=SlowCoordinator(self.indexed),
            retriever=_Retriever(self.retrieved),
            monotonic=lambda: clock[0],
        )
        result = reviewer.review(
            documents=[self.document], issues=[self.issue], remaining_run_tokens=8_000
        )
        self.assertEqual("degraded", result.diagnostics["outcome"])
        self.assertIn("deadline", result.diagnostics["reason_codes"])

    def test_one_deadline_includes_each_query_retrieval(self):
        clock = [0.0]

        class SlowRetriever(_Retriever):
            def retrieve(_, **kwargs):
                result = super().retrieve(**kwargs)
                clock[0] = 2.0
                return result

        reviewer = IssueEvidenceReviewer(
            session_factory=lambda: None,
            settings=self.settings(
                issue_evidence_review_total_deadline_seconds=1.0
            ),
            embedding_provider=object(),
            chat_provider=_ForbiddenChat(),
            coordinator=_Coordinator(self.indexed),
            retriever=SlowRetriever(self.retrieved),
            monotonic=lambda: clock[0],
        )
        result = reviewer.review(
            documents=[self.document], issues=[self.issue], remaining_run_tokens=8_000
        )
        self.assertEqual("degraded", result.diagnostics["outcome"])
        self.assertIn("deadline", result.diagnostics["reason_codes"])

    def test_mixed_project_scope_fails_before_chat(self):
        other_content = "另一个项目中的文本。"
        other = EvidenceDocument(
            snapshot=SnapshotDocumentKey(
                project_id="project-2",
                document_id="document-2",
                document_version=1,
                content_sha256=hashlib.sha256(other_content.encode()).hexdigest(),
            ),
            content=other_content,
        )
        # Use the real coordinator's scope validation. Its provider must never
        # be consulted because mixed projects are rejected first.
        class ForbiddenEmbedding:
            @property
            def profile(self):
                raise AssertionError("embedding provider must not be consulted")

        reviewer = IssueEvidenceReviewer(
            session_factory=lambda: None,
            settings=self.settings(),
            embedding_provider=ForbiddenEmbedding(),
            chat_provider=_ForbiddenChat(),
        )
        result = reviewer.review(
            documents=[self.document, other],
            issues=[self.issue],
            remaining_run_tokens=8_000,
        )
        self.assertEqual("degraded", result.diagnostics["outcome"])
        self.assertEqual({}, result.annotations)

    def test_settings_enforce_resume_safe_upper_bounds(self):
        invalid = (
            {"issue_evidence_review_max_issues": 9},
            {"issue_evidence_review_top_k": 7},
            {"issue_evidence_review_batch_size": 5},
            {"issue_evidence_review_max_evidence_chars": 6_001},
            {"issue_evidence_review_token_budget": 6_001},
            {"issue_evidence_review_timeout_seconds": 21},
            {"issue_evidence_review_total_deadline_seconds": 46},
            {"issue_evidence_review_max_completion_tokens": 1_401},
            {"issue_evidence_review_max_response_bytes": 64_001},
        )
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                Settings(**values)


class IssueEvidenceReviewServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        database = Path(self.temp_dir.name) / "service-review.db"
        self.engine = create_engine(
            f"sqlite:///{database.as_posix()}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)

    def tearDown(self):
        self.engine.dispose()
        self.temp_dir.cleanup()

    def test_service_uses_frozen_input_and_only_adds_metadata_annotation(self):
        content = "规则：银钥仅由林澈持有。\n顾青使用银钥开门。"
        with self.Session() as db:
            project = ProjectRow(name="服务测试", description="")
            db.add(project)
            db.flush()
            document = DocumentRow(
                project_id=project.id,
                name="story.md",
                content=content,
                version=2,
            )
            run = AnalysisRunRow(project_id=project.id)
            db.add_all([document, run])
            db.flush()
            capture_run_inputs(db, run, [document])
            db.commit()
            run_id = run.id
            document_id = document.id

        issue = ConsistencyIssue(
            category=IssueCategory.item_ownership,
            severity=Severity.high,
            confidence=0.91,
            title="银钥持有冲突",
            explanation="顾青使用了仅限林澈持有的银钥",
            evidence=[
                EvidenceSpan(
                    document_id=document_id,
                    document_name="story.md",
                    line_start=1,
                    line_end=2,
                    text=content,
                )
            ],
            suggestion="确认是否发生转移",
            metadata={"rule": "item_state"},
        )

        class FakePipeline:
            def run(_, documents, on_stage, checkpoint):
                checkpoint()
                on_stage("check", 75, "规则检查完成")
                return SimpleNamespace(
                    directives=[],
                    issues=[issue],
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
                        }
                    },
                )

        captured = {}

        class FakeReviewer:
            def __init__(_, **kwargs):
                captured["settings"] = kwargs["settings"]

            def review(_, *, documents, issues, remaining_run_tokens):
                captured["documents"] = documents
                captured["issues"] = issues
                captured["remaining"] = remaining_run_tokens
                return IssueEvidenceReviewResult(
                    annotations={
                        str(issue.id): {
                            "schema_version": "issue-evidence-review-v1",
                            "verdict": "supports_issue",
                            "note": "证据支持",
                            "citations": ["E01"],
                            "consumed_evidence": [],
                        }
                    },
                    diagnostics={
                        "enabled": True,
                        "outcome": "completed",
                        "selected_issues": 1,
                        "reviewed_issues": 1,
                    },
                    prompt_tokens=30,
                    completion_tokens=5,
                    charged_tokens=100,
                )

        settings = Settings(
            database_url="sqlite:///unused.db",
            enable_issue_evidence_review=True,
            openai_api_key="unit-secret",
            openai_model="unit-model",
        )
        with (
            patch("app.service.SessionLocal", self.Session),
            patch("app.service.AnalysisPipeline", FakePipeline),
            patch("app.service.IssueEvidenceReviewer", FakeReviewer),
            patch("app.service.get_settings", return_value=settings),
        ):
            execute_analysis(run_id)

        self.assertEqual(content, captured["documents"][0].content)
        self.assertEqual(2, captured["documents"][0].snapshot.document_version)
        self.assertEqual((issue,), captured["issues"])
        self.assertEqual(19_980, captured["remaining"])
        with self.Session() as db:
            saved_issue = db.scalar(select(IssueRow).where(IssueRow.run_id == run_id))
            saved_run = db.get(AnalysisRunRow, run_id)
            diagnostic = db.get(AnalysisDiagnosticRow, run_id)
            events = db.scalars(
                select(RunEventRow)
                .where(RunEventRow.run_id == run_id)
                .order_by(RunEventRow.id)
            ).all()
        self.assertEqual(issue.category.value, saved_issue.category)
        self.assertEqual(issue.severity.value, saved_issue.severity)
        self.assertEqual(issue.confidence, saved_issue.confidence)
        self.assertEqual(issue.title, saved_issue.title)
        self.assertEqual(issue.explanation, saved_issue.explanation)
        self.assertEqual(issue.suggestion, saved_issue.suggestion)
        self.assertEqual("item_state", saved_issue.extra["rule"])
        self.assertEqual(
            "supports_issue", saved_issue.extra["ai_evidence_review"]["verdict"]
        )
        self.assertEqual(42, saved_run.prompt_tokens)
        self.assertEqual(13, saved_run.completion_tokens)
        self.assertEqual(
            "completed", diagnostic.payload["ai_evidence_review"]["outcome"]
        )
        review_events = [row for row in events if row.stage == "evidence_review"]
        self.assertEqual([82, 88], [row.progress for row in review_events])
        self.assertEqual(
            sorted(row.progress for row in events), [row.progress for row in events]
        )

    def test_model_attempt_then_cancel_persists_conservative_debit(self):
        with self.Session() as db:
            project = ProjectRow(name="取消记账", description="")
            db.add(project)
            db.flush()
            document = DocumentRow(
                project_id=project.id,
                name="story.md",
                content="规则文本",
                version=1,
            )
            run = AnalysisRunRow(project_id=project.id)
            db.add_all([document, run])
            db.flush()
            capture_run_inputs(db, run, [document])
            db.commit()
            run_id = run.id

        issue = ConsistencyIssue(
            category=IssueCategory.fact_conflict,
            severity=Severity.high,
            confidence=0.9,
            title="事实冲突",
            explanation="两项事实不一致",
            evidence=[
                EvidenceSpan(
                    document_id="doc",
                    document_name="story.md",
                    line_start=1,
                    line_end=1,
                    text="规则文本",
                )
            ],
            suggestion="核对事实",
        )

        class FakePipeline:
            def run(_, documents, on_stage, checkpoint):
                on_stage("check", 75, "规则完成")
                return SimpleNamespace(
                    directives=[],
                    issues=[issue],
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
                        }
                    },
                )

        class CancelAfterAttemptReviewer:
            def __init__(_, **kwargs):
                _.account = kwargs["usage_accounting"]

            def review(_, **kwargs):
                _.account(None, True, 0, 0, 1_234, "success")
                raise AnalysisCancelled("cancel after provider")

        settings = Settings(
            enable_issue_evidence_review=True,
            openai_api_key="unit-secret",
            openai_model="unit-model",
            daily_token_budget=1_000,
        )
        with (
            patch("app.service.SessionLocal", self.Session),
            patch("app.service.AnalysisPipeline", FakePipeline),
            patch(
                "app.service.IssueEvidenceReviewer", CancelAfterAttemptReviewer
            ),
            patch("app.service.get_settings", return_value=settings),
        ):
            execute_analysis(run_id)

        with self.Session() as db:
            saved_run = db.get(AnalysisRunRow, run_id)
            diagnostic = db.get(AnalysisDiagnosticRow, run_id)
        self.assertEqual("cancelled", saved_run.status)
        accounting = diagnostic.payload["usage_accounting"]
        self.assertEqual(1_234, accounting["charged_tokens"])
        self.assertEqual(0, saved_run.prompt_tokens + saved_run.completion_tokens)
        with self.Session() as db, patch("app.main.settings", settings):
            with self.assertRaisesRegex(Exception, "Token 预算已用尽"):
                enforce_daily_model_budget(db)

    def _exercise_retried_usage(self, *, second_attempt_succeeds: bool):
        with self.Session() as db:
            project = ProjectRow(name="重试累计", description="")
            db.add(project)
            db.flush()
            document = DocumentRow(
                project_id=project.id,
                name="story.md",
                content="重试文本",
                version=1,
            )
            run = AnalysisRunRow(project_id=project.id)
            db.add_all([document, run])
            db.flush()
            capture_run_inputs(db, run, [document])
            db.commit()
            run_id = run.id

        attempts = [0]

        class SequencedPipeline:
            def run(_, documents, on_stage, checkpoint):
                attempts[0] += 1
                on_stage("check", 75, "规则完成")
                malformed = attempts[0] == 1 or not second_attempt_succeeds
                return SimpleNamespace(
                    directives=[object()] if malformed else [],
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
                        }
                    },
                )

        class AccountingReviewer:
            def __init__(_, **kwargs):
                _.account = kwargs["usage_accounting"]

            def review(_, **kwargs):
                _.account(None, True, 10, 5, 1_000, "success")
                return IssueEvidenceReviewResult(
                    annotations={},
                    diagnostics={"enabled": True, "outcome": "completed"},
                    prompt_tokens=10,
                    completion_tokens=5,
                    charged_tokens=1_000,
                )

        settings = Settings(
            enable_issue_evidence_review=True,
            openai_api_key="unit-secret",
            openai_model="unit-model",
        )
        patches = (
            patch("app.service.SessionLocal", self.Session),
            patch("app.service.AnalysisPipeline", SequencedPipeline),
            patch("app.service.IssueEvidenceReviewer", AccountingReviewer),
            patch("app.service.get_settings", return_value=settings),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaises(AttributeError):
                execute_analysis(
                    run_id,
                    raise_on_failure=True,
                    finalize_failure=False,
                )
            with self.Session() as db:
                queued = db.get(AnalysisRunRow, run_id)
                first_diagnostic = db.get(AnalysisDiagnosticRow, run_id)
                self.assertEqual("queued", queued.status)
                self.assertEqual(
                    1_000,
                    first_diagnostic.payload["usage_accounting"]["charged_tokens"],
                )
            queued_budget = settings.model_copy(update={"daily_token_budget": 999})
            with self.Session() as db, patch("app.main.settings", queued_budget):
                with self.assertRaisesRegex(Exception, "Token 预算已用尽"):
                    enforce_daily_model_budget(db)
            if second_attempt_succeeds:
                execute_analysis(run_id, raise_on_failure=True)
            else:
                with self.assertRaises(AttributeError):
                    execute_analysis(run_id, raise_on_failure=True)

        with self.Session() as db:
            saved_run = db.get(AnalysisRunRow, run_id)
            diagnostic = db.get(AnalysisDiagnosticRow, run_id)
        self.assertEqual(
            "completed" if second_attempt_succeeds else "failed", saved_run.status
        )
        accounting = diagnostic.payload["usage_accounting"]
        self.assertEqual(2, accounting["logical_calls"])
        self.assertEqual(20, accounting["prompt_tokens"])
        self.assertEqual(10, accounting["completion_tokens"])
        self.assertEqual(2_000, accounting["charged_tokens"])
        self.assertEqual(30, saved_run.prompt_tokens + saved_run.completion_tokens)

    def test_retriable_failure_then_success_accumulates_usage(self):
        self._exercise_retried_usage(second_attempt_succeeds=True)

    def test_retriable_failure_then_terminal_failure_accumulates_usage(self):
        self._exercise_retried_usage(second_attempt_succeeds=False)


if __name__ == "__main__":
    unittest.main()
