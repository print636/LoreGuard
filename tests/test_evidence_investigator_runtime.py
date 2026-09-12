from __future__ import annotations

import hashlib
import json

import pytest

from app.config import Settings
from app.domain import AnalysisCancelled, EvidenceSpan, ParsedDirective
from app.embeddings import EmbeddingInputError
from app.evidence_chunks import EvidenceChunker
from app.evidence_investigator_runtime import (
    EvidenceInvestigatorRuntime,
    InvestigatorUsageAccumulator,
    _DeadlineEmbeddingProvider,
    build_frozen_investigation_bundle,
)
from app.pipeline import DocumentInput
from app.provider import (
    OpenAICompatibleProvider,
    ProviderCallTelemetry,
    ProviderRetryExhausted,
    ProviderToolCall,
    ToolCallResult,
)
from app.service import WorkerLeaseLost


CONTENT = "岚的发色是银色。\n岚的发色是黑色。"


def configured_settings(**overrides) -> Settings:
    values = {
        "openai_api_key": "unit-test-placeholder",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock",
        "enable_evidence_investigator": True,
        "enable_embeddings": True,
        "embedding_base_url": "https://embedding.invalid/v1",
        "embedding_model": "mock-embedding",
        "embedding_model_revision": "r1",
        "embedding_deployment_fingerprint": "test-deployment-v1",
        "embedding_dimensions": 2,
        "evidence_investigator_require_hybrid": False,
        "evidence_investigator_max_completion_tokens": 64,
        "evidence_investigator_token_budget": 8_000,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def snapshot_inputs(content: str = CONTENT):
    document = DocumentInput(
        id="doc-1",
        name="chapter.md",
        content=content,
        role="chapter",
        scope="main",
    )
    metadata = {
        "document_id": "doc-1",
        "document_name": "chapter.md",
        "document_version": 3,
        "document_role": "chapter",
        "story_scope": "main",
        "context_explicit": True,
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "char_count": len(content),
        "ordinal": 0,
    }
    return [document], [metadata]


def frozen_bundle(content: str = CONTENT):
    documents, metadata = snapshot_inputs(content)
    return build_frozen_investigation_bundle(
        project_id="project-a", documents=documents, metadata=metadata
    )


def baseline_directive(*, text: str = "岚的发色是银色。") -> ParsedDirective:
    return ParsedDirective(
        kind="fact",
        attrs={
            "subject": "岚",
            "predicate": "发色",
            "value": "银色",
            "modality": "asserted",
            "source_scope": "narrator",
            "certainty": "certain",
            "story_scope": "main",
            "document_role": "chapter",
        },
        evidence=EvidenceSpan(
            document_id="doc-1",
            document_name="chapter.md",
            line_start=1,
            line_end=1,
            text=text,
        ),
        provenance_sources=frozenset({"baseline"}),
    )


def _find_string(value, key: str) -> str:
    if isinstance(value, dict):
        candidate = value.get(key)
        if isinstance(candidate, str):
            return candidate
        for nested in value.values():
            try:
                return _find_string(nested, key)
            except LookupError:
                pass
    elif isinstance(value, list):
        for nested in value:
            try:
                return _find_string(nested, key)
            except LookupError:
                pass
    raise LookupError(key)


def _telemetry(*, category="success", request_id="CANARYREQUEST"):
    return ProviderCallTelemetry(
        input_chars=100,
        response_chars=20,
        received_bytes=20,
        prompt_tokens=7,
        completion_tokens=2,
        elapsed_ms=3,
        category=category,
        attempt_no=1,
        request_id=request_id,
    )


class SuccessfulProvider:
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
                "query": "岚的发色是否矛盾",
                "entity_terms": ["岚"],
            }
        elif self.calls == 2:
            name = "READ_SPAN"
            arguments = {
                "seed_ref": seed_ref,
                "result_ref": _find_string(state, "result_ref"),
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
                        "span_ref": _find_string(state, "span_ref"),
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
                    id=f"call_{self.calls}", name=name, arguments=arguments
                ),
            ),
            prompt_tokens=7,
            completion_tokens=2,
            telemetry=_telemetry(),
        )


class PreparedRag:
    def __init__(self, scope, *, diagnostics=None):
        self.scope = scope
        self.prepare_calls = 0
        self.search_calls = 0
        self._diagnostics = diagnostics
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

    def prepare(self):
        self.prepare_calls += 1

    def search(self, *, seed, query, limit):
        del seed, query
        self.search_calls += 1
        return (self.chunk,)[:limit]

    def safe_diagnostics(self):
        if self._diagnostics is not None:
            return self._diagnostics
        return {
            "index": {
                "outcome": "complete",
                "reason": None,
                "profile_id": "profile@123",
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
                "query_calls": self.search_calls,
                "query_input_chars": 12 * self.search_calls,
            },
        }


def test_bundle_strictly_zips_all_three_frozen_views_and_rejects_mismatch():
    documents, metadata = snapshot_inputs()
    bundle = build_frozen_investigation_bundle(
        project_id="project-a", documents=documents, metadata=metadata
    )

    assert bundle.evidence_documents[0].content == CONTENT
    assert bundle.scoped_documents[0].content == CONTENT
    assert bundle.trusted_contexts[0].document_name == "chapter.md"
    assert len(bundle.fingerprint) == 64

    tampered = [dict(metadata[0], document_version=4, ordinal=True)]
    with pytest.raises(ValueError, match="snapshot bundle"):
        build_frozen_investigation_bundle(
            project_id="project-a", documents=documents, metadata=tampered
        )
    with pytest.raises(ValueError, match="cardinality"):
        build_frozen_investigation_bundle(
            project_id="project-a", documents=documents, metadata=[]
        )


def test_runtime_no_seed_is_zero_provider_embedding_and_rag_activity():
    provider = SuccessfulProvider()
    rag_created = 0

    def rag_factory(**_kwargs):
        nonlocal rag_created
        rag_created += 1
        raise AssertionError("no-seed run must not create a RAG runtime")

    outcome = EvidenceInvestigatorRuntime(
        session_factory=lambda: None,
        settings=configured_settings(),
        native_provider=provider,
        rag_factory=rag_factory,
    ).run(
        run_id="run-a",
        bundle=frozen_bundle(),
        baseline_directives=(),
        baseline_issues=(),
        remaining_run_tokens=20_000,
    )

    assert (outcome.outcome, outcome.reason_code) == ("skipped", "no_seeds")
    assert provider.calls == 0
    assert rag_created == 0
    assert outcome.diagnostics["usage"] is None


def test_runtime_real_loop_promotes_only_deterministically_reproduced_issue():
    provider = SuccessfulProvider()
    created = []

    def rag_factory(**kwargs):
        rag = PreparedRag(kwargs["scope"])
        created.append(rag)
        return rag

    usage = InvestigatorUsageAccumulator()
    outcome = EvidenceInvestigatorRuntime(
        session_factory=lambda: None,
        settings=configured_settings(),
        native_provider=provider,
        embedding_provider=object(),
        rag_factory=rag_factory,
        usage=usage,
    ).run(
        run_id="run-a",
        bundle=frozen_bundle(),
        baseline_directives=(baseline_directive(),),
        baseline_issues=(),
        remaining_run_tokens=20_000,
    )

    assert outcome.outcome == "completed"
    assert outcome.applied is True
    assert outcome.promotion is not None
    assert outcome.promotion.accepted_candidates == 1
    assert len(outcome.promotion.directives) == 2
    assert len(outcome.promotion.issues) == 1
    assert created[0].prepare_calls == 1
    assert created[0].search_calls == 1
    assert provider.calls == 3
    assert usage.logical_calls == 3
    assert usage.reported_prompt_tokens == 21
    assert usage.reported_completion_tokens == 6
    assert usage.charged_tokens >= 27
    serialized = json.dumps(outcome.diagnostics, ensure_ascii=False)
    assert CONTENT not in serialized
    assert "CANARYREQUEST" not in serialized
    assert "span_" not in serialized
    assert "result_" not in serialized
    assert outcome.diagnostics["rag"]["runtime_embedding_quota"] == {
        "calls": 0,
        "input_chars": 0,
        "max_input_chars": 250_000,
        "accounting": "resource_quota_not_chat_tokens_or_api_cost",
    }


def test_embedding_input_char_quota_is_hard_and_separate_from_chat_usage():
    class EmbeddingProvider:
        def __init__(self):
            self.calls = []

        @property
        def profile(self):
            return object()

        def embed(self, texts):
            self.calls.append(tuple(texts))
            return object()

    provider = EmbeddingProvider()
    bounded = _DeadlineEmbeddingProvider(
        provider,
        settings=configured_settings(),
        deadline=100.0,
        checkpoint=lambda: None,
        monotonic=lambda: 0.0,
        max_input_chars=5,
    )

    assert bounded.embed(("abc",)) is not None
    with pytest.raises(EmbeddingInputError, match="resource quota"):
        bounded.embed(("def",))

    assert provider.calls == [("abc",)]
    assert bounded.safe_activity() == {
        "calls": 1,
        "input_chars": 3,
        "max_input_chars": 5,
        "accounting": "resource_quota_not_chat_tokens_or_api_cost",
    }


def test_native_tool_turns_share_one_absolute_runtime_deadline():
    class FakeClock:
        def __init__(self):
            self.value = 0.0

        def __call__(self):
            return self.value

        def advance(self, seconds):
            self.value += seconds

    clock = FakeClock()
    scripted = SuccessfulProvider()

    class BoundedTurn:
        def __init__(self, effective_timeout):
            self.effective_timeout = effective_timeout

        def complete_with_tools(self, *args, **kwargs):
            intended_duration = 25.0
            if self.effective_timeout <= intended_duration:
                clock.advance(self.effective_timeout)
                raise ProviderRetryExhausted(
                    "bounded transport timeout", category="read_timeout"
                )
            clock.advance(intended_duration)
            return scripted.complete_with_tools(*args, **kwargs)

    class RecordingProvider(OpenAICompatibleProvider):
        def __init__(self):
            self.remaining_deadlines = []
            self.effective_timeouts = []

        def fork_for_evidence_investigator(self, *, remaining_deadline_seconds=None):
            self.remaining_deadlines.append(remaining_deadline_seconds)
            effective_timeout = min(30.0, remaining_deadline_seconds)
            self.effective_timeouts.append(effective_timeout)
            return BoundedTurn(effective_timeout)

    provider = RecordingProvider()
    outcome = EvidenceInvestigatorRuntime(
        session_factory=lambda: None,
        settings=configured_settings(),
        native_provider=provider,
        embedding_provider=object(),
        rag_factory=lambda **kwargs: PreparedRag(kwargs["scope"]),
        monotonic=clock,
    ).run(
        run_id="run-a",
        bundle=frozen_bundle(),
        baseline_directives=(baseline_directive(),),
        baseline_issues=(),
        remaining_run_tokens=20_000,
    )

    assert (outcome.outcome, outcome.reason_code) == ("degraded", "deadline")
    assert provider.remaining_deadlines == pytest.approx([60.0, 35.0, 10.0])
    assert provider.effective_timeouts == pytest.approx([30.0, 30.0, 10.0])
    assert clock.value == pytest.approx(60.0)
    assert scripted.calls == 2
    assert outcome.promotion is None


def test_invalid_anchor_fails_before_provider_or_rag_creation():
    provider = SuccessfulProvider()
    created = 0

    def rag_factory(**_kwargs):
        nonlocal created
        created += 1

    outcome = EvidenceInvestigatorRuntime(
        session_factory=lambda: None,
        settings=configured_settings(),
        native_provider=provider,
        rag_factory=rag_factory,
    ).run(
        run_id="run-a",
        bundle=frozen_bundle(),
        baseline_directives=(baseline_directive(text="伪造的锚点"),),
        baseline_issues=(),
        remaining_run_tokens=20_000,
    )

    assert (outcome.outcome, outcome.reason_code) == (
        "degraded",
        "internal_failure",
    )
    assert provider.calls == 0
    assert created == 0


def test_rag_diagnostic_boundary_drops_content_query_endpoint_key_and_refs():
    canary = "CANARY-STORY-BASEURL-APIKEY-SPANREF"
    provider = SuccessfulProvider()

    def rag_factory(**kwargs):
        return PreparedRag(
            kwargs["scope"],
            diagnostics={
                "index": {
                    "outcome": canary,
                    "reason": canary,
                    "profile_id": canary,
                    "chunker_version": canary,
                    "provider_calls": 1,
                },
                "retrievals": [
                    {"strategy": canary, "mode": canary, "query": canary}
                ],
                "total_retrievals": 1,
                "embedding_activity": {
                    "index_calls": 1,
                    "query_calls": 1,
                    "endpoint": canary,
                    "api_key": canary,
                },
                "span_ref": canary,
            },
        )

    outcome = EvidenceInvestigatorRuntime(
        session_factory=lambda: None,
        settings=configured_settings(),
        native_provider=provider,
        embedding_provider=object(),
        rag_factory=rag_factory,
    ).run(
        run_id="run-a",
        bundle=frozen_bundle(),
        baseline_directives=(baseline_directive(),),
        baseline_issues=(),
        remaining_run_tokens=20_000,
    )

    serialized = json.dumps(outcome.diagnostics, ensure_ascii=False)
    assert canary not in serialized
    assert "query" not in outcome.diagnostics["rag"]["retrievals"][0]
    assert "endpoint" not in serialized
    assert "api_key" not in serialized


class _Preflight:
    minimum_path_admissible = False
    minimum_initial_reservation = 9_999

    def safe_dict(self):
        return {
            "minimum_path_admissible": False,
            "minimum_initial_reservation": 9_999,
        }


class PreflightRejectingLoop:
    def __init__(self, **_kwargs):
        pass

    def budget_preflight(self):
        return _Preflight()

    def run(self):
        raise AssertionError("inadmissible preflight must stop before run")


def test_budget_preflight_happens_before_rag_prepare_or_external_call():
    provider = SuccessfulProvider()
    created = []

    def rag_factory(**kwargs):
        rag = PreparedRag(kwargs["scope"])
        created.append(rag)
        return rag

    outcome = EvidenceInvestigatorRuntime(
        session_factory=lambda: None,
        settings=configured_settings(),
        native_provider=provider,
        embedding_provider=object(),
        rag_factory=rag_factory,
        loop_factory=PreflightRejectingLoop,
    ).run(
        run_id="run-a",
        bundle=frozen_bundle(),
        baseline_directives=(baseline_directive(),),
        baseline_issues=(),
        remaining_run_tokens=100,
    )

    assert (outcome.outcome, outcome.reason_code) == ("skipped", "token_budget")
    assert created[0].prepare_calls == 0
    assert provider.calls == 0


class FailingProvider:
    def __init__(self, error):
        self.error = error
        self.calls = 0

    def complete_with_tools(self, *_args, **_kwargs):
        self.calls += 1
        raise self.error


@pytest.mark.parametrize(
    ("error", "expected_prompt"),
    [
        (
            ProviderRetryExhausted(
                "rate limited",
                category="rate_limit",
                http_status=429,
                telemetry=_telemetry(category="rate_limit"),
            ),
            7,
        ),
        (
            ProviderRetryExhausted(
                "rate limited", category="rate_limit", http_status=429
            ),
            0,
        ),
    ],
)
def test_provider_429_and_missing_usage_are_immediately_charged_and_local(error, expected_prompt):
    provider = FailingProvider(error)
    usage = InvestigatorUsageAccumulator()

    outcome = EvidenceInvestigatorRuntime(
        session_factory=lambda: None,
        settings=configured_settings(),
        native_provider=provider,
        embedding_provider=object(),
        rag_factory=lambda **kwargs: PreparedRag(kwargs["scope"]),
        usage=usage,
    ).run(
        run_id="run-a",
        bundle=frozen_bundle(),
        baseline_directives=(baseline_directive(),),
        baseline_issues=(),
        remaining_run_tokens=20_000,
    )

    assert (outcome.outcome, outcome.reason_code) == (
        "degraded",
        "loop_degraded",
    )
    assert usage.logical_calls == 1
    assert usage.reported_prompt_tokens == expected_prompt
    assert usage.charged_tokens > 0
    assert outcome.promotion is None


def test_success_usage_is_recorded_before_post_call_cancellation_propagates():
    cancellation = AnalysisCancelled("cancel after completed call")
    state = {"cancel": False}
    usage = InvestigatorUsageAccumulator()
    delegate = SuccessfulProvider()

    class Provider:
        def complete_with_tools(self, *args, **kwargs):
            result = delegate.complete_with_tools(*args, **kwargs)
            state["cancel"] = True
            return result

    def checkpoint():
        if state["cancel"]:
            raise cancellation

    runtime = EvidenceInvestigatorRuntime(
        session_factory=lambda: None,
        settings=configured_settings(),
        checkpoint=checkpoint,
        native_provider=Provider(),
        embedding_provider=object(),
        rag_factory=lambda **kwargs: PreparedRag(kwargs["scope"]),
        usage=usage,
        passthrough_exceptions=(AnalysisCancelled,),
    )

    with pytest.raises(AnalysisCancelled) as raised:
        runtime.run(
            run_id="run-a",
            bundle=frozen_bundle(),
            baseline_directives=(baseline_directive(),),
            baseline_issues=(),
            remaining_run_tokens=20_000,
        )

    assert raised.value is cancellation
    assert usage.logical_calls == 1
    assert usage.reported_prompt_tokens == 7
    assert usage.reported_completion_tokens == 2


def test_provider_error_usage_is_recorded_before_post_call_cancellation_propagates():
    cancellation = AnalysisCancelled("cancel after failed call")
    state = {"cancel": False}
    usage = InvestigatorUsageAccumulator()
    failure = ProviderRetryExhausted(
        "rate limited",
        category="rate_limit",
        http_status=429,
        telemetry=_telemetry(category="rate_limit"),
    )

    class Provider:
        def complete_with_tools(self, *_args, **_kwargs):
            state["cancel"] = True
            raise failure

    def checkpoint():
        if state["cancel"]:
            raise cancellation

    runtime = EvidenceInvestigatorRuntime(
        session_factory=lambda: None,
        settings=configured_settings(),
        checkpoint=checkpoint,
        native_provider=Provider(),
        embedding_provider=object(),
        rag_factory=lambda **kwargs: PreparedRag(kwargs["scope"]),
        usage=usage,
        passthrough_exceptions=(AnalysisCancelled,),
    )

    with pytest.raises(AnalysisCancelled) as raised:
        runtime.run(
            run_id="run-a",
            bundle=frozen_bundle(),
            baseline_directives=(baseline_directive(),),
            baseline_issues=(),
            remaining_run_tokens=20_000,
        )

    assert raised.value is cancellation
    assert usage.logical_calls == 1
    assert usage.reported_prompt_tokens == 7
    assert usage.reported_completion_tokens == 2


@pytest.mark.parametrize("signal_type", [AnalysisCancelled, WorkerLeaseLost])
def test_checkpoint_control_flow_passes_through_after_preserving_call_charge(signal_type):
    signal = signal_type("stop")
    usage = InvestigatorUsageAccumulator()
    provider = FailingProvider(signal)
    runtime = EvidenceInvestigatorRuntime(
        session_factory=lambda: None,
        settings=configured_settings(),
        native_provider=provider,
        embedding_provider=object(),
        rag_factory=lambda **kwargs: PreparedRag(kwargs["scope"]),
        usage=usage,
        passthrough_exceptions=(signal_type,),
    )

    with pytest.raises(signal_type) as raised:
        runtime.run(
            run_id="run-a",
            bundle=frozen_bundle(),
            baseline_directives=(baseline_directive(),),
            baseline_issues=(),
            remaining_run_tokens=20_000,
        )

    assert raised.value is signal
    assert usage.logical_calls == 1
    assert usage.charged_tokens > 0
