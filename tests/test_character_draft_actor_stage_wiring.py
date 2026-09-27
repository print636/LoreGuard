from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

import app.character_consistency_stage as stage_module
from app.character_consistency_stage import (
    CharacterConsistencyStage,
    _FrozenDocument,
    project_provisional_draft_clues,
)
from app.character_drift import ConfirmedTraitSnapshot
from app.character_scope_review import ScopeReviewSourceIdentity
from app.character_trait_extraction import (
    CharacterSignal,
    CharacterSignalChunk,
    CharacterSignalDiagnostics,
    CharacterSignalExtractionResult,
    CharacterSignalTokenAdmission,
)
from app.config import Settings
from app.domain import EvidenceSpan
from app.narrative_context import NarrativeScopeV1
from app.pipeline import DocumentInput


def _settings(**updates: Any) -> Settings:
    values: dict[str, Any] = {
        "openai_api_key": "stage-wiring-test",
        "openai_base_url": "https://mock.invalid/v1",
        "openai_model": "mock-model",
        "enable_character_consistency": True,
        "provider_max_attempts": 1,
        "character_consistency_stage_token_budget": 20_000,
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def _draft_source(content: str) -> _FrozenDocument:
    document = DocumentInput(
        id="draft-document-1",
        name="draft.md",
        content=content,
        role="chapter",
    )
    return _FrozenDocument(
        input_id="run-input-1",
        document=document,
        document_version=7,
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        ordinal=1,
        source_kind="draft",
        source_reason="draft",
        scope=NarrativeScopeV1(),
        resolution_state="confirmed",
        publication_status="draft",
        authority_tier="working_material",
    )


def _baseline_entry():
    candidate_id = "109a1df4-6d2f-40c1-bc45-70b274bf3a39"
    baseline = ConfirmedTraitSnapshot(
        id=f"ct_{candidate_id}",
        character="祁雾",
        dimension="core_personality",
        trait_key="directness",
        statement="祁雾说话直来直往",
        polarity="positive",
        stability="core",
        origin="explicit_setting",
        evidence=(
            EvidenceSpan(
                document_id="profile-document-1",
                document_name="profile.md",
                line_start=1,
                line_end=1,
                text="祁雾说话直来直往。",
            ),
        ),
    )
    return (
        SimpleNamespace(candidate_id=candidate_id, payload={}),
        baseline,
        NarrativeScopeV1(),
        "祁雾",
    )


def _reverse_action(source: _FrozenDocument, *, line: int = 2) -> CharacterSignal:
    evidence = source.document.content.splitlines()[line - 1]
    return CharacterSignal(
        id="cs_0123456789abcdef0123456789abcdef",
        character="祁雾",
        dimension="core_personality",
        trait_key="directness",
        statement="祁雾用奉承话术迂回交流",
        polarity="negative",
        stability="temporary",
        observation_kind="action",
        source_kind="draft",
        evidence=EvidenceSpan(
            document_id=source.document.id,
            document_name=source.document.name,
            line_start=line,
            line_end=line,
            text=evidence,
        ),
    )


def _result(*signals: CharacterSignal) -> CharacterSignalExtractionResult:
    return CharacterSignalExtractionResult(
        signals=signals,
        draft_observations=signals,
        diagnostics=CharacterSignalDiagnostics(
            outcome="completed",
            attempted_calls=1,
            raw_records=len(signals),
            accepted_records=len(signals),
            rejected_records=0,
            prompt_tokens=3,
            completion_tokens=2,
            charged_tokens=5,
        ),
    )


def _partial_result(
    *provisional_clues: CharacterSignal,
    accepted_signals: tuple[CharacterSignal, ...] = (),
) -> CharacterSignalExtractionResult:
    return CharacterSignalExtractionResult(
        signals=accepted_signals,
        draft_observations=accepted_signals,
        provisional_draft_clues=provisional_clues,
        diagnostics=CharacterSignalDiagnostics(
            outcome="partial",
            attempted_calls=2,
            raw_records=len(provisional_clues) + len(accepted_signals),
            accepted_records=len(accepted_signals),
            rejected_records=0,
            reason_counts={"draft_actor_review_incomplete": 1},
            prompt_tokens=5,
            completion_tokens=3,
            charged_tokens=8,
        ),
    )


class _FrozenStage(CharacterConsistencyStage):
    def __init__(self, *, source, baselines, settings):
        super().__init__(settings=settings, provider=object())
        self._source = source
        self._baselines = baselines

    def _bind_frozen_documents(self, _db, **_kwargs):
        return [self._source]

    def _load_confirmed_traits(self, _db, **_kwargs):
        return list(self._baselines)


@dataclass
class _ExtractionRecorder:
    primary_result: CharacterSignalExtractionResult
    targeted_results: list[CharacterSignalExtractionResult] = field(
        default_factory=list
    )
    primary_calls: list[dict[str, Any]] = field(default_factory=list)
    targeted_calls: list[dict[str, Any]] = field(default_factory=list)

    def extractor_type(self):
        recorder = self

        class RecordingExtractor:
            def __init__(self, _provider, *, settings):
                self.settings = settings

            def extract(self, chunk, **kwargs):
                recorder.primary_calls.append(
                    {"chunk": chunk, "settings": self.settings, **kwargs}
                )
                return recorder.primary_result

            def extract_targeted(self, chunk, targets, **kwargs):
                recorder.targeted_calls.append(
                    {"chunk": chunk, "targets": targets, **kwargs}
                )
                if not recorder.targeted_results:
                    raise AssertionError("unexpected targeted extraction")
                return recorder.targeted_results.pop(0)

        return RecordingExtractor


def _run(
    stage: CharacterConsistencyStage,
    source: _FrozenDocument,
    *,
    remaining_run_tokens: int = 20_000,
):
    return stage.run(
        object(),
        run_id="analysis-run-1",
        project_id="project-1",
        documents=[source.document],
        metadata=[],
        remaining_run_tokens=remaining_run_tokens,
    )


@pytest.mark.parametrize(
    ("actor_review_enabled", "expected_signal_allowance", "expected_reserve"),
    ((True, 6_000, 4_000), (False, 10_000, 0)),
)
def test_draft_actor_review_preserves_drift_reviewer_reserve_under_tight_budget(
    monkeypatch: pytest.MonkeyPatch,
    actor_review_enabled: bool,
    expected_signal_allowance: int,
    expected_reserve: int,
):
    source = _draft_source("祁雾沉默片刻。")
    admission = CharacterSignalExtractionResult(
        diagnostics=CharacterSignalDiagnostics(
            outcome="skipped",
            attempted_calls=0,
            raw_records=0,
            accepted_records=0,
            rejected_records=0,
            reason_counts={"token_budget_preflight": 1},
            token_admission=CharacterSignalTokenAdmission(
                phase="initial",
                estimated_tokens=10_001,
                available_tokens=expected_signal_allowance,
            ),
        )
    )
    recorder = _ExtractionRecorder(primary_result=admission)
    monkeypatch.setattr(
        stage_module, "CharacterSignalExtractor", recorder.extractor_type()
    )
    settings = _settings(
        character_consistency_stage_token_budget=10_000,
        character_signal_token_budget=10_000,
        character_drift_token_budget=4_000,
        character_signal_full_line_prompt_v2=True,
        character_draft_actor_review_v1=actor_review_enabled,
    )

    result = _run(
        _FrozenStage(source=source, baselines=[], settings=settings),
        source,
        remaining_run_tokens=10_000,
    )

    assert recorder.primary_calls[0][
        "settings"
    ].character_signal_token_budget == expected_signal_allowance
    assert result.diagnostics["token_admission_events"] == [
        {
            "stage_phase": "primary_extraction",
            "signal_phase": "initial",
            "chunk_ordinal": 1,
            "target_ordinal": None,
            "estimated_tokens": 10_001,
            "available_tokens": expected_signal_allowance,
            "stage_remaining_before": 10_000,
            "reviewer_reserve_tokens": expected_reserve,
            "model_calls_before_failure": 0,
        }
    ]


def test_partial_actor_clue_reaches_stage_and_api_projection(
    monkeypatch: pytest.MonkeyPatch,
):
    content = "祁雾直截了当地拒绝请求。"
    source = _draft_source(content)
    provisional = _reverse_action(source, line=1).model_copy(
        update={
            "id": "cs_11111111111111111111111111111111",
            "statement": "祁雾直截了当地拒绝请求",
            "polarity": "positive",
        }
    )
    recorder = _ExtractionRecorder(
        primary_result=_partial_result(provisional)
    )
    monkeypatch.setattr(
        stage_module, "CharacterSignalExtractor", recorder.extractor_type()
    )

    result = _run(
        _FrozenStage(
            source=source,
            baselines=[],
            settings=_settings(character_draft_actor_review_v1=True),
        ),
        source,
    )

    assert result.issues == result.review_clues == ()
    assert result.diagnostics["outcome"] == "partial"
    assert result.diagnostics["counts"]["draft_observation_count"] == 0
    assert result.diagnostics["counts"]["pending_candidate_count"] == 0
    assert len(result.provisional_draft_clues) == 1
    clue = result.provisional_draft_clues[0]
    assert (clue.evidence, clue.proposed_statement) == (
        content,
        "祁雾直截了当地拒绝请求",
    )

    projected = project_provisional_draft_clues(
        {
            "items": [clue.model_dump(mode="json")],
            "truncated": result.provisional_draft_clues_truncated,
        },
        [
            SimpleNamespace(
                id=source.input_id,
                document_id=source.document.id,
                document_name=source.document.name,
                document_version=source.document_version,
                content=source.document.content,
                content_sha256=source.content_sha256,
            )
        ],
    )
    assert projected == {
        "items": [
            {
                "id": clue.id,
                "document_id": source.document.id,
                "document_version": source.document_version,
                "document_name": source.document.name,
                "line_start": 1,
                "line_end": 1,
                "evidence": content,
                "character": "祁雾",
                "dimension": "core_personality",
                "proposed_statement": "祁雾直截了当地拒绝请求",
                "reason": "partial_model_package",
            }
        ],
        "truncated": False,
    }


def test_partial_actor_clue_survives_unrelated_accepted_signal(
    monkeypatch: pytest.MonkeyPatch,
):
    content = (
        "祁雾直截了当地拒绝请求。\n"
        "祁雾在窗边停下脚步。"
    )
    source = _draft_source(content)
    provisional = _reverse_action(source, line=1).model_copy(
        update={
            "id": "cs_33333333333333333333333333333333",
            "statement": "祁雾直截了当地拒绝请求",
            "polarity": "positive",
        }
    )
    unrelated = CharacterSignal(
        id="cs_44444444444444444444444444444444",
        character="祁雾",
        dimension="contextual_behavior",
        trait_key="pause_by_window",
        statement="祁雾在窗边停下脚步",
        polarity="negative",
        stability="temporary",
        observation_kind="action",
        source_kind="draft",
        evidence=EvidenceSpan(
            document_id=source.document.id,
            document_name=source.document.name,
            line_start=2,
            line_end=2,
            text=content.splitlines()[1],
        ),
    )
    recorder = _ExtractionRecorder(
        primary_result=_partial_result(
            provisional,
            accepted_signals=(unrelated,),
        )
    )
    monkeypatch.setattr(
        stage_module, "CharacterSignalExtractor", recorder.extractor_type()
    )

    result = _run(
        _FrozenStage(
            source=source,
            baselines=[],
            settings=_settings(character_draft_actor_review_v1=True),
        ),
        source,
    )

    assert result.issues == result.review_clues == ()
    assert result.diagnostics["counts"]["draft_observation_count"] == 1
    assert [clue.line_start for clue in result.provisional_draft_clues] == [1]


def test_flag_on_wires_one_frozen_identity_through_all_three_draft_paths(
    monkeypatch: pytest.MonkeyPatch,
):
    content = (
        "这是未进入当前分块的冻结前言。\n"
        "祁雾用奉承话术迂回交流。\n"
        "祁雾走进空会议室。\n"
        "她随后关上房门。"
    )
    source = _draft_source(content)
    chunk_content = "\n".join(content.splitlines()[1:])
    monkeypatch.setattr(
        stage_module,
        "chunk_document",
        lambda *_args, **_kwargs: [
            SimpleNamespace(
                document_id=source.document.id,
                document_name=source.document.name,
                content=chunk_content,
                global_line_start=2,
            )
        ],
    )
    actor_range_calls: list[dict[str, Any]] = []

    def actor_range(chunk, character, named_line, **kwargs):
        actor_range_calls.append(
            {
                "chunk": chunk,
                "character": character,
                "named_line": named_line,
                **kwargs,
            }
        )
        return (3, 4) if named_line == 3 else None

    monkeypatch.setattr(
        stage_module, "draft_actor_review_evidence_range", actor_range
    )
    recorder = _ExtractionRecorder(
        primary_result=_result(_reverse_action(source)),
        targeted_results=[_result(), _result()],
    )
    monkeypatch.setattr(
        stage_module, "CharacterSignalExtractor", recorder.extractor_type()
    )
    settings = _settings(
        character_signal_full_line_prompt_v2=True,
        character_draft_actor_review_v1=True,
    )

    result = _run(
        _FrozenStage(
            source=source,
            baselines=[_baseline_entry()],
            settings=settings,
        ),
        source,
    )

    assert len(recorder.primary_calls) == 1
    assert len(recorder.targeted_calls) == 2
    primary = recorder.primary_calls[0]
    identity = primary["source_identity"]
    assert isinstance(identity, ScopeReviewSourceIdentity)
    assert identity.model_dump() == {
        "run_input_id": source.input_id,
        "document_id": source.document.id,
        "document_version": source.document_version,
        "content_sha256": source.content_sha256,
    }
    assert primary["frozen_content"] == content
    assert primary["frozen_content"] != primary["chunk"].content
    assert all(
        call["source_identity"] is identity
        and call["frozen_content"] == content
        for call in recorder.targeted_calls
    )
    assert "candidate_evidence_ranges" not in recorder.targeted_calls[0]
    assert recorder.targeted_calls[1]["candidate_evidence_ranges"] == ((3, 4),)
    assert any(
        call["named_line"] == 3
        and call["source_identity"] is identity
        and call["frozen_content"] == content
        for call in actor_range_calls
    )

    assert result.issues == ()
    assert len(result.review_clues) == 1
    assert result.review_clues[0].metadata["final_outcome"] == (
        "needs_confirmation"
    )
    assert result.review_clues[0].metadata["review_reason"] == (
        "single_behavior_is_not_drift"
    )


def test_later_targeted_opposite_axis_signal_vetoes_partial_actor_clue(
    monkeypatch: pytest.MonkeyPatch,
):
    content = (
        "祁雾直截了当地拒绝请求。\n"
        "祁雾用奉承话术迂回交流。"
    )
    source = _draft_source(content)
    provisional = _reverse_action(source, line=1).model_copy(
        update={
            "id": "cs_22222222222222222222222222222222",
            "statement": "祁雾直截了当地拒绝请求",
            "polarity": "positive",
        }
    )
    accepted_conflict = _reverse_action(source, line=2)
    recorder = _ExtractionRecorder(
        primary_result=_partial_result(provisional),
        targeted_results=[_result(accepted_conflict), _result()],
    )
    monkeypatch.setattr(
        stage_module, "CharacterSignalExtractor", recorder.extractor_type()
    )

    result = _run(
        _FrozenStage(
            source=source,
            baselines=[_baseline_entry()],
            settings=_settings(character_draft_actor_review_v1=True),
        ),
        source,
    )

    assert recorder.targeted_calls
    assert recorder.targeted_calls[0]["chunk"].content == content
    assert result.provisional_draft_clues == ()
    assert result.issues == ()


def test_without_confirmed_baseline_draft_signal_never_enters_drift(
    monkeypatch: pytest.MonkeyPatch,
):
    source = _draft_source("前言。\n祁雾用奉承话术迂回交流。")
    recorder = _ExtractionRecorder(primary_result=_result(_reverse_action(source)))
    monkeypatch.setattr(
        stage_module, "CharacterSignalExtractor", recorder.extractor_type()
    )

    def unexpected_drift(_case):
        raise AssertionError("drift preparation requires a confirmed baseline")

    monkeypatch.setattr(stage_module, "prepare_character_drift", unexpected_drift)
    result = _run(
        _FrozenStage(
            source=source,
            baselines=[],
            settings=_settings(
                character_signal_full_line_prompt_v2=True,
                character_draft_actor_review_v1=True,
            ),
        ),
        source,
    )

    assert result.issues == ()
    assert result.review_clues == ()
    assert result.diagnostics["reason_counts"][
        "no_confirmed_character_traits"
    ] == 1


def test_flag_off_keeps_legacy_draft_extraction_call_shape(
    monkeypatch: pytest.MonkeyPatch,
):
    source = _draft_source("祁雾沉默片刻。")
    recorder = _ExtractionRecorder(
        primary_result=_result(), targeted_results=[_result(), _result()]
    )
    monkeypatch.setattr(
        stage_module, "CharacterSignalExtractor", recorder.extractor_type()
    )

    def unexpected_actor_range(*_args, **_kwargs):
        raise AssertionError("draft actor range expansion must remain flag-gated")

    monkeypatch.setattr(
        stage_module,
        "draft_actor_review_evidence_range",
        unexpected_actor_range,
    )
    _run(
        _FrozenStage(
            source=source,
            baselines=[_baseline_entry()],
            settings=_settings(character_draft_actor_review_v1=False),
        ),
        source,
    )

    assert recorder.primary_calls[0]["source_identity"] is None
    assert recorder.primary_calls[0]["frozen_content"] is None
    assert len(recorder.targeted_calls) == 2
    assert set(recorder.targeted_calls[0]) == {"chunk", "targets"}
    assert set(recorder.targeted_calls[1]) == {
        "chunk",
        "targets",
        "candidate_evidence_ranges",
    }
    assert recorder.targeted_calls[1]["candidate_evidence_ranges"] == ((1, 1),)
