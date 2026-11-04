from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.character_consistency_stage import (
    CharacterConsistencyStage, _assess_observation_match,
    _observation_matches_baseline, _ObservationJournal,
)
from app.character_observations import CharacterObservationRecorder, project_character_observations
from app.character_trait_extraction import CharacterSignal
from app.db import (
    AnalysisRunRow, AnalysisRunInputRow, CharacterObservationRecordingRow,
    CharacterTraitCandidateRow, AnalysisRunCharacterTraitInputRow, SessionLocal,
)
from app.domain import EvidenceSpan
from app.main import app, write_limiter
from app.narrative_context import NarrativeScopeV1
from app.provider import OpenAICompatibleProvider
from app.service import _load_verified_snapshot, execute_analysis
from test_character_consistency_stage import (
    QueueProvider, _confirmed_trait, _new_run, _run_stage, _confirm_only_candidate, _confirm_all_candidates,
    _context, _create_document, _record, _response, _settings,
    _single_event_response, _snapshot_stub,
)


@pytest.fixture(autouse=True)
def _reset_write_limiter():
    write_limiter.events.clear()
    yield
    write_limiter.events.clear()


def _new_review(client, project_id, document_id):
    with patch("app.main.dispatch_analysis"):
        response = client.post(
            f"/api/v1/projects/{project_id}/analysis-runs",
            json={"mode": "draft_review", "target_document_ids": [document_id]},
        )
    assert response.status_code == 202, response.text
    return response.json()["id"]


def _run_recorded(run_id, provider, **overrides):
    with SessionLocal() as db:
        run = db.get(AnalysisRunRow, run_id)
        run.status = "running"
        documents, metadata = _load_verified_snapshot(db, run_id)
        result = CharacterConsistencyStage(
            settings=_settings(**overrides), provider=provider,
            observation_recorder=CharacterObservationRecorder(),
        ).run(
            db, run_id=run_id, project_id=run.project_id,
            documents=documents, metadata=metadata, remaining_run_tokens=60_000,
        )
        run.status = "completed"
        db.commit()
        return result, project_character_observations(db, run)


def _preference_project(client):
    project = client.post("/api/v1/projects", json={"name": "recording-baseline-" + uuid4().hex}).json()
    line = "林澈喜欢蜜瓜。"
    _create_document(client, project["id"], name="profile.md", role="character_profile", content=line,
                     narrative_context=_context(publication="published"))
    seed = _new_run(client, project["id"])
    _run_stage(seed, QueueProvider(_response(_record(evidence=line, polarity="positive", kind="explicit_declaration", trait_key="melon_preference"))))
    _confirm_only_candidate(client, project["id"], seed)
    return project


@pytest.mark.parametrize(
    ("dimension", "object_name", "frozen_key", "expected", "reason"),
    [
        ("preference", "蜜瓜", "preference:蜜瓜", True, "matched"),
        ("preference", "葡萄", "preference:蜜瓜", False, "comparison_mismatch"),
        ("preference", "蜜瓜", None, None, "comparison_identity_unavailable"),
        ("preference", "", "preference:蜜瓜", False, "object_identity_unavailable"),
        ("core_personality", "", "preference:蜜瓜", False, "dimension_mismatch"),
    ],
)
def test_assessment_preserves_legacy_tri_state(dimension, object_name, frozen_key, expected, reason):
    entry = (_snapshot_stub("frozen", frozen_key), _confirmed_trait(), NarrativeScopeV1(), "林澈")
    signal = CharacterSignal(
        id="cs_" + "a" * 32, character="林澈", dimension=dimension,
        trait_key="raw_unmodified_label", key_object=object_name,
        statement="林澈拒绝了眼前的点心", polarity="negative",
        stability="temporary", observation_kind="action", source_kind="draft",
        evidence=EvidenceSpan(document_id="draft", document_name="draft.md", line_start=1, line_end=1, text="林澈拒绝了眼前的点心。"),
    )
    assessment = _assess_observation_match(entry, signal)
    assert assessment.result is expected
    assert _observation_matches_baseline(entry, signal) is expected
    assert assessment.reason_code == reason


def test_v3_action_certificate_path_and_object_basis_survive_snapshot_capture():
    from test_character_target_bound_draft_review_v3 import (
        _review_target, _certified_signal, _confirmed_baseline,
    )

    target = _review_target()
    signal = _certified_signal(target, key_object="栗子糕").model_copy(update={"observation_kind": "action"})
    entry = (SimpleNamespace(id=str(uuid4()), payload={"comparison_key": target.comparison_key}),
             _confirmed_baseline(), NarrativeScopeV1(), target.character)
    assessment = _assess_observation_match(entry, signal)
    assert assessment.result is True
    assert assessment.match_method == "target_bound"
    assert _observation_matches_baseline(entry, signal) is True
    journal = _ObservationJournal()
    journal.sources[signal.evidence.document_id] = SimpleNamespace(input_id=str(uuid4()))
    journal.baselines[entry[0].id] = entry
    journal.capture(signal)
    journal.resolve(signal, (target.character,))
    journal.decide(signal, entry, "matched", status="matched", assessment=assessment)
    snapshot = journal.snapshots()[0]
    assert snapshot.target_bound_object_basis_id == "L1:A1"
    assert snapshot.model_dump(mode="json")["target_bound_object_basis_id"] == "L1:A1"
    assert snapshot.target_bound_target == target
    assert snapshot.observation_kind == "action"
    assert snapshot.key_object == "栗子糕"


def test_admitted_unresolved_actor_and_raw_identity_are_persisted_before_matching_rewrites():
    known = "林澈讨厌蜜瓜。"
    unknown = "苏弦喜欢蜜瓜。"
    with TestClient(app) as client:
        project = _preference_project(client)
        document = _create_document(
            client, project["id"], name="private-draft.md", role="chapter",
            content=known + "\n" + unknown, narrative_context=_context(publication="draft"),
        )
        run_id = _new_review(client, project["id"], document["id"])
        with SessionLocal() as db:
            frozen_trait = db.scalar(select(AnalysisRunCharacterTraitInputRow).where(AnalysisRunCharacterTraitInputRow.run_id == run_id))
            # Replay must retain this run's identity after the current trait changes.
            current = db.get(CharacterTraitCandidateRow, frozen_trait.candidate_id)
            current.trait_key = "later_live_identity"
            db.commit()
        records = [
            _record(character=name, dimension="preference", trait_key="raw_unmodified_label", evidence=line,
                    polarity="negative" if index == 1 else "positive", kind="preference_expression", line=index,
                    statement=line.rstrip("。"))
            for index, (name, line) in enumerate((("林澈", known), ("苏弦", unknown)), 1)
        ]
        provider = QueueProvider(_response(), _response(*records), _single_event_response("needs_confirmation"))
        result, saved = _run_recorded(run_id, provider)
        assert saved["recording"]["status"] == "complete", (result.diagnostics.get("reason_counts"), result.diagnostics.get("counts"))
        by_actor = {row["character"]: row for row in saved["items"]}
        assert set(by_actor) == {"林澈", "苏弦"}
        matched = by_actor["林澈"]
        assert matched["trait_key"] == "raw_unmodified_label"
        assert matched["comparison_key"] == "preference:蜜瓜"
        assert matched["evidence"]["text"] == known
        assert matched["matches"][0]["status"] == "matched"
        assert matched["matches"][0]["baseline"]["payload"]["trait_key"] == "melon_preference"
        assert matched["matches"][0]["identity_match"] is True
        assert by_actor["苏弦"]["resolved_character_key"] is None
        assert by_actor["苏弦"]["matches"][0]["reason_code"] == "unmatched_character_alias"
        with SessionLocal() as db:
            source = db.scalar(select(AnalysisRunInputRow).where(AnalysisRunInputRow.run_id == run_id, AnalysisRunInputRow.document_id == document["id"]))
            assert matched["source"]["run_input_id"] == source.id
        # Content lives only in the dedicated private projection.
        assert known not in json.dumps(result.diagnostics, ensure_ascii=False)


def test_real_worker_injects_recorder_through_accounting_provider_and_frozen_source():
    line = "周尧喜欢蜜瓜。"
    calls = []
    settings = _settings(enable_model_extraction=False, enable_review_agent=False)

    def handle(request):
        calls.append(json.loads(request.content))
        record = _record(character="周尧", dimension="preference", trait_key="raw_unmodified_label",
                         evidence=line, polarity="positive", kind="preference_expression")
        return httpx.Response(200, json={
            "choices": [{"message": {"content": _response(record)}}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 7},
        })

    def provider_factory(provider_settings):
        return OpenAICompatibleProvider(provider_settings, transport=httpx.MockTransport(handle))

    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "recording-worker-" + uuid4().hex}).json()
        document = _create_document(client, project["id"], name="draft.md", role="chapter", content=line,
                                    narrative_context=_context(publication="draft"))
        run_id = _new_review(client, project["id"], document["id"])
        runtime = SimpleNamespace(settings=settings, identity="server_default", available=True, unavailable_reason=None)
        with patch("app.service.runtime_for_analysis_run", return_value=runtime), patch("app.service.OpenAICompatibleProvider", side_effect=provider_factory):
            execute_analysis(run_id, raise_on_failure=True)
        assert calls
        with SessionLocal() as db:
            run = db.get(AnalysisRunRow, run_id)
            assert run.status == "completed"
            projection = project_character_observations(db, run)
            assert projection["recording"]["status"] == "complete", client.get(f"/api/v1/analysis-runs/{run_id}/diagnostics").json()["character_consistency"].get("reason_counts")
            assert len(projection["items"]) == 1
            assert projection["items"][0]["evidence"]["text"] == line
            assert projection["items"][0]["matches"][0]["reason_code"] == "unmatched_character_alias"
        diagnostics = client.get(f"/api/v1/analysis-runs/{run_id}/diagnostics").json()
        assert diagnostics["character_consistency"]["observation_recording"]["status"] == "complete"
        assert diagnostics["character_consistency"]["usage"]["attempted_calls"] >= 1


def test_stage_exception_retains_admitted_observation_and_failed_header():
    line = "周尧喜欢蜜瓜。"
    with TestClient(app) as client:
        project = _preference_project(client)
        document = _create_document(client, project["id"], name="draft.md", role="chapter", content=line,
                                    narrative_context=_context(publication="draft"))
        run_id = _new_review(client, project["id"], document["id"])
        record = _record(character="周尧", dimension="preference", trait_key="raw_unmodified_label", evidence=line,
                         polarity="positive", kind="preference_expression")
        with SessionLocal() as db:
            run = db.get(AnalysisRunRow, run_id)
            run.status = "running"
            documents, metadata = _load_verified_snapshot(db, run_id)
            with patch("app.character_consistency_stage.build_pending_trait_candidates", side_effect=RuntimeError("private defect")):
                with pytest.raises(RuntimeError, match="private defect"):
                    CharacterConsistencyStage(settings=_settings(), provider=QueueProvider(_response(), _response(record)),
                        observation_recorder=CharacterObservationRecorder()).run(
                        db, run_id=run_id, project_id=run.project_id, documents=documents, metadata=metadata,
                        remaining_run_tokens=60_000,
                    )
            db.commit()
            saved = project_character_observations(db, run)
            assert saved["recording"]["status"] == "failed"
            assert len(saved["items"]) == 1
            assert saved["items"][0]["evidence"]["text"] == line
            assert saved["items"][0]["matches"][0]["reason_code"] == "baseline_not_evaluated"


def test_worker_writer_failure_is_explicit_and_never_complete():
    line = "普通章节正文。"
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "writer-failure-" + uuid4().hex}).json()
        document = _create_document(client, project["id"], name="draft.md", role="chapter", content=line,
                                    narrative_context=_context(publication="draft"))
        run_id = _new_review(client, project["id"], document["id"])
        with patch("app.service.CharacterObservationRecorder.record", side_effect=RuntimeError("private database defect")):
            execute_analysis(run_id, raise_on_failure=True)
        with SessionLocal() as db:
            assert db.get(CharacterObservationRecordingRow, run_id).status == "failed"
        diagnostics = client.get(f"/api/v1/analysis-runs/{run_id}/diagnostics").json()
        recording = diagnostics["character_consistency"]["observation_recording"]
        assert recording == {"status": "failed", "reason_code": "observation_recording_failed"}
        assert "private database defect" not in json.dumps(diagnostics)


def test_observation_limit_preserves_identity_match_but_does_not_claim_comparison_selection():
    line = "林澈讨厌蜜瓜。"
    with TestClient(app) as client:
        project = _preference_project(client)
        document = _create_document(client, project["id"], name="draft.md", role="chapter", content="\n".join([line] * 25),
                                    narrative_context=_context(publication="draft"))
        run_id = _new_review(client, project["id"], document["id"])
        records = [_record(evidence=line, polarity="negative", kind="preference_expression", line=index) for index in range(1, 26)]
        _, saved = _run_recorded(run_id, QueueProvider(_response(), _response(*records), _single_event_response("needs_confirmation")))
        assert saved["recording"]["status"] == "partial"
        assert len(saved["items"]) == 25
        decisions = [row["matches"][0] for row in saved["items"]]
        assert sum(row["status"] == "matched" for row in decisions) == 24
        omitted = [row for row in decisions if row["reason_code"] == "observation_limit"]
        assert len(omitted) == 1
        assert omitted[0]["identity_match"] is True
        assert omitted[0]["status"] == "not_matched"


def test_baseline_limit_records_bounded_pairs_and_explicit_omitted_summary():
    profile = "林澈喜欢蜜瓜，也喜欢葡萄。"
    melon = _record(evidence=profile, polarity="positive", kind="explicit_declaration", trait_key="food_preference", statement="林澈喜欢蜜瓜")
    grape = {**melon, "statement": "林澈喜欢葡萄", "key_object": "葡萄"}
    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "baseline-limit-" + uuid4().hex}).json()
        _create_document(client, project["id"], name="profile.md", role="character_profile", content=profile,
                         narrative_context=_context(publication="published"))
        seed = _new_run(client, project["id"])
        _run_stage(seed, QueueProvider(_response(melon, grape)))
        assert len(_confirm_all_candidates(client, project["id"], seed)) == 2
        line = "林澈讨厌蜜瓜。"
        document = _create_document(client, project["id"], name="draft.md", role="chapter", content=line,
                                    narrative_context=_context(publication="draft"))
        run_id = _new_review(client, project["id"], document["id"])
        record = _record(evidence=line, polarity="negative", kind="preference_expression")
        result, saved = _run_recorded(run_id, QueueProvider(_response(), _response(record), *[_response()] * 6),
                                      character_consistency_max_candidates_per_run=1)
        summary = result.diagnostics["target_assessment_summary_v1"]
        assert summary["total_targets"] == 2 and summary["unprocessed"] == 1
        assert saved["recording"]["status"] == "partial"
        assert saved["recording"]["coverage"]["omitted_baseline_count"] == 1
        assert saved["recording"]["coverage"]["excluded_baseline_counts"] == {"baseline_limit": 1}
        assert len(saved["items"]) == 1
        assert len(saved["items"][0]["matches"]) == 1
