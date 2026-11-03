from __future__ import annotations

import hashlib
import os
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, delete, inspect, select
from sqlalchemy.engine import make_url
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session
from sqlalchemy.schema import CreateTable

from app.character_observations import (
    CharacterObservationIntegrityError,
    CharacterObservationRecorder,
    ObservationMatchDecision,
    ObservationRecordingCoverage,
    ValidatedCharacterObservation,
    project_character_observations,
)
from app.db import (
    AnalysisRunCharacterTraitInputRow, AnalysisRunInputRow, AnalysisRunRow,
    Base, CharacterObservationRecordingRow, CharacterObservationRow,
    CharacterTraitCandidateRow, CharacterTraitReviewRow, DocumentRow,
    LOCAL_WORKSPACE_ID, ProjectRow, SessionLocal, WorkspaceRow,
    enable_sqlite_foreign_keys,
)
from app.domain import EvidenceSpan
from app.narrative_context import payload_sha256


ROOT = Path(__file__).resolve().parents[1]
BODY = '林舟看着餐盘，说：“我不吃。”\n他把餐盘推开。'


def seed(db, *, workspace_id=LOCAL_WORKSPACE_ID, prefix=""):
    project = ProjectRow(id=prefix + "project", workspace_id=workspace_id, name="Private story")
    db.add(project)
    db.flush()
    run = AnalysisRunRow(id=prefix + "run", project_id=project.id, status="running", batch_mode="draft_review")
    db.add(run)
    db.flush()
    source = AnalysisRunInputRow(id=prefix + "input", run_id=run.id, document_id=prefix + "document", document_name="draft.md", document_version=3, content=BODY, content_sha256=hashlib.sha256(BODY.encode()).hexdigest(), ordinal=0)
    db.add(source)
    db.flush()
    return project, run, source


def observation(source, number=1, **updates):
    values = dict(
        observation_id="cs_" + f"{number:032x}", source_run_input_id=source.id,
        character="林舟", resolved_character_key="林舟", dimension="preference",
        trait_key="进食选择", key_object="餐食", comparison_key="preference:餐食",
        statement="林舟拒绝进食", polarity="negative", stability="situational",
        observation_kind="action", context="住院期间",
        evidence=EvidenceSpan(document_id=source.document_id, document_name=source.document_name, line_start=1, line_end=1, text="我不吃。"),
        matches=(ObservationMatchDecision(status="unresolved", reason_code="no_confirmed_baseline"),),
    )
    values.update(updates)
    return ValidatedCharacterObservation(**values)


def baseline(db, project, run, *, prefix=""):
    origin_run = AnalysisRunRow(id=prefix + "origin-run", project_id=project.id, status="completed")
    db.add(origin_run)
    db.flush()
    candidate = CharacterTraitCandidateRow(id=prefix + "candidate", project_id=project.id, source_run_id=origin_run.id, character_key="林舟", character_display_name="林舟", trait_type="preference", trait_key="进食选择", value="林舟通常正常进食", polarity="positive", stability="stable", origin="explicit_setting", confidence=.9, scope_payload={}, scope_sha256=payload_sha256({}), evidence=[], evidence_sha256=payload_sha256([]), support_binding_mode="legacy_v1", candidate_fingerprint=hashlib.sha256((prefix + "candidate").encode()).hexdigest(), generator_version="test", review_state="confirmed", lock_version=1)
    db.add(candidate)
    db.flush()
    review = CharacterTraitReviewRow(id=prefix + "review", project_id=project.id, candidate_id=candidate.id, decision="confirm", expected_lock_version=0)
    db.add(review)
    db.flush()
    payload = {"schema_version": 1, "candidate_id": candidate.id, "character_display_name": "林舟", "trait_type": "preference", "trait_key": "进食选择", "value": "林舟通常正常进食", "polarity": "positive", "evidence": []}
    frozen = AnalysisRunCharacterTraitInputRow(id=prefix + "baseline", run_id=run.id, project_id=project.id, candidate_id=candidate.id, confirmation_review_id=review.id, candidate_lock_version=1, ordinal=0, payload=payload, payload_sha256=payload_sha256(payload))
    db.add(frozen)
    db.flush()
    return frozen, candidate


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'observations.db').as_posix()}")
    enable_sqlite_foreign_keys(engine)
    Base.metadata.create_all(engine)
    with Session(engine, expire_on_commit=False) as session:
        yield session
    engine.dispose()


def test_records_all_admitted_observations_with_frozen_decisions_and_idempotency(db):
    project, run, source = seed(db)
    frozen, candidate = baseline(db, project, run)
    recorder = CharacterObservationRecorder()
    matched = observation(source, matches=(ObservationMatchDecision(baseline_input_id=frozen.id, status="matched", reason_code="matched", identity_match=True, match_method="comparison_identity"),))
    unresolved = observation(source, 2, character="小舟", resolved_character_key=None, matches=(ObservationMatchDecision(baseline_input_id=frozen.id, status="unresolved", reason_code="unmatched_character_alias"),))
    not_matched = observation(source, 3, matches=(ObservationMatchDecision(baseline_input_id=frozen.id, status="not_matched", reason_code="comparison_mismatch", identity_match=False),))
    recorder.record(db, run_id=run.id, project_id=project.id, observations=[unresolved, matched, not_matched])
    recorder.finish(db, run_id=run.id, project_id=project.id, status="complete")
    db.commit()
    before = project_character_observations(db, run)
    recorder.record(db, run_id=run.id, project_id=project.id, observations=[matched, unresolved, not_matched])
    recorder.finish(db, run_id=run.id, project_id=project.id, status="complete")
    candidate.value = "后来改过的设定"
    candidate.review_state = "withdrawn"
    # A later document revision cannot rewrite the run's frozen evidence.
    db.add(DocumentRow(id=source.document_id, project_id=project.id, name="draft.md", content="后来改过的正文", version=4))
    db.commit()
    after = project_character_observations(db, run)
    assert before == after
    assert after["recording"]["observation_count"] == 3
    assert [item["observation_id"] for item in after["items"]] == [matched.observation_id, unresolved.observation_id, not_matched.observation_id]
    assert after["items"][0]["statement"] == "林舟拒绝进食"
    assert after["items"][0]["evidence"]["text"] == "我不吃。"
    assert after["items"][0]["matches"][0]["baseline"]["payload"]["value"] == "林舟通常正常进食"
    assert after["items"][0]["source"]["document_version"] == 3
    stored = db.get(CharacterObservationRow, (run.id, matched.observation_id))
    assert "payload" not in stored.payload["matches"][0]["baseline"]
    assert "not a consistency verdict" in after["match_semantics"]
    with pytest.raises(CharacterObservationIntegrityError, match="immutable"):
        recorder.record(db, run_id=run.id, project_id=project.id, observations=[matched.model_copy(update={"statement": "新版本"})])
    with pytest.raises(CharacterObservationIntegrityError, match="sealed"):
        recorder.record(db, run_id=run.id, project_id=project.id, observations=[observation(source, 4)])


def test_source_run_project_and_frozen_baseline_bindings_are_rejected(db):
    project, run, source = seed(db)
    other_project, other_run, other_source = seed(db, prefix="other-")
    foreign_baseline, _ = baseline(db, other_project, other_run, prefix="other-")
    recorder = CharacterObservationRecorder()
    with pytest.raises(CharacterObservationIntegrityError, match="ownership"):
        recorder.begin(db, run_id=run.id, project_id=other_project.id)
    with pytest.raises(CharacterObservationIntegrityError, match="source binding"):
        recorder.record(db, run_id=run.id, project_id=project.id, observations=[observation(other_source)])
    with pytest.raises(CharacterObservationIntegrityError, match="baseline binding"):
        recorder.record(db, run_id=run.id, project_id=project.id, observations=[observation(source, matches=(ObservationMatchDecision(baseline_input_id=foreign_baseline.id, status="not_matched", reason_code="comparison_mismatch"),))])
    assert not list(db.scalars(select(CharacterObservationRow)))


@pytest.mark.parametrize("quote,start,end", [("", 1, 1), (" ", 1, 1), ("他把餐盘推开。", 1, 1), ("我不吃。", 2, 2), ("我不吃。", 0, 1), ("我不吃。", 1, 99)])
def test_evidence_requires_nonempty_exact_quote_within_bound_frozen_lines(db, quote, start, end):
    project, run, source = seed(db)
    value = observation(source, evidence=EvidenceSpan(document_id=source.document_id, document_name=source.document_name, line_start=start, line_end=end, text=quote))
    with pytest.raises(CharacterObservationIntegrityError, match="evidence"):
        CharacterObservationRecorder().record(db, run_id=run.id, project_id=project.id, observations=[value])


def test_pagination_and_coverage_are_explicit_and_stable(db):
    project, run, source = seed(db)
    assert project_character_observations(db, run)["recording"]["status"] == "legacy_unavailable"
    recorder = CharacterObservationRecorder()
    recorder.begin(db, run_id=run.id, project_id=project.id)
    assert project_character_observations(db, run)["recording"]["status"] == "in_progress"
    recorder.record(db, run_id=run.id, project_id=project.id, observations=[observation(source, n) for n in [5, 2, 4, 1, 3]])
    first = project_character_observations(db, run, offset=0, limit=2)
    second = project_character_observations(db, run, offset=2, limit=2)
    last = project_character_observations(db, run, offset=4, limit=2)
    assert [item["ordinal"] for item in first["items"] + second["items"] + last["items"]] == list(range(5))
    assert first["page"] == {"offset": 0, "limit": 2, "total_items": 5, "has_more": True}
    assert last["page"]["has_more"] is False
    assert project_character_observations(db, run, offset=20)["items"] == []
    for offset, limit in [(-1, 10), (0, 0), (0, 201)]:
        with pytest.raises(ValueError):
            project_character_observations(db, run, offset=offset, limit=limit)
    recorder.finish(db, run_id=run.id, project_id=project.id, status="partial")
    assert project_character_observations(db, run)["recording"]["status"] == "partial"


@pytest.mark.parametrize("status", ["complete", "failed", "not_applicable"])
def test_empty_recording_has_explicit_terminal_status(db, status):
    project, run, _ = seed(db)
    CharacterObservationRecorder().finish(db, run_id=run.id, project_id=project.id, status=status)
    data = project_character_observations(db, run)
    assert data["recording"]["status"] == status
    assert data["recording"]["observation_count"] == 0
    assert data["recording"]["source_manifest_sha256"]
    assert data["items"] == []


def test_frozen_integrity_and_manifest_changes_fail_closed(db):
    project, run, source = seed(db)
    recorder = CharacterObservationRecorder()
    recorder.begin(db, run_id=run.id, project_id=project.id)
    source.document_version = 9
    with pytest.raises(CharacterObservationIntegrityError, match="recording identity"):
        recorder.record(db, run_id=run.id, project_id=project.id, observations=[observation(source)])
    source.document_version = 3
    source.content = "tampered"
    with pytest.raises(CharacterObservationIntegrityError, match="source integrity"):
        recorder.begin(db, run_id=run.id, project_id=project.id)


def test_unexpanded_baselines_are_counted_as_partial_recording_without_copying_them(db):
    project, run, source = seed(db)
    recorder = CharacterObservationRecorder()
    recorder.record(db, run_id=run.id, project_id=project.id, observations=[observation(source)])
    coverage = ObservationRecordingCoverage(baseline_not_evaluated_count=4740, excluded_baseline_counts={"equivalent_axis_collapsed": 4})
    recorder.finish(db, run_id=run.id, project_id=project.id, status="complete", coverage=coverage)
    data = project_character_observations(db, run)
    assert data["recording"]["status"] == "partial"
    assert data["recording"]["coverage"] == {"baseline_not_evaluated_count": 4740, "excluded_baseline_counts": {"equivalent_axis_collapsed": 4}, "omitted_baseline_count": 4744}
    assert len(data["items"][0]["matches"]) == 1
    recorder.finish(db, run_id=run.id, project_id=project.id, status="complete", coverage=coverage)


def test_api_frozen_baseline_reference_integrity_is_checked(db):
    project, run, source = seed(db)
    frozen, _ = baseline(db, project, run)
    value = observation(source, matches=(ObservationMatchDecision(baseline_input_id=frozen.id, status="matched", reason_code="matched"),))
    CharacterObservationRecorder().record(db, run_id=run.id, project_id=project.id, observations=[value])
    frozen.payload = {**frozen.payload, "value": "corrupted frozen input"}
    with pytest.raises(CharacterObservationIntegrityError, match="baseline integrity"):
        project_character_observations(db, run)


@pytest.mark.parametrize("parent", ["run", "project"])
def test_deleting_parent_never_leaves_observation_or_header_orphans(db, parent):
    project, run, source = seed(db)
    CharacterObservationRecorder().record(db, run_id=run.id, project_id=project.id, observations=[observation(source)])
    db.commit()
    # Existing run-input FKs predate CASCADE. Parent deletion already requires
    # source cleanup; no explicit cleanup of either new table is performed.
    db.execute(delete(AnalysisRunInputRow).where(AnalysisRunInputRow.run_id == run.id))
    db.execute(delete(AnalysisRunRow).where(AnalysisRunRow.id == run.id))
    if parent == "project":
        db.execute(delete(ProjectRow).where(ProjectRow.id == project.id))
    db.commit()
    assert list(db.scalars(select(CharacterObservationRow))) == []
    assert list(db.scalars(select(CharacterObservationRecordingRow))) == []


def test_api_uses_existing_workspace_access_and_disallows_client_writes():
    from app.main import app
    from tests.test_workspace_isolation import register, required_auth_settings

    with required_auth_settings(), TestClient(app) as owner, TestClient(app) as other, TestClient(app) as anonymous:
        identity, _ = register(owner, "observation-owner")
        register(other, "observation-other")
        with SessionLocal() as session:
            project, run, source = seed(session, workspace_id=identity["workspace"]["id"], prefix=uuid4().hex[:8])
            CharacterObservationRecorder().record(session, run_id=run.id, project_id=project.id, observations=[observation(source)])
            session.commit()
            run_id = run.id
        url = f"/api/v1/analysis-runs/{run_id}/character-observations"
        response = owner.get(url)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["recording"]["observation_count"] == 1
        assert other.get(url).status_code == 404
        assert anonymous.get(url).status_code == 401
        assert owner.get(url + "?limit=201").status_code == 422
        assert owner.post(url, json={"items": []}).status_code in {403, 405}


def test_anonymous_local_api_and_baseline_only_history():
    from app.main import app
    with TestClient(app) as client:
        with SessionLocal() as session:
            project, run, _ = seed(session, prefix=uuid4().hex[:8])
            run.batch_mode = "baseline_build"
            session.commit()
            run_id = run.id
        response = client.get(f"/api/v1/analysis-runs/{run_id}/character-observations")
        assert response.status_code == 200
        assert response.json()["recording"]["status"] == "not_applicable"


@pytest.mark.parametrize("run_status,expected", [("completed", "partial"), ("failed", "failed"), ("cancelled", "failed")])
def test_terminal_run_with_unfinished_header_is_projected_without_mutation(run_status, expected):
    from app.main import app
    with TestClient(app) as client:
        with SessionLocal() as session:
            project, run, _ = seed(session, prefix=uuid4().hex[:8])
            CharacterObservationRecorder().begin(session, run_id=run.id, project_id=project.id)
            run.status = run_status
            session.commit()
            run_id = run.id
        response = client.get(f"/api/v1/analysis-runs/{run_id}/character-observations")
        assert response.status_code == 200
        recording = response.json()["recording"]
        assert recording["status"] == expected
        assert recording["stored_status"] == "in_progress"
        assert recording["incomplete_reason"] == "run_ended_before_recording_completed"
        with SessionLocal() as session:
            header = session.get(CharacterObservationRecordingRow, run_id)
            assert header.status == "in_progress"
            assert header.completed_at is None


def test_additive_upgrade_preserves_old_history_and_guards_downgrade(tmp_path):
    url = f"sqlite:///{(tmp_path / 'legacy.db').as_posix()}"
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["database_url"] = url
    command.upgrade(config, "0024_project_import_foundation")
    engine = create_engine(url)
    enable_sqlite_foreign_keys(engine)
    with Session(engine) as session:
        project, run, source = seed(session)
        session.commit()
        identifiers = project.id, run.id, source.id
    command.upgrade(config, "head")
    assert {"character_observations", "character_observation_recordings"} <= set(inspect(engine).get_table_names())
    with Session(engine) as session:
        project = session.get(ProjectRow, identifiers[0])
        run = session.get(AnalysisRunRow, identifiers[1])
        source = session.get(AnalysisRunInputRow, identifiers[2])
        assert source.content == BODY
        assert project_character_observations(session, run)["recording"]["status"] == "legacy_unavailable"
        CharacterObservationRecorder().record(session, run_id=run.id, project_id=project.id, observations=[observation(source)])
        session.commit()
    with pytest.raises(RuntimeError, match="history exists"):
        command.downgrade(config, "0024_project_import_foundation")
    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
        assert connection.exec_driver_sql("SELECT count(*) FROM character_observations").scalar_one() == 1
    engine.dispose()


def test_new_schema_compiles_on_postgresql():
    for row in [CharacterObservationRecordingRow, CharacterObservationRow]:
        ddl = str(CreateTable(row.__table__).compile(dialect=postgresql.dialect()))
        assert "ON DELETE CASCADE" in ddl
        assert "JSON" in ddl if row is CharacterObservationRow else "source_manifest_sha256" in ddl


@pytest.mark.skipif(not os.environ.get("LOREGUARD_TEST_POSTGRES_URL"), reason="requires disposable PostgreSQL test database")
def test_postgres_old_schema_upgrade_snapshot_roundtrip_and_parent_cleanup():
    # Follow the existing PG migration gate convention: an isolated disposable
    # schema in the explicitly configured test database, never product data.
    base_url = os.environ["LOREGUARD_TEST_POSTGRES_URL"]
    schema = "loreguard_observations_" + uuid4().hex[:12]
    admin = create_engine(base_url)
    engine = None
    try:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        url = make_url(base_url).update_query_dict({"options": f"-csearch_path={schema},public"}).render_as_string(hide_password=False)
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        config.attributes["database_url"] = url
        command.upgrade(config, "0024_project_import_foundation")
        engine = create_engine(url)
        with Session(engine, expire_on_commit=False) as session:
            project, run, source = seed(session)
            session.commit()
            identifiers = project.id, run.id, source.id
        command.upgrade(config, "head")
        with Session(engine, expire_on_commit=False) as session:
            project = session.get(ProjectRow, identifiers[0])
            run = session.get(AnalysisRunRow, identifiers[1])
            source = session.get(AnalysisRunInputRow, identifiers[2])
            assert source.content == BODY
            assert project_character_observations(session, run)["recording"]["status"] == "legacy_unavailable"
            recorder = CharacterObservationRecorder()
            value = observation(source)
            recorder.record(session, run_id=run.id, project_id=project.id, observations=[value])
            recorder.finish(session, run_id=run.id, project_id=project.id, status="complete")
            session.commit()
        with Session(engine, expire_on_commit=False) as session:
            run = session.get(AnalysisRunRow, identifiers[1])
            data = project_character_observations(session, run, limit=1)
            assert data["items"][0]["statement"] == value.statement
            assert data["items"][0]["source"]["document_version"] == 3
            assert data["recording"]["status"] == "complete"
            recorder.record(session, run_id=run.id, project_id=run.project_id, observations=[value])
            assert len(project_character_observations(session, run)["items"]) == 1
            session.execute(delete(AnalysisRunInputRow).where(AnalysisRunInputRow.run_id == run.id))
            session.execute(delete(AnalysisRunRow).where(AnalysisRunRow.id == run.id))
            session.commit()
            assert list(session.scalars(select(CharacterObservationRow))) == []
            assert list(session.scalars(select(CharacterObservationRecordingRow))) == []
    finally:
        if engine is not None:
            engine.dispose()
        with admin.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        admin.dispose()
