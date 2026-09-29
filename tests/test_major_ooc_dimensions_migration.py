"""Migration coverage for the two major object-anchored OOC dimensions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import MetaData, Table, create_engine, inspect, select
from sqlalchemy.exc import IntegrityError


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = "0022_character_formal_proof"
HEAD = "0023_major_ooc_dimensions"
LOCAL_ID = "00000000-0000-0000-0000-000000000001"
WHEN = datetime(2026, 9, 28)


def _config(url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["database_url"] = url
    return config


def _candidate(
    candidate_id: str,
    trait_type: str,
    comparison_key: str | None,
    fingerprint: str,
    *,
    key_object: str | None = None,
) -> dict:
    values = {
        "id": candidate_id,
        "project_id": "project-ooc-dimensions",
        "source_run_id": "run-ooc-dimensions",
        "character_key": "character-lin",
        "character_display_name": "Lin",
        "trait_type": trait_type,
        "trait_key": candidate_id,
        "comparison_key": comparison_key,
        "value": candidate_id,
        "polarity": "neutral",
        "stability": "stable",
        "contexts": [],
        "origin": "explicit_setting",
        "authority_tier": "formal_record",
        "confidence": 0.9,
        "scope_payload": {},
        "scope_sha256": "a" * 64,
        "evidence": [],
        "evidence_sha256": "b" * 64,
        "support_binding_mode": "legacy_v1",
        "candidate_fingerprint": fingerprint,
        "generator_version": "migration-test-v1",
        "provenance": {},
        "review_state": "pending",
        "lock_version": 0,
        "created_at": WHEN,
    }
    if key_object is not None:
        values["key_object"] = key_object
    return values


def _candidate_triggers(connection) -> dict[str, str]:
    return dict(connection.exec_driver_sql(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type = 'trigger' AND tbl_name = 'character_trait_candidates' "
        "ORDER BY name"
    ).all())


def _assert_sqlite_integrity(engine) -> None:
    with engine.connect() as connection:
        connection.exec_driver_sql("PRAGMA foreign_keys=ON")
        assert connection.exec_driver_sql("PRAGMA foreign_key_check").all() == []
        assert connection.exec_driver_sql("PRAGMA quick_check").scalar_one() == "ok"


def test_0023_expands_object_anchored_dimensions_and_downgrades_fail_closed(tmp_path):
    url = f"sqlite:///{(tmp_path / 'major-ooc-dimensions.db').as_posix()}"
    config = _config(url)
    command.upgrade(config, PREVIOUS)
    engine = create_engine(url)
    metadata = MetaData()
    projects = Table("projects", metadata, autoload_with=engine)
    runs = Table("analysis_runs", metadata, autoload_with=engine)
    candidates = Table("character_trait_candidates", metadata, autoload_with=engine)

    legacy_values = _candidate(
        "candidate-legacy", "core_personality", None, "0" * 64
    )
    with engine.begin() as connection:
        connection.execute(projects.insert().values(
            id="project-ooc-dimensions",
            workspace_id=LOCAL_ID,
            name="OOC dimensions",
            description="",
            created_at=WHEN,
        ))
        connection.execute(runs.insert().values(
            id="run-ooc-dimensions",
            project_id="project-ooc-dimensions",
            status="completed",
            created_at=WHEN,
            completed_at=WHEN,
            input_chars=20,
            prompt_tokens=0,
            completion_tokens=0,
            estimated_cost_usd=0.0,
            cancel_requested=False,
            batch_mode="full_review",
            sensitivity="balanced",
            batch_coverage={},
        ))
        connection.execute(candidates.insert().values(**legacy_values))
        legacy_before = connection.execute(select(
            candidates.c.id,
            candidates.c.trait_type,
            candidates.c.comparison_key,
            candidates.c.candidate_fingerprint,
        )).one()
        triggers_before = _candidate_triggers(connection)
    assert triggers_before
    engine.dispose()

    command.upgrade(config, HEAD)
    engine = create_engine(url)
    candidates = Table(
        "character_trait_candidates", MetaData(), autoload_with=engine
    )
    columns = {
        item["name"]: item
        for item in inspect(engine).get_columns("character_trait_candidates")
    }
    assert columns["key_object"]["nullable"] is True
    assert columns["key_object"]["type"].length == 80
    checks = {
        item["name"]: str(item.get("sqltext") or "")
        for item in inspect(engine).get_check_constraints(
            "character_trait_candidates"
        )
    }
    assert "relationship_attitude" in checks["ck_character_trait_candidate_type"]
    assert "motivation_goal" in checks["ck_character_trait_candidate_type"]
    assert "ck_character_trait_candidate_object_identity" in checks
    assert "key_object" in checks["ck_character_trait_candidate_object_identity"]
    assert "160" in checks["ck_character_trait_candidate_object_identity"]

    relationship_key = f"relationship_attitude:{'1' * 64}.character-lin"
    motivation_key = f"motivation_goal:{'2' * 64}.survival"
    with engine.begin() as connection:
        assert connection.execute(select(
            candidates.c.id,
            candidates.c.trait_type,
            candidates.c.comparison_key,
            candidates.c.candidate_fingerprint,
        ).where(candidates.c.id == "candidate-legacy")).one() == legacy_before
        assert connection.execute(select(
            candidates.c.key_object
        ).where(candidates.c.id == "candidate-legacy")).one()[0] is None
        assert _candidate_triggers(connection) == triggers_before
        connection.execute(candidates.insert().values(**_candidate(
            "candidate-relationship",
            "relationship_attitude",
            relationship_key,
            "1" * 64,
            key_object="character-lin",
        )))
        connection.execute(candidates.insert().values(**_candidate(
            "candidate-motivation",
            "motivation_goal",
            motivation_key,
            "2" * 64,
            key_object="survival",
        )))

    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(candidates.insert().values(**_candidate(
            "candidate-missing-object",
            "relationship_attitude",
            relationship_key,
            "3" * 64,
        )))
    malformed = "x" * len(motivation_key)
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(candidates.insert().values(**_candidate(
            "candidate-malformed-key",
            "motivation_goal",
            malformed,
            "4" * 64,
            key_object="survival",
        )))
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(candidates.insert().values(**_candidate(
            "candidate-long-key",
            "relationship_attitude",
            f"relationship_attitude:{'x' * 160}",
            "5" * 64,
            key_object="character-lin",
        )))
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(candidates.insert().values(**_candidate(
            "candidate-blank-object",
            "motivation_goal",
            motivation_key,
            "6" * 64,
            key_object="   ",
        )))
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.exec_driver_sql(
            "UPDATE character_trait_candidates SET approved_axis_version = 1 "
            "WHERE id = 'candidate-legacy'"
        )
    _assert_sqlite_integrity(engine)
    engine.dispose()

    with pytest.raises(RuntimeError, match="cannot downgrade 0023"):
        command.downgrade(config, PREVIOUS)

    engine = create_engine(url)
    candidates = Table(
        "character_trait_candidates", MetaData(), autoload_with=engine
    )
    with engine.begin() as connection:
        assert connection.exec_driver_sql(
            "SELECT version_num FROM alembic_version"
        ).scalar_one() == HEAD
        connection.execute(candidates.delete().where(
            candidates.c.trait_type.in_((
                "relationship_attitude", "motivation_goal"
            ))
        ))
    engine.dispose()

    command.downgrade(config, PREVIOUS)
    engine = create_engine(url)
    checks = {
        item["name"]: str(item.get("sqltext") or "")
        for item in inspect(engine).get_check_constraints(
            "character_trait_candidates"
        )
    }
    assert "ck_character_trait_candidate_object_identity" not in checks
    assert "relationship_attitude" not in checks["ck_character_trait_candidate_type"]
    assert "motivation_goal" not in checks["ck_character_trait_candidate_type"]
    assert "key_object" not in {
        item["name"]
        for item in inspect(engine).get_columns("character_trait_candidates")
    }
    candidates = Table(
        "character_trait_candidates", MetaData(), autoload_with=engine
    )
    with engine.connect() as connection:
        assert connection.execute(select(
            candidates.c.id,
            candidates.c.trait_type,
            candidates.c.comparison_key,
            candidates.c.candidate_fingerprint,
        )).one() == legacy_before
        assert _candidate_triggers(connection) == triggers_before
    _assert_sqlite_integrity(engine)
    engine.dispose()
