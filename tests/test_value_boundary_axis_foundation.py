"""Regression gates for adding author-scoped value and boundary axes."""

from __future__ import annotations

import hashlib
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import MetaData, Table, create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from app.character_drift import (
    CharacterDriftCase, ConfirmedTraitSnapshot, prepare_character_drift,
)
from app.character_trait_extraction import CharacterSignal
from app.domain import EvidenceSpan
from app.narrative_context import payload_sha256
from app.service import _approved_axis_snapshot_fields


ROOT = Path(__file__).resolve().parents[1]
LOCAL_ID = "00000000-0000-0000-0000-000000000001"
WHEN = datetime(2026, 9, 27)


def _migrate(url: str, revision: str, *, downgrade: bool = False) -> None:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["database_url"] = url
    if downgrade:
        command.downgrade(config, revision)
    else:
        command.upgrade(config, revision)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _seed_legacy_bound_trait(engine) -> None:
    metadata = MetaData()
    tables = {
        name: Table(name, metadata, autoload_with=engine)
        for name in (
            "projects", "analysis_runs", "character_trait_axes",
            "character_trait_candidates", "character_trait_reviews",
            "analysis_run_character_trait_inputs",
        )
    }
    scope = {"schema_version": 1, "timeline_key": "main"}
    evidence = [{"document_id": "document-old", "text": "角色拒绝冒用签名。"}]
    frozen = {"schema_version": 1, "candidate_id": "candidate-old", "value": "拒绝冒用签名"}
    with engine.begin() as connection:
        for project_id in ("project-old", "project-other"):
            connection.execute(tables["projects"].insert().values(
                id=project_id, workspace_id=LOCAL_ID, name=project_id,
                description="", created_at=WHEN,
            ))
        for run_id in ("run-source", "run-frozen"):
            connection.execute(tables["analysis_runs"].insert().values(
                id=run_id, project_id="project-old", status="completed",
                created_at=WHEN, completed_at=WHEN, input_chars=20,
                prompt_tokens=0, completion_tokens=0, estimated_cost_usd=0.0,
                cancel_requested=False, batch_mode="full_review",
                sensitivity="balanced", batch_coverage={},
            ))
        connection.execute(tables["character_trait_axes"].insert().values(
            id="axis-old", project_id="project-old",
            trait_type="core_personality", version=1,
            display_name="签名行为", definition="是否冒用签名",
            definition_sha256=_sha("是否冒用签名"),
            positive_proposition="角色拒绝冒用签名",
            positive_proposition_sha256=_sha("角色拒绝冒用签名"),
            positive_proposition_authored_by_user_id=LOCAL_ID,
            positive_proposition_authored_at=WHEN,
            created_by_user_id=LOCAL_ID, created_at=WHEN,
        ))
        connection.execute(tables["character_trait_candidates"].insert().values(
            id="candidate-old", project_id="project-old",
            source_run_id="run-source", character_key="角色",
            character_display_name="角色", trait_type="core_personality",
            trait_key="signature_integrity", value="拒绝冒用签名",
            polarity="positive", stability="stable", contexts=[],
            origin="explicit_setting", authority_tier="formal_record",
            confidence=0.9, scope_payload=scope,
            scope_sha256=payload_sha256(scope), evidence=evidence,
            evidence_sha256=payload_sha256(evidence),
            support_binding_mode="legacy_v1", candidate_fingerprint="f" * 64,
            generator_version="test-v1", provenance={}, review_state="confirmed",
            lock_version=1, approved_axis_id="axis-old", approved_axis_version=1,
            axis_alignment="same", axis_polarity="positive",
            axis_positive_proposition_sha256=_sha("角色拒绝冒用签名"),
            reviewed_at=WHEN, reviewed_by_user_id=LOCAL_ID, created_at=WHEN,
        ))
        connection.execute(tables["character_trait_reviews"].insert().values(
            id="confirm-old", project_id="project-old",
            candidate_id="candidate-old", decision="confirm",
            approved_axis_id="axis-old", approved_axis_version=1,
            axis_alignment="same", axis_polarity="positive",
            axis_positive_proposition_sha256=_sha("角色拒绝冒用签名"),
            expected_lock_version=0, comment="", created_by_user_id=LOCAL_ID,
            created_at=WHEN,
        ))
        connection.execute(tables["analysis_run_character_trait_inputs"].insert().values(
            id="snapshot-old", run_id="run-frozen", project_id="project-old",
            candidate_id="candidate-old", confirmation_review_id="confirm-old",
            candidate_lock_version=1, ordinal=0, payload=frozen,
            payload_sha256=payload_sha256(frozen),
        ))


def _triggers(connection) -> set[str]:
    return {
        row[0] for row in connection.execute(text(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND "
            "tbl_name IN ('character_trait_axes', 'character_trait_candidates', "
            "'character_trait_reviews')"
        )).all()
    }


def test_sqlite_upgrade_preserves_old_axis_links_guards_and_frozen_hash():
    with tempfile.TemporaryDirectory() as directory:
        url = f"sqlite:///{(Path(directory) / 'axes.db').as_posix()}"
        _migrate(url, "0019_project_name_sort_key")
        engine = create_engine(url)
        _seed_legacy_bound_trait(engine)
        with engine.connect() as connection:
            before_triggers = _triggers(connection)
            before_snapshot = connection.execute(text(
                "SELECT payload_sha256 FROM analysis_run_character_trait_inputs "
                "WHERE id='snapshot-old'"
            )).scalar_one()
        engine.dispose()

        _migrate(url, "0020_value_boundary_author_axes")
        engine = create_engine(url)
        with engine.connect() as connection:
            assert _triggers(connection) == before_triggers
            index_names = {
                item["name"] for item in inspect(connection).get_indexes(
                    "character_trait_axes"
                )
            }
            assert "uq_character_trait_axis_core_definition" in index_names
            assert "uq_character_trait_axis_scoped_definition" in index_names
            connection.execute(text("PRAGMA foreign_keys=ON"))
            assert connection.execute(text("PRAGMA foreign_key_check")).all() == []
            assert connection.execute(text("PRAGMA quick_check")).scalar_one() == "ok"
            assert connection.execute(text(
                "SELECT payload_sha256 FROM analysis_run_character_trait_inputs "
                "WHERE id='snapshot-old'"
            )).scalar_one() == before_snapshot
            assert connection.execute(text(
                "SELECT comparison_key, applicability_scope, "
                "applicability_scope_sha256 FROM character_trait_axes "
                "WHERE id='axis-old'"
            )).one() == (None, None, None)

        def rejected(statement: str, params: dict | None = None) -> None:
            with pytest.raises(IntegrityError), engine.begin() as connection:
                connection.execute(text(statement), params or {})

        rejected("UPDATE character_trait_axes SET definition='changed' WHERE id='axis-old'")
        rejected(
            "UPDATE character_trait_axes SET positive_proposition='changed' "
            "WHERE id='axis-old'"
        )
        rejected(
            "UPDATE character_trait_axes SET applicability_scope='all' "
            "WHERE id='axis-old'"
        )
        rejected(
            "INSERT INTO character_trait_axes "
            "(id, project_id, trait_type, version, display_name, definition, "
            "definition_sha256, created_at) VALUES "
            "('axis-invalid', 'project-old', 'value', 1, '价值', '帮助家人', "
            ":digest, '2026-09-27 00:00:00')",
            {"digest": _sha("帮助家人")},
        )
        rejected(
            "INSERT INTO character_trait_axes "
            "(id, project_id, trait_type, version, display_name, definition, "
            "definition_sha256, created_at) VALUES "
            "('axis-core-duplicate', 'project-old', 'core_personality', 1, "
            "'另一名称', '是否冒用签名', :digest, '2026-09-27 00:00:00')",
            {"digest": _sha("是否冒用签名")},
        )
        rejected(
            "UPDATE character_trait_candidates SET approved_axis_id='axis-other' "
            "WHERE id='candidate-old'"
        )
        rejected(
            "UPDATE character_trait_reviews SET approved_axis_id='axis-other' "
            "WHERE id='confirm-old'"
        )
        with engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO character_trait_axes "
                "(id, project_id, trait_type, version, display_name, definition, "
                "definition_sha256, comparison_key, applicability_scope, "
                "applicability_scope_sha256, positive_proposition, "
                "positive_proposition_sha256, positive_proposition_authored_at, "
                "created_at) VALUES "
                "('axis-value', 'project-old', 'value', 1, '签名价值', '拒绝冒用签名', "
                ":definition_hash, 'value:签名', '对外签名时', :scope_hash, "
                "'角色拒绝冒用签名', :proposition_hash, "
                "'2026-09-27 00:00:00', '2026-09-27 00:00:00')"
            ), {
                "definition_hash": _sha("拒绝冒用签名"),
                "scope_hash": _sha("对外签名时"),
                "proposition_hash": _sha("角色拒绝冒用签名"),
            })
        def scoped_insert(axis_id: str, key: str, scope: str) -> None:
            with engine.begin() as connection:
                connection.execute(text(
                    "INSERT INTO character_trait_axes "
                    "(id, project_id, trait_type, version, display_name, definition, "
                    "definition_sha256, comparison_key, applicability_scope, "
                    "applicability_scope_sha256, positive_proposition, "
                    "positive_proposition_sha256, positive_proposition_authored_at, "
                    "created_at) VALUES "
                    "(:axis_id, 'project-old', 'value', 1, '签名价值', '拒绝冒用签名', "
                    ":definition_hash, :key, :scope, :scope_hash, "
                    "'角色拒绝冒用签名', :proposition_hash, "
                    "'2026-09-27 00:00:00', '2026-09-27 00:00:00')"
                ), {
                    "axis_id": axis_id, "definition_hash": _sha("拒绝冒用签名"),
                    "key": key, "scope": scope, "scope_hash": _sha(scope),
                    "proposition_hash": _sha("角色拒绝冒用签名"),
                })

        scoped_insert("axis-value-other-object", "value:父亲", "对外签名时")
        scoped_insert("axis-value-other-scope", "value:签名", "内部签名时")
        with pytest.raises(IntegrityError):
            scoped_insert("axis-value-duplicate", "value:签名", "对外签名时")
        rejected(
            "UPDATE character_trait_axes SET applicability_scope='其他时候' "
            "WHERE id='axis-value'"
        )
        rejected(
            "UPDATE character_trait_axes SET comparison_key='value:父亲' "
            "WHERE id='axis-value'"
        )
        with engine.connect() as connection:
            connection.execute(text("PRAGMA foreign_keys=ON"))
            assert connection.execute(text("PRAGMA foreign_key_check")).all() == []
        engine.dispose()


def test_sqlite_downgrade_rejects_new_axes_and_round_trips_core_only():
    with tempfile.TemporaryDirectory() as directory:
        url = f"sqlite:///{(Path(directory) / 'axes.db').as_posix()}"
        _migrate(url, "0020_value_boundary_author_axes")
        engine = create_engine(url)
        with engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO projects (id, workspace_id, name, description, created_at) "
                "VALUES ('project-one', :workspace, 'One', '', '2026-09-27 00:00:00')"
            ), {"workspace": LOCAL_ID})
            connection.execute(text(
                "INSERT INTO character_trait_axes "
                "(id, project_id, trait_type, version, display_name, definition, "
                "definition_sha256, comparison_key, applicability_scope, "
                "applicability_scope_sha256, positive_proposition, "
                "positive_proposition_sha256, positive_proposition_authored_at, "
                "created_at) VALUES "
                "('axis-value', 'project-one', 'value', 1, '价值', '帮助家人', "
                ":definition_hash, 'value:家人', '家庭危机时', :scope_hash, "
                "'角色帮助家人', :proposition_hash, "
                "'2026-09-27 00:00:00', '2026-09-27 00:00:00')"
            ), {
                "definition_hash": _sha("帮助家人"),
                "scope_hash": _sha("家庭危机时"),
                "proposition_hash": _sha("角色帮助家人"),
            })
        engine.dispose()
        with pytest.raises(RuntimeError, match="value/boundary axes exist"):
            _migrate(url, "0019_project_name_sort_key", downgrade=True)
        engine = create_engine(url)
        with engine.begin() as connection:
            connection.execute(text("DELETE FROM character_trait_axes WHERE id='axis-value'"))
        engine.dispose()
        _migrate(url, "0019_project_name_sort_key", downgrade=True)
        engine = create_engine(url)
        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA quick_check")).scalar_one() == "ok"
            assert connection.execute(text(
                "SELECT 1 FROM pragma_table_info('character_trait_axes') "
                "WHERE name='comparison_key'"
            )).first() is None
        engine.dispose()


@pytest.mark.parametrize("dimension", ("value", "behavior_boundary"))
def test_scoped_axis_snapshot_requires_object_and_situation_and_single_behavior_abstains(dimension):
    evidence = EvidenceSpan(
        document_id="profile", document_name="profile.md",
        line_start=1, line_end=1, text="林澈在家庭危机时保护家人。",
    )
    definition = "家庭危机时是否保护家人"
    proposition = "林澈在家庭危机时保护家人"
    baseline_payload = {
        "id": "ct_scoped", "character": "林澈", "dimension": dimension,
        "trait_key": "保护家人", "statement": proposition,
        "polarity": "positive", "stability": "stable",
        "origin": "explicit_setting", "evidence": (evidence,),
        "approved_axis_id": "00000000-0000-4000-8000-000000000001",
        "approved_axis_version": 1, "approved_axis_display_name": "家庭保护",
        "approved_axis_definition": definition,
        "approved_axis_definition_sha256": _sha(definition),
        "approved_axis_comparison_key": f"{dimension}:家人",
        "approved_axis_applicability_scope": "家庭危机时",
        "approved_axis_applicability_scope_sha256": _sha("家庭危机时"),
        "axis_positive_proposition": proposition,
        "axis_positive_proposition_sha256": _sha(proposition),
        "axis_alignment": "same", "axis_polarity": "positive",
    }
    baseline = ConfirmedTraitSnapshot.model_validate(baseline_payload)
    for invalid in (
        {key: value for key, value in baseline_payload.items()
         if key != "approved_axis_applicability_scope"},
        {**baseline_payload, "approved_axis_comparison_key": "value:陌生人"
         if dimension == "behavior_boundary" else "behavior_boundary:家人"},
        {**baseline_payload, "approved_axis_applicability_scope_sha256": "0" * 64},
    ):
        with pytest.raises(ValidationError):
            ConfirmedTraitSnapshot.model_validate(invalid)
    observation = CharacterSignal(
        id="cs_" + "a" * 32, character="林澈", dimension=dimension,
        trait_key="保护家人", statement="林澈离开了家人",
        polarity="negative", stability="situational",
        observation_kind="action", key_object="家人", source_kind="draft",
        evidence=EvidenceSpan(
            document_id="draft", document_name="draft.md",
            line_start=1, line_end=1, text="林澈离开了家人。",
        ),
    )
    case = CharacterDriftCase(
        id="cdc_scoped", baseline=baseline, observations=(observation,),
        scope_compatibility="compatible", material_coverage="complete",
        explanation_coverage="complete",
        approved_axis_bound_observation_ids=(observation.id,),
        approved_axis_observation_polarities=((observation.id, "negative"),),
    )
    prepared = prepare_character_drift(case)
    assert prepared.reason == "single_behavior_is_not_drift"
    assert prepared.reviewer_eligible is True
    assert prepared.deterministic_conflict is False


@pytest.mark.parametrize("dimension", ("value", "behavior_boundary"))
def test_service_freezes_only_exact_object_and_author_scope(monkeypatch, dimension):
    monkeypatch.setattr(
        "app.service.verified_character_trait_review_chain",
        lambda _db, _row: [],
    )
    axis_id = "00000000-0000-4000-8000-000000000001"
    proposition = "林澈在家庭危机时保护家人"
    axis = SimpleNamespace(
        id=axis_id, project_id="project-one", trait_type=dimension,
        version=1, display_name="家庭保护", definition="家庭危机时是否保护家人",
        definition_sha256=_sha("家庭危机时是否保护家人"),
        comparison_key=f"{dimension}:家人",
        applicability_scope="家庭危机时",
        applicability_scope_sha256=_sha("家庭危机时"),
        positive_proposition=proposition,
        positive_proposition_sha256=_sha(proposition),
        positive_proposition_authored_at=WHEN,
    )
    candidate = SimpleNamespace(
        project_id="project-one", trait_type=dimension,
        comparison_key=f"{dimension}:家人", polarity="positive",
        approved_axis_id=axis_id, approved_axis_version=1,
        axis_alignment="same", axis_polarity="positive",
        axis_positive_proposition_sha256=_sha(proposition),
    )
    review = SimpleNamespace(approved_axis_id=axis_id, approved_axis_version=1)
    db = SimpleNamespace(get=lambda _table, _id: axis)
    frozen = _approved_axis_snapshot_fields(db, candidate, review)
    assert frozen["approved_axis_comparison_key"] == f"{dimension}:家人"
    assert frozen["approved_axis_applicability_scope"] == "家庭危机时"
    assert frozen["approved_axis_applicability_scope_sha256"] == _sha("家庭危机时")
    candidate.comparison_key = f"{dimension}:陌生人"
    with pytest.raises(ValueError, match="object or scope"):
        _approved_axis_snapshot_fields(db, candidate, review)
    candidate.comparison_key = f"{dimension}:家人"
    axis.applicability_scope_sha256 = "0" * 64
    with pytest.raises(ValueError, match="object or scope"):
        _approved_axis_snapshot_fields(db, candidate, review)
    axis.applicability_scope_sha256 = _sha("家庭危机时")
    candidate.axis_alignment = None
    candidate.axis_polarity = None
    candidate.axis_positive_proposition_sha256 = None
    with pytest.raises(ValueError, match="alignment is missing"):
        _approved_axis_snapshot_fields(db, candidate, review)


@pytest.mark.skipif(
    not os.environ.get("LOREGUARD_TEST_POSTGRES_URL"),
    reason="isolated PostgreSQL test URL unavailable",
)
def test_postgres_upgrade_preserves_bound_trait_and_enforces_new_axis_scope():
    base_url = os.environ["LOREGUARD_TEST_POSTGRES_URL"]
    schema = f"loreguard_scoped_axis_{uuid.uuid4().hex[:12]}"
    admin = create_engine(base_url)
    with admin.begin() as connection:
        connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
    url = make_url(base_url).update_query_dict(
        {"options": f"-csearch_path={schema},public"}
    ).render_as_string(hide_password=False)
    try:
        _migrate(url, "0019_project_name_sort_key")
        engine = create_engine(url)
        _seed_legacy_bound_trait(engine)
        with engine.connect() as connection:
            frozen_hash = connection.execute(text(
                "SELECT payload_sha256 FROM analysis_run_character_trait_inputs "
                "WHERE id='snapshot-old'"
            )).scalar_one()
        engine.dispose()
        _migrate(url, "0020_value_boundary_author_axes")
        engine = create_engine(url)
        checks = {
            item["name"] for item in inspect(engine).get_check_constraints(
                "character_trait_axes"
            )
        }
        assert "ck_character_trait_axis_object_scope" in checks
        indexes = {
            item["name"] for item in inspect(engine).get_indexes(
                "character_trait_axes"
            )
        }
        assert "uq_character_trait_axis_core_definition" in indexes
        assert "uq_character_trait_axis_scoped_definition" in indexes
        with engine.connect() as connection:
            assert connection.execute(text(
                "SELECT payload_sha256 FROM analysis_run_character_trait_inputs "
                "WHERE id='snapshot-old'"
            )).scalar_one() == frozen_hash
            assert connection.execute(text(
                "SELECT comparison_key, applicability_scope FROM "
                "character_trait_axes WHERE id='axis-old'"
            )).one() == (None, None)
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(text(
                "UPDATE character_trait_axes SET applicability_scope='new' "
                "WHERE id='axis-old'"
            ))
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO character_trait_axes "
                "(id, project_id, trait_type, version, display_name, definition, "
                "definition_sha256, created_at) VALUES "
                "('axis-invalid', 'project-old', 'value', 1, 'Value', 'Help', "
                ":digest, :created)"
            ), {"digest": _sha("Help"), "created": WHEN})
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(text(
                "INSERT INTO character_trait_axes "
                "(id, project_id, trait_type, version, display_name, definition, "
                "definition_sha256, created_at) VALUES "
                "('axis-core-duplicate', 'project-old', 'core_personality', 1, "
                "'Duplicate core', '是否冒用签名', :digest, :created)"
            ), {"digest": _sha("是否冒用签名"), "created": WHEN})

        def scoped_insert(axis_id: str, key: str, scope: str) -> None:
            with engine.begin() as connection:
                connection.execute(text(
                    "INSERT INTO character_trait_axes "
                    "(id, project_id, trait_type, version, display_name, definition, "
                    "definition_sha256, comparison_key, applicability_scope, "
                    "applicability_scope_sha256, positive_proposition, "
                    "positive_proposition_sha256, positive_proposition_authored_at, "
                    "created_at) VALUES "
                    "(:axis_id, 'project-old', 'value', 1, 'Signing value', "
                    "'拒绝冒用签名', :definition_hash, :key, :scope, :scope_hash, "
                    "'角色拒绝冒用签名', :proposition_hash, :created, :created)"
                ), {
                    "axis_id": axis_id, "definition_hash": _sha("拒绝冒用签名"),
                    "key": key, "scope": scope, "scope_hash": _sha(scope),
                    "proposition_hash": _sha("角色拒绝冒用签名"),
                    "created": WHEN,
                })

        scoped_insert("axis-value", "value:签名", "对外签名时")
        scoped_insert("axis-value-other-object", "value:父亲", "对外签名时")
        scoped_insert("axis-value-other-scope", "value:签名", "内部签名时")
        with pytest.raises(IntegrityError):
            scoped_insert("axis-value-duplicate", "value:签名", "对外签名时")
        with engine.begin() as connection:
            connection.execute(text(
                "DELETE FROM character_trait_axes WHERE id IN "
                "('axis-value', 'axis-value-other-object', 'axis-value-other-scope')"
            ))
        engine.dispose()
        _migrate(url, "0019_project_name_sort_key", downgrade=True)
        engine = create_engine(url)
        assert "comparison_key" not in {
            item["name"] for item in inspect(engine).get_columns("character_trait_axes")
        }
        with engine.connect() as connection:
            assert connection.execute(text(
                "SELECT approved_axis_id FROM character_trait_candidates "
                "WHERE id='candidate-old'"
            )).scalar_one() == "axis-old"
        engine.dispose()
    finally:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        admin.dispose()
