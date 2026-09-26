"""PostgreSQL gate for upgrading populated 0017 author-axis data to 0018."""

from __future__ import annotations

import hashlib
import os
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import MetaData, Table, create_engine, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app import main as app_main
from app.narrative_context import payload_sha256


ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = "0017_character_trait_withdraw"
HEAD = "0018_character_axis_direction"
LOCAL_ID = "00000000-0000-0000-0000-000000000001"
WHEN = datetime(2026, 9, 1)


def _config(url: str) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["database_url"] = url
    return config


@contextmanager
def _isolated_postgres_url():
    base_url = os.environ["LOREGUARD_TEST_POSTGRES_URL"]
    schema = f"loreguard_axis_{uuid.uuid4().hex[:12]}"
    admin = create_engine(base_url)
    try:
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        url = make_url(base_url).update_query_dict(
            {"options": f"-csearch_path={schema},public"}
        ).render_as_string(hide_password=False)
        try:
            yield url
        finally:
            with admin.begin() as connection:
                connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
    finally:
        admin.dispose()


def _seed_populated_0017(engine) -> dict[str, Table]:
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
    evidence = [{"document_id": "doc-old", "text": "角色拒绝冒用签名。"}]
    frozen = {
        "schema_version": 1,
        "candidate_id": "candidate-old",
        "approved_axis_id": "axis-old",
        "polarity": "positive",
        "evidence": evidence,
    }
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
        for axis_id, project_id in (
            ("axis-old", "project-old"), ("axis-other", "project-other")
        ):
            connection.execute(tables["character_trait_axes"].insert().values(
                id=axis_id, project_id=project_id, trait_type="core_personality",
                version=1, display_name="签名行为", definition="是否冒用签名",
                definition_sha256="a" * 64, created_by_user_id=LOCAL_ID,
                created_at=WHEN,
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
            reviewed_at=WHEN, reviewed_by_user_id=LOCAL_ID, created_at=WHEN,
        ))
        connection.execute(tables["character_trait_reviews"].insert().values(
            id="confirm-old", project_id="project-old",
            candidate_id="candidate-old", decision="confirm",
            approved_axis_id="axis-old", approved_axis_version=1,
            expected_lock_version=0, idempotency_key="legacy-confirm",
            comment="legacy axis confirmation", created_by_user_id=LOCAL_ID,
            created_at=WHEN,
        ))
        connection.execute(tables["analysis_run_character_trait_inputs"].insert().values(
            id="snapshot-old", run_id="run-frozen", project_id="project-old",
            candidate_id="candidate-old", confirmation_review_id="confirm-old",
            candidate_lock_version=1, ordinal=0, payload=frozen,
            payload_sha256=payload_sha256(frozen),
        ))
    return tables


def _old_rows(engine, tables: dict[str, Table]) -> dict[str, dict]:
    # These Table objects were reflected at 0017, so only historical columns
    # are compared after 0018 adds nullable direction fields.
    identities = {
        "character_trait_axes": "axis-old",
        "character_trait_candidates": "candidate-old",
        "character_trait_reviews": "confirm-old",
        "analysis_run_character_trait_inputs": "snapshot-old",
    }
    with engine.connect() as connection:
        return {
            name: dict(connection.execute(
                select(table).where(table.c.id == identities[name])
            ).mappings().one())
            for name, table in tables.items() if name in identities
        }


@unittest.skipUnless(
    os.environ.get("LOREGUARD_TEST_POSTGRES_URL"),
    "requires an explicitly disposable PostgreSQL/pgvector test database",
)
class PgAxisDirectionMigrationTests(unittest.TestCase):
    def test_populated_0017_upgrade_constraints_and_guarded_downgrade(self):
        with _isolated_postgres_url() as url:
            config = _config(url)
            command.upgrade(config, PREVIOUS)
            engine = create_engine(url)
            try:
                tables = _seed_populated_0017(engine)
                before = _old_rows(engine, tables)
                command.upgrade(config, HEAD)
                self.assertEqual(_old_rows(engine, tables), before)
                with engine.connect() as connection:
                    self.assertEqual(connection.execute(text(
                        "SELECT version_num FROM alembic_version"
                    )).scalar_one(), HEAD)
                    self.assertEqual(connection.execute(text(
                        "SELECT positive_proposition FROM character_trait_axes "
                        "WHERE id='axis-old'"
                    )).scalar_one(), None)
                    self.assertEqual(connection.execute(text(
                        "SELECT axis_alignment, axis_polarity, "
                        "axis_positive_proposition_sha256 "
                        "FROM character_trait_candidates WHERE id='candidate-old'"
                    )).one(), (None, None, None))
                    self.assertEqual(connection.execute(text(
                        "SELECT axis_alignment, axis_polarity, "
                        "axis_positive_proposition_sha256 "
                        "FROM character_trait_reviews WHERE id='confirm-old'"
                    )).one(), (None, None, None))

                def rejected(sql: str, **params) -> None:
                    with self.assertRaises(IntegrityError):
                        with engine.begin() as connection:
                            connection.execute(text(sql), params)

                rejected(
                    "UPDATE character_trait_candidates SET approved_axis_id='axis-other' "
                    "WHERE id='candidate-old'"
                )
                rejected(
                    "UPDATE character_trait_candidates SET approved_axis_version=2 "
                    "WHERE id='candidate-old'"
                )
                rejected(
                    "UPDATE character_trait_candidates SET axis_alignment='same', "
                    "axis_polarity='negative', "
                    "axis_positive_proposition_sha256=:digest "
                    "WHERE id='candidate-old'", digest="c" * 64,
                )
                rejected(
                    "UPDATE character_trait_reviews SET axis_alignment='same' "
                    "WHERE id='confirm-old'"
                )
                rejected(
                    "UPDATE character_trait_axes SET positive_proposition='冒用签名' "
                    "WHERE id='axis-old'"
                )
                self.assertEqual(_old_rows(engine, tables), before)

                command.downgrade(config, PREVIOUS)
                self.assertEqual(_old_rows(engine, tables), before)
                self.assertNotIn("axis_alignment", {
                    column["name"] for column in inspect(engine).get_columns(
                        "character_trait_reviews"
                    )
                })
                command.upgrade(config, HEAD)
                self.assertEqual(_old_rows(engine, tables), before)

                # A confirm review may have direction data even when its
                # candidate is still legacy-unverified. Rollback must keep it.
                with engine.begin() as connection:
                    connection.execute(text(
                        "UPDATE character_trait_reviews SET axis_alignment='same', "
                        "axis_polarity='positive', "
                        "axis_positive_proposition_sha256=:digest "
                        "WHERE id='confirm-old'"
                    ), {"digest": "c" * 64})
                with self.assertRaisesRegex(RuntimeError, "direction decisions"):
                    command.downgrade(config, PREVIOUS)
                with engine.begin() as connection:
                    self.assertEqual(connection.execute(text(
                        "SELECT version_num FROM alembic_version"
                    )).scalar_one(), HEAD)
                    self.assertEqual(connection.execute(text(
                        "SELECT axis_alignment FROM character_trait_reviews "
                        "WHERE id='confirm-old'"
                    )).scalar_one(), "same")
                    connection.execute(text(
                        "UPDATE character_trait_reviews SET axis_alignment=NULL, "
                        "axis_polarity=NULL, axis_positive_proposition_sha256=NULL "
                        "WHERE id='confirm-old'"
                    ))

                proposition = "角色未经授权冒用签名"
                digest = hashlib.sha256(proposition.encode("utf-8")).hexdigest()
                with engine.begin() as connection:
                    connection.execute(text(
                        "UPDATE character_trait_axes SET "
                        "positive_proposition=:proposition, "
                        "positive_proposition_sha256=:digest, "
                        "positive_proposition_authored_at=:authored_at, "
                        "positive_proposition_authored_by_user_id=:user_id "
                        "WHERE id='axis-old'"
                    ), {"proposition": proposition, "digest": digest,
                        "authored_at": WHEN, "user_id": LOCAL_ID})
                    connection.execute(text(
                        "UPDATE character_trait_candidates SET "
                        "axis_alignment='opposite', axis_polarity='negative', "
                        "axis_positive_proposition_sha256=:digest "
                        "WHERE id='candidate-old'"
                    ), {"digest": digest})
                    connection.execute(text(
                        "UPDATE character_trait_reviews SET "
                        "axis_alignment='opposite', axis_polarity='negative', "
                        "axis_positive_proposition_sha256=:digest "
                        "WHERE id='confirm-old'"
                    ), {"digest": digest})
                with engine.connect() as connection:
                    self.assertEqual(connection.execute(text(
                        "SELECT polarity, axis_alignment, axis_polarity "
                        "FROM character_trait_candidates WHERE id='candidate-old'"
                    )).one(), ("positive", "opposite", "negative"))
                    self.assertEqual(connection.execute(text(
                        "SELECT payload_sha256 FROM analysis_run_character_trait_inputs "
                        "WHERE id='snapshot-old'"
                    )).scalar_one(), before[
                        "analysis_run_character_trait_inputs"
                    ]["payload_sha256"])
                rejected(
                    "UPDATE character_trait_axes SET "
                    "positive_proposition='different' WHERE id='axis-old'"
                )
                with self.assertRaisesRegex(RuntimeError, "direction decisions"):
                    command.downgrade(config, PREVIOUS)
                with engine.connect() as connection:
                    self.assertEqual(connection.execute(text(
                        "SELECT version_num FROM alembic_version"
                    )).scalar_one(), HEAD)
                    self.assertEqual(connection.execute(text(
                        "SELECT positive_proposition_sha256 FROM character_trait_axes "
                        "WHERE id='axis-old'"
                    )).scalar_one(), digest)
            finally:
                engine.dispose()

    def test_competing_author_propositions_commit_once(self):
        with _isolated_postgres_url() as url:
            config = _config(url)
            command.upgrade(config, PREVIOUS)
            engine = create_engine(url)
            try:
                _seed_populated_0017(engine)
                command.upgrade(config, HEAD)
                ready = Barrier(2)

                def author(proposition: str) -> tuple[int, str | None]:
                    ready.wait(timeout=10)
                    response = client.post(
                        "/api/v1/projects/project-old/character-trait-axes/"
                        "axis-old/positive-proposition",
                        json={"positive_proposition": proposition,
                              "expected_axis_version": 1},
                    )
                    return (
                        response.status_code,
                        response.json().get("positive_proposition"),
                    )

                pg_session = sessionmaker(bind=engine, expire_on_commit=False)
                with patch.object(app_main, "SessionLocal", pg_session):
                    with TestClient(app_main.app) as client:
                        with ThreadPoolExecutor(max_workers=2) as pool:
                            first = pool.submit(author, "角色主动交谈")
                            second = pool.submit(author, "角色拒绝交谈")
                            results = [first.result(timeout=20), second.result(timeout=20)]
                self.assertEqual(sorted(status for status, _ in results), [200, 409])
                winner = next(value for status, value in results if status == 200)
                with engine.connect() as connection:
                    stored = connection.execute(text(
                        "SELECT positive_proposition, positive_proposition_sha256, "
                        "positive_proposition_authored_by_user_id "
                        "FROM character_trait_axes WHERE id='axis-old'"
                    )).one()
                self.assertEqual(stored[0], winner)
                self.assertEqual(
                    stored[1], hashlib.sha256(winner.encode("utf-8")).hexdigest()
                )
                self.assertEqual(stored[2], LOCAL_ID)
            finally:
                engine.dispose()
