from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import MetaData, Table, create_engine, inspect, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateIndex

from app.db import Base
from app import db as app_db
from app.narrative_context import payload_sha256


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_TABLES = {"embedding_profiles", "evidence_chunks", "evidence_embeddings"}
HEAD_REVISION = "0019_project_name_sort_key"


class EmbeddingMigrationTests(unittest.TestCase):
    def test_axis_direction_migration_keeps_old_confirmations_unverified_and_guards_new_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            url = f"sqlite:///{(Path(directory) / 'axis-direction.db').as_posix()}"
            self.upgrade_to(url, "0017_character_trait_withdraw")
            engine = create_engine(url)
            axis = Table("character_trait_axes", MetaData(), autoload_with=engine)
            candidate = Table("character_trait_candidates", MetaData(), autoload_with=engine)
            review = Table("character_trait_reviews", MetaData(), autoload_with=engine)
            scope = {"schema_version": 1, "timeline_key": "main"}
            evidence = [{"document_id": "doc-old", "text": "角色拒绝冒用签名。"}]
            with engine.begin() as connection:
                connection.execute(axis.insert().values(
                    id="axis-old", project_id="project-old",
                    trait_type="core_personality", version=1,
                    display_name="签名行为", definition="是否冒用签名",
                    definition_sha256="a" * 64,
                    created_at=datetime(2026, 9, 1),
                ))
                connection.execute(candidate.insert().values(
                    id="candidate-old", project_id="project-old",
                    source_run_id="run-old", character_key="角色",
                    character_display_name="角色", trait_type="core_personality",
                    trait_key="signature_integrity", value="拒绝冒用签名",
                    polarity="positive", stability="stable", contexts=[],
                    origin="explicit_setting", authority_tier="formal_record",
                    confidence=0.9, scope_payload=scope,
                    scope_sha256=payload_sha256(scope), evidence=evidence,
                    evidence_sha256=payload_sha256(evidence),
                    support_binding_mode="legacy_v1",
                    candidate_fingerprint="f" * 64,
                    generator_version="test-v1", provenance={},
                    review_state="confirmed", lock_version=1,
                    approved_axis_id="axis-old", approved_axis_version=1,
                    reviewed_at=datetime(2026, 9, 1),
                    created_at=datetime(2026, 9, 1),
                ))
                connection.execute(review.insert().values(
                    id="confirm-old", project_id="project-old",
                    candidate_id="candidate-old", decision="confirm",
                    approved_axis_id="axis-old", approved_axis_version=1,
                    expected_lock_version=0, comment="",
                    created_at=datetime(2026, 9, 1),
                ))
                before_review_triggers = {
                    name for (name,) in connection.exec_driver_sql(
                        "SELECT name FROM sqlite_master WHERE type='trigger' "
                        "AND tbl_name='character_trait_reviews'"
                    ).all()
                }
            engine.dispose()
            self.upgrade(url)
            engine = create_engine(url)
            with engine.begin() as connection:
                after_review_triggers = {
                    name for (name,) in connection.exec_driver_sql(
                        "SELECT name FROM sqlite_master WHERE type='trigger' "
                        "AND tbl_name='character_trait_reviews'"
                    ).all()
                }
                self.assertTrue(before_review_triggers <= after_review_triggers)
                self.assertEqual(connection.exec_driver_sql(
                    "SELECT approved_axis_id, axis_alignment, axis_polarity, "
                    "axis_positive_proposition_sha256 FROM character_trait_candidates "
                    "WHERE id='candidate-old'"
                ).one(), ("axis-old", None, None, None))
                self.assertEqual(connection.exec_driver_sql(
                    "SELECT decision, axis_alignment FROM character_trait_reviews "
                    "WHERE id='confirm-old'"
                ).one(), ("confirm", None))
                with self.assertRaises(IntegrityError):
                    connection.exec_driver_sql(
                        "UPDATE character_trait_candidates SET axis_alignment='opposite' "
                        "WHERE id='candidate-old'"
                    )
                self.assertEqual(
                    connection.exec_driver_sql("PRAGMA quick_check").scalar_one(),
                    "ok",
                )
            engine.dispose()
            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            config.attributes["database_url"] = url
            command.downgrade(config, "0017_character_trait_withdraw")
            engine = create_engine(url)
            self.assertNotIn(
                "axis_alignment",
                {item["name"] for item in inspect(engine).get_columns(
                    "character_trait_candidates"
                )},
            )
            engine.dispose()
            self.upgrade(url)
            engine = create_engine(url)
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "UPDATE character_trait_reviews SET axis_alignment='same', "
                    "axis_polarity='positive', axis_positive_proposition_sha256=? "
                    "WHERE id='confirm-old'",
                    ("c" * 64,),
                )
            engine.dispose()
            with self.assertRaisesRegex(RuntimeError, "direction decisions"):
                command.downgrade(config, "0017_character_trait_withdraw")
            engine = create_engine(url)
            with engine.begin() as connection:
                self.assertEqual(connection.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar_one(), "0018_character_axis_direction")
                self.assertEqual(connection.exec_driver_sql(
                    "SELECT axis_alignment FROM character_trait_reviews "
                    "WHERE id='confirm-old'"
                ).scalar_one(), "same")
                connection.exec_driver_sql(
                    "UPDATE character_trait_reviews SET axis_alignment=NULL, "
                    "axis_polarity=NULL, axis_positive_proposition_sha256=NULL "
                    "WHERE id='confirm-old'"
                )
            engine.dispose()
            engine = create_engine(url)
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "UPDATE character_trait_axes SET positive_proposition='角色冒用签名', "
                    "positive_proposition_sha256=?, "
                    "positive_proposition_authored_at='2026-09-26 00:00:00' "
                    "WHERE id='axis-old'",
                    ("b" * 64,),
                )
            engine.dispose()
            with self.assertRaisesRegex(RuntimeError, "direction decisions"):
                command.downgrade(config, "0017_character_trait_withdraw")

    def test_axis_direction_unmodified_create_all_schema_can_downgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            url = f"sqlite:///{(Path(directory) / 'created-current.db').as_posix()}"
            engine = create_engine(url)
            Base.metadata.create_all(engine)
            engine.dispose()
            self.upgrade(url)
            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            config.attributes["database_url"] = url
            command.downgrade(config, "0017_character_trait_withdraw")
            engine = create_engine(url)
            with engine.connect() as connection:
                self.assertEqual(connection.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar_one(), "0017_character_trait_withdraw")
                self.assertEqual(connection.exec_driver_sql(
                    "PRAGMA quick_check"
                ).scalar_one(), "ok")
            engine.dispose()

    def test_character_withdrawal_migration_preserves_schema_and_is_reversible_only_without_history(self):
        with tempfile.TemporaryDirectory() as directory:
            url = f"sqlite:///{(Path(directory) / 'withdrawal.db').as_posix()}"
            self.upgrade_to(url, "0016_character_support_bindings")
            engine = create_engine(url)

            def protections(table: str) -> dict[str, set[tuple]]:
                inspector = inspect(engine)
                return {
                    "checks": {(item["name"],) for item in inspector.get_check_constraints(table)},
                    "uniques": {(item["name"], tuple(item["column_names"]))
                                for item in inspector.get_unique_constraints(table)},
                    "indexes": {(item["name"], tuple(item["column_names"]))
                                for item in inspector.get_indexes(table)},
                    "foreign_keys": {
                        (tuple(item["constrained_columns"]), item["referred_table"],
                         tuple(item["referred_columns"]))
                        for item in inspector.get_foreign_keys(table)
                    },
                }

            tables = ("character_trait_candidates", "character_trait_reviews")
            before = {table: protections(table) for table in tables}
            with engine.connect() as connection:
                before_triggers = {
                    table: {
                        name for (name,) in connection.exec_driver_sql(
                            "SELECT name FROM sqlite_master WHERE type='trigger' "
                            "AND tbl_name = ?", (table,)
                        ).all()
                    }
                    for table in tables
                }
            candidate_table = Table(tables[0], MetaData(), autoload_with=engine)
            review_table = Table(tables[1], MetaData(), autoload_with=engine)
            scope = {"schema_version": 1, "timeline_key": "main"}
            evidence = [{"document_id": "doc-legacy", "text": "林澈喜欢蜜瓜。"}]
            with engine.begin() as connection:
                connection.execute(candidate_table.insert().values(
                    id="withdraw-candidate", project_id="project-old",
                    source_run_id="run-old", character_key="林澈",
                    character_display_name="林澈", trait_type="preference",
                    trait_key="食物偏好:蜜瓜", value="喜欢蜜瓜", polarity="positive",
                    stability="stable", contexts=[], origin="explicit_setting",
                    authority_tier="formal_record", confidence=0.9,
                    scope_payload=scope, scope_sha256=payload_sha256(scope),
                    evidence=evidence, evidence_sha256=payload_sha256(evidence),
                    support_binding_mode="legacy_v1", candidate_fingerprint="f" * 64,
                    generator_version="test-v1", provenance={},
                    review_state="confirmed", lock_version=1,
                    reviewed_at=datetime(2026, 9, 1), created_at=datetime(2026, 9, 1),
                ))
                connection.execute(review_table.insert().values(
                    id="confirm-review", project_id="project-old",
                    candidate_id="withdraw-candidate", decision="confirm",
                    expected_lock_version=0, comment="", created_at=datetime(2026, 9, 1),
                ))
            engine.dispose()

            self.upgrade_to(url, "0017_character_trait_withdraw")
            engine = create_engine(url)
            self.assertEqual({table: protections(table) for table in tables}, before)
            with engine.connect() as connection:
                after_triggers = {
                    table: {
                        name for (name,) in connection.exec_driver_sql(
                            "SELECT name FROM sqlite_master WHERE type='trigger' "
                            "AND tbl_name = ?", (table,)
                        ).all()
                    }
                    for table in tables
                }
                self.assertEqual(after_triggers, before_triggers)
                self.assertEqual(connection.exec_driver_sql(
                    "SELECT review_state, lock_version FROM character_trait_candidates "
                    "WHERE id='withdraw-candidate'"
                ).one(), ("confirmed", 1))
            engine.dispose()

            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            config.attributes["database_url"] = url
            command.downgrade(config, "0016_character_support_bindings")
            engine = create_engine(url)
            with engine.begin() as connection:
                with self.assertRaises(IntegrityError):
                    connection.exec_driver_sql(
                        "UPDATE character_trait_candidates SET review_state='withdrawn' "
                        "WHERE id='withdraw-candidate'"
                    )
            engine.dispose()

            self.upgrade_to(url, "0017_character_trait_withdraw")
            engine = create_engine(url)
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "UPDATE character_trait_candidates SET review_state='withdrawn', "
                    "lock_version=2 WHERE id='withdraw-candidate'"
                )
                connection.exec_driver_sql(
                    "INSERT INTO character_trait_reviews "
                    "(id, project_id, candidate_id, decision, expected_lock_version, "
                    "comment, created_at) VALUES "
                    "('withdraw-review', 'project-old', 'withdraw-candidate', "
                    "'withdraw', 1, '单独撤销', '2026-09-26 00:00:00')"
                )
            engine.dispose()
            with self.assertRaisesRegex(RuntimeError, "withdrawn traits"):
                command.downgrade(config, "0016_character_support_bindings")
            engine = create_engine(url)
            with engine.connect() as connection:
                self.assertEqual(connection.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar_one(), "0017_character_trait_withdraw")
                self.assertEqual(connection.exec_driver_sql(
                    "SELECT review_state FROM character_trait_candidates "
                    "WHERE id='withdraw-candidate'"
                ).scalar_one(), "withdrawn")
            engine.dispose()

    def test_support_binding_migration_marks_existing_candidate_legacy_without_rehashing(self):
        with tempfile.TemporaryDirectory() as directory:
            url = f"sqlite:///{(Path(directory) / 'support-binding.db').as_posix()}"
            self.upgrade_to(url, "0015_character_trait_axes")
            engine = create_engine(url)
            before = {item["name"] for item in inspect(engine).get_columns(
                "character_trait_candidates"
            )}
            def protections(inspector):
                table = "character_trait_candidates"
                return {
                    kind: {item["name"] for item in getattr(inspector, method)(table)}
                    for kind, method in (
                        ("foreign_keys", "get_foreign_keys"),
                        ("uniques", "get_unique_constraints"),
                        ("checks", "get_check_constraints"),
                        ("indexes", "get_indexes"),
                    )
                }

            before_protections = protections(inspect(engine))
            self.assertNotIn("support_bindings_v1", before)
            old_candidate_table = Table(
                "character_trait_candidates", MetaData(), autoload_with=engine
            )
            scope = {"schema_version": 1, "timeline_key": "main"}
            evidence = [{"document_id": "document-old", "text": "old line"}]
            with engine.begin() as connection:
                connection.execute(old_candidate_table.insert().values(
                    id="candidate-old",
                    project_id="project-old",
                    source_run_id="run-old",
                    character_key="old-character",
                    character_display_name="old-character",
                    trait_type="preference",
                    trait_key="old-preference",
                    value="old-value",
                    polarity="positive",
                    stability="stable",
                    contexts=[],
                    origin="explicit_setting",
                    authority_tier="formal_record",
                    confidence=0.9,
                    scope_payload=scope,
                    scope_sha256=payload_sha256(scope),
                    evidence=evidence,
                    evidence_sha256=payload_sha256(evidence),
                    # Deliberately not recomputable: a pre-0014 row may have
                    # hashed a comparison key that was never persisted.
                    candidate_fingerprint="f" * 64,
                    generator_version="test-v1",
                    provenance={},
                    review_state="pending",
                    lock_version=0,
                    created_at=datetime(2026, 9, 1),
                ))
            engine.dispose()
            self.upgrade(url)
            engine = create_engine(url)
            after = {item["name"] for item in inspect(engine).get_columns(
                "character_trait_candidates"
            )}
            self.assertIn("support_bindings_v1", after)
            self.assertIn("support_bindings_sha256", after)
            self.assertIn("support_binding_mode", after)
            self.assertEqual(protections(inspect(engine)), before_protections)
            with engine.connect() as connection:
                self.assertEqual(connection.exec_driver_sql(
                    "SELECT support_binding_mode, support_bindings_v1, "
                    "support_bindings_sha256, candidate_fingerprint "
                    "FROM character_trait_candidates WHERE id = 'candidate-old'"
                ).one(), ("legacy_v1", None, None, "f" * 64))
            engine.dispose()
            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            config.attributes["database_url"] = url
            command.downgrade(config, "0015_character_trait_axes")
            engine = create_engine(url)
            after_downgrade = {item["name"] for item in inspect(engine).get_columns(
                "character_trait_candidates"
            )}
            self.assertNotIn("support_bindings_v1", after_downgrade)
            self.assertNotIn("support_bindings_sha256", after_downgrade)
            self.assertNotIn("support_binding_mode", after_downgrade)
            self.assertEqual(protections(inspect(engine)), before_protections)
            with engine.connect() as connection:
                self.assertEqual(connection.exec_driver_sql(
                    "SELECT candidate_fingerprint FROM character_trait_candidates "
                    "WHERE id = 'candidate-old'"
                ).scalar_one(), "f" * 64)
            engine.dispose()

    def test_support_binding_downgrade_rejects_bound_candidate_without_data_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            url = f"sqlite:///{(Path(directory) / 'bound-candidate.db').as_posix()}"
            self.upgrade(url)
            engine = create_engine(url)
            candidate_table = Table(
                "character_trait_candidates", MetaData(), autoload_with=engine
            )
            scope = {"schema_version": 1, "timeline_key": "main"}
            evidence = [{"document_id": "document-1", "text": "test line"}]
            with engine.begin() as connection:
                connection.execute(candidate_table.insert().values(
                    id="candidate-bound",
                    project_id="project-1",
                    source_run_id="run-1",
                    character_key="test-character",
                    character_display_name="test-character",
                    trait_type="preference",
                    trait_key="test-preference",
                    value="test-value",
                    polarity="positive",
                    stability="stable",
                    contexts=[],
                    origin="explicit_setting",
                    authority_tier="formal_record",
                    confidence=0.9,
                    scope_payload=scope,
                    scope_sha256=payload_sha256(scope),
                    evidence=evidence,
                    evidence_sha256=payload_sha256(evidence),
                    support_binding_mode="required_v1",
                    support_bindings_v1={"schema_version": "test"},
                    support_bindings_sha256="f" * 64,
                    candidate_fingerprint="e" * 64,
                    generator_version="test-v1",
                    provenance={},
                    review_state="pending",
                    lock_version=0,
                    created_at=datetime(2026, 9, 1),
                ))
            engine.dispose()
            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            config.attributes["database_url"] = url
            with self.assertRaisesRegex(RuntimeError, "support bindings exist"):
                command.downgrade(config, "0015_character_trait_axes")
            engine = create_engine(url)
            with engine.connect() as connection:
                row = connection.exec_driver_sql(
                    "SELECT support_binding_mode, support_bindings_sha256 "
                    "FROM character_trait_candidates WHERE id = 'candidate-bound'"
                ).one()
                revision = connection.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar_one()
            self.assertEqual(row, ("required_v1", "f" * 64))
            self.assertEqual(revision, "0016_character_support_bindings")
            engine.dispose()

    def test_author_axis_migration_round_trips_on_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            url = f"sqlite:///{(Path(directory) / 'approved-axis.db').as_posix()}"
            self.upgrade(url)
            engine = create_engine(url)
            inspector = inspect(engine)
            self.assertTrue(inspector.has_table("character_trait_axes"))
            for table_name in ("character_trait_candidates", "character_trait_reviews"):
                self.assertIn(
                    "approved_axis_id",
                    {item["name"] for item in inspector.get_columns(table_name)},
                )
            with engine.connect() as connection:
                connection.exec_driver_sql("PRAGMA foreign_keys=ON")
                connection.commit()
                connection.exec_driver_sql(
                    "INSERT INTO users "
                    "(id, email, display_name, password_hash, is_active, created_at) "
                    "VALUES ('axis-author', 'axis-author@example.invalid', "
                    "'Axis Author', 'test-hash', 1, '2026-09-24 00:00:00')"
                )
                connection.exec_driver_sql(
                    "INSERT INTO projects "
                    "(id, workspace_id, name, description, created_at) "
                    "VALUES ('axis-project', "
                    "'00000000-0000-0000-0000-000000000001', "
                    "'Axis Project', '', '2026-09-24 00:00:00')"
                )
                connection.exec_driver_sql(
                    "INSERT INTO character_trait_axes "
                    "(id, project_id, trait_type, version, display_name, "
                    "definition, definition_sha256, created_by_user_id, created_at) "
                    "VALUES ('axis-one', 'axis-project', 'core_personality', 1, "
                    "'主动社交', '面对陌生人时是否主动交谈', "
                    "lower(hex(randomblob(32))), 'axis-author', "
                    "'2026-09-24 00:00:00')"
                )
                connection.commit()
                with self.assertRaises(IntegrityError):
                    connection.exec_driver_sql(
                        "UPDATE character_trait_axes "
                        "SET definition = '另一个含义' WHERE id = 'axis-one'"
                    )
                connection.rollback()
                connection.exec_driver_sql(
                    "DELETE FROM users WHERE id = 'axis-author'"
                )
                connection.commit()
                axis_row = connection.exec_driver_sql(
                    "SELECT definition, created_by_user_id "
                    "FROM character_trait_axes WHERE id = 'axis-one'"
                ).one()
                self.assertEqual(axis_row, ("面对陌生人时是否主动交谈", None))
            engine.dispose()
            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            config.attributes["database_url"] = url
            command.downgrade(config, "0014_trait_comparison_key")
            engine = create_engine(url)
            inspector = inspect(engine)
            self.assertFalse(inspector.has_table("character_trait_axes"))
            for table_name in ("character_trait_candidates", "character_trait_reviews"):
                self.assertNotIn(
                    "approved_axis_id",
                    {item["name"] for item in inspector.get_columns(table_name)},
                )
            engine.dispose()

    def test_trait_comparison_key_migration_preserves_legacy_rows_and_snapshot_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trait-comparison.db"
            url = f"sqlite:///{path.as_posix()}"
            self.upgrade_to(url, "0013_account_model_provider")
            engine = create_engine(url)
            before_columns = {
                item["name"]
                for item in inspect(engine).get_columns("character_trait_candidates")
            }
            self.assertNotIn("comparison_key", before_columns)
            metadata = MetaData()
            candidate_table = Table(
                "character_trait_candidates", metadata, autoload_with=engine
            )
            snapshot_table = Table(
                "analysis_run_character_trait_inputs", metadata, autoload_with=engine
            )
            scope = {"schema_version": 1, "timeline_key": "main"}
            evidence = [{"document_id": "document-1", "text": "林澈喜欢蜜瓜。"}]
            legacy_payload = {"schema_version": 1, "trait_key": "食物偏好"}
            legacy_hash = payload_sha256(legacy_payload)
            with engine.begin() as connection:
                connection.execute(
                    candidate_table.insert().values(
                        id="candidate-legacy",
                        project_id="project-legacy",
                        source_run_id="run-source",
                        character_key="林澈",
                        character_display_name="林澈",
                        trait_type="preference",
                        trait_key="食物偏好",
                        value="喜欢蜜瓜",
                        polarity="positive",
                        stability="stable",
                        contexts=[],
                        origin="explicit_setting",
                        authority_tier="formal_record",
                        confidence=0.9,
                        scope_payload=scope,
                        scope_sha256=payload_sha256(scope),
                        evidence=evidence,
                        evidence_sha256=payload_sha256(evidence),
                        candidate_fingerprint="f" * 64,
                        generator_version="test-v1",
                        provenance={},
                        review_state="confirmed",
                        lock_version=1,
                        created_at=datetime(2026, 9, 1),
                    )
                )
                connection.execute(
                    snapshot_table.insert().values(
                        id="snapshot-legacy",
                        run_id="run-frozen",
                        project_id="project-legacy",
                        candidate_id="candidate-legacy",
                        confirmation_review_id="review-legacy",
                        candidate_lock_version=1,
                        ordinal=0,
                        payload=legacy_payload,
                        payload_sha256=legacy_hash,
                    )
                )
            engine.dispose()

            self.upgrade(url)
            engine = create_engine(url)
            comparison_column = next(
                item
                for item in inspect(engine).get_columns("character_trait_candidates")
                if item["name"] == "comparison_key"
            )
            self.assertTrue(comparison_column["nullable"])
            self.assertEqual(comparison_column["type"].length, 200)
            with engine.connect() as connection:
                candidate = connection.exec_driver_sql(
                    "SELECT comparison_key, candidate_fingerprint, "
                    "approved_axis_id, approved_axis_version "
                    "FROM character_trait_candidates WHERE id = 'candidate-legacy'"
                ).one()
                frozen = connection.execute(
                    select(snapshot_table.c.payload, snapshot_table.c.payload_sha256)
                    .where(snapshot_table.c.id == "snapshot-legacy")
                ).one()
            self.assertIsNone(candidate.comparison_key)
            self.assertIsNone(candidate.approved_axis_id)
            self.assertIsNone(candidate.approved_axis_version)
            self.assertTrue(inspect(engine).has_table("character_trait_axes"))
            with engine.begin() as connection:
                with self.assertRaises(IntegrityError):
                    connection.exec_driver_sql(
                        "UPDATE character_trait_candidates "
                        "SET approved_axis_version = 1 WHERE id = 'candidate-legacy'"
                    )
            with engine.connect() as connection:
                still_legacy = connection.exec_driver_sql(
                    "SELECT approved_axis_id, approved_axis_version "
                    "FROM character_trait_candidates WHERE id = 'candidate-legacy'"
                ).one()
            self.assertEqual(still_legacy, (None, None))
            self.assertEqual(candidate.candidate_fingerprint, "f" * 64)
            self.assertEqual(frozen.payload, legacy_payload)
            self.assertEqual(frozen.payload_sha256, legacy_hash)
            engine.dispose()

            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            config.attributes["database_url"] = url
            command.downgrade(config, "0013_account_model_provider")
            engine = create_engine(url)
            self.assertEqual(
                {
                    item["name"]
                    for item in inspect(engine).get_columns("character_trait_candidates")
                },
                before_columns,
            )
            engine.dispose()

    def test_run_idempotency_indexes_compile_for_sqlite_and_postgresql(self):
        table = Base.metadata.tables["analysis_runs"]
        indexes = {
            item.name: item
            for item in table.indexes
            if item.name
            in {
                "uq_analysis_runs_project_id_idempotency_key",
                "ix_analysis_runs_project_created_id",
            }
        }
        self.assertEqual(2, len(indexes))
        for dialect in (sqlite.dialect(), postgresql.dialect()):
            for index in indexes.values():
                sql = str(CreateIndex(index).compile(dialect=dialect))
                self.assertIn("analysis_runs", sql)
                self.assertIn(index.name, sql)
        self.assertTrue(
            indexes["uq_analysis_runs_project_id_idempotency_key"].unique
        )

    def test_revision_identifiers_fit_default_alembic_version_column(self):
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        revisions = list(ScriptDirectory.from_config(config).walk_revisions())
        self.assertTrue(revisions)
        self.assertEqual(len(revisions), len({item.revision for item in revisions}))
        for item in revisions:
            with self.subTest(revision=item.revision):
                self.assertLessEqual(len(item.revision), 32)

    def upgrade_to(self, database_url: str, revision: str) -> None:
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        config.attributes["database_url"] = database_url
        command.upgrade(config, revision)

    def upgrade(self, database_url: str) -> None:
        self.upgrade_to(database_url, "head")

    def stamp(self, database_url: str, revision: str) -> None:
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        config.attributes["database_url"] = database_url
        command.stamp(config, revision)

    def test_document_concurrency_downgrade_preserves_0007_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison-downgrade.db"
            url = f"sqlite:///{path.as_posix()}"
            self.upgrade(url)
            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            config.attributes["database_url"] = url
            command.downgrade(config, "0007_revision_comparisons")
            engine = create_engine(url)
            try:
                columns = {
                    row["name"]
                    for row in inspect(engine).get_columns(
                        "analysis_run_comparisons"
                    )
                }
                self.assertIn("provenance", columns)
            finally:
                engine.dispose()

    def create_prior_wip_embedding_tables(
        self,
        engine,
        *,
        vector_type: str = "JSON",
        embedding_foreign_keys: bool = True,
        simple_chunk_foreign_keys: bool = True,
        reversed_owner_foreign_key: bool = False,
    ) -> None:
        chunk_foreign_keys = []
        if simple_chunk_foreign_keys:
            chunk_foreign_keys.extend(
                [
                    "FOREIGN KEY (project_id) REFERENCES projects(id)",
                    "FOREIGN KEY (document_id) REFERENCES documents(id)",
                ]
            )
        if reversed_owner_foreign_key:
            chunk_foreign_keys.append(
                "FOREIGN KEY (document_id, project_id) "
                "REFERENCES documents(project_id, id)"
            )
        chunk_constraints = ",\n                    ".join(chunk_foreign_keys)
        if chunk_constraints:
            chunk_constraints = ",\n                    " + chunk_constraints

        embedding_constraints = ""
        if embedding_foreign_keys:
            embedding_constraints = (
                ",\n                    FOREIGN KEY (chunk_id) "
                "REFERENCES evidence_chunks(id) ON DELETE CASCADE"
                ",\n                    FOREIGN KEY (profile_id) "
                "REFERENCES embedding_profiles(id) ON DELETE CASCADE"
            )

        with engine.begin() as connection:
            connection.exec_driver_sql(
                """
                CREATE TABLE embedding_profiles (
                    id VARCHAR(68) NOT NULL PRIMARY KEY,
                    provider_kind VARCHAR(40) NOT NULL,
                    provider_namespace VARCHAR(80) NOT NULL,
                    model_identifier VARCHAR(255) NOT NULL,
                    model_revision VARCHAR(120) NOT NULL,
                    dimensions INTEGER NOT NULL,
                    normalized BOOLEAN NOT NULL,
                    created_at DATETIME NOT NULL,
                    CONSTRAINT uq_embedding_profile_identity UNIQUE (
                        provider_kind, provider_namespace, model_identifier,
                        model_revision, dimensions, normalized
                    )
                )
                """
            )
            connection.exec_driver_sql(
                f"""
                CREATE TABLE evidence_chunks (
                    id VARCHAR(68) NOT NULL PRIMARY KEY,
                    project_id VARCHAR(36) NOT NULL,
                    document_id VARCHAR(36) NOT NULL,
                    document_version INTEGER NOT NULL,
                    content_sha256 VARCHAR(64) NOT NULL,
                    chunker_version VARCHAR(80) NOT NULL,
                    ordinal INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    text_sha256 VARCHAR(64) NOT NULL,
                    char_start INTEGER NOT NULL,
                    char_end INTEGER NOT NULL,
                    line_start INTEGER NOT NULL,
                    line_end INTEGER NOT NULL,
                    created_at DATETIME NOT NULL,
                    CONSTRAINT uq_evidence_chunk_snapshot_ordinal UNIQUE (
                        project_id, document_id, document_version,
                        content_sha256, chunker_version, ordinal
                    )
                    {chunk_constraints}
                )
                """
            )
            connection.exec_driver_sql(
                f"""
                CREATE TABLE evidence_embeddings (
                    chunk_id VARCHAR(68) NOT NULL,
                    profile_id VARCHAR(68) NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector {vector_type} NOT NULL,
                    created_at DATETIME NOT NULL,
                    PRIMARY KEY (chunk_id, profile_id)
                    {embedding_constraints}
                )
                """
            )

    def insert_snapshot_fixture(
        self,
        engine,
        *,
        chunk_project_id: str = "p1",
        include_product_rows: bool = True,
    ) -> None:
        with engine.begin() as connection:
            connection.exec_driver_sql(
                "INSERT INTO projects (id, name, description, created_at) "
                "VALUES ('p1', 'one', '', '2026-09-07'), "
                "('p2', 'two', '', '2026-09-07')"
            )
            connection.exec_driver_sql(
                "INSERT INTO documents "
                "(id, project_id, name, content, version, active, created_at) "
                "VALUES ('doc1', 'p1', 'one.md', 'alpha', 1, 1, '2026-09-07')"
            )
            if include_product_rows:
                connection.exec_driver_sql(
                    "INSERT INTO analysis_runs "
                    "(id, project_id, status, created_at, started_at, completed_at, "
                    "input_chars, prompt_tokens, completion_tokens, estimated_cost_usd, "
                    "error, cancel_requested) VALUES "
                    "('run1', 'p1', 'completed', '2026-09-07', NULL, NULL, "
                    "5, 0, 0, 0, NULL, 0)"
                )
                connection.exec_driver_sql(
                    "INSERT INTO issues "
                    "(id, run_id, category, severity, confidence, title, explanation, "
                    "evidence, suggestion, extra) VALUES "
                    "('issue1', 'run1', 'fact_conflict', 'medium', 0.9, 'keep', "
                    "'keep explanation', '[]', 'keep suggestion', '{}')"
                )
                connection.exec_driver_sql(
                    "INSERT INTO issue_feedback "
                    "(id, issue_id, label, comment, created_at) "
                    "VALUES ('feedback1', 'issue1', 'accepted', 'keep feedback', "
                    "'2026-09-07')"
                )
            connection.exec_driver_sql(
                "INSERT INTO embedding_profiles "
                "(id, provider_kind, provider_namespace, model_identifier, "
                "model_revision, dimensions, normalized, created_at) "
                "VALUES ('emb-wip', 'openai-compatible', 'test', 'model', "
                "'r1', 2, 1, '2026-09-07')"
            )
            connection.exec_driver_sql(
                "INSERT INTO evidence_chunks "
                "(id, project_id, document_id, document_version, content_sha256, "
                "chunker_version, ordinal, text, text_sha256, char_start, char_end, "
                "line_start, line_end, created_at) VALUES "
                "('chk-wip', ?, 'doc1', 1, ?, 'wip-v1', 0, 'alpha', ?, "
                "0, 5, 1, 1, '2026-09-07')",
                (chunk_project_id, "1" * 64, "2" * 64),
            )
            connection.exec_driver_sql(
                "INSERT INTO evidence_embeddings "
                "(chunk_id, profile_id, dimensions, vector, created_at) "
                "VALUES ('chk-wip', 'emb-wip', 2, '[1.0, 0.0]', '2026-09-07')"
            )

    def test_empty_sqlite_database_upgrades_to_full_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fresh.db"
            url = f"sqlite:///{path.as_posix()}"
            self.upgrade(url)
            engine = create_engine(url)
            tables = set(inspect(engine).get_table_names())
            self.assertTrue(set(Base.metadata.tables) <= tables)
            for table_name in EMBEDDING_TABLES:
                migrated_columns = {
                    item["name"] for item in inspect(engine).get_columns(table_name)
                }
                self.assertEqual(
                    migrated_columns,
                    {column.name for column in Base.metadata.tables[table_name].columns},
                )
            vector_type = next(
                item["type"]
                for item in inspect(engine).get_columns("evidence_embeddings")
                if item["name"] == "vector"
            )
            self.assertEqual(vector_type.__class__.__name__.upper(), "JSON")
            owner_fks = inspect(engine).get_foreign_keys("evidence_chunks")
            self.assertTrue(
                any(
                    tuple(item.get("constrained_columns") or ())
                    == ("project_id", "document_id")
                    and tuple(item.get("referred_columns") or ())
                    == ("project_id", "id")
                    for item in owner_fks
                )
            )
            with engine.connect() as connection:
                revision = connection.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar_one()
            self.assertEqual(revision, HEAD_REVISION)
            engine.dispose()

    def test_workspace_isolation_migration_round_trips_on_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workspace-round-trip.db"
            url = f"sqlite:///{path.as_posix()}"
            self.upgrade(url)
            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            config.attributes["database_url"] = url
            command.downgrade(config, "0004_auth_foundation")
            engine = create_engine(url)
            inspector = inspect(engine)
            self.assertNotIn(
                "workspace_id",
                {column["name"] for column in inspector.get_columns("projects")},
            )
            self.assertNotIn(
                "requested_by_user_id",
                {
                    column["name"]
                    for column in inspector.get_columns("analysis_runs")
                },
            )
            engine.dispose()

            self.upgrade(url)
            engine = create_engine(url)
            inspector = inspect(engine)
            self.assertIn(
                "workspace_id",
                {column["name"] for column in inspector.get_columns("projects")},
            )
            with engine.connect() as connection:
                self.assertEqual(
                    HEAD_REVISION,
                    connection.exec_driver_sql(
                        "SELECT version_num FROM alembic_version"
                    ).scalar_one(),
                )
            engine.dispose()

    def test_run_idempotency_migration_adds_nullable_column_and_indexes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run-idempotency.db"
            url = f"sqlite:///{path.as_posix()}"
            self.upgrade_to(url, "0005_workspace_isolation")
            engine = create_engine(url)
            inspector = inspect(engine)
            self.assertNotIn(
                "idempotency_key",
                {item["name"] for item in inspector.get_columns("analysis_runs")},
            )
            engine.dispose()

            self.upgrade(url)
            engine = create_engine(url)
            inspector = inspect(engine)
            columns = {
                item["name"]: item
                for item in inspector.get_columns("analysis_runs")
            }
            self.assertIn("idempotency_key", columns)
            self.assertTrue(columns["idempotency_key"]["nullable"])
            indexes = {
                item["name"]: item
                for item in inspector.get_indexes("analysis_runs")
            }
            self.assertEqual(
                ["project_id", "idempotency_key"],
                indexes["uq_analysis_runs_project_id_idempotency_key"][
                    "column_names"
                ],
            )
            self.assertTrue(
                indexes["uq_analysis_runs_project_id_idempotency_key"]["unique"]
            )
            self.assertEqual(
                ["project_id", "created_at", "id"],
                indexes["ix_analysis_runs_project_created_id"]["column_names"],
            )
            engine.dispose()

            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            config.attributes["database_url"] = url
            command.downgrade(config, "0005_workspace_isolation")
            engine = create_engine(url)
            inspector = inspect(engine)
            self.assertNotIn(
                "idempotency_key",
                {item["name"] for item in inspector.get_columns("analysis_runs")},
            )
            index_names = {
                item["name"] for item in inspector.get_indexes("analysis_runs")
            }
            self.assertNotIn(
                "uq_analysis_runs_project_id_idempotency_key", index_names
            )
            self.assertNotIn("ix_analysis_runs_project_created_id", index_names)
            engine.dispose()

    def test_incomplete_legacy_table_stops_before_head_stamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete.db"
            url = f"sqlite:///{path.as_posix()}"
            engine = create_engine(url)
            with engine.begin() as connection:
                connection.exec_driver_sql("CREATE TABLE projects (id TEXT)")
            engine.dispose()
            with self.assertRaisesRegex(RuntimeError, "projects"):
                self.upgrade(url)
            engine = create_engine(url)
            with engine.connect() as connection:
                revision = connection.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar_one_or_none()
            self.assertNotEqual(revision, HEAD_REVISION)
            engine.dispose()

    def test_incomplete_embedding_table_stops_at_baseline_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "incomplete-embedding.db"
            url = f"sqlite:///{path.as_posix()}"
            config = Config(str(ROOT / "alembic.ini"))
            config.set_main_option("script_location", str(ROOT / "migrations"))
            config.attributes["database_url"] = url
            command.upgrade(config, "0001_legacy_schema_baseline")
            engine = create_engine(url)
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "CREATE TABLE embedding_profiles (id VARCHAR(68) PRIMARY KEY)"
                )
            engine.dispose()
            with self.assertRaisesRegex(RuntimeError, "embedding_profiles"):
                self.upgrade(url)
            engine = create_engine(url)
            with engine.connect() as connection:
                revision = connection.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar_one()
            self.assertEqual(revision, "0001_legacy_schema_baseline")
            engine.dispose()

    def test_profile_identity_upgrade_invalidates_only_derived_vector_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior-wip.db"
            url = f"sqlite:///{path.as_posix()}"
            self.upgrade_to(url, "0001_legacy_schema_baseline")
            engine = create_engine(url)
            self.create_prior_wip_embedding_tables(engine)
            self.insert_snapshot_fixture(engine)
            engine.dispose()

            self.upgrade(url)
            engine = create_engine(url)
            owner_fks = inspect(engine).get_foreign_keys("evidence_chunks")
            self.assertTrue(
                any(
                    tuple(item.get("constrained_columns") or ())
                    == ("project_id", "document_id")
                    and item.get("referred_table") == "documents"
                    and tuple(item.get("referred_columns") or ())
                    == ("project_id", "id")
                    for item in owner_fks
                )
            )
            with engine.connect() as connection:
                preserved_chunk = connection.exec_driver_sql(
                    "SELECT text FROM evidence_chunks WHERE id = 'chk-wip'"
                ).scalar_one()
                embedding_count = connection.exec_driver_sql(
                    "SELECT count(*) FROM evidence_embeddings"
                ).scalar_one()
                profile_count = connection.exec_driver_sql(
                    "SELECT count(*) FROM embedding_profiles"
                ).scalar_one()
                document_content = connection.exec_driver_sql(
                    "SELECT content FROM documents WHERE id = 'doc1'"
                ).scalar_one()
                run_status = connection.exec_driver_sql(
                    "SELECT status FROM analysis_runs WHERE id = 'run1'"
                ).scalar_one()
                issue_title = connection.exec_driver_sql(
                    "SELECT title FROM issues WHERE id = 'issue1'"
                ).scalar_one()
                feedback_comment = connection.exec_driver_sql(
                    "SELECT comment FROM issue_feedback WHERE id = 'feedback1'"
                ).scalar_one()
                revision = connection.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar_one()
            self.assertEqual(preserved_chunk, "alpha")
            self.assertEqual(embedding_count, 0)
            self.assertEqual(profile_count, 0)
            self.assertEqual(document_content, "alpha")
            self.assertEqual(run_status, "completed")
            self.assertEqual(issue_title, "keep")
            self.assertEqual(feedback_comment, "keep feedback")
            self.assertEqual(revision, HEAD_REVISION)
            engine.dispose()

    def test_applied_0002_upgrades_to_stronger_profile_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "applied-0002.db"
            url = f"sqlite:///{path.as_posix()}"
            self.upgrade_to(url, "0002_evidence_substrate")
            engine = create_engine(url)
            self.insert_snapshot_fixture(engine)
            engine.dispose()

            self.upgrade(url)
            engine = create_engine(url)
            columns = {
                item["name"]
                for item in inspect(engine).get_columns("embedding_profiles")
            }
            self.assertTrue(
                {
                    "deployment_fingerprint",
                    "document_transform_identity",
                    "query_transform_identity",
                }
                <= columns
            )
            with engine.connect() as connection:
                self.assertEqual(
                    connection.exec_driver_sql(
                        "SELECT count(*) FROM embedding_profiles"
                    ).scalar_one(),
                    0,
                )
                self.assertEqual(
                    connection.exec_driver_sql(
                        "SELECT count(*) FROM evidence_embeddings"
                    ).scalar_one(),
                    0,
                )
                self.assertEqual(
                    connection.exec_driver_sql(
                        "SELECT content FROM documents WHERE id = 'doc1'"
                    ).scalar_one(),
                    "alpha",
                )
                self.assertEqual(
                    connection.exec_driver_sql(
                        "SELECT title FROM issues WHERE id = 'issue1'"
                    ).scalar_one(),
                    "keep",
                )
            engine.dispose()

    def test_stamped_prior_wip_adds_missing_document_owner_unique(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior-wip-no-document-unique.db"
            url = f"sqlite:///{path.as_posix()}"
            engine = create_engine(url)
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "CREATE TABLE projects ("
                    "id VARCHAR(36) PRIMARY KEY, name VARCHAR(200) NOT NULL, "
                    "description TEXT NOT NULL, created_at DATETIME NOT NULL)"
                )
                connection.exec_driver_sql(
                    "CREATE TABLE documents ("
                    "id VARCHAR(36) PRIMARY KEY, project_id VARCHAR(36) NOT NULL, "
                    "name VARCHAR(255) NOT NULL, content TEXT NOT NULL, "
                    "version INTEGER NOT NULL, active BOOLEAN NOT NULL, "
                    "created_at DATETIME NOT NULL, "
                    "FOREIGN KEY (project_id) REFERENCES projects(id))"
                )
            self.create_prior_wip_embedding_tables(engine)
            self.insert_snapshot_fixture(engine, include_product_rows=False)
            engine.dispose()
            self.stamp(url, "0001_legacy_schema_baseline")

            self.upgrade(url)
            engine = create_engine(url)
            inspector = inspect(engine)
            owner_fks = inspector.get_foreign_keys("evidence_chunks")
            self.assertTrue(
                any(
                    tuple(item.get("constrained_columns") or ())
                    == ("project_id", "document_id")
                    and tuple(item.get("referred_columns") or ())
                    == ("project_id", "id")
                    for item in owner_fks
                )
            )
            document_uniques = {
                tuple(item.get("column_names") or ())
                for item in inspector.get_indexes("documents")
                if item.get("unique")
            }
            document_uniques.update(
                tuple(item.get("column_names") or ())
                for item in inspector.get_unique_constraints("documents")
            )
            self.assertIn(("project_id", "id"), document_uniques)
            with engine.connect() as connection:
                self.assertEqual(
                    connection.exec_driver_sql(
                        "SELECT text FROM evidence_chunks WHERE id = 'chk-wip'"
                    ).scalar_one(),
                    "alpha",
                )
            engine.dispose()

    def test_prior_wip_cross_project_chunk_fails_without_head_stamp(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dirty-prior-wip.db"
            url = f"sqlite:///{path.as_posix()}"
            self.upgrade_to(url, "0001_legacy_schema_baseline")
            engine = create_engine(url)
            self.create_prior_wip_embedding_tables(engine)
            self.insert_snapshot_fixture(engine, chunk_project_id="p2")
            engine.dispose()

            with self.assertRaisesRegex(RuntimeError, "document ownership"):
                self.upgrade(url)
            engine = create_engine(url)
            with engine.connect() as connection:
                revision = connection.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar_one()
                owner = connection.exec_driver_sql(
                    "SELECT project_id, document_id FROM evidence_chunks "
                    "WHERE id = 'chk-wip'"
                ).one()
            self.assertEqual(revision, "0001_legacy_schema_baseline")
            self.assertEqual(tuple(owner), ("p2", "doc1"))
            engine.dispose()

    def test_baseline_rejects_analysis_inputs_without_uniques(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs-no-unique.db"
            url = f"sqlite:///{path.as_posix()}"
            engine = create_engine(url)
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "CREATE TABLE analysis_run_inputs ("
                    "id VARCHAR(36) PRIMARY KEY, run_id VARCHAR(36) NOT NULL, "
                    "document_id VARCHAR(36) NOT NULL, document_name VARCHAR(255) NOT NULL, "
                    "document_version INTEGER NOT NULL, content TEXT NOT NULL, "
                    "content_sha256 VARCHAR(64) NOT NULL, ordinal INTEGER NOT NULL)"
                )
            engine.dispose()
            with self.assertRaisesRegex(RuntimeError, "analysis_run_inputs"):
                self.upgrade(url)

    def test_baseline_rejects_analysis_inputs_without_run_foreign_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "inputs-no-fk.db"
            url = f"sqlite:///{path.as_posix()}"
            engine = create_engine(url)
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "CREATE TABLE analysis_run_inputs ("
                    "id VARCHAR(36) PRIMARY KEY, run_id VARCHAR(36) NOT NULL, "
                    "document_id VARCHAR(36) NOT NULL, document_name VARCHAR(255) NOT NULL, "
                    "document_version INTEGER NOT NULL, content TEXT NOT NULL, "
                    "content_sha256 VARCHAR(64) NOT NULL, ordinal INTEGER NOT NULL, "
                    "UNIQUE (run_id, ordinal), UNIQUE (run_id, document_id))"
                )
            engine.dispose()
            with self.assertRaisesRegex(RuntimeError, "analysis_run_inputs"):
                self.upgrade(url)

    def test_sqlite_adoption_rejects_non_json_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-vector-type.db"
            url = f"sqlite:///{path.as_posix()}"
            self.upgrade_to(url, "0001_legacy_schema_baseline")
            engine = create_engine(url)
            self.create_prior_wip_embedding_tables(engine, vector_type="TEXT")
            engine.dispose()
            with self.assertRaisesRegex(RuntimeError, "evidence_embeddings"):
                self.upgrade(url)

    def test_sqlite_adoption_rejects_missing_embedding_foreign_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-embedding-fks.db"
            url = f"sqlite:///{path.as_posix()}"
            self.upgrade_to(url, "0001_legacy_schema_baseline")
            engine = create_engine(url)
            self.create_prior_wip_embedding_tables(
                engine, embedding_foreign_keys=False
            )
            engine.dispose()
            with self.assertRaisesRegex(RuntimeError, "evidence_embeddings"):
                self.upgrade(url)

    def test_reversed_owner_mapping_is_not_accepted_as_owner_fk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reversed-owner.db"
            url = f"sqlite:///{path.as_posix()}"
            self.upgrade_to(url, "0001_legacy_schema_baseline")
            engine = create_engine(url)
            self.create_prior_wip_embedding_tables(
                engine,
                simple_chunk_foreign_keys=False,
                reversed_owner_foreign_key=True,
            )
            engine.dispose()
            with self.assertRaisesRegex(RuntimeError, "evidence_chunks"):
                self.upgrade(url)

    def test_existing_create_all_database_is_adopted_without_data_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.db"
            url = f"sqlite:///{path.as_posix()}"
            engine = create_engine(url)
            legacy = MetaData()
            for table in Base.metadata.sorted_tables:
                if table.name not in EMBEDDING_TABLES:
                    table.to_metadata(legacy)
            legacy.create_all(engine)
            with engine.begin() as connection:
                connection.execute(
                    legacy.tables["projects"].insert().values(
                        id="preserved",
                        name="existing project",
                        description="keep",
                        created_at=datetime(2026, 9, 6),
                    )
                )
            engine.dispose()

            self.upgrade(url)
            engine = create_engine(url)
            with engine.connect() as connection:
                row = connection.exec_driver_sql(
                    "SELECT id, name, description FROM projects WHERE id = 'preserved'"
                ).one()
            self.assertEqual(tuple(row), ("preserved", "existing project", "keep"))
            self.assertTrue(EMBEDDING_TABLES <= set(inspect(engine).get_table_names()))
            engine.dispose()

    def test_current_create_all_database_can_be_stamped_by_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "current.db"
            url = f"sqlite:///{path.as_posix()}"
            engine = create_engine(url)
            Base.metadata.create_all(engine)
            engine.dispose()
            self.upgrade(url)
            engine = create_engine(url)
            self.assertTrue(EMBEDDING_TABLES <= set(inspect(engine).get_table_names()))
            engine.dispose()

    def test_product_init_db_migrates_existing_sqlite_and_preserves_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "startup.db"
            url = f"sqlite:///{path.as_posix()}"
            engine = create_engine(url)
            legacy = MetaData()
            for table in Base.metadata.sorted_tables:
                if table.name not in EMBEDDING_TABLES:
                    table.to_metadata(legacy)
            legacy.create_all(engine)
            with engine.begin() as connection:
                connection.execute(
                    legacy.tables["projects"].insert().values(
                        id="startup-preserved",
                        name="startup",
                        description="keep",
                        created_at=datetime(2026, 9, 6),
                    )
                )
            engine.dispose()
            with patch.object(app_db.settings, "database_url", url):
                app_db.init_db()
            engine = create_engine(url)
            with engine.connect() as connection:
                name = connection.exec_driver_sql(
                    "SELECT name FROM projects WHERE id = 'startup-preserved'"
                ).scalar_one()
                revision = connection.exec_driver_sql(
                    "SELECT version_num FROM alembic_version"
                ).scalar_one()
            self.assertEqual(name, "startup")
            self.assertEqual(revision, HEAD_REVISION)
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
