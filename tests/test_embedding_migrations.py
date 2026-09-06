from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import MetaData, create_engine, inspect

from app.db import Base
from app import db as app_db


ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_TABLES = {"embedding_profiles", "evidence_chunks", "evidence_embeddings"}
HEAD_REVISION = "0003_embedding_identity"


class EmbeddingMigrationTests(unittest.TestCase):
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
