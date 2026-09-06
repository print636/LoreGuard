from __future__ import annotations

import os
import unittest
import uuid
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, delete, inspect
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker

from app.db import (
    DocumentRow,
    EmbeddingProfileRow,
    EvidenceChunkRow,
    EvidenceEmbeddingRow,
    ProjectRow,
)
from app.embeddings import EmbeddingProfile
from app.evidence_chunks import EvidenceChunker, SnapshotDocumentKey
from app.evidence_store import (
    SqlAlchemyEvidenceEmbeddingIndex,
    embedding_coverage,
    store_embeddings,
)


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.environ.get("LOREGUARD_TEST_POSTGRES_URL"),
    "requires an explicitly disposable PostgreSQL/pgvector test database",
)
class PgvectorIntegrationTests(unittest.TestCase):
    def isolated_schema_url(self, base_url: str, schema: str) -> str:
        url = make_url(base_url)
        return url.update_query_dict(
            {"options": f"-csearch_path={schema},public"}
        ).render_as_string(hide_password=False)

    def upgrade_url(self, url: str, revision: str = "head") -> None:
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        config.attributes["database_url"] = url
        command.upgrade(config, revision)

    def test_real_vector_column_and_cosine_distance(self):
        url = os.environ["LOREGUARD_TEST_POSTGRES_URL"]
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        config.attributes["database_url"] = url
        command.upgrade(config, "head")
        engine = create_engine(url)
        with engine.begin() as connection:
            data_type = connection.exec_driver_sql(
                "SELECT udt_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'evidence_embeddings' AND column_name = 'vector'"
            ).scalar_one()
            self.assertEqual(data_type, "vector")
            distance = connection.exec_driver_sql(
                "SELECT '[1,0]'::vector <=> '[0,1]'::vector"
            ).scalar_one()
            self.assertAlmostEqual(float(distance), 1.0)
        engine.dispose()

    def test_exact_cosine_nearest_isolates_snapshot_profile_and_chunker(self):
        url = os.environ["LOREGUARD_TEST_POSTGRES_URL"]
        config = Config(str(ROOT / "alembic.ini"))
        config.set_main_option("script_location", str(ROOT / "migrations"))
        config.attributes["database_url"] = url
        command.upgrade(config, "head")
        engine = create_engine(url)
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        suffix = uuid.uuid4().hex[:12]
        project_id = f"pv-{suffix}"
        document_id = f"doc-{suffix}"
        main_chunker = EvidenceChunker(
            target_chars=6, min_chars=3, max_chars=8, overlap_chars=0
        )
        other_chunker = EvidenceChunker(
            target_chars=7, min_chars=3, max_chars=9, overlap_chars=0
        )
        content = "白塔事实。黑塔事实。"
        chunks = main_chunker.chunk(
            project_id=project_id,
            document_id=document_id,
            document_version=1,
            content=content,
        )
        other_chunks = other_chunker.chunk(
            project_id=project_id,
            document_id=document_id,
            document_version=1,
            content=content,
        )
        profile = EmbeddingProfile.openai_compatible(
            provider_namespace=f"pg-{suffix}",
            model_identifier="mock",
            model_revision="r1",
            deployment_fingerprint="pg-runtime-v1",
            dimensions=2,
        )
        other_profile = EmbeddingProfile.openai_compatible(
            provider_namespace=f"pg-other-{suffix}",
            model_identifier="mock",
            model_revision="r1",
            deployment_fingerprint="pg-runtime-v1",
            dimensions=2,
        )
        try:
            with SessionLocal() as session:
                session.add(ProjectRow(id=project_id, name="pgvector", description=""))
                session.add(
                    DocumentRow(
                        id=document_id,
                        project_id=project_id,
                        name="story.md",
                        content=content,
                        version=1,
                    )
                )
                session.commit()
                store_embeddings(
                    session,
                    profile=profile,
                    chunks=chunks,
                    vectors=[(1.0, 0.0), (0.0, 1.0)],
                )
                store_embeddings(
                    session,
                    profile=profile,
                    chunks=other_chunks,
                    vectors=[(0.6, 0.8), (1.0, 0.0)],
                )
                store_embeddings(
                    session,
                    profile=other_profile,
                    chunks=chunks,
                    vectors=[(0.0, 1.0), (1.0, 0.0)],
                )
                session.commit()
                self.assertTrue(
                    embedding_coverage(
                        session, profile=profile, chunks=chunks
                    ).complete
                )

            index = SqlAlchemyEvidenceEmbeddingIndex(SessionLocal)
            matches = index.nearest(
                snapshots=[chunks[0].snapshot],
                profile_id=profile.profile_id,
                chunker_version=main_chunker.version,
                query_vector=(1.0, 0.0),
                limit=10,
            )
            self.assertEqual(tuple(match.chunk for match in matches), chunks)
            self.assertAlmostEqual(matches[0].score, 1.0)
            self.assertAlmostEqual(matches[1].score, 0.0)
            self.assertEqual(
                index.nearest(
                    snapshots=[
                        SnapshotDocumentKey(project_id, document_id, 1, "0" * 64)
                    ],
                    profile_id=profile.profile_id,
                    chunker_version=main_chunker.version,
                    query_vector=(1.0, 0.0),
                    limit=10,
                ),
                (),
            )
        finally:
            with Session(engine) as session:
                session.execute(
                    delete(EvidenceEmbeddingRow).where(
                        EvidenceEmbeddingRow.profile_id.in_(
                            (profile.profile_id, other_profile.profile_id)
                        )
                    )
                )
                session.execute(
                    delete(EvidenceChunkRow).where(
                        EvidenceChunkRow.project_id == project_id
                    )
                )
                session.execute(
                    delete(EmbeddingProfileRow).where(
                        EmbeddingProfileRow.id.in_(
                            (profile.profile_id, other_profile.profile_id)
                        )
                    )
                )
                session.execute(
                    delete(DocumentRow).where(
                        DocumentRow.project_id == project_id,
                        DocumentRow.id == document_id,
                    )
                )
                session.execute(delete(ProjectRow).where(ProjectRow.id == project_id))
                session.commit()
            engine.dispose()

    def test_fresh_postgresql_schema_upgrades_to_stronger_identity_head(self):
        base_url = os.environ["LOREGUARD_TEST_POSTGRES_URL"]
        schema = f"loreguard_fresh_{uuid.uuid4().hex[:12]}"
        admin = create_engine(base_url)
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        url = self.isolated_schema_url(base_url, schema)
        try:
            self.upgrade_url(url)
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
                        "SELECT version_num FROM alembic_version"
                    ).scalar_one(),
                    "0003_embedding_identity",
                )
            engine.dispose()
        finally:
            with admin.begin() as connection:
                connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
            admin.dispose()

    def test_applied_0002_postgresql_upgrade_preserves_sources_and_drops_cache(self):
        base_url = os.environ["LOREGUARD_TEST_POSTGRES_URL"]
        schema = f"loreguard_upgrade_{uuid.uuid4().hex[:12]}"
        admin = create_engine(base_url)
        with admin.begin() as connection:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
        url = self.isolated_schema_url(base_url, schema)
        try:
            self.upgrade_url(url, "0002_evidence_substrate")
            engine = create_engine(url)
            with engine.begin() as connection:
                connection.exec_driver_sql(
                    "INSERT INTO projects (id, name, description, created_at) "
                    "VALUES ('p', 'keep', '', now())"
                )
                connection.exec_driver_sql(
                    "INSERT INTO documents "
                    "(id, project_id, name, content, version, active, created_at) "
                    "VALUES ('d', 'p', 'keep.md', 'keep source', 1, true, now())"
                )
                connection.exec_driver_sql(
                    "INSERT INTO embedding_profiles "
                    "(id, provider_kind, provider_namespace, model_identifier, "
                    "model_revision, dimensions, normalized, created_at) VALUES "
                    "('emb-old', 'openai-compatible', 'old', 'old', 'r1', "
                    "2, true, now())"
                )
                connection.exec_driver_sql(
                    "INSERT INTO evidence_chunks "
                    "(id, project_id, document_id, document_version, content_sha256, "
                    "chunker_version, ordinal, text, text_sha256, char_start, char_end, "
                    "line_start, line_end, created_at) VALUES "
                    "('chk-old', 'p', 'd', 1, repeat('1', 64), 'old-v1', 0, "
                    "'keep chunk', repeat('2', 64), 0, 10, 1, 1, now())"
                )
                connection.exec_driver_sql(
                    "INSERT INTO evidence_embeddings "
                    "(chunk_id, profile_id, dimensions, vector, created_at) VALUES "
                    "('chk-old', 'emb-old', 2, '[1,0]'::vector, now())"
                )
            engine.dispose()

            self.upgrade_url(url)
            engine = create_engine(url)
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
                        "SELECT content FROM documents WHERE id = 'd'"
                    ).scalar_one(),
                    "keep source",
                )
                self.assertEqual(
                    connection.exec_driver_sql(
                        "SELECT text FROM evidence_chunks WHERE id = 'chk-old'"
                    ).scalar_one(),
                    "keep chunk",
                )
                self.assertEqual(
                    connection.exec_driver_sql(
                        "SELECT version_num FROM alembic_version"
                    ).scalar_one(),
                    "0003_embedding_identity",
                )
            engine.dispose()
        finally:
            with admin.begin() as connection:
                connection.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
            admin.dispose()


if __name__ == "__main__":
    unittest.main()
