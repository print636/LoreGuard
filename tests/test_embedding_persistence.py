from __future__ import annotations

import unittest
from dataclasses import replace

from sqlalchemy import create_engine, func, inspect, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db import (
    Base,
    DocumentRow,
    EmbeddingProfileRow,
    EvidenceChunkRow,
    EvidenceEmbeddingRow,
    ProjectRow,
    enable_sqlite_foreign_keys,
)
from app.embeddings import EmbeddingProfile
from app.evidence_chunks import EvidenceChunk, EvidenceChunker, SnapshotDocumentKey
from app.evidence_store import (
    EvidenceStoreError,
    list_chunks_for_snapshots,
    store_chunks,
    store_embeddings,
)


class EvidencePersistenceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite+pysqlite:///:memory:")
        enable_sqlite_foreign_keys(self.engine)
        Base.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.session.add(ProjectRow(id="p", name="test", description=""))
        self.session.add(
            DocumentRow(
                id="d", project_id="p", name="story.md", content="initial", version=1
            )
        )
        self.session.commit()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()

    def test_sqlite_uses_explicit_json_vector_fallback(self):
        vector_column = inspect(self.engine).get_columns("evidence_embeddings")
        column_type = next(item["type"] for item in vector_column if item["name"] == "vector")
        self.assertEqual(column_type.__class__.__name__.upper(), "JSON")

    def test_store_is_idempotent_and_snapshot_filter_is_exact(self):
        chunker = EvidenceChunker(target_chars=12, min_chars=5, max_chars=20, overlap_chars=3)
        v1 = chunker.chunk(
            project_id="p", document_id="d", document_version=1, content="第一版事实。角色在白塔。"
        )
        v2 = chunker.chunk(
            project_id="p", document_id="d", document_version=2, content="第二版事实。角色在黑塔。"
        )
        profile = EmbeddingProfile.openai_compatible(
            model_identifier="test-embedding", model_revision="r1", dimensions=2
        )
        store_embeddings(
            self.session,
            profile=profile,
            chunks=v1,
            vectors=[(1.0, 0.0) for _ in v1],
        )
        store_embeddings(
            self.session,
            profile=profile,
            chunks=v2,
            vectors=[(0.0, 1.0) for _ in v2],
        )
        self.session.commit()

        selected = list_chunks_for_snapshots(self.session, [v1[0].snapshot])
        self.assertEqual(selected, v1)
        wrong_hash = SnapshotDocumentKey("p", "d", 1, "0" * 64)
        self.assertEqual(list_chunks_for_snapshots(self.session, [wrong_hash]), ())
        wrong_project = SnapshotDocumentKey(
            "other", "d", 1, v1[0].snapshot.content_sha256
        )
        self.assertEqual(list_chunks_for_snapshots(self.session, [wrong_project]), ())

        # A second identical write is a no-op rather than a duplicate.
        store_embeddings(
            self.session,
            profile=profile,
            chunks=v1,
            vectors=[(1.0, 0.0) for _ in v1],
        )
        self.session.commit()

    def test_identity_collision_and_invalid_vectors_fail_closed(self):
        chunk = EvidenceChunker().chunk(
            project_id="p", document_id="d", document_version=1, content="有效事实。"
        )[0]
        store_chunks(self.session, [chunk])
        self.session.commit()
        forged = replace(chunk)
        object.__setattr__(forged, "text", "不同内容")
        with self.assertRaisesRegex(EvidenceStoreError, "chunk is invalid"):
            store_chunks(self.session, [forged])

        profile = EmbeddingProfile.openai_compatible(
            model_identifier="test", model_revision="r1", dimensions=2
        )
        for vector in (
            (1.0,),
            (0.0, 0.0),
            (2.0, 0.0),
            (float("inf"), 1.0),
            (True, 1.0),
        ):
            with self.subTest(vector=vector), self.assertRaises(EvidenceStoreError):
                store_embeddings(
                    self.session, profile=profile, chunks=[chunk], vectors=[vector]
                )
            self.session.rollback()

    def test_invalid_second_vector_leaves_no_partial_rows_after_catch_and_commit(self):
        chunks = EvidenceChunker(
            target_chars=5, min_chars=3, max_chars=6, overlap_chars=0
        ).chunk(
            project_id="p", document_id="d", document_version=1, content="第一句。第二句。"
        )
        self.assertEqual(len(chunks), 2)
        profile = EmbeddingProfile.openai_compatible(
            model_identifier="atomic", model_revision="r1", dimensions=2
        )
        with self.assertRaises(EvidenceStoreError):
            store_embeddings(
                self.session,
                profile=profile,
                chunks=chunks,
                vectors=[(1.0, 0.0), (0.0, 0.0)],
            )
        self.session.commit()
        self.assertEqual(
            self.session.scalar(select(func.count()).select_from(EmbeddingProfileRow)), 0
        )
        self.assertEqual(
            self.session.scalar(select(func.count()).select_from(EvidenceChunkRow)), 0
        )
        self.assertEqual(
            self.session.scalar(select(func.count()).select_from(EvidenceEmbeddingRow)), 0
        )

    def test_duplicate_input_and_existing_row_collision_fail_before_staging(self):
        chunk = EvidenceChunker().chunk(
            project_id="p", document_id="d", document_version=1, content="原始事实。"
        )[0]
        profile = EmbeddingProfile.openai_compatible(
            model_identifier="collision", model_revision="r1", dimensions=2
        )
        with self.assertRaisesRegex(EvidenceStoreError, "duplicate"):
            store_embeddings(
                self.session,
                profile=profile,
                chunks=[chunk, chunk],
                vectors=[(1.0, 0.0), (1.0, 0.0)],
            )
        self.session.commit()
        self.assertEqual(
            self.session.scalar(select(func.count()).select_from(EvidenceChunkRow)), 0
        )

        store_embeddings(
            self.session, profile=profile, chunks=[chunk], vectors=[(1.0, 0.0)]
        )
        self.session.commit()
        self.session.execute(
            update(EvidenceChunkRow)
            .where(EvidenceChunkRow.id == chunk.chunk_id)
            .values(text="数据库中被篡改的内容")
        )
        self.session.commit()
        with self.assertRaisesRegex(EvidenceStoreError, "identity collision"):
            store_embeddings(
                self.session, profile=profile, chunks=[chunk], vectors=[(1.0, 0.0)]
            )
        self.session.commit()
        self.assertEqual(
            self.session.scalar(select(func.count()).select_from(EvidenceEmbeddingRow)), 1
        )

    def test_cross_project_document_ownership_is_rejected_by_composite_fk(self):
        self.session.add(ProjectRow(id="other", name="other", description=""))
        self.session.commit()
        wrong_owner = EvidenceChunker().chunk(
            project_id="other",
            document_id="d",
            document_version=1,
            content="跨项目证据。",
        )[0]
        store_chunks(self.session, [wrong_owner])
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()


if __name__ == "__main__":
    unittest.main()
