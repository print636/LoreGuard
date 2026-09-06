from __future__ import annotations

import hashlib
import inspect
import unittest
from dataclasses import replace

from app.evidence_chunks import EvidenceChunker, EvidenceEmbeddingIndex


class EvidenceChunkerTests(unittest.TestCase):
    def test_nearest_seam_requires_an_explicit_keyword_only_chunker_version(self):
        parameters = inspect.signature(EvidenceEmbeddingIndex.nearest).parameters
        chunker_version = parameters["chunker_version"]
        self.assertEqual(chunker_version.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(chunker_version.default, inspect.Parameter.empty)
        self.assertLess(
            list(parameters).index("chunker_version"),
            list(parameters).index("query_vector"),
        )

    def test_chunk_constructor_rejects_forged_hash_id_and_ranges(self):
        chunk = EvidenceChunker().chunk(
            project_id="p", document_id="d", document_version=1, content="有效事实。"
        )[0]
        with self.assertRaisesRegex(ValueError, "text hash"):
            replace(chunk, text_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "chunk id"):
            replace(chunk, chunk_id="chk-" + ("0" * 64))
        with self.assertRaisesRegex(ValueError, "text length"):
            replace(chunk, char_end=chunk.char_end + 1)
        for invalid_version in ("", " padded ", "v" * 81):
            with self.subTest(invalid_version=invalid_version), self.assertRaisesRegex(
                ValueError, "chunker version"
            ):
                replace(chunk, chunker_version=invalid_version)

    def test_empty_and_whitespace_only_documents_have_no_chunks(self):
        chunker = EvidenceChunker()
        for content in ("", "  \n\t", "\r\n\r\n"):
            with self.subTest(content=repr(content)):
                self.assertEqual(
                    chunker.chunk(
                        project_id="project",
                        document_id="document",
                        document_version=1,
                        content=content,
                    ),
                    (),
                )

    def test_crlf_is_normalized_while_line_ranges_remain_exact(self):
        content = "第一行。\r\n第二行仍在继续。\r\n第三行结束。"
        chunker = EvidenceChunker(target_chars=10, min_chars=5, max_chars=14, overlap_chars=2)
        chunks = chunker.chunk(
            project_id="p", document_id="d", document_version=2, content=content
        )
        normalized = content.replace("\r\n", "\n")
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertEqual(normalized[chunk.char_start : chunk.char_end], chunk.text)
            covered = normalized[chunk.char_start : chunk.char_end]
            expected_start = normalized.count("\n", 0, chunk.char_start) + 1
            expected_end = normalized.count("\n", 0, chunk.char_end - 1) + 1
            self.assertEqual((chunk.line_start, chunk.line_end), (expected_start, expected_end))
            self.assertTrue(covered)

    def test_long_unicode_line_is_bounded_and_makes_forward_progress(self):
        content = "列车驶过银河🌌" * 220
        chunker = EvidenceChunker()
        chunks = chunker.chunk(
            project_id="p", document_id="long", document_version=1, content=content
        )
        self.assertGreater(len(chunks), 2)
        self.assertEqual(chunks[0].char_start, 0)
        self.assertEqual(chunks[-1].char_end, len(content))
        for previous, current in zip(chunks, chunks[1:]):
            self.assertLess(previous.char_start, current.char_start)
            self.assertLessEqual(len(previous.text), 600)
            self.assertLessEqual(previous.char_end - current.char_start, 80)
        self.assertLessEqual(len(chunks[-1].text), 600)
        self.assertTrue(all(chunk.line_start == chunk.line_end == 1 for chunk in chunks))

    def test_chinese_sentence_boundaries_and_ids_are_stable(self):
        content = ("白塔记录角色的记忆。黑潮改变时间线！" * 45) + "终章。"
        chunker = EvidenceChunker()
        first = chunker.chunk(
            project_id="p", document_id="story", document_version=3, content=content
        )
        second = chunker.chunk(
            project_id="p", document_id="story", document_version=3, content=content
        )
        self.assertEqual(first, second)
        self.assertTrue(all(chunk.text.endswith(("。", "！")) for chunk in first))
        changed = chunker.chunk(
            project_id="p", document_id="story", document_version=4, content=content
        )
        self.assertNotEqual(first[0].chunk_id, changed[0].chunk_id)
        different_parameters = EvidenceChunker(
            target_chars=451, min_chars=300, max_chars=600, overlap_chars=80
        ).chunk(
            project_id="p", document_id="story", document_version=3, content=content
        )
        self.assertNotEqual(first[0].chunker_version, different_parameters[0].chunker_version)
        self.assertNotEqual(first[0].chunk_id, different_parameters[0].chunk_id)

    def test_blank_regions_are_skipped_without_breaking_progress_or_lines(self):
        content = ("\n" * 700) + "有效事实。"
        chunks = EvidenceChunker().chunk(
            project_id="p", document_id="blank", document_version=1, content=content
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "有效事实。")
        self.assertEqual(chunks[0].line_start, 701)

    def test_snapshot_hash_is_verified_against_normalized_content(self):
        content = "规则一。\r\n规则二。"
        raw_content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        chunks = EvidenceChunker().chunk(
            project_id="p",
            document_id="d",
            document_version=1,
            content=content,
            content_sha256=raw_content_hash,
        )
        self.assertEqual(chunks[0].snapshot.content_sha256, raw_content_hash)
        with self.assertRaisesRegex(ValueError, "content hash"):
            EvidenceChunker().chunk(
                project_id="p",
                document_id="d",
                document_version=1,
                content=content,
                content_sha256="0" * 64,
            )


if __name__ == "__main__":
    unittest.main()
