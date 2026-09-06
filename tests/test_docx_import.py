from __future__ import annotations

import os
import struct
import unittest
import zipfile
from io import BytesIO
from unittest.mock import patch

# This suite must never inherit a developer model configuration.
os.environ["OPENAI_API_KEY"] = ""
os.environ["ENABLE_MODEL_EXTRACTION"] = "false"

from fastapi.testclient import TestClient

from app.docx_import import DocxImportError, DocxLimits, extract_docx_text
from app.main import app, settings, write_limiter
from app.service import execute_analysis


CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="{main_type}"/>
</Types>"""
ROOT_RELATIONSHIPS = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
STANDARD_MAIN_TYPE = (
    "application/vnd.openxmlformats-officedocument."
    "wordprocessingml.document.main+xml"
)
WORD_PREFIX = """<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<w:body>"""
WORD_SUFFIX = "<w:sectPr/></w:body></w:document>"


def make_docx(
    body: str,
    *,
    document_xml: str | None = None,
    extra_members: dict[str, bytes | str] | None = None,
    main_type: str = STANDARD_MAIN_TYPE,
    root_relationships: str = ROOT_RELATIONSHIPS,
    compression: int = zipfile.ZIP_DEFLATED,
) -> bytes:
    target = BytesIO()
    with zipfile.ZipFile(target, "w", compression=compression) as archive:
        archive.writestr(
            "[Content_Types].xml",
            CONTENT_TYPES.format(main_type=main_type),
        )
        archive.writestr("_rels/.rels", root_relationships)
        archive.writestr(
            "word/document.xml",
            document_xml if document_xml is not None else WORD_PREFIX + body + WORD_SUFFIX,
        )
        for name, payload in (extra_members or {}).items():
            archive.writestr(name, payload)
    return target.getvalue()


def mark_zip_encrypted(payload: bytes) -> bytes:
    """Set the encrypted flag in local and central headers for rejection tests."""

    mutated = bytearray(payload)
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        start = 0
        while True:
            offset = mutated.find(signature, start)
            if offset < 0:
                break
            flags = struct.unpack_from("<H", mutated, offset + flag_offset)[0]
            struct.pack_into("<H", mutated, offset + flag_offset, flags | 0x1)
            start = offset + 4
    return bytes(mutated)


class DocxImportUnitTests(unittest.TestCase):
    def test_accepts_strict_wordprocessingml_namespace(self):
        strict_namespace = "http://purl.oclc.org/ooxml/wordprocessingml/main"
        strict_document = f"""<?xml version="1.0"?>
<w:document xmlns:w="{strict_namespace}">
  <w:body><w:p><w:r><w:t>严格格式正文</w:t></w:r></w:p></w:body>
</w:document>"""
        strict_relationships = ROOT_RELATIONSHIPS.replace(
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
            "http://purl.oclc.org/ooxml/officeDocument/relationships/officeDocument",
        )
        payload = make_docx(
            "",
            document_xml=strict_document,
            root_relationships=strict_relationships,
        )
        self.assertEqual("严格格式正文", extract_docx_text(payload))

    def test_extracts_paragraphs_breaks_unicode_and_table_rows_as_stable_lines(self):
        payload = make_docx(
            """
<w:p><w:r><w:t>第一段：潮汐🌊</w:t><w:br/><w:t>段内换行</w:t></w:r></w:p>
<w:p/>
<w:tbl>
  <w:tr>
    <w:tc><w:p><w:r><w:t>角色</w:t></w:r></w:p></w:tc>
    <w:tc><w:p><w:r><w:t>状态</w:t></w:r></w:p></w:tc>
  </w:tr>
  <w:tr>
    <w:tc><w:p><w:r><w:t>林澈</w:t></w:r></w:p></w:tc>
    <w:tc><w:p><w:r><w:t>清醒</w:t></w:r></w:p><w:p><w:r><w:t>持有钥匙</w:t></w:r></w:p></w:tc>
  </w:tr>
</w:tbl>
<w:p><w:r><w:t>末段</w:t><w:tab/><w:t>附注</w:t></w:r></w:p>
"""
        )

        self.assertEqual(
            "第一段：潮汐🌊\n段内换行\n\n角色\t状态\n林澈\t清醒 / 持有钥匙\n末段\t附注",
            extract_docx_text(payload),
        )

    def test_ignores_deleted_text_and_never_resolves_external_relationships(self):
        external_marker = "https://must-not-be-requested.invalid/private"
        payload = make_docx(
            """
<w:p><w:r><w:t>保留文本</w:t></w:r><w:del><w:r><w:delText>删除文本</w:delText></w:r></w:del></w:p>
<w:p><w:hyperlink r:id="rId9"><w:r><w:t>链接显示文字</w:t></w:r></w:hyperlink></w:p>
""",
            extra_members={
                "word/_rels/document.xml.rels": f"""<?xml version="1.0"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="{external_marker}" TargetMode="External"/>
</Relationships>"""
            },
        )

        text = extract_docx_text(payload)
        self.assertEqual("保留文本\n链接显示文字", text)
        self.assertNotIn("删除文本", text)
        self.assertNotIn(external_marker, text)

    def test_rejects_corrupt_non_docx_empty_or_entity_xml_with_safe_errors(self):
        cases = (
            b"not-a-zip",
            make_docx("<w:p/>"),
            make_docx(
                "",
                document_xml="""<?xml version="1.0"?>
<!DOCTYPE document [<!ENTITY boom "unsafe">]>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body><w:p><w:r><w:t>&boom;</w:t></w:r></w:p></w:body>
</w:document>""",
            ),
        )
        for payload in cases:
            with self.subTest(size=len(payload)):
                with self.assertRaises(DocxImportError) as caught:
                    extract_docx_text(payload)
                self.assertNotIn("Traceback", str(caught.exception))
                self.assertNotIn("word/document.xml", str(caught.exception))

    def test_rejects_macro_enabled_disguised_as_docx(self):
        payload = make_docx(
            "<w:p><w:r><w:t>正文</w:t></w:r></w:p>",
            main_type="application/vnd.ms-word.document.macroEnabled.main+xml",
            extra_members={"word/vbaProject.bin": b"macro"},
        )
        with self.assertRaises(DocxImportError) as caught:
            extract_docx_text(payload)
        self.assertEqual(415, caught.exception.status_code)
        self.assertIn("无宏", str(caught.exception))

    def test_rejects_path_traversal_duplicate_members_and_external_main_part(self):
        external_main = ROOT_RELATIONSHIPS.replace(
            'Target="word/document.xml"',
            'Target="https://invalid.example/document.xml" TargetMode="External"',
        )
        cases = (
            make_docx(
                "<w:p><w:r><w:t>正文</w:t></w:r></w:p>",
                extra_members={"../escape.xml": "unsafe"},
            ),
            make_docx(
                "<w:p><w:r><w:t>正文</w:t></w:r></w:p>",
                extra_members={"WORD/DOCUMENT.XML": "ambiguous"},
            ),
            make_docx(
                "<w:p><w:r><w:t>正文</w:t></w:r></w:p>",
                root_relationships=external_main,
            ),
        )
        for payload in cases:
            with self.subTest(size=len(payload)):
                with self.assertRaises(DocxImportError):
                    extract_docx_text(payload)

    def test_rejects_encrypted_flag_and_bounded_archive_shapes(self):
        ordinary = make_docx("<w:p><w:r><w:t>正文</w:t></w:r></w:p>")
        with self.assertRaisesRegex(DocxImportError, "加密"):
            extract_docx_text(mark_zip_encrypted(ordinary))

        limit_cases = (
            DocxLimits(max_archive_bytes=100),
            DocxLimits(max_members=2),
            DocxLimits(max_total_uncompressed_bytes=100),
            DocxLimits(max_member_uncompressed_bytes=100),
            DocxLimits(max_xml_bytes=100),
            DocxLimits(max_compression_ratio=1.0),
        )
        for limits in limit_cases:
            with self.subTest(limits=limits):
                with self.assertRaises(DocxImportError) as caught:
                    extract_docx_text(ordinary, limits=limits)
                self.assertEqual(413, caught.exception.status_code)

    def test_rejects_extracted_text_over_the_separate_output_limit(self):
        payload = make_docx(
            "<w:p><w:r><w:t>这是超过限制的正文</w:t></w:r></w:p>"
        )
        with self.assertRaises(DocxImportError) as caught:
            extract_docx_text(payload, max_text_bytes=5)
        self.assertEqual(413, caught.exception.status_code)

    def test_rejects_pathological_nested_tables_without_recursion_failure(self):
        nested = "<w:p><w:r><w:t>正文</w:t></w:r></w:p>"
        for _ in range(18):
            nested = f"<w:tbl><w:tr><w:tc>{nested}</w:tc></w:tr></w:tbl>"
        with self.assertRaises(DocxImportError):
            extract_docx_text(make_docx(nested))


class DocxUploadApiTests(unittest.TestCase):
    def setUp(self) -> None:
        write_limiter.events.clear()
        settings.enable_model_extraction = False
        settings.openai_api_key = ""

    def test_docx_upload_runs_analysis_with_extracted_evidence_line_numbers(self):
        payload = make_docx(
            """
<w:p><w:r><w:t>@fact subject="A" predicate="color" value="white" | first</w:t></w:r></w:p>
<w:p/>
<w:p><w:r><w:t>@fact subject="A" predicate="color" value="black" | second</w:t></w:r></w:p>
"""
        )
        with patch("app.main.dispatch_analysis"):
            with TestClient(app) as client:
                project = client.post(
                    "/api/v1/projects", json={"name": "DOCX 证据行号"}
                ).json()
                upload = client.post(
                    f"/api/v1/projects/{project['id']}/documents",
                    files={
                        "file": (
                            "chapter.docx",
                            payload,
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        )
                    },
                    data={"document_role": "chapter", "story_scope": "global"},
                )
                self.assertEqual(201, upload.status_code, upload.text)
                self.assertEqual("chapter.docx", upload.json()["name"])
                self.assertEqual(3, len(upload.json()["content"].splitlines()))

                run = client.post(
                    f"/api/v1/projects/{project['id']}/analysis-runs"
                ).json()
                execute_analysis(run["id"])
                issues = client.get(
                    f"/api/v1/analysis-runs/{run['id']}/issues"
                ).json()
                self.assertEqual(1, len(issues))
                self.assertEqual(
                    {1, 3}, {row["line_start"] for row in issues[0]["evidence"]}
                )

    def test_docx_upload_returns_safe_actionable_errors(self):
        with TestClient(app) as client:
            project = client.post(
                "/api/v1/projects", json={"name": "DOCX 错误"}
            ).json()
            invalid = client.post(
                f"/api/v1/projects/{project['id']}/documents",
                files={"file": ("broken.docx", b"not-a-docx", "application/octet-stream")},
            )
            self.assertEqual(400, invalid.status_code)
            self.assertEqual(
                "DOCX 已损坏、加密或不是标准 Office Open XML 文档",
                invalid.json()["detail"],
            )
            legacy = client.post(
                f"/api/v1/projects/{project['id']}/documents",
                files={"file": ("legacy.doc", b"legacy", "application/msword")},
            )
            self.assertEqual(415, legacy.status_code)
            self.assertIn("标准 DOCX", legacy.json()["detail"])


if __name__ == "__main__":
    unittest.main()
