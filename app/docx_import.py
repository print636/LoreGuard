from __future__ import annotations

import stat
import zipfile
from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import PurePosixPath

from defusedxml import ElementTree as SafeElementTree
from defusedxml.common import DefusedXmlException


_CONTENT_TYPES = "[Content_Types].xml"
_ROOT_RELATIONSHIPS = "_rels/.rels"
_DOCUMENT_XML = "word/document.xml"
_WORDPROCESSINGML_MAIN = (
    "application/vnd.openxmlformats-officedocument."
    "wordprocessingml.document.main+xml"
)
_CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
_RELATIONSHIPS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_OFFICE_DOCUMENT_RELATIONSHIPS = {
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
    "http://purl.oclc.org/ooxml/officeDocument/relationships/officeDocument",
}
_WORD_NAMESPACES = {
    "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "http://purl.oclc.org/ooxml/wordprocessingml/main",
}


def _word_tags(local_name: str) -> set[str]:
    return {f"{{{namespace}}}{local_name}" for namespace in _WORD_NAMESPACES}


_W_DOCUMENT = _word_tags("document")
_W_BODY = _word_tags("body")
_W_PARAGRAPH = _word_tags("p")
_W_TABLE = _word_tags("tbl")
_W_TABLE_ROW = _word_tags("tr")
_W_TABLE_CELL = _word_tags("tc")
_W_TEXT = _word_tags("t")
_W_TAB = _word_tags("tab")
_W_BREAKS = _word_tags("br") | _word_tags("cr")
_W_NO_BREAK_HYPHEN = _word_tags("noBreakHyphen")
_W_SOFT_HYPHEN = _word_tags("softHyphen")
_W_DELETED_CONTAINERS = _word_tags("del") | _word_tags("moveFrom")
_MAX_TABLE_NESTING = 16


@dataclass(frozen=True)
class DocxLimits:
    """Hard resource ceilings for an untrusted Office Open XML package."""

    max_archive_bytes: int = 10 * 1024 * 1024
    max_members: int = 512
    max_total_uncompressed_bytes: int = 64 * 1024 * 1024
    max_member_uncompressed_bytes: int = 16 * 1024 * 1024
    max_xml_bytes: int = 16 * 1024 * 1024
    max_compression_ratio: float = 200.0
    max_text_bytes: int = 10 * 1024 * 1024


DEFAULT_DOCX_LIMITS = DocxLimits()


class DocxImportError(ValueError):
    """A deliberately user-safe DOCX rejection without parser internals."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _invalid_docx() -> DocxImportError:
    return DocxImportError("DOCX 已损坏、加密或不是标准 Office Open XML 文档")


def _oversized_docx() -> DocxImportError:
    return DocxImportError(
        "DOCX 展开后超过安全限制，请拆分文档后重试",
        status_code=413,
    )


def _validate_member_path(name: str) -> str:
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or any(ord(character) < 32 for character in name)
    ):
        raise _invalid_docx()
    without_directory_suffix = name[:-1] if name.endswith("/") else name
    parts = without_directory_suffix.split("/")
    if (
        not without_directory_suffix
        or any(part in {"", ".", ".."} for part in parts)
        or any(":" in part for part in parts)
    ):
        raise _invalid_docx()
    path = PurePosixPath(without_directory_suffix)
    if path.is_absolute() or ".." in path.parts:
        raise _invalid_docx()
    return path.as_posix()


def _validate_archive(archive: zipfile.ZipFile, limits: DocxLimits) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > limits.max_members:
        raise _oversized_docx()

    members: dict[str, zipfile.ZipInfo] = {}
    names_casefolded: set[str] = set()
    total_uncompressed = 0
    for info in infos:
        normalized_name = _validate_member_path(info.filename)
        folded_name = normalized_name.casefold()
        if folded_name in names_casefolded:
            raise _invalid_docx()
        names_casefolded.add(folded_name)
        members[normalized_name] = info

        if info.flag_bits & 0x1:
            raise DocxImportError("不支持加密的 DOCX，请先在 Word 中解除加密")
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise _invalid_docx()
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise DocxImportError("DOCX 使用了不支持的压缩方式")
        if info.file_size < 0 or info.compress_size < 0:
            raise _invalid_docx()
        if info.file_size > limits.max_member_uncompressed_bytes:
            raise _oversized_docx()
        if (
            info.file_size
            and info.file_size / max(info.compress_size, 1)
            > limits.max_compression_ratio
        ):
            raise _oversized_docx()
        if (
            normalized_name.casefold().endswith((".xml", ".rels"))
            and info.file_size > limits.max_xml_bytes
        ):
            raise _oversized_docx()
        total_uncompressed += info.file_size
        if total_uncompressed > limits.max_total_uncompressed_bytes:
            raise _oversized_docx()

    if any(
        required not in members
        for required in (_CONTENT_TYPES, _ROOT_RELATIONSHIPS, _DOCUMENT_XML)
    ):
        raise _invalid_docx()
    if any(name.casefold().endswith("/vbaproject.bin") for name in members):
        raise DocxImportError("仅支持无宏的标准 .docx 文件", status_code=415)
    return members


def _read_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    *,
    byte_limit: int,
) -> bytes:
    try:
        with archive.open(info, "r") as source:
            payload = source.read(byte_limit + 1)
    except (EOFError, NotImplementedError, RuntimeError, zipfile.BadZipFile):
        raise _invalid_docx() from None
    if len(payload) > byte_limit:
        raise _oversized_docx()
    return payload


def _safe_xml_root(payload: bytes):
    try:
        return SafeElementTree.fromstring(payload)
    except (DefusedXmlException, SafeElementTree.ParseError, ValueError):
        raise _invalid_docx() from None


def _validate_content_types(payload: bytes) -> None:
    root = _safe_xml_root(payload)
    if root.tag != f"{{{_CONTENT_TYPES_NS}}}Types":
        raise _invalid_docx()
    document_content_types: list[str] = []
    for item in root:
        content_type = str(item.attrib.get("ContentType", ""))
        if "macroenabled" in content_type.casefold() or "vbaproject" in content_type.casefold():
            raise DocxImportError("仅支持无宏的标准 .docx 文件", status_code=415)
        if item.attrib.get("PartName") == f"/{_DOCUMENT_XML}":
            document_content_types.append(content_type)
    if document_content_types != [_WORDPROCESSINGML_MAIN]:
        raise DocxImportError("仅支持标准 .docx 文件，不支持旧版 .doc 或启用宏的文档", status_code=415)


def _validate_root_relationships(payload: bytes) -> None:
    root = _safe_xml_root(payload)
    if root.tag != f"{{{_RELATIONSHIPS_NS}}}Relationships":
        raise _invalid_docx()
    office_document_targets = []
    for item in root:
        if item.attrib.get("Type") not in _OFFICE_DOCUMENT_RELATIONSHIPS:
            continue
        if str(item.attrib.get("TargetMode", "")).casefold() == "external":
            raise _invalid_docx()
        target = str(item.attrib.get("Target", "")).lstrip("/")
        office_document_targets.append(target)
    if office_document_targets != [_DOCUMENT_XML]:
        raise _invalid_docx()


def _append_word_text(lines: list[str], node) -> None:
    stack = [node]
    while stack:
        current = stack.pop()
        if current.tag in _W_DELETED_CONTAINERS:
            continue
        if current.tag in _W_TEXT:
            value = current.text or ""
            pieces = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            for index, piece in enumerate(pieces):
                if index:
                    lines.append("")
                lines[-1] += piece
            continue
        if current.tag in _W_TAB:
            lines[-1] += "\t"
            continue
        if current.tag in _W_BREAKS:
            lines.append("")
            continue
        if current.tag in _W_NO_BREAK_HYPHEN:
            lines[-1] += "-"
            continue
        if current.tag in _W_SOFT_HYPHEN:
            lines[-1] += "\u00ad"
            continue
        stack.extend(reversed(list(current)))


def _paragraph_lines(paragraph) -> list[str]:
    lines = [""]
    _append_word_text(lines, paragraph)
    return [line.strip() for line in lines]


def _cell_text(cell, *, nesting: int) -> str:
    blocks: list[str] = []
    for child in cell:
        if child.tag in _W_PARAGRAPH:
            blocks.extend(line for line in _paragraph_lines(child) if line)
        elif child.tag in _W_TABLE:
            blocks.extend(
                line for line in _table_lines(child, nesting=nesting + 1) if line.strip()
            )
    return " / ".join(blocks)


def _table_lines(table, *, nesting: int = 0) -> list[str]:
    if nesting > _MAX_TABLE_NESTING:
        raise _invalid_docx()
    lines: list[str] = []
    for row in table:
        if row.tag not in _W_TABLE_ROW:
            continue
        cells = [
            _cell_text(cell, nesting=nesting)
            for cell in row
            if cell.tag in _W_TABLE_CELL
        ]
        if cells:
            lines.append("\t".join(cells))
    return lines


def _extract_document_body(payload: bytes) -> str:
    root = _safe_xml_root(payload)
    if root.tag not in _W_DOCUMENT:
        raise _invalid_docx()
    body = next((child for child in root if child.tag in _W_BODY), None)
    if body is None:
        raise _invalid_docx()

    lines: list[str] = []
    for child in body:
        if child.tag in _W_PARAGRAPH:
            lines.extend(_paragraph_lines(child))
        elif child.tag in _W_TABLE:
            lines.extend(_table_lines(child))

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def extract_docx_text(
    payload: bytes,
    *,
    limits: DocxLimits = DEFAULT_DOCX_LIMITS,
    max_text_bytes: int | None = None,
) -> str:
    """Extract plain, line-stable body text from an untrusted standard DOCX.

    Only ``[Content_Types].xml`` and ``word/document.xml`` are consumed. The
    importer never resolves package relationships, external targets, macros,
    embedded objects, images, headers, comments or tracked deleted text.
    """

    effective_limits = (
        replace(limits, max_text_bytes=max_text_bytes)
        if max_text_bytes is not None
        else limits
    )
    if effective_limits.max_text_bytes < 1:
        raise ValueError("max_text_bytes must be positive")
    if (
        len(payload) > effective_limits.max_archive_bytes
        or not payload.startswith(b"PK\x03\x04")
    ):
        if len(payload) > effective_limits.max_archive_bytes:
            raise _oversized_docx()
        raise _invalid_docx()
    try:
        with zipfile.ZipFile(BytesIO(payload), "r") as archive:
            members = _validate_archive(archive, effective_limits)
            content_types = _read_member(
                archive,
                members[_CONTENT_TYPES],
                byte_limit=effective_limits.max_xml_bytes,
            )
            _validate_content_types(content_types)
            root_relationships = _read_member(
                archive,
                members[_ROOT_RELATIONSHIPS],
                byte_limit=effective_limits.max_xml_bytes,
            )
            _validate_root_relationships(root_relationships)
            document_xml = _read_member(
                archive,
                members[_DOCUMENT_XML],
                byte_limit=effective_limits.max_xml_bytes,
            )
            text = _extract_document_body(document_xml)
    except DocxImportError:
        raise
    except (EOFError, OSError, RuntimeError, zipfile.BadZipFile, zipfile.LargeZipFile):
        raise _invalid_docx() from None

    if not text.strip():
        raise DocxImportError("DOCX 正文为空；请确认内容位于主文档正文或表格中")
    if len(text.encode("utf-8")) > effective_limits.max_text_bytes:
        raise _oversized_docx()
    return text
