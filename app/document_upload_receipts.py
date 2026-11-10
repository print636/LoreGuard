"""Multipart operation digests and project-local successful upload receipts."""
from __future__ import annotations

import hashlib
import json

from fastapi import HTTPException
from sqlalchemy import select

from .db import DocumentRow, DocumentUploadReceiptRow


PARSER_VERSION = "multipart_document_parser_v1"
IMPORT_METADATA_SCHEMA = "document-import-v1"


def import_metadata(*, filename: str, logical_name: str, replaces_document_id: str | None,
                    replaces_document_version: int | None) -> dict:
    """Only server-owned, request-bound fields; never inferred from logical suffix."""
    file_format = {"docx": "docx", "md": "markdown", "txt": "text", "json": "json"}[filename.rsplit(".", 1)[-1].lower()]
    return {
        "schema_version": IMPORT_METADATA_SCHEMA,
        "uploaded_filename": filename, "uploaded_format": file_format,
        "logical_name": logical_name,
        "replaces_document_id": replaces_document_id,
        "replaces_document_version": replaces_document_version,
    }


def upload_response_fields(metadata: dict, *, source: str) -> dict:
    return {key: metadata[key] for key in (
        "uploaded_filename", "uploaded_format", "replaces_document_id", "replaces_document_version",
    )} | {"upload_metadata_source": source}


def replay_upload_fields(db, receipt: DocumentUploadReceiptRow, row: DocumentRow,
                         *, filename: str, replace_document_id: str | None) -> dict:
    """Called only after replay_upload_receipt has verified the request digest.

    Old null metadata stays null. The matching request may describe its upload,
    but an old explicit target is certified only if the saved lineage proves it.
    """
    target = db.scalar(select(DocumentRow).where(
        DocumentRow.project_id == receipt.project_id, DocumentRow.id == replace_document_id,
    )) if replace_document_id else None
    proven = (target is not None and target.id in receipt.superseded_document_ids
              and target.name.lower() == row.name.lower() and target.version + 1 == row.version)
    expected = import_metadata(filename=filename, logical_name=row.name,
                               replaces_document_id=target.id if proven else None,
                               replaces_document_version=target.version if proven else None)
    if receipt.import_metadata is not None:
        if receipt.import_metadata != expected or (replace_document_id is not None and not proven):
            raise HTTPException(409, detail={
                "reason_code": "document_upload_receipt_unavailable",
                "message": "原上传的文件信息或替换关系无法安全核对。请刷新项目资料，不要直接重复创建版本。",
            })
        return upload_response_fields(expected, source="stored_receipt")
    return upload_response_fields(expected, source="verified_request")


def upload_request_sha256(*, filename: str, data: bytes, replace_document_id: str | None,
                          document_role: str | None, story_scope: str | None,
                          narrative_context: str | None) -> str:
    # Keep absent fields distinct from explicit choices. The parser does not
    # infer role/scope from JSON, and same-name inheritance is evaluated only
    # for the first successful operation, not again during a replay.
    semantics = {
        "parser_version": PARSER_VERSION, "filename": filename,
        "bytes_sha256": hashlib.sha256(data).hexdigest(),
        "replace_document_id": replace_document_id,
        "document_role": document_role, "story_scope": story_scope,
        "narrative_context": narrative_context,
    }
    return hashlib.sha256(json.dumps(
        semantics, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()


def find_upload_receipt(db, project_id: str, key: str | None):
    if key is None:
        return None
    return db.scalar(select(DocumentUploadReceiptRow).where(
        DocumentUploadReceiptRow.project_id == project_id,
        DocumentUploadReceiptRow.idempotency_key == key,
    ))


def replay_upload_receipt(db, receipt: DocumentUploadReceiptRow, request_sha256: str) -> tuple[DocumentRow, list[str]]:
    if receipt.request_sha256 != request_sha256:
        raise HTTPException(409, detail={
            "reason_code": "document_upload_idempotency_conflict",
            "message": "该上传操作已用于不同文件或导入选项。请核对原操作；新操作应使用新的标识。",
        })
    row = db.scalar(select(DocumentRow).where(
        DocumentRow.id == receipt.document_id, DocumentRow.project_id == receipt.project_id,
    ))
    superseded = receipt.superseded_document_ids
    if row is None or not isinstance(superseded, list) or any(not isinstance(value, str) for value in superseded):
        raise HTTPException(409, detail={
            "reason_code": "document_upload_receipt_unavailable",
            "message": "原上传记录无法安全核对。请刷新项目资料，不要直接重复创建版本。",
        })
    if superseded:
        owned_ids = set(db.scalars(select(DocumentRow.id).where(
            DocumentRow.project_id == receipt.project_id, DocumentRow.id.in_(superseded),
        )))
        if owned_ids != set(superseded):
            raise HTTPException(409, detail={
                "reason_code": "document_upload_receipt_unavailable",
                "message": "原上传记录无法安全核对。请刷新项目资料，不要直接重复创建版本。",
            })
    return row, list(superseded)
