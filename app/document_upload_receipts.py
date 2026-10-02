"""Multipart operation digests and project-local successful upload receipts."""
from __future__ import annotations

import hashlib
import json

from fastapi import HTTPException
from sqlalchemy import select

from .db import DocumentRow, DocumentUploadReceiptRow


PARSER_VERSION = "multipart_document_parser_v1"


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
