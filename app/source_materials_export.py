"""Explicitly selected, verified run inputs as a readable text ZIP.

This module reads frozen inputs only. It neither follows current documents nor
adds the implicit old sources of confirmed character traits to the selection.
"""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import re
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from sqlalchemy import func, select

from .db import AnalysisRunInputRow, AnalysisRunInputContextRow, AnalysisRunInputNarrativeContextRow
from .narrative_context import NarrativeScopeV1, derive_authority_tier, payload_sha256
from .report_export import _inline


MAX_SOURCE_MATERIALS = 100
MAX_SOURCE_MATERIAL_BYTES = 20 * 1024 * 1024
SOURCE_MATERIAL_HEADERS = {
    "Cache-Control": "private, no-store",
    "X-Content-Type-Options": "nosniff",
}
_ROLES = {"canon": "权威设定", "character_profile": "角色设定", "chapter": "故事正文", "reference": "参考资料"}
_PUBLICATION = {"draft": "草稿", "in_review": "审阅中", "published": "已发布", "retired": "已退役", "unknown": "未核实"}
_RESOLUTION = {"unresolved": "尚未确认", "inferred": "推断，尚未确认", "confirmed": "作者已确认"}


class SourceMaterialsUnavailable(Exception):
    def __init__(self, status_code: int, code: str, message: str, documents: list | None = None):
        self.status_code = status_code
        self.detail = {"code": code, "message": message}
        if documents is not None:
            self.detail["documents"] = documents
        super().__init__(message)


def _too_large():
    raise SourceMaterialsUnavailable(413, "source_materials_too_large",
                                    "所选文稿的 UTF-8 正文合计超过 20 MiB，请减少选择、分批导出。")


def _safe_name(name: str, index: int) -> str:
    leaf = name.replace("\\", "/").rsplit("/", 1)[-1]
    stem = leaf.rsplit(".", 1)[0] if "." in leaf else leaf
    stem = re.sub(r"[^\w\-一-鿿]", "_", stem).strip("._ ")[:48] or "文稿"
    return f"{index:03d}-{stem}.md"


def _metadata(context, narrative):
    role = context.document_role if context else None
    scope = context.story_scope if context else None
    batch = context.batch_role if context else None
    if ((role is not None and role not in _ROLES)
            or (scope is not None and (not isinstance(scope, str) or not 1 <= len(scope) <= 200))
            or (batch is not None and batch not in {"target", "background"})):
        raise ValueError("invalid frozen identity")
    metadata = {"role": _ROLES.get(role, "未保存"), "scope": scope,
                "batch": {"target": "待审稿", "background": "背景资料"}.get(batch, "未保存"),
                "publication": "未保存", "resolution": "未保存", "narrative_scope": None}
    if narrative is None:
        return metadata
    p = narrative.payload
    if (narrative.schema_version != 1 or not isinstance(p, dict) or p.get("schema_version") != 1
            or payload_sha256(p) != narrative.payload_sha256
            or p.get("publication_status") not in _PUBLICATION
            or p.get("resolution_state") not in _RESOLUTION
            or not isinstance(p.get("scope"), dict)
            or payload_sha256(p["scope"]) != p.get("scope_sha256")
            or (context is not None and (p.get("legacy_document_role") != role or p.get("legacy_story_scope") != scope))):
        raise ValueError("invalid frozen narrative context")
    narrative_scope = NarrativeScopeV1.model_validate(p["scope"])
    if role is not None and p.get("authority_tier") != derive_authority_tier(role, p["resolution_state"], p["publication_status"]):
        raise ValueError("invalid frozen narrative authority")
    metadata.update(publication=_PUBLICATION[p["publication_status"]], resolution=_RESOLUTION[p["resolution_state"]],
                    narrative_scope=narrative_scope)
    return metadata


def _directory(project_name, sources):
    lines = ["# LoreGuard 资料目录", "", f"- 作品：{_inline(project_name)}",
             "- 范围：作者在所选已完成审查中明确勾选的资料；不是整个项目的当前版本。",
             "- 正文均为审查时保存的 UTF-8 文本，未增加版本前言或改写原换行。编号不代表故事时序。",
             "- Word 来源是解析后的正文，不含原 Word 排版、批注和完整修订记录。",
             "- 这里核对的是出处与版本，不是新的事实确认或审查结论。", "", "## 资料清单", ""]
    for filename, source, metadata, _ in sources:
        lines.extend([f"### {_inline(filename)}", "", f"- 文稿名称：{_inline(source.document_name)}",
                      f"- 原版本：{source.document_version}", "- 原文核验：已核对审查时保存的该版本原文",
                      f"- 资料身份：{metadata['role']}", f"- 当次用途：{metadata['batch']}",
                      f"- 适用范围：{_inline(metadata['scope']) if metadata['scope'] is not None else '未保存'}",
                      f"- 身份确认：{metadata['resolution']}", f"- 发布状态：{metadata['publication']}"])
        scope = metadata["narrative_scope"]
        if scope is not None:
            lines.append(f"- 时间线：{_inline(scope.timeline_key)}")
            if scope.release:
                lines.append(f"- 剧情版本：{_inline(scope.release.key)}（序号 {scope.release.ordinal}）")
            if scope.branch:
                lines.append(f"- 分支：{' → '.join(_inline(value) for value in scope.branch.path)}")
                if scope.branch.exclusive_group:
                    lines.append(f"- 互斥分支组：{_inline(scope.branch.exclusive_group)}")
            if scope.activity_key:
                lines.append(f"- 活动范围：{_inline(scope.activity_key)}")
        else:
            lines.append("- 结构化时间线／分支：历史记录未保存")
        lines.append("")
    lines.extend(["## 修改后如何返回", "",
        "- 本包是带出处的可读资料导出，不会修改旧报告、发布资料或自动更新已确认设定。",
        "- 上列文稿名称是 LoreGuard 保留的版本链名称，不一定是该次实际上传文件名或格式。",
        "- 修改后可上传 Markdown、TXT、JSON 或标准 DOCX，并明确选择要替换的原文稿；改名或跨格式上传仍保留所选文稿名称和版本链。系统不会根据编号或相似名称猜测对应关系。",
        "- 若未选择替换目标，仍按上传文件名执行原有同名版本规则；不要直接上传带安全编号的文件名并期待自动对应。",
        "- 需要保留 Word 排版时，请在原 Word 文件中编辑后上传；导出的 Markdown 只有解析正文，不会自动转换回原 Word 排版。",
        "- 行号对应 LoreGuard 解析后保存的原文，不是 Word 页码或排版行号。报告引用请按所列版本核对。",
        "- 本包仅含所选本次审查资料；未勾选文稿及此前确认设定的其他来源不在包内，可在报告核对相关引用。",
        "- 本包不自动附带报告。完整报告请通过原有的报告下载入口另行获取。", ""])
    return "\n".join(lines).encode("utf-8")


def build_source_materials_zip(db, run, *, project_name: str, document_ids: list[str]) -> bytes:
    if run.status != "completed":
        raise SourceMaterialsUnavailable(409, "run_not_completed", "仅已完成的审查可导出当次资料。")
    if (not 1 <= len(document_ids) <= MAX_SOURCE_MATERIALS or len(set(document_ids)) != len(document_ids)
            or any(not isinstance(value, str) or not 1 <= len(value) <= 200 for value in document_ids)):
        raise SourceMaterialsUnavailable(422, "invalid_selection", "请明确选择 1 至 100 份不重复的当次资料。")
    # Read cheap metadata first, then one selected body at a time. Upload ZIP
    # bytes (notably DOCX) are not the same limit as these parsed UTF-8 bytes.
    metadata = db.execute(select(AnalysisRunInputRow.id, AnalysisRunInputRow.document_id,
        AnalysisRunInputRow.document_name, AnalysisRunInputRow.ordinal, func.length(AnalysisRunInputRow.content))
        .where(AnalysisRunInputRow.run_id == run.id, AnalysisRunInputRow.document_id.in_(document_ids))
        .order_by(AnalysisRunInputRow.ordinal, AnalysisRunInputRow.id)).all()
    if len(metadata) != len(document_ids):
        raise SourceMaterialsUnavailable(409, "source_materials_unavailable",
            "所选资料已缺失或不属于这份审查，请刷新当次资料清单；未生成不完整资料包。",
            [{"name": None, "reason": "missing"}])
    if sum(row[4] or 0 for row in metadata) > MAX_SOURCE_MATERIAL_BYTES:
        _too_large()
    sources, failures, total_bytes = [], [], 0
    for index, meta in enumerate(metadata, start=1):
        source = db.get(AnalysisRunInputRow, meta.id)
        reason = "invalid_source"
        try:
            if (source is None or source.run_id != run.id or source.document_id not in document_ids
                    or type(source.document_version) is not int or source.document_version < 1
                    or not isinstance(source.document_name, str) or not 1 <= len(source.document_name) <= 255
                    or not isinstance(source.content, str)):
                raise ValueError
            body = source.content.encode("utf-8")
            total_bytes += len(body)
            if total_bytes > MAX_SOURCE_MATERIAL_BYTES:
                _too_large()
            reason = "content_mismatch"
            if sha256(body).hexdigest() != source.content_sha256:
                raise ValueError
            reason = "context_mismatch"
            context = _metadata(db.get(AnalysisRunInputContextRow, source.id),
                                db.get(AnalysisRunInputNarrativeContextRow, source.id))
        except (ValueError, TypeError, UnicodeError):
            failures.append({"name": meta.document_name, "reason": reason})
            continue
        sources.append((_safe_name(source.document_name, index), source, context, body))
    if failures:
        raise SourceMaterialsUnavailable(409, "source_materials_unavailable",
            "部分所选资料的原文或身份无法核验；未生成不完整资料包，也未改用当前新版。", failures)
    out = BytesIO()
    with ZipFile(out, mode="w", compression=ZIP_DEFLATED) as archive:
        for filename, data in [("资料目录.md", _directory(project_name, sources)),
                               *((name, body) for name, _, _, body in sources)]:
            info = ZipInfo(filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, data)
    return out.getvalue()
