"""Portable, evidence-first Markdown export for a completed analysis run."""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any


_CATEGORY_NAMES = {
    "fact_conflict": "事实冲突",
    "location_collision": "同刻多地点",
    "knowledge_without_acquisition": "知识越权",
    "item_ownership": "物品状态",
    "world_rule_conflict": "世界规则",
    "character_drift": "角色漂移",
}
_FEEDBACK_NAMES = {
    "accepted": "已接受",
    "false_positive": "误报",
    "resolved": "已解决",
}
_ROLE_NAMES = {
    "canon": "权威设定",
    "character_profile": "角色设定",
    "reference": "参考资料",
    "chapter": "故事正文",
}
_MARKDOWN_PUNCTUATION = re.compile(r"([\\`*_{}\[\]()#+\-.!|>])")
_TARGET_RESULTS = {
    "conflict": "发现冲突",
    "no_issue": "未发现冲突",
    "no_matching_observation": "未匹配到新稿表现（尚无法判断）",
    "needs_review": "证据不足或待确认",
    "unprocessed": "未处理",
}


def _clean(value: Any) -> str:
    return html.escape(str(value if value is not None else "").replace("\x00", ""), quote=False)


def _inline(value: Any) -> str:
    clean = " ".join(_clean(value).split())
    return _MARKDOWN_PUNCTUATION.sub(r"\\\1", clean)


def _quoted(value: Any) -> list[str]:
    lines = _clean(value).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return [f"> {_MARKDOWN_PUNCTUATION.sub(r'\\\1', line)}" if line else ">" for line in lines]


def _evidence_lines(evidence: Sequence[Mapping[str, Any]]) -> list[str]:
    lines: list[str] = []
    for index, span in enumerate(evidence, start=1):
        start = span.get("line_start")
        end = span.get("line_end")
        location = str(start) if start == end else f"{start}–{end}"
        lines.extend([
            f"- 证据 {index}：{_inline(span.get('document_name', '未知文档'))}，第 {location} 行",
            *_quoted(span.get("text", "")),
        ])
    return lines


def _model_note(diagnostics: Mapping[str, Any] | None) -> str:
    model = diagnostics.get("model") if isinstance(diagnostics, Mapping) else None
    if not isinstance(model, Mapping):
        return "模型执行状态未知；请在网页运行审计中核对覆盖情况。"
    if model.get("partial_fallback") is True:
        return "模型部分结果已降级；已列出的问题仍可核对，未列出的问题不能视为不存在。"
    if model.get("used") is True:
        return "模型已参与；本报告不保证发现全部问题。"
    return "模型未参与；本次只完成有限预检。"


def _character_review_note(diagnostics: Mapping[str, Any] | None) -> str:
    stage = (
        diagnostics.get("character_consistency")
        if isinstance(diagnostics, Mapping) else None
    )
    if not isinstance(stage, Mapping):
        return "缺少可核对的角色一致性阶段记录；无法判断本次是否完成角色 OOC 审查。"

    outcome = stage.get("outcome")
    if outcome == "partial":
        return "角色一致性阶段记录为部分完成；未覆盖内容不能视为没有角色 OOC 问题。"
    if outcome == "degraded":
        return "角色一致性阶段记录为降级；不能据此宣称角色 OOC 审查已经完成。"
    if outcome == "skipped":
        return "角色一致性阶段记录为跳过；本次角色 OOC 未评估。"
    if outcome == "disabled":
        return "角色一致性阶段记录为未启用；本次角色 OOC 未评估。"
    if outcome == "completed":
        if stage.get("material_coverage") == "partial":
            return "角色一致性阶段记录为完成，但材料覆盖标记为部分；未覆盖内容不能视为没有角色 OOC 问题。"
        if (
            stage.get("snapshot_bound") is not True
            or stage.get("material_coverage") != "complete"
            or stage.get("enabled") is False
        ):
            return "角色一致性阶段记录为完成，但冻结材料覆盖无法核对；不能将本次视为完整的角色 OOC 审查。"
        return "角色一致性阶段诊断记录为已完成；这不保证本次有可评估的新稿，也不保证判断准确或没有遗漏。"
    return "角色一致性阶段状态无法核对；不能据此判断角色 OOC 是否存在。"


def _valid_count(value: object) -> bool:
    return type(value) is int and 0 <= value <= 2**53 - 1


def _stage_supports_target_counts(stage: Mapping[str, Any]) -> bool:
    if stage.get("enabled") is not True or stage.get("snapshot_bound") is not True:
        return False
    outcome = stage.get("outcome")
    counts = stage.get("counts", {})
    if not isinstance(counts, Mapping):
        return False
    keys = ("planned_chunks", "model_called_chunks", "model_completed_chunks",
            "model_uncalled_chunks", "model_incomplete_chunks")
    if any(key in counts for key in keys):
        values = [counts.get(key) for key in keys]
        if not all(_valid_count(value) for value in values):
            return False
        planned, called, completed, uncalled, incomplete = values
        if (completed > called or called + uncalled != planned or completed + incomplete != planned
                or outcome == "completed" and (not planned or incomplete)):
            return False
    material = stage.get("material_coverage")
    explanation_incomplete = stage.get("explanation_coverage") in ("partial", "not_run")
    if outcome == "completed":
        return material == "complete" and not explanation_incomplete
    if outcome == "partial":
        return material == "partial" or explanation_incomplete
    return outcome == "degraded"


def _target_assessment_note(diagnostics: Mapping[str, Any] | None) -> str:
    unknown = "缺少可核对的逐项判断记录，无法确认哪些角色设定得到判断。"
    if not isinstance(diagnostics, Mapping):
        return unknown
    batch = diagnostics.get("review_batch")
    if isinstance(batch, Mapping) and batch.get("mode") == "baseline_build":
        return "本次用于整理后续审查的基线，没有待对照的新稿。"
    stage = diagnostics.get("character_consistency")
    if not isinstance(stage, Mapping):
        return unknown
    if stage.get("outcome") in ("disabled", "skipped"):
        return "本次没有执行新稿的角色设定对照。"
    if isinstance(batch, Mapping) and batch.get("character_review_enabled") is False:
        return unknown
    if not _stage_supports_target_counts(stage):
        return unknown
    summary = stage.get("target_assessment_summary_v1")
    if not isinstance(summary, Mapping) or summary.get("schema_version") != "character-target-assessment-summary-v1":
        return unknown
    total = summary.get("total_targets")
    if not _valid_count(total) or not all(_valid_count(summary.get(key)) for key in _TARGET_RESULTS):
        return unknown
    if sum(summary[key] for key in _TARGET_RESULTS) != total:
        return unknown
    if stage["outcome"] == "degraded" and any(
        summary[key] for key in ("conflict", "no_issue", "no_matching_observation")
    ):
        return unknown
    if total == 0:
        return "本次没有可与新稿对照的角色设定。"
    results = "，".join(f"{summary[key]} 项{label}" for key, label in _TARGET_RESULTS.items() if summary[key])
    return f"本次对照范围共 {total} 项角色设定，其中 {results}。"


def render_markdown_report(
    *,
    project_name: str,
    run_id: str,
    completed_at: datetime | None,
    input_documents: Sequence[Mapping[str, Any]],
    issues: Sequence[Mapping[str, Any]],
    latest_feedback: Mapping[str, Mapping[str, Any]],
    clarifications: Sequence[Mapping[str, Any]],
    diagnostics: Mapping[str, Any] | None,
) -> str:
    """Render a current feedback snapshot from immutable run results.

    Only explicitly selected fields enter the file. In particular, raw model
    responses, provider metadata, API keys and developer diagnostics are never
    serialized into the export.
    """
    lines = [
        "# LoreGuard 故事剧情一致性审查报告",
        "",
        f"- 项目：{_inline(project_name)}",
        f"- 运行 ID：{_inline(run_id)}",
        f"- 完成时间：{_inline(completed_at.isoformat(sep=' ', timespec='seconds') if completed_at else '未知')}",
        f"- 一致性问题：{len(issues)} 条",
        f"- 待澄清/开放问题：{len(clarifications)} 条",
        "",
        "## 本次审查范围",
        "",
    ]
    if input_documents:
        for document in input_documents:
            name = _inline(document.get("document_name", "未知文档"))
            version = _inline(document.get("document_version", "?"))
            role = _ROLE_NAMES.get(str(document.get("document_role")), "其他资料")
            batch_role = "待审稿" if document.get("batch_role") == "target" else "背景资料"
            lines.append(f"- {name}（版本 {version}；{role}；{batch_role}）")
    else:
        lines.append("- 历史运行缺少冻结输入清单，无法核对具体文档版本。")
    lines.extend([
        "",
        "## 结果边界",
        "",
        _model_note(diagnostics),
        _character_review_note(diagnostics),
        _target_assessment_note(diagnostics),
        "报告只列出本次运行发现且带原文证据的结果；0 条问题不代表故事绝对无误。",
        "",
        "## 一致性问题",
        "",
    ])
    if not issues:
        lines.extend(["本次未列出有证据支持的一致性问题。", ""])
    for index, issue in enumerate(issues, start=1):
        title = _inline(issue.get("title", "未命名问题"))
        category = _CATEGORY_NAMES.get(str(issue.get("category")), str(issue.get("category", "其他")))
        feedback = latest_feedback.get(str(issue.get("id")), {})
        status = _FEEDBACK_NAMES.get(str(feedback.get("label")), "未反馈")
        lines.extend([
            f"### {index}. {title}",
            "",
            f"- 类别：{_inline(category)}",
            f"- 严重度：{_inline(issue.get('severity', '未知'))}",
            f"- 审阅状态：{status}",
            "",
            "**问题说明**",
            *_quoted(issue.get("explanation", "")),
            "",
            "**原文证据**",
            *_evidence_lines(issue.get("evidence") or []),
        ])
        if feedback.get("comment"):
            lines.extend(["", "**审阅备注**", *_quoted(feedback["comment"])])
        lines.append("")
    lines.extend(["## 待澄清与开放问题", ""])
    if not clarifications:
        lines.extend(["本次未列出待澄清或开放问题。", ""])
    for index, item in enumerate(clarifications, start=1):
        kind = "开放问题" if item.get("kind") == "open_question" else "待澄清"
        lines.extend([
            f"### {index}. {kind}",
            "",
            *_quoted(item.get("text", "")),
            "",
            *_evidence_lines([item["evidence"]] if item.get("evidence") else []),
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"
