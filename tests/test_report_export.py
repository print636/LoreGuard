from __future__ import annotations

from datetime import datetime

import pytest

from app.report_export import render_markdown_report


def test_markdown_export_keeps_evidence_and_feedback_without_rendering_user_markup():
    content = render_markdown_report(
        project_name="星轨 <script> # 一",
        run_id="example-run",
        completed_at=datetime(2026, 9, 23, 12, 30),
        input_documents=[{
            "document_name": "chapter.md",
            "document_version": 2,
            "document_role": "chapter",
            "batch_role": "target",
        }],
        issues=[{
            "id": "issue-1",
            "title": "角色 #冲突 <img>",
            "category": "fact_conflict",
            "severity": "high",
            "explanation": "前文明确内向\n# 后文突然外向",
            "raw_model_response": "SHOULD_NOT_EXPORT",
            "evidence": [{
                "document_name": "chapter.md",
                "line_start": 8,
                "line_end": 9,
                "text": "<script>alert(1)</script>\n# 伪标题",
            }],
        }],
        latest_feedback={
            "issue-1": {"label": "accepted", "comment": "需要核对 *前一章*"}
        },
        clarifications=[{
            "kind": "open_question",
            "text": "他是否知道真相？",
            "evidence": {
                "document_name": "chapter.md",
                "line_start": 10,
                "line_end": 10,
                "text": "他是否知道真相？",
            },
        }],
        diagnostics={
            "model": {"used": True, "partial_fallback": True},
            "provider": {"api_key": "SHOULD_NOT_EXPORT"},
        },
    )

    assert "chapter\\.md（版本 2；故事正文；待审稿）" in content
    assert "模型部分结果已降级" in content
    assert "审阅状态：已接受" in content
    assert "第 8–9 行" in content
    assert "待澄清与开放问题" in content
    assert "<script>" not in content
    assert "&lt;script&gt;" in content
    assert "SHOULD_NOT_EXPORT" not in content
    assert "\n# 后文突然外向" not in content
    assert "\n> \\# 后文突然外向" in content


def test_markdown_export_zero_issues_does_not_claim_story_is_correct():
    content = render_markdown_report(
        project_name="空结果",
        run_id="empty-run",
        completed_at=None,
        input_documents=[],
        issues=[],
        latest_feedback={},
        clarifications=[],
        diagnostics=None,
    )

    assert "0 条问题不代表故事绝对无误" in content
    assert "模型执行状态未知" in content
    assert "缺少可核对的角色一致性阶段记录" in content
    assert "冻结输入清单" in content


def _character_export(stage, *, mode="draft_review"):
    return render_markdown_report(
        project_name="角色审查",
        run_id="character-run",
        completed_at=None,
        input_documents=[{
            "document_name": "profile.md",
            "document_version": 1,
            "document_role": "character_profile",
            "batch_role": "background",
        }],
        issues=[],
        latest_feedback={},
        clarifications=[],
        diagnostics={
            "review_batch": {"mode": mode},
            "model": {"used": True, "partial_fallback": False},
            "character_consistency": stage,
        },
    )


def test_markdown_export_names_character_profile_and_partial_character_coverage():
    content = _character_export({
        "outcome": "partial",
        "material_coverage": "partial",
        "reason_code": "run_token_budget",
    })

    assert "profile\\.md（版本 1；角色设定；背景资料）" in content
    assert "角色一致性阶段记录为部分完成" in content
    assert "未覆盖内容不能视为没有角色 OOC 问题" in content
    assert "模型已参与" in content


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ({"outcome": "degraded"}, "角色一致性阶段记录为降级"),
        ({"outcome": "skipped"}, "本次角色 OOC 未评估"),
        ({"outcome": "disabled"}, "本次角色 OOC 未评估"),
        ({"outcome": "completed", "material_coverage": "partial"},
         "材料覆盖标记为部分"),
        ({"outcome": "completed", "material_coverage": "unknown"},
         "冻结材料覆盖无法核对"),
        ({"outcome": "completed", "material_coverage": "complete",
          "snapshot_bound": True}, "诊断记录为已完成"),
        ({"outcome": "unexpected", "reason_code": "SHOULD_NOT_EXPORT"},
         "角色一致性阶段状态无法核对"),
        (None, "缺少可核对的角色一致性阶段记录"),
    ],
)
def test_markdown_export_reports_only_persisted_character_stage_state(stage, expected):
    content = _character_export(stage)

    assert expected in content
    assert "SHOULD_NOT_EXPORT" not in content
    assert "0 条问题不代表故事绝对无误" in content


def _summary_stage(**updates):
    return {
        "enabled": True, "snapshot_bound": True, "outcome": "partial",
        "material_coverage": "partial", "explanation_coverage": "partial",
        "target_assessment_summary_v1": {
            "schema_version": "character-target-assessment-summary-v1",
            "total_targets": 5, "conflict": 1, "no_issue": 1,
            "no_matching_observation": 1, "needs_review": 1, "unprocessed": 1,
        },
        **updates,
    }


def test_export_separates_frozen_target_decisions_from_formal_issue_count():
    stage = _summary_stage(case_trace=[{"text": "NEVER_EXPORT_CLUE_DETAIL"}])
    output = _character_export(stage)
    assert "本次对照范围共 5 项角色设定，其中 1 项发现冲突，1 项未发现冲突，1 项未匹配到新稿表现（尚无法判断），1 项证据不足或待确认，1 项未处理。" in output
    assert "NEVER_EXPORT_CLUE_DETAIL" not in output
    assert "一致性问题：0 条" in output


@pytest.mark.parametrize("changes", [
    {"schema_version": "unknown"}, {"total_targets": 6}, {"no_issue": True},
    {"conflict": -1}, {"needs_review": "1"}, {"total_targets": 2**53},
])
def test_export_rejects_unverifiable_summary_counts(changes):
    stage = _summary_stage()
    stage["target_assessment_summary_v1"].update(changes)
    assert "缺少可核对的逐项判断记录" in _character_export(stage)


@pytest.mark.parametrize(("stage", "mode", "expected"), [
    (_summary_stage(target_assessment_summary_v1=None), "draft_review", "缺少可核对的逐项判断记录"),
    (_summary_stage(snapshot_bound=False), "draft_review", "缺少可核对的逐项判断记录"),
    (_summary_stage(outcome="completed"), "draft_review", "缺少可核对的逐项判断记录"),
    (_summary_stage(outcome="degraded"), "draft_review", "缺少可核对的逐项判断记录"),
    (_summary_stage(counts={"planned_chunks": 2}), "draft_review", "缺少可核对的逐项判断记录"),
    (_summary_stage(), "baseline_build", "没有待对照的新稿"),
    (_summary_stage(outcome="disabled"), "draft_review", "没有执行新稿的角色设定对照"),
    (_summary_stage(outcome="skipped"), "draft_review", "没有执行新稿的角色设定对照"),
])
def test_export_only_renders_counts_for_a_verifiable_draft_stage(stage, mode, expected):
    content = _character_export(stage, mode=mode)
    assert expected in content
    assert "本次对照范围共" not in content


def test_export_reports_empty_target_universe_separately_from_positive_judgments():
    stage = _summary_stage()
    summary = stage["target_assessment_summary_v1"]
    summary.update({key: 0 for key in summary if key != "schema_version"})
    output = _character_export(stage)
    assert "本次没有可与新稿对照的角色设定" in output
    assert "项未发现冲突" not in output
