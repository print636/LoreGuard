from __future__ import annotations

from datetime import datetime

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
    assert "冻结输入清单" in content
