# 作者报告中的事件比较与材料缺口

本阶段在待复核线索中增加“本次比较”和“仍待复核的原因”。作者可以看到模型实际比较了哪些原文，以及哪些材料尚未核清。每条可定位的材料提供分析时原文入口。

## 作者看到的含义

- “本次比较”来自已经校验的逐事件主审记录。显示的是模型的比较意见，最终条目仍是待复核线索。
- “待核清材料”表示候选核验或处理还存在缺口。其引用通过校验，只能证明出处与冻结原文一致，候选含义仍可能不成立。
- 一个缺口的引用可以与本次比较重叠，也可以位于别处。位置不同只表示两个引用范围不重叠，不能证明它们属于独立事件，也不能排除例外、指令或前后文对本次比较的影响。
- 无法确认出处的缺口按数量与未核清事项汇总，明确标为未能定位。页面不会根据其他线索猜测它发生在上午、晚间或某个事件中。
- 正式问题、待复核线索、习惯适用情境的分类与计数沿用现有规则。本阶段没有调整整稿部分覆盖状态或正式晋升条件。

本阶段沿用当前逐事件审查策略。历史文档中曾采用的两次反向行为门槛，不是这次展示功能的前提。这里的定位核验也不独立证明主审意见正确。

## 报告接口

`GET /api/v1/analysis-runs/{run_id}/review-clues` 的单个条目可以增加 `material_review`。

```json
{
  "comparisons": [
    { "evidence_indices": [1], "opinion": "contradicts" }
  ],
  "gaps": [
    {
      "index": 0,
      "count": 1,
      "reason": "statement_unverified",
      "location": "located",
      "relation": "elsewhere",
      "evidence": {
        "document_id": "opaque-document-id",
        "document_name": "新稿.md",
        "line_start": 1,
        "line_end": 1,
        "text": "从该次运行冻结版本恢复的原文"
      }
    }
  ],
  "omitted_count": 0,
  "total_count": 1
}
```

`comparisons` 最多 8 项，`evidence_indices` 引用同一个线索现有的证据数组。意见取值为 `contradicts`、`explained`、`needs_confirmation` 或 `insufficient_evidence`，界面均转换为明确标注“模型认为”的人话。

`gaps` 最多 12 个分组。每组的 `index` 是保存时的序号，用于原文读取，`count` 是合并在该组内的核验事项数。同一位置、阶段和原因的事项可以合并。`total_count` 等于所有组的 `count` 加上 `omitted_count`，这些数量不是独立事件数或问题数。

`location=unknown` 时不提供 `evidence`，`relation` 为 `unknown`。无法再次核验的已保存坐标也按未知范围呈现。`omitted_count` 表示未逐条保存的事项数量，不补造原文或材料条目。

原因采用以下枚举，页面只显示对应说明。

| API 原因 | 作者说明 |
| --- | --- |
| `actor_unverified` | 行为是否属于这个角色尚未核实 |
| `statement_unverified` | 原文能否支持提取出的含义尚未核实 |
| `object_unverified` | 动作涉及的对象尚未核实 |
| `source_unverified` | 候选材料的原文依据尚未核实 |
| `record_unverified` | 这部分候选材料尚未通过核验 |
| `response_unavailable` | 这部分核验未取得可用结果 |
| `processing_incomplete` | 这部分材料尚未处理完整 |

## 原文阅读与来源保护

已定位缺口复用 `GET /api/v1/analysis-runs/{run_id}/evidence-preview`，参数为 `kind=material_gap`、`item_id=线索ID`、`evidence_index=gap.index`。分页和返回结构沿用现有阅读器。比较意见的引用继续使用 `kind=review_clue` 及原证据序号。

服务端先验证工作区、运行和线索，再验证该次运行冻结的确认项、主审引用与材料记录的绑定。缺口正文只从该次运行的输入快照恢复，输入身份、文档身份、正文摘要、行范围和摘录摘要均须匹配。客户端不能指定任意原文或其他运行的输入进行预览。未知范围没有原文读取入口。

内部 `extra.material_review_v1` 只保留定位、摘要及有限原因，公共元数据中移除该字段。公开报告使用上面的最小投影，页面不展示候选编号、事件摘要、原始声明或内部诊断码。

旧报告没有材料记录时不推断缺口位置。绑定损坏时省略新增说明，已有线索仍按原有规则验证和展示。所有读取使用 SELECT，不发起模型调用、不写反馈，原文响应禁止缓存。

## 验证范围

报告侧集中验证通过。后端报告阅读与线索边界共 53 项，包含本阶段新增的来源、未知范围、越权和损坏数据检查。前端报告与阅读合同共 14 项，生产构建通过。新增 Chromium 用例在 375 像素宽度核对了本次比较、展开待核清材料、两类原文读取及返回，最终分类与计数保持一致。

这些检查使用隔离数据库或模拟接口，验证接线和呈现。真实模型试用结果另记于阶段状态。
