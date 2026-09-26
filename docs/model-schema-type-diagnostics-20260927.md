# 模型抽取类型诊断：开发检查点（2026-09-27）

本次只改善抽取失败的可定位性，并针对已观察到的 `event.location=null` 调整抽取提示；**没有放宽记录准入、证据校验或角色 OOC 判定**。

## 可观察合同

- 原有 `diagnostics.model.record_rejections.schema_wrong_type` 计数保留。运行级 `diagnostics.model.schema_wrong_type_details` 和其中的 `documents[].schema_wrong_type_details` 只按固定白名单输出记录种类与字段类型族的次数，例如 `event.time`、`event.location`、`event.core_list`、`fact.core_text`；未知或混合错误归 `other`。
- 分类只在记录已被 Pydantic 拒收后进行，不保存模型原始响应、字段值或错误路径；行号、证据边界等致命错误仍沿原路径失败，不会被普通类型诊断掩盖。有效的同批记录继续独立接纳。
- 抽取提示进一步声明 `event.time`、`event.location` 必须为非空 JSON 字符串；原文没有明确地点时省略 `event`，不能填 `null`、数组、对象或编造地点。这不是自动修复模型输出，违反协议的记录仍被拒收。

## 小样本真实模型诊断

隔离运行使用此前原创、开发者可见的 `fog-harbor-explicit.md` 文本和本地配置的 `deepseek-v4-pro` 模型别名，仅输出安全计数；不保留原始回复或 Key。提示修改前，先后两次运行分别出现 `event.location` 类型拒收 4 条和 2 条。第二次仅在内存中统计 JSON 类型，看到 2 个字符串地点和 2 个 `null` 地点，因此将原先的宽泛类型错误具体定位到缺失地点。提示修改后的一次运行中，该类别为 0，仍产生一条地点冲突线索，但另有 5 条 `lexical_support` 拒收。

这些是不同模型采样结果，不是受控 A/B，更不是准确率或提示改进已稳定生效的证据。现阶段仍需分别检查词面证据拒收、模型候选覆盖和最终问题的误报/漏报；不能以类型错误计数下降推断角色 OOC 或开放剧情质量改善。

## 工程验证

Mock 回归覆盖固定白名单不泄漏、无效记录不被放行、有效同批记录保留、单文档与批量 Prompt 一致。完整回归以对应提交的 CI 为准。
