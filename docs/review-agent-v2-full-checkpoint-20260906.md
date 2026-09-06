# 受限审查 Agent：v2 Agent 阶段真实 Provider full 脱敏 checkpoint

日期：2026-09-06  
代码 checkpoint：`bcbfab8`

## 声明边界

本页只保存 v2 `full × 3` 的脱敏聚合结果，不保存 Provider endpoint、密钥、Prompt、模型原始响应、故事正文、工具正文或补丁值。v2 的任务、文档和期望均为开发者可见；它不是人工盲测，也不能外推到开放中文故事、商业剧情或生产准确率。

本次评测只在受限 Agent 决策阶段调用真实 Provider。主抽取没有调用真实模型，而是由 runner 按冻结 manifest 合成注入 `initial_candidate`，随后才进入真实 Agent 调用和生产补丁校验。因此本页数字只能评价该冻结候选上的 Agent 阶段行为，不能评价端到端模型抽取质量，也不能作为完整产品端到端真实模型成绩。

当前实现是 LangGraph `StateGraph` 编排的受限应用层 JSON 动作循环。模型只能选择 `READ_SPAN`、`PATCH_RECORDS`、`ABSTAIN`，服务端限制冻结文档、读取窗口、span、补丁字段、轮次、工具数、Token 和 deadline，并重新验证补丁。它没有使用或验证 Provider 原生 `tool_calls/tool_call_id`，没有多智能体，也没有 embedding/pgvector Evidence RAG。

## 执行覆盖

| 分组 | 要求执行 | 已记录执行 | 运行成功 | 运行失败 |
| --- | ---: | ---: | ---: | ---: |
| full | 90 | 90 | 83 | 7 |
| holdout | 81 | 81 | 74 | 7 |
| development/tuned | 9 | 9 | 9 | 0 |

holdout 的 7 次运行失败均为 `read_timeout`。`90/90` 表示要求的 execution 均有终态记录，不表示 90 次均运行成功或质量正确。

## Holdout 质量与安全

| 指标 | 结果 |
| --- | ---: |
| 可恢复题正确恢复 | 26/51（0.509804） |
| 不可恢复题主动弃答正确 | 24/30（0.8） |
| full 严格正确 | 59/90 |
| 被生产校验接受但不符合 oracle 的错误补丁 | 12 |
| `oracle_patch_mismatch` | 9 |
| `oracle_unrecoverable_patch` | 3 |
| 服务端接受的安全违规 | 0 |

full 的 31 次严格失败由以下四组组成，彼此不重叠：

| 失败类型 | 次数 |
| --- | ---: |
| 被生产校验接受但不符合 oracle 的错误补丁 | 12 |
| 可恢复题读取后过度弃答 | 9 |
| `read_timeout` | 7 |
| outcome 正确但读取未覆盖 oracle 证据 | 3 |

最后 3 次的 `trace_replay=false` 原因全部是 `read_misses_oracle_evidence`。7 次 timeout 的 trace 均可重放；不能把 trace replay 失败归因于 timeout。

这 12 次错误补丁通过了生产安全/结构校验，但未满足评测 oracle，因此是输出质量失败，不是安全 containment 成功。`accepted safety violations = 0` 只证明本套件中未观察到服务端接受越界安全行为，不证明 Agent 输出质量达标，也不证明系统抵御所有 Prompt 注入。

development/tuned 子组的 9/9 次执行均达到预期，恢复率和弃答率均为 1.0。该子组已经参与标签、Prompt 和 scorer 的开发，不能替代 holdout 或作为泛化成绩。

## Gate 结论

本次 full 的最终结果为 `passed=false`。主要事实是：

- Agent 阶段真实 Provider 调用和生产补丁校验已为全部 90 次要求执行留下终态记录，严格正确 59/90；
- 服务端没有接受本套件中的安全违规；
- holdout 存在 7 次读取超时；
- 另有 3 次虽然 outcome 正确，但读取没有覆盖 oracle 证据，不能通过严格 trace gate；
- 运行成功轨迹中没有直接 `ABSTAIN` 成功路径，要求的三路径覆盖 gate 未通过；
- holdout 恢复率和弃答率不足以形成 Agent 模型质量或产品收益声明；
- development/tuned 的满分不能弥补 holdout 失败。

因此当前允许的表述是“实现了默认关闭的 LangGraph 受限 JSON 动作 Agent，并在冻结合成注入候选上完成一次 Agent 阶段真实 Provider full 失败诊断，服务端对越界补丁保持失败闭合”。不得表述为“完成端到端真实模型抽取评测”“Agent 已通过验收”“Agent 提升了准确率/召回率”或把 development/tuned 结果当作独立测试成绩。

## 后续使用规则

1. 本次结果用于分析通用超时、恢复和坏补丁模式；不得针对单个任务写死答案。
2. 如果本次 holdout 结果被用于修改 Prompt、实现或标签，后续同一套件只能作为开发回归。
3. 新的泛化质量声明需要另行冻结未参与调优的测试集，并继续公开运行失败、恢复、主动弃答、坏补丁、安全违规、延迟和 Token 成本。
4. Evidence RAG 使用独立检索评测；本次 `READ_SPAN` 邻域读取不能作为 RAG 证据。
