# 受限证据修复 Agent（第一阶段）

状态：代码已接入，默认关闭，尚未通过冻结真实模型评测，因此不能写成默认交付能力或已证明收益。

## 为什么这是 Agent，而不是把固定流程改名

LangGraph 1.2.11 `StateGraph` 只负责编排 `decide -> execute -> decide/finalize` 循环。Agent 行为来自模型在每轮根据服务端校验原因和工具结果动态选择以下应用层 JSON 动作：

- `READ_SPAN`：一次动作可批量读取多个候选附近的有界原文；每个 request 仍独立绑定候选和冻结文档；
- `PATCH_RECORDS`：用先前读取返回的服务端 `span_id` 提交字段补丁；
- `ABSTAIN`：证据不足时明确弃答。

当前 Provider 未验证原生 `tool_calls`，所以这里准确称为“应用层 JSON tool protocol”，不冒充原生 function calling。第一轮提示包含模型候选的核心字段，但不含 `READ_SPAN` 证据正文或文档全文；模型必须先调用 `READ_SPAN`，取得绑定候选、文档、行范围和内容哈希的不可猜测 `span_id`，下一轮才能提交补丁。这使工具结果真实参与后续决策。

## 信任边界

只有已通过响应 envelope、批量 `doc_ref` 映射、服务端 role/scope 归属、Pydantic 核心 schema、证据行范围和非空证据检查，但仍失败于 `lexical_support` 的候选会进入 Agent。仅缺失或写错语义标签的候选继续使用固定 semantic repair pass，不能记作 Agent 经验。

冻结 Agent manifest 的 30 个初始候选现已逐条通过同一生产准入路径验证：结构、文档归属、行号、非空证据和语义三标签均合法，并且只在 `lexical_support` 失败。问句、引用、不确定陈述与互斥分支负例通过“不受原文支持的核心字段 + 合法保守标签”测试 Agent 是否安全弃答；测试不会把旧的 `semantic_quality`、`unsupported_claim` 或标签缺失原因改名冒充词面失败。Oracle、允许证据行和期望补丁仍只供运行后评分，不进入 Agent 输入。

`PATCH_RECORDS` 不能更改 `kind`、`doc_ref`、证据行号、role 或 scope。每个补丁必须携带本次 Agent、同一候选、同一文档且覆盖原证据行的有效 `span_id`，然后重新通过：

1. kind 对应的字段 allowlist；
2. Pydantic `ExtractionRecord`；
3. 原文证据范围与 `lexical_support`；
4. `semantic_quality` 和确定性规则资格重算。

一个 `PATCH_RECORDS` 动作内任意记录失败时全部不提交。伪造、跨运行复用、跨候选复用或跨文档使用 `span_id` 均失败关闭。

## 有界执行和审计

- 整个 analysis run（不是每个文档或每次修复）共享最多 2 个模型决策轮、总计 6 次工具动作；批量 `READ_SPAN` 按一个工具动作计数，内部读取数另记为 `span_read_count`，后续文档不能通过重新创建 Agent 重置额度；
- 单个 READ 动作的 request 数、整次运行的 span 数、单次读取行数、候选附近半径和全部 span 字符数均有硬上限；批量请求中一项越界或跨文档会在读取前原子拒绝；
- 同时受运行剩余 Token、Agent Token、Provider 超时、总 deadline 和取消检查点约束；共享 Agent deadline 在第一次 Agent 调用时启动，主抽取耗时不提前消耗它；
- Provider 已返回后才到达的取消仍会先写入 `ModelEnhancedExtractor` 的内容无关调用/Token 安全账本，再传播取消；service 在取得终态所有权后将已完成 Agent 调用的已知用量和安全遥测原子持久化，并显式标记为 `lower_bound` / `review_agent_completed_calls_only`，不能冒充完整运行用量；其中 `charged_tokens` 另标为保守的内部预算扣减，不冒充 Provider 实际用量；取消后不会继续执行工具，终态重投也不会重复记账；
- 未知工具、非法 JSON、重复循环、越界、跨文档、超预算、超时和补丁重验失败都不会产生确定性记录；
- `invalid_records = recovered_invalid_records + unresolved_invalid_records` 仍是最终诊断约束。

持久化 trace 只包含动作、轮次、候选哈希、服务端 `doc_ref`、行号、span 哈希、补丁字段名、校验原因、Token、耗时和最终状态。不保存思维链、Prompt、模型原始响应、故事/工具正文、补丁值、密钥或 Provider endpoint。取消态安全账本还会无条件丢弃 Provider 控制的原始 request ID，并对聚合 Token、调用次数及逐调用字符/耗时计数应用配置派生的有限上限；聚合计数越界时整份账本不可用，但不能阻止取消终态落库，只有逐调用可选遥测越界时则将该序列标为不可用。诊断最多保留 8 次 Agent run、每次 64 个 trace 事件；发生截断时同时输出原始总数和显式 `*_truncated=true`，不能把截断后的轨迹表述为完整轨迹。

## 启用方式与下一门槛

开发环境显式设置 `ENABLE_REVIEW_AGENT=true` 才会启用。当前仅完成 Mock 动态路径和边界回归，尚未运行真实模型 Agent 验收。启用不代表达到简历表述门槛；仍需完成冻结任务的真实模型动态路径、恢复率、弃答安全、降级率、延迟与成本对照评测。评测前的准确表述是“实现并在 Mock 中验证了受限工具循环，默认关闭”。
