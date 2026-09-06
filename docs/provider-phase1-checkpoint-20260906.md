# Provider / Phase 1 checkpoint（2026-09-06）

## 结论

本轮使用本机未提交的 Provider 配置，对新切换的中转与模型执行了真实调用。本文和生成报告均不记录 API key、endpoint、Prompt 或原始响应正文，也不公开可反查账户的配置值。

生产路径 preflight 成功：thinking 显式为 `disabled`，JSON 契约通过，耗时 4.76 秒，Prompt / Completion Token 为 27 / 5。这只能证明当前凭据、传输和最小 JSON 协议可用，不能证明抽取质量或 AI-first 验收通过。

随后执行冻结的 `full × 1`，即 3 个 persona 各一次。三次正式 Provider 逻辑调用均为单次尝试成功、HTTP 200，无空响应；但严格 gate 为 **0/3**。因此当前新中转/模型不能标记为通过，也没有继续执行 `full × 3` 稳定性轮次。

## 脱敏结果

| Persona | 案例耗时 | Provider 调用 | Token（Prompt / Completion） | accepted | observed/raw invalid | recovered / unresolved | empty | repair | AI 参与 / 完整覆盖 | 语义命中 | 确认冲突 | clarification | forbidden | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|
| 二游剧情策划 | 27.32 s | 1 成功 / 0 失败 | 2,391 / 1,351 | 未提供 | 25 | 0 / 25 | 0 | 未尝试 | 否 / 否 | 4 / 16 | 0 / 2 | 0 / 1 | 0 / 4 | 未通过 |
| 网络小说作者 | 16.18 s | 1 成功 / 0 失败 | 1,827 / 1,795 | 未提供 | 15 | 0 / 15 | 0 | 未尝试 | 是 / 否 | 6 / 12 | 0 / 0 | 0 / 1 | 1 / 5 | 未通过 |
| 长篇编辑/审稿人 | 19.22 s | 1 成功 / 0 失败 | 2,233 / 2,373 | 未提供 | 8 | 0 / 6 | 0 | 未尝试 | 否 / 否 | 3 / 12 | 0 / 3 | 0 / 1 | 1 / 3 | 未通过 |

`accepted` 不是当前安全诊断或验收报告提供的计数，因此明确写为“未提供”，不从最终 baseline/model 混合记录反推。`observed/raw invalid` 对应兼容字段 `invalid_records`，不是模型原始记录总数。第三案还记录了 `batch_protocol`、`batch_failed`、`circuit_open`、`lexical_support` 和 `semantic_labels_quarantined`；前两案主要是 `lexical_support` 拒绝。HTTP 200 与逻辑调用成功没有抵消这些内容/覆盖失败。

正式轮合计 Prompt / Completion Token 为 6,451 / 5,519，共 11,970；墙钟时间约 62.73 秒。三个案例均无完整模型覆盖，未进入严格指标分母，provenance 也均未通过完整验证。

默认 runner 首次真实启动时，旧计数包装器没有透传 repair provider 构造所需的 `transport`、`sleep`、`monotonic`、`wall_time` 和 `random_value`。首案主请求之后进入 repair 路径时进程中断，没有写出报告；当时通过仅存在于该 Python 进程内的只读属性转发完成正式运行。中断前额外发生的一次调用没有安全 telemetry 报告，其耗时与 Token 不可计量，也没有并入上述正式轮数字。该历史事件必须保留，不能因后续修复而追补用量。

随后代码已修复：Phase 1 与 complex runner 现在共用安全 `CountingProvider`，缺失属性会转发给底层 Provider，主调用及其派生 repair 调用都进入同一安全计数，且不保存 Prompt、响应正文、凭据或 endpoint；相关 Mock 回归已通过。因此上述进程内临时转发不再是当前运行要求，但它也不能为首次中断调用补造历史 telemetry。

正式报告写入 Git 忽略目录：`artifacts/phase1-model-acceptance-full-20260906-153423.json`。已检查报告不含本地 key/endpoint 值，也没有 `prompt`、`messages`、`raw_response`、`response_body` 或 `raw_body` 内容字段。报告不是待提交制品。

## 本轮工程 checkpoint

截至本轮，代码与自动化回归已经覆盖以下工程边界，但这些是实现状态，不是简历成绩，也不代表真实 AI gate 已通过：

- semantic trust：显式语气、来源作用域和确定性标签，证据/词面约束，非确定内容隔离，以及最终记录 provenance。
- 冻结运行输入快照、同一次运行的幂等认领、取消检查点、worker lease 与 heartbeat。
- typed diagnostics、无效记录最终处置计数，以及一次性、受限的语义标签 repair pass。repair pass 是结构化补救调用，不是 Agent。
- 批量抽取的文档归属校验、原子提交和按文档失败域；真实第三案仍表明 batch 协议兼容性需要继续修复。
- Provider 的有限重试/总 deadline、安全 telemetry、成功响应硬字节上限、错误正文不读取，以及默认能力中立、可显式设置 `thinking=disabled/enabled` 的请求配置。Compose 已把 thinking 配置同时传入 API 与 worker；空白值统一归一为 `None`，因此未配置时不会意外发送扩展字段。

当前发布判断必须以严格结果为准：**工程安全边界已推进，但新中转/模型的真实 AI-first gate 未通过，LoreGuard 仍不可声称开放文本或冻结 Phase 1 模型能力达标。**

## 与历史成绩的关系

仓库中 2026-09-02 至 2026-09-04 的 complex-v3、短文本延迟和 Provider 切换记录继续作为历史证据保留。它们使用不同的中转/模型或不同版本的完整覆盖判定，不能沿用为 2026-09-06 新中转/模型的成绩，也不能覆盖本轮 `full × 1` 严格 0/3 的结论。
