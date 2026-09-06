# LoreGuard 面试讲稿

## 1 分钟

LoreGuard 是给游戏编剧和叙事设计团队使用的一致性审查平台。它把原创世界观和章节文本解析成事实、事件、角色知识、物品状态与世界规则，再用确定性规则定位冲突。与普通聊天式审稿不同，每个问题都必须返回原文证据、严重度、置信度和修订建议。当前版本已打通 Markdown/TXT/JSON/标准 DOCX 导入、版本比较、五类检查、Celery、SSE、反馈、取消/重试、关系图和时间线。可选 Evidence RAG 使用固定 revision 的 BGE、pgvector 和 keyword+dense RRF 检索当前获授权版本，再由模型给规则问题追加可核验引用；AI 不能删除或改写规则结果。冻结 12 例、每例 3 次的同模型 A/B 中，严格全重复正确数由 4/12 提升到 7/12、情境例外由 1/4 提升到 3/4、黄金证据覆盖由 0/12 提升到 8/12，106/106 引用均在 allowlist 且无排除来源泄漏；但绝对 gate 失败、信息不足类为 0/4，所以我只把它作为真实 RAG 工程与误差分析证据，不声称生产准确率。

## 5 分钟结构

1. 痛点：长篇分支叙事的事实分散、角色认知随时间变化，纯向量 RAG 容易漏掉状态约束。
2. 数据流：冻结文档版本 → 结构化抽取 → 确定性规则问题 → 可选混合检索与 AI 注释 → 问题报告 → 人工反馈。
3. 关键模型：`CanonicalFact`、`NarrativeEvent`、`KnowledgeAcquisition`、`EvidenceSpan`、`ConsistencyIssue`。
4. 权衡：关系表优先、SSE 单向流、确定性基线与 LLM 语义层解耦、反馈进入评测而非直接改规则。
5. 可靠性：输入大小限制、超时/429/非法 JSON 的 provider 边界、持久化事件、取消和重试、无 Key 降级。
6. 评测：检索和复核使用冻结执行包与哈希绑定；真实结果连同失败 gate、延迟和引用隔离一起报告，不把开发者可见小样本外推为开放文本准确率。

## 高频追问

- 当前有 RAG 吗？有，但它是默认关闭的后置 Evidence RAG，不替代确定性规则主路。固定 BGE embedding 写入 PostgreSQL/pgvector，keyword+dense RRF 按项目、文档、版本、内容哈希、chunker 和 profile 隔离检索，再由 `IssueEvidenceReviewer` 消费证据并追加模型注释。首次 retrieval holdout 的混合 Recall@5 为 81.82%、完整证据命中为 71.43%，低词面 79.31% 差 1 条未过门槛；真实 Reviewer A/B 为 4/12 → 7/12，但 absolute gate 失败且信息不足类 0/4。因此可以说真实 RAG 工程闭环已经接通，不能说质量全面达标。规则仍保留，因为向量相似不等于时序与状态冲突。
- 当前有 Agent 吗？有一个默认关闭的第一阶段受限 Agent：LangGraph `StateGraph` 编排模型在 `READ_SPAN`、`PATCH_RECORDS`、`ABSTAIN` 三种应用层 JSON 动作间选择，服务端控制文档窗口、span、补丁字段、轮次、Token 和 deadline。它不是 Provider 原生 `tool_calls`，没有多智能体，也不是 Evidence RAG。v2 Agent 阶段真实 Provider full 完成 90/90 次要求执行，但主抽取候选为冻结合成注入，holdout 仅 74/81 运行成功，可恢复题 26/51、主动弃答 24/30，且没有直接 `ABSTAIN` 成功路径，完整 gate 失败，因此不能声称 Agent 或端到端抽取质量收益。
- 真实 Agent 验收说明了什么？90 次执行中严格正确 59 次。31 次失败分成互不重叠的四组：12 次被生产校验接受但不符合 oracle 的错误补丁、9 次可恢复题读取后过度弃答、7 次 `read_timeout`、3 次 outcome 正确但未覆盖 oracle 证据。只有后 3 次是 `trace_replay=false/read_misses_oracle_evidence`，timeout trace 都可重放。服务端接受的安全违规为 0，说明边界在本套件中没有被突破，但不说明输出质量达标。development/tuned 的 9/9 不能代表泛化，失败的 full 结果也不能包装成简历成绩。
- 如何控制误报？权威来源优先级、时间范围、实体消歧、阈值校准和人工反馈分层统计。
- 如何控制成本？目前有逐分块 Token 门控、有限重试与熔断，已完成报告读取不再调用模型。跨运行缓存和共享原子访客额度尚未完成，不能把当前单实例每日检查当作生产配额保障。
- SSE 断线怎么办？事件落库，客户端携带最后事件编号重连；终态可用状态接口补查。
- 如何防 Prompt 注入？故事文本和工具输出始终视为数据；受限 Agent 的 `READ_SPAN` 由服务端绑定候选、冻结文档、允许窗口和一次性 span，`PATCH_RECORDS` 只能修改按 kind 给出的字段白名单并重新经过证据与语义校验，失败即弃答。响应字节、轮次、工具数、Token 和 deadline 都有上限，安全 trace 不保存 Prompt、原始响应、正文或补丁值。这些边界降低了工具越权风险，但不等于已经证明能抵御所有 Prompt 注入。
