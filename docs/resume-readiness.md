# 简历就绪结论

更新：2026-09-07。

## 结论

LoreGuard 已达到当前简历项目的工程展示停止线，建议立即从“继续开发”切换到“亲自体验 → 学习理解 → 项目讲解 → 更新简历 → 投递”。这不是宣告产品或模型质量全面达标，而是确认已经有足够真实、可验证且与 AI 应用后端岗位相关的工程证据；继续堆叠多智能体、百万字能力或更多基础设施的边际收益低于开始准备投递。

## 为什么现在可以进入投递准备

- 产品端到端可运行：多格式导入、版本化文档、异步分析、SSE、证据化 issue、反馈、图和时间线均已接通。
- 后端工程链路完整：FastAPI、SQLAlchemy、PostgreSQL、Redis/Celery、Docker Compose、CI、Prometheus、限流、预算、重试与降级。
- AI 能力不是占位：真实 BGE embedding 已写入 PostgreSQL/pgvector，三种检索策略已运行冻结 dev/holdout；真实 RAG 证据已经被后置模型 Reviewer 消费。
- 安全边界可解释：项目/文档/版本/内容哈希/profile/chunker 隔离，服务端引用 allowlist，失败不覆盖规则结果，凭据和敏感正文不进入公开报告。
- 有诚实的失败结果：retrieval 低词面门槛差 1 条；Reviewer absolute gate 失败且信息不足类为 0/4。能解释错误比继续包装功能更有面试价值。

## 当前真实指标

### Evidence Retrieval v1 首次 holdout

- 28 问、55 条期望证据。
- Keyword + Dense RRF：Recall@5 81.82%，All-evidence@5 71.43%，MRR 0.8452，低词面 Recall@5 79.31%。
- 三次热运行 P95 557.476 ms；失败、降级、隔离泄漏和排名不稳定均为 0；复用阶段新增 embedding 与调用均为 0。
- 总 gate 为 `false`：低词面 23/29，差 1 条未过门槛；也没有达到相对 keyword 和 dense 各提升 5 个百分点的收益声明条件。

### Issue Review v1 真实 A/B

- 12 例，每例 3 次；local-context 4/12，rag-evidence 7/12。
- 情境例外 1/4 → 3/4；黄金证据覆盖 0/12 → 8/12。
- RAG 引用 106/106 在 allowlist，排除来源泄漏 0；36 次无失败/降级；P95 6.445 秒。
- Absolute gate 为 `false`：未达到总体 10/12 与覆盖 11/12，仍有情境例外误判为 `supports_issue`，`insufficient_evidence` 为 0/4。

### 受限修复 Agent

- 默认关闭的 LangGraph `StateGraph` 应用层动作循环已经实现。
- v2 `full × 3` 的 Agent 阶段真实 Provider 评测严格正确 59/90，但完整 gate 失败，且主抽取候选为冻结 manifest 合成注入。
- 只可作为协议、安全边界和失败分布证据，不作为 Agent 质量收益或端到端抽取成绩。

## 简历表述门槛

可以写实现事实和带边界的测量结果：

- 真实 embedding、pgvector 精确检索、RRF、snapshot/profile 隔离和索引复用。
- RAG 下游 Evidence Reviewer、严格引用校验、规则结果不可变、失败降级。
- Celery/SSE、幂等认领、lease/heartbeat、Token/timeout/response cap 与 Docker/CI。
- Retrieval holdout 的 81.82% Recall@5 和 71.43% All-evidence@5，但必须在项目事实库或面试中保留低词面 79.31% 未过门槛的边界。

不能写：

- “RAG 全面达标”“混合检索显著优于基线”“AI Reviewer 准确率达到上线标准”。
- “模型可靠识别信息不足”；真实结果是 `insufficient_evidence` 0/4。
- 人工盲测、开放文本准确率、商业游戏长篇验证、生产高并发、原生 function calling 或多智能体。
- 受限 Agent 或 Reviewer 的质量收益已经通过 gate。

## 用户验收与学习门槛

更新简历前，用户应亲自完成：

1. 从新建项目、上传普通故事或 DOCX 到生成问题报告，再查看证据、Reviewer 注释、图/时间线和提交反馈。
2. 在关闭模型、模型失败和显式开启 Reviewer 三种状态下，说明页面结果为什么不同。
3. 脱离文档讲清：规则主路与 RAG 后置通路、embedding profile、版本隔离、RRF、Celery/SSE、引用校验、预算和失败恢复。
4. 能主动回答两项失败：为什么低词面少 1 条、为什么 Reviewer 的 `insufficient_evidence` 为 0/4，以及下一版如何用新冻结集验证而不污染本次结果。
5. 准备 1 分钟项目摘要、5 分钟架构讲解和 15 分钟深挖。

完成以上学习验收后，只依据 [项目事实清单](project-facts.md) 更新一页中文简历并开始投递。公网 Demo、演示视频和后续体验优化可以滚动补充，不再阻塞投递。
