# Evidence RAG 与受限 Agent 交付收口

更新：2026-09-13。当前决定：项目已经达到投递停止线，开发可与体验、学习、讲解和投递并行，但不得再用多智能体、百万字生产化、HNSW 或更多功能推迟投递。

## 已交付的主线

### P0：后端正确性与可审计性

- 分析运行冻结文档版本和内容哈希，具备幂等认领、worker lease/heartbeat、跨进程取消检查点、失败恢复和保守旧运行展示。
- 模型抽取、固定语义标签 repair、旧受限修复 Agent、Evidence Investigator 与 Evidence Reviewer 的调用和诊断分别计量；结构化状态不依赖中文 warning 文本推断。
- Provider 超时、429/5xx、非法 JSON、响应过大、Token 预算耗尽和证据越界均有失败或降级路径，规则基线不会因模型失败消失。
- Key、endpoint、Prompt、故事正文和模型原始响应不进入安全 trace 或公开评测报告。

### P1：真实 Evidence RAG

- 独立显式配置的 OpenAI-compatible embedding client，不继承主抽取的 chat 凭据；本地固定运行时使用固定 revision 的 `BAAI/bge-small-zh-v1.5`，512 维。
- 中文行号分块、版本化 embedding profile、项目/文档/版本/内容哈希/chunker 隔离、Alembic 迁移、PostgreSQL `vector` 列和精确余弦检索均已进入真实执行。
- 检索支持 keyword-only、dense-only 与 keyword+dense RRF；索引复用不会重复生成 embedding。旧版本、错误文档、错误项目和错误 profile 被放入同库隔离验证。
- 冻结 retrieval v1 共 44 个原创中文问题。首次 28 问 holdout 中，混合检索命中 45/55 条期望证据：Recall@5 81.82%、All-evidence@5 71.43%、MRR 0.8452、低词面 Recall@5 23/29（79.31%）。运行失败、降级、隔离泄漏和三次热运行不稳定均为 0，P95 为 557.476 ms。
- Retrieval 总 gate 为 `false`：低词面召回差 1 条证据未达到预设门槛；混合检索相对 keyword/dense 的“至少提升 5 个百分点”声明也不成立。完整边界见 [首次 holdout 结果](evidence-retrieval-v1-holdout.md)。

### P2：三个边界不同的 AI 调查与复核能力

旧受限证据修复 Agent：

- LangGraph 1.2.11 `StateGraph` 编排 `READ_SPAN`、`PATCH_RECORDS`、`ABSTAIN`，使用应用层 JSON 动作而非 Provider 原生 `tool_calls`，默认关闭。
- v2 `full × 3` 只在 Agent 阶段调用真实 Provider，主抽取候选由冻结 manifest 合成注入，不是端到端抽取评测。严格正确 59/90，完整 gate 失败；这些数字只用于失败分析，不能写成 Agent 质量收益。
- 详细结果见 [v2 full checkpoint](review-agent-v2-full-checkpoint-20260906.md)。

Evidence Investigator：

- 默认关闭，使用 Provider 原生 function tool calls 在获授权冻结快照内执行 `SEARCH_EVIDENCE`、`READ_SPAN`、`SUBMIT_VERDICT` 或 `ABSTAIN`。
- 模型每个 seed 最多提交一个不可信候选；服务端重新绑定证据并执行确定性 promotion，只有既有规则能够复现的候选才可追加为 issue。
- 被评 commit `618ca991` 的两次连续 DEV 均为 8/8；首次 10 例冻结 holdout 为 8/10（TP 4、FN 1、TN 4、FP 1，precision/recall 均 80%，10/10 正常终止）。Promotion 未接受错误 Agent 候选，系统唯一 FP 来自确定性主链路。
- 该开发者编写小样本已经封存，不能外推为开放文本、生产质量或多智能体成绩。完整边界见 [真实 E2E 评测](evidence-investigator-live-evaluation.md)。

后置 Evidence Reviewer：

- `IssueEvidenceReviewer` 已经是真实 RAG 的下游消费者：规则引擎先产生候选问题，Reviewer 再对本次运行冻结的获授权快照执行 embedding、pgvector、RRF 与结构化模型复核。
- 复核只写入 `metadata.ai_evidence_review` 注释，不删除、不改写规则 issue；失败、降级、引用伪造或越界时保留原问题。
- 功能默认关闭。只有显式设置 `ENABLE_ISSUE_EVIDENCE_REVIEW=true`，并配置 PostgreSQL、embedding 和 chat Provider 后才会运行；默认要求 hybrid 检索。
- 冻结 12 例、每例 3 次的真实 A/B 中，local-context 为 4/12，rag-evidence 为 7/12；情境例外从 1/4 到 3/4，黄金证据覆盖从 0/12 到 8/12。RAG 的 106/106 条引用均在 allowlist，排除来源泄漏为 0，36 次运行无失败或降级，P95 6.445 秒。
- Evidence Reviewer 的 absolute gate 为 `false`：总体未达到 10/12，证据覆盖未达到 11/12，`insufficient_evidence` 为 0/4，仍存在把情境例外判成 `supports_issue` 的情况。完整边界见 [真实 A/B 结果](issue-review-v1-result-20260907.md)。

## 与岗位要求的对应关系

当前版本已经能够真实展示以下工程能力：

- Python/FastAPI、SQLAlchemy、PostgreSQL/pgvector、Redis/Celery、SSE、Docker Compose、CI 和 Prometheus 的端到端后端交付。
- LLM 结构化输出、失败降级、Token/超时/响应上限、证据引用校验与服务端安全边界。
- 真实 embedding、混合检索、版本/租户隔离、下游模型消费和冻结 A/B，而不是只部署向量数据库或把字符哈希称为 embedding。
- 一个使用应用层 JSON 动作的旧 LangGraph 修复循环、一个使用 Provider 原生 function tool calls 的 Evidence Investigator，以及一个独立后置 RAG Reviewer；能够解释三者输入、权限、输出和质量边界。
- 对未过门槛的结果保留真实数字和误差边界，体现实验迭代而非包装成功。

## 不再作为当前投递阻塞项

- 多智能体、HNSW/IVFFlat、图数据库、百万字生产能力、完整商业多租户、Kubernetes 与公网高并发压测。
- 把 retrieval 或 Reviewer 未过的门槛继续调到通过。当前冻结结果不得覆盖重跑；若未来依据错误分析修改 Prompt、查询或实现，必须升级数据集或另建未参与调优的测试集。
- 公网 Demo 和人工录屏可在投递过程中补充，不改变代码已经达到简历工程展示停止线的判断。

## 下一阶段

1. 用户按本地工作流亲自体验普通故事、复杂样例、DOCX、失败降级和可选 Reviewer。
2. 学习并能脱离文档讲清规则主路、RAG 后置复核、Investigator 原生工具状态机与确定性 promotion、snapshot/profile 隔离、Celery/SSE、预算与失败恢复。
3. 准备 1 分钟摘要、5 分钟架构说明、15 分钟深挖与常见追问。
4. 只依据 [项目事实清单](project-facts.md) 核对现有一页中文简历，并同步投递。

## 技术边界

- RAG 已真实完成工程闭环，但冻结检索 gate 未完全通过，不能声称“检索全面达标”或“混合检索显著优于基线”。
- Evidence Reviewer 已真实消费 RAG 证据，但绝对质量 gate 失败，不能声称“AI 复核准确率达标”或“模型质量已上线可用”。
- 旧修复 Agent、Evidence Investigator、Evidence Reviewer 和固定语义 repair 是四条不同路径；只有 Investigator 使用 Provider 原生 function tool calls，四者不得合并包装成多智能体。
- 所有结果均来自开发者可见的小型原创冻结集，不是人工盲测、开放文本或生产质量证明。
