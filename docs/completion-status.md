# 当前完成度与诚实边界

更新：2026-09-13。

LoreGuard 已达到当前“AI 应用后端/游戏平台研发”简历项目的工程展示停止线。代码主线已覆盖真实 embedding、pgvector 检索、后置模型消费、基于 Provider 原生 function tool calls 的证据调查、异步任务、SSE、失败降级、安全引用与冻结评测。开发可与本人体验、学习、讲解和投递并行，但不得阻塞投递。这里的“达到停止线”不代表所有质量 gate 通过，也不代表商业生产就绪。

## 已完成并进入回归的能力

### 产品与后端闭环

- React/Vite 工作台与 FastAPI API；Markdown、TXT、JSON、DOCX 多文件导入。
- 项目、文档版本、冻结运行输入、运行历史、问题、反馈审计、关系图、保守时间线和同名文档版本差异。
- 本地线程与 Redis/Celery 两种执行模式；任务认领、lease/heartbeat、取消、重试、失败恢复、持久化 SSE 断线续传。
- PostgreSQL/SQLite 数据层、Alembic、Docker Compose、GitHub Actions、Prometheus 和结构化诊断。

### 确定性规则与模型抽取

- 无模型时，BaselineExtractor 与规则引擎仍可独立生成证据化问题报告。
- 配置模型后，OpenAI-compatible Provider 抽取 8 类状态记录；Pydantic 严格校验并由服务端按全局行号回填证据。
- 超时、429/5xx、空响应、非法 JSON、响应过大、无效记录、Token 预算和取消均有明确失败/降级路径；模型失败不删除基线结果。
- 五类确定性问题、显式别名、角色知识来源、规则例外、移动许可、物品使用和持续状态等已有自动回归。开发者可见的固定回归不得外推为开放文本准确率。

### 真实 Evidence RAG

- 独立 embedding Provider、固定 BGE 模型 revision、中文行号分块、版本化 profile 和精确 snapshot schema 已实现。
- PostgreSQL 使用真实 `vector` 列与精确余弦检索；keyword-only、dense-only、keyword+dense RRF 三种策略走同一冻结输入与 Top-5 合同。
- 项目、文档、版本、内容哈希、chunker 和 profile 全部进入检索过滤；旧版本、错误文档、错误项目和错误 profile 与合法数据同库验证，未发生越界返回。
- 首次冻结 holdout：28 问、55 条期望证据。混合 Recall@5 为 45/55（81.82%），All-evidence@5 为 20/28（71.43%），MRR 0.8452，低词面 Recall@5 为 23/29（79.31%）；失败、降级、隔离泄漏和三次热运行排名不稳定均为 0，P95 557.476 ms。
- 总 gate 为 `false`，唯一绝对门槛失败项是低词面召回差 1 条证据；混合检索相对 keyword/dense 的预设增益声明也不成立。详见 [Evidence Retrieval v1 Holdout](evidence-retrieval-v1-holdout.md)。

### 后置 Evidence Reviewer

- `IssueEvidenceReviewer` 在确定性规则已经产出 issue 后运行，将获授权的 RAG 证据交给结构化 Chat Provider 复核。
- Reviewer 只追加 `metadata.ai_evidence_review`；不会删除或修改规则 issue 的类别、严重度、证据、解释或建议。模型、检索或引用校验失败时，规则报告原样保留。
- 功能由 `ENABLE_ISSUE_EVIDENCE_REVIEW=true` 显式开启，默认关闭；需要 PostgreSQL、真实 embedding 与 Chat Provider，默认要求 hybrid 检索。
- 冻结 12 例、每例 3 次的真实 A/B：local-context 4/12，rag-evidence 7/12；情境例外 1/4 → 3/4，黄金证据覆盖 0/12 → 8/12。RAG 的 106/106 条引用通过 allowlist，排除来源泄漏为 0，36 次运行无失败或降级，P95 6.445 秒。
- Absolute gate 为 `false`：总体未达 10/12、覆盖未达 11/12，且 `insufficient_evidence` 为 0/4。详见 [Issue Review v1 结果](issue-review-v1-result-20260907.md)。

### 旧受限修复 Agent

- LangGraph 1.2.11 `StateGraph` 受限循环提供 `READ_SPAN`、`PATCH_RECORDS`、`ABSTAIN`，默认关闭；这是应用层 JSON 动作，不是 Provider 原生 `tool_calls`。
- v2 `full × 3` 只对冻结合成候选执行真实 Agent 阶段，并非端到端真实模型抽取。严格正确 59/90，完整 gate 失败；服务端接受的安全违规为 0。
- 这条 LangGraph 修复路径不是 Provider 原生 function calling，也不能把它的失败评测改写成质量收益。

### Evidence Investigator

- 默认关闭；使用 Provider 原生 function tool calls 在获授权冻结快照内执行 `SEARCH_EVIDENCE`、`READ_SPAN`、`SUBMIT_VERDICT` 或 `ABSTAIN`。每个 seed 最多提交一个不可信候选，只有确定性 promotion 能将其转为 issue。
- 被评 commit `618ca991` 上两次连续 DEV 均为 8/8；首次 10 例冻结 holdout 为 8/10（TP 4、FN 1、TN 4、FP 1，precision/recall 均 80%，10/10 正常终止）。
- Promotion 未接受错误 Agent 候选，系统唯一 FP 来自确定性主链路。该开发者编写小样本已封存，不得外推为开放文本、生产质量或多智能体成绩。
- 固定语义 repair、旧修复 Agent、Investigator 与 Evidence Reviewer 是四条独立路径；它们不互相编排，项目没有多智能体。

## 当前验证状态

- 自动测试、前端构建和 Compose 主链路已经覆盖核心行为；具体数量随提交变化，不用测试数量代替产品质量。
- 本机已经真实运行固定 TEI/BGE embedding、PostgreSQL/pgvector、三策略 retrieval holdout、Evidence Reviewer A/B，以及 Investigator 的两次 DEV 资格运行与首次冻结 holdout。
- 最新 Compose 端到端验收中，内置原创样例产生 5 类规则问题，5/5 都得到真实 `keyword+dense-rrf`、模型 verdict 与获授权引用；索引 2 个文档，Reviewer 无跳过、拒绝批次或降级，随后反馈写入和历史读取成功。
- 关闭模型后使用 2 个 Celery worker 并发提交 20 个相同冻结输入任务，20/20 完成、0 失败且问题集合一致，约 9.7 秒收敛。该结果只证明本机队列与状态隔离烟测，不是生产高并发 SLA。
- API Key、服务地址、Prompt、原文、note 与模型原始响应不进入公开结果文档或提交记录。
- 所有质量数据均来自开发者可见的原创冻结集，不是人工盲测、真实用户语料或生产准确率。

## 简历可以写

- 完成 FastAPI、SQLAlchemy、PostgreSQL/pgvector、Redis/Celery、SSE、Docker Compose、CI、Prometheus 的全栈工程闭环。
- 实现真实 BGE embedding、精确余弦检索、RRF 融合、版本/profile 隔离、索引复用和安全降级。
- 将 RAG 证据接入后置模型 Reviewer，并通过服务端引用 allowlist 保证只消费获授权快照；Reviewer 只做注释，不覆盖规则结果。
- 建立冻结评测与 A/B，诚实报告 retrieval holdout 的 81.82% Recall@5、71.43% All-evidence@5，以及低词面 79.31% 差一条未过门槛。
- 实现默认关闭的受限 LangGraph 动作循环，但不把它包装为原生 function calling 或多智能体。
- 实现默认关闭的 Evidence Investigator，以 Provider 原生 function tool calls 执行受限检索、读取和单候选提交，再由确定性 promotion 决定是否追加 issue。

## 不能写成已完成成果

- “混合检索显著优于关键词和 dense 基线”“retrieval 全部门槛通过”。
- “Evidence Reviewer 达到 10/12 或生产可用”“模型能可靠判断信息不足”；真实结果是 7/12、`insufficient_evidence` 0/4、absolute gate false。
- 人工盲测、开放文本准确率、长篇商业剧情验证、高并发 C 端生产容量或公网 SLA。
- 多智能体、HNSW/IVFFlat、图数据库、完整 OpenTelemetry 或生产多租户。
- 旧受限修复 Agent 的质量收益或端到端抽取成绩；其 gate 失败且主抽取候选为冻结合成注入。
- 把旧修复 Agent 说成原生 function calling，或将 Investigator 的 10 例 holdout 外推为 Agent 质量全面达标、端到端抽取收益或生产可用。

## 当前停止线

项目已达到投递停止线，用户可在亲自体验、理解代码和架构、复述真实权衡的同时继续开发与投递，不得再以扩功能阻塞投递。未来优化应由真实体验或新冻结数据驱动；Investigator 首次 holdout 只作为封存历史结果，不用于后续调参。
