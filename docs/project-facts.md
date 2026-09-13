# 项目事实清单

更新：2026-09-13。本文件是简历和面试表述的唯一保守事实源。计划、Mock 结果和未通过门槛不得改写成完成成果。

## 可在简历中表述

- 独立完成面向游戏编剧与叙事设计的剧情一致性审查平台，支持 Markdown、TXT、JSON、DOCX 导入、文档版本管理、异步分析、证据化问题报告、反馈审计、关系图和保守时间线。
- 使用 FastAPI、SQLAlchemy、PostgreSQL/pgvector、Redis/Celery、React/TypeScript、Docker Compose、GitHub Actions 和 Prometheus 构建端到端工程链路。
- 设计 OpenAI-compatible 模型网关与 Pydantic 结构化输出合同；实现超时、429/5xx、非法 JSON、响应大小、Token 预算、取消与降级处理，模型失败时保留确定性基线。
- 实现固定 revision 的中文 BGE embedding、行号感知分块、版本化 embedding profile、PostgreSQL 精确余弦检索和 keyword+dense RRF；按项目、文档、版本、内容哈希、chunker 与 profile 隔离，并验证索引复用不重复调用 embedding。
- 首次冻结 retrieval holdout 含 28 问、55 条期望证据；混合 Recall@5 45/55（81.82%）、All-evidence@5 20/28（71.43%）、MRR 0.8452，三次热运行 P95 557.476 ms，隔离泄漏、失败和降级均为 0。必须同时说明低词面 Recall@5 为 23/29（79.31%），差 1 条未过门槛，整体 gate 为 `false`。
- 实现默认关闭的 `IssueEvidenceReviewer`，将获授权的 RAG 证据交给模型复核，并用服务端签发引用与 allowlist 校验；结果只作为 `ai_evidence_review` 注释持久化，不删除或改写规则 issue。
- 真实 12 例 × 3 次 A/B 中，local-context 4/12，rag-evidence 7/12；情境例外 1/4 → 3/4、黄金证据覆盖 0/12 → 8/12，RAG 引用 106/106 在 allowlist，0 排除来源泄漏，36 次无失败/降级，P95 6.445 秒。只有在同时说明 absolute gate false、`insufficient_evidence` 0/4 时才可引用这些数字。
- 实现默认关闭的 LangGraph 1.2.11 `StateGraph` 受限修复循环，使用应用层 JSON 动作 `READ_SPAN`、`PATCH_RECORDS`、`ABSTAIN`，并限制范围、字段、轮次、Token、deadline 与响应字节；它不是原生 `tool_calls`。
- 实现默认关闭的 Evidence Investigator，使用 Provider 原生 function tool calls 在获授权冻结快照内执行 `SEARCH_EVIDENCE`、`READ_SPAN`、`SUBMIT_VERDICT` 或 `ABSTAIN`；模型每个 seed 最多提交一个不可信候选，只有确定性 promotion 能将其转为 issue。
- 被评 commit `618ca991` 上两次连续 DEV 均为 8/8；首次 10 例冻结 holdout 为 8/10（TP 4、FN 1、TN 4、FP 1，precision/recall 均 80%，10/10 正常终止）。Promotion 未接受错误 Agent 候选，系统唯一 FP 来自确定性主链路。该开发者编写小样本已封存，不得外推为开放文本或生产质量。
- 实现运行输入冻结、幂等认领、worker lease/heartbeat、跨进程取消检查点、持久化 SSE 续传与安全诊断；不能据此宣称生产高并发或 exactly-once 模型计费。
- 本机 Compose 实测 2 个 Celery worker 处理 20 个相同冻结输入任务：20/20 完成、0 失败、问题集合一致，约 9.7 秒收敛；只能称为队列与隔离烟测，不能称为高并发压测或 SLA。

## 面试时必须主动说明

- 确定性主检查器直接生成基线 issue；Evidence Investigator 候选必须通过确定性 promotion 才能追加问题；Evidence Reviewer 在 issue 之后运行，只追加模型证据意见，不删除或改写问题。
- `retrieval.py` 的稳定字符 n-gram 是确定性主路的本地候选信号；真实 BGE/pgvector/RRF 是独立 Evidence RAG 通路，两者不能混称。
- Retrieval holdout 只差低词面 1 条，但仍是 gate 失败；混合 Recall 与 dense 相同，不能声称显著优于两种基线。
- Reviewer A/B 显示 RAG 对情境例外和证据覆盖有帮助，但总体 7/12，信息不足类 0/4，不能声称模型质量达标。
- 全部评测集均为原创、开发者可见、非人工盲测；指标不能外推到开放故事、商业游戏剧情或生产环境。
- 旧 LangGraph 修复 Agent、Evidence Investigator、Evidence Reviewer 和固定语义 repair 是四条独立机制；只有 Investigator 使用 Provider 原生 function tool calls，它们不互相编排，项目没有多智能体。
- Investigator 的 DEV pair 只是 holdout 前的资格门；8/10 holdout 是封存小样本历史结果，不是 Agent 质量或生产可用性全面通过。

## 不可表述为已完成

- 公网在线 Demo、真实用户数据验证、人工盲测、商业游戏语料验证、长篇生产容量或高并发 SLA。
- Retrieval 全门槛通过、混合检索显著领先、Evidence Reviewer 质量达标或模型能够可靠识别信息不足。
- 旧修复 Agent 的质量收益、端到端真实模型抽取收益，或把它表述为原生 `tool_calls`。
- Investigator 的开放文本准确率、生产质量、多智能体协作或端到端抽取收益；HNSW、图数据库、Kubernetes 或完整生产级 OpenTelemetry。
- 任何 API Key、服务地址、Prompt、原文、note、原始响应或内部运行产物。

## 投递停止线与后续

代码已达到当前简历工程展示停止线，可立即开始投递。开发可与亲自体验、学习理解和 1/5/15 分钟讲解准备并行，但不得再成为延迟投递的理由。Investigator 的首次 holdout 已封存；若根据其结果改动 Prompt、规则、阈值或 promotion，只能将该结果作为历史记录，并用新冻结集重新估计质量。
