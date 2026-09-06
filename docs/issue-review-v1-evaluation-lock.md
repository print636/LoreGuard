# Issue Review v1 真实模型 A/B 锁定记录

本记录在首次真实模型 A/B 运行前提交。`issue-review-v1` 是开发者可见的 12 例原创冻结小样本，不是人工盲测、开放文本或生产质量证明。

## 固定身份

- 评测实现提交：`fb57f30a2ffff0fb5816da785cba3f706fbb6127`。相对首次锁定版本仅增加了经审查的显式进程环境凭据通道；未修改数据、Prompt、模型合同、检索配置、评分或门槛。
- Manifest SHA-256：`2e216d6452917d08dec0b909012dde6fe39caa1b967847b79f49a204c7febcee`
- Freeze SHA-256：`d2e418072ecd51ceb8a3183edebed04b7175b42d736a55573741a05a7592f271`
- Full execution SHA-256：`790bdef30c16531199d7a9890edfab75c9c553b63dc0c53fe6279acc14668dca`
- 聊天模型：`deepseek-v4-pro`，thinking disabled，温度由生产 Provider 固定为 0
- Embedding：固定 revision 的 `BAAI/bge-small-zh-v1.5`，512 维，PostgreSQL/pgvector 精确余弦
- 两组模式：`local-context` 与 `rag-evidence`
- 每例重复：3 次；两组都使用同一聊天模型与结构化输出合同
- RAG Top-K：6；单次复核 Token 预算：6000；请求超时：20 秒；整条复核 deadline：45 秒

聊天服务地址只以不可逆 fingerprint 进入报告；API Key、地址、Prompt、原文、模型原始响应和 note 不进入公开报告。真实运行通过隔离容器的进程环境注入现有凭据，不生成临时密钥文件；评测程序不读取 `.env`。

## 运行纪律

本次直接运行完整 12 例，其中包含预先冻结的 6 个 dev 与 6 个 holdout；运行前不根据模型输出修改 Prompt、数据、阈值或检索配置。两组各运行一次 `full × 3`，随后离线评分和对比。无论是否通过，都保留首次结果和哈希，不以重跑覆盖失败。

`local-context` 只提供候选问题已有片段；`rag-evidence` 必须走生产 `IssueEvidenceReviewer`、真实 embedding、pgvector 和严格授权快照。降级到关键词、模型失败、引用越界、缺少答案或重复间不稳定都不能计为正确。

## 预设门槛

- 总体 verdict 至少 10/12，且要求每例 3 次均正确。
- `contextual_exception` 至少 3/4，错误标成 `supports_issue` 的数量为 0。
- 引用 allowlist 命中率 100%，排除证据泄漏为 0。
- 黄金证据覆盖至少 11/12，且要求每例 3 次均覆盖。
- RAG 相比 local-context 至少多答对 2 例，并在 `contextual_exception` 上至少多答对 2 例。

如果任一门槛失败，只能将结果作为工程诊断和误差分析，不得写成模型质量收益。
