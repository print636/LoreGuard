# LoreGuard 文档中心

[产品首页（中文）](../README.zh-CN.md) · [Product overview (English)](../README.md)

首页展示产品与界面；这里按使用、部署、实现和验证整理深入说明。

## 使用 LoreGuard

| 你要完成的事 | 文档 |
| --- | --- |
| 启动项目、配置能力开关 | [中文使用与开发参考](developer-guide.zh-CN.md) · [English reference](developer-guide.en.md) |
| 了解导入、确认资料、角色基线、审查和发布 | [产品内使用指南](product-help-v1.md) · [引导式审查流程](guided-review-batch-v1.md) |
| 整理文稿、版本与资料身份 | [资料库与阅读](document-library-v1.md) · [逐文件导入与恢复](import-queue-v2.md) |
| 阅读报告与原文证据 | [冻结证据阅读](report-evidence-reader-v1.md) · [修订与复检](revision-recheck-v1.md) |
| 使用自己的模型 | [账户模型配置与密钥边界](account-model-provider.md) |

## 部署与运维

- [生产部署与公网安全边界](production-deployment.md)：账户模式、HTTPS、配置与上线检查。
- [账户与资源隔离](auth-security-contract.md)：会话、权限与安全合同。
- [备份与恢复演练](backup-restore-rehearsal.md)：数据、配置和恢复核验。
- [可观测性](observability.md)：日志、指标与运行诊断。

## 实现与扩展

- [系统架构](architecture.md)：数据流、任务执行与核心技术。
- [证据检索](evidence-retrieval-v1.md)：可选 embedding、pgvector 与混合检索。
- [角色一致性](character-consistency-v1.md)：作者确认的角色基线及实验性审查。
- [工具调用调查实验](evidence-investigator-live-evaluation.md)：获授权证据范围、候选与真实失败边界。
- [长篇扩展路线](scalability-roadmap.md)：未来方向，不代表当前已完成能力。

## 进度与质量验证

- [当前阶段状态](current-stage-status.md)：优先查阅的现状与剩余工作入口。
- [人工封闭评测协议](character-ooc-closed-evaluation-v1.md)：独立标注与质量验收方法；协议不等于已取得成绩。
- [系统端到端验证](system-e2e.md)：工程链路验证，不替代开放文本质量评测。
- [ConStory-Bench 参考边界](constory-bench-boundary.md)：外部参考、原创材料与评测口径。

## 怎样理解这些文档

产品合同说明行为和限制，评测报告只对应文中注明的构建、模型与数据集。带日期的检查点记录当时结果，不应代替[当前状态](current-stage-status.md)，也不应将合成开发集成绩外推为真实作者稿件的准确率。

原 README 的长篇技术内容已迁移至[中文](developer-guide.zh-CN.md)与[英文](developer-guide.en.md)参考；实现细节仍保留在各专题文档中。
