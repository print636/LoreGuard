# 修订—复检闭环 V1（后端契约）

LoreGuard 可以从一个已完成且具有冻结输入的分析任务出发，对项目当前活动文档创建新快照并发起复检。复检不是原任务的 `retry`：`retry` 重用原快照，`recheck` 则冻结修改后的当前版本。

## API

### 发起复检

`POST /api/v1/analysis-runs/{baseline_run_id}/rechecks`

- 支持 `Idempotency-Key`，作用域为当前项目。
- 基准任务必须属于当前工作区、状态为 `completed` 且具有可用快照。
- 当前活动文档相对基准快照至少有一个版本、内容或文档集合变化。
- 返回 `202`，主体包含目标 `id`、`baseline_run_id`、`comparison_id`、`comparison_status` 与 `deduplicated`。
- 同一键只能表达同一次复检；复用于普通分析、重试或另一基准时返回 `409 idempotency_key_conflict`。

### 读取对比

`GET /api/v1/analysis-runs/{target_run_id}/comparison`

查询参数：

- `outcome`：可选，`no_longer_detected`、`persisting`、`new` 或 `unverifiable`。
- `limit`：1–200，默认 100。
- `offset`：默认 0。

分页在 outcome 过滤后执行，`page.total` 与 `page.has_more` 也是过滤后口径；全量 `summary` 不受过滤影响。

## 四种机器结论

- `persisting`：新旧运行中找到了唯一的一对一规则身份或证据签名。
- `no_longer_detected`：在两次运行可比且没有身份歧义的前提下，新运行未再检出旧问题。它不是人工确认的“已修复”。
- `new`：仅在两次运行可比时，新运行产生了没有基准对应项的问题；运行不可比时目标侧未匹配项也是 `unverifiable`。
- `unverifiable`：无法可靠判断。包括身份一对多/多对一、快照缺失或损坏、文档被移除、任一侧诊断缺失或降级、模型/Agent/RAG 关键运行配置变化以及数量超过安全上限。

人工反馈与机器对比彼此独立：比较生成时的基准最新反馈会冻结到 item provenance，并作为 `baseline_latest_feedback` 返回；之后新增反馈不会回写历史比较，也不会复制到新问题。人工标记 `resolved` 的问题再次检出仍是 `persisting`。`summary.actionable_no_longer_detected` 排除了冻结反馈为 `false_positive` 的未再检出项。

## 匹配与可比性

匹配器版本为 `issue-match-v1`，属于确定性启发式，不保证语义等价：

1. 优先按类别及规则身份字段分桶，例如 `subject + predicate`、`participant + timestamp`、`character + fact`、`item + actual_user` 或世界规则 `key`。
2. 规则身份未形成唯一配对时，再尝试类别和规范化证据签名完全相同的配对；若两侧都存在但规则身份互相冲突，则标为不可验证。
3. 只有 1:1 候选才配对；1:N、N:1 和 N:N 两侧全部标为 `unverifiable`。
4. 匹配依据只持久化字段名、散列、证据重合率和 matcher version，不把模型凭据写入 provenance。

`no_longer_detected` 还要求：双方任务完成、输入快照哈希完整、没有删除基准文档、双方最终 diagnostics 完整且无 partial fallback，并且关键 capabilities、chat provider 与 RAG 身份一致。V1 采用 fail-closed：宁可返回 `unverifiable`，不把模型超时、跳块或 Agent 降级误报为修复。

## 一致性与容量边界

- comparison lineage 与目标 run 快照在同一创建事务中提交。
- 目标 run 完成 CAS、最终 diagnostics 和 issues 落库后，comparison 才会 materialize 为 `ready`。
- matcher 意外失败时使用数据库 savepoint 回滚局部写入，并持久化全量 `unverifiable(comparison_internal_error)`；不会出现已完成任务静默停留在 pending。
- 每侧最多参与 1000 个问题的 V1 身份匹配；超限 fail-closed。读取接口必须分页。
- 同名文档版本分配在 PostgreSQL 上锁定 project row，并由数据库唯一版本索引和“同一逻辑名称最多一个 active 版本”的部分唯一索引兜底；并发冲突返回安全的 `409 document_version_conflict`。

V1 不宣称：语义级修复确认、跨类别问题迁移识别、自动接受人工反馈、exactly-once 调度或无限规模对比。

## 浏览器验收

`frontend/scripts/revision-browser-smoke.mjs` 可对一个已经完成的复检执行桌面端、360px 移动端、谱系拒绝以及浏览器前进/后退验收。脚本只读取现有运行，不会创建项目或调用模型；需传入正确项目/基线/目标运行，以及另一组项目/基线作为负例。若 Playwright 自带浏览器未安装，可用 `--browser-executable=<本机 Chrome 或 Edge 路径>` 指定已有浏览器。
