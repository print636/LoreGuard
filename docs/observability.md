# 运行指标与进程边界

LoreGuard 的 API 和 Celery worker 是不同进程。进程内 `Counter.inc()` 或
`Histogram.observe()` 不会自动跨进程汇总，而且 API 重启会把内存值归零。因此
`/metrics` 的业务指标在每次抓取时从共享的 `analysis_runs` 与 `run_events` 表生成；
worker 完成、失败或取消任务后，API 无需收到额外回调就能看到相同的持久化状态。

## 暴露的业务指标

| 指标 | 类型 | 语义 |
| --- | --- | --- |
| `loreguard_analysis_runs_total` | Counter | 当前数据库生命周期内已接受并创建的运行记录总数 |
| `loreguard_analysis_run_terminal_transitions_total{status}` | Counter | `run_events` 中按运行 ID 去重、且被运行记录当前同名终态佐证的 `completed/failed/cancelled` 事件数 |
| `loreguard_analysis_runs_current{status}` | Gauge | 抓取时各状态的持久化运行记录数；标签固定为 `queued/running/completed/failed/cancelled/unknown` |
| `loreguard_analysis_seconds{status}` | Histogram | 当前终态与持久化终态事件相互佐证的任务，从首次 worker claim 到终态的非负墙钟耗时；包含中间重试和退避等待，按固定终态标签与固定 bucket 汇总 |
| `loreguard_analysis_duration_unavailable{status}` | Gauge | 已有上述相互佐证、但因缺失时间字段、时间倒置或非法区间而没有进入耗时直方图的终态记录数 |
| `loreguard_analysis_metrics_database_available` | Gauge | 本次业务指标是否成功从数据库生成；数据库不可读时 `/metrics` 返回 503 和值 0，不返回伪造的零业务指标 |

`completed`、`failed` 终态事件和耗时均来自 worker 已提交的数据库事务，而不是 API
进程对任务结果的猜测。`runs_current` 还会诚实展示没有对应终态事件的旧版/异常状态行，
但不会倒推并伪造一条历史终态事件。用户主动重试会创建新的运行记录，因此是独立样本；
Celery 对同一次运行的中间重试沿用首次 `started_at`，所以耗时包含重试退避，并非纯 CPU
或模型调用时间。
取消前尚未启动的任务没有执行耗时，会计入 `duration_unavailable`，不会用创建到取消的
排队时间冒充分析耗时。

## 重启、恢复与删除语义

- API 或 worker 重启不会重置业务指标；新 API 进程会重新读取同一个数据库。
- 两个 `*_total` 的累计语义以当前数据库生命周期为边界。正常应用流程不会删除运行记录，
  所以这些值单调增加；若运维人员清空、回滚或替换数据库，应把它视为指标数据源重置，
  不能解释为任务数倒退。
- `runs_current` 明确是快照 Gauge，任务状态迁移时可以增减。
- 没有引入 Prometheus multiprocess 临时文件，因为它既不能代表数据库恢复后的历史，
  也容易在容器/worker 重启后留下过期分片。
- 数据库不可读时响应中的 `database_available 0` 便于人工诊断；Prometheus 通常不会采纳
  非 2xx 响应中的样本，正式告警应使用抓取目标自带的 `up == 0`。

## 隐私、基数与成本边界

抓取查询只投影运行状态、`started_at`、`completed_at` 和匹配的终态类别；运行 ID
只在数据库子查询内部用于终态事件去重和关联，不进入指标。查询不读取或输出项目名、
文档名、故事正文、错误原文、Provider Key、endpoint 或 Prompt。状态标签采用固定
allowlist，数据库里的未知值统一合并到 `unknown`，因此不会形成用户可控的高基数
时间序列。

当前实现用一次只读查询扫描运行记录并在内存中构造固定 Prometheus bucket，适合当前
单机和作品集规模，但不声称具备海量历史下的抓取性能。若运行记录达到大规模，应增加
事务内的低基数指标账本或按时间分区的汇总表，同时保留原始运行表作为可审计来源，再
用压测决定保留周期；不能为了抓取速度改回彼此隔离的 worker 内存计数器。

当前 `prometheus.yml` 只抓取一个 API target。若未来横向扩容为多个 API 副本，每个副本
都会导出同一份数据库全局业务指标；应改成独立 exporter，或在查询中对副本使用 `max`
而不是 `sum`，否则会重复计算。Python 进程自身指标仍然可以按实例聚合。
