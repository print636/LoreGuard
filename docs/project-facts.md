# 项目事实清单（更新至 2026-09-07）

可在简历中表述：

- 已实现 FastAPI 端到端 API和 SQLite 本地模式；PostgreSQL、Redis、FastAPI、Celery worker、Web/Nginx、Prometheus 主分析 Compose 链路已由 GitHub Actions 实机 smoke 验证。
- 已实现 Markdown、TXT、JSON 导入与文档版本字段。
- 已实现事实冲突、同刻多地点、知识越权、物品持有、世界规则五类确定性检查。
- 已实现证据片段、严重度、置信度、建议、反馈、取消、重试和可恢复 SSE 事件流。
- 已建立 80 条显式指令规则回归；其 100% 结果只代表规则接线正确。
- 已建立 100 条模板生成的合成自然中文案例（40 dev / 60 test，正负样本各半）及 50 条 challenge-v2。状态建模前原 test P/R/F1 为 0.625/1.000/0.769；当前固定回归为 1.000/1.000/1.000，challenge-v2 after 为 0.962/1.000/0.980。三者均非人工盲测，不能写成生产准确率或无偏提升。
- 已建立 14 例原创、多文档复杂验收集，每例至少 4 份文档，五类问题均覆盖正例和困难反例；无模型固定基线 TP 10、FP 0、FN 0、证据对精确命中率 1.0。该数据由开发者编写且可见，只能表述为开发回归结果。
- 历史上曾使用 OpenAI-compatible 模型对上述 14 例执行 3 轮真实模型增强回归：42/42 完整无降级、171/171 次调用成功，类别和严格完整证据评分均为 TP 30、FP 0、FN 0，预测集合三轮一致；P95 84.97 秒。该数据来自不同中转/模型和较早覆盖协议，只能写成“历史固定开发回归”，不能简化为当前模型成绩或生产准确率 100%。
- 历史固定 2000 字单文档真实模型 5 次延迟测试为首进度 P95 5.9 ms、端到端 P50 6.30 s / P95 7.94 s；它是旧模型栈在当时机器上的开发测量，不是当前栈成绩或生产 SLA。
- 已实现默认关闭的 LangGraph `StateGraph` 受限证据修复 Agent，使用应用层 JSON 动作 `READ_SPAN`、`PATCH_RECORDS`、`ABSTAIN`，并由服务端限制文档、窗口、span、补丁字段、轮次、Token 和 deadline。它不是 Provider 原生 `tool_calls`，没有多智能体，也不是 Evidence RAG。
- commit `bcbfab8` 的 v2 `full × 3` 只在 Agent 阶段调用真实 Provider，主抽取候选由冻结 manifest 合成注入，并非端到端真实模型抽取评测。它完成 90/90 次要求执行但严格正确 59/90；holdout 运行成功 74/81、恢复 26/51、主动弃答 24/30，且没有观察到直接 `ABSTAIN` 成功路径，三路径覆盖 gate 失败，完整结果为 `passed=false`。因此只能作为 Agent 阶段失败诊断和安全边界事实，不能写成 Agent、主抽取或产品质量收益成绩。完整口径见 [v2 full 脱敏 checkpoint](review-agent-v2-full-checkpoint-20260906.md)。
- 已提供 React 审查界面、Docker Compose、GitHub Actions、Prometheus 指标入口和中英文 README。

工程 checkpoint（暂不进入简历）：已实现独立显式配置的 OpenAI-compatible embedding client、中文行号分块、版本化 embedding profile、精确 snapshot chunk/vector schema 和 Alembic 迁移；本机 Compose 已验证真实 PostgreSQL `vector` 列、pgvector 排序、snapshot/profile 隔离与跨项目归属约束。尚未运行真实 embedding，向量检索没有接入主分析消费者，也没有混合检索收益评测，所以这些事实不能包装成已完成的 Evidence RAG 或简历成果。

暂不可表述为已完成：公网在线 Demo、真实用户数据验证、人工录制演示视频、Agent 模型质量收益、Provider 原生 `tool_calls`、多智能体、真实 embedding 已运行、向量检索已接入主分析、完整 Evidence RAG、向量数据库线上压测和生产级 OpenTelemetry 链路。
