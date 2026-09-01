# 项目事实清单（2026-08-29）

可在简历中表述：

- 已实现 FastAPI 端到端 API、SQLite 本地模式与 PostgreSQL/pgvector Compose 配置。
- 已实现 Markdown、TXT、JSON 导入与文档版本字段。
- 已实现事实冲突、同刻多地点、知识越权、物品持有、世界规则五类确定性检查。
- 已实现证据片段、严重度、置信度、建议、反馈、取消、重试和可恢复 SSE 事件流。
- 已建立 80 条显式指令规则回归；其 100% 结果只代表规则接线正确。
- 已建立 100 条模板生成的合成自然中文案例（40 dev / 60 test，正负样本各半）及 50 条 challenge-v2。状态建模前原 test P/R/F1 为 0.625/1.000/0.769；当前固定回归为 1.000/1.000/1.000，challenge-v2 after 为 0.962/1.000/0.980。三者均非人工盲测，不能写成生产准确率或无偏提升。
- 已提供 React 审查界面、Docker Compose、GitHub Actions、Prometheus 指标入口和中英文 README。

暂不可表述为已完成：公网在线 Demo、真实用户数据验证、人工录制演示视频、完整 LangGraph/向量数据库线上压测、生产级 OpenTelemetry 链路。
