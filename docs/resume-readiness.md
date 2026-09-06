# 简历就绪门槛

更新时间：2026-09-06

LoreGuard 当前目标是形成一个可以公开体验、指标诚实、本人能够深入讲解的 AI 应用后端项目。百万字剧情、完整图数据库和多人商业化能力只保留扩展边界，不属于当前简历门槛。

2026-09-06 当前 checkpoint：semantic trust、冻结输入快照、幂等认领、worker lease/heartbeat、typed diagnostics、受限语义标签 repair、batch 失败域、Provider response cap 与 thinking 配置等工程边界已经实现并进入自动回归；Phase 1/complex runner 共用的安全计数包装也已覆盖派生 repair 调用，Compose 的 API/worker thinking 空白配置会归一为 `None`。这些内容只说明代码状态，不自动成为简历成果。新中转/模型虽在 `thinking=disabled` 下通过 preflight，但冻结 Phase 1 `full × 1` 严格 gate 为 **0/3**；详情见 [脱敏 checkpoint](provider-phase1-checkpoint-20260906.md)。在真实 AI gate 重新通过前，不更新简历能力表述。

受限 Agent 第一阶段代码也已接入：LangGraph 1.2.11 `StateGraph` 编排默认关闭，通过应用层 JSON 协议提供 `READ_SPAN`、`PATCH_RECORDS`、`ABSTAIN`，并非已验证的原生 `tool_calls`。commit `bcbfab8` 上的 v2 `full × 3` 只在 Agent 阶段调用真实 Provider，主抽取候选由 runner 按冻结 manifest 合成注入，不是端到端真实模型抽取评测。该运行完成 90/90 次要求执行，严格正确 59/90；其中 holdout 为 81 次：运行成功 74/81、7 次 `read_timeout`，可恢复题正确 26/51（0.509804），不可恢复题主动弃答正确 24/30（0.8）。12 次错误补丁已被生产校验接受但不符合 oracle，另有 9 次可恢复题读取后过度弃答、7 次 timeout 和 3 次 outcome 正确但读取未覆盖 oracle 证据；四组互不重叠。3 次 `trace_replay=false` 全部是 `read_misses_oracle_evidence`，7 次 timeout trace 均可重放。服务端接受的安全违规为 0；运行成功轨迹中没有直接 `ABSTAIN` 路径，三路径覆盖 gate 失败，完整结果为 `passed=false`。因此只能证明 Agent 阶段真实协议运行、安全边界与失败分布，不能声称 Agent、主抽取或产品质量收益，也不能把这些数字写成简历成绩。development/tuned 组虽为 9/9，恢复率和弃答率均为 1.0，但已参与开发，不能替代 holdout。固定语义标签 repair 是独立的非 Agent 补救路径；当前也没有多智能体或真实 Evidence RAG。参见 [第一阶段说明](review-agent-phase1.md)、[pilot 记录](review-agent-pilot-20260906.md)、[v2 full checkpoint](review-agent-v2-full-checkpoint-20260906.md)、[v2 冻结 manifest](../data/agent-acceptance-v2/manifest.json) 与 [离线 runner](../scripts/run_agent_acceptance.py)。

## 已满足

- 公开 GitHub 仓库、独立 `main` 分支和可复现 GitHub Actions；自动化后端测试及前端生产构建可重复运行。
- 项目、文档版本、分析运行、SSE、取消/重试、反馈审计和版本差异的端到端工作流。
- OpenAI-compatible 模型抽取、Pydantic 结构化校验、全局证据行号、失败降级、Token 预算和限流。
- 五类证据化一致性检查、关系图、保守时间线和运行诊断。
- 80 条指令回归、100 条合成自然中文案例、50 条 challenge-v2 和 24,418 字生成型长文 smoke；各自限制已公开。
- 14 组原创、多文档复杂验收场景，每类问题同时包含正例和困难反例；当前无模型基线 TP 10 / FP 0 / FN 0、证据对精确命中率 1.0。该数据由开发者编写且可见，不是人工盲测或生产准确率。
- 历史 complex-v3 真实模型完整 14 例 × 3 轮复测已完成：42/42 完整无降级、171/171 次逻辑调用成功，类别级与严格完整证据级均为 TP 30 / FP 0 / FN 0，14/14 案例三轮预测集合一致；P95 84.97 秒。该成绩来自不同中转/模型和较早覆盖协议，只作为开发者可见历史回归，不外推生产准确率，也不沿用为当前栈成绩。
- 页面已补齐项目切换状态隔离、异步错误/忙碌反馈、上传结果汇总、语义空状态和反馈防重复提交。

## 当前阻塞项

针对 Agent 应用开发方向重新评估后，不能再认为仅剩录屏与学习。下列项目工作已列入 [Evidence RAG 与受限审查 Agent 交付计划](agent-rag-delivery-plan.md)，尚未完成：

1. 真正参与下游模型查证的语义检索、pgvector 持久化与独立检索收益评测；现有 `consumed` 标签不能证明候选影响了裁决。
2. 单 Agent 第一阶段已有受限读取/补丁/弃答、安全 trace 和一次完整 Agent 阶段真实 Provider 冻结运行，但主抽取候选由 manifest 合成注入，且 v2 full gate 已失败；仍需提升 holdout 恢复与运行稳定性，并补相对基线/一次检索的收益、延迟和成本对照。Mock 90 次和 development/tuned 9/9 都不能替代质量门槛。多智能体只做有对照的可选实验，不作为投递硬门槛。
3. 当前新中转/模型的冻结 Phase 1 真实 AI gate：`full × 1` 为 0/3，存在词面/证据拒绝、未解决无效记录和 batch 协议失败；修复后还需完成 `full × 3` 稳定性复验。HTTP 200 或 Provider 调用成功不能替代这一门槛。
4. PostgreSQL、Redis、FastAPI、Celery worker、Web/Nginx 与 Prometheus 的 Compose 全链路已在 GitHub Actions 实机 smoke 通过；新增链路仍需验证，有限负载测试和公网 Demo 尚未完成。
5. 项目内门槛完成后，再进入用户体验学习、演示录屏和简历更新；用户学习不能替代项目未完成项。

固定 2000 字单文档真实模型测试已完成 5 次：模型参与率 5/5，首进度 P95 5.9 ms，端到端 P50 6.30 s、P95 7.94 s，达到原定 `<2 s / <90 s` 门槛。该样本只测主链路延迟，不作为准确率样本。

## 进入简历的最低条件

- 复杂原创验收集可重复运行，五类问题均包含非低级正例与困难反例，预期证据固定且误差明细可审计。
- 在不针对单个已知样例继续硬编码的前提下，当前目标中转/模型的真实完整轮次覆盖率 ≥ 90%，Precision ≥ 0.75、Recall ≥ 0.60、证据命中率 ≥ 0.85。历史固定回归曾达到，但 2026-09-06 当前栈 `full × 1` 严格结果为 0/3，历史成绩不得代替当前复验；任何后续通过仍必须声明开发者可见和非盲测边界。
- 若以 Agent 模型质量或收益进入事实清单，必须有通过 gate 的真实模型独立结果，并公开恢复、弃答、安全违规、动态路径、覆盖/降级、延迟和 Token。当前 v2 full 虽已真实执行，但 `passed=false`；若据此继续修改 Prompt、实现或标签，后续同套件只能算开发回归，新的泛化质量声明还需要另行冻结未参与调优的测试集。
- 新用户可从 README 在干净环境完成启动，并通过页面完成创建/导入/分析/查看证据/反馈/版本比较。
- GitHub Actions 通过、无密钥泄漏；Docker/Compose 或替代公开部署至少有一种经过实际验证。
- 用户可以脱离文档完成 1 分钟摘要、5 分钟架构讲解，并回答模型降级、SSE、RAG/规则分工、误报和成本追问。

达到这些条件后，先更新项目事实清单，再由用户学习验收，最后更新一页中文简历。计划功能、生成型固定回归和未验证的扩展能力不得写成完成成果。
