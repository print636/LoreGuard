# LoreGuard 与 AI 应用后端面试指南

更新：2026-09-13。

这份文档只负责**面试表达、项目答辩和真实性校验**，不承担系统知识教学。知识原理、代码阅读和动手练习见 [学习指南](learning-guide.md)；项目可说与不可说的最终边界以 [项目事实清单](project-facts.md) 为准。

目标岗位以“平台研发－后端开发工程师（AI 应用方向）”为主：既考查后端工程能力，也考查 LLM、RAG、Agent 的应用方式、稳定性和能力边界。训练重点是把真实项目讲清楚，不是堆叠术语或把计划包装成成果。

## 一、面试教练的工作协议

每次模拟面试必须遵守以下顺序：

1. 先说明当前阶段：基础筛查、项目深挖、系统设计、行为面，或压力追问。
2. **每轮只问一个问题**，等待候选人完整回答后再继续。
3. 不在提问时附带标准答案，也不一次塞入多个子问题。
4. 反馈只指出一个最高优先级问题，再给一个可执行的改法。
5. 根据上一轮回答动态追问；已经掌握的内容不机械重复。
6. 用户临时问其他问题时先回答，记录原训练位置，然后返回主线。
7. 首次出现术语时用一句通俗话解释；如果用户无法解释术语，转入学习指南对应模块，不能用背稿掩盖知识缺口。

每轮反馈固定使用这张卡片：

```text
当前阶段：
结论：通过 / 知识缺口 / 表达缺口 / 两者都有
做得好的一个点：
最优先修正的一个点：
更好的回答骨架：
下一题：只问一个
```

### 如何区分“不会”和“没说好”

| 检验 | 知识掌握 | 表达合格 |
| --- | --- | --- |
| 是什么 | 能用自己的话定义，不靠同义词循环 | 15 秒内给出结论 |
| 为什么 | 能联系项目目标解释取舍 | 先结论，后两点理由 |
| 怎么做 | 能沿代码或数据流讲到关键边界 | 不罗列所有类名 |
| 如何证明 | 能指向测试、评测或可复现实验 | 明确“测了什么、没证明什么” |
| 失败怎么办 | 能说出真实失败路径和恢复方式 | 不回避失败，不夸大兜底 |

只会背出本指南但不能回答“为什么”或定位实现，仍算知识缺口。理解正确但答案散乱、超时或没有结论，才算表达缺口。

## 二、面试证据规则

回答项目问题时按“结论 → 设计原因 → 实现证据 → 能力边界”组织。证据优先级如下：

1. 生产路径代码：功能确实接入端到端链路。
2. 自动化测试：正常、异常和边界行为可重复验证。
3. 冻结评测产物：指标与数据、配置、代码版本绑定。
4. 本地演示：证明当前环境可运行，但不能替代压力测试或真实用户验证。
5. 设计文档：只能解释方案，不能单独证明已经实现。

LoreGuard 的主要证据入口：

| 主题 | 生产实现 | 验证入口 |
| --- | --- | --- |
| 主分析流水线 | [`app/pipeline.py`](../app/pipeline.py)、[`app/service.py`](../app/service.py) | [`tests/test_engine.py`](../tests/test_engine.py)、[`tests/test_run_reliability.py`](../tests/test_run_reliability.py) |
| 五类确定性检查 | [`app/rules.py`](../app/rules.py) | [`tests/test_engine.py`](../tests/test_engine.py)、[`tests/test_complex_v3_evaluation.py`](../tests/test_complex_v3_evaluation.py) |
| 模型抽取与失败降级 | [`app/model_extractor.py`](../app/model_extractor.py)、[`app/provider.py`](../app/provider.py) | [`tests/test_model_extractor.py`](../tests/test_model_extractor.py)、[`tests/test_model_execution.py`](../tests/test_model_execution.py) |
| DOCX 与版本处理 | [`app/docx_import.py`](../app/docx_import.py)、[`app/document_diff.py`](../app/document_diff.py) | [`tests/test_docx_import.py`](../tests/test_docx_import.py)、[`tests/test_document_diff.py`](../tests/test_document_diff.py) |
| Evidence RAG | [`app/evidence_rag.py`](../app/evidence_rag.py)、[`app/evidence_store.py`](../app/evidence_store.py) | [`tests/test_evidence_rag.py`](../tests/test_evidence_rag.py)、[`tests/test_pgvector_integration.py`](../tests/test_pgvector_integration.py) |
| 后置证据复核 | [`app/issue_evidence_review.py`](../app/issue_evidence_review.py) | [`tests/test_issue_evidence_review.py`](../tests/test_issue_evidence_review.py)、[真实 A/B 结果](issue-review-v1-result-20260907.md) |
| Evidence Investigator（当前 Agent 主讲路径） | [`app/provider.py`](../app/provider.py)、[`app/evidence_investigator_loop.py`](../app/evidence_investigator_loop.py)、[`app/evidence_investigator_runtime.py`](../app/evidence_investigator_runtime.py)、[`app/candidate_promotion.py`](../app/candidate_promotion.py) | [`tests/test_provider_tool_calls.py`](../tests/test_provider_tool_calls.py)、[`tests/test_evidence_investigator_loop.py`](../tests/test_evidence_investigator_loop.py)、[`tests/test_candidate_promotion.py`](../tests/test_candidate_promotion.py)、[真实 E2E 评测](evidence-investigator-live-evaluation.md) |
| 旧受限记录修复 Agent | [`app/review_agent.py`](../app/review_agent.py) | [`tests/test_review_agent.py`](../tests/test_review_agent.py)、[旧完整检查点](review-agent-v2-full-checkpoint-20260906.md) |
| 异步任务与恢复 | [`app/tasks.py`](../app/tasks.py)、[`app/service.py`](../app/service.py)、[`app/main.py`](../app/main.py) | [`tests/test_tasks.py`](../tests/test_tasks.py)、[`tests/test_run_reliability.py`](../tests/test_run_reliability.py)、[`tests/test_api.py`](../tests/test_api.py) |
| 图、时间线与反馈 | [`app/projections.py`](../app/projections.py)、[`app/main.py`](../app/main.py) | [`tests/test_visualization.py`](../tests/test_visualization.py)、[`tests/test_api.py`](../tests/test_api.py) |
| 总体真实性边界 | [项目事实清单](project-facts.md) | [简历就绪结论](resume-readiness.md) |

## 三、与目标岗位的对应关系

| 岗位关注点 | 可用项目证据 | 主动承认的边界 |
| --- | --- | --- |
| AI 能力落地 | 将普通叙事文本转成状态记录，输出带原文证据的问题报告 | 开放长篇与真实用户质量尚未验证 |
| 后端工程 | FastAPI、PostgreSQL/pgvector、Redis/Celery、版本快照、取消/重试、SSE、Docker/CI | 不是生产流量下的高可用服务 |
| RAG | 固定 BGE、真实向量列、精确余弦检索、keyword+dense RRF、隔离与复用 | 混合检索没有显著领先两种基线，低词面门槛未通过 |
| Agent | Evidence Investigator 使用 Provider 原生 function tool calls（下文简称“原生工具调用”），在冻结快照内选择搜索、读取、提交候选或弃答，候选再经确定性 promotion | 默认关闭；首次 holdout 只有 10 例、8/10，不能外推为生产质量；无多智能体 |
| 旧 Agent 实验 | LangGraph 编排受限记录修复，模型在应用层 JSON 的读取、补丁、弃答间选择 | 不是原生工具调用，旧质量 gate 失败，只在追问历史实现时说明 |
| 稳定性 | 超时/429/5xx/非法 JSON 降级，输入冻结、幂等认领、lease/heartbeat、持久化 SSE | 没有 exactly-once 模型计费保证，也没有生产 SLA |
| Agent 周边工具 | 结构化合同、证据白名单、安全 trace、评测 runner 与故障分类 | 是项目内工具链，不是通用 Agent SDK |
| AI 辅助开发 | 使用 Codex、Claude Code、Cursor 做需求拆解、代码阅读、实现与测试辅助 | 必须能亲自解释、运行和校验最终设计，不能把工具产出冒充手写 |
| 高并发 | 两个 worker 的 20 任务队列与隔离烟测 | 不能称为高并发压测或容量结论 |

### 为什么不为了加分项硬加多智能体

这个问题的合格结论是：**当前没有多智能体，这是经过范围控制后的未实现项，不是换名包装。**

Evidence Investigator 已用一个模型和四个受控工具验证“模型选择—服务端执行—确定性复核”的闭环；旧记录修复实验也只是另一条单模型路径。贸然拆成规划、检索、修复、裁决多个 Agent，会增加共享状态一致性、错误传播、Token 成本、超时链路和评测归因难度。未来只有在单 Agent 的失败样本证明角色拆分能带来可测收益时，才值得用新冻结集验证。岗位的加分项不是要求把所有名词都实现一遍；能解释为何暂缓以及如何验证，反而更能体现工程判断。

## 四、三种长度的项目表达

### 1 分钟版本

> LoreGuard 是我独立推进的故事剧情一致性审查平台，面向编剧、小说作者和叙事设计人员。它把 DOCX 或普通文本抽取成带原文位置的结构化记录，再用确定性规则检查事实、地点、知识、物品和世界规则冲突。为补足单轮抽取的遗漏，我实现了默认关闭的 Evidence Investigator：模型通过 Provider 原生工具调用，在冻结文档快照内选择检索、读取、提交一个候选或主动弃答；候选只有经服务端重新绑定证据并由既有规则确定性复现后，才会新增问题。工程上我用 FastAPI、PostgreSQL/pgvector、Redis/Celery 实现异步分析、SSE、取消重试和失败降级。两次 DEV 均为 8/8，首次 10 例冻结 holdout 为 8/10，但样本很小且系统主链路仍有一个误报，所以我不把它说成开放文本或生产质量结论。

控制在 55–75 秒。面试官没有追问时，不主动展开旧 LangGraph 修复 Agent，也不背诵全部指标。

### 5 分钟版本

按以下五段讲，每段只承担一个目的：

1. **问题与用户**：长篇故事的信息分散，单次聊天审稿难以保留角色认知、时间和物品状态；用户需要可回到原文的审查结果。
2. **主数据流**：冻结当前文档版本 → 基线与可选模型抽取 → 语义质量门与归一化 → 五类确定性检查 → 可选 Evidence Investigator → 可选 Evidence Reviewer → 持久化问题、事件和反馈。
3. **AI 与规则分工**：模型擅长把自然语言映射为结构化记录，并在 Investigator 中决定查什么、读什么；模型提交的仍是不可信候选，只有确定性 promotion 能把它变成新增 issue。Reviewer 则只追加证据意见。
4. **工程可靠性**：异步队列、任务认领与租约心跳、取消检查点、有限重试、熔断降级、Token 门控、持久化 SSE；说明这些是已经实现的机制，而不是生产 SLA。
5. **验证与反思**：用一个通过点和一个失败点收尾。例如 Investigator 两次 DEV pair 都是 8/8，首次 holdout 是 8/10 且 10/10 正常终止；同时这只是 10 例小样本，系统确定性主链路仍有一个误报，该 holdout 也不能再拿来调参。

### 15 分钟版本

1. 产品目标、输入输出和为什么不做“聊天框套壳”。
2. 文档版本、运行输入冻结和证据行号如何建立可追溯性。
3. 八类记录、Pydantic 合同、证据回填与逐条错误隔离。
4. 语义质量门、别名归一化、确定性规则与五类冲突。
5. 本地主路候选信号与真实 Evidence RAG 的区别。
6. BGE embedding profile、chunk、snapshot、pgvector 与 RRF 数据流。
7. IssueEvidenceReviewer 为什么只能追加注释，不能覆盖规则结果。
8. Evidence Investigator 的原生工具状态机、冻结授权、确定性 promotion 与安全 trace。
9. Celery、Redis、任务认领、lease/heartbeat、取消/重试与 SSE 续传。
10. DEV pair 与首次 holdout 分别说明什么、没有说明什么。
11. 面向长篇和生产部署的下一步，但不把路线图说成已完成。

旧 LangGraph 记录修复 Agent 不放进默认 1 分钟主线；只有面试官追问简历中的 LangGraph、历史 Agent 路径或失败实验时，再单独说明。15 分钟讲解中每个术语都必须能被打断追问。无法脱离文档解释的部分应回到学习阶段，而不是继续扩展讲稿。

## 五、LoreGuard 简历逐条追问树

以下表述对应最新版一页简历。每一条都先准备“首答”，再准备一层实现追问和一层边界追问。

### 5.1 多格式导入、版本、异步分析与报告

简历表述：支持 DOCX/Markdown/TXT/JSON 导入、版本管理、异步分析、证据化冲突报告、反馈审计及按需关系图/时间线。

首问：**一次分析从上传文件到页面看到问题，经过了哪些步骤？**

回答骨架：

- DOCX 先安全解析成稳定纯文本，其他格式也统一进入文档记录。
- 同名文件创建新版本，运行开始时冻结活动版本快照。
- API 创建分析任务，线程或 Celery worker 执行流水线。
- 抽取、规则检查和可选复核产生记录与问题，进度事件持久化并通过 SSE 推送。
- 页面读取完成结果，图和时间线只投影已持久化记录，不再次调用模型。

继续追问可能落在：

- 为什么版本更新后重试仍使用原快照？
- DOCX 如何防止压缩炸弹、外部关系和宏风险？
- 为什么进度用 SSE 而不是 WebSocket？
- 断线续传依赖什么游标，失败任务如何避免被当成完成？
- 反馈为什么保留审计历史，而不是直接修改规则？
- 图或时间线过大时怎么处理？当前只有限制与 warning，不代表已解决长篇可视化。

证据边界：可以说功能与自动化测试已接通；不能说处理过商业游戏长篇、百万字语料或真实多人协作。

### 5.2 确定性规则、模型抽取与 Evidence RAG

简历表述：确定性规则完成一致性判断，模型用于结构化抽取与证据复核；以 BGE 向量检索和关键词/向量 RRF 构建 Evidence RAG，并以证据白名单限制引用。

首问：**既然已经调用大模型，为什么最终还要由规则判断？**

回答骨架：

- 模型解决普通自然语言到结构化状态的语义映射，适合召回和理解。
- 时间、所有权、知识获得等冲突在结构化后可以用可复现规则比较。
- 规则输出必须带双方证据，便于审计；模型失败不会抹掉基础结果。
- 后置 Reviewer 只给规则 issue 添加证据意见，不拥有删除或改写权。

继续追问可能落在：

- 无 Key 时 BaselineExtractor 能做什么，不能做什么？
- 模型输出如何经 Pydantic、行号范围和证据正文回填校验？
- 五类问题分别依赖哪些结构化状态？
- 为什么字符 n-gram 候选排序不是真实向量 RAG？
- embedding profile 为什么要包含模型、revision、维度和变换身份？
- RRF 是什么，为什么不用直接相加不同通道的原始分数？
- 项目、文档、版本、内容哈希、chunker 和 profile 隔离防止了什么？
- allowlist 校验为何只能降低越权引用，不能保证模型判断正确？

证据边界：真实 BGE、pgvector 与 RRF 已运行；但首次 holdout 的低词面 Recall@5 为 79.31%，差一条未过预设门槛，混合 Recall 与 dense 相同，不能说“混合显著优于所有基线”。Reviewer 的真实 A/B 是 4/12 到 7/12，但 absolute gate 失败，信息不足类为 0/4，不能说质量达标。

### 5.3 Evidence Investigator：当前 Agent 主讲路径

首问：**这里的 Agent 到底自主决定什么，哪些决定仍掌握在服务端？**

回答骨架：

- 确定性主链路先从已有结构化记录建立一个调查 seed；模型不能自行扩大本次任务或选择任意数据库记录。
- 服务端通过 Provider 原生工具调用暴露四种动作。模型按阶段请求 `SEARCH_EVIDENCE`，从结果中选择 `READ_SPAN`，然后只能 `SUBMIT_VERDICT` 一个候选或 `ABSTAIN`。
- 模型只提出工具请求；服务端校验工具名、参数 schema、当前阶段、seed、次数、Token、deadline 与快照授权后才执行。临时 `result_ref` 和 `span_ref` 防止模型伪造文档身份或任意行号。
- `SUBMIT_VERDICT` 的候选仍不可信。服务端按冻结原文重建证据，重新检查字段关系、语义门和对应规则；只有确定性复现了新冲突，promotion 才原子追加 issue。
- 任一模型、检索、协议或 promotion 故障只让该可选阶段跳过或降级，原确定性记录、问题、ID、顺序和 provenance 保持不变。

白板状态机只画这一条：

```text
seed
  → SEARCH_EVIDENCE
  → READ_SPAN
  → SUBMIT_VERDICT → deterministic promotion → add issue / reject
                   ↘ ABSTAIN（search/read/verdict 阶段均可受控结束）
```

安全 trace 只保存已校验并实际执行的动作顺序、运行内文档伪名哈希（不是内容哈希）、读取行号、候选字段形状和完整性标记；非法但未执行的请求只留下受限失败类别或计数。它不保存 Prompt、正文、模型原始响应、Key 或服务地址。

实现与验证入口：

- 原生调用合同：[`app/provider.py`](../app/provider.py) 的 `complete_with_tools`，以及 [`tests/test_provider_tool_calls.py`](../tests/test_provider_tool_calls.py)。
- seed、授权与状态机：[`app/evidence_investigator.py`](../app/evidence_investigator.py)、[`app/evidence_authority.py`](../app/evidence_authority.py)、[`app/evidence_investigator_state.py`](../app/evidence_investigator_state.py)、[`app/evidence_investigator_loop.py`](../app/evidence_investigator_loop.py)。
- RAG 与 promotion：[`app/evidence_investigator_rag.py`](../app/evidence_investigator_rag.py)、[`app/candidate_promotion.py`](../app/candidate_promotion.py)。
- 生产接线：[`app/evidence_investigator_runtime.py`](../app/evidence_investigator_runtime.py)、[`app/service.py`](../app/service.py)。
- E2E runner：[`scripts/run_evidence_investigator_live.py`](../scripts/run_evidence_investigator_live.py) 经公开 HTTP API 建项、上传冻结文档、启动 Celery 分析并在终态后本地判分；[`scripts/check_evidence_investigator_dev_pair.py`](../scripts/check_evidence_investigator_dev_pair.py) 从逐案例结果复算两次 DEV 是否可晋级。完整口径见[真实 E2E 评测](evidence-investigator-live-evaluation.md)。

评测首答：commit `618ca991`、运行时模型别名 `deepseek-v4-pro` 下，两次连续 DEV 都是 8/8，pair checker v3 以相同复现指纹通过；首次冻结 holdout 为 8/10，即 TP 4、FN 1、TN 4、FP 1，precision/recall 都是 80%，10/10 正常终止。Agent 共提交 8 个候选，promotion 接受 4 个目标正例、拒绝 4 个，没有错误 Agent 候选被接受；系统最终仍有 1 个 FP，它来自确定性主链路。必须同时说明：这只有 10 例，模型名只是请求别名，不能证明上游权重；holdout 已封存且不再用于调参，指标不能外推到开放故事或生产质量。

### 5.4 旧 LangGraph 受限记录修复 Agent：只在追问中讲

如果面试官指着简历中的 LangGraph 表述，先主动区分：**这是旧的记录修复实验路径，不是上面的 Evidence Investigator。**

- 它只接收模型抽取后因 `lexical_support` 失败而隔离的候选，不从确定性记录出发寻找新冲突。
- LangGraph `StateGraph` 编排 `READ_SPAN`、`PATCH_RECORDS`、`ABSTAIN`；动作由模型返回应用层 JSON，不是 Provider 原生工具调用。
- 它只能在授权窗口内修补允许字段，之后候选仍要重新校验。固定 semantic-label repair 是另一条受限单轮结构化模型调用，只补正 `modality/source_scope/certainty` 标签，不让模型选择工具。
- 旧真实 Provider 验收的主抽取候选来自冻结 manifest 合成注入，严格正确 59/90、完整 gate 失败，因此只能证明实现、边界和失败诊断，不能声称端到端质量收益。

### 5.5 异步后端与失败恢复

继续追问可能落在：

- Investigator、旧修复 Agent、IssueEvidenceReviewer 和固定 semantic-label repair 的输入输出分别是什么？
- 为什么模型不能直接指定真实 document ID、任意行号或最终 issue？
- 任务重复投递时怎样避免两个 worker 同时执行？
- lease/heartbeat 解决什么，为什么仍不等于 exactly-once？
- 取消如何穿过 API 与 worker 进程？
- Provider 超时、429、5xx、非法 JSON 或压缩响应异常分别如何处理？
- 每日 Token 预算和写接口限流为什么还不是多实例原子配额？
- 如果真正面向高并发 C 端，下一步如何做背压、原子配额、容量测试与 SSE 扇出？

证据边界：可以说已实现原生工具调用的 Investigator、输入冻结、确定性 promotion、安全 trace、幂等认领、lease/heartbeat 和队列烟测；不能说生产高可用、高并发 SLA、多智能体、所有 Agent 路径都用原生调用，或小型 holdout 已证明生产质量。

## 六、华为实习逐条追问树

实习项目的事实证据来自公开仓库与实习报告，不由 LoreGuard 仓库证明。面试前要亲自打开 `toml_lsp4cj`，为下列每项准备一个代码入口和一个测试入口。

### 6.1 JSON-RPC、诊断、补全、定义跳转与 VS Code 接入

首问：**编辑器里打开一个 TOML 文件后，诊断结果是怎样从 Language Server 回到 VS Code 的？**

首答应覆盖：

- VS Code 客户端启动或连接语言服务器。
- 双方按 LSP 约定通过 JSON-RPC 交换初始化、文档同步和功能请求/通知。
- 服务端解析当前文档状态，产生带 range 的诊断或响应补全/定义位置。
- 客户端把协议结果映射为编辑器提示或跳转。

继续追问：请求与通知的区别、请求 ID 和错误响应、能力协商、文档 URI、定义跳转返回位置、进程通信方式、异常消息如何隔离。

不要使用“协议分析”这种模糊词。只能说自己确实做过的 JSON-RPC 消息处理、LSP 生命周期或具体能力。

### 6.2 增量同步、UTF-16 位置与跨平台验证

首问：**为什么 LSP 的字符位置不能直接当成语言运行时的字符串下标？**

首答应覆盖：LSP 位置以行号与 UTF-16 code unit 计数；语言内部字符串可能按 UTF-8 字节、Unicode 标量或其他索引方式处理；遇到中文之外的补充平面字符时差异尤其明显，因此需要显式换算并测试边界。

继续追问：

- 增量编辑的 range、rangeLength、text 怎样应用到文档快照？
- 多次变更、越界位置、换行符差异和 emoji 如何测试？
- 为什么要做 Windows/Linux/macOS 构建验证？路径、换行、进程启动或工具链差异是什么？
- 自动化测试覆盖的是功能、协议边界还是集成链路？不要把测试数量替代测试设计。

如果某个实现细节目前记不清，应明确说“我会回到对应代码确认”，不能临场编造。

## 七、easyMeeting 逐条追问树

easyMeeting 的代码证据在独立仓库，面试前需逐项核对当前分支，尤其确认“失败重试”和“聊天消息分表”确实位于生产路径而非未接入代码。

### 7.1 Netty/WebSocket、ChannelGroup 与心跳

首问：**一个用户加入会议房间后，消息是如何只广播给同房间成员的？**

回答应沿连接建立、身份/房间绑定、ChannelGroup 管理、消息处理和连接清理讲解。继续追问可能涉及 EventLoop 线程模型、Channel 是否线程安全、慢连接、粘包与 WebSocket frame、异常断开、心跳误判和资源释放。

边界：这证明实时连接与房间管理经验，不自动等于完整音视频媒体服务器或 WebRTC 栈。

### 7.2 Redis Pub/Sub 与 RabbitMQ 切换、重试和分表

首问：**为什么同一个消息模块同时支持 Redis Pub/Sub 和 RabbitMQ？**

回答骨架：先说明二者面向的可靠性与运维取舍，再讲配置切换的抽象边界；最后说明失败重试和存储如何避免把“广播成功”与“消息持久化成功”混为一个概念。

继续追问：

- Redis Pub/Sub 在订阅者离线时会发生什么？
- RabbitMQ 的 ack、重试与重复消息如何处理？
- 分表键、路由、分页与跨表查询如何设计？
- 消息顺序、幂等和最终一致性达到什么程度？
- 当前是否做过压力测试？没有就不能给吞吐或 SLA 数字。

## 八、专业技能与 AI 辅助开发追问树

简历中的技能不是关键词清单，面试官可随机挑一个追问。回答时按“我在哪用过 → 解决什么问题 → 一个限制”展开。

### 8.1 AI 编程工具

首问：**你说自己深度使用 AI 编程工具，具体怎样保证生成代码的质量和真实性？**

建议回答骨架：

- 自己负责需求边界、验收条件和最终技术决策，把任务拆成可验证的小块。
- 使用 Codex、Claude Code 或 Cursor 辅助检索代码、提出实现、补测试和解释失败。
- 通过读 diff、运行测试、构造异常输入、核对文档与真实评测来验收；不能解释的代码不作为已掌握能力。
- 保留失败结果和声明边界，例如旧修复 Agent 与 Reviewer 的 gate 失败没有被包装成成功指标，Investigator 的 8/10 也没有外推。
- AI 提高迭代速度，但项目所有权体现在问题定义、取舍、验证和最终责任，而不是声称每行都手写。

禁止回答：

- “让 AI 把整个项目写完，我只负责运行”。
- “测试通过就说明设计正确”。
- 虚构没有使用过的提示词流程、人工协作或线上数据。
- 暴露 API Key、私密 Prompt、原始用户文本或服务地址。

### 8.2 数据库、Redis 与分布式基础

首问队列按岗位相关度依次覆盖：事务边界与并发更新、索引与查询计划、Redis 适用场景、消息重复与幂等、租约与心跳、限流、缓存一致性、水平扩展。每次只取一题。

LoreGuard 当前能支撑的例子是运行快照、条件更新认领、向量 profile/chunk 唯一身份、Celery 队列和 Redis broker；不能因为使用了这些组件就声称完成分布式生产架构。

### 8.3 其他技能标签的抽查边界

简历上出现的每个名词都可能成为一道独立问题。训练时一次只抽一个，并优先要求用项目事实回答：

| 简历标签 | 最小合格解释 | 可用证据 | 不能偷换的概念 |
| --- | --- | --- | --- |
| Java / Python | 能说出本人写过的模块、语言特性和一个调试案例 | easyMeeting / LoreGuard 代码 | “项目用了”不等于掌握语言原理 |
| Spring Boot / FastAPI | 能解释一次请求如何进入业务层、校验、持久化和返回 | 两个项目的 API 入口 | 框架启动成功不等于服务治理经验 |
| MyBatis / SQLAlchemy | 能解释对象与 SQL 的边界、事务何时提交或回滚 | 项目数据访问代码 | ORM 不会自动消除 N+1、锁和索引问题 |
| MySQL / PostgreSQL | 能解释为何不同项目选择不同数据库，以及一个索引或事务问题 | 项目 schema 与查询 | 不能声称做过未发生的海量数据优化 |
| Redis | 能区分缓存、Pub/Sub 与 Celery broker 的使用语义 | easyMeeting / LoreGuard | Pub/Sub 不提供离线消息可靠投递 |
| RabbitMQ / Celery | 能解释消息确认、重试、重复执行与业务幂等 | 两个项目的消息路径 | 使用队列不等于 exactly-once |
| BGE / pgvector | 能解释文本如何变为定长向量、怎样绑定 profile 并检索 | LoreGuard Evidence RAG | pgvector 是存储/检索能力，不是 embedding 模型 |
| REST / WebSocket / SSE | 能根据双向实时通信与单向进度推送说明选择 | easyMeeting / LoreGuard | 三者不能仅按“新旧”比较 |
| Docker Compose / CI | 能解释服务依赖、健康检查和 CI 实际验证了什么 | Compose 与 GitHub Actions | 本地编排不等于 Kubernetes 或生产发布平台 |
| Prometheus | 能指出当前暴露的指标及它能发现的问题 | LoreGuard metrics 入口 | 有 `/metrics` 不等于完整可观测体系 |

### 8.4 教育、获奖与开源入口

教育背景通常只需准确说明专业、毕业时间与求职方向。如果被问“为什么转向 AI 应用后端”，应串起软件工程基础、后端项目、华为实习和 LoreGuard，不需要虚构课程成绩或研究经历。

蓝桥杯省一等奖可被追问算法、C/C++ 或备赛过程。只讲本人真实题目和训练方式；奖项证明过往竞赛能力，不自动证明当前算法面试已准备充分。英语四、六级只按证书事实回答。

开源链接的首问通常是：**仓库中哪些决策是你做的，哪些工作使用了 AI 辅助？** 回答要与 8.1 一致，并能现场定位 README、核心代码、测试和失败评测。公开仓库是可核查证据，也会放大夸大表述的风险。

### 8.5 编码题与后端基础的面试方式

这部分与 LoreGuard 知识学习分开训练，但仍属于岗位面试门槛：

- 编码题一次只做一道，先复述输入输出，再说最直接方案、优化点、复杂度和边界样例，最后编码。
- 优先覆盖数组与哈希、双指针、栈/队列、二分、树与图遍历、堆、常见动态规划；不为偏门竞赛技巧挤占项目理解时间。
- 后端基础优先覆盖进程/线程与并发、TCP/HTTP、数据库索引与事务、缓存与消息队列、分布式幂等和限流。
- 反馈要区分“算法不会”“代码实现错误”“没有说清思路”，每轮只修一个最主要问题。

蓝桥杯经历意味着可以提高编码训练起点，但不能跳过当前基线测试。

## 九、系统设计与压力追问

### 如果流量扩大十倍

先区分当前事实和设计方案：

- 当前 API 与 worker 可分离，任务有持久化状态、认领和租约；这些是扩展基础。
- 未来 API 可无状态横向扩容，Celery worker 按队列和资源类型拆分，模型调用设置全局并发与背压。
- 单进程限流和本地每日预算应迁移到 Redis 原子脚本或事务式配额服务。
- SSE 在多实例下需要共享事件源或专用推送层，不能依赖进程内状态。
- 为数据库补充真实查询计划、连接池和索引监控；向量检索是否切 HNSW 必须由数据规模和延迟测量决定。
- 建立分阶段容量测试、故障注入和 SLO，再谈吞吐与高可用。

这是“会怎样设计”，不是“已经实现”。

### 如果故事从几千字扩大到长篇系列

可以提出：按文档增量索引、分层检索、实体/事件的规范化状态存储、章节摘要与跨章查询、局部重分析、冷热数据分层、可视化聚合。必须主动说明当前只预留了分块、版本、快照和可替换检索边界，没有测出可支持的最高字数，也没有验证商业游戏全量剧情。

### 如果模型服务连续失败

回答顺序：识别失败类型 → 有限重试 → 运行级熔断 → 全文确定性基线继续 → 安全诊断与 SSE 状态 → 用户可重试。再补充：基线能保证流程可完成，但语义覆盖会下降，所以“可恢复”不等于“结果质量不变”。

### 如果模型引用了未授权文档

服务端应拒绝引用：检索先绑定冻结 snapshot，Reviewer 只接收签发标签，返回引用再经 allowlist 校验。若检索、模型或校验失败，保留原规则 issue 并记录降级。这个设计能防止未授权引用进入持久化注释，但不能证明不存在所有形式的 Prompt 注入或数据泄漏。

## 十、评测问答边界

面试官问指标时先说评测对象，不直接报一个百分比。

### 规则评测

- 80 条显式指令回归只证明规则接线。
- 合成自然文本与 14 例多文档复杂集均为开发者可见、原创回归集。
- 它们不能估计开放文本、人工盲测或商业游戏场景的真实准确率。

### Retrieval holdout

- 28 问、55 条期望证据；混合 Recall@5 为 81.82%，All-evidence@5 为 71.43%。
- 低词面 Recall@5 为 79.31%，差 1 条未过门槛；总 gate 为 false。
- 混合 Recall 与 dense 相同，因此不能声称显著优于两个基线。

### Reviewer A/B

- 固定 12 例，每例重复 3 次；严格全重复正确由 4/12 到 7/12。
- 情境例外从 1/4 到 3/4，黄金证据覆盖从 0/12 到 8/12，106/106 引用均在 allowlist。
- absolute gate 失败，`insufficient_evidence` 为 0/4；结论只能是“真实闭环已运行并暴露清晰失败”，不是“质量可上线”。

### Evidence Investigator 首次真实 E2E

- 被评代码为 commit `618ca991`，运行时报告模型别名 `deepseek-v4-pro`；别名不是上游权重身份证明。
- 为隔离归因，运行时关闭通用模型抽取、IssueEvidenceReviewer 与旧修复 Agent，只保留确定性主链路、Evidence Investigator 和真实 embedding/RAG；DEV 与 holdout 的 capability isolation 均为全案例通过。
- 两次连续 DEV 都是 8/8：正例 5/5、负例主动弃答 3/3、正常终止 8/8、错误新增 0。pair checker v3 验证两次运行配置指纹相同且时间不重叠；这只是进入 holdout 的开发门槛，不是开放质量成绩。
- DEV 1 的全案例编排墙钟 P50/P95 为 9.250/10.485 秒，DEV 2 为 10.344/12.312 秒；它包括建项、上传、异步分析、诊断和取问题，不是模型接口延迟或生产 SLA。
- 首次 10 例冻结 holdout 为 8/10：TP 4、FN 1、TN 4、FP 1，precision/recall 均为 80%，10/10 正常终止且无执行类失败。
- Agent 提交 8 个候选，promotion 接受 4 个目标正例、拒绝 4 个，没有错误候选通过 promotion；但系统确定性主链路产生 1 个 FP，因此不能说“系统零误报”。
- holdout 已封存且不再用于 Prompt、规则、阈值或 promotion 调参。10 例原创小样本不能外推到开放故事、商业剧情、生产质量或安全性。

### 旧 LangGraph 修复 Agent 验收

- 真实 Provider 只用于旧 Agent 阶段，主抽取候选是冻结 manifest 合成注入。
- 严格正确 59/90，完整 gate 失败；holdout 有超时、过度弃答和证据覆盖失败。
- 服务端接受的安全违规为 0 只能说明该套件中边界未被突破，不能说明输出质量或安全性全面达标。

## 十一、行为面与 STAR 素材

STAR 指情境、任务、行动、结果。结果既可以是成功指标，也可以是发现方案不成立并及时收缩范围。

准备以下五类素材，每次模拟只问其中一题：

1. **在需求不确定时推进**：LoreGuard 从“聊天式判断”收敛为证据化规则主路，说明如何定义验收标准。
2. **处理失败实验**：Reviewer/旧修复 Agent gate 失败，或 Investigator 首次 holdout 出现漏报与系统误报；说明如何分类错误、保留结果并避免复用 holdout 调参。
3. **质量与交付冲突**：在简历停止线选择先投递学习，而不是继续堆叠多智能体。
4. **跨角色理解需求**：从编剧/小说作者体验反馈反推 DOCX、直接正文、按需图和错误抽取治理。只说真实发生的用户反馈，不虚构公司协作。
5. **AI 辅助开发**：说明如何拆任务、让工具辅助、自己审 diff 与测试，并对最终结果负责。

华为实习可用于跨团队、协议理解或独立交付题，但必须以本人真实经历补充人物、冲突和结果；本指南不替用户虚构行为故事。

## 十二、面试红线速查

### 可以说

- 真实 embedding、pgvector 精确余弦检索、keyword+dense RRF 已接入 Evidence RAG。
- Reviewer 消费获授权 RAG 证据，但只追加注释。
- Evidence Investigator 已实现 Provider 原生工具调用、冻结范围内搜索/读取、单候选提交/弃答、确定性 promotion 和安全 trace。
- 旧 LangGraph 受限修复循环也已实现，但它使用应用层 JSON 动作且真实质量 gate 失败。
- 输入冻结、幂等认领、lease/heartbeat、取消检查点和持久化 SSE 已实现。
- 测试与评测包含失败样本，项目达到了简历展示停止线。

### 必须带边界说

- RAG 指标：同时说低词面门槛未过和未显著领先 dense。
- Reviewer 改善：同时说 absolute gate 失败与信息不足类失败。
- Evidence Investigator：同时说默认关闭、首次 holdout 仅 10 例且为 8/10、系统确定性主链路仍有 1 个 FP、无生产质量结论。
- 旧修复 Agent：同时说使用应用层 JSON 而非原生工具调用、候选为合成注入、完整 gate 失败；不要与 Investigator 合并报成绩。
- 并发：只能说 2 worker/20 任务队列烟测，不称高并发压测。
- AI 独立项目：独立负责目标、决策和验收，同时坦诚使用 AI 编程工具辅助。

### 不能说

- 公网 Demo、真实用户上线、商业游戏语料验证或长篇容量结论。
- 生产高并发、生产 SLA、完整高可用或 exactly-once 模型计费。
- 多智能体、旧修复 Agent 使用原生工具调用、所有 AI 路径都由 Agent 自主执行、HNSW、图数据库、Kubernetes 或生产级 OpenTelemetry。
- RAG、Reviewer、旧 Agent 或 Investigator 已经达到开放文本或生产质量。
- 任何 API Key、模型服务地址、私密 Prompt、原始响应或内部文本。

## 十三、开始模拟面试的顺序

阶段 0 先做表达基线，不预习标准答案：

1. 60 秒介绍 LoreGuard。
2. 解释一次分析的数据流。
3. 解释模型和规则的分工。
4. 解释原生工具调用为什么只是请求，不等于工具已执行。
5. 画出 Investigator 的搜索、读取、提交/弃答状态机。
6. 解释冻结引用、确定性 promotion 与安全 trace。
7. 解释 Evidence RAG 与 Reviewer 的真实实现及差异。
8. 用首次 DEV pair 和 holdout 说明“测到了什么、没证明什么”。
9. 解释 Celery/SSE 和失败恢复。
10. 只有在简历 LangGraph 表述或面试官追问时，解释旧修复 Agent 与 Investigator 的区别。
11. 回答一题华为实习深挖。
12. 回答一题 easyMeeting 深挖。
13. 回答一题行为面。

教练一次只能发送当前序号的一题。每题通过条件记录在 [面试进度模板](interview-progress-template.md)。如果出现知识缺口，就暂停该题并跳转到学习指南的对应模块；知识验证通过后再回到原题重新作答。

## 十四、投递后的最低面试门槛

投递可以与学习同步开始，但进入技术面试前至少应做到：

- 60 秒项目介绍不看稿，结论与边界完整。
- 能画出主流水线并解释每一层为何存在。
- 能闭卷画出 Investigator 的四动作状态机，并说明模型请求、服务端执行和确定性 promotion 的权限边界。
- 能用自己的话区别本地主路候选、Evidence RAG、Reviewer、固定 semantic-label repair、旧 LangGraph 修复 Agent 和 Evidence Investigator。
- 能定位 Provider 原生调用、Investigator 状态机、RAG、promotion、service 接线和 E2E runner，并为其中至少四处指出测试入口。
- 能解释安全 trace 保存什么、明确不保存什么。
- 能主动解释 retrieval、Reviewer、旧 Agent 的失败结果，以及 Investigator 的 DEV pair 与首次 8/10 holdout；不能把小样本或正常终止率偷换成开放质量。
- 能针对高并发、长篇和多智能体问题清楚区分“当前实现”与“未来设计”。
- 华为与 easyMeeting 的每条简历描述都能提供代码或测试依据。
- 准备两个真实 STAR 故事：一个成功交付，一个失败或取舍。

未达到其中某项并不阻塞投递，只决定接下来优先练什么。
