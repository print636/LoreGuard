# LoreGuard 知识学习路线

更新：2026-09-09。

## 这份路线解决什么问题

这是一套“真正读懂并能维护 LoreGuard”的课程，不是背面试答案。学习完成后，你应当能够：

1. 独立完成一次分析演示，并判断这次结果究竟来自基线、模型增强还是降级路径。
2. 沿着一次请求追踪前端、FastAPI、任务执行、抽取、规则、RAG、持久化和 SSE 返回。
3. 用自己的话解释关键设计为什么这样做，以及它牺牲了什么。
4. 根据失败日志和测试定位问题，完成一个小改动并为它补测试。
5. 准确区分已实现能力、未通过质量门槛的实验原型和未来计划。

当前阶段是“项目已经达到简历工程展示停止线，开始投递并完成本人学习验收”。除非真实体验暴露阻断问题，本路线不以继续堆功能为目标。项目事实以[项目事实清单](project-facts.md)为准，完成度以[当前完成度](completion-status.md)为准。

## 学习和面试准备不是一件事

| 事项 | 本路线负责 | 面试训练负责 |
| --- | --- | --- |
| 目标 | 建立可迁移的技术理解，能读代码、排错和维护 | 在有限时间内准确表达经历并应对追问 |
| 判断掌握的证据 | 能预测运行、画数据流、修改代码、解释失败 | 能在 1/5/15 分钟内组织答案并守住事实边界 |
| 学习方式 | 讲解、代码追踪、动手实验、故障演练、间隔复习 | 模拟追问、答案压缩、项目攻防、行为题复盘 |
| 不应做的事 | 为了“像面试答案”跳过原理和代码 | 在现场临时学习从未理解的概念 |

完成知识模块不等于完成面试准备；面试讲稿见[面试指南](interview-guide.md)，但学习期间不要照着讲稿背诵。

## 教学约定

### 学习者起点

暂按“会 Java/Python 基础，知道 HTTP、数据库和常见数据结构，但尚未系统理解本项目”起步。任何更高水平都必须由回答或动手证据确认，不能因为简历写过某个技术就默认已经掌握。

### 每轮如何进行

- 每轮最多提出一个需要你回答的问题。不能一次发送问卷或连续追问多个概念。
- 首次出现术语时先给白话定义，再给项目里的正式名称。例如：先说“把文本变成便于比较的数字列表”，再引入 `embedding`（向量表示）。
- 默认使用“短讲解 → 一个预测或选择 → 运行/读代码 → 复述 → 反馈”的节奏；根据表现切换为画图、对比、故障定位或小改动，不拘泥于一问一答。
- 答错时先定位是哪一级断点：术语不懂、前置知识缺失、数据流断裂，还是知道概念但不能落到代码；一次只补一个断点。
- 教练不能只问“懂了吗”。掌握必须表现为可观察证据，例如预测测试结果、指出代码入口、解释一个反例或亲手完成改动。
- 每次结束只保留一个“下一问题”，并把当前模块、已掌握证据和断点写入[学习进度模板](learning-progress-template.md)。

### 临时岔题如何处理

你可以随时问与上一轮无关的问题。教练先直接回答当前问题，再写一行“返回点：模块 X / 步骤 Y / 下一问题 Z”，然后回到原路线。若岔题暴露必要的前置缺口，把它作为一个短补丁模块插入；若与目标岗位无关，只回答到解决当前疑问所需的深度，不扩展成新课程。

### 状态标签

- `未学`：还没有可靠证据。
- `理解中`：看过或听过，但尚不能独立解释。
- `能解释`：可以不用原文复述，并回答一个反例。
- `能操作`：能够运行、定位或修改相关代码。
- `已掌握`：至少同时具备“能解释”和“能操作”，并在 1 天后的复习中通过。

## 先诊断，不先灌输

诊断按依赖顺序进行，但教练每次只问下列一个问题。回答充分的层级可以跳过，回答模糊时只补当前层级。

1. `D0 产品`：不看文档，你认为 LoreGuard 最重要解决的一个用户问题是什么？
2. `D1 Web`：从点击“分析当前项目”到页面出现进度，你猜浏览器和后端之间发生了什么？
3. `D2 数据`：为什么更新同名文档后，已经开始的旧任务不应该读到新正文？
4. `D3 模型`：如果模型返回一段看似合理但原文中不存在的事实，系统应该在哪些位置阻止它？
5. `D4 RAG`：关键词检索和向量检索分别更擅长找什么，为什么相似文本不能直接等同于剧情冲突？
6. `D5 可靠性`：Celery 把同一任务投递两次时，怎样避免两名 worker 同时提交结果？
7. `D6 验证`：一个固定小测试集得分很高，为什么仍不能说系统对开放长篇故事准确？

诊断不是考试，不要求第一次答对。它只决定课程从哪里开始。当前首次教学从 `D0` 开始，且这一轮只问 `D0`。

## 学习地图与先修关系

```text
M0 产品闭环与证据
 └─M1 领域模型与结构化记录
    └─M2 无模型基线与五类规则
       └─M3 模型增强与语义质量门
          ├─M4 API、数据库与运行快照
          │  └─M5 Celery、SSE 与可靠性
          └─M6 Evidence RAG
             └─M7 Evidence Reviewer
M3 ───────────└─M8 受限 Agent 原型
M2+M5+M6+M7+M8 ──M9 评测、可观测性与安全边界
M0–M9 ───────────M10 独立维护验收
```

建议按 M0 到 M10 前进。M4 与 M6 在学完 M3 后可以互换；其余不要为了追热门术语跳过先修。

## 模块 M0：产品闭环与“证据先于判断”

**目标**

- 说清使用者、输入、输出和五类问题，不把产品误解为自动续写器。
- 独立操作“新建项目 → 导入 → 分析 → 看证据 → 提交反馈”。
- 看懂页面显示的模型模式和 warning，不把基线兜底的正确结果算成模型成功。

**术语先说人话**

- 证据跨度（`EvidenceSpan`）：一条判断对应的原文文件、起止行和文本。
- 运行（`AnalysisRun`）：一次固定输入上的分析任务，而不是“当前项目永远变化的结果”。
- 基线（baseline）：不调用大模型、只用本地可重复逻辑完成的最低可用流程。

**项目证据**

- [README 中文版](../README.zh-CN.md)的“本地项目工作流”“输入格式”。
- [两分钟演示脚本](demo-script.md)。
- `frontend/src/App.tsx`、`app/main.py` 中项目、文档、运行、问题和反馈接口。
- `tests/test_api.py` 中 `test_end_to_end_analysis_and_feedback`、`test_single_plain_body_document_can_complete_without_world_document`。

**动手任务**

1. 用 `启动LoreGuard.cmd` 启动本地应用。
2. 先运行内置原创样例，任选一个问题核对两侧原文行号。
3. 再只上传一份不含独立世界观的正文并运行，确认它可以完成而不是要求必须有设定集。
4. 对一个问题提交反馈，刷新后确认结果和历史仍可恢复。

**掌握门槛**

- 不看 README，能在一张图中画出输入到问题报告的六个主要节点。
- 面对“问题为 0”，能先检查抽取记录、warning 和运行模式，而不是直接断言故事无矛盾。
- 1 天后可再次独立完成上述流程。

## 模块 M1：领域模型与结构化记录

**目标**

- 理解为什么系统先把文字转为结构化状态，再做比较。
- 分清“类型合同”“通用运行记录”和“最终问题”。
- 能从一段原文手写一条最小 `ParsedDirective` 和对应证据。

**术语先说人话**

- 结构化：把散文变成固定字段，例如“谁 / 什么属性 / 值 / 时间”。
- 规范化（canonicalization）：把不同写法统一成可比较的表示，而不是判断哪一句为真。
- 来源轨迹（provenance）：记录某条最终状态来自基线、模型还是两者共同命中。

**项目证据**

- `app/domain.py`：`EvidenceSpan`、`ParsedDirective`、`ConsistencyIssue`、五类 `IssueCategory`。
- `app/pipeline.py`：`DocumentInput`、`PipelineResult`、`NarrativeExtractor`、`ConsistencyChecker`。
- `app/db.py`：项目、文档、运行快照、执行、状态记录、问题和事件表。
- `tests/test_fact_merge.py`、`tests/test_provenance.py`。

**动手任务**

- 把“1026-04-03 12:00，林澈得知星门口令”手写为 `knows` 记录，标出 `character`、`fact`、`time` 和证据行。
- 再把它改成角色台词“林澈说自己知道口令”，预测为什么来源权威性不同，然后用测试验证。

**掌握门槛**

- 能解释 `NarrativeEntity` 等清晰类型与持久化层的通用 `kind/attrs/evidence` 记录为何可以并存。
- 能说明“规范化”和“冲突裁决”不同，且不会把实体共现当成别名声明。

## 模块 M2：无模型基线与五类确定性规则

**目标**

- 跟踪无 Key 时从一行文本到一个问题的完整调用链。
- 理解保守抽取“宁可漏掉也不乱认”的产品权衡。
- 能说清事实、地点、知识、物品和世界规则五类检查需要的最小状态对。

**术语先说人话**

- 确定性：同一输入和版本下，代码按明确条件得到可复现结果，不依赖模型随机输出。
- 语义质量门：先判断候选是否足够像“作者确认的事实”，问题、假设、传闻等仍可展示但不能进入硬规则。
- 词面支持：候选字段的关键内容必须能在授权原文中找到，而不是模型自行补全。

**项目证据**

- `app/parser.py` → `app/natural.py`：显式 `@directive` 与保守中文句式抽取。
- `app/candidate_normalizer.py`、`app/aliases.py`：物品使用、规则、知识、移动许可、身体状态与别名归一。
- `app/semantic_quality.py`：问题、假设、传闻和来源范围的降级或拦截。
- `app/rules.py`：五类最终裁决。
- `tests/test_engine.py`、`tests/test_semantic_quality.py`。

**代码追踪**

从 `BaselineExtractor.extract` 开始，依次找到：

```text
parse_document
→ extract_natural_line
→ DeterministicCandidateNormalizer.enrich
→ apply_semantic_quality_gate_with_provenance
→ canonicalize_entities
→ RuleChecker.check_with_candidates
→ detect_issues
```

注意：`app/retrieval.py` 的本地关键词、稳定字符 n-gram 和实体图在当前主路留下候选轨迹，但 `detect_issues` 仍检查全量规范记录；候选 `consumed=true` 不能解释为排序改变了裁决。

**动手任务**

1. 先预测“如果吃下推车上的食物，是会补充体力吗？”是否应成为确定事实。
2. 运行单测：

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/test_semantic_quality.py::BaselineSemanticQualityTests::test_reported_user_paragraph_keeps_rules_and_questions_without_fake_fact -q
   ```

3. 在测试副本中把问句改成明确叙述，对比抽取记录；不要直接修改生产规则。
4. 任选一种问题，从 `detect_issues` 写出“需要哪些索引、比较什么字段、有哪些例外”。

**掌握门槛**

- 能解释为什么用户截图中的长问句曾被误抽成事实，以及当前语义质量门从哪些维度阻止它。
- 能手工构造五类问题各自的最小正例，并为任意一类补一个不应报警的反例。
- 能解释规则适合时序/状态约束，而向量相似度不能直接替代它。

## 模块 M3：模型增强、结构化输出与失败降级

**目标**

- 理解模型只扩大开放文本的状态候选覆盖，不直接成为最终裁判。
- 跟踪分块、Prompt、Pydantic 校验、证据回填、语义校验、去重合并和熔断。
- 能根据诊断区分完整成功、部分降级、全部基线兜底和主动关闭模型。

**术语先说人话**

- OpenAI-compatible Provider：用近似 OpenAI 请求格式对接不同模型服务的适配层。
- Pydantic 合同：用代码规定模型 JSON 必须有哪些字段、类型和值域，非法结果不能直接进入业务。
- 熔断：确认上游持续故障后，短路本次运行的后续调用，避免每个分块都重复等待。
- Token 预算：在请求前估算并限制模型可消耗的文本额度；兼容服务不回 usage 时也不把成本当作零。

**项目证据**

- `app/chunking.py`：保留全局行号的模型分块。
- `app/provider.py`：超时、429、5xx、非法响应、有限重试与安全遥测。
- `app/model_extractor.py`：8 类结构化输出、证据绑定、词面支持、模型/基线合并、修复分流。
- `app/pipeline.py`：运行级汇总和来源轨迹。
- `tests/test_model_extractor.py`、`tests/test_model_execution.py`、`tests/test_batch_model_extraction.py`。

**动手任务**

- 画出一条模型记录进入规则前必须通过的关卡，并在每个关卡写一个会被拒绝的例子。
- 运行“模型超时”和“非法记录”测试，预测基线记录是否仍保留：

  ```powershell
  .venv\Scripts\python.exe -m pytest tests/test_model_extractor.py::ProviderTests::test_timeout_exhaustion_has_safe_error tests/test_model_extractor.py::ModelExtractorTests::test_all_invalid_records_fall_back_without_losing_baseline -q
  ```

- 在页面或 API 的诊断中找到 `attempted_chunks`、`succeeded_chunks`、`failed_chunks`、`skipped_chunks` 和无效记录计数，给出这次运行模式的结论。

**掌握门槛**

- 能准确复述：“模型负责读懂更多写法并提出可验证候选，规则基于证据化状态作裁决。”
- 能解释为什么“结果正确”不等于“模型调用成功”。
- 能在不看代码的情况下列出至少四条失败保护，并能在代码中找到其中两条。

## 模块 M4：API、数据库与冻结运行输入

**目标**

- 理解 REST 接口、SQLAlchemy 数据模型和事务边界。
- 解释文档版本、活动版本、运行快照和重试之间的关系。
- 跟踪一次创建运行到结果持久化的主要数据库记录。

**术语先说人话**

- REST：用 HTTP 方法和资源路径表达创建、查询或更新操作的一组接口约定。
- 事务：一组数据库改动要么一起成功，要么一起撤销。
- 快照（snapshot）：任务开始时复制并锁定所用文档版本和正文，之后编辑项目不会偷偷改变旧任务输入。
- 幂等：同一个操作重复执行，不会不断产生额外副作用。

**项目证据**

- `app/main.py`：项目、文档、分析、取消、重试、问题、诊断、图、时间线和反馈 API。
- `app/db.py`：`DocumentRow`、`AnalysisRunInputRow`、`AnalysisRunExecutionRow`、`RunEventRow` 等表。
- `app/service.py`：`capture_run_inputs`、`copy_run_inputs`、`_load_verified_snapshot`。
- `tests/test_api.py`、`tests/test_run_reliability.py`。

**动手任务**

1. 创建文档 v1 并启动分析；上传同名 v2 后重试旧任务。
2. 预测重试应使用 v1 还是 v2，再运行：

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/test_run_reliability.py::RunReliabilityTests::test_retry_copies_original_snapshot_not_current_document -q
   ```

3. 用数据库表或 API 响应证明结论，不只看页面文案。

**掌握门槛**

- 能画出 Project、Document、Run、RunInput、Issue、Feedback 的关系。
- 能解释为什么“按项目 ID 过滤”还不足以保护一次旧运行，必须绑定版本和内容哈希。
- 能指出一次写操作的事务边界和一个重复请求场景。

## 模块 M5：Celery、SSE 与任务可靠性

**目标**

- 理解 API 进程与 worker 的职责边界，以及 Redis/Celery 如何承载后台任务。
- 理解任务可能重复投递，所以需要认领、租约、心跳和提交围栏。
- 解释为什么单向进度使用 SSE，以及断线后怎样续传。

**术语先说人话**

- 队列：API 先登记任务，worker 稍后从待办列表取走执行，用户不必让一次 HTTP 请求一直等待。
- 租约（lease）：worker 对任务的有期限执行权；超时才允许另一 worker 接管。
- 心跳：执行者定期续租，证明自己仍活着。
- 提交围栏（fencing）：失去租约的旧 worker 即使后来恢复，也不能再写入最终结果。
- SSE：服务器通过一条 HTTP 长连接持续向浏览器推送单向事件。

**项目证据**

- `app/tasks.py`：Celery 任务入口与自动重试边界。
- `app/service.py`：认领、lease/heartbeat、检查点、取消和终态提交。
- `app/main.py`：`dispatch_analysis`、`stream_events`、取消和重试约束。
- `tests/test_run_reliability.py`、`tests/test_tasks.py`、`tests/test_api.py`。

**故障演练**

- 预测两个 worker 同时收到同一 `run_id` 时会发生什么，再运行 `test_conditional_claim_allows_only_one_concurrent_worker`。
- 预测 worker 在模型调用中超过租约时，心跳与围栏分别解决什么问题。
- 模拟 SSE 断线，用最后事件 ID 重连，确认终态包含 `status/error`。

**掌握门槛**

- 能区分“Celery 重试消息”和“用户重试一个失败运行”。
- 能说明现有机制降低重复执行/错误提交风险，但不能声称 exactly-once 模型计费。
- 能解释本机 2 worker × 20 任务只是队列和隔离烟测，不是生产高并发 SLA。

## 模块 M6：真实 Evidence RAG

**目标**

- 理解 RAG 在本项目中只为已产生的问题补充上下文证据，不替代规则裁决。
- 跟踪分块、embedding、pgvector、关键词/向量排序、RRF 合并和 snapshot/profile 隔离。
- 能读懂 Recall@5、All-evidence@5、MRR 和低词面样本的含义。

**术语先说人话**

- RAG（检索增强生成）：先从受控资料中找证据，再把证据交给模型回答。
- embedding（向量表示）：把一段文字转成数字列表，使意思接近的文本通常距离更近。
- dense retrieval（向量检索）：按向量相似度找文本；它能发现不同措辞，但也可能只找到“主题像”的段落。
- pgvector：PostgreSQL 的向量类型和距离计算扩展。
- RRF（倒数排名融合）：不直接比较两种检索分数，而按各自排名给候选累积分数。
- profile：模型、维度、归一化、查询变换等 embedding 配置的版本身份；配置不同的向量不能混用。

**项目证据**

- `app/evidence_chunks.py`：带快照身份的行号分块。
- `app/embeddings.py`：独立 embedding Provider 和版本化 `EmbeddingProfile`。
- `app/evidence_store.py`：chunk/vector 持久化、覆盖检查与精确快照过滤。
- `app/evidence_rag.py`：索引协调、keyword/dense/RRF 检索和安全诊断。
- `tests/test_evidence_rag.py`、`tests/test_embedding_persistence.py`、`tests/test_pgvector_integration.py`。
- [Evidence Retrieval v1](evidence-retrieval-v1.md)与[首次 holdout](evidence-retrieval-v1-holdout.md)。

**动手任务**

1. 自己计算一个两路 Top-3 的简化 RRF 排名。
2. 预测“错误项目或旧版本的高相似段落”能否进入结果，再运行：

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/test_evidence_rag.py::EvidenceRagTests::test_retriever_requires_exact_explicit_snapshot_authorization tests/test_embedding_persistence.py::EvidencePersistenceTests::test_store_is_idempotent_and_snapshot_filter_is_exact -q
   ```

3. 从冻结报告中任选一个低词面漏检，说明是查询、分块、embedding 还是 Top-K 问题；不能在没有证据时武断归因。

**掌握门槛**

- 能区别 `app/retrieval.py` 的本地主路候选信号与 `app/evidence_rag.py` 的真实 BGE/pgvector 通路。
- 能解释“混合 Recall@5 与 dense 相同”为什么不能声称混合检索显著更优。
- 能主动报出边界：首次 holdout 总 gate 为 `false`，低词面 23/29，差一条未达门槛。

## 模块 M7：后置 Evidence Reviewer

**目标**

- 理解 Reviewer 的输入是规则 issue 与获授权检索证据，输出只是独立 AI 注释。
- 解释服务端引用标签和 allowlist 如何限制模型引用范围。
- 能分析 Reviewer 对情境例外有帮助但对“信息不足”失败的原因假设与验证方法。

**术语先说人话**

- allowlist（允许名单）：模型只能引用服务端这次明确发给它的证据编号，其他编号或来源一律拒绝。
- 后置复核：先由规则产生问题，再让模型看补充上下文；模型不能倒过来静默改掉规则结果。
- A/B：保持其他条件相同，只改变是否提供 RAG 证据，比较两组结果。

**项目证据**

- `app/issue_evidence_review.py`：问题查询、批量 Prompt、引用解析、注释和失败降级。
- `app/service.py`：使用冻结运行输入建立索引，并只追加 issue metadata。
- `tests/test_issue_evidence_review.py`。
- [Issue Review v1 结果](issue-review-v1-result-20260907.md)。

**故障演练**

- 预测模型伪造一个未授权引用时，是丢掉单条引用、单个问题还是整批响应；再运行 `test_forged_citation_rejects_entire_batch` 查证。
- 让检索不可用或模型超时，确认原规则 issue 仍在，并指出降级诊断位置。
- 对 12 例 A/B 结果写三句话：观察到了什么、不能推出什么、下一版如何用新冻结集验证。

**掌握门槛**

- 能画出“规则 issue → 冻结快照索引 → 检索 → Provider → allowlist 校验 → 注释”的闭环。
- 能准确说明真实结果是 local-context 4/12、rag-evidence 7/12，但 absolute gate 失败且 `insufficient_evidence` 为 0/4。
- 不把 Reviewer 说成 Agent，也不把注释说成最终裁决。

## 模块 M8：受限修复 Agent 原型

**目标**

- 理解本项目何处存在多步模型循环，以及为什么它被严格限制并默认关闭。
- 分清应用层 JSON 动作、Provider 原生 function calling 和多智能体。
- 能根据 trace 重放一次 `READ_SPAN → PATCH_RECORDS` 或 `ABSTAIN` 路径。

**术语先说人话**

- Agent：模型不只回答一次，而是在受控循环中根据中间结果选择下一步动作。
- 工具调用：模型请求程序执行某个有明确参数和权限边界的动作；这里使用自定义 JSON 动作，不是 Provider 原生 `tool_calls`。
- LangGraph `StateGraph`：把“当前状态、执行节点和下一步条件”写成可检查的图式编排。
- 最小权限：只开放完成任务必需的文档窗口、字段、轮次和额度。

**项目证据**

- `app/review_agent.py`：`READ_SPAN`、`PATCH_RECORDS`、`ABSTAIN` 合同，窗口、字段和预算限制。
- `app/model_extractor.py`：只有词面支持失败候选进入 Agent；固定语义标签 repair 是另一条路径。
- `tests/test_review_agent.py`。
- [受限 Agent 第一阶段](review-agent-phase1.md)与[v2 完整 checkpoint](review-agent-v2-full-checkpoint-20260906.md)。

**动手任务**

1. 用白纸模拟一轮 Agent：候选缺少词面支持 → 读一个允许窗口 → 只修补允许字段 → 重新校验。
2. 分别预测越权跨文档读取、重复工具动作、Token 超限和禁改语义字段的终态。
3. 运行一个成功路径与一个越权路径：

   ```powershell
   .venv\Scripts\python.exe -m pytest tests/test_review_agent.py::ReviewAgentUnitTests::test_model_dynamically_reads_then_patches_successfully tests/test_review_agent.py::ReviewAgentUnitTests::test_cross_document_read_abstains_before_tool_execution -q
   ```

**掌握门槛**

- 能解释为何“有 LangGraph”不等于“多智能体”，以及为何该实现不是原生 function calling。
- 能说明完整 gate 失败、严格正确 59/90、候选由冻结 manifest 合成注入，所以不能声称端到端 Agent 质量收益。
- 能指出 Agent、固定标签 repair 和 Evidence Reviewer 三条路径的输入、输出和权限差异。

## 模块 M9：评测、可观测性与安全边界

**目标**

- 理解单元、集成、端到端、冻结回归和 A/B 各自回答什么问题。
- 会计算 Precision、Recall、F1、Recall@K，并判断指标边界。
- 能使用结构化诊断、Prometheus 指标和失败分类定位问题，而不是依赖一条人类可读 warning。

**术语先说人话**

- 冻结集：开始正式评测后，不因结果不好偷偷改题或覆盖旧报告。
- 数据泄漏：测试题或答案参与了 Prompt、调参或规则修改，使成绩看起来比真实泛化更好。
- 可观测性：让运行状态、耗时、失败类别和资源使用可查询，而不是“出错了只能猜”。
- gate：评测前写好的通过条件；差一条仍然是不通过。

**项目证据**

- `app/domain.py` 的执行诊断与安全 allowlist；`app/service.py` 的状态和用量持久化；`app/main.py` 的 `/metrics` 与诊断接口。
- `tests/`、`scripts/run_*evaluation.py`、`data/`、`artifacts/`。
- [简历就绪结论](resume-readiness.md)、[项目事实清单](project-facts.md)。

**动手任务**

- 给定 TP/FP/FN 手算 Precision、Recall 和 F1，再解释“证据对完全命中”与“至少命中一条证据”的区别。
- 找出一个模型失败测试，列出日志/诊断允许保存和明确禁止保存的字段。
- 选择一个未通过 gate，提出“假设 → 改动 → 新开发集 → 新冻结测试集 → 比较”的下一轮实验，不覆盖原报告。

**掌握门槛**

- 能指出所有公开成绩都来自原创、开发者可见小型冻结集，不能外推到开放长篇或生产准确率。
- 能解释 API Key、endpoint、Prompt、正文和模型原始响应为何不进入公开诊断。
- 能根据症状在“抽取、语义门、规则、检索、Reviewer、任务执行”中定位首查层。

## 模块 M10：独立维护验收

**目标**

- 在没有逐步提示时完成一次小型、可验证且不越界的项目改动。
- 证明自己不仅能复述架构，也能保持合同、失败降级和测试边界。

**可选任务（只选一个）**

- 为现有一种规则补充一个真实反例，先写失败测试，再做最小修复。
- 为一类诊断增加安全且有界的计数，证明不会泄露正文或把缺失值伪造成零。
- 为一个文档版本/SSE 边界补集成测试，不新增产品功能。

**执行约束**

1. 先写一句问题定义和一句不做什么。
2. 先预测会影响哪些模块和测试。
3. 保持改动最小；不得为了通过单个样本硬编码故事内容。
4. 运行最窄测试，再运行相关测试集；记录失败和修复证据。
5. 自己更新一段相关文档，明确已实现与仍未解决的边界。

**最终掌握门槛**

- 可以从页面动作一路定位到具体后端函数和数据库记录。
- 可以在白板上画出规则主路、可选 RAG Reviewer 支路和受限 Agent 原型，并讲清三者不会互相冒充。
- 可以解释至少两次失败设计：Provider 故障如何降级、worker 失去租约如何阻止错误提交。
- 可以独立完成上述小改动、测试和文档更新。
- 在第 1、3、7 天的间隔复习中，核心概念均达到“能解释”，其中至少一个模块达到“能操作”。

## 推荐学习节奏

每次 45–75 分钟，避免一次把所有术语塞满。

| 阶段 | 模块 | 建议产出 |
| --- | --- | --- |
| 第 0 阶段：定位起点 | D0–D6 | 知识边界和跳过/补课决定 |
| 第 1 阶段：先能用 | M0–M2 | 完整体验、五类规则图、一个误报/漏报解释 |
| 第 2 阶段：理解 AI | M3、M6、M7、M8 | 三条 AI 路径对比图、一次失败降级演练 |
| 第 3 阶段：理解后端 | M4–M5 | 数据表关系、任务状态机、SSE 断线演练 |
| 第 4 阶段：会验证 | M9 | 一页指标与边界说明、一个新实验设计 |
| 第 5 阶段：会维护 | M10 | 一个小改动、相关测试和文档 |

每学完两个新模块，下一次先做 10 分钟闭卷回忆，不重新阅读原文：画数据流、解释三个术语、预测一个失败。第 1、3、7 天复习失败项，已稳定掌握的内容不做机械重复。

## 与目标岗位的范围控制

### 必学

- Python/FastAPI 服务接口，SQLAlchemy/PostgreSQL，Redis/Celery，SSE 与失败恢复。
- LLM 结构化输出、能力边界、RAG、embedding、pgvector、RRF、受限 Agent、工具权限。
- 高可用基础：幂等、租约、心跳、重试、熔断、限流、预算、可观测性。
- 评测设计、数据泄漏、误报/漏报分析和诚实实验迭代。
- AI 编程工具的有效使用要落到需求拆解、验证、代码审查和事实核对，而不只是列工具名。

### 只需知道边界，不深入

- 多智能体：当前未实现；知道何时才值得拆分角色即可。
- HNSW/IVFFlat、Kubernetes、图数据库、完整 OpenTelemetry：当前未实现且不阻塞投递。
- 百万字剧情和生产高并发容量：保留扩展方向，但没有真实验证，不学习不存在的性能结论。

### 当前不学

- 大模型预训练、微调数学推导、CUDA kernel、推理引擎内部优化。
- 与 AI 应用后端/Agent 工程岗位无直接关系的全套前端视觉实现细节。

## 最小命令清单

在项目根目录执行：

```powershell
# 启动本地体验
.\启动LoreGuard.cmd

# 最小规则与语义门测试
.venv\Scripts\python.exe -m pytest tests/test_engine.py tests/test_semantic_quality.py -q

# 模型边界测试（使用 Mock，不需要真实 Key）
.venv\Scripts\python.exe -m pytest tests/test_model_extractor.py tests/test_model_execution.py -q

# RAG 与 Reviewer 逻辑测试
.venv\Scripts\python.exe -m pytest tests/test_evidence_rag.py tests/test_issue_evidence_review.py -q

# 任务可靠性测试
.venv\Scripts\python.exe -m pytest tests/test_run_reliability.py tests/test_tasks.py -q
```

不要把“测试通过”直接等同于“开放故事质量达标”。命令用于验证明确合同，质量结论仍以冻结评测及其边界为准。

## 学完后的自检清单

- [ ] 我可以独立运行普通正文、DOCX 和内置复杂样例。
- [ ] 我能说明无模型、完整模型增强、部分降级和 Reviewer 开启时的差异。
- [ ] 我能从 `app/main.py` 追到 `app/service.py`、`app/pipeline.py` 和最终数据库结果。
- [ ] 我能解释五类规则的最小状态对和至少一个困难反例。
- [ ] 我能区别本地 n-gram 候选轨迹、真实 Evidence RAG 和规则裁决。
- [ ] 我能解释 snapshot/profile 隔离、pgvector 精确检索和 RRF。
- [ ] 我能解释租约/心跳/围栏和持久化 SSE 续传。
- [ ] 我能说明 Reviewer、修复 Agent、固定 repair 三者的边界。
- [ ] 我能主动讲清 retrieval 与 Reviewer 未通过的 gate，而不粉饰。
- [ ] 我完成过一个小改动，并有测试和文档证据。

全勾选后，再进入“项目表达与模拟面试”阶段；没有勾选的条目只补对应模块，不从头重学全部内容。
