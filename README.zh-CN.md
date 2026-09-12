# LoreGuard

面向游戏编剧与叙事设计团队的剧情一致性审查平台。LoreGuard 将世界观、角色设定与章节文本抽取为可追溯的状态记录，通过确定性检查定位时间线、知识、物品、地点和世界规则冲突。

> 本项目只分析用户拥有权利的原创或授权文本，不连接、控制或自动化任何商业游戏。

## 已实现的 MVP

- 一键简单/复杂原创样例、自然文本粘贴及 Markdown、TXT、JSON、标准 DOCX 多文件上传
- 本地项目列表、文档版本历史和运行历史；刷新页面后可恢复最近任务与结果
- 无 API Key 抽取明确中文句式，并展示抽取记录与未抽取提示
- 配置 OpenAI-compatible 模型后，从普通中文故事中结构化抽取 8 类叙事记录；模型失败自动保留基线结果
- 实验性的受限证据修复 Agent 已接入但默认关闭；已完成主抽取候选冻结合成注入、仅 Agent 阶段调用真实 Provider 的验收，但质量 gate 未通过，当前只证明实现与安全边界
- 模型输入按全局行号分块，支持行重叠、超长单行和逐块失败隔离；全文基线不分块
- 仅依据明确“又名/简称/化名/代号”声明做项目级实体别名归一化，保留映射轨迹与歧义警告
- 本地 keyword + 稳定 SHA-256 字符 n-gram + canonical entity graph 三路候选排序并留下分数轨迹；当前 `consumed` 只表示候选类型与检查器兼容，最终仍由规则引擎检查全量记录，不能证明候选排序影响了裁决
- 可选 Evidence RAG 已真实运行固定 BGE embedding、PostgreSQL/pgvector 精确余弦检索与 keyword+dense RRF，并按项目、文档、版本、内容哈希、chunker 和 profile 隔离
- 可选 `IssueEvidenceReviewer` 将获授权的 RAG 证据交给模型复核，并把结论作为独立注释附在规则问题上；它不会删除、改写或替代规则结果，默认关闭
- 检查事实冲突、同时间多地点、知识越权、非持有者使用物品、世界规则冲突
- 每个问题返回两处证据、行号、严重度、置信度和修订建议
- 本地线程/Celery 两种后台执行方式、受状态约束的取消/重试与可断线续传的持久化 SSE 进度流
- 接受、误报、已解决反馈，含最新状态、重复提交抑制和完整审计历史
- 80 条显式指令规则回归、100 条含困难负样本的合成自然中文评测，以及 14 例原创多文档复杂验收集
- React 审查界面、Docker Compose、GitHub Actions 与 Prometheus 指标
- 已完成运行的关系图与保守时间线，支持类别筛选、问题关系筛选、证据详情和问题联动高亮；展示不会再次调用模型
- 同名文档任意两个版本的本地行级差异；项目切换状态隔离、上传结果汇总、错误/空状态与反馈防重复提交

## 快速开始

### Windows 一键体验

先安装 Python 3.12 和 Node.js 22（包含 Corepack），再双击项目根目录的 `启动LoreGuard.cmd`。启动器会创建/同步 `.venv`，并在网页产物缺失或源码更新后自动安装前端依赖和重新构建，随后浏览器打开 <http://127.0.0.1:8000>。进入页面后点击“一键运行原创样例”。

### Docker Compose

```bash
docker compose up --build
```

- Web：<http://localhost:8080>
- API：<http://localhost:8000/docs>
- Prometheus：<http://localhost:9090>

GitHub Actions 会实际构建并启动整套 Compose，在关闭模型的隔离环境中经 Celery worker 完成复杂样例分析，并断言五类问题、关系图、时间线和 SSE 终态；这不是只做 YAML 语法检查。

#### 可选：本地真实 embedding 服务

默认 `docker compose up` 始终保持 embedding 关闭，不会拉取模型或启动 embedding 容器。需要验证本地真实向量时，使用独立 overlay：

```bash
docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.rag.yml up --build
docker compose --env-file .env.example -f docker-compose.yml -f docker-compose.rag.yml --profile rag-smoke run --rm tei-smoke
```

上面的命令只启动和验证本地向量能力，不会自动调用聊天模型。要在页面体验 AI 证据复核，请把 `.env.example` 复制为不提交的 `.env`，填写服务端模型配置并设置 `ENABLE_ISSUE_EVIDENCE_REVIEW=true`，再将命令中的 `--env-file .env.example` 改为 `--env-file .env`。分析完成后，在问题卡片查看“AI 复核”、`hybrid` 标识和获授权引用；该注释不会删除或改写规则问题。

overlay 使用固定 digest 的 TEI 1.9.3 CPU 镜像和固定 revision 的 `BAAI/bge-small-zh-v1.5`，明确采用 float32、CLS pooling、512 维向量及 `auto-truncate=false`。TEI 不映射宿主机端口，只允许 Compose 私网内的 API、worker 和显式 smoke 服务访问；本地 TEI 无需 Key，`EMBEDDING_API_KEY` 保持为空且绝不继承 `OPENAI_API_KEY`。模型文件首次启动时从 Hugging Face 下载到具名卷 `tei_model_cache`，后续复用缓存；首次运行需要网络，实际磁盘、内存和 CPU 延迟取决于机器，不能据此宣称生产吞吐。

`tei-smoke` 会真实检查 `/health`、`/info` 返回的模型 SHA、512 维有限且 L2 归一化的向量、batch/single 近似一致、固定中文相关性排序，以及超出模型输入上限时确实拒绝而非静默截断。[TEI 项目](https://github.com/huggingface/text-embeddings-inference)采用 Apache-2.0，固定 [BGE 模型卡](https://huggingface.co/BAAI/bge-small-zh-v1.5)标注 MIT；实际部署仍需自行复核两者许可。该固定运行时已经由检索评测和可选 Evidence Reviewer 消费，但相关结果只代表小型冻结集，不代表生产吞吐或开放文本质量。

### 本地开发

```bash
python -m venv .venv
.venv/Scripts/activate  # Windows
pip install -r requirements.txt
set DATABASE_URL=sqlite:///./loreguard.db
uvicorn app.main:app --reload
```

默认不需要模型 API，确定性基线可以独立完成全流程。配置 OpenAI-compatible Provider 后，分析流水线会在基线抽取之上调用模型，使用 Pydantic 校验结构化结果，并按文档行号绑定证据：

```bash
set OPENAI_API_KEY=...
set OPENAI_BASE_URL=https://api.openai.com/v1
set OPENAI_MODEL=gpt-4o-mini
# 仅当上游明确支持 thinking 扩展时设置 disabled 或 enabled；默认省略
set PROVIDER_THINKING_MODE=disabled
set ENABLE_MODEL_EXTRACTION=true
set PER_RUN_TOKEN_BUDGET=20000
set DAILY_TOKEN_BUDGET=100000
```

模型 Key 只从服务端环境变量或本地未提交的 `.env` 读取。只有显式启用 `ENABLE_MODEL_EXTRACTION=true` 或 `ENABLE_ISSUE_EVIDENCE_REVIEW=true` 的能力才会调用聊天模型，避免开发和测试意外产生费用。请勿把 Key 写入源码、前端、README 或提交记录。`PROVIDER_THINKING_MODE` 默认不配置，因此通用 OpenAI-compatible 请求不会携带 `thinking`；只有显式设置为 `disabled` 或 `enabled` 时才发送顶层 `thinking={"type": ...}`。Compose 会把该配置同时传入 API 与 worker，空白值统一归一为 `None`。模型抽取支持 `fact`、`event`、`knows`、`claims_knows`、`item`、`uses`、`world_rule` 与 `world_assert`。超时、429、5xx、空响应、非法 JSON、字段校验失败或证据行号越界时，系统会记录非敏感警告并降级到 `BaselineExtractor`；基线与模型结果会去重合并。默认单次请求最多尝试 2 次、每次 30 秒；某分块终态失败后停止当前文档剩余模型分块，一个文档出现终态失败后开启本次运行熔断，后续文档直接走全文基线，避免兼容服务异常时串行等待数分钟。运行事件和诊断会明确区分“完整模型增强”“模型增强（部分分块已降级）”“确定性基线（模型未参与或已降级）”与主动关闭模型的“确定性基线”，结果正确时也不会掩盖模型失败。

模型逐分块调用前会用保守 Token 估算与本次已用额度执行门控；兼容 Provider 不返回 usage 时也按保守估算扣减内部预算。超出单次额度后停止后续模型分块，但全文基线仍会继续。每日额度在创建或重试分析时检查，当前是本地单数据库的非原子配额；多实例生产环境仍需 Redis 或事务式配额服务。写接口另有单进程滑动窗口限流，429 响应包含 `Retry-After`。

模型模式以持久化的结构化执行记录判定，不依赖警告文字。页面会显示逻辑调用成功、失败、跳过和拒绝记录数；任何分块被跳过或记录被拒绝都不能算完整成功。旧运行缺少这些计数时显示“覆盖情况未知”。`python scripts/check_provider.py` 可经生产调用路径测试当前配置，只输出安全状态、耗时与 Token，不输出密钥或上游响应正文。

首页“模型连接”卡会被动读取服务端是否已配置，但不会自动请求模型。只有用户点击“测试模型连接”时才会发起一次可能消耗少量 Token 的最小 JSON 检查；页面仅显示配置状态、JSON 合同、错误分类、延迟、Token 和固定建议，不显示 endpoint、Key、Prompt 或模型原始响应。浏览器不提供 Key 输入框，也不使用浏览器存储保存凭据：请只在本地未提交的 `.env` 中配置 `ENABLE_MODEL_EXTRACTION`、`ENABLE_ISSUE_EVIDENCE_REVIEW`、`OPENAI_BASE_URL`、`OPENAI_API_KEY` 与 `OPENAI_MODEL`，修改后同时重启 API 和 Celery worker。

受限 Agent 第一阶段使用 LangGraph 1.2.11 `StateGraph` 编排，仅在显式设置 `ENABLE_REVIEW_AGENT=true` 时启用，默认关闭。模型每轮返回应用层 JSON 动作 `READ_SPAN`、`PATCH_RECORDS` 或 `ABSTAIN`；这不是、也未冒充 Provider 原生 `tool_calls` / function calling。服务端绑定冻结文档和证据范围、校验 span 与补丁，并只持久化不含 Prompt、原始响应、证据正文、Key 或 endpoint 的安全 trace。固定的 modality/source_scope/certainty 标签 repair pass 是另一条非 Agent 路径，不能混称为 Agent。

冻结 Agent 验收集有 30 个开发者可见任务，每个重复 3 次。Mock oracle/scorer 的 90 次结果只验证工程边界；commit `bcbfab8` 的 v2 `full × 3` 只在 Agent 阶段调用真实 Provider，主抽取候选由 runner 按冻结 manifest 合成注入，并非端到端真实模型抽取评测。该运行记录全部 90 次要求执行，严格正确 59/90，其中 holdout 81 次运行成功 74 次、7 次 `read_timeout`，恢复 26/51、主动弃答 24/30。共有 12 次补丁通过生产校验但不符合评测 oracle；服务端接受的安全违规为 0。运行成功轨迹中没有直接 `ABSTAIN` 路径，要求的三路径覆盖 gate 也失败，完整结果为 `passed=false`，所以不能声称 Agent、主抽取或产品质量收益。当前没有多智能体。Evidence Reviewer 是另一条已经接通真实 RAG 的受限复核消费者，不等同于这个修复 Agent。详细边界见 [`docs/review-agent-phase1.md`](docs/review-agent-phase1.md)，脱敏结果见 [`docs/review-agent-v2-full-checkpoint-20260906.md`](docs/review-agent-v2-full-checkpoint-20260906.md)，冻结任务见 [`data/agent-acceptance-v2/manifest.json`](data/agent-acceptance-v2/manifest.json)，离线 runner 见 [`scripts/run_agent_acceptance.py`](scripts/run_agent_acceptance.py)。

另一条默认关闭的 Evidence Investigator 使用 Provider 原生 function tool calls，在冻结快照内执行 `SEARCH_EVIDENCE`、`READ_SPAN`、`SUBMIT_VERDICT` 或 `ABSTAIN`，候选仍须经过确定性 promotion。commit `618ca991` 上两次 DEV 均为 8/8；首次 10 例冻结 holdout 为 8/10（TP 4、FN 1、TN 4、FP 1，precision/recall 均 80%，10/10 正常终止）。Agent promotion 没有接受错误候选，但系统确定性主链路仍产生 1 个误报。该小样本已封存且不再用于调参，不能外推为开放文本、生产质量或多智能体成绩；详见 [`docs/evidence-investigator-live-evaluation.md`](docs/evidence-investigator-live-evaluation.md)。

Evidence RAG 已实现独立显式配置的 OpenAI-compatible embedding client、中文行号感知分块、版本化 embedding profile、精确 snapshot schema、Alembic、真实 BGE embedding、PostgreSQL/pgvector 精确余弦检索与 keyword+dense RRF。首次冻结 holdout 的混合 Recall@5 为 81.82%、All-evidence@5 为 71.43%、低词面 Recall@5 为 79.31%；最后一项差 1 条证据未过预设门槛，因此整体 gate 为 `false`，也不声称混合检索优于两种基线。`IssueEvidenceReviewer` 默认关闭；只有显式设置 `ENABLE_ISSUE_EVIDENCE_REVIEW=true` 并同时配置模型、embedding 与 PostgreSQL 才会执行。它只写入 `ai_evidence_review` 注释，不修改规则 issue。真实 A/B 结果见 [`docs/issue-review-v1-result-20260907.md`](docs/issue-review-v1-result-20260907.md)。

只有同时配置 `MODEL_INPUT_PRICE_PER_MILLION` 和 `MODEL_OUTPUT_PRICE_PER_MILLION` 时才计算估算成本；未配置时 API/UI 显示“未配置”，不会把未知价格误报为 0 美元。

模型分块默认每块最多 6000 字符、重叠 1 行、每文档最多 24 块，可通过 `MODEL_CHUNK_MAX_CHARS`、`MODEL_CHUNK_OVERLAP_LINES`、`MODEL_MAX_CHUNKS_PER_DOCUMENT` 调整。超过块上限时只截断模型增强部分并明确返回 warning；全文确定性基线仍处理完整文档。模型必须返回原文全局行号，服务端再从完整文档回填证据。

## 本地项目工作流

1. 在“本地项目工作台”新建或选择项目。
2. 一次选择多个 `.md`、`.txt`、`.json`、`.docx` 文件上传。同一项目内再次上传同名文件会自动生成下一版本，并把旧的同名活动版本标记为历史版本；明确选择“替换”时，上传文件名必须与目标一致，否则返回 `409`。
3. 点击“分析当前项目”。所有启动分析的按钮都会提示：仅当服务端显式启用模型时才可能消耗 Token。
4. 运行中可以请求取消；只有 `failed` 或 `cancelled` 任务允许重试，终态任务不能取消。刷新页面后重新选择项目，可从运行历史恢复状态、错误或已完成报告。
5. 问题可标记为接受、误报或已解决并附备注；相同标签与备注不会重复写入，历史反馈仍可由 API 审计。

SSE 客户端可用 `Last-Event-ID` 请求头或 `last_event_id` 查询参数从指定事件之后恢复。终态事件固定返回 `status` 和 `error`；失败任务不会被当成成功结果加载。

## 输入格式

文本 JSON 接口和 multipart 上传接口都接受显式 `document_role` 与
`story_scope`。`document_role` 只能是 `canon`、`character_profile`、
`chapter` 或 `reference`；`story_scope` 是 1–80 位的中文、英文、数字、
下划线或连字符标识。创建同名新版本时，省略某一字段会继承被替换版本
（未指定替换目标时继承该文件最新版本）的对应值；全新文档默认
`chapter/global`。系统不从文件名猜测角色或分支。只上传一份纯正文、
不提供独立世界观，也可以直接启动分析。

`.docx` 只通过 multipart 文件上传接口导入；文本 JSON 接口不接收二进制 Word 内容。导入器只读取标准 Office Open XML 包的主文档正文，将段落、显式换行和表格行转换为稳定的纯文本行，后续证据行号均指向这份导入后的文本。系统限制 ZIP 成员数、单成员/总展开量、XML 大小、压缩比和最终文本大小，拒绝加密、启用宏、损坏、路径异常或结构含糊的包；不会跟随外部关系，也不会导入或执行宏、嵌入对象、图片、批注、页眉和页脚。旧版二进制 `.doc` 不受支持，需先在 Word 或兼容软件中另存为 `.docx`。

网页默认接受保守的自然中文基线，例如：

```text
林澈的发色是银色。
1026-04-03 10:00，林澈在北港。
1026-04-03 12:00，林澈得知星门口令。
星门只能由潮汐晶核驱动。
```

高级用户仍可使用完全确定的行级标注：

```text
@fact subject="林澈" predicate="眼睛颜色" value="金色" | 角色设定：林澈拥有金色瞳孔。
@event id="e1" time="1026-08-03T10:00" location="潮汐港" participants="林澈" | 林澈抵达潮汐港。
@knows character="林澈" fact="星门密码" time="1026-08-03T12:00" | 午后他获知星门密码。
@claims_knows character="林澈" fact="星门密码" time="1026-08-03T09:00" | 清晨他已说出密码。
@item item="潮汐钥匙" owner="苏弦" time="1026-08-03T08:00" | 钥匙由苏弦保管。
@uses item="潮汐钥匙" user="林澈" time="1026-08-03T09:30" | 林澈使用钥匙开门。
@world_rule key="星门能源" value="潮汐晶核" | 星门只能由潮汐晶核驱动。
@world_assert key="星门能源" value="普通火焰" | 本章中星门由火焰直接启动。
```

## 架构

```text
React/Vite ──HTTP/SSE── FastAPI
                           │
                    Analysis Pipeline
                   ┌────────┴────────┐
            NarrativeExtractor  ConsistencyChecker
            ┌──────┴──────┐           │
    Baseline Parser  Model Provider  Rule Checker
            └────去重合并──┘
                   │
              PostgreSQL / SQLite
                   │
               Redis/Celery
```

生产环境通过 `USE_CELERY=true` 将任务交给 Celery；本地演示默认使用后台线程。`NarrativeExtractor` 与 `ConsistencyChecker` 是可替换接口。确定性规则主路仍使用本地可复现的候选排序；真实 embedding/pgvector/RRF 作为独立的可选 Evidence RAG 通路，由后置 `IssueEvidenceReviewer` 消费。复核结果只作为规则 issue 的 AI 注释持久化，不反向修改规则结论。

## 与 ConStory-Bench 的关系

当前版本没有复制 ConStory-Bench 的代码、Prompt、架构或数据。规则回归与合成自然中文评测由 LoreGuard 的固定模板生成，复杂多文档验收集则是独立编写的原创“潮痕群岛”场景。ConStory-Bench 是模型长篇故事一致性的研究基准；LoreGuard 是面向编剧工作流的交互式审查工具。详见 [`docs/constory-bench-boundary.md`](docs/constory-bench-boundary.md)。

## 评测

```bash
python scripts/run_evaluation.py
python scripts/run_natural_evaluation.py
python scripts/run_complex_v3_evaluation.py --require-perfect
python scripts/run_complex_model_evaluation.py --suite full --repeats 3 --max-total-tokens 350000
python scripts/run_model_stability.py --case advanced --repeats 3 --max-total-tokens 15000
python scripts/run_model_stability.py --case long-smoke-2k --repeats 5 --max-total-tokens 100000
python scripts/run_agent_acceptance.py --mock-oracle --require-gates
```

两套报告含义不同：

- `artifacts/directive-regression-report.json`：80 条显式 `@directive` 样本，只验证规则引擎接线，100% 结果不能解释为自然文本准确率。
- `artifacts/natural-evaluation-report.json`：固定 seed 生成的合成自然中文 test 集，共 60 例，其中 30 例无预期问题。状态建模前为 Precision 0.625、Recall 1.000、F1 0.769；加入角色/路线许可、显式知识来源和 actor 规则例外后，当前固定回归为 1.000/1.000/1.000。该 test 已被开发者查看，after 只能作为回归参考，不能称为无偏提升。
- `artifacts/state-modeling-v2/`：保存 natural dev、原 test 和 50 例 challenge-v2 的 before/after 完整报告及误差。challenge-v2 after 为 Precision 0.962、Recall 1.000、F1 0.980，仍保留 1 个移动许可措辞误报；它同样是开发者可见合成数据，不是盲测或人工标注。
- `artifacts/complex-v3-evaluation.json`：14 例原创、多文档复杂验收场景，共 10 个固定预期问题；五类问题均有正例与困难反例。当前无模型固定基线 TP 10、FP 0、FN 0，证据对精确命中率 1.0。该数据由开发者编写且可见，只能作为可审计回归，不能估计开放故事或生产准确率。
- `data/agent-acceptance-v1/`：30 个开发者可见冻结任务，3 类 persona 各 10 个，每任务重复 3 次。`--mock-oracle` 生成的 90 次 execution 仅用于验证离线 scorer、三类动态路径和安全约束；runner 当前不会调用生产 Agent，这些数字不得写成真实 Agent 评测或能力成绩。
- `data/agent-acceptance-v2/`：修正 pilot 暴露的语义标签问题后冻结的 Agent 评测套件。Agent 阶段真实 Provider `full × 3` 已完成但 `passed=false`；主抽取候选为冻结合成注入，仓库只公开[脱敏聚合结论](docs/review-agent-v2-full-checkpoint-20260906.md)，不提交 Prompt、响应正文、凭据或故事运行产物。
- `data/evidence-retrieval-v1/`：44 个原创中文检索问题，其中首次冻结 holdout 为 28 问、55 条期望证据。混合 Recall@5 81.82%、All-evidence@5 71.43%、低词面 Recall@5 79.31%；低词面少 1 条未过门槛，完整结论见 [`docs/evidence-retrieval-v1-holdout.md`](docs/evidence-retrieval-v1-holdout.md)。
- `data/issue-review-v1/`：12 例原创、开发者可见的证据复核 A/B，每例真实调用 3 次。local-context 为 4/12，rag-evidence 为 7/12；绝对 gate 仍为 `false`。只公开脱敏聚合结果，见 [`docs/issue-review-v1-result-20260907.md`](docs/issue-review-v1-result-20260907.md)。

完整数据位于 `data/evaluation-natural/`：40 个 dev 场景和 60 个 test 场景的 `scenario_id` 不重叠。它是模板生成的 `synthetic natural-language` 数据，不是人工标注集，也不能外推为生产准确率。评测 harness 运行 test 时只打开 `test.jsonl`，测试样本不进入 Prompt 或调参示例。

challenge-v2 位于 `data/evaluation-challenge-v2/`，schema、固定 seed、生成器和 SHA-256 均落盘可审计。详细口径见 [`docs/state-modeling-v2-evaluation.md`](docs/state-modeling-v2-evaluation.md)。

复杂验收集位于 `data/evaluation-complex-v3/`，其原创声明、固定证据行、数据哈希与限制见 [`docs/complex-v3-evaluation.md`](docs/complex-v3-evaluation.md)。旧中转/模型的真实重复运行记录见 [`docs/complex-v3-model-evaluation.md`](docs/complex-v3-model-evaluation.md)；2026-09-06 当前新中转/模型的冻结 Phase 1 结果见 [`docs/provider-phase1-checkpoint-20260906.md`](docs/provider-phase1-checkpoint-20260906.md)。简历采用哪些事实则受 [`docs/resume-readiness.md`](docs/resume-readiness.md) 的保守门槛约束。

`run_model_stability.py` 会真实调用已配置的 Provider，记录逐次类别/证据、首进度、P50/P95、Token 与预算停止状态；预期答案只用于运行后评分，不进入 Prompt。报告不保存 Key 或原始响应正文。总预算是运行间停止阈值，单次 Provider 实际 usage 可能让最后一次发生少量越界，报告会单独记录 `budget_overshoot_tokens`。

历史高级开发场景曾在约束不足时得到 P 0.556 / R 1.000；失败记录没有删除。通用结构化与证据修复后，complex-v3 完整 14 例 × 3 轮为 42/42 完整无降级、171/171 次调用成功，类别和严格完整证据评分均为 TP 30、FP 0、FN 0，P95 84.97 秒。它是开发者可见固定回归，不是人工盲测或生产准确率；详情见 [`docs/model-stability-evaluation.md`](docs/model-stability-evaluation.md)。

上述历史模型轮次使用 2026-09-02 的旧覆盖判定。2026-09-04 起完整模型统计必须有结构化执行计数，并排除部分无效记录；旧报告缺少计数时仅保留带“历史未验证”标记的原统计，不能据此声称通过了新协议，也不能移用为新 Provider/模型的成绩。

2026-09-06 新中转/模型在 `thinking=disabled` 下通过最小 JSON preflight，但冻结 Phase 1 `full × 1` 严格 gate 为 0/3。三次正式调用均 HTTP 200 且无空响应，仍因内容校验、未解决无效记录和 batch 协议失败而不具备完整覆盖；HTTP 成功不能解释为产品可用。首次运行曾因旧计数包装器缺少 repair 依赖转发而中断，额外调用不可计量；随后 Phase 1/complex runner 已共用修复后的安全计数包装，派生 repair 调用也纳入统计并通过 Mock 回归。默认报告保存在 Git 忽略的 `artifacts/`，不含 key、endpoint、Prompt 或原始响应。旧模型/中转的历史成绩不得沿用到本轮。

历史固定 2000 字单文档真实模型延迟测试 5 次的首进度 P95 为 5.9 ms，端到端 P50 / P95 为 6.30 / 7.94 秒，总计 11,799 Token。该样本没有准确率标注，使用的是旧中转/模型，只保留为当时的短文本延迟记录。

## 测试

```bash
python -m unittest discover -s tests -v
```

API 测试中的一键自然文本用例会清空模型 Key 并完成“创建项目—分析—读取问题/记录”的 no-model smoke；前端用 `cd frontend && npm run build` 验证 TypeScript 与生产构建。

另有一套 24,418 字、四文档、多章节的原创生成型长文 smoke：

```bash
python scripts/generate_long_text_smoke.py
python scripts/run_long_text_smoke.py
```

当前无模型报告记录为 7 个分块、10 条状态记录、5 类各 1 个问题、无额外问题，本机最近一次约 137 ms；报告位于 `artifacts/long-text-smoke-report.json`。这是生成型管线验收，不是人工长篇准确率或生产性能承诺。

## 公开接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/v1/projects` | 创建项目 |
| GET | `/api/v1/projects` | 最近优先列出项目、活动文档数和最近运行摘要 |
| POST | `/api/v1/demo` | 创建原创自然文本演示项目 |
| GET | `/api/v1/projects/{id}` | 获取项目与当前文档 |
| GET | `/api/v1/projects/{id}/documents?include_history=true` | 获取当前与历史文档版本 |
| GET | `/api/v1/projects/{id}/documents/diff?from_document_id=...&to_document_id=...` | 比较同名文档的两个版本（本地行级 diff） |
| POST | `/api/v1/projects/{id}/documents` | 上传或更新文档 |
| POST | `/api/v1/projects/{id}/documents/text` | 粘贴自然文本 |
| POST | `/api/v1/projects/{id}/analysis-runs` | 启动分析 |
| GET | `/api/v1/projects/{id}/analysis-runs` | 获取项目运行历史 |
| GET | `/api/v1/analysis-runs/{id}` | 查询状态与成本 |
| GET | `/api/v1/analysis-runs/{id}/events` | SSE 进度流 |
| GET | `/api/v1/analysis-runs/{id}/issues` | 获取问题与证据 |
| GET | `/api/v1/analysis-runs/{id}/clarifications` | 获取与确认问题隔离的待澄清/开放问题（仅完成态） |
| GET | `/api/v1/analysis-runs/{id}/records` | 查看实际抽取记录与提示 |
| GET | `/api/v1/analysis-runs/{id}/diagnostics` | 获取分块、别名和检索候选诊断 |
| GET | `/api/v1/analysis-runs/{id}/graph` | 从已保存记录投影证据化关系图（仅完成态） |
| GET | `/api/v1/analysis-runs/{id}/timeline` | 获取精确时间分组及未确定时间记录（仅完成态） |
| POST | `/api/v1/issues/{id}/feedback` | 提交反馈 |
| GET | `/api/v1/issues/{id}/feedback` | 获取最新反馈与审计历史 |
| POST | `/api/v1/analysis-runs/{id}/cancel` | 取消任务 |
| POST | `/api/v1/analysis-runs/{id}/retry` | 重试任务 |
| GET | `/api/v1/evaluations/latest` | 获取内置显式指令规则回归（非自然文本准确率） |

## 项目边界

- MVP 侧重“发现并解释矛盾”，不自动覆写作者原文。
- 大模型输出不是事实来源；所有问题必须绑定输入文档中的证据。
- 在线 Demo 应启用访客限额、文件大小限制和每日 Token 预算。
- 当前已验证模型抽取协议、降级机制、工程安全边界和一次冻结真实 Phase 1 运行；但当前新中转/模型的严格 gate 为 0/3，且尚未完成真实长篇人工标注评测，不能宣称开放文本准确率或当前模型能力达标。
- `document_context` 与运行输入上下文采用附加关联表兼容旧 SQLite；旧文档读取为安全默认 `chapter/global` 并标记上下文并非显式保存。数据库升级现由 Alembic 管理，覆盖空库、旧 `create_all` 库和此前 WIP schema 的保守升级；残缺或约束不完整的同名表会拒绝采用，不再以 `create_all` 充当正式升级路径。
- 版本差异只在当前项目和同名文档边界内运行，不调用模型。默认每个版本最多处理前 20,000 行/2,000,000 字符并最多返回 4,000 行差异，超限会在响应和页面显式提示。
- LangGraph 1.2.11 `StateGraph` 的受限修复循环已作为默认关闭的第一阶段代码接入；v2 Agent 阶段真实 Provider full 已执行但完整 gate 失败，且主抽取候选为冻结合成注入，不能列作已证明收益或端到端抽取成绩。它不是原生 `tool_calls`，也没有多智能体。真实 embedding、pgvector、RRF 和后置 Evidence Reviewer 已完成工程闭环与冻结 A/B；两套 gate 的失败项必须同时公开。项目已达到当前简历工程展示的停止线，后续优先进入本人体验、学习、讲解和投递，不再以扩功能推迟投递。
- 详细完成度见 [`docs/completion-status.md`](docs/completion-status.md)，学习顺序见 [`docs/learning-guide.md`](docs/learning-guide.md)。
- 长篇容量的当前硬限制、实用范围和百万字扩展路线见 [`docs/scalability-roadmap.md`](docs/scalability-roadmap.md)。

## License

Apache-2.0。原创样例世界观可在本仓库演示与评测范围内使用。
