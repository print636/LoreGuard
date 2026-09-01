# LoreGuard

面向游戏编剧与叙事设计团队的剧情一致性审查平台。LoreGuard 将世界观、角色设定与章节文本抽取为可追溯的状态记录，通过确定性检查定位时间线、知识、物品、地点和世界规则冲突。

> 本项目只分析用户拥有权利的原创或授权文本，不连接、控制或自动化任何商业游戏。

## 已实现的 MVP

- 一键简单/复杂原创样例、自然文本粘贴及 Markdown、TXT、JSON 多文件上传
- 本地项目列表、文档版本历史和运行历史；刷新页面后可恢复最近任务与结果
- 无 API Key 抽取明确中文句式，并展示抽取记录与未抽取提示
- 配置 OpenAI-compatible 模型后，从普通中文故事中结构化抽取 8 类叙事记录；模型失败自动保留基线结果
- 模型输入按全局行号分块，支持行重叠、超长单行和逐块失败隔离；全文基线不分块
- 仅依据明确“又名/简称/化名/代号”声明做项目级实体别名归一化，保留映射轨迹与歧义警告
- 本地 keyword + 稳定 SHA-256 字符 n-gram + canonical entity graph 混合检索，候选短名单由检查器消费并留下分数轨迹
- 检查事实冲突、同时间多地点、知识越权、非持有者使用物品、世界规则冲突
- 每个问题返回两处证据、行号、严重度、置信度和修订建议
- 本地线程/Celery 两种后台执行方式、受状态约束的取消/重试与可断线续传的持久化 SSE 进度流
- 接受、误报、已解决反馈，含最新状态、重复提交抑制和完整审计历史
- 80 条显式指令规则回归，以及 100 条含困难负样本的合成自然中文评测（40 dev / 60 test）
- React 审查界面、Docker Compose、GitHub Actions 与 Prometheus 指标
- 已完成运行的关系图与保守时间线，支持类别筛选、问题关系筛选、证据详情和问题联动高亮；展示不会再次调用模型

## 快速开始

### Windows 一键体验

双击项目根目录的 `启动LoreGuard.cmd`。首次运行会创建 `.venv` 并安装依赖，随后浏览器打开 <http://127.0.0.1:8000>。进入页面后点击“一键运行原创样例”。

### Docker Compose

```bash
docker compose up --build
```

- Web：<http://localhost:8080>
- API：<http://localhost:8000/docs>
- Prometheus：<http://localhost:9090>

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
set ENABLE_MODEL_EXTRACTION=true
set PER_RUN_TOKEN_BUDGET=20000
set DAILY_TOKEN_BUDGET=100000
```

模型 Key 只从服务端环境变量或本地未提交的 `.env` 读取，并且只有显式设置 `ENABLE_MODEL_EXTRACTION=true` 才会调用模型，避免开发和测试意外产生费用。请勿把 Key 写入源码、前端、README 或提交记录。模型抽取支持 `fact`、`event`、`knows`、`claims_knows`、`item`、`uses`、`world_rule` 与 `world_assert`。超时、429、5xx、空响应、非法 JSON、字段校验失败或证据行号越界时，系统会记录非敏感警告并降级到 `BaselineExtractor`；基线与模型结果会去重合并。

模型逐分块调用前会用保守 Token 估算与本次已用额度执行门控；兼容 Provider 不返回 usage 时也按保守估算扣减内部预算。超出单次额度后停止后续模型分块，但全文基线仍会继续。每日额度在创建或重试分析时检查，当前是本地单数据库的非原子配额；多实例生产环境仍需 Redis 或事务式配额服务。写接口另有单进程滑动窗口限流，429 响应包含 `Retry-After`。

只有同时配置 `MODEL_INPUT_PRICE_PER_MILLION` 和 `MODEL_OUTPUT_PRICE_PER_MILLION` 时才计算估算成本；未配置时 API/UI 显示“未配置”，不会把未知价格误报为 0 美元。

模型分块默认每块最多 6000 字符、重叠 1 行、每文档最多 24 块，可通过 `MODEL_CHUNK_MAX_CHARS`、`MODEL_CHUNK_OVERLAP_LINES`、`MODEL_MAX_CHUNKS_PER_DOCUMENT` 调整。超过块上限时只截断模型增强部分并明确返回 warning；全文确定性基线仍处理完整文档。模型必须返回原文全局行号，服务端再从完整文档回填证据。

## 本地项目工作流

1. 在“本地项目工作台”新建或选择项目。
2. 一次选择多个 `.md`、`.txt`、`.json` 文件上传。同一项目内再次上传同名文件会自动生成下一版本，并把旧的同名活动版本标记为历史版本；明确选择“替换”时，上传文件名必须与目标一致，否则返回 `409`。
3. 点击“分析当前项目”。所有启动分析的按钮都会提示：仅当服务端显式启用模型时才可能消耗 Token。
4. 运行中可以请求取消；只有 `failed` 或 `cancelled` 任务允许重试，终态任务不能取消。刷新页面后重新选择项目，可从运行历史恢复状态、错误或已完成报告。
5. 问题可标记为接受、误报或已解决并附备注；相同标签与备注不会重复写入，历史反馈仍可由 API 审计。

SSE 客户端可用 `Last-Event-ID` 请求头或 `last_event_id` 查询参数从指定事件之后恢复。终态事件固定返回 `status` 和 `error`；失败任务不会被当成成功结果加载。

## 输入格式

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

生产环境通过 `USE_CELERY=true` 将任务交给 Celery；本地演示默认使用后台线程。`NarrativeExtractor` 与 `ConsistencyChecker` 是可替换接口。当前混合检索是本地可复现字符 n-gram 近似向量，不是真实 embedding；Compose 虽预置 pgvector 镜像，主链路尚未创建或使用向量列。

## 与 ConStory-Bench 的关系

当前版本没有复制 ConStory-Bench 的代码、Prompt、架构或数据。规则回归与自然中文评测内容均由 LoreGuard 的固定模板生成。ConStory-Bench 是模型长篇故事一致性的研究基准；LoreGuard 是面向编剧工作流的交互式审查工具。详见 [`docs/constory-bench-boundary.md`](docs/constory-bench-boundary.md)。

## 评测

```bash
python scripts/run_evaluation.py
python scripts/run_natural_evaluation.py
python scripts/run_model_stability.py --case advanced --repeats 3 --max-total-tokens 15000
```

两套报告含义不同：

- `artifacts/directive-regression-report.json`：80 条显式 `@directive` 样本，只验证规则引擎接线，100% 结果不能解释为自然文本准确率。
- `artifacts/natural-evaluation-report.json`：固定 seed 生成的合成自然中文 test 集，共 60 例，其中 30 例无预期问题。状态建模前为 Precision 0.625、Recall 1.000、F1 0.769；加入角色/路线许可、显式知识来源和 actor 规则例外后，当前固定回归为 1.000/1.000/1.000。该 test 已被开发者查看，after 只能作为回归参考，不能称为无偏提升。
- `artifacts/state-modeling-v2/`：保存 natural dev、原 test 和 50 例 challenge-v2 的 before/after 完整报告及误差。challenge-v2 after 为 Precision 0.962、Recall 1.000、F1 0.980，仍保留 1 个移动许可措辞误报；它同样是开发者可见合成数据，不是盲测或人工标注。

完整数据位于 `data/evaluation-natural/`：40 个 dev 场景和 60 个 test 场景的 `scenario_id` 不重叠。它是模板生成的 `synthetic natural-language` 数据，不是人工标注集，也不能外推为生产准确率。评测 harness 运行 test 时只打开 `test.jsonl`，测试样本不进入 Prompt 或调参示例。

challenge-v2 位于 `data/evaluation-challenge-v2/`，schema、固定 seed、生成器和 SHA-256 均落盘可审计。详细口径见 [`docs/state-modeling-v2-evaluation.md`](docs/state-modeling-v2-evaluation.md)。

`run_model_stability.py` 会真实调用已配置的 Provider，记录逐次类别/证据、首进度、P50/P95、Token 与预算停止状态；预期答案只用于运行后评分，不进入 Prompt。报告不保存 Key 或原始响应正文。总预算是运行间停止阈值，单次 Provider 实际 usage 可能让最后一次发生少量越界，报告会单独记录 `budget_overshoot_tokens`。

固定高级开发场景已经完成 3 次初测和 2 次证据约束后的真实调用。最新两次为 TP 10、FP 8、FN 0，Precision 0.556、Recall 1.000、F1 0.714，尚未达到投递门槛；详情与限制见 [`docs/model-stability-evaluation.md`](docs/model-stability-evaluation.md)。

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
- 当前只证明了模型抽取协议、降级机制和 Mock 异常测试；尚未完成真实长篇人工标注评测，不能宣称开放文本准确率。
- 版本差异只在当前项目和同名文档边界内运行，不调用模型。默认每个版本最多处理前 20,000 行/2,000,000 字符并最多返回 4,000 行差异，超限会在响应和页面显式提示。
- LangGraph、向量列与 OpenTelemetry 链路属于下一阶段，不列为本版完成事实。
- 详细完成度见 [`docs/completion-status.md`](docs/completion-status.md)，学习顺序见 [`docs/learning-guide.md`](docs/learning-guide.md)。
- 长篇容量的当前硬限制、实用范围和百万字扩展路线见 [`docs/scalability-roadmap.md`](docs/scalability-roadmap.md)。

## License

Apache-2.0。原创样例世界观可在本仓库演示与评测范围内使用。
