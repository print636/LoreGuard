# Guided Review Batch 与资料上下文建议 V1

> 阶段范围：当前后端合同及其对应的网页引导流程。本文只描述已有实现与测试覆盖的行为，不代表开放文本准确率、长篇生产性能或自动写作能力。

LoreGuard 当前推荐的审查路径不是把项目内所有文件无差别送入一次分析，而是先建立可信背景，再审查仍可修改的新稿：

```text
导入资料
  ↓
AI 建议资料身份（可选）──→ 人工核对并确认资料上下文
  ↓
baseline_build（只冻结正式背景）
  ↓
人工确认 / 驳回角色资料候选
  ↓
draft_review（新稿作为 target，正式资料作为 background）
  ↓
查看只与新稿有关的问题及证据
```

## 一、先确认每份资料的身份

资料上下文至少包括：

- `document_role`：`canon`（世界观/正式设定）、`character_profile`（角色资料）、`chapter`（章节/剧情文本）或 `reference`（参考资料）。
- `publication_status`：`draft`、`in_review`、`published`、`retired` 或 `unknown`。
- `scope`：主时间线、版本、分支和活动范围。范围缺失时可以保留未知，不应凭空补全。
- `resolution_state`：只有 `confirmed` 才表示用户确认；`inferred` 只是模型建议。

网页中的“AI 自动识别”是一个显式、单文档操作。它读取当前活动版本的正文，要求模型返回简体中文理由和可回指原文的证据，再把建议保存为新的上下文修订。模型建议始终满足以下边界：

- 固定写为 `origin=model_inferred`、`resolution_state=inferred`、`authority_tier=unresolved`，不能自动确认资料或提升权威等级。
- 证据必须逐字对应当前文档的行号，且资料角色至少需要一条明确支持 `document_role` 的原文证据；缺少该证据时整次建议失败且不落库。其他字段缺少证据时，发布状态降为 `unknown`，版本、分支和活动范围不得臆测。
- 模型调用期间不持有数据库写事务；返回后会再次检查文档版本、正文哈希和上下文修订号，发生并发修改时返回 `409`，不会覆盖新数据。
- 模型未配置、超时、限流或结构化输出无效时不会写入建议，也不会向客户端返回 Key、Prompt 或模型原始响应。
- 单次自动识别最多处理 30,000 个字符、2,000 行。超过任一上限时请手动设置资料上下文；当前不会自动拆分超长文档做身份推断。

用户仍需在界面中核对建议，必要时修改角色、发布状态和范围，再明确保存为 `confirmed`。角色修正与上下文确认在同一事务完成，旧的确认请求无需携带 `document_role`，保持兼容。

## 二、建立正式基线

调用 `baseline_build` 时，服务端只选择当前项目中处于活动状态且已确认的正式资料：

- `canon` 与 `character_profile`，且发布状态为 `published` 或 `unknown`。发布状态未标注的已确认正式资料仍允许进入基线；这一规则不只针对旧资料；
- `document_role=chapter` 且 `publication_status=published` 的历史章节。

`reference` 即使标为 `published` 也只是参考资料，不会冒充历史章节。草稿、审阅中、已退役、未确认或不属于上述正式资料/已发布历史的文件不会进入基线，并会在 `review_batch.excluded_documents` 中给出排除原因。草稿与审阅中资料沿用 `draft_excluded_from_baseline` 原因码。`baseline_build` 没有 target，所有入选文件都被冻结为 `background`。

如果服务端显式启用了角色一致性阶段（`ENABLE_CHARACTER_CONSISTENCY=true`），基线运行可能产生带证据的角色资料候选。候选不会自动变成正式角色基线；用户必须逐条确认或驳回。未启用该阶段时，`baseline_build` 仍可冻结并运行背景资料，但不会因此自动生成角色候选。

## 三、审查新稿

调用 `draft_review` 时，可以显式选择一份或多份新稿，也可以让服务端在当前项目中选择符合条件的文件。合法 target 必须同时满足：

- 是当前活动版本；
- `document_role=chapter`；
- 资料上下文已人工确认；
- `publication_status` 为 `draft` 或 `in_review`。

服务端根据自身保存的资料上下文自动加入背景，客户端不能把任意文件伪装成权威资料。背景与建立基线使用同一资格：已确认、未退役、发布状态为 `published` 或 `unknown` 的 `canon` / `character_profile`，以及已确认且 `published` 的历史 `chapter`；还须与 target 范围兼容。草稿/审阅中的非目标资料以 `draft_excluded_from_background` 排除，已发布的 `reference` 也不会变成背景。未确认、已退役或范围不兼容的资料同样会被排除，原因在状态响应中可见。

每个运行冻结以下信息：

- 批次 `mode`、`sensitivity`、target/background ID 和排除原因；
- 每份输入的文档 ID、版本、正文哈希、上下文快照以及 `batch_role`；
- 分析过程中使用的是冻结内容，不会悄悄切换到随后上传的新版本。

`GET /api/v1/analysis-runs/{id}` 的 `review_batch` 汇总批次语义，`input_documents[].batch_role` 展示每份冻结输入是 `target` 还是 `background`。草稿审查最终只保留至少有一条证据指向 target 的问题；纯 background-background 冲突不作为“新稿问题”展示，其抑制数量保留在运行诊断中。

网页引导页通过 `GET /api/v1/projects/{project_id}/character-baseline-status?baseline_run_id={run_id}` 获取项目级汇总，而不是只读取角色列表第一页。汇总返回角色总数、仍有效的作者已确认特征数、所选已完成基线的待确认候选数、来源运行与角色模型覆盖状态；计数由服务端聚合，不因角色超过一页而截断。省略 `baseline_run_id` 时返回最近完成的基线；显式指定时只能读取该项目已完成且仍被允许的基线，避免并发基线完成顺序交错导致当前资料对应的结果被误判过期。待确认候选不会进入正式角色档案，也不会仅因项目中另有候选尚未核对，就阻止使用已确认特征审查新稿。引导页仍要求当前冻结资料对应已完成且完整覆盖的基线、至少一条已确认特征和至少一份人工确认的新稿；未知、旧版本或部分覆盖不会被当作已就绪。

这些是**项目级**条件，不保证所选章节里的每个角色都有已确认特征。未覆盖角色不会获得角色 OOC 判断，其它一致性检查仍可运行；“没有角色问题”不可解释为所有角色都通过。网页会明确提示这一边界。选中的新稿 ID 暂存在当前浏览器标签页内，返回引导页时只恢复仍然有效且允许审查的草稿；不保存正文或角色设定。

## 四、重试与复检

- `retry` 用于失败或取消的运行，复制原运行的冻结输入和 target/background 角色，不重新选择资料。
- `recheck` 用于已完成运行。对于 `draft_review`，它只把原来的逻辑 target 替换成同名文档的当前活动版本，再根据当前已确认资料重新派生 background；它不会退回“分析全部活动文档”。
- 创建分析和用户重试仍支持 `Idempotency-Key`。同一个 Key 绑定规范化后的 `mode`、`sensitivity` 和目标集合；用同一 Key 改变审查意图会返回 `409`。

## 五、API 最小示例

AI 识别资料上下文（建议，不是确认）：

```http
POST /api/v1/projects/{project_id}/documents/{document_id}/narrative-context/inference
Content-Type: application/json

{"expected_revision": 0}
```

人工核对后确认资料身份：

```http
POST /api/v1/projects/{project_id}/documents/{document_id}/narrative-context/revisions
Content-Type: application/json

{
  "expected_revision": 1,
  "document_role": "chapter",
  "resolution_state": "confirmed",
  "publication_status": "draft",
  "scope": {"schema_version": 1, "timeline_key": "main"}
}
```

建立正式基线：

```http
POST /api/v1/projects/{project_id}/analysis-runs
Content-Type: application/json

{"mode": "baseline_build", "sensitivity": "balanced"}
```

审查选中的新稿：

```http
POST /api/v1/projects/{project_id}/analysis-runs
Content-Type: application/json

{
  "mode": "draft_review",
  "target_document_ids": ["document-id"],
  "sensitivity": "balanced"
}
```

旧客户端不传 body 或传 `{}` 时继续使用 `full_review`：当前所有活动文档均作为 target。这是兼容入口，不是推荐的权威资料工作流。

## 六、当前限制

- 资料上下文推断一次只处理一份不超过 30,000 字符、2,000 行的文档，不支持批量推断和自动分块确认。
- AI 只能建议资料身份，不能自动确认上下文、确认角色候选或改变权威等级。
- 角色资料候选依赖默认关闭的角色一致性功能开关和已配置模型；即使生成，也必须人工确认。
- 当前不会自动修改故事正文，也不会生成可直接替换原文的修订内容。
- `full_review` 为旧行为兼容模式，无法表达“可信背景对新稿”的产品语义。
- 已有自动化测试覆盖选择、隔离、冻结、重试/复检、推断结构和并发冲突；这些测试不等于开放文本准确率、真实长篇稳定性或生产高并发验证。

## 七、2026-09-22 验证记录

- 代码回归：后端 `1353 passed, 4 skipped`，另有 620 组参数化子测试通过；前端 113 项测试与生产构建通过。
- 真实资料上下文推断：世界设定、角色档案和待审章节三类样本均返回正确角色与发布状态，每份都有 3 条逐字证据；所有建议仍为 `inferred/unresolved`，未自动提升权威。
- 原创开发者可见角色一致性样例的三次独立全流程中，三次都生成并人工确认 5 个预期角色候选，三次都稳定显示预期的两个冲突且无额外可见问题。两次角色审查完整；一次因上游 `read_timeout` 与阶段预算边界降级为 `partial`，因此严格三次稳定性 gate 为 `passed=false`。这证明了失败时的安全降级与问题语义稳定性，但不能宣称三次稳定运行、开放文本准确率或生产质量。
