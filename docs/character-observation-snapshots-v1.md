# 角色行为记录与观察快照 v1

更新时间：2026-11-04。本文说明角色观察快照 v1 的使用与维护合同，受测实现为 `b9bb07e1c8f296b93e52c435f0d35736be57966c`。实际功能核验、CI 进度和本机服务版本分别记录在文末；功能链通过不代表整体模型质量验收通过。

## 报告中的角色行为记录在哪里

打开一份自己有权访问的运行报告，可以在“角色行为记录”区域查看本次已保存的角色行为与陈述。每条记录显示角色、行为或陈述、对应文稿的冻结版本及行号；展开“查看原文与设定关联”，可以核对原文片段、当时情境，以及这次运行实际采用的冻结角色设定。

记录来自专用角色处理流程中**已经通过现有校验、被接纳的新稿观察**。它保存当时识别出的内容，方便作者和维护者回看“系统看到了什么、与哪条设定进行了比较”。它不承诺记录文稿中的所有行为，也不把未通过校验的模型输出当成已接纳观察。

角色行为记录单独展示，不计入正式问题或待复核线索数量。普通行为、未关联设定的行为，以及角色称呼尚未解析的观察，都可以保留在这里；是否构成角色偏离，仍应查看报告中的实际审查结论。

界面中的“已关联冻结设定”只表示这条观察被选入与该设定的比较。它既不等于发现冲突，也不等于已经证明一致。未关联或未决也不能用来推断“没有问题”。

当前网页面板在已登录且报告所属项目、运行与当前页面相符时读取数据。它提供刷新、原文展开和分页，默认每页 50 条；读取失败会显示失败状态和重试入口，不会把请求失败当成零条记录。

## 先看记录是否完整，再看具体内容

`recording.status` 说明观察记录的保存情况。它与实际角色审查覆盖范围、材料覆盖范围和审查结论分别表达不同的事情。

| 状态 | 含义 | 阅读时如何理解 |
| --- | --- | --- |
| `in_progress` | 记录过程尚未结束 | 目前已保存的内容可能只是部分，可稍后刷新。 |
| `complete` | 本轮记录过程完成 | 不是“角色审查完整”或“没有异常”的证明；零条记录也不能作这种解释。 |
| `partial` | 本轮观察或设定关联明细未完整记录 | 结合覆盖计数和原因阅读，实际审查范围仍看报告中的审查说明。 |
| `failed` | 记录过程未能完成 | 可能已经保存部分观察，也可能没有可读条目，不能据此作一致性判断。 |
| `not_applicable` | 本轮不适用新稿观察记录 | 例如只归纳设定的 `baseline_build`，或角色处理功能未启用；不表示审查通过。 |
| `legacy_unavailable` | 本次运行没有可读取的专用观察记录头 | 无法据此还原当时的观察；不等于历史运行确实识别到零条。 |

没有记录头时，接口不根据运行时间猜测是否“等待记录”，也不把旧历史当成一个已经完成、结果为空的快照。`baseline_build` 可明确返回 `not_applicable`；其他没有记录头的情况返回 `legacy_unavailable`。

运行已经结束而记录头仍为 `in_progress` 时，接口采用只读解释：运行 `completed` 对应显示 `partial`，运行 `failed` 或 `cancelled` 对应显示 `failed`。响应同时保留 `stored_status: "in_progress"`，并给出固定的 `incomplete_reason: "run_ended_before_recording_completed"`。这不会修改数据库，也不会代替失去执行权的 worker 补写历史。

界面保留未完整记录的设定数量及相应原因数量。例如，达到设定比较数量上限、相同特征方向已合并采用、需要作者确认方向等，分别说明为什么没有展开全部关联明细。这些说明用于理解记录的保存范围，不会被转换成“实际审查一定遗漏同样数量”的判断。

## 保存了什么，旧快照为什么不会随正文更新

每条观察保留被接纳时的原始角色称呼、维度、特征键、对象、比较身份、陈述、极性、观察类型、稳定性、情境和证据。解析后的角色键单独保存。后续比较时对角色称呼或特征键所作的对齐，不会反向覆盖原始观察身份。

陈述可以是对行为的规范化表达；原文证据另行保存。证据必须是**本次运行冻结输入中、指定行范围内的非空精确片段**，允许引用行内的一部分，不要求把整行全部复制为引文。空引文、引自其他行或其他输入的片段，以及来源绑定不符的记录都会被拒绝。这里核对的是来源和完整性，不重新判定角色行为真假。

来源绑定包括本轮运行、项目、冻结输入标识、文档标识及名称、文档版本、内容 hash 和输入顺序。关联设定引用本轮 `AnalysisRunCharacterTraitInputRow` 冻结快照，同时保留候选标识、确认审核标识、候选锁版本和设定内容 hash。

之后上传新正文、修改当前设定，或撤销当前候选的确认，不会把旧观察改写成后来内容。读取旧关联时使用本轮冻结设定，不去读取当前候选的值，也不重新调用模型来“补全”旧记录。

## 只读接口

```http
GET /api/v1/analysis-runs/{run_id}/character-observations?offset=0&limit=50
```

`offset` 是从 0 开始的非负偏移；`limit` 默认 50，允许 1 至 200。接口按保存后的 `ordinal`、`observation_id` 稳定排序，返回 `total_items` 和 `has_more`，供调用方逐页读取。

接口复用运行及项目的既有工作区鉴权。启用账号认证时，其他账户无法读取不属于其工作区的运行；未登录请求也不能读取私有数据。项目或运行不可访问时按既有规则返回 404，未认证请求由鉴权层拒绝。显式配置为本机匿名模式时，访问规则与现有运行问题接口一致；这与网页面板当前要求登录后展示的条件分别处理。

成功响应带有 `Cache-Control: no-store`。保存条目或关联冻结设定的完整性校验失败时，接口返回固定的 409 错误，不返回一份被重新推断过的记录。系统没有开放创建或修改这类快照的客户端 API。

响应主要字段如下；字段名保留供接口维护者使用，网页不会把内部标识、hash 或原始原因代码直接当成作者提示。

| 位置 | 字段与用途 |
| --- | --- |
| 顶层 | `schema_version: "character-observations-v1"`、`run_id`、`project_id`；`match_semantics` 固定说明关联不代表一致性判决。 |
| `recording` | `status`、`stored_status`、`incomplete_reason`、`recorded_at`、`completed_at`、`observation_count`、来源与设定的 manifest hash，以及 `coverage`。 |
| `page` | `offset`、`limit`、`total_items`、`has_more`。 |
| `items[]` | `observation_id`、`ordinal`、`character`、`resolved_character_key`、`dimension`、`trait_key`、`key_object`、可为空的 `comparison_key`、`statement`、`polarity`、`stability`、`observation_kind`、`context`。 |
| `items[].source` | `run_input_id`、`document_id`、`document_name`、`document_version`、`content_sha256`、`input_ordinal`。 |
| `items[].evidence` | `document_id`、`document_name`、`line_start`、`line_end`、`text`。行号对应冻结输入。 |
| `items[].matches[]` | `status`、`reason_code`、`match_method`、`identity_match`，以及可为空的 `baseline`。 |
| `matches[].baseline` | 冻结设定的 `input_id`、`candidate_id`、`confirmation_review_id`、`candidate_lock_version`、`payload_sha256` 和读取时附带的冻结 `payload`。 |

有经过现有流程接纳的目标绑定证书时，观察还会保留相应的 `target_bound_target_digest`、`target_bound_target`、`target_bound_target_ordinal`、`target_bound_object_relation` 和 `target_bound_object_basis_id`。这些是业务证书，不是未经校验的模型响应。

`matches[].status` 只有三类：`matched` 表示进入比较，`not_matched` 表示实际路径未选入该比较，`unresolved` 表示尚不能确定或该关联未完成。`reason_code` 来自固定枚举；`match_method` 可记录特征身份、比较身份、目标绑定、偏好修饰或作者确认轴等实际匹配路径。可空的 `identity_match` 保留身份匹配器当时的结果，它与后续范围、发布位置或数量限制筛选后的最终 `status` 可以不同。

## 有界关联明细与遗漏计数

每条观察最多记录 256 条详细关联决策；实际受本轮既有候选数量限制约束，可能更少。系统不会为了追踪记录，将每条观察与最多 5,000 条冻结设定全部展开一遍。记录限制不改变原有判定门槛、匹配规则或问题提升条件。

没有展开为逐条明细的设定，通过记录头的固定结构保存：

```json
{
  "baseline_not_evaluated_count": 0,
  "excluded_baseline_counts": {},
  "omitted_baseline_count": 0
}
```

上述对象位于 `recording.coverage`。`baseline_not_evaluated_count` 是未展开明细的设定数；`excluded_baseline_counts` 按互斥的固定原因统计未展开设定；`omitted_baseline_count` 是前两部分数量之和。它们统计的是**设定条目**，不是观察数、冲突数或遗漏行为数。未提供覆盖元数据时，`coverage` 可以为空对象。

只要这些未展开数量非零，本轮记录完成时就标记为 `partial`。完整来源与冻结设定仍由记录头中的 manifest hash 绑定。网页将数量和允许展示的原因转成可读说明，并提示读者另行核对报告中的实际审查范围。

## 持久化、幂等与 worker 失败

新增持久化由两个表承担：`character_observation_recordings` 保存每轮记录头、状态、覆盖计数及 manifest；`character_observations` 保存每条已接纳观察、证据、稳定顺序、关联决策引用及内容 hash。

数据库中的关联只存冻结设定引用与内容 hash，不为每条观察重复存储整份设定 payload。GET 为当前页批量读取**同一运行、同一项目**的冻结设定，核对引用及 hash 后，才将 payload 附到响应里。

同一运行下，以稳定的 `observation_id` 幂等写入。完全相同的记录可以重复提交；相同标识但内容或关联决策不同会被拒绝。记录一旦以终态封存，不能追加新观察、改写旧观察或改换终态。这里的封存由仓库写入接口执行，不提供客户端绕过入口。

真实 worker 入口强制注入 recorder。已接纳的新稿观察在别名解析和比较期改写之前进入观察 journal；即使未关联、称呼未解析，或后续阶段异常，也会尝试保留这些观察与实际已经执行的匹配决策。未执行的关联明确标为未决，不根据已存业务数据重新判断。

写入沿用服务现有的短事务边界，并在提交前检查 worker token。阶段异常时会尝试封存为 `failed`；写入失败有固定失败诊断，不能宣称记录完成。取消或执行权丢失时，不允许绕过 worker fence 补写；若因此留下 `in_progress` 记录头，使用前述只读终态投影说明记录未完成。

业务内容保存在项目私有领域表，不写入日志或 telemetry；该存储路径也不保存未经校验的模型响应、请求头、密钥或任意异常正文。

## 升级旧数据库与历史兼容

迁移 `0025_character_observations` 接在 `0024_project_import_foundation` 后，增加上述两个表。它采用 SQLAlchemy 通用字段和外键定义，面向 SQLite 与 PostgreSQL；新增表通过运行、项目和冻结输入引用建立删除关系，清理父资源时仍遵循项目既有依赖删除顺序。

升级保留旧运行及其已有正文、设定和问题数据，但不会为旧运行凭空生成“已经校验过的观察”。如果尚无专用记录头，读取时应遵循 `legacy_unavailable` 或明确不适用状态。已有观察记录头时，降级受到数据保护检查；不能静默删除这些历史记录。

旧 `analysis_records` 属于通用抽取链路，不能替代专用角色观察。尤其不能凭旧通用记录倒推当时的角色特征键、对象、比较身份、校验接纳状态或冻结设定关联，再把推断结果标成历史已校验快照。需要新观察快照时，应按既有流程执行新的分析运行。

## 开发定位

维护时可从以下源码入口核对实现：

- `app/character_observations.py`：强类型输入、固定原因枚举、仓库写入、分页投影及完整性检查。
- `app/character_consistency_stage.py`：接纳观察的 journal、实际分支决策、有界明细与覆盖汇总。
- `app/service.py`：真实 worker recorder 注入、短事务及提交前执行权检查。
- `app/main.py`：只读 GET、工作区访问校验及固定完整性错误。
- `frontend/src/features/character-observations/`：作者面板、字段白名单、覆盖计数展示与分页。
- `migrations/versions/0025_character_observations.py`：新增表及降级保护。

## 2026-11-04 实际核验与发布状态

本轮在同一冻结实现上，用不同于医疗案的原公开 case1（周岑，两份文稿）完成一次真实 DEV 功能核验。原生结果为 `completed`，已评估 1／未评估 0，实际形成 `conflict` 并呈现为正式问题。新稿的观察记录为 `complete`，保存 1 条已校验观察和 1 条冻结设定关联；数据库记录与 API 整个返回 payload（包括补入的冻结设定）精确一致。

这次实际观察与基线均采用 `directness`／`speech_pattern:directness` 身份。新稿观察是第 3 行的 `negative`、`situational`、`speech_sample`，实际匹配路径为 `trait_identity`，`identity_match: true`。这是当时真实执行分支的留存及回放核验，没有离线重新归类来补造一次匹配。它证明本例“接纳观察—保存原始身份及冻结引用—鉴权读取”的功能链已经贯通，不提供整体模型质量、命中率或召回率结论。

同版本的医疗 case4 也只执行一次，但停在基线构建：角色阶段因 `scope_review_provider_error` 为 `partial`，没有进入新稿运行，已评估 0／未评估 1。该次只能验证基线构建的 `not_applicable` 空记录状态，不能证明医疗解释链执行过，也不能声称先前的语义匹配问题已经修复。现有安全诊断未保存更细的失败子类，无法据此判断 HTTP、传输或其他具体原因。旧轮次的失败和不确定性保持原记录。

| 核验项 | 本轮证据与边界 |
| --- | --- |
| 源码冻结 | 上述两次 DEV 均为 `b9bb07e1c8f296b93e52c435f0d35736be57966c`；69 个已提交源文件的来源汇总 hash 为 `d2b34bb1defd17a5b2cd7da1ab5cd702477ce86ba22cb47f993661b5ace85c63`，运行前后及已提交字节一致。 |
| 功能与来源核对 | case1 的 12 项功能核对全部通过；来源与执行 receipt 的 13 项身份核对通过。没有改动模型 Prompt、判定门槛或评分规则。 |
| 存储/API 本地验证 | 26 通过、1 个 PostgreSQL 实库用例因本地未提供环境而跳过；覆盖冻结历史、幂等、来源/所有权隔离、分页、失败状态及升级兼容。跳过不能计为本地 PostgreSQL 通过。 |
| 迁移/备份脚本本地回归 | 64 项通过、25 个子测试通过；子测试与重复执行不再累计成新增独立场景。 |
| Pipeline 定向验证 | 12 项专项、3 项 v3 定向回归及 7 项旧用例定选通过；测试 provider 的路径验证与上述真实 DEV 分别记录。 |
| 前端本地验证 | 初始 11 项单元加新增覆盖计数用例，共 12 项；另有 4 个 Mock API 浏览器场景。1440／375 视口截图早于最后文案及覆盖计数改动，只证明当时布局，不冒充最终字节的 UI 截图。 |
| 实现审查 | 一轮审查的一项 P2 已修复；没有第二轮审查结论。 |
| CI 与兼容修复 | 首轮 [41695225974](https://github.com/print636/LoreGuard/actions/runs/41695225974) 四项任务成功，整站检查为 115 通过、2 失败；两个旧运行历史用例在最后的 unexpected 请求断言发现未模拟新增只读接口。仅补严格测试路由，保留所有原业务断言，集中补跑这两个用例，2／2 通过（12.1 秒）。测试修复提交 `6a81ad7ad68c7f1a64c17235d564110322868dbf` 的正常 [CI 41697571874](https://github.com/print636/LoreGuard/actions/runs/41697571874) 中 `test`、`pg-axis-upgrade`、`backup-restore-rehearsal`、`compose-smoke`、`system-e2e` 五项全部实际成功。原失败记录保留；产品源码及模式与上述 DEV 冻结版本相同。 |
| 本机交付流程 | 交付入口为 `http://127.0.0.1:8787`；部署沿用原账户及资料并重建前端，启动时自动增补 `0025` 记录表，实际部署结果以本次启动验证收据为准。本轮自有 8106／8107 DEV 实例已停止，原 8317 服务保持原样。 |

PostgreSQL 实库用例遵循 `LOREGUARD_TEST_POSTGRES_URL` 约定，在隔离 schema 内验证旧库升级、快照回放、幂等及清理。两次代码 CI 中已成功的 `pg-axis-upgrade` 作业均包含新增的 `tests/test_character_observations.py -k postgres` 步骤；它们与本地实库用例跳过分别记录。此后仅文档的提交使用 `[skip ci]`，不算作新一次代码验证，也不重新调用模型。

本地证据名包括 case1 目录 `observation-functional-dev-20261104` 下的 `functional-acceptance.json`、`observation-api-db-evidence.json`、`receipt-crosscheck.json`，以及医疗 case4 目录 `observation-real-dev-20261104` 下的阶段边界和安全诊断工件。专项出处还包括 `observation-storage-final-junit.xml`、`observation-migration-junit.xml`、`pipeline.xml`、`pipeline-v3.xml`。这些是未随仓库发布的内部工件名；仓库读者可使用冻结标识与上方 CI 链接核对公开执行记录。

下一阶段的明确工程待办是补齐 scope 每次调用的安全失败分类，取得证据后再作故障归因。医疗语义匹配在本轮仍没有新稿观察证据；不能跳过这一诊断边界宣称医疗问题已解决。
