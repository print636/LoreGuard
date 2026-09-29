# 角色 OOC 人工封闭评测 V1

本文定义 LoreGuard 首套人工双人独立标注封闭测试的工作协议。它是评测基础设施，
不是已经完成的质量结果；在两位真人完成独立标注、仲裁、封存并运行真实模型之前，
不得声称“通过人工盲测”或给出开放文本准确率。

## 目标与本期边界

本期只验证角色 OOC 主链路及六类常见维度：核心性格、稳定偏好、说话方式、价值观或
行为边界、对特定人物或组织的关系态度、长期动机或目标。

样本同时覆盖常见的非冲突解释：已发布成长、当前有效的伪装或临时状态、单次反常和
材料不足。本期不做跨模型比较、超长文本压力、并发或生产部署验收，也不主动构造大量
文字游戏。后续只根据真实失败簇扩充 DEV/回归集。

## 人工标注原则

1. 每个 case 由两位不同真人独立标注；AI 不得充当其中任何一位标注者，也不得先生成
   holdout 预标注。
2. 两人提交并冻结前，不得看到对方答案、系统预测、私有 gold 或开发者预期。
3. 两人都要独立给出结果层级、六维度、`B/C/G/X/P` 证据、当前事件分组、解释到当前
   证据的绑定、原因码、置信度和说明。
4. 两份原始文件永不覆盖。比较只生成一致性摘要，不回显另一人的说明。
5. 分歧优先交由第三位真人仲裁；如果现实条件只有两人，则必须保留原始提交，并另行
   记录两人共识，不能回改初始答案。
6. 在查看模型预测前完成仲裁并生成私有 gold 的 HMAC-SHA256 commitment。
7. 某个 holdout case 一旦因具体模型错误被用于改代码、规则或 Prompt，就立即降级为
   DEV/回归样本；下一次封闭验收必须用未揭盲的新 case 补位。

标注文件必须声明：`reviewer_kind=human`、手册版本、未查看同伴结果、未查看系统预测和
独立完成。程序只能验证声明与文件绑定，无法替代真实流程纪律。

## 判定层级

结果采用固定三值：

| 冲突级别 | 结果 | 产品展示 | 含义 |
| --- | --- | --- | --- |
| `L0` | `no_issue` | `none` | 无反向行为，或所有表面偏离均被有效成长/伪装/状态解释 |
| `L1/L2` | `indeterminate` | `review_clue` | 主体、轴、情境或证据不充分，或只有一次独立反向行为 |
| `L3` | `conflict` | `formal_issue` | 稳定基线与至少两个独立当前事件同轴相反，且没有有效解释 |

材料覆盖不是 `complete` 时只能标为 `indeterminate`，不能把“没有发现”写成无问题。

证据角色：

- `B`：作者确认且版本有效的稳定基线；
- `C`：待审稿中属于目标角色、实际发生且与同轴同对象相关的行为或发言；
- `G`：在当前事件之前已经发生、能够解释变化的已发布成长；
- `X`：当前事件发生时有效的伪装、身份扮演或临时状态；
- `P`：伏笔、传闻、含糊暗示或有限推断，只能阻止过早晋升，不能证明已经解释。

正式问题必须包含至少一个 `B` 和两个属于不同事件组、互不重叠的 `C`。每个 `G/X`
必须明确绑定它能够解释的 `C`，不能用一处成长或伪装覆盖无关行为。

## 文件隔离

仓库只保存工具、协议和不含答案的公开输入。真人标注、仲裁、私有 gold、HMAC 密钥和
逐案失败不得提交 Git。推荐放在 `C:\Users\dell\Desktop\doc\LoreGuard-封闭评测`，并按
以下结构限制访问：

```text
public/
  public-input.json
  author-setup.json
  freeze.json
  run-config.json
  gold-commitment.json
  cases/<opaque-id>/*.md
offline-pages/
  reviewer-a-v1.html
  reviewer-b-v1.html
  adjudication-v1.html
private/
  reviewer-a.locked.json
  reviewer-b.locked.json
  agreement.json
  adjudication.locked.json
  private-gold.json
  hmac.key
predictions/
  repeat-01.sealed-run.json
reports/
  score-01.json
```

case、world、document、axis 和 evidence ID 必须为中性随机 ID，不能出现 `positive`、
`negative`、`conflict`、`clean`、`B`、`C` 等答案提示。公开 evidence catalogue 采用固定
算法为每个非空行生成一个单行锚点，不能由人工只挑“正确证据”，否则会泄漏答案位置。
`validate_exhaustive_line_catalog()` 会同时校验 UTF-8/LF、文档哈希、路径边界和全量行覆盖。
`freeze.json` 进一步绑定 `dataset_id`、公开输入哈希、作者配置哈希、case 数量和独立故事组
数量；生成 gold 和正式评分时三份公开信封缺一不可，任一内容变化都会使封存失效。

## 离线工具

每位标注者必须使用单独生成的自包含离线页面。两条命令使用相同的冻结公开包和手册
版本，但必须使用两个不同且不透露身份含义的 reviewer ID；页面不包含模型预测、私有
gold 或另一人的标注，输出已存在时拒绝覆盖：

```powershell
# 标注者 A 的离线页面
python scripts/generate_ooc_annotation_page.py `
  --public-input <public-input.json> --bundle-root <public-directory> `
  --author-setup <author-setup.json> --execution-freeze <freeze.json> `
  --reviewer-id <reviewer-a-id> --manual-version ooc-manual-v1 `
  --output <reviewer-a-v1.html>

# 标注者 B 的离线页面
python scripts/generate_ooc_annotation_page.py `
  --public-input <public-input.json> --bundle-root <public-directory> `
  --author-setup <author-setup.json> --execution-freeze <freeze.json> `
  --reviewer-id <reviewer-b-id> --manual-version ooc-manual-v1 `
  --output <reviewer-b-v1.html>
```

两位真人应分别只打开自己的页面，并在完成全部声明与案例后导出各自不可改写的
`*.locked.json`。两份原始提交完成校验并锁定后，才可生成裁决页。优先由未参与前两份
标注的第三位真人裁决：

```powershell
python scripts/generate_ooc_adjudication_page.py `
  --public-input <public-input.json> --bundle-root <public-directory> `
  --author-setup <author-setup.json> --execution-freeze <freeze.json> `
  --left-annotation <reviewer-a.locked.json> `
  --right-annotation <reviewer-b.locked.json> `
  --third-human-id <reviewer-c-id> --default-resolution-mode third_human `
  --output <adjudication-v1.html>
```

确实只有两位真人时，省略 `--third-human-id` 并使用默认的
`--default-resolution-mode reviewer_consensus`；仍须保留两份初始提交，由两人明确确认
共识，不能回改原文件。裁决页不读取系统预测或私有 gold，最终导出
`adjudication.locked.json`。

[`scripts/ooc_annotation_workflow.py`](../scripts/ooc_annotation_workflow.py) 提供：

```powershell
# 校验公开文档哈希、路径与无答案泄漏的全量行目录
python scripts/ooc_annotation_workflow.py verify-public `
  --public <public-input.json> --bundle-root <public-directory>

# 校验一位真人的完整独立提交
python scripts/ooc_annotation_workflow.py validate-annotation `
  --public <public-input.json> --author-setup <author-setup.json> `
  --execution-freeze <freeze.json> --bundle-root <public-directory> `
  --annotation <reviewer.json>

# 两份提交锁定后生成一致性报告
python scripts/ooc_annotation_workflow.py compare `
  --public <public-input.json> --author-setup <author-setup.json> `
  --execution-freeze <freeze.json> --bundle-root <public-directory> `
  --left <reviewer-a.json> `
  --right <reviewer-b.json> --output <agreement.json>

# 仲裁完成后生成私有 gold 和公开 commitment
python scripts/ooc_annotation_workflow.py build-gold `
  --public <public-input.json> --author-setup <author-setup.json> `
  --execution-freeze <freeze.json> --run-config <run-config.json> `
  --bundle-root <public-directory> `
  --left <reviewer-a.json> `
  --right <reviewer-b.json> --adjudication <adjudication.json> `
  --hmac-key-file <hmac.key> --gold-output <private-gold.json> `
  --commitment-output <gold-commitment.json>

# 独立评分进程先验证 commitment 与固定运行配置，再读取执行器原生报告
python scripts/ooc_annotation_workflow.py score `
  --public <public-input.json> --author-setup <author-setup.json> `
  --execution-freeze <freeze.json> --run-config <run-config.json> `
  --bundle-root <public-directory> `
  --gold <private-gold.json> `
  --commitment <gold-commitment.json> --hmac-key-file <hmac.key> `
  --sealed-run-report <repeat-01.sealed-run.json> --output <score-01.json>
```

所有输出默认拒绝覆盖。固定的 `run-config.json` 必须在 `build-gold` 前选定；commitment
同时绑定 gold、两份原始标注摘要、仲裁摘要、公开输入、作者配置、执行冻结、该运行配置
的规范化哈希、手册版本、完整评分门槛和评分代码清单。V1 预注册的是一次固定真实运行
策略：`minimum_repeats=1`、`min_outcome_consistency=null`；其余质量门槛均显式保持严格。
单轮无法计算跨轮一致性，因此不把未定义值伪装成失败，也不据此声称验证了运行稳定性。
正式 `score` 的唯一模型结果入口是 `--sealed-run-report`，且只
接受 [`scripts/run_ooc_sealed_http.py`](../scripts/run_ooc_sealed_http.py) 原生写出的完整
`SealedHttpRunReport`。评分器会先在尚未打开模型报告时验证 commitment 与传入的
`run-config.json`，随后要求报告携带的配置对象与该配置完全相等、配置哈希一致，并继续
核对 `execution_id`、执行代码清单哈希、运行时来源、项目回执、公开输入、作者配置和执行
冻结绑定；评分前还会重建并核对 workflow 与评分契约源码清单，最后才使用报告内部预测。
命令行不接受裸 `SealedPredictionArtifact`；把内层
artifact 手工复制、抽出或重新封装，都不是正式评分输入。模型执行进程不得接收 gold、
commitment 密钥或标注文件路径。

### 密封 HTTP 执行器

[`scripts/run_ooc_sealed_http.py`](../scripts/run_ooc_sealed_http.py) 是与评分进程隔离的
单向执行器。它只读取一个公开包目录中的 `public-input.json`、`author-setup.json`、
`freeze.json`、清单明确列出的 Markdown 文档，以及一份运行配置；命令行没有 gold、
标注、仲裁、commitment 或任意额外数据文件入口，也不会遍历公共包寻找其他文件。

执行器在真正发起 Provider 调用前会校验：

- 三份公开控制文件的规范化哈希、数据集 ID、case/group 数量和作者轴绑定；
- 文档 UUID 路径、UTF-8/LF、内容哈希，以及每个非空行恰好一个单行证据锚；
- 本机专用端口和显式配置的隔离数据库身份，且首次执行时数据库必须为空；
- Character Consistency、精确 support binding 和完整行证据能力已经启用；
- 每个 case 都创建全新项目，baseline 严格按作者配置导入为已确认/已发布，target 只在
  唯一候选确认完成后以已确认草稿身份导入；所有未选候选都显式拒绝。

候选必须同时命中角色、维度、来源类型、方向、稳定度、冻结来源文档、版本、哈希和精确
行区间，且精确 support binding 可核验。运行结果中的 `B` 只能来自该 selector 的来源
文档及行区间，`C` 只能来自 target 文档；所有引用必须与公开目录中的一个单行锚首尾完全
一致，不能用宽区间吸入多行。正式问题还必须有两个经运行诊断确认相互独立的 `C` 事件。
正式问题或待复核线索的报告行必须实际展示非空证据，并与最终诊断轨迹按角色和坐标完全
一致；预测引用只从这组已交叉核验的报告证据投影，不能由内部 trace 补齐报告没有展示的
`B/C/G/X/P`。`no_issue` 没有展示行，只能保留通过同样来源、坐标和角色约束的 trace
解释证据。

报告集合、coverage、trace、引用、候选绑定或接口响应只要有一个不完整/歧义，该 case 就
输出 `unassessed`；执行器不会把“没看到问题”猜成 `no_issue`。最终产物是哈希绑定的
`SealedHttpRunReport`，它本身不包含评分能力；内部 prediction artifact 只是报告结构的
一部分，不能脱离原生报告进入正式评分。

运行配置示例（所有身份值均由独立评测实例的预置流程产生）：

```json
{
  "schema_version": "loreguard-ooc-sealed-http-config-v1",
  "base_url": "http://127.0.0.1:8101",
  "isolation": {
    "schema_version": "character-axis-evaluation-isolation-v1",
    "isolated_sqlite": true,
    "eval_db_id": "<uuid-v4>",
    "database_path_sha256": "<sha256>",
    "instance_id_sha256": "<sha256>"
  },
  "execution_id": "<uuid-v4>",
  "repeat_id": "repeat-01",
  "expected_runtime_provenance_sha256": "<sha256>",
  "expected_runner_source_sha256": "<runner-code-manifest-canonical-sha256>",
  "sensitivity": "balanced",
  "run_timeout_seconds": 1800.0,
  "poll_interval_seconds": 1.0
}
```

```powershell
# 只做本地冻结包与配置校验，不连接 HTTP，也不调用 Provider
python scripts/run_ooc_sealed_http.py `
  --bundle-root <public-directory> --run-config <run-config.json> `
  --output-json <preflight-report.json> --preflight-only

# 真正运行必须显式二次确认 Provider 调用；输出文件已存在时拒绝覆盖
python scripts/run_ooc_sealed_http.py `
  --bundle-root <public-directory> --run-config <run-config.json> `
  --output-json <repeat-01.sealed-run.json> --allow-provider-call
```

当前只用 Mock HTTP 覆盖了完整成功链路和 fail-closed 分支，尚未调用真实模型；这部分测试
不能替代后续的真人双标、封存、一次固定配置真实运行与评分。

执行报告保存完整的运行配置快照及其哈希、执行代码清单哈希、`execution_id`、运行时来源
哈希，以及逐 case 的项目 ID 哈希回执和规范化项目集合哈希；顶层字段与内嵌预测 artifact
逐项强绑定。正式 scorer 应直接读取并严格校验整个 `SealedHttpRunReport` 后再使用其中的
artifact，不能要求操作者手工复制或抽取内层 JSON。该回执用于防止误接旧服务、旧脚本或
复制错产物，并不声称能够对控制本机和全部输入的恶意操作者提供密码学证明。
评分侧可调用只读 helper `current_runner_source_sha256()` 获取当前执行代码清单的规范化哈希；
该清单固定包含 sealed HTTP runner、预测投影器和评测契约三个源码文件，任一依赖变化都会
使已冻结运行配置失效。helper 只读取这三个仓库源码文件，不读取公共包或私有答案。

## 评测指标与声明边界

人工侧先报告仲裁前 outcome/surface 原始一致率、Cohen κ、证据角色 Jaccard 和分歧 case
数量。样本规模较小时只作为标注手册校准依据，不做统计泛化。

系统侧继续使用保守指标：正式问题 precision、最差单轮 recall、困难负例 specificity、
`B+C` 完整率、禁止/非法引用、联合准确率和覆盖率。`partial`、`failed`、缺失或
`unassessed` 一律不得获得“正确无问题”信用。

当前首版公共冻结包包含 36 个 case、12 个互相独立的原创故事组，六维度各 6 个。
在真人仲裁完成前不得把预设构成当作最终标签分布；即使完成评分，这个规模也只能提供
第一轮工程方向，不足以声称生产准确率。如需稳定的分维度统计，应在协议验证后扩大
样本，而不是先用少量 holdout 反复调参。

## 当前状态与尚未完成事项

截至 2026-09-29，36 例公共冻结包与 A/B 两份自包含离线标注页面已经生成并保存在仓库
之外；仓库内具备合同、页面生成器、双提交比较、裁决绑定、gold commitment、强制验签
评分和不挂载私有答案的密封 HTTP runner。这只表示输入与工具准备完成。

两位真人的真实独立标注、锁定提交、分歧裁决、私有 gold/commitment、固定配置真实模型
运行、正式评分以及由真实失败驱动的修复均尚未完成。因此当前没有人工盲测结论、角色
OOC 封闭评测成绩或开放文本准确率，后续也不得把 Mock HTTP 测试或已生成页面描述为
盲测完成。
