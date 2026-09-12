# Evidence Investigator Live E2E 评测夹具

这是一套面向 LoreGuard 原生 Evidence Investigator 工具循环的中文小型评测夹具。它检验模型能否从一个确定性基线种子出发，主动检索另一份文档中的自然叙事证据，并把候选记录提交给现有的授权、语义质量和确定性规则复验链路。

## 边界与许可

- 全部世界、人物、地点、装置和文本均为本仓库原创虚构内容，不使用商业游戏、小说或公司内部语料，可随 LoreGuard 以 MIT 许可证分发。
- 本数据集不是公开文本能力证明，也没有预先声明任何模型精确率、召回率或成功率。`manifest.json` 中的 `live_model_status` 保持 `not_run`，直到一次可复现的真实模型运行完成。
- 显式 `@...` 记录只出现在 `canon.md`，用途是稳定地产生受信任 seed。每个待发现的矛盾侧或干扰侧都写在 `chapter.md` 的自然叙事中，不使用指令捷径。
- Ground truth、候选字段和答案只供评测器使用，绝不能拼入系统提示词、工具说明、检索 query 模板、few-shot 示例或模型上下文。

## v2 冻结说明（2026-09-13）

- 一次保留的 v1 dev-only 真实运行暴露了确定性归一化缺陷：“发动了某动作”中的完成体“了”会被错误纳入动作名。生产归一化器已在源头修复；v1 运行产物只保留为历史诊断，不能与 v2 产物组成 dev pair。
- 为继续隔离验证 Investigator，而不是让修复后的基线直接发现目标，v2 只把开发集 `EIL-D-05` 的实际动作改为当前基线不直接抽取、但既有 promotion grounding 明确支持的“执行了……”表达。候选字段、证据行和预期问题均未改变。
- 本次版本升级没有修改或打开 holdout 正文，也没有修改其标签或预期；没有运行 holdout，更没有使用 holdout 结果选择措辞、规则、Prompt、检索参数或阈值。

## 数据划分

- `dev/`：8 个开发案例。包含五类正例和 3 个困难负例，可用于调试调用接线、观察失败类型和调整通用提示词，但不得把案例原文或答案改写成提示示例。
- `holdout/`：10 个保留案例。包含五类全新正例和梦境、假设、规则例外、分支作用域、地点层级 5 个困难负例。开发期间不得查看运行结果后再改提示词、检索参数、规则或阈值；应在配置冻结后一次性运行并保存原始结果。

每个案例是独立项目切片，默认只加载该案例目录下的两个文档，防止跨案例同名实体或时间记录互相污染。`manifest.json` 是唯一机器可读标注源；模型运行时只应收到其中 `sources` 指向的文档内容及生产代码正常提供的 seed/tool schema，不应收到 `expected`、`candidate`、`lure_candidate`、`difficulty` 等字段。

`freeze.json` 固定了 manifest、说明和全部 36 份模型输入文档的 SHA-256。修改其中任一文件时必须升级数据集版本并重新冻结，不能在查看 holdout 运行结果后原地改写 v1。`validate_fixture.py` 是只读验证器：它不调用模型，检查文件完整性、单 seed 隔离、无模型基线零问题，以及候选经过完整授权与 promotion 链路后的预期结果。

## 覆盖范围

正例覆盖当前确定性闭包中的五类问题：

1. `fact_conflict`
2. `location_collision`
3. `knowledge_without_acquisition`
4. `item_ownership`
5. `world_rule_conflict`

困难负例覆盖：许可不等于实际使用、命令或转述不等于执行、跨时间状态变化、梦境、条件假设、有效规则例外、互斥剧情分支，以及同一地点与内部子区域的层级关系。梦境案例既检验模型能否主动 abstain，也验证模型误提交“现实事件”时，文档上下文语义门会在 promotion 阶段拒绝候选；判分时应区分这两条路径。

## 推荐运行协议

1. 固定代码提交、模型名、provider base URL、温度、工具循环预算、检索索引版本和所有 feature flags。
2. 先在 `dev` 上验证 E2E 接线。允许修复通用 bug，但不允许把任何案例文本或答案加入提示词。
3. 冻结配置后运行 `holdout` 一次。每个案例建立独立项目/analysis run，并仅导入该案例的 `canon.md` 与 `chapter.md`。
4. 保存每轮工具调用、最终 `added_issue`/`abstain`、promotion 拒绝码、token 用量、延迟与运行失败。Provider 失败与模型判断错误必须分开统计。
5. 正例只有在候选证据经过授权绑定、字段 grounding、语义质量门和 `detect_issues` 确定性复验后，才计为成功；模型仅输出一个看似合理的 JSON 不算成功。
6. 负例必须显式 `ABSTAIN`，或由 promotion 安全拒绝且最终未新增问题，才算没有误报。若只是 provider/预算/超时失败，不算正确 abstain。

## 判分建议

不要从这 18 个样本外推生产质量。至少分别报告：

- 正例 `added_issue` 命中数，按五类规则拆分；
- 负例安全 abstain 数，以及“模型主动 abstain”和“promotion 拒绝”两种路径；
- 错误候选、错误证据行、错误字段、未检索到证据、工具预算耗尽、provider 失败；
- 真实 prompt/completion tokens、端到端延迟与模型/provider 配置。

任何指标都必须由实际运行产物计算，不能把 manifest 中的预期标签当成测量结果。

## 文件约定

- `canon.md`：可信基线锚点；允许使用明确指令，以隔离 seed 生成是否成功这一变量。
- `chapter.md`：模型需要检索与理解的自然叙事；禁止添加 `@directive`。
- 行号是 1-based、闭区间。当前目标叙事均位于 `chapter.md:5`，锚点通常位于 `canon.md:3`；评测器仍应以 manifest 为准，而不能硬编码这一巧合。
- `story_scope` 必须来自 manifest 的文档元数据并在导入时保持；特别是 `EIL-H-09` 的两个文档属于不同分支。

## 防泄漏清单

- 不把 dev 或 holdout 原句用作提示示例。
- 不把候选字段、预期类别、允许证据行或案例解释提供给模型。
- 不用 holdout 结果选择模型、改 query、补同义词、调预算或改阈值。
- 不把 holdout 纳入向量索引的跨项目语料库。
- 演示或 README 可以描述覆盖的现象，但不得公布 holdout 答案后继续声称它仍是保留集。
