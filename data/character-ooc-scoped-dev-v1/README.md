# 善潮号｜带对象与情境的角色审查 DEV 集

这是一套原创、开发者可见的合成 DEV 材料。作者确认在评测中只能由开发者显式模拟，不能假称真实作者审核；它不是盲测、真实用户稿件或生产质量证据，不能对外宣称准确率。素材不含商业游戏 IP 或公司内部资料。

目标是检查作者明确设定、当前行为对象、适用情境与行为归属是否被逐一核对，以及两条原文是否真是两次独立事件。`oracle.json` 是评测答案，绝不能上传给模型或放进 Prompt。这里没有模型运行结果；预期结果不是已测成绩。

`01-world-setting.md`、`02-character-profiles.md` 和 `03-published-history-v1.1.md` 是每案共用材料。`drafts/` 下每份文档都是一项独立案例；在全新的隔离项目中只导入该案例的一份草稿。十份草稿不得合并，否则不同案例的行为可能被错误累计。`manifest.json` 明确记录每案要核对的作者轴。`author-plan.json` 仅记录拟由开发者模拟的作者确认选择及各轴唯一的原文短句，默认状态仍为未确认；它不能代替在产品内审核候选、核对短句对应的精确证据和显式确认方向。

基线优先级为正式角色设定，其次是已发布历史；草稿是待审新稿。世界规则将绿讯定义为日常出航状态、红讯定义为迫近危险时的救援状态；公共屏面向不特定船民，照护队专线仅面向指定成员。已经发生的训练可以解释表达方式变化，但不会自行撤销作者确认的设定。

按 `manifest.json` 为每案建立新隔离项目，导入三份共用文档，再导入对应草稿。正式档案的角色特征、对象比较键、适用情境和轴正向命题必须逐条由审核者显式确认；无法确认时该案标为 `unassessed`，不能拿草稿的“0 个问题”作负例成功。运行实验路径须显式开启 `ENABLE_CHARACTER_CONSISTENCY=true`、`CHARACTER_SCOPED_AXIS_DRIFT_V1=true`、`CHARACTER_SIGNAL_FULL_LINE_PROMPT_V2=true`、`CHARACTER_SIGNAL_SUPPORT_ID_V4=true`、`CHARACTER_SIGNAL_SEMANTIC_SCOPE_V5=true` 与 `CHARACTER_SIGNAL_SCOPE_REVIEW_V1=true`。仅开启 V4 不会产生可确认的 `required_v1` 精确绑定；运行时还须报告 `signal_support_segmenter_version=assertion-index-v1`、`signal_semantic_scope_version=semantic-scope-v6`、`signal_scope_review_schema_version=character-scope-review-v2` 和 `signal_scope_review_prompt_version=character-scope-review-prompt-v4`。记录实际生效配置、模型名、构建版本、材料哈希和覆盖状态；只在隔离实例调用模型。

离线可执行 `scripts/run_scoped_ooc_dev_v1.py --phase preflight` 校验固定输入与哈希；真实模型阶段须在独立评测库、非日常服务端口、干净 Git 构建上显式传入 `--phase prepare --allow-provider-call`、隔离数据库 ID/实例 ID/绝对路径和 `--output-json`。所有报告、检查点和作者决定文件路径必须位于本仓库被 Git 忽略的 `artifacts/` 目录内。运行器会在创建项目或调用模型前读取服务 `/health` 的实际运行配置，要求 V4 支持 ID 与指定分句器版本均生效。准备阶段每案建立独立项目，只做基线，不上传草稿；其检查点记录真实候选 ID 与唯一性。随后审核者逐案核对原文、精确证据区间、对象、适用情境和轴方向，在另一个 `artifacts/` JSON 文件写入全部 10 项显式决定，再用 `--phase review --checkpoint-json ... --author-decisions-json ...` 运行草稿审查。决定文件须包含 `schema_version=scoped-ooc-dev-author-decisions-v1`、固定清单哈希、检查点文件 SHA-256、`reviewer_declaration=developer_simulated_author_reviewed_frozen_source_axis_and_direction`，以及每案的 `case_id`、`axis_key`、`candidate_id`、`decision`（`confirm`/`defer`）和 `axis_alignment`（`same`/`opposite`；暂缓时两个字段填 `null`）。脚本不会根据答案自动确认；未提取到唯一且与冻结原文精确匹配的候选时只能暂缓。

首次探索性隔离 prepare 在 `CHARACTER_SIGNAL_SUPPORT_ID_V4=false` 下完成了 10 案基线，但精确目标候选匹配均为 0。这是评测协议与运行配置不一致，结果一律视为 `unassessed`，不作为角色判断能力或准确率指标；原始报告保留在本地排查材料中。修正配置后需使用新的隔离评测库重新运行。

当前 review 阶段不支持中断后在同一评测库逐案续跑。若中途失败，应保留失败报告用于排查，并使用新隔离评测库重新执行 prepare/review；不能直接重放已做出的确认。

评测时按阶段核对：草稿原文是否被抽取；人物、行为、对象和原文行号是否正确绑定；每条当前证据的对象与情境判断；是否确属独立事件；复核结论及 B/C/G 引证；最终是正式问题、待复核线索还是弃权。`conflict` 需要作者已确认且本轮有效的基线、完整材料、正确归属、相同对象与情境、足量独立行为以及双侧原文证据。一个案例可以正确地产生零条角色观察，也可以在证据不足时弃权；后者不能被计为“判断正确的无问题”。

`oracle.json` 为每案都重复标注“开发者可见合成 DEV、模拟作者确认、不可对外宣称准确率”。本集小而公开，只适合暴露错误和比较修复前后行为；即便全部符合预期，也不能换算成开放文本准确率或写成简历里的质量指标。要做质量声明，还需另建冻结、独立标注、与开发过程隔离的测试集。
