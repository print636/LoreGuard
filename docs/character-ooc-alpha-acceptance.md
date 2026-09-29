# 角色 OOC Alpha 功能验收

`data/character-ooc-alpha-v1` 是原创、开发者可见的合成功能开发集。它只回答：
当前构建能否在一个受控故事中走完角色设定确认、六类主要维度抽取、解释检索、
两次独立事件复核和分层报告。它不是人工封闭测试集，也不支持开放文本准确率、
生产质量或真实游戏项目效果的结论。

## 有边界的覆盖范围

- 主要维度：核心性格、稳定偏好、说话方式、价值取向、关系态度、长期动机。
- 价值取向中的唐岫个案不再依赖 legacy 文字标签。作者须明确确认比较轴定义、正向命题
  与适用情境；runner 在唯一候选选定后只从候选冻结快照取得对象键，经公开 API 创建并
  核验轴，再提交轴版本、同向选择、命题/情境哈希与情境适用确认。轴创建或核验失败时，
  不发送任何候选审核 decision。
- 硬负例：已经发布的成长经历、当前新稿中明确生效的伪装。
- 分层结论：两次独立反向事件可进入正式问题；一次偏好反转只能进入待复核线索；
  未通过完整抽取校验的提案始终留在未核实提案层。
- 该套件不穷举文字游戏、隐喻、复杂群像指代或所有行业题材。新的真实失败样本先进入
  开发/回归集，用于优化提示词、检索、结构化抽取或裁决逻辑。

## 通过条件

运行：

```powershell
python scripts/run_character_consistency_live.py --fixture alpha-v1 --trials 1
```

真实 Alpha 运行前须显式启用
`ENABLE_CHARACTER_CONSISTENCY=true`、`CHARACTER_EXPLANATION_REVIEW_V1=true`、
`CHARACTER_SCOPED_AXIS_DRIFT_V1=true`、`CHARACTER_SIGNAL_SUPPORT_ID_V4=true`、
`CHARACTER_SIGNAL_SEMANTIC_SCOPE_V5=true`、`CHARACTER_SIGNAL_SCOPE_REVIEW_V1=true`、
`CHARACTER_DRAFT_ACTOR_REVIEW_V1=true` 和
`CHARACTER_TARGET_BOUND_DRAFT_REVIEW_V3=true`（且 V2 保持关闭），并重启 API（以及使用中的 worker）。
这些实验能力不由 runner 自动开启，也不因此改变生产默认值；preflight 会严格核验
正式候选精确绑定链、scoped-axis v1 开关及 target-bound semantic 当前整组协议版本；
其中 runtime 模型输出合同必须为 request/response/batch V4 与 `character-target-bound-draft-review-prompt-v8`。唯一 raw target literal 的 `matches_target` 使用服务端坐标绑定并要求模型返回 null offsets；`broader`/`narrower` 以及非唯一情形仍要求合法的事实分句区间。坐标绑定不替代任何语义、basis 或来源门。
当前 V4 在定向模型返回 clean-empty 时仍只复核服务端白名单候选；唯一同对象关系由服务端
绑定冻结 raw literal，其余对象关系仍保存经严格区间验证的冻结原文切片。旧 V2/V3
provenance 不会被重解释为 V4，也不能通过当前 Alpha preflight。
解释复核还必须报告 `character-explanation-review-v2` /
`character-explanation-review-prompt-v2`；缺少身份字段的历史 V1 运行可只读，但不能通过
当前 Alpha preflight。
仅开启 support-ID v4 不足以产生 `required_v1`：V4 定位断言，semantic-scope v5
提供承接关系，scope reviewer 才将获支持的主张绑定到冻结运行输入。因此运行来源须为
`assertion-index-v1` / `semantic-scope-v6` / `character-scope-review-v2` /
`character-scope-review-prompt-v4`；这些身份由服务端派生，不是额外环境变量。v6 明确支持
同行后置的核心/稳定总结标签，但仅能通过 `postposed_label_summary` 提交；label 必须是该行
最后一个断言，且独立复核须覆盖从目标到标签的完整分句路径。标签后仍有任何断言、跨行、
换主体、不同语义轴或依据不完整，均失败关闭。路径内同轴事实自身的负向措辞（例如“不
会主动长谈”）不等于否认目标；复核器仍须按目标语义与 polarity 判断，不能仅凭“不”字拒绝。

一次运行会新建独立项目，不复用旧结果。默认只跑一次以控制真实模型调用；需要观察
稳定性时可以显式增加 `--trials`。通过必须同时满足：

1. 基线与新稿运行均结束，角色阶段没有降级。
2. 以 `model_completed_chunks == planned_chunks`、`model_incomplete_chunks == 0`
   为模型抽取完成依据；旧字段 `processed_chunks` 只表示遍历过分块，不能单独证明完成。
3. 每个被评分个案的材料覆盖和解释检索覆盖均为 `complete`，阶段内不存在部分覆盖个案。
4. 分别读取 `/issues`、`/review-clues`、`/provisional-clues`；正式问题与待复核线索
   必须逐案匹配 oracle，且不得存在未核实提案、不可显示旧线索或被截断的线索集合。
5. 成长和伪装两项必须不产生正式问题；一次反转必须只出现在待复核线索中。
6. 唐岫价值轴必须由作者定义并确认情境适用，且报告只保留轴 ID 哈希、版本、命题/情境
   哈希和操作状态，不回显作者正文、候选对象键或模型输出。作为轴来源的正式候选必须
   为 `support_binding_mode=required_v1` 且 `support_bindings_status=verified`；`legacy`
   或 `invalid` 不计为可验收候选。

`status=completed`、`processed_chunks == planned_chunks` 或“没有正式问题”中的任一项，
都不能单独视为通过。材料/解释覆盖为 `partial`、线索无法核验或存在未核实提案时，
本轮属于未充分评估，不能记为正确无问题。

## 后续封闭测试

这套 Alpha 材料和轴计划都对开发者可见，作者确认也只是开发流程合同，不是盲标答案。
真正的质量指标需要另外建立人工标注且在开发期不查看答案的封闭测试集。开发集可反复
用于提示词、检索、规则和模型选择；封闭集只用于阶段验收。若团队根据封闭集中的具体
错误修改系统，该集合随后必须降级为开发/回归集，并准备新的封闭样本。模型微调不是
默认步骤，只有在高质量标注已积累且提示词与工程流程仍存在稳定系统性缺陷时才考虑。

## 最新开发检查点

2026-09-29 的 [Alpha v20 脱敏检查点](character-ooc-alpha-v20-checkpoint-20260929.md)
记录了一次开发者可见合成 DEV 全流程：8 个已知案例的预期分层均匹配，20/20 必要 gate
通过。它只有单次 trial，不是盲测、生产质量或开放文本泛化证明，也未评估跨轮稳定性。
