# 镜潮港角色 OOC Alpha 功能开发集

这是 LoreGuard 的原创、开发者可见合成材料，用于验证角色连续性主链路能否在真实
Provider 下完成“设定候选确认 → 新稿抽取 → 解释检索 → 分类报告”。它不是人工
封闭测试集，不用于声称开放文本准确率、生产质量或真实游戏策划验收结果。

覆盖边界保持有意克制：

- 六类主要角色维度：核心性格、稳定偏好、说话方式、价值取向、关系态度、长期动机。
- 唐岫的价值取向个案必须走产品内的作者比较轴流程：作者在 oracle 中明确给出轴定义、
  正向命题和适用情境；runner 先唯一选中设定候选，再从该候选的冻结快照读取并核验
  `comparison_key`，通过公开 API 创建轴并显式确认同向与情境适用。oracle 不保存或猜测
  该对象键，也不以旧式模型标签代替作者确认。该候选必须带冻结子句的精确绑定且
  `support_bindings_status=verified`；`legacy` / `invalid` 均不得进入轴确认。
- 两类必须抑制的硬负例：已经发生并发布的成长，以及新稿中明确持续生效的伪装。
- 一次偏好反转只能进入“待复核线索”，不得混入正式问题。
- 正式问题、待复核线索和未核实提案必须分别读取与计数。
- 任一相关个案材料覆盖或解释检索覆盖不完整，本轮不得按“正确地没有问题”计分。

基础 Alpha 中的每个个案只验证一项主能力：偏好反转沿用同一完整对象，作者价值轴的
两次事件都明示其适用情境，已发布成长也明示训练与后续表现之间的关系。对象上下位
泛化、隐含多跳因果等更高难度问题应放入独立挑战集，不能混入基础链路验收后再把失败
误归因于角色连续性主流程。

这套材料可以反复用于提示词、检索、结构化抽取和裁决流程的开发。未来应另建由人工
标注、在开发期间不查看答案的封闭测试集；一旦依据某个封闭样本定向修改系统，该样本
就应降级为开发/回归集并更换新的封闭样本。

运行前必须显式启用 `ENABLE_CHARACTER_CONSISTENCY=true`、
`CHARACTER_EXPLANATION_REVIEW_V1=true`、`CHARACTER_SCOPED_AXIS_DRIFT_V1=true`、
`CHARACTER_SIGNAL_SUPPORT_ID_V4=true`、`CHARACTER_SIGNAL_SEMANTIC_SCOPE_V5=true`、
`CHARACTER_SIGNAL_SCOPE_REVIEW_V1=true`、
`CHARACTER_DRAFT_ACTOR_REVIEW_V1=true` 与
`CHARACTER_TARGET_BOUND_DRAFT_REVIEW_V3=true`（同时保持
`CHARACTER_TARGET_BOUND_DRAFT_REVIEW_V2=false`），重启 API（以及 Celery worker，如使用）
后执行。它们仍保持生产默认关闭；runner 只核验已启动进程公布的能力，不会替部署修改默认值：
仅开启 support-ID v4 不会生成 `required_v1` 候选：V4 定位目标断言，V5 提供作用域
关系，scope reviewer 才把获支持的主张绑定到冻结运行输入。运行来源必须同时报告
`signal_support_segmenter_version=assertion-index-v1`、
`signal_semantic_scope_version=semantic-scope-v6`、
`signal_scope_review_schema_version=character-scope-review-v2` 与
`signal_scope_review_prompt_version=character-scope-review-prompt-v4`；这些版本均由服务端
派生，不是额外的手工环境变量。兼容开关仍名为 target-bound V3，但当前 runtime 必须报告
`target_bound_draft_review_schema_version=character-target-bound-draft-review-v4`、
`target_bound_draft_review_batch_schema_version=character-target-bound-draft-review-batch-v4` 与
`target_bound_draft_review_prompt_version=character-target-bound-draft-review-prompt-v8`；
旧 prompt-v5 不能通过当前 preflight。

```powershell
python scripts/run_character_consistency_live.py --fixture alpha-v1 --trials 1
```

一次 V4 / Prompt V8 的真实模型单轮结果与完整声明边界见
[2026-09-29 Alpha v20 脱敏检查点](../../docs/character-ooc-alpha-v20-checkpoint-20260929.md)。
该结果仍只是开发者可见合成 DEV 检查，不是人工盲测、生产质量或开放文本泛化证明。
