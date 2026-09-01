# 状态建模 v2 评测记录

本轮只使用 `natural dev` 的已知错误设计移动许可、知识来源和规则例外状态。原 `natural test` 已在更早阶段被开发者查看，因此它的修复后结果只作为回归参考，不能称为无偏提升。

## 固定数据与哈希

- natural dev（40 例）：`26f65a0c5bb9625eac1e04eb5e993ea1419082c8ae036b4706a5dbe30804d496`
- natural test（60 例）：`e6343eb8a7a5ec08ebe66e7da7f35d70cce66109632ddad9167f3679479bd5e8`
- challenge-v2（50 例）：`37470c6e8479c6bc3e5789f8a800f69ca00a40558a1ee97147866b98dddfa3fe`

challenge-v2 使用固定 seed `20260902`，五类各 10 例、正负各 25 例，覆盖不同实体、措辞和结构。它仍是固定模板生成、开发者可见的 synthetic natural-language 数据，不是人工标注集，也不是盲测。

## Before / After

| 数据 | 角色 | 阶段 | TP | FP | FN | Precision | Recall | F1 | 证据命中率 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| natural dev | 调参与回归 | before | 20 | 12 | 0 | 0.625 | 1.000 | 0.769 | 1.000 |
| natural dev | 调参与回归 | after | 20 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| natural test | 已查看的回归参考 | before | 30 | 18 | 0 | 0.625 | 1.000 | 0.769 | 1.000 |
| natural test | 已查看的回归参考 | after | 30 | 0 | 0 | 1.000 | 1.000 | 1.000 | 1.000 |
| challenge-v2 | 开发者可见合成挑战 | before | 25 | 15 | 0 | 0.625 | 1.000 | 0.769 | 0.800 |
| challenge-v2 | 开发者可见合成挑战 | after | 25 | 1 | 0 | 0.962 | 1.000 | 0.980 | 1.000 |

完整报告与逐案例误差保存在 `artifacts/state-modeling-v2/`。challenge-v2 修复后仍有 1 个 `location_collision` 误报：一种带有效期、双向路线措辞的许可没有被保守抽取。为避免循环调 challenge，本轮保留该错误，不继续扩张正则。

## 状态语义

- 移动许可使用 `fact(predicate=mobility_permission)`，绑定角色、起点、终点、是否双向、状态与有效期。只有角色、路线和时间都适用时才抑制同刻异地问题。
- 知识来源继续使用 `knows`，显式记录来信、目击、告知、阅读或公告来源及时间；只有同角色、同 canonical fact 且早于 claim 的记录有效。
- 规则例外使用 `fact(predicate=rule_exception)`，绑定角色、规则 canonical key、状态和有效期；`world_assert` 记录 actor。只有同 actor、同 key 的有效例外才抑制规则冲突。
- 两条显式带时间且时间不同的普通事实视为状态变化，不直接判冲突；无时间的持续状态冲突规则保持不变。

这些结果只证明固定合成模板上的行为，不能外推到开放长篇中文文本或生产准确率。下一步仍需独立人工标注、未参与开发的盲测集。
