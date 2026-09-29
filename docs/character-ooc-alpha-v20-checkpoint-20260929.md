# 角色 OOC Alpha v20 脱敏检查点（2026-09-29）

本页只记录本地产物 `artifacts/character-ooc-alpha-v1-20260929-v20.json`
可以证明的事实。该产物来自**开发者可见的合成 DEV 材料**，仅执行 **1 次**完整工作流；
它不是人工盲测、不是封闭测试、不是生产质量证明，也不能外推到开放文本、其他故事、
其他模型配置或真实游戏项目。本文不保存 API Key、Provider endpoint、原始 Prompt 或模型
原始响应。

## 结论与运行身份

- 产物 schema：`character-continuity-live-dev-v3`；最终 `passed=true`。
- 必要 gate：`20/20` 通过。唯一 observational gate
  `stable_visible_issue_semantic_identity_across_independent_trials=false`，因为本次只有一轮，
  未评估跨独立轮次稳定性。
- 本地服务产物匹配检查已启用且匹配：
  `local_service_artifact_match_enforced=true`、
  `local_service_artifact_matches_live_api=true`；未使用远程产物绕过。
- 服务产物 SHA-256：
  `cff7ebf903d97999e8ef7e03a6d6abe6a8791e9fb7d27b9e4b2200a5b745df29`。
- target-bound runtime 身份：review schema
  `character-target-bound-draft-review-v4`、batch schema
  `character-target-bound-draft-review-batch-v4`、Prompt
  `character-target-bound-draft-review-prompt-v8`（V4 / V8）。

## 8 个案例的分层结果

| 案例 | 维度 | 本轮分类 |
| --- | --- | --- |
| 黎音 | 核心性格 | 正式问题：`conflict` |
| 白榆 | 说话方式 | 正式问题：`conflict` |
| 唐岫 | 价值取向 | 正式问题：`conflict` |
| 余霁 | 关系态度 | 正式问题：`conflict` |
| 顾潮 | 长期动机 | 正式问题：`conflict` |
| 沈砚 | 稳定偏好 | 待复核线索：`needs_confirmation` |
| 苏弦 | 核心性格 / 已发布成长 | `no_issue` |
| 祁雾 | 说话方式 / 当前伪装 | `no_issue` |

oracle 中的 8 个预期案例全部通过，且没有额外 runtime-bound 案例。正式问题为 `5`，
待复核线索为 `1`，未核实提案为 `0`；正式问题、待复核线索与未核实提案的集合完整且
严格分离。可见正式问题的安全签名、待复核线索的安全签名及报告层级均与 oracle 精确
匹配。这里的 `8/8` 只表示对这 8 个已知 DEV 案例的预期分类一致，不是准确率或召回率。

## 覆盖与 gate 摘要

- 基线：计划 `10` 个分块，模型完成 `10` 个，未调用 `0`、未完成 `0`；阶段完整结束。
- 新稿：计划 `11` 个分块，模型完成 `11` 个，未调用 `0`、未完成 `0`；阶段完整结束。
- 案例材料：`8/8 complete`、`0 partial`。
- 解释复核：要求 `8` 个、完成 `8` 个，整体 `complete`。
- 候选复核：确认 `8`、拒绝 `0`，与预期确认数 `8` 一致；作者轴创建并绑定 `1` 个。
- 其余必要 gate 覆盖基线/新稿完成度、模型 Token 报告、候选集合精确匹配、报告层级
  完整性、案例语义身份、无额外案例、API/worker 与运行前后 provenance 一致，以及严格
  候选选择器身份；本轮均为 `true`。

## 资源记录

| 阶段 | elapsed | run summary prompt / completion Token | stage attempted calls | stage input / completion Token | stage charged Token |
| --- | ---: | ---: | ---: | ---: | ---: |
| 基线 | 84.921 s | 45,355 / 4,706 | 18 | 42,920 / 3,294 | 105,898 |
| 新稿 | 304.375 s | 112,213 / 20,332 | 51 | 109,213 / 15,635 | 241,768 |
| 合计 | 389.296 s | 157,568 / 25,038 | 69 | 152,133 / 18,929 | 347,666 |

`stage charged Token` 是产物中的内部保守计费字段，不等同于 Provider 返回的实际
prompt/completion Token；因此不能与 run summary Token 混作同一口径。

## 数据集 SHA-256

| 文件 | SHA-256 |
| --- | --- |
| `01-world-setting.md` | `6599585a750e2ee1febe3f324a2b47c5bf3ead6136926fafcdbc61732dc13298` |
| `02-character-profiles.md` | `818264485b2beaac51575d915201da8b6e47e5e749ec7b120b54a1b259af3f1d` |
| `03-published-history-v1.0.md` | `df57d852e2196992b40047c00a85556495b36b36809f67e3bb4221488ecde147` |
| `04-draft-event-v1.1.md` | `2c5d823f19fcfb34c1b423f3db061f6f3c6833dffd22dd9f918215d55b13f2e1` |
| `acceptance-oracle.json` | `74cb7a64b96b1f0cde40f92def56dc577a2f019a40538dee6b96eff5a5341957` |

## 声明边界

本轮没有独立重复试验，未评估语义身份跨轮稳定性；产物也显式标记
`blind_holdout=false`、`production_quality=false`、
`open_text_generalization=false`、`semantic_coverage=false` 与
`independent_full_workflow_trials=false`。因此可审计结论仅限于：当前服务产物在一轮
真实工作流中完整执行并正确分类这套已知、开发者可见的 8 案例合成 Alpha 材料。
