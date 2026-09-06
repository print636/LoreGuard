# Issue Review v1 真实模型 A/B 结果

日期：2026-09-07。

本结果来自预先冻结的 12 例原创证据复核集，每例重复 3 次。`local-context` 与 `rag-evidence` 使用相同的聊天模型、thinking 设置、结构化输出合同、Token 预算、超时和 deadline；区别仅是后者通过生产 `IssueEvidenceReviewer` 使用真实 embedding、PostgreSQL/pgvector 与 RRF 补充获授权证据。

这是开发者可见的小型冻结工程评测，不是人工盲测、开放文本准确率或生产质量证明。评测未保存或公开 API Key、服务地址、Prompt、故事原文、note 或模型原始响应。

## 结果摘要

| 指标 | Local context | RAG evidence |
| --- | ---: | ---: |
| 三次均正确的案例 | 4/12 | 7/12 |
| `supports_issue` | 2/4 | 4/4 |
| `contextual_exception` | 1/4 | 3/4 |
| `insufficient_evidence` | 1/4 | 0/4 |
| 三次均覆盖黄金证据的案例 | 0/12 | 8/12 |
| 引用 allowlist | 42/42 | 106/106 |
| 排除来源泄漏 | 0 | 0 |
| 失败 / 降级运行 | 0 / 0 | 0 / 0 |
| P95 | 5.823 秒 | 6.445 秒 |

RAG 相比 local-context 多答对 3 例，情境例外多答对 2 例，黄金证据覆盖增加 8 例。RAG 的 36 次模型复核均完成，无失败或降级；106 条引用全部落在当前项目、当前获授权版本的 allowlist 内，没有引用冻结集中的旧版或错误来源。

## Gate 结论

Absolute gate 为 **`false`**，A/B 总 gate 也为 **`false`**。

已达到的部分：

- RAG 相比 local-context 总体多答对至少 2 例。
- 情境例外多答对至少 2 例。
- 引用 allowlist 命中率 100%，排除来源泄漏为 0。
- `contextual_exception` 达到 3/4。

未达到的部分：

- 总体要求至少 10/12，实际为 7/12。
- 黄金证据覆盖要求至少 11/12，实际为 8/12。
- 情境例外不得被判成 `supports_issue`，实际 3 次运行出现此错误。
- `insufficient_evidence` 为 0/4，说明当前模型合同倾向于在证据不足时作出确定判断，这是最重要的质量缺口。

因此只能声称“真实 RAG 证据已进入下游模型复核，并在冻结小样本上改善了情境例外和证据覆盖”；不能声称 AI Reviewer 质量达标、可靠识别信息不足或已适合生产自动裁决。

## 产品边界

Evidence Reviewer 是规则引擎之后的可选注释层：

- 确定性规则先生成 issue。
- Reviewer 只消费本次运行冻结且获授权的检索证据。
- Reviewer 的 verdict 和引用保存在 `metadata.ai_evidence_review`。
- Reviewer 不删除、不改写规则 issue；模型、检索或引用校验失败时仍保留原规则报告。
- 功能默认关闭，只有显式设置 `ENABLE_ISSUE_EVIDENCE_REVIEW=true` 并配置所需后端服务时才运行。

## 可验证哈希

- Full execution：`790bdef30c16531199d7a9890edfab75c9c553b63dc0c53fe6279acc14668dca`
- Local predictions：`249b69e9a8dfd66c3b75f2844ffaa177925beb389357727e16fd36e03fa313fc`
- RAG predictions：`21b483adf6e997bbf8533ecdba5f3b18ffc437856034dde6a6e80a604325f83b`
- Local report：`37ddd7a777866c91bcac745639f3301143d5a962c2596151d08ab7177f42ec03`
- RAG report：`1b0b036a420bab555b9f022bfbb7bb26339c045dc8bcfff5d99cad86f5c14a56`
- Comparison：`df568f2f54f27723da342fb78233c785b1a8557d12d7757d2ea2193f7df03e41`

这些本机运行产物位于 Git 忽略的 `artifacts/`，公开仓库只保留本脱敏摘要与冻结评测定义。
