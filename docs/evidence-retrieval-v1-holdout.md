# Evidence Retrieval v1 Holdout 结果

本结果来自配置锁定后的首次、唯一一次 holdout 运行。运行使用 28 个查询、55 条期望证据；原始预测和报告保留在本机 `artifacts/`，不将全文语料或逐题答案写入公开摘要。

## 固定条件

- 代码提交：`08797e70765e812ab7f1c93bc26ba2a77de8de27`
- Prepared SHA-256：`b44ae5a52afba2dc4e214874a1eafc826564811eb4dd244fe30763a4157dc251`
- Chunker：target 120、min 80、max 160、overlap 24
- RRF：`rrf_k=60`，branch limit 30，Top-5
- Embedding：固定 revision 的 `BAAI/bge-small-zh-v1.5`，512 维
- 每个查询执行 3 次热运行；数据库分别隔离，三策略使用同一配置身份

## 结果

| 策略 | Evidence Recall@5 | All-evidence@5 | Low-lexical Recall@5 | MRR | 热检索 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Keyword | 80.00% | 67.86% | 79.31% | 0.7827 | 0.985 ms |
| Dense | 81.82% | 67.86% | 75.86% | 0.8494 | 545.805 ms |
| Keyword + Dense RRF | **81.82%** | **71.43%** | **79.31%** | **0.8452** | 557.476 ms |

混合检索命中 45/55 条期望证据，并使 20/28 个查询在 Top-5 内找齐全部证据。冷索引用时 2.918 秒；复用阶段新增 embedding 数和 Provider 调用数均为 0。三种策略的失败、降级、隔离泄漏和热运行排名不稳定均为 0。

## 门槛结论

总门槛 **未通过**。唯一未通过项是低词面证据召回：23/29（79.31%），低于预设的 80%；其余 Recall、All-evidence、MRR、延迟、缓存复用、稳定性和隔离门槛均通过。

因此不得宣称“holdout 全面达标”或“混合检索显著优于两种基线”。允许的限定表述是：项目完成了真实 embedding、pgvector 精确余弦检索与 RRF 三策略对照；在首次冻结 holdout 上总体 Recall@5 为 81.82%、完整证据命中为 71.43%，且无隔离泄漏或运行失败，同时公开记录低词面召回差一条证据未过门槛。

逐题复核显示，低词面遗漏集中在 4 个多跳任务：跨文档二跳因果、角色获得知识的前提、规则例外，以及旅程节点与时间规则。当前单查询 Top-5 RRF 容易被词面或语义相近的单段证据挤占；尚未实现查询分解和迭代取证。这个结论只用于误差说明，不用于修改本版配置后重跑 holdout。

## 可验证哈希

- Keyword predictions：`309a5db8f44ff88ed265488f5f16cbf60e9bc52aa8bd0f68331589ab8dbb3850`
- Dense predictions：`7b93a77d92d1649e2d74a469daf14ce8fd8add660d7455e142ce98c4e33737b1`
- Hybrid predictions：`0cd89c23f4ad85891d7042603d05fe443c028efc966c33682f1e60b513aaf227`
- Keyword report：`4f00dce1d90b6ffe9783665b02182721284c4ebb312ff9b0d30171ab0254d809`
- Dense report：`8d079d0dd3dd2390ac4c9ae2824f201d103bbe27f3a1cd272129cc3af2d1866c`
- Hybrid report：`1cce80076c8caba84e1cc69fcdef437e88d5868c4028f6302039a9c25c2e8d32`
- Comparison：`275643de4f59864c6d5a315735e30a7b9c6bcf16c6981d3a80bb7c9c2281f7db`
