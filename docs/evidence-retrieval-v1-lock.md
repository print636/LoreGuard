# Evidence Retrieval v1 配置锁定记录

本记录在首次运行 holdout 前提交，用于约束 LoreGuard 的检索评测口径。数据集为开发者可见的原创冻结小样本，结果不得外推到开放文本或生产流量。

## 固定运行身份

- 数据集：`evidence-retrieval-v1`
- Manifest SHA-256：`e7f7100116c52e55585a55258914bb22b26b97711e35c0a50d3dae4a8ebbbdca`
- Freeze SHA-256：`3b9b14df06b427318acec0a01d9f30e6225936ed3792799fcd63f3eae1e25e9d`
- Embedding：`BAAI/bge-small-zh-v1.5`
- 模型 revision：`7999e1d3359715c523056ef9478215996d62a620`
- 维度：512，归一化向量，PostgreSQL/pgvector 精确余弦检索
- Chunker：target 120、min 80、max 160、overlap 24
- RRF：`rrf_k=60`，branch limit 30
- 返回数量：Top-5
- 热运行重复次数：3

## Dev 选择结果

在 16 个 dev 查询、29 条期望证据上，三种策略使用同一提交、同一语料、同一 chunker 与同一模型 profile：

| 策略 | Evidence Recall@5 | All-evidence@5 | Low-lexical Recall@5 | MRR | 热检索 P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Keyword | 89.66% | 81.25% | 84.62% | 0.8073 | 1.141 ms |
| Dense | 79.31% | 62.50% | 76.92% | 0.8958 | 209.924 ms |
| Keyword + Dense RRF | **93.10%** | **87.50%** | **92.31%** | **0.8875** | 204.380 ms |

混合检索通过全部预设绝对门槛；相对关键词基线的总体 Recall 提升为 3.45 个百分点，未达到预设的 5 个百分点。因此只允许表述为“对比三种策略后选择混合检索，改善完整证据覆盖和低词面召回”，不得表述为“总体召回显著提升”。

同时观测到：隔离泄漏、失败、降级和三次热运行排名不稳定均为 0；冷索引 16 个 chunk，用时 2.888 秒；复用阶段新增 embedding 和 Provider 调用均为 0。

## Holdout 纪律

提交本记录后，使用上面的固定配置只运行一次 28 题 holdout。无论结果是否通过，都保留原始哈希报告并公开真实结果；不得根据 holdout 修改 chunker、RRF、Top-K、模型或门槛后重跑并覆盖结果。
