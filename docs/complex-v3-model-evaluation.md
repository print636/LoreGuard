# complex-v3 真实模型稳定性评测协议

## 目的与边界

`scripts/run_complex_model_evaluation.py` 用真实 OpenAI-compatible Provider 运行 LoreGuard 的完整“模型增强＋确定性规则”流程。它评测的是开发者编写、开发者可见的 complex-v3 回归集，不是人工盲测，也不能代表开放剧情生产准确率。

评测器不会把 `expected_issues`、类别答案或证据答案传给模型。Pipeline 只接收案例中的原始文档；评分发生在 Pipeline 返回以后。Provider 使用温度 0，但第三方推理服务仍可能产生非确定性，因此必须通过多次 repeat 测量实际稳定性。

## 推荐运行方式

先运行 pilot 检查 Provider 和预算：

```powershell
python scripts/run_complex_model_evaluation.py `
  --suite pilot `
  --repeats 1 `
  --max-total-tokens 25000 `
  --output artifacts/complex-v3-model-pilot.json
```

确认 pilot 没有系统性失败后，再运行 14 案例全量稳定性评测：

```powershell
python scripts/run_complex_model_evaluation.py `
  --suite full `
  --repeats 3 `
  --max-total-tokens 250000 `
  --output artifacts/complex-v3-model-full-r3.json
```

`--max-total-tokens` 是硬性保守预算入口，不是成本预测。运行前应结合具体模型价格和 pilot 实际 Token 重新设置；预算不足时报告会保留已完成轮次并说明停止原因。

## 统计口径

每个案例的每次 repeat 都单独记录：

- 是否有模型结果参与；
- 是否为完整无降级轮次；
- Provider 逻辑调用成功/失败计数；
- prompt、completion、Provider 上报和保守计费 Token；
- 耗时、告警、预测类别与证据行号；
- 类别评分和精确证据评分。

每轮 repeat 汇总尝试案例数、完整模型案例数、Token、耗时及是否完成全部所选案例；全局再汇总总 Token、总耗时和预算超限量。

主指标只纳入“完整模型轮次”：

1. `model_used=true`；
2. 所有被包装的 Provider 逻辑调用成功；
3. 没有分块上限、Token 预算、抽取失败或 Baseline-only 降级告警。

只要有一块模型成功但其他块降级，该案例会进入 `model_participating` 辅助指标，但不会进入 `full_model_only` 主指标。产品流程始终合并确定性 Baseline，因此这里评测的是模型增强产品链路，不是孤立模型能力。

`exact_evidence_metrics` 的 TP 必须同时满足类别相同，且完整的无序 `{文档, 行号}` 证据集合完全一致。类别相同但证据错误会同时计作 1 FP 和 1 FN。原有类别级匹配保留在 `category_metrics`，便于区分“判断方向正确”和“证据可审计”。

## 稳定性

`stability.cases` 按案例汇总：

- 有效完整模型 repeat 数；
- 不同完整预测集合数量；
- 众数预测集合占比；
- 严格证据完全通过率。

全局同时报告全部 repeat 均有效的案例数、严格通过率和平均众数稳定率。预算中断或模型降级不会被当作一次错误预测悄悄混入，而是从严格稳定性分母排除并在 coverage 中显式体现。

## 隐私和可公开性

报告不保存以下内容：

- system/user prompt 正文；
- Provider 原始 response；
- API Key；
- Provider endpoint。

报告只保留数据集哈希、统计量、预测类别、证据文档/行号、确定性规则生成的问题标题和元数据、告警以及异常类型。`artifacts/` 默认不进入公开仓库；对外发布时仍应先执行密钥扫描并人工检查报告。
