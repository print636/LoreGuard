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

2026-09-04 起，以 `model_execution` 中的结构化记录为唯一新运行判据：启用且已配置，分块总数大于 0，全部分块均已调用并成功，失败/跳过/无效记录均为 0，且无降级原因码。只看 `model_used` 或扫描 warning 都不足以证明完整成功。未知执行状态排除出严格分母。旧报告离线重算只保留原有明确标记并注明 `legacy_stored_flags_unverified`，不补造缺失计数；下列历史成绩不是新协议的重新认证。

旧判定（仅用于解释历史报告）：

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

## 2026-09-02 正式复测结果

在通用结构化协议、词面证据约束、时间继承和同证据去重修复后，使用配置的 `deepseek-v4-pro` 对完整 14 例执行 3 轮，共 42 个案例轮次：

- 42/42 轮有模型参与且均为完整无降级轮次；171/171 个逻辑 Provider 调用成功。
- 类别级 TP 30、FP 0、FN 0，Precision / Recall / F1 均为 1.000，证据对命中率 1.000。
- 更严格的“类别＋完整证据集合”同样为 TP 30、FP 0、FN 0；42/42 轮严格完全通过，14/14 个案例三轮预测集合一致。
- 单案例轮次 P50 42.60 秒、P95 84.97 秒、最大 111.77 秒；总墙钟时间约 33 分 58 秒。
- Provider 上报 286,941 Token，保守计数 292,916 Token，350,000 Token 预算未耗尽。

原始安全报告保存在本机忽略目录 `artifacts/complex-v3-model-full-r3-postfix-20260902.json`，未提交 Prompt、响应正文、Key 或 endpoint。这个结果证明固定回归链路已稳定达到项目门槛，但数据是开发者编写、开发者可见的回归集，而且修复参考过其前一轮错误；它不能表述为人工盲测、开放故事泛化准确率或生产 SLA。
