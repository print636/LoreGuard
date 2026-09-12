# Evidence Investigator 真实 Provider E2E 评测

这套 runner 通过 LoreGuard 的公开 HTTP API 创建隔离项目、上传冻结文档、启动异步分析、轮询终态并在服务返回后本地判分。它不会直接调用应用内部函数，也不会把 manifest 中的 `expected`、候选答案、诱饵答案或难度说明发送给模型。

## 运行前提

- PostgreSQL、Redis、API 与 Celery worker 已启动，并指向同一套服务配置。
- `ENABLE_EVIDENCE_INVESTIGATOR=true`，真实模型凭据已通过服务端环境变量配置。为隔离被测能力，应关闭通用模型抽取与 issue evidence review；否则 runner 会拒绝把其它模型阶段先发现的问题归功于调查器。
- 若要求真实混合检索，还需配置 embedding Provider，并保持调查器的 hybrid 要求开启。
- 先固定模型、Prompt、RAG、预算和阈值；holdout 运行结果不得用于调参。
- runner 会产生真实模型调用与费用，因此没有 `--confirm-live-provider` 时必定拒绝执行。

先运行 dev 接线验证：

```powershell
python scripts/run_evidence_investigator_live.py `
  --split dev `
  --base-url http://127.0.0.1:8000 `
  --confirm-live-provider
```

只有在配置冻结后才运行 holdout，并显式指定一个尚不存在的结果文件：

```powershell
python scripts/run_evidence_investigator_live.py `
  --split holdout `
  --base-url http://127.0.0.1:8000 `
  --artifact artifacts/evidence-investigator-live/holdout-frozen-run-01.json `
  --confirm-live-provider
```

所有结果文件都采用“只创建、不覆盖”语义。默认文件名包含 UTC 时间和随机后缀；显式路径若已存在，runner 会退出而不会改写原始结果。每个案例有独立项目，超时后会尽力取消仍在运行的任务。

## 结果边界

产物只保存案例编号、期望类别、分类结果、计数、行号授权判定、耗时、Token 计数、RAG 模式和哈希指纹，不保存故事正文、请求内容、模型原始响应、凭据或服务地址。项目 ID 与 run ID 仅保存单向哈希。

分类包括：

- `positive_added_issue`：规则类别、数量和冻结证据行均命中；
- `active_abstain`：模型通过受控工具协议主动放弃；
- `promotion_rejection`：模型提交了候选，但确定性 promotion 拒绝；
- `provider_failure`：Provider 或调查器降级；
- `timeout_or_budget`：服务或调查器触发超时/预算边界；
- `wrong_candidate`：输出了错误问题、漏掉正例，或证据没有命中冻结授权行；
- `runner_failure`：HTTP/协议/本地执行失败，不得计作模型主动 abstain。

公共诊断接口目前足以提供调查器的 reported prompt/completion、保守 charged token、RAG 策略/模式和 promotion 结果。精确工具调用次数只可在 `loop=completed` 时由“一次 Provider decision 对应一次已执行工具”的协议不变量得到；降级路径不会猜测该值。服务也刻意不公开有效 Provider 身份与密钥配置，因此 `safe_configuration_fingerprint` 只覆盖 runner 参数、冻结哈希与 `/health` 的脱敏能力观察，不应被解释为上游部署身份的证明。

这是一套小型、开发者编写的 E2E 夹具，不是公开盲测基准。任何指标都必须来自保留的真实运行产物，不能从 mock 测试或结构校验推断。
