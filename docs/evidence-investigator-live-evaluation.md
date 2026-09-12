# Evidence Investigator 真实 Provider E2E 评测

这套 runner 通过 LoreGuard 的公开 HTTP API 创建隔离项目、上传冻结文档、启动异步分析、轮询终态并在服务返回后本地判分。它不会直接调用应用内部函数，也不会把 manifest 中的 `expected`、候选答案、诱饵答案或难度说明发送给模型。

## 运行前提

- PostgreSQL、Redis、API 与 Celery worker 已启动，并指向同一套服务配置。
- `ENABLE_EVIDENCE_INVESTIGATOR=true`，真实模型凭据已通过服务端环境变量配置。为隔离被测能力，应关闭通用模型抽取与 issue evidence review；否则 runner 会拒绝把其它模型阶段先发现的问题归功于调查器。
- 若要求真实混合检索，还需配置 embedding Provider，并保持调查器的 hybrid 要求开启。
- API 与 worker 必须设置相同的 `LOREGUARD_BUILD_REVISION`，值为本次评测所用干净工作树的完整 Git commit SHA；修改代码后必须重新构建并更新该值。
- 先固定模型、Prompt、RAG、预算和阈值；holdout 运行结果不得用于调参。
- runner 会产生真实模型调用与费用，因此没有 `--confirm-live-provider` 时必定拒绝执行。

当前部署默认给 Investigator 预留 16,000 个保守计费 Token，单轮 Provider 调用最多等待 30 秒，整个 Investigator 阶段共用 60 秒绝对 deadline。每一轮实际超时取“30 秒、通用 Provider 更严格限制、全局剩余时间”中的最小值，且每轮仍只执行一次 Provider attempt；这组默认值是部署配置说明，不是评测产物对上游模型身份或服务质量的证明。

先运行 dev 接线验证：

```powershell
python scripts/run_evidence_investigator_live.py `
  --split dev `
  --base-url http://127.0.0.1:8000 `
  --confirm-live-provider
```

先取得两次连续、互不重叠且配置指纹相同的合格 dev 产物：

```powershell
python scripts/check_evidence_investigator_dev_pair.py `
  artifacts/evidence-investigator-live/dev-run-01.json `
  artifacts/evidence-investigator-live/dev-run-02.json
```

只有 pair checker 通过后才运行一次 holdout，并显式指定一个尚不存在的结果文件：

```powershell
python scripts/run_evidence_investigator_live.py `
  --split holdout `
  --base-url http://127.0.0.1:8000 `
  --artifact artifacts/evidence-investigator-live/holdout-frozen-run-01.json `
  --confirm-live-provider
```

所有结果文件都采用“只创建、不覆盖”语义。默认文件名包含 UTC 时间和随机后缀；显式路径若已存在，runner 会退出而不会改写原始结果。每个案例有独立项目，超时后会尽力取消仍在运行的任务。

冻结校验始终认证共享 manifest 与 freeze 索引，但 dev 运行只打开并哈希 dev 正文，holdout 运行才打开并哈希 holdout 正文；校验得到的正文 bytes 会直接用于上传，不会再次打开文件。`verify_frozen_dataset(split=None)` 仅用于显式的离线全夹具审计。共享 manifest 仍包含两个 split 的索引与期望元数据，因此这里保证的是“正文 bytes 隔离”，不是 holdout 元数据盲化。任何 holdout 正文篡改都会在建立 HTTP 客户端、创建项目或调用 Provider 前终止 holdout 运行。

## 结果边界

产物只保存案例编号、期望类别、分类结果、计数、行号授权判定、耗时、Token 计数、RAG 模式和哈希指纹，不保存故事正文、请求内容、模型原始响应、凭据或服务地址。项目 ID 与 run ID 仅保存单向哈希。

分类包括：

- `positive_added_issue`：规则类别、数量和冻结证据行均命中；
- `active_abstain`：模型通过受控工具协议主动放弃；
- `promotion_rejection`：模型提交了候选，但确定性 promotion 拒绝；在负例上它可以使最终输出安全并按现有 strict 规则通过，但这只证明系统兜底生效，不等于 Agent 主动判断正确；
- `agent_protocol_failure`：模型违反工具状态机或参数协议，例如重复动作、无效参数或多工具调用；
- `isolation_failure`：主模型抽取或 AI 证据复核未关闭、发生调用，或运行级聊天用量无法与调查器账本对齐；该案例不得归因给 Investigator；
- `provider_failure`：Provider 不可用、拒绝、响应契约失效，或出现未能细分的调查器降级；
- `timeout_or_budget`：服务或调查器触发超时/预算边界；
- `wrong_candidate`：输出了错误问题、漏掉正例，或证据没有命中冻结授权行；
- `runner_failure`：HTTP/协议/本地执行失败，不得计作模型主动 abstain。

产物格式 `evidence-investigator-live-http-v4` 会逐案例保存脱敏的 capability isolation 证明：主模型抽取必须明确为 disabled、unconfigured、unused 且零逻辑/Provider 调用；`ai_evidence_review` 必须按服务分支契约缺席或明确关闭且零调用；运行级 prompt、completion 与 charged token 还必须和 Investigator 账本完全相等。任一条件不满足都会得到 `isolation_failure`，即使最终问题恰好命中也不能通过。v2/v3 产物仅保留作历史记录，不能用于取得 dev pair 资格。

### Dev 晋级门槛

单次 dev 产物必须同时满足：固定的 8 案例结构（5 正例、3 负例）、strict 至少 5/8、正例至少 3/5、负例最终安全 3/3、Agent 主动 abstain 至少 2/3、正常终止 8/8、零执行类失败、零错误新增、8/8 最终输出可观察、能力隔离与 promotion 计数均为 8/8。执行类失败明确为 `timeout_or_budget`、`provider_failure`、`runner_failure`、`isolation_failure` 和 `agent_protocol_failure`；任何一个出现都会阻止晋级。

runtime provenance 还必须证明 API 与全部 worker 案例使用同一服务代码包哈希、Git revision、模型别名、不可逆 endpoint 配置哈希、核心 Investigator 有效上限和 RAG profile/chunker 配置。runner 会用同一算法独立计算本地 `app/**/*.py + requirements.txt` 哈希；API 在任何案例调用前必须匹配，全部 worker 也必须匹配。该哈希不是 OCI image digest，Git revision 也不能证明 relay 背后的真实模型权重。产物不保存 API key、hostname 或完整 base URL。缺少任一 v4 provenance/gate 字段都会 fail closed；两次合格 dev 还必须具有同一 `reproducibility_fingerprint` 且时间不重叠。

pair checker 会按已认证 manifest 复核每个 dev case 的 ID、期望 decision/category 和完整脱敏字段合同，然后从逐案例记录重新计算 summary、provenance gate 与 development gate；不能用手写 summary 掩盖超时或 Provider 失败。该检查只证明产物内部自洽并与当前 evaluator 源码包一致，不提供数字签名或来源真实性证明。

### Summary 指标口径

`passed` / `failed` 和逐案例 `passed` 仍是原有 strict 判分，不会被下面的拆分指标替代。新增指标用于分清“发现了正确问题”“最终没有误报”“Agent 自己选择放弃”“确定性 promotion 拦住候选”和“链路正常结束”这几件不同的事：

- `system_safety_rate = system_safe_cases / case_count`：只有成功读取到 completed 最终问题列表、且其中没有错误新增问题的案例才进入分子；无法观察最终输出的超时、Provider、runner 等失败仍保留在总分母中，不会被当成安全的零输出。正例漏报但没有错误新增可计为系统安全，同时会降低正例召回。
- `erroneous_added_issue_count`：在可观察的最终输出中累计错误新增问题数；负例中的所有问题均为错误新增，正例只允许抵扣一个类别与冻结证据均匹配的目标问题。`final_output_observed_cases` 明示该计数覆盖了多少案例，未完成案例属于未知而不是零错误。
- `capability_isolation_verified` / `capability_isolation_rate`：通过能力隔离证明的案例数及其占全部案例的比例；它只说明结果可归因于 Investigator，不说明结果正确。
- `positive_recall = positive_passed / positive_cases`：strict 通过的正例数除以正例总数。
- `negative_safety_rate = negative_safe_cases / negative_cases`：最终可观察且没有错误新增问题的负例数除以负例总数；未完成或无法观察最终输出的负例不进入分子。
- `agent_active_abstain_rate = negative_active_abstain_cases / negative_cases`：分类为 `active_abstain` 的负例数除以负例总数。`promotion_rejection` 不计入该分子，因为它表示 Agent 已提交候选、随后被确定性层拒绝。
- `promotion_acceptance_rate = promotion_accepted_candidates / promotion_submitted_candidates`：对具有合法 submitted/accepted 诊断的案例汇总后计算；`promotion_accounting_observed_cases` 给出诊断覆盖案例数。没有任何提交时 rate 为 `null`，而不是 0。
- `normal_termination_rate = normally_terminated_cases / case_count`：分类不属于 `timeout_or_budget`、`provider_failure`、`runner_failure`、`isolation_failure` 或 `agent_protocol_failure` 的案例比例。它衡量正常收束，不衡量答案正确性。

因此，负例的 `promotion_rejection` 可能同时表现为 strict 通过、系统安全，但 Agent 主动弃权率不会因此上升。评估 Agent 判断能力时应同时查看正例召回、主动弃权率和 promotion 接受率，不能只引用系统安全率。

公共诊断接口会同时保存安全的 `budget_preflight` 和由其直接证明的有效 token 上限，并将跨案例一致值纳入 `safe_configuration_fingerprint`。这里不会从本地源码或 runner 参数猜测服务端的搜索次数、超时等未公开上限。每案的 `case_wall_latency_ms` 从建项前开始，到文档上传、异步分析、诊断和问题读取全部结束后停止；它是整次评测编排墙钟时间，不是分析接口或模型调用的 P95。

调查器诊断还包括 reported prompt/completion、保守 charged token、RAG 策略/模式和 promotion 结果，以及与 Provider decision 分开统计的已执行工具、检索、读取和可恢复拒绝次数。Token 对账以顶层 `evidence_investigator.usage` 累计账本为准，因此即使 deadline 等边界令 `loop` 诊断缺席也不会丢失已发生调用；loop 内 Token 仅作为旧版诊断兼容回退，且不会和不完整的新账本混用。即使循环后续降级，服务仍能报告成功执行的工具数；旧版 completed-loop 诊断才使用“一次 Provider decision 对应一次已执行工具”的协议不变量。服务只公开评测所需的脱敏 Provider 配置身份，不能据此证明 relay 背后的实际模型权重或供应商部署。

这是一套小型、开发者编写的 E2E 夹具，不是公开盲测基准。任何指标都必须来自保留的真实运行产物，不能从 mock 测试或结构校验推断。
