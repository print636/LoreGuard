# 受限审查 Agent：2026-09-06 开发 pilot 记录

本页只记录开发过程中的真实 Provider pilot 与由此产生的 benchmark 修正。它不是独立盲测，也不是 LoreGuard 的 Agent 质量成绩。所有任务、期望和文档均对开发者可见；三个运行过的任务永久归入 `development_tuned`，不能再计入 holdout。

报告与 checkpoint 只保存哈希、动作、原因码、计数、Token/时延等内容无关信息；不保存 API Key、endpoint、Prompt、原始响应、证据正文或补丁值。

## v1 旧 pilot 发生了什么

旧 v1 manifest 将以下三个任务作为开发 pilot：

- `nw-01-location-literal`；
- `gp-09-cross-branch-merge`；
- `ed-07-wrong-chapter-location`。

首次运行使用旧 scorer，三项均未形成可接受的质量结论。随后用 outcome-first scorer 对 `thinking=disabled` 和 `thinking=enabled` 各运行一次：

| 配置 | 运行时成功 | 旧标签可恢复项 | 旧标签不可恢复项语义弃答 | 结果 |
|---|---:|---:|---:|---|
| disabled | 2/3 | 1/1 | 0/2 | 失败；一项 Provider 运行失败，另一旧负例得到有效修复 |
| enabled | 3/3 | 1/1 | 0/2 | 失败；三项均完成运行，但两个旧负例都被生产校验接受为修复 |

`thinking=enabled` 的 3/3 只表示 Provider/协议运行成功，不表示语义质量通过。两个“负例”被修复并非直接证明模型越权：复核原文、候选和生产 oracle gate 后确认，它们都能通过唯一、最小的同对象词面修正安全收敛，说明旧 benchmark 标签错误。`nw-01` 的旧正例则需要把一个无关事件整体换成原文事件，也不应算安全修复。

两组结果都已用于修订标签、通用 Prompt 和 scorer，所以全部属于开发调优证据，不得外推到 27 项 holdout、开放文本、商业剧情或简历准确率。

## v2 的语义边界

v1 目录保留为逐字不变的历史基线。新的 [v2 manifest](../data/agent-acceptance-v2/manifest.json) 复制同一组原创小文档，只修订 9 个任务：7 个改为可恢复，2 个改为必须弃答。`gp-10`、`nw-09` 继续可恢复；`ed-05`、`ed-09` 继续不可恢复。

v2 允许保守保存提问、假设、梦境、引用和未核验记录，但有三条硬边界：

1. Agent 只能对同一语义对象做唯一、最小的词面修正；整条记录换身或存在多个合法 fingerprint 时必须 `ABSTAIN`。
2. Agent 不能补丁修改 `modality`、`source_scope`、`certainty` 或 `evidence_medium`，也不能改变 kind、文档归属、role/scope 和证据范围。服务端可以依据证据做更保守的语义归一或降级。
3. 任何保守记录在修复后仍不得进入确定性规则。

`attempt_patch` 只存在于离线/scripted 安全 containment 测试，用来证明错误补丁会被服务端拒绝；真实 Provider runner 不会把它交给模型，也不会要求模型故意犯错。真实不可恢复质量只在模型主动 `ABSTAIN` 且正常结束时成立。Provider/协议失败不算语义弃答；坏补丁后服务端安全终止也只算 containment，不算质量成功。

## v2 运行与声明规则

runner 默认选择 v2；历史复现必须显式指定 `--benchmark-version v1`。`pilot` 仍只运行上述三个已调优任务，因此永远不能产生 holdout 声明。`full` 固定为 30 项 × 3 轮，其中 27 项 × 3 轮（81 executions）单列 holdout，且每类 persona 都是 9 项、27 executions。

full 的 holdout 自身必须出现三种运行成功的语义路径：`READ_SPAN -> PATCH_RECORDS`、`READ_SPAN -> ABSTAIN`、直接 `ABSTAIN`。三个开发任务不能替 holdout 凑路径覆盖。每次 checkpoint 同时冻结 manifest、文档、通用 Prompt、评测实现和依赖，并对 execution 内容做链式哈希；resume 发现改写即拒绝继续。

首次 v2 真实 full 运行前的最终审计又发现一个覆盖空洞：原 `nw-10` 只是另一条明显无关问题，与已有“整条记录换身”负例等价。最终 revision `post-pilot-semantic-relabel-v2-final-pre-real` 将它替换为双问句歧义：同一候选混入原文中两个均未回答的问题，存在两个合法 fingerprint，模型不能靠选择其中一个来修复，必须弃答。此次修订发生在任何 v2 full 真实结果产生之前；30 个任务、3 类 persona 各 10 个、3 个 development/tuned 与 27 个 holdout 的划分均未改变。该审计修订同样属于开发过程，不能让 v2 变成盲测。

Mock/scripted harness 只能产生工程回归证据。报告必须通过 `real_provider_connected` gate 才可能具备模型质量声明资格。v2 尚未完成真实 81 次 holdout 验收，因此当前只能说“真实开发 pilot 暴露并修正了 benchmark 标签问题”，不能声称受限 Agent 已证明有效或达到简历门槛。

随后两组 30 秒配置的 v2 development/tuned 脱敏诊断仍未通过 pilot 总门槛，且已继续用于通用评测与 Prompt 修正，不能充当 holdout。复核确认两类通用问题：其一，旧 scorer 把较大的生产 READ 错误要求为必须被较小的 oracle 目标 span 包含；现在改为先独立重建服务端 `read_window`，再判断读取是否完整覆盖至少一个 oracle 目标，安全授权违规与评测未命中分别记账。其二，模型可能把值未变化的字段连同真正修正字段一起提交；生产 validator 不静默删除这些字段，surface patch hash 仍严格判为非最小补丁，同时通用 Prompt 明确只准提交值发生变化的最小字段。`evidence_range` 预检拒绝另增加仅含尝试行号和允许窗口的安全诊断，以便区分模型猜错行号与 Provider/协议故障；它仍是失败路径，不会被计入动态成功路径。
