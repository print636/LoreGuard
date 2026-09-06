# 受限证据修复 Agent（第一阶段）

状态：代码已接入，默认关闭。2026-09-06 已完成 v1 的 3 任务真实 Provider 开发 pilot；pilot 暴露出 benchmark 错标后建立独立 v2。commit `bcbfab8` 上的 v2 `full × 3` 只在 Agent 阶段调用真实 Provider，主抽取候选由冻结 manifest 合成注入；全部 90 次要求执行已经完成，其中 holdout 为 81 次，但完整 gate 为 `passed=false`，且没有直接 `ABSTAIN` 成功路径。因此它既不能写成默认交付能力或已证明收益，也不能作为端到端真实模型抽取评测。

## 为什么这是 Agent，而不是把固定流程改名

LangGraph 1.2.11 `StateGraph` 只负责编排 `decide -> execute -> decide/finalize` 循环。Agent 行为来自模型在每轮根据服务端校验原因和工具结果动态选择以下应用层 JSON 动作：

- `READ_SPAN`：一次动作可批量读取多个候选附近的有界原文；每个 request 仍独立绑定候选和冻结文档；
- `PATCH_RECORDS`：用先前读取返回的服务端 `span_id` 提交字段补丁；
- `ABSTAIN`：证据不足时明确弃答。

当前 Provider 未验证原生 `tool_calls`，所以这里准确称为“应用层 JSON tool protocol”，不冒充原生 function calling。第一轮提示包含模型候选的核心字段，但不含 `READ_SPAN` 证据正文或文档全文；模型必须先调用 `READ_SPAN`，取得绑定候选、文档、行范围和内容哈希的不可猜测 `span_id`，下一轮才能提交补丁。这使工具结果真实参与后续决策。

## 信任边界

只有已通过响应 envelope、批量 `doc_ref` 映射、服务端 role/scope 归属、Pydantic 核心 schema、证据行范围和非空证据检查，但仍失败于 `lexical_support` 的候选会进入 Agent。仅缺失或写错语义标签的候选继续使用固定 semantic repair pass，不能记作 Agent 经验。

v2 冻结 Agent manifest 的 30 个初始候选现已逐条通过同一生产准入路径验证：结构、文档归属、行号、非空证据和语义标签均合法，并且只在 `lexical_support` 失败。提问、引用、假设和梦境可做同一语义对象的唯一最小词面修正，但 Agent 不能修改语义标签；服务端可按证据做保守归一，且最终记录仍隔离于确定性规则。整条记录换身或存在多个合法 fingerprint 时必须弃答。Oracle、允许证据行和期望补丁仍只供运行后评分，不进入 Agent 输入。

`PATCH_RECORDS` 不能更改 `kind`、`doc_ref`、证据行号、role 或 scope。每个补丁必须携带本次 Agent、同一候选、同一文档且覆盖原证据行的有效 `span_id`，然后重新通过：

1. kind 对应的字段 allowlist；
2. Pydantic `ExtractionRecord`；
3. 原文证据范围与 `lexical_support`；
4. `semantic_quality` 和确定性规则资格重算。

一个 `PATCH_RECORDS` 动作内任意记录失败时全部不提交。伪造、跨运行复用、跨候选复用或跨文档使用 `span_id` 均失败关闭。

## 有界执行和审计

- 整个 analysis run（不是每个文档或每次修复）共享最多 2 个模型决策轮、总计 6 次工具动作；批量 `READ_SPAN` 按一个工具动作计数，内部读取数另记为 `span_read_count`，后续文档不能通过重新创建 Agent 重置额度；
- 单个 READ 动作的 request 数、整次运行的 span 数、单次读取行数、候选附近半径和全部 span 字符数均有硬上限；批量请求中一项越界或跨文档会在读取前原子拒绝；`evidence_range` 预检拒绝只额外记录候选/文档标识、模型尝试的起止行和服务端允许窗口，不生成 span、不保存正文，也不能计作成功读取；
- 同时受运行剩余 Token、Agent Token、Provider 超时、总 deadline 和取消检查点约束；共享 Agent deadline 在第一次 Agent 调用时启动，主抽取耗时不提前消耗它；
- Provider 已返回后才到达的取消仍会先写入 `ModelEnhancedExtractor` 的内容无关调用/Token 安全账本，再传播取消；service 在取得终态所有权后将已完成 Agent 调用的已知用量和安全遥测原子持久化，并显式标记为 `lower_bound` / `review_agent_completed_calls_only`，不能冒充完整运行用量；其中 `charged_tokens` 另标为保守的内部预算扣减，不冒充 Provider 实际用量；取消后不会继续执行工具，终态重投也不会重复记账；
- 未知工具、非法 JSON、重复循环、越界、跨文档、超预算、超时和补丁重验失败都不会产生确定性记录；
- `invalid_records = recovered_invalid_records + unresolved_invalid_records` 仍是最终诊断约束。

持久化 trace 只包含动作、轮次、候选哈希、服务端 `doc_ref`、行号、span 哈希、补丁字段名、校验原因、Token、耗时和最终状态。不保存思维链、Prompt、模型原始响应、故事/工具正文、补丁值、密钥或 Provider endpoint。取消态安全账本还会无条件丢弃 Provider 控制的原始 request ID，并对聚合 Token、调用次数及逐调用字符/耗时计数应用配置派生的有限上限；聚合计数越界时整份账本不可用，但不能阻止取消终态落库，只有逐调用可选遥测越界时则将该序列标为不可用。诊断最多保留 8 次 Agent run、每次 64 个 trace 事件；发生截断时同时输出原始总数和显式 `*_truncated=true`，不能把截断后的轨迹表述为完整轨迹。

## 2026-09-06 pilot 后的评测修正

v1 的三个开发任务曾用 outcome-first scorer 在 `thinking=disabled` 和 `thinking=enabled` 下各运行一次。disabled 组运行时成功 2/3，enabled 组运行时成功 3/3，但两组质量 gate 均失败。复核发现：

- 两个旧“不可恢复”任务实际都能通过同一语义对象的唯一、最小词面修正通过生产 oracle gate；
- 旧“可恢复”任务需要同时替换时间、地点和人物，实质是把候选换成另一个事件，应该弃答；
- disabled 组有一项 Provider 运行失败；安全降级不能计为语义弃答。enabled 组的 3/3 仅表示运行完成，不表示质量通过。

因此保留 v1 作为逐字不变的历史基线，新建 benchmark `agent-acceptance-v2` / `post-pilot-semantic-relabel-v2-final-pre-real`，真实报告 schema 为 `real-agent-acceptance-report-v2`。该 revision 在首次 v2 Agent 阶段真实 Provider full 运行前完成最终审计冻结：以一个“单条候选混入两个均未回答问题、存在两个合法 fingerprint”的弃答案例替换旧的等价负例；任务总数、persona 分布和 development/tuned 划分未变。它仍是开发者可见、非盲测套件：

- 可恢复任务仍必须实际 `READ_SPAN -> PATCH_RECORDS`，补丁被完整生产门接受，最终 directive 指纹匹配 scorer 事后计算的 oracle 指纹，同时运行期仅在内存捕获的补丁 fields 稳定哈希必须匹配 expected patch 哈希；报告不保存补丁值；模型提交的 fields 还必须只包含相对候选确实变化的最小字段，服务端/scorer 不会静默剥离重复的未变字段来替模型制造通过结果；
- `allowed_evidence` 是运行后 oracle 目标证据，不是生产读取授权范围。生产授权由候选附近的服务端 `read_window` 单独重建和审计；一个已授权的同文档 READ 只要完整包含至少一个对应 oracle 目标 span 就算证据命中，目标 span 不需要反向包含整个读取窗口。读错文档、没有覆盖任何目标或越过服务端窗口仍失败，其中只有服务端授权边界被已接受 READ 突破才计安全违规，单纯未命中 oracle 只计评测偏离；
- 不可恢复任务只有在模型实际执行 `ABSTAIN`、`final_reason=explicit_abstain`、运行成功且零 recovered directive 时才算语义弃答。manifest 中的 `expected_action_path` 仅是协议覆盖参考，不固定模型必须直接弃答还是读取后弃答；Provider/协议错误、超时、deadline 或预算终止只是安全降级，必须计为质量失败；坏补丁被拒后的服务端 `FINALIZE` 只算 `safe_containment_success`，同样不能计作语义弃答；
- 实际路径仍单独报告；27 项 holdout 自身必须覆盖 3 种 runtime-success 语义路径，分别是直接 `ABSTAIN`、`READ_SPAN -> ABSTAIN`、`READ_SPAN -> accepted PATCH_RECORDS`。Provider/协议失败与坏补丁后的服务端终止只计入 runtime failure 或 containment 指标，不能贡献动态性；
- 本次 pilot 的 3 个任务固定标为 `development_tuned`。full 模式必须将其余 27 个任务作为 holdout 单独报告恢复率、弃答率、安全、trace、遥测和覆盖 gate。旧 pilot 已影响通用 prompt，不能外推成独立 holdout 成绩；
- 真实 runner 在报告中明确记录 `semantic_abstain_rate`、runtime failure 数量/分类、`bad_patch_proposal_rate`、坏补丁诊断原因、`safe_containment_success`、Provider/Agent 配置超时覆盖和 `read_timeout` 次数，但不保存 endpoint、密钥、prompt、正文或响应。无论 rejected PATCH 的原因为通用校验失败、`semantic_promotion` 还是 `semantic_field_forbidden`，都属于坏补丁；生产接受但事后 oracle surface hash/最终指纹不符的 PATCH，以及不可恢复题上的 accepted PATCH，也属于坏补丁。只有被服务端拒绝并随后 `FINALIZE` 的路径可算安全 containment；错误补丁已被接受时不能获得 containment 成绩，任何坏补丁也不能被随后弃答洗成质量成功。Mock/scripted harness 必须在 `real_provider_connected` gate 失败，不能冒充真实 Provider 质量报告。

通用 Agent prompt 同步收紧为 lexical-only：候选语义标签在进入 Agent 前已经合法；若候选字段和 validator reason 已足以证明无法安全修复，可以直接弃答，否则第一轮读取，第二轮必须按 kind 逐项核对核心字段，使用同一 span 的最小原文词组修正或弃答。PATCH 的 fields 只能列值实际变化的最小字段，禁止把未变字段原样重复提交；生产 validator 不替模型静默剥离 no-op 字段，运行后 surface hash 继续严格暴露这种非最小行为。每个候选同时披露服务端计算的 `document_line_count` 和闭区间 `read_window`，`READ_SPAN` 必须完全落在这个窗口，防止模型猜测不存在的行号或利用“只与邻域相交”的宽区间越界读取。仅修改 `modality`、`source_scope`、`certainty` 等标签不能修复 `lexical_support`。这不是针对任务 ID 或 oracle 的提示，也没有放宽服务端校验。

协议失败仍按失败处理，不会从畸形响应中猜测或打捞动作。为定位通用模型兼容性问题，持久化 trace 只新增内容无关的枚举诊断：失败阶段、根形状、截断后的动作数量、白名单动作名，以及经过字段路径与 Pydantic 错误类型白名单归一化的 schema 错误；不保存响应、任意字段名或字段值。

## 启用方式与下一门槛

开发环境显式设置 `ENABLE_REVIEW_AGENT=true` 才会启用。真实 runner 还要求显式 `--execute`，默认选择 v2；历史复现必须显式 `--benchmark-version v1`。pilot 为 3 个 development/tuned 任务各 1 次，full 强制 30 个任务各 3 次。runner 在主抽取阶段合成注入冻结候选，只有 Agent 阶段调用真实 Provider。诊断慢速 Provider 时可显式传入 `--agent-timeout-seconds 30 --agent-total-deadline-seconds 60`；省略参数就继续使用部署配置，runner 不会静默改变产品默认的 15/30 秒边界，报告会记录实际生效值。启用或完成执行不代表达到简历表述门槛；本次 v2 full 的 holdout 运行成功 74/81，恢复 26/51、主动弃答 24/30，且缺少直接 `ABSTAIN` 成功路径，完整 gate 失败。详见 [2026-09-06 pilot 记录](review-agent-pilot-20260906.md) 与 [v2 full 脱敏 checkpoint](review-agent-v2-full-checkpoint-20260906.md)。
