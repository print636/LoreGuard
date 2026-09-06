# Evidence RAG 与受限审查 Agent 交付计划

更新：2026-09-06。状态：P0 核心正确性边界已实现；P2 的受限 Agent 第一阶段代码已接入且默认关闭。commit `bcbfab8` 上的 v2 `full × 3` 只在 Agent 阶段调用真实 Provider，主抽取候选由冻结 manifest 合成注入，不是端到端真实模型抽取评测；90/90 次要求执行已完成，但完整 gate 为 `passed=false`。P1 的真实 Evidence RAG 与 P4 仍未完成。未完成、未通过或未实测项不能作为已完成项目经历。

增量状态：P0 已覆盖冻结输入、幂等认领、lease/heartbeat、跨进程取消检查点、typed diagnostics 和保守旧运行展示。受限 Agent 现有实现比原 P2 设想更窄，只处理通过核心 schema/证据边界但需要词面支持修复的候选；它不是 Evidence RAG，也不因此宣布项目交付就绪。

## 目标与边界

在现有五类确定性检查之外，增加能主动补查原文、寻找规则例外并明确弃答的审查助手。保持原创叙事研发工具定位，不改成玩家聊天机器人，不在此阶段实现百万字生产系统。

当前架构可以复用 Provider、证据类型、版本管理、规则引擎、SSE、反馈和评测工具，但新增检索/Agent 需要真实接入业务链路，不能只增加依赖或改名。

## 代码审计结论

- `retrieval.py` 的向量是字符 n-gram 哈希，不是学习得到的语义 embedding。
- `pipeline.py` 的 `check_with_candidates` 目前按记录类型兼容性设置 `consumed`，最终仍调用 `detect_issues(directives)` 检查全部记录；该计数不证明候选改变了裁决，也没有检索结果回送模型的闭环。
- 主抽取仍是固定 system/user 的结构化抽取；另有 LangGraph 1.2.11 `StateGraph` 编排的受限修复循环。它使用应用层 JSON 动作，不使用、也未验证 Provider 原生 `tool_calls/tool_call_id`。
- 运行输入快照、原子认领、终态幂等和 worker lease/heartbeat 已补齐并有自动回归；共享配额仍不是多实例原子预留，工程测试通过也不等于高并发高可用。
- 固定 semantic label repair pass 只修补 modality/source_scope/certainty，是非 Agent 路径；不得把其调用或恢复结果计入 Agent 经验。
- 冻结 Agent 套件的 90 次完整 Mock oracle/scorer 只证明实现边界可回归。v1 的 3 项真实 Provider pilot 只是已用于调优的开发证据。v2 Agent 阶段真实 Provider full 已完成，主抽取候选由冻结 manifest 合成注入；holdout 运行成功 74/81，7 次为 `read_timeout`，可恢复题正确 26/51，不可恢复题主动弃答正确 24/30，因此既未通过 Agent 质量验收，也不能评价端到端抽取质量。

RAG 不以向量数据库为定义条件，但必须证明检索内容进入下游模型推理/生成。将已有精确规则保留为独立通路是合理的；把未实际影响输出的短名单标成已消费则需要纠正。

## 实施顺序与项目内验收

### P0：先补状态正确性与可审计性

- 在提交运行时冻结文档版本 ID、内容哈希与模型/配置版本；重试同一输入，重新分析才读取新版本。
- 运行原子认领、终态保护、重复投递幂等；恢复过程不得覆盖已有人工反馈。明确 Provider 调用存在至少一次语义，不能承诺模型费用 exactly-once。
- 在分块/工具边界重新读取取消状态；任务失败后的事务回滚与恢复可测试。
- 用结构化执行状态替换中文 warning 字符串判定，覆盖预算耗尽、分块截断、无效记录与模型故障；兼容旧运行的未知状态，不能推断旧运行未调用模型。
- 将当前 retrieval 的兼容类型计数与真正输入模型的证据消费记录分开。

验收：两个 worker 争抢同一任务、任务重投递、队列等待时更新文档、分析中取消、模型预算中途耗尽均有回归；已完成报告/反馈不被重复执行破坏。

### P1：真实且能解释收益的 Evidence RAG

- 引入独立 `EmbeddingProvider` 接口；先验证中文本地模型与可用 embedding 服务的许可证、资源需求、维度、延迟，锁定实际采用的模型和版本。现有 chat Key 可用不代表 embedding 可用。
- PostgreSQL/pgvector 真实保存文档分块向量，包含项目、文档版本、原文行区间、内容哈希、embedding 模型及维度；提供可重复迁移与索引重建入口。不能依赖 `create_all` 升级既有数据库。
- 关键词、语义向量、实体关系召回融合（优先使用可解释的 rank fusion）；检索必须先限定项目与本次运行版本，不能把其他项目/旧版本原文泄漏给模型。
- 小数据先精确向量检索；HNSW 作为可配置选项，使用真实查询计划和召回/延迟对照证明价值，不为简历强行启用近似索引。
- 把召回的带引用 ID 原文交给下游模型补查；记录 query、命中文档/版本、得分与实际消费的 evidence ID，避免只建立无人使用的向量表。
- 模型与索引不可用时明确降级到现有基线；不能将降级运行统计为 RAG 成功。

验收：冻结至少 40 个中文跨段检索问题（包含低词面重叠、别名、例外与错误版本干扰），比较关键词-only 与混合检索的 Recall@5、MRR、延迟、索引/调用成本。暂定 Recall@5 ≥ 0.80，并要求在低词面重叠子集体现收益；若未体现，不声称向量带来提升。所有租户/版本隔离负例通过。该门槛是内部工程目标，不是岗位要求。

### P2：一个真正有用的受限 Agent + 工具调用

第一阶段代码状态：

- 已使用 LangGraph 1.2.11 `StateGraph` 编排 `decide -> execute -> decide/finalize`。是否构成动态 Agent 取决于模型依据服务端校验结果和读取结果选择动作，不是因为采用了 LangGraph；功能由 `ENABLE_REVIEW_AGENT` 控制且默认关闭。
- 可选动作限定为 `READ_SPAN`、`PATCH_RECORDS`、`ABSTAIN`。当前是应用层 JSON 工具协议，Provider 原生 `tools/tool_calls/tool_call_id` 未验证、未使用，不能称为原生 function calling。
- `READ_SPAN` 只能读取候选附近、同一冻结文档和 scope 的有界行范围；`PATCH_RECORDS` 必须携带服务端签发且绑定候选/文档/行范围的 span，不能修改 kind、文档归属、role/scope 或证据行号；验证失败和 `ABSTAIN` 均不生成确定性记录。
- 整个 analysis run 共享最多 2 个模型决策轮和 6 次工具动作，同时限制 span 数/字符、单次行数、Token、deadline、取消检查点和响应字节数。故事文本和工具结果一律视为不可信数据。
- 安全 trace 仅保留动作、轮次、哈希、行号、字段名、原因码、Token、耗时和终态，并有 Run/trace 截断标记；不保存或展示 Prompt、思维链、原始响应、故事/工具正文、补丁值、Key 或 endpoint。页面将其与固定语义标签 repair 分开显示。

当前验证边界：

- [v2 冻结 manifest](../data/agent-acceptance-v2/manifest.json) 含 30 个开发者可见任务，3 类 persona 各 10 个，每任务重复 3 次，共 90 次 execution；30 个初始候选均由生产校验路径证明只失败于 `lexical_support`，而不是把其他 validator reason 改名伪装。v1 开发 pilot 使用的 3 项已标为 development/tuned，其余 27 项是 full 报告单列的 holdout。可恢复项要求 `READ_SPAN -> accepted PATCH_RECORDS` 与 scorer 事后 oracle 指纹匹配；不可恢复项必须由模型实际 `ABSTAIN`。Provider/协议失败和坏补丁 containment 均不计质量成功。Mock/scripted harness 还必须失败 `real_provider_connected` gate。
- commit `bcbfab8` 的 Agent 阶段真实 Provider full 已记录 90/90 次要求执行，严格正确 59/90。holdout 81 次中 74 次运行成功、7 次 `read_timeout`；恢复 26/51（0.509804），弃答 24/30（0.8）。31 次严格失败分成互不重叠的 12 次被生产校验接受但不符合 oracle 的错误补丁（9 次 `oracle_patch_mismatch`、3 次 `oracle_unrecoverable_patch`）、9 次可恢复题读取后过度弃答、7 次 timeout、3 次 outcome 正确但读取未覆盖 oracle 证据。`trace_replay=false` 仅有后 3 次，原因都是 `read_misses_oracle_evidence`；7 次 timeout trace 均可重放。服务端接受的安全违规为 0。运行成功轨迹中没有直接 `ABSTAIN` 路径，因此要求的三路径覆盖 gate 失败。development/tuned 组为 9/9，恢复与弃答率均为 1.0，但不能替代 holdout。完整报告 `passed=false`，详见 [v2 full checkpoint](review-agent-v2-full-checkpoint-20260906.md)。
- [离线 runner](../scripts/run_agent_acceptance.py) 只接受 trace 文件或 `--mock-oracle`；它不会调用生产 Agent。真实 runner 虽走同一生产 Agent 路径，但 scripted harness 会被 `real_provider_connected` gate 显式拦截。
- 完整实现边界见 [第一阶段说明](review-agent-phase1.md)、[pilot 记录](review-agent-pilot-20260906.md) 与 [v2 full checkpoint](review-agent-v2-full-checkpoint-20260906.md)。v2 Agent 阶段真实 Provider holdout 已执行但未通过；它没有评测主抽取，在新的独立质量结果达标前不得声称 Agent 或端到端质量成绩。

仍待 P2 验收：针对本次暴露的恢复不足、坏补丁和 `read_timeout` 做通用改进，并比较基线、一次检索和受限 Agent 的恢复、误报、漏报、证据正确性、安全弃答、覆盖/降级、动态路径、延迟与 Token 成本。不能针对单个任务硬编码；本次 holdout 结果若用于修改 Prompt 或实现，后续同套件只能算开发回归，新的泛化质量声明需要另行冻结未参与调优的测试集。只有真实完整运行达到既定 P ≥ 0.75、R ≥ 0.60、证据命中率 ≥ 0.85 等门槛，且安全违规为 0，才能讨论收益。真正的 Evidence RAG 仍属于 P1：当前 `READ_SPAN` 是候选附近有界读取，不是 embedding/pgvector 语义召回，也没有搜索工具闭环。当前实现也没有多智能体。

### P3：轻量多智能体实验（可选，不阻塞投递）

仅在单 Agent 已过门槛后，对困难候选增加“审查者 + 独立反证复核者”的限定实验。两者需有实际独立的上下文、工具决策与输出，再由确定性聚合器保留分歧；将两个固定 Prompt 节点并排不能算有价值的多 Agent。

同一测试集对照单 Agent 的误报、漏报、延迟和 Token。只有可复现收益且成本可接受才进入默认功能，否则保留实验开关并如实说明未采用原因。不能以多 Agent 互相同意替代真实证据，也不把同模型角色包装成独立人工专家。

### P4：可部署交付与有限负载验证

- 开放模型调用前提供服务端项目权限/访客隔离、共享原子额度预留与结算、排队和背压、全局紧急停用开关；只读样例可单独开放。
- 两个 worker、并发任务与故障重投递验证不串项目/版本、不重复完成报告、不绕过配额。廉价模拟 Provider 用于负载，少量真实调用用于协议与端到端验证，二者分开报告。
- 声明测试机器、并发数、输入规模、队列等待与执行延迟、错误率，不以小规模压测推导大型 C 端生产容量。
- 更新 Compose CI、英文/中文启动说明、架构图、事实清单、已知限制和演示脚本；公网部署所需账号/预算另行确认，不擅自购买资源。

验收：上述主线从干净环境可复现，页面可操作，诊断可解释；新增模型/RAG 模式有独立报告，不挪用旧 14 例固定规则回归成绩。

## 完成通知规则

P0、P1、P2、P4 的项目内门槛达标后，才通知“项目本身达到当前目标，可以进入体验/学习与简历准备”。多智能体、百万字生产能力、完整商业多租户平台不作为无限延期的条件。公网资源尚未获得时明确区分“代码交付完成”与“线上部署未完成”。

原先约 90% 是旧 Alpha 清单的估算，不再用它表示上述目标已经接近全部完成。后续以完成的门槛和实测证据报告进度，不用新增测试数量自动换算百分比。

## 技术依据

- [LangGraph：固定工作流与动态 Agent 的区别、工具循环](https://docs.langchain.com/oss/python/langgraph/workflows-agents)
- [pgvector：真实向量列、精确检索与 HNSW/IVFFlat](https://github.com/pgvector/pgvector)
