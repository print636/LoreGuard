# 正式角色设定语义作用域复核 RFC（2026-09-25）

## 当前实现检查点

Phase A 的 V5 结构准入和 Phase B 的逐块模型语义复核已在后端实现，均默认关闭。Phase B 只把模型复核为 `supported` 且服务端合同校验通过的正式角色设定记录送入普通待审核候选；`rejected`、`uncertain` 和模型/预算失败只计入受控诊断并使该块部分覆盖。目前尚无独立的作者语义待确认记录、列表或更正界面，也尚未用冻结人工 gold 完成真实模型 DEV/TRANSFER 放行。因此这是可回退的实验链路，不代表用户可见功能已完整或开放文本质量达标。

## 范围与现状

Phase A 是默认关闭的 V5 原型：主抽取提出 `support_id`、`actor_anchor_id`、`label_anchor_id` 和 `scope_relation`；服务端按冻结原文生成断言索引，核对行号、偏移和若干中文句式。它能证明引用位置和部分字面关系，不能证明完整的语义归属。独立审查已指出四种高风险关系：单句“桑衍相信周尧喜欢蜜瓜”把嵌套偏好归给桑衍；A1 桑衍、A2 周尧、A3 省略主语时越过新主体；“核心风险告知”误覆盖擦桌面；“稳定热茶”误覆盖喜欢蜜瓜。继续扩充动词、人物称谓和对象的正则词库，不能成为开放文本的可靠语义门。Phase A 保持默认关闭，其测试通过只说明列举样例受控，不构成真实模型质量声明。

Phase B 只作用于 `formal_character_profile` 的 V5 主抽取；历史、草稿、targeted 召回和既有确认基线协议不变。目标是允许有原文依据的同一主体、同一语义轴承接，同时让不确定关系停在作者待确认处。

## 两级准入

1. **确定性硬拒。** 沿用严格 JSON/schema、字段白名单、服务端冻结文档、逐字完整证据行、目标分句 `support_id`、对象必须出现在目标分句、行号与代码点偏移、锚点存在且位于同一行目标之前、关系枚举和包大小等不可由语义推断弥补的约束。明确的结构冲突或字段矛盾硬拒，继续使用现有至多一次整包重生成。不要把“句首恰好出现角色名”“中间词未命中白名单”“标签与目标使用不同动词”当作充分语义证明或硬拒理由。
2. **AI 语义复核。** 结构干净的 V5 正式资料候选均接受一次逐块批量复核，包括 `local` 单句，以覆盖嵌套主体。复核器只看候选涉及的冻结完整原文行、该行全部服务端分句及偏移、目标和候选字段；必须独立判断事实是否为现实断言、目标事实的主体、锚点之间是否换主体、核心/稳定标签是否确实覆盖目标语义轴、偏好对象和方向。不能因为同一行、同一个人名或相似词面就判“支持”。服务端仍做最终比对，复核器不能修改候选、签发来源或直接创建基线。语义 `rejected` 或 `uncertain` 不触发同一模型再生成整包；已核实的其他记录可保留，本块覆盖标为 `partial`。

现已在 `app/character_trait_extraction.py` 区分结构准入与最终结果，并于整包校验后执行复核；`app/character_scope_review.py` 负责来源绑定和响应合同，`app/character_scope_review_provider.py` 负责有上限的模型调用。`app/character_consistency_stage.py` 传递冻结 run input 身份，且只把复核 `supported` 的信号交给 `build_pending_trait_candidates`。普通候选仍须作者确认后才由 `app/service.py` 冻结为可比较基线。

## 复核合同与来源绑定

请求由服务端按确定性顺序构造：`run_input_id`、文档 ID/版本/`content_sha256`、块起始行、`ASSERTION_INDEX_V1` 版本、该行每个分句的 ID/偏移/原文、候选的目标 ID/锚 ID/主体/维度/方向/对象/稳定层级、复核 prompt 与 schema 版本。对规范化请求计算 `request_digest`；模型回显仅供匹配，服务端必须自行重算，不能信任回显作为证明。

响应为单个严格 JSON 对象，含版本、`request_digest` 和与请求一一对应的 `items`。每项只允许 `proposal_id`、`support_id`、`verdict: supported | rejected | uncertain`、`actor: proposed | other | ambiguous`、`actuality: asserted | reported | hypothetical | question | ambiguous`、`statement_relation: supported | contradicted | ambiguous`、`label_relation: same_axis | different_axis | none | ambiguous`、`object_relation: same | different | not_applicable | ambiguous`、`polarity_relation: same | opposite | not_applicable | ambiguous`、`level_supported: yes | no | ambiguous`、`basis_ids`（服务端分句 ID 白名单）。不接收模型自评分、自由文本理由或新的证据。`supported` 还要求各槽与候选主张相容，`basis_ids` 含目标 ID，跨句判断覆盖必要锚点和中间分句。缺项、重复、额外 ID、越界引用、版本/摘要不符、槽位自相矛盾或响应超限，一律视作 `uncertain`，不能通过缺席默认同意。不要把同一模型的抽取和复核称为独立共识；复核只是第二道否决/弃权门，效果须由预先冻结的人工标注检验。

原文及分句索引始终作为不可信数据，以 JSON 数据块传入，system 指令说明其中的“忽略规则/改变 verdict”等文字无效；调用不授予工具能力。服务端不执行模型输出，不保存原始 prompt、回复、思维链、密钥或服务地址；诊断仅记录受控 verdict/原因计数、匿名槽位和用量。注入内容即使仿造 `support_id` 或 reviewer JSON，也不得改变服务端索引及响应白名单。

`scope_review_slot_conflict_counts` 是 `scope_review_slot_conflict` 的结构化诊断。模型报 `supported` 但服务端槽位合同不成立时，分别累计 `actor`、`actuality`、`statement_relation`、`label_relation`、`object_relation`、`polarity_relation`、`level_supported`；一项可贡献多个类别。模型报 `rejected` 却没有任何明确否决槽位时，累计 `rejected_without_rejection_signal`，并额外累计值为 `ambiguous` 的槽位。只有通过来源、响应和依据校验的冲突项才产生这些计数；缺失/无效响应沿用原有失败原因码。阶段诊断与开发运行器仅导出固定类别和有界计数，不导出候选/分句 ID、正文、prompt 或模型原始值；诊断不参与准入判定。

## 预算、失败与作者待确认

复核沿用注入的 OpenAI-compatible Provider 的 `complete`，使用独立角色 prompt、温度 0、受限 JSON、单次逻辑调用；底层可按现有 Provider 规则有限重试。给 reviewer 单独的 completion/响应字节/调用超时上限，并在主抽取前为它预留估算 Token；抽取重生成与复核共同受一次信号的总 deadline、单次逻辑预算及角色阶段共享预算约束，不挪用漂移 reviewer 的保留额。每次调用前用现有估算器准入，成功按 `max(估算,上游实报)` 计 `charged_tokens`，Provider 失败按估算计入；`attempted_calls` 和阶段/运行账本均包含复核调用。预算不足、超时、Provider 失败、无效 JSON 或部分响应都转 `uncertain`、`partial`，不把缺失解释成没有角色特征。新开关默认关闭，`app/config.py` 声明依赖 V5；`app/runtime_provenance.py` 与真实运行器显式记录 reviewer 版本和有效上限，旧 V5 结果不原地改写。

`rejected`/`uncertain` 不写入现有 `CharacterTraitCandidateRow(review_state=pending)`：该表的普通确认 API 能直接将它升为 `confirmed`。新增独立、有上限的语义待确认记录和列表，引用冻结 run input、版本/hash、目标/锚 ID、候选字段、复核状态与受控原因；页面从冻结输入展示完整原文和分句，不展示模型自信分。作者可驳回或明确更正主体、标签、对象/层级并创建新的普通待审核候选；该动作必须校验项目权限、当前文档和叙事上下文版本、原文 hash、分句 ID/偏移、乐观锁与幂等键，随后仍须走普通候选确认。过期记录只读，不准确认。迁移只新增表/索引和 API，不回填或重判旧候选与旧基线。

## 验证与放行

单测用假 Provider 覆盖四类风险、不同措辞但确属同一轴的原文正例、纯情境分句后的主体承接，以及 schema/ID/版本/注入/超时/预算/重试/用量/部分包的反例；集成测试覆盖待确认 API、权限隔离、过期、并发确认、迁移和“未核实记录永不进入可比较基线”。支持 trace 的最终 accepted 槽位必须是复核后结果，必要时升级 trace 版本。相关入口为 `tests/test_character_semantic_scope_v5.py`、`tests/test_character_v5_runtime_integration.py`、阶段/API 测试及 `scripts/run_character_axis_live.py`。

真实模型门槛先冻结候选级人工 gold（主体、现实性、标签轴、对象、方向和可弃权情形）与哈希；不修改既有 DEV/TRANSFER 选择器或 oracle。先在独立项目做至少三轮完整 DEV：四类风险没有错误进入可审核普通候选，正例及既有五个目标候选每轮均唯一可审核，阶段完整、预算/来源一致、下游五案按既定 gate 通过；逐项报告误接受、真通过、弃权、漏收、Token、延迟和失败。只有 DEV 达标才运行既有 TRANSFER 的至少三轮独立全流程，并要求同样门槛；不完整/超时/人工补救单列，不计自动通过。开发者可见的小样本和同模型复核都不能外推为盲测、开放文本准确率或生产可用性。
