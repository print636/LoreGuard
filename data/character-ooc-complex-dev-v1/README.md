# 镜湖闸门：角色连续性 DEV 回归

这组原创中文剧情由开发者编写且答案可见，只验证模拟模型记录经过生产抽取、证据和漂移合同后的行为。它不是人工盲标，也不能衡量真实模型或开放剧情准确率。测试不调用真实模型；Pytest 将数据库隔离在临时 SQLite 中。

`profiles.md` 是已确认的人物设定，`history.md` 是已发布旧章，`draft.md` 是待审新章。每个文件一行一项叙事事实，行号是回归合同的一部分。

| 案例 | 必须区分的事实 |
| --- | --- |
| `growth_after_flood` | 已发生的伤人事故、公开复盘和训练，解释闻岚后来当众反对师父 |
| `scripted_quote_other_actor` | 洛原念出的戏本台词不是其行动；乔唯才实际代签 |
| `ambiguous_pronoun` | 余霁与乔唯同句出现时，“她”没有安全的唯一前指 |
| `unique_pronoun` | 只有余霁在场的相邻两行，可以完整引用后句行动 |
| `medical_food_exception` | 本人真实医嘱和履行，只解释限期拒食；乔唯的真实医嘱不能借给祁棠 |

本套件暂未纳入确认特征的来源生命周期。来源退役后仍生效、作者显式撤销后不进入后续运行的合同，已有独立回归 `tests/test_character_api_contract.py::test_confirmed_trait_survives_source_change_until_explicit_withdrawal` 覆盖；这不是本套件的通过项。

引语误归属与双候选代词误归属曾是已复现缺口，回归现在要求误归因记录不被接受、拒收诊断非空；若旧行为复现，断言会报告被接受的角色、陈述、行动类型和证据行号。同句真实行动及唯一相邻代词各有真正通过的正控。旧 DEV、transfer 与未来 holdout 材料不从这里导入答案。
