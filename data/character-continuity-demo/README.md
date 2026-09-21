# 「浮光列车」角色连续性演示集

这是一套为 LoreGuard 编写的原创合成材料，用于展示角色档案确认与新稿漂移审查。它不是商业游戏语料，也不代表真实游戏策划团队的验收结果。

## 运行前配置

在未提交的 `.env` 中配置 OpenAI-compatible Provider，并显式启用：

```dotenv
ENABLE_CHARACTER_CONSISTENCY=true
CHARACTER_CONSISTENCY_SENSITIVITY=balanced
```

API Key 只放在服务端 `.env`，不要粘贴到浏览器、文档或提交记录。修改配置后需重启 API；若使用 Celery，也要同时重启 worker。`GET /health` 的 `model.configured` 应为 `true`。

## 两轮体验步骤

所有材料使用 `timeline_key=main`、`branch.path=["main"]`。第一轮只导入前三份材料：

| 文件 | 文档角色 | 发布状态 | 版本 |
|---|---|---|---|
| `01-world-setting.md` | 世界/故事设定（canon） | published | `v1.0` / ordinal `10` |
| `02-character-profiles.md` | 角色设定（character_profile） | published | `v1.0` / ordinal `10` |
| `03-published-history-v1.0.md` | 历史章节（chapter） | published | `v1.0` / ordinal `10` |

1. 启动第一次分析，等待任务完整完成。
2. 打开“角色档案”，逐条核对原文证据；只确认你认可的稳定设定或历史归纳。AI 候选不会自动成为裁决依据。
3. 再导入 `04-draft-event-v1.1.md`，标为 `chapter`、`draft`、`v1.1` / ordinal `11`，作用域仍为主时间线与主分支。
4. 启动新的分析（不是重试旧任务）。新运行会冻结当前文档及已经确认的角色档案。
5. 在角色工作台查看覆盖状态，再从漂移摘要进入完整问题报告核对双侧证据。

若页面显示“部分审查”“模型降级”“角色无法对齐”或“未执行”，本轮不能解释为没有问题；请先查看运行诊断并重试。旧候选的来源文档被新版本替代后会标为过期，不能继续确认。

演示集中故意包含：

- 明确的稳定偏好冲突。
- 缺少成长桥梁的核心人格偏移。
- 有明确事件依据的性格变化，不应作为确定冲突。
- 同一次伪装期间发生在不同时间、地点和交互对象上的两次反常说话行为；审查应将它们视为独立行为，但仍应依据伪装情境解释，而不是直接上升为人格改变。
- 只发生一次的反常行为，只能进入待确认。
