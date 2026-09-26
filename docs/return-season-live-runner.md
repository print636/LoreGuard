# 回港季隔离复现实测脚本

`scripts/run_return_season_live.py` 对冻结的 DEV 素材运行两个诊断臂：A 保留候选待审核；选择 `--simulate-author` 时，B 在另一个新项目中按正式档案的冻结原句唯一匹配价值观／行为边界候选，经公开候选审核接口模拟确认，再审查新稿。B 不绑定作者批准比较轴，不代表真实作者确认。两个诊断臂的所有 Oracle 案例都标为 `unassessed`，直至另有经过作者确认、完整覆盖且能绑定案例证据的评分协议。

只做本地文件校验，不启动服务或调用模型：

```powershell
.\.venv\Scripts\python.exe scripts/run_return_season_live.py --preflight-only
```

实测前，先用 `scripts/provision_character_axis_isolation.py --db-path <全新绝对路径>` 在 `Desktop/dev/artifacts/character-axis-direction-v1/` 下创建一个新的 SQLite 文件。记录它返回的 `eval_db_id`。给单独启动的 API 进程设置 `DATABASE_URL` 为该文件的绝对 SQLite URL，`EVAL_ISOLATION_INSTANCE_ID` 为新生成的 UUID4，`LOREGUARD_BUILD_REVISION` 为当前 `git rev-parse HEAD` 的完整提交 SHA；如需本次记录级排障，再设 `CHARACTER_SIGNAL_DRAFT_TRACE_V1=true`。通过 `python -m uvicorn app.main:app --host 127.0.0.1 --port 8877` 启动服务。不要在日常服务的 8000/8080 端口或既有数据库上运行。隔离身份接口会核验数据库文件、标记、进程实例和空项目表，然后脚本才会创建评测项目。

服务启动后，在另一个终端运行（替换尖括号中的值）：

```powershell
.\.venv\Scripts\python.exe scripts/run_return_season_live.py `
  --base-url http://127.0.0.1:8877 `
  --expected-eval-db-id <provision 输出的 eval_db_id> `
  --expected-eval-instance-id <EVAL_ISOLATION_INSTANCE_ID 的值> `
  --expected-eval-db-path <数据库绝对路径> `
  --allow-provider-call `
  --simulate-author `
  --require-draft-trace `
  --output-json artifacts/return-season-<新编号>.json
```

省略 `--simulate-author` 只运行 A。`--require-draft-trace` 会在模型调用前检查服务已开启固定版本的新稿记录诊断；旧服务可省略它，报告会标注 `trace_unavailable`。每次实测都应使用新数据库、进程实例和报告文件；脚本拒绝覆盖既有报告。它固定校验所有素材字节，上传的只有五份故事文档；`review-plan.json` 仅用于本地候选定位，`oracle.json` 在所有运行结束后才解析。报告只保留预设计数、状态、哈希和案例 ID，不记录模型密钥、接口地址、剧情原文、提示词或上游响应。诊断中的 `claimed_start_Dxx` 仅对应模型声称的起始行，`claimed_single_line_Dxx` 只在声称跨度为同一行时出现；二者都未据此校验证据或认定案例命中。`final_accepted` 区分脏包中暂时接受的记录与最终干净包采纳的记录。
