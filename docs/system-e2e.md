# Playwright 生产网页接线测试

这项测试验证“真实浏览器 → `http://127.0.0.1:8080` Nginx 入口 → API → PostgreSQL/Redis/Celery”的接线、排队与历史恢复行为。它不评估大模型的语义质量，也不是 Agent/RAG 验收的替代品。

## 覆盖范围

- 通过页面新建项目，真实点击文件选择器，上传测试内嵌生成的无宏 DOCX。
- 暂停 Celery worker 以稳定观测 `queued`，刷新页面后从运行历史恢复，再恢复 worker 直至 `completed`。
- 主动中断首次 SSE 连接，确认浏览器自动重连，而不把任务误显示为空闲。
- 确认一个确定性事实冲突、两条原文证据及行号在 UI 中可见，且问题和抽取记录不重复。
- 记录浏览器所有 API 请求的 origin，防止测试绕过 Nginx 直连 `8000`。

## 隔离边界

测试始终显式使用 `tests/system/compose.env` 和 `docker-compose.e2e.yml`，不读取本地 `.env`。API 与 worker 中的模型抽取、Embedding、AI 证据复核和 Review Agent 全部强制关闭，Key 为空，因此不调用真实模型或外网。Compose project 固定为 `loreguard-e2e`，数据卷在测试后删除。

## 本地运行

```powershell
cd C:\Users\dell\Desktop\dev\projects\LoreGuard
corepack enable
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend exec playwright install chromium
python scripts/run_system_e2e.py
```

浏览器场景超时为 90 秒，失败时保留截图、Trace、视频和 HTML 报告到 `artifacts/playwright/`。如需保留隔离栈进行排查，可添加 `--keep-stack`。

## 第二阶段接口

后续的 Agent/RAG 浏览器系统测试应使用独立 Compose override 和本地 Stub Provider，并另建语义验收数据集。第一阶段不在本测试中伪造 Agent/RAG 成果。
