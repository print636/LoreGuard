import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

// Synthetic interaction fixtures only: no evaluation cases, human labels or model requests.
const projectId = "tab-draft-history-project";
const now = "2026-10-02T08:00:00Z";
const screenshotFolder = process.env.LOREGUARD_DRAFT_HISTORY_SCREENSHOT_DIR ?? "../artifacts/drafts-and-run-history";
const runId = (index: number) => `history-run-${String(index).padStart(3, "0")}`;
const statuses = ["completed", "failed", "cancelled", "queued", "running"] as const;
const history = Array.from({ length: 61 }, (_, index) => ({
  id: runId(index), project_id: projectId, status: statuses[index % statuses.length],
  created_at: new Date(Date.parse(now) - index * 60000).toISOString(), started_at: now,
  completed_at: index % 5 < 3 ? now : null, input_chars: 1234,
  prompt_tokens: index === 0 ? 0 : 12, completion_tokens: index === 0 ? 0 : 4,
  estimated_cost_usd: null, frozen_document_count: 2, retried_from: null, batch_mode: "draft_review",
}));
const document = {
  id: "history-document", project_id: projectId, name: "当前章节.md", version: 1, active: true,
  created_at: now, document_role: "chapter", story_scope: "main",
  narrative_context: { revision: 1, resolution_state: "confirmed", publication_status: "draft", origin: "explicit", scope: { schema_version: 1, timeline_key: "main" } },
};
const issues = [1, 2].map((number) => ({
  id: `history-issue-${number}`, category: "fact_conflict", severity: "medium", confidence: 0.7,
  title: `用于反馈交互的原创问题 ${number}`, explanation: "这是页面交互场景，不是模型质量评测。", suggestion: "核对原文。",
  evidence: [{ document_id: document.id, document_name: document.name, line_start: 1, line_end: 1, text: "页面交互用原创句子。" }], metadata: {},
}));

function fullRun(index: number) {
  return { ...history[index], input_snapshot_available: true, input_documents: [{ document_id: document.id, document_name: document.name,
    document_version: 1, document_role: "chapter", story_scope: "main", content_sha256: "a".repeat(64), batch_role: "target" }],
    review_batch: { mode: "draft_review", no_formal_context_expected: true, target_document_ids: [document.id], background_document_ids: [] } };
}

function fixtureState() {
  return { user: "author-a", authenticated: true, anonymous: false, unexpected: [] as string[], writes: [] as string[],
    catalogRequests: [] as string[], catalogFailures: 0, feedbackFailure: false, logout401: false,
    textFailure: false, forceEmptyStatus: "", feedback: {} as Record<string, unknown>,
    gateStatus: "", gate: null as Promise<void> | null, releaseGate: null as (() => void) | null,
    feedbackGate: null as Promise<void> | null, releaseFeedback: null as (() => void) | null };
}
type Fixture = ReturnType<typeof fixtureState>;
function identity(state: Fixture) {
  return { mode: state.anonymous ? "anonymous" : "required", user: { id: state.user, email: `${state.user}@example.test`, display_name: state.user },
    workspace: { id: `workspace-${state.user}`, name: "原创故事工作区", kind: "personal", role: "owner" } };
}

async function mockApi(page: Page, state: Fixture) {
  // Catch ALL business endpoints. Unexpected calls fail locally and never reach
  // the actual local database, provider settings, human cases or model gateway.
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const endpoint = url.pathname;
    const method = request.method();
    if (method !== "GET") state.writes.push(`${method} ${endpoint}`);
    if (endpoint === "/api/v1/auth/login" && method === "POST") {
      const email = request.postDataJSON().email as string;
      state.user = email.startsWith("author-b") ? "author-b" : "author-a";
      state.authenticated = true;
      await route.fulfill({ json: identity(state) }); return;
    }
    if (endpoint === "/api/v1/auth/logout" && method === "POST") {
      state.authenticated = false;
      await route.fulfill({ status: state.logout401 ? 401 : 200, json: state.logout401 ? { detail: "expired" } : {} }); return;
    }
    if (!state.authenticated) { await route.fulfill({ status: 401, json: { detail: "fixture session expired" } }); return; }
    let body: unknown;
    if (endpoint === "/api/v1/auth/me") body = identity(state);
    else if (endpoint === "/api/v1/account/model-provider") body = { configured: false, revision: 0 };
    else if (endpoint === "/api/v1/project-catalog") {
      const item = { id: projectId, name: "星港短篇交互工程", description: "隔离页面验收", created_at: now,
        active_document_count: 1, latest_run: fullRun(0) };
      body = { page: 1, page_size: 40, total: 1, items: [item] };
    }
    else if (endpoint === `/api/v1/projects/${projectId}/documents`) body = [document];
    else if (endpoint.endsWith("/narrative-context")) body = { document_id: document.id, current: document.narrative_context, revision_count: 1 };
    else if (endpoint === `/api/v1/projects/${projectId}/analysis-runs` && method === "GET") body = history.map((_, index) => fullRun(index));
    else if (endpoint === `/api/v1/projects/${projectId}/run-catalog`) {
      state.catalogRequests.push(url.search);
      if (state.catalogFailures > 0) { state.catalogFailures -= 1; await route.fulfill({ status: 503, json: { detail: "fixture catalog temporarily unavailable" } }); return; }
      const pageNumber = Number(url.searchParams.get("page"));
      const status = url.searchParams.get("status") || "all";
      const items = status === state.forceEmptyStatus ? [] : history.filter((item) => status === "all" || item.status === status);
      if (status === state.gateStatus && state.gate) await state.gate;
      const offset = (pageNumber - 1) * 20;
      body = { project_id: projectId, page: pageNumber, page_size: 20, total: items.length, has_more: offset + 20 < items.length, items: items.slice(offset, offset + 20) };
    }
    else if (endpoint === "/api/v1/projects" && method === "POST") body = { id: projectId, name: "快速文本独立工程", created_at: now };
    else if (endpoint === `/api/v1/projects/${projectId}/documents/text` && method === "POST") {
      if (state.textFailure) { await route.fulfill({ status: 503, json: { detail: "fixture upload failed" } }); return; }
      body = document;
    }
    else if (endpoint === `/api/v1/projects/${projectId}/analysis-runs` && method === "POST") {
      await route.fulfill({ status: 503, json: { detail: "fixture analysis start failed after successful import" } }); return;
    }
    else if (/^\/api\/v1\/analysis-runs\/history-run-\d+$/.test(endpoint)) body = fullRun(Number(endpoint.slice(-3)));
    else if (/^\/api\/v1\/analysis-runs\/history-run-\d+\/issues$/.test(endpoint)) body = issues;
    else if (/^\/api\/v1\/analysis-runs\/history-run-\d+\/records$/.test(endpoint)) body = { records: [], warnings: [] };
    else if (/^\/api\/v1\/analysis-runs\/history-run-\d+\/diagnostics$/.test(endpoint)) body = {};
    else if (/^\/api\/v1\/analysis-runs\/history-run-\d+\/clarifications$/.test(endpoint)) body = [];
    else if (/^\/api\/v1\/analysis-runs\/history-run-\d+\/review-clues$/.test(endpoint)) body = { items: [], truncated: false, unavailable_count: 0 };
    else if (/^\/api\/v1\/analysis-runs\/history-run-\d+\/provisional-clues$/.test(endpoint)) body = { items: [], truncated: false };
    else if (/^\/api\/v1\/issues\/history-issue-[12]\/feedback$/.test(endpoint)) {
      const id = endpoint.split("/")[4];
      if (method === "POST") {
        if (state.feedbackFailure) { await route.fulfill({ status: 503, json: { detail: "fixture feedback failed" } }); return; }
        body = { id: `feedback-${id}`, ...request.postDataJSON(), created_at: now };
        state.feedback[id] = body;
      } else { if (state.feedbackGate) await state.feedbackGate; body = { latest: state.feedback[id] || null }; }
    }
    else { state.unexpected.push(`${method} ${endpoint}`); await route.fulfill({ status: 404, json: { detail: "Unexpected isolated fixture endpoint" } }); return; }
    try { await route.fulfill({ json: body }); } catch { /* Intentionally aborted delayed fixture request. */ }
  });
}

async function open(page: Page, state: Fixture, url: string) {
  await mockApi(page, state);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(url);
  await page.waitForLoadState("networkidle");
}
async function login(page: Page, who = "author-a") {
  await expect(page.getByRole("heading", { name: "登录", exact: true })).toBeVisible();
  await page.getByLabel("邮箱", { exact: true }).fill(`${who}@example.test`);
  await page.getByLabel("密码", { exact: true }).fill("Synthetic-login-only");
  await page.getByRole("button", { name: "登录", exact: true }).click();
  await page.waitForLoadState("networkidle");
}
async function loginThroughGuestCheckGate(page: Page, who = "author-a") {
  await expect(page).toHaveURL(/\/check$/);
  const guest = page.getByRole("main", { name: "访客空工作区", exact: true });
  await expect(guest).toBeVisible();
  await expect(guest.getByRole("heading", { name: "还没有待审文稿", exact: true })).toBeVisible();
  await guest.getByRole("button", { name: "开始校验", exact: true }).click();
  const reminder = page.getByRole("dialog", { name: "登录后继续", exact: true });
  await expect(reminder).toBeVisible();
  await reminder.getByRole("button", { name: "去登录", exact: true }).click();
  await expect(page).toHaveURL(/\/login\?returnTo=%2Fcheck$/);
  await login(page, who);
  await expect(page).toHaveURL(/\/check$/);
}
async function draftEntries(page: Page) {
  return page.evaluate(() => Object.keys(sessionStorage).filter((key) => key.startsWith("loreguard:tab-draft:v1:")).map((key) => ({ key, value: JSON.parse(sessionStorage.getItem(key)!) })));
}
function bodyEditor(page: Page) { return page.getByRole("textbox", { name: "需要审查的故事正文 / chapter.md", exact: true }); }
function notes(page: Page, number: number) { return page.locator(`#issue-history-issue-${number} .note`); }
function panel(page: Page) { return page.getByRole("region", { name: "运行历史", exact: true }); }
async function noUnexpected(state: Fixture) { expect(state.unexpected, "No request may bypass the fixture").toEqual([]); }

test("未改默认样例不缓存；401后原账户恢复正文、设定及身份设置，刷新也不丢稿", async ({ page }) => {
  const state = fixtureState();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await open(page, state, "/check");
  expect(await draftEntries(page)).toEqual([]);
  await page.getByRole("button", { name: "高级：设定 + 章节", exact: true }).click();
  await page.getByRole("textbox", { name: "可选权威设定 / world.md", exact: true }).fill("原创设定：北港书库只在黎明开门。");
  await page.getByRole("textbox", { name: "待审章节 / chapter.md", exact: true }).fill("我还没有提交的章节。\n夜间书库门前的对白。");
  await page.getByRole("textbox", { name: "快速文本故事作用域" }).fill("北港支线");
  await page.reload();
  await expect(page.getByRole("textbox", { name: "待审章节 / chapter.md", exact: true })).toHaveValue("我还没有提交的章节。\n夜间书库门前的对白。");
  await expect(page.getByRole("textbox", { name: "可选权威设定 / world.md", exact: true })).toHaveValue("原创设定：北港书库只在黎明开门。");
  await expect(page.getByRole("textbox", { name: "快速文本故事作用域" })).toHaveValue("北港支线");
  await expect(page.getByRole("complementary", { name: "正文暂存状态" })).toContainText("已恢复本标签页");
  state.authenticated = false;
  await page.getByRole("button", { name: "开始校验", exact: true }).click();
  await expect(page).toHaveURL(/\/login\?returnTo=/);
  expect(await draftEntries(page)).toHaveLength(1);
  await expect(page.locator("body")).not.toContainText("夜间书库门前的对白");
  await login(page);
  await expect(page.getByRole("textbox", { name: "待审章节 / chapter.md", exact: true })).toHaveValue("我还没有提交的章节。\n夜间书库门前的对白。");
  await expect(page.getByRole("textbox", { name: "快速文本故事作用域" })).toHaveValue("北港支线");
  await page.screenshot({ path: path.join(screenshotFolder, "quick-draft-restored-1440.png") });
  expect((await draftEntries(page))[0].value.payload).not.toHaveProperty("password");
  await noUnexpected(state);
});

test("换账户不泄露也不删除原稿；原账户主动退出清理且不会由卸载复写", async ({ page }) => {
  const state = fixtureState();
  await open(page, state, "/check");
  const privateText = "只属于账户A的未提交原创稿件。";
  await bodyEditor(page).fill(privateText);
  state.authenticated = false;
  await page.getByRole("button", { name: "开始校验", exact: true }).click();
  await login(page, "author-b");
  await expect(bodyEditor(page)).not.toHaveValue(privateText);
  expect(await draftEntries(page)).toHaveLength(1);
  state.authenticated = false;
  await page.getByRole("button", { name: "开始校验", exact: true }).click();
  await login(page, "author-a");
  await expect(bodyEditor(page)).toHaveValue(privateText);
  await page.getByRole("button", { name: "退出", exact: true }).click();
  await expect(page.getByRole("heading", { name: "登录", exact: true })).toBeVisible();
  expect(await draftEntries(page)).toEqual([]);
  await login(page);
  await page.goto("/check");
  await expect(bodyEditor(page)).not.toHaveValue(privateText);
  expect(await draftEntries(page)).toEqual([]);
  await noUnexpected(state);
});

test("主动退出返回401仍清稿；普通auth/me401刷新不清稿", async ({ page }) => {
  const state = fixtureState();
  const privateText = "重新登录之前需要保留的正文。";
  await open(page, state, "/check");
  await bodyEditor(page).fill(privateText);
  state.authenticated = false;
  await page.reload();
  await expect(page).toHaveURL(/\/check$/);
  const guest = page.getByRole("main", { name: "访客空工作区", exact: true });
  await expect(guest).toBeVisible();
  await expect(guest.getByRole("heading", { name: "还没有待审文稿", exact: true })).toBeVisible();
  expect(await draftEntries(page)).toHaveLength(1);
  await expect(page.locator("body")).not.toContainText(privateText);
  await loginThroughGuestCheckGate(page);
  await expect(bodyEditor(page)).toHaveValue(privateText);
  state.logout401 = true;
  await page.getByRole("button", { name: "退出", exact: true }).click();
  await expect(page).toHaveURL(/\/check$/);
  await expect(guest).toBeVisible();
  await expect(guest.getByRole("heading", { name: "还没有待审文稿", exact: true })).toBeVisible();
  expect(await draftEntries(page)).toEqual([]);
  await expect(page.locator("body")).not.toContainText(privateText);
  await noUnexpected(state);
});

test("备注成功只清对应问题，失败与401保留；换run不会误绑", async ({ page }) => {
  const state = fixtureState();
  await open(page, state, `/app/projects/${projectId}/runs/${runId(0)}/report`);
  await notes(page, 1).fill("问题一：已准备提交的备注。");
  await notes(page, 2).fill("问题二：尚未提交的备注。");
  await page.locator("#issue-history-issue-1").getByRole("button", { name: "已接受", exact: true }).click();
  await expect(notes(page, 1)).toHaveValue("");
  const cached = (await draftEntries(page))[0].value.payload.notes;
  expect(cached).not.toHaveProperty("history-issue-1");
  expect(cached["history-issue-2"]).toBe("问题二：尚未提交的备注。");
  state.feedbackFailure = true;
  await page.locator("#issue-history-issue-2").getByRole("button", { name: "已接受", exact: true }).click();
  await expect(page.locator(".railLive")).toContainText("提交反馈失败");
  await page.reload();
  await expect(notes(page, 2)).toHaveValue("问题二：尚未提交的备注。");
  await expect(page.getByRole("complementary", { name: "备注暂存状态" })).toContainText("已恢复本标签页");
  state.authenticated = false;
  await page.locator("#issue-history-issue-2").getByRole("button", { name: "已接受", exact: true }).click();
  await login(page);
  await expect(notes(page, 2)).toHaveValue("问题二：尚未提交的备注。");
  await page.goto(`/app/projects/${projectId}/runs/${runId(5)}/report`);
  await expect(notes(page, 2)).toHaveValue("");
  await page.goto(`/app/projects/${projectId}/runs/${runId(0)}/report`);
  await expect(notes(page, 2)).toHaveValue("问题二：尚未提交的备注。");
  await page.setViewportSize({ width: 375, height: 812 });
  await page.getByRole("complementary", { name: "备注暂存状态" }).scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  await page.screenshot({ path: path.join(screenshotFolder, "feedback-draft-restored-375.png") });
  await noUnexpected(state);
});

test("存储禁用不阻止编辑，也不把最新内容说成已保存", async ({ page }) => {
  const state = fixtureState();
  await page.setViewportSize({ width: 375, height: 812 });
  await open(page, state, "/check");
  await page.evaluate(() => { Storage.prototype.setItem = () => { throw new DOMException("synthetic quota", "QuotaExceededError"); }; });
  await bodyEditor(page).fill("存储失败时仍在页面内的正文。");
  await expect(bodyEditor(page)).toHaveValue("存储失败时仍在页面内的正文。");
  const notice = page.getByRole("complementary", { name: "正文暂存状态" });
  await expect(notice).toContainText("浏览器暂存不可用");
  await expect(notice).not.toContainText("正文已在本标签页暂存");
  expect(await draftEntries(page)).toEqual([]);
  await notice.scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  await page.screenshot({ path: path.join(screenshotFolder, "quick-draft-storage-unavailable-375.png") });
  await noUnexpected(state);
});

test("反馈和暂存载入前备注不可编辑，完成后新输入不会被初始化覆盖", async ({ page }) => {
  const state = fixtureState();
  state.feedbackGate = new Promise<void>((resolve) => { state.releaseFeedback = resolve; });
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId(0)}/report`);
  await expect(notes(page, 1)).toBeDisabled();
  await expect(page.locator(".issues")).toContainText("读取完成后才能编辑");
  state.releaseFeedback!();
  state.feedbackGate = null;
  await expect(notes(page, 1)).toBeEnabled();
  await notes(page, 1).fill("初始化完成后的新备注。");
  await page.reload();
  await expect(notes(page, 1)).toHaveValue("初始化完成后的新备注。");
  await noUnexpected(state);
});

test("匿名体验仍能提交反馈并更新UI，但不缓存未提交文本", async ({ page }) => {
  const state = fixtureState(); state.anonymous = true;
  await open(page, state, `/app/projects/${projectId}/runs/${runId(0)}/report`);
  await notes(page, 1).fill("本地体验的反馈备注。");
  await page.locator("#issue-history-issue-1").getByRole("button", { name: "已接受", exact: true }).click();
  await expect(page.locator("#issue-history-issue-1 .feedbackState")).toContainText("已接受");
  await expect(page.locator(".railLive")).toContainText("反馈已记录");
  expect(await draftEntries(page)).toEqual([]);
  await expect(page.locator(".issues")).toContainText("本地体验模式不暂存未提交备注");
  await noUnexpected(state);
});

test("文本导入成功后清稿，即使启动分析失败；上传失败继续保留", async ({ page }) => {
  const state = fixtureState();
  await open(page, state, "/check");
  await bodyEditor(page).fill("需要导入的原创正文。");
  state.textFailure = true;
  await page.getByRole("button", { name: "开始校验", exact: true }).click();
  await expect(page.locator(".railLive")).toContainText("fixture upload failed");
  expect(await draftEntries(page)).toHaveLength(1);
  state.textFailure = false;
  await page.getByRole("button", { name: "开始校验", exact: true }).click();
  await expect(page.locator(".railLive")).toContainText("fixture analysis start failed after successful import");
  expect(await draftEntries(page)).toEqual([]);
  expect(state.writes.some((value) => value.endsWith("/documents/text"))).toBe(true);
  await noUnexpected(state);
});

test("61条历史服务端分页与状态URL/back生效，页外选中run保持不变", async ({ page }) => {
  const state = fixtureState();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await open(page, state, `/app/projects/${projectId}/runs/${runId(60)}`);
  await expect(panel(page).locator("tbody tr")).toHaveCount(20);
  await expect(panel(page)).toContainText("共 61 次，第 1 / 4 页");
  await expect(panel(page)).toContainText("不在本页中，仍保持选中");
  await expect(page.locator(".status code").nth(1)).toHaveText(runId(60));
  await panel(page).getByRole("button", { name: "下一页", exact: true }).click();
  await expect(page).toHaveURL(/history_page=2/);
  await expect(panel(page)).toContainText("第 2 / 4 页");
  await page.goBack();
  await expect(panel(page)).toContainText("第 1 / 4 页");
  await panel(page).getByRole("combobox", { name: "运行状态" }).selectOption("completed");
  await expect(page).toHaveURL(/history_status=completed/);
  await expect(panel(page).locator("tbody tr")).toHaveCount(13);
  await expect(page.locator(".status code").nth(1)).toHaveText(runId(60));
  await expect(panel(page)).toContainText("已记录用量，不证明模型实际参与");
  await panel(page).getByRole("combobox", { name: "运行状态" }).focus();
  expect(await panel(page).getByRole("combobox", { name: "运行状态" }).evaluate((element) => getComputedStyle(element).outlineStyle)).not.toBe("none");
  await page.screenshot({ path: path.join(screenshotFolder, "run-history-filter-1440.png") });
  expect(await panel(page).evaluate((element) => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
  expect(state.catalogRequests).toContain("?page=2&page_size=20&status=all");
  await noUnexpected(state);
});

test("成功导入后复用可见设定写下一章，新的暂存包保留完整上下文", async ({ page }) => {
  const state = fixtureState();
  await open(page, state, "/check");
  await page.getByRole("button", { name: "高级：设定 + 章节", exact: true }).click();
  await page.getByRole("textbox", { name: "可选权威设定 / world.md", exact: true }).fill("需要在下一章复用的原创世界观。");
  await page.getByRole("textbox", { name: "待审章节 / chapter.md", exact: true }).fill("已导入的第一章。");
  await page.getByRole("button", { name: "开始校验", exact: true }).click();
  await expect(page.locator(".railLive")).toContainText("fixture analysis start failed after successful import");
  expect(await draftEntries(page)).toEqual([]);
  // Navigate within the same mounted workspace so the imported visible values remain available.
  await page.evaluate(() => { history.pushState({}, "", "/check"); window.dispatchEvent(new PopStateEvent("popstate")); });
  const chapter = page.getByRole("textbox", { name: "待审章节 / chapter.md", exact: true });
  await expect(chapter).toHaveValue("已导入的第一章。");
  await chapter.fill("尚未导入的第二章。");
  expect((await draftEntries(page))[0].value.payload.world).toBe("需要在下一章复用的原创世界观。");
  await page.reload();
  await expect(chapter).toHaveValue("尚未导入的第二章。");
  await expect(page.getByRole("textbox", { name: "可选权威设定 / world.md", exact: true })).toHaveValue("需要在下一章复用的原创世界观。");
  await noUnexpected(state);
});

test("迟到状态筛选响应不能覆盖新结果，历史错误可重试与超页可恢复", async ({ page }) => {
  const state = fixtureState();
  state.catalogFailures = 1;
  await open(page, state, `/app/projects/${projectId}/runs/${runId(0)}`);
  await expect(panel(page)).toContainText("运行历史读取失败");
  await panel(page).getByRole("button", { name: "重试读取历史", exact: true }).click();
  await expect(panel(page).locator("tbody tr")).toHaveCount(20);
  state.gateStatus = "failed";
  state.gate = new Promise<void>((resolve) => { state.releaseGate = resolve; });
  await panel(page).getByRole("combobox", { name: "运行状态" }).selectOption("failed");
  await expect.poll(() => state.catalogRequests.some((query) => query.endsWith("status=failed"))).toBe(true);
  await panel(page).getByRole("combobox", { name: "运行状态" }).selectOption("completed");
  await expect(panel(page).locator("tbody tr")).toHaveCount(13);
  state.releaseGate!();
  await page.waitForLoadState("networkidle");
  await expect(panel(page).getByRole("combobox", { name: "运行状态" })).toHaveValue("completed");
  await expect(panel(page).locator("tbody tr")).toHaveCount(13);
  await page.goto(`/app/projects/${projectId}/runs/${runId(0)}?history_page=99`);
  await expect(panel(page)).toContainText("已超出当前结果范围");
  await panel(page).getByRole("button", { name: "前往最后一页", exact: true }).click();
  await expect(panel(page).locator("tbody tr")).toHaveCount(1);
  await expect(panel(page)).toContainText("第 4 / 4 页");
  state.forceEmptyStatus = "cancelled";
  await panel(page).getByRole("combobox", { name: "运行状态" }).selectOption("cancelled");
  await expect(panel(page)).toContainText("没有已取消的运行");
  await expect(page.locator(".status code").nth(1)).toHaveText(runId(0));
  await noUnexpected(state);
});

test("375像素历史重排无横滚，操作与筛选保持44px，打开详情按id查询", async ({ page }) => {
  const state = fixtureState();
  await page.setViewportSize({ width: 375, height: 812 });
  await open(page, state, `/app/projects/${projectId}/runs/${runId(0)}?history_page=4`);
  await expect(panel(page).locator("tbody tr")).toHaveCount(1);
  await panel(page).scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  expect(await panel(page).evaluate((element) => element.scrollWidth <= element.clientWidth + 1), "History itself must not hide horizontal overflow").toBe(true);
  const table = panel(page).locator("table");
  expect(await table.evaluate((element) => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
  const controls = panel(page).locator("button:visible, select:visible");
  for (const control of await controls.all()) expect((await control.boundingBox())!.height).toBeGreaterThanOrEqual(44);
  await page.screenshot({ path: path.join(screenshotFolder, "run-history-last-page-375.png") });
  await panel(page).locator("tbody tr").getByRole("button", { name: /查看报告/ }).click();
  await expect(page).toHaveURL(new RegExp(`/runs/${runId(60)}/report`));
  await expect(page.locator(".issues article")).toHaveCount(2);
  await noUnexpected(state);
});
