import path from "node:path";
import { expect, test, type Dialog, type Page, type Route } from "@playwright/test";

test.setTimeout(45_000);

// Synthetic author-workspace fixtures only. Every API request is intercepted;
// an unknown endpoint is rejected rather than forwarded to a live backend.
const projectId = "context-protection-project";
const documentId = "context-source-one";
const secondDocumentId = "context-source-two";
const now = "2026-10-03T00:00:00Z";
const root = `/api/v1/projects/${projectId}`;
const documentsPath = `/app/projects/${projectId}/documents`;
const checkPath = `/app/projects/${projectId}/check`;
const screenshotFolder = process.env.LOREGUARD_CONTEXT_PROTECTION_SCREENSHOT_DIR
  ?? process.env.LOREGUARD_E2E_SCREENSHOT_DIR
  ?? "../artifacts/context-editor-protection";

type Role = "chapter" | "canon" | "character_profile" | "reference";
type Publication = "unknown" | "draft" | "in_review" | "published" | "retired";
type Scope = { schema_version: 1; timeline_key: string; activity_key?: string };
type Inference = {
  confidence: number; reasoning: string;
  evidence: { document_id: string; document_name: string; line_start: number; line_end: number; text: string; supported_fields: string[] }[];
  usage: { prompt_tokens: number; completion_tokens: number; total_tokens: number };
};
type Context = {
  revision: number; resolution_state: "unresolved" | "inferred" | "confirmed";
  origin: string; publication_status: Publication; scope: Scope;
  scope_sha256: string; inference?: Inference;
};
type Document = {
  id: string; project_id: string; name: string; version: number; active: boolean;
  created_at: string; document_role: Role; story_scope: string; narrative_context: Context;
};
type SavePayload = {
  document_role: Role; resolution_state: "unresolved" | "confirmed";
  publication_status: Publication; scope: Scope; expected_revision: number;
};

function context(timeline = "main"): Context {
  return { revision: 2, resolution_state: "unresolved", origin: "explicit", publication_status: "unknown",
    scope: { schema_version: 1, timeline_key: timeline }, scope_sha256: `${timeline}-scope` };
}

function document(id: string, name: string, timeline = "main", version = 1): Document {
  return { id, project_id: projectId, name, version, active: true, created_at: now,
    document_role: "reference", story_scope: "global", narrative_context: context(timeline) };
}

function fixture() {
  return {
    documents: [document(documentId, "01-活动草稿.md"), document(secondDocumentId, "02-背景资料.md", "side")],
    requests: [] as string[], unexpected: [] as string[],
    contextReads: [] as string[], saves: [] as { documentId: string; payload: SavePayload }[],
    inferences: [] as { documentId: string; payload: { expected_revision: number } }[],
    uploads: [] as string[], saveFailure: false, contextReadUnavailable: false,
    contextMalformed: false, documentsUnavailable: false,
    delayedDocument: "", releaseDelayed: undefined as (() => void) | undefined,
    delayedFulfilled: false, dropOldOnUpload: false,
  };
}
type Fixture = ReturnType<typeof fixture>;

async function fulfill(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, json: body });
}

async function mockApi(page: Page, state: Fixture) {
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const endpoint = url.pathname;
    const method = request.method();
    state.requests.push(`${method} ${endpoint}`);
    let body: unknown;
    if (method === "GET" && endpoint === "/api/v1/auth/me") body = {
      mode: "anonymous", user: { id: "context-author", email: "", display_name: "作者" },
      workspace: { id: "context-workspace", name: "原创故事工作区", kind: "personal", role: "owner" },
    };
    else if (method === "GET" && endpoint === "/api/v1/account/model-provider") body = { configured: false, revision: 0 };
    else if (method === "GET" && endpoint === "/api/v1/project-catalog") body = {
      page: 1, page_size: 40, total: 1,
      items: [{ id: projectId, name: "雾潮剧情工程", description: "原创资料身份编辑验收", created_at: now,
        active_document_count: state.documents.filter((item) => item.active).length, latest_run: null }],
    };
    else if (method === "GET" && endpoint === `${root}/analysis-runs`) body = [];
    else if (method === "GET" && endpoint === `${root}/documents`) {
      if (state.documentsUnavailable) { await fulfill(route, { detail: "项目资料暂时无法读取。" }, 503); return; }
      body = state.documents;
    }
    else if (method === "POST" && endpoint === `${root}/documents`) {
      const name = /filename="([^"]+)"/u.exec(request.postDataBuffer()?.toString("utf8") ?? "")?.[1] ?? "";
      state.uploads.push(name);
      const old = state.documents.find((item) => item.active && item.name === name);
      const imported = document(`context-upload-${state.uploads.length}`, name, "main", (old?.version ?? 0) + 1);
      if (old) old.active = false;
      state.documents.push(imported);
      if (state.dropOldOnUpload) state.documents = state.documents.filter((item) => item.id !== documentId);
      await fulfill(route, { ...imported, superseded_document_ids: old ? [old.id] : [], deduplicated: false }, 201);
      return;
    }
    else if (method === "GET" && /^\/api\/v1\/projects\/[^/]+\/documents\/[^/]+\/preview$/u.test(endpoint)) {
      const id = endpoint.split("/").at(-2);
      const selected = state.documents.find((item) => item.id === id);
      if (!selected) { await fulfill(route, { detail: "资料不存在" }, 404); return; }
      body = { document: { ...selected, char_count: 20, line_count: 1, content_sha256: "a".repeat(64) },
        lines: [{ line_number: 1, text: "原创活动资料，请作者核对故事位置。" }],
        page: { offset: 0, limit: 120, total: 1, has_more: false }, query: "" };
    }
    else if (endpoint.startsWith(`${root}/documents/`) && endpoint.endsWith("/narrative-context") && method === "GET") {
      const id = endpoint.split("/").at(-2) ?? "";
      const selected = state.documents.find((item) => item.id === id);
      state.contextReads.push(id);
      if (!selected) { await fulfill(route, { detail: "资料不存在" }, 404); return; }
      if (state.contextReadUnavailable) { await fulfill(route, { detail: "资料身份暂时无法读取。" }, 503); return; }
      if (state.contextMalformed) { await fulfill(route, { document_id: id, revision_count: 2 }); return; }
      const snapshot = structuredClone(selected.narrative_context);
      if (state.delayedDocument === id) {
        await new Promise<void>((resolve) => { state.releaseDelayed = resolve; });
      }
      await fulfill(route, { document_id: id, current: snapshot, revision_count: snapshot.revision });
      if (state.delayedDocument === id) state.delayedFulfilled = true;
      return;
    }
    else if (endpoint.startsWith(`${root}/documents/`) && endpoint.endsWith("/narrative-context/revisions") && method === "POST") {
      const id = endpoint.split("/").at(-3) ?? "";
      const selected = state.documents.find((item) => item.id === id);
      const payload = request.postDataJSON() as SavePayload;
      state.saves.push({ documentId: id, payload });
      if (state.saveFailure) { await fulfill(route, { detail: "资料服务暂时不可用，请稍后重试。" }, 503); return; }
      if (!selected?.active || selected.narrative_context.revision !== payload.expected_revision) {
        await fulfill(route, { detail: { code: "narrative_context_revision_conflict", message: "资料已更新。" } }, 409); return;
      }
      selected.document_role = payload.document_role;
      selected.narrative_context = { ...selected.narrative_context, revision: selected.narrative_context.revision + 1,
        origin: "explicit", resolution_state: payload.resolution_state,
        publication_status: payload.publication_status, scope: payload.scope };
      await fulfill(route, selected.narrative_context, 201);
      return;
    }
    else if (endpoint.startsWith(`${root}/documents/`) && endpoint.endsWith("/narrative-context/inference") && method === "POST") {
      const id = endpoint.split("/").at(-3) ?? "";
      const selected = state.documents.find((item) => item.id === id);
      const payload = request.postDataJSON() as { expected_revision: number };
      state.inferences.push({ documentId: id, payload });
      if (!selected) { await fulfill(route, { detail: "资料不存在" }, 404); return; }
      const usage = { prompt_tokens: 10, completion_tokens: 20, total_tokens: 30 };
      // This is a mocked service receipt, not a real model call or gold annotation.
      selected.document_role = "chapter";
      selected.narrative_context = { ...selected.narrative_context, revision: selected.narrative_context.revision + 1,
        resolution_state: "inferred", origin: "model_inferred", publication_status: "draft",
        scope: { schema_version: 1, timeline_key: "inferred_main" },
        inference: { confidence: 0.8, reasoning: "文稿中的活动叙述需要由作者核对。",
          evidence: [{ document_id: id, document_name: selected.name, line_start: 1, line_end: 1,
            text: "原创活动草稿。", supported_fields: ["document_role", "publication_status"] }], usage } };
      await fulfill(route, { document_id: id, document_role: selected.document_role, suggestion: selected.narrative_context, usage });
      return;
    }
    else {
      state.unexpected.push(`${method} ${endpoint}`);
      await fulfill(route, { detail: "Unrecognized isolated context fixture endpoint" }, 404);
      return;
    }
    await fulfill(route, body);
  });
}

function editor(page: Page) { return page.getByRole("region", { name: "确认资料身份", exact: true }); }
function documentButton(page: Page, name = "02-背景资料.md") {
  return editor(page).getByRole("navigation", { name: "活动资料列表", exact: true }).getByRole("button", { name: new RegExp(name.replaceAll(".", "\\."), "u") });
}
function timeline(page: Page) { return editor(page).getByRole("textbox", { name: /^时间线/u }); }
function role(page: Page) { return editor(page).getByRole("combobox", { name: /^资料类型/u }); }
function publication(page: Page) { return editor(page).getByRole("combobox", { name: /^发布状态/u }); }
function confirmed(page: Page) { return editor(page).getByRole("checkbox", { name: /我已核对这份资料的身份与故事位置/u }); }

async function open(page: Page, state: Fixture, location = documentsPath) {
  await mockApi(page, state);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(location);
  if (location === documentsPath && !state.delayedDocument && !state.contextReadUnavailable && !state.contextMalformed) await expect(timeline(page)).toHaveValue("main");
}

async function editAll(page: Page) {
  await role(page).selectOption("canon");
  await publication(page).selectOption("published");
  await timeline(page).fill("author_main");
  await confirmed(page).check();
}

async function expectAllDraft(page: Page) {
  await expect(role(page)).toHaveValue("canon");
  await expect(publication(page)).toHaveValue("published");
  await expect(timeline(page)).toHaveValue("author_main");
  await expect(confirmed(page)).toBeChecked();
}

async function confirmation(page: Page, action: () => Promise<unknown>, accept: boolean) {
  const dialogs: { type: string; message: string }[] = [];
  const handle = async (dialog: Dialog) => {
    dialogs.push({ type: dialog.type(), message: dialog.message() });
    if (accept) await dialog.accept(); else await dialog.dismiss();
  };
  page.once("dialog", handle);
  await action();
  page.off("dialog", handle);
  expect(dialogs, "此操作必须先明确确认未保存修改").toHaveLength(1);
  expect(dialogs[0].type).toBe("confirm");
  expect(dialogs[0].message).toMatch(/未保存|尚未保存|改动|修改/u);
}

function assertIsolated(state: Fixture) {
  expect(state.unexpected, "所有API均由合成fixture拦截，不可访问真实服务").toEqual([]);
  expect(state.requests.filter((request) => /^(POST|PATCH|PUT|DELETE) /u.test(request)
    && !new RegExp(`^POST ${root}/documents(?:$|/[^/]+/narrative-context/(?:revisions|inference)$)`, "u").test(request)),
  "资料身份编辑不得启动分析、定稿发布、角色确认或其他业务写入").toEqual([]);
}

async function capture(page: Page, name: string) {
  await editor(page).evaluate((element) => element.scrollIntoView({ block: "start", inline: "nearest", behavior: "instant" }));
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1)).toBe(true);
  await page.screenshot({ path: path.join(screenshotFolder, name) });
}

test("正确保存解除dirty；clean和恢复原值不误拦资料切换", async ({ page }) => {
  const state = fixture();
  state.contextMalformed = true;
  await open(page, state);
  await expect(editor(page).getByRole("alert")).toContainText("资料身份操作未完成");
  await expect(editor(page).getByRole("button", { name: /^保存/u })).toHaveCount(0);
  await expect(timeline(page)).toHaveCount(0);
  expect(state.saves).toEqual([]);
  state.contextMalformed = false;
  state.contextReadUnavailable = true;
  await editor(page).getByRole("button", { name: "重新读取", exact: true }).click();
  await expect(editor(page).getByRole("alert")).toContainText("资料身份操作未完成");
  await expect(timeline(page)).toHaveCount(0);
  expect(state.saves).toEqual([]);
  state.contextReadUnavailable = false;
  await editor(page).getByRole("button", { name: "重新读取", exact: true }).click();
  await expect(timeline(page)).toHaveValue("main");
  let unexpectedDialog = false;
  page.on("dialog", async (dialog) => { unexpectedDialog = true; await dialog.dismiss(); });
  await timeline(page).fill("临时主线");
  await timeline(page).fill("main");
  await documentButton(page).click();
  await expect(timeline(page)).toHaveValue("side");
  await documentButton(page, "01-活动草稿.md").click();
  await expect(timeline(page)).toHaveValue("main");
  await editAll(page);
  await editor(page).getByRole("button", { name: "保存并确认", exact: true }).click();
  await expect(editor(page)).toContainText("已确认“01-活动草稿.md”");
  expect(state.saves).toHaveLength(1);
  expect(state.saves[0]).toEqual({ documentId, payload: { document_role: "canon", resolution_state: "confirmed",
    publication_status: "published", scope: { schema_version: 1, timeline_key: "author_main" }, expected_revision: 2 } });
  await documentButton(page).click();
  await expect(timeline(page)).toHaveValue("side");
  expect(unexpectedDialog).toBe(false);
  assertIsolated(state);
});

test("切另一资料须确认；取消保类型、时间线、发布与作者核对输入", async ({ page }) => {
  const state = fixture();
  await page.setViewportSize({ width: 1440, height: 960 });
  await open(page, state);
  await editAll(page);
  await confirmation(page, () => documentButton(page).click(), false);
  await expectAllDraft(page);
  await capture(page, "context-editor-desktop-1440.png");
  await confirmation(page, () => documentButton(page).click(), true);
  await expect(timeline(page)).toHaveValue("side");
  expect(state.saves).toEqual([]);
  assertIsolated(state);
});

test("中央内部导航不绕过dirty保护；取消不换URL或丢稿", async ({ page }) => {
  const state = fixture();
  await open(page, state);
  await editAll(page);
  await confirmation(page, () => page.getByRole("button", { name: "项目中心", exact: true }).click(), false);
  await expect(page).toHaveURL(new RegExp(`${documentsPath}$`, "u"));
  await expectAllDraft(page);
  await confirmation(page, () => page.getByRole("button", { name: "项目中心", exact: true }).click(), true);
  await expect(page).toHaveURL(/\/app$/u);
  await expect(page.getByRole("heading", { name: "项目列表", exact: true })).toBeVisible();
  expect(state.saves).toEqual([]);
  assertIsolated(state);
});

test("浏览器后退与前进取消保留当前稿；确认后到正确历史位置", async ({ page }) => {
  const state = fixture();
  await open(page, state, "/app");
  await page.getByRole("link", { name: /雾潮剧情工程/u }).click();
  await page.getByRole("button", { name: "项目与文档", exact: true }).click();
  await expect(timeline(page)).toHaveValue("main");
  await editAll(page);
  await confirmation(page, () => page.goBack(), false);
  await expect(page).toHaveURL(new RegExp(`${documentsPath}$`, "u"));
  await expectAllDraft(page);
  await confirmation(page, () => page.goBack(), true);
  await expect(page).toHaveURL(new RegExp(`${checkPath}$`, "u"));
  await page.goForward();
  await expect(timeline(page)).toHaveValue("main");
  // Leave a forward entry, then modify the source form before traversing to it.
  await page.getByRole("button", { name: "文稿校验", exact: true }).click();
  await page.goBack();
  await expect(timeline(page)).toHaveValue("main");
  await editAll(page);
  await confirmation(page, () => page.goForward(), false);
  await expect(page).toHaveURL(new RegExp(`${documentsPath}$`, "u"));
  await expectAllDraft(page);
  await confirmation(page, () => page.goForward(), true);
  await expect(page).toHaveURL(new RegExp(`${checkPath}$`, "u"));
  expect(state.saves).toEqual([]);
  assertIsolated(state);
});

test("失败后重新读取也须确认；取消不请求或丢稿，确认读取最新版本", async ({ page }) => {
  const state = fixture();
  state.saveFailure = true;
  await open(page, state);
  await editAll(page);
  await editor(page).getByRole("button", { name: "保存并确认", exact: true }).click();
  await expect(editor(page).getByRole("alert")).toContainText("保存结果待核对");
  const reads = state.contextReads.length;
  const reread = editor(page).getByRole("button", { name: "重新读取", exact: true });
  await confirmation(page, () => reread.click(), false);
  await expectAllDraft(page);
  expect(state.contextReads).toHaveLength(reads);
  state.contextMalformed = true;
  await confirmation(page, () => reread.click(), true);
  await expect(editor(page).getByRole("alert")).toContainText("无法读取最新资料身份");
  await expectAllDraft(page);
  state.contextMalformed = false;
  state.documents[0].narrative_context = { ...context("server_main"), revision: 3 };
  await confirmation(page, () => reread.click(), true);
  await expect(timeline(page)).toHaveValue("server_main");
  await expect(role(page)).toHaveValue("reference");
  await expect(confirmed(page)).not.toBeChecked();
  expect(state.contextReads).toHaveLength(reads + 2);
  assertIsolated(state);
});

test("刷新确实触发原生beforeunload；取消保输入，确认刷新不自动保存", async ({ page }) => {
  const state = fixture();
  await open(page, state);
  await editAll(page);
  const firstDialog = page.waitForEvent("dialog", { timeout: 10_000 });
  const cancelledReload = page.reload({ timeout: 15_000 }).catch((error: unknown) => error);
  const cancel = await firstDialog;
  expect(cancel.type(), "必须观察到浏览器真实beforeunload，不能用Mock确认替代").toBe("beforeunload");
  await cancel.dismiss();
  await cancelledReload;
  await expectAllDraft(page);
  const secondDialog = page.waitForEvent("dialog", { timeout: 10_000 });
  const acceptedReload = page.reload({ timeout: 15_000 });
  const accept = await secondDialog;
  expect(accept.type()).toBe("beforeunload");
  await accept.accept();
  await acceptedReload;
  await expect(timeline(page)).toHaveValue("main");
  expect(state.saves).toEqual([]);
  assertIsolated(state);
});

test("保存错误保留草稿与离开保护；375px键盘焦点和字段可阅读", async ({ page }) => {
  const state = fixture();
  state.saveFailure = true;
  await page.setViewportSize({ width: 375, height: 812 });
  await open(page, state);
  await editAll(page);
  await editor(page).getByRole("button", { name: "保存并确认", exact: true }).click();
  await expect(editor(page).getByRole("alert")).toContainText("保存结果待核对");
  await expect(editor(page).getByRole("alert")).toBeFocused();
  await expectAllDraft(page);
  await capture(page, "context-editor-mobile-375.png");
  await confirmation(page, () => documentButton(page).click(), false);
  await expectAllDraft(page);
  expect(state.saves).toHaveLength(1);
  expect(state.documents[0].narrative_context.resolution_state).toBe("unresolved");
  assertIsolated(state);
});

test("手工未保存修改启动AI前确认；取消不写入，确认只生成待确认建议", async ({ page }) => {
  const state = fixture();
  await open(page, state);
  await editAll(page);
  const infer = editor(page).getByRole("button", { name: "AI 识别资料", exact: true });
  await confirmation(page, () => infer.click(), false);
  expect(state.inferences).toEqual([]);
  await expectAllDraft(page);
  await confirmation(page, () => infer.click(), true);
  await expect(timeline(page)).toHaveValue("inferred_main");
  await expect(editor(page).getByRole("region", { name: "为什么这样识别", exact: true })).toBeVisible();
  await expect(confirmed(page)).not.toBeChecked();
  expect(state.inferences).toEqual([{ documentId, payload: { expected_revision: 2 } }]);
  expect(state.saves).toEqual([]);
  expect(state.documents[0].narrative_context.resolution_state).toBe("inferred");
  assertIsolated(state);
});

test("真实上传引起同项目refresh保稿；新版本或资料消失不把旧表单套给新版", async ({ page }) => {
  const state = fixture();
  await open(page, state);
  await editAll(page);
  async function upload(name: string, stillCurrent = true) {
    await page.locator("#document-upload").setInputFiles({ name, mimeType: "text/markdown", buffer: Buffer.from("原创补充资料。") });
    await page.getByRole("button", { name: "导入待处理文稿", exact: true }).click();
    const queueRow = page.getByRole("region", { name: "文稿导入队列", exact: true }).getByRole("listitem")
      .filter({ has: page.locator(".importQueueIdentity").getByText(name, { exact: true }) });
    await expect(queueRow).toContainText("已导入");
    if (stillCurrent) await expect(timeline(page)).toBeEnabled();
  }
  state.documentsUnavailable = true;
  await upload("补充参考.md");
  const refreshWarning = page.getByRole("status", { name: "项目资料刷新未完成", exact: true });
  await expect(refreshWarning).toContainText("未保存资料身份修改保持在本页");
  await expectAllDraft(page);
  state.documentsUnavailable = false;
  await refreshWarning.getByRole("button", { name: "重新加载项目资料", exact: true }).click();
  await expect(documentButton(page, "补充参考.md")).toBeVisible();
  await expect(refreshWarning).not.toBeVisible();
  await expectAllDraft(page);
  await upload("01-活动草稿.md", false);
  await expectAllDraft(page);
  await expect(editor(page)).toContainText(/旧版本|不再是活动|已被替换|已移除|当前活动版本/u);
  await expect(editor(page).getByRole("button", { name: "保存并确认", exact: true })).toBeDisabled();
  state.dropOldOnUpload = true;
  // A later catalog refresh may omit the older ID entirely; the retained form
  // must still identify its original source instead of silently becoming v2.
  await page.locator("#document-upload").setInputFiles({ name: "清单更新.md", mimeType: "text/markdown", buffer: Buffer.from("原创目录刷新资料。") });
  await page.getByRole("button", { name: "导入待处理文稿", exact: true }).click();
  await expect.poll(() => state.uploads).toHaveLength(3);
  await expectAllDraft(page);
  await expect(editor(page).getByRole("button", { name: "保存并确认", exact: true })).toBeDisabled();
  expect(state.saves).toEqual([]);
  expect(state.documents.find((item) => item.name === "01-活动草稿.md" && item.active)?.narrative_context.scope.timeline_key).toBe("main");
  assertIsolated(state);
});

test("旧资料读取晚于新选择返回，不覆盖新资料或作者刚填的时间线", async ({ page }) => {
  const state = fixture();
  state.delayedDocument = documentId;
  await open(page, state);
  await expect.poll(() => typeof state.releaseDelayed).toBe("function");
  await documentButton(page).click();
  await expect(timeline(page)).toHaveValue("side");
  await timeline(page).fill("new_author_side");
  state.releaseDelayed!();
  await expect.poll(() => state.delayedFulfilled).toBe(true);
  await expect(timeline(page)).toHaveValue("new_author_side");
  await expect(documentButton(page)).toHaveAttribute("aria-current", "true");
  expect(state.saves).toEqual([]);
  assertIsolated(state);
});
