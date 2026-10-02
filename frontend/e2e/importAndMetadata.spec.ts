import path from "node:path";
import { expect, test, type Locator, type Page, type Request } from "@playwright/test";

test.setTimeout(45_000);

// All stories and accounts below are synthetic UI fixtures, not evaluation cases.
// The catch-all route rejects any API request not explicitly handled here.
const projectId = "d6f543c8-7459-4b68-861e-8201571a2d32";
const otherProjectId = "0e5b3025-a79d-454a-b5a1-c01a2a34b414";
const createdProjectId = "49b769b7-7032-44fb-8f14-6a89e7fd4083";
const runId = "8f3db188-13a8-4978-99d7-ed8d2b8f150d";
const now = "2026-10-02T00:00:00Z";
const screenshotFolder = process.env.LOREGUARD_IMPORT_METADATA_SCREENSHOT_DIR
  ?? process.env.LOREGUARD_E2E_SCREENSHOT_DIR
  ?? "../artifacts/import-and-metadata";

type Role = "chapter" | "canon" | "character_profile" | "reference";
type Project = {
  id: string; name: string; description: string; created_at: string; metadata_revision: number;
};
type Document = {
  id: string; project_id: string; name: string; version: number; active: boolean;
  document_role: Role; story_scope: string; created_at: string;
  narrative_context: {
    revision: number; resolution_state: "unresolved"; origin: "explicit";
    publication_status: "unknown";
    scope: { schema_version: 1; timeline_key: string };
  };
};
type Upload = {
  projectId: string; key: string; name: string; content: string; role: Role;
  scope: string; replaceId: string; signature: string;
};
type UploadResult = "success" | "failure" | "unknown-committed";

const frozenRun = {
  id: runId, project_id: projectId, status: "completed", created_at: now,
  completed_at: now, prompt_tokens: 0, completion_tokens: 0, estimated_cost_usd: 0,
  input_snapshot_available: true, input_documents: [{
    document_id: "source-chapter", document_name: "已存章节.md", document_version: 1,
    document_role: "chapter", story_scope: "global", content_sha256: "a".repeat(64), batch_role: "target",
  }],
};

function document(name: string, role: Role, id: string, owner = projectId): Document {
  return {
    id, project_id: owner, name, version: 1, active: true, document_role: role,
    story_scope: "global", created_at: now,
    narrative_context: {
      revision: 1, resolution_state: "unresolved", origin: "explicit",
      publication_status: "unknown", scope: { schema_version: 1, timeline_key: "main" },
    },
  };
}

function stateFixture() {
  return {
    projects: [
      { id: projectId, name: "C-雾潮主线", description: "游戏剧情策划的独立工作区", created_at: now, metadata_revision: 3 },
      { id: otherProjectId, name: "B-青灯活动", description: "另一份故事", created_at: now, metadata_revision: 1 },
    ] as Project[],
    documents: [document("已存章节.md", "chapter", "source-chapter")],
    uploads: [] as Upload[],
    uploadPlans: new Map<string, UploadResult[]>(),
    receipts: new Map<string, { signature: string; doc: Document }>(),
    creations: [] as { name: string; description: string }[],
    patches: [] as { name: string; description: string; expected_revision: number }[],
    requests: [] as string[],
    unexpected: [] as string[],
    createUnknown: false,
    catalogFailuresAfterUpload: 0,
    metadataReadUnavailable: false,
    metadataConflict: false,
    metadataSaveFailure: false,
    metadataDelay: undefined as (() => Promise<void>) | undefined,
  };
}
type State = ReturnType<typeof stateFixture>;

function file(name: string, content = `${name}的原创叙事文本。`) {
  return { name, mimeType: "text/markdown", buffer: Buffer.from(content, "utf8") };
}

function multipart(request: Request, owner: string): Upload {
  const contentType = request.headers()["content-type"] ?? "";
  const boundary = /boundary=(?:"([^"]+)"|([^;]+))/u.exec(contentType);
  if (!boundary) throw new Error("Fixture expected multipart upload");
  const parts = (request.postDataBuffer()?.toString("utf8") ?? "").split(`--${boundary[1] ?? boundary[2]}`);
  const fields = new Map<string, string>();
  let name = "";
  let content = "";
  for (const part of parts) {
    const separator = part.indexOf("\r\n\r\n");
    if (separator < 0) continue;
    const header = part.slice(0, separator);
    const value = part.slice(separator + 4).replace(/\r\n$/u, "");
    const field = /\bname="([^"]+)"/u.exec(header)?.[1];
    if (field === "file") {
      name = /\bfilename="([^"]+)"/u.exec(header)?.[1] ?? "";
      content = value;
    } else if (field) fields.set(field, value);
  }
  const role = (fields.get("document_role") ?? "chapter") as Role;
  const scope = fields.get("story_scope") ?? "global";
  const replaceId = fields.get("replace_document_id") ?? "";
  return {
    projectId: owner, key: request.headers()["idempotency-key"] ?? "", name, content, role, scope, replaceId,
    signature: JSON.stringify([owner, name, content, role, scope, replaceId]),
  };
}

function catalogItem(project: Project, state: State) {
  return {
    ...project, active_document_count: state.documents.filter((doc) => doc.project_id === project.id && doc.active).length,
    latest_run: project.id === projectId ? frozenRun : null,
  };
}

async function mockApi(page: Page, state: State) {
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const endpoint = url.pathname;
    const method = request.method();
    state.requests.push(`${method} ${endpoint}${url.search}`);
    let body: unknown;

    if (method === "GET" && endpoint === "/api/v1/auth/me") body = {
      mode: "anonymous", user: { id: "import-author", email: "", display_name: "剧情策划" },
      workspace: { id: "import-workspace", name: "原创剧情工作区", kind: "personal", role: "owner" },
    };
    else if (method === "GET" && endpoint === "/api/v1/account/model-provider") body = { configured: false, revision: 0 };
    else if (method === "GET" && endpoint === "/api/v1/project-catalog") {
      if (state.uploads.length > 0 && state.catalogFailuresAfterUpload > 0) {
        state.catalogFailuresAfterUpload -= 1;
        await route.fulfill({ status: 503, json: { detail: "目录暂时无法读取" } });
        return;
      }
      const query = (url.searchParams.get("query") ?? "").trim();
      const owner = url.searchParams.get("project_id");
      let projects = state.projects.filter((project) => (!owner || project.id === owner)
        && (!query || `${project.name} ${project.description}`.includes(query)));
      if (url.searchParams.get("sort") === "name") projects = projects.slice().sort((a, b) => a.name.localeCompare(b.name, "zh-CN"));
      const pageNumber = Number(url.searchParams.get("page") ?? 1);
      const pageSize = Number(url.searchParams.get("page_size") ?? 40);
      body = { page: pageNumber, page_size: pageSize, total: projects.length,
        items: projects.slice((pageNumber - 1) * pageSize, pageNumber * pageSize).map((project) => catalogItem(project, state)) };
    }
    else if (method === "GET" && endpoint === "/api/v1/projects") body = state.projects.map((project) => catalogItem(project, state));
    else if (method === "POST" && endpoint === "/api/v1/projects") {
      const payload = request.postDataJSON() as { name: string; description: string };
      state.creations.push(payload);
      const created = { ...payload, id: createdProjectId, created_at: now, metadata_revision: 1 };
      state.projects.push(created);
      if (state.createUnknown) { await route.abort("failed"); return; }
      await route.fulfill({ status: 201, json: created });
      return;
    }
    else if (/^\/api\/v1\/projects\/[^/]+\/metadata$/u.test(endpoint)) {
      const owner = endpoint.split("/")[4];
      const project = state.projects.find((item) => item.id === owner);
      if (!project) { await route.fulfill({ status: 404, json: { detail: "项目不存在" } }); return; }
      if (method === "GET") {
        if (state.metadataDelay) await state.metadataDelay();
        if (state.metadataReadUnavailable) {
          await route.fulfill({ status: 503, json: { detail: "项目信息暂时无法读取" } });
          return;
        }
        body = project;
      } else if (method === "PATCH") {
        const payload = request.postDataJSON() as State["patches"][number];
        state.patches.push(payload);
        if (state.metadataConflict) {
          state.metadataConflict = false;
          project.name = "C-另一页更新的故事名";
          project.description = "另一页保存的说明";
          project.metadata_revision += 1;
          await route.fulfill({ status: 409, json: { detail: {
            reason_code: "project_metadata_revision_conflict", message: "项目信息已更新，请读取最新版本后再决定。",
          } } });
          return;
        }
        if (state.metadataSaveFailure) {
          await route.fulfill({ status: 503, json: { detail: "本次保存结果暂无法确认" } });
          return;
        }
        if (payload.expected_revision !== project.metadata_revision) {
          await route.fulfill({ status: 409, json: { detail: {
            reason_code: "project_metadata_revision_conflict", message: "项目信息已更新，请重新读取。",
          } } });
          return;
        }
        project.name = payload.name.trim();
        project.description = payload.description;
        project.metadata_revision += 1;
        body = project;
      } else { state.unexpected.push(`${method} ${endpoint}`); await route.fulfill({ status: 404, json: { detail: "Unexpected fixture metadata method" } }); return; }
    }
    else if (/^\/api\/v1\/projects\/[^/]+\/documents$/u.test(endpoint)) {
      const owner = endpoint.split("/")[4];
      if (method === "GET") body = state.documents.filter((doc) => doc.project_id === owner);
      else if (method === "POST") {
        const upload = multipart(request, owner);
        state.uploads.push(upload);
        const receipt = state.receipts.get(upload.key);
        if (receipt) {
          if (receipt.signature !== upload.signature) {
            await route.fulfill({ status: 409, json: { detail: { reason_code: "document_upload_idempotency_conflict", message: "同一导入记录不能对应不同内容" } } });
          } else await route.fulfill({ status: 201, json: { ...receipt.doc, superseded_document_ids: [], deduplicated: true } });
          return;
        }
        const result = state.uploadPlans.get(upload.name)?.shift() ?? "success";
        if (result === "failure") {
          await route.fulfill({ status: 422, json: { detail: "文稿格式无法导入，请核对文件后重试" } });
          return;
        }
        const created = document(upload.name, upload.role, `uploaded-${state.documents.length + 1}`, owner);
        created.story_scope = upload.scope;
        state.documents.push(created);
        state.receipts.set(upload.key, { signature: upload.signature, doc: created });
        if (result === "unknown-committed") { await route.abort("failed"); return; }
        await route.fulfill({ status: 201, json: { ...created, superseded_document_ids: [], deduplicated: false } });
        return;
      } else { state.unexpected.push(`${method} ${endpoint}`); await route.fulfill({ status: 404, json: { detail: "Unexpected fixture document method" } }); return; }
    }
    else if (method === "GET" && /^\/api\/v1\/projects\/[^/]+\/analysis-runs$/u.test(endpoint)) body = endpoint.includes(projectId) ? [frozenRun] : [];
    else if (method === "GET" && endpoint === `/api/v1/projects/${projectId}/run-catalog`) body = {
      project_id: projectId, page: 1, page_size: 20, total: 1, has_more: false,
      items: [{ ...frozenRun, started_at: now, input_chars: 100, frozen_document_count: 1,
        retried_from: null, batch_mode: "draft_review" }],
    };
    else if (method === "GET" && endpoint.endsWith("/narrative-context")) {
      const selected = state.documents.find((doc) => doc.id === endpoint.split("/").at(-2));
      if (!selected) { state.unexpected.push(`${method} ${endpoint}`); await route.fulfill({ status: 404, json: { detail: "Unexpected fixture context" } }); return; }
      body = { document_id: selected.id, current: selected.narrative_context, revision_count: 1 };
    }
    else if (method === "GET" && endpoint === `/api/v1/analysis-runs/${runId}`) body = frozenRun;
    else if (method === "GET" && ["issues", "clarifications"].some((suffix) => endpoint === `/api/v1/analysis-runs/${runId}/${suffix}`)) body = [];
    else if (method === "GET" && endpoint === `/api/v1/analysis-runs/${runId}/records`) body = { records: [], warnings: [] };
    else if (method === "GET" && endpoint === `/api/v1/analysis-runs/${runId}/diagnostics`) body = {};
    else if (method === "GET" && ["review-clues", "provisional-clues"].some((suffix) => endpoint === `/api/v1/analysis-runs/${runId}/${suffix}`)) body = { items: [], truncated: false, unavailable_count: 0 };
    else {
      state.unexpected.push(`${method} ${endpoint}`);
      await route.fulfill({ status: 404, json: { detail: "Unexpected import and metadata fixture endpoint" } });
      return;
    }
    await route.fulfill({ json: body });
  });
}

async function open(page: Page, state: State, location: string) {
  await mockApi(page, state);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(location);
  await page.waitForLoadState("networkidle");
}

function queue(page: Page) { return page.getByRole("region", { name: "文稿导入队列", exact: true }); }
function row(page: Page, name: string) {
  return queue(page).getByRole("listitem").filter({
    has: page.locator(".importQueueIdentity").getByText(name, { exact: true }),
  });
}
function metadata(page: Page) { return page.getByRole("dialog", { name: "编辑项目信息", exact: true }); }

async function selectWorkspaceFiles(page: Page, files: ReturnType<typeof file>[]) {
  await page.locator("#document-upload").setInputFiles(files);
  await expect(queue(page)).toBeVisible();
}

async function selectCenterFiles(page: Page, name: string, files: ReturnType<typeof file>[]) {
  await page.getByRole("button", { name: /^导入已有故事/u }).first().click();
  const form = page.locator(".quickEntry");
  await form.getByLabel("项目名称", { exact: true }).fill(name);
  await form.locator('input[type="file"]').setInputFiles(files);
  await expect(queue(page)).toBeVisible();
}

async function assertNoOtherWrites(state: State) {
  expect(state.unexpected, "未列出的API不得越过Mock请求真实后端").toEqual([]);
  expect(state.requests.filter((request) => /^(POST|PATCH|PUT|DELETE) /u.test(request)
    && !/^POST \/api\/v1\/projects(?:$|\/[^/]+\/documents$)/u.test(request)
    && !/^PATCH \/api\/v1\/projects\/[^/]+\/metadata$/u.test(request)),
  "导入/项目描述不得启动分析、模型、角色确认、资料发布或反馈").toEqual([]);
}

async function assertFits(page: Page, surface: Locator) {
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), "手机/桌面不能横向溢出").toBe(true);
  expect(await surface.evaluate((element) => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
}

async function captureQueueViewport(page: Page, name: string) {
  // The workbench scrolls inside .mainCanvas. A tall locator screenshot includes
  // areas clipped by that scrollport, so record the actual visible viewport.
  await queue(page).evaluate((element) => element.scrollIntoView({ block: "start", inline: "nearest", behavior: "instant" }));
  await page.screenshot({ path: path.join(screenshotFolder, name) });
}

test("工作台：混合资料部分成功后仅重试失败文件，并保留逐文件类型", async ({ page }) => {
  const state = stateFixture();
  state.uploadPlans.set("02-新章节.md", ["failure", "success"]);
  await page.setViewportSize({ width: 1440, height: 960 });
  await open(page, state, `/app/projects/${projectId}/documents`);
  await selectWorkspaceFiles(page, [file("01-世界观.md"), file("02-新章节.md"), file("03-角色资料.md")]);
  await page.getByLabel("01-世界观.md 的文档类型", { exact: true }).selectOption("canon");
  await page.getByLabel("02-新章节.md 的文档类型", { exact: true }).selectOption("chapter");
  await page.getByLabel("03-角色资料.md 的文档类型", { exact: true }).selectOption("character_profile");
  await page.getByRole("button", { name: "导入待处理文稿", exact: true }).click();
  await expect(row(page, "01-世界观.md")).toContainText("已导入");
  await expect(row(page, "02-新章节.md")).toContainText("导入失败");
  await expect(row(page, "03-角色资料.md")).toContainText("已导入");
  expect(state.uploads.map((upload) => upload.role)).toEqual(["canon", "chapter", "character_profile"]);
  await assertFits(page, queue(page));
  await captureQueueViewport(page, "import-desktop-1440.png");
  await page.getByRole("button", { name: "导入待处理文稿", exact: true }).click();
  await expect(row(page, "02-新章节.md")).toContainText("已导入");
  expect(state.uploads.map((upload) => upload.name)).toEqual(["01-世界观.md", "02-新章节.md", "03-角色资料.md", "02-新章节.md"]);
  expect(state.documents.filter((doc) => doc.name === "01-世界观.md")).toHaveLength(1);
  expect(state.documents.filter((doc) => doc.name === "03-角色资料.md")).toHaveLength(1);
  await assertNoOtherWrites(state);
});

test("工作台：响应丢失保留待核对，手动安全重试复用同一key和完整payload", async ({ page }) => {
  const state = stateFixture();
  state.uploadPlans.set("章节响应丢失.md", ["unknown-committed"]);
  await open(page, state, `/app/projects/${projectId}/documents`);
  await selectWorkspaceFiles(page, [file("章节响应丢失.md", "不会因为重试产生第二份的原创章节。")]);
  await page.getByLabel("章节响应丢失.md 的文档类型", { exact: true }).selectOption("chapter");
  await page.getByLabel("章节响应丢失.md 的故事作用域", { exact: true }).fill("主线");
  await page.getByRole("button", { name: "导入待处理文稿", exact: true }).click();
  await expect(row(page, "章节响应丢失.md")).toContainText("结果待核对");
  expect(state.uploads).toHaveLength(1);
  await expect(page.getByLabel("章节响应丢失.md 的文档类型", { exact: true })).toBeDisabled();
  await row(page, "章节响应丢失.md").getByRole("button", { name: "安全重试同一导入", exact: true }).click();
  await expect(row(page, "章节响应丢失.md")).toContainText("已导入");
  expect(state.uploads).toHaveLength(2);
  expect(state.uploads[1].key).toBe(state.uploads[0].key);
  expect(state.uploads[1].signature).toBe(state.uploads[0].signature);
  expect(state.uploads[0].key).toMatch(/^[0-9a-f-]{36}$/iu);
  expect(state.documents.filter((doc) => doc.name === "章节响应丢失.md")).toHaveLength(1);
  await assertNoOtherWrites(state);
});

test("工作台：已收到成功回执后目录刷新失败不能误报上传失败或再次上传", async ({ page }) => {
  const state = stateFixture();
  state.catalogFailuresAfterUpload = 2;
  await open(page, state, `/app/projects/${projectId}/documents`);
  await selectWorkspaceFiles(page, [file("已导入但目录刷新失败.md")]);
  await page.getByRole("button", { name: "导入待处理文稿", exact: true }).click();
  await expect(row(page, "已导入但目录刷新失败.md")).toContainText("已导入");
  await expect(row(page, "已导入但目录刷新失败.md")).not.toContainText("导入失败");
  await expect(row(page, "已导入但目录刷新失败.md").getByRole("button", { name: "安全重试同一导入", exact: true })).toHaveCount(0);
  expect(state.uploads).toHaveLength(1);
  await assertNoOtherWrites(state);
});

test("项目中心：部分导入失败保留创建ID，继续只处理未成功文件", async ({ page }) => {
  const state = stateFixture();
  state.uploadPlans.set("新稿二.md", ["failure", "success"]);
  state.uploadPlans.set("新稿三.md", ["unknown-committed"]);
  await open(page, state, "/app");
  await selectCenterFiles(page, "作者新故事", [file("设定一.md"), file("新稿二.md"), file("新稿三.md")]);
  await page.getByLabel("设定一.md 的文档类型", { exact: true }).selectOption("canon");
  await page.getByLabel("新稿二.md 的文档类型", { exact: true }).selectOption("chapter");
  await page.getByLabel("新稿三.md 的文档类型", { exact: true }).selectOption("chapter");
  await page.getByRole("button", { name: "创建并导入", exact: true }).click();
  await expect(row(page, "新稿二.md")).toContainText("导入失败");
  await expect(row(page, "设定一.md")).toContainText("已导入");
  await expect(row(page, "新稿三.md")).toContainText("结果待核对");
  expect(state.creations).toHaveLength(1);
  const originalUnknown = state.uploads.find((upload) => upload.name === "新稿三.md")!;
  const checkDocuments = row(page, "新稿三.md").getByRole("link", { name: "在新标签页核对项目文档", exact: true });
  await expect(checkDocuments).toHaveAttribute("href", `/app/projects/${createdProjectId}/documents`);
  await expect(checkDocuments).toHaveAttribute("target", "_blank");
  await expect(checkDocuments).toHaveAttribute("rel", "noopener noreferrer");
  // Do not open an un-intercepted popup; attributes and the retained queue are sufficient here.
  expect(new URL(page.url()).pathname).toBe("/app");
  await page.getByRole("button", { name: "继续导入待处理文稿", exact: true }).click();
  await expect(row(page, "新稿二.md")).toContainText("已导入");
  await expect(row(page, "新稿三.md")).toContainText("结果待核对");
  expect(state.uploads.filter((upload) => upload.name === "新稿三.md")).toHaveLength(1);
  await row(page, "新稿三.md").getByRole("button", { name: "安全重试同一导入", exact: true }).click();
  await expect(row(page, "新稿三.md")).toContainText("已导入");
  expect(state.documents.filter((doc) => doc.project_id === createdProjectId).map((doc) => doc.name).sort()).toEqual(["新稿三.md", "新稿二.md", "设定一.md"].sort());
  expect(state.creations).toHaveLength(1);
  expect(state.uploads.filter((upload) => upload.name === "设定一.md")).toHaveLength(1);
  const retriedUnknown = state.uploads.filter((upload) => upload.name === "新稿三.md")[1];
  expect(retriedUnknown.key).toBe(originalUnknown.key);
  expect(retriedUnknown.signature).toBe(originalUnknown.signature);
  expect(state.uploads.every((upload) => upload.projectId === createdProjectId)).toBe(true);
  await assertNoOtherWrites(state);
});

test("项目中心：创建响应丢失不得自动重建项目或上传到未知项目", async ({ page }) => {
  const state = stateFixture();
  state.createUnknown = true;
  await open(page, state, "/app");
  await selectCenterFiles(page, "创建结果待核对的故事", [file("待导入章节.md")]);
  await page.getByRole("button", { name: "创建并导入", exact: true }).click();
  await expect(page.locator(".quickEntry")).toContainText("创建结果待核对");
  expect(state.creations).toHaveLength(1);
  expect(state.uploads).toHaveLength(0);
  await expect(page.getByRole("button", { name: "创建并导入", exact: true })).toBeDisabled();
  const checkCatalog = page.locator(".quickEntry").getByRole("link", { name: "在新标签页核对项目目录", exact: true });
  await expect(checkCatalog).toHaveAttribute("href", "/app");
  await expect(checkCatalog).toHaveAttribute("target", "_blank");
  await expect(checkCatalog).toHaveAttribute("rel", "noopener noreferrer");
  await expect(row(page, "待导入章节.md")).toContainText("等待导入");
  expect(new URL(page.url()).pathname).toBe("/app");
  await assertNoOtherWrites(state);
});

test("项目信息：加载失败不凭列表旧值保存，重读后显示正确revision", async ({ page }) => {
  const state = stateFixture();
  state.metadataReadUnavailable = true;
  await open(page, state, "/app");
  await page.getByRole("button", { name: "编辑C-雾潮主线的项目信息", exact: true }).click();
  await expect(metadata(page)).toContainText("无法读取");
  await expect(metadata(page).getByRole("button", { name: "保存项目信息", exact: true })).toBeDisabled();
  expect(state.patches).toHaveLength(0);
  state.metadataReadUnavailable = false;
  await metadata(page).getByRole("button", { name: /重新读取|重试读取|重试/u }).click();
  await expect(metadata(page).getByLabel("项目名称", { exact: true })).toHaveValue("C-雾潮主线");
  await metadata(page).getByRole("textbox", { name: "项目说明", exact: true }).fill("从正确读取的版本保存说明");
  await metadata(page).getByRole("button", { name: "保存项目信息", exact: true }).click();
  await expect(metadata(page)).toContainText("项目信息已保存");
  await metadata(page).getByRole("button", { name: "取消", exact: true }).click();
  await expect(metadata(page)).not.toBeVisible();
  expect(state.patches[0].expected_revision).toBe(3);
  await assertNoOtherWrites(state);
});

test("项目信息：409保留作者修改，显式合并未修改字段后仍由作者明确保存", async ({ page }) => {
  const state = stateFixture();
  state.metadataConflict = true;
  await open(page, state, `/app/projects/${projectId}/documents`);
  const originalUrl = page.url();
  await page.getByRole("button", { name: "编辑项目信息", exact: true }).click();
  await metadata(page).getByLabel("项目名称", { exact: true }).fill("A-作者保留的故事名");
  await metadata(page).getByRole("button", { name: "保存项目信息", exact: true }).click();
  await expect(metadata(page)).toContainText(/已更新|最新/u);
  await expect(metadata(page).getByLabel("项目名称", { exact: true })).toHaveValue("A-作者保留的故事名");
  await expect(metadata(page).getByRole("textbox", { name: "项目说明", exact: true })).toHaveValue("游戏剧情策划的独立工作区");
  await metadata(page).getByRole("button", { name: /读取最新|重新读取/u }).click();
  await expect(metadata(page)).toContainText("C-另一页更新的故事名");
  await expect(metadata(page).getByLabel("项目名称", { exact: true })).toHaveValue("A-作者保留的故事名");
  await expect(metadata(page).getByRole("textbox", { name: "项目说明", exact: true })).toHaveValue("游戏剧情策划的独立工作区");
  expect(state.patches).toHaveLength(1);
  await metadata(page).getByRole("button", { name: "保留我的改动，合并最新内容", exact: true }).click();
  await expect(metadata(page).getByLabel("项目名称", { exact: true })).toHaveValue("A-作者保留的故事名");
  await expect(metadata(page).getByRole("textbox", { name: "项目说明", exact: true })).toHaveValue("另一页保存的说明");
  expect(state.patches).toHaveLength(1);
  await metadata(page).getByRole("button", { name: "保存项目信息", exact: true }).click();
  await expect(metadata(page)).toContainText("项目信息已保存");
  await metadata(page).getByRole("button", { name: "取消", exact: true }).click();
  await expect(metadata(page)).not.toBeVisible();
  expect(state.patches.map((patch) => patch.expected_revision)).toEqual([3, 4]);
  expect(state.patches[1].description).toBe("另一页保存的说明");
  expect(page.url()).toBe(originalUrl);
  await expect(page.locator(".topContext > button").first()).toContainText("A-作者保留的故事名");
  await page.getByRole("button", { name: "完整报告", exact: true }).click();
  await expect(page).toHaveURL(new RegExp(`/runs/${runId}/report`, "u"));
  expect(state.documents[0].narrative_context.resolution_state).toBe("unresolved");
  await assertNoOtherWrites(state);
});

test("项目信息：保存错误保留输入，Escape/取消须确认放弃修改并返回入口焦点", async ({ page }) => {
  const state = stateFixture();
  state.metadataSaveFailure = true;
  await open(page, state, "/app");
  const trigger = page.getByRole("button", { name: "编辑C-雾潮主线的项目信息", exact: true });
  await trigger.click();
  await metadata(page).getByRole("textbox", { name: "项目说明", exact: true }).fill("未保存的故事备注");
  await metadata(page).getByRole("button", { name: "保存项目信息", exact: true }).click();
  await expect(metadata(page)).toContainText(/结果待核对|结果.*确认|保存.*失败|暂.*保存/u);
  await expect(metadata(page).getByRole("textbox", { name: "项目说明", exact: true })).toHaveValue("未保存的故事备注");
  page.once("dialog", (dialog) => dialog.dismiss());
  await page.keyboard.press("Escape");
  await expect(metadata(page)).toBeVisible();
  await expect(metadata(page).getByRole("textbox", { name: "项目说明", exact: true })).toHaveValue("未保存的故事备注");
  page.once("dialog", (dialog) => dialog.accept());
  await metadata(page).getByRole("button", { name: "取消", exact: true }).click();
  await expect(metadata(page)).not.toBeVisible();
  await expect(trigger).toBeFocused();
  expect(state.projects[0].description).toBe("游戏剧情策划的独立工作区");
  expect(state.patches).toHaveLength(1);
  await assertNoOtherWrites(state);
});

test("项目信息：保存后目录名称排序、说明搜索和当前项目名称同步", async ({ page }) => {
  const state = stateFixture();
  await page.setViewportSize({ width: 1440, height: 960 });
  await open(page, state, "/app?sort=name");
  await page.getByRole("button", { name: "编辑C-雾潮主线的项目信息", exact: true }).click();
  await metadata(page).getByLabel("项目名称", { exact: true }).fill("A-更新后的雾潮主线");
  await metadata(page).getByRole("textbox", { name: "项目说明", exact: true }).fill("策划发布批次说明可被搜索");
  await metadata(page).screenshot({ path: path.join(screenshotFolder, "metadata-desktop-1440.png") });
  await metadata(page).getByRole("button", { name: "保存项目信息", exact: true }).click();
  await expect(metadata(page)).toContainText("项目信息已保存");
  await metadata(page).getByRole("button", { name: "取消", exact: true }).click();
  await expect(metadata(page)).not.toBeVisible();
  const rows = page.locator(".projectList > li");
  await expect(rows.first()).toContainText("A-更新后的雾潮主线");
  await page.getByRole("searchbox", { name: /搜索项目/u }).fill("策划发布批次");
  await expect(rows).toHaveCount(1);
  await expect(rows.first()).toContainText("A-更新后的雾潮主线");
  expect(state.patches).toHaveLength(1);
  await assertNoOtherWrites(state);
});

test("移动端375：失败队列与长项目说明可阅读，键盘操作及关闭入口不溢出", async ({ page }) => {
  const state = stateFixture();
  state.uploadPlans.set("章节-作者修订的活动开场与转折.md", ["failure"]);
  await page.setViewportSize({ width: 375, height: 812 });
  await open(page, state, `/app/projects/${projectId}/documents`);
  await selectWorkspaceFiles(page, [file("章节-作者修订的活动开场与转折.md"), file("世界观补充.md")]);
  await page.getByRole("button", { name: "导入待处理文稿", exact: true }).click();
  await expect(row(page, "章节-作者修订的活动开场与转折.md")).toContainText("导入失败");
  await assertFits(page, queue(page));
  await captureQueueViewport(page, "import-mobile-375.png");
  await row(page, "世界观补充.md").evaluate((element) => element.scrollIntoView({ block: "end", inline: "nearest", behavior: "instant" }));
  await page.screenshot({ path: path.join(screenshotFolder, "import-mobile-375-bottom.png") });
  await page.getByRole("button", { name: "编辑项目信息", exact: true }).click();
  const nameField = metadata(page).getByLabel("项目名称", { exact: true });
  await expect(nameField).toBeFocused();
  await metadata(page).getByRole("textbox", { name: "项目说明", exact: true }).fill("用于策划跨版本集中审查的项目说明。".repeat(12));
  await assertFits(page, metadata(page));
  await expect(metadata(page).getByRole("button", { name: "取消", exact: true })).toBeInViewport();
  await metadata(page).screenshot({ path: path.join(screenshotFolder, "metadata-mobile-375.png") });
  page.once("dialog", (dialog) => dialog.dismiss());
  await page.keyboard.press("Escape");
  await expect(metadata(page)).toBeVisible();
  page.once("dialog", (dialog) => dialog.accept());
  await metadata(page).getByRole("button", { name: "取消", exact: true }).click();
  expect(state.patches).toHaveLength(0);
  await assertNoOtherWrites(state);
});
