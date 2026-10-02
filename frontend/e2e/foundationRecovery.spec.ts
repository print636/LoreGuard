import path from "node:path";
import { expect, test, type Dialog, type Page, type Route } from "@playwright/test";

test.setTimeout(45_000);

// Ordinary synthetic author workflows only: no live API, provider, story corpus,
// evaluation answers or user account data is used by this browser batch.
const projectA = "foundation-author-project";
const projectB = "foundation-second-project";
const runId = "foundation-baseline-run";
const chapterId = "foundation-chapter-one";
const profileId = "foundation-profile-one";
const characterKey = "资料角色";
const firstCandidate = "foundation-candidate-one";
const secondCandidate = "foundation-candidate-two";
const thirdCandidate = "foundation-candidate-three";
const now = "2026-10-03T00:00:00Z";
const root = `/api/v1/projects/${projectA}`;
const characterRoot = `${root}/characters/${characterKey}`;
const revisePath = `/app/projects/${projectA}/runs/${runId}/revise?step=upload&document=${chapterId}`;
const candidatePath = `/app/projects/${projectA}/characters?section=candidates&page=1&character=${encodeURIComponent(characterKey)}&candidate=${firstCandidate}`;
const screenshotFolder = process.env.LOREGUARD_FOUNDATION_SCREENSHOT_DIR
  ?? process.env.LOREGUARD_E2E_SCREENSHOT_DIR
  ?? "../artifacts/foundation-recovery";

function makeDocument(id: string, name: string, version = 1, projectId = projectA, role = "chapter") {
  return { id, project_id: projectId, name, version, active: true, created_at: now,
    content: role === "character_profile" ? "资料角色处理共同事务时，先征询同伴意见。" : "原创章节，作者正在修订叙事细节。",
    document_role: role, story_scope: "global", context_explicit: true,
    narrative_context: { revision: 1, resolution_state: "confirmed", origin: "explicit",
      publication_status: role === "character_profile" ? "published" : "draft",
      scope: { schema_version: 1, timeline_key: "main" }, scope_sha256: "c".repeat(64) } };
}

function baseline() {
  return { id: runId, project_id: projectA, status: "completed", mode: "baseline_build", progress: 100,
    prompt_tokens: 0, completion_tokens: 0, estimated_cost_usd: 0, created_at: now, completed_at: now,
    input_snapshot_available: true,
    input_documents: [makeDocument(chapterId, "章节一.md"), makeDocument(profileId, "角色资料.md", 1, projectA, "character_profile")].map((item, ordinal) => ({
      document_id: item.id, document_name: item.name, document_version: 1, document_role: item.document_role,
      story_scope: "global", content_sha256: "a".repeat(64), ordinal,
      batch_role: item.document_role === "chapter" ? "target" : "background",
      narrative_context: { context_revision: 1, resolution_state: "confirmed",
        publication_status: item.narrative_context.publication_status, scope_sha256: "c".repeat(64) },
    })) };
}

function fixture() {
  return { documents: [makeDocument(chapterId, "章节一.md"), makeDocument(profileId, "角色资料.md", 1, projectA, "character_profile")],
    requests: [] as string[], unexpected: [] as string[],
    uploads: [] as { key: string; payload: string }[], receipts: new Map<string, ReturnType<typeof makeDocument>>(),
    uploadMode: "success" as "success" | "unknown" | "reject" | "delay",
    documentsUnavailable: false, delayedFulfilled: false,
    releaseDelayed: undefined as (() => void) | undefined,
    baselineHistoryPending: false, releaseBaselineHistory: undefined as (() => void) | undefined,
    decisions: [] as unknown[], decisionUnavailable: false, rejected: new Set<string>(), axesCreated: 0 };
}
type Fixture = ReturnType<typeof fixture>;

function candidate(state: Fixture, id: string) {
  const values: Record<string, string> = {
    [firstCandidate]: "共同事务中会先征询同伴意见",
    [secondCandidate]: "需要交接时会说明当前进度",
    [thirdCandidate]: "安排日程时会核对团队时间",
  };
  return { id, character_key: characterKey, trait_type: "core_personality", trait_key: `ordinary_${id}`,
    value: values[id], polarity: "positive", origin: "explicit_setting", confidence: 0.9,
    source_run_id: runId, scope_sha256: "synthetic-frozen-snapshot", revision: state.rejected.has(id) ? 2 : 1,
    review_state: state.rejected.has(id) ? "rejected" : "pending", reviewable: !state.rejected.has(id),
    model_coverage: "full", source_verified: true, contexts: ["共同事务"], limitations: [],
    approved_axis_id: null, approved_axis_version: null, axis_alignment: null, axis_polarity: null,
    axis_positive_proposition_sha256: null, support_bindings_status: "legacy", support_bindings_v1: null,
    evidence: [{ input_id: "foundation-profile-input", document_id: profileId, document_name: "角色资料.md",
      document_version: 1, document_role: "character_profile", publication_status: "published",
      authority_level: "canon", line_start: 1, line_end: 1,
      text: "资料角色处理共同事务时，先征询同伴意见。",
      source_verified: true, source_text_exact: true, context_verified: false }] };
}

function multipartPayload(route: Route): string {
  // Ignore only the random transport boundary, not any file bytes or form fields.
  const request = route.request();
  const boundary = /boundary=(?:"([^"]+)"|([^;]+))/u.exec(request.headers()["content-type"] ?? "");
  const token = boundary?.[1] ?? boundary?.[2];
  const payload = request.postDataBuffer()?.toString("utf8") ?? "";
  return token ? payload.split(token).join("<multipart-boundary>") : payload;
}

async function reply(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, json: body });
}

async function mockApi(page: Page, state: Fixture) {
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const endpoint = decodeURIComponent(url.pathname);
    const method = request.method();
    state.requests.push(`${method} ${endpoint}`);
    let body: unknown;
    if (method === "GET" && endpoint === "/api/v1/auth/me") body = {
      mode: "anonymous", user: { id: "foundation-author", email: "", display_name: "作者" },
      workspace: { id: "foundation-workspace", name: "原创工作区", kind: "personal", role: "owner" },
    };
    else if (method === "GET" && endpoint === "/api/v1/account/model-provider") body = { configured: false, revision: 0 };
    else if (method === "GET" && endpoint === "/api/v1/project-catalog") {
      const items = [
        { id: projectA, name: "原创修订工程", description: "普通作者恢复流程", created_at: now,
          active_document_count: state.documents.filter((item) => item.active).length, latest_run: baseline() },
        { id: projectB, name: "另一个原创工程", description: "独立资料", created_at: now,
          active_document_count: 1, latest_run: null },
      ].filter((item) => !url.searchParams.get("project_id") || item.id === url.searchParams.get("project_id"));
      body = { page: 1, page_size: 40, total: items.length, items };
    }
    else if (method === "GET" && endpoint === `${root}/documents`) {
      if (state.documentsUnavailable) { await reply(route, { detail: "项目资料暂时无法读取。" }, 503); return; }
      body = state.documents;
    }
    else if (method === "GET" && endpoint === `/api/v1/projects/${projectB}/documents`) body = [makeDocument("second-project-document", "独立正文.md", 1, projectB)];
    else if (method === "GET" && endpoint === `${root}/analysis-runs`) {
      if (state.baselineHistoryPending) await new Promise<void>((resolve) => { state.releaseBaselineHistory = resolve; });
      body = [baseline()];
    }
    else if (method === "GET" && endpoint === `/api/v1/projects/${projectB}/analysis-runs`) body = [];
    else if (method === "GET" && endpoint === `/api/v1/analysis-runs/${runId}`) body = baseline();
    else if (method === "GET" && endpoint.startsWith(`/api/v1/analysis-runs/${runId}/`)) {
      const tail = endpoint.split("/").at(-1);
      if (tail === "issues" || tail === "clarifications") body = [];
      else if (tail === "records") body = { records: [], warnings: [] };
      else if (tail === "diagnostics") body = {};
      else if (tail === "provisional-clues") body = { items: [], truncated: false };
      else if (tail === "review-clues") body = { items: [], truncated: false, unavailable_count: 0 };
      else { state.unexpected.push(`${method} ${endpoint}`); await reply(route, { detail: "Unknown isolated run endpoint" }, 404); return; }
    }
    else if (method === "POST" && endpoint === `${root}/documents`) {
      const key = request.headers()["idempotency-key"] ?? "";
      expect(key, "修订上传必须具有可复用的操作收据键").not.toBe("");
      const payload = multipartPayload(route);
      state.uploads.push({ key, payload });
      if (state.uploadMode === "reject") { state.uploadMode = "success"; await reply(route, { detail: "请核对所选资料类型。" }, 422); return; }
      const previous = state.receipts.get(key);
      const saved = previous ?? makeDocument(`foundation-version-${state.receipts.size + 1}`, "章节一.md", 2);
      if (!previous) {
        const old = state.documents.find((item) => item.id === chapterId);
        if (old) old.active = false;
        state.documents.push(saved);
        state.receipts.set(key, saved);
      } else expect(payload).toBe(state.uploads.find((item) => item.key === key)!.payload);
      if (state.uploadMode === "unknown") { state.uploadMode = "success"; await route.abort("failed"); return; }
      if (state.uploadMode === "delay") await new Promise<void>((resolve) => { state.releaseDelayed = resolve; });
      try { await reply(route, { ...saved, superseded_document_ids: [chapterId], deduplicated: Boolean(previous) }, 201); }
      catch { /* A cancelled obsolete transport is allowed; the synthetic receipt is already stored. */ }
      if (state.uploadMode === "delay") state.delayedFulfilled = true;
      return;
    }
    else if (method === "GET" && endpoint === `${root}/characters`) body = {
      items: [{ id: characterKey, canonical_name: characterKey, aliases: [], confirmed_item_count: 0,
        pending_candidate_count: 3 - state.rejected.size, drift_issue_count: 0, profile_revision: 1 }],
      page: 1, page_size: 30, total: 1, has_more: false, readiness: "ready", model_coverage: "full", source_run_id: runId,
    };
    else if (method === "GET" && endpoint === characterRoot) body = {
      character: { id: characterKey, canonical_name: characterKey, aliases: [], confirmed_item_count: 0,
        pending_candidate_count: 3 - state.rejected.size, drift_issue_count: 0 },
      profile_items: [], model_coverage: "full", source_run_id: runId,
    };
    else if (method === "GET" && endpoint === `${characterRoot}/profile-candidates`) {
      const pageNumber = Number(url.searchParams.get("page") || 1);
      const ids = pageNumber === 1 ? [firstCandidate, secondCandidate] : [thirdCandidate];
      const items = url.searchParams.get("state") === "withdrawn" ? [] : ids.filter((id) => !state.rejected.has(id)).map((id) => candidate(state, id));
      body = { items, page: pageNumber, page_size: 2, total: 3 - state.rejected.size, has_more: pageNumber === 1,
        model_coverage: "full" };
    }
    else if (method === "GET" && endpoint === `${root}/character-trait-axes`) body = { items: [], total: 0, limit: 100, offset: 0 };
    else if (method === "GET" && endpoint.startsWith(`${characterRoot}/profile-candidates/`)) {
      const candidateId = endpoint.slice(`${characterRoot}/profile-candidates/`.length).split("/")[0];
      if (![firstCandidate, secondCandidate, thirdCandidate].includes(candidateId)) { state.unexpected.push(`${method} ${endpoint}`); await reply(route, { detail: "Unknown synthetic candidate" }, 404); return; }
      if (endpoint === `${characterRoot}/profile-candidates/${candidateId}/source-neighbors`) body = { candidate_id: candidateId, character_key: characterKey,
        source_run_id: runId, limit: 20, offset: 0, total: 0, has_more: false, items: [], source_groups: [] };
      else if (endpoint === `${characterRoot}/profile-candidates/${candidateId}`) body = candidate(state, candidateId);
      else { state.unexpected.push(`${method} ${endpoint}`); await reply(route, { detail: "Unknown synthetic candidate endpoint" }, 404); return; }
    }
    else if (method === "POST" && endpoint.endsWith("/decisions") && endpoint.startsWith(`${characterRoot}/profile-candidates/`)) {
      const candidateId = endpoint.slice(`${characterRoot}/profile-candidates/`.length).split("/")[0];
      if (![firstCandidate, secondCandidate, thirdCandidate].includes(candidateId) || endpoint !== `${characterRoot}/profile-candidates/${candidateId}/decisions`) {
        state.unexpected.push(`${method} ${endpoint}`); await reply(route, { detail: "Unknown synthetic decision endpoint" }, 404); return;
      }
      const payload = request.postDataJSON();
      state.decisions.push(payload);
      if (state.decisionUnavailable) { await reply(route, { detail: "归纳审核暂时无法完成。" }, 422); return; }
      expect(payload.decision).toBe("reject");
      state.rejected.add(candidateId);
      body = { candidate: candidate(state, candidateId), decision_id: "foundation-explicit-decision", deduplicated: false };
    }
    else if (method === "GET" && ([chapterId, profileId].some((id) => endpoint === `${root}/documents/${id}/narrative-context`)
      || endpoint === `/api/v1/projects/${projectB}/documents/second-project-document/narrative-context`)) {
      const id = endpoint.split("/").at(-2);
      const item = state.documents.find((entry) => entry.id === id) ?? makeDocument("second-project-document", "独立正文.md", 1, projectB);
      body = { document_id: item.id, current: item.narrative_context, revision_count: 1 };
    }
    else {
      state.unexpected.push(`${method} ${endpoint}`);
      await reply(route, { detail: "Unrecognized isolated foundation fixture endpoint" }, 404);
      return;
    }
    await reply(route, body);
  });
}

async function open(page: Page, state: Fixture, location = revisePath) {
  await mockApi(page, state);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(location);
  if (location === revisePath) {
    await expect(page.getByRole("heading", { name: "上传修订后的文档", exact: true })).toBeVisible();
    // The upload shell appears before its exact frozen baseline has finished
    // restoring. Choosing a file during that identity transition is not ready.
    if (!state.baselineHistoryPending) {
      await expect(page.locator(`.revisionHeader code[title="${runId}"]`)).toBeVisible();
      await expect(page.locator(".revisionContext")).toContainText("2 份");
    }
  }
  else await expect(note(page)).toBeEditable();
}

async function selectRevision(page: Page, text = "原创章节修订内容。") {
  await revisionFile(page).setInputFiles({ name: "章节一.md", mimeType: "text/markdown", buffer: Buffer.from(text) });
  await expect.poll(() => revisionFile(page).evaluate((element) => (element as HTMLInputElement).files?.[0]?.name ?? "")).toBe("章节一.md");
}

function revisionFile(page: Page) { return page.getByRole("button", { name: /^修订稿文件/u }); }
function note(page: Page) { return page.getByRole("textbox", { name: "审核备注（可选）", exact: true }); }
function candidateLink(page: Page, id = secondCandidate) { return page.locator(`a[href*="candidate=${id}"]`); }

async function confirmation(page: Page, action: () => Promise<unknown>, accept: boolean) {
  const seen: { type: string; message: string }[] = [];
  const handler = async (dialog: Dialog) => { seen.push({ type: dialog.type(), message: dialog.message() }); if (accept) await dialog.accept(); else await dialog.dismiss(); };
  page.once("dialog", handler);
  await action();
  page.off("dialog", handler);
  expect(seen, "作者须明确决定是否放弃未提交修改").toHaveLength(1);
  expect(seen[0].type).toBe("confirm");
  expect(seen[0].message).toMatch(/未提交|未保存|尚未|修改|改动/u);
}

function expectNoUnexpected(state: Fixture) { expect(state.unexpected).toEqual([]); }
function expectNoAnalysis(state: Fixture) { expect(state.requests.filter((item) => item.startsWith("POST ") && /analysis-runs|inference|model-provider/u.test(item))).toEqual([]); }

test("01 修订上传响应丢失后同键同内容手动核对，不生成重复版本", async ({ page }) => {
  const state = fixture(); state.uploadMode = "unknown";
  await open(page, state);
  await selectRevision(page);
  await page.getByRole("button", { name: "保存新版本", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("上传结果待核对");
  await expect(revisionFile(page)).toBeDisabled();
  expect(state.uploads).toHaveLength(1);
  await page.getByRole("button", { name: /安全重试同一/u }).click();
  await expect(page.getByRole("heading", { name: "确认变化并启动复检", exact: true })).toBeVisible();
  expect(state.uploads).toHaveLength(2);
  expect(state.uploads[1]).toEqual(state.uploads[0]);
  expect(state.receipts.size).toBe(1);
  await expect(page.getByRole("status").filter({ hasText: "没有新增文档版本" })).toBeVisible();
  expectNoAnalysis(state); expectNoUnexpected(state);
});

test("02 明确拒收后作者修改内容，新的请求使用新操作键", async ({ page }) => {
  const state = fixture(); state.uploadMode = "reject";
  await open(page, state); await selectRevision(page, "原始修订内容。");
  await page.getByRole("button", { name: "保存新版本", exact: true }).click();
  await expect(page.getByRole("alert")).toContainText("上传未完成");
  await expect(revisionFile(page)).toBeEnabled();
  await selectRevision(page, "核对后修改的修订内容。");
  await page.getByRole("button", { name: "保存新版本", exact: true }).click();
  await expect(page.getByRole("heading", { name: "确认变化并启动复检", exact: true })).toBeVisible();
  expect(state.uploads).toHaveLength(2);
  expect(state.uploads[1].key).not.toBe(state.uploads[0].key);
  expect(state.uploads[1].payload).not.toBe(state.uploads[0].payload);
  expect(state.receipts.size).toBe(1);
  expectNoAnalysis(state); expectNoUnexpected(state);
});

test("03 文稿确已保存而资料刷新失败时，只重新读取，不重复上传", async ({ page }) => {
  const state = fixture(); state.baselineHistoryPending = true;
  await open(page, state);
  await expect.poll(() => Boolean(state.releaseBaselineHistory)).toBe(true);
  await expect(page.getByRole("status", { name: "复检基线尚未核验", exact: true })).toContainText("基线尚未核验或不可用于复检");
  await expect(revisionFile(page)).toBeDisabled();
  await expect(page.getByRole("button", { name: "保存新版本", exact: true })).toBeDisabled();
  await expect(page.getByRole("combobox", { name: "文档类型", exact: true })).toBeDisabled();
  await expect(page.getByRole("textbox", { name: "故事作用域", exact: true })).toBeDisabled();
  expect(state.uploads).toHaveLength(0);
  state.baselineHistoryPending = false;
  state.releaseBaselineHistory?.();
  await expect(page.locator(`.revisionHeader code[title="${runId}"]`)).toBeVisible();
  await expect(page.locator(".revisionContext")).toContainText("2 份");
  await expect(revisionFile(page)).toBeEnabled();
  await expect(page.getByRole("status", { name: "复检基线尚未核验", exact: true })).toHaveCount(0);
  await selectRevision(page);
  state.documentsUnavailable = true;
  await page.getByRole("button", { name: "保存新版本", exact: true }).click();
  await expect(page.getByRole("status").filter({ hasText: "新版本已保存，但项目资料刷新未完成" })).toBeVisible();
  expect(state.uploads).toHaveLength(1);
  state.documentsUnavailable = false;
  await page.getByRole("button", { name: "重新读取项目资料", exact: true }).click();
  await expect(page.getByRole("heading", { name: "确认变化并启动复检", exact: true })).toBeVisible();
  expect(state.uploads).toHaveLength(1);
  expect(state.receipts.size).toBe(1);
  expectNoAnalysis(state); expectNoUnexpected(state);
});

test("04 离开旧项目后迟到上传响应不更改新项目或当前审查基线", async ({ page }) => {
  const state = fixture(); state.uploadMode = "delay";
  await open(page, state); await selectRevision(page);
  await page.getByRole("button", { name: "保存新版本", exact: true }).click();
  await expect.poll(() => Boolean(state.releaseDelayed)).toBe(true);
  await page.getByRole("button", { name: "项目中心", exact: true }).click();
  await page.getByRole("link").filter({ hasText: "另一个原创工程" }).click();
  await expect(page).toHaveURL(new RegExp(`/app/projects/${projectB}/`));
  const readsBefore = state.requests.filter((item) => item === `GET ${root}/documents`).length;
  state.releaseDelayed?.();
  await expect.poll(() => state.delayedFulfilled).toBe(true);
  await expect(page).toHaveURL(new RegExp(`/app/projects/${projectB}/`));
  await expect(page.getByText("另一个原创工程", { exact: true }).first()).toBeVisible();
  expect(state.requests.filter((item) => item === `GET ${root}/documents`)).toHaveLength(readsBefore);
  await expect(page.getByText("新版本已保存，项目资料已重新读取。", { exact: false })).toHaveCount(0);
  expectNoAnalysis(state); expectNoUnexpected(state);
});

test("05 动态工作台渲染失败使用无私密安全页面，键盘手动重载恢复原路径", async ({ page }) => {
  const state = fixture();
  const sentinel = "SYNTHETIC_PRIVATE_RENDER_FAILURE_9d67b";
  const logs: string[] = [];
  page.on("console", (message) => logs.push(message.text()));
  page.on("pageerror", (error) => logs.push(error.message));
  const chunk = /\/assets\/RevisionReview-[^/]+\.js(?:\?.*)?$/u;
  await page.route(chunk, (route) => route.fulfill({ contentType: "application/javascript", body: `export default function SyntheticFailure(){throw new Error(${JSON.stringify(sentinel)});}` }));
  await mockApi(page, state);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(revisePath);
  const heading = page.getByRole("heading", { name: "页面显示遇到问题", exact: true });
  await expect(heading).toBeVisible();
  await expect(heading).toBeFocused();
  await expect(page.getByRole("heading", { name: "重新加载前请注意", exact: true })).toBeVisible();
  await expect(page.locator("body")).not.toContainText(sentinel);
  expect(logs.join("\n")).not.toContain(sentinel);
  expect(state.requests.filter((item) => !item.startsWith("GET "))).toEqual([]);
  await page.screenshot({ path: path.join(screenshotFolder, "render-recovery-desktop-1440.png"), fullPage: false });
  await page.setViewportSize({ width: 375, height: 900 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: path.join(screenshotFolder, "render-recovery-mobile-375.png"), fullPage: false });
  await page.unroute(chunk);
  await heading.focus();
  await page.keyboard.press("Tab");
  await expect(page.getByRole("button", { name: "重新加载页面", exact: true })).toBeFocused();
  await page.keyboard.press("Enter");
  await expect(page.getByRole("heading", { name: "上传修订后的文档", exact: true })).toBeVisible();
  await expect(page).toHaveURL(new RegExp(`/app/projects/${projectA}/runs/${runId}/revise`));
  expect(state.requests.filter((item) => !item.startsWith("GET "))).toEqual([]);
  expect(logs.join("\n")).not.toContain(sentinel);
  expectNoUnexpected(state);
});

test("06 作者切换候选先决定是否放弃备注，取消保留、明确同意后丢弃本页草稿", async ({ page }) => {
  const state = fixture(); await open(page, state, candidatePath);
  await note(page).fill("待核对共同事务的情境边界。");
  await confirmation(page, () => candidateLink(page).click(), false);
  await expect(note(page)).toHaveValue("待核对共同事务的情境边界。");
  await expect(page).toHaveURL(new RegExp(`candidate=${firstCandidate}`));
  await confirmation(page, () => candidateLink(page).click(), true);
  await expect(page).toHaveURL(new RegExp(`candidate=${secondCandidate}`));
  await expect(note(page)).toHaveValue("");
  expect(state.decisions).toHaveLength(0); expect(state.axesCreated).toBe(0);
  expectNoUnexpected(state);
});

test("07 候选翻页和离开取消保备注，明确审核失败保稿，成功后解除离开保护", async ({ page }) => {
  const state = fixture(); await open(page, state, candidatePath);
  await note(page).fill("本次只驳回这条归纳，不修改其它设定。");
  await confirmation(page, () => page.getByRole("navigation", { name: "待确认归纳分页" }).getByRole("button", { name: "下一页", exact: true }).click(), false);
  await expect(page).toHaveURL(/page=1/u);
  await confirmation(page, () => page.getByRole("button", { name: "项目中心", exact: true }).click(), false);
  await expect(note(page)).toHaveValue("本次只驳回这条归纳，不修改其它设定。");
  state.decisionUnavailable = true;
  await page.getByRole("button", { name: "驳回归纳", exact: true }).click();
  await expect.poll(() => state.decisions.length).toBe(1);
  await expect(note(page)).toBeEditable();
  await expect(note(page)).toHaveValue("本次只驳回这条归纳，不修改其它设定。");
  state.decisionUnavailable = false;
  await page.getByRole("button", { name: "驳回归纳", exact: true }).click();
  await expect.poll(() => state.rejected.has(firstCandidate)).toBe(true);
  await expect(page.getByText("归纳已驳回，决定已保留在审核记录中。", { exact: true })).toBeVisible();
  await expect(page.getByRole("status", { name: "候选审核未提交输入", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "项目中心", exact: true })).toBeEnabled();
  const unexpectedDialogs: string[] = [];
  page.on("dialog", async (dialog) => { unexpectedDialogs.push(dialog.message()); await dialog.dismiss(); });
  await page.getByRole("button", { name: "项目中心", exact: true }).click();
  await expect(page).toHaveURL(/\/app$/u);
  expect(unexpectedDialogs).toEqual([]);
  expect(state.decisions).toHaveLength(2);
  expectNoUnexpected(state);
});

test("08 未创建的作者轴输入切换前有确认，取消保留，明确放弃不向服务器建轴", async ({ page }) => {
  const state = fixture(); await open(page, state, candidatePath);
  await page.getByRole("button", { name: "创建新轴", exact: true }).click();
  await page.getByRole("textbox", { name: "轴名称", exact: true }).fill("共同事务决策边界");
  await page.getByRole("textbox", { name: "轴定义", exact: true }).fill("是否未经同伴同意决定共同事务。");
  await page.getByRole("textbox", { name: "轴的正向命题", exact: true }).fill("角色未经同伴同意决定共同事务。");
  await confirmation(page, () => candidateLink(page).click(), false);
  await expect(page.getByRole("textbox", { name: "轴名称", exact: true })).toHaveValue("共同事务决策边界");
  await expect(page.getByRole("textbox", { name: "轴定义", exact: true })).toHaveValue("是否未经同伴同意决定共同事务。");
  await expect(page.getByRole("textbox", { name: "轴的正向命题", exact: true })).toHaveValue("角色未经同伴同意决定共同事务。");
  await confirmation(page, () => candidateLink(page).click(), true);
  await expect(page).toHaveURL(new RegExp(`candidate=${secondCandidate}`));
  expect(state.requests.filter((item) => item === `POST ${root}/character-trait-axes`)).toEqual([]);
  expect(state.decisions).toHaveLength(0);
  expectNoUnexpected(state);
});
