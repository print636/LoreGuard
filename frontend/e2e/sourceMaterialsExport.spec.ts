import { expect, test, type Page } from "@playwright/test";
import { fulfillLegacyMaterialImpact, fulfillLegacyRunCatalog } from "./materialImpactFixture";

const project = "source-export-project", first = "source-export-one", second = "source-export-two";
const now = "2026-11-10T00:00:00Z";
const input = (id: string, name: string) => ({ document_id: id, document_name: name,
  document_version: 1, document_role: "chapter", story_scope: "main", batch_role: "target", content_sha256: "a".repeat(64) });
const run = (id = first) => ({ id, project_id: project, status: "completed", created_at: now, completed_at: now,
  input_snapshot_available: true, input_documents: id === first ? [input("draft-v1", "原稿.md"), input("profile-v1", "设定.md")] : [input("other-v1", "另一稿.md")],
  prompt_tokens: 0, completion_tokens: 0, estimated_cost_usd: 0,
  review_batch: { mode: "draft_review", sensitivity: "balanced", character_review_enabled: false } });
const routePath = (id = first) => `/app/projects/${project}/runs/${id}/report`;

async function fixture(page: Page) {
  const state = { status: 409, authenticated: true, delayed: false, release: undefined as (() => void) | undefined,
    settled: false, requests: [] as string[], zipSelections: [] as string[][],
    writes: [] as string[], unexpected: [] as string[], downloads: 0 };
  page.on("download", () => { state.downloads++; });
  await page.route("**/api/v1/**", async route => {
    const url = new URL(route.request().url()), p = url.pathname, method = route.request().method();
    state.requests.push(`${method} ${p}${url.search}`);
    if (method !== "GET") state.writes.push(`${method} ${p}`);
    if (await fulfillLegacyMaterialImpact(route, [first, second].map(runId => ({ projectId: project, runId })))) return;
    if (await fulfillLegacyRunCatalog(route, [{ projectId: project, runs: [run(), run(second)] }])) return;
    let body: unknown;
    if (p === "/api/v1/auth/me") {
      if (!state.authenticated) { await route.fulfill({ status: 401, json: { detail: "session expired" } }); return; }
      body = { mode: "required", user: { id: "qa", email: "qa@example.test", display_name: "作者" }, workspace: { id: "qa-space", name: "作品", kind: "personal", role: "owner" } };
    }
    else if (p === "/api/v1/account/model-provider") body = { configured: false, revision: 0 };
    else if (p === "/api/v1/project-catalog") body = { page: 1, page_size: 40, total: 1, items: [{ id: project, name: "资料导出", description: "", created_at: now, active_document_count: 0, latest_run: run() }] };
    else if (p === `/api/v1/projects/${project}/documents`) body = [];
    else if (p === `/api/v1/projects/${project}/analysis-runs`) body = [run(), run(second)];
    else {
      const id = [first, second].find(id => p === `/api/v1/analysis-runs/${id}` || p.startsWith(`/api/v1/analysis-runs/${id}/`));
      const suffix = id ? p.slice(`/api/v1/analysis-runs/${id}`.length) : "invalid";
      if (id && !suffix) body = run(id);
      else if (id && suffix === "/issues") body = [];
      else if (id && suffix === "/records") body = { records: [], warnings: [] };
      else if (id && suffix === "/diagnostics") body = {};
      else if (id && suffix === "/clarifications") body = [];
      else if (id && suffix === "/review-clues") body = { items: [], truncated: false, unavailable_count: 0 };
      else if (id && suffix === "/provisional-clues") body = { items: [], truncated: false };
      else if (id && suffix === "/character-observations") body = { schema_version: "character-observations-v1", project_id: project, run_id: id,
        recording: { status: "complete", observation_count: 0 }, page: { offset: 0, limit: 50, total_items: 0, has_more: false }, items: [] };
      else if (id && suffix === "/source-materials.zip" && method === "GET") {
        expect([...url.searchParams.keys()].every(key => key === "document_id")).toBe(true);
        const selected = url.searchParams.getAll("document_id");
        expect(selected.length).toBeGreaterThan(0);
        expect(selected.every(value => run(id).input_documents.some(row => row.document_id === value))).toBe(true);
        state.zipSelections.push(selected);
        if (state.delayed) await new Promise<void>(resolve => { state.release = resolve; });
        try {
          if (state.status === 200) await route.fulfill({ status: 200, contentType: "application/zip", body: Buffer.from("PK\x05\x06" + "\0".repeat(18), "binary") });
          else { if (state.status === 401) state.authenticated = false;
            await route.fulfill({ status: state.status, json: { detail: { code: "source_materials_unavailable", documents: [{ name: "原稿.md", reason: "content_mismatch" }] } } }); }
        } catch { /* A scope change may have cancelled the old request. */ }
        state.settled = true;
        return;
      } else { state.unexpected.push(`${method} ${p}`); await route.fulfill({ status: 404, json: { detail: "unexpected endpoint" } }); return; }
    }
    await route.fulfill({ json: body });
  });
  return state;
}

async function open(page: Page) {
  await page.goto(routePath());
  const panel = page.locator("details.sourceMaterialsExport");
  await panel.locator("summary").click();
  return panel;
}

test("空选不下载，缺源错误保留选中并可明确重试，375宽不溢出", async ({ page }) => {
  const state = await fixture(page); const panel = await open(page);
  await expect(panel.getByRole("button", { name: "下载所选资料（0份）", exact: true })).toBeDisabled();
  expect(state.zipSelections).toEqual([]);
  const chosen = panel.getByRole("checkbox", { name: "原稿.md · v1", exact: true });
  await chosen.check();
  await panel.getByRole("button", { name: "下载所选资料（1份）", exact: true }).click();
  await expect(panel.getByRole("alert")).toContainText("本次未生成压缩包");
  await expect(chosen).toBeChecked();
  await expect(panel.getByRole("checkbox", { name: "设定.md · v1", exact: true })).not.toBeChecked();
  expect(state.downloads).toBe(0);
  await page.setViewportSize({ width: 375, height: 900 });
  await expect(panel.getByRole("alert")).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  state.status = 200;
  const download = page.waitForEvent("download");
  await panel.getByRole("button", { name: "下载所选资料（1份）", exact: true }).click();
  await download; await expect(panel.getByRole("status")).toContainText("没有导出未勾选资料");
  expect(state.zipSelections).toEqual([["draft-v1"], ["draft-v1"]]);
  expect(state.writes).toEqual([]); expect(state.unexpected).toEqual([]);
});

test("切换报告清空选择并取消迟到下载，不将旧包用于新报告", async ({ page }) => {
  const state = await fixture(page); state.status = 200; state.delayed = true;
  const panel = await open(page);
  await panel.getByRole("checkbox", { name: "原稿.md · v1", exact: true }).check();
  await panel.getByRole("button", { name: "下载所选资料（1份）", exact: true }).click();
  await expect.poll(() => !!state.release).toBe(true);
  await page.evaluate(url => { history.pushState({}, "", url); window.dispatchEvent(new PopStateEvent("popstate")); }, routePath(second));
  const next = page.locator("details.sourceMaterialsExport");
  await next.locator("summary").click();
  await expect(next.getByRole("checkbox", { name: "另一稿.md · v1", exact: true })).not.toBeChecked();
  await expect(next.getByRole("button", { name: "下载所选资料（0份）", exact: true })).toBeDisabled();
  state.release?.(); await expect.poll(() => state.settled).toBe(true);
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  expect(state.downloads).toBe(0); expect(state.zipSelections).toEqual([["draft-v1"]]);
  expect(state.writes).toEqual([]); expect(state.unexpected).toEqual([]);
});

test("会话失效不下载错误响应也不自动重试分析或资料请求", async ({ page }) => {
  const state = await fixture(page); state.status = 401;
  const panel = await open(page);
  await panel.getByRole("checkbox", { name: "原稿.md · v1", exact: true }).check();
  await panel.getByRole("button", { name: "下载所选资料（1份）", exact: true }).click();
  await expect(page).toHaveURL(/\/login\?/);
  await expect(page.getByText("登录状态已失效，请重新登录后继续。登录不会自动重试刚才的操作。", { exact: true })).toBeVisible();
  expect(state.downloads).toBe(0); expect(state.zipSelections).toEqual([["draft-v1"]]);
  expect(state.writes).toEqual([]); expect(state.unexpected).toEqual([]);
});
