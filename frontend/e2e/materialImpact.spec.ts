import { expect, test, type Page } from "@playwright/test";
import { fulfillLegacyRunCatalog } from "./materialImpactFixture";

const projectId = "impact-story", otherProject = "impact-other-story", runId = "impact-report", otherRun = "impact-other-report";
const now = "2026-11-10T00:00:00Z";
const doc = (id: string, name: string, role = "canon") => ({ id, project_id: projectId, name, version: 2, active: true, created_at: now,
  document_role: role, story_scope: "main", context_explicit: true,
  narrative_context: { revision: 1, resolution_state: "confirmed", publication_status: "published", origin: "explicit", scope: { schema_version: 1, timeline_key: "main" } } });
const document = doc("setting-v2", "人物资料.md", "character_profile");
const source = (version: number) => ({ document_id: `setting-v${version}`, document_name: document.name, document_version: version,
  document_role: "character_profile", story_scope: "main", batch_role: "background", publication_status: "published" });
const run = (id = runId, project = projectId) => ({ id, project_id: project, status: "completed", progress: 100, created_at: now, completed_at: now,
  prompt_tokens: 0, completion_tokens: 0, estimated_cost_usd: 0, input_snapshot_available: true,
  input_documents: [{ ...source(1), content_sha256: "a".repeat(64), ordinal: 0 }],
  review_batch: { mode: "full_review", sensitivity: "balanced", character_review_enabled: false } });
const issue = { id: "impact-issue", category: "fact_conflict", severity: "high", confidence: 1, title: "核对旧设定的明确引用",
  explanation: "旧判断仅供核对，不因资料更新自动改写。", suggestion: "查阅依据。", evidence: [{ document_id: "setting-v1", document_name: document.name, line_start: 1, line_end: 1, text: "林澈守在北门。" }], extra: {} };
type Mode = "changed" | "unchanged" | "unselected" | "identity";
function impact(mode: Mode, project = projectId, id = runId) {
  const changed = mode === "changed" || mode === "identity";
  const items = [{ key: "frozen-setting", kind: "input", frozen: source(1), current: mode === "changed" ? source(2) : source(1),
    changes: mode === "changed" ? ["content_changed", "version_changed", "document_replaced"] : mode === "identity" ? ["identity_changed"] : [],
    selected_for_recheck: true, content_changed: mode === "changed", identity_comparison_available: true }];
  if (mode === "identity") items[0].current = { ...source(1), document_role: "reference", publication_status: "retired", story_scope: "branch-b" };
  if (mode === "unselected") items.push({ key: "unselected-draft", kind: "not_included", frozen: null as any,
    current: { ...source(1), document_id: "new-draft", document_name: "未选新稿.md", document_role: "chapter", publication_status: "draft" },
    changes: [], selected_for_recheck: false, content_changed: null as any, identity_comparison_available: true });
  return { schema_version: "material-change-impact-v1", project_id: project, run_id: id, comparison_status: "complete",
    semantic_impact_complete: false, confirmed_traits_auto_invalidated: false, confirmed_traits_changed: false,
    recheck: { inputs_changed: changed, can_prepare: changed, reason: changed ? null : "no_document_changes" },
    sources: { items, offset: 0, limit: 30, total: items.length, has_more: false },
    affected_issues: { items: changed ? [{ id: issue.id, title: issue.title, report_class: "formal", source_keys: ["frozen-setting"], evidence_indices: [0] }] : [], offset: 0, limit: 30, total: changed ? 1 : 0, has_more: false },
    summary: { changed_sources: changed ? 1 : 0, added_sources: 0, not_included_sources: mode === "unselected" ? 1 : 0,
      affected_issues: changed ? 1 : 0, unresolved_evidence: 0, unresolved_sources: 0 } };
}
async function fixture(page: Page, mode: Mode = "changed") {
  const state = { mode, failures: 0, requests: [] as string[], writes: [] as string[], unexpected: [] as string[], delayed: false, release: undefined as (() => void) | undefined };
  await page.route("**/api/v1/**", async route => {
    const url = new URL(route.request().url()), path = url.pathname, method = route.request().method();
    state.requests.push(`${method} ${path}`); if (method !== "GET") state.writes.push(`${method} ${path}`);
    let body: unknown;
    if (await fulfillLegacyRunCatalog(route, [{ projectId, runs: [run()] }, { projectId: otherProject, runs: [run(otherRun, otherProject)] }])) return;
    if (path === "/api/v1/auth/me") body = { mode: "anonymous", user: { id: "impact-author", email: "", display_name: "作者" }, workspace: { id: "impact-space", name: "原创空间", kind: "personal", role: "owner" } };
    else if (path === "/api/v1/account/model-provider") body = { configured: false, revision: 0 };
    else if (path === "/api/v1/project-catalog") {
      const all = [projectId, otherProject].map(id => ({ id, name: id === projectId ? "原创主线" : "独立故事", description: "", created_at: now, active_document_count: 1, latest_run: run(id === projectId ? runId : otherRun, id) }));
      const items = all.filter(i => !url.searchParams.get("project_id") || i.id === url.searchParams.get("project_id")); body = { page: 1, page_size: 40, total: items.length, items };
    } else if ([projectId, otherProject].some(id => path === `/api/v1/projects/${id}/documents`)) body = [{ ...document, project_id: path.split("/")[4] }];
    else if ([projectId, otherProject].some(id => path === `/api/v1/projects/${id}/analysis-runs`)) body = [path.includes(otherProject) ? run(otherRun, otherProject) : run()];
    else if (path === `/api/v1/projects/${projectId}/documents/${document.id}/narrative-context` || path === `/api/v1/projects/${otherProject}/documents/${document.id}/narrative-context`) body = { document_id: document.id, current: document.narrative_context, revision_count: 1 };
    else if (path === `/api/v1/issues/${issue.id}/feedback`) body = { latest: null };
    else if ([runId, otherRun].some(id => path === `/api/v1/analysis-runs/${id}/material-impact`)) {
      if (state.failures) { state.failures--; await route.fulfill({ status: 503, json: { detail: "暂时不可用" } }); return; }
      const other = path.includes(otherRun);
      if (state.delayed && !other) await new Promise<void>(resolve => { state.release = resolve; });
      body = impact(other ? "unchanged" : state.mode, other ? otherProject : projectId, other ? otherRun : runId);
    } else if ([runId, otherRun].some(id => path === `/api/v1/analysis-runs/${id}`)) body = path.includes(otherRun) ? run(otherRun, otherProject) : run();
    else if ([runId, otherRun].some(id => path === `/api/v1/analysis-runs/${id}/issues`)) body = [issue];
    else if ([runId, otherRun].some(id => path === `/api/v1/analysis-runs/${id}/records`)) body = { records: [], warnings: [] };
    else if ([runId, otherRun].some(id => path === `/api/v1/analysis-runs/${id}/diagnostics`)) body = {};
    else if ([runId, otherRun].some(id => path === `/api/v1/analysis-runs/${id}/clarifications`)) body = [];
    else if ([runId, otherRun].some(id => path === `/api/v1/analysis-runs/${id}/review-clues`)) body = { items: [], truncated: false, unavailable_count: 0 };
    else if ([runId, otherRun].some(id => path === `/api/v1/analysis-runs/${id}/provisional-clues`)) body = { items: [], truncated: false };
    else if ([runId, otherRun].some(id => path === `/api/v1/analysis-runs/${id}/evidence-preview`)) { await route.fulfill({ status: 409, json: { detail: { code: "snapshot_unavailable", message: "本交互夹具未提供全文；真实冻结阅读另由QA副本验证。" } } }); return; }
    else { state.unexpected.push(`${method} ${path}`); await route.fulfill({ status: 404, json: { detail: "unexpected fixture endpoint" } }); return; }
    await route.fulfill({ json: body });
  });
  return state;
}
const report = `/app/projects/${projectId}/runs/${runId}/report`;
test("变化引用进入当前问题与复检准备，桌面和窄屏均不自动分析", async ({ page }) => {
  const state = await fixture(page);
  await page.goto(report);
  const panel = page.getByRole("region", { name: "资料变化提示" });
  await expect(panel).toContainText("审查输入已有变化");
  await panel.locator("summary").click();
  await expect(panel).toContainText("人物资料.md · v1"); await expect(panel).toContainText("人物资料.md · v2");
  await panel.getByRole("link", { name: issue.title, exact: true }).click();
  await expect(page).toHaveURL(/issue=impact-issue/);
  await panel.getByRole("link", { name: "准备复检", exact: true }).click();
  await expect(page).toHaveURL(/revise\?step=run/);
  await expect(page.getByRole("button", { name: "开始复检", exact: true })).toBeEnabled();
  await expect(panel).toContainText("可能使用模型额度");
  await page.setViewportSize({ width: 375, height: 900 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  expect(state.writes).toEqual([]); expect(state.unexpected).toEqual([]);
});
test("未选草稿、身份变化和无变化分别表达，不能把零引用当完整无影响", async ({ page }) => {
  const state = await fixture(page, "unselected");
  for (const mode of ["unselected", "identity", "unchanged"] as const) {
    state.mode = mode; await page.goto(report);
    const panel = page.getByRole("region", { name: "资料变化提示" });
    await expect(panel.locator("summary")).toBeVisible(); await panel.locator("summary").click();
    if (mode === "unselected") { await expect(panel).toContainText("尚未纳入本次输入的资料"); await expect(panel).toContainText("未在当前复检选材中"); }
    if (mode === "identity") {
      await expect(panel).toContainText("资料身份或适用范围变化");
      await expect(panel).toContainText("角色档案");
      await expect(panel).toContainText("已归档");
      await expect(panel).toContainText("branch-b");
      await expect(panel).not.toContainText("用途未记录");
    }
    if (mode === "unchanged") await expect(panel).toContainText("未发现明确的输入变化");
    await expect(panel).toContainText("未找到直接引用不等于没有其他影响");
    await expect(panel.getByRole("link", { name: "准备复检", exact: true })).toHaveCount(mode === "identity" ? 1 : 0);
  }
  expect(state.writes).toEqual([]); expect(state.unexpected).toEqual([]);
});
test("读取失败可重试；跨项目迟到变化不能覆盖当前报告或开放复检", async ({ page }) => {
  const state = await fixture(page); state.failures = 1;
  await page.goto(report);
  const panel = page.getByRole("region", { name: "资料变化提示" });
  await expect(panel.getByRole("alert")).toContainText("资料变化未能读完");
  await expect(panel.getByRole("link", { name: "准备复检" })).toHaveCount(0);
  await panel.getByRole("button", { name: "重新核对" }).click(); await expect(panel).toContainText("审查输入已有变化");
  state.delayed = true; await panel.getByRole("button", { name: "重新核对" }).click();
  await expect.poll(() => !!state.release).toBe(true);
  await page.goto(`/app/projects/${otherProject}/runs/${otherRun}/report`);
  await expect(panel).toContainText("未发现明确的输入变化");
  state.release?.(); await expect(panel).not.toContainText("审查输入已有变化");
  await expect(panel.getByRole("link", { name: "准备复检" })).toHaveCount(0);
  expect(state.writes).toEqual([]); expect(state.unexpected).toEqual([]);
});
