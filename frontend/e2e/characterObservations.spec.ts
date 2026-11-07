import { expect, test, type Page } from "@playwright/test";

const projectId = "observations-project", runId = "observations-old", nextRunId = "observations-next";
const now = "2026-11-04T00:00:00Z";
const screenshots = "../artifacts/playwright/screenshots/character-observations";
const source = { document_id: "draft-frozen", document_name: "北港新稿（本次冻结稿）.md", document_version: 3,
  line_start: 7, line_end: 8, text: "林霁收刀入鞘，没有伤害已经放下武器的人。\n她说：让他们走。" };
const run = (id = runId) => ({ id, project_id: projectId, status: "completed", created_at: now, prompt_tokens: 12, completion_tokens: 4, estimated_cost_usd: 0 });
const issue = { id: "formal-issue", category: "fact_conflict", severity: "medium", confidence: 0.8, title: "正式问题示例", explanation: "请核对两处事实。", suggestion: "核对正式设定。", evidence: [{ ...source }], metadata: {} };
const reviewClue = { id: "71475dcc-1d0c-4dd2-a014-06941adf1bf3", report_class: "review_clue", category: "character_drift", severity: "low", confidence: 0.5,
  title: "另一处表现需要作者确认", explanation: "一次不同表现尚不能独立定为冲突。", suggestion: "结合上下文确认。", evidence: [{ ...source, document_id: "profile-frozen", document_name: "角色设定.md", text: "她通常保持沉默。" }, { ...source }], metadata: { final_outcome: "needs_confirmation", review_reason: "single_behavior_is_not_drift" } };

function observationPayload(id = runId, status = "complete", total = 2, offset = 0) {
  return { schema_version: "character-observations-v1", project_id: projectId, run_id: id,
    recording: { status, observation_count: total }, page: { offset, limit: 50, total_items: total, has_more: offset + 50 < total },
    items: Array.from({ length: Math.min(50, Math.max(0, total - offset)) }, (_, index) => ({ ordinal: offset + index,
      observation_id: `cs_PRIVATE_OBSERVATION_${index}`, character: "林霁", dimension: "behavior_boundary", trait_key: "private_trait_key",
      statement: id === nextRunId ? "新运行记录：她留在门前。" : index === 0 ? "收刀后放走了已放下武器的人。" : "她在风里站了一会儿，没有回应远处传来的呼唤。".repeat(3),
      context: "对方已经放下武器。", source: { ...source, run_input_id: "PRIVATE_INPUT", content_sha256: "f".repeat(64) }, evidence: { ...source },
      matches: index === 0 ? [{ status: "matched", reason_code: "matched", baseline: { payload: {
        character_display_name: "林霁", trait_type: "behavior_boundary", trait_key: "private_trait_key", value: "不会伤害失去反抗能力的人。",
        evidence: [{ ...source, document_name: "角色设定（历史冻结）.md", document_version: 1, text: "林霁从不伤害失去反抗能力的人。" }],
        candidate_id: "PRIVATE_CANDIDATE", comparison_key: "PRIVATE_COMPARISON", raw_response: "PRIVATE_RAW_RESPONSE",
      } } }] : [{ status: "not_matched", reason_code: "no_confirmed_baseline", baseline: null }],
    })),
  };
}
type State = { status: string; total: number; failures: number; anonymous: boolean; delay: boolean; release?: () => void; settled: boolean; requests: string[]; unexpected: string[]; writes: string[] };
const initialState = (): State => ({ status: "complete", total: 2, failures: 0, anonymous: false, delay: false, settled: false, requests: [], unexpected: [], writes: [] });

async function mockApi(page: Page, state: State) {
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url()), path = url.pathname, method = route.request().method();
    if (method !== "GET") state.writes.push(`${method} ${path}`);
    let body: unknown;
    if (path === "/api/v1/auth/me") body = { mode: state.anonymous ? "anonymous" : "required", user: { id: "observations-author", email: "author@example.test", display_name: "作者" }, workspace: { id: "observations-workspace", name: "创作工作区", kind: "personal", role: "owner" } };
    else if (path === "/api/v1/auth/logout") { await route.fulfill({ status: 204 }); return; }
    else if (path === "/api/v1/account/model-provider") body = { configured: false, revision: 0 };
    else if (path === "/api/v1/project-catalog") body = { page: 1, page_size: 40, total: 1, items: [{ id: projectId, name: "北港剧情", description: "", created_at: now, active_document_count: 0, latest_run: run() }] };
    else if (path === `/api/v1/projects/${projectId}/documents`) body = [];
    else if (path === `/api/v1/projects/${projectId}/analysis-runs`) body = [run(), run(nextRunId)];
    else if (path === `/api/v1/projects/${projectId}/run-catalog`) body = { project_id: projectId, page: 1, page_size: 20, total: 2, has_more: false, items: [run(), run(nextRunId)].map((item) => ({ ...item, frozen_document_count: 1, input_chars: 100 })) };
    else if (path === `/api/v1/issues/${issue.id}/feedback`) body = { latest: null };
    else if (path.startsWith("/api/v1/analysis-runs/")) {
      const id = path.split("/")[4], endpoint = path.split("/")[5];
      if (!endpoint) body = run(id);
      else if (endpoint === "issues") body = [issue];
      else if (endpoint === "evidence-preview" && [runId, nextRunId].includes(id)
        && url.searchParams.get("kind") === "issue" && url.searchParams.get("item_id") === issue.id
        && url.searchParams.get("evidence_index") === "0") {
        await route.fulfill({ status: 409, json: { detail: "source_snapshot_unavailable" } }); return;
      }
      else if (endpoint === "records") body = { records: [], warnings: [] };
      else if (endpoint === "diagnostics") body = {};
      else if (endpoint === "clarifications") body = [];
      else if (endpoint === "provisional-clues") body = { items: [], truncated: false };
      else if (endpoint === "review-clues") body = { items: [reviewClue], truncated: false, unavailable_count: 0 };
      else if (endpoint === "character-observations") {
        state.requests.push(id);
        if (state.failures > 0) { state.failures--; await route.fulfill({ status: 503, json: { detail: "PRIVATE_SERVER_ERROR" } }); return; }
        if (state.delay && id === runId) {
          await new Promise<void>((resolve) => { state.release = resolve; });
          try { await route.fulfill({ json: observationPayload(id, "complete", 2) }); } catch { /* Expected transport cancellation. */ }
          state.settled = true; return;
        }
        body = observationPayload(id, state.status, state.total, Number(url.searchParams.get("offset") || 0));
      } else { state.unexpected.push(path); await route.fulfill({ status: 404, json: { detail: "unexpected" } }); return; }
    } else { state.unexpected.push(path); await route.fulfill({ status: 404, json: { detail: "unexpected" } }); return; }
    await route.fulfill({ json: body });
  });
}
async function openReport(page: Page, state: State) {
  await mockApi(page, state); await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);
  await expect(page.locator(".issues")).toContainText(issue.title);
}

test("角色记录可读、冻结双侧可展开、计数独立，桌面与手机均无溢出", async ({ page }) => {
  const state = initialState(); await page.setViewportSize({ width: 1440, height: 1000 }); await openReport(page, state);
  const panel = page.getByRole("region", { name: "角色行为记录", exact: true });
  await expect(panel).toContainText("记录已完成");
  await expect(page.locator(".issues article")).toHaveCount(1);
  await expect(page.locator(".provisionalClues > .sectionHead h2")).toContainText("待复核线索 1");
  const summary = panel.locator("summary").first(); await summary.focus(); await summary.press("Enter");
  await expect(panel.locator(".observationDetail").first()).toContainText(source.text);
  await expect(panel.locator(".observationDetail").first()).toContainText("冻结版本 3");
  await expect(panel.locator(".observationBaselines")).toContainText("不会伤害失去反抗能力的人。");
  await expect(panel.locator(".observationBaselines")).toContainText("冻结版本 1");
  await expect(panel).not.toContainText(/PRIVATE_|trait_key=|no_confirmed_baseline/);
  await panel.evaluate((element) => { element.scrollIntoView({ block: "start" }); window.scrollTo(0, 0); });
  await page.screenshot({ path: `${screenshots}/desktop-1440.png` });
  await page.setViewportSize({ width: 375, height: 900 });
  await expect.poll(() => panel.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
  await panel.evaluate((element) => { element.scrollIntoView({ block: "start" }); window.scrollTo(0, 0); });
  await page.screenshot({ path: `${screenshots}/mobile-375.png` });
  await summary.evaluate((element) => { element.scrollIntoView({ block: "start" }); window.scrollTo(0, 0); });
  await page.screenshot({ path: `${screenshots}/mobile-375-detail.png` });
  await panel.locator("summary").nth(1).click();
  await expect(panel.locator(".observationDetail").nth(1)).toContainText("没有可用于比较的已确认角色设定");
  await expect(page.locator(".issues article")).toHaveCount(1);
  await expect(page.locator(".provisionalClues > .sectionHead h2")).toContainText("待复核线索 1");
  expect(state.writes).toEqual([]); expect(state.unexpected).toEqual([]);
});

test("历史未保存、读取失败与可重试、记录失败、完整空和分页分别呈现", async ({ page }) => {
  const state = initialState(); state.status = "legacy_unavailable"; state.total = 0;
  await openReport(page, state); const panel = page.getByRole("region", { name: "角色行为记录", exact: true });
  await expect(panel).toContainText("未保存观察快照"); await expect(panel).toContainText("不等于零条记录或没有异常");
  await expect(panel.locator(".observationPager")).toHaveCount(0);
  state.failures = 1; await panel.getByRole("button", { name: "刷新记录" }).click();
  await expect(panel.getByRole("alert")).toContainText("读取失败"); await expect(panel).not.toContainText("PRIVATE_SERVER_ERROR");
  state.status = "failed"; await panel.getByRole("button", { name: "重试读取记录" }).click();
  await expect(panel).toContainText("记录未能完成");
  state.status = "complete"; await panel.getByRole("button", { name: "刷新记录" }).click();
  await expect(panel).toContainText("未保存到已接纳"); await expect(panel).toContainText("不能说明没有角色异常");
  state.total = 53; await panel.getByRole("button", { name: "刷新记录" }).click();
  await expect(panel.locator(".observationList > li")).toHaveCount(50);
  await panel.getByRole("button", { name: "下一页", exact: true }).click();
  await expect(panel.locator(".observationList > li")).toHaveCount(3);
  await expect(panel.getByRole("button", { name: "下一页", exact: true })).toBeDisabled();
  expect(state.writes).toEqual([]); expect(state.unexpected).toEqual([]);
});

test("切运行或退出会取消私有请求，迟到记录不会回填", async ({ page }) => {
  const state = initialState(); state.delay = true; await openReport(page, state);
  await expect.poll(() => Boolean(state.release)).toBe(true);
  await page.locator(".sideNav").getByRole("button", { name: "运行审计", exact: true }).click();
  await page.getByRole("region", { name: "运行历史", exact: true }).locator(`tr[data-run-id="${nextRunId}"]`).getByRole("button", { name: /^查看报告/ }).click();
  const panel = page.getByRole("region", { name: "角色行为记录", exact: true });
  await expect(panel).toContainText("新运行记录：她留在门前。");
  state.release?.(); await expect.poll(() => state.settled).toBe(true);
  await expect(panel).not.toContainText("收刀后放走了");
  await page.locator(".sideNav").getByRole("button", { name: "运行审计", exact: true }).click();
  state.settled = false; state.release = undefined;
  await page.getByRole("region", { name: "运行历史", exact: true }).locator(`tr[data-run-id="${runId}"]`).getByRole("button", { name: /^查看报告/ }).click();
  await expect.poll(() => Boolean(state.release)).toBe(true);
  await page.getByRole("button", { name: "退出", exact: true }).click();
  await expect(panel).toHaveCount(0); state.release?.(); await expect.poll(() => state.settled).toBe(true);
  await expect(page.locator("body")).not.toContainText("收刀后放走了");
  expect(state.writes).toEqual(["POST /api/v1/auth/logout"]); expect(state.unexpected).toEqual([]);
});

test("匿名工作区不发送私有角色记录请求", async ({ page }) => {
  const state = initialState(); state.anonymous = true; await openReport(page, state);
  await expect(page.getByRole("region", { name: "角色行为记录", exact: true })).toContainText("登录后可查看");
  expect(state.requests).toEqual([]); expect(state.writes).toEqual([]); expect(state.unexpected).toEqual([]);
});
