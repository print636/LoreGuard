import path from "node:path";
import { expect, test, type Page } from "@playwright/test";

const projectId = "guided-status-project";
const root = `/api/v1/projects/${projectId}`;
const secondProjectId = "guided-status-other-project";
const secondRoot = `/api/v1/projects/${secondProjectId}`;
const now = "2026-09-26T00:00:00Z";

const confirmedContext = (publication_status: "published" | "draft") => ({
  revision: 1,
  resolution_state: "confirmed",
  publication_status,
  scope_sha256: "scope-v1",
  scope: { schema_version: 1, timeline_key: "main" },
});

const documents = [
  { id: "canon-1", name: "正式世界设定.md", document_role: "canon", narrative_context: confirmedContext("published") },
  { id: "draft-1", name: "第一章.md", document_role: "chapter", narrative_context: confirmedContext("draft") },
  { id: "draft-2", name: "第二章.md", document_role: "chapter", narrative_context: confirmedContext("draft") },
].map((item) => ({
  ...item,
  project_id: projectId,
  version: 1,
  active: true,
  created_at: now,
  content: "原创剧情正文。",
  story_scope: "main",
}));

const baselineRun = {
  id: "baseline-current",
  project_id: projectId,
  status: "completed",
  created_at: now,
  completed_at: now,
  prompt_tokens: 10,
  completion_tokens: 10,
  estimated_cost_usd: 0,
  input_snapshot_available: true,
  input_documents: [{
    document_id: "canon-1",
    document_version: 1,
    document_role: "canon",
    batch_role: "background",
    narrative_context: {
      context_revision: 1,
      resolution_state: "confirmed",
      publication_status: "published",
      scope_sha256: "scope-v1",
    },
  }],
  review_batch: {
    mode: "baseline_build",
    sensitivity: "balanced",
    background_document_ids: ["canon-1"],
  },
};

type MockState = {
  confirmed?: number;
  pending?: number;
  total?: number;
  sourceRun?: string;
  coverage?: "full" | "partial" | "rules_only";
  statusFailures?: number;
  unconfirmFirstDraft?: boolean;
  userId?: string;
  workspaceId?: string;
  statusRequests?: string[];
  expectedBaselineRunId?: string;
  extraRuns?: Array<typeof baselineRun>;
  secondProject?: boolean;
  holdPreflight?: Promise<void>;
  posts: unknown[];
  unexpected: string[];
};

async function mockApi(page: Page, state: MockState) {
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();
    let body: unknown;
    if (path === "/api/v1/auth/me") body = {
      mode: "anonymous",
      user: { id: state.userId || "author", email: "", display_name: "作者" },
      workspace: { id: state.workspaceId || "workspace", name: "写作工作区", kind: "personal", role: "owner" },
    };
    else if (path === "/api/v1/account/model-provider") body = {};
    else if (path === "/api/v1/project-catalog") {
      const items = [
        { id: projectId, name: "角色审查项目", description: "", created_at: now,
          active_document_count: 3, latest_run: null },
        ...(state.secondProject ? [{ id: secondProjectId, name: "另一故事项目", description: "",
          created_at: now, active_document_count: 1, latest_run: null }] : []),
      ].filter((item) => !url.searchParams.get("project_id") || item.id === url.searchParams.get("project_id"));
      body = { page: 1, page_size: Number(url.searchParams.get("page_size") || "40"),
        total: items.length, items };
    }
    else if (path === `${root}/documents`) body = documents.map((document) =>
      state.unconfirmFirstDraft && document.id === "draft-1"
        ? { ...document, narrative_context: { ...document.narrative_context, resolution_state: "unresolved" } }
        : document,
    );
    else if (state.secondProject && path === `${secondRoot}/documents`) body = [{
      ...documents[0], id: "canon-other", project_id: secondProjectId,
      name: "另一个故事的正式设定.md",
    }];
    else if (path === `${root}/analysis-runs` && method === "GET") body = [baselineRun, ...(state.extraRuns || [])];
    else if (state.secondProject && path === `${secondRoot}/analysis-runs` && method === "GET") body = [];
    else if (path === `/api/v1/analysis-runs/${baselineRun.id}`) body = baselineRun;
    else if (path === `/api/v1/analysis-runs/${baselineRun.id}/issues`) body = [];
    else if (path === `/api/v1/analysis-runs/${baselineRun.id}/records`) body = { records: [], warnings: [] };
    else if (path === `/api/v1/analysis-runs/${baselineRun.id}/diagnostics`) body = {};
    else if (path === `/api/v1/analysis-runs/${baselineRun.id}/clarifications`) body = [];
    else if (path === `${root}/analysis-runs` && method === "POST") {
      state.posts.push(route.request().postDataJSON());
      await route.fulfill({ status: 503, json: { detail: "mocked run submit; request captured" } });
      return;
    }
    else if (path === `${root}/character-baseline-status`) {
      const requestedRunId = url.searchParams.get("baseline_run_id") || "";
      state.statusRequests?.push(requestedRunId);
      if (state.holdPreflight && (state.statusRequests?.length || 0) > 1) await state.holdPreflight;
      if (requestedRunId !== (state.expectedBaselineRunId || baselineRun.id)) {
        state.unexpected.push(`GET ${path}?baseline_run_id=${requestedRunId}`);
        await route.fulfill({ status: 400, json: { detail: "wrong baseline run" } });
        return;
      }
      if ((state.statusFailures || 0) > 0) {
        state.statusFailures = (state.statusFailures || 0) - 1;
        await route.fulfill({ status: 503, json: { detail: "暂时不可用" } });
        return;
      }
      body = {
        total_characters: state.total ?? 142,
        confirmed_trait_count: state.confirmed ?? 3,
        pending_candidate_count: state.pending ?? 206,
        baseline_run_id: state.sourceRun ?? "baseline-current",
        baseline_run_status: "completed",
        model_coverage: state.coverage ?? "full",
        coverage_detail: null,
        readiness: "ready",
      };
    }
    else {
      state.unexpected.push(`${method} ${path}`);
      await route.fulfill({ status: 404, json: { detail: "unexpected mocked endpoint" } });
      return;
    }
    await route.fulfill({ status: 200, json: body });
  });
}

test("超过百名角色且有待确认候选时仍可选择单章或批量审查，未确认候选不成权威", async ({ page }) => {
  const state: MockState = { posts: [], unexpected: [], statusRequests: [] };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  const review = page.locator(".guidedReviewLaunch");
  await expect(review).toContainText("项目级汇总：142 个角色、3 条已确认特征");
  await expect(review.locator(".stagePendingNotice")).toContainText("206 条待确认");
  await expect(review.locator(".stageBaselineSummary")).toHaveAttribute("aria-live", "polite");
  await expect(review.locator(".stagePendingNotice")).toContainText("不会作为正式角色设定参与审查");
  await expect(review).toContainText("未覆盖角色不会获得角色 OOC 判断");
  expect(state.statusRequests).toEqual(["baseline-current"]);
  const visualOutput = process.env.LOREGUARD_GUIDED_REVIEW_SCREENSHOTS_DIR;
  if (visualOutput) {
    await page.screenshot({ path: path.join(visualOutput, "guided-review-desktop.png"), fullPage: true });
    await review.getByRole("button", { name: /^开始校验/ }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(visualOutput, "guided-review-desktop-action.png"), fullPage: true });
    await page.setViewportSize({ width: 375, height: 812 });
    await review.scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(visualOutput, "guided-review-mobile-375.png"), fullPage: true });
    await review.getByRole("button", { name: /^开始校验/ }).scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(visualOutput, "guided-review-mobile-375-action.png"), fullPage: true });
    await page.setViewportSize({ width: 1280, height: 720 });
  }
  const start = review.getByRole("button", { name: /^开始校验/ });
  await expect(start).toBeDisabled();
  await review.getByRole("checkbox", { name: /第一章.md/ }).check();
  await expect(start).toBeEnabled();
  await expect(start).toHaveText("开始校验（1 份）");
  await page.reload();
  await expect(review.getByRole("checkbox", { name: /第一章.md/ })).toBeChecked();
  await review.getByRole("checkbox", { name: /第二章.md/ }).check();
  await expect(start).toHaveText("开始校验（2 份）");
  await review.getByRole("checkbox", { name: /第一章.md/ }).uncheck();
  await page.reload();
  await expect(review.getByRole("checkbox", { name: /第一章.md/ })).not.toBeChecked();
  await expect(review.getByRole("checkbox", { name: /第二章.md/ })).toBeChecked();
  await start.click();
  await expect.poll(() => state.posts.length).toBe(1);
  expect(state.posts[0]).toEqual({
    mode: "draft_review",
    sensitivity: "balanced",
    target_document_ids: ["draft-2"],
  });
  expect(state.unexpected).toEqual([]);
});

test("基线读取失败可重试，选稿保留；窄屏下警告和键盘入口可操作", async ({ page }) => {
  const state: MockState = { posts: [], unexpected: [] };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  const review = page.locator(".guidedReviewLaunch");
  await expect(review.getByRole("checkbox", { name: /第一章.md/ })).toBeVisible();
  await review.getByRole("checkbox", { name: /第一章.md/ }).check();
  state.statusFailures = 1;
  await page.reload();
  await expect(review.getByRole("alert")).toContainText("请重试");
  await expect(review.getByRole("button", { name: /^开始校验/ })).toHaveCount(0);
  const retry = review.getByRole("button", { name: "重试读取基线状态" });
  await retry.focus();
  await retry.press("Enter");
  await expect(review.getByRole("checkbox", { name: /第一章.md/ })).toBeChecked();
  await expect(review.getByRole("button", { name: /^开始校验/ })).toBeEnabled();
  await page.setViewportSize({ width: 375, height: 812 });
  await expect(review.locator(".stagePendingNotice")).toBeVisible();
  await expect(review.getByRole("button", { name: "核对待确认候选" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(state.unexpected).toEqual([]);
});

test("零已确认特征或过期基线不能被项目级待确认数解锁", async ({ page }) => {
  const state: MockState = { confirmed: 0, pending: 206, posts: [], unexpected: [] };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  const review = page.locator(".guidedReviewLaunch");
  await expect(review).toContainText("项目级角色档案中暂无已确认特征");
  await expect(review.getByRole("button", { name: /^开始校验/ })).toHaveCount(0);
  state.confirmed = 3;
  state.sourceRun = "older-run";
  await page.reload();
  await expect(review).toContainText("并非来自当前冻结资料的完整运行");
  await expect(review.getByRole("button", { name: /^开始校验/ })).toHaveCount(0);
  expect(state.posts).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test("恢复的选稿必须仍是当前已确认草稿；失效后再次确认不会自动重选", async ({ page }) => {
  const state: MockState = { posts: [], unexpected: [] };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  const review = page.locator(".guidedReviewLaunch");
  const first = review.getByRole("checkbox", { name: /第一章.md/ });
  await expect(first).toBeVisible();
  await first.check();
  state.unconfirmFirstDraft = true;
  await page.reload();
  await expect(first).toHaveCount(0);
  await expect(review.getByRole("button", { name: /^开始校验/ })).toBeDisabled();
  state.unconfirmFirstDraft = false;
  await page.reload();
  await expect(first).toBeVisible();
  await expect(first).not.toBeChecked();
  expect(state.unexpected).toEqual([]);
});

test("同一标签页切换用户或工作区不会继承另一身份的选稿", async ({ page }) => {
  const state: MockState = { posts: [], unexpected: [] };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  const first = page.locator(".guidedReviewLaunch").getByRole("checkbox", { name: /第一章.md/ });
  await expect(first).toBeVisible();
  await first.check();
  state.userId = "different-author";
  await page.reload();
  await expect(first).toBeVisible();
  await expect(first).not.toBeChecked();
  state.userId = "author";
  state.workspaceId = "different-workspace";
  await page.reload();
  await expect(first).toBeVisible();
  await expect(first).not.toBeChecked();
  state.workspaceId = "workspace";
  await page.reload();
  await expect(first).toBeChecked();
  expect(state.unexpected).toEqual([]);
});

test("旧资料的基线晚完成，也不会盖过当前冻结资料的基线", async ({ page }) => {
  const staleLateRun = {
    ...baselineRun,
    id: "stale-late-run",
    created_at: "2026-09-25T10:00:00Z",
    completed_at: "2026-09-26T03:00:00Z",
    input_documents: [{ ...baselineRun.input_documents[0], document_version: 0 }],
  };
  const state: MockState = { posts: [], unexpected: [], statusRequests: [], extraRuns: [staleLateRun] };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  await expect(page.locator(".guidedReviewLaunch")).toContainText("当前角色基线可用于新稿审查");
  expect(state.statusRequests).toEqual(["baseline-current"]);
  expect(state.unexpected).toEqual([]);
});

test("提交前发现另一标签页撤销最后特征时，刷新卡片且不创建运行", async ({ page }) => {
  const state: MockState = { posts: [], unexpected: [], statusRequests: [] };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  const review = page.locator(".guidedReviewLaunch");
  await expect(review.getByRole("checkbox", { name: /第一章.md/ })).toBeVisible();
  await review.getByRole("checkbox", { name: /第一章.md/ }).check();
  state.confirmed = 0;
  await review.getByRole("button", { name: /^开始校验/ }).click();
  await expect(review.getByRole("alert")).toContainText("本次没有启动任务");
  await expect(review).toContainText("项目级角色档案中暂无已确认特征");
  await expect(review.getByRole("button", { name: /^开始校验/ })).toHaveCount(0);
  expect(state.statusRequests).toEqual(["baseline-current", "baseline-current"]);
  expect(state.posts).toEqual([]);
  state.confirmed = 3;
  await page.reload();
  await expect(review.getByRole("checkbox", { name: /第一章.md/ })).toBeChecked();
  expect(state.unexpected).toEqual([]);
});

test("提交前基线查询断线时保留选稿、不提交任务，并提供重试路径", async ({ page }) => {
  const state: MockState = { posts: [], unexpected: [], statusRequests: [] };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  const review = page.locator(".guidedReviewLaunch");
  await expect(review.getByRole("checkbox", { name: /第一章.md/ })).toBeVisible();
  await review.getByRole("checkbox", { name: /第一章.md/ }).check();
  state.statusFailures = 1;
  await review.getByRole("button", { name: /^开始校验/ }).click();
  await expect(review.getByRole("alert").last()).toContainText("提交前复核失败，本次没有启动任务");
  await expect(review.getByRole("button", { name: /^开始校验/ })).toHaveCount(0);
  expect(state.posts).toEqual([]);
  await review.getByRole("button", { name: "重试读取基线状态" }).click();
  await expect(review.getByRole("checkbox", { name: /第一章.md/ })).toBeChecked();
  await expect(review.getByRole("button", { name: /^开始校验/ })).toBeEnabled();
  expect(state.statusRequests).toEqual(["baseline-current", "baseline-current", "baseline-current"]);
  expect(state.unexpected).toEqual([]);
});

test("提交前复核未返回就切换项目时，旧请求不能提交或锁住新项目", async ({ page }) => {
  let releasePreflight!: () => void;
  const holdPreflight = new Promise<void>((resolve) => { releasePreflight = resolve; });
  const state: MockState = {
    posts: [], unexpected: [], statusRequests: [], secondProject: true, holdPreflight,
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  const review = page.locator(".guidedReviewLaunch");
  await expect(review.getByRole("checkbox", { name: /第一章.md/ })).toBeVisible();
  await review.getByRole("checkbox", { name: /第一章.md/ }).check();
  await review.getByRole("button", { name: /^开始校验/ }).click();
  await expect(review.getByRole("button", { name: "正在复核角色基线…" })).toBeDisabled();
  await expect.poll(() => state.statusRequests?.length).toBe(2);

  // Same-view route switch reuses the React workspace instead of reloading it.
  await page.evaluate((nextProjectId) => {
    const previous = window.history.state || {};
    const previousIndex = Number(previous.__loreguardHistoryIndex || 0);
    window.history.pushState(
      { ...previous, __loreguardHistoryIndex: previousIndex + 1 },
      "",
      `/app/projects/${nextProjectId}/check`,
    );
    window.dispatchEvent(new PopStateEvent("popstate", { state: window.history.state }));
  }, secondProjectId);
  await expect(page).toHaveURL(new RegExp(`/app/projects/${secondProjectId}/check$`));
  const newBaselineButton = review.getByRole("button", { name: "建立角色基线", exact: true });
  await expect(newBaselineButton).toBeEnabled();
  releasePreflight();
  await expect(newBaselineButton).toBeEnabled();
  expect(state.posts).toEqual([]);
  expect(state.unexpected).toEqual([]);
});
