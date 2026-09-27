import { expect, test, type Page } from "@playwright/test";

const projectId = "draft-only-project";
const projectRoot = `/api/v1/projects/${projectId}`;
const completedRunId = "draft-only-completed-run";
const now = "2026-09-27T00:00:00Z";

type MockDocument = {
  id: string;
  name: string;
  document_role: "chapter" | "canon" | "character_profile" | "reference";
  active: boolean;
  version: number;
  narrative_context: {
    revision: number;
    resolution_state: "confirmed" | "unresolved";
    publication_status: "draft" | "published" | "retired" | "unknown";
    scope_sha256: string;
    scope: { schema_version: 1; timeline_key: string };
  };
};

function document(
  id: string,
  role: MockDocument["document_role"],
  status: MockDocument["narrative_context"]["publication_status"],
  resolution: MockDocument["narrative_context"]["resolution_state"] = "confirmed",
): MockDocument {
  return {
    id,
    name: `${id}.md`,
    document_role: role,
    active: true,
    version: 1,
    narrative_context: {
      revision: 1,
      resolution_state: resolution,
      publication_status: status,
      scope_sha256: "scope-1",
      scope: { schema_version: 1, timeline_key: "main" },
    },
  };
}

type MockState = {
  documents: MockDocument[];
  posts: unknown[];
  documentReads: number;
  unexpected: string[];
  postFormalConflict?: boolean;
  mutateOnPreflight?: () => void;
  completedRun?: Record<string, unknown>;
};

async function mockApi(page: Page, state: MockState) {
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();
    let body: unknown;
    if (path === "/api/v1/auth/me") body = {
      mode: "anonymous",
      user: { id: "author", email: "", display_name: "作者" },
      workspace: { id: "workspace", name: "写作工作区", kind: "personal", role: "owner" },
    };
    else if (path === "/api/v1/account/model-provider") body = {};
    else if (path === "/api/v1/project-catalog") body = {
      page: 1, page_size: 40, total: 1,
      items: [{ id: projectId, name: "草稿故事", description: "", created_at: now,
        active_document_count: state.documents.length, latest_run: null }],
    };
    else if (path === `${projectRoot}/documents`) {
      state.documentReads += 1;
      if (state.documentReads > 1) state.mutateOnPreflight?.();
      body = state.documents.map((item) => ({
        ...item,
        project_id: projectId,
        created_at: now,
        content: "原创剧情正文。",
        story_scope: "main",
      }));
    }
    else if (path === `${projectRoot}/analysis-runs` && method === "GET") {
      body = state.completedRun ? [state.completedRun] : [];
    }
    else if (path === `${projectRoot}/analysis-runs` && method === "POST") {
      state.posts.push(route.request().postDataJSON());
      await route.fulfill({
        status: 409,
        json: state.postFormalConflict
          ? { detail: { code: "formal_context_present", message: "formal context changed", blocking_documents: [] } }
          : { detail: "mocked submission; request captured" },
      });
      return;
    }
    else if (state.completedRun && path === `/api/v1/analysis-runs/${completedRunId}`) body = state.completedRun;
    else if (state.completedRun && path === `/api/v1/analysis-runs/${completedRunId}/issues`) body = [];
    else if (state.completedRun && path === `/api/v1/analysis-runs/${completedRunId}/records`) body = { records: [], warnings: [] };
    else if (state.completedRun && path === `/api/v1/analysis-runs/${completedRunId}/diagnostics`) body = {};
    else if (state.completedRun && path === `/api/v1/analysis-runs/${completedRunId}/clarifications`) body = [];
    else if (state.completedRun && path === `/api/v1/analysis-runs/${completedRunId}/provisional-clues`) body = { items: [], truncated: false };
    else if (state.completedRun && path === `/api/v1/analysis-runs/${completedRunId}/review-clues`) body = { items: [], truncated: false, unavailable_count: 0 };
    else {
      state.unexpected.push(`${method} ${path}`);
      await route.fulfill({ status: 404, json: { detail: "unexpected mocked endpoint" } });
      return;
    }
    await route.fulfill({ status: 200, json: body });
  });
}

test("没有正式设定的项目可显式选稿自检，已确认退役和参考文档不成为背景", async ({ page }) => {
  const state: MockState = {
    documents: [
      document("first-draft", "chapter", "draft"),
      document("other-draft", "chapter", "draft"),
      document("old-setting", "canon", "retired"),
      document("inspiration", "reference", "published"),
    ],
    posts: [], documentReads: 0, unexpected: [],
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  const review = page.locator(".guidedReviewLaunch");
  await expect(review.getByRole("heading", { name: "从新稿开始故事自检" })).toBeVisible();
  await expect(review).toContainText("若当前没有已确认特征，角色 OOC 将弃权");
  await expect(review.getByRole("heading", { name: "建立并确认角色基线" })).toHaveCount(0);
  if (process.env.LOREGUARD_DRAFT_ONLY_SCREENSHOTS === "1") {
    await page.screenshot({ path: "../artifacts/playwright/draft-only-desktop.png", fullPage: true });
    await page.setViewportSize({ width: 375, height: 812 });
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.screenshot({ path: "../artifacts/playwright/draft-only-mobile-375.png", fullPage: true });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.setViewportSize({ width: 667, height: 375 });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    await page.setViewportSize({ width: 1280, height: 720 });
  }
  const start = review.getByRole("button", { name: /^开始校验/ });
  await expect(start).toBeDisabled();
  await review.getByRole("checkbox", { name: /first-draft.md/ }).check();
  await expect(start).toBeEnabled();
  await start.click();
  await expect.poll(() => state.posts.length).toBe(1);
  expect(state.posts).toEqual([{
    mode: "draft_review", sensitivity: "balanced",
    target_document_ids: ["first-draft"],
    no_formal_context_expected: true,
  }]);
  expect(state.documentReads).toBeGreaterThanOrEqual(2);
  expect(state.unexpected).toEqual([]);
});

test("待确认正式设定及已发布历史阻止零背景入口", async ({ page }) => {
  const state: MockState = {
    documents: [
      document("draft", "chapter", "draft"),
      document("setting", "canon", "retired", "unresolved"),
    ],
    posts: [], documentReads: 0, unexpected: [],
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  const review = page.locator(".guidedReviewLaunch");
  await expect(review.getByRole("heading", { name: "从正式资料到新稿审查" })).toBeVisible();
  await expect(review.getByRole("button", { name: /^开始校验/ })).toHaveCount(0);
  state.documents = [document("draft", "chapter", "draft"), document("history", "chapter", "published", "unresolved")];
  await page.reload();
  await expect(review.getByRole("heading", { name: "从正式资料到新稿审查" })).toBeVisible();
  await expect(review.getByRole("button", { name: /^开始校验/ })).toHaveCount(0);
  expect(state.posts).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test("提交前刷新发现新增正式资料或新稿版本变化时不发起分析", async ({ page }) => {
  const state: MockState = {
    documents: [document("draft", "chapter", "draft")],
    posts: [], documentReads: 0, unexpected: [],
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  const review = page.locator(".guidedReviewLaunch");
  const start = review.getByRole("button", { name: /^开始校验/ });
  await review.getByRole("checkbox", { name: /draft.md/ }).check();
  state.mutateOnPreflight = () => state.documents.push(document("new-setting", "canon", "unknown", "unresolved"));
  await start.click();
  await expect(review.getByRole("alert")).toContainText("项目已有正式资料");
  expect(state.posts).toEqual([]);

  state.documents = [document("draft", "chapter", "draft")];
  state.mutateOnPreflight = undefined;
  await page.reload();
  await expect(review.getByRole("checkbox", { name: /draft.md/ })).toBeChecked();
  state.mutateOnPreflight = () => { state.documents[0] = { ...state.documents[0], version: 2 }; };
  await start.click();
  await expect(review.getByRole("alert")).toContainText("所选新稿的版本或资料状态已变化");
  expect(state.posts).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test("服务端在提交瞬间检测到正式资料竞态时展示恢复指引；窄屏可键盘操作", async ({ page }) => {
  const state: MockState = {
    documents: [document("draft", "chapter", "draft")],
    posts: [], documentReads: 0, unexpected: [], postFormalConflict: true,
  };
  await mockApi(page, state);
  await page.setViewportSize({ width: 375, height: 812 });
  await page.goto(`/app/projects/${projectId}/check`);
  const review = page.locator(".guidedReviewLaunch");
  const draft = review.getByRole("checkbox", { name: /draft.md/ });
  await draft.focus();
  await draft.press("Space");
  const start = review.getByRole("button", { name: /^开始校验/ });
  await expect(start).toBeEnabled();
  await start.focus();
  await start.press("Enter");
  await expect(review.getByRole("alert")).toContainText("项目新增了正式资料");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(state.unexpected).toEqual([]);
});

test("首章发布后即使没有角色基线，仍可用已确认历史审查新稿的其他一致性", async ({ page }) => {
  const state: MockState = {
    documents: [
      document("published-first", "chapter", "published"),
      document("next-draft", "chapter", "draft"),
    ],
    posts: [], documentReads: 0, unexpected: [],
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  const review = page.locator(".guidedReviewLaunch");
  await expect(review).toContainText("正式背景已确认，但角色基线尚未达到完整可核对状态");
  await expect(review).toContainText("未覆盖角色不会被判为“通过”");
  const start = review.getByRole("button", { name: /^开始校验/ });
  await expect(start).toBeDisabled();
  await review.getByRole("checkbox", { name: /next-draft.md/ }).check();
  await start.click();
  await expect.poll(() => state.posts.length).toBe(1);
  expect(state.posts[0]).toEqual({
    mode: "draft_review", sensitivity: "balanced", target_document_ids: ["next-draft"],
  });
  expect(state.documentReads).toBeGreaterThanOrEqual(2);
  expect(state.unexpected).toEqual([]);
});

test("有已发布历史但另有待确认正式资料时，不能绕过确认进入有限覆盖审查", async ({ page }) => {
  const state: MockState = {
    documents: [
      document("published-first", "chapter", "published"),
      document("next-draft", "chapter", "draft"),
      document("unconfirmed-setting", "canon", "unknown", "unresolved"),
    ],
    posts: [], documentReads: 0, unexpected: [],
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/check`);
  const review = page.locator(".guidedReviewLaunch");
  await expect(review).toContainText("可能作为正式背景的资料仍需核对");
  await expect(review.getByRole("button", { name: /^开始校验/ })).toHaveCount(0);
  expect(state.posts).toEqual([]);
  expect(state.unexpected).toEqual([]);
});

test("完成的仅新稿报告明确写出 1 份目标、0 份背景及 OOC 限制，不将线索冒称确认问题", async ({ page }, testInfo) => {
  const state: MockState = {
    documents: [document("draft", "chapter", "draft")],
    posts: [], documentReads: 0, unexpected: [],
    completedRun: {
      id: completedRunId,
      project_id: projectId,
      status: "completed",
      created_at: now,
      completed_at: now,
      prompt_tokens: 20,
      completion_tokens: 10,
      estimated_cost_usd: 0,
      review_batch: {
        mode: "draft_review",
        sensitivity: "balanced",
        target_document_ids: ["draft"],
        background_document_ids: [],
        no_formal_context_expected: true,
      },
    },
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${completedRunId}/report`);
  const scope = page.getByRole("region", { name: "本次审查范围" });
  await expect(scope).toContainText("仅新稿自检");
  await expect(scope).toContainText("1 份目标章节、0 份正式背景");
  await expect(scope).toContainText("不能据零问题断言角色 OOC 已通过");
  await expect(page.getByRole("heading", { name: /一致性问题/ })).toBeVisible();
  await expect(page.getByRole("heading", { name: /已确认的一致性问题/ })).toHaveCount(0);
  await expect(page.locator(".railSummary")).toContainText("一致性问题");
  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await scope.screenshot({ path: testInfo.outputPath("draft-only-report-desktop.png") });
  }
  await page.setViewportSize({ width: 375, height: 812 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(scope).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await scope.screenshot({ path: testInfo.outputPath("draft-only-report-mobile-375.png") });
  }
  expect(state.unexpected).toEqual([]);
});
