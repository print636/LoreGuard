import path from "node:path";
import { expect, test, type Locator, type Page } from "@playwright/test";

// These are independent reading fixtures, not evaluation cases or model output.
const projectId = "product-help-story";
const runId = "product-help-report";
const issueId = "product-help-issue";
const now = "2026-10-02T00:00:00Z";
const screenshotFolder = process.env.LOREGUARD_PRODUCT_HELP_SCREENSHOT_DIR
  ?? process.env.LOREGUARD_E2E_SCREENSHOT_DIR
  ?? "../artifacts/product-help";
const topics = ["start", "materials", "characters", "review", "report", "coverage"] as const;
type Topic = typeof topics[number];

const evidence = {
  document_id: "help-draft", document_name: "北港章节.md",
  line_start: 3, line_end: 3, text: "本次报告引用的章节原文。",
};
const issue = {
  id: issueId, report_class: "formal", category: "fact_conflict", severity: "medium", confidence: 0.7,
  title: "需要作者核对的正式问题", explanation: "阅读指南后仍需结合原文判断。",
  suggestion: "核对章节上下文。", evidence: [evidence], metadata: {},
};
const secondIssue = {
  ...issue, id: "product-help-other-issue", category: "character_drift", title: "另一类别的问题",
};
const reviewClue = {
  ...issue, id: "71475dcc-1d0c-4dd2-a014-06941adf1bf3", report_class: "review_clue",
  category: "character_drift", title: "角色审查待复核线索", severity: "low", confidence: 0.5,
  evidence: [evidence, { ...evidence, line_start: 4, line_end: 4, text: "线索引用的另一段原文。" }],
  metadata: { final_outcome: "needs_confirmation", review_reason: "reading_contract_fixture" },
};
const provisionalClue = {
  id: `pc_${"a".repeat(32)}`, document_id: evidence.document_id,
  document_name: evidence.document_name, document_version: 1,
  line_start: 3, line_end: 3, evidence: evidence.text,
  character: "阅读角色", dimension: "core_personality",
  proposed_statement: "尚未验证的模型提案。", reason: "partial_model_package",
};
const run = {
  id: runId, project_id: projectId, status: "completed", created_at: now, completed_at: now,
  prompt_tokens: 12, completion_tokens: 4, estimated_cost_usd: 0,
  input_snapshot_available: true,
  input_documents: [{
    document_id: evidence.document_id, document_name: evidence.document_name,
    document_version: 1, document_role: "chapter", story_scope: "main",
    content_sha256: "a".repeat(64), batch_role: "target",
  }],
  review_batch: {
    mode: "draft_review", no_formal_context_expected: true,
    target_document_ids: [evidence.document_id], background_document_ids: [],
  },
};
const documents = [{
  id: evidence.document_id, project_id: projectId, name: evidence.document_name,
  version: 2, active: true, created_at: now, document_role: "chapter", story_scope: "main",
  narrative_context: {
    revision: 1, resolution_state: "confirmed", publication_status: "draft", origin: "explicit",
    scope: { schema_version: 1, timeline_key: "main" },
  },
}];

function initialState() {
  return { requests: [] as string[], writes: [] as string[], unexpected: [] as string[] };
}
type MockState = ReturnType<typeof initialState>;

async function mockApi(page: Page, state: MockState) {
  // Intercept every business endpoint so an unsupported request cannot reach a
  // running backend, its database, a sample creator or a model configuration.
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const { pathname: endpoint } = url;
    const method = route.request().method();
    state.requests.push(`${method} ${endpoint}${url.search}`);
    if (method !== "GET") {
      state.writes.push(`${method} ${endpoint}`);
      await route.fulfill({ status: 503, json: { detail: "Read-only help fixture rejects writes" } });
      return;
    }
    let body: unknown;
    if (endpoint === "/api/v1/auth/me") body = {
      mode: "anonymous", user: { id: "help-author", email: "", display_name: "首次作者" },
      workspace: { id: "help-workspace", name: "故事工作区", kind: "personal", role: "owner" },
    };
    else if (endpoint === "/api/v1/account/model-provider") body = { configured: false, revision: 0 };
    else if (endpoint === "/api/v1/project-catalog") {
      const item = {
        id: projectId, name: "北港游戏剧情", description: "独立帮助阅读场景", created_at: now,
        active_document_count: 1, latest_run: run,
      };
      const matches = (!url.searchParams.get("project_id") || url.searchParams.get("project_id") === projectId)
        && (!url.searchParams.get("query") || item.name.includes(url.searchParams.get("query")!.trim()));
      body = { page: 1, page_size: 40, total: matches ? 1 : 0, items: matches ? [item] : [] };
    }
    else if (endpoint === `/api/v1/projects/${projectId}/documents`) body = documents;
    else if (endpoint === `/api/v1/projects/${projectId}/analysis-runs`) body = [run];
    else if (endpoint === `/api/v1/analysis-runs/${runId}`) body = run;
    else if (endpoint === `/api/v1/analysis-runs/${runId}/issues`) body = [issue, secondIssue];
    else if (endpoint === `/api/v1/analysis-runs/${runId}/records`) body = { records: [], warnings: [] };
    else if (endpoint === `/api/v1/analysis-runs/${runId}/clarifications`) body = [];
    else if (endpoint === `/api/v1/analysis-runs/${runId}/diagnostics`) body = {
      model: {
        enabled: true, configured: true, total_chunks: 2, attempted_chunks: 1,
        succeeded_chunks: 1, failed_chunks: 0, skipped_chunks: 1,
        invalid_records: 0, empty_response_chunks: 0, reason_codes: ["token_budget"],
      },
      character_consistency: {
        enabled: true, outcome: "partial", reason_code: "bounded_partial", snapshot_bound: true,
        material_coverage: "partial", counts: {
          planned_chunks: 2, processed_chunks: 2, model_called_chunks: 1,
          model_completed_chunks: 1, model_uncalled_chunks: 1, model_incomplete_chunks: 1,
        }, reason_counts: { token_budget: 1 },
      },
    };
    else if (endpoint === `/api/v1/analysis-runs/${runId}/review-clues`) body = {
      items: [reviewClue], truncated: false, unavailable_count: 0,
    };
    else if (endpoint === `/api/v1/analysis-runs/${runId}/provisional-clues`) body = {
      items: [provisionalClue], truncated: false,
    };
    else if ([issueId, secondIssue.id].some((id) => endpoint === `/api/v1/issues/${id}/feedback`)) body = { latest: null };
    else if (endpoint.endsWith("/narrative-context")) body = {
      document_id: evidence.document_id, current: documents[0].narrative_context, revision_count: 1,
    };
    else {
      state.unexpected.push(`${method} ${endpoint}`);
      await route.fulfill({ status: 404, json: { detail: "Unexpected help fixture endpoint" } });
      return;
    }
    await route.fulfill({ json: body });
  });
}

function guide(page: Page) {
  return page.getByRole("dialog", { name: "使用指南", exact: true });
}

function topicHeading(dialog: Locator, topic: Topic) {
  return dialog.locator(`h3#user-guide-topic-${topic}`);
}

async function openPage(page: Page, state: MockState, url: string) {
  await mockApi(page, state);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(url);
  await page.waitForLoadState("networkidle");
}

async function readingSnapshot(page: Page, state: MockState) {
  await page.waitForLoadState("networkidle");
  return { url: page.url(), requestCount: state.requests.length };
}

async function assertReadingOnly(page: Page, state: MockState, before: { url: string; requestCount: number }) {
  await page.waitForLoadState("networkidle");
  expect(page.url(), "打开、目录定位与关闭帮助不得修改当前URL").toBe(before.url);
  expect(state.requests.slice(before.requestCount), "帮助不得发起额外业务读取或写入").toEqual([]);
  expect(state.writes, "帮助不得新建项目、启动分析、发布、写反馈或调用模型").toEqual([]);
  expect(state.unexpected).toEqual([]);
}

async function assertCloseVisible(page: Page, dialog: Locator) {
  const close = dialog.getByRole("button", { name: "关闭使用指南", exact: true });
  await expect(close).toBeInViewport({ ratio: 1 });
  const box = await close.boundingBox();
  expect(box!.y).toBeGreaterThanOrEqual(0);
  expect(box!.y + box!.height).toBeLessThanOrEqual(page.viewportSize()!.height);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
  expect(await dialog.evaluate((element) => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
}

async function selectTopic(dialog: Locator, topic: Topic) {
  const navigation = dialog.getByRole("navigation", { name: "使用指南目录" });
  await dialog.getByRole("button", { name: "查看目录", exact: true }).click();
  await expect(navigation.getByRole("heading", { name: "查阅目录", exact: true })).toBeFocused();
  const heading = topicHeading(dialog, topic);
  const title = await heading.innerText();
  await navigation.getByRole("button", { name: title, exact: true }).click();
  await expect(heading).toBeFocused();
  await expect(heading).toBeInViewport({ ratio: 1 });
}

test("首次作者从项目中心读完整指南，目录和键盘关闭保留目录状态", async ({ page }) => {
  const state = initialState();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openPage(page, state, "/app?query=北港&sort=name");
  await expect(page.locator(".projectList li")).toHaveCount(1);
  const trigger = page.locator(".projectCenterHeader").getByRole("button", { name: "使用指南", exact: true });
  await expect(trigger).toBeInViewport({ ratio: 1 });
  const before = await readingSnapshot(page, state);
  await trigger.focus();
  await trigger.press("Enter");
  const dialog = guide(page);
  await expect(dialog).toBeVisible();
  await expect(topicHeading(dialog, "start")).toBeInViewport({ ratio: 1 });
  await expect(dialog.locator('[id^="user-guide-topic-"]')).toHaveCount(6);
  await expect(dialog.getByRole("region", { name: "使用指南正文" })).toBeVisible();
  await expect(dialog.getByRole("navigation", { name: "使用指南目录" }).getByRole("button")).toHaveCount(6);

  const focusable = dialog.locator('button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex="0"]');
  await focusable.first().focus();
  await page.keyboard.press("Shift+Tab");
  await expect(focusable.last()).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(focusable.first()).toBeFocused();
  for (const topic of topics) await selectTopic(dialog, topic);
  await selectTopic(dialog, "start");
  await assertCloseVisible(page, dialog);
  const readingTypography = await dialog.evaluate((element) => {
    const paragraph = getComputedStyle(element.querySelector(".userGuideBlock p")!);
    const step = getComputedStyle(element.querySelector(".userGuideBlock li")!);
    const heading = getComputedStyle(element.querySelector(".userGuideTopic h3")!);
    return {
      paragraph: { fontWeight: paragraph.fontWeight, fontSize: paragraph.fontSize },
      step: { fontWeight: step.fontWeight, fontSize: step.fontSize },
      heading: { fontWeight: heading.fontWeight, fontSize: heading.fontSize },
    };
  });
  console.info("使用指南正文 computed style", JSON.stringify(readingTypography));
  await page.screenshot({ path: path.join(screenshotFolder, "product-help-project-center-1440.png") });
  await page.keyboard.press("Escape");
  await expect(dialog).not.toBeVisible();
  await expect(trigger).toBeFocused();
  await expect(page.getByRole("searchbox", { name: "搜索项目" })).toHaveValue("北港");
  await expect(page.getByRole("combobox", { name: "项目排序" })).toHaveValue("name");
  await expect(page.locator(".projectList li")).toHaveCount(1);
  await assertReadingOnly(page, state, before);
});

test("375像素校验台入口可见，作者阅读帮助后保留未提交正文和文本设置", async ({ page }) => {
  const state = initialState();
  await page.setViewportSize({ width: 375, height: 812 });
  await openPage(page, state, "/check");
  const draft = "我正在写第一章，还没有提交分析。\n第二段对白也需要保留。";
  const editor = page.getByRole("textbox", { name: "需要审查的故事正文 / chapter.md", exact: true });
  await expect(editor).toBeVisible();
  await editor.fill(draft);
  await page.getByRole("textbox", { name: "快速文本故事作用域" }).fill("route_a");
  const trigger = page.locator(".workbenchIntro").getByRole("button", { name: "使用指南", exact: true });
  // Filling the editor may scroll the document; return to the actual title
  // viewport to check the intended small-screen entry rather than a sidebar.
  await page.evaluate(() => window.scrollTo(0, 0));
  await expect(trigger).toBeInViewport({ ratio: 1 });
  const before = await readingSnapshot(page, state);
  await trigger.click();
  const dialog = guide(page);
  await expect(dialog).toBeVisible();
  await expect(topicHeading(dialog, "review")).toBeFocused();
  await selectTopic(dialog, "materials");
  await assertCloseVisible(page, dialog);
  await page.screenshot({ path: path.join(screenshotFolder, "product-help-check-375.png") });
  await dialog.getByRole("button", { name: "关闭使用指南", exact: true }).click();
  await expect(dialog).not.toBeVisible();
  await expect(trigger).toBeFocused();
  await expect(editor).toHaveValue(draft);
  await expect(page.getByRole("textbox", { name: "快速文本故事作用域" })).toHaveValue("route_a");
  await expect(page.getByRole("combobox", { name: "快速文本类型" })).toHaveValue("chapter");
  await assertReadingOnly(page, state, before);
});

test("剧情策划从报告定位阅读与覆盖帮助，保留筛选选中问题和未提交备注", async ({ page }) => {
  const state = initialState();
  await page.setViewportSize({ width: 1440, height: 1000 });
  const query = `?category=fact_conflict&status=unreviewed&issue=${issueId}`;
  await openPage(page, state, `/app/projects/${projectId}/runs/${runId}/report${query}`);
  const article = page.locator(`#issue-${issueId}`);
  await expect(article).toBeVisible();
  const note = article.locator(".note");
  const feedbackDraft = "我还需核对当前支线的前一段对白，尚未提交反馈。";
  await note.fill(feedbackDraft);
  const before = await readingSnapshot(page, state);
  const reportTrigger = page.getByRole("button", { name: "阅读报告帮助", exact: true });
  await reportTrigger.click();
  const dialog = guide(page);
  await expect(topicHeading(dialog, "report")).toBeFocused();
  await expect(topicHeading(dialog, "report")).toBeInViewport({ ratio: 1 });
  const reportTopic = dialog.getByRole("region", { name: "阅读报告与原文", exact: true });
  await expect(reportTopic).toContainText("仍需作者核对");
  await expect(reportTopic).toContainText("角色审查线索");
  await expect(reportTopic).toContainText("未验证模型提案");
  await expect(reportTopic).toContainText("不会拿最新版顶替");
  await expect(reportTopic).toContainText("不会自动修改文稿或训练模型");
  await assertCloseVisible(page, dialog);
  await page.screenshot({ path: path.join(screenshotFolder, "product-help-report-1440.png") });
  await page.keyboard.press("Escape");
  await expect(reportTrigger).toBeFocused();

  const coverageTrigger = page.getByRole("button", { name: "理解覆盖范围", exact: true });
  await coverageTrigger.click();
  await expect(topicHeading(dialog, "coverage")).toBeFocused();
  await expect(topicHeading(dialog, "coverage")).toBeInViewport({ ratio: 1 });
  const coverageTopic = dialog.getByRole("region", { name: "覆盖范围与异常", exact: true });
  await expect(coverageTopic).toContainText("通用抽取与角色审查要分别看");
  await expect(coverageTopic).toContainText("进度 100% 或任务完成，仍可能伴随部分覆盖");
  await expect(coverageTopic).toContainText("零问题只表示当前范围和能力下没有形成正式报告问题");
  await expect(coverageTopic).toContainText("不能替代 AI");
  await selectTopic(dialog, "characters");
  await dialog.getByRole("button", { name: "关闭使用指南", exact: true }).click();
  await expect(coverageTrigger).toBeFocused();
  await expect(note).toHaveValue(feedbackDraft);
  await expect(page.locator(".reportFilters").getByRole("combobox", { name: "问题类别" })).toHaveValue("fact_conflict");
  await expect(page.locator(".reportFilters").getByRole("combobox", { name: "反馈状态" })).toHaveValue("unreviewed");
  await expect(article.getByRole("button", { name: "当前选中问题" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".issues article")).toHaveCount(1);
  await expect(page.getByRole("region", { name: /待复核线索/ })).toContainText("模型提案（未验证）");
  await assertReadingOnly(page, state, before);
});

test("360像素报告指南的目录可定位末节，正文滚动后关闭仍可见", async ({ page }) => {
  const state = initialState();
  await page.setViewportSize({ width: 360, height: 812 });
  await openPage(page, state, `/app/projects/${projectId}/runs/${runId}/report`);
  const trigger = page.getByRole("button", { name: "阅读报告帮助", exact: true });
  const before = await readingSnapshot(page, state);
  await trigger.click();
  const dialog = guide(page);
  await expect(topicHeading(dialog, "report")).toBeInViewport({ ratio: 1 });
  await selectTopic(dialog, "coverage");
  await assertCloseVisible(page, dialog);
  const reading = dialog.getByRole("region", { name: "使用指南正文" });
  await reading.evaluate((element) => { element.scrollTop = element.scrollHeight; });
  await assertCloseVisible(page, dialog);
  expect(await reading.evaluate((element) => element.scrollWidth <= element.clientWidth + 1)).toBe(true);
  await page.screenshot({ path: path.join(screenshotFolder, "product-help-report-360.png") });
  await dialog.getByRole("button", { name: "关闭使用指南", exact: true }).click();
  await expect(dialog).not.toBeVisible();
  await expect(trigger).toBeFocused();
  await assertReadingOnly(page, state, before);
});

test.describe("1440×1000物理像素的200%缩放等效布局", () => {
  // Browser zoom at 200% halves the effective CSS viewport. Render that
  // 720×500 layout at a 2× pixel ratio; do not use CSS zoom, whose viewport
  // units and media queries behave differently. This is an equivalent reflow
  // check, not a claim to have driven Chromium's browser-menu setting.
  test.use({ viewport: { width: 720, height: 500 }, deviceScaleFactor: 2 });

  test("重排后目录和关闭可操作，帮助不改业务状态", async ({ page }) => {
    const state = initialState();
    await openPage(page, state, `/app/projects/${projectId}/runs/${runId}/report`);
    const trigger = page.getByRole("button", { name: "理解覆盖范围", exact: true });
    const before = await readingSnapshot(page, state);
    await trigger.click();
    const dialog = guide(page);
    await expect(topicHeading(dialog, "coverage")).toBeInViewport({ ratio: 1 });
    await selectTopic(dialog, "materials");
    await assertCloseVisible(page, dialog);
    await page.screenshot({ path: path.join(screenshotFolder, "product-help-200-percent-equivalent-1440.png") });
    await page.keyboard.press("Escape");
    await expect(dialog).not.toBeVisible();
    await expect(trigger).toBeFocused();
    await assertReadingOnly(page, state, before);
  });
});
