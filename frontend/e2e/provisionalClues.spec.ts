import { expect, test, type Page } from "@playwright/test";

const projectId = "provisional-project";
const runId = "provisional-run";
const now = "2026-09-27T00:00:00Z";
const run = {
  id: runId,
  project_id: projectId,
  status: "completed",
  prompt_tokens: 12,
  completion_tokens: 4,
  estimated_cost_usd: 0,
  created_at: now,
};
const secondRunId = "provisional-run-2";
const secondRun = { ...run, id: secondRunId };
const clue = {
  id: `pc_${"a".repeat(32)}`,
  document_id: "draft-1",
  document_version: 2,
  document_name: "新稿.md",
  line_start: 12,
  line_end: 13,
  evidence: "林澈收起钥匙。\n她沉默地离开。",
  character: "林澈",
  dimension: "behavior_boundary",
  proposed_statement: "林澈可能改变了既有行为边界。",
  reason: "partial_model_package",
};
const issue = {
  id: "issue-1",
  category: "fact_conflict",
  severity: "medium",
  confidence: 0.8,
  title: "正式问题示例",
  explanation: "两处事实陈述冲突。",
  suggestion: "核对正式设定。",
  evidence: [{ document_id: "draft-1", document_name: "新稿.md", line_start: 20, line_end: 20, text: "正式问题证据。" }],
};
const reviewClue = {
  id: "71475dcc-1d0c-4dd2-a014-06941adf1bf3",
  report_class: "review_clue",
  category: "character_drift",
  severity: "low",
  confidence: 0.52,
  title: "林澈的角色表现需要确认",
  explanation: "当前只见一次不同表现，尚不能判定角色设定改变。",
  suggestion: "核对角色设定与上下文。",
  evidence: [
    { document_id: "profile", document_name: "设定.md", line_start: 3, line_end: 3, text: "林澈在陌生人前沉默寡言。" },
    { document_id: "draft", document_name: "新稿.md", line_start: 12, line_end: 12, text: "林澈主动在众人面前长谈。" },
  ],
  metadata: { final_outcome: "needs_confirmation", review_reason: "single_behavior_is_not_drift" },
};

type MockState = {
  clueFailures: number;
  clueResponse: unknown;
  issueResponse?: unknown[];
  unexpected: string[];
  feedbackPosts: number;
  exportReads: number;
  reviewClueResponse?: unknown;
  reviewClueFailures?: number;
  secondClueResponse?: unknown;
  firstClueDelay?: Promise<void>;
  firstClueRequests?: number;
  firstClueSettled?: boolean;
};

async function mockCompletedReport(page: Page, state: MockState) {
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    let body: unknown;
    if (path === "/api/v1/auth/me") body = {
      mode: "anonymous",
      user: { id: "author", email: "author@example.test", display_name: "作者" },
      workspace: { id: "workspace", name: "创作工作区", kind: "personal", role: "owner" },
    };
    else if (path === "/api/v1/project-catalog") body = {
      page: 1, page_size: 40, total: 1,
      items: [{ id: projectId, name: "线索测试项目", description: "", created_at: now,
        active_document_count: 0, latest_run: { id: runId, status: "completed", created_at: now } }],
    };
    else if (path === "/api/v1/account/model-provider") body = {};
    else if (path === `/api/v1/projects/${projectId}/documents`) body = [];
    else if (path === `/api/v1/projects/${projectId}/analysis-runs`) body = state.secondClueResponse ? [run, secondRun] : [run];
    else if (path === `/api/v1/analysis-runs/${runId}`) body = run;
    else if (path === `/api/v1/analysis-runs/${secondRunId}` && state.secondClueResponse) body = secondRun;
    else if (path === `/api/v1/analysis-runs/${runId}/issues` || (state.secondClueResponse && path === `/api/v1/analysis-runs/${secondRunId}/issues`)) body = state.issueResponse ?? [issue];
    else if (path === `/api/v1/issues/${issue.id}/feedback`) {
      if (route.request().method() === "POST") {
        state.feedbackPosts += 1;
        body = { id: "feedback-1", label: "accepted", comment: "", created_at: now };
      } else body = { latest: null };
    }
    else if (path === `/api/v1/analysis-runs/${runId}/records` || (state.secondClueResponse && path === `/api/v1/analysis-runs/${secondRunId}/records`)) body = { records: [], warnings: [] };
    else if (path === `/api/v1/analysis-runs/${runId}/diagnostics` || (state.secondClueResponse && path === `/api/v1/analysis-runs/${secondRunId}/diagnostics`)) body = {};
    else if (path === `/api/v1/analysis-runs/${runId}/clarifications` || (state.secondClueResponse && path === `/api/v1/analysis-runs/${secondRunId}/clarifications`)) body = [];
    else if (path === `/api/v1/analysis-runs/${runId}/export.md`) {
      state.exportReads += 1;
      await route.fulfill({ status: 200, contentType: "text/markdown", body: "# 正式问题\n" });
      return;
    }
    else if (path === `/api/v1/analysis-runs/${runId}/provisional-clues`) {
      state.firstClueRequests = (state.firstClueRequests || 0) + 1;
      if (state.firstClueDelay) {
        await state.firstClueDelay;
      }
      if (state.clueFailures > 0) {
        state.clueFailures -= 1;
        await route.fulfill({ status: 500, json: { detail: "temporarily unavailable" } });
        return;
      }
      body = state.clueResponse;
    }
    else if (state.secondClueResponse && path === `/api/v1/analysis-runs/${secondRunId}/provisional-clues`) {
      body = state.secondClueResponse;
    }
    else if (path === `/api/v1/analysis-runs/${runId}/review-clues`) {
      if (state.reviewClueFailures) {
        state.reviewClueFailures -= 1;
        await route.fulfill({ status: 500, json: { detail: "temporarily unavailable" } });
        return;
      }
      body = state.reviewClueResponse ?? { items: [], truncated: false };
    }
    else if (state.secondClueResponse && path === `/api/v1/analysis-runs/${secondRunId}/review-clues`) {
      body = { items: [], truncated: false };
    }
    else {
      state.unexpected.push(`${route.request().method()} ${path}`);
      await route.fulfill({ status: 404, json: { detail: "unexpected mocked endpoint" } });
      return;
    }
    if (state.firstClueDelay && path === `/api/v1/analysis-runs/${runId}/provisional-clues`) {
      try {
        await route.fulfill({ status: 200, json: body });
      } catch {
        // The old run's in-flight request may have been aborted on navigation.
      }
      state.firstClueSettled = true;
      return;
    }
    await route.fulfill({ status: 200, json: body });
  });
}

test("pending clues remain a separate reading section and do not become formal issues", async ({ page }, testInfo) => {
  const state: MockState = {
    clueFailures: 0, clueResponse: { items: [clue], truncated: false },
    unexpected: [], feedbackPosts: 0, exportReads: 0,
  };
  await mockCompletedReport(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);

  const clues = page.getByRole("region", { name: /待复核线索/ });
  const formalIssues = page.locator(".issues");
  await expect(clues.getByRole("heading", { name: /待复核线索 1/ })).toBeVisible();
  await expect(clues).toContainText("模型提案（未验证）");
  await expect(clues).toContainText(clue.proposed_statement);
  await expect(clues).toContainText(clue.evidence);
  await expect(clues).toContainText("不计入正式问题数");
  await expect(clues.getByRole("button")).toHaveCount(0);
  await expect(clues.getByRole("link")).toHaveCount(0);
  await expect(formalIssues.getByRole("heading", { name: /正式一致性问题 1/ })).toBeVisible();
  await expect(formalIssues.locator("article")).toHaveCount(1);
  await expect(formalIssues).toContainText(issue.title);
  await expect(formalIssues).not.toContainText(clue.proposed_statement);
  await expect(page.locator(".railSummary")).toContainText("正式一致性问题");
  await expect(page.locator(".railSummary")).not.toContainText("待复核线索");
  await formalIssues.getByRole("button", { name: "已接受" }).click();
  await expect(formalIssues).toContainText("当前反馈：已接受");
  await formalIssues.getByRole("button", { name: "导出 Markdown 报告" }).click();
  await expect.poll(() => state.exportReads).toBe(1);
  await expect(formalIssues.getByRole("status")).toContainText("不包含待复核线索");
  expect(state.feedbackPosts).toBe(1);

  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await page.setViewportSize({ width: 1440, height: 900 });
    await clues.screenshot({ path: testInfo.outputPath("provisional-clues-1440.png") });
    await page.screenshot({ path: "../artifacts/playwright/provisional-report-1440.png", fullPage: true });
  }

  await page.setViewportSize({ width: 375, height: 812 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await expect(clues).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(await page.locator(".mainCanvas").evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await clues.screenshot({ path: testInfo.outputPath("provisional-clues-375.png") });
    await page.screenshot({ path: "../artifacts/playwright/provisional-report-375.png", fullPage: true });
  }
  expect(state.unexpected).toEqual([]);
});

test("zero formal issues with an unverified clue never reads as a clean pass", async ({ page }) => {
  const state: MockState = {
    clueFailures: 0, clueResponse: { items: [clue], truncated: false },
    issueResponse: [], unexpected: [], feedbackPosts: 0, exportReads: 0,
  };
  await mockCompletedReport(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);

  const formalIssues = page.locator(".issues");
  const clues = page.getByRole("region", { name: /待复核线索/ });
  await expect(formalIssues.getByRole("heading", { name: "正式一致性问题 0" })).toBeVisible();
  await expect(formalIssues).toContainText("另有 1 条待复核线索列于下方，未计入正式问题");
  await expect(formalIssues).not.toContainText(clue.proposed_statement);
  await expect(clues).toContainText(clue.proposed_statement);
  await expect(page.locator(".railSummary")).toContainText("正式一致性问题");
  await expect(page.locator(".railSummary")).not.toContainText("待复核线索");
  expect(state.unexpected).toEqual([]);
});

test("one contrary character action is a dual-evidence review clue, not a formal issue", async ({ page }) => {
  const state: MockState = {
    clueFailures: 0,
    clueResponse: { items: [], truncated: false },
    reviewClueResponse: { items: [reviewClue], truncated: false },
    issueResponse: [], unexpected: [], feedbackPosts: 0, exportReads: 0,
  };
  await mockCompletedReport(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);

  const formalIssues = page.locator(".issues");
  const clues = page.getByRole("region", { name: /待复核线索/ });
  await expect(formalIssues.getByRole("heading", { name: "正式一致性问题 0" })).toBeVisible();
  await expect(formalIssues).toContainText("另有 1 条待复核线索列于下方，未计入正式问题");
  await expect(formalIssues).not.toContainText(reviewClue.title);
  await expect(clues.getByRole("heading", { name: "待复核线索 1" })).toBeVisible();
  await expect(clues.getByRole("heading", { name: "角色审查线索 1" })).toBeVisible();
  await expect(clues.getByRole("heading", { name: "模型提案（未验证） 0" })).toBeVisible();
  await expect(clues).toContainText("待作者确认");
  await expect(clues).toContainText(reviewClue.title);
  await expect(clues).toContainText(`既有设定 · ${reviewClue.evidence[0].document_name}`);
  await expect(clues).toContainText(reviewClue.evidence[0].text);
  await expect(clues).toContainText(`当前新稿 · ${reviewClue.evidence[1].document_name}`);
  await expect(clues).toContainText(reviewClue.evidence[1].text);
  await expect(clues.getByRole("button")).toHaveCount(0);
  await expect(clues.getByRole("link")).toHaveCount(0);
  await expect(page.locator(".railSummary")).toContainText("正式一致性问题");
  await expect(page.locator(".railSummary")).not.toContainText(reviewClue.title);
  await page.setViewportSize({ width: 375, height: 812 });
  await expect(clues).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(await page.locator(".mainCanvas").evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
  expect(state.feedbackPosts).toBe(0);
  expect(state.exportReads).toBe(0);
  expect(state.unexpected).toEqual([]);
});

test("legacy clues with unverifiable evidence are disclosed but never promoted or counted", async ({ page }) => {
  const state: MockState = {
    clueFailures: 0, clueResponse: { items: [], truncated: false },
    reviewClueResponse: { items: [], truncated: false, unavailable_count: 2 },
    issueResponse: [], unexpected: [], feedbackPosts: 0, exportReads: 0,
  };
  await mockCompletedReport(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);

  const formalIssues = page.locator(".issues");
  const clues = page.getByRole("region", { name: /待复核线索/ });
  await expect(formalIssues.getByRole("heading", { name: "正式一致性问题 0" })).toBeVisible();
  await expect(formalIssues).toContainText("另有 2 条历史线索因证据不足无法展示，建议重新分析");
  await expect(clues.getByRole("heading", { name: "待复核线索 0" })).toBeVisible();
  await expect(clues.getByRole("alert")).toContainText("2 条历史线索证据不足，无法展示；建议重新分析");
  await expect(clues.getByRole("button")).toHaveCount(0);
  expect(state.feedbackPosts).toBe(0);
  expect(state.exportReads).toBe(0);
  expect(state.unexpected).toEqual([]);
});

test("scan-limited legacy rows do not masquerade as a complete zero clue result", async ({ page }) => {
  const state: MockState = {
    clueFailures: 0, clueResponse: { items: [], truncated: false },
    reviewClueResponse: { items: [], truncated: false, unavailable_count: 1025, scan_limited: true },
    issueResponse: [], unexpected: [], feedbackPosts: 0, exportReads: 0,
  };
  await mockCompletedReport(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);
  const formalIssues = page.locator(".issues");
  const clues = page.getByRole("region", { name: /待复核线索/ });
  await expect(formalIssues).toContainText("历史线索数量超过单页安全扫描上限，后续条目尚未核对");
  await expect(clues.getByRole("alert").last()).toContainText("后续条目尚未核对；当前数量只是已核对部分");
  await expect(clues.getByRole("heading", { name: "待复核线索 0" })).toBeVisible();
  expect(state.unexpected).toEqual([]);
});

test("review clue failure is unknown until retry, while model proposals and formal issues stay readable", async ({ page }) => {
  const state: MockState = {
    clueFailures: 0,
    clueResponse: { items: [clue], truncated: false },
    reviewClueResponse: { items: [reviewClue], truncated: false },
    reviewClueFailures: 1,
    issueResponse: [], unexpected: [], feedbackPosts: 0, exportReads: 0,
  };
  await mockCompletedReport(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);

  const clues = page.getByRole("region", { name: /待复核线索/ });
  const formalIssues = page.locator(".issues");
  await expect(clues).toContainText("角色审查线索读取失败，当前无法确认数量");
  await expect(clues.getByRole("heading", { name: "待复核线索 —" })).toBeVisible();
  await expect(clues).toContainText(clue.proposed_statement);
  await expect(formalIssues).toContainText("部分待复核线索读取失败，无法确认完整数量");
  await clues.getByRole("button", { name: "重试读取角色审查线索" }).click();
  await expect(clues.getByRole("heading", { name: "待复核线索 2" })).toBeVisible();
  await expect(clues).toContainText(reviewClue.evidence[1].text);
  expect(state.unexpected).toEqual([]);
});

test("truncated review clues and model proposals show lower bounds rather than exact totals", async ({ page }) => {
  const state: MockState = {
    clueFailures: 0,
    clueResponse: {
      items: Array.from({ length: 64 }, (_, index) => ({
        ...clue, id: `pc_${index.toString(16).padStart(32, "0")}`,
      })),
      truncated: true,
    },
    reviewClueResponse: {
      items: Array.from({ length: 64 }, (_, index) => ({
        ...reviewClue, id: `00000000-0000-4000-8000-${index.toString(16).padStart(12, "0")}`,
      })),
      truncated: true,
    },
    issueResponse: [], unexpected: [], feedbackPosts: 0, exportReads: 0,
  };
  await mockCompletedReport(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);

  const clues = page.getByRole("region", { name: /待复核线索/ });
  await expect(clues.getByRole("heading", { name: "待复核线索 128+" })).toBeVisible();
  await expect(clues.getByRole("heading", { name: "角色审查线索 64+" })).toBeVisible();
  await expect(clues.getByRole("heading", { name: "模型提案（未验证） 64+" })).toBeVisible();
  await expect(clues).toContainText("角色审查线索超过展示上限");
  await expect(clues).toContainText("模型提案超过展示上限");
  await expect(page.locator(".issues")).toContainText("另有至少 128 条待复核线索列于下方，部分未展示");
  await expect(page.locator(".issues")).not.toContainText("另有 128 条待复核线索");
  expect(state.unexpected).toEqual([]);
});

test("clue loading failure is distinct from zero clues and does not hide formal issues", async ({ page }) => {
  const state: MockState = {
    clueFailures: 1, clueResponse: { items: [], truncated: false },
    unexpected: [], feedbackPosts: 0, exportReads: 0,
  };
  await mockCompletedReport(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);

  const clues = page.getByRole("region", { name: /待复核线索/ });
  await expect(clues).toContainText("模型提案读取失败，当前无法确认本次是否有提案");
  await expect(clues).not.toContainText("本次没有可安全定位的待复核线索");
  await expect(page.locator(".issues")).toContainText(issue.title);
  await clues.getByRole("button", { name: "重试读取模型提案" }).click();
  await expect(clues).toContainText("本次没有可安全定位的模型提案");
  await expect(clues.getByRole("heading", { name: /待复核线索 0/ })).toBeVisible();
  expect(state.unexpected).toEqual([]);
});

test("a late clue response from a previous run cannot replace the selected run", async ({ page }) => {
  let releaseOldClue: () => void = () => {};
  const firstClueDelay = new Promise<void>((resolve) => { releaseOldClue = resolve; });
  const secondClue = {
    ...clue,
    id: `pc_${"b".repeat(32)}`,
    proposed_statement: "第二次运行的独立待复核提案。",
  };
  const state: MockState = {
    clueFailures: 0,
    clueResponse: { items: [clue], truncated: false },
    secondClueResponse: { items: [secondClue], truncated: false },
    firstClueDelay,
    unexpected: [], feedbackPosts: 0, exportReads: 0,
  };
  await mockCompletedReport(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);
  const clues = page.getByRole("region", { name: /待复核线索/ });
  await expect(clues).toContainText("正在读取待复核线索");
  await page.locator(".sideNav").getByRole("button", { name: "运行审计" }).click();
  const history = page.getByRole("heading", { name: "运行历史（冻结输入）" }).locator("..");
  await history.getByRole("row").nth(2).getByRole("button", { name: "查看报告" }).click();
  await expect(page).toHaveURL(new RegExp(`/runs/${secondRunId}/report`));
  await expect(clues).toContainText(secondClue.proposed_statement);
  releaseOldClue();
  await expect.poll(() => state.firstClueSettled).toBe(true);
  await expect(clues).not.toContainText(clue.proposed_statement);
  expect(state.unexpected).toEqual([]);
});
