import { expect, test, type Page } from "@playwright/test";

const projectId = "coverage-project";
const runId = "coverage-run";
const run = {
  id: runId, project_id: projectId, status: "completed",
  prompt_tokens: 12, completion_tokens: 4, estimated_cost_usd: 0,
  created_at: "2026-09-26T00:00:00Z",
};
const completeModel = {
  enabled: true, configured: true, total_chunks: 1, attempted_chunks: 1,
  succeeded_chunks: 1, failed_chunks: 0, skipped_chunks: 0,
  invalid_records: 0, empty_response_chunks: 0, reason_codes: [],
};

type CharacterDiagnostic = Record<string, unknown>;

async function mockCompletedRun(page: Page, state: { character: CharacterDiagnostic; choice?: boolean; unexpected: string[] }) {
  const selectedRun = state.choice === undefined ? run : {
    ...run, review_batch: { mode: "draft_review", sensitivity: "balanced", target_document_ids: ["draft"],
      background_document_ids: ["baseline"], character_review_enabled: state.choice },
  };
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();
    let response: unknown;
    if (path === "/api/v1/auth/me") response = {
      mode: "anonymous",
      user: { id: "author", email: "author@example.test", display_name: "作者" },
      workspace: { id: "workspace", name: "创作工作区", kind: "personal", role: "owner" },
    };
    else if (path === "/api/v1/project-catalog" && method === "GET") {
      const item = {
        id: projectId, name: "分节覆盖验收", description: "", active_document_count: 0,
        created_at: "2026-09-26T00:00:00Z",
        latest_run: { id: runId, status: "completed", created_at: run.created_at, model_execution: null },
      };
      const page = Number(url.searchParams.get("page") || 1);
      const pageSize = Number(url.searchParams.get("page_size") || 40);
      const matches = (!url.searchParams.get("project_id") || url.searchParams.get("project_id") === projectId)
        && (!url.searchParams.get("query") || item.name.includes(url.searchParams.get("query")!.trim()));
      response = { page, page_size: pageSize, total: matches ? 1 : 0,
        items: matches && page === 1 ? [item] : [] };
    }
    else if (path === "/api/v1/account/model-provider") response = {};
    else if (path === `/api/v1/projects/${projectId}/documents`) response = [];
    else if (path === `/api/v1/projects/${projectId}/analysis-runs`) response = [selectedRun];
    else if (path === `/api/v1/projects/${projectId}/run-catalog` && method === "GET") {
      const pageNumber = Number(url.searchParams.get("page") || 1);
      const pageSize = Number(url.searchParams.get("page_size") || 20);
      const status = url.searchParams.get("status") || "all";
      const all = status === "all" || status === run.status ? [run] : [];
      const offset = (pageNumber - 1) * pageSize;
      response = { project_id: projectId, page: pageNumber, page_size: pageSize, total: all.length,
        has_more: offset + pageSize < all.length,
        items: all.slice(offset, offset + pageSize).map((item) => ({
          ...item, started_at: null, completed_at: item.created_at, input_chars: 0,
          frozen_document_count: 0, retried_from: null, batch_mode: null,
        })) };
    }
    else if (path === `/api/v1/analysis-runs/${runId}`) response = selectedRun;
    else if (path === `/api/v1/analysis-runs/${runId}/issues`) response = [];
    else if (path === `/api/v1/analysis-runs/${runId}/records`) response = { records: [], warnings: [] };
    else if (path === `/api/v1/analysis-runs/${runId}/diagnostics`) response = {
      model: completeModel, character_consistency: state.character,
    };
    else if (path === `/api/v1/analysis-runs/${runId}/clarifications`) response = [];
    else if (path === `/api/v1/analysis-runs/${runId}/provisional-clues`) response = { items: [], truncated: false };
    else if (path === `/api/v1/analysis-runs/${runId}/review-clues`) response = { items: [], truncated: false, unavailable_count: 0 };
    else {
      state.unexpected.push(`${route.request().method()} ${path}`);
      await route.fulfill({ status: 404, json: { detail: "unexpected mocked endpoint" } });
      return;
    }
    await route.fulfill({ status: 200, json: response });
  });
}

test("completed task with one uncalled character chunk stays visibly partial", async ({ page }, testInfo) => {
  const state = {
    character: {
      enabled: true, outcome: "partial", reason_code: "bounded_partial",
      snapshot_bound: true, material_coverage: "partial",
      counts: {
        planned_chunks: 7, processed_chunks: 7, model_called_chunks: 6,
        model_completed_chunks: 6, model_uncalled_chunks: 1, model_incomplete_chunks: 1,
      },
      reason_counts: { token_budget: 1 },
    },
    unexpected: [] as string[],
  };
  await mockCompletedRun(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}`);
  const coverage = page.locator(".reviewCoverage");
  await expect(coverage).toContainText("角色审查仅覆盖部分内容");
  await expect(coverage).toContainText("调用模型 6 · 完成 6 · 未完成 1（其中未调用 1）");
  await expect(coverage).toContainText("Token 预算门控");
  await expect(coverage).not.toContainText("bounded_partial");
  await expect(coverage).toBeVisible();
  await expect(coverage).toContainText("角色审查仅覆盖部分内容");
  await page.getByText("分析诊断摘要", { exact: true }).click();
  await expect(page.getByText("角色审查主抽取", { exact: true })).toBeVisible();
  await expect(page.locator(".diagnosticGrid")).toContainText("计划 7 · 调用模型 6 · 完成 6");
  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await coverage.screenshot({ path: testInfo.outputPath("partial-coverage-desktop.png") });
  }
  await page.setViewportSize({ width: 360, height: 1000 });
  await expect(coverage).toBeVisible();
  await expect(coverage).toContainText("未调用 1");
  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await coverage.screenshot({ path: testInfo.outputPath("partial-coverage-mobile.png") });
  }
  expect(state.unexpected).toEqual([]);
});

test("contradictory completed stage cannot claim complete coverage", async ({ page }) => {
  const state = {
    character: {
      enabled: true, outcome: "completed", reason_code: "completed",
      snapshot_bound: true, material_coverage: "partial",
      counts: {
        planned_chunks: 7, processed_chunks: 7, model_called_chunks: 6,
        model_completed_chunks: 6, model_uncalled_chunks: 1, model_incomplete_chunks: 1,
      },
    },
    unexpected: [] as string[],
  };
  await mockCompletedRun(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}`);
  const coverage = page.locator(".reviewCoverage");
  await expect(coverage).toContainText("角色审查覆盖无法核对");
  await expect(coverage).toContainText("阶段完成");
  await expect(coverage).not.toContainText("AI 语义与角色审查完整");
  await expect(coverage).toBeVisible();
  await expect(coverage).toContainText("角色审查覆盖无法核对");
  expect(state.unexpected).toEqual([]);
});

const processedStage = {
  enabled: true, snapshot_bound: true, outcome: "completed", material_coverage: "complete",
  counts: { planned_chunks: 1, model_called_chunks: 1, model_completed_chunks: 1,
    model_uncalled_chunks: 0, model_incomplete_chunks: 0 },
};

test("author report distinguishes a processed draft without a matching observation from a judged target", async ({ page }, testInfo) => {
  const state = {
    choice: true,
    character: {
      ...processedStage,
      target_assessment_summary_v1: {
        schema_version: "character-target-assessment-summary-v1", total_targets: 2,
        conflict: 0, no_issue: 1, no_matching_observation: 1, needs_review: 0, unprocessed: 0,
      },
    }, unexpected: [] as string[],
  };
  await mockCompletedRun(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);
  const scope = page.getByRole("region", { name: "本次审查范围" });
  await expect(scope).toContainText("提交时已勾选角色审查");
  await expect(scope).toContainText("角色审查材料处理完成");
  await expect(scope).toContainText("1 项未发现冲突，1 项未匹配到新稿表现");
  await expect(page.getByRole("heading", { name: "正式一致性问题 0" })).toBeVisible();
  await expect(page.locator(".issues .empty")).toContainText("1 项未匹配到新稿表现");
  await expect(scope).toBeVisible();
  await expect(scope).toContainText("1 项未匹配到新稿表现");
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const [width, height] of [[375, 812], [812, 375]]) {
    await page.setViewportSize({ width, height });
    await expect(scope).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
      await scope.screenshot({ path: testInfo.outputPath(`target-summary-${width}.png`) });
    }
  }
  expect(state.unexpected).toEqual([]);
});

test("a model service failure leaves the affected setting open in the completed report", async ({ page }) => {
  const state = {
    choice: true,
    character: {
      ...processedStage, outcome: "partial", material_coverage: "partial",
      reason_counts: { upstream_5xx: 1, targeted_pass_upstream_5xx: 1 },
      target_assessment_summary_v1: {
        schema_version: "character-target-assessment-summary-v1", total_targets: 1,
        conflict: 0, no_issue: 0, no_matching_observation: 0, needs_review: 1, unprocessed: 0,
      },
    }, unexpected: [] as string[],
  };
  await mockCompletedRun(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);
  const scope = page.getByRole("region", { name: "本次审查范围" });
  await expect(scope).toContainText("模型服务出错");
  await expect(scope).toContainText("服务恢复后发起新一轮校验");
  await expect(page.locator(".issues .empty")).toContainText("1 项证据不足或待确认");
  await expect(scope).not.toContainText("未发现冲突");
  expect(state.unexpected).toEqual([]);
});

test("报告区分未发出的核对、回复超时和未能对应原文的行为", async ({ page }) => {
  const state = {
    choice: true, unexpected: [] as string[],
    character: {
      ...processedStage, outcome: "partial", material_coverage: "partial",
      counts: { planned_chunks: 2, model_called_chunks: 2, model_completed_chunks: 1, model_uncalled_chunks: 0,
        model_incomplete_chunks: 1, draft_observation_count: 0, drift_considered: 0, drift_reviewed: 0 },
      reason_counts: { target_bound_provider_timeout: 5, targeted_pass_target_bound_scene_time_insufficient: 1,
        targeted_pass_habit_paraphrase_ambiguous_source: 1 },
      target_assessment_summary_v1: {
        schema_version: "character-target-assessment-summary-v1", total_targets: 1,
        conflict: 0, no_issue: 0, no_matching_observation: 0, needs_review: 1, unprocessed: 0,
      },
    },
  };
  await mockCompletedRun(page, state);
  await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);
  const coverage = page.getByRole("region", { name: "本次审查范围" });
  await expect(coverage).toContainText("核对请求未发出");
  await expect(coverage).toContainText("核对已发起，但模型请求超时");
  await expect(coverage).toContainText("概括未能对应到原文中的具体动作");
  await expect(coverage).toContainText("如需补查，可重新发起校验");
  await expect(page.locator(".issues .empty")).toContainText("相关设定与新稿是否一致仍未判断");
  await expect(coverage).not.toContainText("未发现冲突");
  expect(state.unexpected).toEqual([]);
});

for (const optedOut of [true, false]) {
  test(`author report preserves ${optedOut ? "explicit opt-out" : "historical unknown choice and judgments"}`, async ({ page }) => {
    const state = {
      choice: optedOut ? false : undefined,
      character: optedOut ? { enabled: false, outcome: "disabled", reason_code: "feature_disabled" }
        : { ...processedStage, case_trace: [{ final_outcome: "no_issue" }] },
      unexpected: [] as string[],
    };
    await mockCompletedRun(page, state);
    await page.goto(`/app/projects/${projectId}/runs/${runId}/report`);
    const scope = page.getByRole("region", { name: "本次审查范围" });
    await expect(scope).toContainText(optedOut ? "提交时未勾选" : "未记录提交时是否勾选");
    await expect(page.locator(".issues .empty")).toContainText(optedOut ? "本次未选择角色审查" : "无法确认哪些角色设定得到判断");
    await expect(scope).not.toContainText("角色审查完整");
    expect(state.unexpected).toEqual([]);
  });
}
