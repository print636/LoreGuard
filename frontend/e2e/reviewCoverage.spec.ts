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

async function mockCompletedRun(page: Page, state: { character: CharacterDiagnostic; unexpected: string[] }) {
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
    else if (path === `/api/v1/projects/${projectId}/analysis-runs`) response = [run];
    else if (path === `/api/v1/analysis-runs/${runId}`) response = run;
    else if (path === `/api/v1/analysis-runs/${runId}/issues`) response = [];
    else if (path === `/api/v1/analysis-runs/${runId}/records`) response = { records: [], warnings: [] };
    else if (path === `/api/v1/analysis-runs/${runId}/diagnostics`) response = {
      model: completeModel, character_consistency: state.character,
    };
    else if (path === `/api/v1/analysis-runs/${runId}/clarifications`) response = [];
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
  await expect(coverage).toContainText("角色审查覆盖不完整");
  await expect(coverage).toContainText("调用模型 6 · 完成 6 · 未完成 1（其中未调用 1）");
  await expect(coverage).toContainText("Token 预算门控");
  await expect(coverage).not.toContainText("bounded_partial");
  await expect(page.locator(".railCoverage")).toContainText("覆盖仍有限");
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
  await expect(page.locator(".railCoverage")).toContainText("覆盖仍有限");
  expect(state.unexpected).toEqual([]);
});
