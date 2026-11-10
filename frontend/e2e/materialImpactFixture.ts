import { expect, type Route } from "@playwright/test";

/** Legacy interaction fixtures have no verified source comparison. Do not invent one. */
export async function fulfillLegacyMaterialImpact(route: Route, runs: Array<{ projectId: string; runId: string; changed?: boolean }>): Promise<boolean> {
  const url = new URL(route.request().url());
  const run = runs.find(item => url.pathname === `/api/v1/analysis-runs/${encodeURIComponent(item.runId)}/material-impact`);
  if (route.request().method() !== "GET" || !run) return false;
  expect([...url.searchParams.keys()].sort()).toEqual(["issue_limit", "issue_offset", "limit", "offset"]);
  expect(url.searchParams.get("limit")).toBe("30");
  expect(url.searchParams.get("issue_limit")).toBe("30");
  for (const key of ["offset", "issue_offset"]) expect(url.searchParams.get(key)).toMatch(/^\d+$/);
  const page = (offset: string) => ({ items: [], offset: Number(url.searchParams.get(offset)), limit: 30, total: 0, has_more: false });
  await route.fulfill({ json: {
    schema_version: "material-change-impact-v1", project_id: run.projectId, run_id: run.runId,
    comparison_status: run.changed ? "complete" : "unavailable", semantic_impact_complete: false,
    confirmed_traits_auto_invalidated: false, confirmed_traits_changed: run.changed ? false : null,
    recheck: { inputs_changed: run.changed ? true : null, can_prepare: run.changed === true, reason: run.changed ? null : "snapshot_unavailable" },
    sources: page("offset"), affected_issues: page("issue_offset"),
    summary: { changed_sources: 0, added_sources: 0, not_included_sources: 0, affected_issues: 0, unresolved_evidence: 0, unresolved_sources: 0 },
  } });
  return true;
}

export async function fulfillLegacyRunCatalog(route: Route, projects: Array<{ projectId: string; runs: Array<Record<string, unknown>> }>): Promise<boolean> {
  const url = new URL(route.request().url());
  const project = projects.find(item => url.pathname === `/api/v1/projects/${encodeURIComponent(item.projectId)}/run-catalog`);
  if (route.request().method() !== "GET" || !project) return false;
  expect([...url.searchParams.keys()].sort()).toEqual(["page", "page_size", "status"]);
  expect(url.searchParams.get("page_size")).toBe("20");
  expect(url.searchParams.get("page")).toMatch(/^[1-9]\d*$/);
  const status = url.searchParams.get("status");
  expect(["all", "completed", "failed", "cancelled", "queued", "running"]).toContain(status);
  const page = Number(url.searchParams.get("page"));
  const rows = project.runs.filter(run => status === "all" || status === run.status);
  const offset = (page - 1) * 20;
  await route.fulfill({ json: { project_id: project.projectId, page, page_size: 20, total: rows.length,
    has_more: offset + 20 < rows.length, items: rows.slice(offset, offset + 20).map(run => ({
      ...run, project_id: project.projectId, started_at: run.started_at ?? null, completed_at: run.completed_at ?? null,
      input_chars: run.input_chars ?? 0, prompt_tokens: run.prompt_tokens ?? 0, completion_tokens: run.completion_tokens ?? 0,
      estimated_cost_usd: run.estimated_cost_usd ?? null, frozen_document_count: Array.isArray(run.input_documents) ? run.input_documents.length : run.frozen_document_count ?? 0,
      retried_from: run.retried_from ?? null, batch_mode: run.batch_mode ?? null,
    })) } });
  return true;
}
