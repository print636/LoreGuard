export const runStatuses = ["all", "queued", "running", "completed", "failed", "cancelled"] as const;
export type RunStatusFilter = typeof runStatuses[number];
export type RunCatalogItem = {
  id: string;
  project_id: string;
  status: Exclude<RunStatusFilter, "all"> | "unknown";
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  input_chars: number;
  prompt_tokens: number;
  completion_tokens: number;
  estimated_cost_usd: number | null;
  frozen_document_count: number;
  retried_from: string | null;
  batch_mode: string | null;
};
export type RunCatalog = {
  project_id: string;
  page: number;
  page_size: number;
  total: number;
  has_more: boolean;
  items: RunCatalogItem[];
};
export const RUN_HISTORY_PAGE_SIZE = 20;
export const runStatusNames: Record<RunStatusFilter, string> = {
  all: "全部状态", queued: "排队中", running: "进行中", completed: "已完成", failed: "失败", cancelled: "已取消",
};
export const runItemStatusNames = { ...runStatusNames, unknown: "状态未知" };

export function runHistoryStateFromSearch(search: string): { page: number; status: RunStatusFilter } {
  const params = new URLSearchParams(search);
  const raw = params.get("history_page") || "1";
  const page = /^\d+$/.test(raw) ? Number(raw) : 1;
  const status = params.get("history_status");
  return { page: Number.isSafeInteger(page) && page >= 1 && page <= 100000 ? page : 1,
    status: runStatuses.includes(status as RunStatusFilter) ? status as RunStatusFilter : "all" };
}

export function runHistorySearch(search: string, next: { page: number; status: RunStatusFilter }): string {
  const params = new URLSearchParams(search);
  if (next.page === 1) params.delete("history_page");
  else params.set("history_page", String(next.page));
  if (next.status === "all") params.delete("history_status");
  else params.set("history_status", next.status);
  const query = params.toString();
  return query ? `?${query}` : "";
}

export function runCatalogPath(projectId: string, state: { page: number; status: RunStatusFilter }): string {
  const params = new URLSearchParams({ page: String(state.page), page_size: String(RUN_HISTORY_PAGE_SIZE), status: state.status });
  return `/api/v1/projects/${encodeURIComponent(projectId)}/run-catalog?${params}`;
}

export function catalogMatchesRequest(value: RunCatalog, projectId: string, page: number): boolean {
  return value?.project_id === projectId && value.page === page && value.page_size === RUN_HISTORY_PAGE_SIZE &&
    Number.isSafeInteger(value.total) && value.total >= 0 && typeof value.has_more === "boolean" &&
    Array.isArray(value.items) && value.items.length <= RUN_HISTORY_PAGE_SIZE && value.items.every((item) =>
      item?.project_id === projectId && typeof item.id === "string" && item.id.length > 0 &&
      [...runStatuses, "unknown"].includes(item.status) && item.status !== ("all" as string) &&
      typeof item.created_at === "string" && Number.isSafeInteger(item.input_chars) && item.input_chars >= 0 &&
      Number.isSafeInteger(item.frozen_document_count) && item.frozen_document_count >= 0 &&
      Number.isSafeInteger(item.prompt_tokens) && item.prompt_tokens >= 0 &&
      Number.isSafeInteger(item.completion_tokens) && item.completion_tokens >= 0,
    );
}

export function runCatalogAction(status: RunCatalogItem["status"]): string {
  return status === "completed" ? "查看报告" : status === "queued" || status === "running" ? "查看进度" : "查看详情";
}
