import { apiJson } from "../../api/client.ts";
import type { BaselineRunSummary } from "./guidedReview.ts";

export type CharacterBaselineStatus = {
  total_characters: number;
  confirmed_trait_count: number;
  pending_candidate_count: number;
  baseline_run_id: string | null;
  baseline_run_status: "completed" | null;
  model_coverage: "full" | "partial" | "rules_only" | "unknown";
  coverage_detail: string | null;
  readiness: "ready" | "no_documents" | "no_completed_run" | "not_generated";
};

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function count(value: unknown): value is number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

export function normalizeCharacterBaselineStatus(value: unknown): CharacterBaselineStatus {
  const source = record(value);
  if (
    !source ||
    !count(source.total_characters) ||
    !count(source.confirmed_trait_count) ||
    !count(source.pending_candidate_count) ||
    !(source.baseline_run_id === null || (typeof source.baseline_run_id === "string" && source.baseline_run_id.trim())) ||
    !(source.baseline_run_status === null || source.baseline_run_status === "completed") ||
    !["full", "partial", "rules_only", "unknown"].includes(String(source.model_coverage)) ||
    !(source.coverage_detail === null || typeof source.coverage_detail === "string") ||
    !["ready", "no_documents", "no_completed_run", "not_generated"].includes(String(source.readiness)) ||
    (source.baseline_run_id === null) !== (source.baseline_run_status === null)
  ) {
    throw new TypeError("角色基线汇总格式不完整，请重试读取；未知状态不会解锁新稿审查。");
  }
  return source as CharacterBaselineStatus;
}

export async function fetchCharacterBaselineStatus(
  projectId: string,
  baselineRunId: string,
  signal?: AbortSignal,
): Promise<CharacterBaselineStatus> {
  return normalizeCharacterBaselineStatus(
    await apiJson<unknown>(
      `/api/v1/projects/${encodeURIComponent(projectId)}/character-baseline-status?baseline_run_id=${encodeURIComponent(baselineRunId)}`,
      { signal },
    ),
  );
}

export function hasReadyCharacterBaseline(
  run: BaselineRunSummary | null,
  status: CharacterBaselineStatus | null,
): boolean {
  return Boolean(
    run &&
    status &&
    status.baseline_run_id === run.id &&
    status.baseline_run_status === "completed" &&
    status.readiness === "ready" &&
    status.model_coverage === "full" &&
    status.confirmed_trait_count > 0,
  );
}
