import { reportSearch, workspacePath } from "./routing.ts";
import { ApiError } from "./api/client.ts";

export type RunIdentity = {
  id: string;
  project_id: string;
  status: string;
};

export type RunSelection<T extends RunIdentity> =
  | { kind: "selected"; run: T; precise: boolean }
  | { kind: "empty" }
  | { kind: "not-found" };

export function resolveRunSelection<T extends RunIdentity>(
  history: T[],
  projectId: string,
  requestedRunId: string | null,
): RunSelection<T> {
  if (!requestedRunId) {
    return history[0]
      ? { kind: "selected", run: history[0], precise: false }
      : { kind: "empty" };
  }
  const exact = history.find((row) => row.id === requestedRunId);
  return exact?.project_id === projectId
    ? { kind: "selected", run: exact, precise: true }
    : { kind: "not-found" };
}

export function terminalRunPath(
  status: string,
  projectId: string,
  runId: string,
): string {
  if (status === "completed") {
    return `${workspacePath("report", projectId, runId)}${reportSearch({
      category: "all",
      status: "all",
      issueId: null,
    })}`;
  }
  return workspacePath("audit", projectId, runId);
}

export type DispatchFailureRecovery = {
  runId: string;
  message: string;
};

export function dispatchFailureRecovery(
  error: unknown,
): DispatchFailureRecovery | null {
  if (!(error instanceof ApiError) || error.status !== 503) return null;
  const envelope = error.detail;
  if (!envelope || typeof envelope !== "object" || !("detail" in envelope)) {
    return null;
  }
  const detail = (envelope as { detail?: unknown }).detail;
  if (!detail || typeof detail !== "object") return null;
  const value = detail as {
    code?: unknown;
    message?: unknown;
    run_id?: unknown;
    retryable?: unknown;
  };
  if (
    value.code !== "analysis_dispatch_failed" ||
    value.retryable !== true ||
    typeof value.run_id !== "string" ||
    !/^[A-Za-z0-9-]{1,128}$/.test(value.run_id)
  ) {
    return null;
  }
  return {
    runId: value.run_id,
    message:
      typeof value.message === "string" && value.message.trim()
        ? value.message
        : "任务未进入执行队列，已保留冻结输入。",
  };
}
