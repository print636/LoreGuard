import type { RunInputSnapshot } from "./runSnapshot";

export const revisionSteps = ["review", "upload", "run", "compare"] as const;
export const comparisonOutcomes = [
  "all",
  "no_longer_detected",
  "persisting",
  "new",
  "unverifiable",
] as const;

export type RevisionStep = (typeof revisionSteps)[number];
export type ComparisonOutcome = (typeof comparisonOutcomes)[number];

export type RevisionRouteState = {
  step: RevisionStep;
  issueId: string | null;
  documentId: string | null;
  recheckRunId: string | null;
  outcome: ComparisonOutcome;
  generateGraph: boolean;
  generateTimeline: boolean;
  page: number;
};

export type RevisionDocument = {
  id: string;
  name: string;
  version: number;
  active: boolean;
  document_role: string;
  story_scope: string;
};

export type RevisionDelta = {
  changed: boolean;
  added: RevisionDocument[];
  removed: RunInputSnapshot[];
  updated: Array<{ before: RunInputSnapshot; after: RevisionDocument }>;
};

export type RevisionComparisonIdentity = {
  project_id?: unknown;
  baseline_run_id?: unknown;
  target_run_id?: unknown;
};

export type RevisionUploadValidation =
  | { ok: true; replacement: RevisionDocument | null }
  | { ok: false; message: string; field: "document" | "file" | "mode" };

const safeIdentifier = /^[A-Za-z0-9_-]{1,128}$/;

function identifier(value: string | null): string | null {
  return value && safeIdentifier.test(value) ? value : null;
}

export function revisionRouteStateFromSearch(search: string): RevisionRouteState {
  const params = new URLSearchParams(search);
  const rawStep = params.get("step") || "review";
  const rawOutcome = params.get("outcome") || "all";
  return {
    step: revisionSteps.includes(rawStep as RevisionStep)
      ? (rawStep as RevisionStep)
      : "review",
    issueId: identifier(params.get("issue")),
    documentId: identifier(params.get("document")),
    recheckRunId: identifier(params.get("recheck")),
    outcome: comparisonOutcomes.includes(rawOutcome as ComparisonOutcome)
      ? (rawOutcome as ComparisonOutcome)
      : "all",
    generateGraph: params.get("graph") === "1",
    generateTimeline: params.get("timeline") === "1",
    page: /^\d{1,4}$/.test(params.get("page") || "")
      ? Math.max(1, Number(params.get("page")))
      : 1,
  };
}

export function revisionSearch(state: RevisionRouteState): string {
  const params = new URLSearchParams({
    step: state.step,
    outcome: state.outcome,
  });
  if (state.issueId) params.set("issue", state.issueId);
  if (state.documentId) params.set("document", state.documentId);
  if (state.recheckRunId) params.set("recheck", state.recheckRunId);
  if (state.generateGraph) params.set("graph", "1");
  if (state.generateTimeline) params.set("timeline", "1");
  if (state.page > 1) params.set("page", String(state.page));
  return `?${params.toString()}`;
}

export function mergeRevisionRouteState(
  current: RevisionRouteState,
  next: Partial<RevisionRouteState>,
): RevisionRouteState {
  const merged = { ...current, ...next };
  return next.outcome !== undefined && next.outcome !== current.outcome
    ? { ...merged, page: 1 }
    : merged;
}

export function comparisonBelongsToRevision(
  comparison: RevisionComparisonIdentity,
  expected: { projectId: string; baselineRunId: string; targetRunId: string },
): boolean {
  return (
    comparison.project_id === expected.projectId &&
    comparison.baseline_run_id === expected.baselineRunId &&
    comparison.target_run_id === expected.targetRunId
  );
}

export function createMutationGuard(): { tryBegin: () => boolean; end: () => void } {
  let active = false;
  return {
    tryBegin() {
      if (active) return false;
      active = true;
      return true;
    },
    end() {
      active = false;
    },
  };
}

export function comparisonRetryDelay(failureCount: number): number {
  const safeFailures = Number.isFinite(failureCount)
    ? Math.max(1, Math.floor(failureCount))
    : 1;
  return Math.min(15_000, 2_500 * 2 ** (safeFailures - 1));
}

export function validateRevisionUploadSelection(
  mode: "replace" | "related",
  fileName: string,
  documentId: string | null,
  documents: RevisionDocument[],
): RevisionUploadValidation {
  const activeDocuments = documents.filter((document) => document.active);
  const normalizedName = fileName.toLocaleLowerCase();
  if (mode === "replace") {
    if (!documentId) {
      return {
        ok: false,
        message: "请选择这个文件替换的当前文档，或改为新增相关文档。",
        field: "document",
      };
    }
    const replacement = activeDocuments.find((document) => document.id === documentId);
    if (!replacement) {
      return {
        ok: false,
        message: "替换目标已经不是当前生效文档，请重新选择后再上传。",
        field: "document",
      };
    }
    if (normalizedName !== replacement.name.toLocaleLowerCase()) {
      return {
        ok: false,
        message: `文件名必须与替换目标“${replacement.name}”一致。改名会成为新文档，不能自动归入原版本链。`,
        field: "file",
      };
    }
    return { ok: true, replacement };
  }

  const sameName = activeDocuments.find(
    (document) => document.name.toLocaleLowerCase() === normalizedName,
  );
  if (sameName) {
    return {
      ok: false,
      message: `项目中已有同名生效文档“${sameName.name}”。请选择“替换现有文档”并指定它，避免断开版本链。`,
      field: "mode",
    };
  }
  return { ok: true, replacement: null };
}

export function pendingVisualizationKinds(
  runId: string,
  generateGraph: boolean,
  generateTimeline: boolean,
  generatedKeys: ReadonlySet<string>,
): Array<"graph" | "timeline"> {
  return ([
    ...(generateGraph ? (["graph"] as const) : []),
    ...(generateTimeline ? (["timeline"] as const) : []),
  ]).filter((kind) => !generatedKeys.has(`${runId}:${kind}`));
}

export function revisionDelta(
  baseline: RunInputSnapshot[],
  documents: RevisionDocument[],
): RevisionDelta {
  const current = documents.filter((document) => document.active);
  const baselineByName = new Map(
    baseline.map((document) => [document.document_name.toLocaleLowerCase(), document]),
  );
  const currentByName = new Map(
    current.map((document) => [document.name.toLocaleLowerCase(), document]),
  );
  const added = current.filter(
    (document) => !baselineByName.has(document.name.toLocaleLowerCase()),
  );
  const removed = baseline.filter(
    (document) => !currentByName.has(document.document_name.toLocaleLowerCase()),
  );
  const updated = baseline.flatMap((before) => {
    const after = currentByName.get(before.document_name.toLocaleLowerCase());
    if (!after) return [];
    return after.id !== before.document_id || after.version !== before.document_version
      ? [{ before, after }]
      : [];
  });
  return {
    changed: added.length > 0 || removed.length > 0 || updated.length > 0,
    added,
    removed,
    updated,
  };
}

export function revisionDocumentForIssue(
  documentName: string | null | undefined,
  documents: RevisionDocument[],
): string | null {
  if (!documentName) return null;
  return (
    documents.find(
      (document) =>
        document.active &&
        document.name.toLocaleLowerCase() === documentName.toLocaleLowerCase(),
    )?.id || null
  );
}

export function comparisonPath(
  runId: string,
  limit = 20,
  offset = 0,
  outcome: ComparisonOutcome = "all",
): string {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  if (outcome !== "all") params.set("outcome", outcome);
  return `/api/v1/analysis-runs/${encodeURIComponent(runId)}/comparison?${params.toString()}`;
}

export function recheckPath(baselineRunId: string): string {
  return `/api/v1/analysis-runs/${encodeURIComponent(baselineRunId)}/rechecks`;
}
