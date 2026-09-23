import { useCallback, useEffect, useState } from "react";

const HISTORY_INDEX_KEY = "__loreguardHistoryIndex";

type BrowserNavigationBlocker = () => boolean;

const browserNavigationBlockers = new Set<BrowserNavigationBlocker>();
let currentHistoryIndex: number | null = null;
let restoringHistoryIndex: number | null = null;
let dispatchingProgrammaticNavigation = false;

function historyIndex(value: unknown): number | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const candidate = (value as Record<string, unknown>)[HISTORY_INDEX_KEY];
  return typeof candidate === "number" && Number.isSafeInteger(candidate) && candidate >= 0
    ? candidate
    : null;
}

function historyState(value: unknown, index: number): Record<string, unknown> {
  const base = value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
  return { ...base, [HISTORY_INDEX_KEY]: index };
}

function navigationAllowed(): boolean {
  for (const blocker of browserNavigationBlockers) {
    try {
      if (!blocker()) return false;
    } catch {
      return false;
    }
  }
  return true;
}

/**
 * Mark the current same-document history entry so a cancelled back/forward
 * traversal can return to it without replacing the URL or remounting the page.
 */
export function initializeBrowserNavigation(): number {
  const existing = historyIndex(window.history.state);
  if (existing !== null) {
    currentHistoryIndex = existing;
    return existing;
  }
  const initial = currentHistoryIndex ?? 0;
  window.history.replaceState(historyState(window.history.state, initial), "");
  currentHistoryIndex = initial;
  return initial;
}

/** Register a synchronous confirmation boundary for native browser traversal. */
export function registerBrowserNavigationBlocker(
  blocker: BrowserNavigationBlocker,
): () => void {
  browserNavigationBlockers.add(blocker);
  return () => browserNavigationBlockers.delete(blocker);
}

/**
 * Return true when RootApp may render the location selected by a popstate.
 * A rejected native traversal is reversed with history.go; its compensating
 * popstate is recognized by index and never asks the user a second time.
 */
export function handleBrowserPopState(event: PopStateEvent): boolean {
  const targetIndex = historyIndex(event.state ?? window.history.state);
  if (dispatchingProgrammaticNavigation) {
    if (targetIndex !== null) currentHistoryIndex = targetIndex;
    return true;
  }
  if (
    restoringHistoryIndex !== null &&
    targetIndex === restoringHistoryIndex
  ) {
    currentHistoryIndex = targetIndex;
    restoringHistoryIndex = null;
    return true;
  }
  if (restoringHistoryIndex !== null && targetIndex !== null) {
    window.history.go(restoringHistoryIndex - targetIndex);
    return false;
  }

  const activeIndex = currentHistoryIndex ?? initializeBrowserNavigation();
  if (targetIndex === null || targetIndex === activeIndex) return true;
  if (browserNavigationBlockers.size === 0 || navigationAllowed()) {
    currentHistoryIndex = targetIndex;
    return true;
  }

  restoringHistoryIndex = activeIndex;
  window.history.go(activeIndex - targetIndex);
  return false;
}

export const workspaceViews = [
  "check",
  "projects",
  "characters",
  "diff",
  "visual",
  "audit",
  "report",
  "revision",
  "provider",
] as const;

export type WorkspaceView = (typeof workspaceViews)[number];

const workspaceViewSet = new Set<string>(workspaceViews);

const nestedViewSegments: Record<string, WorkspaceView> = {
  check: "check",
  documents: "projects",
  characters: "characters",
  compare: "diff",
  visuals: "visual",
  runs: "audit",
  report: "report",
  revise: "revision",
};

export type ProductRoute =
  | { kind: "login" }
  | { kind: "register" }
  | { kind: "projects" }
  | { kind: "settings-account" }
  | { kind: "settings-model" }
  | { kind: "legacy-model-settings" }
  | { kind: "workspace"; projectId: string | null; runId: string | null }
  | { kind: "root" }
  | { kind: "not-found" };

export function shouldResetProductScroll(
  previous: ProductRoute["kind"],
  next: ProductRoute["kind"],
): boolean {
  return (
    (previous === "login" || previous === "register") &&
    (next === "projects" || next === "settings-account" || next === "settings-model" || next === "workspace")
  );
}

export function productRouteFromPath(pathname: string): ProductRoute {
  const normalized = pathname.replace(/\/+$/, "") || "/";
  if (normalized === "/") return { kind: "root" };
  if (normalized === "/login") return { kind: "login" };
  if (normalized === "/register") return { kind: "register" };
  if (normalized === "/app") return { kind: "projects" };
  if (normalized === "/provider") return { kind: "legacy-model-settings" };
  if (normalized === "/revision") {
    return { kind: "workspace", projectId: null, runId: null };
  }
  if (normalized === "/app/settings/account") return { kind: "settings-account" };
  if (normalized === "/app/settings/model") return { kind: "settings-model" };
  if (
    /^\/app\/projects\/[^/]+\/(?:check|documents|characters|compare|visuals|runs|report|revise)$/.test(
      normalized,
    ) ||
    /^\/app\/projects\/[^/]+\/runs\/[^/]+(?:\/(?:report|visuals|revise))?$/.test(
      normalized,
    ) ||
    /^\/(?:check|projects|characters|diff|visual|audit|report|provider)$/.test(normalized)
  ) {
    return {
      kind: "workspace",
      projectId: workspaceProjectIdFromPath(normalized),
      runId: workspaceRunIdFromPath(normalized),
    };
  }
  return { kind: "not-found" };
}

export function workspaceProjectIdFromPath(pathname: string): string | null {
  const match = pathname.match(/^\/app\/projects\/([^/]+)(?:\/|$)/);
  return match ? safeDecodePathSegment(match[1]) : null;
}

function safeDecodePathSegment(value: string): string | null {
  try {
    return decodeURIComponent(value);
  } catch {
    return null;
  }
}

export function workspaceRunIdFromPath(pathname: string): string | null {
  const match = pathname.match(/^\/app\/projects\/[^/]+\/runs\/([^/]+)(?:\/|$)/);
  return match ? safeDecodePathSegment(match[1]) : null;
}

export function workspaceViewFromPath(pathname: string): WorkspaceView {
  if (pathname.replace(/\/+$/, "") === "/revision") return "check";
  if (pathname === "/app/settings/model") return "provider";
  const preciseRun = pathname.match(
    /^\/app\/projects\/[^/]+\/runs\/[^/]+(?:\/(report|visuals|revise))?\/?$/,
  );
  if (preciseRun) {
    if (preciseRun[1] === "report") return "report";
    if (preciseRun[1] === "visuals") return "visual";
    if (preciseRun[1] === "revise") return "revision";
    return "audit";
  }
  const nested = pathname.match(
    /^\/app\/projects\/[^/]+\/(check|documents|characters|compare|visuals|runs|report|revise)(?:\/|$)/,
  );
  if (nested) return nestedViewSegments[nested[1]] || "check";
  const segment = pathname.split("/").filter(Boolean)[0] || "check";
  return workspaceViewSet.has(segment) ? (segment as WorkspaceView) : "check";
}

export type WorkspaceRouteSnapshot = {
  view: WorkspaceView;
  projectId: string | null;
  runId: string | null;
  search: string;
};

export const issueStatusFilters = [
  "all",
  "unreviewed",
  "accepted",
  "false_positive",
  "resolved",
] as const;

export const issueCategoryFilters = [
  "all",
  "fact_conflict",
  "location_collision",
  "knowledge_without_acquisition",
  "item_ownership",
  "world_rule_conflict",
  "character_drift",
] as const;

export type IssueStatusFilter = (typeof issueStatusFilters)[number];

export type ReportRouteState = {
  category: string;
  status: IssueStatusFilter;
  issueId: string | null;
};

export function reportRouteStateFromSearch(search: string): ReportRouteState {
  const params = new URLSearchParams(search);
  const rawCategory = params.get("category") || "all";
  const category = issueCategoryFilters.includes(
    rawCategory as (typeof issueCategoryFilters)[number],
  )
    ? rawCategory
    : "all";
  const rawStatus = params.get("status") || "all";
  const status = issueStatusFilters.includes(rawStatus as IssueStatusFilter)
    ? (rawStatus as IssueStatusFilter)
    : "all";
  return {
    category,
    status,
    issueId: /^[A-Za-z0-9_-]{1,128}$/.test(params.get("issue") || "")
      ? params.get("issue")
      : null,
  };
}

export function reportSearch(state: ReportRouteState): string {
  const params = new URLSearchParams({
    category: state.category || "all",
    status: state.status,
  });
  if (state.issueId) params.set("issue", state.issueId);
  return `?${params.toString()}`;
}

export function workspaceRouteFromPath(
  pathname: string,
  search = "",
): WorkspaceRouteSnapshot {
  return {
    view: workspaceViewFromPath(pathname),
    projectId: workspaceProjectIdFromPath(pathname),
    runId: workspaceRunIdFromPath(pathname),
    search,
  };
}

export function workspacePath(
  view: WorkspaceView,
  projectId?: string | null,
  runId?: string | null,
): string {
  if (view === "provider") return "/app/settings/model";
  if (view === "revision" && !projectId) return "/check";
  if (!projectId) return `/${view}`;
  const encodedId = encodeURIComponent(projectId);
  if (runId && ["audit", "report", "visual", "revision"].includes(view)) {
    const encodedRunId = encodeURIComponent(runId);
    if (view === "report") return `/app/projects/${encodedId}/runs/${encodedRunId}/report`;
    if (view === "visual") return `/app/projects/${encodedId}/runs/${encodedRunId}/visuals`;
    if (view === "revision") return `/app/projects/${encodedId}/runs/${encodedRunId}/revise`;
    return `/app/projects/${encodedId}/runs/${encodedRunId}`;
  }
  const segment: Record<Exclude<WorkspaceView, "provider">, string> = {
    check: "check",
    projects: "documents",
    characters: "characters",
    diff: "compare",
    visual: "visuals",
    audit: "runs",
    report: "report",
    revision: "revise",
  };
  return `/app/projects/${encodedId}/${segment[view as Exclude<WorkspaceView, "provider">]}`;
}

export function safeReturnTo(value: string | null | undefined): string {
  if (!value || !value.startsWith("/") || value.startsWith("//")) return "/app";
  try {
    const url = new URL(value, "https://loreguard.local");
    if (url.origin !== "https://loreguard.local") return "/app";
    return productRouteFromPath(url.pathname).kind === "not-found"
      ? "/app"
      : `${url.pathname}${url.search}${url.hash}`;
  } catch {
    return "/app";
  }
}

export function browserNavigate(path: string, options?: { replace?: boolean }): void {
  const activeIndex = currentHistoryIndex ?? initializeBrowserNavigation();
  const method = options?.replace ? "replaceState" : "pushState";
  const nextIndex = options?.replace ? activeIndex : activeIndex + 1;
  restoringHistoryIndex = null;
  window.history[method](historyState(window.history.state, nextIndex), "", path);
  currentHistoryIndex = nextIndex;
  dispatchingProgrammaticNavigation = true;
  try {
    window.dispatchEvent(
      new PopStateEvent("popstate", { state: window.history.state }),
    );
  } finally {
    dispatchingProgrammaticNavigation = false;
  }
}

export function useWorkspaceRoute(): [
  WorkspaceView,
  (view: WorkspaceView, options?: { replace?: boolean }) => void,
  string | null,
  string | null,
  string,
] {
  const [route, setRoute] = useState<WorkspaceRouteSnapshot>(() =>
    workspaceRouteFromPath(window.location.pathname, window.location.search),
  );

  useEffect(() => {
    const handlePopState = () => {
      setRoute(
        workspaceRouteFromPath(window.location.pathname, window.location.search),
      );
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const navigate = useCallback(
    (nextView: WorkspaceView, options?: { replace?: boolean }) => {
      const projectId = workspaceProjectIdFromPath(window.location.pathname);
      const runId = workspaceRunIdFromPath(window.location.pathname);
      const basePath = workspacePath(nextView, projectId, runId);
      const nextPath =
        nextView === "report"
          ? `${basePath}${workspaceViewFromPath(window.location.pathname) === "report" && window.location.search ? window.location.search : reportSearch({ category: "all", status: "all", issueId: null })}`
          : basePath;
      if (`${window.location.pathname}${window.location.search}` !== nextPath) {
        browserNavigate(nextPath, options);
      } else {
        setRoute(
          workspaceRouteFromPath(
            window.location.pathname,
            window.location.search,
          ),
        );
      }
    },
    [],
  );

  return [route.view, navigate, route.projectId, route.runId, route.search];
}
