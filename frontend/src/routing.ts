import { useCallback, useEffect, useState } from "react";

export const workspaceViews = [
  "check",
  "projects",
  "diff",
  "visual",
  "audit",
  "report",
  "provider",
] as const;

export type WorkspaceView = (typeof workspaceViews)[number];

const workspaceViewSet = new Set<string>(workspaceViews);

const nestedViewSegments: Record<string, WorkspaceView> = {
  check: "check",
  documents: "projects",
  compare: "diff",
  visuals: "visual",
  runs: "audit",
  report: "report",
};

export type ProductRoute =
  | { kind: "login" }
  | { kind: "register" }
  | { kind: "projects" }
  | { kind: "settings-account" }
  | { kind: "workspace"; projectId: string | null; runId: string | null }
  | { kind: "root" }
  | { kind: "not-found" };

export function shouldResetProductScroll(
  previous: ProductRoute["kind"],
  next: ProductRoute["kind"],
): boolean {
  return (
    (previous === "login" || previous === "register") &&
    (next === "projects" || next === "settings-account" || next === "workspace")
  );
}

export function productRouteFromPath(pathname: string): ProductRoute {
  const normalized = pathname.replace(/\/+$/, "") || "/";
  if (normalized === "/") return { kind: "root" };
  if (normalized === "/login") return { kind: "login" };
  if (normalized === "/register") return { kind: "register" };
  if (normalized === "/app") return { kind: "projects" };
  if (normalized === "/app/settings/account") return { kind: "settings-account" };
  if (normalized === "/app/settings/model") {
    return { kind: "workspace", projectId: null, runId: null };
  }
  if (
    /^\/app\/projects\/[^/]+\/(?:check|documents|compare|visuals|runs|report)$/.test(
      normalized,
    ) ||
    /^\/app\/projects\/[^/]+\/runs\/[^/]+(?:\/(?:report|visuals))?$/.test(
      normalized,
    ) ||
    /^\/(?:check|projects|diff|visual|audit|report|provider)$/.test(normalized)
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
  if (pathname === "/app/settings/model") return "provider";
  const preciseRun = pathname.match(
    /^\/app\/projects\/[^/]+\/runs\/[^/]+(?:\/(report|visuals))?\/?$/,
  );
  if (preciseRun) {
    if (preciseRun[1] === "report") return "report";
    if (preciseRun[1] === "visuals") return "visual";
    return "audit";
  }
  const nested = pathname.match(
    /^\/app\/projects\/[^/]+\/(check|documents|compare|visuals|runs|report)(?:\/|$)/,
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
  if (view === "provider" && projectId) return "/app/settings/model";
  if (!projectId) return `/${view}`;
  const encodedId = encodeURIComponent(projectId);
  if (runId && ["audit", "report", "visual"].includes(view)) {
    const encodedRunId = encodeURIComponent(runId);
    if (view === "report") return `/app/projects/${encodedId}/runs/${encodedRunId}/report`;
    if (view === "visual") return `/app/projects/${encodedId}/runs/${encodedRunId}/visuals`;
    return `/app/projects/${encodedId}/runs/${encodedRunId}`;
  }
  const segment: Record<Exclude<WorkspaceView, "provider">, string> = {
    check: "check",
    projects: "documents",
    diff: "compare",
    visual: "visuals",
    audit: "runs",
    report: "report",
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
  const method = options?.replace ? "replaceState" : "pushState";
  window.history[method](null, "", path);
  window.dispatchEvent(new PopStateEvent("popstate"));
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
