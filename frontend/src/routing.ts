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
  | { kind: "workspace" }
  | { kind: "root" }
  | { kind: "not-found" };

export function shouldResetProductScroll(
  previous: ProductRoute["kind"],
  next: ProductRoute["kind"],
): boolean {
  return (
    (previous === "login" || previous === "register") &&
    (next === "projects" || next === "workspace")
  );
}

export function productRouteFromPath(pathname: string): ProductRoute {
  const normalized = pathname.replace(/\/+$/, "") || "/";
  if (normalized === "/") return { kind: "root" };
  if (normalized === "/login") return { kind: "login" };
  if (normalized === "/register") return { kind: "register" };
  if (normalized === "/app") return { kind: "projects" };
  if (
    normalized === "/app/settings/model" ||
    /^\/app\/projects\/[^/]+\/(?:check|documents|compare|visuals|runs|report)(?:\/.*)?$/.test(
      normalized,
    ) ||
    /^\/(?:check|projects|diff|visual|audit|report|provider)$/.test(normalized)
  ) {
    return { kind: "workspace" };
  }
  return { kind: "not-found" };
}

export function workspaceProjectIdFromPath(pathname: string): string | null {
  const match = pathname.match(/^\/app\/projects\/([^/]+)(?:\/|$)/);
  return match ? decodeURIComponent(match[1]) : null;
}

export function workspaceViewFromPath(pathname: string): WorkspaceView {
  if (pathname === "/app/settings/model") return "provider";
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
};

export function workspaceRouteFromPath(pathname: string): WorkspaceRouteSnapshot {
  return {
    view: workspaceViewFromPath(pathname),
    projectId: workspaceProjectIdFromPath(pathname),
  };
}

export function workspacePath(view: WorkspaceView, projectId?: string | null): string {
  if (view === "provider" && projectId) return "/app/settings/model";
  if (!projectId) return `/${view}`;
  const encodedId = encodeURIComponent(projectId);
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
] {
  const [route, setRoute] = useState<WorkspaceRouteSnapshot>(() =>
    workspaceRouteFromPath(window.location.pathname),
  );

  useEffect(() => {
    const handlePopState = () => {
      setRoute(workspaceRouteFromPath(window.location.pathname));
    };
    window.addEventListener("popstate", handlePopState);
    return () => window.removeEventListener("popstate", handlePopState);
  }, []);

  const navigate = useCallback(
    (nextView: WorkspaceView, options?: { replace?: boolean }) => {
      const projectId = workspaceProjectIdFromPath(window.location.pathname);
      const nextPath = workspacePath(nextView, projectId);
      if (window.location.pathname !== nextPath) {
        browserNavigate(nextPath, options);
      } else {
        setRoute(workspaceRouteFromPath(nextPath));
      }
    },
    [],
  );

  return [route.view, navigate, route.projectId];
}
