import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { ApiError, apiJson, SESSION_EXPIRED_EVENT } from "../api/client";
import {
  browserNavigate,
  handleBrowserPopState,
  initializeBrowserNavigation,
  productRouteFromPath,
  safeReturnTo,
  shouldResetProductScroll,
} from "../routing";
import AuthPage from "./AuthPage";
import AccountSettings from "./AccountSettings";
import ProjectCenter from "./ProjectCenter";
import PublicHome from "./PublicHome";
import GuestWorkspace from "./GuestWorkspace";
import { guestWorkspaceViewFromPath } from "./guestWorkspaceViews";
import { canApplySessionProbe, PUBLIC_SESSION_PROBE_TIMEOUT_MS, shouldRecoverExpiredSession } from "./publicEntry";
import { apiErrorDetail, type SessionIdentity } from "./session";
import { activateDraftSession, discardDraftSession, suspendDraftSession } from "../features/drafts/sessionDraftStorage";
import "../features/drafts/draft-notice.css";

const WorkspaceApp = lazy(() => import("../App"));
const ModelSettings = lazy(() => import("./ModelSettings"));

type StartupState =
  | { status: "checking"; identity: null }
  | { status: "ready"; identity: SessionIdentity }
  | { status: "signed-out"; identity: null }
  | { status: "failed"; identity: null; message: string };

function currentLocation(): string {
  return `${window.location.pathname}${window.location.search}${window.location.hash}`;
}

function loginPath(returnTo?: string): string {
  const safe = safeReturnTo(returnTo);
  return safe === "/app" ? "/login" : `/login?returnTo=${encodeURIComponent(safe)}`;
}

export default function RootApp() {
  const [locationKey, setLocationKey] = useState(currentLocation);
  const [startup, setStartup] = useState<StartupState>({ status: "checking", identity: null });
  const route = productRouteFromPath(window.location.pathname);
  const guestView = guestWorkspaceViewFromPath(window.location.pathname);
  const previousRouteKind = useRef(route.kind);
  const [draftLogoutWarning, setDraftLogoutWarning] = useState("");
  const [sessionExpiredNotice, setSessionExpiredNotice] = useState("");
  const identityRef = useRef<SessionIdentity | null>(null);
  const sessionProbe = useRef<{ generation: number; controller: AbortController | null; timer: number | null }>({ generation: 0, controller: null, timer: null });

  const invalidateSessionProbe = useCallback(() => {
    sessionProbe.current.generation += 1;
    sessionProbe.current.controller?.abort();
    if (sessionProbe.current.timer !== null) window.clearTimeout(sessionProbe.current.timer);
    sessionProbe.current.controller = null;
    sessionProbe.current.timer = null;
  }, []);

  const authenticated = useCallback((identity: SessionIdentity) => {
    invalidateSessionProbe();
    setSessionExpiredNotice("");
    identityRef.current = identity;
    if (identity.mode === "required") activateDraftSession({ userId: identity.user.id, workspaceId: identity.workspace.id });
    else suspendDraftSession();
    setStartup({ status: "ready", identity });
  }, [invalidateSessionProbe]);

  const loggedOut = useCallback(() => {
    invalidateSessionProbe();
    setSessionExpiredNotice("");
    const identity = identityRef.current;
    if (identity?.mode === "required") {
      const cleared = discardDraftSession({ userId: identity.user.id, workspaceId: identity.workspace.id });
      setDraftLogoutWarning(cleared ? "" : "已退出，但浏览器未能清理本账户的标签页暂存。请在共享设备上清除该站点数据；本页面不会再次恢复这份旧暂存。");
    } else suspendDraftSession();
    identityRef.current = null;
    setStartup({ status: "signed-out", identity: null });
  }, [invalidateSessionProbe]);

  const checkSession = useCallback(async () => {
    invalidateSessionProbe();
    const generation = sessionProbe.current.generation;
    const controller = new AbortController();
    sessionProbe.current.controller = controller;
    const current = () => canApplySessionProbe(generation, sessionProbe.current.generation, controller.signal.aborted);
    setStartup({ status: "checking", identity: null });
    const timeout = window.setTimeout(() => {
      if (!current()) return;
      controller.abort();
      setStartup({ status: "failed", identity: null, message: "确认登录状态超时，请重新确认或登录。" });
    }, PUBLIC_SESSION_PROBE_TIMEOUT_MS);
    sessionProbe.current.timer = timeout;
    try {
      const identity = await apiJson<SessionIdentity>("/api/v1/auth/me", { signal: controller.signal });
      if (!current()) return;
      authenticated(identity);
    } catch (error) {
      if (!current()) return;
      if (error instanceof ApiError && error.status === 401) {
        suspendDraftSession();
        identityRef.current = null;
        setStartup({ status: "signed-out", identity: null });
        return;
      }
      setStartup({ status: "failed", identity: null, message: apiErrorDetail(error) });
    } finally {
      window.clearTimeout(timeout);
      if (sessionProbe.current.generation === generation) {
        sessionProbe.current.controller = null;
        sessionProbe.current.timer = null;
      }
    }
  }, [authenticated, invalidateSessionProbe]);

  useEffect(() => {
    initializeBrowserNavigation();
    const handleLocation = (event: PopStateEvent) => {
      if (!handleBrowserPopState(event)) return;
      setLocationKey(currentLocation());
    };
    const handleExpired = (event: Event) => {
      const explicitLogout = (event as CustomEvent<{ explicitLogout?: boolean }>).detail?.explicitLogout === true;
      if (explicitLogout) {
        loggedOut();
        return;
      }
      const returnTo = shouldRecoverExpiredSession(identityRef.current?.mode, explicitLogout)
        ? safeReturnTo(currentLocation())
        : null;
      invalidateSessionProbe();
      suspendDraftSession();
      identityRef.current = null;
      setStartup({ status: "signed-out", identity: null });
      if (returnTo !== null) {
        setSessionExpiredNotice("登录状态已失效，请重新登录后继续。登录不会自动重试刚才的操作。");
        browserNavigate(loginPath(returnTo), { replace: true, bypassBlockers: true });
      }
    };
    window.addEventListener("popstate", handleLocation);
    window.addEventListener(SESSION_EXPIRED_EVENT, handleExpired);
    void checkSession();
    return () => {
      invalidateSessionProbe();
      window.removeEventListener("popstate", handleLocation);
      window.removeEventListener(SESSION_EXPIRED_EVENT, handleExpired);
    };
  }, [checkSession, invalidateSessionProbe, loggedOut]);

  useEffect(() => {
    if (startup.status === "ready") {
      if (["root", "login", "register"].includes(route.kind)) {
        browserNavigate("/app", { replace: true });
      }
      if (
        startup.identity.mode === "anonymous" &&
        route.kind === "settings-account"
      ) {
        browserNavigate("/app", { replace: true });
      }
      if (route.kind === "legacy-model-settings") {
        browserNavigate("/app/settings/model", { replace: true });
      }
      return;
    }
    if (
      startup.status === "signed-out" &&
      guestView === null &&
      route.kind !== "root" &&
      route.kind !== "login" &&
      route.kind !== "register"
    ) {
      browserNavigate(
        loginPath(currentLocation()),
        { replace: true, bypassBlockers: true },
      );
    }
  }, [locationKey, route.kind, guestView, startup.status]);

  useEffect(() => {
    void locationKey;
    document.body.classList.toggle("loreguard-workspace", route.kind === "workspace" && startup.status === "ready");
    return () => document.body.classList.remove("loreguard-workspace");
  }, [locationKey, route.kind, startup.status]);

  useEffect(() => {
    const previous = previousRouteKind.current;
    previousRouteKind.current = route.kind;
    if (shouldResetProductScroll(previous, route.kind)) {
      window.scrollTo({ top: 0, left: 0, behavior: "auto" });
    }
  }, [locationKey, route.kind]);

  if (route.kind === "root" && startup.status !== "ready") {
    return <PublicHome status={startup.status} onRetry={() => void checkSession()} draftLogoutWarning={draftLogoutWarning} />;
  }

  if (guestView !== null && startup.status !== "ready") {
    return <GuestWorkspace view={guestView} status={startup.status} onRetry={() => void checkSession()} draftLogoutWarning={draftLogoutWarning} />;
  }

  if (startup.status === "checking") {
    return (
      <main className="startupPage productPage" aria-busy="true" aria-label="正在确认登录状态">
        <div className="startupMark"><span className="productSeal" aria-hidden="true"><i>LG</i></span><b>LoreGuard</b></div>
        <div className="startupLine"><i /></div>
        <p>正在打开故事工作区…</p>
      </main>
    );
  }

  if (route.kind === "login" || route.kind === "register") {
    if (startup.status === "ready") {
      return <main className="startupPage productPage" aria-busy="true"><p>正在返回项目中心…</p></main>;
    }
    return <>
      {draftLogoutWarning && <p className="logoutDraftWarning" role="alert">{draftLogoutWarning}</p>}
      {sessionExpiredNotice && <p className="logoutDraftWarning" role="status">{sessionExpiredNotice}</p>}
      <AuthPage mode={route.kind} onAuthenticated={authenticated} />
    </>;
  }

  if (startup.status === "failed") {
    return (
      <main className="startupPage productPage">
        <div className="startupMark"><span className="productSeal" aria-hidden="true"><i>LG</i></span><b>LoreGuard</b></div>
        <h1>暂时无法连接服务</h1>
        <p>{startup.message} 页面没有读取或修改你的项目。</p>
        <button className="quietPrimary" type="button" onClick={() => void checkSession()}>重新连接</button>
      </main>
    );
  }

  if (startup.status !== "ready") {
    return <main className="startupPage productPage" aria-busy="true"><p>正在前往登录页面…</p></main>;
  }

  if (route.kind === "projects" || route.kind === "root") {
    return <ProjectCenter identity={startup.identity} onLoggedOut={loggedOut} />;
  }

  if (route.kind === "settings-account") {
    if (startup.identity.mode !== "required") {
      return <main className="startupPage productPage" aria-busy="true"><p>正在返回项目中心…</p></main>;
    }
    return <AccountSettings identity={startup.identity} onLoggedOut={loggedOut} />;
  }

  if (route.kind === "settings-model") {
    return (
      <Suspense fallback={<main className="startupPage productPage" aria-busy="true"><p>正在打开模型与密钥设置…</p></main>}>
        <ModelSettings
          identity={startup.identity}
          onLoggedOut={loggedOut}
        />
      </Suspense>
    );
  }

  if (route.kind === "legacy-model-settings") {
    return <main className="startupPage productPage" aria-busy="true"><p>正在前往模型与密钥设置…</p></main>;
  }

  if (route.kind === "workspace") {
    return (
      <Suspense fallback={<main className="startupPage productPage" aria-busy="true"><p>正在打开项目工作台…</p></main>}>
        <WorkspaceApp
          key={`${startup.identity.user.id}:${startup.identity.workspace.id}`}
          identity={startup.identity}
          onLoggedOut={loggedOut}
        />
      </Suspense>
    );
  }

  return (
    <main className="notFoundPage productPage">
      <span className="productSeal" aria-hidden="true"><i>LG</i></span>
      <h1>没有找到这个页面</h1>
      <p>链接可能已经失效，或者你没有访问权限。</p>
      <button className="quietPrimary" type="button" onClick={() => browserNavigate("/app", { replace: true })}>返回项目中心</button>
    </main>
  );
}
