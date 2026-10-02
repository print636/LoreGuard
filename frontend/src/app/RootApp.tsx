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
  const previousRouteKind = useRef(route.kind);
  const [draftLogoutWarning, setDraftLogoutWarning] = useState("");
  const identityRef = useRef<SessionIdentity | null>(null);

  function authenticated(identity: SessionIdentity) {
    identityRef.current = identity;
    if (identity.mode === "required") activateDraftSession({ userId: identity.user.id, workspaceId: identity.workspace.id });
    else suspendDraftSession();
    setStartup({ status: "ready", identity });
  }

  function loggedOut() {
    const identity = identityRef.current;
    if (identity?.mode === "required") {
      const cleared = discardDraftSession({ userId: identity.user.id, workspaceId: identity.workspace.id });
      setDraftLogoutWarning(cleared ? "" : "已退出，但浏览器未能清理本账户的标签页暂存。请在共享设备上清除该站点数据；本页面不会再次恢复这份旧暂存。");
    } else suspendDraftSession();
    identityRef.current = null;
    setStartup({ status: "signed-out", identity: null });
  }

  const checkSession = useCallback(async () => {
    try {
      setStartup({ status: "checking", identity: null });
      const identity = await apiJson<SessionIdentity>("/api/v1/auth/me");
      authenticated(identity);
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        suspendDraftSession();
        identityRef.current = null;
        setStartup({ status: "signed-out", identity: null });
        return;
      }
      setStartup({ status: "failed", identity: null, message: apiErrorDetail(error) });
    }
  }, []);

  useEffect(() => {
    initializeBrowserNavigation();
    const handleLocation = (event: PopStateEvent) => {
      if (!handleBrowserPopState(event)) return;
      setLocationKey(currentLocation());
    };
    const handleExpired = (event: Event) => {
      if ((event as CustomEvent<{ explicitLogout?: boolean }>).detail?.explicitLogout === true) {
        loggedOut();
        return;
      }
      suspendDraftSession();
      identityRef.current = null;
      setStartup({ status: "signed-out", identity: null });
    };
    window.addEventListener("popstate", handleLocation);
    window.addEventListener(SESSION_EXPIRED_EVENT, handleExpired);
    void checkSession();
    return () => {
      window.removeEventListener("popstate", handleLocation);
      window.removeEventListener(SESSION_EXPIRED_EVENT, handleExpired);
    };
  }, [checkSession]);

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
      route.kind !== "login" &&
      route.kind !== "register"
    ) {
      browserNavigate(
        loginPath(route.kind === "root" ? undefined : currentLocation()),
        { replace: true, bypassBlockers: true },
      );
    }
  }, [locationKey, route.kind, startup.status]);

  useEffect(() => {
    void locationKey;
    document.body.classList.toggle("loreguard-workspace", route.kind === "workspace");
    return () => document.body.classList.remove("loreguard-workspace");
  }, [locationKey, route.kind]);

  useEffect(() => {
    const previous = previousRouteKind.current;
    previousRouteKind.current = route.kind;
    if (shouldResetProductScroll(previous, route.kind)) {
      window.scrollTo({ top: 0, left: 0, behavior: "auto" });
    }
  }, [locationKey, route.kind]);

  if (startup.status === "checking") {
    return (
      <main className="startupPage productPage" aria-busy="true" aria-label="正在确认登录状态">
        <div className="startupMark"><span className="productSeal" aria-hidden="true"><i>LG</i></span><b>LoreGuard</b></div>
        <div className="startupLine"><i /></div>
        <p>正在打开故事工作区…</p>
      </main>
    );
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

  if (route.kind === "login" || route.kind === "register") {
    if (startup.status === "ready") {
      return <main className="startupPage productPage" aria-busy="true"><p>正在返回项目中心…</p></main>;
    }
    return <>{draftLogoutWarning && <p className="logoutDraftWarning" role="alert">{draftLogoutWarning}</p>}<AuthPage mode={route.kind} onAuthenticated={authenticated} /></>;
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
