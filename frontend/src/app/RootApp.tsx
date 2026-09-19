import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { ApiError, apiJson, SESSION_EXPIRED_EVENT } from "../api/client";
import {
  browserNavigate,
  productRouteFromPath,
  safeReturnTo,
  shouldResetProductScroll,
} from "../routing";
import AuthPage from "./AuthPage";
import ProjectCenter from "./ProjectCenter";
import { apiErrorDetail, type SessionIdentity } from "./session";

const WorkspaceApp = lazy(() => import("../App"));

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

  const checkSession = useCallback(async () => {
    try {
      setStartup({ status: "checking", identity: null });
      const identity = await apiJson<SessionIdentity>("/api/v1/auth/me");
      setStartup({ status: "ready", identity });
    } catch (error) {
      if (error instanceof ApiError && error.status === 401) {
        setStartup({ status: "signed-out", identity: null });
        return;
      }
      setStartup({ status: "failed", identity: null, message: apiErrorDetail(error) });
    }
  }, []);

  useEffect(() => {
    const handleLocation = () => setLocationKey(currentLocation());
    const handleExpired = () => {
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
      return;
    }
    if (
      startup.status === "signed-out" &&
      route.kind !== "login" &&
      route.kind !== "register"
    ) {
      browserNavigate(
        loginPath(route.kind === "root" ? undefined : currentLocation()),
        { replace: true },
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
    return <AuthPage mode={route.kind} onAuthenticated={(identity) => setStartup({ status: "ready", identity })} />;
  }

  if (startup.status !== "ready") {
    return <main className="startupPage productPage" aria-busy="true"><p>正在前往登录页面…</p></main>;
  }

  if (route.kind === "projects" || route.kind === "root") {
    return <ProjectCenter identity={startup.identity} onLoggedOut={() => setStartup({ status: "signed-out", identity: null })} />;
  }

  if (route.kind === "workspace") {
    return (
      <Suspense fallback={<main className="startupPage productPage" aria-busy="true"><p>正在打开项目工作台…</p></main>}>
        <WorkspaceApp
          identity={startup.identity}
          onLoggedOut={() => setStartup({ status: "signed-out", identity: null })}
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
