import { useEffect, useRef, useState } from "react";
import { browserNavigate } from "../routing";
import UserGuide from "../features/help/UserGuide";
import LoginReminder from "./LoginReminder";
import { guestWorkspaceViews, type GuestWorkspaceView } from "./guestWorkspaceViews";
import { publicEntryActions, publicSessionMessage, type PublicFeatureAction, type PublicSessionStatus } from "./publicEntry";
import "./guest-workspace.css";

type Props = {
  view: GuestWorkspaceView;
  status: PublicSessionStatus;
  onRetry: () => void;
  draftLogoutWarning?: string;
};

const viewActions: Record<GuestWorkspaceView, readonly PublicFeatureAction[]> = {
  center: [],
  check: ["input", "file", "check"],
  projects: ["file"],
  characters: ["baseline"],
  diff: ["compare"],
  revision: ["revise"],
  visual: ["graph", "timeline"],
  audit: ["check"],
  report: ["feedback", "export"],
};

function NavigationIcon({ id }: { id: GuestWorkspaceView | "model" | "account" }) {
  const paths = {
    center: "M3 7h7l2 2h9v11H3V7Zm0 0V4h7l2 3",
    check: "M6 3h9l4 4v14H6V3Zm9 0v5h4M9 14l2 2 5-5",
    projects: "M4 4h12v15H4V4Zm4 0v15M16 8h4v14H8v-3",
    characters: "M8 6a4 4 0 1 0 8 0 4 4 0 0 0-8 0Zm-4 15v-2a8 8 0 0 1 16 0v2",
    diff: "M4 3h6v18H4V3Zm10 0h6v18h-6V3ZM7 8h0m10 6h0",
    revision: "M4 5h10v14H4V5Zm11 4 3-3 3 3m-3-3v10a4 4 0 0 1-4 4",
    visual: "M5 12h14M7 6l10 12M7 18 17 6M3 12a2 2 0 1 0 4 0 2 2 0 0 0-4 0Zm14 0a2 2 0 1 0 4 0 2 2 0 0 0-4 0ZM5 4h4v4H5V4Zm10 12h4v4h-4v-4",
    audit: "M6 3h12v18H6V3Zm3 5h6m-6 4h6m-6 4h6",
    report: "M5 3h10l4 4v14H5V3Zm10 0v5h4M8 12h8m-8 4h6",
    model: "M4 9a4 4 0 1 0 8 0 4 4 0 0 0-8 0Zm8 0h9m-3 0v4m3-4v3",
    account: "M12 3 4 6v6c0 5 8 9 8 9s8-4 8-9V6l-8-3Zm-4 9 3 3 5-6",
  };
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d={paths[id]} /></svg>;
}

/** Guest browsing never mounts account resources, private hooks or draft storage. */
export default function GuestWorkspace({ view, status, onRetry, draftLogoutWarning }: Props) {
  const current = guestWorkspaceViews.find((entry) => entry.id === view)!;
  const [selection, setSelection] = useState<{ action: PublicFeatureAction; trigger: HTMLButtonElement } | null>(null);
  const [helpTrigger, setHelpTrigger] = useState<HTMLButtonElement | null>(null);
  const heading = useRef<HTMLHeadingElement | null>(null);

  useEffect(() => {
    heading.current?.focus({ preventScroll: true });
    window.scrollTo({ top: 0, left: 0, behavior: "auto" });
  }, [view]);

  function featureButton(action: PublicFeatureAction, label = publicEntryActions[action].label, className?: string) {
    return <button type="button" className={className}
      onClick={(event) => setSelection({ action, trigger: event.currentTarget })}>{label}</button>;
  }

  return (
    <div className="productPage projectCenterPage guestWorkspace">
      <a className="skipLink" href="#guest-workspace-main">跳到工作区内容</a>
      <aside className="productSidebar" aria-label="全局导航">
        <button className="brandLockup sidebarBrand" type="button" onClick={() => browserNavigate("/app")}>
          <span className="productSeal" aria-hidden="true"><i>LG</i></span>
          <span><b>LoreGuard</b><small>故事审查工作区</small></span>
        </button>
        <nav aria-label="工作区">
          {guestWorkspaceViews.map((entry) => <button key={entry.id} type="button"
            className={view === entry.id ? "active" : undefined}
            aria-label={entry.label}
            title={entry.label}
            aria-current={view === entry.id ? "page" : undefined}
            onClick={() => browserNavigate(entry.path)}><NavigationIcon id={entry.id} /><span>{entry.label}</span></button>)}
          <button type="button" aria-label="模型与密钥" title="模型与密钥" onClick={(event) => setSelection({ action: "model", trigger: event.currentTarget })}>
            <NavigationIcon id="model" /><span>模型与密钥</span>
          </button>
          <button type="button" aria-label="账户安全" title="账户安全" onClick={(event) => setSelection({ action: "account", trigger: event.currentTarget })}>
            <NavigationIcon id="account" /><span>账户安全</span>
          </button>
        </nav>
        <div className="sidebarAccount guestWorkspaceIdentity">
          <span className="productSeal" aria-hidden="true"><i>LG</i></span>
          <span><b>访客</b><small>空工作区</small></span>
        </div>
      </aside>

      <main id="guest-workspace-main" className="projectCenterMain" aria-label="访客空工作区">
        <header className="projectCenterHeader">
          <div><h1 ref={heading} tabIndex={-1}>{current.title}</h1><p>{current.description}</p></div>
          <div className="projectHelpActions">
            <button type="button" onClick={() => browserNavigate("/")}>返回公开首页</button>
            <button type="button" onClick={(event) => setHelpTrigger(event.currentTarget)}>使用指南</button>
            {featureButton("create", "新建项目", view === "center" ? "quietPrimary compactPrimary" : undefined)}
          </div>
        </header>

        {draftLogoutWarning && <p className="guestWorkspaceWarning" role="alert">{draftLogoutWarning}</p>}
        <div className="guestWorkspaceNotice">
          <p role="status">{publicSessionMessage(status)}</p>
          {status === "failed" && <button type="button" onClick={() => {
            onRetry();
            heading.current?.focus({ preventScroll: true });
          }}>重新确认登录状态</button>}
        </div>

        {view === "center" && <section className="projectEntryActions" aria-label="开始方式">
          <button type="button" aria-label="导入已有故事" aria-describedby="guest-import-description" onClick={(event) => setSelection({ action: "import", trigger: event.currentTarget })}>
            <strong>导入已有故事</strong><span id="guest-import-description">支持 DOCX、Markdown、TXT、JSON</span>
          </button>
          <button type="button" aria-label="新建空项目" aria-describedby="guest-create-description" onClick={(event) => setSelection({ action: "create", trigger: event.currentTarget })}>
            <strong>新建空项目</strong><span id="guest-create-description">先建立项目，再逐步添加正文和设定</span>
          </button>
          <button type="button" aria-label="打开原创样例" aria-describedby="guest-sample-description" onClick={(event) => setSelection({ action: "sample", trigger: event.currentTarget })}>
            <strong>打开原创样例</strong><span id="guest-sample-description">登录后由你决定是否创建样例项目</span>
          </button>
        </section>}

        <section className="projectListSection guestWorkspaceEmpty" aria-labelledby="guest-workspace-empty-title">
          {view === "center" && <div className="projectListHeader"><h2>项目列表</h2><span>尚未加载个人项目</span></div>}
          <div className="projectEmpty">
            <div className="emptyBookSpirit" aria-hidden="true"><i /><i /><b>··</b></div>
            <h2 id="guest-workspace-empty-title">{current.emptyTitle}</h2>
            <p>{view === "center" ? "当前显示的是访客空白态。登录后才能读取自己的项目；本页不会创建匿名项目或样例。" : "当前仅展示空白态，没有载入个人文稿、角色或运行。可以先阅读指南，使用功能时再登录。"}</p>
            <div className="guestWorkspaceFeatureActions">
              {viewActions[view].map((action) => <span key={action}>{featureButton(action, undefined, action === "check" ? "startValidation" : undefined)}</span>)}
            </div>
          </div>
        </section>

        {view !== "center" && <section className="guestWorkspaceNext" aria-label="准备项目">
          <p>准备好故事资料后，再登录创建或导入项目。</p>
          <div>{featureButton("create", "新建项目")}{featureButton("import", "导入文稿")}</div>
        </section>}
      </main>
      {selection && <LoginReminder {...selection} status={status} onRetry={onRetry} onClose={() => setSelection(null)} />}
      {helpTrigger && <UserGuide initialTopic={view === "report" ? "report" : view === "check" ? "review" : "start"} trigger={helpTrigger} onClose={() => setHelpTrigger(null)} />}
    </div>
  );
}
