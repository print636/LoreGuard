import {
  FormEvent,
  MouseEvent,
  useEffect,
  useRef,
  useState,
} from "react";
import { ApiError, apiJson } from "../api/client";
import { browserNavigate, workspacePath } from "../routing";
import { supportedUploadAccept, supportedUploadLabel } from "../uploadFormats";
import ImportQueuePanel from "../features/imports/ImportQueuePanel";
import { useImportQueue } from "../features/imports/useImportQueue";
import ProjectMetadataEditor from "../features/projects/ProjectMetadataEditor";
import { verifiedCreatedProjectId, type ProjectMetadata } from "../features/projects/projectMetadata";
import {
  projectNextAction,
  relativeProjectDate,
} from "./projectPresentation";
import { apiErrorDetail, type SessionIdentity } from "./session";
import {
  modelProviderSourceLabel,
  parseAccountModelProvider,
} from "./modelProviderSettings";
import { describeRunModelExecution } from "../runModelExecution";
import UserGuide from "../features/help/UserGuide";

type RunSummary = {
  id: string;
  status: string;
  created_at: string;
  model_execution?: unknown;
};

type ProjectSummary = {
  id: string;
  name: string;
  description: string;
  created_at: string;
  active_document_count: number;
  latest_run: RunSummary | null;
};

type CatalogResponse = {
  page: number;
  page_size: number;
  total: number;
  items: ProjectSummary[];
};

const CATALOG_PAGE_SIZE = 40;
type CatalogCriteria = { page: number; query: string; sort: "recent" | "name" };

function catalogCriteriaFromUrl(): CatalogCriteria {
  const params = new URLSearchParams(window.location.search);
  const requestedPage = Number(params.get("page"));
  return {
    page: Number.isInteger(requestedPage) && requestedPage >= 1 && requestedPage <= 100_000
      ? requestedPage : 1,
    query: (params.get("query") || "").trim().slice(0, 160),
    sort: params.get("sort") === "name" ? "name" : "recent",
  };
}

function replaceCatalogUrl(criteria: CatalogCriteria) {
  const params = new URLSearchParams(window.location.search);
  if (criteria.page === 1) params.delete("page");
  else params.set("page", String(criteria.page));
  if (criteria.query) params.set("query", criteria.query);
  else params.delete("query");
  if (criteria.sort === "recent") params.delete("sort");
  else params.set("sort", criteria.sort);
  const search = params.toString();
  const path = `${window.location.pathname}${search ? `?${search}` : ""}${window.location.hash}`;
  if (path !== `${window.location.pathname}${window.location.search}${window.location.hash}`) {
    browserNavigate(path, { replace: true });
  }
}

type ProjectCenterProps = {
  identity: SessionIdentity;
  onLoggedOut: () => void;
};

function SearchIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="11" cy="11" r="6.5" />
      <path d="m16 16 4.25 4.25" />
    </svg>
  );
}

function ProjectIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 5.5h6l2 2h8v11H4z" />
      <path d="M4 9h16" />
    </svg>
  );
}

function ShieldIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3.5 19 6v5.3c0 4.2-2.8 7.7-7 9.2-4.2-1.5-7-5-7-9.2V6z" />
      <path d="m9.2 12 1.8 1.8 3.9-4.1" />
    </svg>
  );
}

function KeyIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="8.5" cy="12" r="4.5" />
      <path d="M13 12h8M18 12v3M15.5 12v2" />
    </svg>
  );
}

function statusLabel(status: string | undefined): string {
  const labels: Record<string, string> = {
    queued: "等待中",
    running: "校验中",
    completed: "已完成",
    failed: "失败",
    cancelled: "已取消",
  };
  return status ? labels[status] || status : "尚未校验";
}

function followSpaLink(event: MouseEvent<HTMLAnchorElement>) {
  if (
    event.defaultPrevented ||
    event.button !== 0 ||
    event.metaKey ||
    event.ctrlKey ||
    event.shiftKey ||
    event.altKey
  ) {
    return;
  }
  event.preventDefault();
  browserNavigate(event.currentTarget.getAttribute("href") || "/app");
}

export default function ProjectCenter({ identity, onLoggedOut }: ProjectCenterProps) {
  const initialCatalog = catalogCriteriaFromUrl();
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [page, setPage] = useState(initialCatalog.page);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [catalogError, setCatalogError] = useState("");
  const [error, setError] = useState("");
  const [query, setQuery] = useState(initialCatalog.query);
  const [appliedQuery, setAppliedQuery] = useState(initialCatalog.query);
  const [sort, setSort] = useState<"recent" | "name">(initialCatalog.sort);
  const [samplePending, setSamplePending] = useState(false);
  const [logoutPending, setLogoutPending] = useState(false);
  const [modelSource, setModelSource] = useState(
    "正在读取模型状态",
  );
  const [entryMode, setEntryMode] = useState<"create" | "import" | null>(null);
  const [entryName, setEntryName] = useState("");
  const entryQueue = useImportQueue("project-center-entry");
  const [createdEntryId, setCreatedEntryId] = useState("");
  const [creationUnknown, setCreationUnknown] = useState(false);
  const [entryPending, setEntryPending] = useState(false);
  const [entryError, setEntryError] = useState("");
  const [helpTrigger, setHelpTrigger] = useState<HTMLButtonElement | null>(null);
  const [metadataEditor, setMetadataEditor] = useState<{ projectId: string; trigger: HTMLButtonElement } | null>(null);
  const [metadataNotice, setMetadataNotice] = useState("");
  const entryNameRef = useRef<HTMLInputElement | null>(null);
  const catalogRequestRef = useRef(0);
  const catalogBusyRef = useRef(false);
  const searchTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const catalogCriteriaRef = useRef<CatalogCriteria>(initialCatalog);

  async function loadProjects(next: Partial<typeof catalogCriteriaRef.current> = {}) {
    const criteria = { ...catalogCriteriaRef.current, ...next };
    catalogCriteriaRef.current = criteria;
    const request = ++catalogRequestRef.current;
    catalogBusyRef.current = true;
    setPage(criteria.page);
    setAppliedQuery(criteria.query);
    setLoading(true);
    setCatalogError("");
    setProjects([]);
    const params = new URLSearchParams({
      page: String(criteria.page),
      page_size: String(CATALOG_PAGE_SIZE),
      query: criteria.query,
      sort: criteria.sort,
    });
    try {
      const result = await apiJson<CatalogResponse>(`/api/v1/project-catalog?${params}`);
      if (request !== catalogRequestRef.current) return;
      setProjects(result.items);
      setPage(result.page);
      setTotal(result.total);
      return true;
    } catch (reason) {
      if (request === catalogRequestRef.current) {
        setCatalogError(`${apiErrorDetail(reason)} 请重试加载这一页。`);
      }
      return false;
    } finally {
      if (request === catalogRequestRef.current) {
        catalogBusyRef.current = false;
        setLoading(false);
      }
    }
  }

  function changeCatalog(next: Partial<CatalogCriteria>) {
    const criteria = { ...catalogCriteriaRef.current, ...next };
    void loadProjects(criteria);
    replaceCatalogUrl(criteria);
  }

  function searchProjects(value: string) {
    setQuery(value);
    setPage(1);
    ++catalogRequestRef.current;
    catalogBusyRef.current = true;
    setLoading(true);
    setProjects([]);
    setCatalogError("");
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    searchTimerRef.current = setTimeout(() => {
      searchTimerRef.current = null;
      changeCatalog({ page: 1, query: value.trim() });
    }, 300);
  }

  function sortProjects(value: "recent" | "name") {
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    searchTimerRef.current = null;
    setSort(value);
    changeCatalog({ page: 1, query: query.trim(), sort: value });
  }

  async function loadModelSource() {
    try {
      const profile = parseAccountModelProvider(
        await apiJson("/api/v1/account/model-provider"),
      );
      setModelSource(modelProviderSourceLabel(profile));
    } catch {
      setModelSource("模型状态未知");
    }
  }

  useEffect(() => {
    void loadProjects();
    void loadModelSource();
    const syncCatalogFromHistory = () => {
      setHelpTrigger(null);
      if (window.location.pathname !== "/app") return;
      const criteria = catalogCriteriaFromUrl();
      const current = catalogCriteriaRef.current;
      if (criteria.page === current.page && criteria.query === current.query && criteria.sort === current.sort) return;
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
      searchTimerRef.current = null;
      setQuery(criteria.query);
      setSort(criteria.sort);
      void loadProjects(criteria);
    };
    window.addEventListener("popstate", syncCatalogFromHistory);
    return () => {
      ++catalogRequestRef.current;
      if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
      window.removeEventListener("popstate", syncCatalogFromHistory);
    };
  }, []);

  async function openSample() {
    try {
      setSamplePending(true);
      setError("");
      const created = await apiJson<{ id: string }>("/api/v1/demo/advanced", {
        method: "POST",
      });
      browserNavigate(workspacePath("check", created.id));
    } catch (reason) {
      setError(`${apiErrorDetail(reason)} 原创样例未创建，你可以重试。`);
      setSamplePending(false);
    }
  }

  async function logout() {
    try {
      setLogoutPending(true);
      await apiJson("/api/v1/auth/logout", { method: "POST" });
      onLoggedOut();
      browserNavigate("/login", { replace: true });
    } catch (reason) {
      setError(`${apiErrorDetail(reason)} 退出没有完成，请重试。`);
      setLogoutPending(false);
    }
  }

  function openEntry(mode: "create" | "import") {
    setEntryMode(createdEntryId ? "import" : mode);
    setEntryError("");
    requestAnimationFrame(() => entryNameRef.current?.focus());
  }

  async function createEntry(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const name = entryName.trim();
    if (!name) {
      setEntryError("请输入项目名称。");
      entryNameRef.current?.focus();
      return;
    }
    if (creationUnknown) return;
    if (entryMode === "import" && entryQueue.entries.length === 0 && !createdEntryId) {
      setEntryError("请选择至少一份故事文稿。");
      return;
    }
    let createdId = createdEntryId;
    try {
      setEntryPending(true);
      setEntryError("");
      if (!createdId) {
        const created = await apiJson("/api/v1/projects", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name, description: entryMode === "import" ? "从已有故事文稿创建" : "空白故事审查项目" }),
        });
        createdId = verifiedCreatedProjectId(created) || "";
        if (!createdId) throw new Error("Unknown project creation result");
        setCreatedEntryId(createdId);
      }
      if (entryMode === "import") {
        await entryQueue.run(createdId);
        // Keep the queue visible, including successful receipts and unprocessed files.
        await loadProjects();
      } else {
        browserNavigate(workspacePath("projects", createdId));
      }
    } catch (reason) {
      if (!createdId && (!(reason instanceof ApiError) || reason.status >= 500 || reason.status === 408 || reason.status < 400)) {
        setCreationUnknown(true);
        setEntryError("创建结果待核对。服务器可能已经创建项目；请先核对项目目录，不会自动再次创建，也不会凭同名项目猜测导入目标。选中文稿仍保留在本页。");
        await loadProjects();
        return;
      }
      const recovery = createdId
        ? "项目已经创建，导入状态仍保留在队列中；可继续处理未完成文件。"
        : "项目没有创建，你可以修改后重试。";
      setEntryError(`${apiErrorDetail(reason)} ${recovery}`);
      if (createdId) await loadProjects();
    } finally {
      setEntryPending(false);
    }
  }

  async function retryEntry(id: string) {
    if (!createdEntryId || entryPending) return;
    setEntryPending(true);
    try { await entryQueue.run(createdEntryId, id); await loadProjects(); }
    finally { setEntryPending(false); }
  }

  async function metadataSaved(metadata: ProjectMetadata) {
    setProjects((current) => current.map((row) => row.id === metadata.id ? { ...row, name: metadata.name, description: metadata.description } : row));
    setMetadataNotice("项目信息已保存。项目列表将按当前搜索和排序重新加载。");
    if (searchTimerRef.current) clearTimeout(searchTimerRef.current);
    searchTimerRef.current = null;
    if (await loadProjects({ query: query.trim() }) === false) throw new Error("Catalog refresh failed after metadata save");
  }

  return (
    <div className="productPage projectCenterPage">
      <a className="skipLink" href="#project-center-main">跳到项目列表</a>
      <aside className="productSidebar" aria-label="全局导航">
        <button className="brandLockup sidebarBrand" type="button" onClick={() => browserNavigate("/app")}>
          <span className="productSeal" aria-hidden="true"><i>LG</i></span>
          <span><b>LoreGuard</b><small>故事审查工作区</small></span>
        </button>
        <nav aria-label="工作区">
          <button className="active" type="button" aria-current="page" onClick={() => browserNavigate("/app")}>
            <ProjectIcon /><span>项目</span>
          </button>
          <button type="button" onClick={() => browserNavigate("/app/settings/model")}>
            <KeyIcon /><span className="sidebarNavCopy"><span>模型与密钥</span><small>{modelSource}</small></span>
          </button>
          {identity.mode === "required" && (
            <button type="button" onClick={() => browserNavigate("/app/settings/account")}>
              <ShieldIcon /><span>账户安全</span>
            </button>
          )}
        </nav>
        <div className="sidebarAccount">
          <span className="accountAvatar" aria-hidden="true">{identity.user.display_name.slice(0, 1)}</span>
          <span><b>{identity.user.display_name}</b><small>{identity.mode === "anonymous" ? "本地体验模式" : identity.user.email}</small></span>
          {identity.mode === "required" && (
            <div className="sidebarAccountActions">
              <button className="textButton" type="button" onClick={() => browserNavigate("/app/settings/account")}>账户安全</button>
              <button className="textButton" type="button" disabled={logoutPending} onClick={() => void logout()}>
                {logoutPending ? "退出中…" : "退出"}
              </button>
            </div>
          )}
        </div>
      </aside>

      <main id="project-center-main" className="projectCenterMain" tabIndex={-1}>
        <header className="projectCenterHeader">
          <div>
            <h1>项目</h1>
            <p>继续最近的故事审查，或从一份文稿开始。</p>
          </div>
          <div className="projectHelpActions">
            <button type="button" onClick={(event) => setHelpTrigger(event.currentTarget)}>使用指南</button>
            <button className={`${entryMode ? "" : "quietPrimary"} compactPrimary`} type="button" onClick={() => openEntry("create")}>
              新建项目
            </button>
          </div>
        </header>

        {error && (
          <section className="inlineError" role="alert">
            <div><strong>操作没有完成</strong><p>{error}</p></div>
            <button type="button" onClick={() => void loadProjects()}>重试加载</button>
          </section>
        )}

        <section className="projectEntryActions" aria-label="开始方式">
          <button type="button" onClick={() => openEntry("import")}>
            <strong>导入已有故事</strong><span>支持 DOCX、Markdown、TXT、JSON</span>
          </button>
          <button type="button" onClick={() => openEntry("create")}>
            <strong>新建空项目</strong><span>先建立项目，再逐步添加正文和设定</span>
          </button>
          <button type="button" disabled={samplePending} onClick={() => void openSample()}>
            <strong>{samplePending ? "正在创建样例…" : "打开原创样例"}</strong><span>使用多章节素材了解完整审查流程</span>
          </button>
        </section>

        {entryMode && (
          <section className="quickEntry" aria-labelledby="quick-entry-title">
            <div className="quickEntryHead">
              <div>
                <h2 id="quick-entry-title">{entryMode === "import" ? "导入已有故事" : "新建空项目"}</h2>
                <p>{entryMode === "import" ? "世界观设定不是必需；正文会作为故事内容导入。" : "创建后进入项目工作台添加正文、设定和版本。"}</p>
              </div>
              <button className="textButton" type="button" onClick={() => setEntryMode(null)}>关闭</button>
            </div>
            <form onSubmit={createEntry} noValidate>
              <label className="formField">
                <span>项目名称</span>
                <input
                  ref={entryNameRef}
                  value={entryName}
                  onChange={(event) => { setEntryName(event.target.value); setEntryError(""); }}
                  autoComplete="off"
                  maxLength={200}
                  disabled={entryPending || !!createdEntryId || creationUnknown}
                  aria-invalid={entryError.includes("项目名称") || undefined}
                />
              </label>
              {entryMode === "import" && (
                <label className="formField quickFileField">
                  <span>故事文稿</span>
                  <input
                    type="file"
                    multiple
                    accept={supportedUploadAccept}
                    aria-label={`选择${supportedUploadLabel}文件`}
                    onChange={(event) => {
                      entryQueue.add(event.target.files || [], { documentRole: "reference", storyScope: "global", replaceDocumentId: "" });
                      event.target.value = "";
                      setEntryError("");
                    }}
                    disabled={entryPending}
                  />
                  <small className="fieldHelp">支持 {supportedUploadLabel}；可分批选择多份文件。新项目初始按“参考材料／全局”导入，可以逐份调整。上传仅添加文稿，进入工作台后仍需核对资料身份、发布状态与故事位置。</small>
                </label>
              )}
              {entryMode === "import" && <ImportQueuePanel entries={entryQueue.entries} busy={entryPending || entryQueue.busy} notice={entryQueue.notice} onEdit={entryQueue.edit} onRemove={entryQueue.remove} onClearCompleted={entryQueue.clearCompleted} onRetry={(id) => void retryEntry(id)} checkDocumentsHref={createdEntryId ? workspacePath("projects", createdEntryId) : undefined} />}
              {createdEntryId && <p className="fieldHelp">{entryQueue.counts.pending === 0 && entryQueue.counts.unknown === 0 ? "全部文稿已导入。点击‘进入已创建项目’核对资料并准备首次审查。" : "项目已创建。继续导入会复用这个项目，成功文稿不会重复上传。"}</p>}
              {entryError && <p className="quickEntryError" role="alert">{entryError}</p>}
              <div className="quickEntryActions">
                <button className="quietPrimary" type="submit" disabled={entryPending || creationUnknown || (!!createdEntryId && entryQueue.counts.pending === 0)}>
                  {entryPending ? "正在处理…" : createdEntryId ? "继续导入待处理文稿" : entryMode === "import" ? "创建并导入" : "创建项目"}
                </button>
                {createdEntryId && <button type="button" disabled={entryPending} onClick={() => browserNavigate(workspacePath("projects", createdEntryId))}>进入已创建项目</button>}
                {creationUnknown && <button type="button" onClick={() => { if (!window.confirm("请先核对项目目录：原创建请求可能已经生效。放弃本次创建记录后，下一次提交会创建另一个新项目；不会自动绑定到同名项目。仍要重新创建吗？")) return; setCreationUnknown(false); setEntryError(""); }}>已核对目录，重新创建另一个项目</button>}
                {creationUnknown && <a href="/app" target="_blank" rel="noopener noreferrer">在新标签页核对项目目录</a>}
                <button type="button" disabled={entryPending} onClick={() => browserNavigate("/projects")}>
                  使用完整项目工作台
                </button>
              </div>
            </form>
          </section>
        )}

        <section className="projectListSection" aria-labelledby="project-list-title">
          <div className="projectListHeader">
            <h2 id="project-list-title">项目列表</h2>
            <div className="projectListTools">
              <label className="searchField">
                <span className="srOnly">搜索项目</span>
                <SearchIcon />
                <input value={query} onChange={(event) => searchProjects(event.target.value)} placeholder="搜索项目" type="search" maxLength={160} />
              </label>
              <label className="sortField">
                <span className="srOnly">项目排序</span>
                <select value={sort} onChange={(event) => sortProjects(event.target.value as "recent" | "name")}>
                  <option value="recent">最近创建</option>
                  <option value="name">按名称</option>
                </select>
              </label>
            </div>
          </div>

          {metadataNotice && <p role="status">{metadataNotice}</p>}

          <div className="projectCatalogResults" aria-busy={loading}>
          {loading ? (
            <div className="projectSkeleton" aria-label="正在加载项目" aria-busy="true">
              <i /><i /><i />
            </div>
          ) : catalogError ? (
            <div className="filteredEmpty" role="alert">
              <p>{catalogError}</p>
              <button type="button" onClick={() => void loadProjects()}>重试加载</button>
            </div>
          ) : total === 0 && !appliedQuery ? (
            <div className="projectEmpty">
              <div className="emptyBookSpirit" aria-hidden="true"><i /><i /><b>··</b></div>
              <h3>还没有项目</h3>
              <p>导入现有文稿，或从原创样例了解一次完整校验。无需先准备世界观设定。</p>
              <button className={entryMode ? "" : "quietPrimary"} type="button" onClick={() => openEntry("import")}>导入已有故事</button>
            </div>
          ) : total === 0 ? (
            <div className="filteredEmpty">
              <p>没有匹配“{appliedQuery}”的项目。试试其他关键词。</p>
              <button type="button" onClick={() => searchProjects("")}>清除搜索</button>
            </div>
          ) : projects.length === 0 ? (
            <div className="filteredEmpty">
              <p>这一页没有项目。</p>
              <button type="button" onClick={() => changeCatalog({ page: 1 })}>返回第一页</button>
            </div>
          ) : (
            <ul className="projectList">
              {projects.map((project) => {
                const nextAction = projectNextAction(project);
                const latestModelExecution = project.latest_run
                  ? describeRunModelExecution(
                      project.latest_run.model_execution,
                      project.latest_run.status,
                    )
                  : null;
                return (
                <li key={project.id} className="projectRowWithEdit">
                  <a className="projectRow" href={nextAction.path} onClick={followSpaLink}>
                    <span className="projectRowIdentity"><ProjectIcon /><span><strong>{project.name}</strong><small>{project.description || "尚未添加项目说明"}</small></span></span>
                    <span><small>文档</small><b>{project.active_document_count}</b></span>
                    <span><small>最近运行</small><b>{statusLabel(project.latest_run?.status)}</b>{latestModelExecution && <small className={`projectRunSource ${latestModelExecution.tone}`}>{latestModelExecution.label}</small>}</span>
                    <span><small>创建时间</small><b>{relativeProjectDate(project.created_at)}</b></span>
                    <span className="projectRowAction"><small>下一步</small><b>{nextAction.label}</b></span>
                    <svg className="rowChevron" viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7" /></svg>
                  </a>
                  <button className="projectMetadataTrigger" type="button" data-project-metadata-id={project.id} aria-label={`编辑${project.name}的项目信息`} onClick={(event) => setMetadataEditor({ projectId: project.id, trigger: event.currentTarget })}>编辑信息</button>
                </li>
                );
              })}
            </ul>
          )}
          </div>
          {total > 0 && (
            <nav className="projectPagination" aria-label="项目分页" aria-busy={loading}>
              <span role="status">{loading ? `正在加载第 ${page} 页…` : catalogError ? "这一页加载失败" : `第 ${(page - 1) * CATALOG_PAGE_SIZE + 1}–${Math.min(page * CATALOG_PAGE_SIZE, total)} 项，共 ${total} 项`}</span>
              <div>
                <button type="button" aria-disabled={loading || page <= 1} onClick={() => { if (!catalogBusyRef.current && page > 1) changeCatalog({ page: page - 1 }); }}>上一页</button>
                <span>{loading || catalogError ? `第 ${page} 页` : `第 ${page} / ${Math.ceil(total / CATALOG_PAGE_SIZE)} 页`}</span>
                <button type="button" aria-disabled={loading || page * CATALOG_PAGE_SIZE >= total} onClick={() => { if (!catalogBusyRef.current && page * CATALOG_PAGE_SIZE < total) changeCatalog({ page: page + 1 }); }}>下一页</button>
              </div>
            </nav>
          )}
        </section>
      </main>
      {helpTrigger && <UserGuide initialTopic="start" trigger={helpTrigger} onClose={() => setHelpTrigger(null)} />}
      {metadataEditor && <ProjectMetadataEditor key={`${identity.user.id}:${identity.workspace.id}:${metadataEditor.projectId}`} projectId={metadataEditor.projectId} trigger={metadataEditor.trigger} onClose={() => setMetadataEditor(null)} onSaved={metadataSaved} />}
    </div>
  );
}
