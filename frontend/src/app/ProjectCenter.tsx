import {
  FormEvent,
  MouseEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { apiJson } from "../api/client";
import { browserNavigate, workspacePath } from "../routing";
import { documentRoles, type DocumentRole } from "../documentContext";
import { supportedUploadAccept, supportedUploadLabel } from "../uploadFormats";
import {
  createImportFilePlan,
  updateImportFileRole,
  type ImportFilePlan,
} from "./importPlan";
import {
  backendTimestamp,
  projectNextAction,
  relativeProjectDate,
} from "./projectPresentation";
import { apiErrorDetail, type SessionIdentity } from "./session";
import {
  modelProviderSourceLabel,
  parseAccountModelProvider,
} from "./modelProviderSettings";
import { describeRunModelExecution } from "../runModelExecution";

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
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<"recent" | "name">("recent");
  const [samplePending, setSamplePending] = useState(false);
  const [logoutPending, setLogoutPending] = useState(false);
  const [modelSource, setModelSource] = useState(
    "正在读取模型状态",
  );
  const [entryMode, setEntryMode] = useState<"create" | "import" | null>(null);
  const [entryName, setEntryName] = useState("");
  const [entryFiles, setEntryFiles] = useState<ImportFilePlan[]>([]);
  const [entryPending, setEntryPending] = useState(false);
  const [entryError, setEntryError] = useState("");
  const entryNameRef = useRef<HTMLInputElement | null>(null);

  async function loadProjects() {
    try {
      setLoading(true);
      setError("");
      setProjects(await apiJson<ProjectSummary[]>("/api/v1/projects"));
    } catch (reason) {
      setError(`${apiErrorDetail(reason)} 项目列表没有被清空，你可以重试。`);
    } finally {
      setLoading(false);
    }
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
  }, []);

  const visibleProjects = useMemo(() => {
    const keyword = query.trim().toLocaleLowerCase();
    return projects
      .filter((project) =>
        !keyword ||
        project.name.toLocaleLowerCase().includes(keyword) ||
        project.description.toLocaleLowerCase().includes(keyword),
      )
      .sort((a, b) =>
        sort === "name"
          ? a.name.localeCompare(b.name, "zh-CN")
          : backendTimestamp(b.created_at) - backendTimestamp(a.created_at),
      );
  }, [projects, query, sort]);

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
    setEntryMode(mode);
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
    if (entryMode === "import" && entryFiles.length === 0) {
      setEntryError("请选择至少一份故事文稿。");
      return;
    }
    let createdId = "";
    try {
      setEntryPending(true);
      setEntryError("");
      const created = await apiJson<{ id: string }>("/api/v1/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          description: entryMode === "import" ? "从已有故事文稿创建" : "空白故事审查项目",
        }),
      });
      createdId = created.id;
      if (entryMode === "import") {
        for (const entry of entryFiles) {
          const form = new FormData();
          form.append("file", entry.file);
          form.append("document_role", entry.documentRole);
          form.append("story_scope", "global");
          await apiJson(`/api/v1/projects/${created.id}/documents`, {
            method: "POST",
            body: form,
          });
        }
      }
      browserNavigate(workspacePath("projects", created.id));
    } catch (reason) {
      const recovery = createdId
        ? "项目已经创建；请进入项目工作台重新导入失败的文件。"
        : "项目没有创建，你可以修改后重试。";
      setEntryError(`${apiErrorDetail(reason)} ${recovery}`);
      if (createdId) await loadProjects();
    } finally {
      setEntryPending(false);
    }
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
          <button className={`${entryMode ? "" : "quietPrimary"} compactPrimary`} type="button" onClick={() => openEntry("create")}>
            新建项目
          </button>
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
                  maxLength={120}
                  disabled={entryPending}
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
                      setEntryFiles(createImportFilePlan(event.target.files || []));
                      setEntryError("");
                    }}
                    disabled={entryPending}
                  />
                  <small className="fieldHelp">支持 {supportedUploadLabel}；可一次选择多份文件。默认按参考材料导入，进入工作台后仍需确认发布状态与故事位置。</small>
                </label>
              )}
              {entryMode === "import" && entryFiles.length > 0 && (
                <ul className="quickFileRoles" aria-label="逐文件初步设置资料类型">
                  {entryFiles.map((entry, index) => (
                    <li key={`${entry.file.name}-${entry.file.size}-${entry.file.lastModified}-${index}`}>
                      <span title={entry.file.name}>{entry.file.name}</span>
                      <label>
                        <span className="srOnly">{entry.file.name} 的文档类型</span>
                        <select
                          value={entry.documentRole}
                          disabled={entryPending}
                          onChange={(event) =>
                            setEntryFiles((current) =>
                              updateImportFileRole(
                                current,
                                index,
                                event.target.value as DocumentRole,
                              ),
                            )
                          }
                        >
                          {documentRoles.map(([value, label]) => (
                            <option key={value} value={value}>{label}</option>
                          ))}
                        </select>
                      </label>
                    </li>
                  ))}
                </ul>
              )}
              {entryError && <p className="quickEntryError" role="alert">{entryError}</p>}
              <div className="quickEntryActions">
                <button className="quietPrimary" type="submit" disabled={entryPending}>
                  {entryPending ? "正在处理…" : entryMode === "import" ? "创建并导入" : "创建项目"}
                </button>
                <button type="button" disabled={entryPending} onClick={() => browserNavigate("/projects")}>
                  使用完整项目工作台
                </button>
              </div>
            </form>
          </section>
        )}

        <section className="projectListSection" aria-labelledby="project-list-title">
          <div className="projectListHeader">
            <h2 id="project-list-title">最近项目</h2>
            <div className="projectListTools">
              <label className="searchField">
                <span className="srOnly">搜索项目</span>
                <SearchIcon />
                <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索项目" type="search" />
              </label>
              <label className="sortField">
                <span className="srOnly">项目排序</span>
                <select value={sort} onChange={(event) => setSort(event.target.value as "recent" | "name")}>
                  <option value="recent">最近创建</option>
                  <option value="name">按名称</option>
                </select>
              </label>
            </div>
          </div>

          {loading ? (
            <div className="projectSkeleton" aria-label="正在加载项目" aria-busy="true">
              <i /><i /><i />
            </div>
          ) : projects.length === 0 ? (
            <div className="projectEmpty">
              <div className="emptyBookSpirit" aria-hidden="true"><i /><i /><b>··</b></div>
              <h3>还没有项目</h3>
              <p>导入现有文稿，或从原创样例了解一次完整校验。无需先准备世界观设定。</p>
              <button className={entryMode ? "" : "quietPrimary"} type="button" onClick={() => openEntry("import")}>导入已有故事</button>
            </div>
          ) : visibleProjects.length === 0 ? (
            <div className="filteredEmpty">
              <p>没有匹配“{query}”的项目。</p>
              <button type="button" onClick={() => setQuery("")}>清除搜索</button>
            </div>
          ) : (
            <ul className="projectList">
              {visibleProjects.map((project) => {
                const nextAction = projectNextAction(project);
                const latestModelExecution = project.latest_run
                  ? describeRunModelExecution(
                      project.latest_run.model_execution,
                      project.latest_run.status,
                    )
                  : null;
                return (
                <li key={project.id}>
                  <a className="projectRow" href={nextAction.path} onClick={followSpaLink}>
                    <span className="projectRowIdentity"><ProjectIcon /><span><strong>{project.name}</strong><small>{project.description || "尚未添加项目说明"}</small></span></span>
                    <span><small>文档</small><b>{project.active_document_count}</b></span>
                    <span><small>最近运行</small><b>{statusLabel(project.latest_run?.status)}</b>{latestModelExecution && <small className={`projectRunSource ${latestModelExecution.tone}`}>{latestModelExecution.label}</small>}</span>
                    <span><small>创建时间</small><b>{relativeProjectDate(project.created_at)}</b></span>
                    <span className="projectRowAction"><small>下一步</small><b>{nextAction.label}</b></span>
                    <svg className="rowChevron" viewBox="0 0 24 24" aria-hidden="true"><path d="m9 5 7 7-7 7" /></svg>
                  </a>
                </li>
                );
              })}
            </ul>
          )}
        </section>
      </main>
    </div>
  );
}
