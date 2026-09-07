import { lazy, Suspense, useEffect, useMemo, useRef, useState } from "react";
import {
  describeModelStatus,
  describeRepairStatus,
  describeReviewAgentStatus,
  type ModelDiagnostics,
  type RepairStatusView,
  type ReviewAgentStatusView,
} from "./modelStatus";
import NarrativeTimeline from "./components/NarrativeTimeline";
import type {
  Evidence,
  GraphResponse,
  TimelineResponse,
} from "./types/visualization";
import {
  certaintyNames,
  clarificationKindName,
  completedResultPaths,
  modalityNames,
  recordDeveloperFields,
  recordKindName,
  sourceNames,
  summarizeRecord,
  visualizationPath,
  type ClarificationView,
  type RecordView,
} from "./reviewPresentation";
import {
  documentContext,
  documentRoles,
  quickTextDocuments,
  type DocumentRole,
  type QuickTextMode,
} from "./documentContext";
import {
  retryLineage,
  retryState,
  runInputState,
  runSnapshotDocuments,
  shortIdentifier,
  snapshotDocumentLabels,
  type RunInputSnapshot,
} from "./runSnapshot";
import { describeRunUsage, type RunUsageInfo } from "./runUsage";
import {
  docxImportBoundary,
  supportedUploadAccept,
  supportedUploadLabel,
} from "./uploadFormats";
import {
  describeProviderCheck,
  describeProviderHealth,
  unknownProviderConnection,
  type ProviderConnectionView,
} from "./providerConnection";
import IssueEvidenceReview from "./components/IssueEvidenceReview";
import {
  describeIssueEvidenceReview,
  describeIssueEvidenceReviewDiagnostic,
  type IssueEvidenceReviewDiagnosticView,
} from "./issueEvidenceReview";

const RelationGraph = lazy(() => import("./components/RelationGraph"));

const API = import.meta.env.VITE_API_BASE || "";
type FeedbackState = {
  id: string;
  label: string;
  comment: string;
  created_at: string;
};
type Issue = {
  id: string;
  category: string;
  severity: string;
  confidence: number;
  title: string;
  explanation: string;
  suggestion: string;
  evidence: Evidence[];
  metadata?: unknown;
};
type RecordRow = RecordView;
type Doc = {
  id: string;
  project_id: string;
  name: string;
  version: number;
  active: boolean;
  created_at: string;
  content: string;
  document_role: DocumentRole;
  story_scope: string;
  context_explicit?: boolean;
};
type DiffLine = {
  type: "added" | "removed" | "unchanged";
  content: string;
  old_line: number | null;
  new_line: number | null;
};
type DocumentDiff = {
  from_document: Doc & { char_count: number; line_count: number };
  to_document: Doc & { char_count: number; line_count: number };
  summary: {
    added_lines: number;
    removed_lines: number;
    unchanged_lines: number;
    changed_hunks: number;
    compared_old_lines: number;
    compared_new_lines: number;
    old_total_lines: number;
    new_total_lines: number;
    input_truncated: boolean;
    output_truncated: boolean;
  };
  hunks: Array<{
    old_start: number;
    old_lines: number;
    new_start: number;
    new_lines: number;
    lines: DiffLine[];
  }>;
  warnings: string[];
};
type RunInfo = RunUsageInfo & {
  id: string;
  project_id: string;
  prompt_tokens: number;
  completion_tokens: number;
  estimated_cost_usd: number | null;
  error?: string | null;
  created_at: string;
  input_documents?: RunInputSnapshot[];
  input_snapshot_available?: boolean;
  retried_from?: string | null;
  attempt_no?: number | null;
};
type Project = {
  id: string;
  name: string;
  description: string;
  created_at: string;
  active_document_count: number;
  latest_run: RunInfo | null;
};
type Diagnostics = {
  model?: ModelDiagnostics;
  chunking?: {
    total_chunks: number;
    documents: Array<{
      document_name: string;
      chunk_count: number;
      chars: number;
      would_truncate_model_chunks: boolean;
    }>;
  };
  aliases?: {
    declaration_count: number;
    trace_count: number;
    traces: Array<Record<string, unknown>>;
  };
  retrieval?: {
    candidate_count: number;
    consumed_count: number;
    traces: Array<Record<string, unknown>>;
    boundary?: string;
  };
  timings?: {
    chunk_ms: number;
    extract_ms: number;
    index_ms: number;
    check_ms: number;
    report_ms: number;
    total_ms: number;
    first_progress_ms: number;
  };
  ai_evidence_review?: unknown;
};

function RepairDiagnostic({ status }: { status: RepairStatusView }) {
  return (
    <section className={`repairDiagnostic ${status.state}`}>
      <span>固定语义标签 repair</span>
      <code>{status.label}</code>
      <small>
        {status.detail}
        {status.counts && (
          <>
            <br />
            {status.counts}
          </>
        )}
      </small>
    </section>
  );
}

function ReviewAgentDiagnostic({ status }: { status: ReviewAgentStatusView }) {
  return (
    <section className={`agentDiagnostic ${status.state}`}>
      <span>受限证据修复 Agent</span>
      <code>{status.label}</code>
      <small>
        {status.counts}
        <br />
        {status.detail}
        <br />
        覆盖证明：{status.proofDetail}
      </small>
      {status.runs.length > 0 && (
        <div className="agentRuns">
          {status.runs.map((agentRun) => (
            <section className="agentRun" key={agentRun.index}>
              <b>
                RUN {agentRun.index} · {agentRun.outcome}
              </b>
              <strong>{agentRun.protocol}</strong>
              <small>
                {agentRun.orchestrator}
                <br />
                {agentRun.activity}
                <br />
                {agentRun.tokens}
                <br />
                最终原因：{agentRun.finalReason}
                <br />
                {agentRun.traceCount}
              </small>
              {agentRun.trace.length > 0 && (
                <details className="agentTrace">
                  <summary>查看精简 Trace</summary>
                  <ol>
                    {agentRun.trace.map((trace, index) => (
                      <li key={index}>
                        <b>{trace.label}</b>
                        <small>{trace.metrics}</small>
                      </li>
                    ))}
                  </ol>
                </details>
              )}
            </section>
          ))}
        </div>
      )}
    </section>
  );
}

function IssueEvidenceReviewDiagnostic({
  status,
}: {
  status: IssueEvidenceReviewDiagnosticView;
}) {
  return (
    <section className={`issueReviewDiagnostic ${status.state}`}>
      <span>AI 证据复核</span>
      <code>{status.label}</code>
      <small>
        {status.detail}
        <br />
        {status.counts}
        <br />
        复核结果只作独立注释，不会覆盖规则问题。
      </small>
    </section>
  );
}

const defaultWorld = `# 世界观设定

林澈的发色是银色。
星门只能由潮汐晶核驱动。
1026-04-03 08:00，苏弦保管星门钥匙。
1026-04-03 12:00，林澈得知星门口令。`;
const defaultChapter = `# 第一章

林澈的发色是黑色。
1026-04-03 10:00，林澈在北港。
1026-04-03 10:00，林澈在南塔。
1026-04-03 09:00，林澈说出星门口令。
1026-04-03 09:30，林澈使用星门钥匙。
星门由普通火焰驱动。`;
const categoryNames: Record<string, string> = {
  fact_conflict: "事实冲突",
  location_collision: "同刻多地点",
  knowledge_without_acquisition: "知识越权",
  item_ownership: "物品状态",
  world_rule_conflict: "世界规则",
};
const clarificationCategoryNames: Record<string, string> = {
  scope_unknown: "适用范围未确定",
  missing_causal_bridge: "缺少因果衔接",
  missing_state_transition: "缺少状态变化过程",
  ambiguous_reference: "指代不明确",
  insufficient_evidence: "证据不足",
};
const feedbackNames: Record<string, string> = {
  accepted: "已接受",
  false_positive: "误报",
  resolved: "已解决",
};

type WorkspaceView =
  | "check"
  | "projects"
  | "diff"
  | "visual"
  | "audit"
  | "report"
  | "provider";

export default function App() {
  const [activeView, setActiveView] = useState<WorkspaceView>("check");
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState("");
  const [projectName, setProjectName] = useState("");
  const [docs, setDocs] = useState<Doc[]>([]);
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [replaceId, setReplaceId] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [diffFrom, setDiffFrom] = useState("");
  const [diffTo, setDiffTo] = useState("");
  const [documentDiff, setDocumentDiff] = useState<DocumentDiff | null>(null);
  const [diffBusy, setDiffBusy] = useState(false);
  const [world, setWorld] = useState(defaultWorld);
  const [chapter, setChapter] = useState(defaultChapter);
  const [run, setRun] = useState("");
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState("准备就绪");
  const [issues, setIssues] = useState<Issue[]>([]);
  const [records, setRecords] = useState<RecordRow[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [filter, setFilter] = useState("all");
  const [busy, setBusy] = useState(false);
  const [runInfo, setRunInfo] = useState<RunInfo | null>(null);
  const [action, setAction] = useState("");
  const [feedbacks, setFeedbacks] = useState<
    Record<string, FeedbackState | null>
  >({});
  const [notes, setNotes] = useState<Record<string, string>>({});
  const [feedbackPending, setFeedbackPending] = useState<
    Record<string, boolean>
  >({});
  const [diagnostics, setDiagnostics] = useState<Diagnostics>({});
  const [clarifications, setClarifications] = useState<ClarificationView[]>([]);
  const [graph, setGraph] = useState<GraphResponse | null>(null);
  const [timeline, setTimeline] = useState<TimelineResponse | null>(null);
  const [visualTab, setVisualTab] = useState<"graph" | "timeline">("graph");
  const [focusedIssue, setFocusedIssue] = useState<string | null>(null);
  const [visualLoading, setVisualLoading] = useState<
    "graph" | "timeline" | null
  >(null);
  const [visualError, setVisualError] = useState("");
  const [uploadRole, setUploadRole] = useState<DocumentRole>("chapter");
  const [uploadScope, setUploadScope] = useState("global");
  const [quickMode, setQuickMode] = useState<QuickTextMode>("body");
  const [quickRole, setQuickRole] = useState<DocumentRole>("chapter");
  const [quickScope, setQuickScope] = useState("global");
  const [projectLoading, setProjectLoading] = useState(false);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [providerConnection, setProviderConnection] =
    useState<ProviderConnectionView>(unknownProviderConnection);
  const [providerChecking, setProviderChecking] = useState(false);
  const streamRef = useRef<EventSource | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const viewEpochRef = useRef(0);
  const visibleIssues = useMemo(
    () =>
      filter === "all" ? issues : issues.filter((x) => x.category === filter),
    [issues, filter],
  );
  const modelStatus = useMemo(
    () => describeModelStatus(diagnostics.model),
    [diagnostics.model],
  );
  const repairStatus = useMemo(
    () => describeRepairStatus(diagnostics.model),
    [diagnostics.model],
  );
  const reviewAgentStatus = useMemo(
    () => describeReviewAgentStatus(diagnostics.model),
    [diagnostics.model],
  );
  const documentNames = useMemo(
    () =>
      Object.fromEntries(docs.map((document) => [document.id, document.name])),
    [docs],
  );
  const issueReviewCount = useMemo(
    () =>
      issues.filter(
        (issue) =>
          describeIssueEvidenceReview(issue.metadata, documentNames) !== null,
      ).length,
    [issues, documentNames],
  );
  const issueReviewDiagnostic = useMemo(
    () =>
      describeIssueEvidenceReviewDiagnostic(
        diagnostics.ai_evidence_review,
        issueReviewCount,
      ),
    [diagnostics.ai_evidence_review, issueReviewCount],
  );
  const selectedRunUsage = useMemo(() => describeRunUsage(runInfo), [runInfo]);
  const selectedProject = projects.find((row) => row.id === project);

  async function readJson(response: Response) {
    if (!response.ok) {
      const body = await response
        .json()
        .catch(() => ({ detail: response.statusText }));
      throw new Error(body.detail || response.statusText);
    }
    return response.json();
  }
  function clearAnalysisView() {
    streamRef.current?.close();
    streamRef.current = null;
    setBusy(false);
    setAction("");
    setRun("");
    setRunInfo(null);
    setProgress(0);
    setIssues([]);
    setClarifications([]);
    setRecords([]);
    setWarnings([]);
    setFeedbacks({});
    setFeedbackPending({});
    setNotes({});
    setDiagnostics({});
    setGraph(null);
    setTimeline(null);
    setVisualLoading(null);
    setVisualError("");
    setFocusedIssue(null);
  }
  async function loadProjects() {
    const rows = await readJson(await fetch(`${API}/api/v1/projects`));
    setProjects(rows);
  }
  async function loadProviderHealth() {
    try {
      setProviderConnection(
        describeProviderHealth(await readJson(await fetch(`${API}/health`))),
      );
    } catch {
      setProviderConnection({
        ...unknownProviderConnection,
        tone: "error",
        label: "无法读取模型配置状态",
        detail: "LoreGuard API 暂时不可用；页面没有发起模型调用。",
        suggestions: ["确认 API 服务正常运行后刷新页面。"],
      });
    }
  }
  async function checkProviderConnection() {
    try {
      setProviderChecking(true);
      setProviderConnection({
        ...providerConnection,
        tone: "neutral",
        label: "正在测试模型连接",
        detail: "正在发起一次最小 JSON 调用，可能消耗少量 Token。",
      });
      const payload = await readJson(
        await fetch(`${API}/api/v1/model/provider-check`, { method: "POST" }),
      );
      setProviderConnection(describeProviderCheck(payload));
    } catch {
      setProviderConnection({
        ...unknownProviderConnection,
        tone: "error",
        label: "无法完成模型连接测试",
        detail: "LoreGuard API 没有返回可用的安全检查结果。",
        suggestions: ["确认 API 服务正常运行后重试。"],
      });
    } finally {
      setProviderChecking(false);
    }
  }
  async function refreshProjects() {
    try {
      setProjectsLoading(true);
      await loadProjects();
      setMessage("项目列表已刷新");
    } catch (error) {
      setMessage(`刷新项目失败：${String(error)}`);
    } finally {
      setProjectsLoading(false);
    }
  }
  async function loadProject(id: string) {
    const epoch = ++viewEpochRef.current;
    setProject(id);
    setDocumentDiff(null);
    setDocs([]);
    setRuns([]);
    setDiffFrom("");
    setDiffTo("");
    clearAnalysisView();
    if (!id) {
      setMessage("已取消项目选择");
      return;
    }
    try {
      setProjectLoading(true);
      setMessage("正在加载项目…");
      const [documents, history] = await Promise.all([
        readJson(
          await fetch(
            `${API}/api/v1/projects/${id}/documents?include_history=true`,
          ),
        ),
        readJson(await fetch(`${API}/api/v1/projects/${id}/analysis-runs`)),
      ]);
      if (epoch !== viewEpochRef.current) return;
      const documentRows = documents as Doc[];
      setDocs(documentRows);
      setRuns(history);
      const versionGroup = (
        documentRows
          .map((row) =>
            documentRows.filter(
              (candidate) =>
                candidate.name.toLocaleLowerCase() ===
                row.name.toLocaleLowerCase(),
            ),
          )
          .find((group) => group.length >= 2) || []
      ).sort((a, b) => a.version - b.version);
      if (versionGroup.length >= 2) {
        setDiffFrom(versionGroup[0].id);
        setDiffTo(versionGroup[versionGroup.length - 1].id);
      }
      if (history[0]) await restoreRun(history[0], true, epoch);
      else
        setMessage(
          documentRows.length
            ? "项目已加载，尚未运行分析"
            : "项目已加载，请先上传剧情文档",
        );
    } catch (error) {
      if (epoch === viewEpochRef.current)
        setMessage(`加载项目失败：${String(error)}`);
    } finally {
      if (epoch === viewEpochRef.current) setProjectLoading(false);
    }
  }
  async function loadFeedback(rows: Issue[], epoch: number) {
    const pairs = await Promise.all(
      rows.map(async (issue) => {
        const value = await readJson(
          await fetch(`${API}/api/v1/issues/${issue.id}/feedback`),
        );
        return [issue.id, value.latest] as const;
      }),
    );
    if (epoch === viewEpochRef.current) setFeedbacks(Object.fromEntries(pairs));
  }
  async function loadCompleted(runId: string, epoch = viewEpochRef.current) {
    const paths = completedResultPaths(runId);
    const [ir, rr, sr, dr, cr] = await Promise.all([
      fetch(`${API}${paths.issues}`),
      fetch(`${API}${paths.records}`),
      fetch(`${API}${paths.status}`),
      fetch(`${API}${paths.diagnostics}`),
      fetch(`${API}${paths.clarifications}`),
    ]);
    const loadedIssues = await readJson(ir);
    const rec = await readJson(rr);
    const status = await readJson(sr);
    const diag = await readJson(dr);
    const reviewItems = await readJson(cr);
    if (epoch !== viewEpochRef.current) return;
    setIssues(loadedIssues);
    setClarifications(reviewItems);
    setRecords(rec.records);
    setWarnings(rec.warnings);
    setRunInfo(status);
    setDiagnostics(diag);
    setGraph(null);
    setTimeline(null);
    setVisualLoading(null);
    setVisualError("");
    await loadFeedback(loadedIssues, epoch);
  }
  function subscribe(
    runId: string,
    projectId: string,
    epoch = viewEpochRef.current,
  ) {
    streamRef.current?.close();
    const es = new EventSource(`${API}/api/v1/analysis-runs/${runId}/events`);
    streamRef.current = es;
    es.addEventListener("progress", (event) => {
      if (epoch !== viewEpochRef.current) return;
      const data = JSON.parse((event as MessageEvent).data);
      setProgress(data.progress);
      setMessage(data.message);
    });
    es.addEventListener("terminal", async (event) => {
      const data = JSON.parse((event as MessageEvent).data);
      es.close();
      if (epoch !== viewEpochRef.current) return;
      streamRef.current = null;
      setBusy(false);
      setAction("");
      setMessage(
        data.error
          ? `${data.status}：${data.error}`
          : `任务状态：${data.status}`,
      );
      try {
        const status = await readJson(
          await fetch(`${API}/api/v1/analysis-runs/${runId}`),
        );
        if (epoch !== viewEpochRef.current) return;
        setRunInfo(status);
        if (data.status === "completed") await loadCompleted(runId, epoch);
        await Promise.all([loadProjects(), loadProjectRuns(projectId)]);
      } catch (error) {
        if (epoch === viewEpochRef.current)
          setMessage(
            `任务已结束，但结果加载失败：${String(error)}。可从运行历史再次恢复。`,
          );
      }
    });
    es.onerror = () => {
      es.close();
      if (epoch !== viewEpochRef.current) return;
      streamRef.current = null;
      setMessage("SSE 连接中断，可从运行历史恢复");
      setBusy(false);
    };
  }
  async function loadProjectRuns(id: string) {
    if (!id) return;
    setRuns(
      await readJson(await fetch(`${API}/api/v1/projects/${id}/analysis-runs`)),
    );
  }
  async function restoreRun(
    info: RunInfo,
    subscribeIfActive = true,
    existingEpoch?: number,
  ) {
    if (existingEpoch === undefined) {
      streamRef.current?.close();
      streamRef.current = null;
      setBusy(false);
    }
    const epoch = existingEpoch ?? ++viewEpochRef.current;
    if (epoch !== viewEpochRef.current) return;
    setRun(info.id);
    setRunInfo(info);
    setProgress(info.status === "completed" ? 100 : 0);
    setMessage(info.error || `历史任务：${info.status}`);
    setIssues([]);
    setClarifications([]);
    setRecords([]);
    setWarnings([]);
    setFeedbacks({});
    setFeedbackPending({});
    setNotes({});
    setDiagnostics({});
    setGraph(null);
    setTimeline(null);
    setVisualLoading(null);
    setVisualError("");
    setFocusedIssue(null);
    if (info.status === "completed") await loadCompleted(info.id, epoch);
    else if (subscribeIfActive && ["queued", "running"].includes(info.status)) {
      setBusy(true);
      subscribe(info.id, info.project_id, epoch);
    }
  }
  async function restoreSelectedRun(info: RunInfo) {
    try {
      setAction(`restore:${info.id}`);
      await restoreRun(info);
    } catch (error) {
      setMessage(`恢复运行失败：${String(error)}`);
    } finally {
      setAction("");
    }
  }
  async function runProject(id: string) {
    if (!id) throw new Error("请先选择项目");
    const epoch = ++viewEpochRef.current;
    setProject(id);
    setBusy(true);
    setRun("");
    setIssues([]);
    setClarifications([]);
    setRecords([]);
    setDiagnostics({});
    setGraph(null);
    setTimeline(null);
    setVisualLoading(null);
    setVisualError("");
    setFocusedIssue(null);
    setRunInfo(null);
    setProgress(0);
    setMessage("任务已提交（若服务器启用模型，可能消耗 Token）");
    const created = await readJson(
      await fetch(`${API}/api/v1/projects/${id}/analysis-runs`, {
        method: "POST",
      }),
    );
    if (epoch !== viewEpochRef.current) return;
    setRun(created.id);
    subscribe(created.id, id, epoch);
    try {
      const status = await readJson(
        await fetch(`${API}/api/v1/analysis-runs/${created.id}`),
      );
      if (epoch !== viewEpochRef.current) return;
      setRunInfo(status);
      await loadProjectRuns(id);
    } catch (error) {
      if (epoch === viewEpochRef.current)
        setMessage(
          `任务已启动，但输入快照或运行历史暂时无法刷新：${String(error)}`,
        );
    }
  }
  async function startCurrentProject() {
    try {
      await runProject(project);
    } catch (error) {
      setBusy(false);
      setMessage(`启动分析失败：${String(error)}`);
    }
  }
  async function createProject() {
    try {
      if (!projectName.trim()) return;
      setAction("create");
      const created = await readJson(
        await fetch(`${API}/api/v1/projects`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: projectName.trim(),
            description: "本地工作台项目",
          }),
        }),
      );
      setProjectName("");
      await loadProjects();
      await loadProject(created.id);
    } catch (error) {
      setMessage(`新建项目失败：${String(error)}`);
    } finally {
      setAction("");
    }
  }
  async function uploadDocuments() {
    try {
      if (!project || files.length === 0) return;
      if (replaceId && files.length !== 1)
        throw new Error("替换版本时只能选择一个同名文件");
      const context = documentContext(uploadRole, uploadScope);
      setAction("upload");
      let uploaded = 0;
      const failed: string[] = [];
      for (const file of files) {
        try {
          const form = new FormData();
          form.append("file", file);
          form.append("document_role", context.document_role);
          form.append("story_scope", context.story_scope);
          if (replaceId) form.append("replace_document_id", replaceId);
          await readJson(
            await fetch(`${API}/api/v1/projects/${project}/documents`, {
              method: "POST",
              body: form,
            }),
          );
          uploaded += 1;
        } catch (error) {
          failed.push(`${file.name}（${String(error)}）`);
        }
      }
      if (uploaded > 0) {
        await loadProjects();
        await loadProject(project);
      }
      setFiles([]);
      setReplaceId("");
      if (fileInputRef.current) fileInputRef.current.value = "";
      setMessage(
        failed.length
          ? `已上传 ${uploaded}/${uploaded + failed.length} 个文件；失败：${failed.join("、")}`
          : `已上传 ${uploaded} 个文件；同名文件已自动生成新版本`,
      );
    } catch (error) {
      setMessage(String(error));
    } finally {
      setAction("");
    }
  }
  async function demo(kind: "simple" | "advanced") {
    try {
      setAction(kind);
      const url =
        kind === "advanced" ? "/api/v1/demo/advanced" : "/api/v1/demo";
      const created = await readJson(
        await fetch(`${API}${url}`, { method: "POST" }),
      );
      await loadProjects();
      await loadProject(created.id);
      await runProject(created.id);
    } catch (error) {
      setBusy(false);
      setAction("");
      setMessage(String(error));
    }
  }
  async function custom() {
    try {
      const inputs = quickTextDocuments({
        mode: quickMode,
        world,
        chapter,
        role: quickRole,
        scope: quickScope,
      });
      setBusy(true);
      setAction("custom");
      const created = await readJson(
        await fetch(`${API}/api/v1/projects`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: `自然文本审查 ${new Date().toLocaleString()}`,
            description:
              quickMode === "body" ? "单篇正文审查" : "设定与章节审查",
          }),
        }),
      );
      for (const input of inputs)
        await readJson(
          await fetch(`${API}/api/v1/projects/${created.id}/documents/text`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(input),
          }),
        );
      await loadProjects();
      await loadProject(created.id);
      await runProject(created.id);
    } catch (error) {
      setBusy(false);
      setAction("");
      setMessage(String(error));
    }
  }
  async function cancel() {
    try {
      if (!run) return;
      await readJson(
        await fetch(`${API}/api/v1/analysis-runs/${run}/cancel`, {
          method: "POST",
        }),
      );
      setMessage("取消请求已提交");
    } catch (error) {
      setMessage(String(error));
    }
  }
  async function retry() {
    try {
      if (!run || !runInfo || !retryState(runInfo).allowed) return;
      const projectId = runInfo.project_id;
      const epoch = ++viewEpochRef.current;
      const created = await readJson(
        await fetch(`${API}/api/v1/analysis-runs/${run}/retry`, {
          method: "POST",
        }),
      );
      if (epoch !== viewEpochRef.current) return;
      setRun(created.id);
      setRunInfo(null);
      setBusy(true);
      setMessage(
        `重试已提交，继承运行 ${shortIdentifier(created.retried_from)} 的冻结输入（若服务器启用模型，可能消耗 Token）`,
      );
      subscribe(created.id, projectId, epoch);
      try {
        const status = await readJson(
          await fetch(`${API}/api/v1/analysis-runs/${created.id}`),
        );
        if (epoch !== viewEpochRef.current) return;
        setRunInfo(status);
        await loadProjectRuns(projectId);
      } catch (error) {
        if (epoch === viewEpochRef.current)
          setMessage(
            `重试已启动，但输入快照或运行历史暂时无法刷新：${String(error)}`,
          );
      }
    } catch (error) {
      setBusy(false);
      setMessage(String(error));
    }
  }
  async function submitFeedback(id: string, label: string) {
    if (feedbackPending[id]) return;
    try {
      setFeedbackPending((current) => ({ ...current, [id]: true }));
      const comment = notes[id] || "";
      const value = await readJson(
        await fetch(`${API}/api/v1/issues/${id}/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ label, comment }),
        }),
      );
      setFeedbacks((current) => ({ ...current, [id]: value }));
      setMessage(
        value.duplicate_ignored
          ? "相同反馈已存在，未重复写入"
          : "反馈已记录并保留审计历史",
      );
    } catch (error) {
      setMessage(`提交反馈失败：${String(error)}`);
    } finally {
      setFeedbackPending((current) => ({ ...current, [id]: false }));
    }
  }
  function selectDiffFrom(id: string) {
    setDiffFrom(id);
    setDocumentDiff(null);
    const source = docs.find((row) => row.id === id);
    const candidates = source
      ? docs.filter(
          (row) =>
            row.name.toLocaleLowerCase() === source.name.toLocaleLowerCase() &&
            row.id !== id,
        )
      : [];
    setDiffTo(candidates.sort((a, b) => b.version - a.version)[0]?.id || "");
  }
  async function compareVersions() {
    try {
      if (!project || !diffFrom || !diffTo) return;
      setDiffBusy(true);
      setDocumentDiff(
        await readJson(
          await fetch(
            `${API}/api/v1/projects/${project}/documents/diff?from_document_id=${encodeURIComponent(diffFrom)}&to_document_id=${encodeURIComponent(diffTo)}`,
          ),
        ),
      );
    } catch (error) {
      setMessage(String(error));
      setDocumentDiff(null);
    } finally {
      setDiffBusy(false);
    }
  }
  function selectReplacement(id: string) {
    setReplaceId(id);
    const selected = docs.find((row) => row.id === id);
    if (selected) {
      setUploadRole(selected.document_role || "chapter");
      setUploadScope(selected.story_scope || "global");
    }
  }
  async function loadVisualization(kind: "graph" | "timeline") {
    setVisualTab(kind);
    if (!run || runInfo?.status !== "completed") return;
    if ((kind === "graph" && graph) || (kind === "timeline" && timeline))
      return;
    const epoch = viewEpochRef.current;
    try {
      setVisualLoading(kind);
      setVisualError("");
      const data = await readJson(
        await fetch(`${API}${visualizationPath(run, kind)}`),
      );
      if (epoch !== viewEpochRef.current) return;
      if (kind === "graph") setGraph(data);
      else setTimeline(data);
    } catch (error) {
      if (epoch === viewEpochRef.current)
        setVisualError(
          `加载${kind === "graph" ? "关系图" : "时间线"}失败：${String(error)}`,
        );
    } finally {
      if (epoch === viewEpochRef.current) setVisualLoading(null);
    }
  }

  useEffect(() => {
    loadProjects().catch((error) => setMessage(String(error)));
    void loadProviderHealth();
    return () => streamRef.current?.close();
  }, []);

  return (
    <>
      <div className="ambientDecor" aria-hidden="true">
        <i />
        <i />
        <i />
        <span>✦</span>
      </div>
      <header className="topBar">
        <button
          className="brand"
          onClick={() => setActiveView("check")}
          aria-label="返回文稿校验台"
        >
          <span className="brandSeal" aria-hidden="true">
            <i>LG</i>
          </span>
          <span>
            <strong>LoreGuard</strong>
            <small>剧情一致性审查</small>
          </span>
        </button>
        <div className="topContext">
          <button onClick={() => setActiveView("projects")}>
            <small>当前项目</small>
            <b>{selectedProject?.name || "尚未选择项目"}</b>
            <span>
              {selectedProject
                ? `${selectedProject.active_document_count} 份文档 · ${selectedProject.latest_run?.status || "未运行"}`
                : "点击选择或新建"}
            </span>
          </button>
          <button
            className={`providerPill ${providerConnection.tone}`}
            onClick={() => setActiveView("provider")}
          >
            <i aria-hidden="true" />
            <span>{providerConnection.label}</span>
          </button>
        </div>
        <div className="topActions">
          <button
            onClick={() => {
              setActiveView("projects");
              requestAnimationFrame(() =>
                document.getElementById("project-name")?.focus(),
              );
            }}
          >
            新建项目
          </button>
          <button
            onClick={() => {
              setActiveView("projects");
              if (!project) setMessage("请先新建或选择项目，再导入文稿");
              requestAnimationFrame(() =>
                document.getElementById("document-upload")?.focus(),
              );
            }}
          >
            导入文稿
          </button>
          <button
            className="reportShortcut"
            onClick={() => setActiveView("report")}
          >
            查看报告 <span>{issues.length + clarifications.length}</span>
          </button>
          <button
            className="primary topRunButton"
            disabled={
              !project ||
              busy ||
              projectLoading ||
              docs.filter((row) => row.active).length === 0
            }
            title="分析当前项目；服务端启用模型时可能消耗 Token"
            onClick={startCurrentProject}
          >
            <span aria-hidden="true">✦</span>
            {busy ? "校验中…" : "开始校验"}
          </button>
        </div>
      </header>
      <main className="appShell">
        <aside className="sideNav" aria-label="创作工作台导航">
          <div className="sideNavTitle">
            <span>CREATIVE INDEX</span>
            <b>创作索引</b>
          </div>
          <nav>
            {(
              [
                ["check", "PT", "文稿校验", busy ? "运行中" : ""],
                [
                  "projects",
                  "PJ",
                  "项目与文档",
                  String(docs.filter((row) => row.active).length || ""),
                ],
                ["diff", "DF", "版本对比", documentDiff ? "1" : ""],
                [
                  "visual",
                  "TL",
                  "关系与时间",
                  graph || timeline ? "已生成" : "",
                ],
                ["audit", "AU", "运行审计", String(records.length || "")],
                [
                  "report",
                  "RP",
                  "完整报告",
                  String(issues.length + clarifications.length || ""),
                ],
              ] as Array<[WorkspaceView, string, string, string]>
            ).map(([view, glyph, label, badge]) => (
              <button
                key={view}
                className={activeView === view ? "active" : ""}
                aria-current={activeView === view ? "page" : undefined}
                aria-label={label}
                title={label}
                onClick={() => setActiveView(view)}
              >
                <span className="navGlyph" aria-hidden="true">
                  {glyph}
                </span>
                <span className="navLabel">{label}</span>
                {badge && <small>{badge}</small>}
              </button>
            ))}
          </nav>
          <div className="sideNavBottom">
            <button
              className={`navProvider ${providerConnection.tone}`}
              onClick={() => setActiveView("provider")}
            >
              <i aria-hidden="true" />
              <span>
                <small>模型连接</small>
                <b>{providerConnection.label}</b>
              </span>
            </button>
            <details className="sampleShelf">
              <summary>样例与帮助</summary>
              <p>原创演示文本；运行模型时可能消耗 Token。</p>
              <button disabled={busy} onClick={() => demo("simple")}>
                {action === "simple" ? "分析中…" : "运行简单样例"}
              </button>
              <button disabled={busy} onClick={() => demo("advanced")}>
                {action === "advanced" ? "分析中…" : "运行复杂样例"}
              </button>
            </details>
          </div>
        </aside>

        <div className="mainCanvas">
          {busy && (
            <section className="mobileRunStrip" aria-live="polite">
              <span><i aria-hidden="true" />校验中 · {progress}%</span>
              <div className="meter"><i style={{ width: `${progress}%` }} /></div>
              <button onClick={cancel}>取消</button>
            </section>
          )}
          {activeView === "check" && (
            <section className="workbenchIntro">
              <div>
                <p className="eyebrow">NARRATIVE CONSISTENCY REVIEW</p>
                <h1>文稿校验台</h1>
                <p>让每一条伏笔都沿着星轨归位，所有判断回到原文证据。</p>
              </div>
              <span className="workbenchMascot" aria-hidden="true">
                <i />
                <i />
                <b>··</b>
                <em>✦</em>
              </span>
            </section>
          )}

          {activeView === "provider" && (
            <section
              className={`providerConnection ${providerConnection.tone}`}
              aria-live="polite"
            >
              <div className="providerConnectionHead">
                <div>
                  <p className="eyebrow">MODEL CONNECTION</p>
                  <h2>{providerConnection.label}</h2>
                  <p>{providerConnection.detail}</p>
                </div>
                <button
                  className="providerCheckButton"
                  disabled={providerChecking}
                  onClick={() => void checkProviderConnection()}
                >
                  {providerChecking ? "测试中…" : "测试模型连接（少量 Token）"}
                </button>
              </div>
              <div className="providerMetrics">
                <div>
                  <small>服务端配置</small>
                  <b>
                    {providerConnection.configured === true
                      ? "已配置"
                      : providerConnection.configured === false
                        ? "未配置"
                        : "未知"}
                  </b>
                </div>
                <div>
                  <small>JSON 合同</small>
                  <b>
                    {providerConnection.jsonContractOk === true
                      ? "通过"
                      : providerConnection.jsonContractOk === false
                        ? "未通过"
                        : "未执行"}
                  </b>
                </div>
                <div>
                  <small>安全分类</small>
                  <b>{providerConnection.categoryLabel}</b>
                </div>
                <div>
                  <small>本次延迟</small>
                  <b>{providerConnection.latencyLabel}</b>
                </div>
                <div>
                  <small>本次 Token</small>
                  <b>{providerConnection.tokensLabel}</b>
                </div>
              </div>
              {providerConnection.suggestions.length > 0 && (
                <ul className="providerSuggestions">
                  {providerConnection.suggestions.map((suggestion, index) => (
                    <li key={index}>{suggestion}</li>
                  ))}
                </ul>
              )}
              <details className="providerSetup">
                <summary>如何安全配置模型？</summary>
                <p>
                  为避免凭据进入浏览器、数据库或页面响应，本页不提供 API Key
                  输入框。请只在服务端未提交的 <code>.env</code> 中设置{" "}
                  <code>ENABLE_MODEL_EXTRACTION=true</code>、
                  <code>OPENAI_BASE_URL</code>、<code>OPENAI_API_KEY</code> 和{" "}
                  <code>OPENAI_MODEL</code>，保存后重启 API 与 Celery
                  worker。连接测试不会显示 endpoint、Key、Prompt
                  或模型原始响应。
                </p>
              </details>
            </section>
          )}

          {(activeView === "projects" ||
            activeView === "diff" ||
            activeView === "audit") && (
            <section
              className="projectPanel workspaceView"
              aria-busy={projectLoading}
            >
              {activeView === "projects" && (
                <div className="projectSetup">
                  <div className="sectionHead">
                    <div>
                      <p className="eyebrow">PROJECTS</p>
                      <h2>本地项目工作台</h2>
                    </div>
                    <button
                      disabled={projectsLoading}
                      onClick={refreshProjects}
                    >
                      {projectsLoading ? "刷新中…" : "刷新列表"}
                    </button>
                  </div>
                  <div className="projectControls">
                    <select
                      aria-label="选择项目"
                      disabled={projectLoading}
                      value={project}
                      onChange={(event) => void loadProject(event.target.value)}
                    >
                      <option value="">选择已有项目</option>
                      {projects.map((row) => (
                        <option key={row.id} value={row.id}>
                          {row.name} · {row.active_document_count} 文档 ·{" "}
                          {row.latest_run?.status || "未运行"}
                        </option>
                      ))}
                    </select>
                    <input
                      id="project-name"
                      value={projectName}
                      onChange={(event) => setProjectName(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" && projectName.trim())
                          void createProject();
                      }}
                      placeholder="新项目名称"
                    />
                    <button
                      disabled={!projectName.trim() || action === "create"}
                      onClick={createProject}
                    >
                      {action === "create" ? "创建中…" : "新建项目"}
                    </button>
                    <button
                      className="primary"
                      disabled={
                        !project ||
                        busy ||
                        projectLoading ||
                        docs.filter((row) => row.active).length === 0
                      }
                      onClick={startCurrentProject}
                    >
                      分析当前项目（可能消耗 Token）
                    </button>
                  </div>
                  {projectLoading ? (
                    <p className="hint">正在读取文档版本和运行历史…</p>
                  ) : (
                    selectedProject && (
                      <p className="hint">
                        当前：{selectedProject.name} ·{" "}
                        {selectedProject.description || "无描述"}
                      </p>
                    )
                  )}
                  <div className="uploadRow">
                    <input
                      id="document-upload"
                      ref={fileInputRef}
                      type="file"
                      multiple
                      accept={supportedUploadAccept}
                      aria-label={`选择${supportedUploadLabel}文件`}
                      onChange={(event) =>
                        setFiles(Array.from(event.target.files || []))
                      }
                    />
                    <select
                      aria-label="文档版本处理方式"
                      value={replaceId}
                      onChange={(event) =>
                        selectReplacement(event.target.value)
                      }
                    >
                      <option value="">同名自动新版本 / 新文件</option>
                      {docs
                        .filter((row) => row.active)
                        .map((row) => (
                          <option key={row.id} value={row.id}>
                            明确替换 {row.name} v{row.version}
                          </option>
                        ))}
                    </select>
                    <select
                      aria-label="上传文档类型"
                      value={uploadRole}
                      onChange={(event) =>
                        setUploadRole(event.target.value as DocumentRole)
                      }
                    >
                      {documentRoles.map(([value, label]) => (
                        <option key={value} value={value}>
                          {label}
                        </option>
                      ))}
                    </select>
                    <input
                      aria-label="上传文档故事作用域"
                      maxLength={80}
                      value={uploadScope}
                      onChange={(event) => setUploadScope(event.target.value)}
                      placeholder="故事作用域，如 global 或 route_a"
                    />
                    <button
                      disabled={
                        !project ||
                        files.length === 0 ||
                        action === "upload" ||
                        projectLoading
                      }
                      onClick={uploadDocuments}
                    >
                      {action === "upload"
                        ? "上传中…"
                        : `上传${files.length ? ` ${files.length} ` : " "}个文件`}
                    </button>
                  </div>
                  <p className="contextHint">
                    支持 {supportedUploadLabel}。{docxImportBoundary}
                  </p>
                  {files.length > 0 && (
                    <p
                      className="uploadSelection"
                      title={files.map((file) => file.name).join("\n")}
                    >
                      已选择：{files.map((file) => file.name).join("、")} ·
                      将保存为“
                      {
                        documentRoles.find(
                          ([value]) => value === uploadRole,
                        )?.[1]
                      }
                      ” · 作用域 {uploadScope.trim() || "global"}
                    </p>
                  )}
                </div>
              )}
              {activeView !== "diff" && (
                <div className="tables">
                  {activeView === "projects" && (
                    <div>
                      <h3>文档版本</h3>
                      <table>
                        <thead>
                          <tr>
                            <th>文件</th>
                            <th>版本</th>
                            <th>类型</th>
                            <th>作用域</th>
                            <th>状态</th>
                            <th>创建时间</th>
                          </tr>
                        </thead>
                        <tbody>
                          {docs.length === 0 ? (
                            <tr>
                              <td className="tableEmpty" colSpan={6}>
                                {project
                                  ? `暂无文档，请上传${supportedUploadLabel}`
                                  : "选择项目后查看文档版本"}
                              </td>
                            </tr>
                          ) : (
                            docs.map((row) => (
                              <tr key={row.id}>
                                <td>{row.name}</td>
                                <td>v{row.version}</td>
                                <td>
                                  {documentRoles.find(
                                    ([value]) => value === row.document_role,
                                  )?.[1] || row.document_role}
                                </td>
                                <td>{row.story_scope}</td>
                                <td>
                                  <span
                                    className={`badge ${row.active ? "ok" : "muted"}`}
                                  >
                                    {row.active ? "active" : "history"}
                                  </span>
                                </td>
                                <td>
                                  {new Date(row.created_at).toLocaleString()}
                                </td>
                              </tr>
                            ))
                          )}
                        </tbody>
                      </table>
                    </div>
                  )}
                  {activeView === "audit" && (
                    <div>
                      <h3>运行历史（冻结输入）</h3>
                      <table>
                        <thead>
                          <tr>
                            <th>运行 / 时间</th>
                            <th>状态</th>
                            <th>输入快照</th>
                            <th>Token</th>
                            <th></th>
                          </tr>
                        </thead>
                        <tbody>
                          {runs.length === 0 ? (
                            <tr>
                              <td className="tableEmpty" colSpan={5}>
                                {project
                                  ? "暂无分析运行"
                                  : "选择项目后查看运行历史"}
                              </td>
                            </tr>
                          ) : (
                            runs.map((row) => {
                              const usage = describeRunUsage(row);
                              return (
                                <tr key={row.id}>
                                  <td>
                                    <code title={row.id}>
                                      {shortIdentifier(row.id)}
                                    </code>
                                    <small>
                                      {new Date(
                                        row.created_at,
                                      ).toLocaleString()}
                                    </small>
                                    {retryLineage(row) && (
                                      <small
                                        className="retryLineage"
                                        title={row.retried_from || ""}
                                      >
                                        {retryLineage(row)}
                                      </small>
                                    )}
                                  </td>
                                  <td>
                                    <span className={`badge ${row.status}`}>
                                      {row.status}
                                    </span>
                                  </td>
                                  <td>
                                    {runSnapshotDocuments(row).length ? (
                                      <div className="runSnapshotCompact">
                                        {runSnapshotDocuments(row).map(
                                          (input) => {
                                            const labels =
                                              snapshotDocumentLabels(input);
                                            return (
                                              <span
                                                key={`${input.document_id}:${input.ordinal ?? 0}`}
                                              >
                                                <b>{labels.identity}</b>
                                                <small>{labels.context}</small>
                                                <code
                                                  title={input.content_sha256}
                                                >
                                                  sha256 {labels.hash}
                                                </code>
                                              </span>
                                            );
                                          },
                                        )}
                                      </div>
                                    ) : (
                                      <span className="snapshotUnknown">
                                        <b>{runInputState(row)}</b>
                                        <small>不可同输入重试</small>
                                      </span>
                                    )}
                                  </td>
                                  <td>
                                    <b>{usage.tokens}</b>
                                    {usage.detail && (
                                      <small>{usage.detail}</small>
                                    )}
                                  </td>
                                  <td>
                                    <button
                                      disabled={action.startsWith("restore:")}
                                      onClick={() =>
                                        void restoreSelectedRun(row)
                                      }
                                    >
                                      {action === `restore:${row.id}`
                                        ? "恢复中…"
                                        : "恢复"}
                                    </button>
                                  </td>
                                </tr>
                              );
                            })
                          )}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              )}
              {activeView === "diff" && (
                <div className="diffPanel">
                  <div className="sectionHead">
                    <div>
                      <p className="eyebrow">VERSION DIFF</p>
                      <h3>版本内容差异</h3>
                    </div>
                    <small>本地行级比较，不调用模型、不消耗 Token</small>
                  </div>
                  <div className="diffControls">
                    <select
                      value={diffFrom}
                      onChange={(event) => selectDiffFrom(event.target.value)}
                    >
                      <option value="">选择旧版本</option>
                      {docs.map((row) => (
                        <option key={row.id} value={row.id}>
                          {row.name} · v{row.version}
                        </option>
                      ))}
                    </select>
                    <span>→</span>
                    <select
                      value={diffTo}
                      onChange={(event) => {
                        setDiffTo(event.target.value);
                        setDocumentDiff(null);
                      }}
                    >
                      <option value="">选择新版本</option>
                      {docs
                        .filter((row) => {
                          const source = docs.find(
                            (item) => item.id === diffFrom,
                          );
                          return (
                            source &&
                            row.id !== source.id &&
                            row.name.toLocaleLowerCase() ===
                              source.name.toLocaleLowerCase()
                          );
                        })
                        .map((row) => (
                          <option key={row.id} value={row.id}>
                            {row.name} · v{row.version}
                          </option>
                        ))}
                    </select>
                    <button
                      disabled={!diffFrom || !diffTo || diffBusy}
                      onClick={compareVersions}
                    >
                      {diffBusy ? "比较中…" : "查看差异"}
                    </button>
                  </div>
                  {documentDiff && (
                    <div className="diffResult">
                      <div className="diffSummary">
                        <b>
                          v{documentDiff.from_document.version} → v
                          {documentDiff.to_document.version}
                        </b>
                        <span className="diffAdded">
                          +{documentDiff.summary.added_lines}
                        </span>
                        <span className="diffRemoved">
                          −{documentDiff.summary.removed_lines}
                        </span>
                        <span>
                          {documentDiff.summary.changed_hunks} 个变更区块
                        </span>
                        <small>
                          比较 {documentDiff.summary.compared_old_lines}/
                          {documentDiff.summary.old_total_lines} →{" "}
                          {documentDiff.summary.compared_new_lines}/
                          {documentDiff.summary.new_total_lines} 行
                        </small>
                      </div>
                      {documentDiff.warnings.map((warning, index) => (
                        <p className="visualWarning" key={index}>
                          {warning}
                        </p>
                      ))}
                      {documentDiff.hunks.length === 0 ? (
                        <div className="visualEmpty">
                          两个版本的文本内容相同。
                        </div>
                      ) : (
                        <div className="diffHunks">
                          {documentDiff.hunks.map((hunk, index) => (
                            <div className="diffHunk" key={index}>
                              <div className="diffHeader">
                                @@ -{hunk.old_start},{hunk.old_lines} +
                                {hunk.new_start},{hunk.new_lines} @@
                              </div>
                              {hunk.lines.map((line, lineIndex) => (
                                <div
                                  className={`diffLine ${line.type}`}
                                  key={lineIndex}
                                >
                                  <code>{line.old_line ?? ""}</code>
                                  <code>{line.new_line ?? ""}</code>
                                  <b>
                                    {line.type === "added"
                                      ? "+"
                                      : line.type === "removed"
                                        ? "−"
                                        : " "}
                                  </b>
                                  <pre>{line.content || " "}</pre>
                                </div>
                              ))}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </section>
          )}

          {activeView === "check" && (
            <section className="workspace workspaceView">
              <div className="sectionHead">
                <div>
                  <p className="eyebrow">QUICK TEXT</p>
                  <h2>快速自然文本实验台</h2>
                </div>
                <button
                  className="primary startValidation"
                  disabled={busy}
                  onClick={custom}
                >
                  <span aria-hidden="true">✦</span>
                  {action === "custom"
                    ? "校验中…"
                    : "开始校验（可能消耗 Token）"}
                </button>
              </div>
              <div className="quickMode">
                <button
                  className={quickMode === "body" ? "active" : ""}
                  onClick={() => setQuickMode("body")}
                >
                  只有故事正文
                </button>
                <button
                  className={quickMode === "advanced" ? "active" : ""}
                  onClick={() => setQuickMode("advanced")}
                >
                  高级：设定 + 章节
                </button>
                <select
                  aria-label="快速文本类型"
                  disabled={quickMode === "advanced"}
                  value={quickMode === "advanced" ? "chapter" : quickRole}
                  onChange={(event) =>
                    setQuickRole(event.target.value as DocumentRole)
                  }
                >
                  {documentRoles.map(([value, label]) => (
                    <option key={value} value={value}>
                      {label}
                    </option>
                  ))}
                </select>
                <input
                  aria-label="快速文本故事作用域"
                  maxLength={80}
                  value={quickScope}
                  onChange={(event) => setQuickScope(event.target.value)}
                  placeholder="故事作用域，默认 global"
                />
              </div>
              {quickMode === "body" ? (
                <div className="editors single">
                  <label>
                    <span>需要审查的故事正文 / chapter.md</span>
                    <textarea
                      value={chapter}
                      onChange={(event) => setChapter(event.target.value)}
                    />
                  </label>
                </div>
              ) : (
                <div className="editors">
                  <label>
                    <span>可选权威设定 / world.md</span>
                    <textarea
                      value={world}
                      onChange={(event) => setWorld(event.target.value)}
                    />
                  </label>
                  <label>
                    <span>待审章节 / chapter.md</span>
                    <textarea
                      value={chapter}
                      onChange={(event) => setChapter(event.target.value)}
                    />
                  </label>
                </div>
              )}
              <p className="contextHint">
                高级模式会把设定标为“权威世界观”、章节标为“故事正文”；只有正文时无需提前准备世界观。
              </p>
              <div className="visualDeck">
                <article>
                  <span className="navGlyph" aria-hidden="true">
                    RG
                  </span>
                  <div>
                    <b>角色关系预览</b>
                    <small>
                      {graph
                        ? `${graph.nodes.length} 个节点已就绪`
                        : "按需生成，不自动请求"}
                    </small>
                  </div>
                  <button
                    disabled={
                      !run ||
                      runInfo?.status !== "completed" ||
                      visualLoading !== null
                    }
                    onClick={() =>
                      graph
                        ? setActiveView("visual")
                        : void loadVisualization("graph")
                    }
                  >
                    {visualLoading === "graph"
                      ? "生成中…"
                      : graph
                        ? "查看"
                        : "生成"}
                  </button>
                </article>
                <article>
                  <span className="navGlyph" aria-hidden="true">
                    TL
                  </span>
                  <div>
                    <b>故事时间线</b>
                    <small>
                      {timeline
                        ? `${timeline.groups.reduce((count, group) => count + group.entries.length, 0) + timeline.unscheduled.length} 个事件已就绪`
                        : "按需生成，不自动请求"}
                    </small>
                  </div>
                  <button
                    disabled={
                      !run ||
                      runInfo?.status !== "completed" ||
                      visualLoading !== null
                    }
                    onClick={() =>
                      timeline
                        ? setActiveView("visual")
                        : void loadVisualization("timeline")
                    }
                  >
                    {visualLoading === "timeline"
                      ? "生成中…"
                      : timeline
                        ? "查看"
                        : "生成"}
                  </button>
                </article>
              </div>
            </section>
          )}

          {activeView === "audit" && (
            <>
              <section className="status" aria-live="polite">
                <div>
                  <small>PROJECT</small>
                  <code>{project || "not-selected"}</code>
                </div>
                <div>
                  <small>RUN</small>
                  <code>{run || "not-started"}</code>
                </div>
                <div className="meter">
                  <i style={{ width: `${progress}%` }} />
                </div>
                <strong>
                  {progress}% · {message}
                </strong>
                <div className="runActions">
                  {busy && run && <button onClick={cancel}>取消</button>}
                  {runInfo &&
                    ["failed", "cancelled"].includes(runInfo.status) && (
                      <button
                        disabled={!retryState(runInfo).allowed}
                        title={
                          retryState(runInfo).allowed
                            ? "使用该运行的冻结输入，而非当前文档"
                            : retryState(runInfo).label
                        }
                        onClick={retry}
                      >
                        {retryState(runInfo).label}
                      </button>
                    )}
                </div>
              </section>
              {runInfo && (
                <section
                  className={`runSnapshot ${runSnapshotDocuments(runInfo).length ? "available" : "unknown"}`}
                  aria-live="polite"
                >
                  <div className="sectionHead">
                    <div>
                      <p className="eyebrow">RUN INPUT SNAPSHOT</p>
                      <h2>本次运行使用的输入快照</h2>
                    </div>
                    <div className="snapshotState">
                      <b>{runInputState(runInfo)}</b>
                      {retryLineage(runInfo) && (
                        <small title={runInfo.retried_from || ""}>
                          {retryLineage(runInfo)}
                        </small>
                      )}
                    </div>
                  </div>
                  {runSnapshotDocuments(runInfo).length ? (
                    <div className="snapshotCards">
                      {runSnapshotDocuments(runInfo).map((input) => {
                        const labels = snapshotDocumentLabels(input);
                        return (
                          <article
                            key={`${input.document_id}:${input.ordinal ?? 0}`}
                          >
                            <b>{labels.identity}</b>
                            <span>{labels.context}</span>
                            <code title={input.content_sha256}>
                              sha256 {labels.hash}
                            </code>
                            {input.char_count !== undefined && (
                              <small>{input.char_count} 字符</small>
                            )}
                          </article>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="snapshotUnknown">
                      <b>输入未知 / 不可同输入重试</b>
                      <p>
                        此运行没有服务端冻结快照元数据。页面不会用当前项目文档冒充它当时的输入。
                      </p>
                    </div>
                  )}
                </section>
              )}
              {runInfo?.status === "completed" && (
                <section
                  className={`reviewCoverage ${modelStatus.coverage}`}
                  aria-live="polite"
                >
                  <strong>{modelStatus.label}</strong>
                  <span>{modelStatus.detail}</span>
                  {modelStatus.counts && <small>{modelStatus.counts}</small>}
                </section>
              )}
              <section className="summary">
                <div>
                  <small>当前文档</small>
                  <b>{docs.filter((row) => row.active).length}</b>
                </div>
                <div>
                  <small>抽取记录</small>
                  <b>{records.length}</b>
                </div>
                <div>
                  <small>确认问题</small>
                  <b>{issues.length}</b>
                </div>
                <div>
                  <small>待澄清/开放问题</small>
                  <b>{clarifications.length}</b>
                </div>
                <div>
                  <small>模型 Token</small>
                  <b>{selectedRunUsage.tokens}</b>
                  <small>
                    {selectedRunUsage.detail && (
                      <>
                        {selectedRunUsage.detail}
                        <br />
                      </>
                    )}
                    {selectedRunUsage.cost}
                  </small>
                </div>
              </section>
            </>
          )}
          {activeView === "audit" && (
            <section className="records">
              <details>
                <summary>查看系统实际抽取记录（{records.length} 条）</summary>
                {records.length === 0 ? (
                  <div className="inlineEmpty">
                    完成或恢复一次分析后，这里会显示带原文位置的结构化记录。
                  </div>
                ) : (
                  <div className="recordGrid">
                    {records.map((row) => (
                      <div className="recordCard" key={row.id}>
                        <span>{recordKindName(row.kind)}</span>
                        <p>{summarizeRecord(row)}</p>
                        <small>
                          {row.evidence.document_name}:{row.evidence.line_start}
                          {row.evidence.line_end !== row.evidence.line_start
                            ? `–${row.evidence.line_end}`
                            : ""}
                        </small>
                        <blockquote>{row.evidence.text}</blockquote>
                        <details className="developerFields">
                          <summary>查看开发者字段</summary>
                          {recordDeveloperFields(row).map(([key, value]) => (
                            <code key={key}>
                              {key} = {value}
                            </code>
                          ))}
                        </details>
                      </div>
                    ))}
                  </div>
                )}
                {warnings.length > 0 && (
                  <div className="warnings">
                    <b>抽取提示</b>
                    {warnings.map((warning, index) => (
                      <p key={index}>{warning}</p>
                    ))}
                  </div>
                )}
              </details>
            </section>
          )}
          {activeView === "audit" && (
            <section className="records">
              <details>
                <summary>分析诊断摘要</summary>
                <div className="recordGrid diagnosticGrid">
                  <div>
                    <span>模型语义覆盖</span>
                    <code>{modelStatus.label}</code>
                    <small>
                      {modelStatus.counts}
                      <br />
                      {modelStatus.detail}
                      <br />
                      逻辑分块调用计数，不含内部 HTTP 重试次数。
                    </small>
                  </div>
                  <RepairDiagnostic status={repairStatus} />
                  <ReviewAgentDiagnostic status={reviewAgentStatus} />
                  {issueReviewDiagnostic && (
                    <IssueEvidenceReviewDiagnostic
                      status={issueReviewDiagnostic}
                    />
                  )}
                  <div>
                    <span>文档分块</span>
                    <code>
                      {diagnostics.chunking?.total_chunks || 0} 个分块
                    </code>
                    <small>
                      {diagnostics.chunking?.documents
                        ?.map(
                          (row) => `${row.document_name}: ${row.chunk_count}`,
                        )
                        .join(" · ") || "暂无诊断"}
                    </small>
                  </div>
                  <div>
                    <span>角色别名</span>
                    <code>
                      {diagnostics.aliases?.declaration_count || 0} 条声明 ·{" "}
                      {diagnostics.aliases?.trace_count || 0} 次归一化
                    </code>
                    <small>只接受显式又名/简称/化名/代号声明</small>
                  </div>
                  <div>
                    <span>候选证据检索</span>
                    <code>
                      {diagnostics.retrieval?.candidate_count || 0} 个候选 ·{" "}
                      {diagnostics.retrieval?.consumed_count || 0}{" "}
                      个被检查器消费
                    </code>
                    <small>
                      {diagnostics.retrieval?.boundary ||
                        "本地稳定哈希与实体图候选"}
                    </small>
                  </div>
                  <div>
                    <span>处理耗时</span>
                    <code>
                      总计 {diagnostics.timings?.total_ms?.toFixed(1) || "0.0"}{" "}
                      ms · 首事件{" "}
                      {diagnostics.timings?.first_progress_ms?.toFixed(1) ||
                        "0.0"}{" "}
                      ms
                    </code>
                    <small>
                      分块 {diagnostics.timings?.chunk_ms || 0} · 抽取{" "}
                      {diagnostics.timings?.extract_ms || 0} · 索引{" "}
                      {diagnostics.timings?.index_ms || 0} · 检查{" "}
                      {diagnostics.timings?.check_ms || 0} · 报告{" "}
                      {diagnostics.timings?.report_ms || 0}
                    </small>
                  </div>
                </div>
              </details>
            </section>
          )}
          {activeView === "visual" && (
            <section className="visualization workspaceView">
              <div className="sectionHead">
                <div>
                  <p className="eyebrow">NARRATIVE VIEW</p>
                  <h2>按需查看关系图与时间线</h2>
                </div>
                <div className="visualTabs">
                  <button
                    className={visualTab === "graph" && graph ? "active" : ""}
                    disabled={
                      !run ||
                      runInfo?.status !== "completed" ||
                      visualLoading !== null
                    }
                    onClick={() => void loadVisualization("graph")}
                  >
                    {visualLoading === "graph"
                      ? "加载关系图…"
                      : graph
                        ? "查看关系图"
                        : "生成关系图"}
                  </button>
                  <button
                    className={
                      visualTab === "timeline" && timeline ? "active" : ""
                    }
                    disabled={
                      !run ||
                      runInfo?.status !== "completed" ||
                      visualLoading !== null
                    }
                    onClick={() => void loadVisualization("timeline")}
                  >
                    {visualLoading === "timeline"
                      ? "加载时间线…"
                      : timeline
                        ? "查看时间线"
                        : "生成时间线"}
                  </button>
                </div>
              </div>
              {visualError && (
                <div className="visualWarning">{visualError}</div>
              )}
              {visualTab === "graph" && graph ? (
                <Suspense
                  fallback={
                    <div className="visualEmpty">正在加载关系图组件…</div>
                  }
                >
                  <RelationGraph data={graph} focusedIssueId={focusedIssue} />
                </Suspense>
              ) : visualTab === "timeline" && timeline ? (
                <NarrativeTimeline
                  data={timeline}
                  focusedIssueId={focusedIssue}
                />
              ) : (
                <div className="visualEmpty">
                  关系图和时间线不会自动生成。需要时点击上方按钮，本页才会请求对应数据。
                </div>
              )}
            </section>
          )}
          {activeView === "report" && (
            <>
              <section className="clarifications workspaceView">
                <div className="sectionHead">
                  <div>
                    <p className="eyebrow">REVIEW QUESTIONS</p>
                    <h2>
                      需要作者确认 <em>{clarifications.length}</em>
                    </h2>
                  </div>
                </div>
                {clarifications.length === 0 && !busy && (
                  <div className="empty">
                    {runInfo?.status === "completed"
                      ? `当前未发现需要作者补充的疑问。${modelStatus.emptyCaveat ? ` ${modelStatus.emptyCaveat}` : ""}`
                      : "完成或恢复一次分析后，这里会单独显示待澄清项。"}
                  </div>
                )}
                {clarifications.map((item, index) => (
                  <article key={item.id}>
                    <div className="rank">
                      {String(index + 1).padStart(2, "0")}
                    </div>
                    <div>
                      <p className="tag">
                        {clarificationKindName(item.kind)}
                        {item.category
                          ? ` · ${clarificationCategoryNames[item.category] || item.category}`
                          : ""}
                      </p>
                      <h3>{item.text}</h3>
                      <p>
                        {modalityNames[item.modality] || item.modality} ·{" "}
                        {sourceNames[item.source_scope] || item.source_scope} ·{" "}
                        {certaintyNames[item.certainty] || item.certainty}
                      </p>
                      <div className="evidence">
                        <blockquote>
                          <b>
                            {item.evidence.document_name}:
                            {item.evidence.line_start}
                            {item.evidence.line_end !== item.evidence.line_start
                              ? `–${item.evidence.line_end}`
                              : ""}
                          </b>
                          {item.evidence.text}
                        </blockquote>
                      </div>
                    </div>
                  </article>
                ))}
              </section>
              <section className="issues">
                <div className="sectionHead">
                  <div>
                    <p className="eyebrow">EVIDENCE REPORT</p>
                    <h2>
                      已确认的一致性问题 <em>{visibleIssues.length}</em>
                    </h2>
                  </div>
                  <select
                    value={filter}
                    onChange={(event) => setFilter(event.target.value)}
                  >
                    <option value="all">全部类别</option>
                    {Object.entries(categoryNames).map(([key, value]) => (
                      <option key={key} value={key}>
                        {value}
                      </option>
                    ))}
                  </select>
                </div>
                {issues.length === 0 && !busy && (
                  <div className="empty">
                    {runInfo?.status === "completed"
                      ? `当前未发现有证据支持的一致性问题。${modelStatus.emptyCaveat ? ` ${modelStatus.emptyCaveat}` : ""}`
                      : "选择项目运行，或从历史任务恢复报告。"}
                  </div>
                )}
                {issues.length > 0 && visibleIssues.length === 0 && (
                  <div className="empty">
                    当前筛选类别没有问题，请切换到“全部类别”。
                  </div>
                )}
                {visibleIssues.map((issue, index) => {
                  const evidenceReview = describeIssueEvidenceReview(
                    issue.metadata,
                    documentNames,
                  );
                  return (
                    <article
                      key={issue.id}
                      className={
                        focusedIssue === issue.id ? "focusedIssue" : ""
                      }
                      onClick={() => setFocusedIssue(issue.id)}
                    >
                      <div className="rank">
                        {String(index + 1).padStart(2, "0")}
                      </div>
                      <div>
                        <p className="tag">
                          {categoryNames[issue.category] || issue.category} ·{" "}
                          {issue.severity} ·{" "}
                          {(issue.confidence * 100).toFixed(0)}%
                        </p>
                        <h3>{issue.title}</h3>
                        <p>{issue.explanation}</p>
                        <div className="evidence">
                          {issue.evidence.map((evidence, index) => (
                            <blockquote key={index}>
                              <b>
                                {evidence.document_name}:{evidence.line_start}
                              </b>
                              {evidence.text}
                            </blockquote>
                          ))}
                        </div>
                        <p className="suggestion">建议：{issue.suggestion}</p>
                        {evidenceReview && (
                          <IssueEvidenceReview review={evidenceReview} />
                        )}
                        <div className="feedbackState">
                          当前反馈：
                          {feedbacks[issue.id]
                            ? feedbackNames[feedbacks[issue.id]!.label] ||
                              feedbacks[issue.id]!.label
                            : "未反馈"}
                          {feedbackPending[issue.id] ? " · 提交中…" : ""}
                        </div>
                        <input
                          className="note"
                          value={notes[issue.id] || ""}
                          disabled={feedbackPending[issue.id]}
                          onClick={(event) => event.stopPropagation()}
                          onChange={(event) =>
                            setNotes((current) => ({
                              ...current,
                              [issue.id]: event.target.value,
                            }))
                          }
                          placeholder="可选备注（会进入审计历史）"
                        />
                        <div className="feedback">
                          {Object.entries(feedbackNames).map(
                            ([label, title]) => (
                              <button
                                key={label}
                                disabled={
                                  feedbackPending[issue.id] ||
                                  (feedbacks[issue.id]?.label === label &&
                                    (feedbacks[issue.id]?.comment || "") ===
                                      (notes[issue.id] || ""))
                                }
                                onClick={(event) => {
                                  event.stopPropagation();
                                  void submitFeedback(issue.id, label);
                                }}
                              >
                                {title}
                              </button>
                            ),
                          )}
                        </div>
                      </div>
                    </article>
                  );
                })}
              </section>
            </>
          )}
        </div>

        <aside className="resultRail" aria-label="运行与冲突报告">
          <div className="resultRailHead">
            <div>
              <p className="eyebrow">LIVE REVIEW</p>
              <h2>冲突审查报告</h2>
            </div>
            <strong>{issues.length}</strong>
          </div>
          <p className={`railCoverage ${modelStatus.coverage}`}>
            <span>{modelStatus.label}</span>
            <small>
              {runInfo?.status === "completed"
                ? "结果可追溯"
                : "等待完成一次分析"}
            </small>
          </p>
          <section
            className={`railLive ${busy ? "running" : ""}`}
            aria-live="polite"
          >
            <div className="railLiveTop">
              <span>
                <i aria-hidden="true" />
                {busy
                  ? "正在校验"
                  : runInfo
                    ? `运行 ${runInfo.status}`
                    : "尚未开始"}
              </span>
              <b>{progress}%</b>
            </div>
            <div className="meter">
              <i style={{ width: `${progress}%` }} />
            </div>
            <p>{message}</p>
            <div className="railIds">
              <span>
                PROJECT <code>{shortIdentifier(project) || "—"}</code>
              </span>
              <span>
                RUN <code>{shortIdentifier(run) || "—"}</code>
              </span>
            </div>
            <div className="runActions">
              {busy && run && <button onClick={cancel}>取消任务</button>}
              {runInfo && ["failed", "cancelled"].includes(runInfo.status) && (
                <button
                  disabled={!retryState(runInfo).allowed}
                  title={
                    retryState(runInfo).allowed
                      ? "使用该运行的冻结输入，而非当前文档"
                      : retryState(runInfo).label
                  }
                  onClick={retry}
                >
                  {retryState(runInfo).label}
                </button>
              )}
              <button onClick={() => setActiveView("audit")}>运行审计</button>
            </div>
          </section>
          <section className="railSummary" aria-label="运行摘要">
            <div>
              <small>确认问题</small>
              <b>{issues.length}</b>
            </div>
            <div>
              <small>待澄清</small>
              <b>{clarifications.length}</b>
            </div>
            <div>
              <small>抽取记录</small>
              <b>{records.length}</b>
            </div>
            <div>
              <small>当前文档</small>
              <b>{docs.filter((row) => row.active).length}</b>
            </div>
            <div className="railToken">
              <small>模型 Token</small>
              <b>{selectedRunUsage.tokens}</b>
              <span>{selectedRunUsage.cost}</span>
            </div>
          </section>
          <div className="railPreviewScroll">
            <section className="issuePreview">
              <div className="railSectionHead">
                <h3>问题预览</h3>
                <button onClick={() => setActiveView("report")}>
                  全部 {issues.length}
                </button>
              </div>
              {visibleIssues.length === 0 ? (
                <div className="railEmpty">
                  {runInfo?.status === "completed"
                    ? `当前未发现有证据支持的一致性问题。${modelStatus.emptyCaveat ? ` ${modelStatus.emptyCaveat}` : ""}`
                    : "完成或恢复一次分析后，这里会显示真实问题。"}
                </div>
              ) : (
                visibleIssues.slice(0, 4).map((issue) => (
                  <button
                    className={`issuePreviewCard ${focusedIssue === issue.id ? "focused" : ""}`}
                    key={issue.id}
                    onClick={() => {
                      setFocusedIssue(issue.id);
                      setActiveView("report");
                    }}
                  >
                    <span>
                      {categoryNames[issue.category] || issue.category} ·{" "}
                      {issue.severity} · {(issue.confidence * 100).toFixed(0)}%
                    </span>
                    <b>{issue.title}</b>
                    <p>{issue.explanation}</p>
                    <small>
                      {issue.evidence[0]
                        ? `${issue.evidence[0].document_name}:${issue.evidence[0].line_start}`
                        : "证据待核对"}
                    </small>
                  </button>
                ))
              )}
            </section>
            {clarifications.length > 0 && (
              <section className="questionPreview">
                <div className="railSectionHead">
                  <h3>需要作者确认</h3>
                  <span>{clarifications.length}</span>
                </div>
                {clarifications.slice(0, 2).map((item) => (
                  <button key={item.id} onClick={() => setActiveView("report")}>
                    <span>{clarificationKindName(item.kind)}</span>
                    <b>{item.text}</b>
                  </button>
                ))}
              </section>
            )}
          </div>
          <button
            className="railReportButton"
            onClick={() => setActiveView("report")}
          >
            查看完整报告 <span>{issues.length + clarifications.length}</span>
          </button>
        </aside>
      </main>
      <footer>
        原创演示文本 · 本地项目/版本/运行历史 · Provider 异常时安全降级
      </footer>
    </>
  );
}
