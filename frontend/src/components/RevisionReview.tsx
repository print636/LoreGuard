import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  apiJson,
  apiJsonIdempotent,
  apiUrl,
  createBoundedSessionProbe,
} from "../api/client";
import { documentContext, documentRoles, type DocumentRole } from "../documentContext";
import {
  comparisonBelongsToRevision,
  comparisonOutcomes,
  comparisonPath,
  comparisonRetryDelay,
  createMutationGuard,
  mergeRevisionRouteState,
  pendingVisualizationKinds,
  recheckPath,
  revisionDelta,
  revisionDocumentForIssue,
  revisionRouteStateFromSearch,
  revisionSearch,
  revisionSteps,
  type ComparisonOutcome,
  type RevisionDocument,
  type RevisionRouteState,
  validateRevisionUploadSelection,
} from "../revisionWorkflow";
import { runSnapshotDocuments, shortIdentifier, type RunInputSnapshot } from "../runSnapshot";
import { supportedUploadAccept, supportedUploadLabel } from "../uploadFormats";

type Evidence = {
  document_id: string;
  document_name: string;
  line_start: number;
  line_end: number;
  text: string;
};

type RevisionIssue = {
  id: string;
  category: string;
  severity: string;
  confidence: number;
  title: string;
  explanation: string;
  suggestion: string;
  evidence: Evidence[];
};

type RevisionRun = {
  id: string;
  project_id: string;
  status: string;
  input_documents?: RunInputSnapshot[];
  input_snapshot_available?: boolean;
};

type ComparisonIssue = RevisionIssue & { metadata?: unknown };

type ComparisonItem = {
  id?: string;
  outcome: Exclude<ComparisonOutcome, "all">;
  baseline_issue: ComparisonIssue | null;
  target_issue: ComparisonIssue | null;
  match_method?: string | null;
  match_score?: number | null;
  provenance?: unknown;
  baseline_latest_feedback?: {
    id: string;
    label: string;
    comment?: string | null;
    created_at: string;
  } | null;
};

type ComparisonResponse = {
  status: "pending" | "ready" | "failed" | "cancelled";
  project_id?: string;
  baseline_run_id?: string;
  target_run_id?: string;
  summary?: {
    no_longer_detected: number;
    persisting: number;
    new: number;
    unverifiable: number;
    total_baseline: number;
    total_target: number;
    actionable_no_longer_detected?: number;
    baseline_false_positive?: number;
  };
  items?: ComparisonItem[];
  page?: {
    outcome?: string | null;
    limit: number;
    offset: number;
    returned?: number;
    total: number;
    has_more: boolean;
  } | null;
  provenance?: {
    compatibility?: {
      status?: string;
      reasons?: string[];
    };
  } | null;
};

type ComparisonLoadResult =
  | { kind: "ok"; value: ComparisonResponse }
  | { kind: "retry" }
  | { kind: "fatal" }
  | { kind: "aborted" };

type RecheckCreated = RevisionRun & {
  deduplicated?: boolean;
  baseline_run_id: string;
  comparison_id: string;
  comparison_status: string;
};

type RevisionReviewProps = {
  projectId: string;
  projectName: string;
  baselineRun: RevisionRun | null;
  documents: RevisionDocument[];
  baselineIssues: RevisionIssue[];
  routeSearch: string;
  onRouteChange: (search: string, replace?: boolean) => void;
  onDocumentsChanged: () => Promise<void>;
};

const stepLabels: Record<RevisionRouteState["step"], string> = {
  review: "确认待修订问题",
  upload: "上传新版本",
  run: "启动复检",
  compare: "查看复检差异",
};

const outcomeLabels: Record<ComparisonOutcome, string> = {
  all: "全部变化",
  no_longer_detected: "本次未再检出",
  persisting: "仍存在",
  new: "新增",
  unverifiable: "无法确认",
};

const issueCategoryLabels: Record<string, string> = {
  fact_conflict: "事实冲突",
  location_collision: "同刻多地点",
  knowledge_without_acquisition: "知识越权",
  item_ownership: "物品状态",
  world_rule_conflict: "世界规则",
};

const severityLabels: Record<string, string> = {
  high: "严重",
  medium: "中等",
  low: "轻微",
};

function errorCopy(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return "请求没有完成。请检查网络后重试，已上传的文档不会被删除。";
}

function issueForItem(item: ComparisonItem): ComparisonIssue | null {
  return item.target_issue || item.baseline_issue;
}

function itemIdentifier(item: ComparisonItem, index: number): string {
  return item.id || item.target_issue?.id || item.baseline_issue?.id || `comparison-${index}`;
}

function compatibilityCopy(comparison: ComparisonResponse): string {
  const compatibility = comparison.provenance?.compatibility;
  const statusNames: Record<string, string> = {
    comparable: "可比较",
    degraded: "受限",
    unverifiable: "无法确认",
  };
  const reasonNames: Record<string, string> = {
    baseline_not_completed: "基线运行未完成",
    target_not_completed: "复检运行未完成",
    baseline_snapshot_incomplete: "基线冻结输入不完整",
    target_snapshot_incomplete: "复检冻结输入不完整",
    baseline_diagnostics_missing: "基线诊断记录缺失",
    target_diagnostics_missing: "复检诊断记录缺失",
    baseline_model_diagnostics_incomplete: "基线模型诊断不完整",
    target_model_diagnostics_incomplete: "复检模型诊断不完整",
    baseline_model_partial_fallback: "基线模型覆盖不完整",
    target_model_partial_fallback: "复检模型覆盖不完整",
    baseline_investigator_incomplete: "基线证据调查未完成",
    target_investigator_incomplete: "复检证据调查未完成",
    runtime_provenance_missing: "运行环境记录缺失",
    runtime_capabilities_changed: "两次运行启用的能力不同",
    runtime_chat_provider_changed: "两次运行使用的模型服务不同",
    runtime_rag_changed: "两次运行使用的检索配置不同",
    documents_removed: "复检输入删除了文档",
    document_context_changed: "文档类型或故事作用域已变化",
    issue_limit_exceeded: "问题数量超过安全比较上限",
    ambiguous_identity: "问题身份存在歧义",
    comparison_internal_error: "比较过程未完整完成",
  };
  const status = compatibility?.status
    ? statusNames[compatibility.status] || "受限"
    : "状态未知";
  const reasons = (compatibility?.reasons || []).map(
    (reason) => reasonNames[reason] || "存在未识别的不可比条件，请核对两次运行配置",
  );
  return reasons.length ? `${status} · ${Array.from(new Set(reasons)).join("、")}` : status;
}

function feedbackCopy(item: ComparisonItem): string | null {
  const feedback = item.baseline_latest_feedback;
  if (!feedback) return null;
  if (feedback.label === "resolved") return "历史人工反馈：已标记处理；机器复检结论仍独立计算。";
  if (feedback.label === "false_positive") return "历史人工反馈：误报；不计入可行动的修订结果。";
  if (feedback.label === "accepted") return "历史人工反馈：已接受该问题。";
  return "存在一条历史人工反馈。";
}

function reportHref(projectId: string, runId: string, issueId?: string | null): string {
  const params = new URLSearchParams({ category: "all", status: "all" });
  if (issueId) params.set("issue", issueId);
  return `/app/projects/${encodeURIComponent(projectId)}/runs/${encodeURIComponent(runId)}/report?${params.toString()}`;
}

function matchDescription(item: ComparisonItem): string {
  const labels: Record<string, string> = {
    rule_identity: "依据规则字段确认是同一问题",
    evidence_signature: "依据原文证据确认是同一问题",
  };
  const method = item.match_method
    ? labels[item.match_method] || "系统已核对问题对应关系"
    : "未能确认问题对应关系";
  const reason =
    item.provenance && typeof item.provenance === "object" &&
    "reason" in item.provenance && typeof item.provenance.reason === "string"
      ? item.provenance.reason
      : null;
  const reasonNames: Record<string, string> = {
    ambiguous_identity: "问题身份映射存在歧义",
    runs_not_comparable: "两次运行不可直接比较",
    no_matching_issue_detected: "未找到可一一对应的复检问题",
    no_matching_baseline_issue: "未找到可一一对应的基线问题",
    comparison_internal_error: "比较过程未完整完成",
  };
  return reason ? `${method} · ${reasonNames[reason] || "证据不足，无法确认"}` : method;
}

export default function RevisionReview({
  projectId,
  projectName,
  baselineRun,
  documents,
  baselineIssues,
  routeSearch,
  onRouteChange,
  onDocumentsChanged,
}: RevisionReviewProps) {
  const route = useMemo(() => revisionRouteStateFromSearch(routeSearch), [routeSearch]);
  const baselineDocuments = baselineRun ? runSnapshotDocuments(baselineRun) : [];
  const delta = useMemo(
    () => revisionDelta(baselineDocuments, documents),
    [baselineDocuments, documents],
  );
  const selectedIssue = baselineIssues.find((issue) => issue.id === route.issueId) || null;
  const activeDocuments = documents.filter((document) => document.active);
  const [file, setFile] = useState<File | null>(null);
  const [uploadMode, setUploadMode] = useState<"replace" | "related">("replace");
  const [role, setRole] = useState<DocumentRole>("chapter");
  const [scope, setScope] = useState("global");
  const [uploading, setUploading] = useState(false);
  const [starting, setStarting] = useState(false);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [comparisonLoading, setComparisonLoading] = useState(false);
  const [comparisonRefresh, setComparisonRefresh] = useState(0);
  const [recheckProgress, setRecheckProgress] = useState(0);
  const [recheckMessage, setRecheckMessage] = useState("等待复检运行状态");
  const [streamInterrupted, setStreamInterrupted] = useState(false);
  const [notice, setNotice] = useState("");
  const [problem, setProblem] = useState("");
  const [visualState, setVisualState] = useState<"idle" | "generating" | "ready" | "failed">("idle");
  const fileInput = useRef<HTMLInputElement | null>(null);
  const fileContext = `${projectId}:${baselineRun?.id || ""}`;
  const previousFileContext = useRef(fileContext);
  const generatedVisuals = useRef(new Set<string>());
  const recheckGuard = useRef(createMutationGuard());
  const uploadGuard = useRef(createMutationGuard());
  const comparisonRequest = useRef<{
    epoch: number;
    controller: AbortController | null;
  }>({ epoch: 0, controller: null });
  const safeComparison =
    comparison && route.recheckRunId && baselineRun?.id &&
    comparisonBelongsToRevision(comparison, {
      projectId,
      baselineRunId: baselineRun.id,
      targetRunId: route.recheckRunId,
    })
      ? comparison
      : null;

  function updateRoute(next: Partial<RevisionRouteState>, replace = false) {
    onRouteChange(revisionSearch(mergeRevisionRouteState(route, next)), replace);
  }

  function requestComparisonRefresh() {
    setComparisonRefresh((current) => current + 1);
  }

  useEffect(() => {
    comparisonRequest.current.epoch += 1;
    comparisonRequest.current.controller?.abort();
    comparisonRequest.current.controller = null;
    setComparison(null);
    setComparisonLoading(false);
    setRecheckProgress(0);
    setRecheckMessage("等待复检运行状态");
    setStreamInterrupted(false);
    setVisualState("idle");
    return () => {
      comparisonRequest.current.epoch += 1;
      comparisonRequest.current.controller?.abort();
      comparisonRequest.current.controller = null;
    };
  }, [projectId, baselineRun?.id, route.recheckRunId]);

  useEffect(() => {
    const contextChanged = previousFileContext.current !== fileContext;
    if (route.step !== "upload" || contextChanged) {
      setFile(null);
      if (fileInput.current) fileInput.current.value = "";
    }
    previousFileContext.current = fileContext;
  }, [fileContext, route.step]);

  useEffect(() => {
    if (!route.issueId) return;
    requestAnimationFrame(() => {
      document.getElementById(`revision-issue-${route.issueId}`)?.scrollIntoView({
        block: "nearest",
      });
    });
  }, [route.issueId, baselineIssues]);

  useEffect(() => {
    if (route.documentId && activeDocuments.some((document) => document.id === route.documentId)) {
      const document = activeDocuments.find((row) => row.id === route.documentId)!;
      setRole((document.document_role || "chapter") as DocumentRole);
      setScope(document.story_scope || "global");
      return;
    }
    const inferred = revisionDocumentForIssue(
      selectedIssue?.evidence[0]?.document_name,
      documents,
    );
    if (inferred && route.step === "upload") {
      updateRoute({ documentId: inferred }, true);
    }
  }, [route.documentId, route.step, selectedIssue?.id, documents]);

  async function loadComparison(runId: string, quiet = false): Promise<ComparisonLoadResult> {
    if (!baselineRun?.id) {
      setComparison(null);
      setProblem("无法核验复检基线，请返回报告后重新进入修订流程。");
      return { kind: "fatal" };
    }
    const epoch = comparisonRequest.current.epoch + 1;
    comparisonRequest.current.controller?.abort();
    const controller = new AbortController();
    comparisonRequest.current = { epoch, controller };
    if (!quiet) setComparisonLoading(true);
    try {
      const value = await apiJson<ComparisonResponse>(
        comparisonPath(runId, 20, (route.page - 1) * 20, route.outcome),
        { signal: controller.signal },
      );
      if (comparisonRequest.current.epoch !== epoch) return { kind: "aborted" };
      if (!comparisonBelongsToRevision(value, {
        projectId,
        baselineRunId: baselineRun.id,
        targetRunId: runId,
      })) {
        setComparison(null);
        setProblem("服务器返回的比较结果不属于当前项目、基线或复检运行。为避免串用结果，本页已停止展示；请从基线报告重新进入。" );
        return { kind: "fatal" };
      }
      setComparison(value);
      setProblem("");
      if (value.status === "ready" && route.step !== "compare") {
        updateRoute({ step: "compare", recheckRunId: runId }, true);
      }
      return { kind: "ok", value };
    } catch (error) {
      if (controller.signal.aborted || (error instanceof DOMException && error.name === "AbortError")) {
        return { kind: "aborted" };
      }
      if (comparisonRequest.current.epoch !== epoch) return { kind: "aborted" };
      setProblem(`无法读取复检差异：${errorCopy(error)}`);
      if (!(error instanceof ApiError) || error.status === 429 || error.status >= 500) {
        return { kind: "retry" };
      }
      return { kind: "fatal" };
    } finally {
      if (!quiet && comparisonRequest.current.epoch === epoch) setComparisonLoading(false);
    }
  }

  useEffect(() => {
    if (!route.recheckRunId) {
      setComparison(null);
      return;
    }
    let stopped = false;
    let timer: number | undefined;
    let failures = 0;
    let firstRequest = true;
    setComparison(null);
    const poll = async () => {
      const result = await loadComparison(route.recheckRunId!, !firstRequest);
      firstRequest = false;
      if (stopped || result.kind === "aborted" || result.kind === "fatal") return;
      if (result.kind === "retry") {
        failures += 1;
        timer = window.setTimeout(poll, comparisonRetryDelay(failures));
        return;
      }
      failures = 0;
      if (result.value.status === "pending") {
        timer = window.setTimeout(poll, 2_500);
      }
    };
    void poll();
    return () => {
      stopped = true;
      if (timer !== undefined) window.clearTimeout(timer);
      comparisonRequest.current.controller?.abort();
    };
  }, [route.recheckRunId, route.page, route.outcome, comparisonRefresh, projectId, baselineRun?.id]);

  useEffect(() => {
    if (!route.recheckRunId || safeComparison?.status !== "pending") return;
    const runId = route.recheckRunId;
    const source = new EventSource(
      apiUrl(`/api/v1/analysis-runs/${encodeURIComponent(runId)}/events`),
      { withCredentials: true },
    );
    const probeSession = createBoundedSessionProbe();
    source.addEventListener("progress", (event) => {
      try {
        const data = JSON.parse((event as MessageEvent).data) as {
          progress?: number;
          message?: string;
        };
        if (typeof data.progress === "number") setRecheckProgress(data.progress);
        if (typeof data.message === "string" && data.message.trim()) {
          setRecheckMessage(data.message);
        }
        setStreamInterrupted(false);
      } catch {
        setRecheckMessage("收到无法识别的进度事件，正在读取运行状态");
      }
    });
    source.addEventListener("terminal", (event) => {
      try {
        const data = JSON.parse((event as MessageEvent).data) as {
          status?: string;
          error?: string | null;
        };
        setRecheckProgress((current) =>
          data.status === "completed" ? 100 : current,
        );
        setRecheckMessage(
          data.error ||
            (data.status === "completed"
              ? "分析完成，正在生成复检差异"
              : `复检运行已${data.status === "cancelled" ? "取消" : "结束"}`),
        );
      } catch {
        setRecheckMessage("复检运行已经结束，正在读取比较结果");
      }
      source.close();
      requestComparisonRefresh();
    });
    source.onerror = () => {
      setStreamInterrupted(true);
      setRecheckMessage("进度连接暂时中断，浏览器正在自动重连");
      void probeSession().then((status) => {
        if (status === "expired") {
          source.close();
          setRecheckMessage("登录状态已失效，请重新登录后返回这个精确链接");
        }
      });
    };
    return () => source.close();
  }, [route.recheckRunId, safeComparison?.status]);

  useEffect(() => {
    if (
      safeComparison?.status !== "ready" ||
      !route.recheckRunId ||
      (!route.generateGraph && !route.generateTimeline)
    ) return;
    const kinds = pendingVisualizationKinds(
      route.recheckRunId,
      route.generateGraph,
      route.generateTimeline,
      generatedVisuals.current,
    );
    if (!kinds.length) {
      setVisualState("ready");
      return;
    }
    setVisualState("generating");
    const runId = route.recheckRunId;
    void Promise.allSettled(
      kinds.map(async (kind) => {
        await apiJson(`/api/v1/analysis-runs/${encodeURIComponent(runId)}/${kind}`);
        generatedVisuals.current.add(`${runId}:${kind}`);
      }),
    ).then((results) => {
      setVisualState(results.every((result) => result.status === "fulfilled") ? "ready" : "failed");
    });
  }, [safeComparison?.status, route.recheckRunId, route.generateGraph, route.generateTimeline]);

  async function uploadRevision(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) {
      setProblem("请选择要上传的新版本文件。");
      fileInput.current?.focus();
      return;
    }
    const validation = validateRevisionUploadSelection(
      uploadMode,
      file.name,
      route.documentId,
      documents,
    );
    if (!validation.ok) {
      setProblem(validation.message);
      if (validation.field === "document") {
        document.getElementById("revision-document")?.focus();
      } else if (validation.field === "mode") {
        document.getElementById("revision-mode-replace")?.focus();
      } else {
        fileInput.current?.focus();
      }
      return;
    }
    if (!uploadGuard.current.tryBegin()) return;
    try {
      setUploading(true);
      setProblem("");
      const context = documentContext(role, scope);
      const form = new FormData();
      form.append("file", file);
      form.append("document_role", context.document_role);
      form.append("story_scope", context.story_scope);
      if (uploadMode === "replace" && route.documentId) {
        form.append("replace_document_id", route.documentId);
      }
      await apiJson(`/api/v1/projects/${encodeURIComponent(projectId)}/documents`, {
        method: "POST",
        body: form,
      });
      await onDocumentsChanged();
      setFile(null);
      if (fileInput.current) fileInput.current.value = "";
      setNotice("新版本已保存。请核对输入变化后启动复检。");
      updateRoute({ step: "run", recheckRunId: null, outcome: "all", page: 1 });
    } catch (error) {
      setProblem(`上传失败：${errorCopy(error)}`);
    } finally {
      setUploading(false);
      uploadGuard.current.end();
    }
  }

  async function startRecheck() {
    if (!baselineRun || !recheckGuard.current.tryBegin()) return;
    try {
      setStarting(true);
      setProblem("");
      setNotice("正在创建复检运行…");
      setRecheckProgress(0);
      setRecheckMessage("任务已提交，等待执行队列接收");
      const created = await apiJsonIdempotent<RecheckCreated>(
        recheckPath(baselineRun.id),
        { method: "POST" },
      );
      setNotice(
        created.deduplicated
          ? "相同复检请求已经存在，已恢复原运行。"
          : "复检已提交。页面会持续检查结果，可以安全刷新或稍后返回。",
      );
      updateRoute({ step: "run", recheckRunId: created.id, page: 1 });
    } catch (error) {
      setProblem(`无法启动复检：${errorCopy(error)}`);
    } finally {
      setStarting(false);
      recheckGuard.current.end();
    }
  }

  const baselineReady = baselineRun?.status === "completed" && baselineDocuments.length > 0;
  const visibleComparisonItems = safeComparison?.items || [];

  return (
    <section className="revisionReview workspaceView" aria-labelledby="revision-title">
      <header className="revisionHeader">
        <div>
          <h1 id="revision-title">修订与复检</h1>
          <p>
            以运行 <code title={baselineRun?.id}>{shortIdentifier(baselineRun?.id || "")}</code>
            为固定基线，上传修订稿后只比较这两次报告。
          </p>
        </div>
        {baselineRun ? (
          <a href={`/app/projects/${encodeURIComponent(projectId)}/runs/${encodeURIComponent(baselineRun.id)}/report?category=all&status=all`}>
            返回基线报告
          </a>
        ) : (
          <span className="revisionBaselineUnavailable">基线运行不可用</span>
        )}
      </header>

      <nav className="revisionSteps" aria-label="修订与复检步骤">
        {revisionSteps.map((step, index) => (
          <button
            key={step}
            type="button"
            className={route.step === step ? "active" : ""}
            aria-current={route.step === step ? "step" : undefined}
            disabled={step === "compare" && !route.recheckRunId}
            onClick={() => updateRoute({ step })}
          >
            <span>{index + 1}</span>
            {stepLabels[step]}
          </button>
        ))}
      </nav>

      <div className="revisionContext" aria-label="复检上下文">
        <span><small>项目</small><b>{projectName || projectId}</b></span>
        <span><small>基线运行</small><code title={baselineRun?.id}>{shortIdentifier(baselineRun?.id || "")}</code></span>
        <span><small>冻结输入</small><b>{baselineDocuments.length} 份</b></span>
        <span><small>基线问题</small><b>{baselineIssues.length} 项</b></span>
      </div>

      {(notice || problem) && (
        <div className={`revisionNotice ${problem ? "error" : ""}`} role={problem ? "alert" : "status"} aria-live="polite">
          <p>{problem || notice}</p>
          {problem && route.recheckRunId && (
            <button type="button" onClick={requestComparisonRefresh}>
              重新读取差异
            </button>
          )}
        </div>
      )}

      {route.step === "review" && (
        <div className="revisionStage revisionIssueStage">
          <div className="revisionStageHead">
            <div><h2>选择要处理的问题</h2><p>选择只用于定位与记录修订目标，不会改变原报告或自动改写正文。</p></div>
            <b>{baselineIssues.length} 项</b>
          </div>
          {!baselineReady ? (
            <div className="revisionEmpty">
              <h3>这个运行不能作为复检基线</h3>
              <p>复检需要一条已完成且包含冻结输入的运行，页面不会拿当前文档冒充历史输入。</p>
            </div>
          ) : baselineIssues.length === 0 ? (
            <div className="revisionEmpty"><h3>基线报告没有确认问题</h3><p>可以返回报告查看待澄清项，或在文稿变化后重新开始普通校验。</p></div>
          ) : (
            <div className="revisionIssueList">
              {baselineIssues.map((issue) => (
                <article
                  id={`revision-issue-${issue.id}`}
                  key={issue.id}
                  className={route.issueId === issue.id ? "selected" : ""}
                >
                  <button type="button" aria-pressed={route.issueId === issue.id} onClick={() => updateRoute({ issueId: issue.id })}>
                    <span>{issueCategoryLabels[issue.category] || issue.category} · {severityLabels[issue.severity] || issue.severity} · {Math.round(issue.confidence * 100)}%</span>
                    <b>{issue.title}</b>
                    <p>{issue.explanation}</p>
                    <small>{issue.evidence[0] ? `${issue.evidence[0].document_name}:${issue.evidence[0].line_start}` : "证据待核对"}</small>
                  </button>
                  {route.issueId === issue.id && (
                    <div className="revisionIssueDetail">
                      {issue.evidence.map((evidence, index) => (
                        <blockquote key={`${evidence.document_id}:${evidence.line_start}:${index}`}>
                          <b>{evidence.document_name}:{evidence.line_start}{evidence.line_end !== evidence.line_start ? `–${evidence.line_end}` : ""}</b>
                          {evidence.text}
                        </blockquote>
                      ))}
                      <p><b>修订建议：</b>{issue.suggestion}</p>
                      <button
                        type="button"
                        onClick={() => updateRoute({
                          step: "upload",
                          documentId: revisionDocumentForIssue(issue.evidence[0]?.document_name, documents),
                        })}
                      >
                        上传处理此问题的新版本
                      </button>
                    </div>
                  )}
                </article>
              ))}
            </div>
          )}
        </div>
      )}

      {route.step === "upload" && (
        <div className="revisionStage">
          <div className="revisionStageHead"><div><h2>上传修订后的文档</h2><p>浏览器不会保存所选本地文件；刷新后需要重新选择，但已成功上传的版本不会丢失。</p></div></div>
          {selectedIssue && (
            <aside className="revisionTarget">
              <small>当前修订目标</small><b>{selectedIssue.title}</b>
              <p>{selectedIssue.evidence[0]?.document_name}:{selectedIssue.evidence[0]?.line_start} · {selectedIssue.suggestion}</p>
            </aside>
          )}
          <form className="revisionUpload" onSubmit={(event) => void uploadRevision(event)}>
            <fieldset>
              <legend>版本关系</legend>
              <label htmlFor="revision-mode-replace"><input id="revision-mode-replace" type="radio" name="revision-mode" value="replace" checked={uploadMode === "replace"} onChange={() => {
                setUploadMode("replace");
                if (file) {
                  const validation = validateRevisionUploadSelection("replace", file.name, route.documentId, documents);
                  setProblem(validation.ok ? "" : validation.message);
                }
              }} />替换现有文档，生成新版本</label>
              <label htmlFor="revision-mode-related"><input id="revision-mode-related" type="radio" name="revision-mode" value="related" checked={uploadMode === "related"} onChange={() => {
                setUploadMode("related");
                if (file) {
                  const validation = validateRevisionUploadSelection("related", file.name, route.documentId, documents);
                  setProblem(validation.ok ? "" : validation.message);
                }
              }} />新增相关文档</label>
            </fieldset>
            {uploadMode === "replace" && (
              <label className="revisionField" htmlFor="revision-document">
                <span>替换目标</span>
                <select id="revision-document" name="document" value={route.documentId || ""} onChange={(event) => {
                  const documentId = event.target.value || null;
                  updateRoute({ documentId }, true);
                  if (file) {
                    const validation = validateRevisionUploadSelection("replace", file.name, documentId, documents);
                    setProblem(validation.ok ? "" : validation.message);
                  }
                }}>
                  <option value="">选择当前文档</option>
                  {activeDocuments.map((document) => <option key={document.id} value={document.id}>{document.name} · v{document.version}</option>)}
                </select>
              </label>
            )}
            <label className="revisionField" htmlFor="revision-file">
              <span>修订稿文件</span>
              <input
                ref={fileInput}
                id="revision-file"
                name="file"
                type="file"
                accept={supportedUploadAccept}
                onChange={(event) => {
                  const nextFile = event.target.files?.[0] || null;
                  setFile(nextFile);
                  if (!nextFile) {
                    setProblem("");
                    return;
                  }
                  const validation = validateRevisionUploadSelection(
                    uploadMode,
                    nextFile.name,
                    route.documentId,
                    documents,
                  );
                  setProblem(validation.ok ? "" : validation.message);
                }}
              />
              <small>支持 {supportedUploadLabel}；一次上传 1 份。替换版本时文件名必须与目标相同；改名文件应选择“新增相关文档”。</small>
            </label>
            <label className="revisionField" htmlFor="revision-role">
              <span>文档类型</span>
              <select id="revision-role" name="role" value={role} onChange={(event) => setRole(event.target.value as DocumentRole)}>
                {documentRoles.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
            <label className="revisionField" htmlFor="revision-scope">
              <span>故事作用域</span>
              <input id="revision-scope" name="scope" autoComplete="off" maxLength={80} value={scope} onChange={(event) => setScope(event.target.value)} placeholder="例如 global 或 route_a…" />
            </label>
            <div className="revisionFormActions">
              <button type="button" onClick={() => updateRoute({ step: "review" })}>返回问题</button>
              <button type="submit" disabled={uploading}>{uploading ? "上传中…" : "保存新版本"}</button>
            </div>
          </form>
        </div>
      )}

      {route.step === "run" && (
        <div className="revisionStage">
          <div className="revisionStageHead"><div><h2>确认变化并启动复检</h2><p>复检冻结此刻的 active 文档，并与上方基线运行逐项比较。</p></div></div>
          <div className={`revisionDelta ${delta.changed ? "changed" : "unchanged"}`}>
            <strong>{delta.changed ? "已检测到可复检的输入变化" : "尚未检测到输入变化"}</strong>
            <span>更新 {delta.updated.length} · 新增 {delta.added.length} · 移除 {delta.removed.length}</span>
            {delta.updated.map(({ before, after }) => <p key={after.id}>{before.document_name}：v{before.document_version} → v{after.version}</p>)}
            {delta.added.map((document) => <p key={document.id}>新增：{document.name} · v{document.version}</p>)}
            {delta.removed.map((document) => <p key={document.document_id}>移除：{document.document_name} · v{document.document_version}</p>)}
          </div>
          <fieldset className="revisionVisualOptions">
            <legend>复检完成后的按需视图</legend>
            <label><input type="checkbox" checked={route.generateGraph} onChange={(event) => updateRoute({ generateGraph: event.target.checked }, true)} />生成关系图</label>
            <label><input type="checkbox" checked={route.generateTimeline} onChange={(event) => updateRoute({ generateTimeline: event.target.checked }, true)} />生成时间线</label>
            <small>两项默认关闭；只在复检完成后请求所选视图，不影响问题比较。</small>
          </fieldset>
          {route.recheckRunId ? (
            <div className="recheckProgress" aria-live="polite">
              <span><i aria-hidden="true" />复检运行 {shortIdentifier(route.recheckRunId)}</span>
              <b>{safeComparison?.status === "ready" ? "比较完成" : safeComparison?.status === "failed" ? "运行失败" : safeComparison?.status === "cancelled" ? "运行已取消" : recheckMessage}</b>
              {safeComparison?.status === "pending" && (
                <div
                  className="recheckMeter"
                  role="progressbar"
                  aria-label="复检进度"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={Math.max(0, Math.min(100, recheckProgress))}
                >
                  <i style={{ width: `${Math.max(0, Math.min(100, recheckProgress))}%` }} />
                </div>
              )}
              {streamInterrupted && safeComparison?.status === "pending" && <small>比较状态仍会定时刷新，进度流恢复后继续显示命名阶段。</small>}
              <div>
                <button type="button" disabled={comparisonLoading} onClick={requestComparisonRefresh}>{comparisonLoading ? "读取中…" : "刷新状态"}</button>
                <a href={`/app/projects/${encodeURIComponent(projectId)}/runs/${encodeURIComponent(route.recheckRunId)}`}>打开精确运行详情</a>
                {safeComparison?.status === "ready" && <button type="button" onClick={() => updateRoute({ step: "compare" })}>查看复检差异</button>}
              </div>
            </div>
          ) : (
            <div className="revisionLaunch">
              <p>{!baselineReady ? "基线运行缺少可复现输入，不能复检。" : !delta.changed ? "先上传新版本，再启动复检。" : "启动后可能调用模型并消耗 Token；重复提交会复用同一个复检运行。"}</p>
              <button className="revisionPrimaryAction" type="button" disabled={!baselineReady || !delta.changed || starting} onClick={() => void startRecheck()}>
                {starting ? "提交复检中…" : "开始复检"}
              </button>
            </div>
          )}
        </div>
      )}

      {route.step === "compare" && (
        <div className="revisionStage revisionComparison">
          <div className="revisionStageHead">
            <div><h2>基线与复检结果</h2><p>“本次未再检出、仍存在、新增、无法确认”来自两次冻结报告的匹配，不等于对故事质量的最终裁决。</p></div>
            {route.recheckRunId && <button type="button" disabled={comparisonLoading} onClick={requestComparisonRefresh}>{comparisonLoading ? "刷新中…" : "刷新比较"}</button>}
          </div>
          {safeComparison?.status === "ready" && safeComparison.summary ? (
            <>
              <div className="comparisonSummary" aria-label="复检差异摘要">
                {(["no_longer_detected", "persisting", "new", "unverifiable"] as const).map((outcome) => (
                  <button key={outcome} type="button" className={`${outcome} ${route.outcome === outcome ? "active" : ""}`} aria-pressed={route.outcome === outcome} onClick={() => updateRoute({ outcome, page: 1 }, true)}>
                    <span>{outcomeLabels[outcome]}</span><b>{safeComparison.summary![outcome]}</b>
                  </button>
                ))}
                <button type="button" className={route.outcome === "all" ? "active" : ""} aria-pressed={route.outcome === "all"} onClick={() => updateRoute({ outcome: "all", page: 1 }, true)}><span>查看全部</span><b>{route.outcome === "all" && safeComparison.page ? safeComparison.page.total : safeComparison.summary.no_longer_detected + safeComparison.summary.persisting + safeComparison.summary.new + safeComparison.summary.unverifiable}</b></button>
              </div>
              <div className="comparisonMeta">
                <span>基线 {safeComparison.summary.total_baseline} 项</span><span>复检 {safeComparison.summary.total_target} 项</span>
                <span>可比性：{compatibilityCopy(safeComparison)}</span>
                {typeof safeComparison.summary.actionable_no_longer_detected === "number" && <span>本次未再检出中，可行动项 {safeComparison.summary.actionable_no_longer_detected}</span>}
                {!!safeComparison.summary.baseline_false_positive && <span>基线含 {safeComparison.summary.baseline_false_positive} 项历史误报反馈</span>}
                {visualState === "generating" && <span>正在生成所选视图…</span>}
                {visualState === "ready" && <span>所选关系视图已生成</span>}
                {visualState === "failed" && <span>关系视图生成失败，可稍后从关系与时间页重试</span>}
              </div>
              {safeComparison.page && safeComparison.page.total > safeComparison.page.limit && (
                <nav className="comparisonPagination" aria-label="复检差异分页">
                  <span>
                    第 {Math.floor(safeComparison.page.offset / safeComparison.page.limit) + 1} 页 · 共 {safeComparison.page.total} 项
                  </span>
                  <div>
                    <button type="button" disabled={safeComparison.page.offset === 0} onClick={() => updateRoute({ page: Math.max(1, route.page - 1) })}>上一页</button>
                    <button type="button" disabled={!safeComparison.page.has_more} onClick={() => updateRoute({ page: route.page + 1 })}>下一页</button>
                  </div>
                </nav>
              )}
              {visibleComparisonItems.length ? (
                <div className="comparisonList">
                  {visibleComparisonItems.map((item, index) => {
                    const issue = issueForItem(item);
                    const itemId = itemIdentifier(item, index);
                    return <article id={`comparison-${itemId}`} key={`${item.outcome}:${itemId}`} className={item.outcome}>
                      <header><span>{outcomeLabels[item.outcome] || "无法确认"}</span><small>{matchDescription(item)}</small></header>
                      <h3>{issue?.title || "问题记录不可用"}</h3>
                      {feedbackCopy(item) && <p className="comparisonFeedback">{feedbackCopy(item)}</p>}
                      <div className="comparisonEvidence">
                        <section><b>基线证据</b>{item.baseline_issue?.evidence?.length ? item.baseline_issue.evidence.map((evidence, evidenceIndex) => <blockquote key={`before:${evidenceIndex}`}><b>{evidence.document_name}:{evidence.line_start}</b>{evidence.text}</blockquote>) : <p>基线中没有对应问题。</p>}</section>
                        <section><b>复检证据</b>{item.target_issue?.evidence?.length ? item.target_issue.evidence.map((evidence, evidenceIndex) => <blockquote key={`after:${evidenceIndex}`}><b>{evidence.document_name}:{evidence.line_start}</b>{evidence.text}</blockquote>) : <p>{item.outcome === "no_longer_detected" ? "本次复检未再次检出；这不等同于机器确认问题已修复。" : "没有足够证据确认结果。"}</p>}</section>
                      </div>
                      {item.target_issue && <a className="comparisonReportLink" href={reportHref(projectId, route.recheckRunId!, item.target_issue.id)}>在复检报告中打开</a>}
                    </article>;
                  })}
                </div>
              ) : <div className="revisionEmpty"><h3>该分类下没有问题</h3><p>切换上方分类查看其它变化。</p></div>}
              <div className="comparisonActions">
                {route.recheckRunId && <a href={reportHref(projectId, route.recheckRunId)}>打开完整复检报告</a>}
                {route.recheckRunId && (route.generateGraph || route.generateTimeline) && <a href={`/app/projects/${encodeURIComponent(projectId)}/runs/${encodeURIComponent(route.recheckRunId)}/visuals`}>打开关系与时间视图</a>}
              </div>
            </>
          ) : (
            <div className="revisionEmpty">
              <h3>{safeComparison?.status === "failed" ? "复检运行失败" : safeComparison?.status === "cancelled" ? "复检运行已取消" : "复检结果尚未就绪"}</h3>
              <p>保留这个链接即可稍后返回；页面不会改用其他运行结果。</p>
              {route.recheckRunId && <button type="button" onClick={() => updateRoute({ step: "run" })}>查看复检状态</button>}
            </div>
          )}
        </div>
      )}
    </section>
  );
}
