import { useEffect, useMemo, useRef, useState } from "react";
import type { WorkspaceView } from "../../routing";
import {
  fetchCharacterBaselineStatus,
  hasReadyCharacterBaseline,
  type CharacterBaselineStatus,
} from "./baselineStatus";
import {
  analysisRunRequest,
  findCurrentBaselineRun,
  guidedDocumentState,
  hasCompletedBaselineRun,
  type AnalysisRunRequest,
  type BaselineRunSummary,
  type GuidedDocument,
  type ReviewSensitivity,
} from "./guidedReview";

type RunSummary = BaselineRunSummary;

type Props = {
  projectId: string;
  selectionUserId: string;
  selectionWorkspaceId: string;
  documents: GuidedDocument[];
  runs: RunSummary[];
  busy: boolean;
  projectLoading: boolean;
  providerLabel: string;
  providerTone: string;
  onStart: (request: AnalysisRunRequest) => Promise<void>;
  onNavigate: (view: WorkspaceView) => void;
};

type ProfileState = {
  loading: boolean;
  error: string;
  projectId: string | null;
  requestedRunId: string | null;
  status: CharacterBaselineStatus | null;
};

const sensitivityOptions: Array<{
  value: ReviewSensitivity;
  label: string;
  detail: string;
}> = [
  { value: "conservative", label: "谨慎", detail: "证据不足时更少报告角色变化" },
  { value: "balanced", label: "平衡", detail: "角色变化的准确率与发现能力兼顾" },
  { value: "exploratory", label: "探索", detail: "发现更多角色变化线索，需更多人工复核" },
];

function selectionStorageKey(userId: string, workspaceId: string, projectId: string): string {
  return `loreguard:guided-review:drafts:${JSON.stringify([userId, workspaceId, projectId])}`;
}

function readDraftSelection(storageKey: string): string[] {
  try {
    const value = JSON.parse(window.sessionStorage.getItem(storageKey) || "[]");
    return Array.isArray(value)
      ? Array.from(new Set(value.filter((id): id is string => typeof id === "string" && id.length > 0 && id.length <= 200))).slice(0, 500)
      : [];
  } catch {
    return [];
  }
}

function saveDraftSelection(storageKey: string, ids: string[]): void {
  try {
    window.sessionStorage.setItem(storageKey, JSON.stringify(ids));
  } catch {
    // Session storage is optional; selection still works for this visit.
  }
}

export default function GuidedReviewLaunch({
  projectId,
  selectionUserId,
  selectionWorkspaceId,
  documents,
  runs,
  busy,
  projectLoading,
  providerLabel,
  providerTone,
  onStart,
  onNavigate,
}: Props) {
  const storageKey = selectionStorageKey(selectionUserId, selectionWorkspaceId, projectId);
  const documentState = useMemo(() => guidedDocumentState(documents), [documents]);
  const [sensitivity, setSensitivity] = useState<ReviewSensitivity>("balanced");
  const [draftSelection, setDraftSelection] = useState(() => ({
    storageKey,
    ids: readDraftSelection(storageKey),
  }));
  const [action, setAction] = useState<"baseline" | "review" | null>(null);
  const [actionError, setActionError] = useState("");
  const [reviewError, setReviewError] = useState("");
  const [statusRefresh, setStatusRefresh] = useState(0);
  const preflightControllerRef = useRef<AbortController | null>(null);
  const [profiles, setProfiles] = useState<ProfileState>({
    loading: false,
    error: "",
    projectId: null,
    requestedRunId: null,
    status: null,
  });
  const baselineRunCompleted = hasCompletedBaselineRun(runs);
  const currentBaselineRun = findCurrentBaselineRun(documents, runs);

  const confirmedDraftIdKey = documentState.confirmedDrafts.map((document) => document.id).join(":");
  const confirmedDraftIds = new Set(documentState.confirmedDrafts.map((document) => document.id));
  const selectedDraftIds = draftSelection.storageKey === storageKey
    ? draftSelection.ids.filter((id) => confirmedDraftIds.has(id))
    : [];

  useEffect(() => {
    if (draftSelection.storageKey !== storageKey) {
      setDraftSelection({ storageKey, ids: readDraftSelection(storageKey) });
    }
  }, [storageKey, draftSelection.storageKey]);

  useEffect(() => {
    if (projectLoading || documents.length === 0 || draftSelection.storageKey !== storageKey) return;
    setDraftSelection((current) => {
      if (current.storageKey !== storageKey) return current;
      const next = current.ids.filter((id) => confirmedDraftIds.has(id));
      if (next.length === current.ids.length) return current;
      saveDraftSelection(storageKey, next);
      return { storageKey, ids: next };
    });
  }, [storageKey, projectLoading, documents.length, confirmedDraftIdKey, draftSelection.storageKey]);

  useEffect(() => {
    if (!projectId || !currentBaselineRun || documentState.unresolvedBaseline.length) {
      setProfiles({ loading: false, error: "", projectId: null, requestedRunId: null, status: null });
      return;
    }
    const controller = new AbortController();
    setProfiles({ loading: true, error: "", projectId, requestedRunId: currentBaselineRun.id, status: null });
    void fetchCharacterBaselineStatus(projectId, currentBaselineRun.id, controller.signal)
      .then((status) => {
        setProfiles({
          loading: false,
          error: "",
          projectId,
          requestedRunId: currentBaselineRun.id,
          status,
        });
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setProfiles({
          loading: false,
          error: "暂时无法读取完整的角色基线汇总。请重试；未知状态不会解锁新稿审查。",
          projectId,
          requestedRunId: currentBaselineRun.id,
          status: null,
        });
      });
    return () => controller.abort();
  }, [projectId, currentBaselineRun?.id, documentState.unresolvedBaseline.length, statusRefresh]);

  useEffect(() => {
    // The component can be reused when the selected project, account, or
    // baseline changes. Discard the previous scope's pending action and copy.
    setAction(null);
    setActionError("");
    setReviewError("");
    return () => {
      preflightControllerRef.current?.abort();
      preflightControllerRef.current = null;
    };
  }, [projectId, currentBaselineRun?.id, storageKey]);

  const status = profiles.projectId === projectId && profiles.requestedRunId === currentBaselineRun?.id
    ? profiles.status
    : null;
  const profilesLoading = Boolean(currentBaselineRun) && (
    profiles.loading || profiles.projectId !== projectId || profiles.requestedRunId !== currentBaselineRun?.id
  );
  const pendingCandidates = status?.pending_candidate_count ?? 0;
  const confirmedItems = status?.confirmed_trait_count ?? 0;
  const contextsReady =
    documentState.baseline.length > 0 &&
    documentState.unresolvedBaseline.length === 0;
  const profilesReady =
    Boolean(currentBaselineRun) &&
    !profilesLoading &&
    !profiles.error &&
    hasReadyCharacterBaseline(currentBaselineRun, status);
  const draftsReady = profilesReady && documentState.confirmedDrafts.length > 0;

  async function startBaseline() {
    if (!contextsReady || busy || action) return;
    try {
      setAction("baseline");
      setActionError("");
      await onStart(analysisRunRequest("baseline_build", sensitivity));
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "角色基线任务没有启动，请重试。");
    } finally {
      setAction(null);
    }
  }

  async function startDraftReview() {
    if (!draftsReady || busy || action || preflightControllerRef.current || !currentBaselineRun || !selectedDraftIds.length) return;
    const controller = new AbortController();
    preflightControllerRef.current = controller;
    try {
      setAction("review");
      setReviewError("");
      let freshStatus: CharacterBaselineStatus;
      try {
        freshStatus = await fetchCharacterBaselineStatus(projectId, currentBaselineRun.id, controller.signal);
      } catch {
        if (controller.signal.aborted) return;
        setProfiles({
          loading: false,
          error: "提交前无法复核角色基线。请重试读取状态；未知状态不会解锁审查。",
          projectId,
          requestedRunId: currentBaselineRun.id,
          status: null,
        });
        setReviewError("提交前复核失败，本次没有启动任务。请重试读取角色基线后再校验。");
        return;
      }
      if (controller.signal.aborted) return;
      setProfiles({ loading: false, error: "", projectId, requestedRunId: currentBaselineRun.id, status: freshStatus });
      if (!hasReadyCharacterBaseline(currentBaselineRun, freshStatus)) {
        setReviewError("角色基线状态已变化，本次没有启动任务。请核对上方角色档案状态后再校验。");
        return;
      }
      await onStart(analysisRunRequest("draft_review", sensitivity, selectedDraftIds));
    } catch (error) {
      setReviewError(error instanceof Error ? error.message : "新稿审查没有启动，请重试。");
    } finally {
      if (preflightControllerRef.current === controller) {
        preflightControllerRef.current = null;
        setAction(null);
      }
    }
  }

  function toggleDraft(documentId: string) {
    setDraftSelection((current) => {
      const visible = current.storageKey === storageKey
        ? current.ids.filter((id) => confirmedDraftIds.has(id))
        : [];
      const next = visible.includes(documentId)
        ? visible.filter((id) => id !== documentId)
        : [...visible, documentId];
      saveDraftSelection(storageKey, next);
      return { storageKey, ids: next };
    });
    setReviewError("");
  }

  return (
    <div className="guidedReviewLaunch">
      <header className="guidedReviewHead">
        <div>
          <h2>从正式资料到新稿审查</h2>
          <p>先建立可核对的角色基线，再只选择这次需要审查的新稿。每次运行都会冻结输入。</p>
        </div>
        <fieldset className="sensitivityControl">
          <legend>角色变化敏感度</legend>
          {sensitivityOptions.map((option) => (
            <label key={option.value} title={option.detail}>
              <input
                type="radio"
                name="review-sensitivity"
                value={option.value}
                checked={sensitivity === option.value}
                onChange={() => setSensitivity(option.value)}
                disabled={busy || Boolean(action)}
              />
              <span>{option.label}</span>
            </label>
          ))}
        </fieldset>
      </header>

      {actionError && <div className="guidedActionError" role="alert">{actionError}</div>}

      <ol className="guidedStages" aria-label="故事审查三阶段">
        <li className={contextsReady ? "complete" : "current"}>
          <span className="stageNumber" aria-hidden="true">1</span>
          <div className="stageBody">
            <header>
              <div><h3>确认正式资料</h3><p>角色设定、世界观和已发布剧情先成为可靠依据。</p></div>
              <b>{contextsReady ? "已就绪" : "待处理"}</b>
            </header>
            {projectLoading ? (
              <p className="stageNotice">正在读取资料状态…</p>
            ) : documentState.baseline.length === 0 ? (
              <p className="stageNotice warning">还没有可建立基线的正式资料。请确认世界观、角色档案或已发布历史章节；参考材料不会自动成为基线。</p>
            ) : documentState.unresolvedBaseline.length > 0 ? (
              <p className="stageNotice warning">
                {documentState.unresolvedBaseline.length} 份正式资料仍待确认：
                {documentState.unresolvedBaseline.slice(0, 3).map((document) => document.name).join("、")}
                {documentState.unresolvedBaseline.length > 3 ? " 等" : ""}
              </p>
            ) : (
              <p className="stageNotice success">{documentState.baseline.length} 份正式资料已确认，可建立角色基线。</p>
            )}
            {!contextsReady && <button type="button" onClick={() => onNavigate("projects")}>前往确认资料</button>}
          </div>
        </li>

        <li className={profilesReady ? "complete" : contextsReady ? "current" : "locked"}>
          <span className="stageNumber" aria-hidden="true">2</span>
          <div className="stageBody">
            <header>
              <div><h3>建立并确认角色基线</h3><p>AI 从正式资料归纳候选；只有你确认的条目才进入角色档案。</p></div>
              <b>{profilesReady ? "已就绪" : contextsReady ? "进行中" : "未解锁"}</b>
            </header>
            {!contextsReady ? (
              <p className="stageNotice">完成上一步后才能开始角色归纳，页面不会绕过未确认上下文。</p>
            ) : !currentBaselineRun ? (
              <>
                <p className={`stageNotice ${baselineRunCompleted ? "warning" : ""}`}>
                  {baselineRunCompleted
                    ? "正式资料的版本或上下文已变化，旧角色基线不再匹配当前冻结输入，请重新建立。"
                    : "正式资料只作为背景输入，不会被当作本次待审新稿。"}
                </p>
                <button className="quietPrimary" type="button" disabled={busy || Boolean(action)} onClick={() => void startBaseline()}>
                  {action === "baseline" ? "正在启动…" : baselineRunCompleted ? "重新建立角色基线" : "建立角色基线"}
                </button>
              </>
            ) : profilesLoading ? (
              <p className="stageNotice" aria-busy="true">正在读取角色基线状态…</p>
            ) : profiles.error ? (
              <div className="stageStatusRecovery" role="alert">
                <p className="stageNotice warning">{profiles.error}</p>
                <button type="button" onClick={() => { setReviewError(""); setStatusRefresh((count) => count + 1); }}>重试读取基线状态</button>
              </div>
            ) : !status || status.baseline_run_id !== currentBaselineRun.id || status.baseline_run_status !== "completed" ? (
              <>
                <p className="stageNotice warning">角色基线汇总并非来自当前冻结资料的完整运行，不能用于解锁新稿审查。</p>
                <button className="quietPrimary" type="button" disabled={busy || Boolean(action)} onClick={() => void startBaseline()}>重新建立角色基线</button>
              </>
            ) : status.model_coverage !== "full" ? (
              <>
                <p className="stageNotice warning">角色审查覆盖不完整（{status.model_coverage}），不会把已有少量结果当作完整基线。</p>
                <button className="quietPrimary" type="button" disabled={busy || Boolean(action)} onClick={() => void startBaseline()}>重新建立角色基线</button>
              </>
            ) : status.readiness !== "ready" ? (
              <>
                <p className="stageNotice warning">角色基线尚未就绪（{status.readiness}），不会把未知状态当作可用档案。</p>
                <button className="quietPrimary" type="button" disabled={busy || Boolean(action)} onClick={() => void startBaseline()}>重新建立角色基线</button>
              </>
            ) : confirmedItems === 0 ? (
              <>
                <p className="stageNotice warning">项目级角色档案中暂无已确认特征{pendingCandidates > 0 ? `，还有 ${pendingCandidates} 条待确认候选` : ""}。这不代表资料中没有角色，也不能据此宣称角色审查已就绪。</p>
                <button type="button" onClick={() => onNavigate("characters")}>核对角色档案</button>
              </>
            ) : (
              <div className="stageBaselineSummary" aria-live="polite">
                <p className="stageNotice success">项目级汇总：{status.total_characters} 个角色、{confirmedItems} 条已确认特征；当前角色基线可用于新稿审查。</p>
                {pendingCandidates > 0 && (
                  <div className="stagePendingNotice">
                    <p className="stageNotice warning">另有 {pendingCandidates} 条待确认的 AI 归纳候选。它们不会作为正式角色设定参与审查；你可以先审查新稿，之后再核对候选。</p>
                    <button type="button" onClick={() => onNavigate("characters")}>核对待确认候选</button>
                  </div>
                )}
              </div>
            )}
          </div>
        </li>

        <li className={draftsReady ? "current" : "locked"}>
          <span className="stageNumber" aria-hidden="true">3</span>
          <div className="stageBody">
            <header>
              <div><h3>选择新稿开始审查</h3><p>一次可检查单章，也可勾选同一批次的多个章节。</p></div>
              <b>{draftsReady ? "可开始" : "未解锁"}</b>
            </header>
            {reviewError && <p className="stageReviewError" role="alert">{reviewError}</p>}
            {!profilesReady ? (
              <p className="stageNotice">完成角色基线确认后，才能用它审查新稿中的性格、偏好与行为漂移。</p>
            ) : documentState.confirmedDrafts.length === 0 ? (
              <>
                <p className="stageNotice warning">还没有已确认的草稿。将待审正文设为“故事正文 + 草稿/审阅中”，再确认上下文。</p>
                <button type="button" onClick={() => onNavigate("projects")}>前往设置新稿</button>
              </>
            ) : (
              <div className="draftReviewSelection">
                <p className="stageNotice draftReviewScopeNote">上方为项目级汇总，不保证所选章节中的每个角色都有已确认特征；未覆盖角色不会获得角色 OOC 判断，其他一致性检查仍可运行。</p>
                <fieldset>
                  <legend>本次待审新稿</legend>
                  {documentState.confirmedDrafts.map((document) => (
                    <label key={document.id}>
                      <input
                        type="checkbox"
                        checked={selectedDraftIds.includes(document.id)}
                        onChange={() => toggleDraft(document.id)}
                        disabled={busy || Boolean(action)}
                      />
                      <span><b>{document.name}</b><small>v{document.version} · {document.narrative_context?.publication_status === "draft" ? "草稿" : "审阅中"}</small></span>
                    </label>
                  ))}
                </fieldset>
                <button
                  className="startValidation"
                  type="button"
                  disabled={busy || Boolean(action) || selectedDraftIds.length === 0}
                  onClick={() => void startDraftReview()}
                >
                  {busy ? "正在校验…" : action === "review" ? "正在复核角色基线…" : `开始校验${selectedDraftIds.length ? `（${selectedDraftIds.length} 份）` : ""}`}
                </button>
              </div>
            )}
          </div>
        </li>
      </ol>

      <footer className="guidedReviewBoundary">
        <span className={`boundaryLight ${providerTone}`} aria-hidden="true" />
        <p><b>模型连接：{providerLabel}</b><small>状态仅表示连接准备情况；实际 AI 覆盖以运行完成后的诊断为准。模型不可用时不会把规则结果冒充 AI 判断。</small></p>
      </footer>
    </div>
  );
}
