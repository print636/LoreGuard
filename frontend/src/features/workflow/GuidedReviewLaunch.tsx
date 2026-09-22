import { useEffect, useMemo, useState } from "react";
import type { WorkspaceView } from "../../routing";
import { fetchCharacters } from "../characters/api";
import type { CharacterSummary } from "../characters/types";
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
  items: CharacterSummary[];
  total: number;
  modelCoverage: "full" | "partial" | "rules_only" | "unknown";
  sourceRunId: string | null;
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

export default function GuidedReviewLaunch({
  projectId,
  documents,
  runs,
  busy,
  projectLoading,
  providerLabel,
  providerTone,
  onStart,
  onNavigate,
}: Props) {
  const documentState = useMemo(() => guidedDocumentState(documents), [documents]);
  const [sensitivity, setSensitivity] = useState<ReviewSensitivity>("balanced");
  const [selectedDraftIds, setSelectedDraftIds] = useState<string[]>([]);
  const [action, setAction] = useState<"baseline" | "review" | null>(null);
  const [actionError, setActionError] = useState("");
  const [profiles, setProfiles] = useState<ProfileState>({
    loading: false,
    error: "",
    items: [],
    total: 0,
    modelCoverage: "unknown",
    sourceRunId: null,
  });
  const baselineRunCompleted = hasCompletedBaselineRun(runs);
  const currentBaselineRun = findCurrentBaselineRun(documents, runs);

  useEffect(() => {
    const valid = new Set(documentState.confirmedDrafts.map((document) => document.id));
    setSelectedDraftIds((current) => current.filter((id) => valid.has(id)));
  }, [documentState.confirmedDrafts.map((document) => document.id).join(":")]);

  useEffect(() => {
    if (!projectId || !currentBaselineRun || documentState.unresolvedBaseline.length) {
      setProfiles({ loading: false, error: "", items: [], total: 0, modelCoverage: "unknown", sourceRunId: null });
      return;
    }
    const controller = new AbortController();
    setProfiles((current) => ({ ...current, loading: true, error: "" }));
    void fetchCharacters(
      projectId,
      { page: 1, pageSize: 100 },
      controller.signal,
    )
      .then((page) => {
        if (page.has_more) {
          setProfiles({
            loading: false,
            error: "角色数量超过当前引导页可核对范围，请进入角色档案完成确认。",
            items: page.items,
            total: page.total,
            modelCoverage: page.model_coverage,
            sourceRunId: page.source_run_id,
          });
          return;
        }
        setProfiles({
          loading: false,
          error: "",
          items: page.items,
          total: page.total,
          modelCoverage: page.model_coverage,
          sourceRunId: page.source_run_id,
        });
      })
      .catch((error) => {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setProfiles({
          loading: false,
          error: "暂时无法读取角色基线状态。页面不会把未知状态当作已完成。",
          items: [],
          total: 0,
          modelCoverage: "unknown",
          sourceRunId: null,
        });
      });
    return () => controller.abort();
  }, [projectId, currentBaselineRun?.id, documentState.unresolvedBaseline.length]);

  const pendingCandidates = profiles.items.reduce(
    (sum, character) => sum + character.pending_candidate_count,
    0,
  );
  const confirmedItems = profiles.items.reduce(
    (sum, character) => sum + character.confirmed_item_count,
    0,
  );
  const contextsReady =
    documentState.baseline.length > 0 &&
    documentState.unresolvedBaseline.length === 0;
  const profilesReady =
    Boolean(currentBaselineRun) &&
    !profiles.loading &&
    !profiles.error &&
    profiles.sourceRunId === currentBaselineRun?.id &&
    profiles.modelCoverage === "full" &&
    confirmedItems > 0 &&
    pendingCandidates === 0;
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
    if (!draftsReady || busy || action) return;
    try {
      setAction("review");
      setActionError("");
      await onStart(analysisRunRequest("draft_review", sensitivity, selectedDraftIds));
    } catch (error) {
      setActionError(error instanceof Error ? error.message : "新稿审查没有启动，请重试。");
    } finally {
      setAction(null);
    }
  }

  function toggleDraft(documentId: string) {
    setSelectedDraftIds((current) =>
      current.includes(documentId)
        ? current.filter((id) => id !== documentId)
        : [...current, documentId],
    );
    setActionError("");
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
            ) : profiles.loading ? (
              <p className="stageNotice" aria-busy="true">正在读取角色基线状态…</p>
            ) : profiles.error ? (
              <p className="stageNotice warning">{profiles.error}</p>
            ) : profiles.sourceRunId !== currentBaselineRun.id ? (
              <>
                <p className="stageNotice warning">角色档案并非来自当前冻结资料的基线运行，不能用于解锁新稿审查。</p>
                <button className="quietPrimary" type="button" disabled={busy || Boolean(action)} onClick={() => void startBaseline()}>重新建立角色基线</button>
              </>
            ) : profiles.modelCoverage !== "full" ? (
              <>
                <p className="stageNotice warning">角色审查覆盖不完整（{profiles.modelCoverage}），不会把已有少量结果当作完整基线。</p>
                <button className="quietPrimary" type="button" disabled={busy || Boolean(action)} onClick={() => void startBaseline()}>重新建立角色基线</button>
              </>
            ) : pendingCandidates > 0 ? (
              <>
                <p className="stageNotice warning">还有 {pendingCandidates} 条 AI 归纳候选待确认；当前已有 {confirmedItems} 条正式档案。</p>
                <button type="button" onClick={() => onNavigate("characters")}>核对角色归纳</button>
              </>
            ) : confirmedItems === 0 ? (
              <>
                <p className="stageNotice warning">本次还没有形成已确认角色档案。这不代表资料中没有角色。</p>
                <button type="button" onClick={() => onNavigate("characters")}>查看角色档案</button>
              </>
            ) : (
              <p className="stageNotice success">{profiles.total} 个角色、{confirmedItems} 条档案已确认，可用于新稿审查。</p>
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
            {!profilesReady ? (
              <p className="stageNotice">完成角色基线确认后，才能用它审查新稿中的性格、偏好与行为漂移。</p>
            ) : documentState.confirmedDrafts.length === 0 ? (
              <>
                <p className="stageNotice warning">还没有已确认的草稿。将待审正文设为“故事正文 + 草稿/审阅中”，再确认上下文。</p>
                <button type="button" onClick={() => onNavigate("projects")}>前往设置新稿</button>
              </>
            ) : (
              <div className="draftReviewSelection">
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
                  {busy || action === "review" ? "正在校验…" : `开始校验${selectedDraftIds.length ? `（${selectedDraftIds.length} 份）` : ""}`}
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
