import { type FormEvent, useEffect, useRef, useState } from "react";
import { ApiError } from "../../api/client";
import EvidenceList from "./EvidenceList";
import TargetEvidencePreview from "./TargetEvidencePreview";
import { canCreateNewAxis, confirmedAxisPolarity, isScopedAxisDimension, previewAxisPolarity, scopedCandidateCanBind, selectableAxesForCandidate, selectedProjectAxis, selectedReviewAxis, validateApplicabilityScope, validateAxisDraft, validateAxisPositiveProposition, validScopedComparisonKey } from "./axisReview";
import { advanceReviewScope, isCurrentReviewRequest } from "./reviewScope";
import {
  candidateOriginNames,
  candidateDecisionLabel,
  candidateDisplayStatement,
  candidateReviewState,
  candidateStatusNames,
  candidateStatusExplanation,
  characterDimensionNames,
  describeCoverage,
} from "./presentation";
import type {
  CandidateDecision,
  CandidatePage,
  CharacterTraitAxis,
  ProfileCandidate,
  SourceNeighborPage,
} from "./types";

type CandidateReviewProps = {
  scopeKey: string;
  page: CandidatePage | null;
  selected: ProfileCandidate | null;
  selectedId: string | null;
  neighborPage: SourceNeighborPage | null;
  neighborLoading: boolean;
  neighborMoreBusy: boolean;
  neighborError: string;
  neighborMoreError: string;
  loading: boolean;
  detailLoading: boolean;
  error: string;
  detailError: string;
  decisionBusy: CandidateDecision | null;
  axes: CharacterTraitAxis[];
  axisTotal: number;
  axisFetchedCount: number;
  axisLoaded: boolean;
  axisListDirty: boolean;
  axisLoading: boolean;
  axisMoreBusy: boolean;
  axisCreateBusy: boolean;
  axisError: string;
  axisMoreError: string;
  axisRefreshKey: number;
  pageNumber: number;
  onSelect: (candidateId: string) => void;
  hrefForCandidate: (candidateId: string) => string;
  onPage: (page: number) => void;
  onRetry: () => void;
  onRetryDetail: () => void;
  onRetryNeighbors: () => void;
  onLoadMoreNeighbors: () => void;
  onBack: () => void;
  onRetryAxes: () => void;
  onLoadMoreAxes: () => void;
  onCreateAxis: (input: {
    trait_type: CharacterTraitAxis["trait_type"];
    display_name: string;
    definition: string;
    positive_proposition: string;
    comparison_key?: string;
    applicability_scope?: string;
  }) => Promise<CharacterTraitAxis>;
  onSetAxisProposition: (axis: CharacterTraitAxis, positiveProposition: string) => Promise<CharacterTraitAxis>;
  onDecision: (decision: CandidateDecision, comment: string, axis: CharacterTraitAxis | null, alignment: "same" | "opposite" | null, scopeApplicabilityConfirmed: boolean) => void;
};

const polarityNames: Record<NonNullable<ProfileCandidate["polarity"]>, string> = {
  positive: "positive（相对模型标签的正向）",
  negative: "negative（相对模型标签的反向）",
  neutral: "neutral（无方向）",
  unclear: "unclear（未判定）",
};

function axisCreateError(error: unknown): string {
  if (error instanceof ApiError) {
    const envelope = error.detail;
    const detail = envelope && typeof envelope === "object" && "detail" in envelope
      ? envelope.detail
      : null;
    const code = detail && typeof detail === "object" && "code" in detail
      ? detail.code
      : null;
    if (code === "character_trait_axis_duplicate") {
      return "相同定义的作者轴已经存在。请刷新并手动选择它，不要重复创建。";
    }
    if (error.status === 401) return "登录状态已失效，请重新登录。";
    if (error.status === 403) return "安全校验未通过，请刷新页面后重试。";
    if (error.status === 404) return "项目不存在或无权访问，请返回项目列表核对。";
    if (error.status === 422) return "轴名称或定义未通过校验，请检查后重试。";
    return "创建结果暂无法确认。先刷新轴列表核对，避免重复创建。";
  }
  return "连接中断，作者轴可能已创建。请先刷新轴列表核对，不要立即重复提交。";
}

function axisPropositionError(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 409) return "作者轴可能已由另一位作者定义。原输入已保留，请刷新轴列表核对后再继续。";
    if (error.status === 422) return "正向命题未通过校验，请检查措辞后重试。";
    if (error.status === 403) return "安全校验未通过，请刷新页面后重试。";
    if (error.status === 404) return "作者轴已不可访问，请刷新轴列表。";
  }
  return "命题保存结果暂无法确认。原输入已保留，请刷新轴列表核对，避免重复提交。";
}

export default function CandidateReview({
  scopeKey,
  page,
  selected,
  selectedId,
  neighborPage,
  neighborLoading,
  neighborMoreBusy,
  neighborError,
  neighborMoreError,
  loading,
  detailLoading,
  error,
  detailError,
  decisionBusy,
  axes,
  axisTotal,
  axisFetchedCount,
  axisLoaded,
  axisListDirty,
  axisLoading,
  axisMoreBusy,
  axisCreateBusy,
  axisError,
  axisMoreError,
  axisRefreshKey,
  pageNumber,
  onSelect,
  hrefForCandidate,
  onPage,
  onRetry,
  onRetryDetail,
  onRetryNeighbors,
  onLoadMoreNeighbors,
  onBack,
  onRetryAxes,
  onLoadMoreAxes,
  onCreateAxis,
  onSetAxisProposition,
  onDecision,
}: CandidateReviewProps) {
  const reviewScopeRef = useRef({ key: scopeKey, generation: 0 });
  reviewScopeRef.current = advanceReviewScope(reviewScopeRef.current, scopeKey);
  const [comment, setComment] = useState("");
  const [axisMode, setAxisMode] = useState<"existing" | "new">("existing");
  const [axisChoice, setAxisChoice] = useState("");
  const [axisChoiceError, setAxisChoiceError] = useState("");
  const [axisNeedsRecheck, setAxisNeedsRecheck] = useState(false);
  const [axisName, setAxisName] = useState("");
  const [axisDefinition, setAxisDefinition] = useState("");
  const [axisPositiveProposition, setAxisPositiveProposition] = useState("");
  const [applicabilityScope, setApplicabilityScope] = useState("");
  const [scopeTouched, setScopeTouched] = useState(false);
  const [scopeConfirmed, setScopeConfirmed] = useState(false);
  const [scopeError, setScopeError] = useState("");
  const [legacyAxisProposition, setLegacyAxisProposition] = useState("");
  const [legacyAxisPropositionError, setLegacyAxisPropositionError] = useState("");
  const [legacyAxisPropositionBusy, setLegacyAxisPropositionBusy] = useState(false);
  const [alignmentChoice, setAlignmentChoice] = useState<"same" | "opposite" | "uncertain">("uncertain");
  const [alignmentError, setAlignmentError] = useState("");
  const [updatedAxis, setUpdatedAxis] = useState<CharacterTraitAxis | null>(null);
  const [nameTouched, setNameTouched] = useState(false);
  const [definitionTouched, setDefinitionTouched] = useState(false);
  const [propositionTouched, setPropositionTouched] = useState(false);
  const [createAttempted, setCreateAttempted] = useState(false);
  const [createError, setCreateError] = useState("");
  const [createdMessage, setCreatedMessage] = useState("");
  const [createdAxis, setCreatedAxis] = useState<CharacterTraitAxis | null>(null);
  const axisSelectRef = useRef<HTMLSelectElement | null>(null);
  const axisNameRef = useRef<HTMLInputElement | null>(null);
  const axisDefinitionRef = useRef<HTMLTextAreaElement | null>(null);
  const axisPropositionRef = useRef<HTMLTextAreaElement | null>(null);
  const applicabilityScopeRef = useRef<HTMLTextAreaElement | null>(null);
  const scopeCheckboxRef = useRef<HTMLInputElement | null>(null);
  const alignmentRef = useRef<HTMLFieldSetElement | null>(null);
  const createErrorSummaryRef = useRef<HTMLDivElement | null>(null);
  const createPendingRef = useRef(false);
  const createRequestRef = useRef(0);
  const candidateLinkRefs = useRef(new Map<string, HTMLAnchorElement>());
  const queueTitleRef = useRef<HTMLHeadingElement | null>(null);
  const detailRegionRef = useRef<HTMLElement | null>(null);
  const detailTitleRef = useRef<HTMLHeadingElement | null>(null);
  const previousSelectedIdRef = useRef<string | null>(selectedId);
  const coverage = describeCoverage(
    page?.model_coverage || selected?.model_coverage || "unknown",
    page?.coverage_detail,
  );
  const review = selected ? candidateReviewState(selected) : null;

  useEffect(() => {
    if (reviewScopeRef.current.key !== scopeKey) return;
    createPendingRef.current = false;
    setComment("");
    setAxisMode("existing");
    setAxisChoice("");
    setAxisChoiceError("");
    setAxisNeedsRecheck(false);
    setAxisName("");
    setAxisDefinition("");
    setAxisPositiveProposition("");
    setApplicabilityScope("");
    setScopeTouched(false);
    setScopeConfirmed(false);
    setScopeError("");
    setLegacyAxisProposition("");
    setLegacyAxisPropositionError("");
    setLegacyAxisPropositionBusy(false);
    setAlignmentChoice("uncertain");
    setAlignmentError("");
    setUpdatedAxis(null);
    setNameTouched(false);
    setDefinitionTouched(false);
    setPropositionTouched(false);
    setCreateAttempted(false);
    setCreateError("");
    setCreatedMessage("");
    setCreatedAxis(null);
  }, [scopeKey]);

  useEffect(() => {
    // Preserve the user's choice on a conflict, but require a deliberate
    // re-selection before any newer axis version or proposition is trusted.
    if (axisChoice) {
      setAxisNeedsRecheck(true);
      setAxisChoiceError("轴列表已刷新。原选择已保留，请重新核对轴定义、命题与适用情境后再确认。");
    }
    setUpdatedAxis(null);
    setCreatedAxis(null);
    setAlignmentChoice("uncertain");
    setAlignmentError("");
    setScopeConfirmed(false);
    setScopeError("");
    // This effect intentionally reacts to a server refresh, not each local
    // choice; the current value is captured when that refresh arrives.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [axisRefreshKey]);

  const scopedCandidate = Boolean(selected && isScopedAxisDimension(selected.dimension));
  const scopedEligible = Boolean(selected && scopedCandidateCanBind(selected));
  const candidateAxes = selected ? selectableAxesForCandidate(axes, selected) : [];
  const locallySelectedAxes = selected ? selectableAxesForCandidate(
    [updatedAxis, createdAxis].filter((axis): axis is CharacterTraitAxis => Boolean(axis)), selected,
  ) : [];
  const selectedAxis = selectedReviewAxis(candidateAxes, locallySelectedAxes, axisChoice);
  const confirmedAxis = selected?.approved_axis_id
    ? selectedProjectAxis(candidateAxes, selected.approved_axis_id)
    : null;
  const confirmedMappedPolarity = selected ? confirmedAxisPolarity(selected, confirmedAxis) : null;
  const axisDraft = validateAxisDraft(axisName, axisDefinition, axisPositiveProposition);
  const scopeDraft = validateApplicabilityScope(applicabilityScope);
  const canCreateAxis = canCreateNewAxis({
    loaded: axisLoaded,
    loading: axisLoading,
    error: axisError,
    dirty: axisListDirty,
    fetchedCount: axisFetchedCount,
    total: axisTotal,
  });

  useEffect(() => {
    if (createdMessage && axisMode === "existing") {
      requestAnimationFrame(() => axisSelectRef.current?.focus());
    }
  }, [createdMessage, axisMode]);

  function confirmCandidate() {
    const mustBind = selected?.dimension === "core_personality";
    if (selected && (mustBind || (scopedCandidate && (axisChoice || axisMode === "new")))) {
      if (!selectedAxis) {
        setAxisChoiceError("请先选择适用于此候选的作者轴，或创建新轴后再确认。");
        requestAnimationFrame(() => {
          if (axisMode === "existing") axisSelectRef.current?.focus();
          else axisNameRef.current?.focus();
        });
        return;
      }
      if (axisNeedsRecheck) {
        setAxisChoiceError("请核对刷新后的轴定义和正向命题，并重新选择一次。");
        requestAnimationFrame(() => axisSelectRef.current?.focus());
        return;
      }
      if (!selectedAxis.positive_proposition_sha256) {
        setAxisChoiceError("这条旧作者轴还没有正向命题，请先在下方补写，不能猜测方向。");
        return;
      }
      if (selected.polarity !== "positive" && selected.polarity !== "negative") {
        setAlignmentError("模型原始方向不明确，不能建立有方向的作者轴绑定。请先驳回或等待重新归纳。");
        return;
      }
      if (alignmentChoice === "uncertain") {
        setAlignmentError("请核对证据后选择同向或反向；暂不确定时保留待审，不会自动确认。");
        requestAnimationFrame(() => alignmentRef.current?.focus());
        return;
      }
      if (scopedCandidate) {
        if (!scopedEligible || !selectedAxis.applicability_scope || !selectedAxis.applicability_scope_sha256) {
          setAxisChoiceError("这条候选或作者轴缺少可核对的对象、原文证据或适用情境；请保留待审。");
          requestAnimationFrame(() => axisSelectRef.current?.focus());
          return;
        }
        if (!scopeConfirmed) {
          setScopeError("请根据原文判断这一情境确实适用，再勾选确认；不确定时保留待审。");
          requestAnimationFrame(() => scopeCheckboxRef.current?.focus());
          return;
        }
      }
    }
    setAxisChoiceError("");
    setAlignmentError("");
    setScopeError("");
    onDecision("confirm", comment, selectedAxis, selectedAxis && alignmentChoice !== "uncertain" ? alignmentChoice : null, scopedCandidate && Boolean(selectedAxis) && scopeConfirmed);
  }

  async function submitLegacyAxisProposition(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedAxis || selectedAxis.positive_proposition || legacyAxisPropositionBusy) return;
    const validated = validateAxisPositiveProposition(legacyAxisProposition);
    setLegacyAxisPropositionError(validated.error);
    if (validated.error) return;
    setLegacyAxisPropositionBusy(true);
    try {
      const result = await onSetAxisProposition(selectedAxis, validated.value);
      setUpdatedAxis(result);
      setAxisNeedsRecheck(false);
      setAxisChoiceError("");
      setLegacyAxisPropositionError("");
      setLegacyAxisProposition("");
      setAlignmentChoice("uncertain");
    } catch (error) {
      setLegacyAxisPropositionError(axisPropositionError(error));
    } finally {
      setLegacyAxisPropositionBusy(false);
    }
  }

  async function submitNewAxis(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedId || !review?.allowed || axisCreateBusy || createPendingRef.current || !canCreateAxis) return;
    setCreateAttempted(true);
    setNameTouched(true);
    setDefinitionTouched(true);
    setPropositionTouched(true);
    if (scopedCandidate) setScopeTouched(true);
    setCreateError("");
    const invalid = Object.entries({ ...axisDraft.errors, ...(scopedCandidate ? { applicability_scope: scopeDraft.error } : {}) }).filter(([, message]) => message);
    if (invalid.length) {
      requestAnimationFrame(() => {
        if (invalid.length > 1) createErrorSummaryRef.current?.focus();
        else if (invalid[0]?.[0] === "display_name") axisNameRef.current?.focus();
        else if (invalid[0]?.[0] === "definition") axisDefinitionRef.current?.focus();
        else if (invalid[0]?.[0] === "positive_proposition") axisPropositionRef.current?.focus();
        else applicabilityScopeRef.current?.focus();
      });
      return;
    }
    const startedScope = reviewScopeRef.current;
    const requestId = ++createRequestRef.current;
    const isCurrentRequest = () =>
      isCurrentReviewRequest(reviewScopeRef.current, startedScope, createRequestRef.current, requestId);
    createPendingRef.current = true;
    try {
      if (!selected || (scopedCandidate && !scopedEligible)) return;
      const created = await onCreateAxis({
        trait_type: selected.dimension as CharacterTraitAxis["trait_type"],
        ...axisDraft.value,
        ...(scopedCandidate ? {
          comparison_key: validScopedComparisonKey(selected) || undefined,
          applicability_scope: scopeDraft.value,
        } : {}),
      });
      if (!isCurrentRequest()) return;
      setCreatedAxis(created);
      setAxisChoice(created.id);
      setAxisNeedsRecheck(false);
      setAxisMode("existing");
      setAxisChoiceError("");
      setScopeConfirmed(false);
      setScopeError("");
      setCreatedMessage(`“${created.display_name}”已创建并选中。请核对命题、原文${scopedCandidate ? "与适用情境" : ""}，再明确选择方向。`);
    } catch (error) {
      if (isCurrentRequest()) setCreateError(axisCreateError(error));
    } finally {
      if (isCurrentRequest()) createPendingRef.current = false;
    }
  }

  useEffect(() => {
    const previousId = previousSelectedIdRef.current;
    previousSelectedIdRef.current = selectedId;
    if (previousId && !selectedId) {
      requestAnimationFrame(() => {
        (candidateLinkRefs.current.get(previousId) || queueTitleRef.current)?.focus();
      });
      return;
    }
    if (!selectedId) return;
    requestAnimationFrame(() => {
      if (selected?.id === selectedId) detailTitleRef.current?.focus();
      else detailRegionRef.current?.focus();
    });
  }, [scopeKey, selectedId, selected?.id]);

  return (
    <div className="candidateWorkspace">
      {(page || selected) && (
        <section className={`characterCoverage ${coverage.tone}`} role="status">
          <b>{coverage.label}</b>
          <p>{coverage.detail}</p>
        </section>
      )}

      <div className={`candidateLayout ${selectedId ? "candidateSelected" : ""}`}>
        <section className="candidateQueue" aria-busy={loading}>
          <div className="characterSubhead">
            <h3 ref={queueTitleRef} tabIndex={-1}>待确认归纳</h3>
            <span>{page?.total || 0} 条</span>
          </div>
          {error ? (
            <div className="characterPanelError" role="alert">
              <b>候选列表没有加载</b>
              <p>{error}</p>
              <button type="button" onClick={onRetry}>重试</button>
            </div>
          ) : loading ? (
            <div className="characterInlineLoading">正在读取待确认归纳…</div>
          ) : !page || page.items.length === 0 ? (
            <div className="characterSectionEmpty compact">
              <h3>当前没有待确认的角色归纳</h3>
              <p>
                这只表示当前筛选和本次归纳没有待处理项，不代表角色资料已经完整。
              </p>
            </div>
          ) : (
            <ul>
              {page.items.map((candidate) => (
                <li key={candidate.id}>
                  <a
                    ref={(node) => {
                      if (node) candidateLinkRefs.current.set(candidate.id, node);
                      else candidateLinkRefs.current.delete(candidate.id);
                    }}
                    href={hrefForCandidate(candidate.id)}
                    className={selectedId === candidate.id ? "active" : ""}
                    aria-current={selectedId === candidate.id ? "true" : undefined}
                    onClick={(event) => {
                      if (
                        event.button !== 0 ||
                        event.metaKey ||
                        event.ctrlKey ||
                        event.shiftKey ||
                        event.altKey
                      ) return;
                      event.preventDefault();
                      onSelect(candidate.id);
                    }}
                  >
                    <span>
                      {characterDimensionNames[candidate.dimension]} ·{" "}
                      {candidateStatusNames[candidate.status]}
                    </span>
                    <b>{candidateDisplayStatement(candidate)}</b>
                    <small>{Math.round(candidate.confidence * 100)}% 置信度</small>
                  </a>
                </li>
              ))}
            </ul>
          )}
          {page && (pageNumber > 1 || page.has_more) && (
            <nav className="characterPagination" aria-label="待确认归纳分页">
              <button type="button" disabled={pageNumber <= 1 || loading} onClick={() => onPage(pageNumber - 1)}>上一页</button>
              <span>第 {pageNumber} 页</span>
              <button type="button" disabled={!page.has_more || loading} onClick={() => onPage(pageNumber + 1)}>下一页</button>
            </nav>
          )}
        </section>

        <section
          ref={detailRegionRef}
          className={`candidateDetail ${selectedId ? "selected" : ""}`}
          aria-busy={detailLoading}
          aria-label="归纳证据详情"
          tabIndex={-1}
        >
          {selectedId && (
            <button className="candidateMobileBack" type="button" onClick={onBack}>
              返回候选列表
            </button>
          )}
          {detailError ? (
            <div className="characterPanelError" role="alert">
              <b>这条归纳没有加载</b>
              <p>{detailError}</p>
              <button type="button" onClick={onRetryDetail}>重试</button>
            </div>
          ) : detailLoading ? (
            <div className="characterInlineLoading">正在读取归纳证据…</div>
          ) : !selected ? (
            <div className="characterSectionEmpty compact">
              <h3>选择一条归纳查看证据</h3>
              <p>确认前请同时阅读支持证据、反向证据和适用作用域。</p>
            </div>
          ) : (
            <>
              <header className="candidateStatement">
                <div>
                  <span>{characterDimensionNames[selected.dimension]}</span>
                  <span>{candidateOriginNames[selected.origin]}</span>
                  <strong>{candidateStatusNames[selected.status]}</strong>
                </div>
                <h3 ref={detailTitleRef} tabIndex={-1}>{candidateDisplayStatement(selected)}</h3>
                <p>{candidateStatusExplanation(selected.status)}</p>
              </header>

              <TargetEvidencePreview candidate={selected} />

              <section className="candidateReviewConditions" aria-label="适用条件与例外">
                <h4>适用条件与例外</h4>
                {selected.contexts.length || selected.scopes.length || selected.limitations.length ||
                  selected.valid_from_release_ordinal !== null || selected.valid_until_release_ordinal !== null ? (
                  <ul>
                    {selected.contexts.map((context) => <li key={`context:${context}`}><b>情境</b>{context}</li>)}
                    {selected.scopes.map((scope) => <li key={`scope:${scope.scope_id}`}><b>故事范围</b>{scope.label}</li>)}
                    {selected.valid_from_release_ordinal !== null || selected.valid_until_release_ordinal !== null ? (
                      <li><b>版本序号</b>{selected.valid_from_release_ordinal ?? "起点未标注"} 至 {selected.valid_until_release_ordinal ?? "终点未标注"}</li>
                    ) : null}
                    {selected.limitations.map((item, index) => <li key={`limit:${index}`}><b>模型提示的边界</b>{item}</li>)}
                  </ul>
                ) : <p>模型未给出明确适用条件；这不表示该行为在所有版本和情境下都成立。</p>}
                <p>请以原文核对成长阶段、保密任务、疾病等例外，不要把上下文省略后确认成永久性格。</p>
              </section>

              <EvidenceList
                title="支持这条归纳的证据"
                items={selected.supporting_evidence}
                emptyText="没有可展示的支持证据，因此不能仅凭候选表述作出确认。"
                candidateEvidence
                supportBindingsStatus={selected.support_bindings_status}
                supportBindings={selected.support_bindings_v1?.bindings || []}
              />
              <EvidenceList
                title="反向或例外证据"
                items={selected.contrary_evidence}
                emptyText="本次没有检索到反向证据；这不代表反向证据一定不存在。"
                tone="contrary"
                candidateEvidence
              />

              <section className="candidateTraitMetadata" aria-label="原始特征与权威范围">
                <dl>
                  <div><dt>模型原始标签</dt><dd>{selected.model_trait_key || "未提供"}</dd></div>
                  <div><dt>模型内部方向码</dt><dd>{selected.polarity ? polarityNames[selected.polarity] : "未提供"}</dd></div>
                  <div><dt>对象限定</dt><dd>{selected.key_object || selected.comparison_key || "无对象限定"}</dd></div>
                  <div><dt>来源资料级别</dt><dd>{selected.authority_tier === "core_canon" ? "核心设定来源" : selected.authority_tier === "formal_record" ? "正式资料来源" : "未提供"}</dd></div>
                  <div>
                    <dt>发布范围</dt>
                    <dd>{selected.valid_from_release_ordinal === null && selected.valid_until_release_ordinal === null
                      ? "未标注版本边界（不等于永久有效）"
                      : `${selected.valid_from_release_ordinal ?? "起点不限"} 至 ${selected.valid_until_release_ordinal ?? "后续不限"}`}</dd>
                  </div>
                  {selected.dimension === "core_personality" && (
                    <div>
                      <dt>作者批准轴</dt>
                      <dd>{selected.approved_axis_id
                        ? confirmedAxis?.version === selected.approved_axis_version
                          ? `${confirmedAxis.display_name} · v${selected.approved_axis_version}`
                          : `已绑定 · v${selected.approved_axis_version}（当前轴版本未核对）`
                        : selected.status === "confirmed"
                          ? "未绑定作者轴"
                          : "尚未绑定"}</dd>
                    </div>
                  )}
                  {selected.dimension === "core_personality" && selected.approved_axis_id && (
                    <div>
                      <dt>作者方向映射</dt>
                      <dd>{selected.axis_alignment
                        ? confirmedMappedPolarity
                          ? `${selected.axis_alignment === "same" ? "同向" : "反向"}；按已确认映射，作者比较句${confirmedMappedPolarity === "positive" ? "成立" : "不成立"}（非新稿事实复核）`
                          : "已记录方向；当前轴命题或冻结证据不可核对，暂不换算成立/不成立"
                        : "待作者补认，暂不作同轴方向判定"}</dd>
                    </div>
                  )}
                </dl>
                <p>模型解释：{selected.rationale}</p>
                <p>positive / negative 仅相对模型原始标签，不表示行为好坏，也不能按“不会”等字眼自动反转。正式比较轴须由作者核对证据后定义。</p>
              </section>

              <section className="candidateSourceNeighbors" aria-labelledby={`candidate-source-neighbors-${selected.id}`}>
                <div className="characterSubhead">
                  <h4 id={`candidate-source-neighbors-${selected.id}`}>完全相同证据区间的其他归纳</h4>
                  {neighborPage && <span>{neighborPage.total} 条</span>}
                </div>
                <p className="candidateSourceIntro">仅按本次运行的冻结输入与完全相同的起止行坐标对照；部分重叠区间不在此列。同一区间可能支持不同归纳，并不代表语义相同。每条仍须单独审核。</p>
                {!selected.source_verified ? (
                  <p className="candidateSourceNotice" role="status">冻结证据无法核对，不能显示同源对照或确认此归纳。</p>
                ) : neighborError ? (
                  <div className="characterPanelError" role="alert">
                    <p>{neighborError}</p>
                    <button type="button" onClick={onRetryNeighbors}>重新读取同源归纳</button>
                  </div>
                ) : neighborLoading || !neighborPage ? (
                  <p className="characterInlineEmpty" aria-busy="true">正在核对相同冻结证据区间…</p>
                ) : (
                  <>
                    {neighborPage.source_groups.map((group) => {
                      const related = neighborPage.items.filter((item) => item.shared_evidence.some((anchor) =>
                        anchor.input_id === group.input_id && anchor.line_start === group.line_start && anchor.line_end === group.line_end));
                      const complete = related.length >= group.total;
                      return (
                        <div className="candidateSourceGroup" key={`${group.input_id}:${group.line_start}:${group.line_end}`}>
                          <p className="candidateSourceGroupTitle">
                            <b>{group.document_name} · v{group.document_version} · 第 {group.line_start}{group.line_end === group.line_start ? "" : `–${group.line_end}`} 行</b>
                            <span>相同区间另有 {group.total} 条 · 已显示 {related.length} 条</span>
                          </p>
                          {group.context_verified && group.story_scope && <p className="candidateSourceScope">冻结作用域：{group.story_scope.label}</p>}
                          <div className="candidateSourceGrid">
                            <div className="candidateSourceItem current">
                              <small>当前审核 · {candidateStatusNames[selected.status]}</small>
                              <b>{selected.statement}</b>
                              <span>{selected.polarity ? polarityNames[selected.polarity] : "方向未提供"}</span>
                            </div>
                            {related.map((item) => (
                              <a
                                key={item.id}
                                className="candidateSourceItem"
                                href={hrefForCandidate(item.id)}
                                onClick={(event) => {
                                  if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
                                  event.preventDefault();
                                  onSelect(item.id);
                                }}
                              >
                                <small>查看并独立审核 · {candidateStatusNames[item.status]}</small>
                                <b>{item.statement}</b>
                                <span>{item.polarity ? polarityNames[item.polarity] : "方向未提供"}</span>
                              </a>
                            ))}
                          </div>
                          {group.total === 0 && <p className="characterInlineEmpty">这个证据区间暂无其他可核对的归纳。</p>}
                          {!complete && <p className="characterInlineEmpty">还有 {group.total - related.length} 条未显示；请继续加载后再比较这个区间的全部候选。</p>}
                        </div>
                      );
                    })}
                    {neighborPage.has_more && (
                      <div className="candidateSourcePagination">
                        <p>已加载 {neighborPage.items.length} / {neighborPage.total} 条；其他候选可能来自上述任意证据区间。</p>
                        <button type="button" disabled={neighborMoreBusy} onClick={onLoadMoreNeighbors}>
                          {neighborMoreBusy ? "正在加载…" : "继续加载同源归纳"}
                        </button>
                      </div>
                    )}
                    {neighborMoreError && <p className="candidateSourceNotice" role="alert">{neighborMoreError} <button type="button" onClick={onRetryNeighbors}>重新读取</button></p>}
                  </>
                )}
              </section>

              <section className={`candidateDecision ${review?.allowed ? "ready" : "blocked"}`}>
                <p>{review?.label}</p>
                {scopedCandidate && !scopedEligible && selected.status === "pending" && (
                  <p className="candidateAxisHint" role="status">
                    此候选暂不能绑定对象与情境比较轴：需明确设定来源、单条已核对的原文证据、有效对象键及明确方向。仍可按现有流程审核普通角色特征。
                  </p>
                )}
                {(selected.dimension === "core_personality" || (scopedCandidate && scopedEligible)) && selected.status === "pending" && (
                  <section className="candidateAxisBinding" aria-labelledby={`candidate-axis-heading-${selected.id}`}>
                    <h4 id={`candidate-axis-heading-${selected.id}`}>作者批准比较轴</h4>
                    <p>{scopedCandidate
                      ? "先核对原文，再决定这条候选是否属于所选对象与情境。确认仅建立作者认可的角色基线；价值观和行为边界的新稿漂移检测仍在开发中。"
                      : "轴只定义“比较什么”，不代表当前候选一定正确。创建与确认分两步；绑定只影响之后创建的分析，既有报告不会改写。"}</p>
                    {scopedCandidate && (
                      <div className="candidateAxisScopeReview">
                        <b>当前候选</b>
                        <p>{candidateDisplayStatement(selected)}</p>
                        <p><strong>比较对象：</strong>{validScopedComparisonKey(selected)?.split(":").slice(1).join(":") || "未能核对"}</p>
                        <p><strong>原文证据：</strong>{selected.supporting_evidence[0]?.text || "原文缺失"}</p>
                      </div>
                    )}
                    <div className="candidateAxisMode" role="group" aria-label="作者轴来源">
                      <button
                        type="button"
                        className={axisMode === "existing" ? "active" : ""}
                        aria-pressed={axisMode === "existing"}
                        onClick={() => { setAxisMode("existing"); setAxisChoiceError(""); setAlignmentChoice("uncertain"); setScopeConfirmed(false); setScopeError(""); }}
                      >选择已有轴</button>
                      <button
                        type="button"
                        className={axisMode === "new" ? "active" : ""}
                        aria-pressed={axisMode === "new"}
                        disabled={!review?.allowed || !canCreateAxis || Boolean(decisionBusy) || axisCreateBusy}
                        onClick={() => { setAxisMode("new"); setAxisChoice(""); setUpdatedAxis(null); setAxisNeedsRecheck(false); setAxisChoiceError(""); setAlignmentChoice("uncertain"); setScopeConfirmed(false); setScopeError(""); setCreatedMessage(""); }}
                      >创建新轴</button>
                    </div>
                    {axisLoading && <p className="candidateAxisHint" role="status">正在读取当前项目的作者轴…</p>}
                    {axisError && (
                      <div className="candidateAxisError" role="alert">
                        <p>{axisError} 作者轴列表未核对完成，暂不能新建或确认。</p>
                        <button type="button" onClick={onRetryAxes}>刷新轴列表</button>
                      </div>
                    )}
                    {axisListDirty && !axisError && (
                      <div className="candidateAxisPagination">
                        <p>刚创建的轴已选中。请重新核对候选原文、轴定义和适用情境。若要再建新轴，请先刷新列表。</p>
                        <button type="button" onClick={onRetryAxes}>重新读取作者轴</button>
                      </div>
                    )}
                    {!axisListDirty && !axisError && !axisLoading && axisFetchedCount < axisTotal && (
                      <div className="candidateAxisPagination">
                        <p>已读取 {axisFetchedCount} / {axisTotal} 条作者轴。尚有未加载条目；请先看完再决定是否创建，避免重复定义。</p>
                        <button type="button" disabled={axisMoreBusy} onClick={onLoadMoreAxes}>
                          {axisMoreBusy ? "正在加载…" : "继续加载作者轴"}
                        </button>
                        {axisMoreError && <p role="alert">{axisMoreError}</p>}
                      </div>
                    )}
                    {axisMode === "existing" && !axisError && (
                      <div className="candidateAxisSelect">
                        <label htmlFor={`candidate-axis-${selected.id}`}>项目内已有轴</label>
                        <select
                          ref={axisSelectRef}
                          id={`candidate-axis-${selected.id}`}
                          value={axisChoice}
                          disabled={!review?.allowed || axisLoading || Boolean(decisionBusy) || axisCreateBusy}
                          aria-invalid={Boolean(axisChoiceError)}
                          aria-describedby={`candidate-axis-help-${selected.id}${axisChoiceError ? ` candidate-axis-error-${selected.id}` : ""}`}
                          onChange={(event) => { setAxisChoice(event.target.value); setUpdatedAxis(null); setAxisNeedsRecheck(false); setLegacyAxisProposition(""); setLegacyAxisPropositionError(""); setAlignmentChoice("uncertain"); setAlignmentError(""); setScopeConfirmed(false); setScopeError(""); setAxisChoiceError(""); setCreatedMessage(""); }}
                        >
                          <option value="">请选择比较轴</option>
                          {candidateAxes.map((axis) => (
                            <option key={axis.id} value={axis.id}>{axis.display_name} · v{axis.version}{axis.applicability_scope ? ` · 适用：${Array.from(axis.applicability_scope).slice(0, 18).join("")}${Array.from(axis.applicability_scope).length > 18 ? "…" : ""}` : ""}</option>
                          ))}
                          {createdAxis && locallySelectedAxes.some((axis) => axis.id === createdAxis.id) && !candidateAxes.some((axis) => axis.id === createdAxis.id) && (
                            <option value={createdAxis.id}>{createdAxis.display_name} · v{createdAxis.version}{createdAxis.applicability_scope ? ` · 适用：${Array.from(createdAxis.applicability_scope).slice(0, 18).join("")}${Array.from(createdAxis.applicability_scope).length > 18 ? "…" : ""}` : ""}（刚创建）</option>
                          )}
                        </select>
                        <p id={`candidate-axis-help-${selected.id}`} className="candidateAxisHint">
                          {candidateAxes.length ? "请核对轴定义、对象、情境与原文；相近措辞不等于同一轴。" : scopedCandidate ? "尚无同维度、同对象的轴。可创建新轴，也可仅确认普通角色特征。" : "此项目还没有核心性格轴。读取完成后可创建新轴。"}
                        </p>
                        {selectedAxis && (
                          <div className="candidateAxisDefinition">
                            <b>{selectedAxis.display_name}</b><p>{selectedAxis.definition}</p>
                            {scopedCandidate && <>
                              <p><strong>比较对象：</strong>{selectedAxis.comparison_key?.split(":").slice(1).join(":")}</p>
                              <p><strong>适用情境：</strong>{selectedAxis.applicability_scope}</p>
                            </>}
                            <p><strong>作者比较句（轴正向）：</strong>{selectedAxis.positive_proposition || "尚未由作者定义"}</p>
                            <p>“正向”只是作者定义的比较坐标，不是好坏评价，也不代表这句话已在剧情中发生。</p>
                          </div>
                        )}
                        {selectedAxis && axisNeedsRecheck && (
                          <button
                            type="button"
                            className="candidateAxisRecheck"
                            onClick={() => { setAxisNeedsRecheck(false); setAxisChoiceError(""); setAlignmentChoice("uncertain"); setScopeConfirmed(false); setScopeError(""); }}
                          >我已核对刷新后的轴定义{scopedCandidate ? "、对象、情境" : "与命题"}</button>
                        )}
                        {selectedAxis && !scopedCandidate && !selectedAxis.positive_proposition && (
                          <form className="candidateAxisCreate" onSubmit={(event) => void submitLegacyAxisProposition(event)} noValidate>
                            <label htmlFor={`candidate-axis-legacy-proposition-${selected.id}`}>为旧作者轴补写正向命题</label>
                            <textarea
                              id={`candidate-axis-legacy-proposition-${selected.id}`}
                              value={legacyAxisProposition}
                              maxLength={200}
                              disabled={legacyAxisPropositionBusy || Boolean(decisionBusy)}
                              aria-invalid={Boolean(legacyAxisPropositionError)}
                              aria-describedby={legacyAxisPropositionError ? `candidate-axis-legacy-proposition-error-${selected.id}` : undefined}
                              onChange={(event) => { setLegacyAxisProposition(event.target.value); setLegacyAxisPropositionError(""); }}
                            />
                            <p className="candidateAxisHint">这是一次性的作者定义，不能根据轴名称自动推断。补写后仍须单独选择候选方向。</p>
                            {legacyAxisPropositionError && (
                              <div className="candidateAxisError" role="alert">
                                <p id={`candidate-axis-legacy-proposition-error-${selected.id}`}>{legacyAxisPropositionError}</p>
                                <button type="button" onClick={onRetryAxes}>刷新轴列表核对</button>
                              </div>
                            )}
                            <button type="submit" disabled={legacyAxisPropositionBusy || Boolean(decisionBusy)}>{legacyAxisPropositionBusy ? "正在保存…" : "保存正向命题"}</button>
                          </form>
                        )}
                        {axisChoiceError && <p id={`candidate-axis-error-${selected.id}`} className="candidateFieldError" role="alert">{axisChoiceError}</p>}
                      </div>
                    )}
                    {axisMode === "new" && (
                      <form className="candidateAxisCreate" onSubmit={(event) => void submitNewAxis(event)} noValidate>
                        {axisChoiceError && <p className="candidateFieldError" role="alert">{axisChoiceError}</p>}
                        {createAttempted && Object.values({ ...axisDraft.errors, ...(scopedCandidate ? { applicability_scope: scopeDraft.error } : {}) }).filter(Boolean).length > 1 && (
                          <div className="candidateAxisErrorSummary" role="alert" tabIndex={-1} ref={createErrorSummaryRef}>
                            <b>请先修正以下内容</b>
                            <ul>
                              {axisDraft.errors.display_name && <li><a href={`#candidate-axis-name-${selected.id}`}>{axisDraft.errors.display_name}</a></li>}
                              {axisDraft.errors.definition && <li><a href={`#candidate-axis-definition-${selected.id}`}>{axisDraft.errors.definition}</a></li>}
                              {axisDraft.errors.positive_proposition && <li><a href={`#candidate-axis-proposition-${selected.id}`}>{axisDraft.errors.positive_proposition}</a></li>}
                              {scopedCandidate && scopeDraft.error && <li><a href={`#candidate-axis-scope-${selected.id}`}>{scopeDraft.error}</a></li>}
                            </ul>
                          </div>
                        )}
                        <label htmlFor={`candidate-axis-name-${selected.id}`}>轴名称</label>
                        <input
                          ref={axisNameRef}
                          id={`candidate-axis-name-${selected.id}`}
                          value={axisName}
                          maxLength={80}
                          disabled={!review?.allowed || axisCreateBusy || Boolean(decisionBusy)}
                          aria-invalid={Boolean((nameTouched || createAttempted) && axisDraft.errors.display_name)}
                          aria-describedby={`candidate-axis-name-help-${selected.id}${(nameTouched || createAttempted) && axisDraft.errors.display_name ? ` candidate-axis-name-error-${selected.id}` : ""}`}
                          onChange={(event) => setAxisName(event.target.value)}
                          onBlur={() => setNameTouched(true)}
                        />
                        <p id={`candidate-axis-name-help-${selected.id}`} className="candidateAxisHint">{scopedCandidate ? "给当前对象与情境的比较轴取一个便于识别的名称。" : "例如“涉及同伴安全的路线决策”。不要把正反方向写进名称。"}</p>
                        {(nameTouched || createAttempted) && axisDraft.errors.display_name && <p id={`candidate-axis-name-error-${selected.id}`} className="candidateFieldError" role="alert">{axisDraft.errors.display_name}</p>}
                        <label htmlFor={`candidate-axis-definition-${selected.id}`}>轴定义</label>
                        <textarea
                          ref={axisDefinitionRef}
                          id={`candidate-axis-definition-${selected.id}`}
                          value={axisDefinition}
                          maxLength={200}
                          disabled={!review?.allowed || axisCreateBusy || Boolean(decisionBusy)}
                          aria-invalid={Boolean((definitionTouched || createAttempted) && axisDraft.errors.definition)}
                          aria-describedby={`candidate-axis-definition-help-${selected.id}${(definitionTouched || createAttempted) && axisDraft.errors.definition ? ` candidate-axis-definition-error-${selected.id}` : ""}`}
                          onChange={(event) => setAxisDefinition(event.target.value)}
                          onBlur={() => setDefinitionTouched(true)}
                        />
                        <p id={`candidate-axis-definition-help-${selected.id}`} className="candidateAxisHint">说明比较的{scopedCandidate ? "价值取向或行为边界" : "具体行为边界"}；另用正向命题明确判定方向。</p>
                        {(definitionTouched || createAttempted) && axisDraft.errors.definition && <p id={`candidate-axis-definition-error-${selected.id}`} className="candidateFieldError" role="alert">{axisDraft.errors.definition}</p>}
                        <label htmlFor={`candidate-axis-proposition-${selected.id}`}>轴的正向命题</label>
                        <textarea
                          ref={axisPropositionRef}
                          id={`candidate-axis-proposition-${selected.id}`}
                          value={axisPositiveProposition}
                          maxLength={200}
                          disabled={!review?.allowed || axisCreateBusy || Boolean(decisionBusy)}
                          aria-invalid={Boolean((propositionTouched || createAttempted) && axisDraft.errors.positive_proposition)}
                          aria-describedby={`candidate-axis-proposition-help-${selected.id}${(propositionTouched || createAttempted) && axisDraft.errors.positive_proposition ? ` candidate-axis-proposition-error-${selected.id}` : ""}`}
                          onChange={(event) => setAxisPositiveProposition(event.target.value)}
                          onBlur={() => setPropositionTouched(true)}
                        />
                        <p id={`candidate-axis-proposition-help-${selected.id}`} className="candidateAxisHint">{scopedCandidate ? "写出在当前对象与情境下用于比较的具体正向命题。" : "例如“角色未经授权冒用同伴签名”。"}这只定义何为正向，不表示角色已这样做。</p>
                        {(propositionTouched || createAttempted) && axisDraft.errors.positive_proposition && <p id={`candidate-axis-proposition-error-${selected.id}`} className="candidateFieldError" role="alert">{axisDraft.errors.positive_proposition}</p>}
                        {scopedCandidate && (
                          <>
                            <p className="candidateAxisHint">比较对象取自已核对的候选对象键：{validScopedComparisonKey(selected)?.split(":").slice(1).join(":")}。新轴只对这个对象生效。</p>
                            <label htmlFor={`candidate-axis-scope-${selected.id}`}>轴的适用情境</label>
                            <textarea
                              ref={applicabilityScopeRef}
                              id={`candidate-axis-scope-${selected.id}`}
                              value={applicabilityScope}
                              maxLength={200}
                              disabled={!review?.allowed || axisCreateBusy || Boolean(decisionBusy)}
                              aria-invalid={Boolean((scopeTouched || createAttempted) && scopeDraft.error)}
                              aria-describedby={`candidate-axis-scope-help-${selected.id}${(scopeTouched || createAttempted) && scopeDraft.error ? ` candidate-axis-scope-error-${selected.id}` : ""}`}
                              onChange={(event) => { setApplicabilityScope(event.target.value); setScopeError(""); }}
                              onBlur={() => setScopeTouched(true)}
                            />
                            <p id={`candidate-axis-scope-help-${selected.id}`} className="candidateAxisHint">写出何时、对谁、在什么条件下适用；不要写“任何时候”，除非原文确实如此。</p>
                            {(scopeTouched || createAttempted) && scopeDraft.error && <p id={`candidate-axis-scope-error-${selected.id}`} className="candidateFieldError" role="alert">{scopeDraft.error}</p>}
                          </>
                        )}
                        <button type="submit" disabled={!review?.allowed || axisCreateBusy || Boolean(decisionBusy) || !canCreateAxis}>
                          {axisCreateBusy ? "正在创建作者轴…" : "先创建作者轴"}
                        </button>
                        {createError && <div className="candidateAxisError" role="alert"><p>{createError}</p><button type="button" onClick={onRetryAxes}>刷新轴列表</button></div>}
                      </form>
                    )}
                    {createdMessage && <p className="candidateAxisCreated" role="status">{createdMessage}</p>}
                    {selectedAxis?.positive_proposition_sha256 && (
                      <fieldset
                        ref={alignmentRef}
                        tabIndex={-1}
                        className="candidateAxisAlignment"
                        aria-describedby={alignmentError ? `candidate-axis-alignment-error-${selected.id}` : undefined}
                      >
                        <legend>核对模型标签与作者比较句的方向</legend>
                        <p>模型内部标签：{selected.model_trait_key || "未提供"}；模型方向码：{selected.polarity ? polarityNames[selected.polarity] : "未提供"}。作者比较句：{selectedAxis.positive_proposition}。请按原文判断两种比较坐标是否同向，不要根据否定词自动选择。</p>
                        <label><input type="radio" name={`candidate-alignment-${selected.id}`} checked={alignmentChoice === "same"} onChange={() => { setAlignmentChoice("same"); setAlignmentError(""); }} />同向：模型标签的正向状态与作者比较句同向</label>
                        <label><input type="radio" name={`candidate-alignment-${selected.id}`} checked={alignmentChoice === "opposite"} onChange={() => { setAlignmentChoice("opposite"); setAlignmentError(""); }} />反向：模型标签的正向状态与作者比较句相反</label>
                        <label><input type="radio" name={`candidate-alignment-${selected.id}`} checked={alignmentChoice === "uncertain"} onChange={() => { setAlignmentChoice("uncertain"); setAlignmentError(""); }} />暂不确定，保留待审</label>
                        {previewAxisPolarity(selected.polarity, alignmentChoice) && (
                          <p role="status">映射预览（未经事实复核）：若采用你的方向选择，这条候选表示“作者比较句{previewAxisPolarity(selected.polarity, alignmentChoice) === "positive" ? "成立" : "不成立"}”。这里只做方向换算，不代表模型或系统验证了原文判断。</p>
                        )}
                        {alignmentError && <p id={`candidate-axis-alignment-error-${selected.id}`} className="candidateFieldError" role="alert">{alignmentError}</p>}
                      </fieldset>
                    )}
                    {scopedCandidate && selectedAxis?.applicability_scope && (
                      <div className="candidateAxisScopeAcknowledgment">
                        <label htmlFor={`candidate-axis-scope-confirm-${selected.id}`}>
                          <input
                            ref={scopeCheckboxRef}
                            id={`candidate-axis-scope-confirm-${selected.id}`}
                            type="checkbox"
                            checked={scopeConfirmed}
                            disabled={!review?.allowed || Boolean(decisionBusy) || axisNeedsRecheck}
                            aria-invalid={Boolean(scopeError)}
                            aria-describedby={`candidate-axis-scope-confirm-help-${selected.id}${scopeError ? ` candidate-axis-scope-confirm-error-${selected.id}` : ""}`}
                            onChange={(event) => { setScopeConfirmed(event.target.checked); setScopeError(""); }}
                          />
                          <span>我已根据上方原文判断：这条候选确实适用于“{selectedAxis.applicability_scope}”</span>
                        </label>
                        <p id={`candidate-axis-scope-confirm-help-${selected.id}`}>对象键相同不能证明情境适用；此判断由作者负责。不确定时暂不勾选，可保留待审。</p>
                        {scopeError && <p id={`candidate-axis-scope-confirm-error-${selected.id}`} className="candidateFieldError" role="alert">{scopeError}</p>}
                      </div>
                    )}
                    {scopedCandidate && <p className="candidateAxisHint">若不绑定比较轴，仍可确认普通角色特征；后续需要作者另行建立轴基线。</p>}
                  </section>
                )}
                <label htmlFor={`candidate-comment-${selected.id}`}>
                  审核备注（可选）
                </label>
                <textarea
                  id={`candidate-comment-${selected.id}`}
                  value={comment}
                  maxLength={500}
                  disabled={Boolean(decisionBusy) || !review?.allowed}
                  onChange={(event) => setComment(event.target.value)}
                  placeholder="记录确认依据或驳回原因"
                />
                <div>
                  <button
                    className="characterConfirm"
                    type="button"
                    disabled={Boolean(decisionBusy) || !review?.allowed}
                    onClick={confirmCandidate}
                  >
                    {candidateDecisionLabel("confirm", decisionBusy)}
                  </button>
                  <button
                    type="button"
                    disabled={Boolean(decisionBusy) || !review?.allowed}
                    onClick={() => onDecision("reject", comment, null, null, false)}
                  >
                    {candidateDecisionLabel("reject", decisionBusy)}
                  </button>
                </div>
              </section>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
