import { type FormEvent, useEffect, useRef, useState } from "react";
import { ApiError } from "../../api/client";
import EvidenceList from "./EvidenceList";
import { canCreateNewAxis, selectedProjectAxis, validateAxisDraft } from "./axisReview";
import { advanceReviewScope, isCurrentReviewRequest } from "./reviewScope";
import {
  candidateOriginNames,
  candidateDecisionLabel,
  candidateReviewState,
  candidateStatusNames,
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
  onCreateAxis: (input: { display_name: string; definition: string }) => Promise<CharacterTraitAxis>;
  onDecision: (decision: CandidateDecision, comment: string, axis: CharacterTraitAxis | null) => void;
};

const polarityNames: Record<NonNullable<ProfileCandidate["polarity"]>, string> = {
  positive: "正向",
  negative: "反向",
  neutral: "中性",
  unclear: "方向不明",
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
  onDecision,
}: CandidateReviewProps) {
  const reviewScopeRef = useRef({ key: scopeKey, generation: 0 });
  reviewScopeRef.current = advanceReviewScope(reviewScopeRef.current, scopeKey);
  const [comment, setComment] = useState("");
  const [axisMode, setAxisMode] = useState<"existing" | "new">("existing");
  const [axisChoice, setAxisChoice] = useState("");
  const [axisChoiceError, setAxisChoiceError] = useState("");
  const [axisName, setAxisName] = useState("");
  const [axisDefinition, setAxisDefinition] = useState("");
  const [nameTouched, setNameTouched] = useState(false);
  const [definitionTouched, setDefinitionTouched] = useState(false);
  const [createAttempted, setCreateAttempted] = useState(false);
  const [createError, setCreateError] = useState("");
  const [createdMessage, setCreatedMessage] = useState("");
  const [createdAxis, setCreatedAxis] = useState<CharacterTraitAxis | null>(null);
  const axisSelectRef = useRef<HTMLSelectElement | null>(null);
  const axisNameRef = useRef<HTMLInputElement | null>(null);
  const axisDefinitionRef = useRef<HTMLTextAreaElement | null>(null);
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
    setAxisName("");
    setAxisDefinition("");
    setNameTouched(false);
    setDefinitionTouched(false);
    setCreateAttempted(false);
    setCreateError("");
    setCreatedMessage("");
    setCreatedAxis(null);
  }, [scopeKey]);

  useEffect(() => {
    // A refreshed axis list may have a newer version. Require a deliberate
    // re-selection rather than silently confirming with the new version.
    setAxisChoice("");
    setAxisChoiceError("");
  }, [axisRefreshKey]);

  const selectedAxis = selectedProjectAxis(axes, axisChoice) ||
    (createdAxis?.id === axisChoice ? createdAxis : null);
  const confirmedAxis = selected?.approved_axis_id
    ? selectedProjectAxis(axes, selected.approved_axis_id)
    : null;
  const axisDraft = validateAxisDraft(axisName, axisDefinition);
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
    if (selected?.dimension === "core_personality") {
      if (!selectedAxis) {
        setAxisChoiceError("请先选择已有作者轴，或创建新轴后再确认。");
        requestAnimationFrame(() => {
          if (axisMode === "existing") axisSelectRef.current?.focus();
          else axisNameRef.current?.focus();
        });
        return;
      }
    }
    setAxisChoiceError("");
    onDecision("confirm", comment, selectedAxis);
  }

  async function submitNewAxis(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedId || !review?.allowed || axisCreateBusy || createPendingRef.current || !canCreateAxis) return;
    setCreateAttempted(true);
    setNameTouched(true);
    setDefinitionTouched(true);
    setCreateError("");
    const invalid = Object.entries(axisDraft.errors).filter(([, message]) => message);
    if (invalid.length) {
      requestAnimationFrame(() => {
        if (invalid.length > 1) createErrorSummaryRef.current?.focus();
        else if (invalid[0]?.[0] === "display_name") axisNameRef.current?.focus();
        else axisDefinitionRef.current?.focus();
      });
      return;
    }
    const startedScope = reviewScopeRef.current;
    const requestId = ++createRequestRef.current;
    const isCurrentRequest = () =>
      isCurrentReviewRequest(reviewScopeRef.current, startedScope, createRequestRef.current, requestId);
    createPendingRef.current = true;
    try {
      const created = await onCreateAxis(axisDraft.value);
      if (!isCurrentRequest()) return;
      setCreatedAxis(created);
      setAxisChoice(created.id);
      setAxisMode("existing");
      setAxisChoiceError("");
      setCreatedMessage(`“${created.display_name}”已创建并选中。请再次核对证据，再确认候选。`);
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
                    <b>{candidate.statement}</b>
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
                <h3 ref={detailTitleRef} tabIndex={-1}>{selected.statement}</h3>
                <p>{selected.rationale}</p>
                <ul className="characterScopeList" aria-label="适用作用域与情境">
                  {selected.contexts.map((context) => (
                    <li key={`context:${context}`}>情境：{context}</li>
                  ))}
                  {selected.scopes.map((scope) => (
                    <li key={scope.scope_id}>{scope.label}</li>
                  ))}
                </ul>
              </header>

              <section className="candidateTraitMetadata" aria-label="原始特征与权威范围">
                <dl>
                  <div><dt>模型原始标签</dt><dd>{selected.model_trait_key || "未提供"}</dd></div>
                  <div><dt>相对方向</dt><dd>{selected.polarity ? polarityNames[selected.polarity] : "未提供"}</dd></div>
                  <div><dt>对象限定</dt><dd>{selected.comparison_key || "无对象限定"}</dd></div>
                  <div><dt>资料权威</dt><dd>{selected.authority_tier === "core_canon" ? "核心设定" : selected.authority_tier === "formal_record" ? "正式资料" : "未提供"}</dd></div>
                  <div>
                    <dt>发布范围</dt>
                    <dd>{selected.valid_from_release_ordinal === null && selected.valid_until_release_ordinal === null
                      ? "未限定发布序号"
                      : `${selected.valid_from_release_ordinal ?? "起点不限"} 至 ${selected.valid_until_release_ordinal ?? "后续不限"}`}</dd>
                  </div>
                  {selected.dimension === "core_personality" && (
                    <div>
                      <dt>作者批准轴</dt>
                      <dd>{selected.approved_axis_id
                        ? confirmedAxis
                          ? `${confirmedAxis.display_name} · v${selected.approved_axis_version}`
                          : `已绑定 · v${selected.approved_axis_version}`
                        : selected.status === "confirmed"
                          ? "未绑定作者轴"
                          : "尚未绑定"}</dd>
                    </div>
                  )}
                </dl>
                <p>模型标签只是归纳线索；正式比较轴需要作者根据原文证据确认。</p>
              </section>

              {selected.limitations.length > 0 && (
                <section className="candidateLimitations">
                  <h4>判断边界</h4>
                  <ul>
                    {selected.limitations.map((item, index) => <li key={index}>{item}</li>)}
                  </ul>
                </section>
              )}

              <EvidenceList
                title="支持这条归纳的证据"
                items={selected.supporting_evidence}
                emptyText="没有可展示的支持证据，因此不能仅凭候选表述作出确认。"
                candidateEvidence
              />
              <EvidenceList
                title="反向或例外证据"
                items={selected.contrary_evidence}
                emptyText="本次没有检索到反向证据；这不代表反向证据一定不存在。"
                tone="contrary"
                candidateEvidence
              />

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
                {selected.dimension === "core_personality" && selected.status === "pending" && (
                  <section className="candidateAxisBinding" aria-labelledby={`candidate-axis-heading-${selected.id}`}>
                    <h4 id={`candidate-axis-heading-${selected.id}`}>作者批准比较轴</h4>
                    <p>轴只定义“比较什么”，不代表当前候选一定正确。创建与确认分两步；绑定只影响之后创建的分析，既有报告不会改写。</p>
                    <div className="candidateAxisMode" role="group" aria-label="作者轴来源">
                      <button
                        type="button"
                        className={axisMode === "existing" ? "active" : ""}
                        aria-pressed={axisMode === "existing"}
                        onClick={() => { setAxisMode("existing"); setAxisChoiceError(""); }}
                      >选择已有轴</button>
                      <button
                        type="button"
                        className={axisMode === "new" ? "active" : ""}
                        aria-pressed={axisMode === "new"}
                        disabled={!review?.allowed || !canCreateAxis || Boolean(decisionBusy) || axisCreateBusy}
                        onClick={() => { setAxisMode("new"); setAxisChoice(""); setAxisChoiceError(""); setCreatedMessage(""); }}
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
                        <p>刚创建的轴已选中，可以继续确认本候选。若要再建新轴，请先刷新列表核对其它页面可能新增的轴。</p>
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
                          onChange={(event) => { setAxisChoice(event.target.value); setAxisChoiceError(""); setCreatedMessage(""); }}
                        >
                          <option value="">请选择比较轴</option>
                          {axes.map((axis) => (
                            <option key={axis.id} value={axis.id}>{axis.display_name} · v{axis.version}</option>
                          ))}
                          {createdAxis && !axes.some((axis) => axis.id === createdAxis.id) && (
                            <option value={createdAxis.id}>{createdAxis.display_name} · v{createdAxis.version}（刚创建）</option>
                          )}
                        </select>
                        <p id={`candidate-axis-help-${selected.id}`} className="candidateAxisHint">
                          {axes.length ? "请核对定义与原文是否是同一语义轴；相近措辞不等于同一轴。" : "此项目还没有作者轴。读取完成后可创建新轴。"}
                        </p>
                        {selectedAxis && (
                          <div className="candidateAxisDefinition">
                            <b>{selectedAxis.display_name}</b><p>{selectedAxis.definition}</p>
                          </div>
                        )}
                        {axisChoiceError && <p id={`candidate-axis-error-${selected.id}`} className="candidateFieldError" role="alert">{axisChoiceError}</p>}
                      </div>
                    )}
                    {axisMode === "new" && (
                      <form className="candidateAxisCreate" onSubmit={(event) => void submitNewAxis(event)} noValidate>
                        {axisChoiceError && <p className="candidateFieldError" role="alert">{axisChoiceError}</p>}
                        {createAttempted && axisDraft.errors.display_name && axisDraft.errors.definition && (
                          <div className="candidateAxisErrorSummary" role="alert" tabIndex={-1} ref={createErrorSummaryRef}>
                            <b>请先修正以下内容</b>
                            <ul>
                              <li><a href={`#candidate-axis-name-${selected.id}`}>{axisDraft.errors.display_name}</a></li>
                              <li><a href={`#candidate-axis-definition-${selected.id}`}>{axisDraft.errors.definition}</a></li>
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
                        <p id={`candidate-axis-name-help-${selected.id}`} className="candidateAxisHint">例如“涉及同伴安全的路线决策”。不要把正反方向写进名称。</p>
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
                        <p id={`candidate-axis-definition-help-${selected.id}`} className="candidateAxisHint">说明比较的具体行为边界；方向留给候选的正反方向字段。</p>
                        {(definitionTouched || createAttempted) && axisDraft.errors.definition && <p id={`candidate-axis-definition-error-${selected.id}`} className="candidateFieldError" role="alert">{axisDraft.errors.definition}</p>}
                        <button type="submit" disabled={!review?.allowed || axisCreateBusy || Boolean(decisionBusy) || !canCreateAxis}>
                          {axisCreateBusy ? "正在创建作者轴…" : "先创建作者轴"}
                        </button>
                        {createError && <div className="candidateAxisError" role="alert"><p>{createError}</p><button type="button" onClick={onRetryAxes}>刷新轴列表</button></div>}
                      </form>
                    )}
                    {createdMessage && <p className="candidateAxisCreated" role="status">{createdMessage}</p>}
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
                    onClick={() => onDecision("reject", comment, null)}
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
