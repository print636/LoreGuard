import { useEffect, useRef, useState } from "react";
import EvidenceList from "./EvidenceList";
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
  ProfileCandidate,
} from "./types";

type CandidateReviewProps = {
  page: CandidatePage | null;
  selected: ProfileCandidate | null;
  selectedId: string | null;
  loading: boolean;
  detailLoading: boolean;
  error: string;
  detailError: string;
  decisionBusy: CandidateDecision | null;
  pageNumber: number;
  onSelect: (candidateId: string) => void;
  hrefForCandidate: (candidateId: string) => string;
  onPage: (page: number) => void;
  onRetry: () => void;
  onRetryDetail: () => void;
  onBack: () => void;
  onDecision: (decision: CandidateDecision, comment: string) => void;
};

export default function CandidateReview({
  page,
  selected,
  selectedId,
  loading,
  detailLoading,
  error,
  detailError,
  decisionBusy,
  pageNumber,
  onSelect,
  hrefForCandidate,
  onPage,
  onRetry,
  onRetryDetail,
  onBack,
  onDecision,
}: CandidateReviewProps) {
  const [comment, setComment] = useState("");
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
    setComment("");
  }, [selected?.id]);

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
  }, [selectedId, selected?.id]);

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
              />
              <EvidenceList
                title="反向或例外证据"
                items={selected.contrary_evidence}
                emptyText="本次没有检索到反向证据；这不代表反向证据一定不存在。"
                tone="contrary"
              />

              <section className={`candidateDecision ${review?.allowed ? "ready" : "blocked"}`}>
                <p>{review?.label}</p>
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
                    onClick={() => onDecision("confirm", comment)}
                  >
                    {candidateDecisionLabel("confirm", decisionBusy)}
                  </button>
                  <button
                    type="button"
                    disabled={Boolean(decisionBusy) || !review?.allowed}
                    onClick={() => onDecision("reject", comment)}
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
