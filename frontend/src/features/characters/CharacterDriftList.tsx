import EvidenceList from "./EvidenceList";
import {
  characterDriftReportPath,
  describeCoverage,
  feedbackStatusNames,
} from "./presentation";
import type { DriftIssuePage } from "./types";

type CharacterDriftListProps = {
  projectId: string;
  page: DriftIssuePage | null;
  pageNumber: number;
  loading: boolean;
  error: string;
  onPage: (page: number) => void;
  onRetry: () => void;
};

export default function CharacterDriftList({
  projectId,
  page,
  pageNumber,
  loading,
  error,
  onPage,
  onRetry,
}: CharacterDriftListProps) {
  const coverage = describeCoverage(
    page?.model_coverage || "unknown",
    page?.coverage_detail,
  );

  return (
    <div className="characterDriftWorkspace" aria-busy={loading}>
      {page && (
        <section className={`characterCoverage ${coverage.tone}`} role="status">
          <b>{coverage.label}</b>
          <p>{coverage.detail}</p>
        </section>
      )}
      {error ? (
        <div className="characterPanelError" role="alert">
          <b>角色漂移问题没有加载</b>
          <p>{error}</p>
          <button type="button" onClick={onRetry}>重试</button>
        </div>
      ) : loading ? (
        <div className="characterInlineLoading">正在读取角色漂移问题…</div>
      ) : !page || page.items.length === 0 ? (
        <div className="characterSectionEmpty">
          <h3>本次没有报告角色漂移问题</h3>
          <p>
            这只表示当前覆盖范围内没有形成可报告的问题，不代表角色在全部历史剧情中必然一致。
          </p>
        </div>
      ) : (
        <ul className="characterDriftList">
          {page.items.map((issue) => (
            <li key={issue.issue_id}>
              <article>
                <header>
                  <div>
                    <span>{issue.severity} · {Math.round(issue.confidence * 100)}%</span>
                    <strong>{feedbackStatusNames[issue.feedback_status]}</strong>
                  </div>
                  <h3>{issue.title}</h3>
                  <p>{issue.explanation}</p>
                  <small>作用域：{issue.scope.label}</small>
                </header>
                <EvidenceList
                  title="证据预览"
                  items={issue.evidence_preview}
                  emptyText="摘要没有携带证据预览，请打开完整报告核对。"
                />
                <a
                  className="characterReportLink"
                  href={characterDriftReportPath(
                    projectId,
                    issue.run_id,
                    issue.issue_id,
                  )}
                >
                  查看完整证据与反馈
                </a>
              </article>
            </li>
          ))}
        </ul>
      )}
      {page && (pageNumber > 1 || page.has_more) && (
        <nav className="characterPagination" aria-label="角色漂移问题分页">
          <button type="button" disabled={pageNumber <= 1 || loading} onClick={() => onPage(pageNumber - 1)}>上一页</button>
          <span>第 {pageNumber} 页</span>
          <button type="button" disabled={!page.has_more || loading} onClick={() => onPage(pageNumber + 1)}>下一页</button>
        </nav>
      )}
    </div>
  );
}
