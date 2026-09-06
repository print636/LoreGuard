import type {IssueEvidenceReviewView} from '../issueEvidenceReview';

export default function IssueEvidenceReview({review}:{review:IssueEvidenceReviewView}) {
  return <section className={`issueEvidenceReview ${review.tone}`} aria-label="AI 证据复核">
    <div className="issueEvidenceReviewHead">
      <strong>AI 复核 · {review.verdictLabel}</strong>
      <span>{review.retrievalModeLabel}</span>
    </div>
    <p>{review.note}</p>
    <div className="reviewCitations" aria-label="引用证据">
      {review.evidence.map(evidence=><span key={evidence.citationId}>
        <b>{evidence.citationId}</b> {evidence.documentLabel} · v{evidence.documentVersion} · 第 {evidence.lineStart}{evidence.lineEnd===evidence.lineStart?'':`–${evidence.lineEnd}`} 行
      </span>)}
    </div>
    <small>AI 注释，不会修改规则检测发现的问题、严重度或置信度。</small>
  </section>;
}
