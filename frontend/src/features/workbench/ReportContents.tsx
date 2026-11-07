import { revealWorkbenchTarget } from "./revealTarget";

type Props = {
  issues: Array<{ id: string; title: string }>;
  focusedIssue: string | null;
  onIssue: (id: string) => void;
};

function revealSection(id: string) {
  const section = document.getElementById(id);
  revealWorkbenchTarget(section);
  section?.focus({ preventScroll: true });
}

/** Navigation only: keep the report and its feedback drafts mounted. */
export default function ReportContents({ issues, focusedIssue, onIssue }: Props) {
  return <details className="storyReportContents" open>
    <summary>本页报告目录</summary>
    <nav aria-label="报告内容">
      <button type="button" onClick={() => revealSection("report-coverage")}>本次范围与覆盖</button>
      <button type="button" onClick={() => revealSection("report-formal-issues")}>正式问题</button>
      {issues.length > 0 && <ol>{issues.map(issue => <li key={issue.id}>
        <button type="button" aria-current={focusedIssue === issue.id ? "location" : undefined}
          onClick={() => onIssue(issue.id)}>{issue.title}</button>
      </li>)}</ol>}
      <button type="button" onClick={() => revealSection("report-pending")}>待复核线索</button>
      <button type="button" onClick={() => revealSection("report-questions")}>需要作者确认</button>
    </nav>
  </details>;
}
