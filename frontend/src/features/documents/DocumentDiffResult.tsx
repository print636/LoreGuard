import { emptyDocumentDiffMessage, type DocumentDiff } from "./documentDiff";

export default function DocumentDiffResult({ result }: { result: DocumentDiff }) {
  return <div className="diffResult">
    <div className="diffSummary">
      <b>v{result.from_document.version} → v{result.to_document.version}</b>
      <span className="diffAdded">+{result.summary.added_lines}</span>
      <span className="diffRemoved">−{result.summary.removed_lines}</span>
      <span>{result.summary.changed_hunks} 个变更区块</span>
      <small>比较 {result.summary.compared_old_lines}/{result.summary.old_total_lines} → {result.summary.compared_new_lines}/{result.summary.new_total_lines} 行</small>
    </div>
    {result.warnings.map((warning, index) => <p className="visualWarning" key={index}>{warning}</p>)}
    {result.hunks.length === 0 ? <div className="visualEmpty">{emptyDocumentDiffMessage(result.summary)}</div>
      : <div className="diffHunks">{result.hunks.map((hunk, index) => <div className="diffHunk" key={index}>
        <div className="diffHeader">@@ -{hunk.old_start},{hunk.old_lines} +{hunk.new_start},{hunk.new_lines} @@</div>
        {hunk.lines.map((line, lineIndex) => <div className={`diffLine ${line.type}`} key={lineIndex}>
          <code>{line.old_line ?? ""}</code><code>{line.new_line ?? ""}</code>
          <b>{line.type === "added" ? "+" : line.type === "removed" ? "−" : " "}</b><pre>{line.content || " "}</pre>
        </div>)}
      </div>)}</div>}
  </div>;
}
