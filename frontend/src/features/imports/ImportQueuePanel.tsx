import { documentRoles, type DocumentRole } from "../../documentContext";
import { importCounts, type ImportDefaults, type ImportEntry, type ImportStatus } from "./importQueueModel";
import "./import-queue.css";

const labels: Record<ImportStatus, string> = { queued: "等待导入", uploading: "上传中", succeeded: "已导入", failed: "导入失败", unknown: "结果待核对" };
type Props = {
  entries: ImportEntry[];
  busy: boolean;
  notice: string;
  replacements?: { id: string; name: string; version: number }[];
  onEdit: (id: string, patch: Partial<ImportDefaults>) => void;
  onRemove: (id: string) => void;
  onClearCompleted: () => void;
  onRetry: (id: string) => void;
  onCheckDocuments?: () => void;
  checkDocumentsHref?: string;
};

export default function ImportQueuePanel({ entries, busy, notice, replacements, onEdit, onRemove, onClearCompleted, onRetry, onCheckDocuments, checkDocumentsHref }: Props) {
  const counts = importCounts(entries);
  if (entries.length === 0) return notice ? <p className="importQueueNotice" role="status">{notice}</p> : null;
  return <section className="importQueue" aria-label="文稿导入队列" aria-busy={busy}>
    <div className="importQueueHeading">
      <strong>逐文件导入</strong>
      <span role="status" aria-live="polite">已导入 {counts.succeeded}/{counts.total} 份{counts.unknown ? `，${counts.unknown} 份待核对` : ""}</span>
      {counts.succeeded > 0 && <button type="button" disabled={busy} onClick={onClearCompleted}>移除已导入项</button>}
    </div>
    <p className="importQueueHelp">文件仅保留在本页内存中，刷新、离开页面或会话过期后需要重新选择。未指定时，同名文稿沿用已有类型和作用域，新文稿按正文/全局导入；不会自动确认资料身份、启动分析或发布。</p>
    <ul>
      {entries.map((entry) => {
        const editable = !busy && (entry.status === "queued" || entry.status === "failed");
        return <li key={entry.id} className={`importQueueItem ${entry.status}`}>
          <div className="importQueueIdentity"><strong title={entry.file.name}>{entry.file.name}</strong><span>{labels[entry.status]}{entry.receipt ? ` · v${entry.receipt.version}` : ""}</span></div>
          <div className="importQueueFields">
            <label><span>文档类型</span><select aria-label={`${entry.file.name} 的文档类型`} value={entry.documentRole || ""} disabled={!editable} onChange={(event) => onEdit(entry.id, { documentRole: (event.target.value || null) as DocumentRole | null })}>
              <option value="">沿用默认类型</option>
              {documentRoles.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
            </select></label>
            <label><span>故事作用域</span><input aria-label={`${entry.file.name} 的故事作用域`} value={entry.storyScope} maxLength={80} disabled={!editable} placeholder="沿用默认作用域" onChange={(event) => onEdit(entry.id, { storyScope: event.target.value })} /></label>
            {replacements && <label><span>版本处理</span><select aria-label={`${entry.file.name} 的版本处理`} value={entry.replaceDocumentId} disabled={!editable} onChange={(event) => onEdit(entry.id, { replaceDocumentId: event.target.value })}>
              <option value="">同名自动新版本 / 新文件</option>
              {replacements.map((row) => <option key={row.id} value={row.id}>明确替换 {row.name} v{row.version}</option>)}
            </select></label>}
          </div>
          {entry.message && <p className="importQueueMessage" role={entry.status === "failed" || entry.status === "unknown" ? "alert" : undefined}>{entry.message}</p>}
          <div className="importQueueActions">
            {(entry.status === "failed" || entry.status === "unknown") && <button type="button" disabled={busy} onClick={() => onRetry(entry.id)}>{entry.status === "unknown" ? "安全重试同一导入" : "重试此文件"}</button>}
            {entry.status === "unknown" && (checkDocumentsHref
              ? <a href={checkDocumentsHref} target="_blank" rel="noopener noreferrer">在新标签页核对项目文档</a>
              : onCheckDocuments && <button type="button" disabled={busy} onClick={onCheckDocuments}>核对项目文档</button>)}
            {entry.status !== "uploading" && <button type="button" disabled={busy} onClick={() => {
              if (entry.status === "unknown" && !window.confirm("移出队列不会撤销服务器上可能已导入的文稿。请先核对项目文档；之后重新选择并上传会创建新的导入操作，可能产生重复版本。仍要移出吗？")) return;
              onRemove(entry.id);
            }}>{entry.status === "succeeded" || entry.status === "unknown" ? "移出队列" : "移除文件"}</button>}
          </div>
        </li>;
      })}
    </ul>
    {notice && <p className="importQueueNotice" role="status">{notice}</p>}
  </section>;
}
