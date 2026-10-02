import type { DraftWriteStatus } from "./sessionDraftStorage";
import "./draft-notice.css";

export default function DraftNotice({ status, restored, kind, hasDraft, onDiscard, enabled = true }: {
  status: DraftWriteStatus | "invalid";
  restored: boolean;
  kind: "正文" | "备注";
  hasDraft: boolean;
  onDiscard: () => void;
  enabled?: boolean;
}) {
  if (!enabled) return <p className="draftNotice">本地体验模式不暂存未提交{kind}，刷新或离开前请先复制备份。</p>;
  const failed = status === "unavailable" || status === "too_large" || status === "invalid";
  const description = status === "too_large"
    ? "最新内容超过本标签页暂存上限，仍可编辑。请先复制备份；刷新只能找回最后一次成功暂存的内容。"
    : status === "invalid"
      ? "已有暂存内容无法校验，未恢复。你仍可编辑，或丢弃这份暂存后重新输入。"
      : status === "unavailable"
        ? "浏览器暂存不可用，最新改动只在当前页面中。请先复制备份，刷新或重新登录可能丢失。"
        : status === "saved"
          ? `${restored ? `已恢复本标签页的未提交${kind}。` : `${kind}已在本标签页暂存。`}尚未${kind === "正文" ? "保存到项目" : "提交反馈"}。`
          : `未提交${kind}会在本标签页暂存，刷新或会话过期后仅原账户可恢复。`;
  return (
    <aside className={`draftNotice ${failed ? "draftNoticeWarning" : ""}`} aria-label={`${kind}暂存状态`}>
      <div>
        <p role="status" aria-live="polite">{description}</p>
        <small>仅本标签页暂存于本机浏览器，未加密；不是云端保存。主动退出会清理。不要在公共设备留下私人文稿。</small>
      </div>
      {(hasDraft || status === "invalid") && <button type="button" onClick={onDiscard}>丢弃暂存{kind}</button>}
    </aside>
  );
}
