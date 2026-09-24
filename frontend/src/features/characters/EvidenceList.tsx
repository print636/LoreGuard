import { authorityName } from "./presentation";
import type { ProfileEvidence } from "./types";

const documentRoleNames: Record<string, string> = {
  canon: "世界观设定",
  character_profile: "角色设定",
  chapter: "章节文本",
  reference: "参考资料",
};
const publicationNames: Record<string, string> = {
  draft: "草稿",
  in_review: "审核中",
  published: "已发布",
  retired: "已归档",
  unknown: "未标记",
};

type EvidenceListProps = {
  title: string;
  items: ProfileEvidence[];
  emptyText: string;
  tone?: "supporting" | "contrary";
  candidateEvidence?: boolean;
};

export default function EvidenceList({
  title,
  items,
  emptyText,
  tone = "supporting",
  candidateEvidence = false,
}: EvidenceListProps) {
  return (
    <section className={`characterEvidenceGroup ${tone}`} aria-label={title}>
      <div className="characterSubhead">
        <h4>{title}</h4>
        <span>{items.length} 条</span>
      </div>
      {items.length === 0 ? (
        <p className="characterInlineEmpty">{emptyText}</p>
      ) : (
        <ul className="characterEvidenceList">
          {items.map((evidence, index) => (
            <li
              key={`${evidence.document_id}:${evidence.document_version}:${evidence.line_start}:${index}`}
            >
              <div className="characterEvidenceMeta">
                <b>
                  {candidateEvidence && !evidence.source_verified ? "候选所存来源（未核对） · " : ""}
                  {evidence.document_name}
                  {evidence.document_version === null
                    ? " · 版本未提供"
                    : ` · v${evidence.document_version}`} · 第
                  {evidence.line_start}
                  {evidence.line_end !== evidence.line_start
                    ? `–${evidence.line_end}`
                    : ""}
                  行
                </b>
                {candidateEvidence && evidence.context_verified ? (
                  <>
                    <span>资料角色：{documentRoleNames[evidence.document_role] || evidence.document_role}</span>
                    <span>发布状态：{publicationNames[evidence.publication_status] || evidence.publication_status}</span>
                    <span>{authorityName(evidence.authority_level)}</span>
                    <span>冻结作用域：{evidence.story_scope.label}</span>
                  </>
                ) : !candidateEvidence ? (
                  <>
                    <span>{authorityName(evidence.authority_level)}</span>
                    <span>{evidence.story_scope?.label || "作用域未标注"}</span>
                  </>
                ) : null}
              </div>
              {candidateEvidence && (
                <p className="characterEvidenceLineStatus">
                  {!evidence.source_verified
                    ? "原文行无法与冻结输入核对；以下仅为候选所存文本，不可据此确认。"
                    : evidence.source_text_exact
                      ? "已逐字核对冻结输入的完整原文行；尚未定位到精确子句。"
                      : "原文行已按冻结坐标核对；旧候选未保留首尾空白。下方显示冻结原文完整行，尚未定位精确子句。"}
                </p>
              )}
              <blockquote>{evidence.text}</blockquote>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
