import { authorityName } from "./presentation";
import type { ProfileEvidence } from "./types";

type EvidenceListProps = {
  title: string;
  items: ProfileEvidence[];
  emptyText: string;
  tone?: "supporting" | "contrary";
};

export default function EvidenceList({
  title,
  items,
  emptyText,
  tone = "supporting",
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
                <span>{authorityName(evidence.authority_level)}</span>
                <span>{evidence.story_scope?.label || "作用域未标注"}</span>
              </div>
              <blockquote>{evidence.text}</blockquote>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
