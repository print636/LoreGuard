import { useMemo, useState } from "react";
import EvidenceReader from "../evidence/EvidenceReader";
import type { Evidence } from "../../types/visualization";
import type { EvidenceKind } from "../evidence/evidenceReaderModel";

/** Each visited source stays mounted so switching sides preserves its page and scroll. */
export default function EvidenceWorkspace({ projectId, runId, kind, itemId, evidence, onOpen }: {
  projectId: string; runId: string; kind: EvidenceKind; itemId: string; evidence: Evidence[];
  onOpen: (index: number, trigger: HTMLButtonElement) => void;
}) {
  const [selected, setSelected] = useState(0);
  const [visited, setVisited] = useState([0]);
  const sources = useMemo(() => evidence.map((expected, evidenceIndex) => ({ projectId, runId, kind, itemId, evidenceIndex, expected })),
    [projectId, runId, kind, itemId, evidence]);
  if (evidence.length === 0) return <p className="storyEvidenceMissing">这条结果没有可供展开的原文引用。</p>;
  return <section className="storyEvidenceWorkspace" aria-label="当前问题的原文工作区">
    <div className="storyEvidenceSources" role="group" aria-label="选择证据来源">
      {evidence.map((source, index) => <button type="button" key={index} aria-pressed={selected === index}
        onClick={() => { setSelected(index); setVisited(current => current.includes(index) ? current : [...current, index]); }}>
        <span>依据 {index + 1}</span><b>{source.document_name}</b><small>第 {source.line_start}{source.line_end === source.line_start ? "" : `–${source.line_end}`} 行</small>
      </button>)}
    </div>
    {visited.map(index => evidence[index] && <div key={index} hidden={index !== selected}>
      <EvidenceReader presentation="inline" selection={sources[index]} />
    </div>)}
    <button type="button" className="storyOpenReader" onClick={event => onOpen(selected, event.currentTarget)}>单独阅读这份原文</button>
  </section>;
}
