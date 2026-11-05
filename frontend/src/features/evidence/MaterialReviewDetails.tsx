import type { MaterialReview } from "../../reviewClues";
import type { Evidence } from "../../types/visualization";
import { referenceRange } from "./evidenceReaderModel";

const opinions = {
  contradicts: "模型认为这件事与已确认设定相悖。",
  explained: "模型认为这件事可得到解释，或不适合直接与这条设定比较。",
  needs_confirmation: "模型认为这件事仍有歧义，需要作者确认。",
  insufficient_evidence: "模型未取得足够依据来判断这件事。",
};
const reasons = {
  actor_unverified: "行为是否属于这个角色尚未核实",
  statement_unverified: "原文能否支持提取出的含义尚未核实",
  object_unverified: "动作涉及的对象尚未核实",
  source_unverified: "候选材料的原文依据尚未核实",
  record_unverified: "这部分候选材料尚未通过核验",
  response_unavailable: "这部分核验未取得可用结果",
  processing_incomplete: "这部分材料尚未处理完整",
};

type Props = {
  review: MaterialReview;
  evidence: Evidence[];
  onOpen: (kind: "review_clue" | "material_gap", index: number, span: Evidence, trigger: HTMLButtonElement) => void;
};

export default function MaterialReviewDetails({ review, evidence, onOpen }: Props) {
  const located = review.gaps.filter(gap => gap.location === "located");
  const unknown = review.gaps.filter(gap => gap.location === "unknown");
  const unknownReasons = [...new Set(unknown.map(gap => reasons[gap.reason]))];
  return (
    <div className="materialReviewDetails">
      <h5>本次比较</h5>
      <ul className="materialComparisonList">
        {review.comparisons.map((comparison, index) => (
          <li key={index}>
            <p>{opinions[comparison.opinion]}</p>
            {comparison.evidence_indices.map(i => (
              <button key={i} type="button" className="evidenceReaderTrigger"
                onClick={event => onOpen("review_clue", i, evidence[i], event.currentTarget)}>
                {evidence[i].document_name} · {referenceRange(evidence[i])} · 查看分析时原文
              </button>
            ))}
          </li>
        ))}
      </ul>
      <h5>仍待复核的原因</h5>
      <p>仍有 {review.total_count} 项材料核验未完成，尚不能排除对本次比较的影响，整体结论保留为待复核。</p>
      {located.length > 0 && <p>可定位的待核清材料共 {located.length} 处，展开可对照原文。</p>}
      {located.map(gap => (
        <details key={gap.index} className="materialGap">
          <summary>{gap.evidence!.document_name} · {referenceRange(gap.evidence!)} · {reasons[gap.reason]}{gap.count > 1 ? `（${gap.count} 项）` : ""}</summary>
          <p>{gap.relation === "elsewhere"
            ? "另一个原文位置仍未核清，尚未排除它对本次比较的影响。"
            : "待核清材料与本次比较的引用范围重叠，仍需核实。"}摘录可定位，提取出的含义仍待核验。</p>
          <blockquote>
            <span>{gap.evidence!.text}</span>
            <button type="button" className="evidenceReaderTrigger"
              onClick={event => onOpen("material_gap", gap.index, gap.evidence!, event.currentTarget)}>
              查看分析时原文
            </button>
          </blockquote>
        </details>
      ))}
      {unknown.length > 0 && <p>其中 {unknown.reduce((sum, gap) => sum + gap.count, 0)} 项未能定位到具体原文。{unknownReasons.join("。")}。</p>}
      {review.omitted_count > 0 && <p>另有 {review.omitted_count} 项缺口未逐条保存，本页仅展示已保存的部分。</p>}
    </div>
  );
}
