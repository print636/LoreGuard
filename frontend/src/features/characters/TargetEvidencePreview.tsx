import { verifiedTargetExcerpts } from "./supportBindings";
import type { ProfileCandidate } from "./types";

type Props = { candidate: ProfileCandidate };

export default function TargetEvidencePreview({ candidate }: Props) {
  const excerpts = verifiedTargetExcerpts(candidate);
  const visible = excerpts.slice(0, 3);
  return (
    <section className="candidateTargetPreview" aria-label="冻结原文目标句">
      <h4>先看原文目标句</h4>
      {visible.length ? (
        <>
          <ol>
            {visible.map((item, index) => (
              <li key={`${item.location}:${index}`}>
                <small>{item.location}</small>
                <blockquote>{item.target}</blockquote>
                {item.context.length > 0 && (
                  <dl>
                    {item.context.map((part, partIndex) => (
                      <div key={`${part.label}:${partIndex}`}>
                        <dt>{part.label}</dt><dd>{part.text}</dd>
                      </div>
                    ))}
                  </dl>
                )}
              </li>
            ))}
          </ol>
          {excerpts.length > visible.length && (
            <p>另有 {excerpts.length - visible.length} 处目标句，请查看下方完整证据。</p>
          )}
          <p>摘录来自已核对的冻结原文；位置正确不等于归纳已被作者确认。</p>
        </>
      ) : (
        <p className="candidateTargetUnavailable">
          {candidate.support_bindings_status === "invalid"
            ? "目标句定位核验失败，不能据此确认；请重新分析。"
            : "本候选尚无可展示的精确目标句。请核对下方完整原文行与条件，不能只凭模型归纳标题判断。"}
        </p>
      )}
    </section>
  );
}
