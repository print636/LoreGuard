import { useEffect, useRef, useState } from "react";
import {
  characterDimensionNames,
  profileOriginNames,
} from "./presentation";
import type {
  CandidatePage,
  CharacterDimension,
  CharacterProfileItem,
} from "./types";
import AxisAlignmentPanel from "./AxisAlignmentPanel";

const dimensions = Object.keys(characterDimensionNames) as CharacterDimension[];

export default function CharacterProfile({
  projectId,
  characterId,
  items,
  withdrawnPage,
  withdrawnPageNumber,
  withdrawnLoading,
  withdrawnError,
  withdrawBusyId,
  withdrawError,
  onWithdraw,
  onDismissWithdrawError,
  onWithdrawnPage,
  onRetryWithdrawn,
  onAligned,
}: {
  projectId: string;
  characterId: string;
  items: CharacterProfileItem[];
  withdrawnPage: CandidatePage | null;
  withdrawnPageNumber: number;
  withdrawnLoading: boolean;
  withdrawnError: string;
  withdrawBusyId: string | null;
  withdrawError: { id: string; message: string } | null;
  onWithdraw: (item: CharacterProfileItem) => void;
  onDismissWithdrawError: () => void;
  onWithdrawnPage: (page: number) => void;
  onRetryWithdrawn: () => void;
  onAligned: () => void;
}) {
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [alignId, setAlignId] = useState<string | null>(null);
  const cancelRef = useRef<HTMLButtonElement | null>(null);
  const alignTriggerRefs = useRef(new Map<string, HTMLButtonElement>());

  function closeAlignment(itemId: string) {
    setAlignId(null);
    requestAnimationFrame(() => alignTriggerRefs.current.get(itemId)?.focus());
  }

  useEffect(() => {
    if (confirmId && !items.some((item) => item.id === confirmId)) setConfirmId(null);
    if (alignId && !items.some((item) => item.id === alignId)) setAlignId(null);
  }, [confirmId, alignId, items]);

  useEffect(() => {
    if (confirmId) requestAnimationFrame(() => cancelRef.current?.focus());
  }, [confirmId]);

  return (
    <div className="characterProfileArchive">
      <div className="characterProfilePolicy">
        <p>已确认特征会用于之后新建的剧情审查；来源文档退役或改为参考资料，不会自动撤销这些特征。</p>
        <p>如设定不再生效，请逐条撤销或用新候选替换。历史运行报告始终保留原样。</p>
      </div>

      {items.length === 0 ? (
        <div className="characterSectionEmpty">
          <h3>还没有生效的角色特征</h3>
          <p>AI 归纳不会自动成为正式档案。请到“待确认归纳”逐条核对证据。</p>
        </div>
      ) : (
        <div className="characterProfileGroups">
          {dimensions.map((dimension) => {
            const rows = items.filter((item) => item.dimension === dimension);
            if (rows.length === 0) return null;
            return (
              <section key={dimension} aria-labelledby={`character-dimension-${dimension}`}>
                <div className="characterSubhead">
                  <h3 id={`character-dimension-${dimension}`}>
                    {characterDimensionNames[dimension]}
                  </h3>
                  <span>{rows.length} 条生效</span>
                </div>
                <ul>
                  {rows.map((item) => (
                    <li key={item.id}>
                      <p>{item.statement}</p>
                      <div className="characterProfileMetadata">
                        <span>生效中</span>
                        <span>{profileOriginNames[item.origin]}</span>
                        <span>{item.evidence_count} 条证据</span>
                        {item.dimension === "core_personality" && (
                          <span>{item.approved_axis_id ? "已绑定作者轴" : "未绑定作者轴"}</span>
                        )}
                        {item.dimension === "core_personality" && item.approved_axis_id && (
                          <span>{item.axis_alignment
                            ? `方向已补认：${item.axis_alignment === "same" ? "同向" : "反向"}`
                            : "方向待作者补认，暂不作同轴方向判定"}</span>
                        )}
                        {item.scopes.slice(0, 2).map((scope) => (
                          <span key={scope.scope_id}>{scope.label}</span>
                        ))}
                      </div>
                      {item.dimension === "core_personality" && item.approved_axis_id && (
                        <>
                          <button
                            type="button"
                            ref={(node) => {
                              if (node) alignTriggerRefs.current.set(item.id, node);
                              else alignTriggerRefs.current.delete(item.id);
                            }}
                            className="characterAxisAlignOpen"
                            disabled={withdrawBusyId !== null || item.revision === null}
                            aria-expanded={alignId === item.id}
                            onClick={() => { setConfirmId(null); if (alignId === item.id) closeAlignment(item.id); else setAlignId(item.id); }}
                          >{item.axis_alignment ? "查看作者轴方向与证据" : "核对并补认作者轴方向"}</button>
                          {alignId === item.id && (
                            <AxisAlignmentPanel
                              projectId={projectId}
                              characterId={characterId}
                              item={item}
                              onAligned={() => { closeAlignment(item.id); onAligned(); }}
                              onClose={() => closeAlignment(item.id)}
                            />
                          )}
                        </>
                      )}
                      {confirmId === item.id ? (
                        <div className="characterWithdrawConfirm" role="group" aria-label={`撤销“${item.statement}”`}>
                          <strong>确定撤销这条正式特征？</strong>
                          <p>撤销后，它不再参与之后新建的审查；已有运行报告不会改写。撤销记录会保留，不能在这里直接恢复。</p>
                          <div className="characterWithdrawActions">
                            <button
                              type="button"
                              className="characterWithdrawDanger"
                              disabled={withdrawBusyId !== null || item.revision === null}
                              onClick={() => onWithdraw(item)}
                            >
                              {withdrawBusyId === item.id ? "正在撤销…" : "确认撤销特征"}
                            </button>
                            <button
                              ref={cancelRef}
                              type="button"
                              disabled={withdrawBusyId !== null}
                              onClick={() => { setConfirmId(null); onDismissWithdrawError(); }}
                            >保留特征</button>
                          </div>
                        </div>
                      ) : (
                        <button
                          type="button"
                          className="characterWithdrawOpen"
                          disabled={withdrawBusyId !== null || item.revision === null}
                          aria-expanded={false}
                          onClick={() => { setAlignId(null); setConfirmId(item.id); }}
                        >撤销这条特征</button>
                      )}
                      {withdrawError?.id === item.id && (
                        <p className="characterWithdrawError" role="alert">{withdrawError.message}</p>
                      )}
                      {item.revision === null && (
                        <p className="characterWithdrawUnavailable">档案缺少并发审核版本，暂不能撤销。请刷新档案后重试。</p>
                      )}
                    </li>
                  ))}
                </ul>
              </section>
            );
          })}
        </div>
      )}

      <section className="characterWithdrawnArchive" aria-labelledby="character-withdrawn-title">
        <div className="characterSubhead">
          <h3 id="character-withdrawn-title">已撤销记录</h3>
          <span>{withdrawnPage ? `${withdrawnPage.total} 条` : ""}</span>
        </div>
        <p>这些特征已不再作为新审查的正式基线；旧报告仍保持原状。这里不提供隐式恢复。</p>
        {withdrawnError ? (
          <div className="characterPanelError" role="alert">
            <p>{withdrawnError}</p>
            <button type="button" onClick={onRetryWithdrawn}>重新读取撤销记录</button>
          </div>
        ) : withdrawnLoading ? (
          <p className="characterInlineLoading" aria-busy="true">正在读取撤销记录…</p>
        ) : !withdrawnPage || withdrawnPage.items.length === 0 ? (
          <p className="characterInlineEmpty">尚无撤销记录。</p>
        ) : (
          <>
            <ul>
              {withdrawnPage.items.map((candidate) => (
                <li key={candidate.id}>
                  <p>{candidate.statement}</p>
                  <div className="characterProfileMetadata">
                    <span>已撤销</span>
                    <span>{characterDimensionNames[candidate.dimension]}</span>
                    {candidate.scopes.slice(0, 2).map((scope) => (
                      <span key={scope.scope_id}>{scope.label}</span>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
            {(withdrawnPageNumber > 1 || withdrawnPage.has_more) && (
              <nav className="characterPagination" aria-label="已撤销记录分页">
                <button type="button" disabled={withdrawnPageNumber <= 1 || withdrawnLoading} onClick={() => onWithdrawnPage(withdrawnPageNumber - 1)}>上一页</button>
                <span>第 {withdrawnPageNumber} 页</span>
                <button type="button" disabled={!withdrawnPage.has_more || withdrawnLoading} onClick={() => onWithdrawnPage(withdrawnPageNumber + 1)}>下一页</button>
              </nav>
            )}
          </>
        )}
      </section>
    </div>
  );
}
