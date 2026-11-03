import { useEffect, useId, useState } from "react";
import { apiJson } from "../../api/client";
import type { SessionIdentity } from "../../app/session";
import {
  OBSERVATION_PAGE_SIZE, observationRecordingCopy, observationScope, startObservationRequest,
  type FrozenExcerpt, type ObservationLoadState,
} from "./observationModel";
import "./character-observations.css";

function FrozenQuote({ source }: { source: FrozenExcerpt }) {
  return <blockquote className="observationQuote">
    <p>{source.text}</p>
    <footer>{source.documentName} · 冻结版本 {source.version} · 第 {source.lineStart}{source.lineEnd === source.lineStart ? "" : `–${source.lineEnd}`} 行</footer>
  </blockquote>;
}

function ObservationContents({ projectId, runId, runStatus }: { projectId: string; runId: string; runStatus: string }) {
  const [offset, setOffset] = useState(0);
  const [retry, setRetry] = useState(0);
  const [state, setState] = useState<ObservationLoadState>({ status: "loading", data: null });
  const [loadedKey, setLoadedKey] = useState("");
  const prefix = useId();
  const key = JSON.stringify([projectId, runId, offset, retry, runStatus]);
  useEffect(() => startObservationRequest({ projectId, runId, offset },
    (path, signal) => apiJson(path, { signal }),
    (next) => { setLoadedKey(key); setState(next); }), [projectId, runId, offset, retry, runStatus, key]);
  const current = loadedKey === key ? state : null;
  const data = current?.status === "ready" ? current.data : null;
  const loading = !current || current.status === "loading";
  const recording = data ? observationRecordingCopy(data.status, data.total) : null;
  return <div aria-busy={loading}>
    {loading ? <p className="observationEmpty" role="status">正在读取本次角色行为记录…</p> : current?.status === "error" ?
      <div className="observationEmpty"><p role="alert">角色行为记录读取失败，当前无法确认记录状态或数量。请检查连接后重试。</p><button type="button" onClick={() => setRetry((value) => value + 1)}>重试读取记录</button></div> : data && recording && <>
        <div className={`observationRecording ${recording.tone}`}>
          <div><strong>{recording.label}</strong><p>{recording.detail}</p></div>
          <button type="button" onClick={() => setRetry((value) => value + 1)}>刷新记录</button>
        </div>
        {data.coverage && <div className="observationNote">
          <p>另有 {data.coverage.omittedBaselineCount} 条设定的关联明细未完整记录。</p>
          <ul className="observationReasons">
            {data.coverage.notEvaluatedCount > 0 && <li>未保存逐条关联明细：{data.coverage.notEvaluatedCount} 条设定</li>}
            {data.coverage.reasons.map((reason) => <li key={reason.explanation}>{reason.explanation}：{reason.count} 条设定</li>)}
          </ul>
          <p>以上说明记录保存情况；实际审查范围和结论请查看报告中的审查说明。</p>
        </div>}
        {data.items.length === 0 && data.total > 0 ? <div className="observationEmpty"><p>这一页已超出当前记录范围。</p><button type="button" onClick={() => setOffset(0)}>返回第一页</button></div> : data.items.length > 0 &&
          <ol className="observationList" aria-label="已接纳的角色行为与陈述">{data.items.map((item, index) => {
            const matched = item.matches.filter((match) => match.status === "matched" && match.baseline);
            const reasons = [...new Set(item.matches.filter((match) => match.status !== "matched").map((match) => match.explanation))];
            const detailId = `${prefix}-source-${index}`;
            return <li key={`${offset}:${item.ordinal}`}>
              <article className="observationItem">
                <div className="observationItemHeading"><h3>{item.character}</h3><span>{item.dimension}</span><span className="observationAssociation">{matched.length ? "已关联冻结设定" : "尚未关联设定"}</span></div>
                <p className="observationStatement">{item.statement}</p>
                <p className="observationLocation">{item.source.documentName} · 冻结版本 {item.source.version} · 第 {item.source.lineStart}{item.source.lineEnd === item.source.lineStart ? "" : `–${item.source.lineEnd}`} 行</p>
                <details>
                  <summary aria-controls={detailId}>查看原文与设定关联<span className="observationSrOnly">：{item.character}，本页第 {index + 1} 条</span></summary>
                  <div id={detailId} className="observationDetail">
                    <h4>对应原文</h4><FrozenQuote source={item.source} />
                    {item.context && <p>当时情境：{item.context}</p>}
                    <h4>本次冻结的设定关联</h4>
                    {matched.length > 0 && <><p className="observationNote">关联仅表示该设定被选入比较，不代表行为与设定一致，也不代表存在角色偏离。</p>
                      <ul className="observationBaselines">{matched.map((match, matchIndex) => <li key={matchIndex}>
                        <p><strong>{match.baseline!.character}</strong> · {match.baseline!.dimension}</p><p>{match.baseline!.statement}</p>
                        {match.baseline!.evidence.map((source, evidenceIndex) => <FrozenQuote key={evidenceIndex} source={source} />)}
                      </li>)}</ul></>}
                    {reasons.length > 0 && <><p className="observationNote">{matched.length ? "其他设定未关联的原因：" : "本次尚未关联到可比较的冻结设定："}</p><ul className="observationReasons">{reasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></>}
                    {!matched.length && <p className="observationNote">未关联的普通行为仍是观察记录，不会因此成为待复核线索或正式问题。</p>}
                  </div>
                </details>
              </article>
            </li>;
          })}</ol>}
        {data.total > 0 && <nav className="observationPager" aria-label="角色行为记录分页">
          <p role="status" aria-live="polite">已保存 {data.total} 条{data.status !== "complete" ? "，记录可能不完整" : ""}；第 {Math.floor(offset / OBSERVATION_PAGE_SIZE) + 1} / {Math.max(1, Math.ceil(data.total / OBSERVATION_PAGE_SIZE))} 页</p>
          <div><button type="button" disabled={offset === 0} onClick={() => setOffset((value) => Math.max(0, value - OBSERVATION_PAGE_SIZE))}>上一页</button><button type="button" disabled={!data.hasMore} onClick={() => setOffset((value) => value + OBSERVATION_PAGE_SIZE)}>下一页</button></div>
        </nav>}
      </>}
  </div>;
}

export default function CharacterObservationsPanel({ identity, projectId, runId, runStatus, enabled }: {
  identity: SessionIdentity; projectId: string; runId: string; runStatus: string; enabled: boolean;
}) {
  const scope = observationScope(identity, projectId, runId, enabled);
  return <section className="characterObservations workspaceView" aria-label="角色行为记录">
    <div className="sectionHead"><h2>角色行为记录</h2><span className="observationReadOnly">本次审查记录</span></div>
    <p className="observationBoundary">查看系统从本次文稿中识别出的角色行为、原文及设定关联。行为记录单独展示，不计入正式问题或待复核线索。</p>
    {scope ? <ObservationContents key={scope} projectId={projectId} runId={runId} runStatus={runStatus} /> : <p className="observationEmpty">{identity.mode !== "required" ? "登录后可查看本次运行保存的角色行为记录。" : "打开一份可访问的运行报告后，可查看当时保存的角色行为记录。"}</p>}
  </section>;
}
