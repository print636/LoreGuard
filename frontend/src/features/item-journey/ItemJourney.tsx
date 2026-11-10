import { useEffect, useMemo, useRef, useState } from "react";
import { apiJson } from "../../api/client";
import { reviewTimeLabel } from "../../productLanguage";
import EvidenceReader from "../evidence/EvidenceReader";
import { MaterialImpactPanel } from "../material-impact/MaterialImpact";
import { revealWorkbenchTarget } from "../workbench/revealTarget";
import {
  EVENT_PAGE_SIZE, ITEM_PAGE_SIZE, itemJourneysPath, itemJourneyPath, journeyEventLabel, journeyKindNames,
  journeyRoute, journeySearch, normalizeJourneyList, normalizeJourneyDetail,
  type JourneyList, type JourneyDetail, type JourneyEvidence, type JourneyEvent,
} from "./itemJourneyModel";
import "./item-journey.css";

function useJourneyRequest<T>(path: string | null, projectId: string, parse: (raw: unknown) => T) {
  const [retry, setRetry] = useState(0);
  const key = JSON.stringify([projectId, path, retry]);
  const [state, setState] = useState<{ key: string; value: T | null; error: string } | null>(null);
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    let alive = true;
    const timeout = setTimeout(() => controller.abort(), 15_000);
    void apiJson<unknown>(path, { signal: controller.signal }).then(raw => {
      const value = parse(raw);
      if (alive) setState({ key, value, error: "" });
    }).catch(() => { if (alive) setState({ key, value: null, error: "暂时无法读取所选报告的物品记录，请重试。页面没有改用其他报告或当前文稿。" }); }).finally(() => clearTimeout(timeout));
    return () => { alive = false; clearTimeout(timeout); controller.abort(); };
  }, [key, path, projectId]);
  const current = path && state?.key === key ? state : null;
  return { value: current?.value || null, error: current?.error || "", loading: !!path && !current, retry: () => setRetry(n => n + 1) };
}
function FrozenRecord({ source, projectId, runId }: { source: JourneyEvidence; projectId: string; runId: string }) {
  const selection = useMemo(() => source.evidence ? ({ projectId, runId, kind: "record" as const, itemId: source.record_id, evidenceIndex: 0,
    expected: source.evidence, ...(source.document_version !== null ? { expectedVersion: source.document_version } : {}) }) : null, [projectId, runId, source]);
  return selection ? <EvidenceReader presentation="inline" selection={selection} /> : <p>这条记录的冻结来源尚未核清，不能用当前文稿代替。</p>;
}
function EventNotes({ event }: { event: JourneyEvent }) {
  return <>
    {event.kind === "observed_possession" && <small>只说明此处看到持有，不代表交接过程已核实。</small>}
    {event.kind === "uses" && <small>使用不等于保管、所有权或已经发生转交。</small>}
    {event.kind === "retained_custody" && <small>{event.retention_scope === "bounded" ? `明确未交接的范围截至 ${event.through_time || "终点未明确"}` : event.retention_scope === "ongoing" ? "正文记载持续保管，未写明终点；不能延伸为永久持有。" : "持续保管的适用范围尚未核清。"}</small>}
    {event.kind === "legacy" && <small>历史记录未区分持有方式，不能据此补出交接或持续状态。</small>}
    {(event.semantic_status !== "supported" || event.basis === "unverified") && <small className="itemJourneyWarning">这条记录的语义依据尚未核清。</small>}
    {event.basis === "supported_inference" && <small>依据上下文推断，请结合原文核对。</small>}
  </>;
}
function JourneyDetailView({ projectId, runId, itemKey, focusTrigger }: { projectId: string; runId: string; itemKey: string; focusTrigger: HTMLElement | null }) {
  const [offset, setOffset] = useState(0);
  const model = useJourneyRequest<JourneyDetail>(itemJourneyPath(runId, itemKey, offset), projectId,
    raw => normalizeJourneyDetail(raw, projectId, runId, itemKey, offset));
  const [selected, setSelected] = useState<string | null>(null);
  const [visited, setVisited] = useState<JourneyEvidence[]>([]);
  const [readingOnMobile, setReadingOnMobile] = useState(false);
  const readerTitle = useRef<HTMLHeadingElement>(null);
  const eventTitle = useRef<HTMLHeadingElement>(null);
  const focusFrame = useRef<number | null>(null);
  const detailTitle = useRef<HTMLHeadingElement>(null);
  const initialTrigger = useRef(focusTrigger);
  useEffect(() => () => { if (focusFrame.current !== null) cancelAnimationFrame(focusFrame.current); }, []);
  useEffect(() => {
    const trigger = initialTrigger.current;
    if (!model.value || !trigger) return;
    initialTrigger.current = null;
    focusFrame.current = requestAnimationFrame(() => {
      focusFrame.current = null;
      if (document.activeElement !== trigger && document.activeElement !== document.body) return;
      detailTitle.current?.focus({ preventScroll: true }); revealWorkbenchTarget(detailTitle.current, "nearest");
    });
  }, [model.value]);
  function focusAfterChange(target: "reader" | "events") {
    const previous = document.activeElement;
    if (focusFrame.current !== null) cancelAnimationFrame(focusFrame.current);
    focusFrame.current = requestAnimationFrame(() => {
      focusFrame.current = null;
      if (document.activeElement !== previous && document.activeElement !== document.body) return;
      const heading = target === "reader" ? readerTitle.current : eventTitle.current;
      heading?.focus({ preventScroll: true }); revealWorkbenchTarget(heading, "nearest");
    });
  }
  function read(source: JourneyEvidence) {
    if (!source.evidence) return;
    setSelected(source.record_id);
    setVisited(previous => previous.some(item => item.record_id === source.record_id) ? previous : [...previous.slice(-7), source]);
    setReadingOnMobile(true); focusAfterChange("reader");
  }
  const data = model.value;
  const relations = data?.relations.filter(r => data.events.items.some(e => e.record_id === r.left_record_id || e.record_id === r.right_record_id)) || [];
  const relationName = (id: string) => {
    const event = data?.events.items.find(e => e.record_id === id);
    return event ? `${event.time || "时间未确定"} · ${journeyEventLabel(event)}` : "其他页的记录";
  };
  if (model.loading) return <p role="status">正在核对这件物品的原文记录…</p>;
  if (model.error || !data) return <div role="alert"><p>{model.error}</p><button type="button" onClick={model.retry}>重试读取物品历程</button></div>;
  return <section className="itemJourneyDetail" aria-label="所选物品历程">
    <div className="itemJourneySummary"><h3 tabIndex={-1} ref={detailTitle}>{data.item.name}</h3><span>故事范围：{data.item.story_scope || "未记录"}</span>
      <p>{data.summary.text}</p><small>这是所选报告里有出处的记载，不代表整部作品此刻的所有者。</small>
    </div>
    <div className="itemJourneyLimits" role="status">
      {data.coverage.run_partial && <p>原审查覆盖不完整，查询不会补做审查或把未评估部分视为通过。</p>}
      {data.coverage.truncated && <p>本次仅整理 {data.coverage.processed_records} / {data.coverage.total_records} 条记录，不能据此确定历程末端。</p>}
      {data.coverage.unresolved_time_pairs > 0 && <p>{data.coverage.unresolved_time_pairs} 组记录的时间先后未能确定；下方排列不代表完整剧情顺序。</p>}
      {(data.coverage.source_unavailable > 0 || data.coverage.proof_unavailable > 0 || data.coverage.relation_proof_unavailable > 0) && <p>部分原文或语义、时间关系依据尚未核清，请以每条记录的限制提示为准。</p>}
      {data.summary.status === "unavailable" && <p>现有依据不足以形成持有或保管人的摘要。</p>}
    </div>
    <div className={`itemJourneyReading ${readingOnMobile ? "readingOnMobile" : ""}`}>
      <div className="itemJourneyEvents">
        <h4 tabIndex={-1} ref={eventTitle}>有出处的历程记录</h4>
        <p>仅已列出的时间关系可用于判断先后；记录相邻不表示发生过交接。</p>
        <ol>{data.events.items.map(event => <li key={event.record_id}>
          <span className="itemJourneyKind">{journeyKindNames[event.kind]}</span><time>{event.time || "时间未确定"}</time>
          <h5>{journeyEventLabel(event)}</h5>
          <EventNotes event={event} />
          {data.summary.latest_record_ids.includes(event.record_id) && <small>本次摘要的引用依据</small>}
          {event.evidence ? <><p className="itemJourneyCitation">{event.evidence.document_name} · {event.document_version === null ? "版本待核对" : `v${event.document_version}`} · 第 {event.evidence.line_start}{event.evidence.line_end === event.evidence.line_start ? "" : `–${event.evidence.line_end}`} 行</p>
            <blockquote>{event.evidence.text}</blockquote></> : <p>冻结来源未能核实，暂不展示原文。</p>}
          <button type="button" disabled={!event.evidence || event.source_status !== "verified"} aria-pressed={selected === event.record_id} onClick={() => read(event)}>阅读这条记录的原文</button>
        </li>)}</ol>
        {!data.events.items.length && <p>本页没有可列出的物品记录，不能据此断言物品没有转交。</p>}
        <nav className="itemJourneyPagination" aria-label="物品记录分页"><span>{data.events.items.length ? `${offset + 1}–${offset + data.events.items.length}` : "0"} / {data.events.total} 条</span>
          <button type="button" disabled={offset === 0} onClick={() => setOffset(n => Math.max(0, n - EVENT_PAGE_SIZE))}>上一页记录</button>
          <button type="button" disabled={!data.events.has_more} onClick={() => setOffset(n => n + EVENT_PAGE_SIZE)}>下一页记录</button></nav>
        {relations.length > 0 && <details className="itemJourneyRelations"><summary>本页已核对的时间关系（{relations.length}）</summary><ul>{relations.map((relation, index) => <li key={index}>
          <p>{relationName(relation.left_record_id)}<b>{({ before: "早于", same_time: "与其同时", after: "晚于" })[relation.order]}</b>{relationName(relation.right_record_id)}</p>
          {relation.evidence_record_ids.map(id => { const source = data.relation_evidence.find(e => e.record_id === id);
            return source ? <button type="button" key={id} disabled={!source.evidence} onClick={() => read(source)}>阅读时间关系依据{source.evidence ? `：${source.evidence.document_name} 第 ${source.evidence.line_start} 行` : "（来源未核清）"}</button> : null;
          })}</li>)}</ul></details>}
      </div>
      <section className="itemJourneyReader" aria-label="物品记录冻结原文">
        <button className="itemJourneyBack" type="button" onClick={() => { setReadingOnMobile(false); focusAfterChange("events"); }}>返回物品历程</button>
        <h4 tabIndex={-1} ref={readerTitle}>记录所依据的原文</h4>
        {!selected && <p>选择一条记录阅读分析时的完整原文，包括没有形成正式问题的记录。</p>}
        {visited.map(source => <div key={source.record_id} hidden={source.record_id !== selected}><FrozenRecord source={source} projectId={projectId} runId={runId} /></div>)}
      </section>
    </div>
  </section>;
}

export default function ItemJourney({ projectId, runId, status, createdAt, search, refreshKey, onRouteChange, onChooseRun }: {
  projectId: string; runId: string; status: string | undefined; createdAt: string | undefined; search: string; refreshKey: unknown;
  onRouteChange: (search: string, replace?: boolean) => void; onChooseRun: () => void;
}) {
  const selected = journeyRoute(search);
  const [query, setQuery] = useState("");
  const [applied, setApplied] = useState("");
  const [offset, setOffset] = useState(0);
  const selectionTrigger = useRef<HTMLElement | null>(null);
  const enabled = !!projectId && !!runId && status === "completed";
  const model = useJourneyRequest<JourneyList>(enabled ? itemJourneysPath(runId, applied, offset, selected.issue) : null, projectId,
    raw => normalizeJourneyList(raw, projectId, runId, offset));
  const list = model.value;
  useEffect(() => {
    if (selected.issue && !selected.item && list?.items.length === 1 && !list.page.has_more) onRouteChange(journeySearch(list.items[0].key, selected.issue), true);
  }, [selected.issue, selected.item, list]);
  return <div className="itemJourney">
    <div className="itemJourneyContext"><p><b>物品去向</b> · {runId ? `所选审查：${reviewTimeLabel(createdAt)}` : "尚未选择审查报告"}</p><button type="button" onClick={onChooseRun}>从历史任务选择报告</button></div>
    <p>在这一份报告中查看谁拿着、何时转交和原文依据。读取已保存记录，不会启动新的模型审查。</p>
    {!enabled ? <p role="status">请先选择一份已完成的报告。失败、取消或进行中的任务不能当作完整物品历程。</p> : <>
      <MaterialImpactPanel projectId={projectId} runId={runId} refreshKey={refreshKey} />
      <div className="itemJourneyWorkspace">
        <section className="itemJourneyCatalog" aria-label="物品目录"><h3>选择物品</h3>
          <form onSubmit={event => { event.preventDefault(); setOffset(0); setApplied(query.trim()); }}><label>查找物品<input type="search" maxLength={100} value={query} onChange={event => setQuery(event.target.value)} /></label><button type="submit">查找</button></form>
          <p>按所引物品名称和故事范围分组；同名记录不自动证明是同一件实物。</p>
          {selected.issue && <p>仅查看这个问题关联的物品。<button type="button" onClick={() => { setOffset(0); onRouteChange(journeySearch(null)); }}>查看全部物品</button></p>}
          {model.loading ? <p role="status">正在读取物品目录…</p> : model.error ? <div role="alert"><p>{model.error}</p><button type="button" onClick={model.retry}>重试读取物品目录</button></div> : list && <>
            {list.coverage.truncated && <p className="itemJourneyWarning">目录只整理了部分记录，不是本次所有物品的完整清单。</p>}
            <ul>{list.items.map(item => <li key={item.key}><button type="button" aria-pressed={selected.item === item.key} onClick={event => { selectionTrigger.current = event.currentTarget; onRouteChange(journeySearch(item.key, selected.issue)); }}><b>{item.name}</b><small>故事范围：{item.story_scope || "未记录"} · {item.record_count} 条记录</small></button></li>)}</ul>
            {!list.items.length && <p>{applied ? "未找到匹配物品，可换个原文名称查找。" : "这份报告没有可整理的物品记录；不代表故事中没有物品或交接。"}</p>}
            <nav className="itemJourneyPagination" aria-label="物品目录分页"><span>{list.items.length ? `${offset + 1}–${offset + list.items.length}` : "0"} / {list.page.total} 项</span><button type="button" disabled={!offset} onClick={() => setOffset(n => Math.max(0, n - ITEM_PAGE_SIZE))}>上一页物品</button><button type="button" disabled={!list.page.has_more} onClick={() => setOffset(n => n + ITEM_PAGE_SIZE)}>下一页物品</button></nav>
          </>}
        </section>
        {selected.item ? <JourneyDetailView key={`${projectId}:${runId}:${selected.item}`} projectId={projectId} runId={runId} itemKey={selected.item} focusTrigger={selectionTrigger.current} /> : <p className="itemJourneyEmpty">从目录选一件物品，查看摘要、历程和原文。</p>}
      </div>
    </>}
  </div>;
}
