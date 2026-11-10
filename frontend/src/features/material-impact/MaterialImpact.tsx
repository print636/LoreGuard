import { useEffect, useState, type ReactNode } from "react";
import { apiJson } from "../../api/client";
import { browserNavigate, workspacePath } from "../../routing";
import { revisionSearch } from "../../revisionWorkflow";
import { catalogMatchesRequest, runCatalogPath, runItemStatusNames, type RunCatalog } from "../run-history/runHistoryModel";
import { canPrepareRecheck, impactChangeNames, impactHeadline, sourceLabel, sourceContext, IMPACT_PAGE_SIZE, type ImpactPage } from "./materialImpactModel";
import { useMaterialImpact } from "./useMaterialImpact";
import { revealWorkbenchTarget } from "../workbench/revealTarget";
import "./material-impact.css";

function WorkspaceLink({ href, children }: { href: string; children: ReactNode }) {
  return <a href={href} onClick={event => {
    if (event.button !== 0 || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) return;
    event.preventDefault();
    if (browserNavigate(href) && href.includes("#")) requestAnimationFrame(() => revealWorkbenchTarget(document.getElementById(href.split("#")[1])));
  }}>{children}</a>;
}
export function recheckReason(reason: string | null | undefined): string {
  const names: Record<string, string> = {
    no_document_changes: "当前可用于复检的输入与原报告相同。未纳入的新稿可从审查页另行选择。",
    run_not_completed: "原任务未完成，不能作为复检起点。可从审查页准备新任务。",
    snapshot_unavailable: "原报告冻结输入不完整，暂不能据此复检。",
    logical_target_missing: "原目标文稿没有明确接续版本，请先核对资料。",
    missing_draft_review_targets: "原目标文稿没有明确接续版本，请先核对资料。",
    no_documents: "尚无可用于复检的资料。",
    current_sources_unavailable: "当前资料未能完整核对，请刷新后重试。",
    character_profile_snapshot_limit_exceeded: "当前角色档案超过本次可读取范围，暂不能准备复检。",
  };
  return names[reason || ""] || "当前资料暂不能复检，请先核对目标稿件与比对依据。";
}
function Pagination({ page, label, onOffset }: { page: ImpactPage<unknown>; label: string; onOffset: (n: number) => void }) {
  if (page.total <= IMPACT_PAGE_SIZE && page.offset === 0) return null;
  return <nav className="materialImpactPagination" aria-label={label}>
    <span>{page.items.length ? `${page.offset + 1}–${page.offset + page.items.length}` : "0"} / {page.total}</span>
    <button type="button" disabled={page.offset === 0} onClick={() => onOffset(Math.max(0, page.offset - IMPACT_PAGE_SIZE))}>上一页</button>
    <button type="button" disabled={!page.has_more} onClick={() => onOffset(page.offset + IMPACT_PAGE_SIZE)}>下一页</button>
  </nav>;
}
export function MaterialImpactView({ model, showPrepare = true, expanded = false }: {
  model: ReturnType<typeof useMaterialImpact>; showPrepare?: boolean; expanded?: boolean;
}) {
  const [detailsOpen, setDetailsOpen] = useState(expanded);
  const value = model.value;
  return <section className="materialImpact" aria-label="资料变化提示" aria-busy={model.loading}>
    <div className="materialImpactHeading"><h2>资料变化提示</h2><button type="button" disabled={model.loading} onClick={model.refresh}>重新核对</button></div>
    {model.loading ? <p role="status">正在对照这份报告的原输入与当前资料…</p>
      : model.error ? <p role="alert">{model.error}</p> : value && <>
        <p className="materialImpactHeadline">{impactHeadline(value)}</p>
        <p>这里按明确版本和原文引用提示可能影响，不代表原结论已错或问题已修复。旧报告、原文和反馈继续保留，已确认的角色特征不会自动撤销。</p>
        {value.confirmed_traits_changed === true && <p>已确认的角色设定与原报告使用的档案有变化，复检会重新核对当前档案。</p>}
        {value.summary.unresolved_evidence > 0 && <p className="materialImpactWarning">有 {value.summary.unresolved_evidence} 处旧引用未能建立可靠对应，影响范围尚未确定。</p>}
        {value.summary.unresolved_sources > 0 && <p className="materialImpactWarning">部分来源身份无法完整比较（{value.summary.unresolved_sources} 份），不能据此判断适用范围没有变化。</p>}
        <details open={detailsOpen} onToggle={event => setDetailsOpen(event.currentTarget.open)}>
          <summary>核对资料与引用（变化来源 {value.summary.changed_sources}，直接引用结果 {value.summary.affected_issues}）</summary>
          <p className="materialImpactBoundary">只列明确的输入与引用关系，未找到直接引用不等于没有其他影响。下方引用数不计入正式问题总数。</p>
          <ul className="materialImpactSources">
            {value.sources.items.map(item => <li key={item.key}>
              <strong>{item.kind === "confirmed_source" ? "已确认特征的原始来源" : item.kind === "added" ? "新增的复检输入" : item.kind === "not_included" ? "尚未纳入本次输入的资料" : "原报告输入"}</strong>
              <div className="materialImpactVersions">
                <div><small>原报告保存</small><b>{sourceLabel(item.frozen)}</b><span>{sourceContext(item.frozen)}</span></div>
                <div><small>当前资料</small><b>{sourceLabel(item.current)}</b><span>{sourceContext(item.current)}</span></div>
              </div>
              <p>{item.changes.length ? item.changes.map(c => impactChangeNames[c]).join("、") : "未发现明确变化"}
                {item.content_changed === false && item.changes.length > 0 && "；正文未变化"}</p>
              {item.kind === "confirmed_source" && <p>原始资料更新不会自动改写已确认特征；是否调整仍由作者确认。</p>}
              {item.frozen && !item.identity_comparison_available && <p>该来源的历史身份记录不完整，当前只展示可核对的版本与正文变化。</p>}
              {item.selected_for_recheck !== null && <small>{item.selected_for_recheck ? "在当前复检选材中" : "未在当前复检选材中"}</small>}
            </li>)}
          </ul>
          {!value.sources.items.length && <p>本页没有可展示的来源；来源缺失时不能据此判断没有影响。</p>}
          <Pagination page={value.sources} label="变化来源分页" onOffset={model.setOffset} />
          <h3>直接引用这些来源的结果</h3>
          {value.affected_issues.items.length ? <ul className="materialImpactReferences">{value.affected_issues.items.map(item => <li key={item.id}>
            <span>{item.report_class === "formal" ? "正式问题" : "待复核线索"}</span>
            <WorkspaceLink href={`${workspacePath("report", value.project_id, value.run_id)}?category=all&status=all${item.report_class === "formal" ? `&issue=${encodeURIComponent(item.id)}` : "#report-pending"}`}>{item.title || "查看原报告引用"}</WorkspaceLink>
            <small>引用来源：{item.source_keys.map(key => {
              const source = value.sources.items.find(source => source.key === key);
              return source ? sourceLabel(source.frozen) : "来源见其他分页";
            }).join("、")}</small>
            <small>原报告第 {item.evidence_indices.map(n => n + 1).join("、")} 处证据引用</small>
          </li>)}</ul> : <p>本页没有匹配的直接引用。资料变化仍可能影响尚未建立对应的内容。</p>}
          <Pagination page={value.affected_issues} label="直接引用结果分页" onOffset={model.setIssueOffset} />
        </details>
        <div className="materialImpactActions">
          {showPrepare && canPrepareRecheck(value) && <WorkspaceLink href={`${workspacePath("revision", value.project_id, value.run_id)}${revisionSearch({ step: "run", issueId: null, documentId: null, recheckRunId: null, outcome: "all", generateGraph: false, generateTimeline: false, page: 1 })}`}>准备复检</WorkspaceLink>}
          {!canPrepareRecheck(value) && <p>{recheckReason(value.recheck.reason)}</p>}
          <small>读取提示和进入准备页不会启动审查；确认范围后点击“开始复检”才可能使用模型额度。</small>
        </div>
      </>}
  </section>;
}
export function MaterialImpactPanel({ projectId, runId, refreshKey }: { projectId: string; runId: string; refreshKey: unknown }) {
  return <MaterialImpactView model={useMaterialImpact(projectId, runId, refreshKey)} />;
}

/** Only the selected report is queried; the catalog is not an impact total. */
export function MaterialImpactCatalog({ projectId, refreshKey }: { projectId: string; refreshKey: unknown }) {
  const [page, setPage] = useState(1);
  const [attempt, setAttempt] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const key = `${projectId}:${page}:${attempt}`;
  const [state, setState] = useState<{ key: string; refreshKey: unknown; catalog: RunCatalog | null; error: string } | null>(null);
  useEffect(() => { setPage(1); setSelected(null); }, [projectId]);
  useEffect(() => {
    if (!projectId) return;
    const controller = new AbortController();
    let alive = true;
    const timer = setTimeout(() => controller.abort(), 15_000);
    void apiJson<RunCatalog>(runCatalogPath(projectId, { page, status: "all" }), { signal: controller.signal }).then(catalog => {
      if (!catalogMatchesRequest(catalog, projectId, page)) throw new TypeError("报告目录不匹配");
      if (alive) setState({ key, refreshKey, catalog, error: "" });
    }).catch(() => { if (alive) setState({ key, refreshKey, catalog: null, error: "旧报告目录未能读完，请重试。" }); }).finally(() => clearTimeout(timer));
    return () => { alive = false; clearTimeout(timer); controller.abort(); };
  }, [projectId, page, key, refreshKey]);
  const current = state?.key === key && state.refreshKey === refreshKey ? state : null;
  const catalog = current?.catalog;
  if (!projectId) return null;
  return <section className="materialImpactCatalog" aria-label="资料变化与旧报告">
    <h2>资料变化与旧报告</h2><p>选一份旧报告，核对它使用的版本与当前资料。下方是审查记录，不是全部受影响报告的统计。</p>
    {!current ? <p role="status">正在读取旧报告目录…</p> : current.error ? <div role="alert"><p>{current.error}</p><button type="button" onClick={() => setAttempt(n => n + 1)}>重试读取报告目录</button></div>
      : catalog && <>
        {!catalog.items.length && <p>这一页没有审查记录。保存资料后可前往“文稿校验”准备首次审查。</p>}
        <ul>{catalog.items.map(item => <li key={item.id}>
          <div className="materialImpactRun"><span>{new Date(item.created_at).toLocaleString("zh-CN")} · {runItemStatusNames[item.status]}</span><small>{item.frozen_document_count} 份原输入</small>
            <button type="button" aria-expanded={selected === item.id} onClick={() => setSelected(selected === item.id ? null : item.id)}>{selected === item.id ? "收起变化" : "核对这份报告"}</button>
            <WorkspaceLink href={workspacePath("report", projectId, item.id)}>打开原报告</WorkspaceLink></div>
          {selected === item.id && <MaterialImpactPanel key={`${projectId}:${item.id}`} projectId={projectId} runId={item.id} refreshKey={refreshKey} />}
        </li>)}</ul>
        <nav className="materialImpactPagination" aria-label="旧报告目录分页"><span>第 {page} 页 · 共 {catalog.total} 条审查记录</span>
          <button type="button" disabled={page === 1} onClick={() => { setSelected(null); setPage(p => p - 1); }}>上一页报告</button>
          <button type="button" disabled={!catalog.has_more} onClick={() => { setSelected(null); setPage(p => p + 1); }}>下一页报告</button></nav>
      </>}
  </section>;
}
