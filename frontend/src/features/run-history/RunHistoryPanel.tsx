import { useEffect, useRef, useState } from "react";
import { apiJson } from "../../api/client";
import { apiErrorDetail } from "../../app/session";
import { browserNavigate } from "../../routing";
import { shortIdentifier } from "../../runSnapshot";
import {
  catalogMatchesRequest, runCatalogAction, runCatalogPath, runHistorySearch, runHistoryStateFromSearch,
  runStatusNames, runItemStatusNames, runStatuses, type RunCatalog, type RunCatalogItem, type RunStatusFilter,
} from "./runHistoryModel";
import "./run-history.css";

type HistoryState = { key: string; status: "loading" | "ready" | "error"; data: RunCatalog | null; error: string };

export default function RunHistoryPanel({ projectId, selectedRunId, search, refreshKey, onOpen, disabled }: {
  projectId: string;
  selectedRunId: string;
  search: string;
  refreshKey: number;
  onOpen: (run: RunCatalogItem, signal?: AbortSignal) => Promise<void>;
  disabled: boolean;
}) {
  const criteria = runHistoryStateFromSearch(search);
  const key = JSON.stringify([projectId, criteria.page, criteria.status, refreshKey]);
  const [state, setState] = useState<HistoryState>({ key: "", status: "loading", data: null, error: "" });
  const [retry, setRetry] = useState(0);
  const [opening, setOpening] = useState("");
  const [openError, setOpenError] = useState("");
  const requestRef = useRef(0);
  const openingRef = useRef(0);
  const openingAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const request = ++requestRef.current;
    ++openingRef.current;
    openingAbortRef.current?.abort();
    setOpening("");
    setOpenError("");
    if (!projectId || disabled) return;
    const controller = new AbortController();
    setState({ key, status: "loading", data: null, error: "" });
    void apiJson<RunCatalog>(runCatalogPath(projectId, criteria), { signal: controller.signal }).then((data) => {
      if (controller.signal.aborted || request !== requestRef.current) return;
      if (!catalogMatchesRequest(data, projectId, criteria.page)) throw new Error("invalid catalog response");
      setState({ key, status: "ready", data, error: "" });
    }).catch((error) => {
      if (controller.signal.aborted || request !== requestRef.current) return;
      setState({ key, status: "error", data: null, error: apiErrorDetail(error) });
    });
    return () => { controller.abort(); openingAbortRef.current?.abort(); ++requestRef.current; ++openingRef.current; };
  }, [projectId, criteria.page, criteria.status, refreshKey, retry, disabled]);

  const current = state.key === key ? state : null;
  const data = current?.status === "ready" ? current.data : null;
  const loading = Boolean(projectId) && (disabled || !current || current.status === "loading");
  const totalPages = data ? Math.max(1, Math.ceil(data.total / data.page_size)) : 1;
  const outsidePage = data && selectedRunId && !data.items.some((row) => row.id === selectedRunId);

  function navigate(page: number, status: RunStatusFilter) {
    browserNavigate(`${window.location.pathname}${runHistorySearch(search, { page, status })}${window.location.hash}`);
  }

  async function open(row: RunCatalogItem) {
    if (opening) return;
    const request = ++openingRef.current;
    const controller = new AbortController();
    openingAbortRef.current = controller;
    setOpening(row.id);
    setOpenError("");
    try { await onOpen(row, controller.signal); }
    catch (error) { if (!controller.signal.aborted && request === openingRef.current) setOpenError(`暂时无法打开这次运行：${apiErrorDetail(error)} 请重新点击。`); }
    finally { if (request === openingRef.current) { setOpening(""); openingAbortRef.current = null; } }
  }

  return (
    <section className="runHistoryPanel" aria-label="运行历史">
      <div className="runHistoryHeader">
        <div><h3>运行历史（冻结输入）</h3><p>按创建时间倒序，每页 20 次。翻页和筛选不会改变当前打开的运行。</p></div>
        <label>运行状态<select value={criteria.status} disabled={!projectId || disabled} onChange={(event) => navigate(1, event.target.value as RunStatusFilter)}>
          {runStatuses.map((status) => <option key={status} value={status}>{runStatusNames[status]}</option>)}
        </select></label>
        <button type="button" disabled={!projectId || loading} onClick={() => setRetry((value) => value + 1)}>刷新历史</button>
      </div>
      <p className="runHistoryBoundary">列表仅展示已保存的输入数量与已记录用量，不证明模型实际参与或审查完整；请打开运行详情核对。输入数量也不代表可重试。</p>
      {outsidePage && <p className="runHistorySelectedNotice">当前打开的运行 {shortIdentifier(selectedRunId)} 不在本页中，仍保持选中。</p>}
      {openError && <p className="runHistoryError" role="alert">{openError}</p>}
      {!projectId ? <p className="runHistoryEmpty">选择项目后查看运行历史。</p> : loading ?
        <p className="runHistoryEmpty" role="status" aria-busy="true">正在读取这一页运行历史…</p> : current?.status === "error" ?
          <div className="runHistoryEmpty"><p className="runHistoryError" role="alert">运行历史读取失败：{current.error} 当前运行报告未改变。</p><button type="button" onClick={() => setRetry((value) => value + 1)}>重试读取历史</button></div> : data && data.items.length === 0 ?
            <div className="runHistoryEmpty">{data.total > 0 ? <><p>这一页已超出当前结果范围，共 {data.total} 次运行。</p><button type="button" onClick={() => navigate(totalPages, criteria.status)}>前往最后一页</button></> : <p>{criteria.status === "all" ? "这个项目还没有分析运行。返回文稿校验台，完成一次审查后可在这里查看。" : `没有${runStatusNames[criteria.status]}的运行。可切换为全部状态。`}</p>}</div> : data &&
              <table className="runHistoryTable">
                <caption className="runHistorySrOnly">本页运行记录</caption>
                <thead><tr><th>运行 / 时间</th><th>状态</th><th>冻结输入</th><th>已记录 Token</th><th>操作</th></tr></thead>
                <tbody>{data.items.map((row) => <tr key={row.id} className={row.id === selectedRunId ? "runHistorySelected" : ""}>
                  <th scope="row"><code title={row.id}>{shortIdentifier(row.id)}</code><small>{new Date(row.created_at).toLocaleString("zh-CN")}</small>{row.id === selectedRunId && <small className="runHistoryCurrent">当前打开</small>}{row.retried_from && <small>重试来源 {shortIdentifier(row.retried_from)}</small>}</th>
                  <td data-label="状态"><span className={`badge ${row.status}`}>{runItemStatusNames[row.status]}</span></td>
                  <td data-label="冻结输入"><span>{row.frozen_document_count} 份输入</span><small>{row.input_chars.toLocaleString("zh-CN")} 字符</small></td>
                  <td data-label="已记录 Token"><span>{(row.prompt_tokens + row.completion_tokens).toLocaleString("zh-CN")}</span><small>输入 {row.prompt_tokens.toLocaleString("zh-CN")} / 输出 {row.completion_tokens.toLocaleString("zh-CN")}</small></td>
                  <td data-label="操作"><button type="button" disabled={Boolean(opening)} onClick={() => void open(row)} aria-label={`${runCatalogAction(row.status)} ${shortIdentifier(row.id)}`}>{opening === row.id ? "正在打开…" : runCatalogAction(row.status)}</button></td>
                </tr>)}</tbody>
              </table>}
      {data && <nav className="runHistoryPager" aria-label="运行历史分页">
        <p role="status" aria-live="polite">{runStatusNames[criteria.status]}：共 {data.total} 次，第 {criteria.page} / {totalPages} 页</p>
        <div><button type="button" disabled={criteria.page <= 1 || loading} onClick={() => navigate(criteria.page - 1, criteria.status)}>上一页</button><button type="button" disabled={!data.has_more || loading} onClick={() => navigate(criteria.page + 1, criteria.status)}>下一页</button></div>
      </nav>}
    </section>
  );
}
