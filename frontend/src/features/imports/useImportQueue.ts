import { useEffect, useRef, useState } from "react";
import { apiJson } from "../../api/client";
import { registerBrowserNavigationBlocker } from "../../routing";
import {
  appendImportEntries, emptyImportDefaults, importBatchEntries, importContextFields,
  importCounts, importFailure, ImportResultUnknown, updateImportEntry, verifiedImportReceipt,
  type ImportDefaults, type ImportEntry,
} from "./importQueueModel";

type QueueState = { entries: ImportEntry[]; busy: boolean; notice: string };
const emptyQueue: QueueState = { entries: [], busy: false, notice: "" };

/** File objects live only in this mounted page; queues are never browser-persisted. */
export function useImportQueue(scope: string) {
  const [states, setStates] = useState<Record<string, QueueState>>({});
  const statesRef = useRef(states);
  const alive = useRef(true);
  const controllers = useRef(new Set<AbortController>());
  function change(target: string, update: (state: QueueState) => QueueState) {
    if (!alive.current) return;
    const next = { ...statesRef.current, [target]: update(statesRef.current[target] || emptyQueue) };
    statesRef.current = next;
    setStates(next);
  }
  useEffect(() => {
    alive.current = true;
    const unfinished = () => Object.values(statesRef.current).some((state) => state.entries.some((entry) => entry.status !== "succeeded"));
    const unregister = registerBrowserNavigationBlocker(() => !unfinished() || window.confirm("导入队列还有未完成或待核对的文件，仅保留在本页内存中。离开当前页面后可能需要重新选择；已导入文稿不会撤销。仍要离开吗？"));
    const beforeUnload = (event: BeforeUnloadEvent) => { if (unfinished()) { event.preventDefault(); event.returnValue = ""; } };
    window.addEventListener("beforeunload", beforeUnload);
    return () => { alive.current = false; controllers.current.forEach((controller) => controller.abort()); controllers.current.clear(); unregister(); window.removeEventListener("beforeunload", beforeUnload); };
  }, []);

  function add(files: Iterable<File>, defaults: ImportDefaults = emptyImportDefaults) {
    change(scope, (current) => {
      if (current.busy) return current;
      const added = appendImportEntries(current.entries, files, defaults);
      return { ...current, entries: added.entries, notice: added.duplicates ? `${added.duplicates} 份选中文件已在队列中，没有重复添加。` : "" };
    });
  }
  function edit(id: string, patch: Partial<ImportDefaults>) {
    change(scope, (current) => current.busy ? current : { ...current, entries: current.entries.map((entry) => entry.id === id ? updateImportEntry(entry, patch) : entry) });
  }
  function remove(id: string) {
    change(scope, (current) => current.busy ? current : { ...current, entries: current.entries.filter((entry) => entry.id !== id) });
  }
  function clearCompleted() {
    change(scope, (current) => current.busy ? current : { ...current, entries: current.entries.filter((entry) => entry.status !== "succeeded") });
  }
  function setNotice(message: string) { change(scope, (current) => ({ ...current, notice: message })); }

  async function run(projectId: string, retryId?: string) {
    const targetScope = scope;
    const current = statesRef.current[targetScope] || emptyQueue;
    if (!projectId || current.busy) return { uploaded: 0, stopped: false };
    const batch = importBatchEntries(current.entries, retryId);
    if (batch.length === 0) return { uploaded: 0, stopped: false };
    change(targetScope, (state) => ({ ...state, busy: true, notice: "" }));
    let uploaded = 0;
    let stopped = false;
    try {
      for (const entry of batch) {
        if (!alive.current) { stopped = true; break; }
        let fields: Record<string, string>;
        try { fields = importContextFields(entry); }
        catch (error) {
          change(targetScope, (state) => ({ ...state, entries: state.entries.map((row) => row.id === entry.id ? { ...row, status: "failed", message: error instanceof Error ? error.message : "请检查故事作用域。" } : row) }));
          continue;
        }
        const controller = new AbortController();
        controllers.current.add(controller);
        change(targetScope, (state) => ({ ...state, entries: state.entries.map((row) => row.id === entry.id ? { ...row, status: "uploading", message: "" } : row) }));
        try {
          const form = new FormData();
          form.append("file", entry.file);
          Object.entries(fields).forEach(([name, value]) => form.append(name, value));
          const response = await apiJson(`/api/v1/projects/${projectId}/documents`, {
            method: "POST", headers: { "Idempotency-Key": entry.operationKey }, body: form, signal: controller.signal,
          });
          const receipt = verifiedImportReceipt(response, projectId, entry.file.name);
          if (!receipt) throw new ImportResultUnknown();
          uploaded += 1;
          change(targetScope, (state) => ({ ...state, entries: state.entries.map((row) => row.id === entry.id ? { ...row, status: "succeeded", message: receipt.deduplicated ? "已核对原导入记录，没有新增文档版本。" : "导入完成；资料身份和发布状态仍需核对。", receipt } : row) }));
        } catch (error) {
          if (!alive.current) { stopped = true; break; }
          const failure = importFailure(error);
          change(targetScope, (state) => ({ ...state, entries: state.entries.map((row) => row.id === entry.id ? { ...row, status: failure.status, message: failure.message } : row) }));
          if (failure.stop) { stopped = true; break; }
        } finally { controllers.current.delete(controller); }
      }
    } finally {
      change(targetScope, (state) => {
        const counts = importCounts(state.entries);
        return { ...state, busy: false, notice: `已导入 ${counts.succeeded}/${counts.total} 份文稿${counts.failed ? `，${counts.failed} 份失败` : ""}${counts.unknown ? `，${counts.unknown} 份结果待核对` : ""}${stopped ? "；本轮已停止，未处理项仍在队列" : ""}。没有启动分析或发布。` };
      });
    }
    return { uploaded, stopped };
  }
  const state = states[scope] || emptyQueue;
  return { ...state, counts: importCounts(state.entries), add, edit, remove, clearCompleted, setNotice, run };
}
