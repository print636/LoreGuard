import { useEffect, useId, useRef, useState } from "react";
import { apiFetch } from "../../api/client";
import type { RunInputSnapshot } from "../../runSnapshot";
import { MAX_SOURCE_MATERIALS, SourceMaterialsError, selectedSourceIds, sourceMaterialContext, sourceMaterialsBlob, sourceMaterialsPath } from "./sourceMaterialsModel";
import "./source-materials.css";

type Props = { projectId: string; runId: string; inputs: RunInputSnapshot[] };

function ScopedSourceMaterials({ runId, inputs }: Props) {
  const [open, setOpen] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<{ error: boolean; text: string } | null>(null);
  const active = useRef<AbortController | null>(null);
  const explanationId = useId();
  useEffect(() => () => { active.current?.abort(); active.current = null; }, []);

  function cancel() {
    active.current?.abort(); active.current = null; setBusy(false);
  }
  function change(id: string, checked: boolean) {
    setNotice(null);
    setSelected(previous => { const next = new Set(previous); checked ? next.add(id) : next.delete(id); return next; });
  }
  async function download() {
    if (active.current) return;
    const controller = new AbortController();
    active.current = controller;
    setBusy(true); setNotice(null);
    const timeout = setTimeout(() => controller.abort(), 60_000);
    try {
      const ids = selectedSourceIds(inputs, selected);
      const response = await apiFetch(sourceMaterialsPath(runId, ids), { signal: controller.signal, headers: { Accept: "application/zip" } });
      const blob = await sourceMaterialsBlob(response);
      if (active.current !== controller) return;
      if (controller.signal.aborted) throw new SourceMaterialsError("资料下载准备超时。");
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url; link.download = "LoreGuard-所选审查资料.zip";
      document.body.append(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
      setNotice({ error: false, text: `已下载 ${ids.length} 份所选资料和资料目录。没有导出未勾选资料，没有启动审查或修改文稿。` });
    } catch (error) {
      if (active.current !== controller) return;
      setNotice({ error: true, text: controller.signal.aborted ? "资料下载准备超时，勾选仍保留，请重试。" : error instanceof SourceMaterialsError ? error.message : "资料下载失败，勾选仍保留，请重试。" });
    } finally {
      clearTimeout(timeout);
      if (active.current === controller) { active.current = null; setBusy(false); }
    }
  }
  return <details className="sourceMaterialsExport" aria-label="本次审查资料导出" aria-busy={busy} open={open} onToggle={event => {
    const next = event.currentTarget.open;
    if (!next) { cancel(); setNotice(null); }
    setOpen(next);
  }}>
    <summary>导出本次资料</summary>
    <div className="sourceMaterialsBody">
      <p id={explanationId}>只导出你勾选的本次审查输入，不包含全项目资料或清单外的旧角色来源。下方是保存的输入清单；下载时会核对冻结原文与资料身份，无法核对则不生成压缩包。</p>
      <p>ZIP 内含“资料目录.md”和逐份解析后的 Markdown 正文，正文不插入前言。Word 排版、批注请使用原文件；回传仍须与原文稿同名：Markdown 请对照目录按原名保存，Word 请修改原 Word 文件再上传。不会自动改名或跨格式替换。</p>
      {inputs.length ? <>
        <p className="sourceMaterialsLimit">每次最多 100 份、正文合计 20 MiB；超出请分批选择，系统不会截断资料。</p>
        <fieldset disabled={busy} aria-describedby={explanationId}>
          <legend>勾选本次资料（已选 {selected.size} / {inputs.length} 份）</legend>
          <div className="sourceMaterialsSelectionActions">
            {inputs.length <= MAX_SOURCE_MATERIALS && <button type="button" disabled={selected.size === inputs.length} onClick={() => { setSelected(new Set(inputs.map(input => input.document_id))); setNotice(null); }}>选择本次全部资料</button>}
            <button type="button" disabled={!selected.size} onClick={() => { setSelected(new Set()); setNotice(null); }}>清空勾选</button>
          </div>
          <ul>{inputs.map(input => <li key={input.document_id}><label>
            <input type="checkbox" aria-label={`${input.document_name} · v${input.document_version}`} checked={selected.has(input.document_id)} disabled={!selected.has(input.document_id) && selected.size >= MAX_SOURCE_MATERIALS}
              onChange={event => change(input.document_id, event.target.checked)} />
            <span><b>{input.document_name} · v{input.document_version}</b><small>{sourceMaterialContext(input)}</small></span>
          </label></li>)}</ul>
        </fieldset>
        <button type="button" disabled={busy || selected.size === 0 || selected.size > MAX_SOURCE_MATERIALS} onClick={() => void download()}>{busy ? "正在核对并打包…" : `下载所选资料（${selected.size}份）`}</button>
        {notice && <p className="sourceMaterialsNotice" role={notice.error ? "alert" : "status"}>{notice.text}</p>}
      </> : <p role="status">这份报告没有可用的当次输入清单，无法勾选导出；不会改用当前项目资料。</p>}
    </div>
  </details>;
}

export default function SourceMaterialsExport(props: Props) {
  // A refreshed or different frozen scope must never retain another scope's selection/download.
  return <ScopedSourceMaterials key={JSON.stringify([props.projectId, props.runId, props.inputs])} {...props} />;
}
