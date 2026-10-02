import { useEffect, useLayoutEffect, useRef, useState, type FormEvent, type KeyboardEvent } from "react";
import { ApiError, apiJson } from "../../api/client";
import { apiErrorDetail } from "../../app/session";
import { registerBrowserNavigationBlocker } from "../../routing";
import { metadataDraftDirty, metadataNameError, metadataPatch, rebaseMetadataDraft, verifiedProjectMetadata, type MetadataDraft, type ProjectMetadata } from "./projectMetadata";
import "./project-metadata.css";

type Props = { projectId: string; trigger: HTMLButtonElement; onClose: () => void; onSaved: (metadata: ProjectMetadata) => Promise<void> | void };

export default function ProjectMetadataEditor({ projectId, trigger, onClose, onSaved }: Props) {
  const [base, setBase] = useState<ProjectMetadata | null>(null);
  const [draft, setDraft] = useState<MetadataDraft>({ name: "", description: "" });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [nameError, setNameError] = useState("");
  const [conflict, setConflict] = useState(false);
  const [latest, setLatest] = useState<ProjectMetadata | null>(null);
  const [notice, setNotice] = useState("");
  const dialog = useRef<HTMLDialogElement | null>(null);
  const nameInput = useRef<HTMLInputElement | null>(null);
  const alive = useRef(true);
  const requests = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const dirty = metadataDraftDirty(draft, base);
  const dirtyRef = useRef(dirty);
  const savingRef = useRef(saving);
  dirtyRef.current = dirty;
  savingRef.current = saving;

  function close(force = false) {
    if (!force && savingRef.current) return;
    if (!force && dirtyRef.current && !window.confirm("项目名称和说明尚未保存。关闭后会丢弃这次修改，仍要关闭吗？")) return;
    dialog.current?.close();
    onClose();
    if (trigger.isConnected) trigger.focus({ preventScroll: true });
    else {
      const replacement = Array.from(document.querySelectorAll<HTMLButtonElement>("button[data-project-metadata-id]")).find((button) => button.dataset.projectMetadataId === projectId);
      (replacement || document.querySelector<HTMLElement>("main[tabindex='-1']"))?.focus({ preventScroll: true });
    }
  }

  async function readLatest(compareOnly = false) {
    const sequence = ++requests.current;
    controller.current?.abort();
    const requestController = new AbortController();
    controller.current = requestController;
    setLoading(true);
    setError("");
    try {
      const result = verifiedProjectMetadata(await apiJson(`/api/v1/projects/${projectId}/metadata`, { signal: requestController.signal }), projectId);
      if (!result) throw new Error("Invalid metadata response");
      if (!alive.current || sequence !== requests.current) return;
      if (compareOnly) setLatest(result);
      else { setBase(result); setDraft({ name: result.name, description: result.description }); }
      requestAnimationFrame(() => nameInput.current?.focus());
    } catch (reason) {
      if (!alive.current || sequence !== requests.current || requestController.signal.aborted) return;
      setError(`${apiErrorDetail(reason)} ${compareOnly ? "你填写的修改仍保留，可重新读取最新信息。" : "尚未读取项目信息，请重试；本次没有保存修改。"}`);
    } finally { if (alive.current && sequence === requests.current) setLoading(false); }
  }

  useLayoutEffect(() => {
    const element = dialog.current;
    if (element && !element.open) element.showModal();
    return () => { if (element?.open) element.close(); };
  }, []);
  useEffect(() => {
    alive.current = true;
    void readLatest();
    const leave = () => {
      if (!dirtyRef.current && !savingRef.current) { close(true); return true; }
      const accepted = window.confirm(savingRef.current
        ? "保存请求正在进行，结果可能已经生效。离开后请重新核对项目信息，仍要离开吗？"
        : "项目名称和说明尚未保存，离开会丢弃这次修改。仍要离开吗？");
      if (accepted) close(true);
      return accepted;
    };
    const unregister = registerBrowserNavigationBlocker(leave);
    const beforeUnload = (event: BeforeUnloadEvent) => {
      if (!dirtyRef.current && !savingRef.current) return;
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", beforeUnload);
    return () => {
      alive.current = false;
      ++requests.current;
      controller.current?.abort();
      unregister();
      window.removeEventListener("beforeunload", beforeUnload);
    };
  }, [projectId]);

  async function save(event: FormEvent) {
    event.preventDefault();
    if (!base || loading || saving || conflict) return;
    const validation = metadataNameError(draft.name);
    setNameError(validation);
    if (validation) { nameInput.current?.focus(); return; }
    const submitted = { ...draft };
    setSaving(true);
    savingRef.current = true;
    setError("");
    setNotice("");
    try {
      const result = verifiedProjectMetadata(await apiJson(`/api/v1/projects/${projectId}/metadata`, {
        method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(metadataPatch(submitted, base)),
      }), projectId);
      if (!result || result.metadata_revision < base.metadata_revision) throw new Error("Invalid metadata result");
      if (!alive.current) return;
      setBase(result);
      setDraft({ name: result.name, description: result.description });
      dirtyRef.current = false;
      setNotice("项目信息已保存。正文、设定和运行报告没有改变。");
      try { await onSaved(result); }
      catch { if (alive.current) setNotice("项目信息已保存，但项目目录暂未刷新。请关闭后重试加载列表；不要重复保存。"); }
    } catch (reason) {
      if (!alive.current) return;
      const uncertain = !(reason instanceof ApiError) || reason.status >= 500 || reason.status === 408 || reason.status < 400;
      if (reason instanceof ApiError && reason.status === 409 || uncertain) {
        setConflict(true);
        setLatest(null);
        setError(uncertain ? "保存结果待核对。你填写的修改仍保留；请读取最新项目信息，对比后再决定。" : "项目信息已被其他页面修改。你填写的内容仍保留；请读取最新信息并对比，不会自动覆盖任何版本。");
      } else setError(`${apiErrorDetail(reason)} 你填写的修改仍保留。`);
    } finally { if (alive.current) { setSaving(false); savingRef.current = false; } }
  }

  function loopFocus(event: KeyboardEvent<HTMLDialogElement>) {
    if (event.key !== "Tab") return;
    const elements = Array.from(dialog.current?.querySelectorAll<HTMLElement>("button, input, textarea, [tabindex]") || [])
      .filter((element) => element.tabIndex >= 0 && !element.matches(":disabled") && element.getClientRects().length > 0);
    const first = elements[0], last = elements[elements.length - 1];
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
  }

  return <dialog className="projectMetadataEditor" ref={dialog} aria-labelledby="project-metadata-title" aria-describedby="project-metadata-purpose" onKeyDown={loopFocus} onCancel={(event) => { event.preventDefault(); close(); }}>
    <div className="projectMetadataHeading"><h2 id="project-metadata-title">编辑项目信息</h2><p id="project-metadata-purpose">修改故事名称与说明，不改变正文、资料身份、发布状态或历史报告。修改仅留在本页内存中，保存后才写入项目。</p></div>
    <form onSubmit={save} noValidate>
      {loading && <p role="status">正在读取项目信息…{base ? "你填写的内容不会被替换。" : ""}</p>}
      <label><span>项目名称</span><input ref={nameInput} value={draft.name} maxLength={200} autoComplete="off" disabled={!base || saving} aria-invalid={!!nameError || undefined} aria-describedby={nameError ? "project-metadata-name-error" : undefined} onBlur={() => { if (base) setNameError(metadataNameError(draft.name)); }} onChange={(event) => { setDraft((current) => ({ ...current, name: event.target.value })); setNameError(""); setNotice(""); }} /></label>
      {nameError && <p id="project-metadata-name-error" className="metadataError" role="alert">{nameError}</p>}
      <label><span>项目说明</span><textarea value={draft.description} rows={5} disabled={!base || saving} onChange={(event) => { setDraft((current) => ({ ...current, description: event.target.value })); setNotice(""); }} /></label>
      {error && <p className="metadataError" role="alert">{error}</p>}
      {!base && !loading && <button type="button" onClick={() => void readLatest()}>重试读取项目信息</button>}
      {conflict && <div className="metadataConflict">
        <button type="button" disabled={loading || saving} onClick={() => void readLatest(true)}>{latest ? "重新读取最新项目信息" : "读取最新项目信息"}</button>
        {latest && <section aria-label="服务器最新项目信息"><h3>服务器最新内容</h3><dl><dt>项目名称</dt><dd>{latest.name}</dd><dt>项目说明</dt><dd>{latest.description || "未填写说明"}</dd></dl>
          <p>你改动的字段会保留，未改动的字段会采用以上最新内容。{base && draft.name !== base.name ? " 保存时会用你填写的项目名称替换服务器名称。" : ""}{base && draft.description !== base.description ? " 保存时会用你填写的项目说明替换服务器说明。" : ""}合并后仍需检查并点击保存，不会自动提交。</p>
          <div className="metadataConflictActions"><button type="button" disabled={loading || saving} onClick={() => { if (!base) return; setDraft(rebaseMetadataDraft(draft, base, latest)); setBase(latest); setConflict(false); setLatest(null); setError(""); setNotice("已保留你的改动并采用未修改字段的最新内容。请检查后点击保存，尚未提交。"); }}>保留我的改动，合并最新内容</button>
            <button type="button" disabled={loading || saving} onClick={() => { if (dirty && !window.confirm("采用服务器最新内容会替换你当前未保存的名称和说明。仍要采用吗？")) return; setBase(latest); setDraft({ name: latest.name, description: latest.description }); setConflict(false); setLatest(null); setError(""); setNameError(""); setNotice("已采用服务器最新内容，没有提交新的修改。"); }}>采用最新内容</button></div>
        </section>}
      </div>}
      {notice && <p role="status" className="metadataNotice">{notice}</p>}
      <div className="projectMetadataActions"><button className="quietPrimary" type="submit" disabled={!base || loading || saving || conflict || !dirty}>{saving ? "正在保存…" : "保存项目信息"}</button><button type="button" disabled={saving} onClick={() => close()}>取消</button></div>
    </form>
  </dialog>;
}
