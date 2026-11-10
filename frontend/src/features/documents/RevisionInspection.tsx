import { useEffect, useRef, useState } from "react";
import { apiJson, SESSION_EXPIRED_EVENT } from "../../api/client";
import type { RevisionInspection as Coordinates } from "../../revisionWorkflow";
import { workspacePath } from "../../routing";
import DocumentLibrary from "./DocumentLibrary";
import { documentNameKey, initialLibraryState, type DocumentPreview } from "./libraryModel";
import DocumentDiffResult from "./DocumentDiffResult";
import { documentDiffMatches, documentDiffPath, type DocumentDiff } from "./documentDiff";
import { inspectionDocuments, type InspectionDocument } from "./revisionInspectionModel";
import NarrativeContextWorkbench from "../workflow/NarrativeContextWorkbench";
import type { GuidedDocument, NarrativeContext } from "../workflow/guidedReview";

type Props = {
  projectId: string;
  coordinates: Coordinates;
  documents: InspectionDocument[];
  onRefresh: () => Promise<boolean>;
  onContextSaved: (id: string, role: GuidedDocument["document_role"], context: NarrativeContext) => void;
  onOpenProvider: () => void;
  onContinue: () => void;
};

export default function RevisionInspection({ projectId, coordinates, documents, onRefresh, onContextSaved, onOpenProvider, onContinue }: Props) {
  const pair = inspectionDocuments(projectId, coordinates, documents);
  const [library, setLibrary] = useState(() => ({ ...initialLibraryState(projectId),
    selectedGroupKey: pair ? documentNameKey(pair.saved.name) : "", selectedDocumentId: coordinates.savedDocumentId }));
  const [verifiedSaved, setVerifiedSaved] = useState<DocumentPreview["document"] | null>(null);
  const [diff, setDiff] = useState<DocumentDiff | null>(null);
  const [diffError, setDiffError] = useState("");
  const [diffLoading, setDiffLoading] = useState(false);
  const [refresh, setRefresh] = useState(0);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState("");
  const alive = useRef(true);
  const readingPositions = useRef(new Map<string, number>());
  useEffect(() => {
    alive.current = true;
    const expire = () => { alive.current = false; };
    window.addEventListener(SESSION_EXPIRED_EVENT, expire);
    return () => { alive.current = false; window.removeEventListener(SESSION_EXPIRED_EVENT, expire); };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const from = pair?.previous;
    const to = pair?.saved;
    setDiff(null); setDiffError("");
    if (!from || !to) { setDiffLoading(false); return; }
    setDiffLoading(true);
    void apiJson<DocumentDiff>(documentDiffPath(projectId, from.id, to.id), { signal: controller.signal })
      .then(result => {
        if (controller.signal.aborted || !alive.current) return;
        if (!documentDiffMatches(result, projectId, from, to)) throw new TypeError("版本不一致");
        setDiff(result);
      }).catch(() => {
        if (!controller.signal.aborted && alive.current) setDiffError("暂时无法核对这两个版本的正文差异。所选版本没有改变，请重试读取。");
      }).finally(() => { if (!controller.signal.aborted && alive.current) setDiffLoading(false); });
    return () => controller.abort();
  }, [projectId, pair?.saved.id, pair?.saved.version, pair?.previous?.id, pair?.previous?.version, refresh]);

  async function reread() {
    setRefreshing(true); setRefreshError("");
    try {
      if (!await onRefresh()) throw new Error("unavailable");
      if (alive.current) setRefresh(value => value + 1);
    } catch { if (alive.current) setRefreshError("读取项目资料失败，未切换到其他版本。请稍后重试。"); }
    finally { if (alive.current) setRefreshing(false); }
  }

  if (!pair) return <section className="revisionStage" aria-label="核对新版本">
    <h2>暂时无法定位所选版本</h2>
    <p>文稿或版本与链接不一致。不会改用最新版本，也不会据此恢复上传成功状态。</p>
    {refreshError && <p role="alert">{refreshError}</p>}
    <button type="button" disabled={refreshing} onClick={() => void reread()}>重新读取项目资料</button>
    <a href={workspacePath("projects", projectId)}>返回资料库选择文稿</a>
  </section>;

  const saved = { ...pair.saved, active: pair.saved.active && verifiedSaved?.active !== false && diff?.to_document.active !== false };
  const readerDocuments = [saved, ...(pair.previous ? [pair.previous] : [])];
  const readerState = library.selectedGroupKey ? library : { ...library, selectedGroupKey: documentNameKey(saved.name) };
  return <section className="revisionStage revisionInspection" aria-label="核对新版本">
    <div className="revisionStageHead"><div><h2>核对所选版本</h2>
      <p>{verifiedSaved ? "已读取所选版本原文" : "正在核对所选版本"}：{saved.name} · v{saved.version}。
        {pair.previous ? `正文与原目标 v${pair.previous.version} 比较；` : "此次没有指定原目标，不推测版本对应；"}复检仍以原报告为起点。</p>
      <p>这里比较导入后保存的正文。阅读、比较和切换步骤不会启动审查。</p>
    </div></div>
    {!saved.active && <div className="revisionNotice" role="status">
      <p>v{saved.version} 已不是活动版本。仍可阅读和比较；本步骤不能确认它的资料身份或继续复检，也不会替你改选新版。</p>
      <a href={workspacePath("projects", projectId)}>返回资料库另选活动文稿</a>
    </div>}
    <DocumentLibrary projectId={projectId} documents={readerDocuments} loading={false} readerOnly
      state={readerState} onStateChange={setLibrary} onCompare={() => {}} readingPositions={readingPositions.current}
      onPreview={document => { if (alive.current && document.id === saved.id && document.version === saved.version) setVerifiedSaved(document); }} />
    <section className="revisionBodyComparison" aria-label="原目标与所选版本正文差异">
      <h3>正文改动{pair.previous ? ` · v${pair.previous.version} → v${saved.version}` : ""}</h3>
      <p>这里显示正文的逐行变化；审查问题是否仍存在，要在主动复检后另看“复检差异”。</p>
      {!pair.previous ? <p>未指定原目标版本，本步骤只核对所选稿正文与身份。</p>
        : diffLoading ? <p role="status">正在读取这两个版本的正文差异…</p>
        : diffError ? <div role="alert"><p>{diffError}</p><button type="button" onClick={() => setRefresh(value => value + 1)}>重试读取正文差异</button></div>
        : diff && <DocumentDiffResult result={diff} />}
    </section>
    {verifiedSaved ? <div><p>下方只核对 {saved.name} · v{saved.version} 的资料身份；切换上方阅读版本不会改变编辑目标。确认身份不等于发布。</p>
      <NarrativeContextWorkbench projectId={projectId} documents={[saved]}
      fixedDocument={{ id: saved.id, version: saved.version }} onSaved={onContextSaved} onOpenProvider={onOpenProvider} /></div>
      : <p role="status">读取所选版本原文后，可在这里核对同一版本的资料身份。</p>}
    <div className="revisionFormActions">
      <button type="button" disabled={!verifiedSaved || !saved.active} onClick={onContinue}>返回原报告的复检准备</button>
      <button type="button" disabled={refreshing} onClick={() => void reread()}>{refreshing ? "正在读取项目资料…" : "刷新版本状态"}</button>
    </div>
    {refreshError && <p role="alert">{refreshError}</p>}
    <p>复检准备会重新核对当前选材。只有你主动点击开始，才会创建审查任务。</p>
  </section>;
}
