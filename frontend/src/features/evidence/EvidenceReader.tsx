import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { ApiError, apiJson } from "../../api/client";
import {
  EVIDENCE_PAGE_SIZE, evidenceKindLabel, evidencePreviewPath, isCurrentEvidenceRequest,
  normalizeEvidencePreview, referenceRange, sameEvidenceSource,
  type EvidencePreview, type EvidenceRequest, type EvidenceSelection, type EvidenceSource,
} from "./evidenceReaderModel";
import "./evidence-reader.css";

type Props = { selection: EvidenceSelection; trigger: HTMLButtonElement; onClose: () => void };
type ReadingState = { request: EvidenceRequest | null; status: "loading" | "ready" | "error"; preview: EvidencePreview | null; error: string };

function safeReaderError(error: unknown, timedOut: boolean): string {
  if (timedOut) return "读取冻结原文超时。请检查连接后重试，或关闭面板返回报告。";
  if (error instanceof ApiError) {
    if (error.status === 401) return "登录已过期，请重新登录后阅读。";
    if (error.status === 403 || error.status === 404) return "这条证据不存在、不属于本次分析，或当前账户无法访问。请返回报告核对运行后重试。";
    if (error.status === 409) return "这条证据的冻结来源缺失或无法核验。报告中的摘录仍可查看，可重试读取原文。";
    return "服务器暂时无法读取冻结原文，请稍后重试。";
  }
  if (error instanceof TypeError && (error.message.startsWith("返回的") || error.message.startsWith("冻结来源"))) return error.message;
  return "读取冻结原文失败。请检查连接后重试，或关闭面板返回报告。";
}

export default function EvidenceReader({ selection, trigger, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement | null>(null);
  const readingRegion = useRef<HTMLDivElement | null>(null);
  const pinnedSource = useRef<EvidenceSource | null>(null);
  const sequence = useRef(0);
  const focusReferenceAfterLoad = useRef(false);
  const [offset, setOffset] = useState<number | null>(null);
  const [retry, setRetry] = useState(0);
  const [reading, setReading] = useState<ReadingState>({ request: null, status: "loading", preview: null, error: "" });
  const request = useMemo<EvidenceRequest>(() => ({ selection, offset, limit: EVIDENCE_PAGE_SIZE }), [selection, offset]);
  const current = isCurrentEvidenceRequest(reading.request, request) ? reading : null;
  const preview = current?.status === "ready" ? current.preview : null;
  const busy = !current || current.status === "loading";
  const categoryLabel = selection.kind === "issue" ? "正式问题" : selection.kind === "review_clue" ? "待复核" : "未验证";
  const headerSummary = preview
    ? `冻结 v${preview.source.document_version} · ${preview.source.provenance === "run_input" ? "本次输入" : "既有设定"} · ${categoryLabel}${preview.source.availability === "excerpt_only" ? " · 仅摘录" : ""}`
    : `${categoryLabel} · ${busy ? "来源读取中" : "来源未核验"}`;

  function closeReader() {
    dialog.current?.close();
    onClose();
    // Native dialog keeps the report mounted; preserve the reader's original position.
    if (trigger.isConnected) trigger.focus({ preventScroll: true });
  }

  useLayoutEffect(() => {
    const element = dialog.current;
    if (element && !element.open) element.showModal();
    return () => { if (element?.open) element.close(); };
  }, []);

  useEffect(() => {
    const requestSequence = ++sequence.current;
    const controller = new AbortController();
    let disposed = false;
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 15_000);
    setReading({ request, status: "loading", preview: null, error: "" });
    void Promise.resolve().then(() => apiJson<unknown>(evidencePreviewPath(request), { signal: controller.signal }))
      .then((value) => {
        if (disposed || controller.signal.aborted || requestSequence !== sequence.current) return;
        const preview = normalizeEvidencePreview(value, request);
        if (pinnedSource.current && !sameEvidenceSource(pinnedSource.current, preview.source)) {
          throw new TypeError("冻结来源与上一次读取不一致，请关闭面板后重新核对这条证据。");
        }
        pinnedSource.current = preview.source;
        setReading({ request, status: "ready", preview, error: "" });
      })
      .catch((error: unknown) => {
        if (disposed || requestSequence !== sequence.current) return;
        setReading({ request, status: "error", preview: null, error: safeReaderError(error, timedOut) });
      })
      .finally(() => clearTimeout(timeout));
    return () => { disposed = true; controller.abort(); clearTimeout(timeout); };
  }, [request, retry]);

  useLayoutEffect(() => {
    if (!preview || !readingRegion.current) return;
    if (offset === null) {
      const referenceLine = readingRegion.current.querySelector<HTMLElement>(`[data-evidence-line="${selection.expected.line_start}"]`);
      referenceLine?.scrollIntoView({ behavior: "auto", block: "nearest" });
    } else readingRegion.current.scrollTop = 0;
    if (focusReferenceAfterLoad.current) readingRegion.current.focus({ preventScroll: true });
    focusReferenceAfterLoad.current = false;
  }, [preview, offset, selection.expected.line_start]);

  function returnToReference() {
    if (offset !== null) {
      focusReferenceAfterLoad.current = true;
      setOffset(null);
    } else {
      readingRegion.current?.querySelector<HTMLElement>(`[data-evidence-line="${selection.expected.line_start}"]`)?.scrollIntoView({ behavior: "auto", block: "nearest" });
      readingRegion.current?.focus({ preventScroll: true });
    }
  }

  return (
    <dialog className="evidenceReader" ref={dialog} aria-labelledby="evidence-reader-title" aria-describedby="evidence-reader-boundary"
      onCancel={(event) => { event.preventDefault(); closeReader(); }}>
      <div className="evidenceReaderHeading">
        <div>
          <h2 id="evidence-reader-title">分析时原文</h2>
          <p>{selection.expected.document_name}</p>
          <p className={`evidenceReaderHeaderSummary ${selection.kind !== "issue" ? "pending" : ""}`}>{headerSummary}</p>
        </div>
        <button type="button" autoFocus onClick={closeReader}>关闭阅读面板</button>
      </div>
      <p className="evidenceReaderBoundary" id="evidence-reader-boundary">阅读本次报告绑定的冻结来源。文稿后续修改不会替换这里的原文。</p>
      <div className="evidenceReaderReading" ref={readingRegion} tabIndex={0} role="region" aria-label="冻结原文阅读区" aria-busy={busy}>
        <p className={`evidenceReaderKind ${selection.kind !== "issue" ? "pending" : ""}`}>
          <strong>{evidenceKindLabel(selection.kind)}</strong>
          {selection.kind !== "issue" && <span>原文定位可核对，阅读不会改变线索或提案的审查结论。</span>}
        </p>
        <dl className="evidenceReaderMetadata">
          <div><dt>报告运行</dt><dd><code title={selection.runId}>{selection.runId.slice(0, 8)}</code></dd></div>
          <div><dt>报告引用范围</dt><dd>{referenceRange(selection.expected)}</dd></div>
          {preview && <>
            <div><dt>分析时版本</dt><dd>冻结版本 v{preview.source.document_version}</dd></div>
            <div><dt>原文来源</dt><dd>{preview.source.provenance === "run_input" ? "本次分析的冻结输入" : "既有角色设定的冻结来源"}</dd></div>
            {preview.source.source_run_id && <div><dt>来源运行</dt><dd><code title={preview.source.source_run_id}>{preview.source.source_run_id.slice(0, 8)}</code></dd></div>}
            {preview.source.char_count !== null && <div><dt>正文规模</dt><dd>{preview.source.char_count.toLocaleString("zh-CN")} 字符，{preview.source.line_count?.toLocaleString("zh-CN")} 行</dd></div>}
          </>}
        </dl>
        <details className="evidenceReaderExcerpt" open={preview?.source.availability === "excerpt_only" || current?.status === "error"}>
          <summary>报告中保留的引用 · {referenceRange(selection.expected)}</summary>
          <blockquote>{selection.expected.text}</blockquote>
        </details>
        {busy ? <p className="evidenceReaderStatus" role="status">正在读取这条证据的冻结原文…</p>
          : current?.status === "error" ? <div className="evidenceReaderError" role="alert">
            <p>{current.error}</p><button type="button" onClick={() => setRetry((value) => value + 1)}>重试读取原文</button>
          </div>
          : preview?.source.availability === "excerpt_only" ? <div className="evidenceReaderUnavailable" role="status">
            <h3>只有冻结摘录，完整来源原文不可用</h3>
            <p>完整原文快照已缺失。上方保留报告绑定的证据摘录，无法向前后扩展阅读。</p>
          </div>
          : preview && <>
            {/\.docx$/i.test(selection.expected.document_name) && <p className="evidenceReaderSourceNote">这里是导入后保存的正文，不还原 Word 排版。</p>}
            <p className="evidenceReaderRangeNotice">引用范围：{referenceRange(selection.expected)}。对应原文行标有“引用”，并以边框突出。</p>
            <ol className="evidenceReaderLines" aria-label="冻结原文与原始行号">
              {preview.lines.map((line) => <li key={line.line_number} data-evidence-line={line.line_number} className={line.is_evidence ? "referenced" : ""}>
                <span className="evidenceReaderLineNumber" aria-label={`原文第 ${line.line_number} 行${line.is_evidence ? "，报告引用" : ""}`}>
                  {line.line_number}{line.is_evidence && <small>引用</small>}
                </span>
                <p>{line.text}</p>
              </li>)}
            </ol>
            {preview.lines.length === 0 && <p className="evidenceReaderStatus">此页没有原文行。可返回引用位置继续阅读。</p>}
          </>}
      </div>
      <nav className="evidenceReaderPagination" aria-label="冻结原文分页">
        {preview?.page ? <>
          <p role="status">本页原文第 {Math.min(preview.page.offset + 1, preview.page.total)}–{Math.min(preview.page.offset + preview.lines.length, preview.page.total)} 行，共 {preview.page.total} 行</p>
          <div>
            <button type="button" disabled={busy || preview.page.offset === 0} onClick={() => setOffset(Math.max(0, preview.page!.offset - EVIDENCE_PAGE_SIZE))}>上一页原文</button>
            <button type="button" disabled={busy} onClick={returnToReference}>返回引用位置</button>
            <button type="button" disabled={busy || !preview.page.has_more} onClick={() => setOffset(preview.page!.offset + EVIDENCE_PAGE_SIZE)}>下一页原文</button>
          </div>
        </> : <p>{busy ? "正在核对冻结来源与引用范围…" : "关闭面板后可继续核对报告。"}</p>}
      </nav>
    </dialog>
  );
}
