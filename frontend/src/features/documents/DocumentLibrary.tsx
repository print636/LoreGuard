import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, apiJson } from "../../api/client";
import { documentRoles, type DocumentRole } from "../../documentContext";
import {
  documentPreviewPath, filterDocumentGroups, groupDocuments, initialLibraryState,
  isCurrentPreviewRequest, libraryPage, literalHighlightSegments,
  normalizeDocumentPreview, preferredGroupVersion, PREVIEW_PAGE_SIZE,
  PREVIEW_QUERY_LIMIT, publicationLabel,
  type DocumentPreview, type LibraryDocument, type LibraryState,
  type PreviewRequest, type VersionFilter,
} from "./libraryModel";
import "./document-library.css";

type Props = {
  projectId: string;
  documents: LibraryDocument[];
  loading: boolean;
  state: LibraryState;
  onStateChange: (state: LibraryState) => void;
  onCompare: (fromId: string, toId: string) => void;
};

type ReaderState = {
  request: PreviewRequest | null;
  status: "idle" | "loading" | "ready" | "error";
  preview: DocumentPreview | null;
  error: string;
};

function previewError(error: unknown, timedOut: boolean): string {
  if (timedOut) return "读取原文超时。请检查连接后重试。";
  if (error instanceof ApiError) {
    if (error.status === 401) return "登录已过期，请重新登录后阅读。";
    if (error.status === 403 || error.status === 404) return "这份文稿不存在，或当前账户无法访问。请刷新项目后重试。";
    if (error.status === 422) return "读取范围或搜索文字无效。请清除搜索后重试。";
    return "服务器暂时无法读取原文，请稍后重试。";
  }
  if (error instanceof TypeError && error.message.startsWith("返回的")) return error.message;
  return "读取原文失败。请检查连接后重试。";
}

function LiteralText({ text, query }: { text: string; query: string }) {
  return literalHighlightSegments(text, query).map((segment, index) =>
    segment.match ? <mark key={index}>{segment.text}</mark> : <Fragment key={index}>{segment.text}</Fragment>,
  );
}

function readableDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "时间未记录" : date.toLocaleString("zh-CN");
}

export default function DocumentLibrary({ projectId, documents, loading, state: savedState, onStateChange, onCompare }: Props) {
  const state = savedState.projectId === projectId ? savedState : initialLibraryState(projectId);
  const groups = useMemo(() => groupDocuments(documents, projectId), [documents, projectId]);
  const filtered = useMemo(() => filterDocumentGroups(groups, state), [groups, state.appliedFilenameQuery, state.roleFilter, state.versionFilter]);
  const page = libraryPage(filtered, state.page);
  const selectedGroup = filtered.find((group) => group.key === state.selectedGroupKey) || null;
  const selected = selectedGroup?.versions.find((document) => document.id === state.selectedDocumentId) || null;
  const request = useMemo<PreviewRequest | null>(() => !loading && selected && selected.project_id === projectId ? {
    projectId, documentId: selected.id, version: selected.version, name: selected.name,
    offset: state.offset, limit: PREVIEW_PAGE_SIZE, query: state.appliedQuery,
  } : null, [loading, projectId, selected?.id, selected?.project_id, selected?.version, selected?.name,
    selected?.active, selected?.narrative_context?.publication_status, state.offset, state.appliedQuery]);
  const [reader, setReader] = useState<ReaderState>({ request: null, status: "idle", preview: null, error: "" });
  const [retry, setRetry] = useState(0);
  const requestSequence = useRef(0);
  const catalogList = useRef<HTMLUListElement | null>(null);
  const readerHeading = useRef<HTMLHeadingElement | null>(null);
  const currentReader = isCurrentPreviewRequest(reader.request, request) ? reader : null;
  const preview = currentReader?.status === "ready" ? currentReader.preview : null;
  const metadata = preview?.document || selected;

  function update(next: Partial<LibraryState>) {
    onStateChange({ ...state, ...next, projectId });
  }

  function resetSelection(next: Partial<LibraryState>) {
    update({ ...next, page: 1, selectedGroupKey: "", selectedDocumentId: "", offset: 0, queryDraft: "", appliedQuery: "" });
  }

  function selectVersion(document: LibraryDocument, groupKey: string, focusReader = true) {
    update({ selectedGroupKey: groupKey, selectedDocumentId: document.id, offset: 0, queryDraft: "", appliedQuery: "" });
    if (focusReader) requestAnimationFrame(() => {
      readerHeading.current?.focus({ preventScroll: true });
      readerHeading.current?.scrollIntoView({ block: "start", behavior: "auto" });
    });
  }

  useEffect(() => { if (catalogList.current) catalogList.current.scrollTop = 0; }, [page.page]);

  useEffect(() => {
    const sequence = ++requestSequence.current;
    if (!request) {
      setReader({ request: null, status: "idle", preview: null, error: "" });
      return;
    }
    const controller = new AbortController();
    let disposed = false;
    let timedOut = false;
    const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 15_000);
    setReader({ request, status: "loading", preview: null, error: "" });
    void apiJson<unknown>(documentPreviewPath(request), { signal: controller.signal })
      .then((response) => {
        if (disposed || controller.signal.aborted || sequence !== requestSequence.current) return;
        const preview = normalizeDocumentPreview(response, request);
        setReader({ request, status: "ready", preview, error: "" });
      })
      .catch((error: unknown) => {
        if (disposed || sequence !== requestSequence.current) return;
        setReader({ request, status: "error", preview: null, error: previewError(error, timedOut) });
      })
      .finally(() => clearTimeout(timeout));
    return () => { disposed = true; controller.abort(); clearTimeout(timeout); };
  }, [request, retry]);

  const hasFilters = Boolean(state.appliedFilenameQuery || state.roleFilter !== "all" || state.versionFilter !== "all");
  const currentVersion = selectedGroup?.current;
  const canCompare = Boolean(selected && !selected.active && currentVersion && currentVersion.id !== selected.id);

  return (
    <section className="documentLibrary" aria-labelledby="document-library-title" aria-busy={loading}>
      <div className="documentLibraryHeading">
        <h3 id="document-library-title">资料库</h3>
        {!loading && projectId && <span>{groups.length} 份文稿，{documents.filter((document) => document.project_id === projectId).length} 个版本</span>}
      </div>
      <div className="documentLibrarySurface">
        <div className="documentLibraryCatalog">
          <form className="documentLibrarySearch" onSubmit={(event) => {
            event.preventDefault();
            resetSelection({ appliedFilenameQuery: state.filenameQuery.trim() });
          }}>
            <label htmlFor="library-filename-query">搜索文件名</label>
            <div>
              <input id="library-filename-query" type="search" autoComplete="off" value={state.filenameQuery} maxLength={160}
                disabled={!projectId || loading} onChange={(event) => update({ filenameQuery: event.target.value })} />
              <button type="submit" disabled={!projectId || loading}>搜索</button>
            </div>
          </form>
          <div className="documentLibraryFilters">
            <label htmlFor="library-role-filter">资料类型
              <select id="library-role-filter" value={state.roleFilter} disabled={!projectId || loading}
                onChange={(event) => resetSelection({ roleFilter: event.target.value as DocumentRole | "all" })}>
                <option value="all">全部类型</option>
                {documentRoles.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select>
            </label>
            <label htmlFor="library-version-filter">版本范围
              <select id="library-version-filter" value={state.versionFilter} disabled={!projectId || loading}
                onChange={(event) => resetSelection({ versionFilter: event.target.value as VersionFilter })}>
                <option value="all">全部版本</option>
                <option value="current">当前版本</option>
                <option value="history">历史版本</option>
              </select>
            </label>
          </div>
          <div className="documentLibraryListHeading">
            <h4>文稿</h4>
            {hasFilters && <button type="button" disabled={loading} onClick={() => resetSelection({ filenameQuery: "", appliedFilenameQuery: "", roleFilter: "all", versionFilter: "all" })}>清除筛选</button>}
          </div>
          {loading ? <p className="documentLibraryNotice" role="status">正在读取文稿与版本…</p>
            : !projectId ? <p className="documentLibraryNotice">选择项目后查看文稿。</p>
            : groups.length === 0 ? <p className="documentLibraryNotice">项目中还没有文稿。导入文稿后，可在这里阅读正文与历史版本。</p>
            : page.total === 0 ? <p className="documentLibraryNotice" role="status">没有符合条件的文稿。调整文件名或筛选条件后重试。</p>
            : <>
              <ul className="documentLibraryList" aria-label="文稿列表" tabIndex={0} ref={catalogList}>
                {page.items.map((group) => {
                  const preferred = preferredGroupVersion(group, state);
                  return <li key={group.key}>
                    <button type="button" className={selectedGroup?.key === group.key ? "selected" : ""}
                      aria-pressed={selectedGroup?.key === group.key} aria-controls="document-library-reader"
                      onClick={() => { if (preferred) selectVersion(preferred, group.key); }}>
                      <strong>{group.name}</strong>
                      <span>{group.current ? `当前版本 v${group.current.version}` : "仅有历史版本"}<span>{group.versions.length} 个版本</span></span>
                      <small>{documentRoles.find(([value]) => value === preferred?.document_role)?.[1] || "资料类型未记录"}</small>
                    </button>
                  </li>;
                })}
              </ul>
              <nav className="documentLibraryPagination" aria-label="文稿列表分页">
                <p role="status">第 {page.start + 1}–{Math.min(page.start + page.items.length, page.total)} 份，共 {page.total} 份</p>
                <div>
                  <button type="button" disabled={page.page === 1} onClick={() => update({ page: page.page - 1 })}>上一页</button>
                  <span>{page.page} / {page.pages}</span>
                  <button type="button" disabled={page.page === page.pages} onClick={() => update({ page: page.page + 1 })}>下一页</button>
                </div>
              </nav>
            </>}
        </div>
        <section className="documentLibraryReader" id="document-library-reader" aria-labelledby="document-reader-heading" aria-busy={Boolean(request && (!currentReader || currentReader.status === "loading"))}>
          <h4 id="document-reader-heading" ref={readerHeading} tabIndex={-1}>{!loading && selected ? selected.name : "文稿阅读"}</h4>
          {loading ? <p className="documentLibraryNotice" role="status">正在读取项目资料…</p>
            : !selected || !selectedGroup ? <p className="documentLibraryNotice">选择文稿，阅读当前或历史版本。</p>
            : <>
              <div className="documentLibraryVersionPicker">
                <label htmlFor="library-selected-version">阅读版本
                  <select id="library-selected-version" value={selected.id} onChange={(event) => {
                    const document = selectedGroup.versions.find((version) => version.id === event.target.value);
                    if (document) selectVersion(document, selectedGroup.key, false);
                  }}>
                    {selectedGroup.versions.map((document) => <option key={document.id} value={document.id}>
                      v{document.version} · {document.active ? "当前版本" : "历史版本"} · {publicationLabel(document)}
                    </option>)}
                  </select>
                </label>
                {canCompare && <button type="button" onClick={() => { if (currentVersion) onCompare(selected.id, currentVersion.id); }}>与当前版本比较</button>}
              </div>
              {metadata && <dl className="documentLibraryMetadata">
                <div><dt>版本状态</dt><dd>v{metadata.version} · {metadata.active ? "当前版本（在用）" : "历史版本（仅阅读）"}</dd></div>
                <div><dt>发布状态</dt><dd>{publicationLabel(metadata)}</dd></div>
                <div><dt>资料类型</dt><dd>{documentRoles.find(([value]) => value === metadata.document_role)?.[1] || metadata.document_role}</dd></div>
                <div><dt>保存时间</dt><dd>{readableDate(metadata.created_at)}</dd></div>
                {preview && <div><dt>正文规模</dt><dd>{preview.document.char_count.toLocaleString("zh-CN")} 字符，{preview.document.line_count.toLocaleString("zh-CN")} 行</dd></div>}
              </dl>}
              {/\.docx$/i.test(selected.name) && <p className="documentLibrarySourceNote">这里阅读的是导入后保存的正文，不还原 Word 排版。</p>}
              <form className="documentLibraryTextSearch" onSubmit={(event) => {
                event.preventDefault();
                update({ appliedQuery: state.queryDraft.trim(), offset: 0 });
              }}>
                <label htmlFor="library-text-query">在此版本中搜索原文</label>
                <div>
                  <input id="library-text-query" type="search" autoComplete="off" value={state.queryDraft} maxLength={PREVIEW_QUERY_LIMIT}
                    aria-describedby="library-text-query-help" onChange={(event) => update({ queryDraft: event.target.value })} />
                  <button type="submit">查找</button>
                  {(state.appliedQuery || state.queryDraft) && <button type="button" onClick={() => update({ queryDraft: "", appliedQuery: "", offset: 0 })}>清除</button>}
                </div>
                <p id="library-text-query-help">按文字查找，不区分大小写；结果保留原文行号。</p>
              </form>
              {!currentReader || currentReader.status === "loading" ? <p className="documentLibraryNotice documentLibraryReadingStatus" role="status">正在读取 v{selected.version} 的{state.appliedQuery ? "匹配原文" : "正文"}…</p>
                : currentReader.status === "error" ? <div className="documentLibraryError" role="alert">
                  <p>{currentReader.error}</p>
                  <button type="button" onClick={() => setRetry((value) => value + 1)}>重试读取</button>
                </div>
                : preview && <>
                  <div className="documentLibraryReadingSummary" role="status">
                    {preview.query ? `找到 ${preview.page.total} 行匹配原文` : `正文共 ${preview.document.line_count} 行`}
                    {preview.lines.length > 0 && <span>本页原文第 {preview.lines[0].line_number}–{preview.lines[preview.lines.length - 1].line_number} 行</span>}
                  </div>
                  {preview.lines.length === 0 ? <div className="documentLibraryNotice">
                    <p>{preview.page.total > 0 ? "这一页已没有原文。" : preview.query ? "此版本中没有匹配文字。修改搜索文字，或清除搜索阅读全部正文。" : "此版本保存的正文为空。"}</p>
                    {state.offset > 0 && <button type="button" onClick={() => update({ offset: 0 })}>返回第一页</button>}
                  </div> : <ol className="documentLibraryLines" tabIndex={0} aria-label={preview.query ? "匹配原文与原始行号" : "正文与原始行号"}>
                    {preview.lines.map((line) => <li key={line.line_number}>
                      <span className="documentLibraryLineNumber" aria-label={`原文第 ${line.line_number} 行`}>{line.line_number}</span>
                      <p><LiteralText text={line.text} query={preview.query} /></p>
                    </li>)}
                  </ol>}
                  {preview.page.total > 0 && <nav className="documentLibraryPagination documentLibraryBodyPagination" aria-label="原文分页">
                    <p>{preview.query ? "匹配结果" : "原文"}第 {Math.min(preview.page.offset + 1, preview.page.total)}–{Math.min(preview.page.offset + preview.lines.length, preview.page.total)} 行，共 {preview.page.total} 行</p>
                    <div>
                      <button type="button" disabled={state.offset === 0} onClick={() => update({ offset: Math.max(0, state.offset - PREVIEW_PAGE_SIZE) })}>上一页原文</button>
                      <span>{Math.floor(state.offset / PREVIEW_PAGE_SIZE) + 1} / {Math.ceil(preview.page.total / PREVIEW_PAGE_SIZE)}</span>
                      <button type="button" disabled={!preview.page.has_more} onClick={() => update({ offset: state.offset + PREVIEW_PAGE_SIZE })}>下一页原文</button>
                    </div>
                  </nav>}
                </>}
            </>}
        </section>
      </div>
    </section>
  );
}
