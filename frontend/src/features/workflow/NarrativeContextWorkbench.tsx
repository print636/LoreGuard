import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, apiJson } from "../../api/client";
import { documentRoles } from "../../documentContext";
import {
  contextDraft,
  contextNavigationLocked,
  contextRevision,
  contextStatus,
  isLoadedContextForDocument,
  narrativeContextPayload,
  normalizeNarrativeContextInference,
  publicationStatuses,
  responseBelongsToSelectedDocument,
  type GuidedDocument,
  type NarrativeContext,
  type NarrativeContextDraft,
} from "./guidedReview";

type ContextResponse = {
  document_id: string;
  current: NarrativeContext;
  revision_count: number;
};

type Props = {
  projectId: string;
  documents: GuidedDocument[];
  disabled?: boolean;
  onSaved: (
    documentId: string,
    documentRole: GuidedDocument["document_role"],
    context: NarrativeContext,
  ) => void;
  onOpenProvider: () => void;
};

type InferenceRecovery = "reload" | "provider" | "retry" | "manual";

type InferenceFailure = {
  message: string;
  recovery: InferenceRecovery;
};

function requestMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 409) {
      return "这份资料已在其他页面更新。请重新读取后再保存，当前页面没有覆盖新内容。";
    }
    if (error.status === 422) {
      return "资料上下文不符合服务端约束，请检查版本、时间线和分支标识。";
    }
    return error.message || "资料上下文没有保存，请重试。";
  }
  if (error instanceof Error) return error.message;
  return "资料上下文没有保存，请重试。";
}

function contextPath(projectId: string, documentId: string): string {
  return `/api/v1/projects/${encodeURIComponent(projectId)}/documents/${encodeURIComponent(documentId)}/narrative-context`;
}

function inferenceFailure(error: unknown): InferenceFailure {
  if (error instanceof ApiError) {
    if (error.status === 409) {
      return {
        message: "识别期间资料或上下文发生了变化，本次建议没有保存。请重新读取后再识别。",
        recovery: "reload",
      };
    }
    if (error.status === 503) {
      return {
        message: "模型尚未配置或连接凭据不可用。连接模型后再识别，本页不会把失败当作空建议。",
        recovery: "provider",
      };
    }
    if (error.status === 413) {
      return {
        message: "这份文档超过单次识别上限，请先用下方字段手动设置资料上下文。",
        recovery: "manual",
      };
    }
    if (error.status === 422) {
      return {
        message: "文档内容为空或暂时无法安全分析，请先检查文稿，或使用下方字段手动设置。",
        recovery: "manual",
      };
    }
    if (error.status === 429) {
      return {
        message: "模型服务当前限流，本次没有生成建议。请稍后重试。",
        recovery: "retry",
      };
    }
    if (error.status === 502) {
      return {
        message: "模型没有返回结构完整且有原文证据的建议。你可以重试，或直接手动设置。",
        recovery: "retry",
      };
    }
    if (error.status === 504) {
      return {
        message: "模型响应超时，本次没有生成建议。请检查连接后重试。",
        recovery: "retry",
      };
    }
    return {
      message: error.message || "AI 资料识别失败，本次没有生成建议。请重试或手动设置。",
      recovery: "retry",
    };
  }
  return {
    message: error instanceof Error
      ? error.message
      : "AI 资料识别失败，本次没有生成建议。请重试或手动设置。",
    recovery: "retry",
  };
}

const inferredFieldLabels: Record<string, string> = {
  document_role: "资料类型",
  publication_status: "发布状态",
  release: "版本",
  branch: "分支",
  activity: "活动 / 篇章",
};

export default function NarrativeContextWorkbench({
  projectId,
  documents,
  disabled = false,
  onSaved,
  onOpenProvider,
}: Props) {
  const activeDocuments = useMemo(
    () => documents.filter((document) => document.active),
    [documents],
  );
  const [selectedId, setSelectedId] = useState("");
  const selected =
    activeDocuments.find((document) => document.id === selectedId) ||
    activeDocuments[0] ||
    null;
  const selectedIdRef = useRef("");
  selectedIdRef.current = selected?.id || "";
  const [remoteContext, setRemoteContext] = useState<NarrativeContext | null>(null);
  const [draft, setDraft] = useState<NarrativeContextDraft | null>(null);
  const [loadedDocumentId, setLoadedDocumentId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [inferring, setInferring] = useState(false);
  const [error, setError] = useState("");
  const [inferenceError, setInferenceError] = useState<InferenceFailure | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const errorRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!selected) {
      setSelectedId("");
      setRemoteContext(null);
      setDraft(null);
      setLoadedDocumentId(null);
      return;
    }
    if (selectedId !== selected.id) setSelectedId(selected.id);
  }, [selected?.id]);

  async function loadContext(document: GuidedDocument, signal?: AbortSignal) {
    setLoading(true);
    setLoadedDocumentId(null);
    setError("");
    setInferenceError(null);
    setAnnouncement("");
    try {
      const payload = await apiJson<ContextResponse>(
        contextPath(projectId, document.id),
        { signal },
      );
      if (payload.document_id !== document.id) {
        throw new TypeError("服务返回的资料上下文不属于当前文档。");
      }
      if (!responseBelongsToSelectedDocument(document.id, selectedIdRef.current)) return;
      setRemoteContext(payload.current);
      setDraft(contextDraft(document, payload.current));
      setLoadedDocumentId(document.id);
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === "AbortError") return;
      if (!responseBelongsToSelectedDocument(document.id, selectedIdRef.current)) return;
      setError(requestMessage(reason));
      setRemoteContext(document.narrative_context || null);
      setDraft(contextDraft(document));
      setLoadedDocumentId(document.id);
    } finally {
      if (
        !signal?.aborted &&
        responseBelongsToSelectedDocument(document.id, selectedIdRef.current)
      ) setLoading(false);
    }
  }

  useEffect(() => {
    if (!selected || !projectId) return;
    const controller = new AbortController();
    void loadContext(selected, controller.signal);
    return () => controller.abort();
  }, [projectId, selected?.id]);

  function update<K extends keyof NarrativeContextDraft>(
    key: K,
    value: NarrativeContextDraft[K],
  ) {
    setDraft((current) => (current ? { ...current, [key]: value } : current));
    setError("");
    setInferenceError(null);
    setAnnouncement("");
  }

  async function save() {
    if (
      !selected ||
      !draft ||
      loadedDocumentId !== selected.id ||
      saving ||
      inferring
    ) return;
    const document = selected;
    const draftSnapshot = { ...draft };
    try {
      setSaving(true);
      setError("");
      setAnnouncement("");
      const body = narrativeContextPayload(
        draftSnapshot,
        contextRevision(remoteContext || document.narrative_context),
      );
      const context = await apiJson<NarrativeContext>(
        `${contextPath(projectId, document.id)}/revisions`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
      onSaved(document.id, draftSnapshot.documentRole, context);
      if (!responseBelongsToSelectedDocument(document.id, selectedIdRef.current)) return;
      setRemoteContext(context);
      setDraft(contextDraft({ ...document, document_role: draftSnapshot.documentRole }, context));
      setAnnouncement(
        draftSnapshot.confirmed
          ? `已确认“${document.name}”的资料上下文。`
          : `已保存“${document.name}”，仍需人工确认后才能用于角色审查。`,
      );
    } catch (reason) {
      if (!responseBelongsToSelectedDocument(document.id, selectedIdRef.current)) return;
      setError(requestMessage(reason));
      requestAnimationFrame(() => errorRef.current?.focus());
    } finally {
      setSaving(false);
    }
  }

  async function inferContext() {
    if (!selected || loadedDocumentId !== selected.id || inferring || saving) return;
    const document = selected;
    try {
      setInferring(true);
      setError("");
      setInferenceError(null);
      setAnnouncement("");
      const raw = await apiJson<unknown>(`${contextPath(projectId, document.id)}/inference`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          expected_revision: contextRevision(remoteContext || document.narrative_context),
        }),
      });
      const result = normalizeNarrativeContextInference(raw);
      if (result.document_id !== document.id) {
        throw new TypeError("AI 返回的资料标识与当前文档不一致，请重新读取后重试。");
      }
      const updatedDocument = { ...document, document_role: result.document_role };
      onSaved(document.id, result.document_role, result.suggestion);
      if (!responseBelongsToSelectedDocument(document.id, selectedIdRef.current)) return;
      setRemoteContext(result.suggestion);
      setDraft(contextDraft(updatedDocument, result.suggestion));
      setLoadedDocumentId(document.id);
      setAnnouncement(`已生成“${document.name}”的 AI 资料建议，请核对依据后另行保存并确认。`);
    } catch (reason) {
      if (!responseBelongsToSelectedDocument(document.id, selectedIdRef.current)) return;
      setInferenceError(inferenceFailure(reason));
      requestAnimationFrame(() => errorRef.current?.focus());
    } finally {
      setInferring(false);
    }
  }

  if (!activeDocuments.length) return null;
  const status = contextStatus(remoteContext || selected?.narrative_context);
  const inference = remoteContext?.inference || null;
  const selectedContextReady = isLoadedContextForDocument(loadedDocumentId, selected?.id);
  const editorDisabled = disabled || saving || inferring || !selectedContextReady;
  const navigationLocked = contextNavigationLocked(saving, inferring);

  return (
    <section className="contextWorkbench" aria-labelledby="context-workbench-title">
      <header>
        <div>
          <h3 id="context-workbench-title">确认资料身份</h3>
          <p>先说明每份资料是什么、位于哪条时间线。系统只采用你已确认的上下文。</p>
        </div>
        <span className="contextCount">
          {activeDocuments.filter((document) => document.narrative_context?.resolution_state === "confirmed").length}
          /{activeDocuments.length} 已确认
        </span>
      </header>

      <div className="contextWorkbenchLayout">
        <nav aria-label="活动资料列表" className="contextDocumentList">
          {activeDocuments.map((document) => {
            const documentStatus = contextStatus(document.narrative_context);
            return (
              <button
                key={document.id}
                type="button"
                className={document.id === selected?.id ? "selected" : ""}
                aria-current={document.id === selected?.id ? "true" : undefined}
                disabled={navigationLocked}
                onClick={() => setSelectedId(document.id)}
              >
                <span>
                  <b>{document.name}</b>
                  <small>v{document.version} · {documentRoles.find(([value]) => value === document.document_role)?.[1] || document.document_role}</small>
                </span>
                <em className={documentStatus.tone}>{documentStatus.label}</em>
              </button>
            );
          })}
        </nav>

        <div className="contextEditor" aria-busy={loading || inferring}>
          {loading || !draft || !selected || !selectedContextReady ? (
            <div className="contextLoading">正在读取资料上下文…</div>
          ) : (
            <>
              <div className={`contextOrigin ${status.tone}`} role="status">
                <b>{status.label}</b>
                <p>{status.detail}</p>
              </div>

              {error && (
                <div ref={errorRef} className="contextError" role="alert" tabIndex={-1}>
                  <b>资料上下文没有保存</b>
                  <p>{error}</p>
                  <button type="button" onClick={() => void loadContext(selected)}>重新读取</button>
                </div>
              )}
              {inferenceError && (
                <div ref={errorRef} className="contextError" role="alert" tabIndex={-1}>
                  <b>AI 资料识别未完成</b>
                  <p>{inferenceError.message}</p>
                  {inferenceError.recovery === "reload" && (
                    <button type="button" onClick={() => void loadContext(selected)}>重新读取</button>
                  )}
                  {inferenceError.recovery === "provider" && (
                    <button type="button" onClick={onOpenProvider}>前往模型连接</button>
                  )}
                  {inferenceError.recovery === "retry" && (
                    <button type="button" onClick={() => void inferContext()}>重新识别</button>
                  )}
                </div>
              )}
              <div className="contextAnnouncement" aria-live="polite">{announcement}</div>

              {status.tone !== "confirmed" && (
                <div className="contextInferenceAction">
                  <div>
                    <b>让 AI 先读一遍资料</b>
                    <p>只生成可核对的资料类型、版本与分支建议，不会替你确认，也不会改写正文。</p>
                  </div>
                  <button
                    className="contextInferenceButton"
                    type="button"
                    disabled={editorDisabled}
                    onClick={() => void inferContext()}
                  >
                    {inferring ? "AI 正在识别…" : "AI 识别资料"}
                  </button>
                </div>
              )}
              {inferring && (
                <p className="contextInferenceProgress" role="status" aria-live="polite">
                  正在读取当前文稿并核对原文依据，请勿切换资料…
                </p>
              )}

              {inference && (
                <section className="contextInferenceReview" aria-labelledby="context-inference-review-title">
                  <header>
                    <div>
                      <span>AI 推断待确认</span>
                      <h4 id="context-inference-review-title">为什么这样识别</h4>
                    </div>
                    <strong>置信度 {Math.round(inference.confidence * 100)}%</strong>
                  </header>
                  <p>{inference.reasoning}</p>
                  <ol aria-label="AI 识别使用的原文证据">
                    {inference.evidence.slice(0, 4).map((evidence, index) => (
                      <li key={`${evidence.document_id}:${evidence.line_start}:${index}`}>
                        <div>
                          <b>{evidence.document_name}：第 {evidence.line_start}{evidence.line_end === evidence.line_start ? "" : `–${evidence.line_end}`} 行</b>
                          <span>{evidence.supported_fields.map((field) => inferredFieldLabels[field] || field).join(" · ")}</span>
                        </div>
                        <blockquote>{evidence.text}</blockquote>
                      </li>
                    ))}
                  </ol>
                  {inference.evidence.length > 4 && (
                    <small>另有 {inference.evidence.length - 4} 条原文依据，已收纳在本次推断记录中。</small>
                  )}
                  <footer>本次识别使用 {inference.usage.total_tokens} Token；请以原文为准核对下方字段。</footer>
                </section>
              )}

              <div className="contextFields">
                <label>
                  <span>资料类型</span>
                  <select
                    value={draft.documentRole}
                    disabled={editorDisabled}
                    onChange={(event) => update("documentRole", event.target.value as NarrativeContextDraft["documentRole"])}
                  >
                    {documentRoles.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                  </select>
                  <small>设定和角色档案可作为高权威资料；正文还需说明发布状态。</small>
                </label>
                <label>
                  <span>发布状态</span>
                  <select
                    value={draft.publicationStatus}
                    disabled={editorDisabled}
                    onChange={(event) => update("publicationStatus", event.target.value as NarrativeContextDraft["publicationStatus"])}
                  >
                    {publicationStatuses.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                  </select>
                  <small>草稿/审阅中用于新稿审查；当前只有“已发布”正文会进入角色基线，已归档资料仅保留记录。</small>
                </label>
                <label>
                  <span>时间线</span>
                  <input value={draft.timelineKey} maxLength={80} disabled={editorDisabled} onChange={(event) => update("timelineKey", event.target.value)} />
                  <small>主线通常填写 main；平行世界使用不同标识。</small>
                </label>
                <label>
                  <span>活动 / 篇章标识（可选）</span>
                  <input value={draft.activityKey} maxLength={80} disabled={editorDisabled} onChange={(event) => update("activityKey", event.target.value)} placeholder="例如 summer_event" />
                </label>
                <label>
                  <span>版本标签（可选）</span>
                  <input value={draft.releaseKey} maxLength={80} disabled={editorDisabled} onChange={(event) => update("releaseKey", event.target.value)} placeholder="例如 2.1" />
                </label>
                <label>
                  <span>版本顺序（与标签同时填写）</span>
                  <input type="number" inputMode="numeric" min={0} step={1} value={draft.releaseOrdinal} disabled={editorDisabled} onChange={(event) => update("releaseOrdinal", event.target.value)} placeholder="例如 21" />
                </label>
                <label className="contextWideField">
                  <span>分支路径（可选）</span>
                  <input value={draft.branchPath} disabled={editorDisabled} onChange={(event) => update("branchPath", event.target.value)} placeholder="例如 主线 / 北港分支" />
                  <small>使用斜杠或逗号分隔层级；公开分支信息可以在这里明确标注。</small>
                </label>
                <label>
                  <span>互斥分支组（可选）</span>
                  <input value={draft.exclusiveGroup} maxLength={80} disabled={editorDisabled} onChange={(event) => update("exclusiveGroup", event.target.value)} placeholder="例如 chapter_8_choice" />
                </label>
              </div>

              <label className="contextConfirmation">
                <input
                  type="checkbox"
                  checked={draft.confirmed}
                  disabled={editorDisabled}
                  onChange={(event) => update("confirmed", event.target.checked)}
                />
                <span>
                  <b>我已核对这份资料的身份与故事位置</b>
                  <small>勾选后才会成为可用于角色基线或新稿审查的正式上下文。</small>
                </span>
              </label>

              <div className="contextActions">
                <button className="quietPrimary" type="button" disabled={editorDisabled} onClick={() => void save()}>
                  {saving ? "保存中…" : draft.confirmed ? "保存并确认" : "保存草稿"}
                </button>
                <small>保存会生成新的上下文修订，不会改写文稿内容。</small>
              </div>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
