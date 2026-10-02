import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, apiJson, SESSION_EXPIRED_EVENT } from "../../api/client";
import { documentRoles } from "../../documentContext";
import { registerBrowserNavigationBlocker, type BrowserNavigationApproval } from "../../routing";
import {
  contextDraft,
  contextDraftsEqual,
  contextNavigationLocked,
  contextRevision,
  contextStatus,
  isLoadedContextForDocument,
  narrativeContextPayload,
  normalizeNarrativeContextInference,
  readableNarrativeContext,
  publicationStatuses,
  requiresDedicatedChapterPublish,
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
  onLeaveGuardChange?: (guard: (() => BrowserNavigationApproval | null) | null) => void;
};

type InferenceRecovery = "reload" | "provider" | "retry" | "manual";

type InferenceFailure = {
  message: string;
  recovery: InferenceRecovery;
};

type PublishTarget = {
  projectId: string;
  documentId: string;
  documentName: string;
  documentVersion: number;
  contextRevision: number;
};

type PublishedContext = NarrativeContext & {
  document_id: string;
  document_role: string;
  document_version: number;
};

function publishFailure(error: unknown): string {
  if (error instanceof ApiError) {
    const payload = error.detail && typeof error.detail === "object"
      ? error.detail as { detail?: unknown }
      : null;
    const detail = payload?.detail && typeof payload.detail === "object"
      ? payload.detail as { code?: unknown }
      : null;
    switch (detail?.code) {
      case "document_version_conflict":
      case "narrative_context_revision_conflict":
        return "文稿版本或资料上下文已变化。发布结果未改写当前页面；请重新读取后核对再决定。";
      case "chapter_already_published":
        return "这份章节已在其他页面发布。请重新读取最新状态，勿重复提交。";
      case "document_not_active":
        return "这份文稿已不再是活动版本。请重新读取并选择当前版本。";
      case "document_not_publishable":
      case "chapter_context_unconfirmed":
      case "chapter_not_draft_or_in_review":
        return "文稿已不满足发布条件。请重新读取并确认其类型、发布状态和上下文。";
    }
    if (error.status === 404) return "文稿或项目已不可访问。请重新读取项目资料。";
  }
  return "发布结果暂无法确认；请重新读取文稿状态后再决定，勿直接重复提交。";
}

function requestMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 409) {
      const payload = error.detail && typeof error.detail === "object"
        ? error.detail as { detail?: unknown }
        : null;
      const detail = payload?.detail && typeof payload.detail === "object"
        ? payload.detail as { code?: unknown }
        : null;
      if (detail?.code === "chapter_publish_endpoint_required") {
        return "当前章节不能通过普通资料保存升格为已发布。请先保存为已确认草稿或审阅中，再使用“作者定稿并发布”入口。";
      }
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

function writeResultUnknown(reason: unknown): boolean {
  return !(reason instanceof ApiError && reason.status >= 400 && reason.status < 500);
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
  onLeaveGuardChange,
}: Props) {
  const activeDocuments = useMemo(
    () => documents.filter((document) => document.active),
    [documents],
  );
  const [selectedId, setSelectedId] = useState("");
  const [acceptedDocument, setAcceptedDocument] = useState<GuidedDocument | null>(null);
  const selected =
    activeDocuments.find((document) => document.id === selectedId) ||
    (acceptedDocument?.id === selectedId ? acceptedDocument : null) ||
    activeDocuments[0] ||
    null;
  const selectedIdRef = useRef("");
  selectedIdRef.current = selected?.id || "";
  const [remoteContext, setRemoteContext] = useState<NarrativeContext | null>(null);
  const [draft, setDraft] = useState<NarrativeContextDraft | null>(null);
  const [baseline, setBaseline] = useState<NarrativeContextDraft | null>(null);
  const [loadedDocumentId, setLoadedDocumentId] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [inferring, setInferring] = useState(false);
  const [publishing, setPublishing] = useState(false);
  const [publishTarget, setPublishTarget] = useState<PublishTarget | null>(null);
  const [publishError, setPublishError] = useState("");
  const [error, setError] = useState("");
  const [inferenceError, setInferenceError] = useState<InferenceFailure | null>(null);
  const [announcement, setAnnouncement] = useState("");
  const [writeNeedsReview, setWriteNeedsReview] = useState(false);
  const errorRef = useRef<HTMLDivElement | null>(null);
  const publishErrorRef = useRef<HTMLDivElement | null>(null);
  const publishDialogRef = useRef<HTMLDialogElement | null>(null);
  const publishTriggerRef = useRef<HTMLButtonElement | null>(null);
  const publishInFlightRef = useRef(false);
  const projectIdRef = useRef(projectId);
  projectIdRef.current = projectId;
  const aliveRef = useRef(true);
  const requestRef = useRef(0);
  const editRef = useRef(0);
  const dirty = !!draft && !!baseline && !contextDraftsEqual(draft, baseline);
  const dirtyRef = useRef(false);
  dirtyRef.current = dirty;
  const busyRef = useRef(false);
  busyRef.current = saving || inferring || publishing;
  const currentDocumentRef = useRef<GuidedDocument | null>(null);
  currentDocumentRef.current = activeDocuments.find((document) => document.id === selected?.id) || null;
  const selectedObsolete = !!selected && !currentDocumentRef.current;
  const leaveBlockerRef = useRef(() => {
    if (!dirtyRef.current && !busyRef.current) return true;
    return window.confirm(busyRef.current
      ? "资料身份操作仍在进行，结果可能已写入服务器。离开后本页修改将丢失，返回时需重新读取核对。确定离开吗？"
      : "资料身份还有未保存修改，仅保留在本页内存。确定放弃修改并离开吗？");
  });

  useEffect(() => {
    aliveRef.current = true;
    const expire = () => { aliveRef.current = false; requestRef.current += 1; };
    window.addEventListener(SESSION_EXPIRED_EVENT, expire);
    const unregister = registerBrowserNavigationBlocker(leaveBlockerRef.current);
    onLeaveGuardChange?.(() => {
      if (!leaveBlockerRef.current()) return null;
      const stamp = JSON.stringify([projectIdRef.current, selectedIdRef.current, editRef.current]);
      return {
        blocker: leaveBlockerRef.current,
        stillValid: () => aliveRef.current && stamp === JSON.stringify([projectIdRef.current, selectedIdRef.current, editRef.current]),
      };
    });
    return () => {
      aliveRef.current = false;
      requestRef.current += 1;
      window.removeEventListener(SESSION_EXPIRED_EVENT, expire);
      unregister();
      onLeaveGuardChange?.(null);
    };
  }, []);

  useEffect(() => {
    if (!dirty && !busyRef.current) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ""; };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty, saving, inferring, publishing]);

  function requestCurrent(project: string, documentId: string, generation: number): boolean {
    return aliveRef.current && projectIdRef.current === project &&
      selectedIdRef.current === documentId && requestRef.current === generation;
  }

  function mutationCurrent(project: string, documentId: string, generation: number): boolean {
    return requestCurrent(project, documentId, generation) && currentDocumentRef.current?.id === documentId;
  }

  function confirmReplacement(action: string): boolean {
    return !dirtyRef.current || window.confirm(`${action}将放弃本页尚未保存的资料身份修改，不会自动合并。确定继续吗？`);
  }

  function chooseDocument(document: GuidedDocument) {
    if (document.id === selected?.id || busyRef.current || !confirmReplacement("切换资料")) return;
    requestRef.current += 1;
    editRef.current += 1;
    setAcceptedDocument(null);
    setRemoteContext(null);
    setDraft(null);
    setBaseline(null);
    setLoadedDocumentId(null);
    setWriteNeedsReview(false);
    setSelectedId(document.id);
  }

  function reread() {
    if (!selected || selectedObsolete || busyRef.current || !confirmReplacement("重新读取并采纳服务器最新内容")) return;
    void loadContext(selected);
  }

  useEffect(() => {
    if (!selected) {
      setSelectedId("");
      setRemoteContext(null);
      setDraft(null);
      setBaseline(null);
      setLoadedDocumentId(null);
      return;
    }
    if (selectedId !== selected.id) setSelectedId(selected.id);
  }, [selected?.id]);

  async function loadContext(document: GuidedDocument, signal?: AbortSignal) {
    const targetProject = projectId;
    const generation = ++requestRef.current;
    setLoading(true);
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
      if (!readableNarrativeContext(payload.current)) throw new TypeError("资料身份响应缺少可核对的上下文。");
      if (!requestCurrent(targetProject, document.id, generation) || signal?.aborted) return;
      const nextDraft = contextDraft(document, payload.current);
      setAcceptedDocument(document);
      setRemoteContext(payload.current);
      setDraft(nextDraft);
      setBaseline(nextDraft);
      setLoadedDocumentId(document.id);
      setWriteNeedsReview(false);
      editRef.current += 1;
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === "AbortError") return;
      if (!requestCurrent(targetProject, document.id, generation)) return;
      setError("无法读取最新资料身份。当前修改和已读取基准保持不变，请重试读取；不会以旧列表内容冒充最新结果。");
    } finally {
      if (
        !signal?.aborted &&
        requestCurrent(targetProject, document.id, generation)
      ) setLoading(false);
    }
  }

  useEffect(() => {
    if (!selected || !projectId) return;
    const controller = new AbortController();
    void loadContext(selected, controller.signal);
    return () => controller.abort();
  }, [projectId, selected?.id]);

  useEffect(() => {
    // A route or selected-document change must not keep a confirmation dialog
    // pointing at an earlier project/document.
    if (publishDialogRef.current?.open) publishDialogRef.current.close();
    setPublishTarget(null);
    setPublishError("");
  }, [projectId, selected?.id]);

  function update<K extends keyof NarrativeContextDraft>(
    key: K,
    value: NarrativeContextDraft[K],
  ) {
    setDraft((current) => (current ? { ...current, [key]: value } : current));
    editRef.current += 1;
    if (!writeNeedsReview) {
      setError("");
      setInferenceError(null);
    }
    setAnnouncement("");
  }

  async function save() {
    if (
      !selected ||
      !draft ||
      loadedDocumentId !== selected.id ||
      selectedObsolete || disabled || loading || writeNeedsReview ||
      saving ||
      inferring ||
      publishing
    ) return;
    const document = selected;
    const targetProject = projectId;
    const generation = ++requestRef.current;
    const draftSnapshot = { ...draft };
    let dispatched = false;
    if (requiresDedicatedChapterPublish(
      document.document_role,
      remoteContext || document.narrative_context,
      draftSnapshot.documentRole,
      draftSnapshot.publicationStatus,
    )) {
      setError("当前草稿不能通过普通保存直接转成已发布。请先保存其他改动，再使用下方“作者定稿并发布”入口确认。");
      requestAnimationFrame(() => errorRef.current?.focus());
      return;
    }
    try {
      setSaving(true);
      setError("");
      setAnnouncement("");
      const body = narrativeContextPayload(
        draftSnapshot,
        contextRevision(remoteContext || document.narrative_context),
      );
      dispatched = true;
      const context = await apiJson<NarrativeContext>(
        `${contextPath(projectId, document.id)}/revisions`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        },
      );
      if (!context || !["unresolved", "inferred", "confirmed"].includes(context.resolution_state) || !context.scope || contextRevision(context) <= contextRevision(remoteContext || document.narrative_context)) {
        throw new TypeError("资料身份响应无法核对，请重新读取服务器状态。");
      }
      if (!mutationCurrent(targetProject, document.id, generation)) return;
      onSaved(document.id, draftSnapshot.documentRole, context);
      const updatedDocument = { ...document, document_role: draftSnapshot.documentRole };
      const nextDraft = contextDraft(updatedDocument, context);
      setAcceptedDocument(updatedDocument);
      setRemoteContext(context);
      setDraft(nextDraft);
      setBaseline(nextDraft);
      editRef.current += 1;
      setAnnouncement(
        draftSnapshot.confirmed
          ? `已确认“${document.name}”的资料上下文。`
          : `已保存“${document.name}”，仍需人工确认后才能用于角色审查。`,
      );
    } catch (reason) {
      if (!requestCurrent(targetProject, document.id, generation)) return;
      const unknown = dispatched && writeResultUnknown(reason);
      if (unknown) setWriteNeedsReview(true);
      setError(unknown
        ? "保存结果待核对：请求可能已写入服务器，本页输入仍保留。请重新读取最新资料身份后核对，不要直接重复保存。"
        : requestMessage(reason));
      requestAnimationFrame(() => errorRef.current?.focus());
    } finally {
      if (requestCurrent(targetProject, document.id, generation)) setSaving(false);
    }
  }

  async function inferContext() {
    if (!selected || loadedDocumentId !== selected.id || selectedObsolete || disabled || loading || writeNeedsReview || inferring || saving || publishing) return;
    if (!confirmReplacement("AI 识别会保存一条待确认建议，并替换本页字段；此操作")) return;
    const document = selected;
    const targetProject = projectId;
    const generation = ++requestRef.current;
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
      if (!mutationCurrent(targetProject, document.id, generation)) return;
      onSaved(document.id, result.document_role, result.suggestion);
      const nextDraft = contextDraft(updatedDocument, result.suggestion);
      setAcceptedDocument(updatedDocument);
      setRemoteContext(result.suggestion);
      setDraft(nextDraft);
      setBaseline(nextDraft);
      editRef.current += 1;
      setLoadedDocumentId(document.id);
      setAnnouncement(`已生成“${document.name}”的 AI 资料建议，请核对依据后另行保存并确认。`);
    } catch (reason) {
      if (!requestCurrent(targetProject, document.id, generation)) return;
      if (writeResultUnknown(reason)) {
        setWriteNeedsReview(true);
        setInferenceError({ message: "AI 识别写入结果待核对。当前输入仍保留，但服务器可能已有待确认修订；请重新读取后核对，不要直接重复识别。", recovery: "reload" });
      } else setInferenceError(inferenceFailure(reason));
      requestAnimationFrame(() => errorRef.current?.focus());
    } finally {
      if (requestCurrent(targetProject, document.id, generation)) setInferring(false);
    }
  }

  function openPublishDialog() {
    if (
      !selected || !remoteContext || !draft ||
      selected.document_role !== "chapter" ||
      remoteContext.resolution_state !== "confirmed" ||
      !["draft", "in_review"].includes(remoteContext.publication_status) ||
      contextRevision(remoteContext) < 1 ||
      selected.version < 1 ||
      !contextDraftsEqual(draft, baseline) || selectedObsolete || writeNeedsReview || loading ||
      disabled || saving || inferring || publishing ||
      loadedDocumentId !== selected.id
    ) return;
    setPublishError("");
    setPublishTarget({
      projectId,
      documentId: selected.id,
      documentName: selected.name,
      documentVersion: selected.version,
      contextRevision: contextRevision(remoteContext),
    });
    publishDialogRef.current?.showModal();
  }

  function closePublishDialog() {
    if (publishInFlightRef.current) return;
    publishDialogRef.current?.close();
  }

  async function publishChapter() {
    if (!publishTarget || publishInFlightRef.current) return;
    const target = publishTarget;
    const generation = ++requestRef.current;
    publishInFlightRef.current = true;
    setPublishing(true);
    setPublishError("");
    try {
      const published = await apiJson<PublishedContext>(
        `/api/v1/projects/${encodeURIComponent(target.projectId)}/documents/${encodeURIComponent(target.documentId)}/publish`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            expected_revision: target.contextRevision,
            expected_document_version: target.documentVersion,
          }),
        },
      );
      if (
        published.document_id !== target.documentId ||
        published.document_role !== "chapter" ||
        published.document_version !== target.documentVersion ||
        published.resolution_state !== "confirmed" ||
        published.publication_status !== "published" ||
        contextRevision(published) <= target.contextRevision
      ) {
        throw new TypeError("发布响应与当前文稿不一致，请重新读取状态。");
      }
      if (!mutationCurrent(target.projectId, target.documentId, generation)) return;
      onSaved(target.documentId, "chapter", published);
      const nextDraft = contextDraft({ ...selected!, document_role: "chapter" }, published);
      setRemoteContext(published);
      setDraft(nextDraft);
      setBaseline(nextDraft);
      editRef.current += 1;
      setAnnouncement(`“${target.documentName}”已由你定稿并发布。后续新稿审查可将这份历史正文作为正式背景；系统未据此认定剧情无误。`);
      publishDialogRef.current?.close();
    } catch (reason) {
      if (!requestCurrent(target.projectId, target.documentId, generation)) return;
      if (writeResultUnknown(reason)) setWriteNeedsReview(true);
      setPublishError(publishFailure(reason));
      requestAnimationFrame(() => publishErrorRef.current?.focus());
    } finally {
      if (requestCurrent(target.projectId, target.documentId, generation)) {
        publishInFlightRef.current = false;
        setPublishing(false);
      }
    }
  }

  if (!selected) return null;
  const status = contextStatus(remoteContext || selected?.narrative_context);
  const inference = remoteContext?.inference || null;
  const selectedContextReady = isLoadedContextForDocument(loadedDocumentId, selected?.id);
  const editorDisabled = disabled || loading || saving || inferring || publishing || selectedObsolete || !selectedContextReady;
  const navigationLocked = contextNavigationLocked(saving || publishing, inferring);
  const unsavedContextChanges = dirty;
  const showPublishAction = Boolean(
    selected?.document_role === "chapter" &&
    remoteContext &&
    (remoteContext.publication_status === "draft" || remoteContext.publication_status === "in_review"),
  );
  const canPublish = Boolean(
    showPublishAction &&
    remoteContext?.resolution_state === "confirmed" &&
    contextRevision(remoteContext) >= 1 &&
    selected && selected.version >= 1 &&
    !unsavedContextChanges &&
    !writeNeedsReview &&
    !editorDisabled,
  );

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
                onClick={() => chooseDocument(document)}
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
          {dirty && (
            <p className="contextDraftNotice" role="status">资料身份有未保存修改 · 仅在本页内存，尚未写入项目；刷新或离开将丢失，不会跨刷新恢复。</p>
          )}
          {selectedObsolete && (
            <section className="contextObsoleteNotice" aria-label="旧版本资料身份修改">
              <b>正在保留旧文档“{selected.name}”v{selected.version} 的本页字段</b>
              <p>这份文档已不再是活动版本。字段没有套用到新稿，当前不能保存、AI 识别或发布；请先核对，再明确放弃旧字段并选择活动资料。</p>
              {activeDocuments.length > 0 && <button type="button" disabled={navigationLocked} onClick={() => chooseDocument(activeDocuments[0])}>放弃旧字段并选择活动资料</button>}
            </section>
          )}
          {error && (
            <div ref={errorRef} className="contextError" role="alert" tabIndex={-1}>
              <b>{writeNeedsReview ? "保存结果待核对" : "资料身份操作未完成"}</b>
              <p>{error}</p>
              <button type="button" disabled={navigationLocked || loading || selectedObsolete} onClick={reread}>重新读取</button>
            </div>
          )}
          {inferenceError && (
            <div ref={errorRef} className="contextError" role="alert" tabIndex={-1}>
              <b>AI 资料识别未完成</b>
              <p>{inferenceError.message}</p>
              {inferenceError.recovery === "reload" && <button type="button" disabled={navigationLocked || loading || selectedObsolete} onClick={reread}>重新读取</button>}
              {inferenceError.recovery === "provider" && <button type="button" onClick={onOpenProvider}>前往模型连接</button>}
              {inferenceError.recovery === "retry" && <button type="button" onClick={() => void inferContext()}>重新识别</button>}
            </div>
          )}
          {loading || !draft || !selected || !selectedContextReady ? (
            <div className="contextLoading">{loading ? "正在读取资料上下文…" : "尚未读取可核对的资料身份，请重新读取后再编辑。"}</div>
          ) : (
            <>
              <div className={`contextOrigin ${status.tone}`} role="status">
                <b>{status.label}</b>
                <p>{status.detail}</p>
              </div>

              <div className="contextAnnouncement" aria-live="polite">{announcement}</div>

              {status.tone !== "confirmed" && (
                <div className="contextInferenceAction">
                  <div>
                    <b>让 AI 先读一遍资料</b>
                    <p>会保存一条待确认的资料类型、版本与分支建议并替换本页字段，不会替你确认或改写正文。</p>
                  </div>
                  <button
                    className="contextInferenceButton"
                    type="button"
                    disabled={editorDisabled || writeNeedsReview}
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
                    {publicationStatuses
                      .filter(([value]) => !requiresDedicatedChapterPublish(
                        selected.document_role,
                        remoteContext || selected.narrative_context,
                        draft.documentRole,
                        value,
                      ))
                      .map(([value, label]) => <option key={value} value={value}>{label}</option>)}
                  </select>
                  <small>草稿/审阅中用于新稿审查。已有历史可标注“已发布”；当前草稿定稿请使用下方专用入口。已归档资料仅保留记录。</small>
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
                <button className="quietPrimary" type="button" disabled={editorDisabled || writeNeedsReview} onClick={() => void save()}>
                  {saving ? "保存中…" : draft.confirmed ? "保存并确认" : "保存草稿"}
                </button>
                {!error && inferenceError?.recovery !== "reload" && <button type="button" disabled={editorDisabled} onClick={reread}>重新读取</button>}
                <small>保存会生成新的上下文修订，不会改写文稿内容。</small>
              </div>
              {showPublishAction && (
                <section className="contextPublishAction" aria-label="作者定稿与发布">
                  <div>
                    <h4>作者定稿与发布</h4>
                    <p>发布后，这份章节会成为后续新稿审查可引用的已发布历史。审查报告只提供线索，不是发布许可或质量保证。</p>
                    {!canPublish && (
                      <small>{remoteContext?.resolution_state !== "confirmed"
                        ? "请先确认这份章节的资料上下文。"
                        : unsavedContextChanges
                          ? "请先保存当前上下文改动，再定稿发布。"
                          : "请先完成资料读取，再定稿发布。"}</small>
                    )}
                  </div>
                  <button
                    ref={publishTriggerRef}
                    type="button"
                    disabled={!canPublish}
                    onClick={openPublishDialog}
                  >作者定稿并发布…</button>
                </section>
              )}
              {selected.document_role === "chapter" && remoteContext?.publication_status === "published" && (
                <p className="contextPublishedNote" role="status">此章节已发布；后续新稿审查可将其作为已发布历史。定稿不代表系统认定剧情无误。</p>
              )}
              <dialog
                ref={publishDialogRef}
                className="contextPublishDialog"
                aria-labelledby="context-publish-title"
                onCancel={(event) => { if (publishInFlightRef.current) event.preventDefault(); }}
                onClose={() => {
                  setPublishTarget(null);
                  setPublishError("");
                  publishTriggerRef.current?.focus();
                }}
              >
                <h4 id="context-publish-title">确认由作者定稿并发布</h4>
                <p>将“{publishTarget?.documentName || selected.name}”v{publishTarget?.documentVersion || selected.version} 标为已发布，保留原有时间线与分支。此后它可作为后续新稿的正式历史背景。</p>
                <p className="contextPublishWarning">发布是作者决定，不表示系统认定剧情无误。即使没有审查报告、覆盖不完整或报告仍有问题线索，你仍可自行决定；请先自行核对文稿。</p>
                {publishError && (
                  <div ref={publishErrorRef} className="contextError" role="alert" tabIndex={-1}>
                    <b>发布状态未确认</b>
                    <p>{publishError}</p>
                    <button type="button" disabled={publishing} onClick={() => {
                      closePublishDialog();
                      reread();
                    }}>关闭并重新读取</button>
                  </div>
                )}
                <div className="contextPublishDialogActions">
                  <button type="button" disabled={publishing} onClick={closePublishDialog}>继续修改</button>
                  <button type="button" disabled={publishing || Boolean(publishError)} onClick={() => void publishChapter()}>
                    {publishing ? "发布中…" : "确认发布此章节"}
                  </button>
                </div>
              </dialog>
            </>
          )}
        </div>
      </div>
    </section>
  );
}
