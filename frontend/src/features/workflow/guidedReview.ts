import type { DocumentRole } from "../../documentContext.ts";

export const publicationStatuses = [
  ["unknown", "待确认"],
  ["draft", "草稿"],
  ["in_review", "审阅中"],
  ["published", "已发布"],
  ["retired", "已归档"],
] as const;

export type PublicationStatus = (typeof publicationStatuses)[number][0];
export type ReviewSensitivity = "conservative" | "balanced" | "exploratory";
export type ReviewMode = "baseline_build" | "draft_review";

export type NarrativeScope = {
  schema_version: 1;
  timeline_key: string;
  release?: { key: string; ordinal: number } | null;
  branch?: { path: string[]; exclusive_group?: string | null } | null;
  activity_key?: string | null;
};

export type NarrativeContextInferenceEvidence = {
  document_id: string;
  document_name: string;
  line_start: number;
  line_end: number;
  text: string;
  supported_fields: string[];
};

export type NarrativeContextInference = {
  confidence: number;
  reasoning: string;
  evidence: NarrativeContextInferenceEvidence[];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
};

export type NarrativeContext = {
  revision?: number;
  context_revision?: number;
  resolution_state: "unresolved" | "inferred" | "confirmed";
  origin?: "explicit" | "deterministic_import" | "model_inferred" | "legacy" | string;
  authority_tier?: string;
  publication_status: PublicationStatus;
  scope: NarrativeScope;
  scope_sha256?: string;
  inference_confidence?: number | null;
  inference?: NarrativeContextInference | null;
  legacy_document_role?: string;
};

export type NarrativeContextInferenceResponse = {
  document_id: string;
  document_role: DocumentRole;
  suggestion: NarrativeContext;
  usage: NarrativeContextInference["usage"];
};

export type GuidedDocument = {
  id: string;
  name: string;
  version: number;
  active: boolean;
  document_role: DocumentRole;
  story_scope: string;
  narrative_context?: NarrativeContext;
};

export type NarrativeContextDraft = {
  documentRole: DocumentRole;
  publicationStatus: PublicationStatus;
  timelineKey: string;
  releaseKey: string;
  releaseOrdinal: string;
  branchPath: string;
  exclusiveGroup: string;
  activityKey: string;
  confirmed: boolean;
};

export type AnalysisRunRequest = {
  mode: ReviewMode;
  sensitivity: ReviewSensitivity;
  target_document_ids?: string[];
};

const keyPattern = /^[A-Za-z0-9_.:\-\u4e00-\u9fff]{1,80}$/u;
const roleValues = new Set<DocumentRole>([
  "chapter",
  "canon",
  "character_profile",
  "reference",
]);
const publicationStatusValues = new Set<PublicationStatus>(
  publicationStatuses.map(([value]) => value),
);

function clean(value: string): string {
  return value.trim();
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function finiteNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function nonNegativeInteger(value: unknown): number | null {
  return Number.isSafeInteger(value) && Number(value) >= 0 ? Number(value) : null;
}

function inferenceUsage(value: unknown): NarrativeContextInference["usage"] | null {
  const source = record(value);
  if (!source) return null;
  const promptTokens = nonNegativeInteger(source.prompt_tokens);
  const completionTokens = nonNegativeInteger(source.completion_tokens);
  const totalTokens = nonNegativeInteger(source.total_tokens);
  if (promptTokens === null || completionTokens === null || totalTokens === null) {
    return null;
  }
  return {
    prompt_tokens: promptTokens,
    completion_tokens: completionTokens,
    total_tokens: totalTokens,
  };
}

/**
 * Treat model-backed API data as untrusted. A malformed suggestion must surface
 * as an error instead of silently becoming an empty or apparently confirmed form.
 */
export function normalizeNarrativeContextInference(
  value: unknown,
): NarrativeContextInferenceResponse {
  const payload = record(value);
  const suggestion = record(payload?.suggestion);
  const scope = record(suggestion?.scope);
  const inference = record(suggestion?.inference);
  const documentId = typeof payload?.document_id === "string"
    ? payload.document_id.trim()
    : "";
  const documentRole = payload?.document_role;
  const publicationStatus = suggestion?.publication_status;
  const revision = nonNegativeInteger(suggestion?.revision);
  const confidence = finiteNumber(inference?.confidence);
  const reasoning = typeof inference?.reasoning === "string"
    ? inference.reasoning.trim()
    : "";
  const usage = inferenceUsage(payload?.usage);
  const nestedUsage = inferenceUsage(inference?.usage);
  const timelineKey = typeof scope?.timeline_key === "string"
    ? scope.timeline_key.trim()
    : "";

  if (
    !documentId ||
    typeof documentRole !== "string" ||
    !roleValues.has(documentRole as DocumentRole) ||
    suggestion?.resolution_state !== "inferred" ||
    suggestion?.origin !== "model_inferred" ||
    typeof publicationStatus !== "string" ||
    !publicationStatusValues.has(publicationStatus as PublicationStatus) ||
    revision === null ||
    scope?.schema_version !== 1 ||
    !timelineKey ||
    confidence === null ||
    confidence < 0 ||
    confidence > 1 ||
    !reasoning ||
    !usage ||
    !nestedUsage ||
    !Array.isArray(inference?.evidence)
  ) {
    throw new TypeError("AI 返回的资料识别建议不完整，请重试或手动设置。");
  }

  const evidence = inference.evidence.slice(0, 6).map((item) => {
    const source = record(item);
    const evidenceDocumentId = typeof source?.document_id === "string"
      ? source.document_id.trim()
      : "";
    const documentName = typeof source?.document_name === "string"
      ? source.document_name.trim()
      : "";
    const lineStart = nonNegativeInteger(source?.line_start);
    const lineEnd = nonNegativeInteger(source?.line_end);
    const text = typeof source?.text === "string" ? source.text.trim() : "";
    const supportedFields = Array.isArray(source?.supported_fields)
      ? source.supported_fields.filter(
          (field): field is string => typeof field === "string" && Boolean(field.trim()),
        )
      : [];
    if (
      !evidenceDocumentId ||
      !documentName ||
      lineStart === null ||
      lineStart < 1 ||
      lineEnd === null ||
      lineEnd < lineStart ||
      !text ||
      !supportedFields.length
    ) {
      throw new TypeError("AI 返回的原文证据不完整，请重试或手动设置。");
    }
    return {
      document_id: evidenceDocumentId,
      document_name: documentName,
      line_start: lineStart,
      line_end: lineEnd,
      text,
      supported_fields: supportedFields,
    };
  });
  if (!evidence.length) {
    throw new TypeError("AI 没有返回可核对的原文证据，请重试或手动设置。");
  }

  return {
    document_id: documentId,
    document_role: documentRole as DocumentRole,
    suggestion: {
      ...(suggestion as NarrativeContext),
      revision,
      resolution_state: "inferred",
      origin: "model_inferred",
      publication_status: publicationStatus as PublicationStatus,
      scope: {
        ...(scope as NarrativeScope),
        schema_version: 1,
        timeline_key: timelineKey,
      },
      inference_confidence: confidence,
      inference: {
        confidence,
        reasoning,
        evidence,
        usage: nestedUsage,
      },
    },
    usage,
  };
}

export function contextRevision(context?: NarrativeContext): number {
  const value = context?.revision ?? context?.context_revision ?? 0;
  return Number.isSafeInteger(value) && value >= 0 ? value : 0;
}

export function isLoadedContextForDocument(
  loadedDocumentId: string | null,
  selectedDocumentId: string | null | undefined,
): boolean {
  return responseBelongsToSelectedDocument(loadedDocumentId, selectedDocumentId);
}

export function responseBelongsToSelectedDocument(
  requestDocumentId: string | null | undefined,
  selectedDocumentId: string | null | undefined,
): boolean {
  return Boolean(
    requestDocumentId &&
    selectedDocumentId &&
    requestDocumentId === selectedDocumentId,
  );
}

export function contextNavigationLocked(saving: boolean, inferring: boolean): boolean {
  return saving || inferring;
}

export function contextDraft(
  document: GuidedDocument,
  context = document.narrative_context,
): NarrativeContextDraft {
  const scope = context?.scope;
  return {
    documentRole: document.document_role || "reference",
    publicationStatus: context?.publication_status || "unknown",
    timelineKey: scope?.timeline_key || "main",
    releaseKey: scope?.release?.key || "",
    releaseOrdinal:
      scope?.release?.ordinal === undefined ? "" : String(scope.release.ordinal),
    branchPath: scope?.branch?.path?.join(" / ") || "",
    exclusiveGroup: scope?.branch?.exclusive_group || "",
    activityKey: scope?.activity_key || "",
    confirmed: context?.resolution_state === "confirmed",
  };
}

export function contextStatus(context?: NarrativeContext): {
  tone: "confirmed" | "attention" | "inferred";
  label: string;
  detail: string;
} {
  if (context?.resolution_state === "confirmed") {
    return {
      tone: "confirmed",
      label: "已人工确认",
      detail: "这份资料的身份与故事位置会进入冻结分析输入。",
    };
  }
  if (context?.origin === "model_inferred" || context?.resolution_state === "inferred") {
    return {
      tone: "inferred",
      label: "AI 推断待确认",
      detail: "模型推断不会自动成为正式上下文，请核对后保存。",
    };
  }
  return {
    tone: "attention",
    label: "待人工确认",
    detail: "当前没有可采用的 AI 上下文推断，请由创作者确认资料类型与故事位置。",
  };
}

function splitBranchPath(value: string): string[] {
  return value
    .split(/[/,，>＞]+/u)
    .map(clean)
    .filter(Boolean);
}

function assertKey(value: string, label: string, required = false) {
  if (!value && !required) return;
  if (!keyPattern.test(value)) {
    throw new Error(`${label}只能包含中英文、数字、点、冒号、下划线或连字符，最长 80 位。`);
  }
}

export function narrativeContextPayload(
  draft: NarrativeContextDraft,
  expectedRevision: number,
) {
  const timelineKey = clean(draft.timelineKey) || "main";
  const releaseKey = clean(draft.releaseKey);
  const releaseOrdinalText = clean(draft.releaseOrdinal);
  const branchPath = splitBranchPath(draft.branchPath);
  const exclusiveGroup = clean(draft.exclusiveGroup);
  const activityKey = clean(draft.activityKey);
  assertKey(timelineKey, "时间线标识", true);
  if (releaseKey || releaseOrdinalText) {
    assertKey(releaseKey, "版本标签", true);
    if (!/^\d+$/.test(releaseOrdinalText)) {
      throw new Error("版本顺序必须是 0 或更大的整数，并与版本标签一起填写。");
    }
  }
  for (const [index, part] of branchPath.entries()) {
    assertKey(part, `分支路径第 ${index + 1} 段`, true);
  }
  assertKey(exclusiveGroup, "互斥分支组");
  assertKey(activityKey, "活动标识");

  const scope: NarrativeScope = { schema_version: 1, timeline_key: timelineKey };
  if (releaseKey) {
    scope.release = { key: releaseKey, ordinal: Number(releaseOrdinalText) };
  }
  if (branchPath.length) {
    scope.branch = {
      path: branchPath,
      ...(exclusiveGroup ? { exclusive_group: exclusiveGroup } : {}),
    };
  } else if (exclusiveGroup) {
    throw new Error("填写互斥分支组前，请先填写至少一段分支路径。");
  }
  if (activityKey) scope.activity_key = activityKey;
  return {
    document_role: draft.documentRole,
    resolution_state: draft.confirmed ? "confirmed" : "unresolved",
    publication_status: draft.publicationStatus,
    scope,
    expected_revision: expectedRevision,
  };
}

export function isConfirmed(document: GuidedDocument): boolean {
  return document.narrative_context?.resolution_state === "confirmed";
}

export function isDraftDocument(document: GuidedDocument): boolean {
  const status = document.narrative_context?.publication_status;
  return (
    document.active &&
    document.document_role === "chapter" &&
    (status === "draft" || status === "in_review")
  );
}

export function isBaselineDocument(document: GuidedDocument): boolean {
  if (!document.active) return false;
  const status = document.narrative_context?.publication_status;
  if (document.document_role === "chapter") {
    return status === "published";
  }
  if (document.document_role === "canon" || document.document_role === "character_profile") {
    return status === "published" || status === "unknown";
  }
  return false;
}

export function guidedDocumentState(documents: GuidedDocument[]) {
  const active = documents.filter((document) => document.active);
  const baseline = active.filter(isBaselineDocument);
  const drafts = active.filter(isDraftDocument);
  const unresolved = active.filter((document) => !isConfirmed(document));
  const unresolvedBaseline = baseline.filter((document) => !isConfirmed(document));
  const confirmedDrafts = drafts.filter(isConfirmed);
  return {
    active,
    baseline,
    drafts,
    unresolved,
    unresolvedBaseline,
    confirmedDrafts,
  };
}

export function analysisRunRequest(
  mode: ReviewMode,
  sensitivity: ReviewSensitivity,
  targetDocumentIds: string[] = [],
): AnalysisRunRequest {
  if (mode === "baseline_build") return { mode, sensitivity };
  const uniqueTargets = Array.from(
    new Set(targetDocumentIds.map(clean).filter(Boolean)),
  );
  if (!uniqueTargets.length) throw new Error("请至少选择一份待审新稿。");
  return { mode, sensitivity, target_document_ids: uniqueTargets };
}

export function hasCompletedBaselineRun(
  runs: Array<{
    status?: string;
    mode?: string | null;
    review_batch?: { mode?: string | null } | null;
  }>,
): boolean {
  return runs.some(
    (run) =>
      run.status === "completed" &&
      (run.review_batch?.mode ?? run.mode) === "baseline_build",
  );
}

export type BaselineRunSummary = {
  id: string;
  status?: string;
  mode?: string | null;
  input_snapshot_available?: boolean;
  input_documents?: Array<{
    document_id: string;
    document_version: number;
    document_role: string;
    batch_role?: string;
    narrative_context?: {
      context_revision?: number;
      resolution_state?: string;
      publication_status?: string;
      scope_sha256?: string;
    };
  }>;
  review_batch?: {
    mode?: string | null;
    background_document_ids?: string[];
  } | null;
};

function baselineRunMatchesDocuments(
  run: BaselineRunSummary,
  baseline: GuidedDocument[],
): boolean {
  if (run.input_snapshot_available !== true || !Array.isArray(run.input_documents)) {
    return false;
  }
  const declaredBackground = new Set(
    Array.isArray(run.review_batch?.background_document_ids)
      ? run.review_batch.background_document_ids
      : [],
  );
  const frozen = run.input_documents.filter((item) =>
    item.batch_role === "background" || declaredBackground.has(item.document_id),
  );
  if (frozen.length !== baseline.length) return false;
  const byId = new Map(frozen.map((item) => [item.document_id, item]));
  if (byId.size !== baseline.length) return false;
  return baseline.every((document) => {
    if (!isConfirmed(document)) return false;
    const input = byId.get(document.id);
    const currentContext = document.narrative_context;
    const frozenContext = input?.narrative_context;
    return Boolean(
      input &&
      input.document_version === document.version &&
      input.document_role === document.document_role &&
      currentContext &&
      frozenContext &&
      contextRevision(currentContext) === frozenContext.context_revision &&
      frozenContext.resolution_state === "confirmed" &&
      frozenContext.publication_status === currentContext.publication_status &&
      typeof currentContext.scope_sha256 === "string" &&
      currentContext.scope_sha256.length > 0 &&
      frozenContext.scope_sha256 === currentContext.scope_sha256
    );
  });
}

export function findCurrentBaselineRun<T extends BaselineRunSummary>(
  documents: GuidedDocument[],
  runs: T[],
): T | null {
  const baseline = guidedDocumentState(documents).baseline;
  if (!baseline.length || baseline.some((document) => !isConfirmed(document))) {
    return null;
  }
  return runs.find(
    (run) =>
      run.status === "completed" &&
      (run.review_batch?.mode ?? run.mode) === "baseline_build" &&
      baselineRunMatchesDocuments(run, baseline),
  ) || null;
}
