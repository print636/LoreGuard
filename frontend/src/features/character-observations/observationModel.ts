export const OBSERVATION_PAGE_SIZE = 50;
export type RecordingStatus = "in_progress" | "complete" | "partial" | "failed" | "not_applicable" | "legacy_unavailable";
export type ObservationRequest = { projectId: string; runId: string; offset: number };
export type FrozenExcerpt = { documentName: string; version: number; lineStart: number; lineEnd: number; text: string };
export type ObservationMatch = { status: "matched" | "not_matched" | "unresolved"; explanation: string; baseline: { character: string; statement: string; dimension: string; evidence: FrozenExcerpt[] } | null };
export type Observation = { ordinal: number; character: string; statement: string; dimension: string; context: string; source: FrozenExcerpt; matches: ObservationMatch[] };
export type ObservationCoverage = { omittedBaselineCount: number; notEvaluatedCount: number; reasons: Array<{ explanation: string; count: number }> };
export type ObservationPage = { status: RecordingStatus; total: number; offset: number; hasMore: boolean; items: Observation[]; coverage: ObservationCoverage | null };

const dimensionNames: Record<string, string> = {
  core_personality: "核心人格", preference: "稳定偏好", value: "价值观", relationship_attitude: "关系态度",
  motivation_goal: "长期动机与目标", speech_pattern: "语言表现", behavior_boundary: "行为边界",
  contextual_behavior: "情境表现", current_state: "当前状态",
};
const reasonNames: Record<string, string> = {
  matched: "已关联到本次运行冻结的角色设定，仅表示进入比较，不代表一致或存在冲突。",
  unmatched_character_alias: "记录中的角色称呼尚未对应到角色档案。",
  ambiguous_character_alias: "记录中的角色称呼可能对应多位角色，无法确定关联。",
  no_confirmed_baseline: "本次运行没有可用于比较的已确认角色设定。",
  character_mismatch: "这条记录与该设定中的角色不同。",
  dimension_mismatch: "这条记录与该设定描述的特征类型不同。",
  comparison_identity_unavailable: "缺少可核对的特征方向或对象，尚不能关联。",
  comparison_mismatch: "这条记录与该设定描述的方向或对象不同。",
  axis_direction_ambiguous: "特征方向尚不明确，暂不能关联。",
  source_scope_unavailable: "原文的适用范围不明确，暂不能关联。",
  lower_authority_shadowed: "本次比较采用了优先级更高的设定。",
  release_unknown: "尚不能确认这条记录对应的剧情发布位置。",
  release_inapplicable: "该设定不适用于这条记录对应的剧情发布位置。",
  scope_unknown: "尚不能确认记录与设定的适用范围是否一致。",
  scope_incompatible: "记录与设定的适用范围不同。",
  observation_limit: "本次已达到记录处理上限，部分关联未完成。",
  baseline_not_evaluated: "这条记录与该设定的关联尚未完成。",
  approved_axis_binding_mismatch: "记录与作者确认的特征方向尚未对应。",
  trait_identity_mismatch: "这条记录与该设定描述的特征不同。",
  object_identity_unavailable: "记录中的行为对象尚不明确，暂不能关联。",
  author_alignment_required: "需要先由作者确认特征方向，才能关联这条记录。",
  scoped_axis_review_disabled: "本次未开展带适用范围的特征方向比较。",
  equivalent_axis_collapsed: "相同特征方向的设定已合并采用，未重复比较。",
  lower_authority_baseline_shadowed: "本次比较采用了优先级更高的设定。",
  baseline_limit: "本次已达到设定比较上限，部分关联未完成。",
};
const coverageReasonNames: Record<string, string> = {
  author_alignment_required: "特征方向需要作者确认",
  scoped_axis_review_disabled: "本次未开展限定适用范围的设定比较",
  lower_authority_baseline_shadowed: "比较时采用了优先级更高的设定",
  equivalent_axis_collapsed: "相同特征方向的设定已合并采用",
  baseline_limit: "已达到本次设定比较数量上限",
};

export function observationDimension(value: unknown): string {
  return typeof value === "string" ? dimensionNames[value] || "角色表现" : "角色表现";
}

/** Only strip a known duplicated machine prefix; never display internal keys. */
function statementText(value: unknown, traitKey: unknown): string {
  let text = requiredText(value);
  if (typeof traitKey === "string" && traitKey) {
    for (const prefix of [`trait_key=${traitKey}`, `trait_key: ${traitKey}`, `${traitKey}：`, `${traitKey}:`]) {
      if (text.startsWith(prefix)) text = text.slice(prefix.length).replace(/^[\s:：;；,，]+/, "");
    }
  }
  // A machine annotation is not author-facing prose. Keep the validated original excerpt below.
  return /\b(?:trait_key|comparison_key)\s*[:=]/i.test(text) || !text ? "表述暂无法显示，请核对下方原文。" : text;
}

export function observationRecordingCopy(status: RecordingStatus, total: number): { label: string; detail: string; tone: string } {
  switch (status) {
    case "complete": return { label: "记录已完成", detail: total ? "已保存本次识别出的角色行为与陈述。记录完成不代表角色审查完整或没有异常。" : "本次未保存到已接纳的角色行为或陈述。这不能说明没有角色异常，也不能代替审查结论。", tone: "complete" };
    case "in_progress": return { label: "尚在记录", detail: "本次记录尚未结束，以下仅为当前已保存的部分。可刷新查看后续记录。", tone: "pending" };
    case "partial": return { label: "记录不完整", detail: "部分角色行为或设定关联记录未能完整保存，可能有遗漏。记录保存情况与实际审查范围应分别核对。", tone: "warning" };
    case "failed": return { label: "记录未能完成", detail: "本次记录过程失败，可能仍保留部分记录。不能据此判断角色是否存在异常。", tone: "warning" };
    case "legacy_unavailable": return { label: "未保存观察快照", detail: "这次历史运行没有可读取的角色行为快照，无法还原当时的观察记录；这不等于零条记录或没有异常。", tone: "unavailable" };
    case "not_applicable": return { label: "本次不适用", detail: "这次运行不记录新稿角色行为；角色设定归纳与角色行为观察属于不同的处理环节。", tone: "unavailable" };
  }
}

export function observationScope(identity: { mode: string; user: { id: string }; workspace: { id: string } }, projectId: string, runId: string, enabled: boolean): string | null {
  return identity.mode === "required" && identity.user.id && identity.workspace.id && projectId && runId && enabled
    ? JSON.stringify([identity.user.id, identity.workspace.id, projectId, runId]) : null;
}

export function observationPath(request: ObservationRequest): string {
  if (!request.projectId || !request.runId || !integer(request.offset)) throw new TypeError("invalid observation request");
  return `/api/v1/analysis-runs/${encodeURIComponent(request.runId)}/character-observations?offset=${request.offset}&limit=${OBSERVATION_PAGE_SIZE}`;
}

function record(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new TypeError("invalid observation response");
  return value as Record<string, unknown>;
}
function requiredText(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) throw new TypeError("invalid observation text");
  return value;
}
function integer(value: unknown): value is number { return Number.isSafeInteger(value) && Number(value) >= 0; }
function coverageCount(value: unknown): number {
  if (!integer(value) || value > 5000) throw new TypeError("invalid observation coverage count");
  return value;
}
function observationCoverage(value: unknown): ObservationCoverage | null {
  if (value === undefined || value === null) return null;
  const coverage = record(value);
  const omittedBaselineCount = coverageCount(coverage.omitted_baseline_count ?? 0);
  const notEvaluatedCount = coverageCount(coverage.baseline_not_evaluated_count ?? 0);
  const excluded = record(coverage.excluded_baseline_counts ?? {});
  // Iterate the fixed author-facing allowlist, never server keys or free-form diagnostics.
  const reasons = Object.entries(coverageReasonNames).flatMap(([key, explanation]) => {
    const count = coverageCount(excluded[key] ?? 0);
    return count > 0 ? [{ explanation, count }] : [];
  });
  if (notEvaluatedCount + reasons.reduce((sum, reason) => sum + reason.count, 0) > omittedBaselineCount) throw new TypeError("invalid observation coverage total");
  return omittedBaselineCount > 0 ? { omittedBaselineCount, notEvaluatedCount, reasons } : null;
}
function excerpt(value: unknown, version: unknown): FrozenExcerpt {
  const item = record(value);
  if (!integer(version) || version < 1 || !integer(item.line_start) || item.line_start < 1 || !integer(item.line_end) || item.line_end < item.line_start) throw new TypeError("invalid observation source");
  return { documentName: requiredText(item.document_name), version, lineStart: item.line_start, lineEnd: item.line_end, text: requiredText(item.text) };
}

/** Bind every response to its private run and page; return only an author-facing allowlist. */
export function normalizeObservationPage(value: unknown, request: ObservationRequest): ObservationPage {
  const payload = record(value), recording = record(payload.recording), page = record(payload.page);
  const statuses: RecordingStatus[] = ["in_progress", "complete", "partial", "failed", "not_applicable", "legacy_unavailable"];
  if (payload.schema_version !== "character-observations-v1" || payload.run_id !== request.runId || payload.project_id !== request.projectId ||
    !statuses.includes(recording.status as RecordingStatus) || !integer(page.total_items) || page.offset !== request.offset || page.limit !== OBSERVATION_PAGE_SIZE ||
    recording.observation_count !== page.total_items || typeof page.has_more !== "boolean" || !Array.isArray(payload.items) || payload.items.length > OBSERVATION_PAGE_SIZE ||
    payload.items.length !== Math.min(OBSERVATION_PAGE_SIZE, Math.max(0, page.total_items - request.offset)) || page.has_more !== (request.offset + payload.items.length < page.total_items)) throw new TypeError("invalid observation page");
  const ordinals = new Set<number>();
  const items = payload.items.map((raw): Observation => {
    const item = record(raw), source = record(item.source), evidence = record(item.evidence);
    if (!integer(item.ordinal) || ordinals.has(item.ordinal) || source.document_id !== evidence.document_id || source.document_name !== evidence.document_name ||
      typeof item.context !== "string" || !Array.isArray(item.matches) || item.matches.length < 1 || item.matches.length > 512) throw new TypeError("invalid observation item");
    ordinals.add(item.ordinal);
    const matches = item.matches.map((rawMatch): ObservationMatch => {
      const match = record(rawMatch);
      if (!["matched", "not_matched", "unresolved"].includes(String(match.status)) || (match.status === "matched" && (match.reason_code !== "matched" || !match.baseline))) throw new TypeError("invalid observation match");
      let baseline: ObservationMatch["baseline"] = null;
      if (match.baseline) {
        const frozen = record(record(match.baseline).payload);
        if (!Array.isArray(frozen.evidence) || frozen.evidence.length > 12) throw new TypeError("invalid frozen baseline");
        baseline = { character: requiredText(frozen.character_display_name), statement: statementText(frozen.value, frozen.trait_key), dimension: observationDimension(frozen.trait_type),
          evidence: frozen.evidence.map((rawEvidence) => { const entry = record(rawEvidence); return excerpt(entry, entry.document_version); }) };
      }
      return { status: match.status as ObservationMatch["status"], explanation: reasonNames[String(match.reason_code)] || "本次未能确定这条记录与该设定的关联，请结合原文核对。", baseline };
    });
    return { ordinal: item.ordinal, character: requiredText(item.character), statement: statementText(item.statement, item.trait_key), dimension: observationDimension(item.dimension), context: item.context,
      source: excerpt(evidence, source.document_version), matches };
  });
  return { status: recording.status as RecordingStatus, total: page.total_items, offset: request.offset, hasMore: page.has_more, items, coverage: observationCoverage(recording.coverage) };
}

export type ObservationLoadState = { status: "loading"; data: null } | { status: "ready"; data: ObservationPage } | { status: "error"; data: null };

/** Cleanup also blocks late results from transports that ignore AbortSignal. */
export function startObservationRequest(request: ObservationRequest, load: (path: string, signal: AbortSignal) => Promise<unknown>, emit: (state: ObservationLoadState) => void): () => void {
  const controller = new AbortController();
  let active = true;
  emit({ status: "loading", data: null });
  void Promise.resolve().then(() => active ? load(observationPath(request), controller.signal) : undefined).then((raw) => {
    if (!active) return;
    emit({ status: "ready", data: normalizeObservationPage(raw, request) });
  }).catch(() => { if (active) emit({ status: "error", data: null }); });
  return () => { active = false; controller.abort(); };
}
