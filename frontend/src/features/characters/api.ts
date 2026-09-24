import { apiJson, apiJsonIdempotent } from "../../api/client.ts";
import type {
  CandidateDecisionIn,
  CandidateDecisionOut,
  CandidatePage,
  CharacterTraitAxis,
  CharacterTraitAxisPage,
  CharacterDetail,
  CharacterDimension,
  CharacterPage,
  CharacterProfileItem,
  CharacterReadiness,
  CharacterSummary,
  DriftIssuePage,
  DriftIssueSummary,
  ModelCoverage,
  NarrativeScopeRef,
  ProfileCandidate,
  ProfileCandidateStatus,
  ProfileEvidence,
} from "./types.ts";

type JsonRecord = Record<string, unknown>;

const knownDimensions = new Set<CharacterDimension>([
  "core_personality",
  "preference",
  "value",
  "speech_pattern",
  "behavior_boundary",
  "contextual_behavior",
  "current_state",
]);
const modelCoverages = new Set<ModelCoverage>([
  "full",
  "partial",
  "rules_only",
  "unknown",
]);
const candidateStatuses = new Set<ProfileCandidateStatus>([
  "pending",
  "confirmed",
  "rejected",
  "stale",
]);

function segment(value: string): string {
  return encodeURIComponent(value);
}

function projectRoot(projectId: string): string {
  return `/api/v1/projects/${segment(projectId)}`;
}

function characterRoot(projectId: string): string {
  return `${projectRoot(projectId)}/characters`;
}

function record(value: unknown): JsonRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonRecord)
    : null;
}

function requiredRecord(value: unknown, label: string): JsonRecord {
  const parsed = record(value);
  if (!parsed) throw new TypeError(`${label}格式不受支持`);
  return parsed;
}

function requiredItems(source: JsonRecord, label: string): unknown[] {
  if (!Array.isArray(source.items)) {
    throw new TypeError(`${label}缺少 items 数组`);
  }
  return source.items;
}

function text(value: unknown, fallback = ""): string {
  return typeof value === "string" && value.trim() ? value.trim() : fallback;
}

function optionalText(value: unknown): string | null {
  const parsed = text(value);
  return parsed || null;
}

function integer(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : fallback;
}

function nullableInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : null;
}

function positiveInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 1
    ? value
    : null;
}

function fraction(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.min(1, Math.max(0, value))
    : 0;
}

function stringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => text(item)).filter(Boolean);
}

function coverage(value: unknown): ModelCoverage {
  return modelCoverages.has(value as ModelCoverage)
    ? (value as ModelCoverage)
    : "unknown";
}

function readiness(value: unknown): CharacterReadiness {
  if (
    value === "ready" ||
    value === "no_documents" ||
    value === "no_completed_run" ||
    value === "not_generated"
  ) return value;
  throw new TypeError("角色列表缺少可识别的 readiness 状态");
}

export function normalizeCharacterDimension(value: unknown): CharacterDimension {
  // Never coerce a new server value to the nearest known dimension. Doing so
  // could turn an unknown semantic claim into a confirmable profile fact.
  return knownDimensions.has(value as CharacterDimension)
    ? (value as CharacterDimension)
    : "unknown";
}

function normalizeScope(
  value: unknown,
  seed: string,
  fallbackId?: unknown,
): NarrativeScopeRef {
  const source = record(value) || {};
  const release = record(source.release);
  const branch = record(source.branch);
  const timeline = text(source.timeline_key);
  const releaseKey = text(release?.key);
  const activity = text(source.activity_key);
  const branchPath = stringList(branch?.path);
  const parts: string[] = [];
  if (timeline) parts.push(timeline === "main" ? "主时间线" : `时间线 ${timeline}`);
  if (releaseKey) parts.push(`版本 ${releaseKey}`);
  if (activity) parts.push(`活动 ${activity}`);
  if (branchPath.length) parts.push(`分支 ${branchPath.join(" / ")}`);
  const explicitLabel = text(source.label);
  const semanticKey = [timeline, releaseKey, activity, ...branchPath]
    .filter(Boolean)
    .join(":");
  return {
    scope_id:
      text(source.scope_id) ||
      text(fallbackId) ||
      `scope:${seed}:${semanticKey || "unresolved"}`,
    label: explicitLabel || parts.join(" · ") || "作用域未标注",
    version_id: optionalText(source.version_id) || releaseKey || null,
    activity_id: optionalText(source.activity_id) || activity || null,
    branch_path: branchPath.length ? branchPath : stringList(source.branch_path),
  };
}

function normalizeScopes(value: unknown, seed: string): NarrativeScopeRef[] {
  if (!Array.isArray(value)) return [];
  return value.map((scope, index) => normalizeScope(scope, `${seed}:${index}`));
}

function normalizeEvidence(
  value: unknown,
  index: number,
  fallbackScope?: unknown,
): ProfileEvidence | null {
  const source = record(value);
  if (!source) return null;
  const documentName = text(source.document_name);
  const excerpt = text(source.text);
  const lineStart = integer(source.line_start);
  const lineEnd = integer(source.line_end);
  if (!documentName || !excerpt || lineStart < 1 || lineEnd < lineStart) return null;
  const documentId = text(source.document_id, `unresolved-document-${index}`);
  return {
    document_id: documentId,
    document_name: documentName,
    document_version: nullableInteger(source.document_version),
    document_role: text(source.document_role, "unknown"),
    publication_status: text(source.publication_status, "unknown"),
    authority_level: text(
      source.authority_level ?? source.authority_tier,
      "unresolved",
    ),
    story_scope: normalizeScope(
      source.story_scope ?? source.scope ?? fallbackScope,
      `evidence:${documentId}:${lineStart}:${lineEnd}`,
      source.scope_sha256,
    ),
    line_start: lineStart,
    line_end: lineEnd,
    text: excerpt,
  };
}

function normalizeEvidenceList(
  value: unknown,
  fallbackScope?: unknown,
  label = "角色证据",
): ProfileEvidence[] {
  if (value === undefined || value === null) return [];
  if (!Array.isArray(value)) throw new TypeError(`${label}必须是数组`);
  const parsed = value.map((item, index) =>
    normalizeEvidence(item, index, fallbackScope),
  );
  if (parsed.some((item) => item === null)) {
    throw new TypeError(`${label}包含无法核对的证据定位`);
  }
  return parsed as ProfileEvidence[];
}

function normalizeCharacterSummary(value: unknown): CharacterSummary | null {
  const source = record(value);
  if (!source) return null;
  const id = text(source.id ?? source.character_key);
  const canonicalName = text(
    source.canonical_name ?? source.character_display_name,
    id,
  );
  if (!id || !canonicalName) return null;
  return {
    id,
    canonical_name: canonicalName,
    aliases: stringList(source.aliases),
    applicable_scopes: normalizeScopes(source.applicable_scopes, `character:${id}`),
    confirmed_item_count: integer(
      source.confirmed_item_count ?? source.confirmed_trait_count,
    ),
    pending_candidate_count: integer(source.pending_candidate_count),
    drift_issue_count: nullableInteger(source.drift_issue_count),
    profile_revision: nullableInteger(source.profile_revision),
    updated_at: text(source.updated_at),
  };
}

function candidateStatus(value: unknown): ProfileCandidateStatus {
  if (value === "superseded") return "stale";
  return candidateStatuses.has(value as ProfileCandidateStatus)
    ? (value as ProfileCandidateStatus)
    : "stale";
}

export function normalizeProfileCandidate(
  value: unknown,
  options: { characterId?: string; inheritedCoverage?: ModelCoverage } = {},
): ProfileCandidate {
  const source = requiredRecord(value, "角色归纳候选");
  const id = text(source.id);
  const characterId = text(
    source.character_id ?? source.character_key,
    options.characterId || "",
  );
  if (!id || !characterId) {
    throw new TypeError("角色归纳候选缺少稳定标识");
  }
  const dimension = normalizeCharacterDimension(
    source.dimension ?? source.trait_type,
  );
  const traitKey = text(source.trait_key);
  const traitValue = text(source.value);
  const statement = text(
    source.statement,
    traitKey && traitValue
      ? `${traitKey}：${traitValue}`
      : traitValue || traitKey || "候选表述缺失",
  );
  const rawScope = source.scope;
  const scopes = Array.isArray(source.scopes)
    ? normalizeScopes(source.scopes, `candidate:${id}`)
    : rawScope
      ? [normalizeScope(rawScope, `candidate:${id}`, source.scope_sha256)]
      : [];
  const supportingEvidence = normalizeEvidenceList(
    source.supporting_evidence ?? source.evidence,
    rawScope,
    "支持证据",
  );
  const contraryEvidence = normalizeEvidenceList(
    source.contrary_evidence,
    rawScope,
    "反向证据",
  );
  const status = candidateStatus(source.status ?? source.review_state);
  const origin =
    source.origin === "explicit_setting" || source.origin === "history_inference"
      ? source.origin
      : "unknown";
  const sourceRunId = text(source.source_run_id);
  const sourceSnapshotRevision = text(
    source.source_snapshot_revision ??
      source.scope_sha256 ??
      source.candidate_fingerprint,
  );
  const explicitlyReviewable =
    typeof source.reviewable === "boolean" ? source.reviewable : true;
  const revision = nullableInteger(source.revision);
  const rawAxisId = source.approved_axis_id;
  const approvedAxisId = optionalText(rawAxisId);
  const rawAxisVersion = source.approved_axis_version;
  const approvedAxisVersion = positiveInteger(rawAxisVersion);
  if (
    (rawAxisId !== undefined && rawAxisId !== null && !approvedAxisId) ||
    (rawAxisVersion !== undefined && rawAxisVersion !== null && approvedAxisVersion === null) ||
    Boolean(approvedAxisId) !== Boolean(approvedAxisVersion)
  ) {
    throw new TypeError("角色归纳候选的作者轴绑定无效");
  }
  const reviewable = Boolean(
    explicitlyReviewable &&
      status === "pending" &&
      dimension !== "unknown" &&
      origin !== "unknown" &&
      sourceRunId &&
      sourceSnapshotRevision &&
      supportingEvidence.length &&
      revision !== null,
  );
  let unreviewableReason = optionalText(source.unreviewable_reason);
  if (!reviewable && !unreviewableReason) {
    if (dimension === "unknown") {
      unreviewableReason = "服务端返回了未识别的角色特征类型。";
    } else if (origin === "unknown") {
      unreviewableReason = "候选没有提供可核对的归纳来源。";
    } else if (status === "stale") {
      unreviewableReason = "来源资料或候选状态已经变化。";
    } else if (!supportingEvidence.length) {
      unreviewableReason = "候选没有可核对的原文证据。";
    } else if (!sourceRunId || !sourceSnapshotRevision) {
      unreviewableReason = "候选缺少可验证的冻结运行来源。";
    } else if (revision === null) {
      unreviewableReason = "候选缺少并发审核所需的修订号。";
    }
  }
  return {
    id,
    character_id: characterId,
    dimension,
    origin,
    contexts: Array.from(new Set(stringList(source.contexts))),
    statement,
    model_trait_key: optionalText(source.trait_key),
    polarity:
      source.polarity === "positive" ||
      source.polarity === "negative" ||
      source.polarity === "neutral" ||
      source.polarity === "unclear"
        ? source.polarity
        : null,
    comparison_key: optionalText(source.comparison_key),
    authority_tier:
      source.authority_tier === "core_canon" ||
      source.authority_tier === "formal_record"
        ? source.authority_tier
        : null,
    valid_from_release_ordinal: nullableInteger(source.valid_from_release_ordinal),
    valid_until_release_ordinal: nullableInteger(source.valid_until_release_ordinal),
    approved_axis_id: approvedAxisId,
    approved_axis_version: approvedAxisVersion,
    confidence: fraction(source.confidence),
    rationale: text(
      source.rationale,
      "该候选由冻结分析输入归纳；确认前请核对原文证据与适用作用域。",
    ),
    limitations: stringList(source.limitations),
    scopes,
    supporting_evidence: supportingEvidence,
    contrary_evidence: contraryEvidence,
    status,
    reviewable,
    unreviewable_reason: unreviewableReason,
    source_run_id: sourceRunId,
    source_snapshot_revision: sourceSnapshotRevision,
    model_coverage: coverage(
      source.model_coverage ?? options.inheritedCoverage ?? "unknown",
    ),
    revision: revision ?? 0,
  };
}

function normalizeProfileItem(value: unknown): CharacterProfileItem | null {
  const source = record(value);
  if (!source) return null;
  const id = text(source.id);
  if (!id) return null;
  const traitKey = text(source.trait_key);
  const traitValue = text(source.value);
  const statement = text(
    source.statement,
    traitKey && traitValue ? `${traitKey}：${traitValue}` : traitValue || traitKey,
  );
  if (!statement) return null;
  const scopes = Array.isArray(source.scopes)
    ? normalizeScopes(source.scopes, `profile:${id}`)
    : source.scope
      ? [normalizeScope(source.scope, `profile:${id}`, source.scope_sha256)]
      : [];
  const evidence = normalizeEvidenceList(source.evidence, source.scope);
  return {
    id,
    dimension: normalizeCharacterDimension(
      source.dimension ?? source.trait_type,
    ),
    statement,
    approved_axis_id: optionalText(source.approved_axis_id),
    origin:
      source.origin === "explicit_profile" || source.origin === "explicit_setting"
        ? "explicit_profile"
        : "confirmed_inference",
    scopes,
    evidence_count: integer(source.evidence_count, evidence.length),
    confirmed_at: optionalText(source.confirmed_at ?? source.reviewed_at),
  };
}

export function characterTraitAxesPath(projectId: string): string {
  return `${projectRoot(projectId)}/character-trait-axes`;
}

export function normalizeCharacterTraitAxis(
  value: unknown,
  projectId: string,
): CharacterTraitAxis {
  const source = requiredRecord(value, "作者轴");
  const id = text(source.id);
  const name = text(source.display_name);
  const definition = text(source.definition);
  const version = positiveInteger(source.version);
  const digest = text(source.definition_sha256);
  if (
    !id ||
    source.project_id !== projectId ||
    source.trait_type !== "core_personality" ||
    !version ||
    !name ||
    !definition ||
    !/^[a-f0-9]{64}$/.test(digest)
  ) {
    throw new TypeError("作者轴格式无效或不属于当前项目");
  }
  return {
    id,
    project_id: projectId,
    trait_type: "core_personality",
    version,
    display_name: name,
    definition,
    definition_sha256: digest,
    created_at: optionalText(source.created_at),
  };
}

export async function fetchCharacterTraitAxes(
  projectId: string,
  input: { limit?: number; offset?: number } = {},
  signal?: AbortSignal,
): Promise<CharacterTraitAxisPage> {
  const limit = input.limit ?? 100;
  const offset = input.offset ?? 0;
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  const source = requiredRecord(
    await apiJson<unknown>(`${characterTraitAxesPath(projectId)}?${params}`, { signal }),
    "作者轴列表",
  );
  const rawItems = requiredItems(source, "作者轴列表");
  if (
    nullableInteger(source.total) === null ||
    positiveInteger(source.limit) === null ||
    nullableInteger(source.offset) === null ||
    source.limit !== limit ||
    source.offset !== offset ||
    rawItems.length > limit ||
    (rawItems.length === 0 && offset < (source.total as number)) ||
    (source.total as number) < offset + rawItems.length
  ) {
    throw new TypeError("作者轴分页信息无效");
  }
  const items = rawItems.map((item) => normalizeCharacterTraitAxis(item, projectId));
  if (new Set(items.map((item) => item.id)).size !== items.length) {
    throw new TypeError("作者轴列表包含重复标识");
  }
  return { items, total: source.total as number, limit, offset };
}

export async function createCharacterTraitAxis(
  projectId: string,
  input: { display_name: string; definition: string },
): Promise<CharacterTraitAxis> {
  // The creation endpoint does not yet offer server-side idempotency. Never
  // replay an uncertain request automatically or a transport loss could create
  // two different author axes.
  const payload = await apiJson<unknown>(characterTraitAxesPath(projectId), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ trait_type: "core_personality", ...input }),
  });
  return normalizeCharacterTraitAxis(payload, projectId);
}

function pageFields(
  source: JsonRecord,
  itemCount: number,
) {
  const hasPagePagination =
    source.page !== undefined || source.page_size !== undefined;
  const hasOffsetPagination =
    source.offset !== undefined || source.limit !== undefined;
  if (!hasPagePagination && !hasOffsetPagination) {
    throw new TypeError("分页响应缺少 page/page_size 或 offset/limit");
  }
  const validPositiveInteger = (value: unknown): value is number =>
    typeof value === "number" && Number.isSafeInteger(value) && value >= 1;
  const validNonNegativeInteger = (value: unknown): value is number =>
    typeof value === "number" && Number.isSafeInteger(value) && value >= 0;

  let pageSize: number;
  let page: number;
  if (hasPagePagination) {
    if (!validPositiveInteger(source.page) || !validPositiveInteger(source.page_size)) {
      throw new TypeError("分页响应的 page 或 page_size 无效");
    }
    page = source.page;
    pageSize = source.page_size;
  } else {
    if (!validPositiveInteger(source.limit) || !validNonNegativeInteger(source.offset)) {
      throw new TypeError("分页响应的 offset 或 limit 无效");
    }
    pageSize = source.limit;
    page = Math.floor(source.offset / pageSize) + 1;
  }
  if (!validNonNegativeInteger(source.total) || source.total < itemCount) {
    throw new TypeError("分页响应的 total 无效");
  }
  if (typeof source.has_more !== "boolean") {
    throw new TypeError("分页响应缺少 has_more 布尔值");
  }
  if (itemCount > pageSize) {
    throw new TypeError("分页响应条目数超过 page_size");
  }
  return {
    page,
    page_size: pageSize,
    total: source.total,
    has_more: source.has_more,
  };
}

function characterPageFromArray(
  payload: unknown[],
  input: { page: number; pageSize: number; query?: string },
): CharacterPage {
  const all = payload
    .map(normalizeCharacterSummary)
    .filter((item): item is CharacterSummary => item !== null);
  if (all.length !== payload.length) {
    throw new TypeError("角色列表包含无法识别的角色条目");
  }
  const query = input.query?.trim().toLocaleLowerCase() || "";
  const filtered = query
    ? all.filter((item) =>
        [item.canonical_name, ...item.aliases].some((name) =>
          name.toLocaleLowerCase().includes(query),
        ),
      )
    : all;
  const start = (input.page - 1) * input.pageSize;
  return {
    items: filtered.slice(start, start + input.pageSize),
    page: input.page,
    page_size: input.pageSize,
    total: filtered.length,
    has_more: start + input.pageSize < filtered.length,
    readiness: all.length ? "ready" : "not_generated",
    model_coverage: "unknown",
    coverage_detail: null,
    source_run_id: null,
  };
}

function normalizeCharacterPage(
  payload: unknown,
  input: { page: number; pageSize: number; query?: string },
): CharacterPage {
  if (Array.isArray(payload)) return characterPageFromArray(payload, input);
  const source = requiredRecord(payload, "角色列表");
  const rawItems = requiredItems(source, "角色列表");
  const items = rawItems
    .map(normalizeCharacterSummary)
    .filter((item): item is CharacterSummary => item !== null);
  if (items.length !== rawItems.length) {
    throw new TypeError("角色列表包含无法识别的角色条目");
  }
  return {
    items,
    ...pageFields(source, items.length),
    readiness: readiness(source.readiness),
    model_coverage: coverage(source.model_coverage),
    coverage_detail: optionalText(source.coverage_detail),
    source_run_id: optionalText(source.source_run_id),
  };
}

function normalizeCharacterDetail(payload: unknown): CharacterDetail {
  const source = requiredRecord(payload, "角色档案");
  const rawProfileItems = Array.isArray(source.profile_items)
    ? source.profile_items
    : Array.isArray(source.confirmed_traits)
      ? source.confirmed_traits
      : null;
  if (!rawProfileItems) {
    throw new TypeError("角色档案缺少 profile_items 或 confirmed_traits 数组");
  }
  const profileItems = rawProfileItems
    .map(normalizeProfileItem)
    .filter((item): item is CharacterProfileItem => item !== null);
  if (profileItems.length !== rawProfileItems.length) {
    throw new TypeError("角色档案包含无法识别的已确认条目");
  }
  const canonicalSummary = normalizeCharacterSummary(source.character);
  const rawSummary = normalizeCharacterSummary({
    ...source,
    confirmed_item_count: profileItems.length,
  });
  const character = canonicalSummary || rawSummary;
  if (!character) throw new TypeError("角色档案缺少角色标识");
  const firstTrait = record(rawProfileItems[0]);
  const scopes = profileItems.flatMap((item) => item.scopes);
  if (!character.applicable_scopes.length && scopes.length) {
    character.applicable_scopes = Array.from(
      new Map(scopes.map((scope) => [scope.scope_id, scope])).values(),
    );
  }
  return {
    character,
    profile_items: profileItems,
    source_run_id: optionalText(source.source_run_id ?? firstTrait?.source_run_id),
    source_snapshot_revision: optionalText(
      source.source_snapshot_revision ??
        firstTrait?.scope_sha256 ??
        firstTrait?.candidate_fingerprint,
    ),
    model_coverage: coverage(source.model_coverage),
    coverage_detail: optionalText(source.coverage_detail),
  };
}

function normalizeCandidatePage(
  payload: unknown,
  characterId: string,
): CandidatePage {
  const source = requiredRecord(payload, "角色归纳候选列表");
  const inheritedCoverage = coverage(source.model_coverage);
  const rawItems = requiredItems(source, "角色归纳候选列表");
  const items = rawItems.map((item) =>
    normalizeProfileCandidate(item, { characterId, inheritedCoverage }),
  );
  return {
    items,
    ...pageFields(source, items.length),
    model_coverage: inheritedCoverage,
    coverage_detail: optionalText(source.coverage_detail),
  };
}

function nestedCharacterId(metadata: JsonRecord): string {
  const baseline = record(metadata.baseline);
  const candidate = record(metadata.candidate);
  return text(
    metadata.character_id ??
      metadata.character_key ??
      metadata.character ??
      baseline?.character_id ??
      baseline?.character_key ??
      baseline?.character ??
      candidate?.character_id ??
      candidate?.character_key ??
      candidate?.character,
  );
}

function normalizeDriftIssue(
  value: unknown,
  requestedCharacterId: string,
): DriftIssueSummary | null {
  const source = record(value);
  if (!source) return null;
  const metadata = record(source.metadata) || {};
  const characterId = text(source.character_id) || nestedCharacterId(metadata);
  // The current endpoint is project-scoped. If an item cannot be attributed to
  // this character, omit it rather than leaking another character's report.
  if (!characterId || characterId !== requestedCharacterId) return null;
  const issueId = text(source.issue_id ?? source.id);
  const runId = text(source.run_id);
  if (!issueId || !runId) return null;
  const evidence = normalizeEvidenceList(
    source.evidence_preview ?? source.evidence,
    source.scope ?? metadata.scope,
  );
  const rawFeedback = text(source.feedback_status);
  const feedbackStatus =
    rawFeedback === "accepted" ||
    rawFeedback === "false_positive" ||
    rawFeedback === "resolved"
      ? rawFeedback
      : "unreviewed";
  return {
    issue_id: issueId,
    run_id: runId,
    character_id: characterId,
    title: text(source.title, "未命名角色漂移问题"),
    severity: text(source.severity, "unknown"),
    confidence: fraction(source.confidence),
    explanation: text(source.explanation, "该问题没有提供可展示的解释。"),
    scope: normalizeScope(
      source.scope ?? metadata.scope,
      `issue:${issueId}`,
      source.scope_id ?? metadata.scope_sha256,
    ),
    baseline_profile_item_ids: stringList(
      source.baseline_profile_item_ids ?? metadata.baseline_profile_item_ids,
    ),
    evidence_preview: evidence,
    feedback_status: feedbackStatus,
  };
}

function normalizeDriftIssuePage(
  payload: unknown,
  characterId: string,
): DriftIssuePage {
  const source = requiredRecord(payload, "角色漂移问题列表");
  const rawItems = requiredItems(source, "角色漂移问题列表");
  const items = rawItems
    .map((item) => normalizeDriftIssue(item, characterId))
    .filter((item): item is DriftIssueSummary => item !== null);
  if (items.length !== rawItems.length) {
    throw new TypeError(
      "角色漂移问题包含无法定位到当前角色的条目",
    );
  }
  return {
    items,
    ...pageFields(source, items.length),
    model_coverage: coverage(source.model_coverage),
    coverage_detail: optionalText(source.coverage_detail),
  };
}

export function characterApiPaths(
  projectId: string,
  characterId?: string,
  candidateId?: string,
) {
  const root = characterRoot(projectId);
  const character = characterId ? `${root}/${segment(characterId)}` : root;
  const candidates = `${character}/profile-candidates`;
  const candidate = candidateId
    ? `${candidates}/${segment(candidateId)}`
    : candidates;
  return {
    root,
    character,
    candidates,
    candidate,
    decisions: `${candidate}/decisions`,
    driftIssues: `${projectRoot(projectId)}/drift-issues`,
  };
}

export async function fetchCharacters(
  projectId: string,
  input: { page: number; pageSize?: number; query?: string },
  signal?: AbortSignal,
): Promise<CharacterPage> {
  const pageSize = input.pageSize || 30;
  const params = new URLSearchParams({
    page: String(input.page),
    page_size: String(pageSize),
  });
  if (input.query?.trim()) params.set("query", input.query.trim());
  const payload = await apiJson<unknown>(
    `${characterApiPaths(projectId).root}?${params.toString()}`,
    { signal },
  );
  return normalizeCharacterPage(payload, { ...input, pageSize });
}

export async function fetchCharacter(
  projectId: string,
  characterId: string,
  signal?: AbortSignal,
): Promise<CharacterDetail> {
  const payload = await apiJson<unknown>(
    characterApiPaths(projectId, characterId).character,
    { signal },
  );
  return normalizeCharacterDetail(payload);
}

export async function fetchProfileCandidates(
  projectId: string,
  characterId: string,
  input: { page: number; pageSize?: number },
  signal?: AbortSignal,
): Promise<CandidatePage> {
  const pageSize = input.pageSize || 20;
  const params = new URLSearchParams({
    state: "pending",
    status: "pending",
    page: String(input.page),
    page_size: String(pageSize),
    limit: String(pageSize),
    offset: String((input.page - 1) * pageSize),
  });
  const payload = await apiJson<unknown>(
    `${characterApiPaths(projectId, characterId).candidates}?${params.toString()}`,
    { signal },
  );
  return normalizeCandidatePage(payload, characterId);
}

export async function fetchProfileCandidate(
  projectId: string,
  characterId: string,
  candidateId: string,
  signal?: AbortSignal,
): Promise<ProfileCandidate> {
  const payload = await apiJson<unknown>(
    characterApiPaths(projectId, characterId, candidateId).candidate,
    { signal },
  );
  return normalizeProfileCandidate(payload, { characterId });
}

export async function submitCandidateDecision(
  projectId: string,
  characterId: string,
  candidateId: string,
  decision: CandidateDecisionIn,
): Promise<CandidateDecisionOut> {
  const payload = await apiJsonIdempotent<unknown>(
    characterApiPaths(projectId, characterId, candidateId).decisions,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(decision),
    },
  );
  const source = requiredRecord(payload, "角色归纳审核结果");
  return {
    candidate: normalizeProfileCandidate(source.candidate, { characterId }),
    created_profile_item: source.created_profile_item
      ? normalizeProfileItem(source.created_profile_item)
      : null,
    profile_revision: nullableInteger(source.profile_revision) ?? undefined,
    decision_id: optionalText(source.decision_id) ?? undefined,
    deduplicated:
      typeof source.deduplicated === "boolean"
        ? source.deduplicated
        : undefined,
  };
}

export async function fetchDriftIssues(
  projectId: string,
  characterId: string,
  input: { page: number; pageSize?: number },
  signal?: AbortSignal,
): Promise<DriftIssuePage> {
  const pageSize = input.pageSize || 20;
  const params = new URLSearchParams({
    character_key: characterId,
    page: String(input.page),
    page_size: String(pageSize),
    limit: String(pageSize),
    offset: String((input.page - 1) * pageSize),
  });
  const payload = await apiJson<unknown>(
    `${characterApiPaths(projectId, characterId).driftIssues}?${params.toString()}`,
    { signal },
  );
  return normalizeDriftIssuePage(payload, characterId);
}
