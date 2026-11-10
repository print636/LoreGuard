import { documentRoles } from "../../documentContext.ts";
import { publicationStatusNames } from "../documents/libraryModel.ts";

export type MaterialSource = {
  document_id: string | null; document_name: string | null; document_version: number | null;
  document_role: string | null; story_scope: string | null; batch_role: string | null; publication_status: string | null;
};
export type ImpactSource = {
  key: string; kind: "input" | "confirmed_source" | "added" | "not_included";
  frozen: MaterialSource | null; current: MaterialSource | null; changes: string[];
  selected_for_recheck: boolean | null; content_changed: boolean | null; identity_comparison_available: boolean;
};
export type ImpactIssue = { id: string; title: string; report_class: "formal" | "review_clue"; source_keys: string[]; evidence_indices: number[] };
export type ImpactPage<T> = { items: T[]; offset: number; limit: number; total: number; has_more: boolean };
export type MaterialImpact = {
  schema_version: "material-change-impact-v1"; run_id: string; project_id: string;
  comparison_status: "complete" | "unavailable"; semantic_impact_complete: false;
  confirmed_traits_auto_invalidated: false; confirmed_traits_changed: boolean | null;
  recheck: { inputs_changed: boolean | null; can_prepare: boolean; reason: string | null };
  sources: ImpactPage<ImpactSource>; affected_issues: ImpactPage<ImpactIssue>;
  summary: { changed_sources: number; added_sources: number; not_included_sources: number; affected_issues: number; unresolved_evidence: number; unresolved_sources: number };
};
export const IMPACT_PAGE_SIZE = 30;
const changes = ["content_changed", "version_changed", "document_replaced", "identity_changed", "selection_changed", "inactive", "missing", "unavailable"];
const count = (n: unknown): n is number => Number.isSafeInteger(n) && (n as number) >= 0;
const optionalString = (v: unknown) => v === null || typeof v === "string";
const optionalBoolean = (v: unknown) => v === null || typeof v === "boolean";
function source(value: MaterialSource | null): boolean {
  return value === null || !!value && typeof value === "object" &&
    [value.document_id, value.document_name, value.document_role, value.story_scope, value.batch_role, value.publication_status].every(optionalString) &&
    (value.document_version === null || count(value.document_version));
}
function page<T>(value: ImpactPage<T>, offset: number, valid: (item: T) => boolean): boolean {
  return !!value && value.offset === offset && value.limit === IMPACT_PAGE_SIZE && count(value.total) &&
    typeof value.has_more === "boolean" && Array.isArray(value.items) && value.items.length <= value.limit &&
    value.items.length <= value.total && value.items.every(valid);
}
export function normalizeMaterialImpact(raw: unknown, projectId: string, runId: string, offset = 0, issueOffset = 0): MaterialImpact {
  const value = raw as MaterialImpact;
  if (!value || value.schema_version !== "material-change-impact-v1" || value.project_id !== projectId || value.run_id !== runId ||
      !["complete", "unavailable"].includes(value.comparison_status) || value.semantic_impact_complete !== false ||
      value.confirmed_traits_auto_invalidated !== false || !optionalBoolean(value.confirmed_traits_changed) ||
      !value.recheck || !optionalBoolean(value.recheck.inputs_changed) || typeof value.recheck.can_prepare !== "boolean" || !optionalString(value.recheck.reason) ||
      !value.summary || ![value.summary.changed_sources, value.summary.added_sources, value.summary.not_included_sources, value.summary.affected_issues, value.summary.unresolved_evidence, value.summary.unresolved_sources].every(count) ||
      !page(value.sources, offset, item => !!item && typeof item.key === "string" && ["input", "confirmed_source", "added", "not_included"].includes(item.kind) &&
        source(item.frozen) && source(item.current) && Array.isArray(item.changes) && item.changes.every(c => changes.includes(c)) && optionalBoolean(item.selected_for_recheck) && optionalBoolean(item.content_changed) && typeof item.identity_comparison_available === "boolean") ||
      !page(value.affected_issues, issueOffset, item => !!item && typeof item.id === "string" && typeof item.title === "string" && ["formal", "review_clue"].includes(item.report_class) &&
        Array.isArray(item.source_keys) && item.source_keys.every(k => typeof k === "string") && Array.isArray(item.evidence_indices) && item.evidence_indices.every(count))) {
    throw new TypeError("无法核对资料变化的项目、报告或分页范围。");
  }
  return value;
}
export function materialImpactPath(runId: string, offset = 0, issueOffset = 0): string {
  return `/api/v1/analysis-runs/${encodeURIComponent(runId)}/material-impact?offset=${offset}&limit=${IMPACT_PAGE_SIZE}&issue_offset=${issueOffset}&issue_limit=${IMPACT_PAGE_SIZE}`;
}
export function canPrepareRecheck(value: MaterialImpact | null): boolean {
  return value?.comparison_status === "complete" && value.recheck.can_prepare === true && value.recheck.inputs_changed === true;
}
export const impactChangeNames: Record<string, string> = {
  content_changed: "正文变化", version_changed: "已有新版本", document_replaced: "文稿已替换", identity_changed: "资料身份或适用范围变化",
  selection_changed: "审查选材变化", inactive: "原版本已停用", missing: "当前资料中未找到", unavailable: "无法核对",
};
export function sourceLabel(value: MaterialSource | null): string {
  return value ? `${value.document_name || "名称未记录"}${value.document_version === null ? " · 版本未记录" : ` · v${value.document_version}`}` : "无对应版本";
}
export function sourceContext(source: MaterialSource | null): string {
  if (!source) return "";
  const role = source.document_role ? documentRoles.find(([key]) => key === source.document_role)?.[1] || "其他资料类型" : "用途未记录";
  const publication = source.publication_status ? publicationStatusNames[source.publication_status] || "其他发布状态" : "发布状态未记录";
  return [source.batch_role === "target" ? "目标新稿" : source.batch_role === "background" ? "比对依据" : "",
    role, publication, `故事范围：${source.story_scope || "未记录"}`].filter(Boolean).join(" · ");
}
export function impactHeadline(value: MaterialImpact): string {
  if (value.comparison_status === "unavailable") return "暂时无法核对这份报告的资料变化";
  if (value.recheck.inputs_changed === true) return "这份报告的审查输入已有变化";
  if (value.summary.changed_sources || value.summary.not_included_sources) return "有资料变化，尚未纳入这份报告";
  return "未发现明确的输入变化";
}
