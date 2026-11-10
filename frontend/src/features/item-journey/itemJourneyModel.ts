import type { Evidence } from "../../types/visualization.ts";

export type JourneyItem = { key: string; name: string; story_scope: string; record_count: number; issue_ids: string[] };
export type JourneyEvidence = { record_id: string; evidence: Evidence | null; document_version: number | null };
export type JourneyEvent = JourneyEvidence & {
  kind: "observed_possession" | "custody" | "transfer" | "retained_custody" | "uses" | "legacy";
  holder: string | null; transfer_from: string | null; time: string | null; through_time: string | null;
  retention_scope: "ongoing" | "bounded" | "unclear" | null;
  basis: "explicit" | "supported_inference" | "unverified";
  source_status: "verified" | "unavailable"; semantic_status: "supported" | "unverified";
};
export type JourneyPage = { offset: number; limit: number; total: number; has_more: boolean };
export type JourneyList = { schema_version: "item-journey-v1"; run_id: string; project_id: string; items: JourneyItem[]; page: JourneyPage;
  coverage: { scanned_records: number; total_records: number; truncated: boolean } };
export type JourneyDetail = { schema_version: "item-journey-v1"; run_id: string; project_id: string; item: JourneyItem;
  summary: { status: "ordered_evidence" | "partial" | "unavailable"; text: string; latest_record_ids: string[] };
  events: JourneyPage & { items: JourneyEvent[] };
  relations: Array<{ left_record_id: string; right_record_id: string; order: "before" | "same_time" | "after"; evidence_record_ids: string[] }>;
  relation_evidence: JourneyEvidence[];
  coverage: { processed_records: number; total_records: number; truncated: boolean; source_unavailable: number; proof_unavailable: number;
    unresolved_time_pairs: number; run_partial: boolean; relation_proof_unavailable: number };
};
export const ITEM_PAGE_SIZE = 30;
export const EVENT_PAGE_SIZE = 50;
const count = (value: unknown): value is number => Number.isSafeInteger(value) && (value as number) >= 0;
const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every(item => typeof item === "string");
const nullable = (value: unknown) => value === null || typeof value === "string";
const validItem = (value: JourneyItem) => !!value && typeof value.key === "string" && typeof value.name === "string" && typeof value.story_scope === "string" && count(value.record_count) && strings(value.issue_ids);
function validPage(value: JourneyPage, offset: number, limit: number, length: number): boolean {
  return !!value && value.offset === offset && value.limit === limit && count(value.total) && typeof value.has_more === "boolean" && length <= limit && length <= value.total;
}
function validEvidence(value: JourneyEvidence): boolean {
  const e = value?.evidence;
  return !!value && typeof value.record_id === "string" && (value.document_version === null || count(value.document_version) && value.document_version > 0) &&
    (e === null || !!e && typeof e.document_id === "string" && typeof e.document_name === "string" && typeof e.text === "string" && count(e.line_start) && e.line_start > 0 && count(e.line_end) && e.line_end >= e.line_start);
}
export function itemJourneysPath(runId: string, query = "", offset = 0, issueId: string | null = null): string {
  const params = new URLSearchParams({ q: query.trim(), offset: String(offset), limit: String(ITEM_PAGE_SIZE) });
  if (issueId) params.set("issue_id", issueId);
  return `/api/v1/analysis-runs/${encodeURIComponent(runId)}/item-journeys?${params}`;
}
export function itemJourneyPath(runId: string, itemKey: string, offset = 0): string {
  return `/api/v1/analysis-runs/${encodeURIComponent(runId)}/item-journeys/${encodeURIComponent(itemKey)}?offset=${offset}&limit=${EVENT_PAGE_SIZE}`;
}
export function normalizeJourneyList(raw: unknown, projectId: string, runId: string, offset: number): JourneyList {
  const value = raw as JourneyList;
  if (!value || value.schema_version !== "item-journey-v1" || value.project_id !== projectId || value.run_id !== runId || !Array.isArray(value.items) ||
    !value.items.every(validItem) || !validPage(value.page, offset, ITEM_PAGE_SIZE, value.items.length) || !value.coverage ||
    !count(value.coverage.scanned_records) || !count(value.coverage.total_records) || typeof value.coverage.truncated !== "boolean") throw new TypeError("物品目录与所选报告不一致。");
  return value;
}
export function normalizeJourneyDetail(raw: unknown, projectId: string, runId: string, key: string, offset: number): JourneyDetail {
  const value = raw as JourneyDetail;
  if (!value || value.schema_version !== "item-journey-v1" || value.project_id !== projectId || value.run_id !== runId || !validItem(value.item) || value.item.key !== key ||
    !value.summary || !["ordered_evidence", "partial", "unavailable"].includes(value.summary.status) || typeof value.summary.text !== "string" || !strings(value.summary.latest_record_ids) ||
    !value.events || !Array.isArray(value.events.items) || !validPage(value.events, offset, EVENT_PAGE_SIZE, value.events.items.length) || !value.events.items.every(e =>
      validEvidence(e) && ["observed_possession", "custody", "transfer", "retained_custody", "uses", "legacy"].includes(e.kind) &&
      [e.holder, e.transfer_from, e.time, e.through_time].every(nullable) && (e.retention_scope === null || ["ongoing", "bounded", "unclear"].includes(e.retention_scope)) &&
      ["explicit", "supported_inference", "unverified"].includes(e.basis) && ["verified", "unavailable"].includes(e.source_status) && ["supported", "unverified"].includes(e.semantic_status)) ||
    !Array.isArray(value.relations) || !value.relations.every(r => typeof r.left_record_id === "string" && typeof r.right_record_id === "string" && ["before", "same_time", "after"].includes(r.order) && strings(r.evidence_record_ids)) ||
    !Array.isArray(value.relation_evidence) || !value.relation_evidence.every(validEvidence) || !value.coverage ||
    ![value.coverage.processed_records, value.coverage.total_records, value.coverage.source_unavailable, value.coverage.proof_unavailable, value.coverage.unresolved_time_pairs, value.coverage.relation_proof_unavailable].every(count) ||
    typeof value.coverage.truncated !== "boolean" || typeof value.coverage.run_partial !== "boolean") throw new TypeError("物品历程与所选报告或来源不一致。");
  return value;
}
export const journeyKindNames: Record<JourneyEvent["kind"], string> = { observed_possession: "观察到持有", custody: "保管", transfer: "转交", retained_custody: "明确未交接", uses: "使用", legacy: "持有方式未区分" };
export function journeyEventLabel(event: JourneyEvent): string {
  const who = event.holder || "人物未明确";
  if (event.kind === "transfer") return `${event.transfer_from || "交出者未明确"}交给${who}`;
  return `${who} · ${journeyKindNames[event.kind]}`;
}
export function journeyRoute(search: string): { item: string | null; issue: string | null } {
  const params = new URLSearchParams(search);
  const id = (key: string) => /^[A-Za-z0-9_-]{1,200}$/.test(params.get(key) || "") ? params.get(key) : null;
  return { item: id("item"), issue: id("item_issue") };
}
export function journeySearch(item: string | null, issue: string | null = null): string {
  const params = new URLSearchParams({ view: "items" });
  if (item) params.set("item", item);
  if (issue) params.set("item_issue", issue);
  return `?${params}`;
}
