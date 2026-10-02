import type { DocumentRole } from "../../documentContext.ts";
import { documentContext } from "../../documentContext.ts";
import { ApiError, createIdempotencyKey } from "../../api/client.ts";
import { apiErrorDetail } from "../../app/session.ts";

export type ImportStatus = "queued" | "uploading" | "succeeded" | "failed" | "unknown";
export type ImportDefaults = {
  documentRole: DocumentRole | null;
  storyScope: string;
  replaceDocumentId: string;
};
export type ImportReceipt = { id: string; project_id: string; name: string; version: number; deduplicated?: boolean };
export type ImportFile = { name: string; size: number; lastModified: number };
export type ImportEntry<TFile extends ImportFile = File> = ImportDefaults & {
  id: string;
  operationKey: string;
  file: TFile;
  status: ImportStatus;
  message: string;
  receipt: ImportReceipt | null;
};
export const emptyImportDefaults: ImportDefaults = { documentRole: null, storyScope: "", replaceDocumentId: "" };

export function appendImportEntries<TFile extends ImportFile>(
  current: ImportEntry<TFile>[], files: Iterable<TFile>, defaults: ImportDefaults = emptyImportDefaults,
  key: () => string = createIdempotencyKey,
): { entries: ImportEntry<TFile>[]; duplicates: number } {
  const entries = [...current];
  let duplicates = 0;
  for (const file of files) {
    if (entries.some((entry) => entry.file === file || (entry.file.name === file.name && entry.file.size === file.size && entry.file.lastModified === file.lastModified))) {
      duplicates += 1;
      continue;
    }
    const operationKey = key();
    entries.push({ ...defaults, id: operationKey, operationKey, file, status: "queued", message: "", receipt: null });
  }
  return { entries, duplicates };
}

/** A changed payload is a new operation; ambiguous and successful operations are immutable. */
export function updateImportEntry<TFile extends ImportFile>(
  entry: ImportEntry<TFile>, patch: Partial<ImportDefaults>, key: () => string = createIdempotencyKey,
): ImportEntry<TFile> {
  if (["uploading", "succeeded", "unknown"].includes(entry.status)) return entry;
  const next = { ...entry, ...patch };
  if (next.documentRole === entry.documentRole && next.storyScope === entry.storyScope && next.replaceDocumentId === entry.replaceDocumentId) return entry;
  return { ...next, operationKey: key(), status: "queued", message: "", receipt: null };
}

export function importContextFields(entry: ImportDefaults): Record<string, string> {
  const fields: Record<string, string> = {};
  if (entry.documentRole !== null) fields.document_role = entry.documentRole;
  if (entry.storyScope.trim()) fields.story_scope = documentContext("reference", entry.storyScope).story_scope;
  if (entry.replaceDocumentId) fields.replace_document_id = entry.replaceDocumentId;
  return fields;
}

export function verifiedImportReceipt(value: unknown, projectId: string, filename: string): ImportReceipt | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const row = value as Record<string, unknown>;
  if (typeof row.id !== "string" || !row.id.trim() || row.project_id !== projectId || row.name !== filename || !Number.isSafeInteger(row.version) || Number(row.version) < 1) return null;
  return { id: row.id, project_id: projectId, name: filename, version: Number(row.version), ...(typeof row.deduplicated === "boolean" ? { deduplicated: row.deduplicated } : {}) };
}

export class ImportResultUnknown extends Error {
  constructor() { super("服务器响应不完整，不能确认这份文稿是否已导入。"); }
}

export function importFailure(error: unknown): { status: "failed" | "unknown"; message: string; stop: boolean } {
  if (error instanceof ApiError) {
    const detail = error.detail && typeof error.detail === "object" && "detail" in error.detail
      ? (error.detail as { detail?: { reason_code?: unknown } }).detail : null;
    if (error.status >= 500 || error.status === 408 || error.status < 400 || detail?.reason_code === "document_upload_idempotency_conflict") {
      return { status: "unknown", message: "结果待核对；请先安全重试同一导入，或检查项目文档。不要重新选择文件来重复上传。", stop: false };
    }
    return { status: "failed", message: `${apiErrorDetail(error)}${error.status === 401 ? " 请重新登录后核对已导入文档；未处理文件需要重新选择。" : " 修改设置或解决原因后可重试。"}`, stop: [401, 403, 404].includes(error.status) };
  }
  return { status: "unknown", message: "未收到可确认的导入结果；请安全重试同一导入，或检查项目文档。文件仍保留在本页内存中。", stop: false };
}

export function importCounts(entries: ImportEntry<ImportFile>[]) {
  return {
    total: entries.length,
    succeeded: entries.filter((entry) => entry.status === "succeeded").length,
    failed: entries.filter((entry) => entry.status === "failed").length,
    unknown: entries.filter((entry) => entry.status === "unknown").length,
    pending: entries.filter((entry) => entry.status === "queued" || entry.status === "failed").length,
  };
}

/** Only an explicit one-item retry may include an unknown operation. */
export function importBatchEntries<TFile extends ImportFile>(entries: ImportEntry<TFile>[], retryId?: string): ImportEntry<TFile>[] {
  return entries.filter((entry) => retryId
    ? entry.id === retryId && ["queued", "failed", "unknown"].includes(entry.status)
    : entry.status === "queued" || entry.status === "failed");
}
