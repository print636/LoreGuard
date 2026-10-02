import { createIdempotencyKey } from "./api/client.ts";
import { documentContext, type DocumentRole } from "./documentContext.ts";
import { importContextFields, type ImportEntry, type ImportFile } from "./features/imports/importQueueModel.ts";

export type RevisionUploadOperation<TFile extends ImportFile = File> = {
  projectId: string;
  baselineRunId: string;
  entry: ImportEntry<TFile>;
  fields: Readonly<Record<string, string>>;
  replacementName: string | null;
  replacementVersion: number | null;
  refresh: "idle" | "reading" | "failed" | "ready";
};

export function frozenRevisionUpload<TFile extends ImportFile>(input: {
  projectId: string; baselineRunId: string; file: TFile; documentRole: DocumentRole;
  storyScope: string; replacement?: { id: string; name: string; version: number } | null;
}, key: () => string = createIdempotencyKey): RevisionUploadOperation<TFile> {
  const context = documentContext(input.documentRole, input.storyScope);
  const operationKey = key();
  const entry: ImportEntry<TFile> = {
    id: operationKey, operationKey, file: input.file, documentRole: context.document_role,
    storyScope: context.story_scope, replaceDocumentId: input.replacement?.id || "",
    status: "queued", message: "", receipt: null,
  };
  return { projectId: input.projectId, baselineRunId: input.baselineRunId, entry,
    fields: Object.freeze(importContextFields(entry)), replacementName: input.replacement?.name || null,
    replacementVersion: input.replacement?.version ?? null, refresh: "idle" };
}

export function revisionUploadMatchesInput<TFile extends ImportFile>(operation: RevisionUploadOperation<TFile>, input: {
  projectId: string; baselineRunId: string; file: TFile; documentRole: DocumentRole; storyScope: string; replaceDocumentId: string;
}): boolean {
  const context = documentContext(input.documentRole, input.storyScope);
  return operation.projectId === input.projectId && operation.baselineRunId === input.baselineRunId &&
    operation.entry.file === input.file && operation.entry.documentRole === context.document_role &&
    operation.entry.storyScope === context.story_scope && operation.entry.replaceDocumentId === input.replaceDocumentId;
}

export function revisionUploadBelongsTo<TFile extends ImportFile>(operation: RevisionUploadOperation<TFile>, projectId: string, baselineRunId: string): boolean {
  return operation.projectId === projectId && operation.baselineRunId === baselineRunId;
}

export function revisionUploadMaySend<TFile extends ImportFile>(operation: RevisionUploadOperation<TFile>, explicitRetry: boolean): boolean {
  return operation.entry.status === "queued" || operation.entry.status === "failed" ||
    (operation.entry.status === "unknown" && explicitRetry);
}
