import type { RevisionInspection } from "../../revisionWorkflow.ts";
import type { RevisionUploadOperation } from "../../revisionUploadRecovery.ts";
import type { LibraryDocument } from "./libraryModel.ts";
import type { GuidedDocument } from "../workflow/guidedReview.ts";
import { canPrepareRecheck, materialImpactPath, normalizeMaterialImpact } from "../material-impact/materialImpactModel.ts";

export type InspectionDocument = LibraryDocument & GuidedDocument;

export function inspectionFromUpload(operation: RevisionUploadOperation): RevisionInspection | null {
  const { receipt, replacement } = operation.entry;
  if (operation.entry.status !== "succeeded" || !receipt || receipt.project_id !== operation.projectId) return null;
  if (replacement && (receipt.id === replacement.id || receipt.name !== replacement.name || receipt.version !== replacement.version + 1 ||
    receipt.replaces_document_id !== replacement.id || receipt.replaces_document_version !== replacement.version)) return null;
  return { savedDocumentId: receipt.id, savedVersion: receipt.version,
    previousDocumentId: replacement?.id || null, previousVersion: replacement?.version || null };
}

export function inspectionDocuments(projectId: string, coordinates: RevisionInspection, documents: InspectionDocument[]) {
  const saved = documents.find(doc => doc.project_id === projectId && doc.id === coordinates.savedDocumentId && doc.version === coordinates.savedVersion);
  const previous = coordinates.previousDocumentId
    ? documents.find(doc => doc.project_id === projectId && doc.id === coordinates.previousDocumentId && doc.version === coordinates.previousVersion)
    : null;
  if (!saved || (coordinates.previousDocumentId && (!previous || previous.id === saved.id ||
    previous.version + 1 !== saved.version || previous.name.toLowerCase() !== saved.name.toLowerCase()))) return null;
  return { saved, previous: previous || null };
}

export class RecheckSelectionError extends Error {}

/** Every page belongs to one freshly observed selection; the POST checks it atomically. */
export async function freshRecheckSignature(projectId: string, runId: string, coordinates: RevisionInspection | undefined,
  read: (path: string) => Promise<unknown>): Promise<string> {
  let offset = 0;
  let signature: string | null = null;
  do {
    const impact = normalizeMaterialImpact(await read(materialImpactPath(runId, offset)), projectId, runId, offset);
    const current = impact.recheck.input_signature;
    if (!canPrepareRecheck(impact) || !current || !/^[a-f0-9]{64}$/.test(current)) throw new RecheckSelectionError("当前选材尚不能核对，请重新读取资料变化后再准备复检。");
    if (signature && signature !== current) throw new RecheckSelectionError("读取期间资料已有变化，请重新核对后再开始复检。");
    signature = current;
    if (!coordinates || impact.sources.items.some(source => source.selected_for_recheck === true &&
      source.current?.document_id === coordinates.savedDocumentId && source.current.document_version === coordinates.savedVersion)) return signature;
    if (!impact.sources.has_more) break;
    if (!impact.sources.items.length) throw new RecheckSelectionError("当前选材分页未能读完，请重新读取。");
    offset += impact.sources.items.length;
  } while (true);
  throw new RecheckSelectionError("本次核对的版本已不在当前复检选材中。未改用其他新版，请明确另选活动文稿并核对身份。");
}
