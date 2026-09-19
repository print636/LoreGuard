import type { DocumentRole } from "../documentContext.ts";

export type ImportFilePlan<TFile = File> = {
  file: TFile;
  documentRole: DocumentRole;
};

export function createImportFilePlan<TFile>(
  files: Iterable<TFile>,
): ImportFilePlan<TFile>[] {
  return Array.from(files, (file) => ({
    file,
    documentRole: "chapter" as const,
  }));
}

export function updateImportFileRole<TFile>(
  plan: ImportFilePlan<TFile>[],
  index: number,
  documentRole: DocumentRole,
): ImportFilePlan<TFile>[] {
  return plan.map((entry, entryIndex) =>
    entryIndex === index ? { ...entry, documentRole } : entry,
  );
}
