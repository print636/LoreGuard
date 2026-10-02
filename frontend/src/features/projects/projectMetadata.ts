export type ProjectMetadata = { id: string; name: string; description: string; created_at: string; metadata_revision: number };
export type MetadataDraft = { name: string; description: string };

export function verifiedProjectMetadata(value: unknown, projectId: string): ProjectMetadata | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const row = value as Record<string, unknown>;
  if (row.id !== projectId || typeof row.name !== "string" || !row.name.trim() || row.name.length > 200 || typeof row.description !== "string" || typeof row.created_at !== "string" || !Number.isSafeInteger(row.metadata_revision) || Number(row.metadata_revision) < 1) return null;
  return { id: projectId, name: row.name, description: row.description, created_at: row.created_at, metadata_revision: Number(row.metadata_revision) };
}

export function metadataDraftDirty(draft: MetadataDraft, original: MetadataDraft | null): boolean {
  return original !== null && (draft.name !== original.name || draft.description !== original.description);
}

export function metadataNameError(name: string): string {
  if (!name.trim()) return "请输入项目名称。";
  if (name.trim().length > 200) return "项目名称最多 200 个字符。";
  return "";
}

export function metadataPatch(draft: MetadataDraft, base: ProjectMetadata) {
  const error = metadataNameError(draft.name);
  if (error) throw new Error(error);
  return { name: draft.name.trim(), description: draft.description, expected_revision: base.metadata_revision };
}

/** Rebase only by explicit author action: untouched fields follow the latest server version. */
export function rebaseMetadataDraft(draft: MetadataDraft, base: MetadataDraft, latest: MetadataDraft): MetadataDraft {
  return { name: draft.name === base.name ? latest.name : draft.name, description: draft.description === base.description ? latest.description : draft.description };
}

export function verifiedCreatedProjectId(value: unknown): string | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const id = (value as Record<string, unknown>).id;
  return typeof id === "string" && /^[A-Za-z0-9._:-]{1,128}$/.test(id) ? id : null;
}
