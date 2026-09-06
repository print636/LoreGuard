export type RunInputSnapshot = {
  document_id: string;
  document_name: string;
  document_version: number;
  document_role: string;
  story_scope: string;
  content_sha256: string;
  char_count?: number;
  ordinal?: number;
};

export type SnapshotRun = {
  id: string;
  input_documents?: RunInputSnapshot[];
  input_snapshot_available?: boolean;
  retried_from?: string | null;
};

const roleNames: Record<string,string> = {
  canon: '权威世界观',
  chapter: '故事正文',
  reference: '参考资料',
  story_branch: '故事分支',
};

export function shortIdentifier(value: string, visible=12): string {
  if (!value) return 'unknown';
  return value.length > visible ? `${value.slice(0,visible)}…` : value;
}

export function runSnapshotDocuments(run: SnapshotRun): RunInputSnapshot[] {
  if (run.input_snapshot_available !== true || !Array.isArray(run.input_documents)) return [];
  return run.input_documents;
}

export function hasRunSnapshot(run: SnapshotRun): boolean {
  return runSnapshotDocuments(run).length > 0;
}

export function runInputState(run: SnapshotRun): string {
  const documents=runSnapshotDocuments(run);
  return documents.length ? `${documents.length} 个冻结输入` : '输入未知';
}

export function retryState(run: SnapshotRun): {allowed:boolean; label:string} {
  return hasRunSnapshot(run)
    ? {allowed:true,label:'按本次输入重试（可能消耗 Token）'}
    : {allowed:false,label:'不可同输入重试'};
}

export function snapshotDocumentLabels(document: RunInputSnapshot): {
  identity:string; context:string; hash:string;
} {
  return {
    identity:`${document.document_name} · v${document.document_version}`,
    context:`role ${roleNames[document.document_role]||document.document_role} (${document.document_role}) · scope ${document.story_scope}`,
    hash:shortIdentifier(document.content_sha256),
  };
}

export function retryLineage(run: SnapshotRun): string | null {
  return run.retried_from ? `重试继承自 ${shortIdentifier(run.retried_from)}` : null;
}
