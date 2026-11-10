import type { LibraryDocument } from "./libraryModel.ts";

export type DocumentDiff = {
  from_document: LibraryDocument;
  to_document: LibraryDocument;
  summary: {
    added_lines: number; removed_lines: number; unchanged_lines: number; changed_hunks: number;
    compared_old_lines: number; compared_new_lines: number; old_total_lines: number; new_total_lines: number;
    input_truncated: boolean; output_truncated: boolean;
  };
  hunks: Array<{ old_start: number; old_lines: number; new_start: number; new_lines: number;
    lines: Array<{ type: "added" | "removed" | "unchanged"; content: string; old_line: number | null; new_line: number | null }> }>;
  warnings: string[];
};

export function documentDiffPath(projectId: string, fromId: string, toId: string): string {
  return `/api/v1/projects/${encodeURIComponent(projectId)}/documents/diff?${new URLSearchParams({ from_document_id: fromId, to_document_id: toId })}`;
}

export function documentDiffMatches(result: DocumentDiff, projectId: string, from: LibraryDocument, to: LibraryDocument): boolean {
  return [result.from_document, result.to_document].every((actual, index) => {
    const expected = index === 0 ? from : to;
    return actual?.project_id === projectId && actual.id === expected.id && actual.version === expected.version && actual.name === expected.name;
  });
}

/** No hunks is not proof that unread input (or omitted output) is identical. */
export function emptyDocumentDiffMessage(summary: DocumentDiff["summary"]): string {
  if (summary.output_truncated || summary.changed_hunks > 0 || summary.added_lines > 0 || summary.removed_lines > 0) {
    return "本页没有可展示的变更区块；差异输出不完整，不能据此认定两版正文相同。";
  }
  if (summary.input_truncated || summary.compared_old_lines < summary.old_total_lines || summary.compared_new_lines < summary.new_total_lines) {
    return "已比较范围内未见差异；未覆盖的正文不能确认。";
  }
  return "逐行正文未见差异；这不比较 Word 排版、批注或文件字节。";
}
