import type { DocumentRole } from "../../documentContext.ts";

export const LIBRARY_PAGE_SIZE = 20;
export const PREVIEW_PAGE_SIZE = 120;
export const PREVIEW_QUERY_LIMIT = 160;

export type LibraryDocument = {
  id: string;
  project_id: string;
  name: string;
  version: number;
  active: boolean;
  created_at: string;
  document_role: DocumentRole;
  story_scope: string;
  narrative_context?: { publication_status: string };
};

export type VersionFilter = "all" | "current" | "history";
export type LibraryState = {
  projectId: string;
  filenameQuery: string;
  appliedFilenameQuery: string;
  roleFilter: DocumentRole | "all";
  versionFilter: VersionFilter;
  page: number;
  selectedGroupKey: string;
  selectedDocumentId: string;
  offset: number;
  queryDraft: string;
  appliedQuery: string;
};

export function initialLibraryState(projectId: string): LibraryState {
  return {
    projectId, filenameQuery: "", appliedFilenameQuery: "", roleFilter: "all",
    versionFilter: "all", page: 1, selectedGroupKey: "", selectedDocumentId: "",
    offset: 0, queryDraft: "", appliedQuery: "",
  };
}

/** Match the existing import/version name grouping; filenames are never trimmed. */
export function documentNameKey(name: string): string {
  return name.toLowerCase();
}

export type DocumentGroup = {
  key: string;
  name: string;
  versions: LibraryDocument[];
  current: LibraryDocument | null;
};

export function groupDocuments(documents: LibraryDocument[], projectId: string): DocumentGroup[] {
  const grouped = new Map<string, LibraryDocument[]>();
  for (const document of documents) {
    if (document.project_id !== projectId) continue;
    const key = documentNameKey(document.name);
    const group = grouped.get(key) || [];
    group.push(document);
    grouped.set(key, group);
  }
  return Array.from(grouped, ([key, versions]) => {
    versions.sort((a, b) => b.version - a.version || b.created_at.localeCompare(a.created_at) || a.id.localeCompare(b.id));
    const current = versions.find((document) => document.active) || null;
    return { key, name: (current || versions[0]).name, versions, current };
  }).sort((a, b) => a.name.localeCompare(b.name, "zh-CN", { numeric: true }) || a.key.localeCompare(b.key));
}

export function matchingVersions(group: DocumentGroup, role: LibraryState["roleFilter"], version: VersionFilter): LibraryDocument[] {
  return group.versions.filter((document) =>
    (role === "all" || document.document_role === role) &&
    (version === "all" || (version === "current" ? document.active : !document.active)),
  );
}

export function filterDocumentGroups(groups: DocumentGroup[], state: Pick<LibraryState, "appliedFilenameQuery" | "roleFilter" | "versionFilter">): DocumentGroup[] {
  const query = documentNameKey(state.appliedFilenameQuery.trim());
  return groups.filter((group) =>
    (!query || group.key.includes(query)) &&
    matchingVersions(group, state.roleFilter, state.versionFilter).length > 0,
  );
}

export function preferredGroupVersion(group: DocumentGroup, state: Pick<LibraryState, "roleFilter" | "versionFilter">): LibraryDocument | null {
  const versions = matchingVersions(group, state.roleFilter, state.versionFilter);
  return versions.find((document) => document.active) || versions[0] || null;
}

export function libraryPage(groups: DocumentGroup[], requestedPage: number) {
  const pages = Math.max(1, Math.ceil(groups.length / LIBRARY_PAGE_SIZE));
  const page = Math.min(pages, Math.max(1, Number.isSafeInteger(requestedPage) ? requestedPage : 1));
  const start = (page - 1) * LIBRARY_PAGE_SIZE;
  return { page, pages, total: groups.length, start, items: groups.slice(start, start + LIBRARY_PAGE_SIZE) };
}

export function publicationLabel(document: LibraryDocument): string {
  const names: Record<string, string> = {
    unknown: "待确认", draft: "草稿", in_review: "审阅中", published: "已发布", retired: "已归档",
  };
  return names[document.narrative_context?.publication_status || "unknown"] || "待确认";
}

export type PreviewRequest = {
  projectId: string;
  documentId: string;
  version: number;
  name: string;
  offset: number;
  limit: number;
  query: string;
};

export type DocumentPreview = {
  document: LibraryDocument & { char_count: number; line_count: number; content_sha256: string };
  lines: Array<{ line_number: number; text: string }>;
  page: { offset: number; limit: number; total: number; has_more: boolean };
  query: string;
};

export function isCurrentPreviewRequest(received: PreviewRequest | null, selected: PreviewRequest | null): boolean {
  return Boolean(received && selected &&
    received.projectId === selected.projectId && received.documentId === selected.documentId &&
    received.version === selected.version && received.name === selected.name &&
    received.offset === selected.offset && received.limit === selected.limit && received.query === selected.query);
}

export function documentPreviewPath(request: PreviewRequest): string {
  const params = new URLSearchParams({ offset: String(request.offset), limit: String(request.limit) });
  if (request.query) params.set("query", request.query);
  return `/api/v1/projects/${encodeURIComponent(request.projectId)}/documents/${encodeURIComponent(request.documentId)}/preview?${params}`;
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function nonNegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0;
}

/** Never display a response for another document, version, search or page. */
export function normalizeDocumentPreview(value: unknown, request: PreviewRequest): DocumentPreview {
  const source = record(value);
  const document = record(source?.document);
  const page = record(source?.page);
  const context = record(document?.narrative_context);
  const roles = ["chapter", "canon", "character_profile", "reference"];
  const publications = ["unknown", "draft", "in_review", "published", "retired"];
  if (!source || !document || !page ||
    document.id !== request.documentId || document.project_id !== request.projectId ||
    document.version !== request.version || document.name !== request.name ||
    typeof document.active !== "boolean" || typeof document.created_at !== "string" ||
    typeof document.document_role !== "string" || !roles.includes(document.document_role) ||
    typeof document.story_scope !== "string" ||
    !nonNegativeInteger(document.char_count) || !nonNegativeInteger(document.line_count) ||
    typeof document.content_sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(document.content_sha256) ||
    (document.narrative_context != null && (!context || typeof context.publication_status !== "string" || !publications.includes(context.publication_status))) ||
    source.query !== request.query || !Array.isArray(source.lines) ||
    page.offset !== request.offset || page.limit !== request.limit ||
    !nonNegativeInteger(page.total) || page.total > document.line_count ||
    typeof page.has_more !== "boolean" ||
    (!request.query && page.total !== document.line_count)) {
    throw new TypeError("返回的文稿身份或分页信息不一致，请刷新资料库后重试。");
  }
  const expectedCount = Math.min(request.limit, Math.max(0, page.total - request.offset));
  if (source.lines.length !== expectedCount || page.has_more !== (request.offset + expectedCount < page.total)) {
    throw new TypeError("返回的原文分页不完整，请重试。");
  }
  let previousLine = 0;
  const lineCount = document.line_count;
  const lines = source.lines.map((value, index) => {
    const line = record(value);
    if (!line || !nonNegativeInteger(line.line_number) || line.line_number < 1 ||
      line.line_number <= previousLine || line.line_number > lineCount ||
      typeof line.text !== "string" ||
      (!request.query && line.line_number !== request.offset + index + 1)) {
      throw new TypeError("返回的原文行号不一致，请重试。");
    }
    previousLine = line.line_number;
    return { line_number: line.line_number, text: line.text };
  });
  return {
    // Keep only reader metadata; never retain a server-provided full content field.
    document: {
      id: request.documentId, project_id: request.projectId, name: request.name, version: request.version,
      active: document.active, created_at: document.created_at, document_role: document.document_role as DocumentRole,
      story_scope: document.story_scope, char_count: document.char_count, line_count: document.line_count,
      content_sha256: document.content_sha256,
      ...(context ? { narrative_context: { publication_status: context.publication_status as string } } : {}),
    },
    lines, page: { offset: request.offset, limit: request.limit, total: page.total, has_more: page.has_more },
    query: request.query,
  };
}

export type HighlightSegment = { text: string; match: boolean };

function foldCharacter(character: string): string {
  const expansions: Record<string, string> = { "ß": "ss", "ς": "σ", "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st" };
  const lower = character.toLowerCase();
  return expansions[lower] || lower;
}

/** Produce text nodes and marks, including Unicode expansions without offset drift. */
export function literalHighlightSegments(text: string, query: string): HighlightSegment[] {
  if (!query) return [{ text, match: false }];
  let folded = "";
  const starts: number[] = [];
  const ends: number[] = [];
  let originalOffset = 0;
  for (const character of text) {
    const part = foldCharacter(character);
    folded += part;
    for (let index = 0; index < part.length; index++) {
      starts.push(originalOffset);
      ends.push(originalOffset + character.length);
    }
    originalOffset += character.length;
  }
  const needle = Array.from(query, foldCharacter).join("");
  if (!needle) return [{ text, match: false }];
  const ranges: Array<[number, number]> = [];
  let searchOffset = 0;
  while (searchOffset < folded.length) {
    const found = folded.indexOf(needle, searchOffset);
    if (found < 0) break;
    const start = starts[found];
    const end = ends[found + needle.length - 1];
    const previous = ranges[ranges.length - 1];
    if (previous && previous[1] >= start) previous[1] = Math.max(previous[1], end);
    else ranges.push([start, end]);
    searchOffset = found + needle.length;
  }
  const segments: HighlightSegment[] = [];
  let cursor = 0;
  for (const [start, end] of ranges) {
    if (start > cursor) segments.push({ text: text.slice(cursor, start), match: false });
    segments.push({ text: text.slice(start, end), match: true });
    cursor = end;
  }
  if (cursor < text.length || !segments.length) segments.push({ text: text.slice(cursor), match: false });
  return segments;
}
