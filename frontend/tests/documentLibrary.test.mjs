import test from "node:test";
import assert from "node:assert/strict";
import {
  documentNameKey, documentPreviewPath, filterDocumentGroups, groupDocuments,
  initialLibraryState, isCurrentPreviewRequest, libraryPage, literalHighlightSegments,
  matchingVersions, normalizeDocumentPreview, preferredGroupVersion, publicationLabel,
} from "../src/features/documents/libraryModel.ts";

function document(overrides = {}) {
  return {
    id: "doc-current", project_id: "project-1", name: "chapter.md", version: 2,
    active: true, document_role: "chapter", story_scope: "global",
    created_at: "2026-10-01T08:00:00Z", narrative_context: { publication_status: "draft" },
    ...overrides,
  };
}

function request(overrides = {}) {
  return { projectId: "project-1", documentId: "doc-current", name: "chapter.md", version: 2, offset: 0, limit: 120, query: "", ...overrides };
}

function preview(overrides = {}) {
  return {
    document: { ...document(), char_count: 16, line_count: 3, content_sha256: "b".repeat(64) },
    lines: [{ line_number: 1, text: "首行" }, { line_number: 2, text: "" }, { line_number: 3, text: "<script>literal</script>" }],
    page: { offset: 0, limit: 120, total: 3, has_more: false }, query: "", ...overrides,
  };
}

test("same-name versions group without crossing project boundaries or merging distinct filenames", () => {
  const groups = groupDocuments([
    document({ id: "history", name: "CHAPTER.md", version: 1, active: false }),
    document(), document({ id: "foreign", project_id: "project-2", version: 7 }),
    document({ id: "spaced", name: " chapter.md" }),
    document({ id: "chinese-1", name: "第一章.md", version: 1, active: false }),
    document({ id: "chinese-2", name: "第一章.md" }),
  ], "project-1");
  assert.equal(groups.length, 3);
  const chapter = groups.find((group) => group.key === "chapter.md");
  assert.deepEqual(chapter.versions.map((item) => item.id), ["doc-current", "history"]);
  assert.equal(chapter.current.id, "doc-current");
  assert.equal(documentNameKey("第一章.MD"), "第一章.md");
  assert.equal(groups.find((group) => group.key === "第一章.md").versions.length, 2);
});

test("filters apply to actual matching versions and select a historical version when requested", () => {
  const groups = groupDocuments([
    document(), document({ id: "history", version: 1, active: false, document_role: "reference" }),
    document({ id: "world", name: "世界观.md", document_role: "canon" }),
  ], "project-1");
  const filters = { ...initialLibraryState("project-1"), appliedFilenameQuery: " CHAPTER ", versionFilter: "history", roleFilter: "reference" };
  const filtered = filterDocumentGroups(groups, filters);
  assert.equal(filtered.length, 1);
  assert.equal(preferredGroupVersion(filtered[0], filters).id, "history");
  assert.equal(preferredGroupVersion(filtered[0], initialLibraryState("project-1")).id, "doc-current");
  assert.deepEqual(matchingVersions(filtered[0], "canon", "history"), []);
  assert.equal(filterDocumentGroups(groups, { ...filters, roleFilter: "chapter" }).length, 0);
});

test("logical manuscripts paginate at 20, clamp empty and obsolete pages, and keep every manuscript reachable", () => {
  const groups = groupDocuments(Array.from({ length: 43 }, (_, index) => document({ id: `doc-${index}`, name: `章节${index + 1}.md` })), "project-1");
  const first = libraryPage(groups, 1);
  const second = libraryPage(groups, 2);
  const last = libraryPage(groups, 9);
  assert.deepEqual([first.items.length, second.items.length, last.items.length], [20, 20, 3]);
  assert.deepEqual([last.page, last.pages, last.start, last.total], [3, 3, 40, 43]);
  assert.equal(new Set([...first.items, ...second.items, ...last.items].map((group) => group.key)).size, 43);
  assert.deepEqual(libraryPage([], 5), { page: 1, pages: 1, total: 0, start: 0, items: [] });
});

test("reader request matching rejects stale project, document, version, name, pagination and searches", () => {
  const current = request();
  assert.equal(isCurrentPreviewRequest(current, { ...current }), true);
  for (const [key, value] of Object.entries({ projectId: "project-2", documentId: "old", version: 1, name: "other.md", offset: 120, limit: 100, query: "secret" })) {
    assert.equal(isCurrentPreviewRequest({ ...current, [key]: value }, current), false, key);
  }
  assert.equal(isCurrentPreviewRequest(null, current), false);
});

test("preview path encodes IDs and literal search only in the API request", () => {
  const path = documentPreviewPath(request({ projectId: "project/one", documentId: "doc?one", query: "林澈 & .*[" }));
  const url = new URL(path, "https://example.test");
  assert.equal(url.pathname, "/api/v1/projects/project%2Fone/documents/doc%3Fone/preview");
  assert.equal(url.searchParams.get("query"), "林澈 & .*[");
  assert.equal(url.searchParams.get("limit"), "120");
  assert.equal(new URL(documentPreviewPath(request()), url).searchParams.has("query"), false);
});

test("normalization preserves literal source, blank lines and historical metadata without retaining full content", () => {
  const value = preview();
  value.document.active = false;
  value.document.content = "must not be retained";
  const result = normalizeDocumentPreview(value, request());
  assert.equal(result.document.active, false);
  assert.equal("content" in result.document, false);
  assert.deepEqual(result.lines, value.lines);
  assert.equal(publicationLabel(result.document), "草稿");
});

test("filtered pages preserve sparse original line numbers and accept empty documents and past-end offsets", () => {
  const sparse = preview({ query: "林澈", lines: [{ line_number: 2, text: "林澈" }, { line_number: 120, text: "林澈" }] });
  sparse.document.line_count = 150;
  sparse.page.total = 2;
  assert.deepEqual(normalizeDocumentPreview(sparse, request({ query: "林澈" })).lines.map((line) => line.line_number), [2, 120]);
  const empty = preview({ lines: [], page: { offset: 0, limit: 120, total: 0, has_more: false } });
  empty.document.line_count = 0;
  empty.document.char_count = 0;
  assert.deepEqual(normalizeDocumentPreview(empty, request()).lines, []);
  const pastEnd = preview({ lines: [], page: { offset: 120, limit: 120, total: 3, has_more: false } });
  assert.deepEqual(normalizeDocumentPreview(pastEnd, request({ offset: 120 })).lines, []);
});

test("normalization rejects wrong identity, pagination, duplicate/out-of-order line numbers and malformed source data", () => {
  const mutations = [
    (value) => value.document.id = "wrong", (value) => value.document.project_id = "wrong",
    (value) => value.document.version = 9, (value) => value.document.name = "wrong.md",
    (value) => value.document.content_sha256 = "invalid", (value) => value.document.line_count = -1,
    (value) => value.document.char_count = 1.5, (value) => value.page.offset = 120,
    (value) => value.page.limit = 20, (value) => value.page.total = 4,
    (value) => value.page.has_more = true, (value) => value.query = "old search",
    (value) => value.lines.pop(), (value) => value.lines[1].line_number = 1,
    (value) => value.lines[1].line_number = 3, (value) => value.lines[0].text = {},
    (value) => value.document.narrative_context = { publication_status: "fake" },
  ];
  for (const mutate of mutations) {
    const value = preview();
    mutate(value);
    assert.throws(() => normalizeDocumentPreview(value, request()), TypeError);
  }
});

test("search highlighting uses literal text segments and correct original offsets for case expansion and emoji", () => {
  assert.deepEqual(literalHighlightSegments("甲 .* 乙 .*", ".*"), [
    { text: "甲 ", match: false }, { text: ".*", match: true },
    { text: " 乙 ", match: false }, { text: ".*", match: true },
  ]);
  assert.deepEqual(literalHighlightSegments("🌙 Straße <img>", "STRASSE"), [
    { text: "🌙 ", match: false }, { text: "Straße", match: true }, { text: " <img>", match: false },
  ]);
  assert.deepEqual(literalHighlightSegments("林澈说：<script>", "<script>"), [
    { text: "林澈说：", match: false }, { text: "<script>", match: true },
  ]);
  assert.deepEqual(literalHighlightSegments("保留原文", "不存在"), [{ text: "保留原文", match: false }]);
  assert.deepEqual(literalHighlightSegments("", ""), [{ text: "", match: false }]);
});
