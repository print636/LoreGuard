import test from "node:test";
import assert from "node:assert/strict";
import {
  evidenceKindLabel, evidencePreviewPath, isCurrentEvidenceRequest, normalizeEvidencePreview,
  referenceRange, sameEvidenceSelection, sameEvidenceSource,
} from "../src/features/evidence/evidenceReaderModel.ts";

function selection(overrides = {}) {
  return {
    projectId: "project-1", runId: "run-1", kind: "issue", itemId: "issue-1", evidenceIndex: 0,
    expected: { document_id: "doc-1", document_name: "第一章.md", line_start: 20, line_end: 21, text: "冻结引用\n<script>literal</script>" },
    ...overrides,
  };
}

function request(overrides = {}) {
  return { selection: selection(), offset: null, limit: 120, ...overrides };
}

function fullPreview(input = request()) {
  const reference = input.selection.expected;
  const offset = input.offset ?? Math.max(0, reference.line_start - 1 - 12);
  const total = 250;
  const lines = Array.from({ length: Math.min(input.limit, Math.max(0, total - offset)) }, (_, index) => {
    const line_number = offset + index + 1;
    return { line_number, text: line_number === 20 ? "冻结引用" : line_number === 21 ? "<script>literal</script>" : `原文 ${line_number}`, is_evidence: line_number >= reference.line_start && line_number <= reference.line_end };
  });
  return {
    run: { id: input.selection.runId, project_id: input.selection.projectId },
    reference: { ...reference, kind: input.selection.kind, item_id: input.selection.itemId, evidence_index: input.selection.evidenceIndex },
    source: { availability: "full_context", provenance: "run_input", source_run_id: input.selection.runId,
      document_version: 1, content_sha256: "a".repeat(64), char_count: 1000, line_count: total },
    lines, page: { offset, limit: input.limit, total, has_more: offset + lines.length < total }, message: null,
  };
}

function excerptPreview(input = request()) {
  const preview = fullPreview(input);
  preview.source = { ...preview.source, availability: "excerpt_only", provenance: "confirmed_trait_source", source_run_id: null, char_count: null, line_count: null };
  preview.lines = [];
  preview.page = null;
  preview.message = "完整原文快照已缺失，仅可查看已冻结的证据摘录。";
  return preview;
}

test("GET sends only opaque reference IDs, indices and pagination, and omits the first offset", () => {
  const input = request({ selection: selection({ runId: "run/one", itemId: "item&one" }) });
  const url = new URL(evidencePreviewPath(input), "https://example.test");
  assert.equal(url.pathname, "/api/v1/analysis-runs/run%2Fone/evidence-preview");
  assert.deepEqual(Array.from(url.searchParams.keys()), ["kind", "item_id", "evidence_index", "limit"]);
  assert.equal(url.searchParams.get("item_id"), "item&one");
  assert.equal(url.searchParams.get("limit"), "120");
  assert.equal(url.searchParams.has("offset"), false);
  assert.equal(url.toString().includes(encodeURIComponent(input.selection.expected.text)), false);
  assert.equal(new URL(evidencePreviewPath(request({ offset: 127 })), url).searchParams.get("offset"), "127");
});

test("out-of-range reference indices and pagination cannot become API requests", () => {
  for (const index of [-1, 12, 0.5]) assert.throws(() => evidencePreviewPath(request({ selection: selection({ evidenceIndex: index }) })), TypeError);
  for (const offset of [-1, 0.5]) assert.throws(() => evidencePreviewPath(request({ offset })), TypeError);
  for (const limit of [0, 201]) assert.throws(() => evidencePreviewPath(request({ limit })), TypeError);
});

test("stale response scope includes project, run, item class, index, evidence identity, text and page", () => {
  const selected = selection();
  assert.equal(sameEvidenceSelection(selected, structuredClone(selected)), true);
  for (const [key, value] of Object.entries({ projectId: "project-2", runId: "run-2", kind: "review_clue", itemId: "issue-2", evidenceIndex: 1, expectedVersion: 2 })) {
    assert.equal(sameEvidenceSelection({ ...selected, [key]: value }, selected), false, key);
  }
  for (const [key, value] of Object.entries({ document_id: "doc-2", document_name: "第二章.md", line_start: 19, line_end: 22, text: "当前新稿不可替换" })) {
    assert.equal(sameEvidenceSelection({ ...selected, expected: { ...selected.expected, [key]: value } }, selected), false, key);
  }
  assert.equal(isCurrentEvidenceRequest(request({ offset: 120 }), request()), false);
  assert.equal(isCurrentEvidenceRequest(request({ limit: 200 }), request()), false);
  assert.equal(isCurrentEvidenceRequest(null, request()), false);
});

test("first preview starts before the cited range and preserves original numbers, blank text and literal markup", () => {
  const value = fullPreview();
  value.lines[0].text = "";
  const preview = normalizeEvidencePreview(value, request());
  assert.equal(preview.page.offset, 7);
  assert.equal(preview.lines[0].line_number, 8);
  assert.equal(preview.lines[0].text, "");
  assert.deepEqual(preview.lines.filter((line) => line.is_evidence).map((line) => line.line_number), [20, 21]);
  assert.equal(preview.lines.find((line) => line.line_number === 21).text, "<script>literal</script>");
  assert.equal(referenceRange(preview.reference), "第 20–21 行");
  assert.equal(referenceRange({ line_start: 1, line_end: 1 }), "第 1 行");
});

test("explicit pages preserve source identity, support a partial final page and safe past-end results", () => {
  const input = request({ offset: 240 });
  const preview = normalizeEvidencePreview(fullPreview(input), input);
  assert.equal(preview.lines.length, 10);
  assert.equal(preview.lines[0].line_number, 241);
  assert.equal(preview.page.has_more, false);
  assert.equal(preview.lines.some((line) => line.is_evidence), false);
  const pastEnd = request({ offset: 360 });
  assert.deepEqual(normalizeEvidencePreview(fullPreview(pastEnd), pastEnd).lines, []);
  const beginning = request({ selection: selection({ expected: { ...selection().expected, line_start: 1, line_end: 1 } }) });
  assert.equal(normalizeEvidencePreview(fullPreview(beginning), beginning).page.offset, 0);
});

test("confirmed trait evidence may belong to a historical source run and does not adopt a current document version", () => {
  const input = request({ selection: selection({ kind: "review_clue", itemId: "clue-1" }) });
  const value = fullPreview(input);
  value.source.provenance = "confirmed_trait_source";
  value.source.source_run_id = "historical-source-run";
  assert.equal(normalizeEvidencePreview(value, input).source.source_run_id, "historical-source-run");
  assert.equal(normalizeEvidencePreview(value, input).source.document_version, 1);
  const wrongRunInput = fullPreview();
  wrongRunInput.source.source_run_id = "historical-source-run";
  assert.throws(() => normalizeEvidencePreview(wrongRunInput, request()), TypeError);
});

test("excerpt-only preserves the frozen reference and known version/hash with explicitly unavailable context", () => {
  const value = excerptPreview();
  const preview = normalizeEvidencePreview(value, request());
  assert.deepEqual(preview.reference.text, selection().expected.text);
  assert.equal(preview.source.document_version, 1);
  assert.equal(preview.source.content_sha256, "a".repeat(64));
  assert.deepEqual([preview.source.source_run_id, preview.source.char_count, preview.source.line_count, preview.page], [null, null, null, null]);
  assert.deepEqual(preview.lines, []);
  for (const mutate of [
    (value) => value.source.provenance = "run_input", (value) => value.source.source_run_id = "guessed-run",
    (value) => value.source.char_count = 0, (value) => value.source.line_count = 0,
    (value) => value.page = { offset: 0, limit: 120, total: 0, has_more: false },
    (value) => value.lines.push({ line_number: 20, text: "invented context", is_evidence: true }),
    (value) => value.message = null,
  ]) {
    const wrong = excerptPreview(); mutate(wrong);
    assert.throws(() => normalizeEvidencePreview(wrong, request()), TypeError);
  }
});

test("provisional references retain the unverified label and known frozen version requirement", () => {
  const input = request({ selection: selection({ kind: "provisional_clue", itemId: "pc_1", expectedVersion: 1 }) });
  assert.match(evidenceKindLabel(input.selection.kind), /未验证/);
  assert.match(evidenceKindLabel("review_clue"), /待复核/);
  assert.equal(normalizeEvidencePreview(fullPreview(input), input).reference.kind, "provisional_clue");
  const wrong = fullPreview(input); wrong.source.document_version = 2;
  assert.throws(() => normalizeEvidencePreview(wrong, input), TypeError);
});

test("every reference identity, frozen binding and pagination mismatch is rejected before source display", () => {
  const mutations = [
    (value) => value.run.id = "wrong", (value) => value.run.project_id = "wrong",
    (value) => value.reference.kind = "review_clue", (value) => value.reference.item_id = "wrong",
    (value) => value.reference.evidence_index = 1, (value) => value.reference.document_id = "wrong",
    (value) => value.reference.document_name = "wrong", (value) => value.reference.line_start = 19,
    (value) => value.reference.line_end = 22, (value) => value.reference.text = "current draft",
    (value) => value.source.availability = "live_document", (value) => value.source.provenance = "current_document",
    (value) => value.source.source_run_id = null, (value) => value.source.document_version = 0,
    (value) => value.source.content_sha256 = "invalid", (value) => value.source.char_count = null,
    (value) => value.source.line_count = 20, (value) => value.page.offset = 8,
    (value) => value.page.limit = 200, (value) => value.page.total = 249,
    (value) => value.page.has_more = false, (value) => value.lines.pop(),
    (value) => value.lines[0].line_number = 9, (value) => value.lines[0].text = {},
    (value) => value.lines[12].is_evidence = false, (value) => value.lines[0].is_evidence = true,
    (value) => value.message = "unexpected full-context warning",
  ];
  for (const mutate of mutations) {
    const value = fullPreview(); mutate(value);
    assert.throws(() => normalizeEvidencePreview(value, request()), TypeError);
  }
});

test("a frozen source cannot change between context pages", () => {
  const source = fullPreview().source;
  assert.equal(sameEvidenceSource(source, { ...source }), true);
  for (const [key, value] of Object.entries({ availability: "excerpt_only", provenance: "confirmed_trait_source", source_run_id: "old-run", document_version: 2, content_sha256: "b".repeat(64), char_count: 2000, line_count: 251 })) {
    assert.equal(sameEvidenceSource(source, { ...source, [key]: value }), false, key);
  }
});
