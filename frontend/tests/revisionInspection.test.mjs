import test from "node:test";
import assert from "node:assert/strict";
import { revisionRouteStateFromSearch, revisionSearch } from "../src/revisionWorkflow.ts";
import { frozenRevisionUpload } from "../src/revisionUploadRecovery.ts";
import { freshRecheckSignature, inspectionFromUpload, inspectionDocuments } from "../src/features/documents/revisionInspectionModel.ts";
import { selectedContextDocument } from "../src/features/workflow/guidedReview.ts";
import { documentDiffPath, documentDiffMatches, emptyDocumentDiffMessage } from "../src/features/documents/documentDiff.ts";

const old = { id: "v3", project_id: "project", name: "原稿.docx", version: 3, active: false,
  document_role: "chapter", story_scope: "global", created_at: "2026-11-11" };
const saved = { ...old, id: "v4", version: 4, active: true };
const coordinates = { savedDocumentId: "v4", savedVersion: 4, previousDocumentId: "v3", previousVersion: 3 };

test("inspection route persists exact versions but does not manufacture an upload receipt", () => {
  const route = { ...revisionRouteStateFromSearch("?step=inspect&issue=issue-one"), inspection: coordinates };
  assert.deepEqual(revisionRouteStateFromSearch(revisionSearch(route)), route);
  assert.equal("receipt" in route, false);
  for (const query of ["saved=v4", "saved=v4&saved_version=4&previous=v4&previous_version=3",
    "saved=v4&saved_version=4&previous=v1&previous_version=1", "saved=../secret&saved_version=4"]) {
    assert.equal(revisionRouteStateFromSearch(`?step=inspect&${query}`).inspection, undefined);
  }
});

test("only a succeeded verified receipt yields saved coordinates; target is independent of baseline v1", () => {
  const operation = frozenRevisionUpload({ projectId: "project", baselineRunId: "baseline-v1",
    file: { name: "新版.md", size: 10, lastModified: 1 }, documentRole: "chapter", storyScope: "global", replacement: old }, () => "key");
  const complete = { ...operation, entry: { ...operation.entry, status: "succeeded", receipt: {
    id: saved.id, name: saved.name, version: saved.version, project_id: "project",
    replaces_document_id: old.id, replaces_document_version: old.version } } };
  assert.deepEqual(inspectionFromUpload(complete), coordinates);
  for (const status of ["unknown", "uploading", "failed"]) {
    assert.equal(inspectionFromUpload({ ...complete, entry: { ...complete.entry, status } }), null);
  }
  for (const patch of [{ project_id: "other" }, { replaces_document_id: "v1" }, { version: 5 }, { name: "其他稿.md" }]) {
    assert.equal(inspectionFromUpload({ ...complete, entry: { ...complete.entry, receipt: { ...complete.entry.receipt, ...patch } } }), null);
  }
});

test("later active v5 never substitutes for saved v4 or the explicit previous v3", () => {
  const later = { ...saved, id: "v5", version: 5 };
  const retired = { ...saved, active: false };
  const pair = inspectionDocuments("project", coordinates, [later, retired, old]);
  assert.equal(pair.saved, retired);
  assert.equal(pair.previous, old);
  assert.equal(inspectionDocuments("project", coordinates, [later, old]), null);
  assert.equal(inspectionDocuments("other-project", coordinates, [saved, old]), null);
  assert.equal(inspectionDocuments("project", coordinates, [saved, { ...old, name: "other.md" }]), null);
});

test("fixed identity editor preserves its historical target and never selects another active document", () => {
  const fixed = { id: saved.id, version: saved.version };
  const later = { ...saved, id: "v5", version: 5 };
  const retired = { ...saved, active: false };
  assert.equal(selectedContextDocument([later, retired], "v5", null, fixed), retired);
  assert.equal(selectedContextDocument([later], "v5", retired, fixed), retired);
  assert.equal(selectedContextDocument([later], "v5", null, fixed), null);
  assert.equal(selectedContextDocument([later], "v5", retired, { ...fixed, version: 9 }), null);
  assert.equal(selectedContextDocument([later], "", null), later);
  assert.equal(selectedContextDocument([later], retired.id, retired), retired);
});

test("diff response must match both exact IDs, versions, names and project", () => {
  const result = { from_document: old, to_document: saved };
  assert.equal(documentDiffMatches(result, "project", old, saved), true);
  for (const patch of [{ id: "v5" }, { version: 5 }, { project_id: "other" }, { name: "other.md" }]) {
    assert.equal(documentDiffMatches({ ...result, to_document: { ...saved, ...patch } }, "project", old, saved), false);
    assert.equal(documentDiffMatches({ ...result, from_document: { ...old, ...patch } }, "project", old, saved), false);
  }
  assert.equal(documentDiffPath("p x", "old/id", "new/id"), "/api/v1/projects/p%20x/documents/diff?from_document_id=old%2Fid&to_document_id=new%2Fid");
});

test("empty hunks distinguish complete, input truncation and omitted output", () => {
  const summary = { added_lines: 0, removed_lines: 0, changed_hunks: 0, compared_old_lines: 20, compared_new_lines: 20,
    old_total_lines: 20, new_total_lines: 20, input_truncated: false, output_truncated: false };
  assert.match(emptyDocumentDiffMessage(summary), /^逐行正文未见差异/);
  assert.match(emptyDocumentDiffMessage({ ...summary, input_truncated: true }), /未覆盖的正文不能确认/);
  assert.match(emptyDocumentDiffMessage({ ...summary, old_total_lines: 21 }), /未覆盖的正文不能确认/);
  assert.match(emptyDocumentDiffMessage({ ...summary, output_truncated: true }), /差异输出不完整/);
  assert.match(emptyDocumentDiffMessage({ ...summary, changed_hunks: 2 }), /不能据此认定/);
});

const source = (id = saved.id, version = saved.version, selected = true) => ({ key: id, kind: "input", frozen: null,
  current: { document_id: id, document_name: saved.name, document_version: version, document_role: "chapter",
    story_scope: "global", batch_role: "target", publication_status: "draft" },
  changes: ["version_changed"], selected_for_recheck: selected, content_changed: true, identity_comparison_available: true });
const impact = (items, offset = 0, total = items.length, signature = "a".repeat(64)) => ({
  schema_version: "material-change-impact-v1", project_id: "project", run_id: "baseline-v1", comparison_status: "complete",
  semantic_impact_complete: false, confirmed_traits_auto_invalidated: false, confirmed_traits_changed: false,
  recheck: { inputs_changed: true, can_prepare: true, reason: null, input_signature: signature },
  sources: { items, offset, limit: 30, total, has_more: offset + items.length < total },
  affected_issues: { items: [], offset: 0, limit: 30, total: 0, has_more: false },
  summary: { changed_sources: total, added_sources: 0, not_included_sources: 0, affected_issues: 0, unresolved_evidence: 0, unresolved_sources: 0 },
});

test("fresh recheck selection finds exact saved version on later source page with the same signature", async () => {
  const reads = [];
  const signature = await freshRecheckSignature("project", "baseline-v1", coordinates, async path => {
    reads.push(path);
    return reads.length === 1 ? impact([source("background", 1)], 0, 2) : impact([source()], 1, 2);
  });
  assert.equal(signature, "a".repeat(64));
  assert.equal(reads.length, 2);
  assert.match(reads[1], /offset=1&limit=30/);
});

test("preflight fails closed for newer, unselected, unsigned or changed-between-pages inputs", async () => {
  for (const response of [impact([source("v5", 5)]), impact([source(saved.id, saved.version, false)]),
    impact([source()], 0, 1, null), impact([source()], 0, 1, "invalid")]) {
    await assert.rejects(freshRecheckSignature("project", "baseline-v1", coordinates, async () => response));
  }
  let reads = 0;
  await assert.rejects(freshRecheckSignature("project", "baseline-v1", coordinates, async () =>
    ++reads === 1 ? impact([source("other", 1)], 0, 2) : impact([source()], 1, 2, "b".repeat(64))), /资料已有变化/);
});
