import test from "node:test";
import assert from "node:assert/strict";
import { ApiError } from "../src/api/client.ts";
import {
  appendImportEntries, emptyImportDefaults, importBatchEntries, importContextFields,
  frozenImportReplacement, importCounts, importFailure, updateImportEntry, verifiedImportReceipt,
} from "../src/features/imports/importQueueModel.ts";

const files = [
  { name: "设定.docx", size: 140, lastModified: 10 },
  { name: "chapter.json", size: 230, lastModified: 20 },
];
function seeded() { let sequence = 0; return appendImportEntries([], files, emptyImportDefaults, () => `operation-${++sequence}`).entries; }

test("queued file defaults do not override the server's existing document role, scope or JSON bytes", () => {
  const entries = seeded();
  assert.equal(entries[0].documentRole, null);
  assert.equal(entries[1].file, files[1]);
  assert.deepEqual(importContextFields(entries[1]), {});
  assert.deepEqual(importContextFields({ documentRole: "reference", storyScope: "路线甲", replaceDocumentId: "old-document" }), { document_role: "reference", story_scope: "路线甲", replace_document_id: "old-document" });
});

test("appending files retains queue receipts, failures and per-file settings; duplicates are not appended", () => {
  const entries = seeded();
  const initial = [updateImportEntry(entries[0], { documentRole: "canon" }, () => "canon-operation"), { ...entries[1], status: "failed", message: "UTF-8" }];
  const next = appendImportEntries(initial, [{ ...files[0] }, { name: "第三章.txt", size: 10, lastModified: 30 }], emptyImportDefaults, () => "new-operation");
  assert.equal(next.duplicates, 1);
  assert.equal(next.entries.length, 3);
  assert.equal(next.entries[0].documentRole, "canon");
  assert.equal(next.entries[1].status, "failed");
  assert.equal(initial.length, 2);
});

test("changing a failed file's payload generates a new key without changing row identity", () => {
  const initial = { ...seeded()[0], status: "failed" };
  const next = updateImportEntry(initial, { documentRole: "character_profile", storyScope: "route_a" }, () => "new-payload-key");
  assert.equal(next.id, initial.id);
  assert.equal(next.operationKey, "new-payload-key");
  assert.equal(next.status, "queued");
  assert.equal(updateImportEntry(next, { storyScope: "route_a" }, () => { throw new Error("unchanged payload must retain key"); }), next);
});

test("successful, in-flight and ambiguous entries cannot silently change their payload or stable key", () => {
  for (const status of ["succeeded", "uploading", "unknown"]) {
    const initial = { ...seeded()[0], status };
    assert.equal(updateImportEntry(initial, { storyScope: "other" }, () => "wrong-new-key"), initial);
  }
});

test("bulk continuation excludes successful, unknown and in-flight operations; explicit unknown retry retains same key", () => {
  const template = seeded()[0];
  const entries = ["queued", "failed", "succeeded", "unknown", "uploading"].map((status) => ({ ...template, id: status, operationKey: `${status}-key`, status }));
  assert.deepEqual(importBatchEntries(entries).map((entry) => entry.id), ["queued", "failed"]);
  assert.equal(importBatchEntries(entries, "unknown")[0].operationKey, "unknown-key");
  assert.deepEqual(importBatchEntries(entries, "succeeded"), []);
  assert.deepEqual(importCounts(entries), { total: 5, succeeded: 1, failed: 1, unknown: 1, pending: 2 });
});

test("receipts verify project, filename and positive version, not merely a truthy 2xx payload", () => {
  const receipt = { id: "doc-1", project_id: "p-1", name: files[0].name, version: 2, deduplicated: true };
  assert.deepEqual(verifiedImportReceipt(receipt, "p-1", files[0].name), receipt);
  for (const patch of [{ id: "" }, { project_id: "p-2" }, { name: "other.docx" }, { version: 0 }, { version: 1.5 }]) assert.equal(verifiedImportReceipt({ ...receipt, ...patch }, "p-1", files[0].name), null);
  assert.equal(verifiedImportReceipt({ id: "doc-1" }, "p-1", files[0].name), null);
});

test("transport errors, 5xx, malformed success and idempotency conflict stay ambiguous, not auto-retryable failures", () => {
  const reasons = [new TypeError("network"), new ApiError("unavailable", { status: 503, statusText: "Unavailable" }), new ApiError("bad JSON", { status: 201, statusText: "Created" }), new ApiError("conflict", { status: 409, statusText: "Conflict", detail: { detail: { reason_code: "document_upload_idempotency_conflict" } } })];
  for (const reason of reasons) assert.equal(importFailure(reason).status, "unknown");
});

test("explicit rejected input remains editable while permission/session boundaries stop subsequent files", () => {
  const rejected = importFailure(new ApiError("UTF-8 required", { status: 400, statusText: "Bad Request" }));
  assert.equal(rejected.status, "failed");
  assert.equal(rejected.stop, false);
  for (const status of [401, 403, 404]) assert.equal(importFailure(new ApiError("Denied", { status, statusText: "Denied" })).stop, true);
});

test("invalid story scope fails local validation before a document request is prepared", () => {
  assert.throws(() => importContextFields({ ...emptyImportDefaults, storyScope: "route / secret" }), /作用域/);
  assert.deepEqual(importContextFields({ ...emptyImportDefaults, storyScope: " global " }), { story_scope: "global" });
});

test("explicit renamed upload captures an immutable target separately from the unchanged File", () => {
  const target = { id: "old-doc", name: "原章.docx", version: 3 };
  const file = { name: "001-修订.md", size: 32, lastModified: 42 };
  const entry = appendImportEntries([], [file], { ...emptyImportDefaults, replaceDocumentId: target.id, replacement: target }, () => "fixed").entries[0];
  target.name = "late title"; target.version = 8;
  assert.equal(entry.file, file);
  assert.deepEqual(entry.replacement, { id: "old-doc", name: "原章.docx", version: 3 });
  assert.equal(Object.isFrozen(entry.replacement), true);
  assert.deepEqual(importContextFields(entry), { replace_document_id: "old-doc" });
  assert.equal(frozenImportReplacement("other", entry.replacement), null);
});

test("changing an editable target is a new operation; unknown targets stay fixed including identity", () => {
  const first = { id: "old", name: "原稿.docx", version: 1 };
  const second = { id: "other", name: "另一稿.md", version: 7 };
  const initial = appendImportEntries([], [files[0]], { ...emptyImportDefaults, replaceDocumentId: first.id, replacement: first }, () => "first-key").entries[0];
  const changed = updateImportEntry(initial, { replaceDocumentId: second.id, replacement: second }, () => "new-key");
  assert.equal(changed.operationKey, "new-key"); assert.deepEqual(changed.replacement, second);
  assert.equal(updateImportEntry({ ...initial, status: "unknown" }, { replaceDocumentId: second.id, replacement: second }, () => "forbidden").replacement, initial.replacement);
  assert.equal(updateImportEntry(initial, { replaceDocumentId: "" }, () => "clear").replacement, null);
  assert.equal(updateImportEntry(initial, { replaceDocumentId: "other" }, () => "unverified").replacement, null);
});

test("explicit mapping receipt binds uploaded format, original logical target and its successor", () => {
  const target = { id: "old-doc", name: "原章.docx", version: 3 };
  const receipt = { id: "new-doc", project_id: "p", name: "原章.docx", version: 4, deduplicated: true,
    uploaded_filename: "001-修订.md", uploaded_format: "markdown", replaces_document_id: "old-doc", replaces_document_version: 3,
    superseded_document_ids: ["old-doc"], upload_metadata_source: "stored_receipt" };
  const value = verifiedImportReceipt(receipt, "p", "001-修订.md", target);
  assert.equal(value.name, "原章.docx"); assert.equal(value.uploaded_filename, "001-修订.md");
  for (const source of ["stored_receipt", "verified_request", "current_request"]) {
    assert.equal(verifiedImportReceipt({ ...receipt, upload_metadata_source: source }, "p", "001-修订.md", target).upload_metadata_source, source);
  }
  for (const patch of [{ id: "old-doc" }, { project_id: "other" }, { name: "001-修订.md" }, { version: 3 }, { version: 5 },
    { uploaded_filename: "changed.md" }, { uploaded_format: "docx" }, { replaces_document_id: "other" },
    { replaces_document_version: 2 }, { superseded_document_ids: ["other"] }, { upload_metadata_source: "guessed" }]) {
    assert.equal(verifiedImportReceipt({ ...receipt, ...patch }, "p", "001-修订.md", target), null, JSON.stringify(patch));
  }
  for (const key of ["uploaded_filename", "uploaded_format", "replaces_document_id", "replaces_document_version", "superseded_document_ids"]) {
    const missing = { ...receipt }; delete missing[key];
    assert.equal(verifiedImportReceipt(missing, "p", "001-修订.md", target), null, key);
  }
});

test("unmapped receipts retain old semantics and do not accept a hidden replacement claim", () => {
  const old = { id: "new", project_id: "p", name: "new.md", version: 1 };
  assert.deepEqual(verifiedImportReceipt(old, "p", "new.md"), old);
  const receipt = { ...old, uploaded_filename: "new.md", uploaded_format: "markdown", replaces_document_id: null, replaces_document_version: null };
  assert.equal(verifiedImportReceipt(receipt, "p", "new.md").name, "new.md");
  assert.equal(verifiedImportReceipt({ ...receipt, replaces_document_id: "unexpected" }, "p", "new.md"), null);
  assert.equal(verifiedImportReceipt({ ...receipt, uploaded_format: "docx" }, "p", "new.md"), null);
});
