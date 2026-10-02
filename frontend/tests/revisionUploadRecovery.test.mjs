import test from "node:test";
import assert from "node:assert/strict";
import { ApiError } from "../src/api/client.ts";
import { importFailure, ImportResultUnknown, verifiedImportReceipt } from "../src/features/imports/importQueueModel.ts";
import { frozenRevisionUpload, revisionUploadBelongsTo, revisionUploadMatchesInput, revisionUploadMaySend } from "../src/revisionUploadRecovery.ts";

const file = { name: "chapter.md", size: 50, lastModified: 1234 };
const input = { projectId: "project-a", baselineRunId: "run-a", file, documentRole: "chapter", storyScope: " route_a ", replacement: { id: "old-doc", name: "chapter.md", version: 3 } };

test("revision operation freezes canonical payload and exact target under one stable key", () => {
  let keys = 0;
  const operation = frozenRevisionUpload(input, () => `key-${++keys}`);
  assert.equal(keys, 1);
  assert.equal(operation.entry.operationKey, "key-1");
  assert.equal(operation.entry.file, file);
  assert.deepEqual(operation.fields, { document_role: "chapter", story_scope: "route_a", replace_document_id: "old-doc" });
  assert.equal(Object.isFrozen(operation.fields), true);
  assert.throws(() => { operation.fields.story_scope = "other"; }, TypeError);
  assert.equal(operation.replacementName, "chapter.md");
  assert.equal(operation.replacementVersion, 3);
});

test("unchanged File/context may reuse key but new File or target/context is a different operation", () => {
  const operation = frozenRevisionUpload(input, () => "fixed");
  const candidate = { ...input, replaceDocumentId: "old-doc", storyScope: "route_a" };
  assert.equal(revisionUploadMatchesInput(operation, candidate), true);
  for (const changed of [
    { file: { ...file } }, { replaceDocumentId: "new-doc" }, { documentRole: "reference" },
    { storyScope: "route_b" }, { projectId: "project-b" }, { baselineRunId: "run-b" },
  ]) assert.equal(revisionUploadMatchesInput(operation, { ...candidate, ...changed }), false);
  assert.equal(revisionUploadBelongsTo(operation, "project-a", "run-a"), true);
  assert.equal(revisionUploadBelongsTo(operation, "project-b", "run-a"), false);
  assert.equal(revisionUploadBelongsTo(operation, "project-a", "run-b"), false);
});

test("unknown revision requires explicit original operation retry; saved operation is never sendable", () => {
  const operation = frozenRevisionUpload(input, () => "fixed");
  for (const status of ["queued", "failed"]) assert.equal(revisionUploadMaySend({ ...operation, entry: { ...operation.entry, status } }, false), true);
  const unknown = { ...operation, entry: { ...operation.entry, status: "unknown" } };
  assert.equal(revisionUploadMaySend(unknown, false), false);
  assert.equal(revisionUploadMaySend(unknown, true), true);
  for (const status of ["uploading", "succeeded"]) {
    const locked = { ...operation, entry: { ...operation.entry, status } };
    assert.equal(revisionUploadMaySend(locked, false), false);
    assert.equal(revisionUploadMaySend(locked, true), false);
  }
});

test("related upload omits replacement and empty scope normalizes to global", () => {
  const operation = frozenRevisionUpload({ ...input, replacement: null, storyScope: "" }, () => "related");
  assert.deepEqual(operation.fields, { document_role: "chapter", story_scope: "global" });
  assert.equal(operation.replacementName, null);
  assert.equal(operation.replacementVersion, null);
  assert.throws(() => frozenRevisionUpload({ ...input, storyScope: "route/a" }, () => "bad"));
});

test("revision receipt validates project/name/version and replay retains returned document", () => {
  const response = { id: "saved-doc", project_id: "project-a", name: file.name, version: 4, deduplicated: true };
  assert.deepEqual(verifiedImportReceipt(response, "project-a", file.name), response);
  for (const changed of [{ id: "" }, { project_id: "other" }, { name: "other.md" }, { version: 0 }, { version: 1.5 }, { version: "4" }]) {
    assert.equal(verifiedImportReceipt({ ...response, ...changed }, "project-a", file.name), null);
  }
  assert.equal(verifiedImportReceipt(null, "project-a", file.name), null);
});

test("ambiguous transport/server/malformed responses remain unknown, not falsely failed", () => {
  const apiError = (status, detail = null) => new ApiError("test error", { status, statusText: "test", detail });
  for (const error of [new TypeError("network"), new ImportResultUnknown(), apiError(503), apiError(408), apiError(409, { detail: { reason_code: "document_upload_idempotency_conflict" } })]) {
    assert.equal(importFailure(error).status, "unknown");
  }
  assert.equal(importFailure(apiError(422)).status, "failed");
  assert.equal(importFailure(apiError(401)).stop, true);
});
