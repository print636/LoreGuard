import test from "node:test";
import assert from "node:assert/strict";

import {
  comparisonBelongsToRevision,
  comparisonPath,
  comparisonRetryDelay,
  createMutationGuard,
  mergeRevisionRouteState,
  pendingVisualizationKinds,
  recheckPath,
  revisionDelta,
  revisionDocumentForIssue,
  revisionRouteStateFromSearch,
  revisionSearch,
  validateRevisionUploadSelection,
} from "../src/revisionWorkflow.ts";

test("revision route state round-trips only whitelisted workflow values", () => {
  const search = revisionSearch({
    step: "compare",
    issueId: "issue-1",
    documentId: "doc-2",
    recheckRunId: "run-3",
    outcome: "no_longer_detected",
    generateGraph: true,
    generateTimeline: false,
    page: 3,
  });
  assert.deepEqual(revisionRouteStateFromSearch(search), {
    step: "compare",
    issueId: "issue-1",
    documentId: "doc-2",
    recheckRunId: "run-3",
    outcome: "no_longer_detected",
    generateGraph: true,
    generateTimeline: false,
    page: 3,
  });
  assert.deepEqual(
    revisionRouteStateFromSearch("?step=delete&issue=../../secret&outcome=resolved&recheck=" + "x".repeat(129)),
    {
      step: "review",
      issueId: null,
      documentId: null,
      recheckRunId: null,
      outcome: "all",
      generateGraph: false,
      generateTimeline: false,
      page: 1,
    },
  );
});

test("revision delta distinguishes added, removed, and versioned active inputs", () => {
  const baseline = [
    { document_id: "old-a", document_name: "chapter.md", document_version: 1, document_role: "chapter", story_scope: "global", content_sha256: "a" },
    { document_id: "old-b", document_name: "rules.md", document_version: 1, document_role: "canon", story_scope: "global", content_sha256: "b" },
  ];
  const documents = [
    { id: "old-a", name: "chapter.md", version: 1, active: false, document_role: "chapter", story_scope: "global" },
    { id: "new-a", name: "chapter.md", version: 2, active: true, document_role: "chapter", story_scope: "global" },
    { id: "new-c", name: "notes.md", version: 1, active: true, document_role: "reference", story_scope: "route_a" },
  ];
  const delta = revisionDelta(baseline, documents);
  assert.equal(delta.changed, true);
  assert.equal(delta.updated[0].after.id, "new-a");
  assert.equal(delta.added[0].name, "notes.md");
  assert.equal(delta.removed[0].document_name, "rules.md");
  assert.equal(revisionDocumentForIssue("CHAPTER.md", documents), "new-a");
});

test("revision API paths encode run identifiers", () => {
  assert.equal(recheckPath("run one"), "/api/v1/analysis-runs/run%20one/rechecks");
  assert.equal(comparisonPath("run one"), "/api/v1/analysis-runs/run%20one/comparison?limit=20&offset=0");
  assert.equal(comparisonPath("run one", 25, 50), "/api/v1/analysis-runs/run%20one/comparison?limit=25&offset=50");
  assert.equal(comparisonPath("run one", 20, 0, "new"), "/api/v1/analysis-runs/run%20one/comparison?limit=20&offset=0&outcome=new");
});

test("comparison identity fails closed unless project, baseline, and target all match", () => {
  const expected = { projectId: "project-1", baselineRunId: "base-1", targetRunId: "run-2" };
  assert.equal(comparisonBelongsToRevision({
    project_id: "project-1",
    baseline_run_id: "base-1",
    target_run_id: "run-2",
  }, expected), true);
  assert.equal(comparisonBelongsToRevision({
    project_id: "project-2",
    baseline_run_id: "base-1",
    target_run_id: "run-2",
  }, expected), false);
  assert.equal(comparisonBelongsToRevision({
    baseline_run_id: "base-1",
    target_run_id: "run-2",
  }, expected), false);
});

test("outcome transitions always reset comparison pagination", () => {
  const current = revisionRouteStateFromSearch("?step=compare&outcome=new&page=4");
  assert.equal(mergeRevisionRouteState(current, { outcome: "persisting", page: 9 }).page, 1);
  assert.equal(mergeRevisionRouteState(current, { page: 3 }).page, 3);
});

test("mutation guard rejects same-tick duplicate starts and can be released", () => {
  const guard = createMutationGuard();
  assert.equal(guard.tryBegin(), true);
  assert.equal(guard.tryBegin(), false);
  guard.end();
  assert.equal(guard.tryBegin(), true);
});

test("comparison retry delay grows but stays bounded", () => {
  assert.equal(comparisonRetryDelay(1), 2_500);
  assert.equal(comparisonRetryDelay(2), 5_000);
  assert.equal(comparisonRetryDelay(99), 15_000);
  assert.equal(comparisonRetryDelay(Number.NaN), 2_500);
});

test("revision upload validates active version chains and same-name related files", () => {
  const documents = [
    { id: "old", name: "chapter.md", version: 1, active: false, document_role: "chapter", story_scope: "global" },
    { id: "active", name: "chapter.md", version: 2, active: true, document_role: "chapter", story_scope: "global" },
  ];
  assert.equal(validateRevisionUploadSelection("replace", "chapter.md", "active", documents).ok, true);
  assert.deepEqual(validateRevisionUploadSelection("replace", "chapter.md", "old", documents), {
    ok: false,
    message: "替换目标已经不是当前生效文档，请重新选择后再上传。",
    field: "document",
  });
  assert.equal(validateRevisionUploadSelection("replace", "renamed.md", "active", documents).ok, false);
  assert.deepEqual(validateRevisionUploadSelection("related", "CHAPTER.MD", null, documents), {
    ok: false,
    message: "项目中已有同名生效文档“chapter.md”。请选择“替换现有文档”并指定它，避免断开版本链。",
    field: "mode",
  });
  assert.equal(validateRevisionUploadSelection("related", "notes.md", null, documents).ok, true);
});

test("visualization requests are tracked independently per run and kind", () => {
  const generated = new Set(["run-1:graph"]);
  assert.deepEqual(pendingVisualizationKinds("run-1", true, true, generated), ["timeline"]);
  assert.deepEqual(pendingVisualizationKinds("run-2", true, false, generated), ["graph"]);
  assert.deepEqual(pendingVisualizationKinds("run-1", false, false, generated), []);
});
