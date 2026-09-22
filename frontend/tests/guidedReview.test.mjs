import test from "node:test";
import assert from "node:assert/strict";
import { createImportFilePlan } from "../src/app/importPlan.ts";
import {
  analysisRunRequest,
  contextNavigationLocked,
  contextDraft,
  contextStatus,
  findCurrentBaselineRun,
  guidedDocumentState,
  hasCompletedBaselineRun,
  isLoadedContextForDocument,
  narrativeContextPayload,
  normalizeNarrativeContextInference,
  responseBelongsToSelectedDocument,
} from "../src/features/workflow/guidedReview.ts";

function document(overrides = {}) {
  return {
    id: "doc-1",
    name: "资料.md",
    version: 1,
    active: true,
    document_role: "reference",
    story_scope: "global",
    narrative_context: {
      context_revision: 0,
      resolution_state: "unresolved",
      origin: "deterministic_import",
      publication_status: "unknown",
      scope: { schema_version: 1, timeline_key: "main" },
    },
    ...overrides,
  };
}

test("bulk imports use a neutral role until the creator confirms context", () => {
  const files = [{ name: "第一章.docx" }, { name: "设定.txt" }];
  assert.deepEqual(
    createImportFilePlan(files).map((entry) => entry.documentRole),
    ["reference", "reference"],
  );
});

test("context draft preserves release, branch and activity fields", () => {
  const source = document({
    document_role: "chapter",
    narrative_context: {
      revision: 3,
      resolution_state: "confirmed",
      origin: "explicit",
      publication_status: "published",
      scope: {
        schema_version: 1,
        timeline_key: "main",
        release: { key: "2.1", ordinal: 21 },
        branch: { path: ["主线", "北港"], exclusive_group: "choice_8" },
        activity_key: "summer_event",
      },
    },
  });
  assert.deepEqual(contextDraft(source), {
    documentRole: "chapter",
    publicationStatus: "published",
    timelineKey: "main",
    releaseKey: "2.1",
    releaseOrdinal: "21",
    branchPath: "主线 / 北港",
    exclusiveGroup: "choice_8",
    activityKey: "summer_event",
    confirmed: true,
  });
});

test("confirmed context payload includes document role and optimistic revision", () => {
  assert.deepEqual(
    narrativeContextPayload(
      {
        documentRole: "character_profile",
        publicationStatus: "published",
        timelineKey: "main",
        releaseKey: "v2",
        releaseOrdinal: "2",
        branchPath: "主线 / A线",
        exclusiveGroup: "ending",
        activityKey: "event_2",
        confirmed: true,
      },
      4,
    ),
    {
      document_role: "character_profile",
      resolution_state: "confirmed",
      publication_status: "published",
      scope: {
        schema_version: 1,
        timeline_key: "main",
        release: { key: "v2", ordinal: 2 },
        branch: { path: ["主线", "A线"], exclusive_group: "ending" },
        activity_key: "event_2",
      },
      expected_revision: 4,
    },
  );
  assert.throws(
    () =>
      narrativeContextPayload(
        {
          documentRole: "chapter",
          publicationStatus: "draft",
          timelineKey: "main",
          releaseKey: "v3",
          releaseOrdinal: "",
          branchPath: "",
          exclusiveGroup: "",
          activityKey: "",
          confirmed: false,
        },
        0,
      ),
    /版本顺序/,
  );
});

test("AI inference is never presented as human confirmation", () => {
  assert.deepEqual(
    contextStatus({
      revision: 2,
      resolution_state: "inferred",
      origin: "model_inferred",
      publication_status: "draft",
      scope: { schema_version: 1, timeline_key: "main" },
    }),
    {
      tone: "inferred",
      label: "AI 推断待确认",
      detail: "模型推断不会自动成为正式上下文，请核对后保存。",
    },
  );
});

test("a loaded context is editable only while it still belongs to the selected document", () => {
  assert.equal(isLoadedContextForDocument("doc-a", "doc-a"), true);
  assert.equal(isLoadedContextForDocument("doc-a", "doc-b"), false);
  assert.equal(isLoadedContextForDocument(null, "doc-a"), false);
});

test("context navigation locks only for mutations and stale responses cannot target another document", () => {
  assert.equal(contextNavigationLocked(true, false), true);
  assert.equal(contextNavigationLocked(false, true), true);
  assert.equal(contextNavigationLocked(false, false), false);
  assert.equal(responseBelongsToSelectedDocument("doc-a", "doc-a"), true);
  assert.equal(responseBelongsToSelectedDocument("doc-a", "doc-b"), false);
  assert.equal(responseBelongsToSelectedDocument("doc-a", null), false);
});

test("AI context inference keeps grounded evidence and remains unconfirmed", () => {
  const result = normalizeNarrativeContextInference({
    document_id: "doc-1",
    document_role: "character_profile",
    suggestion: {
      revision: 2,
      resolution_state: "inferred",
      origin: "model_inferred",
      authority_tier: "unresolved",
      publication_status: "published",
      scope: { schema_version: 1, timeline_key: "main" },
      inference: {
        confidence: 0.82,
        reasoning: "正文以角色档案形式列出固定偏好。",
        evidence: [{
          document_id: "doc-1",
          document_name: "角色资料.md",
          line_start: 2,
          line_end: 3,
          text: "角色：岚。长期偏好：蜜瓜。",
          supported_fields: ["document_role", "publication_status"],
        }],
        usage: { prompt_tokens: 100, completion_tokens: 40, total_tokens: 140 },
      },
    },
    usage: { prompt_tokens: 100, completion_tokens: 40, total_tokens: 140 },
  });
  assert.equal(result.document_role, "character_profile");
  assert.equal(result.suggestion.resolution_state, "inferred");
  assert.equal(contextDraft({ ...document(), document_role: result.document_role }, result.suggestion).confirmed, false);
  assert.equal(result.suggestion.inference.evidence[0].line_start, 2);
  assert.equal(result.suggestion.inference.usage.total_tokens, 140);
});

test("malformed AI context responses never become empty suggestions", () => {
  assert.throws(
    () => normalizeNarrativeContextInference({
      document_id: "doc-1",
      document_role: "chapter",
      suggestion: {
        revision: 1,
        resolution_state: "inferred",
        origin: "model_inferred",
        publication_status: "draft",
        scope: { schema_version: 1, timeline_key: "main" },
        inference: {
          confidence: 0.6,
          reasoning: "没有证据",
          evidence: [],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        },
      },
      usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
    }),
    /原文证据/,
  );
  assert.throws(
    () => normalizeNarrativeContextInference({ document_id: "doc-1" }),
    /建议不完整/,
  );
});

test("only confirmed canon, character profiles and published chapters can ready a baseline", () => {
  const documents = [
    document({ id: "canon", document_role: "canon" }),
    document({
      id: "history",
      document_role: "chapter",
      narrative_context: {
        revision: 1,
        resolution_state: "confirmed",
        publication_status: "published",
        scope: { schema_version: 1, timeline_key: "main" },
      },
    }),
    document({
      id: "draft",
      document_role: "chapter",
      narrative_context: {
        revision: 1,
        resolution_state: "confirmed",
        publication_status: "draft",
        scope: { schema_version: 1, timeline_key: "main" },
      },
    }),
    document({
      id: "reference",
      document_role: "reference",
      narrative_context: {
        revision: 1,
        resolution_state: "confirmed",
        publication_status: "published",
        scope: { schema_version: 1, timeline_key: "main" },
      },
    }),
    document({
      id: "retired",
      document_role: "chapter",
      narrative_context: {
        revision: 1,
        resolution_state: "confirmed",
        publication_status: "retired",
        scope: { schema_version: 1, timeline_key: "main" },
      },
    }),
  ];
  const state = guidedDocumentState(documents);
  assert.deepEqual(state.baseline.map((item) => item.id), ["canon", "history"]);
  assert.deepEqual(state.unresolvedBaseline.map((item) => item.id), ["canon"]);
  assert.deepEqual(state.confirmedDrafts.map((item) => item.id), ["draft"]);
});

test("analysis requests keep baseline targets server-derived and draft targets explicit", () => {
  assert.deepEqual(analysisRunRequest("baseline_build", "balanced", ["ignored"]), {
    mode: "baseline_build",
    sensitivity: "balanced",
  });
  assert.deepEqual(
    analysisRunRequest("draft_review", "exploratory", [" doc-a ", "doc-a", "doc-b"]),
    {
      mode: "draft_review",
      sensitivity: "exploratory",
      target_document_ids: ["doc-a", "doc-b"],
    },
  );
  assert.throws(
    () => analysisRunRequest("draft_review", "conservative"),
    /至少选择一份/,
  );
});

test("completed baseline recovery uses the nested durable review batch contract", () => {
  assert.equal(
    hasCompletedBaselineRun([
      { status: "completed", mode: "baseline_build", review_batch: { mode: "full_review" } },
      { status: "completed", review_batch: { mode: "baseline_build" } },
    ]),
    true,
  );
  assert.equal(
    hasCompletedBaselineRun([
      { status: "running", review_batch: { mode: "baseline_build" } },
      { status: "completed", review_batch: { mode: "draft_review" } },
    ]),
    false,
  );
  assert.equal(
    hasCompletedBaselineRun([{ status: "completed", mode: "baseline_build" }]),
    true,
  );
});

test("only a frozen baseline matching current versions and context revisions stays current", () => {
  const canon = document({
    id: "canon",
    version: 3,
    document_role: "canon",
    narrative_context: {
      context_revision: 4,
      resolution_state: "confirmed",
      origin: "explicit",
      publication_status: "published",
      scope: { schema_version: 1, timeline_key: "main" },
      scope_sha256: "scope-4",
    },
  });
  const run = {
    id: "baseline-1",
    status: "completed",
    input_snapshot_available: true,
    review_batch: { mode: "baseline_build", background_document_ids: ["canon"] },
    input_documents: [{
      document_id: "canon",
      document_version: 3,
      document_role: "canon",
      batch_role: "background",
      narrative_context: {
        context_revision: 4,
        resolution_state: "confirmed",
        publication_status: "published",
        scope_sha256: "scope-4",
      },
    }],
  };
  assert.equal(findCurrentBaselineRun([canon], [run])?.id, "baseline-1");
  assert.equal(findCurrentBaselineRun([{ ...canon, version: 4 }], [run]), null);
  assert.equal(findCurrentBaselineRun([{
    ...canon,
    narrative_context: { ...canon.narrative_context, context_revision: 5 },
  }], [run]), null);
  assert.equal(findCurrentBaselineRun([canon, document({ id: "new-canon", document_role: "canon" })], [run]), null);
});
