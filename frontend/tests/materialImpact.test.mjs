import test from "node:test";
import assert from "node:assert/strict";
import { normalizeMaterialImpact, materialImpactPath, canPrepareRecheck, impactHeadline, sourceLabel, sourceContext } from "../src/features/material-impact/materialImpactModel.ts";

function result() {
  return {
    schema_version: "material-change-impact-v1", project_id: "project", run_id: "run",
    comparison_status: "complete", semantic_impact_complete: false, confirmed_traits_auto_invalidated: false, confirmed_traits_changed: false,
    recheck: { inputs_changed: true, can_prepare: true, reason: null },
    sources: { offset: 0, limit: 30, total: 1, has_more: false, items: [{ key: "source", kind: "input", changes: ["version_changed", "content_changed"],
      frozen: { document_id: "old", document_name: "设定.md", document_version: 1, document_role: "canon", story_scope: "global", batch_role: "background", publication_status: "published" },
      current: { document_id: "new", document_name: "设定.md", document_version: 2, document_role: "canon", story_scope: "global", batch_role: "background", publication_status: "published" },
      selected_for_recheck: true, content_changed: true, identity_comparison_available: true }] },
    affected_issues: { offset: 0, limit: 30, total: 1, has_more: false, items: [{ id: "issue", title: "旧问题", report_class: "formal", source_keys: ["source"], evidence_indices: [0] }] },
    summary: { changed_sources: 1, added_sources: 0, not_included_sources: 0, affected_issues: 1, unresolved_evidence: 0, unresolved_sources: 0 },
  };
}
test("material impact keeps explicit frozen and current versions and does not mutate old report data", () => {
  const raw = result(); const original = structuredClone(raw);
  const value = normalizeMaterialImpact(raw, "project", "run");
  assert.equal(sourceLabel(value.sources.items[0].frozen), "设定.md · v1");
  assert.equal(sourceLabel(value.sources.items[0].current), "设定.md · v2");
  assert.equal(canPrepareRecheck(value), true);
  assert.deepEqual(raw, original);
});
test("late response for another project, run or source page is rejected", () => {
  for (const [field, other] of [["project_id", "other"], ["run_id", "other"]]) {
    const raw = result(); raw[field] = other;
    assert.throws(() => normalizeMaterialImpact(raw, "project", "run"));
  }
  assert.throws(() => normalizeMaterialImpact(result(), "project", "run", 30));
  assert.throws(() => normalizeMaterialImpact(result(), "project", "run", 0, 30));
});
test("no changes, unavailable comparison and unselected draft never authorize recheck locally", () => {
  assert.equal(canPrepareRecheck(null), false);
  for (const update of [{ inputs_changed: false }, { inputs_changed: null }, { can_prepare: false }]) {
    const raw = result(); Object.assign(raw.recheck, update);
    assert.equal(canPrepareRecheck(raw), false);
  }
  const raw = result(); raw.comparison_status = "unavailable";
  assert.equal(canPrepareRecheck(raw), false);
  assert.match(impactHeadline(raw), /无法核对/);
});
test("identity-only change can prepare without a different document version", () => {
  const raw = result(); raw.sources.items[0].current = { ...raw.sources.items[0].frozen, document_role: "reference" };
  raw.sources.items[0].content_changed = false; raw.sources.items[0].changes = ["identity_changed"];
  assert.equal(canPrepareRecheck(normalizeMaterialImpact(raw, "project", "run")), true);
});
test("source identity shows shared role/publication names, original scope, and distinguishes absent fields", () => {
  const source = result().sources.items[0].frozen;
  assert.equal(sourceContext({ ...source, document_role: "character_profile", publication_status: "retired", story_scope: "冬季支线" }),
    "比对依据 · 角色档案 · 已归档 · 故事范围：冬季支线");
  assert.match(sourceContext({ ...source, document_role: "chapter", publication_status: "in_review", story_scope: "main" }), /故事正文\/章节 · 审阅中 · 故事范围：main/);
  assert.match(sourceContext({ ...source, publication_status: "unknown" }), /待确认/);
  assert.equal(sourceContext({ ...source, document_role: null, publication_status: null, story_scope: null }),
    "比对依据 · 用途未记录 · 发布状态未记录 · 故事范围：未记录");
  assert.notEqual(sourceContext({ ...source, story_scope: "main" }), sourceContext({ ...source, story_scope: "branch" }));
  assert.equal(sourceContext(null), "");
});
test("unselected materials and missing references remain limitations rather than a clean verdict", () => {
  const raw = result(); raw.recheck = { inputs_changed: false, can_prepare: false, reason: "no_document_changes" };
  raw.summary.not_included_sources = 1; raw.summary.affected_issues = 0; raw.summary.unresolved_evidence = 2;
  raw.summary.unresolved_sources = 1; raw.sources.items[0].identity_comparison_available = false;
  assert.equal(normalizeMaterialImpact(raw, "project", "run").summary.unresolved_sources, 1);
  assert.match(impactHeadline(raw), /尚未纳入/);
  raw.summary.changed_sources = 0; raw.summary.not_included_sources = 0;
  assert.equal(impactHeadline(raw), "未发现明确的输入变化");
});
test("malformed relation, review class, counts and proof-like scope upgrades are rejected", () => {
  for (const mutate of [r => { r.sources.items[0].changes = ["guessed_rename"]; }, r => { r.affected_issues.items[0].report_class = "conflict"; },
    r => { r.affected_issues.items[0].evidence_indices = [-1]; }, r => { r.summary.changed_sources = -1; },
    r => { r.semantic_impact_complete = true; }, r => { r.confirmed_traits_auto_invalidated = true; }]) {
    const raw = result(); mutate(raw); assert.throws(() => normalizeMaterialImpact(raw, "project", "run"));
  }
});
test("separate source and reference pages retain server totals and encoded exact endpoint", () => {
  const raw = result(); raw.sources.offset = 30; raw.sources.total = 61; raw.sources.has_more = true;
  raw.affected_issues.offset = 60; raw.affected_issues.total = 62;
  assert.equal(normalizeMaterialImpact(raw, "project", "run", 30, 60).sources.total, 61);
  assert.equal(materialImpactPath("run / one", 30, 60), "/api/v1/analysis-runs/run%20%2F%20one/material-impact?offset=30&limit=30&issue_offset=60&issue_limit=30");
});
