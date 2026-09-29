import test from "node:test";
import assert from "node:assert/strict";

import {
  fetchProvisionalClues,
  normalizeProvisionalClueResult,
  provisionalClueDimensionLabel,
} from "../src/provisionalClues.ts";

const clue = {
  id: `pc_${"a".repeat(32)}`,
  document_id: "document-1",
  document_version: 2,
  document_name: "新稿.md",
  line_start: 12,
  line_end: 13,
  evidence: "林澈放下钥匙。\n她转身离开。",
  character: "林澈",
  dimension: "behavior_boundary",
  proposed_statement: "林澈可能改变了行为边界。",
  reason: "partial_model_package",
};

test("provisional clue contract keeps model proposals separate from issue fields", () => {
  const result = normalizeProvisionalClueResult({ items: [clue], truncated: false });
  assert.deepEqual(result.items[0], clue);
  assert.equal(result.items[0].reason, "partial_model_package");
  assert.equal(provisionalClueDimensionLabel(result.items[0].dimension), "行为边界");
  assert.equal(provisionalClueDimensionLabel("relationship_attitude"), "关系态度");
  assert.equal(provisionalClueDimensionLabel("motivation_goal"), "长期动机/目标");
  assert.deepEqual(normalizeProvisionalClueResult({ items: [], truncated: false }), {
    items: [], truncated: false,
  });
});

test("invalid or ambiguous clue responses fail closed instead of looking like zero clues", () => {
  for (const payload of [
    null,
    { clues: [clue], truncated: false },
    { items: [], truncated: null },
    { items: [clue], truncated: true },
    { items: [{ ...clue, evidence: "" }], truncated: false },
    { items: [{ ...clue, evidence: "字".repeat(2_001) }], truncated: false },
    { items: [{ ...clue, proposed_statement: "字".repeat(301) }], truncated: false },
    { items: [{ ...clue, proposed_statement: "字" }], truncated: false },
    { items: [{ ...clue, id: "issue-1" }], truncated: false },
    { items: [{ ...clue, line_end: 11 }], truncated: false },
    { items: [{ ...clue, line_start: 10_000_001, line_end: 10_000_001 }], truncated: false },
    { items: [{ ...clue, reason: "confirmed_issue" }], truncated: false },
    { items: [{ ...clue, dimension: "other" }], truncated: false },
    { items: [clue, clue], truncated: false },
    { items: Array.from({ length: 65 }, (_, index) => ({ ...clue, id: `pc_${index.toString(16).padStart(32, "0")}` })), truncated: true },
  ]) {
    assert.throws(() => normalizeProvisionalClueResult(payload), TypeError);
  }
});

test("clue API reads the run-scoped endpoint and preserves truncation", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => { globalThis.fetch = originalFetch; });
  const truncatedClues = Array.from({ length: 64 }, (_, index) => ({
    ...clue,
    id: `pc_${index.toString(16).padStart(32, "0")}`,
  }));
  globalThis.fetch = async (url, init) => {
    assert.equal(String(url), "/api/v1/analysis-runs/run%20one/provisional-clues");
    assert.equal(init.credentials, "include");
    return new Response(JSON.stringify({ items: truncatedClues, truncated: true }));
  };
  const result = await fetchProvisionalClues("run one");
  assert.equal(result.items.length, 64);
  assert.equal(result.truncated, true);
});
