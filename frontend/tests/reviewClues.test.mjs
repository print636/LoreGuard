import test from "node:test";
import assert from "node:assert/strict";

import { fetchReviewClues, normalizeReviewClueResult } from "../src/reviewClues.ts";

const clue = {
  id: "71475dcc-1d0c-4dd2-a014-06941adf1bf3",
  report_class: "review_clue",
  category: "character_drift",
  title: "林澈的角色表现需要确认",
  explanation: "当前只见一次不同表现，尚不能判定角色设定改变。",
  evidence: [
    { document_id: "profile", document_name: "设定.md", line_start: 3, line_end: 3, text: "林澈在陌生人前沉默寡言。" },
    { document_id: "draft", document_name: "新稿.md", line_start: 12, line_end: 12, text: "林澈主动在众人面前长谈。" },
  ],
  metadata: { final_outcome: "needs_confirmation", review_reason: "single_behavior_is_not_drift" },
};

test("review clue requires a separate report class and both source spans", () => {
  const result = normalizeReviewClueResult({ items: [clue], truncated: false });
  assert.equal(result.items[0].report_class, "review_clue");
  assert.equal(result.items[0].evidence[0].text, clue.evidence[0].text);
  assert.equal(result.items[0].evidence[1].text, clue.evidence[1].text);
  const indentedSource = `  ${clue.evidence[1].text}  `;
  assert.equal(
    normalizeReviewClueResult({
      items: [{ ...clue, evidence: [clue.evidence[0], { ...clue.evidence[1], text: indentedSource }] }],
      truncated: false,
    }).items[0].evidence[1].text,
    indentedSource,
  );
  assert.equal(result.items[0].metadata.final_outcome, "needs_confirmation");
  const versionedSameCoordinate = {
    ...clue,
    evidence: [
      clue.evidence[0],
      { ...clue.evidence[0], text: "同一文档坐标的新冻结版本原文。" },
    ],
  };
  assert.equal(
    normalizeReviewClueResult({ items: [versionedSameCoordinate], truncated: false })
      .items[0].evidence[1].text,
    "同一文档坐标的新冻结版本原文。",
  );
  assert.equal(
    normalizeReviewClueResult({
      items: [{ ...clue, metadata: { ...clue.metadata, final_outcome: "unverifiable" } }],
      truncated: false,
    }).items[0].metadata.final_outcome,
    "unverifiable",
  );
  assert.deepEqual(normalizeReviewClueResult({ items: [], truncated: false }), {
    items: [], truncated: false, unavailable_count: 0, scan_limited: false,
  });
  assert.equal(normalizeReviewClueResult({ items: [], truncated: false, unavailable_count: 3 }).unavailable_count, 3);
  assert.equal(normalizeReviewClueResult({ items: [], truncated: false, scan_limited: true }).scan_limited, true);
});

test("invalid review clue payload fails instead of becoming an empty result or formal issue", () => {
  for (const payload of [
    null,
    { items: [], truncated: null },
    { items: [], truncated: false, unavailable_count: -1 },
    { items: [], truncated: false, unavailable_count: 1.5 },
    { items: [], truncated: false, unavailable_count: "2" },
    { items: [], truncated: false, scan_limited: "true" },
    { items: [clue], truncated: true },
    { items: [{ ...clue, report_class: "formal" }], truncated: false },
    { items: [{ ...clue, category: "fact_conflict" }], truncated: false },
    { items: [{ ...clue, evidence: [clue.evidence[0]] }], truncated: false },
    { items: [{ ...clue, evidence: [clue.evidence[0], clue.evidence[0]] }], truncated: false },
    { items: [{ ...clue, evidence: [{ ...clue.evidence[0], line_end: 1 }, clue.evidence[1]] }], truncated: false },
    { items: [{ ...clue, metadata: { ...clue.metadata, final_outcome: "conflict" } }], truncated: false },
    { items: [clue, clue], truncated: false },
  ]) {
    assert.throws(() => normalizeReviewClueResult(payload), TypeError);
  }
});

test("review clues use the run scoped endpoint", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => { globalThis.fetch = originalFetch; });
  globalThis.fetch = async (url, init) => {
    assert.equal(String(url), "/api/v1/analysis-runs/run%20one/review-clues");
    assert.equal(init.credentials, "include");
    return new Response(JSON.stringify({ items: [clue], truncated: false }));
  };
  const result = await fetchReviewClues("run one");
  assert.equal(result.items.length, 1);
});
