import test from "node:test";
import assert from "node:assert/strict";
import {
  normalizeJourneyList, normalizeJourneyDetail, itemJourneysPath, itemJourneyPath,
  journeyRoute, journeySearch, journeyEventLabel, journeyKindNames,
} from "../src/features/item-journey/itemJourneyModel.ts";

const item = { key: "ij_copper", name: "铜钥匙", story_scope: "渡口篇", record_count: 2, issue_ids: [] };
const evidence = { document_id: "doc", document_name: "渡口.md", line_start: 2, line_end: 2, text: "午后，沈宁将铜钥匙交给许遥。" };
const event = { record_id: "r2", kind: "transfer", holder: "许遥", transfer_from: "沈宁", time: "午后", through_time: null,
  retention_scope: null, basis: "explicit", source_status: "verified", semantic_status: "supported", evidence, document_version: 1 };
function list() {
  return { schema_version: "item-journey-v1", project_id: "project", run_id: "run", items: [structuredClone(item)],
    page: { offset: 0, limit: 30, total: 1, has_more: false }, coverage: { scanned_records: 2, total_records: 2, truncated: false } };
}
function detail() {
  return { schema_version: "item-journey-v1", project_id: "project", run_id: "run", item: structuredClone(item),
    summary: { status: "ordered_evidence", text: "最后一条有据记录：午后，沈宁将铜钥匙交给许遥。", latest_record_ids: ["r2"] },
    // Deliberately not narrative order: the client must preserve source order.
    events: { items: [structuredClone(event), { ...structuredClone(event), record_id: "r1", kind: "custody", time: "清晨", holder: "沈宁", transfer_from: null }],
      offset: 0, limit: 50, total: 2, has_more: false },
    relations: [{ left_record_id: "r1", right_record_id: "r2", order: "before", evidence_record_ids: ["scope"] }],
    relation_evidence: [{ record_id: "scope", evidence: { ...evidence, line_start: 1, line_end: 1, text: "同一天，从清晨到午后。" }, document_version: 1 }],
    coverage: { processed_records: 2, total_records: 2, truncated: false, source_unavailable: 0, proof_unavailable: 0,
      unresolved_time_pairs: 0, run_partial: false, relation_proof_unavailable: 0 } };
}

test("read paths encode the exact run/item and optional issue; there is no generation or cutoff argument", () => {
  const url = new URL(itemJourneysPath("run / one", " 铜钥匙 ", 30, "issue / one"), "https://example.test");
  assert.equal(url.pathname, "/api/v1/analysis-runs/run%20%2F%20one/item-journeys");
  assert.deepEqual([...url.searchParams.entries()], [["q", "铜钥匙"], ["offset", "30"], ["limit", "30"], ["issue_id", "issue / one"]]);
  assert.equal(itemJourneyPath("run", "item / one", 50), "/api/v1/analysis-runs/run/item-journeys/item%20%2F%20one?offset=50&limit=50");
});

test("zero-issue catalogue keeps the server scope/grouping and real page totals", () => {
  const raw = list(); raw.items.push({ ...item, key: "ij_other", story_scope: "另一支线" });
  Object.assign(raw.page, { total: 62, has_more: true, offset: 30 });
  const value = normalizeJourneyList(raw, "project", "run", 30);
  assert.equal(value.items.length, 2); assert.equal(value.page.total, 62);
  assert.deepEqual(value.items.map(i => i.story_scope), ["渡口篇", "另一支线"]);
  assert.deepEqual(value.items[0].issue_ids, []);
});

test("project/run/item and pagination mismatches fail closed before display", () => {
  for (const mutate of [v => v.project_id = "other", v => v.run_id = "other", v => v.page.offset = 30, v => v.page.limit = 100]) {
    const raw = list(); mutate(raw); assert.throws(() => normalizeJourneyList(raw, "project", "run", 0), TypeError);
  }
  for (const mutate of [v => v.project_id = "other", v => v.run_id = "other", v => v.item.key = "other", v => v.events.offset = 50]) {
    const raw = detail(); mutate(raw); assert.throws(() => normalizeJourneyDetail(raw, "project", "run", item.key, 0), TypeError);
  }
});

test("summary, source ordering and relation evidence come from the server without reconstructing a chain", () => {
  const raw = detail(); const original = structuredClone(raw);
  const value = normalizeJourneyDetail(raw, "project", "run", item.key, 0);
  assert.deepEqual(value, original);
  assert.deepEqual(value.events.items.map(e => e.time), ["午后", "清晨"]);
  assert.deepEqual(value.summary.latest_record_ids, ["r2"]);
  assert.equal(value.relations[0].order, "before");
  assert.equal(value.relation_evidence[0].evidence.text, "同一天，从清晨到午后。");
});

test("source unavailable is nullable and independent of unresolved meaning or report coverage", () => {
  const raw = detail(); raw.events.items[0] = { ...event, evidence: null, document_version: null, source_status: "unavailable", semantic_status: "unverified", basis: "unverified" };
  raw.summary = { status: "partial", text: "尚不能确定先后", latest_record_ids: [] };
  Object.assign(raw.coverage, { source_unavailable: 1, proof_unavailable: 1, unresolved_time_pairs: 1, run_partial: true });
  const value = normalizeJourneyDetail(raw, "project", "run", item.key, 0);
  assert.equal(value.events.items[0].evidence, null);
  assert.equal(value.summary.status, "partial"); assert.equal(value.coverage.run_partial, true);
  assert.equal(value.events.items[1].semantic_status, "supported");
});

test("truncation preserves full total and does not fabricate a next page or latest holder", () => {
  const raw = detail(); raw.events = { ...raw.events, items: [], offset: 200, total: 231, has_more: false };
  Object.assign(raw.coverage, { processed_records: 200, total_records: 231, truncated: true });
  raw.summary = { status: "partial", text: "记录未整理完", latest_record_ids: [] };
  const value = normalizeJourneyDetail(raw, "project", "run", item.key, 200);
  assert.equal(value.events.total, 231); assert.equal(value.events.has_more, false);
  assert.deepEqual(value.summary.latest_record_ids, []);
});

test("observation, custody, use, transfer and retention remain distinct labels", () => {
  assert.equal(journeyEventLabel(event), "沈宁交给许遥");
  for (const [kind, label] of Object.entries({ observed_possession: "观察到持有", custody: "保管", uses: "使用", retained_custody: "明确未交接", legacy: "持有方式未区分" })) {
    assert.equal(journeyKindNames[kind], label);
    assert.equal(journeyEventLabel({ ...event, kind }), `许遥 · ${label}`);
  }
  const raw = detail(); raw.events.items[0] = { ...event, kind: "retained_custody", retention_scope: "ongoing", through_time: null };
  assert.equal(normalizeJourneyDetail(raw, "project", "run", item.key, 0).events.items[0].through_time, null);
});

test("invalid evidence coordinates, enum upgrades and negative coverage are rejected", () => {
  for (const mutate of [v => v.events.items[0].evidence.line_start = 0, v => v.events.items[0].document_version = 0,
    v => v.events.items[0].kind = "current_owner", v => v.events.items[0].retention_scope = "forever",
    v => v.events.items[0].semantic_status = "certain_owner", v => v.events.items[0].basis = "guessed",
    v => v.coverage.unresolved_time_pairs = -1, v => v.relations[0].order = "story_line_number",
    v => v.relation_evidence[0].evidence.line_end = 0, v => v.summary.status = "all_clear"]) {
    const raw = detail(); mutate(raw); assert.throws(() => normalizeJourneyDetail(raw, "project", "run", item.key, 0), TypeError);
  }
});

test("empty catalogues are valid data, while detail errors cannot be relabelled as empty success", () => {
  const raw = list(); raw.items = []; raw.page.total = 0;
  assert.equal(normalizeJourneyList(raw, "project", "run", 0).items.length, 0);
  for (const raw of [null, {}, { detail: "not found" }, { ...detail(), item: null }]) assert.throws(() => normalizeJourneyDetail(raw, "project", "run", item.key, 0), TypeError);
});

test("deep links preserve item/issue selection without treating a name as an identity", () => {
  assert.equal(journeySearch("ij_copper", "issue-1"), "?view=items&item=ij_copper&item_issue=issue-1");
  assert.deepEqual(journeyRoute(journeySearch("ij_copper", "issue-1")), { item: "ij_copper", issue: "issue-1" });
  assert.deepEqual(journeyRoute("?item=铜钥匙&item_issue=../../other"), { item: null, issue: null });
});
