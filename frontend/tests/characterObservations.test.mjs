import test from "node:test";
import assert from "node:assert/strict";
import {
  normalizeObservationPage, observationDimension, observationPath, observationRecordingCopy,
  observationScope, startObservationRequest,
} from "../src/features/character-observations/observationModel.ts";

const request = { projectId: "project-private", runId: "run-private", offset: 0 };
const identity = { mode: "required", user: { id: "author-a" }, workspace: { id: "workspace-a" } };
const evidence = { document_id: "doc-private", document_name: "未发布的第二章", document_version: 3, line_start: 7, line_end: 8, text: "她把刀收回鞘中。\n她说：让他们走。" };
const baseline = { input_id: "frozen-baseline-private", payload_sha256: "a".repeat(64), payload: {
  character_display_name: "林霁", trait_type: "behavior_boundary", trait_key: "protect_life", value: "保护无辜者。",
  evidence: [{ ...evidence, document_name: "角色设定", document_version: 1, text: "她不伤害无辜者。" }],
  provenance: { raw_response: "must never render" }, candidate_id: "candidate-private", comparison_key: "behavior_boundary:private-key",
} };

function payload({ status = "complete", total = 1, offset = 0 } = {}) {
  return {
    schema_version: "character-observations-v1", project_id: request.projectId, run_id: request.runId,
    recording: { status, observation_count: total, source_manifest_sha256: "b".repeat(64) },
    page: { offset, limit: 50, total_items: total, has_more: offset + Math.min(50, Math.max(0, total - offset)) < total },
    items: Array.from({ length: Math.min(50, Math.max(0, total - offset)) }, (_, index) => ({
      ordinal: offset + index, observation_id: `cs_private_${index}`, character: "林霁", dimension: "behavior_boundary", trait_key: "protect_life",
      statement: "收刀并放走了对方。", context: "对方已经放下武器。", comparison_key: "behavior_boundary:private-key",
      source: { ...evidence, content_sha256: "c".repeat(64), run_input_id: "input-private" }, evidence: { ...evidence },
      matches: [{ status: "matched", reason_code: "matched", baseline: structuredClone(baseline) }],
    })),
  };
}

test("private run and exact pagination identity are checked before displaying observations", () => {
  const data = normalizeObservationPage(payload(), request);
  assert.equal(data.items[0].statement, "收刀并放走了对方。");
  assert.equal(data.items[0].source.version, 3);
  assert.equal(data.items[0].matches[0].baseline.evidence[0].version, 1);
  const mutations = [
    (value) => value.run_id = "other-run", (value) => value.project_id = "other-project",
    (value) => value.schema_version = "unrecognized-version", (value) => value.recording.status = "unknown-private-code",
    (value) => value.page.offset = 50, (value) => value.page.limit = 200,
    (value) => value.page.total_items = 2, (value) => value.page.has_more = true,
    (value) => value.items[0].source.document_id = "another-source",
    (value) => value.items[0].source.document_name = "当前草稿",
    (value) => value.items[0].source.document_version = 0,
    (value) => value.items[0].evidence.line_end = 1,
    (value) => value.items[0].matches[0].baseline = null,
    (value) => value.items[0].matches[0].status = "no_issue",
  ];
  for (const mutate of mutations) { const value = payload(); mutate(value); assert.throws(() => normalizeObservationPage(value, request), TypeError); }
});

test("author-facing projection excludes ids, hashes, comparison keys and arbitrary baseline payload", () => {
  const data = normalizeObservationPage(payload(), request);
  const visible = JSON.stringify(data);
  assert.doesNotMatch(visible, /private|must never render|sha256|comparison_key|candidate_id|observation_id|provenance|payload/);
  assert.match(visible, /保护无辜者/);
  assert.match(data.items[0].matches[0].explanation, /仅表示进入比较/);
  assert.match(data.items[0].matches[0].explanation, /不代表一致或存在冲突/);
});

test("unlinked ordinary observations receive a readable reason without becoming a review clue", () => {
  const value = payload();
  value.items[0].matches = [{ status: "not_matched", reason_code: "no_confirmed_baseline", baseline: null }];
  const item = normalizeObservationPage(value, request).items[0];
  assert.match(item.matches[0].explanation, /没有可用于比较的已确认角色设定/);
  assert.doesNotMatch(item.matches[0].explanation, /待复核|正式问题|no_confirmed_baseline/);
  value.items[0].matches[0].reason_code = "unknown-internal-reason";
  assert.doesNotMatch(normalizeObservationPage(value, request).items[0].matches[0].explanation, /unknown-internal-reason/);
  assert.equal(observationDimension("internal_dimension"), "角色表现");
});

test("known machine prefixes are removed from statements without changing the frozen quotation", () => {
  for (const statement of ["protect_life：保护无辜者。", "trait_key=protect_life；保护无辜者。", "protect_life: 保护无辜者。"]) {
    const value = payload(); value.items[0].statement = statement;
    const item = normalizeObservationPage(value, request).items[0];
    assert.equal(item.statement, "保护无辜者。"); assert.equal(item.source.text, evidence.text);
  }
  const value = payload(); value.items[0].statement = "trait_key=unexpected_private_key";
  assert.doesNotMatch(normalizeObservationPage(value, request).items[0].statement, /trait_key|unexpected_private_key/);
});

test("recording completeness, unavailable history, failure and complete empty are distinct", () => {
  const copies = ["in_progress", "complete", "partial", "failed", "not_applicable", "legacy_unavailable"].map((status) => {
    assert.equal(normalizeObservationPage(payload({ status, total: 0 }), request).status, status);
    return observationRecordingCopy(status, 0);
  });
  assert.equal(new Set(copies.map((copy) => copy.label)).size, 6);
  assert.match(copies[1].detail, /未保存到已接纳/);
  assert.match(copies[1].detail, /不能说明没有角色异常/);
  assert.match(copies[5].detail, /不等于零条记录或没有异常/);
  assert.match(copies[2].detail, /可能有遗漏/);
  assert.match(copies[3].detail, /过程失败/);
  assert.match(copies[0].detail, /尚未结束/);
});

test("recording coverage preserves bounded counts and only fixed author-facing exclusion reasons", () => {
  const value = payload({ status: "partial", total: 0 });
  value.recording.coverage = {
    baseline_not_evaluated_count: 2,
    excluded_baseline_counts: {
      author_alignment_required: 3, scoped_axis_review_disabled: 1, lower_authority_baseline_shadowed: 2,
      equivalent_axis_collapsed: 4, baseline_limit: 5, unknown_private_reason: 7,
    },
    omitted_baseline_count: 24,
    private_diagnostic: "must never render",
  };
  const data = normalizeObservationPage(value, request);
  assert.equal(data.coverage.omittedBaselineCount, 24);
  assert.equal(data.coverage.notEvaluatedCount, 2);
  assert.deepEqual(data.coverage.reasons, [
    { explanation: "特征方向需要作者确认", count: 3 },
    { explanation: "本次未开展限定适用范围的设定比较", count: 1 },
    { explanation: "比较时采用了优先级更高的设定", count: 2 },
    { explanation: "相同特征方向的设定已合并采用", count: 4 },
    { explanation: "已达到本次设定比较数量上限", count: 5 },
  ]);
  assert.doesNotMatch(JSON.stringify(data.coverage), /unknown_private_reason|must never render|author_alignment_required|scoped_axis_review_disabled/);
  assert.doesNotMatch(observationRecordingCopy(data.status, data.total).detail, /中途结束|达到处理上限/);
  assert.match(observationRecordingCopy(data.status, data.total).detail, /记录保存情况与实际审查范围应分别核对/);
  assert.equal(normalizeObservationPage(payload(), request).coverage, null);
});

test("bounded pages preserve frozen evidence and do not mix counts with issues or clues", () => {
  assert.equal(normalizeObservationPage(payload({ total: 53 }), request).hasMore, true);
  const last = normalizeObservationPage(payload({ total: 53, offset: 50 }), { ...request, offset: 50 });
  assert.equal(last.items.length, 3); assert.equal(last.hasMore, false); assert.equal(last.total, 53);
  assert.equal(normalizeObservationPage(payload({ total: 2, offset: 50 }), { ...request, offset: 50 }).items.length, 0);
  assert.equal(observationPath({ ...request, runId: "run/a", offset: 50 }), "/api/v1/analysis-runs/run%2Fa/character-observations?offset=50&limit=50");
  assert.throws(() => observationPath({ ...request, offset: -1 }), TypeError);
  const wrong = payload({ total: 2 }); wrong.items[1].ordinal = wrong.items[0].ordinal;
  assert.throws(() => normalizeObservationPage(wrong, request), TypeError);
  assert.doesNotMatch(JSON.stringify(last), /issue_count|clue_count/);
});

test("anonymous and blocked routes have no private scope; account, workspace and run changes invalidate it", () => {
  assert.equal(observationScope({ ...identity, mode: "anonymous" }, request.projectId, request.runId, true), null);
  assert.equal(observationScope(identity, request.projectId, request.runId, false), null);
  assert.equal(observationScope(identity, "", request.runId, true), null);
  const scope = observationScope(identity, request.projectId, request.runId, true);
  assert.notEqual(scope, observationScope({ ...identity, user: { id: "author-b" } }, request.projectId, request.runId, true));
  assert.notEqual(scope, observationScope({ ...identity, workspace: { id: "workspace-b" } }, request.projectId, request.runId, true));
  assert.notEqual(scope, observationScope(identity, request.projectId, "next-run", true));
});

const tick = () => new Promise((resolve) => setImmediate(resolve));

test("request cleanup aborts transport and prevents stale private success from refilling the panel", async () => {
  let resolve, signal; const states = [];
  const cancel = startObservationRequest(request, (_path, requestSignal) => { signal = requestSignal; return new Promise((done) => { resolve = done; }); }, (state) => states.push(state));
  await tick(); assert.equal(signal.aborted, false); cancel(); assert.equal(signal.aborted, true);
  resolve(payload()); await tick();
  assert.deepEqual(states, [{ status: "loading", data: null }]);
});

test("late failures are suppressed after a run switch; a current failure clears old data", async () => {
  let reject; const states = [];
  const cancel = startObservationRequest(request, () => new Promise((_resolve, fail) => { reject = fail; }), (state) => states.push(state));
  await tick(); cancel(); reject(new Error("private payload must not appear")); await tick();
  assert.equal(states.length, 1);
  startObservationRequest(request, async () => { throw new Error("private payload must not appear"); }, (state) => states.push(state));
  await tick(); assert.deepEqual(states.at(-1), { status: "error", data: null });
  assert.doesNotMatch(JSON.stringify(states), /private payload/);
});

test("a canceled preflight never sends a request, and current valid results become ready", async () => {
  let calls = 0; const states = [];
  const cancel = startObservationRequest(request, async () => { calls++; return payload(); }, (state) => states.push(state));
  cancel(); await tick(); assert.equal(calls, 0);
  startObservationRequest(request, async () => payload(), (state) => states.push(state));
  await tick(); assert.equal(states.at(-1).status, "ready"); assert.equal(states.at(-1).data.total, 1);
});

test("foreign or malformed results produce a safe retry state rather than private content", async () => {
  const states = [], value = payload(); value.run_id = "wrong-run";
  startObservationRequest(request, async () => value, (state) => states.push(state));
  await tick(); assert.deepEqual(states.at(-1), { status: "error", data: null });
});
