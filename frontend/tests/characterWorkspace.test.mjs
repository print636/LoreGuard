import test from "node:test";
import assert from "node:assert/strict";

import {
  characterApiPaths,
  characterTraitAxisPath,
  characterTraitAxesPath,
  createCharacterTraitAxis,
  fetchCharacterTraitAxis,
  fetchCharacter,
  fetchCharacterTraitAxes,
  fetchCharacters,
  fetchDriftIssues,
  fetchProfileCandidates,
  fetchWithdrawnProfileCandidates,
  fetchProfileCandidate,
  fetchSourceNeighbors,
  normalizeCharacterDimension,
  normalizeProfileCandidate,
  setAxisPositiveProposition,
  submitAxisAlignment,
  submitCandidateDecision,
} from "../src/features/characters/api.ts";
import {
  appendAxisPage,
  canCreateNewAxis,
  previewAxisPolarity,
  selectedProjectAxis,
  validateAxisDraft,
  validateAxisPositiveProposition,
} from "../src/features/characters/axisReview.ts";
import {
  advanceReviewScope,
  isCurrentReviewRequest,
  reviewScopeKey,
} from "../src/features/characters/reviewScope.ts";
import {
  candidateDecisionLabel,
  candidateReviewState,
  candidateStatusNames,
  characterDriftReportPath,
  describeCoverage,
  describeReadiness,
} from "../src/features/characters/presentation.ts";
import { splitEvidenceBySupport } from "../src/features/characters/supportBindings.ts";
import {
  characterRouteStateFromSearch,
  characterSearch,
} from "../src/features/characters/routeState.ts";

function candidate(overrides = {}) {
  return {
    id: "candidate-1",
    character_id: "character-1",
    dimension: "core_personality",
    origin: "explicit_setting",
    statement: "在陌生人面前倾向寡言",
    confidence: 0.83,
    rationale: "多段历史台词保持相同表现。",
    limitations: [],
    scopes: [],
    supporting_evidence: [],
    contrary_evidence: [],
    status: "pending",
    reviewable: true,
    unreviewable_reason: null,
    source_run_id: "run-1",
    source_snapshot_revision: "snapshot-1",
    model_coverage: "full",
    support_bindings_status: "legacy",
    support_bindings_v1: null,
    revision: 3,
    ...overrides,
  };
}

function verifiedSupportCandidate(overrides = {}) {
  const line = "  🌙林澈按住剑。随后她退到门边。  ";
  const contextText = "🌙林澈按住剑。";
  const targetText = "随后她退到门边。";
  const contextStart = 2;
  const contextEnd = contextStart + Array.from(contextText).length;
  const targetEnd = contextEnd + Array.from(targetText).length;
  const evidence = {
    input_id: "input-1", document_id: "doc-1", document_name: "角色.md",
    document_version: 2, line_start: 1, line_end: 1, text: line,
    source_verified: true, source_text_exact: true,
  };
  const binding = {
    evidence_index: 0, support_id: "L1:A2",
    target: { support_id: "L1:A2", start_offset: contextEnd, end_offset: targetEnd, role: "target" },
    actor_anchor_id: "L1:A1", label_anchor_id: null,
    scope_relation: "same_actor_continuation",
    context: [{ support_id: "L1:A1", start_offset: contextStart, end_offset: contextEnd, role: "actor_anchor" }],
  };
  return {
    id: "candidate-bound", character_key: "林澈", trait_type: "core_personality",
    trait_key: "谨慎", value: "遇险时先保护同伴", origin: "explicit_setting",
    source_run_id: "run-1", scope_sha256: "snapshot", revision: 1,
    review_state: "pending", reviewable: true, model_coverage: "full",
    source_verified: true, evidence: [evidence],
    support_bindings_status: "verified",
    support_bindings_v1: {
      schema_version: "character-support-bindings-v1",
      index_version: "assertion-index-v1",
      bindings: [binding],
    },
    ...overrides,
  };
}

test("verified support renders the exact indented Unicode line with distinct context and target", () => {
  const raw = verifiedSupportCandidate();
  const parsed = normalizeProfileCandidate(raw, { requireExactEvidence: true });
  assert.equal(parsed.reviewable, true);
  assert.equal(parsed.support_bindings_status, "verified");
  assert.equal(parsed.supporting_evidence[0].text, raw.evidence[0].text);
  assert.deepEqual(
    splitEvidenceBySupport(parsed.supporting_evidence[0].text, parsed.support_bindings_v1.bindings),
    [
      { text: "  ", kind: "plain" },
      { text: "🌙林澈按住剑。", kind: "context" },
      { text: "随后她退到门边。", kind: "target" },
      { text: "  ", kind: "plain" },
    ],
  );
});

test("Unicode combining characters use codepoint offsets rather than UTF-16 indices", () => {
  const raw = verifiedSupportCandidate();
  const line = "e\u0301🌙林澈按住剑。随后她退到门边。";
  const contextEnd = Array.from("e\u0301🌙林澈按住剑。").length;
  const binding = {
    ...raw.support_bindings_v1.bindings[0],
    context: [{ ...raw.support_bindings_v1.bindings[0].context[0], start_offset: 0, end_offset: contextEnd }],
    target: {
      ...raw.support_bindings_v1.bindings[0].target,
      start_offset: contextEnd,
      end_offset: Array.from(line).length,
    },
  };
  const parsed = normalizeProfileCandidate({
    ...raw,
    evidence: [{ ...raw.evidence[0], text: line }],
    support_bindings_v1: { ...raw.support_bindings_v1, bindings: [binding] },
  }, { requireExactEvidence: true });
  assert.deepEqual(splitEvidenceBySupport(line, parsed.support_bindings_v1.bindings), [
    { text: "e\u0301🌙林澈按住剑。", kind: "context" },
    { text: "随后她退到门边。", kind: "target" },
  ]);
});

test("legacy candidates retain whole-line review while missing or damaged new bindings fail closed", () => {
  const base = verifiedSupportCandidate();
  const legacy = normalizeProfileCandidate({
    ...base, support_bindings_status: "legacy", support_bindings_v1: null,
  }, { requireExactEvidence: true });
  assert.equal(legacy.reviewable, true);
  assert.equal(legacy.support_bindings_v1, null);
  for (const malformed of [
    { support_bindings_status: undefined, support_bindings_v1: null },
    { support_bindings_status: "verified", support_bindings_v1: null },
    { support_bindings_status: "invalid", support_bindings_v1: null },
    { support_bindings_status: "legacy", support_bindings_v1: base.support_bindings_v1 },
    { support_bindings_v1: { ...base.support_bindings_v1, bindings: [{
      ...base.support_bindings_v1.bindings[0], evidence_index: 9,
    }] } },
    { support_bindings_v1: { ...base.support_bindings_v1, bindings: [{
      ...base.support_bindings_v1.bindings[0], target: {
        ...base.support_bindings_v1.bindings[0].target, end_offset: 999,
      },
    }] } },
    { support_bindings_v1: { ...base.support_bindings_v1, bindings: [{
      ...base.support_bindings_v1.bindings[0], context: [{
        ...base.support_bindings_v1.bindings[0].context[0], end_offset: 999,
      }],
    }] } },
    { support_bindings_v1: { ...base.support_bindings_v1, bindings: [{
      ...base.support_bindings_v1.bindings[0], context: [{
        ...base.support_bindings_v1.bindings[0].context[0], role: "bridge",
      }],
    }] } },
    { support_bindings_v1: { ...base.support_bindings_v1, bindings: [{
      ...base.support_bindings_v1.bindings[0], actor_anchor_id: null,
      scope_relation: "local",
    }] } },
    { evidence: [{ ...base.evidence[0], source_text_exact: false }] },
    { evidence: [{ ...base.evidence[0], input_id: null }] },
  ]) {
    const parsed = normalizeProfileCandidate({ ...base, ...malformed }, { requireExactEvidence: true });
    assert.equal(parsed.support_bindings_status, "invalid");
    assert.equal(parsed.support_bindings_v1, null);
    assert.equal(candidateReviewState(parsed).allowed, false);
    assert.match(candidateReviewState(parsed).label, /精确证据定位/);
  }
});

test("character route state round-trips linkable selection and filters", () => {
  const state = {
    characterId: "林澈",
    section: "candidates",
    candidateId: "candidate-4",
    page: 3,
    query: "林澈",
  };
  assert.deepEqual(characterRouteStateFromSearch(characterSearch(state)), state);
});

test("character route state rejects untrusted ids, sections and pages", () => {
  assert.deepEqual(
    characterRouteStateFromSearch(
      "?character=%3Cscript%3E&section=edit&candidate=" +
        "x".repeat(200) +
        "&page=-8&q=%00%20%20%E8%8B%8F%E5%BC%A6%20%20",
    ),
    {
      characterId: null,
      section: "profile",
      candidateId: null,
      page: 1,
      query: "苏弦",
    },
  );
  assert.equal(
    characterRouteStateFromSearch(
      "?section=drift&candidate=candidate-hidden",
    ).candidateId,
    null,
  );
});

test("candidate source-neighbor pages preserve distinct frozen-line groups and explicit pagination", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => { globalThis.fetch = originalFetch; });
  const sourceGroups = [1, 2].map((line) => ({
    input_id: "input-1", document_id: "document-1", document_name: "设定.md",
    document_version: 2, line_start: line, line_end: line, total: 1,
    context_verified: false,
  }));
  const neighbor = (id, group) => ({
    id, character_key: "林澈", source_run_id: "run-1", trait_type: "preference",
    trait_key: `归纳${id}`, value: "偏好", polarity: "positive", review_state: "pending",
    shared_evidence: [group],
  });
  const urls = [];
  globalThis.fetch = async (url) => {
    urls.push(String(url));
    const offset = Number(new URL(String(url), "http://local").searchParams.get("offset"));
    return new Response(JSON.stringify({
      candidate_id: "selected", character_key: "林澈", source_run_id: "run-1",
      limit: 1, offset, total: 2, has_more: offset === 0,
      source_groups: sourceGroups,
      items: [offset === 0 ? neighbor("one", sourceGroups[0]) : neighbor("two", sourceGroups[1])],
    }), { status: 200 });
  };
  const first = await fetchSourceNeighbors("project-1", "林澈", "selected", { limit: 1, offset: 0 });
  const second = await fetchSourceNeighbors("project-1", "林澈", "selected", { limit: 1, offset: 1 });
  assert.equal(first.total, 2);
  assert.equal(first.has_more, true);
  assert.equal(second.has_more, false);
  assert.equal(first.items[0].shared_evidence[0].line_start, 1);
  assert.equal(second.items[0].shared_evidence[0].line_start, 2);
  assert.equal(first.source_groups.length, 2);
  assert.match(urls[0], /source-neighbors\?limit=1&offset=0/);
  assert.match(urls[1], /source-neighbors\?limit=1&offset=1/);
});

test("candidate source-neighbor adapter rejects another character, line, and inconsistent pages", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => { globalThis.fetch = originalFetch; });
  const group = {
    input_id: "input-1", document_id: "document-1", document_name: "设定.md",
    document_version: 1, line_start: 1, line_end: 1, total: 1,
    context_verified: false,
  };
  const base = {
    candidate_id: "selected", character_key: "林澈", source_run_id: "run-1",
    limit: 20, offset: 0, total: 1, has_more: false, source_groups: [group],
    items: [{
      id: "neighbor", character_key: "林澈", source_run_id: "run-1",
      trait_type: "preference", trait_key: "偏好", value: "蜜瓜",
      polarity: "positive", review_state: "pending", shared_evidence: [group],
    }],
  };
  for (const malformed of [
    { ...base, character_key: "别的角色" },
    { ...base, items: [{ ...base.items[0], shared_evidence: [{ ...group, line_start: 2, line_end: 2 }] }] },
    { ...base, has_more: true },
    { ...base, items: [base.items[0], base.items[0]], total: 2 },
  ]) {
    globalThis.fetch = async () => new Response(JSON.stringify(malformed), { status: 200 });
    await assert.rejects(fetchSourceNeighbors("project-1", "林澈", "selected"), TypeError);
  }
});

test("candidate detail refuses confirmation when source verification is absent", () => {
  const raw = {
    id: "candidate-1", character_key: "林澈", trait_type: "preference",
    trait_key: "食物偏好", value: "喜欢蜜瓜", origin: "explicit_setting",
    source_run_id: "run-1", scope_sha256: "snapshot", revision: 0,
    review_state: "pending", reviewable: true, model_coverage: "full",
    evidence: [{ document_id: "doc-1", document_name: "设定.md", document_version: 1,
      line_start: 1, line_end: 1, text: "林澈喜欢蜜瓜。",
      document_role: "canon", authority_level: "core_canon" }],
  };
  const candidate = normalizeProfileCandidate(raw);
  assert.equal(candidate.source_verified, false);
  assert.equal(candidate.reviewable, false);
  assert.equal(candidate.supporting_evidence[0].context_verified, false);
  assert.equal(candidateReviewState(candidate).allowed, false);
});

test("decision replay is followed by a verified detail read, not treated as enriched evidence", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalDocument = globalThis.document;
  context.after(() => { globalThis.fetch = originalFetch; globalThis.document = originalDocument; });
  globalThis.document = { cookie: "loreguard_csrf=review-token" };
  const compact = {
    id: "candidate-1", character_key: "林澈", trait_type: "preference",
    trait_key: "食物偏好", value: "喜欢蜜瓜", origin: "explicit_setting",
    source_run_id: "run-1", scope_sha256: "snapshot", revision: 1,
    review_state: "confirmed", reviewable: false, model_coverage: "full",
    evidence: [{ document_id: "doc-1", document_name: "设定.md", document_version: 1,
      line_start: 1, line_end: 1, text: "林澈喜欢蜜瓜。" }],
  };
  let reads = 0;
  globalThis.fetch = async (_url, init) => {
    if (init?.method === "POST") {
      return new Response(JSON.stringify({ candidate: compact }), { status: 201 });
    }
    reads += 1;
    return new Response(JSON.stringify({
      ...compact, source_verified: true,
      evidence: [{ ...compact.evidence[0], input_id: "input-1", source_verified: true,
        source_text_exact: true, context_verified: false }],
    }), { status: 200 });
  };
  const mutation = await submitCandidateDecision("project-1", "林澈", "candidate-1", {
    decision: "confirm", comment: "", expected_revision: 0,
  });
  assert.equal(mutation.candidate.source_verified, false);
  const detail = await fetchProfileCandidate("project-1", "林澈", "candidate-1");
  assert.equal(reads, 1);
  assert.equal(detail.status, "confirmed");
  assert.equal(detail.source_verified, true);
  assert.equal(detail.supporting_evidence[0].source_text_exact, true);
});

test("coverage and readiness copy never turn degraded extraction into a clean result", () => {
  assert.match(describeCoverage("partial").detail, /可能存在遗漏/);
  assert.match(describeCoverage("rules_only").label, /没有完成 AI/);
  assert.match(describeCoverage("unknown").label, /未知/);
  assert.match(describeReadiness("not_generated").detail, /不代表资料中没有角色/);
  assert.equal(describeReadiness("ready"), null);
});

test("candidate decision progress names the action that is actually running", () => {
  assert.equal(candidateDecisionLabel("confirm", "confirm"), "正在确认…");
  assert.equal(candidateDecisionLabel("reject", "reject"), "正在驳回…");
  assert.equal(candidateDecisionLabel("confirm", "reject"), "确认归纳");
});

test("legacy character arrays stay compatible but an empty run is not called ready", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async () =>
    new Response(JSON.stringify([]), { status: 200 });

  const page = await fetchCharacters("project-1", { page: 1 });
  assert.equal(page.readiness, "not_generated");
  assert.equal(page.model_coverage, "unknown");
  assert.deepEqual(page.items, []);

  let envelopeUrl = "";
  globalThis.fetch = async (url) => {
    envelopeUrl = String(url);
    return new Response(
      JSON.stringify({
        items: [],
        page: 1,
        page_size: 30,
        total: 0,
        has_more: false,
        readiness: "not_generated",
        model_coverage: "partial",
        coverage_detail: "一个分块未完成归纳",
        source_run_id: "run-2",
      }),
      { status: 200 },
    );
  };
  const envelope = await fetchCharacters("project-1", {
    page: 1,
    query: " 林澈 ",
  });
  assert.match(envelopeUrl, /[?&]query=%E6%9E%97%E6%BE%88(?:&|$)/);
  assert.doesNotMatch(envelopeUrl, /[?&]q=/);
  assert.equal(envelope.readiness, "not_generated");
  assert.equal(envelope.model_coverage, "partial");
  assert.equal(envelope.source_run_id, "run-2");
});

test("character collection adapters reject malformed success payloads instead of showing clean empty states", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async (url) => {
    const path = String(url);
    if (path.includes("profile-candidates")) {
      return new Response(JSON.stringify({ total: 0, items: {} }), { status: 200 });
    }
    if (path.includes("drift-issues")) {
      return new Response(
        JSON.stringify({
          total: 1,
          has_more: false,
          items: [{ id: "issue-1", run_id: "run-1", title: "漂移" }],
        }),
        { status: 200 },
      );
    }
    return new Response(JSON.stringify({ unexpected: "contract-break" }), {
      status: 200,
    });
  };

  await assert.rejects(
    fetchCharacters("project-1", { page: 1 }),
    /items 数组/,
  );
  await assert.rejects(
    fetchProfileCandidates("project-1", "character-1", { page: 1 }),
    /items 数组/,
  );
  await assert.rejects(
    fetchDriftIssues("project-1", "character-1", { page: 1 }),
    /无法定位到当前角色/,
  );
});

test("stale and rules-only candidates fail closed while current candidates remain reviewable", () => {
  assert.equal(candidateReviewState(candidate()).allowed, true);
  assert.equal(
    candidateReviewState(candidate({ status: "stale" })).allowed,
    false,
  );
  assert.equal(
    candidateReviewState(candidate({ model_coverage: "rules_only" })).allowed,
    false,
  );
  assert.match(
    candidateReviewState(
      candidate({ reviewable: false, unreviewable_reason: "来源文档已更新" }),
    ).label,
    /来源文档已更新/,
  );
  assert.equal(
    candidateReviewState(candidate({ dimension: "unknown" })).allowed,
    false,
  );
  assert.equal(
    candidateReviewState(candidate({ origin: "unknown" })).allowed,
    false,
  );
});

test("frozen character dimensions include current state and fail unknown values closed", () => {
  assert.equal(normalizeCharacterDimension("current_state"), "current_state");
  assert.equal(normalizeCharacterDimension("future_dimension"), "unknown");

  const normalized = normalizeProfileCandidate({
    id: "candidate-raw",
    character_key: "林澈",
    trait_type: "future_dimension",
    origin: "explicit_setting",
    contexts: ["正式会议", "正式会议"],
    trait_key: "社交表现",
    value: "倾向寡言",
    confidence: 0.8,
    scope: { schema_version: 1, timeline_key: "main" },
    scope_sha256: "snapshot-raw",
    source_run_id: "run-raw",
    source_verified: true,
    support_bindings_status: "legacy",
    support_bindings_v1: null,
    review_state: "pending",
    revision: 0,
    evidence: [
      {
        document_id: "document-1",
        document_name: "第一章.md",
        document_version: 2,
        line_start: 3,
        line_end: 3,
        text: "林澈没有主动加入谈话。",
      },
    ],
  });
  assert.equal(normalized.dimension, "unknown");
  assert.equal(normalized.reviewable, false);
  assert.equal(normalized.statement, "社交表现：倾向寡言");
  assert.equal(normalized.origin, "explicit_setting");
  assert.deepEqual(normalized.contexts, ["正式会议"]);
  assert.match(normalized.unreviewable_reason, /未识别/);
});

test("character API and report paths encode opaque identifiers", () => {
  assert.deepEqual(characterApiPaths("project a", "角色/id", "candidate ?"), {
    root: "/api/v1/projects/project%20a/characters",
    character: "/api/v1/projects/project%20a/characters/%E8%A7%92%E8%89%B2%2Fid",
    candidates:
      "/api/v1/projects/project%20a/characters/%E8%A7%92%E8%89%B2%2Fid/profile-candidates",
    candidate:
      "/api/v1/projects/project%20a/characters/%E8%A7%92%E8%89%B2%2Fid/profile-candidates/candidate%20%3F",
    decisions:
      "/api/v1/projects/project%20a/characters/%E8%A7%92%E8%89%B2%2Fid/profile-candidates/candidate%20%3F/decisions",
    alignment:
      "/api/v1/projects/project%20a/characters/%E8%A7%92%E8%89%B2%2Fid/profile-candidates/candidate%20%3F/alignment",
    sourceNeighbors:
      "/api/v1/projects/project%20a/characters/%E8%A7%92%E8%89%B2%2Fid/profile-candidates/candidate%20%3F/source-neighbors",
    driftIssues: "/api/v1/projects/project%20a/drift-issues",
  });
  assert.equal(
    characterDriftReportPath("project a", "run/2", "issue-3"),
    "/app/projects/project%20a/runs/run%2F2/report?category=character_drift&status=all&issue=issue-3",
  );
});

test("candidate decisions send the expected revision with an idempotency key", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalDocument = globalThis.document;
  context.after(() => {
    globalThis.fetch = originalFetch;
    globalThis.document = originalDocument;
  });
  globalThis.document = { cookie: "loreguard_csrf=character-decision-token" };
  globalThis.fetch = async (url, init) => {
    assert.equal(
      url,
      "/api/v1/projects/project-1/characters/character-1/profile-candidates/candidate-1/decisions",
    );
    const headers = new Headers(init.headers);
    assert.ok(headers.get("Idempotency-Key"));
    assert.equal(headers.get("X-CSRF-Token"), "character-decision-token");
    assert.deepEqual(JSON.parse(init.body), {
      decision: "confirm",
      comment: "证据充分",
      expected_revision: 3,
    });
    return new Response(JSON.stringify({ candidate: candidate({ status: "confirmed" }) }), {
      status: 201,
    });
  };

  const result = await submitCandidateDecision(
    "project-1",
    "character-1",
    "candidate-1",
    { decision: "confirm", comment: "证据充分", expected_revision: 3 },
  );
  assert.equal(result.candidate.status, "confirmed");
});

test("withdrawal submits only the selected confirmed revision with CSRF and a stable retry key", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalDocument = globalThis.document;
  context.after(() => { globalThis.fetch = originalFetch; globalThis.document = originalDocument; });
  globalThis.document = { cookie: "loreguard_csrf=withdraw-token" };
  let calls = 0;
  let firstKey = "";
  globalThis.fetch = async (url, init) => {
    calls += 1;
    assert.equal(url, "/api/v1/projects/project-1/characters/%E6%9E%97%E6%BE%88/profile-candidates/trait-1/decisions");
    const headers = new Headers(init.headers);
    assert.equal(headers.get("X-CSRF-Token"), "withdraw-token");
    assert.ok(headers.get("Idempotency-Key"));
    if (firstKey) assert.equal(headers.get("Idempotency-Key"), firstKey);
    firstKey = headers.get("Idempotency-Key");
    assert.deepEqual(JSON.parse(init.body), {
      decision: "withdraw", comment: "", expected_revision: 3,
    });
    if (calls === 1) throw new TypeError("transport lost");
    return new Response(JSON.stringify({
      candidate: {
        id: "trait-1", character_key: "林澈", trait_type: "preference",
        trait_key: "食物偏好", value: "喜欢蜜瓜", review_state: "withdrawn",
        revision: 4, evidence: [],
      },
      deduplicated: true,
    }), { status: 201 });
  };
  const result = await submitCandidateDecision("project-1", "林澈", "trait-1", {
    decision: "withdraw", comment: "", expected_revision: 3,
  });
  assert.equal(calls, 2);
  assert.equal(result.candidate.status, "withdrawn");
  assert.equal(result.candidate.reviewable, false);
  assert.equal(result.deduplicated, true);
  assert.equal(candidateStatusNames[result.candidate.status], "已撤销");
});

test("retired source does not erase a confirmed trait, and withdrawn records are separately queryable", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => { globalThis.fetch = originalFetch; });
  const confirmed = {
    id: "trait-1", character_key: "林澈", trait_type: "preference",
    trait_key: "食物偏好", value: "喜欢蜜瓜", review_state: "confirmed",
    revision: 3, origin: "explicit_setting", source_run_id: "run-older",
    evidence: [{ document_id: "doc-1", document_name: "旧设定.md", document_version: 1,
      document_role: "reference", publication_status: "retired", line_start: 1,
      line_end: 1, text: "林澈喜欢蜜瓜。" }],
  };
  globalThis.fetch = async (url) => {
    if (String(url).includes("state=withdrawn")) {
      const parsed = new URL(String(url), "http://local.invalid");
      assert.equal(parsed.searchParams.get("status"), "withdrawn");
      assert.equal(parsed.searchParams.get("page"), "1");
      return new Response(JSON.stringify({
        items: [{ ...confirmed, id: "trait-0", review_state: "withdrawn", revision: 4 }],
        total: 1, page: 1, page_size: 20, has_more: false,
      }), { status: 200 });
    }
    return new Response(JSON.stringify({
      character_key: "林澈", confirmed_traits: [confirmed],
    }), { status: 200 });
  };
  const profile = await fetchCharacter("project-1", "林澈");
  assert.equal(profile.profile_items[0].id, "trait-1");
  assert.equal(profile.profile_items[0].revision, 3);
  assert.equal(profile.profile_items[0].statement, "食物偏好：喜欢蜜瓜");
  const withdrawn = await fetchWithdrawnProfileCandidates("project-1", "林澈", { page: 1 });
  assert.equal(withdrawn.items[0].status, "withdrawn");
  assert.equal(withdrawn.items[0].reviewable, false);
  globalThis.fetch = async () => new Response(JSON.stringify({
    state: "withdrawn", items: [confirmed], total: 1, page: 1,
    page_size: 20, has_more: false,
  }), { status: 200 });
  await assert.rejects(
    fetchWithdrawnProfileCandidates("project-1", "林澈", { page: 1 }),
    /仍在生效的特征/,
  );
});

test("author axis list is project scoped and rejects malformed or cross-project records", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => { globalThis.fetch = originalFetch; });
  const axis = {
    id: "axis-1",
    project_id: "project-1",
    trait_type: "core_personality",
    version: 1,
    display_name: "撤离路线协作",
    definition: "涉及同伴安全的撤离路线决策是否征询当值伙伴",
    definition_sha256: "a".repeat(64),
    created_at: "2026-09-24T00:00:00",
  };
  assert.equal(characterTraitAxesPath("project a"), "/api/v1/projects/project%20a/character-trait-axes");
  assert.equal(characterTraitAxisPath("project a", "axis/1"), "/api/v1/projects/project%20a/character-trait-axes/axis%2F1");
  globalThis.fetch = async (url) => {
    assert.equal(String(url), "/api/v1/projects/project-1/character-trait-axes?limit=100&offset=0");
    return new Response(JSON.stringify({ items: [axis], total: 2, limit: 100, offset: 0 }), { status: 200 });
  };
  const page = await fetchCharacterTraitAxes("project-1");
  assert.equal(page.items[0].display_name, axis.display_name);
  assert.equal(page.total, 2);
  assert.equal(selectedProjectAxis(page.items, "axis-1")?.version, 1);
  assert.equal(selectedProjectAxis(page.items, "axis-missing"), null);

  globalThis.fetch = async () => new Response(JSON.stringify({
    items: [{ ...axis, project_id: "another-project" }], total: 1, limit: 100, offset: 0,
  }), { status: 200 });
  await assert.rejects(fetchCharacterTraitAxes("project-1"), /不属于当前项目/);
});

test("author axis creation sends CSRF once and does not replay uncertain POST", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalDocument = globalThis.document;
  context.after(() => { globalThis.fetch = originalFetch; globalThis.document = originalDocument; });
  globalThis.document = { cookie: "loreguard_csrf=author-axis-token" };
  let calls = 0;
  globalThis.fetch = async (url, init) => {
    calls += 1;
    assert.equal(String(url), "/api/v1/projects/project-1/character-trait-axes");
    assert.equal(new Headers(init.headers).get("X-CSRF-Token"), "author-axis-token");
    assert.deepEqual(JSON.parse(init.body), {
      trait_type: "core_personality",
      display_name: "路线协作",
      definition: "危急决策是否征询同伴",
      positive_proposition: "危急决策前征询当值同伴",
    });
    throw new TypeError("connection lost");
  };
  await assert.rejects(createCharacterTraitAxis("project-1", {
    display_name: "路线协作", definition: "危急决策是否征询同伴",
    positive_proposition: "危急决策前征询当值同伴",
  }), /connection lost/);
  assert.equal(calls, 1);
});

test("axis draft validates on meaningful normalized content", () => {
  const empty = validateAxisDraft(" \n ", "\t", "  ");
  assert.match(empty.errors.display_name, /请输入/);
  assert.match(empty.errors.definition, /请说明/);
  assert.match(empty.errors.positive_proposition, /正向/);
  const valid = validateAxisDraft("  路线   协作  ", " 决策\n是否征询同伴 ", " 决策前\n征询当值同伴 ");
  assert.deepEqual(valid.value, {
    display_name: "路线 协作",
    definition: "决策 是否征询同伴",
    positive_proposition: "决策前 征询当值同伴",
  });
  assert.deepEqual(valid.errors, { display_name: "", definition: "", positive_proposition: "" });
  assert.match(validateAxisDraft("轴".repeat(81), "定义", "正向").errors.display_name, /80/);
  assert.match(validateAxisDraft("轴", "定".repeat(201), "正向").errors.definition, /200/);
  assert.match(validateAxisPositiveProposition("正".repeat(201)).error, /200/);
});

test("author-axis mapping keeps raw model direction distinct and refuses contradictory payloads", () => {
  assert.equal(previewAxisPolarity("positive", "same"), "positive");
  assert.equal(previewAxisPolarity("positive", "opposite"), "negative");
  assert.equal(previewAxisPolarity("negative", "opposite"), "positive");
  assert.equal(previewAxisPolarity("unclear", "same"), null);
  assert.equal(previewAxisPolarity("positive", "uncertain"), null);
  const base = verifiedSupportCandidate({
    polarity: "positive",
    approved_axis_id: "axis-1",
    approved_axis_version: 1,
  });
  const legacy = normalizeProfileCandidate(base);
  assert.equal(legacy.axis_alignment, null);
  assert.equal(legacy.axis_polarity, null);
  const mapped = normalizeProfileCandidate({
    ...base,
    axis_alignment: "opposite",
    axis_polarity: "negative",
    axis_positive_proposition_sha256: "b".repeat(64),
  });
  assert.equal(mapped.polarity, "positive");
  assert.equal(mapped.axis_polarity, "negative");
  for (const invalid of [
    { axis_alignment: "opposite", axis_polarity: "positive", axis_positive_proposition_sha256: "b".repeat(64) },
    { axis_alignment: "same", axis_polarity: null, axis_positive_proposition_sha256: "b".repeat(64) },
    { axis_alignment: "same", axis_polarity: "positive", axis_positive_proposition_sha256: "invalid" },
  ]) assert.throws(() => normalizeProfileCandidate({ ...base, ...invalid }), /方向映射/);
});

test("one-time axis proposition and legacy alignment use scoped API, CSRF, and review revision", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalDocument = globalThis.document;
  context.after(() => { globalThis.fetch = originalFetch; globalThis.document = originalDocument; });
  globalThis.document = { cookie: "loreguard_csrf=axis-review-token" };
  const axis = {
    id: "axis-1", project_id: "project-1", trait_type: "core_personality",
    version: 1, display_name: "签名自主", definition: "是否冒用他人签名",
    definition_sha256: "a".repeat(64), positive_proposition: "角色冒用他人签名",
    positive_proposition_sha256: "b".repeat(64), created_at: null,
  };
  const requests = [];
  globalThis.fetch = async (url, init) => {
    requests.push({ url: String(url), init });
    if (String(url).endsWith("/positive-proposition")) return new Response(JSON.stringify(axis));
    if (String(url).endsWith("/alignment")) {
      return new Response(JSON.stringify({ candidate: verifiedSupportCandidate({
        review_state: "confirmed", polarity: "positive", approved_axis_id: axis.id,
        approved_axis_version: 1, axis_alignment: "opposite", axis_polarity: "negative",
        axis_positive_proposition_sha256: axis.positive_proposition_sha256,
      }) }));
    }
    return new Response(JSON.stringify(axis));
  };
  assert.equal((await fetchCharacterTraitAxis("project-1", "axis-1")).positive_proposition, axis.positive_proposition);
  const updated = await setAxisPositiveProposition("project-1", "axis-1", {
    positive_proposition: axis.positive_proposition, expected_axis_version: 1,
  });
  assert.equal(updated.positive_proposition_sha256, axis.positive_proposition_sha256);
  const result = await submitAxisAlignment("project-1", "林澈", "candidate-bound", {
    expected_revision: 1, expected_axis_version: 1,
    expected_axis_positive_proposition_sha256: axis.positive_proposition_sha256,
    axis_alignment: "opposite", comment: "与冒签命题相反",
  });
  assert.equal(result.candidate.axis_polarity, "negative");
  assert.equal(requests.length, 3);
  for (const request of requests.slice(1)) {
    assert.equal(new Headers(request.init.headers).get("X-CSRF-Token"), "axis-review-token");
    assert.ok(new Headers(request.init.headers).get("Idempotency-Key"));
  }
  assert.equal(JSON.parse(requests[2].init.body).expected_revision, 1);
  assert.equal(JSON.parse(requests[2].init.body).axis_alignment, "opposite");
});

test("author-axis pagination fails closed on shifted duplicate pages or unreviewed local creation", () => {
  const axis = {
    id: "axis-1", project_id: "project-1", trait_type: "core_personality",
    version: 1, display_name: "路线协作", definition: "是否征询同伴",
    definition_sha256: "a".repeat(64), created_at: null,
  };
  assert.deepEqual(appendAxisPage([], { items: [axis], total: 2, limit: 1, offset: 0 }), [axis]);
  assert.throws(
    () => appendAxisPage([axis], { items: [axis], total: 2, limit: 1, offset: 1 }),
    /分页出现重复项/,
  );
  const full = { loaded: true, loading: false, error: "", dirty: false, fetchedCount: 2, total: 2 };
  assert.equal(canCreateNewAxis(full), true);
  assert.equal(canCreateNewAxis({ ...full, fetchedCount: 1 }), false);
  assert.equal(canCreateNewAxis({ ...full, dirty: true }), false);
  assert.equal(canCreateNewAxis({ ...full, error: "列表变化" }), false);
  assert.equal(canCreateNewAxis({ ...full, loaded: false }), false);
});

test("late decision success cannot overwrite another candidate or a returned old visit", async () => {
  const firstKey = reviewScopeKey("project-1", "character-1", "candidate-1");
  let current = { key: firstKey, generation: 0 };
  const started = current;
  let latestRequest = 1;
  const startedRequest = latestRequest;
  let resolveOld;
  const oldResponse = new Promise((resolve) => { resolveOld = resolve; });
  const view = { candidate: "candidate-1", announcement: "", busy: "confirm" };
  const completed = oldResponse.then((candidate) => {
    if (isCurrentReviewRequest(current, started, latestRequest, startedRequest)) {
      view.candidate = candidate;
      view.announcement = "已确认";
    }
  }).finally(() => {
    if (isCurrentReviewRequest(current, started, latestRequest, startedRequest)) view.busy = "";
  });

  current = advanceReviewScope(current, reviewScopeKey("project-1", "character-1", "candidate-2"));
  view.candidate = "candidate-2";
  view.busy = "";
  resolveOld("candidate-1-confirmed");
  await completed;
  assert.deepEqual(view, { candidate: "candidate-2", announcement: "", busy: "" });

  current = advanceReviewScope(current, firstKey);
  assert.equal(current.generation, 2);
  assert.equal(isCurrentReviewRequest(current, started, latestRequest, startedRequest), false);
  latestRequest += 1;
  assert.equal(isCurrentReviewRequest(current, current, latestRequest, startedRequest), false);
});

test("late axis creation error and finally cannot leak across projects sharing a candidate ID", async () => {
  const firstKey = reviewScopeKey("project-a", "character-1", "candidate-1");
  const secondKey = reviewScopeKey("project-b", "character-1", "candidate-1");
  assert.notEqual(firstKey, secondKey);
  let current = { key: firstKey, generation: 0 };
  const started = current;
  const request = 4;
  let rejectOld;
  const oldRequest = new Promise((_resolve, reject) => { rejectOld = reject; });
  const view = { axes: ["project-b-axis"], error: "", busy: true };
  const completed = oldRequest.catch(() => {
    if (isCurrentReviewRequest(current, started, request, request)) view.error = "旧项目错误";
  }).finally(() => {
    if (isCurrentReviewRequest(current, started, request, request)) view.busy = false;
  });

  current = advanceReviewScope(current, secondKey);
  rejectOld(new Error("old project failed"));
  await completed;
  assert.deepEqual(view, { axes: ["project-b-axis"], error: "", busy: true });
});

test("author-confirmed candidate keeps raw model label and axis identity separate", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalDocument = globalThis.document;
  context.after(() => { globalThis.fetch = originalFetch; globalThis.document = originalDocument; });
  globalThis.document = { cookie: "loreguard_csrf=axis-confirm-token" };
  const raw = {
    id: "candidate-1", character_key: "林澈", trait_type: "core_personality",
    trait_key: "consults_partner", value: "林澈总会征询伙伴。",
    polarity: "positive", authority_tier: "core_canon",
    origin: "explicit_setting", source_run_id: "run-1", scope_sha256: "snapshot-1",
    revision: 3, review_state: "confirmed", approved_axis_id: "axis-1",
    approved_axis_version: 1,
    evidence: [{ document_id: "doc-1", document_name: "设定.md", document_version: 1,
      line_start: 1, line_end: 1, text: "林澈总会征询伙伴。" }],
  };
  globalThis.fetch = async (_url, init) => {
    assert.deepEqual(JSON.parse(init.body), {
      decision: "confirm", comment: "证据充分", expected_revision: 3,
      approved_axis_id: "axis-1", expected_axis_version: 1,
    });
    return new Response(JSON.stringify({ candidate: raw }), { status: 201 });
  };
  const result = await submitCandidateDecision("project-1", "林澈", "candidate-1", {
    decision: "confirm", comment: "证据充分", expected_revision: 3,
    approved_axis_id: "axis-1", expected_axis_version: 1,
  });
  assert.equal(result.candidate.model_trait_key, "consults_partner");
  assert.equal(result.candidate.approved_axis_id, "axis-1");
  assert.equal(result.candidate.polarity, "positive");
  assert.equal(result.candidate.authority_tier, "core_canon");

  globalThis.fetch = async () => new Response(JSON.stringify({
    character_key: "林澈", confirmed_traits: [{ ...raw, approved_axis_id: null, approved_axis_version: null }],
  }), { status: 200 });
  const profile = await fetchCharacter("project-1", "林澈");
  assert.equal(profile.profile_items[0].approved_axis_id, null);
});
