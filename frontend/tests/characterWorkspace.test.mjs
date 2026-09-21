import test from "node:test";
import assert from "node:assert/strict";

import {
  characterApiPaths,
  fetchCharacters,
  fetchDriftIssues,
  fetchProfileCandidates,
  normalizeCharacterDimension,
  normalizeProfileCandidate,
  submitCandidateDecision,
} from "../src/features/characters/api.ts";
import {
  candidateDecisionLabel,
  candidateReviewState,
  characterDriftReportPath,
  describeCoverage,
  describeReadiness,
} from "../src/features/characters/presentation.ts";
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
    revision: 3,
    ...overrides,
  };
}

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
