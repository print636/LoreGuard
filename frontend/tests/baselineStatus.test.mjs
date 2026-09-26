import test from "node:test";
import assert from "node:assert/strict";
import {
  hasReadyCharacterBaseline,
  normalizeCharacterBaselineStatus,
} from "../src/features/workflow/baselineStatus.ts";

function status(overrides = {}) {
  return {
    total_characters: 142,
    confirmed_trait_count: 3,
    pending_candidate_count: 206,
    baseline_run_id: "run-current",
    baseline_run_status: "completed",
    model_coverage: "full",
    coverage_detail: null,
    readiness: "ready",
    ...overrides,
  };
}

test("project-wide baseline counts do not inherit the first character page limit", () => {
  const parsed = normalizeCharacterBaselineStatus(status());
  assert.equal(parsed.total_characters, 142);
  assert.equal(parsed.pending_candidate_count, 206);
  assert.equal(hasReadyCharacterBaseline({ id: "run-current" }, parsed), true);
});

test("pending candidates warn but do not make confirmed traits authoritative by association", () => {
  assert.equal(hasReadyCharacterBaseline({ id: "run-current" }, status({ pending_candidate_count: 999 })), true);
  assert.equal(hasReadyCharacterBaseline({ id: "run-current" }, status({ confirmed_trait_count: 0 })), false);
});

test("stale, incomplete, rules-only or unavailable baseline never unlocks review", () => {
  assert.equal(hasReadyCharacterBaseline({ id: "run-other" }, status()), false);
  assert.equal(hasReadyCharacterBaseline({ id: "run-current" }, status({ model_coverage: "partial" })), false);
  assert.equal(hasReadyCharacterBaseline({ id: "run-current" }, status({ model_coverage: "rules_only" })), false);
  assert.equal(hasReadyCharacterBaseline({ id: "run-current" }, status({ readiness: "not_generated" })), false);
  assert.equal(hasReadyCharacterBaseline(null, status()), false);
});

test("malformed or contradictory baseline summaries fail closed", () => {
  for (const malformed of [
    status({ total_characters: -1 }),
    status({ confirmed_trait_count: "3" }),
    status({ pending_candidate_count: NaN }),
    status({ baseline_run_id: null }),
    status({ baseline_run_status: null }),
    status({ model_coverage: "unverified" }),
    status({ readiness: "complete" }),
  ]) {
    assert.throws(() => normalizeCharacterBaselineStatus(malformed), TypeError);
  }
  assert.equal(normalizeCharacterBaselineStatus(status({ model_coverage: "rules_only" })).model_coverage, "rules_only");
});
