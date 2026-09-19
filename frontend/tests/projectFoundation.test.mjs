import test from "node:test";
import assert from "node:assert/strict";

import {
  createImportFilePlan,
  updateImportFileRole,
} from "../src/app/importPlan.ts";
import {
  backendTimestamp,
  relativeProjectDate,
} from "../src/app/projectPresentation.ts";

test("backend timestamps without a timezone are interpreted as UTC", () => {
  assert.equal(
    backendTimestamp("2026-09-19T10:00:00"),
    Date.parse("2026-09-19T10:00:00Z"),
  );
  assert.equal(
    relativeProjectDate(
      "2026-09-19T10:00:00",
      Date.parse("2026-09-19T10:05:00Z"),
    ),
    "5 分钟前",
  );
});

test("multi-file import defaults safely to chapter and preserves per-file roles", () => {
  const files = [{ name: "world.docx" }, { name: "chapter.docx" }];
  const initial = createImportFilePlan(files);
  assert.deepEqual(initial.map((entry) => entry.documentRole), ["chapter", "chapter"]);

  const updated = updateImportFileRole(initial, 0, "canon");
  assert.deepEqual(updated.map((entry) => entry.documentRole), ["canon", "chapter"]);
  assert.deepEqual(initial.map((entry) => entry.documentRole), ["chapter", "chapter"]);
});
