import test from "node:test";
import assert from "node:assert/strict";

import {
  createImportFilePlan,
  updateImportFileRole,
} from "../src/app/importPlan.ts";
import {
  backendTimestamp,
  projectNextAction,
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

test("multi-file import defaults safely to reference and preserves per-file roles", () => {
  const files = [{ name: "world.docx" }, { name: "chapter.docx" }];
  const initial = createImportFilePlan(files);
  assert.deepEqual(initial.map((entry) => entry.documentRole), ["reference", "reference"]);

  const updated = updateImportFileRole(initial, 0, "canon");
  assert.deepEqual(updated.map((entry) => entry.documentRole), ["canon", "reference"]);
  assert.deepEqual(initial.map((entry) => entry.documentRole), ["reference", "reference"]);
});

test("project center actions lead to the precise next view", () => {
  assert.deepEqual(
    projectNextAction({ id: "p-1", active_document_count: 0, latest_run: null }),
    { label: "导入第一份文稿", path: "/app/projects/p-1/documents" },
  );
  assert.deepEqual(
    projectNextAction({
      id: "p-1",
      active_document_count: 2,
      latest_run: { id: "r-1", status: "completed" },
    }),
    {
      label: "查看最近报告",
      path: "/app/projects/p-1/runs/r-1/report?category=all&status=all",
    },
  );
  assert.deepEqual(
    projectNextAction({
      id: "p-1",
      active_document_count: 2,
      latest_run: { id: "r-2", status: "running" },
    }),
    { label: "查看校验进度", path: "/app/projects/p-1/runs/r-2" },
  );
});
