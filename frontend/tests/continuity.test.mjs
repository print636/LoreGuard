import test from "node:test";
import assert from "node:assert/strict";

import {
  dispatchFailureRecovery,
  resolveRunSelection,
  terminalRunPath,
} from "../src/runContinuity.ts";
import { ApiError } from "../src/api/client.ts";

const history = [
  { id: "latest", project_id: "p-1", status: "completed" },
  { id: "requested", project_id: "p-1", status: "failed" },
];

test("a precise run route restores that run instead of history[0]", () => {
  assert.deepEqual(resolveRunSelection(history, "p-1", "requested"), {
    kind: "selected",
    run: history[1],
    precise: true,
  });
});

test("a missing or cross-project run fails closed instead of selecting latest", () => {
  assert.deepEqual(resolveRunSelection(history, "p-1", "missing"), {
    kind: "not-found",
  });
  assert.deepEqual(
    resolveRunSelection(
      [{ id: "foreign", project_id: "p-2", status: "completed" }],
      "p-1",
      "foreign",
    ),
    { kind: "not-found" },
  );
});

test("terminal navigation opens completed reports and keeps failures on details", () => {
  assert.equal(
    terminalRunPath("completed", "p-1", "r-1"),
    "/app/projects/p-1/runs/r-1/report?category=all&status=all",
  );
  assert.equal(
    terminalRunPath("failed", "p-1", "r-1"),
    "/app/projects/p-1/runs/r-1",
  );
  assert.equal(
    terminalRunPath("cancelled", "p-1", "r-1"),
    "/app/projects/p-1/runs/r-1",
  );
});

test("a safe dispatch failure returns the durable run recovery target", () => {
  const error = new ApiError("safe message", {
    status: 503,
    statusText: "Service Unavailable",
    detail: {
      detail: {
        code: "analysis_dispatch_failed",
        message: "任务暂未进入执行队列，请稍后重试",
        run_id: "run-123",
        retryable: true,
      },
    },
  });
  assert.deepEqual(dispatchFailureRecovery(error), {
    runId: "run-123",
    message: "任务暂未进入执行队列，请稍后重试",
  });
  assert.equal(
    dispatchFailureRecovery(
      new ApiError("unsafe", {
        status: 503,
        statusText: "Service Unavailable",
        detail: { detail: { run_id: "../../secret", retryable: true } },
      }),
    ),
    null,
  );
});
