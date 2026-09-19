import test from "node:test";
import assert from "node:assert/strict";

import {
  productRouteFromPath,
  reportRouteStateFromSearch,
  reportSearch,
  safeReturnTo,
  shouldResetProductScroll,
  workspacePath,
  workspaceProjectIdFromPath,
  workspaceRunIdFromPath,
  workspaceRouteFromPath,
  workspaceViewFromPath,
  workspaceViews,
} from "../src/routing.ts";

test("each workspace view has a stable deep-link path", () => {
  for (const view of workspaceViews) {
    assert.equal(workspacePath(view), `/${view}`);
    assert.equal(workspaceViewFromPath(`/${view}`), view);
  }
});

test("root and unknown paths fail closed to the check workspace", () => {
  assert.equal(workspaceViewFromPath("/"), "check");
  assert.equal(workspaceViewFromPath("/unknown"), "check");
  assert.equal(workspaceViewFromPath("/projects/extra"), "projects");
});

test("product routes distinguish authentication, project center, and workspaces", () => {
  assert.equal(productRouteFromPath("/login").kind, "login");
  assert.equal(productRouteFromPath("/register/").kind, "register");
  assert.equal(productRouteFromPath("/app").kind, "projects");
  assert.equal(productRouteFromPath("/app/projects/p-1/check").kind, "workspace");
  assert.deepEqual(productRouteFromPath("/app/projects/p-1/runs/r-2/report"), {
    kind: "workspace",
    projectId: "p-1",
    runId: "r-2",
  });
  assert.equal(productRouteFromPath("/app/settings/account").kind, "settings-account");
  assert.equal(productRouteFromPath("/missing").kind, "not-found");
});

test("nested project routes preserve project context across old workspace views", () => {
  assert.equal(workspaceProjectIdFromPath("/app/projects/story%201/check"), "story 1");
  assert.equal(workspaceViewFromPath("/app/projects/p-1/documents"), "projects");
  assert.equal(workspaceViewFromPath("/app/projects/p-1/compare"), "diff");
  assert.equal(workspacePath("report", "p-1"), "/app/projects/p-1/report");
  assert.equal(workspacePath("provider", "p-1"), "/app/settings/model");
  assert.equal(workspaceRunIdFromPath("/app/projects/p-1/runs/run%202/report"), "run 2");
  assert.equal(
    workspacePath("report", "p-1", "run 2"),
    "/app/projects/p-1/runs/run%202/report",
  );
  assert.equal(
    workspacePath("visual", "p-1", "run-2"),
    "/app/projects/p-1/runs/run-2/visuals",
  );
  assert.equal(
    workspacePath("audit", "p-1", "run-2"),
    "/app/projects/p-1/runs/run-2",
  );
});

test("same-view project routes remain distinct for browser back and forward", () => {
  assert.deepEqual(workspaceRouteFromPath("/app/projects/p-1/check"), {
    view: "check",
    projectId: "p-1",
    runId: null,
    search: "",
  });
  assert.deepEqual(workspaceRouteFromPath("/app/projects/p-2/check"), {
    view: "check",
    projectId: "p-2",
    runId: null,
    search: "",
  });
  assert.deepEqual(
    workspaceRouteFromPath(
      "/app/projects/p-2/runs/r-9/report",
      "?category=fact_conflict",
    ),
    {
      view: "report",
      projectId: "p-2",
      runId: "r-9",
      search: "?category=fact_conflict",
    },
  );
});

test("report route query round-trips whitelisted filters and issue selection", () => {
  const search = reportSearch({
    category: "fact_conflict",
    status: "false_positive",
    issueId: "issue-9",
  });
  assert.deepEqual(reportRouteStateFromSearch(search), {
    category: "fact_conflict",
    status: "false_positive",
    issueId: "issue-9",
  });
  assert.deepEqual(
    reportRouteStateFromSearch("?category=<script>&status=private&issue=" + "x".repeat(200)),
    { category: "all", status: "all", issueId: null },
  );
});

test("returnTo accepts only known same-origin application paths", () => {
  assert.equal(safeReturnTo("/app/projects/p-1/check?tab=run"), "/app/projects/p-1/check?tab=run");
  assert.equal(
    safeReturnTo(
      "/app/projects/p-1/runs/r-2/report?category=fact_conflict&status=all#issue",
    ),
    "/app/projects/p-1/runs/r-2/report?category=fact_conflict&status=all#issue",
  );
  assert.equal(safeReturnTo("https://evil.example/app"), "/app");
  assert.equal(safeReturnTo("//evil.example/app"), "/app");
  assert.equal(safeReturnTo("/unknown"), "/app");
});

test("leaving authentication resets scroll only when entering product work", () => {
  assert.equal(shouldResetProductScroll("register", "projects"), true);
  assert.equal(shouldResetProductScroll("login", "workspace"), true);
  assert.equal(shouldResetProductScroll("login", "settings-account"), true);
  assert.equal(shouldResetProductScroll("projects", "workspace"), false);
  assert.equal(shouldResetProductScroll("login", "register"), false);
});
