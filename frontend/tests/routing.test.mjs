import test from "node:test";
import assert from "node:assert/strict";

import {
  productRouteFromPath,
  safeReturnTo,
  shouldResetProductScroll,
  workspacePath,
  workspaceProjectIdFromPath,
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
  assert.equal(productRouteFromPath("/missing").kind, "not-found");
});

test("nested project routes preserve project context across old workspace views", () => {
  assert.equal(workspaceProjectIdFromPath("/app/projects/story%201/check"), "story 1");
  assert.equal(workspaceViewFromPath("/app/projects/p-1/documents"), "projects");
  assert.equal(workspaceViewFromPath("/app/projects/p-1/compare"), "diff");
  assert.equal(workspacePath("report", "p-1"), "/app/projects/p-1/report");
  assert.equal(workspacePath("provider", "p-1"), "/app/settings/model");
});

test("same-view project routes remain distinct for browser back and forward", () => {
  assert.deepEqual(workspaceRouteFromPath("/app/projects/p-1/check"), {
    view: "check",
    projectId: "p-1",
  });
  assert.deepEqual(workspaceRouteFromPath("/app/projects/p-2/check"), {
    view: "check",
    projectId: "p-2",
  });
});

test("returnTo accepts only known same-origin application paths", () => {
  assert.equal(safeReturnTo("/app/projects/p-1/check?tab=run"), "/app/projects/p-1/check?tab=run");
  assert.equal(safeReturnTo("https://evil.example/app"), "/app");
  assert.equal(safeReturnTo("//evil.example/app"), "/app");
  assert.equal(safeReturnTo("/unknown"), "/app");
});

test("leaving authentication resets scroll only when entering product work", () => {
  assert.equal(shouldResetProductScroll("register", "projects"), true);
  assert.equal(shouldResetProductScroll("login", "workspace"), true);
  assert.equal(shouldResetProductScroll("projects", "workspace"), false);
  assert.equal(shouldResetProductScroll("login", "register"), false);
});
