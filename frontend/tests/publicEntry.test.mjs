import test from "node:test";
import assert from "node:assert/strict";
import { productRouteFromPath, safeReturnTo } from "../src/routing.ts";
import { guestWorkspaceViewFromPath, guestWorkspaceViews } from "../src/app/guestWorkspaceViews.ts";
import {
  canApplySessionProbe,
  PUBLIC_SESSION_PROBE_TIMEOUT_MS,
  publicEntryActions,
  publicEntryAuthHref,
  publicSessionMessage,
  shouldRecoverExpiredSession,
} from "../src/app/publicEntry.ts";

test("workspace entry is browsing while actual features preserve a destination without an automatic action", () => {
  assert.equal(publicEntryActions.workspace.kind, "navigation");
  assert.equal(publicEntryActions.workspace.returnTo, "/app");
  for (const [action, entry] of Object.entries(publicEntryActions)) {
    assert.equal(safeReturnTo(entry.returnTo), entry.returnTo, action);
    assert.ok(["projects", "workspace", "settings-model", "settings-account"].includes(productRouteFromPath(entry.returnTo).kind), action);
    assert.equal(new URL(entry.returnTo, "https://loreguard.local").search, "", action);
    assert.ok(entry.label, action);
    if (action !== "workspace") {
      assert.equal(entry.kind, "feature", action);
      assert.ok(entry.reason, action);
    }
  }
  assert.equal(publicEntryActions.create.returnTo, "/app");
  assert.equal(publicEntryActions.import.returnTo, "/app");
  assert.equal(publicEntryActions.model.returnTo, "/app/settings/model");
  assert.equal(publicEntryActions.account.returnTo, "/app/settings/account");
  assert.equal(publicEntryActions.sample.returnTo, "/app");
  assert.equal(publicEntryActions.check.returnTo, "/check");
});

test("both authentication modes carry only a safe internal return destination", () => {
  for (const mode of ["login", "register"]) {
    for (const entry of Object.values(publicEntryActions)) {
      if (entry.kind === "navigation") continue;
      const url = new URL(publicEntryAuthHref(mode, entry.returnTo), "https://loreguard.local");
      assert.equal(url.pathname, `/${mode}`);
      assert.equal(url.searchParams.get("returnTo"), entry.returnTo);
    }
    for (const invalid of ["https://evil.example/app", "//evil.example/app", "/\\evil.example/app", "/missing"]) {
      const url = new URL(publicEntryAuthHref(mode, invalid), "https://loreguard.local");
      assert.equal(url.searchParams.get("returnTo"), "/app");
    }
  }
});

test("checking, signed-out and failed states give distinct reading and recovery guidance", () => {
  const messages = ["checking", "signed-out", "failed"].map(publicSessionMessage);
  assert.equal(new Set(messages).size, 3);
  assert.match(publicSessionMessage("checking"), /不能使用个人功能.*可继续浏览空工作区/);
  assert.match(publicSessionMessage("signed-out"), /空工作区无需登录/);
  assert.match(publicSessionMessage("failed"), /仍可阅读本页.*登录/);
});

test("guest browsing allows only exact resource-free paths and keeps private deep links protected", () => {
  assert.deepEqual(guestWorkspaceViews.map((view) => view.path), [
    "/app", "/check", "/projects", "/characters", "/diff", "/revision", "/visual", "/audit", "/report",
  ]);
  for (const view of guestWorkspaceViews) {
    assert.equal(guestWorkspaceViewFromPath(view.path), view.id);
    assert.equal(guestWorkspaceViewFromPath(`${view.path}/`), view.id);
    assert.notEqual(productRouteFromPath(view.path).kind, "not-found");
  }
  for (const path of [
    "/app/projects/p-private/check", "/app/projects/p-private/runs/r-private/report",
    "/app/settings/model", "/app/settings/account", "/provider", "/check/extra", "/characters/p-private",
    "/login", "/register", "/", "//evil.example/check", "/%63heck",
  ]) assert.equal(guestWorkspaceViewFromPath(path), null, path);
});

test("guest view selection ignores resource-looking search and hash values", () => {
  const url = new URL("/report?projectId=p-private&runId=r-private&action=export#private", "https://loreguard.local");
  assert.equal(guestWorkspaceViewFromPath(url.pathname), "report");
  assert.equal(guestWorkspaceViewFromPath("/report?projectId=p-private"), null, "the selector accepts a pathname, never a resource query");
  const authUrl = new URL(publicEntryAuthHref("login", publicEntryActions.export.returnTo), url.origin);
  assert.equal(authUrl.searchParams.get("returnTo"), "/report", "feature entry does not forward guest resource-looking parameters");
});

test("session probe timeout is bounded and stale or aborted replies cannot overwrite a newer identity", () => {
  assert.equal(PUBLIC_SESSION_PROBE_TIMEOUT_MS, 4_000);
  assert.equal(canApplySessionProbe(3, 3, false), true);
  assert.equal(canApplySessionProbe(3, 4, false), false, "login, logout and expiration invalidate the old generation");
  assert.equal(canApplySessionProbe(3, 3, true), false, "timeout rejects a late reply even if transport ignores abort");
  assert.equal(canApplySessionProbe(3, 4, true), false);
});

test("only interruption of an active required account forces login recovery", () => {
  assert.equal(shouldRecoverExpiredSession("required", false), true);
  for (const mode of [null, undefined, "anonymous"]) {
    assert.equal(shouldRecoverExpiredSession(mode, false), false, "initial visits and auth/me 401 remain guest browsing");
  }
  for (const mode of [null, undefined, "anonymous", "required"]) {
    assert.equal(shouldRecoverExpiredSession(mode, true), false, "explicit logout uses existing draft cleanup and destination behavior");
  }
});
