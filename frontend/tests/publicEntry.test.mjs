import test from "node:test";
import assert from "node:assert/strict";
import { productRouteFromPath, safeReturnTo } from "../src/routing.ts";
import {
  canApplySessionProbe,
  PUBLIC_SESSION_PROBE_TIMEOUT_MS,
  publicEntryActions,
  publicEntryAuthHref,
  publicSessionMessage,
} from "../src/app/publicEntry.ts";

test("public entry destinations require authentication and preserve a page without an automatic action", () => {
  assert.deepEqual(Object.keys(publicEntryActions), ["workspace", "create", "import", "model"]);
  for (const [action, entry] of Object.entries(publicEntryActions)) {
    assert.equal(safeReturnTo(entry.returnTo), entry.returnTo, action);
    assert.ok(["projects", "settings-model"].includes(productRouteFromPath(entry.returnTo).kind), action);
    assert.equal(new URL(entry.returnTo, "https://loreguard.local").search, "", action);
    assert.ok(entry.label && entry.reason, action);
  }
  assert.equal(publicEntryActions.create.returnTo, "/app");
  assert.equal(publicEntryActions.import.returnTo, "/app");
  assert.equal(publicEntryActions.model.returnTo, "/app/settings/model");
});

test("both authentication modes carry only a safe internal return destination", () => {
  for (const mode of ["login", "register"]) {
    for (const entry of Object.values(publicEntryActions)) {
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
  assert.match(publicSessionMessage("checking"), /确认完成前不会打开工作区/);
  assert.match(publicSessionMessage("signed-out"), /本页无需登录/);
  assert.match(publicSessionMessage("failed"), /仍可阅读本页.*登录/);
});

test("session probe timeout is bounded and stale or aborted replies cannot overwrite a newer identity", () => {
  assert.equal(PUBLIC_SESSION_PROBE_TIMEOUT_MS, 4_000);
  assert.equal(canApplySessionProbe(3, 3, false), true);
  assert.equal(canApplySessionProbe(3, 4, false), false, "login, logout and expiration invalidate the old generation");
  assert.equal(canApplySessionProbe(3, 3, true), false, "timeout rejects a late reply even if transport ignores abort");
  assert.equal(canApplySessionProbe(3, 4, true), false);
});
