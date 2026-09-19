import test from "node:test";
import assert from "node:assert/strict";

import {
  ApiError,
  SESSION_EXPIRED_EVENT,
  apiJson,
  apiUrl,
  createBoundedSessionProbe,
  probeCurrentSession,
  readCookie,
} from "../src/api/client.ts";

test("API paths stay relative by default and reject ambiguous paths", () => {
  assert.equal(apiUrl("/api/v1/projects"), "/api/v1/projects");
  assert.throws(() => apiUrl("api/v1/projects"), /must start/);
});

test("apiJson sends cookies and returns parsed JSON", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async (url, init) => {
    assert.equal(url, "/health");
    assert.equal(init.credentials, "include");
    return new Response(JSON.stringify({ status: "ok" }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  };

  assert.deepEqual(await apiJson("/health"), { status: "ok" });
});

test("apiJson exposes a bounded typed HTTP error", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async () =>
    new Response(JSON.stringify({ detail: "项目不存在" }), {
      status: 404,
      statusText: "Not Found",
    });

  await assert.rejects(
    apiJson("/api/v1/projects/missing"),
    (error) =>
      error instanceof ApiError &&
      error.status === 404 &&
      error.message === "项目不存在",
  );
});

test("cookie reader decodes the named value without exposing other cookies", () => {
  assert.equal(readCookie("loreguard_csrf", "other=private; loreguard_csrf=abc%20123"), "abc 123");
  assert.equal(readCookie("missing", "other=private"), null);
});

test("unsafe requests copy the readable CSRF cookie into the request header", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalDocument = globalThis.document;
  context.after(() => {
    globalThis.fetch = originalFetch;
    globalThis.document = originalDocument;
  });
  globalThis.document = { cookie: "loreguard_csrf=session-bound-token; unrelated=hidden" };
  globalThis.fetch = async (_url, init) => {
    const headers = new Headers(init.headers);
    assert.equal(init.credentials, "include");
    assert.equal(headers.get("X-CSRF-Token"), "session-bound-token");
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  };

  assert.deepEqual(
    await apiJson("/api/v1/projects", { method: "POST" }),
    { ok: true },
  );
});

test("an explicit CSRF header is never overwritten", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalDocument = globalThis.document;
  context.after(() => {
    globalThis.fetch = originalFetch;
    globalThis.document = originalDocument;
  });
  globalThis.document = { cookie: "loreguard_csrf=cookie-token" };
  globalThis.fetch = async (_url, init) => {
    assert.equal(new Headers(init.headers).get("X-CSRF-Token"), "caller-token");
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  };

  await apiJson("/api/v1/projects", {
    method: "DELETE",
    credentials: "omit",
    headers: { "X-CSRF-Token": "caller-token" },
  });
});

test("callers cannot disable credentialed session requests", async (context) => {
  const originalFetch = globalThis.fetch;
  context.after(() => {
    globalThis.fetch = originalFetch;
  });
  globalThis.fetch = async (_url, init) => {
    assert.equal(init.credentials, "include");
    return new Response(JSON.stringify({ ok: true }), { status: 200 });
  };

  await apiJson("/health", { credentials: "omit" });
});

test("204 responses resolve without pretending to contain JSON", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalDocument = globalThis.document;
  context.after(() => {
    globalThis.fetch = originalFetch;
    globalThis.document = originalDocument;
  });
  globalThis.document = { cookie: "loreguard_csrf=logout-token" };
  globalThis.fetch = async () => new Response(null, { status: 204 });

  assert.equal(
    await apiJson("/api/v1/auth/logout", { method: "POST" }),
    undefined,
  );
});

test("session probe turns auth 401 into the central expiry event", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalWindow = globalThis.window;
  const target = new EventTarget();
  let expiredEvents = 0;
  target.addEventListener(SESSION_EXPIRED_EVENT, () => { expiredEvents += 1; });
  context.after(() => {
    globalThis.fetch = originalFetch;
    globalThis.window = originalWindow;
  });
  globalThis.window = target;
  globalThis.fetch = async () => new Response(null, { status: 401 });

  assert.equal(await probeCurrentSession(), "expired");
  assert.equal(expiredEvents, 1);
});

test("authenticated account endpoints turn 401 into the central expiry event", async (context) => {
  const originalFetch = globalThis.fetch;
  const originalWindow = globalThis.window;
  const originalDocument = globalThis.document;
  const target = new EventTarget();
  let expiredEvents = 0;
  target.addEventListener(SESSION_EXPIRED_EVENT, () => { expiredEvents += 1; });
  context.after(() => {
    globalThis.fetch = originalFetch;
    globalThis.window = originalWindow;
    globalThis.document = originalDocument;
  });
  globalThis.window = target;
  globalThis.document = { cookie: "loreguard_csrf=expired-session-token" };
  globalThis.fetch = async () => new Response(null, { status: 401 });

  await assert.rejects(
    apiJson("/api/v1/auth/password", { method: "POST" }),
    (error) => error instanceof ApiError && error.status === 401,
  );
  assert.equal(expiredEvents, 1);
});

test("bounded session probe collapses parallel EventSource error checks", async () => {
  let calls = 0;
  let resolveProbe;
  const pending = new Promise((resolve) => { resolveProbe = resolve; });
  const probe = createBoundedSessionProbe(5_000, async () => {
    calls += 1;
    return pending;
  });

  const first = probe();
  assert.equal(await probe(), "skipped");
  assert.equal(calls, 1);
  resolveProbe("active");
  assert.equal(await first, "active");
});
