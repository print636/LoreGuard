import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { transformWithOxc } from "vite";
import { renderToStaticMarkup } from "react-dom/server";
import * as recovery from "../src/app/appRenderRecovery.ts";
import { renderStaticAppRenderFailure } from "../src/app/appRenderRecoveryDom.ts";

// Exercise the actual TSX class and static fallback without adding a DOM package.
// React server rendering is NOT claimed to validate catching browser render errors.
const compiled = (await transformWithOxc(
  readFileSync(new URL("../src/app/AppErrorBoundary.tsx", import.meta.url), "utf8"),
  "AppErrorBoundary.tsx",
  { jsx: { runtime: "automatic" } },
)).code
  .replace(/from (["'])(react(?:\/jsx-runtime)?)\1/g, (_match, _quote, name) => `from ${JSON.stringify(import.meta.resolve(name))}`)
  .replace(/from (["'])\.\/appRenderRecovery\1/g, () => `from ${JSON.stringify(new URL("../src/app/appRenderRecovery.ts", import.meta.url).href)}`);
const { default: AppErrorBoundary } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);

test("render boundary preserves children until a failure and never stores its exception", () => {
  const child = "synthetic child";
  const boundary = new AppErrorBoundary({ children: child });
  assert.equal(boundary.render(), child);
  const secretError = new Error("private-story-and-key-marker");
  const failed = AppErrorBoundary.getDerivedStateFromError(secretError);
  assert.deepEqual(Object.keys(failed), ["failed"]);
  assert.equal(failed.failed, true);
  boundary.state = failed;
  const fallback = renderToStaticMarkup(boundary.render());
  assert.ok(!fallback.includes(secretError.message));
  assert.ok(!fallback.includes(secretError.stack));
});

test("fixed fallback has semantic heading, warning and explicitly manual recovery", () => {
  const boundary = new AppErrorBoundary({ children: null });
  boundary.state = AppErrorBoundary.getDerivedStateFromError();
  const markup = renderToStaticMarkup(boundary.render());
  assert.match(markup, /<main[^>]*aria-labelledby="app-render-recovery-heading"/);
  assert.match(markup, /<h1[^>]*tabindex="-1"[^>]*>页面显示遇到问题<\/h1>/);
  assert.match(markup, /role="alert"/);
  assert.match(markup, /<button[^>]*type="button"[^>]*>重新加载页面<\/button>/);
  assert.ok(markup.includes("尚未保存的正文、备注或资料修改可能已丢失"));
  assert.ok(markup.includes("不会自动重新提交操作或启动模型分析"));
});

test("caught and recoverable React errors are discarded without inspecting arguments or logging", () => {
  const hostile = new Proxy({}, { get() { throw new Error("must not read exception fields"); } });
  let restored = 0;
  const handlers = recovery.safeReactRenderErrorHandlers(() => restored++);
  assert.doesNotThrow(() => handlers.onCaughtError(hostile, hostile));
  assert.doesNotThrow(() => handlers.onRecoverableError(hostile, hostile));
  assert.equal(restored, 0);
});

test("uncaught root errors still request a visible safe surface without reading exception contents", () => {
  const hostile = new Proxy({}, { get() { throw new Error("must not read exception fields"); } });
  let restored = 0;
  const handlers = recovery.safeReactRenderErrorHandlers(() => restored++);
  assert.doesNotThrow(() => handlers.onUncaughtError(hostile, hostile));
  assert.equal(restored, 1);
});

test("entrypoint installs safe React callbacks and wraps the complete application", () => {
  const entry = readFileSync(new URL("../src/main.tsx", import.meta.url), "utf8");
  assert.match(entry, /createRoot\([\s\S]*safeReactRenderErrorHandlers\(/);
  assert.match(entry, /<AppErrorBoundary>\s*<RootApp\s*\/>\s*<\/AppErrorBoundary>/);
  assert.match(entry, /renderStaticAppRenderFailure\(rootElement\)/);
  assert.doesNotMatch(entry, /root\.render\(<AppRenderFailure/);
  const boundary = readFileSync(new URL("../src/app/AppErrorBoundary.tsx", import.meta.url), "utf8");
  assert.match(boundary, /onClick=\{\(\) => window\.location\.reload\(\)\}/);
  assert.doesNotMatch(boundary, /console\.|fetch\(|setInterval\(|setTimeout\(|localStorage|sessionStorage/);
});

test("uncaught root recovery uses native fixed DOM, focuses its heading and never reloads automatically", () => {
  let reloads = 0;
  const nodes = [];
  const doc = {
    defaultView: { location: { reload: () => reloads++ } },
    createElement(tag) {
      const node = {
        tag, ownerDocument: doc, children: [], attributes: {}, listeners: {},
        append(child) { this.children.push(child); },
        setAttribute(name, value) { this.attributes[name] = value; },
        addEventListener(name, handler) { this.listeners[name] = handler; },
        focus(options) { this.focused = options; },
      };
      nodes.push(node);
      return node;
    },
  };
  const rootElement = {
    ownerDocument: doc,
    replaceChildren(child) { this.child = child; },
  };
  renderStaticAppRenderFailure(rootElement);
  assert.equal(rootElement.child.tag, "main");
  assert.equal(rootElement.child.attributes["aria-labelledby"], "app-render-recovery-heading");
  const heading = nodes.find((node) => node.tag === "h1");
  assert.equal(heading.textContent, recovery.appRenderRecoveryCopy.heading);
  assert.equal(heading.tabIndex, -1);
  assert.deepEqual(heading.focused, { preventScroll: true });
  assert.equal(nodes.find((node) => node.attributes.role === "alert").textContent, recovery.appRenderRecoveryCopy.explanation);
  const button = nodes.find((node) => node.tag === "button");
  assert.equal(button.type, "button");
  assert.equal(button.textContent, recovery.appRenderRecoveryCopy.reload);
  assert.equal(reloads, 0);
  button.listeners.click();
  assert.equal(reloads, 1);
});
