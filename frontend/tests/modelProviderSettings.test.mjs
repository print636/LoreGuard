import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import {
  MODEL_TEST_DISCLAIMER,
  SAVED_API_KEY_MASK,
  canTestModelProvider,
  describeAccountProviderTest,
  describeAccountProviderConnection,
  emptyModelProviderForm,
  formFromModelProvider,
  modelProviderSavePayload,
  modelProviderDeleteSuccess,
  modelProviderDeleteWarning,
  modelProviderTestIsStale,
  parseAccountModelProvider,
  providerConflictMessage,
  validateModelProviderForm,
} from "../src/app/modelProviderSettings.ts";

const configuredPayload = {
  configured: true,
  revision: 4,
  api_key: { state: "set", masked: "server-value-must-not-render" },
  base_url: "https://api.example.com/v1",
  model: "story-model",
  updated_at: "2026-09-22T10:00:00Z",
  last_test: null,
};

test("unconfigured account metadata stays explicit without inventing a service default", () => {
  const profile = parseAccountModelProvider({
    configured: false,
    revision: 0,
    api_key: { state: "absent", masked: null },
    base_url: null,
    model: null,
    updated_at: null,
    last_test: null,
  });
  assert.equal(profile.configured, false);
  assert.equal(profile.serviceDefaultAvailable, false);
  assert.equal(profile.baseUrl, "");
  assert.equal(profile.model, "");
});

test("configured account uses a constant client mask and never trusts a returned mask", () => {
  const profile = parseAccountModelProvider(configuredPayload);
  assert.equal(profile.configured, true);
  assert.equal(SAVED_API_KEY_MASK, "••••••••••••");
  assert.doesNotMatch(JSON.stringify(profile), /server-value-must-not-render/);
});

test("successful reload clears the unsaved key and replacement state", () => {
  const clean = formFromModelProvider(parseAccountModelProvider(configuredPayload));
  assert.deepEqual(clean, {
    baseUrl: "https://api.example.com/v1",
    model: "story-model",
    apiKey: "",
    replacingKey: false,
  });
});

test("replacement sends a new key while ordinary metadata edits preserve the saved key", () => {
  const profile = parseAccountModelProvider(configuredPayload);
  const preserve = modelProviderSavePayload(
    { ...formFromModelProvider(profile), model: "story-model-2" },
    profile,
  );
  assert.equal(Object.hasOwn(preserve, "api_key"), false);

  const replace = modelProviderSavePayload(
    { ...formFromModelProvider(profile), replacingKey: true, apiKey: "new-secret" },
    profile,
  );
  assert.equal(replace.api_key, "new-secret");
  assert.equal(replace.expected_revision, 4);
});

test("connection testing is disabled for dirty or pending settings", () => {
  const profile = parseAccountModelProvider(configuredPayload);
  const clean = formFromModelProvider(profile);
  assert.equal(canTestModelProvider(profile, clean), true);
  assert.equal(canTestModelProvider(profile, { ...clean, model: "changed" }), false);
  assert.equal(canTestModelProvider(profile, { ...clean, replacingKey: true }), false);
  assert.equal(canTestModelProvider(profile, clean, true), false);
  assert.equal(canTestModelProvider(parseAccountModelProvider({
    configured: false, revision: 0, base_url: null, model: null, last_test: null,
  }), emptyModelProviderForm()), false);
});

test("review launch source distinguishes account, service default, and unknown availability", () => {
  const account = describeAccountProviderConnection(
    parseAccountModelProvider(configuredPayload),
  );
  assert.equal(account.configured, true);
  assert.match(account.label, /账户模型/);
  assert.match(account.detail, /不代表完整故事审查/);

  const service = describeAccountProviderConnection(parseAccountModelProvider({
    configured: false,
    revision: 2,
    base_url: null,
    model: null,
    last_test: null,
    service_default_available: true,
  }));
  assert.match(service.label, /服务默认/);

  const absent = describeAccountProviderConnection(parseAccountModelProvider({
    configured: false,
    revision: 0,
    base_url: null,
    model: null,
    last_test: null,
  }));
  assert.equal(absent.configured, false);
  assert.match(absent.label, /未设置/);
});

test("test results and recovery copy are allowlisted without upstream content", () => {
  const privateText = "sk-private https://internal.invalid raw response";
  const result = describeAccountProviderTest({
    category: "new-secret-category",
    tested_at: "2026-09-22T10:00:00Z",
    profile_revision: 4,
    raw_response: privateText,
    error: privateText,
    suggestions: [privateText],
  });
  assert.equal(result.category, "unknown");
  assert.doesNotMatch(JSON.stringify(result), /sk-private|internal\.invalid|raw response/);
  assert.match(MODEL_TEST_DISCLAIMER, /不代表.*完整分析/);
});

test("test results without a matching profile revision fail closed as stale", () => {
  const missingRevision = describeAccountProviderTest({ category: "success" });
  const matchingRevision = describeAccountProviderTest({
    category: "success",
    profile_revision: 4,
  });
  const olderRevision = describeAccountProviderTest({
    category: "success",
    profile_revision: 3,
  });

  assert.equal(modelProviderTestIsStale(missingRevision, 4), true);
  assert.equal(modelProviderTestIsStale(matchingRevision, 4), false);
  assert.equal(modelProviderTestIsStale(olderRevision, 4), true);
  assert.equal(modelProviderTestIsStale(null, 4), false);
});

test("every backend-safe provider test category has actionable Chinese copy", () => {
  const categories = [
    "success",
    "invalid_response",
    "unauthorized",
    "forbidden",
    "rate_limit",
    "upstream_5xx",
    "connect_timeout",
    "read_timeout",
    "transport",
    "response_too_large",
    "unsupported_content_encoding",
    "response_decompression",
    "nonretry_http",
    "body_json",
    "response_shape",
    "empty_content",
    "usage_shape",
    "truncated",
    "content_json",
    "endpoint_rejected",
    "credential_invalid",
    "credential_revoked",
    "credential_reflected",
    "provider",
  ];
  for (const category of categories) {
    const view = describeAccountProviderTest({ category });
    assert.equal(view.category, category);
    assert.notEqual(view.label, "无法识别连接测试结果");
    assert.ok(view.action.length > 0);
  }
});

test("client rejects a unicode API key before sending it", () => {
  const errors = validateModelProviderForm({
    baseUrl: "https://api.example.com/v1",
    model: "story-model",
    apiKey: "sk-不可回显",
    replacingKey: true,
  }, false);
  assert.match(errors.apiKey || "", /ASCII/);
});

test("revision conflicts and destructive deletion have explicit recovery copy", () => {
  assert.match(providerConflictMessage(409) || "", /重新加载/);
  assert.equal(providerConflictMessage(422), null);
  const requiredProfile = parseAccountModelProvider(configuredPayload);
  assert.doesNotMatch(modelProviderDeleteWarning(requiredProfile), /改用服务默认/);
  assert.match(modelProviderDeleteWarning(requiredProfile), /只执行确定性检查/);
  assert.match(modelProviderDeleteSuccess(requiredProfile), /只执行确定性检查/);

  const anonymousProfile = {
    ...requiredProfile,
    serviceDefaultAvailable: true,
  };
  assert.match(modelProviderDeleteWarning(anonymousProfile), /改用服务默认/);
  assert.match(modelProviderDeleteSuccess(anonymousProfile), /服务默认/);
});

test("model settings markup keeps labels, live status, alert, and keyboard deletion escape", () => {
  const componentPath = fileURLToPath(
    new URL("../src/app/ModelSettings.tsx", import.meta.url),
  );
  const source = readFileSync(componentPath, "utf8");
  assert.match(source, /<label htmlFor="model-baseUrl">Base URL<\/label>/);
  assert.match(source, /<label htmlFor="model-model">模型名称<\/label>/);
  assert.match(source, /role="status"/);
  assert.match(source, /role="alert"/);
  assert.match(source, /event\.key === "Escape"/);
  assert.match(source, /aria-label=\{keyVisible \? "隐藏 API Key" : "显示 API Key"\}/);
});
