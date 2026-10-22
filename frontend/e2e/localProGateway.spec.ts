import { expect, test, type Page } from "@playwright/test";

// Synthetic account, provider and response only. No request reaches a gateway.
async function mockSettings(page: Page, enabled: boolean) {
  const origin = new URL(process.env.LOREGUARD_E2E_BASE_URL || "http://127.0.0.1:8080").origin;
  const writes: string[] = [];
  const unexpected: string[] = [];
  let profile: Record<string, unknown> = {
    configured: false, revision: 0, base_url: null, model: null, last_test: null,
    local_gateway: { enabled, allowed_origins: enabled ? ["http://127.0.0.1:8317"] : [] },
  };
  await page.route("**/*", async (route) => {
    const req = route.request();
    const url = new URL(req.url());
    if (url.origin === origin && req.method() === "GET" && (req.resourceType() === "document" || url.pathname.startsWith("/assets/"))) return route.continue();
    if (url.origin === origin && url.pathname === "/api/v1/auth/me" && req.method() === "GET") {
      return route.fulfill({ json: { mode: "required", user: { id: "gateway-mock", email: "gateway@example.test", display_name: "网关验收" }, workspace: { id: "gateway-workspace", name: "验收", kind: "personal", role: "owner" } } });
    }
    if (url.origin === origin && url.pathname === "/api/v1/account/model-provider") {
      if (req.method() === "PUT" && enabled) {
        const body = req.postDataJSON();
        expect(body).toEqual({ expected_revision: 0, base_url: "http://127.0.0.1:8317/v1", model: "synthetic-model", api_key: "synthetic-local-client-key" });
        writes.push("save");
        profile = { ...profile, configured: true, revision: 1, base_url: body.base_url, model: body.model };
        return route.fulfill({ json: profile });
      }
      if (req.method() === "GET") return route.fulfill({ json: profile });
    }
    if (url.origin === origin && enabled && url.pathname === "/api/v1/account/model-provider/test" && req.method() === "POST") {
      expect(req.postDataJSON()).toEqual({ expected_revision: 1 });
      writes.push("test");
      return route.fulfill({ json: { category: "success", profile_revision: 1, json_contract_ok: true, latency_ms: 120, token_usage: { prompt: 8, completion: 4, total: 12 } } });
    }
    unexpected.push(`${req.method()} ${url.origin}${url.pathname}`);
    return route.abort("blockedbyclient");
  });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/app/settings/model");
  await expect(page.getByRole("heading", { name: "模型与密钥", exact: true })).toBeVisible();
  await expect(page.getByLabel("Base URL", { exact: true })).toBeEnabled();
  return { writes, unexpected };
}

test("enabled local gateway saves through account form and tests without exposing a saved key", async ({ page }) => {
  await page.setViewportSize({ width: 375, height: 812 });
  const state = await mockSettings(page, true);
  await expect(page.locator("#model-baseUrl-help")).toContainText("http://127.0.0.1:8317");
  await page.getByLabel("Base URL", { exact: true }).fill("http://127.0.0.1:8317/v1");
  await page.getByLabel("模型名称", { exact: true }).fill("synthetic-model");
  await page.getByLabel("API Key", { exact: true }).fill("synthetic-local-client-key");
  await page.getByRole("button", { name: "保存模型配置", exact: true }).click();
  await expect(page.getByText("模型配置已保存。API Key 已从页面输入状态中清除。", { exact: true })).toBeVisible();
  await expect(page.locator("body")).not.toContainText("synthetic-local-client-key");
  await page.getByRole("button", { name: "测试已保存的模型连接", exact: true }).click();
  await expect(page.getByText("模型连接检查通过", { exact: true })).toBeVisible();
  expect(state.writes).toEqual(["save", "test"]);
  expect(state.unexpected).toEqual([]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(await page.evaluate(() => JSON.stringify({ ...localStorage, ...sessionStorage }))).not.toContain("synthetic-local-client-key");
  await page.screenshot({ path: "../artifacts/pro-gateway/model-settings-375.png", fullPage: true });
});

test("ordinary deployment still rejects a local HTTP URL before saving", async ({ page }) => {
  const state = await mockSettings(page, false);
  await page.getByLabel("Base URL", { exact: true }).fill("http://127.0.0.1:8317/v1");
  await page.getByLabel("模型名称", { exact: true }).fill("synthetic-model");
  await page.getByLabel("API Key", { exact: true }).fill("synthetic-local-client-key");
  await page.getByRole("button", { name: "保存模型配置", exact: true }).click();
  await expect(page.locator("#model-baseUrl-error")).toHaveText("Base URL 必须使用 HTTPS。");
  expect(state.writes).toEqual([]);
  expect(state.unexpected).toEqual([]);
});
