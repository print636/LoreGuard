import path from "node:path";
import { expect, test, type Locator, type Page } from "@playwright/test";

// Independent public-entry fixtures. Every API and nonlocal request is intercepted;
// no project creation, import, sample creation, analysis, or model call is permitted.
const screenshotFolder = process.env.LOREGUARD_PUBLIC_ENTRY_SCREENSHOT_DIR
  ?? "C:/Users/dell/Desktop/doc/creative-media/LoreGuard-公开信息页-v1";
const author = {
  mode: "required",
  user: { id: "public-entry-author", email: "entry@example.test", display_name: "入口验收作者" },
  workspace: { id: "public-entry-workspace", name: "入口验收工作区", kind: "personal", role: "owner" },
};

type Probe = "signed-out" | "failed" | "slow" | "required" | "anonymous";
type Options = { probe?: Probe; auth?: "login" | "register" };

function initialState(options: Options = {}) {
  let releaseProbe = () => {};
  const heldProbe = new Promise<void>((resolve) => { releaseProbe = resolve; });
  return {
    probe: options.probe ?? "signed-out",
    permittedAuth: options.auth,
    privateAccess: options.probe === "required" || options.probe === "anonymous",
    requests: [] as string[],
    authWrites: [] as string[],
    businessWrites: [] as string[],
    unexpected: [] as string[],
    heldProbe,
    releaseProbe,
  };
}
type MockState = ReturnType<typeof initialState>;

async function mockRequests(page: Page, state: MockState) {
  const baseOrigin = new URL(process.env.LOREGUARD_E2E_BASE_URL || "http://127.0.0.1:8080").origin;
  await page.route("**/*", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const endpoint = url.pathname;
    const method = request.method();
    if (!endpoint.startsWith("/api/")) {
      const staticDocument = request.resourceType() === "document"
        && (["/", "/login", "/register", "/app", "/check", "/projects", "/characters",
          "/diff", "/revision", "/visual", "/audit", "/report", "/provider"].includes(endpoint)
          || endpoint.startsWith("/app/"));
      const staticAsset = method === "GET" && endpoint.startsWith("/assets/");
      if (url.origin === baseOrigin && method === "GET" && (staticDocument || staticAsset)) {
        await route.continue();
      } else {
        state.unexpected.push(`${method} ${url.origin}${endpoint}`);
        await route.abort("blockedbyclient");
      }
      return;
    }

    state.requests.push(`${method} ${endpoint}${url.search}`);
    if (method === "GET" && endpoint === "/api/v1/auth/me") {
      if (state.probe === "failed") {
        await route.abort("failed");
        return;
      }
      if (state.probe === "slow") {
        await state.heldProbe;
        // The bounded session probe intentionally aborts this held request.
        await route.fulfill({ status: 401, json: { detail: "迟到的未登录探测" } }).catch(() => {});
        return;
      }
      if (state.probe === "required" || state.probe === "anonymous") {
        await route.fulfill({ json: { ...author, mode: state.probe } });
      } else {
        await route.fulfill({ status: 401, json: { detail: "请先登录" } });
      }
      return;
    }
    if (method === "POST" && endpoint === `/api/v1/auth/${state.permittedAuth}`) {
      state.authWrites.push(`${method} ${endpoint}`);
      state.privateAccess = true;
      await route.fulfill({ json: author });
      return;
    }
    if (method !== "GET") {
      state.businessWrites.push(`${method} ${endpoint}`);
      state.unexpected.push(`${method} ${endpoint}`);
      await route.fulfill({ status: 503, json: { detail: "Public-entry fixture rejects business writes" } });
      return;
    }
    if (state.privateAccess && endpoint === "/api/v1/project-catalog") {
      await route.fulfill({ json: { page: 1, page_size: 40, total: 0, items: [] } });
      return;
    }
    if (state.privateAccess && endpoint === "/api/v1/account/model-provider") {
      await route.fulfill({ json: { configured: false, revision: 0, service_default_available: false } });
      return;
    }
    state.unexpected.push(`${method} ${endpoint}`);
    await route.fulfill({ status: 404, json: { detail: "Unexpected public-entry fixture endpoint" } });
  });
}

async function openPage(page: Page, state: MockState, url = "/") {
  await mockRequests(page, state);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(url);
}

function reminder(page: Page) {
  return page.getByRole("dialog", { name: "登录后继续", exact: true });
}

async function expectPublicHome(page: Page) {
  await expect(page.getByRole("button", { name: "进入工作区", exact: true })).toBeVisible();
  await expect(page.getByRole("main")).toContainText("原创静态示例");
  expect(new URL(page.url()).pathname).toBe("/");
}

async function expectReadingOnly(page: Page, state: MockState, probeCount = 1) {
  await expectPublicHome(page);
  expect(state.requests).toEqual(Array.from({ length: probeCount }, () => "GET /api/v1/auth/me"));
  expect(state.authWrites).toEqual([]);
  expect(state.businessWrites).toEqual([]);
  expect(state.unexpected).toEqual([]);
}

async function expectNoBusinessWrite(page: Page, state: MockState) {
  await page.waitForLoadState("networkidle");
  expect(state.businessWrites).toEqual([]);
  expect(state.unexpected).toEqual([]);
}

function guestWorkspace(page: Page) {
  return page.getByRole("main", { name: "访客空工作区", exact: true });
}

async function expectGuestReadingOnly(page: Page, state: MockState, probeCount = 1) {
  await expect(guestWorkspace(page)).toBeVisible();
  await page.waitForLoadState("networkidle");
  expect(state.requests).toEqual(Array.from({ length: probeCount }, () => "GET /api/v1/auth/me"));
  expect(state.authWrites).toEqual([]);
  expect(state.businessWrites).toEqual([]);
  expect(state.unexpected).toEqual([]);
  const draftKeys = await page.evaluate(() => Object.keys(sessionStorage)
    .filter((key) => key.startsWith("loreguard:tab-draft:v1:")));
  expect(draftKeys).toEqual([]);
}

async function expectWithinViewport(page: Page, control: Locator) {
  await expect(control).toBeInViewport({ ratio: 1 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true);
}

async function submitAuth(page: Page, mode: "login" | "register") {
  if (mode === "register") await page.getByLabel("显示名称", { exact: true }).fill("入口验收作者");
  await page.getByLabel("邮箱", { exact: true }).fill("entry@example.test");
  await page.getByLabel("密码", { exact: true }).fill("public-entry-fixture-password");
  if (mode === "register") await page.getByLabel("确认密码", { exact: true }).fill("public-entry-fixture-password");
  await page.locator("#auth-form").getByRole("button", { name: mode === "login" ? "登录" : "创建账户", exact: true }).click();
}

test("401 首页公开可读，静态指南不读取项目也不写入", async ({ page }) => {
  const state = initialState();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openPage(page, state);
  await expectPublicHome(page);
  await page.waitForLoadState("networkidle");
  await page.screenshot({ path: path.join(screenshotFolder, "public-home-1440.png"), fullPage: false });
  const trigger = page.getByRole("button", { name: "阅读使用指南", exact: true });
  await trigger.click();
  const guide = page.getByRole("dialog", { name: "使用指南", exact: true });
  await expect(guide).toBeVisible();
  await expect(guide.locator('[id^="user-guide-topic-"]')).toHaveCount(6);
  await expectWithinViewport(page, guide.getByRole("button", { name: "关闭使用指南", exact: true }));
  await page.keyboard.press("Escape");
  await expect(guide).not.toBeVisible();
  await expect(trigger).toBeFocused();
  await expectReadingOnly(page, state);
});

test("慢探测不挡阅读，有限超时后登录且迟到探测不覆盖新身份", async ({ page }) => {
  const state = initialState({ probe: "slow", auth: "login" });
  await page.clock.install();
  await openPage(page, state);
  await expectPublicHome(page);
  await expect.poll(() => state.requests.length).toBe(1);
  const help = page.getByRole("button", { name: "阅读使用指南", exact: true });
  await help.click();
  await expect(page.getByRole("dialog", { name: "使用指南", exact: true })).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(help).toBeFocused();
  await page.getByRole("button", { name: "新建故事项目", exact: true }).click();
  const dialog = reminder(page);
  await expect(dialog).toContainText("正在确认登录状态");
  await expect(dialog.getByRole("button", { name: "去登录", exact: true })).toBeDisabled();
  await expect(dialog.getByRole("button", { name: "创建账户", exact: true })).toBeDisabled();
  await page.clock.fastForward(12_000);
  await expect(dialog).toContainText("暂时无法确认登录状态。");
  await dialog.getByRole("button", { name: "去登录", exact: true }).click();
  await expect(page.getByRole("heading", { name: "登录", exact: true })).toBeVisible();
  await submitAuth(page, "login");
  await expect(page).toHaveURL(/\/app$/);
  await expect(page.getByRole("heading", { name: "项目", exact: true })).toBeVisible();
  state.releaseProbe();
  await page.clock.resume();
  await expectNoBusinessWrite(page, state);
  await expect(page).toHaveURL(/\/app$/);
  await expect(page.getByRole("heading", { name: "项目", exact: true })).toBeVisible();
  expect(state.authWrites).toEqual(["POST /api/v1/auth/login"]);
});

test("探测失败的手机首页仍可读，登录页可返回公开首页", async ({ page }) => {
  const state = initialState({ probe: "failed" });
  await page.setViewportSize({ width: 375, height: 812 });
  await openPage(page, state);
  await expectPublicHome(page);
  await page.waitForLoadState("networkidle");
  await page.screenshot({ path: path.join(screenshotFolder, "public-home-375.png"), fullPage: false });
  await expectWithinViewport(page, page.getByRole("button", { name: "进入工作区", exact: true }));
  await page.getByRole("button", { name: "模型与密钥", exact: true }).click();
  const dialog = reminder(page);
  await expect(dialog).toContainText("暂时无法确认登录状态。");
  await expectWithinViewport(page, dialog.getByRole("button", { name: "继续浏览", exact: true }));
  await page.screenshot({ path: path.join(screenshotFolder, "login-reminder-375.png"), fullPage: false });
  await dialog.getByRole("button", { name: "去登录", exact: true }).click();
  await expect(page.getByRole("heading", { name: "登录", exact: true })).toBeVisible();
  await page.getByRole("button", { name: "返回公开首页", exact: true }).click();
  await expectReadingOnly(page, state);
});

test("需账户按钮仅弹提醒，取消和 Escape 回焦点并保持原页面", async ({ page }) => {
  const state = initialState();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openPage(page, state);
  await expectPublicHome(page);
  await page.waitForLoadState("networkidle");
  const trigger = page.getByRole("button", { name: "模型与密钥", exact: true });
  await trigger.focus();
  await trigger.press("Enter");
  const dialog = reminder(page);
  await expect(dialog).toBeVisible();
  const tabbable = dialog.locator("button:not([disabled]), a[href], [tabindex='0']");
  await tabbable.first().focus();
  await page.keyboard.press("Shift+Tab");
  await expect(tabbable.last()).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(tabbable.first()).toBeFocused();
  await page.screenshot({ path: path.join(screenshotFolder, "login-reminder-1440.png"), fullPage: false });
  await dialog.getByRole("button", { name: "继续浏览", exact: true }).click();
  await expect(dialog).not.toBeVisible();
  await expect(trigger).toBeFocused();
  for (const name of ["新建故事项目", "导入已有故事", "模型与密钥"]) {
    const feature = page.getByRole("button", { name, exact: true });
    await feature.click();
    await expect(dialog).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(dialog).not.toBeVisible();
    await expect(feature).toBeFocused();
  }
  await expectReadingOnly(page, state);
});

test("登录保留模型页目标，站外 returnTo 回退项目页且不发业务 POST", async ({ page }) => {
  const state = initialState({ auth: "login" });
  await openPage(page, state);
  await expectPublicHome(page);
  await page.waitForLoadState("networkidle");
  await page.getByRole("button", { name: "模型与密钥", exact: true }).click();
  await reminder(page).getByRole("button", { name: "去登录", exact: true }).click();
  await expect(page).toHaveURL(/\/login\?returnTo=%2Fapp%2Fsettings%2Fmodel$/);
  await submitAuth(page, "login");
  await expect(page).toHaveURL(/\/app\/settings\/model$/);
  await expect(page.getByRole("heading", { name: "模型与密钥", exact: true })).toBeVisible();
  await expectNoBusinessWrite(page, state);

  await page.goto(`/login?returnTo=${encodeURIComponent("https://outside.example.test/private")}`);
  await expect(page.getByRole("heading", { name: "登录", exact: true })).toBeVisible();
  await submitAuth(page, "login");
  await expect(page).toHaveURL(/\/app$/);
  await expectNoBusinessWrite(page, state);
  expect(state.authWrites).toEqual(["POST /api/v1/auth/login", "POST /api/v1/auth/login"]);
});

test("注册和登录切换保留目标，注册完成仅回跳而不执行导入", async ({ page }) => {
  const state = initialState({ auth: "register" });
  await openPage(page, state);
  await expectPublicHome(page);
  await page.waitForLoadState("networkidle");
  await page.getByRole("button", { name: "导入已有故事", exact: true }).click();
  await reminder(page).getByRole("button", { name: "创建账户", exact: true }).click();
  await expect(page).toHaveURL(/\/register\?returnTo=%2Fapp$/);
  await page.getByRole("button", { name: "已有账户？登录", exact: true }).click();
  await expect(page).toHaveURL(/\/login\?returnTo=%2Fapp$/);
  await page.getByRole("button", { name: "还没有账户？创建账户", exact: true }).click();
  await expect(page).toHaveURL(/\/register\?returnTo=%2Fapp$/);
  await submitAuth(page, "register");
  await expect(page).toHaveURL(/\/app$/);
  await expect(page.getByRole("heading", { name: "项目", exact: true })).toBeVisible();
  await expectNoBusinessWrite(page, state);
  expect(state.authWrites).toEqual(["POST /api/v1/auth/register"]);
});

test("私密深链仍先登录，合法内部 query 和 hash 被保留", async ({ page }) => {
  const state = initialState();
  const target = "/app/projects/public-entry-story/runs/public-entry-run/report?category=fact_conflict&status=unreviewed#issue-public-entry";
  await openPage(page, state, target);
  await expect(page.getByRole("heading", { name: "登录", exact: true })).toBeVisible();
  const url = new URL(page.url());
  expect(url.pathname).toBe("/login");
  expect(url.searchParams.get("returnTo")).toBe(target);
  expect(state.requests).toEqual(["GET /api/v1/auth/me"]);
  for (const settings of ["/app/settings/model", "/app/settings/account"]) {
    await page.goto(settings);
    await expect(page.getByRole("heading", { name: "登录", exact: true })).toBeVisible();
    expect(new URL(page.url()).searchParams.get("returnTo")).toBe(settings);
  }
  await page.getByRole("button", { name: "返回公开首页", exact: true }).click();
  await expectReadingOnly(page, state, 3);
  state.probe = "failed";
  await page.goto(target);
  await expect(page.getByRole("heading", { name: "暂时无法连接服务", exact: true })).toBeVisible();
  expect(state.requests).toEqual(Array.from({ length: 4 }, () => "GET /api/v1/auth/me"));
  await expectNoBusinessWrite(page, state);
});

test("已登录和显式本地 anonymous 的根入口兼容旧行为自动进入项目页", async ({ page }) => {
  const state = initialState({ probe: "required" });
  await openPage(page, state);
  for (const mode of ["required", "anonymous"] as const) {
    state.probe = mode;
    state.privateAccess = true;
    if (mode === "anonymous") await page.goto("/");
    await expect(page).toHaveURL(/\/app$/);
    await expect(page.getByRole("heading", { name: "项目", exact: true })).toBeVisible();
    await expect(page.getByRole("button", { name: "进入工作区", exact: true })).toHaveCount(0);
    await expectNoBusinessWrite(page, state);
  }
  expect(state.requests.filter((request) => request === "GET /api/v1/auth/me")).toHaveLength(2);
  expect(state.authWrites).toEqual([]);
});

test("首页直接进入访客空工作区，通用栏目和 query/hash 不读取私密资源", async ({ page }) => {
  const state = initialState();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openPage(page, state);
  await expectPublicHome(page);
  await page.waitForLoadState("networkidle");
  await page.getByRole("button", { name: "进入工作区", exact: true }).click();
  await expect(page).toHaveURL(/\/app$/);
  await expect(reminder(page)).toHaveCount(0);
  const guest = guestWorkspace(page);
  await expect(guest.getByRole("heading", { name: "项目", exact: true })).toBeVisible();
  await expect(guest.getByRole("heading", { name: "还没有项目", exact: true })).toBeVisible();
  await expectGuestReadingOnly(page, state);
  await page.screenshot({ path: path.join(screenshotFolder, "guest-workspace-1440.png"), fullPage: false });
  const navigation = page.getByRole("navigation", { name: "工作区", exact: true });
  await navigation.getByRole("button", { name: "文稿校验", exact: true }).click();
  await expect(page).toHaveURL(/\/check$/);
  await expect(guest.getByRole("heading", { name: "文稿校验台", exact: true })).toBeVisible();
  await navigation.getByRole("button", { name: "完整报告", exact: true }).click();
  await expect(page).toHaveURL(/\/report$/);
  await expectGuestReadingOnly(page, state);
  await page.goto("/report?projectId=not-a-resource&runId=not-a-run#issue-not-a-resource");
  await expectGuestReadingOnly(page, state, 2);
  await expectNoBusinessWrite(page, state);
});

test("慢探测和失败的手机空工作区可浏览，实际项目及校验功能才提示登录", async ({ page }) => {
  const state = initialState({ probe: "slow" });
  await page.setViewportSize({ width: 375, height: 812 });
  await page.clock.install();
  await openPage(page, state, "/app");
  const guest = guestWorkspace(page);
  await expect(guest).toBeVisible();
  await expect.poll(() => state.requests.length).toBe(1);
  const create = guest.getByRole("button", { name: "新建项目", exact: true });
  await create.click();
  await expect(reminder(page).getByRole("button", { name: "去登录", exact: true })).toBeDisabled();
  await page.keyboard.press("Escape");
  await expect(create).toBeFocused();
  await page.clock.fastForward(12_000);
  await expect(guest).toContainText("暂时无法确认登录状态。");
  state.releaseProbe();
  await page.clock.resume();
  await expectGuestReadingOnly(page, state);
  await expectWithinViewport(page, create);
  await page.evaluate(() => window.scrollTo({ top: 0, left: 0, behavior: "instant" }));
  await page.screenshot({ path: path.join(screenshotFolder, "guest-workspace-375.png"), fullPage: false });
  await guest.getByRole("heading", { name: "还没有项目", exact: true }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(screenshotFolder, "guest-workspace-empty-375.png"), fullPage: false });
  for (const name of ["新建项目", "导入已有故事", "新建空项目", "打开原创样例"]) {
    const feature = guest.getByRole("button", { name, exact: true });
    await feature.click();
    await expect(reminder(page)).toBeVisible();
    await expect(reminder(page).getByRole("button", { name: "去登录", exact: true })).toBeEnabled();
    await reminder(page).getByRole("button", { name: "继续浏览", exact: true }).click();
    await expect(feature).toBeFocused();
    await expectGuestReadingOnly(page, state);
  }
  await page.getByRole("navigation", { name: "工作区", exact: true })
    .getByRole("button", { name: "文稿校验", exact: true }).click();
  await expect(page).toHaveURL(/\/check$/);
  for (const name of ["输入文稿", "选择文稿文件", "开始校验"]) {
    const feature = guest.getByRole("button", { name, exact: true });
    await feature.click();
    await expect(reminder(page)).toBeVisible();
    await page.keyboard.press("Escape");
    await expect(feature).toBeFocused();
    await expectGuestReadingOnly(page, state);
  }
  await page.getByRole("button", { name: "账户安全", exact: true }).click();
  await reminder(page).getByRole("button", { name: "去登录", exact: true }).click();
  await expect(page).toHaveURL(/\/login\?returnTo=%2Fapp%2Fsettings%2Faccount$/);
  await expect(page.getByRole("heading", { name: "登录", exact: true })).toBeVisible();
  await expectNoBusinessWrite(page, state);
  expect(state.requests).toEqual(["GET /api/v1/auth/me"]);
  expect(state.authWrites).toEqual([]);
});
