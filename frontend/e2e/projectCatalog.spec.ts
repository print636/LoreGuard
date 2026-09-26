import { expect, test, type Page } from "@playwright/test";

const PROJECT_COUNT = 2191;
const projectId = (index: number) =>
  `00000000-0000-4000-8000-${String(index).padStart(12, "0")}`;
const projects = Array.from({ length: PROJECT_COUNT }, (_, offset) => {
  const index = offset + 1;
  return {
    id: projectId(index),
    name: `项目 ${String(index).padStart(4, "0")}`,
    description: `故事资料 ${index}`,
    created_at: new Date(Date.UTC(2026, 8, 1, 0, 0, index)).toISOString(),
    active_document_count: index % 3,
    latest_run: null,
  };
});

async function mockCatalog(page: Page) {
  const requests: URL[] = [];
  let legacyListRequests = 0;
  await page.route("**/api/v1/auth/me", (route) => route.fulfill({
    json: {
      mode: "anonymous",
      user: { id: "user-1", email: "", display_name: "测试者" },
      workspace: { id: "workspace-1", name: "测试工作区", kind: "personal", role: "owner" },
    },
  }));
  await page.route("**/api/v1/account/model-provider", (route) => route.fulfill({ json: {} }));
  await page.route("**/api/v1/projects", (route) => {
    if (route.request().method() === "GET") legacyListRequests += 1;
    return route.fulfill({ status: 500, json: { detail: "Legacy list must not be called" } });
  });
  await page.route("**/api/v1/projects/*/documents?*", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/v1/projects/*/analysis-runs", (route) => route.fulfill({ json: [] }));
  await page.route("**/api/v1/project-catalog?*", async (route) => {
    const url = new URL(route.request().url());
    requests.push(url);
    const query = (url.searchParams.get("query") || "").trim().toLocaleLowerCase();
    const exactId = url.searchParams.get("project_id");
    const sort = url.searchParams.get("sort") || "recent";
    const pageNumber = Number(url.searchParams.get("page") || "1");
    const pageSize = Number(url.searchParams.get("page_size") || "40");
    const matching = projects
      .filter((item) => (!exactId || item.id === exactId) &&
        (!query || `${item.name} ${item.description}`.toLocaleLowerCase().includes(query)))
      .sort((a, b) => sort === "name"
        ? a.name.localeCompare(b.name, "zh-CN") || a.id.localeCompare(b.id)
        : b.created_at.localeCompare(a.created_at) || b.id.localeCompare(a.id));
    if (query === "项目 0219") await new Promise((resolve) => setTimeout(resolve, 700));
    await route.fulfill({ json: {
      page: pageNumber,
      page_size: pageSize,
      total: matching.length,
      items: matching.slice((pageNumber - 1) * pageSize, pageNumber * pageSize),
    } });
  });
  return { requests, legacyListRequests: () => legacyListRequests };
}

test("项目中心分页、服务端搜索和过期响应不覆盖新结果", async ({ page }) => {
  const mock = await mockCatalog(page);
  await page.goto("/app");
  const list = page.locator(".projectList li");
  await expect(list).toHaveCount(40);
  await expect(page.getByRole("navigation", { name: "项目分页" })).toContainText("共 2191 项");
  await page.getByRole("navigation", { name: "项目分页" }).getByRole("button", { name: "下一页" }).click();
  await expect(list).toHaveCount(40);
  await expect(page.getByRole("navigation", { name: "项目分页" })).toContainText("第 2 / 55 页");
  expect(mock.requests.some((url) => url.searchParams.get("page") === "2" && url.searchParams.get("page_size") === "40")).toBe(true);
  const centerNext = page.getByRole("navigation", { name: "项目分页" }).getByRole("button", { name: "下一页" });
  await centerNext.focus();
  await centerNext.press("Enter");
  await expect(page.getByRole("navigation", { name: "项目分页" })).toContainText("第 3 / 55 页");
  await expect(centerNext).toBeFocused();

  const search = page.getByRole("searchbox", { name: "搜索项目" });
  const oldRequest = page.waitForRequest((request) => {
    const url = new URL(request.url());
    return url.pathname === "/api/v1/project-catalog" && url.searchParams.get("query") === "项目 0219";
  });
  await search.fill("项目 0219");
  await oldRequest;
  await search.fill("项目 2191");
  await expect(list).toHaveCount(1);
  await expect(list.first()).toContainText("项目 2191");
  await page.waitForTimeout(750);
  await expect(list.first()).toContainText("项目 2191");
  await expect(page.getByRole("navigation", { name: "项目分页" })).toContainText("共 1 项");

  await search.fill("不存在的项目");
  await expect(page.locator(".filteredEmpty")).toContainText("没有匹配“不存在的项目”的项目");
  await page.locator(".filteredEmpty").getByRole("button", { name: "清除搜索" }).click();
  await expect(list).toHaveCount(40);
  await page.getByRole("combobox", { name: "项目排序" }).selectOption("name");
  await expect(list).toHaveCount(40);
  expect(mock.requests.some((url) => url.searchParams.get("sort") === "name")).toBe(true);
  await page.setViewportSize({ width: 375, height: 812 });
  await expect(page.getByRole("navigation", { name: "项目分页" }).getByRole("button", { name: "下一页" })).toBeVisible();
  expect(mock.legacyListRequests()).toBe(0);
});

test("项目中心的 URL 在打开项目、浏览器返回和刷新后恢复目录状态", async ({ page }) => {
  const mock = await mockCatalog(page);
  await page.goto("/app");
  const pagination = page.getByRole("navigation", { name: "项目分页" });
  await pagination.getByRole("button", { name: "下一页" }).click();
  await expect(pagination).toContainText("第 2 / 55 页");
  await expect(page).toHaveURL(/\/app\?page=2$/);
  await expect(page.locator(".projectList li").first()).toContainText("项目 2151");
  await page.locator(".projectList li a").first().click();
  await expect(page).toHaveURL(new RegExp(`/app/projects/${projectId(2151)}/documents$`));
  await page.goBack();
  await expect(page).toHaveURL(/\/app\?page=2$/);
  await expect(pagination).toContainText("第 2 / 55 页");
  await expect(page.locator(".projectList li").first()).toContainText("项目 2151");

  await page.getByRole("searchbox", { name: "搜索项目" }).fill("项目 0042");
  await expect(page.locator(".projectList li")).toHaveCount(1);
  await page.getByRole("combobox", { name: "项目排序" }).selectOption("name");
  await expect(page).toHaveURL(/\/app\?query=.*&sort=name$/);
  await page.locator(".projectList li a").first().click();
  await page.goBack();
  await expect(page.getByRole("searchbox", { name: "搜索项目" })).toHaveValue("项目 0042");
  await expect(page.getByRole("combobox", { name: "项目排序" })).toHaveValue("name");
  await expect(page.locator(".projectList li")).toHaveCount(1);
  await page.reload();
  await expect(page.getByRole("searchbox", { name: "搜索项目" })).toHaveValue("项目 0042");
  await expect(page.getByRole("combobox", { name: "项目排序" })).toHaveValue("name");
  await expect(page.locator(".projectList li")).toHaveCount(1);
  expect(mock.legacyListRequests()).toBe(0);
});

test("项目目录请求失败后可重试当前页", async ({ page }) => {
  const mock = await mockCatalog(page);
  let allowSuccess = false;
  await page.route("**/api/v1/project-catalog?*", async (route) => {
    if (!allowSuccess) {
      await route.fulfill({ status: 503, json: { detail: "暂时不可用" } });
    } else {
      await route.fallback();
    }
  });
  await page.goto("/app");
  await expect(page.locator(".projectCatalogResults")).toContainText("请重试加载这一页");
  allowSuccess = true;
  await page.locator(".projectCatalogResults").getByRole("button", { name: "重试加载" }).click();
  await expect(page.locator(".projectList li")).toHaveCount(40);
  expect(mock.legacyListRequests()).toBe(0);
});

test("旧项目深链接在工作区分页、搜索和刷新后仍显示当前项目", async ({ page }) => {
  const mock = await mockCatalog(page);
  await page.goto(`/app/projects/${projectId(2191)}/documents`);
  await expect(page.getByText("当前：项目 2191", { exact: false })).toBeVisible();
  await expect(page.getByRole("combobox", { name: "选择项目" })).toHaveValue(projectId(2191));
  await expect(page.getByRole("navigation", { name: "工作区项目分页" })).toContainText("共 2191 项");
  await page.getByRole("button", { name: "刷新列表" }).click();
  await expect(page.getByText("当前：项目 2191", { exact: false })).toBeVisible();
  const workspaceNext = page.getByRole("navigation", { name: "工作区项目分页" }).getByRole("button", { name: "下一页" });
  await workspaceNext.focus();
  await workspaceNext.press("Enter");
  await expect(page.getByRole("navigation", { name: "工作区项目分页" })).toContainText("第 2 / 55 页");
  await expect(workspaceNext).toBeFocused();
  await expect(page.getByRole("combobox", { name: "选择项目" })).toHaveValue(projectId(2191));
  await page.getByRole("searchbox", { name: "搜索项目" }).fill("项目 0042");
  await expect(page.getByRole("navigation", { name: "工作区项目分页" })).toContainText("共 1 项");
  await expect(page.getByText("当前：项目 2191", { exact: false })).toBeVisible();
  expect(mock.requests.some((url) => url.searchParams.get("project_id") === projectId(2191))).toBe(true);
  expect(mock.legacyListRequests()).toBe(0);
});
