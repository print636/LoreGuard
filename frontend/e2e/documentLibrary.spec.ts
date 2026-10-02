import { expect, test, type Page } from "@playwright/test";

const projectId = "document-library-project";
const root = `/api/v1/projects/${projectId}`;
const now = "2026-10-02T00:00:00Z";
const screenshotFolder = "C:/Users/dell/Desktop/doc/creative-media/LoreGuard-资料库-v1";

type DocumentFixture = {
  id: string;
  project_id: string;
  name: string;
  version: number;
  active: boolean;
  created_at: string;
  document_role: "chapter" | "canon" | "character_profile" | "reference";
  story_scope: string;
  narrative_context: {
    revision: number;
    resolution_state: "confirmed";
    origin: "explicit";
    publication_status: "draft" | "published";
    scope_sha256: string;
    scope: { schema_version: 1; timeline_key: string };
  };
};

function fixture(id: string, name: string, version: number, active: boolean,
  documentRole: DocumentFixture["document_role"] = "chapter"): DocumentFixture {
  return {
    id, project_id: projectId, name, version, active, created_at: now,
    document_role: documentRole, story_scope: "main",
    narrative_context: {
      revision: 1, resolution_state: "confirmed", origin: "explicit",
      publication_status: documentRole === "chapter" ? "draft" : "published",
      scope_sha256: "main-scope", scope: { schema_version: 1, timeline_key: "main" },
    },
  };
}

function initialState() {
  const documents = [
    fixture("chapter-current", "01-北港主线.md", 2, true),
    fixture("chapter-old", "01-北港主线.md", 1, false),
    fixture("canon-current", "02-世界观设定.docx", 1, true, "canon"),
    fixture("retired-old", "归档参考.txt", 1, false, "reference"),
    ...Array.from({ length: 23 }, (_, index) => fixture(
      `reference-${index}`, `参考资料 ${String(index + 1).padStart(2, "0")}.txt`, 1, true, "reference",
    )),
  ];
  const longLines = Array.from({ length: 245 }, (_, index) => {
    const line = index + 1;
    if (line === 7) return `存档标识：${"LongUnbrokenStoryIdentifier".repeat(16)}`;
    if (line === 121 || line === 241) return `第 ${line} 行：林澈取得[北港密钥]，旧稿保留另一种选择。`;
    return `第 ${line} 行：北港主线当前版本的原创剧情。`;
  });
  return {
    documents,
    bodies: new Map<string, string[]>([
      ["chapter-current", longLines],
      ["chapter-old", ["旧稿第一行：林澈还未取得钥匙。", "旧稿第二行：北港路线仍待策划确认。"]],
      ["canon-current", ["设定第一行：北港由青岚议会管理。", "设定第二行：发布历史高于新稿。"]],
      ["retired-old", ["归档参考的历史正文，当前没有活动版本。"]],
      ...documents.filter((document) => document.id.startsWith("reference-")).map((document) =>
        [document.id, [`${document.name}的原创参考内容。`]] as [string, string[]]),
    ]),
    previews: [] as URL[],
    diffs: [] as URL[],
    writes: [] as string[],
    unexpected: [] as string[],
    failedDocument: "",
    delayedDocument: "",
    releaseDelayed: undefined as (() => void) | undefined,
    delayedFinished: false,
    emptyDocument: "",
  };
}

type MockState = ReturnType<typeof initialState>;

async function mockApi(page: Page, state: MockState) {
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();
    if (method !== "GET") state.writes.push(`${method} ${path}`);
    let body: unknown;
    if (path === "/api/v1/auth/me") body = {
      mode: "anonymous",
      user: { id: "reader-author", email: "", display_name: "剧情策划" },
      workspace: { id: "reader-workspace", name: "原创剧情工作区", kind: "personal", role: "owner" },
    };
    else if (path === "/api/v1/account/model-provider") body = { configured: false, revision: 0 };
    else if (path === "/api/v1/project-catalog") body = {
      page: 1, page_size: 40, total: 1,
      items: [{ id: projectId, name: "北港故事项目", description: "资料阅读验收", created_at: now,
        active_document_count: state.documents.filter((document) => document.active).length, latest_run: null }],
    };
    else if (path === `${root}/documents` && method === "GET") body = state.documents;
    else if (path === `${root}/analysis-runs` && method === "GET") body = [];
    else if (path.endsWith("/narrative-context") && method === "GET") {
      const documentId = decodeURIComponent(path.split("/").at(-2) || "");
      const document = state.documents.find((item) => item.id === documentId);
      body = { document_id: documentId, current: document?.narrative_context, revision_count: 1 };
    }
    else if (path === `${root}/documents/diff` && method === "GET") {
      state.diffs.push(url);
      const from = state.documents.find((document) => document.id === url.searchParams.get("from_document_id"));
      const to = state.documents.find((document) => document.id === url.searchParams.get("to_document_id"));
      body = {
        from_document: { ...from, char_count: 30, line_count: 2 },
        to_document: { ...to, char_count: 4500, line_count: 245 },
        summary: { added_lines: 1, removed_lines: 1, unchanged_lines: 0, changed_hunks: 1,
          compared_old_lines: 2, compared_new_lines: 245, old_total_lines: 2, new_total_lines: 245,
          input_truncated: false, output_truncated: false },
        hunks: [{ old_start: 1, old_lines: 1, new_start: 1, new_lines: 1, lines: [
          { type: "removed", content: "旧稿第一行：林澈还未取得钥匙。", old_line: 1, new_line: null },
          { type: "added", content: "第 1 行：北港主线当前版本的原创剧情。", old_line: null, new_line: 1 },
        ] }], warnings: [],
      };
    }
    else if (path.startsWith(`${root}/documents/`) && path.endsWith("/preview") && method === "GET") {
      state.previews.push(url);
      const documentId = decodeURIComponent(path.split("/").at(-2) || "");
      const document = state.documents.find((item) => item.id === documentId);
      if (!document) {
        await route.fulfill({ status: 404, json: { detail: "文稿不存在" } });
        return;
      }
      if (state.failedDocument === documentId) {
        await route.fulfill({ status: 503, json: { detail: "正文暂时无法读取" } });
        return;
      }
      if (state.delayedDocument === documentId) {
        await new Promise<void>((resolve) => { state.releaseDelayed = resolve; });
      }
      const lines = state.emptyDocument === documentId ? [] : state.bodies.get(documentId) || [];
      const query = url.searchParams.get("query") || "";
      const offset = Number(url.searchParams.get("offset") || 0);
      const limit = Number(url.searchParams.get("limit") || 120);
      const matching = lines.map((text, index) => ({ line_number: index + 1, text }))
        .filter((line) => !query || line.text.toLowerCase().includes(query.toLowerCase()));
      body = {
        document: { ...document, char_count: lines.join("\n").length, line_count: lines.length,
          content_sha256: "a".repeat(64) },
        lines: matching.slice(offset, offset + limit),
        page: { offset, limit, total: matching.length, has_more: offset + limit < matching.length }, query,
      };
      if (state.delayedDocument === documentId) state.delayedFinished = true;
    }
    else {
      state.unexpected.push(`${method} ${path}`);
      await route.fulfill({ status: 404, json: { detail: "unexpected mocked endpoint" } });
      return;
    }
    await route.fulfill({ status: 200, json: body });
  });
}

async function openLibrary(page: Page, state: MockState) {
  await mockApi(page, state);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`/app/projects/${projectId}/documents`);
  await expect(page.locator(".documentLibrary")).toBeVisible();
  await expect(page.getByRole("complementary", { name: "运行与冲突报告" })).toHaveCount(0);
}

function assertReadOnly(state: MockState) {
  expect(state.writes, "资料阅读、字面搜索和版本比较应全部只读").toEqual([]);
  expect(state.unexpected, "验收fixture覆盖全部请求，禁止悄悄调用模型接口").toEqual([]);
}

test("游戏剧情策划：同名文稿聚合、筛选历史稿，比较准确版本后返回继续阅读", async ({ page }) => {
  const state = initialState();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openLibrary(page, state);
  const library = page.locator(".documentLibrary");
  const list = library.getByRole("list", { name: "文稿列表" });
  await expect(library).toContainText("26 份文稿，27 个版本");
  await expect(list.locator(":scope > li")).toHaveCount(20);
  await expect(list.getByRole("button", { name: /01-北港主线\.md/ })).toHaveCount(1);
  const pagination = library.getByRole("navigation", { name: "文稿列表分页" });
  await pagination.getByRole("button", { name: "下一页", exact: true }).click();
  await expect(list.locator(":scope > li")).toHaveCount(6);
  await expect(pagination).toContainText("21–26 份，共 26 份");
  await pagination.getByRole("button", { name: "上一页", exact: true }).click();

  const filenameSearch = library.getByRole("searchbox", { name: "搜索文件名" });
  await filenameSearch.fill("主线");
  await filenameSearch.press("Enter");
  await library.getByRole("combobox", { name: "资料类型", exact: true }).selectOption("chapter");
  await library.getByRole("combobox", { name: "版本范围" }).selectOption("history");
  await expect(list.locator(":scope > li")).toHaveCount(1);
  const chapter = list.getByRole("button", { name: /01-北港主线\.md/ });
  await chapter.focus();
  await chapter.press("Enter");
  const reader = page.locator("#document-library-reader");
  await expect(reader.getByRole("combobox", { name: "阅读版本" })).toHaveValue("chapter-old");
  await expect(reader).toContainText("历史版本（仅阅读）");
  await expect(reader.locator(".documentLibraryLines > li")).toHaveCount(2);
  expect((await reader.boundingBox())?.width, "1440px资料库正文区应有足够阅读宽度").toBeGreaterThanOrEqual(600);
  await expect(reader.getByRole("heading", { name: "01-北港主线.md" })).toBeFocused();
  await reader.getByRole("combobox", { name: "阅读版本" }).selectOption("chapter-current");
  await expect(reader).toContainText("当前版本（在用）");
  await expect(reader.locator(".documentLibraryLines > li")).toHaveCount(120);
  await reader.getByRole("combobox", { name: "阅读版本" }).selectOption("chapter-old");
  await expect(reader.locator(".documentLibraryLines > li")).toHaveCount(2);
  await reader.getByRole("button", { name: "与当前版本比较" }).click();
  await expect(page).toHaveURL(new RegExp(`/app/projects/${projectId}/compare(?:\\?|$)`));
  await expect(page.locator(".diffControls select").nth(0)).toHaveValue("chapter-old");
  await expect(page.locator(".diffControls select").nth(1)).toHaveValue("chapter-current");
  await expect(page.locator(".diffResult")).toContainText("v1 → v2");
  await expect(page.getByRole("complementary", { name: "运行与冲突报告" })).toBeVisible();
  expect(state.diffs.at(-1)?.searchParams.get("from_document_id")).toBe("chapter-old");
  expect(state.diffs.at(-1)?.searchParams.get("to_document_id")).toBe("chapter-current");
  await page.goBack();
  await expect(page).toHaveURL(new RegExp(`/app/projects/${projectId}/documents$`));
  await expect(filenameSearch).toHaveValue("主线");
  await expect(library.getByRole("combobox", { name: "资料类型", exact: true })).toHaveValue("chapter");
  await expect(library.getByRole("combobox", { name: "版本范围" })).toHaveValue("history");
  await expect(reader.getByRole("combobox", { name: "阅读版本" })).toHaveValue("chapter-old");
  await expect(chapter).toHaveAttribute("aria-pressed", "true");
  await expect(reader.locator(".documentLibraryLines > li")).toHaveCount(2);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  assertReadOnly(state);
});

test("独立网文作者：分页阅读并按字面查找，匹配行保留原始行号", async ({ page }) => {
  const state = initialState();
  await openLibrary(page, state);
  const library = page.locator(".documentLibrary");
  await library.getByRole("list", { name: "文稿列表" }).getByRole("button", { name: /01-北港主线\.md/ }).click();
  const reader = page.locator("#document-library-reader");
  const lines = reader.locator(".documentLibraryLines > li");
  const lineNumbers = reader.locator(".documentLibraryLineNumber");
  await expect(lines).toHaveCount(120);
  await expect(lineNumbers.first()).toHaveText("1");
  await expect(lineNumbers.last()).toHaveText("120");
  await reader.getByRole("button", { name: "下一页原文" }).click();
  await expect(lineNumbers.first()).toHaveText("121");
  await expect(lineNumbers.last()).toHaveText("240");
  await reader.getByRole("button", { name: "下一页原文" }).click();
  await expect(lines).toHaveCount(5);
  await expect(lineNumbers.first()).toHaveText("241");
  await expect(reader.getByRole("button", { name: "下一页原文" })).toBeDisabled();
  expect(state.previews.some((url) => url.searchParams.get("offset") === "240" && url.searchParams.get("limit") === "120")).toBe(true);

  const search = reader.getByRole("searchbox", { name: "在此版本中搜索原文" });
  await search.fill("[北港密钥]");
  await search.press("Enter");
  await expect(lines).toHaveCount(2);
  await expect(lineNumbers).toHaveText(["121", "241"]);
  await expect(reader.locator("mark")).toHaveText(["[北港密钥]", "[北港密钥]"]);
  await expect(reader).toContainText("找到 2 行匹配原文");
  expect(state.previews.at(-1)?.searchParams.get("query")).toBe("[北港密钥]");
  expect(state.previews.at(-1)?.searchParams.get("offset")).toBe("0");

  await search.fill("北港");
  await search.press("Enter");
  await expect(lines).toHaveCount(120);
  await expect(lineNumbers.last()).toHaveText("121");
  await reader.getByRole("button", { name: "下一页原文" }).click();
  await expect(lineNumbers.first()).toHaveText("122");
  await search.fill("此版本没有的词句");
  await search.press("Enter");
  await expect(reader).toContainText("此版本中没有匹配文字");
  await expect(lines).toHaveCount(0);
  await reader.getByRole("button", { name: "清除", exact: true }).click();
  await expect(lines).toHaveCount(120);
  await expect(lineNumbers.first()).toHaveText("1");
  await expect(reader.locator("mark")).toHaveCount(0);
  assertReadOnly(state);
});

test("快速切换旧稿与当前稿：迟到正文不能覆盖当前版本", async ({ page }) => {
  const state = initialState();
  state.delayedDocument = "chapter-old";
  await openLibrary(page, state);
  await page.locator(".documentLibraryList").getByRole("button", { name: /01-北港主线\.md/ }).click();
  const reader = page.locator("#document-library-reader");
  const version = reader.getByRole("combobox", { name: "阅读版本" });
  await expect(reader.locator(".documentLibraryLines > li")).toHaveCount(120);
  await version.selectOption("chapter-old");
  await expect.poll(() => Boolean(state.releaseDelayed)).toBe(true);
  await expect(reader).toContainText("正在读取 v1 的正文");
  await expect(reader.locator(".documentLibraryLines > li")).toHaveCount(0);
  await version.selectOption("chapter-current");
  await expect(reader.locator(".documentLibraryLines > li")).toHaveCount(120);
  state.releaseDelayed?.();
  await expect.poll(() => state.delayedFinished).toBe(true);
  await expect(version).toHaveValue("chapter-current");
  await expect(reader).toContainText("当前版本（在用）");
  await expect(reader.locator(".documentLibraryLines > li")).toHaveCount(120);
  await expect(reader).not.toContainText("旧稿第一行");
  assertReadOnly(state);
});

test("正文读取失败后可重试；无匹配资料、空项目和空正文各有明确说明", async ({ page }) => {
  const state = initialState();
  state.failedDocument = "chapter-current";
  await openLibrary(page, state);
  const library = page.locator(".documentLibrary");
  await library.getByRole("list", { name: "文稿列表" }).getByRole("button", { name: /01-北港主线\.md/ }).click();
  const reader = page.locator("#document-library-reader");
  await expect(reader.getByRole("alert")).toContainText("服务器暂时无法读取原文");
  state.failedDocument = "";
  const retry = reader.getByRole("button", { name: "重试读取" });
  await retry.focus();
  await retry.press("Enter");
  await expect(reader.locator(".documentLibraryLines > li")).toHaveCount(120);

  await library.getByRole("searchbox", { name: "搜索文件名" }).fill("不存在的文稿");
  await library.getByRole("searchbox", { name: "搜索文件名" }).press("Enter");
  await expect(library).toContainText("没有符合条件的文稿");
  await expect(reader).toContainText("选择文稿，阅读当前或历史版本");
  await library.getByRole("button", { name: "清除筛选" }).click();
  await library.getByRole("list", { name: "文稿列表" }).getByRole("button", { name: /02-世界观设定\.docx/ }).click();
  await expect(reader).toContainText("导入后保存的正文，不还原 Word 排版");
  await expect(reader.locator(".documentLibraryLines > li")).toHaveCount(2);
  state.emptyDocument = "canon-current";
  await reader.getByRole("searchbox", { name: "在此版本中搜索原文" }).fill("任何文字");
  await reader.getByRole("button", { name: "查找", exact: true }).click();
  await expect(reader).toContainText("此版本中没有匹配文字");
  await reader.getByRole("button", { name: "清除", exact: true }).click();
  await expect(reader).toContainText("此版本保存的正文为空");
  await library.getByRole("searchbox", { name: "搜索文件名" }).fill("归档参考");
  await library.getByRole("searchbox", { name: "搜索文件名" }).press("Enter");
  await library.getByRole("list", { name: "文稿列表" }).getByRole("button", { name: /归档参考\.txt/ }).click();
  await expect(reader).toContainText("历史版本（仅阅读）");
  await expect(reader.locator(".documentLibraryLines > li")).toHaveCount(1);
  await expect(reader.getByRole("button", { name: "与当前版本比较" })).toHaveCount(0);
  state.documents = [];
  await page.reload();
  await expect(library).toContainText("项目中还没有文稿");
  await expect(library.getByRole("list", { name: "文稿列表" })).toHaveCount(0);
  assertReadOnly(state);
});

test("360px与1440px可阅读长标识，键盘可以选稿、搜索和翻页", async ({ page }) => {
  const state = initialState();
  await page.setViewportSize({ width: 360, height: 900 });
  await openLibrary(page, state);
  const library = page.locator(".documentLibrary");
  const chapter = library.getByRole("list", { name: "文稿列表" }).getByRole("button", { name: /01-北港主线\.md/ });
  await chapter.focus();
  await chapter.press("Enter");
  const reader = page.locator("#document-library-reader");
  await expect(reader.locator(".documentLibraryLines > li")).toHaveCount(120);
  await expect(reader.locator(".documentLibraryLines > li").nth(6)).toContainText("LongUnbrokenStoryIdentifier");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await reader.getByRole("heading").scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${screenshotFolder}/document-library-mobile-360.png` });
  const search = reader.getByRole("searchbox", { name: "在此版本中搜索原文" });
  await search.focus();
  await search.fill("[北港密钥]");
  await search.press("Enter");
  await expect(reader.locator(".documentLibraryLineNumber")).toHaveText(["121", "241"]);
  await reader.getByRole("button", { name: "清除", exact: true }).focus();
  await page.keyboard.press("Enter");
  await expect(reader.locator(".documentLibraryLines > li")).toHaveCount(120);
  const next = reader.getByRole("button", { name: "下一页原文" });
  await next.focus();
  await next.press("Enter");
  await expect(reader.locator(".documentLibraryLineNumber").first()).toHaveText("121");
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.setViewportSize({ width: 1440, height: 1000 });
  await reader.getByRole("button", { name: "上一页原文" }).click();
  await expect(reader.locator(".documentLibraryLineNumber").first()).toHaveText("1");
  await library.getByRole("heading", { name: "资料库", exact: true }).evaluate((heading) => {
    heading.scrollIntoView({ block: "start", inline: "nearest", behavior: "instant" });
  });
  await page.evaluate(() => window.scrollTo({ top: 0, left: 0, behavior: "instant" }));
  expect((await reader.boundingBox())?.width, "1440px资料库正文区应有足够阅读宽度").toBeGreaterThanOrEqual(600);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: `${screenshotFolder}/document-library-desktop-1440.png` });
  assertReadOnly(state);
});
