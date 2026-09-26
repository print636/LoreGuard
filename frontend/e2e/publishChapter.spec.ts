import { expect, test, type Page } from "@playwright/test";

const projectId = "publish-chapter-project";
const documentId = "chapter-one";
const root = `/api/v1/projects/${projectId}`;
const now = "2026-09-27T00:00:00Z";

type Context = {
  revision: number;
  resolution_state: "confirmed" | "unresolved";
  origin: string;
  publication_status: "draft" | "in_review" | "published";
  scope_sha256: string;
  scope: { schema_version: 1; timeline_key: string };
};

type MockState = {
  context: Context;
  role: "chapter" | "reference";
  posts: unknown[];
  contextReads: number;
  conflict?: string;
  unexpected: string[];
};

function initialState(): MockState {
  return {
    context: {
      revision: 2,
      resolution_state: "confirmed",
      origin: "explicit",
      publication_status: "draft",
      scope_sha256: "main-scope",
      scope: { schema_version: 1, timeline_key: "main" },
    },
    role: "chapter",
    posts: [], contextReads: 0, unexpected: [],
  };
}

async function mockApi(page: Page, state: MockState) {
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();
    let body: unknown;
    if (path === "/api/v1/auth/me") body = {
      mode: "anonymous",
      user: { id: "author", email: "", display_name: "作者" },
      workspace: { id: "workspace", name: "写作工作区", kind: "personal", role: "owner" },
    };
    else if (path === "/api/v1/account/model-provider") body = {};
    else if (path === "/api/v1/project-catalog") body = {
      page: 1, page_size: 40, total: 1,
      items: [{ id: projectId, name: "连载项目", description: "", created_at: now,
        active_document_count: 1, latest_run: null }],
    };
    else if (path === `${root}/documents` && method === "GET") body = [{
      id: documentId,
      project_id: projectId,
      name: "第一章.md",
      version: 1,
      active: true,
      created_at: now,
      content: "原创剧情正文。",
      document_role: state.role,
      story_scope: "main",
      narrative_context: state.context,
    }];
    else if (path === `${root}/analysis-runs` && method === "GET") body = [];
    else if (path === `${root}/documents/${documentId}/narrative-context` && method === "GET") {
      state.contextReads += 1;
      body = { document_id: documentId, current: state.context, revision_count: state.context.revision };
    }
    else if (path === `${root}/documents/${documentId}/publish` && method === "POST") {
      state.posts.push(route.request().postDataJSON());
      if (state.conflict) {
        await route.fulfill({ status: 409, json: { detail: { code: state.conflict, message: "mock conflict" } } });
        return;
      }
      state.context = { ...state.context, revision: state.context.revision + 1, publication_status: "published" };
      await route.fulfill({ status: 201, json: {
        ...state.context,
        id: "revision-three",
        project_id: projectId,
        document_id: documentId,
        document_role: "chapter",
        document_version: 1,
        created_at: now,
      } });
      return;
    }
    else {
      state.unexpected.push(`${method} ${path}`);
      await route.fulfill({ status: 404, json: { detail: "unexpected mocked endpoint" } });
      return;
    }
    await route.fulfill({ status: 200, json: body });
  });
}

test("作者明确确认后才发布章节；没有审查报告也不自动阻止或宣称 AI 通过", async ({ page }, testInfo) => {
  const state = initialState();
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/documents`);
  const workbench = page.locator(".contextWorkbench");
  const publish = workbench.getByRole("button", { name: "作者定稿并发布…" });
  await expect(publish).toBeEnabled();
  await expect(workbench.getByLabel("发布状态").getByRole("option", { name: "已发布" })).toHaveCount(0);
  await publish.click();
  const dialog = page.getByRole("dialog", { name: "确认由作者定稿并发布" });
  await expect(dialog).toBeVisible();
  await expect(dialog).toContainText("发布是作者决定，不表示系统认定剧情无误");
  await expect(dialog).toContainText("没有审查报告、覆盖不完整或报告仍有问题线索");
  await dialog.press("Escape");
  await expect(dialog).not.toBeVisible();
  expect(state.posts).toEqual([]);

  await publish.click();
  await dialog.getByRole("button", { name: "确认发布此章节" }).click();
  await expect(workbench.getByText(/已由你定稿并发布/)).toBeVisible();
  await expect(workbench.getByText(/此章节已发布/)).toBeVisible();
  await expect(publish).toHaveCount(0);
  expect(state.posts).toEqual([{ expected_revision: 2, expected_document_version: 1 }]);
  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await workbench.screenshot({ path: testInfo.outputPath("published-desktop.png") });
  }
  expect(state.unexpected).toEqual([]);
});

test("版本冲突时不重复发布；375px 弹窗可键盘取消、错误可恢复", async ({ page }, testInfo) => {
  const state = initialState();
  state.conflict = "narrative_context_revision_conflict";
  await mockApi(page, state);
  await page.setViewportSize({ width: 375, height: 812 });
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`/app/projects/${projectId}/documents`);
  const publish = page.getByRole("button", { name: "作者定稿并发布…" });
  await publish.focus();
  await publish.press("Enter");
  const dialog = page.getByRole("dialog", { name: "确认由作者定稿并发布" });
  await expect(dialog).toBeVisible();
  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await dialog.screenshot({ path: testInfo.outputPath("publish-confirm-mobile-375.png") });
  }
  await dialog.getByRole("button", { name: "确认发布此章节" }).click();
  await expect(dialog.getByRole("alert")).toContainText("文稿版本或资料上下文已变化");
  await expect(dialog.getByRole("button", { name: "确认发布此章节" })).toBeDisabled();
  const contextReadsBeforeRecovery = state.contextReads;
  await dialog.getByRole("button", { name: "关闭并重新读取" }).click();
  await expect(dialog).not.toBeVisible();
  await expect.poll(() => state.contextReads).toBeGreaterThan(contextReadsBeforeRecovery);
  expect(state.posts).toEqual([{ expected_revision: 2, expected_document_version: 1 }]);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(state.unexpected).toEqual([]);
});

test("待确认章节不能发布，导入既有历史仍可独立标注已发布", async ({ page }) => {
  const state = initialState();
  state.context = { ...state.context, resolution_state: "unresolved" };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/documents`);
  await expect(page.getByRole("button", { name: "作者定稿并发布…" })).toBeDisabled();
  await expect(page.locator(".contextPublishAction")).toContainText("请先确认这份章节");
  state.role = "reference";
  state.context = { ...state.context, resolution_state: "confirmed", publication_status: "published" };
  await page.reload();
  await expect(page.getByRole("button", { name: "作者定稿并发布…" })).toHaveCount(0);
  await expect(page.locator(".contextWorkbench").getByLabel("发布状态").getByRole("option", { name: "已发布" })).toHaveCount(1);
  expect(state.posts).toEqual([]);
  expect(state.unexpected).toEqual([]);
});
