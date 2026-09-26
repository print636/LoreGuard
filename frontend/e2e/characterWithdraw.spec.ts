import { expect, test } from "@playwright/test";

test("a confirmed trait remains accessible after source retirement and can be explicitly withdrawn", async ({ page }) => {
  let withdrawn = false;
  let postCount = 0;
  let revision = 3;
  let conflictOnce = true;
  const trait = {
    id: "trait-1", character_key: "林澈", trait_type: "preference",
    trait_key: "食物偏好", value: "喜欢蜜瓜", origin: "explicit_setting",
    source_run_id: "run-old", scope_sha256: "old-snapshot",
    review_state: "confirmed", revision: 3, model_coverage: "full",
    evidence: [{ document_id: "doc-retired", document_name: "旧设定.md", document_version: 1,
      document_role: "reference", publication_status: "retired", line_start: 1,
      line_end: 1, text: "林澈喜欢蜜瓜。" }],
  };
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = decodeURIComponent(url.pathname);
    let body: unknown;
    if (path === "/api/v1/auth/me") body = {
      mode: "anonymous",
      user: { id: "user-1", email: "author@example.test", display_name: "作者" },
      workspace: { id: "workspace-1", name: "测试工作区", kind: "personal", role: "owner" },
    };
    else if (path === "/api/v1/projects") body = [{
      id: "project-1", name: "退役来源测试", description: "", created_at: "2026-09-01T00:00:00Z",
      active_document_count: 0, latest_run: null,
    }];
    else if (path === "/api/v1/projects/project-1/documents") body = [];
    else if (path === "/api/v1/projects/project-1/analysis-runs") body = [{
      id: "run-old", project_id: "project-1", status: "completed",
      prompt_tokens: 1, completion_tokens: 1, estimated_cost_usd: 0,
      created_at: "2026-09-01T00:00:00Z",
    }];
    else if (path === "/api/v1/analysis-runs/run-old") body = {
      id: "run-old", project_id: "project-1", status: "completed",
      prompt_tokens: 1, completion_tokens: 1, estimated_cost_usd: 0,
      created_at: "2026-09-01T00:00:00Z",
    };
    else if (path === "/api/v1/analysis-runs/run-old/issues") body = [];
    else if (path === "/api/v1/analysis-runs/run-old/records") body = { records: [], warnings: [] };
    else if (path === "/api/v1/analysis-runs/run-old/diagnostics") body = {};
    else if (path === "/api/v1/analysis-runs/run-old/clarifications") body = [];
    else if (path === "/api/v1/projects/project-1/characters") body = {
      items: url.searchParams.has("query") ? [] : [{ character_key: "林澈", character_display_name: "林澈",
        confirmed_trait_count: withdrawn ? 0 : 1, pending_candidate_count: 0,
        withdrawn_trait_count: withdrawn ? 1 : 0,
        profile_revision: revision, updated_at: "2026-09-01T00:00:00Z" }],
      page: 1, page_size: 30, total: url.searchParams.has("query") ? 0 : 1, has_more: false,
      readiness: "no_documents", model_coverage: "full", source_run_id: "run-old",
    };
    else if (path === "/api/v1/projects/project-1/characters/林澈") body = {
      character_key: "林澈", character_display_name: "林澈",
      confirmed_traits: withdrawn ? [] : [{ ...trait, revision }], pending_candidate_count: 0,
      model_coverage: "full", source_run_id: "run-old",
    };
    else if (path === "/api/v1/projects/project-1/characters/林澈/profile-candidates" &&
      url.searchParams.get("state") === "withdrawn") body = {
      items: withdrawn ? [{ ...trait, review_state: "withdrawn", revision }] : [],
      page: 1, page_size: 20, total: withdrawn ? 1 : 0, has_more: false,
      model_coverage: "full",
    };
    else if (path === "/api/v1/projects/project-1/characters/林澈/profile-candidates/trait-1/decisions" &&
      route.request().method() === "POST") {
      postCount += 1;
      expect(route.request().postDataJSON()).toEqual({
        decision: "withdraw", comment: "", expected_revision: revision,
      });
      if (conflictOnce) {
        conflictOnce = false;
        revision += 1;
        await route.fulfill({ status: 409, json: {
          detail: { code: "character_trait_revision_conflict", message: "并发更新" },
        } });
        return;
      }
      withdrawn = true;
      revision += 1;
      await route.fulfill({ status: 201, json: {
        candidate: { ...trait, review_state: "withdrawn", revision },
        decision_id: "decision-1", deduplicated: false,
      } });
      return;
    } else if (path === "/api/v1/account/model-provider") body = {};
    else {
      await route.fulfill({ status: 404, json: { detail: "not found" } });
      return;
    }
    await route.fulfill({ status: 200, json: body });
  });

  const profilePath = "/app/projects/project-1/characters?section=profile&page=1&character=%E6%9E%97%E6%BE%88";
  await page.goto(`${profilePath}&q=${encodeURIComponent("另一个角色")}`);
  await expect(page.getByText("食物偏好：喜欢蜜瓜")).toBeVisible();
  await expect(page.getByText("没有匹配的角色")).toBeVisible();
  await expect(page.getByText(/来源文档退役或改为参考资料，不会自动撤销/)).toBeVisible();
  await expect(page.getByRole("heading", { name: "已撤销记录" })).toBeVisible();
  await page.goto(profilePath);
  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await page.screenshot({ path: "../artifacts/playwright/character-withdraw-before.png", fullPage: true });
  }

  await page.getByRole("button", { name: "撤销这条特征" }).click();
  await expect(page.getByText(/已有运行报告不会改写/)).toBeVisible();
  await page.getByRole("button", { name: "保留特征" }).click();
  expect(postCount).toBe(0);

  await page.getByRole("button", { name: "撤销这条特征" }).click();
  await page.getByRole("button", { name: "确认撤销特征" }).click();
  await expect(page.getByText(/这条特征已在其他页面更新；档案正在刷新/)).toBeVisible();
  await expect(page.getByRole("button", { name: "撤销这条特征" })).toBeVisible();
  await expect(page.locator(".characterWithdrawError")).toContainText("这条特征已在其他页面更新；档案正在刷新");
  await expect(page.getByRole("alert").filter({ hasText: /这条特征已在其他页面更新；档案正在刷新/ })).toHaveCount(1);
  expect(postCount).toBe(1);

  await page.getByRole("button", { name: "撤销这条特征" }).click();
  await page.getByRole("button", { name: "确认撤销特征" }).click();
  await expect(page.getByText(/特征已撤销，将不再参与之后新建的审查/)).toBeVisible();
  await expect(page.getByText("还没有生效的角色特征")).toBeVisible();
  await expect(page.locator(".characterWithdrawnArchive").getByText("食物偏好：喜欢蜜瓜")).toBeVisible();
  await expect(page.locator(".characterWithdrawnArchive").getByText("已撤销", { exact: true })).toBeVisible();
  await expect(page.locator(".characterRosterList").getByText("1 已撤销")).toBeVisible();
  expect(postCount).toBe(2);

  await page.setViewportSize({ width: 375, height: 780 });
  const archive = await page.locator(".characterWithdrawnArchive").boundingBox();
  expect(archive).not.toBeNull();
  expect(archive!.x).toBeGreaterThanOrEqual(0);
  expect(archive!.x + archive!.width).toBeLessThanOrEqual(376);
  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await page.locator(".characterWithdrawnArchive").scrollIntoViewIfNeeded();
    await page.screenshot({ path: "../artifacts/playwright/character-withdraw-mobile-after.png", fullPage: true });
  }
});
