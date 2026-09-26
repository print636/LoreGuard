import { expect, test } from "@playwright/test";

const sourceLine = "  🌙林澈按住剑。随后她退到门边。  ";
const contextEnd = 2 + Array.from("🌙林澈按住剑。").length;
const targetEnd = contextEnd + Array.from("随后她退到门边。").length;

function candidate(invalid = false) {
  return {
    id: "candidate-1", character_key: "林澈", trait_type: "preference",
    trait_key: "行动倾向", value: "遇险时后退", origin: "explicit_setting",
    source_run_id: "run-1", scope_sha256: "snapshot-1", revision: 1,
    review_state: "pending", reviewable: true, model_coverage: "full",
    source_verified: true, contexts: [], confidence: 0.8,
    evidence: [{
      input_id: "input-1", document_id: "doc-1", document_name: "角色.md",
      document_version: 1, line_start: 1, line_end: 1, text: sourceLine,
      source_verified: true, source_text_exact: true, context_verified: false,
    }],
    support_bindings_status: "verified",
    support_bindings_v1: {
      schema_version: "character-support-bindings-v1",
      index_version: "assertion-index-v1",
      bindings: [{
        evidence_index: 0, support_id: "L1:A2",
        target: {
          support_id: "L1:A2", start_offset: contextEnd,
          end_offset: invalid ? 999 : targetEnd, role: "target",
        },
        actor_anchor_id: "L1:A1", label_anchor_id: null,
        scope_relation: "same_actor_continuation",
        context: [{
          support_id: "L1:A1", start_offset: 2,
          end_offset: contextEnd, role: "actor_anchor",
        }],
      }],
    },
  };
}

test("precise character evidence keeps the original line, distinction and fail-closed review", async ({ page }) => {
  let invalid = false;
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = decodeURIComponent(url.pathname);
    let body: unknown;
    if (path === "/api/v1/auth/me") body = {
      mode: "anonymous",
      user: { id: "user-1", email: "author@example.test", display_name: "作者" },
      workspace: { id: "workspace-1", name: "测试工作区", kind: "personal", role: "owner" },
    };
    else if (path === "/api/v1/project-catalog" && route.request().method() === "GET") {
      const item = {
        id: "project-1", name: "证据测试", description: "", created_at: "2026-09-01T00:00:00Z",
        active_document_count: 1, latest_run: null,
      };
      const page = Number(url.searchParams.get("page") || 1);
      const pageSize = Number(url.searchParams.get("page_size") || 40);
      const matches = (!url.searchParams.get("project_id") || url.searchParams.get("project_id") === item.id)
        && (!url.searchParams.get("query") || item.name.includes(url.searchParams.get("query")!.trim()));
      body = { page, page_size: pageSize, total: matches ? 1 : 0,
        items: matches && page === 1 ? [item] : [] };
    }
    else if (path === "/api/v1/projects/project-1/documents") body = [{
      id: "doc-1", project_id: "project-1", name: "角色.md", version: 1,
      active: true, created_at: "2026-09-01T00:00:00Z", content: sourceLine,
      document_role: "character_profile", story_scope: "主线", context_explicit: true,
      narrative_context: {
        resolution_state: "confirmed", publication_status: "published",
        document_role: "character_profile", scope: { schema_version: 1, timeline_key: "main" },
      },
    }];
    else if (path === "/api/v1/projects/project-1/analysis-runs") body = [{
      id: "run-1", project_id: "project-1", status: "completed",
      prompt_tokens: 1, completion_tokens: 1, estimated_cost_usd: 0,
      created_at: "2026-09-01T00:00:00Z",
    }];
    else if (path === "/api/v1/analysis-runs/run-1") body = {
      id: "run-1", project_id: "project-1", status: "completed",
      prompt_tokens: 1, completion_tokens: 1, estimated_cost_usd: 0,
      created_at: "2026-09-01T00:00:00Z",
    };
    else if (path === "/api/v1/analysis-runs/run-1/issues") body = [];
    else if (path === "/api/v1/analysis-runs/run-1/records") body = { records: [], warnings: [] };
    else if (path === "/api/v1/analysis-runs/run-1/diagnostics") body = {};
    else if (path === "/api/v1/analysis-runs/run-1/clarifications") body = [];
    else if (path === "/api/v1/projects/project-1/characters") body = {
      items: [{ id: "林澈", canonical_name: "林澈", aliases: [], confirmed_item_count: 0,
        pending_candidate_count: 1, drift_issue_count: 0, profile_revision: 0 }],
      page: 1, page_size: 30, total: 1, has_more: false,
      readiness: "ready", model_coverage: "full", source_run_id: "run-1",
    };
    else if (path === "/api/v1/projects/project-1/characters/林澈") body = {
      character: { id: "林澈", canonical_name: "林澈", aliases: [],
        confirmed_item_count: 0, pending_candidate_count: 1, drift_issue_count: 0 },
      profile_items: [], model_coverage: "full", source_run_id: "run-1",
    };
    else if (path === "/api/v1/projects/project-1/characters/林澈/profile-candidates") body = {
      items: [candidate(invalid)], page: 1, page_size: 20, total: 1, has_more: false,
      model_coverage: "full",
    };
    else if (path === "/api/v1/projects/project-1/characters/林澈/profile-candidates/candidate-1") {
      body = candidate(invalid);
    } else if (path.endsWith("/source-neighbors")) body = {
      candidate_id: "candidate-1", character_key: "林澈", source_run_id: "run-1",
      limit: 20, offset: 0, total: 0, has_more: false, items: [],
      source_groups: [{ input_id: "input-1", document_id: "doc-1",
        document_name: "角色.md", document_version: 1, line_start: 1,
        line_end: 1, total: 0, context_verified: false }],
    };
    else if (path === "/api/v1/account/model-provider") body = {};
    else {
      await route.fulfill({ status: 404, json: { detail: "not found" } });
      return;
    }
    await route.fulfill({ status: 200, json: body });
  });

  const path = "/app/projects/project-1/characters?section=candidates&page=1&character=%E6%9E%97%E6%BE%88&candidate=candidate-1";
  await page.goto(path);
  const quote = page.locator(".characterEvidenceGroup.supporting blockquote").first();
  await expect(quote).toBeVisible();
  await expect(quote.locator("mark.characterEvidenceTarget")).toHaveText("随后她退到门边。");
  await expect(quote.locator(".characterEvidenceContext")).toHaveText("🌙林澈按住剑。");
  await expect(quote).toHaveText(sourceLine);
  await expect(page.getByText(/定位不代表归纳已被作者确认/)).toBeVisible();
  const confirm = page.getByRole("button", { name: "确认归纳" });
  await expect(confirm).toBeEnabled();
  await page.getByLabel("审核备注（可选）").focus();
  await page.keyboard.press("Tab");
  await expect(confirm).toBeFocused();
  expect(await confirm.evaluate((button) =>
    Number.parseFloat(getComputedStyle(button).outlineWidth))).toBeGreaterThanOrEqual(2);
  const targetContrast = await quote.locator("mark.characterEvidenceTarget").evaluate((mark) => {
    const style = getComputedStyle(mark);
    const channel = (color: string) => (color.match(/[0-9.]+/g) || []).slice(0, 3)
      .map((raw) => {
        const value = Number(raw) / 255;
        return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
      });
    const luminance = (color: string) => {
      const [red, green, blue] = channel(color);
      return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
    };
    const lighter = Math.max(luminance(style.color), luminance(style.backgroundColor));
    const darker = Math.min(luminance(style.color), luminance(style.backgroundColor));
    return (lighter + 0.05) / (darker + 0.05);
  });
  expect(targetContrast).toBeGreaterThanOrEqual(4.5);
  for (const width of [1024, 360]) {
    await page.setViewportSize({ width, height: 780 });
    await expect(quote).toBeVisible();
    const box = await quote.boundingBox();
    expect(box).not.toBeNull();
    expect(box!.x).toBeGreaterThanOrEqual(0);
    expect(box!.x + box!.width).toBeLessThanOrEqual(width + 1);
    if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
      await quote.scrollIntoViewIfNeeded();
      await page.screenshot({
        path: `../artifacts/playwright/character-support-${width}.png`,
        fullPage: true,
      });
    }
  }

  invalid = true;
  await page.reload();
  await expect(page.getByText(/精确证据定位未通过冻结原文核对/).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "确认归纳" })).toBeDisabled();
  await expect(page.locator(".characterEvidenceGroup.supporting mark")).toHaveCount(0);
});
