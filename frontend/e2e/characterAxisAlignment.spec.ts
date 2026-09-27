import { expect, test, type Page } from "@playwright/test";

const projectId = "axis-demo-project";
const characterId = "林澈";
const candidateId = "axis-candidate";
const axisId = "author-axis";
const proposition = "林澈未经授权冒用同伴签名";
const propositionHash = "b".repeat(64);
const axisDefinitionHash = "a".repeat(64);
const sourceLine = "林澈在需要签字时先征询同伴，从不擅自代签。";
const sourceContext = "林澈在需要签字时先征询同伴，";
const sourceTarget = "从不擅自代签。";
const sourceContextEnd = Array.from(sourceContext).length;
const sourceTargetEnd = sourceContextEnd + Array.from(sourceTarget).length;
const root = `/api/v1/projects/${projectId}`;
const characterRoot = `${root}/characters/${characterId}`;
const candidateRoot = `${characterRoot}/profile-candidates/${candidateId}`;

type MockState = {
  confirmed: boolean;
  mapped: boolean;
  proposition: string | null;
  conflictOnce?: boolean;
  propositionPostCount: number;
  decisionPostCount: number;
  unexpected: string[];
  preciseEvidence?: boolean;
  mappingHashMismatch?: boolean;
  scoped?: boolean;
  scopeConflictOnce?: boolean;
  scopedNoAxis?: boolean;
  axisScope?: string;
  decisionErrorOnce?: "character_trait_axis_source_unverified" | "character_trait_axis_integrity_invalid";
  axisCreated?: boolean;
  axisPostCount?: number;
};

function axis(state: MockState) {
  return {
    id: axisId, project_id: projectId, trait_type: state.scoped ? "behavior_boundary" : "core_personality", version: 1,
    display_name: "代签边界", definition: "是否未经同伴授权代其签名",
    definition_sha256: axisDefinitionHash,
    ...(state.scoped ? {
      comparison_key: "behavior_boundary:同伴签名",
      applicability_scope: state.axisScope ?? "需要同伴签字时",
      applicability_scope_sha256: "d".repeat(64),
    } : {}),
    positive_proposition: state.proposition,
    positive_proposition_sha256: state.proposition ? propositionHash : null,
  };
}

function candidate(state: MockState) {
  return {
    id: candidateId, character_key: characterId, trait_type: state.scoped ? "behavior_boundary" : "core_personality",
    ...(state.scoped ? { comparison_key: "behavior_boundary:同伴签名" } : {}),
    trait_key: "signature_integrity", value: "不会擅自代同伴签名",
    polarity: "positive", origin: "explicit_setting", confidence: 0.91,
    source_run_id: "run-axis", scope_sha256: "frozen-axis-snapshot", revision: 1,
    review_state: state.confirmed ? "confirmed" : "pending",
    reviewable: !state.confirmed, model_coverage: "full", source_verified: true,
    contexts: ["需要签字时"],
    limitations: ["未披露的紧急代签情境需要另行核对"],
    approved_axis_id: state.confirmed ? axisId : null,
    approved_axis_version: state.confirmed ? 1 : null,
    axis_alignment: state.mapped ? "opposite" : null,
    axis_polarity: state.mapped ? "negative" : null,
    axis_positive_proposition_sha256: state.mapped
      ? state.mappingHashMismatch ? "c".repeat(64) : propositionHash
      : null,
    evidence: [{
      input_id: "input-axis", document_id: "doc-axis", document_name: "人物设定.md",
      document_version: 1, document_role: "character_profile", publication_status: "published",
      authority_level: "canon", line_start: 1, line_end: 1, text: sourceLine,
      source_verified: true, source_text_exact: true, context_verified: false,
      ...(state.scoped ? { context_verified: true, story_scope: { timeline_key: "main" } } : {}),
    }],
    support_bindings_status: state.preciseEvidence ? "verified" : "legacy",
    support_bindings_v1: state.preciseEvidence ? {
      schema_version: "character-support-bindings-v1", index_version: "assertion-index-v1",
      bindings: [{
        evidence_index: 0, support_id: "L1:A2",
        target: { support_id: "L1:A2", start_offset: sourceContextEnd,
          end_offset: sourceTargetEnd, role: "target" },
        actor_anchor_id: "L1:A1", label_anchor_id: null,
        scope_relation: "same_actor_continuation",
        context: [{ support_id: "L1:A1", start_offset: 0,
          end_offset: sourceContextEnd, role: "actor_anchor" }],
      }],
    } : null,
  };
}

async function mockApi(page: Page, state: MockState) {
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = decodeURIComponent(url.pathname);
    const method = route.request().method();
    let body: unknown;
    if (path === "/api/v1/auth/me") body = {
      mode: "anonymous",
      user: { id: "author", email: "author@example.test", display_name: "作者" },
      workspace: { id: "workspace", name: "创作工作区", kind: "personal", role: "owner" },
    };
    else if (path === "/api/v1/project-catalog" && method === "GET") {
      const item = {
        id: projectId, name: "作者轴演示", description: "", active_document_count: 1,
        created_at: "2026-09-01T00:00:00Z", latest_run: null,
      };
      const page = Number(url.searchParams.get("page") || 1);
      const pageSize = Number(url.searchParams.get("page_size") || 40);
      const matches = (!url.searchParams.get("project_id") || url.searchParams.get("project_id") === projectId)
        && (!url.searchParams.get("query") || item.name.includes(url.searchParams.get("query")!.trim()));
      body = { page, page_size: pageSize, total: matches ? 1 : 0,
        items: matches && page === 1 ? [item] : [] };
    }
    else if (path === `${root}/documents`) body = [{
      id: "doc-axis", project_id: projectId, name: "人物设定.md", version: 1,
      active: true, created_at: "2026-09-01T00:00:00Z", content: sourceLine,
      document_role: "character_profile", story_scope: "主线", context_explicit: true,
      narrative_context: {
        resolution_state: "confirmed", publication_status: "published",
        document_role: "character_profile", scope: { schema_version: 1, timeline_key: "main" },
      },
    }];
    else if (path === `${root}/analysis-runs`) body = [{
      id: "run-axis", project_id: projectId, status: "completed",
      prompt_tokens: 1, completion_tokens: 1, estimated_cost_usd: 0,
      created_at: "2026-09-01T00:00:00Z",
    }];
    else if (path === "/api/v1/analysis-runs/run-axis") body = {
      id: "run-axis", project_id: projectId, status: "completed",
      prompt_tokens: 1, completion_tokens: 1, estimated_cost_usd: 0,
      created_at: "2026-09-01T00:00:00Z",
    };
    else if (path === "/api/v1/analysis-runs/run-axis/issues") body = [];
    else if (path === "/api/v1/analysis-runs/run-axis/records") body = { records: [], warnings: [] };
    else if (path === "/api/v1/analysis-runs/run-axis/diagnostics") body = {};
    else if (path === "/api/v1/analysis-runs/run-axis/clarifications") body = [];
    else if (path === "/api/v1/analysis-runs/run-axis/provisional-clues") body = { items: [], truncated: false };
    else if (path === "/api/v1/analysis-runs/run-axis/review-clues") body = { items: [], truncated: false, unavailable_count: 0 };
    else if (path === `${root}/characters`) body = {
      items: [{ id: characterId, canonical_name: characterId, aliases: [],
        confirmed_item_count: state.confirmed ? 1 : 0,
        pending_candidate_count: state.confirmed ? 0 : 1,
        drift_issue_count: 0, profile_revision: 1 }],
      page: 1, page_size: 30, total: 1, has_more: false,
      readiness: "ready", model_coverage: "full", source_run_id: "run-axis",
    };
    else if (path === characterRoot) body = {
      character: { id: characterId, canonical_name: characterId, aliases: [],
        confirmed_item_count: state.confirmed ? 1 : 0,
        pending_candidate_count: state.confirmed ? 0 : 1, drift_issue_count: 0 },
      profile_items: state.confirmed ? [candidate(state)] : [],
      model_coverage: "full", source_run_id: "run-axis",
    };
    else if (path === `${characterRoot}/profile-candidates`) body = {
      items: url.searchParams.get("state") === "withdrawn" || state.confirmed ? [] : [candidate(state)],
      page: 1, page_size: 20, total: state.confirmed ? 0 : 1, has_more: false,
      model_coverage: "full",
    };
    else if (path === candidateRoot) body = candidate(state);
    else if (path === `${candidateRoot}/source-neighbors`) body = {
      candidate_id: candidateId, character_key: characterId, source_run_id: "run-axis",
      limit: 20, offset: 0, total: 0, has_more: false, items: [],
      source_groups: [{ input_id: "input-axis", document_id: "doc-axis",
        document_name: "人物设定.md", document_version: 1,
        line_start: 1, line_end: 1, total: 0, context_verified: false }],
    };
    else if (path === `${root}/character-trait-axes` && method === "GET") {
      expect(url.searchParams.get("trait_type")).toBe("all");
      body = {
      items: state.scopedNoAxis && !state.axisCreated ? [] : state.scoped ? [axis(state), {
        ...axis(state), id: "wrong-object-axis", comparison_key: "behavior_boundary:陌生人",
      }, {
        ...axis(state), id: "wrong-dimension-axis", trait_type: "value",
        comparison_key: "value:同伴签名",
      }] : [axis(state)], total: state.scopedNoAxis && !state.axisCreated ? 0 : state.scoped ? 3 : 1, limit: 100, offset: 0,
      };
    }
    else if (path === `${root}/character-trait-axes` && method === "POST" && state.scoped) {
      state.axisPostCount = (state.axisPostCount || 0) + 1;
      expect(route.request().postDataJSON()).toEqual({
        trait_type: "behavior_boundary", display_name: "代签边界",
        definition: "是否未经同伴授权代其签名",
        positive_proposition: proposition,
        comparison_key: "behavior_boundary:同伴签名",
        applicability_scope: "需要同伴签字时",
      });
      state.axisCreated = true;
      await route.fulfill({ status: 201, json: axis(state) });
      return;
    }
    else if (path === `${root}/character-trait-axes/${axisId}` && method === "GET") body = axis(state);
    else if (path === `${root}/character-trait-axes/${axisId}/positive-proposition` && method === "POST") {
      state.propositionPostCount += 1;
      expect(route.request().postDataJSON()).toEqual({
        positive_proposition: proposition, expected_axis_version: 1,
      });
      if (state.conflictOnce) {
        state.conflictOnce = false;
        await route.fulfill({ status: 409, json: {
          detail: { code: "character_trait_axis_proposition_conflict", message: "并发更新" },
        } });
        return;
      }
      state.proposition = proposition;
      body = axis(state);
    }
    else if (path === `${candidateRoot}/decisions` && method === "POST") {
      state.decisionPostCount += 1;
      const input = route.request().postDataJSON();
      if (state.scoped && !input.approved_axis_id) {
        expect(input.scope_applicability_confirmed).toBeUndefined();
        expect(input.expected_axis_applicability_scope_sha256).toBeUndefined();
        state.confirmed = true;
        body = { candidate: candidate(state), decision_id: "review-unbound", deduplicated: false };
        await route.fulfill({ status: 200, json: body });
        return;
      }
      expect(input.approved_axis_id).toBe(axisId);
      expect(input.axis_alignment).toBe("opposite");
      expect(input.expected_axis_positive_proposition_sha256).toBe(propositionHash);
      if (state.scoped) {
        expect(input.expected_axis_applicability_scope_sha256).toBe("d".repeat(64));
        expect(input.scope_applicability_confirmed).toBe(true);
        if (state.decisionErrorOnce) {
          const code = state.decisionErrorOnce;
          state.decisionErrorOnce = undefined;
          await route.fulfill({ status: 409, json: { detail: { code, message: "binding rejected" } } });
          return;
        }
        if (state.scopeConflictOnce) {
          state.scopeConflictOnce = false;
          await route.fulfill({ status: 409, json: {
            detail: { code: "character_trait_axis_scope_conflict", message: "适用情境已变化" },
          } });
          return;
        }
      }
      state.confirmed = true;
      state.mapped = true;
      body = { candidate: candidate(state), decision_id: "review-axis", deduplicated: false };
    }
    else if (path === "/api/v1/account/model-provider") body = {};
    else {
      state.unexpected.push(`${method} ${path}`);
      await route.fulfill({ status: 404, json: { detail: "unexpected mocked endpoint" } });
      return;
    }
    await route.fulfill({ status: 200, json: body });
  });
}

test("legacy axis 409 offers in-page recovery without discarding the author's choice or text", async ({ page }) => {
  const state: MockState = {
    confirmed: false, mapped: false, proposition: null, conflictOnce: true,
    propositionPostCount: 0, decisionPostCount: 0, unexpected: [],
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/characters?section=candidates&page=1&character=${encodeURIComponent(characterId)}&candidate=${candidateId}`);
  const axisSelect = page.getByLabel("项目内已有轴");
  await expect(axisSelect).toBeVisible();
  await axisSelect.selectOption(axisId);
  const input = page.getByLabel("为旧作者轴补写正向命题");
  await input.fill(proposition);
  await page.getByRole("button", { name: "保存正向命题" }).click();
  await expect(page.getByRole("button", { name: "刷新轴列表核对" })).toBeVisible();
  await expect(input).toHaveValue(proposition);
  await expect(axisSelect).toHaveValue(axisId);
  expect(state.decisionPostCount).toBe(0);

  await page.getByRole("button", { name: "刷新轴列表核对" }).click();
  await expect(page.getByRole("button", { name: "我已核对刷新后的轴定义与命题" })).toBeVisible();
  await expect(input).toHaveValue(proposition);
  await expect(axisSelect).toHaveValue(axisId);
  await page.getByRole("button", { name: "我已核对刷新后的轴定义与命题" }).click();
  await page.getByRole("button", { name: "保存正向命题" }).click();
  await expect(page.getByText(`作者比较句（轴正向）：${proposition}`, { exact: true })).toBeVisible();
  await page.getByLabel(/反向：模型标签的正向状态与作者比较句相反/).check();
  await expect(page.getByText(/映射预览（未经事实复核）：若采用你的方向选择，这条候选表示“作者比较句不成立”/)).toBeVisible();
  await page.getByRole("button", { name: "确认归纳" }).click();
  await expect(page.getByText(/归纳已确认并写入角色档案/)).toBeVisible();
  expect(state.propositionPostCount).toBe(2);
  expect(state.decisionPostCount).toBe(1);
  expect(state.unexpected).toEqual([]);
});

test("scoped boundary review requires author applicability and clears it after a 409 refresh", async ({ page }) => {
  const state: MockState = {
    confirmed: false, mapped: false, proposition, preciseEvidence: true,
    scoped: true, scopeConflictOnce: true,
    propositionPostCount: 0, decisionPostCount: 0, unexpected: [],
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/characters?section=candidates&page=1&character=${encodeURIComponent(characterId)}&candidate=${candidateId}`);
  const detail = page.getByRole("region", { name: "归纳证据详情" });
  const axisSelect = detail.getByLabel("项目内已有轴");
  await expect(axisSelect).toBeVisible();
  await expect(axisSelect.locator("option")).toHaveCount(2);
  await axisSelect.selectOption(axisId);
  await expect(detail.locator(".candidateAxisDefinition").getByText(/适用情境：需要同伴签字时/)).toBeVisible();
  await detail.getByRole("radio", { name: /反向：模型标签的正向状态与作者比较句相反/ }).check();
  await detail.getByRole("button", { name: "确认归纳" }).click();
  const scopeAck = detail.getByRole("checkbox", { name: /我已根据上方原文判断/ });
  await page.setViewportSize({ width: 375, height: 900 });
  const panelBox = await detail.getByRole("region", { name: "作者批准比较轴" }).boundingBox();
  expect(panelBox).not.toBeNull();
  expect(panelBox!.x).toBeGreaterThanOrEqual(0);
  expect(panelBox!.x + panelBox!.width).toBeLessThanOrEqual(376);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(376);
  await expect(scopeAck).toBeFocused();
  await expect(detail.getByText(/请根据原文判断这一情境确实适用/)).toBeVisible();
  expect(state.decisionPostCount).toBe(0);
  await scopeAck.check();
  await detail.getByRole("button", { name: "确认归纳" }).click();
  await expect(page.getByText(/作者轴适用情境已变化/)).toBeVisible();
  await expect(axisSelect).toHaveValue("");
  await expect(scopeAck).toHaveCount(0);
  await axisSelect.selectOption(axisId);
  await expect(scopeAck).not.toBeChecked();
  await detail.getByRole("radio", { name: /反向：模型标签的正向状态与作者比较句相反/ }).check();
  await scopeAck.check();
  await detail.getByRole("button", { name: "确认归纳" }).click();
  await expect(page.getByText(/归纳与适用情境已由作者确认/)).toBeVisible();
  expect(state.decisionPostCount).toBe(2);
  expect(state.unexpected).toEqual([]);
});

test("scoped candidate can still be confirmed as an ordinary unbound profile item", async ({ page }) => {
  const state: MockState = {
    confirmed: false, mapped: false, proposition, preciseEvidence: true,
    scoped: true, propositionPostCount: 0, decisionPostCount: 0, unexpected: [],
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/characters?section=candidates&page=1&character=${encodeURIComponent(characterId)}&candidate=${candidateId}`);
  await expect(page.getByText(/若不绑定比较轴，仍可确认普通角色特征/)).toBeVisible();
  await page.getByRole("button", { name: "确认归纳" }).click();
  await expect(page.getByText(/归纳已确认并写入角色档案/)).toBeVisible();
  expect(state.decisionPostCount).toBe(1);
  expect(state.unexpected).toEqual([]);
});

for (const [code, expectedMessage] of [
  ["character_trait_axis_source_unverified", /缺少可核对的模型来源或精确原文证据/],
  ["character_trait_axis_integrity_invalid", /作者轴定义未通过完整性核对/],
] as const) {
  test(`scoped binding explains ${code} and retains the pending candidate`, async ({ page }) => {
    const longScope = "需要同伴签字且持续存在授权边界".repeat(12);
    const state: MockState = {
      confirmed: false, mapped: false, proposition, preciseEvidence: true,
      scoped: true, axisScope: longScope, decisionErrorOnce: code,
      propositionPostCount: 0, decisionPostCount: 0, unexpected: [],
    };
    await mockApi(page, state);
    await page.goto(`/app/projects/${projectId}/characters?section=candidates&page=1&character=${encodeURIComponent(characterId)}&candidate=${candidateId}`);
    const detail = page.getByRole("region", { name: "归纳证据详情" });
    const axisSelect = detail.getByLabel("项目内已有轴");
    await expect(axisSelect.locator(`option[value="${axisId}"]`)).toContainText("适用：");
    await axisSelect.selectOption(axisId);
    const acknowledgment = detail.locator(".candidateAxisScopeAcknowledgment");
    await page.setViewportSize({ width: 375, height: 900 });
    await expect(acknowledgment).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(376);
    await detail.getByRole("radio", { name: /反向：模型标签的正向状态与作者比较句相反/ }).check();
    await detail.getByRole("checkbox", { name: /我已根据上方原文判断/ }).check();
    await detail.getByRole("button", { name: "确认归纳" }).click();
    await expect(page.getByText(expectedMessage)).toBeVisible();
    expect(state.confirmed).toBe(false);
    expect(state.decisionPostCount).toBe(1);
    expect(state.unexpected).toEqual([]);
  });
}

test("author can create an object-scoped boundary axis and then confirm its situation separately", async ({ page }) => {
  const state: MockState = {
    confirmed: false, mapped: false, proposition, preciseEvidence: true,
    scoped: true, scopedNoAxis: true,
    propositionPostCount: 0, decisionPostCount: 0, unexpected: [],
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/characters?section=candidates&page=1&character=${encodeURIComponent(characterId)}&candidate=${candidateId}`);
  await page.getByRole("button", { name: "创建新轴" }).click();
  await page.getByLabel("轴名称").fill("代签边界");
  await page.getByLabel("轴定义").fill("是否未经同伴授权代其签名");
  await page.getByLabel("轴的正向命题").fill(proposition);
  await page.getByLabel("轴的适用情境").fill("需要同伴签字时");
  await page.getByRole("button", { name: "先创建作者轴" }).click();
  await expect(page.getByText(/已创建并选中/)).toBeVisible();
  expect(state.axisPostCount).toBe(1);
  await page.getByRole("radio", { name: /反向：模型标签的正向状态与作者比较句相反/ }).check();
  const scopeAck = page.getByRole("checkbox", { name: /我已根据上方原文判断/ });
  await expect(scopeAck).not.toBeChecked();
  await scopeAck.check();
  await page.getByRole("button", { name: "确认归纳" }).click();
  await expect(page.getByText(/归纳与适用情境已由作者确认/)).toBeVisible();
  expect(state.decisionPostCount).toBe(1);
  expect(state.unexpected).toEqual([]);
});

test("candidate review leads with exact source and conditions without treating positive as a virtue", async ({ page }, testInfo) => {
  const state: MockState = {
    confirmed: false, mapped: false, proposition, preciseEvidence: true,
    propositionPostCount: 0, decisionPostCount: 0, unexpected: [],
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/characters?section=candidates&page=1&character=${encodeURIComponent(characterId)}&candidate=${candidateId}`);
  const detail = page.getByRole("region", { name: "归纳证据详情" });
  const preview = detail.getByRole("region", { name: "冻结原文目标句" });
  await expect(preview.locator("blockquote")).toHaveText(sourceTarget);
  await expect(preview.getByText(sourceContext)).toBeVisible();
  const conditions = detail.getByRole("region", { name: "适用条件与例外" });
  await expect(conditions.getByText("需要签字时")).toBeVisible();
  await expect(conditions.getByText("未披露的紧急代签情境需要另行核对")).toBeVisible();
  await expect(detail.getByText(/positive（相对模型标签的正向）/).first()).toBeVisible();
  await expect(detail.getByText(/不能按“不会”等字眼自动反转/)).toBeVisible();
  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await preview.scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath("candidate-review-desktop.png") });
  }
  const axisSelect = detail.getByLabel("项目内已有轴");
  await axisSelect.focus();
  await axisSelect.press("ArrowDown");
  await expect(axisSelect).toHaveValue(axisId);
  const opposite = detail.getByRole("radio", { name: /反向：模型标签的正向状态与作者比较句相反/ });
  await opposite.focus();
  await opposite.press("Space");
  await expect(opposite).toBeChecked();
  await expect(detail.getByText(/作者比较句不成立/)).toBeVisible();
  expect(state.decisionPostCount).toBe(0);
  await page.setViewportSize({ width: 375, height: 1200 });
  await expect(detail).toBeVisible();
  const widths = await page.evaluate(() => ({
    viewport: window.innerWidth, scroll: document.documentElement.scrollWidth,
  }));
  expect(widths.scroll).toBeLessThanOrEqual(widths.viewport + 1);
  await expect(preview.locator("blockquote")).toHaveText(sourceTarget);
  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await preview.scrollIntoViewIfNeeded();
    await page.screenshot({ path: testInfo.outputPath("candidate-review-375.png") });
  }
  expect(state.unexpected).toEqual([]);
});

test("an already mapped profile keeps its proposition and frozen evidence inspectable on desktop and mobile", async ({ page }, testInfo) => {
  const state: MockState = {
    confirmed: true, mapped: true, proposition, propositionPostCount: 0,
    decisionPostCount: 0, unexpected: [],
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/characters?section=profile&page=1&character=${encodeURIComponent(characterId)}`);
  const open = page.getByRole("button", { name: "查看作者轴方向与证据" });
  await expect(open).toBeVisible();
  await open.click();
  const panel = page.getByRole("region", { name: "补认作者轴方向" });
  await expect(panel.getByText(`作者比较句（轴正向）：${proposition}`)).toBeVisible();
  await expect(panel.getByRole("blockquote")).toHaveText(sourceLine);
  await expect(panel.getByText(/作者已标注比较方向：反向/)).toBeVisible();
  await expect(panel.getByText(/按已确认映射，作者比较句不成立/)).toBeVisible();
  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await panel.screenshot({
      path: testInfo.outputPath("axis-alignment-desktop.png"),
    });
  }
  await page.setViewportSize({ width: 360, height: 1200 });
  await expect(panel).toBeVisible();
  const panelBounds = await panel.boundingBox();
  expect(panelBounds).not.toBeNull();
  expect(panelBounds!.x).toBeGreaterThanOrEqual(0);
  expect(panelBounds!.x + panelBounds!.width).toBeLessThanOrEqual(361);
  const width = await page.evaluate(() => ({
    viewport: window.innerWidth, scroll: document.documentElement.scrollWidth,
  }));
  expect(width.scroll).toBeLessThanOrEqual(width.viewport + 1);
  if (process.env.LOREGUARD_E2E_VISUAL_QA === "1") {
    await panel.scrollIntoViewIfNeeded();
    await panel.screenshot({
      path: testInfo.outputPath("axis-alignment-360.png"),
    });
  }
  await panel.getByRole("button", { name: "关闭补认" }).click();
  await expect(open).toBeVisible();
  await expect(open).toBeFocused();
  await page.goto(`/app/projects/${projectId}/characters?section=candidates&page=1&character=${encodeURIComponent(characterId)}&candidate=${candidateId}`);
  await expect(page.getByRole("region", { name: "归纳证据详情" }).getByText(/按已确认映射，作者比较句不成立/)).toBeVisible();
  expect(state.propositionPostCount).toBe(0);
  expect(state.decisionPostCount).toBe(0);
  expect(state.unexpected).toEqual([]);
});

test("a confirmed direction with a changed axis proposition does not guess its mapped meaning", async ({ page }) => {
  const state: MockState = {
    confirmed: true, mapped: true, proposition, mappingHashMismatch: true,
    propositionPostCount: 0, decisionPostCount: 0, unexpected: [],
  };
  await mockApi(page, state);
  await page.goto(`/app/projects/${projectId}/characters?section=profile&page=1&character=${encodeURIComponent(characterId)}`);
  await page.getByRole("button", { name: "查看作者轴方向与证据" }).click();
  const panel = page.getByRole("region", { name: "补认作者轴方向" });
  await expect(panel.getByText(/不展示成立\/不成立的换算/)).toBeVisible();
  await expect(panel.getByText(/按已确认映射，作者比较句/)).toHaveCount(0);
  await page.goto(`/app/projects/${projectId}/characters?section=candidates&page=1&character=${encodeURIComponent(characterId)}&candidate=${candidateId}`);
  const detail = page.getByRole("region", { name: "归纳证据详情" });
  await expect(detail.getByText(/当前轴命题或冻结证据不可核对，暂不换算成立\/不成立/)).toBeVisible();
  await expect(detail.getByText(/按已确认映射，作者比较句/)).toHaveCount(0);
  expect(state.unexpected).toEqual([]);
});
