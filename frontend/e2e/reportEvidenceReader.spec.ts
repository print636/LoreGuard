import { expect, test, type Page } from "@playwright/test";

const projectId = "evidence-reader-project";
const otherProjectId = "evidence-reader-other-project";
const runId = "run-old-a";
const nextRunId = "run-next-b";
const otherRunId = "run-other-c";
const baselineRunId = "trait-source-old-run";
const primaryId = "issue-primary";
const traitIssueId = "issue-trait-source";
const reviewClueId = "71475dcc-1d0c-4dd2-a014-06941adf1bf3";
const provisionalClueId = `pc_${"a".repeat(32)}`;
const now = "2026-10-02T00:00:00Z";
const screenshotFolder = `${process.env.LOREGUARD_E2E_SCREENSHOT_DIR ?? "../artifacts/playwright/screenshots"}/report-evidence-v1`;
type EvidenceKind = "issue" | "review_clue" | "provisional_clue";
type Evidence = { document_id: string; document_name: string; line_start: number; line_end: number; text: string };

function draftEvidence(activeRunId = runId): Evidence {
  return {
    document_id: activeRunId === runId ? "draft-frozen-v1" : "draft-current-v2",
    document_name: "北港新稿.md", line_start: 130, line_end: 130,
    text: activeRunId === runId ? "冻结V1证据：林澈把北港密钥交给守门人。"
      : activeRunId === nextRunId ? "第二次运行V2证据：林澈收回北港密钥。" : "另一项目冻结证据：青岚独自守门。",
  };
}

const traitEvidence: Evidence = {
  document_id: "profile-shared-id", document_name: "林澈档案.md", line_start: 8, line_end: 8,
  text: "旧特征来源V1：林澈面对陌生人时通常沉默寡言。",
};

function primaryIssue(activeRunId = runId) {
  return {
    id: primaryId, category: "fact_conflict", severity: "medium", confidence: 0.8,
    title: "密钥位置需要核对", explanation: "请核对本次分析的两侧原文。", suggestion: "结合前后剧情确认。",
    evidence: [draftEvidence(activeRunId)], metadata: {},
  };
}

const traitIssue = {
  id: traitIssueId, category: "character_drift", severity: "medium", confidence: 0.75,
  title: "角色档案与新稿表现需要核对", explanation: "正式问题仍需要作者核对原文。", suggestion: "核对档案与新稿上下文。",
  evidence: [traitEvidence, draftEvidence()], metadata: {},
};
const reviewClue = {
  id: reviewClueId, report_class: "review_clue", category: "character_drift", severity: "low", confidence: 0.52,
  title: "一次不同表现仍待作者确认", explanation: "单次行为不能独立确认角色变化。", suggestion: "请对照前后原文。",
  evidence: [traitEvidence, draftEvidence()],
  metadata: { final_outcome: "needs_confirmation", review_reason: "single_behavior_is_not_drift" },
};
const provisionalClue = {
  id: provisionalClueId, document_id: "draft-frozen-v1", document_version: 1, document_name: "北港新稿.md",
  line_start: 130, line_end: 130, evidence: draftEvidence().text, character: "林澈", dimension: "behavior_boundary",
  proposed_statement: "林澈的本次表现尚未验证。", reason: "partial_model_package",
};

function run(activeRunId = runId) {
  const old = activeRunId === runId;
  return {
    id: activeRunId, project_id: activeRunId === otherRunId ? otherProjectId : projectId,
    status: "completed", created_at: now, prompt_tokens: 12, completion_tokens: 4, estimated_cost_usd: 0,
    input_snapshot_available: true,
    input_documents: [
      { document_id: old ? "draft-frozen-v1" : "draft-current-v2", document_name: "北港新稿.md",
        document_version: old ? 1 : 2, document_role: "chapter", story_scope: "main", content_sha256: "a".repeat(64), batch_role: "target" },
      { document_id: "profile-shared-id", document_name: "林澈档案.md", document_version: 2,
        document_role: "character_profile", story_scope: "main", content_sha256: "b".repeat(64), batch_role: "background" },
    ],
    review_batch: { mode: "draft_review", no_formal_context_expected: false,
      target_document_ids: [old ? "draft-frozen-v1" : "draft-current-v2"], background_document_ids: ["profile-shared-id"] },
  };
}

function documents(activeProjectId: string) {
  return [
    { id: "draft-current-v2", project_id: activeProjectId, name: "北港新稿.md", version: 2, active: true,
      created_at: now, document_role: "chapter", story_scope: "main", narrative_context: {
        revision: 1, resolution_state: "confirmed", publication_status: "draft", origin: "explicit",
        scope: { schema_version: 1, timeline_key: "main" },
      } },
    { id: "profile-shared-id", project_id: activeProjectId, name: "林澈档案.md", version: 2, active: true,
      created_at: now, document_role: "character_profile", story_scope: "main", narrative_context: {
        revision: 1, resolution_state: "confirmed", publication_status: "published", origin: "explicit",
        scope: { schema_version: 1, timeline_key: "main" },
      } },
  ];
}

function initialState() {
  return {
    previews: [] as URL[], requests: [] as string[], writes: [] as string[], unexpected: [] as string[],
    failures: 0, excerptOnly: false,
    delayedRunId: "", releaseDelayed: undefined as (() => void) | undefined, delayedSettled: false,
  };
}
type MockState = ReturnType<typeof initialState>;

function previewPayload(url: URL, activeRunId: string, state: MockState) {
  const kind = url.searchParams.get("kind") as EvidenceKind;
  const itemId = url.searchParams.get("item_id") || "";
  const evidenceIndex = Number(url.searchParams.get("evidence_index") || 0);
  const traitSource = (kind === "issue" && itemId === traitIssueId && evidenceIndex === 0)
    || (kind === "review_clue" && evidenceIndex === 0);
  const evidence = traitSource ? traitEvidence : draftEvidence(activeRunId);
  const sourceRunId = traitSource ? baselineRunId : activeRunId;
  const lineCount = traitSource ? 24 : 300;
  const sourceLines = Array.from({ length: lineCount }, (_, index) =>
    traitSource ? `旧特征来源V1第 ${index + 1} 行：作者已保存的角色档案。`
      : `${activeRunId === runId ? "冻结V1" : activeRunId === nextRunId ? "第二次运行V2" : "另一项目冻结"}第 ${index + 1} 行：原创剧情上下文。`);
  sourceLines[evidence.line_start - 1] = evidence.text;
  if (!traitSource) sourceLines[135] = `故事标识：${"LongUnbrokenFrozenStoryIdentifier".repeat(14)}`;
  const excerpt = state.excerptOnly && traitSource;
  const limit = Number(url.searchParams.get("limit") || 120);
  const offset = Number(url.searchParams.get("offset") ?? Math.max(0, evidence.line_start - 1 - 12));
  return {
    run: { id: activeRunId, project_id: activeRunId === otherRunId ? otherProjectId : projectId },
    reference: { kind, item_id: itemId, evidence_index: evidenceIndex, ...evidence },
    source: {
      availability: excerpt ? "excerpt_only" : "full_context",
      provenance: traitSource ? "confirmed_trait_source" : "run_input",
      source_run_id: excerpt ? null : sourceRunId,
      document_version: traitSource || activeRunId === runId ? 1 : 2,
      content_sha256: (traitSource ? "c" : "a").repeat(64),
      char_count: excerpt ? null : sourceLines.join("\n").length, line_count: excerpt ? null : sourceLines.length,
    },
    lines: excerpt ? [] : sourceLines.slice(offset, offset + limit).map((text, index) => ({
      line_number: offset + index + 1, text,
      is_evidence: offset + index + 1 >= evidence.line_start && offset + index + 1 <= evidence.line_end,
    })),
    page: excerpt ? null : { offset, limit, total: sourceLines.length, has_more: offset + limit < sourceLines.length },
    message: excerpt ? "完整原文快照已缺失，仅可查看已冻结的证据摘录。" : null,
  };
}

async function mockApi(page: Page, state: MockState) {
  await page.route("**/api/v1/**", async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const method = route.request().method();
    state.requests.push(`${method} ${path}`);
    if (method !== "GET") state.writes.push(`${method} ${path}`);
    let body: unknown;
    if (path === "/api/v1/auth/me") body = {
      mode: "anonymous", user: { id: "evidence-author", email: "", display_name: "作者" },
      workspace: { id: "evidence-workspace", name: "剧情创作工作区", kind: "personal", role: "owner" },
    };
    else if (path === "/api/v1/account/model-provider") body = { configured: false, revision: 0 };
    else if (path === "/api/v1/project-catalog") {
      const all = [
        { id: projectId, name: "北港游戏剧情", description: "", created_at: now, active_document_count: 2, latest_run: run() },
        { id: otherProjectId, name: "另一部独立故事", description: "", created_at: now, active_document_count: 2, latest_run: run(otherRunId) },
      ];
      const filtered = all.filter((item) => !url.searchParams.get("project_id") || item.id === url.searchParams.get("project_id"));
      body = { page: 1, page_size: 40, total: filtered.length, items: filtered };
    }
    else if (/^\/api\/v1\/projects\/[^/]+\/documents$/.test(path)) body = documents(path.split("/")[4]);
    else if (/^\/api\/v1\/projects\/[^/]+\/analysis-runs$/.test(path)) {
      body = path.includes(otherProjectId) ? [run(otherRunId)] : [run(), run(nextRunId)];
    }
    else if ([projectId, otherProjectId].some((id) => path === `/api/v1/projects/${id}/run-catalog`) && method === "GET") {
      const activeProjectId = path.split("/")[4];
      const pageNumber = Number(url.searchParams.get("page") || 1);
      const pageSize = Number(url.searchParams.get("page_size") || 20);
      const status = url.searchParams.get("status") || "all";
      const all = (activeProjectId === otherProjectId ? [run(otherRunId)] : [run(), run(nextRunId)])
        .filter((item) => status === "all" || item.status === status)
        .sort((left, right) => right.created_at.localeCompare(left.created_at) || right.id.localeCompare(left.id));
      const offset = (pageNumber - 1) * pageSize;
      body = { project_id: activeProjectId, page: pageNumber, page_size: pageSize, total: all.length,
        has_more: offset + pageSize < all.length,
        items: all.slice(offset, offset + pageSize).map((item) => ({
          id: item.id, project_id: item.project_id, status: item.status, created_at: item.created_at,
          started_at: null, completed_at: now, input_chars: 0, prompt_tokens: item.prompt_tokens,
          completion_tokens: item.completion_tokens, estimated_cost_usd: item.estimated_cost_usd,
          frozen_document_count: item.input_documents.length, retried_from: null, batch_mode: item.review_batch.mode,
        })) };
    }
    else if (path.endsWith("/narrative-context")) {
      const parts = path.split("/");
      const document = documents(parts[4]).find((item) => item.id === parts.at(-2));
      body = { document_id: document?.id, current: document?.narrative_context, revision_count: 1 };
    }
    else if (path.startsWith("/api/v1/issues/") && path.endsWith("/feedback") && method === "GET") body = { latest: null };
    else if (path.startsWith("/api/v1/analysis-runs/")) {
      const parts = path.split("/");
      const activeRunId = parts[4];
      const endpoint = parts[5];
      if (!endpoint) body = run(activeRunId);
      else if (endpoint === "issues") body = activeRunId === runId ? [primaryIssue(), traitIssue] : [primaryIssue(activeRunId)];
      else if (endpoint === "records") body = { records: [], warnings: [] };
      else if (endpoint === "diagnostics") body = {};
      else if (endpoint === "clarifications") body = [];
      else if (endpoint === "review-clues") body = { items: activeRunId === runId ? [reviewClue] : [], truncated: false, unavailable_count: 0 };
      else if (endpoint === "provisional-clues") body = { items: activeRunId === runId ? [provisionalClue] : [], truncated: false };
      else if (endpoint === "evidence-preview") {
        state.previews.push(url);
        if (state.failures > 0) {
          state.failures -= 1;
          await route.fulfill({ status: 503, json: { detail: "冻结正文暂时无法读取" } });
          return;
        }
        if (state.delayedRunId === activeRunId) {
          await new Promise<void>((resolve) => { state.releaseDelayed = resolve; });
          try { await route.fulfill({ json: previewPayload(url, activeRunId, state) }); } catch { /* Cancelled old selection. */ }
          state.delayedSettled = true;
          return;
        }
        body = previewPayload(url, activeRunId, state);
      }
      else {
        state.unexpected.push(`${method} ${path}`);
        await route.fulfill({ status: 404, json: { detail: "unexpected mocked endpoint" } });
        return;
      }
    }
    else {
      state.unexpected.push(`${method} ${path}`);
      await route.fulfill({ status: 404, json: { detail: "unexpected mocked endpoint" } });
      return;
    }
    await route.fulfill({ json: body });
  });
}

async function openReport(page: Page, state: MockState, activeRunId = runId, query = "") {
  await mockApi(page, state);
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto(`/app/projects/${activeRunId === otherRunId ? otherProjectId : projectId}/runs/${activeRunId}/report${query}`);
  await expect(page.locator(".issues")).toContainText(primaryIssue().title);
}

function primaryTrigger(page: Page) {
  return page.locator(`#issue-${primaryId}`).getByRole("button", { name: "查看分析时原文", exact: true });
}

function assertReadingOnly(state: MockState) {
  expect(state.writes, "阅读冻结原文不得写反馈、启动分析或调用Provider").toEqual([]);
  expect(state.unexpected).toEqual([]);
  expect(state.requests.some((request) => /\/documents\/[^/]+\/preview$/.test(request)), "不得使用当前文档preview冒充分析时原文").toBe(false);
  for (const url of state.previews) {
    expect(Array.from(url.searchParams.keys()).every((key) => ["kind", "item_id", "evidence_index", "offset", "limit"].includes(key))).toBe(true);
  }
}

test("作者查看旧报告冻结V1，关闭后保留筛选、选中问题、反馈草稿和触发焦点", async ({ page }) => {
  const state = initialState();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await openReport(page, state, runId, `?category=fact_conflict&status=unreviewed&issue=${primaryId}`);
  const article = page.locator(`#issue-${primaryId}`);
  const note = article.locator(".note");
  await note.fill("我还需核对守门人前一段的对白，暂未提交。");
  const trigger = primaryTrigger(page);
  await trigger.focus();
  await trigger.press("Enter");
  const dialog = page.getByRole("dialog", { name: "分析时原文" });
  await expect(dialog).toBeVisible();
  await expect(dialog.getByRole("button", { name: "关闭阅读面板" })).toBeFocused();
  await expect(dialog.locator(".evidenceReaderMetadata")).toContainText("冻结版本 v1");
  await expect(dialog.locator(".evidenceReaderMetadata")).toContainText("本次分析的冻结输入");
  await expect(dialog.locator(".evidenceReaderLines .referenced")).toContainText(draftEvidence().text);
  await expect(dialog).not.toContainText("第二次运行V2证据");
  await expect(dialog).toContainText("文稿后续修改不会替换这里的原文");
  expect(documents(projectId)[0].version).toBe(2);
  expect(state.previews.at(-1)?.searchParams.get("kind")).toBe("issue");
  expect(state.previews.at(-1)?.searchParams.get("item_id")).toBe(primaryId);
  expect(state.previews.at(-1)?.searchParams.get("evidence_index")).toBe("0");
  await dialog.press("Escape");
  await expect(dialog).not.toBeVisible();
  await expect(trigger).toBeFocused();
  await expect(note).toHaveValue("我还需核对守门人前一段的对白，暂未提交。");
  await expect(page.locator(".reportFilters").getByRole("combobox", { name: "问题类别" })).toHaveValue("fact_conflict");
  await expect(page.locator(".reportFilters").getByRole("combobox", { name: "反馈状态" })).toHaveValue("unreviewed");
  await expect(article.getByRole("button", { name: "当前选中问题" })).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator(".issues article")).toHaveCount(1);
  await expect(page).toHaveURL(new RegExp(`category=fact_conflict&status=unreviewed&issue=${primaryId}$`));
  assertReadingOnly(state);
});

test("剧情策划核对旧特征来源与两类待复核证据，阅读不会升级线索或模型提案", async ({ page }) => {
  const state = initialState();
  await openReport(page, state);
  const dialog = page.getByRole("dialog", { name: "分析时原文" });
  const traitTrigger = page.locator(`#issue-${traitIssueId}`).getByRole("button", { name: "查看分析时原文" }).first();
  await traitTrigger.click();
  await expect(dialog.locator(".evidenceReaderMetadata")).toContainText("冻结版本 v1");
  await expect(dialog.locator(".evidenceReaderMetadata")).toContainText("既有角色设定的冻结来源");
  const headerSummary = dialog.locator(".evidenceReaderHeading .evidenceReaderHeaderSummary");
  await expect(headerSummary).toHaveText("冻结 v1 · 既有设定 · 正式问题");
  await expect(dialog.locator(`code[title="${baselineRunId}"]`)).toHaveCount(1);
  await expect(dialog.locator(".evidenceReaderLines .referenced")).toContainText(traitEvidence.text);
  expect(run().input_documents[1].document_id).toBe(traitEvidence.document_id);
  expect(run().input_documents[1].document_version).toBe(2);
  await dialog.getByRole("button", { name: "关闭阅读面板" }).click();
  const clues = page.getByRole("region", { name: /待复核线索/ });
  await expect(clues.getByRole("heading", { name: "待复核线索 2" })).toBeVisible();
  await expect(page.locator(".issues article")).toHaveCount(2);
  const reviewTrigger = clues.locator(".reviewClueList").getByRole("button", { name: "查看分析时原文" }).first();
  await reviewTrigger.click();
  await expect(dialog).toContainText("角色审查线索的引用（待复核）");
  await expect(headerSummary).toHaveText("冻结 v1 · 既有设定 · 待复核");
  await expect(headerSummary).toBeInViewport({ ratio: 1 });
  await expect(dialog).toContainText("阅读不会改变线索或提案的审查结论");
  await expect(dialog.locator(".evidenceReaderLines .referenced")).toContainText(traitEvidence.text);
  expect(state.previews.at(-1)?.searchParams.get("kind")).toBe("review_clue");
  await dialog.press("Escape");
  await expect(reviewTrigger).toBeFocused();
  const proposalTrigger = clues.locator(".provisionalClueList:not(.reviewClueList)").getByRole("button", { name: "查看分析时原文" });
  await proposalTrigger.click();
  await expect(dialog).toContainText("模型提案的引用（未验证）");
  await expect(headerSummary).toHaveText("冻结 v1 · 本次输入 · 未验证");
  await expect(headerSummary).toBeInViewport({ ratio: 1 });
  await expect(dialog.locator(".evidenceReaderLines .referenced")).toContainText(draftEvidence().text);
  expect(state.previews.at(-1)?.searchParams.get("kind")).toBe("provisional_clue");
  await expect(dialog.getByRole("button", { name: /已接受|误报|已解决|导出/ })).toHaveCount(0);
  await expect(dialog.getByRole("textbox")).toHaveCount(0);
  await expect(dialog.getByRole("searchbox")).toHaveCount(0);
  await dialog.press("Escape");
  await expect(proposalTrigger).toBeFocused();
  await expect(clues.getByRole("heading", { name: "待复核线索 2" })).toBeVisible();
  await expect(page.locator(".issues article")).toHaveCount(2);
  await expect(clues.getByRole("button", { name: /已接受|误报|已解决|导出/ })).toHaveCount(0);
  assertReadingOnly(state);
});

test("分页可返回引用位置；读取失败可重试；只剩冻结摘录时不能扩展", async ({ page }) => {
  const state = initialState();
  state.failures = 1;
  await openReport(page, state);
  await primaryTrigger(page).click();
  const dialog = page.getByRole("dialog", { name: "分析时原文" });
  await expect(dialog.getByRole("alert")).toContainText("服务器暂时无法读取冻结原文");
  await expect(dialog.locator(".evidenceReaderExcerpt blockquote")).toContainText(draftEvidence().text);
  await expect(dialog.locator(".evidenceReaderLines > li")).toHaveCount(0);
  const retry = dialog.getByRole("button", { name: "重试读取原文" });
  await retry.focus();
  await retry.press("Enter");
  const lines = dialog.locator(".evidenceReaderLines > li");
  await expect(lines).toHaveCount(120);
  await expect(lines.first()).toHaveAttribute("data-evidence-line", "118");
  await expect(dialog.locator('[data-evidence-line="130"]')).toHaveClass("referenced");
  await dialog.getByRole("button", { name: "下一页原文" }).click();
  await expect(lines).toHaveCount(63);
  await expect(lines.first()).toHaveAttribute("data-evidence-line", "238");
  await expect(dialog.getByRole("button", { name: "下一页原文" })).toBeDisabled();
  await dialog.getByRole("button", { name: "返回引用位置" }).click();
  await expect(lines.first()).toHaveAttribute("data-evidence-line", "118");
  await expect(dialog.locator('[data-evidence-line="130"]')).toBeVisible();
  await expect(dialog.getByRole("region", { name: "冻结原文阅读区" })).toBeFocused();
  await dialog.press("Escape");

  state.excerptOnly = true;
  const traitTrigger = page.locator(`#issue-${traitIssueId}`).getByRole("button", { name: "查看分析时原文" }).first();
  await traitTrigger.click();
  await expect(dialog).toContainText("只有冻结摘录，完整来源原文不可用");
  await expect(dialog).toContainText("无法向前后扩展阅读");
  await expect(dialog.locator(".evidenceReaderExcerpt blockquote")).toContainText(traitEvidence.text);
  await expect(dialog.locator(".evidenceReaderLines > li")).toHaveCount(0);
  await expect(dialog.getByRole("button", { name: /上一页原文|下一页原文|返回引用位置/ })).toHaveCount(0);
  await expect(dialog.locator(".evidenceReaderMetadata")).toContainText("冻结版本 v1");
  await dialog.press("Escape");
  await expect(traitTrigger).toBeFocused();
  assertReadingOnly(state);
});

test("切换运行与项目会清除旧阅读面板，迟到原文不能覆盖新报告", async ({ page }) => {
  const state = initialState();
  state.delayedRunId = runId;
  await openReport(page, state, nextRunId);
  await page.locator(".sideNav").getByRole("button", { name: "运行审计", exact: true }).click();
  const history = page.getByRole("region", { name: "运行历史", exact: true });
  await history.getByRole("row").filter({ has: page.locator(`code[title="${runId}"]`) })
    .getByRole("button", { name: /^查看报告/ }).click();
  await expect(page).toHaveURL(new RegExp(`/runs/${runId}/report`));
  await primaryTrigger(page).click();
  const dialog = page.getByRole("dialog", { name: "分析时原文" });
  await expect(dialog).toContainText("正在读取这条证据的冻结原文");
  await expect.poll(() => Boolean(state.releaseDelayed)).toBe(true);
  await page.goBack();
  await expect(dialog).not.toBeVisible();
  await page.goBack();
  await expect(page).toHaveURL(new RegExp(`/runs/${nextRunId}/report`));
  await primaryTrigger(page).click();
  await expect(dialog.locator(".evidenceReaderLines .referenced")).toContainText(draftEvidence(nextRunId).text);
  state.releaseDelayed?.();
  await expect.poll(() => state.delayedSettled).toBe(true);
  await expect(dialog.locator(".evidenceReaderMetadata")).toContainText("冻结版本 v2");
  await expect(dialog.locator(".evidenceReaderLines .referenced")).toContainText(draftEvidence(nextRunId).text);
  await expect(dialog.locator(".evidenceReaderLines")).not.toContainText(draftEvidence().text);
  await dialog.press("Escape");
  await page.getByRole("button", { name: "项目中心", exact: true }).click();
  await page.locator(".projectList").getByRole("link", { name: /另一部独立故事/ }).click();
  await expect(page).toHaveURL(new RegExp(`/app/projects/${otherProjectId}/`));
  await page.locator(".sideNav").getByRole("button", { name: "完整报告", exact: true }).click();
  await expect(dialog).not.toBeVisible();
  await primaryTrigger(page).click();
  await expect(dialog.locator(".evidenceReaderLines .referenced")).toContainText(draftEvidence(otherRunId).text);
  await expect(dialog.locator(".evidenceReaderLines")).not.toContainText(draftEvidence(nextRunId).text);
  expect(state.previews.at(-1)?.pathname).toBe(`/api/v1/analysis-runs/${otherRunId}/evidence-preview`);
  await dialog.press("Escape");
  assertReadingOnly(state);
});

test("360与1440阅读原文，375与横屏可操作；键盘Escape返回触发按钮", async ({ page }) => {
  const state = initialState();
  await page.setViewportSize({ width: 360, height: 900 });
  await openReport(page, state);
  const dialog = page.getByRole("dialog", { name: "分析时原文" });
  const trigger = primaryTrigger(page);
  for (const viewport of [{ width: 360, height: 900 }, { width: 375, height: 812 }, { width: 812, height: 375 }, { width: 1440, height: 1000 }]) {
    await page.setViewportSize(viewport);
    await trigger.focus();
    await trigger.press("Enter");
    await expect(dialog).toBeVisible();
    const close = dialog.getByRole("button", { name: "关闭阅读面板" });
    await expect(close).toBeFocused();
    await expect(dialog.locator('[data-evidence-line="130"]')).toContainText(draftEvidence().text);
    await expect(dialog.locator('[data-evidence-line="130"]')).toBeVisible();
    await expect(dialog.locator('[data-evidence-line="130"]')).toBeInViewport({ ratio: 0.5 });
    const headerSummary = dialog.locator(".evidenceReaderHeading .evidenceReaderHeaderSummary");
    await expect(headerSummary).toHaveText("冻结 v1 · 本次输入 · 正式问题");
    await expect(headerSummary).toBeInViewport({ ratio: 1 });
    const dialogBox = await dialog.boundingBox();
    const closeBox = await close.boundingBox();
    expect(dialogBox!.width).toBeLessThanOrEqual(viewport.width);
    expect(closeBox!.y).toBeGreaterThanOrEqual(0);
    expect(closeBox!.y + closeBox!.height).toBeLessThanOrEqual(viewport.height);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
    expect(await dialog.evaluate((element) => element.scrollWidth <= element.clientWidth)).toBe(true);
    for (const name of ["上一页原文", "返回引用位置", "下一页原文"]) {
      const footerBox = await dialog.getByRole("button", { name, exact: true }).boundingBox();
      expect(footerBox!.y, `${viewport.width}×${viewport.height}的${name}应在视口内`).toBeGreaterThanOrEqual(0);
      expect(footerBox!.y + footerBox!.height, `${viewport.width}×${viewport.height}的${name}应完整可见`).toBeLessThanOrEqual(viewport.height);
    }
    await dialog.getByRole("button", { name: "上一页原文" }).click();
    await expect(dialog.locator(".evidenceReaderLines > li").first()).toHaveAttribute("data-evidence-line", "1");
    await dialog.getByRole("button", { name: "下一页原文" }).click();
    await expect(dialog.locator(".evidenceReaderLines > li").first()).toHaveAttribute("data-evidence-line", "121");
    const returnButton = dialog.getByRole("button", { name: "返回引用位置" });
    const returnBox = await returnButton.boundingBox();
    expect(returnBox!.y + returnBox!.height, "阅读面板分页操作应在视口内可见").toBeLessThanOrEqual(viewport.height);
    await returnButton.focus();
    await page.keyboard.press("Enter");
    await expect(dialog.locator(".evidenceReaderLines > li").first()).toHaveAttribute("data-evidence-line", "118");
    const readingRegion = dialog.getByRole("region", { name: "冻结原文阅读区" });
    await expect(readingRegion).toBeFocused();
    await readingRegion.evaluate((element) => { element.scrollTop = element.scrollHeight; });
    await expect(headerSummary).toBeInViewport({ ratio: 1 });
    const summaryBox = await headerSummary.boundingBox();
    expect(summaryBox!.y).toBeGreaterThanOrEqual(0);
    expect(summaryBox!.y + summaryBox!.height).toBeLessThanOrEqual(viewport.height);
    await returnButton.focus();
    await page.keyboard.press("Enter");
    await expect(readingRegion).toBeFocused();
    await expect(dialog.locator('[data-evidence-line="130"]')).toBeInViewport({ ratio: 0.5 });
    if (viewport.width === 360 || viewport.width === 1440) {
      await page.screenshot({ path: `${screenshotFolder}/report-evidence-${viewport.width === 360 ? "mobile-360" : "desktop-1440"}.png` });
    }
    await page.keyboard.press("Escape");
    await expect(dialog).not.toBeVisible();
    await expect(trigger).toBeFocused();
  }
  assertReadingOnly(state);
});
