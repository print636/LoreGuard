import test from "node:test";
import assert from "node:assert/strict";
import {
  catalogMatchesRequest, runCatalogAction, runCatalogPath, runHistorySearch, runHistoryStateFromSearch,
} from "../src/features/run-history/runHistoryModel.ts";

test("历史URL状态严格限制范围，独立于正式问题status筛选", () => {
  assert.deepEqual(runHistoryStateFromSearch("?history_page=3&history_status=failed&status=accepted"), { page: 3, status: "failed" });
  for (const raw of ["-1", "0", "100001", "1.5", "NaN"]) assert.equal(runHistoryStateFromSearch(`?history_page=${raw}`).page, 1);
  assert.equal(runHistoryStateFromSearch("?history_status=secret").status, "all");
});

test("翻页保留其他query，默认参数去掉，不改变run路径", () => {
  const search = runHistorySearch("?issue=issue-a&status=accepted", { page: 4, status: "completed" });
  assert.equal(new URLSearchParams(search).get("issue"), "issue-a");
  assert.equal(new URLSearchParams(search).get("status"), "accepted");
  assert.equal(new URLSearchParams(search).get("history_page"), "4");
  assert.equal(runHistorySearch(search, { page: 1, status: "all" }), "?issue=issue-a&status=accepted");
  assert.equal(runCatalogPath("project/a", { page: 2, status: "failed" }), "/api/v1/projects/project%2Fa/run-catalog?page=2&page_size=20&status=failed");
});

test("目录响应绑定project与page且不接受跨项目/超页/非法统计", () => {
  const item = { id: "run-a", project_id: "project-a", status: "completed", created_at: "2026-10-02", input_chars: 100,
    frozen_document_count: 2, prompt_tokens: 12, completion_tokens: 5 };
  const data = { project_id: "project-a", page: 2, page_size: 20, total: 61, has_more: true, items: [item] };
  assert.equal(catalogMatchesRequest(data, "project-a", 2), true);
  assert.equal(catalogMatchesRequest({ ...data, items: [{ ...item, status: "unknown" }] }, "project-a", 2), true);
  assert.equal(catalogMatchesRequest(data, "project-b", 2), false);
  assert.equal(catalogMatchesRequest(data, "project-a", 3), false);
  for (const changes of [{ total: -1 }, { items: Array(21).fill(item) }, { items: [{ ...item, project_id: "other" }] },
    { items: [{ ...item, prompt_tokens: "12" }] }]) assert.equal(catalogMatchesRequest({ ...data, ...changes }, "project-a", 2), false);
});

test("动作仅按状态命名，Token元数据不据此推断模型参与", () => {
  assert.equal(runCatalogAction("completed"), "查看报告");
  assert.equal(runCatalogAction("running"), "查看进度");
  assert.equal(runCatalogAction("failed"), "查看详情");
  assert.equal(runCatalogAction("unknown"), "查看详情");
});
