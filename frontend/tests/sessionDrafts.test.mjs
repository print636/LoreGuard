import test from "node:test";
import assert from "node:assert/strict";
import {
  activateDraftSession, discardDraftSession, draftLease, draftStorageKey,
  isFeedbackDraft, isQuickDraft, MAX_DRAFT_BYTES, readTabDraft, removeTabDraft,
  suspendDraftSession, verifiedFeedbackNotes, writeTabDraft,
} from "../src/features/drafts/sessionDraftStorage.ts";

function storage() {
  const entries = new Map();
  return { get length() { return entries.size; }, key: (index) => [...entries.keys()][index] ?? null,
    getItem: (key) => entries.get(key) ?? null, setItem: (key, value) => entries.set(key, value), removeItem: (key) => entries.delete(key) };
}
const owner = { userId: "author-a", workspaceId: "workspace-a" };
const scope = { ...owner, projectId: "project-1", kind: "quick" };
const quick = { chapter: "本账户未提交的原创章节。", mode: "body", role: "chapter", scope: "main" };
function session(who = owner) { activateDraftSession(who); return draftLease(who); }

test("原账户缓存可刷新读取，文本设置一起保存，不缓存默认设定", () => {
  const cache = storage(); const lease = session();
  assert.equal(writeTabDraft(scope, lease, quick, cache), "saved");
  const restored = readTabDraft(scope, lease, cache);
  assert.equal(restored.status, "ready");
  assert.deepEqual(restored.draft, quick);
  assert.equal("world" in restored.draft, false);
});

test("401保留文本但撤销旧回调；重新认证原账户才恢复", () => {
  const cache = storage(); const oldLease = session();
  writeTabDraft(scope, oldLease, quick, cache);
  suspendDraftSession();
  assert.equal(readTabDraft(scope, oldLease, cache).status, "revoked");
  assert.equal(writeTabDraft(scope, oldLease, { ...quick, chapter: "late callback" }, cache), "revoked");
  const newLease = session();
  assert.deepEqual(readTabDraft(scope, newLease, cache).draft, quick);
});

test("账户、工作区、项目、运行命名空间相互隔离；换账户不清原账户稿", () => {
  const cache = storage(); const lease = session();
  writeTabDraft(scope, lease, quick, cache);
  for (const different of [{ ...owner, userId: "author-b" }, { ...owner, workspaceId: "workspace-b" }]) {
    const otherLease = session(different);
    assert.equal(readTabDraft({ ...scope, ...different }, otherLease, cache).draft, null);
    assert.equal(readTabDraft(scope, otherLease, cache).status, "revoked");
  }
  const current = session();
  assert.equal(readTabDraft({ ...scope, projectId: "project-2" }, current, cache).draft, null);
  assert.deepEqual(readTabDraft(scope, current, cache).draft, quick);
  const noteScope = { ...scope, kind: "feedback", runId: "run-1" };
  writeTabDraft(noteScope, current, { notes: { "issue-1": "备注" } }, cache);
  assert.equal(readTabDraft({ ...noteScope, runId: "run-2" }, current, cache).draft, null);
});

test("主动退出先撤销租约再清本账户缓存，清理后unmount不能复写", () => {
  const cache = storage(); const lease = session();
  cache.setItem("unrelated", "preserve");
  writeTabDraft(scope, lease, quick, cache);
  const other = { userId: "other-account", workspaceId: "other-workspace" };
  const otherScope = { ...scope, ...other };
  writeTabDraft(otherScope, session(other), quick, cache);
  const logoutLease = session();
  assert.equal(discardDraftSession(owner, cache), true);
  assert.equal(cache.getItem(draftStorageKey(scope)), null);
  assert.notEqual(cache.getItem(draftStorageKey(otherScope)), null);
  assert.equal(cache.getItem("unrelated"), "preserve");
  assert.equal(writeTabDraft(scope, logoutLease, quick, cache), "revoked");
  assert.equal(readTabDraft(scope, session(), cache).draft, null);
});

test("禁用/配额异常可恢复错误而非抛出或谎称保存", () => {
  const broken = { get length() { throw new Error("blocked"); }, key() { throw new Error("blocked"); },
    getItem() { throw new Error("blocked"); }, setItem() { throw new Error("quota"); }, removeItem() { throw new Error("blocked"); } };
  const independentOwner = { userId: "blocked-storage-author", workspaceId: "blocked-storage-workspace" };
  const independentScope = { ...scope, ...independentOwner };
  const lease = session(independentOwner);
  assert.equal(readTabDraft(independentScope, lease, broken).status, "unavailable");
  assert.equal(writeTabDraft(independentScope, lease, quick, broken), "unavailable");
  assert.equal(removeTabDraft(independentScope, lease, broken), "unavailable");
  assert.equal(discardDraftSession(independentOwner, broken), false);
});

test("超上限不覆盖最后成功快照，输入层可继续保留当前正文", () => {
  const cache = storage(); const lease = session();
  writeTabDraft(scope, lease, quick, cache);
  assert.equal(writeTabDraft(scope, lease, { ...quick, chapter: "文".repeat(MAX_DRAFT_BYTES) }, cache), "too_large");
  assert.deepEqual(readTabDraft(scope, lease, cache).draft, quick);
});

test("本轮写过的缓存即使logout删除失败，原账号重新登录也不能恢复旧稿", () => {
  const isolated = { userId: "delete-failure-author", workspaceId: "delete-failure-workspace" };
  const isolatedScope = { ...scope, ...isolated };
  const cache = storage();
  const lease = session(isolated);
  assert.equal(writeTabDraft(isolatedScope, lease, quick, cache), "saved");
  const brokenDelete = { ...cache, get length() { return cache.length; }, removeItem() { throw new Error("synthetic blocked delete"); } };
  assert.equal(discardDraftSession(isolated, brokenDelete), false);
  assert.equal(readTabDraft(isolatedScope, session(isolated), cache).draft, null);
  assert.equal(writeTabDraft(isolatedScope, draftLease(isolated), { ...quick, chapter: "新的登录后正文" }, cache), "saved");
  assert.equal(readTabDraft(isolatedScope, draftLease(isolated), cache).draft.chapter, "新的登录后正文");
});

test("损坏包、跨命名空间包、秘密字段与错误类型均不恢复", () => {
  const cache = storage(); const lease = session();
  const key = draftStorageKey(scope);
  for (const raw of ["{", JSON.stringify({ version: 1, scope: { ...scope, userId: "another" }, payload: quick }),
    JSON.stringify({ version: 1, scope, payload: { ...quick, apiKey: "not-a-real-key" } }),
    JSON.stringify({ version: 2, scope, payload: quick })]) {
    cache.setItem(key, raw);
    assert.equal(readTabDraft(scope, lease, cache).draft, null);
  }
  assert.equal(isQuickDraft({ ...quick, role: ["chapter"] }), false);
  assert.equal(isQuickDraft({ ...quick, password: "not-a-password" }), false);
  assert.equal(isFeedbackDraft({ notes: { id: 7 } }), false);
});

test("备注只恢复已验证run中存在的issue，单条提交仅清单条", () => {
  const cache = storage(); const lease = session();
  const noteScope = { ...scope, kind: "feedback", runId: "run-verified" };
  const draft = { notes: { "issue-a": "待提交A", "issue-b": "待提交B", "unknown-issue": "不应显示" } };
  assert.deepEqual(verifiedFeedbackNotes(draft, ["issue-a", "issue-b"]), { "issue-a": "待提交A", "issue-b": "待提交B" });
  writeTabDraft(noteScope, lease, draft, cache);
  writeTabDraft(noteScope, lease, { notes: { "issue-b": "待提交B" } }, cache);
  assert.deepEqual(readTabDraft(noteScope, lease, cache).draft.notes, { "issue-b": "待提交B" });
  removeTabDraft(noteScope, lease, cache);
  assert.equal(readTabDraft(noteScope, lease, cache).draft, null);
});
