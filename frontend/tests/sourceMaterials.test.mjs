import test from "node:test";
import assert from "node:assert/strict";
import {
  MAX_SOURCE_MATERIALS, SourceMaterialsError, selectedSourceIds, sourceMaterialContext,
  sourceMaterialsPath, sourceMaterialsFailure, sourceMaterialsBlob,
} from "../src/features/source-materials/sourceMaterialsModel.ts";

const inputs = [
  { document_id: "old-draft", document_name: "返航.docx", document_version: 2, document_role: "chapter", story_scope: "航行篇",
    batch_role: "target", content_sha256: "a".repeat(64), narrative_context: { publication_status: "draft" } },
  { document_id: "old-profile", document_name: "人物.md", document_version: 1, document_role: "character_profile", story_scope: "主线",
    batch_role: "background", content_sha256: "b".repeat(64), narrative_context: { publication_status: "retired" } },
];

test("only explicit selected run inputs enter the ZIP request; order follows the frozen list", () => {
  assert.deepEqual(selectedSourceIds(inputs, new Set()), []);
  assert.deepEqual(selectedSourceIds(inputs, new Set(["old-profile"])), ["old-profile"]);
  assert.deepEqual(selectedSourceIds(inputs, new Set(["old-profile", "old-draft"])), ["old-draft", "old-profile"]);
  for (const id of ["new-draft", "implicit-confirmed-source", "other-project-doc"]) {
    assert.throws(() => selectedSourceIds(inputs, new Set([id])), /清单不一致/);
  }
  assert.throws(() => selectedSourceIds([inputs[0], inputs[0]], new Set(["old-draft"])), /清单不一致/);
});

test("GET uses repeated encoded document_id and never submits raw content or guessed filenames", () => {
  const path = sourceMaterialsPath("run / one", ["doc/a", "doc & b"]);
  const url = new URL(path, "https://example.test");
  assert.equal(url.pathname, "/api/v1/analysis-runs/run%20%2F%20one/source-materials.zip");
  assert.deepEqual([...url.searchParams.entries()], [["document_id", "doc/a"], ["document_id", "doc & b"]]);
  assert.doesNotMatch(path, /filename|content|format|model|project_id/);
});

test("empty, duplicate and over-limit selections fail without silently truncating", () => {
  for (const ids of [[], ["a", "a"], [""], Array.from({ length: MAX_SOURCE_MATERIALS + 1 }, (_, i) => `id-${i}`)]) {
    assert.throws(() => sourceMaterialsPath("run", ids), TypeError);
  }
  const ids = Array.from({ length: MAX_SOURCE_MATERIALS }, (_, i) => `id-${i}`);
  assert.equal(new URL(sourceMaterialsPath("run", ids), "https://example.test").searchParams.getAll("document_id").length, MAX_SOURCE_MATERIALS);
});

test("saved identity labels remain readable without IDs/hashes or a verified claim", () => {
  assert.equal(sourceMaterialContext(inputs[1]), "比对依据 · 角色档案 · 故事范围：主线 · 已归档");
  assert.equal(sourceMaterialContext(inputs[0]), "待审稿 · 故事正文/章节 · 故事范围：航行篇 · 草稿");
  assert.equal(sourceMaterialContext({ ...inputs[0], document_role: null, batch_role: null, story_scope: "", narrative_context: undefined }),
    "审查用途未记录 · 资料身份未记录 · 故事范围：未记录 · 发布状态未记录");
  assert.doesNotMatch(sourceMaterialContext(inputs[0]), /old-draft|aaa|已核验|已核对/);
});

test("unavailable sources are described per item without displaying internal reasons or server raw text", () => {
  const value = sourceMaterialsFailure(409, { detail: { code: "source_materials_unavailable", message: "internal: do not display",
    documents: [{ name: "返航.docx", reason: "content_mismatch" }, { name: null, reason: "missing" }, { name: "人物.md", reason: "context_mismatch" }] } });
  assert.match(value, /本次未生成压缩包/);
  assert.match(value, /返航.docx：原文无法核对/);
  assert.match(value, /一份所选资料：原文缺失/);
  assert.match(value, /人物.md：资料身份无法核对/);
  assert.doesNotMatch(value, /content_mismatch|context_mismatch|internal/);
});

test("over-size, validation, permission and unavailable-run failures have bounded human messages", () => {
  assert.match(sourceMaterialsFailure(413, { detail: { code: "source_materials_too_large" } }), /20 MiB.*分批.*没有截断/);
  assert.match(sourceMaterialsFailure(422, { detail: [{ msg: "raw validation internals" }] }), /1 至 100/);
  assert.match(sourceMaterialsFailure(401, null), /登录/);
  assert.match(sourceMaterialsFailure(404, { detail: { code: "run_not_found" } }), /无权/);
  assert.match(sourceMaterialsFailure(409, { detail: { code: "run_not_completed" } }), /尚未完成/);
  assert.doesNotMatch(sourceMaterialsFailure(500, { detail: "internal raw response" }), /internal/);
});

test("successful ZIP is passed through, while HTML/empty responses cannot become a downloaded archive", async () => {
  const bytes = new Uint8Array([80, 75, 3, 4, 1, 2]);
  const blob = await sourceMaterialsBlob(new Response(bytes, { headers: { "Content-Type": "application/zip" } }));
  assert.deepEqual(new Uint8Array(await blob.arrayBuffer()), bytes);
  await assert.rejects(sourceMaterialsBlob(new Response("<html>login</html>", { headers: { "Content-Type": "text/html" } })), SourceMaterialsError);
  await assert.rejects(sourceMaterialsBlob(new Response("", { headers: { "Content-Type": "application/zip" } })), /空压缩包/);
});

test("ZIP endpoint errors never become files or successful partial exports", async () => {
  const payload = { detail: { code: "source_materials_unavailable", documents: [{ name: "人物.md", reason: "invalid_source" }] } };
  await assert.rejects(sourceMaterialsBlob(new Response(JSON.stringify(payload), { status: 409, headers: { "Content-Type": "application/json" } })), /人物.md：来源无法核对/);
  await assert.rejects(sourceMaterialsBlob(new Response("upstream", { status: 503 })), /暂时失败/);
});
