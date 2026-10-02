import test from "node:test";
import assert from "node:assert/strict";
import { metadataDraftDirty, metadataNameError, metadataPatch, rebaseMetadataDraft, verifiedCreatedProjectId, verifiedProjectMetadata } from "../src/features/projects/projectMetadata.ts";

const baseline = { id: "project-1", name: "星门纪事", description: "原说明", created_at: "2026-10-02T00:00:00Z", metadata_revision: 2 };

test("metadata dialog requires verified project identity and positive integer CAS revision", () => {
  assert.deepEqual(verifiedProjectMetadata(baseline, "project-1"), baseline);
  for (const patch of [{ id: "project-2" }, { metadata_revision: 0 }, { metadata_revision: 2.5 }, { name: "   " }, { description: null }]) assert.equal(verifiedProjectMetadata({ ...baseline, ...patch }, "project-1"), null);
});

test("metadata name is trimmed and validated without truncating project description", () => {
  const draft = { name: "  故事新名  ", description: "长说明".repeat(800) };
  assert.deepEqual(metadataPatch(draft, baseline), { name: "故事新名", description: draft.description, expected_revision: 2 });
  assert.match(metadataNameError(" "), /项目名称/);
  assert.match(metadataNameError("名".repeat(201)), /200/);
  assert.equal(metadataNameError("名".repeat(200)), "");
});

test("dirty tracking distinguishes unverified initial loading, unchanged draft and real edits", () => {
  assert.equal(metadataDraftDirty({ name: "", description: "" }, null), false);
  assert.equal(metadataDraftDirty(baseline, baseline), false);
  assert.equal(metadataDraftDirty({ ...baseline, description: "新说明" }, baseline), true);
});

test("explicit conflict merge preserves author-edited fields and adopts latest untouched fields", () => {
  const latest = { name: "别处修改的名称", description: "别处新增的说明" };
  assert.deepEqual(rebaseMetadataDraft({ name: "我的新名称", description: baseline.description }, baseline, latest), { name: "我的新名称", description: latest.description });
  assert.deepEqual(rebaseMetadataDraft({ name: baseline.name, description: "我的新说明" }, baseline, latest), { name: latest.name, description: "我的新说明" });
  assert.deepEqual(rebaseMetadataDraft({ name: "我的新名称", description: "我的新说明" }, baseline, latest), { name: "我的新名称", description: "我的新说明" });
  assert.deepEqual(rebaseMetadataDraft(baseline, baseline, latest), latest);
  assert.equal(baseline.metadata_revision, 2);
});

test("ambiguous project creation does not invent an id or bind by a project name", () => {
  assert.equal(verifiedCreatedProjectId({ name: "same name" }), null);
  assert.equal(verifiedCreatedProjectId({ id: "../another/project" }), null);
  assert.equal(verifiedCreatedProjectId({ id: "project-1" }), "project-1");
});
