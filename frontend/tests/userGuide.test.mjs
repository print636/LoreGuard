import test from "node:test";
import assert from "node:assert/strict";
import { userGuideTopics } from "../src/features/help/userGuideContent.ts";

test("guide has exactly the six stable entry-point destinations", () => {
  assert.deepEqual(userGuideTopics.map((topic) => topic.id), ["start", "materials", "characters", "review", "report", "coverage"]);
  assert.equal(new Set(userGuideTopics.map((topic) => topic.title)).size, userGuideTopics.length);
});

test("every guide destination has readable headings and populated text blocks", () => {
  for (const topic of userGuideTopics) {
    assert.ok(topic.title.trim() && topic.summary.trim(), topic.id);
    assert.ok(topic.blocks.length > 0, topic.id);
    assert.equal(new Set(topic.blocks.map((block) => block.title)).size, topic.blocks.length, topic.id);
    for (const block of topic.blocks) {
      assert.ok(block.title.trim(), topic.id);
      const populated = [block.paragraphs, block.steps, block.points].filter((list) => list !== undefined);
      assert.ok(populated.length > 0, `${topic.id}: ${block.title}`);
      for (const list of populated) {
        assert.ok(Array.isArray(list) && list.length > 0, block.title);
        assert.ok(list.every((text) => typeof text === "string" && text.trim()), block.title);
      }
    }
  }
});

test("guide data stays plain local copy, without HTML, remote links or action descriptors", () => {
  for (const topic of userGuideTopics) {
    assert.deepEqual(Object.keys(topic).sort(), ["blocks", "id", "summary", "title"]);
    for (const block of topic.blocks) {
      assert.ok(Object.keys(block).every((key) => ["title", "paragraphs", "steps", "points"].includes(key)));
    }
  }
  const text = JSON.stringify(userGuideTopics);
  assert.doesNotMatch(text, /<\/?[a-z][^>]*>|https?:\/\/|javascript:|data:text\/html/i);
});
