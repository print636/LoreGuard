import test from 'node:test';
import assert from 'node:assert/strict';
import { issueSeverityLabel, reviewRecordLabel, reviewStatusLabel, reviewTimeLabel } from '../src/productLanguage.ts';

test('review statuses use author-facing labels and never echo unknown system codes', () => {
  assert.equal(reviewStatusLabel('completed'), '审查完成');
  assert.equal(reviewStatusLabel('failed'), '审查未完成');
  assert.equal(reviewStatusLabel(null), '尚未审查');
  assert.equal(reviewStatusLabel('unexpected-private-state'), '状态待确认');
});

test('history labels use display position and valid local time, without requiring resource ids', () => {
  assert.equal(reviewTimeLabel(null), '时间未记录');
  assert.equal(reviewTimeLabel('not-a-date'), '时间未记录');
  assert.match(reviewRecordLabel(21, '2026-10-23T12:34:56Z'), /^记录 21 · /);
  assert.doesNotMatch(reviewRecordLabel(21, '2026-10-23T12:34:56Z'), /Invalid Date|undefined/);
});

test('issue severity is readable Chinese, including an honest unknown state', () => {
  assert.equal(issueSeverityLabel('medium'), '一般');
  assert.equal(issueSeverityLabel('high'), '严重');
  assert.equal(issueSeverityLabel('unrecognized-code'), '程度待确认');
});
