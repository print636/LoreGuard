import test from 'node:test';
import assert from 'node:assert/strict';
import {describeModelStatus} from '../src/modelStatus.ts';

const complete = {enabled:true, attempted_chunks:2, succeeded_chunks:2, failed_chunks:0,
  skipped_chunks:0, invalid_records:0, mode:'完整模型增强', reason_codes:[]};

test('missing and legacy counters do not imply baseline execution', () => {
  for (const value of [undefined, {used:true, mode:'完整模型增强'}]) {
    const result = describeModelStatus(value);
    assert.match(result.label, /未知/);
    assert.doesNotMatch(result.detail, /本次未使用模型/);
  }
});
test('complete run displays logical chunk counts', () => {
  assert.match(describeModelStatus(complete).counts, /成功 2/);
  assert.match(describeModelStatus(complete).detail, /不代表/);
});
test('budget exhaustion is not described as a provider failure', () => {
  const result = describeModelStatus({...complete, skipped_chunks:1, reason_codes:['token_budget']});
  assert.equal(result.detail, 'Token 预算不足');
});
test('rejected records are visible even with successful calls', () => {
  const result = describeModelStatus({...complete, invalid_records:1, reason_codes:['evidence_range']});
  assert.match(result.counts, /拒绝记录 1/);
  assert.match(result.detail, /越界/);
});
test('malformed counters and unknown reason values are conservative', () => {
  assert.match(describeModelStatus({...complete, attempted_chunks:-1}).label, /未知/);
  assert.equal(describeModelStatus({...complete, reason_codes:['untrusted-value']}).detail, '其他执行限制');
});
