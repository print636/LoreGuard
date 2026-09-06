import test from 'node:test';
import assert from 'node:assert/strict';
import {describeModelStatus} from '../src/modelStatus.ts';

const complete = {enabled:true, configured:true, total_chunks:2, attempted_chunks:2,
  succeeded_chunks:2, failed_chunks:0, skipped_chunks:0, invalid_records:0,
  empty_response_chunks:0,
  mode:'完整模型增强', reason_codes:[]};

test('missing and legacy counters do not imply baseline execution', () => {
  for (const value of [undefined, {used:true, mode:'完整模型增强'}]) {
    const result = describeModelStatus(value);
    assert.match(result.label, /未知/);
    assert.match(result.emptyCaveat, /不能证明没有/);
  }
});
test('complete run displays logical chunk counts', () => {
  assert.equal(describeModelStatus(complete).label, 'AI 语义审查完整');
  assert.equal(describeModelStatus(complete).coverage, 'full');
  assert.match(describeModelStatus(complete).counts, /成功 2/);
  assert.match(describeModelStatus(complete).detail, /不代表结论/);
});
test('budget exhaustion is not described as a provider failure', () => {
  const result = describeModelStatus({...complete, total_chunks:3, skipped_chunks:1, reason_codes:['token_budget']});
  assert.equal(result.coverage, 'partial');
  assert.equal(result.detail, 'Token 预算不足');
});
test('rejected records are visible even with successful calls', () => {
  const result = describeModelStatus({...complete, invalid_records:1, reason_codes:['evidence_range']});
  assert.equal(result.label, 'AI 语义审查部分降级');
  assert.match(result.counts, /拒绝记录 1/);
  assert.match(result.detail, /越界/);
});
test('successfully repaired raw invalid records remain full and disclose final disposition', () => {
  const result = describeModelStatus({...complete, invalid_records:4,
    unresolved_invalid_records:0, recovered_invalid_records:4,
    repair_succeeded:true, repair_failed:false, repair_post_invalid:0,
    repair_final_path:'repaired', reason_codes:['schema_validation','repair_succeeded']});
  assert.equal(result.coverage, 'full');
  assert.match(result.counts, /原始无效 4/);
  assert.match(result.counts, /最终未解决 0/);
  assert.match(result.counts, /已恢复 4/);
  assert.match(result.counts, /repair repaired/);
  assert.match(result.detail, /repair pass 已恢复 4 条/);
});
test('unresolved, salvaged, failed, and inconsistent repair diagnostics fail closed', () => {
  const unresolved = describeModelStatus({...complete, invalid_records:1,
    unresolved_invalid_records:1, recovered_invalid_records:0,
    repair_failed:true, repair_post_invalid:1, repair_final_path:'salvaged'});
  assert.equal(unresolved.coverage, 'partial');
  assert.match(unresolved.counts, /最终未解决 1/);

  assert.equal(describeModelStatus({...complete, invalid_records:2,
    unresolved_invalid_records:0, recovered_invalid_records:1,
    repair_failed:false, repair_post_invalid:0}).coverage, 'unknown');
  assert.equal(describeModelStatus({...complete, invalid_records:1,
    unresolved_invalid_records:0, recovered_invalid_records:1,
    repair_failed:true, repair_post_invalid:0}).coverage, 'partial');
});
test('schema-valid empty responses are degraded and warn when zero issues are shown', () => {
  const result = describeModelStatus({...complete, empty_response_chunks:1});
  assert.equal(result.coverage, 'partial');
  assert.match(result.label, /部分降级/);
  assert.match(result.counts, /空响应 1/);
  assert.match(result.detail, /返回空结果.*无法证明完整覆盖/);
  assert.match(result.emptyCaveat, /0 个问题不能证明/);
});
test('impossible empty-response counts are unknown, not partial coverage', () => {
  const result = describeModelStatus({...complete, succeeded_chunks:1, failed_chunks:1,
    empty_response_chunks:2});
  assert.equal(result.coverage, 'unknown');
  assert.match(result.label, /未知/);
});
test('malformed counters and unknown reason values are conservative', () => {
  assert.match(describeModelStatus({...complete, attempted_chunks:-1}).label, /未知/);
  assert.match(describeModelStatus({...complete, attempted_chunks:1}).label, /未知/);
  assert.equal(describeModelStatus({...complete, invalid_records:1, reason_codes:['untrusted-value']}).detail, '其他执行限制');
});

test('zero successful chunks is clearly a limited preflight', () => {
  const disabled = {enabled:false, configured:false, total_chunks:2,
    attempted_chunks:0, succeeded_chunks:0, failed_chunks:0, skipped_chunks:2,
    invalid_records:0, empty_response_chunks:0, reason_codes:['disabled']};
  const result = describeModelStatus(disabled);
  assert.equal(result.coverage, 'limited');
  assert.match(result.label, /完全降级.*有限预检/);
  assert.match(result.emptyCaveat, /不能证明没有/);
});
