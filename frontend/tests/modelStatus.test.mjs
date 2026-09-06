import test from 'node:test';
import assert from 'node:assert/strict';
import {
  describeModelStatus, describeRepairStatus, describeReviewAgentStatus,
} from '../src/modelStatus.ts';

const complete = {enabled:true, configured:true, total_chunks:2, attempted_chunks:2,
  succeeded_chunks:2, failed_chunks:0, skipped_chunks:0, invalid_records:0,
  empty_response_chunks:0,
  mode:'完整模型增强', reason_codes:[]};

function agentRun(overrides={}) {
  const candidateHash = 'a'.repeat(64);
  const spanHash = 'b'.repeat(64);
  return {
    protocol:'application_json_tools_v1', orchestrator:'langgraph_stategraph',
    decision_rounds:2, tool_calls:2, span_chars:18, span_read_count:1,
    prompt_tokens:30, completion_tokens:12, charged_tokens:42,
    recovered_records:1, unresolved_records:0, abstained_records:0,
    final_reason:'completed', total_trace_events:4, trace_truncated:false,
    trace:[
      {action:'DECISION', round:1, fields:[], validator_reason:'accepted_protocol',
        prompt_tokens:18, completion_tokens:6, elapsed_ms:20, final:'continue'},
      {action:'READ_SPAN', round:1, candidate_hash:candidateHash, doc_ref:'d1',
        line_start:1, line_end:1, span_hash:spanHash, fields:[],
        validator_reason:'read_ok', prompt_tokens:0, completion_tokens:0,
        elapsed_ms:0, final:'continue'},
      {action:'DECISION', round:2, fields:[], validator_reason:'accepted_protocol',
        prompt_tokens:12, completion_tokens:6, elapsed_ms:11, final:'continue'},
      {action:'PATCH_RECORDS', round:2, candidate_hash:candidateHash, doc_ref:'d1',
        line_start:1, line_end:1, span_hash:spanHash, fields:['value'],
        validator_reason:'patch_ok', prompt_tokens:0, completion_tokens:0,
        elapsed_ms:0, final:'accepted'},
    ],
    ...overrides,
  };
}

function withAgent(overrides={}) {
  return {
    ...complete,
    invalid_records:1, unresolved_invalid_records:0, recovered_invalid_records:1,
    repair_attempted:false, repair_succeeded:false, repair_failed:false,
    repair_pre_invalid:0, repair_post_invalid:0, repair_salvaged:0, repair_dropped:0,
    repair_final_path:'not_needed',
    repair:{attempted:false, succeeded:false, failed:false, pre_invalid:0,
      post_invalid:0, salvaged:0, dropped:0, final_path:'not_needed'},
    review_agent_attempted:true, review_agent_succeeded:true,
    review_agent_abstained:false, review_agent_total_runs:1,
    review_agent_runs_truncated:false, review_agent_runs:[agentRun()],
    ...overrides,
  };
}

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
  assert.match(result.counts, /总恢复 4/);
  assert.match(result.counts, /repair 路径 repaired/);
  assert.match(result.detail, /共恢复 4 条.*两条独立诊断链/);
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

test('legacy runs keep model coverage compatibility while Agent status stays unknown', () => {
  assert.equal(describeModelStatus(complete).coverage, 'full');
  const agent = describeReviewAgentStatus(complete);
  assert.equal(agent.state, 'unknown');
  assert.equal(agent.supplied, false);
  assert.match(agent.detail, /旧运行/);
});

test('Agent recovery is disclosed separately and never relabeled as fixed repair', () => {
  const model = withAgent();
  const coverage = describeModelStatus(model);
  const repair = describeRepairStatus(model);
  const agent = describeReviewAgentStatus(model);
  assert.equal(coverage.coverage, 'full');
  assert.doesNotMatch(coverage.detail, /repair.*恢复 1/);
  assert.equal(repair.state, 'not_attempted');
  assert.match(repair.detail, /不是 Agent/);
  assert.equal(agent.state, 'succeeded');
  assert.equal(agent.coverageProven, true);
  assert.match(agent.detail, /Agent recovery 不是固定语义标签 repair/);
  assert.match(agent.counts, /attempted yes.*succeeded yes.*abstained no/);
});

test('unsuccessful or abstaining Agent cannot promote otherwise full coverage', () => {
  const model = withAgent({
    invalid_records:0, unresolved_invalid_records:0, recovered_invalid_records:0,
    review_agent_succeeded:false, review_agent_abstained:true,
    review_agent_runs:[agentRun({recovered_records:0, abstained_records:1,
      final_reason:'explicit_abstain'})],
  });
  const result = describeModelStatus(model);
  assert.equal(result.coverage, 'partial');
  assert.match(result.detail, /Agent 未成功完成或发生弃答/);
  assert.match(result.emptyCaveat, /不能证明没有/);
});

test('run view names the application JSON protocol and bounded per-run accounting', () => {
  const status = describeReviewAgentStatus(withAgent({
    review_agent_runs_truncated:true, review_agent_total_runs:3,
    review_agent_runs:[agentRun({
      total_trace_events:6, trace_truncated:true,
      prompt:'SECRET_PROMPT', raw:'SECRET_RAW', api_key:'SECRET_KEY',
      endpoint:'SECRET_ENDPOINT', evidence:'SECRET_EVIDENCE',
    })],
  }));
  assert.equal(status.valid, true);
  assert.match(status.counts, /展示 1 \/ 总计 3（已截断）/);
  assert.match(status.runs[0].protocol, /应用层 JSON 工具协议.*非原生 function calling/);
  assert.match(status.runs[0].activity, /决策轮 2.*工具调用 2.*span read 1 次 \/ 18 字符/);
  assert.match(status.runs[0].tokens, /prompt 30.*completion 12.*charged 42/);
  assert.equal(status.runs[0].finalReason, '流程完成');
  assert.match(status.runs[0].traceCount, /展示 4 \/ 总计 6（已截断）/);
  const rendered = JSON.stringify(status);
  for (const secret of ['SECRET_PROMPT','SECRET_RAW','SECRET_KEY','SECRET_ENDPOINT','SECRET_EVIDENCE']) {
    assert.doesNotMatch(rendered, new RegExp(secret));
  }
});

test('mixed fixed repair and Agent paths remain two independently named diagnostics', () => {
  const model = withAgent({
    invalid_records:2, unresolved_invalid_records:0, recovered_invalid_records:2,
    repair_attempted:true, repair_succeeded:true, repair_pre_invalid:1,
    repair_final_path:'repaired',
    repair_final_path:'repaired',
    repair:{attempted:true, succeeded:true, failed:false, pre_invalid:1,
      post_invalid:0, salvaged:0, dropped:0, final_path:'repaired'},
  });
  const repair = describeRepairStatus(model);
  const agent = describeReviewAgentStatus(model);
  assert.equal(repair.label, '固定语义标签 repair 已完成');
  assert.equal(agent.label, '受限证据修复 Agent 已采纳补丁');
  assert.equal(agent.coverageProven, true);
  assert.equal(describeModelStatus(model).coverage, 'full');
  assert.doesNotMatch(repair.detail, /Agent 成功/);
  assert.match(agent.runs[0].outcome, /Agent 恢复 1 条/);
});

test('malformed Agent diagnostics fail closed without exposing an unsafe run', () => {
  const model = withAgent({review_agent_runs:[{...agentRun(), protocol:'native_tools'}]});
  const agent = describeReviewAgentStatus(model);
  assert.equal(agent.state, 'unknown');
  assert.deepEqual(agent.runs, []);
  assert.equal(describeModelStatus(model).coverage, 'unknown');
});

test('reported Agent success needs a nonempty complete replay before full coverage', () => {
  const sentinels = [
    withAgent({review_agent_total_runs:0, review_agent_runs:[]}),
    withAgent({review_agent_runs:[agentRun({protocol:'unknown'})]}),
    withAgent({review_agent_runs:[agentRun({orchestrator:'unknown'})]}),
    withAgent({review_agent_runs:[agentRun({final_reason:'round_limit'})]}),
    withAgent({review_agent_runs:[agentRun({trace_truncated:true,
      total_trace_events:5})]}),
    withAgent({review_agent_runs:[agentRun({recovered_records:2})]}),
  ];
  for (const model of sentinels) {
    const agent = describeReviewAgentStatus(model);
    assert.equal(agent.valid, true);
    assert.equal(agent.coverageProven, false);
    assert.match(agent.label, /覆盖证据不足/);
    assert.notEqual(describeModelStatus(model).coverage, 'full');
  }
});

test('Agent recovery share must subtract independently proven fixed repair recovery', () => {
  const validMixed = withAgent({
    invalid_records:2, unresolved_invalid_records:0, recovered_invalid_records:2,
    repair_attempted:true, repair_succeeded:true, repair_pre_invalid:1,
    repair_post_invalid:0, repair_final_path:'repaired',
    repair:{attempted:true, succeeded:true, failed:false, pre_invalid:1,
      post_invalid:0, salvaged:0, dropped:0, final_path:'repaired'},
  });
  assert.equal(describeReviewAgentStatus(validMixed).coverageProven, true);

  const mismatched = {...validMixed, recovered_invalid_records:3, invalid_records:3};
  assert.equal(describeReviewAgentStatus(mismatched).coverageProven, false);
  assert.notEqual(describeModelStatus(mismatched).coverage, 'full');

  const forgedUnattemptedFixed = {
    ...validMixed,
    repair_attempted:false, repair_succeeded:false, repair_final_path:'not_needed',
    repair:{attempted:false, succeeded:false, failed:false, pre_invalid:1,
      post_invalid:0, salvaged:0, dropped:0, final_path:'not_needed'},
  };
  assert.equal(describeRepairStatus(forgedUnattemptedFixed).state, 'unknown');
  assert.equal(describeReviewAgentStatus(forgedUnattemptedFixed).coverageProven, false);
  assert.notEqual(describeModelStatus(forgedUnattemptedFixed).coverage, 'full');

  const divergentShapes = {
    ...validMixed,
    repair:{...validMixed.repair, pre_invalid:0},
  };
  assert.equal(describeRepairStatus(divergentShapes).state, 'unknown');
  assert.equal(describeReviewAgentStatus(divergentShapes).coverageProven, false);
  assert.notEqual(describeModelStatus(divergentShapes).coverage, 'full');
});
