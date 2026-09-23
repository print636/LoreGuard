import test from 'node:test';
import assert from 'node:assert/strict';
import {
  describeRunModelExecution,
  parseRunModelExecution,
} from '../src/runModelExecution.ts';

const endpointHash = 'a'.repeat(64);
const accountPlan = {
  planned_source: 'account_byok',
  profile_revision: 7,
  model: 'gpt-5',
  endpoint_configuration_sha256: endpointHash,
  configured: true,
  status: 'planned',
  ignored_upstream_payload: 'must never be shown',
};

test('strict execution parsing allowlists fields and shortens the endpoint fingerprint', () => {
  assert.deepEqual(parseRunModelExecution(accountPlan), {
    source: 'account_byok',
    profileRevision: 7,
    model: 'gpt-5',
    endpointFingerprint: 'aaaaaaaa',
    configured: true,
    status: 'planned',
  });
});

test('queued accepted runs disclose their frozen account plan', () => {
  const view = describeRunModelExecution(accountPlan, 'queued');
  assert.equal(view.tone, 'planned');
  assert.equal(view.label, '计划使用账户模型');
  assert.match(view.detail, /配置修订 r7/);
  assert.match(view.detail, /模型 gpt-5/);
  assert.match(view.detail, /冻结快照/);
  assert.doesNotMatch(view.detail, new RegExp(endpointHash));
  assert.doesNotMatch(view.detail, /must never be shown/);
});

test('completed runs distinguish actual use, no call, and deterministic-only review', () => {
  assert.equal(describeRunModelExecution({
    ...accountPlan, status: 'used',
  }, 'completed').label, '实际使用账户模型');
  assert.equal(describeRunModelExecution({
    ...accountPlan,
    planned_source: 'service_default',
    profile_revision: null,
    status: 'not_used',
  }, 'completed').label, '服务默认模型未产生可确认的模型结果');
  assert.equal(describeRunModelExecution({
    planned_source: 'deterministic_only',
    profile_revision: null,
    model: null,
    endpoint_configuration_sha256: null,
    configured: false,
    status: 'not_used',
  }, 'completed').label, '仅运行确定性检查，未调用模型');
});

test('provider failure remains explicit without exposing an upstream response', () => {
  const view = describeRunModelExecution({
    ...accountPlan,
    status: 'provider_failed',
    upstream_response: 'secret raw response',
  }, 'failed');
  assert.equal(view.tone, 'failed');
  assert.equal(view.label, '账户模型调用失败');
  assert.doesNotMatch(`${view.label} ${view.detail}`, /secret raw response/);

  const completedFallback = describeRunModelExecution({
    ...accountPlan,
    status: 'provider_failed',
  }, 'completed');
  assert.equal(completedFallback.tone, 'failed');
  assert.equal(completedFallback.label, '账户模型调用失败');
});

test('missing, malformed, and contradictory snapshots fail closed', () => {
  const malformed = [
    [null, 'completed'],
    [{...accountPlan, planned_source: 'other'}, 'queued'],
    [{...accountPlan, configured: 'yes'}, 'queued'],
    [{...accountPlan, status: 'used'}, 'queued'],
    [{...accountPlan, status: 'planned'}, 'completed'],
    [{
      ...accountPlan,
      planned_source: 'deterministic_only',
      configured: false,
      status: 'used',
    }, 'completed'],
    [{...accountPlan, status: 'used'}, 'mystery'],
  ];
  for (const [payload, runStatus] of malformed) {
    const view = describeRunModelExecution(payload, runStatus);
    assert.equal(view.tone, 'unknown');
    assert.equal(view.label, '历史模型来源未知');
    assert.match(view.detail, /不会用当前账户设置替代/);
  }
});
