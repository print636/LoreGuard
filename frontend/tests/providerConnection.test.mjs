import test from 'node:test';
import assert from 'node:assert/strict';

import {
  describeProviderCheck, describeProviderHealth, unknownProviderConnection,
} from '../src/providerConnection.ts';

test('passive health describes configuration without claiming a live check', () => {
  const configured = describeProviderHealth({model:{configured:true}});
  assert.equal(configured.configured, true);
  assert.equal(configured.category, 'not_tested');
  assert.equal(configured.jsonContractOk, null);
  assert.match(configured.detail, /点击.*测试模型连接/);

  const disabled = describeProviderHealth({model:{configured:false}});
  assert.equal(disabled.configured, false);
  assert.equal(disabled.category, 'not_configured');
  assert.match(disabled.suggestions[0], /服务端未提交的 \.env/);
});

test('successful check shows only bounded safe metrics and JSON contract state', () => {
  const view = describeProviderCheck({
    category:'success', configured:true, json_contract_ok:true,
    reachable:true, authorized:true, latency_ms:321,
    token_usage:{prompt:9, completion:4, total:13},
  });
  assert.equal(view.tone, 'ready');
  assert.equal(view.jsonContractOk, true);
  assert.equal(view.latencyLabel, '321 ms');
  assert.equal(view.tokensLabel, '13（输入 9 / 输出 4）');
});

test('known failures become actionable non-technical categories', () => {
  const forbidden = describeProviderCheck({
    category:'forbidden', configured:true, json_contract_ok:null,
    latency_ms:87, token_usage:{prompt:null, completion:null, total:null},
  });
  assert.equal(forbidden.tone, 'error');
  assert.match(forbidden.label, /访问被拒绝/);
  assert.match(forbidden.suggestions[0], /额度/);

  const invalid = describeProviderCheck({
    category:'invalid_response', configured:true, json_contract_ok:false,
    latency_ms:55, token_usage:{prompt:6, completion:3, total:9},
  });
  assert.equal(invalid.jsonContractOk, false);
  assert.match(invalid.label, /JSON/);
});

test('unknown and malformed fields fail closed without rendering upstream text', () => {
  const privateText = 'sk-private https://private.invalid raw prompt';
  const view = describeProviderCheck({
    category:'brand_new_failure', configured:'yes', json_contract_ok:'yes',
    latency_ms:Number.MAX_SAFE_INTEGER, token_usage:{prompt:1, completion:2, total:99},
    endpoint:privateText, key:privateText, prompt:privateText,
    raw_response:privateText, suggestions:[privateText],
  });
  const rendered = JSON.stringify(view);
  assert.equal(view.category, 'unknown');
  assert.equal(view.configured, null);
  assert.equal(view.jsonContractOk, null);
  assert.equal(view.latencyLabel, '未提供');
  assert.equal(view.tokensLabel, '未提供');
  assert.doesNotMatch(rendered, /sk-private|private\.invalid|raw prompt/);
  assert.equal(unknownProviderConnection.tokensLabel, '未调用');
});
