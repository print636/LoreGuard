import test from 'node:test';
import assert from 'node:assert/strict';
import {describeRunUsage} from '../src/runUsage.ts';

const completed = {
  status:'completed', prompt_tokens:11, completion_tokens:7,
  estimated_cost_usd:0.000123,
};

const cancelled = {
  status:'cancelled', prompt_tokens:11, completion_tokens:7,
  estimated_cost_usd:0.000123,
  usage_accounting:{
    completeness:'lower_bound', scope:'review_agent_completed_calls_only',
    terminal_status:'cancelled', logical_calls:1, prompt_tokens:11,
    completion_tokens:7, charged_tokens:999,
    charged_token_semantics:'conservative_internal_budget_debit',
    provider_calls:null,
  },
};

test('completed runs keep the ordinary provider-token and cost display', () => {
  assert.deepEqual(describeRunUsage(completed), {
    tokens:'18', detail:null, cost:'成本：$0.000123', semantics:'standard',
  });
  assert.deepEqual(describeRunUsage({...completed, usage_accounting:{bad:true}}),
    describeRunUsage(completed));
});

test('qualified cancellation is an Agent-only lower bound and never uses charged tokens', () => {
  assert.deepEqual(describeRunUsage(cancelled), {
    tokens:'≥18', detail:'仅已完成 Review Agent 调用',
    cost:'成本下界：$0.000123', semantics:'lower_bound',
  });
  assert.doesNotMatch(describeRunUsage(cancelled).tokens, /999/);
});

test('missing, malformed, or inconsistent cancellation qualifiers fail closed', () => {
  const malformed = [
    {...cancelled, usage_accounting:undefined},
    {...cancelled, usage_accounting:{...cancelled.usage_accounting,
      scope:'all_model_calls'}},
    {...cancelled, usage_accounting:{...cancelled.usage_accounting,
      terminal_status:'completed'}},
    {...cancelled, usage_accounting:{...cancelled.usage_accounting,
      prompt_tokens:12}},
    {...cancelled, usage_accounting:{...cancelled.usage_accounting,
      logical_calls:0}},
    {...cancelled, usage_accounting:{...cancelled.usage_accounting,
      charged_tokens:17}},
    {...cancelled, usage_accounting:{...cancelled.usage_accounting,
      charged_token_semantics:'provider_actual_usage'}},
    {...cancelled, prompt_tokens:Number.MAX_SAFE_INTEGER + 1},
  ];
  for (const run of malformed) {
    const view = describeRunUsage(run);
    assert.equal(view.tokens, '未知');
    assert.equal(view.semantics, 'unknown');
    assert.match(view.detail, /取消态/);
    assert.equal(view.cost, '成本：不可确认');
  }
});
