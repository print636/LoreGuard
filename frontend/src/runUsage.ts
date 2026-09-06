export type RunUsageInfo = {
  status: string;
  prompt_tokens: unknown;
  completion_tokens: unknown;
  estimated_cost_usd: unknown;
  usage_accounting?: unknown;
};

export type RunUsageView = {
  tokens: string;
  detail: string | null;
  cost: string;
  semantics: 'standard' | 'lower_bound' | 'unknown';
};

function nonnegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && (value as number) >= 0;
}

function formattedCost(value: unknown, prefix: string): string {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0
    ? `${prefix}：$${value.toFixed(6)}`
    : `${prefix}：未配置`;
}

function validCancelledLowerBound(run: RunUsageInfo): boolean {
  const accounting = run.usage_accounting;
  if (!accounting || typeof accounting !== 'object' || Array.isArray(accounting)) {
    return false;
  }
  const source = accounting as Record<string, unknown>;
  if (
    source.completeness !== 'lower_bound'
    || source.scope !== 'review_agent_completed_calls_only'
    || source.terminal_status !== 'cancelled'
    || !nonnegativeInteger(source.logical_calls)
    || source.logical_calls === 0
    || !nonnegativeInteger(source.prompt_tokens)
    || !nonnegativeInteger(source.completion_tokens)
    || !nonnegativeInteger(source.charged_tokens)
    || source.charged_token_semantics !== 'conservative_internal_budget_debit'
  ) {
    return false;
  }
  const providerTokens = source.prompt_tokens + source.completion_tokens;
  return (
    source.prompt_tokens === run.prompt_tokens
    && source.completion_tokens === run.completion_tokens
    && source.charged_tokens >= providerTokens
  );
}

export function describeRunUsage(run: RunUsageInfo | null): RunUsageView {
  if (!run) {
    return {
      tokens: '0', detail: null, cost: '成本：未配置', semantics: 'standard',
    };
  }
  const promptTokens = nonnegativeInteger(run.prompt_tokens)
    ? run.prompt_tokens : null;
  const completionTokens = nonnegativeInteger(run.completion_tokens)
    ? run.completion_tokens : null;
  const countersValid = promptTokens !== null && completionTokens !== null;
  if (run.status === 'cancelled') {
    if (!countersValid || !validCancelledLowerBound(run)) {
      return {
        tokens: '未知',
        detail: '取消态缺少可验证的完整用量，只能确认任务已停止',
        cost: '成本：不可确认',
        semantics: 'unknown',
      };
    }
    return {
      tokens: `≥${promptTokens! + completionTokens!}`,
      detail: '仅已完成 Review Agent 调用',
      cost: formattedCost(run.estimated_cost_usd, '成本下界'),
      semantics: 'lower_bound',
    };
  }
  if (!countersValid) {
    return {
      tokens: '未知', detail: 'Token 计数不可验证', cost: '成本：不可确认',
      semantics: 'unknown',
    };
  }
  return {
    tokens: String(promptTokens! + completionTokens!),
    detail: null,
    cost: formattedCost(run.estimated_cost_usd, '成本'),
    semantics: 'standard',
  };
}
