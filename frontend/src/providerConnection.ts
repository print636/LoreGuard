type JsonObject = Record<string, unknown>;

export type ProviderConnectionTone = 'neutral' | 'ready' | 'warning' | 'error';

export type ProviderConnectionView = {
  tone: ProviderConnectionTone;
  label: string;
  detail: string;
  configured: boolean | null;
  jsonContractOk: boolean | null;
  category: string;
  categoryLabel: string;
  latencyLabel: string;
  tokensLabel: string;
  suggestions: string[];
};

const categoryCopy:Record<string, {
  label:string;
  tone:ProviderConnectionTone;
  detail:string;
  suggestions:string[];
}> = {
  success: {
    label: '连接与 JSON 协议正常', tone: 'ready',
    detail: '模型服务可达、凭据已获授权，并通过最小 JSON 输出检查。',
    suggestions: ['可以开始一次短篇样例分析，确认业务抽取效果。'],
  },
  not_configured: {
    label: '服务端尚未配置模型', tone: 'warning',
    detail: '当前分析仍可运行确定性基线，但不会获得模型语义增强。',
    suggestions: ['在服务端未提交的 .env 中配置模型并启用模型抽取，然后重启 API 与 worker。'],
  },
  invalid_response: {
    label: '连接成功，但 JSON 协议不兼容', tone: 'warning',
    detail: '网络和授权正常，但模型没有按最小 JSON 合同返回；不能据此认为模型增强可用。',
    suggestions: ['检查所选模型是否支持严格 JSON 输出，再重启服务并重新测试。'],
  },
  unauthorized: {
    label: '凭据未获授权', tone: 'error',
    detail: '模型服务已响应，但拒绝了当前凭据。',
    suggestions: ['在服务端检查 API Key 是否有效、未过期且属于当前租户或项目，然后重启 API 与 worker。'],
  },
  forbidden: {
    label: '模型访问被拒绝', tone: 'error',
    detail: '服务已响应，但当前账户、模型、额度或访问策略不允许本次调用。',
    suggestions: ['检查账户额度、模型权限、租户、IP 白名单或区域策略，然后重试。'],
  },
  rate_limit: {
    label: '模型服务正在限流', tone: 'warning',
    detail: '服务可达，但当前调用频率或额度受到限制。',
    suggestions: ['稍后重试，并检查服务方额度与频率限制。'],
  },
  upstream_5xx: {
    label: '模型上游暂时异常', tone: 'warning',
    detail: '服务已响应，但上游暂时无法完成请求。',
    suggestions: ['稍后重试；若持续失败，请查看服务方状态。'],
  },
  connect_timeout: {
    label: '连接模型服务超时', tone: 'error',
    detail: '在限定时间内没有建立到模型服务的连接。',
    suggestions: ['检查服务端网络、Base URL 与代理配置，然后重试。'],
  },
  read_timeout: {
    label: '等待模型响应超时', tone: 'warning',
    detail: '连接已建立，但模型没有在限定时间内完成响应。',
    suggestions: ['稍后重试，或在服务端核对超时与模型负载配置。'],
  },
  transport: {
    label: '模型网络连接失败', tone: 'error',
    detail: '服务端没有完成到模型服务的网络请求。',
    suggestions: ['检查服务端网络、DNS、代理与 Base URL，然后重试。'],
  },
  response_too_large: {
    label: '模型响应超过安全上限', tone: 'warning',
    detail: '服务已响应，但结果超过当前安全大小限制。',
    suggestions: ['检查模型兼容性或在服务端调整经过评估的响应上限。'],
  },
};

function objectValue(value:unknown):JsonObject|null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? value as JsonObject : null;
}

function nullableBoolean(value:unknown):boolean|null {
  return typeof value === 'boolean' ? value : null;
}

function safeCount(value:unknown):number|null {
  return typeof value === 'number' && Number.isSafeInteger(value)
    && value >= 0 && value <= 2_147_483_647 ? value : null;
}

function usageLabel(payload:JsonObject):string {
  const usage = objectValue(payload.token_usage);
  if (!usage) return '未提供';
  const prompt = safeCount(usage.prompt);
  const completion = safeCount(usage.completion);
  const total = safeCount(usage.total);
  if (prompt === null || completion === null || total === null
    || prompt + completion !== total) return '未提供';
  return `${total}（输入 ${prompt} / 输出 ${completion}）`;
}

export const unknownProviderConnection:ProviderConnectionView = {
  tone: 'neutral', label: '正在读取服务端配置状态',
  detail: '页面只读取非敏感状态，不会读取或保存模型凭据。',
  configured: null, jsonContractOk: null, category: 'unknown',
  categoryLabel: '尚未测试', latencyLabel: '未测试', tokensLabel: '未调用',
  suggestions: [],
};

export function describeProviderHealth(payload:unknown):ProviderConnectionView {
  const root = objectValue(payload);
  const model = objectValue(root?.model);
  const configured = nullableBoolean(model?.configured);
  if (configured === true) return {
    ...unknownProviderConnection,
    label: '服务端已配置模型，连接尚未测试',
    detail: '点击“测试模型连接”后才会发起一次最小模型调用，并可能消耗少量 Token。',
    configured: true,
    category: 'not_tested',
  };
  if (configured === false) return {
    ...unknownProviderConnection,
    tone: 'warning', label: categoryCopy.not_configured.label,
    detail: categoryCopy.not_configured.detail,
    configured: false, category: 'not_configured',
    categoryLabel: '未配置',
    suggestions: categoryCopy.not_configured.suggestions,
  };
  return {
    ...unknownProviderConnection,
    tone: 'warning', label: '无法确认模型配置状态',
    detail: '服务端健康信息缺少可信的模型配置字段。',
    suggestions: ['确认 LoreGuard API 正常运行后刷新页面。'],
  };
}

export function describeProviderCheck(payload:unknown):ProviderConnectionView {
  const root = objectValue(payload);
  const rawCategory = typeof root?.category === 'string' ? root.category : 'unknown';
  const category = Object.hasOwn(categoryCopy, rawCategory) ? rawCategory : 'unknown';
  const copy = categoryCopy[category] || {
    label: '模型连接测试未得到可识别结果', tone: 'error' as const,
    detail: '服务端返回了未知的安全分类，页面不会展示未识别字段。',
    suggestions: ['查看服务端日志中的安全诊断后重试。'],
  };
  const configured = nullableBoolean(root?.configured);
  const jsonContractOk = nullableBoolean(root?.json_contract_ok);
  const latency = safeCount(root?.latency_ms);
  return {
    tone: copy.tone,
    label: copy.label,
    detail: copy.detail,
    configured,
    jsonContractOk,
    category,
    categoryLabel: category === 'unknown' ? '未知' : copy.label,
    latencyLabel: latency === null ? '未提供' : `${latency} ms`,
    tokensLabel: usageLabel(root || {}),
    suggestions: copy.suggestions,
  };
}
