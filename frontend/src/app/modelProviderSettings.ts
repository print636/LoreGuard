import type { ProviderConnectionView } from "../providerConnection";

type JsonObject = Record<string, unknown>;

export const SAVED_API_KEY_MASK = "••••••••••••";
export const MODEL_TEST_DISCLAIMER =
  "最小连接测试只验证网络、授权和基础 JSON 响应，不代表长文本、额度、并发或完整分析一定可用。";

export type ModelProviderTestTone = "neutral" | "ready" | "warning" | "error";

export type ModelProviderTestView = {
  category: string;
  label: string;
  detail: string;
  tone: ModelProviderTestTone;
  testedAt: string | null;
  profileRevision: number | null;
  jsonContractOk: boolean | null;
  latencyMs: number | null;
  tokenTotal: number | null;
  action: string;
};

export type AccountModelProviderProfile = {
  configured: boolean;
  revision: number;
  baseUrl: string;
  model: string;
  updatedAt: string | null;
  serviceDefaultAvailable: boolean;
  lastTest: ModelProviderTestView | null;
};

export type ModelProviderForm = {
  baseUrl: string;
  model: string;
  apiKey: string;
  replacingKey: boolean;
};

export type ModelProviderField = "baseUrl" | "model" | "apiKey";
export type ModelProviderErrors = Partial<Record<ModelProviderField, string>>;

const testCategoryCopy: Record<
  string,
  Omit<ModelProviderTestView, "category" | "testedAt" | "profileRevision" | "jsonContractOk" | "latencyMs" | "tokenTotal">
> = {
  success: {
    label: "最小连接测试通过",
    detail: "模型服务可达，当前凭据已授权，并通过基础 JSON 响应检查。",
    tone: "ready",
    action: "可以先用短篇样例确认实际抽取质量。",
  },
  unauthorized: {
    label: "API Key 未获授权",
    detail: "模型服务拒绝了当前账户保存的凭据。",
    tone: "error",
    action: "替换 API Key 后重新测试。",
  },
  forbidden: {
    label: "模型访问被拒绝",
    detail: "服务已响应，但当前凭据、账户额度或模型权限不允许调用。",
    tone: "error",
    action: "检查模型权限与账户额度，必要时更换模型后重试。",
  },
  rate_limit: {
    label: "模型服务正在限流",
    detail: "服务可达，但本次最小请求受到频率或额度限制。",
    tone: "warning",
    action: "稍后重试，并检查服务方的额度与频率限制。",
  },
  upstream_5xx: {
    label: "模型上游暂时异常",
    detail: "服务已响应，但上游暂时无法完成请求。",
    tone: "warning",
    action: "稍后重试；若持续失败，请查看服务方状态。",
  },
  connect_timeout: {
    label: "连接模型服务超时",
    detail: "LoreGuard 未能在限定时间内连接到模型服务。",
    tone: "error",
    action: "检查 Base URL、网络和代理设置后重试。",
  },
  read_timeout: {
    label: "等待模型响应超时",
    detail: "连接已经建立，但模型没有及时完成最小响应。",
    tone: "warning",
    action: "稍后重试，或检查模型服务负载。",
  },
  transport: {
    label: "模型网络连接失败",
    detail: "LoreGuard 没有完成到模型服务的网络请求。",
    tone: "error",
    action: "检查 Base URL、DNS、代理和网络连通性后重试。",
  },
  invalid_response: {
    label: "基础 JSON 响应不兼容",
    detail: "网络和授权可能正常，但模型没有按最小 JSON 合同返回。",
    tone: "warning",
    action: "确认所选模型支持 JSON 输出，或更换兼容模型。",
  },
  response_too_large: {
    label: "模型响应超过安全上限",
    detail: "服务已响应，但最小测试结果超过 LoreGuard 的安全大小限制。",
    tone: "warning",
    action: "检查网关兼容性或更换模型后重试。",
  },
  unsupported_content_encoding: {
    label: "响应压缩格式不受支持",
    detail: "模型服务返回了 LoreGuard 无法安全读取的压缩格式。",
    tone: "warning",
    action: "检查中转网关的响应压缩配置，或改用兼容的服务地址。",
  },
  response_decompression: {
    label: "模型响应解压失败",
    detail: "模型服务返回的压缩响应不完整或格式异常。",
    tone: "warning",
    action: "稍后重试；若持续失败，请检查中转网关兼容性。",
  },
  nonretry_http: {
    label: "模型接口拒绝了请求",
    detail: "服务已返回不可重试的 HTTP 错误，但没有暴露上游正文。",
    tone: "error",
    action: "核对 Base URL 是否为 OpenAI-compatible 接口根路径，并检查模型名。",
  },
  body_json: {
    label: "模型响应不是有效 JSON",
    detail: "服务可达，但响应体不符合 OpenAI-compatible JSON 格式。",
    tone: "warning",
    action: "检查中转网关兼容性，或更换模型后重试。",
  },
  response_shape: {
    label: "模型响应结构不兼容",
    detail: "服务返回了 JSON，但缺少 LoreGuard 所需的标准响应字段。",
    tone: "warning",
    action: "确认接口兼容 Chat Completions 响应结构。",
  },
  empty_content: {
    label: "模型返回了空内容",
    detail: "请求已完成，但模型没有返回可校验的正文。",
    tone: "warning",
    action: "重试一次；若持续发生，请更换模型或检查网关策略。",
  },
  usage_shape: {
    label: "Token 用量字段不兼容",
    detail: "响应正文已到达，但用量字段不是受支持的安全格式。",
    tone: "warning",
    action: "检查中转网关的 OpenAI-compatible usage 字段。",
  },
  truncated: {
    label: "最小测试响应被截断",
    detail: "模型在完成基础 JSON 响应前触发了输出上限。",
    tone: "warning",
    action: "检查模型是否支持简短 JSON 输出，或更换兼容模型。",
  },
  content_json: {
    label: "模型正文不是有效 JSON",
    detail: "标准响应结构存在，但模型正文没有满足最小 JSON 合同。",
    tone: "warning",
    action: "确认模型支持 JSON 输出，或更换兼容模型。",
  },
  endpoint_rejected: {
    label: "模型地址未通过安全校验",
    detail: "Base URL 的 HTTPS 来源不在部署允许列表中。",
    tone: "error",
    action: "请部署管理员核对 ACCOUNT_MODEL_ALLOWED_ORIGINS 后重新保存。",
  },
  credential_invalid: {
    label: "API Key 格式无效",
    detail: "保存的凭据不能安全地写入 Authorization 请求头。",
    tone: "error",
    action: "使用服务商提供的可见 ASCII API Key 重新保存。",
  },
  credential_revoked: {
    label: "模型凭据已被撤销",
    detail: "连接测试期间配置已被替换、删除或失效。",
    tone: "error",
    action: "重新加载当前配置，确认后再次测试。",
  },
  credential_reflected: {
    label: "模型服务返回了敏感凭据",
    detail: "上游响应包含账户凭据，LoreGuard 已丢弃整次响应且未进入分析。",
    tone: "error",
    action: "立即停用并更换该 API Key，同时检查中转服务的安全性。",
  },
  provider: {
    label: "模型连接测试失败",
    detail: "LoreGuard 安全地收敛了无法进一步分类的模型错误。",
    tone: "error",
    action: "重新测试；若持续失败，请由部署管理员查看服务日志。",
  },
};

function objectValue(value: unknown): JsonObject | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as JsonObject)
    : null;
}

function safeString(value: unknown, maxLength: number): string {
  return typeof value === "string" && value.length <= maxLength ? value : "";
}

function safeBoolean(value: unknown, fallback = false): boolean {
  return typeof value === "boolean" ? value : fallback;
}

function safeInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : null;
}

function safeTimestamp(value: unknown): string | null {
  if (typeof value !== "string" || value.length > 80 || !value.trim()) return null;
  return Number.isFinite(Date.parse(value)) ? value : null;
}

function parseTokenTotal(value: unknown): number | null {
  const usage = objectValue(value);
  if (!usage) return null;
  const prompt = safeInteger(usage.prompt);
  const completion = safeInteger(usage.completion);
  const total = safeInteger(usage.total);
  return prompt !== null && completion !== null && total !== null && prompt + completion === total
    ? total
    : null;
}

export function describeAccountProviderTest(payload: unknown): ModelProviderTestView {
  const root = objectValue(payload) || {};
  const rawCategory = safeString(root.category, 80);
  const category = Object.hasOwn(testCategoryCopy, rawCategory) ? rawCategory : "unknown";
  const copy = testCategoryCopy[category] || {
    label: "无法识别连接测试结果",
    detail: "LoreGuard 收到了未知的安全分类，页面不会显示未识别内容。",
    tone: "error" as const,
    action: "重新加载配置后再试；若仍失败，请查看 LoreGuard 服务日志。",
  };
  return {
    category,
    ...copy,
    testedAt: safeTimestamp(root.tested_at),
    profileRevision: safeInteger(root.profile_revision),
    jsonContractOk:
      typeof root.json_contract_ok === "boolean" ? root.json_contract_ok : null,
    latencyMs: safeInteger(root.latency_ms),
    tokenTotal: parseTokenTotal(root.token_usage),
  };
}

export function parseAccountModelProvider(payload: unknown): AccountModelProviderProfile {
  const root = objectValue(payload);
  if (!root || typeof root.configured !== "boolean") {
    throw new TypeError("模型配置响应缺少可信的 configured 字段");
  }
  const revision = safeInteger(root.revision) ?? 0;
  const baseUrl = safeString(root.base_url, 2_048);
  const model = safeString(root.model, 255);
  if (root.configured && (!revision || !baseUrl || !model)) {
    throw new TypeError("已配置模型的安全元数据不完整");
  }
  const lastTestPayload = objectValue(root.last_test);
  return {
    configured: root.configured,
    revision,
    baseUrl,
    model,
    updatedAt: safeTimestamp(root.updated_at),
    serviceDefaultAvailable: safeBoolean(root.service_default_available),
    lastTest: lastTestPayload ? describeAccountProviderTest(lastTestPayload) : null,
  };
}

export function formFromModelProvider(
  profile: AccountModelProviderProfile,
): ModelProviderForm {
  return {
    baseUrl: profile.baseUrl,
    model: profile.model,
    apiKey: "",
    replacingKey: false,
  };
}

export function emptyModelProviderForm(): ModelProviderForm {
  return { baseUrl: "", model: "", apiKey: "", replacingKey: false };
}

export function validateModelProviderForm(
  form: ModelProviderForm,
  configured: boolean,
): ModelProviderErrors {
  const errors: ModelProviderErrors = {};
  const baseUrl = form.baseUrl.trim();
  if (!baseUrl) errors.baseUrl = "请输入 Base URL。";
  else if (form.baseUrl !== baseUrl) errors.baseUrl = "Base URL 首尾不能包含空格。";
  else if (baseUrl.length > 2_048) errors.baseUrl = "Base URL 不能超过 2048 个字符。";
  else {
    try {
      const parsed = new URL(baseUrl);
      if (parsed.protocol !== "https:") errors.baseUrl = "Base URL 必须使用 HTTPS。";
      else if (parsed.username || parsed.password || parsed.search || parsed.hash) {
        errors.baseUrl = "Base URL 不能包含账户信息、查询参数或片段。";
      }
    } catch {
      errors.baseUrl = "请输入完整且有效的 HTTPS Base URL。";
    }
  }

  const model = form.model.trim();
  if (!model) errors.model = "请输入模型名称。";
  else if (form.model !== model) errors.model = "模型名称首尾不能包含空格。";
  else if (model.length > 255) errors.model = "模型名称不能超过 255 个字符。";

  if (!configured || form.replacingKey) {
    if (!form.apiKey.trim()) errors.apiKey = "请输入 API Key。";
    else if (form.apiKey !== form.apiKey.trim()) errors.apiKey = "API Key 首尾不能包含空格。";
    else if (form.apiKey.length > 4_096) errors.apiKey = "API Key 超过安全长度限制。";
    else if ([...form.apiKey].some((character) => {
      const code = character.codePointAt(0) ?? 0;
      return code < 0x21 || code > 0x7e;
    })) errors.apiKey = "API Key 只能包含可安全写入请求头的 ASCII 字符。";
  }
  return errors;
}

export function modelProviderFormDirty(
  form: ModelProviderForm,
  profile: AccountModelProviderProfile,
): boolean {
  return (
    form.baseUrl.trim() !== profile.baseUrl ||
    form.model.trim() !== profile.model ||
    form.replacingKey ||
    Boolean(form.apiKey)
  );
}

export function canTestModelProvider(
  profile: AccountModelProviderProfile,
  form: ModelProviderForm,
  pending = false,
): boolean {
  return profile.configured && !pending && !modelProviderFormDirty(form, profile);
}

export function modelProviderTestIsStale(
  result: ModelProviderTestView | null,
  profileRevision: number,
): boolean {
  return result !== null && result.profileRevision !== profileRevision;
}

export function modelProviderSavePayload(
  form: ModelProviderForm,
  profile: AccountModelProviderProfile,
): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    base_url: form.baseUrl.trim(),
    model: form.model.trim(),
    expected_revision: profile.revision,
  };
  if (!profile.configured || form.replacingKey) payload.api_key = form.apiKey.trim();
  return payload;
}

export function modelProviderSourceLabel(
  profile: AccountModelProviderProfile | null,
): string {
  if (profile?.configured) return `账户模型 · ${profile.model}`;
  if (profile?.serviceDefaultAvailable) return "服务默认模型";
  return "账户模型未设置";
}

export function modelProviderDeleteWarning(
  profile: AccountModelProviderProfile,
): string {
  const nextRun = profile.serviceDefaultAvailable
    ? "新运行可能改用服务默认模型。"
    : "新运行将只执行确定性检查，不会借用服务端模型密钥。";
  return `删除后，${nextRun}已经保存的故事项目不会删除。`;
}

export function modelProviderDeleteSuccess(
  profile: AccountModelProviderProfile,
): string {
  return profile.serviceDefaultAvailable
    ? "账户模型配置已删除。后续新运行可能使用服务默认模型。"
    : "账户模型配置已删除。后续新运行只执行确定性检查。";
}

export function describeAccountProviderConnection(
  profile: AccountModelProviderProfile,
): ProviderConnectionView {
  if (profile.configured) {
    const currentTest =
      profile.lastTest?.profileRevision === profile.revision ? profile.lastTest : null;
    const tested = currentTest !== null;
    return {
      tone: tested ? currentTest.tone : "neutral",
      label: tested ? `账户模型：${currentTest.label}` : "账户模型已设置，尚未测试",
      detail: `${profile.model} · 连接状态不代表完整故事审查可用。`,
      configured: true,
      jsonContractOk: currentTest?.jsonContractOk ?? null,
      category: currentTest?.category ?? "not_tested",
      categoryLabel: currentTest?.label ?? "尚未测试",
      latencyLabel:
        currentTest?.latencyMs === null || currentTest?.latencyMs === undefined
          ? "未提供"
          : `${currentTest.latencyMs} ms`,
      tokensLabel:
        currentTest?.tokenTotal === null || currentTest?.tokenTotal === undefined
          ? "未提供"
          : String(currentTest.tokenTotal),
      suggestions: currentTest ? [currentTest.action] : ["可以在“模型与密钥”中执行最小连接测试。"],
    };
  }
  if (profile.serviceDefaultAvailable) {
    return {
      tone: "neutral",
      label: "将使用服务默认模型",
      detail: "当前没有账户密钥；新运行会使用本机服务默认配置。",
      configured: true,
      jsonContractOk: null,
      category: "service_default",
      categoryLabel: "服务默认",
      latencyLabel: "未测试",
      tokensLabel: "未调用",
      suggestions: ["如需使用自己的模型，请打开“模型与密钥”。"],
    };
  }
  return {
    tone: "warning",
    label: "账户模型未设置",
    detail: "当前没有可确认的账户模型；模型增强是否执行以运行诊断为准。",
    configured: false,
    jsonContractOk: null,
    category: "not_configured",
    categoryLabel: "未设置",
    latencyLabel: "未测试",
    tokensLabel: "未调用",
    suggestions: ["打开“模型与密钥”保存自己的兼容模型。"],
  };
}

export function providerConflictMessage(status: number): string | null {
  return status === 409
    ? "模型配置已在其他页面更新。请重新加载最新配置，再提交你的更改。"
    : null;
}
