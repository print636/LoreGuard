const SOURCES = new Set([
  "account_byok",
  "service_default",
  "service_default_legacy",
  "deterministic_only",
] as const);

const EXECUTION_STATUSES = new Set([
  "planned",
  "used",
  "not_used",
  "provider_failed",
] as const);

export type RunModelSource =
  | "account_byok"
  | "service_default"
  | "service_default_legacy"
  | "deterministic_only";

export type RunModelExecutionStatus =
  | "planned"
  | "used"
  | "not_used"
  | "provider_failed";

export type RunModelExecutionSnapshot = {
  source: RunModelSource | null;
  profileRevision: number | null;
  model: string | null;
  endpointFingerprint: string | null;
  configured: boolean | null;
  status: RunModelExecutionStatus | null;
};

export type RunModelExecutionView = {
  tone: "planned" | "used" | "not-used" | "failed" | "unknown";
  label: string;
  detail: string;
};

function objectValue(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function positiveInteger(value: unknown): number | null {
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0
    ? value
    : null;
}

function safeModel(value: unknown): string | null {
  if (typeof value !== "string" || value.length === 0 || value.length > 255) {
    return null;
  }
  if (value !== value.trim() || /[\u0000-\u001f\u007f]/.test(value)) return null;
  return value;
}

function endpointFingerprint(value: unknown): string | null {
  return typeof value === "string" && /^[a-f0-9]{64}$/.test(value)
    ? value.slice(0, 8)
    : null;
}

export function parseRunModelExecution(
  value: unknown,
): RunModelExecutionSnapshot | null {
  const root = objectValue(value);
  if (!root) return null;
  const source = SOURCES.has(root.planned_source as RunModelSource)
    ? (root.planned_source as RunModelSource)
    : null;
  const status = EXECUTION_STATUSES.has(
    root.status as RunModelExecutionStatus,
  )
    ? (root.status as RunModelExecutionStatus)
    : null;
  return {
    source,
    profileRevision: positiveInteger(root.profile_revision),
    model: safeModel(root.model),
    endpointFingerprint: endpointFingerprint(
      root.endpoint_configuration_sha256,
    ),
    configured:
      typeof root.configured === "boolean" ? root.configured : null,
    status,
  };
}

function sourceLabel(source: RunModelSource): string {
  if (source === "account_byok") return "账户模型";
  if (source === "service_default") return "服务默认模型";
  if (source === "service_default_legacy") {
    return "服务默认模型（历史兼容）";
  }
  return "仅确定性检查";
}

function frozenDetail(snapshot: RunModelExecutionSnapshot): string {
  const parts: string[] = [];
  if (snapshot.source === "account_byok" && snapshot.profileRevision !== null) {
    parts.push(`配置修订 r${snapshot.profileRevision}`);
  }
  if (snapshot.model) parts.push(`模型 ${snapshot.model}`);
  if (
    snapshot.configured === false &&
    snapshot.source !== "deterministic_only"
  ) {
    parts.push("冻结时配置不可用");
  }
  parts.push("本次历史运行的冻结快照，不随当前设置改变");
  return parts.join(" · ");
}

const UNKNOWN_VIEW: RunModelExecutionView = {
  tone: "unknown",
  label: "历史模型来源未知",
  detail: "该运行没有可验证的模型执行快照；不会用当前账户设置替代。",
};

export function describeRunModelExecution(
  value: unknown,
  runStatus: unknown,
): RunModelExecutionView {
  const snapshot = parseRunModelExecution(value);
  if (!snapshot?.source || !snapshot.status || snapshot.configured === null) {
    return UNKNOWN_VIEW;
  }

  const active = runStatus === "queued" || runStatus === "running";
  const completed = runStatus === "completed";
  const failed = runStatus === "failed";
  const cancelled = runStatus === "cancelled";
  if (!active && !completed && !failed && !cancelled) return UNKNOWN_VIEW;
  if (active && snapshot.status !== "planned") return UNKNOWN_VIEW;
  if (
    completed &&
    !["used", "not_used", "provider_failed"].includes(snapshot.status)
  ) {
    return UNKNOWN_VIEW;
  }
  if (
    (failed || cancelled) &&
    !["used", "not_used", "provider_failed"].includes(snapshot.status)
  ) {
    return UNKNOWN_VIEW;
  }
  if (snapshot.source === "deterministic_only" && snapshot.configured) {
    return UNKNOWN_VIEW;
  }
  if (snapshot.source !== "deterministic_only" && !snapshot.configured) {
    return UNKNOWN_VIEW;
  }
  if (
    snapshot.source === "deterministic_only" &&
    (snapshot.status === "used" || snapshot.status === "provider_failed")
  ) {
    return UNKNOWN_VIEW;
  }

  const source = sourceLabel(snapshot.source);
  const detail = frozenDetail(snapshot);
  if (snapshot.source === "deterministic_only") {
    return {
      tone: snapshot.status === "planned" ? "planned" : "not-used",
      label:
        snapshot.status === "planned"
          ? "计划仅运行确定性检查"
          : "仅运行确定性检查，未调用模型",
      detail,
    };
  }
  if (snapshot.status === "planned") {
    return { tone: "planned", label: `计划使用${source}`, detail };
  }
  if (snapshot.status === "used") {
    return { tone: "used", label: `实际使用${source}`, detail };
  }
  if (snapshot.status === "not_used") {
    return {
      tone: "not-used",
      label: `${source}未产生可确认的模型结果`,
      detail,
    };
  }
  return { tone: "failed", label: `${source}调用失败`, detail };
}
