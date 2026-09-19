const API_BASE = (
  (import.meta as ImportMeta & { env?: { VITE_API_BASE?: string } }).env
    ?.VITE_API_BASE || ""
).replace(/\/$/, "");

const SAFE_METHODS = new Set(["GET", "HEAD", "OPTIONS"]);
const CSRF_COOKIE = "loreguard_csrf";
const AUTH_ENTRY_PATHS = new Set([
  "/api/v1/auth/login",
  "/api/v1/auth/register",
  "/api/v1/auth/me",
]);
export const SESSION_EXPIRED_EVENT = "loreguard:session-expired";
export type SessionProbeResult = "active" | "expired" | "unknown" | "skipped";

type ApiErrorOptions = {
  status: number;
  statusText: string;
  detail?: unknown;
};

export class ApiError extends Error {
  readonly status: number;
  readonly statusText: string;
  readonly detail?: unknown;

  constructor(message: string, options: ApiErrorOptions) {
    super(message);
    this.name = "ApiError";
    this.status = options.status;
    this.statusText = options.statusText;
    this.detail = options.detail;
  }
}

export function apiUrl(path: string): string {
  if (!path.startsWith("/")) {
    throw new TypeError(`API path must start with "/": ${path}`);
  }
  return `${API_BASE}${path}`;
}

export function readCookie(name: string, cookie = document.cookie): string | null {
  const prefix = `${encodeURIComponent(name)}=`;
  for (const part of cookie.split(";")) {
    const value = part.trim();
    if (!value.startsWith(prefix)) continue;
    try {
      return decodeURIComponent(value.slice(prefix.length));
    } catch {
      return value.slice(prefix.length);
    }
  }
  return null;
}

function requestMethod(init: RequestInit): string {
  return (init.method || "GET").toUpperCase();
}

function requestHeaders(init: RequestInit): Headers {
  const headers = new Headers(init.headers);
  if (!SAFE_METHODS.has(requestMethod(init)) && !headers.has("X-CSRF-Token")) {
    const token = readCookie(CSRF_COOKIE);
    if (token) headers.set("X-CSRF-Token", token);
  }
  return headers;
}

export async function apiFetch(
  path: string,
  init: RequestInit = {},
): Promise<Response> {
  const response = await fetch(apiUrl(path), {
    ...init,
    credentials: "include",
    headers: requestHeaders(init),
  });
  if (
    response.status === 401 &&
    !AUTH_ENTRY_PATHS.has(path) &&
    typeof window !== "undefined"
  ) {
    dispatchSessionExpired();
  }
  return response;
}

export function dispatchSessionExpired(): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new Event(SESSION_EXPIRED_EVENT));
  }
}

export async function probeCurrentSession(
  timeoutMs = 4_000,
): Promise<Exclude<SessionProbeResult, "skipped">> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await apiFetch("/api/v1/auth/me", {
      signal: controller.signal,
    });
    if (response.status === 401) {
      dispatchSessionExpired();
      return "expired";
    }
    return response.ok ? "active" : "unknown";
  } catch {
    return "unknown";
  } finally {
    clearTimeout(timeout);
  }
}

export function createBoundedSessionProbe(
  minimumIntervalMs = 5_000,
  probe: () => Promise<Exclude<SessionProbeResult, "skipped">> = probeCurrentSession,
): () => Promise<SessionProbeResult> {
  let inFlight: Promise<Exclude<SessionProbeResult, "skipped">> | null = null;
  let lastStartedAt = Number.NEGATIVE_INFINITY;

  return () => {
    const now = Date.now();
    if (inFlight || now - lastStartedAt < minimumIntervalMs) {
      return Promise.resolve("skipped");
    }
    lastStartedAt = now;
    const current = probe().catch(() => "unknown" as const);
    inFlight = current;
    void current.finally(() => {
      if (inFlight === current) inFlight = null;
    });
    return current;
  };
}

function errorMessage(payload: unknown, response: Response): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string" && detail.trim()) return detail;
  }
  return response.statusText || `HTTP ${response.status}`;
}

export async function apiJson<T = unknown>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const response = await apiFetch(path, init);
  const raw = await response.text();
  let payload: unknown = null;

  if (raw) {
    try {
      payload = JSON.parse(raw);
    } catch (cause) {
      throw new ApiError("API 返回了无法解析的 JSON", {
        status: response.status,
        statusText: response.statusText,
        detail: cause,
      });
    }
  }

  if (!response.ok) {
    throw new ApiError(errorMessage(payload, response), {
      status: response.status,
      statusText: response.statusText,
      detail: payload,
    });
  }

  if (response.status === 204) return undefined as T;

  return payload as T;
}
