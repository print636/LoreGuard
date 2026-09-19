import { ApiError } from "../api/client.ts";

export type SessionIdentity = {
  mode: "anonymous" | "required";
  user: {
    id: string;
    email: string;
    display_name: string;
  };
  workspace: {
    id: string;
    name: string;
    kind: string;
    role: string;
  };
};

export function apiErrorDetail(error: unknown): string {
  if (!(error instanceof ApiError)) return "服务暂时不可用，请稍后重试。";
  const payload = error.detail;
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail?: unknown }).detail;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) =>
          item && typeof item === "object" && "msg" in item
            ? String((item as { msg: unknown }).msg)
            : "",
        )
        .filter(Boolean);
      if (messages.length) return messages.join("；");
    }
  }
  if (error.status === 429) return "请求过于频繁，请稍后再试。";
  if (error.status >= 500) return "服务暂时不可用，请稍后重试。";
  return error.message || "请求没有完成，请重试。";
}

export function passwordRuleMessage(value: string): string {
  if (!value) return "请输入密码。";
  if (value.length < 10) return "密码至少需要 10 个字符。";
  if (value.length > 128) return "密码不能超过 128 个字符。";
  return "";
}

export function emailRuleMessage(value: string): string {
  const email = value.trim();
  if (!email) return "请输入邮箱。";
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]+$/.test(email)) return "请输入有效的邮箱地址。";
  return "";
}
