import { passwordRuleMessage } from "./session.ts";

export type AccountSession = {
  id: string;
  current: boolean;
  created_at: string;
  last_seen_at: string;
  expires_at: string;
};

export type AccountSessionsResponse = {
  sessions: AccountSession[];
};

export type PasswordValues = {
  currentPassword: string;
  newPassword: string;
  confirmPassword: string;
};

export type PasswordFieldName = keyof PasswordValues;
export type PasswordFocusTarget = PasswordFieldName | "summary" | null;

const passwordFieldOrder: PasswordFieldName[] = [
  "currentPassword",
  "newPassword",
  "confirmPassword",
];

export function validatePasswordChange(
  values: PasswordValues,
): Partial<Record<PasswordFieldName, string>> {
  const errors: Partial<Record<PasswordFieldName, string>> = {};
  if (!values.currentPassword) errors.currentPassword = "请输入当前密码。";

  const newPasswordError = passwordRuleMessage(values.newPassword);
  if (newPasswordError) errors.newPassword = newPasswordError;
  else if (values.newPassword === values.currentPassword) {
    errors.newPassword = "新密码需要与当前密码不同。";
  }

  if (!values.confirmPassword) errors.confirmPassword = "请再次输入新密码。";
  else if (values.confirmPassword !== values.newPassword) {
    errors.confirmPassword = "两次输入的新密码不一致。";
  }
  return errors;
}

export function passwordErrorFocusTarget(
  errors: Partial<Record<PasswordFieldName, string>>,
  hasServerError = false,
): PasswordFocusTarget {
  if (hasServerError) return "summary";
  const invalidFields = passwordFieldOrder.filter((field) => Boolean(errors[field]));
  if (invalidFields.length > 1) return "summary";
  return invalidFields[0] || null;
}

export function formatSessionTime(value: string): string {
  const normalized = /(?:Z|[+-]\d{2}:?\d{2})$/.test(value) ? value : `${value}Z`;
  const date = new Date(normalized);
  if (Number.isNaN(date.getTime())) return "时间未知";
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}

export function passwordChangeSuccessMessage(revokedSessions: number): string {
  if (revokedSessions > 0) {
    return `密码已更新，其他 ${revokedSessions} 个会话已自动撤销；这些设备需要重新登录。`;
  }
  return "密码已更新。当前设备保持登录，没有其他有效会话需要撤销。";
}

export function passwordApiDetailMessage(status: number, detail: string): string {
  if (status === 400 && detail === "当前密码错误") {
    return "当前密码不正确，请重新输入。";
  }
  return detail;
}
