import assert from "node:assert/strict";
import test from "node:test";
import {
  formatSessionTime,
  passwordApiDetailMessage,
  passwordErrorFocusTarget,
  passwordChangeSuccessMessage,
  validatePasswordChange,
} from "../src/app/accountSecurity.ts";

test("password change validates required fields, length, equality, and confirmation", () => {
  assert.deepEqual(
    validatePasswordChange({ currentPassword: "", newPassword: "short", confirmPassword: "different" }),
    {
      currentPassword: "请输入当前密码。",
      newPassword: "密码至少需要 10 个字符。",
      confirmPassword: "两次输入的新密码不一致。",
    },
  );
  assert.deepEqual(
    validatePasswordChange({ currentPassword: "same-password", newPassword: "same-password", confirmPassword: "same-password" }),
    { newPassword: "新密码需要与当前密码不同。" },
  );
  assert.deepEqual(
    validatePasswordChange({ currentPassword: "old-password", newPassword: "new-password", confirmPassword: "new-password" }),
    {},
  );
});

test("password validation focuses one invalid field or the multi-error summary", () => {
  assert.equal(
    passwordErrorFocusTarget({ currentPassword: "请输入当前密码。" }),
    "currentPassword",
  );
  assert.equal(
    passwordErrorFocusTarget({ newPassword: "密码过短。", confirmPassword: "两次输入不一致。" }),
    "summary",
  );
  assert.equal(passwordErrorFocusTarget({}, true), "summary");
  assert.equal(passwordErrorFocusTarget({}), null);
});

test("password success copy discloses that other sessions are automatically revoked", () => {
  assert.equal(
    passwordChangeSuccessMessage(2),
    "密码已更新，其他 2 个会话已自动撤销；这些设备需要重新登录。",
  );
  assert.equal(
    passwordChangeSuccessMessage(0),
    "密码已更新。当前设备保持登录，没有其他有效会话需要撤销。",
  );
});

test("password API copy only maps the exact current-password error", () => {
  assert.equal(
    passwordApiDetailMessage(400, "当前密码错误"),
    "当前密码不正确，请重新输入。",
  );
  assert.equal(
    passwordApiDetailMessage(400, "新密码不能与当前密码相同"),
    "新密码不能与当前密码相同",
  );
  assert.equal(
    passwordApiDetailMessage(401, "当前密码错误"),
    "当前密码错误",
  );
});

test("session times accept backend UTC-naive timestamps and reject malformed values", () => {
  assert.notEqual(formatSessionTime("2026-09-19T09:30:00"), "时间未知");
  assert.equal(formatSessionTime("not-a-time"), "时间未知");
});
