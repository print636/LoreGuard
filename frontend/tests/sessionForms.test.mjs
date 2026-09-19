import test from "node:test";
import assert from "node:assert/strict";

import { emailRuleMessage, passwordRuleMessage } from "../src/app/session.ts";

test("authentication validation explains the actual server-side boundaries", () => {
  assert.equal(emailRuleMessage(""), "请输入邮箱。");
  assert.equal(emailRuleMessage("author@example.com"), "");
  assert.equal(passwordRuleMessage("short"), "密码至少需要 10 个字符。");
  assert.equal(passwordRuleMessage("correct horse battery staple"), "");
});
