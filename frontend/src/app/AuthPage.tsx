import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, apiJson } from "../api/client";
import { browserNavigate, safeReturnTo } from "../routing";
import {
  apiErrorDetail,
  emailRuleMessage,
  passwordRuleMessage,
  type SessionIdentity,
} from "./session";

type AuthMode = "login" | "register";
type FieldName = "displayName" | "email" | "password" | "confirmPassword";
type FieldErrors = Partial<Record<FieldName, string>>;

type AuthPageProps = {
  mode: AuthMode;
  onAuthenticated: (identity: SessionIdentity) => void;
};

function EyeIcon({ crossed }: { crossed: boolean }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" />
      <circle cx="12" cy="12" r="2.5" />
      {crossed && <path d="m4 4 16 16" />}
    </svg>
  );
}

function validate(
  mode: AuthMode,
  values: Record<FieldName, string>,
): FieldErrors {
  const errors: FieldErrors = {};
  const emailError = emailRuleMessage(values.email);
  if (emailError) errors.email = emailError;
  const passwordError =
    mode === "register"
      ? passwordRuleMessage(values.password)
      : values.password
        ? ""
        : "请输入密码。";
  if (passwordError) errors.password = passwordError;
  if (mode === "register") {
    if (!values.displayName.trim()) errors.displayName = "请输入显示名称。";
    if (!values.confirmPassword) errors.confirmPassword = "请再次输入密码。";
    else if (values.confirmPassword !== values.password)
      errors.confirmPassword = "两次输入的密码不一致。";
  }
  return errors;
}

export default function AuthPage({ mode, onAuthenticated }: AuthPageProps) {
  const isRegister = mode === "register";
  const [values, setValues] = useState<Record<FieldName, string>>({
    displayName: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [touched, setTouched] = useState<Partial<Record<FieldName, boolean>>>({});
  const [submitted, setSubmitted] = useState(false);
  const [pending, setPending] = useState(false);
  const [serverError, setServerError] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmation, setShowConfirmation] = useState(false);
  const summaryRef = useRef<HTMLDivElement | null>(null);
  const errors = useMemo(() => validate(mode, values), [mode, values]);
  const visibleErrors = Object.entries(errors).filter(
    ([name]) => submitted || touched[name as FieldName],
  );

  useEffect(() => {
    setSubmitted(false);
    setServerError("");
    setTouched({});
  }, [mode]);

  function update(name: FieldName, value: string) {
    setValues((current) => ({ ...current, [name]: value }));
    setServerError("");
  }

  function fieldError(name: FieldName): string {
    return submitted || touched[name] ? errors[name] || "" : "";
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitted(true);
    setServerError("");
    if (Object.keys(errors).length) {
      requestAnimationFrame(() => summaryRef.current?.focus());
      return;
    }
    try {
      setPending(true);
      const identity = await apiJson<SessionIdentity>(
        `/api/v1/auth/${isRegister ? "register" : "login"}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(
            isRegister
              ? {
                  display_name: values.displayName.trim(),
                  email: values.email.trim(),
                  password: values.password,
                }
              : { email: values.email.trim(), password: values.password },
          ),
        },
      );
      onAuthenticated(identity);
      const params = new URLSearchParams(window.location.search);
      browserNavigate(safeReturnTo(params.get("returnTo")), { replace: true });
    } catch (error) {
      const detail = apiErrorDetail(error);
      if (error instanceof ApiError && error.status === 409 && detail.includes("邮箱")) {
        setSubmitted(true);
        setServerError("该邮箱已经注册，请直接登录。 ");
      } else if (error instanceof ApiError && error.status === 401) {
        setServerError("邮箱或密码错误，请检查后重试。");
      } else if (error instanceof ApiError && error.status === 409) {
        setServerError("当前服务处于本地体验模式，无需登录即可使用。");
      } else {
        setServerError(detail);
      }
      requestAnimationFrame(() => summaryRef.current?.focus());
    } finally {
      setPending(false);
    }
  }

  const switchTarget = isRegister ? "/login" : "/register";
  const returnTo = new URLSearchParams(window.location.search).get("returnTo");
  const switchHref = returnTo
    ? `${switchTarget}?returnTo=${encodeURIComponent(safeReturnTo(returnTo))}`
    : switchTarget;

  return (
    <main className="authPage productPage" id="main-content">
      <a className="skipLink" href="#auth-form">跳到登录表单</a>
      <section className="authStory" aria-labelledby="auth-story-title">
        <div className="brandLockup">
          <span className="productSeal" aria-hidden="true"><i>LG</i></span>
          <span><b>LoreGuard</b><small>故事剧情一致性审查</small></span>
        </div>
        <div className="authSpirit" aria-hidden="true">
          <i /><i /><b>··</b><span />
        </div>
        <div className="authStoryCopy">
          <h1 id="auth-story-title">让每一条判断<br />回到原文证据</h1>
          <p>直接导入故事正文也可以开始。世界观设定不是使用前提，系统会明确标出证据、判断边界和模型降级情况。</p>
        </div>
      </section>

      <section className="authPanel" aria-labelledby="auth-title">
        <div className="authPanelTop">
          <button className="textButton" type="button" onClick={() => browserNavigate(switchHref)}>
            {isRegister ? "已有账户？登录" : "还没有账户？创建账户"}
          </button>
        </div>
        <div className="authFormWrap">
          <h2 id="auth-title">{isRegister ? "创建账户" : "登录"}</h2>
          <p>{isRegister ? "建立你的个人故事工作区。" : "继续上次的故事审查工作。"}</p>

          {(visibleErrors.length > 1 || serverError) && (
            <div className="formErrorSummary" ref={summaryRef} tabIndex={-1} role="alert">
              <strong>{serverError || "请检查以下内容："}</strong>
              {!serverError && (
                <ul>
                  {visibleErrors.map(([name, message]) => (
                    <li key={name}><a href={`#${name}`}>{message}</a></li>
                  ))}
                </ul>
              )}
            </div>
          )}

          <form id="auth-form" noValidate onSubmit={submit}>
            {isRegister && (
              <div className="formField">
                <label htmlFor="displayName">显示名称</label>
                <input
                  id="displayName"
                  name="name"
                  autoComplete="name"
                  value={values.displayName}
                  onChange={(event) => update("displayName", event.target.value)}
                  onBlur={() => setTouched((current) => ({ ...current, displayName: true }))}
                  aria-invalid={Boolean(fieldError("displayName"))}
                  aria-describedby={fieldError("displayName") ? "displayName-error" : undefined}
                  disabled={pending}
                />
                {fieldError("displayName") && <small id="displayName-error" className="fieldError">{fieldError("displayName")}</small>}
              </div>
            )}
            <div className="formField">
              <label htmlFor="email">邮箱</label>
              <input
                id="email"
                name="email"
                type="email"
                inputMode="email"
                autoCapitalize="none"
                spellCheck={false}
                autoComplete="email"
                value={values.email}
                onChange={(event) => update("email", event.target.value)}
                onBlur={() => setTouched((current) => ({ ...current, email: true }))}
                aria-invalid={Boolean(fieldError("email"))}
                aria-describedby={fieldError("email") ? "email-error" : undefined}
                disabled={pending}
              />
              {fieldError("email") && <small id="email-error" className="fieldError">{fieldError("email")}</small>}
            </div>
            <div className="formField">
              <label htmlFor="password">密码</label>
              <div className="passwordField">
                <input
                  id="password"
                  name="password"
                  type={showPassword ? "text" : "password"}
                  autoComplete={isRegister ? "new-password" : "current-password"}
                  value={values.password}
                  onChange={(event) => update("password", event.target.value)}
                  onBlur={() => setTouched((current) => ({ ...current, password: true }))}
                  aria-invalid={Boolean(fieldError("password"))}
                  aria-describedby={[
                    isRegister ? "password-help" : "",
                    fieldError("password") ? "password-error" : "",
                  ].filter(Boolean).join(" ") || undefined}
                  disabled={pending}
                />
                <button
                  type="button"
                  className="passwordToggle"
                  aria-label={showPassword ? "隐藏密码" : "显示密码"}
                  aria-pressed={showPassword}
                  onClick={() => setShowPassword((value) => !value)}
                >
                  <EyeIcon crossed={showPassword} />
                </button>
              </div>
              {isRegister && <small id="password-help" className="fieldHelp">使用 10–128 个字符；支持粘贴和密码管理器。</small>}
              {fieldError("password") && <small id="password-error" className="fieldError">{fieldError("password")}</small>}
            </div>
            {isRegister && (
              <div className="formField">
                <label htmlFor="confirmPassword">确认密码</label>
                <div className="passwordField">
                  <input
                    id="confirmPassword"
                    name="password-confirmation"
                    type={showConfirmation ? "text" : "password"}
                    autoComplete="new-password"
                    value={values.confirmPassword}
                    onChange={(event) => update("confirmPassword", event.target.value)}
                    onBlur={() => setTouched((current) => ({ ...current, confirmPassword: true }))}
                    aria-invalid={Boolean(fieldError("confirmPassword"))}
                    aria-describedby={fieldError("confirmPassword") ? "confirmPassword-error" : undefined}
                    disabled={pending}
                  />
                  <button
                    type="button"
                    className="passwordToggle"
                    aria-label={showConfirmation ? "隐藏确认密码" : "显示确认密码"}
                    aria-pressed={showConfirmation}
                    onClick={() => setShowConfirmation((value) => !value)}
                  >
                    <EyeIcon crossed={showConfirmation} />
                  </button>
                </div>
                {fieldError("confirmPassword") && <small id="confirmPassword-error" className="fieldError">{fieldError("confirmPassword")}</small>}
              </div>
            )}
            <button className="quietPrimary authSubmit" type="submit" disabled={pending}>
              {pending ? (isRegister ? "正在创建…" : "正在登录…") : (isRegister ? "创建账户" : "登录")}
            </button>
          </form>
          {!isRegister && <p className="authFootnote">忘记密码功能暂未开放；部署管理员可以协助恢复访问。</p>}
        </div>
      </section>
    </main>
  );
}
