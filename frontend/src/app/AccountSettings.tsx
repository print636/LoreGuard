import {
  FormEvent,
  type MouseEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ApiError, apiJson } from "../api/client";
import { browserNavigate } from "../routing";
import {
  formatSessionTime,
  passwordApiDetailMessage,
  passwordErrorFocusTarget,
  passwordChangeSuccessMessage,
  validatePasswordChange,
  type AccountSession,
  type AccountSessionsResponse,
  type PasswordFieldName,
  type PasswordValues,
} from "./accountSecurity";
import { apiErrorDetail, type SessionIdentity } from "./session";

type AccountSettingsProps = {
  identity: SessionIdentity;
  onLoggedOut: () => void;
};

const emptyPasswords: PasswordValues = {
  currentPassword: "",
  newPassword: "",
  confirmPassword: "",
};

function ProjectIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M4 5.5h6l2 2h8v11H4z" />
      <path d="M4 9h16" />
    </svg>
  );
}

function ShieldIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M12 3.5 19 6v5.3c0 4.2-2.8 7.7-7 9.2-4.2-1.5-7-5-7-9.2V6z" />
      <path d="m9.2 12 1.8 1.8 3.9-4.1" />
    </svg>
  );
}

function KeyIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <circle cx="8.5" cy="12" r="4.5" />
      <path d="M13 12h8M18 12v3M15.5 12v2" />
    </svg>
  );
}

function EyeIcon({ crossed }: { crossed: boolean }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <path d="M3 12s3.2-5 9-5 9 5 9 5-3.2 5-9 5-9-5-9-5Z" />
      <circle cx="12" cy="12" r="2.5" />
      {crossed && <path d="m4 4 16 16" />}
    </svg>
  );
}

function SessionIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3.5" y="5" width="17" height="12" rx="2" />
      <path d="M9 20h6M12 17v3" />
    </svg>
  );
}

function passwordApiMessage(error: unknown): string {
  const detail = apiErrorDetail(error);
  return passwordApiDetailMessage(
    error instanceof ApiError ? error.status : 0,
    detail,
  );
}

function followSpaLink(event: MouseEvent<HTMLAnchorElement>) {
  if (
    event.defaultPrevented ||
    event.button !== 0 ||
    event.metaKey ||
    event.ctrlKey ||
    event.shiftKey ||
    event.altKey
  ) {
    return;
  }
  event.preventDefault();
  browserNavigate(event.currentTarget.getAttribute("href") || "/app");
}

export default function AccountSettings({
  identity,
  onLoggedOut,
}: AccountSettingsProps) {
  const [passwords, setPasswords] = useState<PasswordValues>(emptyPasswords);
  const [touched, setTouched] = useState<Partial<Record<PasswordFieldName, boolean>>>({});
  const [submitted, setSubmitted] = useState(false);
  const [passwordPending, setPasswordPending] = useState(false);
  const [passwordError, setPasswordError] = useState("");
  const [passwordSuccess, setPasswordSuccess] = useState("");
  const [visiblePasswords, setVisiblePasswords] = useState<Partial<Record<PasswordFieldName, boolean>>>({});
  const [sessions, setSessions] = useState<AccountSession[]>([]);
  const [sessionsLoading, setSessionsLoading] = useState(true);
  const [sessionsError, setSessionsError] = useState("");
  const [sessionAction, setSessionAction] = useState<string | null>(null);
  const [confirmAction, setConfirmAction] = useState<"others" | string | null>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [logoutPending, setLogoutPending] = useState(false);
  const summaryRef = useRef<HTMLDivElement | null>(null);
  const mainRef = useRef<HTMLElement | null>(null);
  const passwordInputRefs = useRef<Record<PasswordFieldName, HTMLInputElement | null>>({
    currentPassword: null,
    newPassword: null,
    confirmPassword: null,
  });

  const errors = useMemo(() => validatePasswordChange(passwords), [passwords]);
  const visibleErrors = Object.entries(errors).filter(
    ([field]) => submitted || touched[field as PasswordFieldName],
  );

  async function loadSessions() {
    try {
      setSessionsLoading(true);
      setSessionsError("");
      const response = await apiJson<AccountSessionsResponse>("/api/v1/auth/sessions");
      setSessions(response.sessions);
    } catch (error) {
      setSessionsError(`${apiErrorDetail(error)} 会话列表没有更新，你可以重试。`);
    } finally {
      setSessionsLoading(false);
    }
  }

  useEffect(() => {
    mainRef.current?.focus({ preventScroll: true });
    void loadSessions();
  }, []);

  function updatePassword(field: PasswordFieldName, value: string) {
    setPasswords((current) => ({ ...current, [field]: value }));
    setPasswordError("");
    setPasswordSuccess("");
  }

  function fieldError(field: PasswordFieldName): string {
    return submitted || touched[field] ? errors[field] || "" : "";
  }

  function focusPasswordError(
    nextErrors: Partial<Record<PasswordFieldName, string>>,
    hasServerError = false,
  ) {
    const target = passwordErrorFocusTarget(nextErrors, hasServerError);
    requestAnimationFrame(() => {
      if (target === "summary") summaryRef.current?.focus();
      else if (target) passwordInputRefs.current[target]?.focus();
    });
  }

  async function changePassword(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitted(true);
    setPasswordError("");
    setPasswordSuccess("");
    if (Object.keys(errors).length) {
      focusPasswordError(errors);
      return;
    }
    try {
      setPasswordPending(true);
      const result = await apiJson<{ revoked_sessions: number }>("/api/v1/auth/password", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          current_password: passwords.currentPassword,
          new_password: passwords.newPassword,
        }),
      });
      setPasswords(emptyPasswords);
      setTouched({});
      setSubmitted(false);
      setVisiblePasswords({});
      setPasswordSuccess(passwordChangeSuccessMessage(result.revoked_sessions));
      await loadSessions();
    } catch (error) {
      setPasswordError(passwordApiMessage(error));
      focusPasswordError({}, true);
    } finally {
      setPasswordPending(false);
    }
  }

  async function revokeOtherSessions() {
    try {
      setSessionAction("others");
      setSessionsError("");
      setStatusMessage("");
      await apiJson("/api/v1/auth/sessions/revoke-others", { method: "POST" });
      setConfirmAction(null);
      setStatusMessage("其他会话已撤销。当前设备仍保持登录。");
      await loadSessions();
    } catch (error) {
      setSessionsError(`${apiErrorDetail(error)} 其他会话没有被撤销，请重试。`);
    } finally {
      setSessionAction(null);
    }
  }

  async function revokeSession(sessionId: string) {
    try {
      setSessionAction(sessionId);
      setSessionsError("");
      setStatusMessage("");
      await apiJson(`/api/v1/auth/sessions/${encodeURIComponent(sessionId)}`, {
        method: "DELETE",
      });
      setConfirmAction(null);
      setStatusMessage("会话已撤销。");
      await loadSessions();
    } catch (error) {
      setSessionsError(`${apiErrorDetail(error)} 该会话没有被撤销，请重试。`);
    } finally {
      setSessionAction(null);
    }
  }

  async function logout() {
    try {
      setLogoutPending(true);
      await apiJson("/api/v1/auth/logout", { method: "POST" });
      onLoggedOut();
      browserNavigate("/login", { replace: true });
    } catch (error) {
      setSessionsError(`${apiErrorDetail(error)} 退出没有完成，请重试。`);
      setLogoutPending(false);
    }
  }

  return (
    <div className="productPage projectCenterPage accountSettingsPage">
      <a className="skipLink" href="#account-settings-main">跳到账户安全设置</a>
      <aside className="productSidebar" aria-label="全局导航">
        <a className="brandLockup sidebarBrand" href="/app" onClick={followSpaLink}>
          <span className="productSeal" aria-hidden="true"><i>LG</i></span>
          <span><b>LoreGuard</b><small>故事审查工作区</small></span>
        </a>
        <nav aria-label="工作区">
          <a href="/app" onClick={followSpaLink}>
            <ProjectIcon /><span>项目</span>
          </a>
          <a className="active" href="/app/settings/account" aria-current="page" onClick={followSpaLink}>
            <ShieldIcon /><span>账户安全</span>
          </a>
          <a href="/app/settings/model" onClick={followSpaLink}>
            <KeyIcon /><span>模型与密钥</span>
          </a>
        </nav>
        <div className="sidebarAccount">
          <span className="accountAvatar" aria-hidden="true">{identity.user.display_name.slice(0, 1)}</span>
          <span><b>{identity.user.display_name}</b><small>{identity.user.email}</small></span>
          <div className="sidebarAccountActions">
            <button className="textButton" type="button" disabled={logoutPending} onClick={() => void logout()}>
              {logoutPending ? "退出中…" : "退出"}
            </button>
          </div>
        </div>
      </aside>

      <main id="account-settings-main" className="accountSettingsMain" tabIndex={-1} ref={mainRef}>
        <header className="accountSettingsHeader">
          <div>
            <h1>账户安全</h1>
            <p>更新登录密码，并检查仍可访问这个工作区的会话。</p>
          </div>
          <span className="securityState"><ShieldIcon />账户已受保护</span>
        </header>

        <div className="securityLedger">
          <section className="securitySection" aria-labelledby="password-section-title">
            <header>
              <span className="sectionGlyph"><ShieldIcon /></span>
              <div>
                <h2 id="password-section-title">修改密码</h2>
                <p>更新后，当前设备仍保持登录。需要时可单独撤销其他会话。</p>
              </div>
            </header>
            <div className="securitySectionBody">
              {(visibleErrors.length > 1 || passwordError) && (
                <div className="formErrorSummary" ref={summaryRef} tabIndex={-1} role="alert">
                  <strong>{passwordError || "请检查以下内容："}</strong>
                  {!passwordError && (
                    <ul>
                      {visibleErrors.map(([field, message]) => (
                        <li key={field}><a href={`#${field}`}>{message}</a></li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
              {passwordSuccess && <p className="securitySuccess" role="status">{passwordSuccess}</p>}
              <form className="passwordChangeForm" noValidate onSubmit={changePassword}>
                {([
                  ["currentPassword", "当前密码", "current-password"],
                  ["newPassword", "新密码", "new-password"],
                  ["confirmPassword", "确认新密码", "new-password"],
                ] as const).map(([field, label, autoComplete]) => {
                  const error = fieldError(field);
                  const helpId = field === "newPassword" ? `${field}-help` : "";
                  return (
                    <div className="formField" key={field}>
                      <label htmlFor={field}>{label}</label>
                      <div className="passwordField">
                        <input
                          ref={(element) => { passwordInputRefs.current[field] = element; }}
                          id={field}
                          name={field === "currentPassword" ? "current-password" : field === "newPassword" ? "new-password" : "new-password-confirmation"}
                          type={visiblePasswords[field] ? "text" : "password"}
                          autoComplete={autoComplete}
                          value={passwords[field]}
                          onChange={(event) => updatePassword(field, event.target.value)}
                          onBlur={() => setTouched((current) => ({ ...current, [field]: true }))}
                          aria-invalid={Boolean(error)}
                          aria-describedby={[helpId, error ? `${field}-error` : ""].filter(Boolean).join(" ") || undefined}
                          disabled={passwordPending}
                        />
                        <button
                          type="button"
                          className="passwordToggle"
                          aria-label={visiblePasswords[field] ? `隐藏${label}` : `显示${label}`}
                          aria-pressed={Boolean(visiblePasswords[field])}
                          onClick={() => setVisiblePasswords((current) => ({ ...current, [field]: !current[field] }))}
                          disabled={passwordPending}
                        >
                          <EyeIcon crossed={Boolean(visiblePasswords[field])} />
                        </button>
                      </div>
                      {field === "newPassword" && <small className="fieldHelp" id={helpId}>使用 10–128 个字符；支持粘贴和密码管理器。</small>}
                      {error && <small className="fieldError" id={`${field}-error`}>{error}</small>}
                    </div>
                  );
                })}
                <button className="quietPrimary passwordSaveAction" type="submit" disabled={passwordPending}>
                  {passwordPending ? "正在更新…" : "更新密码"}
                </button>
              </form>
            </div>
          </section>

          <section className="securitySection" aria-labelledby="sessions-section-title">
            <header>
              <span className="sectionGlyph"><SessionIcon /></span>
              <div>
                <h2 id="sessions-section-title">登录会话</h2>
                <p>这里只显示会话时间，不采集或伪造设备名称与 IP 地址。</p>
              </div>
            </header>
            <div className="securitySectionBody">
              <div className="sessionToolbar">
                <p>{sessionsLoading ? "正在读取会话…" : `共 ${sessions.length} 个有效会话`}</p>
                {sessions.filter((session) => !session.current).length > 0 && (
                  <button
                    className="dangerTextAction"
                    type="button"
                    disabled={Boolean(sessionAction)}
                    onClick={() => setConfirmAction("others")}
                  >
                    撤销其他会话
                  </button>
                )}
              </div>

              {confirmAction === "others" && (
                <div className="revokeConfirmation" role="group" aria-labelledby="revoke-others-title">
                  <div><strong id="revoke-others-title">撤销所有其他会话？</strong><p>其他设备需要重新登录，当前设备不受影响。</p></div>
                  <span>
                    <button autoFocus className="dangerAction" type="button" disabled={Boolean(sessionAction)} onClick={() => void revokeOtherSessions()}>{sessionAction === "others" ? "正在撤销…" : "确认撤销"}</button>
                    <button type="button" disabled={Boolean(sessionAction)} onClick={() => setConfirmAction(null)}>取消</button>
                  </span>
                </div>
              )}

              {sessionsError && (
                <div className="inlineError" role="alert">
                  <div><strong>会话操作没有完成</strong><p>{sessionsError}</p></div>
                  <button type="button" onClick={() => void loadSessions()}>重新加载</button>
                </div>
              )}
              {statusMessage && <p className="securitySuccess sessionSuccess" role="status">{statusMessage}</p>}

              {sessionsLoading ? (
                <div className="sessionSkeleton" aria-busy="true" aria-label="正在加载登录会话"><i /><i /></div>
              ) : !sessionsError && sessions.length === 0 ? (
                <div className="sessionsEmpty"><strong>没有可显示的有效会话</strong><p>重新加载页面；如果问题持续，请重新登录。</p><button type="button" onClick={() => void loadSessions()}>重新加载</button></div>
              ) : (
                <ul className="sessionList">
                  {sessions.map((session) => (
                    <li className={session.current ? "currentSession" : ""} key={session.id}>
                      <span className="sessionDeviceIcon"><SessionIcon /></span>
                      <div className="sessionIdentity">
                        <span><strong>{session.current ? "当前会话" : "其他登录会话"}</strong>{session.current && <em>正在使用</em>}</span>
                        <dl>
                          <div><dt>最近活动</dt><dd>{formatSessionTime(session.last_seen_at)}</dd></div>
                          <div><dt>登录时间</dt><dd>{formatSessionTime(session.created_at)}</dd></div>
                          <div><dt>到期时间</dt><dd>{formatSessionTime(session.expires_at)}</dd></div>
                        </dl>
                      </div>
                      {!session.current && confirmAction !== session.id && (
                        <button className="dangerTextAction" type="button" disabled={Boolean(sessionAction)} onClick={() => setConfirmAction(session.id)}>撤销</button>
                      )}
                      {!session.current && confirmAction === session.id && (
                        <div className="singleRevokeConfirmation" role="group" aria-label="确认撤销这个会话">
                          <span>撤销这个会话？</span>
                          <button autoFocus className="dangerAction" type="button" disabled={Boolean(sessionAction)} onClick={() => void revokeSession(session.id)}>{sessionAction === session.id ? "撤销中…" : "确认"}</button>
                          <button type="button" disabled={Boolean(sessionAction)} onClick={() => setConfirmAction(null)}>取消</button>
                        </div>
                      )}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>

          <section className="securitySection unavailableSecuritySection" aria-labelledby="recovery-title">
            <header>
              <span className="sectionGlyph"><ShieldIcon /></span>
              <div><h2 id="recovery-title">账户恢复</h2><p>邮箱验证与忘记密码暂未提供。这里不展示无法使用的操作按钮。</p></div>
            </header>
          </section>
        </div>
      </main>
    </div>
  );
}
