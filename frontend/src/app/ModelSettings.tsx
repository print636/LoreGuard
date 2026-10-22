import {
  FormEvent,
  type KeyboardEvent,
  type MouseEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { ApiError, apiJson } from "../api/client";
import { browserNavigate, registerBrowserNavigationBlocker } from "../routing";
import type { SessionIdentity } from "./session";
import TechnicalDetails from "../components/TechnicalDetails";
import {
  MODEL_TEST_DISCLAIMER,
  SAVED_API_KEY_MASK,
  describeAccountProviderTest,
  canTestModelProvider,
  emptyModelProviderForm,
  formFromModelProvider,
  modelProviderFormDirty,
  modelProviderDeleteSuccess,
  modelProviderDeleteWarning,
  modelProviderSavePayload,
  modelProviderSourceLabel,
  modelProviderTestIsStale,
  parseAccountModelProvider,
  providerConflictMessage,
  validateModelProviderForm,
  type AccountModelProviderProfile,
  type ModelProviderErrors,
  type ModelProviderField,
  type ModelProviderForm,
  type ModelProviderTestView,
} from "./modelProviderSettings";

type ModelSettingsProps = {
  identity: SessionIdentity;
  onLoggedOut: () => void;
};

function ProjectIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 5.5h6l2 2h8v11H4z" /><path d="M4 9h16" /></svg>;
}

function ShieldIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3.5 19 6v5.3c0 4.2-2.8 7.7-7 9.2-4.2-1.5-7-5-7-9.2V6z" /><path d="m9.2 12 1.8 1.8 3.9-4.1" /></svg>;
}

function KeyIcon() {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="8.5" cy="12" r="4.5" /><path d="M13 12h8M18 12v3M15.5 12v2" /></svg>;
}

function EyeIcon({ crossed }: { crossed: boolean }) {
  return <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 12s3.2-5 9-5 9 5 9 5-3.2 5-9 5-9-5-9-5Z" /><circle cx="12" cy="12" r="2.5" />{crossed && <path d="m4 4 16 16" />}</svg>;
}

function followSpaLink(event: MouseEvent<HTMLAnchorElement>) {
  if (
    event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey ||
    event.shiftKey || event.altKey
  ) return;
  event.preventDefault();
  browserNavigate(event.currentTarget.getAttribute("href") || "/app");
}

function safeOperationMessage(error: unknown, operation: "load" | "save" | "test" | "delete"): string {
  const conflict = error instanceof ApiError ? providerConflictMessage(error.status) : null;
  if (conflict) return conflict;
  if (error instanceof ApiError && error.status === 401) return "登录状态已失效，请重新登录后继续。";
  if (error instanceof ApiError && error.status === 403) return "当前账户不能执行这项操作。";
  if (error instanceof ApiError && error.status === 422) return "配置格式未通过安全检查，请核对 Base URL、模型名称和 API Key。";
  if (error instanceof ApiError && error.status === 429) return "操作过于频繁，请稍后再试。";
  if (operation === "load") return "模型配置暂时无法读取；页面没有修改已保存内容。请重新加载配置。";
  if (operation === "test") return "连接测试没有完成；已保存配置没有改变。请检查服务状态后重试。";
  if (operation === "delete") return "模型配置没有删除，请稍后重试。";
  return "模型配置没有保存，请稍后重试。";
}

function formatTimestamp(value: string | null): string {
  if (!value) return "尚未记录";
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return "尚未记录";
  return new Intl.DateTimeFormat("zh-CN", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function initialProfile(): AccountModelProviderProfile {
  return {
    configured: false,
    revision: 0,
    baseUrl: "",
    model: "",
    updatedAt: null,
    serviceDefaultAvailable: false,
    lastTest: null,
  };
}

export default function ModelSettings({ identity, onLoggedOut }: ModelSettingsProps) {
  const [profile, setProfile] = useState<AccountModelProviderProfile>(initialProfile);
  const [form, setForm] = useState<ModelProviderForm>(emptyModelProviderForm);
  const [loading, setLoading] = useState(true);
  const [profileLoaded, setProfileLoaded] = useState(false);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [logoutPending, setLogoutPending] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [touched, setTouched] = useState<Partial<Record<ModelProviderField, boolean>>>({});
  const [operationError, setOperationError] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [testResult, setTestResult] = useState<ModelProviderTestView | null>(null);
  const [keyVisible, setKeyVisible] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const mainRef = useRef<HTMLElement | null>(null);
  const summaryRef = useRef<HTMLDivElement | null>(null);
  const deleteTriggerRef = useRef<HTMLButtonElement | null>(null);
  const deleteConfirmRef = useRef<HTMLButtonElement | null>(null);
  const fieldRefs = useRef<Record<ModelProviderField, HTMLInputElement | null>>({
    baseUrl: null,
    model: null,
    apiKey: null,
  });

  const errors = useMemo(
    () => validateModelProviderForm(form, profile.configured, profile.localGateway),
    [form, profile.configured, profile.localGateway],
  );
  const dirty = useMemo(
    () => modelProviderFormDirty(form, profile),
    [form, profile],
  );
  const testAllowed = canTestModelProvider(
    profile,
    form,
    testing || saving || deleting,
  ) && profileLoaded;
  const visibleErrors = Object.entries(errors).filter(
    ([field]) => submitted || touched[field as ModelProviderField],
  );
  const activeTest = testResult || profile.lastTest;
  const testStale = modelProviderTestIsStale(activeTest, profile.revision);

  const loadProfile = useCallback(async (announce = false) => {
    try {
      setLoading(true);
      setProfileLoaded(false);
      setOperationError("");
      const next = parseAccountModelProvider(
        await apiJson("/api/v1/account/model-provider"),
      );
      setProfile(next);
      setProfileLoaded(true);
      setForm(formFromModelProvider(next));
      setSubmitted(false);
      setTouched({});
      setKeyVisible(false);
      setTestResult(null);
      if (announce) setSuccessMessage("已重新加载最新模型配置。");
    } catch (error) {
      setOperationError(safeOperationMessage(error, "load"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    mainRef.current?.focus({ preventScroll: true });
    void loadProfile();
  }, [loadProfile]);

  useEffect(() => {
    if (confirmDelete) deleteConfirmRef.current?.focus();
  }, [confirmDelete]);

  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => event.preventDefault();
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [dirty]);

  useEffect(() => {
    if (!dirty) return;
    return registerBrowserNavigationBlocker(() =>
      window.confirm("模型配置还有未保存更改，确定离开吗？"),
    );
  }, [dirty]);

  function followSettingsLink(event: MouseEvent<HTMLAnchorElement>) {
    followSpaLink(event);
  }

  function clearFeedback() {
    setOperationError("");
    setSuccessMessage("");
  }

  function updateField(field: ModelProviderField, value: string) {
    setForm((current) => ({ ...current, [field]: value }));
    clearFeedback();
  }

  function fieldError(field: ModelProviderField): string {
    return submitted || touched[field] ? errors[field] || "" : "";
  }

  function focusErrors(nextErrors: ModelProviderErrors) {
    const fields = Object.keys(nextErrors) as ModelProviderField[];
    requestAnimationFrame(() => {
      if (fields.length > 1) summaryRef.current?.focus();
      else if (fields[0]) fieldRefs.current[fields[0]]?.focus();
    });
  }

  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitted(true);
    clearFeedback();
    if (Object.keys(errors).length) {
      focusErrors(errors);
      return;
    }
    try {
      setSaving(true);
      const next = parseAccountModelProvider(
        await apiJson("/api/v1/account/model-provider", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(modelProviderSavePayload(form, profile)),
        }),
      );
      setProfile(next);
      setForm(formFromModelProvider(next));
      setSubmitted(false);
      setTouched({});
      setKeyVisible(false);
      setTestResult(null);
      setSuccessMessage("模型配置已保存。API Key 已从页面输入状态中清除。");
    } catch (error) {
      setOperationError(safeOperationMessage(error, "save"));
      requestAnimationFrame(() => summaryRef.current?.focus());
    } finally {
      setSaving(false);
    }
  }

  async function testConnection() {
    if (!profile.configured || dirty || testing) return;
    try {
      setTesting(true);
      clearFeedback();
      const payload = await apiJson("/api/v1/account/model-provider/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ expected_revision: profile.revision }),
      });
      const result = describeAccountProviderTest(payload);
      setTestResult(result);
      setSuccessMessage("模型连接检查已完成。结果仅说明能否正常回复，不代表剧情判断质量。");
    } catch (error) {
      setOperationError(safeOperationMessage(error, "test"));
    } finally {
      setTesting(false);
    }
  }

  function closeDeleteConfirmation() {
    setConfirmDelete(false);
    requestAnimationFrame(() => deleteTriggerRef.current?.focus());
  }

  function deleteConfirmationKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Escape" && !deleting) closeDeleteConfirmation();
  }

  async function deleteConfiguration() {
    try {
      setDeleting(true);
      clearFeedback();
      const next = parseAccountModelProvider(
        await apiJson("/api/v1/account/model-provider", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ expected_revision: profile.revision }),
        }),
      );
      setProfile(next);
      setForm(formFromModelProvider(next));
      setSubmitted(false);
      setTouched({});
      setTestResult(null);
      setKeyVisible(false);
      setConfirmDelete(false);
      setSuccessMessage(modelProviderDeleteSuccess(next));
    } catch (error) {
      setOperationError(safeOperationMessage(error, "delete"));
    } finally {
      setDeleting(false);
    }
  }

  async function logout() {
    if (dirty && !window.confirm("模型配置还有未保存更改，确定退出吗？")) return;
    try {
      setLogoutPending(true);
      await apiJson("/api/v1/auth/logout", { method: "POST" });
      onLoggedOut();
      browserNavigate("/login", { replace: true, bypassBlockers: true });
    } catch {
      setOperationError("退出没有完成，请重试。模型配置没有改变。");
      setLogoutPending(false);
    }
  }

  const statusLabel = loading
    ? "正在读取配置"
    : profile.configured
      ? dirty ? "有未保存更改" : "账户密钥已保存"
      : "尚未设置账户密钥";
  const statusTone = loading ? "neutral" : profile.configured ? dirty ? "warning" : "ready" : "warning";

  return (
    <div className="productPage projectCenterPage modelSettingsPage">
      <a className="skipLink" href="#model-settings-main">跳到模型与密钥设置</a>
      <aside className="productSidebar" aria-label="全局导航">
        <a className="brandLockup sidebarBrand" href="/app" onClick={followSettingsLink}>
          <span className="productSeal" aria-hidden="true"><i>LG</i></span>
          <span><b>LoreGuard</b><small>故事审查工作区</small></span>
        </a>
        <nav aria-label="工作区">
          <a href="/app" onClick={followSettingsLink}><ProjectIcon /><span>项目</span></a>
          <a className="active" href="/app/settings/model" aria-current="page" onClick={followSettingsLink}>
            <KeyIcon /><span>模型与密钥</span>
          </a>
          {identity.mode === "required" && <a href="/app/settings/account" onClick={followSettingsLink}><ShieldIcon /><span>账户安全</span></a>}
        </nav>
        <div className="sidebarAccount">
          <span className="accountAvatar" aria-hidden="true">{identity.user.display_name.slice(0, 1)}</span>
          <span><b>{identity.user.display_name}</b><small>{identity.mode === "anonymous" ? "本地体验账户" : identity.user.email}</small></span>
          {identity.mode === "required" && <div className="sidebarAccountActions">
            <button className="textButton" type="button" disabled={logoutPending} onClick={() => void logout()}>{logoutPending ? "退出中…" : "退出"}</button>
          </div>}
        </div>
      </aside>

      <main id="model-settings-main" className="accountSettingsMain modelSettingsMain" tabIndex={-1} ref={mainRef}>
        <header className="accountSettingsHeader">
          <div><h1>模型与密钥</h1><p>填入模型服务商提供的地址、模型名称和密钥，用于你发起的故事审查。接口需要兼容 OpenAI 格式。</p></div>
          <span className={`modelCredentialState ${statusTone}`}><KeyIcon />{statusLabel}</span>
        </header>

        {operationError && (
          <div className="formErrorSummary modelOperationError" ref={summaryRef} tabIndex={-1} role="alert">
            <strong>操作没有完成</strong><p>{operationError}</p>
            {operationError.includes("重新加载") && <button type="button" onClick={() => void loadProfile(true)}>重新加载配置</button>}
          </div>
        )}
        {successMessage && <p className="securitySuccess modelStatusMessage" role="status">{successMessage}</p>}

        <div className="securityLedger modelSettingsLedger" aria-busy={loading}>
          <section className="securitySection modelOverviewSection" aria-labelledby="model-overview-title">
            <header><span className="sectionGlyph"><KeyIcon /></span><div><h2 id="model-overview-title">当前配置</h2><p>你可以查看服务地址和模型名称，但不能再次读取已保存的密钥。</p></div></header>
            <div className="securitySectionBody">
              {loading ? (
                <div className="modelSettingsSkeleton" aria-label="正在加载模型配置"><i /><i /><i /></div>
              ) : (
                <>
                  <dl className="modelConfigSummary">
                    <div><dt>下次审查使用</dt><dd>{modelProviderSourceLabel(profile)}</dd></div>
                    <div><dt>API Key</dt><dd>{profile.configured ? <code aria-label="API Key 已保存">{SAVED_API_KEY_MASK}</code> : "未设置"}</dd></div>
                    <div><dt>Base URL</dt><dd>{profile.baseUrl || "未设置"}</dd></div>
                    <div><dt>模型</dt><dd>{profile.model || "未设置"}</dd></div>
                    <div><dt>更新时间</dt><dd>{formatTimestamp(profile.updatedAt)}</dd></div>
                  </dl>
                  <TechnicalDetails><p>配置版本：<code>{profile.revision || "尚未创建"}</code></p></TechnicalDetails>
                  <p className="secretBoundary"><strong>LoreGuard 不会再次显示已保存 API Key。</strong><span>固定掩码只表示账户中存在密钥，不包含真实前缀或后缀。</span></p>
                </>
              )}
            </div>
          </section>

          <section className="securitySection" aria-labelledby="model-config-title">
            <header><span className="sectionGlyph"><KeyIcon /></span><div><h2 id="model-config-title">配置账户模型</h2><p>修改地址或模型时可以保留原密钥；只有主动替换时才重新输入。</p></div></header>
            <div className="securitySectionBody">
              {visibleErrors.length > 1 && (
                <div className="formErrorSummary" ref={summaryRef} tabIndex={-1} role="alert">
                  <strong>请检查以下内容：</strong><ul>{visibleErrors.map(([field, message]) => <li key={field}><a href={`#model-${field}`}>{message}</a></li>)}</ul>
                </div>
              )}
              <form className="modelProviderForm" noValidate onSubmit={save}>
                <div className="formField">
                  <label htmlFor="model-baseUrl">Base URL</label>
                  <input ref={(node) => { fieldRefs.current.baseUrl = node; }} id="model-baseUrl" name="model-base-url" type="url" inputMode="url" autoComplete="off" spellCheck={false} value={form.baseUrl} disabled={!profileLoaded || loading || saving || deleting} aria-invalid={Boolean(fieldError("baseUrl"))} aria-describedby={`model-baseUrl-help${fieldError("baseUrl") ? " model-baseUrl-error" : ""}`} onBlur={() => setTouched((current) => ({ ...current, baseUrl: true }))} onChange={(event) => updateField("baseUrl", event.target.value)} />
                  <small className="fieldHelp" id="model-baseUrl-help">{profile.localGateway?.enabled
                    ? `普通服务必须使用 HTTPS；本机开发例外仅限 ${profile.localGateway.allowedOrigins.join("、")}，可在地址后加 /v1。不能包含账户信息或查询参数。`
                    : "例如 https://api.example.com/v1；必须使用 HTTPS，不能包含账户信息或查询参数。"}</small>
                  {fieldError("baseUrl") && <small className="fieldError" id="model-baseUrl-error">{fieldError("baseUrl")}</small>}
                  {profile.localGateway?.enabled && <p className="fieldHelp">本机网关的固定 API Key 与 ChatGPT 浏览器授权相互独立。连接测试最多等待 60 秒；通过不代表完整审查质量、额度或并发保证。仅供本人本机使用，不要通过公网分享此网关。</p>}
                </div>
                <div className="formField">
                  <label htmlFor="model-model">模型名称</label>
                  <input ref={(node) => { fieldRefs.current.model = node; }} id="model-model" name="model-name" autoComplete="off" spellCheck={false} maxLength={255} value={form.model} disabled={!profileLoaded || loading || saving || deleting} aria-invalid={Boolean(fieldError("model"))} aria-describedby={fieldError("model") ? "model-model-error" : undefined} onBlur={() => setTouched((current) => ({ ...current, model: true }))} onChange={(event) => updateField("model", event.target.value)} />
                  {fieldError("model") && <small className="fieldError" id="model-model-error">{fieldError("model")}</small>}
                </div>

                {profile.configured && !form.replacingKey ? (
                  <div className="savedKeyControl">
                    <div><span>API Key</span><strong>{SAVED_API_KEY_MASK}</strong><small>已保存；页面无法读取原值。</small></div>
                    <button type="button" disabled={saving || deleting} onClick={() => { setForm((current) => ({ ...current, replacingKey: true, apiKey: "" })); clearFeedback(); requestAnimationFrame(() => fieldRefs.current.apiKey?.focus()); }}>替换 API Key</button>
                  </div>
                ) : (
                  <div className="formField">
                    <label htmlFor="model-apiKey">{profile.configured ? "新的 API Key" : "API Key"}</label>
                    <div className="passwordField">
                      <input ref={(node) => { fieldRefs.current.apiKey = node; }} id="model-apiKey" name="provider-api-key" type={keyVisible ? "text" : "password"} autoComplete="off" spellCheck={false} maxLength={4096} value={form.apiKey} disabled={!profileLoaded || loading || saving || deleting} aria-invalid={Boolean(fieldError("apiKey"))} aria-describedby={`model-apiKey-help${fieldError("apiKey") ? " model-apiKey-error" : ""}`} onBlur={() => setTouched((current) => ({ ...current, apiKey: true }))} onChange={(event) => updateField("apiKey", event.target.value)} />
                      <button className="passwordToggle" type="button" aria-label={keyVisible ? "隐藏 API Key" : "显示 API Key"} aria-pressed={keyVisible} disabled={saving || deleting} onClick={() => setKeyVisible((current) => !current)}><EyeIcon crossed={keyVisible} /></button>
                    </div>
                    <small className="fieldHelp" id="model-apiKey-help">允许粘贴。密钥只会随本次保存请求发送，不会写入 URL 或浏览器存储。</small>
                    {fieldError("apiKey") && <small className="fieldError" id="model-apiKey-error">{fieldError("apiKey")}</small>}
                    {profile.configured && <button className="textButton cancelKeyReplacement" type="button" disabled={saving} onClick={() => { setForm((current) => ({ ...current, replacingKey: false, apiKey: "" })); setKeyVisible(false); setTouched((current) => ({ ...current, apiKey: false })); }}>取消替换</button>}
                  </div>
                )}

                <div className="modelFormActions">
                  <button className="quietPrimary" type="submit" disabled={!profileLoaded || loading || saving || deleting || !dirty}>{saving ? "正在保存…" : profile.configured && form.replacingKey ? "替换密钥并保存" : "保存模型配置"}</button>
                  {dirty && <span>有未保存更改；连接测试暂不可用。</span>}
                </div>
              </form>
            </div>
          </section>

          <section className="securitySection" aria-labelledby="model-test-title">
            <header><span className="sectionGlyph"><ShieldIcon /></span><div><h2 id="model-test-title">检查模型能否正常回复</h2><p>使用已保存的配置发送一条简短测试，可能产生少量费用。不会发送你的故事。</p></div></header>
            <div className="securitySectionBody modelTestBody">
              <div className="modelTestToolbar"><button type="button" disabled={!testAllowed} onClick={() => void testConnection()}>{testing ? "正在测试…" : "测试已保存的模型连接"}</button><span>{!profile.configured ? "请先保存账户模型配置。" : dirty ? "请先保存或撤销当前更改。" : "测试不会读取未保存输入。"}</span></div>
              <p className="modelTestBoundary"><strong>测试边界</strong>{MODEL_TEST_DISCLAIMER}</p>
              {activeTest ? (
                <div className={`modelTestResult ${testStale ? "warning" : activeTest.tone}`} role="status">
                  <div><strong>{testStale ? "测试结果已过期" : activeTest.label}</strong><span>{testStale ? "配置版本已改变，请重新测试。" : activeTest.detail}</span></div>
                  <dl><div><dt>检查时间</dt><dd>{formatTimestamp(activeTest.testedAt)}</dd></div><div><dt>回复格式检查</dt><dd>{activeTest.jsonContractOk === true ? "通过" : activeTest.jsonContractOk === false ? "未通过" : "未提供"}</dd></div><div><dt>响应时间</dt><dd>{activeTest.latencyMs === null ? "未提供" : `${(activeTest.latencyMs / 1000).toLocaleString("zh-CN", { maximumFractionDigits: 2 })} 秒`}</dd></div><div><dt>模型用量</dt><dd>{activeTest.tokenTotal === null ? "未提供" : `${activeTest.tokenTotal} Token`}</dd></div></dl>
                  <p><strong>下一步：</strong>{activeTest.action}</p>
                </div>
              ) : <div className="modelTestEmpty"><strong>尚未检查模型连接</strong><p>保存配置后点击上方按钮检查；保存成功不代表模型已经可用。</p></div>}
            </div>
          </section>

          {profile.configured && (
            <section className="securitySection modelDangerSection" aria-labelledby="delete-model-title">
              <header><span className="sectionGlyph"><KeyIcon /></span><div><h2 id="delete-model-title">删除账户模型配置</h2><p>删除 Base URL、模型名称和已保存密钥，不影响故事项目与历史报告。</p></div></header>
              <div className="securitySectionBody">
                {!confirmDelete ? <button ref={deleteTriggerRef} className="dangerTextAction" type="button" disabled={saving || testing} onClick={() => { clearFeedback(); setConfirmDelete(true); }}>删除模型配置</button> : (
                  <div className="modelDeleteConfirmation" role="group" aria-labelledby="delete-confirm-title" onKeyDown={deleteConfirmationKeyDown}>
                    <div><strong id="delete-confirm-title">确认删除账户模型配置？</strong><p>{modelProviderDeleteWarning(profile)}</p></div>
                    <span><button ref={deleteConfirmRef} className="dangerAction" type="button" disabled={deleting} onClick={() => void deleteConfiguration()}>{deleting ? "正在删除…" : "确认删除"}</button><button type="button" disabled={deleting} onClick={closeDeleteConfirmation}>取消</button></span>
                  </div>
                )}
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}
