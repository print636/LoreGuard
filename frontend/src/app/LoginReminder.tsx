import { useLayoutEffect, useRef, type KeyboardEvent } from "react";
import { browserNavigate } from "../routing";
import { publicEntryActions, publicEntryAuthHref, publicSessionMessage, type PublicFeatureAction, type PublicSessionStatus } from "./publicEntry";

type Props = {
  action: PublicFeatureAction;
  trigger: HTMLButtonElement;
  status: PublicSessionStatus;
  onRetry: () => void;
  onClose: () => void;
};

export default function LoginReminder({ action, trigger, status, onRetry, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement | null>(null);
  const title = useRef<HTMLHeadingElement | null>(null);
  const entry = publicEntryActions[action];

  function closeReminder() {
    dialog.current?.close();
    onClose();
    if (trigger.isConnected) trigger.focus({ preventScroll: true });
  }

  function loopBoundaryFocus(event: KeyboardEvent<HTMLDialogElement>) {
    if (event.key !== "Tab" || event.altKey || event.ctrlKey || event.metaKey) return;
    const controls = Array.from(dialog.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") ?? []);
    const first = controls[0];
    const last = controls[controls.length - 1];
    if (!first || !last) return;
    if (event.shiftKey && (document.activeElement === first || document.activeElement === title.current)) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  useLayoutEffect(() => {
    const element = dialog.current;
    if (element && !element.open) element.showModal();
    title.current?.focus({ preventScroll: true });
    return () => { if (element?.open) element.close(); };
  }, []);

  return (
    <dialog className="loginReminder productPage" ref={dialog}
      aria-labelledby="login-reminder-title" aria-describedby="login-reminder-purpose login-reminder-status"
      onKeyDown={loopBoundaryFocus}
      onCancel={(event) => { event.preventDefault(); closeReminder(); }}
      onClick={(event) => {
        if (event.target !== event.currentTarget) return;
        const bounds = event.currentTarget.getBoundingClientRect();
        if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) closeReminder();
      }}>
      <div className="loginReminderTop">
        <h2 id="login-reminder-title" ref={title} tabIndex={-1}>登录后继续</h2>
        <button type="button" className="textButton" onClick={closeReminder}>关闭登录提醒</button>
      </div>
      <p id="login-reminder-purpose">{entry.reason}</p>
      <p id="login-reminder-status" className="loginReminderStatus" role="status">{publicSessionMessage(status)}</p>
      {status === "failed" && <button type="button" onClick={() => {
        onRetry();
        title.current?.focus({ preventScroll: true });
      }}>重新确认登录状态</button>}
      <p className="loginReminderNote">登录只会打开目标页面，不会自动创建项目、导入文稿或开始校验。</p>
      <div className="loginReminderActions">
        <button type="button" className="quietPrimary" disabled={status === "checking"}
          onClick={() => browserNavigate(publicEntryAuthHref("login", entry.returnTo))}>去登录</button>
        <button type="button" disabled={status === "checking"}
          onClick={() => browserNavigate(publicEntryAuthHref("register", entry.returnTo))}>创建账户</button>
        <button type="button" onClick={closeReminder}>继续浏览</button>
      </div>
    </dialog>
  );
}
