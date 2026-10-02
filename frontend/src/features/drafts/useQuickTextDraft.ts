import { useMemo, useRef, useState } from "react";
import type { DocumentRole, QuickTextMode } from "../../documentContext";
import {
  draftLease, readTabDraft, removeTabDraft, writeTabDraft,
  type DraftOwner, type DraftScope, type DraftWriteStatus, type QuickDraft,
} from "./sessionDraftStorage";

export function useQuickTextDraft(owner: DraftOwner, defaults: { world: string; chapter: string }, enabled: boolean) {
  const lease = useMemo(() => draftLease(owner), [owner.userId, owner.workspaceId]);
  // This editor creates a new independent project; it is deliberately not bound to a selected project's documents.
  const scope = useMemo<DraftScope>(() => ({ ...owner, projectId: "__quick_text__", kind: "quick" }), [owner.userId, owner.workspaceId]);
  const [initial] = useState(() => enabled ? readTabDraft<QuickDraft>(scope, lease) : { draft: null, status: "empty" as const });
  const [values, setValues] = useState(() => ({
    ...defaults, mode: "body" as QuickTextMode, role: "chapter" as DocumentRole, scope: "global",
    ...initial.draft,
  }));
  const valuesRef = useRef(values);
  const dirty = useRef<{ world?: string; chapter?: string }>({
    ...(initial.draft?.world !== undefined ? { world: initial.draft.world } : {}),
    ...(initial.draft?.chapter !== undefined ? { chapter: initial.draft.chapter } : {}),
  });
  const [restored, setRestored] = useState(initial.status === "ready");
  const [status, setStatus] = useState<DraftWriteStatus | "invalid">(
    initial.status === "ready" ? "saved" : initial.status === "invalid" ? "invalid" :
      initial.status === "unavailable" ? "unavailable" : "empty",
  );

  function update<K extends keyof typeof values>(field: K, value: typeof values[K]) {
    const next = { ...valuesRef.current, [field]: value };
    valuesRef.current = next;
    // After an import the visible values stay in the editor. A later draft may
    // intentionally reuse that world; derive its full context from current values.
    dirty.current = {
      ...(next.world !== defaults.world ? { world: next.world } : {}),
      ...(next.chapter !== defaults.chapter ? { chapter: next.chapter } : {}),
    };
    setValues(next);
    setRestored(false);
    // Write synchronously from the input event: a subsequent 401/refresh cannot race a debounce timer.
    const hasText = Object.keys(dirty.current).length > 0;
    setStatus(!enabled ? "empty" : hasText ? writeTabDraft(scope, lease, {
      ...dirty.current, mode: next.mode, role: next.role, scope: next.scope,
    }) : removeTabDraft(scope, lease));
  }

  function discard() {
    if (Object.keys(dirty.current).length && !window.confirm("丢弃本标签页暂存的未提交正文？这不会修改项目中已保存的文档。")) return;
    const result = enabled ? removeTabDraft(scope, lease) : "empty";
    if (result !== "empty") { setStatus(result); return; }
    const next = { ...defaults, mode: "body" as QuickTextMode, role: "chapter" as DocumentRole, scope: "global" };
    dirty.current = {};
    valuesRef.current = next;
    setValues(next);
    setRestored(false);
    setStatus("empty");
  }

  function markSubmitted(submitted: { world: string; chapter: string; mode: QuickTextMode; role: DocumentRole; scope: string }) {
    // A successful request only clears the version it submitted, not text typed while it was in flight.
    if (JSON.stringify(valuesRef.current) !== JSON.stringify(submitted)) return;
    const result = enabled ? removeTabDraft(scope, lease) : "empty";
    if (result === "empty") dirty.current = {};
    setRestored(false);
    setStatus(result);
  }

  return { ...values, status, restored, hasDraft: Object.keys(dirty.current).length > 0,
    update, discard, markSubmitted };
}
