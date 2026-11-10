import { useEffect, useRef, useState } from "react";
import { apiJson } from "../../api/client";
import { materialImpactPath, normalizeMaterialImpact, type MaterialImpact } from "./materialImpactModel";

export function useMaterialImpact(projectId: string, runId: string | null, refreshKey: unknown, enabled = true) {
  const [offset, setOffset] = useState(0);
  const [issueOffset, setIssueOffset] = useState(0);
  const [attempt, setAttempt] = useState(0);
  const sequence = useRef(0);
  const [state, setState] = useState<{ key: string; refreshKey: unknown; value: MaterialImpact | null; error: string; loading: boolean } | null>(null);
  const key = JSON.stringify([projectId, runId, offset, issueOffset, attempt]);
  useEffect(() => { setOffset(0); setIssueOffset(0); }, [projectId, runId, refreshKey]);
  useEffect(() => {
    if (!enabled || !projectId || !runId) return;
    const current = ++sequence.current;
    const controller = new AbortController();
    let alive = true;
    const timer = setTimeout(() => controller.abort(), 15_000);
    setState({ key, refreshKey, value: null, error: "", loading: true });
    void apiJson<unknown>(materialImpactPath(runId, offset, issueOffset), { signal: controller.signal }).then(raw => {
      const value = normalizeMaterialImpact(raw, projectId, runId, offset, issueOffset);
      if (alive && sequence.current === current) setState({ key, refreshKey, value, error: "", loading: false });
    }).catch(() => {
      if (alive && sequence.current === current) setState({ key, refreshKey, value: null, error: "资料变化未能读完。旧报告仍可阅读，请重试核对后再准备复检。", loading: false });
    }).finally(() => clearTimeout(timer));
    return () => { alive = false; clearTimeout(timer); controller.abort(); };
  }, [key, projectId, runId, offset, issueOffset, refreshKey, enabled]);
  const visible = enabled && state?.key === key && state.refreshKey === refreshKey ? state : null;
  return { value: visible?.value || null, error: visible?.error || "", loading: enabled && !!runId && (!visible || visible.loading),
    offset, issueOffset, setOffset, setIssueOffset, refresh: () => setAttempt(n => n + 1) };
}
