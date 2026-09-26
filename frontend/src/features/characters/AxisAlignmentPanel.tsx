import { type FormEvent, useEffect, useRef, useState } from "react";
import { ApiError } from "../../api/client";
import {
  fetchCharacterTraitAxis,
  fetchProfileCandidate,
  setAxisPositiveProposition,
  submitAxisAlignment,
} from "./api";
import { previewAxisPolarity, validateAxisPositiveProposition } from "./axisReview";
import EvidenceList from "./EvidenceList";
import type { CharacterProfileItem, CharacterTraitAxis, ProfileCandidate } from "./types";

type Props = {
  projectId: string;
  characterId: string;
  item: CharacterProfileItem;
  onAligned: () => void;
  onClose: () => void;
};

function errorMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.status === 409) return "资料已变化。原选择仍保留；请核对当前版本后再提交。";
    if (error.status === 403) return "安全校验未通过，请刷新页面后重试。";
    if (error.status === 404) return "作者轴或角色特征已不可访问，请刷新档案。";
    if (error.status === 422) return "填写内容或方向未通过校验，请核对后重试。";
  }
  return "请求结果暂不能确认。已尝试重新读取资料，请核对状态后再操作。";
}

export default function AxisAlignmentPanel({
  projectId, characterId, item, onAligned, onClose,
}: Props) {
  const [candidate, setCandidate] = useState<ProfileCandidate | null>(null);
  const [axis, setAxis] = useState<CharacterTraitAxis | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [proposition, setProposition] = useState("");
  const [propositionTouched, setPropositionTouched] = useState(false);
  const [alignment, setAlignment] = useState<"same" | "opposite" | "uncertain">("uncertain");
  const [alignmentError, setAlignmentError] = useState("");
  const [comment, setComment] = useState("");
  const headingRef = useRef<HTMLHeadingElement | null>(null);
  const propositionRef = useRef<HTMLTextAreaElement | null>(null);
  const alignmentRef = useRef<HTMLFieldSetElement | null>(null);
  const isCurrentRef = useRef(true);
  const scopeKey = `${projectId}:${characterId}:${item.id}:${item.approved_axis_id || ""}`;
  const scopeKeyRef = useRef(scopeKey);
  scopeKeyRef.current = scopeKey;

  useEffect(() => {
    isCurrentRef.current = true;
    requestAnimationFrame(() => headingRef.current?.focus());
    const controller = new AbortController();
    setLoading(true);
    void Promise.all([
      fetchProfileCandidate(projectId, characterId, item.id, controller.signal),
      fetchCharacterTraitAxis(projectId, item.approved_axis_id || "", controller.signal),
    ]).then(([freshCandidate, freshAxis]) => {
      if (controller.signal.aborted) return;
      if (
        freshCandidate.id !== item.id || freshCandidate.status !== "confirmed" ||
        freshCandidate.approved_axis_id !== freshAxis.id ||
        freshCandidate.approved_axis_version !== freshAxis.version
      ) throw new TypeError("角色特征和作者轴版本不一致");
      setCandidate(freshCandidate);
      setAxis(freshAxis);
      setError("");
    }).catch((reason) => {
      if (!controller.signal.aborted) setError(errorMessage(reason));
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false);
    });
    return () => { isCurrentRef.current = false; controller.abort(); };
  }, [projectId, characterId, item.id, item.approved_axis_id]);

  async function refresh(): Promise<ProfileCandidate> {
    const startedScope = scopeKeyRef.current;
    const [freshCandidate, freshAxis] = await Promise.all([
      fetchProfileCandidate(projectId, characterId, item.id),
      fetchCharacterTraitAxis(projectId, item.approved_axis_id || ""),
    ]);
    if (!isCurrentRef.current || scopeKeyRef.current !== startedScope) return freshCandidate;
    if (freshCandidate.approved_axis_id !== freshAxis.id || freshCandidate.status !== "confirmed") {
      throw new TypeError("这条正式特征或作者轴已经变化");
    }
    setCandidate(freshCandidate);
    setAxis(freshAxis);
    return freshCandidate;
  }

  async function saveProposition(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!axis || axis.positive_proposition || busy) return;
    setPropositionTouched(true);
    const validated = validateAxisPositiveProposition(proposition);
    if (validated.error) { propositionRef.current?.focus(); return; }
    setBusy(true);
    setError("");
    const startedScope = scopeKeyRef.current;
    try {
      const updated = await setAxisPositiveProposition(projectId, axis.id, {
        positive_proposition: validated.value,
        expected_axis_version: axis.version,
      });
      if (!isCurrentRef.current || scopeKeyRef.current !== startedScope) return;
      setAxis(updated);
      setProposition("");
      setPropositionTouched(false);
    } catch (reason) {
      if (!isCurrentRef.current || scopeKeyRef.current !== startedScope) return;
      setError(errorMessage(reason));
      try {
        const freshAxis = await fetchCharacterTraitAxis(projectId, axis.id);
        if (scopeKeyRef.current === startedScope) {
          setAxis(freshAxis);
          if (freshAxis.positive_proposition === validated.value) setError("");
        }
      } catch { /* Keep the uncertainty visible. */ }
    } finally {
      if (isCurrentRef.current && scopeKeyRef.current === startedScope) setBusy(false);
    }
  }

  async function saveAlignment(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!candidate || !axis || !axis.positive_proposition_sha256 || busy) return;
    if (alignment === "uncertain") {
      setAlignmentError("暂不确定时请保留待补认，不会自动套用模型方向。");
      alignmentRef.current?.focus();
      return;
    }
    setBusy(true);
    setError("");
    setAlignmentError("");
    const startedScope = scopeKeyRef.current;
    try {
      await submitAxisAlignment(projectId, characterId, candidate.id, {
        expected_revision: candidate.revision,
        expected_axis_version: axis.version,
        expected_axis_positive_proposition_sha256: axis.positive_proposition_sha256,
        axis_alignment: alignment,
        comment: comment.trim(),
      });
      if (isCurrentRef.current && scopeKeyRef.current === startedScope) onAligned();
    } catch (reason) {
      if (!isCurrentRef.current || scopeKeyRef.current !== startedScope) return;
      setError(errorMessage(reason));
      try {
        const latest = await refresh();
        if (isCurrentRef.current && scopeKeyRef.current === startedScope && latest.axis_alignment === alignment) onAligned();
      } catch { /* Keep the uncertainty visible. */ }
    } finally {
      if (isCurrentRef.current && scopeKeyRef.current === startedScope) setBusy(false);
    }
  }

  const propositionValidation = validateAxisPositiveProposition(proposition);
  const canAlign = candidate?.status === "confirmed" &&
    !candidate.axis_alignment &&
    candidate.polarity !== "neutral" && candidate.polarity !== "unclear" &&
    (candidate.polarity === "positive" || candidate.polarity === "negative") &&
    candidate.source_verified && candidate.support_bindings_status !== "invalid" &&
    Boolean(axis?.positive_proposition_sha256);

  return (
    <section className="characterAxisAlignmentPanel" aria-label="补认作者轴方向" aria-busy={loading || busy}>
      <h4 ref={headingRef} tabIndex={-1}>补认作者轴方向</h4>
      <p>只影响之后新建的分析；原确认和历史报告不改写。原文证据与角色特征会单独核对。</p>
      {loading && <p role="status">正在读取冻结证据与作者轴…</p>}
      {error && <p className="candidateFieldError" role="alert">{error}</p>}
      {!loading && !candidate && <button type="button" onClick={() => {
        setLoading(true);
        setError("");
        void refresh()
          .catch((reason) => { if (isCurrentRef.current) setError(errorMessage(reason)); })
          .finally(() => { if (isCurrentRef.current) setLoading(false); });
      }}>重新读取</button>}
      {candidate && axis && (
        <>
          <div className="candidateAxisDefinition">
            <b>{axis.display_name} · v{axis.version}</b>
            <p>{axis.definition}</p>
            <p><strong>正向命题：</strong>{axis.positive_proposition || "待作者补写"}</p>
          </div>
          <p>原候选：{candidate.statement}；模型原标签：{candidate.model_trait_key || "未提供"}；原方向：{candidate.polarity === "positive" ? "正向" : candidate.polarity === "negative" ? "反向" : "不明确"}。</p>
          <EvidenceList
            title="原确认所依据的冻结证据"
            items={candidate.supporting_evidence}
            emptyText="没有可核对的冻结证据，不能补认方向。"
            candidateEvidence
            supportBindingsStatus={candidate.support_bindings_status}
            supportBindings={candidate.support_bindings_v1?.bindings || []}
          />
          {candidate.axis_alignment ? (
            <p role="status">这条特征已补认：{candidate.axis_alignment === "same" ? "同向" : "反向"}。刷新档案即可查看最新状态。</p>
          ) : !axis.positive_proposition ? (
            <form className="candidateAxisCreate" onSubmit={(event) => void saveProposition(event)} noValidate>
              <label htmlFor={`profile-axis-proposition-${item.id}`}>为旧作者轴补写正向命题</label>
              <textarea
                ref={propositionRef}
                id={`profile-axis-proposition-${item.id}`}
                maxLength={200}
                value={proposition}
                disabled={busy}
                aria-invalid={propositionTouched && Boolean(propositionValidation.error)}
                aria-describedby={propositionTouched && propositionValidation.error ? `profile-axis-proposition-error-${item.id}` : undefined}
                onChange={(event) => setProposition(event.target.value)}
                onBlur={() => setPropositionTouched(true)}
              />
              {propositionTouched && propositionValidation.error && <p className="candidateFieldError" id={`profile-axis-proposition-error-${item.id}`} role="alert">{propositionValidation.error}</p>}
              <p className="candidateAxisHint">命题只定义“何为正向”，不代表人物已这样做；保存后不能在原轴上改写。</p>
              <button type="submit" disabled={busy}>{busy ? "正在保存…" : "先保存正向命题"}</button>
            </form>
          ) : !canAlign ? (
            <p className="candidateFieldError" role="status">原方向或冻结证据不足，不能补认成确定方向；请保留待补认并检查资料。</p>
          ) : (
            <form className="candidateAxisCreate" onSubmit={(event) => void saveAlignment(event)} noValidate>
              <fieldset ref={alignmentRef} tabIndex={-1} className="candidateAxisAlignment" aria-describedby={alignmentError ? `profile-axis-alignment-error-${item.id}` : undefined}>
                <legend>原标签的正向含义与作者轴正向命题</legend>
                <label><input type="radio" name={`profile-axis-alignment-${item.id}`} checked={alignment === "same"} onChange={() => { setAlignment("same"); setAlignmentError(""); }} />同向</label>
                <label><input type="radio" name={`profile-axis-alignment-${item.id}`} checked={alignment === "opposite"} onChange={() => { setAlignment("opposite"); setAlignmentError(""); }} />反向</label>
                <label><input type="radio" name={`profile-axis-alignment-${item.id}`} checked={alignment === "uncertain"} onChange={() => { setAlignment("uncertain"); setAlignmentError(""); }} />暂不确定，保留待补认</label>
                {previewAxisPolarity(candidate.polarity, alignment) && (
                  <p role="status">方向预览：当前特征相对轴正向命题表示“{previewAxisPolarity(candidate.polarity, alignment) === "positive" ? "命题成立" : "命题不成立"}”。这仅根据你的方向选择换算，不替代证据审核。</p>
                )}
                {alignmentError && <p className="candidateFieldError" id={`profile-axis-alignment-error-${item.id}`} role="alert">{alignmentError}</p>}
              </fieldset>
              <label htmlFor={`profile-axis-comment-${item.id}`}>审核备注（可选）</label>
              <textarea id={`profile-axis-comment-${item.id}`} maxLength={2000} value={comment} disabled={busy} onChange={(event) => setComment(event.target.value)} />
              <button type="submit" disabled={busy}>{busy ? "正在补认…" : "确认方向映射"}</button>
            </form>
          )}
        </>
      )}
      <button type="button" onClick={onClose} disabled={busy}>关闭补认</button>
    </section>
  );
}
