export type CandidateDraftInput = {
  comment: string;
  axisChoice: string;
  axisName: string;
  axisDefinition: string;
  axisPositiveProposition: string;
  applicabilityScope: string;
  legacyAxisProposition: string;
  alignmentChoice: "same" | "opposite" | "uncertain";
  scopeConfirmed: boolean;
};

export function hasPendingCandidateInput(draft: CandidateDraftInput): boolean {
  return [draft.comment, draft.axisChoice, draft.axisName, draft.axisDefinition,
    draft.axisPositiveProposition, draft.applicabilityScope, draft.legacyAxisProposition]
    .some((value) => value.trim().length > 0) ||
    draft.alignmentChoice !== "uncertain" || draft.scopeConfirmed;
}

// A successful request only clears what it actually submitted, not later typing.
export function candidateInputAfterSubmission<T>(current: T, submitted: T, empty: T): T {
  return current === submitted ? empty : current;
}

export const candidateLeaveConfirmation = "候选审核还有未提交内容，仅在本页内存。切换或离开将丢失这些输入；已创建的作者轴仍保存在项目中，不会撤销。确定离开吗？";
export const candidateBusyLeaveConfirmation = "候选审核正在提交。离开会丢失本页未提交输入，服务器仍可能完成已发出的请求；已创建的作者轴不会撤销，返回后请读取核对。确定离开吗？";
