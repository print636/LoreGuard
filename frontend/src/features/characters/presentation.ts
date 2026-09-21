import type {
  CandidateDecision,
  CharacterDimension,
  CharacterReadiness,
  ModelCoverage,
  ProfileCandidate,
} from "./types.ts";

export const characterDimensionNames: Record<CharacterDimension, string> = {
  core_personality: "核心人格",
  preference: "稳定偏好",
  value: "价值观",
  speech_pattern: "语言表现",
  behavior_boundary: "行为边界",
  contextual_behavior: "情境表现",
  current_state: "当前状态",
  unknown: "未分类角色特征",
};

export const profileOriginNames = {
  explicit_profile: "明确角色设定",
  confirmed_inference: "已确认的历史归纳",
} as const;

export const candidateOriginNames = {
  explicit_setting: "明确设定归纳",
  history_inference: "历史剧情归纳",
  unknown: "归纳来源未标注",
} as const;

export const candidateStatusNames = {
  pending: "待确认",
  confirmed: "已确认",
  rejected: "已驳回",
  stale: "来源已变化",
} as const;

export const feedbackStatusNames = {
  unreviewed: "未反馈",
  accepted: "已接受",
  false_positive: "误报",
  resolved: "已处理",
} as const;

export function describeCoverage(coverage: ModelCoverage, detail?: string | null) {
  if (coverage === "full") {
    return {
      tone: "full",
      label: "AI 归纳覆盖完整",
      detail: detail || "本次角色归纳的计划分块均完成模型处理。",
    };
  }
  if (coverage === "partial") {
    return {
      tone: "partial",
      label: "AI 归纳仅覆盖部分内容",
      detail:
        detail ||
        "部分文档或分块没有完成模型处理；现有候选仍需逐条核对，也可能存在遗漏。",
    };
  }
  if (coverage === "rules_only") {
    return {
      tone: "rulesOnly",
      label: "本次没有完成 AI 角色归纳",
      detail:
        detail ||
        "当前只有确定性记录，不能据此宣称角色档案完整，也不能确认 AI 归纳候选。",
    };
  }
  return {
    tone: "unknown",
    label: "角色归纳覆盖范围未知",
    detail: detail || "运行没有提供足够的覆盖信息，请先核对运行诊断。",
  };
}

export function describeReadiness(readiness: CharacterReadiness) {
  if (readiness === "no_documents") {
    return {
      title: "还没有可归纳的资料",
      detail: "先导入角色设定或已经发布的剧情，再开始一次校验。",
      action: "前往导入文稿",
      destination: "projects" as const,
    };
  }
  if (readiness === "no_completed_run") {
    return {
      title: "还没有完成的校验运行",
      detail: "角色档案候选来自冻结的分析输入；请先完成一次校验。",
      action: "前往文稿校验",
      destination: "check" as const,
    };
  }
  if (readiness === "not_generated") {
    return {
      title: "本次运行没有生成角色档案",
      detail: "这不代表资料中没有角色。请检查模型覆盖和文档类型后重新校验。",
      action: "查看运行审计",
      destination: "audit" as const,
    };
  }
  return null;
}

export function candidateReviewState(candidate: ProfileCandidate) {
  if (candidate.dimension === "unknown") {
    return {
      allowed: false,
      label: "服务端返回了未识别的角色特征类型。为避免改变语义，当前不能确认。",
    };
  }
  if (candidate.status === "stale") {
    return {
      allowed: false,
      label: "来源资料已经变化，不能确认这条旧归纳。请重新校验。",
    };
  }
  if (candidate.origin === "unknown") {
    return {
      allowed: false,
      label: "候选没有提供可核对的归纳来源，当前不能确认。",
    };
  }
  if (candidate.status !== "pending") {
    return {
      allowed: false,
      label: `这条归纳${candidate.status === "confirmed" ? "已经确认" : "已经驳回"}。`,
    };
  }
  if (candidate.model_coverage === "rules_only") {
    return {
      allowed: false,
      label: "本次没有完成 AI 角色归纳，不能确认该候选。",
    };
  }
  if (!candidate.reviewable) {
    return {
      allowed: false,
      label: candidate.unreviewable_reason || "这条归纳目前不能确认。",
    };
  }
  return { allowed: true, label: "核对证据后确认或驳回这条归纳。" };
}

export function candidateDecisionLabel(
  action: CandidateDecision,
  busyAction: CandidateDecision | null,
): string {
  if (busyAction === action) {
    return action === "confirm" ? "正在确认…" : "正在驳回…";
  }
  return action === "confirm" ? "确认归纳" : "驳回归纳";
}

export function authorityName(value: string): string {
  const names: Record<string, string> = {
    canon: "最高权威设定",
    character_profile: "角色设定",
    published_history: "已发布历史剧情",
    confirmed_inference: "已确认历史归纳",
    draft: "待审新稿",
    core_canon: "最高权威设定",
    formal_record: "正式资料",
    reference: "参考资料",
    unresolved: "权威级别未确认",
  };
  return names[value] || value || "权威级别未标注";
}

export function characterDriftReportPath(
  projectId: string,
  runId: string,
  issueId: string,
): string {
  const project = encodeURIComponent(projectId);
  const run = encodeURIComponent(runId);
  const params = new URLSearchParams({
    category: "character_drift",
    status: "all",
    issue: issueId,
  });
  return `/app/projects/${project}/runs/${run}/report?${params.toString()}`;
}
