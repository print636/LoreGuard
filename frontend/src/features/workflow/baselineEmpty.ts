const reasons = {
  evidence_unconfirmed: {
    title: "提取到了角色描述，但还未核实原文依据",
    detail: "可以对照原文，看看哪条长期设定适用于这个人物，以及它在什么情境下生效。本次建档记录保留了处理情况。",
  },
  no_candidates: {
    title: "这次没有提取到可确认的角色设定",
    detail: "模型已读完本次资料。可以先查看原文，核对是否写清角色、设定和适用情境，也可以直接用现有资料再试。",
  },
  provider_unavailable: {
    title: "模型服务中途未能完成建档",
    detail: "还有资料没有处理完。服务恢复后可以用现有资料重新建档，无需重新上传。",
  },
  provider_configuration: {
    title: "模型连接需要处理",
    detail: "请先到模型设置核对连接，再回来用现有资料重新建档。",
  },
  incomplete: {
    title: "角色资料还没有处理完整",
    detail: "这次没有可确认的设定。可以查看本次运行的进度与诊断，再决定是否用现有资料重新建档。",
  },
  unknown: {
    title: "暂时没有可确认的角色设定",
    detail: "现有记录不足以说明原因。档案为空不代表资料中没有角色。可以先查看本次运行，再决定是否重新建档。",
  },
};

export type BaselineEmptyReason = keyof typeof reasons;

const evidenceChecks = {
  actor: "描述的是哪位角色",
  actuality: "这件事在故事中是否属实",
  statement_relation: "归纳的内容是否符合原意",
  label_relation: "归纳的角色特征是否与原文相符",
  object_relation: "描述涉及的对象是否一致",
  polarity_relation: "归纳的倾向是否与原意一致",
  level_supported: "原文是否足以支持长期设定",
};

export type BaselineEvidenceCheck = keyof typeof evidenceChecks;

export function baselineEvidenceChecks(value: unknown): BaselineEvidenceCheck[] {
  if (!Array.isArray(value)) return [];
  return (Object.keys(evidenceChecks) as BaselineEvidenceCheck[]).filter(check => value.includes(check));
}

export function baselineEmptyReason(value: unknown): BaselineEmptyReason | null {
  return typeof value === "string" && Object.hasOwn(reasons, value) ? value as BaselineEmptyReason : null;
}

export function baselineEmptyMessage(value: BaselineEmptyReason | null | undefined, checks: BaselineEvidenceCheck[] = []) {
  if (value === "evidence_unconfirmed" && checks.length) {
    return {
      title: reasons[value].title,
      detail: `本次建档未能核实的内容包括${checks.map(check => evidenceChecks[check]).join("、")}。可以打开正式资料对照原文，也可以查看本次建档记录。`,
    };
  }
  return reasons[value ?? "unknown"];
}
