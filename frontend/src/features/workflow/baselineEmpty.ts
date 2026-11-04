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

export function baselineEmptyReason(value: unknown): BaselineEmptyReason | null {
  return typeof value === "string" && Object.hasOwn(reasons, value) ? value as BaselineEmptyReason : null;
}

export function baselineEmptyMessage(value: BaselineEmptyReason | null | undefined) {
  return reasons[value ?? "unknown"];
}
