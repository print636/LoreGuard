import { safeReturnTo } from "../routing.ts";

export type PublicSessionStatus = "checking" | "signed-out" | "failed";
export type PublicFeatureAction = "create" | "import" | "sample" | "model" | "account" | "input" | "file" | "check" | "baseline" | "compare" | "revise" | "graph" | "timeline" | "feedback" | "export";
export type PublicEntryAction = "workspace" | PublicFeatureAction;

export const PUBLIC_SESSION_PROBE_TIMEOUT_MS = 4_000;

type PublicFeatureEntry = {
  kind: "feature";
  label: string;
  returnTo: string;
  reason: string;
};

export const publicEntryActions: { workspace: { kind: "navigation"; label: string; returnTo: string } } & Record<PublicFeatureAction, PublicFeatureEntry> = {
  workspace: {
    kind: "navigation",
    label: "进入工作区",
    returnTo: "/app",
  },
  create: {
    kind: "feature",
    label: "新建故事项目",
    returnTo: "/app",
    reason: "创建项目会保存到你的个人工作区，需要先登录。登录后请在项目中心自行创建。",
  },
  import: {
    kind: "feature",
    label: "导入已有故事",
    returnTo: "/app",
    reason: "导入文稿会保存到你的故事项目，需要先登录。登录后请在项目中心自行选择文件并导入。",
  },
  model: {
    kind: "feature",
    label: "模型与密钥",
    returnTo: "/app/settings/model",
    reason: "模型配置与密钥属于你的账户，需要登录后查看和修改。",
  },
  sample: { kind: "feature", label: "打开原创样例", returnTo: "/app", reason: "打开样例会在你的工作区创建故事项目，需要先登录。登录后由你决定是否打开样例。" },
  account: { kind: "feature", label: "账户安全", returnTo: "/app/settings/account", reason: "账户与会话设置包含个人信息，需要登录后访问。" },
  input: { kind: "feature", label: "输入文稿", returnTo: "/check", reason: "处理文稿需要使用你的故事项目与审查设置。请先登录，再自行输入和提交文稿。" },
  file: { kind: "feature", label: "选择文稿文件", returnTo: "/projects", reason: "选择与导入文稿会使用你的个人项目，需要先登录。登录后再选择文件；本页不会读取文件。" },
  check: { kind: "feature", label: "开始校验", returnTo: "/check", reason: "校验会处理你的文稿，并可能调用模型产生费用，需要先登录。登录后请确认项目、资料和范围，再自行开始。" },
  baseline: { kind: "feature", label: "建立角色基线", returnTo: "/characters", reason: "归纳角色基线需要读取你的故事资料，并可能调用模型，需要先登录。登录后再准备资料并主动运行。" },
  compare: { kind: "feature", label: "对比文稿", returnTo: "/diff", reason: "版本对比需要读取项目中的文稿，需要先登录。登录后再选择要比较的文档。" },
  revise: { kind: "feature", label: "上传修订稿", returnTo: "/revision", reason: "保存修订稿需要使用你的个人项目，需要先登录。登录后再选择运行和修订文件。" },
  graph: { kind: "feature", label: "生成关系图", returnTo: "/visual", reason: "生成关系图需要读取所选运行的故事资料，并可能调用模型，需要先登录。登录后再自行选择运行与生成。" },
  timeline: { kind: "feature", label: "生成时间线", returnTo: "/visual", reason: "生成时间线需要读取所选运行的故事资料，并可能调用模型，需要先登录。登录后再自行选择运行与生成。" },
  feedback: { kind: "feature", label: "提交反馈", returnTo: "/report", reason: "反馈需要保存到你的项目报告，需要先登录。登录后再选择问题并填写反馈。" },
  export: { kind: "feature", label: "导出报告", returnTo: "/report", reason: "导出需要读取你的项目报告，需要先登录。登录后再选择报告并自行导出。" },
};

export function publicEntryAuthHref(mode: "login" | "register", returnTo: string): string {
  return `/${mode}?returnTo=${encodeURIComponent(safeReturnTo(returnTo))}`;
}

export function publicSessionMessage(status: PublicSessionStatus): string {
  if (status === "checking") return "正在确认登录状态，确认完成前不能使用个人功能，可继续浏览空工作区。";
  if (status === "failed") return "暂时无法确认登录状态。你仍可阅读本页和浏览空工作区，或前往登录后使用功能。";
  return "本页和空工作区无需登录。创建、导入、校验或使用个人设置时，需要登录账户。";
}

/** A stopped or superseded probe must never replace a newer session decision. */
export function canApplySessionProbe(generation: number, activeGeneration: number, aborted: boolean): boolean {
  return generation === activeGeneration && !aborted;
}
