import { safeReturnTo } from "../routing.ts";

export type PublicSessionStatus = "checking" | "signed-out" | "failed";
export type PublicEntryAction = "workspace" | "create" | "import" | "model";

export const PUBLIC_SESSION_PROBE_TIMEOUT_MS = 4_000;

export const publicEntryActions: Record<PublicEntryAction, {
  label: string;
  returnTo: string;
  reason: string;
}> = {
  workspace: {
    label: "进入工作区",
    returnTo: "/app",
    reason: "工作区包含你的故事项目、文稿和审查记录，需要登录后访问。",
  },
  create: {
    label: "新建故事项目",
    returnTo: "/app",
    reason: "创建项目会保存到你的个人工作区，需要先登录。登录后请在项目中心自行创建。",
  },
  import: {
    label: "导入已有故事",
    returnTo: "/app",
    reason: "导入文稿会保存到你的故事项目，需要先登录。登录后请在项目中心自行选择文件并导入。",
  },
  model: {
    label: "模型与密钥",
    returnTo: "/app/settings/model",
    reason: "模型配置与密钥属于你的账户，需要登录后查看和修改。",
  },
};

export function publicEntryAuthHref(mode: "login" | "register", returnTo: string): string {
  return `/${mode}?returnTo=${encodeURIComponent(safeReturnTo(returnTo))}`;
}

export function publicSessionMessage(status: PublicSessionStatus): string {
  if (status === "checking") return "正在确认登录状态，确认完成前不会打开工作区。";
  if (status === "failed") return "暂时无法确认登录状态。你仍可阅读本页，或前往登录后继续。";
  return "本页无需登录。使用个人项目和模型设置时，需要登录账户。";
}

/** A stopped or superseded probe must never replace a newer session decision. */
export function canApplySessionProbe(generation: number, activeGeneration: number, aborted: boolean): boolean {
  return generation === activeGeneration && !aborted;
}
