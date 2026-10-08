export const guestWorkspaceViews = [
  { id: "center", path: "/app", label: "项目中心", title: "项目", emptyTitle: "还没有项目", description: "访客工作区默认为空。你可以浏览栏目；创建或导入自己的故事时再登录。" },
  { id: "check", path: "/check", label: "文稿校验", title: "文稿校验台", emptyTitle: "还没有待审文稿", description: "先了解校验台。实际输入、选择文稿或开始校验时，需要登录并准备你的故事项目。" },
  { id: "projects", path: "/projects", label: "项目与文档", title: "项目与文档", emptyTitle: "还没有文档", description: "这里展示项目的正文、设定与资料身份。登录后自行选择项目并导入文稿。" },
  { id: "characters", path: "/characters", label: "角色档案", title: "角色档案", emptyTitle: "还没有角色档案", description: "角色基线来自故事资料与作者确认。空工作区没有已确认特征或待审候选。" },
  { id: "diff", path: "/diff", label: "版本对比", title: "版本对比", emptyTitle: "还没有可比较的文稿", description: "比较同一故事中的文稿版本。登录后准备资料，再自行选择要对照的版本。" },
  { id: "revision", path: "/revision", label: "修订复检", title: "修订复检", emptyTitle: "还没有待修订的运行", description: "修订复检衔接已完成的审查。登录后选择项目与运行，再上传新稿并决定是否复检。" },
  { id: "visual", path: "/visual", label: "关系与时间", title: "关系与时间", emptyTitle: "还没有关系图或时间线", description: "关系与时间属于所选运行，需在准备故事资料后主动生成。空工作区不调用模型。" },
  { id: "audit", path: "/audit", label: "运行审计", title: "运行审计", emptyTitle: "还没有运行记录", description: "运行审计说明一次审查使用了哪些材料、版本与模型阶段。空工作区没有运行历史。" },
  { id: "report", path: "/report", label: "完整报告", title: "完整报告", emptyTitle: "还没有审查报告", description: "先看实际覆盖，再核对问题与双方原文。登录并完成审查后，才会出现自己的报告。" },
] as const;

export type GuestWorkspaceView = (typeof guestWorkspaceViews)[number]["id"];

/** Exact paths only. Query/hash values never select a private resource. */
export function guestWorkspaceViewFromPath(pathname: string): GuestWorkspaceView | null {
  const normalized = pathname.replace(/\/+$/, "") || "/";
  return guestWorkspaceViews.find((view) => view.path === normalized)?.id ?? null;
}
