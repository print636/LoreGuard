import { reportSearch, workspacePath } from "../routing.ts";

type ProjectEntrySummary = {
  id: string;
  active_document_count: number;
  latest_run: { id: string; status: string } | null;
};

export function projectNextAction(
  project: ProjectEntrySummary,
): { label: string; path: string } {
  const run = project.latest_run;
  if (!run) {
    return project.active_document_count > 0
      ? { label: "准备首次校验", path: workspacePath("check", project.id) }
      : { label: "导入第一份文稿", path: workspacePath("projects", project.id) };
  }
  if (run.status === "completed") {
    return {
      label: "查看最近报告",
      path: `${workspacePath("report", project.id, run.id)}${reportSearch({
        category: "all",
        status: "all",
        issueId: null,
      })}`,
    };
  }
  if (["queued", "running"].includes(run.status)) {
    return {
      label: "查看校验进度",
      path: workspacePath("audit", project.id, run.id),
    };
  }
  return {
    label: run.status === "failed" ? "查看失败详情" : "查看取消详情",
    path: workspacePath("audit", project.id, run.id),
  };
}

export function backendTimestamp(value: string): number {
  const trimmed = value.trim();
  if (!trimmed) return Number.NaN;
  const hasExplicitZone = /(?:Z|[+-]\d{2}:?\d{2})$/i.test(trimmed);
  return Date.parse(hasExplicitZone ? trimmed : `${trimmed}Z`);
}

export function relativeProjectDate(
  value: string,
  now = Date.now(),
): string {
  const timestamp = backendTimestamp(value);
  if (!Number.isFinite(timestamp)) return "时间未知";
  const minutes = Math.max(0, Math.round((now - timestamp) / 60_000));
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.round(hours / 24);
  return days < 30
    ? `${days} 天前`
    : new Date(timestamp).toLocaleDateString("zh-CN");
}
