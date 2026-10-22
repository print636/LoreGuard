const reviewStatuses: Record<string, string> = {
  queued: "等待审查", running: "正在审查", completed: "审查完成",
  failed: "审查未完成", cancelled: "已取消审查",
};

export function reviewStatusLabel(status: string | null | undefined): string {
  return status ? reviewStatuses[status] || "状态待确认" : "尚未审查";
}

export function reviewTimeLabel(value: string | null | undefined): string {
  if (!value || !Number.isFinite(Date.parse(value))) return "时间未记录";
  return new Date(value).toLocaleString("zh-CN", { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

/** Names describe this displayed list position, not a permanent backend identifier. */
export function reviewRecordLabel(position: number, createdAt: string): string {
  return `记录 ${position} · ${reviewTimeLabel(createdAt)}`;
}

export function issueSeverityLabel(severity: string): string {
  return ({ low: "轻微", medium: "一般", high: "严重", critical: "严重" } as Record<string, string>)[severity] || "程度待确认";
}
