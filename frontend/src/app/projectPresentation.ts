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
