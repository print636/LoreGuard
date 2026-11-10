import type { RunInputSnapshot } from "../../runSnapshot.ts";
import { documentRoles } from "../../documentContext.ts";
import { publicationStatusNames } from "../documents/libraryModel.ts";

export const MAX_SOURCE_MATERIALS = 100;
export class SourceMaterialsError extends Error {}

export function sourceMaterialsPath(runId: string, ids: readonly string[]): string {
  if (!runId || ids.length < 1 || ids.length > MAX_SOURCE_MATERIALS || ids.some(id => !id) || new Set(ids).size !== ids.length) {
    throw new TypeError("请选择 1 至 100 份本次审查资料，不要重复选择。");
  }
  const query = new URLSearchParams();
  ids.forEach(id => query.append("document_id", id));
  return `/api/v1/analysis-runs/${encodeURIComponent(runId)}/source-materials.zip?${query}`;
}

export function selectedSourceIds(inputs: readonly RunInputSnapshot[], selected: ReadonlySet<string>): string[] {
  const ids = inputs.map(input => input.document_id);
  if (new Set(ids).size !== ids.length || [...selected].some(id => !ids.includes(id))) {
    throw new TypeError("勾选与本次审查清单不一致，请重新打开资料导出。");
  }
  return ids.filter(id => selected.has(id));
}

export function sourceMaterialContext(input: RunInputSnapshot): string {
  const roles = Object.fromEntries(documentRoles);
  const batch = input.batch_role === "target" ? "待审稿" : input.batch_role === "background" ? "比对依据" : "审查用途未记录";
  const role = roles[input.document_role] || (input.document_role === "story_branch" ? "故事分支" : "资料身份未记录");
  const publication = input.narrative_context?.publication_status;
  return `${batch} · ${role} · 故事范围：${input.story_scope || "未记录"} · ${publication ? publicationStatusNames[publication] || "发布状态未记录" : "发布状态未记录"}`;
}

export function sourceMaterialsFailure(status: number, payload: unknown): string {
  const detail = payload && typeof payload === "object" && "detail" in payload ? (payload as { detail?: unknown }).detail : null;
  const data = detail && typeof detail === "object" && !Array.isArray(detail) ? detail as Record<string, unknown> : null;
  const messages: Record<string, string> = {
    run_not_found: "这份审查不存在或当前账户无权读取。",
    run_not_completed: "这份审查尚未完成，暂不能导出本次资料。",
    source_materials_unavailable: "所选资料中有原文或版本无法核对，本次未生成压缩包，没有改用当前新版。",
    source_materials_too_large: "所选正文超过 20 MiB，请减少勾选后分批下载；本次没有截断资料。",
    invalid_selection: "请选择 1 至 100 份本次审查资料，再重试下载。",
  };
  let message = typeof data?.code === "string" ? messages[data.code] : undefined;
  if (!message) message = status === 401 ? "登录已过期，请重新登录后导出。" : status === 403 || status === 404 ? messages.run_not_found
    : status === 413 ? messages.source_materials_too_large : status === 422 ? messages.invalid_selection
      : "资料导出暂时失败，请稍后重试。没有生成不完整的压缩包。";
  if (data?.code === "source_materials_unavailable" && Array.isArray(data.documents)) {
    const reasons: Record<string, string> = { missing: "原文缺失", content_mismatch: "原文无法核对", context_mismatch: "资料身份无法核对", invalid_source: "来源无法核对" };
    const rows = data.documents.slice(0, MAX_SOURCE_MATERIALS).flatMap(row => {
      if (!row || typeof row !== "object" || !reasons[row.reason]) return [];
      return [`${typeof row.name === "string" && row.name.trim() ? row.name : "一份所选资料"}：${reasons[row.reason]}`];
    });
    if (rows.length) message += ` ${rows.join("；")}。`;
  }
  return message;
}

/** A successful HTTP response must still be an actual, non-empty ZIP download. */
export async function sourceMaterialsBlob(response: Response): Promise<Blob> {
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new SourceMaterialsError(sourceMaterialsFailure(response.status, detail));
  }
  if (response.headers.get("content-type")?.split(";", 1)[0].trim().toLowerCase() !== "application/zip") {
    throw new SourceMaterialsError("返回内容不是资料压缩包，未保存文件。请稍后重试。");
  }
  const blob = await response.blob();
  if (blob.size === 0) throw new SourceMaterialsError("服务器返回了空压缩包，未保存文件。请稍后重试。");
  return blob;
}
