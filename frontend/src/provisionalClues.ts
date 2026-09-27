import { apiJson } from "./api/client.ts";

export type ProvisionalClue = {
  id: string;
  document_id: string;
  document_version: number;
  document_name: string;
  line_start: number;
  line_end: number;
  evidence: string;
  character: string;
  dimension: string;
  proposed_statement: string;
  reason: "partial_model_package";
};

export type ProvisionalClueResult = {
  items: ProvisionalClue[];
  truncated: boolean;
};

const maximumClues = 64;
const clueIdPattern = /^pc_[0-9a-f]{32}$/;
const dimensionNames: Record<string, string> = {
  core_personality: "核心人格",
  preference: "稳定偏好",
  value: "价值观",
  speech_pattern: "语言表现",
  behavior_boundary: "行为边界",
  contextual_behavior: "情境表现",
  current_state: "当前状态",
  unknown: "未分类角色特征",
};

export function provisionalClueDimensionLabel(dimension: string): string {
  return Object.prototype.hasOwnProperty.call(dimensionNames, dimension)
    ? dimensionNames[dimension]
    : "角色特征";
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function requiredText(value: unknown, field: string, maxLength: number): string {
  if (typeof value !== "string") {
    throw new TypeError(`待复核线索的 ${field} 字段无效`);
  }
  const trimmed = value.trim();
  if (!trimmed || trimmed.length > maxLength) {
    throw new TypeError(`待复核线索的 ${field} 字段无效`);
  }
  return trimmed;
}

function positiveInteger(value: unknown, field: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 1) {
    throw new TypeError(`待复核线索的 ${field} 字段无效`);
  }
  return value as number;
}

function lineNumber(value: unknown, field: string): number {
  const line = positiveInteger(value, field);
  if (line > 10_000_000) throw new TypeError(`待复核线索的 ${field} 字段无效`);
  return line;
}

export function normalizeProvisionalClueResult(payload: unknown): ProvisionalClueResult {
  const source = record(payload);
  if (
    !source || !Array.isArray(source.items) ||
    source.items.length > maximumClues || typeof source.truncated !== "boolean" ||
    (source.truncated && source.items.length !== maximumClues)
  ) {
    throw new TypeError("待复核线索列表格式无效");
  }
  const seen = new Set<string>();
  const items = source.items.map((value): ProvisionalClue => {
    const clue = record(value);
    if (!clue) throw new TypeError("待复核线索条目格式无效");
    const id = requiredText(clue.id, "id", 35);
    if (!clueIdPattern.test(id)) throw new TypeError("待复核线索的 id 字段无效");
    if (seen.has(id)) throw new TypeError("待复核线索包含重复条目");
    seen.add(id);
    const line_start = lineNumber(clue.line_start, "line_start");
    const line_end = lineNumber(clue.line_end, "line_end");
    if (line_end < line_start) throw new TypeError("待复核线索的行号范围无效");
    if (clue.reason !== "partial_model_package") {
      throw new TypeError("待复核线索的原因类型无效");
    }
    const dimension = requiredText(clue.dimension, "dimension", 64);
    if (!Object.prototype.hasOwnProperty.call(dimensionNames, dimension)) {
      throw new TypeError("待复核线索的 dimension 字段无效");
    }
    const proposed_statement = requiredText(clue.proposed_statement, "proposed_statement", 300);
    if (proposed_statement.length < 2) {
      throw new TypeError("待复核线索的 proposed_statement 字段无效");
    }
    return {
      id,
      document_id: requiredText(clue.document_id, "document_id", 200),
      document_version: positiveInteger(clue.document_version, "document_version"),
      document_name: requiredText(clue.document_name, "document_name", 255),
      line_start,
      line_end,
      evidence: requiredText(clue.evidence, "evidence", 2_000),
      character: requiredText(clue.character, "character", 64),
      dimension,
      proposed_statement,
      reason: "partial_model_package",
    };
  });
  return { items, truncated: source.truncated };
}

export async function fetchProvisionalClues(
  runId: string,
  signal?: AbortSignal,
): Promise<ProvisionalClueResult> {
  const payload = await apiJson<unknown>(
    `/api/v1/analysis-runs/${encodeURIComponent(runId)}/provisional-clues`,
    { signal },
  );
  return normalizeProvisionalClueResult(payload);
}
