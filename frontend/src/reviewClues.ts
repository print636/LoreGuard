import { apiJson } from "./api/client.ts";
import type { Evidence } from "./types/visualization";

export type ReviewClue = {
  id: string;
  report_class: "review_clue";
  title: string;
  explanation: string;
  evidence: [Evidence, Evidence, ...Evidence[]];
  metadata: {
    final_outcome: "needs_confirmation" | "unverifiable";
    review_reason: string;
  };
};

export type ReviewClueResult = {
  items: ReviewClue[];
  truncated: boolean;
  unavailable_count: number;
  scan_limited: boolean;
};

const maximumClues = 64;
const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null;
}

function requiredText(value: unknown, field: string, maxLength: number): string {
  if (typeof value !== "string" || !value.trim() || value.length > maxLength) {
    throw new TypeError(`角色审查线索的 ${field} 字段无效`);
  }
  return value.trim();
}

function sourceText(value: unknown): string {
  if (typeof value !== "string" || !value.trim() || value.length > 20_000) {
    throw new TypeError("角色审查线索的原文文本无效");
  }
  return value;
}

function evidenceSpan(value: unknown): Evidence {
  const source = record(value);
  if (!source) throw new TypeError("角色审查线索的原文证据无效");
  const line_start = source.line_start;
  const line_end = source.line_end;
  if (
    !Number.isSafeInteger(line_start) || !Number.isSafeInteger(line_end) ||
    (line_start as number) < 1 || (line_end as number) < (line_start as number) ||
    (line_end as number) > 10_000_000
  ) {
    throw new TypeError("角色审查线索的原文行号无效");
  }
  return {
    document_id: requiredText(source.document_id, "document_id", 200),
    document_name: requiredText(source.document_name, "document_name", 255),
    line_start: line_start as number,
    line_end: line_end as number,
    text: sourceText(source.text),
  };
}

export function normalizeReviewClueResult(payload: unknown): ReviewClueResult {
  const source = record(payload);
  if (
    !source || !Array.isArray(source.items) ||
    source.items.length > maximumClues || typeof source.truncated !== "boolean" ||
    (source.truncated && source.items.length !== maximumClues)
  ) {
    throw new TypeError("角色审查线索列表格式无效");
  }
  const unavailableCount = source.unavailable_count ?? 0;
  if (!Number.isSafeInteger(unavailableCount) || (unavailableCount as number) < 0) {
    throw new TypeError("角色审查线索的历史不可展示数量无效");
  }
  const scanLimited = source.scan_limited ?? false;
  if (typeof scanLimited !== "boolean") {
    throw new TypeError("角色审查线索的扫描上限状态无效");
  }
  const seen = new Set<string>();
  const items = source.items.map((value): ReviewClue => {
    const clue = record(value);
    const metadata = record(clue?.metadata);
    if (!clue || clue.report_class !== "review_clue" || !metadata) {
      throw new TypeError("角色审查线索条目格式无效");
    }
    const id = requiredText(clue.id, "id", 36);
    if (!uuidPattern.test(id) || seen.has(id)) throw new TypeError("角色审查线索的 id 字段无效");
    seen.add(id);
    if (clue.category !== "character_drift") {
      throw new TypeError("角色审查线索的 category 字段无效");
    }
    if (metadata.final_outcome !== "needs_confirmation" && metadata.final_outcome !== "unverifiable") {
      throw new TypeError("角色审查线索的审查结论无效");
    }
    if (!Array.isArray(clue.evidence) || clue.evidence.length < 2 || clue.evidence.length > 12) {
      throw new TypeError("角色审查线索缺少双侧原文证据");
    }
    const evidence = clue.evidence.map(evidenceSpan) as [Evidence, Evidence, ...Evidence[]];
    if (
      evidence[0].document_id === evidence[1].document_id &&
      evidence[0].document_name === evidence[1].document_name &&
      evidence[0].line_start === evidence[1].line_start &&
      evidence[0].line_end === evidence[1].line_end &&
      evidence[0].text === evidence[1].text
    ) {
      throw new TypeError("角色审查线索的双侧证据不可指向同一位置");
    }
    return {
      id,
      report_class: "review_clue",
      title: requiredText(clue.title, "title", 300),
      explanation: requiredText(clue.explanation, "explanation", 4_000),
      evidence,
      metadata: {
        final_outcome: metadata.final_outcome,
        review_reason: requiredText(metadata.review_reason, "review_reason", 128),
      },
    };
  });
  return {
    items,
    truncated: source.truncated,
    unavailable_count: unavailableCount as number,
    scan_limited: scanLimited,
  };
}

export async function fetchReviewClues(
  runId: string,
  signal?: AbortSignal,
): Promise<ReviewClueResult> {
  const payload = await apiJson<unknown>(
    `/api/v1/analysis-runs/${encodeURIComponent(runId)}/review-clues`,
    { signal },
  );
  return normalizeReviewClueResult(payload);
}
