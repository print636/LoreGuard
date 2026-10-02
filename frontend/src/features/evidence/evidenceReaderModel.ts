import type { Evidence } from "../../types/visualization.ts";

export const EVIDENCE_PAGE_SIZE = 120;
export type EvidenceKind = "issue" | "review_clue" | "provisional_clue";
export type EvidenceSelection = {
  projectId: string;
  runId: string;
  kind: EvidenceKind;
  itemId: string;
  evidenceIndex: number;
  expected: Evidence;
  expectedVersion?: number;
};
export type EvidenceRequest = { selection: EvidenceSelection; offset: number | null; limit: number };
export type EvidenceSource = {
  availability: "full_context" | "excerpt_only";
  provenance: "run_input" | "confirmed_trait_source";
  source_run_id: string | null;
  document_version: number;
  content_sha256: string;
  char_count: number | null;
  line_count: number | null;
};
export type EvidencePreview = {
  run: { id: string; project_id: string };
  reference: Evidence & { kind: EvidenceKind; item_id: string; evidence_index: number };
  source: EvidenceSource;
  lines: Array<{ line_number: number; text: string; is_evidence: boolean }>;
  page: { offset: number; limit: number; total: number; has_more: boolean } | null;
  message: string | null;
};

export function evidenceKindLabel(kind: EvidenceKind): string {
  return kind === "issue" ? "正式一致性问题的引用" : kind === "review_clue" ? "角色审查线索的引用（待复核）" : "模型提案的引用（未验证）";
}

export function referenceRange(reference: Pick<Evidence, "line_start" | "line_end">): string {
  return `第 ${reference.line_start}${reference.line_end === reference.line_start ? "" : `–${reference.line_end}`} 行`;
}

export function sameEvidenceSelection(left: EvidenceSelection | null, right: EvidenceSelection | null): boolean {
  return Boolean(left && right && left.projectId === right.projectId && left.runId === right.runId &&
    left.kind === right.kind && left.itemId === right.itemId && left.evidenceIndex === right.evidenceIndex &&
    left.expectedVersion === right.expectedVersion && left.expected.document_id === right.expected.document_id &&
    left.expected.document_name === right.expected.document_name && left.expected.line_start === right.expected.line_start &&
    left.expected.line_end === right.expected.line_end && left.expected.text === right.expected.text);
}

export function isCurrentEvidenceRequest(received: EvidenceRequest | null, selected: EvidenceRequest | null): boolean {
  return Boolean(received && selected && sameEvidenceSelection(received.selection, selected.selection) &&
    received.offset === selected.offset && received.limit === selected.limit);
}

export function evidencePreviewPath(request: EvidenceRequest): string {
  if (!Number.isSafeInteger(request.selection.evidenceIndex) || request.selection.evidenceIndex < 0 || request.selection.evidenceIndex > 11 ||
    (request.offset !== null && (!Number.isSafeInteger(request.offset) || request.offset < 0)) ||
    !Number.isSafeInteger(request.limit) || request.limit < 1 || request.limit > 200) {
    throw new TypeError("返回的引用读取范围无效，请重新打开证据。");
  }
  const params = new URLSearchParams({
    kind: request.selection.kind, item_id: request.selection.itemId,
    evidence_index: String(request.selection.evidenceIndex), limit: String(request.limit),
  });
  if (request.offset !== null) params.set("offset", String(request.offset));
  return `/api/v1/analysis-runs/${encodeURIComponent(request.selection.runId)}/evidence-preview?${params}`;
}

function record(value: unknown): Record<string, unknown> | null {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function nonNegativeInteger(value: unknown): value is number {
  return Number.isSafeInteger(value) && Number(value) >= 0;
}

export function sameEvidenceSource(left: EvidenceSource, right: EvidenceSource): boolean {
  return left.availability === right.availability && left.provenance === right.provenance &&
    left.source_run_id === right.source_run_id && left.document_version === right.document_version &&
    left.content_sha256 === right.content_sha256 && left.char_count === right.char_count && left.line_count === right.line_count;
}

/** Validate the clicked reference and frozen binding before accepting any source text. */
export function normalizeEvidencePreview(value: unknown, request: EvidenceRequest): EvidencePreview {
  const payload = record(value);
  const run = record(payload?.run);
  const reference = record(payload?.reference);
  const source = record(payload?.source);
  const expected = request.selection.expected;
  if (!payload || !run || !reference || !source ||
    run.id !== request.selection.runId || run.project_id !== request.selection.projectId ||
    reference.kind !== request.selection.kind || reference.item_id !== request.selection.itemId ||
    reference.evidence_index !== request.selection.evidenceIndex ||
    reference.document_id !== expected.document_id || reference.document_name !== expected.document_name ||
    reference.line_start !== expected.line_start || reference.line_end !== expected.line_end || reference.text !== expected.text ||
    !nonNegativeInteger(reference.line_start) || reference.line_start < 1 ||
    !nonNegativeInteger(reference.line_end) || reference.line_end < reference.line_start ||
    (source.availability !== "full_context" && source.availability !== "excerpt_only") ||
    (source.provenance !== "run_input" && source.provenance !== "confirmed_trait_source") ||
    !nonNegativeInteger(source.document_version) || source.document_version < 1 ||
    (request.selection.expectedVersion !== undefined && source.document_version !== request.selection.expectedVersion) ||
    typeof source.content_sha256 !== "string" || !/^[a-f0-9]{64}$/i.test(source.content_sha256) ||
    !Array.isArray(payload.lines) || (payload.message !== null && typeof payload.message !== "string")) {
    throw new TypeError("返回的原文与所选报告证据不一致，请重新打开证据或重试。");
  }
  let page: EvidencePreview["page"] = null;
  let lines: EvidencePreview["lines"] = [];
  if (source.availability === "excerpt_only") {
    if (source.provenance !== "confirmed_trait_source" || source.source_run_id !== null ||
      source.char_count !== null || source.line_count !== null || payload.page !== null || payload.lines.length !== 0 ||
      typeof payload.message !== "string" || !payload.message.trim()) {
      throw new TypeError("返回的冻结摘录来源不完整，请重新打开证据或重试。");
    }
  } else {
    const sourcePage = record(payload.page);
    const expectedOffset = request.offset ?? Math.max(0, expected.line_start - 1 - 12);
    if (typeof source.source_run_id !== "string" || !source.source_run_id ||
      (source.provenance === "run_input" && source.source_run_id !== request.selection.runId) ||
      !nonNegativeInteger(source.char_count) || !nonNegativeInteger(source.line_count) ||
      reference.line_end > source.line_count || payload.message !== null || !sourcePage ||
      sourcePage.offset !== expectedOffset || sourcePage.limit !== request.limit ||
      sourcePage.total !== source.line_count || typeof sourcePage.has_more !== "boolean") {
      throw new TypeError("返回的冻结原文来源或分页不一致，请重试。");
    }
    const total = source.line_count;
    const expectedCount = Math.min(request.limit, Math.max(0, total - expectedOffset));
    if (payload.lines.length !== expectedCount || sourcePage.has_more !== (expectedOffset + expectedCount < total)) {
      throw new TypeError("返回的冻结原文分页不完整，请重试。");
    }
    lines = payload.lines.map((value, index) => {
      const line = record(value);
      const number = expectedOffset + index + 1;
      const isEvidence = number >= expected.line_start && number <= expected.line_end;
      if (!line || line.line_number !== number || typeof line.text !== "string" || line.is_evidence !== isEvidence) {
        throw new TypeError("返回的原文行号或引用范围不一致，请重试。");
      }
      return { line_number: number, text: line.text, is_evidence: isEvidence };
    });
    page = { offset: expectedOffset, limit: request.limit, total, has_more: sourcePage.has_more };
  }
  return {
    run: { id: request.selection.runId, project_id: request.selection.projectId },
    reference: { ...expected, kind: request.selection.kind, item_id: request.selection.itemId, evidence_index: request.selection.evidenceIndex },
    source: {
      availability: source.availability, provenance: source.provenance,
      source_run_id: source.source_run_id as string | null, document_version: source.document_version,
      content_sha256: source.content_sha256, char_count: source.char_count as number | null, line_count: source.line_count as number | null,
    },
    lines, page, message: payload.message as string | null,
  };
}
