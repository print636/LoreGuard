import type {
  ProfileEvidence,
  ProfileSupportBinding,
  ProfileSupportBindingsV1,
  ProfileSupportSpan,
  SupportBindingStatus,
} from "./types.ts";

type JsonRecord = Record<string, unknown>;
type SegmentKind = "plain" | "context" | "target";

export type EvidenceSegment = { text: string; kind: SegmentKind };

const supportId = /^L([1-9][0-9]{0,7}):A[1-9][0-9]{0,2}$/;
const spanRoles = new Set(["target", "actor_anchor", "label_anchor", "bridge"]);

function record(value: unknown): JsonRecord | null {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as JsonRecord : null;
}

function exactKeys(value: JsonRecord, keys: string[]): boolean {
  const actual = Object.keys(value);
  return actual.length === keys.length && keys.every((key) => Object.hasOwn(value, key));
}

function validSupportId(value: unknown, line: number): value is string {
  if (typeof value !== "string") return false;
  const match = supportId.exec(value);
  return Boolean(match && Number(match[1]) === line);
}

function hasLoneSurrogate(value: string): boolean {
  for (const point of Array.from(value)) {
    if (point.length === 1) {
      const code = point.charCodeAt(0);
      if (code >= 0xd800 && code <= 0xdfff) return true;
    }
  }
  return false;
}

function parseSpan(
  value: unknown,
  evidence: ProfileEvidence,
  codepoints: string[],
): ProfileSupportSpan | null {
  const raw = record(value);
  if (!raw || !exactKeys(raw, ["support_id", "start_offset", "end_offset", "role"])) return null;
  const start = raw.start_offset;
  const end = raw.end_offset;
  if (
    !validSupportId(raw.support_id, evidence.line_start) ||
    typeof start !== "number" || !Number.isSafeInteger(start) || start < 0 ||
    typeof end !== "number" || !Number.isSafeInteger(end) ||
    end <= start || end > codepoints.length ||
    typeof raw.role !== "string" || !spanRoles.has(raw.role) ||
    !codepoints.slice(start, end).join("").trim()
  ) return null;
  return raw as ProfileSupportSpan;
}

function parseBinding(value: unknown, evidence: ProfileEvidence[]): ProfileSupportBinding | null {
  const raw = record(value);
  if (!raw || !exactKeys(raw, [
    "evidence_index", "support_id", "target", "actor_anchor_id",
    "label_anchor_id", "scope_relation", "context",
  ])) return null;
  const index = raw.evidence_index;
  if (typeof index !== "number" || !Number.isSafeInteger(index) || index < 0 || index >= evidence.length) return null;
  const source = evidence[index];
  if (
    !source.input_id || !source.document_id || source.document_version === null ||
    source.line_start !== source.line_end || !source.text.trim() ||
    hasLoneSurrogate(source.text) || !validSupportId(raw.support_id, source.line_start)
  ) return null;
  const points = Array.from(source.text);
  const target = parseSpan(raw.target, source, points);
  if (!target || target.role !== "target" || target.support_id !== raw.support_id) return null;
  if (!Array.isArray(raw.context)) return null;
  const context = raw.context.map((part) => parseSpan(part, source, points));
  if (context.some((part) => part === null)) return null;
  const parsedContext = context as ProfileSupportSpan[];
  let priorEnd = 0;
  for (const span of parsedContext) {
    if (
      span.role === "target" || span.start_offset < priorEnd ||
      span.end_offset > target.start_offset
    ) return null;
    priorEnd = span.end_offset;
  }
  const actor = raw.actor_anchor_id;
  const label = raw.label_anchor_id;
  if ((actor !== null && !validSupportId(actor, source.line_start)) ||
      (label !== null && !validSupportId(label, source.line_start))) return null;
  if (actor !== null && actor === target.support_id) return null;
  if (actor !== null && !parsedContext.some((span) =>
    span.support_id === actor && span.role === "actor_anchor")) return null;
  if (label !== null && label !== target.support_id &&
      !parsedContext.some((span) => span.support_id === label &&
        (span.role === "label_anchor" || (label === actor && span.role === "actor_anchor")))) return null;
  if (parsedContext.some((span) =>
    (span.role === "actor_anchor" && span.support_id !== actor) ||
    (span.role === "label_anchor" && span.support_id !== label))) return null;
  if (
    (raw.scope_relation === "local" &&
      (actor !== null || (label !== null && label !== target.support_id) || parsedContext.length > 0)) ||
    (raw.scope_relation === "same_actor_continuation" &&
      (actor === null || (label !== null && label !== target.support_id))) ||
    (raw.scope_relation === "labelled_elaboration" && (label === null || label === target.support_id)) ||
    !["local", "same_actor_continuation", "labelled_elaboration"].includes(raw.scope_relation as string)
  ) return null;
  return {
    evidence_index: index,
    support_id: raw.support_id,
    target: target as ProfileSupportBinding["target"],
    actor_anchor_id: actor as string | null,
    label_anchor_id: label as string | null,
    scope_relation: raw.scope_relation as ProfileSupportBinding["scope_relation"],
    context: parsedContext,
  };
}

/** Interpret only the server's verified, versioned coordinates; never guess a clause. */
export function readSupportBindings(
  status: unknown,
  payload: unknown,
  evidence: ProfileEvidence[],
  candidateSourceVerified: boolean,
  requireExactEvidence = false,
): { status: SupportBindingStatus; payload: ProfileSupportBindingsV1 | null } {
  if (status === "legacy" && payload === null) return { status: "legacy", payload: null };
  if (status !== "verified" || !candidateSourceVerified) return { status: "invalid", payload: null };
  const raw = record(payload);
  if (
    !raw || !exactKeys(raw, ["schema_version", "index_version", "bindings"]) ||
    raw.schema_version !== "character-support-bindings-v1" ||
    raw.index_version !== "assertion-index-v1" ||
    !Array.isArray(raw.bindings) || raw.bindings.length < 1 || raw.bindings.length > 48
  ) return { status: "invalid", payload: null };
  const bindings = raw.bindings.map((value) => parseBinding(value, evidence));
  if (bindings.some((binding) => binding === null)) return { status: "invalid", payload: null };
  const parsed = bindings as ProfileSupportBinding[];
  const identities = new Set(parsed.map((binding) =>
    `${binding.evidence_index}:${binding.support_id}:${binding.actor_anchor_id || ""}:${binding.label_anchor_id || ""}`));
  if (identities.size !== parsed.length) return { status: "invalid", payload: null };
  if (requireExactEvidence && parsed.some((binding) => {
    const source = evidence[binding.evidence_index];
    return !source.source_verified || !source.source_text_exact;
  })) return { status: "invalid", payload: null };
  return {
    status: "verified",
    payload: {
      schema_version: "character-support-bindings-v1",
      index_version: "assertion-index-v1",
      bindings: parsed,
    },
  };
}

/** Python's frozen-input offsets count Unicode code points, not JS UTF-16 units. */
export function splitEvidenceBySupport(
  text: string,
  bindings: ProfileSupportBinding[],
): EvidenceSegment[] {
  const points = Array.from(text);
  const spans = bindings.flatMap((binding) => [
    { ...binding.target, kind: "target" as const },
    ...binding.context.map((span) => ({ ...span, kind: "context" as const })),
  ]);
  if (!spans.length) return [{ text, kind: "plain" }];
  const boundaries = Array.from(new Set([0, points.length, ...spans.flatMap((span) =>
    [span.start_offset, span.end_offset])])).sort((a, b) => a - b);
  const segments: EvidenceSegment[] = [];
  for (let index = 0; index < boundaries.length - 1; index += 1) {
    const start = boundaries[index];
    const end = boundaries[index + 1];
    if (start === end) continue;
    const kind: SegmentKind = spans.some((span) =>
      span.kind === "target" && span.start_offset <= start && end <= span.end_offset)
      ? "target"
      : spans.some((span) => span.kind === "context" && span.start_offset <= start && end <= span.end_offset)
        ? "context" : "plain";
    const part = points.slice(start, end).join("");
    const previous = segments.at(-1);
    if (previous?.kind === kind) previous.text += part;
    else segments.push({ text: part, kind });
  }
  return segments;
}
