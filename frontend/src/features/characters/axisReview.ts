import type { CharacterTraitAxis, CharacterTraitAxisPage, ProfileCandidate } from "./types";

export type AxisDraftErrors = {
  display_name: string;
  definition: string;
  positive_proposition: string;
};

function normalizedText(value: string): string {
  return value.replace(/\s+/gu, " ").trim();
}

export function validateAxisDraft(
  displayName: string,
  definition: string,
  positiveProposition: string,
): { value: { display_name: string; definition: string; positive_proposition: string }; errors: AxisDraftErrors } {
  const name = normalizedText(displayName);
  const meaning = normalizedText(definition);
  const proposition = normalizedText(positiveProposition);
  const nameLength = Array.from(name).length;
  const meaningLength = Array.from(meaning).length;
  const propositionLength = Array.from(proposition).length;
  return {
    value: { display_name: name, definition: meaning, positive_proposition: proposition },
    errors: {
      display_name: !name
        ? "请输入轴名称。"
        : nameLength > 80
          ? "轴名称不能超过 80 字。"
          : "",
      definition: !meaning
        ? "请说明比较的具体行为或性格维度。"
        : meaningLength > 200
          ? "轴定义不能超过 200 字。"
          : "",
      positive_proposition: !proposition
        ? "请写出用于判定正向的明确命题。"
        : propositionLength > 200
          ? "正向命题不能超过 200 字。"
          : "",
    },
  };
}

export function validateAxisPositiveProposition(value: string): {
  value: string;
  error: string;
} {
  const proposition = normalizedText(value);
  const length = Array.from(proposition).length;
  return {
    value: proposition,
    error: !proposition
      ? "请写出用于判定正向的明确命题。"
      : length > 200
        ? "正向命题不能超过 200 字。"
        : "",
  };
}

export function previewAxisPolarity(
  rawPolarity: "positive" | "negative" | "neutral" | "unclear" | null,
  alignment: "same" | "opposite" | "uncertain",
): "positive" | "negative" | null {
  if ((rawPolarity !== "positive" && rawPolarity !== "negative") || alignment === "uncertain") return null;
  if (alignment === "same") return rawPolarity;
  return rawPolarity === "positive" ? "negative" : "positive";
}

export function hasVerifiableFrozenEvidence(candidate: ProfileCandidate): boolean {
  return candidate.source_verified && candidate.support_bindings_status !== "invalid" &&
    (candidate.support_bindings_status !== "verified" || Boolean(candidate.support_bindings_v1?.bindings.length)) &&
    candidate.supporting_evidence.length > 0 &&
    candidate.supporting_evidence.every((evidence) => evidence.source_verified);
}

/** Never infer a confirmed axis meaning from an unverified/currently different proposition. */
export function confirmedAxisPolarity(
  candidate: ProfileCandidate,
  axis: CharacterTraitAxis | null,
): "positive" | "negative" | null {
  if (
    candidate.status !== "confirmed" || !axis ||
    candidate.approved_axis_id !== axis.id ||
    candidate.approved_axis_version !== axis.version ||
    !axis.positive_proposition || !axis.positive_proposition_sha256 ||
    candidate.axis_positive_proposition_sha256 !== axis.positive_proposition_sha256 ||
    !candidate.axis_alignment || !hasVerifiableFrozenEvidence(candidate)
  ) return null;
  const mapped = previewAxisPolarity(candidate.polarity, candidate.axis_alignment);
  return mapped && mapped === candidate.axis_polarity ? mapped : null;
}

export function selectedProjectAxis(
  axes: CharacterTraitAxis[],
  selectedId: string,
): CharacterTraitAxis | null {
  return axes.find((axis) => axis.id === selectedId) ?? null;
}

export function appendAxisPage(
  current: CharacterTraitAxis[],
  page: CharacterTraitAxisPage,
): CharacterTraitAxis[] {
  const seen = new Set(current.map((axis) => axis.id));
  if (page.items.some((axis) => seen.has(axis.id))) {
    throw new Error("作者轴分页出现重复项，请重新读取列表");
  }
  return [...current, ...page.items];
}

export function canCreateNewAxis(input: {
  loaded: boolean;
  loading: boolean;
  error: string;
  dirty: boolean;
  fetchedCount: number;
  total: number;
}): boolean {
  return input.loaded && !input.loading && !input.error && !input.dirty &&
    input.fetchedCount >= input.total;
}
