import type { CharacterTraitAxis, CharacterTraitAxisPage } from "./types";

export type AxisDraftErrors = {
  display_name: string;
  definition: string;
};

function normalizedText(value: string): string {
  return value.replace(/\s+/gu, " ").trim();
}

export function validateAxisDraft(
  displayName: string,
  definition: string,
): { value: { display_name: string; definition: string }; errors: AxisDraftErrors } {
  const name = normalizedText(displayName);
  const meaning = normalizedText(definition);
  const nameLength = Array.from(name).length;
  const meaningLength = Array.from(meaning).length;
  return {
    value: { display_name: name, definition: meaning },
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
    },
  };
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
