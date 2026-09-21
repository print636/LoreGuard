import {
  characterSections,
  type CharacterSection,
} from "./types.ts";

export type CharacterRouteState = {
  characterId: string | null;
  section: CharacterSection;
  candidateId: string | null;
  page: number;
  query: string;
};

// Character keys are server-owned, normalized opaque identifiers and commonly
// contain CJK names. Keep them linkable without accepting path separators,
// markup delimiters, whitespace, or control characters.
const idPattern = /^[\p{L}\p{N}_.:-]{1,160}$/u;
const maximumPage = 10_000;
const maximumQueryLength = 120;

function safeId(value: string | null): string | null {
  return value && idPattern.test(value) ? value : null;
}

function safeQuery(value: string | null): string {
  if (!value) return "";
  return value.replace(/[\u0000-\u001f\u007f]/g, "").trim().slice(0, maximumQueryLength);
}

function safePage(value: string | null): number {
  if (!value || !/^\d{1,5}$/.test(value)) return 1;
  const parsed = Number(value);
  return Number.isSafeInteger(parsed) && parsed >= 1 && parsed <= maximumPage
    ? parsed
    : 1;
}

export function characterRouteStateFromSearch(search: string): CharacterRouteState {
  const params = new URLSearchParams(search);
  const rawSection = params.get("section") || "profile";
  const section = characterSections.includes(rawSection as CharacterSection)
    ? (rawSection as CharacterSection)
    : "profile";
  return {
    characterId: safeId(params.get("character")),
    section,
    candidateId:
      section === "candidates" ? safeId(params.get("candidate")) : null,
    page: safePage(params.get("page")),
    query: safeQuery(params.get("q")),
  };
}

export function characterSearch(state: CharacterRouteState): string {
  const params = new URLSearchParams({
    section: characterSections.includes(state.section)
      ? state.section
      : "profile",
    page: String(
      Number.isSafeInteger(state.page) && state.page >= 1 && state.page <= maximumPage
        ? state.page
        : 1,
    ),
  });
  const characterId = safeId(state.characterId);
  const candidateId = safeId(state.candidateId);
  const query = safeQuery(state.query);
  if (characterId) params.set("character", characterId);
  if (state.section === "candidates" && candidateId) {
    params.set("candidate", candidateId);
  }
  if (query) params.set("q", query);
  return `?${params.toString()}`;
}
