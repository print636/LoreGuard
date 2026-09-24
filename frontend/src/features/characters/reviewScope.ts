export type ReviewScopeEpoch = {
  key: string;
  generation: number;
};

export function reviewScopeKey(
  projectId: string,
  characterId: string | null,
  candidateId: string | null,
): string {
  // JSON encoding keeps opaque IDs unambiguous without trusting delimiters.
  return JSON.stringify([projectId, characterId, candidateId]);
}

export function advanceReviewScope(
  current: ReviewScopeEpoch,
  key: string,
): ReviewScopeEpoch {
  return current.key === key
    ? current
    : { key, generation: current.generation + 1 };
}

export function isCurrentReviewScope(
  current: ReviewScopeEpoch,
  started: ReviewScopeEpoch,
): boolean {
  return current.key === started.key && current.generation === started.generation;
}

export function isCurrentReviewRequest(
  current: ReviewScopeEpoch,
  started: ReviewScopeEpoch,
  latestRequestGeneration: number,
  startedRequestGeneration: number,
): boolean {
  return isCurrentReviewScope(current, started) &&
    latestRequestGeneration === startedRequestGeneration;
}
