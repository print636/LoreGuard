/** Private text is cached only in this tab's sessionStorage, never sent by this module. */
export const DRAFT_STORAGE_PREFIX = "loreguard:tab-draft:v1:";
export const MAX_DRAFT_BYTES = 2 * 1024 * 1024;

export type DraftOwner = { userId: string; workspaceId: string };
export type DraftScope = DraftOwner & {
  projectId: string;
  kind: "quick" | "feedback";
  runId?: string;
};
export type DraftLease = DraftOwner & { generation: number };
export type DraftStorage = Pick<Storage, "getItem" | "setItem" | "removeItem" | "key" | "length">;
export type DraftWriteStatus = "saved" | "empty" | "unavailable" | "too_large" | "revoked";
export type QuickDraft = {
  world?: string;
  chapter?: string;
  mode: "body" | "advanced";
  role: "chapter" | "canon" | "character_profile" | "reference";
  scope: string;
};
export type FeedbackDraft = { notes: Record<string, string> };

let activeOwner = "";
let generation = 0;
const ignoredOwners = new Set<string>();
const freshKeys = new Set<string>();

function ownerKey(owner: DraftOwner): string {
  return JSON.stringify([owner.userId, owner.workspaceId]);
}

export function activateDraftSession(owner: DraftOwner): void {
  const next = ownerKey(owner);
  if (activeOwner !== next) {
    generation += 1;
    activeOwner = next;
  }
}

/** A 401 revokes outstanding callbacks but retains the cache for the original account. */
export function suspendDraftSession(): void {
  generation += 1;
  activeOwner = "";
}

export function draftLease(owner: DraftOwner): DraftLease {
  return { ...owner, generation };
}

export function draftLeaseActive(lease: DraftLease): boolean {
  return lease.generation === generation && ownerKey(lease) === activeOwner;
}

function browserStorage(): DraftStorage | null {
  try {
    return typeof window === "undefined" ? null : window.sessionStorage;
  } catch {
    return null;
  }
}

export function draftStorageKey(scope: DraftScope): string {
  return `${DRAFT_STORAGE_PREFIX}${JSON.stringify([
    scope.userId, scope.workspaceId, scope.projectId, scope.kind, scope.runId || "",
  ])}`;
}

function scopeMatchesLease(scope: DraftScope, lease: DraftLease): boolean {
  return scope.userId === lease.userId && scope.workspaceId === lease.workspaceId;
}

function record(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function exactKeys(value: Record<string, unknown>, allowed: string[]): boolean {
  return Object.keys(value).every((key) => allowed.includes(key));
}

export function isQuickDraft(value: unknown): value is QuickDraft {
  if (!record(value) || !exactKeys(value, ["world", "chapter", "mode", "role", "scope"])) return false;
  return (value.world === undefined || typeof value.world === "string") &&
    (value.chapter === undefined || typeof value.chapter === "string") &&
    (typeof value.world === "string" || typeof value.chapter === "string") &&
    (value.mode === "body" || value.mode === "advanced") &&
    typeof value.role === "string" && ["chapter", "canon", "character_profile", "reference"].includes(value.role) &&
    typeof value.scope === "string" && value.scope.length <= 80;
}

export function isFeedbackDraft(value: unknown): value is FeedbackDraft {
  return record(value) && exactKeys(value, ["notes"]) && record(value.notes) &&
    Object.keys(value.notes).length <= 1000 &&
    Object.entries(value.notes).every(([id, text]) => id.length > 0 && id.length <= 200 && typeof text === "string");
}

function validPayload(scope: DraftScope, payload: unknown): boolean {
  return scope.kind === "quick" ? isQuickDraft(payload) : Boolean(scope.runId) && isFeedbackDraft(payload);
}

export function readTabDraft<T extends QuickDraft | FeedbackDraft>(
  scope: DraftScope,
  lease: DraftLease,
  storage: DraftStorage | null = browserStorage(),
): { draft: T | null; status: "ready" | "empty" | "invalid" | "unavailable" | "revoked" } {
  if (!draftLeaseActive(lease) || !scopeMatchesLease(scope, lease)) return { draft: null, status: "revoked" };
  if (!storage) return { draft: null, status: "unavailable" };
  const key = draftStorageKey(scope);
  if (ignoredOwners.has(ownerKey(scope)) && !freshKeys.has(key)) return { draft: null, status: "empty" };
  try {
    const raw = storage.getItem(key);
    if (raw === null) return { draft: null, status: "empty" };
    if (raw.length * 2 > MAX_DRAFT_BYTES) return { draft: null, status: "invalid" };
    let data: unknown;
    try { data = JSON.parse(raw); }
    catch { return { draft: null, status: "invalid" }; }
    if (!record(data) || !exactKeys(data, ["version", "scope", "payload"]) || data.version !== 1 ||
      !record(data.scope) || JSON.stringify(data.scope) !== JSON.stringify(scope) || !validPayload(scope, data.payload)) {
      return { draft: null, status: "invalid" };
    }
    return { draft: data.payload as T, status: "ready" };
  } catch {
    // Neither the stored text nor the error is logged.
    return { draft: null, status: "unavailable" };
  }
}

export function writeTabDraft(
  scope: DraftScope,
  lease: DraftLease,
  payload: QuickDraft | FeedbackDraft,
  storage: DraftStorage | null = browserStorage(),
): DraftWriteStatus {
  if (!draftLeaseActive(lease) || !scopeMatchesLease(scope, lease)) return "revoked";
  if (!storage || !validPayload(scope, payload)) return "unavailable";
  const raw = JSON.stringify({ version: 1, scope, payload });
  if (raw.length * 2 > MAX_DRAFT_BYTES) return "too_large";
  try {
    const key = draftStorageKey(scope);
    storage.setItem(key, raw);
    freshKeys.add(key);
    return "saved";
  } catch {
    return "unavailable";
  }
}

export function removeTabDraft(
  scope: DraftScope,
  lease: DraftLease,
  storage: DraftStorage | null = browserStorage(),
): DraftWriteStatus {
  if (!draftLeaseActive(lease) || !scopeMatchesLease(scope, lease)) return "revoked";
  if (!storage) return "unavailable";
  try {
    const key = draftStorageKey(scope);
    storage.removeItem(key);
    freshKeys.delete(key);
    return "empty";
  } catch {
    return "unavailable";
  }
}

/** Invalidate first: late request completions/unmounts cannot re-create cleared drafts. */
export function discardDraftSession(owner: DraftOwner, storage: DraftStorage | null = browserStorage()): boolean {
  suspendDraftSession();
  ignoredOwners.add(ownerKey(owner));
  // Forget write provenance before attempting deletion: removeItem may fail midway.
  for (const key of [...freshKeys]) {
    try {
      const parts: unknown = JSON.parse(key.slice(DRAFT_STORAGE_PREFIX.length));
      if (Array.isArray(parts) && parts[0] === owner.userId && parts[1] === owner.workspaceId) freshKeys.delete(key);
    } catch { /* No matching owner provenance. */ }
  }
  if (!storage) return false;
  try {
    const keys: string[] = [];
    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index);
      if (!key?.startsWith(DRAFT_STORAGE_PREFIX)) continue;
      try {
        const parts: unknown = JSON.parse(key.slice(DRAFT_STORAGE_PREFIX.length));
        if (Array.isArray(parts) && parts[0] === owner.userId && parts[1] === owner.workspaceId) keys.push(key);
      } catch { /* Unrelated/malformed keys are never interpreted as this account's drafts. */ }
    }
    keys.forEach((key) => { storage.removeItem(key); freshKeys.delete(key); });
    return true;
  } catch {
    return false;
  }
}

/** A cache may contain old issue IDs; only the independently loaded run's issue IDs are restored. */
export function verifiedFeedbackNotes(draft: FeedbackDraft | null, issueIds: string[]): Record<string, string> {
  const allowed = new Set(issueIds);
  return Object.fromEntries(Object.entries(draft?.notes || {}).filter(([id]) => allowed.has(id)));
}
