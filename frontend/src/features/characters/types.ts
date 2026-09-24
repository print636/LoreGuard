export const characterSections = ["profile", "candidates", "drift"] as const;

export type CharacterSection = (typeof characterSections)[number];
export type ModelCoverage = "full" | "partial" | "rules_only" | "unknown";
export type CharacterReadiness =
  | "ready"
  | "no_documents"
  | "no_completed_run"
  | "not_generated";

export type Page<T> = {
  items: T[];
  page: number;
  page_size: number;
  total: number;
  has_more: boolean;
};

export type NarrativeScopeRef = {
  scope_id: string;
  label: string;
  version_id: string | null;
  activity_id: string | null;
  branch_path: string[];
};

export type CharacterSummary = {
  id: string;
  canonical_name: string;
  aliases: string[];
  applicable_scopes: NarrativeScopeRef[];
  confirmed_item_count: number;
  pending_candidate_count: number;
  drift_issue_count: number | null;
  profile_revision: number | null;
  updated_at: string;
};

export type CharacterDimension =
  | "core_personality"
  | "preference"
  | "value"
  | "speech_pattern"
  | "behavior_boundary"
  | "contextual_behavior"
  | "current_state"
  | "unknown";

export type CharacterProfileItem = {
  id: string;
  dimension: CharacterDimension;
  statement: string;
  origin: "explicit_profile" | "confirmed_inference";
  approved_axis_id: string | null;
  scopes: NarrativeScopeRef[];
  evidence_count: number;
  confirmed_at: string | null;
};

export type ProfileEvidence = {
  document_id: string;
  document_name: string;
  document_version: number | null;
  document_role: string;
  publication_status: string;
  authority_level: string;
  story_scope: NarrativeScopeRef;
  line_start: number;
  line_end: number;
  text: string;
};

export type ProfileCandidateStatus =
  | "pending"
  | "confirmed"
  | "rejected"
  | "stale";

export type ProfileCandidate = {
  id: string;
  character_id: string;
  dimension: CharacterDimension;
  origin: "explicit_setting" | "history_inference" | "unknown";
  contexts: string[];
  statement: string;
  model_trait_key: string | null;
  polarity: "positive" | "negative" | "neutral" | "unclear" | null;
  comparison_key: string | null;
  authority_tier: "core_canon" | "formal_record" | null;
  valid_from_release_ordinal: number | null;
  valid_until_release_ordinal: number | null;
  approved_axis_id: string | null;
  approved_axis_version: number | null;
  confidence: number;
  rationale: string;
  limitations: string[];
  scopes: NarrativeScopeRef[];
  supporting_evidence: ProfileEvidence[];
  contrary_evidence: ProfileEvidence[];
  status: ProfileCandidateStatus;
  reviewable: boolean;
  unreviewable_reason: string | null;
  source_run_id: string;
  source_snapshot_revision: string;
  model_coverage: ModelCoverage;
  revision: number;
};

export type CharacterDetail = {
  character: CharacterSummary;
  profile_items: CharacterProfileItem[];
  source_run_id: string | null;
  source_snapshot_revision: string | null;
  model_coverage: ModelCoverage;
  coverage_detail: string | null;
};

export type CharacterPage = Page<CharacterSummary> & {
  readiness: CharacterReadiness;
  model_coverage: ModelCoverage;
  coverage_detail: string | null;
  source_run_id: string | null;
};

export type CandidatePage = Page<ProfileCandidate> & {
  model_coverage: ModelCoverage;
  coverage_detail: string | null;
};

export type CandidateDecision = "confirm" | "reject";

export type CandidateDecisionIn = {
  decision: CandidateDecision;
  comment: string;
  expected_revision: number;
  approved_axis_id?: string;
  expected_axis_version?: number;
};

export type CharacterTraitAxis = {
  id: string;
  project_id: string;
  trait_type: "core_personality";
  version: number;
  display_name: string;
  definition: string;
  definition_sha256: string;
  created_at: string | null;
};

export type CharacterTraitAxisPage = {
  items: CharacterTraitAxis[];
  total: number;
  limit: number;
  offset: number;
};

export type CandidateDecisionOut = {
  candidate: ProfileCandidate;
  created_profile_item?: CharacterProfileItem | null;
  profile_revision?: number;
  decision_id?: string;
  deduplicated?: boolean;
};

export type DriftIssueSummary = {
  issue_id: string;
  run_id: string;
  character_id: string;
  title: string;
  severity: string;
  confidence: number;
  explanation: string;
  scope: NarrativeScopeRef;
  baseline_profile_item_ids: string[];
  evidence_preview: ProfileEvidence[];
  feedback_status: "unreviewed" | "accepted" | "false_positive" | "resolved";
};

export type DriftIssuePage = Page<DriftIssueSummary> & {
  model_coverage: ModelCoverage;
  coverage_detail: string | null;
};
