from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from .character_drift import (
    CharacterConsistencyReviewer,
    CharacterDriftCase,
    CharacterReviewResult,
    ConfirmedTraitSnapshot,
    PreparedCharacterDrift,
    SupportEvidence,
    _evidence_rows,
    prepare_character_drift,
    promote_character_drift,
)
from .character_trait_extraction import (
    MAX_CHARACTER_SIGNAL_BASELINE_HINT_CHARS,
    MAX_CHARACTER_SIGNAL_SERVER_CONTEXT_CHARS,
    MAX_TARGETED_CHARACTER_SIGNAL_CANDIDATE_LINES,
    CharacterSignal,
    CharacterSignalChunk,
    CharacterSignalExtractor,
    CharacterSignalTarget,
    PendingTraitCandidate,
    build_pending_trait_candidates,
    preference_modifier_bridge,
    safe_pronoun_evidence_range,
    stable_trait_identity,
    trait_keys_compatible,
)
from .character_traits import (
    TraitCandidateInput,
    TraitEvidenceInput,
    normalize_character_key,
    upsert_character_trait_candidate,
)
from .chunking import chunk_character_profile_document, chunk_document
from .config import Settings, get_settings
from .db import (
    AnalysisRunCharacterTraitInputRow,
    AnalysisRunInputRow,
)
from .domain import ConsistencyIssue, EvidenceSpan, IssueCategory, Severity
from .narrative_context import NarrativeScopeV1, payload_sha256, scope_relation
from .pipeline import DocumentInput


CHARACTER_CONSISTENCY_CHECKER_VERSION = "character-consistency-stage-v1"
_SUGGESTION = "请核对是否存在尚未记录的成长、伪装或情境依据"
_BRIDGE_PATTERN = re.compile(
    r"成长|训练|逐渐|渐渐|变得|学会|克服|改变|转变|经历.{0,24}(?:后|之后)|"
    r"受到.{0,24}影响"
)
_EXCEPTION_PATTERN = re.compile(
    r"伪装|假装|佯装|撒谎|说谎|演戏|潜伏|梦境|梦中|幻觉|"
    r"被操控|受到控制"
)
_NEGATED_OR_UNCERTAIN = re.compile(
    r"没有|并未|未曾|从未|不存在|不是|并非|毫无|无任何|没提到|未提到|"
    r"没有提到|未说明|可能|也许|或许|疑似|似乎|假如|如果|是否|"
    r"不能|不得|不会|不再|不曾|无法|难以|禁止|否认|拒绝|没|"
    r"不(?:会|能|再|曾|是|愿|想|肯|要|可|应|得)?(?:进行|发生|存在|选择)?"
    r"(?:伪装|假装|佯装|撒谎|说谎|演戏|成长|训练|改变|转变|变得|学会)"
)
_CONDITIONAL_OR_RULE = re.compile(
    r"除非|倘若|假设|前提是|仅当|仅在|只有.{0,16}才|只在|"
    r"只要|一旦|(?:^|[，,。；;：:\s])若|"
    r"若(?:在|有|被|遇|需|要|能|可|将|启动|完成|接受)|"
    r"规则(?:规定|说明|要求)?|按照规则|依据规则|"
    r"规定中|条款|守则|可能|也许|或许|疑似|似乎|如果|假如"
)
_TEMPORARY_CHARACTER_STATE_OR_BEHAVIOR = re.compile(
    r"(?:临时|暂时)(?:地)?(?:"
    r"表现|显得|变得|处于|伪装|假装|装作|扮作|扮演|"
    r"担任|充当|保持|改变|改用|隐藏|隐瞒|回避|沉默|失语|"
    r"失忆|昏迷|失控|受伤|生病|离开|停留|拒绝|接受|喜欢|"
    r"讨厌|信任|敌对|合作|服从)"
)
_MAX_CONFIRMED_TRAITS_IN_SERVER_CONTEXT = 12
_MAX_CASE_TRACE_OBSERVATION_REFS = 12
_MAX_CASE_TRACE_CITATION_REFS = 8
_MAX_CASE_TRACE_LINE = 10_000_000
_CASE_TRACE_CITATION_HANDLE = re.compile(r"^[BCGX][0-9]{2}$")
_MAX_ACCEPTED_DRAFT_OBSERVATION_REFS = 64
_MAX_TOKEN_ADMISSION_EVENTS = 24
_OBJECT_BEARING_TRAIT_DIMENSIONS = frozenset(
    {"preference", "value", "behavior_boundary", "current_state"}
)
_CONTEXT_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_CONTEXT_SECRET_OR_URL = re.compile(
    r"(?:[a-z][a-z0-9+.-]{1,15}://|www\.)|"
    r"(?:api[_\s-]?key|base[_\s-]?url|authorization|bearer\s+|password|"
    r"credential|private[_\s-]?key|(?:^|[^a-z0-9])sk-[a-z0-9]{8,}|\u5bc6\u94a5|\u5bc6\u7801)",
    re.IGNORECASE,
)


class _ChatProvider(Protocol):
    def complete(self, system: str, user: str): ...


class CharacterConsistencyStageResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    issues: tuple[ConsistencyIssue, ...] = ()
    diagnostics: dict[str, Any]
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    charged_tokens: int = Field(default=0, ge=0)
    model_used: bool = False

    def usage_accounting(self, *, terminal_status: str) -> dict[str, Any] | None:
        logical_calls = self.diagnostics.get("usage", {}).get("attempted_calls")
        if type(logical_calls) is not int or logical_calls <= 0:
            return None
        return {
            "completeness": "completed_calls",
            "scope": "character_consistency",
            "terminal_status": terminal_status,
            "logical_calls": logical_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "charged_tokens": max(
                self.charged_tokens,
                self.prompt_tokens + self.completion_tokens,
            ),
            "charged_token_semantics": "heuristic_or_reported_internal_debit",
            "provider_calls": None,
        }


@dataclass(frozen=True, slots=True)
class _FrozenDocument:
    input_id: str
    document: DocumentInput
    document_version: int
    content_sha256: str
    ordinal: int
    source_kind: str | None
    source_reason: str
    scope: NarrativeScopeV1 | None
    resolution_state: str
    publication_status: str
    authority_tier: str


BaselineEntry = tuple[
    AnalysisRunCharacterTraitInputRow,
    ConfirmedTraitSnapshot,
    NarrativeScopeV1,
    str,
]


@dataclass(slots=True)
class _Usage:
    attempted_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    charged_tokens: int = 0

    def add(self, diagnostics: object) -> None:
        attempted = getattr(diagnostics, "attempted_calls", 0)
        prompt = getattr(diagnostics, "prompt_tokens", 0)
        completion = getattr(diagnostics, "completion_tokens", 0)
        charged = getattr(diagnostics, "charged_tokens", 0)
        self.attempted_calls += attempted if type(attempted) is int else 0
        self.prompt_tokens += prompt if type(prompt) is int else 0
        self.completion_tokens += completion if type(completion) is int else 0
        self.charged_tokens += charged if type(charged) is int else 0

    def safe_dict(self) -> dict[str, int]:
        return {
            "attempted_calls": self.attempted_calls,
            "input_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "charged_tokens": self.charged_tokens,
        }


@dataclass(frozen=True, slots=True)
class _SafeServerContext:
    payload: str
    eligible_traits: int = 0
    included_traits: int = 0
    ambiguous_traits: int = 0
    targets: tuple[CharacterSignalTarget, ...] = ()

    @property
    def truncated(self) -> bool:
        return self.included_traits + self.ambiguous_traits < self.eligible_traits

    @property
    def omitted_traits(self) -> int:
        return max(
            0,
            self.eligible_traits - self.included_traits - self.ambiguous_traits,
        )


class CharacterConsistencyStage:
    """Optional, fail-closed character signal and drift stage.

    The caller supplies the already verified immutable document projection.
    This class re-binds it to the corresponding ``AnalysisRunInputRow`` rows
    and reads only the run's frozen confirmed-trait payloads.  It never reads a
    live ``DocumentRow`` or a live character candidate.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        provider: _ChatProvider | None = None,
        checkpoint: Callable[[], None] | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider
        self.checkpoint = checkpoint or (lambda: None)

    def run(
        self,
        db,
        *,
        run_id: str,
        project_id: str,
        documents: list[DocumentInput],
        metadata: list[dict[str, Any]],
        remaining_run_tokens: int,
    ) -> CharacterConsistencyStageResult:
        settings = self.settings
        if not settings.enable_character_consistency:
            return _empty_stage_result("disabled", "feature_disabled")
        if type(remaining_run_tokens) is not int or remaining_run_tokens < 0:
            return _empty_stage_result("degraded", "invalid_remaining_budget")

        stage_budget = min(
            settings.character_consistency_stage_token_budget,
            remaining_run_tokens,
        )
        if stage_budget < 256:
            return _empty_stage_result("skipped", "run_token_budget")

        usage = _Usage()
        reason_counts: Counter[str] = Counter()
        frozen = self._bind_frozen_documents(
            db,
            run_id=run_id,
            documents=documents,
            metadata=metadata,
            reasons=reason_counts,
        )
        eligible = [row for row in frozen if row.source_kind is not None]
        if not eligible:
            return CharacterConsistencyStageResult(
                diagnostics=_diagnostics(
                    outcome="skipped",
                    reason_code="no_eligible_frozen_documents",
                    usage=usage,
                    reasons=reason_counts,
                    source_total=len(frozen),
                    source_eligible=0,
                )
            )

        # Load only the run's immutable confirmed-trait snapshots before model
        # extraction.  This lets draft extraction reuse an exact comparison key
        # without consulting mutable candidate rows or exposing evidence text.
        baselines = self._load_confirmed_traits(
            db,
            run_id=run_id,
            project_id=project_id,
            reasons=reason_counts,
        )
        baselines, shadowed_baselines = _select_authoritative_baselines(baselines)
        if shadowed_baselines:
            reason_counts["lower_authority_baseline_shadowed"] += shadowed_baselines

        planned_chunks: list[tuple[_FrozenDocument, object]] = []
        for source in sorted(eligible, key=lambda row: row.ordinal):
            if (
                source.source_kind == "formal_character_profile"
                and source.document.role == "character_profile"
            ):
                chunks = chunk_character_profile_document(
                    source.document, settings.character_signal_max_chunk_chars
                )
            else:
                chunks = chunk_document(
                    source.document,
                    settings.character_signal_max_chunk_chars,
                    overlap_lines=0,
                )
            for chunk in chunks:
                planned_chunks.append((source, chunk))
        partial = (
            len(planned_chunks) > settings.character_consistency_max_chunks_per_run
            or bool(reason_counts["invalid_confirmed_trait_snapshot"])
        )
        chunk_cap = settings.character_consistency_max_chunks_per_run
        if partial and any(source.source_kind == "draft" for source, _ in planned_chunks):
            # Reserve the first chunk of each draft before filling remaining
            # slots in source order. Process those drafts first so an overlong
            # profile cannot consume the shared stage budget before review.
            reserved: list[int] = []
            seen_drafts: set[str] = set()
            for index, (source, _) in enumerate(planned_chunks):
                if (
                    source.source_kind != "draft"
                    or source.document.id in seen_drafts
                ):
                    continue
                reserved.append(index)
                seen_drafts.add(source.document.id)
                if len(reserved) == chunk_cap:
                    break
            reserved_set = set(reserved)
            fill = [
                index
                for index in range(len(planned_chunks))
                if index not in reserved_set
            ][: chunk_cap - len(reserved)]
            selected_chunks = [planned_chunks[index] for index in (*reserved, *fill)]
        else:
            selected_chunks = planned_chunks[:chunk_cap]
        if partial:
            reason_counts["chunk_limit"] += (
                len(planned_chunks) - len(selected_chunks)
            )

        server_contexts: dict[str, _SafeServerContext] = {}
        context_eligible_traits = 0
        context_included_traits = 0
        ambiguous_hint_traits = 0
        context_truncated_documents = 0
        for source, _ in selected_chunks:
            if source.document.id in server_contexts:
                continue
            context = _safe_server_context(source, baselines=baselines)
            server_contexts[source.document.id] = context
            context_eligible_traits += context.eligible_traits
            context_included_traits += context.included_traits
            ambiguous_hint_traits += context.ambiguous_traits
            if context.ambiguous_traits:
                partial = True
                reason_counts["confirmed_trait_hint_ambiguous"] += (
                    context.ambiguous_traits
                )
            if context.truncated:
                partial = True
                context_truncated_documents += 1
                reason_counts["confirmed_trait_context_truncated"] += (
                    context.omitted_traits
                )

        all_signals: dict[str, CharacterSignal] = {}
        # A model record never carries an approved-axis ID. Only a clean,
        # validated one-target pass can bind its evidence to a frozen axis.
        axis_bindings_by_signal: dict[str, set[tuple[str, int, str]]] = defaultdict(set)
        axis_bindings_by_line: dict[
            tuple[str, str, int], set[tuple[str, int, str]]
        ] = defaultdict(set)

        def bind_approved_axis(
            signal: CharacterSignal, target: CharacterSignalTarget
        ) -> None:
            axis_key = target.approved_axis_identity
            if axis_key is None:
                return
            axis_bindings_by_signal[signal.id].add(axis_key)
            actor = _key(signal.character)
            for line in range(
                signal.evidence.line_start, signal.evidence.line_end + 1
            ):
                axis_bindings_by_line[
                    (actor, signal.evidence.document_id, line)
                ].add(axis_key)

        def bound_observations(
            target: CharacterSignalTarget,
            observations: tuple[CharacterSignal, ...],
        ) -> tuple[CharacterSignal, ...]:
            axis_key = target.approved_axis_identity
            if axis_key is None:
                return observations
            return tuple(
                row for row in observations
                if _trusted_axis_binding(
                    row,
                    axis_bindings_by_signal=axis_bindings_by_signal,
                    axis_bindings_by_line=axis_bindings_by_line,
                ) == axis_key
            )
        selected_draft_chunk_count = sum(
            source.source_kind == "draft" for source, _ in selected_chunks
        )
        processed_chunks = 0
        successful_model_calls = 0
        signal_ignored_duplicates = 0
        targeted_eligible_targets = 0
        targeted_selected_targets = 0
        targeted_truncated_targets = 0
        targeted_passes_scheduled = 0
        targeted_passes_attempted = 0
        targeted_passes_completed = 0
        targeted_empty_passes = 0
        targeted_records_accepted = 0
        targeted_records_rejected = 0
        targeted_records_ignored_duplicates = 0
        targeted_signals_added = 0
        targeted_budget_exhausted_targets = 0
        targeted_fully_processed_draft_chunks = 0
        targeted_verification_first_empty_targets = 0
        targeted_verification_eligible_targets = 0
        targeted_verification_no_candidate_targets = 0
        targeted_verification_candidate_lines = 0
        targeted_verification_candidate_lines_truncated = 0
        targeted_verification_scheduled = 0
        targeted_verification_attempted = 0
        targeted_verification_completed = 0
        targeted_verification_empty = 0
        targeted_verification_signals_added = 0
        targeted_verification_budget_exhausted = 0
        targeted_reviewer_reserve_tokens = min(
            settings.character_drift_token_budget, stage_budget
        )
        token_admission_events: list[dict[str, int | str | None]] = []
        token_admission_omitted = 0

        def record_token_admission(
            phase: str,
            extraction: object,
            *,
            chunk_ordinal: int,
            target_ordinal: int | None = None,
            stage_remaining_before: int,
            reviewer_reserve: int = 0,
        ) -> None:
            nonlocal token_admission_omitted
            diagnostics = getattr(extraction, "diagnostics", None)
            admission = getattr(diagnostics, "token_admission", None)
            if admission is None:
                return
            if len(token_admission_events) >= _MAX_TOKEN_ADMISSION_EVENTS:
                token_admission_omitted += 1
                return
            token_admission_events.append(
                {
                    "stage_phase": phase,
                    "signal_phase": admission.phase,
                    "chunk_ordinal": chunk_ordinal,
                    "target_ordinal": target_ordinal,
                    "estimated_tokens": admission.estimated_tokens,
                    "available_tokens": admission.available_tokens,
                    "stage_remaining_before": stage_remaining_before,
                    "reviewer_reserve_tokens": reviewer_reserve,
                    # Logical package generations, not transport retries.
                    "model_calls_before_failure": diagnostics.attempted_calls,
                }
            )

        for chunk_ordinal, (source, chunk) in enumerate(selected_chunks, start=1):
            self.checkpoint()
            remaining = stage_budget - usage.charged_tokens
            if remaining < 256:
                partial = True
                reason_counts["stage_token_budget"] += 1
                break
            call_settings = settings.model_copy(
                update={
                    "character_signal_token_budget": min(
                        settings.character_signal_token_budget, remaining
                    )
                }
            )
            extraction = CharacterSignalExtractor(
                self.provider, settings=call_settings
            ).extract(
                CharacterSignalChunk(
                    document_id=source.document.id,
                    document_name=source.document.name,
                    content=chunk.content,
                    global_line_start=chunk.global_line_start,
                    source_kind=source.source_kind,  # type: ignore[arg-type]
                    server_context=server_contexts[source.document.id].payload,
                )
            )
            processed_chunks += 1
            record_token_admission(
                "primary_extraction",
                extraction,
                chunk_ordinal=chunk_ordinal,
                stage_remaining_before=remaining,
            )
            usage.add(extraction.diagnostics)
            signal_ignored_duplicates += (
                extraction.diagnostics.ignored_duplicate_records
            )
            reason_counts.update(extraction.diagnostics.reason_counts)
            if extraction.diagnostics.outcome in {"completed", "partial"}:
                successful_model_calls += 1
            if extraction.diagnostics.outcome != "completed":
                # Keeping valid records from a mixed response is useful, but
                # rejected or malformed records mean the chunk was not fully
                # processed and can never support a clean coverage claim.
                partial = True
            for signal in extraction.signals:
                all_signals.setdefault(signal.id, signal)

            # Recall is isolated to one confirmed trait per model call. This
            # prevents an empty or malformed answer for one semantic axis from
            # suppressing another target in the same draft chunk. Every call
            # retains the same polarity/evidence binder and shared stage cap.
            if source.source_kind != "draft":
                continue
            undercovered: list[CharacterSignalTarget] = []
            for target in server_contexts[source.document.id].targets:
                if not _character_appears_in_chunk(target.character, chunk.content):
                    continue
                if _target_has_sufficient_recall_evidence(
                    target,
                    bound_observations(target, extraction.draft_observations),
                ):
                    continue
                undercovered.append(
                    _target_with_existing_evidence_ranges(
                        target,
                        bound_observations(target, extraction.draft_observations),
                    )
                )
            targeted_eligible_targets += len(undercovered)
            target_limit = settings.character_signal_targeted_max_targets_per_chunk
            selected_targets = tuple(undercovered[:target_limit])
            targeted_selected_targets += len(selected_targets)
            if len(undercovered) > len(selected_targets):
                omitted = len(undercovered) - len(selected_targets)
                targeted_truncated_targets += omitted
                reason_counts["targeted_target_limit"] += omitted
                partial = True
            if not selected_targets:
                continue

            targeted_passes_scheduled += len(selected_targets)
            chunk_targets_complete = True
            initial_round_finished = True
            first_empty_targets: list[CharacterSignalTarget] = []
            completed_targets: list[tuple[int, CharacterSignalTarget]] = []
            chunk_observations = list(extraction.draft_observations)
            targeted_chunk = CharacterSignalChunk(
                document_id=source.document.id,
                document_name=source.document.name,
                content=chunk.content,
                global_line_start=chunk.global_line_start,
                source_kind="draft",
            )
            for target_index, target in enumerate(selected_targets):
                self.checkpoint()
                remaining = stage_budget - usage.charged_tokens
                # Targeted recall is secondary to adjudicating observations
                # already found. Preserve one full reviewer call across every
                # focused target so recall cannot consume the last review.
                targeted_budget = remaining - targeted_reviewer_reserve_tokens
                if targeted_budget < 256:
                    exhausted = len(selected_targets) - target_index
                    reason_counts["targeted_reviewer_budget_reserve"] += exhausted
                    targeted_budget_exhausted_targets += exhausted
                    partial = True
                    chunk_targets_complete = False
                    initial_round_finished = False
                    break
                targeted_settings = settings.model_copy(
                    update={
                        "character_signal_token_budget": min(
                            settings.character_signal_token_budget,
                            targeted_budget,
                        ),
                        "character_signal_max_completion_tokens": (
                            _targeted_completion_reserve(
                                settings, targeted_chunk
                            )
                        ),
                    }
                )
                targeted = CharacterSignalExtractor(
                    self.provider, settings=targeted_settings
                ).extract_targeted(targeted_chunk, (target,))
                record_token_admission(
                    "targeted_recall",
                    targeted,
                    chunk_ordinal=chunk_ordinal,
                    target_ordinal=target_index + 1,
                    stage_remaining_before=remaining,
                    reviewer_reserve=targeted_reviewer_reserve_tokens,
                )
                usage.add(targeted.diagnostics)
                targeted_passes_attempted += targeted.diagnostics.attempted_calls
                targeted_records_accepted += targeted.diagnostics.accepted_records
                targeted_records_rejected += targeted.diagnostics.rejected_records
                targeted_records_ignored_duplicates += (
                    targeted.diagnostics.ignored_duplicate_records
                )
                for reason, count in targeted.diagnostics.reason_counts.items():
                    reason_counts[f"targeted_pass_{reason}"] += count
                if targeted.diagnostics.outcome == "completed":
                    targeted_passes_completed += 1
                    successful_model_calls += 1
                    completed_targets.append((target_index, target))
                    if not targeted.signals:
                        targeted_empty_passes += 1
                        first_empty_targets.append(target)
                else:
                    # A partial response is also incomplete: accepted records
                    # may be retained, but coverage remains explicitly partial.
                    chunk_targets_complete = False
                    if targeted.diagnostics.outcome == "partial":
                        successful_model_calls += 1
                    partial = True
                for signal in targeted.signals:
                    if targeted.diagnostics.outcome == "completed":
                        bind_approved_axis(signal, target)
                    if all(existing.id != signal.id for existing in chunk_observations):
                        chunk_observations.append(signal)
                    if signal.id not in all_signals:
                        all_signals[signal.id] = signal
                        targeted_signals_added += 1

            # Only after every selected target has received its initial focused
            # pass may a still-undercovered target receive one candidate-line
            # verification. This includes a clean non-empty pass whose sole
            # result is still only one concrete behavior. The second pass
            # excludes every already-bound span, so it cannot turn repetition
            # into artificial coverage. This ordering also prevents an early
            # target from consuming a later target's first-pass budget.
            verification_queue: list[
                tuple[
                    int,
                    CharacterSignalTarget,
                    tuple[tuple[int, int], ...],
                    int,
                ]
            ] = []
            targeted_verification_first_empty_targets += len(first_empty_targets)
            if initial_round_finished:
                for target_index, target in completed_targets:
                    target = _target_with_existing_evidence_ranges(
                        target,
                        bound_observations(target, tuple(chunk_observations)),
                    )
                    if _target_has_sufficient_recall_evidence(
                        target,
                        bound_observations(target, tuple(chunk_observations)),
                    ):
                        continue
                    candidate_ranges, truncated_lines = (
                        _target_candidate_line_ranges(
                            targeted_chunk,
                            target,
                        )
                    )
                    if not candidate_ranges:
                        targeted_verification_no_candidate_targets += 1
                        continue
                    targeted_verification_eligible_targets += 1
                    targeted_verification_candidate_lines += sum(
                        end - start + 1 for start, end in candidate_ranges
                    )
                    if truncated_lines:
                        targeted_verification_candidate_lines_truncated += (
                            truncated_lines
                        )
                        reason_counts[
                            "targeted_verification_candidate_line_limit"
                        ] += truncated_lines
                        chunk_targets_complete = False
                        partial = True
                    verification_queue.append(
                        (target_index, target, candidate_ranges, truncated_lines)
                    )

            targeted_verification_scheduled += len(verification_queue)
            targeted_passes_scheduled += len(verification_queue)
            for verification_index, (
                target_index,
                target,
                candidate_ranges,
                _,
            ) in enumerate(verification_queue):
                self.checkpoint()
                remaining = stage_budget - usage.charged_tokens
                verification_budget = (
                    remaining - targeted_reviewer_reserve_tokens
                )
                if verification_budget < 256:
                    exhausted = len(verification_queue) - verification_index
                    reason_counts[
                        "targeted_verification_reviewer_budget_reserve"
                    ] += exhausted
                    targeted_budget_exhausted_targets += exhausted
                    targeted_verification_budget_exhausted += exhausted
                    partial = True
                    chunk_targets_complete = False
                    break
                verification_settings = settings.model_copy(
                    update={
                        "character_signal_token_budget": min(
                            settings.character_signal_token_budget,
                            verification_budget,
                        ),
                        "character_signal_max_completion_tokens": (
                            _targeted_completion_reserve(
                                settings,
                                targeted_chunk,
                                candidate_ranges=candidate_ranges,
                            )
                        ),
                    }
                )
                verification = CharacterSignalExtractor(
                    self.provider, settings=verification_settings
                ).extract_targeted(
                    targeted_chunk,
                    (target,),
                    candidate_evidence_ranges=candidate_ranges,
                )
                record_token_admission(
                    "targeted_verification",
                    verification,
                    chunk_ordinal=chunk_ordinal,
                    target_ordinal=target_index + 1,
                    stage_remaining_before=remaining,
                    reviewer_reserve=targeted_reviewer_reserve_tokens,
                )
                usage.add(verification.diagnostics)
                targeted_passes_attempted += verification.diagnostics.attempted_calls
                targeted_verification_attempted += (
                    verification.diagnostics.attempted_calls
                )
                targeted_records_accepted += (
                    verification.diagnostics.accepted_records
                )
                targeted_records_rejected += (
                    verification.diagnostics.rejected_records
                )
                targeted_records_ignored_duplicates += (
                    verification.diagnostics.ignored_duplicate_records
                )
                for reason, count in verification.diagnostics.reason_counts.items():
                    reason_counts[f"targeted_verification_{reason}"] += count
                if verification.diagnostics.outcome == "completed":
                    targeted_passes_completed += 1
                    targeted_verification_completed += 1
                    successful_model_calls += 1
                    if not verification.signals:
                        targeted_empty_passes += 1
                        targeted_verification_empty += 1
                else:
                    chunk_targets_complete = False
                    if verification.diagnostics.outcome == "partial":
                        successful_model_calls += 1
                    partial = True
                for signal in verification.signals:
                    if verification.diagnostics.outcome == "completed":
                        bind_approved_axis(signal, target)
                    if all(existing.id != signal.id for existing in chunk_observations):
                        chunk_observations.append(signal)
                    if signal.id not in all_signals:
                        all_signals[signal.id] = signal
                        targeted_signals_added += 1
                        targeted_verification_signals_added += 1
            if chunk_targets_complete:
                targeted_fully_processed_draft_chunks += 1

        signals = tuple(all_signals.values())
        ambiguous_axis_evidence = sum(
            len(keys) > 1 for keys in axis_bindings_by_line.values()
        )
        if ambiguous_axis_evidence:
            partial = True
            reason_counts["approved_axis_ambiguous_evidence"] += (
                ambiguous_axis_evidence
            )
        pending = list(build_pending_trait_candidates(signals))
        if len(pending) > settings.character_consistency_max_candidates_per_run:
            reason_counts["candidate_limit"] += len(pending) - (
                settings.character_consistency_max_candidates_per_run
            )
            pending = pending[: settings.character_consistency_max_candidates_per_run]
            partial = True

        frozen_by_document = {row.document.id: row for row in frozen}
        persisted_created = 0
        persisted_reused = 0
        persisted_failed = 0
        persisted_scope_skipped = 0
        for candidate in pending:
            self.checkpoint()
            try:
                mapped = _trait_candidate_input(
                    candidate,
                    frozen_by_document=frozen_by_document,
                )
                if mapped is None:
                    persisted_scope_skipped += 1
                    reason_counts["candidate_scope"] += 1
                    continue
                with db.begin_nested():
                    _, created = upsert_character_trait_candidate(
                        db,
                        project_id=project_id,
                        source_run_id=run_id,
                        candidate=mapped,
                        allow_running_source=True,
                    )
                if created:
                    persisted_created += 1
                else:
                    persisted_reused += 1
            except Exception:
                persisted_failed += 1
                reason_counts["candidate_persistence"] += 1
                partial = True

        draft_signals = tuple(
            row for row in signals if row.source_kind == "draft"
        )
        (
            accepted_draft_observation_refs,
            accepted_draft_observation_total,
            accepted_draft_observation_refs_truncated,
        ) = _safe_accepted_draft_observation_refs(draft_signals)
        eligible_draft_count = sum(row.source_kind == "draft" for row in eligible)
        if eligible_draft_count and not draft_signals:
            # A completed targeted pass is allowed to say that the named
            # character has no supported behavior on the requested axes.  It
            # leaves each baseline trace unverifiable, but the processing
            # itself is complete.  Without such a pass, an entirely empty
            # primary extraction remains an explicit coverage gap.
            targeted_empty_was_complete = (
                selected_draft_chunk_count > 0
                and targeted_passes_completed == targeted_passes_scheduled
                and targeted_empty_passes == targeted_passes_completed
                and targeted_fully_processed_draft_chunks
                == selected_draft_chunk_count
            )
            if targeted_empty_was_complete:
                reason_counts["targeted_no_supported_observation"] += 1
            else:
                partial = True
                reason_counts["no_draft_signals"] += eligible_draft_count
        if draft_signals and not baselines:
            partial = True
            reason_counts["no_confirmed_character_traits"] += 1
        alias_map = _unique_alias_map(baselines)
        resolved_drafts: dict[str, list[CharacterSignal]] = defaultdict(list)
        ambiguous_aliases = 0
        for observation in draft_signals:
            alias = _key(observation.character)
            character_keys = alias_map.get(alias, ())
            if len(character_keys) != 1:
                ambiguous_aliases += 1
                reason_counts[
                    "ambiguous_character_alias"
                    if character_keys
                    else "unmatched_character_alias"
                ] += 1
                continue
            resolved_drafts[character_keys[0]].append(observation)
        if ambiguous_aliases:
            partial = True

        issues: list[ConsistencyIssue] = []
        drift_considered = 0
        drift_reviewed = 0
        drift_unverifiable = 0
        drift_no_issue = 0
        drift_scope_skipped = 0
        case_trace: list[dict[str, Any]] = []
        for baseline_row, baseline, baseline_scope, character_key in baselines[
            : settings.character_consistency_max_candidates_per_run
        ]:
            self.checkpoint()
            baseline_entry = (
                baseline_row, baseline, baseline_scope, character_key
            )
            matches: list[CharacterSignal] = []
            draft_scopes: list[NarrativeScopeV1] = []
            for observation in resolved_drafts.get(character_key, ()):
                axis_match = _observation_matches_baseline(
                    baseline_entry,
                    observation,
                    approved_axis_binding=_trusted_axis_binding(
                        observation,
                        axis_bindings_by_signal=axis_bindings_by_signal,
                        axis_bindings_by_line=axis_bindings_by_line,
                    ),
                )
                if axis_match is None:
                    partial = True
                    reason_counts["object_baseline_identity_unavailable"] += 1
                    continue
                if not axis_match:
                    continue
                source = frozen_by_document.get(observation.evidence.document_id)
                if source is None or source.scope is None:
                    drift_scope_skipped += 1
                    reason_counts["drift_scope_unknown"] += 1
                    continue
                if _baseline_shadowed_at_scope(
                    baseline_entry,
                    baselines,
                    source.scope,
                    observation_context=observation.context,
                ):
                    reason_counts["lower_authority_draft_scope_shadowed"] += 1
                    continue
                release_applicability = _trait_applies_to_release(
                    baseline, source.scope
                )
                if release_applicability is None:
                    partial = True
                    reason_counts["drift_release_unknown"] += 1
                    continue
                if not release_applicability:
                    reason_counts["drift_release_inapplicable"] += 1
                    continue
                relation = scope_relation(
                    baseline_scope,
                    source.scope,
                    first_resolution="confirmed",
                    second_resolution=source.resolution_state,
                )
                if relation != "compatible":
                    drift_scope_skipped += 1
                    reason_counts[f"drift_scope_{relation}"] += 1
                    if relation == "unknown":
                        partial = True
                    continue
                matches.append(
                    observation.model_copy(
                        update={
                            "character": baseline.character,
                            **(
                                {} if baseline.approved_axis_identity is not None
                                else {"trait_key": baseline.trait_key}
                            ),
                        }
                    )
                )
                draft_scopes.append(source.scope)
            if not matches:
                case_trace.append(
                    _safe_case_trace(
                        character_key=character_key,
                        baseline=baseline,
                        baseline_entry=baseline_entry,
                        matched_observation_count=0,
                        matched_observations=(),
                        prepare_reason="no_matching_observation",
                        review=None,
                        final_outcome="unverifiable",
                        visible=False,
                        promote_reason="no_matching_observation",
                    )
                )
                continue
            if len(matches) > 24:
                reason_counts["observation_limit"] += len(matches) - 24
                matches = matches[:24]
                draft_scopes = draft_scopes[:24]
                partial = True
            support = _find_support_evidence(
                baseline=baseline,
                baseline_scope=baseline_scope,
                draft_scopes=tuple(draft_scopes),
                documents=frozen,
                limit=settings.character_drift_max_support_evidence,
            )
            case_id = "cdc_" + hashlib.sha256(
                (
                    baseline_row.candidate_id
                    + "|"
                    + "|".join(row.id for row in matches)
                ).encode("utf-8")
            ).hexdigest()[:32]
            case = CharacterDriftCase(
                id=case_id,
                baseline=baseline,
                observations=tuple(matches),
                support_evidence=support,
                scope_compatibility="compatible",
                material_coverage="partial" if partial else "complete",
                approved_axis_bound_observation_ids=(
                    tuple(row.id for row in matches)
                    if baseline.approved_axis_identity is not None else ()
                ),
            )
            prepared = prepare_character_drift(case)
            drift_considered += 1
            review: CharacterReviewResult | None = None
            if prepared.reviewer_eligible:
                remaining = stage_budget - usage.charged_tokens
                if remaining >= 256:
                    review_settings = settings.model_copy(
                        update={
                            "character_drift_token_budget": min(
                                settings.character_drift_token_budget,
                                remaining,
                            )
                        }
                    )
                    review = CharacterConsistencyReviewer(
                        self.provider, settings=review_settings
                    ).review(prepared)
                    usage.add(review.diagnostics)
                    drift_reviewed += review.diagnostics.attempted_calls
                    if review.diagnostics.outcome == "completed":
                        successful_model_calls += 1
                    elif review.diagnostics.outcome in {"degraded", "skipped"}:
                        partial = True
                        reason_counts[
                            f"review_{review.diagnostics.reason}"
                        ] += 1
                else:
                    partial = True
                    reason_counts["stage_token_budget"] += 1

            promoted = promote_character_drift(
                prepared,
                review,
                sensitivity=settings.character_consistency_sensitivity,
            )
            case_trace.append(
                _safe_case_trace(
                    character_key=character_key,
                    baseline=baseline,
                    baseline_entry=baseline_entry,
                    matched_observation_count=len(prepared.matching_observations),
                    matched_observations=prepared.matching_observations,
                    prepared=prepared,
                    prepare_reason=prepared.reason,
                    review=review,
                    final_outcome=promoted.outcome,
                    visible=promoted.visible,
                    promote_reason=promoted.reason,
                )
            )
            if promoted.outcome == "unverifiable":
                drift_unverifiable += 1
                partial = True
                reason_counts[f"drift_{promoted.reason}"] += 1
                continue
            if promoted.outcome == "no_issue":
                drift_no_issue += 1
                continue
            if not promoted.visible:
                reason_counts["below_sensitivity_threshold"] += 1
                continue
            issues.append(
                _to_issue(
                    promoted=promoted,
                    prepared=prepared,
                    confirmed_candidate_id=baseline_row.candidate_id,
                    judgement=(
                        review.decision.verdict
                        if review is not None and review.decision is not None
                        else (
                            "deterministic_conflict"
                            if prepared.deterministic_conflict
                            else promoted.outcome
                        )
                    ),
                )
            )

        if len(baselines) > settings.character_consistency_max_candidates_per_run:
            partial = True
            reason_counts["baseline_limit"] += len(baselines) - (
                settings.character_consistency_max_candidates_per_run
            )

        if usage.attempted_calls and successful_model_calls == 0:
            outcome = "degraded"
            reason_code = "model_stage_unavailable"
        else:
            outcome = "partial" if partial else "completed"
            reason_code = "bounded_partial" if partial else "completed"
        diagnostics = _diagnostics(
            outcome=outcome,
            reason_code=reason_code,
            usage=usage,
            reasons=reason_counts,
            source_total=len(frozen),
            source_eligible=len(eligible),
            planned_chunks=len(planned_chunks),
            processed_chunks=processed_chunks,
            chunk_limit=settings.character_consistency_max_chunks_per_run,
            signal_count=len(signals),
            signal_ignored_duplicate_count=signal_ignored_duplicates,
            draft_observation_count=len(draft_signals),
            pending_candidate_count=len(pending),
            persisted_created=persisted_created,
            persisted_reused=persisted_reused,
            persisted_failed=persisted_failed,
            persisted_scope_skipped=persisted_scope_skipped,
            confirmed_trait_count=len(baselines),
            context_eligible_trait_count=context_eligible_traits,
            context_included_trait_count=context_included_traits,
            ambiguous_hint_trait_count=ambiguous_hint_traits,
            context_truncated_document_count=context_truncated_documents,
            targeted_eligible_target_count=targeted_eligible_targets,
            targeted_selected_target_count=targeted_selected_targets,
            targeted_truncated_target_count=targeted_truncated_targets,
            targeted_pass_scheduled_count=targeted_passes_scheduled,
            targeted_pass_attempted_count=targeted_passes_attempted,
            targeted_pass_completed_count=targeted_passes_completed,
            targeted_empty_pass_count=targeted_empty_passes,
            targeted_record_accepted_count=targeted_records_accepted,
            targeted_record_rejected_count=targeted_records_rejected,
            targeted_record_ignored_duplicate_count=(
                targeted_records_ignored_duplicates
            ),
            targeted_signal_added_count=targeted_signals_added,
            targeted_budget_exhausted_target_count=(
                targeted_budget_exhausted_targets
            ),
            targeted_fully_processed_chunk_count=(
                targeted_fully_processed_draft_chunks
            ),
            targeted_verification_first_empty_target_count=(
                targeted_verification_first_empty_targets
            ),
            targeted_verification_eligible_target_count=(
                targeted_verification_eligible_targets
            ),
            targeted_verification_no_candidate_target_count=(
                targeted_verification_no_candidate_targets
            ),
            targeted_verification_candidate_line_count=(
                targeted_verification_candidate_lines
            ),
            targeted_verification_candidate_line_truncated_count=(
                targeted_verification_candidate_lines_truncated
            ),
            targeted_verification_scheduled_count=(
                targeted_verification_scheduled
            ),
            targeted_verification_attempted_count=(
                targeted_verification_attempted
            ),
            targeted_verification_completed_count=(
                targeted_verification_completed
            ),
            targeted_verification_empty_count=targeted_verification_empty,
            targeted_verification_signal_added_count=(
                targeted_verification_signals_added
            ),
            targeted_verification_budget_exhausted_count=(
                targeted_verification_budget_exhausted
            ),
            targeted_reviewer_reserve_tokens=targeted_reviewer_reserve_tokens,
            token_admission_omitted_count=token_admission_omitted,
            ambiguous_alias_count=ambiguous_aliases,
            drift_considered=drift_considered,
            drift_reviewed=drift_reviewed,
            drift_unverifiable=drift_unverifiable,
            drift_no_issue=drift_no_issue,
            drift_scope_skipped=drift_scope_skipped,
            issue_count=len(issues),
            stage_token_budget=stage_budget,
            configured_stage_token_budget=(
                settings.character_consistency_stage_token_budget
            ),
            remaining_run_tokens_at_stage_start=remaining_run_tokens,
            sensitivity=settings.character_consistency_sensitivity,
            material_coverage="partial" if partial else "complete",
            case_trace=case_trace,
            accepted_draft_observation_refs=accepted_draft_observation_refs,
            accepted_draft_observation_total=accepted_draft_observation_total,
            accepted_draft_observation_refs_truncated=(
                accepted_draft_observation_refs_truncated
            ),
            token_admission_events=token_admission_events,
        )
        return CharacterConsistencyStageResult(
            issues=tuple(issues),
            diagnostics=diagnostics,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            charged_tokens=usage.charged_tokens,
            model_used=successful_model_calls > 0,
        )

    def _bind_frozen_documents(
        self,
        db,
        *,
        run_id: str,
        documents: list[DocumentInput],
        metadata: list[dict[str, Any]],
        reasons: Counter[str],
    ) -> list[_FrozenDocument]:
        rows = list(
            db.scalars(
                select(AnalysisRunInputRow)
                .where(AnalysisRunInputRow.run_id == run_id)
                .order_by(AnalysisRunInputRow.ordinal, AnalysisRunInputRow.id)
            ).all()
        )
        if len(rows) != len(documents) or len(rows) != len(metadata):
            raise ValueError("frozen document projection is incomplete")
        documents_by_id = {row.id: row for row in documents}
        metadata_by_id = {str(row.get("document_id")): row for row in metadata}
        if len(documents_by_id) != len(documents) or len(metadata_by_id) != len(metadata):
            raise ValueError("frozen document projection contains duplicate ids")

        result: list[_FrozenDocument] = []
        for row in rows:
            document = documents_by_id.get(row.document_id)
            item = metadata_by_id.get(row.document_id)
            if document is None or item is None:
                raise ValueError("frozen document projection is misbound")
            content_hash = hashlib.sha256(document.content.encode("utf-8")).hexdigest()
            if (
                content_hash != row.content_sha256
                or item.get("content_sha256") != row.content_sha256
                or item.get("document_version") != row.document_version
                or item.get("ordinal") != row.ordinal
                or document.name != row.document_name
            ):
                raise ValueError("frozen document projection hash mismatch")
            context = item.get("narrative_context")
            if not isinstance(context, dict):
                raise ValueError("frozen narrative context is missing")
            source_kind, source_reason, scope, resolution, publication, authority = (
                _classify_frozen_source(item, context)
            )
            reasons[f"source_{source_reason}"] += 1
            result.append(
                _FrozenDocument(
                    input_id=row.id,
                    document=document,
                    document_version=row.document_version,
                    content_sha256=row.content_sha256,
                    ordinal=row.ordinal,
                    source_kind=source_kind,
                    source_reason=source_reason,
                    scope=scope,
                    resolution_state=resolution,
                    publication_status=publication,
                    authority_tier=authority,
                )
            )
        return result

    def _load_confirmed_traits(
        self,
        db,
        *,
        run_id: str,
        project_id: str,
        reasons: Counter[str],
    ) -> list[
        tuple[
            AnalysisRunCharacterTraitInputRow,
            ConfirmedTraitSnapshot,
            NarrativeScopeV1,
            str,
        ]
    ]:
        rows = list(
            db.scalars(
                select(AnalysisRunCharacterTraitInputRow)
                .where(AnalysisRunCharacterTraitInputRow.run_id == run_id)
                .order_by(
                    AnalysisRunCharacterTraitInputRow.ordinal,
                    AnalysisRunCharacterTraitInputRow.id,
                )
            ).all()
        )
        result = []
        for row in rows:
            try:
                payload = row.payload
                if (
                    row.project_id != project_id
                    or not isinstance(payload, dict)
                    or payload_sha256(payload) != row.payload_sha256
                    or payload.get("candidate_id") != row.candidate_id
                ):
                    raise ValueError("snapshot identity")
                scope = NarrativeScopeV1.model_validate(payload.get("scope"))
                evidence = tuple(
                    EvidenceSpan.model_validate(item)
                    for item in payload.get("evidence", ())
                )
                raw_origin = payload.get("origin")
                if raw_origin == "explicit_setting":
                    origin = "explicit_setting"
                elif raw_origin == "history_inference":
                    origin = "confirmed_history_inference"
                else:
                    raise ValueError("snapshot origin")
                character_key = normalize_character_key(
                    str(payload.get("character_key", ""))
                )
                baseline = ConfirmedTraitSnapshot(
                    id=f"ct_{row.candidate_id}",
                    character=payload.get("character_display_name"),
                    dimension=payload.get("trait_type"),
                    trait_key=payload.get("trait_key"),
                    statement=payload.get("value"),
                    polarity=payload.get("polarity"),
                    stability=payload.get("stability"),
                    contexts=tuple(payload.get("contexts") or ()),
                    origin=origin,
                    authority_tier=(
                        payload.get("authority_tier")
                        if payload.get("authority_tier")
                        in {"core_canon", "formal_record"}
                        else "formal_record"
                    ),
                    valid_from_release_ordinal=payload.get(
                        "valid_from_release_ordinal"
                    ),
                    valid_until_release_ordinal=payload.get(
                        "valid_until_release_ordinal"
                    ),
                    evidence=evidence,
                    approved_axis_id=payload.get("approved_axis_id"),
                    approved_axis_version=payload.get("approved_axis_version"),
                    approved_axis_display_name=payload.get("approved_axis_display_name"),
                    approved_axis_definition=payload.get("approved_axis_definition"),
                    approved_axis_definition_sha256=payload.get(
                        "approved_axis_definition_sha256"
                    ),
                )
            except Exception:
                reasons["invalid_confirmed_trait_snapshot"] += 1
                continue
            result.append((row, baseline, scope, character_key))
        return result


def _classify_frozen_source(
    metadata: dict[str, Any], context: dict[str, Any]
) -> tuple[str | None, str, NarrativeScopeV1 | None, str, str, str]:
    resolution = context.get("resolution_state")
    publication = context.get("publication_status")
    role = metadata.get("document_role")
    if resolution not in {"unresolved", "inferred", "confirmed"}:
        return None, "invalid_resolution", None, "unresolved", "unknown", "unresolved"
    if not isinstance(publication, str):
        publication = "unknown"
    try:
        scope = NarrativeScopeV1.model_validate(context.get("scope"))
    except Exception:
        return None, "invalid_scope", None, resolution, publication, "unresolved"
    if resolution != "confirmed":
        return None, str(resolution), scope, str(resolution), publication, "unresolved"
    authority = context.get("authority_tier")
    if role == "canon":
        return (
            "formal_character_profile",
            "formal",
            scope,
            resolution,
            publication,
            "core_canon" if authority in {None, "core_canon"} else str(authority),
        )
    if role == "character_profile":
        return (
            "formal_character_profile",
            "formal",
            scope,
            resolution,
            publication,
            "formal_record",
        )
    if role == "chapter" and publication in {"published", "retired"}:
        return "published_history", "history", scope, resolution, publication, "formal_record"
    if role == "chapter" and publication in {"draft", "in_review"}:
        return "draft", "draft", scope, resolution, publication, "draft"
    return None, "reference", scope, resolution, publication, "reference"


def _character_appears_in_chunk(character: str, content: str) -> bool:
    character_key = _key(character)
    return bool(character_key) and any(
        character_key in _key(line) for line in content.splitlines()
    )


def _signal_matches_target(
    signal: CharacterSignal, target: CharacterSignalTarget
) -> bool:
    if target.dimension in _OBJECT_BEARING_TRAIT_DIMENSIONS:
        return (
            _key(signal.character) == _key(target.character)
            and signal.dimension == target.dimension
            and (
                target.dimension == "preference"
                or stable_trait_identity(signal.dimension, signal.trait_key)
                == stable_trait_identity(target.dimension, target.trait_key)
            )
            and bool(signal.key_object.strip())
            and (
                stable_trait_identity(
                    signal.dimension, signal.trait_key, signal.key_object
                ) == target.comparison_key
                or preference_modifier_bridge(
                    baseline_comparison_key=target.comparison_key,
                    baseline_polarity=target.baseline_polarity,
                    observation=signal,
                )
            )
        )
    return (
        _key(signal.character) == _key(target.character)
        and signal.dimension == target.dimension
        and trait_keys_compatible(
            dimension=target.dimension,
            baseline_key=target.trait_key,
            observation_key=signal.trait_key,
            observation_object=signal.key_object,
        )
    )


def _trusted_axis_binding(
    signal: CharacterSignal,
    *,
    axis_bindings_by_signal: dict[str, set[tuple[str, int, str]]],
    axis_bindings_by_line: dict[
        tuple[str, str, int], set[tuple[str, int, str]]
    ],
) -> tuple[str, int, str] | None:
    by_signal = axis_bindings_by_signal.get(signal.id, set())
    # The model may quote different substrings of one line for two targets.
    # Keying by the whole source line, not trait_key/statement/text, makes
    # such cross-axis reuse explicitly ambiguous. Overlapping ranges share a
    # line and are ambiguous too, at a conservative recall cost.
    by_line: set[tuple[str, int, str]] = set()
    actor = _key(signal.character)
    for line in range(signal.evidence.line_start, signal.evidence.line_end + 1):
        by_line.update(
            axis_bindings_by_line.get(
                (actor, signal.evidence.document_id, line), set()
            )
        )
    if len(by_signal) != 1 or by_signal != by_line:
        return None
    return next(iter(by_signal))


def _target_has_sufficient_recall_evidence(
    target: CharacterSignalTarget,
    observations: tuple[CharacterSignal, ...],
) -> bool:
    """Mirror drift's threshold on evidence opposing this baseline only."""

    matching = tuple(
        signal
        for signal in observations
        if _signal_matches_target(signal, target)
        and signal.polarity == target.requested_polarity
    )
    if not matching:
        return False
    if target.dimension == "preference":
        if any(
            signal.observation_kind
            in {
                "explicit_declaration",
                "state_description",
                "preference_expression",
            }
            for signal in matching
        ):
            return True
        independent_preference_spans = {
            (
                signal.evidence.document_id,
                signal.evidence.line_start,
                signal.evidence.line_end,
            )
            for signal in matching
            if signal.observation_kind
            in {"action", "decision", "interaction", "dialogue", "speech_sample"}
        }
        return len(independent_preference_spans) >= 2
    if any(
        signal.observation_kind in {"explicit_declaration", "state_description"}
        for signal in matching
    ):
        return True
    independent_spans = {
        (
            signal.evidence.document_id,
            signal.evidence.line_start,
            signal.evidence.line_end,
        )
        for signal in matching
        if signal.observation_kind
        in {"action", "decision", "interaction", "dialogue", "speech_sample"}
    }
    return len(independent_spans) >= 2


def _target_with_existing_evidence_ranges(
    target: CharacterSignalTarget,
    observations: tuple[CharacterSignal, ...],
) -> CharacterSignalTarget:
    """Attach bounded, exact spans already seen in the requested direction."""

    ranges = tuple(
        sorted(
            {
                (signal.evidence.line_start, signal.evidence.line_end)
                for signal in observations
                if _signal_matches_target(signal, target)
                and signal.polarity == target.requested_polarity
            }
        )[:3]
    )
    return CharacterSignalTarget.model_validate(
        {
            **target.model_dump(),
            "existing_evidence_ranges": ranges,
        }
    )


def _target_candidate_line_ranges(
    chunk: CharacterSignalChunk,
    target: CharacterSignalTarget,
) -> tuple[tuple[tuple[int, int], ...], int]:
    """Select bounded, exact named lines or strictly proven adjacent spans.

    Returned ranges retain the original global line numbers. An adjacent
    pronoun line is included only when the extraction binder's same structural
    proof accepts that pair. The extractor still binds output against ``chunk``
    and applies the tuple as an exact server-owned allowlist.
    """

    character_key = _key(target.character)
    candidates: list[tuple[int, int]] = []
    used_lines: set[int] = set()
    for line_number, line in enumerate(
        chunk.content.splitlines(), start=chunk.global_line_start
    ):
        if (
            not character_key
            or character_key not in _key(line)
            or line_number in used_lines
        ):
            continue
        paired = safe_pronoun_evidence_range(
            chunk, target.character, line_number
        )
        candidate = paired or (line_number, line_number)
        if any(
            not (candidate[1] < start or candidate[0] > end)
            for start, end in target.existing_evidence_ranges
        ):
            # A pair could overlap an already-bound following line while the
            # named antecedent remains unused. Keep only that exact named line.
            candidate = (line_number, line_number)
            if any(
                not (candidate[1] < start or candidate[0] > end)
                for start, end in target.existing_evidence_ranges
            ):
                continue
        candidates.append(candidate)
        used_lines.update(range(candidate[0], candidate[1] + 1))
    selected: list[tuple[int, int]] = []
    selected_line_count = 0
    for candidate in candidates:
        span_lines = candidate[1] - candidate[0] + 1
        if selected_line_count + span_lines > MAX_TARGETED_CHARACTER_SIGNAL_CANDIDATE_LINES:
            break
        selected.append(candidate)
        selected_line_count += span_lines
    omitted_lines = sum(
        end - start + 1 for start, end in candidates[len(selected) :]
    )
    return tuple(selected), omitted_lines


def _targeted_completion_reserve(
    settings: Settings,
    chunk: CharacterSignalChunk,
    *,
    candidate_ranges: tuple[tuple[int, int], ...] = (),
) -> int:
    """Reserve for at most three focused records, not a full extraction batch.

    The signal extractor charges its entire completion allowance on admission,
    even when the provider returns a short response. Reusing the 12-record
    allowance for each one-target pass can starve later targets. Keep a generous
    floor for JSON fields and short statements; grow with the three longest
    eligible evidence lines and retain the configured full cap for long prose.
    This changes only capacity planning, never evidence or issue thresholds.
    """

    lines = chunk.content.splitlines()
    if candidate_ranges:
        eligible_indexes = {
            line_number - chunk.global_line_start
            for start, end in candidate_ranges
            for line_number in range(start, end + 1)
            if chunk.global_line_start <= line_number <= chunk.global_line_end
        }
        eligible_lines = (lines[index] for index in sorted(eligible_indexes))
        max_span_lines = max(end - start + 1 for start, end in candidate_ranges)
    else:
        eligible_lines = iter(lines)
        # The binder can accept a strictly proven two-line pronoun span in a
        # full targeted pass. Reserve for up to three such records.
        max_span_lines = 2
    longest_evidence_lines = sorted(
        (min(len(line), 2_000) for line in eligible_lines), reverse=True
    )[: 3 * max_span_lines]
    requested = max(2_048, 1_024 + 2 * sum(longest_evidence_lines))
    return min(settings.character_signal_max_completion_tokens, requested)


def _safe_server_context(
    source: _FrozenDocument,
    *,
    baselines: list[BaselineEntry] | tuple[BaselineEntry, ...] = (),
) -> _SafeServerContext:
    """Build a bounded, content-free alignment hint for draft extraction.

    Only immutable identifiers needed to reuse a comparison axis cross the
    model boundary. An object-bearing comparison key may disclose its short
    object anchor; baseline statements, contexts, evidence, source names,
    URLs and provider configuration are never serialized. Scope and release
    validity are enforced server-side; the scope payload is not sent.
    """

    base_payload: dict[str, Any] = {
        "publication_status": source.publication_status,
    }
    if source.source_kind != "draft" or source.scope is None:
        return _SafeServerContext(payload=_compact_context_json(base_payload))

    eligible_entries: list[BaselineEntry] = []
    for entry in sorted(
        baselines,
        key=lambda item: (
            item[3],
            item[1].dimension,
            item[1].trait_key,
            item[0].candidate_id,
        ),
    ):
        _, baseline, baseline_scope, character_key = entry
        if _trait_applies_to_release(baseline, source.scope) is not True:
            continue
        if _baseline_shadowed_at_scope(entry, baselines, source.scope):
            continue
        if scope_relation(
            baseline_scope,
            source.scope,
            first_resolution="confirmed",
            second_resolution=source.resolution_state,
        ) != "compatible":
            continue
        eligible_entries.append(entry)

    # Contexts deliberately stay off the model boundary. If one comparison
    # key has conflicting polarity, or different behavioral contexts, its
    # first sorted row is not a safe extraction hint for this draft.
    hint_variants: dict[
        tuple[str, str, str, str], set[tuple[tuple[str, ...], str]]
    ] = defaultdict(set)
    unverified_object_keys: set[tuple[str, str, str, str]] = set()
    for entry in eligible_entries:
        _, baseline, _, character_key = entry
        identity = _baseline_hint_identity(entry)
        if baseline.dimension == "contextual_behavior":
            contexts = tuple(
                sorted(value.strip() for value in baseline.contexts if value.strip())
            )
        else:
            contexts = ()
        if (
            baseline.dimension in _OBJECT_BEARING_TRAIT_DIMENSIONS
            and not _frozen_comparison_identity(entry)
        ):
            unverified_object_keys.add(identity)
        hint_variants[identity].add((contexts, baseline.polarity))
    ambiguous_hint_keys = {
        identity for identity, variants in hint_variants.items() if len(variants) > 1
    } | unverified_object_keys

    applicable: list[
        tuple[
            str, str, str, str, str, str,
            tuple[str, int, str] | None, str | None,
        ]
    ] = []
    seen: set[tuple[str, str, str, str]] = set()
    for entry in eligible_entries:
        _, baseline, _, character_key = entry
        identity = _baseline_hint_identity(entry)
        if identity in seen:
            continue
        seen.add(identity)
        if identity in ambiguous_hint_keys:
            applicable.append(("", "", "", "", "", "", None, None))
            continue
        # The approved ID is the *internal* comparison identity. The model
        # still receives a legacy-shaped neutral trait label/key so it cannot
        # claim an axis ID in its output. A second raw label on the same
        # approved axis therefore collapses to this first safe hint.
        comparison_key = (
            stable_trait_identity(baseline.dimension, baseline.trait_key)
            if baseline.approved_axis_identity is not None else identity[2]
        )
        labels = (
            baseline.character,
            baseline.dimension,
            baseline.trait_key,
            comparison_key,
            baseline.polarity,
        )
        axis_key = baseline.approved_axis_identity
        axis_definition = baseline.approved_axis_definition if axis_key else None
        if not all(_safe_context_label(value) for value in labels) or (
            axis_definition is not None
            and not _safe_context_label(axis_definition)
        ):
            # Count the applicable baseline but never serialize a suspicious
            # label.  The resulting partial marker prevents a false clean bill.
            applicable.append(("", "", "", "", "", "", None, None))
            continue
        applicable.append(
            (*labels, _safe_baseline_hint(baseline.statement), axis_key, axis_definition)
        )

    eligible_count = len(applicable)
    selected = [row for row in applicable if row[0]][
        :_MAX_CONFIRMED_TRAITS_IN_SERVER_CONTEXT
    ]
    items = [
        {
            "character": character,
            "dimension": dimension,
            "trait_key": trait_key,
            "comparison_key": comparison_key,
        }
        for character, dimension, trait_key, comparison_key, _, _, _, _ in selected
    ]

    while True:
        state = "complete" if len(items) == eligible_count else "partial"
        payload = {
            **base_payload,
            "confirmed_traits": items,
            "confirmed_traits_coverage": {
                "state": state,
                "included": len(items),
                "eligible": eligible_count,
            },
        }
        serialized = _compact_context_json(payload)
        if len(serialized) <= MAX_CHARACTER_SIGNAL_SERVER_CONTEXT_CHARS:
            targets = tuple(
                CharacterSignalTarget.model_validate(
                    {
                        "character": character,
                        "dimension": dimension,
                        "trait_key": trait_key,
                        "comparison_key": comparison_key,
                        "baseline_polarity": baseline_polarity,
                        "requested_polarity": (
                            "negative" if baseline_polarity == "positive" else "positive"
                        ),
                        "baseline_hint": baseline_hint,
                        **(
                            {
                                "approved_axis_id": axis_key[0],
                                "approved_axis_version": axis_key[1],
                                "approved_axis_definition_sha256": axis_key[2],
                                "approved_axis_definition": axis_definition,
                            }
                            if axis_key is not None else {}
                        ),
                    }
                )
                for (
                    character, dimension, trait_key, comparison_key,
                    baseline_polarity, baseline_hint, axis_key, axis_definition,
                ) in selected
                if baseline_polarity in {"positive", "negative"} and baseline_hint
            )
            return _SafeServerContext(
                payload=serialized,
                eligible_traits=eligible_count,
                included_traits=len(items),
                ambiguous_traits=len(ambiguous_hint_keys),
                targets=targets,
            )
        if not items:
            # The fixed envelope is intentionally tiny; this is a defensive
            # guard in case future fields make even that envelope exceed the
            # CharacterSignalChunk hard limit.
            fallback = {
                "publication_status": "draft",
                "confirmed_traits": [],
                "confirmed_traits_coverage": {
                    "state": "partial",
                    "included": 0,
                    "eligible": eligible_count,
                },
            }
            fallback_payload = _compact_context_json(fallback)
            if len(fallback_payload) > MAX_CHARACTER_SIGNAL_SERVER_CONTEXT_CHARS:
                raise ValueError("fixed character server context exceeds hard limit")
            return _SafeServerContext(
                payload=fallback_payload,
                eligible_traits=eligible_count,
                included_traits=0,
                ambiguous_traits=len(ambiguous_hint_keys),
            )
        items.pop()
        selected.pop()


def _safe_context_label(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 200
        and _CONTEXT_CONTROL.search(value) is None
        and _CONTEXT_SECRET_OR_URL.search(value) is None
    )


def _safe_baseline_hint(value: str) -> str:
    """Create a bounded semantic hint that can never alter prompt structure."""

    cleaned = "".join(
        " "
        if unicodedata.category(character).startswith("C")
        or unicodedata.category(character) in {"Zl", "Zp"}
        else character
        for character in value
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) <= MAX_CHARACTER_SIGNAL_BASELINE_HINT_CHARS:
        return cleaned
    marker = "…[语义提示已截断]"
    boundary = MAX_CHARACTER_SIGNAL_BASELINE_HINT_CHARS - len(marker)
    return f"{cleaned[:boundary].rstrip()}{marker}"


def _compact_context_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _trait_candidate_input(
    candidate: PendingTraitCandidate,
    *,
    frozen_by_document: dict[str, _FrozenDocument],
) -> TraitCandidateInput | None:
    sources: list[_FrozenDocument] = []
    evidence_rows: list[TraitEvidenceInput] = []
    for evidence in candidate.evidence:
        source = frozen_by_document.get(evidence.document_id)
        if (
            source is None
            or source.scope is None
            or source.resolution_state != "confirmed"
            or source.source_kind not in {
                "formal_character_profile",
                "published_history",
            }
        ):
            return None
        sources.append(source)
        evidence_rows.append(
            TraitEvidenceInput(
                input_id=source.input_id,
                document_id=source.document.id,
                document_name=source.document.name,
                document_version=source.document_version,
                content_sha256=source.content_sha256,
                line_start=evidence.line_start,
                line_end=evidence.line_end,
                text=evidence.text,
            )
        )
    scope = _merge_compatible_scopes(sources)
    if scope is None:
        return None
    release_ordinals = [
        source.scope.release.ordinal
        for source in sources
        if source.scope is not None and source.scope.release is not None
    ]
    authority_tier = (
        "core_canon"
        if any(source.authority_tier == "core_canon" for source in sources)
        else "formal_record"
    )
    return TraitCandidateInput(
        character_key=normalize_character_key(candidate.character),
        character_display_name=candidate.character,
        trait_type=candidate.dimension,
        trait_key=candidate.trait_key,
        comparison_key=candidate.comparison_key,
        value=candidate.statement,
        polarity=candidate.polarity,
        stability=candidate.stability,
        contexts=list(candidate.contexts),
        origin=candidate.origin,
        authority_tier=authority_tier,
        confidence=0.9 if candidate.origin == "explicit_setting" else 0.7,
        scope=scope,
        valid_from_release_ordinal=max(release_ordinals) if release_ordinals else None,
        valid_until_release_ordinal=None,
        evidence=evidence_rows,
        generator_version=CHARACTER_CONSISTENCY_CHECKER_VERSION,
        provenance={
            "schema_version": 1,
            "stage": "character_signal_extraction",
            "evidence_count": len(evidence_rows),
            "source_kind": (
                "formal_character_profile"
                if candidate.origin == "explicit_setting"
                else "published_history"
            ),
        },
    )


def _merge_compatible_scopes(
    sources: list[_FrozenDocument],
) -> NarrativeScopeV1 | None:
    if not sources or any(source.scope is None for source in sources):
        return None
    for index, left in enumerate(sources):
        for right in sources[index + 1 :]:
            if (
                scope_relation(
                    left.scope,
                    right.scope,
                    first_resolution=left.resolution_state,
                    second_resolution=right.resolution_state,
                )
                != "compatible"
            ):
                return None
    scopes = [source.scope for source in sources if source.scope is not None]
    timeline = scopes[0].timeline_key
    activities = {row.activity_key for row in scopes if row.activity_key is not None}
    if len(activities) > 1:
        return None
    branches = [row.branch for row in scopes if row.branch is not None]
    branch = max(branches, key=lambda row: len(row.path)) if branches else None
    releases = [row.release for row in scopes if row.release is not None]
    release = max(releases, key=lambda row: row.ordinal) if releases else None
    return NarrativeScopeV1(
        timeline_key=timeline,
        activity_key=next(iter(activities), None),
        branch=branch,
        release=release,
    )


def _trait_applies_to_release(
    baseline: ConfirmedTraitSnapshot,
    target_scope: NarrativeScopeV1,
) -> bool | None:
    """Return None when a bounded trait cannot be placed on the target release."""

    lower = baseline.valid_from_release_ordinal
    upper = baseline.valid_until_release_ordinal
    if lower is None and upper is None:
        return True
    if target_scope.release is None:
        return None
    ordinal = target_scope.release.ordinal
    if lower is not None and ordinal < lower:
        return False
    if upper is not None and ordinal > upper:
        return False
    return True


def _release_range_covers(
    higher: ConfirmedTraitSnapshot, lower: ConfirmedTraitSnapshot
) -> bool:
    """Only discard a lower baseline when no release can still need it."""

    higher_start = higher.valid_from_release_ordinal or 0
    lower_start = lower.valid_from_release_ordinal or 0
    higher_end = (
        higher.valid_until_release_ordinal
        if higher.valid_until_release_ordinal is not None
        else 2_147_483_647
    )
    lower_end = (
        lower.valid_until_release_ordinal
        if lower.valid_until_release_ordinal is not None
        else 2_147_483_647
    )
    return higher_start <= lower_start and higher_end >= lower_end


def _scope_covers(higher: NarrativeScopeV1, lower: NarrativeScopeV1) -> bool:
    """A branch or activity-specific setting cannot erase global history."""

    if higher.timeline_key != lower.timeline_key:
        return False
    if higher.activity_key is not None and higher.activity_key != lower.activity_key:
        return False
    if higher.branch is None:
        return True
    if lower.branch is None:
        return False
    if (
        higher.branch.exclusive_group is not None
        and higher.branch.exclusive_group != lower.branch.exclusive_group
    ):
        return False
    return tuple(lower.branch.path[: len(higher.branch.path)]) == tuple(
        higher.branch.path
    )


def _higher_authority_on_axis(
    lower: BaselineEntry, higher: BaselineEntry
) -> bool:
    _, baseline, _, character_key = lower
    _, other, _, other_character_key = higher
    return (
        (
            other.authority_tier == "core_canon"
            or (
                baseline.authority_tier == "formal_record"
                and baseline.origin == "confirmed_history_inference"
                and other.authority_tier == "formal_record"
                and other.origin == "explicit_setting"
            )
        )
        and other_character_key == character_key
        and other.dimension == baseline.dimension
        and _same_shadow_axis(lower, higher)
    )


def _same_shadow_axis(
    lower: BaselineEntry, higher: BaselineEntry
) -> bool:
    """Require a stable identity before removing a confirmed baseline."""

    lower_baseline = lower[1]
    higher_baseline = higher[1]
    if (
        lower_baseline.approved_axis_identity is not None
        or higher_baseline.approved_axis_identity is not None
    ):
        return (
            lower_baseline.approved_axis_identity is not None
            and lower_baseline.approved_axis_identity
            == higher_baseline.approved_axis_identity
        )
    if lower_baseline.dimension in _OBJECT_BEARING_TRAIT_DIMENSIONS:
        frozen_key = _frozen_comparison_identity(lower)
        return (
            bool(frozen_key)
            and frozen_key == _frozen_comparison_identity(higher)
            and (
                lower_baseline.dimension == "preference"
                or stable_trait_identity(
                    lower_baseline.dimension, lower_baseline.trait_key
                ) == stable_trait_identity(
                    higher_baseline.dimension, higher_baseline.trait_key
                )
            )
        )
    return stable_trait_identity(
        lower_baseline.dimension, lower_baseline.trait_key
    ) == stable_trait_identity(higher_baseline.dimension, higher_baseline.trait_key)


def _observation_matches_baseline(
    entry: BaselineEntry,
    observation: CharacterSignal,
    *,
    approved_axis_binding: tuple[str, int, str] | None = None,
) -> bool | None:
    baseline = entry[1]
    if observation.dimension != baseline.dimension:
        return False
    if baseline.approved_axis_identity is not None:
        # This binding is populated only after a clean, validated, one-target
        # recall call. A primary model key never creates approved identity.
        return approved_axis_binding == baseline.approved_axis_identity
    if baseline.dimension in _OBJECT_BEARING_TRAIT_DIMENSIONS:
        frozen_key = _frozen_comparison_identity(entry)
        if not frozen_key:
            return None
        if (
            baseline.dimension != "preference"
            and stable_trait_identity(baseline.dimension, baseline.trait_key)
            != stable_trait_identity(observation.dimension, observation.trait_key)
        ):
            return False
        return bool(observation.key_object.strip()) and (
            stable_trait_identity(
                observation.dimension, observation.trait_key, observation.key_object
            ) == frozen_key
            or preference_modifier_bridge(
                baseline_comparison_key=frozen_key,
                baseline_polarity=baseline.polarity,
                observation=observation,
            )
        )
    return trait_keys_compatible(
        dimension=baseline.dimension,
        baseline_key=baseline.trait_key,
        observation_key=observation.trait_key,
        observation_object=observation.key_object,
    )


def _frozen_comparison_identity(entry: BaselineEntry) -> str:
    """Read only a validated, hash-bound comparison key from the run snapshot."""

    row, baseline, _, _ = entry
    payload = getattr(row, "payload", None)
    if not isinstance(payload, dict):
        return ""
    value = payload.get("comparison_key")
    if not _safe_context_label(value) or len(value) > 160:
        return ""
    prefix = f"{baseline.dimension}:"
    if not value.startswith(prefix):
        return ""
    anchor = value[len(prefix) :]
    if (
        not anchor
        or ":" in anchor
        or _key(anchor) != anchor
        or _CONTEXT_CONTROL.search(anchor)
    ):
        return ""
    return value


def _baseline_hint_identity(
    entry: BaselineEntry,
) -> tuple[str, str, str, str]:
    _, baseline, _, character_key = entry
    frozen_key = (
        _frozen_comparison_identity(entry)
        if baseline.dimension in _OBJECT_BEARING_TRAIT_DIMENSIONS
        else ""
    )
    return (
        character_key,
        baseline.dimension,
        (
            f"approved-axis:{baseline.approved_axis_identity}"
            if baseline.approved_axis_identity is not None
            else frozen_key or stable_trait_identity(
                baseline.dimension, baseline.trait_key
            )
        ),
        (
            stable_trait_identity(baseline.dimension, baseline.trait_key)
            if frozen_key and baseline.dimension != "preference"
            else ""
        ),
    )


def _context_coverage(
    lower: ConfirmedTraitSnapshot,
    higher: ConfirmedTraitSnapshot,
    *,
    observation_context: str | None = None,
) -> bool:
    if lower.dimension != "contextual_behavior":
        return True
    lower_contexts = {value.strip() for value in lower.contexts if value.strip()}
    higher_contexts = {value.strip() for value in higher.contexts if value.strip()}
    if not lower_contexts or not higher_contexts:
        return False
    if observation_context is not None:
        target = observation_context.strip()
        return bool(target) and target in lower_contexts and target in higher_contexts
    return lower_contexts <= higher_contexts


def _baseline_shadowed_at_scope(
    entry: BaselineEntry,
    baselines: list[BaselineEntry] | tuple[BaselineEntry, ...],
    target_scope: NarrativeScopeV1,
    *,
    observation_context: str | None = None,
) -> bool:
    """Resolve authority for one draft's known release and scope."""

    _, baseline, baseline_scope, _ = entry
    if baseline.authority_tier == "core_canon":
        return False
    if _trait_applies_to_release(baseline, target_scope) is not True:
        return False
    if scope_relation(
        baseline_scope,
        target_scope,
        first_resolution="confirmed",
        second_resolution="confirmed",
    ) != "compatible":
        return False
    for other_entry in baselines:
        _, other, other_scope, _ = other_entry
        if (
            _higher_authority_on_axis(entry, other_entry)
            and _trait_applies_to_release(other, target_scope) is True
            and _scope_covers(other_scope, target_scope)
            and _context_coverage(
                baseline, other, observation_context=observation_context
            )
        ):
            return True
    return False


def _select_authoritative_baselines(
    baselines: list[BaselineEntry],
) -> tuple[list[BaselineEntry], int]:
    """Keep the strongest applicable baseline on a character's trait axis.

    A confirmed character profile is an explicit setting, while a trait
    inferred from published chapters is historical evidence. The latter may
    explain a change later, but it cannot silently replace an overlapping
    explicit setting as the baseline for draft review.
    """

    selected: list[BaselineEntry] = []
    shadowed = 0
    for entry in baselines:
        _, baseline, scope, character_key = entry
        if baseline.authority_tier == "core_canon":
            selected.append(entry)
            continue
        dominated = False
        for other_entry in baselines:
            _, other, other_scope, _ = other_entry
            if (
                not _higher_authority_on_axis(entry, other_entry)
                or not _release_range_covers(other, baseline)
                or not _scope_covers(other_scope, scope)
                or not _context_coverage(baseline, other)
            ):
                continue
            dominated = True
            break
        if dominated:
            shadowed += 1
        else:
            selected.append(entry)
    return selected, shadowed


def _unique_alias_map(
    baselines: list[BaselineEntry],
) -> dict[str, tuple[str, ...]]:
    values: dict[str, set[str]] = defaultdict(set)
    for _, baseline, _, character_key in baselines:
        values[_key(character_key)].add(character_key)
        values[_key(baseline.character)].add(character_key)
    return {alias: tuple(sorted(keys)) for alias, keys in values.items()}


def _find_support_evidence(
    *,
    baseline: ConfirmedTraitSnapshot,
    baseline_scope: NarrativeScopeV1,
    draft_scopes: tuple[NarrativeScopeV1, ...],
    documents: list[_FrozenDocument],
    limit: int,
) -> tuple[SupportEvidence, ...]:
    if limit <= 0:
        return ()
    character = _key(baseline.character)
    baseline_ranges = tuple(
        (row.document_id, row.line_start, row.line_end)
        for row in baseline.evidence
    )
    result: list[SupportEvidence] = []
    seen: set[tuple[str, int]] = set()
    for source in sorted(documents, key=lambda row: row.ordinal):
        if source.scope is None or source.resolution_state != "confirmed":
            continue
        if (
            scope_relation(
                baseline_scope,
                source.scope,
                first_resolution="confirmed",
                second_resolution=source.resolution_state,
            )
            != "compatible"
        ):
            continue
        if any(
            scope_relation(
                draft_scope,
                source.scope,
                first_resolution="confirmed",
                second_resolution=source.resolution_state,
            )
            != "compatible"
            for draft_scope in draft_scopes
        ):
            continue
        for line_number, line in enumerate(source.document.content.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or len(stripped) > 2_000 or character not in _key(stripped):
                continue
            # A baseline line proves the trait, not a later event that explains
            # an apparent change.  Reusing it as G/X would let a conditional or
            # introductory setting sentence explain its own contradiction.
            if any(
                document_id == source.document.id
                and line_start <= line_number <= line_end
                for document_id, line_start, line_end in baseline_ranges
            ):
                continue
            kind = _explicit_support_kind(stripped)
            if kind is None or (source.document.id, line_number) in seen:
                continue
            seen.add((source.document.id, line_number))
            identity = hashlib.sha256(
                f"{source.document.id}:{line_number}:{kind}".encode("utf-8")
            ).hexdigest()[:32]
            result.append(
                SupportEvidence(
                    id=f"se_{identity}",
                    kind=kind,
                    summary=(
                        "原文明示可能存在成长或因果依据"
                        if kind == "causal_bridge"
                        else "原文明示可能存在伪装或临时情境"
                    ),
                    explicit=True,
                    evidence=EvidenceSpan(
                        document_id=source.document.id,
                        document_name=source.document.name,
                        line_start=line_number,
                        line_end=line_number,
                        text=stripped,
                    ),
                )
            )
            if len(result) >= limit:
                return tuple(result)
    return tuple(result)


def _explicit_support_kind(line: str) -> str | None:
    """Return support only for an affirmative clause, never a lexical hit.

    Narrative prose often says that no growth or disguise was described.  A
    raw keyword search would invert that absence statement into an exception
    and suppress a real finding.  Splitting at ordinary Chinese/ASCII clause
    punctuation keeps the conservative filter local and intentionally accepts
    false negatives over unsupported explanations.
    """

    # Conditions and rules often span comma-delimited clauses ("if X, then
    # Y"). Reject the whole line so the consequent cannot masquerade as an
    # event that actually occurred.
    if _CONDITIONAL_OR_RULE.search(line):
        return None

    for clause in re.split(r"[，,。；;！？!?\n]+", line):
        candidate = clause.strip()
        if not candidate or _NEGATED_OR_UNCERTAIN.search(candidate):
            continue
        if _TEMPORARY_CHARACTER_STATE_OR_BEHAVIOR.search(candidate):
            return "exception"
        if _EXCEPTION_PATTERN.search(candidate):
            return "exception"
        if _BRIDGE_PATTERN.search(candidate):
            return "causal_bridge"
    return None


def _to_issue(
    *,
    promoted,
    prepared,
    confirmed_candidate_id: str,
    judgement: str,
) -> ConsistencyIssue:
    baseline_spans = list(prepared.case.baseline.evidence)
    current_spans = [row.evidence for row in prepared.matching_observations]
    if not baseline_spans or not current_spans:
        raise ValueError("character drift issue requires baseline and current evidence")
    evidence: list[EvidenceSpan] = [baseline_spans[0], current_spans[0]]
    seen = {
        (row.document_id, row.line_start, row.line_end) for row in evidence
    }
    for row in promoted.evidence:
        key = (row.document_id, row.line_start, row.line_end)
        if key not in seen and len(evidence) < 12:
            evidence.append(row)
            seen.add(key)
    conflict = promoted.outcome == "conflict"
    severity = Severity.high if conflict else (
        Severity.medium if promoted.confidence_band == "medium" else Severity.low
    )
    confidence = 0.92 if conflict else (
        0.68 if promoted.confidence_band == "medium" else 0.52
    )
    return ConsistencyIssue(
        category=IssueCategory.character_drift,
        severity=severity,
        confidence=confidence,
        title=(
            f"{prepared.case.baseline.character}的角色设定可能冲突"
            if conflict
            else f"{prepared.case.baseline.character}的角色表现需要确认"
        ),
        explanation=promoted.explanation,
        evidence=evidence,
        suggestion=_SUGGESTION,
        metadata={
            "character_key": normalize_character_key(
                prepared.case.baseline.character
            ),
            "dimension": prepared.case.baseline.dimension,
            "trait_key": prepared.case.baseline.trait_key,
            **(
                {
                    "approved_axis_id": prepared.case.baseline.approved_axis_id,
                    "approved_axis_version": (
                        prepared.case.baseline.approved_axis_version
                    ),
                    "approved_axis_definition_sha256": (
                        prepared.case.baseline.approved_axis_definition_sha256
                    ),
                    "observation_axis_binding": "server_targeted_evidence",
                }
                if prepared.case.baseline.approved_axis_identity is not None
                else {}
            ),
            "confirmed_candidate_id": confirmed_candidate_id,
            "subtype": promoted.subtype,
            "judgement": judgement,
            "checker_version": CHARACTER_CONSISTENCY_CHECKER_VERSION,
            "scope_relation": prepared.case.scope_compatibility,
            "sensitivity": promoted.sensitivity,
        },
    )


def _key(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def _safe_trace_identifier(value: str, *, max_chars: int) -> str:
    """Keep trace labels bounded while redacting URL/credential-shaped data."""

    normalized = unicodedata.normalize("NFKC", value).strip()
    if (
        not normalized
        or len(normalized) > max_chars
        or _CONTEXT_CONTROL.search(normalized)
        or _CONTEXT_SECRET_OR_URL.search(normalized)
    ):
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"redacted_{digest}"
    return normalized


def _safe_trace_document_name(value: str) -> str:
    """Expose a file label, never a path, URL, credential or source prose."""

    normalized = unicodedata.normalize("NFKC", value).strip()
    if "/" in normalized or "\\" in normalized:
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
        return f"redacted_{digest}"
    return _safe_trace_identifier(normalized, max_chars=128)


def _safe_observation_refs(
    observations: tuple[CharacterSignal, ...],
) -> list[dict[str, Any]]:
    """Only source coordinates and validated signal labels leave the stage."""

    return [
        {
            "document_name": _safe_trace_document_name(row.evidence.document_name),
            "line_start": row.evidence.line_start,
            "line_end": row.evidence.line_end,
            "observation_kind": row.observation_kind,
            "polarity": row.polarity,
            "key_object_sha256": (
                hashlib.sha256(_key(row.key_object).encode("utf-8")).hexdigest()
                if _key(row.key_object)
                else None
            ),
        }
        for row in observations[:_MAX_CASE_TRACE_OBSERVATION_REFS]
    ]


def _safe_accepted_draft_observation_refs(
    signals: tuple[CharacterSignal, ...],
) -> tuple[list[dict[str, Any]], int, bool]:
    """Summarize every unique, validated draft signal before alias matching.

    Only the stage's accepted signal set reaches this function. Suspicious
    identifiers are omitted rather than exposing a redacted alias as if it
    were an exact actor/document match for later evaluation.
    """

    draft_signals = tuple(row for row in signals if row.source_kind == "draft")
    refs: list[dict[str, Any]] = []
    for row in draft_signals:
        if len(refs) >= _MAX_ACCEPTED_DRAFT_OBSERVATION_REFS:
            break
        character_key = _safe_trace_identifier(_key(row.character), max_chars=64)
        document_name = _safe_trace_document_name(row.evidence.document_name)
        start, end = row.evidence.line_start, row.evidence.line_end
        if (
            character_key.startswith("redacted_")
            or document_name != row.evidence.document_name
            or type(start) is not int
            or type(end) is not int
            or not 1 <= start <= end <= _MAX_CASE_TRACE_LINE
        ):
            continue
        refs.append(
            {
                "character_key": character_key,
                "dimension": row.dimension,
                "polarity": row.polarity,
                "observation_kind": row.observation_kind,
                "document_name": document_name,
                "line_start": start,
                "line_end": end,
            }
        )
    return refs, len(draft_signals), len(refs) < len(draft_signals)


def _safe_citation_refs(
    prepared: PreparedCharacterDrift | None,
    review: CharacterReviewResult | None,
) -> tuple[list[dict[str, Any]], bool]:
    """Resolve reviewed handles against the *same* bounded evidence table.

    Never derive a coordinate from a model explanation or from a guessed handle.
    The reviewer builds B/C/G/X labels through ``_evidence_rows``; this trace
    copies only the citation handle, role, safe file label, and line numbers.
    """

    decision = review.decision if review is not None else None
    if decision is None:
        return [], bool(
            review is not None
            and review.diagnostics.reason == "invalid_model_response"
        )
    if prepared is None or review.diagnostics.outcome != "completed":
        return [], True
    try:
        citations = decision.citations
        if not citations or len(set(citations)) != len(citations):
            return [], True
        evidence_rows, _ = _evidence_rows(prepared)
        by_handle = {row["id"]: row for row in evidence_rows}
        refs: list[dict[str, Any]] = []
        for handle in citations:
            if not isinstance(handle, str) or not _CASE_TRACE_CITATION_HANDLE.fullmatch(handle):
                return [], True
            row = by_handle.get(handle)
            if row is None:
                return [], True
            if row.get("role") != {
                "B": "baseline",
                "C": "current",
                "G": "bridge",
                "X": "exception",
            }[handle[0]]:
                return [], True
            document_name = row.get("document")
            line_start = row.get("line_start")
            line_end = row.get("line_end")
            if (
                not isinstance(document_name, str)
                or _safe_trace_document_name(document_name) != document_name
                or type(line_start) is not int
                or type(line_end) is not int
                or not 1 <= line_start <= line_end <= _MAX_CASE_TRACE_LINE
            ):
                return [], True
            refs.append(
                {
                    "handle": handle,
                    "role": handle[0],
                    "document_name": document_name,
                    "line_start": line_start,
                    "line_end": line_end,
                }
            )
        return refs[:_MAX_CASE_TRACE_CITATION_REFS], (
            len(refs) > _MAX_CASE_TRACE_CITATION_REFS
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return [], True


def _safe_case_trace(
    *,
    character_key: str,
    baseline: ConfirmedTraitSnapshot,
    baseline_entry: BaselineEntry | None = None,
    matched_observation_count: int,
    matched_observations: tuple[CharacterSignal, ...] = (),
    prepared: PreparedCharacterDrift | None = None,
    prepare_reason: str,
    review: CharacterReviewResult | None,
    final_outcome: str,
    visible: bool,
    promote_reason: str,
) -> dict[str, Any]:
    decision = review.decision if review is not None else None
    citation_refs, citation_refs_incomplete = _safe_citation_refs(prepared, review)
    roles = (
        sorted(
            {
                citation[0]
                for citation in decision.citations
                if citation and citation[0] in {"B", "C", "G", "X"}
            },
            key="BCGX".index,
        )
        if decision is not None
        else []
    )
    frozen_key = (
        _frozen_comparison_identity(baseline_entry)
        if baseline_entry is not None
        and baseline.dimension in _OBJECT_BEARING_TRAIT_DIMENSIONS
        else ""
    )
    if baseline.approved_axis_identity is not None:
        comparison_key = (
            f"approved_axis:{baseline.approved_axis_id}:"
            f"{baseline.approved_axis_version}"
        )
    elif frozen_key:
        comparison_key = frozen_key
    elif _CONTEXT_SECRET_OR_URL.search(baseline.trait_key):
        # Legacy snapshots may lack a usable object axis. Preserve the old
        # content-free fallback without exposing a suspicious trait label.
        comparison_key = (
            "redacted_"
            + hashlib.sha256(baseline.trait_key.encode("utf-8")).hexdigest()[:16]
        )
    else:
        # Non-object traits and legacy snapshots retain their prior trace key.
        comparison_key = stable_trait_identity(
            baseline.dimension,
            baseline.trait_key,
        )
    candidate_id_sha256 = None
    if baseline_entry is not None:
        candidate_id = getattr(baseline_entry[0], "candidate_id", None)
        try:
            if (
                isinstance(candidate_id, str)
                and str(UUID(candidate_id)) == candidate_id
                and baseline.id == f"ct_{candidate_id}"
            ):
                candidate_id_sha256 = hashlib.sha256(
                    candidate_id.encode("utf-8")
                ).hexdigest()
        except ValueError:
            pass
    return {
        "character_key": _safe_trace_identifier(character_key, max_chars=64),
        "dimension": baseline.dimension,
        "comparison_key": _safe_trace_identifier(comparison_key, max_chars=128),
        **(
            {
                "approved_axis_definition_sha256": (
                    baseline.approved_axis_definition_sha256
                ),
                "observation_axis_binding": "server_targeted_evidence",
            }
            if baseline.approved_axis_identity is not None else {}
        ),
        "confirmed_candidate_id_sha256": candidate_id_sha256,
        "matched_observation_count": max(0, min(matched_observation_count, 24)),
        "matched_observation_refs": _safe_observation_refs(matched_observations),
        "matched_observation_refs_truncated": (
            len(matched_observations) > _MAX_CASE_TRACE_OBSERVATION_REFS
        ),
        "prepare_reason": prepare_reason,
        "review_outcome": (
            review.diagnostics.outcome if review is not None else "not_run"
        ),
        "review_verdict": decision.verdict if decision is not None else None,
        "citation_roles": roles,
        "citation_refs": citation_refs,
        "citation_refs_incomplete": citation_refs_incomplete,
        "final_outcome": final_outcome,
        "visible": bool(visible),
        "promote_reason": promote_reason,
    }


def _diagnostics(
    *,
    outcome: str,
    reason_code: str,
    usage: _Usage,
    reasons: Counter[str],
    material_coverage: str = "unknown",
    case_trace: list[dict[str, Any]] | None = None,
    accepted_draft_observation_refs: list[dict[str, Any]] | None = None,
    accepted_draft_observation_total: int = 0,
    accepted_draft_observation_refs_truncated: bool = False,
    token_admission_events: list[dict[str, int | str | None]] | None = None,
    **counts: Any,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "outcome": outcome,
        "reason_code": reason_code,
        "checker_version": CHARACTER_CONSISTENCY_CHECKER_VERSION,
        "snapshot_bound": True,
        "material_coverage": material_coverage,
        "counts": counts,
        "case_trace": list(case_trace or ()),
        "accepted_draft_observation_refs": list(
            accepted_draft_observation_refs or ()
        ),
        "accepted_draft_observation_total": accepted_draft_observation_total,
        "accepted_draft_observation_refs_truncated": bool(
            accepted_draft_observation_refs_truncated
        ),
        "token_admission_events": list(token_admission_events or ()),
        "reason_counts": dict(sorted(reasons.items())),
        "usage": usage.safe_dict(),
        "boundary": (
            "Optional frozen-input stage; model output cannot decide authority, "
            "scope, branch compatibility or confirmation state."
        ),
    }


def _empty_stage_result(outcome: str, reason_code: str) -> CharacterConsistencyStageResult:
    enabled = outcome != "disabled"
    return CharacterConsistencyStageResult(
        diagnostics={
            "enabled": enabled,
            "outcome": outcome,
            "reason_code": reason_code,
            "checker_version": CHARACTER_CONSISTENCY_CHECKER_VERSION,
            "snapshot_bound": True,
            "material_coverage": "unknown",
            "counts": {},
            "case_trace": [],
            "accepted_draft_observation_refs": [],
            "accepted_draft_observation_total": 0,
            "accepted_draft_observation_refs_truncated": False,
            "token_admission_events": [],
            "reason_counts": {},
            "usage": _Usage().safe_dict(),
            "boundary": (
                "Optional frozen-input stage; model output cannot decide authority, "
                "scope, branch compatibility or confirmation state."
            ),
        }
    )


def failed_character_consistency_stage() -> CharacterConsistencyStageResult:
    """Content-free optional-stage failure used by the service boundary."""

    return _empty_stage_result("degraded", "internal_failure")
