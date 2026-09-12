from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from itertools import islice
from typing import Literal
from uuid import UUID, uuid5

from .domain import ConsistencyIssue, EvidenceSpan, IssueCategory, ParsedDirective
from .evidence_authority import (
    InvestigationScope,
    ScopedEvidenceDocument,
    clone_investigation_scope,
)
from .evidence_investigator import (
    HASH_PATTERN,
    SEED_REF_PATTERN,
    CandidateRecordSubmission,
    InvestigationSeed,
    candidate_kinds,
    clone_investigation_seed,
    get_candidate_field_contract,
)
from .evidence_investigator_state import UntrustedCandidateEnvelope
from .evidence_investigator_loop import (
    AuthorizedCandidateBinding,
    EvidenceInvestigatorLoopResult,
)
from .evidence_chunks import SnapshotDocumentKey
from .rules import detect_issues
from .semantic_quality import (
    KNOWLEDGE_ACQUISITION_VERB_PATTERN,
    KNOWLEDGE_CLAIM_VERB_PATTERN,
    apply_semantic_quality_gate,
    document_context_has_noncanonical_frame,
    eligible_for_deterministic_rules,
    find_bound_fact_relation_matches,
    find_bound_use_action_matches,
    has_unrealized_heading_frame,
)


PromotionReason = Literal[
    "accepted",
    "invalid_envelope",
    "unknown_seed",
    "candidate_budget",
    "candidate_payload_too_large",
    "candidate_fields_forbidden",
    "candidate_shape_invalid",
    "candidate_not_grounded",
    "candidate_join_mismatch",
    "candidate_evidence_unauthorized",
    "candidate_evidence_reused",
    "candidate_semantics_rejected",
    "candidate_no_rule_conflict",
    "candidate_duplicate",
]

_PROMOTED_ISSUE_NAMESPACE = UUID("d1e64f8c-cde3-4fb9-b90f-fce8be1740f8")
_CONTROL_CHARACTER = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_PRECISE_TIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?$"
)
_WORLD_KEY = re.compile(r"^[^:\s]{1,64}:[^:\s]{1,64}:[^:\s]{1,64}$")


@dataclass(frozen=True, slots=True)
class CandidatePromotionLimits:
    """Hard bounds applied after the permissive tool-transport contract."""

    max_envelopes: int = 32
    max_candidates: int = 32
    max_candidate_payload_bytes: int = 4_096
    max_total_payload_bytes: int = 32_000
    max_field_value_bytes: int = 512
    max_total_field_bytes: int = 1_536
    max_json_nodes: int = 32
    max_rule_context: int = 256
    max_elapsed_ms: int = 5_000

    def __post_init__(self) -> None:
        bounds = (
            (self.max_envelopes, 1, 256),
            (self.max_candidates, 1, 256),
            (self.max_candidate_payload_bytes, 128, 16_384),
            (self.max_total_payload_bytes, 128, 1_000_000),
            (self.max_field_value_bytes, 8, 4_096),
            (self.max_total_field_bytes, 32, 16_384),
            (self.max_json_nodes, 4, 128),
            (self.max_rule_context, 2, 2_048),
            (self.max_elapsed_ms, 1, 60_000),
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
            for value, minimum, maximum in bounds
        ):
            raise ValueError("candidate promotion limit is invalid")


@dataclass(frozen=True, slots=True)
class TrustedDocumentContext:
    """Server-owned metadata for one document in an investigation snapshot."""

    document_id: str
    document_name: str
    story_scope: str = "global"
    document_role: Literal[
        "canon", "character_profile", "chapter", "reference"
    ] | None = None

    def __post_init__(self) -> None:
        for value in (self.document_id, self.document_name, self.story_scope):
            if (
                type(value) is not str
                or not value
                or value != value.strip()
                or len(value.encode("utf-8")) > 512
                or _CONTROL_CHARACTER.search(value)
            ):
                raise ValueError("trusted document context is invalid")
        if self.document_role not in {
            None,
            "canon",
            "character_profile",
            "chapter",
            "reference",
        }:
            raise ValueError("trusted document context is invalid")


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedCandidateEvidence:
    """Exact server-side span binding; never constructed from candidate JSON."""

    evidence: EvidenceSpan
    document_content: str
    story_scope: str
    document_role: str | None


@dataclass(frozen=True, slots=True, repr=False)
class _AuthorizedBinding:
    seed_ref: str
    candidate_payload: str
    span_ref: str
    authorized_span_sha256: str
    snapshot_project_id: str
    snapshot_document_id: str
    snapshot_document_version: int
    snapshot_content_sha256: str
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    text: str
    text_sha256: str
    authorized_span_char_start: int
    authorized_span_char_end: int


class CandidateEvidenceResolver:
    """Bridge opaque span capabilities to exact server-owned source context.

    The investigator envelope intentionally contains no document name, role,
    scope, version, or source text. The service creates this resolver from the
    exact closed snapshot and one in-memory, completed loop result, then injects
    the document metadata it already owns. Opaque authority state is neither
    reconstructed nor accessed outside the loop. This constructor is an
    in-process handoff only: never deserialize, persist, or accept either the
    result or its bindings from a client.
    """

    def __init__(
        self,
        *,
        scope: InvestigationScope,
        documents: Sequence[TrustedDocumentContext],
        investigator_result: EvidenceInvestigatorLoopResult,
    ) -> None:
        prepared_scope = clone_investigation_scope(scope)
        contexts = _bounded_sequence(
            documents,
            maximum=256,
            message="trusted document contexts are invalid",
        )
        if any(type(row) is not TrustedDocumentContext for row in contexts):
            raise ValueError("trusted document contexts are invalid")
        try:
            contexts = tuple(
                TrustedDocumentContext(
                    document_id=row.document_id,
                    document_name=row.document_name,
                    story_scope=row.story_scope,
                    document_role=row.document_role,
                )
                for row in contexts
            )
        except (AttributeError, TypeError, ValueError):
            raise ValueError("trusted document contexts are invalid") from None
        expected_ids = {
            document.snapshot.document_id for document in prepared_scope.documents
        }
        if (
            len({row.document_id for row in contexts}) != len(contexts)
            or {row.document_id for row in contexts} != expected_ids
        ):
            raise ValueError("trusted document contexts do not match the scope")
        self._scope = prepared_scope
        self._documents = {row.snapshot: row for row in prepared_scope.documents}
        self._contexts = {row.document_id: row for row in contexts}
        if type(investigator_result) is not EvidenceInvestigatorLoopResult:
            raise ValueError("completed investigator result is invalid")
        try:
            checked_result = EvidenceInvestigatorLoopResult(
                outcome=investigator_result.outcome,
                reason_code=investigator_result.reason_code,
                envelopes=tuple(investigator_result.envelopes),
                authorized_candidates=tuple(
                    investigator_result.authorized_candidates
                ),
                provider_calls=investigator_result.provider_calls,
                reported_prompt_tokens=investigator_result.reported_prompt_tokens,
                reported_completion_tokens=(
                    investigator_result.reported_completion_tokens
                ),
                charged_tokens=investigator_result.charged_tokens,
                usage_unavailable_calls=investigator_result.usage_unavailable_calls,
                completed_seeds=investigator_result.completed_seeds,
                abstained_seeds=investigator_result.abstained_seeds,
                executed_tool_calls=investigator_result.executed_tool_calls,
                executed_searches=investigator_result.executed_searches,
                executed_reads=investigator_result.executed_reads,
                recoverable_rejections=(
                    investigator_result.recoverable_rejections
                ),
            )
        except (AttributeError, TypeError, ValueError):
            raise ValueError("completed investigator result is invalid") from None
        if checked_result.outcome != "completed" or checked_result.reason_code != "completed":
            raise ValueError("investigator result is not completed")
        try:
            envelopes = tuple(
                _clone_envelope(row, max_payload_bytes=16_384)[0]
                for row in checked_result.envelopes
            )
            bindings = tuple(
                _clone_authorized_binding(row, self._documents)
                for row in checked_result.authorized_candidates
            )
        except (_CandidateRejected, AttributeError, TypeError, ValueError):
            raise ValueError("authorized candidate bindings are invalid") from None
        keys = [
            (row.seed_ref, row.candidate_payload, row.authorized_span_sha256)
            for row in bindings
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("authorized candidate bindings contain duplicates")
        expected = tuple(
            (envelope.seed_ref, payload, span_hash)
            for envelope in envelopes
            for payload, span_hash in zip(
                envelope.candidate_payloads,
                envelope.authorized_span_hashes,
                strict=True,
            )
        )
        if tuple(keys) != expected:
            raise ValueError("authorized candidate bindings do not match envelopes")
        self._envelopes = envelopes
        self._bindings = dict(zip(keys, bindings, strict=True))

    @property
    def run_hash(self) -> str:
        return self._scope.run_hash

    @property
    def envelopes(self) -> tuple[UntrustedCandidateEnvelope, ...]:
        """Return detached envelopes from the exact completed loop result."""

        return tuple(
            UntrustedCandidateEnvelope(
                seed_ref=row.seed_ref,
                candidate_payloads=tuple(row.candidate_payloads),
                authorized_span_hashes=tuple(row.authorized_span_hashes),
            )
            for row in self._envelopes
        )

    def resolve(
        self,
        *,
        seed_ref: str,
        candidate: CandidateRecordSubmission,
        expected_span_hash: str,
    ) -> ResolvedCandidateEvidence:
        if type(candidate) is not CandidateRecordSubmission:
            raise _CandidateRejected("candidate_evidence_unauthorized")
        payload = _canonical_json(candidate.model_dump(mode="json", warnings="error"))
        binding = self._bindings.get((seed_ref, payload, expected_span_hash))
        if binding is None or binding.span_ref != candidate.span_ref:
            raise _CandidateRejected("candidate_evidence_unauthorized")
        document = next(
            (
                row
                for snapshot, row in self._documents.items()
                if snapshot.project_id == binding.snapshot_project_id
                and snapshot.document_id == binding.snapshot_document_id
                and snapshot.document_version == binding.snapshot_document_version
                and snapshot.content_sha256 == binding.snapshot_content_sha256
            ),
            None,
        )
        if document is None:
            raise _CandidateRejected("candidate_evidence_unauthorized")
        context = self._contexts[binding.snapshot_document_id]
        evidence = EvidenceSpan(
            document_id=binding.snapshot_document_id,
            document_name=context.document_name,
            line_start=binding.line_start,
            line_end=binding.line_end,
            text=binding.text,
        )
        return ResolvedCandidateEvidence(
            evidence=evidence.model_copy(deep=True),
            document_content=document.content,
            story_scope=context.story_scope,
            document_role=context.document_role,
        )

    def binds_anchor(self, evidence: EvidenceSpan) -> bool:
        """Confirm a seed anchor belongs to this exact closed snapshot."""

        if type(evidence) is not EvidenceSpan:
            return False
        matching = [
            document
            for document in self._scope.documents
            if document.snapshot.document_id == evidence.document_id
        ]
        if len(matching) != 1:
            return False
        context = self._contexts.get(evidence.document_id)
        if context is None or evidence.document_name != context.document_name:
            return False
        document = matching[0]
        lines = document.lines
        if (
            type(evidence.line_start) is not int
            or type(evidence.line_end) is not int
            or evidence.line_start < 1
            or evidence.line_end < evidence.line_start
            or evidence.line_end > len(lines)
            or type(evidence.text) is not str
        ):
            return False
        source = "\n".join(
            lines[evidence.line_start - 1 : evidence.line_end]
        ).strip()
        if source == evidence.text:
            return True
        if evidence.line_start == evidence.line_end and source.startswith("@") and "|" in source:
            declared = source.partition("|")[2].strip() or source
            return declared == evidence.text
        return False


@dataclass(frozen=True, slots=True)
class CandidatePromotionResult:
    """Additive result: baseline rows stay first and are never rewritten."""

    directives: tuple[ParsedDirective, ...] = field(repr=False)
    issues: tuple[ConsistencyIssue, ...] = field(repr=False)
    promoted_directives: tuple[ParsedDirective, ...] = field(repr=False)
    added_issues: tuple[ConsistencyIssue, ...] = field(repr=False)
    submitted_candidates: int
    accepted_candidates: int
    rejection_counts: tuple[tuple[str, int], ...]

    def safe_dict(self) -> dict[str, object]:
        return {
            "submitted_candidates": _safe_count(self.submitted_candidates),
            "accepted_candidates": _safe_count(self.accepted_candidates),
            "added_issues": min(
                sum(type(row) is ConsistencyIssue for row in self.added_issues), 256
            )
            if type(self.added_issues) is tuple
            else 0,
            "rejection_counts": {
                reason: _safe_count(count)
                for reason, count in self.rejection_counts
                if reason in _REJECTION_REASONS
            }
            if type(self.rejection_counts) is tuple
            else {},
            "boundary": (
                "Candidates are additive only after authorized evidence, semantic "
                "quality, and deterministic-rule validation."
            ),
        }


@dataclass(frozen=True, slots=True)
class _RuleIndex:
    by_join: Mapping[
        tuple[IssueCategory, str], tuple[ParsedDirective, ...]
    ] = field(repr=False)
    mobility_permissions: Mapping[str, tuple[ParsedDirective, ...]] = field(
        repr=False
    )
    mobility_limits: Mapping[str, tuple[ParsedDirective, ...]] = field(
        repr=False
    )
    world_support_by_subject: Mapping[
        str, tuple[ParsedDirective, ...]
    ] = field(repr=False)
    world_support_by_key: Mapping[str, tuple[ParsedDirective, ...]] = field(
        repr=False
    )
    source_order: Mapping[str, int] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _KindSchema:
    required: frozenset[str]
    optional: frozenset[str]
    field_byte_limits: Mapping[str, int]

    @property
    def allowed(self) -> frozenset[str]:
        return self.required | self.optional


_KIND_FIELD_BYTE_LIMITS: dict[str, dict[str, int]] = {
    "fact": {"subject": 128, "predicate": 128, "value": 512, "time": 64},
    "event": {"time": 64, "location": 256, "participants": 512},
    "knows": {"character": 128, "fact": 512, "time": 64},
    "claims_knows": {"character": 128, "fact": 512, "time": 64},
    "item": {"item": 256, "owner": 128, "time": 64},
    "uses": {"item": 256, "user": 128, "time": 64},
    "world_rule": {"key": 384, "value": 256},
    "world_assert": {"key": 384, "value": 256, "actor": 128, "time": 64},
}
if set(_KIND_FIELD_BYTE_LIMITS) != set(candidate_kinds()):
    raise RuntimeError("candidate field limits do not cover candidate kinds")


def _kind_schema(kind: str) -> _KindSchema:
    contract = get_candidate_field_contract(kind)
    field_byte_limits = _KIND_FIELD_BYTE_LIMITS[kind]
    if set(field_byte_limits) != set(contract.required + contract.optional):
        raise RuntimeError("candidate field limits do not match shared contract")
    return _KindSchema(
        frozenset(contract.required),
        frozenset(contract.optional),
        field_byte_limits,
    )


_KIND_SCHEMAS: dict[str, _KindSchema] = {
    kind: _kind_schema(kind) for kind in _KIND_FIELD_BYTE_LIMITS
}

_REJECTION_REASONS = frozenset(
    {
        "invalid_envelope",
        "unknown_seed",
        "candidate_budget",
        "candidate_payload_too_large",
        "candidate_fields_forbidden",
        "candidate_shape_invalid",
        "candidate_not_grounded",
        "candidate_join_mismatch",
        "candidate_evidence_unauthorized",
        "candidate_evidence_reused",
        "candidate_semantics_rejected",
        "candidate_no_rule_conflict",
        "candidate_duplicate",
    }
)


def promote_investigator_candidates(
    *,
    baseline_directives: Sequence[ParsedDirective],
    baseline_issues: Sequence[ConsistencyIssue],
    seeds: Sequence[InvestigationSeed],
    evidence_resolver: CandidateEvidenceResolver,
    envelopes: Sequence[UntrustedCandidateEnvelope] | None = None,
    limits: CandidatePromotionLimits | None = None,
    checkpoint: Callable[[], None] | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> CandidatePromotionResult:
    """Promote model proposals only when existing rules reproduce a conflict.

    A proposal is evaluated against the immutable baseline in isolation. This
    prevents two model-created records from manufacturing their own conflict.
    Baseline issues are copied byte-for-byte (including their existing IDs),
    remain first, and can only be followed by newly validated issues.
    """

    prepared_limits = _copy_limits(limits or CandidatePromotionLimits())
    if checkpoint is not None and not callable(checkpoint):
        raise TypeError("candidate promotion checkpoint is invalid")
    if not callable(monotonic):
        raise TypeError("candidate promotion clock is invalid")
    started_at = _clock_value(monotonic)

    def cancelled_or_expired() -> bool:
        if checkpoint is not None:
            # Cancellation and lease-loss exceptions are control flow and must
            # reach the caller rather than being converted into a rejection.
            checkpoint()
        return (
            _clock_value(monotonic) - started_at
        ) * 1_000 >= prepared_limits.max_elapsed_ms

    if type(evidence_resolver) is not CandidateEvidenceResolver:
        raise TypeError("candidate evidence resolver is invalid")
    baseline = _clone_directives(baseline_directives)
    original_issues = _clone_issues(baseline_issues)
    prepared_seeds = _clone_seeds(seeds, evidence_resolver.run_hash)
    seed_by_ref = {row.seed_ref: row for row in prepared_seeds}
    if len(seed_by_ref) != len(prepared_seeds):
        raise ValueError("candidate seeds contain duplicates")
    baseline_keys = {_directive_identity(row) for row in baseline}
    if any(
        _directive_identity(seed.anchor) not in baseline_keys
        or not evidence_resolver.binds_anchor(seed.anchor.evidence)
        for seed in prepared_seeds
    ):
        raise ValueError("candidate seed anchor is absent from the baseline")

    trusted_envelopes = evidence_resolver.envelopes
    if envelopes is None:
        supplied_envelopes = trusted_envelopes
    else:
        supplied_envelopes = _bounded_sequence(
            envelopes,
            maximum=prepared_limits.max_envelopes,
            message="candidate envelope count is invalid",
        )
        try:
            checked_envelopes = tuple(
                _clone_envelope(
                    row,
                    max_payload_bytes=prepared_limits.max_candidate_payload_bytes,
                )[0]
                for row in supplied_envelopes
            )
        except _CandidateRejected:
            raise ValueError("candidate envelopes are not from the loop result") from None
        if checked_envelopes != trusted_envelopes:
            raise ValueError("candidate envelopes are not from the loop result")
        supplied_envelopes = checked_envelopes
    rejection_counts: Counter[str] = Counter()
    submitted = 0
    total_payload_bytes = 0
    prepared_candidates: list[
        tuple[
            str,
            str,
            InvestigationSeed,
            CandidateRecordSubmission,
            ResolvedCandidateEvidence,
        ]
    ] = []

    # Preflight raw strings before parsing Pydantic payloads. This keeps the
    # semantic boundary from allocating an attacker-controlled JSON tree.
    for supplied_envelope in supplied_envelopes:
        try:
            envelope, payload_bytes = _clone_envelope(
                supplied_envelope,
                max_payload_bytes=prepared_limits.max_candidate_payload_bytes,
            )
        except _CandidateRejected as exc:
            rejection_counts[exc.reason] += 1
            continue
        total_payload_bytes += payload_bytes
        if total_payload_bytes > prepared_limits.max_total_payload_bytes:
            rejection_counts["candidate_budget"] += len(envelope.candidate_payloads)
            continue
        seed = seed_by_ref.get(envelope.seed_ref)
        if seed is None:
            rejection_counts["unknown_seed"] += len(envelope.candidate_payloads)
            continue
        for payload, candidate, span_hash in zip(
            envelope.candidate_payloads,
            envelope.candidates,
            envelope.authorized_span_hashes,
            strict=True,
        ):
            submitted += 1
            if submitted > prepared_limits.max_candidates:
                rejection_counts["candidate_budget"] += 1
                continue
            try:
                resolved = evidence_resolver.resolve(
                    seed_ref=seed.seed_ref,
                    candidate=candidate,
                    expected_span_hash=span_hash,
                )
            except _CandidateRejected as exc:
                rejection_counts[exc.reason] += 1
                continue
            prepared_candidates.append(
                (seed.seed_ref, payload, seed, candidate, resolved)
            )

    # Opaque capability refs are intentionally random.  Sort on semantic
    # fields and trusted source coordinates instead, so a repeated run chooses
    # the same candidate even when freshly minted refs differ.
    prepared_candidates.sort(key=_stable_prepared_candidate_key)
    known_issue_keys = {_issue_identity(row) for row in original_issues}
    wanted_joins = {
        (seed.family, seed.join_key_hash) for seed in prepared_seeds
    }
    rule_index, index_expired = _build_rule_index(
        baseline,
        wanted_joins=wanted_joins,
        cancelled_or_expired=cancelled_or_expired,
    )
    if index_expired:
        baseline_ranges = {}
        ranges_expired = False
    else:
        baseline_ranges, ranges_expired = _build_evidence_range_index(
            baseline,
            cancelled_or_expired=cancelled_or_expired,
        )
    timed_out = index_expired or ranges_expired
    pre_evaluation_rejections = rejection_counts.copy()
    seen_candidate_payloads: set[str] = set()
    accepted_seed_refs: set[str] = set()
    promoted: list[ParsedDirective] = []
    added: list[ConsistencyIssue] = []

    for (
        seed_ref,
        payload,
        seed,
        candidate,
        resolved,
    ) in prepared_candidates:
        if timed_out:
            break
        if cancelled_or_expired():
            timed_out = True
            break
        payload_key = _sha256(payload)
        if payload_key in seen_candidate_payloads or seed_ref in accepted_seed_refs:
            rejection_counts["candidate_duplicate"] += 1
            continue
        seen_candidate_payloads.add(payload_key)
        try:
            attrs = _validate_candidate_fields(candidate, prepared_limits)
            _require_seed_join(seed, candidate.kind, attrs)
            evidence = resolved.evidence
            # A candidate is a proposal for a *missing* record.  Reusing any
            # baseline source range can turn two readings of one sentence into
            # a synthetic contradiction (notably when an @directive anchor
            # stores only the text after ``|``).  Range overlap, rather than a
            # text hash, is therefore the authority boundary.
            if _evidence_range_is_reused(evidence, baseline_ranges):
                raise _CandidateRejected("candidate_evidence_reused")
            if not _fields_are_grounded(candidate.kind, attrs, evidence.text):
                raise _CandidateRejected("candidate_not_grounded")
            server_attrs = dict(attrs)
            if resolved.story_scope:
                server_attrs["story_scope"] = resolved.story_scope
            if resolved.document_role:
                server_attrs["document_role"] = resolved.document_role
            raw = ParsedDirective(
                kind=candidate.kind,
                attrs=server_attrs,
                evidence=evidence,
                noncanonical_frame=document_context_has_noncanonical_frame(
                    resolved.document_content,
                    evidence.line_start,
                    evidence.line_end,
                ),
                provenance_sources=frozenset({"model"}),
            )
            quality = apply_semantic_quality_gate([raw])
            if len(quality.directives) != 1:
                raise _CandidateRejected("candidate_semantics_rejected")
            promoted_candidate = quality.directives[0]
            if (
                promoted_candidate.kind != candidate.kind
                or not eligible_for_deterministic_rules(promoted_candidate)
            ):
                raise _CandidateRejected("candidate_semantics_rejected")
            _require_seed_join(seed, promoted_candidate.kind, promoted_candidate.attrs)
            directive_key = _directive_identity(promoted_candidate)
            if directive_key in baseline_keys:
                raise _CandidateRejected("candidate_duplicate")
            rule_context = _minimal_rule_context(
                seed,
                promoted_candidate,
                rule_index,
                maximum=prepared_limits.max_rule_context,
            )
            if rule_context is None:
                raise _CandidateRejected("candidate_budget")
            reproduced = [
                row
                for row in detect_issues([*rule_context, promoted_candidate])
                if row.category == seed.family
                and _issue_matches_seed(row, seed)
                and _issue_contains_evidence(row, promoted_candidate.evidence)
                and _issue_identity(row) not in known_issue_keys
            ]
            if not reproduced:
                raise _CandidateRejected("candidate_no_rule_conflict")
            reproduced.sort(key=_issue_identity)
            issue = _with_stable_id(reproduced[0])
            issue_key = _issue_identity(issue)
            if issue_key in known_issue_keys:
                raise _CandidateRejected("candidate_duplicate")
        except _CandidateRejected as exc:
            rejection_counts[exc.reason] += 1
            continue
        except (AttributeError, TypeError, ValueError):
            rejection_counts["candidate_shape_invalid"] += 1
            continue

        promoted.append(_clone_directive(promoted_candidate))
        added.append(_clone_issue(issue))
        baseline_keys.add(directive_key)
        known_issue_keys.add(issue_key)
        accepted_seed_refs.add(seed_ref)

    if not timed_out and cancelled_or_expired():
        timed_out = True
    if timed_out:
        # A wall-clock cut must never expose a scheduler-dependent partial
        # promotion.  Discard every candidate-side decision atomically and
        # preserve only the immutable baseline plus preflight diagnostics.
        rejection_counts = pre_evaluation_rejections
        rejection_counts["candidate_budget"] += len(prepared_candidates)
        promoted.clear()
        added.clear()

    promoted.sort(key=_directive_identity)
    added.sort(key=_issue_identity)
    final_directives = (*baseline, *tuple(_clone_directive(row) for row in promoted))
    final_issues = (*original_issues, *tuple(_clone_issue(row) for row in added))
    return CandidatePromotionResult(
        directives=final_directives,
        issues=final_issues,
        promoted_directives=tuple(_clone_directive(row) for row in promoted),
        added_issues=tuple(_clone_issue(row) for row in added),
        submitted_candidates=submitted,
        accepted_candidates=len(promoted),
        rejection_counts=tuple(sorted(rejection_counts.items())),
    )


def _stable_prepared_candidate_key(
    row: tuple[
        str,
        str,
        InvestigationSeed,
        CandidateRecordSubmission,
        ResolvedCandidateEvidence,
    ],
) -> tuple[object, ...]:
    seed_ref, _payload, _seed, candidate, resolved = row
    evidence = resolved.evidence
    return (
        seed_ref,
        evidence.document_id,
        evidence.line_start,
        evidence.line_end,
        candidate.kind,
        _canonical_json(candidate.fields),
        _sha256(evidence.text),
    )


def _build_rule_index(
    baseline: tuple[ParsedDirective, ...],
    *,
    wanted_joins: set[tuple[IssueCategory, str]],
    cancelled_or_expired: Callable[[], bool],
) -> tuple[_RuleIndex, bool]:
    """Build a bounded-by-baseline lookup once, never once per candidate."""

    by_join: dict[tuple[IssueCategory, str], list[ParsedDirective]] = defaultdict(list)
    mobility_permissions: dict[str, list[ParsedDirective]] = defaultdict(list)
    mobility_limits: dict[str, list[ParsedDirective]] = defaultdict(list)
    world_support_by_subject: dict[str, list[ParsedDirective]] = defaultdict(list)
    world_support_by_key: dict[str, list[ParsedDirective]] = defaultdict(list)
    seen: dict[tuple[str, object], set[str]] = defaultdict(set)
    source_order: dict[str, int] = {}
    expired = False

    def append_unique(
        namespace: str,
        bucket_key: object,
        bucket: list[ParsedDirective],
        directive: ParsedDirective,
    ) -> None:
        identity = _rule_semantic_identity(directive)
        marker = (namespace, bucket_key)
        if identity in seen[marker]:
            return
        seen[marker].add(identity)
        bucket.append(directive)

    for position, directive in enumerate(baseline):
        if position % 128 == 0 and cancelled_or_expired():
            expired = True
            break
        if not eligible_for_deterministic_rules(directive):
            continue
        semantic_identity = _rule_semantic_identity(directive)
        source_order.setdefault(semantic_identity, position)
        for join in _directive_rule_joins(directive):
            if join in wanted_joins:
                append_unique("join", join, by_join[join], directive)

        attrs = directive.attrs
        if directive.kind == "fact":
            predicate = attrs.get("predicate", "")
            subject = attrs.get("subject", "")
            if predicate == "mobility_permission" and subject:
                append_unique(
                    "permission",
                    subject,
                    mobility_permissions[subject],
                    directive,
                )
            elif predicate == "mobility_limit" and subject:
                append_unique(
                    "limit", subject, mobility_limits[subject], directive
                )
            if _is_world_rule_support(directive):
                if subject:
                    append_unique(
                        "world_subject",
                        subject,
                        world_support_by_subject[subject],
                        directive,
                    )
                key = attrs.get("key", "")
                if key:
                    append_unique(
                        "world_key",
                        key,
                        world_support_by_key[key],
                        directive,
                    )

    def freeze(
        values: Mapping[object, list[ParsedDirective]],
    ) -> dict[object, tuple[ParsedDirective, ...]]:
        return {key: tuple(rows) for key, rows in values.items()}

    return (
        _RuleIndex(
            by_join=freeze(by_join),
            mobility_permissions=freeze(mobility_permissions),
            mobility_limits=freeze(mobility_limits),
            world_support_by_subject=freeze(world_support_by_subject),
            world_support_by_key=freeze(world_support_by_key),
            source_order=dict(source_order),
        ),
        expired,
    )


def _directive_rule_joins(
    directive: ParsedDirective,
) -> tuple[tuple[IssueCategory, str], ...]:
    attrs = directive.attrs
    kind = directive.kind
    try:
        if kind == "fact":
            if attrs.get("predicate") in {
                "mobility_permission",
                "mobility_limit",
                "rule_exception",
            }:
                return ()
            keys = _candidate_join_keys(kind, attrs)
            family = IssueCategory.fact_conflict
        elif kind == "event":
            keys = _candidate_join_keys(kind, attrs)
            family = IssueCategory.location_collision
        elif kind in {"knows", "claims_knows"}:
            keys = _candidate_join_keys(kind, attrs)
            family = IssueCategory.knowledge_without_acquisition
        elif kind in {"item", "uses"}:
            keys = _candidate_join_keys(kind, attrs)
            family = IssueCategory.item_ownership
        elif kind in {"world_rule", "world_assert"}:
            keys = _candidate_join_keys(kind, attrs)
            family = IssueCategory.world_rule_conflict
        else:
            return ()
    except (KeyError, TypeError, AttributeError):
        return ()
    return tuple((family, _sha256(key)) for key in keys if key)


def _minimal_rule_context(
    seed: InvestigationSeed,
    candidate: ParsedDirective,
    index: _RuleIndex,
    *,
    maximum: int,
) -> tuple[ParsedDirective, ...] | None:
    """Return the smallest baseline closure needed by ``detect_issues``.

    Returning ``None`` is a safe resource rejection.  An empty tuple means the
    candidate has no baseline record on its trusted join and therefore cannot
    reproduce a deterministic conflict.
    """

    rows = list(index.by_join.get((seed.family, seed.join_key_hash), ()))
    attrs = candidate.attrs

    if seed.family == IssueCategory.location_collision:
        participants = tuple(
            value.strip()
            for value in attrs.get("participants", "").split(",")
            if value.strip()
        )
        for participant in participants:
            participant_join = json.dumps(
                ["event", participant, attrs.get("time", "")],
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            if _sha256(participant_join) != seed.join_key_hash:
                continue
            for subject in (
                participant,
                "*",
                "任何人",
                "普通人",
                "所有人",
            ):
                rows.extend(index.mobility_permissions.get(subject, ()))
                rows.extend(index.mobility_limits.get(subject, ()))
    elif seed.family == IssueCategory.world_rule_conflict:
        key = attrs.get("key", "")
        rows.extend(index.world_support_by_key.get(key, ()))
        actors = {attrs.get("actor", "")}
        actors.update(
            row.attrs.get("actor", "")
            for row in rows
            if row.kind == "world_assert"
        )
        for actor in sorted(actor for actor in actors if actor):
            rows.extend(index.world_support_by_subject.get(actor, ()))

    unique: dict[str, ParsedDirective] = {}
    for row in rows:
        unique.setdefault(_rule_semantic_identity(row), row)
    ordered = sorted(
        unique.values(),
        key=lambda row: index.source_order.get(
            _rule_semantic_identity(row), 100_000_000
        ),
    )
    if len(ordered) + 1 > maximum:
        return None
    return tuple(ordered)


def _rule_semantic_identity(value: ParsedDirective) -> str:
    """Identity for rule behavior; source coordinates do not change behavior."""

    return _sha256(
        _canonical_json(
            {
                "kind": value.kind,
                "attrs": sorted(value.attrs.items()),
                "evidence_text_sha256": _sha256(value.evidence.text),
                "noncanonical_frame": value.noncanonical_frame,
            }
        )
    )


def _is_world_rule_support(value: ParsedDirective) -> bool:
    attrs = value.attrs
    if attrs.get("predicate") == "rule_exception":
        return True
    if attrs.get("polarity") != "negative" and attrs.get("modality") != "negated":
        return False
    return bool(
        re.search(
            r"书面授权|授权|许可|豁免|批准|通行证|资格",
            " ".join(
                (
                    attrs.get("predicate", ""),
                    attrs.get("value", ""),
                    value.evidence.text,
                )
            ),
        )
    )


def _issue_matches_seed(issue: ConsistencyIssue, seed: InvestigationSeed) -> bool:
    metadata = issue.metadata
    try:
        if issue.category == IssueCategory.fact_conflict:
            keys = (
                json.dumps(
                    ["fact", metadata["subject"], metadata["predicate"]],
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
            )
        elif issue.category == IssueCategory.location_collision:
            keys = (
                json.dumps(
                    ["event", metadata["participant"], metadata["timestamp"]],
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
            )
        elif issue.category == IssueCategory.knowledge_without_acquisition:
            keys = (
                json.dumps(
                    ["knowledge", metadata["character"], metadata["fact"]],
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
            )
        elif issue.category == IssueCategory.item_ownership:
            keys = (
                json.dumps(
                    ["item", metadata["item"]],
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
            )
        elif issue.category == IssueCategory.world_rule_conflict:
            keys = (
                json.dumps(
                    ["world_rule", metadata["key"]],
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ),
            )
        else:
            return False
    except (KeyError, TypeError, ValueError):
        return False
    return any(_sha256(key) == seed.join_key_hash for key in keys)


def _build_evidence_range_index(
    baseline: tuple[ParsedDirective, ...],
    *,
    cancelled_or_expired: Callable[[], bool],
) -> tuple[dict[str, tuple[tuple[int, ...], tuple[int, ...]]], bool]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for position, directive in enumerate(baseline):
        if position % 128 == 0 and cancelled_or_expired():
            return {}, True
        evidence = directive.evidence
        grouped[evidence.document_id].append(
            (evidence.line_start, evidence.line_end)
        )
    result: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
    for document_id, ranges in grouped.items():
        merged: list[list[int]] = []
        for start, end in sorted(ranges):
            if merged and start <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        result[document_id] = (
            tuple(row[0] for row in merged),
            tuple(row[1] for row in merged),
        )
    return result, False


def _evidence_range_is_reused(
    evidence: EvidenceSpan,
    index: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]],
) -> bool:
    bucket = index.get(evidence.document_id)
    if bucket is None:
        return False
    starts, ends = bucket
    position = bisect_right(starts, evidence.line_end) - 1
    return position >= 0 and ends[position] >= evidence.line_start


def _clock_value(clock: Callable[[], float]) -> float:
    value = clock()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("candidate promotion clock is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise TypeError("candidate promotion clock is invalid")
    return result


class _CandidateRejected(ValueError):
    def __init__(self, reason: PromotionReason) -> None:
        super().__init__(reason)
        self.reason = reason


def _clone_envelope(
    value: object, *, max_payload_bytes: int
) -> tuple[UntrustedCandidateEnvelope, int]:
    if type(value) is not UntrustedCandidateEnvelope or value.trusted is not False:
        raise _CandidateRejected("invalid_envelope")
    try:
        seed_ref = value.seed_ref
        payloads = value.candidate_payloads
        hashes = value.authorized_span_hashes
    except (AttributeError, TypeError, ValueError):
        raise _CandidateRejected("invalid_envelope") from None
    if (
        type(seed_ref) is not str
        or re.fullmatch(SEED_REF_PATTERN, seed_ref) is None
        or type(payloads) is not tuple
        or not 1 <= len(payloads) <= 2
        or type(hashes) is not tuple
        or len(hashes) != len(payloads)
    ):
        raise _CandidateRejected("invalid_envelope")
    total = 0
    for payload in payloads:
        if type(payload) is not str:
            raise _CandidateRejected("invalid_envelope")
        size = len(payload.encode("utf-8"))
        if size > max_payload_bytes:
            raise _CandidateRejected("candidate_payload_too_large")
        total += size
    if any(type(value) is not str or HASH_PATTERN.fullmatch(value) is None for value in hashes):
        raise _CandidateRejected("invalid_envelope")
    try:
        return (
            UntrustedCandidateEnvelope(
                seed_ref=seed_ref,
                candidate_payloads=tuple(payloads),
                authorized_span_hashes=tuple(hashes),
            ),
            total,
        )
    except (AttributeError, TypeError, ValueError):
        raise _CandidateRejected("invalid_envelope") from None


def _validate_candidate_fields(
    candidate: CandidateRecordSubmission,
    limits: CandidatePromotionLimits,
) -> dict[str, str]:
    if type(candidate) is not CandidateRecordSubmission:
        raise _CandidateRejected("candidate_shape_invalid")
    schema = _KIND_SCHEMAS.get(candidate.kind)
    if schema is None or set(candidate.fields) - schema.allowed:
        raise _CandidateRejected("candidate_fields_forbidden")
    if not schema.required.issubset(candidate.fields):
        raise _CandidateRejected("candidate_shape_invalid")
    if _json_node_count(candidate.fields, limit=limits.max_json_nodes) > limits.max_json_nodes:
        raise _CandidateRejected("candidate_shape_invalid")
    attrs: dict[str, str] = {}
    total_bytes = 0
    for key, value in candidate.fields.items():
        if type(value) is not str:
            raise _CandidateRejected("candidate_shape_invalid")
        if (
            not value
            or value != value.strip()
            or "\n" in value
            or "\r" in value
            or _CONTROL_CHARACTER.search(value)
        ):
            raise _CandidateRejected("candidate_shape_invalid")
        size = len(value.encode("utf-8"))
        if size > min(
            limits.max_field_value_bytes,
            schema.field_byte_limits.get(key, limits.max_field_value_bytes),
        ):
            raise _CandidateRejected("candidate_shape_invalid")
        total_bytes += len(key.encode("ascii")) + size
        attrs[key] = value
    if total_bytes > limits.max_total_field_bytes:
        raise _CandidateRejected("candidate_shape_invalid")
    _validate_kind_shape(candidate.kind, attrs)
    return attrs


def _validate_kind_shape(kind: str, attrs: dict[str, str]) -> None:
    if "time" in attrs and not _valid_precise_time(attrs["time"]):
        # The deterministic rules compare timestamps lexicographically.  Only
        # a validated sortable representation may cross this promotion seam;
        # natural-language times such as “清晨/正午” would invert causality.
        raise _CandidateRejected("candidate_shape_invalid")
    if kind == "event":
        participants = [
            value.strip()
            for value in re.split(r"[,，]", attrs["participants"])
            if value.strip()
        ]
        if (
            not participants
            or len(participants) > 16
            or len(participants) != len(set(participants))
        ):
            raise _CandidateRejected("candidate_shape_invalid")
        attrs["participants"] = ",".join(participants)
    elif kind in {"world_rule", "world_assert"}:
        if _WORLD_KEY.fullmatch(attrs["key"]) is None:
            raise _CandidateRejected("candidate_shape_invalid")
        if attrs["key"].startswith("scope_action:"):
            allowed_values = (
                {"disabled", "allowed"}
                if kind == "world_rule"
                else {"performed", "disabled", "allowed", "denied"}
            )
            if attrs["value"] not in allowed_values:
                raise _CandidateRejected("candidate_shape_invalid")
    elif kind == "fact" and attrs["predicate"] in {
        "mobility_permission",
        "mobility_limit",
        "rule_exception",
    }:
        # These state records have wider schemas and are not fact-conflict
        # proposals. A later dedicated contract must validate them separately.
        raise _CandidateRejected("candidate_fields_forbidden")


def _require_seed_join(
    seed: InvestigationSeed, kind: str, attrs: Mapping[str, str]
) -> None:
    if kind not in seed.allowed_candidate_kinds:
        raise _CandidateRejected("candidate_join_mismatch")
    hashes = {_sha256(value) for value in _candidate_join_keys(kind, attrs)}
    if seed.join_key_hash not in hashes:
        raise _CandidateRejected("candidate_join_mismatch")


def _candidate_join_keys(kind: str, attrs: Mapping[str, str]) -> tuple[str, ...]:
    def key(prefix: str, *parts: str) -> str:
        return json.dumps(
            [prefix, *parts],
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    if kind == "fact":
        return (key("fact", attrs["subject"].strip(), attrs["predicate"].strip()),)
    if kind == "event":
        participants = {
            value.strip()
            for value in attrs["participants"].split(",")
            if value.strip()
        }
        return tuple(key("event", value, attrs["time"].strip()) for value in sorted(participants))
    if kind in {"knows", "claims_knows"}:
        return (key("knowledge", attrs["character"].strip(), attrs["fact"].strip()),)
    if kind in {"item", "uses"}:
        return (key("item", attrs["item"].strip()),)
    if kind in {"world_rule", "world_assert"}:
        return (key("world_rule", attrs["key"].strip()),)
    return ()


def _clone_authorized_binding(
    value: object,
    scope_documents: Mapping[SnapshotDocumentKey, ScopedEvidenceDocument],
) -> _AuthorizedBinding:
    """Rebuild one exact loop-produced binding against the frozen scope."""

    if type(value) is not AuthorizedCandidateBinding:
        raise ValueError("authorized candidate binding is invalid")

    seed_ref = getattr(value, "seed_ref")
    candidate_payload = getattr(value, "candidate_payload")
    span_ref = getattr(value, "span_ref")
    authorized_span_sha256 = getattr(value, "authorized_span_sha256")
    snapshot = getattr(value, "snapshot")
    line_start = getattr(value, "line_start")
    line_end = getattr(value, "line_end")
    char_start = getattr(value, "char_start")
    char_end = getattr(value, "char_end")
    text = getattr(value, "text")
    text_sha256 = getattr(value, "text_sha256")
    authorized_span_char_start = getattr(value, "authorized_span_char_start")
    authorized_span_char_end = getattr(value, "authorized_span_char_end")
    if (
        type(seed_ref) is not str
        or re.fullmatch(SEED_REF_PATTERN, seed_ref) is None
        or type(candidate_payload) is not str
        or len(candidate_payload.encode("utf-8")) > 16_384
        or type(span_ref) is not str
        or type(authorized_span_sha256) is not str
        or HASH_PATTERN.fullmatch(authorized_span_sha256) is None
        or type(snapshot) is not SnapshotDocumentKey
        or type(line_start) is not int
        or type(line_end) is not int
        or line_start < 1
        or line_end < line_start
        or type(char_start) is not int
        or type(char_end) is not int
        or char_start < 0
        or char_end <= char_start
        or type(text) is not str
        or not text.strip()
        or len(text.encode("utf-8")) > 100_000
        or char_end - char_start != len(text)
        or type(text_sha256) is not str
        or _sha256(text) != text_sha256
        or type(authorized_span_char_start) is not int
        or type(authorized_span_char_end) is not int
        or authorized_span_char_start < 0
        or authorized_span_char_end <= authorized_span_char_start
        or char_start < authorized_span_char_start
        or char_end > authorized_span_char_end
    ):
        raise ValueError("authorized candidate binding is invalid")
    candidate = CandidateRecordSubmission.model_validate_json(candidate_payload)
    canonical_payload = _canonical_json(
        candidate.model_dump(mode="json", warnings="error")
    )
    if (
        canonical_payload != candidate_payload
        or candidate.span_ref != span_ref
        or candidate.source_line_start != line_start
        or candidate.source_line_end != line_end
    ):
        raise ValueError("authorized candidate binding is invalid")
    checked_snapshot = SnapshotDocumentKey(
        project_id=snapshot.project_id,
        document_id=snapshot.document_id,
        document_version=snapshot.document_version,
        content_sha256=snapshot.content_sha256,
    )
    document = scope_documents.get(checked_snapshot)
    if document is None:
        raise ValueError("authorized candidate binding is invalid")
    normalized = document.normalized_content
    if (
        authorized_span_char_end > len(normalized)
        or _sha256(
            normalized[authorized_span_char_start:authorized_span_char_end]
        )
        != authorized_span_sha256
        or char_end > len(normalized)
        or normalized[char_start:char_end] != text
        or normalized[:char_start].count("\n") + 1 != line_start
        or line_start + text[:-1].count("\n") != line_end
    ):
        raise ValueError("authorized candidate binding is invalid")
    return _AuthorizedBinding(
        seed_ref=seed_ref,
        candidate_payload=candidate_payload,
        span_ref=span_ref,
        authorized_span_sha256=authorized_span_sha256,
        snapshot_project_id=checked_snapshot.project_id,
        snapshot_document_id=checked_snapshot.document_id,
        snapshot_document_version=checked_snapshot.document_version,
        snapshot_content_sha256=checked_snapshot.content_sha256,
        line_start=line_start,
        line_end=line_end,
        char_start=char_start,
        char_end=char_end,
        text=text,
        text_sha256=text_sha256,
        authorized_span_char_start=authorized_span_char_start,
        authorized_span_char_end=authorized_span_char_end,
    )


def _fields_are_grounded(kind: str, attrs: Mapping[str, str], text: str) -> bool:
    compact = _compact(text)

    def visible(value: str) -> bool:
        return bool(value and _compact(value) in compact)

    if kind == "fact":
        predicate = attrs["predicate"]
        value = attrs["value"]
        predicate_visible = visible(predicate)
        value_visible = visible(value)
        if predicate.startswith("body_state:"):
            body = predicate.partition(":")[2]
            predicate_visible = (
                visible(body)
                or visible(body.replace("上肢", "手"))
                or visible(body.replace("下肢", "腿"))
            )
            canonical_values = {
                "missing_or_prosthetic": r"失去|没有了|截去|截除|义肢|缺失",
                "intact": r"完好|健全|没有受伤",
            }
            if value in canonical_values:
                value_visible = bool(re.search(canonical_values[value], text))
        return (
            visible(attrs["subject"])
            and predicate_visible
            and value_visible
            and _optional_time_visible(attrs, text)
            and _fact_relation_grounded(attrs, text)
        )
    if kind == "event":
        return (
            _time_visible(attrs["time"], text)
            and visible(attrs["location"])
            and all(visible(value) for value in attrs["participants"].split(","))
            and _event_relation_grounded(attrs, text)
        )
    if kind in {"knows", "claims_knows"}:
        return (
            visible(attrs["character"])
            and _knowledge_fact_visible(attrs["fact"], text)
            and _time_visible(attrs["time"], text)
            and _knowledge_relation_grounded(kind, attrs, text)
        )
    if kind == "item":
        return (
            visible(attrs["item"])
            and visible(attrs["owner"])
            and _optional_time_visible(attrs, text)
            and _item_relation_grounded(attrs, text)
        )
    if kind == "uses":
        return (
            visible(attrs["item"])
            and visible(attrs["user"])
            and _optional_time_visible(attrs, text)
            and _use_relation_grounded(attrs, text)
        )
    if kind in {"world_rule", "world_assert"}:
        parts = attrs["key"].split(":", 2)
        if len(parts) != 3 or not all(visible(value) for value in parts[1:]):
            return False
        value = attrs["value"]
        if visible(value):
            visible_value = True
        else:
            aliases = {
                "disabled": r"失效|禁止|不得|不能|无法",
                "performed": r"发动|使用|施展|启动|开启|执行",
                "allowed": r"允许|获准|许可|豁免|可以",
                "denied": r"拒绝|禁止|未获|没有.*(?:授权|许可)",
            }
            visible_value = bool(re.search(aliases.get(value, r"(?!x)x"), text))
        return (
            visible_value
            and _optional_time_visible(attrs, text)
            and ("actor" not in attrs or visible(attrs["actor"]))
            and _world_relation_grounded(kind, attrs, text)
        )
    return False


def _fact_relation_grounded(attrs: Mapping[str, str], text: str) -> bool:
    subject = re.escape(_compact(attrs["subject"]))
    predicate = attrs["predicate"]
    value = attrs["value"]
    # Keep colons inside a sentence.  They are both a genuine relation
    # boundary ("终端亮灯：洛棠的…变成…") and part of modal headings such
    # as "操作计划："; stripping them can concatenate the heading with an
    # entity name and defeat exact subject binding.
    clauses = _relation_grounding_clauses(text)
    for clause in clauses:
        if _reported_or_hypothetical(clause):
            continue
        if predicate.startswith("body_state:"):
            body = _compact(predicate.partition(":")[2])
            body_aliases = {body, body.replace("上肢", "手"), body.replace("下肢", "腿")}
            state_patterns = {
                "missing_or_prosthetic": r"失去|没有了|截去|截除|义肢|缺失",
                "intact": r"完好|健全|没有受伤",
            }
            state = state_patterns.get(value)
            if state and any(
                (match := re.search(
                    rf"{subject}(?:的)?(?:左侧|右侧)?{re.escape(alias)}"
                    rf"(?:已经|仍然|依旧|是|变得|变为|变成)?(?:{state})",
                    clause,
                ))
                and not _externally_modalized(clause, match.start())
                for alias in body_aliases
                if alias
            ):
                return True
            continue
        for match in find_bound_fact_relation_matches(
            clause,
            subject=_compact(attrs["subject"]),
            predicate=_compact(predicate),
            value=_compact(value),
        ):
            if not _externally_modalized(clause, match.start()):
                return True
    return False


def _event_relation_grounded(attrs: Mapping[str, str], text: str) -> bool:
    participants = tuple(_compact(row) for row in attrs["participants"].split(","))
    location = re.escape(_compact(attrs["location"]))
    for clause in _grounding_clauses(text):
        if _non_actual_action(clause) or not all(row in clause for row in participants):
            continue
        marker = re.search(rf"(?:在|位于|抵达|到达|进入){location}", clause)
        participant_start = min(clause.find(row) for row in participants)
        if (
            marker is not None
            and all(clause.find(row) < marker.end() for row in participants)
            and not _externally_modalized(clause, participant_start)
        ):
            return True
    return False


def _knowledge_relation_grounded(
    kind: str, attrs: Mapping[str, str], text: str
) -> bool:
    character = re.escape(_compact(attrs["character"]))
    fact = re.escape(_compact(attrs["fact"]))
    if kind == "knows":
        verbs = KNOWLEDGE_ACQUISITION_VERB_PATTERN
    else:
        verbs = KNOWLEDGE_CLAIM_VERB_PATTERN
    pattern = re.compile(
        rf"{character}(?:才|已|已经|终于|随后|此时|后来)?"
        rf"(?:{verbs})(?:了)?(?:关于)?{fact}"
    )
    for clause in _grounding_clauses(text):
        match = pattern.search(clause)
        if (
            match is not None
            and not _non_actual_action(
                clause, allow_reported=(kind == "claims_knows")
            )
            and not _externally_modalized(clause, match.start())
        ):
            return True
    return False


def _item_relation_grounded(attrs: Mapping[str, str], text: str) -> bool:
    owner = re.escape(_compact(attrs["owner"]))
    item = re.escape(_compact(attrs["item"]))
    patterns = (
        re.compile(
            rf"{owner}(?:已经|当前|仍然|正|正在)?"
            rf"(?:持有|保管|拥有|拿着|携带|获得|接过|收下)(?:了)?{item}"
        ),
        re.compile(rf"{item}(?:由{owner}(?:持有|保管)|归{owner}(?:所有)?)"),
        re.compile(rf"(?:将|把)?{item}(?:交给|递给|归还给){owner}"),
    )
    for clause in _grounding_clauses(text):
        if _non_actual_action(clause):
            continue
        for pattern in patterns:
            match = pattern.search(clause)
            if match is not None and not _externally_modalized(clause, match.start()):
                return True
    return False


def _use_relation_grounded(attrs: Mapping[str, str], text: str) -> bool:
    for clause in _relation_grounding_clauses(text):
        if _non_actual_action(clause):
            continue
        for match in find_bound_use_action_matches(
            clause,
            user=attrs["user"],
            item=attrs["item"],
        ):
            if not _externally_modalized(clause, match.start()):
                return True
    return False


def _world_relation_grounded(
    kind: str, attrs: Mapping[str, str], text: str
) -> bool:
    _, scope_value, action_value = attrs["key"].split(":", 2)
    scope = re.escape(_compact(scope_value))
    action = re.escape(_compact(action_value))
    value = attrs["value"]
    actor_value = attrs.get("actor", "")
    actor = re.escape(_compact(actor_value)) if actor_value else ""
    clauses = tuple(dict.fromkeys((*_grounding_clauses(text), _compact(text))))
    for clause in clauses:
        if kind == "world_rule":
            if _reported_or_hypothetical(clause):
                continue
            if value == "disabled" and (
                re.search(
                    rf"(?:在)?{scope}(?:中|内)?(?:的)?{action}"
                    rf"(?:必然|一律|始终|将会|会)?(?:失效|无法|不能|不得)",
                    clause,
                )
                or re.search(rf"{scope}(?:中|内)?(?:禁止|不得|不能){action}", clause)
            ):
                return True
            if value == "allowed" and re.search(
                rf"{scope}(?:中|内)?.{{0,8}}(?:允许|许可|可以).{{0,8}}{action}",
                clause,
            ):
                return True
            if _compact(value) in clause and re.search(
                rf"{scope}.{{0,12}}{action}.{{0,12}}{re.escape(_compact(value))}",
                clause,
            ):
                return True
            continue

        if value == "performed":
            prefix = (
                rf"{actor}(?:(?:获准|得到许可|获得授权)后)?"
                rf"(?:已经|随后|立即|此时|正|正在)?"
                if actor
                else ""
            )
            if _non_actual_action(clause):
                continue
            patterns = (
                rf"{prefix}(?:在){scope}(?:中|内)?(?:发动|使用|施展|启动|开启|执行)(?:了)?{action}",
                rf"{prefix}(?:发动|使用|施展|启动|开启|执行)(?:了)?{action}.{{0,8}}(?:在){scope}",
            )
            for pattern in patterns:
                match = re.search(pattern, clause)
                if match is not None and not _externally_modalized(
                    clause, match.start()
                ):
                    return True
        elif value == "disabled" and re.search(
            rf"{scope}.{{0,12}}{action}.{{0,8}}(?:失效|无法)", clause
        ):
            return not _reported_or_hypothetical(clause)
        elif value == "allowed" and re.search(
            rf"{scope}.{{0,12}}(?:允许|获准|许可|可以).{{0,8}}{action}", clause
        ):
            return not _reported_or_hypothetical(clause)
        elif value == "denied" and re.search(
            rf"{actor}.{{0,8}}(?:被拒绝|未获|没有)(?:授权|许可)?.{{0,8}}{action}",
            clause,
        ):
            return True
        elif _compact(value) in clause and all(
            value_part in clause
            for value_part in (_compact(scope_value), _compact(action_value))
        ):
            return not _reported_or_hypothetical(clause)
    return False


def _grounding_clauses(text: str) -> tuple[str, ...]:
    # Keep a compact clause boundary so a speaker in one clause cannot become
    # the subject of a relation asserted in the next one.
    return tuple(
        compact
        for row in re.split(r"[\r\n。！？!?；;，,]+", text)
        if (compact := _compact(row))
    )


def _relation_grounding_clauses(text: str) -> tuple[str, ...]:
    """Preserve comma/colon syntax used by bound relation grammars."""

    return tuple(
        compact
        for row in re.split(r"[\r\n。！？!?；;]+", text)
        if (
            compact := re.sub(
                r"\s+", "", unicodedata.normalize("NFKC", row).casefold()
            )
        )
    )


def _reported_or_hypothetical(clause: str) -> bool:
    return bool(
        has_unrealized_heading_frame(clause)
        or re.search(
            r"据说|听说|传闻|声称|表示|认为|猜测|可能|或许|似乎|"
            r"假如|如果|若是|说|提问|询问|用户指南|维护规程|操作说明|"
            r"使用说明|应当|应该|务必",
            clause,
        )
    )


def _non_actual_action(clause: str, *, allow_reported: bool = False) -> bool:
    if re.search(
        r"没有|没在|不在|未在|并不|并未|未曾|尚未|不要|禁止|不得|不能|"
        r"警告|劝阻|阻止",
        clause,
    ):
        return True
    if re.search(r"打算|计划|准备|试图|想要|可能|或许|如果|假如|是否", clause):
        return True
    return not allow_reported and _reported_or_hypothetical(clause)


def _externally_modalized(clause: str, relation_start: int) -> bool:
    if type(relation_start) is not int or relation_start < 0:
        return True
    prefix = clause[max(0, relation_start - 32) : relation_start]
    return bool(
        re.search(
            r"(?:允许|准许|准予|容许|获准|许可|授权|同意|批准|答应|"
            r"授意|命令|指示|吩咐|要求|请求|建议|提议|催促|嘱咐|告诉|提醒|"
            r"让|叫|计划|打算|准备|试图|尝试|即将|将要|尚未|还未|并未|未曾|未能)"
            r"[^后]{0,20}$",
            prefix,
        )
    )


def _knowledge_fact_visible(value: str, text: str) -> bool:
    if _compact(value) in _compact(text):
        return True
    replacements = (("经纬坐标", "位置"), ("坐标", "位置"), ("真实", ""), ("的", ""))
    normalized_value = value
    normalized_text = text
    for old, new in replacements:
        normalized_value = normalized_value.replace(old, new)
        normalized_text = normalized_text.replace(old, new)
    return _compact(normalized_value) in _compact(normalized_text)


def _optional_time_visible(attrs: Mapping[str, str], text: str) -> bool:
    return "time" not in attrs or _time_visible(attrs["time"], text)


def _time_visible(value: str, text: str) -> bool:
    canonical = value.replace("T", "").replace(" ", "")
    source = text.replace("T", "").replace(" ", "")
    return bool(canonical and canonical in source)


def _valid_precise_time(value: str) -> bool:
    if _PRECISE_TIME.fullmatch(value) is None:
        return False
    normalized = value.replace("T", " ")
    pattern = "%Y-%m-%d %H:%M:%S" if len(normalized) == 19 else "%Y-%m-%d %H:%M"
    try:
        datetime.strptime(normalized, pattern)
    except ValueError:
        return False
    return True


def _clone_directives(values: Sequence[ParsedDirective]) -> tuple[ParsedDirective, ...]:
    rows = _bounded_sequence(values, maximum=100_000, message="baseline directives are invalid")
    try:
        return tuple(_clone_directive(row) for row in rows)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("baseline directives are invalid") from None


def _clone_issues(values: Sequence[ConsistencyIssue]) -> tuple[ConsistencyIssue, ...]:
    rows = _bounded_sequence(values, maximum=100_000, message="baseline issues are invalid")
    try:
        return tuple(_clone_issue(row) for row in rows)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("baseline issues are invalid") from None


def _clone_seeds(
    values: Sequence[InvestigationSeed], run_hash: str
) -> tuple[InvestigationSeed, ...]:
    rows = _bounded_sequence(values, maximum=64, message="candidate seeds are invalid")
    if not rows:
        raise ValueError("candidate seeds are invalid")
    try:
        result = tuple(clone_investigation_seed(row) for row in rows)
    except (AttributeError, TypeError, ValueError):
        raise ValueError("candidate seeds are invalid") from None
    if any(row.run_hash != run_hash for row in result):
        raise ValueError("candidate seeds are invalid")
    return result


def _clone_directive(value: ParsedDirective) -> ParsedDirective:
    if type(value) is not ParsedDirective:
        raise ValueError("directive is invalid")
    payload = {
        "kind": value.kind,
        "attrs": value.attrs,
        "evidence": value.evidence.model_dump(mode="json", warnings="error"),
        "noncanonical_frame": value.noncanonical_frame,
        "provenance_sources": sorted(value.provenance_sources),
    }
    return ParsedDirective.model_validate(_json_clone(payload))


def _clone_issue(value: ConsistencyIssue) -> ConsistencyIssue:
    if type(value) is not ConsistencyIssue:
        raise ValueError("issue is invalid")
    return ConsistencyIssue.model_validate(
        _json_clone(value.model_dump(mode="json", warnings="error"))
    )


def _copy_limits(value: CandidatePromotionLimits) -> CandidatePromotionLimits:
    if type(value) is not CandidatePromotionLimits:
        raise TypeError("candidate promotion limits are invalid")
    try:
        return CandidatePromotionLimits(
            max_envelopes=value.max_envelopes,
            max_candidates=value.max_candidates,
            max_candidate_payload_bytes=value.max_candidate_payload_bytes,
            max_total_payload_bytes=value.max_total_payload_bytes,
            max_field_value_bytes=value.max_field_value_bytes,
            max_total_field_bytes=value.max_total_field_bytes,
            max_json_nodes=value.max_json_nodes,
            max_rule_context=value.max_rule_context,
            max_elapsed_ms=value.max_elapsed_ms,
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError("candidate promotion limits are invalid") from None


def _bounded_sequence(values: Sequence[object], *, maximum: int, message: str) -> tuple:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(message)
    try:
        rows = tuple(islice(iter(values), maximum + 1))
    except (TypeError, ValueError):
        raise ValueError(message) from None
    if len(rows) > maximum:
        raise ValueError(message)
    return rows


def _directive_identity(value: ParsedDirective) -> str:
    payload = {
        "kind": value.kind,
        "attrs": sorted(value.attrs.items()),
        "evidence": _evidence_payload(value.evidence),
        "noncanonical_frame": value.noncanonical_frame,
    }
    return _sha256(_canonical_json(payload))


def _issue_identity(value: ConsistencyIssue) -> str:
    payload = {
        "category": value.category.value,
        "evidence": sorted(
            (_evidence_payload(row) for row in value.evidence),
            key=_canonical_json,
        ),
    }
    return _sha256(_canonical_json(payload))


def _evidence_payload(value: EvidenceSpan) -> dict[str, object]:
    return {
        "document_id": value.document_id,
        "document_name": value.document_name,
        "line_start": value.line_start,
        "line_end": value.line_end,
        "text_sha256": _sha256(value.text),
    }


def _evidence_identity(value: EvidenceSpan) -> str:
    return _sha256(_canonical_json(_evidence_payload(value)))


def _issue_contains_evidence(issue: ConsistencyIssue, evidence: EvidenceSpan) -> bool:
    expected = _evidence_identity(evidence)
    return any(_evidence_identity(row) == expected for row in issue.evidence)


def _with_stable_id(issue: ConsistencyIssue) -> ConsistencyIssue:
    return issue.model_copy(
        update={"id": uuid5(_PROMOTED_ISSUE_NAMESPACE, _issue_identity(issue))},
        deep=True,
    )


def _json_node_count(value: object, *, limit: int) -> int:
    count = 0
    stack = [value]
    while stack:
        current = stack.pop()
        count += 1
        if count > limit:
            return count
        if type(current) is dict:
            for key, item in current.items():
                stack.extend((key, item))
        elif type(current) in {list, tuple}:
            stack.extend(current)
    return count


def _compact(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[\s，,。；;：:\"'“”‘’（）()【】\[\]]+", "", normalized)


def _json_clone(value: object) -> object:
    return json.loads(_canonical_json(value))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_count(value: object) -> int:
    return value if type(value) is int and 0 <= value <= 1_000_000 else 0
