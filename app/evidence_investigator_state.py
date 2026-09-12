from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from .evidence_authority import (
    EvidenceGrantAuthority,
    InvestigationScope,
    ReadObservation,
    SearchObservation,
    SpanGrant,
    clone_evidence_chunks,
    clone_investigation_scope,
)
from .evidence_chunks import EvidenceChunk
from .evidence_investigator import (
    HASH_PATTERN,
    MAX_LINE_NUMBER,
    SAFE_REASON_CODES,
    SEED_REF_PATTERN,
    AbstainArgs,
    CandidateRecordSubmission,
    InvestigationSeed,
    InvestigatorRejected,
    ReadSpanArgs,
    SearchEvidenceArgs,
    SubmitVerdictArgs,
    ToolArguments,
    ToolName,
    clone_investigation_seed,
)


_TOOL_MODELS = {
    "SEARCH_EVIDENCE": SearchEvidenceArgs,
    "READ_SPAN": ReadSpanArgs,
    "SUBMIT_VERDICT": SubmitVerdictArgs,
    "ABSTAIN": AbstainArgs,
}
_TRACE_ACTIONS = frozenset({"DECISION", *_TOOL_MODELS, "FINALIZE"})
_TRACE_OUTCOMES = frozenset({"accepted", "rejected", "completed", "abstained"})
_MAX_SAFE_COUNTER = 1_000_000_000


@dataclass(frozen=True, slots=True)
class InvestigatorLimits:
    max_decision_rounds: int = 12
    max_tool_calls: int = 24
    max_searches: int = 8
    max_reads: int = 12
    max_results: int = 48
    max_read_lines: int = 12
    max_span_chars: int = 12_000
    max_charged_tokens: int = 8_000
    deadline_seconds: float = 60.0

    def __post_init__(self) -> None:
        bounds = (
            (self.max_decision_rounds, 1, 64),
            (self.max_tool_calls, 1, 128),
            (self.max_searches, 1, 64),
            (self.max_reads, 1, 128),
            (self.max_results, 1, 512),
            (self.max_read_lines, 1, 20),
            (self.max_span_chars, 1, 100_000),
            (self.max_charged_tokens, 1, 100_000),
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or not minimum <= value <= maximum
            for value, minimum, maximum in bounds
        ):
            raise ValueError("investigator limit is invalid")
        if (
            isinstance(self.deadline_seconds, bool)
            or not isinstance(self.deadline_seconds, (int, float))
            or not 0 < float(self.deadline_seconds) <= 300
        ):
            raise ValueError("investigator deadline is invalid")


@dataclass(frozen=True, slots=True)
class InvestigatorTraceEvent:
    action: Literal[
        "DECISION",
        "SEARCH_EVIDENCE",
        "READ_SPAN",
        "SUBMIT_VERDICT",
        "ABSTAIN",
        "FINALIZE",
    ]
    seed_hash: str
    round_no: int
    outcome: Literal["accepted", "rejected", "completed", "abstained"]
    reason_code: str
    action_hash: str | None = None
    result_count: int = 0
    line_start: int | None = None
    line_end: int | None = None
    span_hash: str | None = None
    charged_tokens: int = 0

    def safe_dict(self) -> dict[str, Any]:
        return {
            "action": (
                self.action
                if type(self.action) is str and self.action in _TRACE_ACTIONS
                else "FINALIZE"
            ),
            "seed_hash": self.seed_hash if _is_hash(self.seed_hash) else None,
            "round": _safe_counter(self.round_no),
            "outcome": (
                self.outcome
                if type(self.outcome) is str and self.outcome in _TRACE_OUTCOMES
                else "rejected"
            ),
            "reason_code": (
                self.reason_code
                if type(self.reason_code) is str
                and self.reason_code in SAFE_REASON_CODES
                else "internal_failure"
            ),
            "action_hash": self.action_hash if _is_hash(self.action_hash) else None,
            "result_count": _safe_counter(self.result_count),
            "line_start": _safe_line(self.line_start),
            "line_end": _safe_line(self.line_end),
            "span_hash": self.span_hash if _is_hash(self.span_hash) else None,
            "charged_tokens": _safe_counter(self.charged_tokens),
        }


@dataclass(frozen=True, slots=True)
class UntrustedCandidateEnvelope:
    """Authorized evidence transport, never an accepted issue or fact."""

    seed_ref: str
    candidate_payloads: tuple[str, ...] = field(repr=False)
    authorized_span_hashes: tuple[str, ...]
    trusted: Literal[False] = field(default=False, init=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.seed_ref, str)
            or re.fullmatch(SEED_REF_PATTERN, self.seed_ref) is None
            or type(self.candidate_payloads) is not tuple
            or not 1 <= len(self.candidate_payloads) <= 2
            or type(self.authorized_span_hashes) is not tuple
            or len(self.authorized_span_hashes) != len(self.candidate_payloads)
            or any(not _is_hash(value) for value in self.authorized_span_hashes)
        ):
            raise ValueError("untrusted candidate envelope is invalid")
        for payload in self.candidate_payloads:
            if not isinstance(payload, str):
                raise ValueError("untrusted candidate envelope is invalid")
            try:
                candidate = CandidateRecordSubmission.model_validate_json(payload)
            except ValueError:
                raise ValueError("untrusted candidate envelope is invalid") from None
            if payload != _canonical_candidate_payload(candidate):
                raise ValueError("candidate payload is not canonical")

    @classmethod
    def create(
        cls,
        *,
        seed_ref: str,
        candidates: Sequence[CandidateRecordSubmission],
        authorized_span_hashes: Sequence[str],
    ) -> UntrustedCandidateEnvelope:
        return cls(
            seed_ref=seed_ref,
            candidate_payloads=tuple(
                _canonical_candidate_payload(candidate) for candidate in candidates
            ),
            authorized_span_hashes=tuple(authorized_span_hashes),
        )

    @property
    def candidates(self) -> tuple[CandidateRecordSubmission, ...]:
        """Return detached values; caller mutation cannot alter stored payloads."""

        return tuple(
            CandidateRecordSubmission.model_validate_json(payload)
            for payload in self.candidate_payloads
        )


@dataclass(frozen=True, slots=True)
class InvestigatorRunResult:
    envelopes: tuple[UntrustedCandidateEnvelope, ...] = field(
        repr=False, compare=False
    )
    completed_seed_refs: tuple[str, ...]
    abstained_seed_refs: tuple[str, ...]
    failed_seed_reasons: dict[str, str]
    decision_rounds: int
    tool_calls: int
    searches: int
    reads: int
    result_count: int
    span_chars: int
    charged_tokens: int
    trace: tuple[InvestigatorTraceEvent, ...]

    def safe_dict(self) -> dict[str, Any]:
        safe_trace = tuple(
            row
            for row in _bounded_tuple(self.trace, 128)
            if type(row) is InvestigatorTraceEvent
        )
        safe_failures = tuple(
            reason
            for seed_ref, reason in _mapping_items(self.failed_seed_reasons, 64)
            if type(seed_ref) is str
            and re.fullmatch(SEED_REF_PATTERN, seed_ref) is not None
            and type(reason) is str
            and reason in SAFE_REASON_CODES
            and reason not in {"accepted", "completed"}
        )
        failure_reasons = tuple(sorted(set(safe_failures)))
        return {
            "protocol": "native_tool_contract_v1",
            "decision_rounds": _safe_counter(self.decision_rounds),
            "tool_calls": _safe_counter(self.tool_calls),
            "searches": _safe_counter(self.searches),
            "reads": _safe_counter(self.reads),
            "result_count": _safe_counter(self.result_count),
            "span_chars": _safe_counter(self.span_chars),
            "charged_tokens": _safe_counter(self.charged_tokens),
            "submitted_envelopes": _typed_count(
                self.envelopes, UntrustedCandidateEnvelope, 64
            ),
            "completed_seeds": _seed_ref_count(self.completed_seed_refs),
            "abstained_seeds": _seed_ref_count(self.abstained_seed_refs),
            "failed_seeds": len(safe_failures),
            "failure_reasons": list(failure_reasons),
            "trace": [InvestigatorTraceEvent.safe_dict(row) for row in safe_trace],
            "total_trace_events": _typed_count(
                self.trace, InvestigatorTraceEvent, _MAX_SAFE_COUNTER
            ),
            "trace_truncated": _sequence_length(self.trace) > 128,
            "boundary": "SUBMIT_VERDICT returns untrusted candidates; no issue is created.",
        }


class EvidenceInvestigatorSession:
    """Pure state machine used to test a future provider-driven tool loop."""

    def __init__(
        self,
        *,
        scope: InvestigationScope,
        seeds: Sequence[InvestigationSeed],
        limits: InvestigatorLimits | None = None,
        token_factory: Callable[[], str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(scope) is not InvestigationScope:
            raise TypeError("scope is invalid")
        try:
            scope = clone_investigation_scope(scope)
        except ValueError:
            raise ValueError("scope is invalid") from None
        if not callable(monotonic):
            raise TypeError("monotonic clock must be callable")
        try:
            supplied = tuple(seeds)
        except TypeError:
            raise ValueError("session seeds are invalid") from None
        try:
            prepared = tuple(clone_investigation_seed(seed) for seed in supplied)
        except ValueError:
            raise ValueError("session seeds are invalid") from None
        if not prepared or len(prepared) > 64:
            raise ValueError("session seeds are invalid")
        if len({row.seed_ref for row in prepared}) != len(prepared):
            raise ValueError("session seeds contain duplicates")
        documents = {row.snapshot.document_id: row for row in scope.documents}
        for seed in prepared:
            if seed.run_hash != scope.run_hash:
                raise InvestigatorRejected("cross_run")
            document = documents.get(seed.anchor.evidence.document_id)
            if document is None or not _anchor_matches_document(seed, document.lines):
                raise InvestigatorRejected("snapshot_mismatch")

        self._scope = scope
        self._seeds = {row.seed_ref: row for row in prepared}
        supplied_limits = limits or InvestigatorLimits()
        if type(supplied_limits) is not InvestigatorLimits:
            raise TypeError("limits are invalid")
        self._limits = _copy_limits(supplied_limits)
        self._authority = EvidenceGrantAuthority(
            scope=scope,
            seeds=prepared,
            token_factory=token_factory,
        )
        self.monotonic = monotonic
        self.started = monotonic()
        self.decision_rounds = 0
        self.tool_calls = 0
        self.searches = 0
        self.reads = 0
        self.result_count = 0
        self.span_chars = 0
        self.charged_tokens = 0
        self._pending_seed: str | None = None
        self._action_signatures: set[str] = set()
        self._query_signatures: dict[str, set[str]] = {}
        self._seen_chunks: dict[str, set[str]] = {}
        self._no_progress: dict[str, int] = {}
        self._terminal: dict[str, str] = {}
        self._envelopes: list[UntrustedCandidateEnvelope] = []
        self._trace: list[InvestigatorTraceEvent] = []
        self._finalized = False
        self._final_result: InvestigatorRunResult | None = None

    def begin_decision(self, seed_ref: str, *, charged_tokens: int) -> None:
        self._ensure_open()
        if self._pending_seed is not None:
            self._fail(self._pending_seed, "DECISION", "invalid_state")
        seed = self._seed(seed_ref)
        if seed_ref in self._terminal:
            raise InvestigatorRejected("invalid_state")
        self._check_deadline(seed_ref, "DECISION")
        if (
            isinstance(charged_tokens, bool)
            or not isinstance(charged_tokens, int)
            or charged_tokens < 0
        ):
            self._fail(seed_ref, "DECISION", "token_budget")
        if self.decision_rounds >= self._limits.max_decision_rounds:
            self._fail(seed_ref, "DECISION", "round_budget")
        if self.charged_tokens + charged_tokens > self._limits.max_charged_tokens:
            self._fail(seed_ref, "DECISION", "token_budget")
        self.decision_rounds += 1
        self.charged_tokens += charged_tokens
        self._pending_seed = seed.seed_ref
        self._trace.append(
            InvestigatorTraceEvent(
                action="DECISION",
                seed_hash=_sha256(seed.seed_ref),
                round_no=self.decision_rounds,
                outcome="accepted",
                reason_code="accepted",
                charged_tokens=charged_tokens,
            )
        )

    def search(
        self,
        args: SearchEvidenceArgs,
        chunks: Sequence[EvidenceChunk],
    ) -> tuple[SearchObservation, ...]:
        seed, args = self._preflight_action("SEARCH_EVIDENCE", args)
        try:
            prepared = clone_evidence_chunks(chunks)
        except InvestigatorRejected as exc:
            self._fail(seed.seed_ref, "SEARCH_EVIDENCE", exc.reason_code)
        query_signature = _query_signature(args)
        if query_signature in self._query_signatures.setdefault(seed.seed_ref, set()):
            self._fail(seed.seed_ref, "SEARCH_EVIDENCE", "repeated_query")
        if self.searches >= self._limits.max_searches:
            self._fail(seed.seed_ref, "SEARCH_EVIDENCE", "search_budget")
        if self.result_count + len(prepared) > self._limits.max_results:
            self._fail(seed.seed_ref, "SEARCH_EVIDENCE", "result_budget")

        seen = self._seen_chunks.setdefault(seed.seed_ref, set())
        new_chunks = tuple(chunk for chunk in prepared if chunk.chunk_id not in seen)
        if not new_chunks:
            self._no_progress[seed.seed_ref] = self._no_progress.get(seed.seed_ref, 0) + 1
            if self._no_progress[seed.seed_ref] >= 2:
                self._fail(seed.seed_ref, "SEARCH_EVIDENCE", "no_progress")
        else:
            self._no_progress[seed.seed_ref] = 0
        try:
            grants = self._authority.issue_search_grants(seed.seed_ref, new_chunks)
        except InvestigatorRejected as exc:
            self._fail(seed.seed_ref, "SEARCH_EVIDENCE", exc.reason_code)
        self._query_signatures[seed.seed_ref].add(query_signature)
        seen.update(row.chunk_id for row in new_chunks)
        self.searches += 1
        self.result_count += len(grants)
        self._commit_action(seed, "SEARCH_EVIDENCE", args, result_count=len(grants))
        return tuple(
            SearchObservation(row.result_ref, row.line_start, row.line_end)
            for row in grants
        )

    def read(self, args: ReadSpanArgs) -> ReadObservation:
        seed, args = self._preflight_action("READ_SPAN", args)
        if self.reads >= self._limits.max_reads:
            self._fail(seed.seed_ref, "READ_SPAN", "read_budget")
        try:
            grant = self._authority.read_span(
                args, max_read_lines=self._limits.max_read_lines
            )
        except InvestigatorRejected as exc:
            self._fail(seed.seed_ref, "READ_SPAN", exc.reason_code)
        if self.span_chars + len(grant.text) > self._limits.max_span_chars:
            self._fail(seed.seed_ref, "READ_SPAN", "span_budget")
        self.reads += 1
        self.span_chars += len(grant.text)
        self._commit_action(
            seed,
            "READ_SPAN",
            args,
            line_start=grant.line_start,
            line_end=grant.line_end,
            span_hash=grant.text_sha256,
        )
        return ReadObservation(
            span_ref=grant.span_ref,
            line_start=grant.line_start,
            line_end=grant.line_end,
            char_start=grant.char_start,
            char_end=grant.char_end,
            text=grant.text,
        )

    def submit(self, args: SubmitVerdictArgs) -> UntrustedCandidateEnvelope:
        seed, args = self._preflight_action("SUBMIT_VERDICT", args)
        grants: list[SpanGrant] = []
        try:
            for candidate in args.candidates:
                grants.append(
                    self._authority.authorize_candidate(seed.seed_ref, candidate)
                )
        except InvestigatorRejected as exc:
            self._fail(seed.seed_ref, "SUBMIT_VERDICT", exc.reason_code)
        cloned = tuple(
            CandidateRecordSubmission.model_validate(
                json.loads(
                    json.dumps(
                        row.model_dump(mode="json", warnings="error"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                )
            )
            for row in args.candidates
        )
        envelope = UntrustedCandidateEnvelope.create(
            seed_ref=seed.seed_ref,
            candidates=cloned,
            authorized_span_hashes=tuple(row.text_sha256 for row in grants),
        )
        self._envelopes.append(envelope)
        self._commit_action(seed, "SUBMIT_VERDICT", args)
        self._terminal[seed.seed_ref] = "completed"
        return _copy_envelope(envelope)

    def abstain(self, args: AbstainArgs) -> None:
        seed, args = self._preflight_action("ABSTAIN", args)
        self._commit_action(seed, "ABSTAIN", args, outcome="abstained")
        self._terminal[seed.seed_ref] = "explicit_abstain"

    def result(self) -> InvestigatorRunResult:
        if self._finalized:
            if self._final_result is None:
                raise InvestigatorRejected("internal_failure")
            return _copy_run_result(self._final_result)
        if self._pending_seed is not None:
            self._fail(self._pending_seed, "FINALIZE", "invalid_state")
        for seed_ref in self._seeds:
            self._terminal.setdefault(seed_ref, "unprocessed_seed")
        completed = tuple(
            sorted(key for key, value in self._terminal.items() if value == "completed")
        )
        abstained = tuple(
            sorted(
                key
                for key, value in self._terminal.items()
                if value == "explicit_abstain"
            )
        )
        failed = {
            key: value
            for key, value in sorted(self._terminal.items())
            if value not in {"completed", "explicit_abstain"}
        }
        result = InvestigatorRunResult(
            envelopes=tuple(self._envelopes),
            completed_seed_refs=completed,
            abstained_seed_refs=abstained,
            failed_seed_reasons=failed,
            decision_rounds=self.decision_rounds,
            tool_calls=self.tool_calls,
            searches=self.searches,
            reads=self.reads,
            result_count=self.result_count,
            span_chars=self.span_chars,
            charged_tokens=self.charged_tokens,
            trace=tuple(self._trace),
        )
        self._final_result = result
        self._finalized = True
        return _copy_run_result(result)

    def _seed(self, seed_ref: Any) -> InvestigationSeed:
        if not isinstance(seed_ref, str):
            raise InvestigatorRejected("unknown_seed")
        seed = self._seeds.get(seed_ref)
        if seed is None:
            raise InvestigatorRejected("unknown_seed")
        return seed

    def _preflight_action(
        self, action: ToolName, args: ToolArguments
    ) -> tuple[InvestigationSeed, ToolArguments]:
        self._ensure_open()
        pending_seed_ref = self._pending_seed
        if pending_seed_ref is None:
            raise InvestigatorRejected("invalid_state")
        if type(args) is not _TOOL_MODELS[action]:
            self._fail(pending_seed_ref, action, "invalid_tool_arguments")
        try:
            args = _clone_tool_arguments(args, _TOOL_MODELS[action])
        except (AttributeError, TypeError, ValueError):
            self._fail(pending_seed_ref, action, "invalid_tool_arguments")
        if args.seed_ref != pending_seed_ref:
            self._fail(pending_seed_ref, action, "invalid_state")
        seed = self._seeds[pending_seed_ref]
        if seed.seed_ref in self._terminal:
            raise InvestigatorRejected("invalid_state")
        self._check_deadline(seed.seed_ref, action)
        if self.tool_calls >= self._limits.max_tool_calls:
            self._fail(seed.seed_ref, action, "tool_budget")
        signature = _action_signature(action, args)
        if signature in self._action_signatures:
            self._fail(seed.seed_ref, action, "repeated_action")
        return seed, args

    def _commit_action(
        self,
        seed: InvestigationSeed,
        action: ToolName,
        args: ToolArguments,
        *,
        outcome: Literal["accepted", "abstained"] = "accepted",
        result_count: int = 0,
        line_start: int | None = None,
        line_end: int | None = None,
        span_hash: str | None = None,
    ) -> None:
        signature = _action_signature(action, args)
        self._action_signatures.add(signature)
        self.tool_calls += 1
        self._pending_seed = None
        self._trace.append(
            InvestigatorTraceEvent(
                action=action,
                seed_hash=_sha256(seed.seed_ref),
                round_no=self.decision_rounds,
                outcome=outcome,
                reason_code=(
                    "explicit_abstain" if outcome == "abstained" else "accepted"
                ),
                action_hash=signature,
                result_count=result_count,
                line_start=line_start,
                line_end=line_end,
                span_hash=span_hash,
            )
        )

    def _check_deadline(self, seed_ref: str, action: str) -> None:
        if self.monotonic() - self.started >= float(self._limits.deadline_seconds):
            self._fail(seed_ref, action, "deadline")

    def _fail(self, seed_ref: str, action: str, reason: str) -> None:
        safe_reason = reason if reason in SAFE_REASON_CODES else "internal_failure"
        seed_hash = _sha256(seed_ref) if isinstance(seed_ref, str) else _sha256("unknown")
        safe_action = action if action in _TOOL_MODELS or action == "DECISION" else "FINALIZE"
        if (
            isinstance(seed_ref, str)
            and seed_ref in self._seeds
            and seed_ref not in self._terminal
        ):
            self._terminal[seed_ref] = safe_reason
        if self._pending_seed == seed_ref:
            self._pending_seed = None
        self._trace.append(
            InvestigatorTraceEvent(
                action=safe_action,
                seed_hash=seed_hash,
                round_no=self.decision_rounds,
                outcome="rejected",
                reason_code=safe_reason,
            )
        )
        raise InvestigatorRejected(safe_reason)

    def _ensure_open(self) -> None:
        if self._finalized:
            raise InvestigatorRejected("invalid_state")


def _query_signature(args: SearchEvidenceArgs) -> str:
    query = " ".join(unicodedata.normalize("NFKC", args.query).casefold().split())
    terms = sorted(
        " ".join(unicodedata.normalize("NFKC", value).casefold().split())
        for value in args.entity_terms
    )
    return _sha256(json.dumps([query, terms], ensure_ascii=False, separators=(",", ":")))


def _action_signature(action: str, args: ToolArguments) -> str:
    return _sha256(
        json.dumps(
            {
                "action": action,
                "arguments": args.model_dump(mode="json", warnings="error"),
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    )


def _clone_tool_arguments(args: ToolArguments, model: type) -> ToolArguments:
    payload = json.loads(
        json.dumps(
            args.model_dump(mode="json", warnings="error"),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
    )
    return model.model_validate(payload)


def _canonical_candidate_payload(candidate: CandidateRecordSubmission) -> str:
    if type(candidate) is not CandidateRecordSubmission:
        raise ValueError("candidate is invalid")
    payload = candidate.model_dump(mode="json", warnings="error")
    cloned = CandidateRecordSubmission.model_validate(
        json.loads(
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                separators=(",", ":"),
            )
        )
    )
    return json.dumps(
        cloned.model_dump(mode="json", warnings="error"),
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    )


def _anchor_matches_document(seed: InvestigationSeed, lines: tuple[str, ...]) -> bool:
    evidence = seed.anchor.evidence
    start, end = evidence.line_start, evidence.line_end
    if (
        type(start) is not int
        or type(end) is not int
        or start < 1
        or end < start
        or end > len(lines)
        or not isinstance(evidence.text, str)
    ):
        return False
    source = "\n".join(lines[start - 1 : end]).strip()
    if evidence.text == source:
        return True
    # Explicit @directives intentionally retain the exact text after ``|``.
    if start == end and source.startswith("@") and "|" in source:
        declared = source.partition("|")[2].strip() or source
        return evidence.text == declared
    return False


def _copy_run_result(result: InvestigatorRunResult) -> InvestigatorRunResult:
    return InvestigatorRunResult(
        envelopes=tuple(_copy_envelope(row) for row in result.envelopes),
        completed_seed_refs=tuple(result.completed_seed_refs),
        abstained_seed_refs=tuple(result.abstained_seed_refs),
        failed_seed_reasons=dict(result.failed_seed_reasons),
        decision_rounds=result.decision_rounds,
        tool_calls=result.tool_calls,
        searches=result.searches,
        reads=result.reads,
        result_count=result.result_count,
        span_chars=result.span_chars,
        charged_tokens=result.charged_tokens,
        trace=tuple(_copy_trace_event(row) for row in result.trace),
    )


def _copy_limits(limits: InvestigatorLimits) -> InvestigatorLimits:
    if type(limits) is not InvestigatorLimits:
        raise TypeError("limits are invalid")
    try:
        return InvestigatorLimits(
            max_decision_rounds=limits.max_decision_rounds,
            max_tool_calls=limits.max_tool_calls,
            max_searches=limits.max_searches,
            max_reads=limits.max_reads,
            max_results=limits.max_results,
            max_read_lines=limits.max_read_lines,
            max_span_chars=limits.max_span_chars,
            max_charged_tokens=limits.max_charged_tokens,
            deadline_seconds=limits.deadline_seconds,
        )
    except (AttributeError, TypeError, ValueError):
        raise ValueError("limits are invalid") from None


def _copy_envelope(
    envelope: UntrustedCandidateEnvelope,
) -> UntrustedCandidateEnvelope:
    if type(envelope) is not UntrustedCandidateEnvelope:
        raise InvestigatorRejected("internal_failure")
    return UntrustedCandidateEnvelope(
        seed_ref=envelope.seed_ref,
        candidate_payloads=tuple(envelope.candidate_payloads),
        authorized_span_hashes=tuple(envelope.authorized_span_hashes),
    )


def _copy_trace_event(event: InvestigatorTraceEvent) -> InvestigatorTraceEvent:
    if type(event) is not InvestigatorTraceEvent:
        raise InvestigatorRejected("internal_failure")
    safe = InvestigatorTraceEvent.safe_dict(event)
    return InvestigatorTraceEvent(
        action=safe["action"],
        seed_hash=safe["seed_hash"] or _sha256("invalid-seed"),
        round_no=safe["round"],
        outcome=safe["outcome"],
        reason_code=safe["reason_code"],
        action_hash=safe["action_hash"],
        result_count=safe["result_count"],
        line_start=safe["line_start"],
        line_end=safe["line_end"],
        span_hash=safe["span_hash"],
        charged_tokens=safe["charged_tokens"],
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_hash(value: Any) -> bool:
    return type(value) is str and HASH_PATTERN.fullmatch(value) is not None


def _safe_counter(value: Any) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_SAFE_COUNTER:
        return 0
    return value


def _bounded_tuple(value: Any, limit: int) -> tuple[Any, ...]:
    if type(value) not in {tuple, list}:
        return ()
    return tuple(value[:limit])


def _mapping_items(value: Any, limit: int) -> tuple[tuple[Any, Any], ...]:
    if type(value) is not dict:
        return ()
    return tuple(list(value.items())[:limit])


def _typed_count(value: Any, expected_type: type, limit: int) -> int:
    if type(value) not in {tuple, list}:
        return 0
    return min(sum(type(row) is expected_type for row in value), limit)


def _seed_ref_count(value: Any) -> int:
    return len(
        {
            row
            for row in _bounded_tuple(value, 64)
            if type(row) is str
            and re.fullmatch(SEED_REF_PATTERN, row) is not None
        }
    )


def _sequence_length(value: Any) -> int:
    if type(value) not in {tuple, list}:
        return 0
    return min(len(value), _MAX_SAFE_COUNTER)


def _safe_line(value: Any) -> int | None:
    return value if type(value) is int and 1 <= value <= MAX_LINE_NUMBER else None
