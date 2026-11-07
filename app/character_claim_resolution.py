"""Exact claim identities and bounded final coverage receipts; no text is persisted here."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

from .narrative_context import payload_sha256

CLAIM_RESOLUTION_KEY = "claim_resolution_v1"
CLAIM_RESOLUTION_SCHEMA = "character-claim-resolution-v1"
MAX_CLAIM_RESOLUTION_RECEIPTS = 64
MAX_CLAIM_RESOLUTION_SCOPES = 128


def claim_resolution_target_sha256(target) -> str:
    return payload_sha256(target.model_dump(mode="json", exclude={"target_ordinal"}))


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PendingMaterialClaim:
    """An unresolved original record retained only in this worker's memory."""
    stable_key: str
    deferred: object | None
    package_attempt: int
    target_sha256: str
    retained: object | None = None

    def __post_init__(self):
        if (self.deferred is None) == (self.retained is None):
            raise ValueError("original claim must have exactly one retention basis")

    @property
    def original(self):
        return self.retained if self.retained is not None else self.deferred


@dataclass
class ClaimPackageState:
    pending: dict[str, PendingMaterialClaim] = field(default_factory=dict)
    certificates: list[dict] = field(default_factory=list)
    gaps: tuple = ()
    omitted_count: int = 0
    enumeration_complete: bool = False


def claim_resolution_receipt(proof: dict, *, stable_key: str, method: str,
                             candidate_id: str, candidate_sha256: str) -> dict:
    value = {k: v for k, v in proof.items() if k != "record_index"}
    value.update(primary_claim_sha256=stable_key, method=method,
                 confirmed_candidate_id=candidate_id, confirmed_payload_sha256=candidate_sha256)
    return {**value, "id": "cr_" + payload_sha256(value)}


def claim_resolution_payload(*, receipts: list[dict], final_scopes: list[dict], omitted_count: int = 0) -> dict:
    value = dict(schema_version=CLAIM_RESOLUTION_SCHEMA,
                 receipts=receipts[:MAX_CLAIM_RESOLUTION_RECEIPTS],
                 final_scopes=final_scopes[:MAX_CLAIM_RESOLUTION_SCOPES],
                 omitted_count=omitted_count + max(0, len(receipts) - MAX_CLAIM_RESOLUTION_RECEIPTS)
                 + max(0, len(final_scopes) - MAX_CLAIM_RESOLUTION_SCOPES))
    return {**value, "payload_sha256": payload_sha256(value)}


@dataclass
class ClaimResolutionManager:
    """Final scoped proofs, separate from historical package success counters."""
    primary: list[tuple] = field(default_factory=list)
    targeted: list[tuple] = field(default_factory=list)
    receipts: list[dict] = field(default_factory=list)
    scopes: list[dict] = field(default_factory=list)
    omitted_count: int = 0

    def capture_primary(self, source, chunk, result) -> None:
        if result.diagnostics.outcome != "completed" and result.claim_resolution_state is not None:
            self.primary.append((source, chunk, result.claim_resolution_state))

    def capture_targeted(self, source, chunk, target, result) -> None:
        self.targeted.append((source, chunk, target, result))

    def reconcile(self, *, baselines, provider, settings, budget, charge, checkpoint, admit, ledger, reasons,
                  origin_admitted=None) -> None:
        from .character_trait_extraction import (
            CharacterSignalChunk, _target_bound_review_target, review_original_material_claim, stable_trait_identity,
        )
        from .character_scope_review import ScopeReviewSourceIdentity
        from .config import SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2, SELECTED_CLAIM_RESOLUTION_POLICY_V3

        attempted_targets = set()
        for source, chunk, state in self.primary:
            targets = {}
            for prior_source, prior_chunk, target, result in self.targeted:
                if prior_source.input_id != source.input_id or prior_chunk.global_line_start != chunk.global_line_start:
                    continue
                identity = claim_resolution_target_sha256(_target_bound_review_target(target, target_ordinal=1))
                admitted = tuple(s for s in result.signals if s.target_bound_target is not None
                                 and claim_resolution_target_sha256(s.target_bound_target) == identity)
                if admitted:
                    targets.setdefault(identity, (target, []))[1].append((result, admitted))
            for identity, (target, results) in targets.items():
                candidates = [entry for entry in baselines if (
                    entry[1].dimension == target.dimension
                    and stable_trait_identity(entry[1].dimension, entry[1].trait_key) == target.comparison_key
                    and entry[1].statement == target.baseline_hint and entry[1].contexts == target.habit_contexts
                    and entry[1].character == target.character and entry[1].polarity == target.baseline_polarity
                )]
                if len(candidates) != 1:
                    continue
                row = candidates[0][0]
                binding = dict(source_input_id=source.input_id, source_document_id=source.document.id,
                               source_content_sha256=source.content_sha256, target_sha256=identity,
                               confirmed_candidate_id=row.candidate_id, confirmed_payload_sha256=row.payload_sha256)
                pending = {key: value for key, value in state.pending.items() if value.target_sha256 == identity}
                gaps = {gap.claim_digest: gap for gap in state.gaps if gap.claim_digest is not None}
                pending = {key: value for key, value in pending.items() if key in gaps}
                if not pending:
                    continue
                known = {}
                for result, admitted in results:
                    if result.claim_resolution_state is None:
                        continue
                    admitted_ids = {signal.id for signal in admitted}
                    for proof in result.claim_resolution_state.certificates:
                        key = proof["primary_claim_sha256"]
                        if (key in pending and proof["target_sha256"] == identity
                                and proof["observation_id"] in admitted_ids):
                            known[key] = (proof, "exact_claim")
                for key, original in pending.items():
                    if key in known or row.candidate_id in attempted_targets:
                        continue
                    if original.retained is not None and settings._character_selected_claim_resolution_policy != SELECTED_CLAIM_RESOLUTION_POLICY_V3:
                        continue
                    gap = gaps[key]
                    # Overlap selects a useful question only. The original
                    # claim still needs its own complete, unique certificate.
                    if not any(signal.evidence.line_start <= gap.line_end and gap.line_start <= signal.evidence.line_end
                               for _, admitted in results for signal in admitted):
                        continue
                    origin = None
                    selected_policy = settings._character_selected_claim_resolution_policy
                    prefer_selected = selected_policy in {SELECTED_CLAIM_RESOLUTION_POLICY_V2, SELECTED_CLAIM_RESOLUTION_POLICY_V3}
                    if prefer_selected or not original.original.record.key_object:
                        if selected_policy not in {SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2, SELECTED_CLAIM_RESOLUTION_POLICY_V3}:
                            continue
                        origins = [cert for result, admitted in results for cert in result.selected_action_certificates
                                   if cert.signal.id in {s.id for s in admitted} and origin_admitted is not None and origin_admitted(cert.signal)
                                   and cert.signal.evidence.line_start == gap.line_start
                                   and cert.signal.evidence.line_end == gap.line_end]
                        if not origins:
                            reasons["selected_claim_origin_unavailable"] += 1
                            if not prefer_selected or original.retained is not None:
                                continue
                        # Literal correspondence can prioritize a question; it
                        # cannot discharge the claim or replace the new review.
                        statement = original.original.record.statement
                        origins.sort(key=lambda cert: not any(p.proposal_id == cert.proposal_id and (
                            p.canonical_statement == statement or any(c.support_id == p.fact_clause_id and c.text == statement
                                for line in cert.request.lines for c in line.clauses)) for p in cert.request.proposals))
                        origin = origins[0] if origins else None
                    attempted_targets.add(row.candidate_id)
                    checkpoint()
                    available = budget()
                    if available < 256:
                        reasons["claim_resolution_token_budget"] += 1
                        continue
                    review_function = review_original_material_claim
                    extra = {}
                    if origin is not None:
                        from .character_selected_claim_resolution import review_selected_original_claim
                        review_function = review_selected_original_claim
                        extra["origin"] = origin
                    review = review_function(
                        original, chunk=CharacterSignalChunk(source.document.id, source.document.name, chunk.content,
                                                             chunk.global_line_start, "draft"),
                        source_identity=ScopeReviewSourceIdentity(
                            run_input_id=source.input_id, document_id=source.document.id,
                            document_version=source.document_version, content_sha256=source.content_sha256),
                        frozen_content=source.document.content, provider=provider, settings=settings, token_budget=available, **extra,
                    )
                    charge(review)
                    reasons["claim_resolution_attempted_calls"] += review.attempted_calls
                    reasons.update(dict(review.reason_counts))
                    checkpoint()
                    if origin is not None and review.material_reasons:
                        ledger.reason_scopes[(source.input_id, row.candidate_id, key)] = review.material_reasons[0][1]
                    admitted_ids = set()
                    for signal in review.signals:
                        if admit(signal, source, target):
                            admitted_ids.add(signal.id)
                    for proof in review.claim_certificates:
                        if proof["target_sha256"] == identity and proof["observation_id"] in admitted_ids:
                            method = ("selected_claim_recheck_v2" if prefer_selected else "selected_claim_recheck") if origin is not None else "original_claim_recheck"
                            known[key] = (proof, method)
                    if key not in known:
                        reasons["claim_resolution_unresolved"] += 1
                scope_receipts = []
                for key, (proof, method) in known.items():
                    receipt = claim_resolution_receipt(proof, stable_key=key, method=method,
                                                       candidate_id=row.candidate_id, candidate_sha256=row.payload_sha256)
                    if len(self.receipts) >= MAX_CLAIM_RESOLUTION_RECEIPTS:
                        self.omitted_count += 1
                        break
                    self.receipts.append(receipt)
                    scope_receipts.append(receipt)
                    ledger.resolved_scopes.add((source.input_id, row.candidate_id, key))
                count = len(state.gaps) + state.omitted_count
                unknown = sum(gap.claim_digest not in pending for gap in state.gaps)
                if not state.enumeration_complete:
                    unknown = max(1, unknown)
                resolved = len({receipt["primary_claim_sha256"] for receipt in scope_receipts})
                complete = bool(count and count == resolved and not unknown and not state.omitted_count)
                if len(self.scopes) < MAX_CLAIM_RESOLUTION_SCOPES:
                    self.scopes.append(dict(
                        **binding, chunk_line_start=chunk.global_line_start,
                        chunk_line_end=chunk.global_line_start + len(chunk.content.splitlines()) - 1,
                        status="complete" if complete else "partial", claim_count=max(count, resolved + unknown + state.omitted_count),
                        resolved_count=resolved, unknown_count=unknown, omitted_count=state.omitted_count,
                        receipt_ids=[receipt["id"] for receipt in scope_receipts],
                    ))
                else:
                    self.omitted_count += 1

    def repairs_document(self, document_id: str, candidate_id: str, missing_chunks: int) -> bool:
        scopes = [scope for scope in self.scopes if scope["source_document_id"] == document_id
                  and scope["confirmed_candidate_id"] == candidate_id and scope["status"] == "complete"]
        return not self.omitted_count and missing_chunks > 0 and len({(s["chunk_line_start"], s["chunk_line_end"]) for s in scopes}) == missing_chunks

    def repairs_targeted(self, source_input_id: str, candidate_id: str) -> bool:
        if self.omitted_count:
            return False
        proofs = {(r["target_sha256"], r["primary_claim_sha256"]) for r in self.receipts
                  if r["source_input_id"] == source_input_id and r["confirmed_candidate_id"] == candidate_id}
        if not proofs:
            return False
        identities = {identity for identity, _ in proofs}
        repaired = False
        from .character_trait_extraction import _target_bound_review_target
        bound_targets = {}
        for source, _, target, result in self.targeted:
            if source.input_id != source_input_id:
                continue
            # Ordinary recall targets need not satisfy the stricter semantic
            # review DTO. An actually bound result must first link the target
            # to an existing resolution receipt; absence is unresolved.
            bound_identities = {
                claim_resolution_target_sha256(signal.target_bound_target)
                for signal in result.signals if signal.target_bound_target is not None
            }
            if not identities.intersection(bound_identities):
                continue
            identity = claim_resolution_target_sha256(_target_bound_review_target(target, target_ordinal=1))
            if identity not in identities or identity not in bound_identities:
                continue
            recall_identity = payload_sha256(target.model_dump(mode="json", exclude={"existing_evidence_ranges"}))
            bound_targets[recall_identity] = identity
        for source, _, target, result in self.targeted:
            if source.input_id != source_input_id or result.diagnostics.outcome == "completed":
                continue
            recall_identity = payload_sha256(target.model_dump(mode="json", exclude={"existing_evidence_ranges"}))
            identity = bound_targets.get(recall_identity)
            if identity is None:
                continue
            # A later verification package may contain no admitted signal.
            # Its unresolved claims still belong to the same recall target.
            state = result.claim_resolution_state
            if (state is None or not state.enumeration_complete or state.omitted_count
                    or any((identity, gap.claim_digest) not in proofs for gap in state.gaps)):
                return False
            repaired = repaired or bool(state.gaps)
        return repaired

    def payload(self) -> dict | None:
        if not self.scopes:
            return None
        scopes = [{**scope, "status": "partial"} for scope in self.scopes] if self.omitted_count else self.scopes
        return claim_resolution_payload(receipts=self.receipts, final_scopes=scopes, omitted_count=self.omitted_count)
