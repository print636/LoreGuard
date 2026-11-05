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
    """A validated original record retained only in this worker's memory."""
    stable_key: str
    deferred: object
    package_attempt: int
    target_sha256: str


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

    def reconcile(self, *, baselines, provider, settings, budget, charge, checkpoint, admit, ledger, reasons) -> None:
        from .character_trait_extraction import (
            CharacterSignalChunk, _target_bound_review_target, review_original_material_claim, stable_trait_identity,
        )
        from .character_scope_review import ScopeReviewSourceIdentity

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
                    gap = gaps[key]
                    # Overlap selects a useful question only. The original
                    # claim still needs its own complete, unique certificate.
                    if not any(signal.evidence.line_start <= gap.line_end and gap.line_start <= signal.evidence.line_end
                               for _, admitted in results for signal in admitted):
                        continue
                    attempted_targets.add(row.candidate_id)
                    checkpoint()
                    available = budget()
                    if available < 256:
                        reasons["claim_resolution_token_budget"] += 1
                        continue
                    review = review_original_material_claim(
                        original, chunk=CharacterSignalChunk(source.document.id, source.document.name, chunk.content,
                                                             chunk.global_line_start, "draft"),
                        source_identity=ScopeReviewSourceIdentity(
                            run_input_id=source.input_id, document_id=source.document.id,
                            document_version=source.document_version, content_sha256=source.content_sha256),
                        frozen_content=source.document.content, provider=provider, settings=settings, token_budget=available,
                    )
                    charge(review)
                    reasons["claim_resolution_attempted_calls"] += review.attempted_calls
                    reasons.update(dict(review.reason_counts))
                    checkpoint()
                    admitted_ids = set()
                    for signal in review.signals:
                        if admit(signal, source, target):
                            admitted_ids.add(signal.id)
                    for proof in review.claim_certificates:
                        if proof["target_sha256"] == identity and proof["observation_id"] in admitted_ids:
                            known[key] = (proof, "original_claim_recheck")
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
        identities = {identity for identity, _ in proofs}
        repaired = False
        from .character_trait_extraction import _target_bound_review_target
        for source, _, target, result in self.targeted:
            identity = claim_resolution_target_sha256(_target_bound_review_target(target, target_ordinal=1))
            if source.input_id != source_input_id or identity not in identities or result.diagnostics.outcome == "completed":
                continue
            state = result.claim_resolution_state
            if (state is None or not state.enumeration_complete or state.omitted_count or not state.gaps
                    or any((identity, gap.claim_digest) not in proofs for gap in state.gaps)):
                return False
            repaired = True
        return repaired

    def payload(self) -> dict | None:
        if not self.scopes:
            return None
        scopes = [{**scope, "status": "partial"} for scope in self.scopes] if self.omitted_count else self.scopes
        return claim_resolution_payload(receipts=self.receipts, final_scopes=scopes, omitted_count=self.omitted_count)
