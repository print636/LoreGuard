"""An accepted later event cannot erase an original claim without its own certificate."""
from collections import Counter
from types import SimpleNamespace
import json

import pytest

from app.character_claim_resolution import ClaimResolutionManager
from app.character_material_coverage import DraftMaterialGap, MaterialGapLedger
from app.character_scope_review import ScopeReviewSourceIdentity
from app.character_trait_extraction import CharacterSignalChunk, CharacterSignalExtractor
from tests.test_character_draft_actor_stage_wiring import _draft_source
from tests.test_habit_claim_recovery import HABIT, record
from tests.test_habit_object_chain_v15 import CONTENT, TAIL, response_items
from tests.test_character_target_bound_draft_review_v3 import _settings
from app.character_draft_actor_review import TargetBoundDraftReviewRequest
from app.character_draft_actor_review_provider import TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15


class ResolutionProvider:
    def __init__(self, *, exact=False, unresolved=False, targeted_middle=False):
        self.requests = []
        self.generations = 0
        self.exact = exact
        self.unresolved = unresolved
        self.trait_key = HABIT.trait_key
        self.targeted_middle = targeted_middle

    def complete(self, system, user):
        if system != TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15:
            self.generations += 1
            statement = "廖辰未清点器材箱便发货" if self.exact or self.generations == 1 else "廖辰没有清点器材箱就直接发运"
            if self.targeted_middle and self.generations == 2:
                statement = "廖辰把器材箱放上货台"
            output = {"records": [dict(record(statement), evidence=CONTENT, key_object="器材箱", trait_key=self.trait_key)]}
        else:
            batch = json.loads(user.split("：\n", 1)[1])
            responses = []
            for window in batch["windows"]:
                request = TargetBoundDraftReviewRequest.model_validate_json(json.dumps(window["request"]))
                self.requests.append(request)
                items = response_items(request)
                if self.targeted_middle and len(self.requests) == 2:
                    for item, proposal in zip(items, request.proposals, strict=True):
                        same = "摆上货台" in proposal.canonical_statement
                        item.update(verdict="supported" if same else "rejected",
                                    claim_relation="equivalent" if same else "different", claim_object=None)
                        if same:
                            item["claim_object"] = dict(role="event_object", grounding="unique_reference",
                                                        referent_relation="same", fact_quote="它们",
                                                        antecedent_clause_id=item["action_steps"][0]["clause_id"],
                                                        antecedent_quote="器材箱")
                if len(self.requests) == 1 or self.unresolved and len(self.requests) > 2:
                    next(item for item in items if item["verdict"] == "supported")["claim_object"] = None
                responses.append(dict(schema_version=request.schema_version, request_digest=window["request_digest"], items=items))
            output = dict(schema_version="character-target-bound-draft-review-batch-v4", batch_digest=batch["batch_digest"], responses=responses)
        return SimpleNamespace(text=json.dumps(output, ensure_ascii=False), prompt_tokens=9, completion_tokens=8)


def paired(*, exact=False, unresolved=False, targeted_middle=False):
    settings = _settings(character_signal_package_max_attempts=1, character_signal_draft_trace_v1=True)
    settings._character_habit_object_chain_v1 = settings._character_claim_resolution_v1 = True
    source = _draft_source(CONTENT)
    chunk = CharacterSignalChunk(source.document.id, source.document.name, CONTENT, 1, "draft")
    identity = ScopeReviewSourceIdentity(run_input_id=source.input_id, document_id=source.document.id,
                                        document_version=source.document_version, content_sha256=source.content_sha256)
    provider = ResolutionProvider(exact=exact, unresolved=unresolved, targeted_middle=targeted_middle)
    extractor = CharacterSignalExtractor(provider, settings=settings)
    kwargs = dict(source_identity=identity, frozen_content=CONTENT)
    primary = extractor.extract(chunk, primary_value_targets=(HABIT,), **kwargs)
    targeted = extractor.extract_targeted(chunk, (HABIT,), target_ordinal=1, **kwargs)
    assert primary.diagnostics.outcome != "completed" and targeted.diagnostics.outcome == "completed"
    return source, chunk, primary, targeted, provider, settings


def reconcile(pair, *, budget=30000, checkpoint=lambda: None, admit_ok=True):
    source, chunk, primary, targeted, provider, settings = pair
    manager = ClaimResolutionManager()
    manager.capture_primary(source, chunk, primary)
    manager.capture_targeted(source, chunk, HABIT, targeted)
    baseline = SimpleNamespace(character=HABIT.character, dimension=HABIT.dimension, trait_key=HABIT.trait_key,
                               statement=HABIT.baseline_hint, contexts=HABIT.habit_contexts, polarity=HABIT.baseline_polarity)
    row = SimpleNamespace(candidate_id="candidate-1", payload_sha256="b" * 64)
    ledger = MaterialGapLedger(scoped_resolution=True)
    ledger.add(primary, source, "primary_extraction")
    admitted, charges, reasons = [], [], Counter()
    manager.reconcile(baselines=[(row, baseline, None, HABIT.character)], provider=provider, settings=settings,
                      budget=lambda: budget, charge=charges.append, checkpoint=checkpoint,
                      admit=lambda signal, *_: admitted.append(signal) is None and admit_ok, ledger=ledger, reasons=reasons)
    return manager, ledger, admitted, charges, reasons


def test_original_raw_claim_is_rechecked_and_only_its_target_gets_a_final_receipt():
    pair = paired()
    source, _, primary, targeted, provider, _ = pair
    gap, = primary.material_gaps
    assert gap.reason == "object_unverified"  # The final validated cause replaces stale actor preflight.
    assert gap.claim_digest not in targeted.material_resolved_claims
    manager, ledger, admitted, charges, _ = reconcile(pair)
    assert len(provider.requests) == 3 and provider.generations == 2
    assert provider.requests[2].primary_claim == provider.requests[0].primary_claim
    assert provider.requests[2].primary_claim.statement != provider.requests[1].primary_claim.statement
    receipt, = manager.receipts
    assert receipt["method"] == "original_claim_recheck" and receipt["primary_claim_sha256"] == gap.claim_digest
    assert receipt["review_claim_sha256"] == provider.requests[0].primary_claim.claim_digest
    assert admitted[0].id == receipt["observation_id"] and len(charges) == 1
    assert manager.repairs_document(source.document.id, "candidate-1", 1)
    assert not manager.repairs_document(source.document.id, "candidate-2", 1)
    assert (source.input_id, "candidate-1", gap.claim_digest) in ledger.resolved_scopes
    assert (source.input_id, "candidate-2", gap.claim_digest) not in ledger.resolved_scopes
    assert primary.diagnostics.outcome != "completed" and primary.diagnostics.rejected_records == 1
    serialized = json.dumps(manager.payload(), ensure_ascii=False)
    assert "statement" not in serialized.replace("observation_statement_sha256", "") and TAIL not in serialized


def test_exact_later_raw_claim_certificate_requires_no_third_call():
    pair = paired(exact=True)
    manager, _, admitted, charges, _ = reconcile(pair)
    assert len(pair[4].requests) == 2 and not charges and not admitted
    assert manager.receipts[0]["method"] == "exact_claim"
    assert manager.scopes[0]["status"] == "complete"


def test_newly_supported_action_is_admitted_before_its_resolution_receipt_is_issued():
    pair = paired(targeted_middle=True)
    manager, _, admitted, _, _ = reconcile(pair)
    assert admitted[0].id != pair[3].signals[0].id
    assert manager.receipts[0]["observation_id"] == admitted[0].id
    denied = reconcile(paired(), admit_ok=False)[0]
    assert not denied.receipts and denied.scopes[0]["status"] == "partial"


@pytest.mark.parametrize("problem", ["unresolved", "budget", "unknown", "omitted", "unobserved", "other_target"])
def test_unresolved_or_unenumerated_material_never_claims_scope_complete(problem):
    pair = paired(unresolved=problem == "unresolved")
    state = pair[2].claim_resolution_state
    if problem == "unknown":
        state.gaps = (*state.gaps, DraftMaterialGap(reason="source_unverified"))
    elif problem == "omitted":
        state.omitted_count = 1
    elif problem == "unobserved":
        state.enumeration_complete = False
    elif problem == "other_target":
        signal = pair[3].signals[0]
        changed = signal.target_bound_target.model_copy(update={"habit_contexts": ("别的场合",)})
        target_result = pair[3].model_copy(update={"signals": (signal.model_copy(update={"target_bound_target": changed}),)})
        pair = (*pair[:3], target_result, *pair[4:])
    manager, _, _, _, _ = reconcile(pair, budget=0 if problem == "budget" else 30000)
    assert not manager.repairs_document(pair[0].document.id, "candidate-1", 1)
    assert all(scope["status"] == "partial" for scope in manager.scopes)
    assert len(pair[4].requests) <= 3


def test_cancellation_after_network_does_not_admit_or_clear_the_claim():
    pair = paired()
    calls = []
    def cancel():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        reconcile(pair, checkpoint=cancel)
    assert len(pair[4].requests) == 3


def test_one_confirmed_target_cannot_recheck_multiple_pending_original_claims():
    from dataclasses import replace
    pair = paired()
    state = pair[2].claim_resolution_state
    original = next(iter(state.pending.values()))
    sibling_key = "f" * 64
    state.pending[sibling_key] = replace(original, stable_key=sibling_key)
    state.gaps = (*state.gaps, state.gaps[0].model_copy(update={"claim_digest": sibling_key}))
    manager, _, _, charges, _ = reconcile(pair)
    assert len(pair[4].requests) == 3 and len(charges) == 1
    assert len(manager.receipts) == 1
    assert manager.scopes[0]["claim_count"] == 2 and manager.scopes[0]["status"] == "partial"


def test_native_engine_stage_and_service_persist_final_scope_without_rewriting_primary_history():
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    from app.character_claim_resolution import CLAIM_RESOLUTION_KEY
    from app.character_consistency_stage import CharacterConsistencyStage
    from app.character_drift import CHARACTER_HABIT_CONTEXT_REVIEW_SYSTEM_PROMPT, CHARACTER_REVIEW_POLICY_VERSION
    from app.character_explanation_review import EXPLANATION_REVIEW_SYSTEM_PROMPT
    from app.character_trait_extraction import CHARACTER_SIGNAL_DRAFT_ACTOR_PROMPT_V1, TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT
    from app.config import HABIT_OBJECT_CHAIN_POLICY_V1, CLAIM_RESOLUTION_POLICY_V1
    from app.db import AnalysisDiagnosticRow, AnalysisRunRow, CharacterObservationRow, SessionLocal
    from app.main import app, settings as application_settings, write_limiter
    from app.service import execute_analysis
    from tests.test_character_consistency_stage import (
        QueueProvider, _create_document, _context, _new_run, _run_stage, _response, _confirm_only_candidate,
    )
    from tests.test_character_single_event_stage import _RuntimeSingleEventProvider, _event

    profile = "发货前，廖辰会逐件清点器材箱。"
    formal = dict(record("廖辰会逐件清点器材箱"), evidence=profile, polarity="positive", stability="stable",
                  observation_kind="explicit_declaration", context="发货前", trait_key="inventory")

    class NativeProvider(ResolutionProvider):
        def __init__(self):
            super().__init__()
            self.trait_key = "inventory"
            self.calls = []

        def complete(self, system, user):
            if system == CHARACTER_HABIT_CONTEXT_REVIEW_SYSTEM_PROMPT:
                item = _event("C01")
                item["context_checks"] = [dict(citation="C01", relation="same", quote="装车发运")]
                value = dict(policy_version=CHARACTER_REVIEW_POLICY_VERSION, verdict="contradicts",
                             explanation="本次发货未逐件清点，与已确认习惯相反。", citations=["B01", "C01"], events=[item])
                return SimpleNamespace(text=json.dumps(value, ensure_ascii=False), prompt_tokens=9, completion_tokens=8)
            if system == EXPLANATION_REVIEW_SYSTEM_PROMPT:
                return _RuntimeSingleEventProvider.complete(self, system, user)
            if (system != TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15
                    and CHARACTER_SIGNAL_DRAFT_ACTOR_PROMPT_V1 not in system
                    and not system.startswith(TARGETED_CHARACTER_SIGNAL_SYSTEM_PROMPT)):
                return SimpleNamespace(text=_response(formal), prompt_tokens=9, completion_tokens=8)
            return super().complete(system, user)

    write_limiter.events.clear()
    provider = NativeProvider()
    captured = []
    stage_errors = []
    original_run = CharacterConsistencyStage.run
    stage_settings = _settings(character_consistency_stage_token_budget=100000, character_explanation_review_v1=True)
    stage_settings._character_habit_object_chain_v1 = stage_settings._character_claim_resolution_v1 = True

    def native_run(stage, *args, **kwargs):
        stage.settings, stage.provider = stage_settings, provider
        try:
            result = original_run(stage, *args, **kwargs)
        except Exception as exc:
            stage_errors.append((type(exc).__name__, str(exc)))
            raise
        captured.append(result)
        return result

    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "原主张终态联调"}).json()
        _create_document(client, project["id"], name="profile.md", role="character_profile", content=profile,
                         narrative_context=_context(publication="published"))
        seed = _new_run(client, project["id"])
        _run_stage(seed, QueueProvider(_response(formal)))
        candidate_id = _confirm_only_candidate(client, project["id"], seed)
        _create_document(client, project["id"], name="draft.md", role="chapter", content=CONTENT,
                         narrative_context=_context(publication="draft"))
        run_id = _new_run(client, project["id"])
        with SessionLocal() as db:
            run = db.get(AnalysisRunRow, run_id)
            run.batch_mode = "draft_review"
            run.batch_coverage = {**(run.batch_coverage or {}), "character_review_enabled": True,
                                  "habit_object_chain_policy": HABIT_OBJECT_CHAIN_POLICY_V1,
                                  "claim_resolution_policy": CLAIM_RESOLUTION_POLICY_V1}
            db.commit()
        with patch.object(application_settings, "enable_character_consistency", True), patch.object(
            CharacterConsistencyStage, "run", native_run,
        ):
            execute_analysis(run_id, raise_on_failure=True)
        assert captured, stage_errors
        result, = captured
        assert len(provider.requests) == 3
        assert len(result.issues) == 1 and not result.review_clues
        assert result.diagnostics["counts"]["model_incomplete_chunks"] == 1 and result.diagnostics["outcome"] == "partial"
        assert result.diagnostics["case_trace"][0]["material_coverage"] == "complete"
        payload = result.diagnostics[CLAIM_RESOLUTION_KEY]
        assert payload["final_scopes"][0]["status"] == "complete"
        with SessionLocal() as db:
            durable = db.get(AnalysisDiagnosticRow, run_id).payload["character_consistency"]
            assert durable[CLAIM_RESOLUTION_KEY] == payload
            receipt, = payload["receipts"]
            assert receipt["confirmed_candidate_id"] == candidate_id
            assert db.get(CharacterObservationRow, (run_id, receipt["observation_id"])) is not None
