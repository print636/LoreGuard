"""Original empty-object claims require a new, admitted source-bound proof."""
from collections import Counter
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.character_claim_resolution import ClaimResolutionManager, CLAIM_RESOLUTION_KEY
from app.character_draft_actor_review import (
    TargetBoundDraftReviewRequest, evaluate_target_bound_draft_review,
    target_bound_draft_review_request_digest, verify_target_bound_draft_review_source,
    TARGET_BOUND_DRAFT_REVIEW_PROMPT_V16,
    TARGET_BOUND_DRAFT_REVIEW_PROMPT_V17,
)
from app.character_draft_actor_review_provider import (
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15,
    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V17, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V18,
)
from app.config import SELECTED_CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V2
from app.character_material_coverage import DraftMaterialGap, MaterialGapLedger
from app.character_scope_review import ScopeReviewSourceIdentity
from app.character_selected_claim_resolution import build_selected_claim_review_request
from app.character_trait_extraction import CharacterSignalChunk, CharacterSignalExtractor
from tests.test_character_draft_actor_stage_wiring import _draft_source
from tests.test_character_target_bound_draft_review_v4 import _item
from tests.test_habit_claim_recovery import record
from tests.test_habit_action_selection import (
    CONTENT, OBJECT, TARGET, SelectionProvider, settings as selection_settings, review_items,
    HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16,
)

STATEMENT = "连试音都省了便示意开场"
CONTEXT = "每次开演前"
RAW = dict(record(STATEMENT), character=TARGET.character, trait_key=TARGET.trait_key,
           context=CONTEXT, evidence=CONTENT)


def settings(policy=SELECTED_CLAIM_RESOLUTION_POLICY_V1):
    config = selection_settings().model_copy(update={"character_signal_package_max_attempts": 1})
    config._character_claim_resolution_v1 = True
    config._character_selected_claim_resolution_policy = policy
    return config


class SelectedClaimProvider(SelectionProvider):
    def __init__(self, *, damage=None, key_object=""):
        super().__init__()
        self.damage = damage
        self.raw = {**RAW, "key_object": key_object}
        self.original_requests, self.resolution_requests = [], []

    def complete(self, system, user):
        if system in {HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16}:
            return super().complete(system, user)
        self.calls.append(system)
        if system in {TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15,
                      TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V17, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V18}:
            batch = json.loads(user.split("：\n", 1)[1])
            responses = []
            for window in batch["windows"]:
                request = TargetBoundDraftReviewRequest.model_validate_json(json.dumps(window["request"]))
                if system in {TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15}:
                    self.original_requests.append(request)
                    items = [_item(request, p, verdict="rejected", object_relation="not_applicable",
                        observation_kind="action", axis_relation="different", polarity_relation="neutral",
                        claim_digest=request.primary_claim.claim_digest, claim_relation="different") for p in request.proposals]
                    if system == TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15:
                        for item in items: item["claim_object"] = None
                else:
                    self.resolution_requests.append(request)
                    items = v17_items(request, damage=self.damage)
                responses.append(dict(schema_version=request.schema_version,
                                      request_digest=window["request_digest"], items=items))
            response = dict(schema_version="character-target-bound-draft-review-batch-v4",
                            batch_digest=batch["batch_digest"], responses=responses)
        else:
            response = {"records": [self.raw]}
        return SimpleNamespace(text=json.dumps(response, ensure_ascii=False), prompt_tokens=11, completion_tokens=7)


def v17_items(request, *, damage=None):
    items = review_items(request, damaged=damage)
    for item in items:
        item.update(claim_digest=request.original_claim.claim_digest, claim_relation="equivalent", claim_mapping="unique")
        if request.prompt_version.endswith("-v18"):
            item["original_object_relation"] = "same" if request.original_claim.key_object else "unspecified"
            if damage in {"original_different", "original_uncertain", "original_unspecified", "original_same"}:
                item["original_object_relation"] = damage.removeprefix("original_")
            elif damage == "original_missing":
                item.pop("original_object_relation")
        if damage in {"multiple", "uncertain", "none"}:
            item["claim_mapping"] = damage
        elif damage == "different_claim":
            item["claim_relation"] = "different"
        elif damage == "missing_mapping":
            item.pop("claim_mapping")
        elif damage == "wrong_claim":
            item["claim_digest"] = "0" * 64
    return items


def paired(*, damage=None, enabled=True, policy=SELECTED_CLAIM_RESOLUTION_POLICY_V1, key_object=""):
    config = settings(policy if enabled else None)
    source = _draft_source(CONTENT)
    chunk = CharacterSignalChunk(source.document.id, source.document.name, CONTENT, 1, "draft")
    identity = ScopeReviewSourceIdentity(run_input_id=source.input_id, document_id=source.document.id,
        document_version=source.document_version, content_sha256=source.content_sha256)
    provider = SelectedClaimProvider(damage=damage, key_object=key_object)
    extractor = CharacterSignalExtractor(provider, settings=config)
    primary = extractor.extract(chunk, primary_value_targets=(TARGET,), source_identity=identity, frozen_content=CONTENT)
    targeted = extractor.extract_targeted(chunk, (TARGET,), target_ordinal=1,
                                         source_identity=identity, frozen_content=CONTENT)
    assert primary.diagnostics.outcome != "completed" and len(targeted.signals) == 1
    return source, chunk, primary, targeted, provider, config


def reconcile(pair, *, budget=40000, checkpoint=lambda: None, admit_ok=True, origin_ok=True):
    source, chunk, primary, targeted, provider, config = pair
    manager = ClaimResolutionManager()
    manager.capture_primary(source, chunk, primary)
    manager.capture_targeted(source, chunk, TARGET, targeted)
    baseline = SimpleNamespace(character=TARGET.character, dimension=TARGET.dimension, trait_key=TARGET.trait_key,
        statement=TARGET.baseline_hint, contexts=TARGET.habit_contexts, polarity=TARGET.baseline_polarity)
    row = SimpleNamespace(candidate_id="candidate-1", payload_sha256="b" * 64)
    ledger = MaterialGapLedger(scoped_resolution=True)
    ledger.add(primary, source, "primary_extraction")
    admitted, charges, reasons = [], [], Counter()
    manager.reconcile(baselines=[(row, baseline, None, TARGET.character)], provider=provider, settings=config,
        budget=lambda: budget, charge=charges.append, checkpoint=checkpoint,
        admit=lambda signal, *_: admitted.append(signal) is None and admit_ok,
        origin_admitted=lambda signal: origin_ok and signal in targeted.signals, ledger=ledger, reasons=reasons)
    return manager, ledger, admitted, charges, reasons


def test_original_empty_key_and_full_window_get_an_independent_admitted_certificate():
    pair = paired()
    source, _, primary, targeted, provider, _ = pair
    manager, ledger, admitted, charges, _ = reconcile(pair)
    request, = provider.resolution_requests
    old_claim = provider.original_requests[0].primary_claim
    assert request.primary_claim is None
    assert request.original_claim.statement == old_claim.statement == STATEMENT
    assert request.original_claim.context == old_claim.context == CONTEXT
    assert request.original_claim.key_object == "" and old_claim.key_object is None
    assert request.original_claim.claim_digest == old_claim.claim_digest
    origin, = targeted.selected_action_certificates
    assert request.lines == origin.request.lines and request.lines[0].text == CONTENT
    assert request.origin_request_sha256 == target_bound_draft_review_request_digest(origin.request)
    assert admitted == list(targeted.signals) and len(charges) == 1
    receipt, = manager.receipts
    gap, = primary.material_gaps
    assert receipt["method"] == "selected_claim_recheck" and receipt["primary_claim_sha256"] == gap.claim_digest
    assert receipt["origin_observation_id"] == receipt["observation_id"] == admitted[0].id
    assert receipt["fact_clause_id"] == "L1:A3"
    assert manager.repairs_document(source.document.id, "candidate-1", 1)
    assert not manager.repairs_document(source.document.id, "candidate-2", 1)
    assert (source.input_id, "candidate-1", gap.claim_digest) in ledger.resolved_scopes
    assert (source.input_id, "candidate-2", gap.claim_digest) not in ledger.resolved_scopes
    assert primary.diagnostics.outcome != "completed" and primary.diagnostics.rejected_records == 1


@pytest.mark.parametrize("damage", ["multiple", "uncertain", "none", "different_claim", "missing_mapping",
    "wrong_claim", "actor", "actuality", "scene", "different", "object_quote", "correction"])
def test_window_mapping_and_semantic_failures_cannot_reuse_the_v16_admission(damage):
    pair = paired(damage=damage)
    manager, ledger, admitted, _, _ = reconcile(pair)
    assert not manager.receipts and not admitted and manager.scopes[0]["status"] == "partial"
    assert len(pair[4].resolution_requests) == 1
    if damage in {"multiple", "none", "different_claim"}:
        key = pair[2].material_gaps[0].claim_digest
        assert ledger.reason_scopes[(pair[0].input_id, "candidate-1", key)] == "record_unverified"
        assert (pair[0].input_id, "candidate-2", key) not in ledger.reason_scopes


def test_old_v16_cannot_be_reinterpreted_as_an_original_claim_certificate():
    pair = paired()
    pending = next(iter(pair[2].claim_resolution_state.pending.values()))
    origin = pair[3].selected_action_certificates[0]
    identity = ScopeReviewSourceIdentity(**{k: getattr(origin.request, k) for k in ScopeReviewSourceIdentity.model_fields})
    request = build_selected_claim_review_request(pending, origin, source_identity=identity, frozen_content=CONTENT)
    with pytest.raises(ValidationError):
        TargetBoundDraftReviewRequest.model_validate_json(request.model_copy(update={
            "prompt_version": TARGET_BOUND_DRAFT_REVIEW_PROMPT_V16}).model_dump_json())
    items = review_items(origin.request)
    items[0]["claim_mapping"] = "unique"
    raw = json.dumps(dict(schema_version=origin.request.schema_version,
        request_digest=target_bound_draft_review_request_digest(origin.request), items=items))
    assert evaluate_target_bound_draft_review(origin.request, raw, expected_source=identity,
        frozen_content=CONTENT).decisions[0].verdict != "supported"
    assert not verify_target_bound_draft_review_source(request, identity, frozen_content=CONTENT,
        expected_primary_claim=request.original_claim.model_copy(update={"statement": "另一个原主张"}))


@pytest.mark.parametrize("problem", ["old_policy", "unadmitted_origin", "unadmitted_final", "budget", "unknown", "omitted", "unobserved", "other_source", "other_target"])
def test_unavailable_origin_or_incomplete_material_does_not_repair_coverage(problem):
    pair = paired(enabled=problem != "old_policy")
    state = pair[2].claim_resolution_state
    if problem == "unknown": state.gaps = (*state.gaps, DraftMaterialGap(reason="source_unverified"))
    elif problem == "omitted": state.omitted_count = 1
    elif problem == "unobserved": state.enumeration_complete = False
    elif problem in {"other_source", "other_target"}:
        origin = pair[3].selected_action_certificates[0]
        request = origin.request.model_copy(update={"content_sha256": "0" * 64}) if problem == "other_source" else origin.request.model_copy(
            update={"target": origin.request.target.model_copy(update={"habit_contexts": ("别的场合",)})})
        result = pair[3].model_copy(update={"selected_action_certificates": (replace(origin, request=request),)})
        pair = (*pair[:3], result, *pair[4:])
    manager, _, _, _, _ = reconcile(pair, budget=0 if problem == "budget" else 40000,
        origin_ok=problem != "unadmitted_origin", admit_ok=problem != "unadmitted_final")
    assert not manager.repairs_document(pair[0].document.id, "candidate-1", 1)
    assert len(pair[4].resolution_requests) <= 1
    if problem in {"old_policy", "unadmitted_origin", "budget", "other_source", "other_target"}:
        assert not pair[4].resolution_requests


@pytest.mark.parametrize("key_object", ["", "麦克风"])
def test_v2_prioritizes_the_admitted_path_and_preserves_the_complete_original_key(key_object):
    pair = paired(policy=SELECTED_CLAIM_RESOLUTION_POLICY_V2, key_object=key_object)
    manager, _, admitted, charges, _ = reconcile(pair)
    request, = pair[4].resolution_requests
    assert request.prompt_version.endswith("-v18")
    assert request.original_claim.key_object == key_object
    assert request.original_claim.statement == pair[4].original_requests[0].primary_claim.statement == STATEMENT
    assert request.original_claim.claim_digest == pair[4].original_requests[0].primary_claim.claim_digest
    assert request.proposals[0].fact_clause_id == "L1:A3"
    assert admitted[0].key_object == OBJECT and len(charges) == 1
    assert manager.receipts[0]["method"] == "selected_claim_recheck_v2"
    assert manager.receipts[0]["primary_claim_sha256"] == pair[2].material_gaps[0].claim_digest
    assert manager.scopes[0]["status"] == "complete"
    assert len(pair[4].original_requests) == 1  # No second legacy proposal request.
    if key_object:
        with pytest.raises(ValidationError):
            TargetBoundDraftReviewRequest.model_validate_json(request.model_copy(update={
                "prompt_version": TARGET_BOUND_DRAFT_REVIEW_PROMPT_V17}).model_dump_json())


def test_v18_protocol_accepts_semantically_verified_alias_without_a_new_literal_gate():
    """Protocol fixture only: legacy primary currently does not retain this alias."""
    from app.character_selected_claim_resolution import review_selected_original_claim
    from app.narrative_context import payload_sha256
    pair = paired(policy=SELECTED_CLAIM_RESOLUTION_POLICY_V2, key_object="麦克风")
    pending = next(iter(pair[2].claim_resolution_state.pending.values()))
    original_record = pending.deferred.record.model_copy(update={"key_object": "话筒"})
    # Supply a retained original at this module's boundary, not an assertion
    # that upstream extraction currently admits or retains nonliteral aliases.
    raw_key = payload_sha256({"source": pair[0].content_sha256, "document": pair[0].document.id,
                              "record": {**pair[4].raw, "key_object": "话筒"}})
    pending = replace(pending, stable_key=raw_key, deferred=replace(pending.deferred, record=original_record))
    origin = pair[3].selected_action_certificates[0]
    identity = ScopeReviewSourceIdentity(**{k: getattr(origin.request, k) for k in ScopeReviewSourceIdentity.model_fields})
    outcome = review_selected_original_claim(pending, origin, chunk=pair[1], source_identity=identity,
        frozen_content=CONTENT, provider=pair[4], settings=pair[5], token_budget=40000)
    request, = pair[4].resolution_requests
    assert request.original_claim.key_object == "话筒" and "话筒" not in CONTENT
    assert outcome.signals[0].key_object == OBJECT and outcome.claim_certificates
    assert not pair[2].material_resolved_claims  # No manager/admit or coverage proof in this fixture.


@pytest.mark.parametrize("key_object,damage", [("麦克风", "original_different"), ("麦克风", "original_uncertain"),
    ("麦克风", "original_unspecified"), ("麦克风", "original_missing"), ("麦克风", "multiple"),
    ("", "original_same"), ("", "object_quote")])
def test_v18_requires_original_object_semantics_and_never_falls_back_after_review(key_object, damage):
    pair = paired(policy=SELECTED_CLAIM_RESOLUTION_POLICY_V2, key_object=key_object, damage=damage)
    manager, _, admitted, charges, _ = reconcile(pair)
    assert not manager.receipts and not admitted and manager.scopes[0]["status"] == "partial"
    assert len(pair[4].resolution_requests) == len(pair[4].original_requests) == len(charges) == 1


@pytest.mark.parametrize("policy,origin_ok", [(SELECTED_CLAIM_RESOLUTION_POLICY_V1, True),
    (SELECTED_CLAIM_RESOLUTION_POLICY_V2, False)])
def test_old_policy_or_unadmitted_origin_keeps_the_legacy_nonempty_claim_route(policy, origin_ok):
    pair = paired(policy=policy, key_object="麦克风")
    manager, _, _, _, _ = reconcile(pair, origin_ok=origin_ok)
    assert not pair[4].resolution_requests and len(pair[4].original_requests) == 2
    assert not manager.receipts and manager.scopes[0]["status"] == "partial"


def test_shared_target_limit_and_cancellation_preserve_unresolved_claims():
    pair = paired()
    state = pair[2].claim_resolution_state
    original = next(iter(state.pending.values()))
    state.pending["f" * 64] = replace(original, stable_key="f" * 64,
        deferred=replace(original.deferred, record=original.deferred.record.model_copy(update={"key_object": "麦克风"})))
    state.gaps = (*state.gaps, state.gaps[0].model_copy(update={"claim_digest": "f" * 64}))
    manager, _, _, charges, _ = reconcile(pair)
    assert len(charges) == len(pair[4].resolution_requests) == len(manager.receipts) == 1
    assert manager.scopes[0]["status"] == "partial" and manager.scopes[0]["claim_count"] == 2
    pair = paired()
    checkpoints = []
    def cancel():
        checkpoints.append(1)
        if len(checkpoints) == 2: raise RuntimeError("cancelled")
    with pytest.raises(RuntimeError, match="cancelled"):
        reconcile(pair, checkpoint=cancel)
    assert len(pair[4].resolution_requests) == 1


@pytest.mark.parametrize("policy,key_object", [(SELECTED_CLAIM_RESOLUTION_POLICY_V1, ""),
    (SELECTED_CLAIM_RESOLUTION_POLICY_V2, "麦克风")])
def test_native_service_persists_selected_claim_proof_and_origin_without_rewriting_history(policy, key_object):
    """Real authorization/orchestration/journal/storage; deterministic model fixture."""
    from fastapi.testclient import TestClient
    from sqlalchemy import func, select
    from app.character_consistency_stage import CharacterConsistencyStage
    from app.character_drift import CHARACTER_HABIT_CONTEXT_REVIEW_SYSTEM_PROMPT, CHARACTER_REVIEW_POLICY_VERSION
    from app.config import (HABIT_ACTION_SELECTION_POLICY_V1, HABIT_OBJECT_CHAIN_POLICY_V1,
        HABIT_SCENE_POLICY_V1, CLAIM_RESOLUTION_POLICY_V1, SELECTED_CLAIM_RESOLUTION_POLICY_V1)
    from app.db import AnalysisRunRow, AnalysisDiagnosticRow, AccountProviderConfigRow, CharacterObservationRow, SessionLocal
    from app.main import app, write_limiter
    from app.service import execute_analysis, _CharacterConsistencyAccountingProvider
    from tests.test_habit_scene_policy import AuthorSettings
    from tests.test_character_consistency_stage import (
        _create_document, _context, _new_run, _run_stage, QueueProvider, _response, _confirm_only_candidate,
    )
    from tests.test_character_single_event_stage import _event
    profile = "每次开演前，林砚都会逐支试音。"
    formal = dict(RAW, statement="林砚都会逐支试音", evidence=profile,
                  polarity="positive", stability="stable", observation_kind="explicit_declaration")

    class NativeProvider(SelectedClaimProvider):
        def complete(self, system, user):
            if system == CHARACTER_HABIT_CONTEXT_REVIEW_SYSTEM_PROMPT:
                item = _event("C01")
                item.update(verdict="needs_confirmation")
                item["context_checks"] = [dict(citation="C01", relation="same", quote="示意开场")]
                value = dict(policy_version=CHARACTER_REVIEW_POLICY_VERSION, verdict="needs_confirmation",
                    explanation="需要作者确认这次开场的具体情况。", citations=["B01", "C01"], events=[item])
                return SimpleNamespace(text=json.dumps(value, ensure_ascii=False), prompt_tokens=11, completion_tokens=7)
            if "character_profile" in user and system not in {HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1,
                    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V13, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16,
                    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V15, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V17,
                    TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V18}:
                return SimpleNamespace(text=_response(formal), prompt_tokens=11, completion_tokens=7)
            return super().complete(system, user)

    write_limiter.events.clear()
    provider, captured, errors = NativeProvider(key_object=key_object), [], []
    original_run = CharacterConsistencyStage.run
    def observed_run(stage, *args, **kwargs):
        assert stage.settings._character_selected_claim_resolution_policy == policy
        stage.provider = _CharacterConsistencyAccountingProvider(stage.settings, stage.provider.usage,
            signal_provider=provider, drift_provider=provider)
        try: result = original_run(stage, *args, **kwargs)
        except Exception as exc:
            errors.append((type(exc).__name__, str(exc)))
            raise
        captured.append(result)
        return result

    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "空受事原主张独立核实贯通"}).json()
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
            owner = run.requested_by_user_id
            revision = (db.scalar(select(func.max(AccountProviderConfigRow.revision)).where(
                AccountProviderConfigRow.user_id == owner)) or 0) + 1
            config = AccountProviderConfigRow(user_id=owner, revision=revision, base_url="https://selection.invalid/v1",
                model_name="fixture", endpoint_sha256="a" * 64, state="usable")
            db.add(config)
            db.flush()
            config_id = config.id
            run.provider_config_id = config_id
            run.batch_mode = "draft_review"
            run.batch_coverage = {**(run.batch_coverage or {}), "character_review_enabled": True,
                "character_baseline_run_id": seed, "habit_scene_policy": HABIT_SCENE_POLICY_V1,
                "habit_object_chain_policy": HABIT_OBJECT_CHAIN_POLICY_V1,
                "habit_action_selection_policy": HABIT_ACTION_SELECTION_POLICY_V1,
                "claim_resolution_policy": CLAIM_RESOLUTION_POLICY_V1,
                "selected_claim_resolution_policy": policy}
            db.commit()
        native = AuthorSettings(_env_file=None, **{**settings().model_dump(),
            "provider_runtime_config_id": config_id, "provider_runtime_user_id": owner,
            "character_consistency_stage_token_budget": 100000, "per_run_token_budget": 120000})
        runtime = SimpleNamespace(settings=native, identity=None, available=True, unavailable_reason=None)
        with patch("app.service.runtime_for_analysis_run", return_value=runtime), patch.object(CharacterConsistencyStage, "run", observed_run):
            execute_analysis(run_id, raise_on_failure=True)
        assert captured, errors
        result, = captured
        assert len(provider.selection_requests) == len(provider.review_requests) == len(provider.resolution_requests) == 1
        assert result.diagnostics["counts"]["model_incomplete_chunks"] == 1
        assert result.diagnostics["case_trace"][0]["material_coverage"] == "complete"
        assert not result.issues and len(result.review_clues) == 1  # Main review still controls the verdict.
        payload = result.diagnostics[CLAIM_RESOLUTION_KEY]
        assert payload["final_scopes"][0]["status"] == "complete"
        with SessionLocal() as db:
            observations = db.scalars(select(CharacterObservationRow).where(CharacterObservationRow.run_id == run_id)).all()
            assert len(observations) == 1
            value = observations[0].payload
            assert value["key_object"] == OBJECT and value["evidence"]["text"] == CONTENT
            origin = value["selected_action_origin"]
            receipt, = payload["receipts"]
            assert receipt["method"] == ("selected_claim_recheck_v2" if policy == SELECTED_CLAIM_RESOLUTION_POLICY_V2 else "selected_claim_recheck")
            assert origin["request_sha256"] == receipt["origin_request_sha256"]
            assert receipt["origin_observation_id"] == receipt["observation_id"] == value["observation_id"]
            assert receipt["confirmed_candidate_id"] == candidate_id
            durable = db.get(AnalysisDiagnosticRow, run_id).payload["character_consistency"]
            assert durable[CLAIM_RESOLUTION_KEY] == payload
