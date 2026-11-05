"""Source selection is a proposal, independently checked before observation admission."""
import copy
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.character_action_selection import (
    HABIT_ACTION_SELECTION_SCHEMA_V1, HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1,
    ActionChoice, ActionSelectionRequest, ActionSelectionResponse, selection_request_digest,
    select_action_window, validate_action_choices, build_selected_action_review_request,
    selected_action_path, run_habit_action_selection,
)
from app.character_draft_actor_review import (
    TargetBoundDraftReviewRequest, TargetBoundDraftReviewTarget,
    TARGET_BOUND_DRAFT_REVIEW_PROMPT_V15, build_target_bound_draft_proposals,
    verify_target_bound_draft_review_source, evaluate_target_bound_draft_review,
    target_bound_draft_review_request_digest,
)
from app.character_draft_actor_review_provider import TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16
from app.character_trait_extraction import CharacterSignalChunk, CharacterSignalExtractor, CharacterSignalTarget
from tests.test_character_target_bound_draft_review_v4 import _source, _item
from tests.test_character_target_bound_draft_review_v3 import _settings

CONTENT = "林砚接过新送来的麦克风，沿台口摆成一列，连试音都省了便示意开场。"
OBJECT = "新送来的麦克风"
TARGET = CharacterSignalTarget(character="林砚", dimension="contextual_behavior", trait_key="sound_check",
    comparison_key="contextual_behavior:soundcheck", baseline_polarity="positive", requested_polarity="negative",
    baseline_hint="林砚每次开演前都会逐支试音", habit_contexts=("每次开演前",))


def settings():
    result = _settings(character_signal_token_budget=40000)
    result._character_habit_scene_scheduling_v1 = True
    result._character_habit_object_chain_v1 = True
    result._character_habit_action_selection_v1 = True
    return result


def selection(content=CONTENT, *, existing=()):
    from app.character_trait_extraction import _target_bound_review_target
    source = _source(content)
    request, omitted = select_action_window(source=source, frozen_content=content,
        target=_target_bound_review_target(TARGET, target_ordinal=1), line_start=1,
        line_end=len(content.splitlines()), existing_ranges=existing)
    return source, request, omitted


def choice_response(request, status="selected"):
    ids = tuple(c.support_id for c in request.window)
    return ActionSelectionResponse(schema_version=HABIT_ACTION_SELECTION_SCHEMA_V1,
        request_digest=selection_request_digest(request), status=status,
        items=(ActionChoice(fact_clause_id=ids[-1], actor_anchor_id=ids[0], path_ids=ids),) if status == "selected" else ())


def review_items(request, *, damaged=None):
    items = []
    for proposal in request.proposals:
        path = selected_action_path(request, proposal)
        item = _item(request, proposal, object_relation="not_applicable", observation_kind="action")
        item["action_steps"] = [dict(clause_id=c.support_id, quote=c.text, actor="target", actuality="actual", scene="same") for c in path]
        item["claim_object"] = dict(role="event_object", grounding="action_path", referent_relation="same",
            fact_quote=path[-1].text, antecedent_clause_id=path[0].support_id, antecedent_quote=OBJECT,
            object_steps=[dict(clause_id=c.support_id, quote=OBJECT if i == 0 else c.text,
                               grounding="explicit" if i == 0 else "omitted", referent_relation="same") for i, c in enumerate(path)])
        if damaged in {"actor", "actuality", "scene"}:
            item["action_steps"][-1][damaged] = dict(actor="other", actuality="non_actual", scene="changed")[damaged]
        elif damaged in {"different", "uncertain"}:
            item["claim_object"]["object_steps"][1]["referent_relation"] = damaged
        elif damaged == "object_quote":
            item["claim_object"]["object_steps"][0]["quote"] = "并不存在的物件"
        elif damaged == "claim_identity":
            item.update(claim_digest="a" * 64, claim_relation="equivalent")
        elif damaged == "correction":
            item["correction_relation"] = "corrected"
        items.append(item)
    return items


def evaluate(request, source, items, content=CONTENT):
    return evaluate_target_bound_draft_review(request, json.dumps(dict(schema_version=request.schema_version,
        request_digest=target_bound_draft_review_request_digest(request), items=items)), expected_source=source,
        frozen_content=content)


class SelectionProvider:
    def __init__(self, *, status="selected", damaged=None):
        self.status, self.damaged = status, damaged
        self.selection_requests, self.review_requests, self.calls, self.forks = [], [], [], []

    def fork_for_character_consistency(self, **kwargs):
        self.forks.append(("signal", kwargs["settings"].character_signal_max_completion_tokens))
        return self

    def fork_for_habit_scene_review(self, **kwargs):
        self.forks.append(("review", kwargs["completion_reserve"]))
        return self

    def complete(self, system, user):
        self.calls.append(system)
        if system == HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1:
            request = ActionSelectionRequest.model_validate_json(json.dumps(json.loads(user)["request"]))
            self.selection_requests.append(request)
            response = choice_response(request, self.status).model_dump(mode="json")
        elif system == TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16:
            body = json.loads(user.split("：\n", 1)[1])
            responses = []
            for window in body["windows"]:
                request = TargetBoundDraftReviewRequest.model_validate_json(json.dumps(window["request"]))
                self.review_requests.append(request)
                responses.append(dict(schema_version=request.schema_version, request_digest=window["request_digest"],
                                      items=review_items(request, damaged=self.damaged)))
            response = dict(schema_version="character-target-bound-draft-review-batch-v4",
                            batch_digest=body["batch_digest"], responses=responses)
        else:
            response = {"records": []}
        return SimpleNamespace(text=json.dumps(response, ensure_ascii=False), prompt_tokens=11, completion_tokens=7)


def extract(provider=None, config=None, **kwargs):
    provider = provider or SelectionProvider()
    source = _source(CONTENT)
    chunk = CharacterSignalChunk(source.document_id, "draft.md", CONTENT, 1, "draft")
    result = CharacterSignalExtractor(provider, settings=config or settings()).extract_targeted(
        chunk, (TARGET,), source_identity=source, frozen_content=CONTENT, **kwargs)
    return result, provider


def test_selected_non_template_action_reaches_source_bound_v16_without_legacy_enumeration():
    source, request, _ = selection()
    _, legacy = build_target_bound_draft_proposals(CONTENT, source, request.target, line_start=1, line_end=1,
        protocol_version="v4", primary_claim_review=True, allow_object_chain=True)
    assert not any(p.fact_clause_id == "L1:A3" for p in legacy)
    with patch("app.character_draft_actor_review.build_target_bound_draft_proposals", side_effect=AssertionError("legacy enumeration")):
        selected = build_selected_action_review_request(request, choice_response(request), frozen_content=CONTENT)
        assert verify_target_bound_draft_review_source(selected, source, frozen_content=CONTENT)
        supported, = evaluate(selected, source, review_items(selected)).decisions
    assert supported.verdict == "supported"
    assert supported.verified_claim_digest is None
    assert supported.verified_object.antecedent_quote == OBJECT


def test_existing_whole_line_evidence_does_not_hide_another_action_on_that_line():
    _, request, _ = selection(existing=((1, 1),))
    assert request.window[-1].support_id == "L1:A3"


def test_blank_paragraphs_do_not_invalidate_or_renumber_a_source_window():
    content = "\n\n" + CONTENT + "\n\n"
    source, selection_request, _ = selection(content)
    assert [c.support_id for c in selection_request.window] == ["L3:A1", "L3:A2", "L3:A3"]
    request = build_selected_action_review_request(selection_request, choice_response(selection_request), frozen_content=content)
    assert verify_target_bound_draft_review_source(request, source, frozen_content=content)
    assert request.proposals[0].line_start == request.proposals[0].line_end == 3


@pytest.mark.parametrize("damage", ["skip", "other_actor_anchor", "outside", "duplicate", "digest", "oversized"])
def test_selector_response_cannot_change_the_server_window_or_path(damage):
    _, request, _ = selection()
    value = choice_response(request).model_dump(mode="json")
    row = value["items"][0]
    if damage == "skip": row["path_ids"].pop(1)
    elif damage == "other_actor_anchor": row.update(actor_anchor_id="L1:A2", path_ids=["L1:A2", "L1:A3"])
    elif damage == "outside": row["path_ids"][-1] = row["fact_clause_id"] = "L2:A1"
    elif damage == "duplicate": value["items"].append(copy.deepcopy(row))
    elif damage == "digest": value["request_digest"] = "0" * 64
    else: value["items"] *= 3
    with pytest.raises((ValueError, ValidationError)):
        validate_action_choices(request, ActionSelectionResponse.model_validate_json(json.dumps(value)))


@pytest.mark.parametrize("damage", ["actor", "actuality", "scene", "different", "uncertain", "object_quote", "claim_identity", "correction"])
def test_selection_never_substitutes_for_independent_semantic_or_object_checks(damage):
    source, selected, _ = selection()
    request = build_selected_action_review_request(selected, choice_response(selected), frozen_content=CONTENT)
    assert all(d.verdict != "supported" for d in evaluate(request, source, review_items(request, damaged=damage)).decisions)


def test_new_source_fields_cannot_upgrade_v15_and_mutated_source_cannot_be_reviewed():
    source, selection_request, _ = selection()
    request = build_selected_action_review_request(selection_request, choice_response(selection_request), frozen_content=CONTENT)
    with pytest.raises(ValidationError):
        TargetBoundDraftReviewRequest.model_validate_json(request.model_copy(update={"prompt_version": TARGET_BOUND_DRAFT_REVIEW_PROMPT_V15}).model_dump_json())
    assert not verify_target_bound_draft_review_source(request, source, frozen_content=CONTENT.replace("麦克风", "道具"))
    damaged = request.model_copy(update={"selection_digest": "0" * 64})
    assert not verify_target_bound_draft_review_source(damaged, source, frozen_content=CONTENT)


def test_targeted_pipeline_replaces_generation_and_admits_only_the_reviewed_source_object():
    result, provider = extract()
    assert provider.calls == [HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16]
    observed, = result.signals
    assert observed.key_object == OBJECT and observed.statement == "林砚连试音都省了便示意开场"
    assert observed.evidence.text == CONTENT
    assert result.selection_only and not result.material_resolved_claims and result.claim_resolution_state is None
    assert result.diagnostics.attempted_calls == 2 and result.diagnostics.outcome == "completed"
    assert ("signal", 768) in provider.forks
    assert next(cap for kind, cap in provider.forks if kind == "review") > 768


@pytest.mark.parametrize("status", ["none", "uncertain"])
def test_selector_abstention_is_not_an_observation_or_a_claim_resolution(status):
    result, provider = extract(SelectionProvider(status=status))
    assert len(provider.calls) == 1 and not result.signals and not result.material_resolved_claims
    assert result.diagnostics.outcome == ("partial" if status == "uncertain" else "completed")


def test_budget_and_cancellation_do_not_send_an_unfunded_or_cancelled_review():
    config = settings().model_copy(update={"character_signal_token_budget": 0})
    result, provider = extract(config=config)
    assert not provider.calls and result.diagnostics.attempted_calls == 0
    calls = []
    def checkpoint():
        calls.append(1)
        if len(calls) == 2: raise RuntimeError("cancelled")
    provider = SelectionProvider()
    with pytest.raises(RuntimeError, match="cancelled"):
        extract(provider, checkpoint=checkpoint)
    assert provider.calls == [HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1]


def test_old_authorization_keeps_the_old_targeted_call_without_a_selector():
    config = settings()
    config._character_habit_action_selection_v1 = False
    _, provider = extract(config=config)
    assert HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1 not in provider.calls


def test_native_authorized_stage_service_persists_selected_observation():
    """Frozen author choice + native orchestration; the model responses are a fixture."""
    from fastapi.testclient import TestClient
    from sqlalchemy import func, select
    from app.character_consistency_stage import CharacterConsistencyStage
    from app.character_drift import CHARACTER_HABIT_CONTEXT_REVIEW_SYSTEM_PROMPT, CHARACTER_REVIEW_POLICY_VERSION
    from app.config import HABIT_ACTION_SELECTION_POLICY_V1, HABIT_OBJECT_CHAIN_POLICY_V1, HABIT_SCENE_POLICY_V1
    from app.db import AnalysisRunRow, AnalysisDiagnosticRow, AccountProviderConfigRow, CharacterObservationRow, SessionLocal
    from app.main import app, write_limiter
    from app.service import execute_analysis, _CharacterConsistencyAccountingProvider
    from tests.test_habit_scene_policy import AuthorSettings
    from tests.test_habit_claim_recovery import record
    from tests.test_character_consistency_stage import (
        _create_document, _context, _new_run, _run_stage, QueueProvider, _response, _confirm_only_candidate,
    )
    from tests.test_character_single_event_stage import _event
    profile = "每次开演前，林砚都会逐支试音。"
    formal = dict(record("林砚都会逐支试音"), character="林砚", trait_key="sound_check", evidence=profile,
                  context="每次开演前", polarity="positive", stability="stable", observation_kind="explicit_declaration")

    class NativeProvider(SelectionProvider):
        def complete(self, system, user):
            if system == CHARACTER_HABIT_CONTEXT_REVIEW_SYSTEM_PROMPT:
                item = _event("C01")
                item.update(verdict="needs_confirmation")
                item["context_checks"] = [dict(citation="C01", relation="same", quote="示意开场")]
                value = dict(policy_version=CHARACTER_REVIEW_POLICY_VERSION, verdict="needs_confirmation",
                             explanation="需要作者确认这次开场的具体情况。", citations=["B01", "C01"], events=[item])
                return SimpleNamespace(text=json.dumps(value, ensure_ascii=False), prompt_tokens=11, completion_tokens=7)
            if "character_profile" in user and system not in {HABIT_ACTION_SELECTION_SYSTEM_PROMPT_V1, TARGET_BOUND_DRAFT_REVIEW_SYSTEM_PROMPT_V16}:
                return SimpleNamespace(text=_response(formal), prompt_tokens=11, completion_tokens=7)
            return super().complete(system, user)

    write_limiter.events.clear()
    provider = NativeProvider()
    captured = []
    errors = []
    original_run = CharacterConsistencyStage.run
    def observed_run(stage, *args, **kwargs):
        # Keep the service-derived frozen authorization and real stage/persistence.
        assert stage.settings._character_habit_action_selection_v1
        stage.provider = _CharacterConsistencyAccountingProvider(stage.settings, stage.provider.usage,
                                                                 signal_provider=provider, drift_provider=provider)
        try:
            result = original_run(stage, *args, **kwargs)
        except Exception as exc:
            errors.append((type(exc).__name__, str(exc)))
            raise
        captured.append(result)
        return result

    with TestClient(app) as client:
        project = client.post("/api/v1/projects", json={"name": "固定源文选择贯通"}).json()
        _create_document(client, project["id"], name="profile.md", role="character_profile", content=profile,
                         narrative_context=_context(publication="published"))
        seed = _new_run(client, project["id"])
        _run_stage(seed, QueueProvider(_response(formal)))
        _confirm_only_candidate(client, project["id"], seed)
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
                "habit_action_selection_policy": HABIT_ACTION_SELECTION_POLICY_V1}
            db.commit()
        native = AuthorSettings(_env_file=None, **{**settings().model_dump(),
            "provider_runtime_config_id": config_id, "provider_runtime_user_id": owner,
            "character_consistency_stage_token_budget": 100000, "per_run_token_budget": 120000})
        runtime = SimpleNamespace(settings=native, identity=None, available=True, unavailable_reason=None)
        with patch("app.service.runtime_for_analysis_run", return_value=runtime), patch.object(CharacterConsistencyStage, "run", observed_run):
            execute_analysis(run_id, raise_on_failure=True)
        assert captured, errors
        result, = captured
        assert len(provider.selection_requests) == len(provider.review_requests) == 1
        assert result.diagnostics["counts"]["targeted_record_accepted_count"] == 1
        assert result.diagnostics["reason_counts"]["habit_selection_not_exhaustive"] == 1
        with SessionLocal() as db:
            observations = db.scalars(select(CharacterObservationRow).where(CharacterObservationRow.run_id == run_id)).all()
            assert len(observations) == 1
            value = observations[0].payload
            assert value["key_object"] == OBJECT and value["evidence"]["text"] == CONTENT
            assert value["statement"] == "林砚连试音都省了便示意开场"
            assert any(match.get("baseline") for match in value["matches"])
            durable = db.get(AnalysisDiagnosticRow, run_id).payload["character_consistency"]
            assert durable["counts"]["targeted_record_accepted_count"] == 1
            assert "claim_resolution_v1" not in durable
