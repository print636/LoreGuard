import argparse
import hashlib
import json
import shutil

import pytest

import scripts.run_character_axis_live as axis_live


def _hash(data):
    return hashlib.sha256(data).hexdigest()


def _fixture(tmp_path):
    root = tmp_path / "axis"
    files = {}
    for suite in axis_live.SUITES:
        folder = root / suite
        folder.mkdir(parents=True)
        contents = {
            name: b"source anchor\n" for name, _ in axis_live.BASELINE_FILES
        }
        contents[axis_live.DRAFT_FILE] = b"draft action\n"
        plan = {
            "schema_version": "character-axis-review-plan-v1",
            "suite": suite,
            "world_id": f"world-{suite}",
            "approved_axes": [{
                "axis_key": "safe_axis", "display_name": "Decision axis",
                "definition": "Whether the actor consults a partner",
                "trait_type": "core_personality",
            }],
            "candidate_decisions": [{
                "candidate_key": "wanted", "character_key": "Actor",
                "trait_type": "core_personality", "source_document": "02-character-profiles.md",
                "source_line": 1, "source_quote": "source anchor",
                "polarity": "positive", "stability": "core",
                "key_object": None, "approved_axis_key": "safe_axis",
            }],
        }
        contents["review-plan.json"] = json.dumps(plan).encode()
        # Deliberately invalid gold: preflight hashes it but does not parse it.
        contents["oracle.json"] = b"sealed until scoring"
        hashes = {}
        for name, data in contents.items():
            (folder / name).write_bytes(data)
            hashes[name] = _hash(data)
        files[suite] = {"world_id": f"world-{suite}", "case_count": 1, "files": hashes}
    manifest = {
        "schema_version": "character-axis-fixture-manifest-v1",
        "dataset_boundary": {
            "developer_visible": True, "blind_holdout": False, "production_quality": False,
        },
        "suites": files,
    }
    data = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(data)
    return root, _hash(data)


def _runtime_provenance(*, revision="a" * 40, artifact_hash="c" * 64):
    return {
        "schema_version": "loreguard-runtime-provenance-v3",
        "build": {
            "git_revision": revision,
            "service_artifact_sha256": artifact_hash,
        },
        "chat_provider": {
            "model_alias": "fixture-model",
            "endpoint_configuration_sha256": "e" * 64,
            "temperature": 0,
            "thinking_configured": True,
            "thinking_mode": "disabled",
        },
        "capabilities": {
            "model_extraction": False,
            "character_consistency": True,
            "issue_evidence_review": False,
            "record_repair_agent": False,
            "evidence_investigator": False,
            "embeddings": True,
        },
        "character_consistency_limits": {
            "sensitivity": "balanced",
            "stage_token_budget": 60000,
            "max_chunks_per_run": 24,
            "max_candidates_per_run": 64,
            "signal_max_chunk_chars": 8000,
            "signal_max_records": 64,
            "signal_targeted_max_targets_per_chunk": 8,
            "signal_provider_max_attempts": 2,
            "signal_package_max_attempts": 2,
            "signal_token_budget": 22000,
            "signal_max_completion_tokens": 4096,
            "signal_max_response_bytes": 64000,
            "signal_provider_max_completion_tokens": 4096,
            "signal_provider_max_response_bytes": 64000,
            "signal_timeout_seconds": 30.0,
            "signal_total_deadline_seconds": 30.0,
            "signal_provider_timeout_seconds": 30.0,
            "drift_max_observations": 12,
            "drift_max_support_evidence": 8,
            "drift_max_evidence_chars": 8000,
            "drift_token_budget": 4000,
            "drift_max_completion_tokens": 1024,
            "drift_max_response_bytes": 32000,
            "drift_provider_max_attempts": 2,
            "drift_provider_max_completion_tokens": 1024,
            "drift_provider_max_response_bytes": 32000,
            "drift_timeout_seconds": 30.0,
            "drift_total_deadline_seconds": 30.0,
            "drift_provider_timeout_seconds": 30.0,
            "per_run_token_budget": 100000,
            "daily_token_budget": 1000000,
        },
        "investigator_limits": {
            "max_seeds": 1,
            "max_decision_rounds": 6,
            "max_tool_calls": 6,
            "max_searches": 2,
            "max_reads": 2,
            "max_results": 12,
            "max_read_lines": 12,
            "max_span_chars": 12000,
            "token_budget": 16000,
            "max_agent_input_bytes": 131072,
            "provider_call_timeout_seconds": 30.0,
            "total_deadline_seconds": 60.0,
            "max_completion_tokens": 768,
            "max_response_bytes": 64000,
            "provider_attempts_per_decision": 1,
            "top_k": 6,
            "branch_limit": 30,
            "embedding_max_input_chars": 250000,
            "require_hybrid": True,
            "daily_token_budget": 100000,
        },
        "rag": {
            "strategy": "keyword+vector+entity-rrf",
            "profile_fingerprint": "b" * 64,
            "chunker_fingerprint": "d" * 64,
            "require_hybrid": True,
            "top_k": 6,
            "branch_limit": 30,
        },
    }


def test_verify_fixture_checks_every_file_before_any_http(tmp_path, monkeypatch):
    root, digest = _fixture(tmp_path)
    fixture = axis_live.verify_fixture(root, expected_manifest_sha256=digest)
    assert set(fixture.suites) == {"dev", "transfer"}
    assert fixture.suites["dev"].files["oracle.json"] == b"sealed until scoring"

    (root / "transfer" / axis_live.DRAFT_FILE).write_bytes(b"tampered")
    called = False

    def fail_client(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("no HTTP before all hashes match")

    monkeypatch.setattr(axis_live.httpx, "Client", fail_client)
    args = argparse.Namespace(
        dataset=str(root), manifest_sha256=digest, output_json=None,
        preflight_only=False, base_url="http://127.0.0.1:8000",
        run_timeout_seconds=1, diagnostic_dev_one_trial=False,
    )
    report, code = axis_live.run(args)
    assert code == 1
    assert report["failure"]["code"] == "fixture_file_hash_mismatch"
    assert called is False


def test_manifest_needs_independently_pinned_digest(tmp_path):
    root, digest = _fixture(tmp_path)
    with pytest.raises(axis_live.SafeFailure, match="manifest_digest_required"):
        axis_live.verify_fixture(root, expected_manifest_sha256=None)
    with pytest.raises(axis_live.SafeFailure, match="manifest_hash_mismatch"):
        axis_live.verify_fixture(root, expected_manifest_sha256="0" * 64)
    assert axis_live.verify_fixture(root, expected_manifest_sha256=digest)


def test_candidate_selector_uses_source_and_object_not_raw_model_label():
    selector = {
        "character_key": "Actor", "trait_type": "preference",
        "source_document": "02-character-profiles.md", "source_line": 4,
        "source_quote": "likes pear drink", "polarity": "positive",
        "stability": "stable", "key_object": "pear drink",
    }
    candidate = {
        "reviewable": True, "character_key": "Actor", "trait_type": "preference",
        "trait_key": "unregistered free-form label", "value": "Actor likes pear drink",
        "polarity": "positive",
        "stability": "stable", "origin": "explicit_setting",
        "comparison_key": axis_live.stable_trait_identity("preference", "", "pear drink"),
        "evidence": [{
            "document_name": "02-character-profiles.md", "line_start": 4,
            "line_end": 4, "text": "Actor likes pear drink",
        }],
    }
    assert axis_live._candidate_matches(candidate, selector)
    assert not axis_live._candidate_matches(
        {**candidate, "comparison_key": "preference:icedpeardrink"}, selector
    )
    assert not axis_live._candidate_matches(
        {**candidate, "character_key": "Other"}, selector
    )


def test_candidate_selector_rejects_other_clause_on_same_profile_line():
    selector = {
        "character_key": "桑衍", "trait_type": "core_personality",
        "source_document": "02-character-profiles.md", "source_line": 4,
        "source_quote": "先向共同值守的搭档说明会影响其航船的风险",
        "polarity": "positive", "stability": "core", "key_object": None,
    }
    profile_line = (
        "桑衍的核心性格是先向共同值守的搭档说明会影响其航船的风险；"
        "桑衍做决定很快。"
    )
    candidate = {
        "reviewable": True, "character_key": "桑衍",
        "trait_type": "core_personality", "trait_key": "risk_notice",
        "comparison_key": axis_live.stable_trait_identity("core_personality", "risk_notice"),
        "value": "桑衍做决定很快", "polarity": "positive",
        "stability": "core", "origin": "explicit_setting",
        "evidence": [{
            "document_name": "02-character-profiles.md", "line_start": 4,
            "line_end": 4, "text": profile_line,
        }],
    }
    assert not axis_live._candidate_matches(candidate, selector)
    assert axis_live._candidate_matches({
        **candidate,
        "value": "桑衍先向共同值守的搭档说明会影响其航船的风险。",
    }, selector)
    assert axis_live._candidate_matches({
        **candidate,
        "value": "桑衍先向共同值守的搭档说明，会影响其航船的风险。",
    }, selector)
    assert not axis_live._candidate_matches({
        **candidate, "value": "桑衍向搭档说明航船风险",
    }, selector)
    assert not axis_live._candidate_matches({
        **candidate, "value": "桑衍先向共同值守的搭档说明会影响其航船的风险",
        "comparison_key": "core_personality:another_trait",
    }, selector)
    assert not axis_live._candidate_matches({
        **candidate, "value": None,
    }, selector)


def test_ambiguous_source_anchor_aborts_before_axis_creation(tmp_path, monkeypatch):
    root, digest = _fixture(tmp_path)
    suite = axis_live.verify_fixture(root, expected_manifest_sha256=digest).suites["dev"]
    row = {
        "reviewable": True, "character_key": "Actor", "trait_type": "core_personality",
        "trait_key": "decision_axis", "value": "source anchor",
        "comparison_key": axis_live.stable_trait_identity("core_personality", "decision_axis"),
        "polarity": "positive", "stability": "core", "origin": "explicit_setting",
        "evidence": [{
            "document_name": "02-character-profiles.md", "line_start": 1,
            "line_end": 1, "text": "source anchor",
        }],
    }
    monkeypatch.setattr(axis_live, "_list_pending", lambda *_: [
        {**row, "id": "candidate-a"}, {**row, "id": "candidate-b"},
    ])
    monkeypatch.setattr(axis_live, "_request", lambda *_a, **_k: pytest.fail("unexpected write"))
    state = {}
    with pytest.raises(axis_live.SafeFailure, match="candidate_not_unique"):
        axis_live._review_candidates(object(), "project-id", suite, state)
    assert state["candidate_review"]["selectors"]["wanted"]["match_count"] == 2


def test_explained_case_needs_exact_g_citation_ref():
    candidate_id = "11111111-1111-4111-8111-111111111111"
    suite = axis_live.VerifiedSuite(
        name="dev", world_id="world-dev", case_count=1, files={}, hashes={},
        plan={"candidate_decisions": [{"candidate_key": "wanted", "trait_type": "preference", "approved_axis_key": None}], "approved_axes": []},
    )
    case = {
        "case_id": "growth", "candidate_key": "wanted", "gold_class": "explained",
        "allowed_final_outcomes": ["no_issue"], "required_citation_roles": ["B", "C", "G"],
        "min_independent_observations": 1,
        "evidence": {
            "B": [{"document_name": "profile.md", "line": 1}],
            "C": [{"document_name": "draft.md", "line": 2}],
            "G": [{"document_name": "history.md", "line": 3}], "X": [], "forbidden": [],
        },
    }
    trace = {
        "confirmed_candidate_id_sha256": axis_live._candidate_id_sha256(candidate_id),
        "character_key": "Actor", "final_outcome": "no_issue", "review_verdict": "explained",
        "citation_roles": ["B", "C", "G"],
        "matched_observation_refs": [{"document_name": "draft.md", "line_start": 2, "line_end": 2}],
        "citation_refs": None,
    }
    state = {
        "selected": {"wanted": {
            "id": candidate_id, "approved_axis_id": None,
            "evidence": [{"document_name": "profile.md", "line_start": 1, "line_end": 1}],
        }},
        "draft": {"case_trace": [trace], "visible_issue_cases": []},
    }
    unverified = axis_live._score_case(case, state, suite)
    assert unverified["evidence"]["G"] == "unavailable"
    assert unverified["passed"] is False
    trace["citation_refs"] = [
        {"role": "B", "document_name": "profile.md", "line_start": 1, "line_end": 1},
        {"role": "C", "document_name": "draft.md", "line_start": 2, "line_end": 2},
        {"role": "G", "document_name": "history.md", "line_start": 3, "line_end": 3},
    ]
    verified = axis_live._score_case(case, state, suite)
    assert verified["evidence"]["G"] == "matched"
    assert verified["passed"] is True
    state["draft"]["visible_issue_cases"] = [{
        "confirmed_candidate_id_sha256": axis_live._candidate_id_sha256(candidate_id),
        "judgement": "needs_confirmation",
        "evidence_refs": [
            {"document_name": "profile.md", "line_start": 1, "line_end": 1},
            {"document_name": "draft.md", "line_start": 2, "line_end": 2},
        ],
    }]
    assert axis_live._score_case(case, state, suite)["passed"] is False


def test_every_required_g_and_c_line_must_be_cited():
    case = {
        "required_citation_roles": ["B", "C", "G"],
        "evidence": {
            "B": [{"document_name": "profile.md", "line": 1}],
            "C": [
                {"document_name": "draft.md", "line": 2},
                {"document_name": "draft.md", "line": 4},
            ],
            "G": [
                {"document_name": "history.md", "line": 3},
                {"document_name": "history.md", "line": 5},
            ],
            "X": [],
        },
    }
    candidate = {"evidence": [
        {"document_name": "profile.md", "line_start": 1, "line_end": 1},
    ]}
    trace = {
        "matched_observation_refs": [
            {"document_name": "draft.md", "line_start": 2, "line_end": 2},
            {"document_name": "draft.md", "line_start": 4, "line_end": 4},
        ],
        "citation_refs": [
            {"role": "B", "document_name": "profile.md", "line_start": 1, "line_end": 1},
            {"role": "C", "document_name": "draft.md", "line_start": 2, "line_end": 2},
            {"role": "G", "document_name": "history.md", "line_start": 3, "line_end": 3},
        ],
    }
    scores = axis_live._evidence_scores(case, candidate, trace)
    assert scores == {"B": "matched", "C": "missed", "G": "missed", "X": "not_required"}
    trace["citation_refs"].extend([
        {"role": "C", "document_name": "draft.md", "line_start": 4, "line_end": 4},
        {"role": "G", "document_name": "history.md", "line_start": 5, "line_end": 5},
    ])
    assert axis_live._evidence_scores(case, candidate, trace) == {
        "B": "matched", "C": "matched", "G": "matched", "X": "not_required",
    }
    trace["citation_refs"] = None
    assert axis_live._evidence_scores(case, candidate, trace)["C"] == "unavailable"


def test_exact_lines_and_final_visible_issue_evidence_are_required():
    case = {
        "gold_class": "conflict",
        "evidence": {
            "B": [{"document_name": "profile.md", "line": 1}],
            "C": [
                {"document_name": "draft.md", "line": 3},
                {"document_name": "draft.md", "line": 4},
            ],
            "G": [], "X": [],
        },
    }
    assert axis_live._ref_matches(
        {"document_name": "draft.md", "line_start": 3, "line_end": 4},
        case["evidence"]["C"][0],
    ) is False
    span_scores = axis_live._evidence_scores(
        {**case, "required_citation_roles": []},
        {"evidence": [{"document_name": "profile.md", "line_start": 1, "line_end": 1}]},
        {"matched_observation_refs": [{
            "document_name": "draft.md", "line_start": 3, "line_end": 4,
        }], "citation_refs": None},
    )
    assert span_scores["C"] == "missed"
    trace = {"citation_refs": [
        {"role": "B", "document_name": "profile.md", "line_start": 1, "line_end": 1},
        {"role": "C", "document_name": "draft.md", "line_start": 3, "line_end": 3},
        {"role": "C", "document_name": "draft.md", "line_start": 4, "line_end": 4},
    ]}
    issue = {"evidence_refs": [
        {"document_name": "profile.md", "line_start": 1, "line_end": 1},
        {"document_name": "draft.md", "line_start": 3, "line_end": 3},
    ]}
    assert axis_live._visible_issue_evidence_score(case, [issue], trace) == "missed"
    issue["evidence_refs"][1]["line_end"] = 4
    assert axis_live._visible_issue_evidence_score(case, [issue], trace) == "missed"
    issue["evidence_refs"][1]["line_end"] = 3
    issue["evidence_refs"].append({
        "document_name": "draft.md", "line_start": 4, "line_end": 4,
    })
    assert axis_live._visible_issue_evidence_score(case, [issue], trace) == "matched"
    issue["evidence_refs"].append({
        "document_name": "draft.md", "line_start": 5, "line_end": 5,
    })
    assert axis_live._visible_issue_evidence_score(case, [issue], trace) == "missed"
    issue["evidence_refs"].pop()
    case["evidence"]["G"] = [{"document_name": "history.md", "line": 7}]
    issue["evidence_refs"].append({
        "document_name": "history.md", "line_start": 7, "line_end": 7,
    })
    assert axis_live._visible_issue_evidence_score(case, [issue], trace) == "matched"
    trace["citation_refs"].append({
        "role": "G", "document_name": "history.md", "line_start": 7, "line_end": 7,
    })
    assert axis_live._visible_issue_evidence_score(case, [issue], trace) == "matched"
    issue["evidence_refs"].pop()
    assert axis_live._visible_issue_evidence_score(case, [issue], trace) == "missed"


def test_case_cannot_pass_on_internal_trace_when_visible_issue_cites_wrong_line():
    candidate_id = "22222222-2222-4222-8222-222222222222"
    digest = axis_live._candidate_id_sha256(candidate_id)
    suite = axis_live.VerifiedSuite(
        name="dev", world_id="world-dev", case_count=1, files={}, hashes={},
        plan={"candidate_decisions": [{
            "candidate_key": "wanted", "trait_type": "behavior_boundary",
            "approved_axis_key": None,
        }], "approved_axes": []},
    )
    case = {
        "case_id": "visible-evidence", "candidate_key": "wanted", "gold_class": "conflict",
        "allowed_final_outcomes": ["conflict"], "required_citation_roles": ["B", "C"],
        "min_independent_observations": 1,
        "evidence": {
            "B": [{"document_name": "profile.md", "line": 1}],
            "C": [{"document_name": "draft.md", "line": 2}],
            "G": [], "X": [], "forbidden": [],
        },
    }
    trace = {
        "confirmed_candidate_id_sha256": digest,
        "character_key": "Actor", "final_outcome": "conflict",
        "review_verdict": "contradicts", "citation_roles": ["B", "C"],
        "matched_observation_refs": [{
            "document_name": "draft.md", "line_start": 2, "line_end": 2,
        }],
        "citation_refs": [
            {"role": "B", "document_name": "profile.md", "line_start": 1, "line_end": 1},
            {"role": "C", "document_name": "draft.md", "line_start": 2, "line_end": 2},
        ],
    }
    issue = {
        "confirmed_candidate_id_sha256": digest, "judgement": "contradicts",
        "evidence_refs": [
            {"document_name": "profile.md", "line_start": 1, "line_end": 1},
            {"document_name": "draft.md", "line_start": 3, "line_end": 3},
        ],
    }
    state = {
        "selected": {"wanted": {
            "id": candidate_id, "approved_axis_id": None,
            "evidence": [{"document_name": "profile.md", "line_start": 1, "line_end": 1}],
        }},
        "draft": {"case_trace": [trace], "visible_issue_cases": [issue]},
    }
    wrong = axis_live._score_case(case, state, suite)
    assert wrong["evidence"] == {
        "B": "matched", "C": "matched", "G": "not_required", "X": "not_required",
    }
    assert wrong["visible_issue_evidence"] == "missed"
    assert wrong["passed"] is False
    issue["evidence_refs"][1]["line_start"] = 2
    issue["evidence_refs"][1]["line_end"] = 2
    right = axis_live._score_case(case, state, suite)
    assert right["visible_issue_evidence"] == "matched"
    assert right["passed"] is True
    issue["evidence_refs"].append({
        "document_name": "draft.md", "line_start": 4, "line_end": 4,
    })
    extra_wrong_line = axis_live._score_case(case, state, suite)
    assert extra_wrong_line["visible_issue_evidence"] == "missed"
    assert extra_wrong_line["passed"] is False


def test_visible_false_positive_is_counted_even_when_case_trace_is_missing():
    candidate_id = "33333333-3333-4333-8333-333333333333"
    digest = axis_live._candidate_id_sha256(candidate_id)
    suite = axis_live.VerifiedSuite(
        name="dev", world_id="world-dev", case_count=1, files={}, hashes={},
        plan={"candidate_decisions": [{
            "candidate_key": "wanted", "trait_type": "behavior_boundary",
            "approved_axis_key": None,
        }], "approved_axes": []},
    )
    case = {
        "case_id": "hard-negative", "candidate_key": "wanted", "gold_class": "hard_negative",
        "allowed_final_outcomes": ["no_issue"], "required_citation_roles": [],
        "min_independent_observations": 0,
        "evidence": {
            "B": [{"document_name": "profile.md", "line": 1}],
            "C": [], "G": [], "X": [], "forbidden": [],
        },
    }
    draft = {"case_trace": [], "visible_issue_cases": [{
        "confirmed_candidate_id_sha256": digest, "judgement": "contradicts",
        "evidence_refs": [],
    }]}
    state = {
        "trial": 1,
        "selected": {"wanted": {"id": candidate_id, "approved_axis_id": None}},
        "draft": draft,
    }
    score = axis_live._score_case(case, state, suite)
    assert score["trace_unique"] is False
    assert score["false_positive"] is True
    assert score["visible_false_positive_count"] == 1
    assert score["false_positive_unknown"] is False
    trial = axis_live._score_trial(suite, state, [case])
    assert trial["counts"]["false_positives"] == 1
    assert trial["counts"]["visible_false_positives"] == 1
    assert trial["counts"]["unexpected_conflicts"] == 0

    draft["visible_issue_cases"] = []
    missing = axis_live._score_case(case, state, suite)
    assert missing["false_positive"] is False
    assert missing["false_positive_unknown"] is True
    trial = axis_live._score_trial(suite, state, [case])
    assert trial["counts"]["false_positive_unknown_cases"] == 1


def test_custom_fixture_cannot_claim_strict_pass(tmp_path, monkeypatch):
    root, digest = _fixture(tmp_path)
    monkeypatch.setattr(axis_live, "_code_state", lambda: {
        "git_head": "a" * 40, "worktree_clean": True,
    })
    monkeypatch.setattr(axis_live.httpx, "Client", lambda **_k: pytest.fail("unexpected HTTP"))
    args = argparse.Namespace(
        dataset=str(root), manifest_sha256=digest, output_json=None,
        preflight_only=False, base_url="http://127.0.0.1:8000",
        run_timeout_seconds=1, diagnostic_dev_one_trial=False,
    )
    report, code = axis_live.run(args)
    assert code == 1
    assert report["dataset_kind"] == "custom"
    assert report["failure"]["code"] == "strict_requires_pinned_fixture"


def test_partial_draft_never_passes_even_if_case_score_passes():
    suite = axis_live.VerifiedSuite(
        name="dev", world_id="world-dev", case_count=0, files={}, hashes={},
        plan={"candidate_decisions": [], "approved_axes": []},
    )
    state = {
        "trial": 1, "baseline_admission": {"admitted": True},
        "candidate_review": {"expected": 0, "unique_matches": 0},
        "selected": {},
        "draft": {
            "status": "completed", "stage_outcome": "partial", "material_coverage": "partial",
            "planned_chunks": 1, "processed_chunks": 1,
            "stage_usage": {"attempted_calls": 1}, "visible_issue_cases": [],
        },
    }
    result = axis_live._score_trial(suite, state, [])
    assert result["draft_complete"] is False
    assert result["passed"] is False


@pytest.mark.parametrize(
    ("unselected", "reported_count"),
    [(0, 0), (1, 1), (None, None), ("0", None), (False, None), (-1, None)],
)
def test_strict_trial_fails_closed_on_unselected_reviewable_candidates(
    unselected, reported_count,
):
    suite = axis_live.VerifiedSuite(
        name="dev", world_id="world-dev", case_count=0, files={}, hashes={},
        plan={"candidate_decisions": [], "approved_axes": []},
    )
    review = {"expected": 0, "unique_matches": 0, "reviewable": 0}
    if unselected is not None:
        review["unselected_reviewable"] = unselected
    state = {
        "trial": 1,
        "project_id_sha256": "a" * 64,
        "baseline_admission": {"admitted": True},
        "candidate_review": review,
        "selected": {},
        "draft": {
            "status": "completed", "stage_outcome": "completed",
            "material_coverage": "complete", "planned_chunks": 1,
            "processed_chunks": 1,
            "stage_usage": {"attempted_calls": 1},
            "visible_issue_cases": [],
        },
    }
    result = axis_live._score_trial(suite, state, [])
    assert result["counts"]["unselected_reviewable_candidates"] == reported_count
    assert result["passed"] is (unselected == 0 and type(unselected) is int)
    if unselected == 0 and type(unselected) is int:
        malformed = axis_live._score_trial(
            suite, {**state, "candidate_review": "invalid"}, []
        )
        assert malformed["counts"]["unselected_reviewable_candidates"] is None
        assert malformed["passed"] is False


def test_suite_report_exposes_extra_candidates_and_unknown_counts():
    suite = axis_live.VerifiedSuite(
        name="dev", world_id="world-dev", case_count=0, files={}, hashes={},
        plan={"candidate_decisions": [], "approved_axes": []},
    )
    states = []
    for trial, unselected in ((1, 0), (2, 2), (3, "invalid")):
        states.append({
            "trial": trial,
            "project_id_sha256": axis_live._sha256(f"project-{trial}"),
            "baseline_admission": {"admitted": True},
            "candidate_review": {
                "expected": 0, "unique_matches": 0,
                "reviewable": 0, "unselected_reviewable": unselected,
            },
            "selected": {},
            "draft": {
                "status": "completed", "stage_outcome": "completed",
                "material_coverage": "complete", "planned_chunks": 1,
                "processed_chunks": 1, "stage_usage": {"attempted_calls": 1},
                "visible_issue_cases": [],
            },
        })
    report = axis_live._suite_report(suite, states, [])
    assert report["passed"] is False
    assert report["aggregate"]["unselected_reviewable_candidates"] == 2
    assert report["aggregate"]["unselected_reviewable_unknown_trials"] == 1
    assert [row["passed"] for row in report["trials"]] == [True, False, False]


def test_safe_failure_drops_exception_text():
    failure = axis_live._failure(RuntimeError("sk-secret story text https://private.invalid"))
    assert failure == {
        "code": "runner_error", "stage": "runner",
        "details": {"exception_type": "RuntimeError"},
    }
    assert "sk-secret" not in repr(failure)


def test_execute_trial_uploads_only_story_documents_and_explicit_draft_target(
    tmp_path, monkeypatch,
):
    root, digest = _fixture(tmp_path)
    suite = axis_live.verify_fixture(root, expected_manifest_sha256=digest).suites["dev"]
    uploads = []
    runs = []

    def fake_request(_client, method, _path, route, **kwargs):
        if route == "project_create":
            return {"id": "project-uuid"}
        if route == "upload":
            uploads.append(kwargs["json"])
            return {"id": f"document-{len(uploads)}"}
        raise AssertionError(f"unexpected route {method} {route}")

    def fake_start(_client, _project, *, mode, target_id=None):
        runs.append((mode, target_id))
        return f"run-{len(runs)}"

    monkeypatch.setattr(axis_live, "_request", fake_request)
    monkeypatch.setattr(axis_live, "_start_run", fake_start)
    monkeypatch.setattr(axis_live, "_wait_run", lambda _c, run_id, **_k: {"id": run_id})
    monkeypatch.setattr(axis_live, "_run_summary", lambda _c, run, **_k: {"run_id": run["id"]})
    monkeypatch.setattr(axis_live, "_baseline_admission", lambda _summary: {"admitted": True})
    monkeypatch.setattr(axis_live, "_review_candidates", lambda *_: None)

    state = {}
    axis_live._execute_trial(object(), suite, 1, state, timeout_seconds=1)
    assert [row["name"] for row in uploads] == [
        "01-world-setting.md", "02-character-profiles.md",
        "03-published-history-v1.0.md", "04-draft-event-v1.1.md",
    ]
    assert runs == [("baseline_build", None), ("draft_review", "document-4")]
    assert uploads[-1]["narrative_context"]["publication_status"] == "draft"
    assert all("oracle" not in json.dumps(row) for row in uploads)


def test_six_trials_finish_before_oracle_is_parsed(tmp_path, monkeypatch):
    root, digest = _fixture(tmp_path)
    events = []
    provenance = _runtime_provenance(revision="a" * 64)
    runtime_digest = axis_live._runtime_provenance_digest(provenance)
    assert runtime_digest is not None
    monkeypatch.setattr(axis_live, "DATASET", root)
    monkeypatch.setattr(axis_live, "PINNED_MANIFEST_SHA256", digest)
    monkeypatch.setattr(axis_live, "_code_state", lambda: {
        "git_head": "a" * 64, "worktree_clean": True,
    })
    monkeypatch.setattr(axis_live, "_local_service_artifact_sha256", lambda _root: "c" * 64)

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(axis_live.httpx, "Client", FakeClient)
    monkeypatch.setattr(axis_live, "_request", lambda *_a, **_k: {
        "model": {"configured": True},
        "runtime_provenance": provenance,
    })

    def fake_trial(_client, suite, trial, state, **_kwargs):
        events.append((suite.name, trial))
        state["project_id_sha256"] = axis_live._sha256(f"{suite.name}-{trial}")
        state["baseline"] = {"runtime_provenance_sha256": runtime_digest}
        state["draft"] = {"runtime_provenance_sha256": runtime_digest}

    def fake_oracle(suite):
        assert sum(item[0] != "oracle" for item in events) == 6
        events.append(("oracle", suite.name))
        return []

    monkeypatch.setattr(axis_live, "_execute_trial", fake_trial)
    monkeypatch.setattr(axis_live, "_load_oracle", fake_oracle)
    monkeypatch.setattr(axis_live, "_suite_report", lambda _s, states, _c: {
        "passed": len(states) == 3, "attempted_trials": len(states),
    })
    args = argparse.Namespace(
        dataset=str(root), manifest_sha256=digest, output_json=None,
        preflight_only=False, base_url="http://127.0.0.1:8000",
        run_timeout_seconds=1, diagnostic_dev_one_trial=False,
    )
    report, code = axis_live.run(args)
    assert code == 0
    assert report["passed"] is True
    assert report["independent_projects"] is True
    assert events[:6] == [
        ("dev", 1), ("dev", 2), ("dev", 3),
        ("transfer", 1), ("transfer", 2), ("transfer", 3),
    ]
    events.clear()

    def reused_project(_client, suite, trial, state, **_kwargs):
        events.append((suite.name, trial))
        state["project_id_sha256"] = "0" * 64
        state["baseline"] = {"runtime_provenance_sha256": runtime_digest}
        state["draft"] = {"runtime_provenance_sha256": runtime_digest}

    monkeypatch.setattr(axis_live, "_execute_trial", reused_project)
    duplicate, code = axis_live.run(args)
    assert code == 1
    assert duplicate["independent_projects"] is False
    assert duplicate["passed"] is False


@pytest.mark.parametrize(
    ("build_field", "reported_value", "failure_code"),
    [
        ("git_revision", None, "runtime_provenance_invalid"),
        ("git_revision", "b" * 40, "service_build_revision_mismatch"),
        ("service_artifact_sha256", None, "runtime_provenance_invalid"),
        ("service_artifact_sha256", "d" * 64, "service_artifact_mismatch"),
    ],
)
def test_strict_service_build_mismatch_stops_at_health_before_project_create(
    tmp_path, monkeypatch, build_field, reported_value, failure_code,
):
    root, manifest_digest = _fixture(tmp_path)
    provenance = _runtime_provenance()
    provenance["build"][build_field] = reported_value
    monkeypatch.setattr(axis_live, "DATASET", root)
    monkeypatch.setattr(axis_live, "PINNED_MANIFEST_SHA256", manifest_digest)
    monkeypatch.setattr(axis_live, "_code_state", lambda: {
        "git_head": "a" * 40, "worktree_clean": True,
    })
    monkeypatch.setattr(axis_live, "_local_service_artifact_sha256", lambda _root: "c" * 64)
    calls = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_request(_client, _method, _path, route, **_kwargs):
        calls.append(route)
        assert route == "health", "no project or other API call before build gate"
        return {
            "model": {"configured": True},
            "runtime_provenance": provenance,
        }

    monkeypatch.setattr(axis_live.httpx, "Client", FakeClient)
    monkeypatch.setattr(axis_live, "_request", fake_request)
    report, code = axis_live.run(argparse.Namespace(
        dataset=str(root), manifest_sha256=manifest_digest, output_json=None,
        preflight_only=False, base_url="http://127.0.0.1:8000",
        run_timeout_seconds=1, diagnostic_dev_one_trial=False,
    ))
    assert code == 1
    assert report["failure"]["code"] == failure_code
    assert report["claims"]["independent_full_workflow_trials"] is False
    assert calls == ["health"]


def test_character_runtime_provenance_requires_new_effective_limits_and_hashes_changes():
    base = _runtime_provenance()
    digest = axis_live._runtime_provenance_digest(base)
    assert digest is not None
    new_fields = (
        "signal_max_records", "signal_targeted_max_targets_per_chunk",
        "signal_max_response_bytes", "drift_max_evidence_chars",
        "drift_token_budget", "drift_max_completion_tokens",
        "drift_max_response_bytes", "signal_provider_max_completion_tokens",
        "signal_provider_max_response_bytes", "drift_provider_max_completion_tokens",
        "drift_provider_max_response_bytes", "per_run_token_budget",
        "daily_token_budget", "signal_timeout_seconds", "drift_timeout_seconds",
        "signal_provider_timeout_seconds", "drift_provider_timeout_seconds",
    )
    for field in new_fields:
        missing = json.loads(json.dumps(base))
        del missing["character_consistency_limits"][field]
        assert axis_live._runtime_provenance_digest(missing) is None, field
    for field, changed_value in (
        ("signal_max_records", 63),
        ("signal_targeted_max_targets_per_chunk", 7),
        ("drift_max_evidence_chars", 7999),
        ("signal_provider_max_completion_tokens", 4095),
        ("signal_provider_timeout_seconds", 29.0),
        ("drift_provider_max_response_bytes", 31999),
        ("per_run_token_budget", 100001),
        ("daily_token_budget", 1000001),
    ):
        changed = json.loads(json.dumps(base))
        changed["character_consistency_limits"][field] = changed_value
        changed_digest = axis_live._runtime_provenance_digest(changed)
        assert changed_digest is not None and changed_digest != digest, field
    over_stage_cap = json.loads(json.dumps(base))
    over_stage_cap["character_consistency_limits"]["signal_provider_max_completion_tokens"] = 4097
    assert axis_live._runtime_provenance_digest(over_stage_cap) is None
    no_rag = json.loads(json.dumps(base))
    no_rag["capabilities"]["embeddings"] = False
    no_rag["rag"]["profile_fingerprint"] = None
    assert axis_live._runtime_provenance_digest(no_rag) is not None


def _strict_provenance_trial(tmp_path, monkeypatch, *, worker_change=None,
                             post_revision=None, post_artifact=None):
    root, manifest_digest = _fixture(tmp_path)
    provenance = _runtime_provenance()
    runtime_digest = axis_live._runtime_provenance_digest(provenance)
    assert runtime_digest is not None
    monkeypatch.setattr(axis_live, "DATASET", root)
    monkeypatch.setattr(axis_live, "PINNED_MANIFEST_SHA256", manifest_digest)
    code_states = iter([
        {"git_head": "a" * 40, "worktree_clean": True},
        {"git_head": post_revision or "a" * 40, "worktree_clean": True},
    ])
    monkeypatch.setattr(axis_live, "_code_state", lambda: next(code_states))
    artifact_hashes = iter(["c" * 64, post_artifact or "c" * 64])
    monkeypatch.setattr(
        axis_live, "_local_service_artifact_sha256", lambda _root: next(artifact_hashes)
    )
    calls = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_request(_client, _method, _path, route, **_kwargs):
        calls.append(route)
        assert route in {"health", "health_postrun"}
        return {
            "model": {"configured": True},
            "runtime_provenance": provenance,
        }

    def fake_trial(_client, suite, trial, state, **_kwargs):
        state["project_id_sha256"] = axis_live._sha256(f"{suite.name}-{trial}")
        for stage in ("baseline", "draft"):
            value = runtime_digest
            if worker_change is not None:
                value = worker_change(suite.name, trial, stage, value)
            state[stage] = {"runtime_provenance_sha256": value}

    monkeypatch.setattr(axis_live.httpx, "Client", FakeClient)
    monkeypatch.setattr(axis_live, "_request", fake_request)
    monkeypatch.setattr(axis_live, "_execute_trial", fake_trial)
    monkeypatch.setattr(axis_live, "_load_oracle", lambda _suite: [])
    # Isolate the runtime gate from quality scoring; all six project IDs differ.
    monkeypatch.setattr(axis_live, "_suite_report", lambda _suite, states, _cases: {
        "passed": len(states) == 3, "attempted_trials": len(states),
    })
    report, code = axis_live.run(argparse.Namespace(
        dataset=str(root), manifest_sha256=manifest_digest, output_json=None,
        preflight_only=False, base_url="http://127.0.0.1:8000",
        run_timeout_seconds=1, diagnostic_dev_one_trial=False,
    ))
    return report, code, calls


def test_strict_worker_provenance_requires_all_twelve_matching_runs(tmp_path, monkeypatch):
    valid, code, calls = _strict_provenance_trial(tmp_path, monkeypatch)
    assert code == 0
    assert valid["passed"] is True
    assert valid["provenance_gate"]["worker_runtime_provenance_observed_runs"] == 12
    assert valid["provenance_gate"]["worker_runtime_provenance_matches_api_runs"] == 12
    assert calls == ["health", "health_postrun"]


@pytest.mark.parametrize(
    ("bad_worker_digest", "expected_reason"),
    [
        (None, "worker_runtime_provenance_incomplete"),
        ("f" * 64, "api_worker_runtime_provenance_mismatch"),
    ],
)
def test_strict_missing_or_mixed_worker_provenance_cannot_pass(
    tmp_path, monkeypatch, bad_worker_digest, expected_reason,
):
    def change(suite_name, trial, stage, value):
        if (suite_name, trial, stage) == ("transfer", 3, "draft"):
            return bad_worker_digest
        return value

    report, code, calls = _strict_provenance_trial(
        tmp_path, monkeypatch, worker_change=change,
    )
    assert code == 1
    assert report["passed"] is False
    assert report["independent_projects"] is True
    assert report["claims"]["independent_full_workflow_trials"] is False
    assert report["provenance_gate"]["passed"] is False
    assert expected_reason in report["provenance_gate"]["reason_codes"]
    assert report["provenance_gate"]["worker_runtime_provenance_matches_api_runs"] == 11
    assert calls == ["health", "health_postrun"]


@pytest.mark.parametrize(
    ("post_revision", "post_artifact"),
    [("b" * 40, None), (None, "d" * 64)],
)
def test_strict_local_code_change_during_runs_cannot_pass(
    tmp_path, monkeypatch, post_revision, post_artifact,
):
    report, code, calls = _strict_provenance_trial(
        tmp_path, monkeypatch,
        post_revision=post_revision, post_artifact=post_artifact,
    )
    assert code == 1
    assert report["passed"] is False
    assert report["provenance_gate"]["worker_runtime_provenance_matches_api_runs"] == 12
    assert report["provenance_gate"]["postrun_code_state_stable"] is False
    assert "runner_code_changed_during_run" in report["provenance_gate"]["reason_codes"]
    assert calls == ["health", "health_postrun"]


def test_strict_dirty_blocks_http_but_dev_diagnostic_runs_one_world(
    tmp_path, monkeypatch,
):
    root, digest = _fixture(tmp_path)
    monkeypatch.setattr(axis_live, "DATASET", root)
    monkeypatch.setattr(axis_live, "PINNED_MANIFEST_SHA256", digest)
    monkeypatch.setattr(axis_live, "_code_state", lambda: {
        "git_head": "a" * 64, "worktree_clean": False,
    })
    calls = []

    class FakeClient:
        def __init__(self, **_kwargs):
            calls.append("client")

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(axis_live.httpx, "Client", FakeClient)
    args = argparse.Namespace(
        dataset=str(root), manifest_sha256=digest, output_json=None,
        preflight_only=False, base_url="http://127.0.0.1:8000",
        run_timeout_seconds=1, diagnostic_dev_one_trial=False,
    )
    strict, code = axis_live.run(args)
    assert code == 1
    assert strict["failure"]["code"] == "strict_worktree_not_clean"
    assert calls == []

    monkeypatch.setattr(axis_live, "_request", lambda *_a, **_k: {
        "model": {"configured": True},
        "runtime_provenance": {"capabilities": {"character_consistency": True}},
    })

    def fake_trial(_client, suite, trial, state, **_kwargs):
        calls.append((suite.name, trial))

    monkeypatch.setattr(axis_live, "_execute_trial", fake_trial)
    monkeypatch.setattr(axis_live, "_load_oracle", lambda suite: (
        [] if suite.name == "dev" else pytest.fail("transfer oracle was parsed")
    ))
    diagnostic_args = argparse.Namespace(**{
        **vars(args), "diagnostic_dev_one_trial": True,
    })
    diagnostic, code = axis_live.run(diagnostic_args)
    assert code == 1
    assert diagnostic["mode"] == "developer_diagnostic"
    assert diagnostic["passed"] is False
    assert diagnostic["claims"]["independent_full_workflow_trials"] is False
    assert calls == ["client", ("dev", 1)]
    assert set(diagnostic["suites"]) == {"dev"}
    assert diagnostic["preflight_verified_suites"] == ["dev", "transfer"]


@pytest.mark.parametrize(
    "failure_code", ["baseline_admission_failed", "candidate_review_failed"]
)
def test_dev_failure_before_draft_is_unevaluated_not_false_negative(
    monkeypatch, failure_code,
):
    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_request(_client, _method, _path, route, **_kwargs):
        assert route == "health"
        return {
            "model": {"configured": True},
            "runtime_provenance": {"capabilities": {"character_consistency": True}},
        }

    def fake_trial(_client, _suite, _trial, state, **_kwargs):
        state["baseline"] = {
            "status": "completed", "stage_outcome": "partial",
            "material_coverage": "partial",
        }
        state["baseline_admission"] = {
            "admitted": failure_code != "baseline_admission_failed"
        }
        raise axis_live.SafeFailure(failure_code, "baseline")

    monkeypatch.setattr(axis_live.httpx, "Client", FakeClient)
    monkeypatch.setattr(axis_live, "_request", fake_request)
    monkeypatch.setattr(axis_live, "_execute_trial", fake_trial)
    monkeypatch.setattr(axis_live, "_code_state", lambda: {
        "git_head": "a" * 40, "worktree_clean": False,
    })
    report, code = axis_live.run(argparse.Namespace(
        dataset=str(axis_live.DATASET),
        manifest_sha256=axis_live.PINNED_MANIFEST_SHA256,
        output_json=None, preflight_only=False,
        base_url="http://127.0.0.1:8000", run_timeout_seconds=1,
        diagnostic_dev_one_trial=True,
    ))
    assert code == 1
    assert report["passed"] is False
    trial = report["suites"]["dev"]["trials"][0]
    assert trial["failure"]["code"] == failure_code
    assert trial["draft"] is None
    assert trial["counts"]["false_negatives"] == 0
    assert trial["counts"]["unevaluated_cases"] == 5
    assert report["suites"]["dev"]["aggregate"] == {
        "false_negatives": 0, "unevaluated_cases": 5,
    }
    assert all(
        row["evaluation_state"] == "pipeline_not_reached"
        and row["false_negative"] is None
        and row["passed"] is False
        for row in trial["case_scores"]
    )


def test_checked_in_fixture_matches_pinned_manifest_and_gold_contract():
    assert axis_live.PINNED_MANIFEST_SHA256 == (
        "650dad19bf54fad726b5c0f2f5dfb94d05fc4462babbbb558b0e9aaf40f61140"
    )
    fixture = axis_live.verify_fixture()
    assert fixture.manifest_sha256 == axis_live.PINNED_MANIFEST_SHA256
    assert {name: len(axis_live._load_oracle(suite)) for name, suite in fixture.suites.items()} == {
        "dev": 5, "transfer": 5,
    }


def test_v2_checked_in_fixture_matches_its_own_pin_and_gold_contract():
    assert axis_live.PINNED_MANIFEST_SHA256_V2 == (
        "3d428f156d2b5fd752a46eddb8a23ebeefff83cb221c2d02eaa8ac87891755b3"
    )
    assert axis_live.PINNED_MANIFEST_SHA256_V2 != axis_live.PINNED_MANIFEST_SHA256
    fixture = axis_live.verify_fixture(
        axis_live.DATASET_V2,
        expected_manifest_sha256=axis_live.PINNED_MANIFEST_SHA256_V2,
    )
    assert fixture.manifest_sha256 == axis_live.PINNED_MANIFEST_SHA256_V2
    assert {name: len(axis_live._load_oracle(suite)) for name, suite in fixture.suites.items()} == {
        "dev": 5, "transfer": 6,
    }
    assert fixture.suites["dev"].world_id != fixture.suites["transfer"].world_id


def test_both_official_fixtures_use_their_own_pin_without_http(monkeypatch):
    monkeypatch.setattr(axis_live.httpx, "Client", lambda **_k: pytest.fail("unexpected HTTP"))
    for dataset, digest, kind in (
        (axis_live.DATASET, axis_live.PINNED_MANIFEST_SHA256, "pinned_challenge"),
        (axis_live.DATASET_V2, axis_live.PINNED_MANIFEST_SHA256_V2, "pinned_challenge_v2"),
    ):
        report, code = axis_live.run(argparse.Namespace(
            dataset=str(dataset), manifest_sha256=None,
            output_json=None, preflight_only=True,
            diagnostic_dev_one_trial=False,
            base_url="http://127.0.0.1:8000", run_timeout_seconds=1,
        ))
        assert code == 0
        assert report["dataset_kind"] == kind
        assert report["manifest_sha256"] == digest
        assert report["preflight_verified"] is True
        assert report["passed"] is False


def test_v2_official_digest_override_and_unpinned_custom_fail_before_http(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(axis_live.httpx, "Client", lambda **_k: pytest.fail("unexpected HTTP"))
    wrong, code = axis_live.run(argparse.Namespace(
        dataset=str(axis_live.DATASET_V2),
        manifest_sha256=axis_live.PINNED_MANIFEST_SHA256,
        output_json=None, preflight_only=True,
        diagnostic_dev_one_trial=False,
        base_url="http://127.0.0.1:8000", run_timeout_seconds=1,
    ))
    assert code == 1
    assert wrong["failure"]["code"] == "official_manifest_digest_override"
    custom, _digest = _fixture(tmp_path)
    unpinned, code = axis_live.run(argparse.Namespace(
        dataset=str(custom), manifest_sha256=None,
        output_json=None, preflight_only=True,
        diagnostic_dev_one_trial=False,
        base_url="http://127.0.0.1:8000", run_timeout_seconds=1,
    ))
    assert code == 1
    assert unpinned["dataset_kind"] == "custom"
    assert unpinned["failure"]["code"] == "manifest_digest_required"


def test_v2_is_strict_eligible_but_still_obeys_clean_worktree_gate(monkeypatch):
    monkeypatch.setattr(axis_live.httpx, "Client", lambda **_k: pytest.fail("unexpected HTTP"))
    monkeypatch.setattr(axis_live, "_code_state", lambda: {
        "git_head": "a" * 40, "worktree_clean": False,
    })
    report, code = axis_live.run(argparse.Namespace(
        dataset=str(axis_live.DATASET_V2), manifest_sha256=None,
        output_json=None, preflight_only=False,
        diagnostic_dev_one_trial=False,
        base_url="http://127.0.0.1:8000", run_timeout_seconds=1,
    ))
    assert code == 1
    assert report["dataset_kind"] == "pinned_challenge_v2"
    assert report["failure"]["code"] == "strict_worktree_not_clean"


def test_v2_file_hash_and_source_anchor_are_checked_before_scoring(tmp_path):
    copied = tmp_path / "v2-copy"
    shutil.copytree(axis_live.DATASET_V2, copied)
    draft = copied / "dev" / axis_live.DRAFT_FILE
    draft.write_bytes(draft.read_bytes() + b"\nchanged after freeze")
    with pytest.raises(axis_live.SafeFailure, match="fixture_file_hash_mismatch"):
        axis_live.verify_fixture(
            copied, expected_manifest_sha256=axis_live.PINNED_MANIFEST_SHA256_V2
        )

    anchor_copy = tmp_path / "v2-anchor-copy"
    shutil.copytree(axis_live.DATASET_V2, anchor_copy)
    plan_path = anchor_copy / "dev" / "review-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["candidate_decisions"][0]["source_quote"] = "this quote is absent from the source"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    manifest_path = anchor_copy / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["suites"]["dev"]["files"]["review-plan.json"] = _hash(plan_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(axis_live.SafeFailure, match="review_plan_source_anchor"):
        axis_live.verify_fixture(
            anchor_copy, expected_manifest_sha256=_hash(manifest_path.read_bytes())
        )


def test_preflight_only_is_verified_but_not_a_quality_pass(monkeypatch):
    monkeypatch.setattr(axis_live.httpx, "Client", lambda **_k: pytest.fail("unexpected HTTP"))
    report, code = axis_live.run(argparse.Namespace(
        dataset=str(axis_live.DATASET),
        manifest_sha256=axis_live.PINNED_MANIFEST_SHA256,
        output_json=None, preflight_only=True,
        diagnostic_dev_one_trial=False,
        base_url="http://127.0.0.1:8000", run_timeout_seconds=1,
    ))
    assert code == 0
    assert report["mode"] == "preflight_only"
    assert report["preflight_verified"] is True
    assert report["passed"] is False
    assert report["claims"]["independent_full_workflow_trials"] is False


def test_citation_refs_are_projected_only_when_complete_and_content_free(monkeypatch):
    raw_trace = {
        "character_key": "Actor", "dimension": "core_personality",
        "comparison_key": "approved_axis:11111111-1111-4111-8111-111111111111:1",
        "review_outcome": "completed", "review_verdict": "explained",
        "citation_roles": ["B", "C", "G"],
        "citation_refs_incomplete": False,
        "citation_refs": [{
            "handle": "G01", "role": "G", "document_name": "history.md",
            "line_start": 3, "line_end": 3, "text": "sk-secret never report",
        }],
    }

    def fake_request(_client, _method, _path, route, **_kwargs):
        if route == "diagnostics":
            return {"character_consistency": {
                "case_trace": [raw_trace],
                "accepted_draft_observation_refs": [{
                    "character_key": "Actor", "dimension": "behavior_boundary",
                    "polarity": "positive", "observation_kind": "action",
                    "document_name": "draft.md", "line_start": 2, "line_end": 2,
                    "text": "sk-secret never report",
                }],
                "accepted_draft_observation_total": 1,
                "accepted_draft_observation_refs_truncated": False,
                "accepted_signal_histogram": [{
                    "source_kind": "formal_character_profile",
                    "stability": "stable", "dimension": "behavior_boundary",
                    "count": 1,
                }, {
                    "source_kind": "published_history",
                    "stability": "temporary", "dimension": "contextual_behavior",
                    "count": 1,
                }],
                "candidate_eligibility": {
                    "stable_or_core_formal_signals": 1,
                    "stable_or_core_history_signals": 0,
                    "prelimit_candidates": 1,
                },
                "counts": {
                    "targeted_record_rejected_count": 3,
                    "signal_count": 2,
                },
                "reason_counts": {
                    "candidate_limit": 2,
                    "source_formal": 2,
                    "source_history": 1,
                    "regenerated_from_evidence_mismatch": 5,
                    "statement_support": 4,
                    "sk-secret": 1,
                },
            }}
        if route == "issues":
            return [{
                "category": "character_drift", "explanation": "sk-secret never report",
                "metadata": {"character_key": "Actor", "dimension": "core_personality"},
                "evidence": [{
                    "document_name": "draft.md", "line_start": 2, "line_end": 2,
                    "text": "sk-secret never report",
                }],
            }]
        raise AssertionError(route)

    monkeypatch.setattr(axis_live, "_request", fake_request)
    summary = axis_live._run_summary(
        object(), {"id": "run-id"}, known_documents={"history.md", "draft.md"}
    )
    assert summary["case_trace"][0]["citation_refs"] == [{
        "role": "G", "document_name": "history.md", "line_start": 3, "line_end": 3,
    }]
    assert summary["accepted_draft_observation_refs"] == [{
        "actor_sha256": axis_live._actor_digest("Actor"),
        "dimension": "behavior_boundary", "polarity": "positive",
        "document_name": "draft.md", "line_start": 2, "line_end": 2,
    }]
    assert summary["targeted_record_rejection_events"] == 3
    assert summary["accepted_signal_histogram"] == [{
        "source_kind": "formal_character_profile", "stability": "stable",
        "dimension": "behavior_boundary", "count": 1,
    }, {
        "source_kind": "published_history", "stability": "temporary",
        "dimension": "contextual_behavior", "count": 1,
    }]
    assert summary["candidate_eligibility"] == {
        "stable_or_core_formal_signals": 1,
        "stable_or_core_history_signals": 0,
        "prelimit_candidates": 1,
    }
    assert summary["reason_counts"] == {
        "candidate_limit": 2,
        "source_formal": 2,
        "source_history": 1,
        "regenerated_from_evidence_mismatch": 5,
        "statement_support": 4,
    }
    assert summary["unreported_reason_entries"] == 1
    assert "sk-secret" not in repr(summary)
    raw_trace["citation_refs_incomplete"] = True
    incomplete = axis_live._run_summary(
        object(), {"id": "run-id"}, known_documents={"history.md", "draft.md"}
    )
    assert incomplete["case_trace"][0]["citation_refs"] is None


def test_signal_histogram_projection_rejects_unbounded_or_untrusted_labels():
    valid = {
        "accepted_signal_histogram": [{
            "source_kind": "formal_character_profile", "stability": "stable",
            "dimension": "behavior_boundary", "count": 1,
        }],
        "candidate_eligibility": {
            "stable_or_core_formal_signals": 1,
            "stable_or_core_history_signals": 0,
            "prelimit_candidates": 1,
        },
    }
    assert axis_live._safe_accepted_signal_diagnostics(
        valid, {"signal_count": 1}
    ) == (valid["accepted_signal_histogram"], valid["candidate_eligibility"])
    for field, value in (
        ("source_kind", "sk-secret"),
        ("source_kind", ["formal_character_profile"]),
        ("stability", "secret"),
        ("dimension", "secret"),
        ("count", True),
    ):
        invalid = json.loads(json.dumps(valid))
        invalid["accepted_signal_histogram"][0][field] = value
        assert axis_live._safe_accepted_signal_diagnostics(
            invalid, {"signal_count": 1}
        ) == (None, None)
    unexpected = json.loads(json.dumps(valid))
    unexpected["accepted_signal_histogram"][0]["text"] = "sk-secret"
    assert axis_live._safe_accepted_signal_diagnostics(
        unexpected, {"signal_count": 1}
    ) == (None, None)
    assert axis_live._safe_accepted_signal_diagnostics(
        valid, {"signal_count": 2}
    ) == (None, None)
    assert axis_live._safe_accepted_signal_diagnostics(valid, {}) == (None, None)
    inflated = json.loads(json.dumps(valid))
    inflated["candidate_eligibility"]["prelimit_candidates"] = 2
    assert axis_live._safe_accepted_signal_diagnostics(
        inflated, {"signal_count": 1}
    ) == (None, None)


def test_each_run_summary_records_only_valid_worker_runtime_provenance_digest(monkeypatch):
    provenance = _runtime_provenance()
    provenance["chat_provider"]["model_alias"] = "private-model-alias"
    diagnostics = {
        "runtime_provenance": provenance,
        "character_consistency": {},
    }

    def fake_request(_client, _method, _path, route, **_kwargs):
        if route == "diagnostics":
            return diagnostics
        if route == "issues":
            return []
        raise AssertionError(route)

    monkeypatch.setattr(axis_live, "_request", fake_request)
    summary = axis_live._run_summary(object(), {"id": "run-id"}, known_documents=set())
    assert summary["runtime_provenance_sha256"] == axis_live._runtime_provenance_digest(
        provenance
    )
    assert "private-model-alias" not in repr(summary)

    diagnostics.pop("runtime_provenance")
    missing = axis_live._run_summary(object(), {"id": "run-id"}, known_documents=set())
    assert missing["runtime_provenance_sha256"] is None

    diagnostics["runtime_provenance"] = {**provenance, "build": {
        **provenance["build"], "service_artifact_sha256": None,
    }}
    malformed = axis_live._run_summary(object(), {"id": "run-id"}, known_documents=set())
    assert malformed["runtime_provenance_sha256"] is None


def test_global_accepted_refs_catch_unmatched_wrong_actor_and_fail_closed():
    candidate_id = "11111111-1111-4111-8111-111111111111"
    suite = axis_live.VerifiedSuite(
        name="dev", world_id="world-dev", case_count=1, files={}, hashes={},
        plan={"candidate_decisions": [{"candidate_key": "wanted", "trait_type": "behavior_boundary", "approved_axis_key": None}],
              "approved_axes": []},
    )
    case = {
        "case_id": "wrong-actor", "candidate_key": "wanted", "gold_class": "hard_negative",
        "allowed_final_outcomes": ["no_issue"], "required_citation_roles": [],
        "min_independent_observations": 0,
        "evidence": {
            "B": [{"document_name": "profile.md", "line": 1}],
            "C": [], "G": [], "X": [],
            "forbidden": [{
                "document_name": "draft.md", "line": 5, "character_key": "Actor",
                "role": "C", "polarity": None,
            }],
        },
    }
    state = {
        "selected": {"wanted": {
            "id": candidate_id, "approved_axis_id": None,
            "evidence": [{"document_name": "profile.md", "line_start": 1, "line_end": 1}],
        }},
        "draft": {"case_trace": [{
            "confirmed_candidate_id_sha256": axis_live._candidate_id_sha256(candidate_id),
            "character_key": "Actor", "final_outcome": "no_issue",
            "matched_observation_refs": [], "citation_roles": [], "citation_refs": None,
        }], "visible_issue_cases": []},
    }
    score = axis_live._score_case(case, state, suite)
    assert score["target_dimension_attribution_unavailable"] is True
    assert score["passed"] is False

    state["draft"]["accepted_draft_observation_refs"] = [{
        "actor_sha256": axis_live._actor_digest("Actor"),
        "dimension": "core_personality", "polarity": "negative",
        "document_name": "draft.md", "line_start": 5, "line_end": 5,
    }]
    score = axis_live._score_case(case, state, suite)
    assert score["target_dimension_attribution_unavailable"] is False
    assert score["target_dimension_suspect_attribution_hits"] == 0
    assert score["actor_recall_assessable"] is False
    assert score["passed"] is True

    state["draft"]["accepted_draft_observation_refs"][0]["dimension"] = "behavior_boundary"
    score = axis_live._score_case(case, state, suite)
    assert score["target_dimension_suspect_attribution_hits"] == 1
    assert score["passed"] is False

    state["draft"]["accepted_draft_observation_refs"] = []
    score = axis_live._score_case(case, state, suite)
    assert score["target_dimension_suspect_attribution_hits"] == 0
    assert score["passed"] is True, score

    state["draft"]["case_trace"][0]["matched_observation_refs"] = [{
        "document_name": "draft.md", "line_start": 8, "line_end": 8,
    }]
    score = axis_live._score_case(case, state, suite)
    assert score["evidence"]["C"] == "missed"
    assert score["passed"] is False


def test_accepted_ref_projection_is_complete_or_unavailable_and_never_reports_actor():
    ref = {
        "character_key": "Actor", "dimension": "behavior_boundary",
        "polarity": "positive", "observation_kind": "action",
        "document_name": "draft.md", "line_start": 5, "line_end": 5,
        "text": "sk-secret never report",
    }
    stage = {
        "accepted_draft_observation_refs": [ref],
        "accepted_draft_observation_total": 1,
        "accepted_draft_observation_refs_truncated": False,
    }
    projected = axis_live._safe_accepted_draft_refs(stage, known_documents={"draft.md"})
    assert projected == [{
        "actor_sha256": axis_live._actor_digest("Actor"),
        "dimension": "behavior_boundary", "polarity": "positive",
        "document_name": "draft.md", "line_start": 5, "line_end": 5,
    }]
    assert "Actor" not in repr(projected)
    assert "sk-secret" not in repr(projected)
    stage["accepted_draft_observation_refs_truncated"] = True
    assert axis_live._safe_accepted_draft_refs(stage, known_documents={"draft.md"}) is None
    stage["accepted_draft_observation_refs_truncated"] = False
    stage["accepted_draft_observation_total"] = 2
    assert axis_live._safe_accepted_draft_refs(stage, known_documents={"draft.md"}) is None
