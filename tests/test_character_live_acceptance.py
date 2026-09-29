import hashlib
import json
from copy import deepcopy

import pytest

import scripts.run_character_consistency_live as live_acceptance
from app.config import Settings
from app.runtime_provenance import safe_runtime_provenance as service_runtime_provenance
from scripts.run_character_consistency_live import (
    AcceptanceFailure,
    _baseline_admission,
    _comparison_identity,
    _evaluation_claims,
    _evaluate_gates,
    _evaluate_oracle_trial,
    _evaluate_stability_observations,
    _execute_trial,
    _failure_report,
    _load_oracle,
    _matches_selector,
    _report_gates,
    _resolve_output_json,
    _review_explicit_candidates,
    _safe_unexpected_failure,
    _safe_case_trace_summary,
    _safe_token_admission_events,
    _safe_trace_observation_refs,
    _safe_visible_issue,
    _validate_oracle_payload,
)


def _alpha_runtime_provenance():
    return service_runtime_provenance(Settings(
        loreguard_build_revision="a" * 40,
        openai_api_key="test-only-secret",
        openai_model="fixture-model",
        provider_thinking_mode="disabled",
        enable_character_consistency=True,
        character_explanation_review_v1=True,
        character_scoped_axis_drift_v1=True,
        character_signal_support_id_v4=True,
        character_signal_semantic_scope_v5=True,
        character_signal_scope_review_v1=True,
        character_draft_actor_review_v1=True,
        character_target_bound_draft_review_v3=True,
    ))


def test_qi_disguise_fixture_contains_two_independent_behaviors():
    lines = (
        (live_acceptance.DEMO / live_acceptance.DRAFT_FILE)
        .read_text(encoding="utf-8")
        .splitlines()
    )

    setup, first_behavior, second_behavior, resolution = lines[16:20]
    assert "整个潜入期间持续生效" in setup
    assert all(
        marker in first_behavior
        for marker in ("午后一点", "商会门厅", "守门执事", "没有直接请求放行")
    )
    assert all(
        marker in second_behavior
        for marker in (
            "一小时后",
            "档案室门口",
            "档案管理员",
            "没有直接询问口令",
        )
    )
    assert "镜面身份仍在生效" in second_behavior
    assert "立即结束伪装" in resolution
    assert first_behavior != second_behavior


def test_token_admission_summary_whitelists_only_bounded_numeric_events():
    valid = {
        "stage_phase": "targeted_verification",
        "signal_phase": "initial",
        "chunk_ordinal": 2,
        "target_ordinal": 3,
        "estimated_tokens": 9_000,
        "available_tokens": 7_000,
        "stage_remaining_before": 11_000,
        "reviewer_reserve_tokens": 4_000,
        "model_calls_before_failure": 0,
    }
    injected = [
        {**valid, "prompt": "private source text"},
        {**valid, "api_key": "sk-test-secret-not-for-report"},
        {**valid, "estimated_tokens": "9000"},
        {**valid, "target_ordinal": True},
        {**valid, "stage_phase": ["targeted_verification"]},
        {**valid, "available_tokens": 10_000},
    ]

    safe = _safe_token_admission_events([valid, *injected])
    assert safe == [valid]
    assert "private source text" not in repr(safe)
    assert "sk-test-secret-not-for-report" not in repr(safe)
    assert _safe_token_admission_events(None) == []
    assert len(_safe_token_admission_events([valid] * 50)) == 24


def test_safe_case_trace_summary_keeps_only_bounded_source_provenance():
    row = {
        "character_key": "林澈",
        "dimension": "preference",
        "comparison_key": "preference:蜜瓜",
        "matched_observation_count": 15,
        "matched_observation_refs_truncated": True,
        "matched_observation_refs": [
            {
                "document_name": "draft.md",
                "line_start": index + 1,
                "line_end": index + 1,
                "observation_kind": "preference_expression",
                "polarity": "negative",
                "key_object_sha256": hashlib.sha256(
                    "蜜瓜".encode("utf-8")
                ).hexdigest(),
                "text": "机密原文不应进入报告",
            }
            for index in range(13)
        ] + [{
            "document_name": "sk-1234567890abcdef.md",
            "line_start": 14,
            "line_end": 14,
            "observation_kind": "action",
            "polarity": "negative",
            "key_object_sha256": "not-a-digest",
        }],
        "raw_model_response": "机密模型答复不应进入报告",
    }

    safe = _safe_case_trace_summary(row)
    assert len(safe["matched_observation_refs"]) == 12
    assert safe["matched_observation_refs_truncated"] is True
    assert safe["matched_observation_refs"][0] == {
        "document_name": "draft.md",
        "line_start": 1,
        "line_end": 1,
        "observation_kind": "preference_expression",
        "polarity": "negative",
        "key_object_sha256": hashlib.sha256("蜜瓜".encode("utf-8")).hexdigest(),
    }
    serialized = str(safe)
    assert "机密原文" not in serialized
    assert "机密模型答复" not in serialized
    assert "sk-1234567890abcdef" not in serialized


def test_safe_trace_observation_refs_rejects_untrusted_fields_and_types():
    valid = {
        "document_name": "draft.md",
        "line_start": 9,
        "line_end": 10,
        "observation_kind": "action",
        "polarity": "negative",
        "key_object_sha256": None,
    }
    assert _safe_trace_observation_refs([
        {**valid, "document_name": "C:\\secrets\\draft.md"},
        {**valid, "document_name": "sk-1234567890abcdef.md"},
        {**valid, "line_start": True},
        {**valid, "line_end": 8},
        {**valid, "observation_kind": ["action"]},
        {**valid, "polarity": "unknown"},
        {**valid, "key_object_sha256": "not-a-digest"},
        {**valid, "text": "secret source text"},
    ]) == [valid]


def test_oracle_candidate_selector_requires_semantics_source_and_type():
    selector = {
        "case_id": "lin_melon_preference",
        "character_key": "林澈",
        "trait_type": "preference",
        "source_document": "02-character-profiles.md",
        "source_line": 8,
        "value_contains": "冰镇蜜瓜",
        "polarity": "positive",
        "stability": "stable",
    }
    candidate = {
        "character_key": "林澈",
        "trait_type": "preference",
        "origin": "explicit_setting",
        "reviewable": True,
        "value": "喜欢冰镇蜜瓜",
        "polarity": "positive",
        "stability": "stable",
        "evidence": [
            {
                "document_name": "02-character-profiles.md",
                "line_start": 8,
                "line_end": 8,
            }
        ],
    }
    assert _matches_selector(candidate, selector)
    assert not _matches_selector(
        {**candidate, "trait_type": "core_personality"}, selector
    )
    assert not _matches_selector({**candidate, "polarity": "negative"}, selector)
    assert not _matches_selector({**candidate, "stability": "core"}, selector)
    assert not _matches_selector({**candidate, "value": "喜欢梨"}, selector)
    assert _matches_selector(
        {
            **candidate,
            "value": "林澈长期保持该偏好",
            "evidence": [
                {
                    "document_name": "02-character-profiles.md",
                    "line_start": 8,
                    "line_end": 8,
                    "text": "林澈长期喜欢冰镇蜜瓜。",
                }
            ],
        },
        selector,
    )
    assert not _matches_selector(
        {
            **candidate,
            "value": "林澈长期保持该偏好",
            "evidence": [
                {
                    "document_name": "02-character-profiles.md",
                    "line_start": 8,
                    "line_end": 8,
                    "text": "林澈长期喜欢梨。",
                }
            ],
        },
        selector,
    )
    assert not _matches_selector(
        {
            **candidate,
            "evidence": [
                {
                    "document_name": "02-character-profiles.md",
                    "line_start": 9,
                    "line_end": 9,
                }
            ],
        },
        selector,
    )


def test_alpha_fixture_is_developer_visible_and_covers_bounded_major_scenarios():
    fixture = live_acceptance.FIXTURES["alpha-v1"]
    oracle = _validate_oracle_payload(json.loads(
        (fixture / "acceptance-oracle.json").read_text(encoding="utf-8")
    ))
    expected = {row["case_id"]: row for row in oracle["expected_cases"]}
    assert oracle["dataset_boundary"] == {
        "developer_visible": True,
        "blind_holdout": False,
        "production_quality_claim": False,
        "open_text_generalization_claim": False,
    }
    assert {row["dimension"] for row in expected.values()} == {
        "core_personality", "preference", "speech_pattern", "value",
        "relationship_attitude", "motivation_goal",
    }
    assert expected["suxian_published_growth"]["expected_report_class"] == "none"
    assert expected["qiwu_active_disguise"]["expected_report_class"] == "none"
    assert expected["shenyan_single_preference_reversal"][
        "expected_report_class"
    ] == "review_clue"
    assert sum(
        row["expected_report_class"] == "formal" for row in expected.values()
    ) == 5
    assert len(oracle["author_axes"]) == 1
    tangxiu_axis = oracle["author_axes"][0]
    assert tangxiu_axis["case_id"] == "tangxiu_value_shift"
    assert tangxiu_axis["trait_type"] == "value"
    assert tangxiu_axis["axis_alignment"] == "same"
    assert "comparison_key" not in tangxiu_axis
    boundary_text = (fixture / "README.md").read_text(encoding="utf-8")
    assert "人工" in boundary_text and "封闭测试集" in boundary_text
    assert "作者" in boundary_text and "冻结快照" in boundary_text

    profiles = (fixture / "02-character-profiles.md").read_text(
        encoding="utf-8"
    ).splitlines()
    for selector in oracle["confirm_candidates"]:
        source = profiles[selector["source_line"] - 1]
        assert selector["character_key"] in source
        assert selector["value_contains"] in source
    relationship = next(
        row for row in oracle["confirm_candidates"]
        if row["trait_type"] == "relationship_attitude"
    )
    candidate = {
        **relationship,
        "origin": "explicit_setting",
        "reviewable": True,
        "value": profiles[relationship["source_line"] - 1],
        "evidence": [{
            "document_name": relationship["source_document"],
            "line_start": relationship["source_line"],
            "line_end": relationship["source_line"],
        }],
    }
    assert _matches_selector(candidate, relationship)
    assert not _matches_selector({**candidate, "key_object": "另一角色"}, relationship)

    runtime = _alpha_runtime_provenance()
    assert runtime["capabilities"]["embeddings"] is False
    assert runtime["rag"]["profile_fingerprint"] is None
    assert live_acceptance._safe_runtime_provenance(runtime) == runtime
    assert live_acceptance._alpha_runtime_ready(runtime)
    limits = runtime["character_consistency_limits"]
    for field, invalid in (
        ("explanation_review_v1", False),
        ("explanation_review_schema_version", "character-explanation-review-v1"),
        ("explanation_review_prompt_version", "character-explanation-review-prompt-v1"),
        ("scoped_axis_drift_v1", False),
        ("signal_support_id_v4", False),
        ("signal_support_segmenter_version", "unknown"),
        ("signal_semantic_scope_v5", False),
        ("signal_semantic_scope_version", "unknown"),
        ("signal_scope_review_v1", False),
        ("signal_scope_review_schema_version", "unknown"),
        ("signal_scope_review_prompt_version", "character-scope-review-prompt-v2"),
        ("draft_actor_review_v1", False),
        ("target_bound_draft_review_v3", False),
        ("target_bound_draft_review_schema_version", "unknown"),
    ):
        unavailable = deepcopy(runtime)
        unavailable["character_consistency_limits"][field] = invalid
        assert not live_acceptance._alpha_runtime_ready(unavailable)
    assert limits["scoped_axis_drift_v1"] is True
    assert limits["signal_support_id_v4"] is True
    assert limits["signal_support_segmenter_version"] == "assertion-index-v1"
    assert limits["signal_semantic_scope_v5"] is True
    assert limits["signal_semantic_scope_version"] == "semantic-scope-v6"
    assert limits["signal_scope_review_v1"] is True
    assert limits["signal_scope_review_schema_version"] == (
        "character-scope-review-v2"
    )
    assert limits["signal_scope_review_prompt_version"] == (
        "character-scope-review-prompt-v4"
    )


@pytest.mark.parametrize(
    ("support_identity", "expected_code"),
    (
        ("binding_chain_off", "alpha_ooc_capabilities_unavailable"),
        ("v4_only", "alpha_ooc_capabilities_unavailable"),
        ("v4_v5_only", "alpha_ooc_capabilities_unavailable"),
        ("missing", "runtime_provenance_invalid"),
        ("segmenter_version_mismatch", "runtime_provenance_invalid"),
        ("scope_review_version_mismatch", "runtime_provenance_invalid"),
    ),
)
def test_alpha_runner_preflight_requires_strict_formal_support_binding_identity(
    monkeypatch, support_identity, expected_code,
):
    # run() selects its fixture through a module global; register the current
    # value with monkeypatch so this preflight-only test cannot leak Alpha into
    # the oracle tests that follow.
    monkeypatch.setattr(live_acceptance, "DEMO", live_acceptance.DEMO)
    provenance = _alpha_runtime_provenance()
    limits = provenance["character_consistency_limits"]
    if support_identity == "binding_chain_off":
        limits.update({
            "signal_support_id_v4": False,
            "signal_support_segmenter_version": None,
            "signal_semantic_scope_v5": False,
            "signal_semantic_scope_version": None,
            "signal_scope_review_v1": False,
            "signal_scope_review_schema_version": None,
            "signal_scope_review_prompt_version": None,
        })
    elif support_identity == "v4_only":
        limits.update({
            "signal_semantic_scope_v5": False,
            "signal_semantic_scope_version": None,
            "signal_scope_review_v1": False,
            "signal_scope_review_schema_version": None,
            "signal_scope_review_prompt_version": None,
        })
    elif support_identity == "v4_v5_only":
        limits.update({
            "signal_scope_review_v1": False,
            "signal_scope_review_schema_version": None,
            "signal_scope_review_prompt_version": None,
        })
    elif support_identity == "missing":
        limits.pop("signal_support_id_v4")
        limits.pop("signal_support_segmenter_version")
    elif support_identity == "scope_review_version_mismatch":
            # A v6/v2 producer identity cannot be mixed with a legacy prompt.
            # Legacy reports remain readable only as a consistent v5/v1 triple.
        limits["signal_scope_review_prompt_version"] = (
            "character-scope-review-prompt-v2"
        )
    else:
        limits["signal_support_segmenter_version"] = "assertion-index-v2"

    safe = live_acceptance._safe_runtime_provenance(provenance)
    if expected_code == "alpha_ooc_capabilities_unavailable":
        assert safe == provenance
    else:
        assert safe is None

    monkeypatch.setattr(live_acceptance, "_request", lambda *_args, **_kwargs: {
        "runtime_provenance": provenance,
        "model": {"configured": True},
    })
    monkeypatch.setattr(
        live_acceptance,
        "_execute_trial",
        lambda *_args, **_kwargs: pytest.fail("preflight must fail before trials"),
    )

    with pytest.raises(AcceptanceFailure) as captured:
        live_acceptance.run(
            live_acceptance.parse_args(["--fixture", "alpha-v1"])
        )

    assert captured.value.safe_payload["code"] == expected_code
    assert captured.value.safe_payload["stage"] == "runner_preflight"


def test_author_axis_oracle_is_strict_and_legacy_oracle_defaults_empty():
    fixture = live_acceptance.FIXTURES["alpha-v1"]
    source = json.loads(
        (fixture / "acceptance-oracle.json").read_text(encoding="utf-8")
    )
    plan = source["author_axes"][0]

    legacy = deepcopy(source)
    del legacy["author_axes"]
    assert _validate_oracle_payload(legacy)["author_axes"] == []

    invalid_payloads = []
    unknown_case = deepcopy(source)
    unknown_case["author_axes"][0]["case_id"] = "unknown_case"
    invalid_payloads.append(unknown_case)

    duplicate_plan = deepcopy(source)
    duplicate_plan["author_axes"].append(deepcopy(plan))
    invalid_payloads.append(duplicate_plan)

    duplicate_case = deepcopy(source)
    duplicate = deepcopy(plan)
    duplicate["plan_id"] = "another_tangxiu_axis"
    duplicate_case["author_axes"].append(duplicate)
    invalid_payloads.append(duplicate_case)

    non_scoped = deepcopy(source)
    non_scoped["author_axes"][0].update({
        "case_id": "liyin_core_shift",
        "trait_type": "core_personality",
    })
    invalid_payloads.append(non_scoped)

    missing_scope = deepcopy(source)
    del missing_scope["author_axes"][0]["applicability_scope"]
    invalid_payloads.append(missing_scope)

    guessed_key = deepcopy(source)
    guessed_key["author_axes"][0]["comparison_key"] = "value:货物"
    invalid_payloads.append(guessed_key)

    for invalid in invalid_payloads:
        with pytest.raises(RuntimeError):
            _validate_oracle_payload(invalid)


def _oracle_trial_fixture():
    expected = _load_oracle()["expected_cases"]
    case_traits = {
        row["case_id"]: _comparison_identity(
            dimension=row["dimension"],
            trait_key=f"{row['case_id']}_trait",
        )
        for row in expected
    }
    trace = []
    visible = []
    clues = []
    for row in expected:
        runtime = case_traits[row["case_id"]]
        matched = row.get(
            "matched_observation_count",
            row.get("min_matched_observation_count", 1),
        )
        item = {
            "character_key": row["character_key"],
            "dimension": row["dimension"],
            "comparison_key_sha256": runtime["comparison_key_sha256"],
            "matched_observation_count": matched,
            "material_coverage": "complete",
            "explanation_coverage": "complete",
            "prepare_reason": row.get(
                "prepare_reason",
                row.get("prepare_reason_any_of", [None])[0],
            ),
            "review_outcome": row["review_outcome"],
            "review_verdict": row.get("review_verdict"),
            "citation_roles": row["expected_citation_roles"],
            "final_outcome": row["final_outcome"],
            "visible": row["visible"],
            "promote_reason": row["promote_reason"],
        }
        trace.append(item)
        report_class = row.get("expected_report_class")
        if report_class in {"formal", "review_clue"}:
            issue_contract = row[
                "visible_issue" if report_class == "formal" else "review_clue"
            ]
            evidence_selectors = list(issue_contract["required_evidence"])
            required_any_minimum = issue_contract.get(
                "required_any_evidence_min_matches",
                1 if issue_contract["required_any_evidence"] else 0,
            )
            evidence_selectors.extend(
                issue_contract["required_any_evidence"][:required_any_minimum]
            )
            output = {
                "character_key": row["character_key"],
                "dimension": row["dimension"],
                "comparison_key_sha256": runtime["comparison_key_sha256"],
                "subtype": issue_contract["subtype"],
                "judgement": issue_contract["judgement"],
                "evidence_refs": [
                    {
                        "document_name": selector["document_name"],
                        "line_start": selector.get("line_start", selector["line"]),
                        "line_end": selector.get("line_end", selector["line"]),
                    }
                    for selector in evidence_selectors
                ],
            }
            (visible if report_class == "formal" else clues).append(output)
    return expected, case_traits, trace, visible, clues


def test_oracle_prepare_reason_contract_is_exclusive_nonempty_and_unique():
    oracle = _load_oracle()
    preference_index = next(
        index
        for index, row in enumerate(oracle["expected_cases"])
        if row["case_id"] == "lin_melon_preference"
    )

    conflicting = deepcopy(oracle)
    conflicting["expected_cases"][preference_index]["prepare_reason"] = (
        "explicit_opposed_preference"
    )
    with pytest.raises(RuntimeError, match="exactly one prepare reason"):
        _validate_oracle_payload(conflicting)

    for invalid_allowlist in (
        [],
        ["explicit_opposed_preference", "explicit_opposed_preference"],
        ["explicit_opposed_preference", ""],
        ["explicit_opposed_preference", 1],
    ):
        invalid = deepcopy(oracle)
        invalid["expected_cases"][preference_index][
            "prepare_reason_any_of"
        ] = invalid_allowlist
        with pytest.raises(RuntimeError, match="allowlist is invalid"):
            _validate_oracle_payload(invalid)


def test_preference_prepare_reason_accepts_only_review_equivalent_allowlist():
    expected, case_traits, trace, visible, clues = _oracle_trial_fixture()
    preference_index = next(
        index
        for index, row in enumerate(expected)
        if row["case_id"] == "lin_melon_preference"
    )
    trial = {
        "case_trace": trace,
        "formal_issue_cases": visible,
        "review_clue_cases": clues,
    }
    assert _evaluate_oracle_trial(trial, expected, case_traits)[
        "all_cases_passed"
    ] is True

    reported = deepcopy(trace)
    reported[preference_index]["prepare_reason"] = "reported_opposed_preference"
    assert _evaluate_oracle_trial(
        {"case_trace": reported, "formal_issue_cases": visible,
         "review_clue_cases": clues},
        expected,
        case_traits,
    )["all_cases_passed"] is True

    unrelated = deepcopy(trace)
    unrelated[preference_index]["prepare_reason"] = "two_independent_behaviors"
    assert _evaluate_oracle_trial(
        {"case_trace": unrelated, "formal_issue_cases": visible,
         "review_clue_cases": clues},
        expected,
        case_traits,
    )["all_cases_passed"] is False


def test_oracle_trial_binds_runtime_trait_and_full_visible_issue_signature():
    expected, case_traits, trace, visible, clues = _oracle_trial_fixture()
    trial = {
        "case_trace": trace,
        "formal_issue_cases": visible,
        "review_clue_cases": clues,
    }
    result = _evaluate_oracle_trial(trial, expected, case_traits)
    assert result["all_cases_passed"] is True
    assert result["trace_has_no_unexpected_cases"] is True
    assert result["visible_issues_exact"] is True

    wrong_trait = deepcopy(trace)
    wrong_trait[0]["comparison_key_sha256"] = "0" * 64
    failed_trait = _evaluate_oracle_trial(
        {"case_trace": wrong_trait, "formal_issue_cases": visible,
         "review_clue_cases": clues},
        expected,
        case_traits,
    )
    assert failed_trait["all_cases_passed"] is False
    assert failed_trait["trace_has_no_unexpected_cases"] is False

    wrong_subtype = deepcopy(visible)
    wrong_subtype[0]["subtype"] = "wrong_subtype"
    failed_subtype = _evaluate_oracle_trial(
        {"case_trace": trace, "formal_issue_cases": wrong_subtype,
         "review_clue_cases": clues},
        expected,
        case_traits,
    )
    assert failed_subtype["visible_issues_exact"] is False

    wrong_evidence = deepcopy(visible)
    wrong_evidence[0]["evidence_refs"] = [
        {
            "document_name": "04-draft-event-v1.1.md",
            "line_start": 99,
            "line_end": 99,
        }
    ]
    failed_evidence = _evaluate_oracle_trial(
        {"case_trace": trace, "formal_issue_cases": wrong_evidence,
         "review_clue_cases": clues},
        expected,
        case_traits,
    )
    assert failed_evidence["visible_issues_exact"] is False

    insufficient_distinct_core_evidence = deepcopy(visible)
    core_issue = next(
        issue
        for issue in insufficient_distinct_core_evidence
        if issue["dimension"] == "core_personality"
        and issue["character_key"] == "林澈"
    )
    core_issue["evidence_refs"] = [
        ref
        for ref in core_issue["evidence_refs"]
        if not (
            ref["document_name"] == "04-draft-event-v1.1.md"
            and ref["line_start"] == 5
        )
    ]
    insufficient = _evaluate_oracle_trial(
        {
            "case_trace": trace,
            "formal_issue_cases": insufficient_distinct_core_evidence,
            "review_clue_cases": clues,
        },
        expected,
        case_traits,
    )
    assert insufficient["visible_issues_exact"] is False

    extra_unallowed_evidence = deepcopy(clues)
    preference_issue = next(
        issue
        for issue in extra_unallowed_evidence
        if issue["dimension"] == "preference"
    )
    preference_issue["evidence_refs"].append(
        {
            "document_name": "04-draft-event-v1.1.md",
            "line_start": 11,
            "line_end": 11,
        }
    )
    unallowed = _evaluate_oracle_trial(
        {"case_trace": trace, "formal_issue_cases": visible,
         "review_clue_cases": extra_unallowed_evidence},
        expected,
        case_traits,
    )
    assert unallowed["review_clues_exact"] is False


def test_oracle_never_scores_partial_case_or_wrong_report_class_as_clean():
    expected, case_traits, trace, formal, clues = _oracle_trial_fixture()
    partial = deepcopy(trace)
    partial[0]["material_coverage"] = "partial"
    result = _evaluate_oracle_trial(
        {"case_trace": partial, "formal_issue_cases": formal,
         "review_clue_cases": clues},
        expected,
        case_traits,
    )
    assert result["all_cases_passed"] is False

    preference = next(
        row for row in clues if row["dimension"] == "preference"
    )
    wrong_class = _evaluate_oracle_trial(
        {"case_trace": trace, "formal_issue_cases": [*formal, preference],
         "review_clue_cases": [row for row in clues if row is not preference]},
        expected,
        case_traits,
    )
    assert wrong_class["report_classes_exact"] is False
    assert wrong_class["review_clues_exact"] is False


def test_safe_visible_issue_does_not_copy_trait_or_evidence_text():
    safe = _safe_visible_issue(
        {
            "metadata": {
                "character_key": "林澈",
                "dimension": "preference",
                "trait_key": "melon_preference",
                "subtype": "stable_preference_conflict",
                "judgement": "contradicts",
            },
            "evidence": [
                {
                    "document_name": "02-character-profiles.md",
                    "line_start": 8,
                    "line_end": 8,
                    "text": "must-not-enter-report",
                }
            ],
        }
    )
    assert "trait_key" not in safe
    assert "must-not-enter-report" not in repr(safe)
    assert safe["evidence_refs"] == [
        {
            "document_name": "02-character-profiles.md",
            "line_start": 8,
            "line_end": 8,
        }
    ]


def test_visible_issues_join_distinct_frozen_objects_by_candidate_digest():
    candidate_ids = (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    names = ("蜜瓜", "葡萄")
    trace = [
        _safe_case_trace_summary({
            "character_key": "林澈",
            "dimension": "preference",
            "comparison_key": f"preference:{name}",
            "confirmed_candidate_id_sha256": hashlib.sha256(
                candidate_id.encode("utf-8")
            ).hexdigest(),
        })
        for candidate_id, name in zip(candidate_ids, names)
    ]
    issue_metadata = {
        "character_key": "林澈",
        "dimension": "preference",
        "trait_key": "food_preference",
        "subtype": "stable_preference_conflict",
        "judgement": "contradicts",
    }
    issues = [
        _safe_visible_issue(
            {"metadata": {**issue_metadata, "confirmed_candidate_id": candidate_id}},
            case_trace=trace,
        )
        for candidate_id in candidate_ids
    ]

    assert [row["comparison_key_sha256"] for row in issues] == [
        hashlib.sha256(f"preference:{name}".encode("utf-8")).hexdigest()
        for name in names
    ]
    assert issues[0]["comparison_key_sha256"] != issues[1]["comparison_key_sha256"]
    without_trait_key = {**issue_metadata, "confirmed_candidate_id": candidate_ids[0]}
    del without_trait_key["trait_key"]
    assert _safe_visible_issue(
        {"metadata": without_trait_key},
        case_trace=trace,
    )["comparison_key_sha256"] == issues[0]["comparison_key_sha256"]
    for candidate_id, issue in zip(candidate_ids, issues):
        assert issue["confirmed_candidate_id_sha256"] == hashlib.sha256(
            candidate_id.encode("utf-8")
        ).hexdigest()
        assert candidate_id not in repr(issue)
        assert "food_preference" not in repr(issue)

    runtime = {
        "melon": {
            **_comparison_identity(dimension="preference", trait_key="food_preference"),
            "confirmed_candidate_id_sha256": issues[0][
                "confirmed_candidate_id_sha256"
            ],
        },
        "grape": {
            **_comparison_identity(dimension="preference", trait_key="food_preference"),
            "confirmed_candidate_id_sha256": issues[1][
                "confirmed_candidate_id_sha256"
            ],
        },
    }
    assert live_acceptance._runtime_identity(
        {"case_id": "melon", "character_key": "林澈", "dimension": "preference"},
        runtime,
        trace,
    ) == ("林澈", "preference", issues[0]["comparison_key_sha256"])
    assert live_acceptance._runtime_identity(
        {"case_id": "grape", "character_key": "林澈", "dimension": "preference"},
        runtime,
        trace,
    ) == ("林澈", "preference", issues[1]["comparison_key_sha256"])

    duplicated = [*trace, trace[0]]
    assert _safe_visible_issue(
        {"metadata": {**issue_metadata, "confirmed_candidate_id": candidate_ids[0]}},
        case_trace=duplicated,
    )["comparison_key_sha256"] is None
    assert live_acceptance._runtime_identity(
        {"case_id": "melon", "character_key": "林澈", "dimension": "preference"},
        runtime,
        duplicated,
    ) is None
    assert _safe_visible_issue(
        {"metadata": {**issue_metadata, "confirmed_candidate_id": "33333333-3333-4333-8333-333333333333"}},
        case_trace=trace,
    )["comparison_key_sha256"] is None
    assert _safe_visible_issue(
        {"metadata": {
            **issue_metadata,
            "dimension": "value",
            "confirmed_candidate_id": candidate_ids[0],
        }},
        case_trace=trace,
    )["comparison_key_sha256"] is None


def test_visible_issue_legacy_axis_fallback_requires_unique_trace():
    metadata = {
        "character_key": "林澈",
        "dimension": "preference",
        "trait_key": "食物偏好:蜜瓜",
        "confirmed_candidate_id": "11111111-1111-4111-8111-111111111111",
    }
    legacy_trace = [_safe_case_trace_summary({
        "character_key": "林澈",
        "dimension": "preference",
        "comparison_key": "preference:蜜瓜",
    })]
    expected_hash = hashlib.sha256("preference:蜜瓜".encode("utf-8")).hexdigest()
    assert _safe_visible_issue(
        {"metadata": metadata}, case_trace=legacy_trace
    )["comparison_key_sha256"] == expected_hash
    assert _safe_visible_issue(
        {"metadata": metadata}, case_trace=legacy_trace * 2
    )["comparison_key_sha256"] is None
    malformed_binding = [_safe_case_trace_summary({
        "character_key": "林澈",
        "dimension": "preference",
        "comparison_key": "preference:蜜瓜",
        "confirmed_candidate_id_sha256": "invalid",
    })]
    assert _safe_visible_issue(
        {"metadata": metadata}, case_trace=malformed_binding
    )["comparison_key_sha256"] is None


def test_run_summary_reports_frozen_issue_axis_without_object_text(monkeypatch):
    candidate_id = "11111111-1111-4111-8111-111111111111"
    frozen_key = "preference:蜜瓜"
    def fake_request(_client, method, path, **_kwargs):
        assert method == "GET"
        if path.endswith("/diagnostics"):
            return {"character_consistency": {"case_trace": [{
                "character_key": "林澈",
                "dimension": "preference",
                "comparison_key": frozen_key,
                "confirmed_candidate_id_sha256": hashlib.sha256(
                    candidate_id.encode("utf-8")
                ).hexdigest(),
            }]}}
        if path.endswith("/review-clues"):
            return {"items": [{
                "report_class": "review_clue",
                "category": "character_drift",
                "metadata": {
                    "character_key": "祁雾",
                    "dimension": "core_personality",
                    "trait_key": "social_response",
                    "subtype": "core_trait_drift",
                    "judgement": "needs_confirmation",
                },
                "evidence": [{
                    "document_name": "draft.md", "line_start": 2,
                    "line_end": 2, "text": "另一段机密原文",
                }],
            }], "truncated": False,
                    "unavailable_count": 0, "scan_limited": False}
        assert path.endswith("/provisional-clues")
        return {"items": [{
            "id": "pc_" + "a" * 32,
            "document_name": "draft.md",
            "line_start": 3,
            "line_end": 3,
            "dimension": "speech_pattern",
            "reason": "partial_model_package",
            "evidence": "不应回显的提案原文",
        }], "truncated": False}

    def fake_request_list(_client, method, path, **_kwargs):
        assert method == "GET"
        assert path.endswith("/issues")
        return [{
            "report_class": "formal",
            "category": "character_drift",
            "metadata": {
                "character_key": "林澈",
                "dimension": "preference",
                "trait_key": "food_preference",
                "confirmed_candidate_id": candidate_id,
                "judgement": "contradicts",
            },
            "evidence": [{
                "document_name": "draft.md", "line_start": 1, "line_end": 1,
                "text": "机密草稿原文",
            }],
        }]

    monkeypatch.setattr(live_acceptance, "_request", fake_request)
    monkeypatch.setattr(live_acceptance, "_request_list", fake_request_list)
    report = live_acceptance._run_summary(None, "project-1", {"id": "run-1"})
    issue = report["visible_issue_cases"][0]
    assert issue["comparison_key_sha256"] == hashlib.sha256(
        frozen_key.encode("utf-8")
    ).hexdigest()
    assert frozen_key not in repr(report)
    assert candidate_id not in repr(report)
    assert "机密草稿原文" not in repr(report)
    assert "另一段机密原文" not in repr(report)
    assert "不应回显的提案原文" not in repr(report)
    assert report["formal_character_issue_count"] == 1
    assert report["review_clue_count"] == 1
    assert report["provisional_clue_count"] == 1
    assert report["report_class_integrity"] is True
    # Historical API diagnostics did not include admission events.
    assert report["token_admission_events"] == []


@pytest.mark.parametrize(
    ("field", "secret"),
    (
        ("source_text", "private source text"),
        ("key_object", "private object identity"),
        ("credential", "sk-test-secret-not-for-report"),
    ),
)
def test_run_summary_rejects_extra_worker_provenance_fields_without_echo(
    monkeypatch, field, secret,
):
    raw_provenance = deepcopy(_alpha_runtime_provenance())
    raw_provenance[field] = secret

    def fake_request(_client, method, path, **_kwargs):
        assert method == "GET"
        if path.endswith("/diagnostics"):
            return {
                "runtime_provenance": raw_provenance,
                "character_consistency": {},
            }
        if path.endswith("/review-clues"):
            return {
                "items": [], "unavailable_count": 0,
                "scan_limited": False, "truncated": False,
            }
        assert path.endswith("/provisional-clues")
        return {"items": [], "truncated": False}

    monkeypatch.setattr(live_acceptance, "_request", fake_request)
    monkeypatch.setattr(
        live_acceptance, "_request_list", lambda *_args, **_kwargs: []
    )

    summary = live_acceptance._run_summary(
        None, "project-1", {"id": "run-1"}
    )

    assert summary["runtime_provenance"] is None
    assert secret not in repr(summary)


def test_run_summary_keeps_only_strict_worker_provenance_projection(monkeypatch):
    provenance = _alpha_runtime_provenance()

    def fake_request(_client, method, path, **_kwargs):
        assert method == "GET"
        if path.endswith("/diagnostics"):
            return {
                "runtime_provenance": provenance,
                "character_consistency": {},
            }
        if path.endswith("/review-clues"):
            return {
                "items": [], "unavailable_count": 0,
                "scan_limited": False, "truncated": False,
            }
        assert path.endswith("/provisional-clues")
        return {"items": [], "truncated": False}

    monkeypatch.setattr(live_acceptance, "_request", fake_request)
    monkeypatch.setattr(
        live_acceptance, "_request_list", lambda *_args, **_kwargs: []
    )

    summary = live_acceptance._run_summary(
        None, "project-1", {"id": "run-1"}
    )

    assert summary["runtime_provenance"] == provenance


def _semantic_visible_issue(*, judgement="contradicts"):
    return {
        "character_key": "林澈",
        "dimension": "core_personality",
        "comparison_key_sha256": "a" * 64,
        "subtype": "core_trait_drift",
        "judgement": judgement,
        "evidence_refs": [],
    }


def _complete_stage(
    *,
    issues=("stable-signature",),
    visible_issue_cases=None,
    runtime_provenance=None,
):
    if visible_issue_cases is None:
        visible_issue_cases = [_semantic_visible_issue()]
    case_trace = (
        [{"material_coverage": "complete", "explanation_coverage": "complete"}]
        if visible_issue_cases else []
    )
    result = {
        "status": "completed",
        "stage_outcome": "completed",
        "material_coverage": "complete",
        "planned_chunks": 4,
        "processed_chunks": 4,
        "model_called_chunks": 4,
        "model_completed_chunks": 4,
        "model_uncalled_chunks": 0,
        "model_incomplete_chunks": 0,
        "case_trace": case_trace,
        "case_material_complete_count": len(case_trace),
        "case_material_partial_count": 0,
        "explanation_coverage": "complete" if case_trace else "not_applicable",
        "explanation_review_required_case_count": len(case_trace),
        "explanation_review_complete_case_count": len(case_trace),
        "stage_usage": {"attempted_calls": 4, "input_tokens": 100},
        "issues": list(issues),
        "visible_issue_cases": visible_issue_cases,
        "formal_issue_cases": visible_issue_cases,
        "review_clue_cases": [],
        "provisional_clue_count": 0,
        "report_class_integrity": True,
        "review_clue_collection_complete": True,
        "provisional_clue_collection_complete": True,
    }
    if runtime_provenance is not None:
        result["runtime_provenance"] = runtime_provenance
    return result


def _passing_trial(index: int, *, runtime_provenance=None):
    return {
        "project_id": f"project-{index}",
        "baseline": _complete_stage(
            issues=(),
            visible_issue_cases=[],
            runtime_provenance=runtime_provenance,
        ),
        "draft": _complete_stage(runtime_provenance=runtime_provenance),
        "candidate_review": {
            "expected_confirmed": 5,
            "confirmed": 5,
            "rejected": 0,
            "explicit_reviewable_candidate_count": 5,
            "case_traits": {f"case-{value}": {} for value in range(5)},
        },
        "oracle": {
            "all_cases_passed": True,
            "trace_has_no_unexpected_cases": True,
            "visible_issues_exact": True,
            "review_clues_exact": True,
            "report_classes_exact": True,
        },
    }


def test_alpha_single_fresh_trial_passes_with_stability_observational():
    provenance = _alpha_runtime_provenance()
    passing = [_passing_trial(1, runtime_provenance=provenance)]
    one_trial = _evaluate_gates(
        passing,
        fixture="alpha-v1",
        api_runtime_provenance=provenance,
        post_run_api_runtime_provenance=provenance,
    )
    assert one_trial["at_least_one_fresh_full_workflow_trial"] is True
    assert one_trial["api_and_worker_runtime_provenance_match"] is True
    assert one_trial["visible_issue_semantic_identity_valid_in_each_trial"] is True
    assert all(one_trial.values())
    one_trial_observations = _evaluate_stability_observations(
        passing, fixture="alpha-v1"
    )
    assert one_trial_observations == {
        "multiple_independent_full_workflow_trials": False,
        "stable_visible_issue_semantic_identity_across_independent_trials": False,
    }
    one_trial_claims = _evaluation_claims(
        passing,
        one_trial,
        one_trial_observations,
        diagnostic_source_anchored=False,
    )
    assert one_trial_claims["full_workflow_trial_count"] == 1
    assert one_trial_claims["independent_full_workflow_trials"] is False
    assert one_trial_claims["semantic_identity_stability_assessed"] is False
    assert (
        one_trial_claims[
            "semantic_identity_stable_across_independent_trials"
        ]
        is False
    )
    one_trial_reported_gates = _report_gates(
        one_trial, one_trial_observations
    )
    assert one_trial_reported_gates[
        "stable_visible_issue_semantic_identity_across_independent_trials"
    ] is False
    assert all(
        value
        for name, value in one_trial_reported_gates.items()
        if name != "stable_visible_issue_semantic_identity_across_independent_trials"
    )


@pytest.mark.parametrize("mismatch", ("build", "explanation", "target_v3"))
def test_alpha_rejects_api_worker_runtime_provenance_mismatch(mismatch):
    api_provenance = _alpha_runtime_provenance()
    worker_provenance = deepcopy(api_provenance)
    limits = worker_provenance["character_consistency_limits"]
    if mismatch == "build":
        worker_provenance["build"]["service_artifact_sha256"] = "f" * 64
    elif mismatch == "explanation":
        limits["explanation_max_windows_per_case"] -= 1
    else:
        limits["target_bound_draft_review_v3"] = False
        for field in live_acceptance._TARGET_BOUND_DRAFT_REVIEW_V3:
            limits[field] = None
    assert live_acceptance._safe_runtime_provenance(worker_provenance) is not None
    trials = [_passing_trial(1, runtime_provenance=worker_provenance)]

    gates = _evaluate_gates(
        trials,
        fixture="alpha-v1",
        api_runtime_provenance=api_provenance,
        post_run_api_runtime_provenance=api_provenance,
    )

    assert gates["api_and_worker_runtime_provenance_match"] is False
    assert not all(gates.values())


@pytest.mark.parametrize("phase", ("baseline", "draft"))
def test_alpha_requires_worker_runtime_provenance_from_both_phases(phase):
    provenance = _alpha_runtime_provenance()
    trial = _passing_trial(1, runtime_provenance=provenance)
    trial[phase]["runtime_provenance"] = None

    gates = _evaluate_gates(
        [trial],
        fixture="alpha-v1",
        api_runtime_provenance=provenance,
        post_run_api_runtime_provenance=provenance,
    )

    assert gates["api_and_worker_runtime_provenance_match"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("signal_max_completion_tokens", 4095),
        ("signal_max_response_bytes", 63_999),
        ("signal_timeout_seconds", 29.0),
        ("drift_max_completion_tokens", 999),
        ("drift_max_response_bytes", 31_999),
        ("drift_timeout_seconds", 29.0),
    ),
)
def test_strict_runtime_consumer_rejects_provider_limits_above_stage(
    field, value,
):
    provenance = _alpha_runtime_provenance()
    provenance["character_consistency_limits"][field] = value
    assert live_acceptance._safe_runtime_provenance(provenance) is None


@pytest.mark.parametrize(
    ("section", "field", "value"),
    (
        ("chat_provider", "temperature", False),
        ("build", "git_revision", "a" * 41),
    ),
)
def test_strict_runtime_consumer_rejects_ambiguous_scalar_identity(
    section, field, value,
):
    provenance = _alpha_runtime_provenance()
    provenance[section][field] = value
    assert live_acceptance._safe_runtime_provenance(provenance) is None


def test_alpha_run_uses_safe_health_projection_and_observational_stability(
    monkeypatch,
):
    provenance = _alpha_runtime_provenance()
    reports = []
    monkeypatch.setattr(
        live_acceptance,
        "_local_service_artifact_sha256",
        lambda _root: provenance["build"]["service_artifact_sha256"],
    )
    monkeypatch.setattr(live_acceptance, "_request", lambda *_args, **_kwargs: {
        "runtime_provenance": provenance,
        "model": {"configured": True},
    })
    monkeypatch.setattr(
        live_acceptance,
        "_execute_trial",
        lambda *_args, trial_number, **_kwargs: _passing_trial(
            trial_number, runtime_provenance=provenance
        ),
    )
    monkeypatch.setattr(
        live_acceptance, "_emit_report", lambda report, _path: reports.append(report)
    )

    result = live_acceptance.run(
        live_acceptance.parse_args(["--fixture", "alpha-v1"])
    )

    assert result == 0
    assert reports[0]["passed"] is True
    assert reports[0]["runtime_provenance"] == provenance
    assert reports[0]["claims"]["local_service_artifact_match_enforced"] is True
    assert reports[0]["claims"]["local_service_artifact_matches_live_api"] is True
    assert reports[0]["claims"]["remote_service_artifact_override"] is False
    assert reports[0]["gates"][
        "post_run_api_runtime_provenance_matches_preflight"
    ] is True
    assert "api_and_worker_runtime_provenance_match" in reports[0][
        "gate_policy"
    ]["required_for_pass"]
    assert (
        "stable_visible_issue_semantic_identity_across_independent_trials"
        not in reports[0]["gate_policy"]["required_for_pass"]
    )
    assert reports[0]["gate_policy"]["observational_only"] == [
        "stable_visible_issue_semantic_identity_across_independent_trials"
    ]


@pytest.mark.parametrize(
    ("local_hash", "expected_code"),
    (
        (None, "local_service_artifact_unavailable"),
        ("f" * 64, "service_artifact_mismatch"),
    ),
)
def test_alpha_preflight_fails_closed_without_matching_local_service_artifact(
    monkeypatch, local_hash, expected_code,
):
    provenance = _alpha_runtime_provenance()
    secret = "sk-local-artifact-preflight-secret"
    monkeypatch.setattr(
        live_acceptance,
        "_local_service_artifact_sha256",
        lambda _root: local_hash,
    )
    monkeypatch.setattr(live_acceptance, "_request", lambda *_args, **_kwargs: {
        "runtime_provenance": provenance,
        "model": {"configured": True, "credential": secret},
    })
    monkeypatch.setattr(
        live_acceptance,
        "_execute_trial",
        lambda *_args, **_kwargs: pytest.fail("artifact preflight must fail first"),
    )

    with pytest.raises(AcceptanceFailure) as captured:
        live_acceptance.run(
            live_acceptance.parse_args(["--fixture", "alpha-v1"])
        )

    assert captured.value.safe_payload == {
        "code": expected_code,
        "stage": "runner_preflight",
        "details": {},
    }
    assert secret not in repr(captured.value.safe_payload)


def test_alpha_remote_service_artifact_override_is_explicit_and_audited(
    monkeypatch,
):
    provenance = _alpha_runtime_provenance()
    reports = []
    monkeypatch.setattr(
        live_acceptance,
        "_local_service_artifact_sha256",
        lambda _root: "f" * 64,
    )
    monkeypatch.setattr(live_acceptance, "_request", lambda *_args, **_kwargs: {
        "runtime_provenance": provenance,
        "model": {"configured": True},
    })
    monkeypatch.setattr(
        live_acceptance,
        "_execute_trial",
        lambda *_args, trial_number, **_kwargs: _passing_trial(
            trial_number, runtime_provenance=provenance
        ),
    )
    monkeypatch.setattr(
        live_acceptance, "_emit_report", lambda report, _path: reports.append(report)
    )

    result = live_acceptance.run(live_acceptance.parse_args([
        "--fixture", "alpha-v1", "--allow-remote-service-artifact",
    ]))

    assert result == 0
    assert reports[0]["claims"]["local_service_artifact_match_enforced"] is False
    assert reports[0]["claims"]["local_service_artifact_matches_live_api"] is False
    assert reports[0]["claims"]["remote_service_artifact_override"] is True


@pytest.mark.parametrize("post_mode", ("drift", "invalid", "missing", "unavailable"))
def test_alpha_post_run_health_fails_closed_without_echo(
    monkeypatch, post_mode,
):
    preflight = _alpha_runtime_provenance()
    post_only_secret = f"post-only-{post_mode}-secret"
    if post_mode == "drift":
        post_provenance = deepcopy(preflight)
        post_provenance["chat_provider"]["model_alias"] = post_only_secret
    elif post_mode == "invalid":
        post_provenance = deepcopy(preflight)
        post_provenance["credential"] = post_only_secret
    else:
        post_provenance = None
    reports = []
    health_calls = 0
    monkeypatch.setattr(
        live_acceptance,
        "_local_service_artifact_sha256",
        lambda _root: preflight["build"]["service_artifact_sha256"],
    )

    def fake_request(_client, method, path, **_kwargs):
        nonlocal health_calls
        assert method == "GET"
        assert path == "/health"
        health_calls += 1
        if health_calls == 1:
            return {
                "runtime_provenance": preflight,
                "model": {"configured": True},
            }
        if post_mode == "unavailable":
            raise AcceptanceFailure(
                "http_request_failed",
                safe_payload={
                    "code": "http_request_failed",
                    "stage": "http_api",
                    "details": {"credential": post_only_secret},
                },
            )
        if post_mode == "missing":
            return {"status": "ok", "credential": post_only_secret}
        return {"runtime_provenance": post_provenance}

    monkeypatch.setattr(live_acceptance, "_request", fake_request)
    monkeypatch.setattr(
        live_acceptance,
        "_execute_trial",
        lambda *_args, trial_number, **_kwargs: _passing_trial(
            trial_number, runtime_provenance=preflight
        ),
    )
    monkeypatch.setattr(
        live_acceptance, "_emit_report", lambda report, _path: reports.append(report)
    )

    result = live_acceptance.run(
        live_acceptance.parse_args(["--fixture", "alpha-v1"])
    )

    assert health_calls == 2
    assert result == 1
    assert reports[0]["passed"] is False
    assert reports[0]["runtime_provenance"] == preflight
    assert reports[0]["gates"][
        "post_run_api_runtime_provenance_matches_preflight"
    ] is False
    assert (
        "post_run_api_runtime_provenance_matches_preflight"
        in reports[0]["gate_policy"]["required_for_pass"]
    )
    assert post_only_secret not in repr(reports[0])


@pytest.mark.parametrize(
    ("field", "secret"),
    (
        ("source_text", "private health source"),
        ("key_object", "private health object"),
        ("credential", "sk-private-health-secret"),
    ),
)
def test_run_rejects_unsafe_health_provenance_without_echo(
    monkeypatch, field, secret,
):
    raw_provenance = deepcopy(_alpha_runtime_provenance())
    raw_provenance[field] = secret
    monkeypatch.setattr(live_acceptance, "_request", lambda *_args, **_kwargs: {
        "runtime_provenance": raw_provenance,
        "model": {"configured": True},
    })

    with pytest.raises(AcceptanceFailure) as captured:
        live_acceptance.run(
            live_acceptance.parse_args(["--fixture", "alpha-v1"])
        )

    assert captured.value.safe_payload == {
        "code": "runtime_provenance_invalid",
        "stage": "runner_preflight",
        "details": {},
    }
    assert secret not in repr(captured.value.safe_payload)


def test_legacy_explicit_single_trial_emits_failed_report(monkeypatch):
    provenance = _alpha_runtime_provenance()
    reports = []
    monkeypatch.setattr(live_acceptance, "_request", lambda *_args, **_kwargs: {
        "runtime_provenance": provenance,
        "model": {"configured": True},
    })
    monkeypatch.setattr(
        live_acceptance,
        "_execute_trial",
        lambda *_args, trial_number, **_kwargs: _passing_trial(trial_number),
    )
    monkeypatch.setattr(
        live_acceptance, "_emit_report", lambda report, _path: reports.append(report)
    )

    result = live_acceptance.run(
        live_acceptance.parse_args(
            ["--fixture", "demo", "--trials", "1"]
        )
    )

    assert result == 1
    assert reports[0]["passed"] is False
    assert reports[0]["gates"][
        "at_least_three_independent_full_workflow_trials"
    ] is False
    assert reports[0]["gates"][
        "stable_visible_issue_semantic_identity_across_independent_trials"
    ] is False
    assert (
        "stable_visible_issue_semantic_identity_across_independent_trials"
        in reports[0]["gate_policy"]["required_for_pass"]
    )
    assert reports[0]["gate_policy"]["observational_only"] == []


@pytest.mark.parametrize(
    "fixture", ("demo", "ooc-v1", "ooc-transfer-v1")
)
def test_legacy_fixture_single_trial_generates_failing_gates(fixture):
    gates = _evaluate_gates([_passing_trial(1)], fixture=fixture)
    assert gates["at_least_three_independent_full_workflow_trials"] is False
    assert gates[
        "stable_visible_issue_semantic_identity_across_independent_trials"
    ] is False
    assert not all(gates.values())


def test_legacy_fixture_three_consistent_trials_pass_required_stability():
    passing = [_passing_trial(index) for index in range(3)]
    gates = _evaluate_gates(passing, fixture="demo")
    assert gates["at_least_three_independent_full_workflow_trials"] is True
    assert gates[
        "stable_visible_issue_semantic_identity_across_independent_trials"
    ] is True
    assert all(gates.values())

    observations = _evaluate_stability_observations(passing, fixture="demo")
    assert observations == {
        "multiple_independent_full_workflow_trials": True,
        "stable_visible_issue_semantic_identity_across_independent_trials": True,
    }

    repeated_project = deepcopy(passing)
    repeated_project[2]["project_id"] = repeated_project[1]["project_id"]
    repeated = _evaluate_gates(repeated_project, fixture="demo")
    assert repeated["at_least_three_independent_full_workflow_trials"] is False


def test_alpha_two_consistent_trials_make_stability_observable_only():
    provenance = _alpha_runtime_provenance()
    passing = [
        _passing_trial(index, runtime_provenance=provenance)
        for index in range(2)
    ]
    two_trial_observations = _evaluate_stability_observations(
        passing, fixture="alpha-v1"
    )
    assert two_trial_observations == {
        "multiple_independent_full_workflow_trials": True,
        "stable_visible_issue_semantic_identity_across_independent_trials": True,
    }
    passing[1]["draft"]["visible_issue_cases"][0]["judgement"] = "explained"
    required = _evaluate_gates(
        passing,
        fixture="alpha-v1",
        api_runtime_provenance=provenance,
        post_run_api_runtime_provenance=provenance,
    )
    observations = _evaluate_stability_observations(
        passing, fixture="alpha-v1"
    )
    assert all(required.values())
    assert observations[
        "stable_visible_issue_semantic_identity_across_independent_trials"
    ] is False


@pytest.mark.parametrize(
    ("fixture", "expected"),
    (
        ("alpha-v1", 1),
        ("demo", 3),
        ("ooc-v1", 3),
        ("ooc-transfer-v1", 3),
    ),
)
def test_default_trial_count_is_routed_by_fixture(fixture, expected):
    assert live_acceptance.parse_args(["--fixture", fixture]).trials == expected
    assert live_acceptance.parse_args(
        ["--fixture", fixture, "--trials", "1"]
    ).trials == 1


def test_common_functional_gates_remain_fail_closed():
    passing = [_passing_trial(index) for index in range(3)]
    extra_candidate = deepcopy(passing)
    extra_candidate[0]["candidate_review"]["rejected"] = 1
    extra_candidate[0]["candidate_review"][
        "explicit_reviewable_candidate_count"
    ] = 6
    extras = _evaluate_gates(extra_candidate)
    assert extras["baseline_explicit_candidate_set_matches_oracle_exactly"] is False

    incomplete_baseline = deepcopy(passing)
    incomplete_baseline[0]["baseline"]["model_completed_chunks"] = 3
    incomplete_baseline[0]["baseline"]["model_incomplete_chunks"] = 1
    incomplete = _evaluate_gates(incomplete_baseline)
    assert (
        incomplete[
            "baseline_all_planned_chunks_model_completed_without_degradation"
        ]
        is False
    )

    provisional = deepcopy(passing)
    provisional[0]["draft"]["provisional_clue_count"] = 1
    provisional_gates = _evaluate_gates(provisional)
    assert provisional_gates["draft_has_no_unverified_provisional_clues"] is False

    partial_case = deepcopy(passing)
    partial_case[0]["draft"]["case_trace"][0]["material_coverage"] = "partial"
    partial_case[0]["draft"]["case_material_complete_count"] = 0
    partial_case[0]["draft"]["case_material_partial_count"] = 1
    coverage_gates = _evaluate_gates(partial_case)
    assert coverage_gates[
        "draft_case_material_and_explanation_coverage_complete"
    ] is False


def test_legacy_three_trial_stability_gate_uses_semantic_issue_multiset():
    passing = [_passing_trial(index) for index in range(3)]
    for index, trial in enumerate(passing, start=4):
        trial["draft"]["issues"] = [f"same-issue-with-evidence-line-{index}"]
        trial["draft"]["visible_issue_cases"][0]["evidence_refs"] = [
            {
                "document_name": "04-draft-event-v1.1.md",
                "line_start": index,
                "line_end": index,
            }
        ]
    observations = _evaluate_stability_observations(passing)
    assert observations[
        "multiple_independent_full_workflow_trials"
    ] is True
    assert observations[
        "stable_visible_issue_semantic_identity_across_independent_trials"
    ] is True

    changed_semantics = deepcopy(passing)
    changed_semantics[2]["draft"]["visible_issue_cases"][0][
        "judgement"
    ] = "explained"
    changed_observations = _evaluate_stability_observations(changed_semantics)
    assert changed_observations[
        "stable_visible_issue_semantic_identity_across_independent_trials"
    ] is False
    changed_required_gates = _evaluate_gates(changed_semantics)
    assert changed_required_gates[
        "visible_issue_semantic_identity_valid_in_each_trial"
    ] is True
    assert changed_required_gates[
        "stable_visible_issue_semantic_identity_across_independent_trials"
    ] is False
    assert not all(changed_required_gates.values())

    changed_multiplicity = deepcopy(passing)
    changed_multiplicity[2]["draft"]["visible_issue_cases"].append(
        deepcopy(changed_multiplicity[2]["draft"]["visible_issue_cases"][0])
    )
    multiplicity_observations = _evaluate_stability_observations(
        changed_multiplicity
    )
    assert multiplicity_observations[
        "stable_visible_issue_semantic_identity_across_independent_trials"
    ] is False


def test_baseline_admission_reports_transport_degradation_safely():
    summary = _complete_stage(issues=())
    summary.update(
        {
            "stage_outcome": "degraded",
            "stage_reason": "model_stage_unavailable",
            "material_coverage": "partial",
            "stage_usage": {
                "attempted_calls": 1,
                "input_tokens": 0,
                "completion_tokens": 0,
                "charged_tokens": 512,
            },
            "reason_counts": {"provider_transport": 1},
            "source_text": "must-not-enter-admission",
        }
    )
    admission = _baseline_admission(summary)
    assert admission["admitted"] is False
    assert (
        admission["checks"]["character_stage_completed_without_degradation"]
        is False
    )
    assert admission["checks"]["model_input_tokens_reported"] is False
    assert "must-not-enter-admission" not in repr(admission)


def test_execute_trial_rejects_baseline_before_candidate_selector(monkeypatch):
    review_called = False

    def fake_request(_client, method, path, **_kwargs):
        assert method == "POST"
        assert path == "/api/v1/projects"
        return {"id": "project-baseline-failure"}

    def fake_review(*_args, **_kwargs):
        nonlocal review_called
        review_called = True
        raise AssertionError("candidate review must not run after baseline degradation")

    degraded = _complete_stage(issues=())
    degraded.update(
        {
            "stage_outcome": "degraded",
            "stage_reason": "model_stage_unavailable",
            "material_coverage": "partial",
            "stage_usage": {"attempted_calls": 1, "input_tokens": 0},
        }
    )
    monkeypatch.setattr(live_acceptance, "_request", fake_request)
    monkeypatch.setattr(live_acceptance, "_upload", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(live_acceptance, "_start_run", lambda *_args: "baseline")
    monkeypatch.setattr(
        live_acceptance,
        "_wait_run",
        lambda *_args, **_kwargs: {"id": "baseline", "status": "completed"},
    )
    monkeypatch.setattr(
        live_acceptance,
        "_run_summary",
        lambda *_args, **_kwargs: degraded,
    )
    monkeypatch.setattr(live_acceptance, "_review_explicit_candidates", fake_review)

    with pytest.raises(AcceptanceFailure) as captured:
        _execute_trial(
            object(),
            {"confirm_candidates": [], "expected_cases": []},
            trial_number=1,
            timeout_seconds=1,
        )
    assert review_called is False
    assert captured.value.safe_payload["code"] == "baseline_admission_failed"
    assert captured.value.safe_payload["stage"] == "baseline_admission"
    assert (
        captured.value.safe_payload["details"]["baseline_admission"][
            "safe_summary"
        ]["stage_reason"]
        == "model_stage_unavailable"
    )


def _author_axis_review_http_mock(
    *,
    comparison_key="value:人货取舍",
    axis_mode="valid",
):
    selected_id = "11111111-1111-4111-8111-111111111111"
    rejected_id = "00000000-0000-4000-8000-000000000000"
    axis_id = "22222222-2222-4222-8222-222222222222"
    selector = {
        "case_id": "tangxiu_value_shift",
        "character_key": "唐岫",
        "trait_type": "value",
        "source_document": "02-character-profiles.md",
        "source_line": 17,
        "value_contains": "先救人再保货",
        "polarity": "positive",
        "stability": "stable",
    }
    plan = {
        "plan_id": "tangxiu_value_rescue_before_cargo",
        "case_id": "tangxiu_value_shift",
        "trait_type": "value",
        "display_name": "险情中先救人",
        "definition": "人和货物同时遇险时是否先保障人的安全",
        "positive_proposition": "唐岫在人和货物同时遇险时先救人再保货",
        "applicability_scope": "人和货物同时遇险且只能先处理一方时",
        "axis_alignment": "same",
    }
    selected = {
        "id": selected_id,
        "revision": 0,
        "character_key": "唐岫",
        "trait_type": "value",
        "trait_key": "rescue_before_cargo",
        "comparison_key": comparison_key,
        "key_object": "人货取舍",
        "value": "唐岫坚持先救人再保货",
        "polarity": "positive",
        "stability": "stable",
        "origin": "explicit_setting",
        "reviewable": True,
        "evidence": [{
            "document_name": "02-character-profiles.md",
            "line_start": 17,
            "line_end": 17,
            "text": "唐岫坚持先救人再保货。",
        }],
    }
    rejected = {
        "id": rejected_id,
        "revision": 0,
        "character_key": "唐岫",
        "trait_type": "preference",
        "trait_key": "tea_preference",
        "value": "喜欢清茶",
        "polarity": "positive",
        "stability": "stable",
        "origin": "explicit_setting",
        "reviewable": True,
        "evidence": [{
            "document_name": "02-character-profiles.md",
            "line_start": 18,
            "line_end": 18,
        }],
    }
    axis = {
        "id": axis_id,
        "project_id": "project-id",
        "trait_type": "value",
        "version": 1,
        "display_name": plan["display_name"],
        "definition": plan["definition"],
        "definition_sha256": hashlib.sha256(
            plan["definition"].encode("utf-8")
        ).hexdigest(),
        "positive_proposition": plan["positive_proposition"],
        "positive_proposition_sha256": hashlib.sha256(
            plan["positive_proposition"].encode("utf-8")
        ).hexdigest(),
        "comparison_key": comparison_key,
        "applicability_scope": plan["applicability_scope"],
        "applicability_scope_sha256": hashlib.sha256(
            plan["applicability_scope"].encode("utf-8")
        ).hexdigest(),
        "positive_proposition_authored_at": "2026-09-28T00:00:00Z",
    }
    if axis_mode == "bad_hash":
        axis["positive_proposition_sha256"] = "0" * 64
    events = []

    def fake_request(_client, method, path, **kwargs):
        if method == "GET" and path.endswith("/characters"):
            return {"items": [{"character_key": "唐岫"}]}
        if method == "GET" and "/profile-candidates" in path:
            return {"items": [selected, rejected]}
        if method == "POST" and path.endswith("/character-trait-axes"):
            events.append(("axis", path, kwargs["json"]))
            if axis_mode == "http_failure":
                raise AcceptanceFailure(
                    "http_request_failed",
                    safe_payload={
                        "code": "http_request_failed",
                        "stage": "http_api",
                        "details": {"method": "POST", "status_code": 503},
                    },
                )
            return axis
        if method == "POST" and path.endswith("/decisions"):
            payload = kwargs["json"]
            events.append(("decision", path, payload))
            candidate = selected if selected_id in path else rejected
            reviewed = {**candidate, "review_state": (
                "confirmed" if payload["decision"] == "confirm" else "rejected"
            )}
            if payload["decision"] == "confirm" and "approved_axis_id" in payload:
                reviewed.update({
                    "approved_axis_id": axis_id,
                    "approved_axis_version": 1,
                    "axis_alignment": "same",
                    "axis_polarity": "positive",
                    "axis_positive_proposition_sha256": (
                        axis["positive_proposition_sha256"]
                    ),
                })
            return {"candidate": reviewed}
        raise AssertionError(f"unexpected request: {method} {path}")

    return {
        "fake_request": fake_request,
        "events": events,
        "selector": selector,
        "plan": plan,
        "selected_id": selected_id,
        "axis_id": axis_id,
        "comparison_key": comparison_key,
        "axis": axis,
    }


def test_author_axis_is_created_before_decisions_and_bound_from_candidate(monkeypatch):
    fixture = _author_axis_review_http_mock()
    monkeypatch.setattr(live_acceptance, "_request", fixture["fake_request"])

    review = _review_explicit_candidates(
        object(),
        "project-id",
        [fixture["selector"]],
        author_axes=[fixture["plan"]],
    )

    events = fixture["events"]
    assert events[0][0] == "axis"
    assert all(event[0] == "decision" for event in events[1:])
    axis_payload = events[0][2]
    assert axis_payload["comparison_key"] == fixture["comparison_key"]
    assert "comparison_key" not in fixture["plan"]
    confirm_payload = next(
        event[2] for event in events
        if event[0] == "decision" and event[2]["decision"] == "confirm"
    )
    assert confirm_payload["approved_axis_id"] == fixture["axis_id"]
    assert confirm_payload["expected_axis_version"] == 1
    assert confirm_payload["axis_alignment"] == "same"
    assert confirm_payload["expected_axis_positive_proposition_sha256"] == (
        fixture["axis"]["positive_proposition_sha256"]
    )
    assert confirm_payload["expected_axis_applicability_scope_sha256"] == (
        fixture["axis"]["applicability_scope_sha256"]
    )
    assert confirm_payload["scope_applicability_confirmed"] is True

    trait = review["case_traits"]["tangxiu_value_shift"]
    assert trait == {
        "dimension": "value",
        "comparison_key_sha256": hashlib.sha256(
            fixture["comparison_key"].encode("utf-8")
        ).hexdigest(),
        "confirmed_candidate_id_sha256": hashlib.sha256(
            fixture["selected_id"].encode("utf-8")
        ).hexdigest(),
        "axis_bound": True,
        "axis_id_sha256": hashlib.sha256(
            fixture["axis_id"].encode("utf-8")
        ).hexdigest(),
        "axis_version": 1,
        "approved_axis_runtime_comparison_key_sha256": hashlib.sha256(
            f"approved_axis:{fixture['axis_id']}:1".encode("utf-8")
        ).hexdigest(),
        "axis_alignment": "same",
        "axis_positive_proposition_sha256": (
            fixture["axis"]["positive_proposition_sha256"]
        ),
        "axis_applicability_scope_sha256": (
            fixture["axis"]["applicability_scope_sha256"]
        ),
        "axis_creation_status": "created",
        "axis_decision_status": "confirmed_bound",
    }
    assert review["author_axis_operation_status"] == "created_and_bound"
    assert fixture["axis_id"] not in repr(review)
    assert fixture["selected_id"] not in repr(review)
    assert fixture["comparison_key"] not in repr(review)
    assert fixture["plan"]["definition"] not in repr(review)
    assert fixture["plan"]["positive_proposition"] not in repr(review)
    assert fixture["plan"]["applicability_scope"] not in repr(review)


def test_runtime_identity_uses_prebound_approved_axis_key_not_semantic_key():
    candidate_id = "11111111-1111-4111-8111-111111111111"
    axis_id = "22222222-2222-4222-8222-222222222222"
    candidate_hash = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
    semantic_hash = hashlib.sha256(
        "value:rescue-before-goods".encode("utf-8")
    ).hexdigest()
    runtime_key = f"approved_axis:{axis_id}:3"
    runtime_hash = hashlib.sha256(runtime_key.encode("utf-8")).hexdigest()
    expected = {
        "case_id": "tangxiu_value_shift",
        "character_key": "唐岫",
        "dimension": "value",
    }
    trait = {
        "dimension": "value",
        "comparison_key_sha256": semantic_hash,
        "confirmed_candidate_id_sha256": candidate_hash,
        "axis_bound": True,
        "axis_id_sha256": hashlib.sha256(axis_id.encode("utf-8")).hexdigest(),
        "axis_version": 3,
        "approved_axis_runtime_comparison_key_sha256": runtime_hash,
        "axis_alignment": "same",
        "axis_positive_proposition_sha256": "a" * 64,
        "axis_applicability_scope_sha256": "b" * 64,
        "axis_creation_status": "created",
        "axis_decision_status": "confirmed_bound",
    }
    trace = [{
        "character_key": "唐岫",
        "dimension": "value",
        "comparison_key_sha256": runtime_hash,
        "confirmed_candidate_id_sha256": candidate_hash,
    }]

    assert live_acceptance._runtime_identity(
        expected, {"tangxiu_value_shift": trait}, trace
    ) == ("唐岫", "value", runtime_hash)
    assert runtime_hash != semantic_hash


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("confirmed_candidate_id_sha256", "c" * 64),
        ("approved_axis_runtime_comparison_key_sha256", "d" * 64),
        ("axis_id_sha256", "invalid"),
        ("axis_version", 0),
        ("axis_creation_status", "not_requested"),
        ("axis_decision_status", "confirmed_unbound"),
    ),
)
def test_runtime_identity_rejects_unverified_approved_axis_binding(
    field, replacement,
):
    candidate_id = "11111111-1111-4111-8111-111111111111"
    axis_id = "22222222-2222-4222-8222-222222222222"
    candidate_hash = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()
    runtime_hash = hashlib.sha256(
        f"approved_axis:{axis_id}:1".encode("utf-8")
    ).hexdigest()
    trait = {
        "dimension": "value",
        "comparison_key_sha256": "e" * 64,
        "confirmed_candidate_id_sha256": candidate_hash,
        "axis_bound": True,
        "axis_id_sha256": hashlib.sha256(axis_id.encode("utf-8")).hexdigest(),
        "axis_version": 1,
        "approved_axis_runtime_comparison_key_sha256": runtime_hash,
        "axis_alignment": "same",
        "axis_positive_proposition_sha256": "a" * 64,
        "axis_applicability_scope_sha256": "b" * 64,
        "axis_creation_status": "created",
        "axis_decision_status": "confirmed_bound",
    }
    trait[field] = replacement
    trace = [{
        "character_key": "唐岫",
        "dimension": "value",
        "comparison_key_sha256": runtime_hash,
        "confirmed_candidate_id_sha256": candidate_hash,
    }]

    assert live_acceptance._runtime_identity(
        {
            "case_id": "tangxiu_value_shift",
            "character_key": "唐岫",
            "dimension": "value",
        },
        {"tangxiu_value_shift": trait},
        trace,
    ) is None


@pytest.mark.parametrize(
    ("axis_mode", "failure_code"),
    (
        ("http_failure", "http_request_failed"),
        ("bad_hash", "author_axis_response_invalid"),
    ),
)
def test_author_axis_creation_or_snapshot_failure_sends_no_decisions(
    monkeypatch, axis_mode, failure_code,
):
    fixture = _author_axis_review_http_mock(axis_mode=axis_mode)
    monkeypatch.setattr(live_acceptance, "_request", fixture["fake_request"])

    with pytest.raises(AcceptanceFailure) as captured:
        _review_explicit_candidates(
            object(),
            "project-id",
            [fixture["selector"]],
            author_axes=[fixture["plan"]],
        )

    assert captured.value.safe_payload["code"] == failure_code
    assert [event[0] for event in fixture["events"]] == ["axis"]


def test_author_axis_rejects_candidate_comparison_key_mismatch_before_post(monkeypatch):
    fixture = _author_axis_review_http_mock(
        comparison_key="behavior_boundary:人货取舍"
    )
    monkeypatch.setattr(live_acceptance, "_request", fixture["fake_request"])

    with pytest.raises(AcceptanceFailure) as captured:
        _review_explicit_candidates(
            object(),
            "project-id",
            [fixture["selector"]],
            author_axes=[fixture["plan"]],
        )

    assert captured.value.safe_payload["code"] == (
        "author_axis_candidate_comparison_key_invalid"
    )
    assert fixture["events"] == []


def test_legacy_candidate_review_without_author_axes_still_works(monkeypatch):
    fixture = _author_axis_review_http_mock()
    monkeypatch.setattr(live_acceptance, "_request", fixture["fake_request"])

    review = _review_explicit_candidates(
        object(), "project-id", [fixture["selector"]]
    )

    assert not any(event[0] == "axis" for event in fixture["events"])
    assert review["author_axis_plan_count"] == 0
    assert review["author_axes_created"] == 0
    assert review["author_axis_operation_status"] == "not_requested"
    trait = review["case_traits"]["tangxiu_value_shift"]
    assert trait["axis_bound"] is False
    assert trait["axis_creation_status"] == "not_requested"
    assert trait["axis_decision_status"] == "confirmed_unbound"


def test_output_json_is_confined_to_artifacts():
    resolved = _resolve_output_json("character-live-report.json")
    assert resolved.parent.name == "artifacts"
    assert resolved.name == "character-live-report.json"
    with pytest.raises(ValueError, match="inside the artifacts directory"):
        _resolve_output_json("../outside.json")


def test_selector_failure_is_structured_and_contains_no_candidate_payload(monkeypatch):
    def fake_request(_client, method, path, **_kwargs):
        if path.endswith("/characters"):
            return {"items": [{"character_key": "祁雾"}]}
        if "/profile-candidates" in path and method == "GET":
            return {"items": []}
        raise AssertionError("selector failure must occur before a decision POST")

    monkeypatch.setattr(live_acceptance, "_request", fake_request)
    with pytest.raises(AcceptanceFailure) as captured:
        _review_explicit_candidates(
            object(),
            "project-id",
            [
                {
                    "case_id": "qi_disguise_speech",
                    "character_key": "祁雾",
                    "trait_type": "speech_pattern",
                    "source_document": "02-character-profiles.md",
                    "source_line": 17,
                }
            ],
        )
    assert captured.value.safe_payload == {
        "code": "oracle_selector_match_count",
        "stage": "baseline_candidate_review",
        "details": {
            "case_id": "qi_disguise_speech",
            "matched_candidates": 0,
            "expected_candidates": 1,
        },
    }


def test_unexpected_failure_report_never_copies_exception_message():
    secret = "sk-secret https://private.example/v1 original story body"
    failure = _safe_unexpected_failure(RuntimeError(secret))
    report = _failure_report(failure)
    assert report["failure"] == {
        "code": "runner_runtime_error",
        "stage": "runner",
        "details": {"exception_type": "RuntimeError"},
    }
    assert secret not in repr(report)
    assert "private.example" not in repr(report)
