import hashlib
from copy import deepcopy

import pytest

import scripts.run_character_consistency_live as live_acceptance
from scripts.run_character_consistency_live import (
    AcceptanceFailure,
    _baseline_admission,
    _comparison_identity,
    _evaluate_gates,
    _evaluate_oracle_trial,
    _execute_trial,
    _failure_report,
    _load_oracle,
    _matches_selector,
    _resolve_output_json,
    _review_explicit_candidates,
    _safe_unexpected_failure,
    _safe_case_trace_summary,
    _safe_trace_observation_refs,
    _safe_visible_issue,
    _validate_oracle_payload,
)


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
        if row["visible"]:
            issue_contract = row["visible_issue"]
            evidence_selectors = list(issue_contract["required_evidence"])
            required_any_minimum = issue_contract.get(
                "required_any_evidence_min_matches",
                1 if issue_contract["required_any_evidence"] else 0,
            )
            evidence_selectors.extend(
                issue_contract["required_any_evidence"][:required_any_minimum]
            )
            visible.append(
                {
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
            )
    return expected, case_traits, trace, visible


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
    expected, case_traits, trace, visible = _oracle_trial_fixture()
    preference_index = next(
        index
        for index, row in enumerate(expected)
        if row["case_id"] == "lin_melon_preference"
    )
    trial = {"case_trace": trace, "visible_issue_cases": visible}
    assert _evaluate_oracle_trial(trial, expected, case_traits)[
        "all_cases_passed"
    ] is True

    reported = deepcopy(trace)
    reported[preference_index]["prepare_reason"] = "reported_opposed_preference"
    assert _evaluate_oracle_trial(
        {"case_trace": reported, "visible_issue_cases": visible},
        expected,
        case_traits,
    )["all_cases_passed"] is True

    unrelated = deepcopy(trace)
    unrelated[preference_index]["prepare_reason"] = "two_independent_behaviors"
    assert _evaluate_oracle_trial(
        {"case_trace": unrelated, "visible_issue_cases": visible},
        expected,
        case_traits,
    )["all_cases_passed"] is False


def test_oracle_trial_binds_runtime_trait_and_full_visible_issue_signature():
    expected, case_traits, trace, visible = _oracle_trial_fixture()
    trial = {"case_trace": trace, "visible_issue_cases": visible}
    result = _evaluate_oracle_trial(trial, expected, case_traits)
    assert result["all_cases_passed"] is True
    assert result["trace_has_no_unexpected_cases"] is True
    assert result["visible_issues_exact"] is True

    wrong_trait = deepcopy(trace)
    wrong_trait[0]["comparison_key_sha256"] = "0" * 64
    failed_trait = _evaluate_oracle_trial(
        {"case_trace": wrong_trait, "visible_issue_cases": visible},
        expected,
        case_traits,
    )
    assert failed_trait["all_cases_passed"] is False
    assert failed_trait["trace_has_no_unexpected_cases"] is False

    wrong_subtype = deepcopy(visible)
    wrong_subtype[0]["subtype"] = "wrong_subtype"
    failed_subtype = _evaluate_oracle_trial(
        {"case_trace": trace, "visible_issue_cases": wrong_subtype},
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
        {"case_trace": trace, "visible_issue_cases": wrong_evidence},
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
            "visible_issue_cases": insufficient_distinct_core_evidence,
        },
        expected,
        case_traits,
    )
    assert insufficient["visible_issues_exact"] is False

    extra_unallowed_evidence = deepcopy(visible)
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
        {"case_trace": trace, "visible_issue_cases": extra_unallowed_evidence},
        expected,
        case_traits,
    )
    assert unallowed["visible_issues_exact"] is False


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
        assert path.endswith("/drift-issues")
        return {"items": [{
            "run_id": "run-1",
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
        }]}

    monkeypatch.setattr(live_acceptance, "_request", fake_request)
    report = live_acceptance._run_summary(None, "project-1", {"id": "run-1"})
    issue = report["visible_issue_cases"][0]
    assert issue["comparison_key_sha256"] == hashlib.sha256(
        frozen_key.encode("utf-8")
    ).hexdigest()
    assert frozen_key not in repr(report)
    assert candidate_id not in repr(report)
    assert "机密草稿原文" not in repr(report)


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
    *, issues=("stable-signature",), visible_issue_cases=None
):
    if visible_issue_cases is None:
        visible_issue_cases = [_semantic_visible_issue()]
    return {
        "status": "completed",
        "stage_outcome": "completed",
        "material_coverage": "complete",
        "planned_chunks": 4,
        "processed_chunks": 4,
        "stage_usage": {"attempted_calls": 4, "input_tokens": 100},
        "issues": list(issues),
        "visible_issue_cases": visible_issue_cases,
    }


def _passing_trial(index: int):
    return {
        "project_id": f"project-{index}",
        "baseline": _complete_stage(issues=(), visible_issue_cases=[]),
        "draft": _complete_stage(),
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
        },
    }


def test_gates_require_three_independent_full_workflows_and_exact_candidates():
    passing = [_passing_trial(index) for index in range(3)]
    gates = _evaluate_gates(passing)
    assert all(gates.values())

    one_trial = _evaluate_gates(passing[:1])
    assert one_trial["at_least_three_independent_full_workflow_trials"] is False
    assert (
        one_trial[
            "stable_visible_issue_semantic_identity_across_independent_trials"
        ]
        is False
    )

    repeated_project = deepcopy(passing)
    repeated_project[2]["project_id"] = repeated_project[1]["project_id"]
    repeated = _evaluate_gates(repeated_project)
    assert repeated["at_least_three_independent_full_workflow_trials"] is False

    extra_candidate = deepcopy(passing)
    extra_candidate[0]["candidate_review"]["rejected"] = 1
    extra_candidate[0]["candidate_review"][
        "explicit_reviewable_candidate_count"
    ] = 6
    extras = _evaluate_gates(extra_candidate)
    assert extras["baseline_explicit_candidate_set_matches_oracle_exactly"] is False

    incomplete_baseline = deepcopy(passing)
    incomplete_baseline[0]["baseline"]["processed_chunks"] = 3
    incomplete = _evaluate_gates(incomplete_baseline)
    assert (
        incomplete["baseline_all_planned_chunks_processed_without_degradation"]
        is False
    )


def test_stability_gate_uses_semantic_issue_multiset_not_evidence_serialization():
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
    gates = _evaluate_gates(passing)
    assert gates[
        "stable_visible_issue_semantic_identity_across_independent_trials"
    ] is True

    changed_semantics = deepcopy(passing)
    changed_semantics[2]["draft"]["visible_issue_cases"][0][
        "judgement"
    ] = "explained"
    changed_gates = _evaluate_gates(changed_semantics)
    assert changed_gates[
        "stable_visible_issue_semantic_identity_across_independent_trials"
    ] is False

    changed_multiplicity = deepcopy(passing)
    changed_multiplicity[2]["draft"]["visible_issue_cases"].append(
        deepcopy(changed_multiplicity[2]["draft"]["visible_issue_cases"][0])
    )
    multiplicity_gates = _evaluate_gates(changed_multiplicity)
    assert multiplicity_gates[
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
