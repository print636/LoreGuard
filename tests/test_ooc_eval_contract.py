"""Synthetic-only tests for the sealed OOC evaluation contract.

No repository fixture, evaluation data directory, or holdout is read here.
"""

from __future__ import annotations

import copy
import json

import pytest
from pydantic import ValidationError

from scripts import ooc_eval_contract as contract


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _parse(model, payload):
    return model.model_validate_json(json.dumps(payload, ensure_ascii=False))


def _document(document_id: str, path: str, sha256: str) -> dict:
    return {
        "document_id": document_id,
        "path": path,
        "version": 1,
        "sha256": sha256,
    }


def _case(
    case_id: str,
    group_id: str,
    world_id: str,
    document_id: str,
    document_path: str,
    document_sha: str,
    axis_id: str,
    evidence_ids: tuple[str, ...],
) -> dict:
    return {
        "case_id": case_id,
        "split_id": "sealed-eval",
        "group_id": group_id,
        "world_id": world_id,
        "documents": [
            _document(document_id, document_path, document_sha),
        ],
        "axis": {
            "axis_id": axis_id,
            "version": 1,
            "sha256": document_sha,
        },
        "evidence_catalog": [
            {
                "evidence_id": evidence_id,
                "document_id": document_id,
                "line_start": index,
                "line_end": index,
            }
            for index, evidence_id in enumerate(evidence_ids, start=1)
        ],
    }


def _public_payload() -> dict:
    return {
        "schema_version": contract.PUBLIC_SCHEMA_VERSION,
        "dataset_id": "数据集/Ω — opaque",
        "cases": [
            _case(
                "正例 #1", "group-positive", "world-positive", "doc + positive",
                "documents/positive.md", SHA_A, "axis:positive", ("base B", "current C", "current C2"),
            ),
            _case(
                "难负例 #1", "group-negative", "world-negative", "doc + negative",
                "documents/negative.md", SHA_B, "axis:negative", ("tempting baseline", "tempting but forbidden", "tempting current two"),
            ),
            _case(
                "待定 #1", "group-indeterminate", "world-indeterminate", "doc + pending",
                "documents/pending.md", SHA_C, "axis:pending", ("provenance P",),
            ),
        ],
    }


def _reviewers(outcome: str, surface: str) -> list[dict]:
    return [
        {"reviewer_id": "reviewer α", "outcome": outcome, "surface": surface},
        {"reviewer_id": "reviewer β", "outcome": outcome, "surface": surface},
    ]


def _required(**updates: list[str]) -> dict:
    result = {role: [] for role in ("B", "C", "G", "X", "P")}
    result.update(updates)
    return result


def _gold_payload(public: contract.PublicInputBundle) -> dict:
    positive_outcome = "conflict"
    positive_surface = "formal_issue"
    negative_outcome = "no_issue"
    negative_surface = "none"
    pending_outcome = "indeterminate"
    pending_surface = "review_clue"
    return {
        "schema_version": contract.PRIVATE_GOLD_SCHEMA_VERSION,
        "dataset_id": public.dataset_id,
        "public_input_sha256": contract.canonical_sha256(public),
        "cases": [
            {
                "case_id": "正例 #1",
                "outcome": positive_outcome,
                "surface": positive_surface,
                "required_evidence": _required(B=["base B"], C=["current C", "current C2"]),
                "forbidden_evidence_ids": [],
                "independent_event_groups": [
                    {"event_group_id": "event one", "evidence_ids": ["current C"]},
                    {"event_group_id": "event two", "evidence_ids": ["current C2"]},
                ],
                "phenomena": ["temporal-boundary"],
                "adjudication": {
                    "adjudicated": True,
                    "reviewers": _reviewers(positive_outcome, positive_surface),
                    "final_outcome": positive_outcome,
                    "final_surface": positive_surface,
                },
            },
            {
                "case_id": "难负例 #1",
                "outcome": negative_outcome,
                "surface": negative_surface,
                "required_evidence": _required(),
                "forbidden_evidence_ids": ["tempting but forbidden"],
                "independent_event_groups": [],
                "phenomena": ["hard-negative"],
                "adjudication": {
                    "adjudicated": True,
                    "reviewers": _reviewers(negative_outcome, negative_surface),
                    "final_outcome": negative_outcome,
                    "final_surface": negative_surface,
                },
            },
            {
                "case_id": "待定 #1",
                "outcome": pending_outcome,
                "surface": pending_surface,
                "required_evidence": _required(P=["provenance P"]),
                "forbidden_evidence_ids": [],
                "independent_event_groups": [
                    {"event_group_id": "provenance event", "evidence_ids": ["provenance P"]},
                ],
                "phenomena": ["review-boundary"],
                "adjudication": {
                    "adjudicated": True,
                    "reviewers": _reviewers(pending_outcome, pending_surface),
                    "final_outcome": pending_outcome,
                    "final_surface": pending_surface,
                },
            },
        ],
    }


def _fixtures() -> tuple[contract.PublicInputBundle, contract.PrivateGoldBundle]:
    public = _parse(contract.PublicInputBundle, _public_payload())
    gold = _parse(contract.PrivateGoldBundle, _gold_payload(public))
    contract.validate_evaluation_contract(public, gold)
    return public, gold


def _funnel(*, surfaced: int) -> dict:
    return {
        "input_candidates": 1,
        "retrieved_candidates": 1,
        "grounded_candidates": 1,
        "adjudicated_candidates": 1,
        "surfaced_candidates": surfaced,
    }


def _citation(
    public: contract.PublicInputBundle,
    case_id: str,
    evidence_id: str,
    role: str,
    *,
    snapshot_sha256: str | None = None,
) -> dict:
    case = next(row for row in public.cases if row.case_id == case_id)
    anchor = next(
        row for row in case.evidence_catalog if row.evidence_id == evidence_id
    )
    document = next(
        row for row in case.documents if row.document_id == anchor.document_id
    )
    return {
        "evidence_id": evidence_id,
        "role": role,
        "document_id": document.document_id,
        "snapshot_version": document.version,
        "snapshot_sha256": snapshot_sha256 or document.sha256,
        "line_start": anchor.line_start,
        "line_end": anchor.line_end,
    }


def _perfect_predictions(public: contract.PublicInputBundle) -> list[dict]:
    return [
        {
            "case_id": "正例 #1",
            "outcome": "conflict",
            "surface": "formal_issue",
            "citations": [
                _citation(public, "正例 #1", "base B", "B"),
                _citation(public, "正例 #1", "current C", "C"),
                _citation(public, "正例 #1", "current C2", "C"),
            ],
            "stage_funnel": _funnel(surfaced=1),
        },
        {
            "case_id": "难负例 #1",
            "outcome": "no_issue",
            "surface": "none",
            "citations": [],
            "stage_funnel": _funnel(surfaced=0),
        },
        {
            "case_id": "待定 #1",
            "outcome": "indeterminate",
            "surface": "review_clue",
            "citations": [
                _citation(public, "待定 #1", "provenance P", "P"),
            ],
            "stage_funnel": _funnel(surfaced=1),
        },
    ]


def _run_payload(
    public: contract.PublicInputBundle,
    repeat_id: str,
    predictions: list[dict],
    *,
    status: str = "complete",
) -> dict:
    return {
        "schema_version": contract.PREDICTION_SCHEMA_VERSION,
        "dataset_id": public.dataset_id,
        "public_input_sha256": contract.canonical_sha256(public),
        "run_id": f"run {repeat_id}",
        "repeat_id": repeat_id,
        "run_status": status,
        "predictions": predictions,
    }


def test_opaque_ids_are_bounded_not_semantically_patterned_and_models_are_strict():
    public, _ = _fixtures()
    assert public.dataset_id == "数据集/Ω — opaque"
    assert public.cases[0].case_id == "正例 #1"

    payload = _public_payload()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        _parse(contract.PublicInputBundle, payload)

    payload = _public_payload()
    payload["cases"][0]["documents"][0]["version"] = "1"
    with pytest.raises(ValidationError):
        _parse(contract.PublicInputBundle, payload)

    payload = _public_payload()
    payload["cases"][0]["case_id"] = "x" * 201
    with pytest.raises(ValidationError):
        _parse(contract.PublicInputBundle, payload)


@pytest.mark.parametrize("mutation", ["group_crosses_split", "world_crosses_group"])
def test_cross_split_and_cross_group_leakage_is_rejected(mutation: str):
    payload = _public_payload()
    clone = copy.deepcopy(payload["cases"][0])
    clone["case_id"] = "new independent-looking case"
    clone["documents"] = [_document("new doc", "documents/new.md", "d" * 64)]
    clone["axis"] = {"axis_id": "new axis", "version": 1, "sha256": "d" * 64}
    clone["evidence_catalog"] = [
        {
            "evidence_id": "new evidence",
            "document_id": "new doc",
            "line_start": 1,
            "line_end": 1,
        }
    ]
    if mutation == "group_crosses_split":
        clone["split_id"] = "different split"
        clone["group_id"] = payload["cases"][0]["group_id"]
        clone["world_id"] = "new world"
    else:
        clone["group_id"] = "different group"
        clone["world_id"] = payload["cases"][0]["world_id"]
    payload["cases"].append(clone)
    with pytest.raises(ValidationError, match="cross_split|cross.*group"):
        _parse(contract.PublicInputBundle, payload)


@pytest.mark.parametrize("shared_bytes", ["document", "axis"])
def test_content_hash_reuse_across_groups_is_rejected_even_with_new_version(
    shared_bytes: str,
):
    payload = _public_payload()
    source = payload["cases"][0]
    clone = copy.deepcopy(source)
    clone["case_id"] = f"same {shared_bytes} bytes in another group"
    clone["split_id"] = "another split"
    clone["group_id"] = "another group"
    clone["world_id"] = "another world"
    clone["documents"][0]["document_id"] = "another document"
    clone["documents"][0]["path"] = "documents/another.md"
    clone["documents"][0]["version"] = 2
    clone["axis"]["axis_id"] = "another axis"
    clone["axis"]["version"] = 2
    clone["evidence_catalog"] = [
        {
            "evidence_id": "another evidence",
            "document_id": "another document",
            "line_start": 1,
            "line_end": 1,
        }
    ]
    if shared_bytes == "document":
        clone["axis"]["sha256"] = "f" * 64
    else:
        clone["documents"][0]["sha256"] = "f" * 64
    payload["cases"].append(clone)

    with pytest.raises(ValidationError, match="cross_split_or_group_identity_leakage"):
        _parse(contract.PublicInputBundle, payload)


def test_gold_requires_two_distinct_adjudicators_and_known_grouped_evidence():
    public, _ = _fixtures()
    payload = _gold_payload(public)
    payload["cases"][0]["adjudication"]["reviewers"][1]["reviewer_id"] = "reviewer α"
    with pytest.raises(ValidationError, match="two_distinct"):
        _parse(contract.PrivateGoldBundle, payload)

    payload = _gold_payload(public)
    payload["cases"][2]["independent_event_groups"][0]["evidence_ids"] = ["unknown"]
    gold = _parse(contract.PrivateGoldBundle, payload)
    with pytest.raises(contract.ContractError, match="not_in_public_catalog"):
        contract.validate_evaluation_contract(public, gold)


@pytest.mark.parametrize(
    "mutation, expected",
    (
        ("required_p", "formal_issue_cannot_require_P"),
        ("single_current", "formal_issue_requires_two_current"),
        ("same_event_group", "two_independent_current_events"),
        ("surface_mismatch", "formal_issue_must_match_conflict"),
    ),
)
def test_formal_gold_cannot_encode_review_only_or_single_event_cases(
    mutation, expected
):
    public, _ = _fixtures()
    payload = _gold_payload(public)
    case = payload["cases"][0]
    if mutation == "required_p":
        case["required_evidence"]["P"] = ["current C2"]
        case["required_evidence"]["C"] = ["current C"]
    elif mutation == "single_current":
        case["required_evidence"]["C"] = ["current C"]
    elif mutation == "same_event_group":
        case["independent_event_groups"] = [
            {
                "event_group_id": "one event",
                "evidence_ids": ["current C", "current C2"],
            }
        ]
    else:
        case["outcome"] = "indeterminate"
        case["adjudication"]["final_outcome"] = "indeterminate"
        for reviewer in case["adjudication"]["reviewers"]:
            reviewer["outcome"] = "indeterminate"

    with pytest.raises(ValidationError, match=expected):
        _parse(contract.PrivateGoldBundle, payload)


@pytest.mark.parametrize("mutation, expected", (
    ("required_p", "formal_prediction_cannot_cite_P"),
    ("single_current", "formal_prediction_requires_B_and_two_C"),
    ("surface_mismatch", "formal_prediction_must_match_conflict"),
))
def test_formal_prediction_shape_cannot_bypass_product_gate(mutation, expected):
    public, _ = _fixtures()
    prediction = _perfect_predictions(public)[0]
    if mutation == "required_p":
        prediction["citations"][2]["role"] = "P"
    elif mutation == "single_current":
        prediction["citations"] = prediction["citations"][:2]
    else:
        prediction["outcome"] = "indeterminate"

    with pytest.raises(ValidationError, match=expected):
        _parse(
            contract.PredictionBundle,
            _run_payload(public, "invalid formal", [prediction]),
        )


def test_separate_file_loaders_bind_files_without_reading_document_paths(tmp_path):
    public, gold = _fixtures()
    run = _parse(
        contract.PredictionBundle,
        _run_payload(public, "repeat 1", _perfect_predictions(public)),
    )
    public_path = tmp_path / "public-input.json"
    gold_path = tmp_path / "private-gold.json"
    run_path = tmp_path / "prediction.json"
    public_path.write_text(public.model_dump_json(), encoding="utf-8")
    gold_path.write_text(gold.model_dump_json(), encoding="utf-8")
    run_path.write_text(run.model_dump_json(), encoding="utf-8")

    # No documents/ directory is created.  Successful loading proves document
    # snapshots are inert metadata rather than an implicit corpus read.
    loaded_public, loaded_gold, loaded_runs = contract.load_evaluation_files(
        public_path, gold_path, [run_path]
    )
    assert loaded_public == public
    assert loaded_gold == gold
    assert loaded_runs == (run,)
    with pytest.raises(contract.ContractError, match="must_be_distinct"):
        contract.load_evaluation_files(public_path, public_path)


@pytest.mark.parametrize(
    "unsafe_path",
    (
        "C:/Users/alice/private/chapter.md",
        "C:private/chapter.md",
        "documents/C:/Users/alice/private/chapter.md",
        r"\\server\share\chapter.md",
        "//server/share/chapter.md",
        "https://example.invalid/chapter.md",
        "documents/../private/chapter.md",
    ),
)
def test_public_document_paths_must_be_portable_relative_metadata(unsafe_path):
    payload = _public_payload()
    payload["cases"][0]["documents"][0]["path"] = unsafe_path

    with pytest.raises(ValidationError, match="document_path"):
        _parse(contract.PublicInputBundle, payload)


def test_public_evidence_catalog_rejects_duplicate_source_anchor():
    payload = _public_payload()
    first = payload["cases"][0]["evidence_catalog"][1]
    second = payload["cases"][0]["evidence_catalog"][2]
    second["document_id"] = first["document_id"]
    second["line_start"] = first["line_start"]
    second["line_end"] = first["line_end"]

    with pytest.raises(ValidationError, match="duplicate_evidence_anchor"):
        _parse(contract.PublicInputBundle, payload)


def test_public_case_rejects_duplicate_document_bytes_under_new_identity():
    payload = _public_payload()
    duplicate = copy.deepcopy(payload["cases"][0]["documents"][0])
    duplicate["document_id"] = "same bytes, new id"
    duplicate["path"] = "documents/copied-positive.md"
    duplicate["version"] = 2
    payload["cases"][0]["documents"].append(duplicate)

    with pytest.raises(ValidationError, match="duplicate_document_bytes"):
        _parse(contract.PublicInputBundle, payload)


def test_formal_gold_rejects_overlapping_current_evidence_ranges():
    payload = _public_payload()
    payload["cases"][0]["evidence_catalog"][2]["line_start"] = 2
    payload["cases"][0]["evidence_catalog"][2]["line_end"] = 3
    public = _parse(contract.PublicInputBundle, payload)
    gold = _parse(contract.PrivateGoldBundle, _gold_payload(public))

    with pytest.raises(
        contract.ContractError,
        match="formal_current_evidence_ranges_overlap",
    ):
        contract.validate_evaluation_contract(public, gold)


def test_formal_prediction_rejects_overlapping_current_evidence_ranges():
    public, gold = _fixtures()
    predictions = _perfect_predictions(public)
    predictions[0]["citations"][2]["line_start"] = 2
    predictions[0]["citations"][2]["line_end"] = 2
    run = _parse(
        contract.PredictionBundle,
        _run_payload(public, "overlapping prediction", predictions),
    )

    with pytest.raises(
        contract.ContractError,
        match="formal_prediction_current_evidence_ranges_overlap",
    ):
        contract.validate_evaluation_contract(public, gold, [run])


def test_evaluation_file_loader_rejects_unc_network_locations():
    with pytest.raises(contract.ContractError, match="network_locations"):
        contract.load_public_input_file(r"\\server\share\public.json")


def test_combined_loader_rejects_unc_before_resolving_any_path(monkeypatch):
    resolved: list[str] = []

    def record_resolve(path):
        resolved.append(str(path))
        return path

    monkeypatch.setattr(contract.Path, "resolve", record_resolve)
    with pytest.raises(contract.ContractError, match="network_locations"):
        contract.load_evaluation_files(
            r"\\server\share\public.json", "private-gold.json"
        )
    assert resolved == []


def test_public_binding_and_snapshot_tampering_do_not_receive_credit():
    public, gold = _fixtures()
    rebound = gold.model_copy(update={"public_input_sha256": "f" * 64})
    with pytest.raises(contract.ContractError, match="binding_mismatch"):
        contract.validate_evaluation_contract(public, rebound)

    predictions = _perfect_predictions(public)
    predictions[0]["citations"][0]["snapshot_sha256"] = "f" * 64
    run = _parse(
        contract.PredictionBundle,
        _run_payload(public, "repeat 1", predictions),
    )
    report = contract.score_evaluation(
        public,
        gold,
        [run],
        gate_policy=contract.GatePolicy(minimum_repeats=1),
    )
    assert report.metrics.invalid_citation_count == 1
    assert report.metrics.bc_completeness.value == 0.0
    positive = next(row for row in report.case_scores if row.case_id == "正例 #1")
    assert not positive.joint_correct
    assert "invalid_citation_count:above_maximum" in report.gate.failures


def test_gold_commitments_reject_bare_or_low_entropy_hashing_and_detect_tamper():
    _, gold = _fixtures()
    with pytest.raises(contract.CommitmentError, match="hmac_key_required"):
        contract.create_gold_commitment(gold)
    with pytest.raises(contract.CommitmentError, match="low_entropy"):
        contract.create_gold_commitment(gold, hmac_key=b"x" * 32)

    key = bytes(range(32))
    commitment = contract.create_gold_commitment(gold, hmac_key=key)
    assert contract.verify_gold_commitment(gold, commitment, hmac_key=key)
    tampered = gold.model_copy(update={"dataset_id": "tampered dataset"})
    assert not contract.verify_gold_commitment(tampered, commitment, hmac_key=key)

    with pytest.raises(contract.CommitmentError, match="public_salt"):
        contract.create_gold_commitment(gold, salt=bytes(range(16)))
    with pytest.raises(ValidationError):
        _parse(
            contract.GoldCommitment,
            {"scheme": "salted-sha256-v1", "digest": SHA_A, "salt_hex": "00" * 16},
        )


def test_metric_boundaries_worst_case_missing_forbidden_slices_and_gate():
    public, gold = _fixtures()
    repeat_one = _parse(
        contract.PredictionBundle,
        _run_payload(public, "repeat 1", _perfect_predictions(public)),
    )
    repeat_two_predictions = [
        {
            "case_id": "正例 #1",
            "outcome": "unassessed",
            "surface": "none",
            "citations": [],
            "stage_funnel": {
                "input_candidates": 1,
                "retrieved_candidates": 0,
                "grounded_candidates": 0,
                "adjudicated_candidates": 0,
                "surfaced_candidates": 0,
            },
        },
        {
            "case_id": "难负例 #1",
            "outcome": "conflict",
            "surface": "formal_issue",
            "citations": [
                _citation(public, "难负例 #1", "tempting baseline", "B"),
                _citation(public, "难负例 #1", "tempting but forbidden", "C"),
                _citation(public, "难负例 #1", "tempting current two", "C"),
            ],
            "stage_funnel": _funnel(surfaced=1),
        },
        _perfect_predictions(public)[2],
    ]
    repeat_two = _parse(
        contract.PredictionBundle,
        _run_payload(public, "repeat 2", repeat_two_predictions),
    )
    boundary_policy = contract.GatePolicy(
        min_formal_precision=0.5,
        min_worst_case_recall=0.0,
        min_hard_negative_specificity=0.5,
        min_bc_completeness=0.5,
        max_forbidden_reference_rate=0.2,
        min_joint_accuracy=2 / 3,
        max_sliced_formal_false_positive_rate=0.5,
        min_coverage=5 / 6,
        min_outcome_consistency=1 / 3,
        minimum_repeats=2,
        max_invalid_citation_count=0,
    )
    report = contract.score_evaluation(
        public, gold, [repeat_one, repeat_two], gate_policy=boundary_policy
    )
    metrics = report.metrics
    assert metrics.formal_precision == contract.RateMetric(
        numerator=1, denominator=2, value=0.5
    )
    assert metrics.worst_case_recall.value == 0.0
    assert metrics.hard_negative_specificity.value == 0.5
    assert metrics.bc_completeness.value == 0.5
    assert metrics.forbidden_reference_count == 1
    assert metrics.forbidden_reference_rate.value == 0.125
    assert metrics.joint_accuracy.value == pytest.approx(2 / 3)
    assert metrics.coverage.value == pytest.approx(5 / 6)
    assert metrics.outcome_consistency.value == pytest.approx(1 / 3)
    hard_negative_slice = next(
        row for row in metrics.formal_false_positive_slices
        if row.phenomenon == "hard-negative"
    )
    assert hard_negative_slice.rate == 0.5
    assert report.gate.passed, report.gate.failures

    just_over_boundary = boundary_policy.model_copy(
        update={"min_formal_precision": 0.500001}
    )
    failed = contract.score_evaluation(
        public, gold, [repeat_one, repeat_two], gate_policy=just_over_boundary
    )
    assert not failed.gate.passed
    assert failed.gate.failures == ("formal_precision:below_minimum",)


def test_worst_case_recall_uses_the_lowest_repeat_instead_of_pooled_average():
    public, gold = _fixtures()
    recalled = _parse(
        contract.PredictionBundle,
        _run_payload(public, "recall 1", _perfect_predictions(public)),
    )
    missed_predictions = _perfect_predictions(public)
    missed_predictions[0] = {
        "case_id": "正例 #1",
        "outcome": "unassessed",
        "surface": "none",
        "citations": [],
        "stage_funnel": {
            "input_candidates": 1,
            "retrieved_candidates": 0,
            "grounded_candidates": 0,
            "adjudicated_candidates": 0,
            "surfaced_candidates": 0,
        },
    }
    missed = _parse(
        contract.PredictionBundle,
        _run_payload(public, "recall 0", missed_predictions),
    )

    report = contract.score_evaluation(
        public,
        gold,
        [recalled, missed],
        gate_policy=contract.GatePolicy(minimum_repeats=2),
    )

    assert report.metrics.worst_case_recall == contract.RateMetric(
        numerator=0,
        denominator=1,
        value=0.0,
    )


def test_partial_failed_and_unassessed_never_receive_correctness_credit():
    public, gold = _fixtures()
    partial = _parse(
        contract.PredictionBundle,
        _run_payload(
            public,
            "partial repeat",
            _perfect_predictions(public),
            status="partial",
        ),
    )
    partial_report = contract.score_evaluation(
        public,
        gold,
        [partial],
        gate_policy=contract.GatePolicy(minimum_repeats=1),
    )
    assert partial_report.metrics.formal_precision.value == 0.0
    assert partial_report.metrics.worst_case_recall.value == 0.0
    assert partial_report.metrics.hard_negative_specificity.value == 0.0
    assert partial_report.metrics.coverage.value == 0.0
    assert partial_report.metrics.joint_accuracy.value == 0.0

    failed = _parse(
        contract.PredictionBundle,
        _run_payload(public, "failed repeat", [], status="failed"),
    )
    failed_report = contract.score_evaluation(
        public,
        gold,
        [failed],
        gate_policy=contract.GatePolicy(minimum_repeats=1),
    )
    assert failed_report.metrics.formal_precision.value is None
    assert failed_report.metrics.worst_case_recall.value == 0.0
    assert failed_report.metrics.hard_negative_specificity.value == 0.0
    assert failed_report.metrics.coverage.value == 0.0


def test_stage_funnel_must_narrow_and_prediction_repeat_ids_are_unique():
    public, gold = _fixtures()
    predictions = _perfect_predictions(public)
    predictions[0]["stage_funnel"]["retrieved_candidates"] = 2
    with pytest.raises(ValidationError, match="monotonic"):
        _parse(
            contract.PredictionBundle,
            _run_payload(public, "repeat 1", predictions),
        )

    first = _parse(
        contract.PredictionBundle,
        _run_payload(public, "same repeat", _perfect_predictions(public)),
    )
    second_payload = _run_payload(
        public, "same repeat", _perfect_predictions(public)
    )
    second_payload["run_id"] = "different run"
    second = _parse(contract.PredictionBundle, second_payload)
    with pytest.raises(contract.ContractError, match="duplicate_repeat_id"):
        contract.validate_evaluation_contract(public, gold, [first, second])
