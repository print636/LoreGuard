"""Synthetic tests for the private two-person OOC annotation workflow."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest
from pydantic import ValidationError

from scripts import ooc_annotation_workflow as workflow
from scripts import ooc_closed_bundle as closed_bundle
from scripts import ooc_eval_contract as contract
from scripts import ooc_sealed_prediction as sealed_prediction
from scripts import run_ooc_sealed_http as sealed_http


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _resign_closed_commitment(
    gold: contract.PrivateGoldBundle,
    commitment: workflow.ClosedGoldCommitment,
    key: bytes,
) -> workflow.ClosedGoldCommitment:
    """Test helper for an authentic but non-registered commitment policy."""

    metadata = workflow._closed_commitment_metadata_bytes(
        public_input_sha256=commitment.public_input_sha256,
        author_setup_sha256=commitment.author_setup_sha256,
        execution_freeze_sha256=commitment.execution_freeze_sha256,
        run_config_sha256=commitment.run_config_sha256,
        annotation_sha256=commitment.annotation_sha256,
        adjudication_sha256=commitment.adjudication_sha256,
        manual_version=commitment.manual_version,
        gate_policy_id=commitment.gate_policy_id,
        gate_policy=commitment.gate_policy,
        scorer_manifest=commitment.scorer_manifest,
    )
    payload = (
        workflow.CLOSED_COMMITMENT_DOMAIN
        + contract.canonical_json_bytes(gold)
        + b"\x00"
        + metadata
    )
    return commitment.model_copy(
        update={"digest": hmac.new(key, payload, hashlib.sha256).hexdigest()}
    )


def _setup_bound_fixture():
    baseline_id = "baseline-document"
    target_id = "target-document"
    public = contract.PublicInputBundle(
        dataset_id="setup-bound-dataset",
        cases=(
            contract.PublicInputCase(
                case_id="setup-bound-case",
                split_id="closed-holdout-v1",
                group_id="setup-bound-group",
                world_id="setup-bound-world",
                documents=(
                    contract.DocumentSnapshot(
                        document_id=baseline_id,
                        path="cases/setup-bound-group/baseline.md",
                        version=1,
                        sha256=SHA_A,
                    ),
                    contract.DocumentSnapshot(
                        document_id=target_id,
                        path="cases/setup-bound-group/target.md",
                        version=1,
                        sha256=SHA_B,
                    ),
                ),
                axis=contract.AxisSnapshot(
                    axis_id="setup-bound-axis", version=1, sha256=SHA_C
                ),
                evidence_catalog=(
                    contract.EvidenceAnchor(
                        evidence_id="baseline-selected",
                        document_id=baseline_id,
                        line_start=1,
                        line_end=1,
                    ),
                    contract.EvidenceAnchor(
                        evidence_id="baseline-other",
                        document_id=baseline_id,
                        line_start=2,
                        line_end=2,
                    ),
                    contract.EvidenceAnchor(
                        evidence_id="current-one",
                        document_id=target_id,
                        line_start=1,
                        line_end=1,
                    ),
                    contract.EvidenceAnchor(
                        evidence_id="current-two",
                        document_id=target_id,
                        line_start=2,
                        line_end=2,
                    ),
                ),
            ),
        ),
    )
    setup = closed_bundle.PublicAuthorSetupBundle(
        dataset_id=public.dataset_id,
        public_input_sha256=contract.canonical_sha256(public),
        cases=(
            closed_bundle.PublicAuthorSetupCase(
                case_id="setup-bound-case",
                group_id="setup-bound-group",
                target_document_id=target_id,
                documents=(
                    closed_bundle.PublicDocumentSetup(
                        document_id=baseline_id,
                        phase="baseline",
                        document_role="character_profile",
                        story_scope="main",
                        resolution_state="confirmed",
                        publication_status="published",
                        import_order=1,
                    ),
                    closed_bundle.PublicDocumentSetup(
                        document_id=target_id,
                        phase="target",
                        document_role="chapter",
                        story_scope="main",
                        resolution_state="draft",
                        publication_status="draft",
                        import_order=2,
                    ),
                ),
                candidate_selector=closed_bundle.PublicCandidateSelector(
                    character_key="林澈",
                    trait_type="core_personality",
                    source_document_id=baseline_id,
                    source_line_start=1,
                    source_line_end=1,
                    origin="explicit_setting",
                    polarity="positive",
                    stability="core",
                ),
                author_axis=None,
            ),
        ),
    )
    annotation = workflow.ReviewerAnnotationBundle(
        dataset_id=public.dataset_id,
        public_input_sha256=contract.canonical_sha256(public),
        reviewer_kind="human",
        reviewer_id="setup-reviewer",
        manual_version="ooc-manual-v1",
        blinded_to_peer=True,
        blinded_to_system_prediction=True,
        independence_declaration=True,
        cases=(
            workflow.ReviewerCaseAnnotation(
                case_id="setup-bound-case",
                dimension="core_trait",
                outcome="conflict",
                surface="formal_issue",
                conflict_level="L3",
                material_coverage="complete",
                evidence=(
                    workflow.EvidenceAssignment(
                        evidence_id="baseline-selected", role="B"
                    ),
                    workflow.EvidenceAssignment(
                        evidence_id="current-one", role="C"
                    ),
                    workflow.EvidenceAssignment(
                        evidence_id="current-two", role="C"
                    ),
                ),
                independent_event_groups=(
                    contract.IndependentEventGroup(
                        event_group_id="event-one", evidence_ids=("current-one",)
                    ),
                    contract.IndependentEventGroup(
                        event_group_id="event-two", evidence_ids=("current-two",)
                    ),
                ),
                phenomena=("core_personality",),
                reason_codes=("two_independent_oppositions",),
                confidence=5,
                review_note="两个独立事件都与冻结角色性格设定相反。",
            ),
        ),
    )
    return public, setup, annotation


def _public() -> contract.PublicInputBundle:
    def case(case_id: str, group: str, sha: str, evidence: tuple[str, ...]):
        return {
            "case_id": case_id,
            "split_id": "closed-v1",
            "group_id": group,
            "world_id": f"world-{group}",
            "documents": [
                {
                    "document_id": f"doc-{group}",
                    "path": f"cases/{group}/story.md",
                    "version": 1,
                    "sha256": sha,
                }
            ],
            "axis": {"axis_id": f"axis-{group}", "version": 1, "sha256": sha},
            "evidence_catalog": [
                {
                    "evidence_id": value,
                    "document_id": f"doc-{group}",
                    "line_start": index,
                    "line_end": index,
                }
                for index, value in enumerate(evidence, start=1)
            ],
        }

    return contract.PublicInputBundle.model_validate_json(
        json.dumps({
            "dataset_id": "closed synthetic test",
            "cases": [
                case("case-conflict", "a", SHA_A, ("a-B", "a-C1", "a-C2", "a-D")),
                case("case-clean", "b", SHA_B, ("b-B", "b-C", "b-G", "b-D")),
                case("case-review", "c", SHA_C, ("c-B", "c-C", "c-P", "c-D")),
            ],
        })
    )


def _evidence(*values: tuple[str, str]) -> list[dict[str, str]]:
    return [{"evidence_id": evidence_id, "role": role} for evidence_id, role in values]


def _case_rows(*, changed_review: bool = False) -> list[dict]:
    return [
        {
            "case_id": "case-conflict",
            "dimension": "core_trait",
            "outcome": "conflict",
            "surface": "formal_issue",
            "conflict_level": "L3",
            "material_coverage": "complete",
            "evidence": _evidence(("a-B", "B"), ("a-C1", "C"), ("a-C2", "C")),
            "independent_event_groups": [
                {"event_group_id": "event-a-1", "evidence_ids": ["a-C1"]},
                {"event_group_id": "event-a-2", "evidence_ids": ["a-C2"]},
            ],
            "explanation_links": [],
            "phenomena": ["core_personality"],
            "reason_codes": ["two_independent_oppositions"],
            "confidence": 5,
            "review_note": "角色在两个独立事件中的行为均直接违背稳定性格设定。",
        },
        {
            "case_id": "case-clean",
            "dimension": "core_trait",
            "outcome": "no_issue",
            "surface": "none",
            "conflict_level": "L0",
            "material_coverage": "complete",
            "evidence": _evidence(("b-B", "B"), ("b-C", "C"), ("b-G", "G")),
            "independent_event_groups": [],
            "explanation_links": [
                {
                    "support_evidence_id": "b-G",
                    "applicable_current_ids": ["b-C"],
                    "causal_relation": "explicit",
                    "temporal_relation": "before",
                }
            ],
            "phenomena": ["published_growth"],
            "reason_codes": ["explained_by_published_growth"],
            "confidence": 4,
            "review_note": "已发布的成长事件能够解释新稿中与旧设定不同的行为。",
        },
        {
            "case_id": "case-review",
            "dimension": "stable_preference",
            "outcome": "no_issue" if changed_review else "indeterminate",
            "surface": "none" if changed_review else "review_clue",
            "conflict_level": "L0" if changed_review else "L2",
            "material_coverage": "complete",
            "evidence": _evidence(("c-B", "B"), ("c-C", "C"), ("c-P", "P")),
            "independent_event_groups": [],
            "explanation_links": [
                {
                    "support_evidence_id": "c-P",
                    "applicable_current_ids": ["c-C"],
                    "causal_relation": "ambiguous",
                    "temporal_relation": "unknown",
                }
            ],
            "phenomena": ["single_reversal"],
            "reason_codes": ["single_opposition_only"],
            "confidence": 3,
            "review_note": "目前只有一次反向行为，保留为待复核线索而不形成正式问题。",
        },
    ]


def _annotation(
    public: contract.PublicInputBundle,
    reviewer: str,
    *,
    changed_review: bool = False,
) -> workflow.ReviewerAnnotationBundle:
    return workflow.ReviewerAnnotationBundle.model_validate_json(
        json.dumps({
            "dataset_id": public.dataset_id,
            "public_input_sha256": contract.canonical_sha256(public),
            "reviewer_kind": "human",
            "reviewer_id": reviewer,
            "manual_version": "ooc-manual-v1",
            "blinded_to_peer": True,
            "blinded_to_system_prediction": True,
            "independence_declaration": True,
            "cases": _case_rows(changed_review=changed_review),
        })
    )


def _adjudication(
    public: contract.PublicInputBundle,
    left: workflow.ReviewerAnnotationBundle,
    right: workflow.ReviewerAnnotationBundle,
) -> workflow.AdjudicationBundle:
    rows = _case_rows()
    cases = []
    distractors = {
        "case-conflict": "a-D",
        "case-clean": "b-D",
        "case-review": "c-D",
    }
    for row in rows:
        cases.append(
            {
                "case_id": row["case_id"],
                "dimension": row["dimension"],
                "final_outcome": row["outcome"],
                "final_surface": row["surface"],
                "conflict_level": row["conflict_level"],
                "material_coverage": row["material_coverage"],
                "required_evidence": row["evidence"],
                "forbidden_evidence_ids": [distractors[row["case_id"]]],
                "independent_event_groups": row["independent_event_groups"],
                "explanation_links": row["explanation_links"],
                "phenomena": row["phenomena"],
                "reason_codes": row["reason_codes"],
                "resolution_mode": "reviewer_consensus",
                "adjudicator_id": None,
                "resolution_note": "两位标注者复核证据后共同确认该最终结论。",
            }
        )
    return workflow.AdjudicationBundle.model_validate_json(
        json.dumps({
            "dataset_id": public.dataset_id,
            "public_input_sha256": contract.canonical_sha256(public),
            "reviewer_ids": [left.reviewer_id, right.reviewer_id],
            "annotation_sha256": [
                contract.canonical_sha256(left),
                contract.canonical_sha256(right),
            ],
            "consensus_confirmed_by": [right.reviewer_id, left.reviewer_id],
            "cases": cases,
        })
    )


def _closed_commitment_fixture():
    public = _public()
    left = _annotation(public, "reviewer-a")
    right = _annotation(public, "reviewer-b")
    adjudication = _adjudication(public, left, right)
    gold = workflow._build_private_gold(public, left, right, adjudication)
    key = bytes(range(32))
    commitment = workflow._create_closed_gold_commitment(
        gold,
        left,
        right,
        adjudication,
        author_setup_sha256=SHA_B,
        execution_freeze_sha256=SHA_C,
        run_config_sha256=SHA_D,
        hmac_key=key,
    )
    return public, left, right, adjudication, gold, key, commitment


def test_single_run_formal_gate_policy_is_fully_pinned():
    assert workflow.SINGLE_RUN_FORMAL_GATE_POLICY.model_dump() == {
        "min_formal_precision": 1.0,
        "min_worst_case_recall": 1.0,
        "min_hard_negative_specificity": 1.0,
        "min_bc_completeness": 1.0,
        "max_forbidden_reference_rate": 0.0,
        "min_joint_accuracy": 1.0,
        "max_sliced_formal_false_positive_rate": 0.0,
        "min_coverage": 1.0,
        "min_outcome_consistency": None,
        "minimum_repeats": 1,
        "max_invalid_citation_count": 0,
    }


def test_closed_commitment_hash_covers_policy_and_scorer_sources():
    _, _, _, _, gold, key, commitment = _closed_commitment_fixture()
    assert commitment.schema_version == "loreguard-ooc-closed-commitment-v2"
    assert workflow.CLOSED_COMMITMENT_DOMAIN.endswith(b"/v2\x00")
    assert commitment.scorer_manifest == workflow._current_scorer_manifest()
    assert workflow._verify_closed_gold_commitment(gold, commitment, hmac_key=key)

    legacy = commitment.model_dump(mode="json")
    legacy["schema_version"] = "loreguard-ooc-closed-commitment-v1"
    with pytest.raises(ValidationError):
        workflow.ClosedGoldCommitment.model_validate(legacy)

    tampered_policy = commitment.model_copy(
        update={
            "gate_policy": commitment.gate_policy.model_copy(
                update={"minimum_repeats": 2}
            )
        }
    )
    tampered_scorer_manifest = commitment.model_copy(
        update={
            "scorer_manifest": commitment.scorer_manifest.model_copy(
                update={"scorer_source_sha256": "0" * 64}
            )
        }
    )
    tampered_contract_manifest = commitment.model_copy(
        update={
            "scorer_manifest": commitment.scorer_manifest.model_copy(
                update={"eval_contract_source_sha256": "0" * 64}
            )
        }
    )
    for tampered in (
        tampered_policy,
        tampered_scorer_manifest,
        tampered_contract_manifest,
    ):
        assert not workflow._verify_closed_gold_commitment(
            gold, tampered, hmac_key=key
        )


def test_unregistered_policy_and_dependency_drift_fail_before_scoring(monkeypatch):
    public, _, _, _, gold, key, commitment = _closed_commitment_fixture()
    tampered_policy = commitment.model_copy(
        update={
            "gate_policy": commitment.gate_policy.model_copy(
                update={"minimum_repeats": 2}
            )
        }
    )
    authentic_unregistered_policy = _resign_closed_commitment(
        gold, tampered_policy, key
    )
    assert workflow._verify_closed_gold_commitment(
        gold, authentic_unregistered_policy, hmac_key=key
    )
    with pytest.raises(workflow.AnnotationWorkflowError, match="gate_policy_mismatch"):
        workflow._require_closed_commitment_for_execution(
            gold,
            authentic_unregistered_policy,
            public_input_sha256=contract.canonical_sha256(public),
            author_setup_sha256=SHA_B,
            execution_freeze_sha256=SHA_C,
            run_config_sha256=SHA_D,
            hmac_key=key,
        )

    drifted_manifest = commitment.scorer_manifest.model_copy(
        update={"eval_contract_source_sha256": "1" * 64}
    )
    monkeypatch.setattr(workflow, "_current_scorer_manifest", lambda: drifted_manifest)
    with pytest.raises(
        workflow.AnnotationWorkflowError, match="scorer_manifest_mismatch"
    ):
        workflow._require_closed_commitment_for_execution(
            gold,
            commitment,
            public_input_sha256=contract.canonical_sha256(public),
            author_setup_sha256=SHA_B,
            execution_freeze_sha256=SHA_C,
            run_config_sha256=SHA_D,
            hmac_key=key,
        )


def test_complete_independent_annotations_compare_and_build_sealed_gold():
    public = _public()
    left = _annotation(public, "reviewer-a")
    right = _annotation(public, "reviewer-b")
    workflow.validate_annotation_binding(public, left)
    workflow.validate_annotation_binding(public, right)

    agreement = workflow.compare_annotations(public, left, right)
    assert agreement.case_count == 3
    assert agreement.exact_label_agreement == 1.0
    assert agreement.outcome_kappa == 1.0
    assert agreement.surface_kappa == 1.0
    assert agreement.mean_evidence_jaccard == 1.0
    assert agreement.disagreement_case_ids == ()

    adjudication = _adjudication(public, left, right)
    gold = workflow._build_private_gold(public, left, right, adjudication)
    assert [row.outcome for row in gold.cases] == [
        "conflict",
        "no_issue",
        "indeterminate",
    ]
    assert gold.cases[0].adjudication.reviewers[0].reviewer_id == "reviewer-a"
    commitment = contract.create_gold_commitment(gold, hmac_key=bytes(range(32)))
    assert contract.verify_gold_commitment(
        gold, commitment, hmac_key=bytes(range(32))
    )


def test_disagreement_report_exposes_case_not_notes_or_gold():
    public = _public()
    left = _annotation(public, "reviewer-a")
    right = _annotation(public, "reviewer-b", changed_review=True)
    report = workflow.compare_annotations(public, left, right)
    assert report.exact_label_agreement == pytest.approx(2 / 3)
    assert report.disagreement_case_ids == ("case-review",)
    assert not report.cases[-1].outcome_agrees
    payload = report.model_dump(mode="json")
    assert "review_note" not in json.dumps(payload)
    assert "required_evidence" not in json.dumps(payload)


def test_semantic_level_and_manual_version_disagreements_are_not_hidden():
    public = _public()
    left = _annotation(public, "reviewer-a")
    payload = _annotation(public, "reviewer-b").model_dump(mode="json")
    payload["cases"][2]["conflict_level"] = "L1"
    right = workflow.ReviewerAnnotationBundle.model_validate_json(json.dumps(payload))
    report = workflow.compare_annotations(public, left, right)
    assert report.disagreement_case_ids == ("case-review",)
    assert not report.cases[-1].conflict_level_agrees

    different_manual = right.model_copy(update={"manual_version": "ooc-manual-v2"})
    with pytest.raises(workflow.AnnotationWorkflowError, match="same_manual"):
        workflow.compare_annotations(public, left, different_manual)


def test_annotation_requires_complete_public_binding_and_known_evidence():
    public = _public()
    payload = _annotation(public, "reviewer-a").model_dump(mode="json")
    payload["cases"] = payload["cases"][:-1]
    incomplete = workflow.ReviewerAnnotationBundle.model_validate_json(json.dumps(payload))
    with pytest.raises(
        workflow.AnnotationWorkflowError,
        match="cover_every_public_case",
    ):
        workflow.validate_annotation_binding(public, incomplete)

    payload = _annotation(public, "reviewer-a").model_dump(mode="json")
    payload["cases"][1]["evidence"][0]["evidence_id"] = "unknown"
    unknown = workflow.ReviewerAnnotationBundle.model_validate_json(json.dumps(payload))
    with pytest.raises(
        workflow.AnnotationWorkflowError,
        match="not_in_public_catalog",
    ):
        workflow.validate_annotation_binding(public, unknown)


def test_formal_annotation_requires_two_independent_current_events():
    public = _public()
    payload = _annotation(public, "reviewer-a").model_dump(mode="json")
    payload["cases"][0]["independent_event_groups"] = [
        {"event_group_id": "same-event", "evidence_ids": ["a-C1", "a-C2"]}
    ]
    with pytest.raises(ValidationError, match="two_independent_current_events"):
        workflow.ReviewerAnnotationBundle.model_validate_json(json.dumps(payload))


def test_formal_annotation_rejects_growth_or_exception_that_explains_current():
    public = _public()
    payload = _annotation(public, "reviewer-a").model_dump(mode="json")
    formal = payload["cases"][0]
    formal["evidence"].append({"evidence_id": "a-D", "role": "G"})
    formal["explanation_links"] = [
        {
            "support_evidence_id": "a-D",
            "applicable_current_ids": ["a-C1", "a-C2"],
            "causal_relation": "explicit",
            "temporal_relation": "before",
        }
    ]
    with pytest.raises(ValidationError, match="no_explanation"):
        workflow.ReviewerAnnotationBundle.model_validate_json(json.dumps(payload))


def test_ambiguous_growth_or_exception_cannot_prove_no_issue():
    public = _public()
    payload = _annotation(public, "reviewer-a").model_dump(mode="json")
    payload["cases"][1]["explanation_links"][0]["causal_relation"] = "ambiguous"
    with pytest.raises(ValidationError, match="classified_as_P"):
        workflow.ReviewerAnnotationBundle.model_validate_json(json.dumps(payload))


def test_build_gold_rejects_same_reviewer_and_stale_adjudication_binding():
    public = _public()
    left = _annotation(public, "reviewer-a")
    same = _annotation(public, "reviewer-a")
    with pytest.raises(workflow.AnnotationWorkflowError, match="distinct"):
        workflow.compare_annotations(public, left, same)

    right = _annotation(public, "reviewer-b")
    adjudication = _adjudication(public, left, right)
    changed = right.model_copy(update={"reviewer_id": "reviewer-c"})
    with pytest.raises(workflow.AnnotationWorkflowError, match="reviewer_order"):
        workflow._build_private_gold(public, left, changed, adjudication)


def test_adjudication_requires_both_reviewers_to_confirm():
    public = _public()
    left = _annotation(public, "reviewer-a")
    right = _annotation(public, "reviewer-b")
    payload = _adjudication(public, left, right).model_dump(mode="json")
    payload["consensus_confirmed_by"] = ["reviewer-a", "reviewer-c"]
    with pytest.raises(ValidationError, match="both_reviewers"):
        workflow.AdjudicationBundle.model_validate_json(json.dumps(payload))


def test_exhaustive_catalog_binds_bytes_and_covers_every_nonempty_line(tmp_path):
    dataset_id = "00000000-0000-4000-8000-000000000001"
    case_id = "00000000-0000-4000-8000-000000000002"
    group_id = "00000000-0000-4000-8000-000000000003"
    world_id = "00000000-0000-4000-8000-000000000004"
    document_id = "00000000-0000-4000-8000-000000000005"
    axis_id = "00000000-0000-4000-8000-000000000006"
    evidence_ids = [
        "00000000-0000-4000-8000-000000000007",
        "00000000-0000-4000-8000-000000000008",
        "00000000-0000-4000-8000-000000000009",
    ]
    case_dir = tmp_path / "cases" / group_id
    case_dir.mkdir(parents=True)
    payload = "设定行。\n\n当前行为一。\n当前行为二。\n".encode("utf-8")
    document = case_dir / f"{document_id}.md"
    document.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    public = contract.PublicInputBundle.model_validate_json(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "cases": [
                    {
                        "case_id": case_id,
                        "split_id": "closed-pilot-v1",
                        "group_id": group_id,
                        "world_id": world_id,
                        "documents": [
                            {
                                "document_id": document_id,
                                "path": f"cases/{group_id}/{document_id}.md",
                                "version": 1,
                                "sha256": digest,
                            }
                        ],
                        "axis": {
                            "axis_id": axis_id,
                            "version": 1,
                            "sha256": digest,
                        },
                        "evidence_catalog": [
                            {
                                "evidence_id": evidence_ids[0],
                                "document_id": document_id,
                                "line_start": 1,
                                "line_end": 1,
                            },
                            {
                                "evidence_id": evidence_ids[1],
                                "document_id": document_id,
                                "line_start": 3,
                                "line_end": 3,
                            },
                            {
                                "evidence_id": evidence_ids[2],
                                "document_id": document_id,
                                "line_start": 4,
                                "line_end": 4,
                            },
                        ],
                    }
                ],
            }
        )
    )
    workflow.validate_exhaustive_line_catalog(public, tmp_path)

    missing = public.model_dump(mode="json")
    missing["cases"][0]["evidence_catalog"] = missing["cases"][0][
        "evidence_catalog"
    ][:-1]
    incomplete = contract.PublicInputBundle.model_validate_json(json.dumps(missing))
    with pytest.raises(workflow.AnnotationWorkflowError, match="cover_each_nonempty"):
        workflow.validate_exhaustive_line_catalog(incomplete, tmp_path)

    leaked = public.model_dump(mode="json")
    leaked["cases"][0]["split_id"] = "conflict"
    leaked_public = contract.PublicInputBundle.model_validate_json(json.dumps(leaked))
    with pytest.raises(workflow.AnnotationWorkflowError, match="split_id"):
        workflow.validate_exhaustive_line_catalog(leaked_public, tmp_path)

    leaked = public.model_dump(mode="json")
    leaked["cases"][0]["documents"][0]["path"] = (
        f"cases/conflict/{document_id}.md"
    )
    leaked_public = contract.PublicInputBundle.model_validate_json(json.dumps(leaked))
    with pytest.raises(workflow.AnnotationWorkflowError, match="path_must_be_neutral"):
        workflow.validate_exhaustive_line_catalog(leaked_public, tmp_path)


def test_sealed_scorer_refuses_wrong_key_before_scoring(tmp_path):
    public, left, right, adjudication, gold, key, commitment = (
        _closed_commitment_fixture()
    )
    assert commitment.author_setup_sha256 == SHA_B
    assert commitment.execution_freeze_sha256 == SHA_C
    assert commitment.run_config_sha256 == SHA_D
    assert commitment.annotation_sha256 == (
        contract.canonical_sha256(left),
        contract.canonical_sha256(right),
    )
    assert commitment.adjudication_sha256 == contract.canonical_sha256(adjudication)
    assert commitment.gate_policy_id == workflow.SINGLE_RUN_FORMAL_GATE_POLICY_ID
    assert commitment.gate_policy == workflow.SINGLE_RUN_FORMAL_GATE_POLICY
    assert commitment.gate_policy.minimum_repeats == 1
    assert commitment.gate_policy.min_outcome_consistency is None
    assert commitment.scorer_manifest == workflow._current_scorer_manifest()
    assert workflow._verify_closed_gold_commitment(gold, commitment, hmac_key=key)

    def citation(case_id: str, evidence_id: str, role: str) -> dict:
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
            "snapshot_sha256": document.sha256,
            "line_start": anchor.line_start,
            "line_end": anchor.line_end,
        }

    funnel = {
        "input_candidates": 2,
        "retrieved_candidates": 2,
        "grounded_candidates": 2,
        "adjudicated_candidates": 1,
        "surfaced_candidates": 1,
    }
    prediction = contract.PredictionBundle.model_validate_json(
        json.dumps(
            {
                "dataset_id": public.dataset_id,
                "public_input_sha256": contract.canonical_sha256(public),
                "run_id": "sealed-run-01",
                "repeat_id": "repeat-01",
                "run_status": "complete",
                "predictions": [
                    {
                        "case_id": "case-conflict",
                        "outcome": "conflict",
                        "surface": "formal_issue",
                        "citations": [
                            citation("case-conflict", "a-B", "B"),
                            citation("case-conflict", "a-C1", "C"),
                            citation("case-conflict", "a-C2", "C"),
                        ],
                        "stage_funnel": funnel,
                    },
                    {
                        "case_id": "case-clean",
                        "outcome": "no_issue",
                        "surface": "none",
                        "citations": [
                            citation("case-clean", "b-B", "B"),
                            citation("case-clean", "b-C", "C"),
                            citation("case-clean", "b-G", "G"),
                        ],
                        "stage_funnel": {**funnel, "surfaced_candidates": 0},
                    },
                    {
                        "case_id": "case-review",
                        "outcome": "indeterminate",
                        "surface": "review_clue",
                        "citations": [
                            citation("case-review", "c-B", "B"),
                            citation("case-review", "c-C", "C"),
                            citation("case-review", "c-P", "P"),
                        ],
                        "stage_funnel": funnel,
                    },
                ],
            }
        )
    )
    report = workflow._score_sealed_evaluation(
        public,
        gold,
        commitment,
        [prediction],
        author_setup_sha256=SHA_B,
        execution_freeze_sha256=SHA_C,
        run_config_sha256=SHA_D,
        hmac_key=key,
    )
    assert report.metrics.joint_accuracy.value == 1.0
    assert report.metrics.repeat_count == 1
    assert report.metrics.outcome_consistency.value is None
    assert report.gate.passed
    assert "repeat_count:below_minimum" not in report.gate.failures
    with pytest.raises(workflow.AnnotationWorkflowError, match="commitment_mismatch"):
        workflow._score_sealed_evaluation(
            public,
            gold,
            commitment,
            [prediction],
            author_setup_sha256=SHA_B,
            execution_freeze_sha256=SHA_C,
            run_config_sha256=SHA_D,
            hmac_key=bytes(range(1, 33)),
        )
    tampered = commitment.model_copy(update={"adjudication_sha256": "f" * 64})
    assert not workflow._verify_closed_gold_commitment(
        gold, tampered, hmac_key=key
    )
    stale_freeze = commitment.model_copy(
        update={"execution_freeze_sha256": "f" * 64}
    )
    assert not workflow._verify_closed_gold_commitment(
        gold, stale_freeze, hmac_key=key
    )
    stale_run_config = commitment.model_copy(
        update={"run_config_sha256": "e" * 64}
    )
    assert not workflow._verify_closed_gold_commitment(
        gold, stale_run_config, hmac_key=key
    )
    with pytest.raises(workflow.AnnotationWorkflowError, match="commitment_mismatch"):
        workflow._score_sealed_evaluation(
            public,
            gold,
            commitment,
            [prediction],
            author_setup_sha256="d" * 64,
            execution_freeze_sha256=SHA_C,
            run_config_sha256=SHA_D,
            hmac_key=key,
        )

    artifact = sealed_prediction.SealedPredictionArtifact(
        dataset_id=public.dataset_id,
        public_input_sha256=contract.canonical_sha256(public),
        author_setup_sha256=SHA_B,
        execution_freeze_sha256=SHA_C,
        runtime_provenance_sha256="d" * 64,
        prediction=prediction,
    )
    assert (
        workflow._extract_bound_prediction(
            public,
            artifact,
            author_setup_sha256=SHA_B,
            execution_freeze_sha256=SHA_C,
        )
        is prediction
    )
    stale_artifact = artifact.model_copy(update={"author_setup_sha256": "e" * 64})
    with pytest.raises(workflow.AnnotationWorkflowError, match="author_setup_hash"):
        workflow._extract_bound_prediction(
            public,
            stale_artifact,
            author_setup_sha256=SHA_B,
            execution_freeze_sha256=SHA_C,
        )

    run_config = sealed_http.SealedHttpRunConfig(
        base_url="http://127.0.0.1:8101",
        isolation=sealed_http.IsolationExpectation(
            eval_db_id="11111111-1111-4111-8111-111111111111",
            database_path_sha256=SHA_A,
            instance_id_sha256=SHA_B,
        ),
        execution_id="22222222-2222-4222-8222-222222222222",
        repeat_id="repeat-01",
        expected_runtime_provenance_sha256="d" * 64,
        expected_runner_source_sha256=sealed_http.current_runner_source_sha256(),
        run_timeout_seconds=30.0,
        poll_interval_seconds=0.1,
    )
    receipts = tuple(
        sealed_http.ProjectReceipt(
            case_id=case_id,
            project_id_sha256=value * 64,
        )
        for case_id, value in zip(
            ("case-conflict", "case-clean", "case-review"),
            ("1", "2", "3"),
            strict=True,
        )
    )
    sealed_report = sealed_http.SealedHttpRunReport(
        phase="completed",
        dataset_id=public.dataset_id,
        public_input_sha256=contract.canonical_sha256(public),
        author_setup_sha256=SHA_B,
        execution_freeze_sha256=SHA_C,
        runtime_provenance_sha256="d" * 64,
        run_config_sha256=contract.canonical_sha256(run_config),
        run_config=run_config,
        runner_source_sha256=sealed_http.current_runner_source_sha256(),
        execution_id=run_config.execution_id,
        project_set_sha256=sealed_http._project_set_sha(receipts),
        project_receipts=receipts,
        independent_project_count=3,
        assessed_case_count=3,
        unassessed_case_count=0,
        prediction_artifact=artifact,
    )
    assert (
        workflow._extract_prediction_from_run_report(
            public,
            sealed_report,
            run_config,
            author_setup_sha256=SHA_B,
            execution_freeze_sha256=SHA_C,
        )
        is prediction
    )
    stale_report = sealed_report.model_copy(update={"run_config_sha256": "e" * 64})
    with pytest.raises(workflow.AnnotationWorkflowError, match="config_hash"):
        workflow._extract_prediction_from_run_report(
            public,
            stale_report,
            run_config,
            author_setup_sha256=SHA_B,
            execution_freeze_sha256=SHA_C,
        )

    raw_prediction = tmp_path / "raw-prediction.json"
    raw_prediction.write_text(prediction.model_dump_json(), encoding="utf-8")
    with pytest.raises(ValidationError):
        workflow._load_sealed_prediction_artifact(raw_prediction)


def test_annotation_dimension_and_bc_roles_are_bound_to_frozen_author_setup():
    public, setup, annotation = _setup_bound_fixture()
    workflow._validate_annotation_setup_binding(public, setup, annotation)

    wrong_dimension_case = annotation.cases[0].model_copy(
        update={"dimension": "stable_preference"}
    )
    wrong_dimension = annotation.model_copy(update={"cases": (wrong_dimension_case,)})
    with pytest.raises(workflow.AnnotationWorkflowError, match="dimension_mismatch"):
        workflow._validate_annotation_setup_binding(public, setup, wrong_dimension)

    payload = annotation.model_dump(mode="json")
    payload["cases"][0]["evidence"][0]["evidence_id"] = "baseline-other"
    wrong_baseline = workflow.ReviewerAnnotationBundle.model_validate_json(
        json.dumps(payload)
    )
    with pytest.raises(workflow.AnnotationWorkflowError, match="frozen_selector"):
        workflow._validate_annotation_setup_binding(public, setup, wrong_baseline)

    payload = annotation.model_dump(mode="json")
    payload["cases"][0]["evidence"][1]["evidence_id"] = "baseline-other"
    payload["cases"][0]["independent_event_groups"][0]["evidence_ids"] = [
        "baseline-other"
    ]
    wrong_current = workflow.ReviewerAnnotationBundle.model_validate_json(
        json.dumps(payload)
    )
    with pytest.raises(workflow.AnnotationWorkflowError, match="target_document"):
        workflow._validate_annotation_setup_binding(public, setup, wrong_current)
