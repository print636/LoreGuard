"""Tests for answer-free public OOC package generation."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import ooc_annotation_workflow as annotation
from scripts import ooc_closed_bundle as bundle
from scripts import ooc_eval_contract as contract


UUID4 = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _source_package(tmp_path):
    source = tmp_path / "source"
    source.mkdir(parents=True)
    (source / "world.md").write_text(
        "# 雾岛交通规则\n夜间渡船按照潮钟依次离港。\n",
        encoding="utf-8",
        newline="\n",
    )
    (source / "profiles.md").write_text(
        "# 角色设定\n林澈面对陌生人时一向谨慎，不会主动透露航线。\n"
        "周柚长期偏爱清淡的青梅茶。\n",
        encoding="utf-8",
        newline="\n",
    )
    (source / "history.md").write_text(
        "# 已发布历史\n林澈曾因泄露航线让同伴陷入危险。\n",
        encoding="utf-8",
        newline="\n",
    )
    (source / "draft.md").write_text(
        "# 待审新稿\n林澈在码头遇见新来的旅客。\n周柚端起一杯浓苦咖啡。\n",
        encoding="utf-8",
        newline="\n",
    )
    plan = {
        "schema_version": bundle.SOURCE_SCHEMA_VERSION,
        "split_id": "closed-pilot-v1",
        "groups": [
            {
                "group_key": "island_one",
                "documents": [
                    {
                        "document_key": "world",
                        "source_path": "world.md",
                        "phase": "baseline",
                        "document_role": "canon",
                        "story_scope": "main",
                        "resolution_state": "confirmed",
                        "publication_status": "published",
                        "import_order": 1,
                    },
                    {
                        "document_key": "profiles",
                        "source_path": "profiles.md",
                        "phase": "baseline",
                        "document_role": "character_profile",
                        "story_scope": "main",
                        "resolution_state": "confirmed",
                        "publication_status": "published",
                        "import_order": 2,
                    },
                    {
                        "document_key": "history",
                        "source_path": "history.md",
                        "phase": "baseline",
                        "document_role": "chapter",
                        "story_scope": "main",
                        "resolution_state": "confirmed",
                        "publication_status": "published",
                        "import_order": 3,
                    },
                    {
                        "document_key": "draft",
                        "source_path": "draft.md",
                        "phase": "target",
                        "document_role": "chapter",
                        "story_scope": "main",
                        "resolution_state": "draft",
                        "publication_status": "draft",
                        "import_order": 4,
                    },
                ],
                "cases": [
                    {
                        "case_key": "linche_axis",
                        "candidate_selector": {
                            "character_key": "林澈",
                            "trait_type": "core_personality",
                            "source_document_key": "profiles",
                            "source_line_start": 2,
                            "source_line_end": 2,
                            "origin": "explicit_setting",
                            "polarity": "negative",
                            "stability": "core",
                        },
                        "author_axis": None,
                    },
                    {
                        "case_key": "zhouyou_axis",
                        "candidate_selector": {
                            "character_key": "周柚",
                            "trait_type": "preference",
                            "source_document_key": "profiles",
                            "source_line_start": 3,
                            "source_line_end": 3,
                            "origin": "explicit_setting",
                            "polarity": "positive",
                            "stability": "stable",
                        },
                        "author_axis": None,
                    },
                ],
            }
        ],
    }
    plan_path = tmp_path / "source-plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    return source, plan_path, plan


def test_build_bundle_randomizes_ids_binds_setup_and_covers_all_lines(tmp_path):
    source, plan_path, _ = _source_package(tmp_path)
    output = tmp_path / "public"
    freeze = bundle.build_bundle(plan_path, source, output)

    public = contract.load_public_input_file(output / "public-input.json")
    setup = bundle.PublicAuthorSetupBundle.model_validate_json(
        (output / "author-setup.json").read_bytes()
    )
    persisted_freeze = bundle.ExecutionFreeze.model_validate_json(
        (output / "freeze.json").read_bytes()
    )
    bundle.validate_author_setup_binding(public, setup)
    bundle.validate_execution_freeze_binding(public, setup, persisted_freeze)
    annotation.validate_exhaustive_line_catalog(public, output)
    assert freeze.case_count == 2
    assert freeze.group_count == 1
    assert UUID4.fullmatch(freeze.dataset_id)
    assert setup.public_input_sha256 == contract.canonical_sha256(public)
    assert freeze.author_setup_sha256 == contract.canonical_sha256(setup)

    first, second = public.cases
    assert first.group_id == second.group_id
    assert first.world_id == second.world_id
    assert first.documents == second.documents
    for case in public.cases:
        assert UUID4.fullmatch(case.case_id)
        assert UUID4.fullmatch(case.axis.axis_id)
        assert all(UUID4.fullmatch(row.evidence_id) for row in case.evidence_catalog)
        assert all(UUID4.fullmatch(row.document_id) for row in case.documents)
        assert all(UUID4.fullmatch(row.path.rsplit("/", 1)[-1].split(".")[0]) for row in case.documents)

    serialized = (output / "public-input.json").read_text(encoding="utf-8")
    for forbidden in ("linche_axis", "zhouyou_axis", "conflict", "no_issue"):
        assert forbidden not in serialized


def test_execution_freeze_rejects_every_stale_binding_and_count(tmp_path):
    source, plan_path, _ = _source_package(tmp_path)
    output = tmp_path / "public"
    bundle.build_bundle(plan_path, source, output)
    public = contract.load_public_input_file(output / "public-input.json")
    setup = bundle.PublicAuthorSetupBundle.model_validate_json(
        (output / "author-setup.json").read_bytes()
    )
    freeze = bundle.ExecutionFreeze.model_validate_json(
        (output / "freeze.json").read_bytes()
    )

    mutations = (
        ({"dataset_id": "00000000-0000-4000-8000-000000000000"}, "dataset"),
        ({"public_input_sha256": "f" * 64}, "public_hash"),
        ({"author_setup_sha256": "e" * 64}, "author_setup_hash"),
        ({"case_count": freeze.case_count + 1}, "case_count"),
        ({"group_count": freeze.group_count + 1}, "group_count"),
    )
    for update, message in mutations:
        with pytest.raises(bundle.BundleBuildError, match=message):
            bundle.validate_execution_freeze_binding(
                public, setup, freeze.model_copy(update=update)
            )

    duplicate_case_setup = setup.model_copy(
        update={"cases": (*setup.cases, setup.cases[0])}
    )
    with pytest.raises(bundle.BundleBuildError, match="duplicate_author_setup_case"):
        bundle.validate_execution_freeze_binding(
            public, duplicate_case_setup, freeze
        )


def test_author_setup_semantics_are_bound_to_public_axis_and_lifecycle(tmp_path):
    source, plan_path, _ = _source_package(tmp_path)
    output = tmp_path / "public"
    bundle.build_bundle(plan_path, source, output)
    public = contract.load_public_input_file(output / "public-input.json")
    setup = bundle.PublicAuthorSetupBundle.model_validate_json(
        (output / "author-setup.json").read_bytes()
    )

    first = setup.cases[0]
    altered_selector = first.candidate_selector.model_copy(
        update={"character_key": "另一个角色"}
    )
    altered_case = first.model_copy(
        update={"candidate_selector": altered_selector}
    )
    altered_setup = setup.model_copy(
        update={"cases": (altered_case, *setup.cases[1:])}
    )
    with pytest.raises(bundle.BundleBuildError, match="axis_binding"):
        bundle.validate_author_setup_binding(public, altered_setup)

    documents = list(first.documents)
    baseline_index = next(
        index for index, value in enumerate(documents) if value.phase == "baseline"
    )
    documents[baseline_index] = documents[baseline_index].model_copy(
        update={"resolution_state": "draft"}
    )
    lifecycle_case = first.model_copy(update={"documents": tuple(documents)})
    lifecycle_setup = setup.model_copy(
        update={"cases": (lifecycle_case, *setup.cases[1:])}
    )
    with pytest.raises(bundle.BundleBuildError, match="lifecycle"):
        bundle.validate_author_setup_binding(public, lifecycle_setup)


def test_author_setup_selector_must_point_to_real_nonblank_catalog_lines(tmp_path):
    source, plan_path, _ = _source_package(tmp_path)
    output = tmp_path / "public"
    bundle.build_bundle(plan_path, source, output)
    public = contract.load_public_input_file(output / "public-input.json")
    setup = bundle.PublicAuthorSetupBundle.model_validate_json(
        (output / "author-setup.json").read_bytes()
    )
    first = setup.cases[0]
    altered_selector = first.candidate_selector.model_copy(
        update={"source_line_start": 999, "source_line_end": 999}
    )
    altered_case = first.model_copy(
        update={"candidate_selector": altered_selector}
    )
    altered_setup = setup.model_copy(
        update={"cases": (altered_case, *setup.cases[1:])}
    )
    with pytest.raises(bundle.BundleBuildError, match="public_lines"):
        bundle.validate_author_setup_binding(public, altered_setup)


def test_workflow_script_direct_verify_public_and_score_help(tmp_path):
    source, plan_path, _ = _source_package(tmp_path)
    output = tmp_path / "public"
    bundle.build_bundle(plan_path, source, output)
    repository = Path(__file__).resolve().parents[1]
    script = repository / "scripts" / "ooc_annotation_workflow.py"
    verified = subprocess.run(
        [
            sys.executable,
            str(script),
            "verify-public",
            "--public",
            str(output / "public-input.json"),
            "--bundle-root",
            str(output),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["valid"] is True

    help_result = subprocess.run(
        [sys.executable, str(script), "score", "--help"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--author-setup" in help_result.stdout
    assert "--execution-freeze" in help_result.stdout
    assert "--run-config" in help_result.stdout
    assert "--sealed-run-report" in help_result.stdout
    assert "--sealed-prediction" not in help_result.stdout
    assert "--prediction" not in help_result.stdout


def test_build_bundle_rejects_gold_fields_and_output_overwrite(tmp_path):
    source, plan_path, plan = _source_package(tmp_path)
    plan["groups"][0]["cases"][0]["expected_outcome"] = "conflict"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Exception, match="extra_forbidden"):
        bundle.build_bundle(plan_path, source, tmp_path / "rejected")

    _, clean_plan_path, _ = _source_package(tmp_path / "second")
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("preserve", encoding="utf-8")
    with pytest.raises(bundle.BundleBuildError, match="already_exists"):
        bundle.build_bundle(clean_plan_path, source, output)
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_source_plan_rejects_author_axis_for_object_bound_dimensions(tmp_path):
    source, plan_path, plan = _source_package(tmp_path)
    plan["groups"][0]["cases"][1]["author_axis"] = {
        "display_name": "饮品偏好",
        "definition": "比较角色对饮品的稳定偏好。",
        "positive_proposition": "角色喜欢清淡的青梅茶。",
        "applicability_scope": "身体状态正常的日常饮用。",
        "axis_alignment": "same",
    }
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(Exception, match="author_axis_dimension_not_supported"):
        bundle.build_bundle(plan_path, source, tmp_path / "rejected-axis")


def test_build_bundle_rejects_out_of_range_selector_and_cleans_new_output(tmp_path):
    source, plan_path, plan = _source_package(tmp_path)
    plan["groups"][0]["cases"][0]["candidate_selector"]["source_line_end"] = 999
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")
    output = tmp_path / "partial"
    with pytest.raises(bundle.BundleBuildError, match="line_out_of_range"):
        bundle.build_bundle(plan_path, source, output)
    assert not output.exists()
