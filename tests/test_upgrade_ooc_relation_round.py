"""No real stories or answers: V3-to-V4 immutable-input upgrade checks."""

import json
from pathlib import Path

import pytest

from scripts import generate_ooc_annotation_page as page
from scripts import ooc_annotation_workflow as workflow
from scripts import ooc_closed_bundle as closed
from scripts import ooc_eval_contract as contract
from scripts import prepare_ooc_annotation_round as previous
from scripts import repair_ooc_annotation_round as repair
from scripts import run_ooc_sealed_http as runner
from scripts import upgrade_ooc_relation_round as upgrade


@pytest.fixture
def source_v3(tmp_path):
    stories = tmp_path / "stories"
    stories.mkdir()
    groups = []
    traits = tuple(workflow._TRAIT_DIMENSION)
    for group in range(6):
        (stories / f"baseline-{group}.md").write_bytes(f"合成角色{group}始终遵循合成规则。\n".encode("utf-8"))
        (stories / f"target-{group}.md").write_bytes(f"# 合成标题{group}\n角色{group}完成一事，仍在同一场合；隔日完成另一事。\n".encode("utf-8"))
        documents = [{"document_key": key, "source_path": f"{key}-{group}.md", "phase": phase,
                      "document_role": role, "story_scope": "main", "resolution_state": resolution,
                      "publication_status": status, "import_order": index}
                     for index, (key, phase, role, resolution, status) in enumerate(
                         (("baseline", "baseline", "character_profile", "confirmed", "published"),
                          ("target", "target", "chapter", "draft", "draft")), 1)]
        cases = []
        for offset in range(3):
            trait = traits[(group + offset) % 6]
            row = {"case_key": f"case-{group}-{offset}", "candidate_selector": {
                "character_key": f"合成角色{group}", "trait_type": trait,
                "source_document_key": "baseline", "source_line_start": 1, "source_line_end": 1,
                "origin": "explicit_setting", "polarity": "positive", "stability": "stable"}}
            if trait in {"core_personality", "value"}:
                row["author_axis"] = {"display_name": "合成轴", "definition": "只判断合成规则。",
                    "positive_proposition": f"合成角色{group}遵循合成规则。",
                    "applicability_scope": "仅限合成场合。", "axis_alignment": "same"}
            cases.append(row)
        groups.append({"group_key": f"group-{group}", "documents": documents, "cases": cases})
    plan = tmp_path / "source-plan.json"
    plan.write_text(json.dumps({"split_id": "closed-pilot-v1", "groups": groups}), encoding="utf-8")
    bundle = tmp_path / "v2-bundle"
    freeze = closed.build_bundle(plan, stories, bundle)
    public = contract.load_public_input_file(bundle / "public-input.json")
    setup = closed.PublicAuthorSetupBundle.model_validate_json((bundle / "author-setup.json").read_bytes())
    a, b = tmp_path / "old-a.html", tmp_path / "old-b.html"
    for name, path in (("a", a), ("b", b)):
        data = page.build_page_data(public, bundle, setup, freeze, reviewer_id=name,
                                    manual_version=previous.MANUAL_VERSION)
        path.write_text(page.render_annotation_page(data), encoding="utf-8")
    repaired = tmp_path / "v3"
    repair.repair_annotation_round(a, b, repaired, allowed_output_roots=(tmp_path,))
    return repaired / "reviewer-a-v1.html", repaired / "reviewer-b-v1.html", repaired


def _data(path):
    return previous._read_page(path)[1]


def test_upgrade_changes_policy_freeze_only_and_keeps_every_source_byte_anchor_and_position(source_v3, tmp_path):
    a, b, old_root = source_v3
    old_bytes = (a.read_bytes(), b.read_bytes())
    old_public = contract.load_public_input_file(old_root / "bundle/public-input.json")
    # General reconstruction now accepts V3 while preserving its exact hash.
    rebuilt, _, _, _ = previous.reconstruct_source(_data(a))
    assert rebuilt == old_public
    output = tmp_path / "v4"
    result = upgrade.upgrade_relation_round(a, b, output, allowed_output_roots=(tmp_path,))
    assert result["document_bytes_and_anchors_unchanged"] and result["source_files_unchanged"]
    public = contract.load_public_input_file(output / "bundle/public-input.json")
    assert runner.load_public_bundle(output / "bundle").public == public
    assert public.relation_policy == contract.DIRECTIONAL_RELATION_POLICY
    assert public.cases == old_public.cases and public.dataset_id != old_public.dataset_id
    assert public.assessment_policy == old_public.assessment_policy
    assert (a.read_bytes(), b.read_bytes()) == old_bytes
    for name, raw in zip(("reviewer-a-v1.html", "reviewer-b-v1.html"), old_bytes, strict=True):
        assert (output / "source-v3" / name).read_bytes() == raw
    for document in {d.path: d for case in public.cases for d in case.documents}.values():
        assert (output / "bundle" / document.path).read_bytes() == (old_root / "bundle" / document.path).read_bytes()
    new_a, new_b = _data(output / "reviewer-a-v1.html"), _data(output / "reviewer-b-v1.html")
    assert {k:v for k,v in new_a.items() if k != "reviewer_id"} == {k:v for k,v in new_b.items() if k != "reviewer_id"}
    assert new_a["source_position_map"] == _data(a)["source_position_map"]
    assert new_a["manual_version"] == workflow.RELATION_MANUAL_VERSION
    assert "previous_round" not in new_a
    assert new_a["annotation_round"]["automatic_answer_migration"] is False
    assert len(new_a["cases"]) == 18


@pytest.mark.parametrize("mutation", ["identity", "content", "old_manual", "already_v4", "existing", "outside"])
def test_upgrade_refuses_unsafe_or_mismatched_sources_without_overwriting(source_v3, tmp_path, mutation):
    a, b, _ = source_v3
    out = tmp_path / "bad"
    roots = (tmp_path,)
    if mutation == "identity":
        data = _data(b); data["reviewer_id"] = _data(a)["reviewer_id"]
        b.write_text(page.render_annotation_page(data), encoding="utf-8")
    elif mutation == "content":
        data = _data(b); data["cases"][0]["documents"][0]["lines"][0]["text"] += "tampered"
        b.write_text(page.render_annotation_page(data), encoding="utf-8")
    elif mutation in {"old_manual", "already_v4"}:
        for path in (a, b):
            data = _data(path)
            data["manual_version"] = previous.MANUAL_VERSION if mutation == "old_manual" else workflow.RELATION_MANUAL_VERSION
            path.write_text(page.render_annotation_page(data), encoding="utf-8")
    elif mutation == "existing":
        out.mkdir()
    else:
        allowed = tmp_path / "allowed"; allowed.mkdir(); roots = (allowed,)
    with pytest.raises((upgrade.RelationUpgradeError, repair.AnnotationRepairError)):
        upgrade.upgrade_relation_round(a, b, out, allowed_output_roots=roots)
    if mutation != "existing":
        assert not out.exists()
