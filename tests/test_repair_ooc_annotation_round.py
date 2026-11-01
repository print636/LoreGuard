"""Synthetic-only regressions for the one-line/multiple-events handout repair."""

from collections import Counter
import hashlib
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


@pytest.fixture
def v2_source(tmp_path):
    stories = tmp_path / "stories"
    stories.mkdir()
    groups = []
    traits = tuple(workflow._TRAIT_DIMENSION)
    for group in range(6):
        baseline = f"baseline-{group}.md"
        target = f"target-{group}.md"
        (stories / baseline).write_bytes(f"角色{group}明确的设定，附带适用范围。\n".encode("utf-8"))
        (stories / target).write_bytes((f"# 合成新稿 {group}\n"
                                        f"周一角色{group}做一件事，仍在同一场合；五天后又做另一件事。\n"
                                        f"另一个角色{group}的记事，保持完整上下文。\n").encode("utf-8"))
        documents = [
            {"document_key": "baseline", "source_path": baseline, "phase": "baseline",
             "document_role": "character_profile", "story_scope": "main", "resolution_state": "confirmed",
             "publication_status": "published", "import_order": 1},
            {"document_key": "target", "source_path": target, "phase": "target",
             "document_role": "chapter", "story_scope": "main", "resolution_state": "draft",
             "publication_status": "draft", "import_order": 2},
        ]
        cases = [{"case_key": f"case-{group}-{offset}", "candidate_selector": {
            "character_key": f"角色{group}", "trait_type": traits[(group + offset) % 6],
            "source_document_key": "baseline", "source_line_start": 1, "source_line_end": 1,
            "origin": "explicit_setting", "polarity": "positive", "stability": "stable",
        }} for offset in range(3)]
        for case in cases:
            if case["candidate_selector"]["trait_type"] in {"core_personality", "value"}:
                case["author_axis"] = {
                    "display_name": "合成比较项", "definition": "合成规则的比较范围。",
                    "positive_proposition": f"角色{group}遵循合成规则。",
                    "applicability_scope": "仅用于该合成故事场合。", "axis_alignment": "same",
                }
        groups.append({"group_key": f"group-{group}", "documents": documents, "cases": cases})
    plan = tmp_path / "source-plan.json"
    plan.write_text(json.dumps({"split_id": "closed-pilot-v1", "groups": groups}), encoding="utf-8")
    root = tmp_path / "old-bundle"
    freeze = closed.build_bundle(plan, stories, root)
    public = contract.load_public_input_file(root / "public-input.json")
    setup = closed.PublicAuthorSetupBundle.model_validate_json((root / "author-setup.json").read_bytes())
    a, b = tmp_path / "a.html", tmp_path / "b.html"
    for name, path in (("a", a), ("b", b)):
        data = page.build_page_data(public, root, setup, freeze, reviewer_id=f"synthetic-{name}",
                                    manual_version=previous.MANUAL_VERSION)
        path.write_text(page.render_annotation_page(data), encoding="utf-8")
    return a, b, root, public, setup


def _data(path):
    return json.loads(previous.DATA_PATTERN.findall(path.read_text(encoding="utf-8"))[0])


@pytest.mark.parametrize("ending", ["", "\n"])
def test_fragmentation_only_inserts_lf_and_retains_exact_source_offsets(ending):
    original = ("# 标题，不拆\n甲🙂做了事，\"仍保留引号\"；另一日又做事。\n\n末行" + ending).encode("utf-8")
    new, mapping = repair.fragment_document(original)
    assert new.replace(b"\n", b"") == original.replace(b"\n", b"")
    assert new.endswith(b"\n") == original.endswith(b"\n")
    assert new.splitlines()[0] == original.splitlines()[0]
    old_lines = original.decode().splitlines()
    for row, fragment in zip(mapping, new.decode().splitlines()):
        assert fragment == old_lines[row["source_line_number"] - 1][row["source_char_start"]:row["source_char_end"]]
    assert len([row for row in mapping if row["source_line_number"] == 2]) == 2


@pytest.mark.parametrize(("original", "expected"), [
    ("甲，乙,丙：丁:戊？己?庚！辛!壬。", "甲，乙,丙：丁:戊？己?庚！辛!壬。"),
    ("甲；乙;丙。丁", "甲；\n乙;\n丙。\n丁"),
    ("她说：“甲，乙。”她离开；他点头;结束。", "她说：“甲，乙。”\n她离开；\n他点头;\n结束。"),
    ('他说:"甲，乙。"她离开；他答:\'好的；\'然后结束。', '他说:"甲，乙。"\n她离开；\n他答:\'好的；\'\n然后结束。'),
    ("版本v1.2.3;Dr. Wang... said. 3.14", "版本v1.2.3;\nDr. Wang... said. 3.14"),
])
def test_only_chinese_full_stops_and_semicolons_split_fragments(original, expected):
    reformatted, mapping = repair.fragment_document(original.encode("utf-8"))
    assert reformatted.decode("utf-8") == expected
    assert reformatted.replace(b"\n", b"") == original.encode("utf-8")
    for row, fragment in zip(mapping, expected.splitlines(), strict=True):
        assert row["source_line_number"] == 1
        assert original[row["source_char_start"]:row["source_char_end"]] == fragment


def test_repaired_round_is_new_frozen_policy_with_same_cases_and_original_archives(v2_source, tmp_path):
    a, b, old_root, old_public, old_setup = v2_source
    originals = (a.read_bytes(), b.read_bytes())
    output = tmp_path / "repaired"
    result = repair.repair_annotation_round(a, b, output, allowed_output_roots=(tmp_path,))
    assert result["source_files_unchanged"] and result["original_characters_preserved"]
    public = contract.load_public_input_file(output / "bundle/public-input.json")
    setup = closed.PublicAuthorSetupBundle.model_validate_json((output / "bundle/author-setup.json").read_bytes())
    freeze = closed.ExecutionFreeze.model_validate_json((output / "bundle/freeze.json").read_bytes())
    workflow.validate_exhaustive_line_catalog(public, output / "bundle")
    closed.validate_execution_freeze_binding(public, setup, freeze)
    loaded = runner.load_public_bundle(output / "bundle")
    assert loaded.public == public
    assert public.assessment_policy == contract.EVENT_ASSESSMENT_POLICY
    assert public.dataset_id != old_public.dataset_id
    assert [row.case_id for row in public.cases] == [row.case_id for row in old_public.cases]
    assert freeze.case_count == 18 and freeze.group_count == 6
    assert (a.read_bytes(), b.read_bytes()) == originals
    assert (output / "source-18/reviewer-a-v1.html").read_bytes() == originals[0]
    assert (output / "source-18/reviewer-b-v1.html").read_bytes() == originals[1]
    new_a, new_b = _data(output / "reviewer-a-v1.html"), _data(output / "reviewer-b-v1.html")
    assert {key: value for key, value in new_a.items() if key != "reviewer_id"} == {
        key: value for key, value in new_b.items() if key != "reviewer_id"}
    assert "previous_round" not in new_a
    assert new_a["manual_version"] == repair.MANUAL_VERSION
    assert new_a["annotation_round"]["segmentation_policy"] == "semicolons-and-periods-v2"
    manifest = json.loads((output / "round-manifest.json").read_text(encoding="utf-8"))
    assert manifest["segmentation_policy"] == repair.SEGMENTATION_POLICY
    assert new_a["annotation_round"]["automatic_answer_migration"] is False
    assert set(Counter(row["dimension"] for row in new_a["cases"]).values()) == {3}
    documents_by_id = {row.document_id: row for case in public.cases for row in case.documents}
    new_documents = {mapping["source_document_id"]: documents_by_id[document_id]
                     for document_id, mapping in new_a["source_position_map"].items()}
    for old_document in {row.document_id: row for case in old_public.cases for row in case.documents}.values():
        old_bytes = (old_root / old_document.path).read_bytes()
        new_document = new_documents[old_document.document_id]
        new_bytes = (output / "bundle" / new_document.path).read_bytes()
        assert (output / "source-18" / old_document.path).read_bytes() == old_bytes
        assert new_document.version == 1 and new_document.document_id != old_document.document_id
        assert hashlib.sha256(new_bytes).hexdigest() == new_document.sha256
        assert old_bytes.replace(b"\n", b"") == new_bytes.replace(b"\n", b"")
    first_setup = setup.cases[0]
    assert first_setup.candidate_selector.source_line_start == 1
    assert first_setup.candidate_selector.source_line_end == 1
    target = next(doc for doc in new_a["cases"][0]["documents"] if doc["is_target"])
    mapping = new_a["source_position_map"][target["document_id"]]
    same_original_line = [row for row in mapping["fragments"] if row["source_line_number"] == 2]
    assert len(same_original_line) == 2
    target_lines = {row["line_number"]: row["text"] for row in target["lines"]}
    assert "，仍在同一场合；" in target_lines[same_original_line[0]["line_number"]]
    assert target_lines[same_original_line[1]["line_number"]].startswith("五天后")
    assert mapping["offset_unit"] == "unicode_codepoint"


def test_repair_rejects_mismatched_public_data_before_writing(v2_source, tmp_path):
    a, b, *_ = v2_source
    data = _data(b)
    data["cases"][0]["documents"][0]["lines"][0]["text"] += "changed"
    b.write_text(page.render_annotation_page(data), encoding="utf-8")
    with pytest.raises(repair.AnnotationRepairError, match="public_inputs_differ"):
        repair.repair_annotation_round(a, b, tmp_path / "bad", allowed_output_roots=(tmp_path,))
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("kind", ["existing", "outside", "wrong_manual"])
def test_repair_refuses_unsafe_output_and_wrong_source_version(v2_source, tmp_path, kind):
    a, b, *_ = v2_source
    output = tmp_path / "target"
    roots = (tmp_path,)
    if kind == "existing":
        output.mkdir()
    elif kind == "outside":
        roots = (tmp_path / "allowed",)
        roots[0].mkdir()
    else:
        for path in (a, b):
            data = _data(path)
            data["manual_version"] = repair.MANUAL_VERSION
            path.write_text(page.render_annotation_page(data), encoding="utf-8")
    with pytest.raises(repair.AnnotationRepairError):
        repair.repair_annotation_round(a, b, output, allowed_output_roots=roots)
