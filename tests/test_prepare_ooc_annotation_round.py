"""Purely synthetic public fixtures; never open a real annotation/gold file."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from scripts import generate_ooc_annotation_page as page
from scripts import ooc_annotation_workflow as workflow
from scripts import ooc_closed_bundle as closed
from scripts import ooc_eval_contract as contract
from scripts import prepare_ooc_annotation_round as rounds


def _read_data(path: Path) -> dict:
    return json.loads(rounds.DATA_PATTERN.findall(path.read_text(encoding="utf-8"))[0])


def _write_page(path: Path, data: dict) -> None:
    path.write_text(page.render_annotation_page(data), encoding="utf-8")


@pytest.fixture
def source(tmp_path):
    source_root = tmp_path / "synthetic-source"
    source_root.mkdir()
    groups = []
    traits = tuple(workflow._TRAIT_DIMENSION)
    for group_index in range(12):
        baseline = f"baseline-{group_index}.md"
        target = f"target-{group_index}.md"
        (source_root / baseline).write_bytes(
            f"合成角色{group_index}遵守自己的约定。\n无评测答案的设定。\n".encode("utf-8")
        )
        (source_root / target).write_bytes(
            (f"合成故事{group_index}开始。\n"
             "第一次事件。\n第二次事件。\n"
             "</script><script>globalThis.notExecuted = true</script>\n").encode("utf-8")
        )
        documents = []
        for index, (key, name, phase, role, state, published) in enumerate((
            ("baseline", baseline, "baseline", "character_profile", "confirmed", "published"),
            ("target", target, "target", "chapter", "draft", "draft"),
        ), start=1):
            documents.append({
                "document_key": key, "source_path": name, "phase": phase,
                "document_role": role, "story_scope": f"synthetic-{group_index}-{key}",
                "resolution_state": state, "publication_status": published,
                "import_order": index,
            })
        cases = []
        for offset in range(3):
            trait = traits[(group_index + offset) % 6]
            cases.append({
                "case_key": f"case-{group_index}-{offset}",
                "candidate_selector": {
                    "character_key": f"合成角色{group_index}", "trait_type": trait,
                    "source_document_key": "baseline", "source_line_start": 1,
                    "source_line_end": 1, "origin": "explicit_setting",
                    "polarity": "positive", "stability": "stable",
                },
            })
        groups.append({"group_key": f"group-{group_index}", "documents": documents, "cases": cases})
    plan_path = tmp_path / "synthetic-plan.json"
    plan_path.write_text(json.dumps({"split_id": "closed-pilot-v1", "groups": groups}), encoding="utf-8")
    bundle = tmp_path / "source-bundle"
    freeze = closed.build_bundle(plan_path, source_root, bundle)
    public = contract.load_public_input_file(bundle / "public-input.json")
    setup = closed.PublicAuthorSetupBundle.model_validate_json((bundle / "author-setup.json").read_bytes())
    a_path, b_path = tmp_path / "a.html", tmp_path / "b.html"
    data = page.build_page_data(public, bundle, setup, freeze, reviewer_id="synthetic-reviewer-a", manual_version="synthetic-manual-v1")
    _write_page(a_path, data)
    _write_page(b_path, {**data, "reviewer_id": "synthetic-reviewer-b"})
    return a_path, b_path, data, public, setup, freeze


def _prepare(source, tmp_path, output_name="round"):
    return rounds.prepare_annotation_round(
        source[0], source[1], tmp_path / output_name, allowed_output_roots=(tmp_path,)
    )


def test_full_round_preserves_originals_and_creates_real_balanced_freeze(source, tmp_path):
    originals = tuple(path.read_bytes() for path in source[:2])
    result = _prepare(source, tmp_path)
    output = Path(result["output"])
    assert result["source_files_unchanged"] is True
    assert result["case_count"] == 18 and result["group_count"] == 6
    assert set(result["dimension_counts"].values()) == {3}
    assert tuple(path.read_bytes() for path in source[:2]) == originals
    public = contract.load_public_input_file(output / "bundle/public-input.json")
    setup = closed.PublicAuthorSetupBundle.model_validate_json((output / "bundle/author-setup.json").read_bytes())
    freeze = closed.ExecutionFreeze.model_validate_json((output / "bundle/freeze.json").read_bytes())
    closed.validate_execution_freeze_binding(public, setup, freeze)
    workflow.validate_exhaustive_line_catalog(public, output / "bundle")
    assert public.dataset_id != source[3].dataset_id
    assert len(public.cases) == freeze.case_count == 18
    expected_groups = list(dict.fromkeys(case.group_id for case in source[3].cases))[:6]
    assert list(dict.fromkeys(case.group_id for case in public.cases)) == expected_groups
    assert public.cases == tuple(case for case in source[3].cases if case.group_id in expected_groups)
    assert contract.load_public_input_file(output / "source-36/public-input.json") == source[3]
    assert (output / "source-36/reviewer-a-v1.html").read_bytes() == originals[0]
    assert (output / "source-36/reviewer-b-v1.html").read_bytes() == originals[1]
    for case in public.cases:
        for document in case.documents:
            payload = output.joinpath("bundle", *document.path.split("/")).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == document.sha256
    manifest = json.loads((output / "round-manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["case_mapping"]) == 36
    assert sum(row["selected"] for row in manifest["case_mapping"]) == 18
    assert manifest["selection_policy"] == rounds.SELECTION_POLICY
    assert "36-case denominator" in manifest["statistics_scope"]


def test_ab_pages_have_same_new_cases_and_reviewer_bound_explicit_migration(source, tmp_path):
    _prepare(source, tmp_path)
    a = _read_data(tmp_path / "round/reviewer-a-v1.html")
    b = _read_data(tmp_path / "round/reviewer-b-v1.html")
    assert a["cases"] == b["cases"] and len(a["cases"]) == 18
    assert a["reviewer_id"] != b["reviewer_id"]
    assert a["manual_version"] == b["manual_version"] == rounds.MANUAL_VERSION
    assert a["public_input_sha256"] == b["public_input_sha256"] != source[2]["public_input_sha256"]
    for data in (a, b):
        previous = data["previous_round"]
        assert previous["reviewer_id"] == data["reviewer_id"]
        assert previous["source_case_ids"] == [row["case_id"] for row in source[2]["cases"]]
        assert previous["human_declarations_must_be_reconfirmed"] is True
        assert previous["draft_key"] == ":".join((
            "loreguard-ooc-draft-v1", source[2]["public_input_sha256"], data["reviewer_id"],
            source[2]["manual_version"], source[2]["author_setup_sha256"], source[2]["execution_freeze_sha256"],
        ))
        assert len(previous["case_number_map"]) == 18
        for mapping, case in zip(previous["case_number_map"], data["cases"]):
            assert mapping["case_snapshot"] == {key: value for key, value in case.items() if key != "case_number"}
        assert not {"blinded_to_peer", "independence_declaration", "locked", "reviewer_kind"} & set(previous)
        assert set(Counter(case["dimension"] for case in data["cases"]).values()) == {3}
    assert a["previous_round"]["draft_key"] != b["previous_round"]["draft_key"]


@pytest.mark.parametrize("field", ["public_input_sha256", "author_setup_sha256", "execution_freeze_sha256"])
def test_source_stale_freeze_is_rejected_before_output(source, tmp_path, field):
    for path in source[:2]:
        data = _read_data(path)
        data[field] = "f" * 64
        _write_page(path, data)
    with pytest.raises(rounds.AnnotationRoundError, match=f"source_{field}_mismatch"):
        _prepare(source, tmp_path)
    assert not (tmp_path / "round").exists()


def test_ab_mismatch_or_same_identity_is_rejected(source, tmp_path):
    b = _read_data(source[1])
    b["reviewer_id"] = source[2]["reviewer_id"]
    _write_page(source[1], b)
    with pytest.raises(rounds.AnnotationRoundError, match="identities_must_differ"):
        _prepare(source, tmp_path)
    b["reviewer_id"] = "synthetic-reviewer-b"
    b["cases"][0]["dimension"] = "value_boundary"
    _write_page(source[1], b)
    with pytest.raises(rounds.AnnotationRoundError, match="public_inputs_differ"):
        _prepare(source, tmp_path)
    assert not (tmp_path / "round").exists()


@pytest.mark.parametrize("mutation", ["text", "line_number", "dimension", "case_number"])
def test_corrupted_public_pages_do_not_silently_change_materials(source, tmp_path, mutation):
    for path in source[:2]:
        data = _read_data(path)
        case = data["cases"][0]
        if mutation == "text":
            case["documents"][0]["lines"][0]["text"] += "篡改"
        elif mutation == "line_number":
            case["documents"][0]["lines"][0]["line_number"] = 9
        elif mutation == "dimension":
            case["dimension"] = "motivation_goal"
        else:
            case["case_number"] = 9
        _write_page(path, data)
    with pytest.raises(rounds.AnnotationRoundError):
        _prepare(source, tmp_path)
    assert not (tmp_path / "round").exists()


def test_existing_output_and_outside_desktop_output_are_never_overwritten(source, tmp_path):
    directory = tmp_path / "round"
    directory.mkdir()
    sentinel = directory / "do-not-touch.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(rounds.AnnotationRoundError, match="already_exists"):
        _prepare(source, tmp_path)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    with pytest.raises(rounds.AnnotationRoundError, match="stay_under"):
        rounds.prepare_annotation_round(source[0], source[1], tmp_path / "outside", allowed_output_roots=(allowed,))
    assert not (tmp_path / "outside").exists()


def test_no_balanced_complete_groups_is_an_error_not_a_random_or_partial_fallback():
    cases = [{"group_id": f"g-{group}", "dimension": "core_trait"}
             for group in range(12) for _ in range(3)]
    with pytest.raises(rounds.AnnotationRoundError, match="no_balanced"):
        rounds.select_balanced_groups(cases)


@pytest.mark.parametrize("ending", ["", "\n", "\n\n"])
def test_document_restoration_preserves_exact_final_newlines(ending):
    raw = ("第一行\n第二行" + ending).encode("utf-8")
    document = {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "lines": [{"line_number": i, "text": value} for i, value in enumerate(raw.decode().splitlines(), 1)],
    }
    assert rounds._restore_document(document) == raw


def test_previous_round_contains_only_retained_case_snapshots_and_no_human_answers(source):
    selected = [case["case_id"] for case in source[2]["cases"][:18]]
    previous = rounds.build_previous_round(source[2], selected)
    assert len(previous["source_case_ids"]) == 36
    assert [row["case_id"] for row in previous["case_number_map"]] == selected
    snapshots = previous["case_number_map"]
    assert all("outcome" not in item["case_snapshot"] for item in snapshots)
    with pytest.raises(rounds.AnnotationRoundError, match="migration_case_set"):
        rounds.build_previous_round(source[2], [selected[0], selected[0]])
