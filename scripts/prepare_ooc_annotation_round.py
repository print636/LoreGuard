"""Prepare an independently frozen, balanced 18-case public annotation round.

Only the two explicitly named answer-free HTML pages are read. No directory
discovery, human submission, private gold, prediction, Provider, or API is used.
The source pages and their reconstructed 36-case bundle remain untouched and
are retained inside the new directory. Selection uses public order and trait
dimensions, never expected outcomes or inferred difficulty.
"""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import generate_ooc_annotation_page as page
from scripts import ooc_annotation_workflow as workflow
from scripts import ooc_closed_bundle as closed
from scripts import ooc_eval_contract as contract


ROUND_ID = "ooc-choice-short-round-v2"
MANUAL_VERSION = "ooc-manual-v2-choice"
SELECTION_POLICY = "first-balanced-whole-groups-v1"
DIMENSIONS = tuple(workflow._TRAIT_DIMENSION.values())
MAX_PAGE_BYTES = 16 * 1024 * 1024
DATA_PATTERN = re.compile(
    r'<script id="annotation-data" type="application/json">(.*?)</script>', re.DOTALL
)


class AnnotationRoundError(ValueError):
    """Fail-closed error; messages never include source document contents."""


def _read_page(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = str(path)
    if "://" in raw or raw.startswith(("\\\\", "//")):
        raise AnnotationRoundError("page_must_be_local")
    if workflow._path_chain_has_reparse_point(path):
        raise AnnotationRoundError("page_reparse_path_forbidden")
    if not path.is_file() or path.stat().st_size > MAX_PAGE_BYTES:
        raise AnnotationRoundError("page_missing_or_too_large")
    payload = path.read_bytes()
    matches = DATA_PATTERN.findall(payload.decode("utf-8-sig"))
    if len(matches) != 1:
        raise AnnotationRoundError("page_requires_one_public_data_payload")
    data = json.loads(matches[0])
    if not isinstance(data, dict) or data.get("page_schema_version") != page.PAGE_SCHEMA_VERSION:
        raise AnnotationRoundError("page_schema_invalid")
    if data.get("annotation_schema_version") != workflow.ANNOTATION_SCHEMA_VERSION:
        raise AnnotationRoundError("annotation_schema_invalid")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise AnnotationRoundError("page_cases_missing")
    if not isinstance(data.get("reviewer_id"), str) or not data["reviewer_id"]:
        raise AnnotationRoundError("page_reviewer_missing")
    return payload, data


def _restore_document(document: dict[str, Any]) -> bytes:
    """Recover exact LF bytes, accepting only a matching original SHA-256."""
    lines = document["lines"]
    if not isinstance(lines, list):
        raise AnnotationRoundError("document_lines_invalid")
    texts: list[str] = []
    for number, line in enumerate(lines, start=1):
        if line.get("line_number") != number or not isinstance(line.get("text"), str):
            raise AnnotationRoundError("document_line_number_invalid")
        text = line["text"]
        # splitlines also treats these characters as separators. Requiring the
        # round-trip below prevents manufacturing line positions from them.
        if "\r" in text or "\n" in text:
            raise AnnotationRoundError("document_line_contains_separator")
        texts.append(text)
    base = "\n".join(texts).encode("utf-8")
    candidates = (base, base + b"\n") if lines else (b"",)
    for candidate in candidates:
        if (
            hashlib.sha256(candidate).hexdigest() == document["sha256"]
            and candidate.decode("utf-8").splitlines() == texts
        ):
            return candidate
    raise AnnotationRoundError("document_original_bytes_not_recoverable")


def reconstruct_source(
    data: dict[str, Any],
) -> tuple[
    contract.PublicInputBundle,
    closed.PublicAuthorSetupBundle,
    closed.ExecutionFreeze,
    dict[str, bytes],
]:
    """Reconstruct and hash-bind all public metadata without reading a gold file."""
    public_cases: list[dict[str, Any]] = []
    setup_cases: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    for number, case in enumerate(data["cases"], start=1):
        if case.get("case_number") != number:
            raise AnnotationRoundError("source_case_order_invalid")
        documents: list[dict[str, Any]] = []
        setups: list[dict[str, Any]] = []
        for document in case["documents"]:
            snapshot = contract.DocumentSnapshot.model_validate_json(json.dumps({
                key: document[key] for key in ("document_id", "path", "version", "sha256")
            }))
            payload = _restore_document(document)
            previous = payloads.setdefault(snapshot.path, payload)
            if previous != payload:
                raise AnnotationRoundError("shared_document_bytes_differ")
            documents.append(snapshot.model_dump(mode="json"))
            setups.append({key: document[key] for key in (
                "document_id", "phase", "document_role", "story_scope",
                "resolution_state", "publication_status", "import_order",
            )})
        public_cases.append({
            **{key: case[key] for key in (
                "case_id", "split_id", "group_id", "world_id", "axis", "evidence_catalog",
            )},
            "documents": documents,
        })
        setup_cases.append({
            **{key: case[key] for key in (
                "case_id", "group_id", "target_document_id", "candidate_selector", "author_axis",
            )},
            "documents": setups,
            "unselected_candidate_policy": "reject",
        })
        if case["dimension"] != workflow._TRAIT_DIMENSION[case["candidate_selector"]["trait_type"]]:
            raise AnnotationRoundError("source_dimension_selector_mismatch")
    public = contract.PublicInputBundle.model_validate_json(json.dumps({
        "dataset_id": data["dataset_id"], "cases": public_cases,
    }))
    setup = closed.PublicAuthorSetupBundle.model_validate_json(json.dumps({
        "dataset_id": data["dataset_id"],
        "public_input_sha256": contract.canonical_sha256(public),
        "cases": setup_cases,
    }))
    freeze = closed.ExecutionFreeze(
        dataset_id=public.dataset_id,
        public_input_sha256=contract.canonical_sha256(public),
        author_setup_sha256=contract.canonical_sha256(setup),
        case_count=len(public.cases),
        group_count=len({case.group_id for case in public.cases}),
    )
    closed.validate_execution_freeze_binding(public, setup, freeze)
    for field, model in (
        ("public_input_sha256", public), ("author_setup_sha256", setup),
        ("execution_freeze_sha256", freeze),
    ):
        if data.get(field) != contract.canonical_sha256(model):
            raise AnnotationRoundError(f"source_{field}_mismatch")
    return public, setup, freeze, payloads


def select_balanced_groups(cases: Sequence[dict[str, Any]]) -> tuple[str, ...]:
    """First public-order combination of six whole three-case groups, 3/dimension."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        if case["dimension"] not in DIMENSIONS:
            raise AnnotationRoundError("unknown_dimension")
        groups.setdefault(case["group_id"], []).append(case)
    ordered = list(groups)
    if len(ordered) > 64:
        raise AnnotationRoundError("source_group_limit_exceeded")
    vectors = [tuple(Counter(case["dimension"] for case in groups[group])[dim]
                     for dim in DIMENSIONS) for group in ordered]

    @lru_cache(maxsize=None)
    def choose(index: int, taken: int, counts: tuple[int, ...]) -> tuple[int, ...] | None:
        if taken == 6:
            return () if counts == (3,) * len(DIMENSIONS) else None
        if len(ordered) - index < 6 - taken:
            return None
        for candidate in range(index, len(ordered)):
            if len(groups[ordered[candidate]]) != 3:
                continue
            updated = tuple(left + right for left, right in zip(counts, vectors[candidate]))
            if any(value > 3 for value in updated):
                continue
            rest = choose(candidate + 1, taken + 1, updated)
            if rest is not None:
                return (candidate, *rest)
        return None

    selected = choose(0, 0, (0,) * len(DIMENSIONS))
    if selected is None:
        raise AnnotationRoundError("no_balanced_eighteen_case_whole_group_subset")
    return tuple(ordered[index] for index in selected)


def build_previous_round(
    source: dict[str, Any], selected_case_ids: Sequence[str],
) -> dict[str, Any]:
    """Only public binding metadata; no draft, declaration, label, or answer."""
    source_cases = {case["case_id"]: case for case in source["cases"]}
    if len(set(selected_case_ids)) != len(selected_case_ids) or not set(selected_case_ids) <= set(source_cases):
        raise AnnotationRoundError("migration_case_set_invalid")
    keys = ("dataset_id", "public_input_sha256", "author_setup_sha256",
            "execution_freeze_sha256", "manual_version", "reviewer_id")
    result = {key: source[key] for key in keys}
    result["draft_key"] = ":".join((
        "loreguard-ooc-draft-v1", source["public_input_sha256"], source["reviewer_id"],
        source["manual_version"], source["author_setup_sha256"], source["execution_freeze_sha256"],
    ))
    result["source_case_ids"] = [case["case_id"] for case in source["cases"]]
    result["selected_case_ids"] = list(selected_case_ids)
    result["case_number_map"] = [
        {"case_id": case_id, "source_case_number": source_cases[case_id]["case_number"],
         "case_number": index,
         "case_snapshot": {key: value for key, value in source_cases[case_id].items() if key != "case_number"}}
        for index, case_id in enumerate(selected_case_ids, start=1)
    ]
    result["human_declarations_must_be_reconfirmed"] = True
    return result


def _write_json(path: Path, value: Any) -> None:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    with path.open("x", encoding="utf-8", newline="\n") as target:
        target.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_bundle(
    directory: Path, public: contract.PublicInputBundle,
    setup: closed.PublicAuthorSetupBundle, freeze: closed.ExecutionFreeze,
    payloads: dict[str, bytes],
) -> None:
    directory.mkdir()
    paths = {document.path for case in public.cases for document in case.documents}
    for relative in sorted(paths):
        destination = directory.joinpath(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as target:
            target.write(payloads[relative])
    _write_json(directory / "public-input.json", public)
    _write_json(directory / "author-setup.json", setup)
    _write_json(directory / "freeze.json", freeze)
    workflow.validate_exhaustive_line_catalog(public, directory)


def prepare_annotation_round(
    reviewer_a: Path, reviewer_b: Path, output_root: Path, *,
    allowed_output_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    """Create a new standalone round; never overwrite or edit source handouts."""
    a_bytes, source_a = _read_page(reviewer_a)
    b_bytes, source_b = _read_page(reviewer_b)
    if source_a["reviewer_id"] == source_b["reviewer_id"]:
        raise AnnotationRoundError("reviewer_identities_must_differ")
    if ({key: value for key, value in source_a.items() if key != "reviewer_id"}
            != {key: value for key, value in source_b.items() if key != "reviewer_id"}):
        raise AnnotationRoundError("reviewer_public_inputs_differ")
    original, original_setup, original_freeze, payloads = reconstruct_source(source_a)
    if len(original.cases) != 36 or original_freeze.group_count != 12:
        raise AnnotationRoundError("source_requires_thirty_six_cases_in_twelve_groups")
    if set(Counter(case.group_id for case in original.cases).values()) != {3}:
        raise AnnotationRoundError("source_groups_must_have_three_cases_each")
    if Counter(case["dimension"] for case in source_a["cases"]) != Counter({dimension: 6 for dimension in DIMENSIONS}):
        raise AnnotationRoundError("source_requires_six_cases_per_dimension")
    # Both input pages have already been compared field-for-field; rebuilding
    # page_data after document recovery validates all presentation-only fields.
    selected_groups = select_balanced_groups(source_a["cases"])
    selected = tuple(case for case in original.cases if case.group_id in selected_groups)
    selected_ids = tuple(case.case_id for case in selected)
    dataset_id = str(uuid4())
    public = contract.PublicInputBundle(dataset_id=dataset_id, cases=selected)
    setup = closed.PublicAuthorSetupBundle(
        dataset_id=dataset_id, public_input_sha256=contract.canonical_sha256(public),
        cases=tuple(case for case in original_setup.cases if case.case_id in selected_ids),
    )
    freeze = closed.ExecutionFreeze(
        dataset_id=dataset_id, public_input_sha256=contract.canonical_sha256(public),
        author_setup_sha256=contract.canonical_sha256(setup), case_count=18, group_count=6,
    )
    closed.validate_execution_freeze_binding(public, setup, freeze)
    roots = allowed_output_roots or (Path.home() / "Desktop" / "doc", Path.home() / "Desktop" / "dev")
    raw_output = str(output_root)
    if "://" in raw_output or raw_output.startswith(("\\\\", "//")):
        raise AnnotationRoundError("output_must_be_local")
    if workflow._path_chain_has_reparse_point(output_root):
        raise AnnotationRoundError("output_reparse_path_forbidden")
    resolved = output_root.parent.resolve(strict=True) / output_root.name
    if not any(resolved.is_relative_to(root.resolve(strict=True)) and resolved != root.resolve(strict=True) for root in roots):
        raise AnnotationRoundError("output_must_stay_under_desktop_doc_or_dev")
    if resolved.exists():
        raise AnnotationRoundError("output_already_exists")
    if reviewer_a.read_bytes() != a_bytes or reviewer_b.read_bytes() != b_bytes:
        raise AnnotationRoundError("source_pages_changed_during_preflight")

    # The only newly created directory is a validated, explicit child. Errors
    # leave a partial directory for inspection; no recursive cleanup is used.
    resolved.mkdir()
    _write_bundle(resolved / "source-36", original, original_setup, original_freeze, payloads)
    for name, payload in (("reviewer-a-v1.html", a_bytes), ("reviewer-b-v1.html", b_bytes)):
        with (resolved / "source-36" / name).open("xb") as target:
            target.write(payload)
    rebuilt_a = page.build_page_data(
        original, resolved / "source-36", original_setup, original_freeze,
        reviewer_id=source_a["reviewer_id"], manual_version=source_a["manual_version"],
    )
    if rebuilt_a != source_a:
        raise AnnotationRoundError("source_page_presentation_not_reproducible")
    _write_bundle(resolved / "bundle", public, setup, freeze, payloads)
    round_meta = {
        "round_id": ROUND_ID, "selection_policy": SELECTION_POLICY,
        "source_case_count": len(original.cases), "case_count": 18,
        "source_group_count": original_freeze.group_count, "group_count": 6,
        "dimension_counts": dict(Counter(case["dimension"] for case in source_a["cases"]
                                         if case["case_id"] in selected_ids)),
        "source_public_input_sha256": contract.canonical_sha256(original),
    }
    for name, source in (("reviewer-a-v1.html", source_a), ("reviewer-b-v1.html", source_b)):
        data = page.build_page_data(
            public, resolved / "bundle", setup, freeze,
            reviewer_id=source["reviewer_id"], manual_version=MANUAL_VERSION,
        )
        data["annotation_round"] = round_meta
        data["previous_round"] = build_previous_round(source, selected_ids)
        for mapping, current in zip(data["previous_round"]["case_number_map"], data["cases"]):
            if mapping["case_snapshot"] != {key: value for key, value in current.items() if key != "case_number"}:
                raise AnnotationRoundError("selected_case_content_changed")
        with (resolved / name).open("x", encoding="utf-8", newline="\n") as target:
            target.write(page.render_annotation_page(data))
    manifest = {
        "schema_version": "loreguard-ooc-short-round-manifest-v1",
        **round_meta,
        "dataset_id": dataset_id,
        "public_input_sha256": contract.canonical_sha256(public),
        "author_setup_sha256": contract.canonical_sha256(setup),
        "execution_freeze_sha256": contract.canonical_sha256(freeze),
        "manual_version": MANUAL_VERSION,
        "source_dataset_id": original.dataset_id,
        "source_author_setup_sha256": contract.canonical_sha256(original_setup),
        "source_execution_freeze_sha256": contract.canonical_sha256(original_freeze),
        "source_page_sha256": {"A": hashlib.sha256(a_bytes).hexdigest(), "B": hashlib.sha256(b_bytes).hexdigest()},
        "selected_group_ids": list(selected_groups),
        "case_mapping": [
            {"case_id": case["case_id"], "source_case_number": case["case_number"],
             "selected": case["case_id"] in selected_ids,
             "case_number": selected_ids.index(case["case_id"]) + 1 if case["case_id"] in selected_ids else None}
            for case in source_a["cases"]
        ],
        "statistics_scope": "Independent 18-case / 6-story pilot only; do not reuse the 36-case denominator, gold, predictions, or completion claims.",
    }
    _write_json(resolved / "round-manifest.json", manifest)
    with (resolved / "批次说明.md").open("x", encoding="utf-8", newline="\n") as target:
        target.write(
            "# 18 题简化版：本批次说明\n\n"
            "本轮为独立的小批次：18 题、6 个完整故事组，六个角色维度各 3 题。"
            "按原公开顺序选择首个满足平衡条件的完整故事组组合；没有根据答案、模型结果或猜测难度挑题。\n\n"
            "故事原文、文档哈希和案例/证据身份逐字保留，但本轮 dataset 与冻结校验值重新生成，"
            "不能沿用原 36 题的分母、成绩、标准答案或预测文件，也不代表生产准确率。"
            "精确清单在 round-manifest.json；正式输入在 bundle/。\n\n"
            "原 36 题页面及原冻结材料在 source-36/，原位置文件未改动。"
            "未选入本轮的题目没有被标成已完成。\n\n"
            "只分发本目录的 A/B 作答页与配套手册，不必分发内部 bundle、清单或编号。"
            "旧进度仅在浏览器能读取同一标注者的旧草稿时，由本人明确确认恢复已保留题目；"
            "不继承独立作答声明。file:// 的存储规则依浏览器而异，跨路径恢复不作保证，"
            "原页仍可继续作答和保存，不要删除旧页或旧草稿。\n"
        )
    if reviewer_a.read_bytes() != a_bytes or reviewer_b.read_bytes() != b_bytes:
        raise AnnotationRoundError("source_pages_changed_during_generation")
    return {"output": str(resolved), "case_count": 18, "group_count": 6,
            "dimension_counts": round_meta["dimension_counts"], "round_id": ROUND_ID,
            "source_files_unchanged": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_annotation_round(args.reviewer_a, args.reviewer_b, args.output_root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # Pydantic errors may contain document/selector values: print only a
        # known error code, never the full validation exception or payload.
        code = str(exc) if isinstance(exc, AnnotationRoundError) else type(exc).__name__
        print(json.dumps({"status": "failed", "code": code}))
        return 1
    print(json.dumps({"status": "created", **result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
