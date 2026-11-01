"""Re-freeze the 18-case annotation pilot with neutral, reversible fragments.

Only explicitly supplied, answer-free handouts are read. No annotations,
gold, predictions, model, HTTP service, or case-specific splitting rules are
used. Source bytes stay untouched and are retained in the output directory.
"""

from __future__ import annotations

import argparse
from collections import Counter
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
from scripts import prepare_ooc_annotation_round as previous
from scripts import run_ooc_sealed_http as sealed_http


ROUND_ID = "ooc-event-fragment-round-v3"
MANUAL_VERSION = "ooc-manual-v3-event"
SEGMENTATION_POLICY = "punctuation-fragments-v1"
_BOUNDARY = re.compile(r"[，,。；;！？!?]+[\u201d\u2019」』》】）)]*")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+\S")


class AnnotationRepairError(ValueError):
    """Stable errors never include story contents or secret material."""


def fragment_document(payload: bytes) -> tuple[bytes, list[dict[str, int]]]:
    """Insert LF only, keeping a reversible Unicode-codepoint source map.

    Apply exactly the same punctuation rule to every ordinary line of every
    supplied document. Fragment boundaries are not events or evidence labels.
    Markdown headings remain intact; existing blank lines and final LF remain.
    """
    if b"\r" in payload:
        raise AnnotationRepairError("source_requires_lf")
    text = payload.decode("utf-8")
    output: list[str] = []
    positions: list[dict[str, int]] = []
    new_line = 1
    for source_line, raw in enumerate(text.splitlines(keepends=True), start=1):
        has_lf = raw.endswith("\n")
        original = raw[:-1] if has_lf else raw
        # Embedded Unicode line separators are not accepted as a different
        # line system: the verified source catalogue is LF-only.
        if "\n" in original or any(value in original for value in "\v\f\x1c\x1d\x1e\x85\u2028\u2029"):
            raise AnnotationRepairError("unsupported_line_separator")
        boundaries = [] if _HEADING.match(original) else [
            match.end() for match in _BOUNDARY.finditer(original)
            if match.end() < len(original)
        ]
        cuts = [0, *boundaries, len(original)]
        for index, (start, end) in enumerate(zip(cuts, cuts[1:])):
            fragment = original[start:end]
            positions.append({"line_number": new_line,
                              "source_line_number": source_line,
                              "source_char_start": start,
                              "source_char_end": end})
            output.append(fragment)
            if index < len(cuts) - 2 or has_lf:
                output.append("\n")
            new_line += 1
    result = "".join(output).encode("utf-8")
    if result.replace(b"\n", b"") != payload.replace(b"\n", b""):
        raise AnnotationRepairError("source_characters_changed")
    original_lines = text.splitlines()
    new_lines = result.decode("utf-8").splitlines()
    if len(positions) != len(new_lines):
        raise AnnotationRepairError("fragment_line_mapping_invalid")
    for row, fragment in zip(positions, new_lines):
        original = original_lines[row["source_line_number"] - 1]
        if original[row["source_char_start"]:row["source_char_end"]] != fragment:
            raise AnnotationRepairError("fragment_source_slice_mismatch")
    return result, positions


def _output_path(output_root: Path, allowed_roots: Sequence[Path] | None) -> Path:
    raw = str(output_root)
    if "://" in raw or raw.startswith(("\\\\", "//")):
        raise AnnotationRepairError("output_must_be_local")
    if workflow._path_chain_has_reparse_point(output_root):
        raise AnnotationRepairError("output_reparse_path_forbidden")
    result = output_root.parent.resolve(strict=True) / output_root.name
    roots = allowed_roots or (Path.home() / "Desktop" / "doc", Path.home() / "Desktop" / "dev")
    if not any(result.is_relative_to(root.resolve(strict=True))
               and result != root.resolve(strict=True) for root in roots):
        raise AnnotationRepairError("output_must_stay_under_desktop_doc_or_dev")
    if result.exists():
        raise AnnotationRepairError("output_already_exists")
    return result


def repair_annotation_round(
    reviewer_a: Path, reviewer_b: Path, output_root: Path, *,
    allowed_output_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    a_bytes, source_a = previous._read_page(reviewer_a)
    b_bytes, source_b = previous._read_page(reviewer_b)
    ignore = {"reviewer_id", "previous_round"}
    if source_a["reviewer_id"] == source_b["reviewer_id"]:
        raise AnnotationRepairError("reviewer_identities_must_differ")
    if ({key: value for key, value in source_a.items() if key not in ignore}
            != {key: value for key, value in source_b.items() if key not in ignore}):
        raise AnnotationRepairError("reviewer_public_inputs_differ")
    if source_a.get("manual_version") != previous.MANUAL_VERSION:
        raise AnnotationRepairError("source_requires_v2_choice_manual")
    if source_a.get("assessment_policy") is not None:
        raise AnnotationRepairError("source_must_be_legacy_policy")
    original, old_setup, old_freeze, old_payloads = previous.reconstruct_source(source_a)
    if (len(original.cases) != 18 or old_freeze.group_count != 6
            or set(Counter(case.group_id for case in original.cases).values()) != {3}
            or Counter(case["dimension"] for case in source_a["cases"])
            != Counter({dimension: 3 for dimension in previous.DIMENSIONS})):
        raise AnnotationRepairError("source_requires_balanced_eighteen_cases")
    if original.assessment_policy is not None:
        raise AnnotationRepairError("source_must_be_legacy_policy")

    new_payloads: dict[str, bytes] = {}
    source_positions: dict[str, dict[str, Any]] = {}
    new_documents: dict[str, contract.DocumentSnapshot] = {}
    anchors_by_document: dict[str, tuple[contract.EvidenceAnchor, ...]] = {}
    for case in original.cases:
        for document in case.documents:
            if document.document_id in new_documents:
                continue
            reformatted, mapping = fragment_document(old_payloads[document.path])
            new_document_id = str(uuid4())
            new_path = f"cases/{case.group_id}/{new_document_id}.md"
            new_payloads[new_path] = reformatted
            new_documents[document.document_id] = contract.DocumentSnapshot(
                document_id=new_document_id, path=new_path, version=1,
                sha256=hashlib.sha256(reformatted).hexdigest(),
            )
            source_positions[new_document_id] = {
                "source_document_id": document.document_id,
                "source_version": document.version, "source_sha256": document.sha256,
                "offset_unit": "unicode_codepoint", "fragments": mapping,
            }
            anchors_by_document[document.document_id] = tuple(
                contract.EvidenceAnchor(evidence_id=str(uuid4()), document_id=new_document_id,
                                        line_start=index, line_end=index)
                for index, line in enumerate(reformatted.decode("utf-8").splitlines(), start=1)
                if line.strip()
            )
    new_case_rows: list[contract.PublicInputCase] = []
    new_setup_rows: list[closed.PublicAuthorSetupCase] = []
    setup_by_case = {row.case_id: row for row in old_setup.cases}
    for case in original.cases:
        old_case_setup = setup_by_case[case.case_id]
        selector = old_case_setup.candidate_selector
        selector_document_id = new_documents[selector.source_document_id].document_id
        affected = [row for row in source_positions[selector_document_id]["fragments"]
                    if selector.source_line_start <= row["source_line_number"] <= selector.source_line_end]
        if not affected:
            raise AnnotationRepairError("selector_source_map_missing")
        new_selector = selector.model_copy(update={
            "source_document_id": selector_document_id,
            "source_line_start": min(row["line_number"] for row in affected),
            "source_line_end": max(row["line_number"] for row in affected),
        })
        new_setup_case = old_case_setup.model_copy(update={
            "candidate_selector": new_selector,
            "target_document_id": new_documents[old_case_setup.target_document_id].document_id,
            "documents": tuple(row.model_copy(update={
                "document_id": new_documents[row.document_id].document_id,
            }) for row in old_case_setup.documents),
        })
        new_setup_rows.append(new_setup_case)
        axis_payload = json.dumps({
            "candidate_selector": new_selector.model_dump(mode="json"),
            "author_axis": (old_case_setup.author_axis.model_dump(mode="json")
                            if old_case_setup.author_axis else None),
        }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        new_case_rows.append(contract.PublicInputCase(
            case_id=case.case_id, split_id=case.split_id, group_id=case.group_id,
            world_id=case.world_id,
            documents=tuple(new_documents[document.document_id] for document in case.documents),
            axis=contract.AxisSnapshot(axis_id=str(uuid4()), version=1,
                                       sha256=hashlib.sha256(axis_payload).hexdigest()),
            evidence_catalog=tuple(anchor for document in case.documents
                                   for anchor in anchors_by_document[document.document_id]),
        ))
    public = contract.PublicInputBundle(dataset_id=str(uuid4()),
                                        assessment_policy=contract.EVENT_ASSESSMENT_POLICY,
                                        cases=tuple(new_case_rows))
    setup = closed.PublicAuthorSetupBundle(dataset_id=public.dataset_id,
                                           public_input_sha256=contract.canonical_sha256(public),
                                           cases=tuple(new_setup_rows))
    freeze = closed.ExecutionFreeze(dataset_id=public.dataset_id,
                                    public_input_sha256=contract.canonical_sha256(public),
                                    author_setup_sha256=contract.canonical_sha256(setup),
                                    case_count=18, group_count=6)
    closed.validate_execution_freeze_binding(public, setup, freeze)
    output = _output_path(output_root, allowed_output_roots)
    if reviewer_a.read_bytes() != a_bytes or reviewer_b.read_bytes() != b_bytes:
        raise AnnotationRepairError("source_pages_changed_during_preflight")
    output.mkdir()
    previous._write_bundle(output / "source-18", original, old_setup, old_freeze, old_payloads)
    for name, payload in (("reviewer-a-v1.html", a_bytes), ("reviewer-b-v1.html", b_bytes)):
        with (output / "source-18" / name).open("xb") as target:
            target.write(payload)
    rebuilt = page.build_page_data(original, output / "source-18", old_setup, old_freeze,
                                   reviewer_id=source_a["reviewer_id"],
                                   manual_version=source_a["manual_version"])
    if any(rebuilt[key] != source_a.get(key) for key in rebuilt):
        raise AnnotationRepairError("source_page_presentation_not_reproducible")
    previous._write_bundle(output / "bundle", public, setup, freeze, new_payloads)
    # Purely local contract/byte preflight; this does not contact the API.
    # Fresh evaluation projects require newly frozen documents at version 1.
    try:
        sealed_http.load_public_bundle(output / "bundle")
    except (ValueError, sealed_http.RunnerFailure):
        raise AnnotationRepairError("repaired_bundle_local_preflight_failed") from None
    round_meta = {
        "round_id": ROUND_ID, "case_count": 18, "group_count": 6,
        "source_case_count": 18, "assessment_policy": contract.EVENT_ASSESSMENT_POLICY,
        "segmentation_policy": SEGMENTATION_POLICY,
        "source_public_input_sha256": contract.canonical_sha256(original),
        "human_reconfirmation_required": True, "automatic_answer_migration": False,
        "dimension_counts": dict(Counter(row["dimension"] for row in source_a["cases"])),
    }
    for name, source in (("reviewer-a-v1.html", source_a), ("reviewer-b-v1.html", source_b)):
        data = page.build_page_data(public, output / "bundle", setup, freeze,
                                    reviewer_id=source["reviewer_id"], manual_version=MANUAL_VERSION)
        data["assessment_policy"] = contract.EVENT_ASSESSMENT_POLICY
        data["annotation_round"] = round_meta
        data["source_position_map"] = source_positions
        with (output / name).open("x", encoding="utf-8", newline="\n") as target:
            target.write(page.render_annotation_page(data))
    previous._write_json(output / "round-manifest.json", {
        "schema_version": "loreguard-ooc-fragment-round-manifest-v1", **round_meta,
        "dataset_id": public.dataset_id, "manual_version": MANUAL_VERSION,
        "public_input_sha256": contract.canonical_sha256(public),
        "author_setup_sha256": contract.canonical_sha256(setup),
        "execution_freeze_sha256": contract.canonical_sha256(freeze),
        "source_dataset_id": original.dataset_id,
        "source_author_setup_sha256": contract.canonical_sha256(old_setup),
        "source_execution_freeze_sha256": contract.canonical_sha256(old_freeze),
        "source_page_sha256": {"A": hashlib.sha256(a_bytes).hexdigest(),
                               "B": hashlib.sha256(b_bytes).hexdigest()},
        "case_ids_in_unchanged_order": [case.case_id for case in public.cases],
        "source_position_map": source_positions,
        "statistics_scope": "Independent 18-case/6-story v3 pilot. No inherited gold, predictions or quality claims.",
    })
    with (output / "批次说明.md").open("x", encoding="utf-8", newline="\n") as target:
        target.write(
            "# 18 题修复版：按原文片段记录事件\n\n"
            "仍是同一组18题，原文只按统一标点规则新增换行，其他字符一个不改。"
            "片段不是独立事件；同一件事的多个片段仍选同一事情。没有预选答案。\n\n"
            "单次明确且材料完整、范围一致、没有适用解释的反向行为，可以由真人判为正式冲突；"
            "有歧义时保留待复核，多次仅增强证据，不是硬门槛。选择正式冲突还须主动完成核对。\n\n"
            "原18题页面和冻结材料逐字节保存在source-18。新版使用独立冻结与v3手册，"
            "不自动带入旧答案、结论或声明，不删除浏览器旧进度。请先保留原进度，"
            "再在新版按片段重新核对；不要把旧36题/18题结果与本版混算。\n"
        )
    if reviewer_a.read_bytes() != a_bytes or reviewer_b.read_bytes() != b_bytes:
        raise AnnotationRepairError("source_pages_changed_during_generation")
    return {"status": "created", "output": str(output), "case_count": 18,
            "group_count": 6, "manual_version": MANUAL_VERSION,
            "assessment_policy": contract.EVENT_ASSESSMENT_POLICY,
            "source_files_unchanged": True, "original_characters_preserved": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = repair_annotation_round(args.reviewer_a, args.reviewer_b, args.output_root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        code = str(exc) if isinstance(exc, AnnotationRepairError) else type(exc).__name__
        print(json.dumps({"status": "failed", "code": code}))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
