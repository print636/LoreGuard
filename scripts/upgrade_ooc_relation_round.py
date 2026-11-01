"""Upgrade answer-free V3 handouts to explicit per-target evidence directions.

No segmentation, story rewriting, answer migration, Provider or HTTP request.
The new public policy is independently frozen; the original pages and bundle
are retained byte-for-byte. This tool never discovers human answers or gold.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
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
from scripts import repair_ooc_annotation_round as repair
from scripts import run_ooc_sealed_http as sealed_http


ROUND_ID = "ooc-directional-relation-round-v4"
MANUAL_VERSION = workflow.RELATION_MANUAL_VERSION


class RelationUpgradeError(ValueError):
    """Stable failures never contain the source story or personal answers."""


def upgrade_relation_round(
    reviewer_a: Path, reviewer_b: Path, output_root: Path, *,
    allowed_output_roots: Sequence[Path] | None = None,
) -> dict[str, Any]:
    a_bytes, source_a = previous._read_page(reviewer_a)
    b_bytes, source_b = previous._read_page(reviewer_b)
    if source_a["reviewer_id"] == source_b["reviewer_id"]:
        raise RelationUpgradeError("reviewer_identities_must_differ")
    if ({key: value for key, value in source_a.items() if key != "reviewer_id"}
            != {key: value for key, value in source_b.items() if key != "reviewer_id"}):
        raise RelationUpgradeError("reviewer_public_inputs_differ")
    if (source_a.get("manual_version") != workflow.EVENT_MANUAL_VERSION
            or source_a.get("assessment_policy") != contract.EVENT_ASSESSMENT_POLICY
            or source_a.get("relation_policy") is not None):
        raise RelationUpgradeError("source_requires_v3_event_handouts")
    original, old_setup, old_freeze, payloads = previous.reconstruct_source(source_a)
    if (len(original.cases) != 18 or old_freeze.group_count != 6
            or set(Counter(case.group_id for case in original.cases).values()) != {3}
            or Counter(case["dimension"] for case in source_a["cases"])
            != Counter({dimension: 3 for dimension in previous.DIMENSIONS})):
        raise RelationUpgradeError("source_requires_balanced_eighteen_cases")
    public = contract.PublicInputBundle(
        dataset_id=str(uuid4()), cases=original.cases,
        assessment_policy=contract.EVENT_ASSESSMENT_POLICY,
        relation_policy=contract.DIRECTIONAL_RELATION_POLICY,
    )
    setup = closed.PublicAuthorSetupBundle(
        dataset_id=public.dataset_id, public_input_sha256=contract.canonical_sha256(public),
        cases=old_setup.cases,
    )
    freeze = closed.ExecutionFreeze(
        dataset_id=public.dataset_id, public_input_sha256=contract.canonical_sha256(public),
        author_setup_sha256=contract.canonical_sha256(setup), case_count=18, group_count=6,
    )
    closed.validate_execution_freeze_binding(public, setup, freeze)
    output = repair._output_path(output_root, allowed_output_roots)
    if reviewer_a.read_bytes() != a_bytes or reviewer_b.read_bytes() != b_bytes:
        raise RelationUpgradeError("source_pages_changed_during_preflight")
    output.mkdir()
    previous._write_bundle(output / "source-v3", original, old_setup, old_freeze, payloads)
    for name, raw in (("reviewer-a-v1.html", a_bytes), ("reviewer-b-v1.html", b_bytes)):
        with (output / "source-v3" / name).open("xb") as target:
            target.write(raw)
    rebuilt = page.build_page_data(
        original, output / "source-v3", old_setup, old_freeze,
        reviewer_id=source_a["reviewer_id"], manual_version=source_a["manual_version"],
    )
    if any(value != source_a.get(key) for key, value in rebuilt.items()):
        raise RelationUpgradeError("source_page_presentation_not_reproducible")
    previous._write_bundle(output / "bundle", public, setup, freeze, payloads)
    try:
        sealed_http.load_public_bundle(output / "bundle")
    except (ValueError, sealed_http.RunnerFailure):
        raise RelationUpgradeError("upgraded_bundle_local_preflight_failed") from None
    metadata = {
        **source_a.get("annotation_round", {}),
        "round_id": ROUND_ID, "case_count": 18, "group_count": 6,
        "relation_policy": contract.DIRECTIONAL_RELATION_POLICY,
        "source_public_input_sha256": contract.canonical_sha256(original),
        "human_reconfirmation_required": True, "automatic_answer_migration": False,
    }
    for name, source in (("reviewer-a-v1.html", source_a), ("reviewer-b-v1.html", source_b)):
        data = page.build_page_data(
            public, output / "bundle", setup, freeze,
            reviewer_id=source["reviewer_id"], manual_version=MANUAL_VERSION,
        )
        data["relation_policy"] = contract.DIRECTIONAL_RELATION_POLICY
        data["annotation_round"] = metadata
        if "source_position_map" in source:
            data["source_position_map"] = source["source_position_map"]
        with (output / name).open("x", encoding="utf-8", newline="\n") as target:
            target.write(page.render_annotation_page(data))
    previous._write_json(output / "round-manifest.json", {
        "schema_version": "loreguard-ooc-relation-upgrade-manifest-v1", **metadata,
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
        "document_bytes_and_anchors_unchanged": True,
        "statistics_scope": "Independent 18-case / 6-story pilot; no inherited answers or quality scores.",
    })
    with (output / "批次说明.md").open("x", encoding="utf-8", newline="\n") as target:
        target.write(
            "# 18题关系方向版\n\n"
            "原文、分号句号分片、题目顺序与证据位置不变。成长与每条新稿表现的关系分别选择："
            "能解释、与表现相反、尚不能确定。同一成长可以解释一件事，同时反对另一件事。\n\n"
            "相反关系只是反向证据，不自动判定冲突；时间、人物、适用情境与人工确认仍须核对。"
            "不明确的关系保留待复核，不把反向关系保存成正向解释。\n\n"
            "原V3页面与冻结材料保留在source-v3。新版独立冻结，不自动继承旧答案、关系、声明或确认；"
            "不要删除旧文件或清理原浏览器记录。两位标注者仍须独立复核同一套18题。\n"
        )
    if reviewer_a.read_bytes() != a_bytes or reviewer_b.read_bytes() != b_bytes:
        raise RelationUpgradeError("source_pages_changed_during_generation")
    return {"status": "created", "output": str(output), "case_count": 18,
            "group_count": 6, "manual_version": MANUAL_VERSION,
            "relation_policy": contract.DIRECTIONAL_RELATION_POLICY,
            "document_bytes_and_anchors_unchanged": True, "source_files_unchanged": True}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = upgrade_relation_round(args.reviewer_a, args.reviewer_b, args.output_root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"status": "failed", "code": str(exc) if isinstance(exc, RelationUpgradeError) else type(exc).__name__}))
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
