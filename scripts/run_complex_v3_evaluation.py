from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.candidate_normalizer import DeterministicCandidateNormalizer
from app.natural_evaluation import run_natural_evaluation
from app.pipeline import AnalysisPipeline, BaselineExtractor


def build_report(dataset_root: Path) -> dict:
    manifest = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
    pipeline = AnalysisPipeline(
        extractor=BaselineExtractor(),
        normalizer=DeterministicCandidateNormalizer(enable_state_modeling=True),
    )
    report = run_natural_evaluation("test", dataset_root, pipeline=pipeline)
    report["benchmark"].update(
        {
            "name": manifest["name"],
            "version": manifest["version"],
            "dataset_kind": manifest["dataset_kind"],
            "human_annotated": False,
            "developer_visible": True,
            "blind_test": False,
            "evaluation_role": "developer-authored, developer-visible acceptance regression",
            "model_enabled": False,
            "provider_calls": 0,
            "extractor": "BaselineExtractor + DeterministicCandidateNormalizer",
            "expected_issue_count": manifest["expected_issue_count"],
            "dataset_sha256": manifest["sha256"],
            "evidence_contract": manifest["evidence_contract"],
            "limitations": manifest["limitations"],
        }
    )
    report["manifest"] = manifest
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score the fixed complex-v3 acceptance set without a model provider."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional report path; defaults to artifacts/complex-v3-evaluation.json.",
    )
    parser.add_argument(
        "--require-perfect",
        action="store_true",
        help="Exit 1 unless TP/FP/FN and exact evidence are all perfect.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    dataset_root = root / "data" / "evaluation-complex-v3"
    output = args.output or root / "artifacts" / "complex-v3-evaluation.json"
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    report = build_report(dataset_root)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics = report["metrics"]["overall"]
    summary = {
        "dataset": report["benchmark"]["name"],
        "version": report["benchmark"]["version"],
        "case_count": report["benchmark"]["case_count"],
        "expected_issue_count": report["benchmark"]["expected_issue_count"],
        "model_enabled": report["benchmark"]["model_enabled"],
        "provider_calls": report["benchmark"]["provider_calls"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "precision": round(metrics["precision"], 4),
        "recall": round(metrics["recall"], 4),
        "f1": round(metrics["f1"], 4),
        "evidence_pair_line_hit_rate": round(metrics["evidence_pair_line_hit_rate"], 4),
        "report": str(output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    perfect = (
        metrics["fp"] == 0
        and metrics["fn"] == 0
        and metrics["evidence_pair_line_hit_rate"] == 1.0
    )
    return 1 if args.require_perfect and not perfect else 0


if __name__ == "__main__":
    raise SystemExit(main())
