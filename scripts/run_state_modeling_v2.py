from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.natural_evaluation import run_natural_evaluation
from app.candidate_normalizer import DeterministicCandidateNormalizer
from app.pipeline import AnalysisPipeline, BaselineExtractor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("before", "after"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifact_root = root / "artifacts" / "state-modeling-v2"
    artifact_root.mkdir(parents=True, exist_ok=True)
    datasets = {
        "natural-dev": (root / "data" / "evaluation-natural", "dev"),
        "natural-test": (root / "data" / "evaluation-natural", "test"),
        "challenge-v2": (root / "data" / "evaluation-challenge-v2", "test"),
    }
    pipeline = AnalysisPipeline(
        extractor=BaselineExtractor(),
        normalizer=DeterministicCandidateNormalizer(enable_state_modeling=args.phase == "after"),
    )
    for label, (dataset_root, split) in datasets.items():
        report = run_natural_evaluation(split, dataset_root, pipeline=pipeline)
        report["benchmark"]["evaluation_role"] = (
            "known test regression reference"
            if label == "natural-test"
            else "developer-visible synthetic challenge; not a blind test"
        )
        report_path = artifact_root / f"{args.phase}-{label}.json"
        error_path = artifact_root / f"{args.phase}-{label}-errors.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        error_path.write_text(json.dumps({
            "benchmark": report["benchmark"],
            "error_summary": report["error_summary"],
            "errors": report["errors"],
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        metrics = report["metrics"]["overall"]
        print(
            f"{args.phase} {label}: sha256={report['benchmark']['dataset_sha256']} "
            f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
