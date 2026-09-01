from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.natural_evaluation import run_natural_evaluation


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = run_natural_evaluation("test")
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    report_path = artifacts / "natural-evaluation-report.json"
    error_path = artifacts / "natural-evaluation-errors.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    error_path.write_text(
        json.dumps(
            {
                "benchmark": report["benchmark"],
                "error_summary": report["error_summary"],
                "errors": report["errors"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics = report["metrics"]["overall"]
    print(report_path)
    print(error_path)
    print(
        f"cases={report['benchmark']['case_count']} precision={metrics['precision']:.3f} "
        f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} "
        f"evidence={metrics['evidence_pair_line_hit_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
