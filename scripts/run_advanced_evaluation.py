from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import AnalysisPipeline, DocumentInput
from app.config import get_settings
from scripts.run_complex_model_evaluation import _execution_classification


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Original advanced-scenario model smoke; not a blind accuracy test")
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "advanced-evaluation.json")
    args = parser.parse_args()
    data_dir = root / "data" / "advanced"
    documents = [
        DocumentInput(id=path.stem, name=path.name, content=path.read_text(encoding="utf-8"))
        for path in sorted(data_dir.glob("*.md"))
    ]
    expected = json.loads((data_dir / "expected.json").read_text(encoding="utf-8"))
    events: list[dict] = []
    started = perf_counter()
    result = AnalysisPipeline().run(
        documents,
        on_stage=lambda stage, progress, message: events.append({"stage": stage, "progress": progress, "message": message}),
    )
    expected_categories = {item["category"] for item in expected["expected_issue_families"]}
    detected_categories = {issue.category.value for issue in result.issues}
    participating, full_model = _execution_classification(result.diagnostics.get("model"))
    report = {
        "model": get_settings().openai_model,
        "duration_ms": round((perf_counter() - started) * 1000, 3),
        "full_model_execution": full_model,
        "diagnostics": result.diagnostics,
        "scenario": expected["scenario"],
        "model_used": result.model_used,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "record_count": len(result.directives),
        "issue_count": len(result.issues),
        "expected_categories": sorted(expected_categories),
        "detected_categories": sorted(detected_categories),
        "category_recall": len(expected_categories & detected_categories) / len(expected_categories),
        "missing_categories": sorted(expected_categories - detected_categories),
        "unexpected_categories": sorted(detected_categories - expected_categories),
        "warnings": result.warnings,
        "events": events,
        "records": [item.model_dump(mode="json") for item in result.directives],
        "issues": [item.model_dump(mode="json") for item in result.issues],
    }
    target = args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model", "duration_ms", "model_used", "full_model_execution", "prompt_tokens", "completion_tokens", "record_count", "issue_count", "category_recall", "missing_categories", "unexpected_categories", "warnings")}, ensure_ascii=False, indent=2))
    print(target)
    return 0 if participating and full_model else 2


if __name__ == "__main__":
    raise SystemExit(main())
