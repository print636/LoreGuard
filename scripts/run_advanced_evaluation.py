from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import AnalysisPipeline, DocumentInput


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data" / "advanced"
    documents = [
        DocumentInput(id=path.stem, name=path.name, content=path.read_text(encoding="utf-8"))
        for path in sorted(data_dir.glob("*.md"))
    ]
    expected = json.loads((data_dir / "expected.json").read_text(encoding="utf-8"))
    events: list[dict] = []
    result = AnalysisPipeline().run(
        documents,
        on_stage=lambda stage, progress, message: events.append({"stage": stage, "progress": progress, "message": message}),
    )
    expected_categories = {item["category"] for item in expected["expected_issue_families"]}
    detected_categories = {issue.category.value for issue in result.issues}
    report = {
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
    target = root / "artifacts" / "advanced-evaluation.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("model_used", "prompt_tokens", "completion_tokens", "record_count", "issue_count", "category_recall", "missing_categories", "unexpected_categories", "warnings")}, ensure_ascii=False, indent=2))
    print(target)
    return 0 if result.model_used else 2


if __name__ == "__main__":
    raise SystemExit(main())
