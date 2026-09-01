import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.evaluation import run_evaluation

result = run_evaluation()
target = Path(__file__).resolve().parents[1] / "artifacts" / "directive-regression-report.json"
target.parent.mkdir(exist_ok=True)
payload = {
    "benchmark_kind": "rule-engine synthetic directive regression",
    "natural_language_evaluation": False,
    "human_annotated": False,
    "warning": "These explicit @directive cases test deterministic rule wiring only; they are not open-text precision/recall.",
    "metrics": result.model_dump(mode="json"),
}
target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(target)
print(f"samples={result.sample_count} precision={result.precision:.3f} recall={result.recall:.3f} evidence_hit={result.evidence_hit_rate:.3f}")
