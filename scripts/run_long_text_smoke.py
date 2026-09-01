from __future__ import annotations

import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import AnalysisPipeline, BaselineExtractor, DocumentInput


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_root = root / "data" / "long-text-smoke"
    paths = sorted(data_root.glob("*.md"))
    documents = [DocumentInput(id=path.stem, name=path.name, content=path.read_text(encoding="utf-8")) for path in paths]
    started = perf_counter()
    result = AnalysisPipeline(extractor=BaselineExtractor()).run(documents)
    elapsed_ms = (perf_counter() - started) * 1000
    payload = {
        "dataset_kind": "generated long-text smoke",
        "human_annotated": False,
        "model_enabled": False,
        "document_count": len(documents),
        "character_count": sum(len(document.content) for document in documents),
        "elapsed_ms": round(elapsed_ms, 3),
        "record_count": len(result.directives),
        "issue_count": len(result.issues),
        "issue_categories": sorted(issue.category.value for issue in result.issues),
        "chunking": result.diagnostics["chunking"],
        "alias_summary": {
            "declaration_count": result.diagnostics["aliases"]["declaration_count"],
            "trace_count": result.diagnostics["aliases"]["trace_count"],
        },
        "retrieval_summary": {
            "candidate_count": result.diagnostics["retrieval"]["candidate_count"],
            "consumed_count": result.diagnostics["retrieval"]["consumed_count"],
        },
        "evidence": [
            [{"document": span.document_name, "line": span.line_start} for span in issue.evidence]
            for issue in result.issues
        ],
        "limitations": "Generated smoke only; no model/embedding accuracy or production latency claim.",
    }
    artifact = root / "artifacts" / "long-text-smoke-report.json"
    artifact.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"chars={payload['character_count']} issues={payload['issue_count']} chunks={payload['chunking']['total_chunks']} elapsed_ms={payload['elapsed_ms']}")


if __name__ == "__main__":
    main()
