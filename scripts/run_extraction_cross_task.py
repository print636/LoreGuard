"""Read-only live-model transfer check on the frozen Investigator fixture.

Only source documents enter AnalysisPipeline. Labels are opened for scoring
after each run; API credentials, raw completions and source text never enter
the emitted JSON. This is cross-task diagnostic coverage, not a blind benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "app" / "model_extractor.py").is_file():
    ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.account_provider import runtime_for_analysis_run
from app.db import AnalysisRunRow, SessionLocal
from app.model_extractor import ModelEnhancedExtractor
from app.pipeline import AnalysisPipeline, DocumentInput
from app.provider import OpenAICompatibleProvider


FIXTURE = ROOT / "data" / "evaluation" / "evidence_investigator_live"


def _selected_cases(manifest: dict, split: str) -> list[dict]:
    if manifest.get("status") != "ready_unscored":
        raise ValueError("fixture is not frozen and ready")
    cases = [row for row in manifest["cases"] if row["split"] == split]
    expected_count = manifest["splits"][split]["case_count"]
    if len(cases) != expected_count or len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("fixture split count or IDs do not match manifest")
    return cases


def _documents(case: dict) -> tuple[list[DocumentInput], dict[str, str]]:
    documents: list[DocumentInput] = []
    names: dict[str, str] = {}
    for source in case["sources"]:
        path = (FIXTURE / source["path"]).resolve()
        if not path.is_relative_to(FIXTURE.resolve()) or not path.is_file():
            raise ValueError("fixture source path is invalid")
        name = path.name
        names[source["document_id"]] = name
        documents.append(DocumentInput(
            id=f"{case['case_id']}:{source['document_id']}",
            name=name,
            content=path.read_text(encoding="utf-8"),
            role=source["role"],
            scope=source.get("story_scope", "global"),
        ))
    if len(set(names.values())) != len(names):
        raise ValueError("fixture document names are not unique")
    return documents, names


def _expected_evidence(case: dict, names: dict[str, str]) -> set[tuple[str, int]]:
    return {
        (names[item["document_id"]], int(item["lines"][0]))
        for item in case["expected"]["allowed_evidence"]
    }


def _score_case(case: dict, issues: list, names: dict[str, str]) -> dict:
    expected = case["expected"]
    target_category = expected["issue_category"]
    target_evidence = _expected_evidence(case, names)
    matching = [
        issue for issue in issues
        if issue.category.value == target_category
        and {(span.document_name, span.line_start) for span in issue.evidence}
        == target_evidence
    ]
    positive = expected["decision"] == "added_issue"
    if expected["decision"] not in {"added_issue", "abstain"}:
        raise ValueError("fixture has an unsupported expected decision")
    return {
        "expected_decision": expected["decision"],
        "expected_category": target_category,
        "exact_evidence_hit": bool(matching) if positive else None,
        "clean_abstain": not issues if not positive else None,
        "issue_count": len(issues),
        "unexpected_issue_count": len(issues) - min(len(matching), 1)
        if positive else len(issues),
        "issue_categories": sorted(issue.category.value for issue in issues),
    }


def evaluate(split: str, source_run_id: str, max_total_tokens: int) -> dict:
    manifest = json.loads((FIXTURE / "manifest.json").read_text(encoding="utf-8"))
    cases = _selected_cases(manifest, split)
    with SessionLocal() as db:
        source_run = db.get(AnalysisRunRow, source_run_id)
        if source_run is None:
            raise ValueError("provider source run does not exist")
        runtime = runtime_for_analysis_run(db, source_run)
    if not runtime.available or not runtime.settings.enable_model_extraction:
        raise ValueError("frozen provider is unavailable for extraction")

    rows: list[dict] = []
    charged_total = 0
    for case in cases:
        remaining = max_total_tokens - charged_total
        if remaining < 3000:
            break
        documents, names = _documents(case)
        settings = runtime.settings.model_copy(update={
            "per_run_token_budget": min(runtime.settings.per_run_token_budget, remaining)
        })
        extractor = ModelEnhancedExtractor(OpenAICompatibleProvider(settings))
        started = perf_counter()
        try:
            result = AnalysisPipeline(extractor=extractor).run(documents)
        except Exception as exc:
            # Never surface upstream exception messages: relays sometimes put
            # request or response material into them.
            rows.append({"case_id": case["case_id"], "error_type": type(exc).__name__})
            print(f"{case['case_id']}: error {type(exc).__name__}", file=sys.stderr)
            break
        charged = extractor.conservative_run_token_debit()
        charged_total += charged
        model = result.diagnostics.get("model", {})
        row = {
            "case_id": case["case_id"],
            **_score_case(case, result.issues, names),
            "model_used": result.model_used,
            "invalid_records": model.get("invalid_records"),
            "unresolved_invalid_records": model.get("unresolved_invalid_records"),
            "failed_chunks": model.get("failed_chunks"),
            "record_rejections": model.get("record_rejections", {}),
            "reason_codes": model.get("reason_codes", []),
            "conservative_tokens": charged,
            "duration_ms": round((perf_counter() - started) * 1000),
        }
        rows.append(row)
        print(
            f"{case['case_id']}: issues={row['issue_count']} "
            f"invalid={row['invalid_records']} charged={charged}",
            file=sys.stderr,
        )

    return {
        "scope": "cross-task developer-authored transfer diagnostic; not blind accuracy",
        "fixture": manifest["dataset_id"],
        "split": split,
        "selected_cases": len(cases),
        "attempted_cases": len(rows),
        "completed_cases": sum("error_type" not in row for row in rows),
        "positive_exact_hits": sum(row.get("exact_evidence_hit") is True for row in rows),
        "negative_clean_abstains": sum(row.get("clean_abstain") is True for row in rows),
        "invalid_records": sum(row.get("invalid_records") or 0 for row in rows),
        "conservative_tokens": charged_total,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "holdout"), required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--max-total-tokens", type=int, default=200_000)
    args = parser.parse_args()
    if args.max_total_tokens < 3000:
        parser.error("max-total-tokens must be at least 3000")
    try:
        report = evaluate(args.split, args.source_run_id, args.max_total_tokens)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"error_type": type(exc).__name__}), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0 if report["completed_cases"] == report["selected_cases"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
