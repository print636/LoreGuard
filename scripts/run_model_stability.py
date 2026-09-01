from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.pipeline import AnalysisPipeline, DocumentInput


EXPECTED_CATEGORIES = {
    "fact_conflict", "location_collision", "knowledge_without_acquisition",
    "item_ownership", "world_rule_conflict",
}


def load_case(case_name: str, root: Path | None = None) -> list[DocumentInput]:
    project_root = root or Path(__file__).resolve().parents[1]
    if case_name == "advanced":
        data_root = project_root / "data" / "advanced"
        paths = [data_root / "world.md", data_root / "chapter-01.md", data_root / "chapter-02.md"]
        return [DocumentInput(id=path.stem, name=path.name, content=path.read_text(encoding="utf-8")) for path in paths]
    if case_name == "long-smoke-subset":
        data_root = project_root / "data" / "long-text-smoke"
        selected = {}
        for path in sorted(data_root.glob("*.md")):
            lines = path.read_text(encoding="utf-8").splitlines()
            if path.name == "world.md":
                chosen = lines[:6]
            elif path.name == "chapter-01.md":
                chosen = [lines[0], lines[-1]]
            elif path.name == "chapter-02.md":
                chosen = [lines[0], *lines[-2:]]
            else:
                chosen = [lines[0], *lines[-3:]]
            selected[path.name] = "\n".join(chosen)
        return [DocumentInput(id=name.rsplit(".", 1)[0], name=name, content=content) for name, content in selected.items()]
    if case_name == "long-smoke-2k":
        data_root = project_root / "data" / "long-text-smoke"
        sections = []
        for path in sorted(data_root.glob("*.md")):
            sections.append(f"# {path.name}\n{path.read_text(encoding='utf-8')}")
        content = "\n\n".join(sections)[:2_000]
        return [DocumentInput(id="long-smoke-2k", name="long-smoke-2k.md", content=content)]
    raise ValueError("case must be advanced, long-smoke-subset or long-smoke-2k")


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile_value + 0.999999)))
    return ordered[index]


def load_expected_issues(case_name: str, root: Path | None = None) -> list[dict]:
    if case_name != "advanced":
        return []
    project_root = root or Path(__file__).resolve().parents[1]
    payload = json.loads(
        (project_root / "data" / "advanced" / "expected.json").read_text(
            encoding="utf-8"
        )
    )
    result = []
    for row in payload["expected_issue_families"]:
        evidence = []
        for location in row["evidence"]:
            document, line = location.rsplit(":", 1)
            evidence.append((document, int(line)))
        result.append({"category": row["category"], "evidence": tuple(sorted(evidence))})
    return result


def aggregate_runs(
    runs: list[dict], expected: set[str], expected_issues: list[dict] | None = None
) -> dict:
    appearances = Counter()
    evidence_signatures: dict[str, Counter] = defaultdict(Counter)
    complete = unexpected_runs = 0
    for run in runs:
        categories = set(run["issue_categories"])
        complete += int(expected.issubset(categories))
        unexpected_runs += int(bool(categories - expected))
        grouped_signatures: dict[str, list[tuple]] = defaultdict(list)
        for issue in run["issues"]:
            category = issue["category"]
            signature = tuple((row["document"], row["line"]) for row in issue["evidence"])
            grouped_signatures[category].append(signature)
        for category, signatures in grouped_signatures.items():
            appearances[category] += 1
            evidence_signatures[category][tuple(sorted(signatures))] += 1
    durations = [float(run["duration_ms"]) for run in runs]
    first_events = [float(run["first_progress_ms"]) for run in runs]
    per_category = {}
    for category in sorted(expected | set(appearances)):
        count = appearances[category]
        most_common = evidence_signatures[category].most_common(1)
        per_category[category] = {
            "appearance_rate": count / len(runs) if runs else 0.0,
            "evidence_pair_stability_rate": most_common[0][1] / count if count and most_common else 0.0,
        }
    issue_scoring = None
    if expected_issues:
        tp = fp = fn = perfect_runs = 0
        for run in runs:
            predictions = [
                {
                    "category": issue["category"],
                    "evidence": tuple(
                        sorted(
                            (row["document"], int(row["line"]))
                            for row in issue["evidence"]
                        )
                    ),
                }
                for issue in run["issues"]
            ]
            unmatched = set(range(len(predictions)))
            run_tp = 0
            for expected_issue in expected_issues:
                match = next(
                    (
                        index
                        for index in unmatched
                        if predictions[index] == expected_issue
                    ),
                    None,
                )
                if match is None:
                    fn += 1
                else:
                    unmatched.remove(match)
                    tp += 1
                    run_tp += 1
            fp += len(unmatched)
            perfect_runs += int(
                run_tp == len(expected_issues) and not unmatched
            )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        issue_scoring = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            ),
            "perfect_run_rate": perfect_runs / len(runs) if runs else 0.0,
        }
    return {
        "run_count": len(runs),
        "per_category": per_category,
        "complete_target_success_rate": complete / len(runs) if runs else 0.0,
        "unexpected_run_rate": unexpected_runs / len(runs) if runs else 0.0,
        "model_run_rate": sum(bool(run["model_used"]) for run in runs) / len(runs) if runs else 0.0,
        "duration_ms": {"p50": statistics.median(durations) if durations else 0.0, "p95": percentile(durations, 0.95)},
        "first_progress_ms": {"p50": statistics.median(first_events) if first_events else 0.0, "p95": percentile(first_events, 0.95)},
        "total_tokens": sum(run["total_tokens"] for run in runs),
        "exact_issue_scoring": issue_scoring,
    }


def run_stability(
    case_name: str,
    repeats: int,
    max_total_tokens: int,
    pipeline_factory: Callable[[], AnalysisPipeline] = AnalysisPipeline,
    root: Path | None = None,
) -> dict:
    documents = load_case(case_name, root)
    expected_issues = load_expected_issues(case_name, root)
    expected_categories = EXPECTED_CATEGORIES if case_name != "long-smoke-2k" else set()
    runs = []
    budget_exhausted = max_total_tokens <= 0
    for repeat in range(repeats):
        total_so_far = sum(run["total_tokens"] for run in runs)
        if total_so_far >= max_total_tokens:
            budget_exhausted = True
            break
        pipeline = pipeline_factory()
        remaining_tokens = max_total_tokens - total_so_far
        provider_settings = getattr(
            getattr(getattr(pipeline, "extractor", None), "provider", None),
            "settings",
            None,
        )
        original_run_budget = None
        if provider_settings is not None:
            original_run_budget = provider_settings.per_run_token_budget
            provider_settings.per_run_token_budget = min(
                original_run_budget, remaining_tokens
            )
        first_progress: float | None = None
        started = perf_counter()

        def on_stage(_stage: str, _progress: int, _message: str) -> None:
            nonlocal first_progress
            if first_progress is None:
                first_progress = (perf_counter() - started) * 1000

        try:
            result = pipeline.run(documents, on_stage=on_stage)
        finally:
            if provider_settings is not None and original_run_budget is not None:
                provider_settings.per_run_token_budget = original_run_budget
        duration_ms = (perf_counter() - started) * 1000
        stopped_before_model = (
            not result.model_used
            and result.prompt_tokens + result.completion_tokens == 0
            and any("Token 预算不足" in warning for warning in result.warnings)
        )
        if stopped_before_model:
            budget_exhausted = True
            break
        issues = [{
            "category": issue.category.value,
            "evidence": [{"document": span.document_name, "line": span.line_start} for span in issue.evidence],
        } for issue in result.issues]
        total_tokens = result.prompt_tokens + result.completion_tokens
        runs.append({
            "repeat": repeat + 1,
            "model_used": result.model_used,
            "duration_ms": round(duration_ms, 3),
            "first_progress_ms": round(first_progress or 0.0, 3),
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": total_tokens,
            "issue_categories": sorted(issue["category"] for issue in issues),
            "issues": issues,
            "warnings": result.warnings,
        })
        if sum(run["total_tokens"] for run in runs) >= max_total_tokens:
            budget_exhausted = repeat + 1 < repeats
            break
    total_tokens = sum(run["total_tokens"] for run in runs)
    return {
        "benchmark": {
            "name": "LoreGuard model repeat stability",
            "case": case_name,
            "requested_repeats": repeats,
            "max_total_tokens": max_total_tokens,
            "expected_usage": "Scoring metadata only; expected categories are never inserted into model prompts.",
            "secrets_or_response_bodies_recorded": False,
        },
        "budget_exhausted": budget_exhausted,
        "budget_overshoot_tokens": max(0, total_tokens - max_total_tokens),
        "runs": runs,
        "summary": aggregate_runs(runs, expected_categories, expected_issues),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("advanced", "long-smoke-subset", "long-smoke-2k"),
        default="advanced",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-total-tokens", type=int, default=50_000)
    parser.add_argument("--output", type=Path, default=Path("artifacts/model-stability-report.json"))
    parser.add_argument(
        "--rescore",
        type=Path,
        help="Recompute summary from an existing report without calling the provider.",
    )
    args = parser.parse_args()
    if args.rescore:
        report = json.loads(args.rescore.read_text(encoding="utf-8"))
        case_name = report["benchmark"]["case"]
        expected_categories = EXPECTED_CATEGORIES if case_name != "long-smoke-2k" else set()
        report["summary"] = aggregate_runs(
            report["runs"],
            expected_categories,
            load_expected_issues(case_name),
        )
        report["benchmark"]["rescored_without_provider_call"] = True
    else:
        report = run_stability(args.case, args.repeats, args.max_total_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"runs={len(report['runs'])} tokens={report['summary']['total_tokens']} budget_exhausted={report['budget_exhausted']}")
    if not report["runs"] or report["summary"]["model_run_rate"] == 0:
        print("No model-backed run completed; check provider configuration or Token budget.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
