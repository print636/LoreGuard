from __future__ import annotations

import argparse
from datetime import UTC, datetime
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model_extractor import ModelEnhancedExtractor
from app.domain import IssueCategory
from app.natural_evaluation import (
    NaturalEvaluationCase,
    aggregate_results,
    load_cases,
    score_case,
)
from app.pipeline import AnalysisPipeline, DocumentInput
from app.provider import OpenAICompatibleProvider


PILOT_CASE_IDS = (
    "complex-v3-integrated-conflicts",
    "complex-v3-fact-state-transition",
    "complex-v3-location-valid-transport",
    "complex-v3-knowledge-incomplete-ledger",
    "complex-v3-item-authorized-transfer",
    "complex-v3-rule-actor-exception",
)


class CountingProvider:
    """Count logical completion requests without recording prompts or responses."""

    def __init__(self, delegate: OpenAICompatibleProvider) -> None:
        self.delegate = delegate
        self.settings = delegate.settings
        self.requested = 0
        self.succeeded = 0
        self.failed = 0
        self.last_error_type: str | None = None

    @property
    def configured(self) -> bool:
        return self.delegate.configured

    def complete(self, system: str, user: str):
        self.requested += 1
        try:
            result = self.delegate.complete(system, user)
        except Exception as exc:
            self.failed += 1
            self.last_error_type = type(exc).__name__
            raise
        self.succeeded += 1
        return result


def model_pipeline() -> AnalysisPipeline:
    provider = CountingProvider(OpenAICompatibleProvider())
    return AnalysisPipeline(extractor=ModelEnhancedExtractor(provider=provider))


def select_cases(
    cases: list[NaturalEvaluationCase], suite: str
) -> list[NaturalEvaluationCase]:
    if suite == "full":
        return cases
    by_id = {case.case_id: case for case in cases}
    missing = [case_id for case_id in PILOT_CASE_IDS if case_id not in by_id]
    if missing:
        raise ValueError(f"pilot cases missing from dataset: {', '.join(missing)}")
    return [by_id[case_id] for case_id in PILOT_CASE_IDS]


def _predictions(result) -> list[dict]:
    return [
        {
            "category": issue.category.value,
            "title": issue.title,
            "evidence": [
                {"document": span.document_name, "line": span.line_start}
                for span in issue.evidence
            ],
            "metadata": issue.metadata,
        }
        for issue in result.issues
    ]


def _provider_stats(pipeline) -> dict:
    provider = getattr(getattr(pipeline, "extractor", None), "provider", None)
    settings = getattr(provider, "settings", None)
    return {
        "instrumented": hasattr(provider, "requested"),
        "model": getattr(settings, "openai_model", None),
        "requested": int(getattr(provider, "requested", 0)),
        "succeeded": int(getattr(provider, "succeeded", 0)),
        "failed": int(getattr(provider, "failed", 0)),
        "last_error_type": getattr(provider, "last_error_type", None),
    }


def _exact_metrics(case_results: list[dict]) -> dict:
    """Treat a category match with wrong evidence as both one FP and one FN."""
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    overall = {"tp": 0, "fp": 0, "fn": 0}
    for result in case_results:
        for match in result["matches"]:
            category = match["category"]
            if match["evidence_exact"]:
                overall["tp"] += 1
                counts[category]["tp"] += 1
            else:
                overall["fp"] += 1
                overall["fn"] += 1
                counts[category]["fp"] += 1
                counts[category]["fn"] += 1
        for issue in result["false_positives"]:
            overall["fp"] += 1
            counts[issue["category"]]["fp"] += 1
        for issue in result["false_negatives"]:
            overall["fn"] += 1
            counts[issue["category"]]["fn"] += 1

    def add_rates(row: dict) -> dict:
        precision = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        recall = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {**row, "precision": precision, "recall": recall, "f1": f1}

    return {
        "overall": add_rates(overall),
        "per_category": {
            category.value: add_rates(counts[category.value])
            for category in IssueCategory
        },
        "contract": (
            "TP requires both category and the complete unordered set of "
            "document/line evidence references to match. A wrong evidence set "
            "counts as one FP and one FN."
        ),
    }


def _prediction_signature(scored: dict) -> tuple[str, ...]:
    signatures = []
    for issue in scored["predicted"]:
        evidence = sorted(
            f"{row['document']}:{row['line']}" for row in issue["evidence"]
        )
        signatures.append(f"{issue['category']}|{'|'.join(evidence)}")
    return tuple(sorted(signatures))


def _stability_summary(attempts: list[dict], repeats: int) -> dict:
    eligible = [attempt for attempt in attempts if attempt["included_in_strict_metrics"]]
    by_case: dict[str, list[dict]] = defaultdict(list)
    for attempt in eligible:
        by_case[attempt["case_id"]].append(attempt)

    case_rows = []
    all_case_ids = sorted({attempt["case_id"] for attempt in attempts})
    for case_id in all_case_ids:
        rows = by_case.get(case_id, [])
        if not rows:
            case_rows.append(
                {
                    "case_id": case_id,
                    "eligible_repeat_count": 0,
                    "requested_repeat_count": repeats,
                    "distinct_prediction_set_count": 0,
                    "modal_prediction_set_rate": 0.0,
                    "exact_pass_rate": 0.0,
                }
            )
            continue
        signatures = [_prediction_signature(row["score"]) for row in rows]
        modes = Counter(signatures)
        modal_count = modes.most_common(1)[0][1]
        exact_passes = sum(
            not _exact_metrics([row["score"]])["overall"]["fp"]
            and not _exact_metrics([row["score"]])["overall"]["fn"]
            for row in rows
        )
        case_rows.append(
            {
                "case_id": case_id,
                "eligible_repeat_count": len(rows),
                "requested_repeat_count": repeats,
                "distinct_prediction_set_count": len(modes),
                "modal_prediction_set_rate": modal_count / len(rows),
                "exact_pass_rate": exact_passes / len(rows),
            }
        )

    eligible_count = len(eligible)
    exact_pass_count = sum(
        not _exact_metrics([attempt["score"]])["overall"]["fp"]
        and not _exact_metrics([attempt["score"]])["overall"]["fn"]
        for attempt in eligible
    )
    eligible_case_rows = [row for row in case_rows if row["eligible_repeat_count"]]
    return {
        "scope": "Only full-model attempts included in strict metrics.",
        "eligible_attempt_count": eligible_count,
        "selected_case_count": len(case_rows),
        "case_count_with_any_eligible_repeat": len(eligible_case_rows),
        "case_count_without_any_eligible_repeat": len(case_rows) - len(eligible_case_rows),
        "case_count_with_all_repeats_eligible": sum(
            row["eligible_repeat_count"] == repeats for row in case_rows
        ),
        "exact_pass_rate": exact_pass_count / eligible_count if eligible_count else 0.0,
        "mean_modal_prediction_set_rate": (
            sum(row["modal_prediction_set_rate"] for row in eligible_case_rows)
            / len(eligible_case_rows)
            if eligible_case_rows
            else 0.0
        ),
        "all_selected_case_mean_modal_rate": (
            sum(row["modal_prediction_set_rate"] for row in case_rows) / len(case_rows)
            if case_rows
            else 0.0
        ),
        "cases": case_rows,
    }


def _strict_errors(attempts: list[dict]) -> list[dict]:
    errors = []
    for attempt in attempts:
        if not attempt["included_in_strict_metrics"]:
            continue
        row = attempt["score"]
        evidence_misses = [
            match for match in row["matches"] if not match["evidence_exact"]
        ]
        if not row["false_positives"] and not row["false_negatives"] and not evidence_misses:
            continue
        errors.append(
            {
                "repeat_index": attempt["repeat_index"],
                "case_id": row["case_id"],
                "false_positives": row["false_positives"],
                "false_negatives": row["false_negatives"],
                "evidence_misses": evidence_misses,
            }
        )
    return errors


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int((len(ordered) - 1) * fraction + 0.999999)))
    return ordered[index]


def rescore_report(report: dict) -> dict:
    """Refresh derived metrics from stored safe attempt rows without a provider call."""
    attempts = report["attempts"]
    repeats = int(report["coverage"]["requested_repeat_count"])
    participating_results = [
        attempt["score"] for attempt in attempts if attempt["included_in_model_metrics"]
    ]
    strict_results = [
        attempt["score"] for attempt in attempts if attempt["included_in_strict_metrics"]
    ]
    strict_category_metrics = aggregate_results(strict_results)
    report["metrics"] = strict_category_metrics
    report["category_metrics"] = {
        "model_participating": aggregate_results(participating_results),
        "full_model_only": strict_category_metrics,
    }
    report["exact_evidence_metrics"] = {
        "model_participating": _exact_metrics(participating_results),
        "full_model_only": _exact_metrics(strict_results),
    }
    report["stability"] = _stability_summary(attempts, repeats)
    report["errors"] = _strict_errors(attempts)
    durations = [float(attempt["duration_ms"]) for attempt in attempts]
    report.setdefault("timing", {}).update(
        {
            "case_attempt_p50_ms": _percentile(durations, 0.50),
            "case_attempt_p95_ms": _percentile(durations, 0.95),
            "case_attempt_max_ms": max(durations) if durations else 0.0,
        }
    )
    report["benchmark"]["rescored_without_provider_call"] = True
    report["benchmark"]["rescored_at"] = datetime.now(UTC).isoformat()
    return report


def run_evaluation(
    selected_cases: list[NaturalEvaluationCase],
    max_total_tokens: int,
    repeats: int = 1,
    pipeline_factory: Callable[[], AnalysisPipeline] = model_pipeline,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    attempts: list[dict] = []
    participating_results: list[dict] = []
    strict_results: list[dict] = []
    repeat_summaries: list[dict] = []
    accounted_total = 0
    reported_total = 0
    prompt_total = 0
    completion_total = 0
    duration_total_ms = 0.0
    budget_exhausted = max_total_tokens <= 0
    stop_reason: str | None = "token_budget_not_positive" if budget_exhausted else None

    halt = budget_exhausted
    for repeat_index in range(1, repeats + 1):
        if halt or accounted_total >= max_total_tokens:
            budget_exhausted = accounted_total >= max_total_tokens or budget_exhausted
            break
        repeat_started = perf_counter()
        start_attempt_count = len(attempts)
        start_reported = reported_total
        start_accounted = accounted_total
        start_prompt = prompt_total
        start_completion = completion_total
        for case in selected_cases:
            if halt or accounted_total >= max_total_tokens:
                budget_exhausted = accounted_total >= max_total_tokens or budget_exhausted
                halt = True
                break
            pipeline = pipeline_factory()
            remaining = max_total_tokens - accounted_total
            settings = getattr(
                getattr(getattr(pipeline, "extractor", None), "provider", None),
                "settings",
                None,
            )
            original_budget = None
            if settings is not None:
                original_budget = settings.per_run_token_budget
                settings.per_run_token_budget = min(original_budget, remaining)

            documents = [
                DocumentInput(
                    id=f"{case.case_id}:{index}",
                    name=document.name,
                    content=document.content,
                )
                for index, document in enumerate(case.documents)
            ]
            started = perf_counter()
            try:
                result = pipeline.run(documents)
            finally:
                if settings is not None and original_budget is not None:
                    settings.per_run_token_budget = original_budget

            duration_ms = (perf_counter() - started) * 1000
            duration_total_ms += duration_ms
            reported_tokens = result.prompt_tokens + result.completion_tokens
            conservative_tokens = int(
                getattr(getattr(pipeline, "extractor", None), "_run_tokens_used", 0)
            )
            accounted_tokens = max(reported_tokens, conservative_tokens)
            accounted_total += accounted_tokens
            reported_total += reported_tokens
            prompt_total += result.prompt_tokens
            completion_total += result.completion_tokens

            predictions = _predictions(result)
            scored = score_case(case, predictions)
            provider_stats = _provider_stats(pipeline)
            fallback_markers = (
                "模型分块超过上限",
                "模型单次运行 Token 预算不足",
                "抽取不可用",
                "降级到 BaselineExtractor",
            )
            no_fallback_warning = not any(
                marker in warning
                for warning in result.warnings
                for marker in fallback_markers
            )
            provider_complete = (
                not provider_stats["instrumented"]
                or (
                    provider_stats["requested"] > 0
                    and provider_stats["failed"] == 0
                    and provider_stats["succeeded"] == provider_stats["requested"]
                )
            )
            full_model_attempt = bool(result.model_used) and no_fallback_warning and provider_complete
            participating = bool(result.model_used)
            if participating:
                participating_results.append(scored)
            if full_model_attempt:
                strict_results.append(scored)
            attempts.append(
                {
                    "repeat_index": repeat_index,
                    "case_id": case.case_id,
                    "polarity": case.polarity,
                    "category_focus": case.category_focus.value,
                    "document_count": len(case.documents),
                    "model_used": bool(result.model_used),
                    "full_model_attempt": full_model_attempt,
                    "included_in_model_metrics": participating,
                    "included_in_strict_metrics": full_model_attempt,
                    "duration_ms": round(duration_ms, 3),
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "reported_tokens": reported_tokens,
                    "conservative_accounted_tokens": accounted_tokens,
                    "provider_completions": provider_stats,
                    "warnings": result.warnings,
                    "score": scored,
                }
            )
            if progress is not None:
                progress(
                    {
                        "event": "case_completed",
                        "repeat_index": repeat_index,
                        "case_id": case.case_id,
                        "attempted_case_count": len(attempts),
                        "requested_attempt_count": len(selected_cases) * repeats,
                        "model_used": bool(result.model_used),
                        "full_model_attempt": full_model_attempt,
                        "reported_tokens_so_far": reported_total,
                        "conservative_tokens_so_far": accounted_total,
                        "duration_ms": round(duration_ms, 3),
                    }
                )
            if not result.model_used and any(
                "Token 预算不足" in warning for warning in result.warnings
            ):
                budget_exhausted = True
                stop_reason = "token_budget_exhausted_before_model_success"
                halt = True
                break
            if (
                not result.model_used
                and provider_stats["requested"] > 0
                and provider_stats["succeeded"] == 0
                and provider_stats["failed"] == provider_stats["requested"]
            ):
                stop_reason = "systemic_provider_failure"
                halt = True
                break

        repeat_attempts = attempts[start_attempt_count:]
        repeat_summaries.append(
            {
                "repeat_index": repeat_index,
                "attempted_case_count": len(repeat_attempts),
                "model_participating_case_count": sum(
                    attempt["included_in_model_metrics"] for attempt in repeat_attempts
                ),
                "full_model_case_count": sum(
                    attempt["included_in_strict_metrics"] for attempt in repeat_attempts
                ),
                "duration_ms": round((perf_counter() - repeat_started) * 1000, 3),
                "prompt_tokens": prompt_total - start_prompt,
                "completion_tokens": completion_total - start_completion,
                "reported_tokens": reported_total - start_reported,
                "conservative_accounted_tokens": accounted_total - start_accounted,
                "completed_all_selected_cases": len(repeat_attempts) == len(selected_cases),
            }
        )
        if halt:
            break

    requested_attempt_count = len(selected_cases) * repeats
    if len(attempts) < requested_attempt_count and accounted_total >= max_total_tokens:
        budget_exhausted = True
        stop_reason = stop_reason or "token_budget_exhausted"
    participating_category_metrics = aggregate_results(participating_results)
    strict_category_metrics = aggregate_results(strict_results)
    participating_exact_metrics = _exact_metrics(participating_results)
    strict_exact_metrics = _exact_metrics(strict_results)
    errors = _strict_errors(attempts)
    attempt_durations = [float(attempt["duration_ms"]) for attempt in attempts]
    return {
        "budget": {
            "max_total_tokens": max_total_tokens,
            "prompt_tokens": prompt_total,
            "completion_tokens": completion_total,
            "provider_reported_tokens": reported_total,
            "conservative_accounted_tokens": accounted_total,
            "overshoot_tokens": max(0, accounted_total - max_total_tokens),
            "exhausted": budget_exhausted,
        },
        "coverage": {
            "selected_case_count": len(selected_cases),
            "requested_repeat_count": repeats,
            "requested_attempt_count": requested_attempt_count,
            "attempted_case_count": len(attempts),
            "model_scored_case_count": len(participating_results),
            "strict_model_scored_case_count": len(strict_results),
            "full_model_attempt_case_count": sum(
                attempt["full_model_attempt"] for attempt in attempts
            ),
        },
        "provider_completions": {
            key: sum(attempt["provider_completions"][key] for attempt in attempts)
            for key in ("requested", "succeeded", "failed")
        },
        "stop_reason": stop_reason,
        "timing": {
            "case_attempt_duration_sum_ms": round(duration_total_ms, 3),
            "wall_duration_ms": round(
                sum(repeat_row["duration_ms"] for repeat_row in repeat_summaries), 3
            ),
            "case_attempt_p50_ms": _percentile(attempt_durations, 0.50),
            "case_attempt_p95_ms": _percentile(attempt_durations, 0.95),
            "case_attempt_max_ms": max(attempt_durations) if attempt_durations else 0.0,
        },
        "repeat_summaries": repeat_summaries,
        "metrics": strict_category_metrics,
        "category_metrics": {
            "model_participating": participating_category_metrics,
            "full_model_only": strict_category_metrics,
        },
        "exact_evidence_metrics": {
            "model_participating": participating_exact_metrics,
            "full_model_only": strict_exact_metrics,
        },
        "stability": _stability_summary(attempts, repeats),
        "attempts": attempts,
        "errors": errors,
    }


def build_report(
    dataset_root: Path,
    suite: str,
    max_total_tokens: int,
    repeats: int = 1,
    pipeline_factory: Callable[[], AnalysisPipeline] = model_pipeline,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    cases = select_cases(load_cases("test", dataset_root), suite)
    evaluated = run_evaluation(
        cases, max_total_tokens, repeats, pipeline_factory, progress
    )
    provider_models = sorted(
        {
            attempt["provider_completions"]["model"]
            for attempt in evaluated["attempts"]
            if attempt["provider_completions"]["model"]
        }
    )
    return {
        "benchmark": {
            "name": "LoreGuard complex-v3 model-enhanced evaluation",
            "suite": suite,
            "dataset_version": "3.0.0",
            "dataset_sha256": hashlib.sha256(
                (dataset_root / "test.jsonl").read_bytes()
            ).hexdigest(),
            "provider_models": provider_models,
            "human_annotated": False,
            "developer_visible": True,
            "blind_test": False,
            "metric_scope": (
                "Primary metrics include only full-model attempts: model_used=true, "
                "all logical provider completions succeeded, and no chunk-limit, "
                "budget, extraction, or baseline-only fallback warning occurred. "
                "Model-participating partial attempts are reported separately."
            ),
            "answer_isolation": (
                "The pipeline receives only case documents. Expected categories and "
                "evidence are loaded by the scorer and are never included in provider input."
            ),
            "determinism": (
                "Provider temperature is 0. Repeats measure observed stability but "
                "cannot guarantee deterministic third-party inference."
            ),
            "prompts_or_response_bodies_recorded": False,
            "credentials_or_endpoint_recorded": False,
            "generated_at": datetime.now(UTC).isoformat(),
            "limitations": (
                "Developer-authored acceptance data, not production accuracy. "
                "Pilot mode covers one five-category positive case and one hard "
                "negative per category."
            ),
        },
        **evaluated,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a token-bounded real-provider evaluation on complex-v3."
    )
    parser.add_argument("--suite", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--max-total-tokens", type=int, default=25_000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--rescore",
        type=Path,
        help="Recompute derived metrics from an existing safe report without calling the provider.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/complex-v3-model-evaluation.json"),
    )
    args = parser.parse_args()
    if args.max_total_tokens <= 0:
        parser.error("--max-total-tokens must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    if args.rescore:
        report = rescore_report(
            json.loads(args.rescore.read_text(encoding="utf-8"))
        )
    else:
        report = build_report(
            root / "data" / "evaluation-complex-v3",
            args.suite,
            args.max_total_tokens,
            args.repeats,
            progress=lambda event: print(
                json.dumps(event, ensure_ascii=False), flush=True
            ),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    overall = report["metrics"]["overall"]
    exact_overall = report["exact_evidence_metrics"]["full_model_only"]["overall"]
    print(
        json.dumps(
            {
                "suite": report["benchmark"]["suite"],
                "repeats": report["coverage"]["requested_repeat_count"],
                **report["coverage"],
                **report["provider_completions"],
                "stop_reason": report["stop_reason"],
                **{key: overall[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1", "evidence_pair_line_hit_rate")},
                "exact_evidence": exact_overall,
                **report["budget"],
                "report": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["coverage"]["strict_model_scored_case_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
