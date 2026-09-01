from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.model_extractor import ModelEnhancedExtractor
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
        self.last_safe_error: str | None = None

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
            self.last_safe_error = str(exc)
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
    return {
        "requested": int(getattr(provider, "requested", 0)),
        "succeeded": int(getattr(provider, "succeeded", 0)),
        "failed": int(getattr(provider, "failed", 0)),
        "last_error_type": getattr(provider, "last_error_type", None),
        "last_safe_error": getattr(provider, "last_safe_error", None),
    }


def run_evaluation(
    selected_cases: list[NaturalEvaluationCase],
    max_total_tokens: int,
    pipeline_factory: Callable[[], AnalysisPipeline] = model_pipeline,
) -> dict:
    attempts: list[dict] = []
    scored_results: list[dict] = []
    accounted_total = 0
    reported_total = 0
    budget_exhausted = max_total_tokens <= 0
    stop_reason: str | None = "token_budget_not_positive" if budget_exhausted else None

    for case in selected_cases:
        if accounted_total >= max_total_tokens:
            budget_exhausted = True
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
        reported_tokens = result.prompt_tokens + result.completion_tokens
        conservative_tokens = int(
            getattr(getattr(pipeline, "extractor", None), "_run_tokens_used", 0)
        )
        accounted_tokens = max(reported_tokens, conservative_tokens)
        accounted_total += accounted_tokens
        reported_total += reported_tokens

        predictions = _predictions(result)
        scored = score_case(case, predictions)
        provider_stats = _provider_stats(pipeline)
        fallback_markers = (
            "模型分块超过上限",
            "模型单次运行 Token 预算不足",
            "抽取不可用",
            "降级到 BaselineExtractor",
        )
        full_model_attempt = bool(result.model_used) and not any(
            marker in warning
            for warning in result.warnings
            for marker in fallback_markers
        )
        included = bool(result.model_used)
        if included:
            scored_results.append(scored)
        attempts.append(
            {
                "case_id": case.case_id,
                "polarity": case.polarity,
                "category_focus": case.category_focus.value,
                "document_count": len(case.documents),
                "model_used": bool(result.model_used),
                "full_model_attempt": full_model_attempt,
                "included_in_model_metrics": included,
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
        if not result.model_used and any(
            "Token 预算不足" in warning for warning in result.warnings
        ):
            budget_exhausted = True
            stop_reason = "token_budget_exhausted_before_model_success"
            break
        if (
            not result.model_used
            and provider_stats["requested"] > 0
            and provider_stats["succeeded"] == 0
            and provider_stats["failed"] == provider_stats["requested"]
        ):
            stop_reason = "systemic_provider_failure"
            break

    if len(attempts) < len(selected_cases) and accounted_total >= max_total_tokens:
        budget_exhausted = True
        stop_reason = stop_reason or "token_budget_exhausted"
    metrics = aggregate_results(scored_results)
    errors = [
        {
            "case_id": row["case_id"],
            "false_positives": row["false_positives"],
            "false_negatives": row["false_negatives"],
            "evidence_misses": [
                match for match in row["matches"] if not match["evidence_exact"]
            ],
        }
        for row in scored_results
        if row["false_positives"]
        or row["false_negatives"]
        or any(not match["evidence_exact"] for match in row["matches"])
    ]
    return {
        "budget": {
            "max_total_tokens": max_total_tokens,
            "provider_reported_tokens": reported_total,
            "conservative_accounted_tokens": accounted_total,
            "overshoot_tokens": max(0, accounted_total - max_total_tokens),
            "exhausted": budget_exhausted,
        },
        "coverage": {
            "selected_case_count": len(selected_cases),
            "attempted_case_count": len(attempts),
            "model_scored_case_count": len(scored_results),
            "full_model_attempt_case_count": sum(
                attempt["full_model_attempt"] for attempt in attempts
            ),
        },
        "provider_completions": {
            key: sum(attempt["provider_completions"][key] for attempt in attempts)
            for key in ("requested", "succeeded", "failed")
        },
        "stop_reason": stop_reason,
        "metrics": metrics,
        "attempts": attempts,
        "errors": errors,
    }


def build_report(
    dataset_root: Path,
    suite: str,
    max_total_tokens: int,
    pipeline_factory: Callable[[], AnalysisPipeline] = model_pipeline,
) -> dict:
    cases = select_cases(load_cases("test", dataset_root), suite)
    evaluated = run_evaluation(cases, max_total_tokens, pipeline_factory)
    return {
        "benchmark": {
            "name": "LoreGuard complex-v3 model-enhanced evaluation",
            "suite": suite,
            "dataset_version": "3.0.0",
            "dataset_sha256": hashlib.sha256(
                (dataset_root / "test.jsonl").read_bytes()
            ).hexdigest(),
            "human_annotated": False,
            "developer_visible": True,
            "blind_test": False,
            "metric_scope": (
                "Only cases where at least one model chunk succeeded are included; "
                "the product pipeline still merges deterministic baseline records."
            ),
            "prompts_or_response_bodies_recorded": False,
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/complex-v3-model-evaluation.json"),
    )
    args = parser.parse_args()
    if args.max_total_tokens <= 0:
        parser.error("--max-total-tokens must be positive")

    root = Path(__file__).resolve().parents[1]
    output = args.output if args.output.is_absolute() else root / args.output
    report = build_report(
        root / "data" / "evaluation-complex-v3",
        args.suite,
        args.max_total_tokens,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    overall = report["metrics"]["overall"]
    print(
        json.dumps(
            {
                "suite": args.suite,
                **report["coverage"],
                **report["provider_completions"],
                "stop_reason": report["stop_reason"],
                **{key: overall[key] for key in ("tp", "fp", "fn", "precision", "recall", "f1", "evidence_pair_line_hit_rate")},
                **report["budget"],
                "report": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["coverage"]["model_scored_case_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
