from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .domain import IssueCategory
from .pipeline import AnalysisPipeline, BaselineExtractor, DocumentInput


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")
    document: str
    line: int = Field(ge=1)


class ExpectedIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: IssueCategory
    evidence: list[EvidenceRef] = Field(min_length=1)


class EvaluationDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    content: str


class NaturalEvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    scenario_id: str
    split: str
    category_focus: IssueCategory
    polarity: str
    documents: list[EvaluationDocument] = Field(min_length=1)
    expected_issues: list[ExpectedIssue]
    generation: dict[str, Any]

    @model_validator(mode="after")
    def validate_case(self):
        if self.split not in {"dev", "test"}:
            raise ValueError("split must be dev or test")
        if self.polarity not in {"positive", "hard-negative"}:
            raise ValueError("invalid polarity")
        names = [document.name for document in self.documents]
        if len(names) != len(set(names)):
            raise ValueError("document names must be unique within a case")
        name_set = set(names)
        for issue in self.expected_issues:
            for evidence in issue.evidence:
                if evidence.document not in name_set:
                    raise ValueError("expected evidence references an unknown document")
                document = next(row for row in self.documents if row.name == evidence.document)
                if evidence.line > len(document.content.splitlines()):
                    raise ValueError("expected evidence line is outside the document")
        return self


def dataset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "evaluation-natural"


def load_cases(split: str, root: Path | None = None) -> list[NaturalEvaluationCase]:
    if split not in {"dev", "test"}:
        raise ValueError("split must be dev or test")
    path = (root or dataset_root()) / f"{split}.jsonl"
    # Open only the requested split. Test evaluation never reads dev examples.
    return [
        NaturalEvaluationCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _evidence_set(rows: list[dict]) -> set[tuple[str, int]]:
    return {(str(row["document"]), int(row["line"])) for row in rows}


def _prediction_evidence(issue) -> list[dict]:
    return [
        {"document": span.document_name, "line": span.line_start}
        for span in issue.evidence
    ]


def score_case(case: NaturalEvaluationCase, predictions: list[dict]) -> dict:
    expected = [
        {
            "category": issue.category.value,
            "evidence": [row.model_dump() for row in issue.evidence],
        }
        for issue in case.expected_issues
    ]
    unmatched_predictions = set(range(len(predictions)))
    matches: list[dict] = []
    false_negatives: list[dict] = []

    for expected_index, expected_issue in enumerate(expected):
        expected_evidence = _evidence_set(expected_issue["evidence"])
        candidates = [
            index
            for index in unmatched_predictions
            if predictions[index]["category"] == expected_issue["category"]
        ]
        if not candidates:
            false_negatives.append(expected_issue)
            continue
        prediction_index = max(
            candidates,
            key=lambda index: len(
                expected_evidence & _evidence_set(predictions[index]["evidence"])
            ),
        )
        unmatched_predictions.remove(prediction_index)
        predicted_evidence = _evidence_set(predictions[prediction_index]["evidence"])
        matches.append(
            {
                "expected_index": expected_index,
                "prediction_index": prediction_index,
                "category": expected_issue["category"],
                "evidence_exact": predicted_evidence == expected_evidence,
                "expected_evidence": expected_issue["evidence"],
                "predicted_evidence": predictions[prediction_index]["evidence"],
            }
        )

    false_positives = [predictions[index] for index in sorted(unmatched_predictions)]
    return {
        "case_id": case.case_id,
        "scenario_id": case.scenario_id,
        "split": case.split,
        "polarity": case.polarity,
        "category_focus": case.category_focus.value,
        "expected": expected,
        "predicted": predictions,
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def aggregate_results(case_results: list[dict]) -> dict:
    counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "evidence_hits": 0})
    tp = fp = fn = evidence_hits = 0
    for result in case_results:
        for match in result["matches"]:
            tp += 1
            counts[match["category"]]["tp"] += 1
            if match["evidence_exact"]:
                evidence_hits += 1
                counts[match["category"]]["evidence_hits"] += 1
        for issue in result["false_positives"]:
            fp += 1
            counts[issue["category"]]["fp"] += 1
        for issue in result["false_negatives"]:
            fn += 1
            counts[issue["category"]]["fn"] += 1

    def metrics(row: dict) -> dict:
        precision = row["tp"] / (row["tp"] + row["fp"]) if row["tp"] + row["fp"] else 0.0
        recall = row["tp"] / (row["tp"] + row["fn"]) if row["tp"] + row["fn"] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {
            **row,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "evidence_pair_line_hit_rate": row["evidence_hits"] / row["tp"] if row["tp"] else 0.0,
        }

    overall = metrics({"tp": tp, "fp": fp, "fn": fn, "evidence_hits": evidence_hits})
    per_category = {
        category.value: metrics(counts[category.value]) for category in IssueCategory
    }
    return {"overall": overall, "per_category": per_category}


def evaluate_cases(cases: list[NaturalEvaluationCase], pipeline: AnalysisPipeline | None = None) -> dict:
    pipeline = pipeline or AnalysisPipeline(extractor=BaselineExtractor())
    results: list[dict] = []
    for case in cases:
        analysis = pipeline.run(
            [
                DocumentInput(
                    id=f"{case.case_id}:{index}",
                    name=document.name,
                    content=document.content,
                )
                for index, document in enumerate(case.documents)
            ]
        )
        predictions = [
            {
                "category": issue.category.value,
                "title": issue.title,
                "evidence": _prediction_evidence(issue),
                "metadata": issue.metadata,
            }
            for issue in analysis.issues
        ]
        results.append(score_case(case, predictions))
    return {"metrics": aggregate_results(results), "case_results": results}


def run_natural_evaluation(
    split: str = "test",
    root: Path | None = None,
    pipeline: AnalysisPipeline | None = None,
) -> dict:
    selected_root = root or dataset_root()
    split_path = selected_root / f"{split}.jsonl"
    cases = load_cases(split, selected_root)
    evaluated = evaluate_cases(cases, pipeline=pipeline)
    errors = [
        {
            "case_id": row["case_id"],
            "scenario_id": row["scenario_id"],
            "polarity": row["polarity"],
            "category_focus": row["category_focus"],
            "false_positives": row["false_positives"],
            "false_negatives": row["false_negatives"],
            "evidence_misses": [match for match in row["matches"] if not match["evidence_exact"]],
        }
        for row in evaluated["case_results"]
        if row["false_positives"]
        or row["false_negatives"]
        or any(not match["evidence_exact"] for match in row["matches"])
    ]
    error_counts = defaultdict(int)
    for row in errors:
        for issue in row["false_positives"]:
            error_counts[f"false_positive:{issue['category']}"] += 1
        for issue in row["false_negatives"]:
            error_counts[f"false_negative:{issue['category']}"] += 1
        if row["evidence_misses"]:
            error_counts["evidence_pair_or_line_miss"] += len(row["evidence_misses"])
    return {
        "benchmark": {
            "name": "LoreGuard synthetic natural-language evaluation",
            "dataset_kind": "synthetic natural-language",
            "human_annotated": False,
            "split": split,
            "test_isolation": f"Only {split}.jsonl was opened; the other split was not loaded by this run.",
            "model_enabled": False,
            "extractor": "BaselineExtractor + DeterministicCandidateNormalizer",
            "case_count": len(cases),
            "positive_case_count": sum(bool(case.expected_issues) for case in cases),
            "negative_case_count": sum(not case.expected_issues for case in cases),
            "dataset_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "generated_at": datetime.now(UTC).isoformat(),
            "limitations": "Template-generated, not human-annotated, and not an estimate of production accuracy.",
        },
        **evaluated,
        "error_summary": dict(sorted(error_counts.items())),
        "errors": errors,
    }
