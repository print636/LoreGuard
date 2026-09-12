"""Read-only structural and deterministic-closure checks for this fixture.

This script never calls a model and never writes a result.  It is evaluator-side
code: neither the manifest nor its ground truth may be exposed to the provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = DATASET_ROOT.parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from app.candidate_promotion import (  # noqa: E402
    CandidateEvidenceResolver,
    TrustedDocumentContext,
    _fields_are_grounded,
    promote_investigator_candidates,
)
from app.domain import EvidenceSpan, ParsedDirective  # noqa: E402
from app.evidence_authority import (  # noqa: E402
    EvidenceGrantAuthority,
    InvestigationScope,
    ScopedEvidenceDocument,
)
from app.evidence_chunks import EvidenceChunker, SnapshotDocumentKey  # noqa: E402
from app.evidence_investigator import (  # noqa: E402
    CandidateRecordSubmission,
    ReadSpanArgs,
    build_investigation_seeds,
)
from app.evidence_investigator_loop import (  # noqa: E402
    AuthorizedCandidateBinding,
    EvidenceInvestigatorLoopResult,
)
from app.evidence_investigator_state import UntrustedCandidateEnvelope  # noqa: E402
from app.pipeline import (  # noqa: E402
    AnalysisPipeline,
    BaselineExtractor,
    DocumentInput,
)
from app.parser import parse_document  # noqa: E402
from app.rules import detect_issues  # noqa: E402
from app.semantic_quality import (  # noqa: E402
    apply_semantic_quality_gate,
    document_context_has_noncanonical_frame,
    eligible_for_deterministic_rules,
)


def _fail(case_id: str, message: str) -> None:
    raise AssertionError(f"{case_id}: {message}")


def _source_path(relative: str) -> Path:
    candidate = (DATASET_ROOT / relative).resolve()
    if DATASET_ROOT not in candidate.parents:
        raise AssertionError("source path escapes the dataset root")
    if not candidate.is_file():
        raise AssertionError(f"source file does not exist: {relative}")
    return candidate


def _bind_context(
    directive: ParsedDirective,
    *,
    role: str,
    story_scope: str,
    content: str,
) -> ParsedDirective:
    attrs = dict(directive.attrs)
    attrs["document_role"] = role
    attrs["story_scope"] = story_scope
    return directive.model_copy(
        update={
            "attrs": attrs,
            "noncanonical_frame": (
                directive.noncanonical_frame
                or document_context_has_noncanonical_frame(
                    content,
                    directive.evidence.line_start,
                    directive.evidence.line_end,
                )
            ),
        }
    )


def _range_text(content: str, line_range: list[int]) -> str:
    if (
        not isinstance(line_range, list)
        or len(line_range) != 2
        or not all(type(value) is int and value >= 1 for value in line_range)
        or line_range[1] < line_range[0]
    ):
        raise AssertionError("invalid line range")
    lines = content.splitlines()
    if line_range[1] > len(lines):
        raise AssertionError("line range exceeds source")
    return "\n".join(lines[line_range[0] - 1 : line_range[1]])


def _run_actual_promotion(
    *,
    case_id: str,
    documents: dict[str, tuple[dict, str, Path]],
    baseline: list[ParsedDirective],
    seed,
    target: dict,
):
    project_id = case_id.lower()
    scoped_documents: list[ScopedEvidenceDocument] = []
    snapshots: dict[str, SnapshotDocumentKey] = {}
    contexts: list[TrustedDocumentContext] = []
    for document_id, (source, content, path) in documents.items():
        snapshot = SnapshotDocumentKey(
            project_id=project_id,
            document_id=f"{case_id}:{document_id}",
            document_version=1,
            content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        )
        snapshots[document_id] = snapshot
        scoped_documents.append(
            ScopedEvidenceDocument(snapshot=snapshot, content=content)
        )
        contexts.append(
            TrustedDocumentContext(
                document_id=snapshot.document_id,
                document_name=path.name,
                story_scope=source["story_scope"],
                document_role=source["role"],
            )
        )
    scope = InvestigationScope.create(
        run_id=case_id,
        project_id=project_id,
        documents=tuple(scoped_documents),
    )
    authority = EvidenceGrantAuthority(scope=scope, seeds=(seed,))

    _, content, _ = documents[target["document_id"]]
    snapshot = snapshots[target["document_id"]]
    size = max(1, len(content))
    chunk = EvidenceChunker(
        target_chars=size,
        min_chars=1,
        max_chars=size,
        overlap_chars=0,
    ).chunk(
        project_id=project_id,
        document_id=snapshot.document_id,
        document_version=snapshot.document_version,
        content=content,
        content_sha256=snapshot.content_sha256,
    )[0]
    search_grant = authority.issue_search_grants(seed.seed_ref, (chunk,))[0]
    line_start, line_end = target["lines"]
    span = authority.read_span(
        ReadSpanArgs(
            seed_ref=seed.seed_ref,
            result_ref=search_grant.result_ref,
            line_start=line_start,
            line_end=line_end,
        ),
        max_read_lines=20,
    )
    candidate = CandidateRecordSubmission.model_validate(
        {
            "kind": target["kind"],
            "span_ref": span.span_ref,
            "source_line_start": line_start,
            "source_line_end": line_end,
            "fields": target["fields"],
        }
    )
    envelope = UntrustedCandidateEnvelope.create(
        seed_ref=seed.seed_ref,
        candidates=(candidate,),
        authorized_span_hashes=(span.text_sha256,),
    )
    binding = AuthorizedCandidateBinding(
        seed_ref=seed.seed_ref,
        candidate_payload=envelope.candidate_payloads[0],
        span_ref=span.span_ref,
        snapshot=snapshot,
        line_start=line_start,
        line_end=line_end,
        char_start=span.char_start,
        char_end=span.char_end,
        text=span.text,
        text_sha256=span.text_sha256,
        authorized_span_char_start=span.char_start,
        authorized_span_char_end=span.char_end,
        authorized_span_sha256=span.text_sha256,
    )
    loop_result = EvidenceInvestigatorLoopResult(
        outcome="completed",
        reason_code="completed",
        envelopes=(envelope,),
        authorized_candidates=(binding,),
        provider_calls=1,
        completed_seeds=1,
        executed_tool_calls=1,
    )
    resolver = CandidateEvidenceResolver(
        scope=scope,
        documents=tuple(contexts),
        investigator_result=loop_result,
    )
    return promote_investigator_candidates(
        baseline_directives=tuple(baseline),
        baseline_issues=tuple(detect_issues(baseline)),
        envelopes=(envelope,),
        seeds=(seed,),
        evidence_resolver=resolver,
    )


def main(*, selected_split: str | None = None) -> None:
    manifest = json.loads((DATASET_ROOT / "manifest.json").read_text(encoding="utf-8"))
    freeze = json.loads((DATASET_ROOT / "freeze.json").read_text(encoding="utf-8"))
    frozen_rows = freeze["frozen_files"]
    frozen_paths = [row["path"] for row in frozen_rows]
    if len(frozen_paths) != len(set(frozen_paths)):
        raise AssertionError("freeze file contains duplicate paths")
    expected_frozen_paths = {
        "manifest.json",
        "README.md",
        *(
            path.relative_to(DATASET_ROOT).as_posix()
            for path in DATASET_ROOT.glob("*/*/*.md")
        ),
    }
    if set(frozen_paths) != expected_frozen_paths:
        raise AssertionError("freeze file list does not match the evaluator inputs")
    for row in frozen_rows:
        path = _source_path(row["path"])
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != row["sha256"]:
            raise AssertionError(f"frozen file hash mismatch: {row['path']}")
    all_cases = manifest["cases"]
    if len(all_cases) < 12:
        raise AssertionError("at least 12 cases are required")
    case_ids = [case["case_id"] for case in all_cases]
    if len(case_ids) != len(set(case_ids)):
        raise AssertionError("case ids must be unique")

    split_counts = {
        split: sum(case["split"] == split for case in all_cases)
        for split in ("dev", "holdout")
    }
    declared_counts = {
        split: manifest["splits"][split]["case_count"]
        for split in ("dev", "holdout")
    }
    if split_counts != declared_counts:
        raise AssertionError("declared split counts do not match cases")
    if manifest["live_model_status"]["state"] != "not_run":
        raise AssertionError("fixture validation cannot claim a live-model run")
    if selected_split not in {None, "dev", "holdout"}:
        raise AssertionError("selected split must be dev, holdout, or omitted")
    cases = [
        case
        for case in all_cases
        if selected_split is None or case["split"] == selected_split
    ]

    positive_count = 0
    negative_count = 0
    covered_positive_families: dict[str, set[str]] = {"dev": set(), "holdout": set()}

    for case in cases:
        case_id = case["case_id"]
        if case.get("prompt_example_allowed") is not False:
            _fail(case_id, "case content must be forbidden as a prompt example")

        documents: dict[str, tuple[dict, str, Path]] = {}
        production_documents: list[DocumentInput] = []
        fixture_baseline: list[ParsedDirective] = []
        for source in case["sources"]:
            document_id = source["document_id"]
            if document_id in documents:
                _fail(case_id, f"duplicate document id: {document_id}")
            path = _source_path(source["path"])
            content = path.read_text(encoding="utf-8")
            documents[document_id] = (source, content, path)
            production_documents.append(
                DocumentInput(
                    id=f"{case_id}:{document_id}",
                    name=path.name,
                    content=content,
                    role=source["role"],
                    scope=source["story_scope"],
                )
            )
            if source["role"] == "chapter" and any(
                line.lstrip().startswith("@") for line in content.splitlines()
            ):
                _fail(case_id, "chapter side contains a directive")
            parsed = parse_document(
                f"{case_id}:{document_id}",
                path.name,
                content,
            )
            fixture_baseline.extend(
                _bind_context(
                    directive,
                    role=source["role"],
                    story_scope=source["story_scope"],
                    content=content,
                )
                for directive in parsed.directives
            )

        # Exercise the same deterministic production closure used by an
        # analysis run: parser -> candidate normalizer -> semantic quality ->
        # canonical rules.  BaselineExtractor is explicit, so validation can
        # never contact a provider even when a developer has API credentials.
        production_baseline = AnalysisPipeline(extractor=BaselineExtractor()).run(
            production_documents
        )
        baseline = fixture_baseline
        baseline_issues = detect_issues(baseline)
        expected = case["expected"]
        if len(baseline_issues) != expected["baseline_issue_count"]:
            _fail(case_id, "the fixture parse baseline emits an unexpected issue")
        if len(production_baseline.issues) != expected["baseline_issue_count"]:
            _fail(
                case_id,
                "the production normalizer/quality/rules closure emits an unexpected issue",
            )

        anchor = case["anchor"]
        anchor_source, anchor_content, _ = documents[anchor["document_id"]]
        _range_text(anchor_content, anchor["lines"])
        matching_anchors = [
            directive
            for directive in baseline
            if directive.evidence.document_id == f"{case_id}:{anchor['document_id']}"
            and directive.evidence.line_start == anchor["lines"][0]
            and directive.evidence.line_end == anchor["lines"][1]
            and directive.kind == anchor["kind"]
            and all(
                directive.attrs.get(key) == value
                for key, value in anchor["fields"].items()
            )
        ]
        if len(matching_anchors) != 1:
            _fail(case_id, "the declared explicit anchor was not parsed exactly once")

        seeds = build_investigation_seeds(case_id, baseline, limit=64)
        if len(seeds) != 1:
            _fail(case_id, f"expected one isolated seed, found {len(seeds)}")
        seed = seeds[0]
        if (
            seed.family.value != expected["issue_category"]
            or seed.anchor.evidence.document_id != f"{case_id}:{anchor['document_id']}"
            or seed.anchor.evidence.line_start != anchor["lines"][0]
        ):
            _fail(case_id, "the sole seed is not the declared anchor")

        target = case.get("candidate") or case.get("lure_candidate")
        target_source, target_content, target_path = documents[target["document_id"]]
        target_text = _range_text(target_content, target["lines"])
        target_start, target_end = target["lines"]
        exact_baseline_target = any(
            directive.evidence.document_id == f"{case_id}:{target['document_id']}"
            and directive.evidence.line_start <= target_start
            and directive.evidence.line_end >= target_end
            and directive.kind == target["kind"]
            and all(
                directive.attrs.get(key) == value
                for key, value in target["fields"].items()
            )
            for directive in baseline
        )
        if exact_baseline_target:
            _fail(case_id, "the no-model baseline already extracted the target record")

        grounded = _fields_are_grounded(target["kind"], target["fields"], target_text)
        raw_candidate = ParsedDirective(
            kind=target["kind"],
            attrs={
                **target["fields"],
                "document_role": target_source["role"],
                "story_scope": target_source["story_scope"],
            },
            evidence=EvidenceSpan(
                document_id=f"{case_id}:{target['document_id']}",
                document_name=target_path.name,
                line_start=target_start,
                line_end=target_end,
                text=target_text,
            ),
            noncanonical_frame=document_context_has_noncanonical_frame(
                target_content,
                target_start,
                target_end,
            ),
            provenance_sources=frozenset({"model"}),
        )
        quality = apply_semantic_quality_gate([raw_candidate])
        eligible = bool(
            len(quality.directives) == 1
            and eligible_for_deterministic_rules(quality.directives[0])
        )
        reproduced = (
            []
            if not eligible
            else detect_issues([*baseline, quality.directives[0]])
        )
        reproduced_families = {issue.category.value for issue in reproduced}
        promotion_result = _run_actual_promotion(
            case_id=case_id,
            documents=documents,
            baseline=baseline,
            seed=seed,
            target=target,
        )

        if expected["decision"] == "added_issue":
            positive_count += 1
            covered_positive_families[case["split"]].add(expected["issue_category"])
            if not grounded:
                _fail(case_id, "positive candidate is outside grounding closure")
            if not eligible:
                _fail(case_id, "positive candidate is outside semantic closure")
            if expected["issue_category"] not in reproduced_families:
                _fail(case_id, "positive candidate does not reproduce its rule family")
            if (
                promotion_result.accepted_candidates != 1
                or len(promotion_result.added_issues) != expected["added_issue_count"]
                or promotion_result.added_issues[0].category.value
                != expected["issue_category"]
            ):
                _fail(case_id, "the full promotion path did not add the expected issue")
        elif expected["decision"] == "abstain":
            negative_count += 1
            promotion = expected["promotion"]
            if promotion.startswith("candidate_not_grounded") and grounded:
                _fail(case_id, "declared grounding rejection is not reproducible")
            if promotion.startswith("candidate_semantics_rejected") and (
                not grounded or eligible
            ):
                _fail(case_id, "declared semantic rejection is not reproducible")
            if promotion.startswith("candidate_no_rule_conflict") and (
                not grounded
                or not eligible
                or expected["issue_category"] in reproduced_families
            ):
                _fail(case_id, "declared no-rule-conflict result is not reproducible")
            expected_rejection = promotion.removesuffix("_if_submitted")
            if (
                promotion_result.accepted_candidates != 0
                or promotion_result.added_issues
                or dict(promotion_result.rejection_counts)
                != {expected_rejection: 1}
            ):
                _fail(
                    case_id,
                    "the full promotion path did not produce the declared rejection: "
                    f"expected={expected_rejection}, "
                    f"actual={dict(promotion_result.rejection_counts)}",
                )
        else:
            _fail(case_id, "unknown expected decision")

        for evidence in expected["allowed_evidence"]:
            _, content, _ = documents[evidence["document_id"]]
            _range_text(content, evidence["lines"])

    required_families = set(manifest["issue_families"])
    checked_splits = (
        (selected_split,) if selected_split is not None else ("dev", "holdout")
    )
    for split in checked_splits:
        covered = covered_positive_families[split]
        if covered != required_families:
            raise AssertionError(
                f"{split}: positive coverage differs from the five rule families"
            )

    split_summary = (
        f"{split_counts['dev']} dev / {split_counts['holdout']} holdout"
        if selected_split is None
        else f"{selected_split} only"
    )
    print(
        "fixture valid: "
        f"{len(cases)} cases ({split_summary}), "
        f"{positive_count} positive / {negative_count} abstain; "
        "baseline issues=0; five positive families present in checked splits; "
        "live model not run"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "holdout"))
    args = parser.parse_args()
    main(selected_split=args.split)
