"""Generate a sealed, offline Simplified-Chinese OOC adjudication page.

The generator accepts only the frozen public bundle and two already-locked human
reviewer annotations.  It validates both annotations before embedding their
snapshots, and never loads model predictions or private gold.  The generated
single-file page exports an ``AdjudicationBundle`` bound to the ordered inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import generate_ooc_annotation_page as annotation_page
from scripts import ooc_annotation_workflow as workflow
from scripts import ooc_closed_bundle as closed_bundle
from scripts import ooc_eval_contract as contract


PAGE_SCHEMA_VERSION = "loreguard-ooc-adjudication-page-v1"
ResolutionMode = Literal["reviewer_consensus", "third_human"]


class AdjudicationPageError(ValueError):
    """Stable error for unsafe or inconsistent adjudication-page input."""


def _local_path(value: str | Path, *, name: str) -> Path:
    raw = str(value)
    if "://" in raw or raw.startswith(("\\\\", "//")):
        raise AdjudicationPageError(f"{name}_must_be_local")
    return Path(value)


def _validate_identity(value: str | None, *, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 200
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AdjudicationPageError(f"{name}_invalid")
    return value


def _load_locked_annotation(path: str | Path) -> workflow.ReviewerAnnotationBundle:
    source = _local_path(path, name="annotation")
    if not source.name.lower().endswith(".locked.json"):
        raise AdjudicationPageError("annotation_must_be_locked_json")
    payload = workflow._read_limited_local(source, limit=contract.MAX_JSON_BYTES)
    return workflow.ReviewerAnnotationBundle.model_validate_json(payload)


def _public_cases(
    public: contract.PublicInputBundle,
    bundle_root: str | Path,
    author_setup: closed_bundle.PublicAuthorSetupBundle,
    execution_freeze: closed_bundle.ExecutionFreeze,
    *,
    reviewer_id: str,
    manual_version: str,
) -> list[dict[str, Any]]:
    return annotation_page.build_page_data(
        public,
        bundle_root,
        author_setup,
        execution_freeze,
        reviewer_id=reviewer_id,
        manual_version=manual_version,
    )["cases"]


def build_page_data(
    public: contract.PublicInputBundle,
    bundle_root: str | Path,
    author_setup: closed_bundle.PublicAuthorSetupBundle,
    execution_freeze: closed_bundle.ExecutionFreeze,
    left: workflow.ReviewerAnnotationBundle,
    right: workflow.ReviewerAnnotationBundle,
    *,
    third_human_id: str | None = None,
    default_resolution_mode: ResolutionMode = "reviewer_consensus",
) -> dict[str, Any]:
    """Build the blinded data envelope embedded into the adjudication page."""

    closed_bundle.validate_execution_freeze_binding(
        public,
        author_setup,
        execution_freeze,
    )
    workflow.validate_annotation_binding(public, left)
    workflow.validate_annotation_binding(public, right)
    workflow._validate_annotation_setup_binding(public, author_setup, left)
    workflow._validate_annotation_setup_binding(public, author_setup, right)
    if left.reviewer_id == right.reviewer_id:
        raise AdjudicationPageError("two_distinct_reviewers_required")
    if left.manual_version != right.manual_version:
        raise AdjudicationPageError("reviewers_must_use_same_manual_version")
    if default_resolution_mode not in {"reviewer_consensus", "third_human"}:
        raise AdjudicationPageError("default_resolution_mode_invalid")
    third_human_id = _validate_identity(third_human_id, name="third_human_id")
    if default_resolution_mode == "third_human" and third_human_id is None:
        raise AdjudicationPageError("third_human_mode_requires_identity")
    if third_human_id in {left.reviewer_id, right.reviewer_id}:
        raise AdjudicationPageError("third_human_must_be_distinct_from_reviewers")

    annotation_digests = (
        contract.canonical_sha256(left),
        contract.canonical_sha256(right),
    )
    if annotation_digests[0] == annotation_digests[1]:
        raise AdjudicationPageError("locked_annotations_must_be_distinct")
    agreement = workflow.compare_annotations(public, left, right)
    left_cases = {row.case_id: row for row in left.cases}
    right_cases = {row.case_id: row for row in right.cases}
    agreement_cases = {row.case_id: row for row in agreement.cases}
    cases = _public_cases(
        public,
        bundle_root,
        author_setup,
        execution_freeze,
        reviewer_id=left.reviewer_id,
        manual_version=left.manual_version,
    )
    for row in cases:
        case_id = row["case_id"]
        row["reviewer_a"] = left_cases[case_id].model_dump(mode="json")
        row["reviewer_b"] = right_cases[case_id].model_dump(mode="json")
        row["agreement"] = agreement_cases[case_id].model_dump(mode="json")

    return {
        "page_schema_version": PAGE_SCHEMA_VERSION,
        "adjudication_schema_version": workflow.ADJUDICATION_SCHEMA_VERSION,
        "dataset_id": public.dataset_id,
        "public_input_sha256": contract.canonical_sha256(public),
        "author_setup_sha256": contract.canonical_sha256(author_setup),
        "execution_freeze_sha256": contract.canonical_sha256(execution_freeze),
        "manual_version": left.manual_version,
        "reviewers": [
            {
                "side": "A",
                "reviewer_id": left.reviewer_id,
                "annotation_sha256": annotation_digests[0],
            },
            {
                "side": "B",
                "reviewer_id": right.reviewer_id,
                "annotation_sha256": annotation_digests[1],
            },
        ],
        "third_human_id": third_human_id,
        "default_resolution_mode": default_resolution_mode,
        "agreement_summary": agreement.model_dump(mode="json", exclude={"cases"}),
        "cases": cases,
    }


def render_adjudication_page(data: dict[str, Any]) -> str:
    """Render a self-contained page without external runtime dependencies."""

    return _HTML_TEMPLATE.replace(
        "__ADJUDICATION_DATA__",
        annotation_page._safe_json_for_script(data),
    )


def generate_adjudication_page(
    public_path: str | Path,
    bundle_root: str | Path,
    author_setup_path: str | Path,
    execution_freeze_path: str | Path,
    left_annotation_path: str | Path,
    right_annotation_path: str | Path,
    *,
    output_path: str | Path,
    third_human_id: str | None = None,
    default_resolution_mode: ResolutionMode = "reviewer_consensus",
) -> Path:
    """Validate both locked submissions and exclusively create one HTML file."""

    public_file = _local_path(public_path, name="public_input")
    root = _local_path(bundle_root, name="bundle_root")
    setup_file = _local_path(author_setup_path, name="author_setup")
    freeze_file = _local_path(execution_freeze_path, name="execution_freeze")
    left_file = _local_path(left_annotation_path, name="left_annotation")
    right_file = _local_path(right_annotation_path, name="right_annotation")
    output = _local_path(output_path, name="output")
    workflow._require_distinct_local_paths(
        (public_file, root, setup_file, freeze_file, left_file, right_file, output)
    )
    public = contract.load_public_input_file(public_file)
    author_setup = closed_bundle.PublicAuthorSetupBundle.model_validate_json(
        workflow._read_limited_local(setup_file, limit=contract.MAX_JSON_BYTES)
    )
    execution_freeze = closed_bundle.ExecutionFreeze.model_validate_json(
        workflow._read_limited_local(freeze_file, limit=contract.MAX_JSON_BYTES)
    )
    left = _load_locked_annotation(left_file)
    right = _load_locked_annotation(right_file)
    data = build_page_data(
        public,
        root,
        author_setup,
        execution_freeze,
        left,
        right,
        third_human_id=third_human_id,
        default_resolution_mode=default_resolution_mode,
    )
    if output.suffix.lower() not in {".html", ".htm"}:
        raise AdjudicationPageError("output_must_be_html")
    created = False
    try:
        if workflow._path_chain_has_reparse_point(output.parent):
            raise AdjudicationPageError("output_reparse_point_forbidden")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as target:
            created = True
            target.write(render_adjudication_page(data))
    except FileExistsError as exc:
        raise AdjudicationPageError("output_already_exists") from exc
    except AdjudicationPageError:
        if created:
            output.unlink(missing_ok=True)
        raise
    except OSError as exc:
        if created:
            output.unlink(missing_ok=True)
        raise AdjudicationPageError("output_unwritable") from exc
    return output.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-input", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--author-setup", required=True)
    parser.add_argument("--execution-freeze", required=True)
    parser.add_argument("--left-annotation", required=True)
    parser.add_argument("--right-annotation", required=True)
    parser.add_argument("--third-human-id")
    parser.add_argument(
        "--default-resolution-mode",
        choices=("reviewer_consensus", "third_human"),
        default="reviewer_consensus",
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = generate_adjudication_page(
            args.public_input,
            args.bundle_root,
            args.author_setup,
            args.execution_freeze,
            args.left_annotation,
            args.right_annotation,
            output_path=args.output,
            third_human_id=args.third_human_id,
            default_resolution_mode=args.default_resolution_mode,
        )
    except (
        AdjudicationPageError,
        contract.ContractError,
        workflow.AnnotationWorkflowError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {"status": "failed", "code": str(exc)},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        return 1
    print(
        json.dumps(
            {"status": "created", "output": str(output)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-Hans">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <title>LoreGuard OOC 离线仲裁台</title>
  <style>
    :root {
      --deep: #090716; --page: #0d0a1e; --surface: #1d1739;
      --raised: #291f4b; --field: #100d24; --text: #f4f0ff;
      --secondary: #cfc6e3; --muted: #aea5c2; --lavender: #b7a2ff;
      --cyan: #8be9f0; --pink: #ff91bd; --warning: #ffbd82;
      --success: #8de5b8; --danger: #ff7fad;
      --border: rgba(179,154,255,.22); --border-strong: rgba(183,161,255,.46);
      --shadow: 0 18px 48px rgba(0,0,0,.24); --header: 82px;
      color: var(--text); background: var(--deep);
      font-family: "Noto Sans SC", "Microsoft YaHei UI", "PingFang SC", system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; scroll-padding-top: calc(var(--header) + 18px); }
    body { margin: 0; min-width: 320px; min-height: 100dvh; background: var(--page); }
    button, input, select, textarea { font: inherit; }
    button, select, input[type="checkbox"], input[type="radio"], label[for] { cursor: pointer; }
    button, select, input[type="text"], textarea {
      min-height: 44px; color: var(--text); background: var(--field);
      border: 1px solid var(--border-strong); border-radius: 8px;
    }
    button { padding: 10px 14px; font-weight: 650; }
    button:hover:not(:disabled) { border-color: var(--lavender); background: var(--raised); }
    button:active:not(:disabled) { opacity: .82; }
    button:disabled, select:disabled, input:disabled, textarea:disabled { cursor: not-allowed; opacity: .48; }
    :focus-visible { outline: 3px solid var(--cyan); outline-offset: 3px; }
    .skip-link { position: fixed; z-index: 1000; top: 8px; left: 8px; transform: translateY(-160%); padding: 10px 14px; color: var(--deep); background: var(--cyan); border-radius: 8px; }
    .skip-link:focus { transform: none; }
    .app-header { position: sticky; z-index: 40; top: 0; min-height: var(--header); display: flex; align-items: center; justify-content: space-between; gap: 18px; padding: 14px clamp(16px,3vw,40px); background: rgba(13,10,30,.96); border-bottom: 1px solid var(--border); backdrop-filter: blur(14px); }
    .brand h1 { margin: 0; font-size: clamp(20px,2.2vw,28px); line-height: 1.2; }
    .brand p { margin: 5px 0 0; color: var(--secondary); font-size: 13px; }
    .toolbar { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
    .toolbar .lock { color: #180d20; background: var(--pink); border-color: var(--pink); }
    .toolbar .danger { color: var(--danger); }
    .file-input { position: absolute; inline-size: 1px; block-size: 1px; opacity: 0; pointer-events: none; }
    .freeze-strip { display: grid; grid-template-columns: auto minmax(0,1fr); gap: 8px 20px; padding: 11px clamp(16px,3vw,40px); color: var(--secondary); background: #15102c; border-bottom: 1px solid var(--border); font-size: 13px; }
    .freeze-strip strong { color: var(--success); }
    .binding { overflow-wrap: anywhere; font-family: "Cascadia Code", Consolas, monospace; }
    .freeze-binding { grid-column: 1 / -1; color: var(--muted); font-size: 12px; }
    .error-summary { width: min(1540px,calc(100% - 32px)); margin: 18px auto 0; padding: 16px 20px; background: rgba(255,127,173,.12); border: 1px solid rgba(255,127,173,.72); border-radius: 12px; }
    .error-summary[hidden], [hidden] { display: none !important; }
    .error-summary h2 { margin: 0 0 8px; font-size: 18px; }
    .error-summary ul { margin: 0; padding-left: 22px; }
    .error-summary button { min-height: 32px; padding: 2px 4px; color: var(--text); background: none; border: 0; text-align: left; text-decoration: underline; }
    .workspace { width: min(100%,1720px); margin: 0 auto; padding: 20px clamp(12px,2.5vw,34px) 72px; display: grid; grid-template-columns: 210px minmax(0,1fr); align-items: start; gap: 18px; }
    .case-rail { position: sticky; top: calc(var(--header) + 18px); overflow: hidden; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; }
    .rail-head { padding: 16px; border-bottom: 1px solid var(--border); }
    .rail-head h2 { margin: 0; font-size: 16px; }
    .rail-head p { margin: 6px 0 0; color: var(--muted); font-size: 13px; }
    .case-list { max-height: calc(100dvh - 192px); overflow: auto; padding: 8px; }
    .case-button { width: 100%; display: grid; grid-template-columns: 32px minmax(0,1fr) auto; align-items: center; gap: 8px; margin: 0 0 6px; padding: 8px; color: var(--secondary); background: transparent; border-color: transparent; text-align: left; }
    .case-button[aria-current="true"] { color: var(--text); background: var(--raised); border-color: var(--lavender); }
    .case-number { color: var(--lavender); font-variant-numeric: tabular-nums; }
    .case-state { color: var(--muted); font-size: 12px; }
    .case-mobile { display: none; padding: 12px 16px; }
    .case-mobile label { display: block; margin-bottom: 6px; font-weight: 650; }
    .case-mobile select { width: 100%; padding: 8px 10px; }
    .case-work { min-width: 0; }
    .case-intro { margin-bottom: 16px; padding: 18px 20px; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; }
    .case-intro h2 { margin: 0 0 7px; font-size: 24px; }
    .case-intro p { margin: 5px 0; color: var(--secondary); overflow-wrap: anywhere; }
    .target-context { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); }
    .target-context h3 { margin: 0 0 10px; font-size: 17px; }
    .context-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px 14px; }
    .context-item { min-width: 0; padding: 9px 10px; background: var(--field); border-left: 3px solid var(--lavender); }
    .context-item dt { margin: 0 0 4px; color: var(--muted); font-size: 12px; }
    .context-item dd { margin: 0; line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere; }
    .context-item.wide { grid-column: 1 / -1; }
    .shortcut { color: var(--cyan) !important; font-size: 13px; }
    .case-grid { display: grid; grid-template-columns: minmax(440px,1.08fr) minmax(520px,.92fr); align-items: start; gap: 16px; }
    .panel { min-width: 0; margin-bottom: 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; }
    .panel-head { padding: 14px 18px; background: #17122f; border-bottom: 1px solid var(--border); }
    .panel-head h3 { margin: 0; font-size: 17px; }
    .panel-head p { margin: 5px 0 0; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .panel[data-target="true"] { border-color: rgba(255,145,189,.58); }
    .panel[data-selector-source="true"] .panel-head { box-shadow: inset 4px 0 var(--lavender); }
    .document-lines { padding: 7px 0; }
    .evidence-row { display: grid; grid-template-columns: 42px minmax(0,1fr) 142px; gap: 10px; align-items: start; padding: 9px 12px 9px 8px; border-left: 4px solid transparent; border-bottom: 1px solid rgba(179,154,255,.09); }
    .evidence-row[data-final-role="B"] { border-left-color: var(--lavender); }
    .evidence-row[data-final-role="C"] { border-left-color: var(--pink); }
    .evidence-row[data-final-role="G"] { border-left-color: var(--success); }
    .evidence-row[data-final-role="X"] { border-left-color: var(--warning); }
    .evidence-row[data-final-role="P"] { border-left-color: var(--cyan); }
    .evidence-row[data-forbidden="true"] { border-left-color: var(--danger); background: rgba(255,127,173,.06); }
    .evidence-row[data-selector="true"] { background: rgba(183,162,255,.08); }
    .line-no { padding-top: 7px; color: var(--muted); text-align: right; font: 12px "Cascadia Code",Consolas,monospace; }
    .line-copy { margin: 0; padding: 4px 0; color: #fbf8ff; font-family: "Noto Serif SC","Source Han Serif SC","Songti SC",SimSun,serif; font-size: 17px; line-height: 1.78; white-space: pre-wrap; overflow-wrap: anywhere; }
    .evidence-meta { grid-column: 2; display: flex; flex-wrap: wrap; gap: 6px; }
    .role-chip { padding: 3px 7px; color: var(--secondary); background: var(--field); border: 1px solid var(--border); border-radius: 999px; font-size: 12px; }
    .evidence-controls { grid-column: 3; grid-row: 1 / span 2; display: grid; gap: 7px; }
    .evidence-controls select { width: 100%; padding: 7px; }
    .forbidden-check { display: flex; align-items: center; gap: 7px; min-height: 36px; color: var(--secondary); font-size: 13px; }
    .blank-line { min-height: 18px; border-bottom: 1px solid rgba(179,154,255,.05); }
    .comparison { padding: 16px; }
    .comparison-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
    .comparison-actions button { flex: 1 1 150px; }
    .compare-row { display: grid; grid-template-columns: 110px minmax(0,1fr) minmax(0,1fr); border-top: 1px solid var(--border); }
    .compare-row:first-of-type { border-top: 0; }
    .compare-row.disagrees { background: rgba(255,127,173,.07); box-shadow: inset 3px 0 var(--danger); }
    .compare-row.agrees { box-shadow: inset 3px 0 rgba(141,229,184,.65); }
    .compare-label, .compare-value { min-width: 0; padding: 10px; overflow-wrap: anywhere; }
    .compare-label { color: var(--secondary); font-weight: 650; }
    .compare-label small { display: block; margin-top: 4px; color: var(--muted); font-weight: 400; }
    .compare-value { border-left: 1px solid var(--border); line-height: 1.55; }
    .compare-value strong { display: block; margin-bottom: 3px; color: var(--lavender); font-size: 12px; }
    .adopt-one { min-height: 30px; margin-top: 7px; padding: 3px 8px; font-size: 12px; }
    .review-notes { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 14px; }
    .review-note { padding: 11px; background: var(--field); border: 1px solid var(--border); border-radius: 8px; }
    .review-note h4 { margin: 0 0 6px; color: var(--lavender); font-size: 13px; }
    .review-note p { margin: 0; color: var(--secondary); line-height: 1.65; white-space: pre-wrap; }
    .adjudication { padding: 18px; }
    .adjudication h3 { margin: 0 0 5px; }
    .adjudication > p { margin: 0 0 16px; color: var(--secondary); font-size: 13px; }
    fieldset { min-width: 0; margin: 0; padding: 0; border: 0; }
    .form-section { padding: 16px 0; border-top: 1px solid var(--border); }
    .form-section:first-child { padding-top: 0; border-top: 0; }
    .form-section h4 { margin: 0 0 12px; font-size: 16px; }
    .field { margin-bottom: 14px; }
    .field > label, .field-label { display: block; margin-bottom: 6px; font-weight: 650; }
    .field select, .field input[type="text"], .field textarea { width: 100%; padding: 9px 11px; }
    .field textarea { min-height: 92px; resize: vertical; line-height: 1.55; }
    .helper { margin: 6px 0 0; color: var(--muted); font-size: 12px; line-height: 1.55; }
    .derived-result { min-height: 44px; padding: 10px 11px; color: var(--text); background: var(--field); border: 1px solid var(--border-strong); border-radius: 8px; overflow-wrap: anywhere; }
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .radio-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .radio-card, .target-check { display: flex; align-items: flex-start; gap: 8px; min-height: 44px; padding: 9px 10px; background: var(--field); border: 1px solid var(--border-strong); border-radius: 8px; }
    .radio-card:has(input:checked), .target-check:has(input:checked) { border-color: var(--lavender); background: var(--raised); }
    .radio-card input, .target-check input { margin-top: 4px; accent-color: var(--lavender); }
    .structure-empty { padding: 12px; color: var(--muted); background: var(--field); border-radius: 8px; }
    .structure-row { margin-bottom: 12px; padding: 12px; background: var(--field); border: 1px solid var(--border); border-radius: 10px; }
    .structure-row h5 { margin: 0 0 9px; font-size: 14px; overflow-wrap: anywhere; }
    .structure-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .structure-grid label { display: block; margin-bottom: 5px; color: var(--secondary); font-size: 12px; }
    .structure-grid select, .structure-grid input { width: 100%; padding: 8px; }
    .targets { display: grid; gap: 6px; margin: 8px 0 12px; }
    .case-errors { margin-bottom: 16px; padding: 12px 14px; background: rgba(255,127,173,.1); border: 1px solid rgba(255,127,173,.55); border-radius: 8px; }
    .case-errors strong { color: var(--danger); }
    .case-errors ul { margin: 7px 0 0; padding-left: 20px; }
    .confirmation-panel { margin-top: 16px; padding: 16px; background: rgba(139,233,240,.055); border: 1px solid rgba(139,233,240,.35); border-radius: 10px; }
    .confirmation-panel h4 { margin: 0 0 6px; }
    .confirmation-panel .target-check { margin-top: 9px; }
    .locked-banner { margin-bottom: 14px; padding: 12px; color: var(--success); background: rgba(141,229,184,.09); border: 1px solid rgba(141,229,184,.48); border-radius: 8px; }
    .status { position: fixed; z-index: 100; right: 18px; bottom: 18px; max-width: min(440px,calc(100vw - 36px)); padding: 12px 16px; background: var(--raised); border: 1px solid var(--border-strong); border-radius: 10px; box-shadow: var(--shadow); }
    .status:empty { display: none; }
    @media (max-width: 1280px) { .case-grid { grid-template-columns: 1fr; } }
    @media (max-width: 767px) {
      :root { --header: 0px; }
      .app-header { position: static; align-items: flex-start; flex-direction: column; }
      .toolbar { width: 100%; justify-content: flex-start; }
      .toolbar button { flex: 1 1 145px; }
      .freeze-strip { grid-template-columns: 1fr; }
      .workspace { display: block; padding-inline: 10px; }
      .case-rail { position: static; margin-bottom: 14px; }
      .rail-head, .case-list { display: none; }
      .case-mobile { display: block; }
      .evidence-row { grid-template-columns: 34px minmax(0,1fr); }
      .evidence-controls { grid-column: 2; grid-row: auto; }
      .line-copy { font-size: 16px; }
      .compare-row { grid-template-columns: 1fr 1fr; }
      .compare-label { grid-column: 1 / -1; border-bottom: 1px solid var(--border); }
      .review-notes, .form-grid, .structure-grid { grid-template-columns: 1fr; }
      .context-grid { grid-template-columns: 1fr; }
      .context-item.wide { grid-column: auto; }
      .status { right: 10px; bottom: 10px; }
    }
    @media (max-width: 420px) { .radio-grid { grid-template-columns: 1fr; } }
    @media (prefers-reduced-motion: reduce) { *,*::before,*::after { scroll-behavior: auto !important; transition: none !important; } }
  </style>
</head>
<body>
  <a class="skip-link" href="#main-content">跳到当前仲裁案例</a>
  <header class="app-header">
    <div class="brand"><h1>LoreGuard OOC 离线仲裁台</h1><p>先对照两份已冻结人工提交，再明确形成最终裁决。</p></div>
    <div class="toolbar" aria-label="仲裁文件操作">
      <button id="save-draft" type="button">保存本地草稿</button>
      <button id="import-trigger" type="button">导入本页锁定 JSON</button>
      <input class="file-input" id="import-file" type="file" accept="application/json,.json" aria-label="选择本页生成的锁定仲裁 JSON">
      <button class="danger" id="clear-draft" type="button">清除本地草稿</button>
      <button class="lock" id="download-locked" type="button">校验并下载锁定 JSON</button>
      <button id="unlock-editing" type="button" hidden>继续修订</button>
    </div>
  </header>
  <div class="freeze-strip">
    <div><strong>双份提交已冻结</strong>　此页未载入系统预测或私有答案，也不会联网。</div>
    <div class="binding" id="binding-summary"></div>
    <div class="binding freeze-binding" id="freeze-summary"></div>
  </div>
  <section class="error-summary" id="error-summary" role="alert" tabindex="-1" hidden>
    <h2>锁定前还需修正</h2><ul id="error-list"></ul>
  </section>
  <main class="workspace" id="main-content" tabindex="-1">
    <nav class="case-rail" aria-label="仲裁案例导航">
      <div class="rail-head"><h2>案例</h2><p id="progress-label"></p></div>
      <div class="case-list" id="case-list"></div>
      <div class="case-mobile"><label for="case-select">当前案例</label><select id="case-select"></select></div>
    </nav>
    <article class="case-work" id="case-work"></article>
  </main>
  <div class="status" id="status" aria-live="polite" aria-atomic="true"></div>
  <script id="adjudication-data" type="application/json">__ADJUDICATION_DATA__</script>
  <script>
  (() => {
    "use strict";

    const DATA = JSON.parse(document.getElementById("adjudication-data").textContent);
    const ROLES = [["", "不采用"], ["B", "B · 稳定基线"], ["C", "C · 当前行为"], ["G", "G · 已发生成长"], ["X", "X · 有效例外"], ["P", "P · 可能解释"]];
    const DIMENSIONS = [["core_trait", "核心性格"], ["stable_preference", "长期偏好"], ["speech_pattern", "说话方式"], ["value_boundary", "价值观／行为边界"], ["relationship_attitude", "关系态度"], ["motivation_goal", "长期动机／目标"]];
    const OUTCOMES = [["conflict", "conflict · 正式冲突"], ["no_issue", "no_issue · 无问题"], ["indeterminate", "indeterminate · 待复核"]];
    const SURFACES = [["formal_issue", "formal_issue · 正式问题"], ["none", "none · 不展示"], ["review_clue", "review_clue · 复核线索"]];
    const LEVELS = { L0: { outcome: "no_issue", surface: "none" }, L1: { outcome: "indeterminate", surface: "review_clue" }, L2: { outcome: "indeterminate", surface: "review_clue" }, L3: { outcome: "conflict", surface: "formal_issue" } };
    const CASE_KEYS = ["case_id", "dimension", "final_outcome", "final_surface", "conflict_level", "material_coverage", "required_evidence", "forbidden_evidence_ids", "independent_event_groups", "explanation_links", "phenomena", "reason_codes", "resolution_mode", "adjudicator_id", "adjudicator_kind", "adjudicator_blinded_to_system_prediction", "resolution_note"];
    const BUNDLE_KEYS = ["schema_version", "dataset_id", "public_input_sha256", "reviewer_ids", "annotation_sha256", "consensus_confirmed_by", "cases"];
    const AGREEMENT_FIELDS = [
      ["dimension", "维度", "dimension_agrees"], ["finalOutcome", "结果", "outcome_agrees"],
      ["finalSurface", "展示面", "surface_agrees"], ["conflictLevel", "冲突等级", "conflict_level_agrees"],
      ["materialCoverage", "材料覆盖", "material_coverage_agrees"], ["evidence", "证据角色", "evidence_agrees"],
      ["eventGroups", "C 事件组", "event_groups_agree"], ["explanations", "解释绑定", "explanation_links_agree"],
      ["phenomena", "现象标签", "phenomena_agree"], ["reasonCodes", "原因码", "reason_codes_agree"]
    ];
    const ui = {
      work: document.getElementById("case-work"), caseList: document.getElementById("case-list"),
      caseSelect: document.getElementById("case-select"), progress: document.getElementById("progress-label"),
      status: document.getElementById("status"), errorSummary: document.getElementById("error-summary"),
      errorList: document.getElementById("error-list"), importFile: document.getElementById("import-file"),
      download: document.getElementById("download-locked"), unlock: document.getElementById("unlock-editing")
    };
    const draftKey = ["loreguard-ooc-adjudication-draft-v1", DATA.public_input_sha256, DATA.author_setup_sha256, DATA.execution_freeze_sha256, ...DATA.reviewers.map(row => row.annotation_sha256), DATA.third_human_id || "consensus-only"].join(":");

    function blankCase() {
      return { source: "", finalOutcome: "", finalSurface: "", conflictLevel: "", materialCoverage: "complete", evidenceRoles: {}, forbidden: {}, eventGroups: {}, explanations: {}, phenomena: "", reasonCodes: "", resolutionMode: DATA.default_resolution_mode, resolutionNote: "" };
    }
    function blankConfirmations() { return { reviewerA: false, reviewerB: false, thirdHuman: false, thirdHumanBlind: false }; }
    const state = { activeCase: 0, locked: false, confirmations: blankConfirmations(), cases: DATA.cases.map(blankCase) };
    const evidenceMetaCache = new Map();

    function element(tag, className, text) { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; }
    function exactKeys(value, keys) { if (!value || typeof value !== "object" || Array.isArray(value)) return false; const actual = Object.keys(value).sort(); const expected = [...keys].sort(); return actual.length === expected.length && actual.every((key, index) => key === expected[index]); }
    function deepEqual(left, right) { if (Object.is(left, right)) return true; if (Array.isArray(left) || Array.isArray(right)) return Array.isArray(left) && Array.isArray(right) && left.length === right.length && left.every((value, index) => deepEqual(value, right[index])); if (!left || !right || typeof left !== "object" || typeof right !== "object") return false; const a = Object.keys(left).sort(), b = Object.keys(right).sort(); return a.length === b.length && a.every((key, index) => key === b[index] && deepEqual(left[key], right[key])); }
    function announce(message) { ui.status.textContent = message; window.clearTimeout(announce.timer); announce.timer = window.setTimeout(() => { ui.status.textContent = ""; }, 4500); }
    function parseTags(value) { const seen = new Set(); return String(value || "").split(/[\n,，]+/).map(item => item.trim()).filter(item => { if (!item || seen.has(item)) return false; seen.add(item); return true; }); }
    function makeSelect(options, value) { const select = element("select"); options.forEach(([key, label]) => { const option = element("option", "", label); option.value = key; option.selected = key === value; select.append(option); }); return select; }
    function formField(labelText, control, helper, id) { const wrap = element("div", "field"), label = element("label", "", labelText); label.htmlFor = id; control.id = id; wrap.append(label, control); if (helper) { const note = element("p", "helper", helper); note.id = `${id}-help`; control.setAttribute("aria-describedby", note.id); wrap.append(note); } return wrap; }
    function radioGroup(name, options, value) { const group = element("div", "radio-grid"); options.forEach(([key, label, help, disabled]) => { const wrap = element("label", "radio-card"), input = document.createElement("input"); input.type = "radio"; input.name = name; input.value = key; input.checked = key === value; input.disabled = Boolean(disabled); const copy = element("span", "", label); if (help) copy.append(element("small", "helper", help)); wrap.append(input, copy); group.append(wrap); }); return group; }
    function evidenceMeta(caseIndex) { if (evidenceMetaCache.has(caseIndex)) return evidenceMetaCache.get(caseIndex); const map = new Map(), caseRow = DATA.cases[caseIndex]; caseRow.documents.forEach(documentRow => documentRow.lines.forEach(line => { if (line.evidence_id) map.set(line.evidence_id, { evidenceId: line.evidence_id, documentId: documentRow.document_id, path: documentRow.path, lineNumber: line.line_number, text: line.text, phase: documentRow.phase, isTarget: documentRow.is_target, isSelector: documentRow.document_id === caseRow.candidate_selector.source_document_id && caseRow.candidate_selector.source_line_start <= line.line_number && line.line_number <= caseRow.candidate_selector.source_line_end }); })); evidenceMetaCache.set(caseIndex, map); return map; }
    function roleAllowed(caseIndex, evidenceId, role) { if (!role || role === "X" || role === "P") return true; const item = evidenceMeta(caseIndex).get(evidenceId); if (!item) return false; if (role === "B") return item.isSelector; if (role === "C") return item.isTarget; if (role === "G") return item.phase === "baseline"; return false; }
    function reviewerRole(row, evidenceId) { return row.evidence.find(item => item.evidence_id === evidenceId)?.role || "—"; }
    function caseStarted(row) { return Boolean(row.source || row.finalOutcome || row.finalSurface || row.conflictLevel || row.resolutionNote.trim() || Object.values(row.evidenceRoles).some(Boolean) || Object.values(row.forbidden).some(Boolean)); }
    function caseApparentlyComplete(row) { return Boolean(row.finalOutcome && row.finalSurface && row.conflictLevel && row.materialCoverage && parseTags(row.phenomena).length && parseTags(row.reasonCodes).length && row.resolutionMode && row.resolutionNote.trim()); }
    function formatEvidence(caseIndex, rows) { const meta = evidenceMeta(caseIndex); return rows.length ? rows.map(item => { const place = meta.get(item.evidence_id); return `${item.role} · ${place ? `${place.path} 第 ${place.lineNumber} 行` : item.evidence_id}`; }).join("\n") : "未采用"; }
    function formatGroups(caseIndex, groups) { const meta = evidenceMeta(caseIndex); return groups.length ? groups.map(group => `${group.event_group_id}: ${group.evidence_ids.map(id => meta.get(id) ? `第 ${meta.get(id).lineNumber} 行` : id).join("、")}`).join("\n") : "无"; }
    function formatLinks(caseIndex, links) { const meta = evidenceMeta(caseIndex); return links.length ? links.map(link => { const source = meta.get(link.support_evidence_id); const targets = link.applicable_current_ids.map(id => meta.get(id) ? `第 ${meta.get(id).lineNumber} 行` : id).join("、"); return `${source ? `第 ${source.lineNumber} 行` : link.support_evidence_id} → ${targets}；${link.causal_relation}/${link.temporal_relation}`; }).join("\n") : "无"; }
    function displayValue(caseIndex, field, row) { if (field === "dimension") return row.dimension; if (field === "finalOutcome") return row.outcome; if (field === "finalSurface") return row.surface; if (field === "conflictLevel") return row.conflict_level; if (field === "materialCoverage") return row.material_coverage; if (field === "evidence") return formatEvidence(caseIndex, row.evidence); if (field === "eventGroups") return formatGroups(caseIndex, row.independent_event_groups); if (field === "explanations") return formatLinks(caseIndex, row.explanation_links); if (field === "phenomena") return row.phenomena.join("、"); if (field === "reasonCodes") return row.reason_codes.join("、"); return ""; }

    function updateNavigation() {
      ui.caseList.textContent = ""; ui.caseSelect.textContent = ""; let complete = 0;
      DATA.cases.forEach((caseRow, index) => { const row = state.cases[index]; const status = caseApparentlyComplete(row) ? "已填写" : caseStarted(row) ? "进行中" : "未开始"; if (status === "已填写") complete += 1; const button = element("button", "case-button"); button.type = "button"; button.setAttribute("aria-current", index === state.activeCase ? "true" : "false"); button.title = caseRow.case_id; button.append(element("span", "case-number", String(index + 1).padStart(2, "0")), element("span", "", `案例 ${index + 1}`), element("span", "case-state", status)); button.addEventListener("click", () => switchCase(index)); ui.caseList.append(button); const option = element("option", "", `案例 ${index + 1} · ${status}`); option.value = String(index); option.selected = index === state.activeCase; ui.caseSelect.append(option); });
      ui.progress.textContent = `${complete} / ${DATA.cases.length} 个案例已填写主要字段`;
    }
    function switchCase(index, focus = true) { if (!Number.isInteger(index) || index < 0 || index >= DATA.cases.length) return; state.activeCase = index; updateNavigation(); renderCase(); scheduleDraft(false); if (focus) document.getElementById("case-heading")?.focus(); }

    function copyReviewer(caseIndex, side) {
      if (state.locked) return;
      const source = side === "A" ? DATA.cases[caseIndex].reviewer_a : DATA.cases[caseIndex].reviewer_b;
      const previousMode = state.cases[caseIndex].resolutionMode;
      const previousNote = state.cases[caseIndex].resolutionNote;
      const next = blankCase(); next.source = side; next.finalOutcome = source.outcome; next.finalSurface = source.surface; next.conflictLevel = source.conflict_level; next.materialCoverage = source.material_coverage; next.phenomena = source.phenomena.join("\n"); next.reasonCodes = source.reason_codes.join("\n"); next.resolutionMode = previousMode; next.resolutionNote = previousNote;
      source.evidence.forEach(item => { next.evidenceRoles[item.evidence_id] = item.role; });
      source.independent_event_groups.forEach(group => group.evidence_ids.forEach(id => { next.eventGroups[id] = group.event_group_id; }));
      source.explanation_links.forEach(link => { next.explanations[link.support_evidence_id] = { applicableCurrentIds: [...link.applicable_current_ids], causalRelation: link.causal_relation, temporalRelation: link.temporal_relation }; });
      state.cases[caseIndex] = next; updateNavigation(); renderCase(); scheduleDraft(); announce(`已把标注者 ${side} 的提交作为可编辑起点；尚未形成最终裁决。`);
    }

    function adoptField(caseIndex, field) {
      if (state.locked) return false;
      const caseRow = DATA.cases[caseIndex], source = caseRow.reviewer_a, row = state.cases[caseIndex];
      if (field === "dimension") {
        announce(source.dimension === caseRow.dimension ? "维度已由作者冻结，无需采纳。 " : "冻结维度与提交不一致，已拒绝采纳。 ");
        return false;
      }
      if (field === "finalOutcome") row.finalOutcome = source.outcome;
      if (field === "finalSurface") row.finalSurface = source.surface;
      if (field === "conflictLevel") row.conflictLevel = source.conflict_level;
      if (field === "materialCoverage") row.materialCoverage = source.material_coverage;
      if (field === "evidence") { row.evidenceRoles = {}; source.evidence.forEach(item => { row.evidenceRoles[item.evidence_id] = item.role; row.forbidden[item.evidence_id] = false; }); }
      if (field === "eventGroups") { row.eventGroups = {}; source.independent_event_groups.forEach(group => group.evidence_ids.forEach(id => { row.eventGroups[id] = group.event_group_id; })); }
      if (field === "explanations") { row.explanations = {}; source.explanation_links.forEach(link => { row.explanations[link.support_evidence_id] = { applicableCurrentIds: [...link.applicable_current_ids], causalRelation: link.causal_relation, temporalRelation: link.temporal_relation }; }); }
      if (field === "phenomena") row.phenomena = source.phenomena.join("\n");
      if (field === "reasonCodes") row.reasonCodes = source.reason_codes.join("\n");
      return true;
    }
    function adoptAgreed(caseIndex, onlyField = null) { const agreement = DATA.cases[caseIndex].agreement; let adopted = 0; AGREEMENT_FIELDS.forEach(([field, , flag]) => { if ((!onlyField || field === onlyField) && agreement[flag] && adoptField(caseIndex, field)) adopted += 1; }); updateNavigation(); renderCase(); if (adopted) scheduleDraft(); announce(adopted ? `已填入 ${adopted} 个一致项；仍需人工确认其余字段并写明裁决说明。` : onlyField === "dimension" ? "维度已由作者冻结，无需采纳。 " : "没有可快速采纳的一致项。 "); }

    function renderEvidence(caseIndex) {
      const caseRow = DATA.cases[caseIndex], caseState = state.cases[caseIndex], host = element("div");
      caseRow.documents.forEach((documentRow, documentIndex) => {
        const panel = element("section", "panel"), head = element("header", "panel-head"), title = element("h3", "", documentRow.path);
        panel.dataset.target = String(documentRow.is_target);
        panel.dataset.selectorSource = String(documentRow.is_selector_source);
        title.id = `document-${caseIndex}-${documentIndex}`;
        panel.setAttribute("aria-labelledby", title.id);
        head.append(
          title,
          element("p", "", `冻结文档 v${documentRow.version} · ${documentRow.sha256}`),
          element("p", "", `${documentRow.phase === "target" ? "目标文档" : "基线文档"} · ${documentRow.document_role} · ${documentRow.story_scope} · ${documentRow.resolution_state}/${documentRow.publication_status} · 导入序 ${documentRow.import_order}`)
        );
        panel.append(head);
        const lines = element("div", "document-lines");
        documentRow.lines.forEach(line => {
          if (!line.evidence_id) { lines.append(element("div", "blank-line")); return; }
          const evidenceId = line.evidence_id;
          const isSelector = documentRow.document_id === caseRow.candidate_selector.source_document_id && caseRow.candidate_selector.source_line_start <= line.line_number && line.line_number <= caseRow.candidate_selector.source_line_end;
          const row = element("div", "evidence-row");
          row.dataset.finalRole = caseState.evidenceRoles[evidenceId] || "";
          row.dataset.forbidden = caseState.forbidden[evidenceId] ? "true" : "false";
          row.dataset.selector = String(isSelector);
          row.append(element("span", "line-no", String(line.line_number)), element("p", "line-copy", line.text));
          const badges = element("div", "evidence-meta");
          badges.append(element("span", "role-chip", `A：${reviewerRole(caseRow.reviewer_a, evidenceId)}`), element("span", "role-chip", `B：${reviewerRole(caseRow.reviewer_b, evidenceId)}`));
          if (isSelector) badges.append(element("span", "role-chip", "作者 selector"));
          row.append(badges);
          const controls = element("div", "evidence-controls"), role = element("select");
          role.id = `evidence-${caseIndex}-${documentIndex}-${line.line_number}`;
          ROLES.forEach(([value, label]) => { const option = element("option", "", label); option.value = value; option.selected = value === (caseState.evidenceRoles[evidenceId] || ""); option.disabled = !roleAllowed(caseIndex, evidenceId, value); role.append(option); });
          role.setAttribute("aria-label", `${documentRow.path} 第 ${line.line_number} 行最终必要证据角色`);
          role.disabled = state.locked;
          role.addEventListener("keydown", event => {
            const key = event.key.toUpperCase();
            if (!["B","C","G","X","P"].includes(key) || event.ctrlKey || event.altKey || event.metaKey) return;
            event.preventDefault();
            if (!roleAllowed(caseIndex, evidenceId, key)) {
              announce(`${key} 不能用于这一行；B 仅限 selector，C 仅限目标文档，G 仅限基线文档。`);
              return;
            }
            role.value = key;
            role.dispatchEvent(new Event("change", { bubbles: true }));
          });
          role.addEventListener("change", () => {
            if (!roleAllowed(caseIndex, evidenceId, role.value)) {
              role.value = "";
              announce("该证据角色不符合冻结来源约束，已拒绝。 ");
            }
            caseState.evidenceRoles[evidenceId] = role.value;
            if (role.value) caseState.forbidden[evidenceId] = false;
            renderCase();
            scheduleDraft();
          });
          const forbiddenLabel = element("label", "forbidden-check"), forbidden = document.createElement("input");
          forbidden.type = "checkbox";
          forbidden.checked = Boolean(caseState.forbidden[evidenceId]);
          forbidden.disabled = state.locked;
          forbidden.addEventListener("change", () => {
            caseState.forbidden[evidenceId] = forbidden.checked;
            if (forbidden.checked) caseState.evidenceRoles[evidenceId] = "";
            renderCase();
            scheduleDraft();
          });
          forbiddenLabel.append(forbidden, element("span", "", "标为禁止证据"));
          controls.append(role, forbiddenLabel);
          row.append(controls);
          lines.append(row);
        });
        panel.append(lines);
        host.append(panel);
      });
      return host;
    }

    function renderComparison(caseIndex) {
      const caseRow = DATA.cases[caseIndex], panel = element("section", "panel comparison");
      const actions = element("div", "comparison-actions"), left = element("button", "", `以 A · ${DATA.reviewers[0].reviewer_id} 为起点`), right = element("button", "", `以 B · ${DATA.reviewers[1].reviewer_id} 为起点`), agreed = element("button", "", "填入全部一致项"); [left, right, agreed].forEach(button => { button.type = "button"; button.disabled = state.locked; }); left.addEventListener("click", () => copyReviewer(caseIndex, "A")); right.addEventListener("click", () => copyReviewer(caseIndex, "B")); agreed.addEventListener("click", () => adoptAgreed(caseIndex)); actions.append(left, right, agreed); panel.append(actions);
      AGREEMENT_FIELDS.forEach(([field, label, flag]) => { const agrees = caseRow.agreement[flag], row = element("div", `compare-row ${agrees ? "agrees" : "disagrees"}`), labelCell = element("div", "compare-label", label); labelCell.append(element("small", "", agrees ? "一致" : "有分歧")); if (agrees) { const adopt = element("button", "adopt-one", "采纳一致值"); adopt.type = "button"; adopt.disabled = state.locked; adopt.addEventListener("click", () => adoptAgreed(caseIndex, field)); labelCell.append(adopt); } const a = element("div", "compare-value"), b = element("div", "compare-value"); a.append(element("strong", "", "标注者 A"), document.createTextNode(displayValue(caseIndex, field, caseRow.reviewer_a))); b.append(element("strong", "", "标注者 B"), document.createTextNode(displayValue(caseIndex, field, caseRow.reviewer_b))); row.append(labelCell, a, b); panel.append(row); });
      const notes = element("div", "review-notes"); [["A", caseRow.reviewer_a], ["B", caseRow.reviewer_b]].forEach(([side, source]) => { const note = element("article", "review-note"); note.append(element("h4", "", `标注者 ${side} 的人工说明`), element("p", "", source.review_note)); notes.append(note); }); panel.append(notes); return panel;
    }

    function renderStructures(caseIndex, host) {
      const row = state.cases[caseIndex], meta = evidenceMeta(caseIndex), required = Object.entries(row.evidenceRoles).filter(([id, role]) => meta.has(id) && role), currents = required.filter(([, role]) => role === "C").map(([id]) => id), supports = required.filter(([, role]) => ["G","X","P"].includes(role));
      const groupTitle = element("h4", "", "C 事件组"), groups = element("div"); host.append(groupTitle, groups); if (!currents.length) groups.append(element("p", "structure-empty", "把最终必要证据标为 C 后，在这里记录事件组。")); currents.forEach((id, index) => { const info = meta.get(id), wrap = element("div", "structure-row"); wrap.append(element("h5", "", `C${String(index + 1).padStart(2,"0")} · ${info.path} 第 ${info.lineNumber} 行`)); const input = document.createElement("input"); input.type = "text"; input.maxLength = 200; input.value = row.eventGroups[id] || ""; input.disabled = state.locked; input.addEventListener("input", () => { row.eventGroups[id] = input.value; scheduleDraft(); }); wrap.append(formField("事件组 ID", input, "同一事件填相同 ID；独立事件填不同 ID。", `event-${caseIndex}-${index}`)); groups.append(wrap); });
      const linkTitle = element("h4", "", "G／X／P 到 C 的解释绑定"), links = element("div"); host.append(linkTitle, links); if (!supports.length) links.append(element("p", "structure-empty", "采用 G、X 或 P 后，在这里绑定其解释的 C。")); supports.forEach(([id, role], supportIndex) => { const info = meta.get(id), existing = row.explanations[id] || { applicableCurrentIds: [], causalRelation: role === "P" ? "ambiguous" : "bounded", temporalRelation: role === "G" ? "before" : role === "X" ? "active_during" : "unknown" }; row.explanations[id] = existing; if ((role === "G" || role === "X") && existing.causalRelation === "ambiguous") existing.causalRelation = "bounded"; if (role === "G") existing.temporalRelation = "before"; if (role === "X") existing.temporalRelation = "active_during"; existing.applicableCurrentIds = (existing.applicableCurrentIds || []).filter(target => currents.includes(target)); const wrap = element("div", "structure-row"); wrap.append(element("h5", "", `${role} · ${info.path} 第 ${info.lineNumber} 行`), element("p", "helper", role === "P" ? "P 可不建立链接；一旦选择 C 就写入绑定。" : `${role} 必须至少绑定一个 C。`)); const targets = element("div", "targets"); currents.forEach((target, targetIndex) => { const targetInfo = meta.get(target), label = element("label", "target-check"), checkbox = document.createElement("input"); checkbox.type = "checkbox"; checkbox.checked = existing.applicableCurrentIds.includes(target); checkbox.disabled = state.locked; checkbox.addEventListener("change", () => { const selected = new Set(existing.applicableCurrentIds); checkbox.checked ? selected.add(target) : selected.delete(target); existing.applicableCurrentIds = [...selected]; scheduleDraft(); }); label.append(checkbox, element("span", "", `C${String(targetIndex + 1).padStart(2,"0")} · 第 ${targetInfo.lineNumber} 行`)); targets.append(label); }); if (!currents.length) targets.append(element("p", "structure-empty", "尚未采用 C，无法建立解释绑定。")); wrap.append(targets); const grid = element("div", "structure-grid"), causalWrap = element("div"), causalLabel = element("label", "", "因果关系"), causalOptions = role === "P" ? [["explicit","明确因果"],["bounded","有限解释"],["ambiguous","含糊／不确定"]] : [["explicit","明确因果"],["bounded","有限解释"]], causal = makeSelect(causalOptions, existing.causalRelation); causal.id = `causal-${caseIndex}-${supportIndex}`; causal.disabled = state.locked; causalLabel.htmlFor = causal.id; causal.addEventListener("change", () => { existing.causalRelation = causal.value; scheduleDraft(); }); causalWrap.append(causalLabel, causal); const temporalWrap = element("div"), temporalLabel = element("label", "", "时间关系"), temporalOptions = role === "G" ? [["before","发生在 C 之前"]] : role === "X" ? [["active_during","C 发生时有效"]] : [["before","发生在 C 之前"],["active_during","C 发生时有效"],["after","发生在 C 之后"],["unknown","时间不明"]], temporal = makeSelect(temporalOptions, existing.temporalRelation); temporal.id = `temporal-${caseIndex}-${supportIndex}`; temporal.disabled = state.locked || role === "G" || role === "X"; temporalLabel.htmlFor = temporal.id; temporal.addEventListener("change", () => { existing.temporalRelation = temporal.value; scheduleDraft(); }); temporalWrap.append(temporalLabel, temporal); grid.append(causalWrap, temporalWrap); wrap.append(grid); links.append(wrap); });
    }

    function renderAdjudication(caseIndex) {
      const row = state.cases[caseIndex], caseRow = DATA.cases[caseIndex], panel = element("section", "panel adjudication");
      panel.append(element("h3", "", "最终人工裁决"), element("p", "", "快速采纳只填入字段，不会自动定案；请检查证据并完成说明。"));
      const locked = element("div", "locked-banner", "此工作副本已锁定。继续修订会生成新文件，不会改变已下载 JSON。");
      locked.hidden = !state.locked;
      panel.append(locked);
      const errors = element("div", "case-errors");
      errors.id = "case-errors";
      errors.setAttribute("role", "alert");
      errors.hidden = true;
      panel.append(errors);
      const fields = document.createElement("fieldset");
      fields.disabled = state.locked;

      const labels = element("section", "form-section");
      labels.append(element("h4", "", "最终标签"));
      const dimensionField = element("div", "field"), dimensionLabel = element("div", "field-label", "OOC 维度 · 已冻结"), dimension = element("div", "derived-result", `${DIMENSIONS.find(([value]) => value === caseRow.dimension)?.[1] || caseRow.dimension}（${caseRow.dimension}）`);
      dimension.id = `dimension-${caseIndex}`;
      dimension.tabIndex = -1;
      dimensionField.append(dimensionLabel, dimension, element("p", "helper", "维度由作者冻结的 trait_type 映射，仲裁者不能改选。"));
      labels.append(dimensionField);
      const labelGrid = element("div", "form-grid"), outcome = makeSelect([["","请选择结果"], ...OUTCOMES], row.finalOutcome), surface = makeSelect([["","请选择展示面"], ...SURFACES], row.finalSurface);
      outcome.addEventListener("change", () => { row.finalOutcome = outcome.value; scheduleDraft(); });
      surface.addEventListener("change", () => { row.finalSurface = surface.value; scheduleDraft(); });
      labelGrid.append(formField("最终 outcome *", outcome, "必须与 L0–L3 对应。", `outcome-${caseIndex}`), formField("最终 surface *", surface, "必须与 outcome 对应。", `surface-${caseIndex}`));
      labels.append(labelGrid);
      const level = makeSelect([["","请选择等级"],["L0","L0 · 无问题"],["L1","L1 · 轻度待复核"],["L2","L2 · 明确待复核"],["L3","L3 · 正式冲突"]], row.conflictLevel);
      level.addEventListener("change", () => { row.conflictLevel = level.value; scheduleDraft(); });
      const coverage = makeSelect([["complete","完整"],["partial","部分缺失"],["missing","关键材料缺失"]], row.materialCoverage);
      coverage.addEventListener("change", () => { row.materialCoverage = coverage.value; scheduleDraft(); });
      const secondGrid = element("div", "form-grid");
      secondGrid.append(formField("冲突等级 *", level, "L3 需要 B、两条 C 和两个独立事件组。", `level-${caseIndex}`), formField("材料覆盖 *", coverage, "不完整时只能为 L1 或 L2。", `coverage-${caseIndex}`));
      labels.append(secondGrid);
      fields.append(labels);

      const structures = element("section", "form-section");
      renderStructures(caseIndex, structures);
      fields.append(structures);

      const codes = element("section", "form-section");
      codes.append(element("h4", "", "标签、模式与说明"));
      const phenomena = document.createElement("textarea");
      phenomena.value = row.phenomena;
      phenomena.maxLength = 6500;
      phenomena.addEventListener("input", () => { row.phenomena = phenomena.value; scheduleDraft(); });
      const reasons = document.createElement("textarea");
      reasons.value = row.reasonCodes;
      reasons.maxLength = 6500;
      reasons.addEventListener("input", () => { row.reasonCodes = reasons.value; scheduleDraft(); });
      codes.append(formField("现象标签 *", phenomena, "每行或逗号分隔；1–64 个，每个不超过 100 字符。", `phenomena-${caseIndex}`), formField("原因码 *", reasons, "使用中性、可复核短语。", `reasons-${caseIndex}`));
      const mode = radioGroup(`mode-${caseIndex}`, [
        ["reviewer_consensus","两位标注者共识","本案例最终结论由 A、B 共同确认。",false],
        ["third_human","第三人裁决",DATA.third_human_id ? `第三人：${DATA.third_human_id}；第三人需在下方主动确认身份与系统预测盲法。` : "生成页面时未绑定第三人。",!DATA.third_human_id]
      ], row.resolutionMode);
      mode.id = `mode-${caseIndex}`;
      mode.addEventListener("change", event => { if (event.target instanceof HTMLInputElement) { row.resolutionMode = event.target.value; scheduleDraft(); } });
      const modeField = element("div", "field");
      modeField.append(element("div", "field-label", "裁决模式 *"), mode);
      codes.append(modeField);
      const note = document.createElement("textarea");
      note.value = row.resolutionNote;
      note.maxLength = 8000;
      note.addEventListener("input", () => { row.resolutionNote = note.value; updateNavigation(); scheduleDraft(); });
      codes.append(formField("裁决说明 *", note, "说明采用或否决了哪些证据、如何处理分歧；第三人裁决还要说明第三人复核流程；1–8000 字符。", `note-${caseIndex}`));
      fields.append(codes);

      const confirmations = element("section", "confirmation-panel");
      confirmations.id = "final-confirmations";
      confirmations.tabIndex = -1;
      confirmations.append(element("h4", "", "锁定前最终确认"), element("p", "helper", "页面不会默认勾选。按当前协议，即使部分案例由第三人裁决，A、B 仍须确认整份最终仲裁稿。"));
      [
        ["reviewerA", `标注者 A（${DATA.reviewers[0].reviewer_id}）已查看并确认最终仲裁稿。`],
        ["reviewerB", `标注者 B（${DATA.reviewers[1].reviewer_id}）已查看并确认最终仲裁稿。`]
      ].forEach(([key, copy]) => {
        const label = element("label", "target-check"), checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = state.confirmations[key];
        checkbox.addEventListener("change", () => { state.confirmations[key] = checkbox.checked; scheduleDraft(false); });
        label.append(checkbox, element("span", "", copy));
        confirmations.append(label);
      });
      if (DATA.third_human_id) {
        [
          ["thirdHuman", `第三人（${DATA.third_human_id}）确认由本人完成人工裁决。`],
          ["thirdHumanBlind", `第三人（${DATA.third_human_id}）确认裁决期间未查看系统／模型预测。`]
        ].forEach(([key, copy]) => {
          const label = element("label", "target-check"), checkbox = document.createElement("input");
          checkbox.type = "checkbox";
          checkbox.checked = state.confirmations[key];
          checkbox.addEventListener("change", () => { state.confirmations[key] = checkbox.checked; scheduleDraft(false); });
          label.append(checkbox, element("span", "", copy));
          confirmations.append(label);
        });
      }
      fields.append(confirmations);
      panel.append(fields);
      return panel;
    }

    function renderCase() {
      const caseIndex = state.activeCase, caseRow = DATA.cases[caseIndex];
      ui.work.textContent = "";
      const intro = element("header", "case-intro"), heading = element("h2", "", `案例 ${caseIndex + 1} · 双人仲裁`);
      heading.id = "case-heading";
      heading.tabIndex = -1;
      const diffCount = AGREEMENT_FIELDS.filter(([, , flag]) => !caseRow.agreement[flag]).length;
      intro.append(heading, element("p", "", `案例 ID：${caseRow.case_id}`), element("p", "", `轴快照：${caseRow.axis.axis_id} · v${caseRow.axis.version}`), element("p", "", `${diffCount} / ${AGREEMENT_FIELDS.length} 个协议比较字段存在分歧`));
      const context = element("section", "target-context");
      context.append(element("h3", "", "本案例的作者冻结目标"));
      const contextGrid = element("dl", "context-grid"), addContext = (label, value, wide = false) => { const item = element("div", `context-item${wide ? " wide" : ""}`); item.append(element("dt", "", label), element("dd", "", value)); contextGrid.append(item); };
      const selectorDocument = caseRow.documents.find(row => row.document_id === caseRow.candidate_selector.source_document_id);
      addContext("目标角色", caseRow.candidate_selector.character_key);
      addContext("锁定维度", `${caseRow.candidate_selector.trait_type} → ${caseRow.dimension}`);
      addContext("基线 selector", `${selectorDocument?.path || caseRow.candidate_selector.source_document_id} · 第 ${caseRow.candidate_selector.source_line_start}–${caseRow.candidate_selector.source_line_end} 行`);
      addContext("selector 属性", `${caseRow.candidate_selector.origin} / ${caseRow.candidate_selector.polarity} / ${caseRow.candidate_selector.stability}`);
      if (caseRow.author_axis) {
        addContext("作者语义轴", caseRow.author_axis.display_name, true);
        addContext("定义", caseRow.author_axis.definition, true);
        addContext("正向命题", caseRow.author_axis.positive_proposition, true);
        addContext("适用范围", caseRow.author_axis.applicability_scope, true);
        addContext("轴向对齐", caseRow.author_axis.axis_alignment);
      } else {
        addContext("作者语义轴", "未额外提供；按冻结 selector 的稳定特征判断。", true);
      }
      context.append(contextGrid);
      intro.append(context, element("p", "shortcut", "证据角色可直接按 B、C、G、X、P；Alt + ↑/↓ 切换案例，Ctrl + S 保存草稿。"));
      ui.work.append(intro);
      const grid = element("div", "case-grid"), left = element("div"), right = element("div");
      left.append(renderEvidence(caseIndex));
      right.append(renderComparison(caseIndex), renderAdjudication(caseIndex));
      grid.append(left, right);
      ui.work.append(grid);
      document.getElementById("save-draft").disabled = state.locked;
      document.getElementById("import-trigger").disabled = state.locked;
      document.getElementById("clear-draft").disabled = state.locked;
      ui.unlock.hidden = !state.locked;
    }

    function buildCase(caseIndex) {
      const source = DATA.cases[caseIndex], row = state.cases[caseIndex], meta = evidenceMeta(caseIndex), required_evidence = Object.entries(row.evidenceRoles).filter(([id, role]) => meta.has(id) && ["B","C","G","X","P"].includes(role)).map(([evidence_id, role]) => ({ evidence_id, role })), requiredIds = new Set(required_evidence.map(item => item.evidence_id)), forbidden_evidence_ids = Object.entries(row.forbidden).filter(([id, value]) => value && meta.has(id) && !requiredIds.has(id)).map(([id]) => id), currentIds = new Set(required_evidence.filter(item => item.role === "C").map(item => item.evidence_id)), grouped = new Map(); currentIds.forEach(id => { const group = String(row.eventGroups[id] || "").trim(); if (group) { const values = grouped.get(group) || []; values.push(id); grouped.set(group, values); } }); const independent_event_groups = [...grouped.entries()].map(([event_group_id, evidence_ids]) => ({ event_group_id, evidence_ids })), explanation_links = [];
      required_evidence.filter(item => ["G","X","P"].includes(item.role)).forEach(item => { const link = row.explanations[item.evidence_id] || {}, targets = Array.isArray(link.applicableCurrentIds) ? [...new Set(link.applicableCurrentIds.filter(id => currentIds.has(id)))] : []; if (item.role === "G" || item.role === "X" || targets.length) { const allowed = item.role === "P" ? ["explicit","bounded","ambiguous"] : ["explicit","bounded"]; explanation_links.push({ support_evidence_id: item.evidence_id, applicable_current_ids: targets, causal_relation: allowed.includes(link.causalRelation) ? link.causalRelation : item.role === "P" ? "ambiguous" : "bounded", temporal_relation: item.role === "G" ? "before" : item.role === "X" ? "active_during" : ["before","active_during","after","unknown"].includes(link.temporalRelation) ? link.temporalRelation : "unknown" }); } });
      const third = row.resolutionMode === "third_human";
      return { case_id: source.case_id, dimension: source.dimension, final_outcome: row.finalOutcome, final_surface: row.finalSurface, conflict_level: row.conflictLevel, material_coverage: row.materialCoverage, required_evidence, forbidden_evidence_ids, independent_event_groups, explanation_links, phenomena: parseTags(row.phenomena), reason_codes: parseTags(row.reasonCodes), resolution_mode: row.resolutionMode, adjudicator_id: third ? DATA.third_human_id : null, adjudicator_kind: third && state.confirmations.thirdHuman ? "human" : null, adjudicator_blinded_to_system_prediction: third && state.confirmations.thirdHumanBlind ? true : null, resolution_note: row.resolutionNote.trim() };
    }
    function buildBundle() { return { schema_version: DATA.adjudication_schema_version, dataset_id: DATA.dataset_id, public_input_sha256: DATA.public_input_sha256, reviewer_ids: DATA.reviewers.map(row => row.reviewer_id), annotation_sha256: DATA.reviewers.map(row => row.annotation_sha256), consensus_confirmed_by: [state.confirmations.reviewerA ? DATA.reviewers[0].reviewer_id : null, state.confirmations.reviewerB ? DATA.reviewers[1].reviewer_id : null].filter(Boolean), cases: DATA.cases.map((_, index) => buildCase(index)) }; }

    function validateCase(caseIndex) {
      const row = buildCase(caseIndex), errors = [], add = (field, message) => errors.push({ caseIndex, field, message });
      if (row.dimension !== DATA.cases[caseIndex].dimension) add(`dimension-${caseIndex}`, "维度与作者冻结目标不一致。 ");
      if (!LEVELS[row.conflict_level]) add(`level-${caseIndex}`, "请选择 L0–L3。 ");
      if (!["conflict","no_issue","indeterminate"].includes(row.final_outcome)) add(`outcome-${caseIndex}`, "请选择最终 outcome。 ");
      if (!["formal_issue","none","review_clue"].includes(row.final_surface)) add(`surface-${caseIndex}`, "请选择最终 surface。 ");
      if (LEVELS[row.conflict_level] && (LEVELS[row.conflict_level].outcome !== row.final_outcome || LEVELS[row.conflict_level].surface !== row.final_surface)) add(`level-${caseIndex}`, "outcome、surface 与冲突等级不符合协议对应关系。 ");
      if (!["complete","partial","missing"].includes(row.material_coverage)) add(`coverage-${caseIndex}`, "请选择材料覆盖。 ");
      if (row.material_coverage !== "complete" && row.final_outcome !== "indeterminate") add(`coverage-${caseIndex}`, "材料不完整时 outcome 必须为 indeterminate。 ");
      const roles = role => row.required_evidence.filter(item => item.role === role).map(item => item.evidence_id), b = roles("B"), c = roles("C"), g = roles("G"), x = roles("X"), p = roles("P");
      row.required_evidence.forEach(item => { if (!roleAllowed(caseIndex, item.evidence_id, item.role)) add("evidence-final", "证据来源不符合冻结约束：B 仅限 selector，C 仅限目标文档，G 仅限基线文档。 "); });
      if (row.conflict_level === "L3") { if (!b.length || c.length < 2) add("evidence-final", "L3 至少需要 1 条 B 和 2 条 C。 "); if (g.length || x.length || p.length) add("evidence-final", "L3 不得采用 G、X 或 P。 "); const byEvidence = new Map(); row.independent_event_groups.forEach(group => group.evidence_ids.forEach(id => byEvidence.set(id, group.event_group_id))); const groups = new Set(c.map(id => byEvidence.get(id))); if (groups.has(undefined) || groups.size < 2) add("structures", "L3 的每条 C 都要分组，且至少属于两个独立事件组。 "); }
      if (["L1","L2"].includes(row.conflict_level) && !row.required_evidence.length) add("evidence-final", "L1／L2 至少需要一条必要证据。 ");
      const linked = new Set(row.explanation_links.map(link => link.support_evidence_id)); [...g,...x].forEach(id => { if (!linked.has(id)) add("structures", "每条 G 或 X 都必须绑定至少一个 C。 "); }); row.explanation_links.forEach(link => { if (!link.applicable_current_ids.length) add("structures", "解释绑定至少需要一个 C。 "); const role = row.required_evidence.find(item => item.evidence_id === link.support_evidence_id)?.role; if ((role === "G" || role === "X") && link.causal_relation === "ambiguous") add("structures", "含糊支持必须标为 P。 "); });
      if (!row.phenomena.length || row.phenomena.length > 64 || row.phenomena.some(value => value.length > 100)) add(`phenomena-${caseIndex}`, "现象标签需为 1–64 个非空值，每个不超过 100 字符。 ");
      if (!row.reason_codes.length || row.reason_codes.length > 64 || row.reason_codes.some(value => value.length > 100)) add(`reasons-${caseIndex}`, "原因码需为 1–64 个非空值，每个不超过 100 字符。 ");
      if (!["reviewer_consensus","third_human"].includes(row.resolution_mode)) add(`mode-${caseIndex}`, "请选择裁决模式。 ");
      if (row.resolution_mode === "third_human" && !DATA.third_human_id) add(`mode-${caseIndex}`, "第三人模式需要生成时冻结第三人 ID。 ");
      if (row.resolution_mode === "third_human" && (!state.confirmations.thirdHuman || !state.confirmations.thirdHumanBlind)) add("final-confirmations", "第三人裁决需要第三人主动确认本人裁决与系统预测盲法。 ");
      if (!row.resolution_note || row.resolution_note.length > 8000) add(`note-${caseIndex}`, "裁决说明需为 1–8000 字符。 ");
      return errors;
    }
    function validateAll() { const errors = DATA.cases.flatMap((_, index) => validateCase(index)); if (!state.confirmations.reviewerA) errors.push({ caseIndex: state.activeCase, field: "final-confirmations", message: "请由标注者 A 主动确认最终仲裁稿。 " }); if (!state.confirmations.reviewerB) errors.push({ caseIndex: state.activeCase, field: "final-confirmations", message: "请由标注者 B 主动确认最终仲裁稿。 " }); return errors; }
    function showErrors(errors) { ui.errorList.textContent = ""; errors.forEach(error => { const item = element("li"), button = element("button", "", `案例 ${error.caseIndex + 1}：${error.message.trim()}`); button.type = "button"; button.addEventListener("click", () => { switchCase(error.caseIndex, false); window.setTimeout(() => document.getElementById(error.field)?.focus(), 0); }); item.append(button); ui.errorList.append(item); }); ui.errorSummary.hidden = !errors.length; const local = errors.filter(error => error.caseIndex === state.activeCase), host = document.getElementById("case-errors"); if (host) { host.textContent = ""; host.hidden = !local.length; if (local.length) { host.append(element("strong", "", "本案例还需修正")); const list = element("ul"); local.forEach(error => list.append(element("li", "", error.message.trim()))); host.append(list); } } if (errors.length) ui.errorSummary.focus(); }

    function saveDraft(silent = false) { if (state.locked) return; try { localStorage.setItem(draftKey, JSON.stringify({ draft_version: 1, public_input_sha256: DATA.public_input_sha256, author_setup_sha256: DATA.author_setup_sha256, execution_freeze_sha256: DATA.execution_freeze_sha256, manual_version: DATA.manual_version, reviewer_ids: DATA.reviewers.map(row => row.reviewer_id), annotation_sha256: DATA.reviewers.map(row => row.annotation_sha256), third_human_id: DATA.third_human_id, active_case: state.activeCase, confirmations: state.confirmations, cases: state.cases })); if (!silent) announce("本地草稿已保存。数据没有离开此浏览器。 "); } catch (_) { if (!silent) announce("浏览器拒绝本地存储。请尽快下载锁定 JSON。 "); } }
    function invalidateFinalConfirmations() { if (!Object.values(state.confirmations).some(Boolean)) return; state.confirmations = blankConfirmations(); document.querySelectorAll('#final-confirmations input[type="checkbox"]').forEach(input => { input.checked = false; }); announce("仲裁内容已修改，最终确认已撤回；锁定前请 A、B 重新确认。 "); }
    function scheduleDraft(contentChanged = true) { if (state.locked) return; if (contentChanged) invalidateFinalConfirmations(); window.clearTimeout(scheduleDraft.timer); scheduleDraft.timer = window.setTimeout(() => saveDraft(true), 500); }
    function loadDraft() { try { const raw = localStorage.getItem(draftKey); if (!raw) return false; const draft = JSON.parse(raw), reviewerIds = DATA.reviewers.map(row => row.reviewer_id), digests = DATA.reviewers.map(row => row.annotation_sha256); if (draft.draft_version !== 1 || draft.public_input_sha256 !== DATA.public_input_sha256 || draft.author_setup_sha256 !== DATA.author_setup_sha256 || draft.execution_freeze_sha256 !== DATA.execution_freeze_sha256 || draft.manual_version !== DATA.manual_version || !deepEqual(draft.reviewer_ids, reviewerIds) || !deepEqual(draft.annotation_sha256, digests) || draft.third_human_id !== DATA.third_human_id || !Array.isArray(draft.cases) || draft.cases.length !== DATA.cases.length) return false; const confirmations = draft.confirmations || {}; state.confirmations = { reviewerA: confirmations.reviewerA === true, reviewerB: confirmations.reviewerB === true, thirdHuman: confirmations.thirdHuman === true, thirdHumanBlind: confirmations.thirdHumanBlind === true }; state.cases = draft.cases.map(row => ({ ...blankCase(), ...row, evidenceRoles: { ...(row.evidenceRoles || {}) }, forbidden: { ...(row.forbidden || {}) }, eventGroups: { ...(row.eventGroups || {}) }, explanations: { ...(row.explanations || {}) } })); state.activeCase = Number.isInteger(draft.active_case) && draft.active_case >= 0 && draft.active_case < DATA.cases.length ? draft.active_case : 0; return true; } catch (_) { return false; } }

    function stateFromLocked(caseIndex, row) { const next = blankCase(); if (row.dimension !== DATA.cases[caseIndex].dimension) throw new Error("导入文件的维度与作者冻结目标不一致。 "); next.source = "locked"; next.finalOutcome = row.final_outcome; next.finalSurface = row.final_surface; next.conflictLevel = row.conflict_level; next.materialCoverage = row.material_coverage; next.phenomena = row.phenomena.join("\n"); next.reasonCodes = row.reason_codes.join("\n"); next.resolutionMode = row.resolution_mode; next.resolutionNote = row.resolution_note; row.required_evidence.forEach(item => { next.evidenceRoles[item.evidence_id] = item.role; }); row.forbidden_evidence_ids.forEach(id => { next.forbidden[id] = true; }); row.independent_event_groups.forEach(group => group.evidence_ids.forEach(id => { next.eventGroups[id] = group.event_group_id; })); row.explanation_links.forEach(link => { next.explanations[link.support_evidence_id] = { applicableCurrentIds: [...link.applicable_current_ids], causalRelation: link.causal_relation, temporalRelation: link.temporal_relation }; }); const known = evidenceMeta(caseIndex); if ([...Object.keys(next.evidenceRoles), ...Object.keys(next.forbidden)].some(id => !known.has(id))) throw new Error("导入文件引用了本案例不存在的证据。 "); if (Object.entries(next.evidenceRoles).some(([id, role]) => !roleAllowed(caseIndex, id, role))) throw new Error("导入文件的 B/C/G 来源不符合作者冻结目标。 "); return next; }
    function validateImported(bundle) { if (!exactKeys(bundle, BUNDLE_KEYS)) throw new Error("导入文件不是当前 AdjudicationBundle 结构。 "); const reviewerIds = DATA.reviewers.map(row => row.reviewer_id), digests = DATA.reviewers.map(row => row.annotation_sha256); if (bundle.schema_version !== DATA.adjudication_schema_version || bundle.dataset_id !== DATA.dataset_id || bundle.public_input_sha256 !== DATA.public_input_sha256 || !deepEqual(bundle.reviewer_ids, reviewerIds) || !deepEqual(bundle.annotation_sha256, digests) || !deepEqual(bundle.consensus_confirmed_by, reviewerIds) || !Array.isArray(bundle.cases)) throw new Error("导入文件与当前公开数据或两份冻结提交不一致。 "); const byCase = new Map(); bundle.cases.forEach(row => { if (!exactKeys(row, CASE_KEYS) || byCase.has(row.case_id)) throw new Error("导入文件含重复案例或未知字段。 "); if (!Array.isArray(row.required_evidence) || !row.required_evidence.every(item => exactKeys(item, ["evidence_id","role"]))) throw new Error("导入文件的必要证据结构无效。 "); if (!Array.isArray(row.forbidden_evidence_ids) || !Array.isArray(row.independent_event_groups) || !row.independent_event_groups.every(item => exactKeys(item, ["event_group_id","evidence_ids"])) || !Array.isArray(row.explanation_links) || !row.explanation_links.every(item => exactKeys(item, ["support_evidence_id","applicable_current_ids","causal_relation","temporal_relation"])) || !Array.isArray(row.phenomena) || !Array.isArray(row.reason_codes)) throw new Error("导入文件的案例结构无效。 "); byCase.set(row.case_id, row); }); if (byCase.size !== DATA.cases.length || DATA.cases.some(row => !byCase.has(row.case_id))) throw new Error("导入文件没有完整覆盖公开案例。 "); const next = DATA.cases.map((caseRow, index) => stateFromLocked(index, byCase.get(caseRow.case_id))), previous = state.cases, previousConfirmations = state.confirmations; state.cases = next; const hasThird = bundle.cases.some(row => row.resolution_mode === "third_human"); state.confirmations = { reviewerA: true, reviewerB: true, thirdHuman: hasThird, thirdHumanBlind: hasThird }; const errors = validateAll(); if (errors.length) { state.cases = previous; state.confirmations = previousConfirmations; throw new Error(`导入文件未通过协议校验：${errors[0].message.trim()}`); } const rebuilt = new Map(DATA.cases.map((caseRow, index) => [caseRow.case_id, buildCase(index)])); if (bundle.cases.some(row => !deepEqual(row, rebuilt.get(row.case_id)))) { state.cases = previous; state.confirmations = previousConfirmations; throw new Error("导入文件不能由本页无损还原；可能含重复值或非法类型。 "); } return next; }

    function downloadLocked() { const errors = validateAll(); if (errors.length) { showErrors(errors); announce(`还有 ${errors.length} 项需要修正。`); return; } showErrors([]); const payload = JSON.stringify(buildBundle(), null, 2) + "\n", blob = new Blob([payload], { type: "application/json;charset=utf-8" }), url = URL.createObjectURL(blob), anchor = document.createElement("a"); anchor.href = url; anchor.download = `ooc-adjudication-${DATA.public_input_sha256.slice(0,12)}.locked.json`; document.body.append(anchor); anchor.click(); anchor.remove(); window.setTimeout(() => URL.revokeObjectURL(url), 1000); state.locked = true; try { localStorage.removeItem(draftKey); } catch (_) {} renderCase(); announce("锁定仲裁 JSON 已下载；当前工作副本已锁定。 "); }

    document.getElementById("binding-summary").textContent = `A ${DATA.reviewers[0].reviewer_id} · ${DATA.reviewers[0].annotation_sha256}　B ${DATA.reviewers[1].reviewer_id} · ${DATA.reviewers[1].annotation_sha256}`;
    document.getElementById("freeze-summary").textContent = `public ${DATA.public_input_sha256} · author setup ${DATA.author_setup_sha256} · execution freeze ${DATA.execution_freeze_sha256} · manual ${DATA.manual_version}`;
    document.getElementById("save-draft").addEventListener("click", () => saveDraft(false));
    document.getElementById("import-trigger").addEventListener("click", () => ui.importFile.click());
    document.getElementById("clear-draft").addEventListener("click", () => { if (!window.confirm("清除本浏览器中的仲裁草稿？已下载文件不受影响。")) return; try { localStorage.removeItem(draftKey); } catch (_) {} state.cases = DATA.cases.map(blankCase); state.confirmations = blankConfirmations(); state.activeCase = 0; showErrors([]); updateNavigation(); renderCase(); announce("本地草稿已清除。 "); });
    ui.download.addEventListener("click", downloadLocked);
    ui.unlock.addEventListener("click", () => { state.locked = false; renderCase(); saveDraft(true); announce("已进入修订状态；再次锁定会下载新文件。 "); });
    ui.caseSelect.addEventListener("change", () => switchCase(Number(ui.caseSelect.value)));
    ui.importFile.addEventListener("change", async () => { const file = ui.importFile.files && ui.importFile.files[0]; ui.importFile.value = ""; if (!file) return; if (file.size > 8 * 1024 * 1024) { announce("导入文件超过 8 MiB，未载入。 "); return; } try { const bundle = JSON.parse(await file.text()); state.cases = validateImported(bundle); state.activeCase = 0; state.locked = true; showErrors([]); updateNavigation(); renderCase(); announce("已导入并锁定与当前双份提交绑定的仲裁 JSON。 "); } catch (error) { announce(error instanceof Error ? error.message : "导入文件无效。 "); } });
    document.addEventListener("keydown", event => { if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") { event.preventDefault(); saveDraft(false); } if (event.altKey && event.key === "ArrowDown") { event.preventDefault(); switchCase(Math.min(DATA.cases.length - 1, state.activeCase + 1)); } if (event.altKey && event.key === "ArrowUp") { event.preventDefault(); switchCase(Math.max(0, state.activeCase - 1)); } });
    const restored = loadDraft(); updateNavigation(); renderCase(); if (restored) announce("已恢复与当前双份提交绑定的本地草稿。 ");
  })();
  </script>
</body>
</html>
'''


if __name__ == "__main__":
    raise SystemExit(main())
