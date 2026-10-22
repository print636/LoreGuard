"""Generate a self-contained, offline Simplified-Chinese OOC annotation page.

The generated page receives a frozen public input bundle, its answer-free author
setup, execution freeze, and document snapshots.  It contains no model prediction,
private gold, peer annotation, or network dependency.  A reviewer can autosave a
browser-local draft and export a JSON object matching ``ReviewerAnnotationBundle``
exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import ooc_annotation_workflow as workflow
from scripts import ooc_closed_bundle as closed_bundle
from scripts import ooc_eval_contract as contract


PAGE_SCHEMA_VERSION = "loreguard-ooc-annotation-page-v1"
MAX_IDENTITY_LENGTH = 200
MAX_MANUAL_VERSION_LENGTH = 100
TRAIT_DIMENSION = dict(workflow._TRAIT_DIMENSION)


class AnnotationPageError(ValueError):
    """Stable error raised for unsafe or inconsistent page-generation input."""


def _validate_text_identity(value: str, *, name: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise AnnotationPageError(f"{name}_invalid")
    return value


def _local_path(value: str | Path, *, name: str) -> Path:
    raw = str(value)
    if "://" in raw or raw.startswith(("\\\\", "//")):
        raise AnnotationPageError(f"{name}_must_be_local")
    return Path(value)


def _load_document_text(
    root: Path,
    document: contract.DocumentSnapshot,
) -> str:
    path = root.joinpath(*document.path.split("/"))
    try:
        if path.is_symlink():
            raise AnnotationPageError("public_document_symlink_forbidden")
        resolved = path.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(root):
            raise AnnotationPageError("public_document_path_escapes_bundle")
        if resolved.stat().st_size > contract.MAX_JSON_BYTES:
            raise AnnotationPageError("public_document_too_large")
        payload = resolved.read_bytes()
    except OSError as exc:
        raise AnnotationPageError("public_document_unreadable") from exc
    if hashlib.sha256(payload).hexdigest() != document.sha256:
        raise AnnotationPageError("public_document_sha256_mismatch")
    if b"\r" in payload:
        raise AnnotationPageError("public_document_must_use_lf")
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise AnnotationPageError("public_document_must_be_utf8") from exc


def build_page_data(
    public: contract.PublicInputBundle,
    bundle_root: str | Path,
    author_setup: closed_bundle.PublicAuthorSetupBundle,
    execution_freeze: closed_bundle.ExecutionFreeze,
    *,
    reviewer_id: str,
    manual_version: str,
) -> dict[str, Any]:
    """Validate public material and build the only data embedded in the page."""

    reviewer_id = _validate_text_identity(
        reviewer_id,
        name="reviewer_id",
        maximum=MAX_IDENTITY_LENGTH,
    )
    manual_version = _validate_text_identity(
        manual_version,
        name="manual_version",
        maximum=MAX_MANUAL_VERSION_LENGTH,
    )
    root_path = _local_path(bundle_root, name="bundle_root")
    workflow.validate_exhaustive_line_catalog(public, root_path)
    closed_bundle.validate_execution_freeze_binding(
        public,
        author_setup,
        execution_freeze,
    )
    root = root_path.resolve(strict=True)
    setup_cases = {row.case_id: row for row in author_setup.cases}

    cached_text: dict[tuple[str, str], str] = {}
    cases: list[dict[str, Any]] = []
    for case_index, case in enumerate(public.cases, start=1):
        setup_case = setup_cases[case.case_id]
        setup_documents = {
            row.document_id: row for row in setup_case.documents
        }
        anchors_by_document: dict[str, dict[int, contract.EvidenceAnchor]] = {}
        for anchor in case.evidence_catalog:
            anchors_by_document.setdefault(anchor.document_id, {})[
                anchor.line_start
            ] = anchor
        documents: list[dict[str, Any]] = []
        for document in case.documents:
            document_setup = setup_documents[document.document_id]
            cache_key = (document.path, document.sha256)
            text = cached_text.get(cache_key)
            if text is None:
                text = _load_document_text(root, document)
                cached_text[cache_key] = text
            anchors = anchors_by_document.get(document.document_id, {})
            lines = [
                {
                    "line_number": line_number,
                    "text": line,
                    "evidence_id": (
                        anchors[line_number].evidence_id
                        if line_number in anchors
                        else None
                    ),
                }
                for line_number, line in enumerate(text.splitlines(), start=1)
            ]
            documents.append(
                {
                    "document_id": document.document_id,
                    "path": document.path,
                    "version": document.version,
                    "sha256": document.sha256,
                    "phase": document_setup.phase,
                    "document_role": document_setup.document_role,
                    "story_scope": document_setup.story_scope,
                    "resolution_state": document_setup.resolution_state,
                    "publication_status": document_setup.publication_status,
                    "import_order": document_setup.import_order,
                    "is_target": document.document_id == setup_case.target_document_id,
                    "is_selector_source": (
                        document.document_id
                        == setup_case.candidate_selector.source_document_id
                    ),
                    "lines": lines,
                }
            )
        cases.append(
            {
                "case_number": case_index,
                "case_id": case.case_id,
                "split_id": case.split_id,
                "group_id": case.group_id,
                "world_id": case.world_id,
                "axis": case.axis.model_dump(mode="json"),
                "dimension": TRAIT_DIMENSION[
                    setup_case.candidate_selector.trait_type
                ],
                "target_document_id": setup_case.target_document_id,
                "candidate_selector": setup_case.candidate_selector.model_dump(
                    mode="json"
                ),
                "author_axis": (
                    setup_case.author_axis.model_dump(mode="json")
                    if setup_case.author_axis is not None
                    else None
                ),
                "documents": documents,
                "evidence_catalog": [
                    row.model_dump(mode="json") for row in case.evidence_catalog
                ],
            }
        )

    return {
        "page_schema_version": PAGE_SCHEMA_VERSION,
        "annotation_schema_version": workflow.ANNOTATION_SCHEMA_VERSION,
        "dataset_id": public.dataset_id,
        "public_input_sha256": contract.canonical_sha256(public),
        "author_setup_sha256": contract.canonical_sha256(author_setup),
        "execution_freeze_sha256": contract.canonical_sha256(execution_freeze),
        "reviewer_id": reviewer_id,
        "manual_version": manual_version,
        "cases": cases,
    }


def _safe_json_for_script(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_annotation_page(data: dict[str, Any]) -> str:
    """Render a complete HTML document with no external runtime dependency."""

    return _HTML_TEMPLATE.replace("__ANNOTATION_DATA__", _safe_json_for_script(data))


def generate_annotation_page(
    public_path: str | Path,
    bundle_root: str | Path,
    author_setup_path: str | Path,
    execution_freeze_path: str | Path,
    *,
    reviewer_id: str,
    manual_version: str,
    output_path: str | Path,
) -> Path:
    """Load, verify, and write a new annotation page without overwriting files."""

    public_file = _local_path(public_path, name="public_input")
    setup_file = _local_path(author_setup_path, name="author_setup")
    freeze_file = _local_path(execution_freeze_path, name="execution_freeze")
    output = _local_path(output_path, name="output")
    workflow._require_distinct_local_paths(
        (public_file, bundle_root, setup_file, freeze_file, output)
    )
    public = contract.load_public_input_file(public_file)
    author_setup = closed_bundle.PublicAuthorSetupBundle.model_validate_json(
        workflow._read_limited_local(setup_file, limit=contract.MAX_JSON_BYTES)
    )
    execution_freeze = closed_bundle.ExecutionFreeze.model_validate_json(
        workflow._read_limited_local(freeze_file, limit=contract.MAX_JSON_BYTES)
    )
    data = build_page_data(
        public,
        bundle_root,
        author_setup,
        execution_freeze,
        reviewer_id=reviewer_id,
        manual_version=manual_version,
    )
    if output.suffix.lower() not in {".html", ".htm"}:
        raise AnnotationPageError("output_must_be_html")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as target:
            target.write(render_annotation_page(data))
    except FileExistsError as exc:
        raise AnnotationPageError("output_already_exists") from exc
    except OSError as exc:
        raise AnnotationPageError("output_unwritable") from exc
    return output.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-input", required=True)
    parser.add_argument("--bundle-root", required=True)
    parser.add_argument("--author-setup", required=True)
    parser.add_argument("--execution-freeze", required=True)
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--manual-version", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output = generate_annotation_page(
            args.public_input,
            args.bundle_root,
            args.author_setup,
            args.execution_freeze,
            reviewer_id=args.reviewer_id,
            manual_version=args.manual_version,
            output_path=args.output,
        )
    except (
        AnnotationPageError,
        closed_bundle.BundleBuildError,
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
  <meta name="color-scheme" content="light">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <title>LoreGuard 剧情阅读与判断</title>
  <style>
    :root {
      --page-deep: #f1f2f7;
      --page: #f1f2f7;
      --surface: #ffffff;
      --surface-raised: #eeebfa;
      --field: #ffffff;
      --text: #262337;
      --text-secondary: #504b63;
      --text-muted: #665e77;
      --border: #dad6e5;
      --border-strong: #8f859f;
      --lavender: #6244a3;
      --cyan: #215d98;
      --pink: #8b3c68;
      --warning: #795019;
      --success: #246044;
      --danger: #a32a43;
      --role-b: #6244a3;
      --role-c: #287392;
      --role-g: #28734f;
      --role-x: #986525;
      --role-p: #6d6391;
      --shadow: 0 8px 24px rgba(44, 31, 67, .1);
      --radius-control: 8px;
      --radius-panel: 14px;
      --header-height: 82px;
      font-family: "Noto Sans SC", "Microsoft YaHei UI", "PingFang SC", system-ui, sans-serif;
      color: var(--text);
      background: var(--page-deep);
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; scroll-padding-top: calc(var(--header-height) + 20px); }
    body { margin: 0; min-width: 320px; min-height: 100dvh; background: var(--page); font-size: 17px; line-height: 1.8; }
    button, input, select, textarea { font: inherit; }
    button, select, input[type="checkbox"], input[type="radio"], label[for] { cursor: pointer; }
    button, select, input[type="text"], textarea {
      min-height: 44px;
      color: var(--text);
      background: var(--field);
      border: 1px solid var(--border-strong);
      border-radius: var(--radius-control);
    }
    button { padding: 10px 14px; font-weight: 650; line-height: 1.5; touch-action: manipulation; }
    button:hover:not(:disabled) { border-color: var(--lavender); background: var(--surface-raised); }
    button:active:not(:disabled) { opacity: .82; }
    button:disabled, select:disabled, input:disabled, textarea:disabled {
      cursor: not-allowed;
      opacity: .48;
    }
    :focus-visible { outline: 3px solid var(--cyan); outline-offset: 3px; }
    a { color: var(--cyan); }
    code, .data-id { font-family: "Cascadia Code", "SFMono-Regular", Consolas, monospace; }
    .skip-link {
      position: fixed;
      z-index: 1000;
      top: 8px;
      left: 8px;
      transform: translateY(-160%);
      padding: 10px 14px;
      color: #fff;
      background: var(--cyan);
      border-radius: 8px;
    }
    .skip-link:focus { transform: none; }

    .app-header {
      position: sticky;
      z-index: 40;
      top: 0;
      min-height: var(--header-height);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      padding: 14px clamp(16px, 3vw, 40px);
      background: rgba(255, 255, 255, .98);
      border-bottom: 1px solid var(--border);
    }
    .brand-block { min-width: 0; }
    .brand-block h1 { margin: 0; font-size: clamp(20px, 2.2vw, 28px); line-height: 1.2; }
    .brand-block p { margin: 5px 0 0; color: var(--text-secondary); font-size: 15px; }
    .toolbar { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px; }
    .toolbar .primary, .toolbar .lock-action { color: #fff; background: var(--lavender); border-color: var(--lavender); }
    .toolbar .primary:hover, .toolbar .lock-action:hover { background: #4e3485; }
    .toolbar .quiet-danger { color: var(--danger); }
    .file-control { position: absolute; inline-size: 1px; block-size: 1px; opacity: 0; pointer-events: none; }

    .privacy-strip {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 10px clamp(16px, 3vw, 40px);
      color: var(--text-secondary);
      background: #eeebf6;
      border-bottom: 1px solid var(--border);
      font-size: 14px;
    }
    .privacy-strip strong { color: var(--success); }
    .privacy-strip .identity { overflow-wrap: anywhere; }
    .file-tools { position: relative; }
    summary { min-height: 44px; padding: 8px 0; cursor: pointer; font-weight: 650; }
    .file-tools > summary { padding: 10px 12px; }
    .file-tools > div { position: absolute; z-index: 50; right: 0; width: 280px; max-width: calc(100vw - 32px); padding: 16px; background: var(--surface); border: 1px solid var(--border-strong); box-shadow: var(--shadow); border-radius: 8px; }
    .file-tools button { display: block; width: 100%; margin-bottom: 10px; }
    .file-tools p { margin: 0; font-size: 14px; color: var(--text-secondary); }
    .technical-details { margin-top: 12px; color: var(--text-muted); font-size: 14px; }
    .technical-details > summary { font-weight: 500; }
    .technical-details p { margin: 6px 0; overflow-wrap: anywhere; }
    .quick-start { max-width: 1160px; margin: 24px auto 0; padding: 0 24px; }
    .quick-start p { margin: 8px 0; }
    .reading-steps { display: flex; flex-wrap: wrap; gap: 8px 28px; margin: 0; padding-left: 24px; color: var(--text-secondary); }
    .reading-steps a { font-weight: 650; text-decoration-thickness: 1px; text-underline-offset: 4px; }
    .freeze-binding { flex: 1 0 100%; color: var(--text-muted); font: 12px/1.5 "Cascadia Code", Consolas, monospace; overflow-wrap: anywhere; }

    .error-summary {
      max-width: 1112px;
      margin: 18px auto 0;
      padding: 16px 20px;
      background: #fff0f2;
      border: 1px solid var(--danger);
      border-radius: 12px;
    }
    .error-summary[hidden] { display: none; }
    .error-summary h2 { margin: 0 0 8px; font-size: 18px; }
    .error-summary ul { margin: 0; padding-left: 22px; }
    .error-summary button { min-height: 32px; padding: 2px 4px; color: var(--text); background: none; border: 0; text-align: left; text-decoration: underline; }

    .workspace {
      width: min(100%, 1160px);
      margin: 0 auto;
      padding: 24px 24px 72px;
      display: grid;
      grid-template-columns: 170px minmax(0, 1fr);
      align-items: start;
      gap: 24px;
    }
    .case-rail, .annotation-pane {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-panel);
    }
    .case-rail { position: sticky; top: calc(var(--header-height) + 18px); overflow: hidden; grid-row: 1 / span 2; }
    .rail-heading { padding: 16px; border-bottom: 1px solid var(--border); }
    .rail-heading h2 { margin: 0; font-size: 16px; }
    .rail-heading p { margin: 6px 0 0; color: var(--text-muted); font-size: 13px; }
    .case-list { max-height: calc(100dvh - 250px); overflow: auto; padding: 8px; }
    .case-button {
      width: 100%;
      display: grid;
      grid-template-columns: 24px minmax(0, 1fr);
      align-items: center;
      gap: 8px;
      min-height: 52px;
      margin: 0 0 6px;
      padding: 8px;
      color: var(--text-secondary);
      background: transparent;
      border-color: transparent;
      text-align: left;
    }
    .case-button[aria-current="true"] { color: var(--text); background: var(--surface-raised); border-color: var(--lavender); }
    .case-number { font-variant-numeric: tabular-nums; color: var(--lavender); }
    .case-label { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .case-state { font-size: 12px; color: var(--text-muted); }
    .case-button .case-state { grid-column: 2; }
    .case-mobile { display: none; padding: 12px 16px; }
    .case-mobile label { display: block; margin-bottom: 6px; font-weight: 650; }
    .case-mobile select { width: 100%; padding: 8px 10px; }

    .evidence-pane { min-width: 0; grid-column: 2; }
    .case-intro {
      margin-bottom: 14px;
      padding: 24px 28px;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-panel);
    }
    .case-intro h2 { margin: 0 0 8px; font-size: 24px; }
    .case-intro p { margin: 5px 0; color: var(--text-secondary); overflow-wrap: anywhere; }
    .case-intro .shortcut-note { color: var(--text-secondary); font-size: 14px; }
    .case-question { color: var(--text) !important; font-size: 21px; font-weight: 700; line-height: 1.7; }
    .target-context { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); }
    .target-context h3 { margin: 0 0 10px; font-size: 17px; }
    .context-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 9px 16px; }
    .context-item { min-width: 0; padding: 9px 12px; background: #f5f3fa; border-left: 3px solid var(--lavender); }
    .context-item dt { margin: 0 0 4px; color: var(--text-muted); font-size: 14px; }
    .context-item dd { margin: 0; color: var(--text); line-height: 1.55; white-space: pre-wrap; overflow-wrap: anywhere; }
    .context-item.wide { grid-column: 1 / -1; }
    .document-block {
      margin-bottom: 16px;
      overflow: hidden;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius-panel);
    }
    .document-header { padding: 18px 24px; background: #f2eff9; border-bottom: 1px solid var(--border); }
    .document-header h3 { margin: 0; font-size: 20px; overflow-wrap: anywhere; }
    .document-header p { margin: 5px 0 0; color: var(--text-secondary); font-size: 15px; overflow-wrap: anywhere; }
    .document-block[data-target="true"] { border-color: #9bbeca; }
    .document-block[data-target="true"] .document-header { background: #edf6f8; }
    .document-block[data-selector-source="true"] .document-header { box-shadow: inset 4px 0 var(--lavender); }
    .document-lines { padding: 8px 0; }
    .evidence-row {
      display: grid;
      grid-template-columns: 30px minmax(0, 1fr) 174px;
      align-items: start;
      gap: 12px;
      padding: 12px 20px 12px 12px;
      border-left: 4px solid transparent;
      border-bottom: 1px solid #ece9f1;
    }
    .evidence-row:hover { background: #faf9fd; }
    .evidence-row[data-role="B"] { border-left-color: var(--role-b); }
    .evidence-row[data-role="C"] { border-left-color: var(--role-c); }
    .evidence-row[data-role="G"] { border-left-color: var(--role-g); }
    .evidence-row[data-role="X"] { border-left-color: var(--role-x); }
    .evidence-row[data-role="P"] { border-left-color: var(--role-p); }
    .evidence-row[data-selector="true"] { background: #f5f2fb; }
    .line-number { padding-top: 10px; color: var(--text-muted); text-align: right; font-family: "Cascadia Code", Consolas, monospace; font-size: 12px; }
    .line-copy {
      min-width: 0;
      margin: 0;
      padding: 7px 0;
      color: var(--text);
      font-family: inherit;
      font-size: 18px;
      line-height: 1.85;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
    }
    .role-select { width: 100%; padding: 8px 4px; font-size: 15px; }
    .blank-line { min-height: 18px; border-bottom: 1px solid rgba(179, 154, 255, .05); }

    .annotation-pane { padding: 28px; min-width: 0; grid-column: 2; }
    .annotation-pane h2 { margin: 0 0 6px; font-size: 24px; }
    .annotation-pane > p { margin: 0 0 20px; color: var(--text-secondary); font-size: 16px; }
    fieldset { min-width: 0; margin: 0; padding: 0; border: 0; }
    .form-section { padding: 24px 0; border-top: 1px solid var(--border); }
    .form-section:first-child { border-top: 0; padding-top: 0; }
    .form-section h3 { margin: 0 0 12px; font-size: 20px; }
    .field { margin-bottom: 24px; }
    .field > label, .field-label { display: block; margin-bottom: 6px; font-weight: 650; }
    .field select, .field input[type="text"], .field textarea { width: 100%; padding: 9px 11px; }
    .field textarea { min-height: 94px; resize: vertical; line-height: 1.8; }
    .helper { margin: 6px 0 0; color: var(--text-muted); font-size: 15px; line-height: 1.8; }
    .derived-result { padding: 12px; background: var(--field); border-left: 3px solid var(--cyan); color: var(--text-secondary); }
    .derived-result strong { color: var(--text); }
    .radio-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .radio-card, .check-target {
      display: flex;
      align-items: flex-start;
      gap: 8px;
      min-height: 44px;
      padding: 9px 10px;
      background: var(--field);
      border: 1px solid var(--border-strong);
      border-radius: 8px;
    }
    .radio-card:has(input:checked), .check-target:has(input:checked) { border-color: var(--lavender); background: var(--surface-raised); }
    .radio-card input, .check-target input { margin-top: 4px; accent-color: var(--lavender); }
    .radio-card span { display: block; }
    .radio-card small { display: block; margin-top: 6px; color: var(--text-muted); line-height: 1.65; font-size: 14px; }
    .confidence-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px; }
    .confidence-grid .radio-card { align-items: center; justify-content: center; padding: 8px 4px; }
    .structure-empty { padding: 12px; color: var(--text-muted); background: var(--field); border-radius: 8px; }
    .structure-row { margin-bottom: 12px; padding: 12px; background: var(--field); border: 1px solid var(--border); border-radius: 10px; }
    .structure-row h4 { margin: 0 0 8px; font-size: 17px; overflow-wrap: anywhere; }
    .evidence-excerpt { margin: 0 0 12px; color: var(--text-secondary); font-size: 17px; line-height: 1.8; white-space: pre-wrap; overflow-wrap: anywhere; }
    .support-targets { display: grid; gap: 6px; margin: 8px 0 12px; }
    .structure-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .structure-grid label { display: block; margin-bottom: 5px; color: var(--text-secondary); font-size: 15px; }
    .structure-grid select, .structure-grid input { width: 100%; padding: 8px; }
    .case-errors { margin: 0 0 16px; padding: 12px 14px; background: #fff0f2; border: 1px solid var(--danger); border-radius: 8px; }
    .case-errors[hidden] { display: none; }
    .case-errors strong { color: var(--danger); }
    .case-errors ul { margin: 7px 0 0; padding-left: 20px; }
    [aria-invalid="true"] { border-color: var(--danger) !important; }
    .locked-banner { margin-bottom: 14px; padding: 12px; color: var(--success); background: #edf7f0; border: 1px solid var(--success); border-radius: 8px; }
    .locked-banner[hidden] { display: none; }
    .case-actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 20px; padding-top: 24px; border-top: 1px solid var(--border); }
    .role-help { padding: 16px 20px; margin-bottom: 20px; background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }
    .role-help dl { margin: 12px 0 0; }
    .role-help dt { font-weight: 650; }
    .role-help dd { margin: 0 0 12px; color: var(--text-secondary); font-size: 15px; }
    .structure-section[hidden] { display: none; }
    .declaration-details { padding: 16px; background: #f8f7fb; border: 1px solid var(--border); border-radius: 8px; }
    .declaration-details .check-target { margin-top: 10px; }

    .status-region {
      position: fixed;
      z-index: 100;
      right: 18px;
      bottom: 18px;
      max-width: min(420px, calc(100vw - 36px));
      padding: 12px 16px;
      color: var(--text);
      background: var(--surface-raised);
      border: 1px solid var(--border-strong);
      border-radius: 10px;
      box-shadow: var(--shadow);
    }
    .status-region:empty { display: none; }

    @media (max-width: 1000px) {
      .app-header { position: static; flex-wrap: wrap; }
      .workspace { grid-template-columns: 150px minmax(0, 1fr); gap: 18px; }
      .evidence-row { grid-template-columns: 30px minmax(0, 1fr); }
      .role-select { grid-column: 2; width: min(100%, 230px); }
      .case-rail { top: 18px; }
    }
    @media (max-width: 767px) {
      :root { --header-height: 0px; }
      .app-header { position: static; align-items: flex-start; flex-direction: column; }
      .toolbar { width: 100%; justify-content: flex-start; }
      .toolbar button { flex: 1 1 145px; }
      .toolbar #download-locked { flex-basis: 100%; }
      .file-tools > div { left: 0; right: auto; max-width: calc(100vw - 32px); }
      .privacy-strip { align-items: flex-start; flex-direction: column; }
      .privacy-strip .identity { text-align: left; }
      .workspace { display: block; padding: 16px 12px 48px; }
      .quick-start { padding-inline: 16px; margin-top: 16px; }
      .reading-steps { gap: 4px 24px; }
      .case-rail { position: static; margin-bottom: 14px; }
      .rail-heading, .case-list { display: none; }
      .case-mobile { display: block; }
      .annotation-pane { margin-top: 14px; }
      .evidence-row { grid-template-columns: 38px minmax(0, 1fr); gap: 8px; padding-right: 10px; }
      .role-select { grid-column: 2; width: min(100%, 180px); }
      .line-copy { font-size: 18px; }
      .case-intro, .annotation-pane { padding: 20px 18px; }
      .document-header { padding: 16px 18px; }
      .context-grid { grid-template-columns: 1fr; }
      .context-item.wide { grid-column: auto; }
      .radio-grid { grid-template-columns: 1fr; }
      .status-region { bottom: 10px; right: 10px; }
    }
    @media (max-width: 420px) {
      .structure-grid { grid-template-columns: 1fr; }
      .confidence-grid { grid-template-columns: repeat(3, 1fr); }
    }
    @media (prefers-reduced-motion: reduce) {
      *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
    }
  </style>
</head>
<body>
  <a class="skip-link" href="#main-content">跳到当前案例</a>
  <header class="app-header">
    <div class="brand-block">
      <h1>剧情阅读与判断</h1>
      <p>一次只做一题。<span id="save-state">填写后自动保存进度。</span></p>
    </div>
    <div class="toolbar" aria-label="标注文件操作">
      <button id="save-draft" type="button">保存进度</button>
      <button class="lock-action" id="download-locked" type="button">完成任务并下载结果</button>
      <button id="unlock-editing" type="button" hidden>修改这份结果</button>
      <details class="file-tools">
        <summary>文件与备份</summary>
        <div>
          <button id="import-trigger" type="button">导入我的结果文件</button>
          <input class="file-control" id="import-file" type="file" accept="application/json,.json" aria-label="选择我的结果文件">
          <button class="quiet-danger" id="clear-draft" type="button">清除本地进度</button>
          <p>进度通常会自动保存在当前浏览器。导入的是你之前下载的结果文件；此页面不会自动上传。</p>
        </div>
      </details>
    </div>
  </header>
  <div class="privacy-strip">
    <div><strong>请独立完成</strong>　只根据题目判断；先不要和另一位标注者对答案，也不要让 AI 代做。</div>
    <div class="identity">下载后把文件发回负责人。</div>
    <details class="technical-details"><summary>文件信息（通常不用看）</summary><div id="reviewer-summary"></div><div class="freeze-binding" id="freeze-summary"></div></details>
  </div>
  <section class="quick-start" aria-label="操作顺序">
    <ol class="reading-steps">
      <li><a href="#evidence-pane">看材料</a></li>
      <li><a href="#role-help">标出原文依据</a></li>
      <li><a href="#annotation-pane">作出判断并说明</a></li>
    </ol>
    <p class="helper">先通读这一题，再给有用的原文选用途。不相关的行留空；不用每行都选。遇到不确定的情况，可以如实记录。</p>
  </section>
  <section class="error-summary" id="error-summary" role="alert" tabindex="-1" hidden>
    <h2 id="error-summary-title">还有这些地方需要补充</h2>
    <ul id="error-summary-list"></ul>
  </section>
  <main class="workspace" id="main-content" tabindex="-1">
    <nav class="case-rail" aria-label="案例导航">
      <div class="rail-heading">
        <h2>题目进度</h2>
        <p id="progress-label"></p>
      </div>
      <div class="case-list" id="case-list"></div>
      <div class="case-mobile">
        <label for="case-select">切换题目</label>
        <select id="case-select"></select>
      </div>
    </nav>
    <article class="evidence-pane" id="evidence-pane" aria-label="阅读材料与选择原文依据"></article>
    <aside class="annotation-pane" id="annotation-pane" aria-label="人工标注表单"></aside>
  </main>
  <div class="status-region" id="status-region" aria-live="polite" aria-atomic="true"></div>
  <script id="annotation-data" type="application/json">__ANNOTATION_DATA__</script>
  <script>
  (() => {
    "use strict";

    const DATA = JSON.parse(document.getElementById("annotation-data").textContent);
    const ROLE_OPTIONS = [
      ["", "不选／与判断无关"], ["B", "原设定（B）"], ["C", "新稿表现（C）"],
      ["G", "成长经历（G）"], ["X", "当时的特殊状态（X）"], ["P", "可能的解释（P）"]
    ];
    const DIMENSIONS = [
      ["core_trait", "核心性格"], ["stable_preference", "长期偏好"],
      ["speech_pattern", "说话方式"], ["value_boundary", "价值观／行为边界"],
      ["relationship_attitude", "关系态度"], ["motivation_goal", "长期动机／目标"]
    ];
    const LEVELS = {
      L0: { outcome: "no_issue", surface: "none", label: "未发现冲突／变化有明确解释" },
      L1: { outcome: "indeterminate", surface: "review_clue", label: "信息不够，暂时判断不了" },
      L2: { outcome: "indeterminate", surface: "review_clue", label: "有疑点，还不能定为冲突" },
      L3: { outcome: "conflict", surface: "formal_issue", label: "证据充分，存在冲突" }
    };
    const CASE_KEYS = [
      "case_id", "dimension", "outcome", "surface", "conflict_level",
      "material_coverage", "evidence", "independent_event_groups",
      "explanation_links", "phenomena", "reason_codes", "confidence", "review_note"
    ];
    const BUNDLE_KEYS = [
      "schema_version", "dataset_id", "public_input_sha256", "reviewer_kind",
      "reviewer_id", "manual_version", "blinded_to_peer",
      "blinded_to_system_prediction", "independence_declaration", "cases"
    ];
    const draftKey = [
      "loreguard-ooc-draft-v1", DATA.public_input_sha256, DATA.reviewer_id,
      DATA.manual_version, DATA.author_setup_sha256, DATA.execution_freeze_sha256
    ].join(":");

    const ui = {
      caseList: document.getElementById("case-list"),
      caseSelect: document.getElementById("case-select"),
      evidence: document.getElementById("evidence-pane"),
      annotation: document.getElementById("annotation-pane"),
      progress: document.getElementById("progress-label"),
      status: document.getElementById("status-region"),
      errorSummary: document.getElementById("error-summary"),
      errorList: document.getElementById("error-summary-list"),
      importFile: document.getElementById("import-file"),
      download: document.getElementById("download-locked"),
      unlock: document.getElementById("unlock-editing"),
      saveState: document.getElementById("save-state")
    };

    function blankCase() {
      return {
        conflictLevel: "", materialCoverage: "complete",
        evidenceRoles: {}, eventGroups: {}, explanations: {}, phenomena: "",
        reasonCodes: "", confidence: "", reviewNote: ""
      };
    }

    function blankDeclarations() {
      return {
        humanConfirmed: false,
        peerBlindConfirmed: false,
        systemBlindConfirmed: false,
        independenceConfirmed: false
      };
    }

    const state = {
      activeCase: 0,
      locked: false,
      declarations: blankDeclarations(),
      cases: DATA.cases.map(() => blankCase())
    };

    function element(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function exactKeys(value, keys) {
      if (!value || typeof value !== "object" || Array.isArray(value)) return false;
      const actual = Object.keys(value).sort();
      const expected = [...keys].sort();
      return actual.length === expected.length && actual.every((key, index) => key === expected[index]);
    }

    function deepEqual(left, right) {
      if (Object.is(left, right)) return true;
      if (Array.isArray(left) || Array.isArray(right)) {
        return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
          left.every((value, index) => deepEqual(value, right[index]));
      }
      if (!left || !right || typeof left !== "object" || typeof right !== "object") return false;
      const leftKeys = Object.keys(left).sort();
      const rightKeys = Object.keys(right).sort();
      return leftKeys.length === rightKeys.length &&
        leftKeys.every((key, index) => key === rightKeys[index] && deepEqual(left[key], right[key]));
    }

    function announce(message) {
      ui.status.textContent = message;
      window.clearTimeout(announce.timer);
      announce.timer = window.setTimeout(() => { ui.status.textContent = ""; }, 4500);
    }

    function parseTags(value) {
      const seen = new Set();
      return String(value || "").split(/[\n,，]+/).map(item => item.trim()).filter(item => {
        if (!item || seen.has(item)) return false;
        seen.add(item);
        return true;
      });
    }

    function evidenceMeta(caseIndex) {
      const result = new Map();
      const row = DATA.cases[caseIndex];
      row.documents.forEach(documentRow => {
        documentRow.lines.forEach(line => {
          if (line.evidence_id) {
            result.set(line.evidence_id, {
              evidenceId: line.evidence_id,
              documentId: documentRow.document_id,
              path: documentRow.path,
              lineNumber: line.line_number,
              text: line.text,
              phase: documentRow.phase,
              isTarget: documentRow.is_target,
              isSelector: (
                documentRow.document_id === row.candidate_selector.source_document_id &&
                row.candidate_selector.source_line_start <= line.line_number &&
                line.line_number <= row.candidate_selector.source_line_end
              )
            });
          }
        });
      });
      return result;
    }

    function roleAllowed(caseIndex, evidenceId, role) {
      if (!role || role === "X" || role === "P") return true;
      const item = evidenceMeta(caseIndex).get(evidenceId);
      if (!item) return false;
      if (role === "B") return item.isSelector;
      if (role === "C") return item.isTarget;
      if (role === "G") return item.phase === "baseline";
      return false;
    }

    function selectedByRole(caseIndex, roles) {
      const caseState = state.cases[caseIndex];
      const metadata = evidenceMeta(caseIndex);
      return Object.entries(caseState.evidenceRoles)
        .filter(([evidenceId, role]) => roles.includes(role) && metadata.has(evidenceId))
        .map(([evidenceId, role]) => ({ ...metadata.get(evidenceId), role }));
    }

    function caseIsStarted(caseState) {
      return Boolean(
        caseState.conflictLevel || caseState.confidence ||
        caseState.reviewNote.trim() || Object.values(caseState.evidenceRoles).some(Boolean)
      );
    }

    function caseIsApparentlyComplete(caseState) {
      return Boolean(
        caseState.conflictLevel && caseState.materialCoverage &&
        caseState.confidence && parseTags(caseState.phenomena).length &&
        parseTags(caseState.reasonCodes).length && caseState.reviewNote.trim()
      );
    }

    function updateNavigation() {
      ui.caseList.textContent = "";
      ui.caseSelect.textContent = "";
      let complete = 0;
      DATA.cases.forEach((caseRow, index) => {
        const caseState = state.cases[index];
        const status = caseIsApparentlyComplete(caseState) ? "已填写" : caseIsStarted(caseState) ? "进行中" : "未开始";
        if (status === "已填写") complete += 1;
        const button = element("button", "case-button");
        button.type = "button";
        button.dataset.index = String(index);
        button.setAttribute("aria-current", index === state.activeCase ? "true" : "false");
        button.title = `第 ${index + 1} 题：${status}`;
        button.append(element("span", "case-number", String(index + 1).padStart(2, "0")));
        button.append(element("span", "case-label", `第 ${index + 1} 题`));
        button.append(element("span", "case-state", status));
        button.addEventListener("click", () => switchCase(index));
        ui.caseList.append(button);

        const option = element("option", "", `第 ${index + 1} 题 · ${status}`);
        option.value = String(index);
        option.selected = index === state.activeCase;
        ui.caseSelect.append(option);
      });
      ui.progress.textContent = `${complete} / ${DATA.cases.length} 题已填主要字段（不代表已通过检查）`;
    }

    function switchCase(index, focusHeading = true) {
      if (!Number.isInteger(index) || index < 0 || index >= DATA.cases.length) return;
      state.activeCase = index;
      updateNavigation();
      renderCase();
      scheduleDraft();
      if (focusHeading) document.getElementById("case-heading")?.focus();
    }

    function renderEvidence(caseIndex) {
      const caseRow = DATA.cases[caseIndex];
      const caseState = state.cases[caseIndex];
      ui.evidence.textContent = "";
      const intro = element("header", "case-intro");
      const heading = element("h2", "", `第 ${caseIndex + 1} 题，共 ${DATA.cases.length} 题`);
      heading.id = "case-heading";
      heading.tabIndex = -1;
      intro.append(heading);
      const dimensionName = DIMENSIONS.find(([value]) => value === caseRow.dimension)?.[1] || caseRow.dimension;
      const traitName = caseRow.author_axis?.display_name || dimensionName;
      intro.append(element("p", "case-question", `只判断「${caseRow.candidate_selector.character_key}」的「${traitName}」，新稿有没有违背原设定？`));
      const context = element("section", "target-context");
      context.append(element("h3", "", "这题要关注什么"));
      const contextGrid = element("dl", "context-grid");
      const addContext = (label, value, wide = false) => {
        const item = element("div", `context-item${wide ? " wide" : ""}`);
        item.append(element("dt", "", label), element("dd", "", value));
        contextGrid.append(item);
      };
      const selectorDocument = caseRow.documents.find(
        row => row.document_id === caseRow.candidate_selector.source_document_id
      );
      addContext("指定人物", caseRow.candidate_selector.character_key);
      addContext("只看这一项特点", traitName);
      addContext(
        "对应原设定在哪里",
        `下方材料 ${caseRow.documents.indexOf(selectorDocument) + 1}，第 ${caseRow.candidate_selector.source_line_start}–${caseRow.candidate_selector.source_line_end} 行（浅紫底）`, true
      );
      addContext(
        "本题原设定（原文）",
        (selectorDocument?.lines || []).filter(line =>
          caseRow.candidate_selector.source_line_start <= line.line_number &&
          line.line_number <= caseRow.candidate_selector.source_line_end
        ).map(line => line.text).join("\n"), true
      );
      if (caseRow.author_axis) {
        addContext("这个特点指什么", caseRow.author_axis.definition, true);
        addContext("在哪些情况下适用", caseRow.author_axis.applicability_scope, true);
      } else {
        addContext("判断范围", "按上面指定位置的原设定判断，不扩大到人物的其他特点。", true);
      }
      context.append(contextGrid);
      intro.append(context);
      const technical = element("details", "technical-details");
      technical.append(element("summary", "", "题目信息（通常不用看）"));
      technical.append(element("p", "data-id", `案例 ID：${caseRow.case_id}`));
      technical.append(element("p", "data-id", `轴快照：${caseRow.axis.axis_id} · v${caseRow.axis.version} · ${caseRow.axis.sha256}`));
      technical.append(element("p", "data-id", `目标属性：${caseRow.candidate_selector.trait_type} → ${caseRow.dimension}；${caseRow.candidate_selector.origin} / ${caseRow.candidate_selector.polarity} / ${caseRow.candidate_selector.stability}`));
      if (caseRow.author_axis) {
        technical.append(element("p", "", `比较时采用的说法（不一定是人物原设定）：${caseRow.author_axis.positive_proposition}`));
        technical.append(element("p", "", "这只是比较坐标的一端，不是本题答案；人物实际原设定以上面的高亮原文为准。"));
        technical.append(element("p", "data-id", `轴向对齐：${caseRow.author_axis.axis_alignment}`));
      }
      technical.append(element("p", "shortcut-note", "可用 Alt + ↑/↓ 切换题目、Ctrl + S 保存；下拉框保留 B、C、G、X、P 快捷键。"));
      intro.append(technical);
      intro.append(element("h3", "", "1 看材料"));
      intro.append(element("p", "", "先看原设定和已有剧情，再看待审新稿。判断只针对本题，不要带入题外的故事。"));
      ui.evidence.append(intro);

      const roleHelp = element("details", "role-help");
      roleHelp.id = "role-help";
      roleHelp.append(element("summary", "", "2 标出原文依据：每行旁边应该怎么选？"));
      roleHelp.append(element("p", "", "只选支撑你判断的行。不相关的行留空；每行只能选一种用途。这些选择不是结论。"));
      const roleDefinitions = element("dl");
      [
        ["原设定（B）", "指定人物原先稳定的特点。只能在本题标出的浅紫色原设定行选择。"],
        ["新稿表现（C）", "待审新稿里，需要拿来与原设定比较的行为或说法。不是选了就代表有错。"],
        ["成长经历（G）", "在新稿表现之前已经发生、能够解释变化的经历。只可选已有材料中的行，不能把未来计划当成已发生。"],
        ["当时的特殊状态（X）", "在这次表现发生时确实生效的特殊情况，例如伪装或被控制；必须说明它解释哪次表现。"],
        ["可能的解释（P）", "材料只暗示了一种解释，还不足以当成明确的成长或特殊状态。不要自行补写未提供的经历。"]
      ].forEach(([label, help]) => roleDefinitions.append(element("dt", "", label), element("dd", "", help)));
      roleHelp.append(roleDefinitions);
      ui.evidence.append(roleHelp);

      caseRow.documents.forEach((documentRow, documentIndex) => {
        const section = element("section", "document-block");
        section.dataset.target = String(documentRow.is_target);
        section.dataset.selectorSource = String(documentRow.is_selector_source);
        section.setAttribute("aria-labelledby", `doc-${caseIndex}-${documentIndex}`);
        const header = element("header", "document-header");
        const documentType = documentRow.is_target ? "待审新稿" : documentRow.document_role === "character_profile" ? "角色设定" : documentRow.document_role === "canon" ? "故事设定" : documentRow.publication_status === "published" ? "已发布历史" : "已有剧情";
        const originalTitle = documentRow.lines.find(line => /^\s{0,3}#{1,6}\s+\S/.test(line.text));
        const title = element("h3", "", `材料 ${documentIndex + 1}：${documentType}${originalTitle ? ` · ${originalTitle.text.replace(/^\s{0,3}#{1,6}\s+/, "")}` : ""}`);
        title.id = `doc-${caseIndex}-${documentIndex}`;
        header.append(title);
        header.append(element("p", "", documentRow.is_target ? "这是这次要检查的新内容，可以拿它与已有材料对照。" : "这是新稿之前的参考依据；不要把它当成本次新稿表现。"));
        const documentDetails = element("details", "technical-details");
        documentDetails.append(element("summary", "", "文档信息（通常不用看）"));
        documentDetails.append(element("p", "data-id", `${documentRow.path} · 快照 v${documentRow.version} · ${documentRow.sha256}`));
        documentDetails.append(element("p", "data-id", `${documentRow.phase} · ${documentRow.document_role} · ${documentRow.story_scope} · ${documentRow.resolution_state}/${documentRow.publication_status} · 导入序 ${documentRow.import_order}`));
        header.append(documentDetails);
        section.append(header);
        const lines = element("div", "document-lines");
        documentRow.lines.forEach(line => {
          if (!line.evidence_id) {
            lines.append(element("div", "blank-line"));
            return;
          }
          const row = element("div", "evidence-row");
          const role = caseState.evidenceRoles[line.evidence_id] || "";
          row.dataset.role = role;
          row.dataset.selector = String(
            documentRow.document_id === caseRow.candidate_selector.source_document_id &&
            caseRow.candidate_selector.source_line_start <= line.line_number &&
            line.line_number <= caseRow.candidate_selector.source_line_end
          );
          row.append(element("span", "line-number", String(line.line_number)));
          row.append(element("p", "line-copy", line.text));
          const select = element("select", "role-select");
          select.id = `evidence-${caseIndex}-${line.evidence_id}`;
          select.disabled = state.locked;
          select.setAttribute("aria-label", `材料 ${documentIndex + 1} 第 ${line.line_number} 行：选择这段原文的用途`);
          ROLE_OPTIONS.forEach(([value, label]) => {
            const option = element("option", "", label);
            option.value = value;
            option.selected = value === role;
            option.disabled = !roleAllowed(caseIndex, line.evidence_id, value);
            select.append(option);
          });
          select.addEventListener("keydown", event => {
            const roleKey = event.key.toUpperCase();
            if (!["B", "C", "G", "X", "P"].includes(roleKey) || event.ctrlKey || event.altKey || event.metaKey) return;
            if (!roleAllowed(caseIndex, line.evidence_id, roleKey)) {
              announce("这一行不能选该用途：原设定只能选浅紫色指定行，新稿表现只能选待审新稿，成长经历只能选已有材料。");
              return;
            }
            event.preventDefault();
            select.value = roleKey;
            select.dispatchEvent(new Event("change", { bubbles: true }));
          });
          select.addEventListener("change", () => {
            if (!roleAllowed(caseIndex, line.evidence_id, select.value)) {
              select.value = "";
              announce("这一行不能用作你选的用途，已取消该选择。可展开“每行旁边应该怎么选”查看范围。 ");
            }
            caseState.evidenceRoles[line.evidence_id] = select.value;
            row.dataset.role = select.value;
            renderStructures(caseIndex);
            updateNavigation();
            scheduleDraft();
          });
          row.append(select);
          lines.append(row);
        });
        section.append(lines);
        ui.evidence.append(section);
      });
    }

    function formField(labelText, control, helperText, id) {
      const wrapper = element("div", "field");
      const label = element("label", "", labelText);
      label.htmlFor = id;
      control.id = id;
      wrapper.append(label, control);
      if (helperText) {
        const helper = element("p", "helper", helperText);
        helper.id = `${id}-help`;
        control.setAttribute("aria-describedby", helper.id);
        wrapper.append(helper);
      }
      return wrapper;
    }

    function makeSelect(options, value) {
      const select = element("select");
      options.forEach(([optionValue, label]) => {
        const option = element("option", "", label);
        option.value = optionValue;
        option.selected = optionValue === value;
        select.append(option);
      });
      return select;
    }

    function radioGroup(name, options, value, className = "radio-grid") {
      const group = element("div", className);
      options.forEach(([optionValue, label, help]) => {
        const wrapper = element("label", "radio-card");
        const input = document.createElement("input");
        input.type = "radio";
        input.name = name;
        input.value = optionValue;
        input.checked = optionValue === value;
        const copy = element("span", "", label);
        if (help) copy.append(element("small", "", help));
        wrapper.append(input, copy);
        group.append(wrapper);
      });
      return group;
    }

    function updateDerived(caseIndex) {
      const level = state.cases[caseIndex].conflictLevel;
      const node = document.getElementById("derived-result");
      if (!node) return;
      node.textContent = "";
      if (!LEVELS[level]) {
        node.append("这里没有默认结论。请根据材料，自己选择上面四项中的一项。");
      } else {
        node.append(element("strong", "", `你选的是：${LEVELS[level].label}`));
        node.append(element("p", "helper", level === "L3" ? "这项会作为正式冲突。提交前仍需满足下方的证据与独立事件要求。" : level === "L0" ? "这项不会作为问题或待复核线索。" : "这项只作为待复核线索，不算正式冲突。"));
        const technical = element("details", "technical-details");
        technical.append(element("summary", "", "对应记录值（不用填写）"));
        technical.append(element("p", "data-id", `${level}；outcome=${LEVELS[level].outcome}；surface=${LEVELS[level].surface}`));
        node.append(technical);
      }
    }

    function renderStructures(caseIndex) {
      const currentHost = document.getElementById("current-event-groups");
      const supportHost = document.getElementById("explanation-links");
      if (!currentHost || !supportHost || caseIndex !== state.activeCase) return;
      const caseState = state.cases[caseIndex];
      const currents = selectedByRole(caseIndex, ["C"]);
      const supports = selectedByRole(caseIndex, ["G", "X", "P"]);
      document.getElementById("current-event-section").hidden = !currents.length;
      document.getElementById("support-section").hidden = !supports.length;
      currentHost.textContent = "";
      supportHost.textContent = "";

      if (!currents.length) {
        currentHost.append(element("p", "structure-empty", "选出新稿表现后，这里会请你区分它们属于哪一次事情。"));
      } else {
        currents.forEach((row, index) => {
          const wrapper = element("div", "structure-row");
          wrapper.append(element("h4", "", `新稿表现 ${index + 1}（第 ${row.lineNumber} 行）`));
          wrapper.append(element("p", "evidence-excerpt", row.text));
          const input = document.createElement("input");
          input.type = "text";
          input.maxLength = 200;
          input.value = caseState.eventGroups[row.evidenceId] || "";
          const id = `event-group-${caseIndex}-${index}`;
          wrapper.append(formField("这是哪一次事情？", input, "给事情起个简短名字，如“码头争执”。同一件事的几段描写填同名；真正不同的事情填不同名。名字由你决定，但同名或不同名会用于区分事件。", id));
          input.addEventListener("input", () => {
            caseState.eventGroups[row.evidenceId] = input.value;
            scheduleDraft();
          });
          currentHost.append(wrapper);
        });
      }

      if (!supports.length) {
        supportHost.append(element("p", "structure-empty", "选出成长经历、特殊状态或可能的解释后，这里会请你指明它解释哪次表现。"));
      } else {
        supports.forEach((support, supportIndex) => {
          const linkState = caseState.explanations[support.evidenceId] || {
            applicableCurrentIds: [], causalRelation: support.role === "P" ? "ambiguous" : "bounded",
            temporalRelation: support.role === "G" ? "before" : support.role === "X" ? "active_during" : "unknown"
          };
          caseState.explanations[support.evidenceId] = linkState;
          if ((support.role === "G" || support.role === "X") && linkState.causalRelation === "ambiguous") {
            linkState.causalRelation = "bounded";
          }
          if (support.role === "G") linkState.temporalRelation = "before";
          if (support.role === "X") linkState.temporalRelation = "active_during";
          linkState.applicableCurrentIds = linkState.applicableCurrentIds.filter(id => currents.some(row => row.evidenceId === id));

          const wrapper = element("div", "structure-row");
          wrapper.append(element("h4", "", `${ROLE_OPTIONS.find(([value]) => value === support.role)[1]}（第 ${support.lineNumber} 行）`));
          wrapper.append(element("p", "evidence-excerpt", support.text));
          wrapper.append(element("p", "helper", support.role === "P" ? "勾选这段话可能解释的新稿表现；如果还无法对应，可以暂时不勾选。" : "请勾选这段话能够解释的至少一条新稿表现。不要把只能解释一件事的理由套到所有事情上。"));
          const targets = element("div", "support-targets");
          if (!currents.length) {
            targets.append(element("p", "structure-empty", "请先回到原文，把要比较的行为选为“新稿表现”，再在这里勾选。"));
          } else {
            currents.forEach((current, currentIndex) => {
              const label = element("label", "check-target");
              const checkbox = document.createElement("input");
              checkbox.type = "checkbox";
              checkbox.checked = linkState.applicableCurrentIds.includes(current.evidenceId);
              checkbox.addEventListener("change", () => {
                const selected = new Set(linkState.applicableCurrentIds);
                checkbox.checked ? selected.add(current.evidenceId) : selected.delete(current.evidenceId);
                linkState.applicableCurrentIds = [...selected];
                scheduleDraft();
              });
              label.append(checkbox, element("span", "", `新稿表现 ${currentIndex + 1}（第 ${current.lineNumber} 行）：${current.text}`));
              targets.append(label);
            });
          }
          wrapper.append(targets);

          const grid = element("div", "structure-grid");
          const causalWrap = element("div");
          const causalLabel = element("label", "", "这段解释说得有多明确？");
          const causalOptions = support.role === "P"
            ? [["explicit", "材料明确说明原因"], ["bounded", "有明确依据，但只解释特定情况"], ["ambiguous", "仅有暗示，尚不确定"]]
            : [["explicit", "材料明确说明原因"], ["bounded", "有明确依据，但只解释特定情况"]];
          const causal = makeSelect(causalOptions, linkState.causalRelation);
          causal.id = `causal-${caseIndex}-${supportIndex}`;
          causalLabel.htmlFor = causal.id;
          causal.addEventListener("change", () => { linkState.causalRelation = causal.value; scheduleDraft(); });
          causalWrap.append(causalLabel, causal);
          grid.append(causalWrap);

          const temporalWrap = element("div");
          const temporalLabel = element("label", "", "解释发生或生效在什么时候？");
          const temporalOptions = support.role === "G"
            ? [["before", "在所选新稿表现之前已发生"]]
            : support.role === "X"
              ? [["active_during", "所选新稿表现发生时有效"]]
              : [["before", "在所选表现之前"], ["active_during", "所选表现发生时有效"], ["after", "在所选表现之后"], ["unknown", "时间不明"]];
          const temporal = makeSelect(temporalOptions, linkState.temporalRelation);
          temporal.id = `temporal-${caseIndex}-${supportIndex}`;
          temporalLabel.htmlFor = temporal.id;
          temporal.disabled = support.role === "G" || support.role === "X";
          temporal.addEventListener("change", () => { linkState.temporalRelation = temporal.value; scheduleDraft(); });
          temporalWrap.append(temporalLabel, temporal);
          grid.append(temporalWrap);
          wrapper.append(grid);
          supportHost.append(wrapper);
        });
      }
    }

    function renderAnnotation(caseIndex) {
      const caseState = state.cases[caseIndex];
      ui.annotation.textContent = "";
      ui.annotation.append(element("h2", "", "3 作出判断并说明"));
      ui.annotation.append(element("p", "", "看完材料、选好依据后再填写。没有唯一措辞要求；判断不确定，也可以如实说明。"));
      const locked = element("div", "locked-banner", "结果文件已下载，本页暂时不能改动。点击“修改这份结果”可继续编辑；已下载或已发出的文件不会改变。");
      locked.hidden = !state.locked;
      ui.annotation.append(locked);
      const errors = element("div", "case-errors");
      errors.id = "case-errors";
      errors.setAttribute("role", "alert");
      errors.hidden = true;
      ui.annotation.append(errors);
      const fields = document.createElement("fieldset");
      fields.id = "editor-fields";
      fields.disabled = state.locked;

      const judgement = element("section", "form-section");
      const judgementTitle = element("h3", "", "你认为属于哪种情况？");
      judgement.append(judgementTitle);
      const dimensionField = element("div", "field");
      dimensionField.append(element("div", "field-label", "这题只判断"));
      const dimensionValue = element(
        "div", "derived-result",
        `${DATA.cases[caseIndex].candidate_selector.character_key}的${DATA.cases[caseIndex].author_axis?.display_name || DIMENSIONS.find(([value]) => value === DATA.cases[caseIndex].dimension)?.[1] || DATA.cases[caseIndex].dimension}`
      );
      dimensionValue.id = `dimension-${caseIndex}`;
      dimensionValue.tabIndex = -1;
      dimensionField.append(
        dimensionValue,
        element("p", "helper", "判断范围已经指定，不需要另选人物特点。")
      );
      judgement.append(dimensionField);

      const levelLabel = element("div", "field-label", "请选择一个结论（必填）");
      levelLabel.id = `level-label-${caseIndex}`;
      const levels = radioGroup(`level-${caseIndex}`, [
        ["L0", LEVELS.L0.label, "没有违背这项特点，或变化已有能覆盖对应表现的明确解释。（L0）"],
        ["L1", LEVELS.L1.label, "例如还弄不清是谁做的、针对谁、什么情境，或这件事是否真的发生。（L1）"],
        ["L2", LEVELS.L2.label, "已有局部反向表现，但只有一次事情、材料不足，或解释尚不能确定。（L2）"],
        ["L3", LEVELS.L3.label, "原设定明确；至少两次独立事情中的反向表现，没有适用的成长、特殊状态或可能解释。（L3）"]
      ], caseState.conflictLevel);
      levels.id = `level-${caseIndex}`;
      levels.setAttribute("role", "radiogroup");
      levels.setAttribute("aria-labelledby", levelLabel.id);
      levels.addEventListener("change", event => {
        if (event.target instanceof HTMLInputElement) {
          caseState.conflictLevel = event.target.value;
          updateDerived(caseIndex);
          updateNavigation();
          scheduleDraft();
        }
      });
      const levelField = element("div", "field");
      levelField.append(levelLabel, levels);
      judgement.append(levelField);
      const derived = element("div", "derived-result");
      derived.id = "derived-result";
      judgement.append(derived);

      const coverage = makeSelect([
        ["complete", "材料足够覆盖本题需要判断的内容"], ["partial", "有部分相关材料缺失"], ["missing", "缺了关键材料，无法充分判断"]
      ], caseState.materialCoverage);
      coverage.addEventListener("change", () => { caseState.materialCoverage = coverage.value; scheduleDraft(); });
      judgement.append(formField("这题的材料够不够？（必填，请核对）", coverage, "有部分或关键材料缺失时，只能选“暂时判断不了”或“有疑点”，不能直接定为冲突或无问题。", `coverage-${caseIndex}`));
      fields.append(judgement);

      const structures = element("section", "form-section");
      structures.append(element("h3", "", "把所选原文对应清楚"));
      structures.append(element("p", "helper", "下面只显示与你已经选出的原文有关的项目；不用在没有材料时编造经历或解释。"));
      const currentSection = element("section", "structure-section");
      currentSection.id = "current-event-section";
      currentSection.append(element("h3", "", "这些表现是同一次事情，还是不同事情？"));
      currentSection.append(element("p", "helper", "认定正式冲突时必须填写，并至少有两次独立事情；其他结论可按需要补充。"));
      const currentHost = element("div");
      currentHost.id = "current-event-groups";
      currentSection.append(currentHost);
      structures.append(currentSection);
      const supportSection = element("section", "structure-section");
      supportSection.id = "support-section";
      supportSection.append(element("h3", "", "这段解释说明了哪次表现？"));
      const supportHost = element("div");
      supportHost.id = "explanation-links";
      supportSection.append(supportHost);
      structures.append(supportSection);
      fields.append(structures);

      const notes = element("section", "form-section");
      notes.append(element("h3", "", "记录你自己的判断依据"));
      notes.append(element("p", "helper", "前两项写短语，方便负责人以后分类；最后一项用完整的话说明理由。都可以用普通中文，不需要写英文编码。"));
      const phenomena = document.createElement("textarea");
      phenomena.value = caseState.phenomena;
      phenomena.maxLength = 6500;
      phenomena.addEventListener("input", () => { caseState.phenomena = phenomena.value; updateNavigation(); scheduleDraft(); });
      notes.append(formField("你观察到了什么？（短标签，必填）", phenomena, "概括你在材料里看到的现象，至少写一项。多项可换行或用逗号分开；每项不超过 100 字，最多 64 项。", `phenomena-${caseIndex}`));
      const reasons = document.createElement("textarea");
      reasons.value = caseState.reasonCodes;
      reasons.maxLength = 6500;
      reasons.addEventListener("input", () => { caseState.reasonCodes = reasons.value; updateNavigation(); scheduleDraft(); });
      notes.append(formField("你据此判断的关键理由（短标签，必填）", reasons, "概括你为什么选这个结论，至少写一项。这里不是让你猜项目内部的代码；多项可换行或用逗号分开。", `reasons-${caseIndex}`));

      const confidenceLabel = element("div", "field-label", "你对自己的判断有多确定？（必填）");
      confidenceLabel.id = `confidence-label-${caseIndex}`;
      const confidence = radioGroup(`confidence-${caseIndex}`, [
        ["1", "1", ""], ["2", "2", ""], ["3", "3", ""], ["4", "4", ""], ["5", "5", ""]
      ], String(caseState.confidence || ""), "confidence-grid");
      confidence.id = `confidence-${caseIndex}`;
      confidence.setAttribute("role", "radiogroup");
      confidence.setAttribute("aria-labelledby", confidenceLabel.id);
      confidence.addEventListener("change", event => {
        if (event.target instanceof HTMLInputElement) {
          caseState.confidence = event.target.value;
          updateNavigation();
          scheduleDraft();
        }
      });
      const confidenceField = element("div", "field");
      confidenceField.append(confidenceLabel, confidence, element("p", "helper", "1 很不确定；2 不太确定；3 一般；4 比较确定；5 很确定。这不是冲突严重程度。"));
      notes.append(confidenceField);
      const reviewNote = document.createElement("textarea");
      reviewNote.value = caseState.reviewNote;
      reviewNote.maxLength = 8000;
      reviewNote.addEventListener("input", () => { caseState.reviewNote = reviewNote.value; updateNavigation(); scheduleDraft(); });
      notes.append(formField("用自己的话说明判断（必填）", reviewNote, "指出原设定、新稿哪几行支持你的结论。若有成长或特殊状态，说清它解释哪次表现；若不确定，说清缺什么。不必把所有名词都写一遍。最多 8000 字。", `review-note-${caseIndex}`));
      fields.append(notes);

      const declarations = element("details", "form-section declaration-details");
      declarations.id = "declarations";
      declarations.tabIndex = -1;
      declarations.append(element("summary", "", "全部题目做完后：确认独立完成（提交时必填）"));
      declarations.append(element("p", "helper", "下载最终结果之前，请你亲自勾选下列四项。现在没有做完，可以先不勾。"));
      [
        ["humanConfirmed", "我是本文件所列标注者本人，且由真人完成判断。"],
        ["peerBlindConfirmed", "标注期间我没有查看另一位标注者的答案。"],
        ["systemBlindConfirmed", "标注期间我没有查看系统／模型预测。"],
        ["independenceConfirmed", "我独立完成了全部题目，并确认这份结果作为本次提交。"]
      ].forEach(([key, copy]) => {
        const label = element("label", "check-target");
        const checkbox = document.createElement("input");
        checkbox.type = "checkbox";
        checkbox.checked = state.declarations[key];
        checkbox.addEventListener("change", () => {
          state.declarations[key] = checkbox.checked;
          scheduleDraft();
        });
        label.append(checkbox, element("span", "", copy));
        declarations.append(label);
      });
      fields.append(declarations);
      ui.annotation.append(fields);
      const actions = element("nav", "case-actions");
      actions.setAttribute("aria-label", "本题操作");
      const previous = element("button", "", "上一题");
      previous.type = "button";
      previous.id = "previous-case";
      previous.disabled = caseIndex === 0;
      previous.addEventListener("click", () => switchCase(caseIndex - 1));
      const check = element("button", "", "检查本题填写情况");
      check.type = "button";
      check.id = "check-current-case";
      check.addEventListener("click", () => {
        const missing = validateCase(caseIndex);
        showErrorSummary(missing);
        announce(missing.length ? "请按提示补充本题；不会替你判断答案对错。" : "本题必填内容和格式检查通过，不代表你的判断就是标准答案。");
      });
      const next = element("button", "", "下一题");
      next.type = "button";
      next.id = "next-case";
      next.disabled = caseIndex === DATA.cases.length - 1;
      next.addEventListener("click", () => switchCase(caseIndex + 1));
      actions.append(previous, check, next);
      ui.annotation.append(actions);
      updateDerived(caseIndex);
      renderStructures(caseIndex);
    }

    function renderCase() {
      renderEvidence(state.activeCase);
      renderAnnotation(state.activeCase);
      ui.download.disabled = false;
      ui.unlock.hidden = !state.locked;
      document.getElementById("save-draft").disabled = state.locked;
      document.getElementById("import-trigger").disabled = state.locked;
      document.getElementById("clear-draft").disabled = state.locked;
      if (state.locked) ui.saveState.textContent = "结果已生成，请把下载的文件发回负责人。";
    }

    function buildCase(caseIndex) {
      const caseRow = DATA.cases[caseIndex];
      const caseState = state.cases[caseIndex];
      const metadata = evidenceMeta(caseIndex);
      const evidence = Object.entries(caseState.evidenceRoles)
        .filter(([evidenceId, role]) => metadata.has(evidenceId) && ["B", "C", "G", "X", "P"].includes(role))
        .map(([evidenceId, role]) => ({ evidence_id: evidenceId, role }));
      const currentIds = new Set(evidence.filter(row => row.role === "C").map(row => row.evidence_id));
      const grouped = new Map();
      currentIds.forEach(evidenceId => {
        const groupId = String(caseState.eventGroups[evidenceId] || "").trim();
        if (groupId) {
          const values = grouped.get(groupId) || [];
          values.push(evidenceId);
          grouped.set(groupId, values);
        }
      });
      const independent_event_groups = [...grouped.entries()].map(([event_group_id, evidence_ids]) => ({
        event_group_id, evidence_ids
      }));
      const explanation_links = [];
      evidence.filter(row => ["G", "X", "P"].includes(row.role)).forEach(row => {
        const link = caseState.explanations[row.evidence_id] || {};
        const targets = Array.isArray(link.applicableCurrentIds)
          ? [...new Set(link.applicableCurrentIds.filter(value => currentIds.has(value)))] : [];
        if ((row.role === "G" || row.role === "X") || targets.length) {
          const allowedCausal = row.role === "P"
            ? ["explicit", "bounded", "ambiguous"] : ["explicit", "bounded"];
          explanation_links.push({
            support_evidence_id: row.evidence_id,
            applicable_current_ids: targets,
            causal_relation: allowedCausal.includes(link.causalRelation)
              ? link.causalRelation : row.role === "P" ? "ambiguous" : "bounded",
            temporal_relation: row.role === "G" ? "before" : row.role === "X"
              ? "active_during" : ["before", "active_during", "after", "unknown"].includes(link.temporalRelation)
                ? link.temporalRelation : "unknown"
          });
        }
      });
      const level = LEVELS[caseState.conflictLevel] || { outcome: "", surface: "" };
      return {
        case_id: caseRow.case_id,
        dimension: caseRow.dimension,
        outcome: level.outcome,
        surface: level.surface,
        conflict_level: caseState.conflictLevel,
        material_coverage: caseState.materialCoverage,
        evidence,
        independent_event_groups,
        explanation_links,
        phenomena: parseTags(caseState.phenomena),
        reason_codes: parseTags(caseState.reasonCodes),
        confidence: Number(caseState.confidence),
        review_note: caseState.reviewNote.trim()
      };
    }

    function buildBundle() {
      return {
        schema_version: DATA.annotation_schema_version,
        dataset_id: DATA.dataset_id,
        public_input_sha256: DATA.public_input_sha256,
        reviewer_kind: state.declarations.humanConfirmed ? "human" : null,
        reviewer_id: DATA.reviewer_id,
        manual_version: DATA.manual_version,
        blinded_to_peer: state.declarations.peerBlindConfirmed,
        blinded_to_system_prediction: state.declarations.systemBlindConfirmed,
        independence_declaration: state.declarations.independenceConfirmed,
        cases: DATA.cases.map((_, index) => buildCase(index))
      };
    }

    function validateCase(caseIndex) {
      const row = buildCase(caseIndex);
      const errors = [];
      const add = (field, message) => errors.push({ caseIndex, field, message });
      if (row.dimension !== DATA.cases[caseIndex].dimension) add(`dimension-${caseIndex}`, "判断的人物特点与本题指定范围不一致。 ");
      if (!LEVELS[row.conflict_level]) add(`level-${caseIndex}`, "请在四种结论中选择一项。 ");
      if (!["complete", "partial", "missing"].includes(row.material_coverage)) add(`coverage-${caseIndex}`, "请确认这题的材料够不够。 ");
      if (row.material_coverage !== "complete" && !["L1", "L2"].includes(row.conflict_level)) {
        add(`coverage-${caseIndex}`, "材料有缺失时，请选“暂时判断不了”或“有疑点”，不要直接认定冲突或无问题。 ");
      }
      const roles = role => row.evidence.filter(item => item.role === role).map(item => item.evidence_id);
      const b = roles("B"), c = roles("C"), g = roles("G"), x = roles("X"), p = roles("P");
      row.evidence.forEach(item => {
        if (!roleAllowed(caseIndex, item.evidence_id, item.role)) {
          add("evidence-pane", "原文用途不符合本题范围：原设定只能选浅紫色指定行，新稿表现只能选待审新稿，成长经历只能选已有材料。 ");
        }
      });
      if (row.conflict_level === "L3") {
        if (!b.length || c.length < 2) add("evidence-pane", "认定正式冲突，至少要选出 1 行原设定和 2 行新稿表现，并确认这些表现与原设定相反。 ");
        if (g.length || x.length || p.length) add("evidence-pane", "你同时选了成长、特殊状态或可能解释，不能直接认定正式冲突。请复核解释是否适用，不要为了通过检查删除有效依据。 ");
        const byEvidence = new Map();
        row.independent_event_groups.forEach(group => group.evidence_ids.forEach(id => byEvidence.set(id, group.event_group_id)));
        const groups = new Set(c.map(id => byEvidence.get(id)));
        if (groups.has(undefined) || groups.size < 2) add("current-event-groups", "认定正式冲突时，每条新稿表现都要注明属于哪件事，并且确实至少有两次独立事情。同一件事的复述不算两次。 ");
      }
      if (["L1", "L2"].includes(row.conflict_level) && row.evidence.length === 0) add("evidence-pane", "选择“暂时判断不了”或“有疑点”时，也请至少选一行支撑判断的原文。 ");
      const linked = new Set(row.explanation_links.map(link => link.support_evidence_id));
      [...g, ...x].forEach(id => {
        const link = row.explanation_links.find(item => item.support_evidence_id === id);
        if (!linked.has(id) || !link || !link.applicable_current_ids.length) add("explanation-links", "每段成长经历或特殊状态，都要勾选它能解释的至少一条新稿表现。 ");
      });
      row.explanation_links.forEach(link => {
        if (!link.applicable_current_ids.length) add("explanation-links", "建立解释对应关系时，至少勾选一条新稿表现。 ");
        const supportRole = row.evidence.find(item => item.evidence_id === link.support_evidence_id)?.role;
        if ((supportRole === "G" || supportRole === "X") && link.causal_relation === "ambiguous") {
          add("explanation-links", "如果材料只有含糊暗示，请把该行改为“可能的解释”；成长和特殊状态需要明确依据。 ");
        }
      });
      if (!row.phenomena.length || row.phenomena.length > 64 || row.phenomena.some(value => value.length > 100)) add(`phenomena-${caseIndex}`, "请用短标签记录观察到的现象：至少 1 项，每项不超过 100 字，最多 64 项。 ");
      if (!row.reason_codes.length || row.reason_codes.length > 64 || row.reason_codes.some(value => value.length > 100)) add(`reasons-${caseIndex}`, "请用短标签记录判断的关键理由：至少 1 项，每项不超过 100 字，最多 64 项。 ");
      if (!Number.isInteger(row.confidence) || row.confidence < 1 || row.confidence > 5) add(`confidence-${caseIndex}`, "请在 1–5 中选择你对判断有多确定。 ");
      if (!row.review_note || row.review_note.length > 8000) add(`review-note-${caseIndex}`, "请用自己的话说明判断，不能为空，最多 8000 字。 ");
      return errors;
    }

    function showCaseErrors(errors) {
      const host = document.getElementById("case-errors");
      if (!host) return;
      host.textContent = "";
      const local = errors.filter(error => error.caseIndex === state.activeCase);
      host.hidden = !local.length;
      if (local.length) {
        host.append(element("strong", "", "本题还有这些地方需要补充"));
        const list = element("ul");
        local.forEach(error => list.append(element("li", "", error.message.trim())));
        host.append(list);
      }
    }

    function showErrorSummary(errors) {
      ui.errorList.textContent = "";
      errors.forEach(error => {
        const item = element("li");
        const button = element("button", "", `第 ${error.caseIndex + 1} 题：${error.message.trim()}`);
        button.type = "button";
        button.addEventListener("click", () => {
          switchCase(error.caseIndex, false);
          window.setTimeout(() => {
            const field = document.getElementById(error.field);
            if (!field) return;
            for (let parent = field; parent; parent = parent.parentElement) {
              if (parent instanceof HTMLDetailsElement) parent.open = true;
            }
            field.tabIndex = field.tabIndex < 0 ? -1 : field.tabIndex;
            field.focus();
          }, 0);
        });
        item.append(button);
        ui.errorList.append(item);
      });
      ui.errorSummary.hidden = !errors.length;
      showCaseErrors(errors);
      if (errors.length) ui.errorSummary.focus();
    }

    function validateAll() {
      const errors = DATA.cases.flatMap((_, index) => validateCase(index));
      const declarations = state.declarations;
      if (!declarations.humanConfirmed) errors.push({ caseIndex: state.activeCase, field: "declarations", message: "请主动确认由标注者本人以真人身份完成判断。 " });
      if (!declarations.peerBlindConfirmed) errors.push({ caseIndex: state.activeCase, field: "declarations", message: "请主动确认未查看同伴答案。 " });
      if (!declarations.systemBlindConfirmed) errors.push({ caseIndex: state.activeCase, field: "declarations", message: "请主动确认未查看系统／模型预测。 " });
      if (!declarations.independenceConfirmed) errors.push({ caseIndex: state.activeCase, field: "declarations", message: "请亲自确认独立完成全部题目，并确认这份结果作为本次提交。 " });
      return errors;
    }

    function saveDraft(silent = false) {
      if (state.locked) return;
      try {
        localStorage.setItem(draftKey, JSON.stringify({
          draft_version: 1,
          public_input_sha256: DATA.public_input_sha256,
          author_setup_sha256: DATA.author_setup_sha256,
          execution_freeze_sha256: DATA.execution_freeze_sha256,
          reviewer_id: DATA.reviewer_id,
          manual_version: DATA.manual_version,
          active_case: state.activeCase,
          declarations: state.declarations,
          cases: state.cases
        }));
        ui.saveState.textContent = "进度已保存在当前浏览器。";
        if (!silent) announce("进度已保存。换浏览器或设备不会自动带过去；此页面没有上传答案。");
      } catch (_) {
        ui.saveState.textContent = "浏览器未能保存进度，请先不要关闭页面。";
        if (!silent) announce("浏览器拒绝保存进度，请先不要关闭页面。全部填完后可下载结果；若中途要离开，请联系负责人协助备份。 ");
      }
    }

    function scheduleDraft() {
      if (state.locked) return;
      window.clearTimeout(scheduleDraft.timer);
      scheduleDraft.timer = window.setTimeout(() => saveDraft(true), 500);
    }

    function loadDraft() {
      try {
        const raw = localStorage.getItem(draftKey);
        if (!raw) return false;
        const draft = JSON.parse(raw);
        if (
          draft.draft_version !== 1 || draft.public_input_sha256 !== DATA.public_input_sha256 ||
          draft.author_setup_sha256 !== DATA.author_setup_sha256 ||
          draft.execution_freeze_sha256 !== DATA.execution_freeze_sha256 ||
          draft.reviewer_id !== DATA.reviewer_id || draft.manual_version !== DATA.manual_version ||
          !Array.isArray(draft.cases) || draft.cases.length !== DATA.cases.length
        ) return false;
        state.declarations = { ...blankDeclarations(), ...(draft.declarations || {}) };
        state.cases = draft.cases.map(row => ({ ...blankCase(), ...row, evidenceRoles: { ...(row.evidenceRoles || {}) }, eventGroups: { ...(row.eventGroups || {}) }, explanations: { ...(row.explanations || {}) } }));
        state.activeCase = Number.isInteger(draft.active_case) && draft.active_case >= 0 && draft.active_case < DATA.cases.length ? draft.active_case : 0;
        return true;
      } catch (_) {
        return false;
      }
    }

    function stateFromLockedCase(caseIndex, row) {
      const next = blankCase();
      if (row.dimension !== DATA.cases[caseIndex].dimension) throw new Error("导入文件的维度与作者冻结目标不一致。 ");
      next.conflictLevel = row.conflict_level;
      next.materialCoverage = row.material_coverage;
      next.phenomena = row.phenomena.join("\n");
      next.reasonCodes = row.reason_codes.join("\n");
      next.confidence = String(row.confidence);
      next.reviewNote = row.review_note;
      row.evidence.forEach(item => { next.evidenceRoles[item.evidence_id] = item.role; });
      row.independent_event_groups.forEach(group => group.evidence_ids.forEach(id => { next.eventGroups[id] = group.event_group_id; }));
      row.explanation_links.forEach(link => {
        next.explanations[link.support_evidence_id] = {
          applicableCurrentIds: [...link.applicable_current_ids],
          causalRelation: link.causal_relation,
          temporalRelation: link.temporal_relation
        };
      });
      const known = evidenceMeta(caseIndex);
      if (Object.keys(next.evidenceRoles).some(id => !known.has(id))) throw new Error("导入文件引用了本案例不存在的证据。 ");
      if (Object.entries(next.evidenceRoles).some(([id, role]) => !roleAllowed(caseIndex, id, role))) throw new Error("导入文件的 B/C/G 来源不符合作者冻结目标。 ");
      return next;
    }

    function validateImportedBundle(bundle) {
      if (!exactKeys(bundle, BUNDLE_KEYS)) throw new Error("文件格式不对，请选择从这份页面下载的本人结果文件。 ");
      if (
        bundle.schema_version !== DATA.annotation_schema_version || bundle.dataset_id !== DATA.dataset_id ||
        bundle.public_input_sha256 !== DATA.public_input_sha256 || bundle.reviewer_kind !== "human" ||
        bundle.reviewer_id !== DATA.reviewer_id || bundle.manual_version !== DATA.manual_version ||
        bundle.blinded_to_peer !== true || bundle.blinded_to_system_prediction !== true ||
        bundle.independence_declaration !== true || !Array.isArray(bundle.cases)
      ) throw new Error("导入文件与当前数据集、标注者或手册版本不一致。不会载入他人的标注。 ");
      const byCase = new Map();
      bundle.cases.forEach(row => {
        if (!exactKeys(row, CASE_KEYS) || byCase.has(row.case_id)) throw new Error("导入文件含重复案例或未知字段。 ");
        if (!Array.isArray(row.evidence) || !row.evidence.every(item => exactKeys(item, ["evidence_id", "role"]))) throw new Error("导入文件的证据结构无效。 ");
        if (!Array.isArray(row.independent_event_groups) || !row.independent_event_groups.every(item => exactKeys(item, ["event_group_id", "evidence_ids"]))) throw new Error("导入文件的事件组结构无效。 ");
        if (!Array.isArray(row.explanation_links) || !row.explanation_links.every(item => exactKeys(item, ["support_evidence_id", "applicable_current_ids", "causal_relation", "temporal_relation"]))) throw new Error("导入文件的解释绑定结构无效。 ");
        if (!Array.isArray(row.phenomena) || !Array.isArray(row.reason_codes)) throw new Error("导入文件的标签结构无效。 ");
        byCase.set(row.case_id, row);
      });
      if (byCase.size !== DATA.cases.length || DATA.cases.some(row => !byCase.has(row.case_id))) throw new Error("导入文件没有完整覆盖当前公开案例。 ");
      const nextCases = DATA.cases.map((caseRow, index) => stateFromLockedCase(index, byCase.get(caseRow.case_id)));
      const previous = state.cases;
      const previousDeclarations = state.declarations;
      state.cases = nextCases;
      state.declarations = {
        humanConfirmed: bundle.reviewer_kind === "human",
        peerBlindConfirmed: bundle.blinded_to_peer === true,
        systemBlindConfirmed: bundle.blinded_to_system_prediction === true,
        independenceConfirmed: bundle.independence_declaration === true
      };
      const errors = validateAll();
      if (errors.length) {
        state.cases = previous;
        state.declarations = previousDeclarations;
        throw new Error(`导入文件未通过当前协议校验：${errors[0].message.trim()}`);
      }
      const rebuiltByCase = new Map(
        DATA.cases.map((caseRow, index) => [caseRow.case_id, buildCase(index)])
      );
      if (bundle.cases.some(row => !deepEqual(row, rebuiltByCase.get(row.case_id)))) {
        state.cases = previous;
        state.declarations = previousDeclarations;
        throw new Error("导入文件不能由本页无损还原；可能含重复值、非法类型或不受支持的协议字段。 ");
      }
      return nextCases;
    }

    function downloadLocked() {
      const errors = validateAll();
      if (errors.length) {
        showErrorSummary(errors);
        announce(`还有 ${errors.length} 项需要修正。`);
        return;
      }
      showErrorSummary([]);
      const payload = JSON.stringify(buildBundle(), null, 2) + "\n";
      const blob = new Blob([payload], { type: "application/json;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      const safeReviewer = DATA.reviewer_id.replace(/[^\p{L}\p{N}._-]+/gu, "-").slice(0, 80) || "reviewer";
      anchor.href = url;
      anchor.download = `ooc-annotation-${safeReviewer}.locked.json`;
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
      state.locked = true;
      try { localStorage.removeItem(draftKey); } catch (_) {}
      renderCase();
      announce("结果文件已下载。请把文件发回负责人；当前为只读，后续修改不会改变已下载的文件。 ");
    }

    document.getElementById("reviewer-summary").textContent = `标注者：${DATA.reviewer_id}　手册：${DATA.manual_version}`;
    document.getElementById("freeze-summary").textContent = `作者设置 ${DATA.author_setup_sha256}　执行冻结 ${DATA.execution_freeze_sha256}`;
    document.getElementById("save-draft").addEventListener("click", () => saveDraft(false));
    document.getElementById("import-trigger").addEventListener("click", () => ui.importFile.click());
    document.getElementById("clear-draft").addEventListener("click", () => {
      if (!window.confirm("清除这份文件在当前浏览器保存的全部进度？这会删掉未提交的填写内容；已下载的结果文件不受影响。")) return;
      try { localStorage.removeItem(draftKey); } catch (_) {}
      state.cases = DATA.cases.map(() => blankCase());
      state.declarations = blankDeclarations();
      state.activeCase = 0;
      showErrorSummary([]);
      updateNavigation();
      renderCase();
      ui.saveState.textContent = "进度已清除，之后填写会重新保存。";
      announce("当前浏览器的本地进度已清除。 ");
    });
    ui.download.addEventListener("click", downloadLocked);
    ui.unlock.addEventListener("click", () => {
      state.locked = false;
      renderCase();
      saveDraft(true);
      announce("现在可以修改了。再次完成任务会下载一份新结果；已下载或已发出的旧文件不会改变。 ");
    });
    ui.caseSelect.addEventListener("change", () => switchCase(Number(ui.caseSelect.value)));
    ui.importFile.addEventListener("change", async () => {
      const file = ui.importFile.files && ui.importFile.files[0];
      ui.importFile.value = "";
      if (!file) return;
      if (file.size > 8 * 1024 * 1024) { announce("导入文件超过 8 MiB，未载入。 "); return; }
      try {
        const bundle = JSON.parse(await file.text());
        state.cases = validateImportedBundle(bundle);
        state.activeCase = 0;
        state.locked = true;
        showErrorSummary([]);
        updateNavigation();
        renderCase();
        announce("已打开你的结果文件，当前为只读。需要改动时，点击“修改这份结果”。 ");
      } catch (error) {
        announce(error instanceof Error ? error.message : "导入文件无效。 ");
      }
    });
    document.addEventListener("keydown", event => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        saveDraft(false);
      }
      if (event.altKey && event.key === "ArrowDown") {
        event.preventDefault();
        switchCase(Math.min(DATA.cases.length - 1, state.activeCase + 1));
      }
      if (event.altKey && event.key === "ArrowUp") {
        event.preventDefault();
        switchCase(Math.max(0, state.activeCase - 1));
      }
    });
    window.addEventListener("pagehide", () => saveDraft(true));

    const restored = loadDraft();
    updateNavigation();
    renderCase();
    if (restored) {
      ui.saveState.textContent = "已恢复当前浏览器保存的进度。";
      announce("已恢复你在当前浏览器保存的进度，可以接着填写。 ");
    }
  })();
  </script>
</body>
</html>
'''


if __name__ == "__main__":
    raise SystemExit(main())
