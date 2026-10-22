"""Reissue existing offline annotation handouts without changing their inputs.

Reads only the two explicitly named public HTML pages and presentation guides.
Never locates a dataset directory, human submission, gold answer, or Provider.
The default is a read-only preflight; replacement requires --apply and retains
byte-for-byte backups of all four existing handouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_ooc_annotation_page import (  # noqa: E402
    PAGE_SCHEMA_VERSION,
    render_annotation_page,
)


PAGE_NAMES = ("reviewer-a-v1.html", "reviewer-b-v1.html")
MANUAL_HTML = "人工标注手册-零基础完整版.html"
MANUAL_MD = "人工标注手册-v1.md"
DATA_PATTERN = re.compile(
    r'<script id="annotation-data" type="application/json">(.*?)</script>',
    re.DOTALL,
)
MAX_HANDOUT_BYTES = 16 * 1024 * 1024


def _read_local(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Required regular local file missing: {path.name}")
    if path.stat().st_size > MAX_HANDOUT_BYTES:
        raise ValueError(f"Handout exceeds size limit: {path.name}")
    return path.read_bytes()


def _page_data(html: bytes) -> dict:
    matches = DATA_PATTERN.findall(html.decode("utf-8-sig"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one public annotation-data payload")
    data = json.loads(matches[0])
    if not isinstance(data, dict) or data.get("page_schema_version") != PAGE_SCHEMA_VERSION:
        raise ValueError("Unsupported annotation page schema")
    if not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("Annotation cases missing")
    if not isinstance(data.get("reviewer_id"), str) or not data["reviewer_id"]:
        raise ValueError("Reviewer identity missing")
    return data


def _backup(directory: Path, originals: dict[str, bytes]) -> Path:
    # mkdir(exist_ok=False) prevents overwriting any previous backup.
    for number in range(1, 1001):
        suffix = "" if number == 1 else f"-{number}"
        target = directory / f"原版备份-易读版升级{suffix}"
        try:
            target.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise ValueError("No available backup directory name")
    for name, payload in originals.items():
        shutil.copy2(directory / name, target / name)
        if hashlib.sha256((target / name).read_bytes()).digest() != hashlib.sha256(payload).digest():
            raise ValueError("Backup verification failed; no handouts replaced")
    return target


def refresh_handouts(
    directory: Path,
    guide_html: Path,
    guide_markdown: Path,
    *,
    apply: bool = False,
) -> dict:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Handout directory must be an existing local directory")
    directory = directory.resolve(strict=True)
    names = (*PAGE_NAMES, MANUAL_HTML, MANUAL_MD)
    originals = {name: _read_local(directory / name) for name in names}
    datasets = [_page_data(originals[name]) for name in PAGE_NAMES]
    if datasets[0]["reviewer_id"] == datasets[1]["reviewer_id"]:
        raise ValueError("A and B must have different reviewer identities")
    first = {key: value for key, value in datasets[0].items() if key != "reviewer_id"}
    second = {key: value for key, value in datasets[1].items() if key != "reviewer_id"}
    if first != second:
        raise ValueError("A and B public inputs differ; refusing presentation-only reissue")
    outputs = {
        name: render_annotation_page(data).encode("utf-8")
        for name, data in zip(PAGE_NAMES, datasets)
    }
    for name, data in zip(PAGE_NAMES, datasets):
        if _page_data(outputs[name]) != data:
            raise ValueError("Rendered public inputs changed; refusing replacement")
    outputs[MANUAL_HTML] = _read_local(guide_html)
    outputs[MANUAL_MD] = _read_local(guide_markdown)
    result = {
        "mode": "apply" if apply else "check-only",
        "case_count_per_reviewer": len(datasets[0]["cases"]),
        "public_inputs_unchanged": True,
        "reviewer_identities_preserved": True,
        "prepared_file_count": len(outputs),
    }
    if not apply:
        return result
    # Do not replace a file that changed between preflight and backup.
    for name, payload in originals.items():
        if _read_local(directory / name) != payload:
            raise ValueError("Handout changed during preflight; refusing replacement")
    backup = _backup(directory, originals)
    # All outputs are prepared and all original bytes backed up before writes.
    for name, payload in outputs.items():
        (directory / name).write_bytes(payload)
        if (directory / name).read_bytes() != payload:
            raise ValueError(f"Replacement verification failed; recover from {backup.name}")
    result["backup_directory"] = str(backup)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--guide-html", type=Path, required=True, help="Previously rendered public guide HTML")
    parser.add_argument("--guide-markdown", type=Path, default=REPO_ROOT / "docs/annotation-reader-guide.md")
    parser.add_argument("--apply", action="store_true", help="Back up and replace the four named handouts")
    args = parser.parse_args(argv)
    try:
        result = refresh_handouts(args.directory, args.guide_html, args.guide_markdown, apply=args.apply)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(1, f"Handouts not refreshed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
