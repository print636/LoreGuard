"""Verify two sanitized Evidence Investigator dev artifacts without model calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_evidence_investigator_live import compare_dev_artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed comparison of two qualified Evidence Investigator dev artifacts."
        )
    )
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = compare_dev_artifacts(args.first, args.second)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
