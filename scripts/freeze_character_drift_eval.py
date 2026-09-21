from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "data" / "evaluation-character-drift-v1"


def file_summary(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    nonempty_lines = [line for line in payload.decode("utf-8").splitlines() if line.strip()]
    return {
        "file": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "case_count": len(nonempty_lines),
        "bytes": len(payload),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report immutable hashes for existing character-drift splits."
    )
    parser.add_argument(
        "--split", choices=("dev", "holdout", "all"), default="all"
    )
    args = parser.parse_args()
    names = ("dev", "holdout") if args.split == "all" else (args.split,)
    rows = []
    missing = []
    for name in names:
        path = ROOT / f"{name}.jsonl"
        if not path.exists():
            missing.append(path.name)
            continue
        rows.append(file_summary(path))
    print(
        json.dumps(
            {
                "dataset": "LoreGuard character drift v1",
                "files": rows,
                "missing": missing,
                "note": (
                    "Hashes describe existing bytes only; this tool never creates, "
                    "loads or scores absent HOLDOUT answers."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
