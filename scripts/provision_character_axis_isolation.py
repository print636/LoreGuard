"""Provision one fresh, marker-only SQLite file for character-axis evaluation.

No service is started and no model is called.  The path must be a new direct
child of Desktop/dev/artifacts/character-axis-direction-v1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.evaluation_isolation import (
    EvaluationIsolationUnavailable,
    provision_new_database,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", required=True)
    args = parser.parse_args()
    try:
        result = provision_new_database(Path(args.db_path))
    except EvaluationIsolationUnavailable:
        print(json.dumps({"error": "evaluation_isolation_provision_failed"}))
        return 1
    print(json.dumps({
        "schema_version": "character-axis-evaluation-isolation-v1",
        **result,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
