from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    ROOT
    / "data"
    / "evaluation"
    / "evidence_investigator_live"
    / "validate_fixture.py"
)


def _load_validator():
    spec = importlib.util.spec_from_file_location(
        "evidence_investigator_fixture_validator", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dev_fixture_validation_never_opens_holdout_source_bytes(monkeypatch):
    validator = _load_validator()
    original_read_bytes = Path.read_bytes

    def guarded_read_bytes(path):
        if "holdout" in path.parts:
            raise AssertionError("dev validation opened holdout source bytes")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    validator.main(selected_split="dev")
