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


def test_dev_fixture_validation_never_opens_holdout_sources(monkeypatch):
    validator = _load_validator()
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text
    original_open = Path.open

    def reject_holdout(path, operation):
        if "holdout" in path.parts:
            raise AssertionError(f"dev validation {operation} a holdout source")

    def guarded_read_bytes(path):
        reject_holdout(path, "read")
        return original_read_bytes(path)

    def guarded_read_text(path, *args, **kwargs):
        reject_holdout(path, "read")
        return original_read_text(path, *args, **kwargs)

    def guarded_open(path, *args, **kwargs):
        reject_holdout(path, "opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)
    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "open", guarded_open)

    validator.main(selected_split="dev")
