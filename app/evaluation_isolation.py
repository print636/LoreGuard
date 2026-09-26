"""Read-only identity checks for explicitly provisioned local evaluation databases.

The identity table is deliberately outside normal migrations.  Ordinary
databases never acquire a marker from application startup or a GET request.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sqlite3
from uuid import UUID, uuid4

from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError


SCHEMA_VERSION = "character-axis-evaluation-isolation-v1"
IDENTITY_TABLE = "character_axis_eval_identity_v1"
_PATH_DOMAIN = "loreguard-character-axis-eval-sqlite-path-v1\0"
_INSTANCE_DOMAIN = "loreguard-character-axis-eval-instance-v1\0"


class EvaluationIsolationUnavailable(Exception):
    """Fixed-code failure; never include a path, URL, or database error."""


def allowed_root() -> Path:
    # For this repository layout: Desktop/dev/projects/LoreGuard ->
    # Desktop/dev/artifacts/character-axis-direction-v1.
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root.parent.parent / "artifacts" / "character-axis-direction-v1"


def _uuid4(value: str) -> bool:
    try:
        parsed = UUID(value)
    except (TypeError, ValueError, AttributeError):
        return False
    return parsed.version == 4 and str(parsed) == value


def _canonical_path(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise EvaluationIsolationUnavailable()
        resolved = path.resolve(strict=True)
        root = allowed_root()
        # The designated artifacts directory must not itself redirect to
        # another location (in particular a daily workspace directory).
        if root.resolve(strict=True) != root.absolute():
            raise EvaluationIsolationUnavailable()
        if (
            not resolved.is_file()
            or resolved.parent != root
            or resolved == (Path(__file__).resolve().parents[1] / "loreguard.db").resolve()
            or resolved.stat().st_nlink != 1
        ):
            raise EvaluationIsolationUnavailable()
        return resolved
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise EvaluationIsolationUnavailable() from exc


def database_path_sha256(path: Path) -> str:
    resolved = _canonical_path(path)
    encoded = (_PATH_DOMAIN + os.path.normcase(str(resolved))).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def instance_id_sha256(instance_id: str) -> str:
    if not _uuid4(instance_id):
        raise EvaluationIsolationUnavailable()
    return hashlib.sha256((_INSTANCE_DOMAIN + instance_id).encode("utf-8")).hexdigest()


def provision_new_database(path: Path) -> dict[str, str]:
    """Create an exclusive, marker-only SQLite file for a future eval API.

    This is never called from the API.  The product schema is migrated later
    by its ordinary startup path, after the marker has been validated.
    """

    root = allowed_root()
    try:
        if not path.is_absolute() or path.parent != root or path.exists():
            raise EvaluationIsolationUnavailable()
        root.mkdir(parents=True, exist_ok=True)
        if root.resolve(strict=True) != root.absolute():
            raise EvaluationIsolationUnavailable()
        with path.open("xb"):
            pass
        digest = database_path_sha256(path)
        eval_db_id = str(uuid4())
        with sqlite3.connect(str(path)) as connection:
            connection.execute(
                f"CREATE TABLE {IDENTITY_TABLE} ("
                "singleton INTEGER PRIMARY KEY CHECK (singleton = 1), "
                "eval_db_id TEXT NOT NULL, path_sha256 TEXT NOT NULL)"
            )
            connection.execute(
                f"INSERT INTO {IDENTITY_TABLE} "
                "(singleton, eval_db_id, path_sha256) VALUES (1, ?, ?)",
                (eval_db_id, digest),
            )
        return {"eval_db_id": eval_db_id, "database_path_sha256": digest}
    except (OSError, sqlite3.Error) as exc:
        raise EvaluationIsolationUnavailable() from exc


def read_live_identity(
    engine: Engine, instance_id: str, *, require_product_schema: bool,
) -> dict[str, str | bool]:
    """Read the actual connected SQLite file, sentinel and optional freshness.

    No URL, raw path, provider setting, or exception details leave this module.
    """

    instance_hash = instance_id_sha256(instance_id)
    if engine.dialect.name != "sqlite":
        raise EvaluationIsolationUnavailable()
    try:
        configured_file = engine.url.database
        configured_path = Path(configured_file) if configured_file else None
        if (
            configured_path is None
            or not configured_path.is_absolute()
            or configured_path.resolve(strict=True) != configured_path.absolute()
        ):
            raise EvaluationIsolationUnavailable()
        with engine.connect() as connection:
            files = connection.exec_driver_sql("PRAGMA database_list").all()
            if len(files) != 1 or files[0][1] != "main" or not files[0][2]:
                raise EvaluationIsolationUnavailable()
            connected_path = Path(files[0][2])
            if connected_path.resolve(strict=True) != configured_path.resolve(strict=True):
                raise EvaluationIsolationUnavailable()
            path_hash = database_path_sha256(connected_path)
            rows = connection.exec_driver_sql(
                f"SELECT eval_db_id, path_sha256 FROM {IDENTITY_TABLE}"
            ).all()
            if (
                len(rows) != 1
                or not _uuid4(rows[0][0])
                or rows[0][1] != path_hash
            ):
                raise EvaluationIsolationUnavailable()
            result: dict[str, str | bool] = {
                "schema_version": SCHEMA_VERSION,
                "isolated_sqlite": True,
                "eval_db_id": rows[0][0],
                "database_path_sha256": path_hash,
                "instance_id_sha256": instance_hash,
            }
            if require_product_schema:
                result["fresh_for_prepare"] = not bool(connection.exec_driver_sql(
                    "SELECT 1 FROM projects LIMIT 1"
                ).first())
            return result
    except (SQLAlchemyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise EvaluationIsolationUnavailable() from exc
