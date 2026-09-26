import asyncio
import json
from pathlib import Path
import shutil

from alembic import command
from alembic.config import Config
from fastapi import Response
import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

from app import evaluation_isolation as isolation
from app.config import Settings


INSTANCE_ID = "44737d93-c1e2-44a2-9295-90e54837f31b"
ROOT = Path(__file__).resolve().parents[1]


def _engine(path):
    return create_engine(URL.create("sqlite", database=str(path)))


def _migrate(path):
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["database_url"] = str(URL.create("sqlite", database=str(path)))
    command.upgrade(config, "head")


def test_provision_migrate_identity_get_and_restart_are_stable(tmp_path, monkeypatch):
    from app import main
    from scripts import run_character_axis_direction_live as direction

    monkeypatch.setattr(isolation, "allowed_root", lambda: tmp_path)
    path = tmp_path / "dev-eval.sqlite3"
    provisioned = isolation.provision_new_database(path)
    assert isolation._uuid4(provisioned["eval_db_id"])
    with pytest.raises(isolation.EvaluationIsolationUnavailable):
        isolation.provision_new_database(path)
    engine = _engine(path)
    before = isolation.read_live_identity(
        engine, INSTANCE_ID, require_product_schema=False
    )
    assert before["eval_db_id"] == provisioned["eval_db_id"]
    assert before["database_path_sha256"] == provisioned["database_path_sha256"]
    assert "fresh_for_prepare" not in before
    _migrate(path)

    monkeypatch.setattr(main, "engine", engine)
    monkeypatch.setattr(main, "settings", Settings(
        database_url=str(URL.create("sqlite", database=str(path))),
        eval_isolation_instance_id=INSTANCE_ID,
    ))
    response = Response()
    payload = main.evaluation_isolation_identity(response)
    assert payload["fresh_for_prepare"] is True
    assert response.headers["Cache-Control"] == "no-store"
    assert payload["instance_id_sha256"] == isolation.instance_id_sha256(INSTANCE_ID)
    assert direction._expected_isolation(
        provisioned["eval_db_id"], INSTANCE_ID, str(path)
    ) == {key: value for key, value in payload.items() if key != "fresh_for_prepare"}
    assert str(path) not in json.dumps(payload)
    assert "sqlite:" not in json.dumps(payload)

    with engine.begin() as connection:
        connection.exec_driver_sql(
            "INSERT INTO projects (id, workspace_id, name, description, created_at) "
            "VALUES ('project-1', 'workspace-1', 'test', '', CURRENT_TIMESTAMP)"
        )
    assert main.evaluation_isolation_identity(Response())["fresh_for_prepare"] is False
    engine.dispose()
    restarted = _engine(path)
    after = isolation.read_live_identity(
        restarted, INSTANCE_ID, require_product_schema=True
    )
    assert after["eval_db_id"] == before["eval_db_id"]
    assert after["database_path_sha256"] == before["database_path_sha256"]
    assert after["fresh_for_prepare"] is False
    restarted.dispose()


def test_daily_database_copy_cannot_claim_eval_identity(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    monkeypatch.setattr(isolation, "allowed_root", lambda: root)
    path = root / "dev-eval.sqlite3"
    isolation.provision_new_database(path)

    daily = tmp_path / "daily.sqlite3"
    shutil.copy2(path, daily)
    daily_engine = _engine(daily)
    with pytest.raises(isolation.EvaluationIsolationUnavailable):
        isolation.read_live_identity(
            daily_engine, INSTANCE_ID, require_product_schema=False
        )
    daily_engine.dispose()


def test_symlink_cannot_claim_eval_identity(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    root.mkdir()
    monkeypatch.setattr(isolation, "allowed_root", lambda: root)
    path = root / "dev-eval.sqlite3"
    isolation.provision_new_database(path)

    symlink = root / "aliased.sqlite3"
    try:
        symlink.symlink_to(path)
    except (OSError, NotImplementedError):
        pytest.skip("file symlinks unavailable on this host")
    linked_engine = _engine(symlink)
    with pytest.raises(isolation.EvaluationIsolationUnavailable):
        isolation.read_live_identity(
            linked_engine, INSTANCE_ID, require_product_schema=False
        )
    linked_engine.dispose()


def test_disabled_or_wrong_instance_identity_is_not_reported(tmp_path, monkeypatch):
    from app import main

    monkeypatch.setattr(main, "settings", Settings(eval_isolation_instance_id=""))
    with pytest.raises(main.HTTPException) as exc:
        main.evaluation_isolation_identity(Response())
    assert exc.value.status_code == 404

    monkeypatch.setattr(isolation, "allowed_root", lambda: tmp_path)
    path = tmp_path / "dev-eval.sqlite3"
    isolation.provision_new_database(path)
    engine = _engine(path)
    assert isolation.read_live_identity(
        engine, INSTANCE_ID, require_product_schema=False
    )["instance_id_sha256"] != isolation.instance_id_sha256(
        "8a7a5a9a-7ef0-4cb0-b81f-3eadc4ec5c3f"
    )
    engine.dispose()


def test_eval_startup_refuses_unmarked_daily_db_before_migration(monkeypatch):
    from app import main

    monkeypatch.setattr(main, "settings", Settings(
        eval_isolation_instance_id=INSTANCE_ID,
    ))
    monkeypatch.setattr(
        main, "validate_provider_security_configuration", lambda _settings: None
    )
    monkeypatch.setattr(
        main, "read_live_identity",
        lambda *_a, **_k: (_ for _ in ()).throw(
            isolation.EvaluationIsolationUnavailable()
        ),
    )
    monkeypatch.setattr(
        main, "init_db", lambda: pytest.fail("migration attempted before isolation gate")
    )

    async def attempt_startup():
        async with main.lifespan(main.app):
            pass

    with pytest.raises(RuntimeError, match="evaluation_isolation_unavailable"):
        asyncio.run(attempt_startup())
