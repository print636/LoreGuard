"""The optional launcher must never inherit another LoreGuard instance's config."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("runtime", ["", "relative/path"])
def test_local_gateway_factory_rejects_missing_or_relative_runtime(monkeypatch, runtime):
    from scripts import local_pro_gateway_app

    monkeypatch.setenv("LOREGUARD_PRO_GATEWAY_RUNTIME", runtime)
    with pytest.raises(RuntimeError, match="fresh Python process|absolute runtime"):
        local_pro_gateway_app.create_app()


def test_local_gateway_factory_excludes_env_and_dotenv_and_keeps_login(tmp_path):
    runtime = tmp_path / "gateway"
    runtime.mkdir()
    (runtime / "loreguard-data").mkdir()
    (runtime / "installation.json").write_text(
        json.dumps(
            {
                "owner": "LoreGuardLocalProGateway/v1",
                "version": "8.0.33",
                "runtime": str(runtime),
            }
        ),
        encoding="utf-8",
    )
    # A synthetic conflicting dotenv must never be constructed/read by Settings.
    (tmp_path / ".env").write_text(
        "AUTH_MODE=anonymous\nDEPLOYMENT_ENVIRONMENT=production\n"
        "OPENAI_API_KEY=synthetic-must-not-inherit\nDATABASE_URL=not-a-database\n",
        encoding="utf-8",
    )
    code = r'''
import json
import sys
from pathlib import Path
from fastapi.testclient import TestClient
sys.path.insert(0, sys.argv[1])
from scripts import local_pro_gateway_app as factory
factory._RUNTIME = Path(sys.argv[2])
application = factory.create_app()
from app.config import get_settings
settings = get_settings()
assert settings.auth_mode == "required"
assert settings.deployment_environment == "local"
assert settings.openai_api_key == ""
assert settings.account_model_keyring_json == ""
assert settings.account_model_keyring_file == ""
assert settings.local_model_gateway_enabled is True
assert settings.local_model_gateway_origins == "http://127.0.0.1:8317"
assert settings.enable_model_extraction is True
assert settings.enable_review_agent is False
assert settings.enable_embeddings is False
assert settings.use_celery is False
assert str(Path(sys.argv[2]) / "loreguard-data") in settings.account_model_local_key_path
with TestClient(application, base_url="http://127.0.0.1:8787") as client:
    health = client.get("/health")
    assert health.status_code == 200
    assert health.headers["x-loreguard-local-pro"] == "1"
    assert client.get("/api/v1/auth/me").status_code == 401
    assert client.get("/api/v1/projects").status_code == 401
    assert client.get("/api/v1/account/model-provider").status_code == 401
    assert client.get("/health", headers={"Host": "untrusted.example"}).status_code == 400
try:
    factory.create_app()
except RuntimeError as failure:
    assert "fresh Python process" in str(failure)
else:
    raise AssertionError("A second factory initialization must fail closed")
print("isolated gateway application checks passed")
'''
    environment = {
        "LOREGUARD_PRO_GATEWAY_RUNTIME": str(runtime),
        "AUTH_MODE": "anonymous",
        "DEPLOYMENT_ENVIRONMENT": "production",
        "DATABASE_URL": "not-a-database",
        "OPENAI_API_KEY": "synthetic-must-not-inherit",
        "ACCOUNT_MODEL_KEYRING_FILE": str(tmp_path / "must-not-read.key"),
    }
    # Python on Windows needs the system directory; this is not an app setting.
    if "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    result = subprocess.run(
        [sys.executable, "-c", code, str(REPOSITORY), str(runtime)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "isolated gateway application checks passed" in result.stdout
