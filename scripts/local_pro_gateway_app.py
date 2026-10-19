"""Fresh-process, loopback-only LoreGuard instance for the local Pro gateway.

The launcher supplies only LOREGUARD_PRO_GATEWAY_RUNTIME (a non-secret fixed
directory). All normal environment/.env/file-secret settings are excluded.
Never import this factory into an already initialized application process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import stat
import sys

from starlette.middleware.trustedhost import TrustedHostMiddleware


_REPOSITORY = Path(__file__).resolve().parents[1]
_RUNTIME = _REPOSITORY.parents[1] / "tools" / "LoreGuard-Pro-Gateway"
_OWNER = "LoreGuardLocalProGateway/v1"


def _no_reparse(path: Path) -> None:
    for candidate in (path, *path.parents):
        try:
            details = candidate.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(details.st_mode) or getattr(details, "st_file_attributes", 0) & 0x400:
            raise RuntimeError("Local Pro runtime must not contain symlinks or junctions")


def _auth_secret(path: Path) -> str:
    _no_reparse(path)
    try:
        with path.open("x", encoding="utf-8") as output:
            output.write(secrets.token_urlsafe(48))
    except FileExistsError:
        pass
    value = path.read_text(encoding="utf-8").strip()
    if len(value) < 32:
        raise RuntimeError("Isolated authentication secret is invalid")
    return value


class LocalProBoundary:
    def __init__(self, application):
        self.application = application

    async def __call__(self, scope, receive, send):
        async def with_local_marker(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                message["headers"] = list(message.get("headers", [])) + [
                    (b"x-loreguard-local-pro", b"1")
                ]
            await send(message)

        await self.application(scope, receive, with_local_marker)


def create_app():
    if any(name.startswith("app.") and name != "app.config" for name in sys.modules):
        raise RuntimeError("Local Pro service requires a fresh Python process")
    raw_runtime = os.environ.get("LOREGUARD_PRO_GATEWAY_RUNTIME", "")
    runtime = Path(raw_runtime)
    if not raw_runtime or not runtime.is_absolute():
        raise RuntimeError("The launcher must provide its dedicated absolute runtime directory")
    _no_reparse(runtime)
    if runtime.resolve() != _RUNTIME.resolve():
        raise RuntimeError("The runtime directory must match the local gateway installation")
    manifest_path = runtime / "installation.json"
    _no_reparse(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("owner") != _OWNER or manifest.get("version") != "8.0.33" or Path(manifest.get("runtime", "")).resolve() != runtime.resolve():
        raise RuntimeError("Local gateway installation ownership/version mismatch")
    directory = runtime / "loreguard-data"
    _no_reparse(directory)
    if not directory.is_dir():
        raise RuntimeError("Create the private application directory through the launcher first")
    database = directory / "loreguard-pro-gateway.sqlite3"
    account_key = directory / "account-model-master.key"
    for path in (database, account_key):
        _no_reparse(path)

    from app import config
    from pydantic_settings import SettingsConfigDict

    class IsolatedSettings(config.Settings):
        model_config = SettingsConfigDict(env_file=None)

        @classmethod
        def settings_customise_sources(
            cls, settings_cls, init_settings, env_settings, dotenv_settings, file_secret_settings
        ):
            return (init_settings,)

    settings = IsolatedSettings(
        _env_file=None,
        deployment_environment="local",
        auth_mode="required",
        auth_secret_key=_auth_secret(directory / "auth-secret.txt"),
        auth_cookie_secure=False,
        auth_cookie_samesite="lax",
        cors_allowed_origins="http://127.0.0.1:8787,http://localhost:8787",
        database_url=f"sqlite:///{database.as_posix()}",
        use_celery=False,
        openai_api_key="",
        enable_model_extraction=True,
        account_model_local_key_path=str(account_key),
        local_model_gateway_enabled=True,
        local_model_gateway_origins="http://127.0.0.1:8317",
    )
    # Patch before the first module imports bind settings/engine/dependencies.
    config.get_settings.cache_clear()
    config.get_settings = lambda: settings
    from app.main import app

    return TrustedHostMiddleware(
        LocalProBoundary(app), allowed_hosts=["127.0.0.1", "localhost"]
    )
