import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docker-compose.yml"
PRODUCTION = ROOT / "docker-compose.production.yml"

PRODUCTION_ENV = {
    "PUBLIC_ORIGIN": "https://loreguard.example.test",
    "AUTH_SECRET_KEY": "test-only-auth-secret-that-is-long-enough",
    "POSTGRES_DB": "loreguard_test",
    "POSTGRES_USER": "loreguard_test_user",
    "POSTGRES_PASSWORD": "test-only-database-password",
    "DATABASE_URL": (
        "postgresql+psycopg://loreguard_test_user:"
        "test-only-database-password@postgres:5432/loreguard_test"
    ),
}


def _compose_config(
    *, env: dict[str, str], quiet: bool = False, observability: bool = False
):
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not installed; static overlay checks still ran")
    command = [
        "docker",
        "compose",
        "--project-directory",
        str(ROOT),
        "-f",
        str(BASE),
        "-f",
        str(PRODUCTION),
    ]
    if observability:
        command.extend(["--profile", "observability"])
    command.append("config")
    if quiet:
        command.append("--quiet")
    return subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, **env},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _rendered_config() -> dict:
    result = _compose_config(env=PRODUCTION_ENV)
    assert result.returncode == 0, result.stderr
    return yaml.safe_load(result.stdout)


def test_production_overlay_requires_all_security_and_database_inputs():
    for missing in PRODUCTION_ENV:
        env = {key: value for key, value in PRODUCTION_ENV.items() if key != missing}
        # Ensure a developer shell or repository .env cannot satisfy the value
        # whose fail-closed interpolation is under test.
        env[missing] = ""
        result = _compose_config(env=env, quiet=True)
        assert result.returncode != 0
        assert missing in result.stderr


def test_rendered_production_config_is_required_auth_and_loopback_only():
    config = _rendered_config()
    services = config["services"]

    for service_name in ("api", "worker"):
        environment = services[service_name]["environment"]
        assert environment["DEPLOYMENT_ENVIRONMENT"] == "production"
        assert environment["AUTH_MODE"] == "required"
        assert environment["AUTH_COOKIE_SECURE"] == "true"
        assert environment["AUTH_SECRET_KEY"] == PRODUCTION_ENV["AUTH_SECRET_KEY"]
        assert environment["CORS_ALLOWED_ORIGINS"] == PRODUCTION_ENV["PUBLIC_ORIGIN"]
        assert environment["DATABASE_URL"] == PRODUCTION_ENV["DATABASE_URL"]

    assert "ports" not in services["api"]
    # Profiled services are omitted unless explicitly selected.
    assert "prometheus" not in services
    assert "ports" not in services["postgres"]
    assert "ports" not in services["redis"]

    web_ports = services["web"]["ports"]
    assert len(web_ports) == 1
    assert web_ports[0]["host_ip"] == "127.0.0.1"
    assert web_ports[0]["published"] == "8080"
    assert web_ports[0]["target"] == 80

    database_environment = services["postgres"]["environment"]
    assert database_environment == {
        "POSTGRES_DB": PRODUCTION_ENV["POSTGRES_DB"],
        "POSTGRES_PASSWORD": PRODUCTION_ENV["POSTGRES_PASSWORD"],
        "POSTGRES_USER": PRODUCTION_ENV["POSTGRES_USER"],
    }
    assert database_environment["POSTGRES_PASSWORD"] != "loreguard"

    monitored = _compose_config(env=PRODUCTION_ENV, observability=True)
    assert monitored.returncode == 0, monitored.stderr
    monitored_services = yaml.safe_load(monitored.stdout)["services"]
    assert "ports" not in monitored_services["prometheus"]


def test_compose_merge_operator_support_is_validated_without_starting_containers():
    result = _compose_config(env=PRODUCTION_ENV, quiet=True)
    assert result.returncode == 0, result.stderr
