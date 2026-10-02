from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import backup_restore_rehearsal as rehearsal


OWNER = "a" * 16
CONTAINER_ID = "b" * 64
PASSWORD = "c" * 64
IMAGE_ID = "sha256:" + "d" * 64


def _owned():
    return rehearsal.OwnedContainer(CONTAINER_ID, rehearsal.NAME_PREFIX + OWNER, OWNER)


def _metadata():
    return {
        "id": CONTAINER_ID, "name": "/" + rehearsal.NAME_PREFIX + OWNER,
        "labels": {rehearsal.PURPOSE_LABEL: rehearsal.PURPOSE, rehearsal.OWNER_LABEL: OWNER},
        "ports": {"5432/tcp": [{"HostIp": "127.0.0.1", "HostPort": "45678"}]},
    }


@pytest.mark.parametrize("change", (
    {"id": "d" * 64}, {"name": "/some-user-container"}, {"labels": {}},
    {"labels": {rehearsal.PURPOSE_LABEL: rehearsal.PURPOSE, rehearsal.OWNER_LABEL: "d" * 16}},
    {"labels": {rehearsal.PURPOSE_LABEL: "not-a-rehearsal", rehearsal.OWNER_LABEL: OWNER}},
))
def test_cleanup_refuses_every_ownership_mismatch(monkeypatch, change):
    metadata = {**_metadata(), **change}
    calls = []

    def fake_docker(args, **_kwargs):
        calls.append(args)
        return json.dumps(metadata).encode()

    monkeypatch.setattr(rehearsal, "docker_command", fake_docker)
    with pytest.raises(rehearsal.RehearsalError, match="container_ownership_unverified"):
        rehearsal.remove_owned_container(_owned())
    assert len(calls) == 1 and calls[0][0] == "inspect"


def test_cleanup_removes_only_verified_exact_id_and_its_volumes(monkeypatch):
    calls = []

    def fake_docker(args, **_kwargs):
        calls.append(args)
        return json.dumps(_metadata()).encode() if args[0] == "inspect" else b"removed"

    monkeypatch.setattr(rehearsal, "docker_command", fake_docker)
    rehearsal.remove_owned_container(_owned())
    assert calls[-1] == ["rm", "--force", "--volumes", CONTAINER_ID]
    assert not any("prune" in args or "down" in args for args in calls)


def test_invalid_container_id_is_rejected_before_any_docker_command(monkeypatch):
    monkeypatch.setattr(rehearsal, "docker_command", lambda *_a, **_k: pytest.fail("must not run Docker"))
    with pytest.raises(rehearsal.RehearsalError, match="container_ownership_unverified"):
        rehearsal.inspect_owned(rehearsal.OwnedContainer("--all", _owned().name, OWNER))


@pytest.mark.parametrize("binding", (
    [{"HostIp": "0.0.0.0", "HostPort": "45678"}],
    [{"HostIp": "127.0.0.1", "HostPort": "0"}],
    [{"HostIp": "127.0.0.1", "HostPort": "65536"}],
    [{"HostIp": "127.0.0.1", "HostPort": "abc"}], [], [None],
))
def test_database_port_must_be_one_verified_loopback_binding(binding):
    with pytest.raises(rehearsal.RehearsalError, match="container_port_unverified"):
        rehearsal._published_port({"ports": {"5432/tcp": binding}})
    assert rehearsal._published_port(_metadata()) == 45678


def test_credentials_only_travel_in_process_environment(monkeypatch):
    calls = []
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-forwarded")
    monkeypatch.setenv("DATABASE_URL", "must-not-be-forwarded")

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"result", stderr=b"")

    monkeypatch.setattr(rehearsal.subprocess, "run", fake_run)
    result = rehearsal.docker_command(["create", "--env", "POSTGRES_PASSWORD"], password=PASSWORD)
    args, kwargs = calls[0]
    assert result == b"result"
    assert PASSWORD not in " ".join(args)
    assert kwargs["env"]["POSTGRES_PASSWORD"] == kwargs["env"]["PGPASSWORD"] == PASSWORD
    assert "OPENAI_API_KEY" not in kwargs["env"] and "DATABASE_URL" not in kwargs["env"]
    assert kwargs["shell"] is False and kwargs["stderr"] == subprocess.PIPE


@pytest.mark.parametrize("failure", ("returncode", "timeout", "missing"))
def test_command_failure_never_exposes_stderr_or_credentials(monkeypatch, failure):
    def fake_run(*_args, **_kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(["fake", PASSWORD], 1, stderr=PASSWORD.encode())
        if failure == "missing":
            raise OSError(PASSWORD)
        return SimpleNamespace(returncode=1, stdout=PASSWORD.encode(), stderr=PASSWORD.encode())

    monkeypatch.setattr(rehearsal.subprocess, "run", fake_run)
    with pytest.raises(rehearsal.RehearsalError) as error:
        rehearsal.docker_command(["version"], code="docker_unavailable")
    assert str(error.value) == "docker_unavailable"
    assert PASSWORD not in str(error.value)
    assert str(rehearsal.RehearsalError(f"database error at {PASSWORD}")) == "rehearsal_internal_error"


@pytest.mark.parametrize("database,port,password", (
    ("loreguard", 5432, PASSWORD), (rehearsal.SOURCE_DB, 0, PASSWORD),
    (rehearsal.SOURCE_DB, 5432, "user-secret"),
))
def test_target_allowlist_cannot_address_an_existing_user_database(database, port, password):
    with pytest.raises(rehearsal.RehearsalError, match="database_target_unverified"):
        rehearsal._connection_parameters(port, password, database)
    assert rehearsal._connection_parameters(45678, PASSWORD, rehearsal.RESTORED_DB)["host"] == "127.0.0.1"


def test_artifact_location_is_unique_under_repository_and_rejects_path_fragments(tmp_path):
    artifact = rehearsal.artifact_directory(tmp_path, OWNER)
    assert artifact.is_relative_to(tmp_path.resolve())
    assert artifact.parent.name == "backup-restore-rehearsal"
    with pytest.raises(FileExistsError):
        rehearsal.artifact_directory(tmp_path, OWNER)
    with pytest.raises(rehearsal.RehearsalError, match="artifact_path_unverified"):
        rehearsal.artifact_directory(tmp_path, "../../")


def test_canonical_row_fingerprints_include_json_values_and_unicode():
    first = [{"正文": "合成 😀", "nested": {"version": 1, "flags": [True, False]}}]
    reordered = [{"nested": {"flags": [True, False], "version": 1}, "正文": "合成 😀"}]
    changed = [{"正文": "合成 😀", "nested": {"version": 2, "flags": [True, False]}}]
    assert rehearsal.fingerprint_rows(first) == rehearsal.fingerprint_rows(reordered)
    assert rehearsal.fingerprint_rows(first) != rehearsal.fingerprint_rows(changed)


@pytest.mark.parametrize("host", ("ssh://some-server", "tcp://127.0.0.1:2375", "tcp://10.0.0.1:2376", "npipe:////other-server/pipe/docker_engine"))
def test_remote_docker_transport_is_rejected_before_daemon_or_database_use(monkeypatch, host):
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setenv("DOCKER_HOST", host)
    monkeypatch.setattr(rehearsal, "docker_command", lambda *_a, **_k: pytest.fail("must not address remote daemon"))
    with pytest.raises(rehearsal.RehearsalError, match="remote_docker_context_unsupported"):
        rehearsal._require_local_docker_transport()


@pytest.mark.parametrize("endpoint", ("unix:///var/run/docker.sock", "npipe:////./pipe/dockerDesktopLinuxEngine"))
def test_local_socket_context_is_accepted_and_explicit_context_wins(monkeypatch, endpoint):
    monkeypatch.setenv("DOCKER_CONTEXT", "local-desktop")
    monkeypatch.setenv("DOCKER_HOST", "ssh://ignored-because-context-wins")
    calls = []

    def fake_docker(args, **_kwargs):
        calls.append(args)
        return json.dumps(endpoint).encode()

    monkeypatch.setattr(rehearsal, "docker_command", fake_docker)
    rehearsal._require_local_docker_transport()
    assert calls[0][-1] == "local-desktop"


def test_invalid_image_identity_prevents_container_creation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(rehearsal, "_require_local_docker_transport", lambda: None)

    def fake_docker(args, **_kwargs):
        calls.append(args)
        return b"not-an-immutable-image-id" if args[0] == "image" else b"ok"

    monkeypatch.setattr(rehearsal, "docker_command", fake_docker)
    result = rehearsal.run_rehearsal(tmp_path)
    assert result["status"] == "failed" and result["failure_code"] == "image_identity_unverified"
    assert not any(args[0] == "create" for args in calls)


def test_unavailable_docker_writes_safe_failure_without_touching_databases(tmp_path, monkeypatch):
    def unavailable(*_a, **_k):
        raise rehearsal.RehearsalError("docker_unavailable")

    monkeypatch.setattr(rehearsal, "docker_command", unavailable)
    monkeypatch.setattr(rehearsal, "connect_rehearsal", lambda *_a: pytest.fail("must not connect"))
    result = rehearsal.run_rehearsal(tmp_path)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert result["status"] == "failed" and result["failure_code"] == "docker_unavailable"
    assert report["container_cleanup"] == "not_created"
    assert report["stage"] == "docker_preflight"
    assert not list(Path(result["report"]).parent.glob("*.pgcustom"))


@pytest.mark.parametrize("restore_fails", (False, True))
def test_mock_pipeline_restores_exact_binary_and_always_cleans_only_own_container(tmp_path, monkeypatch, restore_fails):
    # An orchestration contract, not a real PostgreSQL/restore quality result.
    calls = []
    passwords = []
    token_counter = iter((OWNER, PASSWORD))
    monkeypatch.setattr(rehearsal.secrets, "token_hex", lambda _size: next(token_counter))
    monkeypatch.setattr(rehearsal, "_require_local_docker_transport", lambda: None)

    def fake_docker(args, **kwargs):
        calls.append(args)
        if kwargs.get("password"):
            passwords.append(kwargs["password"])
        if args[0] == "inspect":
            return json.dumps(_metadata()).encode()
        if args[0] == "create":
            return CONTAINER_ID.encode()
        if args[0] == "image":
            return IMAGE_ID.encode()
        if "pg_dump" in args:
            return b"PGDMP-synthetic-unit-only"
        if "pg_restore" in args:
            assert kwargs["input_bytes"] == b"PGDMP-synthetic-unit-only"
            if restore_fails:
                raise rehearsal.RehearsalError("restore_failed")
        return b"ok"

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *_a):
            return False

        def execute(self, statement):
            assert statement == f"CREATE DATABASE {rehearsal.RESTORED_DB}"

    monkeypatch.setattr(rehearsal, "docker_command", fake_docker)
    monkeypatch.setattr(rehearsal, "connect_rehearsal", lambda *_a: FakeConnection())
    monkeypatch.setattr(rehearsal, "_migrate_source", lambda *_a: "synthetic-head")
    monkeypatch.setattr(rehearsal, "_assert_empty", lambda _c: None)
    monkeypatch.setattr(rehearsal, "_seed", lambda *_a: None)
    monkeypatch.setattr(rehearsal, "_snapshot", lambda _c: {"tables": {"synthetic": {"row_count": 2, "sha256": "a" * 64}}})
    monkeypatch.setattr(rehearsal, "_verify_data", lambda *_a: {
        "encryption_key_persisted": False, "migration_head": "synthetic-head",
        "workspace_query_isolation_verified": True,
    })
    monkeypatch.setattr(rehearsal, "_verify_constraints", lambda _c: ["synthetic-unit-probe"])
    result = rehearsal.run_rehearsal(tmp_path)
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    assert result["status"] == ("failed" if restore_fails else "passed")
    if restore_fails:
        assert report["failure_code"] == "restore_failed"
    assert report["container_cleanup"] == "removed_owned_container_and_its_anonymous_volumes"
    assert calls[-1] == ["rm", "--force", "--volumes", CONTAINER_ID]
    create = next(args for args in calls if args[0] == "create")
    assert create[-1] == IMAGE_ID and rehearsal.IMAGE not in create
    ready = next(args for args in calls if "pg_isready" in args)
    assert ready[ready.index("--host") + 1] == "127.0.0.1"
    assert all(PASSWORD not in " ".join(args) for args in calls)
    assert set(passwords) == {PASSWORD}
    assert PASSWORD not in json.dumps(report)
    assert (Path(result["report"]).parent / "synthetic.pgcustom").read_bytes() == b"PGDMP-synthetic-unit-only"
    assert not any("prune" in args or "down" in args for args in calls)
    restore = next(args for args in calls if "pg_restore" in args)
    assert "--clean" not in restore and rehearsal.RESTORED_DB in restore
