"""Rehearse migrations + pg_dump/pg_restore on disposable synthetic databases.

No .env, application settings, user database, provider or evaluation data is
read. This is deliberately NOT a command for backing up a deployment database.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import secrets
import subprocess
import sys
import time
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE = "pgvector/pgvector:pg16"
NAME_PREFIX = "loreguard-rehearsal-"
PURPOSE_LABEL = "org.loreguard.rehearsal.purpose"
OWNER_LABEL = "org.loreguard.rehearsal.id"
PURPOSE = "backup-restore-v1"
SOURCE_DB = "loreguard_rehearsal_source"
RESTORED_DB = "loreguard_rehearsal_restored"
DATABASES = frozenset({SOURCE_DB, RESTORED_DB})
DB_USER = "rehearsal"
MAX_DUMP_BYTES = 64 * 1024 * 1024
IDENTITY_FORMAT = (
    '{"id":{{json .Id}},"name":{{json .Name}},'
    '"labels":{{json .Config.Labels}},"ports":{{json .NetworkSettings.Ports}}}'
)


class RehearsalError(Exception):
    """A fixed safe code; never expose subprocess stderr or database URLs."""

    def __init__(self, code: str):
        safe = code if isinstance(code, str) and re.fullmatch(r"[a-z_]{1,80}", code) else "rehearsal_internal_error"
        super().__init__(safe)


@dataclass(frozen=True)
class OwnedContainer:
    container_id: str
    name: str
    owner: str


def _command_env(password: str | None = None) -> dict[str, str]:
    # Do not forward unrelated application/provider credentials to Docker.
    allowed = {
        "PATH", "SystemRoot", "SYSTEMROOT", "WINDIR", "TEMP", "TMP",
        "HOME", "USERPROFILE", "DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG",
        "SSH_AUTH_SOCK", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    if password is not None:
        environment["POSTGRES_PASSWORD"] = password
        environment["PGPASSWORD"] = password
    return environment


def docker_command(
    args: list[str], *, password: str | None = None,
    input_bytes: bytes | None = None, code: str = "docker_command_failed",
    timeout: float = 60,
) -> bytes:
    try:
        result = subprocess.run(
            ["docker", *args], input=input_bytes,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=_command_env(password), timeout=timeout, check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RehearsalError(code) from None
    if result.returncode != 0:
        raise RehearsalError(code)
    return result.stdout


def _require_local_docker_transport() -> None:
    environment = _command_env()
    context = environment.get("DOCKER_CONTEXT", "").strip()
    explicit_host = environment.get("DOCKER_HOST", "").strip()
    if context:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", context):
            raise RehearsalError("docker_context_unavailable")
        inspect_args = ["context", "inspect", "--format", "{{json .Endpoints.docker.Host}}", context]
    elif explicit_host:
        inspect_args = None
    else:
        inspect_args = ["context", "inspect", "--format", "{{json .Endpoints.docker.Host}}"]
    if inspect_args is not None:
        try:
            endpoint = json.loads(docker_command(inspect_args, code="docker_context_unavailable", timeout=10))
        except (ValueError, TypeError):
            raise RehearsalError("docker_context_unavailable") from None
    else:
        endpoint = explicit_host
    if not isinstance(endpoint, str):
        raise RehearsalError("docker_context_unavailable")
    endpoint = endpoint.replace("\\", "/")
    # Port forwarding belongs to the daemon host, not a remote CLI client.
    # Even localhost TCP tunnels are not accepted as proof of daemon locality.
    if not (endpoint.startswith("unix:///") or endpoint.startswith("npipe:////./pipe/")):
        raise RehearsalError("remote_docker_context_unsupported")


def validate_owned_container(metadata: object, owned: OwnedContainer) -> dict:
    """Refuse cleanup unless ID, unique name, purpose and random label agree."""
    if (
        not isinstance(metadata, dict)
        or not re.fullmatch(r"[a-f0-9]{64}", owned.container_id)
        or not re.fullmatch(r"[a-f0-9]{16}", owned.owner)
        or owned.name != NAME_PREFIX + owned.owner
        or metadata.get("id") != owned.container_id
        or metadata.get("name") != "/" + owned.name
        or not isinstance(metadata.get("labels"), dict)
        or metadata["labels"].get(PURPOSE_LABEL) != PURPOSE
        or metadata["labels"].get(OWNER_LABEL) != owned.owner
    ):
        raise RehearsalError("container_ownership_unverified")
    return metadata


def inspect_owned(owned: OwnedContainer) -> dict:
    if (
        not re.fullmatch(r"[a-f0-9]{64}", owned.container_id)
        or not re.fullmatch(r"[a-f0-9]{16}", owned.owner)
        or owned.name != NAME_PREFIX + owned.owner
    ):
        raise RehearsalError("container_ownership_unverified")
    try:
        metadata = json.loads(docker_command(
            ["inspect", "--format", IDENTITY_FORMAT, owned.container_id],
            code="container_inspection_failed",
        ))
    except (ValueError, TypeError):
        raise RehearsalError("container_inspection_failed") from None
    return validate_owned_container(metadata, owned)


def remove_owned_container(owned: OwnedContainer) -> None:
    inspect_owned(owned)
    # No Compose down, prune, name glob, host filesystem removal or shared volume.
    docker_command(
        ["rm", "--force", "--volumes", owned.container_id], code="container_cleanup_failed",
    )


def _published_port(metadata: dict) -> int:
    ports = metadata.get("ports")
    binding = ports.get("5432/tcp") if isinstance(ports, dict) else None
    if (
        not isinstance(binding, list) or len(binding) != 1 or not isinstance(binding[0], dict)
        or binding[0].get("HostIp") != "127.0.0.1"
        or not str(binding[0].get("HostPort", "")).isdigit()
    ):
        raise RehearsalError("container_port_unverified")
    port = int(binding[0]["HostPort"])
    if not 1 <= port <= 65535:
        raise RehearsalError("container_port_unverified")
    return port


def artifact_directory(repo_root: Path, owner: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{16}", owner):
        raise RehearsalError("artifact_path_unverified")
    root = repo_root.resolve()
    target = (root / ".artifacts" / "backup-restore-rehearsal" / (NAME_PREFIX + owner)).resolve()
    if not target.is_relative_to(root) or target == root:
        raise RehearsalError("artifact_path_unverified")
    target.mkdir(parents=True, exist_ok=False)
    return target


def _connection_parameters(port: int, password: str, database: str) -> dict:
    if (
        type(port) is not int or not 1 <= port <= 65535
        or not re.fullmatch(r"[a-f0-9]{64}", password)
        or database not in DATABASES
    ):
        raise RehearsalError("database_target_unverified")
    return {
        "host": "127.0.0.1", "port": port, "user": DB_USER,
        "password": password, "dbname": database, "connect_timeout": 5,
        "autocommit": True,
    }


def connect_rehearsal(port: int, password: str, database: str):
    import psycopg

    parameters = _connection_parameters(port, password, database)
    try:
        connection = psycopg.connect(**parameters)
        if connection.execute("SELECT current_database(), current_user").fetchone() != (database, DB_USER):
            connection.close()
            raise RehearsalError("database_target_unverified")
        return connection
    except RehearsalError:
        raise
    except Exception:
        raise RehearsalError("database_connection_failed") from None


def _database_url(port: int, password: str, database: str) -> str:
    from sqlalchemy.engine import URL

    _connection_parameters(port, password, database)
    return URL.create(
        "postgresql+psycopg", username=DB_USER, password=password,
        host="127.0.0.1", port=port, database=database,
    ).render_as_string(hide_password=False)


def _alembic_config(repo_root: Path):
    from alembic.config import Config

    config = Config(str(repo_root / "alembic.ini"))
    config.set_main_option("script_location", str(repo_root / "migrations"))
    return config


def _migrate_source(repo_root: Path, port: int, password: str) -> str:
    from alembic import command
    from alembic.script import ScriptDirectory

    config = _alembic_config(repo_root)
    config.attributes["database_url"] = _database_url(port, password, SOURCE_DB)
    heads = ScriptDirectory.from_config(config).get_heads()
    if len(heads) != 1:
        raise RehearsalError("migration_head_ambiguous")
    # No app.db/Settings import: migrations/env.py honours this explicit URL.
    buffer = io.StringIO()
    sys.path.insert(0, str(repo_root))
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            command.upgrade(config, "head")
    except Exception:
        raise RehearsalError("migration_failed") from None
    finally:
        sys.path.pop(0)
        buffer.close()
    return heads[0]


def _assert_empty(connection) -> None:
    count = connection.execute(
        "SELECT count(*) FROM pg_tables WHERE schemaname = 'public'"
    ).fetchone()[0]
    if count != 0:
        raise RehearsalError("database_not_empty")


def _seed(connection, encryption_key: bytes) -> None:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from psycopg.types.json import Jsonb

    fixed_time = datetime(2026, 1, 1, 8)
    bodies = ("合成历史：纸铃留在仓库。\n此文仅用于备份演练。", "合成新稿：纸铃仍在仓库。\n版本二 😀。")
    with connection.transaction():
        for side in ("a", "b"):
            user, workspace, project = (f"rehearsal-{kind}-{side}" for kind in ("user", "workspace", "project"))
            connection.execute(
                "INSERT INTO users (id,email,display_name,password_hash,is_active,created_at) "
                "VALUES (%s,%s,%s,%s,true,%s)",
                (user, f"{side}@rehearsal.invalid", f"演练账户{side}", "!rehearsal-no-login", fixed_time),
            )
            connection.execute(
                "INSERT INTO workspaces (id,name,kind,created_at) VALUES (%s,%s,'personal',%s)",
                (workspace, f"演练工作区{side}", fixed_time),
            )
            connection.execute(
                "INSERT INTO workspace_members (id,workspace_id,user_id,role,created_at) VALUES (%s,%s,%s,'owner',%s)",
                (f"rehearsal-member-{side}", workspace, user, fixed_time),
            )
            connection.execute(
                "INSERT INTO projects (id,workspace_id,name,name_sort_key,description,created_at) VALUES (%s,%s,%s,%s,%s,%s)",
                (project, workspace, f"演练项目{side}", f"rehearsal-{side}".encode(), "合成资料，不可用于评测", fixed_time),
            )
            connection.execute(
                "INSERT INTO auth_sessions (id,user_id,token_hash,csrf_hash,created_at,expires_at,last_seen_at) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (f"rehearsal-session-{side}", user, hashlib.sha256(f"fake-session-{side}".encode()).hexdigest(),
                 hashlib.sha256(f"fake-csrf-{side}".encode()).hexdigest(), fixed_time, fixed_time + timedelta(days=7), fixed_time),
            )
            for version, body in enumerate(bodies, 1):
                connection.execute(
                    "INSERT INTO documents (id,project_id,name,content,version,active,created_at) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    (f"rehearsal-doc-{side}-v{version}", project, "chapter.md", body, version, version == 2, fixed_time),
                )
            run = f"rehearsal-run-{side}"
            connection.execute(
                "INSERT INTO analysis_runs (id,project_id,requested_by_user_id,status,created_at,started_at,completed_at,"
                "input_chars,prompt_tokens,completion_tokens,estimated_cost_usd,cancel_requested,batch_mode,sensitivity,batch_coverage) "
                "VALUES (%s,%s,%s,'completed',%s,%s,%s,%s,0,0,0,false,'draft_review','balanced',%s)",
                (run, project, user, fixed_time, fixed_time, fixed_time, len(bodies[0]),
                 Jsonb({"target_document_ids": [f"rehearsal-doc-{side}-v1"], "synthetic": True})),
            )
            connection.execute(
                "INSERT INTO analysis_run_inputs (id,run_id,document_id,document_name,document_version,content,content_sha256,ordinal) "
                "VALUES (%s,%s,%s,'chapter.md',1,%s,%s,0)",
                (f"rehearsal-input-{side}", run, f"rehearsal-doc-{side}-v1", bodies[0], hashlib.sha256(bodies[0].encode()).hexdigest()),
            )
            evidence = [{"document_id": f"rehearsal-doc-{side}-v1", "document_name": "chapter.md", "line_start": 1,
                         "line_end": 1, "text": bodies[0].splitlines()[0]}]
            issue = f"rehearsal-issue-{side}"
            connection.execute(
                "INSERT INTO issues (id,run_id,report_class,category,severity,confidence,title,explanation,evidence,suggestion,extra) "
                "VALUES (%s,%s,'formal','fact_conflict','low',0.5,%s,%s,%s,%s,%s)",
                (issue, run, "合成问题", "此记录仅验证备份关系", Jsonb(evidence), "无修改建议", Jsonb({"synthetic": True, "nested": {"版本": 1}})),
            )
            connection.execute(
                "INSERT INTO issue_feedback (id,issue_id,created_by_user_id,label,comment,created_at) VALUES (%s,%s,%s,'accepted',%s,%s)",
                (f"rehearsal-feedback-{side}", issue, user, "仅用于验证反馈恢复", fixed_time),
            )

        # A separate throwaway AEAD key proves ciphertext alone is insufficient.
        # This is not an app Provider envelope and is never a usable credential.
        nonce = secrets.token_bytes(12)
        ciphertext = AESGCM(encryption_key).encrypt(nonce, b"synthetic-no-provider-key", b"rehearsal-only")
        connection.execute(
            "INSERT INTO account_provider_configs (id,user_id,revision,base_url,model_name,endpoint_sha256,"
            "secret_ciphertext,secret_nonce,encryption_key_id,secret_schema_version,state,created_by_user_id,created_at) "
            "VALUES ('rehearsal-provider-a','rehearsal-user-a',1,%s,'synthetic-model',%s,%s,%s,'rehearsal-v1',1,'usable','rehearsal-user-a',%s)",
            ("https://provider.rehearsal.invalid/v1", hashlib.sha256(b"synthetic-endpoint").hexdigest(), ciphertext, nonce, fixed_time),
        )
        connection.execute(
            "INSERT INTO account_provider_bindings (user_id,current_config_id,lock_version,created_at,updated_at) "
            "VALUES ('rehearsal-user-a','rehearsal-provider-a',1,%s,%s)", (fixed_time, fixed_time),
        )


def fingerprint_rows(rows: list[Any]) -> str:
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def schema_entry_fingerprints(*, indexes, constraints, columns, triggers, extensions) -> dict:
    """Keep names and hashes only; never return SQL definitions/defaults."""
    return {
        "indexes": {f"{row[0]}.{row[1]}": fingerprint_rows([row]) for row in indexes},
        "constraints": {f"{row[0]}.{row[1]}": fingerprint_rows([row]) for row in constraints},
        "columns": {f"{row[0]}.{row[1]}": fingerprint_rows([row]) for row in columns},
        "triggers": {f"{row[0]}.{row[1]}": fingerprint_rows([row]) for row in triggers},
        "extensions": {row[0]: fingerprint_rows([row]) for row in extensions},
    }


def snapshot_difference(source: dict, restored: dict, *, limit: int = 20) -> dict:
    """Bounded safe diagnostics, without weakening exact snapshot comparison."""
    changed_sections = [key for key in sorted(set(source) | set(restored)) if source.get(key) != restored.get(key)]
    source_tables, restored_tables = source.get("tables", {}), restored.get("tables", {})
    table_changes = []
    for name in sorted(set(source_tables) | set(restored_tables)):
        before, after = source_tables.get(name, {}), restored_tables.get(name, {})
        if before != after:
            table_changes.append({
                "name": name, "source_row_count": before.get("row_count"),
                "restored_row_count": after.get("row_count"),
                "source_sha256": before.get("sha256"), "restored_sha256": after.get("sha256"),
            })
    source_schema, restored_schema = source.get("schema_entries", {}), restored.get("schema_entries", {})
    schema_changes = []
    for section in sorted(set(source_schema) | set(restored_schema)):
        before, after = source_schema.get(section, {}), restored_schema.get(section, {})
        for name in sorted(set(before) | set(after)):
            if before.get(name) != after.get(name):
                schema_changes.append({
                    "section": section, "name": name,
                    "source_sha256": before.get(name), "restored_sha256": after.get(name),
                })
    return {
        "changed_sections": changed_sections,
        "table_change_count": len(table_changes), "schema_change_count": len(schema_changes),
        "table_changes": table_changes[:limit], "schema_changes": schema_changes[:limit],
        "detail_limit": limit,
    }


def _snapshot(connection) -> dict:
    from psycopg import sql

    tables = [row[0] for row in connection.execute(
        "SELECT tablename FROM pg_tables WHERE schemaname = 'public' ORDER BY tablename"
    ).fetchall()]
    table_fingerprints = {}
    for table in tables:
        rows = [row[0] for row in connection.execute(sql.SQL(
            "SELECT to_jsonb(t) FROM {} AS t ORDER BY to_jsonb(t)::text"
        ).format(sql.Identifier("public", table))).fetchall()]
        table_fingerprints[table] = {"row_count": len(rows), "sha256": fingerprint_rows(rows)}
    indexes = connection.execute(
        "SELECT tablename,indexname,indexdef FROM pg_indexes WHERE schemaname='public' ORDER BY tablename,indexname"
    ).fetchall()
    constraints = connection.execute(
        "SELECT c.relname,k.conname,k.contype,pg_get_constraintdef(k.oid) "
        "FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname='public' ORDER BY c.relname,k.conname"
    ).fetchall()
    columns = connection.execute(
        "SELECT table_name,column_name,data_type,udt_name,is_nullable,column_default "
        "FROM information_schema.columns WHERE table_schema='public' ORDER BY table_name,ordinal_position"
    ).fetchall()
    triggers = connection.execute(
        "SELECT c.relname,t.tgname,pg_get_triggerdef(t.oid) FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid "
        "JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public' AND NOT t.tgisinternal ORDER BY c.relname,t.tgname"
    ).fetchall()
    extensions = connection.execute("SELECT extname,extversion FROM pg_extension ORDER BY extname").fetchall()
    return {
        "tables": table_fingerprints,
        "indexes_sha256": fingerprint_rows(indexes),
        "constraints_sha256": fingerprint_rows(constraints),
        "columns_sha256": fingerprint_rows(columns),
        "triggers_sha256": fingerprint_rows(triggers),
        "extensions_sha256": fingerprint_rows(extensions),
        "schema_entries": schema_entry_fingerprints(
            indexes=indexes, constraints=constraints, columns=columns,
            triggers=triggers, extensions=extensions,
        ),
    }


def _verify_data(connection, expected_head: str, encryption_key: bytes) -> dict:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    if connection.execute("SELECT version_num FROM alembic_version").fetchall() != [(expected_head,)]:
        raise RehearsalError("restored_migration_revision_mismatch")
    inputs = connection.execute(
        "SELECT i.content,i.content_sha256,i.document_version,d.version,d.content "
        "FROM analysis_run_inputs i JOIN documents d ON d.project_id IN ('rehearsal-project-a','rehearsal-project-b') "
        "AND d.name=i.document_name AND d.active WHERE i.run_id IN ('rehearsal-run-a','rehearsal-run-b') "
        "AND d.project_id=(SELECT project_id FROM analysis_runs WHERE id=i.run_id) ORDER BY i.id"
    ).fetchall()
    if len(inputs) != 2 or any(
        hashlib.sha256(body.encode()).hexdigest() != digest or version != 1 or current_version != 2 or body == current_body
        for body, digest, version, current_version, current_body in inputs
    ):
        raise RehearsalError("restored_frozen_input_mismatch")
    for side, other in (("a", "b"), ("b", "a")):
        workspace = f"rehearsal-workspace-{side}"
        for table, join, identifier in (
            ("documents", "JOIN projects p ON p.id=r.project_id", f"rehearsal-doc-{other}-v2"),
            ("analysis_runs", "JOIN projects p ON p.id=r.project_id", f"rehearsal-run-{other}"),
            ("issues", "JOIN analysis_runs a ON a.id=r.run_id JOIN projects p ON p.id=a.project_id", f"rehearsal-issue-{other}"),
            ("issue_feedback", "JOIN issues i ON i.id=r.issue_id JOIN analysis_runs a ON a.id=i.run_id JOIN projects p ON p.id=a.project_id", f"rehearsal-feedback-{other}"),
        ):
            # Identifiers below are a fixed internal table allowlist, not user input.
            if connection.execute(f"SELECT r.id FROM {table} r {join} WHERE p.workspace_id=%s AND r.id=%s", (workspace, identifier)).fetchall():
                raise RehearsalError("restored_workspace_isolation_failed")
        own = connection.execute(
            "SELECT r.id FROM issues r JOIN analysis_runs a ON a.id=r.run_id JOIN projects p ON p.id=a.project_id "
            "WHERE p.workspace_id=%s", (workspace,),
        ).fetchall()
        if own != [(f"rehearsal-issue-{side}",)]:
            raise RehearsalError("restored_workspace_isolation_failed")
    ciphertext, nonce = connection.execute(
        "SELECT secret_ciphertext,secret_nonce FROM account_provider_configs WHERE id='rehearsal-provider-a'"
    ).fetchone()
    if AESGCM(encryption_key).decrypt(bytes(nonce), bytes(ciphertext), b"rehearsal-only") != b"synthetic-no-provider-key":
        raise RehearsalError("restored_synthetic_ciphertext_mismatch")
    return {
        "migration_head": expected_head, "frozen_input_hashes_verified": 2,
        "workspace_query_isolation_verified": True,
        "synthetic_ciphertext_decrypted_with_separate_in_memory_key": True,
        "encryption_key_persisted": False,
    }


def _verify_constraints(connection) -> list[str]:
    import psycopg

    probes = (
        ("active_document_unique", "23505", "INSERT INTO documents (id,project_id,name,content,version,active,created_at) VALUES ('rehearsal-probe-active','rehearsal-project-a','CHAPTER.md','probe',3,true,now())"),
        ("document_version_unique", "23505", "INSERT INTO documents (id,project_id,name,content,version,active,created_at) VALUES ('rehearsal-probe-version','rehearsal-project-a','CHAPTER.md','probe',1,false,now())"),
        ("feedback_foreign_key", "23503", "INSERT INTO issue_feedback (id,issue_id,label,comment,created_at) VALUES ('rehearsal-probe-feedback','rehearsal-missing-issue','accepted','probe',now())"),
        ("membership_unique", "23505", "INSERT INTO workspace_members (id,workspace_id,user_id,role,created_at) VALUES ('rehearsal-probe-member','rehearsal-workspace-a','rehearsal-user-a','owner',now())"),
        ("project_workspace_foreign_key", "23503", "INSERT INTO projects (id,workspace_id,name,name_sort_key,description,created_at) VALUES ('rehearsal-probe-project','rehearsal-missing-workspace','probe','x','',now())"),
        ("provider_revision_check", "23514", "UPDATE account_provider_configs SET revision=0 WHERE id='rehearsal-provider-a'"),
    )
    passed = []
    for name, code, statement in probes:
        try:
            with connection.transaction():
                connection.execute(statement)
                raise RehearsalError("restored_constraint_not_enforced")
        except psycopg.Error as error:
            if error.sqlstate != code:
                raise RehearsalError("restored_constraint_probe_failed") from None
            passed.append(name)
    names = {row[0] for row in connection.execute(
        "SELECT indexname FROM pg_indexes WHERE schemaname='public'"
    ).fetchall()}
    if not {
        "uq_documents_project_lower_name_active", "uq_documents_project_lower_name_version",
        "ix_analysis_runs_project_created_id", "ix_projects_workspace_name_sort_id",
    }.issubset(names):
        raise RehearsalError("restored_required_index_missing")
    return passed


def _wait_ready(owned: OwnedContainer, password: str) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            docker_command(
                ["exec", owned.container_id, "pg_isready", "--host", "127.0.0.1", "--username", DB_USER, "--dbname", SOURCE_DB],
                password=password, code="postgres_not_ready", timeout=5,
            )
            return
        except RehearsalError:
            time.sleep(1)
    raise RehearsalError("postgres_not_ready")


def run_rehearsal(repo_root: Path = REPO_ROOT) -> dict:
    owner = secrets.token_hex(8)
    password = secrets.token_hex(32)
    artifact = artifact_directory(repo_root, owner)
    owned = None
    report: dict[str, Any] = {
        "schema_version": 1, "purpose": PURPOSE,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "failed", "stage": "docker_preflight", "image": IMAGE,
        "synthetic_data_only": True, "real_credentials_or_user_database_used": False,
        "provider_or_model_calls": 0, "container_cleanup": "not_created",
    }
    try:
        _require_local_docker_transport()
        docker_command(["version", "--format", "{{.Server.Version}}"], code="docker_unavailable", timeout=10)
        image_id = docker_command(["image", "inspect", IMAGE, "--format", "{{.Id}}"], code="image_not_available", timeout=10).decode().strip()
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", image_id):
            raise RehearsalError("image_identity_unverified")
        report["image_id"] = image_id
        report["stage"] = "create_disposable_container"
        report["disposable_container_name"] = NAME_PREFIX + owner
        report["container_cleanup"] = "creation_attempted_without_verified_id"
        container_id = docker_command([
            "create", "--pull=never", "--name", NAME_PREFIX + owner,
            "--label", f"{PURPOSE_LABEL}={PURPOSE}", "--label", f"{OWNER_LABEL}={owner}",
            "--publish", "127.0.0.1::5432", "--env", f"POSTGRES_USER={DB_USER}",
            "--env", f"POSTGRES_DB={SOURCE_DB}", "--env", "POSTGRES_PASSWORD", image_id,
        ], password=password, code="container_creation_failed").decode().strip()
        owned = OwnedContainer(container_id, NAME_PREFIX + owner, owner)
        report["container_cleanup"] = "owned_container_pending_cleanup"
        inspect_owned(owned)
        docker_command(["start", container_id], code="container_start_failed")
        _wait_ready(owned, password)
        port = _published_port(inspect_owned(owned))
        report["stage"] = "source_migration"
        with connect_rehearsal(port, password, SOURCE_DB) as source:
            _assert_empty(source)
        head = _migrate_source(repo_root, port, password)
        key = secrets.token_bytes(32)
        report["stage"] = "synthetic_seed"
        with connect_rehearsal(port, password, SOURCE_DB) as source:
            _seed(source, key)
            _verify_data(source, head, key)
            source_snapshot = _snapshot(source)
            report["source_fingerprints"] = source_snapshot
            # No DROP/clean operation: the restored target is newly created here.
            source.execute(f"CREATE DATABASE {RESTORED_DB}")
        report["stage"] = "dump"
        inspect_owned(owned)
        dump = docker_command([
            "exec", "--env", "PGPASSWORD", container_id, "pg_dump", "--username", DB_USER,
            "--dbname", SOURCE_DB, "--format=custom", "--no-owner", "--no-acl",
        ], password=password, code="dump_failed", timeout=120)
        if not dump.startswith(b"PGDMP") or len(dump) > MAX_DUMP_BYTES:
            raise RehearsalError("dump_format_or_size_invalid")
        dump_path = artifact / "synthetic.pgcustom"
        with dump_path.open("xb") as output:
            output.write(dump)
        report["dump_sha256"] = hashlib.sha256(dump).hexdigest()
        report["dump_bytes"] = len(dump)
        report["stage"] = "restore_empty_database"
        with connect_rehearsal(port, password, RESTORED_DB) as restored:
            _assert_empty(restored)
        inspect_owned(owned)
        docker_command([
            "exec", "--interactive", "--env", "PGPASSWORD", container_id, "pg_restore",
            "--username", DB_USER, "--dbname", RESTORED_DB, "--exit-on-error", "--no-owner", "--no-acl",
        ], password=password, input_bytes=dump, code="restore_failed", timeout=120)
        report["stage"] = "restored_verification"
        with connect_rehearsal(port, password, RESTORED_DB) as restored:
            restored_snapshot = _snapshot(restored)
            report["restored_fingerprints"] = restored_snapshot
            if restored_snapshot != source_snapshot:
                report["snapshot_difference"] = snapshot_difference(source_snapshot, restored_snapshot)
                raise RehearsalError("restored_schema_or_data_mismatch")
            report["verification"] = _verify_data(restored, head, key)
            report["verification"]["constraint_probes"] = _verify_constraints(restored)
            after_probes = _snapshot(restored)
            if after_probes != source_snapshot:
                report["restored_fingerprints"] = after_probes
                report["snapshot_difference"] = snapshot_difference(source_snapshot, after_probes)
                raise RehearsalError("constraint_probe_modified_restored_data")
        report["source_and_restored_fingerprints"] = source_snapshot
        report["status"] = "passed"
        report["stage"] = "complete"
    except RehearsalError as error:
        report["failure_code"] = str(error)
    except Exception:
        # Database/import/subprocess exceptions can contain passwords or URLs.
        report["failure_code"] = "rehearsal_internal_error"
    finally:
        if owned is not None:
            try:
                remove_owned_container(owned)
                report["container_cleanup"] = "removed_owned_container_and_its_anonymous_volumes"
            except RehearsalError as error:
                report["container_cleanup"] = str(error)
                report["status"] = "failed"
                report.setdefault("failure_code", "container_cleanup_incomplete")
        report["completed_at"] = datetime.now(timezone.utc).isoformat()
        with (artifact / "report.json").open("x", encoding="utf-8") as output:
            json.dump(report, output, ensure_ascii=False, indent=2)
    summary = {
        "status": report["status"], "failure_code": report.get("failure_code"),
        "stage": report["stage"], "container_cleanup": report["container_cleanup"],
        "image_id": report.get("image_id"), "report": str(artifact / "report.json"),
    }
    if report.get("snapshot_difference") is not None:
        summary["snapshot_difference"] = report["snapshot_difference"]
    if report["status"] == "passed":
        summary.update({
            "migration_head": report["verification"]["migration_head"],
            "table_count": len(report["source_and_restored_fingerprints"]["tables"]),
            "constraint_probe_count": len(report["verification"]["constraint_probes"]),
            "workspace_query_isolation_verified": report["verification"]["workspace_query_isolation_verified"],
            "dump_sha256": report["dump_sha256"], "dump_bytes": report["dump_bytes"],
        })
    return summary


def main() -> int:
    try:
        result = run_rehearsal()
    except Exception:
        result = {"status": "failed", "failure_code": "rehearsal_artifact_initialization_failed"}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
