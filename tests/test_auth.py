from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import sleep
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import sessionmaker

from app.auth import SESSION_COOKIE, _ensure_local_identity, _hash_password
from app.config import Settings
from app.db import (
    AuthSessionRow,
    Base,
    LOCAL_USER_ID,
    LOCAL_WORKSPACE_ID,
    SessionLocal,
    UserRow,
    WorkspaceMemberRow,
    WorkspaceRow,
    enable_sqlite_foreign_keys,
)
from app.main import app, settings, write_limiter


@contextmanager
def required_auth_settings():
    fields = {
        "auth_mode": settings.auth_mode,
        "auth_secret_key": settings.auth_secret_key,
        "auth_cookie_secure": settings.auth_cookie_secure,
        "auth_cookie_samesite": settings.auth_cookie_samesite,
    }
    settings.auth_mode = "required"
    settings.auth_secret_key = "test-only-auth-secret-key-32-bytes-minimum"
    settings.auth_cookie_secure = False
    settings.auth_cookie_samesite = "lax"
    write_limiter.events.clear()
    try:
        yield
    finally:
        for name, value in fields.items():
            setattr(settings, name, value)
        write_limiter.events.clear()


class AuthConfigurationTests(unittest.TestCase):
    def test_required_auth_fails_closed_without_secret(self):
        with self.assertRaisesRegex(ValidationError, "AUTH_SECRET_KEY"):
            Settings(auth_mode="required", auth_secret_key="")

    def test_production_rejects_insecure_cookie(self):
        with self.assertRaisesRegex(ValidationError, "must be Secure"):
            Settings(
                deployment_environment="production",
                auth_mode="required",
                auth_secret_key="x" * 32,
                auth_cookie_secure=False,
                cors_allowed_origins="https://loreguard.example",
            )

    def test_production_rejects_anonymous_mode(self):
        with self.assertRaisesRegex(ValidationError, "AUTH_MODE=required"):
            Settings(
                deployment_environment="production",
                auth_mode="anonymous",
                auth_cookie_secure=True,
                cors_allowed_origins="https://loreguard.example",
            )

    def test_production_rejects_non_postgresql_database(self):
        for database_url in (
            "sqlite:///./loreguard.db",
            "mysql+pymysql://user:password@database/loreguard",
            "postgresql://user:password@database/loreguard",
            "postgresql+asyncpg://user:password@database/loreguard",
        ):
            with self.subTest(database_url=database_url), self.assertRaisesRegex(
                ValidationError, "DATABASE_URL must use postgresql"
            ):
                Settings(
                    deployment_environment="production",
                    auth_mode="required",
                    auth_secret_key="x" * 32,
                    auth_cookie_secure=True,
                    cors_allowed_origins="https://loreguard.example",
                    database_url=database_url,
                )

    def test_production_accepts_installed_psycopg_database_scheme(self):
        database_url = "postgresql+psycopg://user:password@database/loreguard"
        configured = Settings(
            deployment_environment="production",
            auth_mode="required",
            auth_secret_key="x" * 32,
            auth_cookie_secure=True,
            cors_allowed_origins="https://loreguard.example",
            database_url=database_url,
            account_model_active_key_id="prod-v1",
            account_model_keyring_json=(
                '{"keys":{"prod-v1":'
                '"a2tra2tra2tra2tra2tra2tra2tra2tra2tra2tra2s="}}'
            ),
        )
        self.assertEqual(database_url, configured.database_url)

    def test_credentialed_cors_rejects_wildcard(self):
        with self.assertRaisesRegex(ValidationError, "invalid exact CORS origin"):
            Settings(cors_allowed_origins="*")


class AuthApiTests(unittest.TestCase):
    def register(self, client: TestClient, *, origin: str | None = None):
        email = f"author-{uuid4().hex}@example.com"
        headers = {"Origin": origin} if origin else {}
        response = client.post(
            "/api/v1/auth/register",
            headers=headers,
            json={
                "email": email.upper(),
                "password": "correct horse battery staple",
                "display_name": "星轨作者",
            },
        )
        return email, response

    def test_register_creates_personal_workspace_and_hashed_session(self):
        with required_auth_settings(), TestClient(app) as client:
            email, response = self.register(client, origin="http://localhost:5173")

            self.assertEqual(201, response.status_code, response.text)
            payload = response.json()
            self.assertEqual(email, payload["user"]["email"])
            self.assertEqual("owner", payload["workspace"]["role"])
            self.assertEqual("personal", payload["workspace"]["kind"])
            raw_session = client.cookies.get(SESSION_COOKIE)
            self.assertTrue(raw_session)
            set_cookie_headers = response.headers.get_list("set-cookie")
            session_cookie = next(
                item for item in set_cookie_headers if item.startswith(f"{SESSION_COOKIE}=")
            )
            self.assertIn("HttpOnly", session_cookie)
            self.assertIn("SameSite=lax", session_cookie)

            with SessionLocal() as db:
                user = db.execute(
                    select(UserRow).where(UserRow.email == email)
                ).scalar_one()
                self.assertTrue(user.password_hash.startswith("$argon2id$"))
                membership = db.execute(
                    select(WorkspaceMemberRow).where(
                        WorkspaceMemberRow.user_id == user.id
                    )
                ).scalar_one()
                workspace = db.get(WorkspaceRow, membership.workspace_id)
                self.assertIsNotNone(workspace)
                self.assertEqual("owner", membership.role)
                session = db.execute(
                    select(AuthSessionRow).where(AuthSessionRow.user_id == user.id)
                ).scalar_one()
                self.assertEqual(64, len(session.token_hash))
                self.assertNotEqual(raw_session, session.token_hash)
                self.assertNotIn(raw_session, session.token_hash)

    def test_me_and_logout_require_valid_session_and_csrf(self):
        with required_auth_settings(), TestClient(app) as client:
            _, response = self.register(client)
            self.assertEqual(201, response.status_code, response.text)
            csrf = response.headers["X-CSRF-Token"]

            me = client.get("/api/v1/auth/me")
            self.assertEqual(200, me.status_code)
            self.assertEqual("required", me.json()["mode"])

            rejected = client.post("/api/v1/auth/logout")
            self.assertEqual(403, rejected.status_code)
            accepted = client.post(
                "/api/v1/auth/logout", headers={"X-CSRF-Token": csrf}
            )
            self.assertEqual(204, accepted.status_code)
            self.assertEqual(401, client.get("/api/v1/auth/me").status_code)

    def test_cross_origin_registration_is_rejected_before_handler(self):
        with required_auth_settings(), TestClient(app) as client:
            _, response = self.register(client, origin="https://evil.example")
            self.assertEqual(403, response.status_code)
            self.assertEqual("请求来源不受信任", response.json()["detail"])

    def test_login_error_does_not_reveal_account_existence(self):
        with required_auth_settings(), TestClient(app) as client:
            email, registered = self.register(client)
            self.assertEqual(201, registered.status_code, registered.text)
            csrf = registered.headers["X-CSRF-Token"]
            self.assertEqual(
                204,
                client.post(
                    "/api/v1/auth/logout", headers={"X-CSRF-Token": csrf}
                ).status_code,
            )
            wrong_password = client.post(
                "/api/v1/auth/login",
                json={"email": email, "password": "definitely-wrong"},
            )
            missing_account = client.post(
                "/api/v1/auth/login",
                json={
                    "email": f"missing-{uuid4().hex}@example.com",
                    "password": "definitely-wrong",
                },
            )
            self.assertEqual(401, wrong_password.status_code)
            self.assertEqual(wrong_password.json(), missing_account.json())

    def test_anonymous_mode_has_durable_local_workspace_context(self):
        previous = settings.auth_mode
        settings.auth_mode = "anonymous"
        try:
            with TestClient(app) as client:
                response = client.get("/api/v1/auth/me")
                self.assertEqual(200, response.status_code, response.text)
                self.assertEqual("anonymous", response.json()["mode"])
                self.assertEqual("owner", response.json()["workspace"]["role"])
        finally:
            settings.auth_mode = previous


class AuthConcurrencyTests(unittest.TestCase):
    def test_password_hash_work_is_bounded_inside_process(self):
        lock = Lock()
        active = 0
        peak = 0

        class SlowHasher:
            def hash(self, password: str) -> str:
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                sleep(0.02)
                with lock:
                    active -= 1
                return f"hashed:{password}"

        with patch("app.auth._PASSWORD_HASHER", SlowHasher()):
            with ThreadPoolExecutor(max_workers=10) as pool:
                results = list(pool.map(_hash_password, map(str, range(10))))
        self.assertEqual(10, len(results))
        self.assertLessEqual(peak, 2)

    def test_concurrent_local_identity_initialization_is_idempotent(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "local-identity.db"
            engine = create_engine(
                f"sqlite:///{path.as_posix()}",
                connect_args={"check_same_thread": False},
            )
            enable_sqlite_foreign_keys(engine)
            Base.metadata.create_all(engine)
            factory = sessionmaker(bind=engine, expire_on_commit=False)
            with factory() as db:
                db.execute(delete(WorkspaceMemberRow))
                db.execute(delete(WorkspaceRow).where(WorkspaceRow.id == LOCAL_WORKSPACE_ID))
                db.execute(delete(UserRow).where(UserRow.id == LOCAL_USER_ID))
                db.commit()

            def initialize(_index: int) -> tuple[str, str]:
                with factory() as db:
                    user, workspace, _role = _ensure_local_identity(db)
                    return user.id, workspace.id

            with ThreadPoolExecutor(max_workers=8) as pool:
                identities = list(pool.map(initialize, range(8)))
            self.assertEqual(
                {(LOCAL_USER_ID, LOCAL_WORKSPACE_ID)}, set(identities)
            )
            with factory() as db:
                self.assertEqual(
                    1,
                    db.scalar(
                        select(func.count())
                        .select_from(UserRow)
                        .where(UserRow.id == LOCAL_USER_ID)
                    ),
                )
                self.assertEqual(
                    1,
                    db.scalar(
                        select(func.count())
                        .select_from(WorkspaceMemberRow)
                        .where(
                            WorkspaceMemberRow.user_id == LOCAL_USER_ID,
                            WorkspaceMemberRow.workspace_id == LOCAL_WORKSPACE_ID,
                        )
                    ),
                )
            engine.dispose()


if __name__ == "__main__":
    unittest.main()
