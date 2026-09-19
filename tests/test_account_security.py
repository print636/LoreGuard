from __future__ import annotations

import unittest
from contextlib import contextmanager
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql

from app.auth import (
    CSRF_COOKIE,
    _locked_user_by_email_query,
    _locked_user_by_id_query,
)
from app.main import app, settings, write_limiter


PASSWORD = "correct horse battery staple"
NEW_PASSWORD = "violet constellations stay coherent"


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


def register(client: TestClient, *, display_name: str = "星轨作者") -> str:
    email = f"author-{uuid4().hex}@example.com"
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": PASSWORD,
            "display_name": display_name,
        },
    )
    assert response.status_code == 201, response.text
    return email


def login(client: TestClient, email: str, password: str = PASSWORD) -> str:
    response = client.post(
        "/api/v1/auth/login",
        json={"email": email, "password": password},
    )
    assert response.status_code == 200, response.text
    csrf = client.cookies.get(CSRF_COOKIE)
    assert csrf
    return csrf


def csrf_headers(client: TestClient) -> dict[str, str]:
    csrf = client.cookies.get(CSRF_COOKIE)
    assert csrf
    return {"X-CSRF-Token": csrf}


class AccountSecurityApiTests(unittest.TestCase):
    def test_password_paths_compile_to_postgresql_row_locks(self):
        dialect = postgresql.dialect()
        statements = (
            _locked_user_by_email_query("author@example.com"),
            _locked_user_by_id_query(str(uuid4())),
        )
        for statement in statements:
            with self.subTest(statement=statement):
                sql = str(statement.compile(dialect=dialect))
                self.assertIn(" FOR UPDATE", sql)

    def test_session_list_is_current_user_only_and_marks_current_session(self):
        with required_auth_settings(), TestClient(app) as first, TestClient(
            app
        ) as second, TestClient(app) as other_user:
            email = register(first)
            login(second, email)
            register(other_user, display_name="另一位作者")

            first_sessions = first.get("/api/v1/auth/sessions")
            second_sessions = second.get("/api/v1/auth/sessions")
            other_sessions = other_user.get("/api/v1/auth/sessions")

            self.assertEqual(200, first_sessions.status_code, first_sessions.text)
            self.assertEqual(200, second_sessions.status_code, second_sessions.text)
            self.assertEqual(200, other_sessions.status_code, other_sessions.text)
            first_rows = first_sessions.json()["sessions"]
            second_rows = second_sessions.json()["sessions"]
            other_rows = other_sessions.json()["sessions"]
            self.assertEqual(2, len(first_rows))
            self.assertEqual(2, len(second_rows))
            self.assertEqual(1, sum(row["current"] for row in first_rows))
            self.assertEqual(1, sum(row["current"] for row in second_rows))
            self.assertEqual(1, len(other_rows))
            self.assertTrue(other_rows[0]["current"])
            self.assertTrue(
                {"id", "created_at", "last_seen_at", "expires_at", "current"}
                <= first_rows[0].keys()
            )
            self.assertTrue(
                {row["id"] for row in first_rows}.isdisjoint(
                    {row["id"] for row in other_rows}
                )
            )

    def test_revoke_others_requires_csrf_and_preserves_current_session(self):
        with required_auth_settings(), TestClient(app) as current, TestClient(
            app
        ) as other_session:
            email = register(current)
            login(other_session, email)

            rejected = current.post("/api/v1/auth/sessions/revoke-others")
            self.assertEqual(403, rejected.status_code)

            revoked = current.post(
                "/api/v1/auth/sessions/revoke-others",
                headers=csrf_headers(current),
            )
            self.assertEqual(200, revoked.status_code, revoked.text)
            self.assertEqual({"revoked_sessions": 1}, revoked.json())
            self.assertEqual(200, current.get("/api/v1/auth/me").status_code)
            self.assertEqual(401, other_session.get("/api/v1/auth/me").status_code)
            remaining = current.get("/api/v1/auth/sessions").json()["sessions"]
            self.assertEqual(1, len(remaining))
            self.assertTrue(remaining[0]["current"])

    def test_delete_session_hides_other_accounts_and_rejects_current_session(self):
        with required_auth_settings(), TestClient(app) as current, TestClient(
            app
        ) as other_session, TestClient(app) as other_user:
            email = register(current)
            login(other_session, email)
            register(other_user, display_name="另一位作者")

            sessions = current.get("/api/v1/auth/sessions").json()["sessions"]
            current_id = next(row["id"] for row in sessions if row["current"])
            other_session_id = next(row["id"] for row in sessions if not row["current"])
            foreign_id = other_user.get("/api/v1/auth/sessions").json()["sessions"][0]["id"]

            self.assertEqual(
                403,
                current.delete(
                    f"/api/v1/auth/sessions/{other_session_id}"
                ).status_code,
            )
            current_rejected = current.delete(
                f"/api/v1/auth/sessions/{current_id}",
                headers=csrf_headers(current),
            )
            self.assertEqual(409, current_rejected.status_code)
            self.assertIn("退出登录", current_rejected.json()["detail"])

            for hidden_id in (foreign_id, str(uuid4())):
                hidden = current.delete(
                    f"/api/v1/auth/sessions/{hidden_id}",
                    headers=csrf_headers(current),
                )
                self.assertEqual(404, hidden.status_code)
                self.assertEqual("会话不存在", hidden.json()["detail"])

            revoked = current.delete(
                f"/api/v1/auth/sessions/{other_session_id}",
                headers=csrf_headers(current),
            )
            self.assertEqual(200, revoked.status_code, revoked.text)
            self.assertEqual({"revoked_sessions": 1}, revoked.json())
            self.assertEqual(401, other_session.get("/api/v1/auth/me").status_code)

    def test_change_password_validates_current_password_and_revokes_other_sessions(self):
        with required_auth_settings(), TestClient(app) as current, TestClient(
            app
        ) as other_session:
            email = register(current)
            login(other_session, email)

            no_csrf = current.post(
                "/api/v1/auth/password",
                json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
            )
            self.assertEqual(403, no_csrf.status_code)

            invalid_new = current.post(
                "/api/v1/auth/password",
                headers=csrf_headers(current),
                json={"current_password": PASSWORD, "new_password": "short"},
            )
            self.assertEqual(422, invalid_new.status_code)

            wrong_current = current.post(
                "/api/v1/auth/password",
                headers=csrf_headers(current),
                json={
                    "current_password": "a convincing but wrong password",
                    "new_password": NEW_PASSWORD,
                },
            )
            self.assertEqual(400, wrong_current.status_code)
            self.assertEqual("当前密码错误", wrong_current.json()["detail"])
            self.assertEqual(200, other_session.get("/api/v1/auth/me").status_code)

            unchanged = current.post(
                "/api/v1/auth/password",
                headers=csrf_headers(current),
                json={"current_password": PASSWORD, "new_password": PASSWORD},
            )
            self.assertEqual(400, unchanged.status_code)
            self.assertEqual(
                "新密码不能与当前密码相同", unchanged.json()["detail"]
            )
            self.assertEqual(200, other_session.get("/api/v1/auth/me").status_code)

            changed = current.post(
                "/api/v1/auth/password",
                headers=csrf_headers(current),
                json={"current_password": PASSWORD, "new_password": NEW_PASSWORD},
            )
            self.assertEqual(200, changed.status_code, changed.text)
            self.assertEqual({"revoked_sessions": 1}, changed.json())
            self.assertEqual(200, current.get("/api/v1/auth/me").status_code)
            self.assertEqual(401, other_session.get("/api/v1/auth/me").status_code)

            logout = current.post(
                "/api/v1/auth/logout", headers=csrf_headers(current)
            )
            self.assertEqual(204, logout.status_code)
            old_login = current.post(
                "/api/v1/auth/login",
                json={"email": email, "password": PASSWORD},
            )
            self.assertEqual(401, old_login.status_code)
            new_login = current.post(
                "/api/v1/auth/login",
                json={"email": email, "password": NEW_PASSWORD},
            )
            self.assertEqual(200, new_login.status_code, new_login.text)

    def test_account_security_endpoints_explicitly_reject_anonymous_mode(self):
        previous = settings.auth_mode
        settings.auth_mode = "anonymous"
        try:
            with TestClient(app) as client:
                responses = [
                    client.get("/api/v1/auth/sessions"),
                    client.post("/api/v1/auth/sessions/revoke-others"),
                    client.delete(f"/api/v1/auth/sessions/{uuid4()}"),
                    client.post(
                        "/api/v1/auth/password",
                        json={
                            "current_password": PASSWORD,
                            "new_password": NEW_PASSWORD,
                        },
                    ),
                ]
                for response in responses:
                    self.assertEqual(409, response.status_code, response.text)
                    self.assertEqual(
                        "本地匿名模式不支持账户安全操作",
                        response.json()["detail"],
                    )
        finally:
            settings.auth_mode = previous


if __name__ == "__main__":
    unittest.main()
