from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import dataclass
from datetime import timedelta
from threading import BoundedSemaphore, Lock

from argon2 import PasswordHasher, Type
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from .config import Settings, get_settings
from .db import (
    AuthSessionRow,
    LOCAL_USER_ID,
    LOCAL_WORKSPACE_ID,
    SessionLocal,
    UserRow,
    WorkspaceMemberRow,
    WorkspaceRow,
)
from .time_utils import utc_now_naive


SESSION_COOKIE = "loreguard_session"
CSRF_COOKIE = "loreguard_csrf"
_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PASSWORD_HASHER = PasswordHasher(
    time_cost=3,
    memory_cost=65_536,
    parallelism=4,
    hash_len=32,
    salt_len=16,
    type=Type.ID,
)
_DUMMY_PASSWORD_HASH = _PASSWORD_HASHER.hash("not-a-real-account-password")
# Each Argon2id operation uses roughly 64 MiB with the parameters above.  A
# small process-local gate prevents a burst of login/register requests from
# multiplying that allocation across the entire ASGI thread pool.
_PASSWORD_OPERATION_SLOTS = BoundedSemaphore(value=2)
_LOCAL_IDENTITY_LOCK = Lock()


@dataclass(frozen=True)
class AuthContext:
    user_id: str
    workspace_id: str
    role: str
    anonymous: bool
    session_id: str | None = None
    csrf_hash: str | None = None


class RegisterIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=10, max_length=128)
    display_name: str = Field(min_length=1, max_length=80)


class LoginIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=1, max_length=128)


class ChangePasswordIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=10, max_length=128)


class ExactOriginMiddleware(BaseHTTPMiddleware):
    """Reject browser state changes from origins outside the exact allowlist.

    Requests without ``Origin`` remain available to non-browser clients. CSRF
    protection for an authenticated browser session is a separate, mandatory
    double-submit check on protected write endpoints.
    """

    def __init__(self, app, allowed_origins: list[str]):
        super().__init__(app)
        self.allowed_origins = frozenset(origin.rstrip("/") for origin in allowed_origins)

    async def dispatch(self, request: Request, call_next):
        if request.method in _UNSAFE_METHODS:
            origin = request.headers.get("origin")
            if origin and origin.rstrip("/") not in self.allowed_origins:
                return JSONResponse(status_code=403, content={"detail": "请求来源不受信任"})
        return await call_next(request)


router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


def normalize_email(value: str) -> str:
    email = value.strip().casefold()
    if len(email) > 320 or not _EMAIL_PATTERN.fullmatch(email):
        raise HTTPException(status_code=422, detail="邮箱格式无效")
    return email


def _digest_token(token: str, settings: Settings) -> str:
    if len(settings.auth_secret_key) < 32:
        # Settings validation normally makes this unreachable. Keeping the
        # guard here prevents a test/runtime mutation from failing open.
        raise RuntimeError("authentication secret is not configured")
    return hmac.new(
        settings.auth_secret_key.encode("utf-8"),
        token.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _serialize_identity(user: UserRow, workspace: WorkspaceRow, role: str) -> dict:
    return {
        "user": {
            "id": user.id,
            "email": user.email,
            "display_name": user.display_name,
        },
        "workspace": {
            "id": workspace.id,
            "name": workspace.name,
            "kind": workspace.kind,
            "role": role,
        },
    }


def _personal_workspace(db, user_id: str) -> tuple[WorkspaceRow, str] | None:
    row = db.execute(
        select(WorkspaceRow, WorkspaceMemberRow.role)
        .join(
            WorkspaceMemberRow,
            WorkspaceMemberRow.workspace_id == WorkspaceRow.id,
        )
        .where(
            WorkspaceMemberRow.user_id == user_id,
            WorkspaceRow.kind == "personal",
        )
        .order_by(WorkspaceRow.created_at, WorkspaceRow.id)
    ).first()
    return (row[0], row[1]) if row else None


def _ensure_local_identity(db) -> tuple[UserRow, WorkspaceRow, str]:
    """Return the fixed anonymous identity without a first-request race.

    Alembic normally seeds these rows.  The retry remains important for a
    manually emptied local database: two initial requests may both observe
    absence, but a fixed-primary-key conflict must never escape as a 500.
    """

    with _LOCAL_IDENTITY_LOCK:
        return _ensure_local_identity_locked(db)


def _ensure_local_identity_locked(db) -> tuple[UserRow, WorkspaceRow, str]:
    for _attempt in range(2):
        user = db.get(UserRow, LOCAL_USER_ID)
        workspace = db.get(WorkspaceRow, LOCAL_WORKSPACE_ID)
        membership = db.execute(
            select(WorkspaceMemberRow).where(
                WorkspaceMemberRow.workspace_id == LOCAL_WORKSPACE_ID,
                WorkspaceMemberRow.user_id == LOCAL_USER_ID,
            )
        ).scalar_one_or_none()
        if user is not None and workspace is not None and membership is not None:
            return user, workspace, membership.role
        try:
            if user is None:
                db.add(
                    UserRow(
                        id=LOCAL_USER_ID,
                        email="local@loreguard.invalid",
                        display_name="本地体验用户",
                        password_hash="!anonymous-local-account",
                    )
                )
            if workspace is None:
                db.add(
                    WorkspaceRow(
                        id=LOCAL_WORKSPACE_ID,
                        name="本地工作区",
                        kind="personal",
                    )
                )
            # The ORM models intentionally do not expose auth relationships;
            # explicitly flush parents before their membership foreign keys.
            db.flush()
            if membership is None:
                db.add(
                    WorkspaceMemberRow(
                        workspace_id=LOCAL_WORKSPACE_ID,
                        user_id=LOCAL_USER_ID,
                        role="owner",
                    )
                )
            db.commit()
        except IntegrityError:
            db.rollback()
            db.expire_all()
            continue
    user = db.get(UserRow, LOCAL_USER_ID)
    workspace = db.get(WorkspaceRow, LOCAL_WORKSPACE_ID)
    membership = db.execute(
        select(WorkspaceMemberRow).where(
            WorkspaceMemberRow.workspace_id == LOCAL_WORKSPACE_ID,
            WorkspaceMemberRow.user_id == LOCAL_USER_ID,
        )
    ).scalar_one_or_none()
    if user is None or workspace is None or membership is None:
        raise RuntimeError("local anonymous identity could not be initialized")
    return user, workspace, membership.role


def _hash_password(password: str) -> str:
    with _PASSWORD_OPERATION_SLOTS:
        return _PASSWORD_HASHER.hash(password)


def _verify_password(password_hash: str, password: str) -> None:
    with _PASSWORD_OPERATION_SLOTS:
        _PASSWORD_HASHER.verify(password_hash, password)


def _locked_user_by_email_query(email: str):
    """Build the password-path query whose PostgreSQL form locks the user row."""

    return select(UserRow).where(UserRow.email == email).with_for_update()


def _locked_user_by_id_query(user_id: str):
    """Build the account-mutation query whose PostgreSQL form locks the user row."""

    return select(UserRow).where(UserRow.id == user_id).with_for_update()


def _session_context(db, raw_token: str, settings: Settings) -> AuthContext | None:
    token_hash = _digest_token(raw_token, settings)
    session = db.execute(
        select(AuthSessionRow).where(AuthSessionRow.token_hash == token_hash)
    ).scalar_one_or_none()
    now = utc_now_naive()
    if session is None or session.revoked_at is not None or session.expires_at <= now:
        return None
    user = db.get(UserRow, session.user_id)
    workspace_and_role = _personal_workspace(db, session.user_id)
    if user is None or not user.is_active or workspace_and_role is None:
        return None
    if (now - session.last_seen_at).total_seconds() >= 300:
        session.last_seen_at = now
        db.commit()
    workspace, role = workspace_and_role
    return AuthContext(
        user_id=user.id,
        workspace_id=workspace.id,
        role=role,
        anonymous=False,
        session_id=session.id,
        csrf_hash=session.csrf_hash,
    )


def get_auth_context(request: Request) -> AuthContext:
    """Resolve the current user and personal workspace for future ownership checks."""

    settings = get_settings()
    with SessionLocal() as db:
        if settings.auth_mode == "anonymous":
            user, workspace, role = _ensure_local_identity(db)
            return AuthContext(
                user_id=user.id,
                workspace_id=workspace.id,
                role=role,
                anonymous=True,
            )
        raw_token = request.cookies.get(SESSION_COOKIE)
        if not raw_token:
            raise HTTPException(status_code=401, detail="请先登录")
        context = _session_context(db, raw_token, settings)
        if context is None:
            raise HTTPException(status_code=401, detail="登录状态已失效")
        return context


def require_csrf(
    request: Request,
    context: AuthContext = Depends(get_auth_context),
    csrf_header: str | None = Header(default=None, alias="X-CSRF-Token"),
) -> AuthContext:
    if context.anonymous:
        return context
    csrf_cookie = request.cookies.get(CSRF_COOKIE)
    if not csrf_header or not csrf_cookie or not secrets.compare_digest(
        csrf_header, csrf_cookie
    ):
        raise HTTPException(status_code=403, detail="CSRF 校验失败")
    expected = _digest_token(csrf_header, get_settings())
    if not context.csrf_hash or not secrets.compare_digest(expected, context.csrf_hash):
        raise HTTPException(status_code=403, detail="CSRF 校验失败")
    return context


def _set_auth_cookies(response: Response, session_token: str, csrf_token: str) -> None:
    settings = get_settings()
    cookie_options = {
        "secure": settings.auth_cookie_secure,
        "samesite": settings.auth_cookie_samesite,
        "path": "/",
        "max_age": settings.auth_session_ttl_seconds,
    }
    response.set_cookie(
        SESSION_COOKIE,
        session_token,
        httponly=True,
        **cookie_options,
    )
    # Double-submit value is intentionally readable by the same-origin
    # frontend; it cannot authenticate a request without the HttpOnly cookie.
    response.set_cookie(CSRF_COOKIE, csrf_token, httponly=False, **cookie_options)
    response.headers["X-CSRF-Token"] = csrf_token


def _clear_auth_cookies(response: Response) -> None:
    settings = get_settings()
    for name in (SESSION_COOKIE, CSRF_COOKIE):
        response.delete_cookie(
            name,
            path="/",
            secure=settings.auth_cookie_secure,
            httponly=name == SESSION_COOKIE,
            samesite=settings.auth_cookie_samesite,
        )


def _new_session(db, user_id: str) -> tuple[str, str]:
    settings = get_settings()
    raw_token = secrets.token_urlsafe(32)
    csrf_token = secrets.token_urlsafe(32)
    now = utc_now_naive()
    db.add(
        AuthSessionRow(
            user_id=user_id,
            token_hash=_digest_token(raw_token, settings),
            csrf_hash=_digest_token(csrf_token, settings),
            expires_at=now + timedelta(seconds=settings.auth_session_ttl_seconds),
        )
    )
    return raw_token, csrf_token


def _require_enabled() -> None:
    if get_settings().auth_mode != "required":
        raise HTTPException(status_code=409, detail="本地匿名模式未启用账号登录")


def _require_account_security(context: AuthContext) -> None:
    """Reject account-only operations in the explicit anonymous/demo mode."""

    if context.anonymous or get_settings().auth_mode != "required":
        raise HTTPException(status_code=409, detail="本地匿名模式不支持账户安全操作")


def _revoke_other_active_sessions(db, context: AuthContext) -> int:
    """Revoke this user's other live sessions and return the affected count."""

    if context.session_id is None:
        raise HTTPException(status_code=401, detail="登录状态已失效")
    now = utc_now_naive()
    result = db.execute(
        update(AuthSessionRow)
        .where(
            AuthSessionRow.user_id == context.user_id,
            AuthSessionRow.id != context.session_id,
            AuthSessionRow.revoked_at.is_(None),
            AuthSessionRow.expires_at > now,
        )
        .values(revoked_at=now)
    )
    return int(result.rowcount or 0)


@router.post("/register", status_code=201)
def register(payload: RegisterIn, response: Response) -> dict:
    _require_enabled()
    email = normalize_email(payload.email)
    display_name = payload.display_name.strip()
    if not display_name:
        raise HTTPException(status_code=422, detail="显示名称不能为空")
    with SessionLocal() as db:
        if db.execute(select(UserRow.id).where(UserRow.email == email)).first():
            raise HTTPException(status_code=409, detail="该邮箱已注册")
        user = UserRow(
            email=email,
            display_name=display_name,
            password_hash=_hash_password(payload.password),
        )
        workspace = WorkspaceRow(name=f"{display_name}的工作区", kind="personal")
        db.add_all([user, workspace])
        db.flush()
        db.add(
            WorkspaceMemberRow(
                workspace_id=workspace.id,
                user_id=user.id,
                role="owner",
            )
        )
        session_token, csrf_token = _new_session(db, user.id)
        try:
            db.commit()
        except IntegrityError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail="该邮箱已注册") from exc
        result = _serialize_identity(user, workspace, "owner")
    _set_auth_cookies(response, session_token, csrf_token)
    return {"mode": "required", **result}


@router.post("/login")
def login(payload: LoginIn, response: Response) -> dict:
    _require_enabled()
    email = normalize_email(payload.email)
    with SessionLocal() as db:
        # Serialize login with password changes for this account. If login wins
        # the row lock, its new session is committed before password change
        # revokes other sessions. If password change wins, this verification
        # observes the new hash and rejects the old password.
        user = db.execute(_locked_user_by_email_query(email)).scalar_one_or_none()
        invalid = user is None or not user.is_active
        candidate_hash = user.password_hash if user is not None else _DUMMY_PASSWORD_HASH
        try:
            _verify_password(candidate_hash, payload.password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            invalid = True
        if invalid:
            raise HTTPException(status_code=401, detail="邮箱或密码错误")
        if _PASSWORD_HASHER.check_needs_rehash(user.password_hash):
            user.password_hash = _hash_password(payload.password)
        workspace_and_role = _personal_workspace(db, user.id)
        if workspace_and_role is None:
            raise HTTPException(status_code=409, detail="账号缺少个人工作区")
        workspace, role = workspace_and_role
        session_token, csrf_token = _new_session(db, user.id)
        db.commit()
        result = _serialize_identity(user, workspace, role)
    _set_auth_cookies(response, session_token, csrf_token)
    return {"mode": "required", **result}


@router.post("/logout", status_code=204, response_class=Response)
def logout(
    context: AuthContext = Depends(require_csrf),
) -> Response:
    if context.session_id:
        with SessionLocal() as db:
            session = db.get(AuthSessionRow, context.session_id)
            if session is not None and session.revoked_at is None:
                session.revoked_at = utc_now_naive()
                db.commit()
    response = Response(status_code=204)
    _clear_auth_cookies(response)
    return response


@router.get("/me")
def me(context: AuthContext = Depends(get_auth_context)) -> dict:
    with SessionLocal() as db:
        user = db.get(UserRow, context.user_id)
        workspace = db.get(WorkspaceRow, context.workspace_id)
        if user is None or workspace is None:
            raise HTTPException(status_code=401, detail="账号上下文不可用")
        return {
            "mode": "anonymous" if context.anonymous else "required",
            **_serialize_identity(user, workspace, context.role),
        }


@router.get("/sessions")
def list_sessions(context: AuthContext = Depends(get_auth_context)) -> dict:
    _require_account_security(context)
    now = utc_now_naive()
    with SessionLocal() as db:
        rows = db.scalars(
            select(AuthSessionRow)
            .where(
                AuthSessionRow.user_id == context.user_id,
                AuthSessionRow.revoked_at.is_(None),
                AuthSessionRow.expires_at > now,
            )
            .order_by(AuthSessionRow.created_at.desc(), AuthSessionRow.id)
        ).all()
        return {
            "sessions": [
                {
                    "id": row.id,
                    "created_at": row.created_at,
                    "last_seen_at": row.last_seen_at,
                    "expires_at": row.expires_at,
                    "current": row.id == context.session_id,
                }
                for row in rows
            ]
        }


@router.post("/password")
def change_password(
    payload: ChangePasswordIn,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    _require_account_security(context)
    with SessionLocal() as db:
        # Keep verification, password replacement, and other-session
        # revocation in one transaction while holding the same row lock used
        # by login. This prevents concurrent password writes or an old-password
        # login from escaping the revocation boundary.
        user = db.execute(
            _locked_user_by_id_query(context.user_id)
        ).scalar_one_or_none()
        if user is None or not user.is_active:
            raise HTTPException(status_code=401, detail="账号上下文不可用")
        try:
            _verify_password(user.password_hash, payload.current_password)
        except (VerifyMismatchError, VerificationError, InvalidHashError) as exc:
            raise HTTPException(status_code=400, detail="当前密码错误") from exc

        if payload.new_password == payload.current_password:
            raise HTTPException(status_code=400, detail="新密码不能与当前密码相同")

        user.password_hash = _hash_password(payload.new_password)
        revoked_sessions = _revoke_other_active_sessions(db, context)
        db.commit()
        return {"revoked_sessions": revoked_sessions}


@router.post("/sessions/revoke-others")
def revoke_other_sessions(
    context: AuthContext = Depends(require_csrf),
) -> dict:
    _require_account_security(context)
    with SessionLocal() as db:
        revoked_sessions = _revoke_other_active_sessions(db, context)
        db.commit()
        return {"revoked_sessions": revoked_sessions}


@router.delete("/sessions/{session_id}")
def revoke_session(
    session_id: str,
    context: AuthContext = Depends(require_csrf),
) -> dict:
    _require_account_security(context)
    if session_id == context.session_id:
        raise HTTPException(status_code=409, detail="当前会话请使用退出登录")

    now = utc_now_naive()
    with SessionLocal() as db:
        session = db.scalar(
            select(AuthSessionRow).where(
                AuthSessionRow.id == session_id,
                AuthSessionRow.user_id == context.user_id,
                AuthSessionRow.revoked_at.is_(None),
                AuthSessionRow.expires_at > now,
            )
        )
        if session is None:
            # Deliberately do not reveal whether this id belongs to another
            # account, is already revoked, expired, or never existed.
            raise HTTPException(status_code=404, detail="会话不存在")
        session.revoked_at = now
        db.commit()
        return {"revoked_sessions": 1}
