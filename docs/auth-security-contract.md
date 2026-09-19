# Authentication security contract

This document describes the account, session, and personal-workspace isolation
boundary. It is intentionally narrower than a claim of complete production
deployment hardening or team-workspace authorization.

## Deployment modes

- `AUTH_MODE=anonymous` is for an explicit local/demo deployment. Requests use
  one durable local user and personal workspace without a login cookie.
- `AUTH_MODE=required` enables account registration and opaque server sessions.
  It fails startup unless `AUTH_SECRET_KEY` contains at least 32 characters.
- `DEPLOYMENT_ENVIRONMENT=production` fails startup unless authentication is
  required, cookies are `Secure`, all configured CORS origins use HTTPS, and
  `DATABASE_URL` uses the supported `postgresql+psycopg` SQLAlchemy scheme.
  SQLite, bare `postgresql` (which would require the uninstalled psycopg2
  driver), and other driver schemes are rejected before the application starts.

The authentication secret is a server-only trust root. It must not be reused as
an LLM provider credential or exposed through the frontend.

## Session and password storage

- Passwords use Argon2id through `argon2-cffi`.
- A login session is a cryptographically random opaque value stored only in an
  `HttpOnly` cookie. The database stores an HMAC-SHA256 digest, never the token.
- Sessions have a fixed server-side expiry and can be revoked on logout.
- `last_seen_at` is updated at most once every five minutes to avoid a write for
  every authenticated read.
- Account settings can list only the current user's unexpired, unrevoked
  sessions. The API deliberately does not infer or fabricate device, IP, or
  user-agent metadata that is not collected by the server.
- A user can revoke another live session, revoke every other live session, or
  change the account password. These writes require the session-bound CSRF
  token. Changing the password preserves the current session and revokes every
  other live session in the same transaction.
- Login and password change acquire the same PostgreSQL user-row lock. Password
  verification, hash replacement, session creation, and other-session
  revocation therefore cannot interleave in a way that lets an old-password
  login escape a completed password change. SQLite ignores `FOR UPDATE`; unit
  tests compile both password-path queries with the PostgreSQL dialect, while a
  real PostgreSQL concurrency rehearsal remains part of pre-release validation.
- Password-form validation errors such as an incorrect current password or an
  unchanged replacement return `400`. A `401` remains reserved for an absent,
  expired, or otherwise invalid login session so clients do not mistake a form
  error for a global authentication failure.
- Session identifiers from another account, expired/revoked sessions, and
  nonexistent identifiers all return the same `404`; the current session can
  only be ended through logout. Account-security endpoints return `409` in
  explicit anonymous mode rather than simulating account state.

## Browser request boundary

- Credentialed CORS uses an exact configured origin allowlist; wildcards are
  rejected.
- Browser write requests with an `Origin` outside that allowlist are rejected.
  Requests without `Origin` remain available to non-browser API clients.
- Authenticated writes use a session-bound double-submit CSRF token: the
  readable CSRF cookie must equal `X-CSRF-Token`, and its HMAC must match the
  value stored for the opaque session. The `HttpOnly` session cookie remains
  the authentication credential.

## Resource isolation boundary

- Projects carry a non-null workspace owner. Documents inherit that boundary
  through their project; runs, issues, feedback, visualizations, diagnostics,
  and SSE streams inherit it through their project/run chain.
- Every business read resolves an authenticated or explicit local context and
  filters the requested resource through the current workspace. A UUID from a
  different workspace returns the same `404` as a missing resource.
- Every business write additionally requires the session-bound CSRF value in
  required-auth mode. Demo creation and model-provider checks follow the same
  write boundary.
- Analysis runs record the requesting user and feedback records its author.
  Workers consume server-created run identifiers and frozen run inputs; the
  public HTTP boundary never accepts an arbitrary workspace identity.
- Legacy local data is migrated into one fixed local workspace. In required
  mode, an omitted project workspace fails closed rather than falling back to
  that local identity.

This establishes personal-workspace data isolation, not team sharing or a full
role/permission matrix. Team invitations, shared workspaces, account recovery,
email verification, and distributed authentication rate limits remain later
product work.

## Public deployment boundary

The application security checks fail closed only when the deployment declares
`DEPLOYMENT_ENVIRONMENT=production`. A public deployment must also provide
`AUTH_MODE=required`, a unique high-entropy `AUTH_SECRET_KEY`, secure cookies,
the exact HTTPS browser origin, and a supported PostgreSQL `DATABASE_URL`. The
default Compose stack remains a local development stack: it exposes service
ports, uses development database credentials, and leaves `/metrics` for an
operator-controlled monitoring boundary. It must not be published unchanged.

The workspace migration is exercised through SQLite upgrade/downgrade tests.
A real PostgreSQL migration rehearsal and backup/restore drill are still
required before the first public release. Migration `0004` rejects incompatible
column sets in pre-existing authentication tables, but complete validation of
pre-existing unique constraints and foreign-key deletion policies remains a
hardening item; clean databases receive the intended constraints directly.
