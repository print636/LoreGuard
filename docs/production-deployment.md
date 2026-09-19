# Production deployment boundary

LoreGuard supplies a defensive Docker Compose overlay, not a turnkey public
hosting platform. The overlay makes unsafe omissions fail during Compose
configuration and narrows host exposure, but the operator remains responsible
for the domain, TLS termination, secret management, backups, monitoring access,
host firewall, updates, and incident response.

## Prepare configuration

Copy `.env.production.example` to a server-only path outside the repository,
replace every placeholder, and restrict its filesystem permissions. Required
values are:

- `PUBLIC_ORIGIN`: one exact HTTPS origin, with scheme and hostname and without
  a path or wildcard, such as `https://loreguard.example.com`;
- `AUTH_SECRET_KEY`: a unique, high-entropy, server-only value of at least 32
  characters, unrelated to any model-provider key;
- `POSTGRES_DB`, `POSTGRES_USER`, and `POSTGRES_PASSWORD`: independent
  production database credentials, not the development `loreguard/loreguard`
  credential;
- `DATABASE_URL`: the matching SQLAlchemy PostgreSQL URL, using exactly
  `postgresql+psycopg://`, the installed psycopg 3 driver. URL-encode reserved
  characters in its user/password components. Production startup rejects
  SQLite, non-PostgreSQL databases, bare `postgresql://` (which defaults to the
  uninstalled psycopg2 driver), and other unsupported driver schemes.

The API and worker receive the same authentication, origin, and database
configuration. The overlay fixes `DEPLOYMENT_ENVIRONMENT=production`,
`AUTH_MODE=required`, and `AUTH_COOKIE_SECURE=true`; these cannot be weakened by
values in the env file.

Before building or starting containers, validate the fully merged model. This
does not contact a registry or require a running Docker daemon:

```powershell
docker compose --env-file C:\secure\loreguard.production.env `
  -f docker-compose.yml -f docker-compose.production.yml config --quiet
```

To inspect the result without printing secrets, prefer `config --services` or
`config --images`. Plain `docker compose config` interpolates secret values and
must not be copied into tickets or logs.

Start the normal stack only after a database backup and migration plan exists:

```powershell
docker compose --env-file C:\secure\loreguard.production.env `
  -f docker-compose.yml -f docker-compose.production.yml up -d --build
```

Prometheus is disabled by default. Add `--profile observability` only when an
operator-controlled collector needs it on the private Compose network.

## Network boundary

The merged production configuration has these deliberate properties:

- API port `8000` and Prometheus port `9090` are not published to the host;
- PostgreSQL and Redis have no published host ports;
- the Web container is published only on `127.0.0.1:${WEB_BIND_PORT}`;
- browser API requests pass through the Web container's `/api/` proxy.

An external reverse proxy on the same host must terminate HTTPS for
`PUBLIC_ORIGIN` and forward to that loopback Web port. Configure certificate
issuance/renewal, request-size and timeout limits, security headers, access
logging, and trusted proxy rules there. Do not bind the Web port to `0.0.0.0`
or expose API/Prometheus directly merely to work around proxy configuration.

## Operator responsibilities before public traffic

1. Take and restore-test encrypted PostgreSQL backups; snapshot the volume
   before every migration and define retention off the application host.
2. Rehearse Alembic upgrade and rollback against the same PostgreSQL major
   version and a sanitized copy of production-scale data.
3. Put provider keys and auth/database secrets in a restricted secret store or
   server-only env file; rotate them after any suspected disclosure.
4. Restrict SSH and host firewall access, patch images and the host, and pin or
   review image updates according to the deployment policy.
5. Keep Prometheus and operational endpoints on a private monitoring boundary.
   The overlay removes their host ports; it does not add monitoring auth.
6. Verify account flows, workspace isolation, uploads, SSE reconnects, provider
   failures, rate limits, and backup restoration through the public HTTPS URL.

These controls establish a safer deployment boundary; they do not certify the
application as highly available, audited, or ready for unreviewed public use.
