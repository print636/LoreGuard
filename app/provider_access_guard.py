from __future__ import annotations

import hmac

from .db import AccountProviderConfigRow, SessionLocal
from .provider_credentials import ProviderEndpointRejected, provider_endpoint_sha256


class ProviderCredentialRevoked(RuntimeError):
    """A content-free signal that a previously resolved credential was revoked."""


def assert_account_provider_still_authorized(
    *,
    config_id: str,
    user_id: str,
    revision: int,
    endpoint_sha256: str,
    model_name: str,
) -> None:
    """Recheck the frozen provider identity immediately before HTTP transport."""

    if (
        not config_id
        or not user_id
        or revision < 1
        or len(endpoint_sha256) != 64
        or not model_name
    ):
        raise ProviderCredentialRevoked("provider credential is unavailable")
    with SessionLocal() as db:
        row = db.get(AccountProviderConfigRow, config_id)
        try:
            current_endpoint_sha256 = (
                provider_endpoint_sha256(row.base_url) if row is not None else ""
            )
        except ProviderEndpointRejected:
            current_endpoint_sha256 = ""
        valid = (
            row is not None
            and hmac.compare_digest(row.user_id, user_id)
            and row.revision == revision
            and hmac.compare_digest(row.endpoint_sha256, endpoint_sha256)
            and hmac.compare_digest(current_endpoint_sha256, endpoint_sha256)
            # Model aliases are public metadata and may legitimately contain
            # Unicode. ``compare_digest(str, str)`` only accepts ASCII and
            # would raise TypeError for such a saved alias.
            and row.model_name == model_name
            and row.state in {"usable", "superseded"}
            and row.secret_ciphertext is not None
            and row.secret_nonce is not None
            and row.encryption_key_id is not None
        )
    if not valid:
        raise ProviderCredentialRevoked("provider credential is unavailable")
