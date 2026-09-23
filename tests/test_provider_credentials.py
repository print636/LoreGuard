from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.provider_credentials import (
    CredentialAAD,
    EncryptedCredential,
    ProviderCredentialConfigurationError,
    ProviderCredentialDecryptionError,
    ProviderEndpointRejected,
    ProviderKeyring,
    allowed_provider_origins,
    load_or_create_local_keyring,
    load_provider_keyring,
    normalize_provider_base_url,
    normalize_provider_origin,
    provider_endpoint_sha256,
    safe_provider_identity,
    validate_provider_security_configuration,
    validate_provider_request_url,
)


CANARY_SECRET = "sk-provider-canary-must-never-leak"
ALLOWED_ORIGINS = ("https://api.example.com", "https://relay.example:8443")


@pytest.mark.parametrize(
    ("field_name", "prefix", "length"),
    [
        ("openai_api_key", "api-key-validation-canary-", 4_100),
        (
            "account_model_keyring_json",
            "keyring-validation-canary-",
            32_800,
        ),
    ],
)
def test_settings_validation_text_never_echoes_secret_inputs(
    field_name: str, prefix: str, length: int
):
    oversized_value = prefix + "X" * length
    canary = "validation-canary"

    with pytest.raises(ValidationError) as raised:
        Settings(**{field_name: oversized_value})

    assert canary not in str(raised.value)
    assert canary not in repr(raised.value)
    assert canary not in repr(raised.value.errors())


def test_settings_reject_unicode_authorization_key_without_echoing_it():
    unicode_canary = "sk-设置不可回显"

    with pytest.raises(ValidationError) as raised:
        Settings(openai_api_key=unicode_canary)

    assert unicode_canary not in str(raised.value)
    assert unicode_canary not in repr(raised.value)
    assert unicode_canary not in repr(raised.value.errors())


def encoded_key(byte: int) -> str:
    return base64.b64encode(bytes([byte]) * 32).decode("ascii")


def keyring(*, byte: int = 1, key_id: str = "master-v1") -> ProviderKeyring:
    return ProviderKeyring.from_json(
        json.dumps({"version": 1, "keys": {key_id: encoded_key(byte)}}),
        active_key_id=key_id,
    )


def aad(**changes) -> CredentialAAD:
    values = {
        "user_id": "user-1",
        "config_id": "config-1",
        "revision": 3,
        "endpoint_sha256": "a" * 64,
        "model_name": "model-v1",
    }
    values.update(changes)
    return CredentialAAD(**values)


def test_aes_gcm_round_trip_uses_random_nonce_and_redacted_repr():
    configured = keyring()

    first = configured.encrypt(CANARY_SECRET, aad())
    second = configured.encrypt(CANARY_SECRET, aad())

    assert first.nonce != second.nonce
    assert first.ciphertext != second.ciphertext
    assert configured.decrypt(first, aad()) == CANARY_SECRET
    assert configured.decrypt(second, aad()) == CANARY_SECRET
    assert CANARY_SECRET not in repr(first)
    assert encoded_key(1) not in repr(configured)


@pytest.mark.parametrize(
    "field", ["user_id", "config_id", "revision", "endpoint_sha256", "model_name"]
)
def test_every_bound_aad_field_is_authenticated(field: str):
    configured = keyring()
    envelope = configured.encrypt(CANARY_SECRET, aad())
    replacements = {
        "user_id": "user-2",
        "config_id": "config-2",
        "revision": 4,
        "endpoint_sha256": "b" * 64,
        "model_name": "model-v2",
    }

    with pytest.raises(ProviderCredentialDecryptionError) as raised:
        configured.decrypt(envelope, aad(**{field: replacements[field]}))

    assert CANARY_SECRET not in str(raised.value)


def test_ciphertext_nonce_and_wrong_key_tampering_fail_content_free():
    configured = keyring(byte=1)
    envelope = configured.encrypt(CANARY_SECRET, aad())
    tampered_ciphertext = bytes([envelope.ciphertext[0] ^ 1]) + envelope.ciphertext[1:]
    tampered_nonce = bytes([envelope.nonce[0] ^ 1]) + envelope.nonce[1:]

    attempts = (
        EncryptedCredential(
            ciphertext=tampered_ciphertext,
            nonce=envelope.nonce,
            key_id=envelope.key_id,
        ),
        EncryptedCredential(
            ciphertext=envelope.ciphertext,
            nonce=tampered_nonce,
            key_id=envelope.key_id,
        ),
    )
    for tampered in attempts:
        with pytest.raises(ProviderCredentialDecryptionError) as raised:
            configured.decrypt(tampered, aad())
        assert CANARY_SECRET not in str(raised.value)

    with pytest.raises(ProviderCredentialDecryptionError):
        keyring(byte=2).decrypt(envelope, aad())


def test_keyring_keeps_old_decryption_key_during_rotation():
    old = keyring(byte=1, key_id="master-v1")
    envelope = old.encrypt(CANARY_SECRET, aad())
    rotated = ProviderKeyring.from_json(
        json.dumps(
            {
                "keys": {
                    "master-v1": encoded_key(1),
                    "master-v2": encoded_key(2),
                }
            }
        ),
        active_key_id="master-v2",
    )

    assert rotated.decrypt(envelope, aad()) == CANARY_SECRET
    assert rotated.encrypt("new-secret", aad()).key_id == "master-v2"


@pytest.mark.parametrize(
    "payload",
    [
        "not-json",
        "{}",
        '{"keys": {"master-v1": "not base64"}}',
        json.dumps({"keys": {"master-v1": base64.b64encode(b"short").decode()}}),
        '{"keys":{"master-v1":"' + encoded_key(1) + '","master-v1":"' + encoded_key(2) + '"}}',
        json.dumps({"version": 2, "keys": {"master-v1": encoded_key(1)}}),
    ],
)
def test_keyring_json_is_strict_and_does_not_echo_input(payload: str):
    with pytest.raises(ProviderCredentialConfigurationError) as raised:
        ProviderKeyring.from_json(payload, active_key_id="master-v1")

    assert encoded_key(1) not in str(raised.value)
    assert encoded_key(2) not in str(raised.value)


def test_local_keyring_is_persisted_and_reused(tmp_path: Path):
    path = tmp_path / "secrets" / "account-model.key"

    first = load_or_create_local_keyring(path)
    envelope = first.encrypt(CANARY_SECRET, aad())
    second = load_or_create_local_keyring(path)

    assert path.is_file()
    assert second.decrypt(envelope, aad()) == CANARY_SECRET
    assert CANARY_SECRET not in path.read_text(encoding="utf-8")


def test_settings_loader_never_reuses_auth_secret(tmp_path: Path):
    raw_key = b"A" * 32
    same_secret = raw_key.decode("ascii")
    settings = SimpleNamespace(
        account_model_active_key_id="master-v1",
        account_model_keyring_json=json.dumps(
            {"keys": {"master-v1": base64.b64encode(raw_key).decode("ascii")}}
        ),
        account_model_keyring_file="",
        account_model_local_key_path=str(tmp_path / "unused.key"),
        deployment_environment="local",
        auth_secret_key=same_secret,
    )

    with pytest.raises(ProviderCredentialConfigurationError, match="independent"):
        load_provider_keyring(settings)


def test_settings_loader_rejects_ambiguous_explicit_sources(tmp_path: Path):
    settings = SimpleNamespace(
        account_model_active_key_id="master-v1",
        account_model_keyring_json=json.dumps(
            {"keys": {"master-v1": encoded_key(1)}}
        ),
        account_model_keyring_file=str(tmp_path / "also-configured.json"),
        account_model_local_key_path=str(tmp_path / "unused.key"),
        deployment_environment="local",
        auth_secret_key="different-authentication-secret-value",
    )

    with pytest.raises(ProviderCredentialConfigurationError, match="exactly one"):
        load_provider_keyring(settings)


def test_base_url_and_origin_are_canonicalized_and_exactly_allowlisted():
    assert normalize_provider_origin("HTTPS://API.Example.COM:443/") == "https://api.example.com"
    assert (
        normalize_provider_base_url(
            "HTTPS://API.Example.COM:443/v1/", allowed_origins=ALLOWED_ORIGINS
        )
        == "https://api.example.com/v1"
    )
    assert (
        validate_provider_request_url(
            "https://relay.example:8443/openai/v1", allowed_origins=ALLOWED_ORIGINS
        )
        == "https://relay.example:8443/openai/v1"
    )

    with pytest.raises(ProviderEndpointRejected, match="not allowed"):
        normalize_provider_base_url(
            "https://api.example.com:8443/v1", allowed_origins=ALLOWED_ORIGINS
        )


@pytest.mark.parametrize(
    "url",
    [
        "http://api.example.com/v1",
        "https://user@api.example.com/v1",
        "https://user:secret@api.example.com/v1",
        "https://api.example.com/v1?token=secret",
        "https://api.example.com/v1?",
        "https://api.example.com/v1#fragment",
        "https://api.example.com/v1#",
        "https://127.0.0.1/v1",
        "https://[::1]/v1",
        "https://2130706433/v1",
        "https://api.example.com:/v1",
        "https://api.example.com:0/v1",
        "https://localhost/v1",
        "https://provider.local/v1",
        "https://singlelabel/v1",
        "https://api.example.com/v1\nother",
        "https://api.example.com/v1\\..\\admin",
        "https://api.example.com/v1/../admin",
        "https://api.example.com/v1/%2e%2e/admin",
        "https://api.example.com/v1/%252e%252e/admin",
        "https://api.example.com/v1%2f%2fadmin",
        "https://api.example.com/v1%3fsecret=value",
        "https://api.example.com//v1",
    ],
)
def test_ssrf_and_ambiguous_endpoint_forms_are_rejected(url: str):
    with pytest.raises(ProviderEndpointRejected):
        normalize_provider_base_url(url, allowed_origins=ALLOWED_ORIGINS)

    with pytest.raises(ProviderEndpointRejected):
        validate_provider_request_url(url, allowed_origins=ALLOWED_ORIGINS)


@pytest.mark.parametrize(
    "origin",
    [
        "https://api.example.com/v1",
        "https://api.example.com?query=1",
        "http://api.example.com",
        "https://localhost",
    ],
)
def test_allowlist_accepts_origins_only(origin: str):
    with pytest.raises(ProviderEndpointRejected):
        normalize_provider_origin(origin)


def test_settings_origins_include_normalized_server_provider_origin():
    settings = SimpleNamespace(
        auth_mode="anonymous",
        account_model_allowed_origins=(
            "https://Relay.Example:8443, https://api.example.com:443/"
        ),
        openai_base_url="https://API.OpenAI.COM:443/v1/",
    )

    assert allowed_provider_origins(settings) == (
        "https://api.example.com",
        "https://api.openai.com",
        "https://relay.example:8443",
    )


def test_required_mode_uses_only_explicit_account_provider_origins():
    settings = SimpleNamespace(
        auth_mode="required",
        account_model_allowed_origins="https://api.example.com",
        openai_base_url="https://internal-relay.example/v1",
    )

    assert allowed_provider_origins(settings) == ("https://api.example.com",)

    settings.account_model_allowed_origins = ""
    assert allowed_provider_origins(settings) == ()
    with pytest.raises(ProviderEndpointRejected):
        normalize_provider_base_url(
            "https://internal-relay.example/v1",
            allowed_origins=allowed_provider_origins(settings),
        )


def test_endpoint_hash_requires_normalized_safe_url():
    normalized = normalize_provider_base_url(
        "https://api.example.com/v1/", allowed_origins=ALLOWED_ORIGINS
    )

    assert len(provider_endpoint_sha256(normalized)) == 64
    with pytest.raises(ProviderEndpointRejected, match="not normalized"):
        provider_endpoint_sha256("https://API.example.com:443/v1/")


def test_safe_identity_is_content_free_and_exact():
    endpoint_hash = "c" * 64
    identity = safe_provider_identity("config-7", 7, "model-v7", endpoint_hash)
    serialized = json.dumps(identity, sort_keys=True)

    assert identity == {
        "schema_version": 1,
        "source": "account",
        "config_id": "config-7",
        "config_revision": 7,
        "model_alias": "model-v7",
        "endpoint_configuration_sha256": endpoint_hash,
    }
    assert "https://" not in serialized
    assert CANARY_SECRET not in serialized


def test_production_boot_validation_rejects_malformed_explicit_keyring():
    settings = Settings(
        _env_file=None,
        deployment_environment="production",
        auth_mode="required",
        auth_secret_key="independent-authentication-secret-32-bytes",
        auth_cookie_secure=True,
        cors_allowed_origins="https://loreguard.example",
        database_url="postgresql+psycopg://user:password@db/loreguard",
        account_model_active_key_id="prod-v1",
        account_model_keyring_json="not-json",
        account_model_allowed_origins="https://api.example.com",
    )
    with pytest.raises(ProviderCredentialConfigurationError):
        validate_provider_security_configuration(settings)


def test_production_boot_loads_the_mounted_keyring_file(tmp_path: Path):
    keyring_file = tmp_path / "account-model-keyring.json"
    keyring_file.write_text(
        json.dumps(
            {
                "version": 1,
                "keys": {
                    "prod-v1": base64.b64encode(b"K" * 32).decode("ascii")
                },
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    settings = Settings(
        _env_file=None,
        deployment_environment="production",
        auth_mode="required",
        auth_secret_key="independent-authentication-secret-32-bytes",
        auth_cookie_secure=True,
        cors_allowed_origins="https://loreguard.example",
        database_url="postgresql+psycopg://user:password@db/loreguard",
        account_model_active_key_id="prod-v1",
        account_model_keyring_file=str(keyring_file),
        account_model_allowed_origins="https://api.example.com",
    )

    validate_provider_security_configuration(settings)

    keyring_file.write_text("not-json", encoding="utf-8")
    with pytest.raises(ProviderCredentialConfigurationError):
        validate_provider_security_configuration(settings)
