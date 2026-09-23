from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import os
import re
import stat
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlsplit, urlunsplit

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import Settings


CREDENTIAL_SCHEMA_VERSION = 1
PROVIDER_IDENTITY_SCHEMA_VERSION = 1
_AAD_SCHEMA = "loreguard-account-provider-credential-v1"
_KEYRING_FILE_SCHEMA_VERSION = 1
_AES_KEY_BYTES = 32
_NONCE_BYTES = 12
_MAX_SECRET_CHARACTERS = 4_096
_MAX_SECRET_BYTES = _MAX_SECRET_CHARACTERS * 4
_MAX_CIPHERTEXT_BYTES = _MAX_SECRET_BYTES + 16
_MAX_KEYRING_BYTES = 32_768
_MAX_KEY_COUNT = 32
_MAX_ENDPOINT_LENGTH = 2_048
_KEY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_HEX_SHA256_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_HOST_LABEL_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_PERCENT_ESCAPE_PATTERN = re.compile(r"%(?![0-9A-Fa-f]{2})")


class ProviderCredentialError(RuntimeError):
    """Base class for content-free account-provider credential failures."""


class ProviderCredentialConfigurationError(ProviderCredentialError):
    """The dedicated provider credential trust root is not safely configured."""


class ProviderCredentialDecryptionError(ProviderCredentialError):
    """A credential envelope could not be authenticated and decrypted."""


class ProviderEndpointRejected(ValueError):
    """A provider endpoint failed the exact-origin SSRF boundary."""


@dataclass(frozen=True, slots=True)
class CredentialAAD:
    """Immutable semantic identity authenticated with a provider secret."""

    user_id: str
    config_id: str
    revision: int
    endpoint_sha256: str
    model_name: str

    def __post_init__(self) -> None:
        _validate_identifier(self.user_id, field="user id", maximum=128)
        _validate_identifier(self.config_id, field="config id", maximum=128)
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise ValueError("provider revision must be an integer")
        if self.revision < 1:
            raise ValueError("provider revision must be positive")
        _validate_endpoint_hash(self.endpoint_sha256)
        _validate_model_name(self.model_name)

    def canonical_bytes(self) -> bytes:
        """Return a stable, unambiguous AES-GCM additional-data payload."""

        return json.dumps(
            {
                "config_id": self.config_id,
                "endpoint_sha256": self.endpoint_sha256,
                "model_name": self.model_name,
                "revision": self.revision,
                "schema": _AAD_SCHEMA,
                "user_id": self.user_id,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")


@dataclass(frozen=True, slots=True, repr=False)
class EncryptedCredential:
    """An AES-GCM envelope safe to split across database columns."""

    ciphertext: bytes
    nonce: bytes
    key_id: str
    schema_version: int = CREDENTIAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.ciphertext, bytes)
            or not 16 < len(self.ciphertext) <= _MAX_CIPHERTEXT_BYTES
        ):
            raise ValueError("provider credential ciphertext is invalid")
        if not isinstance(self.nonce, bytes) or len(self.nonce) != _NONCE_BYTES:
            raise ValueError("provider credential nonce is invalid")
        _validate_key_id(self.key_id)
        if self.schema_version != CREDENTIAL_SCHEMA_VERSION:
            raise ValueError("provider credential schema is unsupported")

    def __repr__(self) -> str:
        return (
            "EncryptedCredential("
            f"ciphertext=<redacted:{len(self.ciphertext)} bytes>, "
            f"nonce=<redacted:{len(self.nonce)} bytes>, "
            f"key_id={self.key_id!r}, schema_version={self.schema_version})"
        )


class ProviderKeyring:
    """A bounded AES-256-GCM keyring with one active encryption key."""

    __slots__ = ("_active_key_id", "_keys")

    def __init__(self, keys: Mapping[str, bytes], *, active_key_id: str):
        _validate_key_id(active_key_id)
        if not keys or len(keys) > _MAX_KEY_COUNT:
            raise ProviderCredentialConfigurationError(
                "provider keyring must contain between 1 and 32 keys"
            )
        checked: dict[str, bytes] = {}
        for key_id, key in keys.items():
            _validate_key_id(key_id)
            if not isinstance(key, bytes) or len(key) != _AES_KEY_BYTES:
                raise ProviderCredentialConfigurationError(
                    "each provider master key must be exactly 32 bytes"
                )
            checked[key_id] = key
        if active_key_id not in checked:
            raise ProviderCredentialConfigurationError(
                "active provider master key is absent from the keyring"
            )
        self._active_key_id = active_key_id
        self._keys = MappingProxyType(checked)

    @classmethod
    def from_json(cls, raw_json: str, *, active_key_id: str) -> ProviderKeyring:
        """Parse a strict JSON mapping of key IDs to base64 AES-256 keys.

        Both ``{"keys": {"key-id": "base64..."}, "version": 1}`` and a
        compact top-level ``{"key-id": "base64..."}`` mapping are accepted.
        No secret values are included in parsing errors.
        """

        if not isinstance(raw_json, str) or not raw_json:
            raise ProviderCredentialConfigurationError("provider keyring JSON is empty")
        try:
            raw_size = len(raw_json.encode("utf-8"))
        except UnicodeEncodeError:
            raise ProviderCredentialConfigurationError(
                "provider keyring JSON is invalid"
            ) from None
        if raw_size > _MAX_KEYRING_BYTES:
            raise ProviderCredentialConfigurationError("provider keyring JSON is too large")
        try:
            payload = json.loads(raw_json, object_pairs_hook=_unique_json_object)
        except (json.JSONDecodeError, UnicodeError):
            raise ProviderCredentialConfigurationError(
                "provider keyring JSON is invalid"
            ) from None
        if not isinstance(payload, dict) or not payload:
            raise ProviderCredentialConfigurationError(
                "provider keyring JSON must be an object"
            )
        if "keys" in payload:
            if set(payload) - {"keys", "version"}:
                raise ProviderCredentialConfigurationError(
                    "provider keyring JSON contains unsupported fields"
                )
            version = payload.get("version", _KEYRING_FILE_SCHEMA_VERSION)
            if (
                isinstance(version, bool)
                or version != _KEYRING_FILE_SCHEMA_VERSION
            ):
                raise ProviderCredentialConfigurationError(
                    "provider keyring schema is unsupported"
                )
            encoded_keys = payload["keys"]
        else:
            encoded_keys = payload
        if not isinstance(encoded_keys, dict) or not encoded_keys:
            raise ProviderCredentialConfigurationError(
                "provider keyring keys must be a non-empty object"
            )
        if len(encoded_keys) > _MAX_KEY_COUNT:
            raise ProviderCredentialConfigurationError("provider keyring has too many keys")
        decoded: dict[str, bytes] = {}
        for key_id, encoded_key in encoded_keys.items():
            _validate_key_id(key_id)
            if not isinstance(encoded_key, str) or not encoded_key:
                raise ProviderCredentialConfigurationError(
                    "provider master keys must be base64 strings"
                )
            try:
                key = base64.b64decode(encoded_key, validate=True)
            except (binascii.Error, ValueError):
                raise ProviderCredentialConfigurationError(
                    "provider master key base64 is invalid"
                ) from None
            if len(key) != _AES_KEY_BYTES:
                raise ProviderCredentialConfigurationError(
                    "each provider master key must decode to exactly 32 bytes"
                )
            decoded[key_id] = key
        return cls(decoded, active_key_id=active_key_id)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], *, active_key_id: str) -> ProviderKeyring:
        keyring_path = Path(path)
        _reject_secret_symlink(keyring_path)
        try:
            file_stat = keyring_path.stat()
            if not stat.S_ISREG(file_stat.st_mode):
                raise ProviderCredentialConfigurationError(
                    "provider keyring path is not a regular file"
                )
            if file_stat.st_size > _MAX_KEYRING_BYTES:
                raise ProviderCredentialConfigurationError(
                    "provider keyring file is too large"
                )
            raw = keyring_path.read_text(encoding="utf-8")
        except ProviderCredentialConfigurationError:
            raise
        except (OSError, UnicodeError) as exc:
            raise ProviderCredentialConfigurationError(
                "provider keyring file could not be read"
            ) from exc
        return cls.from_json(raw, active_key_id=active_key_id)

    @property
    def active_key_id(self) -> str:
        return self._active_key_id

    @property
    def key_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._keys))

    def encrypt(self, secret: str, aad: CredentialAAD) -> EncryptedCredential:
        if not isinstance(aad, CredentialAAD):
            raise TypeError("aad must be CredentialAAD")
        if (
            not isinstance(secret, str)
            or not secret
            or len(secret) > _MAX_SECRET_CHARACTERS
        ):
            raise ValueError("provider credential must contain 1 to 4096 characters")
        try:
            plaintext = secret.encode("utf-8")
        except UnicodeEncodeError:
            raise ValueError("provider credential must be valid UTF-8") from None
        nonce = os.urandom(_NONCE_BYTES)
        ciphertext = AESGCM(self._keys[self._active_key_id]).encrypt(
            nonce,
            plaintext,
            aad.canonical_bytes(),
        )
        return EncryptedCredential(
            ciphertext=ciphertext,
            nonce=nonce,
            key_id=self._active_key_id,
        )

    def decrypt(self, envelope: EncryptedCredential, aad: CredentialAAD) -> str:
        if not isinstance(envelope, EncryptedCredential):
            raise TypeError("envelope must be an EncryptedCredential")
        if not isinstance(aad, CredentialAAD):
            raise TypeError("aad must be CredentialAAD")
        key = self._keys.get(envelope.key_id)
        if key is None:
            raise ProviderCredentialDecryptionError(
                "provider credential could not be decrypted"
            )
        try:
            plaintext = AESGCM(key).decrypt(
                envelope.nonce,
                envelope.ciphertext,
                aad.canonical_bytes(),
            )
            secret = plaintext.decode("utf-8")
        except (InvalidTag, UnicodeDecodeError, ValueError):
            raise ProviderCredentialDecryptionError(
                "provider credential could not be decrypted"
            ) from None
        if not secret or len(secret) > _MAX_SECRET_CHARACTERS:
            raise ProviderCredentialDecryptionError(
                "provider credential could not be decrypted"
            )
        return secret

    def assert_independent_from(self, auth_secret: object) -> None:
        """Reject an accidentally duplicated authentication trust root."""

        if not isinstance(auth_secret, str) or not auth_secret:
            return
        try:
            candidates = [auth_secret.encode("utf-8")]
        except UnicodeEncodeError:
            raise ProviderCredentialConfigurationError(
                "authentication trust root is not valid UTF-8"
            ) from None
        try:
            candidates.append(base64.b64decode(auth_secret, validate=True))
        except (binascii.Error, ValueError):
            pass
        if any(
            hmac.compare_digest(key, candidate)
            for key in self._keys.values()
            for candidate in candidates
        ):
            raise ProviderCredentialConfigurationError(
                "provider and authentication trust roots must be independent"
            )

    def __repr__(self) -> str:
        return (
            "ProviderKeyring("
            f"key_ids={self.key_ids!r}, active_key_id={self.active_key_id!r})"
        )


def load_provider_keyring(settings: Settings) -> ProviderKeyring:
    """Load only the dedicated account-model key source configured in Settings.

    Production never generates a key. Local development may persist a
    dedicated key at ``account_model_local_key_path``. ``auth_secret_key`` is
    neither a fallback nor a key-derivation input.
    """

    active_key_id = settings.account_model_active_key_id.strip()
    _validate_key_id(active_key_id)
    inline_json = settings.account_model_keyring_json.strip()
    keyring_file = settings.account_model_keyring_file.strip()
    if inline_json and keyring_file:
        raise ProviderCredentialConfigurationError(
            "configure exactly one explicit provider keyring source"
        )
    if inline_json:
        keyring = ProviderKeyring.from_json(inline_json, active_key_id=active_key_id)
    elif keyring_file:
        keyring = ProviderKeyring.from_file(keyring_file, active_key_id=active_key_id)
    else:
        if settings.deployment_environment == "production":
            raise ProviderCredentialConfigurationError(
                "production requires an explicit provider keyring"
            )
        local_path = settings.account_model_local_key_path.strip()
        if not local_path:
            raise ProviderCredentialConfigurationError(
                "local provider master-key path is not configured"
            )
        keyring = load_or_create_local_keyring(local_path, key_id=active_key_id)
    keyring.assert_independent_from(settings.auth_secret_key)
    return keyring


def load_or_create_local_keyring(
    path: str | os.PathLike[str], *, key_id: str = "local-v1"
) -> ProviderKeyring:
    """Load or exclusively create a persistent local-development keyring."""

    _validate_key_id(key_id)
    keyring_path = Path(path)
    try:
        keyring_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise ProviderCredentialConfigurationError(
            "local provider key directory could not be created"
        ) from exc
    _reject_secret_symlink(keyring_path)
    if keyring_path.exists():
        _restrict_local_key_permissions(keyring_path)
        return ProviderKeyring.from_file(keyring_path, active_key_id=key_id)

    encoded_key = base64.b64encode(AESGCM.generate_key(bit_length=256)).decode("ascii")
    payload = json.dumps(
        {
            "version": _KEYRING_FILE_SCHEMA_VERSION,
            "keys": {key_id: encoded_key},
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(keyring_path, flags, 0o600)
    except FileExistsError:
        # A concurrent process won the exclusive create. Its complete keyring
        # is now the single trust root for both processes.
        for _ in range(50):
            try:
                return ProviderKeyring.from_file(keyring_path, active_key_id=key_id)
            except ProviderCredentialConfigurationError:
                time.sleep(0.01)
        raise ProviderCredentialConfigurationError(
            "concurrently-created local provider keyring is not readable"
        )
    except OSError as exc:
        raise ProviderCredentialConfigurationError(
            "local provider keyring could not be created"
        ) from exc
    write_failed = False
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    except OSError as exc:
        write_failed = True
        raise ProviderCredentialConfigurationError(
            "local provider keyring could not be persisted"
        ) from exc
    finally:
        os.close(descriptor)
        if write_failed:
            try:
                keyring_path.unlink()
            except OSError:
                pass
    _restrict_local_key_permissions(keyring_path)
    return ProviderKeyring.from_file(keyring_path, active_key_id=key_id)


def allowed_provider_origins(settings: Settings) -> tuple[str, ...]:
    """Return the exact HTTPS origins accounts may select.

    Anonymous/local mode preserves the legacy single-user behavior by trusting
    the deployment's ``OPENAI_BASE_URL`` origin.  Required-auth deployments
    fail closed and trust only the administrator's explicit account allowlist;
    a stale internal service-default URL must not silently become user
    selectable.
    """

    configured = settings.account_model_allowed_origins.strip()
    entries: list[str] = []
    if configured:
        entries = [item.strip() for item in configured.split(",")]
        if any(not item for item in entries):
            raise ProviderEndpointRejected("provider origin allowlist is invalid")
    origins: set[str] = set()
    if settings.auth_mode == "anonymous":
        legacy_base = _parse_https_url(settings.openai_base_url, origin_only=False)
        origins.add(_origin_from_parts(legacy_base))
    if entries:
        origins.update(_normalize_allowed_origins(entries))
    return tuple(sorted(origins))


def validate_provider_security_configuration(settings: Settings) -> None:
    """Fail closed on malformed production keyrings and endpoint policy."""

    allowed_provider_origins(settings)
    if settings.deployment_environment == "production":
        load_provider_keyring(settings)


def normalize_provider_origin(value: str) -> str:
    """Normalize one safe HTTPS origin for an exact allowlist."""

    return _origin_from_parts(_parse_https_url(value, origin_only=True))


def normalize_provider_base_url(
    value: str, *, allowed_origins: str | Iterable[str]
) -> str:
    """Normalize a provider base URL and enforce its exact HTTPS origin."""

    parts = _parse_https_url(value, origin_only=False)
    normalized_origin = _origin_from_parts(parts)
    if normalized_origin not in _normalize_allowed_origins(allowed_origins):
        raise ProviderEndpointRejected("provider endpoint origin is not allowed")
    path = parts.path.rstrip("/")
    return urlunsplit(("https", parts.netloc, path, "", ""))


def validate_provider_request_url(
    value: str, *, allowed_origins: str | Iterable[str]
) -> str:
    """Re-apply the save-time endpoint policy immediately before a request."""

    return normalize_provider_base_url(value, allowed_origins=allowed_origins)


def provider_endpoint_sha256(normalized_base_url: str) -> str:
    """Fingerprint a safe canonical endpoint without persisting the URL."""

    parts = _parse_https_url(normalized_base_url, origin_only=False)
    normalized = urlunsplit(
        ("https", parts.netloc, parts.path.rstrip("/"), "", "")
    )
    if normalized != normalized_base_url:
        raise ProviderEndpointRejected("provider endpoint is not normalized")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def safe_provider_identity(
    config_id: str,
    revision: int,
    model_name: str,
    endpoint_sha256: str,
) -> dict[str, Any]:
    """Build the only provider identity safe for run/context persistence."""

    _validate_identifier(config_id, field="config id", maximum=128)
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise ValueError("provider revision must be positive")
    _validate_model_name(model_name)
    _validate_endpoint_hash(endpoint_sha256)
    return {
        "schema_version": PROVIDER_IDENTITY_SCHEMA_VERSION,
        "source": "account",
        "config_id": config_id,
        "config_revision": revision,
        "model_alias": model_name,
        "endpoint_configuration_sha256": endpoint_sha256,
    }


def _normalize_allowed_origins(values: str | Iterable[str]) -> frozenset[str]:
    if isinstance(values, str):
        raw_values = [item.strip() for item in values.split(",")]
    else:
        try:
            raw_values = list(values)
        except TypeError as exc:
            raise ProviderEndpointRejected("provider origin allowlist is invalid") from exc
    if not raw_values or any(not isinstance(item, str) or not item for item in raw_values):
        raise ProviderEndpointRejected("provider origin allowlist is empty or invalid")
    return frozenset(normalize_provider_origin(item) for item in raw_values)


def _parse_https_url(value: object, *, origin_only: bool):
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > _MAX_ENDPOINT_LENGTH
        or _has_forbidden_codepoint(value)
        or "\\" in value
        or "?" in value
        or "#" in value
    ):
        raise ProviderEndpointRejected("provider endpoint is invalid")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except (UnicodeError, ValueError) as exc:
        raise ProviderEndpointRejected("provider endpoint is invalid") from exc
    if (
        parsed.scheme.lower() != "https"
        or not parsed.netloc
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "%" in parsed.netloc
        or parsed.netloc.endswith(":")
        or port == 0
    ):
        raise ProviderEndpointRejected("provider endpoint must be an HTTPS URL")
    normalized_hostname = _normalize_public_hostname(hostname)
    if port == 443 or port is None:
        netloc = normalized_hostname
    else:
        netloc = f"{normalized_hostname}:{port}"
    path = parsed.path
    if origin_only and path not in {"", "/"}:
        raise ProviderEndpointRejected("provider allowlist entries must be origins")
    _validate_safe_path(path)
    return parsed._replace(scheme="https", netloc=netloc, path=path, query="", fragment="")


def _normalize_public_hostname(hostname: str) -> str:
    if hostname.endswith("."):
        raise ProviderEndpointRejected("provider hostname is invalid")
    try:
        normalized = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ProviderEndpointRejected("provider hostname is invalid") from exc
    if not 1 <= len(normalized) <= 253:
        raise ProviderEndpointRejected("provider hostname is invalid")
    try:
        ipaddress.ip_address(normalized)
    except ValueError:
        pass
    else:
        raise ProviderEndpointRejected("provider IP literals are not allowed")
    if (
        normalized == "localhost"
        or normalized.endswith(".localhost")
        or normalized == "local"
        or normalized.endswith(".local")
        or "." not in normalized
    ):
        raise ProviderEndpointRejected("local provider hostnames are not allowed")
    # Numeric and legacy IPv4 spellings can be interpreted as addresses by
    # operating-system resolvers even when ipaddress does not recognize them.
    compact = normalized.replace(".", "")
    if compact.isdigit() or normalized.startswith(("0x", "0X")):
        raise ProviderEndpointRejected("provider IP literals are not allowed")
    labels = normalized.split(".")
    if any(not _HOST_LABEL_PATTERN.fullmatch(label) for label in labels):
        raise ProviderEndpointRejected("provider hostname is invalid")
    return normalized


def _validate_safe_path(path: str) -> None:
    if not path:
        return
    if not path.startswith("/") or path.startswith("//") or "//" in path:
        raise ProviderEndpointRejected("provider endpoint path is invalid")
    if _PERCENT_ESCAPE_PATTERN.search(path):
        raise ProviderEndpointRejected("provider endpoint path encoding is invalid")
    candidate = path
    for _ in range(5):
        _reject_unsafe_decoded_path(candidate)
        try:
            decoded = unquote(candidate, errors="strict")
        except UnicodeDecodeError as exc:
            raise ProviderEndpointRejected(
                "provider endpoint path encoding is invalid"
            ) from exc
        if decoded == candidate:
            break
        candidate = decoded
    else:
        raise ProviderEndpointRejected("provider endpoint path encoding is ambiguous")
    _reject_unsafe_decoded_path(candidate)


def _reject_unsafe_decoded_path(path: str) -> None:
    if (
        "\\" in path
        or "?" in path
        or "#" in path
        or path.startswith("//")
        or "//" in path
        or _has_forbidden_codepoint(path)
    ):
        raise ProviderEndpointRejected("provider endpoint path is invalid")
    if any(segment in {".", ".."} for segment in path.split("/")):
        raise ProviderEndpointRejected("provider endpoint path traversal is not allowed")


def _origin_from_parts(parts) -> str:
    return urlunsplit(("https", parts.netloc, "", "", ""))


def _validate_key_id(value: object) -> None:
    if not isinstance(value, str) or not _KEY_ID_PATTERN.fullmatch(value):
        raise ProviderCredentialConfigurationError("provider key id is invalid")


def _validate_identifier(value: object, *, field: str, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or _has_forbidden_codepoint(value)
    ):
        raise ValueError(f"provider {field} is invalid")


def _validate_model_name(value: object) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 255
        or _has_control_codepoint(value)
    ):
        raise ValueError("provider model name is invalid")


def _validate_endpoint_hash(value: object) -> None:
    if not isinstance(value, str) or not _HEX_SHA256_PATTERN.fullmatch(value):
        raise ValueError("provider endpoint hash is invalid")


def _has_forbidden_codepoint(value: str) -> bool:
    return any(
        character.isspace()
        or ord(character) == 127
        or _has_control_codepoint(character)
        for character in value
    )


def _has_control_codepoint(value: str) -> bool:
    return any(
        ord(character) == 127
        or unicodedata.category(character).startswith("C")
        for character in value
    )


def _reject_secret_symlink(path: Path) -> None:
    try:
        if path.is_symlink():
            raise ProviderCredentialConfigurationError(
                "provider keyring path must not be a symbolic link"
            )
    except OSError as exc:
        raise ProviderCredentialConfigurationError(
            "provider keyring path could not be inspected"
        ) from exc


def _restrict_local_key_permissions(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError as exc:
        # Windows ACLs do not implement POSIX mode bits; exclusive creation is
        # still required. Deployment environments must use an explicit secret.
        if os.name != "nt":
            raise ProviderCredentialConfigurationError(
                "local provider keyring permissions could not be restricted"
            ) from exc


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProviderCredentialConfigurationError(
                "provider keyring JSON contains duplicate fields"
            )
        result[key] = value
    return result
