"""Stable, database-independent project-name ordering.

The persisted key is versioned because pinyin data and transliteration choices
can change. A future algorithm change must backfill existing keys in a new
migration before newly created projects use that version.
"""

from __future__ import annotations

import unicodedata

from pypinyin import Style, lazy_pinyin


PROJECT_NAME_SORT_KEY_V1_PREFIX = b"v1\0"


def project_name_sort_key_v1(name: str) -> bytes:
    """Return the immutable v1 binary pinyin key for a project name.

    Binary keys compare identically in SQLite and PostgreSQL, independent of
    the database's text collation. Equal transliterations are ordered by ID
    in the catalog query.
    """

    normalized = unicodedata.normalize("NFKC", name).casefold()
    transliterated = "".join(lazy_pinyin(normalized, style=Style.NORMAL))
    return PROJECT_NAME_SORT_KEY_V1_PREFIX + transliterated.encode("utf-8")


def project_name_sort_key(name: str) -> bytes:
    """Return the active project-name key; migrations call their fixed version."""

    return project_name_sort_key_v1(name)
