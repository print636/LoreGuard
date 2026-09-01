from datetime import UTC, datetime


def utc_now_naive() -> datetime:
    """Return current UTC while preserving the app's existing naive DB format."""
    return datetime.now(UTC).replace(tzinfo=None)
