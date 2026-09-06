from __future__ import annotations

import os
import shutil
import tempfile


# Pytest must never discover the repository's developer .env or loreguard.db.
# Keep this process in an isolated working directory before application modules
# are imported during collection. CI's unittest runner also sets DATABASE_URL.
_ORIGINAL_WORKING_DIRECTORY = os.getcwd()
_TEST_WORKSPACE = tempfile.mkdtemp(prefix="loreguard-pytest-")
os.chdir(_TEST_WORKSPACE)
os.environ["DATABASE_URL"] = (
    f"sqlite:///{_TEST_WORKSPACE.replace(os.sep, '/')}/tests.db"
)
os.environ["OPENAI_API_KEY"] = ""
os.environ["ENABLE_MODEL_EXTRACTION"] = "false"
os.environ["ENABLE_REVIEW_AGENT"] = "false"
os.environ["ENABLE_EMBEDDINGS"] = "false"


def pytest_sessionfinish(session, exitstatus) -> None:
    del session, exitstatus
    try:
        from app import db as app_db

        app_db.engine.dispose()
    finally:
        os.chdir(_ORIGINAL_WORKING_DIRECTORY)
        # A third-party SQLite handle on Windows must not turn a green suite
        # into a false failure. Best-effort cleanup leaves only a temp artifact.
        shutil.rmtree(_TEST_WORKSPACE, ignore_errors=True)
