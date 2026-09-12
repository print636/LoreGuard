from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
COMPOSE_ENV = ROOT / "tests" / "system" / "compose.env"
COMPOSE_OVERRIDE = ROOT / "docker-compose.e2e.yml"
WEB_BASE_URL = "http://127.0.0.1:8080"


def compose_command(project_name: str, *args: str) -> list[str]:
    return [
        "docker",
        "compose",
        "--project-name",
        project_name,
        "--env-file",
        str(COMPOSE_ENV),
        "-f",
        str(ROOT / "docker-compose.yml"),
        "-f",
        str(COMPOSE_OVERRIDE),
        *args,
    ]


def wait_for_web(base_url: str, timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (URLError, OSError, TimeoutError) as exc:
            last_error = exc
        time.sleep(2)
    raise RuntimeError(
        f"Nginx E2E entry did not become ready: {type(last_error).__name__}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the isolated, model-free Playwright system test."
    )
    parser.add_argument("--project-name", default="loreguard-e2e")
    parser.add_argument("--startup-timeout-seconds", type=float, default=120)
    parser.add_argument(
        "--keep-stack",
        action="store_true",
        help="Keep the isolated stack after the test for local diagnosis.",
    )
    args = parser.parse_args()

    pnpm = shutil.which("pnpm")
    if pnpm is None:
        raise RuntimeError("pnpm is required; run corepack enable first")

    compose = lambda *parts: compose_command(args.project_name, *parts)
    result_code = 1
    started = False
    try:
        subprocess.run(
            compose("up", "--build", "--detach"), cwd=ROOT, check=True
        )
        started = True
        wait_for_web(WEB_BASE_URL, args.startup_timeout_seconds)
        environment = os.environ.copy()
        environment.update(
            {
                "LOREGUARD_E2E_COMPOSE_PROJECT": args.project_name,
            }
        )
        completed = subprocess.run(
            [pnpm, "run", "test:system"],
            # Corepack selects the package-manager version before pnpm handles
            # --dir. Starting in the frontend directory makes that selection
            # honor frontend/package.json in clean CI workspaces as well.
            cwd=ROOT / "frontend",
            env=environment,
            check=False,
        )
        result_code = completed.returncode
        if result_code != 0:
            subprocess.run(compose("logs", "--no-color"), cwd=ROOT, check=False)
    except (subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"system E2E could not complete: {exc}", file=sys.stderr)
        result_code = getattr(exc, "returncode", 1) or 1
        if started:
            subprocess.run(compose("logs", "--no-color"), cwd=ROOT, check=False)
    finally:
        if started and not args.keep_stack:
            subprocess.run(
                compose("down", "--volumes", "--remove-orphans"),
                cwd=ROOT,
                check=False,
            )
    return result_code


if __name__ == "__main__":
    sys.exit(main())
