from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.provider import OpenAICompatibleProvider, ProviderError


def check_provider(provider=None) -> tuple[dict, int]:
    """Use the production JSON path, never echo upstream bodies or credentials."""
    provider = provider if provider is not None else OpenAICompatibleProvider()
    report = {"configured": provider.configured, "json_contract_ok": False}
    if not provider.configured:
        report["error_type"] = "ProviderNotConfigured"
        return report, 2
    started = time.perf_counter()
    try:
        result = provider.complete(
            "Return only valid JSON.",
            'Return {"status":"ok"} without additional fields.',
        )
        report["json_contract_ok"] = json.loads(result.text) == {"status": "ok"}
        report["prompt_tokens"] = result.prompt_tokens
        report["completion_tokens"] = result.completion_tokens
        if not report["json_contract_ok"]:
            report["error_type"] = "UnexpectedJSON"
    except ProviderError as exc:
        report["error_type"] = type(exc).__name__
        # Only extract a status code from the safe gateway exception.
        match = re.search(r"HTTP (\d{3})", str(exc))
        if match:
            report["http_status"] = int(match.group(1))
    except (ValueError, TypeError, KeyError, IndexError):
        report["error_type"] = "InvalidResponse"
    report["latency_seconds"] = round(time.perf_counter() - started, 3)
    return report, 0 if report["json_contract_ok"] else 1


def main() -> int:
    report, code = check_provider()
    print(json.dumps(report, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
