from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import get_settings


def main() -> int:
    settings = get_settings()
    if not settings.openai_api_key:
        print("provider_check=missing_key")
        return 2

    roots = [settings.openai_base_url.rstrip("/")]
    if not roots[0].endswith("/v1"):
        roots.append(f"{roots[0]}/v1")
    headers = {"Authorization": f"Bearer {settings.openai_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": "只返回 JSON。"},
            {"role": "user", "content": "返回 {\"status\":\"ok\"}，不要添加其他文字。"},
        ],
        "temperature": 0,
        "max_tokens": 32,
    }

    with httpx.Client(timeout=httpx.Timeout(45, connect=10), follow_redirects=True) as client:
        for root in roots:
            started = time.perf_counter()
            try:
                models = client.get(f"{root}/models", headers=headers)
                completion = client.post(f"{root}/chat/completions", headers=headers, json=payload)
                elapsed = time.perf_counter() - started
                model_count = None
                model_ids: list[str] = []
                if models.is_success:
                    body = models.json()
                    model_count = len(body.get("data", [])) if isinstance(body, dict) else None
                    model_ids = [str(item.get("id")) for item in body.get("data", []) if isinstance(item, dict) and item.get("id")]
                response_preview = ""
                error_preview = ""
                if completion.is_success:
                    body = completion.json()
                    response_preview = str(body.get("choices", [{}])[0].get("message", {}).get("content", ""))[:80]
                else:
                    error_preview = completion.text[:240]
                print(json.dumps({
                    "base_url": root,
                    "models_status": models.status_code,
                    "model_count": model_count,
                    "model_ids": model_ids,
                    "completion_status": completion.status_code,
                    "latency_seconds": round(elapsed, 3),
                    "response_preview": response_preview,
                    "error_preview": error_preview,
                }, ensure_ascii=False))
                if completion.is_success:
                    return 0
            except Exception as exc:
                print(json.dumps({"base_url": root, "error_type": type(exc).__name__, "error": str(exc)[:160]}, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
