from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable

import httpx
from pydantic import BaseModel

from .config import Settings, get_settings


class ModelResult(BaseModel):
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class ProviderError(RuntimeError):
    """Safe provider failure that never includes credentials or response bodies."""


class ProviderNotConfigured(ProviderError):
    pass


class ProviderRetryExhausted(ProviderError):
    pass


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay_seconds: float = 0.25


class OpenAICompatibleProvider:
    """Small, injectable OpenAI-compatible gateway with bounded retries."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        retry_policy: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or get_settings()
        self.transport = transport
        self.retry_policy = retry_policy or RetryPolicy(
            max_attempts=self.settings.provider_max_attempts
        )
        self.sleep = sleep

    @property
    def configured(self) -> bool:
        return self.settings.enable_model_extraction and bool(
            self.settings.openai_api_key.strip()
        )

    def complete(self, system: str, user: str) -> ModelResult:
        if not self.configured:
            raise ProviderNotConfigured("模型未配置")

        payload = {
            "model": self.settings.openai_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"
        last_reason = "unknown"

        with httpx.Client(
            timeout=httpx.Timeout(self.settings.provider_timeout_seconds, connect=10),
            follow_redirects=False,
            transport=self.transport,
        ) as client:
            for attempt in range(1, self.retry_policy.max_attempts + 1):
                try:
                    response = client.post(endpoint, json=payload, headers=headers)
                    if response.status_code == 429 or response.status_code >= 500:
                        last_reason = f"HTTP {response.status_code}"
                        raise httpx.HTTPStatusError(
                            last_reason, request=response.request, response=response
                        )
                    if response.status_code >= 300:
                        raise ProviderError(f"模型服务拒绝请求（HTTP {response.status_code}）")
                    body: Any = response.json()
                    text = body["choices"][0]["message"]["content"]
                    if not isinstance(text, str) or not text.strip():
                        last_reason = "empty response"
                        raise ValueError(last_reason)
                    # Structured extraction cannot consume prose or fenced JSON.
                    # Validate here so malformed model output follows the same
                    # bounded retry policy as transient HTTP failures.
                    json.loads(text)
                    usage = body.get("usage", {}) if isinstance(body, dict) else {}
                    return ModelResult(
                        text=text,
                        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                        completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    )
                except ProviderError:
                    raise
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_reason = type(exc).__name__
                except (
                    httpx.HTTPStatusError,
                    json.JSONDecodeError,
                    KeyError,
                    IndexError,
                    TypeError,
                    ValueError,
                ) as exc:
                    # Exception messages can contain provider-controlled values
                    # (for example an invalid usage field). Persist only types.
                    last_reason = type(exc).__name__

                if attempt < self.retry_policy.max_attempts:
                    self.sleep(self.retry_policy.base_delay_seconds * (2 ** (attempt - 1)))

        raise ProviderRetryExhausted(
            f"模型调用在 {self.retry_policy.max_attempts} 次尝试后失败：{last_reason}"
        )
