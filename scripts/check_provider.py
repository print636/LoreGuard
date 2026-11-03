from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.provider import (
    OpenAICompatibleProvider,
    ProviderError,
    ProviderNotConfigured,
    ProviderRetryExhausted,
    ProviderToolCallError,
    safe_thinking_configuration,
)


# Keep complete() diagnostics finite, like the persisted provider categories.
# Never serialize arbitrary exception fields, response bodies or subclass names.
_SAFE_ERROR_CATEGORIES = frozenset({
    "not_configured", "provider", "rate_limit", "upstream_5xx", "unauthorized",
    "forbidden", "nonretry_http", "unsupported_content_encoding",
    "response_decompression", "body_json", "response_shape", "empty_content",
    "usage_shape", "truncated", "content_json", "connect_timeout", "read_timeout",
    "transport", "response_too_large", "credential_reflected", "endpoint_rejected",
    "credential_invalid", "credential_revoked", "invalid_response",
})
_RESPONSE_CATEGORIES = frozenset({
    "unsupported_content_encoding", "response_decompression", "body_json",
    "response_shape", "empty_content", "usage_shape", "truncated", "content_json",
    "response_too_large", "invalid_response",
})
_ERROR_HINTS = {
    "not_configured": "未配置可用模型：请检查模型功能开关和服务端凭据。",
    "transport": "无法连接模型服务；检查服务是否启动、地址、端口及网络配置。",
    "connect_timeout": "连接模型服务超时；检查网络与服务状态后重试。",
    "read_timeout": "等待模型响应超时；检查网关与上游状态后重试。",
    "unauthorized": "鉴权未通过；检查当前服务的密钥或登录授权状态，不要输出凭据。",
    "forbidden": "请求被服务端拒绝；检查当前账户、模型权限和访问策略。",
    "rate_limit": "服务限流；按服务端限制等待后重试，不要并行反复探测。",
    "upstream_5xx": "网关或上游暂不可用；检查网关、上游鉴权与网络。仅凭此状态不能认定余额不足或需要重新授权。",
    "nonretry_http": "请求被服务端拒绝；检查地址、模型名称和接口配置。",
    "endpoint_rejected": "模型服务地址未通过本地安全策略；检查精确允许地址，不要放宽到任意主机。",
    "credential_invalid": "模型凭据格式无效；在本机安全配置中检查，不要输出密钥。",
    "credential_revoked": "当前模型凭据已撤销；请核对账户配置，不借用其他账户凭据。",
    "credential_reflected": "响应触发凭据泄漏保护；停止使用该响应并在本机核查，不要发布响应正文。",
    "invalid_response": "模型响应未通过 JSON 探测合同；检查响应格式与 Provider 兼容配置。",
    "provider": "模型调用失败；根据安全分类和 HTTP 状态检查配置，不要发布上游正文或凭据。",
}


def _safe_http_status(error: ProviderError) -> int | None:
    value = error.http_status
    if type(value) is int and 100 <= value <= 599:
        return value
    if value is not None:
        return None
    # Backward compatibility for older safe ProviderError messages. Do not call
    # custom __str__, retain text, or parse a four-digit/nonstandard status.
    message = error.args[0] if error.args and type(error.args[0]) is str else ""
    match = re.search(r"\bHTTP ([1-5][0-9]{2})\b", message[:512])
    return int(match.group(1)) if match else None


def _add_failure_guidance(report: dict, category: str) -> None:
    report["category"] = category
    report["hint"] = _ERROR_HINTS[
        "invalid_response" if category in _RESPONSE_CATEGORIES else category
    ]


def check_provider(provider=None) -> tuple[dict, int]:
    """Use the production JSON path, never echo upstream bodies or credentials."""
    provider = provider if provider is not None else OpenAICompatibleProvider()
    report = {
        "configured": provider.configured,
        "json_contract_ok": False,
        "thinking": safe_thinking_configuration(
            getattr(provider, "settings", None)
        ),
    }
    if not provider.configured:
        report["error_type"] = "ProviderNotConfigured"
        _add_failure_guidance(report, "not_configured")
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
            _add_failure_guidance(report, "invalid_response")
    except ProviderError as exc:
        if isinstance(exc, ProviderNotConfigured):
            report["error_type"] = "ProviderNotConfigured"
        elif isinstance(exc, ProviderRetryExhausted):
            report["error_type"] = "ProviderRetryExhausted"
        elif isinstance(exc, ProviderToolCallError):
            report["error_type"] = "ProviderToolCallError"
        else:
            report["error_type"] = "ProviderError"
        category = (
            exc.category
            if type(exc.category) is str and exc.category in _SAFE_ERROR_CATEGORIES
            else "provider"
        )
        _add_failure_guidance(report, category)
        status = _safe_http_status(exc)
        if status is not None:
            report["http_status"] = status
    except (ValueError, TypeError, KeyError, IndexError):
        report["error_type"] = "InvalidResponse"
        _add_failure_guidance(report, "invalid_response")
    report["latency_seconds"] = round(time.perf_counter() - started, 3)
    return report, 0 if report["json_contract_ok"] else 1


def main() -> int:
    report, code = check_provider()
    print(json.dumps(report, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
