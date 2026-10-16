from __future__ import annotations

import json
import http.client
import os
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from validation import require_integer, require_number


DEFAULT_LLM_BASE_URL = "http://172.20.16.1:60500"
DEFAULT_LLM_MODEL = "gpt-5.5"


class LLMClientError(RuntimeError):
    pass


class LLMTransportError(LLMClientError):
    pass


class LLMResponseError(LLMClientError):
    pass


@dataclass(frozen=True)
class LLMConfig:
    api_key: str | None
    base_url: str = DEFAULT_LLM_BASE_URL
    model: str = DEFAULT_LLM_MODEL
    temperature: float = 0.2
    max_tokens: int = 300
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        if self.api_key is not None and (
            not isinstance(self.api_key, str) or not self.api_key.strip()
        ):
            raise ValueError("api_key must be a nonempty string or None")
        if not isinstance(self.base_url, str):
            raise ValueError("base_url must be an HTTP(S) URL")
        try:
            url = urlsplit(self.base_url)
            valid_url = url.scheme in ("http", "https") and bool(url.hostname)
            url.port
        except ValueError:
            valid_url = False
        if not valid_url or any(c.isspace() for c in self.base_url) or url.query or url.fragment:
            raise ValueError("base_url must be an HTTP(S) URL without query or fragment")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a nonempty string")
        require_number("timeout_seconds", self.timeout_seconds, minimum=0, exclusive=True)
        require_number("temperature", self.temperature, minimum=0)
        if self.temperature > 2:
            raise ValueError("temperature must be <= 2")
        require_integer("max_tokens", self.max_tokens, minimum=1)


def _load_local_api_config() -> dict[str, Any]:
    try:
        from local_api_config import API_CONFIG
    except ModuleNotFoundError:
        return {}

    if not isinstance(API_CONFIG, dict):
        return {}
    return API_CONFIG


def load_llm_config() -> LLMConfig:
    local_config = _load_local_api_config()
    api_key = (
        os.getenv("MINI_SIM_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or local_config.get("api_key")
    )
    base_url = (
        os.getenv("MINI_SIM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or local_config.get("base_url")
        or DEFAULT_LLM_BASE_URL
    )
    model = os.getenv("MINI_SIM_MODEL") or local_config.get("model") or DEFAULT_LLM_MODEL

    def setting(env_name: str, key: str, default: Any) -> Any:
        value = os.getenv(env_name)
        return local_config.get(key, default) if value is None else value

    def numeric_setting(env_name: str, key: str, default: Any, convert: Any) -> Any:
        value = setting(env_name, key, default)
        try:
            if isinstance(value, bool):
                raise ValueError
            if convert is int and not isinstance(value, (str, int)):
                raise ValueError
            return convert(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{env_name} / {key} has an invalid numeric value") from None

    timeout_seconds = numeric_setting("MINI_SIM_LLM_TIMEOUT", "timeout_seconds", 30.0, float)
    temperature = numeric_setting("MINI_SIM_LLM_TEMPERATURE", "temperature", 0.2, float)
    max_tokens = numeric_setting("MINI_SIM_LLM_MAX_TOKENS", "max_tokens", 300, int)
    return LLMConfig(
        api_key=api_key,
        base_url=str(base_url).rstrip("/"),
        model=str(model),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
    )


class LLMClient:
    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or load_llm_config()
        self.disabled_reason: str | None = None
        self._request_state = threading.local()

    @property
    def last_error(self) -> str | None:
        return getattr(self._request_state, "last_error", None)

    @last_error.setter
    def last_error(self, value: str | None) -> None:
        self._request_state.last_error = value

    @property
    def last_error_code(self) -> str | None:
        return getattr(self._request_state, "last_error_code", None)

    @last_error_code.setter
    def last_error_code(self, value: str | None) -> None:
        self._request_state.last_error_code = value

    @property
    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    def is_available(self) -> bool:
        return self.is_configured and self.disabled_reason is None

    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        self.last_error = None
        self.last_error_code = None
        if not self.config.api_key:
            self.last_error = "missing API key"
            self.last_error_code = "missing_api_key"
            return None
        if self.disabled_reason is not None:
            self.last_error = self.disabled_reason
            self.last_error_code = "client_disabled"
            return None

        try:
            content = self._complete(messages)
            return self._extract_json_object(content)
        except LLMTransportError as exc:
            self.disabled_reason = str(exc)
            self.last_error = str(exc)
            self.last_error_code = "transport_error"
            return None
        except LLMResponseError as exc:
            self.last_error = str(exc)
            self.last_error_code = "response_error"
            return None

    def _complete(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        encoded_payload = json.dumps(payload).encode("utf-8")

        last_error: LLMTransportError | None = None
        for url in self._completion_urls():
            try:
                return self._post_completion(url, encoded_payload)
            except LLMTransportError as exc:
                last_error = exc
                if "HTTP 404" not in str(exc):
                    break

        if last_error is not None:
            raise last_error
        raise LLMTransportError("LLM API request failed: no endpoint URL available")

    def _completion_urls(self) -> list[str]:
        if self.config.base_url.endswith("/v1"):
            return [f"{self.config.base_url}/chat/completions"]
        return [
            f"{self.config.base_url}/v1/chat/completions",
            f"{self.config.base_url}/chat/completions",
        ]

    def _post_completion(self, url: str, encoded_payload: bytes) -> str:
        request = urllib.request.Request(
            url=url,
            data=encoded_payload,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                response_payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise LLMTransportError(f"LLM API HTTP {exc.code}") from exc
        except (TimeoutError, urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            raise LLMTransportError(f"LLM API request failed: {exc}") from exc
        except UnicodeDecodeError as exc:
            raise LLMResponseError("LLM API returned invalid UTF-8") from exc

        try:
            parsed_payload = json.loads(response_payload)
            return parsed_payload["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("LLM API returned an unexpected response") from exc

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any]:
        if not isinstance(content, str):
            raise LLMResponseError("LLM response content must be text")
        text = content.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise LLMResponseError("LLM response did not contain a JSON object")

        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMResponseError("LLM response JSON could not be parsed") from exc

        if not isinstance(payload, dict):
            raise LLMResponseError("LLM response JSON must be an object")
        return payload


_DEFAULT_CLIENT: LLMClient | None = None
_DEFAULT_CLIENT_LOCK = threading.Lock()


def get_default_llm_client() -> LLMClient:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        with _DEFAULT_CLIENT_LOCK:
            if _DEFAULT_CLIENT is None:
                _DEFAULT_CLIENT = LLMClient()
    return _DEFAULT_CLIENT
