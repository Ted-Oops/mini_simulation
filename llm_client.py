from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


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
    timeout_seconds = float(
        os.getenv("MINI_SIM_LLM_TIMEOUT")
        or local_config.get("timeout_seconds")
        or 30.0
    )
    temperature = float(
        os.getenv("MINI_SIM_LLM_TEMPERATURE")
        or local_config.get("temperature")
        or 0.2
    )
    max_tokens = int(
        os.getenv("MINI_SIM_LLM_MAX_TOKENS")
        or local_config.get("max_tokens")
        or 300
    )
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
        self.last_error: str | None = None

    @property
    def is_configured(self) -> bool:
        return bool(self.config.api_key)

    def is_available(self) -> bool:
        return self.is_configured and self.disabled_reason is None

    def complete_json(self, messages: list[dict[str, str]]) -> dict[str, Any] | None:
        if not self.config.api_key:
            self.last_error = "missing API key"
            return None
        if self.disabled_reason is not None:
            self.last_error = self.disabled_reason
            return None

        try:
            content = self._complete(messages)
            return self._extract_json_object(content)
        except LLMTransportError as exc:
            self.disabled_reason = str(exc)
            self.last_error = str(exc)
            return None
        except LLMResponseError as exc:
            self.last_error = str(exc)
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
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMTransportError(
                f"LLM API HTTP {exc.code}: {detail[:240]}"
            ) from exc
        except (TimeoutError, urllib.error.URLError, OSError) as exc:
            raise LLMTransportError(f"LLM API request failed: {exc}") from exc

        try:
            parsed_payload = json.loads(response_payload)
            return parsed_payload["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise LLMResponseError("LLM API returned an unexpected response") from exc

    @staticmethod
    def _extract_json_object(content: str) -> dict[str, Any]:
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


def get_default_llm_client() -> LLMClient | None:
    global _DEFAULT_CLIENT
    if _DEFAULT_CLIENT is None:
        _DEFAULT_CLIENT = LLMClient()
    return _DEFAULT_CLIENT
