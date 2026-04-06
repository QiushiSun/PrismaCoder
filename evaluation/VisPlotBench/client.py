from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


def _norm_base_url(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url = base_url + "/v1"
    return base_url


@dataclass
class OpenAICompatClient:
    model: str
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    timeout: float = 600.0
    max_retries: int = 6
    default_headers: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_url = _norm_base_url(self.base_url)
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        headers.update(self.default_headers)
        self._headers = headers

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)

        url = f"{self.base_url}/chat/completions"
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout, headers=self._headers) as client:
            for attempt in range(self.max_retries):
                try:
                    resp = await client.post(url, json=payload)
                    if resp.status_code in (429, 500, 502, 503, 504):
                        raise httpx.HTTPStatusError(
                            f"retryable status={resp.status_code}", request=resp.request, response=resp
                        )
                    resp.raise_for_status()
                    data = resp.json()
                    choice = data["choices"][0]["message"]
                    content = choice.get("content", "")
                    if isinstance(content, list):
                        text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
                        content = "".join(text_parts)
                    data["response_text"] = content
                    return data
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if attempt == self.max_retries - 1:
                        break
                    await asyncio.sleep(min(30.0, (2 ** attempt) + random.random()))
        raise RuntimeError(f"chat_completion failed after {self.max_retries} tries: {last_err}")


def image_to_content_part(image_b64_or_path: str, detail: str = "auto") -> dict[str, Any]:
    if os.path.exists(image_b64_or_path):
        with open(image_b64_or_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
    else:
        encoded = image_b64_or_path
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{encoded}",
            "detail": detail,
        },
    }


def now_ts() -> float:
    return time.time()
