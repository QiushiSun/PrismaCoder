"""
Multimodal Chat-Completions client (OpenAI-compatible gateway).

Supports image_url + text content blocks and writes API traces to a local
`api-logs/` directory inside this pipeline folder.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import random
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logger import APILogger  # noqa: E402

API_LOG_DIR = Path(__file__).resolve().parent / "api-logs"


def encode_image_to_data_url(img_path: Path | str, max_side: int = 1280, quality: int = 88) -> str:
    """Resize-if-large + JPEG-encode + base64 wrap into a data URL.

    Resizing is essential: HumanEval-V originals are ~1024×1024 RGBA which
    explodes prompt tokens; we down-scale to 1280-max-side JPEG to keep ~50KB
    per image while preserving readability of small numbers/labels.
    """
    img = Image.open(img_path)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


async def call_llm(
    messages: list[dict],
    *,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float = 0.7,
    max_tokens: int = 8192,
    max_retries: int = 6,
    logger: logging.Logger | None = None,
    api_logger: APILogger | None = None,
    user_tag: str = "humanevalv-aug",
) -> tuple[str, dict]:
    """Returns (content, metadata). Handles retry on 429/5xx and connection errors."""
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    url += "/chat/completions"

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with httpx.AsyncClient(timeout=600.0, headers=headers, follow_redirects=True) as client:
        for attempt in range(max_retries):
            t0 = time.time()
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code in (429, 500, 502, 503, 504):
                    wait = min(60.0, (2 ** attempt) + random.random() * 2)
                    if logger:
                        logger.warning(
                            f"[{model}] retryable status {resp.status_code}, "
                            f"waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})"
                        )
                    await asyncio.sleep(wait)
                    continue
                if resp.status_code == 403:
                    raise RuntimeError(f"403 Forbidden: {resp.text[:300]}")
                resp.raise_for_status()
                data = resp.json()
                duration_ms = (time.time() - t0) * 1000
                msg = data["choices"][0]["message"]
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
                usage = data.get("usage", {})
                if api_logger:
                    try:
                        api_logger.log_call(
                            model=model,
                            request_data={
                                "messages": [
                                    {"role": m["role"], "content_preview": _preview_content(m["content"])}
                                    for m in messages
                                ]
                            },
                            response_data=data,
                            user=user_tag,
                            duration_ms=duration_ms,
                            success=True,
                        )
                    except Exception:
                        pass
                return content, {"usage": usage, "duration_ms": duration_ms}
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError) as e:
                wait = min(60.0, (2 ** attempt) + random.random() * 2)
                if logger:
                    logger.warning(
                        f"[{model}] network error: {type(e).__name__}: {e}, "
                        f"waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})"
                    )
                await asyncio.sleep(wait)
            except RuntimeError:
                raise
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"LLM call failed after {max_retries} retries: {e}") from e
                wait = min(60.0, (2 ** attempt) + random.random() * 2)
                if logger:
                    logger.warning(
                        f"[{model}] error {type(e).__name__}: {e}, "
                        f"waiting {wait:.1f}s (attempt {attempt+1}/{max_retries})"
                    )
                await asyncio.sleep(wait)
    raise RuntimeError(f"LLM call exhausted {max_retries} retries")


def _preview_content(content: Any, max_len: int = 200) -> str:
    if isinstance(content, str):
        return content[:max_len] + ("…" if len(content) > max_len else "")
    if isinstance(content, list):
        parts = []
        for p in content:
            if not isinstance(p, dict):
                continue
            t = p.get("type")
            if t == "text":
                txt = p.get("text", "")
                parts.append(f"[text] {txt[:120]}{'…' if len(txt) > 120 else ''}")
            elif t == "image_url":
                url = p.get("image_url", {}).get("url", "")
                parts.append(f"[image_url] {url[:60]}…")
        return " | ".join(parts)[:max_len]
    return str(content)[:max_len]
