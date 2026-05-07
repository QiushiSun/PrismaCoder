"""
Step 2 — translate Japanese problem statements to English.

For every seed with lang=='jp', send Claude a text-only translation request
and cache the EN result to translations/<sid>.txt.

Resume-aware: skips sids whose translation file already exists.

Usage:
  conda run -n vpb python translate_jp.py --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import call_llm  # noqa: E402
from logger import APILogger     # noqa: E402

BASE = Path(__file__).resolve().parent
API_LOG_DIR = BASE / "api_logs"
TRANS_DIR = BASE / "translations"
SEED_PATH = BASE / "seeds" / "records.jsonl"

SYSTEM_PROMPT = """You are a precise translator for competitive-programming problems.
Translate the user's Japanese problem statement into clear, idiomatic English.

Rules:
  • Preserve ALL inline code, formula tokens, mathematical notation
    (e.g. $$$x_a$$$, x^2, |a-b|, mod 998244353) verbatim.
  • Preserve numbered constraints / structural sections like "入力", "出力",
    "制約", "Constraints" — translate the section heading to English but keep
    the structure (Input / Output / Constraints / Examples).
  • Preserve example I/O blocks exactly — do NOT translate the numerals or
    swap whitespace inside them.
  • Do NOT add commentary, do NOT wrap in code fences.
  • Output ONLY the English translation, nothing else."""


def setup_logging() -> logging.Logger:
    log_dir = BASE / "logs"
    log_dir.mkdir(exist_ok=True)
    lg = logging.getLogger("comp_translate")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"translate_{ts}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(fh)
    return lg


async def translate_one(
    rec: dict,
    *,
    sem: asyncio.Semaphore,
    api_logger: APILogger,
    model: str,
    base_url: str,
    api_key: str,
    logger: logging.Logger,
) -> dict:
    sid = rec["sid"]
    out_path = TRANS_DIR / f"{sid}.txt"
    if out_path.exists():
        return {"sid": sid, "status": "cached"}

    text_jp = rec.get("problem_text_jp") or ""
    if not text_jp:
        return {"sid": sid, "status": "empty_jp"}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text_jp},
    ]

    t0 = time.time()
    async with sem:
        try:
            text, meta = await call_llm(
                messages,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=0.2,
                max_tokens=4096,
                logger=logger,
                api_logger=api_logger,
                user_tag="comp-translate",
            )
        except Exception as e:
            logger.warning(f"[{sid}] translate fail: {e}")
            return {"sid": sid, "status": "llm_fail", "error": str(e)}
    elapsed = round(time.time() - t0, 1)
    text = (text or "").strip()
    if not text:
        return {"sid": sid, "status": "empty_response"}

    out_path.write_text(text, encoding="utf-8")
    usage = (meta or {}).get("usage", {}) if isinstance(meta, dict) else {}
    logger.info(
        f"[{sid}] ok  jp_chars={len(text_jp)} en_chars={len(text)} "
        f"tokens={usage.get('total_tokens','?')} elapsed={elapsed}s"
    )
    return {"sid": sid, "status": "ok"}


async def amain(args):
    TRANS_DIR.mkdir(exist_ok=True)
    logger = setup_logging()

    seeds: list[dict] = []
    with SEED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    jp_seeds = [r for r in seeds if r.get("lang") == "jp"]
    print(f"Loaded {len(seeds)} seeds, {len(jp_seeds)} need translation")

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://az.gptplus5.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    api_logger = APILogger(API_LOG_DIR)

    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(translate_one(
            r, sem=sem, api_logger=api_logger,
            model=args.model, base_url=base_url, api_key=api_key, logger=logger,
        ))
        for r in jp_seeds
    ]

    by_status: dict[str, int] = {}
    pbar = atqdm(total=len(tasks), desc="translate", unit="rec")
    for fut in asyncio.as_completed(tasks):
        r = await fut
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        pbar.update(1)
        pbar.set_postfix(
            ok=by_status.get("ok", 0),
            cached=by_status.get("cached", 0),
            fail=by_status.get("llm_fail", 0) + by_status.get("empty_response", 0),
        )
    pbar.close()

    summary = {
        "ts":         datetime.now().isoformat(timespec="seconds"),
        "n_jp":       len(jp_seeds),
        "by_status":  by_status,
        "model":      args.model,
    }
    (BASE / "translation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6-aws")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
