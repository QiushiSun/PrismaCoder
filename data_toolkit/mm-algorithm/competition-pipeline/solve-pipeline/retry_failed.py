"""
Step 4 — retry failed solutions with a stronger prompt + lower temperature.

Inputs:
  failed/<sid>.json            — sandbox-rejected solutions
  pre-skipped solutions (empty_code / syntax_fail / missing_extra / no_tests)
  llm_fail sids (no solution file exists)

For each candidate, we rebuild a multimodal prompt with:
  • a tougher system prompt that demands "this is the second attempt"
  • the original problem + diagrams
  • forced AUGMENT_TESTS=1 (always emit yaml block — we can't trust parseable
    existing tests if the first attempt couldn't pass them)
  • temperature 0.2, max_tokens 16k

Output:
  retry_solutions/<sid>.json    — same schema as solutions/
  retry_summary.json

After this, run recover_solutions.py (which works on solutions/ AND
retry_solutions/) and re-verify.

Usage:
  conda run -n vpb python retry_failed.py --concurrency 32
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

from llm_client import call_llm, encode_image_to_data_url   # noqa: E402
from logger import APILogger                                # noqa: E402
from solve_with_claude import (                             # noqa: E402
    extract_solution, extract_extra_tests,
    render_existing_tests, MAX_IMAGES,
)

BASE = Path(__file__).resolve().parent
COMP_ROOT = BASE / "seeds_sample"
SEEDS_PATH = BASE / "seeds" / "records.jsonl"
TRANS_DIR = BASE / "translations"
SOL_DIR = BASE / "solutions"
RETRY_DIR = BASE / "retry_solutions"
VER_DIR = BASE / "verified"
FAIL_DIR = BASE / "failed"
API_LOG_DIR = BASE / "api_logs"


RETRY_SYSTEM_PROMPT = """You are an expert competitive-programming solver. The previous
attempt at this problem either crashed, produced no Python code, or gave wrong
answers on at least one example. Read the problem carefully, paying close
attention to:

  • input/output FORMAT (one item per line vs. one big line)
  • edge cases (zero-size arrays, single elements, max bounds)
  • whether this is an INTERACTIVE problem — if it is, output a Python file
    that flushes stdout after each query (or print "INTERACTIVE_NOT_SUPPORTED"
    in a comment and provide a best-effort offline solution)
  • whether the diagram conveys structural information that the prose only
    hints at (graph layout, geometry, sequence indexing convention).

Your output MUST be:
  1. A SINGLE fenced ```python``` block — the complete solution, reading
     stdin, printing to stdout, no I/O wrappers, only stdlib + numpy.
  2. THEN a SINGLE fenced ```yaml``` block holding 3 self-consistent test
     cases that you VERIFIED by running through your solution mentally:

       ```yaml
       tests:
         - input: |
             <stdin lines>
           output: |
             <stdout lines>
         - input: |
             ...
           output: |
             ...
         - input: |
             ...
           output: |
             ...
       ```

  Rules for the test cases:
    • Pick small inputs (each array ≤ 30 entries, integers ≤ 1000).
    • Each input MUST be solvable by YOUR solution.
    • Multi-line stdin MUST use literal newlines under `input: |` (real
      newlines, not the two characters '\\n').
    • Output formatting MUST match the problem's specification exactly
      (including trailing newline if specified).

No prose outside the two code blocks. No explanations, no headers, no
"Looking at this problem...". Be terse. Output the python block, then the
yaml block, in that order. Then stop."""


USER_TEMPLATE_RETRY = """## Problem ({source}, {problem_id})

Title: {title}

{problem_text}

## Existing test cases (parseable subset)

{existing_tests_block}

## Your task (RETRY — be careful with input format and edge cases)

Output ONE python block, then ONE yaml block with 3 augmented test cases.
Nothing else."""


async def retry_one(
    seed: dict,
    *,
    sem: asyncio.Semaphore,
    api_logger: APILogger,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    logger: logging.Logger,
    img_cache: dict[str, str],
) -> dict:
    sid = seed["sid"]
    out_path = RETRY_DIR / f"{sid}.json"
    if out_path.exists():
        return {"sid": sid, "status": "cached"}

    if seed.get("lang") == "jp":
        trans_path = TRANS_DIR / f"{sid}.txt"
        if not trans_path.exists():
            return {"sid": sid, "status": "no_translation"}
        problem_text = trans_path.read_text(encoding="utf-8")
    else:
        problem_text = seed.get("problem_text_en") or ""
        if not problem_text:
            return {"sid": sid, "status": "no_text"}

    image_data_urls: list[str] = []
    for rel in (seed.get("image_paths") or [])[:MAX_IMAGES]:
        if rel not in img_cache:
            img_abs = COMP_ROOT / rel
            try:
                img_cache[rel] = encode_image_to_data_url(img_abs)
            except Exception as e:
                logger.warning(f"[{sid}] image encode fail: {e}")
                continue
        image_data_urls.append(img_cache[rel])
    if not image_data_urls:
        return {"sid": sid, "status": "no_images"}

    parseable = seed.get("parseable_tests") or []
    user_text = USER_TEMPLATE_RETRY.format(
        source=seed["source"],
        problem_id=seed["problem_id"],
        title=seed.get("title") or "(no title)",
        problem_text=problem_text,
        existing_tests_block=render_existing_tests(parseable),
    )

    user_content: list[dict] = []
    for i, url in enumerate(image_data_urls, 1):
        user_content.append({"type": "text", "text": f"**Diagram #{i}:**"})
        user_content.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
    user_content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": RETRY_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    t0 = time.time()
    async with sem:
        try:
            text, meta = await call_llm(
                messages,
                model=model, base_url=base_url, api_key=api_key,
                temperature=temperature, max_tokens=max_tokens,
                logger=logger, api_logger=api_logger, user_tag="comp-retry",
            )
        except Exception as e:
            logger.warning(f"[{sid}] retry LLM fail: {e}")
            return {"sid": sid, "status": "llm_fail", "error": str(e)}
    elapsed = round(time.time() - t0, 1)
    usage = (meta or {}).get("usage", {}) if isinstance(meta, dict) else {}
    code = extract_solution(text or "")
    extra = extract_extra_tests(text or "")
    syntax_ok = True
    syntax_msg = ""
    try:
        compile(code, f"<{sid}>", "exec")
    except SyntaxError as e:
        syntax_ok = False
        syntax_msg = f"SyntaxError: {e.msg} (line {e.lineno})"

    out = {
        "sid":           sid,
        "source":        seed["source"],
        "problem_id":    seed["problem_id"],
        "title":         seed.get("title"),
        "url":           seed.get("url"),
        "lang":          seed.get("lang"),
        "model":         model,
        "elapsed_sec":   elapsed,
        "usage":         usage,
        "augment_tests": 1,
        "n_parseable":   seed.get("n_parseable_tests", 0),
        "raw_text":      text or "",
        "code":          code,
        "extra_tests":   extra,
        "syntax_ok":     syntax_ok,
        "syntax_msg":    syntax_msg,
        "n_images_sent": len(image_data_urls),
        "ts":            datetime.now().isoformat(timespec="seconds"),
        "is_retry":      True,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    if not syntax_ok:
        return {"sid": sid, "status": "syntax_fail"}
    if not code.strip():
        return {"sid": sid, "status": "empty_code"}
    if not extra:
        return {"sid": sid, "status": "missing_extra"}

    logger.info(
        f"[{sid}] retry-ok  code_chars={len(code)}  extra={len(extra)}  "
        f"tokens={usage.get('total_tokens','?')}  elapsed={elapsed}s"
    )
    return {"sid": sid, "status": "ok"}


def setup_logging() -> logging.Logger:
    log_dir = BASE / "logs"
    log_dir.mkdir(exist_ok=True)
    lg = logging.getLogger("comp_retry")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"retry_{ts}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(fh)
    return lg


def collect_retry_candidates() -> set[str]:
    """All sids that need a retry: failed verify + pre-skipped + missing files."""
    all_seeds = set()
    with SEEDS_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            all_seeds.add(r["sid"])

    candidates: set[str] = set()

    # 1. Sandbox-failed
    for p in FAIL_DIR.glob("*.json"):
        candidates.add(p.stem)

    # 2. Solutions that exist but didn't pass pre-checks
    #    (empty_code / syntax_fail / missing_extra / no_tests)
    sol_sids = {p.stem for p in SOL_DIR.glob("*.json")}
    ver_sids = {p.stem for p in VER_DIR.glob("*.json")}
    fail_sids = {p.stem for p in FAIL_DIR.glob("*.json")}
    pre_skipped = sol_sids - ver_sids - fail_sids
    candidates.update(pre_skipped)

    # 3. Seeds with no solution file at all (llm_fail or never reached)
    missing = all_seeds - sol_sids
    candidates.update(missing)

    return candidates


async def amain(args):
    RETRY_DIR.mkdir(exist_ok=True)
    logger = setup_logging()

    seeds: dict[str, dict] = {}
    with SEEDS_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            seeds[r["sid"]] = r

    candidates = collect_retry_candidates()
    candidate_seeds = [seeds[sid] for sid in sorted(candidates) if sid in seeds]
    if args.limit:
        candidate_seeds = candidate_seeds[: args.limit]
    print(f"Retrying {len(candidate_seeds)} solutions  concurrency={args.concurrency}")

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://az.gptplus5.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_logger = APILogger(API_LOG_DIR)

    img_cache: dict[str, str] = {}
    sem = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(retry_one(
            r, sem=sem, api_logger=api_logger,
            model=args.model, base_url=base_url, api_key=api_key,
            temperature=args.temperature, max_tokens=args.max_tokens,
            logger=logger, img_cache=img_cache,
        ))
        for r in candidate_seeds
    ]

    by_status: dict[str, int] = {}
    pbar = atqdm(total=len(tasks), desc="retry", unit="rec")
    for fut in asyncio.as_completed(tasks):
        r = await fut
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        pbar.update(1)
        pbar.set_postfix(
            ok=by_status.get("ok", 0),
            cached=by_status.get("cached", 0),
            sfail=by_status.get("syntax_fail", 0) + by_status.get("empty_code", 0)
                  + by_status.get("missing_extra", 0),
            llmfail=by_status.get("llm_fail", 0),
        )
    pbar.close()

    summary = {
        "ts":        datetime.now().isoformat(timespec="seconds"),
        "n":         len(candidate_seeds),
        "by_status": by_status,
        "model":     args.model,
    }
    (BASE / "retry_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6-aws")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
