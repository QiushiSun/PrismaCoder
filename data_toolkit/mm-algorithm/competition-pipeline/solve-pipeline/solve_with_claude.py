"""
Step 3 — solve each Competition-Data problem via Claude (multimodal).

For each seed:
  1. Build a multimodal prompt:
       - up to MAX_IMAGES image data URLs
       - cleaned problem text (EN translation if seed.lang == 'jp')
       - parseable test cases (if any)
  2. Ask Claude for:
       - a Python STDIN/STDOUT solution (single fenced ```python``` block)
       - if seed.parseable_tests has < 2 entries, ALSO 3 additional test
         cases (separate fenced ```yaml``` block, deterministic format)
  3. Parse response: solution + extra_tests
  4. Cache full audit to solutions/<sid>.json

NOTE: this step does NOT verify solutions; that's the next step
      (verify_sandbox.py).  Keeping LLM-IO and execution in separate
      scripts means a sandbox crash never wastes API tokens.

Usage:
  conda run -n vpb python solve_with_claude.py --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm.asyncio import tqdm as atqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llm_client import call_llm, encode_image_to_data_url  # noqa: E402
from logger import APILogger                               # noqa: E402

BASE = Path(__file__).resolve().parent
COMP_ROOT = BASE / "seeds_sample"  # original repo: data-augmentation/Competition-Data
SEEDS_PATH = BASE / "seeds" / "records.jsonl"
TRANS_DIR = BASE / "translations"
SOL_DIR = BASE / "solutions"
API_LOG_DIR = BASE / "api_logs"

MAX_IMAGES = 6   # cap per problem to keep token usage bounded


# ────────────────────────── prompt ──────────────────────────
SYSTEM_PROMPT = """You are an expert competitive-programming solver. Your task is to
read a problem statement (with diagram), produce a working Python 3 solution
that reads from stdin and writes to stdout, and optionally generate
additional test cases.

Hard requirements:
  • Output a SINGLE Python solution as a fenced ```python``` block.
    The solution MUST read all input from stdin (input(), sys.stdin, etc.)
    and print the answer to stdout.  No `argparse`, no `pathlib`, no
    `argv`-based input.
  • The solution must be self-contained: any helper functions live inside
    the same module.  Use only the Python standard library; numpy is OK.
  • If the problem hints at large recursion depth, set
    `sys.setrecursionlimit(...)` near the top.
  • Do NOT include any commentary outside the code block (your reasoning
    is fine to include AS A leading docstring inside the code, briefly).

Test-case augmentation:
  • If the user explicitly asks for "AUGMENT_TESTS=1", you MUST also output
    a SECOND fenced block, language `yaml`, containing exactly 3 additional
    test cases obtained by RUNNING YOUR OWN SOLUTION mentally — do NOT
    invent values you cannot defend.  Format:

      ```yaml
      tests:
        - input: |
            <stdin contents, one item per line>
          output: |
            <expected stdout>
        - input: |
            ...
          output: |
            ...
        - input: |
            ...
          output: |
            ...
      ```

  • If "AUGMENT_TESTS=0", output ONLY the python block, no yaml.

  • Each test case must be solvable by your solution.  Pick small inputs
    (≤ 30 elements per array, integers ≤ 1000) — they're for verification,
    not stress tests.

Be honest: if the problem is ambiguous due to a missing diagram detail or
a Japanese term that hasn't translated cleanly, output a best-effort
solution and note your assumption in a leading code-comment line.

Style: prefer iterative over deeply recursive; vectorise with numpy only
when the problem is array-heavy.  Always close I/O with a final newline."""


USER_TEMPLATE = """## Problem ({source}, {problem_id})

Title: {title}
URL:   {url}

{problem_text}

## Existing test cases (already verified on the original judge)

{existing_tests_block}

## Configuration

AUGMENT_TESTS={augment_tests}

## Your task

Write a Python 3 STDIN/STDOUT solution.{augment_hint}

Output: ONE python block{maybe_yaml}, nothing else."""


def render_existing_tests(tests: list[dict]) -> str:
    if not tests:
        return "(none provided that survived ingestion)"
    out = []
    for i, t in enumerate(tests, 1):
        out.append(f"### Test {i}\n")
        out.append("```\n")
        out.append("Input:\n")
        out.append((t.get("input") or "").rstrip() + "\n")
        out.append("Output:\n")
        out.append((t.get("output") or "").rstrip() + "\n")
        out.append("```\n")
    return "".join(out)


# ────────────────────────── output parsing ──────────────────────────
_PY_FENCE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_PY_FENCE_GENERIC_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)
_YAML_FENCE_RE = re.compile(r"```ya?ml\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_solution(text: str) -> str:
    """Extract the LAST non-empty python block.

    LLMs sometimes iteratively rewrite the code in their chain-of-thought,
    so we want the FINAL version. We also skip the spurious empty-body match
    that arises when a python block's closing ``` is immediately followed by
    a yaml block's opening ```.
    """
    if not text:
        return ""
    candidates: list[str] = []
    for m in _PY_FENCE_RE.finditer(text):
        body = m.group(1).strip()
        if body and "import" in body or "def " in body or "print(" in body or "input(" in body:
            candidates.append(m.group(1))
        elif body:
            candidates.append(m.group(1))
    if not candidates:
        # Fallback to generic ``` blocks if no explicit python tag found
        for m in _PY_FENCE_GENERIC_RE.finditer(text):
            body = m.group(1).strip()
            if body and len(body) > 30:
                candidates.append(m.group(1))
    return candidates[-1] if candidates else ""


def extract_extra_tests(text: str) -> list[dict]:
    """Extract test cases from a yaml fenced block.  Falls back to lenient
    parsing if PyYAML isn't available."""
    m = _YAML_FENCE_RE.search(text or "")
    if not m:
        return []
    body = m.group(1)
    try:
        import yaml
        data = yaml.safe_load(body)
        if isinstance(data, dict) and isinstance(data.get("tests"), list):
            out: list[dict] = []
            for t in data["tests"]:
                if not isinstance(t, dict):
                    continue
                inp = t.get("input")
                expected = t.get("output")
                if inp is None or expected is None:
                    continue
                out.append({
                    "input": str(inp),
                    "output": str(expected),
                })
            return out
    except ImportError:
        pass

    # Lenient fallback: split into "- input: |" / "output: |" stanzas
    out_tests: list[dict] = []
    cur: dict[str, str] | None = None
    cur_field: str | None = None
    cur_lines: list[str] = []
    base_indent = 0
    lines = body.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("- input:"):
            if cur is not None and cur_field is not None:
                cur[cur_field] = "\n".join(cur_lines)
                if "input" in cur and "output" in cur:
                    out_tests.append(cur)
            cur = {}
            cur_field = "input"
            cur_lines = []
            base_indent = len(line) - len(line.lstrip())
        elif stripped.startswith("output:"):
            if cur is not None and cur_field is not None:
                cur[cur_field] = "\n".join(cur_lines)
            cur_field = "output"
            cur_lines = []
        elif cur is not None and cur_field is not None:
            # accumulate lines that are deeper indented than the field marker
            if line.strip() == "":
                cur_lines.append("")
            elif len(line) - len(line.lstrip()) > base_indent:
                # strip the leading indent
                cur_lines.append(line[base_indent + 2:] if line.startswith(" " * (base_indent + 2)) else line.lstrip())
            else:
                # next stanza
                if cur_field is not None:
                    cur[cur_field] = "\n".join(cur_lines).rstrip()
                if "input" in cur and "output" in cur:
                    out_tests.append(cur)
                cur = None
                cur_field = None
                cur_lines = []
                continue  # don't advance i; reprocess this line
        i += 1
    if cur is not None and cur_field is not None:
        cur[cur_field] = "\n".join(cur_lines).rstrip()
        if "input" in cur and "output" in cur:
            out_tests.append(cur)

    # Final cleanup: ensure newline-terminated input/output
    cleaned: list[dict] = []
    for t in out_tests:
        inp = t.get("input") or ""
        outp = t.get("output") or ""
        cleaned.append({"input": inp.rstrip("\n") + "\n", "output": outp.rstrip("\n") + "\n"})
    return cleaned


# ────────────────────────── per-problem solver ──────────────────────────
async def solve_one(
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
    resume: bool,
    img_cache: dict[str, str],
) -> dict:
    sid = seed["sid"]
    out_path = SOL_DIR / f"{sid}.json"
    if resume and out_path.exists():
        return {"sid": sid, "status": "cached"}

    # Resolve problem text (EN)
    if seed.get("lang") == "jp":
        trans_path = TRANS_DIR / f"{sid}.txt"
        if not trans_path.exists():
            return {"sid": sid, "status": "no_translation"}
        problem_text = trans_path.read_text(encoding="utf-8")
    else:
        problem_text = seed.get("problem_text_en") or ""
        if not problem_text:
            return {"sid": sid, "status": "no_text"}

    # Encode images (cap at MAX_IMAGES, cache across calls)
    image_data_urls: list[str] = []
    for rel in (seed.get("image_paths") or [])[:MAX_IMAGES]:
        if rel not in img_cache:
            img_abs = COMP_ROOT / rel
            try:
                img_cache[rel] = encode_image_to_data_url(img_abs)
            except Exception as e:
                logger.warning(f"[{sid}] image encode fail for {rel}: {e}")
                continue
        image_data_urls.append(img_cache[rel])
    if not image_data_urls:
        return {"sid": sid, "status": "no_images"}

    n_parseable = seed.get("n_parseable_tests", 0) or 0
    parseable = seed.get("parseable_tests") or []
    augment_tests = 1 if n_parseable < 2 else 0
    augment_hint = (
        f"\n\nThis problem only has {n_parseable} parseable existing test "
        "case(s) — please ALSO emit 3 additional self-consistent test cases "
        "in the YAML block as instructed."
        if augment_tests else
        f"\n\nNo extra test cases needed (we have {n_parseable} parseable existing tests)."
    )
    maybe_yaml = " followed by ONE yaml block" if augment_tests else ""

    user_text = USER_TEMPLATE.format(
        source=seed["source"],
        problem_id=seed["problem_id"],
        title=seed.get("title") or "(no title)",
        url=seed.get("url") or "(no url)",
        problem_text=problem_text,
        existing_tests_block=render_existing_tests(parseable),
        augment_tests=augment_tests,
        augment_hint=augment_hint,
        maybe_yaml=maybe_yaml,
    )

    user_content: list[dict] = []
    for i, url in enumerate(image_data_urls, 1):
        user_content.append({"type": "text", "text": f"**Diagram #{i}:**"})
        user_content.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
    user_content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    t0 = time.time()
    async with sem:
        try:
            text, meta = await call_llm(
                messages,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
                max_tokens=max_tokens,
                logger=logger,
                api_logger=api_logger,
                user_tag="comp-solve",
            )
        except Exception as e:
            logger.warning(f"[{sid}] LLM fail: {e}")
            return {"sid": sid, "status": "llm_fail", "error": str(e)}
    elapsed = round(time.time() - t0, 1)
    usage = (meta or {}).get("usage", {}) if isinstance(meta, dict) else {}

    code = extract_solution(text or "")
    extra = extract_extra_tests(text or "") if augment_tests else []

    syntax_ok = True
    syntax_msg = ""
    try:
        compile(code, f"<{sid}>", "exec")
    except SyntaxError as e:
        syntax_ok = False
        syntax_msg = f"SyntaxError: {e.msg} (line {e.lineno})"

    result = {
        "sid":             sid,
        "source":          seed["source"],
        "problem_id":      seed["problem_id"],
        "title":           seed.get("title"),
        "url":             seed.get("url"),
        "lang":            seed.get("lang"),
        "model":           model,
        "elapsed_sec":     elapsed,
        "usage":           usage,
        "augment_tests":   augment_tests,
        "n_parseable":     n_parseable,
        "raw_text":        text or "",
        "code":            code,
        "extra_tests":     extra,
        "syntax_ok":       syntax_ok,
        "syntax_msg":      syntax_msg,
        "n_images_sent":   len(image_data_urls),
        "ts":              datetime.now().isoformat(timespec="seconds"),
    }
    out_path.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    if not syntax_ok:
        logger.warning(f"[{sid}] solve_done  SYNTAX_FAIL  {syntax_msg}  elapsed={elapsed}s")
        return {"sid": sid, "status": "syntax_fail"}
    if not code.strip():
        logger.warning(f"[{sid}] solve_done  EMPTY_CODE  elapsed={elapsed}s")
        return {"sid": sid, "status": "empty_code"}
    if augment_tests and not extra:
        logger.warning(f"[{sid}] solve_done  MISSING_EXTRA_TESTS  elapsed={elapsed}s")
        return {"sid": sid, "status": "missing_extra"}

    logger.info(
        f"[{sid}] ok  code_chars={len(code)}  extra_tests={len(extra)}  "
        f"tokens={usage.get('total_tokens','?')}  elapsed={elapsed}s"
    )
    return {"sid": sid, "status": "ok"}


def setup_logging() -> logging.Logger:
    log_dir = BASE / "logs"
    log_dir.mkdir(exist_ok=True)
    lg = logging.getLogger("comp_solve")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"solve_{ts}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(fh)
    return lg


async def amain(args):
    SOL_DIR.mkdir(exist_ok=True)
    logger = setup_logging()

    seeds: list[dict] = []
    with SEEDS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seeds.append(json.loads(line))
    if args.source:
        seeds = [r for r in seeds if r.get("source") == args.source]
    if args.limit:
        seeds = seeds[: args.limit]
    print(f"Solving {len(seeds)} seeds  concurrency={args.concurrency}")

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://az.gptplus5.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")
    api_logger = APILogger(API_LOG_DIR)

    img_cache: dict[str, str] = {}
    sem = asyncio.Semaphore(args.concurrency)

    tasks = [
        asyncio.create_task(solve_one(
            r, sem=sem, api_logger=api_logger,
            model=args.model, base_url=base_url, api_key=api_key,
            temperature=args.temperature, max_tokens=args.max_tokens,
            logger=logger, resume=args.resume, img_cache=img_cache,
        ))
        for r in seeds
    ]

    by_status: dict[str, int] = {}
    pbar = atqdm(total=len(tasks), desc="solve", unit="rec")
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
        "n":         len(seeds),
        "by_status": by_status,
        "model":     args.model,
    }
    (BASE / "solve_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n{summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-sonnet-4-6-aws")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--source", default=None,
                    help="optional filter: codeforces / atcoder / aizu")
    ap.add_argument("--limit", type=int, default=0,
                    help="optional cap on seeds (0 = all)")
    ap.add_argument("--resume", action="store_true", default=True,
                    help="skip seeds whose solution file already exists")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
