"""
Cross-source HARD problem-template generator.

Differences from leetcode-newgen:
  • Difficulty distribution biased to Hard (5 / 25 / 70 vs. 10 / 40 / 50).
  • Uses ingredients.json (extended algo tag set + more domains).
  • Uses TWO seed pools:
      seeds/icl_records.jsonl       — LC-native + LC-newgen Hard (ICL pool)
      seeds/inspiration_records.jsonl — CF/AC/Aizu/MMCode_full (style cue)
  • Each prompt may include 0 or 1 inspiration cue (50% prob).

All other behaviours (validation, retry, audit JSON, records.jsonl) are
identical to leetcode-newgen so the run/audit/build scripts can be reused.

Usage:
  conda run -n vpb python generate_templates.py --target 500 --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from tqdm.asyncio import tqdm as atqdm

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from llm_client import call_llm, encode_image_to_data_url     # noqa: E402
from logger import APILogger                                  # noqa: E402
from prompts import build_messages_cross                       # noqa: E402

API_LOG_DIR = BASE / "api-logs"

REQUIRED_SYMBOLS = (
    "TITLE", "DESCRIPTION", "METHOD_NAME", "SOLUTION_SIGNATURE",
    "Solution", "sample_examples", "render", "make_test_cases",
)

# Cross-source recipe weights
DIFFICULTY_MIX = {"Easy": 0.05, "Medium": 0.25, "Hard": 0.70}
N_EXAMPLES_MIX = {1: 0.10, 2: 0.70, 3: 0.20}
INSP_PROB = 0.55  # probability of attaching an inspiration record


def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("cross_newgen")
    lg.setLevel(logging.INFO)
    if not lg.handlers:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fh = logging.FileHandler(log_dir / f"newgen_{ts}.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        lg.addHandler(fh)
    return lg


# ── recipe sampling ──────────────────────────────────────────────────────────
def sample_recipe(ingredients: dict, *, rng: random.Random) -> dict:
    diff = rng.choices(list(DIFFICULTY_MIX.keys()),
                       weights=list(DIFFICULTY_MIX.values()), k=1)[0]
    n_examples = rng.choices(list(N_EXAMPLES_MIX.keys()),
                              weights=list(N_EXAMPLES_MIX.values()), k=1)[0]
    visual = rng.choice(ingredients["visual_aspects"])

    algo_pool = list(ingredients["algo_tags"])
    struct_pool = list(ingredients["struct_tags"])
    rng.shuffle(algo_pool); rng.shuffle(struct_pool)
    n_algo = rng.choice([1, 2, 2])
    n_struct = rng.choice([1, 2])
    algo_tags = algo_pool[:n_algo]
    struct_tags = struct_pool[:n_struct]
    domain = rng.choice(ingredients["domains"])

    return {
        "difficulty":     diff,
        "algo_tags":      algo_tags,
        "struct_tags":    struct_tags,
        "visual_aspect":  visual,
        "domain":         domain,
        "n_examples":     n_examples,
    }


# ── ICL matching ─────────────────────────────────────────────────────────────
def _tag_set(rec: dict) -> set[str]:
    return set(rec.get("tags") or [])


def match_icl_examples(seeds: list[dict], recipe: dict, rng: random.Random, k: int = 2) -> list[dict]:
    """Pick k ICL examples by:
        + 3 × tag overlap
        + 2 × difficulty match
        + Hard records preferred when recipe.difficulty == Hard
    """
    target = set(recipe.get("algo_tags") or []) | set(recipe.get("struct_tags") or [])
    target_diff = recipe["difficulty"]
    scored: list[tuple[float, dict]] = []
    for r in seeds:
        tag_overlap = len(target & _tag_set(r))
        diff_match = 1 if r.get("difficulty") == target_diff else 0
        # Bias toward Hard when recipe is Hard
        hard_bonus = 1 if (target_diff == "Hard" and r.get("difficulty") == "Hard") else 0
        score = tag_overlap * 3 + diff_match * 2 + hard_bonus + rng.random() * 0.5
        scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    pool = [r for _, r in scored[: max(k * 8, 16)]]
    if len(pool) <= k:
        return pool
    return rng.sample(pool, k)


def maybe_pick_inspiration(insp_seeds: list[dict], rng: random.Random) -> dict | None:
    if not insp_seeds or rng.random() > INSP_PROB:
        return None
    return rng.choice(insp_seeds)


# ── output extraction ───────────────────────────────────────────────────────
_OPEN_FENCE_RE = re.compile(r"```(?:python)?[ \t]*\n", re.MULTILINE)


def strip_code_fence(text: str) -> str:
    if not text:
        return ""
    m = _OPEN_FENCE_RE.search(text)
    if not m:
        return text.strip()
    body_start = m.end()
    last_close = text.rfind("```")
    if last_close > body_start:
        return text[body_start:last_close]
    return text[body_start:].strip()


def has_required_symbols(src: str) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for sym in REQUIRED_SYMBOLS:
        if not re.search(
            rf"^(def\s+{sym}\b|class\s+{sym}\b|{sym}\s*[:=])",
            src, re.MULTILINE,
        ):
            missing.append(sym)
    return (not missing), missing


def has_input_output_placeholders(src: str, n_examples: int) -> tuple[bool, str]:
    m = re.search(r"DESCRIPTION\s*=\s*[rRbB]?([\"']{3})(.*?)\1", src, re.DOTALL)
    if not m:
        return False, "DESCRIPTION not found / not triple-quoted"
    body = m.group(2)
    missing = []
    for k in range(1, n_examples + 1):
        if f"__INPUT_{k}__" not in body:
            missing.append(f"__INPUT_{k}__")
        if f"__OUTPUT_{k}__" not in body:
            missing.append(f"__OUTPUT_{k}__")
    if missing:
        return False, f"missing placeholders: {missing}"
    return True, ""


# ── per-problem driver ──────────────────────────────────────────────────────
async def process_one(
    *,
    newgen_id: str,
    recipe: dict,
    icl: list[dict],
    inspiration: dict | None,
    rollout_dir: Path,
    sem: asyncio.Semaphore,
    api_logger: APILogger,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    logger: logging.Logger,
    resume: bool,
    retry_errors: list[dict] | None = None,
) -> dict:
    tpl_path = rollout_dir / "templates" / f"{newgen_id}.py"
    audit_path = rollout_dir / "audit" / f"{newgen_id}.json"

    if resume and tpl_path.exists() and audit_path.exists() and not retry_errors:
        return {"id": newgen_id, "status": "cached"}

    messages = build_messages_cross(recipe=recipe, icl_examples=icl,
                                    inspiration=inspiration,
                                    retry_errors=retry_errors)

    t0 = time.time()
    async with sem:
        try:
            text, meta = await call_llm(
                messages, model=model, base_url=base_url, api_key=api_key,
                temperature=temperature, max_tokens=max_tokens,
                logger=logger, api_logger=api_logger, user_tag="cross-newgen",
            )
        except Exception as e:
            logger.warning(f"[{newgen_id}] LLM call failed: {e}")
            return {"id": newgen_id, "status": "llm_fail", "error": str(e)}
    elapsed = round(time.time() - t0, 1)
    usage = (meta or {}).get("usage", {}) if isinstance(meta, dict) else {}

    code = strip_code_fence(text or "")
    syntax_ok = True
    syntax_msg = ""
    try:
        compile(code, f"<{newgen_id}>", "exec")
    except SyntaxError as e:
        syntax_ok = False
        syntax_msg = f"SyntaxError: {e.msg} (line {e.lineno})"

    symbols_ok, missing = has_required_symbols(code)
    placeholders_ok, ph_msg = has_input_output_placeholders(code, recipe["n_examples"])

    audit = {
        "id": newgen_id,
        "model": model,
        "difficulty": recipe["difficulty"],
        "algo_tags": recipe["algo_tags"],
        "struct_tags": recipe["struct_tags"],
        "visual_aspect": recipe["visual_aspect"],
        "domain": recipe["domain"],
        "n_examples": recipe["n_examples"],
        "icl_qids": [r["qid"] for r in icl],
        "inspiration_qid": inspiration.get("qid") if inspiration else None,
        "inspiration_source": inspiration.get("source") if inspiration else None,
        "elapsed_sec": elapsed,
        "usage": usage,
        "raw_chars": len(text or ""),
        "stripped_chars": len(code),
        "syntax_ok": syntax_ok,
        "syntax_msg": syntax_msg,
        "symbols_ok": symbols_ok,
        "missing_symbols": missing,
        "placeholders_ok": placeholders_ok,
        "placeholders_msg": ph_msg,
        "retry": retry_errors is not None,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }

    if not (syntax_ok and symbols_ok and placeholders_ok):
        fail_dir = rollout_dir / "failed_templates"
        fail_dir.mkdir(exist_ok=True)
        (fail_dir / f"{newgen_id}.txt").write_text(text or "", encoding="utf-8")
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.warning(
            f"[{newgen_id}] FAIL  syntax_ok={syntax_ok}  symbols_ok={symbols_ok}  "
            f"ph_ok={placeholders_ok}  missing={missing}  ph_msg={ph_msg}"
        )
        return {"id": newgen_id, "status": "parse_fail"}

    tpl_path.write_text(code, encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        f"[{newgen_id}] ok  diff={recipe['difficulty']}  "
        f"visual={recipe['visual_aspect'][:30]}  insp={'yes' if inspiration else 'no'}  "
        f"chars={len(code)}  tokens={usage.get('total_tokens', '?')}  elapsed={elapsed}s"
    )
    return {"id": newgen_id, "status": "ok"}


# ── main ────────────────────────────────────────────────────────────────────
async def amain(args):
    rollout_dir = BASE
    for sub in ("templates", "audit", "failed_templates", "logs"):
        (rollout_dir / sub).mkdir(exist_ok=True)
    logger = setup_logging(rollout_dir / "logs")

    ing_path = BASE / "ingredients.json"
    if not ing_path.exists():
        logger.error("ingredients.json missing — run extract_ingredients.py")
        sys.exit(1)
    ingredients = json.loads(ing_path.read_text(encoding="utf-8"))

    icl_path = BASE / "seeds" / "icl_records.jsonl"
    insp_path = BASE / "seeds" / "inspiration_records.jsonl"
    if not icl_path.exists():
        logger.error(f"missing {icl_path}")
        sys.exit(1)
    icl_seeds: list[dict] = [json.loads(l) for l in icl_path.open() if l.strip()]
    insp_seeds: list[dict] = [json.loads(l) for l in insp_path.open() if l.strip()] if insp_path.exists() else []
    print(f"Loaded ICL pool: {len(icl_seeds)}, inspiration pool: {len(insp_seeds)}")

    image_cache: dict[str, str] = {}
    def get_data_url(rec: dict) -> str:
        qid = rec["qid"]
        if qid not in image_cache:
            image_cache[qid] = encode_image_to_data_url(
                rollout_dir / "seeds" / rec["image_path"]
            )
        return image_cache[qid]

    existing_ids: set[str] = set()
    if args.resume:
        for p in (rollout_dir / "templates").glob("*.py"):
            existing_ids.add(p.stem)
    n_to_generate = args.target - len(existing_ids)
    print(f"target={args.target}  done={len(existing_ids)}  to_generate={max(0, n_to_generate)}")

    if args.retry_failed:
        failed_ids = sorted(p.stem for p in (rollout_dir / "failed_templates").glob("*.txt"))
        ids_to_run = failed_ids
        print(f"retry-failed: {len(failed_ids)}")
    elif n_to_generate <= 0:
        print("Nothing to do (target reached).")
        return
    else:
        prefix = args.id_prefix
        next_idx = 0
        for eid in existing_ids:
            try:
                idx = int(eid.split("_")[-1])
                next_idx = max(next_idx, idx + 1)
            except ValueError:
                pass
        ids_to_run = [f"{prefix}{next_idx + i:06d}" for i in range(n_to_generate)]

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL", "https://az.gptplus5.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.error("OPENAI_API_KEY not set in env")
        sys.exit(1)
    api_logger = APILogger(API_LOG_DIR)

    async def _one(newgen_id: str) -> dict:
        local_rng = random.Random(f"{args.seed}::{newgen_id}")
        recipe = sample_recipe(ingredients, rng=local_rng)
        icl = match_icl_examples(icl_seeds, recipe, local_rng, k=2)
        icl_with_imgs = []
        for r in icl:
            icl_with_imgs.append({
                "qid": r["qid"],
                "title": r.get("title") or "",
                "description": r.get("description") or "",
                "solution": r.get("solution") or "",
                "tags": r.get("tags") or [],
                "difficulty": r.get("difficulty") or "?",
                "image_data_url": get_data_url(r),
            })

        insp = maybe_pick_inspiration(insp_seeds, local_rng)
        insp_data = None
        if insp:
            insp_data = {
                "qid": insp["qid"],
                "source": insp.get("source") or "?",
                "title": insp.get("title") or "",
                "description": insp.get("description") or "",
                "difficulty": insp.get("difficulty") or "?",
                "image_data_url": get_data_url(insp),
            }

        retry_errors: list[dict] | None = None
        if args.retry_failed:
            audit_path = rollout_dir / "audit" / f"{newgen_id}.json"
            if audit_path.exists():
                try:
                    prev_audit = json.loads(audit_path.read_text(encoding="utf-8"))
                    retry_errors = [{
                        "seed": -1,
                        "error_type": "TemplateGenError",
                        "error": prev_audit.get("syntax_msg")
                                 or prev_audit.get("placeholders_msg")
                                 or f"missing={prev_audit.get('missing_symbols')}",
                    }]
                except Exception:
                    pass
            prev_fail_path = rollout_dir / "failed_templates" / f"{newgen_id}.txt"
            if prev_fail_path.exists():
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                arc_dir = rollout_dir / "_retry_archive" / ts
                arc_dir.mkdir(parents=True, exist_ok=True)
                prev_fail_path.rename(arc_dir / prev_fail_path.name)

        return await process_one(
            newgen_id=newgen_id, recipe=recipe, icl=icl_with_imgs,
            inspiration=insp_data, rollout_dir=rollout_dir,
            sem=sem, api_logger=api_logger,
            model=args.model, base_url=base_url, api_key=api_key,
            temperature=args.temperature, max_tokens=args.max_tokens,
            logger=logger, resume=args.resume, retry_errors=retry_errors,
        )

    sem = asyncio.Semaphore(args.concurrency)
    print(
        f"Generating {len(ids_to_run)} templates  "
        f"concurrency={args.concurrency}  model={args.model}  temp={args.temperature}"
    )
    tasks = [asyncio.create_task(_one(nid)) for nid in ids_to_run]
    results: list[dict] = []
    by_status_running: dict[str, int] = {"ok": 0, "parse_fail": 0, "llm_fail": 0, "cached": 0}

    pbar = atqdm(total=len(tasks), desc="generate", unit="tpl")
    for fut in asyncio.as_completed(tasks):
        r = await fut
        results.append(r)
        s = r["status"]
        by_status_running[s] = by_status_running.get(s, 0) + 1
        pbar.update(1)
        pbar.set_postfix(
            ok=by_status_running.get("ok", 0),
            fail=by_status_running.get("parse_fail", 0),
            llmfail=by_status_running.get("llm_fail", 0),
        )
    pbar.close()

    # Write synthetic seed-records
    out_records: list[dict] = []
    for tpl_path in sorted((rollout_dir / "templates").glob("*.py")):
        nid = tpl_path.stem
        audit_path = rollout_dir / "audit" / f"{nid}.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
        out_records.append({
            "qid":           nid,
            "difficulty":    audit.get("difficulty"),
            "algo_tags":     audit.get("algo_tags") or [],
            "struct_tags":   audit.get("struct_tags") or [],
            "visual_aspect": audit.get("visual_aspect"),
            "domain":        audit.get("domain"),
            "n_examples":    audit.get("n_examples"),
            "icl_qids":      audit.get("icl_qids") or [],
            "inspiration_qid": audit.get("inspiration_qid"),
            "inspiration_source": audit.get("inspiration_source"),
        })
    (rollout_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_records) + "\n",
        encoding="utf-8",
    )

    by_status: dict[str, int] = {}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    summary = {
        "rollout":           rollout_dir.name,
        "model":             args.model,
        "target":            args.target,
        "n_attempted":       len(results),
        "n_total_templates": len(out_records),
        "by_status":         by_status,
        "ts":                datetime.now().isoformat(timespec="seconds"),
    }
    (rollout_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Generation complete: {summary}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--model", default="claude-sonnet-4-6-aws")
    ap.add_argument("--id-prefix", default="cross_aug_")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--seed", type=int, default=20260506)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
